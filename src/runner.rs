//! End-to-end dense generation runner — band B3 `PH-305`.
//!
//! Materializes every tensor the forward loop reads (embed, per-block attn/ffn weights,
//! output norm + output projection) via `MmapModel::dequant`, then walks a token sequence
//! through `block_forward_seq` per layer and projects the last hidden to `n_vocab` logits.
//! This is a **shape/determinism** gate for the full stack: real weights, real geometry,
//! greedy decoding. Numeric parity vs a reference (llama.cpp) is `PH-405` and lives in a
//! separate band.

use crate::core::{EngineError, Result};
use crate::forward::{
    block_forward_cached, block_forward_prefill_cached, block_forward_seq, last_logits,
    logits_from_hidden, BlockWeights, Geometry, LayerKv,
};
use crate::infer::{apply_repetition_penalty, argmax, rmsnorm, sample, GenerationOptions, SeedRng};
use crate::loader::MmapModel;
use crate::model::ModelConfig;
use crate::qmat::QMat;

/// A fully dequantized dense model, ready to run a forward pass over token ids.
pub struct DenseRunner {
    pub geom: Geometry,
    pub n_layers: usize,
    pub n_vocab: usize,
    /// `token_embd.weight` dequantized, row-major `n_vocab × d_model`.
    pub embed: Vec<f32>,
    /// Per-block `BlockWeights` (dequantized attn/ffn projections + norms).
    pub blocks: Vec<BlockWeights>,
    /// `output_norm.weight` (or `norm.weight`) dequantized, length `d_model`.
    pub output_norm: Vec<f32>,
    /// `output.weight` in stored quant blocks, row-major `n_vocab × d_model` (fused matvec).
    pub output: QMat,
    /// Optional EOS token id (from `tokenizer.ggml.eos_token_id`).
    pub eos_token_id: Option<u32>,
}

fn need(v: Result<Vec<f32>>, name: &str) -> Result<Vec<f32>> {
    v.map_err(|_| EngineError {
        message: format!("runner: tensor '{name}' missing or unsized"),
    })
}

impl DenseRunner {
    /// Load + dequantize every tensor the forward loop needs. Assumes `cfg` matches
    /// `model.header()` (callers normally run `ModelConfig::validate` first).
    pub fn from_model(model: &MmapModel, cfg: &ModelConfig) -> Result<Self> {
        let geom = Geometry::new(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.d_ff,
            cfg.rope_freq_base,
            cfg.rms_eps,
        );
        let d_model = cfg.d_model;
        let kv_dim = geom.n_kv_heads * geom.head_dim;
        let d_ff = cfg.d_ff;
        let embed = need(model.dequant("token_embd.weight"), "token_embd.weight")?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("blk.{i}.");
            let attn_norm = need(model.dequant(&format!("{p}attn_norm.weight")), "attn_norm")?;
            let q = QMat::from_model(model, &format!("{p}attn_q.weight"), d_model, d_model)?;
            let k = QMat::from_model(model, &format!("{p}attn_k.weight"), kv_dim, d_model)?;
            let v = QMat::from_model(model, &format!("{p}attn_v.weight"), kv_dim, d_model)?;
            let o = QMat::from_model(model, &format!("{p}attn_output.weight"), d_model, d_model)?;
            let ffn_norm = need(model.dequant(&format!("{p}ffn_norm.weight")), "ffn_norm")?;
            let ffn_gate = QMat::from_model(model, &format!("{p}ffn_gate.weight"), d_ff, d_model)?;
            let ffn_up = QMat::from_model(model, &format!("{p}ffn_up.weight"), d_ff, d_model)?;
            let ffn_down = QMat::from_model(model, &format!("{p}ffn_down.weight"), d_model, d_ff)?;
            blocks.push(BlockWeights {
                attn_norm,
                q,
                k,
                v,
                o,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }
        let output_norm = match model.dequant("output_norm.weight") {
            Ok(v) => v,
            Err(_) => need(model.dequant("norm.weight"), "norm.weight")?,
        };
        let output = QMat::from_model(model, "output.weight", cfg.n_vocab, d_model)?;
        Ok(Self {
            geom,
            n_layers: cfg.n_layers,
            n_vocab: cfg.n_vocab,
            embed,
            blocks,
            output_norm,
            output,
            eos_token_id: cfg.eos_token_id,
        })
    }

    /// Row `t` of the embedding table (length `d_model`).
    fn embed_row(&self, tok: u32) -> &[f32] {
        let t = tok as usize;
        &self.embed[t * self.geom.d_model..(t + 1) * self.geom.d_model]
    }

    /// Prefill a token sequence: returns the last-position logits. Attention sees the
    /// whole prefix (causal), so this is the correct "next-token from a prompt" call.
    pub fn prefill_logits(&self, tokens: &[u32]) -> Vec<f32> {
        assert!(!tokens.is_empty(), "prefill_logits: empty prompt");
        let d = self.geom.d_model;
        let mut hidden = vec![0.0f32; tokens.len() * d];
        for (i, &t) in tokens.iter().enumerate() {
            hidden[i * d..(i + 1) * d].copy_from_slice(self.embed_row(t));
        }
        for blk in &self.blocks {
            hidden = block_forward_seq(&hidden, blk, &self.geom, 0);
        }
        last_logits(
            &hidden,
            &self.geom,
            &self.output_norm,
            &self.output,
            self.n_vocab,
        )
    }

    /// Greedy next-token (argmax of last-position logits). `None` if `n_vocab == 0`.
    pub fn greedy(&self, tokens: &[u32]) -> Option<usize> {
        argmax(&self.prefill_logits(tokens))
    }

    /// Greedy loop: prefill the prompt, then append argmax tokens up to `max_new`,
    /// stopping early if `eos_token_id` fires. Runs full prefill each step (no KV cache
    /// yet — the shape gate is `PH-305`; incremental decode + KV lands with B4).
    pub fn generate(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        let mut seq: Vec<u32> = prompt.to_vec();
        for _ in 0..max_new {
            match self.greedy(&seq) {
                Some(t) if Some(t as u32) == self.eos_token_id => break,
                Some(t) => seq.push(t as u32),
                None => break,
            }
        }
        seq
    }

    /// Sampled generation under `opts` (temperature/top-k/top-p/repetition) driven by the
    /// deterministic `rng`. `temperature<=0` reduces to greedy, so `generate_with` with the
    /// greedy opts reproduces [`generate`] exactly. Repetition penalty applies to the prior
    /// token ids; stops early on `eos_token_id`.
    pub fn generate_with(
        &self,
        prompt: &[u32],
        opts: &GenerationOptions,
        max_new: usize,
        rng: &mut SeedRng,
    ) -> Vec<u32> {
        assert!(!prompt.is_empty(), "generate_with: empty prompt");
        let mut seq: Vec<u32> = prompt.to_vec();
        for _ in 0..max_new {
            let logits = self.prefill_logits(&seq);
            let penalized = apply_repetition_penalty(
                &logits,
                &seq.iter().map(|&t| t as usize).collect::<Vec<usize>>(),
                opts.repetition_penalty,
            );
            match sample(&penalized, opts, rng.next_f32()) {
                Some(t) if Some(t as u32) == self.eos_token_id => break,
                Some(t) => seq.push(t as u32),
                None => break,
            }
        }
        seq
    }
}

/// Incremental greedy decoder over a [`DenseRunner`] — band B3 `PH-307`.
///
/// Keeps one [`LayerKv`] cache per layer and advances position-by-position, so each new
/// token costs a single row through every block (O(n) per step over the growing cache)
/// instead of re-prefilling the whole prefix. Produces **exactly** the same token stream
/// as [`DenseRunner::generate`] (causal equivalence, asserted by a test).
pub struct DenseDecoder<'a> {
    runner: &'a DenseRunner,
    caches: Vec<LayerKv>,
    seq_len: usize,
}

impl<'a> DenseDecoder<'a> {
    pub fn new(runner: &'a DenseRunner) -> Self {
        Self {
            runner,
            caches: vec![LayerKv::default(); runner.blocks.len()],
            seq_len: 0,
        }
    }

    /// Feed one token, return last-position logits after all layers.
    pub fn step(&mut self, tok: u32) -> Vec<f32> {
        let pos = self.seq_len;
        let mut hidden = self.runner.embed_row(tok).to_vec();
        for (l, blk) in self.runner.blocks.iter().enumerate() {
            hidden =
                block_forward_cached(&hidden, blk, &self.runner.geom, pos, &mut self.caches[l]);
        }
        self.seq_len += 1;
        let normed = rmsnorm(&hidden, &self.runner.output_norm, self.runner.geom.rms_eps);
        logits_from_hidden(
            &normed,
            &self.runner.output,
            &self.runner.geom,
            self.runner.n_vocab,
        )
    }

    /// Batched prompt prefill: fill the KV cache in one pass (projections batched, so the
    /// quant dequant is paid once) and return the last-position logits. Equivalent to
    /// calling [`step`](Self::step) once per prompt token, but O(1) batched passes per
    /// layer instead of `prompt.len()`.
    pub fn prefill(&mut self, prompt: &[u32]) -> Vec<f32> {
        assert!(!prompt.is_empty(), "prefill: empty prompt");
        let g = &self.runner.geom;
        let d = g.d_model;
        let mut hidden = vec![0.0f32; prompt.len() * d];
        for (i, &t) in prompt.iter().enumerate() {
            hidden[i * d..(i + 1) * d].copy_from_slice(self.runner.embed_row(t));
        }
        let pos_start = self.seq_len;
        for (l, blk) in self.runner.blocks.iter().enumerate() {
            hidden = block_forward_prefill_cached(&hidden, blk, g, pos_start, &mut self.caches[l]);
        }
        self.seq_len += prompt.len();
        let last = &hidden[(prompt.len() - 1) * d..prompt.len() * d];
        let normed = rmsnorm(last, &self.runner.output_norm, g.rms_eps);
        logits_from_hidden(&normed, &self.runner.output, g, self.runner.n_vocab)
    }

    /// Prefill the prompt, then greedily append up to `max_new` tokens (stop at EOS).
    /// `step` returns the distribution for the token *after* its input, so the last
    /// prefill step's logits already selects the first generated token — exactly the
    /// re-prefill `DenseRunner::generate` semantics.
    pub fn generate_incremental(&mut self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        assert!(!prompt.is_empty(), "generate_incremental: empty prompt");
        let mut seq: Vec<u32> = Vec::with_capacity(prompt.len() + max_new);
        let mut logits = self.prefill(prompt);
        seq.extend_from_slice(prompt);
        for _ in 0..max_new {
            match argmax(&logits) {
                Some(t) if Some(t as u32) == self.runner.eos_token_id => break,
                Some(t) => {
                    seq.push(t as u32);
                    logits = self.step(t as u32);
                }
                None => break,
            }
        }
        seq
    }
}
