//! End-to-end dense generation runner — band B3 `PH-305`.
//!
//! Materializes every tensor the forward loop reads (embed, per-block attn/ffn weights,
//! output norm + output projection) via `MmapModel::dequant`, then walks a token sequence
//! through `block_forward_seq` per layer and projects the last hidden to `n_vocab` logits.
//! This is a **shape/determinism** gate for the full stack: real weights, real geometry,
//! greedy decoding. Numeric parity vs a reference (llama.cpp) is `PH-405` and lives in a
//! separate band.

use crate::core::{EngineError, Result};
use crate::forward::{block_forward_seq, last_logits, BlockWeights, Geometry};
use crate::infer::argmax;
use crate::loader::MmapModel;
use crate::model::ModelConfig;

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
    /// `output.weight` dequantized, row-major `n_vocab × d_model`.
    pub output: Vec<f32>,
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
        let embed = need(model.dequant("token_embd.weight"), "token_embd.weight")?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("blk.{i}.");
            let attn_norm = need(model.dequant(&format!("{p}attn_norm.weight")), "attn_norm")?;
            let q = need(model.dequant(&format!("{p}attn_q.weight")), "attn_q")?;
            let k = need(model.dequant(&format!("{p}attn_k.weight")), "attn_k")?;
            let v = need(model.dequant(&format!("{p}attn_v.weight")), "attn_v")?;
            let o = need(
                model.dequant(&format!("{p}attn_output.weight")),
                "attn_output",
            )?;
            let ffn_norm = need(model.dequant(&format!("{p}ffn_norm.weight")), "ffn_norm")?;
            let ffn_gate = need(model.dequant(&format!("{p}ffn_gate.weight")), "ffn_gate")?;
            let ffn_up = need(model.dequant(&format!("{p}ffn_up.weight")), "ffn_up")?;
            let ffn_down = need(model.dequant(&format!("{p}ffn_down.weight")), "ffn_down")?;
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
        let output = need(model.dequant("output.weight"), "output.weight")?;
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
}
