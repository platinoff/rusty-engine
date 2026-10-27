//! End-to-end `qwen35` hybrid runner — band B13 `PH-1304`.
//!
//! The text core of MiniCPM-V 4.7 is 18 GDN ( [`crate::gdn`] ) layers + 6 full-attention
//! ( [`crate::q35`] ) layers interleaved by `recurrent_layers`/`full_attention_interval`.
//! This module loads every tensor the two blocks need (projections stay quantized
//! through [`QMat`] — only the small F32 vectors dequant), and runs the Qwen3-Next
//! residual topology per layer:
//!
//! ```text
//! h = x + attn(rmsnorm(x, attn_norm))          // GDN step or q35 step
//! y = h + ffn(rmsnorm(h, post_attention_norm)) // NO ffn_norm on this arch
//! ```
//!
//! Decode is greedy, per-token, state-carrying: GDN blocks advance their 1 MiB delta
//! state, full blocks append to their KV cache — one fixed cost per token regardless
//! of context length. Prefill steps token-by-token for now (the batched kernels exist;
//! fusing them into this loop is `PH-1305` perf work).

use crate::core::{EngineError, Result};
use crate::gdn::{gdn_decode_step, GdnGeom, GdnState, GdnWeights};
use crate::infer::{argmax, rmsnorm, top_k};
use crate::loader::MmapModel;
use crate::model::{LayerKind, ModelConfig};
use crate::moe::swiglu;
use crate::q35::{q35_decode_step, Q35Geom, Q35Kv, Q35Weights};
use crate::qmat::QMat;

/// Owned half of a recurrent block (F32 vectors + quantized projections).
pub struct GdnBlock {
    pub qkv: QMat,
    pub gate: QMat,
    pub alpha: QMat,
    pub beta: QMat,
    pub out: QMat,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub conv_w: Vec<f32>,
    pub norm_w: Vec<f32>,
}

impl GdnBlock {
    fn weights(&self) -> GdnWeights<'_> {
        GdnWeights {
            qkv: &self.qkv,
            gate: &self.gate,
            alpha: &self.alpha,
            beta: &self.beta,
            out: &self.out,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            conv_w: &self.conv_w,
            norm_w: &self.norm_w,
            eps: 0.0, // filled by the runner (needs the config eps; see block_norm below)
        }
    }
}

/// Owned half of a full-attention block.
pub struct FullBlock {
    pub q: QMat,
    pub k: QMat,
    pub v: QMat,
    pub o: QMat,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

impl FullBlock {
    fn weights(&self) -> Q35Weights<'_> {
        Q35Weights {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            o: &self.o,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
        }
    }
}

/// One transformer block of the hybrid.
pub struct Q35Block {
    pub kind: LayerKind,
    pub attn_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub gdn: Option<GdnBlock>,
    pub full: Option<FullBlock>,
    pub ffn_gate: QMat,
    pub ffn_up: QMat,
    pub ffn_down: QMat,
}

/// A loaded `qwen35` hybrid model, ready for greedy generation.
pub struct Q35Runner {
    pub n_layers: usize,
    pub n_vocab: usize,
    pub d_model: usize,
    pub eps: f32,
    pub gdn_geom: GdnGeom,
    pub q35_geom: Q35Geom,
    pub embed: Vec<f32>,
    pub blocks: Vec<Q35Block>,
    pub output_norm: Vec<f32>,
    pub output: QMat,
    pub eos_token_id: Option<u32>,
}

fn need(v: Result<Vec<f32>>, name: &str) -> Result<Vec<f32>> {
    v.map_err(|_| EngineError {
        message: format!("runner35: tensor '{name}' missing or unsized"),
    })
}

impl Q35Runner {
    /// Load the hybrid: `cfg` must validate against `model` (the `PH-1301` gate).
    /// Both block geometries must exist — a hybrid file always ships `ssm.*` keys even
    /// if every layer were full, but a missing geometry is a hard error, not a fallback.
    pub fn from_model(model: &MmapModel, cfg: &ModelConfig) -> Result<Self> {
        let gdn_geom = GdnGeom::from_config(cfg).ok_or_else(|| EngineError {
            message: "runner35: config lacks a complete ssm.* GDN geometry".to_string(),
        })?;
        let q35_geom = Q35Geom::from_config(cfg).ok_or_else(|| EngineError {
            message: "runner35: config lacks a qwen35 full-attention geometry".to_string(),
        })?;
        let dm = cfg.d_model;
        let embed = need(model.dequant("token_embd.weight"), "token_embd.weight")?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("blk.{i}.");
            let attn_norm = need(model.dequant(&format!("{p}attn_norm.weight")), "attn_norm")?;
            let post_attention_norm = need(
                model.dequant(&format!("{p}post_attention_norm.weight")),
                "post_attention_norm",
            )?;
            let kind = cfg.layer_kind(i);
            let (gdn, full) = match kind {
                LayerKind::Recurrent => {
                    let b = GdnBlock {
                        qkv: QMat::from_model(
                            model,
                            &format!("{p}attn_qkv.weight"),
                            gdn_geom.qkv_dim(),
                            dm,
                        )?,
                        gate: QMat::from_model(
                            model,
                            &format!("{p}attn_gate.weight"),
                            gdn_geom.inner,
                            dm,
                        )?,
                        alpha: QMat::from_model(
                            model,
                            &format!("{p}ssm_alpha.weight"),
                            gdn_geom.groups,
                            dm,
                        )?,
                        beta: QMat::from_model(
                            model,
                            &format!("{p}ssm_beta.weight"),
                            gdn_geom.groups,
                            dm,
                        )?,
                        out: QMat::from_model(
                            model,
                            &format!("{p}ssm_out.weight"),
                            dm,
                            gdn_geom.inner,
                        )?,
                        a_log: need(model.dequant(&format!("{p}ssm_a")), "ssm_a")?,
                        dt_bias: need(model.dequant(&format!("{p}ssm_dt.bias")), "ssm_dt.bias")?,
                        conv_w: need(
                            model.dequant(&format!("{p}ssm_conv1d.weight")),
                            "ssm_conv1d.weight",
                        )?,
                        norm_w: need(
                            model.dequant(&format!("{p}ssm_norm.weight")),
                            "ssm_norm.weight",
                        )?,
                    };
                    (Some(b), None)
                }
                LayerKind::Full => {
                    let b = FullBlock {
                        q: QMat::from_model(
                            model,
                            &format!("{p}attn_q.weight"),
                            2 * q35_geom.q_dim(),
                            dm,
                        )?,
                        k: QMat::from_model(
                            model,
                            &format!("{p}attn_k.weight"),
                            q35_geom.kv_dim(),
                            dm,
                        )?,
                        v: QMat::from_model(
                            model,
                            &format!("{p}attn_v.weight"),
                            q35_geom.kv_dim(),
                            dm,
                        )?,
                        o: QMat::from_model(
                            model,
                            &format!("{p}attn_output.weight"),
                            dm,
                            q35_geom.q_dim(),
                        )?,
                        q_norm: need(
                            model.dequant(&format!("{p}attn_q_norm.weight")),
                            "attn_q_norm",
                        )?,
                        k_norm: need(
                            model.dequant(&format!("{p}attn_k_norm.weight")),
                            "attn_k_norm",
                        )?,
                    };
                    (None, Some(b))
                }
            };
            let ffn_gate = QMat::from_model(model, &format!("{p}ffn_gate.weight"), cfg.d_ff, dm)?;
            let ffn_up = QMat::from_model(model, &format!("{p}ffn_up.weight"), cfg.d_ff, dm)?;
            let ffn_down = QMat::from_model(model, &format!("{p}ffn_down.weight"), dm, cfg.d_ff)?;
            blocks.push(Q35Block {
                kind,
                attn_norm,
                post_attention_norm,
                gdn,
                full,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }
        let output_norm = need(model.dequant("output_norm.weight"), "output_norm.weight")?;
        let output = QMat::from_model(model, "output.weight", cfg.n_vocab, dm)?;
        Ok(Self {
            n_layers: cfg.n_layers,
            n_vocab: cfg.n_vocab,
            d_model: dm,
            eps: cfg.rms_eps,
            gdn_geom,
            q35_geom,
            embed,
            blocks,
            output_norm,
            output,
            eos_token_id: cfg.eos_token_id,
        })
    }

    /// Row `t` of the embedding table.
    fn embed_row(&self, tok: u32) -> &[f32] {
        let t = (tok as usize).min(self.n_vocab.saturating_sub(1));
        &self.embed[t * self.d_model..(t + 1) * self.d_model]
    }

    /// One block over one hidden row (shared by prefill and decode).
    fn block_row(&self, i: usize, x: &[f32], pos: usize, dec: &mut Q35Decoder) -> Vec<f32> {
        let b = &self.blocks[i];
        let normed = rmsnorm(x, &b.attn_norm, self.eps);
        let attn = match b.kind {
            LayerKind::Recurrent => {
                let g = b.gdn.as_ref().expect("recurrent block has gdn");
                let mut w = g.weights();
                w.eps = self.eps;
                let st = &mut dec.gdn[i];
                gdn_decode_step(&w, &self.gdn_geom, &normed, st)
            }
            LayerKind::Full => {
                let f = b.full.as_ref().expect("full block has q35");
                let w = f.weights();
                let kv = &mut dec.kv[i];
                q35_decode_step(&w, &self.q35_geom, &normed, pos, kv)
                    .expect("q35 shape guarded at load")
            }
        };
        let h: Vec<f32> = x.iter().zip(&attn).map(|(a, c)| a + c).collect();
        let hn = rmsnorm(&h, &b.post_attention_norm, self.eps);
        let g = b.ffn_gate.matvec(&hn);
        let u = b.ffn_up.matvec(&hn);
        let ffn = b.ffn_down.matvec(&swiglu(&g, &u));
        h.iter().zip(&ffn).map(|(a, c)| a + c).collect()
    }
}

/// Streaming decoder over a [`Q35Runner`]: one GDN state per recurrent block, one KV
/// cache per full block, fixed memory, O(1) work per token per layer.
pub struct Q35Decoder<'a> {
    runner: &'a Q35Runner,
    gdn: Vec<GdnState>,
    kv: Vec<Q35Kv>,
    seq_len: usize,
}

/// One step of a greedy trace: the token argmax picked plus the `k` best logits that
/// produced it (the `PH-1305` numeric fingerprint — see [`crate::infer::top_k`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TraceStep {
    /// Token emitted at this step.
    pub token: u32,
    /// Descending `(token, logit)` pairs for the step's logit vector.
    pub top: Vec<(usize, f32)>,
}

impl<'a> Q35Decoder<'a> {
    pub fn new(runner: &'a Q35Runner) -> Self {
        Self {
            runner,
            gdn: (0..runner.n_layers)
                .map(|_| GdnState::new(&runner.gdn_geom))
                .collect(),
            kv: vec![Q35Kv::default(); runner.n_layers],
            seq_len: 0,
        }
    }

    /// Feed one token, get its next-token logits.
    pub fn step(&mut self, tok: u32) -> Vec<f32> {
        let pos = self.seq_len;
        let mut hidden = self.runner.embed_row(tok).to_vec();
        for i in 0..self.runner.n_layers {
            hidden = self.runner.block_row(i, &hidden, pos, self);
        }
        self.seq_len += 1;
        let normed = rmsnorm(&hidden, &self.runner.output_norm, self.runner.eps);
        self.runner.output.matvec(&normed)
    }

    /// Prefill = stepping the prompt (bit-exact base path; batched fusion is `PH-1305`).
    pub fn prefill(&mut self, prompt: &[u32]) -> Vec<f32> {
        let mut logits = Vec::new();
        for &t in prompt {
            logits = self.step(t);
        }
        logits
    }

    /// Greedy generate: prefill, then argmax-append up to `max_new` (EOS stops).
    pub fn generate_incremental(&mut self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        assert!(!prompt.is_empty(), "q35 generate: empty prompt");
        let mut seq: Vec<u32> = prompt.to_vec();
        let mut logits = self.prefill(prompt);
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

    /// Greedy generate **with** the per-step top-`k` logits: same token stream as
    /// [`Self::generate_incremental`] (one argmax per step), but every step also keeps
    /// the ranking around the argmax — the `PH-1305` parity fingerprint against
    /// llama.cpp's `logprobs` response.
    pub fn generate_trace(&mut self, prompt: &[u32], max_new: usize, k: usize) -> Vec<TraceStep> {
        assert!(!prompt.is_empty(), "q35 trace: empty prompt");
        let mut logits = self.prefill(prompt);
        let mut steps = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let tok = match argmax(&logits) {
                Some(t) => t as u32,
                None => break,
            };
            steps.push(TraceStep {
                token: tok,
                top: top_k(&logits, k),
            });
            if Some(tok) == self.runner.eos_token_id {
                break;
            }
            logits = self.step(tok);
        }
        steps
    }
}
