//! Qwen3 forward pass, numerically faithful to HuggingFace `transformers`
//! `Qwen3Model` (`5.17.0`): RMSNorm(eps=1e-6), GQA (16 q-heads / 8 kv-heads,
//! head_dim 128), QK per-head RMSNorm, RoPE rotate-half (θ=1e6), SwiGLU MLP
//! (3072), causal attention.
//!
//! Weights are F32 zero-copy views into the mmap'd `best.safetensors`.
//!
//! The forward processes one leaf at a time (no batching): each leaf is
//! short (~40–160 tokens) and the whole question set fits comfortably; the
//! matvec dominates and is applied with a small std-threads fan-out.
//!
//! Precision: HF `transformers` runs the `best_checkpoint` under `bf16`
//! autocast. Set [`BF16_MODE`] (`--precision bf16`) to reproduce that
//! semantics exactly (every intermediate rounded to bf16); default off
//! (fp32 — stricter than reference, deterministic).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::core::{EngineError, Result};
use crate::nanojev::safetensors::SafeTensors;

const HIDDEN: usize = 1024;
const N_LAYERS: usize = 28;
const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE: usize = 3072;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;

/// Global precision mode (see module docs).
static BF16_MODE: AtomicBool = AtomicBool::new(false);

pub fn set_bf16_mode(on: bool) {
    BF16_MODE.store(on, Ordering::Relaxed);
}

pub fn bf16_mode() -> bool {
    BF16_MODE.load(Ordering::Relaxed)
}

/// Round-to-nearest-even f32 -> bf16 -> f32 (matches HF autocast storage).
#[inline(always)]
fn round_bf16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(round) & 0xffff_0000)
}

/// Identity when bf16 mode is off; bf16-rounded otherwise.
#[inline(always)]
fn rb(x: f32) -> f32 {
    if bf16_mode() {
        round_bf16(x)
    } else {
        x
    }
}

/// Round a whole slice in place (identity when bf16 mode is off).
#[inline]
fn rb_slice(x: &mut [f32]) {
    if !bf16_mode() {
        return;
    }
    for v in x.iter_mut() {
        *v = round_bf16(*v);
    }
}

/// Per-layer tensor views into the mmap.
#[derive(Clone, Copy)]
struct LayerTensors<'a> {
    input_norm: &'a [f32],
    post_norm: &'a [f32],
    q_norm: &'a [f32],
    k_norm: &'a [f32],
    q_proj: &'a [f32],
    k_proj: &'a [f32],
    v_proj: &'a [f32],
    o_proj: &'a [f32],
    gate_proj: &'a [f32],
    up_proj: &'a [f32],
    down_proj: &'a [f32],
}

/// Qwen3 backbone with zero-copy weights.
pub struct Qwen3<'a> {
    embed_tokens: &'a [f32],
    norm: &'a [f32],
    layers: Vec<LayerTensors<'a>>,
}

impl<'a> Qwen3<'a> {
    /// Bind all tensor views from an open safetensors file.
    pub fn bind(st: &'a SafeTensors) -> Result<Self> {
        let f32_t = |name: &str| -> Result<&'a [f32]> {
            let (_, slice) = st.f32_view(name)?;
            Ok(slice)
        };
        let embed_tokens = f32_t("backbone.embed_tokens.weight")?;
        let norm = f32_t("backbone.norm.weight")?;
        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let p = format!("backbone.layers.{i}");
            layers.push(LayerTensors {
                input_norm: f32_t(&format!("{p}.input_layernorm.weight"))?,
                post_norm: f32_t(&format!("{p}.post_attention_layernorm.weight"))?,
                q_norm: f32_t(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: f32_t(&format!("{p}.self_attn.k_norm.weight"))?,
                q_proj: f32_t(&format!("{p}.self_attn.q_proj.weight"))?,
                k_proj: f32_t(&format!("{p}.self_attn.k_proj.weight"))?,
                v_proj: f32_t(&format!("{p}.self_attn.v_proj.weight"))?,
                o_proj: f32_t(&format!("{p}.self_attn.o_proj.weight"))?,
                gate_proj: f32_t(&format!("{p}.mlp.gate_proj.weight"))?,
                up_proj: f32_t(&format!("{p}.mlp.up_proj.weight"))?,
                down_proj: f32_t(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        Ok(Self {
            embed_tokens,
            norm,
            layers,
        })
    }

    /// Forward the full token sequence; returns the hidden state at the LAST
    /// position only (what the decision head consumes).
    pub fn last_hidden(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let n = ids.len();
        if n == 0 {
            return Err(EngineError {
                message: "qwen3: empty token sequence".into(),
            });
        }
        // Embedding lookup
        let mut hidden = Vec::with_capacity(n * HIDDEN);
        for &id in ids {
            let start = (id as usize) * HIDDEN;
            let mut row = self.embed_tokens[start..start + HIDDEN].to_vec();
            rb_slice(&mut row);
            hidden.extend_from_slice(&row);
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // ── attention block ─────────────────────────────────────────
            let mut normed = hidden.clone();
            rms_norm_rows(&mut normed, layer.input_norm);

            // Per-token projections (matvec is [m,k]·[k] -> [m])
            let mut q = vec![0.0f32; n * N_HEADS * HEAD_DIM];
            let mut k = vec![0.0f32; n * N_KV_HEADS * HEAD_DIM];
            let mut v = vec![0.0f32; n * N_KV_HEADS * HEAD_DIM];
            for t in 0..n {
                let xrow = &normed[t * HIDDEN..(t + 1) * HIDDEN];
                q[t * N_HEADS * HEAD_DIM..(t + 1) * N_HEADS * HEAD_DIM].copy_from_slice(&matvec(
                    layer.q_proj,
                    N_HEADS * HEAD_DIM,
                    HIDDEN,
                    xrow,
                ));
                k[t * N_KV_HEADS * HEAD_DIM..(t + 1) * N_KV_HEADS * HEAD_DIM]
                    .copy_from_slice(&matvec(layer.k_proj, N_KV_HEADS * HEAD_DIM, HIDDEN, xrow));
                v[t * N_KV_HEADS * HEAD_DIM..(t + 1) * N_KV_HEADS * HEAD_DIM]
                    .copy_from_slice(&matvec(layer.v_proj, N_KV_HEADS * HEAD_DIM, HIDDEN, xrow));
            }

            // Per-head QK RMSNorm (over head_dim)
            for h in 0..N_HEADS {
                rms_norm_inplace(&mut q[h * HEAD_DIM..(h + 1) * HEAD_DIM], layer.q_norm);
            }
            for h in 0..N_KV_HEADS {
                rms_norm_inplace(&mut k[h * HEAD_DIM..(h + 1) * HEAD_DIM], layer.k_norm);
            }

            // RoPE rotate-half, position = token index (leaf-local)
            rope_inplace(&mut q, N_HEADS, n);
            rope_inplace(&mut k, N_KV_HEADS, n);

            if std::env::var("NANOJEV_DEBUG_SCORES").is_ok() && li == 0 {
                for h in 0..N_HEADS {
                    let qb = (n - 1) * N_HEADS * HEAD_DIM + h * HEAD_DIM;
                    let qnorm: f32 = q[qb..qb + HEAD_DIM]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f32>()
                        .sqrt();
                    let kv = h / (N_HEADS / N_KV_HEADS);
                    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
                    let mut scores: Vec<(usize, f32)> = (0..n)
                        .map(|j| {
                            let kb = (j * N_KV_HEADS + kv) * HEAD_DIM;
                            (j, dot(&q[qb..qb + HEAD_DIM], &k[kb..kb + HEAD_DIM]) * scale)
                        })
                        .collect();
                    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let top: Vec<String> = scores
                        .iter()
                        .take(3)
                        .map(|(j, s)| format!("{j}:{s:.3}"))
                        .collect();
                    eprintln!(
                        "nanojev: L0 dbg h{h:02} norm(q)={:.2} top3=[{}]",
                        qnorm,
                        top.join(", ")
                    );
                }
            }

            // Causal attention, GQA repeat (q_h pairs with kv_{h/2})
            let attn_out = attention_gqa(&q, &k, &v, n);
            let mut attn_proj = vec![0.0f32; n * HIDDEN];
            for t in 0..n {
                let arow = &attn_out[t * N_HEADS * HEAD_DIM..(t + 1) * N_HEADS * HEAD_DIM];
                attn_proj[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&matvec(
                    layer.o_proj,
                    HIDDEN,
                    N_HEADS * HEAD_DIM,
                    arow,
                ));
            }
            for (dst, src) in hidden.iter_mut().zip(attn_proj) {
                *dst = rb(*dst + src);
            }

            // ── MLP block (SwiGLU) ─────────────────────────────────────
            let mut normed2 = hidden.clone();
            rms_norm_rows(&mut normed2, layer.post_norm);
            let mut gate = vec![0.0f32; n * INTERMEDIATE];
            let mut up = vec![0.0f32; n * INTERMEDIATE];
            for t in 0..n {
                let xrow = &normed2[t * HIDDEN..(t + 1) * HIDDEN];
                gate[t * INTERMEDIATE..(t + 1) * INTERMEDIATE].copy_from_slice(&matvec(
                    layer.gate_proj,
                    INTERMEDIATE,
                    HIDDEN,
                    xrow,
                ));
                up[t * INTERMEDIATE..(t + 1) * INTERMEDIATE].copy_from_slice(&matvec(
                    layer.up_proj,
                    INTERMEDIATE,
                    HIDDEN,
                    xrow,
                ));
            }
            let mut act = Vec::with_capacity(n * INTERMEDIATE);
            for (g, u) in gate.iter().zip(&up) {
                act.push(silu(rb(*g)) * rb(*u));
            }
            let mut mlp_out = vec![0.0f32; n * HIDDEN];
            for t in 0..n {
                let arow = &act[t * INTERMEDIATE..(t + 1) * INTERMEDIATE];
                mlp_out[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&matvec(
                    layer.down_proj,
                    HIDDEN,
                    INTERMEDIATE,
                    arow,
                ));
            }
            for (dst, src) in hidden.iter_mut().zip(mlp_out) {
                *dst = rb(*dst + src);
            }
        }

        // Final RMSNorm over the full sequence; return the LAST position.
        let mut final_normed = hidden;
        rms_norm_rows(&mut final_normed, self.norm);
        let start = (n - 1) * HIDDEN;
        Ok(final_normed[start..start + HIDDEN].to_vec())
    }
}

/// RMSNorm over rows of length 1024 (in-place).
fn rms_norm_rows(x: &mut [f32], weight: &[f32]) {
    debug_assert_eq!(weight.len(), HIDDEN);
    for row in x.chunks_mut(HIDDEN) {
        rms_norm_inplace(row, weight);
    }
}

fn rms_norm_inplace(row: &mut [f32], weight: &[f32]) {
    debug_assert_eq!(row.len(), weight.len());
    let ms = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let scale = 1.0 / (ms + RMS_EPS).sqrt();
    let mut out = Vec::with_capacity(row.len());
    for (v, w) in row.iter().zip(weight) {
        out.push(rb(*v * scale * w));
    }
    row.copy_from_slice(&out);
}

/// Naive matvec with a small std-threads fan-out over output rows.
/// `rows` is [m, k] row-major; returns out[m] = rows · x.
fn matvec(rows: &[f32], m: usize, k: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(rows.len(), m * k);
    assert_eq!(x.len(), k);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 12);
    if m < 256 || threads == 1 {
        return matvec_scalar(rows, m, k, x);
    }
    let mut out = vec![0.0f32; m];
    let chunk = m.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, out_chunk) in out.chunks_mut(chunk).enumerate() {
            let rows_chunk =
                &rows[t * chunk * k..((t * chunk + out_chunk.len()) * k).min(rows.len())];
            s.spawn(move || {
                for (j, o) in out_chunk.iter_mut().enumerate() {
                    let r = &rows_chunk[j * k..(j + 1) * k];
                    *o = rb(dot(r, x));
                }
            });
        }
    });
    out
}

fn matvec_scalar(rows: &[f32], m: usize, k: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    for i in 0..m {
        out[i] = rb(dot(&rows[i * k..(i + 1) * k], x));
    }
    out
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |acc, (x, y)| acc + x * y)
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Rotate-half RoPE (HF layout): for each of the first 64 freqs,
/// x'[i] = x[i]·cos(i·θᵢ) − x[i+64]·sin(i·θᵢ),
/// x'[i+64] = x[i+64]·cos(i·θᵢ) + x[i]·sin(i·θᵢ),  θᵢ = θ^(-2i/128).
fn rope_inplace(x: &mut [f32], n_heads: usize, seq: usize) {
    let head_dim = HEAD_DIM;
    let half = head_dim / 2;
    for t in 0..seq {
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            let mut cos_buf = [0.0f32; 64];
            let mut sin_buf = [0.0f32; 64];
            for i in 0..half {
                let inv_freq = ROPE_THETA.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = t as f32 * inv_freq;
                cos_buf[i] = angle.cos();
                sin_buf[i] = angle.sin();
            }
            for i in 0..half {
                let x1 = x[base + i];
                let x2 = x[base + i + half];
                x[base + i] = rb(x1 * cos_buf[i] - x2 * sin_buf[i]);
                x[base + i + half] = rb(x2 * cos_buf[i] + x1 * sin_buf[i]);
            }
        }
    }
}

/// Causal GQA attention over one leaf (short sequences, scalar kernels).
fn attention_gqa(q: &[f32], k: &[f32], v: &[f32], seq: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * N_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    for t in 0..seq {
        for h in 0..N_HEADS {
            let kv = h / (N_HEADS / N_KV_HEADS); // h/2
            let q_base = (t * N_HEADS + h) * HEAD_DIM;
            let mut scores = Vec::with_capacity(t + 1);
            for j in 0..=t {
                let k_base = (j * N_KV_HEADS + kv) * HEAD_DIM;
                scores.push(rb(dot(
                    &q[q_base..q_base + HEAD_DIM],
                    &k[k_base..k_base + HEAD_DIM],
                ) * scale));
            }
            softmax_inplace(&mut scores);
            let mut acc = [0.0f32; HEAD_DIM];
            for (j, &p) in scores.iter().enumerate() {
                let v_base = (j * N_KV_HEADS + kv) * HEAD_DIM;
                for d in 0..HEAD_DIM {
                    acc[d] = rb(acc[d] + p * v[v_base + d]);
                }
            }
            let o_base = (t * N_HEADS + h) * HEAD_DIM;
            out[o_base..o_base + HEAD_DIM].copy_from_slice(&acc);
        }
    }
    out
}

fn softmax_inplace(xs: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in xs.iter_mut() {
            *x = rb(*x / sum);
        }
    }
}
