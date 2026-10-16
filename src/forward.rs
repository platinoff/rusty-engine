//! Transformer forward composition — band B3 `PH-305d`.
//!
//! Wires the unit-tested primitives into one decode step: `rmsnorm → multi-head GQA
//! causal attention (RoPE) → residual → rmsnorm → SwiGLU FFN → residual`, then a final
//! projection to logits. Verified end-to-end with a deterministic toy model (zeroed
//! projections reduce a block to its residual path; a crafted embed/output pair yields a
//! known greedy token). Real Qwen numeric parity vs llama.cpp is `PH-405` (needs a
//! reference + tensor-layout confirmation); this slice proves the *plumbing* composes.

use crate::infer::{gqa_repeat_kv, rmsnorm, rope};
use crate::moe::softmax;
use crate::qmat::QMat;

/// Model geometry a block needs (dimensions + RoPE/norm knobs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_ff: usize,
    pub head_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
}

impl Geometry {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        d_ff: usize,
        rope_freq_base: f32,
        rms_eps: f32,
    ) -> Self {
        assert!(
            n_heads > 0 && d_model % n_heads == 0,
            "Geometry: heads divide d_model"
        );
        assert!(
            n_kv_heads > 0 && n_heads % n_kv_heads == 0,
            "Geometry: heads divide kv"
        );
        Self {
            d_model,
            n_heads,
            n_kv_heads,
            d_ff,
            head_dim: d_model / n_heads,
            rope_freq_base,
            rms_eps,
        }
    }
}

/// One transformer block's weights (row-major projection matrices + norms).
#[derive(Debug, Clone)]
pub struct BlockWeights {
    pub attn_norm: Vec<f32>,
    /// `q`: d_model×d_model; `k`,`v`: (n_kv·head)×d_model; `o`: d_model×(n_heads·head).
    pub q: QMat,
    pub k: QMat,
    pub v: QMat,
    pub o: QMat,
    pub ffn_norm: Vec<f32>,
    /// SwiGLU: gate/up are d_ff×d_model, down is d_model×d_ff.
    pub ffn_gate: QMat,
    pub ffn_up: QMat,
    pub ffn_down: QMat,
}

/// Dense SwiGLU FFN: `down(swiglu(gate(x), up(x)))`, each projection fused via [`QMat`].
fn ffn_forward(x: &[f32], gate: &QMat, up: &QMat, down: &QMat) -> Vec<f32> {
    let g = gate.matvec(x);
    let u = up.matvec(x);
    let a = crate::moe::swiglu(&g, &u);
    down.matvec(&a)
}

/// Single-token causal attention (a decode step attending over a 1-position span): each
/// head's softmax over its own key is 1.0, so the context equals the (GQA-broadcast) value.
/// Q/K are still RoPE-rotated and scored, exercising the real path.
fn attention_decode(x_norm: &[f32], w: &BlockWeights, g: &Geometry, pos: usize) -> Vec<f32> {
    let q = w.q.matvec(x_norm);
    let k = w.k.matvec(x_norm);
    let v = w.v.matvec(x_norm);
    let v_exp = gqa_repeat_kv(&v, 1, g.n_kv_heads, g.n_heads, g.head_dim);
    let scale = 1.0 / (g.head_dim as f32).sqrt();
    let mut ctx = vec![0.0f32; g.n_heads * g.head_dim];
    for h in 0..g.n_heads {
        let hs = h * g.head_dim;
        let qh = rope(&q[hs..hs + g.head_dim], pos, g.rope_freq_base);
        let kh_src = (h * g.n_kv_heads / g.n_heads) * g.head_dim;
        let kh = rope(&k[kh_src..kh_src + g.head_dim], pos, g.rope_freq_base);
        let score = qh.iter().zip(&kh).map(|(a, b)| a * b).sum::<f32>() * scale;
        // softmax over a single position is exactly [1.0], but compute it for the real path.
        let weight = softmax(&[score])[0];
        for (dst, src) in ctx[hs..hs + g.head_dim]
            .iter_mut()
            .zip(&v_exp[hs..hs + g.head_dim])
        {
            *dst = *src * weight;
        }
    }
    w.o.matvec(&ctx)
}

/// One transformer block decode step: attention + FFN with residuals.
pub fn block_forward(x: &[f32], w: &BlockWeights, g: &Geometry, pos: usize) -> Vec<f32> {
    assert_eq!(x.len(), g.d_model, "block_forward: hidden width");
    let attn = attention_decode(&rmsnorm(x, &w.attn_norm, g.rms_eps), w, g, pos);
    let y: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
    let ffn = ffn_forward(
        &rmsnorm(&y, &w.ffn_norm, g.rms_eps),
        &w.ffn_gate,
        &w.ffn_up,
        &w.ffn_down,
    );
    y.iter().zip(&ffn).map(|(a, b)| a + b).collect()
}

/// Project a final hidden state to `n_vocab` logits (`out` is n_vocab×d_model).
pub fn logits_from_hidden(x: &[f32], out_w: &QMat, g: &Geometry, n_vocab: usize) -> Vec<f32> {
    debug_assert_eq!(out_w.rows(), n_vocab, "logits_from_hidden: vocab rows");
    debug_assert_eq!(out_w.cols(), g.d_model, "logits_from_hidden: hidden width");
    out_w.matvec(x)
}

/// Per-layer K/V cache (row-major `seq × kv_dim`), appended one position per decode step.
/// This is the incremental-decoding substrate for `PH-307` (the paged version lives in
/// `crate::infer::PagedKvCache`; a dense linear buffer is enough for the shape gate).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Head-only causal attention over the KV cache (no FFN, no residual): rmsnorm(`x`) → q/k/v
/// → append K/V to `cache` → head-major causal attention over `0..=pos` → o-projection.
/// Extracted so `block_forward_cached` and the MoE block (`PH-403b`) share one attention core.
#[allow(clippy::needless_range_loop)]
pub fn attention_forward_cached(
    x: &[f32],
    w: &BlockWeights,
    g: &Geometry,
    pos: usize,
    cache: &mut LayerKv,
) -> Vec<f32> {
    assert_eq!(x.len(), g.d_model, "attention_forward_cached: hidden width");
    let kv_dim = g.n_kv_heads * g.head_dim;
    let q_dim = g.n_heads * g.head_dim;
    let n = rmsnorm(x, &w.attn_norm, g.rms_eps);
    let q = w.q.matvec(&n);
    let k = w.k.matvec(&n);
    let v = w.v.matvec(&n);
    cache.k.extend_from_slice(&k);
    cache.v.extend_from_slice(&v);
    let scale = 1.0 / (g.head_dim as f32).sqrt();
    let mut ctx = vec![0.0f32; q_dim];
    for h in 0..g.n_heads {
        let hs = h * g.head_dim;
        let kv_h = (h * g.n_kv_heads / g.n_heads) * g.head_dim;
        let q_r = rope(&q[hs..hs + g.head_dim], pos, g.rope_freq_base);
        let probs = softmax(
            &(0..=pos)
                .map(|j| {
                    let k_raw = &cache.k[j * kv_dim + kv_h..j * kv_dim + kv_h + g.head_dim];
                    let k_r = rope(k_raw, j, g.rope_freq_base);
                    q_r.iter().zip(&k_r).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect::<Vec<f32>>(),
        );
        for d in 0..g.head_dim {
            let mut acc = 0.0f32;
            for j in 0..=pos {
                acc += probs[j] * cache.v[j * kv_dim + kv_h + d];
            }
            ctx[hs + d] = acc;
        }
    }
    w.o.matvec(&ctx)
}

/// Decode **one** new token at `pos`, attending over every cached key/value (causal), and
/// append this step's K/V into `cache`. Mathematically identical to running
/// [`block_forward_seq`] over the whole prefix — RoPE is applied per stored position and
/// scores accumulate in the same order — so `cache.len()/kv_dim` must equal `pos + 1`.
#[allow(clippy::needless_range_loop)]
pub fn block_forward_cached(
    x: &[f32],
    w: &BlockWeights,
    g: &Geometry,
    pos: usize,
    cache: &mut LayerKv,
) -> Vec<f32> {
    let attn = attention_forward_cached(x, w, g, pos, cache);
    let y: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
    let ffn = ffn_forward(
        &rmsnorm(&y, &w.ffn_norm, g.rms_eps),
        &w.ffn_gate,
        &w.ffn_up,
        &w.ffn_down,
    );
    y.iter().zip(&ffn).map(|(a, b)| a + b).collect()
}

/// Multi-position prefill: one transformer block over a `seq × d_model` row-major buffer.
/// Attention is causal + GQA + RoPE (head-major); FFN is applied per row. Returns a
/// `seq × d_model` buffer. `pos_start` is the absolute position of row 0 (so prompts
/// continue RoPE from the previous context on decode steps).
pub fn block_forward_seq(
    x_seq: &[f32],
    w: &BlockWeights,
    g: &Geometry,
    pos_start: usize,
) -> Vec<f32> {
    let seq = x_seq.len() / g.d_model;
    assert_eq!(
        seq * g.d_model,
        x_seq.len(),
        "block_forward_seq: seq × d_model"
    );

    // Attention: rmsnorm(x) per row → q/k/v projection per row → head-major causal
    // attention over [0..=i] at position `pos_start + i`, RoPE on q/k → per-row concat.
    let kv_dim = g.n_kv_heads * g.head_dim;
    let q_dim = g.n_heads * g.head_dim;
    let mut x_norm = vec![0.0f32; seq * g.d_model];
    let mut q_all = vec![0.0f32; seq * q_dim];
    let mut k_all = vec![0.0f32; seq * kv_dim];
    let mut v_all = vec![0.0f32; seq * kv_dim];
    for (i, row) in x_seq.chunks_exact(g.d_model).enumerate() {
        let n = rmsnorm(row, &w.attn_norm, g.rms_eps);
        x_norm[i * g.d_model..(i + 1) * g.d_model].copy_from_slice(&n);
        q_all[i * q_dim..(i + 1) * q_dim].copy_from_slice(&w.q.matvec(&n));
        k_all[i * kv_dim..(i + 1) * kv_dim].copy_from_slice(&w.k.matvec(&n));
        v_all[i * kv_dim..(i + 1) * kv_dim].copy_from_slice(&w.v.matvec(&n));
    }
    let scale = 1.0 / (g.head_dim as f32).sqrt();
    let mut attn_ctx = vec![0.0f32; seq * q_dim];
    for h in 0..g.n_heads {
        let hs = h * g.head_dim;
        let kv_h = (h * g.n_kv_heads / g.n_heads) * g.head_dim;
        // Pre-rotate K once per head; V needs no rotation.
        let mut k_rot: Vec<Vec<f32>> = Vec::with_capacity(seq);
        for j in 0..seq {
            let k_raw = &k_all[j * kv_dim + kv_h..j * kv_dim + kv_h + g.head_dim];
            k_rot.push(rope(k_raw, pos_start + j, g.rope_freq_base));
        }
        for i in 0..seq {
            let q_raw = &q_all[i * q_dim + hs..i * q_dim + hs + g.head_dim];
            let q_r = rope(q_raw, pos_start + i, g.rope_freq_base);
            // Causal scores over 0..=i, softmax, then weighted sum of v-rows.
            let scores: Vec<f32> = (0..=i)
                .map(|j| q_r.iter().zip(&k_rot[j]).map(|(a, b)| a * b).sum::<f32>() * scale)
                .collect();
            let probs = softmax(&scores);
            for d in 0..g.head_dim {
                let mut acc = 0.0f32;
                for (j, p) in probs.iter().enumerate() {
                    acc += *p * v_all[j * kv_dim + kv_h + d];
                }
                attn_ctx[i * q_dim + hs + d] = acc;
            }
        }
    }
    // Output projection per row + residual.
    let mut y = vec![0.0f32; seq * g.d_model];
    for i in 0..seq {
        let o_row = w.o.matvec(&attn_ctx[i * q_dim..(i + 1) * q_dim]);
        for d in 0..g.d_model {
            y[i * g.d_model + d] = x_seq[i * g.d_model + d] + o_row[d];
        }
    }
    // FFN per row (SwiGLU): rmsnorm → gate/up/down, add residual.
    let mut z = vec![0.0f32; seq * g.d_model];
    for i in 0..seq {
        let row = &y[i * g.d_model..(i + 1) * g.d_model];
        let ffn = ffn_forward(
            &rmsnorm(row, &w.ffn_norm, g.rms_eps),
            &w.ffn_gate,
            &w.ffn_up,
            &w.ffn_down,
        );
        for d in 0..g.d_model {
            z[i * g.d_model + d] = row[d] + ffn[d];
        }
    }
    z
}

/// Final RMS-norm over the last row, then project to logits.
pub fn last_logits(
    hidden_seq: &[f32],
    g: &Geometry,
    out_norm: &[f32],
    out_w: &QMat,
    n_vocab: usize,
) -> Vec<f32> {
    let seq = hidden_seq.len() / g.d_model;
    assert!(seq >= 1, "last_logits: at least one row");
    let last = &hidden_seq[(seq - 1) * g.d_model..seq * g.d_model];
    let normed = rmsnorm(last, out_norm, g.rms_eps);
    logits_from_hidden(&normed, out_w, g, n_vocab)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::argmax;

    fn geom() -> Geometry {
        Geometry::new(4, 2, 1, 8, 10_000.0, 1e-5) // d_model4 heads2 kv1 head2 ff8
    }

    fn zeros_block() -> BlockWeights {
        BlockWeights {
            attn_norm: vec![1.0; 4],
            q: QMat::from_f32(vec![0.0; 16], 4, 4),
            k: QMat::from_f32(vec![0.0; 8], 2, 4),
            v: QMat::from_f32(vec![0.0; 8], 2, 4),
            o: QMat::from_f32(vec![0.0; 16], 4, 4),
            ffn_norm: vec![1.0; 4],
            ffn_gate: QMat::from_f32(vec![0.0; 32], 8, 4),
            ffn_up: QMat::from_f32(vec![0.0; 32], 8, 4),
            ffn_down: QMat::from_f32(vec![0.0; 32], 4, 8),
        }
    }

    #[test]
    fn block_reduces_to_residual_when_branches_zeroed() {
        // o_proj=0 and ffn=0 ⇒ attention and FFN contribute nothing ⇒ x unchanged.
        let g = geom();
        let w = zeros_block();
        let x = [0.3, -0.1, 0.5, 0.2];
        let out = block_forward(&x, &w, &g, 0);
        assert_eq!(out, x.to_vec());
    }

    #[test]
    fn block_is_finite_and_shape_preserved_with_randomish_weights() {
        let g = geom();
        let w = BlockWeights {
            attn_norm: vec![1.0; 4],
            q: QMat::from_f32((0..16).map(|i| i as f32 * 0.01 - 0.07).collect(), 4, 4),
            k: QMat::from_f32((0..8).map(|i| i as f32 * 0.02 - 0.06).collect(), 2, 4),
            v: QMat::from_f32((0..8).map(|i| 0.1 - i as f32 * 0.03).collect(), 2, 4),
            o: QMat::from_f32((0..16).map(|i| i as f32 * 0.015 - 0.11).collect(), 4, 4),
            ffn_norm: vec![1.0; 4],
            ffn_gate: QMat::from_f32((0..32).map(|i| i as f32 * 0.01 - 0.15).collect(), 8, 4),
            ffn_up: QMat::from_f32((0..32).map(|i| 0.2 - i as f32 * 0.01).collect(), 8, 4),
            ffn_down: QMat::from_f32((0..32).map(|i| i as f32 * 0.005 - 0.08).collect(), 4, 8),
        };
        let out = block_forward(&[0.5, -0.5, 1.0, 0.0], &w, &g, 3);
        assert_eq!(out.len(), g.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn toy_forward_reaches_expected_greedy_token() {
        // A 1-block "model": embed → block (zeroed branches → residual-only) → output.
        let g = geom();
        let n_vocab = 5;
        let w = zeros_block();
        // embedding for token 2 = [1,2,0,0]; block leaves it (branches zeroed).
        let embed = vec![
            0.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            1.0, 2.0, 0.0, 0.0, // token 2  <-- query token
            0.0, 0.0, 0.0, 0.0, // token 3
            0.0, 0.0, 0.0, 0.0, // token 4
        ];
        let hidden = embed[2 * g.d_model..3 * g.d_model].to_vec();
        let after_block = block_forward(&hidden, &w, &g, 0);
        assert_eq!(after_block, hidden); // branches zeroed
                                         // output rows: token 4 dot [1,2,0,0] = largest.
        let mut out_w = vec![0.0f32; n_vocab * g.d_model];
        out_w[4 * g.d_model] = 1.0;
        out_w[4 * g.d_model + 1] = 1.0; // [1,1,0,0]·[1,2,0,0]=3 > others
        out_w[0] = 0.1; // token 0 row [0.1,0,..]·..=0.1
        let out_w = QMat::from_f32(out_w, n_vocab, g.d_model);
        let logits = logits_from_hidden(&after_block, &out_w, &g, n_vocab);
        assert_eq!(logits.len(), n_vocab);
        assert_eq!(argmax(&logits), Some(4));
    }

    #[test]
    fn geometry_head_dim_and_gqa() {
        let g = geom();
        assert_eq!(g.head_dim, 2);
        // kv head for query heads {0,1} with n_kv=1 must both map to kv head 0.
        let kv_of = |h: usize| h * g.n_kv_heads / g.n_heads;
        assert_eq!(kv_of(0), 0);
        assert_eq!(kv_of(1), 0);
    }

    fn nonzero_block() -> BlockWeights {
        // attention + FFN branches active, so causal context really matters.
        BlockWeights {
            attn_norm: vec![1.0; 4],
            q: QMat::from_f32((0..16).map(|i| i as f32 * 0.03 - 0.2).collect(), 4, 4),
            k: QMat::from_f32((0..8).map(|i| i as f32 * 0.02 - 0.06).collect(), 2, 4),
            v: QMat::from_f32((0..8).map(|i| 0.11 - i as f32 * 0.04).collect(), 2, 4),
            o: QMat::from_f32((0..16).map(|i| i as f32 * 0.015 - 0.1).collect(), 4, 4),
            ffn_norm: vec![1.0; 4],
            ffn_gate: QMat::from_f32((0..32).map(|i| i as f32 * 0.01 - 0.14).collect(), 8, 4),
            ffn_up: QMat::from_f32((0..32).map(|i| 0.2 - i as f32 * 0.012).collect(), 8, 4),
            ffn_down: QMat::from_f32((0..32).map(|i| i as f32 * 0.006 - 0.08).collect(), 4, 8),
        }
    }

    #[test]
    fn block_forward_cached_matches_seq_over_prefix() {
        // Feeding rows one-by-one through the KV cache reproduces block_forward_seq's
        // per-position outputs exactly (causal attention over 0..=pos).
        let g = geom();
        let w = nonzero_block();
        let x_seq: Vec<f32> = vec![
            0.5, -0.2, 0.1, 0.4, // pos 0
            -0.3, 0.7, 0.2, -0.1, // pos 1
            0.9, 0.1, -0.4, 0.3, // pos 2
        ];
        let seq = x_seq.len() / g.d_model;
        let reference = block_forward_seq(&x_seq, &w, &g, 0);
        let mut cache = LayerKv::default();
        for i in 0..seq {
            let row = &x_seq[i * g.d_model..(i + 1) * g.d_model];
            let out = block_forward_cached(row, &w, &g, i, &mut cache);
            let want = &reference[i * g.d_model..(i + 1) * g.d_model];
            for (a, b) in out.iter().zip(want) {
                assert!((a - b).abs() < 1e-5, "pos {i}: cached={a} seq={b}");
            }
        }
        assert_eq!(cache.k.len(), seq * (g.n_kv_heads * g.head_dim));
    }
}
