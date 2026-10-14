//! Transformer forward composition — band B3 `PH-305d`.
//!
//! Wires the unit-tested primitives into one decode step: `rmsnorm → multi-head GQA
//! causal attention (RoPE) → residual → rmsnorm → SwiGLU FFN → residual`, then a final
//! projection to logits. Verified end-to-end with a deterministic toy model (zeroed
//! projections reduce a block to its residual path; a crafted embed/output pair yields a
//! known greedy token). Real Qwen numeric parity vs llama.cpp is `PH-405` (needs a
//! reference + tensor-layout confirmation); this slice proves the *plumbing* composes.

use crate::infer::{gqa_repeat_kv, rmsnorm, rope};
use crate::moe::{expert_ffn, matvec, softmax};

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
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub o: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    /// SwiGLU: gate/up are d_ff×d_model, down is d_model×d_ff.
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
}

/// Single-token causal attention (a decode step attending over a 1-position span): each
/// head's softmax over its own key is 1.0, so the context equals the (GQA-broadcast) value.
/// Q/K are still RoPE-rotated and scored, exercising the real path.
fn attention_decode(x_norm: &[f32], w: &BlockWeights, g: &Geometry, pos: usize) -> Vec<f32> {
    let q = matvec(&w.q, g.d_model, g.d_model, x_norm);
    let kv_dim = g.n_kv_heads * g.head_dim;
    let k = matvec(&w.k, kv_dim, g.d_model, x_norm);
    let v = matvec(&w.v, kv_dim, g.d_model, x_norm);
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
    matvec(&w.o, g.d_model, g.n_heads * g.head_dim, &ctx)
}

/// One transformer block decode step: attention + FFN with residuals.
pub fn block_forward(x: &[f32], w: &BlockWeights, g: &Geometry, pos: usize) -> Vec<f32> {
    assert_eq!(x.len(), g.d_model, "block_forward: hidden width");
    let attn = attention_decode(&rmsnorm(x, &w.attn_norm, g.rms_eps), w, g, pos);
    let y: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
    let ffn = expert_ffn(
        &rmsnorm(&y, &w.ffn_norm, g.rms_eps),
        &w.ffn_gate,
        &w.ffn_up,
        &w.ffn_down,
        g.d_model,
        g.d_ff,
    );
    y.iter().zip(&ffn).map(|(a, b)| a + b).collect()
}

/// Project a final hidden state to `n_vocab` logits (`out` is n_vocab×d_model).
pub fn logits_from_hidden(x: &[f32], out_w: &[f32], g: &Geometry, n_vocab: usize) -> Vec<f32> {
    matvec(out_w, n_vocab, g.d_model, x)
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
            q: vec![0.0; 16],
            k: vec![0.0; 8],
            v: vec![0.0; 8],
            o: vec![0.0; 16],
            ffn_norm: vec![1.0; 4],
            ffn_gate: vec![0.0; 32],
            ffn_up: vec![0.0; 32],
            ffn_down: vec![0.0; 32],
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
            q: (0..16).map(|i| i as f32 * 0.01 - 0.07).collect(),
            k: (0..8).map(|i| i as f32 * 0.02 - 0.06).collect(),
            v: (0..8).map(|i| 0.1 - i as f32 * 0.03).collect(),
            o: (0..16).map(|i| i as f32 * 0.015 - 0.11).collect(),
            ffn_norm: vec![1.0; 4],
            ffn_gate: (0..32).map(|i| i as f32 * 0.01 - 0.15).collect(),
            ffn_up: (0..32).map(|i| 0.2 - i as f32 * 0.01).collect(),
            ffn_down: (0..32).map(|i| i as f32 * 0.005 - 0.08).collect(),
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
}
