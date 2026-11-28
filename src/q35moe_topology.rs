//! A2 (pure prep for `PH-1206`): the topology/state half of a qwen35moe block.
//!
//! Pure structures only — residual stream, RMSNorm placement, the three
//! attention-state kinds used by the hybrid arch (dense KV, qwen35 full KV,
//! GDN recurrent), and the routed/shared FFN half. Callers supply the toy
//! weights and the expert loader; this module never loads a model, never
//! decodes a token and makes no quality claim (`G13` is blocked).
//!
//! `PH-403b` gate: with one routed expert, softmax top-1 and no shared expert,
//! [`MoeBlockTopology::compose`] reproduces `moe::moe_block_forward` within
//! 1e-5, including the carried attention state.
//!
//! Arch note: qwen35 calls its post-attention norm `post_attention_norm` and
//! has no separate `ffn_norm` field. That slice is the `norm` argument of
//! `compose`, playing the same role as dense `BlockWeights::ffn_norm`.

use crate::cache::ExpertCache;
use crate::forward::LayerKv;
use crate::gdn::{GdnGeom, GdnState};
use crate::infer::rmsnorm;
use crate::moe::{moe_layer_forward_cached, ExpertWeights, RouterConfig};
use crate::q35::Q35Kv;

/// Attention state carried between qwen35moe block steps.
#[derive(Debug, Clone, PartialEq)]
pub enum MoeAttnState {
    /// Dense GQA K/V — the parity path shared with `PH-403b`.
    Dense(LayerKv),
    /// qwen35 full-attention growing K/V.
    Full(Q35Kv),
    /// qwen35 GDN recurrent state (conv window + delta-state).
    Recurrent(GdnState),
}

impl MoeAttnState {
    pub fn dense() -> Self {
        Self::Dense(LayerKv::default())
    }
    pub fn full() -> Self {
        Self::Full(Q35Kv::default())
    }
    pub fn recurrent(geom: &GdnGeom) -> Self {
        Self::Recurrent(GdnState::new(geom))
    }
}

/// Routed/shared FFN half of a qwen35moe block. The expert bytes are supplied
/// by the caller's `load` closure, so this struct stays node- and cache-free.
#[derive(Debug, Clone, Copy)]
pub struct MoeFfnTopology<'a> {
    pub router_w: &'a [f32],
    pub n_experts: usize,
    pub layer: usize,
    pub shared: Option<&'a ExpertWeights>,
    pub inter: usize,
    pub cfg: RouterConfig,
}

/// Per-step inputs: hidden, attention output and the post-attention norm.
#[derive(Debug, Clone, Copy)]
pub struct MoeBlockInput<'a> {
    pub x: &'a [f32],
    pub attn_out: &'a [f32],
    pub post_norm: &'a [f32],
    pub eps: f32,
}

/// Per-layer state: the residual stream plus one attention-state kind.
#[derive(Debug, Clone)]
pub struct MoeBlockTopology {
    pub residual: Vec<f32>,
    pub attn: MoeAttnState,
}

impl MoeBlockTopology {
    /// Zero residual + the chosen attention state.
    pub fn new(d_model: usize, attn: MoeAttnState) -> Self {
        Self {
            residual: vec![0.0f32; d_model],
            attn,
        }
    }

    /// Compose one block over an attention output:
    /// `h = x + attn_out`; `normed = RMSNorm(h, post_attention_norm, eps)`;
    /// `out = h + MoE FFN(normed)`. The updated residual is stored in `self`.
    pub fn compose<L>(
        &mut self,
        input: MoeBlockInput<'_>,
        ffn: &MoeFfnTopology<'_>,
        cache: &mut ExpertCache,
        load: L,
    ) -> &[f32]
    where
        L: FnMut(usize) -> ExpertWeights,
    {
        let MoeBlockInput {
            x,
            attn_out,
            post_norm,
            eps,
        } = input;
        assert_eq!(
            x.len(),
            self.residual.len(),
            "compose: hidden width must match topology"
        );
        assert_eq!(attn_out.len(), x.len(), "compose: attention width mismatch");
        assert_eq!(post_norm.len(), x.len(), "compose: norm width mismatch");

        let h: Vec<f32> = x.iter().zip(attn_out).map(|(a, b)| a + b).collect();
        let normed = rmsnorm(&h, post_norm, eps);
        let moe = moe_layer_forward_cached(
            &normed,
            ffn.router_w,
            ffn.n_experts,
            ffn.layer,
            cache,
            ffn.shared,
            ffn.inter,
            ffn.cfg,
            load,
        );
        assert_eq!(moe.len(), h.len(), "compose: FFN width mismatch");
        self.residual = h.iter().zip(&moe).map(|(a, b)| a + b).collect();
        &self.residual
    }
}
