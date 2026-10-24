//! `qwen35` full-attention block (the every-`full_attention_interval`-th layer of the
//! MiniCPM-V 4.7 hybrid) — band B13 `PH-1303`.
//!
//! The GQA core matches the qwen2 path (`attention_forward_cached`) with three qwen35
//! twists read straight off the GGUF (see `docs/research/minicpm-v-4.7.md`):
//!
//! 1. **256-wide heads** (`attention.key_length`, not `d_model / n_heads`) — 8 q heads,
//!    2 kv heads; `attn_q.weight` is `[d_model, 2·q_dim]`: the SECOND half is a
//!    per-element **output gate** `o ·= sigmoid(gate)` applied after attention and
//!    before `attn_output` (the Qwen3-Next `attention_output_gate`).
//! 2. **QK-RMSNorm**: `attn_q_norm`/`attn_k_norm` are per-head RMSNorm weights of
//!    length `hd`, applied after projection and before RoPE.
//! 3. **Partial sectioned RoPE**: `rope.dimension_count = 64` of 256 dims, NeoX
//!    rotate-half pairs `(i, i + 32)`; `rope.dimension_sections = [11, 11, 10, 0]` +
//!    `rope.section_order = "zyxt"` split the 32 pairs into M-RoPE position slots
//!    (llama.cpp PR #29416: each section is fed the t/y/x/z slot in section order —
//!    for TEXT all slots equal the token position, so this module degrades to the
//!    standard partial RoPE; the multi-slot split arrives with the image prefix,
//!    `PH-1204`).
//!
//! As in [`crate::gdn`], [`q35_prefill`] and [`q35_decode_step`] share one row kernel in
//! one fixed summation order, so a fresh prefill is bit-identical to stepping (tested);
//! keys are stored already normed+rotated (the `PH-203g` policy).

use crate::core::Result;
use crate::infer::rmsnorm;
use crate::kernel::{axpy_f32_dispatch, dot_f32_dispatch};
use crate::model::ModelConfig;
use crate::moe::softmax_in_place;
use crate::qmat::QMat;

/// Geometry of one qwen35 full-attention block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Q35Geom {
    pub n_heads: usize,
    pub n_kv: usize,
    /// Per-head width (`attention.key_length`; qwen35: 256).
    pub hd: usize,
    /// Rotated prefix of each head (`rope.dimension_count`; qwen35: 64).
    pub rope_dim: usize,
    pub rope_base: f32,
    pub eps: f32,
}

impl Q35Geom {
    /// Build the full-attention geometry from a config: only a hybrid arch with an
    /// explicit `key_length` and a partial-RoPE width qualifies.
    pub fn from_config(c: &ModelConfig) -> Option<Self> {
        let g = Self {
            n_heads: c.n_heads,
            n_kv: c.n_kv_heads,
            hd: c.key_length,
            rope_dim: c.rope_dimension_count,
            rope_base: c.rope_freq_base,
            eps: c.rms_eps,
        };
        (c.is_hybrid()
            && g.hd > 0
            && g.rope_dim >= 2
            && g.rope_dim <= g.hd
            && g.rope_dim % 2 == 0
            && g.n_heads > 0
            && g.n_kv > 0
            && g.n_heads % g.n_kv == 0)
            .then_some(g)
    }

    /// Total q (and gate) width: `n_heads · hd`.
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.hd
    }

    /// K/V row width: `n_kv · hd`.
    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.hd
    }
}

/// Borrowed projections of one full-attention block (`blk.N.attn_*`).
pub struct Q35Weights<'a> {
    /// `attn_q.weight` — `[d_model, 2·q_dim]` (q rows, then the gate half).
    pub q: &'a QMat,
    /// `attn_k.weight` — `[d_model, kv_dim]`.
    pub k: &'a QMat,
    /// `attn_v.weight` — `[d_model, kv_dim]`.
    pub v: &'a QMat,
    /// `attn_output.weight` — `[q_dim, d_model]`.
    pub o: &'a QMat,
    /// `attn_q_norm.weight` — `[hd]` per-head RMSNorm scale.
    pub q_norm: &'a [f32],
    /// `attn_k_norm.weight` — `[hd]`.
    pub k_norm: &'a [f32],
}

/// Per-layer growing K/V cache for a qwen35 full block (normed + rotated keys).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Q35Kv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl Q35Kv {
    /// Tokens cached so far.
    pub fn len(&self, g: &Q35Geom) -> usize {
        self.k.len() / g.kv_dim()
    }

    /// Whether nothing is cached yet.
    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }
}

/// The one gate convention `PH-1305` pins against the oracle: the attention output gate
/// is the sigmoid of the second half of the q projection (Qwen3-Next `attention_output_gate`).
fn gate_act(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Per-head RMSNorm into `dst` (`rmsnorm` from B3 keeps its exact operation order).
fn head_norm(head: &[f32], weight: &[f32], eps: f32, dst: &mut [f32]) {
    dst.copy_from_slice(&rmsnorm(head, weight, eps));
}

/// NeoX rotate-half over the first `rope_dim` dims of one head (`pos`-th table entry).
/// Text degeneracy of the `[11,11,10,0]`/`zyxt` M-RoPE split: every section sees the
/// same scalar position, so plain partial RoPE is exact for 1-D streams.
fn rope_partial_inplace(head: &mut [f32], pos: usize, g: &Q35Geom) {
    let half = g.rope_dim / 2;
    for i in 0..half {
        let theta = (i as f32) * 2.0 / (g.rope_dim as f32);
        let inv = g.rope_base.powf(-theta);
        let (s, c) = ((pos as f32) * inv).sin_cos();
        let a = head[i];
        let b = head[i + half];
        head[i] = a * c - b * s;
        head[i + half] = b * c + a * s;
    }
}

/// The row kernel both passes share: attend head `h` of `q_head` (already normed and
/// rotated) over cache rows `0..=pos`, writing the raw `hd` output into `dst`.
fn attend_row(q_head: &[f32], kv: &Q35Kv, g: &Q35Geom, h: usize, pos: usize, dst: &mut [f32]) {
    let kv_h = (h * g.n_kv / g.n_heads) * g.hd;
    let scale = 1.0 / (g.hd as f32).sqrt();
    let span = pos + 1;
    let mut scores = vec![0.0f32; span];
    for (j, s) in scores.iter_mut().enumerate() {
        let k_row = &kv.k[j * g.kv_dim() + kv_h..j * g.kv_dim() + kv_h + g.hd];
        *s = dot_f32_dispatch(q_head, k_row) * scale;
    }
    softmax_in_place(&mut scores);
    dst.fill(0.0);
    for (j, &p) in scores.iter().enumerate() {
        let v_row = &kv.v[j * g.kv_dim() + kv_h..j * g.kv_dim() + kv_h + g.hd];
        axpy_f32_dispatch(dst, v_row, p);
    }
}

/// Project + norm + rotate one row's q/k/v and append the K/V rows to the cache.
/// Returns the normed+rotated q (width `2·q_dim`: q rows then the raw gate rows).
fn project_rope_append(
    w: &Q35Weights<'_>,
    g: &Q35Geom,
    x: &[f32],
    pos: usize,
    kv: &mut Q35Kv,
) -> Vec<f32> {
    let qg = w.q.matvec(x);
    let mut k = w.k.matvec(x);
    let v = w.v.matvec(x);
    let qd = g.q_dim();
    let mut out = vec![0.0f32; 2 * qd];
    let mut buf = vec![0.0f32; g.hd];
    // The gate half is copied raw (post-attention it feeds `gate_act`).
    out[qd..2 * qd].copy_from_slice(&qg[qd..2 * qd]);
    for h in 0..g.n_heads {
        head_norm(&qg[h * g.hd..(h + 1) * g.hd], w.q_norm, g.eps, &mut buf);
        rope_partial_inplace(&mut buf, pos, g);
        out[h * g.hd..(h + 1) * g.hd].copy_from_slice(&buf);
    }
    for h in 0..g.n_kv {
        head_norm(&k[h * g.hd..(h + 1) * g.hd], w.k_norm, g.eps, &mut buf);
        rope_partial_inplace(&mut buf, pos, g);
        k[h * g.hd..(h + 1) * g.hd].copy_from_slice(&buf);
    }
    kv.k.extend_from_slice(&k);
    kv.v.extend_from_slice(&v);
    out
}

/// One decode token through the block. Returns the `d_model` attention output
/// (residual/norm live in the runner, `PH-1304`).
pub fn q35_decode_step(
    w: &Q35Weights<'_>,
    g: &Q35Geom,
    x: &[f32],
    pos: usize,
    kv: &mut Q35Kv,
) -> Result<Vec<f32>> {
    if w.q.rows() != 2 * g.q_dim() || w.k.rows() != g.kv_dim() || w.v.rows() != g.kv_dim() {
        return Err(crate::core::EngineError {
            message: format!(
                "q35: projection rows {:?} != [{}, {}, {}]",
                (w.q.rows(), w.k.rows(), w.v.rows()),
                2 * g.q_dim(),
                g.kv_dim(),
                g.kv_dim()
            ),
        });
    }
    let qg = project_rope_append(w, g, x, pos, kv);
    let qd = g.q_dim();
    let mut core = vec![0.0f32; qd];
    let mut o = vec![0.0f32; g.hd];
    for h in 0..g.n_heads {
        attend_row(&qg[h * g.hd..(h + 1) * g.hd], kv, g, h, pos, &mut o);
        for j in 0..g.hd {
            core[h * g.hd + j] = o[j] * gate_act(qg[qd + h * g.hd + j]);
        }
    }
    Ok(w.o.matvec(&core))
}

/// Prefill `rows` tokens starting at absolute position `pos_start`: projections are done
/// row-wise through the SAME [`project_rope_append`] (ascending positions), then each
/// row attends through the SAME [`attend_row`] — so the result is bit-identical to
/// calling [`q35_decode_step`] per token (unit-tested). Appends everything to `kv`.
pub fn q35_prefill(
    w: &Q35Weights<'_>,
    g: &Q35Geom,
    x_seq: &[f32],
    rows: usize,
    pos_start: usize,
    kv: &mut Q35Kv,
) -> Vec<f32> {
    debug_assert_eq!(x_seq.len(), rows * w.o.rows(), "q35 prefill: X rows");
    let dm = w.o.rows();
    let qd = g.q_dim();
    let mut qgs = Vec::with_capacity(rows);
    for i in 0..rows {
        qgs.push(project_rope_append(
            w,
            g,
            &x_seq[i * dm..(i + 1) * dm],
            pos_start + i,
            kv,
        ));
    }
    let mut core = vec![0.0f32; rows * qd];
    let mut o = vec![0.0f32; g.hd];
    for i in 0..rows {
        let pos = pos_start + i;
        let qg = &qgs[i];
        for h in 0..g.n_heads {
            attend_row(&qg[h * g.hd..(h + 1) * g.hd], kv, g, h, pos, &mut o);
            for j in 0..g.hd {
                core[i * qd + h * g.hd + j] = o[j] * gate_act(qg[qd + h * g.hd + j]);
            }
        }
    }
    w.o.matmul(&core, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::rope_in_place;

    fn qwen35_cfg() -> ModelConfig {
        ModelConfig {
            arch: "qwen35".into(),
            n_layers: 24,
            d_model: 1024,
            n_heads: 8,
            n_kv_heads: 2,
            d_ff: 3584,
            n_vocab: 248094,
            rope_freq_base: 1e7,
            rms_eps: 1e-6,
            n_experts: 0,
            n_experts_used: 0,
            eos_token_id: None,
            key_length: 256,
            full_attention_interval: 4,
            recurrent_layers: vec![],
            ssm_conv_kernel: 4,
            ssm_state_size: 128,
            ssm_group_count: 16,
            ssm_time_step_rank: 16,
            ssm_inner_size: 2048,
            rope_dimension_count: 64,
            rope_sections: vec![11, 11, 10, 0],
            rope_section_order: "zyxt".into(),
        }
    }

    #[test]
    fn geom_contract_matches_the_real_file() {
        let g = Q35Geom::from_config(&qwen35_cfg()).expect("qwen35 full-attn geom");
        assert_eq!(g.q_dim(), 2048);
        assert_eq!(g.kv_dim(), 512);
        assert_eq!(g.rope_dim, 64);
        // Plain qwen2 (no key_length) must NOT qualify as a q35 block.
        let mut c = qwen35_cfg();
        c.key_length = 0;
        assert_eq!(Q35Geom::from_config(&c), None);
    }

    fn toy() -> Q35Geom {
        Q35Geom {
            n_heads: 2,
            n_kv: 1,
            hd: 4,
            rope_dim: 4,
            rope_base: 100.0,
            eps: 1e-5,
        }
    }

    struct Toy {
        q: QMat,
        k: QMat,
        v: QMat,
        o: QMat,
        qn: Vec<f32>,
        kn: Vec<f32>,
    }

    fn toy_weights() -> Toy {
        let g = toy();
        let dm = 3usize;
        let ramp = |m: usize, n: usize| -> Vec<f32> {
            (0..m * n)
                .map(|i| ((i as i32 * 7 % 13) as f32) * 0.05 - 0.25)
                .collect()
        };
        Toy {
            q: QMat::from_f32(ramp(2 * g.q_dim(), dm), 2 * g.q_dim(), dm),
            k: QMat::from_f32(ramp(g.kv_dim(), dm), g.kv_dim(), dm),
            v: QMat::from_f32(
                ramp(g.kv_dim(), dm)
                    .iter()
                    .map(|x| x * 1.7)
                    .collect::<Vec<_>>(),
                g.kv_dim(),
                dm,
            ),
            o: QMat::from_f32(ramp(dm, g.q_dim()), dm, g.q_dim()),
            qn: vec![1.0, 0.9, 1.1, 0.8],
            kn: vec![1.0, 1.0, 1.0, 1.0],
        }
    }

    fn toy_views(t: &Toy) -> Q35Weights<'_> {
        Q35Weights {
            q: &t.q,
            k: &t.k,
            v: &t.v,
            o: &t.o,
            q_norm: &t.qn,
            k_norm: &t.kn,
        }
    }

    #[test]
    fn rope_is_identity_at_pos_zero_and_full_width_matches_infer() {
        let g = toy();
        let mut h = [1.0f32, 2.0, 3.0, 4.0];
        let before = h;
        rope_partial_inplace(&mut h, 0, &g);
        assert_eq!(h, before, "pos 0 must not move a value");
        // With rope_dim == hd this is exactly infer::rope's NeoX rotate-half.
        let freqs = crate::infer::rope_freqs(g.hd, g.rope_base);
        let mut a = [0.4f32, -0.7, 0.1, 0.9];
        let mut b = a;
        rope_in_place(&mut a, 5, &freqs);
        rope_partial_inplace(&mut b, 5, &g);
        for i in 0..4 {
            assert!(
                (a[i] - b[i]).abs() < 1e-6,
                "dim {i}: infer {:?} vs q35 {:?}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn prefill_is_bit_identical_to_decode_steps() {
        let g = toy();
        let t = toy_weights();
        let w = toy_views(&t);
        let dm = 3usize;
        let rows = 7;
        let x: Vec<f32> = (0..rows * dm)
            .map(|i| ((i as i32 * 11 % 17) as f32) * 0.06 - 0.4)
            .collect();
        let mut kv_p = Q35Kv::default();
        let pre = q35_prefill(&w, &g, &x, rows, 0, &mut kv_p);
        let mut kv_d = Q35Kv::default();
        let mut steps = Vec::new();
        for i in 0..rows {
            steps.extend(q35_decode_step(&w, &g, &x[i * dm..(i + 1) * dm], i, &mut kv_d).unwrap());
        }
        assert_eq!(pre, steps, "prefill drifted from stepping");
        assert_eq!(kv_p, kv_d, "caches must be bit-identical");
    }

    #[test]
    fn zero_input_yields_zero_block_output() {
        let g = toy();
        let t = toy_weights();
        let w = toy_views(&t);
        let mut kv = Q35Kv::default();
        let out = q35_decode_step(&w, &g, &[0.0f32; 3], 0, &mut kv).unwrap();
        assert_eq!(out, vec![0.0f32; 3], "0 v ⇒ 0 core; o·0 = 0");
    }

    #[test]
    fn mismatched_projection_shapes_are_rejected() {
        let g = toy();
        let t = toy_weights();
        let mut w = Q35Weights {
            q: &t.q,
            k: &t.k,
            v: &t.v,
            o: &t.o,
            q_norm: &t.qn,
            k_norm: &t.kn,
        };
        let bad = QMat::from_f32(vec![0.0f32; 24], 8, 3);
        w.q = &bad;
        let mut kv = Q35Kv::default();
        let e = q35_decode_step(&w, &g, &[0.0f32; 3], 0, &mut kv).expect_err("rows checked");
        assert!(e.message.contains("projection rows"), "{e:?}");
    }
}
