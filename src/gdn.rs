//! Gated DeltaNet (GDN) — the recurrent attention block of the `qwen35` hybrid
//! (MiniCPM-V 4.7-1B text core, band B13 `PH-1302`).
//!
//! Every recurrent layer holds a per-group matrix state `S ∈ R^(dk×dv)` instead of a KV
//! cache: each token decays the state, writes the **delta correction** of the new value
//! against what the state already predicts for the key, and reads out with the query —
//! all `O(dk·dv)` per token, so decode is O(1) in context length (the differentiator
//! after MoE; full-attention layers every `full_attention_interval`-th come in
//! `PH-1303`).
//!
//! The arithmetic is fixed to one order of operations shared by both passes:
//! [`gdn_prefill`] projects + convolves the whole sequence, then walks the tokens in
//! order through the same [`delta_step`] that [`gdn_decode_step`] runs — so a fresh
//! prefill and a step-by-step decode land on bit-identical states (unit-tested).
//!
//! Constants follow fla/Qwen3-Next Gated DeltaNet: L2-normalized q/k heads, query scale
//! `dk^-0.5`, per-group log-decay `g = ssm_a·softplus(a + dt_bias)` (the stored `ssm_a`
//! is already `-exp(A_log)`), `beta =
//! sigmoid(b)`, and a **gated** head RMSNorm `rmsnorm(o)·weight·swish(z)` before
//! `ssm_out`. `PH-1305` pins every constant against the llama.cpp-omni oracle; where a
//! convention could differ (swish vs sigmoid gate), the code says so in ONE place.
//!
//! Pure `std`, no I/O — weight tensors are borrowed via [`GdnWeights`]; the GGUF
//! wiring lands with the runner slice (`PH-1304`).

use crate::model::ModelConfig;
use crate::qmat::QMat;

/// Per-group geometry of one GDN block (from `qwen35.ssm.*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnGeom {
    /// `ssm.group_count` — value/query/key head count for the recurrence.
    pub groups: usize,
    /// `ssm.state_size` — key head width (`dk`).
    pub dk: usize,
    /// value head width (`dv`); qwen35 keeps `dk == dv`.
    pub dv: usize,
    /// `ssm.conv_kernel` — depthwise causal conv width.
    pub conv: usize,
    /// `ssm.inner_size` — one of q/k/v (and the gate) width: `groups · dk`.
    pub inner: usize,
}

impl GdnGeom {
    /// The GDN geometry a config implies, or `None` when the file is not hybrid
    /// (any required `ssm.*` key missing ⇒ the block does not exist).
    pub fn from_config(c: &ModelConfig) -> Option<Self> {
        let g = Self {
            groups: c.ssm_group_count,
            dk: c.ssm_state_size,
            dv: c.ssm_state_size,
            conv: c.ssm_conv_kernel,
            inner: c.ssm_inner_size,
        };
        (g.groups > 0 && g.dk > 0 && g.conv > 0 && g.inner == g.groups * g.dk).then_some(g)
    }

    /// Width of the mixed `attn_qkv` projection (q ‖ k ‖ v).
    pub fn qkv_dim(&self) -> usize {
        3 * self.inner
    }

    /// Bytes of the per-sequence recurrent state: `groups · dk · dv · f32` (the matrix
    /// state is why decode is O(1) — one fixed blob replaces the growing KV cache).
    pub fn state_bytes(&self) -> usize {
        self.groups * self.dk * self.dv * 4
    }
}

/// The eight projections/vectors of one GDN block, borrowed from the runner.
pub struct GdnWeights<'a> {
    /// `blk.N.attn_qkv` — `[3·inner, d_model]`.
    pub qkv: &'a QMat,
    /// `blk.N.attn_gate` — the gated-RMSNorm gate `z`, `[inner, d_model]`.
    pub gate: &'a QMat,
    /// `blk.N.ssm_alpha.weight` — log-decay projection `a`, `[groups, d_model]`.
    pub alpha: &'a QMat,
    /// `blk.N.ssm_beta.weight` — write-strength projection `b`, `[groups, d_model]`.
    pub beta: &'a QMat,
    /// `blk.N.ssm_out.weight` — output projection, `[d_model, inner]`.
    pub out: &'a QMat,
    /// `blk.N.ssm_a` — stored **already negated-and-exponentiated** as `-exp(A_log)`
    /// (OpenBMB converter `conversion/qwen.py:297`); the oracle multiplies it straight
    /// into the softplus (`qwen35.cpp:379`), so we must not re-apply `exp`, `[groups]`.
    pub a_log: &'a [f32],
    /// `blk.N.ssm_dt.bias` — `[groups]`.
    pub dt_bias: &'a [f32],
    /// `blk.N.ssm_conv1d.weight` — `[conv][qkv_dim]` (ggml ne0 = kernel ⇒ channel-major
    /// bytes read `conv_w[c · conv + j]`), no bias on this arch.
    pub conv_w: &'a [f32],
    /// `blk.N.ssm_norm.weight` — `[dv]` head RMSNorm scale.
    pub norm_w: &'a [f32],
    /// RMS epsilon from the config.
    pub eps: f32,
}

/// The carried recurrent state of one GDN block for one sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct GdnState {
    /// Last `conv-1` mixed-qkv rows, oldest first (the causal conv window).
    conv_win: Vec<f32>,
    /// `groups` row-major `dk × dv` matrices.
    s: Vec<f32>,
}

impl GdnState {
    /// Zero state: `groups · dk · dv` matrix plus the conv window.
    pub fn new(g: &GdnGeom) -> Self {
        Self {
            conv_win: vec![0.0f32; g.conv.saturating_sub(1) * g.qkv_dim()],
            s: vec![0.0f32; g.groups * g.dk * g.dv],
        }
    }
}

/// Stable `softplus(x) = ln(1 + eˣ)` in f32 (fla's `F.softplus` cut-off is the same
/// plateau regime; beyond ±20 the tails are exact to f32).
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Per-token, per-group gated delta-rule step. Order of operations is the contract:
/// decay → predict → delta-write → rank-1 update → read.
///
/// `s` is the row-major `dk × dv` state, `o` a reused `dv` output buffer (zeroed here).
#[allow(clippy::too_many_arguments)]
fn delta_step(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
    o: &mut [f32],
    dk: usize,
    dv: usize,
) {
    debug_assert_eq!(s.len(), dk * dv);
    debug_assert_eq!((q.len(), k.len(), v.len(), o.len()), (dk, dk, dv, dv));
    let mut pred = [0.0f32; 128];
    debug_assert!(dv <= pred.len(), "test geoms only; dv = state_size");
    let pred = &mut pred[..dv];
    let mut vnew = [0.0f32; 128];
    let vnew = &mut vnew[..dv];
    for x in s.iter_mut() {
        *x *= decay;
    }
    for (i, &ki) in k.iter().enumerate() {
        let row = &s[i * dv..i * dv + dv];
        for (pj, &rv) in pred.iter_mut().zip(row) {
            *pj += ki * rv;
        }
    }
    for (j, (vn, &vj)) in vnew.iter_mut().zip(v).enumerate() {
        *vn = (vj - pred[j]) * beta;
    }
    for (i, &ki) in k.iter().enumerate() {
        let row = &mut s[i * dv..i * dv + dv];
        for (rv, &vn) in row.iter_mut().zip(vnew.iter()) {
            *rv += ki * vn;
        }
    }
    o.fill(0.0);
    for (i, &qi) in q.iter().enumerate() {
        let row = &s[i * dv..i * dv + dv];
        for (oj, &rv) in o.iter_mut().zip(row) {
            *oj += qi * rv;
        }
    }
}

/// The one constant that `PH-1305` must pin against the oracle: the gate activation of
/// the head RMSNorm (fla `FusedRMSNormGated` default = swish/SiLU).
fn gate_act(z: f32) -> f32 {
    z * (1.0 / (1.0 + (-z).exp())) // silu
}

/// Normalize a q/k head (L2) and fold the query scale `dk^-0.5` into q — fla's
/// `use_qk_l2norm_in_kernel`. Matches the oracle's `build_gdn_l2_norm`
/// (`src/models/models.h:14-18`): `ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrt(n))` collapses
/// to `x / sqrt(Σx² + eps)` — the eps sits **inside** the root (no `max(norm, 1e-6)` floor)
/// and there is no scale of its own; the kernel applies `1/sqrt(dk)` at the readout
/// (`ggml/src/ggml-cpu/ops.cpp`, `const float scale = 1.0f / sqrtf((float) S_v)`), which is
/// the same product in a different order, so folding it into q here is equivalent.
fn l2norm_scale(x: &mut [f32], scale: f32, eps: f32) {
    let sum = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() as f32;
    let f = scale / (sum + eps).sqrt();
    for v in x.iter_mut() {
        *v *= f;
    }
}

/// Per-token, per-group scalars: log-decay `g = ssm_a·softplus(a + dt_bias)` (stored
/// `ssm_a` is already `-exp(A_log)` — re-exponentiating it was the PH-1305 pin bug) and
/// write strength `beta = sigmoid(b)` (shared by the two passes for identical bits).
fn gates(w: &GdnWeights<'_>, a_row: &[f32], b_row: &[f32], h: usize) -> (f32, f32) {
    let g_log = w.a_log[h] * softplus(a_row[h] + w.dt_bias[h]);
    (g_log.exp(), 1.0 / (1.0 + (-b_row[h]).exp()))
}

/// One decode token through the block: projections → conv → per-group delta steps →
/// gated head norm → output projection. `st` is updated in place; `o_scratch` is
/// borrowed scratch of length `dv` (reused across groups).
pub fn gdn_decode_step(w: &GdnWeights<'_>, g: &GdnGeom, x: &[f32], st: &mut GdnState) -> Vec<f32> {
    debug_assert_eq!(x.len(), w.qkv.cols(), "gdn: input width");
    let mixed = w.qkv.matvec(x);
    let conv_out = conv_row(w, g, &st.conv_win, &mixed);
    push_conv_window(g, &mut st.conv_win, &mixed);
    let z = w.gate.matvec(x);
    let a = w.alpha.matvec(x);
    let b = w.beta.matvec(x);
    let mut core = vec![0.0f32; g.inner];
    let mut o = vec![0.0f32; g.dv];
    let mut qh = vec![0.0f32; g.dk];
    let mut kh = vec![0.0f32; g.dk];
    let q_scale = 1.0 / (g.dk as f32).sqrt();
    for h in 0..g.groups {
        qh.copy_from_slice(&conv_out[h * g.dk..(h + 1) * g.dk]);
        kh.copy_from_slice(&conv_out[g.inner + h * g.dk..g.inner + (h + 1) * g.dk]);
        l2norm_scale(&mut qh, q_scale, w.eps);
        l2norm_scale(&mut kh, 1.0, w.eps);
        let v = &conv_out[2 * g.inner + h * g.dv..2 * g.inner + (h + 1) * g.dv];
        let (decay, beta) = gates(w, &a, &b, h);
        let s = &mut st.s[h * g.dk * g.dv..(h + 1) * g.dk * g.dv];
        delta_step(s, &qh, &kh, v, decay, beta, &mut o, g.dk, g.dv);
        head_norm_gate(
            &o,
            w.norm_w,
            w.eps,
            &z[h * g.dv..(h + 1) * g.dv],
            g.dv,
            &mut core[h * g.dv..(h + 1) * g.dv],
        );
    }
    w.out.matvec(&core)
}

/// Head RMSNorm with gate: `rmsnorm(o) · weight · swish(z)` (see [`gate_act`]).
fn head_norm_gate(o: &[f32], weight: &[f32], eps: f32, z: &[f32], dv: usize, dst: &mut [f32]) {
    let ms = o.iter().map(|v| v * v).sum::<f32>() / (dv as f32);
    let inv = 1.0 / (ms + eps).sqrt();
    for (j, dst) in dst.iter_mut().enumerate() {
        *dst = o[j] * inv * weight[j] * gate_act(z[j]);
    }
}

/// Causal depthwise conv over the mixed qkv vector: the window (`conv-1` rows, oldest
/// first) plus the fresh row, kernel applied in chronological order, then **SiLU on
/// every channel** — the oracle activates `conv_output_silu = ggml_silu(ssm_conv(...))`
/// before the q/k/v view and L2-norm (`qwen35.cpp:397-403,432-435`); the window keeps
/// the raw pre-conv projections.
fn conv_row(w: &GdnWeights<'_>, g: &GdnGeom, win: &[f32], fresh: &[f32]) -> Vec<f32> {
    let c = g.qkv_dim();
    let k = g.conv;
    debug_assert_eq!(fresh.len(), c);
    debug_assert_eq!(win.len(), (k - 1) * c);
    let mut out = vec![0.0f32; c];
    for ch in 0..c {
        let mut acc = 0.0f32;
        for j in 0..k {
            let src = if j + 1 == k {
                fresh[ch]
            } else {
                win[j * c + ch]
            };
            acc += w.conv_w[ch * k + j] * src;
        }
        out[ch] = gate_act(acc);
    }
    out
}

/// Slide the conv window: drop the oldest row, append the fresh one.
fn push_conv_window(g: &GdnGeom, win: &mut [f32], fresh: &[f32]) {
    let c = g.qkv_dim();
    if g.conv < 2 {
        return;
    }
    win.copy_within(c.., 0);
    let last = win.len() - c..;
    win[last].copy_from_slice(fresh);
}

/// Batched prefill of a whole prompt: projections and the conv run row-tiled, then the
/// recurrence walks the tokens in order through [`delta_step`] — **bit-identical** to
/// feeding the same rows one at a time through [`gdn_decode_step`] (unit-tested).
/// Returns the `rows × d_model` block output and fills `st` with the end state.
pub fn gdn_prefill(
    w: &GdnWeights<'_>,
    g: &GdnGeom,
    x_seq: &[f32],
    rows: usize,
    st: &mut GdnState,
) -> Vec<f32> {
    debug_assert_eq!(x_seq.len(), rows * w.qkv.cols(), "gdn prefill: X rows");
    let c = g.qkv_dim();
    let mixed = w.qkv.matmul(x_seq, rows);
    // Conv over the sequence needs the window carried across rows: do it row-wise.
    let mut conv_out = vec![0.0f32; rows * c];
    for i in 0..rows {
        let row = &mixed[i * c..(i + 1) * c];
        let cr = conv_row(w, g, &st.conv_win, row);
        push_conv_window(g, &mut st.conv_win, row);
        conv_out[i * c..(i + 1) * c].copy_from_slice(&cr);
    }
    let gate = w.gate.matmul(x_seq, rows);
    let a = w.alpha.matmul(x_seq, rows);
    let b = w.beta.matmul(x_seq, rows);
    let mut core = vec![0.0f32; rows * g.inner];
    let mut o = vec![0.0f32; g.dv];
    let mut qh = vec![0.0f32; g.dk];
    let mut kh = vec![0.0f32; g.dk];
    let q_scale = 1.0 / (g.dk as f32).sqrt();
    for h in 0..g.groups {
        // One head across all rows: the state is per-group, so each group walks the
        // sequence alone (the same token order as decode ⇒ identical bits).
        let mut sg = vec![0.0f32; g.dk * g.dv];
        sg.copy_from_slice(&st.s[h * g.dk * g.dv..(h + 1) * g.dk * g.dv]);
        for i in 0..rows {
            let xrow = &conv_out[i * c..(i + 1) * c];
            qh.copy_from_slice(&xrow[h * g.dk..(h + 1) * g.dk]);
            kh.copy_from_slice(&xrow[g.inner + h * g.dk..g.inner + (h + 1) * g.dk]);
            l2norm_scale(&mut qh, q_scale, w.eps);
            l2norm_scale(&mut kh, 1.0, w.eps);
            let v = &xrow[2 * g.inner + h * g.dv..2 * g.inner + (h + 1) * g.dv];
            let (decay, beta) = gates(
                w,
                &a[i * g.groups..(i + 1) * g.groups],
                &b[i * g.groups..(i + 1) * g.groups],
                h,
            );
            delta_step(&mut sg, &qh, &kh, v, decay, beta, &mut o, g.dk, g.dv);
            head_norm_gate(
                &o,
                w.norm_w,
                w.eps,
                &gate[i * g.inner + h * g.dv..i * g.inner + (h + 1) * g.dv],
                g.dv,
                &mut core[i * g.inner + h * g.dv..i * g.inner + (h + 1) * g.dv],
            );
        }
        st.s[h * g.dk * g.dv..(h + 1) * g.dk * g.dv].copy_from_slice(&sg);
    }
    w.out.matmul(&core, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

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
    fn gdn_geom_from_qwen35_config() {
        let g = GdnGeom::from_config(&qwen35_cfg()).expect("hybrid config yields a GDN geom");
        assert_eq!(g.groups, 16);
        assert_eq!(g.dk, 128);
        assert_eq!(g.qkv_dim(), 6144, "matches the real attn_qkv rows");
        // 16 heads × 128² × f32 = exactly 1 MiB of state per recurrent block.
        assert_eq!(g.state_bytes(), 16 * 128 * 128 * 4);
        assert_eq!(g.state_bytes(), 1 << 20);
    }

    #[test]
    fn gdn_geom_absent_for_plain_arches() {
        let mut c = qwen35_cfg();
        c.ssm_inner_size = 0;
        assert_eq!(GdnGeom::from_config(&c), None);
        let mut c = qwen35_cfg();
        c.ssm_inner_size = 1024; // != groups · dk — corrupt key
        assert_eq!(GdnGeom::from_config(&c), None);
    }

    fn toy() -> (GdnGeom, usize) {
        (
            GdnGeom {
                groups: 2,
                dk: 2,
                dv: 2,
                conv: 3,
                inner: 4,
            },
            3usize, // d_model
        )
    }

    struct Toy {
        qkv: QMat,
        gate: QMat,
        alpha: QMat,
        beta: QMat,
        out: QMat,
        a_log: Vec<f32>,
        dt: Vec<f32>,
        conv_w: Vec<f32>,
        norm_w: Vec<f32>,
    }

    fn toy_weights() -> Toy {
        let (g, dm) = toy();
        let c = g.qkv_dim();
        let ramp = |m: usize, n: usize| -> Vec<f32> {
            (0..m * n)
                .map(|i| ((i as i32 * 7 % 11) as f32) * 0.05 - 0.2)
                .collect()
        };
        Toy {
            qkv: QMat::from_f32(ramp(c, dm), c, dm),
            gate: QMat::from_f32(ramp(g.inner, dm), g.inner, dm),
            alpha: QMat::from_f32(
                ramp(g.groups, dm).into_iter().map(|x| x * 0.3).collect(),
                g.groups,
                dm,
            ),
            beta: QMat::from_f32(
                ramp(g.groups, dm).into_iter().map(|x| x * -0.2).collect(),
                g.groups,
                dm,
            ),
            out: QMat::from_f32(ramp(dm, g.inner), dm, g.inner),
            a_log: vec![0.5, -0.25],
            dt: vec![0.1, -0.1],
            conv_w: (0..g.conv * c)
                .map(|i| (i as f32 * 3.7).sin() * 0.5)
                .collect(),
            norm_w: vec![1.0, 0.5, 1.25, -0.75, 0.9, 1.1, 0.8, -1.0],
        }
    }

    fn toy_views<'a>(t: &'a Toy) -> GdnWeights<'a> {
        let (g, _) = toy();
        GdnWeights {
            qkv: &t.qkv,
            gate: &t.gate,
            alpha: &t.alpha,
            beta: &t.beta,
            out: &t.out,
            a_log: &t.a_log,
            dt_bias: &t.dt,
            conv_w: &t.conv_w,
            norm_w: &t.norm_w[..g.dv],
            eps: 1e-6,
        }
    }

    #[test]
    fn delta_step_matches_the_textbook_rule() {
        // Tiny case by hand: S = [[1,2],[3,4]], decay 1, beta 1 →
        // pred = kᵀS = [3·1+4·3? k=(3,4)... ] — computed straight from the formulas.
        let (dk, dv) = (2usize, 2usize);
        let mut s = vec![1.0f32, 2.0, 3.0, 4.0]; // row-major dk × dv
        let q = [0.5f32, -0.5];
        let k = [3.0f32, 4.0];
        let v = [7.0f32, -1.0];
        let mut o = [0.0f32; 2];
        delta_step(&mut s, &q, &k, &v, 1.0, 1.0, &mut o, dk, dv);
        // pred = Σ_i k_i S[i,j] = [3·1+4·3, 3·2+4·4] = [15, 22]
        // vnew = v - pred = [-8, -23]
        // S += k ⊗ vnew = [[1+3·-8, 2+3·-23], [3+4·-8, 4+4·-23]] = [[-23,-67],[-29,-88]]
        assert_eq!(s, vec![-23.0, -67.0, -29.0, -88.0]);
        // o = Σ_i q_i S[i,:] = 0.5·[-23,-67] - 0.5·[-29,-88] = [3, 10.5]
        assert_eq!(o, [3.0, 10.5]);
    }

    #[test]
    fn prefill_is_bit_identical_to_decode_steps() {
        let (g, dm) = toy();
        let t = toy_weights();
        let w = toy_views(&t);
        let rows = 6;
        let x: Vec<f32> = (0..rows * dm)
            .map(|i| ((i as i32 * 13 % 17) as f32) * 0.04 - 0.3)
            .collect();
        let mut st_batch = GdnState::new(&g);
        let batch = gdn_prefill(&w, &g, &x, rows, &mut st_batch);
        let mut st_step = GdnState::new(&g);
        let mut steps = Vec::with_capacity(rows * dm);
        for i in 0..rows {
            steps.extend(gdn_decode_step(
                &w,
                &g,
                &x[i * dm..(i + 1) * dm],
                &mut st_step,
            ));
        }
        assert_eq!(
            batch, steps,
            "prefill must not move a bit vs step-by-step decode"
        );
        assert_eq!(st_batch.s, st_step.s, "end states must be bit-identical");
        assert_eq!(st_batch.conv_win, st_step.conv_win);
    }

    #[test]
    fn conv_is_causal_and_channel_major() {
        let (g, _dm) = toy();
        let t = toy_weights();
        let w = toy_views(&t);
        // Hand-roll the conv of a 5-row sequence of CONSTANT mixed vectors: every
        // channel collapses to w·x summed over the full window.
        let c = g.qkv_dim();
        let mixed = vec![0.2f32; c];
        let mut win = vec![0.0f32; (g.conv - 1) * c];
        let mut acc_row = vec![];
        for _ in 0..5 {
            let r = conv_row(&w, &g, &win, &mixed);
            push_conv_window(&g, &mut win, &mixed);
            acc_row = r;
        }
        for (ch, &got) in acc_row.iter().enumerate() {
            let raw: f32 = (0..g.conv).map(|j| w.conv_w[ch * g.conv + j] * 0.2).sum();
            let want = gate_act(raw);
            assert!((got - want).abs() < 1e-6, "channel {ch}");
        }
    }

    #[test]
    fn zero_input_yields_zero_block_output() {
        let (g, dm) = toy();
        let t = toy_weights();
        let w = toy_views(&t);
        let mut st = GdnState::new(&g);
        let out = gdn_decode_step(&w, &g, &vec![0.0f32; dm], &mut st);
        assert_eq!(out, vec![0.0f32; dm], "v=0 ⇒ no writes ⇒ core=0 ⇒ out=0");
    }

    // ── PH-1305 convention pins (each one is a place the oracle could have differed) ──

    /// The gated-norm activation is **SiLU**, not sigmoid: `build_norm_gated` multiplies by
    /// `ggml_silu(z)` (`qwen35.cpp:255-264`). `x=2` separates them (1.7616 vs 0.8808).
    #[test]
    fn gate_act_is_silu_not_sigmoid() {
        let silu2 = 2.0f32 * (1.0 / (1.0 + (-2.0f32).exp()));
        assert!((gate_act(2.0) - silu2).abs() < 1e-6, "silu(2) = {silu2}");
        assert!(
            (gate_act(2.0) - 1.0 / (1.0 + (-2.0f32).exp())).abs() > 0.5,
            "sigmoid was the rejected candidate"
        );
        assert_eq!(gate_act(0.0), 0.0, "silu(0) = 0");
    }

    /// `ssm_a` is stored **already negated and exponentiated** (`-exp(A_log)`, negative in
    /// the real file: `blk.0.ssm_a = -1.2890509 …`), and the oracle multiplies it straight
    /// into the softplus (`qwen35.cpp:379`). Re-applying `exp` — the bug this band fixed —
    /// would turn the decay into `exp(exp(·))`.
    #[test]
    fn ssm_a_is_used_verbatim_never_re_exponentiated() {
        // The real file's first two values (`blk.0.ssm_a`), both negative.
        let real = [-1.289_050_9f32, -0.131_171_45];
        let t = toy_weights();
        let mut w = toy_views(&t);
        w.a_log = &real;
        let (a_row, b_row) = ([0.3f32, -0.2], [0.5f32, 1.0]);
        let (decay, beta) = gates(&w, &a_row, &b_row, 0);
        let want = (real[0] * softplus(a_row[0] + w.dt_bias[0])).exp();
        assert_eq!(decay, want, "decay = exp(ssm_a · softplus(a + dt_bias))");
        assert!(
            decay < 1.0,
            "a negative ssm_a must decay the state: {decay}"
        );
        // Re-exponentiating the stored value would land somewhere else entirely.
        let re_exp = (real[0].exp() * softplus(a_row[0] + w.dt_bias[0])).exp();
        assert!(
            (decay - re_exp).abs() > 1e-2,
            "decay {decay} is too close to the re-exponentiated {re_exp}"
        );
        assert_eq!(beta, 1.0 / (1.0 + (-b_row[0]).exp()), "beta = sigmoid(b)");
    }

    /// L2 norm with **eps inside the root** and no scale of its own — `build_gdn_l2_norm`
    /// (`models.h:14-18`). A `max(norm, 1e-6)` floor (the rejected candidate) blows a tiny
    /// head up to unit length instead of shrinking it.
    #[test]
    fn l2norm_puts_eps_inside_the_root() {
        let mut tiny = [1e-4f32];
        l2norm_scale(&mut tiny, 1.0, 1e-6);
        let want = 1e-4 / (1e-8f32 + 1e-6).sqrt();
        assert!((tiny[0] - want).abs() < 1e-9, "got {} want {want}", tiny[0]);
        assert!(tiny[0] < 0.1, "a floored norm would give 100, not {want}");

        // Unit-scale head: scale folds in linearly, and `dk^-0.5` is the only extra factor.
        let mut head = [3.0f32, 4.0];
        l2norm_scale(&mut head, 0.5, 0.0);
        assert!((head[0] - 0.3).abs() < 1e-6 && (head[1] - 0.4).abs() < 1e-6);
    }

    /// The conv carries **raw** projections in its window and applies SiLU once, on the
    /// output (`conv_output_silu = ggml_silu(ssm_conv(...))`, `qwen35.cpp:397-403`).
    /// Pre-activating the window would make the second token see `silu(x)` instead of `x`.
    #[test]
    fn conv_window_holds_raw_rows_and_silu_is_applied_once() {
        let (g, _dm) = toy();
        let t = toy_weights();
        let w = toy_views(&t);
        let c = g.qkv_dim();
        let fresh = vec![0.4f32; c];
        let mut win = vec![0.0f32; (g.conv - 1) * c];
        let _ = conv_row(&w, &g, &win, &fresh);
        push_conv_window(&g, &mut win, &fresh);
        let last = &win[win.len() - c..];
        assert_eq!(last, &fresh[..], "the window keeps the pre-conv projection");
        assert!(
            (last[0] - gate_act(0.4)).abs() > 1e-3,
            "the window must not be pre-activated"
        );
    }
}
