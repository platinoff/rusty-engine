//! Transformer forward composition — band B3 `PH-305d`.
//!
//! Wires the unit-tested primitives into one decode step: `rmsnorm → multi-head GQA
//! causal attention (RoPE) → residual → rmsnorm → SwiGLU FFN → residual`, then a final
//! projection to logits. Verified end-to-end with a deterministic toy model (zeroed
//! projections reduce a block to its residual path; a crafted embed/output pair yields a
//! known greedy token). Real Qwen numeric parity vs llama.cpp is `PH-405` (needs a
//! reference + tensor-layout confirmation); this slice proves the *plumbing* composes.

use crate::infer::{gqa_repeat_kv, rmsnorm, rope_freqs, rope_in_place};
use crate::moe::{softmax, softmax_in_place};
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
    /// **Qwen2** q/k/v biases (empty for Llama-style models that carry none): `q` is
    /// `d_model`, `k`/`v` are `n_kv·head`. Adding these is mandatory for Qwen2 parity —
    /// omitting them is a systematic forward error (PH-405).
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    /// SwiGLU: gate/up are d_ff×d_model, down is d_model×d_ff.
    pub ffn_gate: QMat,
    pub ffn_up: QMat,
    pub ffn_down: QMat,
}

/// Add a per-output bias to a single vector (no-op when the bias slice is empty).
fn add_bias(mut v: Vec<f32>, b: &[f32]) -> Vec<f32> {
    for (x, &b) in v.iter_mut().zip(b) {
        *x += b;
    }
    v
}

/// Add a per-output bias to every row of a row-major `rows × b.len()` buffer (no-op when empty).
fn add_bias_rows(v: &mut [f32], b: &[f32]) {
    if b.is_empty() {
        return;
    }
    for row in v.chunks_mut(b.len()) {
        for (x, &b) in row.iter_mut().zip(b) {
            *x += b;
        }
    }
}

/// Dense SwiGLU FFN: `down(swiglu(gate(x), up(x)))`, each projection fused via [`QMat`].
fn ffn_forward(x: &[f32], gate: &QMat, up: &QMat, down: &QMat) -> Vec<f32> {
    let g = gate.matvec(x);
    let u = up.matvec(x);
    let a = crate::moe::swiglu(&g, &u);
    down.matvec(&a)
}

/// Rotate every head of a `heads × head_dim` row in place at absolute position `pos`.
fn rotate_heads(row: &mut [f32], heads: usize, head_dim: usize, pos: usize, freqs: &[f32]) {
    for h in 0..heads {
        rope_in_place(&mut row[h * head_dim..(h + 1) * head_dim], pos, freqs);
    }
}

/// Sequence length from which prefill attention is split across heads (below it the pool's
/// job hand-off costs more than the work).
const PAR_SEQ_MIN: usize = 32;

/// Causal attention for **one** head over already-rotated K/V rows: `seq × head_dim` output.
///
/// `k_rows`/`v_rows` are `pos_end × kv_dim` row-major, where `pos_end = pos_start + seq`
/// (the KV cache or a prompt-local buffer — same layout).
///
/// `out` holds the rows `round, round + rounds, round + 2·rounds, …` of the head, in that
/// order, so callers may hand interleaved row bundles of the same head to different workers.
///
/// Scores go through [`crate::kernel::dot_f32_x4_dispatch`] (one query load per four keys) and
/// the value accumulation through [`crate::kernel::axpy_f32_dispatch`], which is bit-identical
/// to the scalar `o += p * v` loop it replaces (`PH-203o`). The score change swaps the plain
/// sequential f32 sum for the 8-lane ascending accumulator, i.e. a numeric slice.
#[allow(clippy::too_many_arguments)]
fn attention_head_ctx(
    q_all: &[f32],
    k_rows: &[f32],
    v_rows: &[f32],
    g: &Geometry,
    seq: usize,
    pos_start: usize,
    h: usize,
    round: usize,
    rounds: usize,
) -> Vec<f32> {
    let hd = g.head_dim;
    let kv_dim = g.n_kv_heads * hd;
    let q_dim = g.n_heads * hd;
    let kv_h = (h * g.n_kv_heads / g.n_heads) * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let rows = seq.saturating_sub(round).div_ceil(rounds.max(1));
    let mut out = vec![0.0f32; rows * hd];
    let mut probs = vec![0.0f32; pos_start + seq];
    for (k, o) in out.chunks_exact_mut(hd).enumerate() {
        let i = round + k * rounds;
        let pos = pos_start + i;
        let qh = &q_all[i * q_dim + h * hd..i * q_dim + h * hd + hd];
        // Causal scores over 0..=pos against the already-rotated keys, four keys per query load.
        let mut j = 0;
        while j + 4 <= pos + 1 {
            let k0 = &k_rows[j * kv_dim + kv_h..j * kv_dim + kv_h + hd];
            let k1 = &k_rows[(j + 1) * kv_dim + kv_h..(j + 1) * kv_dim + kv_h + hd];
            let k2 = &k_rows[(j + 2) * kv_dim + kv_h..(j + 2) * kv_dim + kv_h + hd];
            let k3 = &k_rows[(j + 3) * kv_dim + kv_h..(j + 3) * kv_dim + kv_h + hd];
            let got = crate::kernel::dot_f32_x4_dispatch(qh, [k0, k1, k2, k3]);
            probs[j] = got[0] * scale;
            probs[j + 1] = got[1] * scale;
            probs[j + 2] = got[2] * scale;
            probs[j + 3] = got[3] * scale;
            j += 4;
        }
        while j <= pos {
            let kh = &k_rows[j * kv_dim + kv_h..j * kv_dim + kv_h + hd];
            probs[j] = crate::kernel::dot_f32_dispatch(qh, kh) * scale;
            j += 1;
        }
        softmax_in_place(&mut probs[..=pos]);
        for (j, p) in probs[..=pos].iter().enumerate() {
            let vh = &v_rows[j * kv_dim + kv_h..j * kv_dim + kv_h + hd];
            crate::kernel::axpy_f32_dispatch(o, vh, *p);
        }
    }
    out
}

/// Per-head prefill attention assembled back into a `seq × q_dim` buffer. Heads are
/// independent (each owns its own `head_dim` column block), so long prompts fan them out on
/// the shared pool; the per-element summation order is unchanged either way.
fn attention_prefill(
    q_all: &[f32],
    k_rows: &[f32],
    v_rows: &[f32],
    g: &Geometry,
    seq: usize,
    pos_start: usize,
) -> Vec<f32> {
    crate::profile::timed(crate::profile::Stage::Attention, || {
        attention_prefill_inner(q_all, k_rows, v_rows, g, seq, pos_start)
    })
}

/// Untimed body of [`attention_prefill`].
fn attention_prefill_inner(
    q_all: &[f32],
    k_rows: &[f32],
    v_rows: &[f32],
    g: &Geometry,
    seq: usize,
    pos_start: usize,
) -> Vec<f32> {
    let hd = g.head_dim;
    let q_dim = g.n_heads * hd;
    let mut ctx = vec![0.0f32; seq * q_dim];
    // Jobs are `(head, interleaved row bundle)` pairs: causal rows are independent, and the
    // cost of a row grows with its position, so rows are dealt round-robin instead of in
    // contiguous blocks (`PH-203o`). Four rounds keeps all twelve workers busy when there are
    // only `n_heads = 14` heads.
    let rounds = if seq >= PAR_SEQ_MIN && g.n_heads > 1 {
        let want = crate::pool::requested_threads() * 4;
        (want / g.n_heads).clamp(1, seq)
    } else {
        1
    };
    let jobs = g.n_heads * rounds;
    let collected = std::sync::Mutex::new(Vec::with_capacity(jobs));
    if jobs > 1 {
        crate::pool::parallel_for(jobs, |ji| {
            let (h, round) = (ji % g.n_heads, ji / g.n_heads);
            let c = attention_head_ctx(q_all, k_rows, v_rows, g, seq, pos_start, h, round, rounds);
            collected.lock().unwrap().push((h, round, c));
        });
    } else {
        for h in 0..g.n_heads {
            let c = attention_head_ctx(q_all, k_rows, v_rows, g, seq, pos_start, h, 0, 1);
            collected.lock().unwrap().push((h, 0, c));
        }
    }
    for (h, round, c) in collected.into_inner().unwrap() {
        for (k, row) in c.chunks_exact(hd).enumerate() {
            let i = round + k * rounds;
            let a = i * q_dim + h * hd;
            ctx[a..a + hd].copy_from_slice(row);
        }
    }
    ctx
}

/// Single-token causal attention (a decode step attending over a 1-position span): each
/// head's softmax over its own key is 1.0, so the context equals the (GQA-broadcast) value.
/// Q/K are still RoPE-rotated and scored, exercising the real path.
fn attention_decode(x_norm: &[f32], w: &BlockWeights, g: &Geometry, pos: usize) -> Vec<f32> {
    let freqs = rope_freqs(g.head_dim, g.rope_freq_base);
    let mut q = add_bias(w.q.matvec(x_norm), &w.q_bias);
    let mut k = add_bias(w.k.matvec(x_norm), &w.k_bias);
    let v = add_bias(w.v.matvec(x_norm), &w.v_bias);
    rotate_heads(&mut q, g.n_heads, g.head_dim, pos, &freqs);
    rotate_heads(&mut k, g.n_kv_heads, g.head_dim, pos, &freqs);
    let v_exp = gqa_repeat_kv(&v, 1, g.n_kv_heads, g.n_heads, g.head_dim);
    let scale = 1.0 / (g.head_dim as f32).sqrt();
    let mut ctx = vec![0.0f32; g.n_heads * g.head_dim];
    for h in 0..g.n_heads {
        let hs = h * g.head_dim;
        let kh_src = (h * g.n_kv_heads / g.n_heads) * g.head_dim;
        let score = q[hs..hs + g.head_dim]
            .iter()
            .zip(&k[kh_src..kh_src + g.head_dim])
            .map(|(a, b)| a * b)
            .sum::<f32>()
            * scale;
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
/// Keys are stored **already RoPE-rotated** at their absolute position, so the attention
/// loops below never re-rotate a cached key (they used to re-rotate every key once per head
/// per token, which dominated decode at long contexts).
/// This is the incremental-decoding substrate for `PH-307` (the paged version lives in
/// `crate::infer::PagedKvCache`; a dense linear buffer is enough for the shape gate).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Head-only causal attention over the KV cache (no FFN, no residual): rmsnorm(`x`) → q/k/v
/// → rotate Q/K → append rotated K/V to `cache` → head-major causal attention over `0..=pos`
/// → o-projection. Extracted so `block_forward_cached` and the MoE block (`PH-403b`) share
/// one attention core.
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
    let freqs = rope_freqs(g.head_dim, g.rope_freq_base);
    let n = rmsnorm(x, &w.attn_norm, g.rms_eps);
    let mut q = add_bias(w.q.matvec(&n), &w.q_bias);
    let mut k = add_bias(w.k.matvec(&n), &w.k_bias);
    let v = add_bias(w.v.matvec(&n), &w.v_bias);
    rotate_heads(&mut q, g.n_heads, g.head_dim, pos, &freqs);
    rotate_heads(&mut k, g.n_kv_heads, g.head_dim, pos, &freqs);
    cache.k.extend_from_slice(&k);
    cache.v.extend_from_slice(&v);
    let scale = 1.0 / (g.head_dim as f32).sqrt();
    let mut ctx = vec![0.0f32; q_dim];
    let mut probs = vec![0.0f32; pos + 1];
    for h in 0..g.n_heads {
        let hs = h * g.head_dim;
        let kv_h = (h * g.n_kv_heads / g.n_heads) * g.head_dim;
        let qh = &q[hs..hs + g.head_dim];
        for (j, p) in probs.iter_mut().enumerate() {
            let kh = &cache.k[j * kv_dim + kv_h..j * kv_dim + kv_h + g.head_dim];
            *p = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        softmax_in_place(&mut probs);
        let out = &mut ctx[hs..hs + g.head_dim];
        out.fill(0.0);
        for (j, p) in probs.iter().enumerate() {
            let vh = &cache.v[j * kv_dim + kv_h..j * kv_dim + kv_h + g.head_dim];
            for (o, vd) in out.iter_mut().zip(vh) {
                *o += *p * *vd;
            }
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
    for (i, row) in x_seq.chunks_exact(g.d_model).enumerate() {
        let n = rmsnorm(row, &w.attn_norm, g.rms_eps);
        x_norm[i * g.d_model..(i + 1) * g.d_model].copy_from_slice(&n);
    }
    // Batched projections: each weight block is dequantized once for the whole prompt
    // (row-wise `matvec` would repeat that `seq×` — the dequant-heavy part of prefill).
    let mut q_all = w.q.matmul(&x_norm, seq);
    let mut k_all = w.k.matmul(&x_norm, seq);
    let mut v_all = w.v.matmul(&x_norm, seq);
    add_bias_rows(&mut q_all, &w.q_bias);
    add_bias_rows(&mut k_all, &w.k_bias);
    add_bias_rows(&mut v_all, &w.v_bias);
    let freqs = rope_freqs(g.head_dim, g.rope_freq_base);
    for i in 0..seq {
        let pos = pos_start + i;
        rotate_heads(
            &mut q_all[i * q_dim..(i + 1) * q_dim],
            g.n_heads,
            g.head_dim,
            pos,
            &freqs,
        );
        rotate_heads(
            &mut k_all[i * kv_dim..(i + 1) * kv_dim],
            g.n_kv_heads,
            g.head_dim,
            pos,
            &freqs,
        );
    }
    let attn_ctx = attention_prefill(&q_all, &k_all, &v_all, g, seq, pos_start);
    // Output projection (batched) + residual.
    let o_all = w.o.matmul(&attn_ctx, seq);
    let mut y = vec![0.0f32; seq * g.d_model];
    for i in 0..seq * g.d_model {
        y[i] = x_seq[i] + o_all[i];
    }
    // FFN (SwiGLU), batched: one rmsnorm pass, then gate/up/down over the whole prompt.
    let mut y_norm = vec![0.0f32; seq * g.d_model];
    for (i, row) in y.chunks_exact(g.d_model).enumerate() {
        let n = rmsnorm(row, &w.ffn_norm, g.rms_eps);
        y_norm[i * g.d_model..(i + 1) * g.d_model].copy_from_slice(&n);
    }
    let gate = w.ffn_gate.matmul(&y_norm, seq);
    let up = w.ffn_up.matmul(&y_norm, seq);
    let act = crate::moe::swiglu(&gate, &up);
    let down = w.ffn_down.matmul(&act, seq);
    let mut z = vec![0.0f32; seq * g.d_model];
    for i in 0..seq * g.d_model {
        z[i] = y[i] + down[i];
    }
    z
}

/// Batched prefill that also fills the KV cache: one pass over the `seq`-row prompt with
/// the same attention as [`block_forward_seq`], but every row's K/V is appended to
/// `cache` **already RoPE-rotated**, matching [`attention_forward_cached`]'s on-read-free
/// contract. This
/// turns an O(`seq`) sequence of single-token decode steps into one batched pass, so the
/// quant dequant cost is paid once instead of `seq×`. Returns the `seq × d_model` hidden
/// state; the caller keeps the last row for logits. `cache` must already hold `pos_start`
/// positions (0 for a fresh prompt).
#[allow(clippy::needless_range_loop)]
pub fn block_forward_prefill_cached(
    x_seq: &[f32],
    w: &BlockWeights,
    g: &Geometry,
    pos_start: usize,
    cache: &mut LayerKv,
) -> Vec<f32> {
    let seq = x_seq.len() / g.d_model;
    assert_eq!(
        seq * g.d_model,
        x_seq.len(),
        "block_forward_prefill_cached: seq × d_model"
    );
    let kv_dim = g.n_kv_heads * g.head_dim;
    let q_dim = g.n_heads * g.head_dim;
    assert_eq!(
        cache.k.len() / kv_dim,
        pos_start,
        "block_forward_prefill_cached: cache/pos mismatch"
    );

    let mut x_norm = vec![0.0f32; seq * g.d_model];
    for (i, row) in x_seq.chunks_exact(g.d_model).enumerate() {
        let n = rmsnorm(row, &w.attn_norm, g.rms_eps);
        x_norm[i * g.d_model..(i + 1) * g.d_model].copy_from_slice(&n);
    }
    let mut q_all = w.q.matmul(&x_norm, seq);
    let mut k_all = w.k.matmul(&x_norm, seq);
    let mut v_all = w.v.matmul(&x_norm, seq);
    add_bias_rows(&mut q_all, &w.q_bias);
    add_bias_rows(&mut k_all, &w.k_bias);
    add_bias_rows(&mut v_all, &w.v_bias);
    let freqs = rope_freqs(g.head_dim, g.rope_freq_base);
    for i in 0..seq {
        let pos = pos_start + i;
        rotate_heads(
            &mut q_all[i * q_dim..(i + 1) * q_dim],
            g.n_heads,
            g.head_dim,
            pos,
            &freqs,
        );
        rotate_heads(
            &mut k_all[i * kv_dim..(i + 1) * kv_dim],
            g.n_kv_heads,
            g.head_dim,
            pos,
            &freqs,
        );
    }
    cache.k.extend_from_slice(&k_all);
    cache.v.extend_from_slice(&v_all);

    let attn_ctx = attention_prefill(&q_all, &cache.k, &cache.v, g, seq, pos_start);
    let o_all = w.o.matmul(&attn_ctx, seq);
    let mut y = vec![0.0f32; seq * g.d_model];
    for i in 0..seq * g.d_model {
        y[i] = x_seq[i] + o_all[i];
    }
    let mut y_norm = vec![0.0f32; seq * g.d_model];
    for (i, row) in y.chunks_exact(g.d_model).enumerate() {
        let n = rmsnorm(row, &w.ffn_norm, g.rms_eps);
        y_norm[i * g.d_model..(i + 1) * g.d_model].copy_from_slice(&n);
    }
    let gate = w.ffn_gate.matmul(&y_norm, seq);
    let up = w.ffn_up.matmul(&y_norm, seq);
    let act = crate::moe::swiglu(&gate, &up);
    let down = w.ffn_down.matmul(&act, seq);
    let mut z = vec![0.0f32; seq * g.d_model];
    for i in 0..seq * g.d_model {
        z[i] = y[i] + down[i];
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
            q_bias: Vec::new(),
            k_bias: Vec::new(),
            v_bias: Vec::new(),
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
            q_bias: Vec::new(),
            k_bias: Vec::new(),
            v_bias: Vec::new(),
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
            q_bias: Vec::new(),
            k_bias: Vec::new(),
            v_bias: Vec::new(),
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

    #[test]
    fn prefill_attention_row_rounds_are_bit_identical() {
        // PH-203o: splitting a head's rows round-robin across jobs only regroups independent
        // rows, so the assembled context must not move a single bit.
        let g = Geometry::new(8, 4, 2, 4, 10_000.0, 1e-5);
        let seq = 23usize;
        let pos_start = 5usize;
        let (q_dim, kv_dim) = (g.n_heads * g.head_dim, g.n_kv_heads * g.head_dim);
        let q: Vec<f32> = (0..seq * q_dim)
            .map(|i| ((i as i32 * 7 % 19) as f32) * 0.03 - 0.25)
            .collect();
        let k: Vec<f32> = (0..(pos_start + seq) * kv_dim)
            .map(|i| ((i as i32 * 11 % 23) as f32) * 0.02 - 0.2)
            .collect();
        let v: Vec<f32> = (0..(pos_start + seq) * kv_dim)
            .map(|i| ((i as i32 * 5 % 17) as f32) * 0.04 - 0.3)
            .collect();
        let whole = attention_head_ctx(&q, &k, &v, &g, seq, pos_start, 1, 0, 1);
        let mut stitched = vec![f32::NAN; seq * g.head_dim];
        for (k, row) in whole.chunks_exact(g.head_dim).enumerate() {
            stitched[k * g.head_dim..(k + 1) * g.head_dim].copy_from_slice(row);
        }
        for rounds in [3usize, 7] {
            let mut got = vec![f32::NAN; seq * g.head_dim];
            for round in 0..rounds {
                let part = attention_head_ctx(&q, &k, &v, &g, seq, pos_start, 1, round, rounds);
                for (n, row) in part.chunks_exact(g.head_dim).enumerate() {
                    let i = round + n * rounds;
                    got[i * g.head_dim..(i + 1) * g.head_dim].copy_from_slice(row);
                }
            }
            assert_eq!(got, stitched, "rounds {rounds} changed the context");
        }
    }
}
