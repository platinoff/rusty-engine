//! `PH-203s` — prefill attention softmax A/B, **measured and rejected**: the blocked online
//! softmax with the `exp2` table (and the table alone) is consistently 8–15 % *slower* than
//! the three-pass libm shape kept in production, and skipping dead keys (`p < 1e-8`) never
//! fires while `1e-4` saves 8 % of the axpy calls for nothing. The stage is score-dot bound,
//! not exp bound — `f32::exp` on this box's libm is faster than the table chain. Live `:8090`
//! walls drift ±25 %, so the verdict is min-of-5 here, in one process: one head's full causal
//! workload (961 rows, hd 64, `kv_dim` 128 — Qwen2.5-0.5B geometry), with the stage estimate
//! scaled by the 336 head-passes a 961-token prefill runs (24 layers × 14 heads).
//!
//! std-only, deterministic xorshift fills, same discipline as `benches/quant.rs`.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use rusty_engine::kernel as k;
use rusty_engine::moe::softmax_in_place;

/// head_dim and the packed K/V row width for Qwen2.5-0.5B (2 KV heads × 64).
const HD: usize = 64;
const KV_DIM: usize = 128;
/// Prompt rows in the 961-token probe.
const SEQ: usize = 961;
/// KV keys per online-softmax block (the rejected candidate's blocking, kept for the A/B).
const KV_BLOCK: usize = 128;
/// Head-passes per prefill: 24 layers × 14 heads.
const HEAD_PASSES: f64 = 336.0;

/// Deterministic floats in `[-0.5, 0.5)`.
fn fill_floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 16_777_216.0) - 0.5
        })
        .collect()
}

fn time(iters: u32, mut f: impl FnMut()) -> Duration {
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed()
}

/// The `PH-203o` kernel: scores into one big row buffer, three-pass libm softmax, axpy loop.
fn head_ctx_reference(q: &[f32], k_rows: &[f32], v_rows: &[f32]) -> Vec<f32> {
    let scale = 1.0 / (HD as f32).sqrt();
    let mut out = vec![0.0f32; SEQ * HD];
    let mut probs = vec![0.0f32; SEQ];
    for i in 0..SEQ {
        let pos = i;
        let o = &mut out[i * HD..i * HD + HD];
        let qh = &q[i * HD..i * HD + HD];
        let mut j = 0;
        while j + 4 <= pos + 1 {
            let k0 = &k_rows[j * KV_DIM..j * KV_DIM + HD];
            let k1 = &k_rows[(j + 1) * KV_DIM..(j + 1) * KV_DIM + HD];
            let k2 = &k_rows[(j + 2) * KV_DIM..(j + 2) * KV_DIM + HD];
            let k3 = &k_rows[(j + 3) * KV_DIM..(j + 3) * KV_DIM + HD];
            let got = k::dot_f32_x4_dispatch(qh, [k0, k1, k2, k3]);
            probs[j] = got[0] * scale;
            probs[j + 1] = got[1] * scale;
            probs[j + 2] = got[2] * scale;
            probs[j + 3] = got[3] * scale;
            j += 4;
        }
        while j <= pos {
            let kh = &k_rows[j * KV_DIM..j * KV_DIM + HD];
            probs[j] = k::dot_f32_dispatch(qh, kh) * scale;
            j += 1;
        }
        softmax_in_place(&mut probs[..=pos]);
        for (j, &p) in probs[..=pos].iter().enumerate() {
            let vh = &v_rows[j * KV_DIM..j * KV_DIM + HD];
            k::axpy_f32_dispatch(o, vh, p);
        }
    }
    out
}

/// The `PH-203s` candidate (rejected): blocked online softmax (running max + denominator with
/// accumulator rescale) over 128-key blocks, weights from the `exp2` table.
fn head_ctx_online(q: &[f32], k_rows: &[f32], v_rows: &[f32]) -> Vec<f32> {
    let scale = 1.0 / (HD as f32).sqrt();
    let mut out = vec![0.0f32; SEQ * HD];
    let mut scores = vec![0.0f32; KV_BLOCK];
    let tab = k::exp2_table_ref();
    for i in 0..SEQ {
        let span = i + 1;
        let o = &mut out[i * HD..i * HD + HD];
        let qh = &q[i * HD..i * HD + HD];
        let mut run_max = f32::NEG_INFINITY;
        let mut den = 0.0f32;
        let mut j0 = 0;
        while j0 < span {
            let n = (span - j0).min(KV_BLOCK);
            let mut j = 0;
            while j + 4 <= n {
                let b = j0 + j;
                let k0 = &k_rows[b * KV_DIM..b * KV_DIM + HD];
                let k1 = &k_rows[(b + 1) * KV_DIM..(b + 1) * KV_DIM + HD];
                let k2 = &k_rows[(b + 2) * KV_DIM..(b + 2) * KV_DIM + HD];
                let k3 = &k_rows[(b + 3) * KV_DIM..(b + 3) * KV_DIM + HD];
                let got = k::dot_f32_x4_dispatch(qh, [k0, k1, k2, k3]);
                scores[j] = got[0] * scale;
                scores[j + 1] = got[1] * scale;
                scores[j + 2] = got[2] * scale;
                scores[j + 3] = got[3] * scale;
                j += 4;
            }
            while j < n {
                let kh = &k_rows[(j0 + j) * KV_DIM..(j0 + j) * KV_DIM + HD];
                scores[j] = k::dot_f32_dispatch(qh, kh) * scale;
                j += 1;
            }
            let bmax = scores[..n]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let m_new = run_max.max(bmax);
            let rescale = k::exp_neg_fast(run_max - m_new, tab);
            if rescale != 1.0 {
                for v in o.iter_mut() {
                    *v *= rescale;
                }
                den *= rescale;
            }
            let mut bsum = 0.0f32;
            for s in scores[..n].iter_mut() {
                let p = k::exp_neg_fast(*s - m_new, tab);
                *s = p;
                bsum += p;
            }
            den += bsum;
            for (jj, &p) in scores[..n].iter().enumerate() {
                if p != 0.0 {
                    let vh = &v_rows[(j0 + jj) * KV_DIM..(j0 + jj) * KV_DIM + HD];
                    k::axpy_f32_dispatch(o, vh, p);
                }
            }
            run_max = m_new;
            j0 += n;
        }
        let inv = 1.0 / den;
        for v in o.iter_mut() {
            *v *= inv;
        }
    }
    out
}

/// The control: the old three-pass structure with the table exp swapped in for libm — tells
/// the online blocking cost apart from the exp-cost change.
fn head_ctx_table_three_pass(q: &[f32], k_rows: &[f32], v_rows: &[f32]) -> Vec<f32> {
    let scale = 1.0 / (HD as f32).sqrt();
    let mut out = vec![0.0f32; SEQ * HD];
    let mut probs = vec![0.0f32; SEQ];
    let tab = k::exp2_table_ref();
    for i in 0..SEQ {
        let pos = i;
        let o = &mut out[i * HD..i * HD + HD];
        let qh = &q[i * HD..i * HD + HD];
        let mut j = 0;
        while j + 4 <= pos + 1 {
            let k0 = &k_rows[j * KV_DIM..j * KV_DIM + HD];
            let k1 = &k_rows[(j + 1) * KV_DIM..(j + 1) * KV_DIM + HD];
            let k2 = &k_rows[(j + 2) * KV_DIM..(j + 2) * KV_DIM + HD];
            let k3 = &k_rows[(j + 3) * KV_DIM..(j + 3) * KV_DIM + HD];
            let got = k::dot_f32_x4_dispatch(qh, [k0, k1, k2, k3]);
            probs[j] = got[0] * scale;
            probs[j + 1] = got[1] * scale;
            probs[j + 2] = got[2] * scale;
            probs[j + 3] = got[3] * scale;
            j += 4;
        }
        while j <= pos {
            let kh = &k_rows[j * KV_DIM..j * KV_DIM + HD];
            probs[j] = k::dot_f32_dispatch(qh, kh) * scale;
            j += 1;
        }
        let row = &mut probs[..=pos];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = k::exp_neg_fast(*v - max, tab);
            *v = e;
            sum += e;
        }
        if sum != 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        for (j, &p) in probs[..=pos].iter().enumerate() {
            let vh = &v_rows[j * KV_DIM..j * KV_DIM + HD];
            k::axpy_f32_dispatch(o, vh, p);
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Three-pass libm softmax, but the value pass skips keys with `p < thr` (worst dropped
/// mass `SEQ · thr`). Also reports the fraction of keys that still ran an axpy.
fn head_ctx_ref_skip(q: &[f32], k_rows: &[f32], v_rows: &[f32], thr: f32) -> (Vec<f32>, f64) {
    let scale = 1.0 / (HD as f32).sqrt();
    let mut out = vec![0.0f32; SEQ * HD];
    let mut probs = vec![0.0f32; SEQ];
    let mut kept = 0usize;
    let mut total = 0usize;
    for i in 0..SEQ {
        let pos = i;
        let o = &mut out[i * HD..i * HD + HD];
        let qh = &q[i * HD..i * HD + HD];
        let mut j = 0;
        while j + 4 <= pos + 1 {
            let k0 = &k_rows[j * KV_DIM..j * KV_DIM + HD];
            let k1 = &k_rows[(j + 1) * KV_DIM..(j + 1) * KV_DIM + HD];
            let k2 = &k_rows[(j + 2) * KV_DIM..(j + 2) * KV_DIM + HD];
            let k3 = &k_rows[(j + 3) * KV_DIM..(j + 3) * KV_DIM + HD];
            let got = k::dot_f32_x4_dispatch(qh, [k0, k1, k2, k3]);
            probs[j] = got[0] * scale;
            probs[j + 1] = got[1] * scale;
            probs[j + 2] = got[2] * scale;
            probs[j + 3] = got[3] * scale;
            j += 4;
        }
        while j <= pos {
            let kh = &k_rows[j * KV_DIM..j * KV_DIM + HD];
            probs[j] = k::dot_f32_dispatch(qh, kh) * scale;
            j += 1;
        }
        softmax_in_place(&mut probs[..=pos]);
        for (j, &p) in probs[..=pos].iter().enumerate() {
            total += 1;
            if p >= thr {
                kept += 1;
                let vh = &v_rows[j * KV_DIM..j * KV_DIM + HD];
                k::axpy_f32_dispatch(o, vh, p);
            }
        }
    }
    (out, kept as f64 / total as f64)
}

fn main() {
    // q rows carry head_dim floats each (one head only); K/V rows are KV_DIM-wide.
    // Q/K at ±2 gives score std ≈ 2.7 after the 1/√hd scale — peaked enough that
    // `p < 1e-8` keys actually exist; flat random data would overstate every variant.
    let wide = |seed: u64, n: usize| -> Vec<f32> {
        fill_floats(seed, n).iter().map(|x| x * 4.0).collect()
    };
    let q = wide(7, SEQ * HD);
    let k_rows = wide(11, SEQ * KV_DIM);
    let v_rows = fill_floats(13, SEQ * KV_DIM);

    let variants: [(&str, fn(&[f32], &[f32], &[f32]) -> Vec<f32>); 3] = [
        (
            "ph203o reference (3-pass libm exp, PROD)",
            head_ctx_reference,
        ),
        ("ph203s online (128-key blocks, exp table)", head_ctx_online),
        ("3-pass, exp table only", head_ctx_table_three_pass),
    ];

    let reference = variants[0].1(&q, &k_rows, &v_rows);
    for (name, f) in variants {
        let once = f(&q, &k_rows, &v_rows);
        black_box(&once);
        let mut best = Duration::MAX;
        for _ in 0..5 {
            let t = time(3, || {
                black_box(f(&q, &k_rows, &v_rows));
            });
            best = best.min(t / 3);
        }
        let per_head = best.as_secs_f64() * 1e3;
        let stage = per_head * HEAD_PASSES;
        let drift = max_abs_diff(&once, &reference);
        println!(
            "{name:46} head {per_head:6.2} ms · stage x336 {stage:6.1} ms · max|Δ| vs ref {drift:.2e}"
        );
    }

    for thr in [1e-8_f32, 1e-6, 1e-4] {
        let mut best = Duration::MAX;
        for _ in 0..5 {
            let t = time(3, || {
                black_box(head_ctx_ref_skip(&q, &k_rows, &v_rows, thr).0);
            });
            best = best.min(t / 3);
        }
        let (once, kept) = head_ctx_ref_skip(&q, &k_rows, &v_rows, thr);
        let per_head = best.as_secs_f64() * 1e3;
        let stage = per_head * HEAD_PASSES;
        let drift = max_abs_diff(&once, &reference);
        let label = format!("3-pass libm + skip p<{thr:.0e}");
        println!(
            "{label:46} head {per_head:6.2} ms · stage x336 {stage:6.1} ms · kept {:.1} % · max|Δ| vs ref {drift:.2e}",
            kept * 100.0
        );
    }
}
