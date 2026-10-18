//! `PH-205` — scalar reference vs runtime-dispatched SIMD micro-benchmark.
//!
//! `cargo bench --bench quant` (release). For each quant kernel it prints the scalar
//! reference cost, the cost of the `*_dispatch` path (AVX2 → SSE2 → scalar ladder,
//! `is_x86_feature_detected!`), the speedup, and the absolute difference between the two
//! results (a sanity check — the scalar kernel is the correctness oracle the SIMD path
//! must match). Nothing here is compared to llama-rs; this isolates the SIMD band payoff.
//!
//! Deliberately **std-only** (no `criterion`): the crate stays offline-green, and a
//! deterministic xorshift feeds the buffers so runs are reproducible.
#![allow(clippy::too_many_arguments)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use rusty_engine::kernel as k;

/// Deterministic xorshift64 byte filler (avoids a `rand` dependency).
fn fill_bytes(seed: u64, out: &mut [u8]) {
    let mut s = seed | 1;
    for b in out.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = (s & 0xFF) as u8;
    }
}

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

/// `n` blocks of `block_bytes` filled deterministically, with a finite f16 scale stamped at
/// `d_off` (and, when set, a finite `min` at `min_off`) so the scalar and SIMD results are
/// comparable instead of propagating f16 NaN/Inf from random scale bytes. Q4_0/Q8_0/Q5_0 and
/// Q4_K carry `d` at offset 0; Q6_K carries it at the block tail (`block_bytes - 2`).
fn blocks(
    seed: u64,
    n: usize,
    block_bytes: usize,
    d_off: usize,
    min_off: Option<usize>,
) -> Vec<u8> {
    const HALF: [u8; 2] = [0x00, 0x38]; // f16 0.5, little-endian
    let mut v = vec![0u8; n * block_bytes];
    fill_bytes(seed, &mut v);
    for blk in v.chunks_exact_mut(block_bytes) {
        blk[d_off..d_off + 2].copy_from_slice(&HALF);
        if let Some(m) = min_off {
            blk[m..m + 2].copy_from_slice(&HALF);
        }
    }
    v
}

fn time<F: FnMut()>(iters: u32, mut f: F) -> Duration {
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed()
}

fn bench_dot(
    name: &str,
    a: &[u8],
    b: &[f32],
    scalar: fn(&[u8], &[f32]) -> f32,
    dispatch: fn(&[u8], &[f32]) -> f32,
    iters: u32,
) {
    let s_val = scalar(a, b);
    let d_val = dispatch(a, b);
    let s_ns = time(iters, || {
        black_box(scalar(a, b));
    })
    .as_nanos() as f64
        / iters as f64;
    let d_ns = time(iters, || {
        black_box(dispatch(a, b));
    })
    .as_nanos() as f64
        / iters as f64;
    println!(
        "{name:<16} scalar {s_ns:>10.1} ns | dispatch {d_ns:>10.1} ns | speedup {:>5.2}x | |Δ|={:.2e}",
        s_ns / d_ns,
        (s_val - d_val).abs()
    );
}

fn bench_matvec(
    name: &str,
    w: &[u8],
    m: usize,
    n: usize,
    x: &[f32],
    scalar: fn(&[u8], usize, usize, &[f32]) -> Vec<f32>,
    dispatch: fn(&[u8], usize, usize, &[f32]) -> Vec<f32>,
    iters: u32,
) {
    let s_val = scalar(w, m, n, x);
    let d_val = dispatch(w, m, n, x);
    let s_ns = time(iters, || {
        black_box(scalar(w, m, n, x));
    })
    .as_nanos() as f64
        / iters as f64;
    let d_ns = time(iters, || {
        black_box(dispatch(w, m, n, x));
    })
    .as_nanos() as f64
        / iters as f64;
    let max_abs = s_val
        .iter()
        .zip(&d_val)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "{name:<16} scalar {s_ns:>10.1} ns | dispatch {d_ns:>10.1} ns | speedup {:>5.2}x | max|Δ|={max_abs:.2e}",
        s_ns / d_ns
    );
}

fn main() {
    // ~8k-element vectors (256 blocks for the 32-wide formats, 32 super-blocks for K-quant).
    let n_blocks = 256usize;
    let iters = 4_000u32;

    println!("== dots (scalar reference vs *_dispatch) ==");
    let b32 = fill_floats(2, n_blocks * 32);

    let a4 = blocks(1, n_blocks, 18, 0, None);
    bench_dot(
        "dot_q4_0",
        &a4,
        &b32,
        k::dot_q4_0,
        k::dot_q4_0_dispatch,
        iters,
    );

    let a8 = blocks(3, n_blocks, 34, 0, None);
    bench_dot(
        "dot_q8_0",
        &a8,
        &b32,
        k::dot_q8_0,
        k::dot_q8_0_dispatch,
        iters,
    );

    let a5 = blocks(5, n_blocks, 22, 0, None);
    bench_dot(
        "dot_q5_0",
        &a5,
        &b32,
        k::dot_q5_0,
        k::dot_q5_0_dispatch,
        iters,
    );

    let super_blocks = 32usize; // 256-value super-blocks
    let b256 = fill_floats(7, super_blocks * 256);
    let a4k = blocks(9, super_blocks, 144, 0, Some(2));
    bench_dot(
        "dot_q4_k",
        &a4k,
        &b256,
        k::dot_q4_k,
        k::dot_q4_k_dispatch,
        iters,
    );

    let a6k = blocks(11, super_blocks, 210, 208, None);
    bench_dot(
        "dot_q6_k",
        &a6k,
        &b256,
        k::dot_q6_k,
        k::dot_q6_k_dispatch,
        iters,
    );

    println!("\n== matvec (Qwen-shaped 896x896 row-major) ==");
    let m = 896usize;
    let n = 896usize;
    let w4 = blocks(13, m * (n / 32), 18, 0, None);
    let x = fill_floats(15, n);
    bench_matvec(
        "matvec_q4_0",
        &w4,
        m,
        n,
        &x,
        k::matvec_q4_0,
        k::matvec_q4_0_dispatch,
        200,
    );

    let w8 = blocks(17, m * (n / 32), 34, 0, None);
    bench_matvec(
        "matvec_q8_0",
        &w8,
        m,
        n,
        &x,
        k::matvec_q8_0,
        k::matvec_q8_0_dispatch,
        200,
    );
}
