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

    // `PH-203i`: integer dot (i8 weights × i8 activations) vs the float dequant-in-register
    // path. Scalar and dispatch must agree exactly (same per-block `i32` sum, one scale step).
    println!("\n== q8_0 integer dot (i8 activations) ==");
    let xq8 = k::quantize_q8_0(&b32);
    let s_int = k::dot_q8_0_i8(&a8, &xq8);
    let d_int = k::dot_q8_0_i8_dispatch(&a8, &xq8);
    let s_ns = time(iters, || {
        black_box(k::dot_q8_0_i8(&a8, &xq8));
    })
    .as_nanos() as f64
        / iters as f64;
    let d_ns = time(iters, || {
        black_box(k::dot_q8_0_i8_dispatch(&a8, &xq8));
    })
    .as_nanos() as f64
        / iters as f64;
    println!(
        "{:<16} scalar {s_ns:>10.1} ns | dispatch {d_ns:>10.1} ns | speedup {:>5.2}x | |Δ|={:.2e}",
        "dot_q8_0_i8",
        s_ns / d_ns,
        (s_int - d_int).abs()
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

    // `PH-203i`: serial 896² matvec, float fused dot vs the i8 dot (activation quantized
    // once per call, exactly like `QMat::matvec`). GMAC/s = m·n / ns.
    let xq = k::quantize_q8_0(&x);
    let float_ns = time(200, || {
        black_box(k::matvec_q8_0_dispatch(&w8, m, n, &x));
    })
    .as_nanos() as f64
        / 200.0;
    let int_ns = time(200, || {
        black_box(k::matvec_q8_0_i8(&w8, m, n, &xq));
    })
    .as_nanos() as f64
        / 200.0;
    let macs = (m * n) as f64;
    println!(
        "{:<16} float {float_ns:>10.1} ns ({:.2} GMAC/s) | i8 {int_ns:>10.1} ns ({:.2} GMAC/s) | speedup {:>5.2}x",
        "matvec_q8_0_i8",
        macs / float_ns,
        macs / int_ns,
        float_ns / int_ns
    );

    // `PH-203e` follow-up: is a persistent pool worth it? Compare one decode-shaped
    // projection (gate/up, 4864x896 Q5_0) on the serial kernel, the row-parallel `QMat`
    // path, and the cost of the `thread::scope` spawn+join that path pays per call.
    println!("\n== parallel projection vs spawn overhead ==");
    let spawn_iters = 200u32;
    let t = time(spawn_iters, || {
        std::thread::scope(|s| {
            for _ in 0..12 {
                s.spawn(|| black_box(0u8));
            }
        });
    });
    let spawn_us = t.as_secs_f64() * 1e6 / spawn_iters as f64;
    println!("scope spawn+join (12 th)  {spawn_us:>10.1} us");

    let gm = 4864usize;
    let wg = blocks(19, gm * (n / 32), 22, 0, None);
    let serial = time(20, || {
        black_box(k::matvec_q5_0_dispatch(&wg, gm, n, &x));
    })
    .as_secs_f64()
        * 1e3
        / 20.0;
    let qm = rusty_engine::qmat::QMat::from_bytes(6, wg.clone(), gm, n);
    let par = time(20, || {
        black_box(qm.matvec(&x));
    })
    .as_secs_f64()
        * 1e3
        / 20.0;
    println!(
        "gate/up 4864x896 Q5_0    serial {serial:>8.3} ms | QMat {par:>8.3} ms | speedup {:>5.2}x",
        serial / par
    );
    println!(
        "per-call scope would add {:.1} ms/token (~120 projections); the pool pays the spawn once",
        spawn_us / 1000.0 * 120.0
    );

    // Prefill shape: the same weight applied to 961 prompt rows.
    let rows = 961usize;
    let xb = fill_floats(21, rows * n);
    let t = time(1, || {
        black_box(k::matmul_q5_0_dispatch(&wg, gm, n, &xb, rows));
    });
    let serial_mm = t.as_secs_f64() * 1e3;
    let t = time(3, || {
        black_box(qm.matmul(&xb, rows));
    });
    let par_mm = t.as_secs_f64() * 1e3 / 3.0;
    println!("prefill matmul 4864x896 x{rows}  serial {serial_mm:>8.1} ms | QMat {par_mm:>8.1} ms | speedup {:>5.2}x", serial_mm / par_mm);

    // Interleaved A/B, min of 3 rounds: this box drifts ~2x between runs and the first
    // measurement after a cache-cold start is biased, so only repeats inside one process
    // are comparable. Rows = the decomposition `QMat::matmul` now uses, cols = the old one.
    let (mut row_best, mut col_best) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..7 {
        let t = time(1, || {
            black_box(qm.matmul(&xb, rows));
        });
        row_best = row_best.min(t.as_secs_f64() * 1e3);
        let t = time(1, || {
            black_box(qm.matmul_split_cols(&xb, rows));
        });
        col_best = col_best.min(t.as_secs_f64() * 1e3);
    }
    println!(
        "prefill split A/B (min of 3):  rows {row_best:>7.1} ms | cols {col_best:>7.1} ms | rows/cols {:>5.2}x",
        col_best / row_best
    );

    // Activation-tile sweep, same interleaved min-of-3 discipline.
    let tiles = [961usize, 128, 64, 32, 16];
    let mut best = vec![f64::INFINITY; tiles.len()];
    for _ in 0..7 {
        for (i, &tile) in tiles.iter().enumerate() {
            let t = time(1, || {
                black_box(k::matmul_q5_0_tiled(&wg, gm, n, &xb, rows, tile));
            });
            best[i] = best[i].min(t.as_secs_f64() * 1e3);
        }
    }
    println!("prefill matmul tile sweep (min of 3, single thread, Q5_0 4864x896 x961):");
    for (i, &tile) in tiles.iter().enumerate() {
        let ms = best[i];
        println!(
            "  tile {tile:>4}  {ms:>8.1} ms  {:>6.2} GMAC/s",
            (rows as f64 * gm as f64 * n as f64 / 1e9) / (ms / 1e3)
        );
    }

    // PH-203l numeric slice: full-row f32 accumulator (one hsum per row) vs the per-block
    // accumulator the recorded goldens came from. Same rounding class, different order.
    let (mut row_best, mut block_best) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..7 {
        let t = time(1, || {
            black_box(k::matmul_q5_0_tiled(&wg, gm, n, &xb, rows, 16));
        });
        row_best = row_best.min(t.as_secs_f64() * 1e3);
        let t = time(1, || {
            black_box(k::matmul_q5_0_blockwise(&wg, gm, n, &xb, rows, 16));
        });
        block_best = block_best.min(t.as_secs_f64() * 1e3);
    }
    println!(
        "prefill accumulator A/B (min of 7, single thread):  row {row_best:>7.1} ms | block {block_best:>7.1} ms | row/block {:>5.2}x",
        block_best / row_best
    );

    // PH-203n: register blocking. `rows1` = one activation per weight load, `x4` = four.
    // Measured on both the narrow (n = 896, gate/up) and the wide (n = 4864, down) shape,
    // because the profiler showed the wide one at 10 GMAC/s against 35 for the rest.
    let shapes = [("gate/up 4864x896", gm, n), ("down    896x4864", n, gm)];
    for (label, mm, nn) in shapes {
        let w_s: Vec<u8> = (0..mm * nn / 32 * 22).map(|i| (i % 251) as u8).collect();
        let x_s = fill_floats(31, rows * nn);
        let mac = rows as f64 * mm as f64 * nn as f64 / 1e9;

        // Through the real 12-thread pool: row split vs output-row split. The row split only
        // wins while the activation matrix as a whole stays in L3.
        let qs = rusty_engine::qmat::QMat::from_bytes(6, w_s.clone(), mm, nn);
        let (mut rsplit, mut csplit) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..5 {
            let t = time(1, || {
                black_box(qs.matmul(&x_s, rows));
            });
            rsplit = rsplit.min(t.as_secs_f64() * 1e3);
            let t = time(1, || {
                black_box(qs.matmul_split_cols(&x_s, rows));
            });
            csplit = csplit.min(t.as_secs_f64() * 1e3);
        }
        println!(
            "pool split A/B {label} (12 threads, min of 5):  rows {rsplit:>7.1} ms ({:>5.2} GMAC/s) | out-rows {csplit:>7.1} ms ({:>5.2} GMAC/s) | {:>5.2}x",
            mac / (rsplit / 1e3),
            mac / (csplit / 1e3),
            rsplit / csplit
        );
        let (mut g1, mut g4) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..7 {
            let t = time(1, || {
                black_box(k::matmul_q5_0_rows1(&w_s, mm, nn, &x_s, rows, 16));
            });
            g1 = g1.min(t.as_secs_f64() * 1e3);
            let t = time(1, || {
                black_box(k::matmul_q5_0_tiled(&w_s, mm, nn, &x_s, rows, 16));
            });
            g4 = g4.min(t.as_secs_f64() * 1e3);
        }
        println!(
            "prefill row blocking {label} (min of 7, single thread):  1 row {g1:>7.1} ms ({:>5.2} GMAC/s) | 4 rows {g4:>7.1} ms ({:>5.2} GMAC/s) | {:>5.2}x",
            mac / (g1 / 1e3),
            mac / (g4 / 1e3),
            g1 / g4
        );
        // PH-203q: the accumulator is one dependent FMA chain per output, so sweep the block
        // width until the 16 AVX2 registers stop holding the accumulators.
        let groups = [1usize, 2, 4, 6, 8];
        let mut bg = vec![f64::INFINITY; groups.len()];
        for _ in 0..5 {
            for (i, &group) in groups.iter().enumerate() {
                let t = time(1, || {
                    black_box(k::matmul_q5_0_grouped(&w_s, mm, nn, &x_s, rows, group));
                });
                bg[i] = bg[i].min(t.as_secs_f64() * 1e3);
            }
        }
        println!("  register-block sweep on {label} (min of 5, single thread):");
        for (i, &group) in groups.iter().enumerate() {
            println!(
                "    group {group:>2}  {:>8.1} ms  {:>6.2} GMAC/s",
                bg[i],
                mac / (bg[i] / 1e3)
            );
        }
        // Tile size now only buys down the dequant pass count (one dequant per weight row per
        // tile), so sweep upwards on both shapes.
        let tiles = [16usize, 24, 32, 48, 64, 96];
        let mut best = vec![f64::INFINITY; tiles.len()];
        for _ in 0..5 {
            for (i, &tile) in tiles.iter().enumerate() {
                let t = time(1, || {
                    black_box(k::matmul_q5_0_tiled(&w_s, mm, nn, &x_s, rows, tile));
                });
                best[i] = best[i].min(t.as_secs_f64() * 1e3);
            }
        }
        println!("  tile sweep on {label} (min of 5, single thread):");
        for (i, &tile) in tiles.iter().enumerate() {
            println!(
                "    tile {tile:>3}  {:>8.1} ms  {:>6.2} GMAC/s",
                best[i],
                mac / (best[i] / 1e3)
            );
        }
        // Weight-row blocking: `rb1` re-reads the whole X for every weight row.
        let (mut rb1, mut rbn) = (f64::INFINITY, f64::INFINITY);
        let tile = 96usize;
        for _ in 0..5 {
            let t = time(1, || {
                black_box(k::matmul_q5_0_rb1(&w_s, mm, nn, &x_s, rows, tile));
            });
            rb1 = rb1.min(t.as_secs_f64() * 1e3);
            let t = time(1, || {
                black_box(k::matmul_q5_0_tiled(&w_s, mm, nn, &x_s, rows, tile));
            });
            rbn = rbn.min(t.as_secs_f64() * 1e3);
        }
        println!(
            "  weight-row blocking on {label} (min of 5, single thread, tile {tile}):  1 row {rb1:>7.1} ms ({:>5.2} GMAC/s) | 64 rows {rbn:>7.1} ms ({:>5.2} GMAC/s) | {:>5.2}x",
            mac / (rb1 / 1e3),
            mac / (rbn / 1e3),
            rb1 / rbn
        );
        // Two-dimensional sweep: tile fixes the dequant pass count (MAC/tile) and the weight
        // block fixes how often the activation tile is re-streamed (m/rb passes).
        let combos = [
            (96usize, 64usize),
            (96, 256),
            (96, 1024),
            (128, 64),
            (128, 256),
            (128, 1024),
            (192, 1024),
            (256, 1024),
        ];
        let mut bc = vec![f64::INFINITY; combos.len()];
        for _ in 0..5 {
            for (i, &(tile, rb)) in combos.iter().enumerate() {
                let t = time(1, || {
                    black_box(k::matmul_q5_0_blocked(&w_s, mm, nn, &x_s, rows, tile, rb));
                });
                bc[i] = bc[i].min(t.as_secs_f64() * 1e3);
            }
        }
        println!("  2-D sweep on {label} (min of 5, single thread):");
        for (i, &(tile, rb)) in combos.iter().enumerate() {
            println!(
                "    tile {tile:>3} rb {rb:>4}  {:>8.1} ms  {:>6.2} GMAC/s",
                bc[i],
                mac / (bc[i] / 1e3)
            );
        }
    }
}
