//! `PH-203q` — activation-tile sweep for the 256-value K-quants.
//!
//! `cargo bench --bench kquant` (release). Single thread, min-of-5 per tile, deterministic
//! xorshift buffers: the sweep isolates what a K-quant tile buys, because the tile is the only
//! thing that decides how often a 256-value super-block is dequantized.
//!
//! Shapes are the real ones from `Qwen2.5-0.5B Q4_K_M`: `ffn_down` is the only K-quant tensor
//! (`Q6_K`, `896 × 4864`), so the wide shape leads; the narrow `4864 × 896` row is the counter
//! case where a larger tile hurts the L2-resident activation tile.
//!
//! Deliberately **std-only** (no `criterion`), same discipline as `benches/quant.rs`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use rusty_engine::kernel as k;

fn fill_bytes(seed: u64, out: &mut [u8]) {
    let mut s = seed | 1;
    for b in out.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = (s & 0xFF) as u8;
    }
}

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

/// `n` super-blocks of `block_bytes`, with a finite f16 `d` stamped at the type's offset
/// (Q4_K/Q5_K at 0, Q6_K at the tail) so the results stay finite.
fn blocks(seed: u64, n: usize, block_bytes: usize, d_off: usize) -> Vec<u8> {
    const HALF: [u8; 2] = [0x00, 0x38]; // f16 0.5, little-endian
    let mut v = vec![0u8; n * block_bytes];
    fill_bytes(seed, &mut v);
    for blk in v.chunks_exact_mut(block_bytes) {
        blk[d_off..d_off + 2].copy_from_slice(&HALF);
    }
    v
}

fn time(iters: usize, mut f: impl FnMut()) -> Duration {
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed()
}

/// The production prefill split (`QMat::matmul_out_rows`) reduced to its kernel calls: split
/// `Out` by weight rows, let the shared pool run one [`k::matmul_k_tiled`] per worker.
///
/// This is the shape the live server actually runs, and it is the only way to see the tile
/// trade-off: single-threaded (above) a bigger tile always wins because dequant is amortized,
/// while under 12 workers the activation tile is shared but each worker adds its own row
/// buffers, so the pooled optimum is not the single-thread optimum.
#[allow(clippy::too_many_arguments)]
fn pool_matmul(
    ty: u32,
    w: &[u8],
    m: usize,
    n: usize,
    x: &[f32],
    rows: usize,
    tile: usize,
    w_rows: usize,
) -> Vec<f32> {
    let threads = rusty_engine::pool::requested_threads().max(1);
    let chunk = m.div_ceil(threads);
    let jobs = m.div_ceil(chunk);
    if jobs <= 1 {
        return k::matmul_k_tiled_rows(ty, w, m, n, x, rows, tile, w_rows);
    }
    let row_bytes = k::quant_row_bytes(ty, n).expect("kquant: fused row");
    let parts = std::sync::Mutex::new(Vec::<(usize, Vec<f32>)>::with_capacity(jobs));
    rusty_engine::pool::parallel_for(jobs, |ji| {
        let r0 = ji * chunk;
        let rc = chunk.min(m - r0);
        let got = k::matmul_k_tiled_rows(
            ty,
            &w[r0 * row_bytes..(r0 + rc) * row_bytes],
            rc,
            n,
            x,
            rows,
            tile,
            w_rows,
        );
        parts.lock().expect("parts").push((ji, got));
    });
    let mut out = vec![0f32; rows * m];
    for (ji, got) in parts.into_inner().expect("parts") {
        let r0 = ji * chunk;
        let rc = chunk.min(m - r0);
        for row in 0..rows {
            out[row * m + r0..row * m + r0 + rc].copy_from_slice(&got[row * rc..(row + 1) * rc]);
        }
    }
    out
}

fn main() {
    let rows = 961usize;
    let tiles = [16usize, 24, 32, 48, 64, 96, 128, 192, 256];
    // (label, ggml_type, block_bytes, d_off, m, n)
    let shapes = [
        (
            "down   896x4864 Q6_K",
            14u32,
            210usize,
            208usize,
            896usize,
            4864usize,
        ),
        ("gate/up 4864x896 Q6_K", 14, 210, 208, 4864, 896),
        ("down   896x4864 Q4_K", 12, 144, 0, 896, 4864),
    ];

    for (label, ty, block_bytes, d_off, m, n) in shapes {
        let w = blocks(41, m * n / 256, block_bytes, d_off);
        let x = fill_floats(43, rows * n);
        let mac = rows as f64 * m as f64 * n as f64 / 1e9;
        let mut best = vec![f64::INFINITY; tiles.len()];
        for _ in 0..5 {
            for (i, &tile) in tiles.iter().enumerate() {
                let t = time(1, || {
                    black_box(k::matmul_k_tiled(ty, &w, m, n, &x, rows, tile));
                });
                best[i] = best[i].min(t.as_secs_f64() * 1e3);
            }
        }
        println!("{label}  rows {rows}  ({mac:.2} GMAC/run, min of 5, single thread):");
        for (i, &tile) in tiles.iter().enumerate() {
            println!(
                "  tile {tile:>3}  {:>8.1} ms  {:>6.2} GMAC/s",
                best[i],
                mac / (best[i] / 1e3)
            );
        }
        let base = best[0];
        println!(
            "  best {:>4} vs tile {}: {:>5.2}x",
            tiles[best
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
                .expect("tiles")
                .0],
            tiles[0],
            base / best.iter().cloned().fold(f64::INFINITY, f64::min)
        );

        // Same sweep through the production split (output rows across the pool): this is where
        // the tile actually has to be chosen, because each worker gets its own activation tile.
        let mut pooled = vec![f64::INFINITY; tiles.len()];
        for _ in 0..3 {
            for (i, &tile) in tiles.iter().enumerate() {
                let t = time(1, || {
                    black_box(pool_matmul(ty, &w, m, n, &x, rows, tile, 2));
                });
                pooled[i] = pooled[i].min(t.as_secs_f64() * 1e3);
            }
        }
        println!(
            "  pooled ({}-thread out-rows split, min of 3):",
            rusty_engine::pool::requested_threads()
        );
        for (i, &tile) in tiles.iter().enumerate() {
            println!(
                "    tile {tile:>3}  {:>8.1} ms  {:>6.2} GMAC/s",
                pooled[i],
                mac / (pooled[i] / 1e3)
            );
        }
        let (best_i, _) = pooled
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .expect("tiles");
        println!(
            "  pooled best tile {}: {:>5.2}x vs tile {}",
            tiles[best_i],
            pooled[0] / pooled[best_i],
            tiles[0]
        );

        // PH-203r: pairing two weight rows changes which tile the pool wants, so the two knobs
        // are measured together instead of reusing the single-row conclusion.
        println!("  pooled w_rows x tile (min of 3, GMAC/s):");
        for wr in [1usize, 2] {
            let mut line = format!("    w_rows {wr}:");
            for &tile in &[24usize, 48, 96, 192] {
                let mut ms = f64::INFINITY;
                for _ in 0..3 {
                    let t = time(1, || {
                        black_box(pool_matmul(ty, &w, m, n, &x, rows, tile, wr));
                    });
                    ms = ms.min(t.as_secs_f64() * 1e3);
                }
                line.push_str(&format!("  t{tile:<3} {:>6.2}", mac / (ms / 1e3)));
            }
            println!("{line}");
        }
    }
}
