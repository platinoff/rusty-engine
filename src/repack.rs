//! Compute-friendly weight layout (repack) — band B2 `PH-202`.
//!
//! The GGML on-disk Q4_0/Q8_0 blocks interleave the two nibbles of each byte and pack
//! the f16 scale inline, which is compact for storage but awkward for a vectorized inner
//! loop. `repack` splits each weight into **contiguous arrays** — a `f32` scale stream and
//! de-interleaved nibble planes (`lo`/`hi`) — so the SIMD kernels (`PH-203`) can broadcast
//! a scale and stride a full byte plane, and a scalar `matvec` reads no f16 per element.
//!
//! The repacked matvec is proven **exactly equal** to the on-disk fused kernel
//! [`crate::kernel::dot_q4_0`] (and to `dequant → matvec`), so repack is a pure layout
//! transform, never a numeric one. `std`-only, no new dependencies.

use crate::core::{EngineError, Result};
use crate::quant::f16_to_f32;

/// Q4_0 super-blocks de-interleaved: `d` (one f32 per 32-elem block) + `lo`/`hi` nibble
/// planes. Element `j` (0..16) uses `lo[j]`; element `j+16` uses `hi[j]` (both biased `-8`).
/// Values are stored as the raw nibble (`0..=15`), not pre-biased, so `d*(v-8)` reproduces
/// GGML bit-for-bit.
#[derive(Debug, Clone, PartialEq)]
pub struct RepackedQ4_0 {
    pub d: Vec<f32>,
    pub lo: Vec<u8>,
    pub hi: Vec<u8>,
    /// Number of 32-element blocks (== `d.len()`).
    pub blocks: usize,
}

/// Q8_0 super-blocks split: one f32 scale per 32-elem block + the raw `i8` byte stream.
#[derive(Debug, Clone, PartialEq)]
pub struct RepackedQ8_0 {
    pub d: Vec<f32>,
    /// Raw signed bytes, 32 per block (kept `u8`, read back as `i8`).
    pub v: Vec<u8>,
    pub blocks: usize,
}

/// Repack a Q4_0 buffer (length a multiple of 18) into [`RepackedQ4_0`].
pub fn repack_q4_0(raw: &[u8]) -> Result<RepackedQ4_0> {
    if raw.len() % 18 != 0 {
        return Err(EngineError {
            message: format!("repack_q4_0: len {} not a multiple of 18", raw.len()),
        });
    }
    let blocks = raw.len() / 18;
    let mut d = Vec::with_capacity(blocks);
    let mut lo = Vec::with_capacity(blocks * 16);
    let mut hi = Vec::with_capacity(blocks * 16);
    for blk in raw.chunks_exact(18) {
        d.push(f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
        for &q in &blk[2..] {
            lo.push(q & 0x0F);
            hi.push(q >> 4);
        }
    }
    Ok(RepackedQ4_0 { d, lo, hi, blocks })
}

/// Repack a Q8_0 buffer (length a multiple of 34) into [`RepackedQ8_0`].
pub fn repack_q8_0(raw: &[u8]) -> Result<RepackedQ8_0> {
    if raw.len() % 34 != 0 {
        return Err(EngineError {
            message: format!("repack_q8_0: len {} not a multiple of 34", raw.len()),
        });
    }
    let blocks = raw.len() / 34;
    let mut d = Vec::with_capacity(blocks);
    let mut v = Vec::with_capacity(blocks * 32);
    for blk in raw.chunks_exact(34) {
        d.push(f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
        v.extend_from_slice(&blk[2..]);
    }
    Ok(RepackedQ8_0 { d, v, blocks })
}

/// `dot(RepackedQ4_0, b)` — the fused read of a repacked weight against a `b` of
/// `blocks*32` floats. Identical summation order per element to [`crate::kernel::dot_q4_0`].
pub fn dot_repacked_q4_0(w: &RepackedQ4_0, b: &[f32]) -> f32 {
    debug_assert_eq!(b.len(), w.blocks * 32, "dot_repacked_q4_0: b length");
    let mut s = 0.0f32;
    for k in 0..w.blocks {
        let d = w.d[k];
        let (lo, hi) = (&w.lo[k * 16..k * 16 + 16], &w.hi[k * 16..k * 16 + 16]);
        let (b0, b1) = b[k * 32..k * 32 + 32].split_at(16);
        for j in 0..16 {
            s += d * (lo[j] as i32 - 8) as f32 * b0[j];
            s += d * (hi[j] as i32 - 8) as f32 * b1[j];
        }
    }
    s
}

/// `dot(RepackedQ8_0, b)` — fused read against `b` of `blocks*32` floats.
pub fn dot_repacked_q8_0(w: &RepackedQ8_0, b: &[f32]) -> f32 {
    debug_assert_eq!(b.len(), w.blocks * 32, "dot_repacked_q8_0: b length");
    let mut s = 0.0f32;
    for k in 0..w.blocks {
        let d = w.d[k];
        for j in 0..32 {
            s += (w.v[k * 32 + j] as i8) as f32 * d * b[k * 32 + j];
        }
    }
    s
}

/// Row-major `W[m×n]` (n multiple of 32) repacked to Q4_0, times `x[n]` → `out[m]`.
pub fn matvec_repacked_q4_0(w: &RepackedQ4_0, m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matvec_repacked_q4_0: n%32");
    debug_assert_eq!(w.blocks * 32, m * n, "matvec_repacked_q4_0: size");
    let blocks_per_row = n / 32;
    (0..m)
        .map(|r| {
            let view = RepackedQ4_0 {
                d: w.d[r * blocks_per_row..(r + 1) * blocks_per_row].to_vec(),
                lo: w.lo[r * n / 2..(r * n / 2) + n / 2].to_vec(),
                hi: w.hi[r * n / 2..(r * n / 2) + n / 2].to_vec(),
                blocks: blocks_per_row,
            };
            dot_repacked_q4_0(&view, x)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{dot_q4_0, dot_q8_0, matvec_q4_0};
    use crate::moe::matvec;
    use crate::quant::{dequant_q4_0, dequant_q8_0};

    const SCALE_1: [u8; 2] = [0x00, 0x3C]; // f16 1.0 (LE)

    fn q4_blocks(nb: usize) -> Vec<u8> {
        let mut a = vec![0u8; nb * 18];
        for (bi, blk) in a.chunks_mut(18).enumerate() {
            blk[0] = SCALE_1[0];
            blk[1] = SCALE_1[1];
            for j in 0..16 {
                blk[2 + j] = ((bi * 37 + j * 13) % 256) as u8;
            }
        }
        a
    }

    fn q8_blocks(nb: usize) -> Vec<u8> {
        let mut a = vec![0u8; nb * 34];
        for (bi, blk) in a.chunks_mut(34).enumerate() {
            blk[0] = SCALE_1[0];
            blk[1] = SCALE_1[1];
            for j in 0..32 {
                blk[2 + j] = ((bi * 53 + j * 29) % 256) as u8;
            }
        }
        a
    }

    #[test]
    fn repack_q4_0_reverses_the_interleave() {
        let raw = q4_blocks(3);
        let r = repack_q4_0(&raw).unwrap();
        assert_eq!(r.blocks, 3);
        assert!(r.d.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        for k in 0..3 {
            for j in 0..16 {
                let q = raw[k * 18 + 2 + j];
                assert_eq!(r.lo[k * 16 + j], q & 0x0F);
                assert_eq!(r.hi[k * 16 + j], q >> 4);
            }
        }
    }

    #[test]
    fn dot_repacked_q4_0_equals_ondisk_kernel() {
        let raw = q4_blocks(4);
        let b: Vec<f32> = (0..raw.len() / 18 * 32)
            .map(|i| (i as f32) * 0.1 - 30.0)
            .collect();
        let r = repack_q4_0(&raw).unwrap();
        let repacked = dot_repacked_q4_0(&r, &b);
        let ondisk = dot_q4_0(&raw, &b);
        let reference = crate::infer::dot(&dequant_q4_0(&raw), &b);
        assert_eq!(repacked, ondisk);
        assert!((repacked - reference).abs() < 1e-3);
    }

    #[test]
    fn dot_repacked_q8_0_equals_ondisk_kernel() {
        let raw = q8_blocks(4);
        let b: Vec<f32> = (0..raw.len() / 34 * 32)
            .map(|i| (i as f32) * 0.05 - 10.0)
            .collect();
        let r = repack_q8_0(&raw).unwrap();
        let repacked = dot_repacked_q8_0(&r, &b);
        let ondisk = dot_q8_0(&raw, &b);
        let reference = crate::infer::dot(&dequant_q8_0(&raw), &b);
        assert_eq!(repacked, ondisk);
        assert!((repacked - reference).abs() < 1e-2);
    }

    #[test]
    fn matvec_repacked_q4_0_matches_fused_and_f32() {
        let (m, n) = (3usize, 64usize);
        let raw = q4_blocks(m * n / 32);
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 - 5.0).collect();
        let r = repack_q4_0(&raw).unwrap();
        let repacked = matvec_repacked_q4_0(&r, m, n, &x);
        let fused = matvec_q4_0(&raw, m, n, &x);
        let reference = matvec(&dequant_q4_0(&raw), m, n, &x);
        assert_eq!(repacked, fused);
        for (a, b) in repacked.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-2);
        }
    }

    #[test]
    fn repack_rejects_bad_block_length() {
        assert!(repack_q4_0(&[0u8; 19]).is_err());
        assert!(repack_q8_0(&[0u8; 35]).is_err());
    }
}
