//! Quantized-block sizes + scalar dequant — band B1 `PH-104` + band B2 `PH-202`.
//!
//! Authoritative layouts + exact `dequantize_row_q{4,5,6}_K` scalar references cached in
//! `docs/research/ggml-blocks.md` (from `ggml-common.h`/`ggml-quants.c`, fetched via
//! explore-subagent). Implements: Q4_0 (split-half) + Q8_0 (signed, `PH-104`), block sizes
//! for all formats, and now the **Q4_K/Q5_K/Q6_K** scalar reference dequant (`PH-202`),
//! transcribed faithfully from the C (6-bit shared scale/min `get_scale_min_k4`, signed
//! 8-bit Q6_K scales with `-32` bias, no lookup table). SIMD + repack follow in `PH-203/204`.

use crate::core::{EngineError, Result};

/// Elements + bytes of one quant block (or one element for f32/f16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockInfo {
    pub elements: usize,
    pub bytes: usize,
}

/// IEEE-754 binary16 → f32 (scale fields are little-endian `u16`).
///
/// `PH-203i`: this used to go through `f64` + `powi`, which cost more than the dot itself
/// (one call per 32-value block; q4_k amortizes it over 256 values, q8_0 does not — the
/// 1.5 GMAC/s mystery). The bit widening below is exact (same bits as IEEE, including
/// subnormals / Inf / NaN) and lowers to a handful of integer ops.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = (h >> 10) & 0x1F;
    let frac = (h & 0x3FF) as u32;
    let bits = match exp {
        0 if frac == 0 => sign, // ±0
        0 => {
            // Subnormal: frac · 2^-24. Normalize so bit 10 is the implicit leading 1.
            let mut e: u32 = 113; // 127 - 14
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3FF) << 13)
        }
        0x1F => sign | 0x7F80_0000 | (frac << 13), // ±Inf (frac 0) / NaN payload
        _ => sign | (((exp as u32) + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// `GGML_TYPE_*` code → block geometry (`None` for formats we don't size yet).
pub fn block_info(ggml_type: u32) -> Option<BlockInfo> {
    let (elements, bytes) = match ggml_type {
        0 => (1, 4),      // F32
        1 | 30 => (1, 2), // F16 / BF16 (canonical ggml.h: I8=24, BF16=30 — PH-1201 mmproj)
        2 => (32, 18),    // Q4_0
        6 => (32, 22),    // Q5_0
        8 => (32, 34),    // Q8_0
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        _ => return None,
    };
    Some(BlockInfo { elements, bytes })
}

/// Bytes occupied by `numel` elements of `ggml_type`, or `None` if the count is not a
/// whole number of blocks (or the type is unknown).
pub fn bytes_for_numel(ggml_type: u32, numel: usize) -> Option<usize> {
    let bi = block_info(ggml_type)?;
    if bi.elements == 1 {
        return Some(numel * bi.bytes);
    }
    if numel % bi.elements != 0 {
        return None;
    }
    Some(numel / bi.elements * bi.bytes)
}

/// Scalar dequant of a whole Q4_0 buffer (length must be a multiple of 18).
///
/// Split-half mapping: low nibble of byte `j` → element `j`, high nibble → element `j+16`.
pub fn dequant_q4_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 18 * 32);
    for blk in data.chunks_exact(18) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let mut vals = [0f32; 32];
        for j in 0..16 {
            let q = blk[2 + j];
            vals[j] = ((q & 0x0F) as i32 - 8) as f32 * d;
            vals[j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
        }
        out.extend_from_slice(&vals);
    }
    out
}

/// Scalar dequant of a whole Q5_0 buffer (length must be a multiple of 22).
///
/// Current-master `block_q5_0` layout: `d`@0 (f16), `qh`@2 (u32 LE, 5th bits), `qs`@6 (16 bytes).
/// Element value = `d * ((nibble | (qh_bit << 4)) - 16)` — bias is −16 (range −16..15); the 5th
/// bit index equals the element index (low nibble → element `j`, high nibble → element `j+16`).
pub fn dequant_q5_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 22 * 32);
    for blk in data.chunks_exact(22) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        let mut vals = [0f32; 32];
        for j in 0..16 {
            let bit0 = ((qh >> j) & 1) << 4;
            let bit1 = ((qh >> (j + 16)) & 1) << 4;
            let x0 = ((qs[j] & 0x0F) as u32 | bit0) as i32 - 16;
            let x1 = ((qs[j] >> 4) as u32 | bit1) as i32 - 16;
            vals[j] = x0 as f32 * d;
            vals[j + 16] = x1 as f32 * d;
        }
        out.extend_from_slice(&vals);
    }
    out
}

/// Scalar dequant of a whole Q8_0 buffer (length must be a multiple of 34).
pub fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 34 * 32);
    for blk in data.chunks_exact(34) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let mut vals = [0f32; 32];
        for j in 0..32 {
            vals[j] = (blk[2 + j] as i8) as f32 * d;
        }
        out.extend_from_slice(&vals);
    }
    out
}

/// 6-bit scale/min extraction shared by Q4_K and Q5_K (llama.cpp `get_scale_min_k4`).
/// `q` is the 12-byte super-block scale array; returns `(scale, min)` for sub-block `j`.
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Scalar reference dequant of Q4_K blocks (256 elements / 144 bytes). Faithful
/// transcription of `dequantize_row_q4_K`; `y = d*sc*q - dmin*m` with 6-bit scales/mins.
#[allow(clippy::needless_range_loop)]
pub fn dequant_q4_k(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 144 * 256);
    for b in data.chunks_exact(144) {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let min = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let scales = &b[4..16];
        let qs = &b[16..144];
        for grp in 0..4 {
            let (sc0, m0) = get_scale_min_k4(grp * 2, scales);
            let (sc1, m1) = get_scale_min_k4(grp * 2 + 1, scales);
            let d1 = d * sc0 as f32;
            let dm1 = min * m0 as f32;
            let d2 = d * sc1 as f32;
            let dm2 = min * m1 as f32;
            let base = grp * 32;
            for l in 0..32 {
                let q = qs[base + l];
                out.push(d1 * (q & 0xF) as f32 - dm1);
            }
            for l in 0..32 {
                let q = qs[base + l];
                out.push(d2 * (q >> 4) as f32 - dm2);
            }
        }
    }
    out
}

/// Scalar reference dequant of Q5_K blocks (256 elements / 176 bytes). Shares Q4_K's
/// 6-bit scale layout; adds the 5th bit from `qh` (mask `1<<(2*grp)` low / `2<<(2*grp)` high).
#[allow(clippy::needless_range_loop)]
pub fn dequant_q5_k(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 176 * 256);
    for b in data.chunks_exact(176) {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let min = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let scales = &b[4..16];
        let qh = &b[16..48];
        let qs = &b[48..176];
        for grp in 0..4 {
            let (sc0, m0) = get_scale_min_k4(grp * 2, scales);
            let (sc1, m1) = get_scale_min_k4(grp * 2 + 1, scales);
            let d1 = d * sc0 as f32;
            let dm1 = min * m0 as f32;
            let d2 = d * sc1 as f32;
            let dm2 = min * m1 as f32;
            let base = grp * 32;
            let u1 = 1u8 << (2 * grp);
            let u2 = 2u8 << (2 * grp);
            for l in 0..32 {
                let lo = (qs[base + l] & 0xF) as i32 + if qh[l] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * lo as f32 - dm1);
            }
            for l in 0..32 {
                let hi = (qs[base + l] >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * hi as f32 - dm2);
            }
        }
    }
    out
}

/// Scalar reference dequant of Q6_K blocks (256 elements / 210 bytes). Signed 8-bit
/// `scales[16]`, no min; value = `ql_nibble | (qh_2bits << 4)` biased by `-32`, result `d*sc*q`.
#[allow(clippy::needless_range_loop)]
pub fn dequant_q6_k(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 210 * 256);
    for b in data.chunks_exact(210) {
        let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
        let ql = &b[0..128];
        let qh = &b[128..192];
        let sc = &b[192..208];
        for half in 0..2 {
            let mut tmp = [0.0f32; 128];
            let qb = half * 64;
            let hb = half * 32;
            let sb = half * 8;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[qb + l] & 0xF) as i32 | ((qh[hb + l] & 3) as i32) << 4) - 32;
                let q2 =
                    ((ql[qb + l + 32] & 0xF) as i32 | (((qh[hb + l] >> 2) & 3) as i32) << 4) - 32;
                let q3 = ((ql[qb + l] >> 4) as i32 | (((qh[hb + l] >> 4) & 3) as i32) << 4) - 32;
                let q4 =
                    ((ql[qb + l + 32] >> 4) as i32 | (((qh[hb + l] >> 6) & 3) as i32) << 4) - 32;
                tmp[l] = d * sc[sb + is] as i8 as f32 * q1 as f32;
                tmp[l + 32] = d * sc[sb + is + 2] as i8 as f32 * q2 as f32;
                tmp[l + 64] = d * sc[sb + is + 4] as i8 as f32 * q3 as f32;
                tmp[l + 96] = d * sc[sb + is + 6] as i8 as f32 * q4 as f32;
            }
            out.extend_from_slice(&tmp);
        }
    }
    out
}

/// Dispatch scalar dequant by `GGML_TYPE_*` for all supported formats.
/// F32/F16 are read directly (they carry no quantization — norms and small tensors
/// stay in these types on real Q4_K_M models). Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K use the
/// block oracles above.
pub fn dequant(ggml_type: u32, data: &[u8]) -> Result<Vec<f32>> {
    match ggml_type {
        0 => {
            if data.len() % 4 != 0 {
                return Err(EngineError {
                    message: format!("F32 data len {} not a multiple of 4", data.len()),
                });
            }
            Ok(data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        1 => {
            if data.len() % 2 != 0 {
                return Err(EngineError {
                    message: format!("F16 data len {} not a multiple of 2", data.len()),
                });
            }
            Ok(data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        // GGML_TYPE_BF16 = 30 (canonical ggml.h) — the MiniCPM-V 4.7 mmproj stores every
        // vision weight as a 2-byte BF16 (`PH-1201`); bf16 is the truncated top half of an
        // f32, so widening is a left-shift, no table.
        30 => {
            if data.len() % 2 != 0 {
                return Err(EngineError {
                    message: format!("BF16 data len {} not a multiple of 2", data.len()),
                });
            }
            Ok(data
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect())
        }
        2 => {
            if data.len() % 18 != 0 {
                return Err(EngineError {
                    message: format!("Q4_0 data len {} not a multiple of 18", data.len()),
                });
            }
            Ok(dequant_q4_0(data))
        }
        6 => {
            if data.len() % 22 != 0 {
                return Err(EngineError {
                    message: format!("Q5_0 data len {} not a multiple of 22", data.len()),
                });
            }
            Ok(dequant_q5_0(data))
        }
        8 => {
            if data.len() % 34 != 0 {
                return Err(EngineError {
                    message: format!("Q8_0 data len {} not a multiple of 34", data.len()),
                });
            }
            Ok(dequant_q8_0(data))
        }
        12 => {
            if data.len() % 144 != 0 {
                return Err(EngineError {
                    message: format!("Q4_K data len {} not a multiple of 144", data.len()),
                });
            }
            Ok(dequant_q4_k(data))
        }
        13 => {
            if data.len() % 176 != 0 {
                return Err(EngineError {
                    message: format!("Q5_K data len {} not a multiple of 176", data.len()),
                });
            }
            Ok(dequant_q5_k(data))
        }
        14 => {
            if data.len() % 210 != 0 {
                return Err(EngineError {
                    message: format!("Q6_K data len {} not a multiple of 210", data.len()),
                });
            }
            Ok(dequant_q6_k(data))
        }
        t => Err(EngineError {
            message: format!("no scalar dequant for ggml type {t}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2f32.powi(-24)); // largest subnormal
        assert!(f16_to_f32(0x7E00).is_nan());
        assert!(f16_to_f32(0xFE00).is_nan());
    }

    /// `PH-203i`: the fast widening must be bit-exact with the slow f64 reference for every
    /// `u16` pattern (subnormals, Inf, NaN included).
    #[test]
    fn f16_bit_exact_against_f64_reference() {
        fn slow(h: u16) -> f32 {
            let sign = if (h >> 15) & 1 == 1 { -1.0f64 } else { 1.0f64 };
            let exp = ((h >> 10) & 0x1F) as i32;
            let frac = (h & 0x3FF) as f64;
            let v = match exp {
                0 => frac / 1024.0 * 2f64.powi(-14),
                31 if (h & 0x3FF) == 0 => f64::INFINITY,
                31 => f64::NAN,
                _ => (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
            };
            (sign * v) as f32
        }
        for h in 0..=u16::MAX {
            let got = f16_to_f32(h);
            let want = slow(h);
            if want.is_nan() {
                assert!(got.is_nan(), "h={h:#06x}: {got} not NaN");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "h={h:#06x}");
            }
        }
    }

    #[test]
    fn block_sizes_match_ggml() {
        assert_eq!(
            block_info(2).unwrap(),
            BlockInfo {
                elements: 32,
                bytes: 18
            }
        );
        assert_eq!(
            block_info(8).unwrap(),
            BlockInfo {
                elements: 32,
                bytes: 34
            }
        );
        assert_eq!(
            block_info(6).unwrap(),
            BlockInfo {
                elements: 32,
                bytes: 22
            }
        );
        assert_eq!(
            block_info(12).unwrap(),
            BlockInfo {
                elements: 256,
                bytes: 144
            }
        );
        assert_eq!(
            block_info(13).unwrap(),
            BlockInfo {
                elements: 256,
                bytes: 176
            }
        );
        assert_eq!(
            block_info(14).unwrap(),
            BlockInfo {
                elements: 256,
                bytes: 210
            }
        );
        assert!(block_info(999).is_none());
    }

    #[test]
    fn bytes_for_numel_block_math() {
        assert_eq!(bytes_for_numel(2, 64), Some(36)); // 2 Q4_0 blocks
        assert_eq!(bytes_for_numel(0, 10), Some(40)); // F32
        assert_eq!(bytes_for_numel(2, 33), None); // not a whole Q4_0 block
    }

    #[test]
    fn q4_0_split_half_and_offset() {
        // scale = 1.0 (0x3C00); byte0 low nibble=15 -> element0 = (15-8)=7 ;
        // byte0 high nibble=0 -> element16 = (0-8) = -8 ; rest of bytes 0 -> -8.
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        blk[2] = 0x0F;
        let out = dequant_q4_0(&blk);
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], 7.0);
        assert_eq!(out[16], -8.0);
        assert_eq!(out[1], -8.0);
        assert_eq!(out[31], -8.0);
    }

    #[test]
    fn q4_0_all_zero_nibbles_minus_eight() {
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1
        let out = dequant_q4_0(&blk);
        assert!(out.iter().all(|&v| v == -8.0));
    }

    #[test]
    fn q8_0_signed_values() {
        let mut blk = vec![0u8; 34];
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1
        blk[2] = 0x7F; // 127
        blk[3] = 0xFF; // -1 (int8)
        let out = dequant_q8_0(&blk);
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], 127.0);
        assert_eq!(out[1], -1.0);
    }

    #[test]
    fn q5_0_values_and_5th_bit() {
        let mut blk = vec![0u8; 22];
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1
                                                             // layout [d@0, qh@2..6, qs@6..22]. qs all 0x11 -> low=1, high=1, no 5th bits => 1-16=-15.
        blk[6..22].fill(0x11);
        let out = dequant_q5_0(&blk);
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| v == -15.0));

        // 5th bit (qh bit0 set): low element0 = (0 | 0x10) - 16 = 0 ; high elem16 = 0 - 16 = -16.
        let mut b2 = vec![0u8; 22];
        b2[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d=1
        b2[2..6].copy_from_slice(&1u32.to_le_bytes()); // qh=1 (bit0)
                                                       // qs zero -> low nibble 0, high 0
        let out2 = dequant_q5_0(&b2);
        assert_eq!(out2[0], 0.0); // ((0 | (1<<4)) - 16) = 0
        assert_eq!(out2[16], -16.0); // (0 | 0) - 16

        // high nibble 15 (0xF0): low elem0 = 0-16=-16 ; high elem16 = 15-16=-1
        let mut b3 = vec![0u8; 22];
        b3[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        b3[6] = 0xF0;
        let out3 = dequant_q5_0(&b3);
        assert_eq!(out3[0], -16.0);
        assert_eq!(out3[16], -1.0);
    }

    #[test]
    fn dequant_dispatch_lengths() {
        assert!(dequant(2, &[0u8; 17]).is_err()); // ragged Q4_0
        assert!(dequant(6, &[0u8; 21]).is_err()); // ragged Q5_0
        assert_eq!(dequant(6, &[0u8; 22]).unwrap().len(), 32);
        assert!(dequant(12, &[0u8; 143]).is_err()); // ragged Q4_K
        assert!(dequant(99, &[0u8; 8]).is_err()); // unsupported type
        let mut ok = vec![0u8; 18];
        ok[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        assert_eq!(dequant(2, &ok).unwrap().len(), 32);
        assert_eq!(dequant(12, &[0u8; 144]).unwrap().len(), 256);
    }

    #[test]
    fn q4_k_zero_block_all_zero() {
        // d=1, min=0, scales=0, qs=0 => y = d*0*q - min*0 = 0 for all 256.
        let mut b = vec![0u8; 144];
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        let out = dequant_q4_k(&b);
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q4_k_uniform_scale_and_min() {
        // d=1, min=1; sub-block0 scale=2 (scales[0]), min=1 (scales[4]); qs low nibble=3.
        // out[0..32] (grp0 low) = d*sc*q - min*m = 1*2*3 - 1*1 = 5.
        let mut b = vec![0u8; 144];
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        b[2..4].copy_from_slice(&0x3C00u16.to_le_bytes());
        b[4] = 2;
        b[8] = 1;
        b[16..144].fill(0x03);
        let out = dequant_q4_k(&b);
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[31] - 5.0).abs() < 1e-5);
        assert_eq!(out[32], 0.0); // grp0 high uses sub-block1 scale = 0
    }

    #[test]
    fn q5_k_low_group() {
        // d=1, min=0, sub-block0 scale=1; qs low nibble=3, qh=0 => out[0] = 1*1*3 = 3.
        let mut b = vec![0u8; 176];
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        b[4] = 1;
        b[48..176].fill(0x03);
        let out = dequant_q5_k(&b);
        assert_eq!(out.len(), 256);
        assert!((out[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn q6_k_signed_zero_bias() {
        // d=1, all scales=1(i8), ql=qh=0 => q = 0 - 32 = -32 => y = 1*1*(-32) = -32 everywhere.
        let mut b = vec![0u8; 210];
        b[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
        b[192..208].fill(1);
        let out = dequant_q6_k(&b);
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| (v + 32.0).abs() < 1e-5));
    }

    #[test]
    fn dequant_f32_roundtrips_le_bytes() {
        // 3 f32 values little-endian.
        let mut bytes = Vec::new();
        for v in [1.5f32, -2.0, 0.125] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = dequant(0, &bytes).expect("F32 dequant");
        assert_eq!(out, vec![1.5, -2.0, 0.125]);
    }

    #[test]
    fn dequant_f16_uses_f16_to_f32() {
        // 0x3C00 = 1.0, 0x4000 = 2.0, 0xCC00 = -16.0 (sign=1, exp=19-15=4, frac=0).
        let bytes: Vec<u8> = [0x3C00u16, 0x4000, 0xCC00]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let out = dequant(1, &bytes).expect("F16 dequant");
        assert_eq!(out, vec![1.0, 2.0, -16.0]);
    }

    #[test]
    fn dequant_bf16_widens_by_left_shift() {
        // PH-1201: GGML_TYPE_BF16 = 30, every MiniCPM-V mmproj weight. 1.0 = 0x3F80,
        // 0.5 = 0x3F00, -2.0 = 0xC000, 1234.5 keeps the top 16 bits of its f32.
        let bytes: Vec<u8> = [0x3F80u16, 0x3F00, 0xC000, 0x4452]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let out = dequant(30, &bytes).expect("BF16 dequant");
        assert_eq!(out, vec![1.0, 0.5, -2.0, 840.0]);
        assert_eq!(
            dequant(30, &[0u8; 3])
                .expect_err("BF16 length must be even")
                .message,
            "BF16 data len 3 not a multiple of 2"
        );
    }

    #[test]
    fn dequant_f32_rejects_partial_word() {
        let err = dequant(0, &[0u8; 3]).expect_err("F32 length must be multiple of 4");
        assert!(err.message.contains("F32"), "{err:?}");
    }
}
