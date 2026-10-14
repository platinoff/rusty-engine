//! Quantized-block sizes + scalar dequant — band B1 `PH-104`.
//!
//! Authoritative layouts cached in `docs/research/ggml-blocks.md` (from
//! `ggml-common.h`/`ggml-quants.c`). This slice implements the two formats with an
//! unambiguous scalar reference — **Q4_0** (split-half nibbles) and **Q8_0** — plus the
//! exact block sizes for **Q4_K/Q5_K/Q6_K** so tensor byte-ranges (and the loader's
//! `slice`) compute correctly. K-quant *dequant math* lands in band **B2** (`PH-201`).

use crate::core::{EngineError, Result};

/// Elements + bytes of one quant block (or one element for f32/f16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockInfo {
    pub elements: usize,
    pub bytes: usize,
}

/// IEEE-754 binary16 → f32 (scale fields are little-endian `u16`).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if (h >> 15) & 1 == 1 { -1.0f64 } else { 1.0f64 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let frac = (h & 0x3FF) as f64;
    let v = match exp {
        0 => frac / 1024.0 * 2f64.powi(-14), // subnormal (frac 0 → ±0)
        31 => {
            if (h & 0x3FF) == 0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    };
    (sign * v) as f32
}

/// `GGML_TYPE_*` code → block geometry (`None` for formats we don't size yet).
pub fn block_info(ggml_type: u32) -> Option<BlockInfo> {
    let (elements, bytes) = match ggml_type {
        0 => (1, 4),      // F32
        1 | 24 => (1, 2), // F16 / BF16
        2 => (32, 18),    // Q4_0
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

/// Dispatch scalar dequant by `GGML_TYPE_*`. Q4_0/Q8_0 supported here; K-quants are
/// sized but their dequant math is `PH-201` (band B2) → `Err`.
pub fn dequant(ggml_type: u32, data: &[u8]) -> Result<Vec<f32>> {
    match ggml_type {
        2 => {
            if data.len() % 18 != 0 {
                return Err(EngineError {
                    message: format!("Q4_0 data len {} not a multiple of 18", data.len()),
                });
            }
            Ok(dequant_q4_0(data))
        }
        8 => {
            if data.len() % 34 != 0 {
                return Err(EngineError {
                    message: format!("Q8_0 data len {} not a multiple of 34", data.len()),
                });
            }
            Ok(dequant_q8_0(data))
        }
        12..=14 => Err(EngineError {
            message: format!(
                "K-quant dequant (ggml {ggml_type}) is band B2; block size via block_info"
            ),
        }),
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
    fn dequant_dispatch_gates_kquants_and_ragged() {
        assert!(dequant(12, &[0u8; 144]).is_err()); // K-quant -> B2
        assert!(dequant(2, &[0u8; 17]).is_err()); // ragged Q4_0
        let mut ok = vec![0u8; 18];
        ok[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        assert_eq!(dequant(2, &ok).unwrap().len(), 32);
    }
}
