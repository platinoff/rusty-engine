//! Quantized weight matrix — band B2 wiring (`PH-201`/`PH-204` in the forward path).
//!
//! A [`QMat`] holds a projection's **stored GGUF bytes** and runs the fused
//! dequant-on-the-fly `matvec` (kernel B2) instead of materializing a full f32 copy.
//! The dense forward then reads ~4 bits/weight per token instead of 32, which is the
//! memory-bandwidth bound that dominated `rusty serve` (see `docs/HANDOFF_NEW_SESSION.md`).
//! `F32` stays available for synthetic/test weights and any format without a fused kernel.

use crate::core::{EngineError, Result};

/// A row-major `m × n` weight matrix: either f32, or its stored quant blocks.
#[derive(Debug, Clone)]
pub enum QMat {
    F32 {
        data: Vec<f32>,
        m: usize,
        n: usize,
    },
    Quant {
        ggml_type: u32,
        bytes: Vec<u8>,
        m: usize,
        n: usize,
    },
}

impl QMat {
    /// Wrap an f32 matrix (row-major `m × n`).
    pub fn from_f32(data: Vec<f32>, m: usize, n: usize) -> Self {
        assert_eq!(data.len(), m * n, "QMat::from_f32: shape mismatch");
        Self::F32 { data, m, n }
    }

    /// Wrap raw GGUF tensor bytes (`ggml_type` selects the fused matvec at call time).
    pub fn from_bytes(ggml_type: u32, bytes: Vec<u8>, m: usize, n: usize) -> Self {
        Self::Quant {
            ggml_type,
            bytes,
            m,
            n,
        }
    }

    /// Rows (`m`) of the matrix.
    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { m, .. } | Self::Quant { m, .. } => *m,
        }
    }

    /// Columns (`n`, the inner width) of the matrix.
    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { n, .. } | Self::Quant { n, .. } => *n,
        }
    }

    /// `W[m×n] · x[n] → out[m]`, fused per row (no full f32 materialization for quant).
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Self::F32 { data, m, n } => crate::moe::matvec(data, *m, *n, x),
            Self::Quant {
                ggml_type,
                bytes,
                m,
                n,
            } => match *ggml_type {
                2 => crate::kernel::matvec_q4_0_dispatch(bytes, *m, *n, x),
                6 => crate::kernel::matvec_q5_0_dispatch(bytes, *m, *n, x),
                8 => crate::kernel::matvec_q8_0_dispatch(bytes, *m, *n, x),
                12 => crate::kernel::matvec_q4_k_dispatch(bytes, *m, *n, x),
                13 => crate::kernel::matvec_q5_k(bytes, *m, *n, x),
                14 => crate::kernel::matvec_q6_k_dispatch(bytes, *m, *n, x),
                t => {
                    // No fused kernel (e.g. F32/F16): scalar dequant then f32 matvec.
                    let d = crate::quant::dequant(t, bytes).unwrap_or_else(|_| {
                        panic!("QMat: no fused kernel and dequant failed for ggml type {t}")
                    });
                    crate::moe::matvec(&d, *m, *n, x)
                }
            },
        }
    }

    /// Build from a mapped model tensor by name; `m`/`n` come from the model geometry
    /// (GGUF stores `W[m×n]` as `dims = [n, m]`). Errors when the tensor is absent.
    pub fn from_model(
        model: &crate::loader::MmapModel,
        name: &str,
        m: usize,
        n: usize,
    ) -> Result<Self> {
        let e = model
            .header()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| EngineError {
                message: format!("QMat: tensor '{name}' missing"),
            })?;
        let bytes = model.tensor_bytes(name).ok_or_else(|| EngineError {
            message: format!("QMat: tensor '{name}' bytes out of bounds"),
        })?;
        Ok(Self::from_bytes(e.ggml_type, bytes.to_vec(), m, n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f32_matches_plain_matvec() {
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect();
        let qm = QMat::from_f32(data.clone(), 3, 4);
        let x = [1.0f32, -2.0, 0.5, 3.0];
        assert_eq!(qm.matvec(&x), crate::moe::matvec(&data, 3, 4, &x));
        assert_eq!(qm.rows(), 3);
        assert_eq!(qm.cols(), 4);
    }

    #[test]
    fn quant_f32_fallback_matches() {
        // ggml type 0 (F32) has no fused kernel -> dequant fallback must match.
        let data: Vec<f32> = (0..6).map(|i| 0.5 - i as f32 * 0.1).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let qm = QMat::from_bytes(0, bytes, 2, 3);
        let x = [2.0f32, -1.0, 0.25];
        assert_eq!(qm.matvec(&x), crate::moe::matvec(&data, 2, 3, &x));
    }
}
