//! Quantized weight matrix — band B2 wiring (`PH-201`/`PH-204` in the forward path).
//!
//! A [`QMat`] holds a projection's **stored GGUF bytes** and runs the fused
//! dequant-on-the-fly `matvec` (kernel B2) instead of materializing a full f32 copy.
//! The dense forward then reads ~4 bits/weight per token instead of 32, which is the
//! memory-bandwidth bound that dominated `rusty serve` (see `docs/HANDOFF_NEW_SESSION.md`).
//! `F32` stays available for synthetic/test weights and any format without a fused kernel.

use crate::core::{EngineError, Result};

/// Minimum MACs (`m × n`, and `rows × m × n` for [`QMat::matmul`]) before a projection is
/// split across worker threads — below this the thread spawn costs more than it saves.
const PAR_MAC_MIN: usize = 1 << 18;

/// Worker count for a projection of `work` MACs, capped to `rows` (each worker needs ≥1 row).
fn workers_for(work: usize, rows: usize) -> usize {
    if work < PAR_MAC_MIN || rows < 2 {
        return 1;
    }
    crate::pool::requested_threads().min(rows).max(1)
}

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
    ///
    /// Rows are independent, so a large enough quant projection is split across worker
    /// threads (`RUSTY_THREADS` / available parallelism). Each row runs the identical
    /// scalar sequence, so results are bit-for-bit independent of the thread count.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Self::F32 { .. } => self.matvec_serial(x),
            Self::Quant {
                ggml_type,
                bytes,
                m,
                n,
            } => {
                let Some(rb) = crate::kernel::quant_row_bytes(*ggml_type, *n) else {
                    return self.matvec_serial(x);
                };
                let threads = workers_for(m * n, *m);
                if threads <= 1 {
                    return self.matvec_serial(x);
                }
                let mut out = vec![0f32; *m];
                let chunk = (*m).div_ceil(threads);
                let t = *ggml_type;
                let mm = *m;
                let jobs = mm.div_ceil(chunk);
                let parts = std::sync::Mutex::new(Vec::<(usize, Vec<f32>)>::with_capacity(jobs));
                crate::pool::parallel_for(jobs, |ci| {
                    let base = ci * chunk;
                    let end = (base + chunk).min(mm);
                    let mut v = Vec::with_capacity(end.saturating_sub(base));
                    for r in base..end {
                        v.push(crate::kernel::quant_row_dot(
                            t,
                            &bytes[r * rb..(r + 1) * rb],
                            x,
                        ));
                    }
                    parts.lock().unwrap().push((ci, v));
                });
                for (ci, v) in parts.into_inner().unwrap() {
                    let base = ci * chunk;
                    out[base..base + v.len()].copy_from_slice(&v);
                }
                out
            }
        }
    }

    /// Serial body of [`QMat::matvec`] (also the fallback for types without a fused kernel).
    fn matvec_serial(&self, x: &[f32]) -> Vec<f32> {
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

    /// Batched `W[m×n] · X[rows×n] → Out[rows×m]` for prefill. Quant formats swap each
    /// block's dequant out of the row loop (see `kernel::matmul_*`), so the int→float
    /// conversions that dominate `matvec` are paid once instead of once per row.
    ///
    /// Split by **input rows**: every worker streams the same weights (a few MiB, re-read
    /// `threads` times) while each owns a `rows/threads` slice of `X` (~290 KiB, L2-sized).
    /// Splitting by output columns instead makes all workers walk the whole activation
    /// matrix per weight row — a shared 16 GiB `X` stream per projection against ~36 MiB of
    /// weight re-reads — which pins the projection to L3 bandwidth (`PH-203k`). Each
    /// worker's rows are contiguous in the row-major `Out`.
    pub fn matmul(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let (m, n) = (self.rows(), self.cols());
        let threads = workers_for(rows.saturating_mul(m).saturating_mul(n), rows);
        let (ggml_type, bytes) = match self {
            Self::Quant {
                ggml_type, bytes, ..
            } => (*ggml_type, bytes.as_slice()),
            Self::F32 { .. } => return self.matmul_serial(x, rows),
        };
        let Some(_rb) = crate::kernel::quant_row_bytes(ggml_type, n) else {
            return self.matmul_serial(x, rows);
        };
        if threads <= 1 {
            return self.matmul_serial(x, rows);
        }
        let chunk = rows.div_ceil(threads);
        let jobs = rows.div_ceil(chunk);
        let parts = std::sync::Mutex::new(Vec::<(usize, Vec<f32>)>::with_capacity(jobs));
        crate::pool::parallel_for(jobs, |ji| {
            let row0 = ji * chunk;
            let rc = chunk.min(rows - row0);
            let got = crate::kernel::matmul_dispatch(
                ggml_type,
                bytes,
                m,
                n,
                &x[row0 * n..(row0 + rc) * n],
                rc,
            );
            parts.lock().unwrap().push((ji, got));
        });
        let mut out = vec![0f32; rows * m];
        for (ji, got) in parts.into_inner().unwrap() {
            let row0 = ji * chunk;
            let rc = chunk.min(rows - row0);
            out[row0 * m..(row0 + rc) * m].copy_from_slice(&got);
        }
        out
    }

    /// Bench A/B (`PH-203k`): the previous prefill decomposition, split by **output
    /// columns**. Kept behind `#[doc(hidden)]` so one process can compare it against the
    /// row split under identical machine noise; production always uses [`QMat::matmul`].
    #[doc(hidden)]
    pub fn matmul_split_cols(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let (m, n) = (self.rows(), self.cols());
        let threads = workers_for(rows.saturating_mul(m).saturating_mul(n), m);
        let (ggml_type, bytes) = match self {
            Self::Quant {
                ggml_type, bytes, ..
            } => (*ggml_type, bytes.as_slice()),
            Self::F32 { .. } => return self.matmul_serial(x, rows),
        };
        let Some(rb) = crate::kernel::quant_row_bytes(ggml_type, n) else {
            return self.matmul_serial(x, rows);
        };
        if threads <= 1 {
            return self.matmul_serial(x, rows);
        }
        let chunk = m.div_ceil(threads);
        let jobs = m.div_ceil(chunk);
        let parts = std::sync::Mutex::new(Vec::<(usize, Vec<f32>)>::with_capacity(jobs));
        crate::pool::parallel_for(jobs, |ci| {
            let col0 = ci * chunk;
            let mc = chunk.min(m - col0);
            let wsub = &bytes[col0 * rb..(col0 + mc) * rb];
            let got = crate::kernel::matmul_dispatch(ggml_type, wsub, mc, n, x, rows);
            parts.lock().unwrap().push((ci, got));
        });
        let mut out = vec![0f32; rows * m];
        for (ci, got) in parts.into_inner().unwrap() {
            let col0 = ci * chunk;
            let mc = chunk.min(m - col0);
            for (r, dst) in out.chunks_mut(m).enumerate() {
                dst[col0..col0 + mc].copy_from_slice(&got[r * mc..(r + 1) * mc]);
            }
        }
        out
    }

    /// Serial body of [`QMat::matmul`].
    fn matmul_serial(&self, x: &[f32], rows: usize) -> Vec<f32> {
        match self {
            Self::F32 { data, m, n } => {
                let mut out = Vec::with_capacity(rows * m);
                for row in x.chunks_exact(*n).take(rows) {
                    out.extend_from_slice(&crate::moe::matvec(data, *m, *n, row));
                }
                out
            }
            Self::Quant {
                ggml_type,
                bytes,
                m,
                n,
            } => match *ggml_type {
                2 => crate::kernel::matmul_q4_0_dispatch(bytes, *m, *n, x, rows),
                6 => crate::kernel::matmul_q5_0_dispatch(bytes, *m, *n, x, rows),
                8 => crate::kernel::matmul_q8_0_dispatch(bytes, *m, *n, x, rows),
                12 => crate::kernel::matmul_q4_k_dispatch(bytes, *m, *n, x, rows),
                13 => crate::kernel::matmul_q5_k(bytes, *m, *n, x, rows),
                14 => crate::kernel::matmul_q6_k_dispatch(bytes, *m, *n, x, rows),
                t => {
                    let d = crate::quant::dequant(t, bytes).unwrap_or_else(|_| {
                        panic!("QMat: no fused matmul and dequant failed for ggml type {t}")
                    });
                    let mut out = Vec::with_capacity(rows * m);
                    for row in x.chunks_exact(*n).take(rows) {
                        out.extend_from_slice(&crate::moe::matvec(&d, *m, *n, row));
                    }
                    out
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

    #[test]
    fn matmul_matches_rowwise_matvec() {
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect();
        let qm = QMat::from_f32(data, 3, 4);
        let x = [1.0f32, -2.0, 0.5, 3.0, 0.0, 1.0, -1.0, 2.0];
        let got = qm.matmul(&x, 2);
        let want: Vec<f32> = (0..2)
            .flat_map(|r| qm.matvec(&x[r * 4..(r + 1) * 4]))
            .collect();
        assert_eq!(got, want);
    }

    /// Q8_0 blocks (34 B: f16 scale = 1.0, then 32 `i8`) for a row-major `m × n`.
    fn q8_blocks(m: usize, n: usize) -> Vec<u8> {
        let nb = m * n / 32;
        let mut a = vec![0u8; nb * 34];
        for (bi, blk) in a.chunks_mut(34).enumerate() {
            blk[0] = 0x00;
            blk[1] = 0x3C; // f16 1.0 (LE)
            for j in 0..32 {
                blk[2 + j] = (((bi * 31 + j * 7) % 251) as i32 - 125) as i8 as u8;
            }
        }
        a
    }

    /// The threaded path is row-independent, so it must be bit-identical to the serial body
    /// (work crosses `PAR_MAC_MIN`, so `matvec`/`matmul` take the parallel branch).
    #[test]
    fn parallel_matvec_is_bit_identical_to_serial() {
        let (m, n) = (64usize, 4096usize);
        let qm = QMat::from_bytes(8, q8_blocks(m, n), m, n);
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 2.0).collect();
        assert_eq!(qm.matvec(&x), qm.matvec_serial(&x));
    }

    #[test]
    fn parallel_matmul_is_bit_identical_to_serial() {
        let (m, n, rows) = (64usize, 4096usize, 8usize);
        let qm = QMat::from_bytes(8, q8_blocks(m, n), m, n);
        let x: Vec<f32> = (0..rows * n).map(|i| (i as f32) * 0.0007 - 1.5).collect();
        assert_eq!(qm.matmul(&x, rows), qm.matmul_serial(&x, rows));
    }
}
