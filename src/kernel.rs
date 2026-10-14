//! Scalar reference kernels — band B2 `PH-201` + `PH-204`.
//!
//! Dequant-on-the-fly `dot`/`gemv` for the quant formats whose dequant is already
//! validated (`Q4_0`, `Q8_0` from `PH-104`, K-quant from `PH-201+`). These are the
//! **correctness oracle** the SIMD backends (`PH-203`) must match, so each fused kernel
//! here is tested against the naive `dequant → dot` path. The K-quant loops dequantize
//! a 256-element super-block into a fixed stack buffer and fold it straight into the dot
//! (no heap `Vec`), matching the scalar reference element-for-element.
//!
//! Layouts per `docs/research/ggml-blocks.md` (Q4_0 split-half: low nibble → element `j`,
//! high nibble → element `j+16`).

use crate::quant::{f16_to_f32, get_scale_min_k4};

/// `dot(dequant_q4_0(a), b)` without materializing the `f32` buffer. `a` is a whole number
/// of 18-byte Q4_0 blocks; `b` is `a.len()/18*32` floats.
pub fn dot_q4_0(a: &[u8], b: &[f32]) -> f32 {
    let nb = a.len() / 18;
    debug_assert_eq!(a.len(), nb * 18, "dot_q4_0: partial block");
    debug_assert_eq!(b.len(), nb * 32, "dot_q4_0: vector length mismatch");
    let mut s = 0.0f32;
    for (block, bchunk) in a.chunks_exact(18).zip(b.chunks(32)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let (lo, hi) = bchunk.split_at(16);
        for (&q, (l, h)) in block[2..].iter().zip(lo.iter().zip(hi.iter())) {
            s += d * ((q & 0x0F) as i32 - 8) as f32 * l;
            s += d * ((q >> 4) as i32 - 8) as f32 * h;
        }
    }
    s
}

/// `dot(dequant_q8_0(a), b)` fused. `a` is a whole number of 34-byte Q8_0 blocks.
pub fn dot_q8_0(a: &[u8], b: &[f32]) -> f32 {
    let nb = a.len() / 34;
    debug_assert_eq!(a.len(), nb * 34, "dot_q8_0: partial block");
    debug_assert_eq!(b.len(), nb * 32, "dot_q8_0: vector length mismatch");
    let mut s = 0.0f32;
    for (block, bchunk) in a.chunks_exact(34).zip(b.chunks(32)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (&q, bx) in block[2..].iter().zip(bchunk) {
            s += (q as i8) as f32 * d * bx;
        }
    }
    s
}

/// Row-major `W[m×n]` stored as Q4_0 blocks, times `x[n]` → `out[m]`, fused per row.
pub fn matvec_q4_0(w: &[u8], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matvec_q4_0: n must be a multiple of 32");
    let row_bytes = n / 32 * 18;
    debug_assert_eq!(w.len(), m * row_bytes, "matvec_q4_0: weight size mismatch");
    debug_assert_eq!(x.len(), n, "matvec_q4_0: input width mismatch");
    (0..m)
        .map(|r| dot_q4_0(&w[r * row_bytes..(r + 1) * row_bytes], x))
        .collect()
}

/// Q8_0 analogue of [`matvec_q4_0`].
pub fn matvec_q8_0(w: &[u8], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matvec_q8_0: n must be a multiple of 32");
    let row_bytes = n / 32 * 34;
    debug_assert_eq!(w.len(), m * row_bytes, "matvec_q8_0: weight size mismatch");
    debug_assert_eq!(x.len(), n, "matvec_q8_0: input width mismatch");
    (0..m)
        .map(|r| dot_q8_0(&w[r * row_bytes..(r + 1) * row_bytes], x))
        .collect()
}

/// `dot(dequant_q5_0(a), b)` fused. `a` is a whole number of 22-byte Q5_0 blocks
/// (`d`@0, `qh`@2, `qs`@6); `b` is `a.len()/22*32` floats. Mirrors `quant::dequant_q5_0`
/// exactly (bias −16, 5th bit from `qh`), so it equals the `dequant → dot` oracle.
pub fn dot_q5_0(a: &[u8], b: &[f32]) -> f32 {
    let nb = a.len() / 22;
    debug_assert_eq!(a.len(), nb * 22, "dot_q5_0: partial block");
    debug_assert_eq!(b.len(), nb * 32, "dot_q5_0: vector length mismatch");
    let mut s = 0.0f32;
    for (blk, bchunk) in a.chunks_exact(22).zip(b.chunks(32)) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        let (lo, hi) = bchunk.split_at(16);
        for j in 0..16 {
            let bit0 = (((qh >> j) & 1) << 4) as i32;
            let bit1 = (((qh >> (j + 16)) & 1) << 4) as i32;
            let x0 = ((qs[j] & 0x0F) as i32 | bit0) - 16;
            let x1 = ((qs[j] >> 4) as i32 | bit1) - 16;
            s += x0 as f32 * d * lo[j];
            s += x1 as f32 * d * hi[j];
        }
    }
    s
}

/// Row-major `W[m×n]` stored as Q5_0 blocks, times `x[n]` → `out[m]`, fused per row.
pub fn matvec_q5_0(w: &[u8], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matvec_q5_0: n must be a multiple of 32");
    let row_bytes = n / 32 * 22;
    debug_assert_eq!(w.len(), m * row_bytes, "matvec_q5_0: weight size mismatch");
    debug_assert_eq!(x.len(), n, "matvec_q5_0: input width mismatch");
    (0..m)
        .map(|r| dot_q5_0(&w[r * row_bytes..(r + 1) * row_bytes], x))
        .collect()
}

/// Dequantize one 144-byte Q4_K super-block into `out` (256 f32), mirroring
/// `quant::dequant_q4_k` exactly.
fn dq_q4_k_block(b: &[u8], out: &mut [f32; 256]) {
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
            out[grp * 64 + l] = d1 * (q & 0xF) as f32 - dm1;
            out[grp * 64 + 32 + l] = d2 * (q >> 4) as f32 - dm2;
        }
    }
}

/// Dequantize one 176-byte Q5_K super-block into `out` (256 f32), mirroring
/// `quant::dequant_q5_k`.
fn dq_q5_k_block(b: &[u8], out: &mut [f32; 256]) {
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
            let hi = (qs[base + l] >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 };
            out[grp * 64 + l] = d1 * lo as f32 - dm1;
            out[grp * 64 + 32 + l] = d2 * hi as f32 - dm2;
        }
    }
}

/// Dequantize one 210-byte Q6_K super-block into `out` (256 f32), mirroring
/// `quant::dequant_q6_k`.
fn dq_q6_k_block(b: &[u8], out: &mut [f32; 256]) {
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
            let q2 = ((ql[qb + l + 32] & 0xF) as i32 | (((qh[hb + l] >> 2) & 3) as i32) << 4) - 32;
            let q3 = ((ql[qb + l] >> 4) as i32 | (((qh[hb + l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[qb + l + 32] >> 4) as i32 | (((qh[hb + l] >> 6) & 3) as i32) << 4) - 32;
            tmp[l] = d * sc[sb + is] as i8 as f32 * q1 as f32;
            tmp[l + 32] = d * sc[sb + is + 2] as i8 as f32 * q2 as f32;
            tmp[l + 64] = d * sc[sb + is + 4] as i8 as f32 * q3 as f32;
            tmp[l + 96] = d * sc[sb + is + 6] as i8 as f32 * q4 as f32;
        }
        out[half * 128..half * 128 + 128].copy_from_slice(&tmp);
    }
}

/// `dot(dequant_q4_k(a), b)` fused — dequantizes each 256-element super-block into a stack
/// buffer and folds it into the dot without materializing the whole row.
pub fn dot_q4_k(a: &[u8], b: &[f32]) -> f32 {
    let nb = a.len() / 144;
    debug_assert_eq!(a.len(), nb * 144, "dot_q4_k: partial block");
    debug_assert_eq!(b.len(), nb * 256, "dot_q4_k: vector length mismatch");
    let mut s = 0.0f32;
    let mut buf = [0.0f32; 256];
    for (block, bchunk) in a.chunks_exact(144).zip(b.chunks(256)) {
        dq_q4_k_block(block, &mut buf);
        s += buf.iter().zip(bchunk).map(|(v, x)| v * x).sum::<f32>();
    }
    s
}

/// `dot(dequant_q5_k(a), b)` fused (176-byte blocks, 5th bit from `qh`).
pub fn dot_q5_k(a: &[u8], b: &[f32]) -> f32 {
    let nb = a.len() / 176;
    debug_assert_eq!(a.len(), nb * 176, "dot_q5_k: partial block");
    debug_assert_eq!(b.len(), nb * 256, "dot_q5_k: vector length mismatch");
    let mut s = 0.0f32;
    let mut buf = [0.0f32; 256];
    for (block, bchunk) in a.chunks_exact(176).zip(b.chunks(256)) {
        dq_q5_k_block(block, &mut buf);
        s += buf.iter().zip(bchunk).map(|(v, x)| v * x).sum::<f32>();
    }
    s
}

/// `dot(dequant_q6_k(a), b)` fused (210-byte blocks, signed 8-bit scales, −32 bias).
pub fn dot_q6_k(a: &[u8], b: &[f32]) -> f32 {
    let nb = a.len() / 210;
    debug_assert_eq!(a.len(), nb * 210, "dot_q6_k: partial block");
    debug_assert_eq!(b.len(), nb * 256, "dot_q6_k: vector length mismatch");
    let mut s = 0.0f32;
    let mut buf = [0.0f32; 256];
    for (block, bchunk) in a.chunks_exact(210).zip(b.chunks(256)) {
        dq_q6_k_block(block, &mut buf);
        s += buf.iter().zip(bchunk).map(|(v, x)| v * x).sum::<f32>();
    }
    s
}

macro_rules! kmatvec {
    ($name:ident, $bytes:expr, $dot:ident, $why:literal) => {
        /// Row-major `W[m×n]` stored as K-quant super-blocks, times `x[n]` → `out[m]`.
        pub fn $name(w: &[u8], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
            debug_assert_eq!(n % 256, 0, $why);
            let row_bytes = n / 256 * $bytes;
            debug_assert_eq!(
                w.len(),
                m * row_bytes,
                concat!(stringify!($name), ": weight size")
            );
            debug_assert_eq!(x.len(), n, concat!(stringify!($name), ": input width"));
            (0..m)
                .map(|r| $dot(&w[r * row_bytes..(r + 1) * row_bytes], x))
                .collect()
        }
    };
}

kmatvec!(
    matvec_q4_k,
    144,
    dot_q4_k,
    "matvec_q4_k: n must be a multiple of 256"
);
kmatvec!(
    matvec_q5_k,
    176,
    dot_q5_k,
    "matvec_q5_k: n must be a multiple of 256"
);
kmatvec!(
    matvec_q6_k,
    210,
    dot_q6_k,
    "matvec_q6_k: n must be a multiple of 256"
);

// ── `PH-203` SSE2 fused Q4_0 dot (x86_64) ────────────────────────────────────

/// Horizontal sum of a `__m128` (SSE2-only: movehl + shuffle + add_ss).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hsum_ps_sse2(v: std::arch::x86_64::__m128) -> f32 {
    use std::arch::x86_64::*;
    let t = _mm_add_ps(v, _mm_movehl_ps(v, v));
    let t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55));
    _mm_cvtss_f32(t)
}

/// SSE2 `dot(dequant_q4_0(a), b)`. Processes the split-half nibbles four elements at a
/// time (four lo + four hi lanes), scale broadcast via `_mm_set1_ps`, and reduces with
/// SSE2 shuffles. Parity with the scalar [`dot_q4_0`] oracle is asserted to 1e-3 (fused
/// accumulation order differs from the sequential reference, so bit-exactness isn't the
/// contract — the value is 4-wide throughput).
///
/// # Safety
/// Requires SSE2 (`is_x86_feature_detected!("sse2")` or an x86_64 baseline target). The
/// caller must guarantee `a.len()` is a multiple of 18 and `b.len() == a.len()/18*32`
/// (both debug-asserted); the `*_loadu_*` intrinsics read exactly `b`'s window per block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn dot_q4_0_sse2(a: &[u8], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.len() / 18;
    debug_assert_eq!(a.len(), nb * 18, "dot_q4_0_sse2: partial block");
    debug_assert_eq!(b.len(), nb * 32, "dot_q4_0_sse2: vector length mismatch");
    let mut acc_lo = _mm_setzero_ps();
    let mut acc_hi = _mm_setzero_ps();
    for (k, blk) in a.chunks_exact(18).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dv = _mm_set1_ps(d);
        let qs = &blk[2..];
        let bo = k * 32;
        for jj in (0..16).step_by(4) {
            let mut lo = [0i32; 4];
            let mut hi = [0i32; 4];
            for t in 0..4 {
                let q = qs[jj + t];
                lo[t] = (q & 0x0F) as i32 - 8;
                hi[t] = (q >> 4) as i32 - 8;
            }
            let lv = _mm_mul_ps(dv, _mm_cvtepi32_ps(_mm_loadu_si128(lo.as_ptr().cast())));
            let hv = _mm_mul_ps(dv, _mm_cvtepi32_ps(_mm_loadu_si128(hi.as_ptr().cast())));
            let bl = _mm_loadu_ps(b.as_ptr().add(bo + jj));
            let bh = _mm_loadu_ps(b.as_ptr().add(bo + 16 + jj));
            acc_lo = _mm_add_ps(acc_lo, _mm_mul_ps(lv, bl));
            acc_hi = _mm_add_ps(acc_hi, _mm_mul_ps(hv, bh));
        }
    }
    hsum_ps_sse2(acc_lo) + hsum_ps_sse2(acc_hi)
}

/// Runtime-dispatched Q4_0 dot: SSE2 when the CPU reports support, else the scalar
/// oracle. `backend::select()` reports the compiled feature tier; this adds the dynamic
/// guard so the same binary is safe on any x86_64 host.
pub fn dot_q4_0_dispatch(a: &[u8], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse2") {
            // SAFETY: gated by the runtime SSE2 check above.
            return unsafe { dot_q4_0_sse2(a, b) };
        }
    }
    dot_q4_0(a, b)
}

/// Row-major `W[m×n]` Q4_0, times `x[n]`, via the dispatched dot.
pub fn matvec_q4_0_dispatch(w: &[u8], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matvec_q4_0_dispatch: n%32");
    let row_bytes = n / 32 * 18;
    debug_assert_eq!(w.len(), m * row_bytes, "matvec_q4_0_dispatch: weight size");
    debug_assert_eq!(x.len(), n, "matvec_q4_0_dispatch: input width");
    (0..m)
        .map(|r| dot_q4_0_dispatch(&w[r * row_bytes..(r + 1) * row_bytes], x))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::dot;
    use crate::moe::matvec;
    use crate::quant::{dequant_q4_0, dequant_q5_0, dequant_q8_0};

    const SCALE_1: [u8; 2] = [0x00, 0x3C]; // f16 1.0 (LE)

    /// Build `nb` Q4_0 blocks with scale 1.0 and varied nibble bytes.
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

    /// `nb` Q5_0 blocks: d=f16 1.0, a varied `qh` 5th-bit word + varied `qs` nibbles.
    fn q5_blocks(nb: usize) -> Vec<u8> {
        let mut a = vec![0u8; nb * 22];
        for (bi, blk) in a.chunks_mut(22).enumerate() {
            blk[0] = SCALE_1[0];
            blk[1] = SCALE_1[1];
            let qh = ((bi as u32).wrapping_mul(2654435761) ^ 0x3C0A5E7F).to_le_bytes();
            blk[2..6].copy_from_slice(&qh);
            for j in 0..16 {
                blk[6 + j] = ((bi * 41 + j * 17) % 256) as u8;
            }
        }
        a
    }

    #[test]
    fn dot_q5_0_fused_matches_reference() {
        let a = q5_blocks(4);
        let b: Vec<f32> = (0..a.len() / 22 * 32)
            .map(|i| (i as f32) * 0.07 - 20.0)
            .collect();
        let fused = dot_q5_0(&a, &b);
        let reference = dot(&dequant_q5_0(&a), &b);
        assert!(
            (fused - reference).abs() < 1e-2,
            "fused={fused} ref={reference}"
        );
    }

    #[test]
    fn matvec_q5_0_matches_f32_matvec() {
        let (m, n) = (3usize, 64usize);
        let w = q5_blocks(m * n / 32);
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.15 - 3.0).collect();
        let fused = matvec_q5_0(&w, m, n, &x);
        let reference = matvec(&dequant_q5_0(&w), m, n, &x);
        assert_eq!(fused.len(), m);
        for (f, r) in fused.iter().zip(&reference) {
            assert!((f - r).abs() < 1e-2, "fused={f} ref={r}");
        }
    }

    #[test]
    fn dot_q4_0_fused_matches_reference() {
        let a = q4_blocks(4);
        let b: Vec<f32> = (0..a.len() / 18 * 32)
            .map(|i| (i as f32) * 0.1 - 30.0)
            .collect();
        let fused = dot_q4_0(&a, &b);
        let reference = dot(&dequant_q4_0(&a), &b);
        assert!(
            (fused - reference).abs() < 1e-3,
            "fused={fused} ref={reference}"
        );
    }

    #[test]
    fn dot_q4_0_dispatch_matches_scalar() {
        // Dispatch must be numerically equivalent to the scalar oracle within tolerance
        // (SSE2 path on x86_64; scalar elsewhere / no-feature fallback).
        let a = q4_blocks(6);
        let b: Vec<f32> = (0..a.len() / 18 * 32)
            .map(|i| (i as f32) * 0.07 - 21.0)
            .collect();
        let d = dot_q4_0_dispatch(&a, &b);
        let s = dot_q4_0(&a, &b);
        assert!((d - s).abs() < 1e-3, "dispatch={d} scalar={s}");
        let mv = matvec_q4_0_dispatch(&a, 3, 64, &b[..64]);
        let ms = matvec_q4_0(&a, 3, 64, &b[..64]);
        for (x, y) in mv.iter().zip(&ms) {
            assert!((x - y).abs() < 1e-3);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_q4_0_sse2_matches_scalar() {
        if !std::is_x86_feature_detected!("sse2") {
            return; // SSE2 is x86_64 baseline; skip only on exotic hosts.
        }
        let a = q4_blocks(5);
        let b: Vec<f32> = (0..a.len() / 18 * 32)
            .map(|i| (i as f32) * 0.05 - 12.0)
            .collect();
        let simd = unsafe { dot_q4_0_sse2(&a, &b) };
        let scalar = dot_q4_0(&a, &b);
        assert!((simd - scalar).abs() < 1e-3, "sse2={simd} scalar={scalar}");
    }

    #[test]
    fn dot_q8_0_fused_matches_reference() {
        let a = q8_blocks(4);
        let b: Vec<f32> = (0..a.len() / 34 * 32)
            .map(|i| (i as f32) * 0.05 - 10.0)
            .collect();
        let fused = dot_q8_0(&a, &b);
        let reference = dot(&dequant_q8_0(&a), &b);
        assert!(
            (fused - reference).abs() < 1e-2,
            "fused={fused} ref={reference}"
        );
    }

    #[test]
    fn matvec_q4_0_matches_f32_matvec() {
        let (m, n) = (3usize, 64usize); // 2 blocks/row
        let w = q4_blocks(m * n / 32);
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 - 5.0).collect();
        let fused = matvec_q4_0(&w, m, n, &x);
        let reference = matvec(&dequant_q4_0(&w), m, n, &x);
        assert_eq!(fused.len(), m);
        for (f, r) in fused.iter().zip(&reference) {
            assert!((f - r).abs() < 1e-2, "fused={f} ref={r}");
        }
    }

    #[test]
    fn matvec_q8_0_matches_f32_matvec() {
        let (m, n) = (2usize, 32usize);
        let w = q8_blocks(m * n / 32);
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 3.0).collect();
        let fused = matvec_q8_0(&w, m, n, &x);
        let reference = matvec(&dequant_q8_0(&w), m, n, &x);
        for (f, r) in fused.iter().zip(&reference) {
            assert!((f - r).abs() < 1e-2, "fused={f} ref={r}");
        }
    }

    /// Fill a super-block buffer of `block_bytes` with scale/min f16=1.0 at the front and
    /// varied data bytes so the fused and reference paths see identical bit patterns.
    fn k_blocks(kind: usize, nb: usize) -> Vec<u8> {
        let (bytes, _els) = match kind {
            0 => (144usize, 256usize),
            1 => (176, 256),
            _ => (210, 256),
        };
        let mut a = vec![0u8; nb * bytes];
        for (bi, blk) in a.chunks_mut(bytes).enumerate() {
            // d = f16 1.0 at bytes [0,1]; Q4_K/Q5_K min f16 1.0 at [2,3]; Q6_K d at tail [208,209].
            if kind != 2 {
                blk[0] = 0x00;
                blk[1] = 0x3C;
                blk[2] = 0x00;
                blk[3] = 0x3C;
            } else {
                blk[bytes - 2] = 0x00;
                blk[bytes - 1] = 0x3C;
            }
            for j in 0..bytes.saturating_sub(4) {
                blk[4 + j] = ((bi * 91 + j * 7) % 251) as u8;
            }
        }
        a
    }

    #[test]
    fn dot_q4_k_fused_matches_reference() {
        let a = k_blocks(0, 3);
        let b: Vec<f32> = (0..a.len() / 144 * 256)
            .map(|i| (i as f32) * 0.01 - 1.3)
            .collect();
        let fused = dot_q4_k(&a, &b);
        let reference = dot(&crate::quant::dequant_q4_k(&a), &b);
        assert!(
            (fused - reference).abs() <= reference.abs() * 1e-4 + 1e-4,
            "fused={fused} ref={reference}"
        );
    }

    #[test]
    fn dot_q5_k_fused_matches_reference() {
        let a = k_blocks(1, 3);
        let b: Vec<f32> = (0..a.len() / 176 * 256)
            .map(|i| (i as f32) * 0.01 - 1.3)
            .collect();
        let fused = dot_q5_k(&a, &b);
        let reference = dot(&crate::quant::dequant_q5_k(&a), &b);
        assert!(
            (fused - reference).abs() <= reference.abs() * 1e-4 + 1e-4,
            "fused={fused} ref={reference}"
        );
    }

    #[test]
    fn dot_q6_k_fused_matches_reference() {
        let a = k_blocks(2, 3);
        let b: Vec<f32> = (0..a.len() / 210 * 256)
            .map(|i| (i as f32) * 0.01 - 1.3)
            .collect();
        let fused = dot_q6_k(&a, &b);
        let reference = dot(&crate::quant::dequant_q6_k(&a), &b);
        assert!(
            (fused - reference).abs() <= reference.abs() * 1e-4 + 1e-4,
            "fused={fused} ref={reference}"
        );
    }

    #[test]
    fn matvec_k_quants_match_f32_matvec() {
        let (m, n) = (2usize, 512usize); // 2 super-blocks/row
        for kind in [0usize, 1, 2] {
            let w = k_blocks(kind, m * n / 256);
            let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002 - 1.0).collect();
            let (fused, reference): (Vec<f32>, Vec<f32>) = match kind {
                0 => (
                    matvec_q4_k(&w, m, n, &x),
                    matvec(&crate::quant::dequant_q4_k(&w), m, n, &x),
                ),
                1 => (
                    matvec_q5_k(&w, m, n, &x),
                    matvec(&crate::quant::dequant_q5_k(&w), m, n, &x),
                ),
                _ => (
                    matvec_q6_k(&w, m, n, &x),
                    matvec(&crate::quant::dequant_q6_k(&w), m, n, &x),
                ),
            };
            for (f, r) in fused.iter().zip(&reference) {
                assert!(
                    (f - r).abs() <= r.abs() * 1e-4 + 1e-4,
                    "kind={kind} fused={f} ref={r}"
                );
            }
        }
    }
}
