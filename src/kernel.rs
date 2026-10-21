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

/// Q8_0-quantized **activation** row (`PH-203i`): one `f32` scale per 32-value block plus
/// the `i8` values. GGML's `block_q8_0` stores the scale as f16; keeping it `f32` here means
/// the integer dot only pays one scale multiply per block and the scalar/SSE2/AVX2 paths
/// agree bit-for-bit (`f32(isum) * dw * dx`, same association everywhere).
#[derive(Clone, Debug, PartialEq)]
pub struct Q8Activation {
    pub scales: Vec<f32>,
    pub qs: Vec<i8>,
}

impl Q8Activation {
    /// Quantize a whole number of 32-value blocks: `d = max|x| / 127` (GGML `q8_0`), values
    /// `round(x / d)` clamped to `[-127, 127]`. An all-zero block gets `d = 0`, `qs = 0`.
    pub fn from_f32(x: &[f32]) -> Self {
        assert_eq!(
            x.len() % 32,
            0,
            "Q8Activation::from_f32: length must be a multiple of 32"
        );
        let mut qs = vec![0i8; x.len()];
        let mut scales = Vec::with_capacity(x.len() / 32);
        for (blk, out) in x.chunks_exact(32).zip(qs.chunks_exact_mut(32)) {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for (q, v) in out.iter_mut().zip(blk) {
                *q = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
            scales.push(d);
        }
        Self { scales, qs }
    }

    /// Number of 32-value blocks (the weight row must carry the same count).
    pub fn blocks(&self) -> usize {
        self.scales.len()
    }
}

/// Quantize a whole number of 32-value blocks for the `PH-203i` integer q8_0 dot.
pub fn quantize_q8_0(x: &[f32]) -> Q8Activation {
    Q8Activation::from_f32(x)
}

/// Scalar integer `dot(dequant_q8_0(a), dequant_q8_0(x))` — the oracle for the `PH-203i`
/// i8 kernels. `a` is whole 34-byte Q8_0 weight blocks, `x` the matching quantized
/// activation: per block the exact `i32` sum of `w_i · x_i`, then one `f32` scale step.
///
/// Measured after the exact `f16_to_f32` widening fix, the float AVX2 path is **faster**
/// (0.64 µs vs 0.74 µs per 8k-element dot): the per-block `i32` horizontal reduction costs
/// more than the i8→f32 conversion it removes. The kernel stays as a bit-exact, oracle-
/// tested alternative (i8 activation caches, future int8 matmul) and is deliberately not
/// wired into `QMat` — see `benches/quant.rs` and the handoff.
pub fn dot_q8_0_i8(a: &[u8], x: &Q8Activation) -> f32 {
    let nb = a.len() / 34;
    debug_assert_eq!(a.len(), nb * 34, "dot_q8_0_i8: partial block");
    debug_assert_eq!(
        x.qs.len(),
        nb * 32,
        "dot_q8_0_i8: activation length mismatch"
    );
    debug_assert_eq!(x.scales.len(), nb, "dot_q8_0_i8: scale count mismatch");
    let mut s = 0.0f32;
    for (blk, (qs, &dx)) in a.chunks_exact(34).zip(x.qs.chunks_exact(32).zip(&x.scales)) {
        let dw = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let mut isum = 0i32;
        for (&w, &q) in blk[2..].iter().zip(qs) {
            isum += (w as i8 as i32) * (q as i32);
        }
        s += (isum as f32) * dw * dx;
    }
    s
}

/// Row-major `W[m×n]` Q8_0 times a pre-quantized activation (`PH-203i`).
pub fn matvec_q8_0_i8(w: &[u8], m: usize, n: usize, x: &Q8Activation) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matvec_q8_0_i8: n must be a multiple of 32");
    let row_bytes = n / 32 * 34;
    debug_assert_eq!(
        w.len(),
        m * row_bytes,
        "matvec_q8_0_i8: weight size mismatch"
    );
    (0..m)
        .map(|r| dot_q8_0_i8_dispatch(&w[r * row_bytes..(r + 1) * row_bytes], x))
        .collect()
}

/// Batched Q8_0 `W[m×n] · X[rows×n]` over pre-quantized activation rows (`PH-203i`).
/// Row-wise, so it matches [`matvec_q8_0_i8`] on every output row.
pub fn matmul_q8_0_i8(w: &[u8], m: usize, n: usize, x: &[Q8Activation], rows: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * m);
    for xq in x.iter().take(rows) {
        out.extend_from_slice(&matvec_q8_0_i8(w, m, n, xq));
    }
    out
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
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: gated by the runtime AVX2+FMA check above.
            return unsafe { dot_q4_0_avx2(a, b) };
        }
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

/// SSE2 `dot(dequant_q8_0(a), b)` — `PH-203b`. Four `i8` per lane via a `[i32;4]` gather,
/// `_mm_cvtepi32_ps` × broadcast scale × `b`, SSE2 reduce. Mirrors [`dot_q8_0`] to 1e-2.
///
/// # Safety
/// Requires SSE2 (runtime-detected or x86_64 baseline). `a.len()` multiple of 34 and
/// `b.len() == a.len()/34*32` (both debug-asserted).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn dot_q8_0_sse2(a: &[u8], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.len() / 34;
    debug_assert_eq!(a.len(), nb * 34, "dot_q8_0_sse2: partial block");
    debug_assert_eq!(b.len(), nb * 32, "dot_q8_0_sse2: vector length mismatch");
    let mut acc = _mm_setzero_ps();
    for (k, blk) in a.chunks_exact(34).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dv = _mm_set1_ps(d);
        let qs = &blk[2..];
        let bo = k * 32;
        for jj in (0..32).step_by(4) {
            let mut v = [0i32; 4];
            for t in 0..4 {
                v[t] = qs[jj + t] as i8 as i32;
            }
            let vv = _mm_mul_ps(dv, _mm_cvtepi32_ps(_mm_loadu_si128(v.as_ptr().cast())));
            acc = _mm_add_ps(acc, _mm_mul_ps(vv, _mm_loadu_ps(b.as_ptr().add(bo + jj))));
        }
    }
    hsum_ps_sse2(acc)
}

/// AVX2+FMA `dot(dequant_q8_0(a), b)` — `PH-203c`. Eight `i8` per lane via
/// `_mm256_cvtepi8_epi32`, `_mm256_cvtepi32_ps` × broadcast scale × `b` fused with
/// `_mm256_fmadd_ps` into two independent accumulators (the `d` scale is constant per
/// 34-byte block, so `d·(q·b)` folds cleanly), reduced with a `_mm256` fold into SSE2's
/// `hsum_ps_sse2`.
///
/// # Safety
/// Requires AVX2+FMA (runtime-detected via [`dot_q8_0_dispatch`]). `a.len()` multiple of 34
/// and `b.len() == a.len()/34*32` (both debug-asserted).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q8_0_avx2(a: &[u8], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.len() / 34;
    debug_assert_eq!(a.len(), nb * 34, "dot_q8_0_avx2: partial block");
    debug_assert_eq!(b.len(), nb * 32, "dot_q8_0_avx2: vector length mismatch");
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    for (k, blk) in a.chunks_exact(34).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dv = _mm256_set1_ps(d);
        let qs = &blk[2..];
        let bo = k * 32;
        for jj in (0..32).step_by(16) {
            let r0 = _mm_loadl_epi64(qs.as_ptr().add(jj).cast());
            let v0 = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r0)));
            acc0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(b.as_ptr().add(bo + jj)), acc0);
            let r1 = _mm_loadl_epi64(qs.as_ptr().add(jj + 8).cast());
            let v1 = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(r1)));
            acc1 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(b.as_ptr().add(bo + jj + 8)), acc1);
        }
    }
    // Reduce 256-bit lanes to 128-bit, then SSE2 horizontal sum.
    let acc = _mm256_add_ps(acc0, acc1);
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    hsum_ps_sse2(_mm_add_ps(lo, hi))
}

/// Runtime-dispatched Q8_0 dot: AVX2 → SSE2 → scalar, in that order.
pub fn dot_q8_0_dispatch(a: &[u8], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: gated by the runtime AVX2+FMA check above.
            return unsafe { dot_q8_0_avx2(a, b) };
        }
        if std::is_x86_feature_detected!("sse2") {
            // SAFETY: gated by the runtime SSE2 check above.
            return unsafe { dot_q8_0_sse2(a, b) };
        }
    }
    dot_q8_0(a, b)
}

/// SSE2 integer `dot(dequant_q8_0(a), dequant_q8_0(x))` — `PH-203i`. Sign-extends eight
/// `i8` per `__m128i` (`unpack` with the sign mask from `pcmpgtb`) and folds pairs with
/// `_mm_madd_epi16`, so no element ever becomes `f32`. Scale is applied once per block;
/// the result is bit-identical to [`dot_q8_0_i8`].
///
/// # Safety
/// Requires SSE2 (x86_64 baseline; runtime-detected in [`dot_q8_0_i8_dispatch`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn dot_q8_0_i8_sse2(a: &[u8], x: &Q8Activation) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.len() / 34;
    debug_assert_eq!(a.len(), nb * 34, "dot_q8_0_i8_sse2: partial block");
    debug_assert_eq!(
        x.qs.len(),
        nb * 32,
        "dot_q8_0_i8_sse2: activation length mismatch"
    );
    let mut s = 0.0f32;
    let zero = _mm_setzero_si128();
    for (k, blk) in a.chunks_exact(34).enumerate() {
        let dw = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dx = x.scales[k];
        let mut acc = _mm_setzero_si128();
        for jj in (0..32).step_by(16) {
            let w8 = _mm_loadu_si128(blk.as_ptr().add(2 + jj).cast());
            let q8 = _mm_loadu_si128(x.qs.as_ptr().add(k * 32 + jj).cast());
            let ws = _mm_cmpgt_epi8(zero, w8);
            let qs = _mm_cmpgt_epi8(zero, q8);
            let prod = _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi8(w8, ws), _mm_unpacklo_epi8(q8, qs)),
                _mm_madd_epi16(_mm_unpackhi_epi8(w8, ws), _mm_unpackhi_epi8(q8, qs)),
            );
            acc = _mm_add_epi32(acc, prod);
        }
        let t = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0b10_11_00_01));
        let t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0b01_00_11_10));
        let isum = _mm_cvtsi128_si32(t);
        s += (isum as f32) * dw * dx;
    }
    s
}

/// AVX2 integer `dot(dequant_q8_0(a), dequant_q8_0(x))` — `PH-203i`. Sixteen `i8` per
/// `__m256i` (`_mm256_cvtepi8_epi16`), `_mm256_madd_epi16` pairs into eight `i32` lanes.
/// One scale step per block; bit-identical to [`dot_q8_0_i8`].
///
/// # Safety
/// Requires AVX2 (runtime-detected in [`dot_q8_0_i8_dispatch`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q8_0_i8_avx2(a: &[u8], x: &Q8Activation) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.len() / 34;
    debug_assert_eq!(a.len(), nb * 34, "dot_q8_0_i8_avx2: partial block");
    debug_assert_eq!(
        x.qs.len(),
        nb * 32,
        "dot_q8_0_i8_avx2: activation length mismatch"
    );
    let mut s = 0.0f32;
    for (k, blk) in a.chunks_exact(34).enumerate() {
        let dw = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dx = x.scales[k];
        let mut acc = _mm256_setzero_si256();
        for jj in (0..32).step_by(16) {
            let w8 = _mm_loadu_si128(blk.as_ptr().add(2 + jj).cast());
            let q8 = _mm_loadu_si128(x.qs.as_ptr().add(k * 32 + jj).cast());
            let prod = _mm256_madd_epi16(_mm256_cvtepi8_epi16(w8), _mm256_cvtepi8_epi16(q8));
            acc = _mm256_add_epi32(acc, prod);
        }
        let t = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256(acc, 1),
        );
        let t = _mm_hadd_epi32(t, t);
        let t = _mm_hadd_epi32(t, t);
        let isum = _mm_cvtsi128_si32(t);
        s += (isum as f32) * dw * dx;
    }
    s
}

/// Runtime-dispatched integer Q8_0 dot: AVX2 → SSE2 → scalar.
pub fn dot_q8_0_i8_dispatch(a: &[u8], x: &Q8Activation) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: gated by the runtime AVX2 check above.
            return unsafe { dot_q8_0_i8_avx2(a, x) };
        }
        if std::is_x86_feature_detected!("sse2") {
            // SAFETY: gated by the runtime SSE2 check above.
            return unsafe { dot_q8_0_i8_sse2(a, x) };
        }
    }
    dot_q8_0_i8(a, x)
}

/// Row-major `W[m×n]` Q8_0, times `x[n]`, via the dispatched dot.
pub fn matvec_q8_0_dispatch(w: &[u8], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matvec_q8_0_dispatch: n%32");
    let row_bytes = n / 32 * 34;
    debug_assert_eq!(w.len(), m * row_bytes, "matvec_q8_0_dispatch: weight size");
    debug_assert_eq!(x.len(), n, "matvec_q8_0_dispatch: input width");
    (0..m)
        .map(|r| dot_q8_0_dispatch(&w[r * row_bytes..(r + 1) * row_bytes], x))
        .collect()
}

// ── `PH-203c` AVX2 fused dots (x86_64) ───────────────────────────────────────

/// AVX2+FMA `dot(dequant_q4_0(a), b)` — split-half nibbles eight at a time (`lo` → `b[0..16]`,
/// `hi` → `b[16..32]`) into two independent accumulators via `_mm256_fmadd_ps`, reduced with a
/// `__m256` fold into SSE2's `hsum_ps_sse2`.
///
/// # Safety
/// Requires AVX2+FMA (runtime-detected via [`dot_q4_0_dispatch`]). `a.len()` multiple of 18 and
/// `b.len() == a.len()/18*32` (both debug-asserted).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_0_avx2(a: &[u8], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.len() / 18;
    debug_assert_eq!(a.len(), nb * 18, "dot_q4_0_avx2: partial block");
    debug_assert_eq!(b.len(), nb * 32, "dot_q4_0_avx2: vector length mismatch");
    let mask = _mm256_set1_epi32(0x0F);
    let eight = _mm256_set1_epi32(8);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    for (k, blk) in a.chunks_exact(18).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dv = _mm256_set1_ps(d);
        let qs = &blk[2..];
        let bo = k * 32;
        for jj in (0..16).step_by(8) {
            let raw = _mm_loadl_epi64(qs.as_ptr().add(jj).cast());
            let v = _mm256_cvtepu8_epi32(raw);
            let lo = _mm256_sub_epi32(_mm256_and_si256(v, mask), eight);
            let hi = _mm256_sub_epi32(_mm256_srli_epi32(v, 4), eight);
            let lov = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(lo));
            let hiv = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(hi));
            acc0 = _mm256_fmadd_ps(lov, _mm256_loadu_ps(b.as_ptr().add(bo + jj)), acc0);
            acc1 = _mm256_fmadd_ps(hiv, _mm256_loadu_ps(b.as_ptr().add(bo + 16 + jj)), acc1);
        }
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    hsum_ps_sse2(_mm_add_ps(lo, hi))
}

/// AVX2 `dot(dequant_q4_k(a), b)` — per 64-element group the low/high nibbles are widened
/// eight at a time and scaled by the group's `d·sc` / `min·m` (`get_scale_min_k4`), then
/// folded into the dot. Mirrors [`dot_q4_k`] to the shared 1e-4 tolerance.
///
/// # Safety
/// Requires AVX2 (runtime-detected via [`dot_q4_k_dispatch`]). `a.len()` multiple of 144 and
/// `b.len() == a.len()/144*256` (both debug-asserted).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q4_k_avx2(a: &[u8], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.len() / 144;
    debug_assert_eq!(a.len(), nb * 144, "dot_q4_k_avx2: partial block");
    debug_assert_eq!(b.len(), nb * 256, "dot_q4_k_avx2: vector length mismatch");
    let mask = _mm256_set1_epi32(0x0F);
    let mut acc = _mm256_setzero_ps();
    for (k, blk) in a.chunks_exact(144).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let min = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let bo = k * 256;
        for grp in 0..4 {
            let (sc0, m0) = get_scale_min_k4(grp * 2, scales);
            let (sc1, m1) = get_scale_min_k4(grp * 2 + 1, scales);
            let d1 = _mm256_set1_ps(d * sc0 as f32);
            let dm1 = _mm256_set1_ps(min * m0 as f32);
            let d2 = _mm256_set1_ps(d * sc1 as f32);
            let dm2 = _mm256_set1_ps(min * m1 as f32);
            let base = grp * 32;
            let ob = bo + grp * 64;
            for jj in (0..32).step_by(8) {
                let raw = _mm_loadl_epi64(qs.as_ptr().add(base + jj).cast());
                let v = _mm256_cvtepu8_epi32(raw);
                let lo = _mm256_cvtepi32_ps(_mm256_and_si256(v, mask));
                let hi = _mm256_cvtepi32_ps(_mm256_srli_epi32(v, 4));
                let lov = _mm256_sub_ps(_mm256_mul_ps(lo, d1), dm1);
                let hiv = _mm256_sub_ps(_mm256_mul_ps(hi, d2), dm2);
                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(lov, _mm256_loadu_ps(b.as_ptr().add(ob + jj))),
                );
                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(hiv, _mm256_loadu_ps(b.as_ptr().add(ob + 32 + jj))),
                );
            }
        }
    }
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    hsum_ps_sse2(_mm_add_ps(lo, hi))
}

/// AVX2 `dot(dequant_q6_k(a), b)` — 6-bit values (−32 bias) packed low-nibble + 2 high bits,
/// with signed 8-bit sub-block scales, widened eight at a time per half-block.
///
/// # Safety
/// Requires AVX2 (runtime-detected via [`dot_q6_k_dispatch`]). `a.len()` multiple of 210 and
/// `b.len() == a.len()/210*256` (both debug-asserted).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_q6_k_avx2(a: &[u8], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.len() / 210;
    debug_assert_eq!(a.len(), nb * 210, "dot_q6_k_avx2: partial block");
    debug_assert_eq!(b.len(), nb * 256, "dot_q6_k_avx2: vector length mismatch");
    let mask = _mm256_set1_epi32(0x0F);
    let m2 = _mm256_set1_epi32(0x03);
    let bias = _mm256_set1_epi32(32);
    let mut acc = _mm256_setzero_ps();
    for (k, blk) in a.chunks_exact(210).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let bo = k * 256;
        for half in 0..2 {
            let qb = half * 64;
            let hb = half * 32;
            let sb = half * 8;
            let ob = bo + half * 128;
            for l in (0..32).step_by(8) {
                let is = l / 16;
                let la = _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql.as_ptr().add(qb + l).cast()));
                let lb = _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql.as_ptr().add(qb + l + 32).cast()));
                let h = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qh.as_ptr().add(hb + l).cast()));
                let q1 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_and_si256(la, mask),
                        _mm256_slli_epi32(_mm256_and_si256(h, m2), 4),
                    ),
                    bias,
                );
                let q2 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_and_si256(lb, mask),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 2), m2), 4),
                    ),
                    bias,
                );
                let q3 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_srli_epi32(la, 4),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 4), m2), 4),
                    ),
                    bias,
                );
                let q4 = _mm256_sub_epi32(
                    _mm256_or_si256(
                        _mm256_srli_epi32(lb, 4),
                        _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 6), m2), 4),
                    ),
                    bias,
                );
                let s0 = _mm256_set1_ps(d * sc[sb + is] as i8 as f32);
                let s2 = _mm256_set1_ps(d * sc[sb + is + 2] as i8 as f32);
                let s4 = _mm256_set1_ps(d * sc[sb + is + 4] as i8 as f32);
                let s6 = _mm256_set1_ps(d * sc[sb + is + 6] as i8 as f32);
                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(q1), s0),
                        _mm256_loadu_ps(b.as_ptr().add(ob + l)),
                    ),
                );
                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(q2), s2),
                        _mm256_loadu_ps(b.as_ptr().add(ob + l + 32)),
                    ),
                );
                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(q3), s4),
                        _mm256_loadu_ps(b.as_ptr().add(ob + l + 64)),
                    ),
                );
                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(q4), s6),
                        _mm256_loadu_ps(b.as_ptr().add(ob + l + 96)),
                    ),
                );
            }
        }
    }
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    hsum_ps_sse2(_mm_add_ps(lo, hi))
}

/// AVX2+FMA `dot(dequant_q5_0(a), b)` — split-half nibbles eight at a time; the 5th bit is
/// gathered from `qh` per lane with `_mm256_srlv_epi32` (bias −16); the two halves fold into
/// independent accumulators via `_mm256_fmadd_ps`.
///
/// # Safety
/// Requires AVX2+FMA (runtime-detected via [`dot_q5_0_dispatch`]). `a.len()` multiple of 22 and
/// `b.len() == a.len()/22*32` (both debug-asserted).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q5_0_avx2(a: &[u8], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let nb = a.len() / 22;
    debug_assert_eq!(a.len(), nb * 22, "dot_q5_0_avx2: partial block");
    debug_assert_eq!(b.len(), nb * 32, "dot_q5_0_avx2: vector length mismatch");
    let mask = _mm256_set1_epi32(0x0F);
    let sixteen = _mm256_set1_epi32(16);
    let ones = _mm256_set1_epi32(1);
    let shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    for (k, blk) in a.chunks_exact(22).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dv = _mm256_set1_ps(d);
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        let bo = k * 32;
        for jj in (0..16).step_by(8) {
            let v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs.as_ptr().add(jj).cast()));
            let lo_nib = _mm256_and_si256(v, mask);
            let hi_nib = _mm256_srli_epi32(v, 4);
            let b0 = _mm256_set1_epi32(((qh >> jj) & 0xFF) as i32);
            let b1 = _mm256_set1_epi32(((qh >> (jj + 16)) & 0xFF) as i32);
            let bit0 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(b0, shifts), ones), 4);
            let bit1 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(b1, shifts), ones), 4);
            let lo = _mm256_sub_epi32(_mm256_or_si256(lo_nib, bit0), sixteen);
            let hi = _mm256_sub_epi32(_mm256_or_si256(hi_nib, bit1), sixteen);
            let lov = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(lo));
            let hiv = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(hi));
            acc0 = _mm256_fmadd_ps(lov, _mm256_loadu_ps(b.as_ptr().add(bo + jj)), acc0);
            acc1 = _mm256_fmadd_ps(hiv, _mm256_loadu_ps(b.as_ptr().add(bo + 16 + jj)), acc1);
        }
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    hsum_ps_sse2(_mm_add_ps(lo, hi))
}

/// Runtime-dispatched Q5_0 dot: AVX2 when the CPU reports support, else the scalar oracle.
pub fn dot_q5_0_dispatch(a: &[u8], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: gated by the runtime AVX2+FMA check above.
            return unsafe { dot_q5_0_avx2(a, b) };
        }
    }
    dot_q5_0(a, b)
}

/// Row-major `W[m×n]` Q5_0, times `x[n]`, via the dispatched dot.
pub fn matvec_q5_0_dispatch(w: &[u8], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matvec_q5_0_dispatch: n%32");
    let row_bytes = n / 32 * 22;
    debug_assert_eq!(w.len(), m * row_bytes, "matvec_q5_0_dispatch: weight size");
    debug_assert_eq!(x.len(), n, "matvec_q5_0_dispatch: input width");
    (0..m)
        .map(|r| dot_q5_0_dispatch(&w[r * row_bytes..(r + 1) * row_bytes], x))
        .collect()
}

/// Runtime-dispatched Q4_K dot: AVX2 when the CPU reports support, else the scalar oracle.
pub fn dot_q4_k_dispatch(a: &[u8], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: gated by the runtime AVX2 check above.
            return unsafe { dot_q4_k_avx2(a, b) };
        }
    }
    dot_q4_k(a, b)
}

/// Runtime-dispatched Q6_K dot: AVX2 when the CPU reports support, else the scalar oracle.
pub fn dot_q6_k_dispatch(a: &[u8], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: gated by the runtime AVX2 check above.
            return unsafe { dot_q6_k_avx2(a, b) };
        }
    }
    dot_q6_k(a, b)
}

/// Row-major `W[m×n]` Q4_K, times `x[n]`, via the dispatched dot.
pub fn matvec_q4_k_dispatch(w: &[u8], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(n % 256, 0, "matvec_q4_k_dispatch: n%256");
    let row_bytes = n / 256 * 144;
    debug_assert_eq!(w.len(), m * row_bytes, "matvec_q4_k_dispatch: weight size");
    debug_assert_eq!(x.len(), n, "matvec_q4_k_dispatch: input width");
    (0..m)
        .map(|r| dot_q4_k_dispatch(&w[r * row_bytes..(r + 1) * row_bytes], x))
        .collect()
}

/// Row-major `W[m×n]` Q6_K, times `x[n]`, via the dispatched dot.
pub fn matvec_q6_k_dispatch(w: &[u8], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(n % 256, 0, "matvec_q6_k_dispatch: n%256");
    let row_bytes = n / 256 * 210;
    debug_assert_eq!(w.len(), m * row_bytes, "matvec_q6_k_dispatch: weight size");
    debug_assert_eq!(x.len(), n, "matvec_q6_k_dispatch: input width");
    (0..m)
        .map(|r| dot_q6_k_dispatch(&w[r * row_bytes..(r + 1) * row_bytes], x))
        .collect()
}

// ── `PH-305` prefill: batched matmul with per-block weight reuse ─────────────
//
// Row-wise `matvec` re-reads and re-dequantizes a whole weight row for every input row,
// so a `seq`-token prefill pays `seq×` the dequant work — and the int→float conversion is
// what dominates the quant kernels. The `matmul_*` variants iterate weight rows outer and
// input rows inner, dequantizing each block **once** into a small f32 buffer reused across
// all rows. Memory stays bounded (one block buffer); only pure-f32 MACs scale with `seq`.

/// Dequantize one 18-byte Q4_0 block into 32 floats (split-half nibbles, bias −8).
fn dq_q4_0_block(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let qs = &b[2..18];
    for j in 0..16 {
        out[j] = d * ((qs[j] & 0x0F) as i32 - 8) as f32;
        out[16 + j] = d * ((qs[j] >> 4) as i32 - 8) as f32;
    }
}

/// Dequantize one 22-byte Q5_0 block into 32 floats (5th bit from `qh`, bias −16).
fn dq_q5_0_block(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..22];
    for j in 0..16 {
        let bit0 = (((qh >> j) & 1) << 4) as i32;
        let bit1 = (((qh >> (j + 16)) & 1) << 4) as i32;
        out[j] = d * (((qs[j] & 0x0F) as i32 | bit0) - 16) as f32;
        out[16 + j] = d * (((qs[j] >> 4) as i32 | bit1) - 16) as f32;
    }
}

/// Dequantize one 34-byte Q8_0 block into 32 floats (`d * i8`).
fn dq_q8_0_block(b: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    for j in 0..32 {
        out[j] = (b[2 + j] as i8) as f32 * d;
    }
}

/// f32 dot, AVX2+FMA when available (a genuine win for pure f32 MACs).
///
/// # Safety
/// Requires AVX2+FMA (gated by [`dot_f32_dispatch`]); `a` and `b` must be equal length
/// (debug-asserted) — the `*_loadu_ps` intrinsics read only within that window.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(a.len(), b.len(), "dot_f32_avx2: length mismatch");
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc,
        );
        i += 8;
    }
    let mut s = hsum_ps_sse2(_mm_add_ps(
        _mm256_castps256_ps128(acc),
        _mm256_extractf128_ps(acc, 1),
    ));
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Runtime-dispatched f32 dot: AVX2+FMA when the CPU reports support, else a scalar fold.
fn dot_f32_dispatch(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: gated by the runtime AVX2+FMA check above.
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Batched `W[m×n] · X[rows×n] → Out[rows×m]`, dequantizing each `block_bytes` block once
/// (via `dq`) and reusing it across every row.
#[allow(clippy::too_many_arguments)]
fn matmul_reuse(
    w: &[u8],
    m: usize,
    n: usize,
    x: &[f32],
    rows: usize,
    block_bytes: usize,
    block_vals: usize,
    dq: impl Fn(&[u8], &mut [f32]),
) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * n, "matmul: X rows × n");
    let blocks = n / block_vals;
    let row_bytes = blocks * block_bytes;
    debug_assert_eq!(w.len(), m * row_bytes, "matmul: weight size");
    let mut out = vec![0.0f32; rows * m];
    let mut buf = vec![0.0f32; block_vals];
    let mut acc = vec![0.0f32; rows];
    for r in 0..m {
        acc.iter_mut().for_each(|a| *a = 0.0);
        let wr = &w[r * row_bytes..(r + 1) * row_bytes];
        for b in 0..blocks {
            dq(&wr[b * block_bytes..(b + 1) * block_bytes], &mut buf);
            let xoff = b * block_vals;
            for (row, a) in acc.iter_mut().enumerate() {
                let xb = &x[row * n + xoff..row * n + xoff + block_vals];
                *a += dot_f32_dispatch(&buf, xb);
            }
        }
        for (row, a) in acc.iter().enumerate() {
            out[row * m + r] = *a;
        }
    }
    out
}

/// Batched Q4_0 `W[m×n] · X[rows×n]` (see [`matmul_reuse`]).
pub fn matmul_q4_0_dispatch(w: &[u8], m: usize, n: usize, x: &[f32], rows: usize) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matmul_q4_0_dispatch: n%32");
    matmul_reuse(w, m, n, x, rows, 18, 32, dq_q4_0_block)
}

/// Batched Q5_0 `W[m×n] · X[rows×n]` (see [`matmul_reuse`]).
pub fn matmul_q5_0_dispatch(w: &[u8], m: usize, n: usize, x: &[f32], rows: usize) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matmul_q5_0_dispatch: n%32");
    matmul_reuse(w, m, n, x, rows, 22, 32, dq_q5_0_block)
}

/// Batched Q8_0 `W[m×n] · X[rows×n]` (see [`matmul_reuse`]).
pub fn matmul_q8_0_dispatch(w: &[u8], m: usize, n: usize, x: &[f32], rows: usize) -> Vec<f32> {
    debug_assert_eq!(n % 32, 0, "matmul_q8_0_dispatch: n%32");
    matmul_reuse(w, m, n, x, rows, 34, 32, dq_q8_0_block)
}

/// Batched Q4_K `W[m×n] · X[rows×n]` (see [`matmul_reuse`]).
pub fn matmul_q4_k_dispatch(w: &[u8], m: usize, n: usize, x: &[f32], rows: usize) -> Vec<f32> {
    debug_assert_eq!(n % 256, 0, "matmul_q4_k_dispatch: n%256");
    matmul_reuse(w, m, n, x, rows, 144, 256, |b, o| {
        dq_q4_k_block(b, o.try_into().expect("q4_k block buffer is 256"))
    })
}

/// Batched Q5_K `W[m×n] · X[rows×n]` (see [`matmul_reuse`]).
pub fn matmul_q5_k(w: &[u8], m: usize, n: usize, x: &[f32], rows: usize) -> Vec<f32> {
    debug_assert_eq!(n % 256, 0, "matmul_q5_k: n%256");
    matmul_reuse(w, m, n, x, rows, 176, 256, |b, o| {
        dq_q5_k_block(b, o.try_into().expect("q5_k block buffer is 256"))
    })
}

/// Batched Q6_K `W[m×n] · X[rows×n]` (see [`matmul_reuse`]).
pub fn matmul_q6_k_dispatch(w: &[u8], m: usize, n: usize, x: &[f32], rows: usize) -> Vec<f32> {
    debug_assert_eq!(n % 256, 0, "matmul_q6_k_dispatch: n%256");
    matmul_reuse(w, m, n, x, rows, 210, 256, |b, o| {
        dq_q6_k_block(b, o.try_into().expect("q6_k block buffer is 256"))
    })
}

/// Bytes of one `n`-wide row for a fused kernel type, or `None` when the type has no
/// fused kernel (caller falls back to dequant + f32 matvec).
pub fn quant_row_bytes(ggml_type: u32, n: usize) -> Option<usize> {
    match ggml_type {
        2 => Some(n / 32 * 18),
        6 => Some(n / 32 * 22),
        8 => Some(n / 32 * 34),
        12 => Some(n / 256 * 144),
        13 => Some(n / 256 * 176),
        14 => Some(n / 256 * 210),
        _ => None,
    }
}

/// `dot(row, x)` for a single stored quantized row — the body every `matvec_*` repeats.
/// Lets a caller drive row-parallel matvec/matmul without re-deriving per-type geometry.
/// Panics when `ggml_type` has no fused kernel (guard with [`quant_row_bytes`]).
pub fn quant_row_dot(ggml_type: u32, row: &[u8], x: &[f32]) -> f32 {
    match ggml_type {
        2 => dot_q4_0_dispatch(row, x),
        6 => dot_q5_0_dispatch(row, x),
        8 => dot_q8_0_dispatch(row, x),
        12 => dot_q4_k_dispatch(row, x),
        13 => dot_q5_k(row, x),
        14 => dot_q6_k_dispatch(row, x),
        t => panic!("quant_row_dot: no fused kernel for ggml type {t}"),
    }
}

/// Batched dispatch `W[m×n] · X[rows×n]` for a fused kernel type.
/// Panics when `ggml_type` has no fused kernel (guard with [`quant_row_bytes`]).
pub fn matmul_dispatch(
    ggml_type: u32,
    w: &[u8],
    m: usize,
    n: usize,
    x: &[f32],
    rows: usize,
) -> Vec<f32> {
    match ggml_type {
        2 => matmul_q4_0_dispatch(w, m, n, x, rows),
        6 => matmul_q5_0_dispatch(w, m, n, x, rows),
        8 => matmul_q8_0_dispatch(w, m, n, x, rows),
        12 => matmul_q4_k_dispatch(w, m, n, x, rows),
        13 => matmul_q5_k(w, m, n, x, rows),
        14 => matmul_q6_k_dispatch(w, m, n, x, rows),
        t => panic!("matmul_dispatch: no fused kernel for ggml type {t}"),
    }
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
    fn dot_q8_0_dispatch_matches_scalar() {
        let a = q8_blocks(6);
        let b: Vec<f32> = (0..a.len() / 34 * 32)
            .map(|i| (i as f32) * 0.03 - 9.0)
            .collect();
        let d = dot_q8_0_dispatch(&a, &b);
        let s = dot_q8_0(&a, &b);
        assert!((d - s).abs() < 1e-2, "dispatch={d} scalar={s}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_q8_0_sse2_matches_scalar() {
        if !std::is_x86_feature_detected!("sse2") {
            return;
        }
        let a = q8_blocks(5);
        let b: Vec<f32> = (0..a.len() / 34 * 32)
            .map(|i| (i as f32) * 0.04 - 11.0)
            .collect();
        let simd = unsafe { dot_q8_0_sse2(&a, &b) };
        let scalar = dot_q8_0(&a, &b);
        assert!((simd - scalar).abs() < 1e-2, "sse2={simd} scalar={scalar}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_q8_0_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            eprintln!("PH-203c: AVX2+FMA absent on this host; skipping explicit avx2 check");
            return;
        }
        let a = q8_blocks(5);
        let b: Vec<f32> = (0..a.len() / 34 * 32)
            .map(|i| (i as f32) * 0.045 - 13.0)
            .collect();
        let simd = unsafe { dot_q8_0_avx2(&a, &b) };
        let scalar = dot_q8_0(&a, &b);
        assert!((simd - scalar).abs() < 1e-2, "avx2={simd} scalar={scalar}");
    }

    fn q8_test_vector(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.037 - 7.0).collect()
    }

    /// Dequantize a [`Q8Activation`] back to f32 (`q · d`) — the exact reference the integer
    /// dot must reproduce (it isolates the kernel from the quantization loss itself).
    fn dequant_activation(xq: &Q8Activation) -> Vec<f32> {
        xq.qs
            .iter()
            .enumerate()
            .map(|(i, q)| (*q as f32) * xq.scales[i / 32])
            .collect()
    }

    #[test]
    fn q8_activation_roundtrip_error_is_bounded() {
        let x = q8_test_vector(96);
        let xq = Q8Activation::from_f32(&x);
        assert_eq!(xq.blocks(), 3);
        assert_eq!(xq.qs.len(), x.len());
        for ((blk, qb), &d) in x
            .chunks_exact(32)
            .zip(xq.qs.chunks_exact(32))
            .zip(&xq.scales)
        {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                (d - amax / 127.0).abs() < 1e-6,
                "scale {d} vs {}",
                amax / 127.0
            );
            for (v, q) in blk.iter().zip(qb) {
                assert!(((*q as f32) * d - v).abs() <= d / 2.0 + 1e-6);
            }
        }
    }

    #[test]
    fn q8_activation_zero_block_stays_zero() {
        let xq = Q8Activation::from_f32(&vec![0.0f32; 64]);
        assert_eq!(xq.scales, vec![0.0, 0.0]);
        assert!(xq.qs.iter().all(|&q| q == 0));
    }

    #[test]
    fn dot_q8_0_i8_matches_dequantized_activation_dot() {
        let a = q8_blocks(6);
        let xq = Q8Activation::from_f32(&q8_test_vector(a.len() / 34 * 32));
        let x = dequant_activation(&xq);
        let int = dot_q8_0_i8(&a, &xq);
        let float = dot_q8_0(&a, &x);
        let rel = (int - float).abs() / float.abs().max(1.0);
        assert!(rel < 1e-3, "int={int} float={float} rel={rel}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_q8_0_i8_sse2_is_bit_exact_with_scalar() {
        if !std::is_x86_feature_detected!("sse2") {
            return;
        }
        let a = q8_blocks(7);
        let xq = Q8Activation::from_f32(&q8_test_vector(a.len() / 34 * 32));
        let simd = unsafe { dot_q8_0_i8_sse2(&a, &xq) };
        let scalar = dot_q8_0_i8(&a, &xq);
        assert_eq!(
            simd.to_bits(),
            scalar.to_bits(),
            "sse2={simd} scalar={scalar}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_q8_0_i8_avx2_is_bit_exact_with_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("PH-203i: AVX2 absent on this host; skipping explicit avx2 check");
            return;
        }
        let a = q8_blocks(7);
        let xq = Q8Activation::from_f32(&q8_test_vector(a.len() / 34 * 32));
        let simd = unsafe { dot_q8_0_i8_avx2(&a, &xq) };
        let scalar = dot_q8_0_i8(&a, &xq);
        assert_eq!(
            simd.to_bits(),
            scalar.to_bits(),
            "avx2={simd} scalar={scalar}"
        );
    }

    #[test]
    fn dot_q8_0_i8_dispatch_is_bit_exact_with_scalar() {
        let a = q8_blocks(9);
        let xq = Q8Activation::from_f32(&q8_test_vector(a.len() / 34 * 32));
        let dispatch = dot_q8_0_i8_dispatch(&a, &xq);
        let scalar = dot_q8_0_i8(&a, &xq);
        assert_eq!(
            dispatch.to_bits(),
            scalar.to_bits(),
            "dispatch={dispatch} scalar={scalar}"
        );
    }

    #[test]
    fn matvec_q8_0_i8_matches_float_matvec() {
        let (m, n) = (3usize, 64usize);
        let w = q8_blocks(m * n / 32);
        let xq = Q8Activation::from_f32(&q8_test_vector(n));
        let x = dequant_activation(&xq);
        let int = matvec_q8_0_i8(&w, m, n, &xq);
        let float = matvec_q8_0(&w, m, n, &x);
        assert_eq!(int.len(), m);
        for (i, f) in int.iter().zip(&float) {
            let rel = (i - f).abs() / f.abs().max(1.0);
            assert!(rel < 1e-3, "int={i} float={f} rel={rel}");
        }
    }

    #[test]
    fn matmul_q8_0_i8_matches_rowwise_matvec() {
        let (m, n, rows) = (3usize, 64usize, 2usize);
        let w = q8_blocks(m * n / 32);
        let x = q8_test_vector(rows * n);
        let xq: Vec<_> = x.chunks_exact(n).map(Q8Activation::from_f32).collect();
        let got = matmul_q8_0_i8(&w, m, n, &xq, rows);
        let want: Vec<f32> = xq
            .iter()
            .flat_map(|x| matvec_q8_0_i8(&w, m, n, x))
            .collect();
        assert_eq!(got, want);
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

    #[test]
    fn dot_q4_0_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let a = q4_blocks(4);
        let b: Vec<f32> = (0..a.len() / 18 * 32)
            .map(|i| (i as f32) * 0.03 - 1.1)
            .collect();
        let want = dot_q4_0(&a, &b);
        let got = unsafe { dot_q4_0_avx2(&a, &b) };
        assert!(
            (got - want).abs() <= want.abs() * 1e-3 + 1e-3,
            "got={got} want={want}"
        );
    }

    #[test]
    fn dot_q5_0_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let a = q5_blocks(4);
        let b: Vec<f32> = (0..a.len() / 22 * 32)
            .map(|i| (i as f32) * 0.02 - 1.2)
            .collect();
        let want = dot_q5_0(&a, &b);
        let got = unsafe { dot_q5_0_avx2(&a, &b) };
        assert!(
            (got - want).abs() <= want.abs() * 1e-3 + 1e-3,
            "got={got} want={want}"
        );
    }

    #[test]
    fn dot_q4_k_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let a = k_blocks(0, 3);
        let b: Vec<f32> = (0..a.len() / 144 * 256)
            .map(|i| (i as f32) * 0.01 - 1.3)
            .collect();
        let want = dot_q4_k(&a, &b);
        let got = unsafe { dot_q4_k_avx2(&a, &b) };
        assert!(
            (got - want).abs() <= want.abs() * 1e-4 + 1e-4,
            "got={got} want={want}"
        );
    }

    #[test]
    fn dot_q6_k_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let a = k_blocks(2, 3);
        let b: Vec<f32> = (0..a.len() / 210 * 256)
            .map(|i| (i as f32) * 0.01 - 1.3)
            .collect();
        let want = dot_q6_k(&a, &b);
        let got = unsafe { dot_q6_k_avx2(&a, &b) };
        assert!(
            (got - want).abs() <= want.abs() * 1e-4 + 1e-4,
            "got={got} want={want}"
        );
    }

    #[test]
    fn matmul_matches_rowwise_matvec() {
        let (m, n, rows) = (5usize, 64usize, 3usize);
        let x: Vec<f32> = (0..rows * n)
            .map(|i| ((i as i32 * 7 % 13) as f32) * 0.1 - 0.6)
            .collect();
        let close = |a: &[f32], b: &[f32]| {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(p, q)| (p - q).abs() <= 1e-2 * (1.0 + p.abs()))
        };
        let rowwise = |w: &[u8], mv: fn(&[u8], usize, usize, &[f32]) -> Vec<f32>| -> Vec<f32> {
            (0..rows)
                .flat_map(|r| mv(w, m, n, &x[r * n..(r + 1) * n]))
                .collect()
        };

        let q4 = q4_blocks(m * n / 32);
        assert!(close(
            &matmul_q4_0_dispatch(&q4, m, n, &x, rows),
            &rowwise(&q4, matvec_q4_0_dispatch)
        ));

        let q5 = q5_blocks(m * n / 32);
        assert!(close(
            &matmul_q5_0_dispatch(&q5, m, n, &x, rows),
            &rowwise(&q5, matvec_q5_0_dispatch)
        ));

        let q8 = q8_blocks(m * n / 32);
        assert!(close(
            &matmul_q8_0_dispatch(&q8, m, n, &x, rows),
            &rowwise(&q8, matvec_q8_0_dispatch)
        ));

        // K-quants need n % 256, so use a 256-wide row.
        let (km, kn) = (3usize, 256usize);
        let kx: Vec<f32> = (0..rows * kn)
            .map(|i| ((i as i32 * 5 % 11) as f32) * 0.1 - 0.5)
            .collect();
        let krowwise = |w: &[u8], mv: fn(&[u8], usize, usize, &[f32]) -> Vec<f32>| -> Vec<f32> {
            (0..rows)
                .flat_map(|r| mv(w, km, kn, &kx[r * kn..(r + 1) * kn]))
                .collect()
        };
        let q6 = k_blocks(2, km * kn / 256);
        assert!(close(
            &matmul_q6_k_dispatch(&q6, km, kn, &kx, rows),
            &krowwise(&q6, matvec_q6_k_dispatch)
        ));
    }
}
