//! Compile-time SIMD backend selection — band B6 `PH-606`.
//!
//! One source, many targets: a single `rusty` binary exposes a *preferred* vector backend
//! chosen from the enabled Cargo features (`sse` / `avx` / `neon`) and the host arch. This
//! is the dispatch substrate the `PH-203` intrinsics plug into — today the kernels are the
//! scalar [`crate::kernel`] oracle, and `select()` always yields [`Backend::Scalar`] unless
//! a marker feature is on. No `unsafe`, no new deps: `std` + `cfg` only.

/// A quantized-kernel execution backend, weakest to strongest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Portable `std` reference (`crate::kernel`), always available.
    Scalar,
    /// x86_64 SSE2 baseline.
    Sse2,
    /// x86_64 AVX/AVX2.
    Avx2,
    /// arm64 NEON.
    Neon,
}

impl Backend {
    /// Stable lowercase name for the `--version`/metrics surface.
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Sse2 => "sse2",
            Backend::Avx2 => "avx2",
            Backend::Neon => "neon",
        }
    }
}

/// Pure backend list implied by feature flags, strongest-last: `Scalar` is always present;
/// the strongest enabled SIMD tier is appended after it. Ordering lets [`select`] just take
/// the last entry without re-scanning.
pub fn from_flags(sse: bool, avx: bool, neon: bool) -> Vec<Backend> {
    let mut v = vec![Backend::Scalar];
    if sse {
        v.push(Backend::Sse2);
    }
    if avx {
        v.push(Backend::Avx2);
    }
    if neon {
        v.push(Backend::Neon);
    }
    v
}

/// Backends available under the crate's *compiled* feature set.
pub fn compile_backends() -> Vec<Backend> {
    from_flags(
        cfg!(feature = "sse"),
        cfg!(feature = "avx"),
        cfg!(feature = "neon"),
    )
}

/// Preferred backend for the current binary = the strongest compiled one (`Scalar` when
/// no SIMD feature is on). Deterministic given the feature set.
pub fn select() -> Backend {
    let list = compile_backends();
    *list.last().unwrap_or(&Backend::Scalar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_always_first_and_last_only_when_no_flags() {
        assert_eq!(from_flags(false, false, false), vec![Backend::Scalar]);
        let only = from_flags(false, false, false);
        assert_eq!(only.first(), Some(&Backend::Scalar));
    }

    #[test]
    fn strongest_enabled_backend_is_last() {
        let all = from_flags(true, true, true);
        assert_eq!(all.first(), Some(&Backend::Scalar));
        assert_eq!(all.last(), Some(&Backend::Neon), "neon appended last");
        // A single tier: scalar + that one.
        assert_eq!(
            from_flags(true, false, false),
            vec![Backend::Scalar, Backend::Sse2]
        );
        assert_eq!(
            from_flags(false, false, true),
            vec![Backend::Scalar, Backend::Neon]
        );
    }

    #[test]
    #[cfg(not(any(feature = "sse", feature = "avx", feature = "neon")))]
    fn select_is_scalar_under_default_features() {
        // With no SIMD marker feature on, `select` resolves to the portable scalar path.
        assert_eq!(select(), Backend::Scalar);
        assert_eq!(Backend::Scalar.as_str(), "scalar");
    }

    #[test]
    #[cfg(feature = "neon")]
    fn select_prefers_neon_when_enabled() {
        assert_eq!(select(), Backend::Neon);
    }
}
