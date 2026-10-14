//! `rusty_engine` — pure-Rust Mixture-of-Experts inference engine (GGUF-first).
//!
//! The differentiator vs [`llama-rs`](https://github.com/platinoff) is that the
//! MoE router, per-expert FFN and quantized kernels live in **Rust**, not in a
//! C++ backend. GGUF is the first model container so the same crate cross-builds
//! to `exe` (Windows), `bin` (Linux amd64/arm64) and `apk` (Android arm64).
//!
//! Module map (see `docs/ARCHITECTURE.md`):
//! - [`core`]  — dtypes, tensor descriptors, error/result types
//! - [`gguf`]  — GGUF v3 header + tensor-info parsing (mmap staged loader in B1)
//! - [`moe`]   — top-k router + expert FFN (the differentiator, B4)
//! - [`infer`] — transformer forward, paged KV-cache, sampler (B3)
//! - [`serve`] — OpenAI-compat HTTP server (B5)
//! - [`node`]  — slave/worker runtime over local Wi-Fi (B7/B8)
//! - [`hub`]   — GSV-native master supervisor (Rust k8s-lite, B8)

pub mod core;
pub mod gguf;
pub mod hub;
pub mod infer;
pub mod moe;
pub mod node;
pub mod serve;

/// Crate name as reported by `rusty --version` / `banner()`.
pub const CRATE_NAME: &str = "rusty-engine";

/// Semantic version pulled from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// One-line identity banner.
pub fn banner() -> String {
    format!("{CRATE_NAME} {VERSION} — pure-Rust MoE engine (GGUF-first)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_cargo_pkg() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn banner_includes_crate_name() {
        assert!(banner().contains(CRATE_NAME));
    }
}
