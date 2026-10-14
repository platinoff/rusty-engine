//! `rusty_engine` — pure-Rust Mixture-of-Experts inference engine (GGUF-first).
//!
//! The differentiator vs [`llama-rs`](https://github.com/platinoff) is that the
//! MoE router, per-expert FFN and quantized kernels live in **Rust**, not in a
//! C++ backend. GGUF is the first model container so the same crate cross-builds
//! to `exe` (Windows), `bin` (Linux amd64/arm64) and `apk` (Android arm64).
//!
//! Module map (see `docs/ARCHITECTURE.md`):
//! - [`core`]  — dtypes, tensor descriptors, error/result types
//! - [`gguf`]  — GGUF v3 header + tensor-info parsing (B1)
//! - [`loader`] — mmap + staged disk→RAM loaders with progress (PH-103, first non-std dep)
//! - [`moe`]   — top-k router + expert FFN (the differentiator, B4)
//! - [`infer`] — transformer forward, paged KV-cache, sampler (B3)
//! - [`forward`] — single-block transformer composition (PH-305d) + seq prefill (PH-305)
//! - [`model`] — GGUF model-config + tensor-geometry validation (PH-305c/e)
//! - [`runner`] — dense end-to-end generation on a real GGUF (PH-305)
//! - [`serve`] — OpenAI-compat HTTP server (B5)
//! - [`node`]  — slave/worker runtime over local Wi-Fi (B7/B8)
//! - [`hub`]   — GSV-native master supervisor (Rust k8s-lite, B8)

pub mod core;
pub mod forward;
pub mod gguf;
pub mod hub;
pub mod infer;
pub mod kernel;
pub mod loader;
pub mod model;
pub mod moe;
pub mod node;
pub mod quant;
pub mod runner;
pub mod serve;
pub mod tokenizer;

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
