//! `nanojev` — Rust-native port of the open NanoJev decision model
//! (`C-Tianyu/NanoJev`, unified-games-v1 step-400, `Qwen/Qwen3-0.6B@c1899de…`
//! backbone + attention set-head). Reads the checkpoint bundle directly:
//!
//! - `best.safetensors` — F32 weights (2.39 GB, `f68c47d6…`), mmap'd, zero-copy;
//! - `backbone_config/config.json` — Qwen3 arch (28 × full attention, GQA 16/8,
//!   head_dim 128, RMSNorm 1e-6, RoPE θ=1e6, SwiGLU 3072);
//! - `tokenizer/tokenizer.json` — HF byte-level BPE, Qwen2 splitter (`pre=qwen2`);
//! - `source/scripts/{train_toy_decisions.py, predict_toy_decisions.py}` — reference
//!   DecisionModel/set-head implementation this port replicates.
//!
//! Module map:
//! - [`safetensors`] — mmap + header parse + zero-copy F32 tensor views
//! - [`qwen3`] — Qwen3 forward (GQA, QK-RMSNorm, RoPE rotate-half, SwiGLU)
//! - [`prepare`] — request schema (`validate_request` port) + leaf prompt building
//! - [`model`] — full pipeline: leaf tokenize → forward → set-head → probabilities
//! - [`replay`] — deterministic replay of the frozen `test` cohort vs the
//!   recorded `predictions_test.jsonl` (top-1 agreement + |Δlogit| stats)

pub mod model;
pub mod prepare;
pub mod qwen3;
pub mod replay;
pub mod safetensors;
