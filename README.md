# rusty-engine

**Pure-Rust Mixture-of-Experts inference engine (GGUF-first), orchestrated by a
GSV-native master/slave supervisor.**

Single portable binary — `rusty.exe` / `rusty` / `rusty.apk` — for amd64 & arm64.
The MoE router, per-expert FFN and quantized kernels are **native Rust** (unlike
llama-rs, which delegates MoE to llama.cpp's C++ path), so the same engine ships to
Windows/Linux hosts and Android slaves over local Wi-Fi/LAN.

- **License:** MIT — `Copyright (c) 2026 platinoff / Madevinc`.
- **Toolchain:** pinned `1.92.0-x86_64-pc-windows-gnu`; MSYS2 bash (never PowerShell).
- **Ratio target:** Rust 95–100% (UI is thin glue; Rust renders the pages).

## Docs

| Doc | Что |
|-----|-----|
| [`docs/STATUS.md`](docs/STATUS.md) | **VERIFIED project state** (tests/commits/ratio/real-model validation) |
| [`docs/concept.md`](docs/concept.md) | концепція / vision / non-goals / джерела методів |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | crate/модуль-мапа, дані-потік, MoE-ядро, оркестрація |
| [`docs/roadmap.md`](docs/roadmap.md) | логічні бенди B0..B11 + PH-* тікети |
| [`docs/UI.md`](docs/UI.md) | веб UI design system (токени, сторінки, хоткеї, chrome-only law) |
| [`docs/research/engine-landscape-2026.md`](docs/research/engine-landscape-2026.md) | **research: порівняння з vLLM/SGLang/TRT-LLM/llama.cpp/ktransformers** |
| [`docs/research/`](docs/research/) | кеш фактажу (GGUF/quant/offsets/BPE) + real-model evidence |
| [`docs/HANDOFF_NEW_SESSION.md`](docs/HANDOFF_NEW_SESSION.md) | журнал сесій |
| [`docs/NEXT_SESSION_PROMPT.md`](docs/NEXT_SESSION_PROMPT.md) | наступний крок (owner pick) |

## Quick start

```bash
cargo xtask check        # fmt → clippy -D warnings → test
cargo build --release    # -> target/release/rusty(.exe)
cargo run --bin rusty -- --version
cargo run --bin rusty -- --help
```

## Status (verified 2026-10-14)

**19 commits · 79 tests · clippy `-D` clean · Rust ratio 96%.** End-to-end plumbing works:
`ModelConfig → validate → GGUF dequant (Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K) → embed →
`moe_layer_forward`/`block_forward` → sampler → byte-level BPE`. A **real** Qwen2.5-0.5B GGUF
is parsed and geometry-validated (`rusty validate` = OK, 24 layers), first real weight
materialized (`rusty tensor`). See [`docs/STATUS.md`](docs/STATUS.md) and
[`docs/research/engine-landscape-2026.md`](docs/research/engine-landscape-2026.md).

Remaining: `PH-305` real multi-layer forward (numeric parity needs a llama-rs/llama.cpp
reference → `PH-405`), HF-exact Qwen tokenization (`PH-305b`), then serving/build-matrix/fleet.
See [`docs/roadmap.md`](docs/roadmap.md) and `S:/rust/GSV/docs/gsv/PRODUCTS.md` (row `rusty-engine`).
