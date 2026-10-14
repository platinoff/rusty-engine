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
| [`docs/concept.md`](docs/concept.md) | концепція / vision / non-goals / джерела методів |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | crate/модуль-мапа, дані-потік, MoE-ядро, оркестрація |
| [`docs/roadmap.md`](docs/roadmap.md) | логічні бенди B0..B11 + PH-* тікети |
| [`docs/UI.md`](docs/UI.md) | веб UI design system (токени, сторінки, хоткеї, chrome-only law) |
| [`docs/HANDOFF_NEW_SESSION.md`](docs/HANDOFF_NEW_SESSION.md) | журнал сесій |
| [`docs/NEXT_SESSION_PROMPT.md`](docs/NEXT_SESSION_PROMPT.md) | наступний крок (owner pick) |

## Quick start

```bash
cargo xtask check        # fmt → clippy -D warnings → test
cargo build --release    # -> target/release/rusty(.exe)
cargo run --bin rusty -- --version
cargo run --bin rusty -- --help
```

## Status (B0)

Workspace scaffold compiles green: `rusty-engine` lib (`rusty_engine`) + `rusty` bin +
`xtask`; modules `core / gguf / moe / infer / serve / node / hub`; **17 tests pass**
(MoE top-k router proven in pure Rust before any model loads). GGUF parser + quant
kernels land in B1–B2; native MoE forward in B3–B4; fleet supervisor in B8.

See [`docs/roadmap.md`](docs/roadmap.md) for the band-by-band plan and
`S:/rust/GSV/docs/gsv/PRODUCTS.md` for the registration row (`rusty-engine`).
