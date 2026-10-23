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

## Status (verified 2026-10-22, PH-203p)

**100 commits · 222 tests (+5 ignored parity) · clippy `-D` clean · Rust ratio 98.9 % (13890 LOC).**
End-to-end plumbing works: `ModelConfig → validate → GGUF dequant (Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K)
→ embed → `moe_layer_forward`/`block_forward` → sampler → byte-level BPE`. A **real**
Qwen2.5-0.5B GGUF is parsed and run (24 layers, greedy output byte-identical to the llama-rs
oracle).

Live `:8090`, Qwen2.5-0.5B Q4_K_M, 12 threads:

| Path | Now | Session start |
|------|-----|---------------|
| prefill 961 tok | **6.90–7.84 s** | 190.9 s (~26×) |
| prefill 321 tok | 5.6 s | 45.6 s |
| decode | **10.4–27.6 tok/s** (load-dependent) | 2.27 tok/s |
| Q5_0 matmul 4864×896×961 | 199 ms single thread / 56 ms pool | 1009 ms |

Profiled split of prefill (`RUSTY_PROFILE=1`): matmul 5.80–6.60 s (~84 %), attention 0.66–0.77 s
(9.8 %), SwiGLU 116–123 ms, RoPE 33–41 ms, norm/residual/kv-copy <70 ms each — the old ~11 %
"other" is down to **3.4–3.8 %** (SwiGLU moved onto the worker pool, RoPE uses one `(cos,sin)`
table per position). Golden fingerprint `tests/data/qwen25_golden.txt` has not moved through the
whole PH-203j…p series — every slice stayed bit-identical on the greedy output.

Remaining: `PH-203q` (K-quant dequant-once, tile multiple of 256), attention 9.8 %,
NEON, then `B6` build-matrix, `B7` docker, `B8` supervisor, `B9` dist-MoE, `B10` UI, `B11`
distributions. See [`docs/STATUS.md`](docs/STATUS.md), [`docs/NEXT_SESSION_PROMPT.md`](docs/NEXT_SESSION_PROMPT.md)
and `S:/rust/GSV/docs/gsv/PRODUCTS.md` (row `rusty-engine`).
