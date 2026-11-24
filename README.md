# rusty-engine

**Pure-Rust Mixture-of-Experts inference engine (GGUF-first), orchestrated by a
GSV-native master/slave supervisor.**

Single portable binary — `rusty.exe` / `rusty` / `rusty.apk` — for amd64 & arm64.
The MoE router, per-expert FFN and quantized kernels are **native Rust** (unlike
llama-rs, which delegates MoE to llama.cpp's C++ path), so the same engine ships to
Windows/Linux hosts and Android slaves over local Wi-Fi/LAN.

- **License:** MIT — `Copyright (c) 2026 platinoff / Madevinc`.
- **Toolchain:** pinned `1.100.0-x86_64-pc-windows-gnu`; MSYS2 bash (never PowerShell).
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

## Secrets

This tree is safe to publish because it does not carry hub credentials.
`GSV_EDGE_TOKEN` and any `.env` live only on the machine that runs `rusty`.
Someone else using the GSV ecosystem creates those on their own hub and
exports them in the environment. Do not commit them.

## Status (verified 2026-11-24, HEAD `61bb0ba` — PH-1204 plumbing; PH-1305 parity ✅ 2026-11-11)

**160 commits · ~316 `#[test]` (`cargo xtask check` green at `61bb0ba`) · clippy `-D` clean · Rust ratio ≥99 % (25 794 LOC) · 107 tracked files.** PH-1204 vision-prefix hook landed (`--vision-prefix <f32>`, `runner35::prefill_embeddings`, state-cache bypass).
End-to-end plumbing works: `ModelConfig → validate → GGUF dequant (Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K)
→ embed → `moe_layer_forward`/`block_forward` → sampler → byte-level BPE`. A **real**
Qwen2.5-0.5B GGUF is parsed and run (24 layers, greedy output byte-identical to the llama-rs
oracle).

Live `:8090`, Qwen2.5-0.5B Q4_K_M, 12 threads:

| Path | Now | Session start |
|------|-----|---------------|
| prefill 961 tok | **5.3–6.4 s** (box drifts) | 190.9 s (~30–36×) |
| prefill 321 tok | 5.6 s | 45.6 s |
| decode | **10.4–27.6 tok/s** (load-dependent) | 2.27 tok/s |
| Q6_K down 896×4864×961 (pool) | **64–69 GMAC/s** (1.46–1.56 s) | 10.0 GMAC/s, 4.2 s |
| Q5_0 matmul 4864×896×961 | 199 ms single thread / 56 ms pool | 1009 ms |

Profiled split of prefill (`RUSTY_PROFILE=1`): matmul 4.1–4.7 s (~77 %), attention 0.74–0.78 s
(13.9 %), SwiGLU 122–132 ms, RoPE 32–35 ms, norm/residual/kv-copy <70 ms each — the old ~11 %
"other" is down to **2.4–3 %**. Three levers got there: `PH-203p` put SwiGLU on the pool and
RoPE on a per-position `(cos,sin)` table, and `PH-203r` dequantizes **two weight rows per pass**
(`dot_f32_x4x2`: `6 loads / 8 FMA` instead of `5 / 4`), which took the pooled kernels from 76 to
102 GMAC/s on gate/up and 77 to 108 on the Q6_K down projection. A 256-multiple K-quant tile,
8-row register blocking and a per-worker L3 tile are all measured *and rejected* — the numbers
live in `benches/kquant.rs` / `benches/quant.rs`. Golden fingerprint
`tests/data/qwen25_golden.txt` has not moved through the whole PH-203j…r series — every slice
stayed bit-identical on the greedy output.

Remaining: **PH-1204 full image→answer pipeline / PH-1205 multimodal serve / G13 GDN-perf** (PH-1204 plumbing landed in `61bb0ba`; GSV still producing the full-depth 1024-dim vision vector — no quality claim). `PH-203s`/`PH-203t` were measured and rejected 2026-10-24; then NEON, then
`B6` build-matrix, `B7` docker, `B8` supervisor, `B9` dist-MoE, `B10` UI, `B11` distributions. See [`docs/STATUS.md`](docs/STATUS.md), [`docs/NEXT_SESSION_PROMPT.md`](docs/NEXT_SESSION_PROMPT.md)
and `S:/rust/GSV/docs/gsv/PRODUCTS.md` (row `rusty-engine`).
