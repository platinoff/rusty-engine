# rusty-engine — PROJECT STATUS

**Snapshot:** 2026-10-18 (agi-ланцюг: B0→B1 GGUF→B2 quant→B3 dense/KV/sampler/BPE→B4 MoE→B5
serve/heartbeat → **PH-405b forward-parity з llama-rs ✅** → **PH-507 hub provider-row E2E ✅
(JSON + SSE) через живий хаб `:9999`** → **B2 SIMD bench ✅ (`PH-203` x86 / `PH-205`, 1.8–5.3×)**
→ **`PH-505` residual ✅ (per-process RAM у `/metrics`)** → **`PH-203d` AVX2+FMA fused dots ✅
(Q8_0 dot 1.72→2.09×, live decode 2.27→2.68 tok/s)** → **`PH-203e` row-parallel matvec/matmul ✅
(decode 2.68→4.2–5.1 tok/s, prefill 1.85×, `RUSTY_THREADS`)**) · **без push** (remote-only у B11).

## Верифікований стан (zmіряно цього сеансу, не «на памʼять»)

| Метрика | Значення |
|---------|----------|
| `git status` | **clean** (0 drift) |
| Локальні коміти | **28** (ahead of `origin/main`; B0→B5 serve/heartbeat + PH-405b + PH-507 + B2 SIMD bench + PH-505 RAM + PH-203d FMA + PH-203e parallel, без push — remote = B11) |
| Тести | **200** (181 lib + 5 dense-gen + 4 scaffold + 5 serve + 5 text-gen; +5 ignored parity), 0 failed |
| `cargo clippy --all-targets -- -D warnings` | **0** (чисто) |
| `cargo fmt --check` | **ok** |
| Rust LOC | **11895** (`src` 10039 + `tests` 1317 + `xtask` 319 + `benches` 220) |
| UI LOC (glue) | **157** (html+css+js) |
| **Rust ratio** | **98.7%** (ціль 95–100 ✓) |
| Docs LOC | 1709 (concept/arch/roadmap/UI/research/handoff/NEXT) |
| Tracked files | 64 |
| Реальна модель | `data/models/qwen2.5-0.5b-q4_k_m.gguf` (491 400 032 B, **gitignored**, sha256 `74a4da8c…`) |
| Трекінг `data/` | лише `.gitkeep` (модель не в репо ✓) |

## Що зроблено (канон смуг → `docs/roadmap.md`)

| Бенд | Статус | Ядро в Rust |
|------|--------|-------------|
| **B0** scaffold | ✅ | workspace + bin `rusty` + `xtask`; MoE `top_k`; 7 модулів |
| **B1** GGUF-чанкер | ✅ **complete** | header+KV+tensor-info, mmap/staged, sizes+Q4_0/Q8_0/Q4_K/Q5_K/Q6_K dequant, tokenizer-meta, `rusty inspect` |
| B1-real parity | ✅ | реальний Qwen2.5: парсер+`ggml_dtype` (баг off-by-K знайдено/виправлено) |
| **B2** kernels | 🟡 | `PH-201/201+/201c/201d` fused dot/gemv Q4_0/Q5_0/Q8_0 + oracle K-кванти; `PH-204` fused K-quant dot/gemv; `PH-202` repack; **`PH-203` x86 SIMD (SSE2 Q4_0/Q8_0, AVX2 Q4_0/Q5_0/Q8_0/Q4_K/Q6_K) + `PH-205` bench ✅ (1.8–5.3×)**; **`PH-203d` AVX2+FMA fused dots ✅ (Q8_0 dot 1.72→2.09×)**; **`PH-203e` row-parallel matvec/matmul ✅ (decode 2.68→4.2–5.1 tok/s, prefill 45.6→24.7 s, `RUSTY_THREADS`)**; лишився NEON |
| **B3** forward | ✅ **complete** | `PH-301..306` + `PH-305a/c/d/e` + **`PH-305`/`PH-306`** (multi-layer dense runner + CLI `rusty run`, real Qwen2.5 24-layer prefill) + **`PH-305b`** (BPE byte-fallback + atomic specials) + **`PH-306b`** (text pipeline: `src/textgen.rs` `TextRunner` + `rusty run -p "text"` + `tests/text_gen.rs` synth round-trip); **`PH-405`** numeric regression-gate + **`PH-405b`** semantic greedy-parity з llama-rs ✅ |
| **B4** MoE-core | 🟡 | `PH-401..403` (router softmax/sigmoid/group, SwiGLU, shared) + `PH-406a` `moe_layer_forward` + **`PH-404` expert-cache LRU** (`src/cache.rs`, budgeted + pin + prefetch); parity/run/bench (405/406/407) open |
| **B5** serve | 🟡 | PH-501a/b (std HTTP + real TextRunner), PH-502 (SSE), PH-503 (slot-cache), PH-504 (heartbeat), PH-505 (**metrics + per-process RAM**), PH-506 (serve integration tests ✅); **PH-507 hub provider-row E2E ✅ (`rusty-serve` через хаб, JSON+SSE)** |
| B6 build-matrix(exe/bin/apk) / B7 docker / B8 supervisor / B9 dist-MoE / B10 UI / B11 distribs | ☐ | заплановано (бенди в роадмепі) |

**Між конекцій:** від токена до logits стек повний — `config → validate → dequant → embed →
block_forward → sampler → BPE` — і реальна геометрія Qwen2.5 підтверджена (`rusty validate` = OK).

## Поточна межа (чому не «далі сліпо»)

1. **`PH-405` числовий regression-gate готовий** (`tests/numeric_parity.rs` + golden на
   справжній Qwen2.5) — guard реґресій forward/quant/KV/sampler. **`PH-405b` semantic
   llama-rs parity — ✅ ВИПРАВЛЕНО (2026-10-17):** биті q/k/v-біаси Qwen2 + RoPE
   GPT-J→NeoX; наш greedy тепер байт-у-байт збігається із записаним оракулом (55/12/48),
   golden перезахоплено. Гейти — `tests/moe_parity.rs`.
2. HF-exact Qwen BPE (`PH-305b`: `qwen2` pre-regex, byte-fallback, specials) — для читабельного тексту.
3. Далі — важкі/інфра-бенди (NEON arm64, axum-server, fleet/docker, build-matrix/apk) — кожен окремий великий зріз. **B2 x86 SIMD + bench закрито** (`PH-203`/`PH-205`), **hot dots переведено на AVX2+FMA** (`PH-203d`), **проєкції паралеляться по рядках** (`PH-203e`, `RUSTY_THREADS`, дефолт = available_parallelism).

## Research / позиціонування

Див. **[`docs/research/engine-landscape-2026.md`](research/engine-landscape-2026.md)**:
vLLM v0.28 / SGLang v0.5.19 (RadixArk) / TensorRT-LLM v1.2.1 / llama.cpp / ktransformers.
`rusty-engine` — **не** GPU-cluster-throughput-ліга; ніша = pure-Rust MoE + один exe/bin/**apk**
+ edge/Android-no-Docker + GSV-native master/slave флот + нативний Rust-MoE (чого немає в llama-rs).

## Наступна `agi` (точка входу)

Див. **`docs/NEXT_SESSION_PROMPT.md`** (`Owner product pick: rusty-engine`). **`PH-507` E2E
зроблено** (лишається WiFi multi-host band 233 + IDE-chat `rusty/*`). **B2 x86 SIMD + bench
(`PH-203`/`PH-205`) закрито.** Готові green-зрізи: NEON arm64 (`PH-203`) · `PH-501` (axum serve)
· важкі/інфра-бенди (B6/B7).
