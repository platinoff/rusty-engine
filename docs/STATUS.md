# rusty-engine — PROJECT STATUS

**Snapshot:** 2026-10-14 (agi-ланцюг: B0→B1 GGUF→B2 quant→B3 dense/KV/sampler/BPE→B4 MoE→B5 serve/heartbeat) · **без push** (remote-only у B11).

## Верифікований стан (zmіряно цього сеансу, не «на памʼять»)

| Метрика | Значення |
|---------|----------|
| `git status` | **clean** (0 drift) |
| Локальні коміти | **37** (B0→B5 serve/heartbeat, без push — remote = B11) |
| Тести | **146** (134 lib unit + 5 dense-gen + 3 text-gen + 4 integration `scaffold_contracts`), 0 failed |
| `cargo clippy --all-targets -- -D warnings` | **0** (чисто) |
| `cargo fmt --check` | **ok** |
| Rust LOC | **3909** (`src`+`tests`+`xtask`) |
| UI LOC (glue) | **157** (html+css+js) |
| **Rust ratio** | **96.1%** (ціль 95–100 ✓) |
| Docs LOC | 969 (concept/arch/roadmap/UI/research) |
| Tracked files | 41 |
| Реальна модель | `data/models/qwen2.5-0.5b-q4_k_m.gguf` (491 400 032 B, **gitignored**, sha256 `74a4da8c…`) |
| Трекінг `data/` | лише `.gitkeep` (модель не в репо ✓) |

## Що зроблено (канон смуг → `docs/roadmap.md`)

| Бенд | Статус | Ядро в Rust |
|------|--------|-------------|
| **B0** scaffold | ✅ | workspace + bin `rusty` + `xtask`; MoE `top_k`; 7 модулів |
| **B1** GGUF-чанкер | ✅ **complete** | header+KV+tensor-info, mmap/staged, sizes+Q4_0/Q8_0/Q4_K/Q5_K/Q6_K dequant, tokenizer-meta, `rusty inspect` |
| B1-real parity | ✅ | реальний Qwen2.5: парсер+`ggml_dtype` (баг off-by-K знайдено/виправлено) |
| **B2** kernels | 🟡 | `PH-201/201+/201c/201d` fused dot/gemv Q4_0/Q5_0/Q8_0 + oracle K-кванти; `PH-204` fused K-quant dot/gemv; `PH-202` repack; SIMD (`PH-203`) / bench (`PH-205`) open |
| **B3** forward | ✅ **complete** | `PH-301..306` + `PH-305a/c/d/e` + **`PH-305`/`PH-306`** (multi-layer dense runner + CLI `rusty run`, real Qwen2.5 24-layer prefill) + **`PH-305b`** (BPE byte-fallback + atomic specials) + **`PH-306b`** (text pipeline: `src/textgen.rs` `TextRunner` + `rusty run -p "text"` + `tests/text_gen.rs` synth round-trip); llama-parity `PH-405` open |
| **B4** MoE-core | 🟡 | `PH-401..403` (router softmax/sigmoid/group, SwiGLU, shared) + `PH-406a` `moe_layer_forward` + **`PH-404` expert-cache LRU** (`src/cache.rs`, budgeted + pin + prefetch); parity/run/bench (405/406/407) open |
| B5 serve / B6 build-matrix(exe/bin/apk) / B7 docker / B8 supervisor / B9 dist-MoE / B10 UI / B11 distribs | ☐ | заплановано (бенди в роадмепі) |

**Між конекцій:** від токена до logits стек повний — `config → validate → dequant → embed →
block_forward → sampler → BPE` — і реальна геометрія Qwen2.5 підтверджена (`rusty validate` = OK).

## Поточна межа (чому не «далі сліпо»)

1. Реальний `PH-305` multi-layer forward верифікується **тільки проти референсу** (row/col-major,
   RoPE-scaling, qk-norm, GQA-мапа) → gate **`PH-405`**; `llama-rs` живий → кандидат на оракул.
2. HF-exact Qwen BPE (`PH-305b`: `qwen2` pre-regex, byte-fallback, specials) — для читабельного тексту.
3. Далі — важкі/інфра-бенди (SIMD, axum-server, fleet/docker, build-matrix/apk) — кожен окремий великий зріз.

## Research / позиціонування

Див. **[`docs/research/engine-landscape-2026.md`](research/engine-landscape-2026.md)**:
vLLM v0.28 / SGLang v0.5.19 (RadixArk) / TensorRT-LLM v1.2.1 / llama.cpp / ktransformers.
`rusty-engine` — **не** GPU-cluster-throughput-ліга; ніша = pure-Rust MoE + один exe/bin/**apk**
+ edge/Android-no-Docker + GSV-native master/slave флот + нативний Rust-MoE (чого немає в llama-rs).

## Наступна `agi` (точка входу)

Див. **`docs/NEXT_SESSION_PROMPT.md`** (`Owner product pick: rusty-engine`). Готові green-зрізи:
`PH-305` real multi-layer forward (**потрібен дозвіл на llama-rs-референс** для числової parity) ·
`PH-202/203` (repack/SIMD, std-testable) · `PH-405` (llama/HF greedy-parity) · `PH-501` (axum serve).
