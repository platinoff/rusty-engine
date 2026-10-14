# NEXT_SESSION_PROMPT — rusty-engine

## Owner product pick: `rusty-engine`

> Unknown pick = AskQuestion; інакше продовжувати з `docs/HANDOFF_NEW_SESSION.md`.

## Стан

**B0 = `[x]`** (2026-10-14) — portable-plugin `S:/rust/rusty-engine`: pure-Rust MoE-двигун
(GGUF-first), MIT, під GSV-хабом; workspace + bin `rusty` + `xtask`; Rust-MoE `top_k`-роутер.
**B1 = `[x]`** (2026-10-14) — GGUF-чанкер завершено: header+KV→tensor-info→mmap/staged→
block-sizes+Q4_0/Q8_0 dequant→tokenizer-meta. **35 тестів**, clippy `-D` 0. Коміти локальні,
**без push** (GitHub remote — B11 `PH-1105`).

## Наступний крок (B3 attention/KV primitives — std, далі модель)

Done цього сеансу: **B1 `PH-101..107`** (GGUF-контейнер, завершено), **B4-core `PH-401/402/403`**
(router/SwiGLU/shared), **B3-primitives `PH-301/304`** (RMSNorm + sampler) — **50 тестів**, clippy `-D` 0.

Наступний std-testable зріз (без моделі):
1. `PH-302` **attention primitives** (`src/infer/attn.rs`): causal-mask, GQA/MQA broadcast, RoPE —
   unit на синтетичних Q/K/V.
2. `PH-303` **paged KV-cache** (`src/infer/kvcache.rs`): slab alloc + round-trip.
Далі залежне від моделі/dep: `PH-305` dense golden (+`PH-306` `rusty run`), B2 `PH-201`
(K-quant dequant + SIMD), B4 `PH-404..407` (expert-cache/parity/run/bench — реальна GGUF-модель).

## Контекст-дисципліна (кожен бенд «мільйон»)

Один agi-зріз = обмежений тікетами → std-only попереду → deps-зріз окремо → HANDOFF/NEXT
несуть стан → **один локальний коміт**. Фактаж кешуємо в `docs/research/`, ресерч — у
subagent-ах. Не тягну весь llama.cpp/vLLM у вікно за раз.

## Тікети на дошці

- `rusty-engine-bootstrap` (B0) — 10/10 **done** (сесія 2026-10-14).
- `rusty-engine-gguf` (B1) — банда **завершена**: `PH-101..107` усі **done** (2026-10-14).
- `rusty-engine-moe-core` (B4) — банда створена: `PH-401/402/403` **done** (router+SwiGLU+shared), `PH-404..407` **open** (потребують реальної GGUF-моделі).

## Reminder (session law)

Global-first (`S:/rust/GSV/AGENTS.md`); MSYS2 bash; ratio 95–100%; Rust-MoE-ядро (не лінкуємо
C++); Docker-only-Win/Linux + Android-no-Docker; GSV hub = master supervisor; dry-run default;
push тільки B11.
