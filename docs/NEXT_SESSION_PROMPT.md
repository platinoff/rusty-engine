# NEXT_SESSION_PROMPT — rusty-engine

## Owner product pick: `rusty-engine`

> Unknown pick = AskQuestion; інакше продовжувати з `docs/HANDOFF_NEW_SESSION.md`.

## Стан

**B0 = `[x]`** (2026-10-14) — portable-plugin `S:/rust/rusty-engine`: pure-Rust MoE-двигун
(GGUF-first), MIT, під GSV-хабом; workspace + bin `rusty` + `xtask`; Rust-MoE `top_k`-роутер.
**B1 = `[x]`** (2026-10-14) — GGUF-чанкер завершено: header+KV→tensor-info→mmap/staged→
block-sizes+Q4_0/Q8_0 dequant→tokenizer-meta. **35 тестів**, clippy `-D` 0. Коміти локальні,
**без push** (GitHub remote — B11 `PH-1105`).

## Наступний крок (межа: K-quant ref через C-фактаж, або реальна модель)

Done цього сеансу: **B1 `PH-101..107`** (контейнер завершено), **B4-core `PH-401/402/403`**
(router/SwiGLU/shared), **B3-primitives `PH-301..304`** (RMSNorm + attention + paged-KV + sampler),
**B2 `PH-201`** (fused scalar reference Q4_0/Q8_0, == dequant-dot oracle) — **57 тести**, clippy `-D` 0.

Чистих **без-модельних std-зрізів більше немає**. Наступна `agi` обирає напрямок:
1. **B2 `PH-202..205`** — K-quant dequant/SIMD: спочатку **explore-subagent тягне точний
   `dequantize_q{4,5,6}_K_ref` з llama.cpp** у `docs/research/`, потім ручні block-тести → repack/SIMD.
2. **`PH-305`/`PH-406`/`PH-405`** — завантажити реальну tiny/real GGUF → dense/MoE golden + parity.

## Контекст-дисципліна (кожен бенд «мільйон»)

Один agi-зріз = обмежений тікетами → std-only попереду → deps-зріз окремо → HANDOFF/NEXT
несуть стан → **один локальний коміт**. Фактаж кешуємо в `docs/research/`, ресерч — у
subagent-ах. Не тягну весь llama.cpp/vLLM у вікно за раз.

## Тікети на дошці

- `rusty-engine-bootstrap` (B0) — 10/10 **done** (сесія 2026-10-14).
- `rusty-engine-gguf` (B1) — банда **завершена**: `PH-101..107` усі **done** (2026-10-14).
- `rusty-engine-moe-core` (B4) — банда створена: `PH-401/402/403` **done** (router+SwiGLU+shared), `PH-404..407` **open** (потребують реальної GGUF-моделі).
- B3 primitives (standalone тікети `PH-301/302/303/304`) — **done**; `PH-305/306` **open** (модель).
- B2 (standalone `PH-201`) — **done** (fused scalar reference Q4_0/Q8_0); `PH-202..205` **open** (K-quant ref → C-fактаж).

## Reminder (session law)

Global-first (`S:/rust/GSV/AGENTS.md`); MSYS2 bash; ratio 95–100%; Rust-MoE-ядро (не лінкуємо
C++); Docker-only-Win/Linux + Android-no-Docker; GSV hub = master supervisor; dry-run default;
push тільки B11.
