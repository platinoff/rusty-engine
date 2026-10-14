# NEXT_SESSION_PROMPT — rusty-engine

## Owner product pick: `rusty-engine`

> Unknown pick = AskQuestion; інакше продовжувати з `docs/HANDOFF_NEW_SESSION.md`.

## Стан

**B0 = `[x]`** (2026-10-14). Створено portable-plugin `S:/rust/rusty-engine`:
pure-Rust MoE-двигун (GGUF-first), MIT, під GSV-хабом. Workspace + bin `rusty` + `xtask`
збираються, **17 тестів green**, clippy 0. Rust-MoE `top_k`-роутер реалізовано й
протестовано до будь-якого модельного IO. Зареєстровано в `PRODUCTS.md` +
`gsv.code-workspace` + ticket-сценарії. Перший локальний коміт зроблено (**без push** —
push лише в B11 з згодою owner).

## Наступний крок (B1 — лишився PH-105, далі B3/B4)

Done: `PH-101/102` (`parse()`), `PH-103` (mmap+staged loaders `src/loader.rs`, `memmap2`),
`PH-104` (quant `src/quant.rs`: block-sizes усіх 5 + scalar dequant Q4_0/Q8_0; K-quant dequant
→ B2), `PH-106` (disk round-trip), `PH-107` (`rusty inspect`). Фактаж — `docs/research/`.

1. `PH-105` tokenizer-**metadata ingest** (`tokenizer.ggml.tokens/scores/token_type/merges/bos/eos/pre`)
   → `src/tokenizer.rs` `TokenizerMeta::from_file(&GgufFile)` (не сам BPE-алгоритм — той у B3).
2. Далі — **B3 dense-forward** (RMSNorm/attn/RoPE/paged-KV/sampler) і **B4 MoE-ядро**
   (`rusty-engine-moe-core` ready) — марquee-діференціатор.

## Контекст-дисципліна (кожен бенд «мільйон»)

Один agi-зріз = обмежений тікетами → std-only попереду → deps-зріз окремо → HANDOFF/NEXT
несуть стан → **один локальний коміт**. Фактаж кешуємо в `docs/research/`, ресерч — у
subagent-ах. Не тягну весь llama.cpp/vLLM у вікно за раз.

## Тікети на дошці

- `rusty-engine-bootstrap` (B0) — 10/10 **done** (сесія 2026-10-14).
- `rusty-engine-gguf` (B1) — банда створена: `PH-101/102/103/104/106/107` **done**, `PH-105` **open**.
- `rusty-engine-moe-core` (B4) — ready-to-hook (марquee-діференціатор: native MoE forward).

## Reminder (session law)

Global-first (`S:/rust/GSV/AGENTS.md`); MSYS2 bash; ratio 95–100%; Rust-MoE-ядро (не лінкуємо
C++); Docker-only-Win/Linux + Android-no-Docker; GSV hub = master supervisor; dry-run default;
push тільки B11.
