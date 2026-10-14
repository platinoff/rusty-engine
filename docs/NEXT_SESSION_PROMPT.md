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

## Наступний крок (B1 продовження — PH-103..107)

`PH-101` (header+KV) і `PH-102` (tensor-info) **done** — `src/gguf.rs` `parse()`,
`docs/research/gguf-format.md` (кеш фактажу). Далі — зріз, що **потребує залежностей**:

1. `PH-103` mmap/mlock staged loader → додати `memmap2` (перша не-std залежність;
   тепер легітимно; перевірити, чи cargo має мережу/vendor-регістр ДО старту).
2. `PH-104` dequant Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 → точні block-sizes з `ggml-common.h`.
3. `PH-105` BPE-токенізер (`tokenizer.ggml.*`) · `PH-106` **real** tiny-GGUF fixture
   round-trip (валідація проти справжнього файлу, не synth) · `PH-107` `rusty inspect`.
4. Research → **explore-subagent** (тягне llama.cpp-семантику окремо; основний контекст не
   засмічує) → результат дописати в `docs/research/`.

## Контекст-дисципліна (кожен бенд «мільйон»)

Один agi-зріз = обмежений тікетами → std-only попереду → deps-зріз окремо → HANDOFF/NEXT
несуть стан → **один локальний коміт**. Фактаж кешуємо в `docs/research/`, ресерч — у
subagent-ах. Не тягну весь llama.cpp/vLLM у вікно за раз.

## Тікети на дошці

- `rusty-engine-bootstrap` (B0) — 10/10 **done** (сесія 2026-10-14).
- `rusty-engine-gguf` (B1) — банда створена: `PH-101`/`PH-102` **done**, `PH-103..107` **open**.
- `rusty-engine-moe-core` (B4) — ready-to-hook (марquee-діференціатор: native MoE forward).

## Reminder (session law)

Global-first (`S:/rust/GSV/AGENTS.md`); MSYS2 bash; ratio 95–100%; Rust-MoE-ядро (не лінкуємо
C++); Docker-only-Win/Linux + Android-no-Docker; GSV hub = master supervisor; dry-run default;
push тільки B11.
