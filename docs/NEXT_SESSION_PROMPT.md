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

## Наступний крок (B1 — GGUF-чанкер)

1. `gsv_tickets_create { product:"rusty-engine", scenario_id:"rusty-engine-gguf" }` (або
   `gsv_tickets_hook { source:"scenario", id:"rusty-engine-gguf" }`) — 7 тікетів `PH-101..107`.
2. Drain: GGUF v3 header+KV (`src/gguf.rs`) → tensor-info table → mmap/mlock staged loader
   → dequant-блоки Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 → tiny-GGUF fixture `tests/gguf_roundtrip.rs`
   → `rusty inspect <model.gguf>`.
3. Додати перші нестд-залежності (`memmap2`, опційно `half`) — тепер це легітимно (B1).
4. `cargo xtask check` → record + один локальний коміт (без push).

## Тікети на дошці

- `rusty-engine-bootstrap` (B0) — відкрити→done у сесії 2026-10-14.
- `rusty-engine-gguf` (B1) — ready-to-hook.
- `rusty-engine-moe-core` (B4) — ready-to-hook (марquee-діференціатор: native MoE forward).

## Reminder (session law)

Global-first (`S:/rust/GSV/AGENTS.md`); MSYS2 bash; ratio 95–100%; Rust-MoE-ядро (не лінкуємо
C++); Docker-only-Win/Linux + Android-no-Docker; GSV hub = master supervisor; dry-run default;
push тільки B11.
