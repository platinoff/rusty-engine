# NEXT_SESSION_PROMPT — rusty-engine

## Owner product pick: `rusty-engine`

> Unknown pick = AskQuestion; інакше продовжувати з `docs/HANDOFF_NEW_SESSION.md`.

## Стан

**B0 = `[x]`** (2026-10-14) — portable-plugin `S:/rust/rusty-engine`: pure-Rust MoE-двигун
(GGUF-first), MIT, під GSV-хабом; workspace + bin `rusty` + `xtask`; Rust-MoE `top_k`-роутер.
**B1 = `[x]`** (2026-10-14) — GGUF-чанкер завершено: header+KV→tensor-info→mmap/staged→
block-sizes+Q4_0/Q8_0 dequant→tokenizer-meta. **35 тестів**, clippy `-D` 0. Коміти локальні,
**без push** (GitHub remote — B11 `PH-1105`).

## Наступний крок (B2 залізо: repack/SIMD, або реальна модель)

Done цього сеансу: **B1 `PH-101..107` + `PH-105b`** (контейнер + real-file parity + **byte-level BPE
encode/decode**), баг `ggml_dtype` знайдено/виправлено + регресійний тест; **B4-core `PH-401/402/403`
+ `PH-406a`** (router/SwiGLU/shared + **`moe_layer_forward` шар**); **B3-primitives `PH-301..304`**
(RMSNorm+attention+paged-KV+sampler) + **`PH-305a`** (mmap→byte-range→**real Q6_K dequant 4.36M finite**);
**B2 `PH-201`+`PH-201+`+`PH-201c`** (fused dot/gemv Q4_0/Q8_0 + скалярний reference-dequant Q4_K/Q5_K/Q6_K **+ Q5_0** —
real `token_embd` dequantize 136M finite → **embedding розблоковано**) —
**77 тести**, clippy `-D` 0. Модель у `data/models/` (gitignored); offset-семантика + BPE — у `docs/research/`.

Наступна `agi` обирає:
1. **`PH-305` dense forward** (⭐ embedding тепер деquantizeʼться — Q5_0 готовий): лишилось
   **multi-layer loop** (embed→rmsnorm→attn q/k/v/o GQA+RoPE→FFN ×24 + residual → `output`→logits)
   + **HF-exact Qwen parity** (`PH-305b`: `pre`-regex, byte-fallback, specials) → golden-токен.
2. **B2 залізо** `PH-202/203/204` — repack + SSE/AVX/NEON dispatch, `cargo bench` проти oracle.

## Контекст-дисципліна (кожен бенд «мільйон»)

Один agi-зріз = обмежений тікетами → std-only попереду → deps-зріз окремо → HANDOFF/NEXT
несуть стан → **один локальний коміт**. Фактаж кешуємо в `docs/research/`, ресерч — у
subagent-ах. Не тягну весь llama.cpp/vLLM у вікно за раз.

## Тікети на дошці

- `rusty-engine-bootstrap` (B0) — 10/10 **done** (сесія 2026-10-14).
- `rusty-engine-gguf` (B1) — банда **завершена**: `PH-101..107` + **`PH-105b`** (BPE) **done** (2026-10-14).
- `rusty-engine-moe-core` (B4) — `PH-401/402/403` **done** (router+SwiGLU+shared) + **`PH-406a` `moe_layer_forward`** (повний цикл шару, детермін.), `PH-404/405/406/407` **open** (потребують реальної GGUF-моделі).
- B3 primitives (`PH-301..304`) + **`PH-305a`** (real mmap→dequant) + **`PH-305c`** (`ModelConfig`) + **`PH-305d`** (`forward::block_forward` decode-step) — **done**; `PH-305` (multi-layer loop over real tensors + KV cache)/`PH-306`/`PH-305b` (HF-parity) **open**.
- B2 — `PH-201` + `PH-201+` + **`PH-201c` (Q5_0)** **done** (fused dot Q4_0/Q8_0 + K-quant + Q5_0 scalar reference); `PH-202..204` **open** (repack/SIMD/fused-gemv).

## Reminder (session law)

Global-first (`S:/rust/GSV/AGENTS.md`); MSYS2 bash; ratio 95–100%; Rust-MoE-ядро (не лінкуємо
C++); Docker-only-Win/Linux + Android-no-Docker; GSV hub = master supervisor; dry-run default;
push тільки B11.
