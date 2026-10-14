# NEXT_SESSION_PROMPT — rusty-engine

## Owner product pick: `rusty-engine`

> Unknown pick = AskQuestion; iнакше продовжувати з `docs/HANDOFF_NEW_SESSION.md`.

## Стан

**B0 = `[x]`** (2026-10-14) — portable-plugin `S:/rust/rusty-engine`: pure-Rust MoE-двигун
(GGUF-first), MIT, під GSV-хабом; workspace + bin `rusty` + `xtask`; Rust-MoE `top_k`-роутер.
**B1 = `[x]`** (2026-10-14) — GGUF-чанкер завершено: header+KV→tensor-info→mmap/staged→
block-sizes+Q4_0/Q8_0 dequant→tokenizer-meta. clippy `-D` 0. Коміти локальні,
**без push** (GitHub remote — B11 `PH-1105`).

## Наступний крок

Done цього сеансу: **B1** (контейнер + real-file parity + **byte-level BPE encode/decode**),
баг `ggml_dtype` знайдено/виправлено; **B4-core** (`PH-401/402/403/406a` router/SwiGLU/shared
+ `moe_layer_forward`); **B3-primitives** `PH-301..304` + `PH-305a/c/d/e` (RMSNorm/attention/
paged-KV/sampler + real mmap→Q6_K dequant + `ModelConfig` + `block_forward` + `validate`).
**B3 `PH-305`/`PH-306` ЗАВЕРШЕНО** — `src/runner.rs` `DenseRunner` = повний multi-layer forward
на справжній Qwen2.5: `block_forward_seq` (causal-GQA + RoPE, per-row SwiGLU + residual) →
final rmsnorm → `output` → logits. `quant::dequant` додав **F32/F16** гілки (норми реальних
моделей). `tests/dense_gen.rs`: synth 2-layer greedy→token 4 + `#[ignore]` real-Qwen prefill
→ **151 936 finite logits** (усі 24 шари, pure-Rust). CLI `rusty run -m <model> -t "ids"
[--max-tokens N]` (PH-306, numeric tokens). **84 тести** (78 lib + 2 dense + 4 scaffold),
clippy `-D` 0, ratio Rust ~96%. Модель у `data/models/` (gitignored).
**Checkpoint:** зріз стану — `docs/STATUS.md`; порівняння рушіїв —
`docs/research/engine-landscape-2026.md`; offset-семантика — `docs/research/`.

Наступна `agi` обирає:
1. **`PH-305b` HF-exact Qwen BPE** (⭐ читабельний текст): qwen2 `pre`-regex, byte-fallback,
   specials → `rusty run` друкує слова, не id.
2. **`PH-405` llama/HF greedy-parity** (⭐ gate): perplexity-smoke + greedy token match на
   коротких промптах проти llama-rs (підлога вже живе) — чисельна довіра до `PH-305`.
3. **B2 залізо** `PH-202/203/204` — repack + SSE/AVX/NEON dispatch, `cargo bench` проти oracle
   (std-testable, без референсу).
4. **B4 `PH-406`** run Qwen3-30B-A3B / Mixtral GGUF (реальна MoE-модель + expert-цикл на
   деquant-тензорах; потребує `PH-305`-подібного runner для MoE-шарів).

## Контекст-дисципліна (кожен бенд «мільйон»)

Один agi-зріз = обмежений тікетами → std-only попереду → deps-зріз окремо → HANDOFF/NEXT
несуть стан → **один локальний коміт**. Фактаж кешуємо в `docs/research/`, ресерч — у
subagent-ах. Не тягну весь llama.cpp/vLLM у вікно за раз.

## Тікети на дошці

- `rusty-engine-bootstrap` (B0) — 10/10 **done** (сесія 2026-10-14).
- `rusty-engine-gguf` (B1) — банда **завершена**: `PH-101..107` + **`PH-105b`** (BPE) **done** (2026-10-14).
- `rusty-engine-moe-core` (B4) — `PH-401/402/403` + `PH-406a` **done**; `PH-404/405/406/407` **open**.
- B3 primitives + **`PH-305a/c/d/e`** + **`PH-305`/`PH-306`** **done** (real Qwen2.5 24-layer
  dense forward OK, CLI `rusty run`). `PH-305b` (HF BPE) + `PH-405` (llama-parity) **open**.
- B2 — `PH-201` + `PH-201+` + **`PH-201c` (Q5_0)** + **F32/F16 dequant** **done**; `PH-202..204` **open**.

## Reminder (session law)

Global-first (`S:/rust/GSV/AGENTS.md`); MSYS2 bash; ratio 95–100%; Rust-MoE-ядро (не лінкуємо
C++); Docker-only-Win/Linux + Android-no-Docker; GSV hub = master supervisor; dry-run default;
push тільки B11.
