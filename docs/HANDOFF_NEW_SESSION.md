# HANDOFF_NEW_SESSION — rusty-engine

**Продукт:** `rusty-engine` · Сесійна мова: «концепція → роадмеп → тікети → drain».
**Host:** GSV hub (`S:/rust/GSV`, `:9999`) = master; це — portable plugin.

## Що було зроблено (B0, campaign 2026-10-14)

Створено новий **portable-plugin** репо `S:/rust/rusty-engine` — унікальний **pure-Rust
MoE-двигун** (GGUF-first), MIT, під GSV-хабом. Рішення owner (5 пунктів) зафіксовані в
`docs/concept.md` §11. Коміт: **B0 skeleton** (див. git log).

**B0 — Фундамент та реєстрація (17 тестів, gate зелений; clippy 0):**

- `Cargo.toml` — workspace (`members=["xtask"]`) + root pkg `rusty-engine` (lib
  `rusty_engine`, bin `rusty`), MIT metadata, `authors=["platinoff"]`, `repository`
  `github.com/platinoff/rusty-engine`. `rust-toolchain.toml` pin `1.92.0-x86_64-pc-windows-gnu`.
- `src/lib.rs` — модуль-карта (`core/gguf/moe/infer/serve/node/hub`), `CRATE_NAME`/`VERSION`/`banner()`.
- `src/core.rs` — `DType`(F32/F16/Q4_0..Q8_0/I8), `TensorInfo.numel()`, `EngineError`/`Result` — 2 unit.
- `src/gguf.rs` — `GGUF_MAGIC` + `matches_magic()` — 2 unit.
- `src/moe.rs` — **ДИФЕРЕНЦІЯТОР**: `softmax`, `top_k`→`Vec<ExpertScore>` (top-k gating +
  renorm) у Rust — 3 unit.
- `src/infer.rs` — `GenerationOptions` (temp/top_p/top_k/max_tokens) — 1 unit.
- `src/serve.rs` — OpenAI-route constи — 1 unit. `src/node.rs` — `Role{Master,Slave}` — 1 unit.
- `src/hub.rs` — heartbeat-tick 15s + fresh 60s (compile-time invariant) + `Worker` — 1 unit.
- `src/bin/rusty.rs` — `rusty` CLI (banner / `--version` / `--help`, заготови `serve/node/hub`).
- `xtask/` + `.cargo/config.toml` — `cargo xtask check|fmt|clippy|test`.
- `tests/scaffold_contracts.rs` — 4 contract-тести (GGUF magic, MoE top-k на 8-expert
  Qwen-гейті, tensor-descriptor, fleet-worker) — **підтверджують Rust-MoE-роутер до
  будь-якого модельного IO**.
- `LICENSE` (MIT, platinoff/Madevinc), `.gitignore`, `data/.gitkeep`.
- `docs/`: `concept.md` · `ARCHITECTURE.md` · `roadmap.md` (B0..B11) · `UI.md` ·
  `HANDOFF` · `NEXT`. `README.md`, `AGENTS.md` (local).
- `ui/`: shell `index.html` + `css/style.css` (токени) + `js/app.js` (chrome-only placeholder).
- Реєстрація в `S:/rust/GSV/docs/gsv/PRODUCTS.md` (row `rusty-engine`) + `gsv.code-workspace`
  folder + ticket-сценарії (`rusty-engine-bootstrap` / `-gguf` / `-moe-core`).

**Smoke:** `cargo xtask check` → `cargo clippy -D warnings` 0 → `cargo test` **17 passed**
(13 unit + 4 contract). `cargo run --bin rusty -- --help` друкує банер.

## Наступний крок

## Наступний крок

**Сесія 2026-10-14 «verify + research»:** стан верифіковано (git чистий, **19 комітів**, **79 тестів**,
clippy `-D` 0, Rust ratio **96%**, модель gitignored), додано [`docs/STATUS.md`](STATUS.md) +
[`docs/research/engine-landscape-2026.md`](research/engine-landscape-2026.md) (порівняння з
vLLM/SGLang/TensorRT-LLM/llama.cpp/ktransformers + позиціонування rusty-engine). Код не змінено.

**B1 — GGUF-чанкер:** `[x]` **ЗАВЕРШЕНО** (усі PH-101..107) — GGUF-контейнер читається
повністю в Rust без llama.cpp. **35 тестів**, clippy `-D` 0. Фактаж — `docs/research/gguf-format.md`
+ `docs/research/ggml-blocks.md` (cached через explore-subagent).

- `PH-101/102` `src/gguf.rs` — `parse()` header+KV (усі `GGUF_TYPE_*`, вкладені arrays) + tensor-info (`TensorEntry`/`ggml_dtype`).
- `PH-103` `src/loader.rs` — `MmapModel::open` (zero-copy, `memmap2` — **перша не-std залежність**, регістр перевірено на старті), `lock_pages`(unix-gated), `staged_read` disk→RAM + progress, `load_staged`.
- `PH-104` `src/quant.rs` — `f16_to_f32`; `block_info`/`bytes_for_numel` для **усіх 5** (18/34/144/176/210 B/block); scalar `dequant_q4_0` (**split-half**) + `dequant_q8_0` (signed); K-quant *dequant math* → **B2**.
- `PH-105` `src/tokenizer.rs` — `TokenizerMeta::from_file` (model/tokens/token_type/scores/merges/bos/eos/pre + accessors);
  **PH-105b** — `ByteEncoder` (GPT-2 byte↔char, invertible 0..=255, 0x20→Ġ) + `Bpe` (pre-tokenize + merge-by-rank)
  **encode/decode round-trip** (HF-exact Qwen parity → PH-305b).
- `PH-106` disk round-trip (`parse_file` read synth `.gguf` з диска); `PH-107` `rusty inspect` CLI.
- **PH-106-real** — парсер прогнано на **справжньому** `Qwen2.5-0.5B-Instruct Q4_K_M` (491 400 032 B,
  sha256 `74a4da8c…`, gitignored у `data/models/`): 291 tensors / 26 kv / arch `qwen2`. При цьому
  **знайдено й виправлено баг** `ggml_dtype` (було `14→Q4_K`, канон `Q4_K=12/Q5_K=13/Q6_K=14`) +
  регресійний тест `ggml_dtype_maps_canonical_ggml_codes`; `inspect` тепер друкує dtype. Доказ —
  `docs/research/real-model-inspect.md`. **64 тести** загалом, clippy `-D` 0.

**B4 — MoE-ядро (частково):** `[~]` — **PH-401/402/403/404/406a done** (`src/moe.rs`, `src/cache.rs`, pure-f32): роутер
`route` (softmax-renorm) + `top_k_sigmoid` (Qwen3/DeepSeek renorm) + `route_group` (noaux_tc-lite:
group-score + top_k_group restrict); `expert_ffn` = `down(swiglu(gate,up))` (`silu`/`matvec`);
`combine` shared-expert merge; **`moe_layer_forward` = повний цикл шару** (router→top-k experts
weighted-sum→+shared) з детермінованим інтеграційним тестом; **`PH-404` expert-cache LRU**
(`ExpertCache::get_or_load/put/prefetch/set_pinned` — byte-budget + pinned-skipped eviction,
oversized→uncached, 7 unit). Лишилось: `PH-405` llama/HF-parity (оракул) + `PH-406` real
MoE GGUF run + `PH-407` bench. **95 тестів**, clippy `-D` 0.
Фактаж — `docs/research/moe-router.md`.

**B3 — Dense forward (частково):** `[~]` — **PH-301/302/303/304 done** (`src/infer.rs`, pure-f32):
`rmsnorm`; attention primitives `rope`/`apply_causal_mask`/`attention_causal`/`gqa_repeat_kv`;
`PagedKvCache` (append/get across pages); sampler chain `sample`/`apply_temperature`/`mask_top_k`/
`mask_top_p`/`apply_repetition_penalty`/`argmax` (детермінований `r`).
**PH-305a** — `MmapModel::tensor_bytes`/`dequant` + `TensorEntry::nbytes` (GGUF byte-offset =
`data_start + stored byte delta`; `nbytes`==`ggml_nbytes`==`bytes_for_numel`), CLI `rusty tensor`.
**Живе читання реальної Qwen2.5:** `ffn_down` Q6_K → **4 358 144 finite** ваг, `attn_norm` F32 → 896.
Offset-семантика кеш — `docs/research/gguf-tensor-offsets.md`.

**B2 — Quantized kernels (частково):** `[~]` — **PH-201 + PH-201+ + PH-201c + PH-204 done**:
fused scalar dot/gemv Q4_0/Q8_0 (`src/kernel.rs`, `== dequant-then-dot`) **і** скалярний
reference-dequant Q4_K/Q5_K/Q6_K (`src/quant.rs`) **+ PH-204 fused K-quant dot/gemv**
(`dot_q{4,5,6}_k` + `matvec_q{4,5,6}_k` macro — per-block dequantize into a stack `[f32;256]`
and fold into the dot without materializing the full f32 row; parity vs `dot(dequant_qX_k, b)`
relative tol). Реальна Qwen2.5 вага — Q4_K_M мікс: attn/ffn Q4_K + ffn_down Q6_K — саме ці
ядра дадуть ~10× менше памʼяті під forward (B4 expert-cache). **`PH-202` repack** (`src/repack.rs`,
compute-friendly layout, exact vs kernel) + **`PH-203/203b` SSE2 dispatch** (`dot_q{4,8}_0_sse2`
`core::arch` + runtime-detect `dot_*_dispatch`, scalar fallback) — **done**. Лишилось: AVX2/NEON
(`PH-203c`) + bench (`PH-205`).

**Числовий gate B4:** **`PH-405`** (`tests/numeric_parity.rs` + `tests/data/qwen25_golden.txt`) —
детермінований greedy-відбиток повного 24-шарового `DenseRunner.generate` на справжній Qwen2.5;
`#[ignore]` release-job (~60s), CI без gitignored-моделі skip. Guard реґресій quant/forward/KV/sampler.

**B3 — Dense forward (control path):** `[x]` — **PH-301..306 + PH-305a/c/d/e/b + PH-305 + PH-306b done** (2026-10-14):
повний **multi-layer dense forward** на справжній Qwen2.5-0.5B (`src/runner.rs` `DenseRunner::
from_model/prefill_logits/greedy/generate` + `forward::block_forward_seq` causal-GQA + RoPE +
per-row SwiGLU + residual → final rmsnorm → `output` → logits). `quant::dequant` додав **F32/F16**
гілки. **`PH-305b`** — BPE **byte-fallback + atomic specials** (`src/tokenizer.rs`) + std-only
**GPT-2/Qwen2 pre-regex** (`src/gpt2.rs`, wired into `Bpe::encode`). **`PH-306b`** — **text
pipeline** `src/textgen.rs` `TextRunner` (Bpe + DenseRunner) + `rusty run -p "text"`. CLI
`rusty run -m <model> -t "ids" | -p "text" [--max-tokens N]` — greedy на числових id / тексті.
**161 тестів** (147 lib + 5 dense + 3 text + 4 scaffold + 2 xtask; +1 ignored numeric parity), clippy `-D` 0, ratio ~96%. Real-Qwen
`#[ignore]`-smoke: prefill `[151644, 872, 198]` → **151 936 finite logits** (усі 24 шари pure-Rust).
**Milestone (22 agi-бендів, B0→B6):** рушій числово закріплений `PH-405` golden-gate проти реального 491 MB Qwen2.5.

**Наступний крок (черга):** B2 `PH-203c` AVX2 (x86 runtime-detect) поверх SSE2-dispatch **або** B6
cross-builds (`PH-601..604`, потрібні `rustup target add` msvc/aarch64 + NDK) **або** llama-rs
**semantic** parity (зовнішній оракул) **або** B7 `PH-701` docker-slave. `PH-405` numeric gate — **done**.
План — `docs/roadmap.md`, `NEXT_SESSION_PROMPT.md`.

**Session-wire fix (2026-10-15):** вчорашнє «agi не працює» = фантомний проєкт OpenCode
Desktop `rusty-engine(moe)` → неіснуючий шлях `CoreBoard_Labs_shop\rusty-engine(moe)`
(ENOENT/Aborted у лозі), а не помилка коду. Додано проєктний `opencode.json` (kit-skills →
GSV `.agents/skills`, hub-MCP stdio `target/live/gsv-mcp.exe`, instructions GSV+local+NEXT);
ендпоінти підтверджив: hub omni `:9999/api/omni/v1` 200+SSE, llama-rs `:8080/:8082` 200;
OmniRoute `:20128` policy-down. Owner-згодою створено **private** GitHub-remote
`platinoff/rusty-engine` і запушено цей fix-comit; MIT-публікація коду — далі за B11.
Деталі — `NEXT_SESSION_PROMPT.md` § «Session wire».
