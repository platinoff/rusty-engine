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

**Worker/APK роль + `build-matrix --build` (2026-10-16):** за owner-директивою
(`GSV_AGI_PATH.md` §worker/APK, коміт `ae83e24`) worker/APK роль перейшла від Telenetis до
двигунів — `rusty-engine` під id **`rusty-serve`**. Додано **`src/worker.rs`** (PH-706, новий
band B7-зріз): std-only hub-edge worker — `WorkerConfig` (LAN-first, `GSV_LOCAL_ADDR`/UDP-probe),
мінімальний `parse_json`/`parse_task`, `execute` (`ping` / `rusty_chat` → `serve_url`
`/v1/chat/completions`; dry-run без сокетів), білдери тіл, blocking `http_call`/`join_once`/
`heartbeat_once`/`poll_once` (header `x-gsv-edge-token`, origin `rusty_edge`, role `virtual_node`),
`plan_lines`+`mask` (токен ніколи не логується) — **17 unit**. CLI `rusty worker [--live] …`
(dry-run default, `--live` = register→join→heartbeat/poll loop). **`xtask build-matrix --build`**
(PH-605): гейтить `cargo build --release --target` по `rustup target list --installed`/host,
пропускає відсутні/невдалі cross-ланки з причиною, не валить матрицю (host gnu `rusty.exe` →
sha256 `48b1f196…`, 617 984 B; решта — skip). Gate: fmt + clippy `-D` 0 + **181 тест**, один
локальний коміт, **без push** (B11).

**Провайдер живий + perf (2026-10-16, продовження того ж дрain):** `rusty serve` піднято живим
(`rusty-serve` provider-row → `gsv_keep_live` **`rusty: alive`**). Ще два локальні коміти:
**`50cd422`** fix(serve) **PH-504** — keep-live heartbeat писався лише на startup+per-connection,
тож після 60 с простою гас (hub `host_ready("rusty-serve")=false`, доведено емпірично); додано
потік-тікер кожні `HEARTBEAT_TICK_SECS` (15 с). **`4359eb4`** perf(textgen) **PH-307** —
`TextRunner::run_text`/`complete` викликали `DenseRunner::generate[_with]`, які **re-prefillять
увесь префікс на кожен токен** (O(n²)); перемкнуто на наявний `DenseDecoder::generate_incremental`
(KV-кеш, O(n) на токен), вихід ідентичний (temp 0 == greedy argmax). Заміри на
`qwen2.5-0.5b-q4_k_m`: `run --max-tokens 4` **debug 91 с → release 6.4 с** (~14×, debug-білд був
головним фактором); `POST /v1/chat/completions` 8 токенів **>150 с** (зависав, однопоточний loop)
→ release+incremental **~5.3 с** стабільно (**~28×**). Лишковий ботлнек — `matvec` по
f32-де-квантованих вагах (≈2 ГБ RAM, ≈2 ГБ трафіку на токен → межа пам'яті), тобто ціль — band
**B2** (тримати ваги Q4_K/Q6_K + fused `matvec` + AVX2).

**B2 QMat + PH-203c AVX2 (2026-10-16, той самий drain):** **`99ec49a`** feat(forward) — новий
`src/qmat.rs` (`QMat::{F32,Quant}`, fused `matvec`-dispatch `q4_0/q5_0/q8_0/q4_k/q5_k/q6_k` + scalar
dequant-fallback, `from_model`); `BlockWeights` проєкції та `output` стали `QMat`, тож dense-forward
**більше не матеріалізує ~2 ГБ f32** ваг (serve RSS ~2 ГБ → ~540 МБ) і golden `PH-405` відтворюється
**точно** (quant-шлях == попереднє f32-золото). **`46cebf6`** perf(kernel) — **`PH-203c`**:
`dot_q4_0_avx2` / `dot_q4_k_avx2` / `dot_q6_k_avx2` (8 lanes, `_mm256_cvtepu8_epi32` над split-half
ниблами / per-group `d·sc`+`min·m` / 6-bit пакуванням), AVX2-гілка в `dot_q4_0_dispatch` + нові
`dot_q4_k/q6_k_dispatch` та `matvec_q4_k/q6_k_dispatch`; `QMat` типи 12/14 через них; parity-тести
avx2-vs-scalar. Заміри (Qwen2.5-0.5B Q4_K_M, цей box): serve 8-ток decode **~6.1 с → ~5.1 с** (~16 %);
O(n²) `DenseRunner::generate` parity **128.5 с → 102.5 с** (~20 %). Лишається ALU-bound — наступний
зріз: AVX2/FMA для Q5_K + батч-`matvec` (кілька рядків на прохід ваг) або B6 cross-builds / B7
docker-slave. **7 локальних комітів** за сесію, **без push** (B11).

**Факт-корекція форматів + `max_tokens` + Q5_0 (той самий drain):** `rusty inspect` на реальній
Qwen2.5-0.5B Q4_K_M показав, що 896-ширинні тензори **не** K-quant: `output` **Q8_0**,
`token_embd`/attn q·k·v·o/`ffn_gate`/`ffn_up` **Q5_0**, лише `ffn_down` (4864) **Q6_K** — тож
домінантний скалярний шлях це **Q5_0** (~51 % MAC/токен), а не Q4_K. Коміти: **`ca935d6`**
perf(serve) **PH-501c** — `serve` хардкодив `max_new=8` (кожен запит = 8 кроків ~5 с, незалежно від
`max_tokens`); додано `http::max_tokens_of` (clamp 1..=512, default 8) → 1-токен запит **5.09 с →
0.67 с**. **`fc308af`** perf(kernel) — `dot_q5_0_avx2` (+`_dispatch`/`matvec_*`, `QMat` type 6):
split-half ниббл + 5-й біт через `_mm256_srlv_epi32`; golden `PH-405` досі **точний**; decode
**~0.56 → ~0.33 с/токен**, 8-токен chat **5.06 → 3.01 с**, O(n²) parity-path **128.5 → 52.6 с**
(**2.4×**). **9 локальних комітів**, без push (B11). Наступне: `Q8_0` (output, ~28 % MAC) вже AVX2;
лишається FMA для Q5_0/Q6_K + батч-`matvec` (рядки на прохід ваг).

**FMA — відкинуто (негативний результат):** FMA-варіанти тих самих дотів (Q4_0/Q5_0/Q6_K,
`fmadd` замість mul+add) заміряно на parity-path: **52.6 → 52.2 с** (у межах шуму, ~1 %). Отже
bottleneck — не множення-додавання, а **int→float конверсія / трафік пам'яті**; FMA не дає виграшу
і лише додає вимогу `fma` — відкочено (`git checkout -- src/kernel.rs`). Сервер перезібрано з
відкоченого джерела й перезапущено hidden: decode стабільно **~0.33–0.35 с/токен** (chat1 0.68 с,
chat8 3.07 с, chat16 5.90 с). Наступний реальний важіль — **батч-`matvec` (prefill/GEMM)**: читати
ваги раз на кілька рядків замість разу на рядок.

**Batched prefill (зроблено).** Дві причини повільного промпту: (1) `block_forward_seq` кликав
`matvec` по рядку → кожен блок ваг де-квантувався `seq×`; (2) `serve` взагалі не робив батч-prefill —
`DenseDecoder::generate_incremental` годував промпт **по токену** через `block_forward_cached`
(`TextRunner::complete`/`run_text`). Додано:
- `kernel::matmul_{q4_0,q5_0,q8_0,q4_k,q5_k,q6_k}_dispatch` + `dot_f32_dispatch` (AVX2+FMA) —
  зовнішній цикл по рядках ваг, внутрішній по рядках входу: блок де-квантується **один раз** у
  малий f32-буфер і переюзується для всіх рядків (пам'ять O(1), MACs скалються з `seq`);
- `QMat::matmul(x, rows)` (F32 → по-рядковий fallback);
- `forward::block_forward_seq` тепер батчений (q/k/v/o + SwiGLU gate/up/down одним `matmul`);
- `forward::block_forward_prefill_cached` — батч-prefill, що **наповнює KV-кеш** (k/v сирі, RoPE на
  читанні) → `DenseDecoder::prefill`, який тепер кличе `generate_incremental`.
Заміри: re-prefill parity-path **52.2 → 19.2 с** (2.7×); serve на ~700-токенному промпті
**>180 с (таймаут) → ~80 с** (2.2×); decode без регресії (~0.33 с/токен). Golden `PH-405` точний —
і розширено: той самий golden тепер перевіряє **й інкрементальний/serve шлях**
(`fingerprint_incremental`). Лишається bound на пропускну здатність f32-MAC (`seq × ваги`); наступне —
**упакований int8 SIMD dot** (ggml-style `q8_0×q8` з `vec_dot`) або register-blocked GEMM.

**13 локальних комітів**, без push (B11).
