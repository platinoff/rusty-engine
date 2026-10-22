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

### Продовження 4 — f32-matmul: три форми циклу, один результат

Спробувано три переструктурування батч-`matmul` (обидва — з AVX2+FMA, відкочено якщо не дали
виграшу): (1) per-block `dot` + hsum — базовий; (2) вектор-акумулятор на рядок + hsum раз на
(weight-row, input-row); (3) row-parallel GEMM — транспонувати `X[rows×n] → xt[n×rows]`, кожен
скалярний ваговий елемент × неперервний відрізок рядків. Заміри на ~700-токенному промпті:
**80 / 75 / 73 с**, і *той самий базовий* код через кілька прогонів знову дав **88–90 с** — тобто
розкид **±10 %** між прогонами (фонові збірки/навантаження) більший за різницю форм. **Висновок:**
f32-`matmul` нечутливий до форми циклу (~9 GFLOPS незалежно); обидва експерименти **відкочено**
(`git checkout -- src/kernel.rs`; дерево чисте на `9878b03`). Реальний наступний крок — **int8
активації** (ggml `vec_dot`: `_mm256_maddubs_epi16`+`madd_epi16`, без `set1`/конверсій на елемент),
але це **змінює числівник** → треба пере-базувати golden як окрему «llama.cpp-parity» (PH-405b)
гілку, не чіпаючи f32-golden.

**13 локальних комітів**, без push (B11).

### Продовження 5 — int8 vec_dot: реалізовано, виміряно, ~1.8–2× ПОВІЛЬНІШЕ; відкочено

Реалізовано ggml-style int8-активації (окрема гілка за `RUSTY_INT8=1`, за замовчуванням **off** —
щоб f32-golden лишався чинним): `quant::f32_to_f16` + `quant::quantize_row_q8_0` (`d=amax/127`,
`q=round(x/d)`); `kernel::dot_{q8_0,q5_0}_q8_0` (AVX2 `_mm256_maddubs_epi16`+`_mm256_madd_epi16`;
для q5_0 — беззнакові `0..31` + поправка `−16·Σx`); `kernel::matmul_{q5_0,q8_0}_q8` з
**block-outer reuse** (кожен блок ваг декодується раз і переюзується по всіх рядках активацій — та
сама схема, що в f32 `matmul_reuse`); `QMat::matmul_q8`/`QMat::int8_enabled`. Юніт-тести
(AVX2 vs scalar, int8 vs f32 у межах допуску) проходять; golden на дефолтному f32-шляху **точний**
(регресії немає).

Заміри на **тій самій** release-збірці, ~700-токенний промпт і chat8:

| шлях | prefill ~700 tok | decode (chat8) |
|---|---|---|
| f32 (дефолт) | **~81 с** | **~3.07 с** |
| int8 (`RUSTY_INT8=1`) | ~143 с | ~6.0 с |

Тобто int8 **у 1.7–2× повільніший** (наївна версія — декод блоку на кожен елемент — була ще гірша:
>180 с/таймаут; block-outer reuse підняв до 143 с, але все одно програш). **Причина:** у цієї моделі
блоки квантування — **32 елементи** (Q5_0/Q8_0, `d_model=896`), тож на блок припадає `hsum` +
множення на scale + розкиданий доступ до `qx`; це дорожче за f32-шлях із векторним dequant+FMA, який
уже амортизує декод ваг через block-outer reuse. Перевага int8 у кількості MAC-інструкцій на
32-елементному блоці не матеріалізується. **Відкочено** (`git checkout -- src/{kernel,qmat,quant}.rs`;
дерево чисте; сервер перезібрано з відкоченого джерела). Щоб int8 справді виграв, потрібен
**репак/ре-квантування ваг при завантаженні** у блочний layout з 256-елементними груповими scale
(менше `hsum`/scale-операцій) — велика зміна з непевним виграшем; наразі не робиться. f32-шлях
лишається оптимальним; інший вектор — `PH-405b` (семантичний паритет з llama.cpp) або кросс-збірки
(B6/B7).

**15 локальних комітів**, без push (B11).

### Продовження 6 — PH-405b: forward-parity з llama-rs ВИПРАВЛЕНО (биті біаси + RoPE)

Гейт `tests/moe_parity.rs` уже існував (Гейт A — tokenizer byte-exact; Гейт B —
greedy-parity проти живої llama-rs), але Гейт B **розходився на першому ж токені**
(префікс-збіг = 0, аргмакс — сміття), а `min_prefix` був 0.

**Знайдено дві систематичні помилки forward-стека** (вхід був правильний:
токенізація byte-exact, геометрія валідна, logits finite):

1. **Пропущені q/k/v-біаси Qwen2 — головна причина.** GGUF несе
   `blk.N.attn_{q,k,v}.bias` (291 тензор = 219 + 3×24), `model.rs` їх лише
   *валідував*, а forward їх не додавав (у `BlockWeights` не було полів). Qwen2
   (на відміну від Llama) має qkv-біаси — без них вихід шумовий. Фікс:
   `BlockWeights.{q,k,v}_bias` + `add_bias`/`add_bias_rows` у всіх attention-шляхах;
   `runner.rs` тягне біаси **опційно** (порожні для Llama-style моделей).
2. **RoPE-mapping.** `infer::rope` робив GPT-J interleaved (пари `2i,2i+1`), тоді як
   Qwen2/llama.cpp використовують **NeoX rotate-half** (пари `i, i+d/2`). Фікс:
   rotate-half (коментар обіцяв «llama.cpp convention», а код був GPT-J).

**Результат (офлайн-гейт `greedy_text_against_recorded_oracle`, без мережі)** — наш
greedy-стрім збігається з **записаним** оракулом байт-у-байт на всіх трьох рядках:

| prompt | oracle | rusty | lcp |
|---|---|---|---|
| `The capital of France is` | ` Paris. It is the largest city in Europe and the second` | той самий | **55/55** |
| `2+2=` | `4, 3+3=6, 4+` | той самий | **12/12** |
| `Berlin is a` | ` city in the state of Berlin, Germany. It is the` | той самий | **48/48** |

Floor-и `PARITY_CASES` піднято до повної довжини мінус запас (52/10/45). Гейт A —
PASS (vocab 151936, merges 151387, ids точні). Числовий відпечаток
`tests/data/qwen25_golden.txt` **перезахоплено** — старі «шумові» токени
(`127498 27277 …`) замінено на зв'язні (`7063 8749 14358 …`). Факти —
`docs/research/llama-rs-parity.md`. Перевірено: `cargo test` (усі ok), clippy чисто,
`cargo test --release -- --ignored` (moe_parity 3 ok, numeric_parity ok). Живий
`rusty serve` на `0.0.0.0:8090` після фіксу віддає **зв'язний текст** (chat8:
`I'm trying to create a simple`).

**16 локальних комітів**, без push (B11).

### Продовження 7 — PH-507: hub-провайдер `rusty-serve` працює E2E (кривий JSON + читання тіла)

Через GSV-хаб (`POST /api/omni/v1/chat/completions`, провайдер `rusty-serve`,
модель `rusty-05`) відповідь приходила як **`null`**, хоч прямий виклик `rusty serve`
на `:8090` віддавав коректний OpenAI-обʼєкт. Діагностика (canned-listener `:8123`,
stub-serve, окремий діагностичний хаб `:9998` з `GSV_OMNI_DEBUG`, перехоплення байтів
upstream-відповіді):

**Корінь — невалідний JSON у `textgen::chat_response`.** Формат-рядок закривав
обʼєкт `usage`, але **не закривав корінь** — у тілі бракувало однієї `}` (4 `{`, 3 `}`).
`curl` шле байти без валідації, а хаб робить
`serde_json::from_slice(&bytes).unwrap_or(Value::Null)` (`GSV/src/boxes/omni/proxy.rs`)
→ `null` при HTTP 200. Помилка парсера: `EOF while parsing an object at line 1 column 256`.

- `textgen.rs`: виділено `chat_completion_body(content, prompt_len, generated)`,
  додано пропущену кореневу `}`; тіло тестується строгим std-only `worker::parse_json`.
- `serve.rs`/`http.rs`: читати тіло за оголошеним `Content-Length` (раніше зупинка на
  першому `\r\n\r\n` губила розділені head/body); кап піднято до 1 MiB з ростом буфера;
  додано `http::content_length`.

**E2E перевірка (живий хаб `:9999` → serve `:8090`):**
`The capital of France is` → ` Paris. It is the largest city in` (HTTP 200, валідний JSON);
`stream:true` → SSE-чанки пасструузятся хабом (`x-gsv-mcp: sse`). 178 lib-тестів ok,
clippy чисто. Лишається в межах PH-507: WiFi multi-host (band 233) і провайдер `rusty/*`
у OpenCode/Cursor.

**17 локальних комітів**, без push (B11).

### Продовження 8 — B2 SIMD bench (`PH-205`) + закриття x86-диспатчу (`PH-203`)

`benches/quant.rs` (новий, **std-only**, без `criterion`, `harness=false`) міряє scalar-оракул проти
`*_dispatch` для кожного quant-ядра і друкує ns + speedup + `|Δ|` (SIMD має збігатися зі скаляром).
`Cargo.toml` отримав `[[bench]] name = "quant"`.

**Заміри (release, Qwen-форма 896²):** dot `q4_0` **2.32×**, `q8_0` **1.78×**, `q5_0` **2.12×**,
`q4_k` **3.94×**, `q6_k` **5.31×**; matvec `q4_0` **2.30×**, `q8_0` **1.78×**; `|Δ| ≤ 4e-3`
(SIMD==scalar до f32-округлення).

**Граблищі — у фікстурах, не в ядрах.** Перші прогони дали `|Δ|` до сотень і **NaN**. Причини: (1)
`fill_floats` ділив xorshift на 4096 замість 2²⁴ → значення до ~4096 (не ±0.5); (2) масштаб блоку
`d` (f16) залишався random-байтами → Inf/NaN; (3) **`Q6_K` тримає `d` у хвості блоку, offset 208**,
а не 0. Після стампу фінітних `d`/`min` (і правильного offset) — тихо.

**Інвентар SIMD (x86 закрито):** SSE2 `Q4_0`/`Q8_0`; AVX2 `Q4_0`/`Q5_0`/`Q8_0`/`Q4_K`/`Q6_K`;
`dot/matvec/matmul_*_dispatch` ланцюг AVX2→SSE2→scalar, підключений до forward через `qmat.rs`
(`QMat::matvec/matmul`). **Лишився тільки NEON (arm64)**, тому `PH-203` тримає `[~]`.

Gate: `cargo test` **197 green** (178 lib + 5 dense-gen + 4 scaffold + 5 serve + 5 text-gen, +5 ignored),
clippy `--all-targets` 0, fmt ok. **23 локальних комітів**, без push (B11).

### Продовження 9 — PH-505 residual: per-process RAM у `/metrics`

`/metrics` віддавав tok/s + experts, але не RAM. Додано **`hub::ram_bytes()`** (std-only, без
крейтів): Windows — working set через `K32GetProcessMemoryInfo` (`extern "system"` FFI,
kernel32/psapi), Linux — resident-сторінки з `/proc/self/statm` × 4096, інакше 0. `Metrics::to_json`
семплить живий RSS → у відповідь додано `ram_bytes` та `ram_mb`. `ChatOutcome` отримав поле
`active_experts`, яке `serve` передає в `metrics.record(..)` (на dense-шляху 0 — наповниться з
`PH-406` MoE).

Перевірка: unit `ram_bytes_reads_the_process_where_supported` + `metrics_json_shape`; live `:8090` →
`"ram_bytes":1502031872,"ram_mb":1432.4` (модель 0.5B завантажена); окремий прогон на `:8095` (stub)
→ `ram_mb 4.6`. **Увага:** живий serve `:8090` (старий PID 9992) встиг впасти між сеансами — піднято
заново (`target/live/serve.sh`, PID 13844). Порт `:8091` належить poolAI (не наш).

Gate: `cargo test` **198 green** (179 lib + 5 dense-gen + 4 scaffold + 5 serve + 5 text-gen, +5 ignored),
clippy 0, fmt ok. **24 локальних комітів**, без push (B11).

### Продовження 10 — PH-203d: hot AVX2-ядра переведено на FMA

Скалярні передумови: три найгарячіші `dot_*_avx2` робили `mul_ps` + `add_ps` в **один** акумулятор
(довгий ланцюг залежності по `add`, latency ~4 цикли на кожен), а Q8_0 ще й збирав ваги через
скалярний `[i32;8]`-цикл (`qs[jj+t] as i8 as i32`).

Що змінено (`src/kernel.rs`):

- **Q8_0**: замість скалярного гезеру — `_mm_loadl_epi64` + `_mm256_cvtepi8_epi32`; множення й
  накопичення злито у `_mm256_fmadd_ps` у **два** акумулятори (jj та jj+8). Оскільки `d` стала на
  блок (34 B), `d·(q·b)` складається коректно; FMA дає одне округлення замість двох.
- **Q4_0 / Q5_0**: low/high половини розведено на два акумулятори (`acc0`/`acc1`), `mul`+`add`
  замінено на `fmadd`, напр. `acc0 = _mm256_fmadd_ps(lov, loadu(b+bo+jj), acc0)`.
- **Диспатч і явні parity-тести** гейтовано `avx2 && fma` (раніше лише `avx2`); FMA присутня на
  кожному AVX2-хості з Haswell (2013). Q4_K/Q6_K ядра не чіпали (лишились `avx2`).

Виміри (`benches/quant.rs`, 5 прогонів, best-of, dot 8192 елементів / matvec 896²):

| kernel | було (dispatch) | стало | speedup vs scalar |
|--------|-----------------|-------|-------------------|
| dot_q8_0 | 6740 ns | **5185 ns** | 1.72× → **2.09×** |
| dot_q4_0 | 5584 ns | 5348 ns | 2.31× → 2.41× |
| dot_q5_0 | 6388 ns | 6124 ns | 2.14× → 2.30× |
| matvec_q8_0 | 604 984 ns | **518 189 ns** | 1.77× → **2.06×** |

`|Δ|` проти скаляра **без регресії** (q8_0: 1.34e-3 → **9.8e-4** — FMA точніший). Q8_0 — це LM-head
(`output.weight`), тож виграш видно на живому шляху: decode `:8090` **2.27 → 2.68 tok/s**
(Qwen2.5-0.5B Q4_K_M).

Інфра-нотатки сеансу: живий `:8090` знову впав між сеансами — перезапуск робив через
`Invoke-CimMethod Win32_Process Create` з `cmd.exe /c "cd … && target\release\rusty.exe serve …"`
(короткий `bash -lc` під PowerShell дає job-kill після завершення tool-call; WMI+cmd переживає).
Живий хаб `S:/rust/GSV/target/live/gsv-server.exe` (`:9999`) не чіпали.

Gate: fmt 0, clippy `--all-targets -D warnings` 0, **198 тестів** green. **25 локальних комітів**
(`c2102bb perf(kernel): AVX2+FMA fused dots for Q4_0/Q5_0/Q8_0 (PH-203d)`), без push (B11).

### Продовження 11 — PH-203e: проєкції паралеляться по рядках

Двигун декодував на одному ядрі з 12 (Ryzen 5 5500U). Рядки `W[m×n] · x[n]` незалежні й кожен — та
сама скалярна послідовність, тож їх можна роздати воркерам без зміни результату.

- **`src/qmat.rs`**: `QMat::matvec` ділить `m` рядків, `QMat::matmul` ділить **рядки ваг**
  (вихідні стовпці) — і збирає блоки назад у row-major `Out`. Поріг `PAR_MAC_MIN = 2¹⁸`;
  воркерів дає `RUSTY_THREADS` (1–256) або `available_parallelism()`. Старі тіла —
  `matvec_serial`/`matmul_serial` (fallback: F32, типи без fused-ядра, малі проєкції).
- **`src/kernel.rs`**: `quant_row_bytes`, `quant_row_dot`, `matmul_dispatch` — щоб qmat не
  дублював геометрію блоків по типах.
- **Пастка, яку зловили виміром:** перша версія ділила `matmul` по вхідних рядках. `matmul_reuse`
  де-квантизує блок один раз і переюзує на всі вхідні рядки — тому кожен воркер де-квантизував
  **усю** матрицю, і prefill не прискорювався взагалі (186 с на 961 токен, як до змін). Після
  поділу по рядках ваг: 321 токен 45.6 с → 24.7 с.
- **Біт-у-біт:** `parallel_matvec_is_bit_identical_to_serial`,
  `parallel_matmul_is_bit_identical_to_serial` (порівняння з `*_serial`).
- **Замір:** prefill A/B `RUSTY_THREADS=1` 45.6 с vs `12` 24.7 с (**1.85×**); decode
  2.68 → **4.2–5.1 tok/s** (розкид — сусідні GSV-сервіси тримають ~60% CPU). Стеля 1.85× —
  це 6 фізичних ядер під навантаженням, не 12.
- **Інфра:** живий `:8090` піднімається **без вікна**: `wscript.exe` (GUI-підсистема, без консолі)
  → `cmd /c` з `sh.Run(cmd, 0, False)` (window style 0). Скрипти: `target/live/launch_serve.vbs`,
  `spawn_wscript.ps1`, `spawn_threads.ps1` (читає `target/live/threads.txt` → `RUSTY_THREADS`).
  WMI `Win32_ProcessStartup.CreateFlags` дає rc=21 на цій машині — не використовувати.

Gate: fmt 0, clippy `--all-targets -D warnings` 0, **200 тестів** green. **28 локальних комітів**
(`e2646a4` parallel + `6336afb` matmul weight-row fix), без push (B11).

### Продовження 12 — PH-203f: постійний пул воркерів замість `thread::scope`

Спершу виміряли ціну попереднього рішення: `thread::scope` з 12 потоками коштує **750–820 µs**
spawn+join (секція в `benches/quant.rs`), а проєкцій на токен ~120 → **~90 ms/токен** тільки на
створення потоків. Це більше, ніж сама проєкція (0.68 ms), тож пул був обґрунтований числом, а не
інтуїцією.

- **`src/pool.rs`** (std-only, без rayon): воркери живуть увесь процес, робота — черга на
  `Mutex<VecDeque<Job>>` + `Condvar`; глобальний `pending` рахує незавершені job-и, і
  `parallel_for(jobs, |i| …)` повертається лише коли `pending == 0`.
- **Позичення через `Handle`:** `transmute` замикання в `'static` не компілюється (вимагає
  `F: 'static`). Тому `Handle` тримає тип-стертий `*const ()` + мономорфізовану
  `unsafe fn(*const (), usize)`; безпечність тримається на тому, що `wait()` дочікується **всіх**
  job-ів до повернення. Панель-паніки — `catch_unwind`, інакше воркер і `pending` гинуть назавжди.
- **`QMat::matvec`/`matmul`** збирають блоки через `Mutex<Vec<(usize, Vec<f32>)>>`; кількість
  job-ів = `m.div_ceil(chunk)`, **не** `threads` — інакше останній воркер отримує `base > m` і
  панікує на `out[base..]` (зловили тестом `parallel_matvec_is_bit_identical_to_serial`).
- **Ніколи не вкладати** `parallel_for` у job (сам себе чекатиме) — записано в doc-comment.
- **Замір:** gate/up 4864×896 Q5_0 matvec 1.270 → **0.684 ms** (4.3–5.1× vs serial, було 2.66×);
  prefill matmul 4864×896×961 1009 → **247 ms (4.08×)**; live decode 4.4 → **7.2–7.9 tok/s**
  (накопичувально 2.27 → 7.8 ≈ **3.4×**). Біт-у-біт збережено.
- **Prefill (321 tok) не зрушив** (~25 с до і після): matmul тепер 4.08×, але ~19 з 25 с — це
  послідовний attention/KV. Це наступна ціль, а не кернели.
- **Виміри на цьому боксі шумні:** свіжий `:8090` дає стабільні 7.2–7.9 tok/s, а той самий процес
  після важкого prefill/`cargo bench` — 2–4 tok/s (сторінки моделі витісняються). Міряти decode
  лише після перезапуску сервера і кількома послідовними викликами.
- **Тести:** `parallel_for` індекси/join, inline-випадок (1 job), панікуючий job — плюс наявні
  біт-ідентичні тести тепер покривають і пул.

Gate: fmt 0, clippy `--all-targets -D warnings` 0, **203 тести** green. **30 локальних комітів**
(`72b570a perf(pool): persistent worker pool for parallel projections (PH-203f)`), без push (B11).

### Продовження 13 — PH-203g: RoPE раз на позицію, KV зберігає повернуті ключі

Prefill стояв на 25 с для 321 токена навіть після того, як matmul став 4.08× швидшим. Причина
була не в attention-математиці, а в RoPE: `block_forward_prefill_cached` викликав `rope()` на
кожну пару (query, key) — ~723k разів, — і кожен виклик рахував `powf` **на елемент** плюс
алокував `Vec`. Decode страждав тим самим, переобертаючи кожен кешований ключ на кожну голову.

- **`src/infer.rs`**: `rope_freqs(d, base)` — таблиця частот один раз на модель;
  `rope_in_place(head, pos, freqs)`; `rope()` тепер обгортка над ними (f32-результат той самий).
- **`src/forward.rs`**: `LayerKv` зберігає **вже повернуті** ключі. Контракт змінено з «raw,
  rotate on read» на «rotated, read as-is»; оновлено обидва письменники
  (`attention_forward_cached`, `block_forward_prefill_cached`) і єдиного читача. Prefill
  повертає цілі Q/K-рядки раз на позицію, а не раз на голову на query; один scratch-буфер для
  scores/probs замість `Vec` на кожен query; V-накопичення `for j { for d }` (порядок сум на
  елемент збережено, внутрішній цикл суміжний).
- **`src/moe.rs`**: `softmax_in_place` (та сама послідовність операцій), `softmax` кличе його.
- **Пастка:** при переписуванні `block_forward_seq` загубився `add_bias_rows(&mut v_all,
  v_bias)` — усі швидкі тести були зелені, а зловив це `PH-405` fingerprint
  (`cargo test --release -- --ignored`). **Висновок: після будь-якої зміни в forward завжди
  ганяти `--ignored`**, не лише `cargo test`.
- **Замір:** prefill 321 tok 25.5 → **5.89 s** (4.3×), 961 tok 190.9 → **29.1 s** (6.6×);
  decode 7.4 → **10.35 tok/s** (накопичувально 2.27 → 10.35 = 4.6×).
- **Що лишилось у prefill:** matmul + послідовний прохід по головах. Кандидати:
  head-parallel attention через `pool::parallel_for`; окремо — q8_0-ядро (серial matvec 896²
  = 0.55 ms ≈ 1.5 GMAC/s, підозріло мало для AVX2).

Gate: fmt 0, clippy `--all-targets -D warnings` 0, **203 тести** green (+ усі `--ignored`).
**32 локальні коміти** (`5f1641b`), без push (B11).

### Продовження 14 — PH-203h: head-parallel prefill attention

- **`src/forward.rs`**: `attention_head_ctx(q_all, k_rows, v_rows, g, seq, pos_start, h)` —
  єдиний per-head кернел. Його використовують **обидва** prefill-шляхи: `block_forward_seq`
  (prompt-local K/V) і `block_forward_prefill_cached` (KV-кеш), бо layout однаковий
  (`pos_end × kv_dim`, рядки вже повернуті RoPE). Два дубльовані inline-цикли прибрано.
- **`attention_prefill(...)`** збирає голови в `seq × q_dim`; для `seq >= PAR_SEQ_MIN = 32`
  роздає їх через `pool::parallel_for`, нижче — серially (hand-off у пул дорожчий за роботу).
  Голови пишуть у власні `head_dim`-блоки, тому біт-у-біт збіг з серial збережено.
- **Замір:** prefill 961 tok 29.1 → **23.5 s** (1.24×; від початку сесії 190.9 s = 8.1×),
  321 tok 5.89 → 5.56 s; decode без змін (10.4–10.9 tok/s).
- **Підсумок перф-серії за сесію:** decode 2.27 → **10.9 tok/s** (4.8×), prefill 961 tok
  190.9 → **23.5 s** (8.1×). Коміти: `c2102bb` FMA, `e2646a4`/`6336afb` row-parallel,
  `72b570a` pool, `5f1641b` RoPE/KV, `fecffa1` head-parallel.
- **Що лишилось у prefill:** matmul (17 GMAC/s). Найбільший резерв — q8_0-ядро:
  serial matvec 896² ≈ 0.55 ms ≈ 1.5 GMAC/s, що мало для AVX2 (конвертація i8→f32, немає
  int8-множення). Це змінює числову схему → окремий зріз з оракулом і `RUSTY_CAPTURE_GOLDEN`
  за потреби.

Gate: fmt 0, clippy `--all-targets -D warnings` 0, **203 тести** green + усі `--ignored`.
**34 локальні коміти** (`fecffa1`), без push (B11).

### Продовження 15 — 2026-10-21: синк доків/боксів + push (старт наступної сесії)

- Власник 2026-10-21 зняв B11-обмеження: **сесію закрито push-ом** — `origin/main` = `ef0378a`
  (**35 комітів**, PH-203f/g/h + доки). Далі звичайний контур: один коміт на зріз + `git push`
  у кінці сесії; MIT/license/CHANGELOG лишаються за B11 (`PH-1105`).
- GSV-бік (2026-10-21): бокси синхронізовано зі станом репо — `cargo test` **480 s ok**
  (41 `gsv_watchdog_contracts`), clippy **0/0**, `docs/vision/speed_index.json` +
  `rust_diagnostics.json` перезаписані на `36e5eca`, vision rev **516**, drift `ok`. Знайдено й
  виправлено контракт-баг `debug_newer_telenetis_compares_telenetis_crate` (фікстура без
  `telenetis/Cargo.toml` → `telenetis_crate_dir` свідомо йшла в sibling; тест проходив лише з
  випадково виставленим `CARGO_TARGET_DIR`).
- Наступна сесія (GSV + двигуни): тікети `t-1792585619484828300` (екосистема: `rusty-serve` у
  `routing.fallback_order` хаба) і `t-1792585623805073700` (двигуни: q8_0 integer-dot kernel).
- **Рантайм на момент синку був мертвий:** `rusty.exe` не існувало в процесах, `rusty_heartbeat.json`
  стояв з 2026-10-19 → keep-live показував rusty down. Піднято **windowless** за каноном
  (`target/live/spawn_threads.ps1 -Threads 12` → WMI `wscript.exe launch_serve.vbs`, `RUSTY_THREADS=12`);
  `GET :8090/health` = 0.1.0, `/v1/chat/completions` відповідає, heartbeat свіжий. Telenetis теж був
  down — піднято так само через новий `telenetis/target/live/launch_live.vbs` (wscript, без консолі);
  `:9800/health` = 0.2.0. keep-live після цього: gsv + telenetis + llama_rs + rusty up, omniroute
  policy-down. **Не запускати `cmd.exe /c` через WMI напряму** — з'являється консоль.

### Продовження 16 — 2026-10-21: PH-203i — f16-widening + q8_0 integer-dot probe

- **Справжній ботлнек «підозріло низьких» 1.5 GMAC/s знайдено:** `quant::f16_to_f32` рахував
  через `f64` + `powi` на **кожен блок** (32 значення для q8_0, 256 для K-quant). Замінено на
  бітове розширення (`sign/exp/frac → f32::from_bits`); subnormal/Inf/NaN збережено, тест
  `f16_bit_exact_against_f64_reference` перевіряє всі 65 536 `u16`. Числова схема **не змінилась**:
  `cargo test --release -- --ignored` (dense_real, 3× moe_parity, numeric_parity) зелений без
  `RUSTY_CAPTURE_GOLDEN`.
- **Бенч (`cargo bench --bench quant`, release):** dot_q8_0 5.34 → **0.64 µs** (9.55×), matvec 896²
  527 → **68.5 µs = 11.8 GMAC/s**; dot_q4_0 2.03×→9.06×, dot_q5_0 2.48×→5.38×, q4_k 3.92×→7.63×,
  q6_k 5.36×→8.15×; matvec q4_0 dispatch 8.60×; QMat prefill matmul 179 → **156 ms** (5.44× над
  serial), gate/up Q5_0 0.968 → 0.309 ms.
- **Live `:8090` (свіжий рестарт, `RUSTY_THREADS=12`):** decode **27.6 tok/s** (64-токен probe:
  64 / 2.316 s; було 10.9) — фінальна q8_0-проєкція більше не домінує. Prefill 961 tok
  **23.46 s** (було 23.5 — q8_0 задіяний лише на останній позиції), 321 tok 5.23 s.
- **q8_0 integer-dot (за тікетом):** `Q8Activation` (f32 scale на 32 + i8), scalar `dot_q8_0_i8`,
  SSE2/AVX2 (`cvtepi8_epi16` + `madd_epi16`, `dot_q8_0_i8_{sse2,avx2,dispatch}`),
  `matvec_q8_0_i8`/`matmul_q8_0_i8`; SIMD **біт-у-біт** з scalar-оракулом. Після f16-фіксу
  int8-шлях **повільніший** за float (0.89×: 76.7 vs 68.3 µs на matvec 896²), тому в `QMat`
  **не підключений** — лишається протестованою альтернативою (i8-кеші активацій / майбутній
  int8-matmul); forward-числа не змінені.
- Gate: fmt 0, clippy `--all-targets -D warnings` 0, **193 lib + 19 інтеграційних** зелені,
  `--ignored` 5/5. **36 локальних комітів**, push у кінці сесії.

### Продовження 17 — 2026-10-21: row-batched prefill matmul — виміряно, 3–4× ПОВІЛЬНІШЕ; відкочено

- Гіпотеза: `matmul_reuse` кличе `dot_f32_dispatch` на кожен (блок, рядок) → одна 4-FMA-ланцюгова
  залежність + hsum на 32 MAC; якщо 4 вхідні рядки ділять один де-квантований блок у регістрах,
  ILP мав закрити латентність (без зміни порядку сум → біт-у-біт).
- Реалізовано `matmul_reuse_avx2` (4 рядки/блок, `p0..p3` у `__m256` поза циклом груп, той самий
  `hsum_ps_sse2`), плюс `matmul_reuse_rows` як scalar-оракул і тест
  `matmul_row_batched_is_bit_identical_to_scalar` (m/n/rows 3·64·6, 2·32·4, 4·64·5, 2·64·1) —
  **тест зелений**, тобто числа справді біт-у-біт.
- Замір (`cargo bench --bench quant`, один і той самий бінарник, prefill 4864×896×961): старий код
  (HEAD) **321 ms**, refactor-fallback **213 ms**, row-batched **573–648 ms**; serial-варіанти
  1609 / 1279 / 5818–6718 ms. Батч стабільно **3–4× повільніший** за fallback при тому ж числі
  FMA → гіпотеза про латентність не підтвердилась (як і §Продовження 4: f32-matmul нечутливий до
  форми циклу). Код **відкочено** (`git checkout -- src/kernel.rs`), дерево чисте.
- **Шум боксу:** та сама (незмінена) реалізація дала 156 ms у попередній сесії і 321 ms сьогодні —
  розкид ~2× через фонове навантаження; порівнювати можна лише прогін у прогін, не через сесії.
- Живий `:8090` перезібрано з відкоченого джерела й перезапущено (`spawn_threads.ps1 -Threads 12`):
  decode 23.7 tok/s (64 tok / 2.70 s), `/health` ok.
- **Висновок:** залишок prefill не лікується перестановкою циклів; наступний кандидат — або
  dequant-once у блоці 256 (K-quant) / інший layout ваг (repack PH-202), або вимірювання частки
  attention у 23.5 s перед подальшими змінами.

### Продовження 18 — 2026-10-22: PH-203k — активаційний tiling + row-split проєкцій (знайдено справжній ботлнек)

- Розбір після §17: `matmul_reuse` сканує **весь** `X` (961×896×4 = 3.3 MiB) на кожен ваговий
  рядок, тобто 4864 проходів ≈ 16 GiB з L3 на одну проєкцію. Замір скаляром підтвердив: та сама
  вага на 32 рядках дає ~6.8 GMAC/s, на 961 — ~3.2 GMAC/s, тобто ціна за MAC росте з розміром
  `X` (не «форма циклу», а трафік).
- **`src/kernel.rs`:** `matmul_reuse` отримав параметр `tile` і обходить `X` тайлами по
  `MATMUL_ROW_TILE = 16` рядків (57 KiB — L2). Порядок додавання блоків на `(row, r)` той самий →
  **біт-у-біт**; тест `matmul_activation_tiling_is_bit_identical` (tile 1/3/31/32/64/128 == один
  тайл, 70 рядків = 2 тайли + хвіст). `matmul_q5_0_tiled` — бенч-хвіст для A/B (production
  завжди через константу). Бенч (min of 3, один процес): untiled 1469 / 1114 ms → tile 16
  **897 / 1103 ms** (~1.2–1.35×), tile 32 проміжний, tile 128 майже як untiled.
- **`src/qmat.rs`:** `QMat::matmul` ділив роботу по **стовпцях ваг** — усі 12 воркерів годували
  один і той самий 3.3 MiB `X` у L3 (≈16 GiB сумарного потоку, проєкція впиралась у L3). Тепер
  ділиться по **вхідних рядках**: кожен воркер тримає свої ~290 KiB `X` (L2) і перечитує ваги
  `threads` разів (≈36 MiB) — на два порядки менше трафіку; рядки в `Out` суміжні, копіювання
  без scatter. Старий варіант лишено як `matmul_split_cols` (`#[doc(hidden)]`) для A/B.
- A/B у одному процесі: **cols 247.6 → rows 190.0 ms (1.30×)**, повтор 344.3 → 291.3 (1.18×),
  третій (бокс під навантаженням) 363.5 → 356.6 (1.02×). Напрям стабільний, величина — в межах шуму.
- **End-to-end не заявляємо:** живий prefill 961 tok = **26.9–27.3 s**, але бокс завантажений
  (rebook + 4 llama-процеси + gsv/watchdog/telenetis), тоді як сьогодні раніше було 25.0 s, а в
  §16 — 23.46 s. Порівнювати через сесії не можна (див. §17).
- Побічна знахідка: **rebook слухає `127.0.0.1:8090`** і перехоплює loopback-проби, тоді як
  `rusty.exe` слухає `0.0.0.0:8090`; health/probe треба гнати на LAN-адресу
  (`bash target/live/prefill_probe_lan.sh`) або інший порт. Не лікувати — це чужий продукт.
- Гейт: `cargo fmt --all` ok, **194 lib** + integration green (включно з
  `qmat::parallel_matmul_is_bit_identical_to_serial`), `clippy --all-targets -D warnings` 0.
- **Далі:** залишок prefill — це вже не layout, а латентність `dot(32)`+hsum на блок (повний
  f32-акумулятор по рядку з одним hsum змінив би порядок сум → окремий числовий зріз з
  recapture); альтернативи: dequant-once 256 для K-квантів, repack layout (PH-202), виміряти
  частку attention у 23.5 s.


### Продовження 19 — 2026-10-22: PH-203l — full-row f32 акумулятор (числовий зріз, golden не зрушив)

- Після §18 вузьким місцем лишалась **латентність на кожен `(row, r)`**: `matmul_reuse` кликав
  `dot` на кожен 32- (Q4_0/Q5_0/Q8_0) або 256-елементний (K-кванти) блок, тобто на 896-широкому
  рядку — 28 викликів з dispatch і 28 hsum, а кожен hsum це серійний 5-оп ланцюг на 4-FMA залежності.
- **`src/kernel.rs`:** ваговий рядок тепер дейквантизується **цілком** (`dequant_row` у буфер `n`
  f32, ті самі блокові `dq`), після чого один `dot_f32_dispatch(&rowbuf, x_row)` на рядок.
  Обхід `X` тайлами лишився, але результат від тайла більше не залежить узагалі.
- Старий шлях збережено як `matmul_reuse_blocks` + бенч-хвіст `matmul_q5_0_blockwise`
  (`#[doc(hidden)]`): це оракул для нового тесту і пара для A/B (новий хвіст — `matmul_q5_0_tiled`).
- **Числовий зріз:** порядок сум інший (один 8-смуговий акumulator по всьому рядку проти
  hsum-на-блок + f32-додавання). Тест `matmul_row_wise_accumulator_matches_block_wise`
  (9×896, 37 рядків) тримає ≤1e-4 відносної; фактична розбіжність на першому елементі
  −1.0500221 vs −1.0499992 (~2.2e-5).
- **Golden не перезахоплювався — і не потребував:** `cargo test --release -- --ignored`
  (qwen25 fingerprint re-prefill **і** incremental/serve, `greedy_text_against_recorded_oracle`,
  `tokenizer_json_byte_exact_gate`, `dense_runner_loads_real_qwen`) — усе green на новому
  акумуляторі. Це прямий доказ, що зріз семантично нейтральний. (`rusty.exe` перед release-тестом
  треба вбити: `taskkill //F //PID <n>`, інакше лінк у `target/release/rusty.exe` падає з os error 5.)
- Заміри `cargo bench --bench quant` (min-of-7, interleaved, один процес, `target/live/bench_ph203l.txt`):
  single-thread Q5_0 4864×896×961 **block 805.9 → row 606.3 ms (1.33×)**; serial-рядок бенча
  **859.1 ms** (було 1469 у PH-203j, 897–1103 у PH-203k); QMat parallel **92.5 ms**
  (= 45 GMAC/s, 9.29× над serial). **Tile sweep плоский:** 961→606.7, 128→616.4, 64→612.1,
  32→617.4, 16→610.8 ms — тайлінг став зайвим (рядковий `dot` читає `X` суміжно), це
  кандидат на спрощення наступним зрізом.
- Гейт: `cargo fmt --all -- --check` ok, **195 lib** + integration green,
  `clippy --all-targets -D warnings` 0.
- **End-to-end тепер заявляємо:** release перезібрано, `:8090` піднято
  `spawn_threads.ps1 -Threads 12`, проби на LAN (`bash target/live/prefill_probe_lan.sh`):
  961 tok = **18.93 / 19.04 s** проти **26.9–27.3 s** у §18 на тому ж завантаженому боксі
  (1.4×) і 23.46 s у §16 — це перший end-to-end виграш після PH-203h.
- **Далі:** (1) прибрати/спростити `MATMUL_ROW_TILE`, якщо flat-свіп підтвердиться; (2) 4-рядковий
  блок — один прохід ваги проти 4 рядків `X` (більше FMA на hsum, новий числовий зріз);
  (3) заміряти частку attention у prefill; (4) dequant-once 256 для K-квантів; (5) repack PH-202.
