# HANDOFF_NEW_SESSION — rusty-engine

**Продукт:** `rusty-engine` · Сесійна мова: «концепція → роадмеп → тікети → drain».
**Host:** GSV hub (`S:/rust/GSV`, `:9999`) = master; це — portable plugin.

## Сталий стан (2026-10-24, перевірено в репо)

| Що | Значення |
|----|----------|
| HEAD / origin | PH-1304 `src/runner35.rs` (див. §Продовження 26–32 і `git log -1`), `origin/main` синхронний, `git status` clean; продакшн-forward = PH-203o, ядра matmul = PH-203r без змін; ваги MiniCPM-V 4.7-1B у `data/models/` (gitignored) |
| Гейт | fmt ok · clippy `-D` **0** · **244** тести (222 lib + 22 integration) + 7 `#[ignore]` parity (incl. real MiniCPM-V runner35 smokes), усі зелені |
| Код | Rust **17426** LOC (src 14177 / tests 1780 / xtask 319 / benches 1150), UI 157 → ratio **99.0 %** · 73 tracked files · docs 2781+ LOC |
| Prefill (live `:8090`, 12 потоків, 961 tok) | **5.3–6.4 s** (було 190.9 s на старті перф-серії; бокс дрейфує під фоновим Chrome/Avast — міряти лише свіжим процесом) · matmul 4.1–4.7 s (~77 %) · attention 0.74–0.78 s (13.9 %) · down `896×4864` **64–69 GMAC/s** (було 36–42) · gate/up **81–92** · swiglu 122–132 ms · rope 32–35 ms · other **2.4–3 %** |
| Decode (live) | **10.4–27.6 tok/s** — дуже залежить від фонового навантаження; міряти на свіжому процесі |
| Числова політика | golden-fingerprint `tests/data/qwen25_golden.txt` не зрушив усю серію PH-203j…r; recapture PH-203s збігся з ним байт-у-байт і не застосовувався (зріз відкату) |
| Живий білд | `target/release/rusty.exe` + `target/live/launch_serve.vbs` (CRLF!), `threads.txt`=12, `profile.txt`=1, звіт `target/live/rusty_prof.err.log` |

**Далі:** обидва perf-левери prefill закриті виміром: `PH-203s` (attention softmax — не
exp-bound, `benches/attn.rs`) і `PH-203t` (matmul `w_rows=4` quad-pass + NT-store — без
стійкого пул-виграшу, на wide down мінус, §Продовження 27). Продакшн ядра лишились
PH-203r-подібні (bit-identical, golden не рухався). **`PH-1201` spike зроблено
2026-10-24 (§Продовження 28):** MiniCPM-V 4.7-1B = text-core **`qwen35`-гібрид**
(18 GDN + 6 full-attn, QK-norm, zyxt-rope, vocab 248 094) + vision-clip BF16
(`ggml_dtype(30)` фікс). **`PH-1301` закрито того ж дня (§Продовження 29: config/geometry
qwen35 + `rusty validate` на реальному файлі = OK); **`PH-1302` теж закрито** (§30:
`src/gdn.rs` — delta-rule стан, prefill==steps bit-identical); **`PH-1303` закрито**
(§31: `src/q35.rs`, zyxt = M-RoPE слоти за llama.cpp PR #29416) і **`PH-1304` закрито**
(§32: `src/runner35.rs` — реальний MiniCPM-V генерує, 244 тести + 7 ignored); **наступний
зріз — 1305** (pin-пара проти `llama.cpp-omni` + batched-prefill perf)
→ 1304 pre-tokenizer/runner → 1305 E2E-паритет проти
`llama.cpp-omni` → 1306 serve; B12-vision (1202/1203) будується паралельно, image-prefix
(1204) чекає B13. **Паралельно
(owner 2026-10-24): модельна політика** — цільова модель **MiniCPM-V 4.7** (official GGUF,
без Python; `src/vit.rs` + 4×/16× compressor + image-prefix + multimodal serve;
`PH-1201..1206` + `PH-1301..1306` на борді GSV; Qwen3/Mixtral з цілей прибрано,
Qwen2.5 = оракул).

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
OmniRoute `:20128` policy-down. Owner-згодою створено GitHub-remote
`platinoff/rusty-engine` (спочатку private; з 2026-10-29 public, MIT).
CHANGELOG і third-party notices — далі за B11.
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

### Продовження 20 — 2026-10-22: PH-203m — per-stage профайлер (де насправді йде prefill)

- Замість вгадування додано `src/profile.rs`: `RUSTY_PROFILE=1` вмикає облік трьох стадій
  (`matvec`, `matmul`, `attention`) атомарними лічильниками ns/calls. Вимкнено → один relaxed
  load на проєкцію, без алокацій; увімкнено → ще й розбивка `matmul` **за формою**
  (`note_matmul`, `Mutex<Vec<ShapeRow>>`, top-10 за часом).
- Хуки: `QMat::matvec`/`QMat::matmul` (обгортки над `*_inner`), `forward::attention_prefill`
  (обгортка над `attention_prefill_inner`). Звіт: `DenseDecoder::prefill` (production serve-шлях,
  один рядок у stderr на запит) + `print_report_once` у `DenseRunner::prefill_logits` (re-prefill
  CLI викликає його на кожен токен — тому лише перший раз).
- **Лайв-вимір** (release, `RUSTY_THREADS=12`, 961-токенний промпт на LAN, `rusty_prof.err.log`):
  wall **20.2 s** → matmul **84.4 %** (168 викликів = 24 шари × 7 проєкцій), attention **9.4 %**
  (1.9 s, 24 × 79 ms), matvec 0.2 % (1 виклик = LM head), other ~6 % (норми, RoPE, KV-копії,
  семплінг). Отже «вузьке місце attention» — міф: attention у 9 разів дешевший за matmul.
- **Головна знахідка — форми нерівні:**

  | форма | час | GMAC/s |
  |---|---|---|
  | `896×4864` (down) | 10051 ms / 24 = **419 ms** | **10.0** |
  | `4864×896` (gate/up) | 5779 ms / 48 = 120 ms | 34.8 |
  | `896×896` (q/o) | 1028 ms / 48 = 21.4 ms | 36.0 |
  | `128×896` (k/v) | 177 ms / 48 = 3.7 ms | 29.8 |

  Тобто 24 down-проєкції з'їдають **~60 % усього matmul** при тій самій кількості MAC, що й
  gate+up разом. Причина: `n = 4864` → rowbuf 19.5 KiB + тайл `X` 16×19.5 = 312 KiB, тобто
  робочий набір не влазить в L1 (32 KiB) і кожен `dot` тягне 19.5 KiB з L2; для `n = 896`
  (3.5 KiB) усе лишається в L1. PH-203l (один hsum на рядок) виграв саме на вузьких формах, а
  найширшу — зіпсував.
- **Аналіз стелі:** `dot_f32` на 8 смугах робить 1 FMA на 2 завантаження (rowbuf + x) — це рівно
  ліміт load-портів AVX2 (2/цикл), тому 6.9 GMAC/s single-thread = 8.2 MAC/цикл ≈ 82 % від
  8/цикл. Далі без зміни співвідношення load:FMA не рухається: потрібен **register blocking по
  рядках** (4 рядки `X` на одне завантаження ваги → 4 FMA / 5 loads ≈ 1.9×), і він **біт-у-біт**
  (той самий порядок сум на `(row, r)`). Або K-chunking 1024 для широких форм (повертає L1, але
  додає hsum → числовий зріз).
- Гейт: 197 lib (+2 тести профайлера) + integration green, clippy 0. Живий сервер перезапущено
  через `spawn_threads.ps1`; `target/live/profile.txt` (вміст `1`) вмикає профіль у
  `launch_serve.vbs`, вимкнути = видалити файл. **Логи тепер `rusty_prof.{out,err}.log`** (старі
  `rusty_serve.*` лишились зайнятими попереднім cmd-обгорткою і давали «дірки» нулів у файлі —
  читати їх після `rm` нового імені, не через `tr -d`).
- **Далі:** 4-рядковий мікрокернел (`dot4`, біт-у-біт, найбільший важіль: −30…40 % matmul),
  потім K-chunking для `n>1024`, потім attention (9.4 % — низький пріоритет).

### Продовження 21 — 2026-10-22: PH-203n — register blocking + 2-D cache blocking + split axis (prefill 15.2 → 10.4 s)

Три незалежні зміни, **усі біт-у-біт** (той самий порядок сум на `(row, r)`, змінюється лише
порядок незалежних ітерацій), тому golden не перезахоплювався.

1. **4-рядковий мікрокернел** (`dot_f32_avx2_x4`, `dot_f32_x4_dispatch`): одне завантаження
   ваги на 4 FMA замість 1 → 4 FMA / 5 loads замість 1 FMA / 2 loads. `MATMUL_ROW_GROUP = 4`,
   `matmul_reuse` отримав параметр `group`; хвіст рядків падає на `dot_f32_dispatch`.
   Тест `matmul_row_grouping_is_bit_identical` (70 = 4×17 + 2).
2. **Двовимірне блокування** в `matmul_reuse`: тайл тепер **зовні** циклу ваг
   (`for r_block { for tile { for r { for row group } } }`), `MATMUL_R_BLOCK = 1024` (≈ «уся
   матриця»), `MATMUL_ROW_TILE = 96`. Старий порядок `for r { for every row }` перечитував
   **весь** `X` з DRAM на кожен ваговий рядок — 16.8 GiB на одну down-проєкцію. Тайл — це вже
   насамперед амортизація дейкванту (дейквант на ваговий рядок **на тайл**, тобто `MAC/tile`),
   а не L2-трюк: 16 → 96 дає −30 %, 192+ знову падає (тайл перестає влазити).
   Хуки A/B: `matmul_q5_0_rows1` (group 1), `matmul_q5_0_rb1` (r_block 1),
   `matmul_q5_0_blocked(tile, rb)`; тест `matmul_weight_row_blocking_is_bit_identical`.
3. **Вісь паралелізму за розміром `X`** (`QMat::matmul`, `MATMUL_X_L3_BUDGET = 6 MiB`): якщо
   `rows·n·4` влазить у L3 — рядковий поділ (як було), інакше поділ за **вихідними рядками**
   (`matmul_out_rows`, той самий код, що старий `matmul_split_cols`). Для down (`X` = 18.7 MiB)
   12 рядкових слайсів вибивають один одного з 8 MiB L3.

Заміри (один процес, min-of-7/5, Q5_0, 961 рядків):

| форма | PH-203l | PH-203n (1 поток) | прискорення |
|---|---|---|---|
| `4864×896` gate/up | 606 ms / 6.9 GMAC/s | **199–201 ms / 21.0** | 3.0× |
| `896×4864` down | 688–914 ms / 4.6–6.1 | **218–222 ms / 18.9** | 2.8–4.2× |

Пул 12 потоків, A/B двох осей: gate/up — рядки **56.2 ms / 74.5 GMAC/s** проти 67.1 / 62.4
(рядковий поділ 1.19× кращий); down — рядки 256.7 ms / 16.3 проти **72.5 ms / 57.7 GMAC/s**
(поділ за вихідними рядками **3.54×** кращий).

**Лайв 961 tok (release, 12 потоків, LAN):** 15.2 → **10.44 s** — matmul 7.22 s (69.2 %),
attention 2.02 s (19.3 %), matvec 0.02 s. Форми: `4864×896` 53.9 GMAC/s, `896×4864` **36.4**
(було 10.0), `896×896` 60.3, `128×896` 48.0. Ланцюг сесії: 26.9–27.3 s (PH-203k) → 20.2 (203m)
→ 15.2 (x4) → **10.4 s**.

Гейт: 199 lib + integration green, clippy 0, fmt ok, `cargo test --release -- --ignored` —
`qwen25_greedy_fingerprint_is_stable` + `moe_parity` (3) + `dense_runner_loads_real_qwen` **ok**
без перезахоплення golden.

- **Далі:** attention 19.3 % (2.0 s, 24 × 84 ms) — вже head-parallel, тож дивитись на
  `attention_prefill_inner` (онлайн-softmax по 961 рядках × 3.6 MB scores?) і на `other` ~11 %;
  потім K-chunking/`rb` для `m > 1024` (зараз `12640→7220 ms` — це вже майже стеля 74 GMAC/s
  на вузьких формах), NEON.

### Продовження 22 — 2026-10-22: PH-203o — prefill attention 2.0 → 0.84 s (x4 scores + vector axpy + row rounds)

Профіль після PH-203n показав attention **2.02 s (19.3 %, 24 × 84 ms)**. Порахували роботу:
≈9.9 G MAC на оцінки + стільки ж на зважену суму значень, тобто attention був **compute-bound на
скалярному** `qh.iter().zip(kh).map(..).sum()` (1 MAC/цикл), а не memory-bound.

- **Оцінки** — `kernel::dot_f32_x4_dispatch(qh, [k_j..k_j+3])`: одне завантаження `q` на 4 ключі
  (той самий 8-смуговий акумулятор, що й `dot_f32_dispatch`; хвіст — `dot_f32_dispatch`).
  Це **числовий зріз** (послідовна f32-сума → 8-смуговий акумулятор).
- **Зважена сума значень** — новий `kernel::axpy_f32_dispatch` (`y += a·x`, AVX2 8-wide,
  `mul`+`add` без FMA-контракції) — **біт-у-біт** дорівнює скалярному `*o += p * *v`;
  тест `axpy_dispatch_is_bit_exact_with_scalar` (довжини 1/7/8/9/64/67).
- **Розподіл роботи**: 14 голів на 12 потоків давали дві хвилі (12 + 2). Тепер робота =
  `(head, round)` з `rounds = clamp(4·threads / heads, 1, seq) = 4` → 56 завдань, рядки
  роздаються **round-robin** (рядок `i` коштує O(i), тому суцільні блоки нерівні). Кожне
  завдання пише свій набір рядків у спільний `ctx`; тест
  `prefill_attention_row_rounds_are_bit_identical` (rounds 3 і 7 проти одного потоку).

Лайв (release, 12 потоків, 4 прогони 961 tok): attention **812–880 ms** (було 2018), prefill
**9.12–9.90 s** (було 10.44). `matmul` 7.07–7.95 s (77 %), форми стабільні: `4864×896`
51–55 GMAC/s, `896×4864` 31–38, `896×896` 55–58, `128×896` 46–52. `other` ≈ 11 % (норми, RoPE,
KV-копії, семплінг) — наступний за величиною після matmul.

Гейт: 201 lib + integration green, clippy 0, fmt ok; `cargo test --release -- --ignored` —
fingerprint + `moe_parity` (3) + `dense_runner_loads_real_qwen` **ok** (greedy-токени не
змінились, golden не перезахоплювався, попри числовий зріз в оцінках).

- **Далі:** `other` ≈ 1 s — розкласти профайлером (rmsnorm/RoPE/KV-copy/swiglu —
  `note_matmul` вже є, треба `Stage::Other` або лічильник на норми); далі K-квантовий
  dequant-once 256 (`MATMUL_ROW_TILE` зараз 96 → для 256-блочних форматів тайл кратний 256?),
  NEON.

### Продовження 23 — 2026-10-22: PH-203p — «other» розкладено профайлером і зрізано (serial 1010 → 260–297 ms)

Після PH-203o профіль показував matmul ~77 %, attention 0.81–0.88 s і ~1 s «other» без
розкладу. Спершу інструментарій, потім різанина; обидва зрізи **біт-у-біт**.

- **`src/profile.rs`:** `Stage` розширено до `{Matvec,Matmul,Attention,Norm,Rope,KvCopy,Swiglu,
  Residual,Embed}` (масиви `[AtomicU64; N_STAGES]`), у звіті з'явились `[pool]`-маркер,
  підсумок **serial** (стадії від `Norm` міряються на викликаючому потоці) та
  **`unaccounted = wall − serial`**. Тест `report_splits_pool_and_serial_stages`.
- **Хуки:** `forward.rs` — `norm_row`/`norm_rows`, `residual_add`, `swiglu_batch`, `rope_rows`,
  `kv_append` (усі три шляхи — `block_forward_seq`, cached-prefill, decode — тепер кличуть
  спільні хелпери, тож дубльовані цикли зникли); `runner.rs` — `embed_batch` + фінальна норма.
- **Замір до зрізу (live 961 tok, 12 потоків):** wall **8493 ms** → serial **1010 ms (11.9 %)** =
  **swiglu 676 ms (8.0 %)** + rope 205 ms (2.4 %) + residual 69 + norm 53 + kv-copy 6 + embed 1.
  Тобто «other» — це на 67 % `silu` (112 M `exp` на 24 шари) і на 20 % RoPE-тригонометрія.
- **SwiGLU на пулі:** 4·threads діапазонів по `seq×d_ff`, кожен пише свій `lo..hi` у спільний
  вихід (покажчик через `as_mut_ptr`; `parallel_for` дочікується всіх job-ів; `SWIGLU_PAR_MIN`
  16 K елементів). Порядок арифметики на елемент той самий → біт-у-біт. **676 → 116–123 ms (5.7×)**.
- **RoPE-таблиця:** `infer::rope_table(seq, pos_start, d, base)` + `rope_in_place_tabled`
  рахують `(cos,sin)` раз на `(позиція, пара)` (той самий `angle = pos·freq` і ті самі виклики
  `cos`/`sin`, що в `rope_in_place`), а не `heads×`. Тест
  `rope_table_is_bit_identical_to_per_position_rope`. **205 → 33–41 ms (6×)**.
- **Після (3 прогони, LAN-проба):** wall **6.90 / 7.35 / 7.84 s** (було 8.49 на тому ж боксі),
  matmul 5.80–6.60 s (**84 %**), attention 657–766 ms (9.8 %), swiglu 116–123 ms, rope 33–41 ms,
  norm 50–54, residual 53–65, kv-copy 6, **serial 260–297 ms = 3.4–3.8 %**.
- **Гейт:** 203 lib + integration green, clippy 0, fmt ok; `cargo test --release -- --ignored` —
  `qwen25_greedy_fingerprint_is_stable` + `moe_parity` (3) + `dense_runner_loads_real_qwen` **ok**
  без перезахоплення golden.
- **Далі:** matmul 84 % → `PH-203q` (K-квантовий dequant-once на тайлі 256; down-форми
  36–42 GMAC/s проти 63–72 у `4864×896`), потім attention 9.8 %, NEON.

### Продовження 24 — 2026-10-22: PH-203q — гіпотезу «тайл 256» відкинуто виміром; тайл став похідною від L3 (down +25 %)

Гіпотеза з роадмепу: K-кванти мають блок 256, тож тайл 96 «де-квантує блок 2.67×», і 256 дасть
економію. Перевірка показала протилежне — і знайшла справжній механізм.

- **`src/kernel.rs`:** три K-квантові диспетчери зведено в один `matmul_k(ggml_type, …)` +
  `k_quant()` (144/176/210 B, `type KBlockDequant`), щоб тайл жив в одному місці.
  `matmul_k_tiled` — вимірювальний хук (біт-у-біт, тест `matmul_k_quant_tiling_is_bit_identical`).
- **`benches/kquant.rs` (новий, `harness = false`):** тайл-свіп на реальних формах
  (`down 896×4864` Q6_K — єдиний K-квант 0.5B-моделі, `gate/up 4864×896`, `down` Q4_K), окремо
  один потік і **пул 12 потоків** (`pool_matmul` повторює прод-поділ за вихідними рядками).
- **Один потік:** down Q6_K — 32 → 310.9 ms, 48 → 278.0, 64 → 261.9, 96 → 245.0
  (**17.1 GMAC/s**), 128 → 263.4, 256 → 439.9 (**−44 %**). Тобто дейквант K-квантів — це одиниці
  відсотків, а 96 рядків × 4864 × 4 B = 1.87 MiB уже не влазить у L2, тому далі винна памʼять,
  не дейквант.
- **Пул 12 потоків (головне):** down Q6_K — тайл 96 = **109.4 ms (38.3 GMAC/s)**, 48 = 75.9,
  32 = 71.9, **24 = 66.1 ms (63.4 GMAC/s)**, 16 = 77.1. Кожен воркер тримає **свій** тайл, тож
  вузьке місце — L3 на всіх: 12 × 1.87 MiB = 22 MiB проти 8 MiB L3. Вузька форма навпаки
  (`48 → 64 GMAC/s` при 96), тому тайл став функцією: `k_row_tile_with(n, threads)` =
  `clamp(6 MiB / (threads·n·4), 8, 96)`, кратне 8. Тести: `k_row_tile_shrinks_with_row_width_and_threads`
  (896/12 → 96, 4864/12 → 24, 4864/1 → 96, 16384/12 → 8).
- **Register blocking x8 — виміряно, ВІДКИНУТО:** 8 незалежних акумуляторів прибирають
  латентність `vfmadd` (один потік: down 18.81 → 20.68 GMAC/s, gate/up 19.18 → 21.67), але в
  пулі кожен воркер отримує 8 потоків X замість 4 → down **падає 36.8–41.9 → 23.3–25.1 GMAC/s**,
  wall 6.9 → 9.1–10.3 s. Відкочено, `MATMUL_ROW_GROUP` лишився 4; хук `matmul_q5_0_grouped` +
  свіп у `benches/quant.rs` лишились як доказ. Заодно виправлено латентний баг `dot_f32_avx2_x4`:
  хвіст `n % 8` додавався лише в `out[0]` (на `n = 32k` не проявлявся, golden не зрушив).
- **Live (той самий бокс, але з дрейфом навантаження):** down **2.40–2.73 → 1.79–2.02 s**
  (36.8–41.9 → **49.7–56.0 GMAC/s**), gate/up 58–65, attention 0.72–0.93 s; wall 7.1–8.9 s
  (розкид > ефекту, бокс ділить CPU з Chrome/Avast — тому доказ = in-process A/B вище).
- **Гейт:** 205 lib + integration = **224** тести, clippy 0, fmt ok; `--ignored` (golden +
  moe-parity + dense real-model) зелені без перезахоплення.
- **Далі:** `PH-203r` — два рядки ваг на прохід (2 `rowbuf`), щоб зняти load-port тиск
  `5 loads / 4 FMA`; потім attention (exp-bound) і NEON.

### Продовження 25 — 2026-10-22: PH-203r — два рядки ваг на прохід (prefill 7.1–8.9 → 5.3–6.4 s)

- **`src/kernel.rs`:** `dot_f32_avx2_x4x2` + `dot_f32_x4x2_dispatch` (AVX2 → 2× скалярний
  fallback): чотири активації проти **двох** дейквантованих рядків ваг. Внутрішній крок став
  `2 ваги + 4 активації = 6 завантажень / 8 FMA` (0.75) замість `5 / 4` (1.25); кожен вихід
  зберігає власний 8-ланьовий ланцюг, тому біт-у-біт (`matmul_two_weight_rows_is_bit_identical`).
- **`matmul_reuse`** розкладено на `dot_tile_rows` + `dot_tile_rows_pair`; новий параметр
  `w_rows` (`MATMUL_W_ROWS = 2`, `clamp(1, 2)`), пара вмикається лише при `group == 4` і
  непарному залишку рядків — решта йде старим шляхом. Хуки: `matmul_q5_0_wrows`,
  `matmul_k_tiled_rows` (усі біт-у-біт).
- **Таблиця, яка змінила тайл:** пул 12 потоків, `benches/quant.rs` + `benches/kquant.rs`
  (min of 3): gate/up Q5_0 — 1 рядок 54.9 ms (76.3 GMAC/s) → **2 рядки 41.2 ms (101.7)**;
  down Q5_0 — 54.2 (77.2) → **38.6 (108.5)**. Один потік: 19.1 → 30.3 і 17.7 → 24.0 GMAC/s.
- **Гіпотеза PH-203q про «тайл ділиться на воркерів» — відкинута:** тайл — це **спільний**
  зріз `X`, у воркера лише власні `rowbuf`-и. З парою рядків pooled-оптимум down Q6_K:
  24 → 75, 48 → 99, **96 → 108**, 128 → 119, 192 → 80 GMAC/s. 128 дав +4 % у бенчі, але в live
  down **55.4** проти **64–69** при 96, тому `MATMUL_ROW_TILE_K = MATMUL_ROW_TILE` (96) для
  обох шляхів, а `k_row_tile_with`/`k_row_tile` і тест на форму видалено.
- **Live (свіжий процес, 12 потоків, LAN-проба):** wall **5.34/5.60/5.81/6.03/6.11/6.41 s**
  (було 7.1–8.9); matmul **4.1–4.7 s (77 %)**; down `896×4864` **1.46–1.56 s = 64–69 GMAC/s**
  (було 1.79–2.02 s / 50–56); gate/up **81–92**; attention QK `896×896` **84–92**;
  attention 0.74–0.78 s (13.9 %); swiglu 122–132 ms; rope 32–35 ms; other 2.4–3 %.
- **Гейт:** 205 lib + integration = **224** тести, clippy 0, fmt ok, `--ignored` (golden +
  moe-parity + dense real-model) зелені; golden не перезахоплювався.
- **Висновок:** matmul тепер упирається у FMA/load межу (2 FMA/цикл × 16 MAC), а не в тайл чи
  дейквант. Наступний великий резерв — attention (13.9 %), далі памʼять (`r_block`,
  non-temporal store) і NEON.

### Продовження 26 — 2026-10-24: PH-203s — онлайн-softmax + exp-таблиця в attention: ВІМІРЯНО й ВІДКИНУТО

- **Owner GO** на зріз із новим golden (2026-10-24, question-пік «PH-203s: attention»).
- **Реалізовано повністю:** `exp_neg_fast` (`exp2(x·log2e)`, 1024+1 entry таблиця, лінійна
  інтерполяція, `OnceLock` хост поза хот-лупом) + `attention_head_ctx` на онлайн-softmax
  блоками по 128 ключів (running max/sum, рескейл акумулятора, пропуск нульових ваг).
  Спершу fused-варіант (exp+axpy в одному лупі) — −15 % на live; перекомпонування на
  split-pass (exp+sum, потім чистий axpy-пас) — повернуло live до паритету з PH-203o.
- **Golden перезахоплено (`RUSTY_CAPTURE_GOLDEN=1`) — і він збігся з старим байт-у-байт**
  (16-токенний greedy fingerprint не зрушив, drift ~3e-7 на логітах); повна `--ignored`
  parity включно з llama-rs oracle — зелена. Тобто чисельно зріз був безпечний.
- **Вердикт дав in-process бенч `benches/attn.rs` (нова A/B-підмодуля, std-only,
  min-of-5, Qwen-0.5B геометрія 961 рядок × hd 64, 336 head-pass):**
  | варіант | ms/head (3 прогони) |
  |---|---|
  | PH-203o 3-pass libm (продакшн) | **9.41 / 10.07 / 10.47** |
  | онлайн-128 + exp-таблиця | 10.73 / 10.56 / 10.93 |
  | 3-pass, лише таблиця замість libm exp | 10.49 / 10.52 / 10.57 |
  | 3-pass + skip `p<1e-8` / `1e-6` | 9.56 / 9.59 — **kept 100 %**, нульових ключів нема |
  | 3-pass + skip `p<1e-4` | 9.71 — зрізає 8 % axpy, виграшу немає (гілка бʼє SIMD-стрім) |
- **Висновок:** преміза «attention exp-bound» — хибна. `f32::exp` на libm цього бокса
  швидший за ланцюжок таблиці (roundss + 2 конверсії + залежні gather-Loads), а онлайн-
  блокування додає ще 3–6 %. Attention-стейдж спирається у `dot_f32_x4` + axpy, обидва
  вже AVX2-фордовані; softmax-паси — одиниці відсотків стейджу.
- **Рішення (прецедент PH-203q):** `src/forward.rs` **повернуто до PH-203o** (біт-у-біт зі
  старим golden — recapture не потрібен). Хуки лишились: `kernel::exp_neg_fast` /
  `exp2_table_ref` (pub, з тестом `< 1e-5` rel на [-87, 0]), `dot_f32_dispatch` /
  `dot_f32_x4_dispatch` / `axpy_f32_dispatch` promoted to pub для бенча, `benches/attn.rs`
  збережено як доказ. `Cargo.toml`: `[[bench]] attn`.
- **Гейт після відкату:** fmt ok, clippy `-D` 0, **206** lib (205 + exp-тест) + integration
  = **225** (+5 ignored) зелені; `--ignored` parity проти старого golden — ok; реліз-серв
  перезапущено (`launch_serve.vbs`), heartbeat свіжий.
- **Далі:** attention у prefill закрито виміром (не softmax-lever). Наступний зріз
  `PH-203t` — памʼять matmul (`r_block`, non-temporal store на wide-out) з A/B у
  `benches/kquant.rs`; потім NEON (B2) і B6–B11.

### Продовження 27 — 2026-10-24: PH-203t — w_rows=4 quad-pass і NT-store: ВІМІРЯНО й ВІДКИНУТО

- **Plumbing:** `matmul_reuse` clamp піднято до 4; quad-pass = два `dot_f32_x4x2` по **одному
  й тому самому** activation-quad → X-тайл стрімиться `m/4`, а не `m/2` разів на проєкцію;
  `rowbuf3/4` аллокуються лише при `w_rows == 4` (продакшн-шлях не видихає нічого).
  Біт-ідентичність гарантована структурою (кожен вихід — своя 8-ланьова цеп):
  `matmul_four_weight_rows_is_bit_identical` (m=9 — лесідра 4→2→1).
- **Пулове A/B, три проходи `benches/kquant.rs` + `quant.rs` (GMAC/s, min):**
  | форма · тайл | wr2 | wr4 |
  |---|---|---|
  | gate/up Q6_K t96 (3 проходи) | 87.8 / 92.0 / 86.7 | 89.2 / 89.4 / 82.9 |
  | down Q6_K t96 | 78.7 / 79.7 / 98.3 | **39.0 / 76.2 / 65.5** |
  | down Q6_K t48 | 94.3 / 104.6 / 98.1 | 91.8 / 96.6 / 93.8 |
  | gate/up Q5_0 t96 (1 прохід quant.rs) | 92.4 | 96.2 |
  Жодного стійкого виграшу; на wide down — стійкий мінус (4×19.4 KiB rowbuf-ів на воркер
  звалив спільний L3 — четверте підтвердження ефекту PH-203n/q).
- **NT-store:** пораховано, не мірялось — `out` пишеться по одному разу; write+RFO трафік
  gate/up ≈ 37 MB/проєкцію проти ~250–400 GB/s сукупного stage → верхня межа виграшу
  **<1 %**; `stream`-store вимагав би вирівнювання стридів — не варте риски.
- **Побічна знахідка (методологія):** послідовні тайл-свіпи брешуть на цьому боксі — новий
  **interleaved t48-vs-t96 (12 раундів, per-round win count)** у `kquant.rs`: min
  43.5 vs 43.5 мс, **t48 виграв 11/12 по медіані** → тайл впливає на хвіст розподілу, не на
  best-case; `MATMUL_ROW_TILE_K = 96` лишається, але черговий тайл-зріз мірити лише
  альтернованими раундами.
- **Рішення:** продакшн `MATMUL_W_ROWS = 2` без змін; чисельно все біт-у-біт — golden не
  рухався (`--ignored` 5/5 ok, включно з llama-rs oracle). Хуки `wrows(...,4)` +
  інтерлейвд-блок у бенчах лишились як доказ.
- **Гейт:** fmt ok · clippy `-D` 0 · **226** тестів (207 lib + 5 dense + 4 scaffold + 5 serve
  + 5 text-gen) + 5 ignored; Rust **15073** LOC, ratio **99.0 %**; serve перезапущено.
- **Далі:** perf-черга matmul закрита до NEON/B12-контексту; наступний зріз — **`PH-1201`**
  (B12 spike: official MiniCPM-V 4.7-1B GGUF + `rusty inspect` vision-layout).

### Продовження 28 — 2026-10-24: PH-1201 — MiniCPM-V 4.7 spike: core = qwen35-ГІБРИД, vision = BF16-clip

- **Ваги:** official HF `openbmb/MiniCPM-V-4.7-1B-gguf` → `data/models/` (gitignored):
  `MiniCPM-V-4_7-1B-Q4_K_M.gguf` 672 MB + `mmproj-model-f16.gguf` 1.11 GB (548M, arch
  `clip`, type `mmproj`).
- **`rusty inspect` прокачано** (PH-1201 скоуп, без forward-правок): повний KV-дамп
  (скорочений — chat_template 7 249 chars обрізається), **tensor-name prefix
  histogram**, `[N]` — скільки тензорів показати; дампи:
  `target/live/inspect_ph1201_{main,mmproj}_full.txt`, факты:
  [`research/minicpm-v-4.7.md`](research/minicpm-v-4.7.md).
- **Головне відкриття:** text-core 4.7-1B — **НЕ qwen2, а `qwen35`** — Qwen3-Next-подібний
  **гібрид**: 18 шарів GDN (Gated DeltaNet: mixed `attn_qkv` 6144, depthwise
  `ssm_conv1d` k4, delta-rule стан 16 груп × 128², `ssm_norm` 128, out-gate) + 6
  full-attn шарів (кожний 4-й; GQA 8q/2kv head_dim 256, **QK-norm**, q-gate 4096) +
  sectioned RoPE `[11,11,10,0]` "zyxt" + pre-tokenizer `qwen35` + vocab 248 094.
  → новий **бенд B13** (`PH-1301..1306`) у roadmap; B12-PH-1204 (image-prefix) тепер
  чекає B13. GDN-декод = O(1)/токен — наступний диференціатор після MoE.
- **Баг-фікс по дорозі:** канон ggml.h — `BF16 = 30` (не 24; 24 = I8). MMproj тримає
  **усі** vision-ваги як BF16. Додано: `ggml_dtype(30)=BF16` (+тест), `block_info(30)`,
  `dequant` arm 30 (bf16 = верхні 16 біт f32, `<< 16` без таблиці) +
  `dequant_bf16_widens_by_left_shift`. Без цього mmproj не читався б взагалі.
- **Гейт:** fmt ok · clippy `-D` 0 · **227** тестів (208 lib + 19 integration) + 5
  ignored parity; Rust **15173** LOC (src 12387 / tests 1317 / xtask 319 / benches
  1150), ratio **99.0 %**; serve перезапущено після збірки.
- **Далі:** **`PH-1301`** — `model.rs`: config `qwen35` + Geometry + validate; далі —
  1302 GDN-шар, 1303 full-attn, 1304 pre-tokenizer+runner, 1305 E2E-паритет проти
  `llama.cpp-omni` CLI, 1306 serve.

### Продовження 29 — 2026-10-24: PH-1301 — `model.rs` читає qwen35-гібрид; `rusty validate` на реальному MiniCPM-V = OK

- **`LayerKind::{Recurrent, Full}` + `layer_kind(i)`** (Bool-массив `recurrent_layers`
  вигріває над правилом `(i+1) % full_attention_interval == 0`; не-гібриди = завжди
  Full, qwen2-шлях не чіпається). `head_dim()` тепер = `attention.key_length` (256),
  якщо він є — інакше стара формула d/n_heads.
- **Нові поля config** (усі опційні, default 0/порожні): key_length, interval,
  recurrent_layers, ssm_* (conv4/state128/groups16/rank16/inner2048),
  rope.dimension_count + dimension_sections [11,11,10,0] + section_order "zyxt";
  хелпери `gdn_qkv_dim()` (=3·inner) і `full_q_dim()` (=n_heads·head_dim).
- **`validate`-гібрид**: GDN-шар вимагає attn_qkv [dm,3·inner], attn_gate [dm,inner],
  ssm_a/alpha/beta/dt [rank], ssm_conv1d [conv,3·inner], ssm_norm [state], ssm_out
  [inner,dm]; full-шар — attn_q [dm,2·q], k/v [dm,kv], attn_output [q,dm],
  attn_q_norm/attn_k_norm [hd]; обидва — attn_norm + **post_attention_norm**
  (являється і перед-FFN нормою: **`ffn_norm` у qwen35 немає** — це єдине, на що
  реальний файл спочатку порігся, фікс закріплений тестом).
- **Тести:** synth-config (2 шари R+F, усі поля + kind-матриця), synth-validate
  (чисто) + 2 missmatch-тести (attn_k у qwen2, ssm_conv1d у GDN) +
  `#[ignore] real_minicpm_v47_1b_config_and_validate` — **на реальному файлі green**:
  validate == [], 24 шари, key_length 256, sections [11,11,10,0], 6 full шарів;
  CLI `rusty validate data/models/MiniCPM-V-4_7-1B-Q4_K_M.gguf` → OK.
- **Гейт:** fmt ok · clippy `-D` 0 · **230** тестів (211 lib + 19 integration) +
  6 ignored (усі зелені); Rust **15535** LOC (src 12749 / tests 1317 / xtask 319 /
  benches 1150), ratio **99.0 %**; qwen2-шлях біт-у-біт (golden не рухався).
- **Далі:** **`PH-1302`** — `src/gdn.rs` (Gated DeltaNet: conv1d k4 + delta-rule
  стан 16×128², O(1) decode). Увага при впровадженні: `post_attention_norm` =
  pre-FFN norm; q/k/v у GDN — один `attn_qkv` 6144 (2048×3), head_dim GDN =
  state_size = 128 (groups 16), а НЕ 256.

### Продовження 30 — 2026-10-24: PH-1302 — `src/gdn.rs`: Gated DeltaNet з O(1)-decode

- **Новий модуль `gdn.rs`** (pure std, без I/O — проекції позичені `QMat`-ами):
  `GdnGeom::from_config` (валідний лише при `inner == groups·dk`), `GdnWeights`
  (qkv/gate/alpha/beta/out + a_log/dt_bias/conv_w[ch·K+j]/norm_w), `GdnState`
  (conv-вікно + `groups × dk × dv` матриць).
- **Ядро — `delta_step`**: `S·=e^g → pred=kᵀS → vnew=(v−pred)·β → S+=k⊗vnew → o=Sᵀq̃`
  (row-major, порядок фіксований). Гейти: `g = -exp(A_log)·softplus(a+dt_bias)`
  (стабільний softplus, плато |x|>20), `β=σ(b)`; q/k — L2-норм на голову + `dk^-0.5`
  на q (fla `use_qk_l2norm`); після скану — **gated head-RMSNorm
  `rmsnorm(o)·w·swish(z)`** і `ssm_out`. `gate_act` — єдине місце з конвенцією
  (swish vs sigmoid gate), яку запінить `PH-1305` проти оракула.
- **Інваріанта зрізу:** `gdn_prefill` (батч-проекції + row-conv зі станом вікна) веде
  токени тим самим `delta_step` — **bit-identical** крокуючому `gdn_decode_step`
  (тест — і стан, і conv-вікно). `delta_step_matches_the_textbook_rule` — точні
  ручні f32; contract стану qwen35: **16·128·128·f32 = рівно 1 MiB** на
  recurrent-блок (×18 = 18 MiB фіксованого стану замість KV-кешу, що росте).
- **Обмеження v1 (зафіксовані в коді):** scratch `delta_step` — фіксований `dv ≤ 128`
  (qwen35 = 128); prefill-скан послідовний per-group — пул-розклад по (layer, group)
  і чанк-версія поїдуть у `PH-1304` wire; llama.cpp-omni parity-фіксації — `PH-1305`
  (оракул-білд на боксі ще не зібрано).
- **qwen2-шлях не чіпаний:** golden + 5 старих ignored зелені; новий 6-й ignored
  (real qwen35 validate) green.
- **Гейт:** fmt ok · clippy `-D` 0 · **236** тестів (217 lib + 19 integration) +
  6 ignored; Rust **16084** LOC (src 13298 / tests 1317 / xtask 319 / benches 1150),
  ratio **99.0 %**; serve перезапущено на новому білді.
- **Далі:** **`PH-1303`** — full-attn шар qwen35 (GQA 8/2×256 + QK-norm + q-gate +
  zyxt-sectioned rope) у `forward.rs`-гілці.

### Продовження 31 — 2026-10-24: PH-1303 — `src/q35.rs`: GQA 256 + QK-norm + q-gate; rope-секції розшифровано (M-RoPE)

- **Розшифровка з llama.cpp PR #29416:** `[11,11,10,0]`/`"zyxt"` = **M-RoPE slots**
  (кожна секція пари-частот живить свій t/y/x/z-слот); для тексту всі слоти
  збігаються ⇒ `src/q35.rs` = коректна **partial NeoX rope (64/256)**, крос-тест
  збігається з `infer::rope_in_place` при rope_dim==hd. Плюс знайдено
  `qwen35.nextn_predict_layers = 1` (dspark/MTP head — кандидата future spec-decode).
- **`Q35Geom::from_config`** — лише hybrid + `key_length>0` + парний `rope_dim ≤ hd`
  (qwen2 не кваліфікується); `q_dim` 2048, `kv_dim` 512, GQA 8/2.
- **`q35_decode_step` / `q35_prefill`** — загальні `project_rope_append` (QK-RMSNorm
  `attn_q_norm/attn_k_norm` після проекції, до rope; повернуті+нормовані K в кеш,
  policy PH-203g) і `attend_row` (scale `hd^-0.5`, libm softmax — lesson PH-203s).
  **attn_q [dm,2·q]: друга половина = gate**, `o ·= sigmoid(gate)` перед `attn_output`
  (`gate_act` — єдине місце, конвенція під pin `PH-1305`). Shape-guard на rows.
- **Інваріанта:** prefill == покроковий decode **біт-у-біт** (і виходи, і кеш) — 5
  нових unit; zero-in→zero-out; geom-контракт з реальних значень.
- **Гейт:** fmt ok · clippy `-D` 0 · **241** тест (222 lib + 19 integration) + 6
  ignored (qwen2 golden не рухався); Rust **16523** LOC (src 13737, 71 tracked).
- **Далі:** **`PH-1304`** — pre-tokenizer `qwen35` + specials + `DenseRunner`-wire
  18 GDN + 6 full за `recurrent_layers` (residual/norms тут же; M-RoPE слоти — на
  image-prefix 1204).

### Продовження 32 — 2026-10-24: PH-1304 — `src/runner35.rs`: гібрид біжить РЕАЛЬНИЙ MiniCPM-V

- **`runner35.rs`**: `Q35Runner::from_model` будує блоки за `layer_kind()` (GDN-частини
  + full-частини + FFN; проекції лишаються квантовими `QMat`, дейквантуються лише дрібні
  F32-вектори), `Q35Decoder` несе 18 delta-станів + 6 KV-кешів; топологія Qwen3-Next:
  `h = x + attn(rmsnorm(x, attn_norm)); y = h + ffn(rmsnorm(h, post_attention_norm))`
  (без `ffn_norm` — факт §29).
- **pre-tokenizer `qwen35`** (`tokenizer.rs`): мапиться на qwen-сьомейний single-digit
  splitter (гіпотеза до пін-тесту 1305); specials 248044–7 атомарні з token_type GGUF.
- **CLI `rusty run`**: `-t` іди → `Q35Runner`; `-p` текст → `textgen::TextRunner35`
  (BPE-in/BPE-out). **Реальний запуск:** prompt `[151644,872,198]` + 6 токени =
  `…24273,1710,271,12,271,328` за **2.36 s**; `-p "The capital of France is"` →
  звʼязний text-вихід за 2.10 s. Семантика ще не запинена (gate-конвенції, pre-split) —
  це предмет 1305; панік/NaN немає, рядки детерміновані.
- **Тести `tests/runner35_gen.rs`:** synth-qwen35 GGUF (R+F, 16 KV, 28 тензорів):
  config→validate→load обох kindʼів; residual-only greedy = той самий токен, що
  qwen2-synth (wire-еквівалентність); live-GDN: скінченні лоґіти + два декодери
  збігаються; `#[ignore]` real-смoke (24 шари, prefill finite, 7-токенний greedy
  deterministic) — **ok у release**.
- **Гейт:** fmt ok · clippy `-D` 0 · **244** тести (222 lib + 22 integration) +
  **7 ignored** green; Rust **17426** LOC (src 14177 / tests 1780 / xtask 319 /
  benches 1150), ratio **99.0 %**; serve перезапущено (qwen2-серв не зачеплений).
- **Далі:** **`PH-1305`** — build `llama.cpp-omni` CLI як оракул; pin `gate_act`
  (GDN swish vs sigmoid; attention σ), pre-split, softplus-плато; golden-файкстури
  `tests/data/qwen35_*.txt`; batched-prefill perf (GDN-скан по групах на пулі).

