# NEXT_SESSION_PROMPT — rusty-engine

## Owner product pick: `rusty-engine`

> Unknown pick = AskQuestion; iнакше продовжувати з `docs/HANDOFF_NEW_SESSION.md`.

## Owner-директива (2026-10-16) — worker/APK роль перейшла в ДВИГУНИ

Canon: `S:/rust/GSV/docs/gsv/GSV_AGI_PATH.md` (~§worker/APK, коміт `ae83e24` «Telenetis commutation
role (worker/APK to engines)») + `S:/rust/GSV/docs/gsv/PRODUCTS.md` (row `rusty-engine`):
**Telenetis = чиста комутація**, а локально-мережевий провайдер/WiFi-worker/APK несуть самі
двигуни — `rusty-engine` під id **`rusty-serve`** (`llama-rs` = `bunke-rock`). Двигун реєструється
як worker у GSV-хабі (single-entry edge `/api/edge`, фронт poolAI `/api/v1` — **не** `:8091` off-box):
register → heartbeat → `virtual-nodes/{peer}/pool/join` → `tasks/poll` → complete/requeue.
Owner-обсяг цього drain: **билд (exe/bin/apk) + runtime-worker роль**, два коміти (легі двигунів —
окремі репо), без push (B11).

## Зроблено цього сеансу (2026-10-16, rusty-engine)

- **`src/worker.rs`** (новий, PH-706) — std-only worker/APK runtime: `WorkerConfig` (LAN-first
  defaults, `GSV_LOCAL_ADDR`/UDP-route probe), мінімальний `parse_json`/`parse_task`, `execute`
  (`ping` / `rusty_chat` → `serve_url` `/v1/chat/completions`, dry-run без сокетів), білдери тіл
  (`register_body`/`heartbeat_body`/`join_body`/`complete_body`/`requeue_body`), blocking
  `http_call`/`join_once`/`heartbeat_once`/`poll_once` (header `x-gsv-edge-token`, origin
  `rusty_edge`, role `virtual_node`), `plan_lines`+`mask` — **токен ніколи не логується**. 17 unit.
- **CLI** `rusty worker [--live] [--peer-id] [--hub] [--serve] [--model] [--token-env] [--memory-mb]`
  — dry-run default друкує план (токен masked); `--live` = blocking register→join→heartbeat/poll loop.
- **`xtask build-matrix --build`** (PH-605) — гейтить `cargo build --release --target` по
  `rustup target list --installed`/host (`rustc -vV`); пропускає відсутні/невдалі cross-ланки з
  причиною (Android NDK / MSVC / cross-linker), не валить матрицю. Демо: host gnu `rusty.exe` →
  **sha256 `48b1f196…`** (617 984 B); linux/msvc/aarch64/android — skip з причиною.
- Gate: `cargo xtask check` зелений (fmt + clippy 0 + **181 тест**: 165 lib + 5 dense + 4 scaffold
  + 4 text + 3 xtask; +4 `#[ignore]` parity), dry-run `rusty worker` при порожньому env-токені показує
  `<unset>` і LAN-адресу. Один локальний коміт, **без push** (B11).

### Продовження (2026-10-16, provider live + perf)

- Провайдер `rusty-serve` піднято живим: `rusty serve -m data/models/qwen2.5-0.5b-q4_k_m.gguf`
  (`target/release/rusty.exe`, `0.0.0.0:8090`); `GET /health`/`/v1/models` OK, heartbeat тікає,
  `gsv_keep_live` → **`rusty: alive`**.
- **`50cd422`** fix(serve) **PH-504** — heartbeat-тікер кожні 15 с (idle-провайдер більше не гасне
  за 60 с; було startup+per-connection → `host_ready=false`).
- **`4359eb4`** perf(textgen) **PH-307** — `run_text`/`complete` декодують інкрементально
  (`DenseDecoder`, KV-кеш, O(n)); прибрано re-prefill усього префікса (O(n²)). Вихід незмінний.
- Заміри (0.5B Q4_K_M): `run -4` debug **91 с** → release **6.4 с**; chat 8 ток **>150 с** →
  release+incremental **~5.3 с** стабільно.
- **Наступний ботлнек → B2 `PH-203c` AVX2 / fused q4_k·q6_k matvec** (f32→quant; межа пам'яті).

### Продовження (2026-10-16, B2 QMat + PH-203c AVX2)

- **`99ec49a`** feat(forward) **B2** — `src/qmat.rs` (`QMat::{F32,Quant}` + fused `matvec`-dispatch,
  `from_model`); `BlockWeights`/`output` → `QMat`, dense-forward більше **не матеріалізує ~2 ГБ f32**
  (serve RSS ~540 МБ); golden `PH-405` відтворюється **точно**.
- **`46cebf6`** perf(kernel) **`PH-203c`** — `dot_q4_0/q4_k/q6_k_avx2` (8 lanes) + `dot_q4_k/q6_k_dispatch`
  / `matvec_*`; `QMat` типи 12/14 через них; parity avx2-vs-scalar. Заміри: serve 8-ток decode
  **~6.1 с → ~5.1 с** (~16 %); O(n²) parity-path **128.5 с → 102.5 с** (~20 %).
- **Далі:** лишається ALU-bound — AVX2/FMA для Q5_K (зараз scalar) + батч-`matvec` (кілька рядків за
  прохід вагами) **або** `PH-405b` llama-rs semantic parity **або** B6 cross-builds / B7 docker-slave.
  **7 локальних комітів** за сесію, **без push** (B11).

### Продовження 2 (2026-10-16, формати + max_tokens + Q5_0)

- **`rusty inspect`** реальної Qwen2.5-0.5B Q4_K_M: 896-ширинні тензори **не** K-quant — `output`
  **Q8_0**, `token_embd`/attn q·k·v·o/`ffn_gate`/`ffn_up` **Q5_0**, лише `ffn_down` (4864) **Q6_K**;
  домінантний скалярний шлях — **Q5_0** (~51 % MAC/токен).
- **`ca935d6`** perf(serve) **PH-501c** — `serve` хардкодив `max_new=8` (усі запити ~5 с незалежно
  від `max_tokens`); `http::max_tokens_of` (clamp 1..=512) → 1-токен **5.09 с → 0.67 с**.
- **`fc308af`** perf(kernel) — `dot_q5_0_avx2` (+dispatch, `QMat` type 6); golden точний; decode
  **~0.56 → ~0.33 с/токен**, 8-токен chat **5.06 → 3.01 с**, parity-path **128.5 → 52.6 с** (2.4×).
- **Далі:** FMA-варіанти Q5_0/Q6_K + батч-`matvec` (prefill/GEMM) **або** `PH-405b` llama-rs parity
  **або** B6 cross-builds / B7 docker-slave. **9 локальних комітів**, без push (B11).

### Продовження 3 (батч-prefill)

- Причини: `block_forward_seq` викликав `matvec` по рядку (де-квант `seq×`), а `serve` годував промпт
  по токену (`generate_incremental` → `block_forward_cached`).
- Додано `kernel::matmul_*_dispatch` + `dot_f32_dispatch` (AVX2+FMA; блок де-квантиться раз і
  переюзується по рядках), `QMat::matmul`, батчений `block_forward_seq`,
  `block_forward_prefill_cached` + `DenseDecoder::prefill`.
- Заміри: parity-path **52.2 → 19.2 с**; serve ~700-токен промпт **>180 с → ~80 с**; decode без
  регресії. Golden розширено на інкрементальний шлях.
- **Далі:** упакований int8 SIMD dot (`q8_0×q8` `vec_dot`) або register-blocked GEMM **або**
  `PH-405b` llama-rs parity **або** B6 cross-builds / B7 docker-slave. **13 комітів**, без push.

## AUTO-DIRECTIVE (owner, 2026-10-15; оновлено 2026-10-17) — на наступне `agi`

Наступного `agi`: **НЕ питати вибору бенду** — одразу брати з черги.

- **`PH-405b` — ✅ ЗРОБЛЕНО (2026-10-17).** llama-rs semantic greedy-parity: знайдено
  й виправлено **биті q/k/v-біаси Qwen2** (`BlockWeights.{q,k,v}_bias`) + **RoPE
  GPT-J→NeoX**; наш greedy тепер байт-у-байт із записаним оракулом (55/12/48). Гейти —
  `tests/moe_parity.rs` (Гейт A tokenizer byte-exact PASS; Гейт B офлайн+live-оракул),
  факти — `docs/research/llama-rs-parity.md`. Golden перезахоплено. Деталі — «Продовження 6»
  у `docs/HANDOFF_NEW_SESSION.md`.
- **Наступний пункт черги — `PH-507`** (owner-директива 2026-10-15, bunke-rock-подібний
  provider-row у hub-каталозі; § «Session wire» нижче): зареєструвати `rusty serve` як
  `kind:local` рядок у GSV `src/boxes/omni/catalog.rs` (`host_ready`-гейт на
  `target/live/rusty_heartbeat.json` + gguf-файл, fail-open), авто-перемикання local↔cloud
  у `GET /api/omni/route`, SSE через hub, WiFi multi-host (LAN-first band 233).
  Product = `rusty-engine`, той же ticket-цикл create→claim→drain→done, один локальний
  коміт, **без push** (B11). Тікети/keep-live за `agi`-скіллом; перед зрізом перевірити
  hub-сервіси через `gsv_keep_live`.
- Якщо `PH-507` заблокований (потрібен GSV-side код/дозвіл) — альтернативи: `PH-202/203`
  (repack/SIMD), `PH-501` (axum serve), B6 cross-builds / B7 docker-slave.

## Session wire (виправлено 2026-10-15, owner-згода на push налаштувань)

- **Проєктна папка — тільки `S:/rust/rusty-engine`.** Причина вчорашнього збою: OpenCode
  Desktop зберіг проєкт `rusty-engine(moe)` під неіснуючим шляхом
  `S:/rust/CoreBoard_Labs_shop/rusty-engine(moe)` (у лозі — `ENOENT` + сесійні `Aborted`),
  тому повідомлення «з’їдалися», а модель-сервер виглядав мертвим. Видаліть той запис із
  recents Desktop-а; відкривайте або `S:/rust/rusty-engine`, або `S:/rust/GSV/gsv.code-workspace`
  (GSV-first) — папка rusty-engine там уже є.
- **`opencode.json`** (корінь проєкту) вмикає в будь-якому вікні: skills = GSV-кит
  (`S:/rust/GSV/.agents/skills`, не копіюємо сюди за `GSV_AGI_PATH`), MCP `gsv_mcp_openbot` =
  stdio `S:/rust/GSV/target/live/gsv-mcp.exe --repo-root S:/rust/GSV` (тікети/keep-live/board),
  instructions = GSV `AGENTS.md` + локальний `AGENTS.md` + цей NEXT. Тобто `agi` у цьому
  вікні = той самий дрain-контур, що у GSV/poolAI.
- **Модель-сервери / routing (перевірено 2026-10-15: HTTP 200 + SSE, hub-route Ok):**
  чат-тести йдуть **через hub omni** `http://127.0.0.1:9999/api/omni/v1` — provider `gsv`
  у глобальному `~/.config/opencode/opencode.jsonc` з **обома локальними тієрами**
  (`lama-2.8` deep, `lama-1.5` fast). Hub сам перемикає ланку: живі локальні моделі →
  локальний лейн, хмара-фолбек з cooldown-скіпом мертвих хостів (offline-first, `data/omni_quota.json`).
  Прямі fallback-провайдери без хаба: `llama-deep` = `:8080/v1` (довгенні генерації — напряму
  швидше), `llama-fast` = `:8082/v1`. **Дев-сесії `agi` — на `opencode-go`** (дефолт; хмара,
  завжди жива; `small_model` теж `qwen3.8-flash`, бо muse-spark рядки блокує Privacy-політика
  Go — звідси й «ламалося» вчора на тайтлах). Ручне перемикання в OpenCode — вибір моделі;
  у Cursor — Settings → Models → **Override OpenAI Base URL** → `http://127.0.0.1:9999/api/omni/v1`
  + custom-моделі з `GET /v1/models` (canon `GSV_OMNI_CATALOG.md`). OmniRoute `:20128` —
  **policy-down**, не піднімати з `agi`.
- **Owner-директива (2026-10-15) — `PH-507` після `PH-405b`:** зареєструвати `rusty serve`
  як **локальний provider-row у GSV hub-каталозі за зразком `bunke-rock`** (`catalog::host_ready`
  гейт на heartbeat `target/live/rusty_heartbeat.json` + модельний файл, kind `local`, fail-open):
  OpenAI-compat поверх B5-serve, WiFi **multi-host** (LAN-first band 233, `0.0.0.0`-bind через
  hub), авто-перемикання local↔cloud у `GET /api/omni/route`, SSE-стрімінг. Тест-чат чіпає
  `rusty/*` так само, як нині `gsv/lama-*`. Код хаба — окремий зріз у `S:/rust/GSV`
  (`src/boxes/omni/catalog.rs`), код двигуна — у `S:/rust/rusty-engine`.
- **Push-згода owner (2026-10-15):** створено GitHub-remote `platinoff/rusty-engine` —
  **private**. MIT-публікація та загальний push-контур коду залишаються за B11 (`PH-1105`);
  коміти продукту далі — локальні, без push, дозвіл only owner-згодою.

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
[--max-tokens N]` (PH-306, numeric tokens). **+ B2 `PH-204` fused K-quant dot/gemv** —
`dot_q{4,5,6}_k`+`matvec_q{4,5,6}_k` per-block stack `[f32;256]`, relative-parity vs
`dot(dequant_qX_k, b)`. **+ B4 `PH-404` expert-cache LRU** (`src/cache.rs` — budgeted
LRU + pin + prefetch, longest-first → atomic ids); **`src/gpt2.rs`** std-only GPT-2/Qwen2 pre-tokenizer
(`gpt2_pre_split`, wired into `Bpe::encode`). **+ B2 `PH-202` repack** (`src/repack.rs`,
`RepackedQ4_0`/`Q8_0` — contiguous scales + de-interleaved nibbles, **exact** vs the
fused kernels). **+ B3 `PH-306b` text pipeline** (`src/textgen.rs` `TextRunner`
= Bpe + DenseRunner; `rusty run -p "text"` prints generated text; `tests/text_gen.rs`
synth round-trip). **+ B3 `PH-307` incremental decode + KV** (`forward::LayerKv`+
`block_forward_cached`, `runner::DenseDecoder` — O(1)/крок, `full == incremental`
еквівалентність). **+ B3 `PH-304b` sampler wiring** (`GenerationOptions.repetition_penalty`,
`infer::SeedRng` xorshift64*, `DenseRunner::generate_with` — temp/top-k/top-p/repetition,
`temp<=0`==greedy). **+ B4 `PH-404b` MoE forward через cache** (`moe_layer_forward_cached`
→ `get_or_load` per expert; == resident forward при детерм. loader). **+ B2 `PH-201d` fused Q5_0 dot/gemv** (closes quant matrix). **+ B4 `PH-401b` exact `noaux_tc` routing** (`route_noaux_tc` — group score = sum of top-2, distinguishes from `route_group`). **+ B4 `PH-403b` MoE block forward** (`attention_forward_cached` core + `moe_block_forward`; 1-expert==dense 1e-5). **+ B5 `PH-501a` std-only HTTP serve** (`src/http.rs` pure `route()` + `src/serve.rs` TcpListener pump + `rusty serve`; loopback wire-тест). **+ B5 `PH-502` SSE-стрімінг** (`sse_body` chunk-події + `[DONE]`, `route` → `text/event-stream`). **+ B5 `PH-504` keep-live heartbeat** (`hub::Heartbeat`/`is_fresh`/`write_heartbeat`; serve stamp при startup+кожне conn; CLI `rusty heartbeat`). **+ B5 `PH-505` metrics** (`hub::Metrics` record/tok_per_sec + `GET /metrics`). **+ B5 `PH-501b` serve_with_model** (real TextRunner → `POST /v1/chat/completions` BPE→generate→decode; `extract_message_content` std JSON reader; CLI `rusty serve -m`). **146 тести** (134 lib + 5 dense + 3 text + 4
scaffold), clippy `-D` 0, ratio Rust ~96%. Модель у `data/models/` (gitignored).
**Checkpoint:** зріз стану — `docs/STATUS.md`; порівняння рушіїв —
`docs/research/engine-landscape-2026.md`; offset-семантика — `docs/research/`.

## Порядок `agi` за логікою побудови (strict queue)

Порядок смуг (roadmap): `фундамент → GGUF → quant → dense → MoE → **serve → build-matrix →
docker → supervisor → dist-MoE → UI → distribs/publish**`. B1–B4 закриті (крім parity/run на
зовнішніх даних). Черга `agi` по максимуму, кожен зріз = тікет + один локальний коміт:

1. **B5 serve** — `PH-504` keep-live heartbeat `target/live/rusty_heartbeat.json` (15s tick,
   `hub.rs`) · `PH-505` метрики tok/s·RAM·active-experts endpoint · `PH-503` prompt-slot.
2. **B6 build-matrix** — `PH-601` `rusty.exe` gnu+msvc · `PH-602` Linux x86_64 · `PH-603`
   aarch64 · `PH-604` Android arm64 (cargo-ndk/Termux, **no Docker**) · `PH-605`
   `cargo xtask build-matrix` + SHA-256 · `PH-606` feature-gated backends.
3. **B7 docker** — `PH-701` bollard client · `PH-702` образ · `PH-703` LAN/Wi-Fi expose
   (band-233 LAN-first) · `PH-704` dry-run/`--live` gate · `PH-705` lifecycle contract.
4. **B8 supervisor (Rust k8s-lite)** — `PH-801..807` registry/heartbeat/lease/placement/
   watchdog + MCP `gsv://` fleet.
5. **B9 dist-MoE** — `PH-901..906` expert-parallel sharding + cross-node KV.
6. **B10 UI** — `PH-1001..1006` Rust-рендер `src/ui.rs` + ui-glue LOC-ratchet.
7. **B11 distribs/publish** — `PH-1101..1106` semver/CHANGELOG · `xtask dist` · MIT
   attribution · **GitHub push (owner згода — єдиний дозволений push)**.

**Зовнішні гейти (чекають owner/підлогу, NOT pure-`agi` std-slice):** ~~`PH-405` llama-rs
greedy-parity~~ ✅ **`PH-405b` done 2026-10-17** · `PH-406` real MoE `.gguf` у `data/models/` ·
`PH-203` SIMD (`core::arch`, arch-gated). Ці робляться окремими зрізами коли підлога готова.

## llama-rs parity (`PH-405b`) — ✅ ЗРОБЛЕНО (2026-10-17)

Гейт уже живе в `tests/moe_parity.rs`; запуск (оракул — на тій самій 0.5B, `:8085`):

```bash
# оракул (S:/rust/llama-rs) — НЕ чіпати живої llama-fast :8082 (1.5B)
./target/release/llama_serve.exe models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  --port 8085 --model-name qwen05-oracle --max-tokens 12 --temperature 0.0
# Гейт A (tokenizer byte-exact) + Гейт B офлайн (записаний оракул, без мережі)
cargo test --release --test moe_parity -- --ignored
# Гейт B live (потребує PH405B_ORACLE_URL)
PH405B_ORACLE_URL=http://127.0.0.1:8085/v1/chat/completions \
  cargo test --release --test moe_parity -- --ignored greedy_parity
```

Знайдені й виправлені помилки: **q/k/v-біаси Qwen2** (головна) + **RoPE NeoX**. Результат —
байт-у-байт збіг (55/12/48). Деталі — `docs/research/llama-rs-parity.md`, «Продовження 6»
у HANDOFF.

## Контекст-дисципліна (кожен бенд «мільйон»)

Один agi-зріз = обмежений тікетами → std-only попереду → deps-зріз окремо → HANDOFF/NEXT
несуть стан → **один локальний коміт**. Фактаж кешуємо в `docs/research/`, ресерч — у
subagent-ах. Не тягну весь llama.cpp/vLLM у вікно за раз.

## Тікети на дошці

- `rusty-engine-bootstrap` (B0) — 10/10 **done** (сесія 2026-10-14).
- `rusty-engine-gguf` (B1) — банда **завершена**: `PH-101..107` + **`PH-105b`** (BPE) **done** (2026-10-14).
- `rusty-engine-moe-core` (B4) — `PH-401/402/403` + `PH-406a` + **`PH-404` expert-cache LRU** + **`PH-404b` MoE forward через cache** **done**; `PH-405/406/407` **open** (реальна MoE GGUF + llama-rs-оракул).
- B3 primitives + **`PH-305a/c/d/e`** + **`PH-305`/`PH-306`** **done** (real Qwen2.5 24-layer
  dense forward OK, CLI `rusty run`). `PH-305b` (HF BPE) + `PH-405` (llama-parity) **open**.
- B2 — `PH-201` + `PH-201+` + **`PH-201c` (Q5_0)** + **F32/F16 dequant** **done**; `PH-202..204` **open**.

## Reminder (session law)

Global-first (`S:/rust/GSV/AGENTS.md`); MSYS2 bash; ratio 95–100%; Rust-MoE-ядро (не лінкуємо
C++); Docker-only-Win/Linux + Android-no-Docker; GSV hub = master supervisor; dry-run default;
push тільки B11.
