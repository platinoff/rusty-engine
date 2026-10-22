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

### Продовження 4 (2026-10-17, B2 SIMD bench — `PH-203`/`PH-205`)

- **`benches/quant.rs`** (новий, `PH-205`) — std-only bench (**без `criterion`**), `harness=false`,
  детермінований xorshift; для кожного quant-ядра друкує scalar-час, `*_dispatch`-час, speedup і
  `|Δ|` (sanity: SIMD == scalar-оракул). `Cargo.toml` → `[[bench]] name="quant"`.
- **Заміри (release, Qwen-форма 896²):** dot `q4_0` **2.32×**, `q8_0` **1.78×**, `q5_0` **2.12×**,
  `q4_k` **3.94×**, `q6_k` **5.31×**; matvec `q4_0` **2.30×**, `q8_0` **1.78×**; `|Δ| ≤ 4e-3`
  (SIMD==scalar до f32-округлення).
- **Фікс фікстур, не ядер:** перші прогони давали NaN/велике `|Δ|` — це були артефакти даних
  (f16-scale з random-байтів → Inf/NaN; `fill_floats` ділив на 4096, не 2²⁴; `Q6_K` має `d` у
  хвості блоку, offset **208**, не 0). Після фіксу — тихо.
- **`PH-203` x86 закрито:** SSE2 `Q4_0`/`Q8_0`; AVX2 `Q4_0`/`Q5_0`/`Q8_0`/`Q4_K`/`Q6_K`;
  `dot/matvec/matmul_*_dispatch` (AVX2→SSE2→scalar) підключені до forward через `qmat.rs`. Лишився
  **NEON arm64**.
- Gate: `cargo test` **197 green**, clippy 0, fmt ok. **23 локальних комітів**, без push (B11).
- **Далі:** NEON (`PH-203` arm64) **або** `PH-501` (axum serve) **або** B6 cross-builds / B7 docker-slave.

### Продовження 5 (2026-10-17, PH-505 residual — RAM у `/metrics`)

- **`hub::ram_bytes()`** (новий, std-only, без крейтів): Windows — working set через
  `K32GetProcessMemoryInfo` (FFI `extern "system"`, kernel32/psapi); Linux — resident-сторінки з
  `/proc/self/statm` × 4096; інакше 0. `Metrics::to_json` тепер семплить живий RSS → `/metrics`
  віддає `ram_bytes` + `ram_mb` (окрім `tok_per_sec`/`active_experts`/`peak_experts`).
- **`ChatOutcome.active_experts`** — поле додано й протягнуто в `serve` → `metrics.record(..)`;
  на dense-шляху = 0 (наповниться з `PH-406` MoE, коли зʼявиться MoE-модель).
- Перевірка: unit `ram_bytes_reads_the_process_where_supported` + `metrics_json_shape`; live
  `:8090` → `"ram_bytes":1502031872,"ram_mb":1432.4` (модель завантажена). Окремий прогон на
  `:8095` (stub) → `ram_mb 4.6`.
- Gate: `cargo test` **198 green**, clippy 0, fmt ok. **24 локальних комітів**, без push (B11).
- **Далі:** NEON (`PH-203` arm64) **або** `PH-501` (axum serve) **або** B6/B7; `PH-505` закрито по
  суті (лишається наповнити `active_experts` з MoE-шляху `PH-406`).

### Продовження 6 (2026-10-18, PH-203d — AVX2+FMA fused dots)

- **`dot_q4_0/q5_0/q8_0_avx2`** переписано на **fused `_mm256_fmadd_ps` + два незалежні
  акумулятори** (low/high halves) замість `mul_ps`+`add_ps` в один акумулятор (розриває ланцюг
  залежності по `add`). Q8_0 додатково позбувся скалярного `[i32;8]`-гезеру → `_mm256_cvtepi8_epi32`.
- Dispatch (`dot_q4_0/q5_0/q8_0_dispatch`) і явні parity-тести гейтовано `avx2 && fma` (FMA є на
  кожному AVX2-хості з Haswell 2013). Q4_K/Q6_K лишились на `avx2` (не чіпали).
- Bench (`benches/quant.rs`, best-of-5, 8192-elem dot / 896² matvec): dot q8_0 **1.72→2.09×**
  (6740→5185 ns), q4_0 2.31→2.41×, q5_0 2.14→2.30×; matvec q8_0 **1.77→2.06×** (605k→518k ns).
  `|Δ|` без регресії (q8_0 1.34e-3→**9.8e-4** — FMA точніший за роздільні mul+add).
- Live `:8090` decode **2.27→2.68 tok/s** (Qwen2.5-0.5B Q4_K_M) — Q8_0 = LM-head, тож виграш
  одразу видно на реальному шляху.
- Gate: fmt 0, clippy `--all-targets -D warnings` 0, **198 тестів** green. **25 локальних
  комітів** (`c2102bb`), без push (B11).
- **Далі:** NEON (`PH-203` arm64, все ще без aarch64-тулчейну) **або** `PH-501` (axum serve)
  **або** `PH-502` (справжній per-token SSE) / B6/B7.

### Продовження 7 (2026-10-18, PH-203e — row-parallel matvec/matmul)

- **`QMat::matvec`/`matmul`** тепер розбивають проєкцію по рядках на воркерів
  (`std::thread::scope`), коли робота ≥ `PAR_MAC_MIN` (2¹⁸ MAC). Старі тіла збережено як
  `matvec_serial`/`matmul_serial` (fallback для типів без fused-ядра і малих проєкцій).
  Кількість воркерів — `RUSTY_THREADS` (1–256) або `available_parallelism()`.
- **`src/kernel.rs`**: `quant_row_bytes(ggml_type, n)` (страйд рядка), `quant_row_dot(t, row, x)`
  (один рядок), `matmul_dispatch(t, w, m, n, x, rows)`.
- **Головна пастка (виправлено):** перша версія ділила `matmul` по **вхідних** рядках — і кожен
  воркер де-квантизував усю матрицю (`matmul_reuse` кешує блок на всі вхідні рядки), тобто
  prefill не вигравав. Правильно ділити по **рядках ваг** (вихідних стовпцях) і збирати блоки
  назад у row-major `Out`. Після фіксу: prefill 45.6→24.7 с.
- **Біт-у-біт:** кожен рядок виконує ту саму скалярну послідовність, тож результат не залежить
  від кількості потоків — 2 нові тести `parallel_matvec_is_bit_identical_to_serial`,
  `parallel_matmul_is_bit_identical_to_serial`.
- A/B (321-tok, той самий бінарник/модель): prefill `RUSTY_THREADS=1` 45.6 с → `12` 24.7 с
  (**1.85×**); decode 2.68 → **4.2–5.1 tok/s**. Стеля = Ryzen 5 5500U (6 фіз. ядер / 12 потоків)
  під ~60% навантаженням інших GSV-сервісів.
- Gate: fmt 0, clippy `--all-targets -D warnings` 0, **200 тестів** green. **28 локальних
  комітів** (`e2646a4` + `6336afb`), без push (B11).
- **Далі (perf):** сталий thread-pool замість `thread::scope` на кожну проєкцію (~120 spawnів/токен,
  оцінно ~20% часу) — дасть ще ~1.2–1.3×.

### Продовження 8 (2026-10-18, PH-203f — persistent worker pool)

- **Спершу виміряли, чи варто:** додали в `benches/quant.rs` секцію «parallel projection vs spawn
  overhead». `thread::scope` з 12 потоками = **750–820 µs** spawn+join, тобто при ~120 проєкціях
  на токен це ~**90 ms/токен** — більше, ніж сама робота проєкції. Оцінка «~20%» була занизькою
  (~46%), тож пул робився вже не навмання.
- **`src/pool.rs`** (std-only): 12 від'єднаних воркерів, черга на `Mutex`+`Condvar`, глобальний
  `pending`-лічильник. `parallel_for(jobs, |i| …)` чекає `pending == 0`, тому позичення в замиканні
  можна безпечно стерти через `Handle` (сирий `*const ()` + мономорфізована `call`-функція) —
  `transmute` тут не компілюється (вимагає `F: 'static`). Панель-паніки ловить `catch_unwind`,
  інакше воркер і лічильник гинуть.
- **`QMat::matvec`/`matmul`** тепер віддають блоки в `Mutex<Vec<(usize, Vec<f32>)>>`, а не пишуть у
  нарізані `&mut`-слайси. **Пастка:** кількість job-ів мусить бути `m.div_ceil(chunk)`, а не
  `threads` — інакше останній воркер отримує `base > m` і панікує на `out[base..]`.
- **Не можна вкладати** `parallel_for` всередині job-а (сам себе чекатиме) — зафіксовано в doc-comment.
- Результат: gate/up 4864×896 Q5_0 **1.270→0.684 ms** (4.3–5.1× vs serial, було 2.66×);
  prefill matmul 4864×896×961 **1009→247 ms (4.08×)**; live decode **4.4→7.2–7.9 tok/s**
  (накопичувально 2.27→7.8 ≈ 3.4×). Біт-у-біт збережено (ті самі 2 тести).
- **Prefill не покращився** (321 tok ≈ 25 с і до, і після): matmul тепер 4.08×, але ~19 з 25 с — це
  послідовний attention/KV-шлях. Це наступний великий перф-зріз, не кернели.
- **Обережно з вимірами:** на цьому боксі цифри decode стрибають 2–8 tok/s залежно від того, що
  ще працює (свіжий процес дає стабільні 7.2–7.9; після важкого prefill/бенчу — 2–4). Міряти
  лише на щойно перезапущеному `:8090` і кількома послідовними викликами.
- Gate: fmt 0, clippy `--all-targets -D warnings` 0, **203 тести** green. **30 локальних комітів**
  (`72b570a`), без push (B11).
- **Далі:** паралельний/векторизований attention prefill (найбільший залишок), далі NEON (`PH-203`
  arm64 — без тулчейну) / `PH-501` (axum) / `PH-502` (per-token SSE).

## AUTO-DIRECTIVE (owner, 2026-10-15; оновлено 2026-10-17) — на наступне `agi`

Наступного `agi`: **НЕ питати вибору бенду** — одразу брати з черги.

- **`PH-405b` — ✅ ЗРОБЛЕНО (2026-10-17).** llama-rs semantic greedy-parity: знайдено
  й виправлено **биті q/k/v-біаси Qwen2** (`BlockWeights.{q,k,v}_bias`) + **RoPE
  GPT-J→NeoX**; наш greedy тепер байт-у-байт із записаним оракулом (55/12/48). Гейти —
  `tests/moe_parity.rs` (Гейт A tokenizer byte-exact PASS; Гейт B офлайн+live-оракул),
  факти — `docs/research/llama-rs-parity.md`. Golden перезахоплено. Деталі — «Продовження 6»
  у `docs/HANDOFF_NEW_SESSION.md`.
- **`PH-507` — ✅ E2E ЗРОБЛЕНО (2026-10-17).** Провайдер `rusty-serve` у hub-каталозі працює
  через живий хаб `:9999` → `rusty serve :8090`: JSON-відповідь і SSE-пасстру (`.x-gsv-mcp: sse`).
  Корінь давнього `null` — **незакритий корінь** у JSON `textgen::chat_response` (фікс
  `chat_completion_body` + тест `worker::parse_json`); плюс читання тіла за `Content-Length`
  (`serve.rs`/`http.rs`). Деталі — «Продовження 7» у `docs/HANDOFF_NEW_SESSION.md`. **Лишається
  в межах PH-507:** WiFi multi-host (band 233) і provider `rusty/*` у OpenCode/Cursor.
- **Next:** або довиконати решту `PH-507` (multi-host + IDE-chat), або **NEON (`PH-203` arm64)**,
  `PH-501` (axum serve), B6 cross-builds / B7 docker-slave. **B2 x86 SIMD + bench (`PH-203`/`PH-205`)
  — ✅ зроблено** (див. «Продовження 4»).

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
- **Push-згода owner (2026-10-15, оновлено 2026-10-21):** GitHub-remote `platinoff/rusty-engine` —
  **private**. 2026-10-21 власник зняв B11-обмеження: сесію закрито **push-ом** (`origin/main` =
  `ef0378a`, 35 комітів). Далі — звичайний контур: один коміт на зріз + `git push` у кінці сесії;
  MIT-публікація/license/CHANGELOG лишаються за B11 (`PH-1105`).

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

### Продовження 9 (2026-10-18, PH-203g — RoPE раз на позицію + повернутий KV)

- **Знайшли справжнє вузьке місце prefill:** `block_forward_prefill_cached` кликав `rope()`
  на кожну пару (query, key) — ~723k разів на 321 токен, — а `rope()` сам рахує `powf` **на
  кожен елемент** і алокує `Vec`. Це і були ті ~19 з 25 секунд, а не attention-математика.
- **`infer::rope_freqs(d, base)`** — таблиця `base^(-2i/d)` один раз на модель;
  `rope_in_place` крутить голову з нею; `rope()` лишився обгорткою (той самий f32-результат).
- **KV-кеш тепер зберігає вже повернуті ключі** (був контракт «raw, rotate on read»). Це
  прибирає з decode повторне обертання кожного кешованого ключа на кожну голову на кожен
  токен. Оновлено обидва письменники (`attention_forward_cached`,
  `block_forward_prefill_cached`) і єдиного читача; еквівалентність тримає
  `block_forward_cached_matches_seq_over_prefix`.
- **Prefill attention:** Q/K-рядки повертаються цілком раз на позицію (а не раз на голову на
  query), один scratch для scores/probs замість `Vec` на кожен query. V-накопичення
  переставлено на `for j { for d }` — порядок сум на елемент той самий (j зростає), але
  внутрішній цикл суміжний → векторизується.
- **Gates:** оракул llama-rs (55/12/48) і `PH-405` greedy fingerprint (re-prefill **і**
  incremental) проходять без змін. Fingerprint зловив реальний баг у середині зрізу —
  загублений `add_bias_rows(v_all, v_bias)` у `block_forward_seq` (не в тому файлі, який я
  правив останнім, тому «на око» не видно).
- **Замір (свіжий процес, 12 потоків):** prefill 321 tok **25.5 → 5.89 s** (4.3×),
  961 tok **190.9 → 29.1 s** (6.6×); decode **7.4 → 10.35 tok/s**. Накопичувально з початку
  сесії: decode 2.27 → 10.35 (**4.6×**), prefill 45.6 → 5.9 s (**7.7×** на 321 токені).
- Gate: fmt 0, clippy `--all-targets -D warnings` 0, **203 тести** green (+ `--ignored`:
  dense_gen, 3× moe_parity, numeric_parity). **32 локальні коміти** (`5f1641b`), без push.
- **Далі:** (а) паралелити attention по головах через `pool::parallel_for` (14 голів,
  залишок prefill), (б) q8_0-ядро: seril matvec 896² = 0.55 ms ≈ 1.5 GMAC/s — значно нижче
  очікуваного для AVX2, схоже на вузьке місце в конвертації i8→f32.

### Продовження 10 (2026-10-18, PH-203h — head-parallel prefill attention)

- Після `PH-203g` у prefill лишився **послідовний цикл по головах**. Голови незалежні й
  пишуть у власні `head_dim`-блоки, тому їх можна роздати наявному пулу без зміни результату.
- **`src/forward.rs`**: `attention_head_ctx(...)` — один per-head кернел; його ж
  використовують обидва prefill-шляхи (`block_forward_seq` з prompt-local K/V і
  `block_forward_prefill_cached` з KV-кешем), бо layout однаковий (`pos_end × kv_dim`).
  `attention_prefill(...)` збирає голови в `seq × q_dim`; паралельно — від
  `PAR_SEQ_MIN = 32` токенів (нижче hand-off у пул дорожчий за роботу).
- Два дубльовані inline-цикли attention видалено; числовий порядок сум на елемент не змінено.
- **Замір:** prefill 961 tok 29.1 → **23.5 s** (1.24×; від початку сесії 190.9 s = **8.1×**),
  321 tok 5.89 → 5.56 s; decode без змін (10.4–10.9 tok/s).
- Gate: fmt 0, clippy `-D warnings` 0, **203 тести** green + усі `--ignored` (dense_gen,
  3× moe_parity, numeric_parity). **34 локальні коміти** (`fecffa1`), без push.
- **Далі:** залишок prefill — сам matmul (17 GMAC/s). Великий резерв: q8_0-ядро
  (serial matvec 896² ≈ 0.55 ms ≈ 1.5 GMAC/s — підозріло мало для AVX2; ймовірно конвертація
  i8→f32 і відсутність int8-множення). Це вже зміна числової схеми, тому окремий зріз з
  оракулом.

### Продовження 11 (2026-10-21 — синк боксів/доків + push, старт наступної сесії)

- Owner-директива 2026-10-21: синхронізувати бокси/доки зі станом репо **і зробити push**
  (B11 «без push» знято власником; GSV-канон: `git push` + самарі — останній крок сесії).
- Стан гілки: **35 локальних комітів** (`ef0378a`, PH-203f/g/h + доки) → `origin/main` (**запушено 2026-10-21**).
- Дошки GSV (2026-10-21): створено два open-тікети на наступну сесію —
  `t-1792585619484828300` (екосистема: `rusty-serve` у `routing.fallback_order` + `host_ready` + keep-live)
  і `t-1792585623805073700` (двигуни: q8_0 integer-dot kernel з oracle + golden).
- **Наступне (окремий зріз):** q8_0-ядро з int8-множенням (зараз serial matvec 896² ≈ 0.55 ms ≈ 1.5 GMAC/s,
  конверсія i8→f32 на блок); або matmul throughput (~17 GMAC/s). Зміна числової схеми →
  оракул + `RUSTY_CAPTURE_GOLDEN=1`, і **завжди** `cargo test --release -- --ignored` після правок
  `src/forward.rs`. Decode міряти лише після свіжого рестарту `:8090`.

### Продовження 12 (2026-10-21, PH-203i — f16-widening + q8_0 integer-dot)

- Міф про «i8→f32 конвертацію» розвіяно: вузьким місцем був `quant::f16_to_f32` з `f64`+`powi`
  **на блок**. Точне бітове розширення (subnormal/Inf/NaN збережено, enumeration-тест на всі
  `u16`) дало dot_q8_0 **9.55×** (5.34 → 0.64 µs), matvec 896² **11.8 GMAC/s** (було 1.5),
  QMat-prefill matmul 5.44×; числа forward **не змінились** (`--ignored` 5/5 без recapture).
- Live `:8090` після свіжого рестарту: decode **27.6 tok/s** (було 10.9), prefill 961 tok
  **23.46 s** (без змін — q8_0 лише фінальна проєкція).
- Для тікета додано `Q8Activation` + `dot_q8_0_i8` (scalar/SSE2/AVX2, біт-у-біт) і
  `matvec_q8_0_i8`/`matmul_q8_0_i8`; після f16-фіксу int8 **повільніший** за float (0.89×), тому
  в `QMat` не підключений. Див. `benches/quant.rs`, HANDOFF §Продовження 16.
- **Далі:** залишок prefill — q5_0/q4_k matmul (QMat 5.4× над serial) і, за потреби,
  head-parallel matmul; decode-стеля тепер інша — профілювати наступний домінантний блок.
- Gate: fmt 0, clippy `-D warnings` 0, **193 lib + 19 інтеграційних** + 5 `--ignored`.

### Продовження 13 (2026-10-21, row-batched prefill matmul — виміряно, відкочено)

- Гіпотеза: `matmul_reuse` латентнісно впирається в один 4-FMA-ланцюг + hsum на кожен (блок, рядок);
  батч 4 вхідних рядків на один де-квантований блок мав дати ILP без зміни порядку сум (біт-у-біт).
- Результат: `matmul_reuse_avx2` + scalar-оракул (`matmul_row_batched_is_bit_identical_to_scalar`,
  біт-у-біт зелено), але **повільніше**: 4864×896×961 — старий 321 ms, refactor 213 ms, батч
  **573–648 ms** на тому ж бінарнику. **Відкочено**; дерево чисте. Узгоджується з §Продовження 4
  (f32-matmul нечутливий до форми циклу).
- Урок про заміри: незмінений код давав 156 ms у сесії PH-203i і 321 ms сьогодні (фон ~2×) —
  A/B тільки в межах одного прогону.
- **Далі:** перед новими правками prefill виміряти частку attention у 23.5 s; для matmul —
  repack layout (PH-202) або K-quant dequant-once на 256, не перестановка циклів.

### Продовження 14 (2026-10-22, PH-203k — activation tiling + row-split проєкцій)

- Знайдено справжній ботлнек prefill-matmul: це **трафік `X`**, не форма циклу. Старий
  `matmul_reuse` сканував усі 3.3 MiB активацій на кожен ваговий рядок (≈16 GiB з L3 на проєкцію),
  а `QMat::matmul` ділив роботу по стовпцях, тобто 12 воркерів годували той самий потік.
- Зроблено: `MATMUL_ROW_TILE = 16` (57 KiB, L2) в `kernel::matmul_reuse` — біт-у-біт, тест
  `matmul_activation_tiling_is_bit_identical`; `QMat::matmul` ділиться по **вхідних рядках**
  (кожен воркер ~290 KiB `X`, ваги ≈36 MiB на всіх), старий варіант — `matmul_split_cols` для A/B.
- Заміри (один процес): serial 4864×896×961 1469→**897–1103 ms**; decomposition A/B
  cols→rows **248→190 ms (1.30×)**, повтор 344→291 (1.18×). Live prefill 961 tok 26.9–27.3 s на
  завантаженому боксі — end-to-end не заявляємо (rebook + llamas + gsv у фоні).
- **Порт:** rebook зайняв `127.0.0.1:8090` (rusty слухає `0.0.0.0:8090`) — проби гнати на LAN:
  `bash target/live/prefill_probe_lan.sh`, не через `127.0.0.1`.
- **Далі:** латентність `dot(32)`+hsum на блок — повний f32-акумулятор по рядку (один hsum)
  змінює порядок сум, тому це окремий числовий зріз з `RUSTY_CAPTURE_GOLDEN=1`; інакше
  dequant-once 256 / repack PH-202 / замір частки attention.

### Продовження 15 (2026-10-22, PH-203l — full-row f32 акумулятор, числовий зріз)

- `kernel::matmul_reuse` більше не кличе `dot` на кожен 32/256-блок: ваговий рядок
  дейквантизується **цілком** (`dequant_row`) у буфер `n` f32 і дотримується проти рядка `X`
  одним викликом `dot_f32_dispatch` → **один hsum на `(row, r)`** замість 28 (для K-квантів — 3–19).
- Порядок сум інший (8-смуговий акumulator по всьому рядку vs hsum-на-блок + f32-додавання),
  тому це числовий зріз. Старий шлях лишено як `matmul_reuse_blocks` +
  `matmul_q5_0_blockwise` (`#[doc(hidden)]`): оракул і бенч-A/B. Новий тест
  `matmul_row_wise_accumulator_matches_block_wise` тримає розбіжність ≤1e-4 відносної на
  896-широкому рядку (факт: ~2e-5).
- **Golden не перезахоплювався:** `cargo test --release -- --ignored` з новим акумулятором
  пройшов без recapture — greedy-fingerprint (re-prefill + incremental/serve) **не зрушив**.
  Це головний доказ, що зміна — лише f32-округлення, а не семантика.
- Заміри (`cargo bench --bench quant`, один процес, min-of-7 interleaved):
  single-thread Q5_0 4864×896×961 **block 805.9 → row 606.3 ms (1.33×)**; serial 1469 (PH-203j)
  → 859 ms; QMat parallel 92.5 ms = **45 GMAC/s** (9.29× над serial). Tile sweep став плоским
  (961→606.7, 16→610.8) — тайлінг більше нічого не дає, кандидат на спрощення.
- Гейт: fmt ok, **195 lib** + integration green, clippy 0, 5 ignored parity green.
- **Live:** release + `spawn_threads.ps1 -Threads 12`, проба на LAN — prefill 961 tok
  **18.93 / 19.04 s** (було 26.9–27.3 s у §14) — перший end-to-end виграш після PH-203h.
- **Далі:** tile-цикл тепер зайвий (спростити або прибрати `MATMUL_ROW_TILE`); далі — заміряти
  частку attention у prefill, dequant-once 256 для K-квантів, repack layout (PH-202), 4-рядковий
  блок (`dot` по 4 рядках `X` за один прохід ваги — більше FMA без нових hsum).

### Продовження 16 (2026-10-22, PH-203m — профайлер: matmul 84.4 %, attention 9.4 %)

- `src/profile.rs` + `RUSTY_PROFILE=1`: стадії `matvec`/`matmul`/`attention` атомарними
  лічильниками, звіт у `DenseDecoder::prefill` (serve) і один раз у `prefill_logits`; плюс
  розбивка `matmul` за формою. Вимкнено — один relaxed load на проєкцію.
- Live 961 tok: wall 20.2 s = matmul **84.4 %**, attention 9.4 % (1.9 s), other ~6 %. Attention
  не ботлнек (у 9× дешевший за matmul).
- **Down-проєкція `896×4864` = 10.0 GMAC/s** проти 34.8–36.0 у `4864×896`/`896×896`/`128×896`:
  24 down-виклики = ~60 % усього matmul-часу. Причина — `n=4864` → rowbuf 19.5 KiB + тайл
  `X` 312 KiB не влазять в L1, `dot` тягне з L2.
- Стеля: 1 FMA на 2 завантаження = ліміт load-портів AVX2 (6.9 GMAC/s ≈ 82 % від 8 MAC/цикл).
  Важіль — **register blocking на 4 рядки `X`** (4 FMA / 5 loads ≈ 1.9×, біт-у-біт); альтернатива
  для широких форм — K-chunking 1024 (числовий зріз).
- Логи профілю: `target/live/rusty_prof.{out,err}.log`; `target/live/profile.txt` вмикає профіль
  у `launch_serve.vbs` (видалити файл = вимкнути).


