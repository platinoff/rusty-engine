# rusty-engine — Roadmap

**Product id:** `rusty-engine` · Root `S:/rust/rusty-engine` · Rust-first plugin (MIT).

No open PH-* without a band. «agi» у сесії (GSV hub) створює тікети по цьому роадмепу
через `gsv_tickets_create { product:"rusty-engine", scenario_id }` / `gsv_tickets_hook`
(catalog id `rusty-engine-bootstrap` / `rusty-engine-gguf` / `rusty-engine-moe-core` /
`rusty-engine-vision`, додаються за потребою). Канон концепції: [`concept.md`](./concept.md).
Глобальні правила: `S:/rust/GSV/AGENTS.md`.

> **Модельна політика (owner, 2026-10-24).** Цільова модель двигуна =
> **MiniCPM-V 4.7** (HF `openbmb/MiniCPM-V-4.7-1B-gguf` + `...-35B-A3B-gguf`, Apache-2.0,
> GGUF-official). Qwen2.5-0.5B — **лише внутрішній числовий оракул** (dev-фікстура,
> golden `PH-405`, еталонні заміри). Інші модельні цілі з планів **прибрано**
> (Qwen3-30B-A3B / Mixtral → не підтримуємо як ціль; llama-rs лишається інструментом-
> оракулом, не продуктом). Віжен-бенд = **B12** (нижче).

## Легенда стану

- `[ ]` — не почато · `[~]` — в роботі · `[x]` — закрито (доказ + тест).
- ID формату `PH-<band><seq>` (band B4 → `PH-401..`).

## Порядок смуг (логіка побудови)

`фундамент → контейнер(GGUF) → ядра(quant) → dense-контроль → **MoE-ядро** → сервер →
портативність(exe/bin/apk) → Rust-docker-хости → Rust-супервізор(master/slave) →
розподілений MoE → Web-UI → дистр/версії/publish → **text-core qwen35-hybrid (B13)** →
**vision MiniCPM-V 4.7 (B12, image-prefix після B13)**`.

---

## Band B0 — Фундамент та реєстрація  **`[x]` (campaign 2026-10-14)**

> Скейтоут: cargo-workspace + канон-доки + MIT + локальні правила + UI-токени +
> реєстрація в GSV + перші тікети. 17 тестів (13 unit + 4 contract); clippy 0.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
|   | Workspace `rusty-engine` + bin `rusty` + `xtask` | `[x]` | `Cargo.toml`, `src/bin/rusty.rs`, `xtask/` |
| PH-001 | Модулі core/gguf/moe/infer/serve/node/hub (std-only) | `[x]` | `src/*.rs` + 13 unit-тестів |
| PH-002 | MoE top-k router у Rust + scaffold-contracts | `[x]` | `src/moe.rs`, `tests/scaffold_contracts.rs` |
| PH-003 | `cargo xtask check` (fmt→clippy -D→test) | `[x]` | `xtask/src/main.rs` + `.cargo/config.toml` |
| PH-004 | MIT LICENSE + Cargo metadata (author platinoff) | `[x]` | `LICENSE`, `Cargo.toml` |
| PH-005 | docs: concept + ARCHITECTURE | `[x]` | `docs/concept.md`, `docs/ARCHITECTURE.md` |
| PH-006 | docs: roadmap (B0..B11) + UI design-system | `[x]` | `docs/roadmap.md`, `docs/UI.md` |
| PH-007 | HANDOFF + NEXT + README + local AGENTS | `[x]` | `docs/HANDOFF_NEW_SESSION.md`, `README.md`, `AGENTS.md` |
| PH-008 | UI shell: index.html + css-токени + js-chrome-only | `[x]` | `ui/{index.html,css/style.css,js/app.js}` |
| PH-009 | Реєстрація в `GSV/docs/gsv/PRODUCTS.md` + workspace folder | `[x]` | row `rusty-engine` |
| PH-010 | Ticket scenarios на GSV-дошці + створення B0-тікети | `[x]` | `ticket_scenarios.json` |
| PH-011 | git init + перший локальний коміт (без push) | `[x]` | commit у `rusty-engine` |

---

## Band B1 — GGUF-чанкер (pure Rust)  **`[x]` (PH-101..107 + 105b done, 2026-10-14)**

> Мета: прочитати реальний GGUF-MoE-файл без llama.cpp. **Досягнуто** — контейнер цілком
> в Rust. Фактаж: [`docs/research/gguf-format.md`](research/gguf-format.md) +
> [`docs/research/ggml-blocks.md`](research/ggml-blocks.md).
> **Контекст-дисципліна:** std-only (101/102/106/107), mmap `103` (`memmap2`), `104`
> (block-sizes усіх 5 + scalar dequant Q4_0/Q8_0) і `105` (tokenizer-metadata ingest) —
> **B1 завершено**. K-quant dequant math → B2; BPE-алгоритм → B3.
> **Real-file parity (2026-10-14):** парсер перевірено на справжньому
> [`Qwen2.5-0.5B Q4_K_M`](research/real-model-inspect.md) (291 tensors / 26 kv / arch qwen2);
> там же знайдено й виправлено `ggml_dtype` off-by-K-types (12/13/14) + регресійний тест.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-101 | GGUF v3 header + KV-metadata парсер (std-only) | `[x]` | `src/gguf.rs` — `parse()`/`Value`/`ValueKind`, 7 unit |
| PH-102 | tensor-info table (name/dims/dtype/offset) | `[x]` | `src/gguf.rs` — `TensorEntry`/`ggml_dtype`, round-trip synth |
| PH-103 | mmap/mlock staged loader (disk→RAM, progress) | `[x]` | `src/loader.rs` — `MmapModel::open`/`lock_pages`(unix)/`staged_read`(+progress)/`load_staged`; `memmap2`; 3 unit |
| PH-104 | quant-блоки: sizes усіх 5 + scalar dequant Q4_0/Q8_0 (K-quant → B2) | `[x]` | `src/quant.rs` — `block_info`/`bytes_for_numel`/`dequant`+`f16_to_f32`; 6 unit |
| PH-105 | tokenizer-metadata ingest (model/tokens/type/scores/merges/bos/eos/pre) | `[x]` | `src/tokenizer.rs` — `TokenizerMeta::from_file`; 2 unit |
| PH-105b | byte-level **BPE encode/decode** (GPT-2 style) | `[x]` | `src/tokenizer.rs` — `ByteEncoder` (invertible 0..=255) + `Bpe` (pre-tokenize/merge-by-rank); round-trip + **byte-fallback** unseen merges + **atomic specials** (token_type≥3, `<|…|>` split) via `special_tokens()`/`split_on_specials` (PH-305b); HF-exact regex parity → later |
| PH-106 | round-trip на disk (write synth GGUF → read via `parse_file`) | `[x]` | `src/gguf.rs` — `round_trips_a_real_file_via_std_io` (real `.gguf` bytes) |
| PH-107 | CLI `rusty inspect model.gguf` | `[x]` | `src/bin/rusty.rs` — `inspect()` summary + graceful errors |

---

## Band B2 — Quantized kernels + SIMD  **`[~]` (PH-201 / PH-201+ / PH-201c / PH-201d / PH-204 / PH-202 / PH-203(x86+AVX2/FMA) / PH-203e / PH-203f / PH-203k…r / PH-205 done, 2026-10-22)**

> Мета: швидкі ядра для квантів, x86 + arm64, feature-gated dispatch.
> Скалярний **reference-оракул** для Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K + fused dot/gemv
> для Q4_0/Q8_0 **і K-квантів** (`PH-204`). Лишився тільки NEON (`PH-203` arm64) —
> x86 SSE2/AVX2 dispatch (`PH-203`), AVX2+FMA hot dots (`PH-203d`) і bench (`PH-205`) закриті.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-201 | scalar reference (dot, gemv) — Q4_0/Q8_0 fused | `[x]` | `src/kernel.rs` — `dot_q4_0`/`dot_q8_0`/`matvec_q4_0`/`matvec_q8_0` (== dequant-then-dot, 4 unit) |
| PH-201+ | scalar reference dequant Q4_K/Q5_K/Q6_K (all-block oracle) | `[x]` | `src/quant.rs` — `dequant_q{4,5,6}_k`+`get_scale_min_k4`; 4 parity; C-ref кеш |
| PH-201c | scalar reference dequant **Q5_0** (unblocks real embeddings) | `[x]` | `src/quant.rs` — `dequant_q5_0`(22B, bias −16) + `DType::Q5_0`/`ggml_dtype(6)`; real token_embd 136M finite; 71 unit |
| PH-201d | **fused Q5_0 dot/gemv** (closes the quant matrix) | `[x]` | `src/kernel.rs` — `dot_q5_0`(d@0/qh@2/qs@6, −16 bias, 5th-bit `qh>>j`) + `matvec_q5_0`; == `dot(dequant_q5_0, b)` / `matvec(dequant_q5_0)`; 2 unit |
| PH-204 | **fused dequant+dot** K-quant (Q4_K/Q5_K/Q6_K, stack `[f32;256]`, no Vec) | `[x]` | `src/kernel.rs` — `dot_q{4,5,6}_k` + `matvec_q{4,5,6}_k` macro (relative-parity vs `dot(dequant_qX_k, b)`, 4 unit) |
| PH-202 | **repack** у compute-friendly layout (contiguous scales + de-interleaved nibble planes) | `[x]` | `src/repack.rs` — `RepackedQ4_0 {d,lo,hi}` + `RepackedQ8_0 {d,v}` + `repack_q{4,5}_*` + `dot_repacked_q{4,5}_*` + `matvec_repacked_q4_0`; **exact** vs `dot_q4_0` / `dot_q8_0` / `matvec(dequant, x)` (5 unit) |
| PH-203 | **SSE/AVX/NEON dispatch** (x86: SSE2 Q4_0+Q8_0, AVX2 Q4_0/Q5_0/Q8_0/Q4_K/Q6_K; NEON → далі) | `[~]` | `src/kernel.rs` — `dot_*_sse2` (Q4_0/Q8_0) + `dot_*_avx2` (Q4_0/Q5_0/Q8_0/Q4_K/Q6_K) (`core::arch`, `#[target_feature]`, `hsum_ps_sse2`) + `dot/matvec/matmul_*_dispatch` (AVX2→SSE2→scalar ladder, runtime `is_x86_feature_detected!`) + `qmat.rs` wired у forward; `#[cfg(x86_64)]` parity SSE2/AVX2==scalar; bench `PH-205` 1.8–5.3×. Лише NEON лишився |
| PH-203d | **AVX2+FMA fused dots** Q4_0/Q5_0/Q8_0 (2 accumulators, `_mm256_fmadd_ps`, Q8_0 `_mm256_cvtepi8_epi32` gather) | `[x]` | `src/kernel.rs` — `dot_q4_0/q5_0/q8_0_avx2` переписано на fused `fmadd` + подвійний акумулятор; dispatch (і parity-тести) гейтовано `avx2 && fma` (FMA є на кожному AVX2-хості з Haswell). Bench: dot q8_0 1.72→2.09×, q4_0 2.31→2.41×, q5_0 2.14→2.30×; matvec q8_0 1.77→2.06×; \|Δ\| без регресії (q8_0 1.34e-3→9.8e-4). Live `:8090` decode 2.27→2.68 tok/s |
| PH-203e | **row-parallel projections** (std `thread::scope`, `RUSTY_THREADS`, `PAR_MAC_MIN` 2¹⁸) | `[x]` | `src/qmat.rs` — `QMat::matvec`/`matmul` розбивають роботу по рядках на воркерів (`matvec_serial`/`matmul_serial` лишаються як fallback і для малих проєкцій); `matmul` ділиться по **рядках ваг** (інакше `matmul_reuse` де-квантизує всю матрицю на кожного воркера). `src/kernel.rs` — `quant_row_bytes`/`quant_row_dot`/`matmul_dispatch`. Результат біт-у-біт той самий (2 тести `*_is_bit_identical_to_serial`). A/B 321-tok: prefill 45.6→24.7 с (1.85×); decode 2.68→4.2–5.1 tok/s. Стеля боксу: Ryzen 5 5500U 6C/12T під ~60% чужим навантаженням |
| PH-203f | **persistent worker pool** замість `thread::scope` на кожну проєкцію | `[x]` | `src/pool.rs` (std-only) — від'єднані воркери + черга `Mutex`/`Condvar`, `pending`-лічильник; `parallel_for(jobs, f)` повертається лише коли pending==0, тому борг позичальника стирається через `Handle` (сирий вказівник + мономорфізована `call`-функція) безпечно; job-паніки ловляться `catch_unwind`. Виміряно: `thread::scope` на 12 потоків = **750–820 µs**, тобто ~90 ms/токен на ~120 проєкціях. `QMat::matvec`/`matmul` тепер збирають блоки через `Mutex`, а кількість job-ів = фактична кількість чанків (не `threads`), інакше воркер отримував base за межами `out`. Результат: gate/up 4864×896 Q5_0 **1.270→0.684 ms** (4.3–5.1× vs serial), prefill matmul 4864×896×961 **1009→247 ms (4.08×)**, live decode **4.4→7.2–7.9 tok/s** (накопичувально 2.27→7.8 ≈ 3.4×). Prefill 321-tok лишився ~25 с → вузьке місце перейшло на послідовний attention/KV |
| PH-203g | **RoPE раз на позицію + повернутий KV-кеш** | `[x]` | `src/infer.rs` — `rope_freqs(d, base)` (таблиця `base^(-2i/d)` один раз на модель) + `rope_in_place`; `rope()` лишився тонкою обгорткою (результат той самий). `src/forward.rs` — `LayerKv` зберігає **вже повернуті** ключі (контракт «raw → rotate on read» замінено; обидва письменники й єдиний читач оновлені разом, еквівалентність тримає `block_forward_cached_matches_seq_over_prefix`); prefill повертає цілі Q/K-рядки **раз на позицію** замість разу на (head, query, key) і переюзує один scratch для scores/probs; V-накопичення переставлено на `for j { for d }` (той самий порядок сум, але внутрішній цикл суміжний → векторизується). `src/moe.rs` — `softmax_in_place` (та сама послідовність операцій). Причина: `rope()` робив `powf` на елемент і алокацію, а prefill кликав його ~723k разів на 321 токен. Результат: prefill 321 tok 25.5→**5.89 s** (4.3×), 961 tok 190.9→**29.1 s** (6.6×); decode 7.4→**10.35 tok/s**. Оракул (55/12/48) і `PH-405` fingerprint не зрушили |
| PH-203h | **head-parallel prefill attention** | `[x]` | `src/forward.rs` — `attention_head_ctx` (один per-head кернел для обох prefill-шляхів: KV-кеш і prompt-local K/V мають той самий layout) + `attention_prefill`, що збирає голови в `seq × q_dim`; `pool::parallel_for` від `PAR_SEQ_MIN = 32` (нижче hand-off дорожчий за роботу). Голови пишуть у власні `head_dim`-блоки, порядок сум на елемент не змінено. Прибрано два дубльовані inline-цикли. Prefill 961 tok 29.1→**23.5 s** (1.24×; 190.9 s на початку сесії = 8.1×), 321 tok 5.89→5.56 s; decode без змін. Оракул + `PH-405` fingerprint green |
| PH-203i | **f16 widening + q8_0 integer-dot probe** | `[x]` | `src/quant.rs` — `f16_to_f32` переписано з `f64`+`powi` (на кожен блок!) на точне бітове розширення `f32::from_bits`; enumeration-тест на всі 65 536 `u16` (subnormal/Inf/NaN) + `--ignored` golden/oracle без recapture. `src/kernel.rs` — `Q8Activation` (f32 scale/32 + i8), scalar `dot_q8_0_i8`, SSE2/AVX2 (`cvtepi8_epi16`+`madd_epi16`, `dot_q8_0_i8_{sse2,avx2,dispatch}`), `matvec_q8_0_i8`/`matmul_q8_0_i8`, біт-у-біт з оракулом. Бенч: dot_q8_0 5.34→**0.64 µs (9.55×)**, matvec 896² **11.8 GMAC/s** (було 1.5); live decode **27.6 tok/s** (було 10.9), prefill 961 tok 23.46 s (без змін). Int8-шлях після фіксу повільніший за float (0.89×) → у `QMat` не підключений |
| PH-203k | **prefill: активаційний tiling + row-split проєкцій** | `[x]` | `src/kernel.rs` — `matmul_reuse` тепер обходить `X` тайлами по `MATMUL_ROW_TILE=16` рядків (ваговий рядок сканує 57 KiB замість 3.3 MiB; порядок додавання блоків на `(row, r)` не змінено → біт-у-біт), `matmul_q5_0_tiled` як бенч-хвіст, тест `matmul_activation_tiling_is_bit_identical` (tile 1/3/31/32/64/128 == один тайл). `src/qmat.rs` — `QMat::matmul` ділиться по **вхідних рядках** (кожен воркер тримає свої ~290 KiB `X` в L2, ваги перечитуються `threads` разів ≈36 MiB) замість колонок (усі 12 воркерів годували спільний 16 GiB `X`-потік у L3); `matmul_split_cols` лишено `#[doc(hidden)]` для A/B. Бенч: serial 4864×896×961 1469→**897–1103 ms** (tile 16 vs untiled), prefill-decomposition A/B у одному процесі: cols **248→190 ms** (1.30×), повтор 344→291 (1.18×); live prefill 961 tok 26.9–27.3 s на завантаженому боксі (rebook+llamas+gsv) — end-to-end не порівнюється. 194 lib + integration green, clippy 0 |
| PH-203l | **prefill: full-row f32 акумулятор (один hsum на рядок)** | `[x]` | `src/kernel.rs` — `matmul_reuse` дейквантизує ваговий рядок **цілком** (`dequant_row` → буфер `n` f32) і робить один `dot_f32_dispatch` проти рядка `X`, тобто один hsum на `(row, r)` замість одного на 32/256-блок; `matmul_reuse_blocks` + `matmul_q5_0_blockwise` лишено `#[doc(hidden)]` як числовий оракул і бенч-A/B. Порядок сум інший (8-смуговий акumulator по всьому рядку vs hsum-на-блок + f32), тому це числовий зріз: новий тест `matmul_row_wise_accumulator_matches_block_wise` (≤1e-4 відносної, факт ~2e-5), і `cargo test --release -- --ignored` пройшов **без перезахоплення golden** — greedy-fingerprint (re-prefill + incremental/serve) не зрушив. Бенч (min-of-7, один процес): single-thread Q5_0 4864×896×961 block **805.9 → row 606.3 ms (1.33×)**; serial 1469 (PH-203j) → **859 ms**; QMat parallel **92.5 ms = 45 GMAC/s** (9.29× над serial). Tile sweep став плоским (961→606.7, 16→610.8 ms) → тайлінг тепер зайвий. Live prefill 961 tok **18.93–19.04 s** (було 26.9–27.3 у PH-203k). 195 lib + integration green, clippy 0 |
| PH-203m | **per-stage профайлер forward-шляху (`RUSTY_PROFILE=1`)** | `[x]` | `src/profile.rs` (std-only) — `Stage::{Matvec,Matmul,Attention}` з атомарними лічильниками ns+calls, `timed()` = один relaxed-load бранч коли вимкнено; `report()` друкує таблицю проти виміряного wall-clock виклику плюс **розбивку matmul за формою** (`note_matmul(ns,m,n,rows)` → GMAC/s на форму). Хуки: `QMat::matvec/matmul`, `forward::attention_prefill`, звіт у `DenseDecoder::prefill` (serve-шлях, → `rusty_prof.err.log`) і один раз у `DenseRunner::prefill_logits` (re-prefill). Live prefill 961 tok (LAN, 12 потоків): wall 20.2 s = **matmul 84.4 %** (168 викликів), attention 9.4 % (1.9 s), matvec 0.2 %, other ~6 %. Головна знахідка: down-проєкція `896×4864` дає **10.0 GMAC/s**, тоді як `4864×896` 34.8, `896×896` 36.0, `128×896` 29.8 — тобто широкий `n` упирається в L2 (rowbuf 19.5 KiB + 16 рядків `X` = 312 KiB не влазить в L1). 197 lib + integration green, clippy 0 |
| PH-203n | **prefill: 4-рядковий register blocking + двовимірне cache-блокування + вісь поділу за `X`** | `[x]` | `src/kernel.rs` — (1) `dot_f32_avx2_x4`/`dot_f32_x4_dispatch`: одне завантаження ваги на 4 FMA (4 FMA / 5 loads замість 1/2), `MATMUL_ROW_GROUP = 4`, хвіст рядків → одиночний dot; (2) `matmul_reuse` тепер `for r_block { for tile { for r { for row group } } }` — тайл **зовні** циклу ваг (`MATMUL_ROW_TILE = 96`, `MATMUL_R_BLOCK = 1024`), бо старий порядок перечитував весь `X` з DRAM на кожен ваговий рядок (16.8 GiB на down-проєкцію), а тайл тепер амортизує дейквант (`MAC/tile`): 16 → 96 це −30 %, 192+ падає; (3) `QMat::matmul` обирає вісь за `MATMUL_X_L3_BUDGET = 6 MiB` — рядковий поділ, поки `X` влазить у L3, інакше `matmul_out_rows` (колишній `matmul_split_cols`). Усе **біт-у-біт** (інший лише порядок незалежних ітерацій): тести `matmul_row_grouping_is_bit_identical` (70 = 4×17+2) і `matmul_weight_row_blocking_is_bit_identical`, хуки `matmul_q5_0_{rows1,rb1,blocked}` для A/B. Один потік Q5_0 961 рядків: gate/up 606 → **199–201 ms (21.0 GMAC/s, 3.0×)**, down 688–914 → **218–222 ms (18.9)**. Пул 12 потоків, A/B осей: gate/up рядки **56.2 ms = 74.5 GMAC/s** проти 67.1 (вихідні рядки); down рядки 256.7 ms = 16.3 проти **72.5 ms = 57.7 (3.54×)**. Live prefill 961 tok **15.2 → 10.44 s** (matmul 69.2 %, attention 19.3 %, `896×4864` 10.0 → **36.4 GMAC/s**); `cargo test --release -- --ignored` зелений без перезахоплення golden. 199 lib + integration green, clippy 0 |
| PH-203o | **prefill attention: x4 score dots + векторний axpy + round-robin рядки** | `[x]` | `src/forward.rs` + `src/kernel.rs` — оцінки уваги йшли скалярним `zip().map().sum()` (≈9.9 G MAC при 1 MAC/цикл), а зважена сума значень — поелементним циклом, тому attention у профілі PH-203n коштував **2.02 s (19.3 %, 24 × 84 ms)**. Тепер: (1) оцінки через `dot_f32_x4_dispatch(qh, [k_j..k_j+3])` — одне завантаження запиту на 4 ключі, хвіст через `dot_f32_dispatch` (**числовий зріз**: послідовна f32-сума → 8-смуговий акумулятор, greedy-fingerprint не зрушив); (2) новий `kernel::axpy_f32_dispatch` (`y += a·x`, AVX2 `mul`+`add` без FMA-контракції) — **біт-у-біт** як скалярний цикл, тест `axpy_dispatch_is_bit_exact_with_scalar` (1/7/8/9/64/67); (3) `attention_prefill_inner` ділить роботу на `(head, round)` з `rounds = clamp(4·threads/heads, 1, seq) = 4` → 56 завдань (було 14 голів = дві хвилі на 12 потоках), рядки роздані round-robin, бо рядок `i` коштує O(i); тест `prefill_attention_row_rounds_are_bit_identical` (rounds 3/7 == один потік). Лайв 961 tok, 4 прогони: attention **812–880 ms** (було 2018, 2.4×), prefill **9.12–9.90 s** (було 10.44). 201 lib + integration green, clippy 0, `--ignored` зелений без перезахоплення golden |
| PH-203p | **prefill: інструментувати й різати «other» (~11 %)** | `[x]` | Крок 1 — `src/profile.rs`: `Stage::{Norm,Rope,KvCopy,Swiglu,Residual,Embed}` (плюс `serial`-підсумок і `unaccounted = wall − serial` у звіті, `[pool]`-маркер), хуки в `forward.rs` (`norm_row`/`norm_rows`, `rope_rows`, `kv_append`, `swiglu_batch`, `residual_add`) і `runner.rs` (`embed_batch`, фінальна норма). Лайв 961 tok, 12 потоків: serial **1010 ms (11.9 %)** = **swiglu 676 ms (8.0 %)** + rope 205 ms (2.4 %) + residual 69 ms + norm 53 ms + kv-copy 6 ms + embed 1 ms. Крок 2 — обидва великі шматки, **біт-у-біт**: (a) `swiglu_batch` ділить `seq×d_ff` на 4·threads діапазонів і виконує на пулі (той самий `silu(g)*u` поелементно, посилання на `out` через `as_mut_ptr`, `SWIGLU_PAR_MIN` 16 K) → **676 → 116–123 ms (5.7×)**; (b) `infer::rope_table`/`rope_in_place_tabled` будують `(cos,sin)` раз на `(позиція, пара)` замість `heads×` на голову — той самий `angle = pos·freq`, ті самі `cos`/`sin`, тест `rope_table_is_bit_identical_to_per_position_rope` → **205 → 33–41 ms (6×)**. Після: serial **260–297 ms (3.4–3.8 %)**, live prefill **8.49 → 6.90–7.84 s** (matmul 84 %, attention 9.8 %), `--ignored` зелений без перезахоплення golden. 203 lib + integration green, clippy 0 |
| PH-203q | **K-квантовий тайл: гіпотеза «256» відкинута, тайл = L3/воркери** | `[x]` | `src/kernel.rs` — три K-квантові диспетчери зведено в `matmul_k` (+ `k_quant`, `type KBlockDequant`), хук `matmul_k_tiled`, тест `matmul_k_quant_tiling_is_bit_identical`. Новий `benches/kquant.rs` міряє тайл і на 1 потоці, і через **прод-поділ 12 потоків**: тайл 256 на `down 896×4864` Q6_K ПРОГРАВ 1.7× (439.9 проти 259 ms), бо дейквант — одиниці відсотків, а L2/L3 — ні. У пулі кожен воркер тримає свій тайл, тому оптимум 24 (66.1 ms = 63.4 GMAC/s проти 109.4 = 38.3 при 96), а для вузьких `n=896` — 96 (64 GMAC/s). Тайл став функцією `k_row_tile_with(n, threads) = clamp(6 MiB/(threads·n·4), 8, 96)` (кратне 8), тест `k_row_tile_shrinks_with_row_width_and_threads`. Register blocking x8 виміряно й **відкинуто** (один потік +10 %, пул −40 %: 8 потоків X на воркера) — лишився хук `matmul_q5_0_grouped`; заодно виправлено хвіст `n%8` у `dot_f32_avx2_x4` (додавався лише в `out[0]`). Live: down 2.40–2.73 → **1.79–2.02 s** (36.8–41.9 → **49.7–56.0 GMAC/s**), gate/up 58–65; golden не зрушив. 205 lib + integration green, clippy 0 |
| PH-203r | **два рядки ваг на прохід (2 `rowbuf`) — prefill 7.1–8.9 → 5.3–6.4 s** | `[x]` | `src/kernel.rs` — `dot_f32_avx2_x4x2` + `dot_f32_x4x2_dispatch`: чотири активації проти **двох** дейквантованих вагових рядків, тобто `2 ваги + 4 активації = 6 loads / 8 FMA` (0.75/load) замість `5 / 4` (1.25); кожен вихід тримає власний 8-ланьовий акумулятор, тому **біт-у-біт** (`matmul_two_weight_rows_is_bit_identical`, m = 5 непарний, rows = 70). `matmul_reuse` розкладено на `dot_tile_rows` + `dot_tile_rows_pair`, новий параметр `w_rows` (`MATMUL_W_ROWS = 2`, пара лише при `group == 4`), хуки `matmul_q5_0_wrows` / `matmul_k_tiled_rows`. Вимір (min of 3, 12 потоків, прод-поділ): gate/up Q5_0 54.9 → **41.2 ms (76.3 → 101.7 GMAC/s)**, down 54.2 → **38.6 (77.2 → 108.5)**; один потік 19.1 → 30.3 і 17.7 → 24.0. **Гіпотеза PH-203q про per-worker тайл відкинута**: тайл — спільний зріз `X`, у воркера лише `rowbuf`-и, тому pooled-оптимум down Q6_K знову 96 (24 → 75, 48 → 99, 96 → 108, 128 → 119, 192 → 80); 128 у live дав 55.4 проти 64–69 при 96, тож `MATMUL_ROW_TILE_K = MATMUL_ROW_TILE`, `k_row_tile_with`/`k_row_tile` видалено. Live: wall **5.34/5.60/5.81/6.03/6.11/6.41 s**, matmul **4.1–4.7 s (77 %)**, down **1.46–1.56 s = 64–69 GMAC/s**, gate/up 81–92, attention 0.74–0.78 s (13.9 %), other 2.4–3 %; golden не зрушив. 205 lib + integration green, clippy 0 |
| PH-203s | **attention 13.9 % — онлайн-softmax блоками по KV** | `[ ]` | Після PH-203r matmul на межі FMA/load (`2 FMA/цикл`), а attention — 0.74–0.78 s (13.9 %, 30–32 ms на шар). `attention_prefill_inner` тримає повний рядок оцінок і два проходи (max, потім нормований exp + axpy); перехід на одно-прохідний онлайн-softmax блоками по 64 ключі (як у flash-attn, але CPU: max/sum у регістрах) скорочує прохід по `scores` із 3 до 1. **Увага:** порядок `exp`/нормалізації змінить f32-округлення → потрібен новий golden, тому тільки з дозволу owner; спершу виміряти частку `expf` у 30 ms (таблиця `exp` на 2^k кроків — біт-у-біт варіант) |
| PH-205 | bench vs scalar (std-only, no criterion) | `[x]` | `benches/quant.rs` (`harness=false`) — scalar vs `*_dispatch`, ns + speedup + \|Δ\| (SIMD==scalar до f32-округлення, \|Δ\|≤4e-3): dot q4_0 2.41×, q8_0 2.09×, q5_0 2.30×, q4_k 3.98×, q6_k 5.32×; matvec q4_0 2.38×, q8_0 2.06× (Qwen 896², best-of-5); плюс `scope spawn+join` overhead, QMat matvec/matmul vs serial |

---

## Band B3 — Dense forward (control path)  **`[x]` (PH-301..306 done, 2026-10-14)**

> Мета: згенерувати текст на DENSE-моделі як контроль перед MoE. **Досягнуто** —
> `DenseRunner` веде повний multi-layer forward (embed→24×block_forward_seq→rmsnorm→
> output→argmax) на **справжній** Qwen2.5-0.5B GGUF; CLI `rusty run` генерує greedy
> за числовими id. Числова parity з llama.cpp/HF (`PH-405`) + HF-exact BPE
> (`PH-305b`: `pre`-regex/byte-fallback/specials) → окремі смуги.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-301 | RMSNorm (+ SiLU/MatMul — уже в `moe.rs`) | `[x]` | `src/infer.rs` — `rmsnorm`; 50 tests |
| PH-302 | causal attention GQA/MQA + RoPE (primitives) | `[x]` | `src/infer.rs` — `rope`/`apply_causal_mask`/`attention_causal`/`gqa_repeat_kv` (multi-head loop → PH-305) |
| PH-303 | paged KV-cache (page alloc, vLLM-ідея) | `[x]` | `src/infer.rs` — `PagedKvCache::append/get` round-trip across pages |
| PH-304 | sampler chain (temp/top-k/top-p/repetition/argmax) | `[x]` | `src/infer.rs` — `sample`/`mask_top_k`/`mask_top_p`/`apply_repetition_penalty`/`argmax` (grammar →later) |
| PH-304b | **sampler wiring into the runner** | `[x]` | `GenerationOptions` += `repetition_penalty`; `infer::SeedRng` (xorshift64*, `next_f32`∈[0,1), детерм.); `DenseRunner::generate_with(prompt,opts,max_new,&mut rng)` — penalty→sample, `temp<=0`==greedy (`generate_with_greedy_matches_generate`); 1 infer unit + 2 integration |
| PH-305a | real-tensor **mmap→byte-range→dequant** (single weight) | `[x]` | `src/loader.rs` — `tensor_bytes`/`dequant` + `gguf` `TensorEntry::nbytes`; CLI `rusty tensor`; real Q6_K 4.36M finite; synthetic layout test |
| PH-305c | GGUF **model config reader** (forward-loop geometry) | `[x]` | `src/model.rs` — `ModelConfig::from_file`/`head_dim`/`is_moe`; CLI `rusty config`; real qwen2 24/896/14/2/4864; synth+missing-arch tests |
| PH-305d | **transformer decode-step forward** (attn+FFN+residual) | `[x]` | `src/forward.rs` — `block_forward`/`attention_decode`/`logits_from_hidden`/`Geometry`/`BlockWeights`; toy-oracle: residual-identity + known greedy token + GQA; llama-parity → PH-405 |
| PH-305e | **tensor-geometry validation** (name+dims vs config) | `[x]` | `src/model.rs` — `ModelConfig::validate` (token_embd/output/output_norm + per-block attn/ffn req, qk_norm/bias opt); CLI `rusty validate`; real Qwen2.5 24-layer OK; synth match+mismatch tests |
| PH-305 | **end-to-end dense forward** (multi-layer loop, real GGUF), greedy | `[x]` | `src/runner.rs` `DenseRunner::from_model/prefill_logits/greedy/generate` + `forward::block_forward_seq`/`last_logits` (causal GQA+RoPE, per-row FFN, residual); quant `F32`/`F16` dequant branches (norms/small tensors); `tests/dense_gen.rs` synth 2-layer greedy=token4 + `#[ignore]` real Qwen2.5 24-layer prefill 151 936 logits finite |
| PH-306 | CLI `rusty run -m <model> -t "ids" [--max-tokens N]` | `[x]` | `src/bin/rusty.rs` — `run()` numeric-token greedy (BPE HF-parity → `PH-305b`) |
| PH-306b | **text pipeline** BPE encode → generate → decode | `[x]` | `src/textgen.rs` — `TextRunner::from_model/run_text` ties Bpe + DenseRunner; `rusty run -p "text"` prints text; CLI `-p` uses `TextRunner` (byte-level or model's real vocab), `-t` numeric fallback. `tests/text_gen.rs` synth round-trip "hi"→`~~~` (2 int) + tokenize+decode "a<|x|>b". |
| PH-307 | **incremental decode + KV** (O(1)/крок) | `[x]` | `forward.rs` — `LayerKv` + `block_forward_cached` (append K/V, causal over cache, `#[allow(needless_range_loop)]`); `runner.rs` — `DenseDecoder::step/generate_incremental`; `full == incremental` (1 forward unit + 1 integration `incremental_decode_matches_full_prefill`). Paged `infer::PagedKvCache` → B4 |
| PH-305b | BPE **byte-fallback + specials + HF pre-regex** (readable text) | `[x]` | `src/tokenizer.rs` — unseen merges split to byte tokens; `token_type≥3` specials atomic via `split_on_specials`; **`src/gpt2.rs`** std-only GPT-2/Qwen2 pre-tokenizer (`gpt2_pre_split`: contractions, `[^…]L+`, `N{1,3}`, symbol+`\r\n`, `\s*[\r\n]+`, `\s+`) wired into `Bpe::encode`; 6 gpt2 + 3 tokenizer unit. Byte-exact HF `tokenizer.json` parity → `PH-405` gate |

---

## Band B4 — MoE-ядро (ДИФЕРЕНЦІЯТОР)  **`[~]` (PH-401/401b/402/403/403b/404/404b/406a done, 2026-10-14)**

> Мета: **MiniCPM-V 4.7-35B-A3B** (перший on-device MoE у серії) GGUF **нативно в Rust**,
> минаючи llama.cpp. (Ретаргет 2026-10-24: Qwen3-30B-A3B / Mixtral із цілей прибрано.)
> **Повний compute-цикл MoE-шара вже зʼєднаний і детермінований** (`moe_layer_forward`);
> лишилось підвʼязати реальні деquant-тензори (PH-406) + оптимізації (PH-404/407) + parity (PH-405).
> Еталонна підлога (інструмент, не ціль): llama-rs Qwen3-30B-A3B ≈ 0.497 tok/s decode — ціль ≥ parity.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-401 | top-k router: softmax-renorm + sigmoid + noaux_tc group | `[x]` | `src/moe.rs` — `route`/`top_k_sigmoid`/`route_group` (simplified; exact top-2/grp → PH-405) |
| PH-401b | **exact `noaux_tc` group routing** (DeepSeek-V3/Qwen3) | `[x]` | `src/moe.rs` — `route_noaux_tc`: group score = **sum of top-2** (not all-sum), select `top_k_group`, global top-k renorm, stable ties; distinguishes from `route_group` (3 unit) |
| PH-402 | expert FFN SwiGLU (`ffn_gate/up/down_exps`) | `[x]` | `src/moe.rs` — `expert_ffn`/`swiglu`/`silu`/`matvec`; 44 unit |
| PH-403 | shared-expert merge (DeepSeek/Qwen3) | `[x]` | `src/moe.rs` — `combine` (shared=Some/None, shape-validated) |
| PH-403b | **MoE transformer block forward** (attn+KV → cached MoE FFN) | `[x]` | `forward.rs` `attention_forward_cached` (head-only, shared core) + `moe.rs` `moe_block_forward` — attn+residual→rmsnorm(ffn_norm)→`moe_layer_forward_cached`→residual; 1-expert top-1 == `block_forward_cached` (1e-5, kv equal) |
| PH-406a | MoE FFN **layer forward** (router→experts→shared) | `[x]` | `src/moe.rs` — `moe_layer_forward`/`ExpertWeights`; 2 integration (63 total) |
| PH-404 | **expert-cache LRU** + byte-budget + pin + prefetch | `[x]` | `src/cache.rs` — `ExpertCache::get_or_load/put/prefetch/set_pinned` + `ExpertKey`/`CacheStats`, budgeted LRU eviction (pinned skipped), oversized→uncached; 7 unit |
| PH-404b | **MoE forward через cache** (top-k experts on demand) | `[x]` | `src/moe.rs` — `moe_layer_forward_cached(...&mut ExpertCache, load)`: route→`get_or_load` per expert; == `moe_layer_forward` (deterministic loader, 1e-6) навіть при budget<resident; hit⇒loader not called (2 unit) |
| PH-405 | **numeric regression gate** (real Qwen2.5 greedy fingerprint) | `[x]` | `tests/numeric_parity.rs` + `tests/data/qwen25_golden.txt` — детерм. greedy-відбиток 24-шарового `DenseRunner.generate` на справжній моделі (guard quant/forward/KV/sampler); `#[ignore]` release-job (~60s), CI без моделі skip. llama-rs semantic parity → окремий зріз |
| PH-405b | **llama-rs semantic greedy-parity + HF `tokenizer.json` byte-exact** | `[x]` | `tests/moe_parity.rs` — Гейт A (vocab 151936/merges 151387/id-точки/`qwen2` pre-regex) PASS; Гейт B: наш greedy байт-у-байт із записаним оракулом (55/12/48). **Фікс:** додано q/k/v-біаси Qwen2 (`BlockWeights.{q,k,v}_bias`) + RoPE GPT-J→NeoX rotate-half. Факти — `docs/research/llama-rs-parity.md`; golden перезахоплено |
| PH-406 | run **MiniCPM-V 4.7-35B-A3B** GGUF pure-Rust (text/MoE path; ваги через B12 spike PH-1201; Qwen3-30B/Mixtral — не цілі) | `[ ]` | `src/moe/forward.rs` |
| PH-407 | bench tok/s vs llama-rs baseline | `[ ]` | `benches/moe.rs` |

---

## Band B5 — Single-node server  **`[~]` (PH-501a/b, 502, 503, 504, 505, 506, 507 done, 2026-10-17)**

> Мета: `rusty serve` — OpenAI-compat, стрімінг, keep-live в GSV. **Перший зріз —
> суто `std` (без axum/tokio):** чистий `http::route(req)->Response` + `TcpListener`-насос
> `serve::serve_one`, loopback-тест по дроту. Повний async + стрімінг + real-model wiring — далі.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-501a | **std-only HTTP layer** (pure `route()` + TcpListener pump) | `[x]` | `src/http.rs` — `Response::to_http_bytes`(Content-Length exact)/`parse_request_line`/`route`: GET /health, GET /v1/models, POST /v1/chat/completions (OpenAI stub), 404/405/400; `src/serve.rs` — `serve_one/handle_conn/serve` (loopback, fail-open); CLI `rusty serve [--port N]`; 7 http unit + 1 wire test |
| PH-501b | **real TextRunner у HTTP handler** (end-to-end) | `[x]` | `textgen.rs`: `extract_message_content` (std JSON reader, escape) + `TextRunner::complete` + `chat_response`; `serve_with_model`/`handle_conn(_,_,Some(&tr))` POST→BPE→generate→decode; CLI `rusty serve -m <gguf>`. Test `serve_with_model_completes_over_the_wire` (synth GGUF, loopback, `"content":"~~~"` + usage). |
| PH-501 | HTTP server (tokio+axum) — `/v1/models`,`/v1/chat/completions` | `[ ]` | `src/serve/http.rs` |
| PH-501 | HTTP server (tokio+axum) — `/v1/models`,`/v1/chat/completions` | `[ ]` | `src/serve/http.rs` |
| PH-502 | **SSE-стрімінг токенів, stop-tokens, usage** | `[~]` | `src/http.rs` — `sse_body` (OpenAI `chat.completion.chunk` + `finish_reason:stop` + `data:[DONE]`, event-framed) + `body_of`/`wants_stream`; `route` → `text/event-stream`; serve.rs wire-тест `[DONE]`. Інкрементні socket-writes + usage → далі |
| PH-503 | prompt-cache / multi-slot | `[x]` | `src/slot.rs` — `Slot`/`SlotTable` (capacity LRU, `open`/`get`/`append`/`close`/`prompt_cached` + hits/misses/evictions); KV per-slot → `DenseDecoder`. 6 unit (evict/open-preserves/append/prefix-cache/close) |
| PH-504 | **GSV keep-live `target/live/rusty_heartbeat.json`** | `[x]` | `src/hub.rs` — `Heartbeat` (`to_json` без serde) + `is_fresh` (saturating, clock-skew-safe) + `write_heartbeat`; serve stamp при startup+кожне зʼєднання (15s/60s cadence з `hub.rs`); CLI `rusty heartbeat`. 3 hub unit |
| PH-505 | метрики tok/s·RAM·active-experts endpoint | `[x]` | `src/hub.rs` — `Metrics` (`zero`/`record`/`tok_per_sec` zero-guard/`to_json`) + `ram_bytes()` (std-only: Windows `K32GetProcessMemoryInfo` working set / Linux `/proc/self/statm`, 0-fallback) + `GET /metrics` (`http::route_with`). Serve пише реальні токени+час+експертів: `ChatOutcome { response, generated_tokens, decode_secs, active_experts }`, `serve` → `metrics.record(..)`. `/metrics` віддає живий `tok_per_sec` + `ram_bytes`/`ram_mb` (тести `chat_response_reports_decode_stats`, `ram_bytes_reads_the_process_where_supported`, `metrics_json_shape`; live `:8090` — `ram_mb` 1432.4). `active_experts` = 0 на dense-шляху (наповниться з `PH-406` MoE) |
| PH-506 | integration test + smoke curl | `[x]` | `tests/serve.rs` — 5 над-дротові тести (health, /v1/models, stub chat, stub SSE, 400/404/405) зі **строгим** парсом тіла (`worker::parse_json`) + перевіркою `Content-Length`; wire-тести `tests/text_gen.rs` теж посилено (строгий парс + `Content-Length`, регресійний лок на PH-507). Curl-smoke скриптовано (`target/live/`) |
| PH-507 | **GSV hub provider-row (зразок `bunke-rock`):** `rusty serve` як `kind:local` рядок у hub-каталозі — `host_ready`-гейт на `target/live/rusty_heartbeat.json` + gguf-файл, авто-перемикання local↔cloud у `GET /api/omni/route` (offline-first, cooldown), SSE через hub, WiFi multi-host (LAN-first band 233). Чат-тести: provider `rusty/*` у OpenCode/Cursor тими ж рутинами, що `gsv/lama-*` | `[~]` | **E2E через живий хаб `:9999` → `rusty-serve` ok** (JSON + SSE passthrough `x-gsv-mcp: sse`). Корінь `null`-відповіді — незакритий корінь у JSON `textgen::chat_response` (фікс `chat_completion_body` + `worker::parse_json`-тест); плюс читання тіла за `Content-Length` (`serve.rs`/`http.rs`). Wire-тести посилено (строгий парс). Далі: WiFi multi-host (band 233) + IDE-chat `rusty/*` | GSV `src/boxes/omni/catalog.rs` (+ heartbeat-read) · engine `src/serve.rs`/`hub.rs` (serve-addr + heartbeat already done) |

---

## Band B6 — Build matrix: exe / bin / apk  **`[~]` (PH-605 `--build` + PH-606 done, 2026-10-14/16)**

> Мета: один портативний бінарник на OS×arch.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-601 | Windows x86_64 `rusty.exe` (gnu primary + msvc) | `[ ]` | `xtask/build_matrix.rs` |
| PH-602 | Linux x86_64 `rusty` | `[ ]` | matrix |
| PH-603 | Linux aarch64 `rusty` (Pi-class) | `[ ]` | matrix |
| PH-604 | Android arm64 `rusty.apk`/`.so` (cargo-ndk/Termux, **no Docker**) | `[ ]` | `.cargo/config` + ndk |
| PH-605 | **`cargo xtask build-matrix [--build]`** + SHA-256 артефактів | `[x]` | `xtask/src/main.rs` — `TARGETS` matrix (win-gnu/msvc, linux-x86_64/aarch64, android-arm64 no-Docker) + `sha256_hex` (pure `std`, FIPS 180-4, multi-block) + `build-matrix` рендер з хешем наявних бінарників; **`--build`** гейтить `cargo build --release --target` по `rustup target list --installed`/host (`rustc -vV`), пропускає відсутні/невдалі cross-ланки з причиною (Android NDK / MSVC / cross-linker), не валить матрицю; 3 unit (NIST vectors + coverage + `capture`). Фікс `main` argv `skip(2)`→`skip(1)` |
| PH-606 | **feature-gated backends** (sse/avx/neon single-binary) | `[x]` | `Cargo.toml [features] sse/avx/neon` (default none) + `src/backend.rs` `Backend`/`from_flags`/`compile_backends`/`select` (найсильніший ввімкнений, Scalar-fallback); `cfg(test)`-обмежені під кожен feature-набір. Dispatch-substrate для `PH-203` |

---

## Band B7 — Rust-docker на Win/Linux хостах  **`[~]` (PH-706 worker/APK runtime done, 2026-10-16)**

> Мета: slave у Rust-docker-контейнері на localhost, exposed через локальний Wi-Fi/LAN.
> **Android — поза Docker** (B6/B8). Dry-run за замовчуванням, `--live` за згодою owner.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-701 | Rust docker client (bollard): build+run slave-образу | `[ ]` | `src/docker/client.rs` |
| PH-702 | образ на таргет + `.dockerignore` | `[ ]` | `docker/` |
| PH-703 | LAN/Wi-Fi expose (bind LAN addr, GSV band-233 LAN-first) | `[ ]` | serve bind |
| PH-704 | dry-run default + `--live` owner-consent gate | `[ ]` | network module |
| PH-705 | container lifecycle contract (create/start/stop/inspect) | `[ ]` | `tests/docker_contract.rs` |
| PH-706 | **worker/APK runtime** (`rusty worker`, hub-edge role 2026-10-16): register→heartbeat→pool/join→poll→complete/requeue поверх GSV hub `/api/edge` (фронт poolAI `/api/v1`, **не** `:8091` off-box), origin `rusty_edge` role `virtual_node`, dry-run default + `--live`; std-only JSON+HTTP, token masked | `[x]` | `src/worker.rs` — `WorkerConfig` (LAN-first defaults) · `parse_json`/`parse_task` · `execute` (`ping`/`rusty_chat`→`serve_url` `/v1/chat/completions`) · request builders (`register_body`/`heartbeat_body`/`join_body`/`complete_body`/`requeue_body`) · `http_call`/`join_once`/`heartbeat_once`/`poll_once` · `plan_lines`+`mask` (token ніколи не логується) · CLI `rusty worker [--live]`; 17 unit |

---

## Band B8 — GSV-native Rust-супервізор («Rust k8s-lite»)  **`[ ]`**

> Мета: hub `:9999` = master, `rusty` = slaves; registry/heartbeat/lease/placement,
> fail-open. Reuse: llama-rs heartbeat + poolAI ed25519 job-lease + `gsv_grid`.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-801 | master registry + heartbeat (15s tick, ed25519-підпис) | `[ ]` | `src/hub/registry.rs` |
| PH-802 | slave-виявлення по Wi-Fi/LAN + lease/reclaim | `[ ]` | `src/hub/lease.rs` |
| PH-803 | placement моделі на вибраний slave + status-ring (gsv_grid-форма) | `[ ]` | `src/hub/placement.rs` |
| PH-804 | health/watchdog + fail-open (вузол впав → master тримає) | `[ ]` | `src/hub/watchdog.rs` |
| PH-805 | CLI `rusty hub` / `rusty node` → supervisor API | `[ ]` | `src/bin/rusty.rs` |
| PH-806 | max GSV-взаємодія: MCP/`gsv://` для флоту, ticket-per-worker | `[ ]` | GSV integration |
| PH-807 | тест: fake-fleet (2 slaves) → schedule → done | `[ ]` | `tests/fleet.rs` |

---

## Band B9 — Розподілений MoE (expert-parallel)  **`[ ]`**

> Мета: expert-паралелізм по воркерах (MixtureServe / MoE-Exchange) через supervisor B8.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-901 | expert-parallel sharding across slaves | `[ ]` | `src/hub/shard.rs` |
| PH-902 | token-routing до expert-вузла (latency-aware) | `[ ]` | `src/hub/route.rs` |
| PH-903 | hot/cold expert migration + prefetch | `[ ]` | `src/moe/migrate.rs` |
| PH-904 | MTP / speculative decoding cross-node (llama-rs mtp-ідея) | `[ ]` | `src/infer/mtp.rs` |
| PH-905 | cross-node KV sync draft-verify | `[ ]` | `src/hub/kv_sync.rs` |
| PH-906 | bench distributed-throughput vs single-node | `[ ]` | `benches/dist_moe.rs` |

---

## Band B10 — Web UI (design system + Rust-рендер)  **`[ ]`**

> Мета: керування/тюнінг MoE — сторінки рендерить **Rust**; JS лише chrome. Spec: [`UI.md`](./UI.md).

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-1001 | `src/ui.rs`: `/page/{name}` + `GET /api/ui/card/{name}` | `[ ]` | `src/ui.rs` |
| PH-1002 | design-system токени + тема auto/light/dark (persist) | `[ ]` | `ui/css/style.css` |
| PH-1003 | сайдбар + хоткеї `d m i k f s o l` + `?`-панель (chrome-only, XSS-safe) | `[ ]` | `ui/js/app.js` |
| PH-1004 | сторінки: dashboard(fleet)/models/playground/tuning/nodes/settings | `[ ]` | `src/ui.rs` |
| PH-1005 | ui-glue-контракт: LOC-ratchet ≥0.95 + JS-chrome-only інваріант | `[ ]` | `tests/ui_glue_contracts.rs` |
| PH-1006 | без auto-generate зайвого UI; дії = GET/POST-форми | `[ ]` | review |

---

## Band B11 — Distribs · версії · портативність · publish  **`[ ]`**

> Мета: релізний канал + відкриття MIT-репо (згода owner).

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-1101 | semver + `CHANGELOG.md` | `[ ]` | `CHANGELOG.md` |
| PH-1102 | portable distribs: tar.gz/zip/apk per target + checksums (`xtask dist`) | `[ ]` | `xtask/dist.rs` |
| PH-1103 | update-channel + binary-vs-source mtime (gsv_update-модель) | `[ ]` | xtask |
| PH-1104 | MIT-compliance: third-party attribution scan → `THIRD-PARTY-NOTICES.md` | `[ ]` | `xtask/notice.rs` |
| PH-1105 | GitHub publish (private→MIT) `github.com/platinoff/rusty-engine` (згода owner, push тільки тут) | `[ ]` | remote `origin` |
| PH-1106 | ratio/LOC-gate + docs final sync (`xtask record-speed`/`record-rust`/`sync`) | `[ ]` | `docs/vision/*.json` |

---

## Band B12 — Vision: MiniCPM-V 4.7 (цільова модель, owner 2026-10-24)  **`[ ]`**

> Мета: **MiniCPM-V 4.7** — цільова модель двигуна — біжить у чистій Rust без Python:
> ваги official-GGUF (Apache-2.0, HF `openbmb/MiniCPM-V-4.7-1B-gguf` / `...-35B-A3B-gguf`),
> віжен-башта + компресор токенов + image-prefix — нові модулі; текст-путь = **бенд B13**
> (`PH-1201` spike 2026-10-24 показав: LLM-core = `qwen35`-гібрид, НЕ qwen2 — див.
> [`research/minicpm-v-4.7.md`](research/minicpm-v-4.7.md); image-prefix і serve чекають
> на B13). Оракул чисельності — `llama.cpp-omni` (mtmd) CLI як **вимірювальний
> інструмент** (прецедент llama-rs-оракула), у продукт не входить. 1B — перший; 35B-A3B —
> через expert-cache (PH-404) і лише на вузлі з диском.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-1201 | **spike: official GGUF + layout** — завантажити `MiniCPM-V-4.7-1B-gguf` (gitignored `data/models/`), `rusty inspect`: arch KV, vision-тензори, розміри/кванти, image-token vocab; факти → `docs/research/minicpm-v-4.7.md` | `[ ]` | `src/bin/rusty.rs` inspect + research-дока |
| PH-1202 | **vision tower у Rust** (`src/vit.rs`): 27× `v.blk` (attn q/k/v/out на 16 голів, ln1/ln2, GELU-MLP 1152→4304), patch-embed [14,14,3→1152], position_embd 70×70 + 2D-інтерполяція, post_ln — усі ваги **BF16(30)** через дейквант з PH-1201; поверх існуючих `QMat`/`matmul`-ядер; golden-фіксайти з `llama.cpp-omni` mtmd CLI (in-process bench-паритет, як `benches/attn.rs`) | `[ ]` | `src/vit.rs` + unit + `#[ignore]` parity |
| PH-1203 | **`v.vit_merger` (4×/16× компресор: ds_ln 4608 + attn + ds_ffn 4608↔17216) + `mm.{input_norm,up,down}` MLP → 1024-dim image-токени**; slice-nums ≤ 9, `wa_layer_indexes=[6]` | `[ ]` | `src/vit.rs` (merge/project) + фіксація форми |
| PH-1204 | **image-token prefix у prefill** (чекає B13-core): вставка vision-ембедингів за placeholder-токенами, chat-шаблон + specials для мови MiniCPM; `DenseDecoder::prefill` + `block_forward_seq` — embedding-комбінування без re-prefill | `[ ]` | `src/runner.rs`/`textgen.rs` + інтеграція (text-only regression green) |
| PH-1205 | **serve мультимодальний**: `messages[].images` base64 (PNG/JPEG через `image` crate — pure Rust; MVP: PPM/BMP без залежностей), E2E greedy-parity з mtmd-оракулом на 3–5 зображеннях, provider-row `rusty-vision` у GSV-каталозі (heartbeat-gated) | `[ ]` | `src/http.rs`/`serve.rs` + `tests/vision.rs` + GSV catalog |
| PH-1206 | **MiniCPM-V 4.7-35B-A3B** (MoE) через `ExpertCache` (PH-404): longest-first pin + NVMe-offload + LES-style lookahead-прелоад оцінка; runs only на виділеному вузлі (диск-gate S0) | `[ ]` | `src/moe.rs` + bench + fleet-дрен |

---

## Band B13 — Text core `qwen35` (ГІБРИД GDN + full-attention)  **`[~]`** (`PH-1301` done, 2026-10-24)

> Мета: LLM-каркас MiniCPM-V 4.7 — **`qwen35`**: 24 шари × d_model 1024, **18 GDN-шарів
> (Gated DeltaNet: mixed qkv 6144, depthwise conv k4, delta-rule стан 16×128×128, out
> gate 2048)** + **6 full-attn шарів (кожний 4-й; GQA 8q/2kv, head_dim 256, QK-norm,
> q-gate 4096)** + sectioned RoPE `[11,11,10,0]` "zyxt" + pre-tokenizer `qwen35` +
> vocab 248 094. Це головний диференціатор after-MoE: **лінійна увага в чистій Rust**
> (декод O(1)/токен — стан, а не KV). Без нього B12 image-prefix і serve не куди
> вставляти. Оракул: `llama.cpp-omni` CLI greedy-паритет (вимірювальний інструмент).
> Деталі layout — [`research/minicpm-v-4.7.md`](research/minicpm-v-4.7.md).

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-1301 | `model.rs`: arch `qwen35` — config-ключі (ssm.*, full_attention_interval, key/value_length, rope sections) + `Geometry`-розширення + `validate` під нові імена тензорів (attn_qkv/attn_gate/ssm_* vs attn_q/k/v+norm) | `[x]` | `src/model.rs` — `LayerKind::{Recurrent,Full}`, `layer_kind()` (Bool-масив > interval), `head_dim()=key_length`, `gdn_qkv_dim`/`full_q_dim`; validate-гілка per-kind (pre-FFN norm = `post_attention_norm`, `ffn_norm` немає); 4 synth-тести + `#[ignore]` real: `rusty validate MiniCPM-V-4_7-1B-Q4_K_M.gguf` = OK |
| PH-1302 | **GDN-шар** (`src/gdn.rs`): mixed qkv → depthwise conv1d(k4, F32) → delta-rule recurrent (16 груп × 128×128 стан, dt/alpha/beta, ssm_norm) → out-gate → 1024; prefill-скан по рядках + O(1) decode-крок; bit-stable oracle на synth | `[x]` | `src/gdn.rs` — `delta_step` (фікс-порядок pred→Δ→update→read), L2 q/k + `dk^-0.5`, gated head-RMSNorm `swish(z)` (конвенція — `PH-1305`); `prefill_is_bit_identical_to_decode_steps` + textbook-тест + contract 1 MiB/блок; scratch `dv ≤ 128` |
| PH-1303 | **full-attn шар**: GQA 8/2 head_dim 256 + **QK-norm** + q-gate (attn_q 4096) + zyxt-sectioned rope [11,11,10,0]; reuse `attention_*` з винесеними нормами | `[x]` | `src/q35.rs` — `Q35Geom::from_config`, `project_rope_append` (norm→rope→кеш), `attend_row` (scale `hd^-0.5`, libm softmax), `o·σ(gate)` (конвенція — 1305), shape-guard; 5 unit incl. prefill==steps біт-і + cross-check rope==`infer::rope_in_place`; zyxt = M-RoPE слоти (PR #29416), text = partial NeoX 64/256 |
| PH-1304 | pre-tokenizer `qwen35` (`gpt2.rs`-сімейство) + chat-template specials (248045/6) + `DenseRunner`-wire 18+6 шарів; CLI `rusty run` на qwen35-файлі | `[ ]` | `src/gpt2.rs`/`runner.rs` + text round-trip |
| PH-1305 | **text E2E greedy-паритет** 1B-Q4_K_M проти `llama.cpp-omni` CLI (3–5 промптів, `#[ignore]` release-тест + golden-файкстура) + perf-зріз (prefill/decode на 12 потоків) | `[ ]` | `tests/qwen35_parity.rs` + `docs/research/qwen35-parity.md` |
| PH-1306 | (далі) serve на qwen35-core: text-only provider + перемикання id `rusty-05`→`rusty-mcp-v47` у GSV-каталозі; B12-PH-1204 розблоковано | `[ ]` | `src/serve.rs` + GSV catalog row |
