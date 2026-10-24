# rusty-engine — PROJECT STATUS

**Snapshot:** 2026-10-24 (останнє: **`PH-1201` spike ✅ — MiniCPM-V 4.7-1B у `data/models/`,
`rusty inspect` прокачано (KV-дамп+histogram+`[N]`); text-core = **`qwen35`-гібрид**
(18 GDN + 6 full-attn, QK-norm, zyxt-rope, vocab 248 094) → **новий бенд B13**; vision =
clip-548M **BF16(30)** (+фікс dtype-таблиці й dequant). Раніше цього ж дня: **`PH-203s` ❌**
і **`PH-203t` ❌** (обидва левери prefill виміряно й відкинуто; продакшн = PH-203o/PH-203r,
golden не рухався). Модельна політика: цільова модель = **MiniCPM-V 4.7**. Попередній ланцюг:
2026-10-22 agi-ланцюг: B0→B1 GGUF→B2 quant→B3 dense/KV/sampler/BPE→B4 MoE→B5
serve/heartbeat → **PH-405b forward-parity з llama-rs ✅** → **PH-507 hub provider-row E2E ✅
(JSON + SSE) через живий хаб `:9999`** → **B2 SIMD bench ✅ (`PH-203` x86 / `PH-205`, 1.8–5.3×)**
→ **`PH-505` residual ✅ (per-process RAM у `/metrics`)** → **`PH-203d` AVX2+FMA fused dots ✅
(Q8_0 dot 1.72→2.09×, live decode 2.27→2.68 tok/s)** → **`PH-203e` row-parallel matvec/matmul ✅
(decode 2.68→4.2–5.1 tok/s, prefill 1.85×, `RUSTY_THREADS`)** → **`PH-203f` persistent worker pool ✅
(decode 4.4→7.2–7.9 tok/s, matvec 5.1×, prefill matmul 4.08×)** → **`PH-203g` RoPE once per
position + rotated KV cache ✅ (decode 7.4→10.4 tok/s, prefill 321 tok 25.5→5.9 s, 961 tok
190.9→29.1 s)** → **`PH-203h` head-parallel prefill attention ✅ (961 tok 29.1→23.5 s)** →
**`PH-203i` f16-widening + q8_0 integer-dot probe ✅ (dot q8_0 9.55×, matvec 896² 11.8 GMAC/s,
live decode 10.9→27.6 tok/s; int8 slower than the fixed float path, not wired))** →
**`PH-203j` row-batched matmul ❌ (bit-exact але 3–4× повільніше, відкочено)** →
**`PH-203k` activation tiling + row-split projections ✅ (serial 1469→897 ms, prefill matmul
cols→rows 248→190 ms 1.30× in-process)** → **`PH-203l` full-row f32 accumulator ✅ (serial
859 ms, single thread 806→606 ms 1.33×, prefill parallel 92.5 ms / 45 GMAC/s; **live prefill
961 tok 26.9–27.3 → 18.9–19.0 s** на тому ж завантаженому боксі; golden fingerprint не зрушив —
recapture не потрібен)** → **`PH-203m` per-stage профайлер ✅ (`RUSTY_PROFILE=1`; live prefill
961 tok: matmul **84.4%**, attention 9.4%, other ~6%; знайдено, що down-проєкція 896×4864
іде **10.0 GMAC/s** проти 34.8–36.0 у решти форм)** → **`PH-203n` register blocking + 2-D
cache blocking + вісь поділу за `X` ✅ (усе біт-у-біт: один потік gate/up 606 → 199 ms,
down 688 → 218 ms; пул 74.5 / 57.7 GMAC/s; **live prefill 961 tok 15.2 → 10.44 s**,
down 10.0 → 36.4 GMAC/s; golden fingerprint не зрушив)** → **`PH-203o` prefill attention ✅
(скалярні оцінки → `dot_f32_x4` (одне завантаження `q` на 4 ключі) + `axpy_f32_dispatch` для
значень (біт-у-біт) + 14 голів × 4 round-robin раунди = 56 завдань: attention **2.02 → 0.81–0.88 s**,
live prefill **9.12–9.90 s**; fingerprint збігся)** → **`PH-203p` «other» розкладено й зрізано ✅
(нові стадії `norm/rope/kv-copy/swiglu/residual/embed` + `serial`/`unaccounted` у звіті; SwiGLU
на пулі 676 → **118 ms**, RoPE через таблицю `(cos,sin)` 205 → **33–41 ms**, serial 1010 → **260–297 ms**,
live prefill **8.49 → 6.90–7.84 s**, «other» 11.9 % → **3.4–3.8 %**; усе біт-у-біт)** →
**`PH-203q` K-тайл від L3 ✅ (down 36.8–41.9 → 49.7–56.0 GMAC/s, live 6.90–7.84 → 7.1–8.9 s на
завантаженому боксі; тайл 256 і x8-blocking виміряно й відкинуто)** → **`PH-203r` два рядки ваг
на прохід ✅ (пул: gate/up 76 → 102, down 77 → 108 GMAC/s; live wall **5.3–6.4 s**, matmul
4.1–4.7 s / 77 %, down 64–69 GMAC/s; golden не зрушив)** · **запушено 2026-10-22** (після
`PH-203r`).

## Верифікований стан (зміряно цього сеансу, не «на памʼять»)

| Метрика | Значення |
|---------|----------|
| `git status` | **clean** (0 drift) |
| Коміти | **107** усього (`git rev-list --count HEAD`); **48** у drain-серії B0→PH-1201 (B0→B5 serve/heartbeat + PH-405b + PH-507 + B2 SIMD bench + PH-505 RAM + PH-203d FMA + PH-203e parallel + PH-203f pool + PH-203g RoPE + PH-203h head-parallel attn + PH-203i f16/q8-int + PH-203j/K prefill matmul + PH-203l row accumulator + PH-203m stage profiler + PH-203n row/cache blocking + PH-203o attention x4/axpy + PH-203p «other» split + PH-203q K-tile/L3 + PH-203r two weight rows + PH-203s measured-rejection bench/hooks + PH-203t measured-rejection wrows4/interleaved (docs+bench) + model-policy B12 docs + PH-1201 MiniCPM-V spike (inspect KV/histogram, BF16-fix); **pushed 2026-10-24**) |
| Тести | **227** (208 lib + 5 dense-gen + 4 scaffold + 5 serve + 5 text-gen; +5 ignored parity), 0 failed |
| `cargo clippy --all-targets -- -D warnings` | **0** (чисто) |
| `cargo fmt --check` | **ok** |
| Rust LOC | **15173** (`src` 12387 + `tests` 1317 + `xtask` 319 + `benches` 1150; `wc -l`) |
| UI LOC (glue) | **157** (html+css+js) |
| **Rust ratio** | **99.0%** (ціль 95–100 ✓) |
| Docs LOC | 2524 (`git ls-files docs \| xargs wc -l`) |
| Tracked files | 68 |
| Реальна модель | `data/models/qwen2.5-0.5b-q4_k_m.gguf` (491 400 032 B, **gitignored**, sha256 `74a4da8c…`) — **лише внутрішній числовий оракул**; цільова модель = **MiniCPM-V 4.7** (B12, owner 2026-10-24) |
| Трекінг `data/` | лише `.gitkeep` (модель не в репо ✓) |
| Live `:8090` (12 потоків, 961-tok prefill, `RUSTY_PROFILE=1`) | **5.3–6.4 s** (бокс дрейфує ±20 % на однаковій роботі; міряти лише свіжим процесом) · matmul **4.1–4.7 s** (~77 %) · attention **0.74–0.78 s** (13.9 %) · down `896×4864` **64–69 GMAC/s** (було 36–42; 1.46–1.56 s) · gate/up **81–92** · attention QK 84–92 · swiglu 122–132 ms · rope 32–35 ms · **other 2.4–3 %** · decode 10.4–27.6 tok/s (сильно залежить від фонового навантаження боксу) |

## Що зроблено (канон смуг → `docs/roadmap.md`)

| Бенд | Статус | Ядро в Rust |
|------|--------|-------------|
| **B0** scaffold | ✅ | workspace + bin `rusty` + `xtask`; MoE `top_k`; 7 модулів |
| **B1** GGUF-чанкер | ✅ **complete** | header+KV+tensor-info, mmap/staged, sizes+Q4_0/Q8_0/Q4_K/Q5_K/Q6_K dequant, tokenizer-meta, `rusty inspect` |
| B1-real parity | ✅ | реальний Qwen2.5: парсер+`ggml_dtype` (баг off-by-K знайдено/виправлено) |
| **B2** kernels | 🟡 | `PH-201/201+/201c/201d` fused dot/gemv Q4_0/Q5_0/Q8_0 + oracle K-кванти; `PH-204` fused K-quant dot/gemv; `PH-202` repack; **`PH-203` x86 SIMD (SSE2 Q4_0/Q8_0, AVX2 Q4_0/Q5_0/Q8_0/Q4_K/Q6_K) + `PH-205` bench ✅ (1.8–5.3×)**; **`PH-203d` AVX2+FMA fused dots ✅ (Q8_0 dot 1.72→2.09×)**; **`PH-203e` row-parallel matvec/matmul ✅ (decode 2.68→4.2–5.1 tok/s, prefill 45.6→24.7 s, `RUSTY_THREADS`)**; **`PH-203f` persistent pool ✅ (decode 4.4→7.2–7.9 tok/s, gate/up matvec 1.270→0.684 ms, prefill matmul 4.08×)**; **`PH-203g` RoPE once/position + rotated KV ✅ (decode→10.4 tok/s, prefill 321 tok 25.5→5.9 s, 961 tok 190.9→29.1 s)**; **`PH-203h` head-parallel prefill attention ✅ (961 tok →23.5 s, 8.1× від початку сесії)**; **`PH-203i` f16-widening + q8_0 integer-dot probe ✅ (dot q8_0 9.55×, matvec 896² 11.8 GMAC/s, decode 27.6 tok/s; int8 не wired — повільніший за виправлений float)**; **`PH-203k` activation tiling (16 рядків, 57 KiB) + row-split проєкцій ✅ (prefill matmul serial 1469→897 ms, cols→rows 1.30× in-process)**; **`PH-203l` full-row f32 акумулятор ✅ (один hsum на рядок замість 28: single-thread 806→606 ms 1.33×, serial 859 ms, QMat 92.5 ms = 45 GMAC/s; порядок сум інший, але greedy-fingerprint збігся — golden не перезахоплювався; live prefill 961 tok 23.5→18.9 s)**; **`PH-203m` per-stage профайлер ✅ (`src/profile.rs`, `RUSTY_PROFILE=1`, атомарні лічильники + розбивка matmul за формою; live: matmul 84.4%, attention 9.4%; down 896×4864 = 10.0 GMAC/s проти 34.8–36.0 у решти — L2-bound)**; **`PH-203n` 4-рядковий register blocking + двовимірне cache-блокування (тайл 96 зовні циклу ваг, `rb` 1024) + вісь поділу за `X` (`MATMUL_X_L3_BUDGET` 6 MiB) ✅ (біт-у-біт, golden не зрушив: один потік gate/up 606→199 ms / 21.0 GMAC/s, down 688→218 ms / 18.9; пул 74.5 та 57.7 GMAC/s; live prefill 961 tok 15.2→10.44 s, down 10.0→36.4 GMAC/s)**; **`PH-203o` prefill attention ✅ (оцінки через `dot_f32_x4` — одне завантаження `q` на 4 ключі, значення через `axpy_f32_dispatch` (біт-у-біт), 14 голів × 4 round-robin раунди = 56 завдань: attention 2.02→0.84 s, live prefill 10.44→9.1–9.9 s; greedy-fingerprint збігся)**; **`PH-203p` «other» ✅ (профайлер дістав `Stage::{Norm,Rope,KvCopy,Swiglu,Residual,Embed}` + `serial`/`unaccounted`: swiglu 676→118 ms на пулі, RoPE через `rope_table` 205→33–41 ms, serial 1010→260–297 ms; live prefill 8.49→6.90–7.84 s, other 11.9→3.4–3.8 %; усе біт-у-біт, fingerprint не зрушив)**; лишився NEON |
| **B3** forward | ✅ **complete** | `PH-301..306` + `PH-305a/c/d/e` + **`PH-305`/`PH-306`** (multi-layer dense runner + CLI `rusty run`, real Qwen2.5 24-layer prefill) + **`PH-305b`** (BPE byte-fallback + atomic specials) + **`PH-306b`** (text pipeline: `src/textgen.rs` `TextRunner` + `rusty run -p "text"` + `tests/text_gen.rs` synth round-trip); **`PH-405`** numeric regression-gate + **`PH-405b`** semantic greedy-parity з llama-rs ✅ |
| **B4** MoE-core | 🟡 | `PH-401..403` (router softmax/sigmoid/group, SwiGLU, shared) + `PH-406a` `moe_layer_forward` + **`PH-404` expert-cache LRU** (`src/cache.rs`, budgeted + pin + prefetch); parity/run/bench (405/406/407) open |
| **B5** serve | 🟡 | PH-501a/b (std HTTP + real TextRunner), PH-502 (SSE), PH-503 (slot-cache), PH-504 (heartbeat), PH-505 (**metrics + per-process RAM**), PH-506 (serve integration tests ✅); **PH-507 hub provider-row E2E ✅ (`rusty-serve` через хаб, JSON+SSE)** |
| B6 build-matrix(exe/bin/apk) / B7 docker / B8 supervisor / B9 dist-MoE / B10 UI / B11 distribs | ☐ | заплановано (бенди в роадмепі) |
| **B12** vision (MiniCPM-V 4.7) | 🟡 | **`PH-1201` spike ✅ 2026-10-24** (`rusty inspect` KV+histogram, ваги у data/models/,research-дока); PH-1202/1203 (ViT-1152/27× + vit_merger, BF16) — далі; image-prefix (1204) **чекає B13** |
| **B13** text-core qwen35 (гібрид GDN+full) | ☐ | відкриття `PH-1201`: 18 GDN-шарів (mixed qkv 6144, conv1d k4, delta-rule 16×128²) + 6 full-attn (GQA 8/2×256, **QK-norm**, q-gate) + zyxt-rope + pre `qwen35` + vocab 248 094; `PH-1301..1306`, оракул `llama.cpp-omni` |

**Між конекцій:** від токена до logits стек повний — `config → validate → dequant → embed →
block_forward → sampler → BPE` — і реальна геометрія Qwen2.5 підтверджена (`rusty validate` = OK).

## Поточна межа (чому не «далі сліпо»)

1. **`PH-405` числовий regression-gate готовий** (`tests/numeric_parity.rs` + golden на
   справжній Qwen2.5) — guard реґресій forward/quant/KV/sampler. **`PH-405b` semantic
   llama-rs parity — ✅ ВИПРАВЛЕНО (2026-10-17):** биті q/k/v-біаси Qwen2 + RoPE
   GPT-J→NeoX; наш greedy тепер байт-у-байт збігається із записаним оракулом (55/12/48),
   golden перезахоплено. Гейти — `tests/moe_parity.rs`.
2. HF-exact Qwen BPE (`PH-305b`: `qwen2` pre-regex, byte-fallback, specials) — для читабельного тексту.
3. Далі — важкі/інфра-бенди (NEON arm64, axum-server, fleet/docker, build-matrix/apk) — кожен окремий великий зріз. **B2 x86 SIMD + bench закрито** (`PH-203`/`PH-205`), **hot dots переведено на AVX2+FMA** (`PH-203d`), **проєкції паралеляться по рядках** (`PH-203e`, `RUSTY_THREADS`, дефолт = available_parallelism) через **постійний пул** (`PH-203f`: `thread::scope` давав 750–820 µs на проєкцію, ~90 ms/токен). **RoPE тепер рахується раз на позицію**, а KV-кеш зберігає вже повернуті ключі (`PH-203g`) — prefill більше не платить `powf` за кожну пару (query, key). Prefill-attention тепер теж паралелиться по головах (`PH-203h`, `pool::parallel_for`, поріг 32 токени); залишок prefill — сам matmul. `PH-203p` закрив останній «other»: після розкладу профайлером (swiglu 8 %, rope 2.4 %, norm/residual/kv <1 % кожен) SwiGLU пішов на пул, а RoPE отримав таблицю `(cos,sin)` на позицію; serial-частка впала з 11.9 % до 3.4–3.8 %, і prefill тепер на **84 %** складається з matmul + 9.8 % attention. `PH-203q` перевірив гіпотезу «K-тайл 256» і **відкинув її виміром** (256 на down-формі −44 %), замінивши константу на похідну від L3: `k_row_tile_with(n, threads) = clamp(6 MiB/(threads·n·4), 8, 96)` — кожен воркер тримає свій активаційний тайл, тож 4864-широкий вихід отримує 24 рядки, а 896-широкий — 96. Down-проєкція **36.8–41.9 → 49.7–56.0 GMAC/s** (2.40–2.73 → 1.79–2.02 s), gate/up без змін. Register blocking x8 виміряно й відкинуто (один потік +10 %, пул −40 %), заодно виправлено хвіст `n%8` у `dot_f32_avx2_x4`. `PH-203r` пішов далі й зняв load-port тиск: два дейквантованих рядки ваг на прохід
(`dot_f32_avx2_x4x2`, `6 loads / 8 FMA`) — gate/up у пулі **76 → 102 GMAC/s**, down
**77 → 108**, live wall **7.1–8.9 → 5.3–6.4 s**, matmul 4.1–4.7 s (77 %), attention 13.9 %.
Заодно **відкинуто** гіпотезу PH-203q про «тайл ділиться на воркерів»: тайл — спільний зріз `X`,
і з парою рядків pooled-оптимум повертається до 96 (down Q6_K: 24 → 75, 96 → 108, 128 → 119
бенч / 55 live), тож `MATMUL_ROW_TILE_K = MATMUL_ROW_TILE` і функцію форми видалено. `PH-203i` знайшов справжній ботлнек q8_0 (повільний `f16_to_f32` на блок, не i8→f32): точне бітове розширення дало 9.55× на dot і 11.8 GMAC/s на matvec без зміни чисел; int8-ядро лишилось оракул-тестованою альтернативою (0.89× після фіксу). **`PH-203s` (2026-10-24) перевірив тезу «attention exp-bound» і спростував її:** онлайн-softmax блоками по 128 ключів + `exp2`-таблиця (1024+1 entry, лінійна інтерполяція) реалізовані й прогнані в новому `benches/attn.rs` (min-of-5, один процес, Qwen-геометрія 961×64): варіанти **10.5–10.9 ms/head проти 9.4–10.5 у 3-pass libm** — таблиця не швидша за `f32::exp` цього бокса, а блокування додає +3–6 %; skip мертвих ключів не спрацьовує (`p<1e-8`: kept 100 %; `1e-4`: −8 % axpy без виграшу). Продакшн `forward.rs` відкату до PH-203o (біт-у-біт, golden не зрушив — recapture був байт-ідентичний); хуки `kernel::exp_neg_fast`/`exp2_table_ref` + бенч лишились як доказ.   Attention-левер закрито: стейдж спирається у `dot_f32_x4`/axpy (обидва AVX2), далі — лише памʼять matmul (`PH-203t`) і NEON. **`PH-203t` (того ж дня) закрив і цю тезу:** quad-pass `w_rows=4` (X-тайл `m/4`) пул-A/B у `benches/kquant.rs`+`quant.rs` (3 проходи) — без стійкого виграшу (gate/up −4.4…+1.5 %) і з провалом на wide down Q6_K (−20…−50 %: чотири 19.4 KiB rowbuf-и на воркер звалив спільний L3); NT-store на `out` пораховано <1 % і пропущено. `MATMUL_W_ROWS = 2`, біт-у-біт, golden не рухався. Побічно: **interleaved** t48-vs-t96 (12 раундів) показав паритет min при 11/12 переваги t48 по медіані — тайл руляє хвостом розподілу; метод альтернованих раундів — канон для майбутніх свіпів. далі — NEON і B12 (MiniCPM-V).

## Research / позиціонування

Див. **[`docs/research/engine-landscape-2026.md`](research/engine-landscape-2026.md)**:
vLLM v0.28 / SGLang v0.5.19 (RadixArk) / TensorRT-LLM v1.2.1 / llama.cpp / ktransformers.
`rusty-engine` — **не** GPU-cluster-throughput-ліга; ніша = pure-Rust MoE + один exe/bin/**apk**
+ edge/Android-no-Docker + GSV-native master/slave флот + нативний Rust-MoE (чого немає в llama-rs).

## Наступна `agi` (точка входу)

Див. **`docs/NEXT_SESSION_PROMPT.md`** (`Owner product pick: rusty-engine`). **`PH-507` E2E
зроблено** (лишається WiFi multi-host band 233 + IDE-chat `rusty/*`). **B2 x86 SIMD + bench
закрито**, matmul виведено на load-port/FMA-межу (77 % prefill, gate/up 81–92 GMAC/s, down
64–69). **`PH-203s` (attention softmax) і `PH-203t` (matmul w_rows=4 + NT-store) — обидва
виміряно й відкинуто 2026-10-24; левери prefill закрито до NEON.**
Наступний зріз — **`PH-1301`**: бенд **B13** — text-core `qwen35`-гібрид (`model.rs`
config/Geometry/validate → GDN 1302 → full-attn 1303 → pre-tokenizer/runner 1304 →
E2E greedy-паритет проти `llama.cpp-omni` CLI + perf 1305 → serve 1306). B12 vision
(1202/1203, `PH-1201` spike готовий — `docs/research/minicpm-v-4.7.md`) — паралельно;
image-prefix (1204) чекає B13. Готові green-зрізи далі: NEON arm64 (`PH-203`) · `PH-501`
(axum serve) · важкі/інфра-бенди (B6/B7).
