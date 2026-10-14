# rusty-engine — Roadmap

**Product id:** `rusty-engine` · Root `S:/rust/rusty-engine` · Rust-first plugin (MIT).

No open PH-* without a band. «agi» у сесії (GSV hub) створює тікети по цьому роадмепу
через `gsv_tickets_create { product:"rusty-engine", scenario_id }` / `gsv_tickets_hook`
(catalog id `rusty-engine-bootstrap` / `rusty-engine-gguf` / `rusty-engine-moe-core`,
додаються за потребою). Канон концепції: [`concept.md`](./concept.md). Глобальні правила:
`S:/rust/GSV/AGENTS.md`.

## Легенда стану

- `[ ]` — не почато · `[~]` — в роботі · `[x]` — закрито (доказ + тест).
- ID формату `PH-<band><seq>` (band B4 → `PH-401..`).

## Порядок смуг (логіка побудови)

`фундамент → контейнер(GGUF) → ядра(quant) → dense-контроль → **MoE-ядро** → сервер →
портативність(exe/bin/apk) → Rust-docker-хости → Rust-супервізор(master/slave) →
розподілений MoE → Web-UI → дистр/версії/publish`.

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

## Band B1 — GGUF-чанкер (pure Rust)  **`[x]` (усі PH-101..107 done, 2026-10-14)**

> Мета: прочитати реальний GGUF-MoE-файл без llama.cpp. **Досягнуто** — контейнер цілком
> в Rust. Фактаж: [`docs/research/gguf-format.md`](research/gguf-format.md) +
> [`docs/research/ggml-blocks.md`](research/ggml-blocks.md).
> **Контекст-дисципліна:** std-only (101/102/106/107), mmap `103` (`memmap2`), `104`
> (block-sizes усіх 5 + scalar dequant Q4_0/Q8_0) і `105` (tokenizer-metadata ingest) —
> **B1 завершено**. K-quant dequant math → B2; BPE-алгоритм → B3.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-101 | GGUF v3 header + KV-metadata парсер (std-only) | `[x]` | `src/gguf.rs` — `parse()`/`Value`/`ValueKind`, 7 unit |
| PH-102 | tensor-info table (name/dims/dtype/offset) | `[x]` | `src/gguf.rs` — `TensorEntry`/`ggml_dtype`, round-trip synth |
| PH-103 | mmap/mlock staged loader (disk→RAM, progress) | `[x]` | `src/loader.rs` — `MmapModel::open`/`lock_pages`(unix)/`staged_read`(+progress)/`load_staged`; `memmap2`; 3 unit |
| PH-104 | quant-блоки: sizes усіх 5 + scalar dequant Q4_0/Q8_0 (K-quant → B2) | `[x]` | `src/quant.rs` — `block_info`/`bytes_for_numel`/`dequant`+`f16_to_f32`; 6 unit |
| PH-105 | tokenizer-metadata ingest (model/tokens/type/scores/merges/bos/eos/pre) | `[x]` | `src/tokenizer.rs` — `TokenizerMeta::from_file`; 2 unit |
| PH-106 | round-trip на disk (write synth GGUF → read via `parse_file`) | `[x]` | `src/gguf.rs` — `round_trips_a_real_file_via_std_io` (real `.gguf` bytes) |
| PH-107 | CLI `rusty inspect model.gguf` | `[x]` | `src/bin/rusty.rs` — `inspect()` summary + graceful errors |

---

## Band B2 — Quantized kernels + SIMD  **`[~]` (PH-201 / PH-201+ done, 2026-10-14)**

> Мета: швидкі ядра для квантів, x86 + arm64, feature-gated dispatch.
> Скалярний **reference-оракул** для ВСІХ 5 форматів готовий (Q4_0/Q8_0 fused dot +
> Q4_K/Q5_K/Q6_K dequant). Лишились: repack (PH-202), SIMD (PH-203), fused K-quant gemv (PH-204).

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-201 | scalar reference (dot, gemv) — Q4_0/Q8_0 fused | `[x]` | `src/kernel.rs` — `dot_q4_0`/`dot_q8_0`/`matvec_q4_0`/`matvec_q8_0` (== dequant-then-dot, 4 unit) |
| PH-201+ | scalar reference dequant Q4_K/Q5_K/Q6_K (all-block oracle) | `[x]` | `src/quant.rs` — `dequant_q{4,5,6}_k`+`get_scale_min_k4`; 4 parity; C-ref кеш |
| PH-202 | repack у compute-friendly layout | `[ ]` | `src/core/kernel/repack.rs` |
| PH-203 | SSE/AVX/AVX2 (x86) + NEON (arm64), runtime dispatch | `[ ]` | `src/core/kernel/simd.rs` |
| PH-204 | fused dequant+dot inner loops | `[ ]` | `src/core/kernel/mod.rs` |
| PH-205 | bench vs scalar (+ vs llama-rs підлога) | `[ ]` | `benches/quant.rs` |

---

## Band B3 — Dense forward (control path)  **`[~]` (PH-301..304 done, 2026-10-14)**

> Мета: згенерувати текст на DENSE-моделі як контроль перед MoE. Примітиви готові;
> решта (`PH-305/306`) потребує реальної GGUF-моделі.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-301 | RMSNorm (+ SiLU/MatMul — уже в `moe.rs`) | `[x]` | `src/infer.rs` — `rmsnorm`; 50 tests |
| PH-302 | causal attention GQA/MQA + RoPE (primitives) | `[x]` | `src/infer.rs` — `rope`/`apply_causal_mask`/`attention_causal`/`gqa_repeat_kv` (multi-head loop → PH-305) |
| PH-303 | paged KV-cache (page alloc, vLLM-ідея) | `[x]` | `src/infer.rs` — `PagedKvCache::append/get` round-trip across pages |
| PH-304 | sampler chain (temp/top-k/top-p/repetition/argmax) | `[x]` | `src/infer.rs` — `sample`/`mask_top_k`/`mask_top_p`/`apply_repetition_penalty`/`argmax` (grammar →later) |
| PH-305 | end-to-end dense-генерація (TinyLlama/Qwen2 GGUF), golden tokens | `[ ]` | `tests/dense_gen.rs` |
| PH-306 | CLI `rusty run -m model.gguf -p "..."` | `[ ]` | `src/bin/rusty.rs` |

---

## Band B4 — MoE-ядро (ДИФЕРЕНЦІЯТОР)  **`[~]` (PH-401/402/403/406a done, 2026-10-14)**

> Мета: Qwen3-30B-A3B / Mixtral GGUF **нативно в Rust**, минаючи llama.cpp.
> **Повний compute-цикл MoE-шара вже зʼєднаний і детермінований** (`moe_layer_forward`);
> лишилось підвʼязати реальні деquant-тензори (PH-406) + оптимізації (PH-404/407) + parity (PH-405).
> Підлога: llama-rs Qwen3-30B-A3B ≈ 0.497 tok/s decode — ціль ≥ parity.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-401 | top-k router: softmax-renorm + sigmoid + noaux_tc group | `[x]` | `src/moe.rs` — `route`/`top_k_sigmoid`/`route_group` (simplified; exact top-2/grp → PH-405) |
| PH-402 | expert FFN SwiGLU (`ffn_gate/up/down_exps`) | `[x]` | `src/moe.rs` — `expert_ffn`/`swiglu`/`silu`/`matvec`; 44 unit |
| PH-403 | shared-expert merge (DeepSeek/Qwen3) | `[x]` | `src/moe.rs` — `combine` (shared=Some/None, shape-validated) |
| PH-406a | MoE FFN **layer forward** (router→experts→shared) | `[x]` | `src/moe.rs` — `moe_layer_forward`/`ExpertWeights`; 2 integration (63 total) |
| PH-404 | expert-cache LRU + RAM/NVMe offload + hot prefetch | `[ ]` | `src/moe/cache.rs` |
| PH-405 | greedy-parity vs llama.cpp-референс + perplexity smoke | `[ ]` | `tests/moe_parity.rs` |
| PH-406 | run Qwen3-30B-A3B / Mixtral GGUF pure-Rust | `[ ]` | `src/moe/forward.rs` |
| PH-407 | bench tok/s vs llama-rs baseline | `[ ]` | `benches/moe.rs` |

---

## Band B5 — Single-node server  **`[ ]`**

> Мета: `rusty serve` — OpenAI-compat, стрімінг, keep-live в GSV.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-501 | HTTP server (tokio+axum) — `/v1/models`,`/v1/chat/completions` | `[ ]` | `src/serve/http.rs` |
| PH-502 | SSE-стрімінг токенів, stop-tokens, usage | `[ ]` | `src/serve/stream.rs` |
| PH-503 | prompt-cache / multi-slot | `[ ]` | `src/serve/slot.rs` |
| PH-504 | GSV keep-live `target/live/rusty_heartbeat.json` (GSV_LIVE=1) | `[ ]` | `src/serve/heartbeat.rs` |
| PH-505 | метрики tok/s·RAM·active-experts endpoint | `[ ]` | `src/serve/metrics.rs` |
| PH-506 | integration test + smoke curl | `[ ]` | `tests/serve.rs` |

---

## Band B6 — Build matrix: exe / bin / apk  **`[ ]`**

> Мета: один портативний бінарник на OS×arch.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-601 | Windows x86_64 `rusty.exe` (gnu primary + msvc) | `[ ]` | `xtask/build_matrix.rs` |
| PH-602 | Linux x86_64 `rusty` | `[ ]` | matrix |
| PH-603 | Linux aarch64 `rusty` (Pi-class) | `[ ]` | matrix |
| PH-604 | Android arm64 `rusty.apk`/`.so` (cargo-ndk/Termux, **no Docker**) | `[ ]` | `.cargo/config` + ndk |
| PH-605 | `cargo xtask build-matrix` + SHA-256 артефактів | `[ ]` | xtask |
| PH-606 | feature-gated backends (neon/sse/avx, mmap) single-binary | `[ ]` | `Cargo.toml [features]` |

---

## Band B7 — Rust-docker на Win/Linux хостах  **`[ ]`**

> Мета: slave у Rust-docker-контейнері на localhost, exposed через локальний Wi-Fi/LAN.
> **Android — поза Docker** (B6/B8). Dry-run за замовчуванням, `--live` за згодою owner.

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-701 | Rust docker client (bollard): build+run slave-образу | `[ ]` | `src/docker/client.rs` |
| PH-702 | образ на таргет + `.dockerignore` | `[ ]` | `docker/` |
| PH-703 | LAN/Wi-Fi expose (bind LAN addr, GSV band-233 LAN-first) | `[ ]` | serve bind |
| PH-704 | dry-run default + `--live` owner-consent gate | `[ ]` | network module |
| PH-705 | container lifecycle contract (create/start/stop/inspect) | `[ ]` | `tests/docker_contract.rs` |

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
