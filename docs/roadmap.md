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

## Band B2 — Quantized kernels + SIMD  **`[~]` (PH-201 / PH-201+ / PH-201c / PH-204 / PH-202 done, 2026-10-14)**

> Мета: швидкі ядра для квантів, x86 + arm64, feature-gated dispatch.
> Скалярний **reference-оракул** для Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K + fused dot/gemv
> для Q4_0/Q8_0 **і K-квантів** (`PH-204`). Лишились: repack (`PH-202`), SIMD (`PH-203`),
> bench (`PH-205`).

| ID | Задача | Стан | Доказ |
|----|--------|------|-------|
| PH-201 | scalar reference (dot, gemv) — Q4_0/Q8_0 fused | `[x]` | `src/kernel.rs` — `dot_q4_0`/`dot_q8_0`/`matvec_q4_0`/`matvec_q8_0` (== dequant-then-dot, 4 unit) |
| PH-201+ | scalar reference dequant Q4_K/Q5_K/Q6_K (all-block oracle) | `[x]` | `src/quant.rs` — `dequant_q{4,5,6}_k`+`get_scale_min_k4`; 4 parity; C-ref кеш |
| PH-201c | scalar reference dequant **Q5_0** (unblocks real embeddings) | `[x]` | `src/quant.rs` — `dequant_q5_0`(22B, bias −16) + `DType::Q5_0`/`ggml_dtype(6)`; real token_embd 136M finite; 71 unit |
| PH-201d | **fused Q5_0 dot/gemv** (closes the quant matrix) | `[x]` | `src/kernel.rs` — `dot_q5_0`(d@0/qh@2/qs@6, −16 bias, 5th-bit `qh>>j`) + `matvec_q5_0`; == `dot(dequant_q5_0, b)` / `matvec(dequant_q5_0)`; 2 unit |
| PH-204 | **fused dequant+dot** K-quant (Q4_K/Q5_K/Q6_K, stack `[f32;256]`, no Vec) | `[x]` | `src/kernel.rs` — `dot_q{4,5,6}_k` + `matvec_q{4,5,6}_k` macro (relative-parity vs `dot(dequant_qX_k, b)`, 4 unit) |
| PH-202 | **repack** у compute-friendly layout (contiguous scales + de-interleaved nibble planes) | `[x]` | `src/repack.rs` — `RepackedQ4_0 {d,lo,hi}` + `RepackedQ8_0 {d,v}` + `repack_q{4,5}_*` + `dot_repacked_q{4,5}_*` + `matvec_repacked_q4_0`; **exact** vs `dot_q4_0` / `dot_q8_0` / `matvec(dequant, x)` (5 unit) |
| PH-203 | **SSE/AVX/NEON dispatch** (SSE2 Q4_0+Q8_0, AVX2 Q8_0) | `[~]` | `src/kernel.rs` — `dot_q4_0_sse2` + `dot_q8_0_sse2` + `dot_q8_0_avx2` (`core::arch`, `#[target_feature]`, `hsum_ps_sse2` reduce) + `dot/matvec_{q4,q8}_0_dispatch` (AVX2→SSE2→scalar ladder, runtime `is_x86_feature_detected!`); `#[cfg(x86_64)]` parity SSE2+AVX2==scalar. NEON → далі |
| PH-205 | bench vs scalar (+ vs llama-rs підлога) | `[ ]` | `benches/quant.rs` |

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

> Мета: Qwen3-30B-A3B / Mixtral GGUF **нативно в Rust**, минаючи llama.cpp.
> **Повний compute-цикл MoE-шара вже зʼєднаний і детермінований** (`moe_layer_forward`);
> лишилось підвʼязати реальні деquant-тензори (PH-406) + оптимізації (PH-404/407) + parity (PH-405).
> Підлога: llama-rs Qwen3-30B-A3B ≈ 0.497 tok/s decode — ціль ≥ parity.

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
| PH-406 | run Qwen3-30B-A3B / Mixtral GGUF pure-Rust | `[ ]` | `src/moe/forward.rs` |
| PH-407 | bench tok/s vs llama-rs baseline | `[ ]` | `benches/moe.rs` |

---

## Band B5 — Single-node server  **`[~]` (PH-501a std-only done, 2026-10-14)**

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
| PH-505 | метрики tok/s·RAM·active-experts endpoint | `[~]` | `src/hub.rs` — `Metrics` (`zero`/`record`/`tok_per_sec` zero-guard/`to_json`) + `GET /metrics` (`http::route_with`); serve рахує calls per-conn. RAM + tok/s на реальних токенах → B5 далі |
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
| PH-605 | **`cargo xtask build-matrix`** + SHA-256 артефактів | `[x]` | `xtask/src/main.rs` — `TARGETS` matrix (win-gnu/msvc, linux-x86_64/aarch64, android-arm64 no-Docker) + `sha256_hex` (pure `std`, FIPS 180-4, multi-block) + `build-matrix` рендер з хешем наявних бінарників; 2 unit (NIST vectors + coverage). Фікс `main` argv `skip(2)`→`skip(1)` |
| PH-606 | **feature-gated backends** (sse/avx/neon single-binary) | `[x]` | `Cargo.toml [features] sse/avx/neon` (default none) + `src/backend.rs` `Backend`/`from_flags`/`compile_backends`/`select` (найсильніший ввімкнений, Scalar-fallback); `cfg(test)`-обмежені під кожен feature-набір. Dispatch-substrate для `PH-203` |

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
