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

**B1 — GGUF-чанкер:** `[x]` **ЗАВЕРШЕНО** (усі PH-101..107) — GGUF-контейнер читається
повністю в Rust без llama.cpp. **35 тестів**, clippy `-D` 0. Фактаж — `docs/research/gguf-format.md`
+ `docs/research/ggml-blocks.md` (cached через explore-subagent).

- `PH-101/102` `src/gguf.rs` — `parse()` header+KV (усі `GGUF_TYPE_*`, вкладені arrays) + tensor-info (`TensorEntry`/`ggml_dtype`).
- `PH-103` `src/loader.rs` — `MmapModel::open` (zero-copy, `memmap2` — **перша не-std залежність**, регістр перевірено на старті), `lock_pages`(unix-gated), `staged_read` disk→RAM + progress, `load_staged`.
- `PH-104` `src/quant.rs` — `f16_to_f32`; `block_info`/`bytes_for_numel` для **усіх 5** (18/34/144/176/210 B/block); scalar `dequant_q4_0` (**split-half**) + `dequant_q8_0` (signed); K-quant *dequant math* → **B2**.
- `PH-105` `src/tokenizer.rs` — `TokenizerMeta::from_file` (model/tokens/token_type/scores/merges/bos/eos/pre + accessors); BPE-алгоритм → **B3**.
- `PH-106` disk round-trip (`parse_file` read synth `.gguf` з диска); `PH-107` `rusty inspect` CLI.

**Наступний крок:** B2 (`PH-201` scalar reference kernels) — або стрибок до **B4 MoE-ядро**
(`rusty-engine-moe-core` ready: `PH-401/402/403` router/swiglu/shared-expert — pure-f32, testable
без повного пайплайну). Канон: `docs/roadmap.md`, `NEXT_SESSION_PROMPT.md`.
