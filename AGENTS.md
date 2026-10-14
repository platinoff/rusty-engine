# AGENTS.md — rusty-engine (portable plugin)

This tree is a **portable plugin** of the rust-folder environment (`S:/rust`), a sibling
of llama-rs/poolAI. GSV is the **mother hub / master**. Canon сесії:
[`docs/concept.md`](docs/concept.md) + [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) +
[`docs/roadmap.md`](docs/roadmap.md).

## Global first

1. Read **`S:/rust/GSV/AGENTS.md`** and `S:/rust/GSV/docs/gsv/GSV_AGI_PATH.md`
   (`gsv://docs/agi-path`).
2. Then this product's rows — PRODUCTS.md row `rusty-engine`; HANDOFF
   `docs/HANDOFF_NEW_SESSION.md`; NEXT `docs/NEXT_SESSION_PROMPT.md`.
3. Local rules below cover **this product only**. Do **not** restate S0 / MSYS2 /
   `абракадабра` / MCP sandbox here.

## Product profile

- **id** `rusty-engine` · **Root** `S:/rust/rusty-engine` · **Kind** rust · **Role** plugin
- **Concept/Arch/Roadmap** — `docs/concept.md`, `docs/ARCHITECTURE.md`, `docs/roadmap.md` (bands B0–B11, `PH-<band><seq>`).
- **License** MIT (`LICENSE`, `Copyright (c) 2026 platinoff / Madevinc`) — публікований MIT-двигун.
- Tests: `cargo xtask check` (→ `cargo fmt --all` → `cargo clippy --all-targets -- -D warnings` → `cargo test`).
- Ratio: Rust **95–100%**; UI — тонкий glue, не heavy JS. Plugin-гейти (з B10):
  `tests/ui_glue_contracts.rs` (LOC-ratchet ≥0.95 + UI-інваріанти). **Не** `gsv-loc-audit` —
  його бін містить `CARGO_MANIFEST_DIR` і аудитує лише GSV-дерево, не plugin.
- UI-рендер — **Rust** (`src/ui.rs`: `/page/{name}` + `GET /api/ui/card/{name}`); дії —
  GET/POST-форми. `ui/js/app.js` — лише chrome (тема, сайдбар, хоткеї `d m i k f s o l`,
  панель `?`). Ніколи не повертати рендер у JS, не послаблювати рашет нижче 0.95.
- Toolchain: `rust-toolchain.toml` pin `1.92.0-x86_64-pc-windows-gnu` (той самий, що GSV). Shell — MSYS2 bash.

## Domain rules (local)

- **Rust-MoE-ядро:** роутер (top-k / sigmoid / noaux_tc group), expert-FFN (SwiGLU),
  shared-expert merge і quant-kernels мають бути **в Rust**; llama.cpp-методи
  **переносимо, не лінкуємо** (C++ core — заборонений шляхом).
- **GGUF-first:** єдиний контейнер моделей до B-safetensors (опційно пізніше). Target
  артефакти: `rusty.exe` (win) / `rusty` (linux) / `rusty.apk|.so` (android arm64) —
  `cargo xtask build-matrix`.
- **Docker = тільки Win/Linux хости** через Rust-клієнт (bollard), **dry-run завжди
  дефолт**, `--live` лише з явною згодою owner. **Android — без Docker** (cargo-ndk/Termux
  `.apk`/`.bin` як slave). Токени/секрети в `data/` (gitignored) — ніколи в коді/логах.
- **Оркестрація = GSV-native Rust-супервізор** («Rust k8s-lite»): hub `:9999` = master,
  `rusty` = slaves; registry/heartbeat(15s)/lease/placement, fail-open. Не k3s/Go.
- **Плагін GSV:** keep-live (`target/live/rusty_heartbeat.json`), алерти/тікети/fleet —
  через hub (`gsv_mcp_openbot`); не копіюємо `.agents/skills/` і не піднімаємо другий `:9999`.

## Must not

- Link/vendor llama.cpp or any C++ engine as the compute core.
- Copy `.agents/skills/` або GSV globals у цю папку.
- `git add -A`; ставимо лише взяті в спринт файли. Ніколи: `.env*`, `*.pem`/`*.key`, `data/*` (окрім `.gitkeep`).
- Push mid-спринту; один коміт наприкінці. GitHub-remote (MIT, `platinoff/rusty-engine`) —
  лише з B11 (PH-1105) і з явної згоди owner.
- Вимикати `--live`-гейт у docker-модулі без згоди owner.
