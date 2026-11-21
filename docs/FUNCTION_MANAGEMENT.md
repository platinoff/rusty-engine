# rusty-engine — FUNCTION_MANAGEMENT (симбіоз)

**Companion:** `S:/rust/GSV/docs/gsv/GSV_FUNCTION_MANAGEMENT.md` · [`GSV_ECOSYSTEM_SYMBIOSIS.md`](../../GSV/docs/gsv/GSV_ECOSYSTEM_SYMBIOSIS.md) · [`GSV_AGI_PATH.md`](../../GSV/docs/gsv/GSV_AGI_PATH.md)  
**Шар:** plugin (engine) · **Repo:** `https://github.com/platinoff/rusty-engine` · **HEAD:** see `git log -1`

## 1. Оголошені функції

| Функція | Модулі/шляхи | Статус | Примітка |
|---------|--------------|--------|----------|
| Pure-Rust MoE inference (GGUF-first) | `src/`, `Cargo.toml` (`rusty-engine`) | active | MiniCPM-V 4.7 target; OpenAI-compat serve surface |
| OpenAI-compat serve / chat | `src/bin/rusty.rs`, serve/chatml paths | active | Local hybrid lane; heartbeat-gated in GSV omni catalog |
| NanoJev DecisionModel eval/replay | `src/nanojev/`, `src/bin/nanojev.rs`, `docs/NANOJEV_RUNBOOK.md` | active (eval) | **Safetensors eval/research**, not GGUF omni serve lane |
| WSP/Astra research hooks | `docs/WSP_*`, nanojev/checkpoint workflows | research | Owner-gated for Phase 2 deploy |
| Bench/xtask gates | `benches/`, `xtask/` | active | `cargo xtask check` (fmt+clippy -D warnings+test) |

## 2. Звірка / gates

- Local dev gate: `cargo xtask check` (per `AGENTS.md`).
- NanoJev inspect smoke (2026-11-21): `nanojev inspect --checkpoint data/models/nanojev` → **322 tensors** readable.
- Model weights/checkpoints are local data assets; do not commit.

## 3. Дублі / межі (посилання на master §3)

- **D1** OpenAI-compat serve: intentional parallel engine plugin (with llama-rs). GSV does not host inference.
- **D9** 35B-A3B: engine execute; GSV mesh dispatches — normal boundary.
- NanoJev safetensors ≠ GGUF omni catalog model (GSV `LOCAL_MODEL_ROLES_2026-11-21.md`).

## 4. Owner-рішення / наступні кроки

- Keep NanoJev artifacts local; use replay/predict smoke on demand.
- WSP Phase 2 / deploy decisions remain owner-gated.
- When adding features, update this skeleton + GSV symbiosis/FM canon if boundary changes.
