# NanoJev runbook (rusty-engine)

**Status:** Operational (inspect verified 2026-11-21; replay smoke attempted with small limit)  
**Scope:** Local NanoJev DecisionModel artifacts in `rusty-engine` and their role in GSV ecosystem.

## 1. What NanoJev is (and is not)

- **Is:** safetensors DecisionModel checkpoint (`nanojev-unified-games-v1`, Qwen3-0.6B backbone, SFT unified games policy: maze/snake/shooting basic+predict_position). Used for eval/predict/replay in `rusty-engine`.
- **Is not:** a GGUF model for omni/bunke-rock/rusty-serve GGUF routing lanes. Do not register it as a GGUF omni catalog model.

## 2. Artifact location

- Checkpoint dir: `S:\rust\rusty-engine\data\models\nanojev\`
- Key files:
  - `best.safetensors` (main weights)
  - `config.json`, `summary.json`
  - `tokenizer/` (`tokenizer.json`, `tokenizer_config.json`)
  - `unified_hard_test.jsonl`, `predictions_test.jsonl`, `unified_hard_manifest.json`
  - `TRAINING_RECIPE.md`, `backbone_config/`, sample `req_*.json`

Model weights are local data assets (not committed).

## 3. CLI (rusty-engine)

Binary: `cargo run -p rusty-engine --bin nanojev -- ...` (or existing `target/release/nanojev.exe`).

### Inspect
```bash
cd /s/rust/rusty-engine
./target/release/nanojev.exe inspect --checkpoint data/models/nanojev
```
**Verified 2026-11-21:** `tensors: 322`, readable safetensors map (Qwen3-0.6B backbone, `embed_tokens [151936,1024]`, 28-layer blocks).

### Predict
```bash
./target/release/nanojev.exe predict \
  --checkpoint data/models/nanojev \
  --input data/models/nanojev/req_one_maze.json
```
Optional: `--max-length N`, `--precision ...` (see `src/bin/nanojev.rs`).

### Replay (cohort comparison)
```bash
./target/release/nanojev.exe replay \
  --checkpoint data/models/nanojev \
  --cohort data/models/nanojev/unified_hard_test.jsonl \
  --predictions data/models/nanojev/predictions_test.jsonl \
  --limit 2 \
  --out /tmp/nanojev_replay_smoke.json
```
Use `--limit` for cheap smoke checks; full cohort replay is heavier.

## 4. Symbiosis / ecosystem role

- **GGUF omni serving/routing:** Granite, MiniCPM 1B/35B, Inkling (GGUF+mmproj) — cataloged in GSV omni (`S:/rust/GSV/src/boxes/omni/catalog.rs`, `docs/gsv/GSV_OMNI_CATALOG.md`).
- **NanoJev safetensors:** local eval/replay/predict research asset in rusty-engine. Relevant to WSP/Astra/NanoJev feasibility work, but separate from GGUF serve lanes.
- GSV canon reference: `S:/rust/GSV/docs/gsv/LOCAL_MODEL_ROLES_2026-11-21.md`.

## 5. Smoke verification (2026-11-21)

- **Inspect:** `./target/release/nanojev.exe inspect --checkpoint data/models/nanojev` → **322 tensors** ✅ (readable checkpoint).
- **Replay smoke (`--limit 2`):** attempted with `--cohort .../unified_hard_test.jsonl --predictions .../predictions_test.jsonl --limit 2`; process loaded large model (~1.7 GB RSS) and was long-running on this constrained box with no output yet; **terminated** to protect RAM/contention. Treat replay smoke as on-demand/manual on a suitable box, or run with small limit and patience.

## 6. Gates / caution

- MSYS2 bash; do not commit model weights.
- Prefer `--limit` for smoke replay; avoid long/heavy runs on constrained box without need.
- For rusty-engine code changes: `cargo xtask check` (fmt+clippy -D warnings+test) per `AGENTS.md`. Full model-eval runs are on-demand.
