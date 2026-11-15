# WSP trial 01 — fact-2 («Кобзар→Pushkin») · layer-23 FFN full ablation

**Ticket:** MC-6 (`t-1794758436128993400`) · **Date:** 2026-11-15 · **Model:** MiniCPM-V-4_7-1B-Q4_K_M (Q6_K ffn_down)
**Status:** **ROLLBACK** — candidate rejected (fixtures clearly worse; result is an ablation probe, not a patch)

## Setup

- Audit (pre-trial, `rusty audit`): fact-1 vs fact-2 — per-layer stats nearly identical; final full-attention layer 23 `delta_rms_max`: fact-2 0.4509 vs fact-1 0.3254 (only material difference)
- Pack: `rusty pack-mint --all --zero` on `blk.23.ffn_down.weight` → 14 336 blocks × 210 B = **all-zero replaceable bits** (valid base_hash each block) — full zero of layer-23 FFN down projection
- Apply: `rusty serve --patch pack_l23_ffndn_zero_all.wsp` (staged `patched_11952.gguf`, in-memory `MmapModel`, base GGUF untouched)

## Critical fixtures before → after

| id | before (base) | after (layer-23 FFN zero) | verdict |
|---|---|---|---|
| uk-fact-1 | "Kyiv" ✅ | "The answer is 1" loops ❌ | degraded |
| uk-fact-2 | "Pushkin" ❌ (known failure) | «Кобза»-цикли ❌ | not fixed, collapsed |
| uk-fact-3 | (unknown-ish) | "unknown" loops ❌ | degraded |
| uk-reasoning-1 | 43 ✅ | **"Answer 43. 43." ✅** | survivor |
| uk-code-1 | works ✅ | scaffolded, restyled ⚠️ | survives |

## Lessons

- Layer-23 FFN is **not** a patch target; full ablation is strictly worse (Ukrainian generation collapses first — consistent with Uk-class being the most damage-prone cluster)
- Factual arithmetic (43) and broad code scaffolding **survive** the layer — the model is not uniformly computing through this matrix
- A single FFN-down layer is not the carrier of «Кобзар→Pushkin»; next candidates: (a) sparse/strided block ablation of layer 23 or 15 (audit-guided), (b) layer-15 full-attention (audit flip 0.269→0.166), (c) rescale-instead-of-zero probes
- Uk-golden critical fixtures are the gate arbiter — one "fix" on one fixture never enough; overall health decides keep/rollback

## Instruments

`rusty pack-mint -m <model> -t <tensor> [--zero] (-b <idx> | --all)` — emits one wsp line per selected block with true base_hash. `rusty serve --patch <wsp>` stages a patched copy and serves it. Base GGUF is never modified; rollback = restart without `--patch`.
