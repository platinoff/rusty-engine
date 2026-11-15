# WSP trial 02 — fact-2 («Кобзар→Pushkin») · layer-15 full-attn output zero

**Ticket:** MC-6 (`t-1794758436128993400`) · **Date:** 2026-11-15 · **Model:** MiniCPM-V-4_7-1B-Q4_K_M
**Status:** **ROLLBACK** — candidate rejected (clear regression vs base)
Field/candidate: `blk.15.attn_output.weight` zero-ablation (audit flip 0.269 vs 0.166 in fact-1 vs fact-2)

## Critical fixtures before → after

| id | before (base) | after (l15 attn_out zero) | verdict |
|---|---|---|---|
| uk-fact-1 | "Kyiv" ✅ | "London" loops ❌ | wrong + repetitive |
| uk-fact-2 | "Pushkin" ❌ | «пому»-цикли ❌ | not fixed, collapsed |
| uk-fact-3 | n/a | garbage ❌ | degraded |
| uk-reasoning-1 | 43 ✅ | **"33"** ❌ | arithmetic broken |
| uk-code-1 | works ✅ | scaffolds code ⚠️ | survives |

## Lessons

- Layer-15 full attention is a **critical path** (arithmetic regresses 43→33); it is not patchable
- Two ablations (l23 ffn_down, l15 attn_out) → two rollbacks; fact-2 is **not** isolatable by layer-level block edits
- Every probe so far says the failure is distributed and the only surviving behavior is code scaffold (and 43 was fragile too)
- Next honest candidate classes: (a) strided/sparse 1/8/stride block ablation of mid layers (layer 14/15/and output-proj), (b) token-level counterfactual re-rank idea (not a weight patch; invalid for WSP scope), (c) different failure class / different target analysis

## Note

Trial 01 + 02 both rollback — no keeps yet. Gate arbitrates everything; nothing kept on the board that regresses the critical fixtures.
