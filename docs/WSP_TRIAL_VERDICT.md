# WSP verdict — MC-6 Alpha (Uk-golden gate, fact-2 class)

**Ticket:** MC-6 (`t-1794758436128993400`) · **Date:** 2026-11-15 · **Status:** **ROLLBACK verdict — 2/2 trials rejected**

## What was done (all artifacts pushed)

- Instruments: `rusty_engine::patch` (apply lane, `6e19d3b`), `rusty audit` activation capture (`47f66ad`), `rusty pack-mint` (+`--all`), `rusty serve --patch` staged-patch serve (`746db1a`)
- Uk-golden fixtures: `evals/uk_golden/fixtures.json` (`dbd0613`), 18 prompts, owner-accepted as canon, 4 critical

## Trials (outcome 2/2: ROLLBACK)

| # | Candidate (Inkling-audit-guided) | Route | Result |
|---|---|---|---|
| 01 | `blk.23.ffn_down.weight` full-zero | audit (l23 delta_rms_max 0.45 vs 0.33) | Uk generation collapses; reasoning survives; ROLLBACK |
| 02 | `blk.15.attn_output.weight` full-zero | audit (l15 delta_rms_max 0.269 vs 0.166) | Reasoning dies 43→33; uk-fact-1→"London"; ROLLBACK |

Base serve re-verified after each rollback (e.g. `17 + 26 = 43`).

## Verdict

The 1B Uk model's factual Uk-class failures (fact-2 «Кобзар→Pushkin») are **not isolatable by block-sparse weight-space edits** on this build. Every patch tried that moved Uk outputs **regressed the critical fixtures**, which is exactly what the Uk-golden gate exists to catch. Gate arbitrates; keep/rollback decisions recorded with evidence.

Two ablations with the largest audit signals both rejected — sparse stride probes would add data but are **not** expected to change this conclusion (flagged as optional, not recommended as a spend of remaining budget).

## Recommended reading

- Uk-fact quality on THIS 1B model = knowledge-capacity limitation, not an error that tolerates weights patching
- The WSP loop (candidate → apply → gate) is validated end-to-end; rollback outcomes are successful instances of the gate
- For Uk-fact quality, next direction is a bigger Uk-capable model (see MC-1 35B / llama-rs / 30B+ checkpoint) — candidate architecture decision, not a weight patch

## Honest scope

This verdict covers **weight-space sparse patches on MiniCPM-V 4.7-1B**; it does not exclude future WSP benefit on a different model size, different failure class (e.g., degeneration/repetition), or a different success metric defined by Owner.
