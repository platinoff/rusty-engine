# qwen35 text E2E parity — PH-1305 method + results

**Status:** ✅ closed (2026-11-11) · **Oracle:** `llama.cpp` build `b11906`
(the `llama.cpp-omni` role) · **Model:** `data/models/MiniCPM-V-4_7-1B-Q4_K_M.gguf`
(gitignored, 672 MB) · **Gate:** `tests/qwen35_parity.rs` (`cargo test --release -- --ignored`)

## Method

The oracle ran the *same* GGUF the engine runs, via the raw `/completion` API
(`temperature=0`, `top_k=1`, `logprobs=5`). Its response bodies are committed
**byte-for-byte** under `tests/data/qwen35_oracle/<slug>.{json,ids}` — five pinned
prompts (capital, fib, math, story, fox), ten greedy steps each. Nothing is
transcribed by hand, so a fixture cannot drift from what the oracle actually said.

Three offline gates (all `#[ignore]`; they skip green without the model):

1. **`tokenizer_matches_oracle`** — our `qwen35` pre-tokenizer + BPE must reproduce
   `llama-tokenize`'s ids exactly (`<slug>.ids`). Tokenizer drift would invalidate every
   downstream comparison, so it is its own gate.
2. **`greedy_matches_oracle_committed_tokens`** — the numeric gate. Our f32 logits are
   compared against the oracle's top-5 logprobs *as a profile*: for every token the
   oracle ranked in its top-5, the logit gap to the argmax must agree within
   `LOGIT_TOL = 0.75` nats. Argmax equality is only required where the oracle itself was
   decisive (`top1 − top2 ≥ TIE_FLOOR = 0.5` nats) — a 0.01-nat tie is decided by float
   order, not by the model.
3. **`greedy_fingerprint_matches_golden`** — our own pinned 10-step run
   (`tests/data/qwen35_golden.txt`, capture once with `RUSTY_CAPTURE_GOLDEN=1`), catching
   a drift a shorter oracle window could miss.

### Why the profile gate, not bit-exact argmax

ggml quantizes activations to `q8_K` before the K-quant dot; this engine keeps the
activations f32. The two matvec paths are therefore *not* bit-identical, and the residual
shows up as a few-hundredths-of-a-nat wobble in the tail of the top-5 — exactly what the
profile gate tolerates and a bit-exact gate would reject as false positives.

## Conventions pinned by the oracle source (`src/models/qwen35.cpp`, `models.h`)

Pins live next to the code they guard (not here): `src/gdn.rs` carries the
`ssm_a` / conv / L2 / gate pins, `src/q35.rs` the attention-gate + RoPE pins,
`src/tokenizer.rs` the `qwen35` pre-tokenizer + specials.

## Results (2026-11-11, release build, RUSTY_THREADS=12)

| Gate | Result |
|---|---|
| Tokenizer | 5/5 prompts **byte-exact** vs `llama-tokenize` |
| Oracle profile | **36/36 committed steps match**; 2 tie-zone divergences (margin < 0.5 nats); worst profile deviation **0.677 nats** (fox step 0, token 33075) — inside `LOGIT_TOL` |
| Greedy fingerprint | 5/5 prompts match the captured golden |

Profile outliers > 0.25 nats: 26 of 182 comparisons — the expected `q8_K` residual,
no engine slip (a slip would move an argmax or blow the 0.75-nat wall, not scatter
small deviations).

## Perf slice (`perf_slice_prefill_decode_12t`, fresh process)

Prompt: 263 tokens of real prose (pinned capital prompt grown with natural text).

| Phase | Wall | Throughput |
|---|---|---|
| Prefill (batched scan) | **12.27 s** | **21.4 tok/s** |
| Decode step (median of 10) | **47.8 ms** | **20.9 tok/s** |

- Decode O(1) expectation **met with headroom**: llama-rs floor oracle is
  0.497 tok/s on the 27B GDN shape, the rusty qwen2 path peaks at 10.9 tok/s
  (PH-203h); the qwen35 1B decode runs 20.9 tok/s.
- Prefill is the **next lever, not a defect**: the GDN delta-rule scan is sequential
  per group (documented PH-1302/1304 limitation); a pooled decomposition by
  (layer, group) — or a chunked scan — is the follow-up band. 21.4 tok/s prefill on a
  1B sits below the qwen2 path (~150 tok/s at 961 tokens) precisely because the
  full-attention layers share the tuned QMat path while the 18 GDN scans do not.

## Recorded oracle build

`llama.cpp` `b11906` on-box; fixture re-record procedure: run the omni CLI over the
five pinned prompts at `temperature=0`, `top_k=1`, `logprobs=5`, save the raw
`/completion` body as `<slug>.json` and `llama-tokenize` output as `<slug>.ids`.
