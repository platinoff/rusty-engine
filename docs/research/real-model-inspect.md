# Real-model validation — evidence (2026-10-14)

First **real production GGUF** run through `rusty inspect` (not synth). Proves the B1
container parser + `ggml_dtype` map on genuine bytes, and **caught a real off-by-K-types
mapping bug** now fixed.

## Model

- File: `data/models/qwen2.5-0.5b-q4_k_m.gguf` (gitignored — **not committed**)
- Source: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` @ HF, `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- Size: **491 400 032 bytes** (== HTTP `content-length`, download verified complete)
- SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`

## `rusty inspect` output (real file)

```
[GGUF v3]  291 tensors  26 kv  align 32
  architecture: qwen2
  output.weight        [896,151936] ggml=8  dtype=Some(Q8_0)
  token_embd.weight    [896,151936] ggml=6  dtype=None        (Q5_0, unmapped — raw kept)
  blk.0.attn_norm      [896]        ggml=0  dtype=Some(F32)
  blk.0.ffn_down       [4864,896]   ggml=14 dtype=Some(Q6_K)  (was the BUG: 14 mis-mapped)
  blk.0.ffn_gate/up    [896,4864]   ggml=6  dtype=None
  blk.0.attn_k.bias    [128]        ggml=0  dtype=Some(F32)
  ...
```

## Bug found + fixed

`ggml_dtype` mapped `14→Q4_K, 15→Q5_K, 16→Q6_K` — **wrong**. Canonical GGML_TYPE_* is
`Q4_K=12, Q5_K=13, Q6_K=14` (matches `quant.rs block_info` + `dequant_q{4,5,6}_k`, which
were already correct). Real file showed `ffn_down` as `ggml=14`; before the fix it reported
`Q4_K` (should be `Q6_K`). Fix: map 12/13/14; added regression test
`ggml_dtype_maps_canonical_ggml_codes`. Tests 64, clippy `-D` 0.

## Still deferred (honest scope)

Header/KV/tensor-info/dtype parsing = **validated on real bytes**. Full **token generation**
(`PH-305`) additionally needs: BPE decode (PH-105→algorithm), multi-layer loop
(attn + MoE/dense FFN stacked over all 24 blocks + RoPE-cache), vocab-sized logit projection,
and the **dequant of real tensors** via the loader `slice()` — a larger, separate slice.
