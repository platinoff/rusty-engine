# WSP apply lane — block-bitmap patch-at-dequant

**Ticket:** WSP-G1 · `t-1794500787890861700` · **ECO-9.2 sketch phase 9.3b** · **Date:** 2026-11-15
**Parent:** `GSV/docs/superpowers/plans/2026-11-08-poolai-wsp-feasibility.md` · `poolAI/docs/development/WSP_HYBRID_ARCHITECTURE_SKETCH.md`

## What this is

The engine-side apply primitive for **Weight-Space Programming** patches: a sparse,
block-level bitmap over a tensor's quantized blocks, applied to a *copy* of the raw
block bytes immediately before dequant. Untouched blocks stay byte-identical —
no requant noise, no kernel edits, zero overhead when no pack is applied.

## Guarantees

- Block granularity only: Q8_0 (32 el / 34 B) · Q4_K (256 / 144) · Q5_K (256 / 176) · Q6_K (256 / 210).
- `base_hash` (hex SHA-256 of the block being replaced) — patch applies only to the
  exact base model it was minted against (grid validator mirror: `validate_pack_against_file`).
- Strict, fail-closed: unknown tensor → error; duplicate (tensor, block_idx) → error;
  ragged tensor → error; wrong block length → error; hash drift → error.
- Instant rollback: drop the pack → base bytes (bitmap is audit surface, `AppliedBlockBitmap`).
- Router/gate weights: nothing in this module chooses which tensor to patch — the
  caller is responsible for the freeze-router/gate guard (ECO-9.2 guard 1).

## Non-goals (by design)

- No single-FP16 surgery (impossible under block quant).
- No kernel hot-path branching — the engine dequantizes once at load; the raw block
  bytes are the hook.
- No F32/F16/BF16/Q4_0/Q5_0 (not MiniCPM-V/llama-rs FFN targets).
- No JSON interchange / quorum / Solana — that is the poolAI grid lane (MC-6).

## API (module `rusty_engine::patch`)

```rust
PatchPack::new(id).with_entry(PatchEntry::new(tensor, block_idx, new_block_bits, base_hash))
dequant_tensor_with_patches(ggml_type, bytes, tensor, pack) -> Result<(Vec<f32>, AppliedBlockBitmap), PatchError>
dequant_with_patches(&GgufFile, name, pack)          // one tensor from an open container
validate_pack_against_file(&GgufFile, pack)          // fail-closed full-pack validation, applies nothing
patch_tensor_blocks(ggml_type, bytes, tensor, pack)  // raw buffer → patched buffer (copy)
sha256_hex(bytes)                                    // base_hash minting + verification
```

## Dequant hook equivalence (honest note)

"patch-at-dequant" here = patch the raw block bytes *before* the existing scalar
dequant oracle, i.e. mathematically identical to a block-granular rewrite of the
tensor followed by the unchanged dequant. There is no separate "patched kernel"
path to drift against the base numerics; the base dequant contracts are reused.

## Evidence

Unit tests in `src/patch.rs`: SHA-256 NIST known-answer vectors; Q8_0/Q4_K/Q6_K
roundtrips; base_hash drift rejection; ragged data; overlapping patch; empty pack
is byte- and f32-identical to base dequant. Run: `cargo test -p rusty-engine patch`.
