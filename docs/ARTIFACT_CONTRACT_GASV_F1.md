# GASV-F1 — partial shexp transplant artifact contract

**Status:** implemented 2026-11-27 · **Gate:** honesty fields are unit-tested in
`src/bin/rusty.rs` (`artifact_contract_is_partial_not_runnable`,
`artifact_contract_honesty_fields_are_constants`).

A partial shared-expert transplant pack is **not a runnable LLM**. The contract
is emitted by `rusty artifact-report <file.gguf> [flags]` as JSON on stdout:

```json
{
  "schema": "gasv.artifact-manifest.v1",
  "kind": "partial-shexp-pack",
  "runnable": false,
  "artifact": "<path>",
  "arch": "gsv",
  "seed": 0,
  "context_length": 0,
  "tensor_count": 0,
  "bytes": 0,
  "sha256": "<sha256 of the artifact file>",
  "ggml_types": ["f32"],
  "source": "tensors.json+chunks",
  "coverage": "blk.0..4 shexp-only",
  "transplant_report_sha256": "<sha256 of the transplant report>",
  "nanojev_role": "eval-oracle-only",
  "no_quality_parity_claim": true
}
```

CLI flags: `--kind K` (default `partial-shexp-pack`),
`--runnable true|false` (default `false`), `--source S`, `--coverage C`,
`--transplant-report-sha256 H`.

## Honesty rules (never relax without owner go)

- `runnable` defaults to `false` — only an explicit `--runnable true` for a
  complete model may flip it.
- `nanojev_role` is always `eval-oracle-only`: NanoJev-derived artifacts are
  evaluation oracles, not answer-producing models.
- `no_quality_parity_claim` is always `true`: the report explicitly denies an
  image→answer / quality-parity claim while G13 is blocked.
- No weights are stored in git. The report is a small JSON summary; the
  heavy artifact stays on the mirror/temp evidence path.

`build_artifact_report()` in `src/bin/rusty.rs` is the pure schema builder; the
CLI wrapper only reads GGUF metadata + sha256 and prints it. GASV-F2 (Q8_0/Q4_0
export evaluation) remains a separate ticket; Q4_K_M stays parity-gated.
