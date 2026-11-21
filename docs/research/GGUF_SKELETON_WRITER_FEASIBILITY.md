# GASV-A2 — GGUF writer/skeleton feasibility + API sketch

**Date:** 2026-11-21  
**Ticket:** `t-1795270403090130500` (rusty-engine, in_progress)  
**Scope:** research/docs only. No GGUF writes, no weights.  
**Plan:** `S:\rust\GSV\docs\superpowers/plans/2026-11-21-gasv-versus-ai-wgpu-cluster-skeleton.md`

## 1. Current state (rusty-engine)

| Component | Status | Evidence |
|---|---|---|
| GGUF reader | **exists, read-only** | `src/gguf.rs` (562 LOC): magic/ver/KV/tensor-info parse, `data_offset`, `alignment()`, `ggml_dtype`, `TensorEntry::nbytes`; std-only, no mmap tensor data |
| GGUF writer | **missing** | grep `write_gguf/GgufWriter/save_gguf/serialize_gguf` = 0 |
| Patch/copy splice | **exists** | `src/patch.rs` (1128 LOC): `parse_wsp`, block patch validation, `stage_patch` copies existing GGUF and splices regions; SHA-256 std-only; supports Q8_0/Q4_K/Q5_K/Q6_K block replacement |
| DType/block info | **exists** | `src/core.rs` DType; `src/quant.rs` block_info/bytes_for_numel/dequant for F32/F16/BF16/Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K |
| Model config/validation | **exists** | `src/model.rs` ModelConfig from GGUF KV, validate tensor names/dims; qwen2/qwen35/qwen35moe paths |
| Loader/mmap/dequant | **exists** | `src/loader.rs` MmapModel, tensor_bytes/slice/dequant/staged_read |
| Random init | **missing** | no init_weights/randn/xavier/kaiming API |
| Quant encoder | **missing** | dequant exists; no f32->Q4_K/Q8_0 GGUF encoder |

## 2. GGUF writer requirements (from-scratch)

A minimal GGUF v3 writer must emit, in order:

1. **Header:** `GGUF` magic, `version:u32` (3), `n_tensors:u64`, `n_kv:u64`.
2. **KV metadata:** for each `(key, value_kind, value)` — key string (u64 len + bytes), kind u32, encoded value; arrays recursive with elem kind + count.
3. **Tensor info table:** for each tensor — name string, `n_dims:u32`, dims as `u64[n_dims]` (GGUF order as in file), `ggml_type:u32`, `offset:u64` (relative to tensor data start).
4. **Alignment padding:** pad to `general.alignment` (default 32) before tensor data blob.
5. **Tensor data blob:** tensor bytes at declared offsets; each tensor region aligned/padded as needed.

### Must-match invariants
- `offset` is relative to aligned tensor data start, not file start.
- Tensor data layout/endianness matches GGUF/ggml expectations.
- `data_offset` (reader-computed) == aligned position after tensor info table.
- String bytes are UTF-8; counts/sizes checked for overflow.
- Unknown/unsupported ggml types: writer should allow explicit raw bytes only when caller declares type+size truthfully.

## 3. Feasibility assessment

| Capability | Feasibility | Risk | Notes |
|---|---|---|---|
| GGUF v3 writer (F32/F16/BF16 raw tensors) | **High** | Low | Straightforward byte serialization; mirror existing parser structs |
| KV writer for scalar/string/array | **High** | Low | `ValueKind` already modeled in reader |
| Deterministic random-init skeleton (F32) | **High** | Med | Std-only seeded PRNG (xorshift64*/SplitMix) for reproducibility; seed in manifest; **not crypto** |
| Tensor shape/name schema from ModelConfig/arch | **Medium** | Med | Need arch-specific tensor naming map for qwen35/qwen35moe/qwen2; validate with existing `ModelConfig::validate` expectations |
| F32 -> Q8_0 quant encoder | **Medium** | Med | Block formula known; must match dequant oracle and llama.cpp layout |
| F32 -> Q4_K/Q5_K/Q6_K quant encoder | **Med/Low** | High | K-quant packing is fiddly; easy to be "almost right" but incompatible; need golden parity against dequant/reference |
| Transplant compatible GGUF tensor bytes as-is | **High** | Med | Copy raw tensor bytes only when same arch/tensor/type/shape/quant semantics proven |
| Transplant with dtype/quant conversion | **Medium** | High | Requires dequant->transform->requant; quality/compat risk |
| Transplant across incompatible arch/shapes | **No (silent)** | Critical | Must be rejected; only explicit research projection lane with owner gate |

## 4. Proposed API sketch (rusty-engine)

```rust
// src/gguf_write.rs (new, std-only)
pub struct GgufWriter {
    version: u32,              // 3
    alignment: u64,            // default 32
    metadata: Vec<(String, Value)>,
    tensors: Vec<TensorPlan>,
}

pub struct TensorPlan {
    pub name: String,
    pub dims: Vec<usize>,      // gguf dims
    pub ggml_type: u32,
    pub bytes: TensorBytes,    // Raw(Vec<u8>) | F32(Vec<f32>) | Zeroed(len) | Seeded{..}
}

pub enum TensorBytes {
    Raw(Vec<u8>),
    F32(Vec<f32>),
}

impl GgufWriter {
    pub fn new_v3() -> Self;
    pub fn alignment(mut self, a: u64) -> Self;
    pub fn kv(mut self, key: impl Into<String>, value: Value) -> Self;
    pub fn tensor(mut self, plan: TensorPlan) -> GgufResult<Self>;
    pub fn write_to_path(&self, path: &Path) -> GgufResult<WriteReport>;
}

pub struct WriteReport {
    pub path: PathBuf,
    pub bytes_written: u64,
    pub sha256: String,
    pub n_tensors: usize,
    pub n_kv: usize,
}

// src/skeleton.rs (new, depends on gguf_write + model schema)
pub struct SkeletonSpec {
    pub arch: String,          // e.g. qwen35 / qwen35moe
    pub seed: u64,
    pub dtype: SkeletonDType,  // F32 first, then Q8_0/Q4_K
    pub context_length: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub d_ff: u32,
    pub n_experts: Option<u32>,
    pub n_experts_used: Option<u32>,
    pub vocab_size: u32,
    pub extra_kv: Vec<(String, Value)>,
}

pub fn build_skeleton(spec: &SkeletonSpec) -> GgufResult<GgufWriter>;
```

Notes:
- Keep writer std-only, no serde dependency required for initial GGUF binary.
- Reuse `gguf::{Value, ValueKind, ggml_dtype, DEFAULT_ALIGNMENT}` where possible; avoid duplicating parser types.
- Deterministic PRNG: std-only `SplitMix64`/xorshift for skeleton bytes; record seed+algo in KV/manifest.
- `TensorBytes::Raw` enables compatible transplant; `F32` enables random init / conversion staging.

## 5. Validation plan

1. **Round-trip read:** write GGUF -> `gguf::parse_file` reads same header/KV/tensor table.
2. **Byte invariants:** tensor offsets aligned; `nbytes()` matches actual data span.
3. **Model validate:** for supported arch, `ModelConfig::validate` passes.
4. **Inspect:** `rusty inspect` reports expected tensors/metadata.
5. **Smoke infer:** tiny skeleton can load/dequant where dtype supported (F32/F16 first).
6. **Golden parity for quant encoders:** dequant(quant(f32)) error within dtype tolerance; compare against known-good vectors before Q4_K export.

## 6. GASV pipeline implications

- **GASV-C1:** implement `GgufWriter` + round-trip tests.
- **GASV-C2:** deterministic random-init skeleton from `SkeletonSpec`; no claims of model quality.
- **GASV-D1:** compatible tensor transplant using `TensorBytes::Raw` with manifest + checksum.
- **GASV-D3:** Q4_K_M export only after quant encoder parity gate.
- **Artifact:** `gasv-versus-ai.gguf` emitted outside git (`L:\gsv_mirror\models\gasv-versus-ai\` or owner-approved path) with SHA256 + manifest.

## 7. Risks / non-goals

- No silent shape coercion or dtype guessing.
- No K-quant encoder without golden parity.
- No claim that random-init skeleton is useful before eval.
- NanoJev safetensors are not GGUF source tensors; do not route through GGUF writer without explicit conversion contract.
- `patch.rs` remains useful for existing GGUF block patching; it is not a substitute for full GGUF writer.

## 8. Verdict

**Feasible in staged Rust work.** Highest-confidence first slice: **GGUF v3 writer for raw/F32 tensors + round-trip tests**. K-quant encoding and cross-model transplant require separate gated tickets.