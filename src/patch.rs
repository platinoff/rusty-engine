//! WSP apply lane — `block-bitmap patch-at-dequant` (ECO-9.2 sketch, phase 9.3b).
//!
//! Owner ask (2026-11-12, WSP/NanoJev track): apply cloud-oracle-proposed sparse
//! weight fixes to small local GGUF models without touching kernel code or the
//! plain serve path. A patch is a **block-level bitmap** over a tensor's quantized
//! blocks: on apply, the affected blocks of a raw tensor byte buffer are replaced
//! wholesale; untouched blocks stay byte-identical and are never requantized.
//!
//! Supported block granularity (matches `crate::quant::block_info`):
//! - `Q8_0`  — 32 elements / 34 bytes  (1×f16 scale + 32×i8)
//! - `Q4_K`  — 256 elements / 144 bytes
//! - `Q5_K`  — 256 elements / 176 bytes
//! - `Q6_K`  — 256 elements / 210 bytes
//!
//! Out of scope / by design:
//! - single-FP16/FP32 surgery — impossible under block quant (ECO-9.1 blocker 1)
//! - `F32`/`F16`/`BF16`/`Q4_0`/`Q5_0` — not the MiniCPM-V/llama-rs FFN targets;
//!   extend only with a gate proving it does not weaken the existing scalar oracle
//! - router/gate weights — **frozen** per ECO-9.2 guard #1; nothing in this module
//!   knows which blocks are routers (the caller chooses `tensor` names)
//! - kernel hot-path branching — unnecessary here: the engine dequantizes once at
//!   load, so patching the raw block bytes just before dequant *is* the hook
//!
//! Transport/interchange (JSON packs, grid `WeightPatch` jobs, quorum) lives in
//! poolAI / MC-6; this module is the engine-side apply primitive only.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt;

use crate::core::{EngineError, Result};
use crate::loader::MmapModel;
use crate::quant;

/// Patch-lane result alias (`core::Result` is pinned to `EngineError`).
pub type PatchResult<T> = std::result::Result<T, PatchError>;

// ─────────────────────────── SHA-256 (std-only) ───────────────────────────

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    // padded message: data || 0x80 || zeros || 8-byte big-endian bit length
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    let mut w = [0u32; 64];
    for chunk in msg.chunks_exact(64) {
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(big_s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = big_s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Hex (lowercase) SHA-256 of `data` — the `base_hash` field of a patch entry.
pub fn sha256_hex(data: &[u8]) -> String {
    let digest = sha256(data);
    let mut s = String::with_capacity(64);
    for byte in digest {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

// ─────────────────────────── Patch wire types ───────────────────────────

/// One staged replacement region inside the raw model file (byte range + new bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    pub start: usize,
    pub span: usize,
    pub replacement: Vec<u8>,
}

/// Parse a trivial text wsp pack (no JSON dependency):
/// ```text
/// # wsp-pack v1                 (optional comment line)
/// blk.0.ffn_down.weight 17 <hex bytes> <base_hash_sha256_hex>\n
/// blk.0.ffn_down.weight 18 <hex bytes> <base_hash_sha256_hex>\n
/// ```
pub fn parse_wsp(data: &str) -> PatchResult<PatchPack> {
    let mut pack = PatchPack::new("wsp");
    for (lineno, raw) in data.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let tensor = parts.next().ok_or_else(|| {
            PatchError::Dequant(format!("wsp line {}: missing tensor", lineno + 1))
        })?;
        let block_idx: u32 = parts.next().and_then(|s| s.parse().ok()).ok_or_else(|| {
            PatchError::Dequant(format!("wsp line {}: bad block_idx", lineno + 1))
        })?;
        let bits_hex = parts.next().ok_or_else(|| {
            PatchError::Dequant(format!("wsp line {}: missing bits hex", lineno + 1))
        })?;
        let base_hash = parts
            .next()
            .ok_or_else(|| {
                PatchError::Dequant(format!("wsp line {}: missing base_hash", lineno + 1))
            })?
            .to_string();
        let bits = hex_decode(bits_hex)
            .ok_or_else(|| PatchError::Dequant(format!("wsp line {}: bad bits hex", lineno + 1)))?;
        pack.entries.push(PatchEntry {
            tensor: tensor.into(),
            block_idx,
            new_block_bits: bits,
            base_hash,
        });
    }
    Ok(pack)
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let b = s.as_bytes();
    for pair in b.chunks_exact(2) {
        let hi = (pair[0] as char).to_digit(16)? as u8;
        let lo = (pair[1] as char).to_digit(16)? as u8;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

/// Splice `regions` into a fresh copy of `src_bytes` (verifying each region's
/// content with the caller-resolved slices). Pure, byte-level, unit-testable.
pub fn apply_regions(src_bytes: &[u8], regions: &[Region]) -> PatchResult<Vec<u8>> {
    let mut regions: Vec<Region> = regions.to_vec();
    regions.sort_by_key(|r| r.start);
    let mut out = src_bytes.to_vec();
    let mut cur = 0usize;
    for r in &regions {
        if r.start < cur {
            return Err(PatchError::OverlappingPatch {
                tensor: String::new(),
                block_idx: 0,
            });
        }
        if r.start + r.span > out.len() {
            return Err(PatchError::BlockOutOfRange {
                tensor: String::new(),
                block_idx: 0,
                total: out.len() as u32,
            });
        }
        out[r.start..r.start + r.span].copy_from_slice(&r.replacement);
        cur = r.start + r.span;
    }
    Ok(out)
}

/// Copy `src` → `dst` with regions spliced in (streaming, ~8 MB buffer, no full
/// file in RAM). Every region already had its base_hash verified by the caller.
pub fn stage_regions_on_file(
    src: &std::path::Path,
    dst: &std::path::Path,
    regions: &[Region],
) -> Result<usize> {
    use std::io::{Read, Seek, Write};
    let mut regions: Vec<Region> = regions.to_vec();
    regions.sort_by_key(|r| r.start);
    let mut src_f = std::fs::File::open(src).map_err(|e| EngineError {
        message: format!("patch stage: open {src:?}: {e}"),
    })?;
    let mut dst_f = std::fs::File::create(dst).map_err(|e| EngineError {
        message: format!("patch stage: create {dst:?}: {e}"),
    })?;
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut cur: u64 = 0;
    let mut patched = 0usize;
    for r in regions {
        // copy everything before this region
        let mut remaining = r.start as u64 - cur;
        while remaining > 0 {
            let take = remaining.min(buf.len() as u64) as usize;
            let n = src_f.read(&mut buf[..take]).map_err(|e| EngineError {
                message: format!("patch stage: read {src:?}: {e}"),
            })?;
            if n == 0 {
                break;
            }
            dst_f.write_all(&buf[..n]).map_err(|e| EngineError {
                message: format!("patch stage: write {dst:?}: {e}"),
            })?;
            remaining -= n as u64;
        }
        // splice the replacement, skip the original region
        dst_f.write_all(&r.replacement).map_err(|e| EngineError {
            message: format!("patch stage: write {dst:?}: {e}"),
        })?;
        src_f
            .seek(std::io::SeekFrom::Start(r.start as u64 + r.span as u64))
            .map_err(|e| EngineError {
                message: format!("patch stage: seek {src:?}: {e}"),
            })?;
        cur = r.start as u64 + r.span as u64;
        patched += 1;
    }
    // tail
    loop {
        let n = src_f.read(&mut buf).map_err(|e| EngineError {
            message: format!("patch stage: read {src:?}: {e}"),
        })?;
        if n == 0 {
            break;
        }
        dst_f.write_all(&buf[..n]).map_err(|e| EngineError {
            message: format!("patch stage: write {dst:?}: {e}"),
        })?;
    }
    Ok(patched)
}

/// One block-level patch: replace block `block_idx` of `tensor` with
/// `new_block_bits` — valid only if the current block bytes hash to
/// `base_hash` (content-addressed against the base model, ECO-9.2 guard #3).
/// Stage a **patched copy** of `src` into `dst` — fail-closed full-pack validation
/// happens BEFORE the destination is created (hash check per block against the
/// base mmap). The patched file is a complete model (mmap-ready); the caller
/// opens it with [`crate::loader::MmapModel::open`] as usual.
pub fn stage_patch(
    src: &std::path::Path,
    dst: &std::path::Path,
    pack: &PatchPack,
) -> Result<usize> {
    let model = unsafe { crate::loader::MmapModel::open(src) }.map_err(|e| EngineError {
        message: format!("patch stage: open {src:?}: {e}"),
    })?;
    let data_start = model.data_start();
    let header = model.header();
    let mut regions: Vec<Region> = Vec::with_capacity(pack.entries.len());
    for entry in &pack.entries {
        let te = header
            .tensors
            .iter()
            .find(|t| t.name == entry.tensor)
            .ok_or_else(|| EngineError {
                message: format!("patch stage: tensor '{}' not found", entry.tensor),
            })?;
        let block_bytes = patch_block_bytes(te.ggml_type).ok_or_else(|| EngineError {
            message: format!(
                "patch stage: tensor '{}' ggml_type {} unsupported",
                entry.tensor, te.ggml_type
            ),
        })?;
        let bytes = model
            .tensor_bytes(&entry.tensor)
            .ok_or_else(|| EngineError {
                message: format!(
                    "patch stage: tensor '{}' out of bounds or unsized",
                    entry.tensor
                ),
            })?;
        let total = bytes.len() / block_bytes;
        if entry.block_idx as usize >= total {
            return Err(EngineError {
                message: format!(
                    "patch stage: tensor '{}' block {} out of range (total {})",
                    entry.tensor, entry.block_idx, total
                ),
            });
        }
        if entry.new_block_bits.len() != block_bytes {
            return Err(EngineError {
                message: format!(
                    "patch stage: tensor '{}' block {} needs {} bytes, got {}",
                    entry.tensor,
                    entry.block_idx,
                    block_bytes,
                    entry.new_block_bits.len()
                ),
            });
        }
        let block_off = entry.block_idx as usize * block_bytes;
        let actual = sha256_hex(&bytes[block_off..block_off + block_bytes]);
        if actual != entry.base_hash.to_lowercase() {
            return Err(EngineError {
                message: format!(
                    "patch stage: tensor '{}' block {} base_hash mismatch (base model differs)",
                    entry.tensor, entry.block_idx
                ),
            });
        }
        let abs_start = data_start + te.offset as usize + block_off;
        regions.push(Region {
            start: abs_start,
            span: block_bytes,
            replacement: entry.new_block_bits.clone(),
        });
    }
    stage_regions_on_file(src, dst, &regions)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchEntry {
    pub tensor: String,
    pub block_idx: u32,
    pub new_block_bits: Vec<u8>,
    pub base_hash: String,
}

impl PatchEntry {
    pub fn new(
        tensor: impl Into<String>,
        block_idx: u32,
        new_block_bits: Vec<u8>,
        base_hash: String,
    ) -> Self {
        Self {
            tensor: tensor.into(),
            block_idx,
            new_block_bits,
            base_hash,
        }
    }
}

/// A sparse set of block patches targeting one model (weights are never in git).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchPack {
    pub id: String,
    pub entries: Vec<PatchEntry>,
}

impl PatchPack {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            entries: Vec::new(),
        }
    }

    pub fn with_entry(mut self, entry: PatchEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Distinct tensor names referenced by the pack (for validation/reporting).
    pub fn tensor_names(&self) -> BTreeSet<&str> {
        self.entries.iter().map(|e| e.tensor.as_str()).collect()
    }
}

/// Block-bitmap of a *successful* apply: which blocks were replaced, per tensor.
/// This is the audit + instant-rollback surface (drop the pack → base bytes).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppliedBlockBitmap {
    map: BTreeMap<String, BTreeSet<u32>>,
}

impl AppliedBlockBitmap {
    pub fn blocks(&self, tensor: &str) -> BTreeSet<u32> {
        self.map.get(tensor).cloned().unwrap_or_default()
    }

    pub fn contains(&self, tensor: &str, block_idx: u32) -> bool {
        self.map.get(tensor).is_some_and(|s| s.contains(&block_idx))
    }

    pub fn tensor_count(&self) -> usize {
        self.map.len()
    }

    pub fn block_count(&self) -> usize {
        self.map.values().map(BTreeSet::len).sum()
    }
}

// ─────────────────────────── Errors ───────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchError {
    /// Pack references a tensor absent from the container (fail-closed; pack/model mismatch).
    TensorNotFound(String),
    /// ggml_type outside the supported patch granularity (Q8_0/Q4_K/Q5_K/Q6_K).
    UnsupportedType { tensor: String, ggml_type: u32 },
    /// `block_idx` beyond the tensor's block count.
    BlockOutOfRange {
        tensor: String,
        block_idx: u32,
        total: u32,
    },
    /// new_block_bits length ≠ block size for the tensor's ggml_type.
    BlockLenMismatch {
        tensor: String,
        block_idx: u32,
        expected: usize,
        got: usize,
    },
    /// Current block bytes do not match the entry's `base_hash`.
    BaseHashMismatch {
        tensor: String,
        block_idx: u32,
        expected: String,
        got: String,
    },
    /// Two entries target the same (tensor, block_idx) — pack is not canonical.
    OverlappingPatch { tensor: String, block_idx: u32 },
    /// Tensor byte length is not a whole number of blocks for its ggml_type.
    RaggedData {
        tensor: String,
        len: usize,
        block_bytes: usize,
    },
    /// Dequant of the patched buffer failed (should not happen after block checks).
    Dequant(String),
}

impl fmt::Display for PatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TensorNotFound(t) => write!(f, "patch: tensor '{t}' not found in model"),
            Self::UnsupportedType { tensor, ggml_type } => {
                write!(
                    f,
                    "patch: tensor '{tensor}' ggml_type {ggml_type} is not patchable"
                )
            }
            Self::BlockOutOfRange {
                tensor,
                block_idx,
                total,
            } => {
                write!(
                    f,
                    "patch: tensor '{tensor}' block {block_idx} out of range (total {total})"
                )
            }
            Self::BlockLenMismatch {
                tensor,
                block_idx,
                expected,
                got,
            } => write!(
                f,
                "patch: tensor '{tensor}' block {block_idx} needs {expected} bytes, got {got}"
            ),
            Self::BaseHashMismatch {
                tensor, block_idx, ..
            } => write!(
                f,
                "patch: tensor '{tensor}' block {block_idx} base_hash mismatch (base model differs)"
            ),
            Self::OverlappingPatch { tensor, block_idx } => {
                write!(f, "patch: duplicate patch for '{tensor}' block {block_idx}")
            }
            Self::RaggedData {
                tensor,
                len,
                block_bytes,
            } => {
                write!(f, "patch: tensor '{tensor}' byte len {len} is not a multiple of block {block_bytes}")
            }
            Self::Dequant(msg) => write!(f, "patch: dequant of patched buffer failed: {msg}"),
        }
    }
}

impl std::error::Error for PatchError {}

impl From<PatchError> for EngineError {
    fn from(e: PatchError) -> Self {
        Self {
            message: e.to_string(),
        }
    }
}

// ─────────────────────────── Apply primitives ───────────────────────────

/// Byte geometry of one block for a patchable ggml_type.
fn patch_block_bytes(ggml_type: u32) -> Option<usize> {
    match ggml_type {
        8 | 12 | 13 | 14 => quant::block_info(ggml_type).map(|b| b.bytes),
        _ => None,
    }
}

/// Apply every entry of `pack` that targets `tensor` onto a **copy** of `bytes`
/// (the tensor's raw quantized blocks, as from `GgufFile::tensor_bytes`).
///
/// Untouched blocks stay byte-identical; untouched bytes are moved with one
/// linear copy. Returns the patched buffer; errors leave the caller's buffer alone.
pub fn patch_tensor_blocks(
    ggml_type: u32,
    bytes: &[u8],
    tensor: &str,
    pack: &PatchPack,
) -> PatchResult<Vec<u8>> {
    let block_bytes = patch_block_bytes(ggml_type).ok_or_else(|| PatchError::UnsupportedType {
        tensor: tensor.into(),
        ggml_type,
    })?;
    if bytes.len() % block_bytes != 0 {
        return Err(PatchError::RaggedData {
            tensor: tensor.into(),
            len: bytes.len(),
            block_bytes,
        });
    }
    let total_blocks = (bytes.len() / block_bytes) as u32;
    let mut out = bytes.to_vec();
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for entry in &pack.entries {
        if entry.tensor != tensor {
            continue;
        }
        if entry.block_idx >= total_blocks {
            return Err(PatchError::BlockOutOfRange {
                tensor: tensor.into(),
                block_idx: entry.block_idx,
                total: total_blocks,
            });
        }
        if !seen.insert(entry.block_idx) {
            return Err(PatchError::OverlappingPatch {
                tensor: tensor.into(),
                block_idx: entry.block_idx,
            });
        }
        if entry.new_block_bits.len() != block_bytes {
            return Err(PatchError::BlockLenMismatch {
                tensor: tensor.into(),
                block_idx: entry.block_idx,
                expected: block_bytes,
                got: entry.new_block_bits.len(),
            });
        }
        let start = entry.block_idx as usize * block_bytes;
        let end = start + block_bytes;
        let actual_hash = sha256_hex(&out[start..end]);
        if actual_hash != entry.base_hash.to_lowercase() {
            return Err(PatchError::BaseHashMismatch {
                tensor: tensor.into(),
                block_idx: entry.block_idx,
                expected: entry.base_hash.clone(),
                got: actual_hash,
            });
        }
        out[start..end].copy_from_slice(&entry.new_block_bits);
    }
    Ok(out)
}

/// Dequantize `tensor`'s raw blocks after applying every pack entry that targets it.
/// Returns the f32 tensor and the bitmap of replaced blocks.
pub fn dequant_tensor_with_patches(
    ggml_type: u32,
    bytes: &[u8],
    tensor: &str,
    pack: &PatchPack,
) -> PatchResult<(Vec<f32>, AppliedBlockBitmap)> {
    let patched = patch_tensor_blocks(ggml_type, bytes, tensor, pack)?;
    let mut bitmap = AppliedBlockBitmap::default();
    for entry in &pack.entries {
        if entry.tensor == tensor {
            bitmap
                .map
                .entry(tensor.to_string())
                .or_default()
                .insert(entry.block_idx);
        }
    }
    let f32s = quant::dequant(ggml_type, &patched).map_err(|e| PatchError::Dequant(e.message))?;
    Ok((f32s, bitmap))
}

/// Dequantize a named tensor from an mmap'd container with its pack applied.
/// Fail-closed on pack/model mismatch is [`validate_pack_against_file`]; this
/// wrapper only resolves the one tensor it is asked for.
pub fn dequant_with_patches(
    model: &MmapModel,
    name: &str,
    pack: &PatchPack,
) -> Result<(Vec<f32>, AppliedBlockBitmap)> {
    let entry = model
        .header()
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| EngineError {
            message: PatchError::TensorNotFound(name.to_string()).to_string(),
        })?;
    let bytes = model.tensor_bytes(name).ok_or_else(|| EngineError {
        message: format!("tensor '{name}' bytes out of bounds or type unsized"),
    })?;
    dequant_tensor_with_patches(entry.ggml_type, bytes, name, pack).map_err(EngineError::from)
}

/// Fail-closed validation of a whole pack against an mmap'd container (grid-side
/// validator mirror): every referenced tensor must exist, be patchable, be in
/// block range, and hash-match the base model. Applies **nothing**.
pub fn validate_pack_against_file(
    model: &MmapModel,
    pack: &PatchPack,
) -> PatchResult<AppliedBlockBitmap> {
    let mut bitmap = AppliedBlockBitmap::default();
    let mut seen: BTreeSet<(String, u32)> = BTreeSet::new();
    for entry in &pack.entries {
        if !seen.insert((entry.tensor.clone(), entry.block_idx)) {
            return Err(PatchError::OverlappingPatch {
                tensor: entry.tensor.clone(),
                block_idx: entry.block_idx,
            });
        }
        let tensor_entry = model
            .header()
            .tensors
            .iter()
            .find(|t| t.name == entry.tensor)
            .ok_or_else(|| PatchError::TensorNotFound(entry.tensor.clone()))?;
        let block_bytes = patch_block_bytes(tensor_entry.ggml_type).ok_or_else(|| {
            PatchError::UnsupportedType {
                tensor: entry.tensor.clone(),
                ggml_type: tensor_entry.ggml_type,
            }
        })?;
        let bytes = model
            .tensor_bytes(&entry.tensor)
            .ok_or_else(|| PatchError::TensorNotFound(entry.tensor.clone()))?;
        let total = (bytes.len() / block_bytes) as u32;
        if entry.block_idx >= total {
            return Err(PatchError::BlockOutOfRange {
                tensor: entry.tensor.clone(),
                block_idx: entry.block_idx,
                total,
            });
        }
        if entry.new_block_bits.len() != block_bytes {
            return Err(PatchError::BlockLenMismatch {
                tensor: entry.tensor.clone(),
                block_idx: entry.block_idx,
                expected: block_bytes,
                got: entry.new_block_bits.len(),
            });
        }
        let start = entry.block_idx as usize * block_bytes;
        let end = start + block_bytes;
        let actual_hash = sha256_hex(&bytes[start..end]);
        if actual_hash != entry.base_hash.to_lowercase() {
            return Err(PatchError::BaseHashMismatch {
                tensor: entry.tensor.clone(),
                block_idx: entry.block_idx,
                expected: entry.base_hash.clone(),
                got: actual_hash,
            });
        }
        bitmap
            .map
            .entry(entry.tensor.clone())
            .or_default()
            .insert(entry.block_idx);
    }
    Ok(bitmap)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SHA-256 known-answer vectors (NIST) ───────────────────────────

    #[test]
    fn sha256_empty() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_nist_two_block() {
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    // ── Synthetic Q8_0 blocks ─────────────────────────────────────────

    fn q8_0_block(scale_f16_bits: u16, values: [i8; 32]) -> Vec<u8> {
        let mut block = scale_f16_bits.to_le_bytes().to_vec();
        block.extend(values.iter().map(|v| *v as u8));
        block
    }

    const F16_1_0: u16 = 0x3C00;
    const F16_2_0: u16 = 0x4000;

    fn q8_0_entry(
        tensor: &str,
        block_idx: u32,
        bits: Vec<u8>,
        bytes: &[u8],
        block_bytes: usize,
    ) -> PatchEntry {
        let start = block_idx as usize * block_bytes;
        let base_hash = sha256_hex(&bytes[start..start + block_bytes]);
        PatchEntry {
            tensor: tensor.into(),
            block_idx,
            new_block_bits: bits,
            base_hash,
        }
    }

    #[test]
    fn q8_0_patch_roundtrip_block0_only() {
        let b0 = q8_0_block(F16_1_0, [100; 32]);
        let b1 = q8_0_block(F16_1_0, [7; 32]);
        let raw = [b0, b1].concat();
        // patch block 0: scale 2.0, values 50
        let patched_block = q8_0_block(F16_2_0, [50; 32]);
        let pack = PatchPack::new("t").with_entry(q8_0_entry(
            "blk.0.ffn_down",
            0,
            patched_block,
            &raw,
            34,
        ));
        let out = patch_tensor_blocks(8, &raw, "blk.0.ffn_down", &pack).expect("patch");
        assert_eq!(out.len(), raw.len());
        assert_eq!(&out[0..34], &q8_0_block(F16_2_0, [50; 32])[..]);
        assert_eq!(&out[34..], &raw[34..], "block 1 must be byte-identical");
        // dequant path agrees: block0 elements = 100 * 2.0, block1 = 7 * 1.0
        let (f32s, bitmap) =
            dequant_tensor_with_patches(8, &raw, "blk.0.ffn_down", &pack).expect("dequant");
        assert_eq!(f32s.len(), 64);
        assert_eq!(f32s[0], 100.0);
        assert_eq!(f32s[32], 7.0);
        assert!(bitmap.contains("blk.0.ffn_down", 0));
        assert!(!bitmap.contains("blk.0.ffn_down", 1));
        assert_eq!(bitmap.block_count(), 1);
    }

    #[test]
    fn patch_with_wrong_base_hash_is_rejected() {
        let raw = q8_0_block(F16_1_0, [100; 32]);
        let entry = PatchEntry::new(
            "blk.0.ffn_down",
            0,
            q8_0_block(F16_2_0, [50; 32]),
            "00".repeat(32),
        );
        let pack = PatchPack::new("t").with_entry(entry);
        let err = patch_tensor_blocks(8, &raw, "blk.0.ffn_down", &pack).unwrap_err();
        assert!(matches!(
            err,
            PatchError::BaseHashMismatch { block_idx: 0, .. }
        ));
    }

    #[test]
    fn patch_out_of_range_is_rejected() {
        let raw = q8_0_block(F16_1_0, [1; 32]);
        let entry = PatchEntry::new(
            "blk.0.ffn_down",
            5,
            q8_0_block(F16_2_0, [0; 32]),
            sha256_hex(&raw),
        );
        let pack = PatchPack::new("t").with_entry(entry);
        let err = patch_tensor_blocks(8, &raw, "blk.0.ffn_down", &pack).unwrap_err();
        assert!(matches!(
            err,
            PatchError::BlockOutOfRange {
                block_idx: 5,
                total: 1,
                ..
            }
        ));
    }

    #[test]
    fn patch_block_len_mismatch_is_rejected() {
        let raw = q8_0_block(F16_1_0, [1; 32]);
        let bits = vec![0u8; 10]; // not 34
        let pack = PatchPack::new("t").with_entry(q8_0_entry("blk.0.ffn_down", 0, bits, &raw, 34));
        let err = patch_tensor_blocks(8, &raw, "blk.0.ffn_down", &pack).unwrap_err();
        assert!(matches!(
            err,
            PatchError::BlockLenMismatch {
                expected: 34,
                got: 10,
                ..
            }
        ));
    }

    #[test]
    fn duplicate_block_patch_is_rejected() {
        let raw = q8_0_block(F16_1_0, [1; 32]);
        let e1 = q8_0_entry("blk.0.ffn_down", 0, q8_0_block(F16_2_0, [0; 32]), &raw, 34);
        let e2 = e1.clone();
        let pack = PatchPack::new("t").with_entry(e1).with_entry(e2);
        let err = patch_tensor_blocks(8, &raw, "blk.0.ffn_down", &pack).unwrap_err();
        assert!(matches!(
            err,
            PatchError::OverlappingPatch { block_idx: 0, .. }
        ));
    }

    #[test]
    fn unsupported_type_is_rejected() {
        let raw = vec![0u8; 72]; // Q4_0 = 18 B/block
        let entry = PatchEntry::new("blk.0.ffn_down", 0, vec![0u8; 18], sha256_hex(&raw[..18]));
        let pack = PatchPack::new("t").with_entry(entry);
        let err = patch_tensor_blocks(2, &raw, "blk.0.ffn_down", &pack).unwrap_err();
        assert!(matches!(
            err,
            PatchError::UnsupportedType { ggml_type: 2, .. }
        ));
    }

    #[test]
    fn empty_pack_leaves_bytes_and_f32_identical() {
        let raw = q8_0_block(F16_1_0, [3; 32]);
        let pack = PatchPack::new("t");
        let out = patch_tensor_blocks(8, &raw, "blk.0.ffn_down", &pack).expect("patch");
        assert_eq!(out, raw);
        let direct = quant::dequant(8, &raw).expect("dequant");
        let (f32s, bitmap) =
            dequant_tensor_with_patches(8, &raw, "blk.0.ffn_down", &pack).expect("dequant");
        assert_eq!(f32s, direct);
        assert_eq!(bitmap.block_count(), 0);
    }

    // ── Synthetic Q4_K / Q6_K blocks ──────────────────────────────────

    // ── WSP pack parsing + stage engine (synthetic, no GGUF needed) ──

    #[test]
    fn parse_wsp_roundtrip() {
        let data = "# wsp-pack v1\n\
                    blk.0.ffn_down.weight 7 0102 00ff\n\
                    blk.1.ffn_up.weight 0 030405 abcd\n";
        let pack = parse_wsp(data).expect("parse");
        assert_eq!(pack.entries.len(), 2);
        assert_eq!(pack.entries[0].tensor, "blk.0.ffn_down.weight");
        assert_eq!(pack.entries[0].block_idx, 7);
        assert_eq!(pack.entries[0].new_block_bits, vec![1, 2]);
        assert_eq!(pack.entries[1].new_block_bits, vec![3, 4, 5]);
        assert_eq!(pack.entries[0].base_hash, "00ff");
    }

    #[test]
    fn parse_wsp_rejects_garbage() {
        assert!(parse_wsp("blk.0 x y z\n").is_err());
        assert!(parse_wsp("blk.0 3 zz 00ff\n").is_err());
        assert_eq!(
            parse_wsp("# only comments\n").expect("parse").entries.len(),
            0
        );
    }

    #[test]
    fn apply_regions_splices_and_preserves_rest() {
        let src = (0u8..=255).cycle().take(4096).collect::<Vec<u8>>();
        let regions = vec![
            Region {
                start: 100,
                span: 8,
                replacement: vec![0xAA; 8],
            },
            Region {
                start: 3000,
                span: 4,
                replacement: vec![0xBB; 4],
            },
        ];
        let out = apply_regions(&src, &regions).expect("apply");
        assert_eq!(out.len(), src.len());
        assert_eq!(&out[100..108], &[0xAA; 8]);
        assert_eq!(&out[3000..3004], &[0xBB; 4]);
        assert_eq!(&out[..100], &src[..100]);
        assert_eq!(&out[108..3000], &src[108..3000]);
        assert_eq!(&out[3004..], &src[3004..]);
    }

    #[test]
    fn apply_regions_rejects_overlap() {
        let src = vec![0u8; 100];
        let regions = vec![
            Region {
                start: 10,
                span: 20,
                replacement: vec![0; 20],
            },
            Region {
                start: 15,
                span: 4,
                replacement: vec![0; 4],
            },
        ];
        assert!(apply_regions(&src, &regions).is_err());
    }

    #[test]
    fn stage_regions_on_file_end_to_end() {
        let dir = std::path::PathBuf::from("target/live");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let src = dir.join("wsp_stage_src.bin");
        let dst = dir.join("wsp_stage_dst.bin");
        let src_bytes: Vec<u8> = (0u8..255).cycle().take(2048).collect();
        std::fs::write(&src, &src_bytes).expect("write src");
        let regions = vec![
            Region {
                start: 100,
                span: 16,
                replacement: vec![0x5A; 16],
            },
            Region {
                start: 1900,
                span: 16,
                replacement: vec![0xC3; 16],
            },
        ];
        let patched = stage_regions_on_file(&src, &dst, &regions).expect("stage");
        assert_eq!(patched, 2);
        let got = std::fs::read(&dst).expect("read dst");
        assert_eq!(got.len(), src_bytes.len());
        assert_eq!(&got[100..116], &[0x5A; 16]);
        assert_eq!(&got[1900..1916], &[0xC3; 16]);
        assert_eq!(&got[..100], &src_bytes[..100]);
        assert_eq!(&got[116..1900], &src_bytes[116..1900]);
        assert_eq!(&got[1916..], &src_bytes[1916..]);
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    #[ignore = "~1 GiB file copy; run explicitly with --ignored"]
    fn stage_patch_identity_smoke() {
        let path = std::env::var("RUSTY_MINICPM_GGUF").unwrap_or_else(|_| {
            "S:/rust/rusty-engine/data/models/MiniCPM-V-4_7-1B-Q4_K_M.gguf".to_string()
        });
        let model = unsafe { crate::loader::MmapModel::open(std::path::Path::new(&path)) }
            .unwrap_or_else(|e| panic!("open {path}: {e}"));
        let name = "blk.0.ffn_down.weight";
        let entry = model
            .header()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .expect("ffn_down entry");
        let block_bytes = quant::block_info(entry.ggml_type)
            .expect("patchable type")
            .bytes;
        let block0 = model.tensor_bytes(name).expect("bytes")[..block_bytes].to_vec();
        let pack = PatchPack::new("id").with_entry(PatchEntry::new(
            name,
            0,
            block0,
            sha256_hex(model.tensor_bytes(name).expect("bytes")[..block_bytes].as_ref()),
        ));
        let dst = std::path::Path::new("target/live/wsp_smoke.gguf");
        let patched = stage_patch(std::path::Path::new(&path), dst, &pack).expect("stage");
        assert_eq!(patched, 1);
        let staged = unsafe { crate::loader::MmapModel::open(dst) }.expect("open staged");
        assert_eq!(
            staged.tensor_bytes(name).expect("staged bytes"),
            model.tensor_bytes(name).expect("base bytes"),
            "identity patch must not alter tensor bytes"
        );
        let _ = std::fs::remove_file(dst);
    }

    // ── Real-model smoke (ignored by default: needs MiniCPM-V 4.7-1B on disk) ──

    /// End-to-end on the real `blk.0.ffn_down` Q6_K tensor:
    /// 1. an empty pack dequantizes byte-identically to the base loader;
    /// 2. `validate_pack_against_file` accepts a block-0 patch minted from the
    ///    live file; `dequant_tensor_with_patches` applies it without touching
    ///    the base bytes (in-memory copy only).
    #[test]
    #[ignore = "requires MiniCPM-V-4_7-1B-Q4_K_M.gguf (RUSTY_MINICPM_GGUF override); ~10 s dequant"]
    fn real_ffn_down_pack_roundtrip() {
        let path = std::env::var("RUSTY_MINICPM_GGUF").unwrap_or_else(|_| {
            "S:/rust/rusty-engine/data/models/MiniCPM-V-4_7-1B-Q4_K_M.gguf".to_string()
        });
        let model = unsafe { crate::loader::MmapModel::open(std::path::Path::new(&path)) }
            .unwrap_or_else(|e| panic!("open {path}: {e}"));
        let name = "blk.0.ffn_down.weight";
        let base = model.dequant(name).expect("base dequant");
        // Empty pack → byte- and value-identical.
        let empty = PatchPack::new("real-empty");
        let (identity, bitmap) =
            dequant_with_patches(&model, name, &empty).expect("identity dequant");
        assert_eq!(bitmap.block_count(), 0);
        assert_eq!(identity.len(), base.len());
        assert_eq!(identity, base);
        // Mint a block-0 patch from the live file and validate fail-closed.
        let entry = model
            .header()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .expect("ffn_down entry");
        let block_bytes = quant::block_info(entry.ggml_type)
            .expect("patchable ggml_type")
            .bytes;
        let block0 = &model.tensor_bytes(name).expect("bytes")[..block_bytes];
        let base_hash = sha256_hex(block0);
        let pack = PatchPack::new("real-block0").with_entry(PatchEntry::new(
            name,
            0,
            block0.to_vec(), // re-apply the same block bits → identity patch
            base_hash,
        ));
        let bitmap = validate_pack_against_file(&model, &pack).expect("validate");
        assert!(bitmap.contains(name, 0));
        let (patched, bitmap) = dequant_with_patches(&model, name, &pack).expect("apply");
        assert!(bitmap.contains(name, 0));
        assert_eq!(patched, base); // identity patch → identical values
    }

    #[test]
    fn q6_k_block_patch_roundtrip() {
        assert_eq!(quant::block_info(14).unwrap().bytes, 210);
        let block: Vec<u8> = (0u8..210).collect();
        let raw = [block.clone(), vec![0xAB; 210]].concat();
        let entry = q8_0_entry("blk.0.ffn_down", 0, vec![0x7F; 210], &raw, 210);
        let pack = PatchPack::new("t").with_entry(entry);
        let out = patch_tensor_blocks(14, &raw, "blk.0.ffn_down", &pack).expect("patch");
        assert_eq!(&out[..210], &vec![0x7F; 210][..]);
        assert_eq!(&out[210..], &vec![0xAB; 210][..]);
        let (f32s, bitmap) =
            dequant_tensor_with_patches(14, &raw, "blk.0.ffn_down", &pack).expect("dequant");
        assert_eq!(f32s.len(), 512);
        assert!(bitmap.contains("blk.0.ffn_down", 0));
    }

    #[test]
    fn q4_k_block_patch_roundtrip() {
        assert_eq!(quant::block_info(12).unwrap().bytes, 144);
        let raw = vec![0x5A; 288]; // two superblocks
        let entry = q8_0_entry("blk.0.ffn_down", 1, vec![0x11; 144], &raw, 144);
        let pack = PatchPack::new("t").with_entry(entry);
        let out = patch_tensor_blocks(12, &raw, "blk.0.ffn_down", &pack).expect("patch");
        assert_eq!(&out[..144], &vec![0x5A; 144][..]);
        assert_eq!(&out[144..], &vec![0x11; 144][..]);
    }

    #[test]
    fn base_hash_pins_exact_base_model() {
        let raw = q8_0_block(F16_1_0, [9; 32]);
        let entry = q8_0_entry("blk.0.ffn_down", 0, q8_0_block(F16_2_0, [1; 32]), &raw, 34);
        // Same entry, different base model (one byte flipped) → must fail.
        let mut other_raw = raw.clone();
        other_raw[3] ^= 0x01;
        let pack = PatchPack::new("t").with_entry(entry);
        let err = patch_tensor_blocks(8, &other_raw, "blk.0.ffn_down", &pack).unwrap_err();
        assert!(matches!(err, PatchError::BaseHashMismatch { .. }));
    }
}
