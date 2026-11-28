//! A1 (pure half of `PH-406`): map `(layer, expert, role)` to a quantized byte
//! range in a GGUF or a GASV expert pack, and dequantize it into
//! [`crate::moe::ExpertWeights`].
//!
//! This is deliberately std-only and node-free: it never loads a model, never
//! touches an `ExpertCache` and makes no quality claim. The real-tensor wiring
//! into the `load` closure of `moe_layer_forward_cached` stays owner-gated.
//!
//! The layout mirrors GSV `src/boxes/moe_forward.rs::ExpertAddr::byte_range`
//! and the registered `gasv.expert-pack.v1` sidecar manifest: one expert is
//! `values_per_expert / 256` blocks, Q4_K = 144 B/block (gate/up), Q6_K =
//! 210 B/block (down).

use std::io::{Read, Seek, SeekFrom};

use crate::moe::ExpertWeights;
use crate::quant::dequant;

/// Values in one K-quant super-block.
pub const Q_BLOCK_VALS: u64 = 256;
/// Bytes per Q4_K super-block (GGML type 12).
pub const Q4_K_BLOCK_BYTES: u64 = 144;
/// Bytes per Q6_K super-block (GGML type 14).
pub const Q6_K_BLOCK_BYTES: u64 = 210;

/// A routed expert tensor inside a layer (`ffn_{gate,up,down}_exps.weight`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRole {
    Gate,
    Up,
    Down,
}

impl ExpertRole {
    /// GGUF tensor name for a zero-indexed layer.
    pub fn tensor_name(self, layer: usize) -> String {
        let role = match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        };
        format!("blk.{layer}.ffn_{role}_exps.weight")
    }

    /// GGML quant type: gate/up are Q4_K (12), down is Q6_K (14).
    pub fn ggml_type(self) -> u32 {
        match self {
            Self::Gate | Self::Up => 12,
            Self::Down => 14,
        }
    }

    fn block_bytes(self) -> u64 {
        match self {
            Self::Gate | Self::Up => Q4_K_BLOCK_BYTES,
            Self::Down => Q6_K_BLOCK_BYTES,
        }
    }
}

/// Byte range `(start, len)` of one expert inside a quantized tensor.
///
/// `data_start` is the aligned tensor-data start and `tensor_offset` is the
/// per-tensor offset from the GGUF manifest (as in GSV `manifest_offsets`).
/// Returns `None` on a zero/partial super-block or arithmetic overflow.
pub fn gguf_expert_byte_range(
    data_start: u64,
    tensor_offset: u64,
    values_per_expert: u64,
    expert: usize,
    role: ExpertRole,
) -> Option<(u64, u64)> {
    if values_per_expert == 0 || values_per_expert % Q_BLOCK_VALS != 0 {
        return None;
    }
    let len = (values_per_expert / Q_BLOCK_VALS).checked_mul(role.block_bytes())?;
    if len == 0 {
        return None;
    }
    let stride = u64::try_from(expert).ok()?.checked_mul(len)?;
    let start = data_start.checked_add(tensor_offset)?.checked_add(stride)?;
    Some((start, len))
}

/// Tensor geometry shared by gate/up and down: `d_model * ffn` values per expert.
#[derive(Debug, Clone, Copy)]
pub struct ExpertGeometry {
    pub d_model: u64,
    pub ffn: u64,
}

impl ExpertGeometry {
    pub fn values_per_expert(self) -> Option<u64> {
        self.d_model.checked_mul(self.ffn)
    }
}

/// Manifest offsets for the three tensors of one layer (already aligned).
#[derive(Debug, Clone, Copy)]
pub struct ExpertTensorOffsets {
    pub data_start: u64,
    pub gate: u64,
    pub up: u64,
    pub down: u64,
    pub geometry: ExpertGeometry,
}

impl ExpertTensorOffsets {
    fn offset(self, role: ExpertRole) -> u64 {
        match role {
            ExpertRole::Gate => self.gate,
            ExpertRole::Up => self.up,
            ExpertRole::Down => self.down,
        }
    }
}

/// Read and dequantize one routed expert into row-major FFN weights.
pub fn load_expert_gguf<R: Read + Seek>(
    reader: &mut R,
    offsets: &ExpertTensorOffsets,
    expert: usize,
) -> Result<ExpertWeights, String> {
    let values = offsets
        .geometry
        .values_per_expert()
        .ok_or("expert geometry overflows")?;
    let mut load = |role: ExpertRole| -> Result<Vec<f32>, String> {
        let (start, len) = gguf_expert_byte_range(
            offsets.data_start,
            offsets.offset(role),
            values,
            expert,
            role,
        )
        .ok_or_else(|| {
            format!(
                "{} expert {expert}: invalid byte range",
                role.tensor_name(0)
            )
        })?;
        reader
            .seek(SeekFrom::Start(start))
            .map_err(|e| format!("seek {start}: {e}"))?;
        let mut bytes = vec![0u8; usize::try_from(len).map_err(|_| "range too large")?];
        reader
            .read_exact(&mut bytes)
            .map_err(|e| format!("read {start}..{}: {e}", start + len))?;
        let out = dequant(role.ggml_type(), &bytes).map_err(|e| e.message)?;
        if out.len() as u64 != values {
            return Err(format!(
                "{}: dequantized {} values, expected {values}",
                role.tensor_name(0),
                out.len()
            ));
        }
        Ok(out)
    };
    Ok(ExpertWeights {
        gate: load(ExpertRole::Gate)?,
        up: load(ExpertRole::Up)?,
        down: load(ExpertRole::Down)?,
    })
}

/// Three in-pack byte ranges of one expert record in a
/// `gasv.expert-pack.v1` sidecar manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackExpertRanges {
    pub file: String,
    pub sha256: String,
    pub gate: (u64, u64),
    pub up: (u64, u64),
    pub down: (u64, u64),
}

/// Cross-check a registered pack record against the GGUF byte-range formula.
/// Fail-closed on wrong schema, layout, size or missing fields.
pub fn pack_expert_ranges(
    manifest: &serde_json::Value,
    expert: usize,
    geometry: ExpertGeometry,
) -> Result<PackExpertRanges, String> {
    if manifest.get("schema").and_then(|v| v.as_str()) != Some("gasv.expert-pack.v1") {
        return Err("not a gasv.expert-pack.v1 manifest".into());
    }
    let entries = manifest
        .get("entries")
        .and_then(|v| v.as_array())
        .ok_or("pack manifest has no entries")?;
    let row = entries
        .iter()
        .find(|r| r.get("expert").and_then(|v| v.as_u64()) == Some(expert as u64))
        .ok_or_else(|| format!("no pack entry for expert {expert}"))?;
    if row.get("quant_layout").and_then(|v| v.as_str()) != Some("q4k_gate_up_q6k_down") {
        return Err("unexpected pack quant_layout".into());
    }
    let values = geometry
        .values_per_expert()
        .ok_or("expert geometry overflows")?;
    let gate_len = gguf_expert_byte_range(0, 0, values, 0, ExpertRole::Gate)
        .ok_or("invalid gate range")?
        .1;
    let up_len = gguf_expert_byte_range(0, 0, values, 0, ExpertRole::Up)
        .ok_or("invalid up range")?
        .1;
    let down_len = gguf_expert_byte_range(0, 0, values, 0, ExpertRole::Down)
        .ok_or("invalid down range")?
        .1;
    let expected = gate_len
        .checked_add(up_len)
        .and_then(|v| v.checked_add(down_len))
        .ok_or("pack record size overflows")?;
    let field = |name: &str| -> Result<u64, String> {
        row.get(name)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("pack entry has no {name}"))
    };
    if field("payload_bytes")? != expected || field("len")? != expected {
        return Err(format!(
            "pack record size mismatch: expected {expected}, got len={}, payload_bytes={}",
            field("len")?,
            field("payload_bytes")?
        ));
    }
    let offset = field("offset")?;
    let sha256 = row
        .get("sha256")
        .and_then(|v| v.as_str())
        .ok_or("pack entry has no sha256")?;
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("pack entry has invalid sha256".into());
    }
    let file = row
        .get("file")
        .and_then(|v| v.as_str())
        .ok_or("pack entry has no file")?;
    let up_start = offset.checked_add(gate_len).ok_or("pack offset overflow")?;
    let down_start = up_start.checked_add(up_len).ok_or("pack offset overflow")?;
    let _end = down_start
        .checked_add(down_len)
        .ok_or("pack offset overflow")?;
    Ok(PackExpertRanges {
        file: file.to_string(),
        sha256: sha256.to_string(),
        gate: (offset, gate_len),
        up: (up_start, up_len),
        down: (down_start, down_len),
    })
}
