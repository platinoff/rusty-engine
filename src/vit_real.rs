//! PH-1202 real-weight binding: read the MiniCPM-V 4.7-1B `mmproj` GGUF
//! (BF16(30) vision tensors) into the structural [`crate::vit`] blocks.
//!
//! This is still **not** a model forward or an image→answer claim: no mtmd
//! golden parity, no trained alignment, no token output (`G13` blocked). It
//! only makes the shape/canary check run on the real registered tensors.

use std::path::Path;

use crate::loader::MmapModel;
use crate::qmat::QMat;
use crate::vit::{VitBlockWeights, VitGeom};

/// Default 1B mmproj artifact (gitignored, 1.1 GiB, 460 tensors).
pub const MM_PROJ_REAL: &str = "data/models/mmproj-model-f16.gguf";

/// Real geometry from `docs/research/minicpm-v-4.7.md` §2.
pub const VIT_DIM: usize = 1152;
pub const VIT_FFN: usize = 4304;
pub const VIT_HEADS: usize = 16;
pub const VIT_GRID: usize = 70;
pub const VIT_PATCH: usize = 14;
pub const VIT_LAYERS: usize = 27;
pub const VIT_EPS: f32 = 1e-6;

/// Real vision tensors dequantized to f32 and shaped for the structural ViT.
#[derive(Debug)]
pub struct VitRealWeights {
    pub geom: VitGeom,
    pub blocks: Vec<VitBlockWeights>,
    /// `[dim, patch²×3]` row-major projection (GGUF stores `[patch²×3, dim]`).
    pub patch_w: QMat,
    pub patch_bias: Vec<f32>,
    pub position: Vec<f32>,
    pub post_w: Vec<f32>,
    pub post_b: Vec<f32>,
}

fn take(model: &MmapModel, name: &str) -> Result<Vec<f32>, String> {
    model.dequant(name).map_err(|e| e.message)
}

fn shaped(v: Vec<f32>, rows: usize, cols: usize, name: &str) -> Result<QMat, String> {
    if v.len() != rows.checked_mul(cols).ok_or("shape overflow")? {
        return Err(format!(
            "{name}: {} values, expected {rows}x{cols}",
            v.len()
        ));
    }
    Ok(QMat::from_f32(v, rows, cols))
}

fn exact(v: Vec<f32>, len: usize, name: &str) -> Result<Vec<f32>, String> {
    if v.len() != len {
        return Err(format!("{name}: {} values, expected {len}", v.len()));
    }
    Ok(v)
}

/// Load `layers` ViT blocks (0..=27) plus patch/position/post tensors.
/// Memory: ~22 MB for the position table, ~8 MB per block (f32 after dequant).
pub fn load_vit_weights(path: &str, layers: usize) -> Result<VitRealWeights, String> {
    if layers > VIT_LAYERS {
        return Err(format!("layers {layers} exceeds {VIT_LAYERS}"));
    }
    // SAFETY: the caller owns the artifact and must not mutate it while mapped;
    // this function never writes to the mmap and keeps `model` alive locally.
    let model = unsafe { MmapModel::open(Path::new(path)).map_err(|e| e.to_string())? };

    let mut blocks = Vec::with_capacity(layers);
    for i in 0..layers {
        let p = format!("v.blk.{i}");
        let w = VitBlockWeights {
            ln1_w: exact(take(&model, &format!("{p}.ln1.weight"))?, VIT_DIM, "ln1.w")?,
            ln1_b: exact(take(&model, &format!("{p}.ln1.bias"))?, VIT_DIM, "ln1.b")?,
            q: shaped(
                take(&model, &format!("{p}.attn_q.weight"))?,
                VIT_DIM,
                VIT_DIM,
                "attn_q",
            )?,
            k: shaped(
                take(&model, &format!("{p}.attn_k.weight"))?,
                VIT_DIM,
                VIT_DIM,
                "attn_k",
            )?,
            v: shaped(
                take(&model, &format!("{p}.attn_v.weight"))?,
                VIT_DIM,
                VIT_DIM,
                "attn_v",
            )?,
            o: shaped(
                take(&model, &format!("{p}.attn_out.weight"))?,
                VIT_DIM,
                VIT_DIM,
                "attn_out",
            )?,
            q_bias: exact(
                take(&model, &format!("{p}.attn_q.bias"))?,
                VIT_DIM,
                "attn_q.bias",
            )?,
            k_bias: exact(
                take(&model, &format!("{p}.attn_k.bias"))?,
                VIT_DIM,
                "attn_k.bias",
            )?,
            v_bias: exact(
                take(&model, &format!("{p}.attn_v.bias"))?,
                VIT_DIM,
                "attn_v.bias",
            )?,
            o_bias: exact(
                take(&model, &format!("{p}.attn_out.bias"))?,
                VIT_DIM,
                "attn_out.bias",
            )?,
            ln2_w: exact(take(&model, &format!("{p}.ln2.weight"))?, VIT_DIM, "ln2.w")?,
            ln2_b: exact(take(&model, &format!("{p}.ln2.bias"))?, VIT_DIM, "ln2.b")?,
            // GGUF ffn_up is [dim, ffn] -> QMat [ffn, dim]; down is the transpose.
            mlp_in: shaped(
                take(&model, &format!("{p}.ffn_up.weight"))?,
                VIT_FFN,
                VIT_DIM,
                "ffn_up",
            )?,
            mlp_out: shaped(
                take(&model, &format!("{p}.ffn_down.weight"))?,
                VIT_DIM,
                VIT_FFN,
                "ffn_down",
            )?,
        };
        blocks.push(w);
    }

    let patch_vals = VIT_PATCH * VIT_PATCH * 3;
    let raw = exact(
        take(&model, "v.patch_embd.weight")?,
        patch_vals * VIT_DIM,
        "patch_embd",
    )?;
    let mut transposed = vec![0.0f32; VIT_DIM * patch_vals];
    for p in 0..patch_vals {
        for o in 0..VIT_DIM {
            transposed[o * patch_vals + p] = raw[p * VIT_DIM + o];
        }
    }

    Ok(VitRealWeights {
        geom: VitGeom {
            patch: VIT_PATCH,
            grid: VIT_GRID,
            dim: VIT_DIM,
            heads: VIT_HEADS,
            mlp: VIT_FFN,
            layers,
            eps: VIT_EPS,
        },
        blocks,
        patch_w: QMat::from_f32(transposed, VIT_DIM, patch_vals),
        patch_bias: exact(take(&model, "v.patch_embd.bias")?, VIT_DIM, "patch_bias")?,
        position: exact(
            take(&model, "v.position_embd.weight")?,
            VIT_GRID * VIT_GRID * VIT_DIM,
            "position_embd",
        )?,
        post_w: exact(take(&model, "v.post_ln.weight")?, VIT_DIM, "post_ln.w")?,
        post_b: exact(take(&model, "v.post_ln.bias")?, VIT_DIM, "post_ln.b")?,
    })
}
