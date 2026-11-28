//! PH-1202 real-weight canary (ignored by default: needs the gitignored
//! `mmproj-model-f16.gguf`). Shape/finiteness only — no mtmd golden, no token,
//! no image→answer or quality claim (G13 blocked).

use rusty_engine::vit::{patch_embed, vit_block_forward, VitGeom};
use rusty_engine::vit_real::{load_vit_weights, MM_PROJ_REAL, VIT_DIM};

#[test]
#[ignore = "requires data/models/mmproj-model-f16.gguf (gitignored)"]
fn real_mmproj_block_loads_with_registered_shapes() {
    let w = load_vit_weights(MM_PROJ_REAL, 1).expect("load real mmproj");
    assert_eq!((w.geom.dim, w.geom.heads, w.geom.mlp), (1152, 16, 4304));
    assert_eq!((w.geom.grid, w.geom.patch), (70, 14));
    assert_eq!(w.blocks.len(), 1);
    let b = &w.blocks[0];
    assert_eq!((b.q.rows(), b.q.cols()), (1152, 1152));
    assert_eq!((b.mlp_in.rows(), b.mlp_in.cols()), (4304, 1152));
    assert_eq!((b.mlp_out.rows(), b.mlp_out.cols()), (1152, 4304));
    assert_eq!(b.q_bias.len(), VIT_DIM);
    assert_eq!(w.patch_w.rows(), VIT_DIM);
    assert_eq!(w.patch_w.cols(), 14 * 14 * 3);
    assert_eq!(w.position.len(), 70 * 70 * VIT_DIM);
    assert!(w.patch_bias.iter().all(|v| v.is_finite()));
    assert!(w.position[..64].iter().all(|v| v.is_finite()));

    // Two-token canary on the real block (no 4900x4900 attention matrix).
    let geom = VitGeom {
        grid: 2,
        layers: 1,
        ..w.geom
    };
    let x = vec![0.1f32; 2 * VIT_DIM];
    let out = vit_block_forward(&x, b, &geom);
    assert_eq!(out.len(), 2 * VIT_DIM);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "real block must stay finite on a toy input"
    );
}

#[test]
#[ignore = "requires data/models/mmproj-model-f16.gguf (gitignored)"]
fn real_patch_embed_produces_4900_tokens() {
    let w = load_vit_weights(MM_PROJ_REAL, 0).expect("load real mmproj");
    let side = w.geom.grid * w.geom.patch; // 448
    let image: Vec<f32> = (0..side * side * 3)
        .map(|i| (i % 255) as f32 / 255.0)
        .collect();
    let patches = patch_embed(&image, &w.geom, &w.patch_w, &w.patch_bias);
    assert_eq!(patches.len(), w.geom.n_tokens() * VIT_DIM);
    assert_eq!(patches.len(), 4900 * 1152);
    assert!(patches[..64].iter().all(|v| v.is_finite()));
}
