//! PH-1203: 4x/16x visual-token compressor + projector on the PH-1202 ViT
//! tower output. Structural/shape gate only — no real model, no image→answer
//! or quality claim (G13 blocked).

use rusty_engine::qmat::QMat;
use rusty_engine::vit::{
    visual_merger_forward, visual_pool, visual_project, VisualCompressor, VisualProjector,
};

/// Token `(y,x)` is `[ (y*4+x)*10, (y*4+x)*10 + 1 ]` on a 4×4 grid.
fn toy_tokens() -> Vec<f32> {
    let mut out = Vec::with_capacity(4 * 4 * 2);
    for y in 0..4 {
        for x in 0..4 {
            let base = ((y * 4 + x) * 10) as f32;
            out.extend([base, base + 1.0]);
        }
    }
    out
}

fn identity_projector() -> VisualProjector {
    VisualProjector {
        project: QMat::from_f32(vec![1.0, 0.0, 0.0, 1.0], 2, 2),
        bias: vec![0.0; 2],
        ln_w: vec![1.0; 2],
        ln_b: vec![0.0; 2],
    }
}

#[test]
fn pool_2x_reduces_4x4_grid_to_2x2_with_hand_averages() {
    let pooled = visual_pool(&toy_tokens(), 4, 2, 2);
    assert_eq!(pooled.len(), 2 * 2 * 2);
    // Blocks in row-major order: (0,0)=[25,26], (0,1)=[45,46],
    // (1,0)=[105,106], (1,1)=[125,126].
    assert_eq!(
        pooled,
        vec![25.0, 26.0, 45.0, 46.0, 105.0, 106.0, 125.0, 126.0]
    );
}

#[test]
fn pool_4x_reduces_16_tokens_to_one() {
    let pooled = visual_pool(&toy_tokens(), 4, 2, 4);
    assert_eq!(pooled.len(), 2);
    // 16 tokens: dim0 values 0..150 step 10 (mean 75), dim1 mean 76.
    assert_eq!(pooled, vec![75.0, 76.0]);
}

#[test]
fn projector_maps_tokens_to_hidden_and_normalizes_each_row() {
    let pooled = vec![25.0, 26.0, 45.0, 46.0]; // two tokens of width 2
    let out = visual_project(&pooled, &identity_projector(), 0.0);
    assert_eq!(out.len(), 4);
    // LayerNorm([25,26]) = [-1,1]; LayerNorm([45,46]) = [-1,1].
    assert_eq!(out, vec![-1.0, 1.0, -1.0, 1.0]);
}

#[test]
fn merger_end_to_end_is_16x_and_hidden_width() {
    let comp = VisualCompressor {
        grid: 4,
        dim: 2,
        factor: 4,
        projector: identity_projector(),
    };
    let out = visual_merger_forward(&toy_tokens(), &comp, 0.0);
    assert_eq!(out.len(), 2, "16x compression => one token x hidden");
    assert_eq!(out, vec![-1.0, 1.0]);
}

#[test]
#[should_panic(expected = "visual_pool: grid not divisible by factor")]
fn pool_rejects_grid_not_divisible_by_factor() {
    let tokens = vec![0.0f32; 3 * 3 * 2];
    let _ = visual_pool(&tokens, 3, 2, 2);
}
