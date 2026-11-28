//! PH-1202: pure-Rust ViT tower on deterministic toy weights.
//! Structural/unit gate only — no real model, no golden mtmd parity, no
//! image→answer or quality claim (G13 blocked).

use rusty_engine::qmat::QMat;
use rusty_engine::vit::{
    gelu, layer_norm, patch_embed, pos_embed_2d, vit_attention_forward, vit_block_forward,
    vit_tower_forward, VitBlockWeights, VitGeom,
};

fn tiny_geom() -> VitGeom {
    VitGeom {
        patch: 2,
        grid: 2,
        dim: 4,
        heads: 2,
        mlp: 8,
        layers: 1,
        eps: 1e-5,
    }
}

fn identity_qmat(dim: usize) -> QMat {
    let mut w = vec![0.0f32; dim * dim];
    for i in 0..dim {
        w[i * dim + i] = 1.0;
    }
    QMat::from_f32(w, dim, dim)
}

fn zero_block(geom: &VitGeom) -> VitBlockWeights {
    let dim = geom.dim;
    VitBlockWeights {
        ln1_w: vec![1.0; dim],
        ln1_b: vec![0.0; dim],
        q: zero_qmat(dim, dim),
        k: zero_qmat(dim, dim),
        v: zero_qmat(dim, dim),
        o: zero_qmat(dim, dim),
        ln2_w: vec![1.0; dim],
        ln2_b: vec![0.0; dim],
        mlp_in: zero_qmat(geom.mlp, dim),
        mlp_out: zero_qmat(dim, geom.mlp),
    }
}

fn zero_qmat(rows: usize, cols: usize) -> QMat {
    QMat::from_f32(vec![0.0f32; rows * cols], rows, cols)
}

#[test]
fn patch_embed_takes_14_style_row_major_patches_plus_bias() {
    let geom = tiny_geom(); // 2x2 patches of 2x2x3 = 12 values
    let image: Vec<f32> = (0..48).map(|i| i as f32).collect();
    let mut w = vec![0.0f32; geom.dim * 12];
    for i in 0..geom.dim {
        w[i * 12 + i] = 1.0;
    }
    let patches = patch_embed(&image, &geom, &QMat::from_f32(w, geom.dim, 12), &[1.0; 4]);
    assert_eq!(patches.len(), geom.grid * geom.grid * geom.dim);
    // patch (0,0): first pixel 0 -> [0,1,2], + bias -> [1,2,3,4]
    assert_eq!(&patches[..4], &[1.0, 2.0, 3.0, 4.0]);
    // patch (0,1) starts at pixel 2 -> [6,7,8, 9,10,11, 18,19,20, 21,22,23]
    assert_eq!(&patches[4..8], &[7.0, 8.0, 9.0, 10.0]);
    // patch (1,1) starts at pixel 10 -> [30,31,32, ...]
    assert_eq!(&patches[12..16], &[31.0, 32.0, 33.0, 34.0]);
}

#[test]
fn layer_norm_centers_and_scales_with_weight_and_bias() {
    let out = layer_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], &[0.0; 4], 0.0);
    // mean 2.5, variance 1.25, sigma ~1.1180339887
    for (got, want) in out
        .iter()
        .zip([-1.3416408_f32, -0.4472136, 0.4472136, 1.3416408])
    {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }
    let shifted = layer_norm(&[1.0, 2.0, 3.0, 4.0], &[2.0; 4], &[0.5; 4], 0.0);
    assert!((shifted[0] - (2.0 * -1.3416408 + 0.5)).abs() < 1e-6);
}

#[test]
fn gelu_uses_tanh_approximation_and_is_odd() {
    assert_eq!(gelu(0.0), 0.0);
    assert!((gelu(1.0) - 0.841192).abs() < 1e-3);
    assert!((gelu(-1.0) + 0.158808).abs() < 1e-3);
}

#[test]
fn pos_embed_2d_bilinearly_interpolates_2x2_table_to_3x3() {
    let table = [0.0, 10.0, 20.0, 30.0]; // row-major, dim = 1
    let out = pos_embed_2d(&table, 2, 3, 1);
    assert_eq!(out.len(), 9);
    for (idx, want) in [
        (0, 0.0),
        (1, 5.0),
        (2, 10.0),
        (4, 15.0),
        (6, 20.0),
        (8, 30.0),
    ] {
        assert!(
            (out[idx] - want).abs() < 1e-6,
            "idx {idx}: {} != {want}",
            out[idx]
        );
    }
}

#[test]
fn vit_attention_is_non_causal_over_all_tokens() {
    let geom = VitGeom {
        dim: 2,
        heads: 1,
        ..tiny_geom()
    };
    let w = VitBlockWeights {
        ln1_w: vec![1.0; 2],
        ln1_b: vec![0.0; 2],
        q: identity_qmat(2),
        k: identity_qmat(2),
        v: identity_qmat(2),
        o: identity_qmat(2),
        ln2_w: vec![1.0; 2],
        ln2_b: vec![0.0; 2],
        mlp_in: zero_qmat(4, 2),
        mlp_out: zero_qmat(2, 4),
    };
    let tokens = [1.0, 0.0, 0.0, 1.0]; // two tokens
    let out = vit_attention_forward(&tokens, &w, &geom);
    // scale 1/sqrt(2): softmax([0.7071, 0]) = [0.6698, 0.3302]
    assert!((out[0] - 0.6698).abs() < 1e-3, "out[0]={}", out[0]);
    assert!((out[1] - 0.3302).abs() < 1e-3, "out[1]={}", out[1]);

    // A later token must be able to change an earlier token: no causal mask.
    let changed = [1.0, 0.0, 1.0, 0.0];
    let out2 = vit_attention_forward(&changed, &w, &geom);
    assert!(
        (out2[0] - out[0]).abs() > 1e-3,
        "attention must be non-causal"
    );
}

#[test]
fn zero_vit_block_is_pure_residual() {
    let geom = tiny_geom();
    let block = zero_block(&geom);
    let x = [0.6, -0.3, 0.4, 0.1];
    let out = vit_block_forward(&x, &block, &geom);
    assert_eq!(
        out,
        x.to_vec(),
        "zero attention + zero MLP => residual only"
    );
}

#[test]
fn vit_tower_keeps_token_shape_and_is_deterministic() {
    let geom = tiny_geom();
    let block = zero_block(&geom);
    let image: Vec<f32> = (0..48).map(|i| i as f32 * 0.01).collect();
    let mut pw = vec![0.0f32; geom.dim * 12];
    for i in 0..geom.dim {
        pw[i * 12 + i] = 1.0;
    }
    let tokens = patch_embed(&image, &geom, &QMat::from_f32(pw, geom.dim, 12), &[0.0; 4]);
    let table: Vec<f32> = (0..2 * 2 * geom.dim).map(|i| i as f32 * 0.01).collect();
    let pos = pos_embed_2d(&table, 2, geom.grid, geom.dim);
    let positioned: Vec<f32> = tokens.iter().zip(&pos).map(|(a, b)| a + b).collect();
    let a = vit_tower_forward(
        &positioned,
        std::slice::from_ref(&block),
        &[1.0; 4],
        &[0.0; 4],
        &geom,
    );
    let b = vit_tower_forward(&positioned, &[block], &[1.0; 4], &[0.0; 4], &geom);
    assert_eq!(a.len(), geom.grid * geom.grid * geom.dim);
    assert_eq!(a, b);
    assert!(a.iter().all(|v| v.is_finite()));
}
