//! PH-1204: dense-path image-token prefix (no real model, no quality claim).
//! Text-only prefill stays bit-identical; only an explicit non-empty prefix
//! changes the embedded sequence fed through the cached block path.

use rusty_engine::chatml::{apply_chatml_template, apply_minicpm_template};
use rusty_engine::forward::{BlockWeights, Geometry};
use rusty_engine::qmat::QMat;
use rusty_engine::runner::{DenseDecoder, DenseRunner};

fn zero_block(geom: &Geometry) -> BlockWeights {
    let dim = geom.d_model;
    let zeros = |r: usize, c: usize| QMat::from_f32(vec![0.0f32; r * c], r, c);
    BlockWeights {
        attn_norm: vec![1.0; dim],
        q: zeros(dim, dim),
        k: zeros(geom.n_kv_heads * geom.head_dim, dim),
        v: zeros(geom.n_kv_heads * geom.head_dim, dim),
        o: zeros(dim, dim),
        q_bias: Vec::new(),
        k_bias: Vec::new(),
        v_bias: Vec::new(),
        ffn_norm: vec![1.0; dim],
        ffn_gate: zeros(geom.d_ff, dim),
        ffn_up: zeros(geom.d_ff, dim),
        ffn_down: zeros(dim, geom.d_ff),
    }
}

fn toy_runner() -> DenseRunner {
    let geom = Geometry::new(4, 2, 1, 8, 10_000.0, 0.0);
    let mut embed = vec![0.0f32; 6 * 4];
    embed[0] = 1.0; // token 0 -> [1,0,0,0]
    embed[5] = 1.0; // token 1 -> [0,1,0,0]
    embed[10] = 1.0; // token 2 -> [0,0,1,0]
    let mut output = vec![0.0f32; 6 * 4];
    for i in 0..4 {
        output[i * 4 + i] = 1.0; // identity rows 0..4
    }
    DenseRunner {
        geom,
        n_layers: 1,
        n_vocab: 6,
        embed,
        blocks: vec![zero_block(&geom)],
        output_norm: vec![1.0; 4],
        output: QMat::from_f32(output, 6, 4),
        eos_token_id: None,
    }
}

#[test]
fn empty_vision_prefix_is_bit_identical_to_plain_prefill() {
    let runner = toy_runner();
    let prompt = [0u32, 1, 2];
    let plain = DenseDecoder::new(&runner).prefill(&prompt);
    let merged = DenseDecoder::new(&runner)
        .prefill_with_vision(&prompt, &[], 0)
        .unwrap();
    assert_eq!(plain, merged);
}

#[test]
fn vision_rows_after_prompt_define_the_last_logits() {
    let runner = toy_runner();
    let logits = DenseDecoder::new(&runner)
        .prefill_with_vision(&[0, 1, 2], &[vec![1.0, 2.0, 3.0, 4.0]], 3)
        .unwrap();
    // RMSNorm([1,2,3,4], eps=0) = [0.36514837, 0.73029674, 1.0954451, 1.4605935]
    for (got, want) in
        logits[..4]
            .iter()
            .zip([0.365_148_37_f32, 0.730_296_74, 1.095_445_1, 1.460_593_5])
    {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }
}

#[test]
fn vision_rows_at_zero_leave_the_last_token_as_final_row() {
    let runner = toy_runner();
    let logits = DenseDecoder::new(&runner)
        .prefill_with_vision(&[0, 1, 2], &[vec![9.0, 9.0, 9.0, 9.0]], 0)
        .unwrap();
    // Last row = embed(token 2) = [0,0,1,0]; RMSNorm scales by 1/sqrt(1/4) = 2.
    assert!((logits[0] - 0.0).abs() < 1e-6);
    assert!((logits[1] - 0.0).abs() < 1e-6);
    assert!((logits[2] - 2.0).abs() < 1e-6);
    assert!((logits[3] - 0.0).abs() < 1e-6);
}

#[test]
fn vision_prefix_rejects_wrong_embedding_width() {
    let runner = toy_runner();
    let got = DenseDecoder::new(&runner).prefill_with_vision(&[0], &[vec![1.0, 2.0]], 0);
    assert!(got.is_err(), "wrong-width vision row must fail closed");
}

#[test]
fn minicpm_template_inserts_image_placeholders_before_assistant_turn() {
    let json = r#"[{"role":"user","content":"what is this"}]"#;
    let base = apply_chatml_template(json).unwrap();
    let with_images = apply_minicpm_template(json, 2).unwrap();
    assert!(with_images.contains("<image><image>"));
    let marker = "<|im_start|>assistant\n";
    let pos = with_images.rfind(marker).unwrap();
    assert_eq!(
        &with_images[pos..],
        marker,
        "placeholders sit before the assistant turn"
    );
    assert_eq!(apply_minicpm_template(json, 0).unwrap(), base);
}
