//! B3 `PH-305` — dense end-to-end forward over a synthetic GGUF + optional real-model
//! smoke. The main test proves the whole stack (embed -> N x block_forward_seq -> final
//! rmsnorm -> output -> argmax) composes on real dequantized F32 tensors and yields a
//! deterministic greedy token. Numeric parity vs llama.cpp is `PH-405` (separate band).

use std::path::Path;

use rusty_engine::infer::{GenerationOptions, SeedRng};
use rusty_engine::loader::MmapModel;
use rusty_engine::model::ModelConfig;
use rusty_engine::runner::{DenseDecoder, DenseRunner};

fn u32le(b: &mut Vec<u8>, x: u32) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn u64le(b: &mut Vec<u8>, x: u64) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn s(b: &mut Vec<u8>, v: &str) {
    u64le(b, v.len() as u64);
    b.extend_from_slice(v.as_bytes());
}
fn kv_str(b: &mut Vec<u8>, k: &str, v: &str) {
    s(b, k);
    u32le(b, 8);
    s(b, v);
}
fn kv_u32(b: &mut Vec<u8>, k: &str, v: u32) {
    s(b, k);
    u32le(b, 4);
    u32le(b, v);
}

/// One F32 tensor-info entry + its data bytes (GGML column-major: dims [ne0, ne1]).
/// Norm vectors use `ne1 == 1` and are written as 1-D `[ne0]` (matching ModelConfig
/// validate's expectation).
fn f32_tensor(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    name: &str,
    ne0: usize,
    ne1: usize,
    vals: &[f32],
) {
    s(meta, name);
    if ne1 == 1 {
        u32le(meta, 1);
        u64le(meta, ne0 as u64);
    } else {
        u32le(meta, 2);
        u64le(meta, ne0 as u64);
        u64le(meta, ne1 as u64);
    }
    u32le(meta, 0);
    u64le(meta, data.len() as u64);
    for v in vals {
        data.extend_from_slice(&v.to_le_bytes());
    }
}

/// Tiny dense qwen2-shaped model: d_model=4, heads=2, kv=2, d_ff=8, layers=2, vocab=5.
/// attn `o` and all ffn projections are ZERO -> every block is residual-only; token 2
/// embeds to [1,2,0,0]; output row 4 [1,1,0,0] dominates -> greedy next token = 4.
fn synth_dense_gguf() -> Vec<u8> {
    const DM: usize = 4;
    const NV: usize = 5;
    const FF: usize = 8;
    const NH: usize = 4;
    const KV: usize = 4;
    let mut meta = Vec::new();
    meta.extend_from_slice(rusty_engine::gguf::GGUF_MAGIC);
    u32le(&mut meta, 3);
    u64le(&mut meta, 0);
    u64le(&mut meta, 7);
    kv_str(&mut meta, "general.architecture", "qwen2");
    kv_u32(&mut meta, "qwen2.embedding_length", DM as u32);
    kv_u32(&mut meta, "qwen2.block_count", 2);
    kv_u32(&mut meta, "qwen2.attention.head_count", 2);
    kv_u32(&mut meta, "qwen2.attention.head_count_kv", 2);
    kv_u32(&mut meta, "qwen2.feed_forward_length", FF as u32);
    kv_u32(&mut meta, "qwen2.vocab_size", NV as u32);

    let mut data = Vec::new();
    let mut n_tensors: u64 = 0;
    let mut emb = [0.0f32; DM * NV];
    emb[2 * DM] = 1.0;
    emb[2 * DM + 1] = 2.0;
    f32_tensor(&mut meta, &mut data, "token_embd.weight", DM, NV, &emb);
    n_tensors += 1;
    let mut out = [0.0f32; DM * NV];
    out[4 * DM] = 1.0;
    out[4 * DM + 1] = 1.0;
    out[0] = 0.1;
    f32_tensor(&mut meta, &mut data, "output.weight", DM, NV, &out);
    n_tensors += 1;
    let ones = [1.0f32; DM];
    f32_tensor(&mut meta, &mut data, "output_norm.weight", DM, 1, &ones);
    n_tensors += 1;
    let q = [0.3f32; DM * NH];
    let k = [0.2f32; DM * KV];
    let v = [0.1f32; DM * KV];
    let zo = [0.0f32; DM * NH];
    let zff = [0.0f32; DM * FF];
    let zdown = [0.0f32; FF * DM];
    for i in 0..2 {
        let p = format!("blk.{i}.");
        f32_tensor(
            &mut meta,
            &mut data,
            &format!("{p}attn_norm.weight"),
            DM,
            1,
            &ones,
        );
        f32_tensor(
            &mut meta,
            &mut data,
            &format!("{p}attn_q.weight"),
            DM,
            NH,
            &q,
        );
        f32_tensor(
            &mut meta,
            &mut data,
            &format!("{p}attn_k.weight"),
            DM,
            KV,
            &k,
        );
        f32_tensor(
            &mut meta,
            &mut data,
            &format!("{p}attn_v.weight"),
            DM,
            KV,
            &v,
        );
        f32_tensor(
            &mut meta,
            &mut data,
            &format!("{p}attn_output.weight"),
            DM,
            NH,
            &zo,
        );
        f32_tensor(
            &mut meta,
            &mut data,
            &format!("{p}ffn_norm.weight"),
            DM,
            1,
            &ones,
        );
        f32_tensor(
            &mut meta,
            &mut data,
            &format!("{p}ffn_gate.weight"),
            DM,
            FF,
            &zff,
        );
        f32_tensor(
            &mut meta,
            &mut data,
            &format!("{p}ffn_up.weight"),
            DM,
            FF,
            &zff,
        );
        f32_tensor(
            &mut meta,
            &mut data,
            &format!("{p}ffn_down.weight"),
            FF,
            DM,
            &zdown,
        );
        n_tensors += 9;
    }

    let patch = n_tensors.to_le_bytes();
    meta[8..16].copy_from_slice(&patch);

    let mut file = meta;
    let aligned = rusty_engine::loader::aligned_start(file.len(), 32);
    file.resize(aligned, 0);
    file.extend_from_slice(&data);
    file
}

fn fixture_path(tag: &str) -> std::path::PathBuf {
    std::path::Path::new("target").join(format!("dense_gen_{tag}.gguf"))
}

#[test]
fn dense_runner_prefills_to_known_greedy_token() {
    let bytes = synth_dense_gguf();
    std::fs::create_dir_all("target").unwrap();
    let path = fixture_path("prefill");
    std::fs::write(&path, &bytes).unwrap();

    let model = unsafe { MmapModel::open(&path) }.expect("mmap open");
    let cfg = ModelConfig::from_file(model.header()).expect("config");
    assert!(cfg.validate(model.header()).is_empty(), "geometry invalid");

    let runner = DenseRunner::from_model(&model, &cfg).expect("runner load");
    assert_eq!(runner.n_layers, 2);
    assert_eq!(runner.n_vocab, 5);

    let logits = runner.prefill_logits(&[2]);
    assert_eq!(logits.len(), 5);
    assert!(logits.iter().all(|v| v.is_finite()), "logits finite");
    assert_eq!(runner.greedy(&[2]), Some(4), "known greedy token");

    let again = runner.prefill_logits(&[2]);
    assert_eq!(logits, again);

    drop(model);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dense_generate_is_greedy_and_bounded() {
    let bytes = synth_dense_gguf();
    std::fs::create_dir_all("target").unwrap();
    let path = fixture_path("generate");
    std::fs::write(&path, &bytes).unwrap();
    let model = unsafe { MmapModel::open(&path) }.expect("mmap open");
    let cfg = ModelConfig::from_file(model.header()).expect("config");
    let runner = DenseRunner::from_model(&model, &cfg).expect("runner");

    let seq = runner.generate(&[2], 3);
    assert_eq!(seq, vec![2, 4, 4, 4]);

    drop(model);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn incremental_decode_matches_full_prefill() {
    // PH-307: causal equivalence — stepping token-by-token with the KV cache yields the
    // exact same greedy stream as re-prefilling the whole prefix each step.
    let bytes = synth_dense_gguf();
    std::fs::create_dir_all("target").unwrap();
    let path = fixture_path("incremental");
    std::fs::write(&path, &bytes).unwrap();
    let model = unsafe { MmapModel::open(&path) }.expect("mmap open");
    let cfg = ModelConfig::from_file(model.header()).expect("config");
    let runner = DenseRunner::from_model(&model, &cfg).expect("runner");

    let full = runner.generate(&[2, 3], 5);
    let mut dec = DenseDecoder::new(&runner);
    let inc = dec.generate_incremental(&[2, 3], 5);
    assert_eq!(full, inc, "incremental == re-prefill greedy stream");

    drop(model);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn generate_with_greedy_opts_matches_generate() {
    // PH-304b: temperature<=0 sampling reduces to argmax => identical to DenseRunner::generate.
    let bytes = synth_dense_gguf();
    std::fs::create_dir_all("target").unwrap();
    let path = fixture_path("sample_greedy");
    std::fs::write(&path, &bytes).unwrap();
    let model = unsafe { MmapModel::open(&path) }.expect("mmap open");
    let cfg = ModelConfig::from_file(model.header()).expect("config");
    let runner = DenseRunner::from_model(&model, &cfg).expect("runner");

    let greedy_opts = GenerationOptions {
        temperature: 0.0,
        ..Default::default()
    };
    let mut rng = SeedRng::new(42);
    let via_sample = runner.generate_with(&[2, 3], &greedy_opts, 5, &mut rng);
    let via_argmax = runner.generate(&[2, 3], 5);
    assert_eq!(via_sample, via_argmax, "greedy sampling == argmax greedy");

    drop(model);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn generate_with_is_deterministic_under_a_seed() {
    let bytes = synth_dense_gguf();
    std::fs::create_dir_all("target").unwrap();
    let path = fixture_path("sample_det");
    std::fs::write(&path, &bytes).unwrap();
    let model = unsafe { MmapModel::open(&path) }.expect("mmap open");
    let cfg = ModelConfig::from_file(model.header()).expect("config");
    let runner = DenseRunner::from_model(&model, &cfg).expect("runner");

    let warm = GenerationOptions {
        temperature: 0.8,
        top_k: 40,
        top_p: 1.0,
        ..Default::default()
    };
    let a = runner.generate_with(&[2, 3], &warm, 6, &mut SeedRng::new(7));
    let b = runner.generate_with(&[2, 3], &warm, 6, &mut SeedRng::new(7));
    assert_eq!(a, b, "same seed => same sampled stream");

    drop(model);
    let _ = std::fs::remove_file(&path);
}

/// Real Qwen2.5-0.5B GGUF load + single prefill (needs the gitignored model file).
/// Run with: `cargo test -- --ignored dense_runner_loads_real_qwen`.
#[test]
#[ignore]
fn dense_runner_loads_real_qwen() {
    let path = Path::new("data/models/qwen2.5-0.5b-q4_k_m.gguf");
    if !path.exists() {
        eprintln!("skip: real model not present at {path:?}");
        return;
    }
    let model = unsafe { MmapModel::open(path) }.expect("open real model");
    let cfg = ModelConfig::from_file(model.header()).expect("real config");
    assert!(
        cfg.validate(model.header()).is_empty(),
        "real geometry invalid"
    );
    let runner = DenseRunner::from_model(&model, &cfg).expect("real runner load");
    assert!(runner.n_layers > 1 && runner.n_vocab > 1000);
    let logits = runner.prefill_logits(&[151644, 872, 198]);
    assert_eq!(logits.len(), runner.n_vocab);
    assert!(logits.iter().all(|v| v.is_finite()), "real logits finite");
}
