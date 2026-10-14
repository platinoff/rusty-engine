//! B3 `PH-306b` — text pipeline integration test on a synthetic tiny GGUF carrying a
//! byte-level vocab (all 127 ASCII bytes as their own tokens). Zeroed attention/FFN
//! branches keep the forward path residual-only, so the greedy argmax over the crafted
//! `output` rows is a stable, hand-verifiable function of the last prompt token.

use std::path::Path;

use rusty_engine::loader::MmapModel;
use rusty_engine::textgen::TextRunner;
use rusty_engine::tokenizer::ByteEncoder;

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
fn kv_arr_str(b: &mut Vec<u8>, k: &str, items: &[String]) {
    s(b, k);
    u32le(b, 9);
    u32le(b, 8); // Str inner kind
    u64le(b, items.len() as u64);
    for it in items {
        s(b, it);
    }
}
fn kv_arr_i32(b: &mut Vec<u8>, k: &str, items: &[i32]) {
    s(b, k);
    u32le(b, 9);
    u32le(b, 5); // I32 inner kind
    u64le(b, items.len() as u64);
    for &v in items {
        u32le(b, v as u32);
    }
}

/// One F32 tensor-info entry + its data bytes (GGML column-major: dims [ne0, ne1]).
/// 1-D when ne1 == 1 (norms).
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

/// 1-layer tiny qwen2 model with byte-level vocab (all 127 ASCII chars as their own
/// tokens). Zeroed attn `o` and FFN projections keep blocks residual-only. Output rows
/// are `[t*0.01, 0, 0, 0]` → argmax = 126 (byte 0x7E '~') for any 1-D prompt.
fn synth_text_gguf() -> Vec<u8> {
    const DM: usize = 4;
    const FF: usize = 8;
    const NV: usize = 127;
    let encoder = ByteEncoder::new();
    let byte_tokens: Vec<String> = (0..=126u8).map(|b| encoder.bytes_to_chars(&[b])).collect();
    let token_type: Vec<i32> = vec![1; NV]; // NORMAL

    let mut meta = Vec::new();
    meta.extend_from_slice(rusty_engine::gguf::GGUF_MAGIC);
    u32le(&mut meta, 3); // version
    u64le(&mut meta, 0); // n_tensors (patched below)
    u64le(&mut meta, 11); // n_kv
    kv_str(&mut meta, "general.architecture", "qwen2");
    kv_u32(&mut meta, "qwen2.embedding_length", DM as u32);
    kv_u32(&mut meta, "qwen2.block_count", 1);
    kv_u32(&mut meta, "qwen2.attention.head_count", 2);
    kv_u32(&mut meta, "qwen2.attention.head_count_kv", 2);
    kv_u32(&mut meta, "qwen2.feed_forward_length", FF as u32);
    kv_u32(&mut meta, "qwen2.vocab_size", NV as u32);
    kv_str(&mut meta, "tokenizer.ggml.model", "gpt2");
    kv_arr_str(&mut meta, "tokenizer.ggml.tokens", &byte_tokens);
    kv_arr_i32(&mut meta, "tokenizer.ggml.token_type", &token_type);
    // pre-tokenizer hint (unused by TextRunner, present for completeness)
    kv_str(&mut meta, "tokenizer.ggml.pre", "qwen2");

    let mut data = Vec::new();
    let mut n_tensors: u64 = 0;
    let mut embed = vec![0.0f32; DM * NV];
    for t in 0..NV {
        embed[t * DM] = t as f32 * 0.01;
    }
    f32_tensor(&mut meta, &mut data, "token_embd.weight", DM, NV, &embed);
    n_tensors += 1;
    let mut out = vec![0.0f32; DM * NV];
    for t in 0..NV {
        out[t * DM] = t as f32 * 0.01;
    }
    f32_tensor(&mut meta, &mut data, "output.weight", DM, NV, &out);
    n_tensors += 1;
    let ones = [1.0f32; DM];
    f32_tensor(&mut meta, &mut data, "output_norm.weight", DM, 1, &ones);
    n_tensors += 1;
    f32_tensor(&mut meta, &mut data, "blk.0.attn_norm.weight", DM, 1, &ones);
    let q = [0.3f32; DM * 4];
    let k = [0.2f32; DM * 4];
    let v = [0.1f32; DM * 4];
    let z4 = [0.0f32; DM * 4];
    let zff = [0.0f32; DM * FF];
    let zdown = [0.0f32; FF * DM];
    f32_tensor(&mut meta, &mut data, "blk.0.attn_q.weight", DM, 4, &q);
    f32_tensor(&mut meta, &mut data, "blk.0.attn_k.weight", DM, 4, &k);
    f32_tensor(&mut meta, &mut data, "blk.0.attn_v.weight", DM, 4, &v);
    f32_tensor(&mut meta, &mut data, "blk.0.attn_output.weight", DM, 4, &z4);
    f32_tensor(&mut meta, &mut data, "blk.0.ffn_norm.weight", DM, 1, &ones);
    f32_tensor(&mut meta, &mut data, "blk.0.ffn_gate.weight", DM, FF, &zff);
    f32_tensor(&mut meta, &mut data, "blk.0.ffn_up.weight", DM, FF, &zff);
    f32_tensor(
        &mut meta,
        &mut data,
        "blk.0.ffn_down.weight",
        FF,
        DM,
        &zdown,
    );
    n_tensors += 9;

    let patch = n_tensors.to_le_bytes();
    meta[8..16].copy_from_slice(&patch);

    let mut file = meta;
    let aligned = rusty_engine::loader::aligned_start(file.len(), 32);
    file.resize(aligned, 0);
    file.extend_from_slice(&data);
    file
}

#[test]
fn text_pipeline_roundtrips_prompt_and_generates() {
    let bytes = synth_text_gguf();
    std::fs::create_dir_all("target").unwrap();
    let path = Path::new("target/text_gen_fixture.gguf");
    std::fs::write(path, &bytes).unwrap();

    let model = unsafe { MmapModel::open(path) }.expect("open synth");
    let tr = TextRunner::from_model(&model).expect("TextRunner build");
    assert_eq!(tr.runner.n_layers, 1);
    assert_eq!(tr.runner.n_vocab, 127);

    // "hi" -> gpt2 split -> ["hi"] -> bytes_to_chars -> "hi" -> vocab tokens 104, 105.
    let ids = tr.tokenize("hi").expect("tokenize");
    assert_eq!(ids, vec![104, 105]);

    // Greedy argmax on residual-only forward is stable: hidden row = embed[105],
    // output row t = [t*0.01, 0, 0, 0] → dot with rmsnorm([1.05,0,0,0]) is monotone
    // in t → argmax = 126 → '~'. Repeat → '~' forever.
    let out = tr.run_text("hi", 3).expect("run_text");
    assert_eq!(out, "~~~", "greedy on a monotone output matrix is stable");

    // Determinism: same prompt -> same generated text.
    let again = tr.run_text("hi", 3).expect("run_text 2");
    assert_eq!(out, again);

    // Empty-prompt path errors.
    assert!(tr.run_text("", 1).is_err());

    drop(model);
    let _ = std::fs::remove_file(path);
}

#[test]
fn text_pipeline_roundtrips_symbol_text() {
    let bytes = synth_text_gguf();
    let path = Path::new("target/text_gen_specials.gguf");
    std::fs::write(path, &bytes).unwrap();
    let model = unsafe { MmapModel::open(path) }.expect("open synth");
    let tr = TextRunner::from_model(&model).expect("TextRunner");
    // byte-level vocab covers every ASCII char, so symbols round-trip losslessly.
    let text = "a<|x|>b";
    let ids = tr.tokenize(text).expect("tokenize symbols");
    assert!(!ids.is_empty());
    assert_eq!(tr.bpe.decode(&ids), text);
    drop(model);
    let _ = std::fs::remove_file(path);
}
