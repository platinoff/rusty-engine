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

/// Slice the HTTP response body (everything after the header terminator).
fn http_body(resp: &str) -> &str {
    resp.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("")
}

#[test]
fn serve_with_model_completes_over_the_wire() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    // PH-501b end-to-end: bind the real TextRunner into serve_one, POST a chat body over
    // loopback, and assert an OpenAI completion with the generated assistant text.
    let bytes = synth_text_gguf();
    std::fs::create_dir_all("target").unwrap();
    let path = Path::new("target/text_gen_serve.gguf");
    std::fs::write(path, &bytes).unwrap();

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let fp = path.to_path_buf();
    let th = std::thread::spawn(move || {
        let model = unsafe { MmapModel::open(&fp) }.expect("open in server");
        let tr = TextRunner::from_model(&model).expect("TextRunner in server");
        let mut m = rusty_engine::hub::Metrics::zero();
        rusty_engine::serve::serve_one(&listener, &mut m, Some(&tr)).expect("serve_one+model");
    });

    // Declare `Content-Length` so the serve loop reads the body via the header (PH-507),
    // not just whatever happened to arrive in the first segment.
    let req_body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        req_body.len()
    );
    let mut client = TcpStream::connect(addr).expect("connect");
    client.write_all(head.as_bytes()).expect("write head");
    client.write_all(req_body.as_bytes()).expect("write body");
    client.flush().expect("flush");
    let mut resp = String::new();
    client.read_to_string(&mut resp).expect("read");
    th.join().expect("server thread");

    assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"), "{resp}");
    // PH-507 regression: the body must be a *complete* JSON object. Substring checks alone
    // let the missing-root-`}` bug through (curl tolerated it, the GSV hub's `serde_json`
    // did not, surfacing as a null proxy reply) — parse the body strictly instead.
    let body = http_body(&resp);
    let json = rusty_engine::worker::parse_json(body)
        .unwrap_or_else(|e| panic!("response body is not valid JSON: {e}\n{body}"));
    assert_eq!(
        json.get("object").and_then(|v| v.as_str()),
        Some("chat.completion")
    );
    assert_eq!(
        json.get("model").and_then(|v| v.as_str()),
        Some("rusty-engine")
    );
    let content = json
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .expect("choices[0].message.content");
    // monotone synth model generates '~' (token 126) greedily after "hi"
    assert!(content.contains("~~~"), "generated text present: {content}");
    let finish = json
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str());
    assert_eq!(finish, Some("stop"));
    let total = json
        .get("usage")
        .and_then(|u| u.get("total_tokens"))
        .and_then(|v| v.as_u64());
    assert!(
        total.is_some_and(|t| t > 0),
        "usage.total_tokens: {total:?}"
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn chat_response_reports_decode_stats() {
    // PH-505: chat_response returns the response *plus* the decode stats the serve loop
    // feeds into /metrics (tokens + wall clock), instead of hardcoded zeros.
    let bytes = synth_text_gguf();
    std::fs::create_dir_all("target").unwrap();
    let path = Path::new("target/text_gen_stats.gguf");
    std::fs::write(path, &bytes).unwrap();
    let model = unsafe { MmapModel::open(path) }.expect("open synth");
    let tr = TextRunner::from_model(&model).expect("TextRunner");

    let out = rusty_engine::textgen::chat_response(
        &tr,
        r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        3,
    );
    assert_eq!(out.response.status, 200);
    assert!(
        out.generated_tokens >= 1,
        "generated tokens counted: {}",
        out.generated_tokens
    );
    assert!(out.decode_secs >= 0.0);
    let json = rusty_engine::worker::parse_json(&out.response.body).expect("valid JSON");
    assert_eq!(
        json.get("usage")
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_u64()),
        Some(out.generated_tokens),
        "usage.completion_tokens mirrors the reported stat"
    );

    drop(model);
    let _ = std::fs::remove_file(path);
}

#[test]
fn serve_with_model_streams_sse_over_the_wire() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    // PH-507: stream:true on the real-model path returns text/event-stream with a
    // chat.completion.chunk delta and the terminating [DONE] — the bunke-rock
    // streaming contract the GSV hub can mirror for rusty/*.
    let bytes = synth_text_gguf();
    std::fs::create_dir_all("target").unwrap();
    let path = Path::new("target/text_gen_serve_sse.gguf");
    std::fs::write(path, &bytes).unwrap();

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let fp = path.to_path_buf();
    let th = std::thread::spawn(move || {
        let model = unsafe { MmapModel::open(&fp) }.expect("open in server");
        let tr = TextRunner::from_model(&model).expect("TextRunner in server");
        let mut m = rusty_engine::hub::Metrics::zero();
        rusty_engine::serve::serve_one(&listener, &mut m, Some(&tr)).expect("serve_one+model");
    });

    let mut client = TcpStream::connect(addr).expect("connect");
    let req_body = r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        req_body.len()
    );
    client.write_all(head.as_bytes()).expect("write head");
    client.write_all(req_body.as_bytes()).expect("write body");
    client.flush().expect("flush");
    let mut resp = String::new();
    client.read_to_string(&mut resp).expect("read");
    th.join().expect("server thread");

    assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"), "{resp}");
    assert!(
        resp.contains("Content-Type: text/event-stream\r\n"),
        "event-stream content type: {resp}"
    );
    assert!(resp.contains("chat.completion.chunk"));
    assert!(
        resp.ends_with("data: [DONE]\n\n"),
        "terminated by DONE: {resp}"
    );
    // PH-507 regression for the stream path too: every `data:` chunk is itself a complete
    // JSON object (a missing root `}` here would break streaming consumers).
    let mut chunks = 0;
    for line in resp.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            continue;
        }
        rusty_engine::worker::parse_json(payload)
            .unwrap_or_else(|e| panic!("SSE chunk is not valid JSON: {e}\n{payload}"));
        chunks += 1;
    }
    assert!(chunks >= 2, "expected delta + finish chunks, got {chunks}");

    let _ = std::fs::remove_file(path);
}
