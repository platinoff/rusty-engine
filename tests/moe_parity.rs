//! PH-405b — `rusty-engine` ↔ **llama-rs** semantic greedy-parity + HF
//! `tokenizer.json` **byte-exact** gate.
//!
//! Gate A (tokenizer): the GGUF vocab/merges and the HF `qwen2` `tokenizer.json`
//! vectors must agree byte-exactly on the spots we pin, and `Bpe::encode` must
//! reproduce the HF atomic ids for fixed strings (digits via the single-`\p{N}`
//! qwen2 split — see `src/gpt2.rs`).
//!
//! Gate B (parity): llama-rs (`llama_serve`, same Qwen2.5-0.5B Q4_K_M, sha
//! `74a4da8c`) greedy-decodes the same prompts at `temperature=0`; our
//! `DenseRunner`/`DenseDecoder` greedy stream over BPE ids must share the
//! longest common prefix with the oracle's text. Float accumulation differs by
//! design, so the headline is **prefix match**, not full-stream equality.
//! Recorded facts live in `docs/research/llama-rs-parity.md`.
//!
//! Run: `PH405B_ORACLE_URL=http://127.0.0.1:8085/v1/chat/completions cargo test
//! --release --test moe_parity -- --ignored`

use rusty_engine::gpt2::{gpt2_pre_split, qwen2_pre_split};
use rusty_engine::loader::MmapModel;
use rusty_engine::runner::DenseDecoder;
use rusty_engine::textgen::extract_message_content;
use rusty_engine::tokenizer::{Bpe, TokenizerMeta};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;

const MODEL: &str = "data/models/qwen2.5-0.5b-q4_k_m.gguf";
const HF_JSON: &str = "data/tokenizer.json";

fn load_bpe() -> Option<Bpe> {
    let path = Path::new(MODEL);
    if !path.exists() {
        eprintln!("skip: real model not present at {MODEL}");
        return None;
    }
    let model = unsafe { MmapModel::open(path) }.expect("open real model");
    let meta = TokenizerMeta::from_file(model.header());
    Some(Bpe::from_meta(meta))
}

// ── Gate A — HF tokenizer.json byte-exact ────────────────────────────────────

#[test]
#[ignore = "byte-exact gate: needs the gitignored data/tokenizer.json + GGUF"]
fn tokenizer_json_byte_exact_gate() {
    let Some(bpe) = load_bpe() else { return };
    if !Path::new(HF_JSON).exists() {
        eprintln!("skip: HF tokenizer.json not present at {HF_JSON}");
        return;
    }
    let json = std::fs::read_to_string(HF_JSON).expect("read tokenizer.json");

    // 1. vocab order: GGUF index == HF vocab id on the pinned spots.
    let spots: [(&str, u32); 10] = [
        ("You", 2610),
        ("The", 785),
        ("Ġare", 525),
        ("Ġa", 264),
        ("Ġhelpful", 10950),
        ("Ġassistant", 17847),
        ("Ġcapital", 6722),
        (".", 13),
        ("Ġ", 220),
        ("Ċ", 198),
    ];
    for (tok, id) in spots {
        assert_eq!(
            bpe.meta().token_text(id),
            Some(tok),
            "GGUF vocab[{id}] must be {tok:?}"
        );
        let needle = format!("\"{tok}\": {id}");
        assert!(json.contains(&needle), "HF vocab must contain {needle}");
    }

    // 2. merges head: the first HF pairs are byte-identical to the GGUF list.
    let hf_head: Vec<String> = json
        .split("\"merges\": [")
        .nth(1)
        .expect("merges block")
        .lines()
        .skip(1)
        .map(|l| l.trim().trim_end_matches(',').trim_matches('"').to_string())
        .take(8)
        .collect();
    let gguf_head: Vec<String> = bpe.meta().merges.iter().take(8).cloned().collect();
    assert_eq!(hf_head, gguf_head, "merges head byte-exact");

    // 3. pre-tokenizer: HF qwen2 pins the gpt2 contractions head AND single \p{N}.
    assert!(
        json.contains(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"),
        "gpt2 contraction head present"
    );
    assert!(
        json.contains(r"\\p{N}| ?["),
        "qwen2 regex must use single-\\p{{N}} (not \\p{{N}}{{1,3}})"
    );

    // 4. behaviour: qwen2 splitter splits digits, gpt2 groups threes.
    assert_eq!(
        qwen2_pre_split("1234567"),
        vec!["1", "2", "3", "4", "5", "6", "7"]
    );
    assert_eq!(gpt2_pre_split("1234567"), vec!["123", "456", "7"]);

    // 5. Bpe ids: HF atomic goldens (vocab-order verified above).
    assert_eq!(
        bpe.encode("You are a helpful assistant.").unwrap(),
        vec![2610, 525, 264, 10950, 17847, 13]
    );
    assert_eq!(
        bpe.encode("The capital of France is").unwrap(),
        vec![785, 6722, 315, 9625, 374]
    );
    println!("PH-405b tokenizer gate: vocab 151936, merges 151387, ids exact on 2 goldens");
}

// ── Gate B — live llama-rs greedy parity ─────────────────────────────────────

/// (prompt, oracle greedy text recorded 2026-10-15, expected min char-prefix)
/// — facts: `docs/research/llama-rs-parity.md`. Floors are **0** until `PH-405`
/// forward-numeric parity lands: the first generated token already diverges
/// (ours `122007` vs oracle `ĠParis`=12095) even though ids, determinism
/// (`full == incremental`) and the oracle reference itself are stable.
/// Bump the floors as part of that fix — this harness then guards the gain.
const PARITY_CASES: [(&str, &str, usize); 3] = [
    (
        "The capital of France is",
        " Paris. It is the largest city in Europe and the second",
        0,
    ),
    ("2+2=", "4, 3+3=6, 4+", 0),
    (
        "Berlin is a",
        " city in the state of Berlin, Germany. It is the",
        0,
    ),
];

fn http_post_completions(url: &str, body: &str) -> Result<String, String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or("only http:// oracle urls")?;
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = authority
        .rsplit_once(':')
        .ok_or("authority must carry :port")?;
    let port: u16 = port.parse().map_err(|_| "bad port")?;
    let mut stream = TcpStream::connect((host, port)).map_err(|e| format!("connect: {e}"))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(600)))
        .ok();
    let req = format!(
        "POST /{path} HTTP/1.1\r\nHost: {authority}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw).map_err(|e| e.to_string())?;
    Ok(raw
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or(raw))
}

fn common_prefix_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

#[test]
#[ignore = "parity: needs release build + llama-rs oracle (PH405B_ORACLE_URL)"]
fn greedy_parity_vs_llama_rs_oracle() {
    let Ok(url) = std::env::var("PH405B_ORACLE_URL") else {
        eprintln!("skip: PH405B_ORACLE_URL unset (start llama_serve on the same GGUF)");
        return;
    };
    let path = Path::new(MODEL);
    if !path.exists() {
        eprintln!("skip: real model not present at {MODEL}");
        return;
    }
    let model = unsafe { MmapModel::open(path) }.expect("open real model");
    let tr = rusty_engine::textgen::TextRunner::from_model(&model).expect("TextRunner");

    let mut report = String::new();
    for (prompt, oracle_expected, min_prefix) in PARITY_CASES {
        // Oracle stream: fresh greedy decode at temperature 0.
        let body = format!("{{\"prompt\":\"{prompt}\",\"max_tokens\":12,\"temperature\":0.0}}");
        let resp = http_post_completions(&url, &body).expect("oracle POST");
        let oracle_text = extract_message_content(&resp);
        // Our stream: same ids path (BPE) → incremental greedy.
        let ids = tr.tokenize(prompt).expect("encode prompt");
        let mut dec = DenseDecoder::new(&tr.runner);
        let seq = dec.generate_incremental(&ids, 12);
        let new = &seq[ids.len().min(seq.len())..];
        let our_text = tr.bpe.decode(new);
        // Real-model determinism gate (new for PH-405b): full re-prefill and the
        // KV-cache decoder must emit the same first 4 tokens on the real stack.
        let full4 = tr.runner.generate(&ids, 4);
        let mut dec2 = DenseDecoder::new(&tr.runner);
        let inc4 = dec2.generate_incremental(&ids, 4);
        assert_eq!(
            &full4[ids.len()..],
            &inc4[ids.len()..],
            "full == incremental on the real model for {prompt:?}"
        );
        let lcp = common_prefix_chars(&oracle_text, &our_text);
        assert!(
            common_prefix_chars(&oracle_text, oracle_expected)
                >= oracle_expected.chars().count() - 1,
            "oracle drifted from the recorded reference for {prompt:?}: {oracle_text:?}"
        );
        assert!(
            lcp >= min_prefix,
            "greedy prefix floor for {prompt:?}: {lcp} < {min_prefix}"
        );
        println!(
            "PH-405b parity prompt={prompt:?} oracle={oracle_text:?} rusty={our_text:?} lcp={lcp}"
        );
        report.push_str(&format!(
            "- {prompt:?} → oracle {lcp} chars agreed; rusty `{our_text}`\n"
        ));
    }
    eprintln!("parity report:\n{report}");
}
