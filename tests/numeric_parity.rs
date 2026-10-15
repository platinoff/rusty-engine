//! B4 `PH-405` — numeric regression gate.
//!
//! A **deterministic greedy fingerprint** of the whole real-model stack on the gitignored
//! Qwen2.5-0.5B GGUF (`config → validate → dequant → embed → 24× block_forward_seq → argmax`
//! sampled repeatedly). The forward is pure `f32`, so the generated token stream is stable
//! across runs on the same binary — freezing it in `tests/data/qwen25_golden.txt` turns this
//! into a regression guard for the quant/forward/KV/sampler path: any numeric slip changes
//! the continuation and fails the test.
//!
//! - **default**: verify — needs the model; if `data/models/*.gguf` is absent (CI without the
//!   gitignored 491 MB download) the test **skips green**.
//! - `RUSTY_CAPTURE_GOLDEN=1`: regenerate the golden (run once locally with the model).
//!
//! This is a *fingerprint*, not an HF-semantic token check — that equality is the separate
//! `qwen2` pre-regex parity slice.
//!
//! Marked `#[ignore]`: the real-model pass needs the 491 MB download and a release build
//! (~60 s), so it is an explicit numeric-parity job — `cargo test --release -- --ignored`.
//! The default `cargo test` suite stays fast; the no-model CI path would also skip it.

use std::path::Path;

use rusty_engine::loader::MmapModel;
use rusty_engine::model::ModelConfig;
use rusty_engine::runner::DenseRunner;

const MODEL: &str = "data/models/qwen2.5-0.5b-q4_k_m.gguf";
const GOLDEN: &str = "tests/data/qwen25_golden.txt";
const PROMPT: &[u32] = &[151644, 872, 198, 3837, 1059, 8701]; // fixed seed ids
const GEN: usize = 16;

/// Greedy continuation fingerprint for the fixed prompt.
fn fingerprint(model_path: &Path) -> Vec<u32> {
    let m = unsafe { MmapModel::open(model_path) }.expect("open model");
    let cfg = ModelConfig::from_file(m.header()).expect("config");
    assert!(cfg.validate(m.header()).is_empty(), "real geometry invalid");
    let runner = DenseRunner::from_model(&m, &cfg).expect("runner");
    let out = runner.generate(PROMPT, GEN);
    out[PROMPT.len()..].to_vec()
}

fn encode(prompt: &[u32], gen: &[u32]) -> String {
    let nums = |v: &[u32]| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!(
        "# rusty-engine PH-405 greedy fingerprint (prompt {GEN}-token continuation)\nprompt: {}\ngen: {}\n",
        nums(prompt),
        nums(gen)
    )
}

fn decode(s: &str) -> (Vec<u32>, Vec<u32>) {
    let mut prompt = Vec::new();
    let mut gen = Vec::new();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("prompt:") {
            prompt = rest
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
        } else if let Some(rest) = line.strip_prefix("gen:") {
            gen = rest
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
        }
    }
    (prompt, gen)
}

#[test]
#[ignore = "numeric parity: needs the gitignored real model + release build (~60s); run: cargo test --release -- --ignored"]
fn qwen25_greedy_fingerprint_is_stable() {
    let model = Path::new(MODEL);
    if !model.exists() {
        eprintln!("PH-405 skip: real model absent at {MODEL} (gitignored download)");
        return;
    }
    let gen = fingerprint(model);

    if std::env::var("RUSTY_CAPTURE_GOLDEN").is_ok() {
        std::fs::create_dir_all("tests/data").expect("mkdir");
        std::fs::write(GOLDEN, encode(PROMPT, &gen)).expect("write golden");
        eprintln!("PH-405: captured golden -> {GOLDEN}");
        return;
    }

    let golden_path = Path::new(GOLDEN);
    if !golden_path.exists() {
        panic!("PH-405: model present but {GOLDEN} missing — run with RUSTY_CAPTURE_GOLDEN=1");
    }
    let (gp, gg) = decode(&std::fs::read_to_string(golden_path).expect("read golden"));
    assert_eq!(gp, PROMPT.to_vec(), "golden prompt must match");
    assert_eq!(
        gen, gg,
        "PH-405 numeric regression: greedy continuation drifted from the golden fingerprint"
    );
}
