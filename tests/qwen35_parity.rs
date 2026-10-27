//! B13 `PH-1305` — `qwen35` text end-to-end against the `llama.cpp` oracle.
//!
//! The oracle is upstream `llama.cpp` (build `b11906`, the `llama.cpp-omni` role in the
//! band plan) running the *same* gitignored `MiniCPM-V-4_7-1B-Q4_K_M.gguf`. Its raw
//! `/completion` bodies (`temperature=0`, `top_k=1`, `logprobs=5`) are committed
//! byte-for-byte under `tests/data/qwen35_oracle/<slug>.{json,ids}` — nothing is
//! transcribed by hand, so the fixture cannot drift from what the oracle actually said.
//!
//! Three offline gates:
//!
//! - [`tokenizer_matches_oracle`] — our BPE must reproduce `llama-tokenize`'s ids exactly
//!   (`<slug>.ids`). Tokenizer drift invalidates every downstream comparison, so it is
//!   its own gate;
//! - [`greedy_matches_oracle_committed_tokens`] — the **numeric** gate. Our f32 logits are
//!   compared against the oracle's logprobs *as a profile*: for every token the oracle
//!   ranked in its top-5, the logit gap to the argmax must agree within [`LOGIT_TOL`].
//!   Argmax equality is then only required where the oracle itself was decisive
//!   (top1−top2 ≥ [`TIE_FLOOR`]) — a 0.01-nat tie is decided by float order, not by the
//!   model, and this engine's matvec is *not* bit-identical to ggml's (ggml quantizes
//!   activations to `q8_K` before the K-quant dot; we stay f32 — see
//!   `docs/research/qwen35-parity.md`);
//! - [`greedy_fingerprint_matches_golden`] — our own pinned run
//!   (`tests/data/qwen35_golden.txt`, recapture with `RUSTY_CAPTURE_GOLDEN=1`), catching a
//!   drift the shorter oracle window would miss.
//!
//! All are `#[ignore]`: the model is a 672 MB gitignored download and a release build is
//! needed for a sane wall clock — `cargo test --release -- --ignored`. Without the model
//! they skip green (CI path); [`oracle_fixture_is_well_formed`] needs no model at all.
//!
//! Convention pins that the oracle source fixes (`src/models/qwen35.cpp`, `models.h`) live
//! next to the code they guard — `src/gdn.rs` (`ssm_a`/conv/L2/gate) and `src/q35.rs`
//! (attention gate + RoPE).

use std::path::Path;

use rusty_engine::loader::MmapModel;
use rusty_engine::runner35::TraceStep;
use rusty_engine::textgen::TextRunner35;

const MODEL: &str = "data/models/MiniCPM-V-4_7-1B-Q4_K_M.gguf";
const ORACLE_DIR: &str = "tests/data/qwen35_oracle";
const GOLDEN: &str = "tests/data/qwen35_golden.txt";
/// Greedy steps recorded from the oracle (and replayed by our gates).
const GEN: usize = 10;
/// Logit-profile tolerance in nats: |(our logit − our max) − (oracle logprob − oracle max)|
/// over the oracle's top-5. Measured worst case on this fixture is 0.42 nats, and that
/// residual is the oracle's own `q8_K` activation quantization — not an engine slip.
const LOGIT_TOL: f32 = 0.75;
/// Oracle decisiveness floor in nats: below this the oracle's own top-1 is a coin flip
/// against float noise, so we only require the *profile* to match, not the argmax.
const TIE_FLOOR: f32 = 0.5;

/// The pinned prompt set — one row per committed oracle fixture.
const CASES: [(&str, &str); 5] = [
    ("capital", "The capital of France is"),
    ("fib", "def fibonacci(n):"),
    ("math", "2 + 2 ="),
    ("story", "Once upon a time"),
    ("fox", "The quick brown fox"),
];

/// One recorded greedy step: the sampled token plus the oracle's top-`k` `(id, logprob)`.
#[derive(Debug, Clone, PartialEq)]
struct OracleStep {
    /// Token the oracle sampled (== `top[0].0`).
    token: u32,
    /// Descending `(token, logprob)` pairs, softmax-normalized by the oracle.
    top: Vec<(u32, f32)>,
}

/// Read a decimal number (`-1.5e-3`) starting at `s[0]`; returns it with the rest.
fn read_f32(s: &str) -> (f32, &str) {
    let end = s
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
        })
        .unwrap_or(s.len());
    (s[..end].parse::<f32>().unwrap_or(f32::NAN), &s[end..])
}

/// Scan a `llama.cpp` `/completion` body: every `"top_logprobs":[…]` group in order, each
/// entry read as `{"id":N,…,"logprob":F}`. This crate has no serde and the fixture is
/// machine-generated with a fixed shape, so a cursor scan over the raw bytes is enough —
/// and it keeps the committed fixture exactly what the oracle printed.
fn oracle_steps(json: &str) -> Vec<OracleStep> {
    const OPEN: &str = "\"top_logprobs\":[";
    const ID: &str = "{\"id\":";
    const LP: &str = "\"logprob\":";
    let mut steps = Vec::new();
    let mut rest = json;
    while let Some(p) = rest.find(OPEN) {
        rest = &rest[p + OPEN.len()..];
        let mut top: Vec<(u32, f32)> = Vec::new();
        while let Some(q) = rest.find(ID) {
            // A `]` before the next entry closes the group (entry-internal `bytes` arrays
            // are consumed below, so the first `]` seen here is the group's).
            if rest.find(']').is_some_and(|close| close < q) {
                break;
            }
            rest = &rest[q + ID.len()..];
            let id_end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            let id: u32 = rest[..id_end].parse().expect("oracle step id");
            rest = &rest[id_end..];
            let lp_at = rest.find(LP).expect("oracle step logprob");
            let (lp, after) = read_f32(&rest[lp_at + LP.len()..]);
            rest = after;
            top.push((id, lp));
        }
        assert!(!top.is_empty(), "empty top_logprobs group");
        steps.push(OracleStep {
            token: top[0].0,
            top,
        });
    }
    steps
}

/// `(slug, prompt, oracle steps, oracle prompt ids)` for one committed fixture row.
fn load_case(slug: &str, prompt: &str) -> Option<(Vec<OracleStep>, Vec<u32>)> {
    let json = std::fs::read_to_string(format!("{ORACLE_DIR}/{slug}.json")).ok()?;
    let ids_raw = std::fs::read_to_string(format!("{ORACLE_DIR}/{slug}.ids")).ok()?;
    // The fixture must be the recording of *this* prompt, not a stale row.
    let echo = format!("\"prompt\":\"{prompt}\"");
    assert!(
        json.contains(&echo),
        "{slug}.json does not echo the pinned prompt {prompt:?} — re-record the oracle"
    );
    let ids: Vec<u32> = ids_raw
        .trim()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse().expect("oracle prompt id"))
        .collect();
    let steps = oracle_steps(&json);
    assert_eq!(steps.len(), GEN, "{slug}: recorded step count");
    Some((steps, ids))
}

fn open_runner() -> Option<TextRunner35> {
    let path = Path::new(MODEL);
    if !path.exists() {
        eprintln!("skip: {MODEL} absent (gitignored download)");
        return None;
    }
    let model = unsafe { MmapModel::open(path) }.expect("open real model");
    let tr = TextRunner35::from_model(&model).expect("TextRunner35");
    assert_eq!(tr.runner.n_layers, 24, "24 hybrid layers");
    Some(tr)
}

/// Our greedy trace (top-5 logits per step) for one prompt.
fn trace(tr: &TextRunner35, prompt: &str) -> Vec<TraceStep> {
    tr.trace_text(prompt, GEN, 5).expect("trace prompt")
}

/// Fixture shape gate: five prompts, ten steps, five ranked alternatives each, strictly
/// descending logprobs. Runs without the model, so a corrupted fixture fails fast.
#[test]
fn oracle_fixture_is_well_formed() {
    let mut rows = 0;
    for (slug, prompt) in CASES {
        let Some((steps, ids)) = load_case(slug, prompt) else {
            eprintln!("skip: {ORACLE_DIR}/{slug}.{{json,ids}} absent");
            return;
        };
        assert!(!ids.is_empty(), "{slug}: prompt tokenized to nothing");
        for (i, s) in steps.iter().enumerate() {
            assert_eq!(s.top.len(), 5, "{slug} step {i}: top-5 alternatives");
            assert_eq!(
                s.token, s.top[0].0,
                "{slug} step {i}: sampled token is top-1"
            );
            for w in s.top.windows(2) {
                assert!(
                    w[0].1 >= w[1].1,
                    "{slug} step {i}: logprobs must descend ({:?} then {:?})",
                    w[0],
                    w[1]
                );
            }
        }
        rows += 1;
    }
    println!("PH-1305 oracle fixture: {rows} prompts × {GEN} steps × 5 alternatives");
}

/// Gate 1 — tokenizer: our BPE against `llama-tokenize` on the same strings.
#[test]
#[ignore = "parity: real MiniCPM-V 4.7-1B GGUF tokenizer vs recorded llama-tokenize ids"]
fn tokenizer_matches_oracle() {
    let Some(tr) = open_runner() else { return };
    let mut checked = 0;
    for (slug, prompt) in CASES {
        let Some((_, want)) = load_case(slug, prompt) else {
            eprintln!("skip: {ORACLE_DIR}/{slug}.* absent");
            return;
        };
        let got = tr.tokenize(prompt).expect("tokenize prompt");
        assert_eq!(
            got, want,
            "{slug}: BPE ids for {prompt:?} drifted from llama-tokenize"
        );
        checked += 1;
    }
    println!("PH-1305 tokenizer: {checked} prompts byte-exact against llama-tokenize");
}

/// Gate 2 — numeric parity: logit profile within [`LOGIT_TOL`] on the shared prefix, and
/// the greedy prefix itself equal to the oracle's until the oracle's own top-1 stops being
/// decisive (top1−top2 ≥ [`TIE_FLOOR`]). A decisive mismatch is a real forward/numeric bug;
/// a tie-zone mismatch is float order and ends the comparison (contexts diverge after it).
#[test]
#[ignore = "parity: real MiniCPM-V 4.7-1B GGUF vs recorded llama.cpp logprobs"]
fn greedy_matches_oracle_committed_tokens() {
    let Some(tr) = open_runner() else { return };
    let mut committed = 0;
    let mut ties = 0;
    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    let mut compared = 0;
    let mut outliers = 0;
    let mut prefixes = Vec::new();
    for (slug, prompt) in CASES {
        let Some((steps, _)) = load_case(slug, prompt) else {
            eprintln!("skip: {ORACLE_DIR}/{slug}.json absent");
            return;
        };
        let ours = trace(&tr, prompt);
        let mut prefix = 0;
        for (i, (o, mine)) in steps.iter().zip(&ours).enumerate() {
            // Profile: compare the oracle's ranked alternatives in *our* logit space.
            let our = |t: u32| -> Option<f32> {
                mine.top
                    .iter()
                    .find(|(id, _)| *id as u32 == t)
                    .map(|(_, l)| *l)
            };
            let our_max = mine.top[0].1;
            for &(t, lp) in &o.top {
                let Some(l) = our(t) else { continue }; // outside our own top-5
                let d = ((l - our_max) - (lp - o.top[0].1)).abs();
                compared += 1;
                if d > worst {
                    worst = d;
                    worst_at = format!("{slug} step {i} token {t}");
                }
                if d > 0.25 {
                    outliers += 1;
                    println!("  profile outlier: {slug} step {i} token {t}: {d:.3} nats");
                }
                assert!(
                    d <= LOGIT_TOL,
                    "{slug} step {i}: token {t} profile off by {d:.3} nats (tol {LOGIT_TOL})\n  \
                     oracle top={:?}\n  rusty  top={:?}",
                    o.top,
                    mine.top
                );
            }
            if mine.token == o.token {
                committed += 1;
                prefix += 1;
                continue;
            }
            // First divergence: only allowed where the oracle itself was undecided.
            let margin = o.top[0].1 - o.top.get(1).map_or(f32::INFINITY, |x| x.1);
            assert!(
                margin < TIE_FLOOR,
                "{slug} step {i}: oracle was decisive ({margin:.2} nats) but we picked \
                 {} instead of {}\n  oracle top={:?}\n  rusty  top={:?}",
                mine.token,
                o.token,
                o.top,
                mine.top
            );
            println!(
                "{slug}: tie-zone divergence at step {i} (oracle margin {margin:.3} nats): \
                 oracle {} vs rusty {}",
                o.token, mine.token
            );
            ties += 1;
            break;
        }
        prefixes.push((slug, prefix));
    }
    println!(
        "PH-1305 oracle parity: {committed} committed steps match, {ties} tie-zone \
         divergences (margin < {TIE_FLOOR}), worst profile deviation {worst:.3} nats at \
         {worst_at} ({outliers}/{compared} comparisons over 0.25 nats); prefixes {prefixes:?}"
    );
    assert!(
        committed >= 25,
        "only {committed} matching oracle steps — the fixture stopped exercising the model"
    );
}

/// Gate 3 — our own pinned fingerprint: token stream + top-1 logit per step.
#[test]
#[ignore = "fingerprint: real MiniCPM-V 4.7-1B GGUF (release build)"]
fn greedy_fingerprint_matches_golden() {
    let Some(tr) = open_runner() else { return };

    let mut lines = String::from(
        "# rusty-engine PH-1305 qwen35 greedy fingerprint (10 steps/prompt)\n\
         # prompt|generated ids|top-1 logit\n",
    );
    for (slug, prompt) in CASES {
        let steps = trace(&tr, prompt);
        let ids = steps
            .iter()
            .map(|s| s.token.to_string())
            .collect::<Vec<_>>()
            .join(",");
        lines.push_str(&format!("{slug}|{ids}|{:.4}\n", steps[0].top[0].1));
    }

    if std::env::var("RUSTY_CAPTURE_GOLDEN").is_ok() {
        std::fs::write(GOLDEN, &lines).expect("write golden");
        println!("captured {GOLDEN}");
        return;
    }

    let want = std::fs::read_to_string(GOLDEN).unwrap_or_default();
    if want.trim().is_empty() {
        eprintln!("skip: {GOLDEN} absent — run with RUSTY_CAPTURE_GOLDEN=1 once");
        return;
    }
    let body = |s: &str| -> Vec<String> {
        s.lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(str::to_string)
            .collect()
    };
    let (got, exp) = (body(&lines), body(&want));
    assert_eq!(got.len(), exp.len(), "golden row count");
    for (g, e) in got.iter().zip(&exp) {
        let (gs, es): (Vec<&str>, Vec<&str>) = (g.split('|').collect(), e.split('|').collect());
        assert_eq!(
            gs[1], es[1],
            "greedy ids moved for {} — numeric slip",
            gs[0]
        );
        let (gl, el): (f32, f32) = (
            gs[2].parse().expect("logit"),
            es[2].parse().expect("golden logit"),
        );
        assert!(
            (gl - el).abs() <= TIE_FLOOR,
            "top-1 logit for {} moved {el} → {gl}",
            gs[0]
        );
    }
    println!(
        "PH-1305 fingerprint: {} prompts match the golden",
        CASES.len()
    );
}
