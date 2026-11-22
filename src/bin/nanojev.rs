//! `nanojev` — CLI for the NanoJev decision model (unified-games-v1
//! step-400 checkpoint `f68c47d6…`, Qwen3-0.6B backbone + attention set-head).
//!
//! Subcommands:
//!   predict --checkpoint <dir> --input <request.json> [--max-length N]
//!       Run the decision model on a `{"states": [...]}` request and print
//!       the result JSON to stdout (same shape as the reference predictor).
//!   replay --checkpoint <dir> [--cohort <test.jsonl>] [--predictions <predictions_test.jsonl>]
//!       [--limit N] [--skip N] [--out <report.json>]
//!       Replay the frozen `test` cohort and compare our probabilities to the
//!       recorded `student_probs`/`student_logits` (top-1 agreement, |Δlogits|).
//!   inspect --checkpoint <dir>
//!       Dump tensor count/names + vocab size (diagnostic).

use std::path::PathBuf;

use rusty_engine::nanojev::model::Nanojev;
use rusty_engine::nanojev::replay::replay_cohort;
use rusty_engine::nanojev::safetensors::SafeTensors;

fn usage() -> ! {
    eprintln!(
        "usage: nanojev <subcommand> [args]\n\
         \n\
         subcommands:\n\
         \x20   predict --checkpoint <dir> --input <request.json> [--max-length N]\n\
         \x20   replay --checkpoint <dir> [--cohort <test.jsonl>] [--predictions <predictions_test.jsonl>] [--limit N] [--skip N] [--out <report.json>]\n\
         \x20   inspect --checkpoint <dir>\n\
         "
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first().cloned() else {
        usage();
    };
    let rest = &args[1..];
    match cmd.as_str() {
        "predict" => {
            let mut checkpoint: Option<PathBuf> = None;
            let mut input: Option<PathBuf> = None;
            let mut max_length: Option<usize> = None;
            let mut precision: Option<String> = None;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--checkpoint" | "-c" => {
                        i += 1;
                        checkpoint = rest.get(i).map(PathBuf::from);
                    }
                    "--input" | "-i" => {
                        i += 1;
                        input = rest.get(i).map(PathBuf::from);
                    }
                    "--max-length" => {
                        i += 1;
                        max_length = rest.get(i).and_then(|v| v.parse().ok());
                    }
                    "--precision" => {
                        i += 1;
                        precision = rest.get(i).cloned();
                    }
                    other => {
                        eprintln!("unknown flag: {other}");
                        usage();
                    }
                }
                i += 1;
            }
            let (Some(checkpoint), Some(input)) = (checkpoint, input) else {
                usage();
            };
            let raw = std::fs::read_to_string(&input).unwrap_or_else(|e| {
                eprintln!("nanojev predict: cannot read {}: {e}", input.display());
                std::process::exit(1);
            });
            let payload: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| {
                eprintln!("nanojev predict: invalid JSON in {}: {e}", input.display());
                std::process::exit(1);
            });
            let precision = match precision.as_deref() {
                Some("bf16") => rusty_engine::nanojev::model::Precision::Bf16,
                Some("fp32") | None => rusty_engine::nanojev::model::Precision::Fp32,
                Some(other) => {
                    eprintln!("nanojev predict: unknown --precision {other} (fp32|bf16)");
                    std::process::exit(2);
                }
            };
            let nj = Nanojev::open_with_precision(&checkpoint, precision).unwrap_or_else(|e| {
                eprintln!(
                    "nanojev predict: open checkpoint {}: {e}",
                    checkpoint.display()
                );
                std::process::exit(1);
            });
            let result = nj.predict(&payload, max_length).unwrap_or_else(|e| {
                eprintln!("nanojev predict: {e}");
                std::process::exit(1);
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&result).unwrap_or_else(|e| {
                    eprintln!("nanojev predict: serialize: {e}");
                    std::process::exit(1);
                })
            );
        }
        "replay" => {
            let mut checkpoint: Option<PathBuf> = None;
            let mut cohort: Option<PathBuf> = None;
            let mut predictions: Option<PathBuf> = None;
            let mut limit: usize = 16;
            let mut skip: usize = 0;
            let mut out: Option<PathBuf> = None;
            let mut precision: Option<String> = None;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--checkpoint" | "-c" => {
                        i += 1;
                        checkpoint = rest.get(i).map(PathBuf::from);
                    }
                    "--cohort" => {
                        i += 1;
                        cohort = rest.get(i).map(PathBuf::from);
                    }
                    "--predictions" | "-p" => {
                        i += 1;
                        predictions = rest.get(i).map(PathBuf::from);
                    }
                    "--limit" => {
                        i += 1;
                        limit = rest.get(i).and_then(|v| v.parse().ok()).unwrap_or(16);
                    }
                    "--skip" => {
                        i += 1;
                        skip = rest.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
                    }
                    "--precision" => {
                        i += 1;
                        precision = rest.get(i).cloned();
                    }
                    "--out" => {
                        i += 1;
                        out = rest.get(i).map(PathBuf::from);
                    }
                    other => {
                        eprintln!("unknown flag: {other}");
                        usage();
                    }
                }
                i += 1;
            }
            let Some(checkpoint) = checkpoint else {
                usage();
            };
            let cohort = cohort.unwrap_or_else(|| PathBuf::from("unified_hard_test.jsonl"));
            let predictions =
                predictions.unwrap_or_else(|| PathBuf::from("predictions_test.jsonl"));
            let precision = match precision.as_deref() {
                Some("bf16") => rusty_engine::nanojev::model::Precision::Bf16,
                Some("fp32") | None => rusty_engine::nanojev::model::Precision::Fp32,
                Some(other) => {
                    eprintln!("nanojev replay: unknown --precision {other} (fp32|bf16)");
                    std::process::exit(2);
                }
            };
            let nj = Nanojev::open_with_precision(&checkpoint, precision).unwrap_or_else(|e| {
                eprintln!(
                    "nanojev replay: open checkpoint {}: {e}",
                    checkpoint.display()
                );
                std::process::exit(1);
            });
            let report =
                replay_cohort(&nj, &cohort, &predictions, limit, skip).unwrap_or_else(|e| {
                    eprintln!("nanojev replay: {e}");
                    std::process::exit(1);
                });
            let text = serde_json::to_string_pretty(&serde_json::json!({
                "questions": report.questions,
                "top1_agree": report.top1_agree,
                "top1_agreement_rate": if report.questions > 0 {
                    report.top1_agree as f64 / report.questions as f64
                } else { 0.0 },
                "max_abs_logit_delta": report.max_abs_logit_delta,
                "mean_abs_logit_delta": report.mean_abs_logit_delta,
                "skipped": report.skipped,
                "rows": report.rows,
            }))
            .unwrap_or_else(|e| {
                eprintln!("nanojev replay: serialize: {e}");
                std::process::exit(1);
            });
            match out {
                Some(path) => {
                    std::fs::write(&path, text).unwrap_or_else(|e| {
                        eprintln!("nanojev replay: write {}: {e}", path.display());
                        std::process::exit(1);
                    });
                    println!("wrote {}", path.display());
                }
                None => println!("{text}"),
            }
        }
        "debug-leaf" => {
            let mut checkpoint: Option<PathBuf> = None;
            let mut input: Option<PathBuf> = None;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--checkpoint" | "-c" => {
                        i += 1;
                        checkpoint = rest.get(i).map(PathBuf::from);
                    }
                    "--input" | "-i" => {
                        i += 1;
                        input = rest.get(i).map(PathBuf::from);
                    }
                    other => {
                        eprintln!("unknown flag: {other}");
                        usage();
                    }
                }
                i += 1;
            }
            let (Some(checkpoint), Some(input)) = (checkpoint, input) else {
                usage();
            };
            let raw = std::fs::read_to_string(&input).unwrap_or_else(|e| {
                eprintln!("nanojev debug-leaf: cannot read {}: {e}", input.display());
                std::process::exit(1);
            });
            let payload: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| {
                eprintln!("nanojev debug-leaf: invalid JSON: {e}");
                std::process::exit(1);
            });
            let nj = Nanojev::open(&checkpoint).unwrap_or_else(|e| {
                eprintln!("nanojev debug-leaf: {e}");
                std::process::exit(1);
            });
            let t0 = std::time::Instant::now();
            let lengths = nj.debug_leaf_lengths(&payload, 8192).unwrap_or_else(|e| {
                eprintln!("nanojev debug-leaf: {e}");
                std::process::exit(1);
            });
            let t1 = std::time::Instant::now();
            for (id, len) in &lengths {
                println!("{id}  tokens={len}");
            }
            eprintln!("elapsed {:?}", t1 - t0);
        }
        "debug-tensor" => {
            let mut checkpoint: Option<PathBuf> = None;
            let mut tensor: Option<String> = None;
            let mut count: usize = 8;
            let mut offset: usize = 0;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--checkpoint" | "-c" => {
                        i += 1;
                        checkpoint = rest.get(i).map(PathBuf::from);
                    }
                    "--tensor" | "-t" => {
                        i += 1;
                        tensor = rest.get(i).cloned();
                    }
                    "--count" | "-n" => {
                        i += 1;
                        count = rest.get(i).and_then(|v| v.parse().ok()).unwrap_or(8);
                    }
                    "--offset" | "-o" => {
                        i += 1;
                        offset = rest.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
                    }
                    other => {
                        eprintln!("unknown flag: {other}");
                        usage();
                    }
                }
                i += 1;
            }
            let (Some(checkpoint), Some(tensor)) = (checkpoint, tensor) else {
                usage();
            };
            let st = SafeTensors::open(&checkpoint.join("best.safetensors")).unwrap_or_else(|e| {
                eprintln!("nanojev debug-tensor: {e}");
                std::process::exit(1);
            });
            let (shape, view) = st.f32_view(&tensor).unwrap_or_else(|e| {
                eprintln!("nanojev debug-tensor: {e}");
                std::process::exit(1);
            });
            println!(
                "tensor={tensor} shape={shape:?} dtype=F32 count={} slice=[{offset}..{})",
                view.len(),
                (offset + count).min(view.len())
            );
            let stats = view.iter().skip(offset).take(count).fold(
                (f32::INFINITY, f32::NEG_INFINITY, 0.0f64, 0.0f64),
                |(mn, mx, sum, sq), v| {
                    (
                        mn.min(*v),
                        mx.max(*v),
                        sum + *v as f64,
                        sq + (*v as f64) * (*v as f64),
                    )
                },
            );
            let n = count.min(view.len() - offset) as f64;
            if n > 0.0 {
                let mean = stats.2 / n;
                let var = stats.3 / n - mean * mean;
                println!(
                    "slice min={} max={} mean={mean:.6} var={var:.6}",
                    stats.0, stats.1
                );
            }
            for (i, v) in view.iter().skip(offset).take(count).enumerate() {
                println!("  [{offset}+{i}] = {v:.8}");
            }
        }
        "inspect" => {
            let mut checkpoint: Option<PathBuf> = None;
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--checkpoint" | "-c" => {
                        i += 1;
                        checkpoint = rest.get(i).map(PathBuf::from);
                    }
                    other => {
                        eprintln!("unknown flag: {other}");
                        usage();
                    }
                }
                i += 1;
            }
            let Some(checkpoint) = checkpoint else {
                usage();
            };
            let st = SafeTensors::open(&checkpoint.join("best.safetensors")).unwrap_or_else(|e| {
                eprintln!("nanojev inspect: {e}");
                std::process::exit(1);
            });
            println!("tensors: {}", st.tensor_count());
            for name in st.tensor_names() {
                if let Ok(shape) = st.shape(name) {
                    println!(
                        "  {name}  [{}]",
                        shape
                            .iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                }
            }
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            usage();
        }
    }
}
