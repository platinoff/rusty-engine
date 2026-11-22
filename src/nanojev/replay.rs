//! Evaluation helpers shared by `nanojev-replay` and unit tests.
//!
//! `answer_from_probabilities` is the port of `answer_from_probabilities` in
//! `predict_toy_decisions.py`: validate probs (finite, [0,1], sum≈1), pick the
//! argmax, and shape the per-type result record.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::core::{EngineError, Result};
use crate::nanojev::model::Nanojev;
use crate::nanojev::prepare::{Example, QuestionType};

/// Port of `answer_from_probabilities`: probabilities must be finite, in
/// `[0, 1]`, and sum to 1 within 1e-5; the argmax candidate is the answer.
pub fn answer_from_probabilities(example: &Example, probabilities: &[f32]) -> Value {
    let ids = &example.candidate_ids;
    assert_eq!(
        probabilities.len(),
        ids.len(),
        "probability vector length mismatch"
    );
    assert!(
        probabilities
            .iter()
            .all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0),
        "invalid probability value"
    );
    let sum: f32 = probabilities.iter().sum();
    assert!(
        (sum - 1.0).abs() <= 1e-5,
        "probabilities do not sum to 1 (sum={sum})"
    );
    let best = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut result = json!({
        "type": match example.qtype {
            QuestionType::Boolean => "boolean",
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
        },
        "probabilities": ids
            .iter()
            .zip(probabilities)
            .map(|(k, p)| (k.clone(), Value::from(*p as f64)))
            .collect::<serde_json::Map<String, Value>>(),
    });
    match example.qtype {
        QuestionType::Boolean => {
            result["p_true"] = Value::from(probabilities[1] as f64);
            result["value"] = Value::Bool(best == 1);
        }
        QuestionType::Choice => {
            result["choice"] = Value::String(ids[best].clone());
            result["value"] = Value::String(ids[best].clone());
        }
        QuestionType::Score => {
            let score: f64 = probabilities
                .iter()
                .enumerate()
                .map(|(i, p)| i as f64 * *p as f64)
                .sum();
            result["score"] = Value::from(score);
            result["level"] = Value::from(best as u64);
            result["value"] = Value::from(score);
        }
    }
    result
}

/// Load the frozen test cohort (`unified/hard/test.jsonl`) into a map
/// `record_id -> Value`.
pub fn load_cohort(path: &std::path::Path) -> Result<HashMap<String, Value>> {
    let raw = std::fs::read_to_string(path).map_err(|e| EngineError {
        message: format!("cohort: read {}: {e}", path.display()),
    })?;
    let mut out = HashMap::with_capacity(2500);
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| EngineError {
            message: format!("cohort: parse: {e}"),
        })?;
        if let Some(id) = v.get("id").and_then(Value::as_str) {
            out.insert(id.to_string(), v);
        }
    }
    Ok(out)
}

/// Load recorded predictions (`predictions_test.jsonl`) — one row per
/// `record_id:qid`, carrying `student_logits`/`student_probs`.
pub fn load_predictions(path: &std::path::Path) -> Result<Vec<Value>> {
    let raw = std::fs::read_to_string(path).map_err(|e| EngineError {
        message: format!("predictions: read {}: {e}", path.display()),
    })?;
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        out.push(serde_json::from_str(line).map_err(|e| EngineError {
            message: format!("predictions: parse: {e}"),
        })?);
    }
    Ok(out)
}

/// Replay report: per-question top-1 agreement and |logit| deltas.
#[derive(Debug, Default)]
pub struct ReplayReport {
    pub questions: usize,
    pub top1_agree: usize,
    pub max_abs_logit_delta: f64,
    pub mean_abs_logit_delta: f64,
    pub skipped: usize,
    pub rows: Vec<Value>,
}

/// Replay the frozen cohort, comparing our probabilities to the recorded
/// `student_probs`/`student_logits`. `limit` caps the number of questions
/// (0 = no cap); `skip` skips the first N predictions.
pub fn replay_cohort(
    nanojev: &Nanojev,
    cohort_path: &std::path::Path,
    predictions_path: &std::path::Path,
    limit: usize,
    skip: usize,
) -> Result<ReplayReport> {
    let cohort = load_cohort(cohort_path)?;
    let predictions = load_predictions(predictions_path)?;
    let mut report = ReplayReport::default();
    let start = skip.min(predictions.len());
    let end = if limit == 0 {
        predictions.len()
    } else {
        (start + limit).min(predictions.len())
    };
    let mut sum_abs = 0.0f64;
    let mut count = 0usize;
    for pred in &predictions[start..end] {
        let pred_id = pred.get("id").and_then(Value::as_str).unwrap_or("");
        let base_id = pred_id.split(':').next().unwrap_or("").to_string();
        let qid = pred
            .get("qid")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let Some(record) = cohort.get(&base_id) else {
            report.skipped += 1;
            continue;
        };
        // Rebuild this one question as a standalone request and score it.
        let payload = json!({ "states": [record] });
        let nj = nanojev;
        let out = nj.predict(&payload, None)?;
        let answers = out
            .pointer(&format!("/states/0/answers/{qid}"))
            .cloned()
            .ok_or(EngineError {
                message: format!("replay: no answer for {base_id}:{qid}"),
            })?;
        let our: Vec<f32> = pred
            .pointer("/candidate_ids")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .map(|id| {
                        answers
                            .pointer(&format!("/probabilities/{id}"))
                            .and_then(Value::as_f64)
                            .map(|f| f as f32)
                            .unwrap_or(f32::NAN)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let recorded_logits: Vec<f32> = pred
            .pointer("/student_logits")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();
        let max_abs = our
            .iter()
            .zip(&recorded_logits)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0f64, f64::max);
        report.max_abs_logit_delta = report.max_abs_logit_delta.max(max_abs);
        sum_abs += max_abs;
        count += 1;
        let our_best = our
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(usize::MAX);
        let rec_best = recorded_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(usize::MAX);
        let agree = our_best == rec_best && our_best != usize::MAX;
        if agree {
            report.top1_agree += 1;
        }
        report.questions += 1;
        report.rows.push(json!({
            "id": pred_id,
            "our_top1": our_best,
            "recorded_top1": rec_best,
            "agree": agree,
            "max_abs_logit_delta": max_abs,
        }));
    }
    report.mean_abs_logit_delta = if count > 0 {
        sum_abs / count as f64
    } else {
        0.0
    };
    Ok(report)
}
