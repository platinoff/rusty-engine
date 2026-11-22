//! Port of `predict_toy_decisions.py`'s `validate_request` + `prepare_examples`:
//! request schema (`{"states": [...]}`) and derivation of the per-candidate
//! "leaf" token sequences fed to the Qwen3 backbone.
//!
//! Leaf format (exact, matches the trainer):
//!
//! ```text
//! State:\n{state}\nQuestion type: {typ}\nQuestion:\n{instructions}\n
//! [{False|True} criterion: …\n]          // boolean with criteria only
//! Candidate:\n{candidate_text}\nDecision:<|im_end|>
//! ```

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::core::{EngineError, Result};
use crate::tokenizer::{Bpe, TokenizerMeta};

/// One encoded candidate path (`leaf`), flattened per question.
#[derive(Debug, Clone)]
pub struct Example {
    pub id: String,
    pub state_id: String,
    pub qid: String,
    pub qtype: QuestionType,
    pub candidate_ids: Vec<String>,
    pub leaf_tokens: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    Boolean,
    Choice,
    Score,
}

/// Decode `tokenizer/tokenizer.json` into the engine's `TokenizerMeta` shape.
pub fn tokenizer_meta_from_tokenizer_json(path: &std::path::Path) -> Result<TokenizerMeta> {
    let raw = std::fs::read_to_string(path).map_err(|e| EngineError {
        message: format!("tokenizer.json: read {}: {e}", path.display()),
    })?;
    let v: Value = serde_json::from_str(&raw).map_err(|e| EngineError {
        message: format!("tokenizer.json: parse: {e}"),
    })?;

    // Invert model.vocab (token -> id) into id-sorted Vec<String>.
    let vocab = v
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(Value::as_object)
        .ok_or(EngineError {
            message: "tokenizer.json: model.vocab missing".into(),
        })?;
    let mut id_of: HashMap<String, u32> = HashMap::with_capacity(vocab.len());
    for (tok, id_v) in vocab {
        id_of.insert(tok.clone(), id_v.as_u64().unwrap_or(u64::MAX) as u32);
    }
    let max_id = id_of.values().copied().max().unwrap_or(0) as usize;
    let mut tokens = vec![String::new(); max_id + 1];
    for (tok, id) in &id_of {
        tokens[*id as usize] = tok.clone();
    }

    let merges: Vec<String> = v
        .get("model")
        .and_then(|m| m.get("merges"))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    // HF tokenizers stores merges either as ["a", "b"] pairs or as
                    // space-joined "a b" strings — normalize to "a b" for `Bpe::from_meta`.
                    match m {
                        Value::String(s) => Some(s.clone()),
                        Value::Array(pair) => {
                            let a = pair.first().and_then(Value::as_str)?;
                            let b = pair.get(1).and_then(Value::as_str)?;
                            Some(format!("{a} {b}"))
                        }
                        _ => None,
                    }
                })
                .collect()
        })
        .ok_or(EngineError {
            message: "tokenizer.json: model.merges missing".into(),
        })?;

    // specials: added_tokens → token_type=3 (control/user-defined)
    let added = v
        .get("added_tokens")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut token_type = vec![0i32; tokens.len()];
    for at in &added {
        if let (Some(id), Some(content)) = (
            at.get("id").and_then(Value::as_u64),
            at.get("content").and_then(Value::as_str),
        ) {
            let _ = content;
            if (id as usize) < token_type.len() {
                token_type[id as usize] = 3;
            }
        }
    }

    let eos = v
        .get("added_tokens")
        .and_then(Value::as_array)
        .and_then(|a| {
            a.iter()
                .find(|t| t.get("content").and_then(Value::as_str) == Some("<|im_end|>"))
        })
        .and_then(|t| t.get("id").and_then(Value::as_u64))
        .map(|n| n as u32);
    let pad = v
        .get("added_tokens")
        .and_then(Value::as_array)
        .and_then(|a| {
            a.iter()
                .find(|t| t.get("content").and_then(Value::as_str) == Some("<|endoftext|>"))
        })
        .and_then(|t| t.get("id").and_then(Value::as_u64))
        .map(|n| n as u32);

    Ok(TokenizerMeta {
        model: "qwen2".to_string(),
        tokens,
        token_type,
        scores: Vec::new(),
        merges,
        bos_token_id: None,
        eos_token_id: eos,
        padding_token_id: pad,
        unknown_token_id: None,
        add_bos_token: Some(false),
        add_eos_token: Some(false),
        add_space_prefix: Some(false),
        remove_extra_whitespaces: Some(false),
        pre: Some("qwen2".to_string()),
    })
}

/// Tokenize a text exactly like the trainer (`add_special_tokens=False`),
/// returning token ids.
pub fn encode(bpe: &Bpe, text: &str) -> Result<Vec<u32>> {
    bpe.encode(text)
}

/// The EOS token id from the payload tokenizer (defaults to `<|im_end|>` = 151645).
pub fn eos_id(meta: &TokenizerMeta) -> u32 {
    meta.eos_token_id.unwrap_or(151_645)
}

/// Port of `validate_request` — strict shape checking of `{"states": [...]}`.
/// Returns the list of states on success.
pub fn validate_request(payload: &Value) -> Result<Vec<Value>> {
    let states = payload
        .get("states")
        .ok_or(EngineError {
            message: "input must contain exactly {\"states\": [...]}".into(),
        })?
        .as_array()
        .cloned()
        .ok_or(EngineError {
            message: "states must be a non-empty array".into(),
        })?;
    if states.is_empty() {
        return Err(EngineError {
            message: "states must be a non-empty array".into(),
        });
    }
    let mut seen = std::collections::HashSet::new();
    for state in &states {
        let id = state
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if id.is_empty() || !seen.insert(id.clone()) {
            return Err(EngineError {
                message: format!("state id must be a unique non-empty string (got {id:?})"),
            });
        }
        let Some(state_val) = state.get("state") else {
            return Err(EngineError {
                message: format!("{id}: missing 'state'"),
            });
        };
        match state_val {
            Value::String(s) if s.trim().is_empty() => {
                return Err(EngineError {
                    message: format!("{id}: state must not be empty"),
                })
            }
            Value::String(_) | Value::Object(_) | Value::Array(_) => {}
            _ => {
                return Err(EngineError {
                    message: format!("{id}: state must be a string, object or array"),
                })
            }
        }
        let questions = state
            .get("questions")
            .and_then(Value::as_object)
            .ok_or(EngineError {
                message: format!("{id}: questions must be a non-empty object"),
            })?;
        if questions.is_empty() {
            return Err(EngineError {
                message: format!("{id}: questions must be a non-empty object"),
            });
        }
        for (qid, question) in questions {
            if qid.trim().is_empty() {
                return Err(EngineError {
                    message: format!("{id}: question id must be a non-empty string"),
                });
            }
            let q = question;
            if !q.is_object() {
                return Err(EngineError {
                    message: format!("{id}:{qid}: question must be an object"),
                });
            }
            if let Some(extra) = q.as_object().and_then(|o| {
                o.keys()
                    .find(|k| !["type", "instructions", "criteria"].contains(&k.as_str()))
                    .cloned()
            }) {
                return Err(EngineError {
                    message: format!("{id}:{qid}: unsupported question field {extra:?}"),
                });
            }
            let typ = q.get("type").and_then(Value::as_str).ok_or(EngineError {
                message: format!("{id}:{qid}: question type missing"),
            })?;
            match typ {
                "boolean" | "choice" | "score" => {}
                _ => {
                    return Err(EngineError {
                        message: format!("{id}:{qid}: invalid question type {typ:?}"),
                    })
                }
            }
            if !q
                .get("instructions")
                .and_then(Value::as_str)
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false)
            {
                return Err(EngineError {
                    message: format!("{id}:{qid}: instructions must be a non-empty string"),
                });
            }
            match typ {
                "boolean" => {
                    if let Some(criteria) = q.get("criteria") {
                        let c = criteria.as_object().ok_or(EngineError {
                            message: format!("{id}:{qid}: boolean criteria must be an object"),
                        })?;
                        for (k, v) in c {
                            if k != "false" && k != "true" {
                                return Err(EngineError {
                                    message: format!(
                                        "{id}:{qid}: boolean criteria keys must be false/true"
                                    ),
                                });
                            }
                            if !v.as_str().map(|s| !s.trim().is_empty()).unwrap_or(false) {
                                return Err(EngineError {
                                    message: format!(
                                        "{id}:{qid}: boolean criterion {k:?} must be a non-empty string"
                                    ),
                                });
                            }
                        }
                    }
                }
                "choice" => {
                    let criteria =
                        q.get("criteria")
                            .and_then(Value::as_object)
                            .ok_or(EngineError {
                                message: format!("{id}:{qid}: choice criteria must be an object"),
                            })?;
                    if criteria.len() < 2 || criteria.len() > 255 {
                        return Err(EngineError {
                            message: format!("{id}:{qid}: choice criteria must hold 2–255 items"),
                        });
                    }
                    for (k, v) in criteria {
                        if k.trim().is_empty()
                            || !v.as_str().map(|s| !s.trim().is_empty()).unwrap_or(false)
                        {
                            return Err(EngineError {
                                message: format!(
                                    "{id}:{qid}: choice criteria must have non-empty keys and values"
                                ),
                            });
                        }
                    }
                }
                "score" => {
                    let criteria =
                        q.get("criteria")
                            .and_then(Value::as_array)
                            .ok_or(EngineError {
                                message: format!("{id}:{qid}: score criteria must be an array"),
                            })?;
                    if criteria.len() < 2 || criteria.len() > 10 {
                        return Err(EngineError {
                            message: format!("{id}:{qid}: score criteria must hold 2–10 levels"),
                        });
                    }
                    for (i, v) in criteria.iter().enumerate() {
                        if !v.as_str().map(|s| !s.trim().is_empty()).unwrap_or(false) {
                            return Err(EngineError {
                                message: format!(
                                    "{id}:{qid}: score criterion {i} must be a non-empty string"
                                ),
                            });
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(states)
}

/// Render the state exactly as the trainer serializes it: raw string state as-
/// is, object/array state as compact JSON (sorted keys? the trainer used
/// `json.dumps(..., sort_keys=..., separators=(',', ':'))` — mirror with
/// `serde_json::to_string` which is compact).
pub fn state_text(state_val: &Value) -> Result<String> {
    match state_val {
        Value::String(s) => Ok(s.clone()),
        other => serde_json::to_string(other).map_err(|e| EngineError {
            message: format!("state serialization: {e}"),
        }),
    }
}

struct CandidatePieces {
    segments: Vec<String>,
    texts: Vec<String>,
    ids: Vec<String>,
}

fn candidate_pieces(typ: &str, question: &Value) -> Result<CandidatePieces> {
    let criteria = question.get("criteria");
    match typ {
        "boolean" => {
            let mut seg1_extra = String::new();
            if let Some(c) = criteria {
                let obj = c.as_object().ok_or(EngineError {
                    message: "boolean criteria must be an object".into(),
                })?;
                for (key, label) in [("false", "False"), ("true", "True")] {
                    if let Some(v) = obj.get(key).and_then(Value::as_str) {
                        seg1_extra.push_str(&format!("{label} criterion: {v}\n"));
                    }
                }
            }
            Ok(CandidatePieces {
                segments: vec![
                    // first segment is filled in by caller (needs state)
                    String::new(),
                    format!(
                        "Question type: {typ}\nQuestion:\n{}\n{seg1_extra}",
                        question
                            .get("instructions")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                    ),
                ],
                texts: vec!["The proposition is true.".to_string()],
                ids: vec!["false".to_string(), "true".to_string()],
            })
        }
        "choice" => {
            let obj = criteria.and_then(Value::as_object).ok_or(EngineError {
                message: "choice criteria must be an object".into(),
            })?;
            let mut ids = Vec::with_capacity(obj.len());
            let mut texts = Vec::with_capacity(obj.len());
            for (k, v) in obj {
                ids.push(k.clone());
                texts.push(format!("{k}: {}", v.as_str().unwrap_or("")));
            }
            Ok(CandidatePieces {
                segments: vec![
                    String::new(),
                    format!(
                        "Question type: {typ}\nQuestion:\n{}\n",
                        question
                            .get("instructions")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                    ),
                ],
                texts,
                ids,
            })
        }
        "score" => {
            let arr = criteria.and_then(Value::as_array).ok_or(EngineError {
                message: "score criteria must be an array".into(),
            })?;
            let ids: Vec<String> = (0..arr.len()).map(|i| i.to_string()).collect();
            let texts: Vec<String> = arr
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect();
            Ok(CandidatePieces {
                segments: vec![
                    String::new(),
                    format!(
                        "Question type: {typ}\nQuestion:\n{}\n",
                        question
                            .get("instructions")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                    ),
                ],
                texts,
                ids,
            })
        }
        _ => unreachable!(),
    }
}

/// Port of `prepare_examples`: derive the per-candidate leaf token sequences
/// for every (state, question) pair in the request.
pub fn prepare_examples(
    states: &[Value],
    bpe: &Bpe,
    meta: &TokenizerMeta,
    max_length: usize,
) -> Result<Vec<Example>> {
    let eos = eos_id(meta);
    let mut out = Vec::new();
    for state in states {
        let state_id = state
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let state_text = state_text(state.get("state").unwrap())?;
        let questions = state.get("questions").and_then(Value::as_object).unwrap();
        for (qid, question) in questions {
            let typ = question
                .get("type")
                .and_then(Value::as_str)
                .unwrap()
                .to_string();
            let mut pieces = candidate_pieces(&typ, question)?;
            pieces.segments[0] = format!("State:\n{state_text}\n");
            let prefix: Vec<u32> = pieces
                .segments
                .iter()
                .map(|s| bpe.encode(s))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();
            let leaves: Vec<Vec<u32>> = pieces
                .texts
                .iter()
                .map(|t| {
                    let mut ids = prefix.clone();
                    ids.extend(bpe.encode(&format!("Candidate:\n{t}\nDecision:"))?);
                    ids.push(eos);
                    Ok(ids)
                })
                .collect::<Result<Vec<_>>>()?;
            let largest = leaves.iter().map(Vec::len).max().unwrap_or(0);
            if largest > max_length {
                return Err(EngineError {
                    message: format!(
                        "{state_id}:{qid}: candidate path {largest} tokens exceeds max_length={max_length}"
                    ),
                });
            }
            out.push(Example {
                id: format!("{state_id}:{qid}"),
                state_id: state_id.clone(),
                qid: qid.clone(),
                qtype: match typ.as_str() {
                    "boolean" => QuestionType::Boolean,
                    "choice" => QuestionType::Choice,
                    "score" => QuestionType::Score,
                    _ => unreachable!(),
                },
                candidate_ids: pieces.ids,
                leaf_tokens: leaves,
            });
        }
    }
    Ok(out)
}

/// Build the canonical JSON request envelope for `{"states": [...]}` (used by
/// tests to construct payloads without hand-rolled JSON).
pub fn canonical_request(states: Value) -> Result<Value> {
    let states = states
        .get("states")
        .cloned()
        .unwrap_or(Value::Array(vec![]));
    Ok(json!({ "states": states }))
}
