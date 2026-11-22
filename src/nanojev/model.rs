//! DecisionModel pipeline: request → leaf tokens → Qwen3 backbone →
//! LayerNorm + scalar + (choice) attention set-head → probabilities.
//!
//! Numerics mirror `train_toy_decisions.py::DecisionModel.forward` with
//! `set_head == "attention"`, evaluated per question (no cross-question
//! batching): `kmax == candidate_count`.
//!
//! Output contract is the same shape as `predict_toy_decisions.py`'s stdout:
//! `{"schema_version": "openjev-toy-inference-v1", "states": [...], ...}`.

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::core::{EngineError, Result};
use crate::nanojev::prepare::{self, Example, QuestionType};
use crate::nanojev::qwen3::Qwen3;
use crate::nanojev::replay::answer_from_probabilities;
use crate::nanojev::safetensors::SafeTensors;
use crate::tokenizer::Bpe;

const HEAD_HIDDEN: usize = 1024;
const SET_HEAD_DIM: usize = 128;
const SET_HEAD_HEADS: usize = 4;

/// One call's worth of tensor views into the mmap'd checkpoint.
struct BoundCheckpoint<'a> {
    scalar_weight: &'a [f32],                 // [1, 1024]
    scalar_bias: &'a [f32],                   // [1]
    norm_weight: &'a [f32],                   // [1024] (LayerNorm)
    norm_bias: &'a [f32],                     // [1024]
    set_project_weight: &'a [f32],            // [128, 1025]
    set_project_bias: &'a [f32],              // [128]
    set_attention_in_proj_weight: &'a [f32],  // [384, 128]
    set_attention_in_proj_bias: &'a [f32],    // [384]
    set_attention_out_proj_weight: &'a [f32], // [128, 128]
    set_attention_out_proj_bias: &'a [f32],   // [128]
    set_output_weight: &'a [f32],             // [1, 128]
    set_output_bias: &'a [f32],               // [1]
    qwen3: Qwen3<'a>,
    #[allow(dead_code)]
    pad_token_id: u32,
}

impl<'a> BoundCheckpoint<'a> {
    fn bind(st: &'a SafeTensors, pad_token_id: u32) -> Result<Self> {
        let f32_t = |name: &str| -> Result<&'a [f32]> {
            let (_, slice) = st.f32_view(name)?;
            Ok(slice)
        };
        Ok(Self {
            scalar_weight: f32_t("scalar.weight")?,
            scalar_bias: f32_t("scalar.bias")?,
            norm_weight: f32_t("norm.weight")?,
            norm_bias: f32_t("norm.bias")?,
            set_project_weight: f32_t("set_project.weight")?,
            set_project_bias: f32_t("set_project.bias")?,
            set_attention_in_proj_weight: f32_t("set_attention.in_proj_weight")?,
            set_attention_in_proj_bias: f32_t("set_attention.in_proj_bias")?,
            set_attention_out_proj_weight: f32_t("set_attention.out_proj.weight")?,
            set_attention_out_proj_bias: f32_t("set_attention.out_proj.bias")?,
            set_output_weight: f32_t("set_output.weight")?,
            set_output_bias: f32_t("set_output.bias")?,
            qwen3: Qwen3::bind(st)?,
            pad_token_id,
        })
    }
}

/// Loaded checkpoint + tokenizer, reusable across predictions.
pub struct Nanojev {
    st: SafeTensors,
    bpe: Bpe,
    pad_token_id: u32,
    vocab_size: usize,
    precision: Precision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    Fp32,
    Bf16,
}

impl Nanojev {
    /// Open a checkpoint bundle directory (as released on HF:
    /// `config.json`, `backbone_config/`, `tokenizer/`, `best.safetensors`).
    pub fn open(checkpoint_dir: &Path) -> Result<Self> {
        Self::open_with_precision(checkpoint_dir, Precision::Fp32)
    }

    /// Open with explicit precision mode (bf16 = HF `torch.autocast` semantics).
    pub fn open_with_precision(checkpoint_dir: &Path, precision: Precision) -> Result<Self> {
        let weights = checkpoint_dir.join("best.safetensors");
        if !weights.is_file() {
            return Err(EngineError {
                message: format!(
                    "nanojev: checkpoint missing best.safetensors in {}",
                    checkpoint_dir.display()
                ),
            });
        }
        let st = SafeTensors::open(&weights)?;

        let tokenizer_path = checkpoint_dir.join("tokenizer").join("tokenizer.json");
        let meta = prepare::tokenizer_meta_from_tokenizer_json(&tokenizer_path)?;
        let pad_token_id = meta.padding_token_id.unwrap_or(151_643);
        let _ = meta.eos_token_id; // eos is consumed via bpe.meta() in prepare
        let bpe = Bpe::from_meta(meta);
        let vocab_size = bpe.meta().vocab_size();

        Ok(Self {
            st,
            bpe,
            pad_token_id,
            vocab_size,
            precision,
        })
    }

    /// Tokenizer vocab size (diagnostics).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Debug: token id count per leaf for the first `n` examples, plus a
    /// decode round-trip sanity check (`text == decode(encode(text))`).
    pub fn debug_leaf_lengths(
        &self,
        payload: &Value,
        max_length: usize,
    ) -> Result<Vec<(String, usize)>> {
        let states = prepare::validate_request(payload)?;
        crate::nanojev::qwen3::set_bf16_mode(self.precision == Precision::Bf16);
        let examples = prepare::prepare_examples(&states, &self.bpe, self.bpe.meta(), max_length)?;
        let mut out = Vec::with_capacity(examples.len());
        for ex in &examples {
            out.push((
                ex.id.clone(),
                ex.leaf_tokens.iter().map(Vec::len).max().unwrap_or(0),
            ));
        }
        // Round-trip the FIRST example's first leaf and print token ids.
        if let (Some(ex), Some(first_leaf)) = (
            examples.first(),
            examples.first().and_then(|e| e.leaf_tokens.first()),
        ) {
            let decoded = self.bpe.decode(first_leaf);
            // Compare against the exact prompt reconstruction we encoded:
            // find the prompt text pieces in the request to rebuild? We don't
            // keep them; print the decoded text length instead.
            eprintln!(
                "debug-leaf: first leaf id_count={} decoded_utf8_bytes={}",
                first_leaf.len(),
                decoded.len()
            );
            eprintln!(
                "debug-leaf: decoded[0..200]={:?}",
                &decoded[..decoded.len().min(200)]
            );
            eprintln!(
                "debug-leaf: first 30 ids = {:?}",
                &first_leaf[..first_leaf.len().min(30)]
            );
            eprintln!(
                "debug-leaf: last 12 ids = {:?}",
                &first_leaf[first_leaf.len().saturating_sub(12)..]
            );
            for (i, leaf) in ex.leaf_tokens.iter().enumerate().take(4) {
                eprintln!(
                    "debug-leaf: candidate {} last 30 ids = {:?}",
                    i,
                    &leaf[leaf.len().saturating_sub(30)..]
                );
            }
        }
        Ok(out)
    }

    /// Predict from a request payload. `max_length` defaults to the
    /// checkpoint's trained `max_length` (8192, from `config.json`).
    pub fn predict(&self, payload: &Value, max_length: Option<usize>) -> Result<Value> {
        let states = prepare::validate_request(payload)?;
        crate::nanojev::qwen3::set_bf16_mode(self.precision == Precision::Bf16);
        let limit = max_length.unwrap_or(8192);
        let ckpt = BoundCheckpoint::bind(&self.st, self.pad_token_id)?;
        let examples = prepare::prepare_examples(&states, &self.bpe, self.bpe.meta(), limit)?;
        let mut outputs: HashMap<String, Value> = HashMap::with_capacity(states.len());
        for state in &states {
            outputs.insert(
                state
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                json!({ "id": state.get("id"), "answers": {} }),
            );
        }
        for ex in &examples {
            let probs = self.score_example(&ckpt, ex)?;
            let answer = answer_from_probabilities(ex, &probs);
            if let Some(ans) = outputs.get_mut(&ex.state_id) {
                ans.get_mut("answers")
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert(ex.qid.clone(), answer);
            }
        }
        let mut states_out: Vec<Value> = states
            .into_iter()
            .map(|s| {
                let id = s
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                outputs
                    .remove(&id)
                    .unwrap_or(json!({ "id": id, "answers": {} }))
            })
            .collect();
        states_out.sort_by_key(|s| {
            s.get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        });
        Ok(json!({
            "schema_version": "openjev-toy-inference-v1",
            "checkpoint": {
                "directory": "<open>",
                "base_model": "Qwen/Qwen3-0.6B",
                "resolved_model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "set_head": "attention",
            },
            "states": states_out,
        }))
    }

    /// Forward one example (all its leaves) and return candidate probabilities.
    fn score_example(&self, ckpt: &BoundCheckpoint<'_>, example: &Example) -> Result<Vec<f32>> {
        // 1) per-leaf forward, take LAST hidden state
        let mut leaves = Vec::with_capacity(example.leaf_tokens.len());
        for ids in &example.leaf_tokens {
            let hidden = ckpt.qwen3.last_hidden(ids)?;
            leaves.push(hidden);
        }

        // 2) LayerNorm(hidden) with bias (torch nn.LayerNorm, eps=1e-5)
        let mut h: Vec<f32> = leaves.into_iter().flatten().collect();
        for row in h.chunks_mut(HEAD_HIDDEN) {
            layer_norm_inplace(row, ckpt.norm_weight, ckpt.norm_bias, 1e-5);
        }

        // 3) z = scalar(norm(h)).squeeze(-1) → [kmax]
        let kmax = example.candidate_ids.len();
        let mut z = Vec::with_capacity(kmax);
        for i in 0..kmax {
            let row = &h[i * HEAD_HIDDEN..(i + 1) * HEAD_HIDDEN];
            z.push(dot1024(row, ckpt.scalar_weight) + ckpt.scalar_bias[0]);
        }
        if std::env::var("NANOJEV_DEBUG_Z").is_ok() {
            eprintln!("nanojev: z(pre) = {z:?}");
            for i in 0..kmax {
                let row = &h[i * HEAD_HIDDEN..(i + 1) * HEAD_HIDDEN];
                eprintln!("nanojev: hidden[{i}][0..8] = {:?}", &row[..8]);
            }
        }

        // 4) choice set-head (attention): rebuild u and add delta to z
        if example.qtype == QuestionType::Choice {
            let log_k = (kmax as f32).ln();
            let mut u_rows = Vec::with_capacity(kmax * SET_HEAD_DIM);
            for i in 0..kmax {
                let row = &h[i * HEAD_HIDDEN..(i + 1) * HEAD_HIDDEN];
                // u = set_project([h, log_k]) → [128]
                let mut u = vec![0.0f32; SET_HEAD_DIM];
                for (o, w) in u.iter_mut().enumerate() {
                    let mut acc = ckpt.set_project_bias[o];
                    let wrow = &ckpt.set_project_weight
                        [o * (HEAD_HIDDEN + 1)..(o + 1) * (HEAD_HIDDEN + 1)];
                    for (x, w) in row.iter().zip(&wrow[..HEAD_HIDDEN]) {
                        acc += x * w;
                    }
                    acc += log_k * wrow[HEAD_HIDDEN];
                    *w = acc;
                }
                u_rows.extend_from_slice(&u);
            }
            // MultiheadAttention(128, 4 heads) self-attention over kmax rows;
            // key_padding_mask = all-valid by construction (per-question batch).
            let mixed = multihead_attention(
                &u_rows,
                kmax,
                ckpt.set_attention_in_proj_weight,
                ckpt.set_attention_in_proj_bias,
                ckpt.set_attention_out_proj_weight,
                ckpt.set_attention_out_proj_bias,
            )?;
            // delta = set_output(tanh(u + mixed)).squeeze(-1)
            for i in 0..kmax {
                let mut acc = ckpt.set_output_bias[0];
                for d in 0..SET_HEAD_DIM {
                    let a = u_rows[i * SET_HEAD_DIM + d] + mixed[i * SET_HEAD_DIM + d];
                    acc += a.tanh() * ckpt.set_output_weight[d];
                }
                z[i] += acc;
            }
        }
        if std::env::var("NANOJEV_DEBUG_Z").is_ok() {
            eprintln!("nanojev: z(final) = {z:?}");
        }

        // 5) boolean: logits = [0, z]; others: z as-is
        let logits = match example.qtype {
            QuestionType::Boolean => vec![0.0, z[0]],
            _ => z,
        };
        // 6) softmax → probabilities (training used temperature=1.0)
        Ok(softmax(&logits))
    }
}

/// `{"states": [...]}` convenience constructor matching the trainer entry point.
pub type NativeRequest = Value;

/// Convenience one-shot: payload JSON → result JSON value.
pub fn nanojev_predict(payload: &Value, checkpoint_dir: &Path) -> Result<Value> {
    let nj = Nanojev::open(checkpoint_dir)?;
    nj.predict(payload, None)
}

/// Same, with explicit `--max-length`.
pub fn nanojev_predict_with_limit(
    payload: &Value,
    checkpoint_dir: &Path,
    max_length: usize,
) -> Result<Value> {
    let nj = Nanojev::open(checkpoint_dir)?;
    nj.predict(payload, Some(max_length))
}

fn dot1024(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), HEAD_HIDDEN);
    debug_assert_eq!(b.len(), HEAD_HIDDEN);
    let mut acc = [0.0f32; 4];
    for i in (0..a.len()).step_by(4) {
        acc[0] += a[i] * b[i];
        acc[1] += a[i + 1] * b[i + 1];
        acc[2] += a[i + 2] * b[i + 2];
        acc[3] += a[i + 3] * b[i + 3];
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

fn layer_norm_inplace(row: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    debug_assert_eq!(row.len(), HEAD_HIDDEN);
    let mean = row.iter().sum::<f32>() / HEAD_HIDDEN as f32;
    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / HEAD_HIDDEN as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for (v, (w, b)) in row.iter_mut().zip(weight.iter().zip(bias)) {
        *v = (*v - mean) * inv * w + b;
    }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    for v in &mut out {
        *v /= sum;
    }
    out
}

/// MultiheadAttention(128, 4 heads, batch_first) self-attention over rows.
/// `in_proj_weight` is `[3*128, 128]` (q | k | v stacked), `in_proj_bias` likewise.
#[allow(clippy::too_many_arguments)]
fn multihead_attention(
    u: &[f32],
    seq: usize,
    in_proj_weight: &[f32],
    in_proj_bias: &[f32],
    out_proj_weight: &[f32],
    out_proj_bias: &[f32],
) -> Result<Vec<f32>> {
    debug_assert_eq!(u.len(), seq * SET_HEAD_DIM);
    let head_dim = SET_HEAD_DIM / SET_HEAD_HEADS; // 32
                                                  // 1) in_proj → q,k,v each [seq, 128]
    let mut q = vec![0.0f32; seq * SET_HEAD_DIM];
    let mut k = vec![0.0f32; seq * SET_HEAD_DIM];
    let mut v = vec![0.0f32; seq * SET_HEAD_DIM];
    for i in 0..seq {
        let row = &u[i * SET_HEAD_DIM..(i + 1) * SET_HEAD_DIM];
        for o in 0..SET_HEAD_DIM * 3 {
            let wrow = &in_proj_weight[o * SET_HEAD_DIM..(o + 1) * SET_HEAD_DIM];
            let mut acc = in_proj_bias[o];
            for (x, w) in row.iter().zip(wrow) {
                acc += x * w;
            }
            if o < SET_HEAD_DIM {
                q[i * SET_HEAD_DIM + o] = acc;
            } else if o < 2 * SET_HEAD_DIM {
                k[i * SET_HEAD_DIM + (o - SET_HEAD_DIM)] = acc;
            } else {
                v[i * SET_HEAD_DIM + (o - 2 * SET_HEAD_DIM)] = acc;
            }
        }
    }
    // 2) per head: scores = q·k / sqrt(head_dim), softmax, weighted v
    let mut mixed = vec![0.0f32; seq * SET_HEAD_DIM];
    for h in 0..SET_HEAD_HEADS {
        let hs = h * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        for i in 0..seq {
            let q_row = &q[i * SET_HEAD_DIM + hs..i * SET_HEAD_DIM + hs + head_dim];
            let mut scores = Vec::with_capacity(seq);
            for j in 0..seq {
                let k_row = &k[j * SET_HEAD_DIM + hs..j * SET_HEAD_DIM + hs + head_dim];
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += q_row[d] * k_row[d];
                }
                scores.push(acc * scale);
            }
            softmax_inplace(&mut scores);
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, &p) in scores.iter().enumerate() {
                    acc += p * v[j * SET_HEAD_DIM + hs + d];
                }
                mixed[i * SET_HEAD_DIM + hs + d] = acc;
            }
        }
    }
    // 3) out_proj: [128 -> 128]
    let mut out = vec![0.0f32; seq * SET_HEAD_DIM];
    for i in 0..seq {
        let row = &mixed[i * SET_HEAD_DIM..(i + 1) * SET_HEAD_DIM];
        for o in 0..SET_HEAD_DIM {
            let wrow = &out_proj_weight[o * SET_HEAD_DIM..(o + 1) * SET_HEAD_DIM];
            let mut acc = out_proj_bias[o];
            for (x, w) in row.iter().zip(wrow) {
                acc += x * w;
            }
            out[i * SET_HEAD_DIM + o] = acc;
        }
    }
    Ok(out)
}

fn softmax_inplace(xs: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in xs.iter_mut() {
            *x /= sum;
        }
    }
}
