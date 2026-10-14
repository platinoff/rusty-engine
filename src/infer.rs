//! Inference plumbing: generation options, math primitives and the sampler chain.
//!
//! B3 `PH-301` (RMSNorm) and `PH-304` (sampler) live here as pure-`f32`, unit-testable
//! primitives that need no model. Attention/RoPE (`PH-302`), the paged KV-cache
//! (`PH-303`) and end-to-end dense generation (`PH-305`, needs a real GGUF) follow.

use crate::moe::softmax;
use std::cmp::Ordering;

/// Sampling / decode options (OpenAI-compat surface).
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 40,
        }
    }
}

/// Root-mean-square layer norm (RMSNorm): `x * rsqrt(mean(x^2)+eps) * weight`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "rmsnorm: shape mismatch");
    let n = x.len() as f32;
    let ms = if n == 0.0 {
        0.0
    } else {
        x.iter().map(|v| v * v).sum::<f32>() / n
    };
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
}

/// Index of the maximum logit (`None` for an empty slice).
pub fn argmax(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
}

/// Softmax temperature scaling (`temp<=0` is a no-op; greedy handled by [`argmax`]).
pub fn apply_temperature(logits: &[f32], temp: f32) -> Vec<f32> {
    if temp <= 0.0 {
        return logits.to_vec();
    }
    logits.iter().map(|&v| v / temp).collect()
}

/// Zero out all but the `k` largest logits (set the rest to `-inf`).
pub fn mask_top_k(logits: &[f32], k: usize) -> Vec<f32> {
    if k == 0 || logits.is_empty() {
        return logits.to_vec();
    }
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    let threshold = sorted[(k - 1).min(sorted.len() - 1)];
    logits
        .iter()
        .map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY })
        .collect()
}

/// Nucleus (top-p) masking: keep the smallest set of highest-probability logits whose
/// cumulative softmax mass reaches `p`; the rest go to `-inf`. `p>=1` is a no-op.
pub fn mask_top_p(logits: &[f32], p: f32) -> Vec<f32> {
    if p >= 1.0 || logits.is_empty() {
        return logits.to_vec();
    }
    let probs = softmax(logits);
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(Ordering::Equal));
    let mut keep = vec![f32::NEG_INFINITY; logits.len()];
    let mut cum = 0.0f32;
    for &i in &order {
        keep[i] = logits[i];
        cum += probs[i];
        if cum >= p {
            break;
        }
    }
    keep
}

/// Repetition penalty on tokens already generated (llama.cpp-style): divide positive
/// logits and multiply negative ones by `penalty` (a no-op when `penalty == 1`).
pub fn apply_repetition_penalty(logits: &[f32], tokens: &[usize], penalty: f32) -> Vec<f32> {
    if (penalty - 1.0).abs() < f32::EPSILON {
        return logits.to_vec();
    }
    let mut out = logits.to_vec();
    for &t in tokens {
        if let Some(v) = out.get_mut(t) {
            *v = if *v > 0.0 { *v / penalty } else { *v * penalty };
        }
    }
    out
}

/// Draw a token id from `logits` under `opts` given a deterministic uniform draw
/// `r` in `[0,1)`. Greedy when `temperature<=0`. `r` lets callers (and tests) be
/// reproducible; production feeds an RNG.
pub fn sample(logits: &[f32], opts: &GenerationOptions, r: f32) -> Option<usize> {
    if logits.is_empty() {
        return None;
    }
    if opts.temperature <= 0.0 {
        return argmax(logits);
    }
    let scaled = apply_temperature(logits, opts.temperature);
    let kmasked = mask_top_k(&scaled, opts.top_k);
    let pmasked = mask_top_p(&kmasked, opts.top_p);
    let probs = softmax(&pmasked);
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let mut cum = 0.0f32;
        for (i, &pr) in probs.iter().enumerate() {
            cum += pr / sum;
            if cum >= r {
                return Some(i);
            }
        }
    }
    argmax(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn defaults_are_conservative() {
        let o = GenerationOptions::default();
        assert_eq!(o.max_tokens, 256);
        assert!((o.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn rmsnorm_normalizes() {
        // x=[3,4] => mean sq = 12.5 => rsqrt = 1/sqrt(12.5)
        let out = rmsnorm(&[3.0, 4.0], &[1.0, 1.0], 0.0);
        assert!((out[0] - 3.0 / 12.5f32.sqrt()).abs() < EPS);
        assert!((out[1] - 4.0 / 12.5f32.sqrt()).abs() < EPS);
    }

    #[test]
    fn argmax_and_temperature() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), Some(1));
        assert_eq!(argmax(&[]), None);
        assert_eq!(
            apply_temperature(&[2.0, 4.0, 6.0], 2.0),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn top_k_masks_low_logits() {
        let m = mask_top_k(&[1.0, 2.0, 3.0, 4.0], 2);
        assert!(m[0].is_infinite() && m[0] < 0.0);
        assert!(m[1].is_infinite() && m[1] < 0.0);
        assert!((m[2] - 3.0).abs() < EPS);
        assert!((m[3] - 4.0).abs() < EPS);
    }

    #[test]
    fn top_p_masks_tail() {
        // probs roughly [.49,.49,.01,.01]; p=0.6 keeps the first two.
        let m = mask_top_p(&[3.0, 3.0, -3.0, -3.0], 0.6);
        assert!(m[2] == f32::NEG_INFINITY);
        assert!(m[3] == f32::NEG_INFINITY);
        assert!(m[0] > 0.0 && m[1] > 0.0);
    }

    #[test]
    fn repetition_penalty_sign_dependent() {
        let out = apply_repetition_penalty(&[2.0, -2.0, 2.0], &[0, 1], 2.0);
        assert!((out[0] - 1.0).abs() < EPS); // positive -> /2
        assert!((out[1] + 4.0).abs() < EPS); // negative -> *2
        assert!((out[2] - 2.0).abs() < EPS); // untouched
                                             // penalty 1.0 is a no-op
        assert_eq!(apply_repetition_penalty(&[5.0], &[0], 1.0), vec![5.0]);
    }

    #[test]
    fn sample_greedy_and_dominant() {
        // temperature 0 -> argmax
        let opts = GenerationOptions {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample(&[1.0, 5.0, 3.0], &opts, 0.5), Some(1));
        // dominant third token -> every r picks it
        let warm = GenerationOptions {
            temperature: 1.0,
            top_k: 40,
            top_p: 1.0,
            max_tokens: 8,
        };
        assert_eq!(sample(&[0.0, 0.0, 10.0], &warm, 0.5), Some(2));
        assert_eq!(sample(&[], &warm, 0.5), None);
    }
}
