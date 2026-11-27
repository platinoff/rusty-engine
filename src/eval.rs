//! Model-free evaluation primitives — `PH-405c` perplexity smoke gate.
//!
//! This module deliberately knows nothing about GGUF, tokenizers or the
//! network. It turns logits / log-probabilities into a perplexity number so
//! the real-model gate can be a thin wrapper once the gitignored Qwen2.5 GGUF
//! and the llama.cpp oracle are available. **Not a quality claim** — G13 is
//! blocked; a low perplexity here says nothing about answer quality.

/// Numerically stable log-softmax of a logit slice (`f32` in/out).
///
/// Uses the max-subtraction trick in `f64` accumulation so large logits
/// (e.g. ±1000) do not overflow. An empty slice or all-non-finite input
/// yields an empty / all-NaN result rather than a panic.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let Some(max) = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f32::max)
    else {
        return vec![f32::NAN; logits.len()];
    };
    let max = max as f64;
    let sum: f64 = logits
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (*v as f64 - max).exp())
        .sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![f32::NAN; logits.len()];
    }
    let ln_sum = sum.ln();
    logits
        .iter()
        .map(|v| {
            if v.is_finite() {
                (*v as f64 - max - ln_sum) as f32
            } else {
                f32::NAN // non-finite input ⇒ undefined log-prob, never a fake number
            }
        })
        .collect()
}

/// Perplexity from the correct-token log-probabilities of a token sequence:
/// `exp(-mean(log p(token_i)))`.
///
/// Returns `NaN` for an empty sequence (no evidence ⇒ no score). A `p = 0`
/// token gives `+inf`, which is the honest "impossible under this model"
/// result — callers should not clamp it to a friendly number.
pub fn perplexity_from_logprobs(logprobs: &[f32]) -> f64 {
    if logprobs.is_empty() {
        return f64::NAN;
    }
    let neg_log_sum: f64 = logprobs.iter().map(|lp| -(*lp as f64)).sum();
    (neg_log_sum / logprobs.len() as f64).exp()
}

/// Convenience: perplexity of a single next-token distribution.
pub fn perplexity_from_logits(logits: &[f32], target: usize) -> f64 {
    let lp = log_softmax(logits);
    match lp.get(target) {
        Some(&v) => perplexity_from_logprobs(&[v]),
        None => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_logits_give_vocab_size_perplexity() {
        let logits = [0.0f32; 8];
        let lp = log_softmax(&logits);
        assert!(lp.iter().all(|v| (v + 8f32.ln()).abs() < 1e-5), "{lp:?}");
        for target in 0..8 {
            let ppl = perplexity_from_logits(&logits, target);
            assert!((ppl - 8.0).abs() < 1e-3, "target {target}: {ppl}");
        }
    }

    #[test]
    fn concentrated_logit_gives_perplexity_near_one() {
        let logits = [10.0f32, 0.0, 0.0, 0.0];
        let ppl = perplexity_from_logits(&logits, 0);
        assert!(ppl > 1.0 && ppl < 1.01, "{ppl}");
        // A surprising token is penalized.

        let wrong = perplexity_from_logits(&logits, 1);
        assert!(wrong > 20_000.0, "{wrong}");
    }

    #[test]
    fn log_softmax_is_stable_for_huge_logits() {
        let lp = log_softmax(&[1000.0f32, 1000.0]);
        assert!(lp.iter().all(|v| (v + 2f32.ln()).abs() < 1e-5), "{lp:?}");
        // Non-finite logits never poison the finite ones.
        let mixed = log_softmax(&[f32::NAN, 0.0, 0.0]);
        assert!(mixed[0].is_nan());
        assert!((mixed[1] + 2f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn empty_sequence_has_no_perplexity() {
        assert!(perplexity_from_logprobs(&[]).is_nan());
        assert!(perplexity_from_logits(&[], 0).is_nan());
        assert!(perplexity_from_logits(&[0.0], 9).is_nan());
    }

    #[test]
    fn zero_probability_token_is_infinite_not_clamped() {
        // log p = -inf ⇒ p = 0 ⇒ cross-entropy = +inf, never a fake number.
        let ppl = perplexity_from_logprobs(&[f32::NEG_INFINITY, -1.0]);
        assert!(ppl.is_infinite() && ppl.is_sign_positive(), "{ppl}");
    }
}
