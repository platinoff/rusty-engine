//! Mixture-of-Experts router — the core differentiator vs llama-rs.
//!
//! In llama-rs the MoE routing / expert FFN runs inside llama.cpp's C++ path.
//! Here the router is **native Rust** so it can be extended (shared experts,
//! sigmoid gating, expert offload, cross-node expert-parallel) without touching
//! a C++ backend. B0 ships softmax + top-k gating; B4 adds SwiGLU experts,
//! shared experts and noaux_tc group routing.

/// One routed expert with its normalized contribution weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertScore {
    pub id: usize,
    pub weight: f32,
}

/// Numerically stable softmax.
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return exps;
    }
    exps.iter().map(|e| e / sum).collect()
}

/// Keep the `k` highest-scoring experts, renormalized so their weights sum to 1.
pub fn top_k(scores: &[f32], k: usize) -> Vec<ExpertScore> {
    if scores.is_empty() {
        return Vec::new();
    }
    let k = k.min(scores.len());
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_scores: Vec<f32> = idx[..k].iter().map(|&i| scores[i]).collect();
    let weights = softmax(&top_scores);
    idx[..k]
        .iter()
        .zip(weights.iter())
        .map(|(&id, &weight)| ExpertScore { id, weight })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let w = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_k_picks_largest_and_renormalizes() {
        let s = top_k(&[0.1, 0.9, 0.2, 0.8], 2);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].id, 1);
        assert_eq!(s[1].id, 3);
        let sum: f32 = s.iter().map(|x| x.weight).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_k_clamps_when_k_exceeds_experts() {
        let s = top_k(&[0.5, 0.4], 8);
        assert_eq!(s.len(), 2);
    }
}
