//! Mixture-of-Experts core — the differentiator vs llama-rs (band B4, PH-401..403).
//!
//! The router (top-k gating), the per-expert SwiGLU FFN and the shared-expert merge
//! live here in **native Rust**, operating on `f32` slices. In llama-rs this whole path
//! is computed inside llama.cpp's C++ backend; here it is ours, so it can be extended
//! (group/noaux routing, expert offload, cross-node expert-parallel) freely. Math +
//! rationale cached in `docs/research/moe-router.md`.

use crate::core::{EngineError, Result};

/// One routed expert with its normalized contribution weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertScore {
    pub id: usize,
    pub weight: f32,
}

/// Gating function used to turn router logits into expert scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    /// Mixtral-style: softmax over all experts.
    Softmax,
    /// Qwen3 / DeepSeek-style: independent sigmoid per expert, renormalized after top-k.
    Sigmoid,
}

/// Router configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterConfig {
    pub gate: Gate,
    pub top_k: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            gate: Gate::Softmax,
            top_k: 2,
        }
    }
}

/// Logistic sigmoid.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Swish / SiLU activation `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
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

/// Keep the `k` highest-scoring experts, renormalized (softmax) so weights sum to 1.
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

/// Qwen3/DeepSeek gating: per-expert sigmoid, keep top-`k`, renormalize over the chosen
/// scores so the weights sum to 1.
pub fn top_k_sigmoid(scores: &[f32], k: usize) -> Vec<ExpertScore> {
    if scores.is_empty() {
        return Vec::new();
    }
    let sig: Vec<f32> = scores.iter().map(|&x| sigmoid(x)).collect();
    let k = k.min(sig.len());
    let mut idx: Vec<usize> = (0..sig.len()).collect();
    idx.sort_by(|&a, &b| {
        sig[b]
            .partial_cmp(&sig[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let wsum: f32 = idx[..k].iter().map(|&i| sig[i]).sum();
    idx[..k]
        .iter()
        .map(|&i| ExpertScore {
            id: i,
            weight: if wsum == 0.0 { 0.0 } else { sig[i] / wsum },
        })
        .collect()
}

/// Dispatch the router by [`Gate`].
pub fn route(logits: &[f32], cfg: RouterConfig) -> Vec<ExpertScore> {
    match cfg.gate {
        Gate::Softmax => top_k(logits, cfg.top_k),
        Gate::Sigmoid => top_k_sigmoid(logits, cfg.top_k),
    }
}

/// Simplified `noaux_tc`-style group routing: scores are `sigmoid(logits + bias)`, experts
/// are split into contiguous `n_groups`, at most `top_k_group` groups (by summed score) may
/// contribute, and the final `top_k` experts are renormalized among the eligible ones.
///
/// The exact llama.cpp heuristic (top-2 experts per group) is refined at parity time
/// (`PH-405`); the selection/renorm contract here is what the FFN consumes.
pub fn route_group(
    logits: &[f32],
    bias: &[f32],
    n_groups: usize,
    top_k: usize,
    top_k_group: Option<usize>,
) -> Vec<ExpertScore> {
    let n = logits.len();
    if n == 0 {
        return Vec::new();
    }
    let exp: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| sigmoid(x + bias.get(i).copied().unwrap_or(0.0)))
        .collect();

    let ng = n_groups.max(1).min(n);
    let group_of = |i: usize| i * ng / n;

    let mut gscore = vec![0.0f32; ng];
    for (i, &e) in exp.iter().enumerate() {
        gscore[group_of(i)] += e;
    }

    let allowed: Vec<bool> = match top_k_group {
        None => vec![true; ng],
        Some(tkg) => {
            let mut order: Vec<usize> = (0..ng).collect();
            order.sort_by(|&a, &b| {
                gscore[b]
                    .partial_cmp(&gscore[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut a = vec![false; ng];
            for &g in order.iter().take(tkg.min(ng)) {
                a[g] = true;
            }
            a
        }
    };

    let mut cand: Vec<usize> = (0..n).filter(|&i| allowed[group_of(i)]).collect();
    cand.sort_by(|&a, &b| {
        exp[b]
            .partial_cmp(&exp[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = top_k.min(cand.len());
    let wsum: f32 = cand[..k].iter().map(|&i| exp[i]).sum();
    cand[..k]
        .iter()
        .map(|&i| ExpertScore {
            id: i,
            weight: if wsum == 0.0 { 0.0 } else { exp[i] / wsum },
        })
        .collect()
}

/// Row-major `W[m×n]` (flat, `m*n` long) times `x[n]` → `out[m]`.
pub fn matvec(w: &[f32], m: usize, n: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), n, "matvec: input width mismatch");
    assert_eq!(w.len(), m * n, "matvec: weight size mismatch");
    w.chunks_exact(n)
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

/// SwiGLU gating: `silu(gate) * up`, elementwise (equal-length inputs).
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len(), "swiglu: shape mismatch");
    gate.iter().zip(up).map(|(g, u)| silu(*g) * u).collect()
}

/// One expert FFN: `down(swiglu(gate(x), up(x)))`.
///
/// `x` is length `hidden`; `gate_w`/`up_w` are `inter×hidden`; `down_w` is `hidden×inter`.
pub fn expert_ffn(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let g = matvec(gate_w, inter, hidden, x);
    let u = matvec(up_w, inter, hidden, x);
    let a = swiglu(&g, &u);
    matvec(down_w, hidden, inter, &a)
}

/// Combine routed expert outputs (weighted) and an optional dense shared-expert output.
///
/// `routed[k]` is expert `k`'s `hidden`-dim output; `weights` are the router weights;
/// `shared` is added verbatim when present (DeepSeek/Qwen3 have a shared expert).
pub fn combine(routed: &[Vec<f32>], weights: &[f32], shared: Option<&[f32]>) -> Result<Vec<f32>> {
    if routed.len() != weights.len() {
        return Err(EngineError {
            message: format!(
                "combine: {} outputs vs {} weights",
                routed.len(),
                weights.len()
            ),
        });
    }
    let dim = routed
        .first()
        .map(|r| r.len())
        .or_else(|| shared.map(|s| s.len()))
        .unwrap_or(0);
    if routed.iter().any(|r| r.len() != dim) {
        return Err(EngineError {
            message: "combine: ragged routed outputs".to_string(),
        });
    }
    if let Some(s) = shared {
        if s.len() != dim {
            return Err(EngineError {
                message: format!("combine: shared dim {} != {dim}", s.len()),
            });
        }
    }
    let mut out = vec![0.0f32; dim];
    for (row, wt) in routed.iter().zip(weights) {
        for (o, rv) in out.iter_mut().zip(row) {
            *o += rv * wt;
        }
    }
    if let Some(s) = shared {
        for (o, sv) in out.iter_mut().zip(s) {
            *o += sv;
        }
    }
    Ok(out)
}

/// Per-expert FFN weights, row-major: `gate`/`up` are `inter×hidden`, `down` is `hidden×inter`.
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

/// One MoE FFN layer forward on an `hidden`-dim activation:
/// router logits → top-k routed experts (SwiGLU FFN each, weighted-summed) → + shared expert.
///
/// This is the differentiator's real compute loop (llama.cpp's `qwen3moe`/`mixtral` FFN), in
/// native Rust on synthetic weights; binding it to real dequantized GGUF tensors is `PH-406`.
pub fn moe_layer_forward(
    hidden: &[f32],
    router_w: &[f32],
    n_experts: usize,
    experts: &[ExpertWeights],
    shared: Option<&ExpertWeights>,
    inter: usize,
    cfg: RouterConfig,
) -> Vec<f32> {
    let dmodel = hidden.len();
    let logits = matvec(router_w, n_experts, dmodel, hidden);
    let chosen = route(&logits, cfg);
    let mut out = vec![0.0f32; dmodel];
    for e in &chosen {
        if let Some(w) = experts.get(e.id) {
            let eo = expert_ffn(hidden, &w.gate, &w.up, &w.down, dmodel, inter);
            for (o, v) in out.iter_mut().zip(eo) {
                *o += e.weight * v;
            }
        }
    }
    if let Some(s) = shared {
        let so = expert_ffn(hidden, &s.gate, &s.up, &s.down, dmodel, inter);
        for (o, v) in out.iter_mut().zip(so) {
            *o += v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn softmax_sums_to_one() {
        let w = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < EPS);
    }

    #[test]
    fn top_k_picks_largest_and_renormalizes() {
        let s = top_k(&[0.1, 0.9, 0.2, 0.8], 2);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].id, 1);
        assert_eq!(s[1].id, 3);
        let sum: f32 = s.iter().map(|x| x.weight).sum();
        assert!((sum - 1.0).abs() < EPS);
    }

    #[test]
    fn top_k_clamps_when_k_exceeds_experts() {
        assert_eq!(top_k(&[0.5, 0.4], 8).len(), 2);
    }

    #[test]
    fn sigmoid_silu_basics() {
        assert!((sigmoid(0.0) - 0.5).abs() < EPS);
        assert!(sigmoid(10.0) > 0.999);
        assert!((silu(0.0) - 0.0).abs() < EPS);
        assert!((silu(1.0) - 1.0f32 / (1.0 + (-1.0f32).exp())).abs() < EPS);
    }

    #[test]
    fn sigmoid_router_renormalizes_to_one() {
        let s = top_k_sigmoid(&[1.0, 5.0, 2.0, 4.0], 2);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].id, 1);
        assert_eq!(s[1].id, 3);
        let sum: f32 = s.iter().map(|x| x.weight).sum();
        assert!((sum - 1.0).abs() < EPS);
    }

    #[test]
    fn route_dispatches_by_gate() {
        let cfg = RouterConfig {
            gate: Gate::Sigmoid,
            top_k: 2,
        };
        let s = route(&[0.0, 3.0, 1.0, 2.0], cfg);
        assert_eq!(s.iter().map(|e| e.id).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn route_group_restricts_to_best_group() {
        // 6 experts, 2 contiguous groups {0,1,2} / {3,4,5}; group 1 is stronger.
        let logits = [0.0, 0.0, 0.0, 5.0, 4.0, 3.0];
        let s = route_group(&logits, &[], 2, 2, Some(1));
        assert_eq!(s.len(), 2);
        for e in &s {
            assert!((3..=5).contains(&e.id), "id {} outside chosen group", e.id);
        }
        let sum: f32 = s.iter().map(|x| x.weight).sum();
        assert!((sum - 1.0).abs() < EPS);
    }

    #[test]
    fn matvec_identity_and_sum() {
        // identity 2x2 * [3,4] = [3,4]
        let out = matvec(&[1.0, 0.0, 0.0, 1.0], 2, 2, &[3.0, 4.0]);
        assert_eq!(out, vec![3.0, 4.0]);
        // 1x2 row [1,1] * [3,4] = 7
        assert_eq!(matvec(&[1.0, 1.0], 1, 2, &[3.0, 4.0]), vec![7.0]);
    }

    #[test]
    fn swiglu_elementwise() {
        let g = [0.0, 1.0];
        let u = [2.0, 3.0];
        let out = swiglu(&g, &u);
        assert!((out[0] - 0.0).abs() < EPS); // silu(0)*2 = 0
        assert!((out[1] - silu(1.0) * 3.0).abs() < EPS);
    }

    #[test]
    fn expert_ffn_pipeline() {
        // hidden=2, inter=2, all projections identity: out = silu(x) * x componentwise.
        let id2 = [1.0, 0.0, 0.0, 1.0];
        let x = [1.0, 0.0];
        let out = expert_ffn(&x, &id2, &id2, &id2, 2, 2);
        // g=[1,0], u=[1,0] => swiglu=[silu(1)*1, silu(0)*0]=[silu(1),0]; down(id)=[silu(1),0]
        assert!((out[0] - silu(1.0)).abs() < EPS);
        assert!((out[1] - 0.0).abs() < EPS);
    }

    #[test]
    fn combine_routed_plus_shared() {
        let routed = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = [0.5, 0.5];
        let out = combine(&routed, &weights, Some(&[1.0, 1.0])).unwrap();
        // 0.5*[1,2] + 0.5*[3,4] + [1,1] = [3, 4]
        assert_eq!(out, vec![3.0, 4.0]);
        // no shared expert (Mixtral)
        let out2 = combine(&routed, &weights, None).unwrap();
        assert_eq!(out2, vec![2.0, 3.0]);
    }

    #[test]
    fn combine_rejects_shape_mismatch() {
        assert!(combine(&[vec![1.0, 2.0]], &[0.5, 0.5], None).is_err());
        assert!(combine(&[vec![1.0], vec![2.0, 3.0]], &[0.5, 0.5], None).is_err());
    }

    fn id_expert(scale: f32) -> ExpertWeights {
        ExpertWeights {
            gate: vec![1.0, 0.0, 0.0, 1.0],
            up: vec![1.0, 0.0, 0.0, 1.0],
            down: vec![scale, 0.0, 0.0, scale],
        }
    }

    #[test]
    fn moe_layer_routes_to_best_expert() {
        // expert1 (down*2) must win; hidden=[1,0] -> routed out = [2*silu(1), 0].
        let experts = vec![id_expert(1.0), id_expert(2.0)];
        let router_w = vec![0.0, 0.0, 10.0, 0.0];
        let cfg = RouterConfig {
            gate: Gate::Softmax,
            top_k: 1,
        };
        let out = moe_layer_forward(&[1.0, 0.0], &router_w, 2, &experts, None, 2, cfg);
        assert!((out[0] - 2.0 * silu(1.0)).abs() < EPS);
        assert!(out[1].abs() < EPS);
    }

    #[test]
    fn moe_layer_adds_shared_expert() {
        // expert0 wins (weight 1) -> [silu1,0]; shared (identity) adds -> [2*silu1, 0].
        let id = id_expert(1.0);
        let experts = vec![id.clone(), id.clone()];
        let router_w = vec![10.0, 0.0, 0.0, 0.0];
        let cfg = RouterConfig {
            gate: Gate::Sigmoid,
            top_k: 1,
        };
        let out = moe_layer_forward(&[1.0, 0.0], &router_w, 2, &experts, Some(&id), 2, cfg);
        assert!((out[0] - 2.0 * silu(1.0)).abs() < EPS);
        assert!(out[1].abs() < EPS);
    }
}
