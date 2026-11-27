//! `PH-407` — MoE layer-forward A/B: resident (all experts materialized) vs
//! byte-budgeted `ExpertCache` (top-k only, evict/reload on misses).
//!
//! std-only, deterministic xorshift fills, same discipline as `benches/attn.rs`.
//! Synthetic F32 only — this benchmarks the cache substrate, NOT real GGUF
//! expert binding (PH-406 stays open) and makes no quality claim.
//!
//!   cargo bench --bench moe

use std::hint::black_box;
use std::time::Instant;

use rusty_engine::cache::{expert_bytes, ExpertCache};
use rusty_engine::moe::{
    moe_layer_forward, moe_layer_forward_cached, ExpertWeights, Gate, RouterConfig,
};

const D_MODEL: usize = 512;
const INTER: usize = 2048;
const N_EXPERTS: usize = 16;
const TOP_K: usize = 8;

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed ^ 0x9E3779B97F4A7C15;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (u32::MAX as f32)) - 0.5
        })
        .collect()
}

fn experts() -> Vec<ExpertWeights> {
    (0..N_EXPERTS)
        .map(|e| ExpertWeights {
            gate: fill(e as u64 + 1, INTER * D_MODEL),
            up: fill(e as u64 + 101, INTER * D_MODEL),
            down: fill(e as u64 + 201, D_MODEL * INTER),
        })
        .collect()
}

fn main() {
    let router_w = fill(7, N_EXPERTS * D_MODEL);
    let hidden = fill(11, D_MODEL);
    let exps = experts();
    let cfg = RouterConfig {
        gate: Gate::Sigmoid,
        top_k: TOP_K,
    };
    let one = expert_bytes(&exps[0]);

    for (name, n, cached) in [("resident", 20usize, false), ("cache_topk", 20, true)] {
        let t0 = Instant::now();
        for i in 0..n {
            if cached {
                // Budget fits only 4 of the 8 selected experts -> forced churn.
                let mut cache = ExpertCache::new(one * 4);
                let out = moe_layer_forward_cached(
                    &hidden,
                    &router_w,
                    N_EXPERTS,
                    i,
                    &mut cache,
                    None,
                    INTER,
                    cfg,
                    |id| exps[id].clone(),
                );
                black_box(out);
            } else {
                let out = moe_layer_forward(&hidden, &router_w, N_EXPERTS, &exps, None, INTER, cfg);
                black_box(out);
            }
        }
        let ns = t0.elapsed().as_nanos() / n as u128;
        println!("ph407 {name}: mean {ns} ns/iter ({n} runs, {N_EXPERTS} experts, top{TOP_K})");
    }
}
