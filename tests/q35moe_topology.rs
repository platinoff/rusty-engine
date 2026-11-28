//! A2 (PH-1206 pure prep): qwen35moe block topology on deterministic toy
//! weights. No real model, no token, no quality claim (G13 blocked).
//!
//! The parity gate is the PH-403b one: a single routed expert with softmax
//! top-1 must reproduce `moe_block_forward` within 1e-5, including the
//! carried attention state.

use rusty_engine::cache::{expert_bytes, ExpertCache};
use rusty_engine::forward::{attention_forward_cached, BlockWeights, Geometry, LayerKv};
use rusty_engine::gdn::GdnGeom;
use rusty_engine::infer::rmsnorm;
use rusty_engine::moe::{moe_block_forward, moe_layer_forward, ExpertWeights, Gate, RouterConfig};
use rusty_engine::q35moe_topology::{
    MoeAttnState, MoeBlockInput, MoeBlockTopology, MoeFfnTopology,
};
use rusty_engine::qmat::QMat;

/// The same toy block as `src/moe.rs::moe_block_single_expert_equals_dense_block`.
fn toy_block() -> (Geometry, BlockWeights, ExpertWeights, Vec<f32>) {
    let g = Geometry::new(4, 2, 1, 8, 10_000.0, 1e-5);
    let gate_w: Vec<f32> = (0..32).map(|i| i as f32 * 0.01 - 0.15).collect();
    let up_w: Vec<f32> = (0..32).map(|i| 0.2 - i as f32 * 0.012).collect();
    let down_w: Vec<f32> = (0..32).map(|i| i as f32 * 0.006 - 0.09).collect();
    let w = BlockWeights {
        attn_norm: vec![1.0; 4],
        q: QMat::from_f32((0..16).map(|i| i as f32 * 0.02 - 0.15).collect(), 4, 4),
        k: QMat::from_f32((0..8).map(|i| i as f32 * 0.03 - 0.1).collect(), 2, 4),
        v: QMat::from_f32((0..8).map(|i| 0.2 - i as f32 * 0.05).collect(), 2, 4),
        o: QMat::from_f32((0..16).map(|i| i as f32 * 0.025 - 0.18).collect(), 4, 4),
        q_bias: Vec::new(),
        k_bias: Vec::new(),
        v_bias: Vec::new(),
        ffn_norm: vec![1.0; 4],
        ffn_gate: QMat::from_f32(gate_w.clone(), 8, 4),
        ffn_up: QMat::from_f32(up_w.clone(), 8, 4),
        ffn_down: QMat::from_f32(down_w.clone(), 4, 8),
    };
    let ffn = ExpertWeights {
        gate: gate_w,
        up: up_w,
        down: down_w,
    };
    (g, w, ffn, vec![1.0, 0.0, 0.0, 0.0])
}

fn single_expert_cfg<'a>(
    router_w: &'a [f32],
    shared: Option<&'a ExpertWeights>,
) -> MoeFfnTopology<'a> {
    MoeFfnTopology {
        router_w,
        n_experts: 1,
        layer: 0,
        shared,
        inter: 8,
        cfg: RouterConfig {
            gate: Gate::Softmax,
            top_k: 1,
        },
    }
}

#[test]
fn single_expert_topology_matches_moe_block_forward_at_1e_5() {
    let (g, w, ffn, router_w) = toy_block();
    let cfg = RouterConfig {
        gate: Gate::Softmax,
        top_k: 1,
    };
    let inputs = [
        [0.6, -0.3, 0.4, 0.1],
        [0.1, 0.5, -0.2, 0.7],
        [-0.4, 0.2, 0.9, -0.1],
    ];
    let mut dense_kv = LayerKv::default();
    let mut cache = ExpertCache::new(expert_bytes(&ffn));
    let mut topo = MoeBlockTopology::new(g.d_model, MoeAttnState::dense());

    for (pos, x) in inputs.iter().enumerate() {
        let dense = moe_block_forward(
            x,
            &w,
            &g,
            pos,
            &mut dense_kv,
            &router_w,
            1,
            0,
            &mut cache,
            None,
            g.d_ff,
            cfg,
            |_| ffn.clone(),
        );
        let attn_out = match &mut topo.attn {
            MoeAttnState::Dense(kv) => attention_forward_cached(x, &w, &g, pos, kv),
            _ => unreachable!("topology was built with dense state"),
        };
        let ffn_cfg = single_expert_cfg(&router_w, None);
        let input = MoeBlockInput {
            x,
            attn_out: &attn_out,
            post_norm: &w.ffn_norm,
            eps: g.rms_eps,
        };
        let got = topo
            .compose(input, &ffn_cfg, &mut cache, |_| ffn.clone())
            .to_vec();
        assert_eq!(got.len(), dense.len());
        for (a, b) in got.iter().zip(&dense) {
            assert!(
                (a - b).abs() < 1e-5,
                "pos {pos}: topology={a} moe_block={b}"
            );
        }
    }
    assert_eq!(
        topo.attn,
        MoeAttnState::Dense(dense_kv),
        "attention state must be identical"
    );
}

#[test]
fn all_attention_state_variants_keep_the_residual_stream() {
    let gd = GdnGeom {
        groups: 2,
        dk: 2,
        dv: 2,
        conv: 3,
        inner: 4,
    };
    let zero_ffn = ExpertWeights {
        gate: vec![0.0; 32],
        up: vec![0.0; 32],
        down: vec![0.0; 32],
    };
    let router_w = [1.0f32, 0.0, 0.0, 0.0];
    let x = [0.6, -0.3, 0.4, 0.1];
    let no_attn = [0.0f32; 4];
    let mut topologies = [
        MoeBlockTopology::new(4, MoeAttnState::dense()),
        MoeBlockTopology::new(4, MoeAttnState::full()),
        MoeBlockTopology::new(4, MoeAttnState::recurrent(&gd)),
    ];
    assert!(matches!(topologies[0].attn, MoeAttnState::Dense(_)));
    assert!(matches!(topologies[1].attn, MoeAttnState::Full(_)));
    assert!(matches!(topologies[2].attn, MoeAttnState::Recurrent(_)));

    for topo in topologies.iter_mut() {
        assert!(topo.residual.iter().all(|v| *v == 0.0));
        let ffn_cfg = single_expert_cfg(&router_w, None);
        let mut cache = ExpertCache::new(expert_bytes(&zero_ffn));
        let input = MoeBlockInput {
            x: &x,
            attn_out: &no_attn,
            post_norm: &[1.0; 4],
            eps: 1e-5,
        };
        let out = topo
            .compose(input, &ffn_cfg, &mut cache, |_| zero_ffn.clone())
            .to_vec();
        assert_eq!(out, x.to_vec(), "zero FFN leaves only the residual");
    }
}

#[test]
fn shared_expert_is_added_after_the_routed_ffn() {
    let x = [1.0f32, 0.0, 0.0, 0.0];
    let no_attn = [0.0f32; 4];
    let norm = [1.0f32; 4];
    let routed_zero = ExpertWeights {
        gate: vec![0.0; 32],
        up: vec![0.0; 32],
        down: vec![0.0; 32],
    };
    let shared = ExpertWeights {
        gate: (0..32).map(|i| i as f32 * 0.01 - 0.15).collect(),
        up: (0..32).map(|i| 0.2 - i as f32 * 0.012).collect(),
        down: (0..32).map(|i| i as f32 * 0.006 - 0.09).collect(),
    };
    let router_w = [1.0f32, 0.0, 0.0, 0.0];
    let cfg = RouterConfig {
        gate: Gate::Softmax,
        top_k: 1,
    };
    let expected_ffn = moe_layer_forward(
        &rmsnorm(&x, &norm, 1e-5),
        &router_w,
        1,
        std::slice::from_ref(&routed_zero),
        Some(&shared),
        8,
        cfg,
    );

    let ffn_cfg = single_expert_cfg(&router_w, Some(&shared));
    let mut topo = MoeBlockTopology::new(4, MoeAttnState::dense());
    let mut cache = ExpertCache::new(expert_bytes(&routed_zero));
    let input = MoeBlockInput {
        x: &x,
        attn_out: &no_attn,
        post_norm: &norm,
        eps: 1e-5,
    };
    let got = topo
        .compose(input, &ffn_cfg, &mut cache, |_| routed_zero.clone())
        .to_vec();
    for (i, (a, b)) in got.iter().zip(&expected_ffn).enumerate() {
        assert!(
            (*a - (x[i] + *b)).abs() < 1e-5,
            "dim {i}: topology={a} residual+shared={}",
            x[i] + *b
        );
    }
}
