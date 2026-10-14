//! Cross-module scaffold contracts (B0 gate). These prove the crate graph is
//! wired and the MoE router behaves before any model is loaded.

use rusty_engine::core::{DType, TensorInfo};
use rusty_engine::gguf::matches_magic;
use rusty_engine::hub::Worker;
use rusty_engine::moe::top_k;
use rusty_engine::node::Role;

#[test]
fn gguf_magic_is_detected_end_to_end() {
    let header = b"GGUF\x03\x00\x00\x00";
    assert!(matches_magic(header));
}

#[test]
fn moe_router_selects_k_experts_for_a_qwen_shaped_gate() {
    // 8 experts, pick top-2 (Qwen3-MoE / Mixtral style activation ratio).
    let gate = [0.02, 0.91, 0.05, 0.03, 0.88, 0.01, 0.04, 0.06];
    let chosen = top_k(&gate, 2);
    assert_eq!(chosen.len(), 2);
    assert_eq!(chosen[0].id, 1);
    assert_eq!(chosen[1].id, 4);
}

#[test]
fn tensor_descriptor_carries_dtype_and_shape() {
    let t = TensorInfo {
        name: "blk.5.ffn_gate_exps.weight".into(),
        dims: vec![2048, 512, 64],
        dtype: DType::Q4_K,
    };
    assert_eq!(t.numel(), 2048 * 512 * 64);
}

#[test]
fn fleet_worker_is_a_slave_with_a_lan_addr() {
    let w = Worker {
        id: "android-edge".into(),
        role: Role::Slave,
        addr: "192.168.2.55:8080".into(),
    };
    assert_eq!(w.role, Role::Slave);
}
