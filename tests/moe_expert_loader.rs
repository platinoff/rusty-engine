//! A1 (PH-406 pure half): expert byte ranges + dequant, without a model node.
//! Every expected value below is hand-derived from the registered GSV pack layout
//! (589 824 B Q4_K gate/up, 860 160 B Q6_K down) — not from the code under test.

use rusty_engine::moe_expert_loader::{
    gguf_expert_byte_range, load_expert_gguf, pack_expert_ranges, ExpertGeometry, ExpertRole,
    ExpertTensorOffsets,
};
use rusty_engine::patch::sha256_hex;
use rusty_engine::quant::quantize_q4_k;
use serde_json::json;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

fn geometry_small() -> ExpertGeometry {
    ExpertGeometry {
        d_model: 32,
        ffn: 32,
    }
}

fn geometry_registered() -> ExpertGeometry {
    ExpertGeometry {
        d_model: 2048,
        ffn: 512,
    }
}

fn manifest_record() -> serde_json::Value {
    json!({
        "schema": "gasv.expert-pack.v1",
        "layer": 0,
        "alignment": 4096,
        "entries": [{
            "layer": 0,
            "expert": 7,
            "file": "experts-L00.pack",
            "offset": 4096,
            "len": 2_039_808,
            "payload_bytes": 2_039_808,
            "quant_layout": "q4k_gate_up_q6k_down",
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }]
    })
}

/// Four Q6_K blocks whose every dequantized value is exactly +1.0:
/// raw 6-bit = low nibble 1 | qh bits 2<<n => 33, minus bias 32; d = f16(1.0), sc = 1.
fn q6k_ones_block() -> Vec<u8> {
    let mut b = vec![0u8; 210];
    for x in b[0..128].iter_mut() {
        *x = 0x11;
    }
    for x in b[128..192].iter_mut() {
        *x = 0xAA;
    }
    for x in b[192..208].iter_mut() {
        *x = 0x01;
    }
    b[208] = 0x00;
    b[209] = 0x3C;
    b
}

#[test]
fn gguf_expert_byte_range_uses_tensor_offset_and_expert_stride() {
    let got = gguf_expert_byte_range(10_985_088, 942_302_464, 2048 * 512, 7, ExpertRole::Gate);
    // 10_985_088 + 942_302_464 + 7 * (1_048_576 / 256 * 144) = 957_416_320
    assert_eq!(got, Some((957_416_320, 589_824)));
}

#[test]
fn down_expert_byte_range_uses_q6_k_block_bytes() {
    let got = gguf_expert_byte_range(10_985_088, 942_302_464, 2048 * 512, 7, ExpertRole::Down);
    // 10_985_088 + 942_302_464 + 7 * (1_048_576 / 256 * 210) = 959_308_672
    assert_eq!(got, Some((959_308_672, 860_160)));
}

#[test]
fn expert_role_maps_to_registered_tensor_names_and_ggml_types() {
    assert_eq!(
        ExpertRole::Gate.tensor_name(5),
        "blk.5.ffn_gate_exps.weight"
    );
    assert_eq!(ExpertRole::Up.tensor_name(5), "blk.5.ffn_up_exps.weight");
    assert_eq!(
        ExpertRole::Down.tensor_name(5),
        "blk.5.ffn_down_exps.weight"
    );
    assert_eq!(ExpertRole::Gate.ggml_type(), 12);
    assert_eq!(ExpertRole::Up.ggml_type(), 12);
    assert_eq!(ExpertRole::Down.ggml_type(), 14);
}

#[test]
fn gguf_expert_byte_range_rejects_zero_stride_and_overflow() {
    assert_eq!(gguf_expert_byte_range(0, 0, 0, 0, ExpertRole::Gate), None);
    assert_eq!(
        gguf_expert_byte_range(u64::MAX, 1, 1024, 3, ExpertRole::Gate),
        None
    );
}

#[test]
fn load_expert_gguf_dequantizes_bytes_at_manifest_offsets() {
    let geometry = geometry_small(); // 1024 values/expert => 576 B Q4_K, 840 B Q6_K
    let gate_src: Vec<f32> = (0..1024).map(|i| i as f32 / 1024.0).collect();
    let up_src: Vec<f32> = (0..1024).map(|i| (1023 - i) as f32 / 1024.0).collect();
    let gate = quantize_q4_k(&gate_src).unwrap();
    let up = quantize_q4_k(&up_src).unwrap();
    let down: Vec<u8> = (0..4).flat_map(|_| q6k_ones_block()).collect();
    assert_eq!((gate.len(), up.len(), down.len()), (576, 576, 840));

    // Offsets are relative to `data_start`, as in GSV `manifest_offsets`.
    let gate_off = 0u64;
    let up_off = gate_off + gate.len() as u64; // 576
    let down_off = up_off + up.len() as u64; // 1152
    let mut bytes = vec![0u8; 64];
    bytes.extend_from_slice(&gate);
    bytes.extend_from_slice(&up);
    bytes.extend_from_slice(&down);

    let offsets = ExpertTensorOffsets {
        data_start: 64,
        gate: gate_off,
        up: up_off,
        down: down_off,
        geometry,
    };
    let mut reader = std::io::Cursor::new(bytes);
    let w = load_expert_gguf(&mut reader, &offsets, 0).unwrap();

    assert_eq!((w.gate.len(), w.up.len(), w.down.len()), (1024, 1024, 1024));
    assert!((w.gate[0] - gate_src[0]).abs() < 0.05);
    assert!((w.gate[900] - gate_src[900]).abs() < 0.05);
    assert!((w.up[0] - up_src[0]).abs() < 0.05);
    assert!((w.down[0] - 1.0).abs() < 1e-6);
    assert!((w.down[1023] - 1.0).abs() < 1e-6);
}

#[test]
fn pack_expert_ranges_splits_registered_manifest_record() {
    let ranges = pack_expert_ranges(&manifest_record(), 7, geometry_registered()).unwrap();
    assert_eq!(ranges.file, "experts-L00.pack");
    assert_eq!(ranges.gate, (4096, 589_824));
    assert_eq!(ranges.up, (593_920, 589_824));
    assert_eq!(ranges.down, (1_183_744, 860_160));
    assert_eq!(ranges.sha256.len(), 64);
}

#[test]
fn pack_expert_ranges_rejects_tampered_payload_bytes() {
    let mut bad = manifest_record();
    bad["entries"][0]["payload_bytes"] = json!(2_039_809);
    assert!(pack_expert_ranges(&bad, 7, geometry_registered()).is_err());
}

/// Cross-check against the already-registered GSV pack, not a synthetic copy.
#[test]
#[ignore = "requires L:/gsv_mirror/models/gsv-versus-ai/expert-pack"]
fn registered_pack_manifest_ranges_match_real_record_sha256() {
    let dir = Path::new("L:/gsv_mirror/models/gsv-versus-ai/expert-pack");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("manifest-L00.json")).unwrap()).unwrap();
    let ranges = pack_expert_ranges(&manifest, 7, geometry_registered()).unwrap();

    // 7 * 2_039_808 = 14_278_656; gate 589_824 B, up 589_824 B, down 860_160 B.
    assert_eq!(ranges.gate, (14_278_656, 589_824));
    assert_eq!(ranges.up, (14_868_480, 589_824));
    assert_eq!(ranges.down, (15_458_304, 860_160));

    let mut pack = File::open(dir.join(&ranges.file)).unwrap();
    pack.seek(SeekFrom::Start(ranges.gate.0)).unwrap();
    let mut payload = vec![0u8; 2_039_808];
    pack.read_exact(&mut payload).unwrap();
    assert_eq!(sha256_hex(&payload), ranges.sha256);
}
