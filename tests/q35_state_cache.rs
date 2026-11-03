//! `EN-7b` — SSD delta-state cache contracts.
//!
//! A hand-built 2-layer hybrid runner (GDN + full attention, all `Q35Runner` fields
//! are public) exercises the whole snapshot wire without any GGUF or tokenizer:
//! capture → bytes → parse → restore must be **bit-exact** against the continuous
//! decode, and every corrupt/foreign/mismatched byte must degrade to a miss.

use rusty_engine::core::EngineError;
use rusty_engine::gdn::{GdnGeom, GdnState};
use rusty_engine::model::LayerKind;
use rusty_engine::q35::{Q35Geom, Q35Kv};
use rusty_engine::qmat::QMat;
use rusty_engine::runner35::{Q35Decoder, Q35Runner, Q35Snapshot};
use rusty_engine::statecache::{prompt_key, StateCache};

const DM: usize = 4;
const NV: usize = 5;
const FF: usize = 8;

fn ramp(seed: i32, m: usize, n: usize) -> Vec<f32> {
    (0..m * n)
        .map(|i| {
            let v = (i as i32 + seed) * 7 % 11;
            (v as f32) * 0.05 - 0.2
        })
        .collect()
}

fn tiny_gdn_block(seed: i32, g: &GdnGeom) -> rusty_engine::runner35::GdnBlock {
    let qkv_dim = 3 * g.inner;
    rusty_engine::runner35::GdnBlock {
        qkv: QMat::from_f32(ramp(seed, qkv_dim, DM), qkv_dim, DM),
        gate: QMat::from_f32(ramp(seed + 1, g.inner, DM), g.inner, DM),
        alpha: QMat::from_f32(
            ramp(seed + 2, g.groups, DM)
                .into_iter()
                .map(|x| x * 0.3)
                .collect(),
            g.groups,
            DM,
        ),
        beta: QMat::from_f32(
            ramp(seed + 3, g.groups, DM)
                .into_iter()
                .map(|x| x * -0.2)
                .collect(),
            g.groups,
            DM,
        ),
        out: QMat::from_f32(ramp(seed + 4, DM, g.inner), DM, g.inner),
        a_log: vec![0.5, -0.25],
        dt_bias: vec![0.1, -0.1],
        conv_w: (0..g.conv * qkv_dim)
            .map(|i| ((i as f32) * 0.7 + 0.3).sin() * 0.5)
            .collect(),
        norm_w: vec![1.0, 0.5, 1.25, -0.75, 0.9, 1.1, 0.8, -1.0],
    }
}

fn tiny_full_block(seed: i32, g: &Q35Geom) -> rusty_engine::runner35::FullBlock {
    let q_dim = g.q_dim();
    rusty_engine::runner35::FullBlock {
        q: QMat::from_f32(ramp(seed, 2 * q_dim, DM), 2 * q_dim, DM),
        k: QMat::from_f32(ramp(seed + 1, g.kv_dim(), DM), g.kv_dim(), DM),
        v: QMat::from_f32(ramp(seed + 2, g.kv_dim(), DM), g.kv_dim(), DM),
        o: QMat::from_f32(ramp(seed + 3, DM, q_dim), DM, q_dim),
        q_norm: vec![1.0, 0.5],
        k_norm: vec![1.0, 1.0],
    }
}

fn tiny_runner() -> Q35Runner {
    let gdn_geom = GdnGeom {
        groups: 2,
        dk: 2,
        dv: 2,
        conv: 2,
        inner: 4,
    };
    let q35_geom = Q35Geom {
        n_heads: 2,
        n_kv: 1,
        hd: 2,
        rope_dim: 2,
        rope_base: 10_000.0,
        eps: 1e-5,
    };
    let blocks = vec![
        rusty_engine::runner35::Q35Block {
            kind: LayerKind::Recurrent,
            attn_norm: ramp(10, DM, 1),
            post_attention_norm: ramp(11, DM, 1),
            gdn: Some(tiny_gdn_block(20, &gdn_geom)),
            full: None,
            ffn_gate: QMat::from_f32(ramp(12, FF, DM), FF, DM),
            ffn_up: QMat::from_f32(ramp(13, FF, DM), FF, DM),
            ffn_down: QMat::from_f32(ramp(14, DM, FF), DM, FF),
        },
        rusty_engine::runner35::Q35Block {
            kind: LayerKind::Full,
            attn_norm: ramp(15, DM, 1),
            post_attention_norm: ramp(16, DM, 1),
            gdn: None,
            full: Some(tiny_full_block(30, &q35_geom)),
            ffn_gate: QMat::from_f32(ramp(17, FF, DM), FF, DM),
            ffn_up: QMat::from_f32(ramp(18, FF, DM), FF, DM),
            ffn_down: QMat::from_f32(ramp(19, DM, FF), DM, FF),
        },
    ];
    Q35Runner {
        n_layers: 2,
        n_vocab: NV,
        d_model: DM,
        eps: 1e-5,
        gdn_geom,
        q35_geom,
        embed: ramp(1, NV, DM),
        blocks,
        output_norm: ramp(2, DM, 1),
        output: QMat::from_f32(ramp(3, NV, DM), NV, DM),
        eos_token_id: None, // never EOS: the greedy tail always runs to max_new
    }
}

#[test]
fn snapshot_restore_is_bit_exact_against_continuous_decode() {
    let r = tiny_runner();
    let prompt = [1u32, 2, 3, 4];

    // Baseline: one continuous greedy run.
    let baseline = Q35Decoder::new(&r).generate_incremental(&prompt, 8);

    // Cache path: prefill, freeze (case A straight, case B through bytes), restore in a
    // VIIRGIN decoder, then greedy-continue from the frozen logits.
    let mut a = Q35Decoder::new(&r);
    let logits = a.prefill(&prompt);
    let snap = a.capture(&logits);

    let round = Q35Snapshot::from_bytes(&snap.to_bytes()).expect("snapshot parses");
    assert_eq!(snap, round, "bytes round trip must be identity");

    let mut b = Q35Decoder::new(&r);
    b.restore_from(&round)
        .expect("restore on the same geometry");
    assert_eq!(b.generate_after(&prompt, round.logits, 8), baseline);

    // The straight path matches too.
    assert_eq!(a.generate_after(&prompt, logits, 8), baseline);
}

#[test]
fn continuation_after_snapshot_continues_the_sequence() {
    let r = tiny_runner();
    let prompt = [2u32, 3];
    let mut live = Q35Decoder::new(&r);
    let tail = live.prefill(&prompt);

    // Two segments must equal one continuous run: prefill(2,3) then step the tail.
    let mut fresh = Q35Decoder::new(&r);
    let seg = fresh.generate_incremental(&prompt, 4);
    let mut restored = Q35Decoder::new(&r);
    let snap = restored.capture(&tail);
    restored.restore_from(&snap).unwrap();
    let seq = restored.generate_after(&prompt, snap.logits, 4);
    assert_eq!(seq, seg);
    assert_eq!(seq.len(), prompt.len() + 4);
}

#[test]
fn corrupt_and_foreign_snapshots_reject_never_decode() {
    let r = tiny_runner();
    let prompt = [1u32, 2];
    let mut d = Q35Decoder::new(&r);
    let logits = d.prefill(&prompt);
    let snap = d.capture(&logits);
    let bytes = snap.to_bytes();

    // Truncated tail: hard error, no panic, no partial decode.
    assert!(Q35Snapshot::from_bytes(&bytes[..bytes.len() / 2]).is_err());
    // Bad magic: rejected before any shape is trusted.
    let mut wrong = bytes.clone();
    wrong[0] = b'X';
    assert!(matches!(
        Q35Snapshot::from_bytes(&wrong),
        Err(EngineError { message }) if message.contains("magic")
    ));
    // Trailing garbage: also rejected — the file is exactly one snapshot.
    let mut tail = bytes.clone();
    tail.extend_from_slice(&[0u8; 4]);
    assert!(matches!(
        Q35Snapshot::from_bytes(&tail),
        Err(EngineError { message }) if message.contains("trailing")
    ));
    // A virgin geometry vs a snapshot sized for a bigger one: layer-count mismatch.
    let foreign = tiny_runner(); // same size — instead fake with a wrong-shape snapshot
    let mut fake = snap.clone();
    fake.gdn.push(GdnState::new(&tiny_runner().gdn_geom));
    fake.gdn.push(GdnState::new(&tiny_runner().gdn_geom));
    fake.gdn.push(GdnState::new(&tiny_runner().gdn_geom));
    let _ = foreign;
    assert!(Q35Decoder::new(&tiny_runner()).restore_from(&fake).is_err());
}

#[test]
fn state_cache_roundtrip_miss_and_corruption() {
    let r = tiny_runner();
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("statecache");
    let _ = std::fs::remove_dir_all(&dir);
    let cache = StateCache::at(&dir);

    let mut d = Q35Decoder::new(&r);
    let prompt = [1u32, 2, 3];
    let logits = d.prefill(&prompt);
    let snap = d.capture(&logits);
    let key = prompt_key(&prompt);

    assert!(cache.load(key).is_none(), "empty store is a miss");
    cache.store(key, &snap).expect("store lands");
    assert_eq!(cache.count(), 1, "one snapshot file");

    let back = cache.load(key).expect("hit");
    assert_eq!(back, snap, "stored bytes parse back to the same snapshot");

    let other_key = prompt_key(&[4u32, 5]);
    assert!(cache.load(other_key).is_none(), "unknown key misses");

    // Corrupt file: load degrades to None — a miss falls back to a prefill.
    let corrupt = dir.join(format!("{key:016x}.q35state"));
    std::fs::write(corrupt, b"junk-not-a-snapshot").unwrap();
    assert!(cache.load(key).is_none());

    // Oversized snapshots are refused (context runaway must not eat the SSD).
    let mut big = Q35Snapshot {
        seq_len: 0,
        logits: vec![0.0; (300 * 1024 * 1024 / 4) / NV.max(1) + NV],
        gdn: vec![GdnState::new(&r.gdn_geom); r.n_layers],
        kv: vec![Q35Kv::default(); r.n_layers],
    };
    while big.logits.len() < 70_000_000 {
        big.logits.push(0.0);
    }
    assert!(cache.store(key, &big).is_err(), "256 MiB cap holds");
}
