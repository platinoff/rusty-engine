//! B13 `PH-1304` — hybrid `qwen35` runner end-to-end over a synthetic GGUF.
//!
//! Two tiny layers — blk.0 Recurrent (GDN), blk.1 Full — exercise the whole wire:
//! config → validate → `Q35Runner::from_model` → residual topology with
//! `post_attention_norm` as the pre-FFN norm → greedy decode. With the attention and
//! FFN projections zeroed every block is residual-only, so the known greedy token
//! must survive the full stack exactly like the dense synth (`dense_gen.rs`).

use rusty_engine::loader::MmapModel;
use rusty_engine::model::ModelConfig;
use rusty_engine::runner35::{Q35Decoder, Q35Runner};

fn u32le(b: &mut Vec<u8>, x: u32) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn u64le(b: &mut Vec<u8>, x: u64) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn s(b: &mut Vec<u8>, v: &str) {
    u64le(b, v.len() as u64);
    b.extend_from_slice(v.as_bytes());
}
fn kv_str(b: &mut Vec<u8>, k: &str, v: &str) {
    s(b, k);
    u32le(b, 8);
    s(b, v);
}
fn kv_u32(b: &mut Vec<u8>, k: &str, v: u32) {
    s(b, k);
    u32le(b, 4);
    u32le(b, v);
}
fn kv_bool_array(b: &mut Vec<u8>, k: &str, vals: &[bool]) {
    s(b, k);
    u32le(b, 9); // Array
    u32le(b, 7); // of Bool
    u64le(b, vals.len() as u64);
    for &v in vals {
        b.push(u8::from(v));
    }
}

fn f32_tensor(
    meta: &mut Vec<u8>,
    data: &mut Vec<u8>,
    name: &str,
    ne0: usize,
    ne1: usize,
    vals: &[f32],
) {
    s(meta, name);
    if ne1 == 1 {
        u32le(meta, 1);
        u64le(meta, ne0 as u64);
    } else {
        u32le(meta, 2);
        u64le(meta, ne0 as u64);
        u64le(meta, ne1 as u64);
    }
    u32le(meta, 0);
    u64le(meta, data.len() as u64);
    for v in vals {
        data.extend_from_slice(&v.to_le_bytes());
    }
}

/// d_model=4, vocab=5, ffn=8; GDN groups=2 dk=dv=2 conv=2 inner=4 (qkv 12);
/// full heads=2 kv=1 hd=2 rope_dim=2. Zero attention/FFN ⇒ residual-only blocks.
fn synth_q35_gguf(nonzero_gdn: bool) -> Vec<u8> {
    const DM: usize = 4;
    const NV: usize = 5;
    const FF: usize = 8;
    const GR: usize = 2; // gdn groups
    const DK: usize = 2; // gdn head dim
    const INNER: usize = 4; // gdn inner = GR·DK
    const QK: usize = 3 * INNER; // 12
    const HD: usize = 2; // full head dim
    const QD: usize = 4; // full q_dim = heads·hd
    const KVD: usize = 2; // full kv_dim

    let mut meta = Vec::new();
    meta.extend_from_slice(rusty_engine::gguf::GGUF_MAGIC);
    u32le(&mut meta, 3);
    u64le(&mut meta, 0); // patched below
    u64le(&mut meta, 16);
    kv_str(&mut meta, "general.architecture", "qwen35");
    kv_u32(&mut meta, "qwen35.embedding_length", DM as u32);
    kv_u32(&mut meta, "qwen35.block_count", 2);
    kv_u32(&mut meta, "qwen35.attention.head_count", 2);
    kv_u32(&mut meta, "qwen35.attention.head_count_kv", 1);
    kv_u32(&mut meta, "qwen35.attention.key_length", HD as u32);
    kv_u32(&mut meta, "qwen35.feed_forward_length", FF as u32);
    kv_u32(&mut meta, "qwen35.vocab_size", NV as u32);
    kv_u32(&mut meta, "qwen35.full_attention_interval", 2);
    kv_bool_array(
        &mut meta,
        "qwen35.attention.recurrent_layers",
        &[true, false],
    );
    kv_u32(&mut meta, "qwen35.ssm.conv_kernel", 2);
    kv_u32(&mut meta, "qwen35.ssm.state_size", DK as u32);
    kv_u32(&mut meta, "qwen35.ssm.group_count", GR as u32);
    kv_u32(&mut meta, "qwen35.ssm.time_step_rank", GR as u32);
    kv_u32(&mut meta, "qwen35.ssm.inner_size", INNER as u32);
    kv_u32(&mut meta, "qwen35.rope.dimension_count", HD as u32);

    let mut data = Vec::new();
    let mut n_tensors: u64 = 0;
    let mut emb = [0.0f32; DM * NV];
    emb[2 * DM] = 1.0;
    emb[2 * DM + 1] = 2.0;
    f32_tensor(&mut meta, &mut data, "token_embd.weight", DM, NV, &emb);
    n_tensors += 1;
    let mut out = [0.0f32; DM * NV];
    out[4 * DM] = 1.0;
    out[4 * DM + 1] = 1.0;
    f32_tensor(&mut meta, &mut data, "output.weight", DM, NV, &out);
    n_tensors += 1;
    let ones = [1.0f32; DM];
    f32_tensor(&mut meta, &mut data, "output_norm.weight", DM, 1, &ones);
    n_tensors += 1;

    // blk.0 — GDN.
    let z_qkv = [0.0f32; QK * DM];
    let z_gate = [0.0f32; INNER * DM];
    let z_small = [0.0f32; GR * DM];
    let z_out = [0.0f32; DM * INNER];
    let a_log = [0.5f32, -0.25];
    let dt = [0.1f32, -0.1];
    let mut conv = [0.0f32; 2 * QK];
    let mut normw = [1.0f32; DK];
    if nonzero_gdn {
        // q/k/v + beta carry signal; alpha≈0 ⇒ decay≈exp(-softplus(0))≈0.5, writes ≠ 0.
        for (i, c) in conv.iter_mut().enumerate() {
            *c = ((i as f32) * 0.11).sin() * 0.3;
        }
        normw = [1.0, 1.0];
    }
    let zff = [0.0f32; FF * DM];
    let zdown = [0.0f32; DM * FF];
    let p0 = "blk.0.";
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}attn_norm.weight"),
        DM,
        1,
        &ones,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}post_attention_norm.weight"),
        DM,
        1,
        &ones,
    );
    let nz_qkv: Vec<f32> = if nonzero_gdn {
        (0..QK * DM)
            .map(|i| ((i as i32 * 7 % 13) as f32) * 0.03 - 0.15)
            .collect()
    } else {
        z_qkv.to_vec()
    };
    let nz_small: Vec<f32> = if nonzero_gdn {
        (0..GR * DM).map(|i| 0.7 - i as f32 * 0.1).collect()
    } else {
        z_small.to_vec()
    };
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}attn_qkv.weight"),
        DM,
        QK,
        &nz_qkv,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}attn_gate.weight"),
        DM,
        INNER,
        &z_gate,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}ssm_alpha.weight"),
        DM,
        GR,
        &z_small,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}ssm_beta.weight"),
        DM,
        GR,
        &nz_small,
    );
    f32_tensor(&mut meta, &mut data, &format!("{p0}ssm_a"), GR, 1, &a_log);
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}ssm_dt.bias"),
        GR,
        1,
        &dt,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}ssm_conv1d.weight"),
        2,
        QK,
        &conv,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}ssm_norm.weight"),
        DK,
        1,
        &normw,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}ssm_out.weight"),
        INNER,
        DM,
        &z_out,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}ffn_gate.weight"),
        DM,
        FF,
        &zff,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}ffn_up.weight"),
        DM,
        FF,
        &zff,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p0}ffn_down.weight"),
        FF,
        DM,
        &zdown,
    );
    n_tensors += 14;

    // blk.1 — full attention (zeroed: residual-only even when nonzero_gdn).
    let p1 = "blk.1.";
    let z_q = [0.0f32; 2 * QD * DM];
    let z_kv = [0.0f32; KVD * DM];
    let z_o = [0.0f32; DM * QD];
    let qkn = [1.0f32; HD];
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}attn_norm.weight"),
        DM,
        1,
        &ones,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}post_attention_norm.weight"),
        DM,
        1,
        &ones,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}attn_q.weight"),
        DM,
        2 * QD,
        &z_q,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}attn_k.weight"),
        DM,
        KVD,
        &z_kv,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}attn_v.weight"),
        DM,
        KVD,
        &z_kv,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}attn_q_norm.weight"),
        HD,
        1,
        &qkn,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}attn_k_norm.weight"),
        HD,
        1,
        &qkn,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}attn_output.weight"),
        QD,
        DM,
        &z_o,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}ffn_gate.weight"),
        DM,
        FF,
        &zff,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}ffn_up.weight"),
        DM,
        FF,
        &zff,
    );
    f32_tensor(
        &mut meta,
        &mut data,
        &format!("{p1}ffn_down.weight"),
        FF,
        DM,
        &zdown,
    );
    n_tensors += 11;

    let patch = n_tensors.to_le_bytes();
    meta[8..16].copy_from_slice(&patch);

    let mut file = meta;
    let aligned = rusty_engine::loader::aligned_start(file.len(), 32);
    file.resize(aligned, 0);
    file.extend_from_slice(&data);
    file
}

fn fixture(tag: &str) -> std::path::PathBuf {
    std::path::Path::new("target").join(format!("runner35_{tag}.gguf"))
}

fn load(tag: &str, bytes: &[u8]) -> (MmapModel, std::path::PathBuf) {
    std::fs::create_dir_all("target").unwrap();
    let path = fixture(tag);
    std::fs::write(&path, bytes).unwrap();
    let model = unsafe { MmapModel::open(&path) }.expect("mmap open");
    (model, path)
}

#[test]
fn hybrid_config_validates_and_loads() {
    let bytes = synth_q35_gguf(false);
    let (model, path) = load("load", &bytes);
    let cfg = ModelConfig::from_file(model.header()).expect("config");
    assert!(cfg.validate(model.header()).is_empty(), "geometry invalid");
    assert!(cfg.is_hybrid());
    let runner = Q35Runner::from_model(&model, &cfg).expect("runner35 load");
    assert_eq!(runner.n_layers, 2);
    assert_eq!(
        runner.blocks[0].kind,
        rusty_engine::model::LayerKind::Recurrent
    );
    assert_eq!(runner.blocks[1].kind, rusty_engine::model::LayerKind::Full);
    drop(model);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn residual_only_hybrid_greedy_is_the_dense_token() {
    // Zeroed attn/FFN on both block kinds ⇒ x passes through untouched ⇒ the same
    // known greedy token as the qwen2 dense synth proves the wire end-to-end.
    let bytes = synth_q35_gguf(false);
    let (model, path) = load("greedy", &bytes);
    let cfg = ModelConfig::from_file(model.header()).unwrap();
    let runner = Q35Runner::from_model(&model, &cfg).unwrap();
    let mut dec = Q35Decoder::new(&runner);
    let seq = dec.generate_incremental(&[2], 3);
    assert_eq!(seq, vec![2, 4, 4, 4], "hybrid greedy stream drifted");
    drop(model);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn live_gdn_state_stays_finite_and_deterministic() {
    // Nonzero qkv/beta/conv ⇒ the delta state really writes every token; the stream
    // must stay finite and reproducible across two fresh decoders.
    let bytes = synth_q35_gguf(true);
    let (model, path) = load("state", &bytes);
    let cfg = ModelConfig::from_file(model.header()).unwrap();
    let runner = Q35Runner::from_model(&model, &cfg).unwrap();
    let mut a = Q35Decoder::new(&runner);
    let logits = a.prefill(&[2, 3]);
    assert_eq!(logits.len(), 5);
    assert!(logits.iter().all(|v| v.is_finite()), "state logits finite");
    let mut b = Q35Decoder::new(&runner);
    let s1 = b.generate_incremental(&[2, 3], 3);
    let mut c = Q35Decoder::new(&runner);
    let s2 = c.generate_incremental(&[2, 3], 3);
    assert_eq!(s1, s2, "fresh decoders must agree");
    assert!(s1.iter().all(|&t| t < 5), "ids in vocab");
    drop(model);
    let _ = std::fs::remove_file(&path);
}

/// Real MiniCPM-V 4.7-1B hybrid load + a short greedy prefill/decode (needs the
/// gitignored 672 MB download; text correctness is `PH-1305` parity work — this only
/// proves the whole 24-layer wire runs on real bytes without panics or NaN).
#[test]
#[ignore = "real MiniCPM-V 4.7-1B GGUF (gitignored download); run: cargo test --release -- --ignored"]
fn runner35_loads_and_generates_real_minicpm() {
    let path = std::path::Path::new("data/models/MiniCPM-V-4_7-1B-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("skip: MiniCPM-V 4.7-1B GGUF absent (gitignored download)");
        return;
    }
    let model = unsafe { MmapModel::open(path) }.expect("open real model");
    let cfg = ModelConfig::from_file(model.header()).expect("real config");
    assert!(
        cfg.validate(model.header()).is_empty(),
        "real geometry invalid"
    );
    let runner = Q35Runner::from_model(&model, &cfg).expect("real runner35 load");
    assert_eq!(runner.n_layers, 24);
    let mut dec = Q35Decoder::new(&runner);
    let logits = dec.prefill(&[151644, 872, 198]);
    assert_eq!(logits.len(), runner.n_vocab);
    assert!(logits.iter().all(|v| v.is_finite()), "real logits finite");
    let mut a = Q35Decoder::new(&runner);
    let s1 = a.generate_incremental(&[151644, 872, 198], 4);
    let mut b = Q35Decoder::new(&runner);
    let s2 = b.generate_incremental(&[151644, 872, 198], 4);
    assert_eq!(s1, s2, "real greedy stream must be deterministic");
    assert_eq!(s1.len(), 7);
}
