//! GGUF model hyperparameter reader — band B3 `PH-305c`.
//!
//! Pulls `general.architecture` + `<arch>.*` (falling back to `general.*`) KV into a
//! [`ModelConfig`] that the forward loop (`PH-305`) needs: layer count, widths, head
//! counts (GQA), FFN size, RoPE base, RMSNorm eps, vocab, and MoE expert counts. Pure
//! `std`, unit-tested on a synthetic config and validated live on a real Qwen2.5 GGUF.
//!
//! `PH-305e` adds [`ModelConfig::validate`]: resolves every expected tensor by name and
//! checks its `dims` against the config-derived geometry — a genuinely verifiable gate
//! (independent of attention numerics) that a real model exposes exactly the tensors the
//! forward loop will read. Optional per-head norms / biases are checked only if present.

use crate::core::{EngineError, Result};
use crate::gguf::{GgufFile, Value};

/// What a `qwen35` hybrid layer runs for its attention block (`PH-1301`, the `PH-1201`
/// spike finding: MiniCPM-V 4.7-1B is NOT plain attention — 18 of its 24 layers are
/// Gated-DeltaNet recurrences and only every `full_attention_interval`-th is a real
/// GQA block with QK-norm and a q-gate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// GDN: mixed `attn_qkv` (3·inner) + `attn_gate` + `ssm_*` delta-rule state.
    Recurrent,
    /// GQA block: `attn_q` (q+gate = 2·q_dim), `attn_k`/`attn_v`, `attn_{q,k}_norm`,
    /// `attn_output`.
    Full,
}

/// Model geometry + knobs needed to run a forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_ff: usize,
    pub n_vocab: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub eos_token_id: Option<u32>,
    /// `attention.key_length` — the real per-head width when it is not `d_model / n_heads`
    /// (qwen35 full layers: 256; 0 ⇒ derive, the qwen2/llama convention).
    pub key_length: usize,
    /// `full_attention_interval`: layer `i` is full-attention iff `(i + 1) % interval == 0`
    /// (unless `recurrent_layers` overrides it). 0 ⇒ plain architecture, every layer full.
    pub full_attention_interval: usize,
    /// `attention.recurrent_layers` Bool array — per-layer override of the interval rule.
    pub recurrent_layers: Vec<bool>,
    /// GDN geometry (`ssm.*`): depthwise conv kernel, per-head state side, head groups,
    /// dt/alpha/bias rank, and the total inner width (mixed qkv = 3·inner). All 0 for
    /// non-hybrid arches.
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_inner_size: usize,
    /// Partial-RoPE width + sections (`rope.dimension_count`, `rope.dimension_sections`,
    /// `rope.section_order`) — qwen35 ships `[11,11,10,0]` / `"zyxt"`; defaults ⇒ full RoPE.
    pub rope_dimension_count: usize,
    pub rope_sections: Vec<usize>,
    pub rope_section_order: String,
}

impl ModelConfig {
    /// Per-head dimension: `attention.key_length` when the arch carries one (qwen35: 256),
    /// otherwise `d_model / n_heads` (qwen2/llama convention).
    pub fn head_dim(&self) -> usize {
        if self.key_length > 0 {
            self.key_length
        } else {
            self.d_model / self.n_heads.max(1)
        }
    }

    pub fn is_moe(&self) -> bool {
        self.n_experts > 0
    }

    /// The `qwen35` hybrid layout is active when the file says which layers are recurrent
    /// or how often a full-attention layer comes.
    pub fn is_hybrid(&self) -> bool {
        self.full_attention_interval > 0 || !self.recurrent_layers.is_empty()
    }

    /// Per-layer attention kind. `recurrent_layers` (Bool[n]) wins; else the interval rule
    /// `full iff (i + 1) % interval == 0` (qwen35-1B: i = 3, 7, … — matches the tensor
    /// names on the real file); non-hybrid arches are all [`LayerKind::Full`].
    pub fn layer_kind(&self, i: usize) -> LayerKind {
        if let Some(&rec) = self.recurrent_layers.get(i) {
            return if rec {
                LayerKind::Recurrent
            } else {
                LayerKind::Full
            };
        }
        if self.full_attention_interval == 0 {
            return LayerKind::Full;
        }
        if (i + 1) % self.full_attention_interval == 0 {
            LayerKind::Full
        } else {
            LayerKind::Recurrent
        }
    }

    /// Mixed GDN projection width: q/k/v stacked (`blk.i.attn_qkv`, 3 × inner).
    pub fn gdn_qkv_dim(&self) -> usize {
        3 * self.ssm_inner_size
    }

    /// Full-attention q width without the fused gate (`n_heads × head_dim`).
    pub fn full_q_dim(&self) -> usize {
        self.n_heads * self.head_dim()
    }

    /// Extract a config from a parsed GGUF header.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        let arch = match file.get("general.architecture") {
            Some(Value::Str(s)) => s.clone(),
            _ => {
                return Err(EngineError {
                    message: "missing general.architecture".to_string(),
                })
            }
        };
        let req = |suffix: &str| -> Result<usize> {
            u32_of(
                file,
                &[&format!("{arch}.{suffix}"), &format!("general.{suffix}")],
            )
            .map(|v| v as usize)
            .ok_or_else(|| EngineError {
                message: format!("missing required key '{arch}.{suffix}'"),
            })
        };
        let opt_usize = |suffix: &str, dflt: usize| -> usize {
            u32_of(
                file,
                &[&format!("{arch}.{suffix}"), &format!("general.{suffix}")],
            )
            .map(|v| v as usize)
            .unwrap_or(dflt)
        };
        let opt_f32 = |suffix: &str, dflt: f32| -> f32 {
            f32_of(
                file,
                &[&format!("{arch}.{suffix}"), &format!("general.{suffix}")],
            )
            .unwrap_or(dflt)
        };

        let n_heads = req("attention.head_count")?;
        let d_model = req("embedding_length")?;
        let n_layers = req("block_count")?;
        let d_ff = req("feed_forward_length")?;
        let n_kv_heads = opt_usize("attention.head_count_kv", n_heads);
        let n_vocab = opt_usize(
            "vocab_size",
            match file.get("tokenizer.ggml.tokens") {
                Some(Value::Array(_, items)) => items.len(),
                _ => 0,
            },
        );
        let eos_token_id = u32_of(file, &["tokenizer.ggml.eos_token_id"]);

        // `PH-1301`: the qwen35 hybrid block — every key optional so qwen2/llama files keep
        // the legacy zero/default behavior.
        let key_length = opt_usize("attention.key_length", 0);
        let full_attention_interval = opt_usize("full_attention_interval", 0);
        let rec_key = format!("{arch}.attention.recurrent_layers");
        let recurrent_layers = match find(
            file,
            &[rec_key.as_str(), "general.attention.recurrent_layers"],
        ) {
            Some(Value::Array(_, items)) => items
                .iter()
                .map(|v| matches!(v, Value::Bool(true)))
                .collect(),
            _ => Vec::new(),
        };
        let ssm_conv_kernel = opt_usize("ssm.conv_kernel", 0);
        let ssm_state_size = opt_usize("ssm.state_size", 0);
        let ssm_group_count = opt_usize("ssm.group_count", 0);
        let ssm_time_step_rank = opt_usize("ssm.time_step_rank", 0);
        let ssm_inner_size = opt_usize("ssm.inner_size", 0);
        let rope_dimension_count = opt_usize("rope.dimension_count", 0);
        let sec_key = format!("{arch}.rope.dimension_sections");
        let rope_sections = match find(file, &[sec_key.as_str(), "general.rope.dimension_sections"])
        {
            Some(Value::Array(_, items)) => items
                .iter()
                .filter_map(|v| match v {
                    Value::I32(x) => usize::try_from(*x).ok(),
                    Value::U32(x) => Some(*x as usize),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        let order_key = format!("{arch}.rope.section_order");
        let rope_section_order =
            match find(file, &[order_key.as_str(), "general.rope.section_order"]) {
                Some(Value::Str(s)) => s.clone(),
                _ => String::new(),
            };

        Ok(Self {
            arch: arch.clone(),
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            d_ff,
            n_vocab,
            rope_freq_base: opt_f32("rope.freq_base", 10_000.0),
            rms_eps: opt_f32("attention.layer_norm_rms_epsilon", 1e-5),
            n_experts: opt_usize("expert_count", 0),
            n_experts_used: opt_usize("expert_used_count", 0),
            eos_token_id,
            key_length,
            full_attention_interval,
            recurrent_layers,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_inner_size,
            rope_dimension_count,
            rope_sections,
            rope_section_order,
        })
    }

    /// Resolve every tensor the forward loop will read and check its `dims` against the
    /// config-derived geometry. Returns a list of problems (empty ⇒ the file exposes exactly
    /// what the architecture promises). A verifiable name+dims gate — independent of attention
    /// numerics — so a real model is checked before we trust a forward pass over it.
    pub fn validate(&self, file: &GgufFile) -> Vec<String> {
        let mut errs = Vec::new();
        let (dm, hd) = (self.d_model, self.head_dim());
        let (kv, nh, ff, nv) = (
            self.n_kv_heads * self.head_dim(),
            self.n_heads * self.head_dim(),
            self.d_ff,
            self.n_vocab,
        );
        let req = |name: &str, dims: &[usize], errs: &mut Vec<String>| match file
            .tensors
            .iter()
            .find(|t| t.name == name)
        {
            Some(t) if t.dims == dims => {}
            Some(t) => errs.push(format!("{name}: dims {:?} != expected {dims:?}", t.dims)),
            None => errs.push(format!("missing required tensor '{name}'")),
        };
        let opt = |name: &str, dims: &[usize], errs: &mut Vec<String>| {
            if let Some(t) = file.tensors.iter().find(|t| t.name == name) {
                if t.dims != dims {
                    errs.push(format!("{name}: dims {:?} != expected {dims:?}", t.dims));
                }
            }
        };
        // At least one of the names must be present with these dims (naming varies by arch:
        // llama/qwen2 use `output_norm.weight`; others `norm.weight`).
        let req_any = |names: &[&str], dims: &[usize], errs: &mut Vec<String>| match names
            .iter()
            .find_map(|n| file.tensors.iter().find(|t| &t.name == n))
        {
            Some(t) if t.dims == dims => {}
            Some(t) => errs.push(format!(
                "{}: dims {:?} != expected {dims:?}",
                t.name, t.dims
            )),
            None => errs.push(format!("missing any of {names:?}")),
        };
        req("token_embd.weight", &[dm, nv], &mut errs);
        req("output.weight", &[dm, nv], &mut errs);
        req_any(&["output_norm.weight", "norm.weight"], &[dm], &mut errs);
        if self.is_hybrid() && self.ssm_inner_size > 0 {
            // qwen35 (`PH-1301`): the layer's tensor set depends on its kind. The full
            // layers fold a q-gate into `attn_q` (2·q columns) and carry per-head QK-norm;
            // the recurrent layers replace q/k/v/out with the mixed GDN stack.
            let (q, gdn_qkv, r) = (
                self.full_q_dim(),
                self.gdn_qkv_dim(),
                self.ssm_time_step_rank.max(1),
            );
            for i in 0..self.n_layers {
                let p = format!("blk.{i}.");
                // The real MiniCPM-V 4.7 file has NO `ffn_norm` — qwen35 reuses
                // `post_attention_norm` as the pre-FFN norm (two norms per block).
                req(&format!("{p}attn_norm.weight"), &[dm], &mut errs);
                req(&format!("{p}post_attention_norm.weight"), &[dm], &mut errs);
                match self.layer_kind(i) {
                    LayerKind::Recurrent => {
                        req(&format!("{p}attn_qkv.weight"), &[dm, gdn_qkv], &mut errs);
                        req(
                            &format!("{p}attn_gate.weight"),
                            &[dm, self.ssm_inner_size],
                            &mut errs,
                        );
                        req(&format!("{p}ssm_a"), &[r], &mut errs);
                        req(&format!("{p}ssm_alpha.weight"), &[dm, r], &mut errs);
                        req(&format!("{p}ssm_beta.weight"), &[dm, r], &mut errs);
                        req(
                            &format!("{p}ssm_conv1d.weight"),
                            &[self.ssm_conv_kernel, gdn_qkv],
                            &mut errs,
                        );
                        req(&format!("{p}ssm_dt.bias"), &[r], &mut errs);
                        req(
                            &format!("{p}ssm_norm.weight"),
                            &[self.ssm_state_size],
                            &mut errs,
                        );
                        req(
                            &format!("{p}ssm_out.weight"),
                            &[self.ssm_inner_size, dm],
                            &mut errs,
                        );
                    }
                    LayerKind::Full => {
                        req(&format!("{p}attn_q.weight"), &[dm, 2 * q], &mut errs);
                        req(&format!("{p}attn_k.weight"), &[dm, kv], &mut errs);
                        req(&format!("{p}attn_v.weight"), &[dm, kv], &mut errs);
                        req(&format!("{p}attn_output.weight"), &[q, dm], &mut errs);
                        req(&format!("{p}attn_q_norm.weight"), &[hd], &mut errs);
                        req(&format!("{p}attn_k_norm.weight"), &[hd], &mut errs);
                    }
                }
                req(&format!("{p}ffn_gate.weight"), &[dm, ff], &mut errs);
                req(&format!("{p}ffn_up.weight"), &[dm, ff], &mut errs);
                req(&format!("{p}ffn_down.weight"), &[ff, dm], &mut errs);
            }
            return errs;
        }
        for i in 0..self.n_layers {
            let p = format!("blk.{i}.");
            req(&format!("{p}attn_norm.weight"), &[dm], &mut errs);
            req(&format!("{p}attn_q.weight"), &[dm, dm], &mut errs);
            req(&format!("{p}attn_k.weight"), &[dm, kv], &mut errs);
            req(&format!("{p}attn_v.weight"), &[dm, kv], &mut errs);
            req(&format!("{p}attn_output.weight"), &[dm, nh], &mut errs);
            opt(&format!("{p}attn_q_norm.weight"), &[hd], &mut errs);
            opt(&format!("{p}attn_k_norm.weight"), &[hd], &mut errs);
            opt(&format!("{p}attn_q.bias"), &[dm], &mut errs);
            opt(&format!("{p}attn_k.bias"), &[kv], &mut errs);
            opt(&format!("{p}attn_v.bias"), &[kv], &mut errs);
            req(&format!("{p}ffn_norm.weight"), &[dm], &mut errs);
            req(&format!("{p}ffn_gate.weight"), &[dm, ff], &mut errs);
            req(&format!("{p}ffn_up.weight"), &[dm, ff], &mut errs);
            req(&format!("{p}ffn_down.weight"), &[ff, dm], &mut errs);
        }
        errs
    }
}

fn find<'a>(file: &'a GgufFile, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|k| file.get(k))
}

fn u32_of(file: &GgufFile, keys: &[&str]) -> Option<u32> {
    find(file, keys).and_then(|v| match v {
        Value::U8(x) => Some(u32::from(*x)),
        Value::U16(x) => Some(u32::from(*x)),
        Value::U32(x) => Some(*x),
        Value::I32(x) => u32::try_from(*x).ok(),
        Value::U64(x) => u32::try_from(*x).ok(),
        Value::I64(x) => u32::try_from(*x).ok(),
        _ => None,
    })
}

fn f32_of(file: &GgufFile, keys: &[&str]) -> Option<f32> {
    find(file, keys).and_then(|v| match v {
        Value::F32(x) => Some(*x),
        Value::F64(x) => Some(*x as f32),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GGUF_MAGIC;

    fn u32le(b: &mut Vec<u8>, x: u32) {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fn u64le(b: &mut Vec<u8>, x: u64) {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fn put_str(b: &mut Vec<u8>, s: &str) {
        u64le(b, s.len() as u64);
        b.extend_from_slice(s.as_bytes());
    }
    fn kv_u32(b: &mut Vec<u8>, k: &str, v: u32) {
        put_str(b, k);
        u32le(b, 4); // U32
        u32le(b, v);
    }
    fn kv_f32(b: &mut Vec<u8>, k: &str, v: f32) {
        put_str(b, k);
        u32le(b, 6); // F32
        b.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    fn synth_qwen_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(GGUF_MAGIC);
        u32le(&mut b, 3);
        u64le(&mut b, 0); // n_tensors
        u64le(&mut b, 7); // n_kv
                          // architecture
        put_str(&mut b, "general.architecture");
        u32le(&mut b, 8); // Str
        put_str(&mut b, "qwen2");
        kv_u32(&mut b, "qwen2.embedding_length", 896);
        kv_u32(&mut b, "qwen2.block_count", 24);
        kv_u32(&mut b, "qwen2.attention.head_count", 14);
        kv_u32(&mut b, "qwen2.attention.head_count_kv", 2);
        kv_u32(&mut b, "qwen2.feed_forward_length", 4864);
        kv_f32(&mut b, "qwen2.rope.freq_base", 1_000_000.0);
        b
    }

    #[test]
    fn parses_qwen2_config() {
        let file = crate::gguf::parse(&synth_qwen_gguf()).expect("parse");
        let c = ModelConfig::from_file(&file).expect("config");
        assert_eq!(c.arch, "qwen2");
        assert_eq!(c.n_layers, 24);
        assert_eq!(c.d_model, 896);
        assert_eq!(c.n_heads, 14);
        assert_eq!(c.n_kv_heads, 2);
        assert_eq!(c.d_ff, 4864);
        assert!((c.rope_freq_base - 1_000_000.0).abs() < 1.0);
        assert_eq!(c.head_dim(), 896 / 14);
        assert_eq!(c.n_experts, 0); // dense qwen2
        assert!(!c.is_moe());
        assert!((c.rms_eps - 1e-5).abs() < 1e-9); // default
    }

    #[test]
    fn missing_architecture_errors() {
        let mut b = Vec::new();
        b.extend_from_slice(GGUF_MAGIC);
        u32le(&mut b, 3);
        u64le(&mut b, 0);
        u64le(&mut b, 0);
        let file = crate::gguf::parse(&b).unwrap();
        assert!(ModelConfig::from_file(&file).is_err());
    }

    fn tninfo(b: &mut Vec<u8>, name: &str, dims: &[usize], ggml: u32, offset: u64) {
        put_str(b, name);
        u32le(b, dims.len() as u32);
        for &d in dims {
            u64le(b, d as u64);
        }
        u32le(b, ggml);
        u64le(b, offset);
    }

    /// qwen2 config (dm4, heads2, kv1 -> head2, kv_dim2, nh_dim4, ff8, vocab5) + 12 tensors.
    fn synth_model(bad_k: bool) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(GGUF_MAGIC);
        u32le(&mut b, 3);
        u64le(&mut b, 12); // n_tensors
        u64le(&mut b, 7); // n_kv
        put_str(&mut b, "general.architecture");
        u32le(&mut b, 8);
        put_str(&mut b, "qwen2");
        kv_u32(&mut b, "qwen2.embedding_length", 4);
        kv_u32(&mut b, "qwen2.block_count", 1);
        kv_u32(&mut b, "qwen2.attention.head_count", 2);
        kv_u32(&mut b, "qwen2.attention.head_count_kv", 1);
        kv_u32(&mut b, "qwen2.feed_forward_length", 8);
        kv_u32(&mut b, "qwen2.vocab_size", 5);
        tninfo(&mut b, "token_embd.weight", &[4, 5], 0, 0);
        tninfo(&mut b, "output.weight", &[4, 5], 0, 0);
        tninfo(&mut b, "norm.weight", &[4], 0, 0);
        tninfo(&mut b, "blk.0.attn_norm.weight", &[4], 0, 0);
        tninfo(&mut b, "blk.0.attn_q.weight", &[4, 4], 0, 0);
        tninfo(
            &mut b,
            "blk.0.attn_k.weight",
            &[4, if bad_k { 3 } else { 2 }],
            0,
            0,
        );
        tninfo(&mut b, "blk.0.attn_v.weight", &[4, 2], 0, 0);
        tninfo(&mut b, "blk.0.attn_output.weight", &[4, 4], 0, 0);
        tninfo(&mut b, "blk.0.ffn_norm.weight", &[4], 0, 0);
        tninfo(&mut b, "blk.0.ffn_gate.weight", &[4, 8], 0, 0);
        tninfo(&mut b, "blk.0.ffn_up.weight", &[4, 8], 0, 0);
        tninfo(&mut b, "blk.0.ffn_down.weight", &[8, 4], 0, 0);
        b
    }

    #[test]
    fn validates_matching_tensor_geometry() {
        let file = crate::gguf::parse(&synth_model(false)).unwrap();
        let c = ModelConfig::from_file(&file).unwrap();
        assert_eq!(c.validate(&file), Vec::<String>::new());
    }

    #[test]
    fn validate_flags_dim_mismatch() {
        let file = crate::gguf::parse(&synth_model(true)).unwrap();
        let c = ModelConfig::from_file(&file).unwrap();
        let errs = c.validate(&file);
        assert_eq!(errs.len(), 1, "got {errs:?}");
        assert!(errs[0].contains("attn_k"), "{errs:?}");
    }

    // ---- PH-1301: the qwen35 hybrid (MiniCPM-V 4.7 text core, PH-1201 spike) ----

    fn kv_str(b: &mut Vec<u8>, k: &str, v: &str) {
        put_str(b, k);
        u32le(b, 8); // Str
        put_str(b, v);
    }

    fn kv_bool_array(b: &mut Vec<u8>, k: &str, vals: &[bool]) {
        put_str(b, k);
        u32le(b, 9); // Array
        u32le(b, 7); // of Bool
        u64le(b, vals.len() as u64);
        for &v in vals {
            b.push(u8::from(v));
        }
    }

    fn kv_i32_array(b: &mut Vec<u8>, k: &str, vals: &[i32]) {
        put_str(b, k);
        u32le(b, 9); // Array
        u32le(b, 5); // of I32
        u64le(b, vals.len() as u64);
        for &v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// qwen35-1B scaled to a toy: dm 8, 2 heads × head_dim 4 (key_length!), kv 1,
    /// ffn 8, vocab 5, interval 2, layers [Recurrent, Full], GDN inner 8 (qkv 24),
    /// rank 2, conv 2, state 4. `bad_conv` shrinks `blk.0.ssm_conv1d` by one.
    fn synth_qwen35(bad_conv: bool) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(GGUF_MAGIC);
        u32le(&mut b, 3);
        u64le(&mut b, 28); // n_tensors (3 embd + GDN layer 14 + full layer 11; no ffn_norm)
        u64le(&mut b, 18); // n_kv
        kv_str(&mut b, "general.architecture", "qwen35");
        kv_u32(&mut b, "qwen35.embedding_length", 8);
        kv_u32(&mut b, "qwen35.block_count", 2);
        kv_u32(&mut b, "qwen35.attention.head_count", 2);
        kv_u32(&mut b, "qwen35.attention.head_count_kv", 1);
        kv_u32(&mut b, "qwen35.attention.key_length", 4);
        kv_u32(&mut b, "qwen35.feed_forward_length", 8);
        kv_u32(&mut b, "qwen35.vocab_size", 5);
        kv_u32(&mut b, "qwen35.full_attention_interval", 2);
        kv_bool_array(&mut b, "qwen35.attention.recurrent_layers", &[true, false]);
        kv_u32(&mut b, "qwen35.ssm.conv_kernel", 2);
        kv_u32(&mut b, "qwen35.ssm.state_size", 4);
        kv_u32(&mut b, "qwen35.ssm.group_count", 2);
        kv_u32(&mut b, "qwen35.ssm.time_step_rank", 2);
        kv_u32(&mut b, "qwen35.ssm.inner_size", 8);
        kv_u32(&mut b, "qwen35.rope.dimension_count", 6);
        kv_i32_array(&mut b, "qwen35.rope.dimension_sections", &[1, 1, 2, 0]);
        kv_str(&mut b, "qwen35.rope.section_order", "zyxt");
        tninfo(&mut b, "token_embd.weight", &[8, 5], 0, 0);
        tninfo(&mut b, "output.weight", &[8, 5], 0, 0);
        tninfo(&mut b, "output_norm.weight", &[8], 0, 0);
        // blk.0 — Recurrent (GDN)
        tninfo(&mut b, "blk.0.attn_norm.weight", &[8], 0, 0);
        tninfo(&mut b, "blk.0.post_attention_norm.weight", &[8], 0, 0);
        tninfo(&mut b, "blk.0.attn_qkv.weight", &[8, 24], 0, 0);
        tninfo(&mut b, "blk.0.attn_gate.weight", &[8, 8], 0, 0);
        tninfo(&mut b, "blk.0.ssm_a", &[2], 0, 0);
        tninfo(&mut b, "blk.0.ssm_alpha.weight", &[8, 2], 0, 0);
        tninfo(&mut b, "blk.0.ssm_beta.weight", &[8, 2], 0, 0);
        tninfo(
            &mut b,
            "blk.0.ssm_conv1d.weight",
            if bad_conv { &[2, 23] } else { &[2, 24] },
            0,
            0,
        );
        tninfo(&mut b, "blk.0.ssm_dt.bias", &[2], 0, 0);
        tninfo(&mut b, "blk.0.ssm_norm.weight", &[4], 0, 0);
        tninfo(&mut b, "blk.0.ssm_out.weight", &[8, 8], 0, 0);
        tninfo(&mut b, "blk.0.ffn_gate.weight", &[8, 8], 0, 0);
        tninfo(&mut b, "blk.0.ffn_up.weight", &[8, 8], 0, 0);
        tninfo(&mut b, "blk.0.ffn_down.weight", &[8, 8], 0, 0);
        // blk.1 — Full (q+gate 16 = 2·8, kv 4, q/k_norm 4, output [8, 8])
        tninfo(&mut b, "blk.1.attn_norm.weight", &[8], 0, 0);
        tninfo(&mut b, "blk.1.post_attention_norm.weight", &[8], 0, 0);
        tninfo(&mut b, "blk.1.attn_q.weight", &[8, 16], 0, 0);
        tninfo(&mut b, "blk.1.attn_k.weight", &[8, 4], 0, 0);
        tninfo(&mut b, "blk.1.attn_v.weight", &[8, 4], 0, 0);
        tninfo(&mut b, "blk.1.attn_q_norm.weight", &[4], 0, 0);
        tninfo(&mut b, "blk.1.attn_k_norm.weight", &[4], 0, 0);
        tninfo(&mut b, "blk.1.attn_output.weight", &[8, 8], 0, 0);
        tninfo(&mut b, "blk.1.ffn_gate.weight", &[8, 8], 0, 0);
        tninfo(&mut b, "blk.1.ffn_up.weight", &[8, 8], 0, 0);
        tninfo(&mut b, "blk.1.ffn_down.weight", &[8, 8], 0, 0);
        b
    }

    #[test]
    fn parses_qwen35_hybrid_config() {
        let file = crate::gguf::parse(&synth_qwen35(false)).unwrap();
        let c = ModelConfig::from_file(&file).expect("qwen35 config");
        assert_eq!(c.arch, "qwen35");
        assert_eq!(c.d_model, 8);
        assert_eq!(c.key_length, 4);
        // The hybrid head is key_length WIDE, not d_model / n_heads.
        assert_eq!(c.head_dim(), 4);
        assert_eq!(c.n_layers, 2);
        assert!(c.is_hybrid());
        assert_eq!(c.full_attention_interval, 2);
        assert_eq!(c.recurrent_layers, vec![true, false]);
        assert_eq!(c.layer_kind(0), LayerKind::Recurrent);
        assert_eq!(c.layer_kind(1), LayerKind::Full);
        assert_eq!(c.ssm_conv_kernel, 2);
        assert_eq!(c.ssm_state_size, 4);
        assert_eq!(c.ssm_group_count, 2);
        assert_eq!(c.ssm_time_step_rank, 2);
        assert_eq!(c.ssm_inner_size, 8);
        assert_eq!(c.gdn_qkv_dim(), 24);
        assert_eq!(c.full_q_dim(), 8);
        assert_eq!(c.rope_dimension_count, 6);
        assert_eq!(c.rope_sections, vec![1, 1, 2, 0]);
        assert_eq!(c.rope_section_order, "zyxt");
        assert!(!c.is_moe());
    }

    #[test]
    fn validates_qwen35_hybrid_layer_kinds() {
        let file = crate::gguf::parse(&synth_qwen35(false)).unwrap();
        let c = ModelConfig::from_file(&file).unwrap();
        assert_eq!(c.validate(&file), Vec::<String>::new());
    }

    #[test]
    fn validate_qwen35_flags_gdn_dim_mismatch() {
        let file = crate::gguf::parse(&synth_qwen35(true)).unwrap();
        let c = ModelConfig::from_file(&file).unwrap();
        let errs = c.validate(&file);
        assert_eq!(errs.len(), 1, "got {errs:?}");
        assert!(errs[0].contains("ssm_conv1d"), "{errs:?}");
    }

    #[test]
    #[ignore = "real MiniCPM-V 4.7-1B GGUF (gitignored download); run: cargo test --release -- --ignored"]
    fn real_minicpm_v47_1b_config_and_validate() {
        let path = std::path::Path::new("data/models/MiniCPM-V-4_7-1B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("PH-1301 skip: MiniCPM-V 4.7-1B GGUF absent (gitignored download)");
            return;
        }
        let file = crate::gguf::parse_file(path, crate::gguf::INSPECT_WINDOW).expect("parse");
        let c = ModelConfig::from_file(&file).expect("qwen35 config");
        assert_eq!(c.n_layers, 24);
        assert_eq!(c.d_model, 1024);
        assert_eq!(c.key_length, 256);
        assert_eq!(c.head_dim(), 256);
        assert_eq!(c.rope_sections, vec![11, 11, 10, 0]);
        assert_eq!(c.rope_section_order, "zyxt");
        assert_eq!(c.ssm_inner_size, 2048);
        assert_eq!(c.ssm_group_count, 16);
        assert_eq!(c.gdn_qkv_dim(), 6144);
        assert_eq!(c.full_q_dim(), 2048);
        // The truth gate: names + dims per kind must match the shipped file exactly.
        assert_eq!(c.validate(&file), Vec::<String>::new());
        // Whatever the Bool[24] says, the full layers are the ones with split q/k/v.
        let fulls: Vec<usize> = (0..24)
            .filter(|&i| c.layer_kind(i) == LayerKind::Full)
            .collect();
        eprintln!("PH-1301: full-attention layers = {fulls:?}");
        assert_eq!(
            fulls.len(),
            6,
            "interval 4 over 24 layers ⇒ six full blocks"
        );
        assert!(
            fulls.contains(&3),
            "blk.3 carries attn_q/k/v on the real file"
        );
    }
}
