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
}

impl ModelConfig {
    /// Per-head dimension (`d_model / n_heads`; assumes `n_heads` divides `d_model`).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads.max(1)
    }

    pub fn is_moe(&self) -> bool {
        self.n_experts > 0
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
}
