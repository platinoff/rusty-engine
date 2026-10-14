//! GGUF tokenizer **metadata ingest** — band B1 `PH-105`.
//!
//! Reads the `tokenizer.ggml.*` KV block into a typed [`TokenizerMeta`] (vocab, types,
//! scores, merges, special-token ids, pre-tokenizer selector). Keys/semantics cached in
//! `docs/research/ggml-blocks.md`. This is *not* the BPE encode/decode algorithm — that
//! lands in band B3 once there is a model to run; here we only surface what the container
//! declares so the sampler/tokenizer in B3 has its vocabulary.

use crate::gguf::{GgufFile, Value, ValueKind};

/// Tokenizer description pulled from GGUF metadata.
#[derive(Debug, Clone, Default)]
pub struct TokenizerMeta {
    pub model: String,
    pub tokens: Vec<String>,
    pub token_type: Vec<i32>,
    pub scores: Vec<f32>,
    pub merges: Vec<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub unknown_token_id: Option<u32>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub add_space_prefix: Option<bool>,
    pub remove_extra_whitespaces: Option<bool>,
    pub pre: Option<String>,
}

impl TokenizerMeta {
    /// Extract every `tokenizer.ggml.*` field present in `file`; absent fields become
    /// `None`/empty (a container may carry a subset).
    pub fn from_file(file: &GgufFile) -> Self {
        Self {
            model: s(file.get("tokenizer.ggml.model")).unwrap_or_default(),
            tokens: arr_str(file.get("tokenizer.ggml.tokens")),
            token_type: arr_i32(file.get("tokenizer.ggml.token_type")),
            scores: arr_f32(file.get("tokenizer.ggml.scores")),
            merges: arr_str(file.get("tokenizer.ggml.merges")),
            bos_token_id: u32v(file.get("tokenizer.ggml.bos_token_id")),
            eos_token_id: u32v(file.get("tokenizer.ggml.eos_token_id")),
            padding_token_id: u32v(file.get("tokenizer.ggml.padding_token_id")),
            unknown_token_id: u32v(file.get("tokenizer.ggml.unknown_token_id")),
            add_bos_token: boolv(file.get("tokenizer.ggml.add_bos_token")),
            add_eos_token: boolv(file.get("tokenizer.ggml.add_eos_token")),
            add_space_prefix: boolv(file.get("tokenizer.ggml.add_space_prefix")),
            remove_extra_whitespaces: boolv(file.get("tokenizer.ggml.remove_extra_whitespaces")),
            pre: s(file.get("tokenizer.ggml.pre")),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Token text by id, if within the vocab.
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Whether BOS should be prepended (defaults to `false` if unset).
    pub fn prepend_bos(&self) -> bool {
        self.add_bos_token.unwrap_or(false)
    }
}

fn s(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::Str(x)) => Some(x.clone()),
        _ => None,
    }
}

fn u32v(v: Option<&Value>) -> Option<u32> {
    match v {
        Some(Value::U32(x)) => Some(*x),
        Some(Value::U64(x)) => u32::try_from(*x).ok(),
        _ => None,
    }
}

fn boolv(v: Option<&Value>) -> Option<bool> {
    match v {
        Some(Value::Bool(b)) => Some(*b),
        _ => None,
    }
}

fn arr_str(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(ValueKind::Str, items)) => items
            .iter()
            .filter_map(|i| match i {
                Value::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn arr_i32(v: Option<&Value>) -> Vec<i32> {
    match v {
        Some(Value::Array(ValueKind::I32, items)) => items
            .iter()
            .filter_map(|i| match i {
                Value::I32(x) => Some(*x),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn arr_f32(v: Option<&Value>) -> Vec<f32> {
    match v {
        Some(Value::Array(ValueKind::F32, items)) => items
            .iter()
            .filter_map(|i| match i {
                Value::F32(x) => Some(*x),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
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
    fn f32le(b: &mut Vec<u8>, x: f32) {
        b.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    fn s(b: &mut Vec<u8>, v: &str) {
        u64le(b, v.len() as u64);
        b.extend_from_slice(v.as_bytes());
    }
    fn kv_str(b: &mut Vec<u8>, key: &str, val: &str) {
        s(b, key);
        u32le(b, 8);
        s(b, val);
    }
    fn kv_u32(b: &mut Vec<u8>, key: &str, val: u32) {
        s(b, key);
        u32le(b, 4);
        u32le(b, val);
    }
    fn kv_bool(b: &mut Vec<u8>, key: &str, val: bool) {
        s(b, key);
        u32le(b, 7);
        b.push(val as u8);
    }
    fn kv_arr_str(b: &mut Vec<u8>, key: &str, items: &[&str]) {
        s(b, key);
        u32le(b, 9);
        u32le(b, 8);
        u64le(b, items.len() as u64);
        for it in items {
            s(b, it);
        }
    }
    fn kv_arr_i32(b: &mut Vec<u8>, key: &str, items: &[i32]) {
        s(b, key);
        u32le(b, 9);
        u32le(b, 5);
        u64le(b, items.len() as u64);
        for &it in items {
            u32le(b, it as u32);
        }
    }
    fn kv_arr_f32(b: &mut Vec<u8>, key: &str, items: &[f32]) {
        s(b, key);
        u32le(b, 9);
        u32le(b, 6);
        u64le(b, items.len() as u64);
        for &it in items {
            f32le(b, it);
        }
    }

    fn synth_tokenizer_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(GGUF_MAGIC);
        u32le(&mut b, 3);
        u64le(&mut b, 0); // n_tensors
        u64le(&mut b, 8); // n_kv
        kv_str(&mut b, "tokenizer.ggml.model", "gpt2");
        kv_arr_str(&mut b, "tokenizer.ggml.tokens", &["a", "b", "c"]);
        kv_arr_i32(&mut b, "tokenizer.ggml.token_type", &[1, 3, 3]);
        kv_arr_f32(&mut b, "tokenizer.ggml.scores", &[0.5, 0.25, 0.125]);
        kv_arr_str(&mut b, "tokenizer.ggml.merges", &["a b", "ab c"]);
        kv_u32(&mut b, "tokenizer.ggml.bos_token_id", 1);
        kv_u32(&mut b, "tokenizer.ggml.eos_token_id", 2);
        kv_bool(&mut b, "tokenizer.ggml.add_bos_token", true);
        b
    }

    #[test]
    fn ingests_tokenizer_metadata() {
        let file = crate::gguf::parse(&synth_tokenizer_gguf()).expect("parse");
        let t = TokenizerMeta::from_file(&file);
        assert_eq!(t.model, "gpt2");
        assert_eq!(t.vocab_size(), 3);
        assert_eq!(t.token_text(0), Some("a"));
        assert_eq!(t.token_text(5), None);
        assert_eq!(t.token_type, vec![1, 3, 3]);
        assert_eq!(t.scores, vec![0.5, 0.25, 0.125]);
        assert_eq!(t.merges.len(), 2);
        assert_eq!(t.bos_token_id, Some(1));
        assert_eq!(t.eos_token_id, Some(2));
        assert!(t.prepend_bos());
        assert_eq!(t.padding_token_id, None); // absent key
        assert_eq!(t.pre, None);
    }

    #[test]
    fn default_when_no_tokenizer_keys() {
        // A GGUF header with no tokenizer.* KV → empty meta (not a panic).
        let mut b = Vec::new();
        b.extend_from_slice(GGUF_MAGIC);
        u32le(&mut b, 3);
        u64le(&mut b, 0);
        u64le(&mut b, 0);
        let file = crate::gguf::parse(&b).unwrap();
        let t = TokenizerMeta::from_file(&file);
        assert!(t.model.is_empty());
        assert_eq!(t.vocab_size(), 0);
        assert!(!t.prepend_bos());
    }
}
