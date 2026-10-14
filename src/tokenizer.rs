//! GGUF tokenizer metadata ingest + a **byte-level BPE** codec — band B1/B3.
//!
//! `PH-105`: read the `tokenizer.ggml.*` block into [`TokenizerMeta`] (keys cached in
//! `docs/research/ggml-blocks.md`). `PH-105b` adds [`Bpe`]: a GPT-2-style byte-level
//! BPE encoder/decoder (byte↔char table, whitespace pre-tokenizer, merge-by-rank). It is
//! verified by round-trip tests on synthetic vocab — the *algorithm* is correct; matching a
//! specific model's HF tokenizer (Qwen `pre`-regex + byte-fallback + specials) is a separate
//! parity slice (`PH-305b`). Ingest-only fields (`model/tokens/scores/…`) come from B1.

use crate::core::{EngineError, Result};
use crate::gguf::{GgufFile, Value, ValueKind};
use std::collections::HashMap;

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
    /// Extract every `tokenizer.ggml.*` field present in `file`.
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

    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn prepend_bos(&self) -> bool {
        self.add_bos_token.unwrap_or(false)
    }

    /// Special/control tokens: `token_type >= CONTROL (3)` and registered in the vocab.
    /// llama.cpp marks CONTROL=3, USER_DEFINED=4 — those must be atomic in encoding.
    pub fn special_tokens(&self) -> Vec<(String, u32)> {
        self.tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| self.token_type.get(*i).copied().unwrap_or(0) >= 3)
            .map(|(i, t)| (t.clone(), i as u32))
            .collect()
    }
}

/// GPT-2 byte→visible-unicode table (the classic reversible mapping so any byte survives BPE).
#[derive(Debug, Clone)]
pub struct ByteEncoder {
    byte_to_char: Vec<char>,
    char_to_byte: HashMap<char, u8>,
}

impl ByteEncoder {
    pub fn new() -> Self {
        // NB: byte 0..=255; the "printable-ish" ranges get identity, the rest remap to 0x100+.
        let mut bs: Vec<u32> = Vec::new();
        for c in 0x21u32..=0x7E {
            bs.push(c); // '!'..'~'
        }
        for c in 0xA1u32..=0xAC {
            bs.push(c); // '¡'..'¬'
        }
        for c in 0xAEu32..=0xFF {
            bs.push(c); // '®'..'ÿ'
        }
        let mut cs: Vec<u32> = bs.clone();
        let mut n = 0u32;
        for b in 0u32..256 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(0x100 + n);
                n += 1;
            }
        }
        // zip bs(byte) -> cs(char), ordered by byte id.
        let mut pairs: Vec<(u32, u32)> = bs.iter().cloned().zip(cs.iter().cloned()).collect();
        pairs.sort_by_key(|(b, _)| *b);
        let mut byte_to_char = vec!['?'; 256];
        let mut char_to_byte = HashMap::new();
        for (b, c) in pairs {
            let ch = char::from_u32(c).unwrap_or('\u{FFFD}');
            byte_to_char[b as usize] = ch;
            char_to_byte.insert(ch, b as u8);
        }
        Self {
            byte_to_char,
            char_to_byte,
        }
    }

    /// Map raw bytes to the visible-unicode space (injective).
    pub fn bytes_to_chars(&self, bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|&b| self.byte_to_char[b as usize])
            .collect()
    }

    /// Inverse: visible-unicode space back to raw bytes.
    pub fn chars_to_bytes(&self, s: &str) -> Vec<u8> {
        s.chars()
            .map(|c| *self.char_to_byte.get(&c).unwrap_or(&0))
            .collect()
    }
}

impl Default for ByteEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Whitespace pre-tokenizer marker: a space byte (0x20) is mapped to `Ġ` by [`ByteEncoder`].
const SPACE_MARK: char = '\u{0120}'; // == byte_to_char[0x20]

/// Byte-level BPE codec (GPT-2 style) over a [`TokenizerMeta`].
#[derive(Debug, Clone)]
pub struct Bpe {
    meta: TokenizerMeta,
    encoder: ByteEncoder,
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    /// Atomic special/control tokens (`token_type >= 3`), matched before BPE.
    specials: HashMap<String, u32>,
}

impl Bpe {
    /// Build the codec from tokenizer metadata (`tokens` → vocab, `merges` → pair ranks).
    pub fn from_meta(meta: TokenizerMeta) -> Self {
        let encoder = ByteEncoder::new();
        let mut vocab = HashMap::with_capacity(meta.tokens.len());
        for (i, tok) in meta.tokens.iter().enumerate() {
            vocab.insert(tok.clone(), i as u32);
        }
        let mut ranks = HashMap::with_capacity(meta.merges.len());
        for (rank, m) in meta.merges.iter().enumerate() {
            // merge entry is "tokA tokB" (space-separated token pair)
            if let Some((a, b)) = m.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), rank);
            }
        }
        let specials = meta
            .special_tokens()
            .into_iter()
            .filter(|(t, _)| t.contains("<|") || t.contains("|>"))
            .collect();
        Self {
            meta,
            encoder,
            vocab,
            ranks,
            specials,
        }
    }

    /// Split byte-encoded text into word chunks at the space marker (first char of a non-initial
    /// chunk keeps its `Ġ` so a space round-trips).
    fn pre_tokenize(&self, byte_text: &str) -> Vec<String> {
        let mut words = Vec::new();
        for (i, part) in byte_text.split(SPACE_MARK).enumerate() {
            if part.is_empty() && i > 0 {
                continue;
            }
            if i == 0 {
                if !part.is_empty() {
                    words.push(part.to_string());
                }
            } else {
                words.push(format!("{}{part}", SPACE_MARK));
            }
        }
        words
    }

    /// Apply BPE merges to one byte-encoded word (list of single-char symbols) until no merge
    /// applies; returns the resulting tokens (which may still be missing from `vocab`).
    fn merge_word(&self, word: &str) -> Vec<String> {
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        while syms.len() > 1 {
            let mut best: Option<(usize, usize)> = None; // (position, rank)
            for i in 0..syms.len() - 1 {
                if let Some(&r) = self.ranks.get(&(syms[i].clone(), syms[i + 1].clone())) {
                    if best.is_none() || r < best.unwrap().1 {
                        best = Some((i, r));
                    }
                }
            }
            match best {
                Some((i, _)) => {
                    syms[i] = format!("{}{}", syms[i], syms[i + 1]);
                    syms.remove(i + 1);
                }
                None => break,
            }
        }
        syms
    }

    /// Emit the byte ids of a merged symbol: exact vocab hit → one id; otherwise
    /// **byte-fallback** — split into single chars (each a ByteEncoder output that is a
    /// vocab byte-token). Errors only if even a single byte char is missing.
    fn encode_symbol(&self, sym: &str, ids: &mut Vec<u32>) -> Result<()> {
        if let Some(&id) = self.vocab.get(sym) {
            ids.push(id);
            return Ok(());
        }
        // byte-fallback: the symbol came from pre_tokenize over byte-mapped chars, so its
        // constituent chars are all in the byte-token vocab.
        for c in sym.chars() {
            let one = c.to_string();
            match self.vocab.get(&one) {
                Some(&id) => ids.push(id),
                None => {
                    return Err(EngineError {
                        message: format!("BPE: byte-fallback char {one:?} not in vocab"),
                    })
                }
            }
        }
        Ok(())
    }

    /// Encode text to token ids. Special/control tokens (`token_type >= 3`) are atomic;
    /// between them the byte-level BPE runs with byte-fallback for unseen merges.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for segment in split_on_specials(text, &self.specials) {
            match segment {
                Segment::Special(s) => ids.push(self.specials[&s]),
                Segment::Text(s) => {
                    let byte_text = self.encoder.bytes_to_chars(s.as_bytes());
                    for word in self.pre_tokenize(&byte_text) {
                        for sym in self.merge_word(&word) {
                            self.encode_symbol(&sym, &mut ids)?;
                        }
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Decode token ids to text (vocab tokens → byte-space → UTF-8).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            if let Some(tok) = self.meta.token_text(id) {
                s.push_str(tok);
            }
        }
        String::from_utf8_lossy(&self.encoder.chars_to_bytes(&s)).into_owned()
    }

    pub fn meta(&self) -> &TokenizerMeta {
        &self.meta
    }
}

// ── special-token splitter (B3 `PH-305b`) ─────────────────────────────────────

/// A piece of a decoded prompt: either an atomic special token or ordinary BPE text.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment {
    Special(String),
    Text(String),
}

/// Split `text` into [`Segment`]s on registered special tokens (longest match wins).
/// Without specials this is a single `Text` segment.
fn split_on_specials(text: &str, specials: &HashMap<String, u32>) -> Vec<Segment> {
    if specials.is_empty() {
        return vec![Segment::Text(text.to_string())];
    }
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut rest = text;
    while !rest.is_empty() {
        if rest.starts_with("<|") {
            let hit = specials
                .keys()
                .filter(|s| rest.starts_with(s.as_str()))
                .max_by_key(|s| s.len());
            if let Some(sp) = hit {
                if !buf.is_empty() {
                    out.push(Segment::Text(std::mem::take(&mut buf)));
                }
                out.push(Segment::Special(sp.clone()));
                rest = &rest[sp.len()..];
                continue;
            }
        }
        let c = rest.chars().next().unwrap();
        buf.push(c);
        rest = &rest[c.len_utf8()..];
    }
    if !buf.is_empty() {
        out.push(Segment::Text(buf));
    }
    out
}

// ── metadata ingest helpers (B1, unchanged) ──────────────────────────────────

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
    fn s_(b: &mut Vec<u8>, v: &str) {
        u64le(b, v.len() as u64);
        b.extend_from_slice(v.as_bytes());
    }
    fn kv_str(b: &mut Vec<u8>, key: &str, val: &str) {
        s_(b, key);
        u32le(b, 8);
        s_(b, val);
    }
    fn kv_u32(b: &mut Vec<u8>, key: &str, val: u32) {
        s_(b, key);
        u32le(b, 4);
        u32le(b, val);
    }
    fn kv_bool(b: &mut Vec<u8>, key: &str, val: bool) {
        s_(b, key);
        u32le(b, 7);
        b.push(val as u8);
    }
    fn kv_arr_str(b: &mut Vec<u8>, key: &str, items: &[&str]) {
        s_(b, key);
        u32le(b, 9);
        u32le(b, 8);
        u64le(b, items.len() as u64);
        for it in items {
            s_(b, it);
        }
    }
    fn kv_arr_i32(b: &mut Vec<u8>, key: &str, items: &[i32]) {
        s_(b, key);
        u32le(b, 9);
        u32le(b, 5);
        u64le(b, items.len() as u64);
        for &it in items {
            u32le(b, it as u32);
        }
    }
    fn kv_arr_f32(b: &mut Vec<u8>, key: &str, items: &[f32]) {
        s_(b, key);
        u32le(b, 9);
        u32le(b, 6);
        u64le(b, items.len() as u64);
        for &it in items {
            b.extend_from_slice(&it.to_bits().to_le_bytes());
        }
    }

    fn synth_tokenizer_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(GGUF_MAGIC);
        u32le(&mut b, 3);
        u64le(&mut b, 0);
        u64le(&mut b, 8);
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
        assert_eq!(t.padding_token_id, None);
    }

    #[test]
    fn default_when_no_tokenizer_keys() {
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

    #[test]
    fn byte_encoder_is_invertible_over_all_256_bytes() {
        let e = ByteEncoder::new();
        let bytes: Vec<u8> = (0..256u32).map(|b| b as u8).collect();
        let chars = e.bytes_to_chars(&bytes);
        assert_eq!(e.chars_to_bytes(&chars), bytes);
        // space (0x20) maps to the 'Ġ' marker used as the word separator
        assert_eq!(e.bytes_to_chars(&[0x20]), "Ġ");
    }

    #[test]
    fn bpe_merges_by_rank_and_roundtrips() {
        // vocab: a b c d ab abc abcd ; merges: a b, ab c, abc d
        let meta = TokenizerMeta {
            tokens: ["a", "b", "c", "d", "ab", "abc", "abcd"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            merges: ["a b", "ab c", "abc d"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        };
        let bpe = Bpe::from_meta(meta);
        let ids = bpe.encode("abcd").unwrap();
        assert_eq!(ids, vec![6]); // fully merged to "abcd"
        assert_eq!(bpe.decode(&ids), "abcd");
    }

    #[test]
    fn bpe_roundtrips_whitespace_words() {
        // byte-level vocab: every ASCII byte as its own token, no merges -> round-trip holds.
        let e = ByteEncoder::new();
        let mut tokens = Vec::new();
        for b in 0..=127u32 {
            tokens.push(e.bytes_to_chars(&[b as u8]));
        }
        let meta = TokenizerMeta {
            tokens,
            ..Default::default()
        };
        let bpe = Bpe::from_meta(meta);
        let text = "hi there";
        let ids = bpe.encode(text).unwrap();
        assert_eq!(bpe.decode(&ids), text);
    }

    #[test]
    fn bpe_unknown_symbol_errors() {
        // only "a" in vocab; "b" cannot merge and its single-byte token is absent too.
        let meta = TokenizerMeta {
            tokens: vec!["a".to_string()], // only "a"
            ..Default::default()
        };
        let bpe = Bpe::from_meta(meta);
        assert!(bpe.encode("b").is_err());
    }

    #[test]
    fn bpe_byte_fallback_handles_unseen_merge() {
        // vocab a,b,c,ab ; merge "a b" makes "ab" a token. "ba" is not in vocab and has no
        // merge -> byte-falls-back to the single-byte tokens [b,a].
        let meta = TokenizerMeta {
            tokens: ["a", "b", "c", "ab"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            merges: ["a b"].iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        let bpe = Bpe::from_meta(meta);
        assert_eq!(bpe.encode("ab").unwrap(), vec![3]);
        assert_eq!(bpe.encode("ba").unwrap(), vec![1, 0]);
        assert_eq!(bpe.decode(&bpe.encode("abba").unwrap()), "abba");
    }

    #[test]
    fn specials_are_atomic_and_split() {
        let mut meta = TokenizerMeta {
            tokens: ["a", "b", "c", "<|x|>"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            token_type: vec![1, 1, 1, 3],
            ..Default::default()
        };
        meta.merges.clear();
        let bpe = Bpe::from_meta(meta);
        // "<|x|>" is special (atomic id 3); "ab"/"c" are byte tokens 0/1/2.
        let ids = bpe.encode("a<|x|>c").unwrap();
        assert_eq!(ids, vec![0, 3, 2]);
        assert_eq!(bpe.decode(&ids), "a<|x|>c");
    }
}
