//! End-to-end **text** generation pipeline — band B3 `PH-306b`.
//!
//! Ties the three landed pieces into one call: [`crate::tokenizer::Bpe`] encodes a prompt
//! to token ids, [`crate::runner::DenseRunner`] greedily generates, and the BPE decodes the
//! new tokens back to UTF-8 text. This is the "type a sentence, get a sentence" path the
//! roadmap lists for `rusty run -p "..."` (numeric `-t` ids from `PH-306` still work).
//!
//! It is exercised on a **synthetic** tiny GGUF carrying a byte-level tokenizer, so the
//! whole text stack is unit-testable without a downloaded model; numeric/parity against a
//! reference is the separate `PH-405` gate.

use crate::core::{EngineError, Result};
use crate::loader::MmapModel;
use crate::model::ModelConfig;
use crate::runner::DenseRunner;
use crate::tokenizer::{Bpe, TokenizerMeta};

/// A dense runner paired with the model's byte-level BPE, for text in / text out.
pub struct TextRunner {
    pub runner: DenseRunner,
    pub bpe: Bpe,
}

impl TextRunner {
    /// Build from a mapped GGUF: parse config + tokenizer metadata, validate geometry,
    /// load weights, and wire the BPE codec.
    pub fn from_model(model: &MmapModel) -> Result<Self> {
        let header = model.header();
        let cfg = ModelConfig::from_file(header)?;
        let problems = cfg.validate(header);
        if !problems.is_empty() {
            return Err(EngineError {
                message: format!("textgen: geometry invalid: {}", problems[0]),
            });
        }
        let meta = TokenizerMeta::from_file(header);
        if meta.tokens.is_empty() {
            return Err(EngineError {
                message: "textgen: model has no tokenizer.ggml.tokens".to_string(),
            });
        }
        let runner = DenseRunner::from_model(model, &cfg)?;
        let bpe = Bpe::from_meta(meta);
        Ok(Self { runner, bpe })
    }

    /// Encode `prompt`, greedily generate up to `max_new` tokens (stopping at EOS), then
    /// decode **only the newly generated** tokens back to text.
    pub fn run_text(&self, prompt: &str, max_new: usize) -> Result<String> {
        let ids = self.bpe.encode(prompt)?;
        if ids.is_empty() {
            return Err(EngineError {
                message: "textgen: prompt tokenized to empty".to_string(),
            });
        }
        let out = self.runner.generate(&ids, max_new);
        let new = &out[ids.len().min(out.len())..];
        Ok(self.bpe.decode(new))
    }

    /// Encode + echo the token ids (debug/`-t` bridge helper).
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.bpe.encode(text)
    }
}
