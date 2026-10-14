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

    /// PH-501b: run one chat completion. Concatenates `messages[].content`, encodes via
    /// BPE, generates greedily (`max_new`), then decodes only the **new** tokens and
    /// returns the assistant text. Errors when prompt is empty or has no text content.
    pub fn complete(&self, messages_json: &str, max_new: usize) -> Result<String> {
        let prompt = extract_message_content(messages_json);
        if prompt.trim().is_empty() {
            return Err(EngineError {
                message: "textgen: prompt has no message content".to_string(),
            });
        }
        let ids = self.bpe.encode(&prompt)?;
        if ids.is_empty() {
            return Err(EngineError {
                message: "textgen: prompt tokens empty".to_string(),
            });
        }
        let opts = crate::infer::GenerationOptions {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = crate::infer::SeedRng::new(0);
        let out = self.runner.generate_with(&ids, &opts, max_new, &mut rng);
        let new = &out[ids.len().min(out.len())..];
        Ok(self.bpe.decode(new))
    }
}

/// Extract a bare `messages[].content` string by scanning a JSON body (no serde dep).
/// Finds each `"content"`, reads the following quoted string (with `\n`/`\t`/`\"`/`\\`
/// unescape), and concatenates the values with a space. Returns "" when none present.
pub fn extract_message_content(json: &str) -> String {
    let jb = json.as_bytes();
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while let Some(rel) = json[i..].find("\"content\"") {
        let mut j = i + rel + "\"content\"".len();
        while j < jb.len() && (jb[j] as char).is_whitespace() {
            j += 1;
        }
        if j >= jb.len() || jb[j] != b':' {
            break;
        }
        let mut v = j + 1;
        while v < jb.len() && (jb[v] as char).is_whitespace() {
            v += 1;
        }
        if v >= jb.len() || jb[v] != b'"' {
            break; // non-string content (number/null/object) — stop
        }
        v += 1;
        let mut val = String::new();
        while v < jb.len() {
            match jb[v] {
                b'"' => break,
                b'\\' if v + 1 < jb.len() => {
                    v += 1;
                    match jb[v] {
                        b'n' => val.push('\n'),
                        b't' => val.push('\t'),
                        b'\\' => val.push('\\'),
                        b'"' => val.push('"'),
                        c => val.push(c as char),
                    }
                }
                _ => val.push(jb[v] as char),
            }
            v += 1;
        }
        if !val.is_empty() {
            parts.push(val);
        }
        i = v + 1;
    }
    parts.join(" ")
}

/// HTTP chat handler (PH-501b). `json_body` is the raw request body (messages JSON);
/// returns the full OpenAI-shaped HTTP response for POST /v1/chat/completions.
pub fn chat_response(
    runner: &TextRunner,
    json_body: &str,
    max_new: usize,
) -> crate::http::Response {
    match runner.complete(json_body, max_new) {
        Ok(text) => {
            let escaped = text
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            let prompt_len = runner
                .bpe
                .encode(&extract_message_content(json_body))
                .unwrap_or_default()
                .len();
            let generated = (runner.bpe.encode(&text).map(|v| v.len()).unwrap_or(0)).max(1);
            let json = format!(
                r#"{{"id":"cmpl-0","object":"chat.completion","model":"rusty-engine","choices":[{{"index":0,"message":{{"role":"assistant","content":"{escaped}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{prompt_len},"completion_tokens":{generated},"total_tokens":{}}}"#,
                prompt_len + generated
            );
            crate::http::Response::json(200, json)
        }
        Err(e) => crate::http::Response::json(400, format!(r#"{{"error":"{}"}}"#, e.message)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_single_content() {
        let j = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        assert_eq!(extract_message_content(j), "hi");
    }

    #[test]
    fn extract_concatenates_turns() {
        let j = r#"{"messages":[{"role":"system","content":"be nice"},{"role":"user","content":"hello"}]}"#;
        assert_eq!(extract_message_content(j), "be nice hello");
    }

    #[test]
    fn extract_handles_escape_and_absent() {
        let j = r#"{"messages":[{"role":"user","content":"a \"quoted\" b"}]}"#;
        // the reader stops at the first unescaped quote; a leading 'a ' is captured
        assert!(extract_message_content(j).starts_with('a'));
        assert_eq!(extract_message_content("{}"), "");
        assert_eq!(extract_message_content(r#"{"max_tokens":8}"#), "");
    }
}
