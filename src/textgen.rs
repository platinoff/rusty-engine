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
use crate::runner::{DenseDecoder, DenseRunner};
use crate::runner35::{Q35Decoder, Q35Runner};
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
        // PH-307: decode incrementally over the KV cache (O(n) per token) instead of
        // re-prefilling the whole prefix every step (O(n^2)); identical greedy stream.
        let mut dec = DenseDecoder::new(&self.runner);
        let out = dec.generate_incremental(&ids, max_new);
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
        // PH-307: KV-cache incremental greedy (temperature 0 == argmax) instead of the
        // re-prefill `generate_with`; the token stream is identical (see dense_gen test).
        let mut dec = DenseDecoder::new(&self.runner);
        let out = dec.generate_incremental(&ids, max_new);
        let new = &out[ids.len().min(out.len())..];
        Ok(self.bpe.decode(new))
    }
}

/// `qwen35` text pipeline — band B13 `PH-1304`. Same shape as [`TextRunner`] (BPE in,
/// greedy hybrid generation, BPE out) over the [`Q35Runner`] delta-state decoder. The
/// serve wiring lands in `PH-1306`; this is the CLI + test path.
pub struct TextRunner35 {
    pub runner: Q35Runner,
    pub bpe: Bpe,
}

impl TextRunner35 {
    /// Build from a mapped GGUF (config must be the validated `qwen35` geometry).
    pub fn from_model(model: &MmapModel) -> Result<Self> {
        let header = model.header();
        let cfg = ModelConfig::from_file(header)?;
        let problems = cfg.validate(header);
        if !problems.is_empty() {
            return Err(EngineError {
                message: format!("textgen35: geometry invalid: {}", problems[0]),
            });
        }
        let meta = TokenizerMeta::from_file(header);
        if meta.tokens.is_empty() {
            return Err(EngineError {
                message: "textgen35: model has no tokenizer.ggml.tokens".to_string(),
            });
        }
        let runner = Q35Runner::from_model(model, &cfg)?;
        let bpe = Bpe::from_meta(meta);
        Ok(Self { runner, bpe })
    }

    /// Encode, greedily generate up to `max_new` tokens (EOS stops), decode the new
    /// tokens only.
    pub fn run_text(&self, prompt: &str, max_new: usize) -> Result<String> {
        let ids = self.bpe.encode(prompt)?;
        if ids.is_empty() {
            return Err(EngineError {
                message: "textgen35: prompt tokenized to empty".to_string(),
            });
        }
        let mut dec = Q35Decoder::new(&self.runner);
        let out = dec.generate_incremental(&ids, max_new);
        let new = &out[ids.len().min(out.len())..];
        Ok(self.bpe.decode(new))
    }

    /// Encode + echo token ids.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.bpe.encode(text)
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

/// One chat completion plus the decode stats the serve loop reports (PH-505).
pub struct ChatOutcome {
    pub response: crate::http::Response,
    /// Newly generated tokens (`usage.completion_tokens`).
    pub generated_tokens: u64,
    /// Wall-clock seconds spent generating this completion.
    pub decode_secs: f64,
    /// MoE experts materialized for this completion. The dense runner activates none, so
    /// this is 0 until the MoE path (PH-406) reports its top-k breadth.
    pub active_experts: u64,
}

/// HTTP chat handler (PH-501b). `json_body` is the raw request body (messages JSON);
/// returns the full OpenAI-shaped HTTP response for POST /v1/chat/completions plus the
/// decode stats (PH-505).
pub fn chat_response(runner: &TextRunner, json_body: &str, max_new: usize) -> ChatOutcome {
    let started = std::time::Instant::now();
    match runner.complete(json_body, max_new) {
        Ok(text) => {
            let decode_secs = started.elapsed().as_secs_f64();
            let escaped = escape_json_text(&text);
            let prompt_len = runner
                .bpe
                .encode(&extract_message_content(json_body))
                .unwrap_or_default()
                .len();
            let generated = (runner.bpe.encode(&text).map(|v| v.len()).unwrap_or(0)).max(1);
            let json = chat_completion_body(&escaped, prompt_len, generated);
            ChatOutcome {
                response: crate::http::Response::json(200, json),
                generated_tokens: generated as u64,
                decode_secs,
                active_experts: 0,
            }
        }
        Err(e) => ChatOutcome {
            response: crate::http::Response::json(400, format!(r#"{{"error":"{}"}}"#, e.message)),
            generated_tokens: 0,
            decode_secs: 0.0,
            active_experts: 0,
        },
    }
}

/// OpenAI-shaped `chat.completion` body. `content` must already be JSON-escaped
/// (see [`escape_json_text`]). Split out so the wire contract is unit-testable
/// without a loaded model: a missing **root** `}` here produced invalid JSON that
/// `curl` and the tokenizer gates accepted but the hub's `serde_json` rejected,
/// surfacing as a `null` proxy reply end-to-end (PH-507).
pub fn chat_completion_body(content: &str, prompt_len: usize, generated: usize) -> String {
    let mut body = format!(
        r#"{{"id":"cmpl-0","object":"chat.completion","model":"rusty-engine","choices":[{{"index":0,"message":{{"role":"assistant","content":"{content}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{prompt_len},"completion_tokens":{generated},"total_tokens":{total}"#,
        total = prompt_len + generated
    );
    // Close the `usage` object and the root object.
    body.push_str("}}");
    body
}

/// PH-507: SSE variant of [`chat_response`] — the completion is emitted as one
/// `chat.completion.chunk` delta event plus the terminating `[DONE]`, the same
/// minimal streaming contract `llama_serve` (bunke-rock) exposes. The hub can
/// hold `text/event-stream` against the engine without a client-visible gap.
pub fn chat_stream_response(runner: &TextRunner, json_body: &str, max_new: usize) -> ChatOutcome {
    let started = std::time::Instant::now();
    match runner.complete(json_body, max_new) {
        Ok(text) => {
            let decode_secs = started.elapsed().as_secs_f64();
            let escaped = escape_json_text(&text);
            let generated = (runner.bpe.encode(&text).map(|v| v.len()).unwrap_or(0)).max(1);
            let body = crate::http::sse_body("cmpl-0", "rusty-engine", &[&escaped]);
            ChatOutcome {
                response: crate::http::Response {
                    status: 200,
                    reason: "OK",
                    content_type: "text/event-stream",
                    body,
                },
                generated_tokens: generated as u64,
                decode_secs,
                active_experts: 0,
            }
        }
        Err(e) => ChatOutcome {
            response: crate::http::Response::json(400, format!(r#"{{"error":"{}"}}"#, e.message)),
            generated_tokens: 0,
            decode_secs: 0.0,
            active_experts: 0,
        },
    }
}

/// JSON-escape a completion for embedding in a response body (PH-501b/507).
fn escape_json_text(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
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

    #[test]
    fn chat_completion_body_is_valid_json_with_root_brace() {
        // PH-507: the body must be a complete JSON object — curl tolerated the
        // missing root `}` but the hub's `serde_json` did not (null proxy reply).
        let body = chat_completion_body(" Paris.", 5, 8);
        assert!(body.ends_with("}}"), "{body}");
        assert!(crate::worker::parse_json(&body).is_ok(), "{body}");
        assert!(body.contains("\"total_tokens\":13"), "{body}");
        assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
        assert!(body.contains("\"content\":\" Paris.\""), "{body}");
    }

    #[test]
    fn chat_completion_body_keeps_escaped_content_parseable() {
        let content = escape_json_text("a \"q\" \\ c\nd");
        let body = chat_completion_body(&content, 1, 2);
        assert!(crate::worker::parse_json(&body).is_ok(), "{body}");
    }
}
