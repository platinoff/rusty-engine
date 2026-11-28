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

use crate::chatml;
use crate::core::{EngineError, Result};
use crate::loader::MmapModel;
use crate::model::ModelConfig;
use crate::runner::{DenseDecoder, DenseRunner};
use crate::runner35::{Q35Decoder, Q35Runner, TraceStep};
use crate::tokenizer::{Bpe, TokenizerMeta};
/// A dense runner paired with the model's byte-level BPE, for text in / text out.
pub struct TextRunner {
    pub runner: DenseRunner,
    pub bpe: Bpe,
}

/// PH-1306: engine shape the OpenAI serve path dispatches over. Dense
/// ([`TextRunner`]) and qwen35 hybrid ([`TextRunner35`]) both satisfy it, so one
/// `rusty serve` wire carries the whole family (B5 chat contract, B13 geometry).
pub trait Completes: Sync {
    /// One chat completion: extract `messages[].content`, encode, greedy-generate
    /// (`max_new`), decode only the **new** tokens back to text.
    fn complete(&self, messages_json: &str, max_new: usize) -> Result<String>;
    /// PH-1205: per-request multimodal prefix (raw little-endian f32 bytes,
    /// K vectors of d_model). Engines without vision ignore it; text-only
    /// behaviour stays bit-identical.
    fn complete_with_vision(
        &self,
        messages_json: &str,
        max_new: usize,
        _prefix_bytes: &[u8],
    ) -> Result<String> {
        self.complete(messages_json, max_new)
    }
    /// PH-1205: per-request `messages[].images`, already decoded from base64.
    /// Engines without a vision tower ignore them; text-only is unchanged.
    fn complete_with_images(
        &self,
        messages_json: &str,
        max_new: usize,
        _images: &[Vec<u8>],
    ) -> Result<String> {
        self.complete(messages_json, max_new)
    }
    /// BPE token count used for the `usage` block (0 encodes as max(1) upstairs).
    fn token_len(&self, text: &str) -> usize;
}

impl Completes for TextRunner {
    fn complete(&self, messages_json: &str, max_new: usize) -> Result<String> {
        TextRunner::complete(self, messages_json, max_new)
    }
    fn token_len(&self, text: &str) -> usize {
        self.bpe.encode(text).map(|v| v.len()).unwrap_or(0)
    }
}

impl Completes for TextRunner35 {
    fn complete(&self, messages_json: &str, max_new: usize) -> Result<String> {
        self.complete_impl(messages_json, max_new, None)
    }
    fn complete_with_vision(
        &self,
        messages_json: &str,
        max_new: usize,
        prefix_bytes: &[u8],
    ) -> Result<String> {
        let vs = decode_vision_prefix_bytes(prefix_bytes, self.runner.d_model)?;
        self.complete_impl(messages_json, max_new, Some(&vs))
    }
    fn token_len(&self, text: &str) -> usize {
        self.bpe.encode(text).map(|v| v.len()).unwrap_or(0)
    }
}

/// Read a `gsv.vision-prefix.v1` raw LE f32 file into K vectors of `d_model`.
pub fn load_vision_prefix(path: &str, d_model: usize) -> Result<Vec<Vec<f32>>> {
    if d_model == 0 {
        return Err(EngineError {
            message: "vision prefix: d_model must be > 0".to_string(),
        });
    }
    let bytes = std::fs::read(path).map_err(|e| EngineError {
        message: format!("vision prefix read {path}: {e}"),
    })?;
    if bytes.is_empty() || bytes.len() % (4 * d_model) != 0 {
        return Err(EngineError {
            message: format!(
                "vision prefix: {} bytes is not a multiple of {} (d_model={d_model})",
                bytes.len(),
                4 * d_model
            ),
        });
    }
    decode_f32_rows_msrv(&bytes, d_model)
}

/// MSRV-1.85-safe f32 row decoder (replaces `[u8]::as_chunks`, stable only
/// since 1.88). Same byte layout: rows of `d_model` little-endian f32 words.
fn decode_f32_rows_msrv(bytes: &[u8], d_model: usize) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::new();
    for row in bytes.chunks_exact(4 * d_model) {
        let v: Vec<f32> = row
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if !v.iter().all(|x| x.is_finite()) {
            return Err(EngineError {
                message: "vision prefix: non-finite value".to_string(),
            });
        }
        out.push(v);
    }
    Ok(out)
}

/// Decode raw little-endian f32 bytes into K vectors of `d_model`.
pub fn decode_vision_prefix_bytes(bytes: &[u8], d_model: usize) -> Result<Vec<Vec<f32>>> {
    if d_model == 0 || bytes.is_empty() || bytes.len() % (4 * d_model) != 0 {
        return Err(EngineError {
            message: format!(
                "vision prefix: {} bytes is not a multiple of {} (d_model={d_model})",
                bytes.len(),
                4 * d_model
            ),
        });
    }
    decode_f32_rows_msrv(bytes, d_model)
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

    /// PH-501b: run one chat completion. Applies the model's ChatML template
    /// server-side then encodes via BPE, generates greedily (`max_new`), and
    /// decodes only the **new** tokens back to text. `RUSTY_CHAT_TEMPLATE=raw`
    /// reverts to the pre-template wire for Uk-golden baseline comparisons.
    pub fn complete(&self, messages_json: &str, max_new: usize) -> Result<String> {
        let prompt = chat_prompt(messages_json).prompt;
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
    /// `EN-7b` SSD delta-state cache (None when disabled): repeated prompts restore
    /// their GDN/KV state from a snapshot file and skip the prefill. Sync — safe
    /// under the thread-per-connection staging lane.
    pub state_cache: Option<crate::statecache::StateCache>,
    /// PH-1204: precomputed vision-prefix embeddings (len = d_model each).
    /// Empty = bit-exact text-only path.
    pub vision_prefix: Vec<Vec<f32>>,
    /// PH-1204: insert the vision vectors after this many prompt tokens
    /// (ChatML image placeholder), not necessarily at position 0.
    pub vision_at: Option<usize>,
}

/// Clamp an insertion index into a prompt of `len` tokens.
pub fn vision_split_at(len: usize, at: usize) -> usize {
    at.min(len)
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
        Ok(Self {
            runner,
            bpe,
            state_cache: crate::statecache::StateCache::open_default(),
            vision_prefix: Vec::new(),
            vision_at: None,
        })
    }

    /// PH-1204: attach K precomputed vision vectors. Width must match d_model.
    pub fn set_vision_prefix(&mut self, vs: Vec<Vec<f32>>) -> Result<()> {
        for v in &vs {
            if v.len() != self.runner.d_model || !v.iter().all(|x| x.is_finite()) {
                return Err(EngineError {
                    message: format!(
                        "textgen35: vision prefix vector width/finite check failed (want {})",
                        self.runner.d_model
                    ),
                });
            }
        }
        self.vision_prefix = vs;
        Ok(())
    }

    /// PH-1204: insert image embeddings after `at` prompt tokens (clamped).
    pub fn set_vision_at(&mut self, at: usize) {
        self.vision_at = Some(at);
    }

    /// Shared completion path; an explicit per-request prefix overrides the
    /// server-level static one without mutating the runner (Sync + &self).
    fn complete_impl(
        &self,
        messages_json: &str,
        max_new: usize,
        prefix_override: Option<&[Vec<f32>]>,
    ) -> Result<String> {
        let prompt = chat_prompt(messages_json).prompt;
        if prompt.trim().is_empty() {
            return Err(EngineError {
                message: "textgen35: prompt has no message content".to_string(),
            });
        }
        let ids = self.bpe.encode(&prompt)?;
        if ids.is_empty() {
            return Err(EngineError {
                message: "textgen35: prompt tokens empty".to_string(),
            });
        }
        self.generate_cached(&ids, max_new, prefix_override)
    }

    /// `ActivationAudit` capture (ECO-9.2 phase 9.3a): per-(token, layer) stats
    /// over the prompt's prefill. Returns one [`crate::runner35::AuditEntry`]
    /// per (prompt token, layer). No state persisted; the decoder is fresh.
    pub fn audit(&mut self, prompt: &str) -> Result<Vec<crate::runner35::AuditEntry>> {
        let ids = self.bpe.encode(prompt)?;
        if ids.is_empty() {
            return Err(EngineError {
                message: "textgen35: audit prompt tokenized to empty".to_string(),
            });
        }
        let mut dec = crate::runner35::Q35Decoder::new(&self.runner);
        let (_logits, audit) = dec.prefill_with_audit(&ids);
        Ok(audit)
    }

    /// Shared `EN-7b` path: load-or-prefill the prompt state, then greedy-generate.
    /// Misses at the restore step rebuild a virgin decoder so a corrupt snapshot can
    /// never poison a prefill; stores are best-effort (a busy disk stays a miss).
    fn generate_cached(
        &self,
        ids: &[u32],
        max_new: usize,
        prefix_override: Option<&[Vec<f32>]>,
    ) -> Result<String> {
        let mut dec = Q35Decoder::new(&self.runner);
        let vision = prefix_override.unwrap_or(&self.vision_prefix);
        let logits = if !vision.is_empty() {
            // PH-1204/1205: the text-only SSD cache is keyed by token ids, so a
            // cache hit would silently drop the image prefix. Bypass it;
            // prefill text up to the placeholder, inject K image embeddings,
            // then prefill the rest of the prompt.
            let at = vision_split_at(ids.len(), self.vision_at.unwrap_or(0));
            let _ = dec.prefill(&ids[..at]);
            let logits = dec.prefill_embeddings(vision)?;
            let tail = dec.prefill(&ids[at..]);
            if tail.is_empty() {
                logits
            } else {
                tail
            }
        } else {
            let cache = self.state_cache.as_ref();
            let key = cache.map(|_| crate::statecache::prompt_key(ids));
            let (logits, hit) = match key.and_then(|k| cache.unwrap().load(k)) {
                Some(snap) => match dec.restore_from(&snap) {
                    Ok(()) => (snap.logits, true),
                    Err(_) => {
                        dec = Q35Decoder::new(&self.runner);
                        (dec.prefill(ids), false)
                    }
                },
                None => (dec.prefill(ids), false),
            };
            if !hit {
                if let (Some(c), Some(k)) = (cache, key) {
                    let snap = dec.capture(&logits);
                    c.store(k, &snap).ok();
                }
            }
            logits
        };
        let out = dec.generate_after(ids, logits, max_new);
        let new = &out[ids.len().min(out.len())..];
        Ok(self.bpe.decode(new))
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
        self.generate_cached(&ids, max_new, None)
    }

    /// Encode + echo token ids.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.bpe.encode(text)
    }

    /// Encode, then greedy-generate **keeping** the top-`k` logits of every step — the
    /// `PH-1305` numeric fingerprint (llama.cpp `logprobs` gives the same quantity).
    pub fn trace_text(&self, prompt: &str, max_new: usize, k: usize) -> Result<Vec<TraceStep>> {
        let ids = self.bpe.encode(prompt)?;
        if ids.is_empty() {
            return Err(EngineError {
                message: "textgen35: prompt tokenized to empty".to_string(),
            });
        }
        let mut dec = Q35Decoder::new(&self.runner);
        Ok(dec.generate_trace(&ids, max_new, k))
    }
}

/// Extract a bare `messages[].content` string by scanning a JSON body (no serde dep).
/// Finds each `"content"`, reads the following quoted string (with `\n`/`\t`/`\"`/`\\`
/// unescape), and concatenates the values with a space. Returns "" when none present.
///
/// UTF-8-safe rewrite (PH: Uk-golden gate): the previous byte scanner pushed
/// `jb[v] as char`, decoding every byte as Latin-1 — ANY non-ASCII content
/// (Cyrillic…) reached the model double-encoded. This version walks `chars`,
/// which is exact for the valid-UTF-8 slices a JSON string can legally hold.
pub struct ChatPrompt {
    pub prompt: String,
}

/// Resolve the wire prompt for a chat completion. Priority: ChatML template
/// (default, matches the model's `<|im_start|>` framing) unless the caller
/// explicitly sets `RUSTY_CHAT_TEMPLATE=raw` (keeps the pre-PH-1306 raw-concat
/// path available for audits against the old baseline).
pub fn chat_prompt(json: &str) -> ChatPrompt {
    match std::env::var_os("RUSTY_CHAT_TEMPLATE")
        .map(|v| v.to_string_lossy().into_owned())
        .as_deref()
    {
        Some("raw") => ChatPrompt {
            prompt: extract_message_content(json),
        },
        _ => match chatml::apply_chatml_template(json) {
            Ok(prompt) => ChatPrompt { prompt },
            Err(_) => ChatPrompt {
                prompt: extract_message_content(json),
            },
        },
    }
}

pub fn extract_message_content(json: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut rest: &str = json;
    while let Some(rel) = rest.find("\"content\"") {
        let after_key = rest[rel + "\"content\"".len()..].trim_start();
        let Some(after_colon) = after_key.strip_prefix(':') else {
            break;
        };
        let after_colon = after_colon.trim_start();
        if !after_colon.starts_with('"') {
            break; // non-string content (number/null/object) — stop
        }
        let s = &after_colon[1..];
        let mut val = String::new();
        let mut consumed = 0usize; // bytes of `s` consumed (incl. closing quote)
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            consumed += c.len_utf8();
            match c {
                '"' => break,
                '\\' => match chars.next() {
                    Some(e) => {
                        consumed += e.len_utf8();
                        match e {
                            'n' => val.push('\n'),
                            't' => val.push('\t'),
                            'r' => val.push('\r'),
                            '\\' => val.push('\\'),
                            '"' => val.push('"'),
                            other => val.push(other),
                        }
                    }
                    None => break,
                },
                c => val.push(c),
            }
        }
        if !val.is_empty() {
            parts.push(val);
        }
        rest = &s[consumed..];
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
pub fn chat_response(runner: &dyn Completes, json_body: &str, max_new: usize) -> ChatOutcome {
    chat_response_impl(runner, json_body, max_new, None, None)
}

/// PH-1205: same HTTP chat shape, but with a per-request vision prefix
/// (raw LE f32 bytes, K x d_model) injected at the placeholder position.
pub fn chat_response_with_vision(
    runner: &dyn Completes,
    json_body: &str,
    max_new: usize,
    prefix_bytes: &[u8],
) -> ChatOutcome {
    chat_response_impl(runner, json_body, max_new, Some(prefix_bytes), None)
}

/// PH-1205: a chat request carrying decoded `messages[].images` bytes.
/// Takes precedence over `vision_prefix_b64` when both are present.
pub fn chat_response_with_images(
    runner: &dyn Completes,
    json_body: &str,
    max_new: usize,
    images: &[Vec<u8>],
) -> ChatOutcome {
    chat_response_impl(runner, json_body, max_new, None, Some(images))
}

fn chat_response_impl(
    runner: &dyn Completes,
    json_body: &str,
    max_new: usize,
    prefix_bytes: Option<&[u8]>,
    images: Option<&[Vec<u8>]>,
) -> ChatOutcome {
    let started = std::time::Instant::now();
    let completed = match (images, prefix_bytes) {
        (Some(imgs), _) => runner.complete_with_images(json_body, max_new, imgs),
        (None, Some(bytes)) => runner.complete_with_vision(json_body, max_new, bytes),
        (None, None) => runner.complete(json_body, max_new),
    };
    match completed {
        Ok(text) => {
            let decode_secs = started.elapsed().as_secs_f64();
            let escaped = escape_json_text(&text);
            let prompt_len = runner.token_len(&extract_message_content(json_body));
            let generated = runner.token_len(&text).max(1);
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
pub fn chat_stream_response(
    runner: &dyn Completes,
    json_body: &str,
    max_new: usize,
) -> ChatOutcome {
    let started = std::time::Instant::now();
    match runner.complete(json_body, max_new) {
        Ok(text) => {
            let decode_secs = started.elapsed().as_secs_f64();
            let escaped = escape_json_text(&text);
            let generated = runner.token_len(&text).max(1);
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
    fn vision_at_clamps_to_prompt_length() {
        assert_eq!(vision_split_at(10, 3), 3);
        assert_eq!(vision_split_at(10, 99), 10);
        assert_eq!(vision_split_at(0, 5), 0);
    }

    #[test]
    fn vision_prefix_file_chunks_and_validates() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vision_prefix_test.f32");
        let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).unwrap();
        let vs = load_vision_prefix(path.to_str().unwrap(), 3).unwrap();
        assert_eq!(vs.len(), 2);
        assert_eq!(vs[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(vs[1], vec![4.0, 5.0, 6.0]);
        assert!(load_vision_prefix(path.to_str().unwrap(), 4).is_err());
        let _ = std::fs::remove_file(&path);
    }

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
    fn extract_preserves_utf8_cyrillic() {
        let j = r#"{"messages":[{"role":"user","content":"Яка столиця України?"}]}"#;
        assert_eq!(extract_message_content(j), "Яка столиця України?");
    }

    #[test]
    fn extract_preserves_utf8_mixed_escapes() {
        let j = r#"{"messages":[{"role":"user","content":"Привіт\nяк справи"}]}"#;
        assert_eq!(extract_message_content(j), "Привіт\nяк справи");
    }

    #[test]
    fn extract_utf8_concatenates_turns() {
        let j = r#"{"messages":[{"role":"system","content":"Ти україномовний"},{"role":"user","content":"Хто написав Кобзар?"}]}"#;
        assert_eq!(
            extract_message_content(j),
            "Ти україномовний Хто написав Кобзар?"
        );
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
