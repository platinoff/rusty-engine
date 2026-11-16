//! ChatML wrapper for the qwen35 / MiniCPM-V chat wire (B13 / PH-1306 follow-up).
//!
//! The OpenAI-shaped `POST /v1/chat/completions` used to hand the runner a bare
//! `messages[].content` concat, which produced incoherent Uk/Cyrillic outputs
//! (the model expects `<|im_start|>role\n…<|im_end|>` framing). `apply` builds the
//! MiniCPM-V 4.7 / Qwen-ChatML prompt server-side, preserving roles and order.
//!
//! Kept std-only and hand-rolled (no serde dep) to match the rest of the crate.

use crate::core::EngineError;

/// Pair of (role, content) extracted from the `messages[]` array in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Scan `"role"` → `"content"` pairs in order. Tolerates escaped quotes and
/// non-ASCII (walks chars, matching the PH: Uk-golden rewrite style).
/// Returns an error when a content value is not a JSON string.
pub fn extract_messages(json: &str) -> Result<Vec<Message>, EngineError> {
    let mut out: Vec<Message> = Vec::new();
    let mut rest = json;
    while let Some(rel_role) = rest.find("\"role\"") {
        let after_role_key = rest[rel_role + "\"role\"".len()..].trim_start();
        let Some(after_role_colon) = after_role_key.strip_prefix(':') else {
            break;
        };
        let after_role_colon = after_role_colon.trim_start();
        if !after_role_colon.starts_with('"') {
            break;
        }
        let role_str = &after_role_colon[1..];
        let mut chars = role_str.chars();
        let mut role = String::new();
        let mut role_consumed: usize = 0;
        while let Some(c) = chars.next() {
            role_consumed += c.len_utf8();
            match c {
                '"' => break,
                '\\' => match chars.next() {
                    Some(e) => {
                        role_consumed += e.len_utf8();
                        match e {
                            'n' => role.push('\n'),
                            't' => role.push('\t'),
                            'r' => role.push('\r'),
                            '\\' => role.push('\\'),
                            '"' => role.push('"'),
                            other => role.push(other),
                        }
                    }
                    None => break,
                },
                c => role.push(c),
            }
        }

        // Find the next `"content"` *after* the role string closes.
        let after_role = &role_str[role_consumed..];
        let rel_content = match after_role.find("\"content\"") {
            Some(i) => i,
            None => break,
        };
        let after_content_key = after_role[rel_content + "\"content\"".len()..].trim_start();
        let Some(after_content_colon) = after_content_key.strip_prefix(':') else {
            break;
        };
        let after_content_colon = after_content_colon.trim_start();
        if !after_content_colon.starts_with('"') {
            return Err(EngineError {
                message: "chatml: message content is not a JSON string".to_string(),
            });
        }
        let content_str = &after_content_colon[1..];
        let mut chars = content_str.chars();
        let mut content = String::new();
        let mut content_consumed: usize = 0;
        while let Some(c) = chars.next() {
            content_consumed += c.len_utf8();
            match c {
                '"' => break,
                '\\' => match chars.next() {
                    Some(e) => {
                        content_consumed += e.len_utf8();
                        match e {
                            'n' => content.push('\n'),
                            't' => content.push('\t'),
                            'r' => content.push('\r'),
                            '\\' => content.push('\\'),
                            '"' => content.push('"'),
                            other => content.push(other),
                        }
                    }
                    None => break,
                },
                c => content.push(c),
            }
        }

        if !role.is_empty() && !content.is_empty() {
            out.push(Message { role, content });
        }
        rest = &content_str[content_consumed..];
    }

    Ok(out)
}

/// Render `messages[]` as MiniCPM-V 4.7 / Qwen-ChatML:
///   `<|im_start|>{role}\n{content}<|im_end|>\n` × N `<|im_start|>assistant\n`
///
/// No system-instruction fallback: callers that wire a `system` role will see it
/// echoed faithfully; the model is an uncluttered base and we don't inject one.
pub fn apply_chatml_template(messages_json: &str) -> Result<String, EngineError> {
    let messages = extract_messages(messages_json)?;
    if messages.is_empty() {
        return Err(EngineError {
            message: "chatml: empty messages array".to_string(),
        });
    }
    let mut out = String::new();
    for m in &messages {
        out.push_str("<|im_start|>");
        out.push_str(&m.role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_user_roundtrips() {
        let p = apply_chatml_template(r#"{"messages":[{"role":"user","content":"Привіт"}]}"#)
            .expect("render");
        assert_eq!(
            p,
            "<|im_start|>user\nПривіт<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn order_preserved_with_system() {
        let p = apply_chatml_template(
            r#"{"messages":[{"role":"system","content":"Ти бот."},{"role":"user","content":"Привіт"}]}"#,
        )
        .expect("render");
        assert_eq!(
            p,
            "<|im_start|>system\nТи бот.<|im_end|>\n<|im_start|>user\nПривіт<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn escape_sequences_in_content() {
        let p =
            apply_chatml_template(r#"{"messages":[{"role":"user","content":"рядок\nрядково"}]}"#)
                .expect("render");
        assert_eq!(
            p,
            "<|im_start|>user\nрядок\nрядково<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn multi_turn_preserved() {
        let p = apply_chatml_template(
            r#"{"messages":[
                {"role":"user","content":"q1"},
                {"role":"assistant","content":"a1"},
                {"role":"user","content":"q2"}
            ]}"#,
        )
        .expect("render");
        assert_eq!(
            p,
            "<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\na1<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn empty_messages_errors() {
        let err = apply_chatml_template(r#"{"messages":[]}"#).unwrap_err();
        assert!(err.message.contains("empty messages"));
    }

    #[test]
    fn non_string_content_errors() {
        let err =
            apply_chatml_template(r#"{"messages":[{"role":"user","content":123}]}"#).unwrap_err();
        assert!(err.message.contains("not a JSON string"));
    }
}
