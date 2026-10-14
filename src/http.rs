//! Minimal **std-only** OpenAI-shaped HTTP layer — band B5 `PH-501a`.
//!
//! Deliberately dependency-free (`std`, no `axum`/`tokio`): the request→response mapping is
//! a **pure function** [`route`] so it is unit-testable without any socket, and the runtime
//! loop in [`crate::serve`] is a thin `std::net::TcpListener` pump over it. Loopback-only by
//! default (dry-run posture, matching the hub LAN-first rule). Full streaming/multi-slot and
//! real-model wiring land later in B5; this slice locks the wire contract.

/// A self-contained HTTP/1.1 response (fixed `Content-Length`, `Connection: close`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub reason: &'static str,
    pub content_type: &'static str,
    pub body: String,
}

impl Response {
    pub fn json(status: u16, body: String) -> Self {
        Self {
            status,
            reason: reason_for(status),
            content_type: "application/json",
            body,
        }
    }

    /// Serialize to the exact bytes written on the socket.
    pub fn to_http_bytes(&self) -> Vec<u8> {
        format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.status,
            self.reason,
            self.content_type,
            self.body.len(),
            self.body,
        )
        .into_bytes()
    }
}

fn reason_for(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Error",
    }
}

/// `(method, path)` split from the request line; `None` if it is malformed.
pub fn parse_request_line(first_line: &str) -> Option<(&str, &str)> {
    let mut parts = first_line.split_whitespace();
    let method = parts.next()?;
    let path = parts.next()?;
    Some((method, path))
}

/// Route a raw HTTP request (only the request line + method are needed) to a [`Response`].
/// Pure and side-effect-free — the whole wire contract lives here.
pub fn route(request: &[u8]) -> Response {
    let text = match std::str::from_utf8(request) {
        Ok(t) => t,
        Err(_) => return Response::json(400, r#"{"error":"invalid utf-8"}"#.to_string()),
    };
    let first_line = text.lines().next().unwrap_or("");
    let (method, path) = match parse_request_line(first_line) {
        Some(mp) => mp,
        None => return Response::json(400, r#"{"error":"malformed request line"}"#.to_string()),
    };

    match (method, path) {
        ("GET", "/health") => Response::json(
            200,
            format!(
                r#"{{"ok":true,"name":"{}","version":"{}"}}"#,
                crate::CRATE_NAME,
                crate::VERSION
            ),
        ),
        ("GET", crate::serve::PATH_MODELS) => Response::json(
            200,
            r#"{"object":"list","data":[{"id":"rusty-engine","object":"model","owned_by":"platinoff"}]}"#
                .to_string(),
        ),
        ("GET", crate::serve::PATH_CHAT_COMPLETIONS) => {
            Response::json(405, r#"{"error":"method not allowed, use POST"}"#.to_string())
        }
        ("POST", crate::serve::PATH_CHAT_COMPLETIONS) => Response::json(
            200,
            r#"{"id":"cmpl-0","object":"chat.completion","model":"rusty-engine","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}"#
                .to_string(),
        ),
        ("GET", _) | ("POST", _) => {
            Response::json(404, format!(r#"{{"error":"no route {path}"}}"#))
        }
        (m, _) => Response::json(405, format!(r#"{{"error":"unsupported method {m}"}}"#)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(line: &str) -> Vec<u8> {
        format!("{line}\r\nHost: 127.0.0.1\r\n\r\n").into_bytes()
    }

    #[test]
    fn health_reports_name_and_version() {
        let r = route(&req("GET /health HTTP/1.1"));
        assert_eq!(r.status, 200);
        assert!(r
            .body
            .contains(&format!("\"version\":\"{}\"", crate::VERSION)));
        assert!(r.body.contains("\"ok\":true"));
    }

    #[test]
    fn models_lists_the_engine() {
        let r = route(&req("GET /v1/models HTTP/1.1"));
        assert_eq!(r.status, 200);
        assert!(r.body.contains("rusty-engine"));
        assert_eq!(r.content_type, "application/json");
    }

    #[test]
    fn chat_completion_post_ok_get_not_allowed() {
        let post = route(&req("POST /v1/chat/completions HTTP/1.1"));
        assert_eq!(post.status, 200);
        assert!(post.body.contains("chat.completion"));
        assert!(post.body.contains("\"finish_reason\":\"stop\""));
        let get = route(&req("GET /v1/chat/completions HTTP/1.1"));
        assert_eq!(get.status, 405);
    }

    #[test]
    fn unknown_path_is_404_and_bad_line_400() {
        assert_eq!(route(&req("GET /nope HTTP/1.1")).status, 404);
        assert_eq!(route(b"garbage").status, 400);
    }

    #[test]
    fn unsupported_method_405() {
        assert_eq!(route(&req("DELETE /health HTTP/1.1")).status, 405);
    }

    #[test]
    fn response_serializes_with_correct_length() {
        let r = Response::json(200, "{\"ok\":true}".to_string());
        let bytes = r.to_http_bytes();
        let s = String::from_utf8(bytes).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(s.contains("Content-Length: 11\r\n"));
        assert!(s.ends_with("{\"ok\":true}"));
    }

    #[test]
    fn parse_request_line_splits_method_and_path() {
        assert_eq!(
            parse_request_line("POST /v1/chat/completions HTTP/1.1"),
            Some(("POST", "/v1/chat/completions"))
        );
        assert!(parse_request_line("onlyone").is_none());
    }
}
