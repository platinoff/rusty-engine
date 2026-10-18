//! B5 `PH-506` — end-to-end wire contract for the std-only serve loop.
//!
//! Drives `serve::serve_one` over real loopback sockets (stub completions, no model) and
//! asserts the exact HTTP surface: status lines, `Content-Type`/`Content-Length`, the
//! OpenAI-shaped JSON bodies (parsed strictly with `worker::parse_json`, so an incomplete
//! body like the PH-507 missing-root-`}` cannot slip through), SSE framing, and the
//! 400/404/405 error paths.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

/// Bind an ephemeral loopback port, serve exactly one connection with the stub path
/// (no model), and return `(addr, join handle)`.
fn serve_once_stub() -> (SocketAddr, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let addr = listener.local_addr().expect("local_addr");
    let th = std::thread::spawn(move || {
        let mut m = rusty_engine::hub::Metrics::zero();
        rusty_engine::serve::serve_one(&listener, &mut m, None).expect("serve_one (stub)");
    });
    (addr, th)
}

/// Send a raw request and read the whole response (the server closes the connection).
fn request(addr: SocketAddr, raw: &str) -> String {
    let mut client = TcpStream::connect(addr).expect("connect");
    client.write_all(raw.as_bytes()).expect("write");
    client.flush().expect("flush");
    let mut resp = String::new();
    client.read_to_string(&mut resp).expect("read");
    resp
}

/// One canned request/response round-trip against a fresh stub server.
fn round_trip(raw: &str) -> String {
    let (addr, th) = serve_once_stub();
    let resp = request(addr, raw);
    th.join().expect("server thread");
    resp
}

fn body_of(resp: &str) -> &str {
    resp.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("")
}

fn header(resp: &str, name: &str) -> Option<String> {
    resp.lines()
        .skip(1) // status line
        .take_while(|l| !l.is_empty())
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.trim().to_string())
}

fn get(path: &str) -> String {
    round_trip(&format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"))
}

fn post(path: &str, body: &str) -> String {
    round_trip(&format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    ))
}

#[test]
fn health_is_ok_json_over_the_wire() {
    let resp = get("/health");
    assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"), "{resp}");
    assert_eq!(
        header(&resp, "content-type").as_deref(),
        Some("application/json")
    );
    let body = body_of(&resp);
    assert_eq!(
        header(&resp, "content-length").as_deref(),
        Some(body.len().to_string().as_str())
    );
    let json = rusty_engine::worker::parse_json(body).expect("valid JSON");
    assert!(
        !json.get("ok").expect("ok field").is_null(),
        "ok should be a real value"
    );
    assert_eq!(
        json.get("name").and_then(|v| v.as_str()),
        Some(rusty_engine::CRATE_NAME)
    );
    assert_eq!(
        json.get("version").and_then(|v| v.as_str()),
        Some(rusty_engine::VERSION)
    );
}

#[test]
fn models_lists_the_engine_over_the_wire() {
    let resp = get("/v1/models");
    assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"), "{resp}");
    let json = rusty_engine::worker::parse_json(body_of(&resp)).expect("valid JSON");
    assert_eq!(json.get("object").and_then(|v| v.as_str()), Some("list"));
    let data = json
        .get("data")
        .and_then(|d| d.as_array())
        .expect("data array");
    assert!(!data.is_empty(), "at least one model row");
    assert_eq!(
        data[0].get("id").and_then(|v| v.as_str()),
        Some("rusty-engine")
    );
    assert_eq!(
        data[0].get("owned_by").and_then(|v| v.as_str()),
        Some("platinoff")
    );
}

#[test]
fn stub_chat_completion_is_valid_json_over_the_wire() {
    let resp = post("/v1/chat/completions", "{}");
    assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"), "{resp}");
    let json = rusty_engine::worker::parse_json(body_of(&resp)).expect("valid JSON");
    assert_eq!(
        json.get("object").and_then(|v| v.as_str()),
        Some("chat.completion")
    );
    assert_eq!(
        json.get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str()),
        Some("stop")
    );
    assert_eq!(
        json.get("usage")
            .and_then(|u| u.get("total_tokens"))
            .and_then(|v| v.as_u64()),
        Some(0)
    );
}

#[test]
fn stub_stream_is_event_framed_over_the_wire() {
    let resp = post("/v1/chat/completions", r#"{"stream":true}"#);
    assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"), "{resp}");
    assert_eq!(
        header(&resp, "content-type").as_deref(),
        Some("text/event-stream")
    );
    assert!(resp.ends_with("data: [DONE]\n\n"), "{resp}");
    // every non-DONE `data:` line is a complete JSON chunk
    let mut chunks = 0;
    for line in resp.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            continue;
        }
        rusty_engine::worker::parse_json(payload)
            .unwrap_or_else(|e| panic!("SSE chunk not valid JSON: {e}\n{payload}"));
        chunks += 1;
    }
    assert!(chunks >= 1, "expected the finish chunk, got {chunks}");
}

#[test]
fn error_paths_400_404_405_over_the_wire() {
    assert!(get("/nope").starts_with("HTTP/1.1 404 "));
    assert!(post("/nope", "{}").starts_with("HTTP/1.1 404 "));
    assert!(get("/v1/chat/completions").starts_with("HTTP/1.1 405 "));
    let del = round_trip("DELETE /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
    assert!(del.starts_with("HTTP/1.1 405 "), "{del}");
    let garbage = round_trip("garbage\r\n\r\n");
    assert!(garbage.starts_with("HTTP/1.1 400 "), "{garbage}");
}
