//! OpenAI-compatible HTTP surface (PH-501, band B5). B0 pins the route constants
//! so the server, client and UI agree on paths; `PH-501a` adds a **std-only** request→
//! response layer (`crate::http::route`) and this thin `std::net` pump over it. PH-507
//! moves the bind to LAN-first (band 233): `rusty serve` defaults to `0.0.0.0`,
//! `--host` overrides. Model chat streams SSE on `stream:true` (PH-507, bunke-rock
//! contract); full streaming/multi-slot wiring stays in B5.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

pub const PATH_MODELS: &str = "/v1/models";
pub const PATH_CHAT_COMPLETIONS: &str = "/v1/chat/completions";
/// Cap on bytes read per request (header + body). Prompts can be large, so this is
/// generous; the loop still streams only what `Content-Length` declares.
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
/// Initial buffer; the loop grows it on demand up to [`MAX_REQUEST_BYTES`].
const READ_CHUNK: usize = 16 * 1024;
/// Generated-token cap when the request carries no usable `max_tokens`.
const DEFAULT_MAX_NEW: usize = 8;
/// Hard ceiling on `max_tokens` accepted from a client (guards a runaway request).
const MAX_NEW_CAP: usize = 512;

/// Accept, serve and close exactly one connection — the unit `serve` loops over. `metrics`
/// accumulates across the process lifetime (PH-505 counters); `model` wires real generation.
pub fn serve_one(
    listener: &TcpListener,
    metrics: &mut crate::hub::Metrics,
    model: Option<&crate::textgen::TextRunner>,
) -> std::io::Result<()> {
    let (mut stream, _peer) = listener.accept()?;
    handle_conn(&mut stream, metrics, model)
}

/// Read one request from `stream`, route it against `metrics` (+ optional real `model`),
/// write the response. With a model, POST /v1/chat/completions runs BPE→generate→decode
/// (PH-501b); otherwise the pure `route_with` stub answers.
pub fn handle_conn(
    stream: &mut TcpStream,
    metrics: &mut crate::hub::Metrics,
    model: Option<&crate::textgen::TextRunner>,
) -> std::io::Result<()> {
    let mut buf = vec![0u8; READ_CHUNK];
    let mut filled = 0usize;
    let mut header_end: Option<usize> = None;
    // Read the head, then keep reading until the whole `Content-Length` body has
    // arrived (hyper/reqwest split head and body across segments — PH-507), or the cap.
    while filled < MAX_REQUEST_BYTES {
        if filled == buf.len() {
            buf.resize((buf.len() * 2).min(MAX_REQUEST_BYTES), 0);
        }
        let n = stream.read(&mut buf[filled..])?;
        if n == 0 {
            break;
        }
        filled += n;
        if header_end.is_none() {
            if let Some(i) = buf[..filled].windows(4).position(|w| w == b"\r\n\r\n") {
                header_end = Some(i + 4);
            }
        }
        if let Some(he) = header_end {
            let head = String::from_utf8_lossy(&buf[..he]);
            let want = he.saturating_add(crate::http::content_length(&head));
            if filled >= want.min(MAX_REQUEST_BYTES) {
                break;
            }
        }
    }
    let text = String::from_utf8_lossy(&buf[..filled]).into_owned();
    let head = text.lines().next().unwrap_or("");
    let is_chat = head.starts_with("POST ") && head.contains(PATH_CHAT_COMPLETIONS);
    let resp = match model {
        Some(tr) if is_chat => {
            // PH-501c: honour the client's max_tokens (clamped) instead of a fixed cap —
            // a 1-token request must not pay for 8 decode steps.
            let body = crate::http::body_of(&text);
            let max_new = crate::http::max_tokens_of(body)
                .unwrap_or(DEFAULT_MAX_NEW)
                .clamp(1, MAX_NEW_CAP);
            let r = if crate::http::wants_stream(&text) {
                crate::textgen::chat_stream_response(tr, body, max_new)
            } else {
                crate::textgen::chat_response(tr, body, max_new)
            };
            // PH-505: record the real decode stats (tokens + wall clock + MoE experts);
            // the runner is the dense path, so `active_experts` is 0 until PH-406.
            metrics.record(r.generated_tokens, r.decode_secs, r.active_experts);
            r.response
        }
        _ => {
            let r = crate::http::route_with(&buf[..filled], metrics);
            if is_chat && r.status == 200 {
                metrics.record(0, 0.0, 0);
            }
            r
        }
    };
    stream.write_all(&resp.to_http_bytes())?;
    stream.flush()?;
    Ok(())
}

/// Bind `addr` (typically `127.0.0.1:PORT`) and serve forever (stub completions). Blocks.
pub fn serve(addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    accept_loop(&listener, None)
}

/// PH-501b: serve with a real [`crate::textgen::TextRunner`] bound — POST /v1/chat/completions
/// runs BPE→generate→decode over the loaded GGUF instead of the empty stub.
pub fn serve_with_model(addr: &str, model: &crate::textgen::TextRunner) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    accept_loop(&listener, Some(model))
}

/// Shared accept loop: keep-live stamping (PH-504) + per-connection routing (PH-505 counters,
/// PH-501b model). Fail-open per connection so a broken client never kills the loop.
fn accept_loop(
    listener: &TcpListener,
    model: Option<&crate::textgen::TextRunner>,
) -> std::io::Result<()> {
    let live_dir = std::path::Path::new("target/live");
    let start = std::time::Instant::now();
    let mut metrics = crate::hub::Metrics::zero();
    let _ = crate::hub::write_heartbeat(
        live_dir,
        &crate::hub::Heartbeat::new(start.elapsed().as_secs()),
    );
    // PH-504: keep-live must re-stamp on a fixed tick regardless of traffic — the hub's
    // freshness window is 60s, so an idle provider that only wrote at startup +
    // per-connection fell out of host-readiness. A dedicated thread stamps every tick.
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(
            crate::hub::HEARTBEAT_TICK_SECS,
        ));
        let _ = crate::hub::write_heartbeat(
            live_dir,
            &crate::hub::Heartbeat::new(start.elapsed().as_secs()),
        );
    });
    for stream in listener.incoming() {
        match stream {
            Ok(mut s) => {
                let _ = handle_conn(&mut s, &mut metrics, model);
            }
            Err(_) => continue,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    #[test]
    fn serve_one_answers_health_over_the_wire() {
        // Bind an ephemeral loopback port, serve exactly one connection on a thread.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        let addr = listener.local_addr().expect("local_addr");
        let th = std::thread::spawn(move || {
            let mut m = crate::hub::Metrics::zero();
            serve_one(&listener, &mut m, None).expect("serve_one");
        });
        let mut client = TcpStream::connect(addr).expect("connect");
        client
            .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            .expect("write");
        client.flush().expect("flush");
        let mut resp = String::new();
        client.read_to_string(&mut resp).expect("read");
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(resp.contains("\"ok\":true"));
        th.join().expect("server thread");
    }

    #[test]
    fn serve_one_streams_sse_when_requested() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        let addr = listener.local_addr().expect("local_addr");
        let th = std::thread::spawn(move || {
            let mut m = crate::hub::Metrics::zero();
            serve_one(&listener, &mut m, None).expect("serve_one");
        });
        let mut client = TcpStream::connect(addr).expect("connect");
        client
            .write_all(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n{\"stream\":true}",
            )
            .expect("write");
        client.flush().expect("flush");
        let mut resp = String::new();
        client.read_to_string(&mut resp).expect("read");
        assert!(resp.contains("Content-Type: text/event-stream\r\n"));
        assert!(resp.contains("chat.completion.chunk"));
        assert!(resp.ends_with("data: [DONE]\n\n"), "{resp}");
        th.join().expect("server thread");
    }

    #[test]
    fn routes_are_openai_shaped() {
        assert_eq!(PATH_MODELS, "/v1/models");
        assert_eq!(PATH_CHAT_COMPLETIONS, "/v1/chat/completions");
    }

    #[test]
    fn reads_the_body_when_head_and_body_arrive_separately() {
        // hyper/reqwest write the head, then the body — a reader that stops at the
        // `\r\n\r\n` terminator would see an empty body (the PH-507 hub `null` bug).
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        let addr = listener.local_addr().expect("local_addr");
        let th = std::thread::spawn(move || {
            let mut m = crate::hub::Metrics::zero();
            serve_one(&listener, &mut m, None).expect("serve_one");
        });
        let body = b"{\"model\":\"rusty-05\",\"stream\":true}";
        let head = format!(
            "POST {} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            PATH_CHAT_COMPLETIONS,
            body.len()
        );
        let mut client = TcpStream::connect(addr).expect("connect");
        client.write_all(head.as_bytes()).expect("write head");
        client.flush().expect("flush head");
        std::thread::sleep(std::time::Duration::from_millis(50));
        client.write_all(body).expect("write body");
        client.flush().expect("flush body");
        let mut resp = String::new();
        client.read_to_string(&mut resp).expect("read");
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"), "{resp}");
        // `stream:true` is in the body only — SSE proves the body was fully read.
        assert!(
            resp.contains("Content-Type: text/event-stream\r\n"),
            "{resp}"
        );
        th.join().expect("server thread");
    }
}
