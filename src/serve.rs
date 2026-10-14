//! OpenAI-compatible HTTP surface (PH-501, band B5). B0 pins the route constants
//! so the server, client and UI agree on paths; `PH-501a` adds a **std-only** request→
//! response layer (`crate::http::route`) and this thin `std::net` pump over it. Loopback
//! by default (dry-run posture); streaming/multi-slot and real-model wiring land later in B5.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

pub const PATH_MODELS: &str = "/v1/models";
pub const PATH_CHAT_COMPLETIONS: &str = "/v1/chat/completions";
/// Cap on bytes read per request (headers only in the `PH-501a` slice).
const MAX_REQUEST_BYTES: usize = 8 * 1024;

/// Accept, serve and close exactly one connection — the unit `serve` loops over.
pub fn serve_one(listener: &TcpListener) -> std::io::Result<()> {
    let (mut stream, _peer) = listener.accept()?;
    handle_conn(&mut stream)
}

/// Read one request from `stream`, route it (pure, `crate::http`), write the response.
pub fn handle_conn(stream: &mut TcpStream) -> std::io::Result<()> {
    let mut buf = vec![0u8; MAX_REQUEST_BYTES];
    let mut filled = 0usize;
    // Read until the header terminator or the cap, so the request line is always present.
    while filled < MAX_REQUEST_BYTES {
        let n = stream.read(&mut buf[filled..])?;
        if n == 0 {
            break;
        }
        filled += n;
        if buf[..filled].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    let resp = crate::http::route(&buf[..filled]);
    stream.write_all(&resp.to_http_bytes())?;
    stream.flush()
}

/// Bind `addr` (typically `127.0.0.1:PORT`) and serve forever. Blocks the calling thread.
/// Keeps the GSV hub alive mirror (`PH-504`) by stamping `rusty_heartbeat.json` at startup
/// and after every handled connection (a busy loopback tool stays fresh; the hub's
/// `HEARTBEAT_FRESH_SECS=60s` window tolerates idle gaps between requests).
pub fn serve(addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    let live_dir = std::path::Path::new("target/live");
    let start = std::time::Instant::now();
    let _ = crate::hub::write_heartbeat(
        live_dir,
        &crate::hub::Heartbeat::new(start.elapsed().as_secs()),
    );
    for stream in listener.incoming() {
        match stream {
            Ok(mut s) => {
                if handle_conn(&mut s).is_err() {
                    // Fail-open: a broken client must not kill the loop (watchdog-friendly).
                    continue;
                }
                let _ = crate::hub::write_heartbeat(
                    live_dir,
                    &crate::hub::Heartbeat::new(start.elapsed().as_secs()),
                );
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
            serve_one(&listener).expect("serve_one");
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
            serve_one(&listener).expect("serve_one");
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
}
