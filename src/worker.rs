//! Worker / edge runtime (band B6/B7 `PH-601..604` + worker/APK role, 2026-10-16).
//!
//! Telenetis became a pure **commutation** plugin; the worker + APK role moved to the
//! model engines. This module is `rusty`'s half of that role: a std-only worker that
//! registers with the GSV hub single-entry edge (`/api/edge`, which fronts poolAI
//! `/api/v1`), heartbeats, polls `virtual-nodes/{peer}/tasks/poll`, executes the task
//! types the MoE engine owns (`ping`, `rusty_chat`) and reports complete/requeue.
//!
//! Posture: **dry-run by default** — [`plan_lines`] prints exactly what live mode would
//! send (never the edge token); `rusty worker --live` opens sockets. The pure core
//! ([`WorkerConfig`] builders, [`parse_task`], [`execute`]) is unit-tested without a
//! network, mirroring the `llama-rs` `llama_edge` protocol but routed through the hub
//! (never poolAI `:8091` off-box) and dependency-free (`std`, hand-built JSON).

use std::io::{Read, Write};
use std::time::Duration;

/// Edge-plane origin advertised by a `rusty` engine worker.
pub const ORIGIN: &str = "rusty_edge";
/// poolAI virtual-node role required by `pool/join`.
pub const ROLE: &str = "virtual_node";

/// Capability tags advertised in register/heartbeat.
pub fn capabilities() -> Vec<&'static str> {
    vec!["inference:cpu", "rusty_moe"]
}

/// Everything a worker needs to talk to the hub edge; `token` is never logged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerConfig {
    pub peer_id: String,
    /// Hub edge base, e.g. `http://192.168.2.238:9999/api/edge` (never `:8091`).
    pub hub: String,
    /// GSV edge token (header `x-gsv-edge-token`); never echoed.
    pub token: String,
    /// Local `rusty serve` base, e.g. `http://192.168.2.238:8090`.
    pub serve_url: String,
    /// Model id advertised to the hub.
    pub model: String,
    pub max_memory_mb: usize,
    pub cpu_cores: usize,
}

impl WorkerConfig {
    /// A config with LAN-first defaults (env overrides applied by the CLI).
    pub fn local(peer_id: impl Into<String>, token: impl Into<String>) -> Self {
        let addr = local_addr();
        Self {
            peer_id: peer_id.into(),
            hub: format!("http://{addr}:9999/api/edge"),
            token: token.into(),
            serve_url: format!("http://{addr}:8090"),
            model: "rusty-05".to_string(),
            max_memory_mb: 0,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }
}

/// This machine's LAN address (`GSV_LOCAL_ADDR` wins), else a UDP-route probe, else loopback.
pub fn local_addr() -> String {
    if let Ok(v) = std::env::var("GSV_LOCAL_ADDR") {
        let v = v.trim();
        if !v.is_empty() {
            return v.to_string();
        }
    }
    if let Ok(sock) = std::net::UdpSocket::bind("0.0.0.0:0") {
        if sock.connect("8.8.8.8:80").is_ok() {
            if let Ok(a) = sock.local_addr() {
                return a.ip().to_string();
            }
        }
    }
    "127.0.0.1".to_string()
}

// ---------------------------------------------------------------------------
// Edge-plane paths (relative to the hub `/api/edge`; forwarded to poolAI `/api/v1`)
// ---------------------------------------------------------------------------

pub fn register_path() -> &'static str {
    "discovery/register-remote"
}
pub fn heartbeat_path() -> &'static str {
    "discovery/heartbeat-remote"
}
pub fn join_path(peer: &str) -> String {
    format!("virtual-nodes/{peer}/pool/join")
}
pub fn poll_path(peer: &str) -> String {
    format!("virtual-nodes/{peer}/tasks/poll")
}
pub fn complete_path(peer: &str, task_id: &str) -> String {
    format!("virtual-nodes/{peer}/tasks/{task_id}/complete")
}
pub fn requeue_path(peer: &str) -> String {
    format!("virtual-nodes/{peer}/tasks")
}
pub fn chat_path() -> &'static str {
    "/v1/chat/completions"
}

// ---------------------------------------------------------------------------
// Minimal std-only JSON (parse + emit) — enough for the poll/complete contract.
// ---------------------------------------------------------------------------

/// A tiny JSON value. Objects keep insertion order (`Vec`) so re-emitted payloads are stable.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Object member by key.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) if *n >= 0.0 => Some(*n as u64),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    /// Compact JSON serialization.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write_to(&mut out);
        out
    }

    fn write_to(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.is_finite() && n.abs() < 1e15 {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{n}"));
                }
            }
            Json::Str(s) => {
                out.push('"');
                escape_into(s, out);
                out.push('"');
            }
            Json::Arr(items) => {
                out.push('[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    it.write_to(out);
                }
                out.push(']');
            }
            Json::Obj(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push('"');
                    escape_into(k, out);
                    out.push_str("\":");
                    v.write_to(out);
                }
                out.push('}');
            }
        }
    }
}

/// Percent-style JSON string escaping (quotes, backslash, control chars).
pub fn escape_into(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
}

/// Escape a string for embedding (without surrounding quotes).
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    escape_into(s, &mut out);
    out
}

/// Parse a JSON document (strict: trailing non-whitespace is an error).
pub fn parse_json(input: &str) -> Result<Json, String> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(format!("trailing bytes at {}", p.pos));
    }
    Ok(v)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at {}", c as char, self.pos))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(_) => self.number(),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn literal(&mut self, word: &str, val: Json) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(val)
        } else {
            Err(format!("bad literal at {}", self.pos))
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut members = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(members));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            let val = self.value()?;
            members.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(members));
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or("unterminated string")?;
            self.pos += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.peek().ok_or("unterminated escape")?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'u' => {
                            let hex = self
                                .bytes
                                .get(self.pos..self.pos + 4)
                                .ok_or("short \\u escape")?;
                            let s = std::str::from_utf8(hex).map_err(|e| e.to_string())?;
                            let cp = u32::from_str_radix(s, 16).map_err(|e| e.to_string())?;
                            self.pos += 4;
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(format!("bad escape at {}", self.pos)),
                    }
                }
                _ => {
                    // Collect a UTF-8 scalar from the byte stream.
                    let start = self.pos - 1;
                    let len = utf8_len(c);
                    self.pos = start + len;
                    let chunk = self.bytes.get(start..self.pos).ok_or("truncated utf-8")?;
                    out.push_str(std::str::from_utf8(chunk).map_err(|e| e.to_string())?);
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        while matches!(
            self.peek(),
            Some(b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        ) {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| e.to_string())?;
        s.parse::<f64>().map(Json::Num).map_err(|e| e.to_string())
    }
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

// ---------------------------------------------------------------------------
// Task model + request bodies
// ---------------------------------------------------------------------------

/// A popped `virtual-nodes/{peer}/tasks` entry.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    pub id: String,
    pub kind: String,
    pub payload: Json,
}

/// Parse a poll response `{"task": {...}}`; `null`/absent → `None` (no work).
pub fn parse_task(body: &str) -> Option<Task> {
    let v = parse_json(body).ok()?;
    let t = v.get("task")?;
    if t.is_null() {
        return None;
    }
    Some(Task {
        id: t.get("id")?.as_str()?.to_string(),
        kind: t.get("task_type")?.as_str()?.to_string(),
        payload: t.get("payload").cloned().unwrap_or(Json::Null),
    })
}

/// The engine's task outcome. `NotOurs` → the caller requeues for the right worker.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Completed(String),
    Failed(String),
    NotOurs,
}

/// Execute a task type this engine owns. dry-run (`live=false`) never opens a socket.
pub fn execute(kind: &str, payload: &Json, cfg: &WorkerConfig, live: bool) -> Outcome {
    match kind {
        "ping" => Outcome::Completed(format!(
            "pong from {} worker {}",
            crate::CRATE_NAME,
            cfg.peer_id
        )),
        "rusty_chat" => execute_chat(payload, cfg, live),
        _ => Outcome::NotOurs,
    }
}

/// Build the local `rusty serve` chat body from a `rusty_chat` payload.
pub fn chat_body(payload: &Json, model: &str) -> String {
    let prompt = payload
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("Hello");
    let max_tokens = payload
        .get("max_tokens")
        .and_then(|n| n.as_u64())
        .map(|n| n.clamp(1, 512))
        .unwrap_or(64);
    format!(
        r#"{{"model":"{}","messages":[{{"role":"user","content":"{}"}}],"max_tokens":{},"temperature":0}}"#,
        escape(model),
        escape(prompt),
        max_tokens
    )
}

fn execute_chat(payload: &Json, cfg: &WorkerConfig, live: bool) -> Outcome {
    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(&cfg.model);
    let body = chat_body(payload, model);
    if !live {
        return Outcome::Completed(format!(
            "dry-run: would POST {}{} ({})",
            cfg.serve_url.trim_end_matches('/'),
            chat_path(),
            body.chars().take(80).collect::<String>()
        ));
    }
    let url = format!("{}{}", cfg.serve_url.trim_end_matches('/'), chat_path());
    match http_call("POST", &url, None, Some(&body), 600) {
        Ok((st, resp)) if (200..300).contains(&st) => match extract_content(&resp) {
            Some(text) => Outcome::Completed(text),
            None => Outcome::Failed("chat: no choices[0].message.content".to_string()),
        },
        Ok((st, resp)) => Outcome::Failed(format!(
            "rusty serve HTTP {st}: {}",
            resp.chars().take(200).collect::<String>()
        )),
        Err(e) => Outcome::Failed(format!("rusty serve: {e}")),
    }
}

/// Pull `choices[0].message.content` (non-streaming JSON) out of a chat body.
pub fn extract_content(body: &str) -> Option<String> {
    let v = parse_json(body).ok()?;
    v.get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?
        .as_str()
        .map(str::to_string)
}

/// Register request body (`POST discovery/register-remote`).
pub fn register_body(cfg: &WorkerConfig, addr: &str, mem_mb: usize) -> String {
    format!(
        r#"{{"peer_id":"{}","address":"{}","port":0,"capabilities":{{"cpu_cores":{},"memory_mb":{},"gpu_devices":[],"supports_tensor_parallelism":false,"supports_pipeline_parallelism":false}},"metadata":{{"origin":"{}","role":"{}","engine":"{}","model":"{}","serve_url":"{}"}}}}"#,
        escape(&cfg.peer_id),
        escape(addr),
        cfg.cpu_cores,
        mem_mb,
        ORIGIN,
        ROLE,
        crate::CRATE_NAME,
        escape(&cfg.model),
        escape(&cfg.serve_url),
    )
}

/// Heartbeat request body (`POST discovery/heartbeat-remote`).
pub fn heartbeat_body(cfg: &WorkerConfig, mem_mb: usize) -> String {
    format!(
        r#"{{"peer_id":"{}","capabilities":{{"cpu_cores":{},"memory_mb":{},"gpu_devices":[],"supports_tensor_parallelism":false,"supports_pipeline_parallelism":false,"active_requests":0,"capacity":4,"current_load":0.0}}}}"#,
        escape(&cfg.peer_id),
        cfg.cpu_cores,
        mem_mb
    )
}

/// Pool-join request body (`POST virtual-nodes/{peer}/pool/join`).
pub fn join_body(mem_mb: usize) -> String {
    format!(r#"{{"max_memory_mb":{mem_mb},"max_concurrent_requests":4}}"#)
}

/// Complete request body (`POST …/tasks/{id}/complete`).
pub fn complete_body(outcome: &Outcome) -> Option<String> {
    match outcome {
        Outcome::Completed(detail) => Some(format!(
            r#"{{"status":"completed","detail":"{}"}}"#,
            escape(detail)
        )),
        Outcome::Failed(detail) => Some(format!(
            r#"{{"status":"failed","detail":"{}"}}"#,
            escape(detail)
        )),
        Outcome::NotOurs => None,
    }
}

/// Requeue request body (`POST virtual-nodes/{peer}/tasks`).
pub fn requeue_body(task: &Task) -> String {
    format!(
        r#"{{"task_type":"{}","payload":{}}}"#,
        escape(&task.kind),
        task.payload.to_json()
    )
}

/// Mask a secret for logs: shows only the last 4 chars (never the whole token).
pub fn mask(secret: &str) -> String {
    let s = secret.trim();
    if s.is_empty() {
        return "<unset>".to_string();
    }
    let tail: String = s
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("…{tail}")
}

/// Ordered dry-run plan (never includes the token).
pub fn plan_lines(cfg: &WorkerConfig, mem_mb: usize) -> Vec<String> {
    let addr = local_addr();
    vec![
        format!("hub edge:   {}", cfg.hub),
        format!(
            "peer id:    {}  (origin {ORIGIN}, role {ROLE})",
            cfg.peer_id
        ),
        format!("serve url:  {}  (model {})", cfg.serve_url, cfg.model),
        format!("token:      {} (header x-gsv-edge-token)", mask(&cfg.token)),
        format!("capabilities: {}", capabilities().join(",")),
        format!("ram/cpu:    {mem_mb} MiB / {} cores", cfg.cpu_cores),
        format!(
            "1. POST {}/{}",
            cfg.hub.trim_end_matches('/'),
            register_path()
        ),
        format!("   body {}", register_body(cfg, &addr, mem_mb)),
        format!(
            "2. POST {}/{}",
            cfg.hub.trim_end_matches('/'),
            join_path(&cfg.peer_id)
        ),
        format!("   body {}", join_body(mem_mb)),
        format!(
            "loop: heartbeat → {}/{} every {HEARTBEAT_SECS}s; poll → {}/{} every {POLL_SECS}s",
            cfg.hub.trim_end_matches('/'),
            heartbeat_path(),
            cfg.hub.trim_end_matches('/'),
            poll_path(&cfg.peer_id),
        ),
    ]
}

/// Heartbeat cadence (matches the hub keep-live 15s window).
pub const HEARTBEAT_SECS: u64 = 15;
/// Task poll cadence.
pub const POLL_SECS: u64 = 5;

// ---------------------------------------------------------------------------
// Blocking HTTP (used only by `--live`)
// ---------------------------------------------------------------------------

/// One blocking HTTP call: `(status, body)`. Adds the edge token header for hub calls.
pub fn http_call(
    method: &str,
    url: &str,
    token: Option<&str>,
    body: Option<&str>,
    timeout_secs: u64,
) -> Result<(u16, String), String> {
    let (host_port, path) = split_url(url)?;
    let timeout = Duration::from_secs(timeout_secs.max(1));
    let mut stream = std::net::TcpStream::connect_timeout(&parse_addr(&host_port)?, timeout)
        .map_err(|e| format!("connect {host_port}: {e}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    let mut token_line = String::new();
    if let Some(t) = token.filter(|t| !t.trim().is_empty()) {
        token_line = format!("\r\nx-gsv-edge-token: {}", t.trim());
    }
    let len = body.map(str::len).unwrap_or(0);
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json{token_line}\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{}",
        body.unwrap_or("")
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("send: {e}"))?;
    let mut raw = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => raw.extend_from_slice(&buf[..n]),
            Err(e) => return Err(format!("recv: {e}")),
        }
    }
    let text = String::from_utf8_lossy(&raw).to_string();
    let status: u16 = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| "bad status line".to_string())?;
    let body = text
        .split_once("\r\n\r\n")
        .map(|x| x.1.to_string())
        .unwrap_or_default();
    Ok((status, body))
}

fn split_url(url: &str) -> Result<(String, String), String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("only http:// URLs: {url}"))?;
    match rest.split_once('/') {
        Some((h, p)) => Ok((h.to_string(), format!("/{p}"))),
        None => Ok((rest.to_string(), "/".to_string())),
    }
}

fn parse_addr(host_port: &str) -> Result<std::net::SocketAddr, String> {
    let (host, port) = host_port
        .rsplit_once(':')
        .ok_or_else(|| format!("bad host:port: {host_port}"))?;
    let port: u16 = port.parse().map_err(|_| format!("bad port: {host_port}"))?;
    use std::net::ToSocketAddrs as _;
    (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("resolve {host_port}: {e}"))?
        .next()
        .ok_or_else(|| format!("no addr: {host_port}"))
}

/// Join the hub edge (register + pool/join). Returns the register response body.
pub fn join_once(cfg: &WorkerConfig, mem_mb: usize) -> Result<String, String> {
    let hub = cfg.hub.trim_end_matches('/');
    let addr = local_addr();
    let (st, body) = http_call(
        "POST",
        &format!("{hub}/{}", register_path()),
        Some(&cfg.token),
        Some(&register_body(cfg, &addr, mem_mb)),
        HEARTBEAT_SECS,
    )?;
    if !(200..300).contains(&st) {
        return Err(format!(
            "register HTTP {st}: {}",
            body.chars().take(200).collect::<String>()
        ));
    }
    let (st, body) = http_call(
        "POST",
        &format!("{hub}/{}", join_path(&cfg.peer_id)),
        Some(&cfg.token),
        Some(&join_body(mem_mb)),
        HEARTBEAT_SECS,
    )?;
    // 503 = pool not ready yet: stay discovery-only, like llama_edge.
    if !(200..300).contains(&st) && st != 503 {
        return Err(format!(
            "join HTTP {st}: {}",
            body.chars().take(200).collect::<String>()
        ));
    }
    Ok(body)
}

/// One heartbeat pass (`POST discovery/heartbeat-remote`).
pub fn heartbeat_once(cfg: &WorkerConfig, mem_mb: usize) -> Result<u16, String> {
    let hub = cfg.hub.trim_end_matches('/');
    let (st, _) = http_call(
        "POST",
        &format!("{hub}/{}", heartbeat_path()),
        Some(&cfg.token),
        Some(&heartbeat_body(cfg, mem_mb)),
        HEARTBEAT_SECS,
    )?;
    Ok(st)
}

/// One poll pass: pop a task, execute, complete or requeue. Returns a log line.
pub fn poll_once(cfg: &WorkerConfig, live: bool) -> Result<String, String> {
    let hub = cfg.hub.trim_end_matches('/');
    let (st, body) = http_call(
        "GET",
        &format!("{hub}/{}", poll_path(&cfg.peer_id)),
        Some(&cfg.token),
        None,
        HEARTBEAT_SECS,
    )?;
    if !(200..300).contains(&st) {
        return Ok(format!("poll HTTP {st}"));
    }
    let Some(task) = parse_task(&body) else {
        return Ok(String::new());
    };
    let outcome = execute(&task.kind, &task.payload, cfg, live);
    match complete_body(&outcome) {
        Some(cb) => {
            let (st, _) = http_call(
                "POST",
                &format!("{hub}/{}", complete_path(&cfg.peer_id, &task.id)),
                Some(&cfg.token),
                Some(&cb),
                HEARTBEAT_SECS,
            )?;
            Ok(format!(
                "task {} ({}) {} HTTP {}",
                task.id,
                task.kind,
                outcome_label(&outcome),
                st
            ))
        }
        None => {
            let (st, _) = http_call(
                "POST",
                &format!("{hub}/{}", requeue_path(&cfg.peer_id)),
                Some(&cfg.token),
                Some(&requeue_body(&task)),
                HEARTBEAT_SECS,
            )?;
            Ok(format!(
                "task {} ({}) requeued for its worker HTTP {}",
                task.id, task.kind, st
            ))
        }
    }
}

fn outcome_label(o: &Outcome) -> &'static str {
    match o {
        Outcome::Completed(_) => "completed",
        Outcome::Failed(_) => "failed",
        Outcome::NotOurs => "not-ours",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> WorkerConfig {
        WorkerConfig {
            peer_id: "rusty-01".into(),
            hub: "http://127.0.0.1:9999/api/edge".into(),
            token: "super-secret-edge-token".into(),
            serve_url: "http://127.0.0.1:8090".into(),
            model: "rusty-05".into(),
            max_memory_mb: 4096,
            cpu_cores: 8,
        }
    }

    #[test]
    fn json_round_trips_nested() {
        let src = r#"{"a":1,"b":[true,null,"x\n"],"c":{"d":-2.5}}"#;
        let v = parse_json(src).expect("parse");
        assert_eq!(v.get("a").and_then(Json::as_u64), Some(1));
        assert_eq!(
            v.get("b").and_then(Json::as_array).map(<[Json]>::len),
            Some(3)
        );
        assert_eq!(
            v.get("c").and_then(|c| c.get("d")).and_then(Json::as_u64),
            None
        );
        // Re-emit is stable JSON.
        let out = v.to_json();
        assert!(out.starts_with('{') && out.ends_with('}'));
        assert_eq!(parse_json(&out).expect("reparse"), v);
    }

    #[test]
    fn parse_json_rejects_trailing_garbage() {
        assert!(parse_json("{} x").is_err());
        assert!(parse_json("{").is_err());
        assert!(parse_json("").is_err());
    }

    #[test]
    fn escape_handles_quotes_and_controls() {
        assert_eq!(escape("a\"b\\c"), "a\\\"b\\\\c");
        assert_eq!(escape("l1\nl2"), "l1\\nl2");
    }

    #[test]
    fn parse_task_extracts_id_kind_payload() {
        let body = r#"{"task":{"id":"t-1","task_type":"rusty_chat","payload":{"prompt":"hi","max_tokens":4}}}"#;
        let t = parse_task(body).expect("task");
        assert_eq!(t.id, "t-1");
        assert_eq!(t.kind, "rusty_chat");
        assert_eq!(t.payload.get("prompt").and_then(Json::as_str), Some("hi"));
    }

    #[test]
    fn parse_task_null_or_absent_is_none() {
        assert!(parse_task(r#"{"task":null}"#).is_none());
        assert!(parse_task(r#"{}"#).is_none());
        assert!(parse_task("not json").is_none());
    }

    #[test]
    fn register_body_carries_role_and_is_token_free() {
        let b = register_body(&cfg(), "192.168.2.238", 4096);
        assert!(b.contains(r#""peer_id":"rusty-01""#));
        assert!(b.contains(&format!(r#""origin":"{ORIGIN}""#)));
        assert!(b.contains(&format!(r#""role":"{ROLE}""#)));
        assert!(b.contains(r#""memory_mb":4096"#));
        assert!(
            !b.contains("super-secret"),
            "token must never appear in a body"
        );
        // valid JSON shape
        assert!(parse_json(&b).is_ok(), "{b}");
    }

    #[test]
    fn heartbeat_and_join_bodies_are_valid_json() {
        let cfg = cfg();
        assert!(parse_json(&heartbeat_body(&cfg, 2048)).is_ok());
        assert!(parse_json(&join_body(2048)).is_ok());
        assert!(heartbeat_body(&cfg, 8).contains(r#""active_requests":0"#));
    }

    #[test]
    fn paths_are_hub_edge_relative() {
        assert_eq!(register_path(), "discovery/register-remote");
        assert_eq!(poll_path("rusty-01"), "virtual-nodes/rusty-01/tasks/poll");
        assert_eq!(
            complete_path("rusty-01", "t-9"),
            "virtual-nodes/rusty-01/tasks/t-9/complete"
        );
    }

    #[test]
    fn execute_ping_and_unknown() {
        let cfg = cfg();
        let pong = execute("ping", &Json::Null, &cfg, false);
        assert!(matches!(pong, Outcome::Completed(ref d) if d.contains("pong")));
        assert_eq!(
            execute("test_ping", &Json::Null, &cfg, false),
            Outcome::NotOurs
        );
        assert_eq!(
            execute("llama_shard", &Json::Null, &cfg, false),
            Outcome::NotOurs
        );
    }

    #[test]
    fn execute_chat_dry_run_never_opens_a_socket() {
        let payload = parse_json(r#"{"prompt":"hi","max_tokens":4}"#).expect("payload");
        let out = execute("rusty_chat", &payload, &cfg(), false);
        match out {
            Outcome::Completed(d) => assert!(
                d.starts_with("dry-run: would POST http://127.0.0.1:8090/v1/chat/completions")
            ),
            other => panic!("expected dry-run completion, got {other:?}"),
        }
    }

    #[test]
    fn chat_body_escapes_and_caps() {
        let payload = parse_json(r#"{"prompt":"a\"b","max_tokens":9999}"#).expect("payload");
        let b = chat_body(&payload, "rusty-05");
        assert!(b.contains(r#""content":"a\"b""#));
        assert!(b.contains(r#""max_tokens":512"#));
        assert!(parse_json(&b).is_ok());
    }

    #[test]
    fn extract_content_shapes() {
        let ok = r#"{"choices":[{"message":{"role":"assistant","content":"hey"}}]}"#;
        assert_eq!(extract_content(ok).as_deref(), Some("hey"));
        assert_eq!(extract_content(r#"{"choices":[]}"#), None);
        assert_eq!(extract_content("nope"), None);
    }

    #[test]
    fn complete_and_requeue_bodies() {
        let done = complete_body(&Outcome::Completed("ok".into())).expect("body");
        assert!(done.contains(r#""status":"completed""#));
        let failed = complete_body(&Outcome::Failed("bad".into())).expect("body");
        assert!(failed.contains(r#""status":"failed""#));
        assert!(complete_body(&Outcome::NotOurs).is_none());

        let task = Task {
            id: "t-1".into(),
            kind: "test_ping".into(),
            payload: parse_json(r#"{"x":1}"#).expect("payload"),
        };
        let r = requeue_body(&task);
        assert!(r.contains(r#""task_type":"test_ping""#));
        assert!(parse_json(&r).is_ok());
    }

    #[test]
    fn mask_never_reveals_the_token() {
        assert_eq!(mask(""), "<unset>");
        let m = mask("abcdef123456");
        assert_eq!(m, "…3456");
        assert!(!m.contains("abcdef"));
    }

    #[test]
    fn plan_lines_never_contain_the_token() {
        let cfg = cfg();
        for line in plan_lines(&cfg, 4096) {
            assert!(
                !line.contains("super-secret-edge-token"),
                "plan leaked the token: {line}"
            );
        }
        let joined = plan_lines(&cfg, 4096).join("\n");
        assert!(joined.contains("register-remote"));
        assert!(joined.contains(ROLE));
    }

    #[test]
    fn split_url_and_bad_inputs() {
        assert_eq!(
            split_url("http://127.0.0.1:9999/api/edge").unwrap(),
            ("127.0.0.1:9999".to_string(), "/api/edge".to_string())
        );
        assert!(split_url("ftp://x").is_err());
        assert!(parse_addr("nohostport").is_err());
    }
}
