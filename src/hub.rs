//! GSV-native master supervisor — "Rust kubernetes-lite".
//!
//! The GSV hub (`:9999`) is the master; `rusty` slaves register over local
//! Wi-Fi/LAN, heartbeat on a fixed tick, take model-placement leases and report
//! health (reuse of the llama-rs heartbeat + poolAI ed25519 job-lease patterns).
//! B0 fixes the tick and the worker descriptor; scheduling is band B8.

/// Heartbeat tick seconds, aligned with the llama-rs `keep_live` 15s cadence.
pub const HEARTBEAT_TICK_SECS: u64 = 15;

/// Freshness window — a worker is considered alive if seen within this many secs.
pub const HEARTBEAT_FRESH_SECS: u64 = 60;

/// A registered worker in the fleet.
#[derive(Debug, Clone, PartialEq)]
pub struct Worker {
    pub id: String,
    pub role: crate::node::Role,
    pub addr: String,
}

/// keep-live heartbeat payload written to `target/live/rusty_heartbeat.json` (PH-504), in
/// the llama-rs `keep_live` shape the GSV hub mirrors. `alive` is the self-report; the hub
/// applies [`is_fresh`] against its own clock to decide liveness (fail-open on peers).
#[derive(Debug, Clone, PartialEq)]
pub struct Heartbeat {
    pub name: String,
    pub version: String,
    pub uptime_secs: u64,
    pub alive: bool,
    pub tick_secs: u64,
    pub fresh_secs: u64,
}

impl Heartbeat {
    /// A heartbeat stamped from `uptime_secs` with the crate identity + hub cadence.
    pub fn new(uptime_secs: u64) -> Self {
        Self {
            name: crate::CRATE_NAME.to_string(),
            version: crate::VERSION.to_string(),
            uptime_secs,
            alive: true,
            tick_secs: HEARTBEAT_TICK_SECS,
            fresh_secs: HEARTBEAT_FRESH_SECS,
        }
    }

    /// Serialize to the exact JSON written on disk (hand-built, `std`-only, no serde).
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"name":"{}","version":"{}","uptime_secs":{},"alive":{},"tick_secs":{},"fresh_secs":{}}}"#,
            self.name, self.version, self.uptime_secs, self.alive, self.tick_secs, self.fresh_secs
        )
    }
}

/// Whether a heartbeat seen at `last_secs` is still fresh against `now_secs`, using the hub
/// window. Saturating so a clock skew (`now < last`) never underflows to a huge gap.
pub fn is_fresh(last_secs: u64, now_secs: u64) -> bool {
    now_secs.saturating_sub(last_secs) <= HEARTBEAT_FRESH_SECS
}

/// Write the heartbeat to `<dir>/rusty_heartbeat.json`, creating `dir` if needed. Returns
/// the path written. The `serve` loop calls this every [`HEARTBEAT_TICK_SECS`]; the hub
/// `keep_live` probe reads it (band-233 LAN-first mirror).
pub fn write_heartbeat(
    dir: &std::path::Path,
    hb: &Heartbeat,
) -> std::io::Result<std::path::PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("rusty_heartbeat.json");
    std::fs::write(&path, hb.to_json())?;
    Ok(path)
}

// Compile-time invariant: a worker is only "fresh" over a window at least as
// wide as its own tick.
const _: () = assert!(HEARTBEAT_FRESH_SECS >= HEARTBEAT_TICK_SECS);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_roundtrip_clone_eq() {
        let w = Worker {
            id: "edge-1".into(),
            role: crate::node::Role::Slave,
            addr: "192.168.2.40:8080".into(),
        };
        assert_eq!(w.clone(), w);
    }

    #[test]
    fn heartbeat_json_carries_identity_and_cadence() {
        let hb = Heartbeat::new(123);
        let json = hb.to_json();
        assert!(json.contains(&format!("\"name\":\"{}\"", crate::CRATE_NAME)));
        assert!(json.contains(&format!("\"version\":\"{}\"", crate::VERSION)));
        assert!(json.contains("\"uptime_secs\":123"));
        assert!(json.contains("\"alive\":true"));
        assert!(json.contains(&format!("\"tick_secs\":{HEARTBEAT_TICK_SECS}")));
        // valid-shaped: starts with { and ends with }
        assert!(json.starts_with('{') && json.ends_with('}'), "{json}");
    }

    #[test]
    fn is_fresh_respects_the_window() {
        assert!(is_fresh(100, 100)); // same tick
        assert!(is_fresh(100, 100 + HEARTBEAT_FRESH_SECS)); // boundary still fresh
        assert!(!is_fresh(100, 100 + HEARTBEAT_FRESH_SECS + 1)); // stale past window
        assert!(is_fresh(200, 150), "clock skew must not underflow to stale");
    }

    #[test]
    fn write_heartbeat_lands_a_file() {
        let dir = std::path::Path::new("target/live-test");
        let path = write_heartbeat(dir, &Heartbeat::new(1)).expect("write");
        assert!(path.exists());
        let s = std::fs::read_to_string(&path).expect("read");
        assert!(s.contains("\"alive\":true"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
