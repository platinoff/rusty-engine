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

/// Resident set size (RAM in use) of **this** process, in bytes, for the `/metrics`
/// panel (PH-505). `std`-only, no crates: Windows reads the working set via
/// `K32GetProcessMemoryInfo` (kernel32/psapi), Linux the resident page count from
/// `/proc/self/statm`. Returns 0 where the platform call is unavailable, so the
/// endpoint stays total on any target.
pub fn ram_bytes() -> u64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        extern "system" {
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
            fn K32GetProcessMemoryInfo(
                process: *mut core::ffi::c_void,
                counters: *mut ProcessMemoryCounters,
                cb: u32,
            ) -> i32;
        }
        let mut pmc: ProcessMemoryCounters = unsafe { core::mem::zeroed() };
        pmc.cb = core::mem::size_of::<ProcessMemoryCounters>() as u32;
        // SAFETY: `GetCurrentProcess` yields a pseudo-handle that is never closed and is
        // valid for the call; `pmc` is a live, correctly-sized `PROCESS_MEMORY_COUNTERS`.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) };
        if ok != 0 {
            pmc.working_set_size as u64
        } else {
            0
        }
    }
    #[cfg(target_os = "linux")]
    {
        // `/proc/self/statm` field 2 is resident pages; x86_64/arm64 pages are 4 KiB.
        let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
        statm
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
            .saturating_mul(4096)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        0
    }
}

/// Runtime metrics exposed at `GET /metrics` (PH-505). Pure counters the serve loop
/// accumulates; `tok_per_sec` is derived and zero-guarded so it never divides by 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub calls: u64,
    pub generated_tokens: u64,
    pub decode_secs: f64,
    pub active_experts: u64,
    pub peak_experts: u64,
}

impl Metrics {
    pub const fn zero() -> Self {
        Self {
            calls: 0,
            generated_tokens: 0,
            decode_secs: 0.0,
            active_experts: 0,
            peak_experts: 0,
        }
    }

    /// Record one completion: `tokens` generated over `secs` of decode, `active_experts`
    /// materialized in the top layer.
    pub fn record(&mut self, tokens: u64, secs: f64, active_experts: u64) {
        self.calls += 1;
        self.generated_tokens += tokens;
        self.decode_secs += secs;
        self.active_experts = active_experts;
        if active_experts > self.peak_experts {
            self.peak_experts = active_experts;
        }
    }

    /// Tokens per second, 0.0 when no decode time observed.
    pub fn tok_per_sec(&self) -> f64 {
        if self.decode_secs <= 0.0 {
            0.0
        } else {
            self.generated_tokens as f64 / self.decode_secs
        }
    }

    /// JSON (hand-built, std-only). Samples the live process RSS via [`ram_bytes`] so the
    /// `/metrics` panel reports tok/s · RAM · experts (PH-505).
    pub fn to_json(&self) -> String {
        let ram = ram_bytes();
        format!(
            r#"{{"calls":{},"generated_tokens":{},"decode_secs":{},"tok_per_sec":{},"active_experts":{},"peak_experts":{},"ram_bytes":{},"ram_mb":{:.1}}}"#,
            self.calls,
            self.generated_tokens,
            self.decode_secs,
            self.tok_per_sec(),
            self.active_experts,
            self.peak_experts,
            ram,
            ram as f64 / (1024.0 * 1024.0)
        )
    }
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

    #[test]
    fn metrics_zero_never_divides() {
        assert_eq!(Metrics::zero().tok_per_sec(), 0.0);
        assert_eq!(Metrics::zero().calls, 0);
    }

    #[test]
    fn metrics_record_tracks_rate_and_peak() {
        let mut m = Metrics::zero();
        m.record(30, 1.0, 8);
        m.record(20, 1.0, 4);
        assert_eq!(m.calls, 2);
        assert_eq!(m.generated_tokens, 50);
        assert!((m.tok_per_sec() - 25.0).abs() < 1e-9);
        assert_eq!(m.active_experts, 4, "last call's active count");
        assert_eq!(m.peak_experts, 8, "peak retained");
    }

    #[test]
    fn metrics_json_shape() {
        let mut m = Metrics::zero();
        m.record(1, 0.0, 1);
        let j = m.to_json();
        assert!(j.starts_with('{') && j.ends_with('}'));
        assert!(j.contains("\"generated_tokens\":1"));
        assert!(j.contains("\"tok_per_sec\":0"));
        assert!(j.contains("\"ram_bytes\":"));
        assert!(j.contains("\"ram_mb\":"));
    }

    #[test]
    fn ram_bytes_reads_the_process_where_supported() {
        if cfg!(windows) || cfg!(target_os = "linux") {
            assert!(ram_bytes() > 0, "expected a positive working set");
        }
    }
}
