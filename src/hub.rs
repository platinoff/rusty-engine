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
}
