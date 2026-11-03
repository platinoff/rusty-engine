//! `EN-7b` — SSD delta-state cache: persist [`crate::runner35::Q35Snapshot`] files so
//! a repeated prompt restores its carried decoder state from disk and **skips the
//! prefill entirely** (the "time out from headers" storm root cause — the 641 MB
//! MiniCPM hybrid recomputing the same system prompt on every connection).
//!
//! Design shards:
//! - keyed by FNV-1a over the prompt **token ids** (byte-exact prompt identity);
//! - one file per key under the cache dir (default `target/live/state`, inside the
//!   repo tree; `RUSTY_STATE_DIR=off` disables, or points elsewhere within a tree);
//! - writes are atomic (`{key}.tmp` → rename), readers accept only snapshots that
//!   still validate against the live geometry — a corrupt/foreign file degrades to an
//!   ordinary prefill, never to a wrong token;
//! - thread-per-connection serve may hit the same key concurrently: both writers
//!   produce identical bytes for the same key, and the rename makes the swap atomic.
//!
//! std-only; no eviction yet (steady state is one file per distinct prompt — the
//! lane's traffic is a handful of system prompts; eviction is a later slice).

use std::io::Write;
use std::path::PathBuf;

use crate::core::{EngineError, Result};
use crate::runner35::Q35Snapshot;

/// FNV-1a 64-bit over the prompt token ids (u32 LE) — fast, stable across runs.
pub fn prompt_key(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in ids {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// SSD snapshot store. `Sync` — the EN-7 thread-per-connection lane shares one.
pub struct StateCache {
    dir: PathBuf,
}

impl StateCache {
    /// Open/resolve the cache dir: `RUSTY_STATE_DIR` env when set (`off` disables),
    /// else the repo-internal default. `None` = feature disabled; every caller then
    /// behaves exactly as before `EN-7b`.
    pub fn open_default() -> Option<Self> {
        match std::env::var_os("RUSTY_STATE_DIR") {
            Some(v) if v == *"off" => None,
            Some(v) => Some(Self {
                dir: PathBuf::from(v),
            }),
            None => Some(Self {
                dir: PathBuf::from("target/live/state"),
            }),
        }
    }

    /// Explicit directory (tests, CLI override).
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn file(&self, key: u64) -> PathBuf {
        self.dir.join(format!("{key:016x}.q35state"))
    }

    /// Store a snapshot under the prompt key. Directory is created on demand; a
    /// failing disk never fails the request — we log nothing and return `Err`, and
    /// the caller keeps serving uncached. Oversized snapshots beyond 256 MiB are
    /// refused (a runaway context must not fill the SSD while the owner is away).
    pub fn store(&self, key: u64, snap: &Q35Snapshot) -> Result<()> {
        const MAX_BYTES: u64 = 256 * 1024 * 1024;
        let bytes = snap.to_bytes();
        if bytes.len() as u64 > MAX_BYTES {
            return Err(EngineError {
                message: "statecache: snapshot exceeds 256 MiB cap".to_string(),
            });
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| EngineError {
            message: format!("statecache: create_dir_all: {e}"),
        })?;
        let tmp = self.dir.join(format!("{key:016x}.tmp"));
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| EngineError {
                message: format!("statecache: create tmp: {e}"),
            })?;
            f.write_all(&bytes).map_err(|e| EngineError {
                message: format!("statecache: write tmp: {e}"),
            })?;
            f.sync_all().ok();
        }
        std::fs::rename(&tmp, self.file(key)).map_err(|e| EngineError {
            message: format!("statecache: rename: {e}"),
        })?;
        Ok(())
    }

    /// Load the snapshot for a prompt key, if it exists and parses. Any failure is a
    /// miss (`None`) — the caller falls back to an ordinary prefill.
    pub fn load(&self, key: u64) -> Option<Q35Snapshot> {
        let p = self.file(key);
        let data = std::fs::read(p).ok()?;
        Q35Snapshot::from_bytes(&data).ok()
    }

    /// How many snapshot files the store currently holds (diagnostic only).
    pub fn count(&self) -> usize {
        std::fs::read_dir(&self.dir)
            .map(|d| {
                d.filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .to_str()
                            .map(|p| p.ends_with(".q35state"))
                            .unwrap_or(false)
                    })
                    .count()
            })
            .unwrap_or(0)
    }
}
