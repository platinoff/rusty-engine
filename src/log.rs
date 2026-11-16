//! Std-only rolling file sink for the live `rusty serve` process.
//!
//! No external deps — this plugin keeps the build offline-green by policy.
//! `target/live/logs/rusty-serve.log` is capped at [`MAX_BYTES`]; when full it
//! is rolled to `.1`..`.7` (oldest dropped). Resolve via `RUSTY_LOG_DIR` or
//! the crate root — never from CWD.
//!
//! Timestamps are RFC3339-ish UTC derived from `SystemTime` (no chrono dep).

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Rotate when active file reaches this size (4 MiB).
pub const MAX_BYTES: u64 = 4 * 1024 * 1024;
/// Rolled files to keep (active + 7 generations = 8 * MAX_BYTES worst case).
pub const KEEP: usize = 7;

/// Rolling sink directory: env wins, else crate's `target/live/logs`.
pub fn log_dir() -> PathBuf {
    std::env::var_os("RUSTY_LOG_DIR")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/live/logs"))
}

fn roll(dir: &std::path::Path, active: &std::path::Path) {
    // drop .KEEP, shift .n-1 -> .n, .1 -> ... then archive the active file as .1
    let _ = fs::remove_file(dir.join(format!("rusty-serve.log.{KEEP}")));
    for i in (1..KEEP).rev() {
        let from = dir.join(format!("rusty-serve.log.{i}"));
        let to = dir.join(format!("rusty-serve.log.{}", i + 1));
        if from.exists() {
            let _ = fs::rename(from, to);
        }
    }
    let _ = fs::rename(active, dir.join("rusty-serve.log.1"));
}

fn rolling_open() -> Option<std::fs::File> {
    let dir = log_dir();
    fs::create_dir_all(&dir).ok()?;
    let active = dir.join("rusty-serve.log");
    if let Ok(md) = fs::metadata(&active) {
        if md.len() >= MAX_BYTES {
            roll(&dir, &active);
        }
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(&active)
        .ok()
}

fn stamp() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", d.as_secs())
}

/// Append one line; falls back to stderr rather than dropping a diagnostic.
pub fn log_line(tag: &str, msg: &str) {
    let line = format!("{}\t{}\t{}\n", stamp(), tag, msg);
    match rolling_open() {
        Some(mut f) => {
            let _ = f.write_all(line.as_bytes());
        }
        None => eprintln!("{}", line.trim_end()),
    }
}
