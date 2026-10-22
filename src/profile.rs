//! Opt-in wall-clock accounting of the hot forward stages (`RUSTY_PROFILE=1`).
//!
//! std-only and allocation-free in the hot path: one relaxed atomic load decides whether a
//! projection is timed at all, so a normal release build pays a single predictable branch.
//! The counters are per-stage atomics summed across the worker pool, i.e. **thread time**,
//! which is why [`report`] prints them against the measured wall clock of the caller and
//! leaves the remainder as `other` (serial parts: norms, rope, KV copies, sampling).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Timed stage (index into the counter arrays).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// One [`crate::qmat::QMat::matvec`] call (decode-shaped projection).
    Matvec = 0,
    /// One [`crate::qmat::QMat::matmul`] call (prefill-shaped projection).
    Matmul = 1,
    /// One [`crate::forward`] prefill attention pass.
    Attention = 2,
}

const STAGES: [&str; 3] = ["matvec", "matmul", "attention"];

static INIT: AtomicBool = AtomicBool::new(false);
static ENABLED: AtomicBool = AtomicBool::new(false);
static REPORTED: AtomicBool = AtomicBool::new(false);
static NS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static CALLS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

fn init_once() {
    if INIT.swap(true, Ordering::Relaxed) {
        return;
    }
    let on = std::env::var_os("RUSTY_PROFILE").is_some();
    ENABLED.store(on, Ordering::Relaxed);
}

/// Is stage timing on for this process?
#[inline]
pub fn enabled() -> bool {
    if !INIT.load(Ordering::Relaxed) {
        init_once();
    }
    ENABLED.load(Ordering::Relaxed)
}

/// Run `f`, adding its elapsed time to `stage` when profiling is enabled.
#[inline]
pub fn timed<R>(stage: Stage, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let r = f();
    add(stage, t.elapsed().as_nanos() as u64);
    r
}

/// Add `ns` to a stage counter (called by wall-clock-multiplexing callers).
pub fn add(stage: Stage, ns: u64) {
    let i = stage as usize;
    NS[i].fetch_add(ns, Ordering::Relaxed);
    CALLS[i].fetch_add(1, Ordering::Relaxed);
}

fn take() -> ([u64; 3], [u64; 3]) {
    let mut ns = [0u64; 3];
    let mut calls = [0u64; 3];
    for i in 0..3 {
        ns[i] = NS[i].swap(0, Ordering::Relaxed);
        calls[i] = CALLS[i].swap(0, Ordering::Relaxed);
    }
    (ns, calls)
}

/// Per-shape matmul accounting (profiling only): which `m × n × rows` projections eat the
/// prefill. The hot path never touches this unless `RUSTY_PROFILE` is set.
#[derive(Clone, Copy)]
struct ShapeRow {
    m: usize,
    n: usize,
    rows: usize,
    ns: u64,
    calls: u64,
}

static SHAPES: std::sync::Mutex<Vec<ShapeRow>> = std::sync::Mutex::new(Vec::new());

/// Record one timed `matmul` (no-op when profiling is off).
pub fn note_matmul(ns: u64, m: usize, n: usize, rows: usize) {
    add(Stage::Matmul, ns);
    if !enabled() {
        return;
    }
    let mut rows_v = match SHAPES.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(r) = rows_v
        .iter_mut()
        .find(|r| r.m == m && r.n == n && r.rows == rows)
    {
        r.ns += ns;
        r.calls += 1;
    } else {
        rows_v.push(ShapeRow {
            m,
            n,
            rows,
            ns,
            calls: 1,
        });
    }
}

fn take_shapes() -> Vec<ShapeRow> {
    let mut g = match SHAPES.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let mut v: Vec<ShapeRow> = std::mem::take(&mut *g);
    v.sort_by(|a, b| b.ns.cmp(&a.ns));
    v
}

/// Human-readable per-stage table against `total_ns` of caller-measured wall clock, with
/// the accounting reset so the next prefill starts clean.
pub fn report(total_ns: u64, label: &str) -> String {
    let (ns, calls) = take();
    let total = total_ns.max(1) as f64 / 1e6;
    let mut out = format!("[profile] {label}: wall {total:.1} ms\n");
    let mut sum = 0.0f64;
    for i in 0..3 {
        let ms = ns[i] as f64 / 1e6;
        sum += ms;
        let per_call = if calls[i] == 0 {
            0.0
        } else {
            ns[i] as f64 / calls[i] as f64 / 1000.0
        };
        out.push_str(&format!(
            "  {:<9} {:>9.1} ms  {:>5.1}%  {:>6} calls  {:>8.1} µs/call\n",
            STAGES[i],
            ms,
            100.0 * ms / total,
            calls[i],
            per_call
        ));
    }
    out.push_str(&format!(
        "  {:<9} {:>9.1} ms  {:>5.1}%  (thread time of the pool, not wall)\n",
        "measured",
        sum,
        100.0 * sum / total
    ));
    let shapes = take_shapes();
    if !shapes.is_empty() {
        out.push_str("  matmul by shape (worst first):\n");
        for s in shapes.iter().take(10) {
            let mac = (s.rows as f64) * (s.m as f64) * (s.n as f64);
            let ms = s.ns as f64 / 1e6;
            let gmac_s = if s.ns == 0 {
                0.0
            } else {
                mac * s.calls as f64 / 1e9 / (s.ns as f64 / 1e9)
            };
            out.push_str(&format!(
                "    {:>5}x{:<5} rows {:>5}  {:>8.1} ms  {:>4} calls  {:>6.1} GMAC/s\n",
                s.m, s.n, s.rows, ms, s.calls, gmac_s
            ));
        }
    }
    out
}

/// Print [`report`] to stderr (no-op when profiling is off).
pub fn print_report(total_ns: u64, label: &str) {
    if enabled() {
        eprint!("{}", report(total_ns, label));
    }
}

/// Like [`print_report`], but only the first call in the process reports (the re-prefill
/// CLI path is called once per generated token, so it would otherwise spam).
pub fn print_report_once(total_ns: u64, label: &str) {
    if enabled() && !REPORTED.swap(true, Ordering::Relaxed) {
        eprint!("{}", report(total_ns, label));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_by_default_in_tests_and_timed_still_runs() {
        // The suite runs without RUSTY_PROFILE, so the counters stay at zero but the
        // closure must still run and return its value.
        let got = timed(Stage::Matvec, || 41 + 1);
        assert_eq!(got, 42);
        assert!(!enabled());
        assert_eq!(report(1_000_000, "test"), report(1_000_000, "test"));
    }

    #[test]
    fn report_accounts_calls_and_share() {
        add(Stage::Matmul, 3_000_000);
        add(Stage::Matmul, 1_000_000);
        let s = report(10_000_000, "unit");
        assert!(s.contains("matmul"), "{s}");
        assert!(s.contains("2 calls"), "{s}");
        assert!(s.contains("40.0%"), "{s}");
        // Accounting is consumed by the report.
        assert!(report(10_000_000, "unit").contains("0 calls"));
    }
}
