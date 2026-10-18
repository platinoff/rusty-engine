//! Minimal std-only persistent thread pool for row-parallel projections (`PH-203e`).
//!
//! `std::thread::scope` creates and joins OS threads on every call; on this box that is
//! ~66 µs per thread, and a decode step issues ~120 projections, so spawn+join was ~46%
//! of the token time (measured in `benches/quant.rs`). Workers here live for the whole
//! process and jobs are queued instead.
//!
//! Borrowed data is supported the same way scoped threads are: [`parallel_for`] erases the
//! closure's lifetime to `'static` and blocks until the pool reports **no pending jobs at
//! all**, so no queued job can outlive the caller's borrows. Consequently `parallel_for`
//! must never be called from inside a pool job (that would wait on itself).

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Inner {
    jobs: VecDeque<Job>,
    pending: usize,
}

struct Shared {
    inner: Mutex<Inner>,
    work: Condvar,
    done: Condvar,
}

struct Pool {
    shared: Arc<Shared>,
}

impl Pool {
    fn new(workers: usize) -> Self {
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                jobs: VecDeque::new(),
                pending: 0,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
        });
        for _ in 0..workers.max(1) {
            let s = Arc::clone(&shared);
            // Detached: workers live for the process and are never joined.
            let _ = std::thread::Builder::new()
                .name("rusty-pool".into())
                .spawn(move || worker(s));
        }
        Self { shared }
    }

    fn submit(&self, job: Job) {
        let mut g = self.shared.inner.lock().unwrap();
        g.pending += 1;
        g.jobs.push_back(job);
        drop(g);
        self.shared.work.notify_one();
    }

    fn wait(&self) {
        let mut g = self.shared.inner.lock().unwrap();
        while g.pending > 0 {
            g = self.shared.done.wait(g).unwrap();
        }
    }
}

fn worker(shared: Arc<Shared>) {
    loop {
        let job = {
            let mut g = shared.inner.lock().unwrap();
            loop {
                if let Some(j) = g.jobs.pop_front() {
                    break j;
                }
                g = shared.work.wait(g).unwrap();
            }
        };
        // A panicking job must not take the worker (or the `pending` count) with it.
        let _ = catch_unwind(AssertUnwindSafe(job));
        let mut g = shared.inner.lock().unwrap();
        g.pending -= 1;
        if g.pending == 0 {
            shared.done.notify_all();
        }
    }
}

/// Requested worker count: `RUSTY_THREADS` when set (1–256), else the machine's available
/// parallelism; never zero.
pub fn requested_threads() -> usize {
    if let Ok(v) = std::env::var("RUSTY_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            if n >= 1 {
                return n.min(256);
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::new(requested_threads()))
}

/// Type-erased handle to the borrowed job body.
///
/// The pointee is a shared `&F` valid for the whole [`parallel_for`] call, and
/// `parallel_for` does not return until every job using it has finished, so copying this
/// handle into a `'static` job is sound.
#[derive(Clone, Copy)]
struct Handle {
    body: *const (),
    call: unsafe fn(*const (), usize),
}

// SAFETY: jobs only ever call `&F` concurrently, and `F: Sync` (checked in `parallel_for`)
// makes that sound.
unsafe impl Send for Handle {}

impl Handle {
    fn new<F: Fn(usize)>(f: &F) -> Self {
        unsafe fn call<F: Fn(usize)>(body: *const (), i: usize) {
            (*(body as *const F))(i);
        }
        Self {
            body: f as *const F as *const (),
            call: call::<F>,
        }
    }

    /// SAFETY: the closure `new` was given must still be alive.
    unsafe fn run(self, i: usize) {
        (self.call)(self.body, i)
    }
}

/// Run `f(i)` for every `i in 0..jobs` on the persistent pool and return once all of them
/// (and any other job submitted concurrently) have finished.
pub fn parallel_for<F>(jobs: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if jobs <= 1 {
        f(0);
        return;
    }
    let p = pool();
    let handle = Handle::new(&f);
    for i in 0..jobs {
        let h = handle;
        p.submit(Box::new(move || {
            // SAFETY: `f` outlives this job — `p.wait()` below joins every job first.
            unsafe { h.run(i) };
        }));
    }
    p.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_for_runs_every_index_and_joins_before_returning() {
        let data = [7u32; 16];
        let seen = Mutex::new(Vec::new());
        parallel_for(8, |i| {
            // Borrowed data must still be alive inside the job and visible after the join.
            let v = data[i] * i as u32;
            seen.lock().unwrap().push(v);
        });
        let mut got = seen.into_inner().unwrap();
        got.sort_unstable();
        assert_eq!(got, (0..8u32).map(|i| i * 7).collect::<Vec<_>>());
    }

    #[test]
    fn parallel_for_single_job_runs_inline() {
        let hit = std::sync::atomic::AtomicUsize::new(0);
        parallel_for(1, |i| {
            hit.store(i + 1, std::sync::atomic::Ordering::Relaxed)
        });
        assert_eq!(hit.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn parallel_for_survives_a_panicking_job() {
        let seen = Mutex::new(Vec::new());
        parallel_for(4, |i| {
            if i == 2 {
                panic!("deliberate job panic");
            }
            seen.lock().unwrap().push(i);
        });
        let mut got = seen.into_inner().unwrap();
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 3]);
    }
}
