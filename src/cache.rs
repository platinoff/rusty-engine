//! Byte-budgeted LRU expert cache — band B4 `PH-404`.
//!
//! A real MoE model (Qwen3-30B-A3B / Mixtral) has 8–128 experts **per layer**, each a
//! full SwiGLU FFN (`inter×hidden` gate/up + `hidden×inter` down). Only the router's
//! top-k are touched by any token, so keeping every expert materialized in `f32` is
//! impossible on edge hosts. This cache bounds total resident bytes and evicts
//! least-recently-used experts (pinned experts are skipped), the substrate for the
//! NVMe/disk offload and hot-prefetch that `PH-406` binds to real GGUF tensors.
//!
//! The cache is **deterministic and `std`-only** — no async, no timers — so the LRU
//! policy and budget accounting are unit-testable without a model.

use crate::moe::ExpertWeights;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

/// Which expert, at which MoE layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub layer: usize,
    pub expert: usize,
}

impl ExpertKey {
    pub fn new(layer: usize, expert: usize) -> Self {
        Self { layer, expert }
    }
}

/// Approximate resident size of an expert's three weight matrices, in bytes (`f32`).
pub fn expert_bytes(w: &ExpertWeights) -> usize {
    (w.gate.len() + w.up.len() + w.down.len()) * std::mem::size_of::<f32>()
}

/// Cumulative cache behavior, for metrics (`PH-505`) and tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub loads: u64,
}

struct Entry {
    weights: Rc<ExpertWeights>,
    bytes: usize,
    pinned: bool,
}

/// Byte-budgeted, LRU-evicting cache of materialized [`ExpertWeights`].
pub struct ExpertCache {
    budget: usize,
    used: usize,
    /// Most-recently-used at the front; evict from the back.
    order: VecDeque<ExpertKey>,
    map: HashMap<ExpertKey, Entry>,
    stats: CacheStats,
}

impl ExpertCache {
    /// A cache holding at most `budget` bytes of expert weights.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            used: 0,
            order: VecDeque::new(),
            map: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }
    pub fn bytes_used(&self) -> usize {
        self.used
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Return a cached expert if present (and mark it most-recently-used).
    pub fn get(&mut self, key: &ExpertKey) -> Option<Rc<ExpertWeights>> {
        if self.map.contains_key(key) {
            self.touch(key);
            self.stats.hits += 1;
            self.map.get(key).map(|e| Rc::clone(&e.weights))
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Fetch, or call `load` on a miss, insert, and evict LRU until within budget.
    ///
    /// A single expert larger than the whole budget is *not* cached (nothing can be
    /// evicted that would admit it without dropping it immediately) — the loader result
    /// is still returned so the caller can use it transiently.
    pub fn get_or_load<L>(&mut self, key: ExpertKey, load: L) -> Rc<ExpertWeights>
    where
        L: FnOnce(ExpertKey) -> ExpertWeights,
    {
        if let Some(w) = self.get(&key) {
            return w;
        }
        self.stats.loads += 1;
        let weights = Rc::new(load(key));
        let bytes = expert_bytes(&weights);
        if bytes > self.budget {
            return weights; // too big to ever fit; hand back uncached
        }
        self.evict_to(self.budget - bytes);
        self.used += bytes;
        self.map.insert(
            key,
            Entry {
                weights: Rc::clone(&weights),
                bytes,
                pinned: false,
            },
        );
        self.order.push_front(key);
        weights
    }

    /// Warm the cache without returning handles (caller discards). Prefetch hints from the
    /// router's next-k; already-resident experts are skipped (no double load).
    pub fn prefetch<L>(&mut self, keys: &[ExpertKey], mut load: L)
    where
        L: FnMut(ExpertKey) -> ExpertWeights,
    {
        for &k in keys {
            if !self.map.contains_key(&k) {
                self.get_or_load(k, &mut load);
            }
        }
    }

    /// Insert an already-loaded expert (e.g. promoted from disk). Evicts LRU as needed.
    pub fn put(&mut self, key: ExpertKey, weights: ExpertWeights) {
        if let Some(old) = self.map.remove(&key) {
            self.used -= old.bytes;
            self.order.retain(|k| *k != key);
        }
        let bytes = expert_bytes(&weights);
        if bytes > self.budget {
            return;
        }
        self.evict_to(self.budget - bytes);
        self.used += bytes;
        self.map.insert(
            key,
            Entry {
                weights: Rc::new(weights),
                bytes,
                pinned: false,
            },
        );
        self.order.push_front(key);
    }

    /// Pin/unpin an expert so LRU eviction skips it (hot shared experts, layer-0 bias, …).
    pub fn set_pinned(&mut self, key: &ExpertKey, pinned: bool) {
        if let Some(e) = self.map.get_mut(key) {
            e.pinned = pinned;
        }
    }

    /// Drop the least-recently-used unpinned entries until `used <= ceiling`.
    fn evict_to(&mut self, ceiling: usize) {
        while self.used > ceiling {
            match self
                .order
                .iter()
                .rev()
                .copied()
                .find(|k| self.map.get(k).map(|e| !e.pinned).unwrap_or(false))
            {
                Some(victim) => {
                    if let Some(e) = self.map.remove(&victim) {
                        self.used -= e.bytes;
                        self.order.retain(|k| *k != victim);
                        self.stats.evictions += 1;
                    }
                }
                None => break, // everything pinned; can't evict (over budget, but safe)
            }
        }
    }

    /// Move `key` to the MRU position.
    fn touch(&mut self, key: &ExpertKey) {
        self.order.retain(|k| k != key);
        self.order.push_front(*key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One `f32` element each matrix ⇒ `3*4 = 12` bytes per expert.
    fn tiny(tag: f32) -> ExpertWeights {
        ExpertWeights {
            gate: vec![tag],
            up: vec![tag],
            down: vec![tag],
        }
    }
    const E: usize = 12; // bytes for `tiny`

    #[test]
    fn expert_bytes_counts_all_three_matrices() {
        assert_eq!(expert_bytes(&tiny(1.0)), E);
        let big = ExpertWeights {
            gate: vec![0.0; 8],
            up: vec![0.0; 8],
            down: vec![0.0; 8],
        };
        assert_eq!(expert_bytes(&big), 24 * std::mem::size_of::<f32>());
    }

    #[test]
    fn load_then_hit_bumps_stats() {
        let mut c = ExpertCache::new(10 * E);
        let w = c.get_or_load(ExpertKey::new(0, 0), |_| tiny(3.0));
        assert_eq!(w.gate[0], 3.0);
        assert_eq!(c.stats().loads, 1);
        let hit = c.get(&ExpertKey::new(0, 0));
        assert!(hit.is_some());
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.evictions), (1, 1, 0));
    }

    #[test]
    fn evicts_least_recently_used_when_over_budget() {
        // room for exactly 2 experts
        let mut c = ExpertCache::new(2 * E);
        c.get_or_load(ExpertKey::new(0, 0), |_| tiny(0.0));
        c.get_or_load(ExpertKey::new(0, 1), |_| tiny(1.0));
        // touch expert 0 so expert 1 becomes LRU
        assert!(c.get(&ExpertKey::new(0, 0)).is_some());
        c.get_or_load(ExpertKey::new(0, 2), |_| tiny(2.0));
        assert_eq!(c.stats().evictions, 1);
        assert!(c.get(&ExpertKey::new(0, 1)).is_none(), "LRU evicted");
        assert!(c.get(&ExpertKey::new(0, 0)).is_some());
        assert!(c.get(&ExpertKey::new(0, 2)).is_some());
        assert_eq!(c.bytes_used(), 2 * E);
    }

    #[test]
    fn oversized_expert_is_returned_but_not_cached() {
        let mut c = ExpertCache::new(E - 1); // cannot hold a `tiny`
        let w = c.get_or_load(ExpertKey::new(0, 0), |_| tiny(7.0));
        assert_eq!(w.down[0], 7.0);
        assert!(c.is_empty());
        assert_eq!(c.bytes_used(), 0);
    }

    #[test]
    fn pinned_experts_survive_eviction() {
        let mut c = ExpertCache::new(2 * E);
        c.get_or_load(ExpertKey::new(0, 0), |_| tiny(0.0));
        c.set_pinned(&ExpertKey::new(0, 0), true);
        c.get_or_load(ExpertKey::new(0, 1), |_| tiny(1.0));
        // filling a 3rd must evict the unpinned expert 1, never pinned expert 0
        c.get_or_load(ExpertKey::new(0, 2), |_| tiny(2.0));
        assert!(c.get(&ExpertKey::new(0, 0)).is_some(), "pinned kept");
        assert_eq!(c.bytes_used(), 2 * E);
    }

    #[test]
    fn prefetch_warms_without_double_load() {
        let mut c = ExpertCache::new(10 * E);
        let keys = [ExpertKey::new(1, 0), ExpertKey::new(1, 1)];
        c.prefetch(&keys, |_| tiny(5.0));
        assert_eq!(c.stats().loads, 2);
        // second pass over already-resident keys issues no new loads
        c.prefetch(&keys, |_| tiny(5.0));
        assert_eq!(c.stats().loads, 2);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn put_replaces_in_place() {
        let mut c = ExpertCache::new(10 * E);
        c.get_or_load(ExpertKey::new(0, 0), |_| tiny(1.0));
        c.put(ExpertKey::new(0, 0), tiny(9.0));
        let w = c.get(&ExpertKey::new(0, 0)).expect("present");
        assert_eq!(w.gate[0], 9.0);
        assert_eq!(c.bytes_used(), E); // replaced, not doubled
    }
}
