//! Multi-slot table + prompt-reuse — band B5 `PH-503`.
//!
//! llama.cpp/vLLM style **slots**: a bounded set of concurrent conversation contexts, each
//! identified by a `u64` slot id and owning its generated token history. A single-threaded
//! `serve` loop maps an incoming request to a slot (its KV cache would live in a
//! [`crate::runner::DenseDecoder`] per slot; here the table owns the canonical token
//! sequence so the same prompt can be resumed without re-tokenizing). Eviction is
//! least-recently-used, mirroring [`crate::cache::ExpertCache`], so long-lived servers never
//! grow unbounded. `std`-only, deterministic, no timers/threads.

use std::collections::{HashMap, VecDeque};

/// One conversation slot: the token ids produced so far (prompt prefix + generated tail).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    pub id: u64,
    pub history: Vec<u32>,
}

impl Slot {
    pub fn new(id: u64, prompt: &[u32]) -> Self {
        Self {
            id,
            history: prompt.to_vec(),
        }
    }
}

/// Byte-free **capacity-budgeted** LRU table of [`Slot`]s. `open` reuses an existing slot
/// (returns its history) or inserts a new one, evicting the least-recently-used past `cap`.
pub struct SlotTable {
    capacity: usize,
    /// MRU at the front; evict from the back.
    order: VecDeque<u64>,
    slots: HashMap<u64, Slot>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl SlotTable {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: VecDeque::new(),
            slots: HashMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Stats (hit = reopened an existing slot; miss = opened a new one).
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.evictions)
    }

    fn touch(&mut self, id: u64) {
        self.order.retain(|&x| x != id);
        self.order.push_front(id);
    }

    /// Look up a slot's history (also marks it MRU). `None` if absent.
    pub fn get(&mut self, id: u64) -> Option<&[u32]> {
        if self.slots.contains_key(&id) {
            self.touch(id);
            self.hits += 1;
            self.slots.get(&id).map(|s| s.history.as_slice())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Return the slot's current history, creating it from `prompt` on a miss (LRU-evicting
    /// when full). Existing slots are **not** overwritten — the prompt is only the seed for
    /// a fresh slot; use [`append`](Self::append) to extend a live one.
    pub fn open(&mut self, id: u64, prompt: &[u32]) -> &[u32] {
        if self.slots.contains_key(&id) {
            self.touch(id);
            self.hits += 1;
        } else {
            self.misses += 1;
            while self.slots.len() >= self.capacity {
                if let Some(victim) = self.order.pop_back() {
                    self.slots.remove(&victim);
                    self.evictions += 1;
                } else {
                    break;
                }
            }
            self.slots.insert(id, Slot::new(id, prompt));
            self.order.push_front(id);
        }
        self.slots.get(&id).map(|s| s.history.as_slice()).unwrap()
    }

    /// Append a generated token to an existing slot. Returns `false` for unknown ids.
    pub fn append(&mut self, id: u64, tok: u32) -> bool {
        match self.slots.get_mut(&id) {
            Some(s) => {
                s.history.push(tok);
                self.touch(id);
                true
            }
            None => false,
        }
    }

    /// `true` when the exact `prompt` tokens are already the prefix of a live slot —
    /// the serve loop then skips re-tokenizing and resumes decode from `history.len()`.
    pub fn prompt_cached(&mut self, prompt: &[u32]) -> Option<u64> {
        let hit = self
            .slots
            .iter()
            .find(|(_, s)| s.history.len() >= prompt.len() && s.history[..prompt.len()] == *prompt)
            .map(|(id, _)| *id);
        match hit {
            Some(id) => {
                self.touch(id);
                self.hits += 1;
                Some(id)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Drop a slot explicitly (e.g. client closed the conversation).
    pub fn close(&mut self, id: u64) -> bool {
        let removed = self.slots.remove(&id).is_some();
        if removed {
            self.order.retain(|&x| x != id);
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_existing_does_not_reset_history() {
        let mut t = SlotTable::new(2);
        t.open(1, &[10, 20]);
        t.append(1, 30);
        // reopen must not overwrite the live [10,20,30] with a fresh prompt.
        assert_eq!(t.open(1, &[99]), &[10, 20, 30][..]);
    }

    #[test]
    fn evicts_least_recently_used() {
        let mut t = SlotTable::new(2);
        t.open(1, &[1]);
        t.open(2, &[2]);
        t.get(1); // 1 is now MRU, 2 becomes LRU
        t.open(3, &[3]); // must evict 2
        assert_eq!(t.stats().2, 1, "one eviction");
        assert!(t.get(1).is_some());
        assert!(t.get(3).is_some());
        assert!(t.get(2).is_none(), "LRU evicted");
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn append_unknown_is_false() {
        let mut t = SlotTable::new(1);
        assert!(!t.append(42, 7));
    }

    #[test]
    fn prompt_cache_matches_live_prefix() {
        let mut t = SlotTable::new(3);
        t.open(7, &[5, 6, 7]);
        assert_eq!(t.prompt_cached(&[5, 6]), Some(7), "prompt is a live prefix");
        assert_eq!(t.prompt_cached(&[5, 6, 7]), Some(7), "exact match");
        assert_eq!(t.prompt_cached(&[9]), None, "no slot starts with 9");
    }

    #[test]
    fn close_removes_slot() {
        let mut t = SlotTable::new(2);
        t.open(1, &[0]);
        assert!(t.close(1));
        assert!(!t.close(1));
        assert!(t.is_empty());
    }

    #[test]
    fn capacity_is_at_least_one() {
        let t = SlotTable::new(0);
        assert_eq!(t.capacity(), 1);
    }
}
