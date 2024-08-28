use std::sync::atomic::{AtomicU64, Ordering};

/// Stats store stores statistics about running miner in memory.
pub struct StatsStore {
    hashes_per_second: AtomicU64,
}

impl StatsStore {
    pub fn new() -> Self {
        Self {
            hashes_per_second: AtomicU64::new(0),
        }
    }

    pub fn update_hashes_per_second(&self, new_value: u64) {
        self.hashes_per_second.store(new_value, Ordering::SeqCst);
    }

    pub fn hashes_per_second(&self) -> u64 {
        self.hashes_per_second.load(Ordering::SeqCst)
    }
}
