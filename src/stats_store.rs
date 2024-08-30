use std::sync::atomic::{AtomicU64, Ordering};

/// Stats store stores statistics about running miner in memory.
pub struct StatsStore {
    hashes_per_second: AtomicU64,
    accepted_blocks: AtomicU64,
    rejected_blocks: AtomicU64,
}

impl StatsStore {
    pub fn new() -> Self {
        Self {
            hashes_per_second: AtomicU64::new(0),
            accepted_blocks: AtomicU64::new(0),
            rejected_blocks: AtomicU64::new(0),
        }
    }

    pub fn update_hashes_per_second(&self, new_value: u64) {
        self.hashes_per_second.store(new_value, Ordering::SeqCst);
    }

    pub fn inc_accepted_blocks(&self) {
        self.accepted_blocks.fetch_add(1, Ordering::SeqCst);
    }

    pub fn inc_rejected_blocks(&self) {
        self.accepted_blocks.fetch_add(1, Ordering::SeqCst);
    }

    pub fn hashes_per_second(&self) -> u64 {
        self.hashes_per_second.load(Ordering::SeqCst)
    }

    pub fn accepted_blocks(&self) -> u64 {
        self.accepted_blocks.load(Ordering::SeqCst)
    }

    pub fn rejected_blocks(&self) -> u64 {
        self.rejected_blocks.load(Ordering::SeqCst)
    }
}
