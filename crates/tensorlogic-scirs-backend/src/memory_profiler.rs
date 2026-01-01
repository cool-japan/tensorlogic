//! Memory Profiling Utilities for TensorLogic
//!
//! This module provides comprehensive memory profiling capabilities for
//! tracking tensor allocations, detecting memory leaks, and optimizing
//! memory usage in TensorLogic execution.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Memory allocation record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRecord {
    /// Allocation ID
    pub id: usize,

    /// Size in bytes
    pub size_bytes: u64,

    /// Allocation timestamp (relative)
    pub timestamp_ms: u64,

    /// Source location (tensor name or operation)
    pub source: String,

    /// Whether this allocation is still alive
    pub alive: bool,

    /// Duration the allocation was alive (if deallocated)
    pub lifetime_ms: Option<u64>,
}

/// Memory profiler that tracks allocations and deallocations.
#[derive(Clone)]
pub struct MemoryProfiler {
    inner: Arc<Mutex<MemoryProfilerInner>>,
}

struct MemoryProfilerInner {
    /// All allocation records
    allocations: HashMap<usize, AllocationRecord>,

    /// Next allocation ID
    next_id: usize,

    /// Start time for relative timestamps
    start_time: std::time::Instant,

    /// Current memory usage
    current_usage: u64,

    /// Peak memory usage
    peak_usage: u64,

    /// Total allocations
    total_allocations: usize,

    /// Total deallocations
    total_deallocations: usize,
}

impl MemoryProfiler {
    /// Create a new memory profiler.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MemoryProfilerInner {
                allocations: HashMap::new(),
                next_id: 0,
                start_time: std::time::Instant::now(),
                current_usage: 0,
                peak_usage: 0,
                total_allocations: 0,
                total_deallocations: 0,
            })),
        }
    }

    /// Record a tensor allocation.
    pub fn record_allocation(&self, size_bytes: u64, source: String) -> usize {
        let mut inner = self.inner.lock().unwrap();

        let id = inner.next_id;
        inner.next_id += 1;

        let timestamp_ms = inner.start_time.elapsed().as_millis() as u64;

        let record = AllocationRecord {
            id,
            size_bytes,
            timestamp_ms,
            source,
            alive: true,
            lifetime_ms: None,
        };

        inner.allocations.insert(id, record);
        inner.current_usage += size_bytes;
        inner.peak_usage = inner.peak_usage.max(inner.current_usage);
        inner.total_allocations += 1;

        id
    }

    /// Record a tensor deallocation.
    pub fn record_deallocation(&self, id: usize) {
        let mut inner = self.inner.lock().unwrap();

        // Extract timestamp before mutable borrow
        let now = inner.start_time.elapsed().as_millis() as u64;

        if let Some(record) = inner.allocations.get_mut(&id) {
            if record.alive {
                let size_bytes = record.size_bytes; // Copy before mutation
                record.lifetime_ms = Some(now - record.timestamp_ms);
                record.alive = false;

                inner.current_usage = inner.current_usage.saturating_sub(size_bytes);
                inner.total_deallocations += 1;
            }
        }
    }

    /// Get current memory usage in bytes.
    pub fn current_usage(&self) -> u64 {
        self.inner.lock().unwrap().current_usage
    }

    /// Get peak memory usage in bytes.
    pub fn peak_usage(&self) -> u64 {
        self.inner.lock().unwrap().peak_usage
    }

    /// Get memory usage statistics.
    pub fn get_stats(&self) -> MemoryStats {
        let inner = self.inner.lock().unwrap();

        let active_count = inner.allocations.values().filter(|r| r.alive).count();
        let leaked_bytes: u64 = inner
            .allocations
            .values()
            .filter(|r| r.alive)
            .map(|r| r.size_bytes)
            .sum();

        let avg_lifetime_ms = if inner.total_deallocations > 0 {
            let total_lifetime: u64 = inner
                .allocations
                .values()
                .filter_map(|r| r.lifetime_ms)
                .sum();
            total_lifetime / inner.total_deallocations as u64
        } else {
            0
        };

        MemoryStats {
            current_usage_bytes: inner.current_usage,
            peak_usage_bytes: inner.peak_usage,
            total_allocations: inner.total_allocations,
            total_deallocations: inner.total_deallocations,
            active_allocations: active_count,
            leaked_allocations: active_count,
            leaked_bytes,
            avg_allocation_lifetime_ms: avg_lifetime_ms,
        }
    }

    /// Get all allocation records.
    pub fn get_allocations(&self) -> Vec<AllocationRecord> {
        self.inner
            .lock()
            .unwrap()
            .allocations
            .values()
            .cloned()
            .collect()
    }

    /// Get active (not yet deallocated) allocations.
    pub fn get_active_allocations(&self) -> Vec<AllocationRecord> {
        self.inner
            .lock()
            .unwrap()
            .allocations
            .values()
            .filter(|r| r.alive)
            .cloned()
            .collect()
    }

    /// Reset the profiler.
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.allocations.clear();
        inner.next_id = 0;
        inner.start_time = std::time::Instant::now();
        inner.current_usage = 0;
        inner.peak_usage = 0;
        inner.total_allocations = 0;
        inner.total_deallocations = 0;
    }

    /// Export memory timeline to CSV.
    pub fn export_timeline(&self) -> String {
        let inner = self.inner.lock().unwrap();

        let mut csv = String::from("timestamp_ms,event,size_bytes,source\n");

        let mut events: Vec<_> = inner
            .allocations
            .values()
            .flat_map(|r| {
                let mut evs = vec![(r.timestamp_ms, "alloc", r.size_bytes, r.source.clone())];
                if let Some(lifetime) = r.lifetime_ms {
                    evs.push((
                        r.timestamp_ms + lifetime,
                        "dealloc",
                        r.size_bytes,
                        r.source.clone(),
                    ));
                }
                evs
            })
            .collect();

        events.sort_by_key(|(t, _, _, _)| *t);

        for (timestamp, event, size, source) in events {
            csv.push_str(&format!("{},{},{},{}\n", timestamp, event, size, source));
        }

        csv
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage_bytes: u64,

    /// Peak memory usage in bytes
    pub peak_usage_bytes: u64,

    /// Total number of allocations
    pub total_allocations: usize,

    /// Total number of deallocations
    pub total_deallocations: usize,

    /// Number of currently active allocations
    pub active_allocations: usize,

    /// Number of leaked allocations (allocated but not deallocated)
    pub leaked_allocations: usize,

    /// Total bytes leaked
    pub leaked_bytes: u64,

    /// Average allocation lifetime in milliseconds
    pub avg_allocation_lifetime_ms: u64,
}

impl MemoryStats {
    /// Get memory efficiency (deallocated / allocated).
    pub fn memory_efficiency(&self) -> f64 {
        if self.total_allocations == 0 {
            1.0
        } else {
            self.total_deallocations as f64 / self.total_allocations as f64
        }
    }

    /// Get leak rate (leaked / total allocations).
    pub fn leak_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.leaked_allocations as f64 / self.total_allocations as f64
        }
    }

    /// Format memory usage as human-readable string.
    pub fn format_usage(&self) -> String {
        format!(
            "Current: {} | Peak: {} | Active: {} | Leaked: {}",
            Self::format_bytes(self.current_usage_bytes),
            Self::format_bytes(self.peak_usage_bytes),
            self.active_allocations,
            Self::format_bytes(self.leaked_bytes)
        )
    }

    fn format_bytes(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }
}

/// Thread-safe atomic memory counter.
#[derive(Debug)]
pub struct AtomicMemoryCounter {
    current_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    num_allocations: AtomicUsize,
}

impl AtomicMemoryCounter {
    /// Create a new atomic memory counter.
    pub fn new() -> Self {
        Self {
            current_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
            num_allocations: AtomicUsize::new(0),
        }
    }

    /// Record an allocation.
    pub fn allocate(&self, bytes: u64) {
        let current = self.current_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.num_allocations.fetch_add(1, Ordering::Relaxed);

        // Update peak using compare-and-swap loop
        let mut peak = self.peak_bytes.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Record a deallocation.
    pub fn deallocate(&self, bytes: u64) {
        self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Get current usage.
    pub fn current(&self) -> u64 {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Get peak usage.
    pub fn peak(&self) -> u64 {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Get number of allocations.
    pub fn num_allocations(&self) -> usize {
        self.num_allocations.load(Ordering::Relaxed)
    }

    /// Reset the counter.
    pub fn reset(&self) {
        self.current_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
        self.num_allocations.store(0, Ordering::Relaxed);
    }
}

impl Default for AtomicMemoryCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler_basic() {
        let profiler = MemoryProfiler::new();

        let id1 = profiler.record_allocation(1000, "tensor1".to_string());
        assert_eq!(profiler.current_usage(), 1000);
        assert_eq!(profiler.peak_usage(), 1000);

        let id2 = profiler.record_allocation(2000, "tensor2".to_string());
        assert_eq!(profiler.current_usage(), 3000);
        assert_eq!(profiler.peak_usage(), 3000);

        profiler.record_deallocation(id1);
        assert_eq!(profiler.current_usage(), 2000);
        assert_eq!(profiler.peak_usage(), 3000); // Peak doesn't decrease

        profiler.record_deallocation(id2);
        assert_eq!(profiler.current_usage(), 0);
    }

    #[test]
    fn test_memory_stats() {
        let profiler = MemoryProfiler::new();

        profiler.record_allocation(1000, "tensor1".to_string());
        let id2 = profiler.record_allocation(2000, "tensor2".to_string());
        profiler.record_deallocation(id2);

        let stats = profiler.get_stats();

        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.leaked_allocations, 1);
        assert_eq!(stats.leaked_bytes, 1000);
    }

    #[test]
    fn test_memory_efficiency() {
        let stats = MemoryStats {
            current_usage_bytes: 0,
            peak_usage_bytes: 1000,
            total_allocations: 10,
            total_deallocations: 8,
            active_allocations: 2,
            leaked_allocations: 2,
            leaked_bytes: 200,
            avg_allocation_lifetime_ms: 100,
        };

        assert_eq!(stats.memory_efficiency(), 0.8);
        assert_eq!(stats.leak_rate(), 0.2);
    }

    #[test]
    fn test_active_allocations() {
        let profiler = MemoryProfiler::new();

        let id1 = profiler.record_allocation(1000, "tensor1".to_string());
        let _id2 = profiler.record_allocation(2000, "tensor2".to_string());

        assert_eq!(profiler.get_active_allocations().len(), 2);

        profiler.record_deallocation(id1);
        assert_eq!(profiler.get_active_allocations().len(), 1);
    }

    #[test]
    fn test_profiler_reset() {
        let profiler = MemoryProfiler::new();

        profiler.record_allocation(1000, "tensor1".to_string());
        assert_eq!(profiler.current_usage(), 1000);

        profiler.reset();
        assert_eq!(profiler.current_usage(), 0);
        assert_eq!(profiler.peak_usage(), 0);
        assert_eq!(profiler.get_allocations().len(), 0);
    }

    #[test]
    fn test_export_timeline() {
        let profiler = MemoryProfiler::new();

        let id1 = profiler.record_allocation(1000, "tensor1".to_string());
        profiler.record_deallocation(id1);

        let csv = profiler.export_timeline();

        assert!(csv.contains("timestamp_ms,event,size_bytes,source"));
        assert!(csv.contains("alloc"));
        assert!(csv.contains("dealloc"));
    }

    #[test]
    fn test_atomic_memory_counter() {
        let counter = AtomicMemoryCounter::new();

        counter.allocate(1000);
        assert_eq!(counter.current(), 1000);
        assert_eq!(counter.peak(), 1000);
        assert_eq!(counter.num_allocations(), 1);

        counter.allocate(2000);
        assert_eq!(counter.current(), 3000);
        assert_eq!(counter.peak(), 3000);

        counter.deallocate(1000);
        assert_eq!(counter.current(), 2000);
        assert_eq!(counter.peak(), 3000); // Peak doesn't decrease

        counter.reset();
        assert_eq!(counter.current(), 0);
        assert_eq!(counter.peak(), 0);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(MemoryStats::format_bytes(512), "512 B");
        assert_eq!(MemoryStats::format_bytes(1024), "1.00 KB");
        assert_eq!(MemoryStats::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(MemoryStats::format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
