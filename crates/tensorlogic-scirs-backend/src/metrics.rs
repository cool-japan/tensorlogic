//! Comprehensive performance monitoring and metrics collection.
//!
//! This module provides detailed performance monitoring capabilities for tracking
//! execution times, memory usage, throughput, and operation statistics.
//!
//! ## Features
//!
//! - **Execution Time Tracking**: Per-operation timing with statistics
//! - **Memory Usage Monitoring**: Track allocations, peak usage, and memory efficiency
//! - **Throughput Measurement**: Operations per second, elements processed
//! - **Telemetry Export**: JSON and CSV export for analysis
//! - **Operation Profiling**: Detailed per-operation statistics
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_scirs_backend::metrics::{MetricsCollector, MetricsConfig};
//!
//! let mut metrics = MetricsCollector::new(MetricsConfig::default());
//!
//! // Track operation execution
//! let result = metrics.time_operation("einsum:ij,jk->ik", || {
//!     // Perform operation
//!     42
//! });
//!
//! // Record memory allocation
//! metrics.record_allocation(1024);
//!
//! // Get summary
//! let summary = metrics.summary();
//! println!("{}", summary);
//!
//! // Export to JSON
//! let json = metrics.export_json().unwrap();
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Configuration for metrics collection.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Enable detailed per-operation timing
    pub detailed_timing: bool,

    /// Enable memory tracking
    pub track_memory: bool,

    /// Enable throughput calculation
    pub track_throughput: bool,

    /// Maximum number of operation records to keep
    pub max_records: usize,

    /// Sampling rate for high-frequency operations (1.0 = all, 0.1 = 10%)
    pub sampling_rate: f64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            detailed_timing: true,
            track_memory: true,
            track_throughput: true,
            max_records: 10000,
            sampling_rate: 1.0,
        }
    }
}

impl MetricsConfig {
    /// Create a lightweight configuration for production.
    pub fn lightweight() -> Self {
        Self {
            detailed_timing: false,
            track_memory: false,
            track_throughput: true,
            max_records: 1000,
            sampling_rate: 0.1,
        }
    }

    /// Create a comprehensive configuration for debugging.
    pub fn debug() -> Self {
        Self {
            detailed_timing: true,
            track_memory: true,
            track_throughput: true,
            max_records: 50000,
            sampling_rate: 1.0,
        }
    }
}

/// Record of a single operation execution.
#[derive(Debug, Clone)]
pub struct OperationRecord {
    /// Operation name/type
    pub operation: String,

    /// Execution duration
    pub duration: Duration,

    /// Timestamp when operation started
    pub timestamp: u64,

    /// Memory allocated during operation (bytes)
    pub memory_allocated: usize,

    /// Number of elements processed
    pub elements_processed: usize,
}

/// Statistics for a specific operation type.
#[derive(Debug, Clone, Default)]
pub struct OperationStats {
    /// Total number of calls
    pub call_count: u64,

    /// Total execution time
    pub total_time: Duration,

    /// Minimum execution time
    pub min_time: Duration,

    /// Maximum execution time
    pub max_time: Duration,

    /// Total elements processed
    pub total_elements: u64,

    /// Total memory allocated
    pub total_memory: u64,
}

impl OperationStats {
    /// Get average execution time.
    pub fn avg_time(&self) -> Duration {
        if self.call_count == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.call_count as u32
        }
    }

    /// Get throughput in operations per second.
    pub fn ops_per_sec(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs == 0.0 {
            0.0
        } else {
            self.call_count as f64 / secs
        }
    }

    /// Get elements per second throughput.
    pub fn elements_per_sec(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs == 0.0 {
            0.0
        } else {
            self.total_elements as f64 / secs
        }
    }

    /// Update stats with a new operation record.
    pub fn update(&mut self, duration: Duration, elements: usize, memory: usize) {
        if self.call_count == 0 {
            self.min_time = duration;
            self.max_time = duration;
        } else {
            self.min_time = self.min_time.min(duration);
            self.max_time = self.max_time.max(duration);
        }
        self.call_count += 1;
        self.total_time += duration;
        self.total_elements += elements as u64;
        self.total_memory += memory as u64;
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Current memory usage (bytes)
    pub current_usage: usize,

    /// Peak memory usage (bytes)
    pub peak_usage: usize,

    /// Total allocations
    pub total_allocations: u64,

    /// Total deallocations
    pub total_deallocations: u64,

    /// Total bytes allocated
    pub total_bytes_allocated: u64,

    /// Total bytes deallocated
    pub total_bytes_deallocated: u64,
}

impl MemoryStats {
    /// Record an allocation.
    pub fn record_allocation(&mut self, bytes: usize) {
        self.current_usage = self.current_usage.saturating_add(bytes);
        self.peak_usage = self.peak_usage.max(self.current_usage);
        self.total_allocations += 1;
        self.total_bytes_allocated += bytes as u64;
    }

    /// Record a deallocation.
    pub fn record_deallocation(&mut self, bytes: usize) {
        self.current_usage = self.current_usage.saturating_sub(bytes);
        self.total_deallocations += 1;
        self.total_bytes_deallocated += bytes as u64;
    }

    /// Get memory efficiency (deallocated / allocated).
    pub fn efficiency(&self) -> f64 {
        if self.total_bytes_allocated == 0 {
            1.0
        } else {
            self.total_bytes_deallocated as f64 / self.total_bytes_allocated as f64
        }
    }

    /// Format current usage in human-readable form.
    pub fn format_current(&self) -> String {
        format_bytes(self.current_usage)
    }

    /// Format peak usage in human-readable form.
    pub fn format_peak(&self) -> String {
        format_bytes(self.peak_usage)
    }
}

/// Global throughput metrics.
#[derive(Debug, Clone, Default)]
pub struct ThroughputStats {
    /// Total operations executed
    pub total_operations: u64,

    /// Total execution time
    pub total_time: Duration,

    /// Total elements processed
    pub total_elements: u64,

    /// Start time of metrics collection
    pub start_time: Option<Instant>,
}

impl ThroughputStats {
    /// Get operations per second.
    pub fn ops_per_sec(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs == 0.0 {
            0.0
        } else {
            self.total_operations as f64 / secs
        }
    }

    /// Get elements per second.
    pub fn elements_per_sec(&self) -> f64 {
        let secs = self.total_time.as_secs_f64();
        if secs == 0.0 {
            0.0
        } else {
            self.total_elements as f64 / secs
        }
    }

    /// Get wall clock throughput (ops/sec since start).
    pub fn wall_clock_ops_per_sec(&self) -> f64 {
        match self.start_time {
            Some(start) => {
                let elapsed = start.elapsed().as_secs_f64();
                if elapsed == 0.0 {
                    0.0
                } else {
                    self.total_operations as f64 / elapsed
                }
            }
            None => 0.0,
        }
    }
}

/// Comprehensive metrics collector.
#[derive(Debug)]
pub struct MetricsCollector {
    /// Configuration
    config: MetricsConfig,

    /// Per-operation statistics
    operation_stats: HashMap<String, OperationStats>,

    /// Memory statistics
    memory_stats: MemoryStats,

    /// Throughput statistics
    throughput_stats: ThroughputStats,

    /// Detailed operation records (if enabled)
    records: Vec<OperationRecord>,

    /// Collection start time
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            config,
            operation_stats: HashMap::new(),
            memory_stats: MemoryStats::default(),
            throughput_stats: ThroughputStats {
                start_time: Some(Instant::now()),
                ..Default::default()
            },
            records: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Time an operation and record metrics.
    pub fn time_operation<F, T>(&mut self, operation: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        self.time_operation_with_elements(operation, 0, f)
    }

    /// Time an operation with element count tracking.
    pub fn time_operation_with_elements<F, T>(
        &mut self,
        operation: &str,
        elements: usize,
        f: F,
    ) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        self.record_operation(operation, duration, elements, 0);

        result
    }

    /// Record an operation execution.
    pub fn record_operation(
        &mut self,
        operation: &str,
        duration: Duration,
        elements: usize,
        memory: usize,
    ) {
        // Update operation stats
        let stats = self
            .operation_stats
            .entry(operation.to_string())
            .or_default();
        stats.update(duration, elements, memory);

        // Update throughput stats
        self.throughput_stats.total_operations += 1;
        self.throughput_stats.total_time += duration;
        self.throughput_stats.total_elements += elements as u64;

        // Store detailed record if enabled
        if self.config.detailed_timing && self.records.len() < self.config.max_records {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            self.records.push(OperationRecord {
                operation: operation.to_string(),
                duration,
                timestamp,
                memory_allocated: memory,
                elements_processed: elements,
            });
        }
    }

    /// Record a memory allocation.
    pub fn record_allocation(&mut self, bytes: usize) {
        if self.config.track_memory {
            self.memory_stats.record_allocation(bytes);
        }
    }

    /// Record a memory deallocation.
    pub fn record_deallocation(&mut self, bytes: usize) {
        if self.config.track_memory {
            self.memory_stats.record_deallocation(bytes);
        }
    }

    /// Get operation statistics.
    pub fn operation_stats(&self) -> &HashMap<String, OperationStats> {
        &self.operation_stats
    }

    /// Get memory statistics.
    pub fn memory_stats(&self) -> &MemoryStats {
        &self.memory_stats
    }

    /// Get throughput statistics.
    pub fn throughput_stats(&self) -> &ThroughputStats {
        &self.throughput_stats
    }

    /// Get detailed operation records.
    pub fn records(&self) -> &[OperationRecord] {
        &self.records
    }

    /// Get a summary of all metrics.
    pub fn summary(&self) -> MetricsSummary {
        let mut slowest_ops: Vec<_> = self.operation_stats.iter().collect();
        slowest_ops.sort_by(|a, b| b.1.avg_time().cmp(&a.1.avg_time()));

        let mut most_called: Vec<_> = self.operation_stats.iter().collect();
        most_called.sort_by(|a, b| b.1.call_count.cmp(&a.1.call_count));

        MetricsSummary {
            total_operations: self.throughput_stats.total_operations,
            total_time: self.throughput_stats.total_time,
            ops_per_sec: self.throughput_stats.ops_per_sec(),
            elements_per_sec: self.throughput_stats.elements_per_sec(),
            peak_memory: self.memory_stats.peak_usage,
            current_memory: self.memory_stats.current_usage,
            memory_efficiency: self.memory_stats.efficiency(),
            unique_operations: self.operation_stats.len(),
            slowest_operation: slowest_ops
                .first()
                .map(|(k, v)| ((*k).clone(), v.avg_time())),
            most_called_operation: most_called
                .first()
                .map(|(k, v)| ((*k).clone(), v.call_count)),
            collection_duration: self.start_time.elapsed(),
        }
    }

    /// Export metrics to JSON format.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let export = MetricsExport {
            summary: self.summary(),
            operation_stats: self
                .operation_stats
                .iter()
                .map(|(k, v)| (k.clone(), OperationStatsExport::from(v)))
                .collect(),
            memory_stats: MemoryStatsExport::from(&self.memory_stats),
            throughput_stats: ThroughputStatsExport::from(&self.throughput_stats),
        };

        serde_json::to_string_pretty(&export)
    }

    /// Export metrics to CSV format (operation stats only).
    pub fn export_csv(&self) -> String {
        let mut csv = String::from(
            "operation,call_count,total_time_ms,avg_time_us,min_time_us,max_time_us,ops_per_sec,elements_per_sec\n",
        );

        for (name, stats) in &self.operation_stats {
            csv.push_str(&format!(
                "{},{},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2}\n",
                name,
                stats.call_count,
                stats.total_time.as_secs_f64() * 1000.0,
                stats.avg_time().as_secs_f64() * 1_000_000.0,
                stats.min_time.as_secs_f64() * 1_000_000.0,
                stats.max_time.as_secs_f64() * 1_000_000.0,
                stats.ops_per_sec(),
                stats.elements_per_sec(),
            ));
        }

        csv
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        self.operation_stats.clear();
        self.memory_stats = MemoryStats::default();
        self.throughput_stats = ThroughputStats {
            start_time: Some(Instant::now()),
            ..Default::default()
        };
        self.records.clear();
        self.start_time = Instant::now();
    }

    /// Get elapsed time since collection started.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new(MetricsConfig::default())
    }
}

/// Summary of collected metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricsSummary {
    /// Total operations executed
    pub total_operations: u64,

    /// Total execution time
    #[serde(with = "duration_serde")]
    pub total_time: Duration,

    /// Operations per second
    pub ops_per_sec: f64,

    /// Elements per second
    pub elements_per_sec: f64,

    /// Peak memory usage (bytes)
    pub peak_memory: usize,

    /// Current memory usage (bytes)
    pub current_memory: usize,

    /// Memory efficiency (0.0 - 1.0)
    pub memory_efficiency: f64,

    /// Number of unique operation types
    pub unique_operations: usize,

    /// Slowest operation (name, avg time)
    pub slowest_operation: Option<(String, Duration)>,

    /// Most called operation (name, count)
    pub most_called_operation: Option<(String, u64)>,

    /// Total collection duration
    #[serde(with = "duration_serde")]
    pub collection_duration: Duration,
}

impl std::fmt::Display for MetricsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Metrics Summary ===")?;
        writeln!(f, "Total Operations: {}", self.total_operations)?;
        writeln!(f, "Total Time: {:.3}s", self.total_time.as_secs_f64())?;
        writeln!(f, "Throughput: {:.2} ops/sec", self.ops_per_sec)?;
        writeln!(f, "Elements: {:.2}/sec", self.elements_per_sec)?;
        writeln!(f, "Peak Memory: {}", format_bytes(self.peak_memory))?;
        writeln!(f, "Current Memory: {}", format_bytes(self.current_memory))?;
        writeln!(
            f,
            "Memory Efficiency: {:.1}%",
            self.memory_efficiency * 100.0
        )?;
        writeln!(f, "Unique Operations: {}", self.unique_operations)?;

        if let Some((name, time)) = &self.slowest_operation {
            writeln!(
                f,
                "Slowest: {} ({:.3}ms)",
                name,
                time.as_secs_f64() * 1000.0
            )?;
        }

        if let Some((name, count)) = &self.most_called_operation {
            writeln!(f, "Most Called: {} ({} calls)", name, count)?;
        }

        Ok(())
    }
}

/// Thread-safe atomic metrics counter.
#[derive(Debug, Default)]
pub struct AtomicMetrics {
    /// Total operations
    pub operations: AtomicU64,

    /// Total time in nanoseconds
    pub time_nanos: AtomicU64,

    /// Peak memory
    pub peak_memory: AtomicUsize,

    /// Current memory
    pub current_memory: AtomicUsize,
}

impl AtomicMetrics {
    /// Create new atomic metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an operation.
    pub fn record_operation(&self, duration: Duration) {
        self.operations.fetch_add(1, Ordering::Relaxed);
        self.time_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record memory allocation.
    pub fn record_allocation(&self, bytes: usize) {
        let current = self.current_memory.fetch_add(bytes, Ordering::Relaxed) + bytes;
        let mut peak = self.peak_memory.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_memory.compare_exchange_weak(
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

    /// Record memory deallocation.
    pub fn record_deallocation(&self, bytes: usize) {
        self.current_memory.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Get current statistics snapshot.
    pub fn snapshot(&self) -> (u64, Duration, usize) {
        let ops = self.operations.load(Ordering::Relaxed);
        let nanos = self.time_nanos.load(Ordering::Relaxed);
        let peak = self.peak_memory.load(Ordering::Relaxed);
        (ops, Duration::from_nanos(nanos), peak)
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.operations.store(0, Ordering::Relaxed);
        self.time_nanos.store(0, Ordering::Relaxed);
        self.peak_memory.store(0, Ordering::Relaxed);
        self.current_memory.store(0, Ordering::Relaxed);
    }
}

/// Shareable atomic metrics wrapper.
pub type SharedMetrics = Arc<AtomicMetrics>;

/// Create a new shared metrics instance.
pub fn shared_metrics() -> SharedMetrics {
    Arc::new(AtomicMetrics::new())
}

// Export structures for serialization

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MetricsExport {
    summary: MetricsSummary,
    operation_stats: HashMap<String, OperationStatsExport>,
    memory_stats: MemoryStatsExport,
    throughput_stats: ThroughputStatsExport,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OperationStatsExport {
    call_count: u64,
    total_time_ms: f64,
    avg_time_us: f64,
    min_time_us: f64,
    max_time_us: f64,
    ops_per_sec: f64,
    elements_per_sec: f64,
    total_elements: u64,
    total_memory_bytes: u64,
}

impl From<&OperationStats> for OperationStatsExport {
    fn from(stats: &OperationStats) -> Self {
        Self {
            call_count: stats.call_count,
            total_time_ms: stats.total_time.as_secs_f64() * 1000.0,
            avg_time_us: stats.avg_time().as_secs_f64() * 1_000_000.0,
            min_time_us: stats.min_time.as_secs_f64() * 1_000_000.0,
            max_time_us: stats.max_time.as_secs_f64() * 1_000_000.0,
            ops_per_sec: stats.ops_per_sec(),
            elements_per_sec: stats.elements_per_sec(),
            total_elements: stats.total_elements,
            total_memory_bytes: stats.total_memory,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MemoryStatsExport {
    current_usage_bytes: usize,
    peak_usage_bytes: usize,
    total_allocations: u64,
    total_deallocations: u64,
    total_bytes_allocated: u64,
    total_bytes_deallocated: u64,
    efficiency: f64,
}

impl From<&MemoryStats> for MemoryStatsExport {
    fn from(stats: &MemoryStats) -> Self {
        Self {
            current_usage_bytes: stats.current_usage,
            peak_usage_bytes: stats.peak_usage,
            total_allocations: stats.total_allocations,
            total_deallocations: stats.total_deallocations,
            total_bytes_allocated: stats.total_bytes_allocated,
            total_bytes_deallocated: stats.total_bytes_deallocated,
            efficiency: stats.efficiency(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ThroughputStatsExport {
    total_operations: u64,
    total_time_ms: f64,
    total_elements: u64,
    ops_per_sec: f64,
    elements_per_sec: f64,
}

impl From<&ThroughputStats> for ThroughputStatsExport {
    fn from(stats: &ThroughputStats) -> Self {
        Self {
            total_operations: stats.total_operations,
            total_time_ms: stats.total_time.as_secs_f64() * 1000.0,
            total_elements: stats.total_elements,
            ops_per_sec: stats.ops_per_sec(),
            elements_per_sec: stats.elements_per_sec(),
        }
    }
}

// Serde helper for Duration
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

/// Format bytes in human-readable form.
pub fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();
        assert!(config.detailed_timing);
        assert!(config.track_memory);
        assert!(config.track_throughput);
        assert_eq!(config.max_records, 10000);
        assert_eq!(config.sampling_rate, 1.0);
    }

    #[test]
    fn test_metrics_config_lightweight() {
        let config = MetricsConfig::lightweight();
        assert!(!config.detailed_timing);
        assert!(!config.track_memory);
        assert!(config.track_throughput);
        assert_eq!(config.sampling_rate, 0.1);
    }

    #[test]
    fn test_operation_stats_update() {
        let mut stats = OperationStats::default();

        stats.update(Duration::from_millis(10), 100, 1024);
        assert_eq!(stats.call_count, 1);
        assert_eq!(stats.total_elements, 100);
        assert_eq!(stats.min_time, Duration::from_millis(10));
        assert_eq!(stats.max_time, Duration::from_millis(10));

        stats.update(Duration::from_millis(20), 200, 2048);
        assert_eq!(stats.call_count, 2);
        assert_eq!(stats.total_elements, 300);
        assert_eq!(stats.min_time, Duration::from_millis(10));
        assert_eq!(stats.max_time, Duration::from_millis(20));
    }

    #[test]
    fn test_operation_stats_throughput() {
        let mut stats = OperationStats::default();
        stats.update(Duration::from_secs(1), 1000, 0);
        stats.update(Duration::from_secs(1), 1000, 0);

        assert_eq!(stats.ops_per_sec(), 1.0);
        assert_eq!(stats.elements_per_sec(), 1000.0);
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::default();

        stats.record_allocation(1024);
        assert_eq!(stats.current_usage, 1024);
        assert_eq!(stats.peak_usage, 1024);

        stats.record_allocation(2048);
        assert_eq!(stats.current_usage, 3072);
        assert_eq!(stats.peak_usage, 3072);

        stats.record_deallocation(1024);
        assert_eq!(stats.current_usage, 2048);
        assert_eq!(stats.peak_usage, 3072);
    }

    #[test]
    fn test_memory_stats_efficiency() {
        let stats = MemoryStats {
            total_bytes_allocated: 1000,
            total_bytes_deallocated: 800,
            ..Default::default()
        };
        assert!((stats.efficiency() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_metrics_collector_basic() {
        let mut collector = MetricsCollector::new(MetricsConfig::default());

        let result = collector.time_operation("test_op", || 42);
        assert_eq!(result, 42);

        let stats = collector.operation_stats();
        assert!(stats.contains_key("test_op"));
        assert_eq!(stats["test_op"].call_count, 1);
    }

    #[test]
    fn test_metrics_collector_with_elements() {
        let mut collector = MetricsCollector::new(MetricsConfig::default());

        collector.time_operation_with_elements("einsum", 1000, || {
            thread::sleep(Duration::from_micros(100));
        });

        let stats = collector.operation_stats();
        assert_eq!(stats["einsum"].total_elements, 1000);
    }

    #[test]
    fn test_metrics_collector_memory() {
        let mut collector = MetricsCollector::new(MetricsConfig::default());

        collector.record_allocation(1024);
        collector.record_allocation(2048);
        collector.record_deallocation(1024);

        let memory = collector.memory_stats();
        assert_eq!(memory.current_usage, 2048);
        assert_eq!(memory.peak_usage, 3072);
    }

    #[test]
    fn test_metrics_collector_summary() {
        let mut collector = MetricsCollector::new(MetricsConfig::default());

        for i in 0..5 {
            collector.time_operation(&format!("op_{}", i), || {
                thread::sleep(Duration::from_micros(10));
            });
        }

        let summary = collector.summary();
        assert_eq!(summary.total_operations, 5);
        assert_eq!(summary.unique_operations, 5);
    }

    #[test]
    fn test_metrics_collector_export_json() {
        let mut collector = MetricsCollector::new(MetricsConfig::default());
        collector.time_operation("test", || {});

        let json = collector.export_json().unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("total_operations"));
    }

    #[test]
    fn test_metrics_collector_export_csv() {
        let mut collector = MetricsCollector::new(MetricsConfig::default());
        collector.time_operation("op1", || {});
        collector.time_operation("op2", || {});

        let csv = collector.export_csv();
        assert!(csv.contains("op1"));
        assert!(csv.contains("op2"));
        assert!(csv.contains("operation,call_count"));
    }

    #[test]
    fn test_metrics_collector_reset() {
        let mut collector = MetricsCollector::new(MetricsConfig::default());
        collector.time_operation("test", || {});

        assert!(!collector.operation_stats().is_empty());
        collector.reset();
        assert!(collector.operation_stats().is_empty());
    }

    #[test]
    fn test_atomic_metrics() {
        let metrics = AtomicMetrics::new();

        metrics.record_operation(Duration::from_millis(10));
        metrics.record_operation(Duration::from_millis(20));
        metrics.record_allocation(1024);

        let (ops, time, peak) = metrics.snapshot();
        assert_eq!(ops, 2);
        assert_eq!(time, Duration::from_millis(30));
        assert_eq!(peak, 1024);
    }

    #[test]
    fn test_shared_metrics() {
        let metrics = shared_metrics();
        let metrics_clone = Arc::clone(&metrics);

        metrics.record_operation(Duration::from_millis(10));
        metrics_clone.record_operation(Duration::from_millis(20));

        let (ops, _, _) = metrics.snapshot();
        assert_eq!(ops, 2);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1572864), "1.50 MB");
        assert_eq!(format_bytes(1610612736), "1.50 GB");
    }

    #[test]
    fn test_metrics_summary_display() {
        let mut collector = MetricsCollector::new(MetricsConfig::default());
        collector.time_operation("test", || {});

        let summary = collector.summary();
        let display = format!("{}", summary);
        assert!(display.contains("Metrics Summary"));
        assert!(display.contains("Total Operations"));
    }

    #[test]
    fn test_detailed_records() {
        let mut collector = MetricsCollector::new(MetricsConfig::default());

        collector.time_operation("op1", || {});
        collector.time_operation("op2", || {});

        let records = collector.records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].operation, "op1");
        assert_eq!(records[1].operation, "op2");
    }

    #[test]
    fn test_max_records_limit() {
        let config = MetricsConfig {
            max_records: 2,
            ..Default::default()
        };
        let mut collector = MetricsCollector::new(config);

        collector.time_operation("op1", || {});
        collector.time_operation("op2", || {});
        collector.time_operation("op3", || {});

        assert_eq!(collector.records().len(), 2);
    }
}
