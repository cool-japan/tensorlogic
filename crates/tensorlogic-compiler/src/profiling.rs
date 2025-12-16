//! Compilation profiling and performance tracking.
//!
//! This module provides tools for profiling the compilation process,
//! tracking performance metrics, and identifying bottlenecks.
//!
//! # Overview
//!
//! Compilation profiling helps developers:
//! - Identify slow compilation passes
//! - Track memory usage during compilation
//! - Optimize compilation performance
//! - Compare different compilation strategies
//!
//! # Features
//!
//! - **Time Tracking**: Measure time spent in each compilation phase
//! - **Memory Tracking**: Monitor memory allocations and peak usage
//! - **Pass Analysis**: Identify expensive optimization passes
//! - **Cache Statistics**: Track cache hit rates and effectiveness
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_compiler::profiling::{CompilationProfiler, ProfileConfig};
//! use tensorlogic_compiler::compile_to_einsum;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let mut profiler = CompilationProfiler::new();
//! profiler.start_phase("compilation");
//!
//! let expr = TLExpr::pred("p", vec![Term::var("x")]);
//! let _graph = compile_to_einsum(&expr).unwrap();
//!
//! profiler.end_phase("compilation");
//!
//! let report = profiler.generate_report();
//! println!("{}", report);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for compilation profiling.
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Enable time tracking
    pub track_time: bool,
    /// Enable memory tracking
    pub track_memory: bool,
    /// Enable detailed pass-level profiling
    pub track_passes: bool,
    /// Enable cache statistics
    pub track_cache: bool,
    /// Minimum duration to report (filter noise)
    pub min_duration_ms: u64,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            track_time: true,
            track_memory: true,
            track_passes: true,
            track_cache: true,
            min_duration_ms: 1,
        }
    }
}

/// Time spent in a compilation phase.
#[derive(Debug, Clone)]
pub struct PhaseTime {
    /// Phase name
    pub name: String,
    /// Total duration
    pub duration: Duration,
    /// Number of times this phase was executed
    pub count: usize,
    /// Child phases
    pub children: Vec<PhaseTime>,
}

impl PhaseTime {
    /// Create a new phase time entry.
    pub fn new(name: String, duration: Duration) -> Self {
        Self {
            name,
            duration,
            count: 1,
            children: Vec::new(),
        }
    }

    /// Get average duration per execution.
    pub fn average_duration(&self) -> Duration {
        if self.count == 0 {
            Duration::from_secs(0)
        } else {
            self.duration / self.count as u32
        }
    }

    /// Get total time including children.
    pub fn total_time_with_children(&self) -> Duration {
        let mut total = self.duration;
        for child in &self.children {
            total += child.total_time_with_children();
        }
        total
    }
}

/// Memory usage snapshot.
#[derive(Debug, Clone, Default)]
pub struct MemorySnapshot {
    /// Timestamp of snapshot
    pub timestamp: Option<Instant>,
    /// Estimated heap usage in bytes
    pub heap_bytes: usize,
    /// Number of active allocations
    pub allocation_count: usize,
}

impl MemorySnapshot {
    /// Create a new memory snapshot.
    pub fn new() -> Self {
        Self {
            timestamp: Some(Instant::now()),
            heap_bytes: 0,
            allocation_count: 0,
        }
    }

    /// Record an allocation.
    pub fn record_allocation(&mut self, size: usize) {
        self.heap_bytes += size;
        self.allocation_count += 1;
    }

    /// Record a deallocation.
    pub fn record_deallocation(&mut self, size: usize) {
        self.heap_bytes = self.heap_bytes.saturating_sub(size);
        self.allocation_count = self.allocation_count.saturating_sub(1);
    }
}

/// Pass-level profiling information.
#[derive(Debug, Clone)]
pub struct PassProfile {
    /// Pass name
    pub name: String,
    /// Number of times executed
    pub execution_count: usize,
    /// Total time spent
    pub total_time: Duration,
    /// Number of optimizations applied
    pub optimizations_applied: usize,
    /// Memory allocated during pass
    pub memory_allocated: usize,
}

impl PassProfile {
    /// Create a new pass profile.
    pub fn new(name: String) -> Self {
        Self {
            name,
            execution_count: 0,
            total_time: Duration::from_secs(0),
            optimizations_applied: 0,
            memory_allocated: 0,
        }
    }

    /// Record an execution of this pass.
    pub fn record_execution(&mut self, duration: Duration, optimizations: usize) {
        self.execution_count += 1;
        self.total_time += duration;
        self.optimizations_applied += optimizations;
    }

    /// Get average time per execution.
    pub fn average_time(&self) -> Duration {
        if self.execution_count == 0 {
            Duration::from_secs(0)
        } else {
            self.total_time / self.execution_count as u32
        }
    }

    /// Get optimizations per execution.
    pub fn optimizations_per_execution(&self) -> f64 {
        if self.execution_count == 0 {
            0.0
        } else {
            self.optimizations_applied as f64 / self.execution_count as f64
        }
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache lookups
    pub lookups: usize,
    /// Cache hits
    pub hits: usize,
    /// Cache misses
    pub misses: usize,
    /// Cache evictions
    pub evictions: usize,
}

impl CacheStats {
    /// Calculate hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            (self.hits as f64 / self.lookups as f64) * 100.0
        }
    }

    /// Calculate miss rate as a percentage.
    pub fn miss_rate(&self) -> f64 {
        100.0 - self.hit_rate()
    }

    /// Record a cache lookup.
    pub fn record_lookup(&mut self, hit: bool) {
        self.lookups += 1;
        if hit {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
    }
}

/// Main compilation profiler.
pub struct CompilationProfiler {
    config: ProfileConfig,
    phases: Vec<PhaseTime>,
    active_phases: Vec<(String, Instant)>,
    memory_snapshots: Vec<MemorySnapshot>,
    pass_profiles: HashMap<String, PassProfile>,
    cache_stats: CacheStats,
    start_time: Option<Instant>,
}

impl CompilationProfiler {
    /// Create a new profiler with default configuration.
    pub fn new() -> Self {
        Self::with_config(ProfileConfig::default())
    }

    /// Create a new profiler with custom configuration.
    pub fn with_config(config: ProfileConfig) -> Self {
        Self {
            config,
            phases: Vec::new(),
            active_phases: Vec::new(),
            memory_snapshots: Vec::new(),
            pass_profiles: HashMap::new(),
            cache_stats: CacheStats::default(),
            start_time: None,
        }
    }

    /// Start overall compilation profiling.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.phases.clear();
        self.active_phases.clear();
    }

    /// Start profiling a compilation phase.
    pub fn start_phase(&mut self, name: &str) {
        if !self.config.track_time {
            return;
        }

        self.active_phases.push((name.to_string(), Instant::now()));
    }

    /// End profiling a compilation phase.
    pub fn end_phase(&mut self, name: &str) {
        if !self.config.track_time {
            return;
        }

        if let Some(pos) = self.active_phases.iter().rposition(|(n, _)| n == name) {
            let (phase_name, start_time) = self.active_phases.remove(pos);
            let duration = start_time.elapsed();

            if duration.as_millis() >= self.config.min_duration_ms as u128 {
                self.phases.push(PhaseTime::new(phase_name, duration));
            }
        }
    }

    /// Record a pass execution.
    pub fn record_pass(&mut self, pass_name: &str, duration: Duration, optimizations: usize) {
        if !self.config.track_passes {
            return;
        }

        let profile = self
            .pass_profiles
            .entry(pass_name.to_string())
            .or_insert_with(|| PassProfile::new(pass_name.to_string()));

        profile.record_execution(duration, optimizations);
    }

    /// Take a memory snapshot.
    pub fn snapshot_memory(&mut self) {
        if !self.config.track_memory {
            return;
        }

        self.memory_snapshots.push(MemorySnapshot::new());
    }

    /// Record a cache lookup.
    pub fn record_cache_lookup(&mut self, hit: bool) {
        if !self.config.track_cache {
            return;
        }

        self.cache_stats.record_lookup(hit);
    }

    /// Get total compilation time.
    pub fn total_time(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }

    /// Get peak memory usage.
    pub fn peak_memory(&self) -> usize {
        self.memory_snapshots
            .iter()
            .map(|s| s.heap_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Get the slowest compilation phase.
    pub fn slowest_phase(&self) -> Option<&PhaseTime> {
        self.phases.iter().max_by_key(|p| p.duration)
    }

    /// Get the most expensive pass (by total time).
    pub fn most_expensive_pass(&self) -> Option<&PassProfile> {
        self.pass_profiles.values().max_by_key(|p| p.total_time)
    }

    /// Generate a human-readable profiling report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Compilation Profiling Report ===\n\n");

        // Overall stats
        if let Some(total) = self.total_time() {
            report.push_str(&format!("Total Time: {:.2?}\n", total));
        }

        if self.config.track_memory {
            report.push_str(&format!("Peak Memory: {} bytes\n", self.peak_memory()));
        }

        report.push('\n');

        // Phase breakdown
        if self.config.track_time && !self.phases.is_empty() {
            report.push_str("=== Phase Breakdown ===\n");
            for phase in &self.phases {
                report.push_str(&format!(
                    "  {}: {:.2?} ({} times, avg: {:.2?})\n",
                    phase.name,
                    phase.duration,
                    phase.count,
                    phase.average_duration()
                ));
            }
            report.push('\n');
        }

        // Pass profiles
        if self.config.track_passes && !self.pass_profiles.is_empty() {
            report.push_str("=== Optimization Passes ===\n");
            let mut passes: Vec<_> = self.pass_profiles.values().collect();
            passes.sort_by_key(|p| std::cmp::Reverse(p.total_time));

            for pass in passes.iter().take(10) {
                report.push_str(&format!(
                    "  {}: {:.2?} ({} execs, {:.1} opts/exec)\n",
                    pass.name,
                    pass.total_time,
                    pass.execution_count,
                    pass.optimizations_per_execution()
                ));
            }
            report.push('\n');
        }

        // Cache statistics
        if self.config.track_cache && self.cache_stats.lookups > 0 {
            report.push_str("=== Cache Statistics ===\n");
            report.push_str(&format!("  Lookups: {}\n", self.cache_stats.lookups));
            report.push_str(&format!("  Hits: {}\n", self.cache_stats.hits));
            report.push_str(&format!("  Misses: {}\n", self.cache_stats.misses));
            report.push_str(&format!(
                "  Hit Rate: {:.1}%\n",
                self.cache_stats.hit_rate()
            ));
            report.push('\n');
        }

        // Recommendations
        if let Some(slowest) = self.slowest_phase() {
            report.push_str("=== Recommendations ===\n");
            report.push_str(&format!(
                "  Slowest phase: {} ({:.2?})\n",
                slowest.name, slowest.duration
            ));

            if let Some(expensive_pass) = self.most_expensive_pass() {
                report.push_str(&format!(
                    "  Most expensive pass: {} ({:.2?})\n",
                    expensive_pass.name, expensive_pass.total_time
                ));
            }

            if self.config.track_cache && self.cache_stats.hit_rate() < 50.0 {
                report.push_str("  Consider increasing cache size (low hit rate)\n");
            }
        }

        report
    }

    /// Generate JSON profiling report.
    pub fn generate_json_report(&self) -> String {
        // Simple JSON serialization
        let mut json = String::from("{\n");

        if let Some(total) = self.total_time() {
            json.push_str(&format!("  \"total_time_ms\": {},\n", total.as_millis()));
        }

        json.push_str(&format!(
            "  \"peak_memory_bytes\": {},\n",
            self.peak_memory()
        ));

        // Phases
        json.push_str("  \"phases\": [\n");
        for (i, phase) in self.phases.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"name\": \"{}\", \"duration_ms\": {}, \"count\": {}}}",
                phase.name,
                phase.duration.as_millis(),
                phase.count
            ));
            if i < self.phases.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("  ],\n");

        // Cache stats
        json.push_str("  \"cache\": {\n");
        json.push_str(&format!("    \"lookups\": {},\n", self.cache_stats.lookups));
        json.push_str(&format!("    \"hits\": {},\n", self.cache_stats.hits));
        json.push_str(&format!(
            "    \"hit_rate\": {:.2}\n",
            self.cache_stats.hit_rate()
        ));
        json.push_str("  }\n");

        json.push_str("}\n");
        json
    }
}

impl Default for CompilationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_basic() {
        let mut profiler = CompilationProfiler::new();
        profiler.start();

        profiler.start_phase("test_phase");
        thread::sleep(Duration::from_millis(10));
        profiler.end_phase("test_phase");

        assert!(!profiler.phases.is_empty());
    }

    #[test]
    fn test_phase_time() {
        let phase = PhaseTime::new("test".to_string(), Duration::from_secs(1));
        assert_eq!(phase.name, "test");
        assert_eq!(phase.count, 1);
        assert_eq!(phase.average_duration(), Duration::from_secs(1));
    }

    #[test]
    fn test_memory_snapshot() {
        let mut snapshot = MemorySnapshot::new();
        snapshot.record_allocation(1000);
        snapshot.record_allocation(500);

        assert_eq!(snapshot.heap_bytes, 1500);
        assert_eq!(snapshot.allocation_count, 2);

        snapshot.record_deallocation(500);
        assert_eq!(snapshot.heap_bytes, 1000);
        assert_eq!(snapshot.allocation_count, 1);
    }

    #[test]
    fn test_pass_profile() {
        let mut profile = PassProfile::new("constant_folding".to_string());
        profile.record_execution(Duration::from_millis(10), 5);
        profile.record_execution(Duration::from_millis(15), 3);

        assert_eq!(profile.execution_count, 2);
        assert_eq!(profile.optimizations_applied, 8);
        assert!(profile.average_time().as_millis() >= 10);
        assert_eq!(profile.optimizations_per_execution(), 4.0);
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::default();
        stats.record_lookup(true); // hit
        stats.record_lookup(true); // hit
        stats.record_lookup(false); // miss

        assert_eq!(stats.lookups, 3);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 66.67).abs() < 0.1);
    }

    #[test]
    fn test_generate_report() {
        let mut profiler = CompilationProfiler::new();
        profiler.start();
        profiler.start_phase("compilation");
        thread::sleep(Duration::from_millis(10));
        profiler.end_phase("compilation");

        let report = profiler.generate_report();
        assert!(report.contains("Compilation Profiling Report"));
        assert!(report.contains("Total Time"));
    }

    #[test]
    fn test_slowest_phase() {
        let mut profiler = CompilationProfiler::new();
        profiler.start();

        profiler.start_phase("fast");
        thread::sleep(Duration::from_millis(5));
        profiler.end_phase("fast");

        profiler.start_phase("slow");
        thread::sleep(Duration::from_millis(20));
        profiler.end_phase("slow");

        let slowest = profiler.slowest_phase().unwrap();
        assert_eq!(slowest.name, "slow");
    }

    #[test]
    fn test_most_expensive_pass() {
        let mut profiler = CompilationProfiler::new();
        profiler.record_pass("pass1", Duration::from_millis(10), 5);
        profiler.record_pass("pass2", Duration::from_millis(50), 10);

        let expensive = profiler.most_expensive_pass().unwrap();
        assert_eq!(expensive.name, "pass2");
    }

    #[test]
    fn test_json_report() {
        let mut profiler = CompilationProfiler::new();
        profiler.start();
        profiler.record_cache_lookup(true);
        profiler.record_cache_lookup(false);

        let json = profiler.generate_json_report();
        assert!(json.contains("total_time_ms"));
        assert!(json.contains("cache"));
        assert!(json.contains("hit_rate"));
    }

    #[test]
    fn test_config_filtering() {
        let config = ProfileConfig {
            track_time: true,
            track_memory: false,
            track_passes: true,
            track_cache: false,
            min_duration_ms: 100,
        };

        let mut profiler = CompilationProfiler::with_config(config);
        profiler.start();

        // Short phase should be filtered out
        profiler.start_phase("short");
        thread::sleep(Duration::from_millis(1));
        profiler.end_phase("short");

        assert!(profiler.phases.is_empty());
    }

    #[test]
    fn test_nested_phases() {
        let mut profiler = CompilationProfiler::new();
        profiler.start();

        profiler.start_phase("outer");
        thread::sleep(Duration::from_millis(5));

        profiler.start_phase("inner");
        thread::sleep(Duration::from_millis(5));
        profiler.end_phase("inner");

        profiler.end_phase("outer");

        assert_eq!(profiler.phases.len(), 2);
    }
}
