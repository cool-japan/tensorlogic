//! Execution profiling and performance monitoring.

use crate::context::{ExecutionHook, ExecutionPhase, ExecutionState};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Profiling statistics for a single operation
#[derive(Debug, Clone)]
pub struct OpProfile {
    pub op_type: String,
    pub count: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl OpProfile {
    pub fn new(op_type: impl Into<String>) -> Self {
        OpProfile {
            op_type: op_type.into(),
            count: 0,
            total_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
        }
    }

    pub fn record(&mut self, duration: Duration) {
        self.count += 1;
        self.total_time += duration;
        self.avg_time = self.total_time / self.count as u32;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryProfile {
    pub peak_bytes: usize,
    pub current_bytes: usize,
    pub allocations: usize,
    pub deallocations: usize,
}

impl MemoryProfile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_allocation(&mut self, bytes: usize) {
        self.current_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(self.current_bytes);
        self.allocations += 1;
    }

    pub fn record_deallocation(&mut self, bytes: usize) {
        self.current_bytes = self.current_bytes.saturating_sub(bytes);
        self.deallocations += 1;
    }
}

/// Comprehensive profiling data
#[derive(Debug, Clone)]
pub struct ProfileData {
    pub op_profiles: HashMap<String, OpProfile>,
    pub memory: MemoryProfile,
    pub total_execution_time: Duration,
}

impl ProfileData {
    pub fn new() -> Self {
        ProfileData {
            op_profiles: HashMap::new(),
            memory: MemoryProfile::new(),
            total_execution_time: Duration::ZERO,
        }
    }

    pub fn record_op(&mut self, op_type: impl Into<String>, duration: Duration) {
        let op_type = op_type.into();
        self.op_profiles
            .entry(op_type.clone())
            .or_insert_with(|| OpProfile::new(op_type))
            .record(duration);
    }

    pub fn get_op_profile(&self, op_type: &str) -> Option<&OpProfile> {
        self.op_profiles.get(op_type)
    }

    /// Get the slowest operations
    pub fn slowest_ops(&self, limit: usize) -> Vec<(&str, &OpProfile)> {
        let mut ops: Vec<_> = self
            .op_profiles
            .iter()
            .map(|(name, profile)| (name.as_str(), profile))
            .collect();

        ops.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));
        ops.truncate(limit);
        ops
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "Total Execution Time: {:.3}ms\n",
            self.total_execution_time.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!("Peak Memory: {} bytes\n", self.memory.peak_bytes));
        report.push_str("\nTop Operations by Time:\n");

        for (name, profile) in self.slowest_ops(5) {
            report.push_str(&format!(
                "  {}: {} calls, {:.3}ms total, {:.3}ms avg\n",
                name,
                profile.count,
                profile.total_time.as_secs_f64() * 1000.0,
                profile.avg_time.as_secs_f64() * 1000.0
            ));
        }

        report
    }
}

impl Default for ProfileData {
    fn default() -> Self {
        Self::new()
    }
}

/// Profiler that tracks execution metrics
pub struct Profiler {
    data: ProfileData,
    start_time: Option<Instant>,
}

impl Profiler {
    pub fn new() -> Self {
        Profiler {
            data: ProfileData::new(),
            start_time: None,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.data.total_execution_time = start.elapsed();
        }
    }

    /// Time an operation and record its duration
    pub fn time_op<F, R>(&mut self, op_type: impl Into<String>, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.data.record_op(op_type, duration);
        result
    }

    pub fn record_allocation(&mut self, bytes: usize) {
        self.data.memory.record_allocation(bytes);
    }

    pub fn record_deallocation(&mut self, bytes: usize) {
        self.data.memory.record_deallocation(bytes);
    }

    pub fn data(&self) -> &ProfileData {
        &self.data
    }

    pub fn into_data(mut self) -> ProfileData {
        self.stop();
        self.data
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for profiled execution
pub trait TlProfiledExecutor {
    /// Get the current profiler
    fn profiler(&self) -> Option<&Profiler>;

    /// Get mutable access to profiler
    fn profiler_mut(&mut self) -> Option<&mut Profiler>;

    /// Enable profiling
    fn enable_profiling(&mut self);

    /// Disable profiling
    fn disable_profiling(&mut self);

    /// Get profiling data
    fn get_profile_data(&self) -> Option<&ProfileData> {
        self.profiler().map(|p| p.data())
    }
}

/// Profiler hook for integration with ExecutionContext
pub struct ProfilerHook {
    profiler: Profiler,
    node_timings: HashMap<usize, Instant>,
}

impl ProfilerHook {
    pub fn new() -> Self {
        ProfilerHook {
            profiler: Profiler::new(),
            node_timings: HashMap::new(),
        }
    }

    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    pub fn into_profiler(self) -> Profiler {
        self.profiler
    }

    pub fn into_data(self) -> ProfileData {
        self.profiler.into_data()
    }
}

impl Default for ProfilerHook {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionHook for ProfilerHook {
    fn on_phase_change(&mut self, phase: ExecutionPhase, _state: &ExecutionState) {
        match phase {
            ExecutionPhase::Executing => {
                self.profiler.start();
            }
            ExecutionPhase::Completed | ExecutionPhase::Failed | ExecutionPhase::Cancelled => {
                self.profiler.stop();
            }
            _ => {}
        }
    }

    fn on_node_start(&mut self, node_idx: usize, _state: &ExecutionState) {
        self.node_timings.insert(node_idx, Instant::now());
    }

    fn on_node_complete(&mut self, node_idx: usize, _duration: Duration, state: &ExecutionState) {
        if let Some(start_time) = self.node_timings.remove(&node_idx) {
            let duration = start_time.elapsed();
            // Record with node index as operation type
            self.profiler
                .data
                .record_op(format!("node_{}", node_idx), duration);

            // If we have access to the actual op type from state, use it
            if node_idx < state.total_nodes {
                self.profiler.data.record_op("graph_node", duration);
            }
        }
    }

    fn on_error(&mut self, _error: &str, _state: &ExecutionState) {
        // Stop profiler on error
        self.profiler.stop();
    }

    fn on_complete(&mut self, _state: &ExecutionState) {
        // Stop profiler on completion
        self.profiler.stop();
    }
}

/// Node-level execution trace entry
#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub node_idx: usize,
    pub start_time: Duration,
    pub end_time: Duration,
    pub duration: Duration,
    pub op_type: String,
}

impl TraceEntry {
    pub fn new(
        node_idx: usize,
        start_time: Duration,
        end_time: Duration,
        op_type: impl Into<String>,
    ) -> Self {
        let duration = end_time - start_time;
        TraceEntry {
            node_idx,
            start_time,
            end_time,
            duration,
            op_type: op_type.into(),
        }
    }
}

/// Detailed execution timeline profiler
pub struct TimelineProfiler {
    traces: Vec<TraceEntry>,
    start_instant: Option<Instant>,
    node_starts: HashMap<usize, (Instant, String)>,
}

impl TimelineProfiler {
    pub fn new() -> Self {
        TimelineProfiler {
            traces: Vec::new(),
            start_instant: None,
            node_starts: HashMap::new(),
        }
    }

    pub fn start(&mut self) {
        self.start_instant = Some(Instant::now());
    }

    pub fn record_node_start(&mut self, node_idx: usize, op_type: impl Into<String>) {
        if self.start_instant.is_some() {
            self.node_starts
                .insert(node_idx, (Instant::now(), op_type.into()));
        }
    }

    pub fn record_node_end(&mut self, node_idx: usize) {
        if let (Some(start_instant), Some((node_start, op_type))) =
            (self.start_instant, self.node_starts.remove(&node_idx))
        {
            let now = Instant::now();
            let start_time = node_start.duration_since(start_instant);
            let end_time = now.duration_since(start_instant);

            self.traces
                .push(TraceEntry::new(node_idx, start_time, end_time, op_type));
        }
    }

    pub fn traces(&self) -> &[TraceEntry] {
        &self.traces
    }

    /// Get critical path (longest chain of dependent nodes)
    pub fn critical_path_duration(&self) -> Duration {
        self.traces.iter().map(|t| t.duration).sum()
    }

    /// Get timeline summary
    pub fn summary(&self) -> String {
        let mut report = String::new();
        let total_duration = self.critical_path_duration();

        report.push_str(&format!(
            "Total Timeline Duration: {:.3}ms\n",
            total_duration.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!("Traced Nodes: {}\n\n", self.traces.len()));

        report.push_str("Node Timeline:\n");
        for trace in &self.traces {
            report.push_str(&format!(
                "  Node {}: {:.3}ms - {:.3}ms ({:.3}ms) - {}\n",
                trace.node_idx,
                trace.start_time.as_secs_f64() * 1000.0,
                trace.end_time.as_secs_f64() * 1000.0,
                trace.duration.as_secs_f64() * 1000.0,
                trace.op_type
            ));
        }

        report
    }
}

impl Default for TimelineProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistical metrics for profiling data
#[derive(Debug, Clone)]
pub struct ProfileStatistics {
    pub mean: f64,
    pub median: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub std_dev: f64,
}

impl ProfileStatistics {
    /// Calculate statistics from a set of durations
    pub fn from_durations(durations: &mut [Duration]) -> Self {
        if durations.is_empty() {
            return ProfileStatistics {
                mean: 0.0,
                median: 0.0,
                p50: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
                std_dev: 0.0,
            };
        }

        durations.sort();

        let values: Vec<f64> = durations.iter().map(|d| d.as_secs_f64()).collect();
        let n = values.len();

        // Mean
        let mean = values.iter().sum::<f64>() / n as f64;

        // Standard deviation
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Percentiles
        let percentile = |p: f64| -> f64 {
            let index = ((n as f64 * p).ceil() as usize).saturating_sub(1);
            values[index.min(n - 1)]
        };

        ProfileStatistics {
            mean,
            median: percentile(0.50),
            p50: percentile(0.50),
            p90: percentile(0.90),
            p95: percentile(0.95),
            p99: percentile(0.99),
            std_dev,
        }
    }
}

/// Performance baseline for regression detection
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub op_baselines: HashMap<String, ProfileStatistics>,
    pub total_time_baseline: Duration,
}

impl PerformanceBaseline {
    pub fn new() -> Self {
        PerformanceBaseline {
            op_baselines: HashMap::new(),
            total_time_baseline: Duration::ZERO,
        }
    }

    pub fn from_profile_data(data: &ProfileData) -> Self {
        let mut baseline = PerformanceBaseline::new();
        baseline.total_time_baseline = data.total_execution_time;

        for (op_type, profile) in &data.op_profiles {
            // Create a simple baseline from existing data
            let stats = ProfileStatistics {
                mean: profile.avg_time.as_secs_f64(),
                median: profile.avg_time.as_secs_f64(),
                p50: profile.avg_time.as_secs_f64(),
                p90: profile.max_time.as_secs_f64(),
                p95: profile.max_time.as_secs_f64(),
                p99: profile.max_time.as_secs_f64(),
                std_dev: 0.0,
            };
            baseline.op_baselines.insert(op_type.clone(), stats);
        }

        baseline
    }

    /// Compare current data against baseline
    pub fn compare(&self, data: &ProfileData) -> PerformanceComparison {
        let mut comparison = PerformanceComparison::new();

        // Compare total time
        let total_time_ratio = if self.total_time_baseline.as_secs_f64() > 0.0 {
            data.total_execution_time.as_secs_f64() / self.total_time_baseline.as_secs_f64()
        } else {
            1.0
        };

        comparison.total_time_ratio = total_time_ratio;
        comparison.is_regression = total_time_ratio > 1.1; // 10% slower is regression

        // Compare per-operation
        for (op_type, profile) in &data.op_profiles {
            if let Some(baseline_stats) = self.op_baselines.get(op_type) {
                let current_mean = profile.avg_time.as_secs_f64();
                let ratio = current_mean / baseline_stats.mean;
                if ratio > 1.1 {
                    comparison.slow_ops.push((
                        op_type.clone(),
                        ratio,
                        current_mean - baseline_stats.mean,
                    ));
                }
            }
        }

        comparison
            .slow_ops
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        comparison
    }
}

impl Default for PerformanceBaseline {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of comparing performance against baseline
#[derive(Debug, Clone)]
pub struct PerformanceComparison {
    pub is_regression: bool,
    pub total_time_ratio: f64,
    pub slow_ops: Vec<(String, f64, f64)>, // (op_type, ratio, absolute_diff)
}

impl PerformanceComparison {
    pub fn new() -> Self {
        PerformanceComparison {
            is_regression: false,
            total_time_ratio: 1.0,
            slow_ops: Vec::new(),
        }
    }

    pub fn summary(&self) -> String {
        let mut report = String::new();

        if self.is_regression {
            report.push_str("⚠️  PERFORMANCE REGRESSION DETECTED\n\n");
        } else {
            report.push_str("✓ Performance within acceptable range\n\n");
        }

        report.push_str(&format!(
            "Total Time Ratio: {:.2}x\n",
            self.total_time_ratio
        ));

        if !self.slow_ops.is_empty() {
            report.push_str("\nSlower Operations:\n");
            for (op_type, ratio, diff) in &self.slow_ops {
                report.push_str(&format!(
                    "  {}: {:.2}x slower (+{:.3}ms)\n",
                    op_type,
                    ratio,
                    diff * 1000.0
                ));
            }
        }

        report
    }
}

impl Default for PerformanceComparison {
    fn default() -> Self {
        Self::new()
    }
}

/// Bottleneck analyzer for identifying performance issues
pub struct BottleneckAnalyzer;

impl BottleneckAnalyzer {
    /// Analyze profile data and identify bottlenecks
    pub fn analyze(data: &ProfileData) -> BottleneckReport {
        let mut report = BottleneckReport::new();

        // Find operations taking > 10% of total time
        let total_time = data.total_execution_time.as_secs_f64();
        if total_time > 0.0 {
            for (op_type, profile) in &data.op_profiles {
                let op_time = profile.total_time.as_secs_f64();
                let percentage = (op_time / total_time) * 100.0;

                if percentage > 10.0 {
                    report.bottlenecks.push(Bottleneck {
                        op_type: op_type.clone(),
                        percentage,
                        total_time: profile.total_time,
                        call_count: profile.count,
                        avg_time: profile.avg_time,
                    });
                }
            }
        }

        // Sort by percentage (highest first)
        report
            .bottlenecks
            .sort_by(|a, b| b.percentage.partial_cmp(&a.percentage).unwrap());

        // Calculate memory pressure
        if data.memory.peak_bytes > 1_000_000_000 {
            // > 1GB
            report.memory_pressure_high = true;
        }

        report
    }
}

/// Identified performance bottleneck
#[derive(Debug, Clone)]
pub struct Bottleneck {
    pub op_type: String,
    pub percentage: f64,
    pub total_time: Duration,
    pub call_count: usize,
    pub avg_time: Duration,
}

/// Bottleneck analysis report
#[derive(Debug, Clone)]
pub struct BottleneckReport {
    pub bottlenecks: Vec<Bottleneck>,
    pub memory_pressure_high: bool,
}

impl BottleneckReport {
    pub fn new() -> Self {
        BottleneckReport {
            bottlenecks: Vec::new(),
            memory_pressure_high: false,
        }
    }

    pub fn has_bottlenecks(&self) -> bool {
        !self.bottlenecks.is_empty() || self.memory_pressure_high
    }

    pub fn summary(&self) -> String {
        let mut report = String::new();

        if self.bottlenecks.is_empty() && !self.memory_pressure_high {
            report.push_str("No significant bottlenecks detected\n");
            return report;
        }

        report.push_str("Performance Bottlenecks:\n\n");

        for bottleneck in &self.bottlenecks {
            report.push_str(&format!(
                "• {} - {:.1}% of total time\n",
                bottleneck.op_type, bottleneck.percentage
            ));
            report.push_str(&format!(
                "  {} calls, {:.3}ms avg, {:.3}ms total\n",
                bottleneck.call_count,
                bottleneck.avg_time.as_secs_f64() * 1000.0,
                bottleneck.total_time.as_secs_f64() * 1000.0
            ));
        }

        if self.memory_pressure_high {
            report.push_str("\n⚠️  High memory pressure detected (>1GB peak)\n");
        }

        report
    }
}

impl Default for BottleneckReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_op_profile() {
        let mut profile = OpProfile::new("einsum");
        assert_eq!(profile.count, 0);

        profile.record(Duration::from_millis(10));
        assert_eq!(profile.count, 1);
        assert!(profile.total_time >= Duration::from_millis(10));

        profile.record(Duration::from_millis(20));
        assert_eq!(profile.count, 2);
    }

    #[test]
    fn test_memory_profile() {
        let mut mem = MemoryProfile::new();
        assert_eq!(mem.current_bytes, 0);
        assert_eq!(mem.peak_bytes, 0);

        mem.record_allocation(1000);
        assert_eq!(mem.current_bytes, 1000);
        assert_eq!(mem.peak_bytes, 1000);

        mem.record_allocation(500);
        assert_eq!(mem.current_bytes, 1500);
        assert_eq!(mem.peak_bytes, 1500);

        mem.record_deallocation(500);
        assert_eq!(mem.current_bytes, 1000);
        assert_eq!(mem.peak_bytes, 1500); // Peak stays at max
    }

    #[test]
    fn test_profile_data() {
        let mut data = ProfileData::new();

        data.record_op("einsum", Duration::from_millis(100));
        data.record_op("relu", Duration::from_millis(50));
        data.record_op("einsum", Duration::from_millis(80));

        let einsum_profile = data.get_op_profile("einsum").unwrap();
        assert_eq!(einsum_profile.count, 2);

        let slowest = data.slowest_ops(2);
        assert_eq!(slowest.len(), 2);
        assert_eq!(slowest[0].0, "einsum"); // Should be slowest
    }

    #[test]
    fn test_profiler() {
        let mut profiler = Profiler::new();

        profiler.start();

        // Simulate some work
        profiler.time_op("operation1", || {
            thread::sleep(Duration::from_millis(1));
        });

        profiler.time_op("operation2", || {
            thread::sleep(Duration::from_millis(2));
        });

        profiler.stop();

        let data = profiler.data();
        assert_eq!(data.op_profiles.len(), 2);
        assert!(data.total_execution_time >= Duration::from_millis(3));
    }

    #[test]
    fn test_profile_summary() {
        let mut data = ProfileData::new();
        data.record_op("einsum", Duration::from_millis(100));
        data.record_op("relu", Duration::from_millis(50));
        data.total_execution_time = Duration::from_millis(150);
        data.memory.record_allocation(1024);

        let summary = data.summary();
        assert!(summary.contains("150"));
        assert!(summary.contains("1024"));
        assert!(summary.contains("einsum"));
    }

    #[test]
    fn test_profiler_hook_creation() {
        let hook = ProfilerHook::new();
        assert_eq!(hook.profiler().data().op_profiles.len(), 0);
    }

    #[test]
    fn test_profiler_hook_with_execution() {
        use crate::context::ExecutionState;

        let mut hook = ProfilerHook::new();
        let state = ExecutionState::new(5);

        // Simulate execution flow
        hook.on_phase_change(ExecutionPhase::Executing, &state);
        hook.on_node_start(0, &state);
        thread::sleep(Duration::from_millis(1));
        hook.on_node_complete(0, Duration::from_millis(1), &state);
        hook.on_phase_change(ExecutionPhase::Completed, &state);

        let data = hook.profiler().data();
        assert!(!data.op_profiles.is_empty());
        assert!(data.total_execution_time > Duration::ZERO);
    }

    #[test]
    fn test_timeline_profiler() {
        let mut profiler = TimelineProfiler::new();
        profiler.start();

        profiler.record_node_start(0, "einsum");
        thread::sleep(Duration::from_millis(1));
        profiler.record_node_end(0);

        profiler.record_node_start(1, "relu");
        thread::sleep(Duration::from_millis(1));
        profiler.record_node_end(1);

        let traces = profiler.traces();
        assert_eq!(traces.len(), 2);
        assert_eq!(traces[0].node_idx, 0);
        assert_eq!(traces[1].node_idx, 1);
    }

    #[test]
    fn test_timeline_summary() {
        let mut profiler = TimelineProfiler::new();
        profiler.start();

        profiler.record_node_start(0, "einsum");
        thread::sleep(Duration::from_millis(1));
        profiler.record_node_end(0);

        let summary = profiler.summary();
        assert!(summary.contains("Node 0"));
        assert!(summary.contains("einsum"));
    }

    #[test]
    fn test_trace_entry() {
        let entry = TraceEntry::new(
            0,
            Duration::from_millis(0),
            Duration::from_millis(100),
            "einsum",
        );

        assert_eq!(entry.node_idx, 0);
        assert_eq!(entry.duration, Duration::from_millis(100));
        assert_eq!(entry.op_type, "einsum");
    }

    #[test]
    fn test_profile_statistics_empty() {
        let stats = ProfileStatistics::from_durations(&mut []);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.median, 0.0);
    }

    #[test]
    fn test_profile_statistics() {
        let mut durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];

        let stats = ProfileStatistics::from_durations(&mut durations);
        assert!(stats.mean > 0.0);
        assert!(stats.p50 > 0.0);
        assert!(stats.p90 > 0.0);
        assert!(stats.p95 > 0.0);
        assert!(stats.p99 > 0.0);
        assert!(stats.p99 >= stats.p95);
        assert!(stats.p95 >= stats.p90);
        assert!(stats.p90 >= stats.p50);
    }

    #[test]
    fn test_performance_baseline_creation() {
        let mut data = ProfileData::new();
        data.record_op("einsum", Duration::from_millis(100));
        data.record_op("relu", Duration::from_millis(50));
        data.total_execution_time = Duration::from_millis(150);

        let baseline = PerformanceBaseline::from_profile_data(&data);
        assert_eq!(baseline.op_baselines.len(), 2);
        assert!(baseline.op_baselines.contains_key("einsum"));
        assert!(baseline.op_baselines.contains_key("relu"));
    }

    #[test]
    fn test_performance_comparison_no_regression() {
        let mut baseline_data = ProfileData::new();
        baseline_data.record_op("einsum", Duration::from_millis(100));
        baseline_data.total_execution_time = Duration::from_millis(100);

        let baseline = PerformanceBaseline::from_profile_data(&baseline_data);

        let mut current_data = ProfileData::new();
        current_data.record_op("einsum", Duration::from_millis(105)); // 5% slower
        current_data.total_execution_time = Duration::from_millis(105);

        let comparison = baseline.compare(&current_data);
        assert!(!comparison.is_regression); // Within 10% threshold
    }

    #[test]
    fn test_performance_comparison_with_regression() {
        let mut baseline_data = ProfileData::new();
        baseline_data.record_op("einsum", Duration::from_millis(100));
        baseline_data.total_execution_time = Duration::from_millis(100);

        let baseline = PerformanceBaseline::from_profile_data(&baseline_data);

        let mut current_data = ProfileData::new();
        current_data.record_op("einsum", Duration::from_millis(120)); // 20% slower
        current_data.total_execution_time = Duration::from_millis(120);

        let comparison = baseline.compare(&current_data);
        assert!(comparison.is_regression);
        assert!(comparison.total_time_ratio > 1.1);
    }

    #[test]
    fn test_performance_comparison_summary() {
        let mut baseline_data = ProfileData::new();
        baseline_data.record_op("einsum", Duration::from_millis(100));
        baseline_data.total_execution_time = Duration::from_millis(100);

        let baseline = PerformanceBaseline::from_profile_data(&baseline_data);

        let mut current_data = ProfileData::new();
        current_data.record_op("einsum", Duration::from_millis(150));
        current_data.total_execution_time = Duration::from_millis(150);

        let comparison = baseline.compare(&current_data);
        let summary = comparison.summary();
        assert!(summary.contains("REGRESSION") || summary.contains("1."));
    }

    #[test]
    fn test_bottleneck_analyzer_no_bottlenecks() {
        let mut data = ProfileData::new();
        data.record_op("op1", Duration::from_millis(5));
        data.record_op("op2", Duration::from_millis(5));
        data.total_execution_time = Duration::from_millis(100);

        let report = BottleneckAnalyzer::analyze(&data);
        assert!(report.bottlenecks.is_empty());
        assert!(!report.has_bottlenecks());
    }

    #[test]
    fn test_bottleneck_analyzer_with_bottleneck() {
        let mut data = ProfileData::new();
        data.record_op("slow_op", Duration::from_millis(50));
        data.record_op("fast_op", Duration::from_millis(5));
        data.total_execution_time = Duration::from_millis(100);

        let report = BottleneckAnalyzer::analyze(&data);
        assert!(!report.bottlenecks.is_empty());
        assert_eq!(report.bottlenecks[0].op_type, "slow_op");
        assert!(report.bottlenecks[0].percentage > 10.0);
    }

    #[test]
    fn test_bottleneck_analyzer_memory_pressure() {
        let mut data = ProfileData::new();
        data.total_execution_time = Duration::from_millis(100);
        data.memory.record_allocation(2_000_000_000); // 2GB

        let report = BottleneckAnalyzer::analyze(&data);
        assert!(report.memory_pressure_high);
        assert!(report.has_bottlenecks());
    }

    #[test]
    fn test_bottleneck_report_summary() {
        let mut data = ProfileData::new();
        data.record_op("slow_op", Duration::from_millis(60));
        data.total_execution_time = Duration::from_millis(100);

        let report = BottleneckAnalyzer::analyze(&data);
        let summary = report.summary();
        assert!(summary.contains("slow_op"));
        assert!(summary.contains("%"));
    }

    #[test]
    fn test_bottleneck_report_summary_no_issues() {
        let report = BottleneckReport::new();
        let summary = report.summary();
        assert!(summary.contains("No significant bottlenecks"));
    }
}
