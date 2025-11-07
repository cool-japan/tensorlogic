//! Performance regression testing framework.
//!
//! This module provides infrastructure for tracking performance over time
//! and detecting regressions. Backend developers can use this to ensure
//! optimizations don't regress and to track performance improvements.
//!
//! # Example
//!
//! ```ignore
//! use tensorlogic_infer::perfregression::{PerfRegression, BenchmarkConfig};
//!
//! let mut perf = PerfRegression::new("my_backend");
//!
//! // Run benchmarks
//! perf.benchmark("matmul_1000x1000", || {
//!     executor.matmul(&a, &b)
//! })?;
//!
//! // Save baseline
//! perf.save_baseline("baselines/")?;
//!
//! // Later, compare against baseline
//! let report = perf.compare_to_baseline("baselines/")?;
//! if report.has_regressions() {
//!     eprintln!("Performance regressions detected!");
//!     report.print_regressions();
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Configuration for performance benchmarks
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Regression threshold (as percentage, e.g., 10.0 means 10% slower is a regression)
    pub regression_threshold_percent: f64,
    /// Improvement threshold (as percentage)
    pub improvement_threshold_percent: f64,
    /// Minimum execution time to consider (filter out noise)
    pub min_time_ns: u64,
    /// Whether to save detailed timing distributions
    pub save_distribution: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            regression_threshold_percent: 5.0,
            improvement_threshold_percent: 5.0,
            min_time_ns: 1000, // 1 microsecond
            save_distribution: false,
        }
    }
}

impl BenchmarkConfig {
    /// Create a quick config with fewer iterations
    pub fn quick() -> Self {
        BenchmarkConfig {
            warmup_iterations: 3,
            measurement_iterations: 20,
            ..Default::default()
        }
    }

    /// Create a thorough config with more iterations
    pub fn thorough() -> Self {
        BenchmarkConfig {
            warmup_iterations: 20,
            measurement_iterations: 200,
            save_distribution: true,
            ..Default::default()
        }
    }

    /// Set warmup iterations
    pub fn with_warmup(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Set measurement iterations
    pub fn with_measurements(mut self, iterations: usize) -> Self {
        self.measurement_iterations = iterations;
        self
    }

    /// Set regression threshold
    pub fn with_regression_threshold(mut self, percent: f64) -> Self {
        self.regression_threshold_percent = percent;
        self
    }
}

/// Statistics for a single benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStats {
    /// Benchmark name
    pub name: String,
    /// Number of samples
    pub samples: usize,
    /// Mean execution time (nanoseconds)
    pub mean_ns: f64,
    /// Median execution time
    pub median_ns: f64,
    /// Standard deviation
    pub std_dev_ns: f64,
    /// Minimum execution time
    pub min_ns: u64,
    /// Maximum execution time
    pub max_ns: u64,
    /// Timestamp when benchmark was run
    pub timestamp: String,
    /// Optional: full distribution of timings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distribution: Option<Vec<u64>>,
}

impl BenchmarkStats {
    /// Create from a list of timing samples
    pub fn from_samples(name: String, samples: Vec<u64>) -> Self {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<u64>() as f64 / n;

        let mut sorted = samples.clone();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2] as f64;

        let variance = samples
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        BenchmarkStats {
            name,
            samples: samples.len(),
            mean_ns: mean,
            median_ns: median,
            std_dev_ns: std_dev,
            min_ns: min,
            max_ns: max,
            timestamp: chrono::Utc::now().to_rfc3339(),
            distribution: None,
        }
    }

    /// Calculate coefficient of variation (CV)
    pub fn coefficient_of_variation(&self) -> f64 {
        self.std_dev_ns / self.mean_ns
    }

    /// Check if measurements are stable (low CV)
    pub fn is_stable(&self, max_cv: f64) -> bool {
        self.coefficient_of_variation() < max_cv
    }

    /// Format duration in human-readable form
    pub fn format_mean(&self) -> String {
        format_duration_ns(self.mean_ns as u64)
    }

    /// Format median duration
    pub fn format_median(&self) -> String {
        format_duration_ns(self.median_ns as u64)
    }
}

/// Comparison between current and baseline benchmarks
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    pub name: String,
    pub current: BenchmarkStats,
    pub baseline: BenchmarkStats,
    pub change_percent: f64,
    pub is_regression: bool,
    pub is_improvement: bool,
}

impl BenchmarkComparison {
    /// Create a comparison
    pub fn new(
        current: BenchmarkStats,
        baseline: BenchmarkStats,
        regression_threshold: f64,
        improvement_threshold: f64,
    ) -> Self {
        let change_percent = ((current.mean_ns - baseline.mean_ns) / baseline.mean_ns) * 100.0;

        BenchmarkComparison {
            name: current.name.clone(),
            is_regression: change_percent > regression_threshold,
            is_improvement: change_percent < -improvement_threshold,
            current,
            baseline,
            change_percent,
        }
    }

    /// Get status symbol
    pub fn status_symbol(&self) -> &str {
        if self.is_regression {
            "⚠️"
        } else if self.is_improvement {
            "✨"
        } else {
            "✓"
        }
    }

    /// Summary line
    pub fn summary(&self) -> String {
        format!(
            "{} {}: {} -> {} ({:+.2}%)",
            self.status_symbol(),
            self.name,
            self.baseline.format_mean(),
            self.current.format_mean(),
            self.change_percent
        )
    }
}

/// Collection of benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkBaseline {
    /// Backend/system identifier
    pub backend_name: String,
    /// When baseline was created
    pub created_at: String,
    /// Benchmarks in this baseline
    pub benchmarks: HashMap<String, BenchmarkStats>,
    /// Metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl BenchmarkBaseline {
    /// Create a new baseline
    pub fn new(backend_name: String) -> Self {
        BenchmarkBaseline {
            backend_name,
            created_at: chrono::Utc::now().to_rfc3339(),
            benchmarks: HashMap::new(),
            metadata: None,
        }
    }

    /// Add a benchmark result
    pub fn add(&mut self, stats: BenchmarkStats) {
        self.benchmarks.insert(stats.name.clone(), stats);
    }

    /// Save to JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let baseline = serde_json::from_str(&json)?;
        Ok(baseline)
    }
}

/// Performance regression testing framework
pub struct PerfRegression {
    backend_name: String,
    config: BenchmarkConfig,
    current_results: HashMap<String, BenchmarkStats>,
}

impl PerfRegression {
    /// Create a new performance regression tester
    pub fn new(backend_name: impl Into<String>) -> Self {
        PerfRegression {
            backend_name: backend_name.into(),
            config: BenchmarkConfig::default(),
            current_results: HashMap::new(),
        }
    }

    /// Create with custom config
    pub fn with_config(backend_name: impl Into<String>, config: BenchmarkConfig) -> Self {
        PerfRegression {
            backend_name: backend_name.into(),
            config,
            current_results: HashMap::new(),
        }
    }

    /// Run a benchmark
    pub fn benchmark<F, R>(
        &mut self,
        name: impl Into<String>,
        mut f: F,
    ) -> Result<BenchmarkStats, String>
    where
        F: FnMut() -> R,
    {
        let name = name.into();

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = f();
        }

        // Measurements
        let mut samples = Vec::with_capacity(self.config.measurement_iterations);
        for _ in 0..self.config.measurement_iterations {
            let start = Instant::now();
            let _ = f();
            let duration = start.elapsed();

            let ns = duration.as_nanos() as u64;
            if ns >= self.config.min_time_ns {
                samples.push(ns);
            }
        }

        if samples.is_empty() {
            return Err(format!(
                "No valid samples for benchmark '{}' (all below min_time_ns threshold)",
                name
            ));
        }

        let mut stats = BenchmarkStats::from_samples(name.clone(), samples.clone());
        if self.config.save_distribution {
            stats.distribution = Some(samples);
        }

        self.current_results.insert(name, stats.clone());
        Ok(stats)
    }

    /// Get current results
    pub fn results(&self) -> &HashMap<String, BenchmarkStats> {
        &self.current_results
    }

    /// Save current results as baseline
    pub fn save_baseline<P: AsRef<Path>>(&self, dir: P) -> std::io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        let filename = format!("{}_baseline.json", self.backend_name);
        let path = dir.join(filename);

        let mut baseline = BenchmarkBaseline::new(self.backend_name.clone());
        for stats in self.current_results.values() {
            baseline.add(stats.clone());
        }

        baseline.save(path)
    }

    /// Compare current results to baseline
    pub fn compare_to_baseline<P: AsRef<Path>>(&self, dir: P) -> std::io::Result<RegressionReport> {
        let dir = dir.as_ref();
        let filename = format!("{}_baseline.json", self.backend_name);
        let path = dir.join(filename);

        let baseline = BenchmarkBaseline::load(path)?;

        let mut comparisons = Vec::new();
        for (name, current_stats) in &self.current_results {
            if let Some(baseline_stats) = baseline.benchmarks.get(name) {
                let comparison = BenchmarkComparison::new(
                    current_stats.clone(),
                    baseline_stats.clone(),
                    self.config.regression_threshold_percent,
                    self.config.improvement_threshold_percent,
                );
                comparisons.push(comparison);
            }
        }

        Ok(RegressionReport {
            backend_name: self.backend_name.clone(),
            comparisons,
            regression_threshold: self.config.regression_threshold_percent,
        })
    }

    /// Clear current results
    pub fn clear(&mut self) {
        self.current_results.clear();
    }
}

/// Report of regression testing results
#[derive(Debug)]
pub struct RegressionReport {
    pub backend_name: String,
    pub comparisons: Vec<BenchmarkComparison>,
    pub regression_threshold: f64,
}

impl RegressionReport {
    /// Check if there are any regressions
    pub fn has_regressions(&self) -> bool {
        self.comparisons.iter().any(|c| c.is_regression)
    }

    /// Get all regressions
    pub fn regressions(&self) -> Vec<&BenchmarkComparison> {
        self.comparisons
            .iter()
            .filter(|c| c.is_regression)
            .collect()
    }

    /// Get all improvements
    pub fn improvements(&self) -> Vec<&BenchmarkComparison> {
        self.comparisons
            .iter()
            .filter(|c| c.is_improvement)
            .collect()
    }

    /// Get unchanged benchmarks
    pub fn unchanged(&self) -> Vec<&BenchmarkComparison> {
        self.comparisons
            .iter()
            .filter(|c| !c.is_regression && !c.is_improvement)
            .collect()
    }

    /// Print regressions
    pub fn print_regressions(&self) {
        let regressions = self.regressions();
        if regressions.is_empty() {
            println!("No performance regressions detected! ✓");
            return;
        }

        println!(
            "\n⚠️  Performance Regressions Detected (threshold: {:.1}%):",
            self.regression_threshold
        );
        for comp in regressions {
            println!("  {}", comp.summary());
        }
    }

    /// Print improvements
    pub fn print_improvements(&self) {
        let improvements = self.improvements();
        if improvements.is_empty() {
            return;
        }

        println!("\n✨ Performance Improvements:");
        for comp in improvements {
            println!("  {}", comp.summary());
        }
    }

    /// Print full report
    pub fn print_report(&self) {
        println!(
            "\n=== Performance Regression Report: {} ===",
            self.backend_name
        );
        println!("Total benchmarks: {}", self.comparisons.len());
        println!("Regressions: {}", self.regressions().len());
        println!("Improvements: {}", self.improvements().len());
        println!("Unchanged: {}", self.unchanged().len());

        self.print_regressions();
        self.print_improvements();

        if !self.unchanged().is_empty() {
            println!("\n✓ Unchanged:");
            for comp in self.unchanged() {
                println!("  {}", comp.summary());
            }
        }
    }

    /// Generate HTML report
    pub fn to_html(&self) -> String {
        let mut html = String::from("<html><head><title>Performance Report</title></head><body>");
        html.push_str(&format!(
            "<h1>Performance Report: {}</h1>",
            self.backend_name
        ));
        html.push_str(&format!(
            "<p>Total: {} | Regressions: {} | Improvements: {}</p>",
            self.comparisons.len(),
            self.regressions().len(),
            self.improvements().len()
        ));

        if !self.regressions().is_empty() {
            html.push_str("<h2>⚠️ Regressions</h2><ul>");
            for comp in self.regressions() {
                html.push_str(&format!("<li style='color:red'>{}</li>", comp.summary()));
            }
            html.push_str("</ul>");
        }

        if !self.improvements().is_empty() {
            html.push_str("<h2>✨ Improvements</h2><ul>");
            for comp in self.improvements() {
                html.push_str(&format!("<li style='color:green'>{}</li>", comp.summary()));
            }
            html.push_str("</ul>");
        }

        html.push_str("</body></html>");
        html
    }
}

/// Format duration in human-readable form
fn format_duration_ns(ns: u64) -> String {
    if ns < 1_000 {
        format!("{} ns", ns)
    } else if ns < 1_000_000 {
        format!("{:.2} μs", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", ns as f64 / 1_000_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert!(config.warmup_iterations > 0);
        assert!(config.measurement_iterations > 0);
        assert!(config.regression_threshold_percent > 0.0);
    }

    #[test]
    fn test_benchmark_config_quick() {
        let quick = BenchmarkConfig::quick();
        let default = BenchmarkConfig::default();
        assert!(quick.measurement_iterations < default.measurement_iterations);
    }

    #[test]
    fn test_benchmark_config_builder() {
        let config = BenchmarkConfig::default()
            .with_warmup(5)
            .with_measurements(50)
            .with_regression_threshold(10.0);

        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measurement_iterations, 50);
        assert_eq!(config.regression_threshold_percent, 10.0);
    }

    #[test]
    fn test_benchmark_stats_from_samples() {
        let samples = vec![100, 110, 105, 108, 102];
        let stats = BenchmarkStats::from_samples("test".to_string(), samples);

        assert_eq!(stats.name, "test");
        assert_eq!(stats.samples, 5);
        assert!(stats.mean_ns > 100.0);
        assert!(stats.mean_ns < 110.0);
        assert_eq!(stats.min_ns, 100);
        assert_eq!(stats.max_ns, 110);
    }

    #[test]
    fn test_benchmark_stats_cv() {
        let samples = vec![100, 100, 100, 100, 100]; // No variation
        let stats = BenchmarkStats::from_samples("test".to_string(), samples);

        assert!(stats.coefficient_of_variation() < 0.01);
        assert!(stats.is_stable(0.1));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration_ns(500), "500 ns");
        assert_eq!(format_duration_ns(5_000), "5.00 μs");
        assert_eq!(format_duration_ns(5_000_000), "5.00 ms");
        assert_eq!(format_duration_ns(5_000_000_000), "5.00 s");
    }

    #[test]
    fn test_perf_regression_creation() {
        let perf = PerfRegression::new("test_backend");
        assert_eq!(perf.backend_name, "test_backend");
        assert!(perf.current_results.is_empty());
    }

    #[test]
    fn test_perf_regression_benchmark() {
        let mut perf = PerfRegression::with_config("test", BenchmarkConfig::quick());

        let stats = perf
            .benchmark("simple", || {
                std::thread::sleep(std::time::Duration::from_micros(10));
            })
            .unwrap();

        assert_eq!(stats.name, "simple");
        assert!(stats.samples > 0);
        assert!(stats.mean_ns > 10_000.0); // At least 10 microseconds
    }

    #[test]
    fn test_benchmark_comparison() {
        let baseline = BenchmarkStats::from_samples("test".to_string(), vec![100, 100, 100]);
        let current = BenchmarkStats::from_samples("test".to_string(), vec![110, 110, 110]);

        let comp = BenchmarkComparison::new(current, baseline, 5.0, 5.0);

        assert!(comp.change_percent > 5.0); // 10% slower
        assert!(comp.is_regression);
        assert!(!comp.is_improvement);
    }

    #[test]
    fn test_benchmark_improvement() {
        let baseline = BenchmarkStats::from_samples("test".to_string(), vec![100, 100, 100]);
        let current = BenchmarkStats::from_samples("test".to_string(), vec![90, 90, 90]);

        let comp = BenchmarkComparison::new(current, baseline, 5.0, 5.0);

        assert!(comp.change_percent < -5.0); // 10% faster
        assert!(!comp.is_regression);
        assert!(comp.is_improvement);
    }

    #[test]
    fn test_regression_report() {
        let baseline = BenchmarkStats::from_samples("test1".to_string(), vec![100, 100, 100]);
        let current = BenchmarkStats::from_samples("test1".to_string(), vec![110, 110, 110]);
        let comp = BenchmarkComparison::new(current, baseline, 5.0, 5.0);

        let report = RegressionReport {
            backend_name: "test".to_string(),
            comparisons: vec![comp],
            regression_threshold: 5.0,
        };

        assert!(report.has_regressions());
        assert_eq!(report.regressions().len(), 1);
        assert_eq!(report.improvements().len(), 0);
    }

    #[test]
    fn test_clear_results() {
        let mut config = BenchmarkConfig::quick();
        config.min_time_ns = 0; // Accept all samples
        let mut perf = PerfRegression::with_config("test", config);
        // Use a non-empty function to ensure it takes some time
        perf.benchmark("test", || {
            let _x = (0..100).sum::<i32>();
        })
        .unwrap();
        assert!(!perf.results().is_empty());

        perf.clear();
        assert!(perf.results().is_empty());
    }
}
