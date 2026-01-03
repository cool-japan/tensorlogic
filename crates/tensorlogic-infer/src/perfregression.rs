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

    /// Calculate percentile (0.0 to 100.0)
    pub fn percentile(&self, p: f64) -> Option<f64> {
        self.distribution.as_ref().and_then(|dist| {
            if dist.is_empty() || !(0.0..=100.0).contains(&p) {
                return None;
            }

            let mut sorted = dist.clone();
            sorted.sort_unstable();

            let index = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
            Some(sorted[index] as f64)
        })
    }

    /// Get P50 (median) from distribution
    pub fn p50(&self) -> Option<f64> {
        self.percentile(50.0)
    }

    /// Get P95 (95th percentile)
    pub fn p95(&self) -> Option<f64> {
        self.percentile(95.0)
    }

    /// Get P99 (99th percentile)
    pub fn p99(&self) -> Option<f64> {
        self.percentile(99.0)
    }

    /// Calculate 95% confidence interval for the mean
    pub fn confidence_interval_95(&self) -> (f64, f64) {
        // Using t-distribution approximation for 95% CI
        // t ≈ 1.96 for large samples (normal approximation)
        let margin = 1.96 * (self.std_dev_ns / (self.samples as f64).sqrt());
        (self.mean_ns - margin, self.mean_ns + margin)
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers(&self) -> Option<Vec<u64>> {
        self.distribution.as_ref().map(|dist| {
            let mut sorted = dist.clone();
            sorted.sort_unstable();

            let q1_idx = sorted.len() / 4;
            let q3_idx = 3 * sorted.len() / 4;
            let q1 = sorted[q1_idx] as f64;
            let q3 = sorted[q3_idx] as f64;
            let iqr = q3 - q1;

            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;

            dist.iter()
                .filter(|&&x| {
                    let val = x as f64;
                    val < lower_bound || val > upper_bound
                })
                .copied()
                .collect()
        })
    }

    /// Create a new BenchmarkStats with outliers removed
    pub fn without_outliers(&self) -> Option<Self> {
        self.distribution.as_ref().map(|dist| {
            let mut sorted = dist.clone();
            sorted.sort_unstable();

            let q1_idx = sorted.len() / 4;
            let q3_idx = 3 * sorted.len() / 4;
            let q1 = sorted[q1_idx] as f64;
            let q3 = sorted[q3_idx] as f64;
            let iqr = q3 - q1;

            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;

            let filtered: Vec<u64> = dist
                .iter()
                .filter(|&&x| {
                    let val = x as f64;
                    val >= lower_bound && val <= upper_bound
                })
                .copied()
                .collect();

            if filtered.is_empty() {
                // If all data is outliers, return original
                return self.clone();
            }

            BenchmarkStats::from_samples(self.name.clone(), filtered)
        })
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
    /// Statistical significance (p-value from Mann-Whitney U test, if distributions available)
    pub p_value: Option<f64>,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Whether the change is statistically significant (p < 0.05)
    pub is_significant: bool,
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

        // Calculate effect size (Cohen's d)
        let pooled_std = ((current.std_dev_ns.powi(2) + baseline.std_dev_ns.powi(2)) / 2.0).sqrt();
        let effect_size = if pooled_std > 0.0 {
            (current.mean_ns - baseline.mean_ns) / pooled_std
        } else {
            0.0
        };

        // Perform Mann-Whitney U test if distributions are available
        let p_value = match (&current.distribution, &baseline.distribution) {
            (Some(curr_dist), Some(base_dist)) => mann_whitney_u_test(curr_dist, base_dist),
            _ => None,
        };

        let is_significant = p_value.map(|p| p < 0.05).unwrap_or(false);

        BenchmarkComparison {
            name: current.name.clone(),
            is_regression: change_percent > regression_threshold,
            is_improvement: change_percent < -improvement_threshold,
            current,
            baseline,
            change_percent,
            p_value,
            effect_size,
            is_significant,
        }
    }

    /// Get effect size interpretation
    pub fn effect_size_interpretation(&self) -> &str {
        let abs_d = self.effect_size.abs();
        if abs_d < 0.2 {
            "negligible"
        } else if abs_d < 0.5 {
            "small"
        } else if abs_d < 0.8 {
            "medium"
        } else {
            "large"
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

/// Mann-Whitney U test for non-parametric comparison of two distributions
///
/// Returns the p-value for the two-sided test.
/// Lower p-value indicates stronger evidence that the distributions are different.
/// p < 0.05 is typically considered statistically significant.
fn mann_whitney_u_test(sample1: &[u64], sample2: &[u64]) -> Option<f64> {
    let n1 = sample1.len();
    let n2 = sample2.len();

    if n1 == 0 || n2 == 0 {
        return None;
    }

    // Combine and rank all values
    let mut combined: Vec<(u64, usize)> = Vec::new();
    for &val in sample1 {
        combined.push((val, 1)); // 1 for sample1
    }
    for &val in sample2 {
        combined.push((val, 2)); // 2 for sample2
    }

    // Sort by value
    combined.sort_unstable_by_key(|(val, _)| *val);

    // Assign ranks (average rank for ties)
    let mut ranks = vec![0.0; combined.len()];
    let mut i = 0;
    while i < combined.len() {
        let mut j = i;
        let current_value = combined[i].0;

        // Find all tied values
        while j < combined.len() && combined[j].0 == current_value {
            j += 1;
        }

        // Average rank for tied values
        let avg_rank = ((i + 1) + j) as f64 / 2.0;
        for rank in ranks.iter_mut().take(j).skip(i) {
            *rank = avg_rank;
        }

        i = j;
    }

    // Calculate U statistic for sample1
    let r1: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter(|((_, sample), _)| *sample == 1)
        .map(|(_, &rank)| rank)
        .sum();

    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;

    // Use smaller U
    let u = u1.min(u2);

    // Calculate z-score for large samples (normal approximation)
    // Valid when both n1 and n2 > 20
    if n1 > 20 && n2 > 20 {
        let mean_u = (n1 * n2) as f64 / 2.0;
        let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
        let z = (u - mean_u) / std_u;

        // Two-tailed p-value using normal distribution approximation
        // P(|Z| > |z|) = 2 * P(Z > |z|) = 2 * (1 - Φ(|z|))
        let abs_z = z.abs();
        let p = 2.0 * (1.0 - standard_normal_cdf(abs_z));
        Some(p)
    } else {
        // For small samples, we'd need exact tables or permutation tests
        // For now, return None (could be extended with exact tests)
        None
    }
}

/// Cumulative distribution function for standard normal distribution
/// Approximation using error function
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun formula 7.1.26)
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
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

    #[test]
    fn test_percentile_calculation() {
        let samples = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let mut stats = BenchmarkStats::from_samples("test".to_string(), samples.clone());
        stats.distribution = Some(samples);

        // Test various percentiles
        assert_eq!(stats.percentile(0.0), Some(10.0));
        // P50 should be close to 50-60 range (linear interpolation)
        assert!(stats.percentile(50.0).unwrap() >= 50.0 && stats.percentile(50.0).unwrap() <= 60.0);
        assert_eq!(stats.percentile(100.0), Some(100.0));

        // Test P50, P95, P99 exist
        assert!(stats.p50().is_some());
        assert!(stats.p95().is_some());
        assert!(stats.p99().is_some());
    }

    #[test]
    fn test_percentile_without_distribution() {
        let samples = vec![10, 20, 30];
        let stats = BenchmarkStats::from_samples("test".to_string(), samples);
        // No distribution saved
        assert_eq!(stats.p50(), None);
        assert_eq!(stats.p95(), None);
    }

    #[test]
    fn test_confidence_interval() {
        let samples = vec![100, 105, 110, 95, 102, 108, 97, 103];
        let stats = BenchmarkStats::from_samples("test".to_string(), samples);

        let (lower, upper) = stats.confidence_interval_95();
        assert!(lower < stats.mean_ns);
        assert!(upper > stats.mean_ns);
        assert!(upper - lower > 0.0); // CI should have non-zero width
    }

    #[test]
    fn test_outlier_detection() {
        // Create data with clear outliers
        let mut samples = vec![100; 20]; // Most values around 100
        samples.push(1000); // Clear outlier
        samples.push(2000); // Another outlier

        let mut stats = BenchmarkStats::from_samples("test".to_string(), samples.clone());
        stats.distribution = Some(samples);

        let outliers = stats.detect_outliers().unwrap();
        assert!(!outliers.is_empty());
        assert!(outliers.contains(&1000));
        assert!(outliers.contains(&2000));
    }

    #[test]
    fn test_without_outliers() {
        let mut samples = vec![100, 102, 98, 101, 99, 103, 97];
        samples.push(1000); // Add outlier

        let mut stats = BenchmarkStats::from_samples("test".to_string(), samples.clone());
        stats.distribution = Some(samples);

        let filtered = stats.without_outliers().unwrap();
        assert!(filtered.mean_ns < stats.mean_ns); // Mean should be lower without outlier
        assert!(filtered.std_dev_ns < stats.std_dev_ns); // Std dev should be lower
    }

    #[test]
    fn test_effect_size_calculation() {
        // Use samples with variation for meaningful std dev
        let baseline =
            BenchmarkStats::from_samples("test".to_string(), vec![95, 100, 105, 98, 102]);
        let current =
            BenchmarkStats::from_samples("test".to_string(), vec![105, 110, 115, 108, 112]);

        let comp = BenchmarkComparison::new(current, baseline, 5.0, 5.0);

        // Effect size should be positive (current is slower)
        assert!(comp.effect_size > 0.0);
    }

    #[test]
    fn test_effect_size_interpretation() {
        // Use samples with variation for meaningful effect size
        let baseline = BenchmarkStats::from_samples(
            "test".to_string(),
            vec![95, 98, 100, 102, 105, 97, 103, 99, 101, 104],
        );

        // Very small effect - minimal increase
        let current_small = BenchmarkStats::from_samples(
            "test".to_string(),
            vec![96, 99, 101, 103, 106, 98, 104, 100, 102, 105],
        );
        let comp_small = BenchmarkComparison::new(current_small, baseline.clone(), 5.0, 5.0);
        // Effect could be negligible or small depending on variation
        assert!(
            comp_small.effect_size.abs() < 1.0,
            "Effect size should be less than 1.0 for small differences"
        );

        // Large effect - significant increase (100 to 200 = doubling)
        let current_large = BenchmarkStats::from_samples(
            "test".to_string(),
            vec![195, 198, 200, 202, 205, 197, 203, 199, 201, 204],
        );
        let comp_large = BenchmarkComparison::new(current_large, baseline, 5.0, 5.0);
        assert_eq!(comp_large.effect_size_interpretation(), "large");
        assert!(comp_large.effect_size > 1.0);
    }

    #[test]
    fn test_mann_whitney_u_test_identical_distributions() {
        let sample1 = vec![100; 50];
        let sample2 = vec![100; 50];

        let p = mann_whitney_u_test(&sample1, &sample2);
        // Identical distributions should have high p-value (close to 1.0)
        assert!(p.is_some());
        assert!(p.unwrap() > 0.5);
    }

    #[test]
    fn test_mann_whitney_u_test_different_distributions() {
        let sample1 = vec![100; 50];
        let sample2 = vec![150; 50];

        let p = mann_whitney_u_test(&sample1, &sample2);
        // Very different distributions should have low p-value (close to 0.0)
        assert!(p.is_some());
        assert!(p.unwrap() < 0.05); // Statistically significant
    }

    #[test]
    fn test_mann_whitney_u_test_small_samples() {
        let sample1 = vec![100, 110, 105];
        let sample2 = vec![120, 125, 130];

        let p = mann_whitney_u_test(&sample1, &sample2);
        // Small samples (< 20) should return None
        assert!(p.is_none());
    }

    #[test]
    fn test_statistical_significance() {
        // Create distributions with large difference
        let baseline_samples: Vec<u64> = (0..100).map(|_| 100).collect();
        let current_samples: Vec<u64> = (0..100).map(|_| 150).collect();

        let mut baseline =
            BenchmarkStats::from_samples("test".to_string(), baseline_samples.clone());
        baseline.distribution = Some(baseline_samples);

        let mut current = BenchmarkStats::from_samples("test".to_string(), current_samples.clone());
        current.distribution = Some(current_samples);

        let comp = BenchmarkComparison::new(current, baseline, 5.0, 5.0);

        assert!(comp.is_significant); // Should be statistically significant
        assert!(comp.p_value.is_some());
        assert!(comp.p_value.unwrap() < 0.05);
    }

    #[test]
    fn test_erf_function() {
        // Test error function with known values
        assert!((erf(0.0) - 0.0).abs() < 0.01);
        assert!((erf(1.0) - 0.8427).abs() < 0.01);
        assert!((erf(-1.0) - (-0.8427)).abs() < 0.01);
    }

    #[test]
    fn test_standard_normal_cdf() {
        // Test CDF with known values
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!((standard_normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((standard_normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }
}
