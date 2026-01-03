//! Data models for benchmark tracking

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistical estimate with confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Estimate {
    pub point_estimate: f64,
    pub standard_error: f64,
    pub confidence_interval: ConfidenceInterval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub confidence_level: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

/// Criterion estimates JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionEstimates {
    pub mean: Estimate,
    pub median: Estimate,
    pub median_abs_dev: Estimate,
    pub slope: Option<Estimate>,
    pub std_dev: Estimate,
}

/// Benchmark result for a single parameter set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name (e.g., "e2e_simple_predicate")
    pub name: String,
    /// Parameter value (e.g., "10", "100")
    pub parameter: Option<String>,
    /// Statistical estimates
    pub estimates: CriterionEstimates,
    /// Timestamp when benchmark was run
    pub timestamp: DateTime<Utc>,
}

/// Collection of benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    /// Baseline name/tag
    pub name: String,
    /// When this baseline was created
    pub created_at: DateTime<Utc>,
    /// Git commit hash (if available)
    pub commit: Option<String>,
    /// All benchmark results, keyed by full benchmark ID
    pub results: HashMap<String, BenchmarkResult>,
}

impl BenchmarkSuite {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            created_at: Utc::now(),
            commit: Self::get_git_commit(),
            results: HashMap::new(),
        }
    }

    fn get_git_commit() -> Option<String> {
        use std::process::Command;

        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8(output.stdout)
                        .ok()
                        .map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
    }

    pub fn add_result(&mut self, result: BenchmarkResult) {
        let key = Self::result_key(&result.name, result.parameter.as_deref());
        self.results.insert(key, result);
    }

    pub fn get_result(&self, name: &str, parameter: Option<&str>) -> Option<&BenchmarkResult> {
        let key = Self::result_key(name, parameter);
        self.results.get(&key)
    }

    fn result_key(name: &str, parameter: Option<&str>) -> String {
        match parameter {
            Some(param) => format!("{}/{}", name, param),
            None => name.to_string(),
        }
    }
}

/// Comparison result between two benchmarks
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub name: String,
    pub parameter: Option<String>,
    pub baseline_mean: f64,
    pub current_mean: f64,
    pub change_percent: f64,
    pub is_regression: bool,
    pub is_improvement: bool,
}

impl ComparisonResult {
    pub fn new(baseline: &BenchmarkResult, current: &BenchmarkResult, threshold: f64) -> Self {
        let baseline_mean = baseline.estimates.mean.point_estimate;
        let current_mean = current.estimates.mean.point_estimate;
        let change_percent = ((current_mean - baseline_mean) / baseline_mean) * 100.0;

        let is_regression = change_percent > threshold;
        let is_improvement = change_percent < -threshold;

        Self {
            name: current.name.clone(),
            parameter: current.parameter.clone(),
            baseline_mean,
            current_mean,
            change_percent,
            is_regression,
            is_improvement,
        }
    }
}
