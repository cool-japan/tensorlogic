//! Benchmarking module for CLI
//!
//! Provides performance benchmarking for:
//! - Compilation speed
//! - Execution speed
//! - Optimization speed

use std::time::{Duration, Instant};

use anyhow::Result;
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{EinsumGraph, TLExpr};

use crate::executor::{Backend, CliExecutor, ExecutionConfig};
use crate::optimize::{optimize_einsum_graph, OptimizationConfig, OptimizationLevel};
use crate::output::{print_header, print_info};

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Compilation timings (in milliseconds)
    pub compilation_times: Vec<f64>,
    /// Execution timings (in milliseconds)
    pub execution_times: Vec<f64>,
    /// Optimization timings (in milliseconds)
    pub optimization_times: Vec<f64>,
}

impl BenchmarkResults {
    /// Create new empty results
    pub fn new() -> Self {
        Self {
            compilation_times: Vec::new(),
            execution_times: Vec::new(),
            optimization_times: Vec::new(),
        }
    }

    /// Calculate statistics for a timing series
    fn stats(times: &[f64]) -> (f64, f64, f64, f64) {
        if times.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let n = times.len() as f64;
        let mean = times.iter().sum::<f64>() / n;
        let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        (mean, std_dev, min, max)
    }

    /// Print benchmark summary
    pub fn print_summary(&self) {
        if !self.compilation_times.is_empty() {
            let (mean, std, min, max) = Self::stats(&self.compilation_times);
            println!("\nCompilation Benchmark:");
            println!("  Iterations: {}", self.compilation_times.len());
            println!("  Mean: {:.3} ms", mean);
            println!("  Std Dev: {:.3} ms", std);
            println!("  Min: {:.3} ms", min);
            println!("  Max: {:.3} ms", max);
            println!("  Throughput: {:.2} compilations/sec", 1000.0 / mean);
        }

        if !self.execution_times.is_empty() {
            let (mean, std, min, max) = Self::stats(&self.execution_times);
            println!("\nExecution Benchmark:");
            println!("  Iterations: {}", self.execution_times.len());
            println!("  Mean: {:.3} ms", mean);
            println!("  Std Dev: {:.3} ms", std);
            println!("  Min: {:.3} ms", min);
            println!("  Max: {:.3} ms", max);
            println!("  Throughput: {:.2} executions/sec", 1000.0 / mean);
        }

        if !self.optimization_times.is_empty() {
            let (mean, std, min, max) = Self::stats(&self.optimization_times);
            println!("\nOptimization Benchmark:");
            println!("  Iterations: {}", self.optimization_times.len());
            println!("  Mean: {:.3} ms", mean);
            println!("  Std Dev: {:.3} ms", std);
            println!("  Min: {:.3} ms", min);
            println!("  Max: {:.3} ms", max);
        }
    }

    /// Export results as JSON
    pub fn to_json(&self) -> serde_json::Value {
        let mut result = serde_json::Map::new();

        if !self.compilation_times.is_empty() {
            let (mean, std, min, max) = Self::stats(&self.compilation_times);
            result.insert(
                "compilation".to_string(),
                serde_json::json!({
                    "iterations": self.compilation_times.len(),
                    "mean_ms": mean,
                    "std_dev_ms": std,
                    "min_ms": min,
                    "max_ms": max,
                    "times_ms": self.compilation_times,
                }),
            );
        }

        if !self.execution_times.is_empty() {
            let (mean, std, min, max) = Self::stats(&self.execution_times);
            result.insert(
                "execution".to_string(),
                serde_json::json!({
                    "iterations": self.execution_times.len(),
                    "mean_ms": mean,
                    "std_dev_ms": std,
                    "min_ms": min,
                    "max_ms": max,
                    "times_ms": self.execution_times,
                }),
            );
        }

        if !self.optimization_times.is_empty() {
            let (mean, std, min, max) = Self::stats(&self.optimization_times);
            result.insert(
                "optimization".to_string(),
                serde_json::json!({
                    "iterations": self.optimization_times.len(),
                    "mean_ms": mean,
                    "std_dev_ms": std,
                    "min_ms": min,
                    "max_ms": max,
                    "times_ms": self.optimization_times,
                }),
            );
        }

        serde_json::Value::Object(result)
    }
}

impl Default for BenchmarkResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark runner
pub struct Benchmarker {
    iterations: usize,
    verbose: bool,
    quiet: bool,
}

impl Benchmarker {
    /// Create new benchmarker
    #[allow(dead_code)]
    pub fn new(iterations: usize, verbose: bool) -> Self {
        Self {
            iterations,
            verbose,
            quiet: false,
        }
    }

    /// Create benchmarker with quiet mode (for JSON output)
    pub fn with_quiet(iterations: usize, verbose: bool, quiet: bool) -> Self {
        Self {
            iterations,
            verbose,
            quiet,
        }
    }

    /// Run compilation benchmark
    pub fn benchmark_compilation(
        &self,
        expr: &TLExpr,
        context: &CompilerContext,
    ) -> Result<Vec<f64>> {
        let mut times = Vec::with_capacity(self.iterations);

        if !self.quiet {
            print_header("Benchmarking compilation...");
        }

        // Warmup run
        let mut ctx = context.clone();
        compile_to_einsum_with_context(expr, &mut ctx)?;

        for i in 0..self.iterations {
            let mut ctx = context.clone();
            let start = Instant::now();
            compile_to_einsum_with_context(expr, &mut ctx)?;
            let elapsed = start.elapsed();
            let ms = elapsed.as_secs_f64() * 1000.0;
            times.push(ms);

            if self.verbose && !self.quiet {
                print_info(&format!("  Iteration {}: {:.3} ms", i + 1, ms));
            }
        }

        Ok(times)
    }

    /// Run execution benchmark
    pub fn benchmark_execution(&self, graph: &EinsumGraph, backend: Backend) -> Result<Vec<f64>> {
        let mut times = Vec::with_capacity(self.iterations);

        if !self.quiet {
            print_header(&format!("Benchmarking execution ({})...", backend.name()));
        }

        let config = ExecutionConfig {
            backend,
            device: tensorlogic_scirs_backend::DeviceType::Cpu,
            show_metrics: false,
            show_intermediates: false,
            validate_shapes: false,
            trace: false,
        };

        let executor = CliExecutor::new(config)?;

        // Warmup run
        let _ = executor.execute(graph);

        for i in 0..self.iterations {
            let start = Instant::now();
            let _ = executor.execute(graph);
            let elapsed = start.elapsed();
            let ms = elapsed.as_secs_f64() * 1000.0;
            times.push(ms);

            if self.verbose && !self.quiet {
                print_info(&format!("  Iteration {}: {:.3} ms", i + 1, ms));
            }
        }

        Ok(times)
    }

    /// Run optimization benchmark
    pub fn benchmark_optimization(
        &self,
        expr: &TLExpr,
        context: &CompilerContext,
    ) -> Result<Vec<f64>> {
        let mut times = Vec::with_capacity(self.iterations);

        if !self.quiet {
            print_header("Benchmarking optimization...");
        }

        let opt_config = OptimizationConfig {
            level: OptimizationLevel::Basic, // Use Basic for benchmarking speed
            enable_dce: true,
            enable_cse: true,
            enable_identity: true,
            show_stats: false,
            verbose: false,
        };

        // Warmup run
        let mut ctx = context.clone();
        let graph = compile_to_einsum_with_context(expr, &mut ctx)?;
        let _ = optimize_einsum_graph(graph, &opt_config);

        for i in 0..self.iterations {
            let mut ctx = context.clone();
            let graph = compile_to_einsum_with_context(expr, &mut ctx)?;

            let start = Instant::now();
            let _ = optimize_einsum_graph(graph, &opt_config);
            let elapsed = start.elapsed();
            let ms = elapsed.as_secs_f64() * 1000.0;
            times.push(ms);

            if self.verbose && !self.quiet {
                print_info(&format!("  Iteration {}: {:.3} ms", i + 1, ms));
            }
        }

        Ok(times)
    }
}

/// Format duration for display
#[allow(dead_code)]
pub fn format_duration(duration: Duration) -> String {
    let ms = duration.as_secs_f64() * 1000.0;
    if ms < 1.0 {
        format!("{:.1} µs", ms * 1000.0)
    } else if ms < 1000.0 {
        format!("{:.3} ms", ms)
    } else {
        format!("{:.3} s", ms / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_results_stats() {
        let times = vec![10.0, 20.0, 30.0];
        let (mean, std, min, max) = BenchmarkResults::stats(&times);

        assert!((mean - 20.0).abs() < 0.001);
        assert!(std > 0.0);
        assert_eq!(min, 10.0);
        assert_eq!(max, 30.0);
    }

    #[test]
    fn test_benchmark_results_empty() {
        let times: Vec<f64> = vec![];
        let (mean, std, min, max) = BenchmarkResults::stats(&times);

        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_benchmark_results_json() {
        let mut results = BenchmarkResults::new();
        results.compilation_times = vec![10.0, 20.0, 30.0];

        let json = results.to_json();
        assert!(json.get("compilation").is_some());
    }
}
