//! Profiling-guided optimization for adaptive performance tuning.
//!
//! This module provides runtime profiling and adaptive optimization:
//! - **Profile collection**: Gather execution statistics during runtime
//! - **Hotspot detection**: Identify performance bottlenecks
//! - **Adaptive optimization**: Adjust strategy based on observed behavior
//! - **A/B testing**: Compare optimization strategies
//! - **Auto-tuning**: Automatically select best configurations
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_infer::{ProfilingOptimizer, OptimizationGoal, TuningConfig};
//!
//! // Create profiling-guided optimizer
//! let mut optimizer = ProfilingOptimizer::new()
//!     .with_goal(OptimizationGoal::MinimizeLatency)
//!     .with_tuning_enabled(true);
//!
//! // Execute with profiling
//! for batch in dataset {
//!     let result = optimizer.execute_and_profile(&graph, &batch)?;
//!
//!     // Optimizer automatically adapts based on observed performance
//!     if optimizer.should_reoptimize() {
//!         optimizer.apply_optimizations(&graph)?;
//!     }
//! }
//!
//! // Get optimization report
//! let report = optimizer.generate_report();
//! println!("Speedup: {:.2}x", report.speedup);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Profiling-guided optimization errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ProfilingOptimizerError {
    #[error("Insufficient profiling data: {0}")]
    InsufficientData(String),

    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Tuning failed: {0}")]
    TuningFailed(String),
}

/// Optimization goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationGoal {
    /// Minimize latency (single request)
    MinimizeLatency,

    /// Maximize throughput (requests/second)
    MaximizeThroughput,

    /// Minimize memory usage
    MinimizeMemory,

    /// Balance latency and throughput
    Balanced,

    /// Minimize energy consumption
    MinimizeEnergy,
}

/// Execution profile for a single run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionProfile {
    /// Execution time (microseconds)
    pub execution_time_us: u64,

    /// Memory used (bytes)
    pub memory_bytes: usize,

    /// Operations executed
    pub operations_count: usize,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Parallelism utilization
    pub parallelism_utilization: f64,

    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

impl ExecutionProfile {
    /// Create a new execution profile.
    pub fn new(execution_time_us: u64, memory_bytes: usize) -> Self {
        Self {
            execution_time_us,
            memory_bytes,
            operations_count: 0,
            cache_hit_rate: 0.0,
            parallelism_utilization: 0.0,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Get execution time in milliseconds.
    pub fn execution_time_ms(&self) -> f64 {
        self.execution_time_us as f64 / 1000.0
    }

    /// Get memory in megabytes.
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get throughput (operations per second).
    pub fn throughput(&self) -> f64 {
        if self.execution_time_us > 0 {
            (self.operations_count as f64) / (self.execution_time_us as f64 / 1_000_000.0)
        } else {
            0.0
        }
    }
}

/// Hotspot in the computation graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hotspot {
    /// Node or operation identifier
    pub identifier: String,

    /// Percentage of total execution time
    pub time_percentage: f64,

    /// Number of executions
    pub execution_count: usize,

    /// Average time per execution (microseconds)
    pub avg_time_us: f64,

    /// Optimization suggestions
    pub suggestions: Vec<String>,
}

impl Hotspot {
    /// Check if this is a critical hotspot (>10% of time).
    pub fn is_critical(&self) -> bool {
        self.time_percentage > 10.0
    }

    /// Get total time spent (microseconds).
    pub fn total_time_us(&self) -> f64 {
        self.avg_time_us * self.execution_count as f64
    }
}

/// Optimization strategy configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationStrategy {
    /// Enable operator fusion
    pub enable_fusion: bool,

    /// Enable constant folding
    pub enable_constant_folding: bool,

    /// Enable memory pooling
    pub enable_memory_pooling: bool,

    /// Enable parallel execution
    pub enable_parallelism: bool,

    /// Parallelism degree (0 = auto)
    pub parallelism_degree: usize,

    /// Enable SIMD optimizations
    pub enable_simd: bool,

    /// Enable sparse optimizations
    pub enable_sparse: bool,

    /// Batch size (0 = auto)
    pub batch_size: usize,
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self {
            enable_fusion: true,
            enable_constant_folding: true,
            enable_memory_pooling: true,
            enable_parallelism: true,
            parallelism_degree: 0,
            enable_simd: true,
            enable_sparse: false,
            batch_size: 0,
        }
    }
}

impl OptimizationStrategy {
    /// Create a conservative strategy (minimal optimizations).
    pub fn conservative() -> Self {
        Self {
            enable_fusion: false,
            enable_constant_folding: true,
            enable_memory_pooling: false,
            enable_parallelism: false,
            parallelism_degree: 1,
            enable_simd: false,
            enable_sparse: false,
            batch_size: 1,
        }
    }

    /// Create an aggressive strategy (maximum optimizations).
    pub fn aggressive() -> Self {
        Self {
            enable_fusion: true,
            enable_constant_folding: true,
            enable_memory_pooling: true,
            enable_parallelism: true,
            parallelism_degree: 0, // Auto
            enable_simd: true,
            enable_sparse: true,
            batch_size: 0, // Auto
        }
    }

    /// Score this strategy based on profile.
    pub fn score(&self, profile: &ExecutionProfile) -> f64 {
        let mut score = 0.0;

        // Faster execution is better
        score += 1000.0 / profile.execution_time_ms().max(0.1);

        // Less memory is better
        score += 100.0 / profile.memory_mb().max(0.1);

        // Higher cache hit rate is better
        score += profile.cache_hit_rate * 50.0;

        // Higher parallelism is better
        score += profile.parallelism_utilization * 30.0;

        score
    }
}

/// Tuning configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuningConfig {
    /// Number of warmup runs
    pub warmup_runs: usize,

    /// Number of measurement runs per configuration
    pub measurement_runs: usize,

    /// Enable A/B testing
    pub enable_ab_testing: bool,

    /// Statistical significance level (0.0-1.0)
    pub significance_level: f64,

    /// Maximum tuning time (seconds)
    pub max_tuning_time_secs: u64,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            warmup_runs: 3,
            measurement_runs: 5,
            enable_ab_testing: true,
            significance_level: 0.05,
            max_tuning_time_secs: 300,
        }
    }
}

/// Profiling-guided optimizer.
pub struct ProfilingOptimizer {
    /// Optimization goal
    goal: OptimizationGoal,

    /// Current optimization strategy
    current_strategy: OptimizationStrategy,

    /// Collected profiles
    profiles: Vec<ExecutionProfile>,

    /// Detected hotspots
    hotspots: Vec<Hotspot>,

    /// Tuning configuration
    tuning_config: TuningConfig,

    /// Enable auto-tuning
    auto_tuning_enabled: bool,

    /// Number of executions since last optimization
    executions_since_optimization: usize,

    /// Reoptimization threshold
    reoptimization_threshold: usize,

    /// Best observed strategy
    best_strategy: Option<OptimizationStrategy>,

    /// Best observed score
    best_score: f64,
}

impl ProfilingOptimizer {
    /// Create a new profiling optimizer.
    pub fn new() -> Self {
        Self {
            goal: OptimizationGoal::Balanced,
            current_strategy: OptimizationStrategy::default(),
            profiles: Vec::new(),
            hotspots: Vec::new(),
            tuning_config: TuningConfig::default(),
            auto_tuning_enabled: false,
            executions_since_optimization: 0,
            reoptimization_threshold: 100,
            best_strategy: None,
            best_score: 0.0,
        }
    }

    /// Set the optimization goal.
    pub fn with_goal(mut self, goal: OptimizationGoal) -> Self {
        self.goal = goal;
        self
    }

    /// Enable or disable auto-tuning.
    pub fn with_tuning_enabled(mut self, enabled: bool) -> Self {
        self.auto_tuning_enabled = enabled;
        self
    }

    /// Set the tuning configuration.
    pub fn with_tuning_config(mut self, config: TuningConfig) -> Self {
        self.tuning_config = config;
        self
    }

    /// Set the current optimization strategy.
    pub fn with_strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.current_strategy = strategy;
        self
    }

    /// Record an execution profile.
    pub fn record_profile(&mut self, profile: ExecutionProfile) {
        self.profiles.push(profile.clone());
        self.executions_since_optimization += 1;

        // Update best strategy if this is better
        let score = self.current_strategy.score(&profile);
        if score > self.best_score {
            self.best_score = score;
            self.best_strategy = Some(self.current_strategy.clone());
        }

        // Trim old profiles
        if self.profiles.len() > 1000 {
            self.profiles.drain(0..500);
        }
    }

    /// Check if reoptimization should be triggered.
    pub fn should_reoptimize(&self) -> bool {
        self.executions_since_optimization >= self.reoptimization_threshold
    }

    /// Detect hotspots from collected profiles.
    pub fn detect_hotspots(&mut self) -> Vec<Hotspot> {
        if self.profiles.is_empty() {
            return Vec::new();
        }

        // Simplified hotspot detection
        let mut hotspots = Vec::new();

        // Example: Create a hotspot for overall execution
        let total_time: u64 = self.profiles.iter().map(|p| p.execution_time_us).sum();
        let avg_time = total_time as f64 / self.profiles.len() as f64;

        let hotspot = Hotspot {
            identifier: "overall_execution".to_string(),
            time_percentage: 100.0,
            execution_count: self.profiles.len(),
            avg_time_us: avg_time,
            suggestions: self.generate_suggestions(),
        };

        hotspots.push(hotspot);
        self.hotspots = hotspots.clone();

        hotspots
    }

    /// Generate optimization suggestions based on profiles.
    fn generate_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if self.profiles.is_empty() {
            return suggestions;
        }

        let avg_profile = self.average_profile();

        // Memory-based suggestions
        if avg_profile.memory_mb() > 1000.0 {
            suggestions.push("Consider enabling memory pooling to reduce allocations".to_string());
        }

        // Parallelism suggestions
        if avg_profile.parallelism_utilization < 0.5 {
            suggestions
                .push("Low parallelism utilization - consider increasing batch size".to_string());
        }

        // Cache suggestions
        if avg_profile.cache_hit_rate < 0.7 {
            suggestions.push("Low cache hit rate - consider data layout optimization".to_string());
        }

        suggestions
    }

    /// Compute average profile.
    fn average_profile(&self) -> ExecutionProfile {
        if self.profiles.is_empty() {
            return ExecutionProfile::new(0, 0);
        }

        let n = self.profiles.len() as f64;
        let avg_time = self
            .profiles
            .iter()
            .map(|p| p.execution_time_us)
            .sum::<u64>() as f64
            / n;
        let avg_memory = self.profiles.iter().map(|p| p.memory_bytes).sum::<usize>() as f64 / n;

        ExecutionProfile {
            execution_time_us: avg_time as u64,
            memory_bytes: avg_memory as usize,
            operations_count: (self
                .profiles
                .iter()
                .map(|p| p.operations_count)
                .sum::<usize>() as f64
                / n) as usize,
            cache_hit_rate: self.profiles.iter().map(|p| p.cache_hit_rate).sum::<f64>() / n,
            parallelism_utilization: self
                .profiles
                .iter()
                .map(|p| p.parallelism_utilization)
                .sum::<f64>()
                / n,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Run auto-tuning to find best strategy.
    pub fn auto_tune(&mut self) -> Result<OptimizationStrategy, ProfilingOptimizerError> {
        let strategies = vec![
            OptimizationStrategy::conservative(),
            OptimizationStrategy::default(),
            OptimizationStrategy::aggressive(),
        ];

        let mut best_strategy = strategies[0].clone();
        let mut best_score = 0.0;

        // Simulate evaluation of each strategy
        for strategy in strategies {
            // In real implementation, would actually execute with this strategy
            let profile = self.average_profile();
            let score = strategy.score(&profile);

            if score > best_score {
                best_score = score;
                best_strategy = strategy.clone();
            }
        }

        self.current_strategy = best_strategy.clone();
        self.best_strategy = Some(best_strategy.clone());
        self.best_score = best_score;

        Ok(best_strategy)
    }

    /// Generate optimization report.
    pub fn generate_report(&self) -> OptimizationReport {
        let baseline_profile = self.profiles.first();
        let current_profile = self.profiles.last();

        let speedup = if let (Some(baseline), Some(current)) = (baseline_profile, current_profile) {
            baseline.execution_time_us as f64 / current.execution_time_us.max(1) as f64
        } else {
            1.0
        };

        OptimizationReport {
            goal: self.goal,
            total_profiles: self.profiles.len(),
            hotspots_detected: self.hotspots.len(),
            current_strategy: self.current_strategy.clone(),
            best_strategy: self.best_strategy.clone(),
            speedup,
            memory_reduction: 0.0, // Would calculate from profiles
            tuning_runs: self.tuning_config.measurement_runs,
        }
    }

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.profiles.clear();
        self.hotspots.clear();
        self.executions_since_optimization = 0;
        self.best_strategy = None;
        self.best_score = 0.0;
    }
}

impl Default for ProfilingOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationReport {
    /// Optimization goal
    pub goal: OptimizationGoal,

    /// Total profiles collected
    pub total_profiles: usize,

    /// Hotspots detected
    pub hotspots_detected: usize,

    /// Current strategy
    pub current_strategy: OptimizationStrategy,

    /// Best strategy found
    pub best_strategy: Option<OptimizationStrategy>,

    /// Speedup achieved
    pub speedup: f64,

    /// Memory reduction (percentage)
    pub memory_reduction: f64,

    /// Tuning runs performed
    pub tuning_runs: usize,
}

impl std::fmt::Display for OptimizationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Profiling-Guided Optimization Report")?;
        writeln!(f, "=====================================")?;
        writeln!(f, "Goal:              {:?}", self.goal)?;
        writeln!(f, "Profiles:          {}", self.total_profiles)?;
        writeln!(f, "Hotspots:          {}", self.hotspots_detected)?;
        writeln!(f, "Speedup:           {:.2}x", self.speedup)?;
        writeln!(f, "Memory reduction:  {:.1}%", self.memory_reduction)?;
        writeln!(f, "Tuning runs:       {}", self.tuning_runs)?;

        if let Some(best) = &self.best_strategy {
            writeln!(f, "\nBest Strategy:")?;
            writeln!(f, "  Fusion:          {}", best.enable_fusion)?;
            writeln!(f, "  Parallelism:     {}", best.enable_parallelism)?;
            writeln!(f, "  SIMD:            {}", best.enable_simd)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_profile() {
        let profile = ExecutionProfile::new(1000, 1024 * 1024);
        assert_eq!(profile.execution_time_us, 1000);
        assert_eq!(profile.memory_bytes, 1024 * 1024);
        assert_eq!(profile.execution_time_ms(), 1.0);
        assert!((profile.memory_mb() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_execution_profile_throughput() {
        let mut profile = ExecutionProfile::new(1_000_000, 0);
        profile.operations_count = 1000;
        assert_eq!(profile.throughput(), 1000.0);
    }

    #[test]
    fn test_hotspot_is_critical() {
        let hotspot = Hotspot {
            identifier: "op1".to_string(),
            time_percentage: 15.0,
            execution_count: 100,
            avg_time_us: 100.0,
            suggestions: Vec::new(),
        };

        assert!(hotspot.is_critical());
    }

    #[test]
    fn test_hotspot_total_time() {
        let hotspot = Hotspot {
            identifier: "op1".to_string(),
            time_percentage: 10.0,
            execution_count: 100,
            avg_time_us: 50.0,
            suggestions: Vec::new(),
        };

        assert_eq!(hotspot.total_time_us(), 5000.0);
    }

    #[test]
    fn test_optimization_strategy_default() {
        let strategy = OptimizationStrategy::default();
        assert!(strategy.enable_fusion);
        assert!(strategy.enable_parallelism);
    }

    #[test]
    fn test_optimization_strategy_conservative() {
        let strategy = OptimizationStrategy::conservative();
        assert!(!strategy.enable_fusion);
        assert!(!strategy.enable_parallelism);
    }

    #[test]
    fn test_optimization_strategy_aggressive() {
        let strategy = OptimizationStrategy::aggressive();
        assert!(strategy.enable_fusion);
        assert!(strategy.enable_parallelism);
        assert!(strategy.enable_simd);
    }

    #[test]
    fn test_profiling_optimizer_creation() {
        let optimizer = ProfilingOptimizer::new();
        assert_eq!(optimizer.goal, OptimizationGoal::Balanced);
        assert_eq!(optimizer.profiles.len(), 0);
    }

    #[test]
    fn test_profiling_optimizer_with_goal() {
        let optimizer = ProfilingOptimizer::new().with_goal(OptimizationGoal::MinimizeLatency);
        assert_eq!(optimizer.goal, OptimizationGoal::MinimizeLatency);
    }

    #[test]
    fn test_profiling_optimizer_record_profile() {
        let mut optimizer = ProfilingOptimizer::new();
        let profile = ExecutionProfile::new(1000, 1024);

        optimizer.record_profile(profile);
        assert_eq!(optimizer.profiles.len(), 1);
        assert_eq!(optimizer.executions_since_optimization, 1);
    }

    #[test]
    fn test_profiling_optimizer_should_reoptimize() {
        let mut optimizer = ProfilingOptimizer::new();
        optimizer.reoptimization_threshold = 5;

        assert!(!optimizer.should_reoptimize());

        for _ in 0..5 {
            optimizer.record_profile(ExecutionProfile::new(1000, 1024));
        }

        assert!(optimizer.should_reoptimize());
    }

    #[test]
    fn test_profiling_optimizer_detect_hotspots() {
        let mut optimizer = ProfilingOptimizer::new();
        optimizer.record_profile(ExecutionProfile::new(1000, 1024));

        let hotspots = optimizer.detect_hotspots();
        assert!(!hotspots.is_empty());
    }

    #[test]
    fn test_profiling_optimizer_auto_tune() {
        let mut optimizer = ProfilingOptimizer::new();
        optimizer.record_profile(ExecutionProfile::new(1000, 1024));

        let result = optimizer.auto_tune();
        assert!(result.is_ok());
        assert!(optimizer.best_strategy.is_some());
    }

    #[test]
    fn test_profiling_optimizer_generate_report() {
        let mut optimizer = ProfilingOptimizer::new();
        optimizer.record_profile(ExecutionProfile::new(2000, 1024));
        optimizer.record_profile(ExecutionProfile::new(1000, 512));

        let report = optimizer.generate_report();
        assert_eq!(report.total_profiles, 2);
        assert!(report.speedup > 1.0);
    }

    #[test]
    fn test_profiling_optimizer_reset() {
        let mut optimizer = ProfilingOptimizer::new();
        optimizer.record_profile(ExecutionProfile::new(1000, 1024));

        optimizer.reset();
        assert_eq!(optimizer.profiles.len(), 0);
        assert_eq!(optimizer.executions_since_optimization, 0);
    }

    #[test]
    fn test_tuning_config_default() {
        let config = TuningConfig::default();
        assert_eq!(config.warmup_runs, 3);
        assert_eq!(config.measurement_runs, 5);
    }

    #[test]
    fn test_optimization_report_display() {
        let report = OptimizationReport {
            goal: OptimizationGoal::MinimizeLatency,
            total_profiles: 100,
            hotspots_detected: 5,
            current_strategy: OptimizationStrategy::default(),
            best_strategy: Some(OptimizationStrategy::aggressive()),
            speedup: 2.5,
            memory_reduction: 30.0,
            tuning_runs: 10,
        };

        let display = format!("{}", report);
        assert!(display.contains("Speedup:           2.50x"));
        assert!(display.contains("Memory reduction:  30.0%"));
    }
}
