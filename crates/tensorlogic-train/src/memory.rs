//! Memory profiling and management utilities for training.
//!
//! This module provides tools to monitor and manage memory usage during training:
//! - Memory profiling with allocation tracking
//! - Gradient checkpointing for memory-efficient training
//! - Memory estimation utilities

use crate::{Callback, TrainResult, TrainingState};
use std::collections::HashMap;
use std::time::Instant;

/// Memory statistics for a training session.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Current allocated memory in bytes (estimated).
    pub current_allocated: usize,
    /// Peak allocated memory in bytes.
    pub peak_allocated: usize,
    /// Number of allocations tracked.
    pub allocation_count: usize,
    /// Memory usage history (epoch -> bytes).
    pub history: Vec<(usize, usize)>,
}

impl MemoryStats {
    /// Create new memory stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a memory measurement.
    pub fn record(&mut self, epoch: usize, bytes: usize) {
        self.current_allocated = bytes;
        if bytes > self.peak_allocated {
            self.peak_allocated = bytes;
        }
        self.allocation_count += 1;
        self.history.push((epoch, bytes));
    }

    /// Get memory usage as formatted string.
    pub fn format_bytes(bytes: usize) -> String {
        if bytes >= 1_073_741_824 {
            format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
        } else if bytes >= 1_048_576 {
            format!("{:.2} MB", bytes as f64 / 1_048_576.0)
        } else if bytes >= 1024 {
            format!("{:.2} KB", bytes as f64 / 1024.0)
        } else {
            format!("{} bytes", bytes)
        }
    }

    /// Get summary of memory usage.
    pub fn summary(&self) -> String {
        format!(
            "Memory: current={}, peak={}, allocations={}",
            Self::format_bytes(self.current_allocated),
            Self::format_bytes(self.peak_allocated),
            self.allocation_count
        )
    }
}

/// Gradient checkpointing configuration.
///
/// Gradient checkpointing reduces memory usage by recomputing activations
/// during the backward pass instead of storing them. This trades compute
/// for memory.
#[derive(Debug, Clone)]
pub struct GradientCheckpointConfig {
    /// Whether gradient checkpointing is enabled.
    pub enabled: bool,
    /// Checkpointing strategy.
    pub strategy: CheckpointStrategy,
    /// Layers to checkpoint (by name pattern).
    pub checkpoint_layers: Vec<String>,
    /// Memory threshold to trigger checkpointing (bytes).
    pub memory_threshold: Option<usize>,
}

impl Default for GradientCheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: CheckpointStrategy::Uniform,
            checkpoint_layers: Vec::new(),
            memory_threshold: None,
        }
    }
}

impl GradientCheckpointConfig {
    /// Create a new gradient checkpoint config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable gradient checkpointing.
    pub fn enabled(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Set checkpointing strategy.
    pub fn with_strategy(mut self, strategy: CheckpointStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set layers to checkpoint.
    pub fn with_layers(mut self, layers: Vec<String>) -> Self {
        self.checkpoint_layers = layers;
        self
    }

    /// Set memory threshold for automatic checkpointing.
    pub fn with_memory_threshold(mut self, threshold: usize) -> Self {
        self.memory_threshold = Some(threshold);
        self
    }
}

/// Gradient checkpointing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointStrategy {
    /// Checkpoint every N layers uniformly.
    Uniform,
    /// Checkpoint based on memory estimates.
    MemoryBased,
    /// Custom checkpointing (user-specified layers).
    Custom,
    /// Square root strategy (checkpoint sqrt(L) layers for L total).
    SqrtStrategy,
}

/// Memory profiler callback for tracking memory usage during training.
///
/// # Example
/// ```no_run
/// use tensorlogic_train::{MemoryProfilerCallback, Callback};
///
/// let mut profiler = MemoryProfilerCallback::new()
///     .with_epoch_tracking(true)
///     .with_batch_tracking(false);
///
/// // Use in training loop
/// ```
#[derive(Debug, Clone)]
pub struct MemoryProfilerCallback {
    /// Memory statistics.
    pub stats: MemoryStats,
    /// Whether to track at epoch level.
    track_epoch: bool,
    /// Whether to track at batch level.
    track_batch: bool,
    /// Logging frequency (every N epochs/batches).
    log_frequency: usize,
    /// Start time for duration tracking.
    start_time: Option<Instant>,
    /// Memory usage per batch in current epoch.
    batch_memory: Vec<usize>,
}

impl MemoryProfilerCallback {
    /// Create a new memory profiler callback.
    pub fn new() -> Self {
        Self {
            stats: MemoryStats::new(),
            track_epoch: true,
            track_batch: false,
            log_frequency: 1,
            start_time: None,
            batch_memory: Vec::new(),
        }
    }

    /// Enable epoch-level tracking.
    pub fn with_epoch_tracking(mut self, enabled: bool) -> Self {
        self.track_epoch = enabled;
        self
    }

    /// Enable batch-level tracking.
    pub fn with_batch_tracking(mut self, enabled: bool) -> Self {
        self.track_batch = enabled;
        self
    }

    /// Set logging frequency.
    pub fn with_log_frequency(mut self, frequency: usize) -> Self {
        self.log_frequency = frequency.max(1);
        self
    }

    /// Get memory statistics.
    pub fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Estimate current memory usage based on tensors.
    ///
    /// This is a simplified estimation - actual memory usage depends
    /// on allocator behavior, alignment, and fragmentation.
    pub fn estimate_tensor_memory(tensors: &[&[f64]]) -> usize {
        tensors.iter().map(|t| std::mem::size_of_val(*t)).sum()
    }

    /// Estimate memory for parameters.
    pub fn estimate_parameter_memory(
        parameters: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::IxDyn>>,
    ) -> usize {
        parameters.values().map(|p| p.len() * 8).sum()
    }

    /// Get memory usage report.
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Memory Profiling Report ===\n");
        report.push_str(&format!(
            "Current Memory: {}\n",
            MemoryStats::format_bytes(self.stats.current_allocated)
        ));
        report.push_str(&format!(
            "Peak Memory: {}\n",
            MemoryStats::format_bytes(self.stats.peak_allocated)
        ));
        report.push_str(&format!(
            "Total Allocations: {}\n",
            self.stats.allocation_count
        ));

        if !self.stats.history.is_empty() {
            report.push_str("\nMemory History:\n");
            for (epoch, bytes) in &self.stats.history {
                report.push_str(&format!(
                    "  Epoch {}: {}\n",
                    epoch,
                    MemoryStats::format_bytes(*bytes)
                ));
            }
        }

        report
    }
}

impl Default for MemoryProfilerCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl Callback for MemoryProfilerCallback {
    fn on_train_begin(&mut self, _state: &TrainingState) -> TrainResult<()> {
        self.start_time = Some(Instant::now());
        Ok(())
    }

    fn on_epoch_begin(&mut self, _epoch: usize, _state: &TrainingState) -> TrainResult<()> {
        self.batch_memory.clear();
        Ok(())
    }

    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        if !self.track_epoch {
            return Ok(());
        }

        // Estimate memory from state
        // In a real implementation, this would use system memory APIs
        let estimated_memory = estimate_training_memory(state);
        self.stats.record(epoch, estimated_memory);

        if epoch.is_multiple_of(self.log_frequency) {
            println!(
                "Epoch {}: Memory usage ~ {}",
                epoch,
                MemoryStats::format_bytes(estimated_memory)
            );
        }

        Ok(())
    }

    fn on_batch_end(&mut self, batch: usize, state: &TrainingState) -> TrainResult<()> {
        if !self.track_batch {
            return Ok(());
        }

        let estimated_memory = estimate_training_memory(state);
        self.batch_memory.push(estimated_memory);

        if batch.is_multiple_of(self.log_frequency) && self.log_frequency > 1 {
            println!(
                "  Batch {}: Memory ~ {}",
                batch,
                MemoryStats::format_bytes(estimated_memory)
            );
        }

        Ok(())
    }

    fn on_train_end(&mut self, _state: &TrainingState) -> TrainResult<()> {
        if let Some(start) = self.start_time {
            let duration = start.elapsed();
            println!("\n{}", self.report());
            println!("Training duration: {:.2?}", duration);
        }
        Ok(())
    }
}

/// Estimate training memory from state.
///
/// This is a rough estimate based on the training state.
/// Actual memory usage may differ significantly.
fn estimate_training_memory(state: &TrainingState) -> usize {
    // Base overhead for training state
    let base_overhead = 1024 * 1024; // 1 MB base

    // Estimate based on metrics count
    let metrics_memory = state.metrics.len() * 1024;

    // Total estimate
    base_overhead + metrics_memory
}

/// Memory-efficient training utilities.
pub struct MemoryEfficientTraining;

impl MemoryEfficientTraining {
    /// Calculate optimal batch size for available memory.
    ///
    /// # Arguments
    /// * `available_memory` - Available memory in bytes
    /// * `sample_size` - Size of a single sample in bytes
    /// * `model_memory` - Model memory footprint in bytes
    /// * `overhead_factor` - Overhead multiplier (typically 2-4 for gradients)
    pub fn optimal_batch_size(
        available_memory: usize,
        sample_size: usize,
        model_memory: usize,
        overhead_factor: f64,
    ) -> usize {
        let available_for_batch = available_memory.saturating_sub(model_memory);
        let sample_total = (sample_size as f64 * overhead_factor) as usize;

        if sample_total == 0 {
            return 1;
        }

        (available_for_batch / sample_total).max(1)
    }

    /// Estimate memory for a model with given parameter count.
    ///
    /// # Arguments
    /// * `num_parameters` - Number of model parameters
    /// * `with_gradients` - Whether to include gradient storage
    /// * `with_optimizer_state` - Whether to include optimizer state (e.g., Adam moments)
    pub fn estimate_model_memory(
        num_parameters: usize,
        with_gradients: bool,
        with_optimizer_state: bool,
    ) -> usize {
        let param_size = num_parameters * std::mem::size_of::<f64>();
        let mut total = param_size;

        if with_gradients {
            total += param_size; // Gradients same size as params
        }

        if with_optimizer_state {
            // Adam has 2 moment tensors per parameter
            total += param_size * 2;
        }

        total
    }

    /// Calculate gradient accumulation steps for target batch size.
    ///
    /// # Arguments
    /// * `target_batch_size` - Desired effective batch size
    /// * `actual_batch_size` - Batch size that fits in memory
    pub fn gradient_accumulation_steps(
        target_batch_size: usize,
        actual_batch_size: usize,
    ) -> usize {
        if actual_batch_size == 0 {
            return 1;
        }
        target_batch_size.div_ceil(actual_batch_size).max(1)
    }

    /// Get recommended memory settings for a given GPU memory.
    pub fn recommended_settings(gpu_memory_gb: f64) -> MemorySettings {
        let memory_bytes = (gpu_memory_gb * 1024.0 * 1024.0 * 1024.0) as usize;

        MemorySettings {
            max_batch_size: (memory_bytes / (100 * 1024 * 1024)).max(1), // ~100MB per batch
            use_gradient_checkpointing: gpu_memory_gb < 16.0,
            use_mixed_precision: gpu_memory_gb < 24.0,
            gradient_accumulation: if gpu_memory_gb < 8.0 { 4 } else { 1 },
        }
    }
}

/// Recommended memory settings.
#[derive(Debug, Clone)]
pub struct MemorySettings {
    /// Maximum recommended batch size.
    pub max_batch_size: usize,
    /// Whether to use gradient checkpointing.
    pub use_gradient_checkpointing: bool,
    /// Whether to use mixed precision.
    pub use_mixed_precision: bool,
    /// Gradient accumulation steps.
    pub gradient_accumulation: usize,
}

/// Memory budget manager for training.
#[derive(Debug, Clone)]
pub struct MemoryBudgetManager {
    /// Total memory budget in bytes.
    budget: usize,
    /// Current allocated memory.
    allocated: usize,
    /// Allocation tracking.
    allocations: HashMap<String, usize>,
}

impl MemoryBudgetManager {
    /// Create a new memory budget manager.
    ///
    /// # Arguments
    /// * `budget_bytes` - Total memory budget in bytes
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget: budget_bytes,
            allocated: 0,
            allocations: HashMap::new(),
        }
    }

    /// Create from GB specification.
    pub fn from_gb(gb: f64) -> Self {
        let bytes = (gb * 1024.0 * 1024.0 * 1024.0) as usize;
        Self::new(bytes)
    }

    /// Try to allocate memory.
    ///
    /// Returns true if allocation succeeded, false if would exceed budget.
    pub fn try_allocate(&mut self, name: &str, bytes: usize) -> bool {
        if self.allocated + bytes > self.budget {
            return false;
        }

        self.allocated += bytes;
        *self.allocations.entry(name.to_string()).or_default() += bytes;
        true
    }

    /// Free allocated memory.
    pub fn free(&mut self, name: &str) {
        if let Some(bytes) = self.allocations.remove(name) {
            self.allocated = self.allocated.saturating_sub(bytes);
        }
    }

    /// Get available memory.
    pub fn available(&self) -> usize {
        self.budget.saturating_sub(self.allocated)
    }

    /// Get utilization percentage.
    pub fn utilization(&self) -> f64 {
        if self.budget == 0 {
            return 0.0;
        }
        (self.allocated as f64 / self.budget as f64) * 100.0
    }

    /// Get allocation summary.
    pub fn summary(&self) -> String {
        format!(
            "Memory Budget: {:.2}% used ({} / {})",
            self.utilization(),
            MemoryStats::format_bytes(self.allocated),
            MemoryStats::format_bytes(self.budget)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new();

        stats.record(0, 1024 * 1024);
        stats.record(1, 2 * 1024 * 1024);
        stats.record(2, 1024 * 1024);

        assert_eq!(stats.current_allocated, 1024 * 1024);
        assert_eq!(stats.peak_allocated, 2 * 1024 * 1024);
        assert_eq!(stats.allocation_count, 3);
        assert_eq!(stats.history.len(), 3);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(MemoryStats::format_bytes(500), "500 bytes");
        assert_eq!(MemoryStats::format_bytes(2048), "2.00 KB");
        assert_eq!(MemoryStats::format_bytes(2 * 1024 * 1024), "2.00 MB");
        assert_eq!(MemoryStats::format_bytes(3 * 1024 * 1024 * 1024), "3.00 GB");
    }

    #[test]
    fn test_gradient_checkpoint_config() {
        let config = GradientCheckpointConfig::new()
            .enabled()
            .with_strategy(CheckpointStrategy::SqrtStrategy)
            .with_layers(vec!["layer1".to_string(), "layer2".to_string()]);

        assert!(config.enabled);
        assert_eq!(config.strategy, CheckpointStrategy::SqrtStrategy);
        assert_eq!(config.checkpoint_layers.len(), 2);
    }

    #[test]
    fn test_memory_profiler_callback() {
        let profiler = MemoryProfilerCallback::new()
            .with_epoch_tracking(true)
            .with_batch_tracking(false)
            .with_log_frequency(5);

        assert!(profiler.track_epoch);
        assert!(!profiler.track_batch);
        assert_eq!(profiler.log_frequency, 5);
    }

    #[test]
    fn test_optimal_batch_size() {
        // 8 GB available, 1 MB per sample, 1 GB model, 3x overhead
        let batch_size = MemoryEfficientTraining::optimal_batch_size(
            8 * 1024 * 1024 * 1024, // 8 GB
            1024 * 1024,            // 1 MB
            1024 * 1024 * 1024,     // 1 GB model
            3.0,                    // 3x overhead
        );

        // (8GB - 1GB) / (1MB * 3) = ~2333
        assert!(batch_size > 2000);
        assert!(batch_size < 2500);
    }

    #[test]
    fn test_estimate_model_memory() {
        let params = 1_000_000;

        // Just parameters
        let base = MemoryEfficientTraining::estimate_model_memory(params, false, false);
        assert_eq!(base, params * 8);

        // With gradients
        let with_grads = MemoryEfficientTraining::estimate_model_memory(params, true, false);
        assert_eq!(with_grads, params * 8 * 2);

        // With optimizer state (Adam)
        let with_adam = MemoryEfficientTraining::estimate_model_memory(params, true, true);
        assert_eq!(with_adam, params * 8 * 4);
    }

    #[test]
    fn test_gradient_accumulation_steps() {
        assert_eq!(
            MemoryEfficientTraining::gradient_accumulation_steps(64, 16),
            4
        );
        assert_eq!(
            MemoryEfficientTraining::gradient_accumulation_steps(100, 32),
            4 // ceil(100/32) = 4
        );
        assert_eq!(
            MemoryEfficientTraining::gradient_accumulation_steps(32, 32),
            1
        );
    }

    #[test]
    fn test_recommended_settings() {
        let small = MemoryEfficientTraining::recommended_settings(8.0);
        assert!(small.use_gradient_checkpointing);
        assert!(small.use_mixed_precision);

        let large = MemoryEfficientTraining::recommended_settings(32.0);
        assert!(!large.use_gradient_checkpointing);
        assert!(!large.use_mixed_precision);
    }

    #[test]
    fn test_memory_budget_manager() {
        let mut manager = MemoryBudgetManager::new(100 * 1024 * 1024); // 100 MB

        // Allocate 50 MB
        assert!(manager.try_allocate("model", 50 * 1024 * 1024));
        assert_eq!(manager.utilization(), 50.0);

        // Allocate another 30 MB
        assert!(manager.try_allocate("gradients", 30 * 1024 * 1024));
        assert_eq!(manager.utilization(), 80.0);

        // Try to allocate 30 MB more (should fail)
        assert!(!manager.try_allocate("overflow", 30 * 1024 * 1024));

        // Free gradients
        manager.free("gradients");
        assert_eq!(manager.utilization(), 50.0);

        // Now can allocate
        assert!(manager.try_allocate("new", 30 * 1024 * 1024));
    }

    #[test]
    fn test_memory_budget_from_gb() {
        let manager = MemoryBudgetManager::from_gb(4.0);
        assert_eq!(manager.budget, 4 * 1024 * 1024 * 1024);
    }
}
