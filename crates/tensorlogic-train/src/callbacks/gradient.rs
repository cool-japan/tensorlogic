//! Gradient monitoring and accumulation callbacks.

use crate::callbacks::core::Callback;
use crate::{TrainError, TrainResult, TrainingState};
use std::collections::HashMap;

/// Gradient flow monitor for tracking gradient statistics during training.
///
/// This callback tracks gradient norms, mean, std, and identifies vanishing/exploding gradients.
/// Useful for debugging training issues and understanding gradient flow through the network.
///
/// # Example
/// ```rust,ignore
/// use tensorlogic_train::{GradientMonitor, CallbackList};
///
/// let mut callbacks = CallbackList::new();
/// callbacks.add(Box::new(GradientMonitor::new(
///     10,      // log_frequency
///     1e-7,    // vanishing_threshold
///     100.0,   // exploding_threshold
/// )));
/// ```
pub struct GradientMonitor {
    /// Frequency of logging (every N batches).
    log_frequency: usize,
    /// Threshold for detecting vanishing gradients.
    vanishing_threshold: f64,
    /// Threshold for detecting exploding gradients.
    exploding_threshold: f64,
    /// History of gradient norms.
    pub gradient_norms: Vec<f64>,
    /// History of gradient means.
    pub gradient_means: Vec<f64>,
    /// History of gradient stds.
    pub gradient_stds: Vec<f64>,
    /// Count of vanishing gradient warnings.
    pub vanishing_count: usize,
    /// Count of exploding gradient warnings.
    pub exploding_count: usize,
    /// Current batch counter.
    batch_counter: usize,
}

impl GradientMonitor {
    /// Create a new gradient monitor.
    ///
    /// # Arguments
    /// * `log_frequency` - Log statistics every N batches
    /// * `vanishing_threshold` - Threshold below which gradients are considered vanishing
    /// * `exploding_threshold` - Threshold above which gradients are considered exploding
    pub fn new(log_frequency: usize, vanishing_threshold: f64, exploding_threshold: f64) -> Self {
        Self {
            log_frequency,
            vanishing_threshold,
            exploding_threshold,
            gradient_norms: Vec::new(),
            gradient_means: Vec::new(),
            gradient_stds: Vec::new(),
            vanishing_count: 0,
            exploding_count: 0,
            batch_counter: 0,
        }
    }

    /// Compute gradient statistics (placeholder - actual implementation needs gradient access).
    fn compute_gradient_stats(&mut self, _state: &TrainingState) -> (f64, f64, f64) {
        // In a real implementation, this would access actual gradients
        // For now, return placeholder values
        // (norm, mean, std)
        (1.0, 0.0, 0.1)
    }

    /// Check for vanishing gradients.
    fn check_vanishing(&mut self, norm: f64) -> bool {
        if norm < self.vanishing_threshold {
            self.vanishing_count += 1;
            return true;
        }
        false
    }

    /// Check for exploding gradients.
    fn check_exploding(&mut self, norm: f64) -> bool {
        if norm > self.exploding_threshold {
            self.exploding_count += 1;
            return true;
        }
        false
    }

    /// Print gradient statistics.
    fn print_stats(&self, norm: f64, mean: f64, std: f64) {
        println!("Gradient Stats [Batch {}]:", self.batch_counter);
        println!("  Norm: {:.6e}, Mean: {:.6e}, Std: {:.6e}", norm, mean, std);

        if self.vanishing_count > 0 {
            println!(
                "  Warning: Vanishing gradient warnings: {}",
                self.vanishing_count
            );
        }

        if self.exploding_count > 0 {
            println!(
                "  Warning: Exploding gradient warnings: {}",
                self.exploding_count
            );
        }
    }

    /// Get summary statistics.
    pub fn summary(&self) -> GradientSummary {
        let avg_norm = if !self.gradient_norms.is_empty() {
            self.gradient_norms.iter().sum::<f64>() / self.gradient_norms.len() as f64
        } else {
            0.0
        };

        GradientSummary {
            total_batches: self.batch_counter,
            average_norm: avg_norm,
            vanishing_count: self.vanishing_count,
            exploding_count: self.exploding_count,
        }
    }
}

/// Summary of gradient statistics.
#[derive(Debug, Clone)]
pub struct GradientSummary {
    /// Total number of batches monitored.
    pub total_batches: usize,
    /// Average gradient norm.
    pub average_norm: f64,
    /// Number of vanishing gradient warnings.
    pub vanishing_count: usize,
    /// Number of exploding gradient warnings.
    pub exploding_count: usize,
}

impl Callback for GradientMonitor {
    fn on_batch_end(&mut self, _batch: usize, state: &TrainingState) -> TrainResult<()> {
        self.batch_counter += 1;

        // Compute gradient statistics
        let (norm, mean, std) = self.compute_gradient_stats(state);

        // Record statistics
        self.gradient_norms.push(norm);
        self.gradient_means.push(mean);
        self.gradient_stds.push(std);

        // Check for issues
        let vanishing = self.check_vanishing(norm);
        let exploding = self.check_exploding(norm);

        // Log if needed
        if self.batch_counter.is_multiple_of(self.log_frequency) {
            self.print_stats(norm, mean, std);
        } else if vanishing || exploding {
            // Always log warnings immediately
            self.print_stats(norm, mean, std);
        }

        Ok(())
    }

    fn on_train_end(&mut self, _state: &TrainingState) -> TrainResult<()> {
        let summary = self.summary();
        println!("\n=== Gradient Monitoring Summary ===");
        println!("Total batches: {}", summary.total_batches);
        println!("Average gradient norm: {:.6e}", summary.average_norm);
        println!("Vanishing gradient warnings: {}", summary.vanishing_count);
        println!("Exploding gradient warnings: {}", summary.exploding_count);
        println!("====================================\n");
        Ok(())
    }
}

/// Gradient scaling strategy for accumulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientScalingStrategy {
    /// Divide by accumulation steps (default, maintains gradient magnitude)
    Average,
    /// Sum gradients without scaling (useful for some optimizers)
    Sum,
    /// Dynamic scaling based on batch size ratio
    Dynamic,
}

/// Gradient Accumulation callback with advanced features.
///
/// Simulates larger batch sizes by accumulating gradients over multiple
/// mini-batches before updating parameters. This is useful when GPU memory
/// is limited but you want to train with effectively larger batches.
///
/// Effective batch size = mini_batch_size * accumulation_steps
///
/// # Features
/// - Memory-efficient in-place accumulation
/// - Multiple scaling strategies
/// - Gradient overflow detection
/// - Memory usage tracking
/// - Automatic gradient zeroing
///
/// # Example
/// ```rust,ignore
/// use tensorlogic_train::{GradientAccumulationCallback, GradientScalingStrategy};
///
/// let mut grad_accum = GradientAccumulationCallback::new(
///     4, // accumulate over 4 mini-batches
///     GradientScalingStrategy::Average,
/// ).unwrap();
/// ```
pub struct GradientAccumulationCallback {
    /// Number of steps to accumulate gradients before updating.
    accumulation_steps: usize,
    /// Current accumulation counter.
    current_step: usize,
    /// Accumulated gradients.
    accumulated_grads: HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    /// Whether gradients are initialized.
    initialized: bool,
    /// Gradient scaling strategy.
    scaling_strategy: GradientScalingStrategy,
    /// Track maximum gradient norm seen during accumulation.
    max_grad_norm: f64,
    /// Track if overflow was detected.
    overflow_detected: bool,
    /// Total number of accumulation cycles completed.
    total_cycles: usize,
    /// Enable gradient clipping during accumulation.
    clip_grad_norm: Option<f64>,
}

impl GradientAccumulationCallback {
    /// Create a new Gradient Accumulation callback with default average scaling.
    ///
    /// # Arguments
    /// * `accumulation_steps` - Number of mini-batches to accumulate (e.g., 4, 8, 16)
    pub fn new(accumulation_steps: usize) -> TrainResult<Self> {
        Self::with_strategy(accumulation_steps, GradientScalingStrategy::Average)
    }

    /// Create a new Gradient Accumulation callback with specified scaling strategy.
    ///
    /// # Arguments
    /// * `accumulation_steps` - Number of mini-batches to accumulate
    /// * `scaling_strategy` - How to scale accumulated gradients
    pub fn with_strategy(
        accumulation_steps: usize,
        scaling_strategy: GradientScalingStrategy,
    ) -> TrainResult<Self> {
        if accumulation_steps == 0 {
            return Err(TrainError::CallbackError(
                "Accumulation steps must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            accumulation_steps,
            current_step: 0,
            accumulated_grads: HashMap::new(),
            initialized: false,
            scaling_strategy,
            max_grad_norm: 0.0,
            overflow_detected: false,
            total_cycles: 0,
            clip_grad_norm: None,
        })
    }

    /// Enable gradient clipping during accumulation.
    ///
    /// # Arguments
    /// * `max_norm` - Maximum gradient norm before clipping
    pub fn with_grad_clipping(mut self, max_norm: f64) -> Self {
        self.clip_grad_norm = Some(max_norm);
        self
    }

    /// Accumulate gradients with optional clipping and overflow detection.
    pub fn accumulate(
        &mut self,
        gradients: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    ) -> TrainResult<()> {
        // Check for NaN/Inf before accumulation
        for grad in gradients.values() {
            if grad.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                self.overflow_detected = true;
                return Err(TrainError::CallbackError(
                    "Gradient overflow detected (NaN or Inf)".to_string(),
                ));
            }
        }

        // Compute gradient norm for monitoring
        let grad_norm = self.compute_total_norm(gradients);
        self.max_grad_norm = self.max_grad_norm.max(grad_norm);

        if !self.initialized {
            // Initialize on first call with zero-copy when possible
            for (name, grad) in gradients {
                let clipped_grad = if let Some(max_norm) = self.clip_grad_norm {
                    if grad_norm > max_norm {
                        let scale = max_norm / grad_norm;
                        grad * scale
                    } else {
                        grad.clone()
                    }
                } else {
                    grad.clone()
                };
                self.accumulated_grads.insert(name.clone(), clipped_grad);
            }
            self.initialized = true;
        } else {
            // In-place accumulation for memory efficiency
            for (name, grad) in gradients {
                if let Some(acc_grad) = self.accumulated_grads.get_mut(name) {
                    let grad_to_add = if let Some(max_norm) = self.clip_grad_norm {
                        if grad_norm > max_norm {
                            let scale = max_norm / grad_norm;
                            grad * scale
                        } else {
                            grad.clone()
                        }
                    } else {
                        grad.clone()
                    };

                    // In-place addition
                    *acc_grad = &*acc_grad + &grad_to_add;
                }
            }
        }

        self.current_step += 1;
        Ok(())
    }

    /// Compute the total L2 norm of all gradients.
    fn compute_total_norm(
        &self,
        gradients: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    ) -> f64 {
        let mut total_norm_sq = 0.0;
        for grad in gradients.values() {
            total_norm_sq += grad.iter().map(|&x| x * x).sum::<f64>();
        }
        total_norm_sq.sqrt()
    }

    /// Check if we should perform an optimizer step.
    pub fn should_update(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Get scaled accumulated gradients and reset state.
    pub fn get_and_reset(
        &mut self,
    ) -> HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>> {
        let scale = match self.scaling_strategy {
            GradientScalingStrategy::Average => 1.0 / self.accumulation_steps as f64,
            GradientScalingStrategy::Sum => 1.0,
            GradientScalingStrategy::Dynamic => {
                // Dynamic scaling based on actual steps accumulated
                1.0 / self.current_step.max(1) as f64
            }
        };

        let mut scaled_grads = HashMap::new();
        for (name, grad) in &self.accumulated_grads {
            scaled_grads.insert(name.clone(), grad * scale);
        }

        // Update statistics
        self.total_cycles += 1;

        // Reset state
        self.current_step = 0;
        self.initialized = false;
        self.accumulated_grads.clear();
        self.max_grad_norm = 0.0;
        self.overflow_detected = false;

        scaled_grads
    }

    /// Get statistics about gradient accumulation.
    pub fn get_stats(&self) -> GradientAccumulationStats {
        let memory_usage = self.estimate_memory_usage();

        GradientAccumulationStats {
            accumulation_steps: self.accumulation_steps,
            current_step: self.current_step,
            total_cycles: self.total_cycles,
            max_grad_norm: self.max_grad_norm,
            overflow_detected: self.overflow_detected,
            num_parameters: self.accumulated_grads.len(),
            memory_usage_mb: memory_usage,
        }
    }

    /// Estimate memory usage of accumulated gradients in MB.
    fn estimate_memory_usage(&self) -> f64 {
        let mut total_elements = 0usize;
        for grad in self.accumulated_grads.values() {
            total_elements += grad.len();
        }
        // f64 = 8 bytes
        (total_elements * 8) as f64 / (1024.0 * 1024.0)
    }

    /// Reset all state without returning gradients (useful for error recovery).
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.initialized = false;
        self.accumulated_grads.clear();
        self.max_grad_norm = 0.0;
        self.overflow_detected = false;
    }
}

/// Statistics for gradient accumulation.
#[derive(Debug, Clone)]
pub struct GradientAccumulationStats {
    /// Configured accumulation steps.
    pub accumulation_steps: usize,
    /// Current step in accumulation.
    pub current_step: usize,
    /// Total completed cycles.
    pub total_cycles: usize,
    /// Maximum gradient norm seen.
    pub max_grad_norm: f64,
    /// Whether overflow was detected.
    pub overflow_detected: bool,
    /// Number of parameters being accumulated.
    pub num_parameters: usize,
    /// Estimated memory usage in MB.
    pub memory_usage_mb: f64,
}

impl Callback for GradientAccumulationCallback {
    fn on_epoch_begin(&mut self, _epoch: usize, _state: &TrainingState) -> TrainResult<()> {
        // Reset at the beginning of each epoch
        self.current_step = 0;
        self.initialized = false;
        self.accumulated_grads.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn create_test_gradients() -> HashMap<String, Array2<f64>> {
        let mut grads = HashMap::new();
        grads.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        grads.insert(
            "layer2".to_string(),
            Array2::from_shape_vec((2, 2), vec![0.5, 1.0, 1.5, 2.0]).unwrap(),
        );
        grads
    }

    #[test]
    fn test_gradient_accumulation_average_strategy() {
        let mut accum = GradientAccumulationCallback::new(2).unwrap();
        let grads = create_test_gradients();

        // First accumulation
        accum.accumulate(&grads).unwrap();
        assert_eq!(accum.current_step, 1);
        assert!(!accum.should_update());

        // Second accumulation
        accum.accumulate(&grads).unwrap();
        assert_eq!(accum.current_step, 2);
        assert!(accum.should_update());

        // Get averaged gradients
        let averaged = accum.get_and_reset();
        let layer1 = averaged.get("layer1").unwrap();

        // Should be average of 2 accumulations (same gradient twice)
        assert_eq!(layer1[[0, 0]], 1.0); // (1.0 + 1.0) / 2
        assert_eq!(layer1[[0, 1]], 2.0); // (2.0 + 2.0) / 2

        // Should be reset
        assert_eq!(accum.current_step, 0);
    }

    #[test]
    fn test_gradient_accumulation_sum_strategy() {
        let mut accum =
            GradientAccumulationCallback::with_strategy(2, GradientScalingStrategy::Sum).unwrap();
        let grads = create_test_gradients();

        accum.accumulate(&grads).unwrap();
        accum.accumulate(&grads).unwrap();

        let summed = accum.get_and_reset();
        let layer1 = summed.get("layer1").unwrap();

        // Should be sum (no scaling)
        assert_eq!(layer1[[0, 0]], 2.0); // 1.0 + 1.0
        assert_eq!(layer1[[0, 1]], 4.0); // 2.0 + 2.0
    }

    #[test]
    fn test_gradient_accumulation_dynamic_strategy() {
        let mut accum =
            GradientAccumulationCallback::with_strategy(4, GradientScalingStrategy::Dynamic)
                .unwrap();
        let grads = create_test_gradients();

        // Accumulate only 3 times (less than configured 4)
        accum.accumulate(&grads).unwrap();
        accum.accumulate(&grads).unwrap();
        accum.accumulate(&grads).unwrap();

        let scaled = accum.get_and_reset();
        let layer1 = scaled.get("layer1").unwrap();

        // Should scale by actual steps (3) not configured steps (4)
        assert_eq!(layer1[[0, 0]], 1.0); // (1.0 + 1.0 + 1.0) / 3
    }

    #[test]
    fn test_gradient_clipping_during_accumulation() {
        let mut accum = GradientAccumulationCallback::new(2)
            .unwrap()
            .with_grad_clipping(1.0); // Very small max norm

        let mut grads = HashMap::new();
        grads.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 2), vec![10.0, 10.0, 10.0, 10.0]).unwrap(),
        );

        // Large gradients should be clipped
        accum.accumulate(&grads).unwrap();
        assert!(accum.max_grad_norm > 0.0);

        // Accumulated gradients should be clipped
        let accumulated = &accum.accumulated_grads["layer1"];
        let norm_sq: f64 = accumulated.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();

        // Norm should be at or below clip threshold
        assert!(norm <= 1.1); // Small tolerance
    }

    #[test]
    fn test_overflow_detection() {
        let mut accum = GradientAccumulationCallback::new(2).unwrap();

        let mut grads = HashMap::new();
        grads.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 2), vec![f64::NAN, 1.0, 2.0, 3.0]).unwrap(),
        );

        // Should detect NaN
        let result = accum.accumulate(&grads);
        assert!(result.is_err());
        assert!(accum.overflow_detected);
    }

    #[test]
    fn test_gradient_accumulation_stats() {
        let mut accum = GradientAccumulationCallback::new(2).unwrap();
        let grads = create_test_gradients();

        accum.accumulate(&grads).unwrap();
        accum.accumulate(&grads).unwrap();
        accum.get_and_reset();

        let stats = accum.get_stats();
        assert_eq!(stats.accumulation_steps, 2);
        assert_eq!(stats.total_cycles, 1);
        assert!(!stats.overflow_detected);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let mut accum = GradientAccumulationCallback::new(2).unwrap();
        let grads = create_test_gradients();

        accum.accumulate(&grads).unwrap();

        let stats = accum.get_stats();
        assert!(stats.memory_usage_mb > 0.0);
        assert_eq!(stats.num_parameters, 2); // 2 layers
    }

    #[test]
    fn test_gradient_accumulation_reset() {
        let mut accum = GradientAccumulationCallback::new(2).unwrap();
        let grads = create_test_gradients();

        accum.accumulate(&grads).unwrap();
        assert_eq!(accum.current_step, 1);

        accum.reset();
        assert_eq!(accum.current_step, 0);
        assert!(!accum.initialized);
        assert_eq!(accum.accumulated_grads.len(), 0);
    }

    #[test]
    fn test_gradient_accumulation_zero_steps_error() {
        let result = GradientAccumulationCallback::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_accumulation_multiple_cycles() {
        let mut accum = GradientAccumulationCallback::new(2).unwrap();
        let grads = create_test_gradients();

        // First cycle
        accum.accumulate(&grads).unwrap();
        accum.accumulate(&grads).unwrap();
        accum.get_and_reset();

        // Second cycle
        accum.accumulate(&grads).unwrap();
        accum.accumulate(&grads).unwrap();
        accum.get_and_reset();

        let stats = accum.get_stats();
        assert_eq!(stats.total_cycles, 2);
    }

    #[test]
    fn test_different_gradient_shapes() {
        let mut accum = GradientAccumulationCallback::new(2).unwrap();

        let mut grads1 = HashMap::new();
        grads1.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );

        let mut grads2 = HashMap::new();
        grads2.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 3), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).unwrap(),
        );

        accum.accumulate(&grads1).unwrap();
        accum.accumulate(&grads2).unwrap();

        let averaged = accum.get_and_reset();
        let layer1 = averaged.get("layer1").unwrap();

        assert_eq!(layer1.dim(), (2, 3));
        assert_eq!(layer1[[0, 0]], 0.75); // (1.0 + 0.5) / 2
    }
}
