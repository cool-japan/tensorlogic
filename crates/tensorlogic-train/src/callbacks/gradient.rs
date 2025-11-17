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

/// Gradient Accumulation callback.
///
/// Simulates larger batch sizes by accumulating gradients over multiple
/// mini-batches before updating parameters. This is useful when GPU memory
/// is limited but you want to train with effectively larger batches.
///
/// Effective batch size = mini_batch_size * accumulation_steps
pub struct GradientAccumulationCallback {
    /// Number of steps to accumulate gradients before updating.
    accumulation_steps: usize,
    /// Current accumulation counter.
    current_step: usize,
    /// Accumulated gradients.
    accumulated_grads: HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    /// Whether gradients are initialized.
    initialized: bool,
}

impl GradientAccumulationCallback {
    /// Create a new Gradient Accumulation callback.
    ///
    /// # Arguments
    /// * `accumulation_steps` - Number of mini-batches to accumulate (e.g., 4, 8, 16)
    pub fn new(accumulation_steps: usize) -> TrainResult<Self> {
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
        })
    }

    /// Accumulate gradients.
    pub fn accumulate(
        &mut self,
        gradients: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    ) -> TrainResult<()> {
        if !self.initialized {
            // Initialize on first call
            for (name, grad) in gradients {
                self.accumulated_grads.insert(name.clone(), grad.clone());
            }
            self.initialized = true;
        } else {
            // Accumulate
            for (name, grad) in gradients {
                if let Some(acc_grad) = self.accumulated_grads.get_mut(name) {
                    *acc_grad = &*acc_grad + grad;
                }
            }
        }

        self.current_step += 1;
        Ok(())
    }

    /// Check if we should perform an optimizer step.
    pub fn should_update(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Get averaged accumulated gradients and reset.
    pub fn get_and_reset(
        &mut self,
    ) -> HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>> {
        let scale = 1.0 / self.accumulation_steps as f64;

        let mut averaged_grads = HashMap::new();
        for (name, grad) in &self.accumulated_grads {
            averaged_grads.insert(name.clone(), grad * scale);
        }

        // Reset
        self.current_step = 0;
        self.initialized = false;
        self.accumulated_grads.clear();

        averaged_grads
    }
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
