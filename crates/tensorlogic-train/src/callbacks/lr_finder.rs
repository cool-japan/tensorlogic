//! Learning rate finder callback using the LR range test.

use crate::callbacks::core::Callback;
use crate::{TrainResult, TrainingState};

/// Learning rate finder callback using the LR range test.
///
/// This callback implements the learning rate range test proposed by Leslie N. Smith.
/// It gradually increases the learning rate from a minimum to a maximum value over
/// a specified number of iterations/epochs and tracks the loss at each step.
///
/// The optimal learning rate is typically found just before the loss starts to increase.
///
/// # Example
/// ```rust,ignore
/// use tensorlogic_train::{LearningRateFinder, CallbackList};
///
/// let mut callbacks = CallbackList::new();
/// callbacks.add(Box::new(LearningRateFinder::new(
///     1e-7,   // start_lr
///     10.0,   // end_lr
///     100,    // num_steps
/// )));
/// ```
pub struct LearningRateFinder {
    /// Starting learning rate.
    start_lr: f64,
    /// Ending learning rate.
    end_lr: f64,
    /// Number of steps to test.
    num_steps: usize,
    /// Current step.
    current_step: usize,
    /// History of (lr, loss) pairs.
    pub history: Vec<(f64, f64)>,
    /// Whether to use exponential or linear scaling.
    exponential: bool,
    /// Smoothing factor for loss (0.0 = no smoothing, 0.9 = heavy smoothing).
    smoothing: f64,
    /// Smoothed loss.
    smoothed_loss: Option<f64>,
}

impl LearningRateFinder {
    /// Create a new learning rate finder.
    ///
    /// # Arguments
    /// * `start_lr` - Starting learning rate (e.g., 1e-7)
    /// * `end_lr` - Ending learning rate (e.g., 10.0)
    /// * `num_steps` - Number of steps to test
    pub fn new(start_lr: f64, end_lr: f64, num_steps: usize) -> Self {
        Self {
            start_lr,
            end_lr,
            num_steps,
            current_step: 0,
            history: Vec::with_capacity(num_steps),
            exponential: true, // Exponential scaling is recommended
            smoothing: 0.0,    // No smoothing by default
            smoothed_loss: None,
        }
    }

    /// Enable exponential scaling (recommended, default).
    pub fn with_exponential_scaling(mut self) -> Self {
        self.exponential = true;
        self
    }

    /// Enable linear scaling.
    pub fn with_linear_scaling(mut self) -> Self {
        self.exponential = false;
        self
    }

    /// Set loss smoothing factor (0.0-1.0).
    ///
    /// Recommended: 0.9 for noisy losses, 0.0 for smooth losses.
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing.clamp(0.0, 1.0);
        self
    }

    /// Compute the current learning rate based on step.
    fn compute_lr(&self) -> f64 {
        if self.num_steps <= 1 {
            return self.start_lr;
        }

        let step_ratio = self.current_step as f64 / (self.num_steps - 1) as f64;

        if self.exponential {
            // Exponential scaling: lr = start_lr * (end_lr/start_lr)^step_ratio
            self.start_lr * (self.end_lr / self.start_lr).powf(step_ratio)
        } else {
            // Linear scaling: lr = start_lr + (end_lr - start_lr) * step_ratio
            self.start_lr + (self.end_lr - self.start_lr) * step_ratio
        }
    }

    /// Get the smoothed loss.
    fn smooth_loss(&mut self, loss: f64) -> f64 {
        if self.smoothing == 0.0 {
            return loss;
        }

        match self.smoothed_loss {
            None => {
                self.smoothed_loss = Some(loss);
                loss
            }
            Some(prev) => {
                let smoothed = self.smoothing * prev + (1.0 - self.smoothing) * loss;
                self.smoothed_loss = Some(smoothed);
                smoothed
            }
        }
    }

    /// Find the suggested optimal learning rate.
    ///
    /// Returns the learning rate with the steepest negative gradient (fastest decrease in loss).
    pub fn suggest_lr(&self) -> Option<f64> {
        if self.history.len() < 3 {
            return None;
        }

        let mut best_lr = None;
        let mut best_gradient = f64::INFINITY;

        // Compute gradients and find steepest descent
        for i in 1..self.history.len() {
            let (lr1, loss1) = self.history[i - 1];
            let (lr2, loss2) = self.history[i];

            let gradient = (loss2 - loss1) / (lr2 - lr1);

            if gradient < best_gradient {
                best_gradient = gradient;
                best_lr = Some(lr2);
            }
        }

        best_lr
    }

    /// Print the LR finder results.
    pub fn print_results(&self) {
        println!("\n=== Learning Rate Finder Results ===");
        println!(
            "Tested {} learning rates from {:.2e} to {:.2e}",
            self.history.len(),
            self.start_lr,
            self.end_lr
        );

        if let Some(suggested_lr) = self.suggest_lr() {
            println!("Suggested optimal LR: {:.2e}", suggested_lr);
            println!(
                "Consider using LR between {:.2e} and {:.2e}",
                suggested_lr / 10.0,
                suggested_lr
            );
        }

        println!("\nLR, Loss:");
        for (lr, loss) in &self.history {
            println!("{:.6e}, {:.6}", lr, loss);
        }
        println!("===================================\n");
    }
}

impl Callback for LearningRateFinder {
    fn on_batch_end(&mut self, _batch: usize, state: &TrainingState) -> TrainResult<()> {
        if self.current_step >= self.num_steps {
            return Ok(());
        }

        // Get current loss and smooth it
        let loss = self.smooth_loss(state.batch_loss);

        // Record (lr, loss) pair
        let lr = self.compute_lr();
        self.history.push((lr, loss));

        self.current_step += 1;

        // Note: The actual LR update happens via the trainer's optimizer
        // This callback just tracks the relationship

        Ok(())
    }

    fn should_stop(&self) -> bool {
        // Stop after testing all LR values
        self.current_step >= self.num_steps
    }
}
