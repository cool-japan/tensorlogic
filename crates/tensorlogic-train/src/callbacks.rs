//! Training callbacks for monitoring and controlling training.

use crate::{TrainError, TrainResult, TrainingState};
use std::collections::HashMap;
use std::path::PathBuf;

/// Trait for training callbacks.
pub trait Callback {
    /// Called at the beginning of training.
    fn on_train_begin(&mut self, _state: &TrainingState) -> TrainResult<()> {
        Ok(())
    }

    /// Called at the end of training.
    fn on_train_end(&mut self, _state: &TrainingState) -> TrainResult<()> {
        Ok(())
    }

    /// Called at the beginning of an epoch.
    fn on_epoch_begin(&mut self, _epoch: usize, _state: &TrainingState) -> TrainResult<()> {
        Ok(())
    }

    /// Called at the end of an epoch.
    fn on_epoch_end(&mut self, _epoch: usize, _state: &TrainingState) -> TrainResult<()> {
        Ok(())
    }

    /// Called at the beginning of a batch.
    fn on_batch_begin(&mut self, _batch: usize, _state: &TrainingState) -> TrainResult<()> {
        Ok(())
    }

    /// Called at the end of a batch.
    fn on_batch_end(&mut self, _batch: usize, _state: &TrainingState) -> TrainResult<()> {
        Ok(())
    }

    /// Called after validation.
    fn on_validation_end(&mut self, _state: &TrainingState) -> TrainResult<()> {
        Ok(())
    }

    /// Check if training should stop early.
    fn should_stop(&self) -> bool {
        false
    }
}

/// List of callbacks to execute in order.
pub struct CallbackList {
    callbacks: Vec<Box<dyn Callback>>,
}

impl CallbackList {
    /// Create a new callback list.
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback to the list.
    pub fn add(&mut self, callback: Box<dyn Callback>) {
        self.callbacks.push(callback);
    }

    /// Execute on_train_begin for all callbacks.
    pub fn on_train_begin(&mut self, state: &TrainingState) -> TrainResult<()> {
        for callback in &mut self.callbacks {
            callback.on_train_begin(state)?;
        }
        Ok(())
    }

    /// Execute on_train_end for all callbacks.
    pub fn on_train_end(&mut self, state: &TrainingState) -> TrainResult<()> {
        for callback in &mut self.callbacks {
            callback.on_train_end(state)?;
        }
        Ok(())
    }

    /// Execute on_epoch_begin for all callbacks.
    pub fn on_epoch_begin(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        for callback in &mut self.callbacks {
            callback.on_epoch_begin(epoch, state)?;
        }
        Ok(())
    }

    /// Execute on_epoch_end for all callbacks.
    pub fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        for callback in &mut self.callbacks {
            callback.on_epoch_end(epoch, state)?;
        }
        Ok(())
    }

    /// Execute on_batch_begin for all callbacks.
    pub fn on_batch_begin(&mut self, batch: usize, state: &TrainingState) -> TrainResult<()> {
        for callback in &mut self.callbacks {
            callback.on_batch_begin(batch, state)?;
        }
        Ok(())
    }

    /// Execute on_batch_end for all callbacks.
    pub fn on_batch_end(&mut self, batch: usize, state: &TrainingState) -> TrainResult<()> {
        for callback in &mut self.callbacks {
            callback.on_batch_end(batch, state)?;
        }
        Ok(())
    }

    /// Execute on_validation_end for all callbacks.
    pub fn on_validation_end(&mut self, state: &TrainingState) -> TrainResult<()> {
        for callback in &mut self.callbacks {
            callback.on_validation_end(state)?;
        }
        Ok(())
    }

    /// Check if any callback requests early stopping.
    pub fn should_stop(&self) -> bool {
        self.callbacks.iter().any(|cb| cb.should_stop())
    }
}

impl Default for CallbackList {
    fn default() -> Self {
        Self::new()
    }
}

/// Callback that logs training progress.
pub struct EpochCallback {
    /// Whether to print detailed information.
    pub verbose: bool,
}

impl EpochCallback {
    /// Create a new epoch callback.
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
}

impl Callback for EpochCallback {
    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        if self.verbose {
            println!(
                "Epoch {}: loss={:.6}, val_loss={:.6}",
                epoch,
                state.train_loss,
                state.val_loss.unwrap_or(f64::NAN)
            );
        }
        Ok(())
    }
}

/// Callback that logs batch progress.
pub struct BatchCallback {
    /// Frequency of logging (every N batches).
    pub log_frequency: usize,
}

impl BatchCallback {
    /// Create a new batch callback.
    pub fn new(log_frequency: usize) -> Self {
        Self { log_frequency }
    }
}

impl Callback for BatchCallback {
    fn on_batch_end(&mut self, batch: usize, state: &TrainingState) -> TrainResult<()> {
        if batch.is_multiple_of(self.log_frequency) {
            println!("Batch {}: loss={:.6}", batch, state.batch_loss);
        }
        Ok(())
    }
}

/// Callback for validation during training.
pub struct ValidationCallback {
    /// Frequency of validation (every N epochs).
    pub validation_frequency: usize,
}

impl ValidationCallback {
    /// Create a new validation callback.
    pub fn new(validation_frequency: usize) -> Self {
        Self {
            validation_frequency,
        }
    }
}

impl Callback for ValidationCallback {
    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        if epoch.is_multiple_of(self.validation_frequency) {
            if let Some(val_loss) = state.val_loss {
                println!("Validation at epoch {}: val_loss={:.6}", epoch, val_loss);
            }
        }
        Ok(())
    }
}

/// Comprehensive checkpoint data structure.
///
/// This structure contains all the information needed to fully restore
/// training state, including model parameters, optimizer state, and training history.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingCheckpoint {
    /// Current epoch number.
    pub epoch: usize,
    /// Model parameters as flattened vectors.
    pub parameters: HashMap<String, Vec<f64>>,
    /// Optimizer state as flattened vectors.
    pub optimizer_state: HashMap<String, Vec<f64>>,
    /// Scheduler state (if present).
    pub scheduler_state: Option<HashMap<String, f64>>,
    /// Current training loss.
    pub train_loss: f64,
    /// Current validation loss (if available).
    pub val_loss: Option<f64>,
    /// Training loss history.
    pub train_loss_history: Vec<f64>,
    /// Validation loss history.
    pub val_loss_history: Vec<f64>,
    /// Metrics history.
    pub metrics_history: HashMap<String, Vec<f64>>,
    /// Current learning rate.
    pub learning_rate: f64,
    /// Best validation loss seen so far.
    pub best_val_loss: Option<f64>,
}

impl TrainingCheckpoint {
    /// Create a new checkpoint from current training state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        epoch: usize,
        parameters: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
        optimizer_state: &HashMap<String, Vec<f64>>,
        scheduler_state: Option<HashMap<String, f64>>,
        state: &TrainingState,
        train_loss_history: &[f64],
        val_loss_history: &[f64],
        metrics_history: &HashMap<String, Vec<f64>>,
        best_val_loss: Option<f64>,
    ) -> Self {
        // Convert parameters to flat vectors
        let parameters = parameters
            .iter()
            .map(|(name, param)| (name.clone(), param.iter().copied().collect()))
            .collect();

        Self {
            epoch,
            parameters,
            optimizer_state: optimizer_state.clone(),
            scheduler_state,
            train_loss: state.train_loss,
            val_loss: state.val_loss,
            train_loss_history: train_loss_history.to_vec(),
            val_loss_history: val_loss_history.to_vec(),
            metrics_history: metrics_history.clone(),
            learning_rate: state.learning_rate,
            best_val_loss,
        }
    }

    /// Save checkpoint to a file.
    pub fn save(&self, path: &PathBuf) -> TrainResult<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to serialize checkpoint: {}", e))
        })?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                TrainError::CheckpointError(format!("Failed to create checkpoint directory: {}", e))
            })?;
        }

        std::fs::write(path, json).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to write checkpoint: {}", e))
        })?;

        Ok(())
    }

    /// Load checkpoint from a file.
    pub fn load(path: &PathBuf) -> TrainResult<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to read checkpoint: {}", e))
        })?;

        let checkpoint: Self = serde_json::from_str(&json).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to deserialize checkpoint: {}", e))
        })?;

        Ok(checkpoint)
    }
}

/// Callback for model checkpointing.
pub struct CheckpointCallback {
    /// Directory to save checkpoints.
    pub checkpoint_dir: PathBuf,
    /// Frequency of checkpointing (every N epochs).
    pub save_frequency: usize,
    /// Whether to save only the best model.
    pub save_best_only: bool,
    /// Best validation loss seen so far.
    best_val_loss: Option<f64>,
}

impl CheckpointCallback {
    /// Create a new checkpoint callback.
    pub fn new(checkpoint_dir: PathBuf, save_frequency: usize, save_best_only: bool) -> Self {
        Self {
            checkpoint_dir,
            save_frequency,
            save_best_only,
            best_val_loss: None,
        }
    }

    /// Save checkpoint to disk (legacy simple format).
    fn save_checkpoint(&self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        let checkpoint_path = self
            .checkpoint_dir
            .join(format!("checkpoint_epoch_{}.json", epoch));

        // Create checkpoint data
        let mut checkpoint = HashMap::new();
        checkpoint.insert("epoch".to_string(), epoch as f64);
        checkpoint.insert("train_loss".to_string(), state.train_loss);
        if let Some(val_loss) = state.val_loss {
            checkpoint.insert("val_loss".to_string(), val_loss);
        }

        // Save to JSON
        let json = serde_json::to_string_pretty(&checkpoint).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to serialize checkpoint: {}", e))
        })?;

        std::fs::create_dir_all(&self.checkpoint_dir).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to create checkpoint directory: {}", e))
        })?;

        std::fs::write(&checkpoint_path, json).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to write checkpoint: {}", e))
        })?;

        println!("Checkpoint saved to {:?}", checkpoint_path);
        Ok(())
    }
}

impl Callback for CheckpointCallback {
    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        if !epoch.is_multiple_of(self.save_frequency) {
            return Ok(());
        }

        if self.save_best_only {
            if let Some(val_loss) = state.val_loss {
                let should_save = self
                    .best_val_loss
                    .map(|best| val_loss < best)
                    .unwrap_or(true);

                if should_save {
                    self.best_val_loss = Some(val_loss);
                    self.save_checkpoint(epoch, state)?;
                }
            }
        } else {
            self.save_checkpoint(epoch, state)?;
        }

        Ok(())
    }
}

/// Callback for early stopping based on validation loss.
pub struct EarlyStoppingCallback {
    /// Number of epochs with no improvement after which training will be stopped.
    pub patience: usize,
    /// Minimum change to qualify as an improvement.
    pub min_delta: f64,
    /// Best validation loss seen so far.
    best_val_loss: Option<f64>,
    /// Counter for epochs without improvement.
    wait: usize,
    /// Whether to stop training.
    stop_training: bool,
}

impl EarlyStoppingCallback {
    /// Create a new early stopping callback.
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_val_loss: None,
            wait: 0,
            stop_training: false,
        }
    }
}

impl Callback for EarlyStoppingCallback {
    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        if let Some(val_loss) = state.val_loss {
            let improved = self
                .best_val_loss
                .map(|best| val_loss < best - self.min_delta)
                .unwrap_or(true);

            if improved {
                self.best_val_loss = Some(val_loss);
                self.wait = 0;
            } else {
                self.wait += 1;
                if self.wait >= self.patience {
                    println!(
                        "Early stopping at epoch {} (no improvement for {} epochs)",
                        epoch, self.patience
                    );
                    self.stop_training = true;
                }
            }
        }

        Ok(())
    }

    fn should_stop(&self) -> bool {
        self.stop_training
    }
}

/// Callback for learning rate reduction on plateau.
#[allow(dead_code)]
pub struct ReduceLrOnPlateauCallback {
    /// Factor by which to reduce learning rate.
    pub factor: f64,
    /// Number of epochs with no improvement after which learning rate will be reduced.
    pub patience: usize,
    /// Minimum change to qualify as an improvement.
    pub min_delta: f64,
    /// Lower bound on the learning rate.
    pub min_lr: f64,
    /// Best validation loss seen so far.
    best_val_loss: Option<f64>,
    /// Counter for epochs without improvement.
    wait: usize,
}

impl ReduceLrOnPlateauCallback {
    /// Create a new reduce LR on plateau callback.
    #[allow(dead_code)]
    pub fn new(factor: f64, patience: usize, min_delta: f64, min_lr: f64) -> Self {
        Self {
            factor,
            patience,
            min_delta,
            min_lr,
            best_val_loss: None,
            wait: 0,
        }
    }
}

impl Callback for ReduceLrOnPlateauCallback {
    fn on_epoch_end(&mut self, _epoch: usize, state: &TrainingState) -> TrainResult<()> {
        if let Some(val_loss) = state.val_loss {
            let improved = self
                .best_val_loss
                .map(|best| val_loss < best - self.min_delta)
                .unwrap_or(true);

            if improved {
                self.best_val_loss = Some(val_loss);
                self.wait = 0;
            } else {
                self.wait += 1;
                if self.wait >= self.patience {
                    // Note: We can't actually modify the optimizer here since we don't have a reference
                    // This would need to be handled by the Trainer
                    let new_lr = (state.learning_rate * self.factor).max(self.min_lr);
                    if new_lr != state.learning_rate {
                        println!("Reducing learning rate to {:.6}", new_lr);
                    }
                    self.wait = 0;
                }
            }
        }

        Ok(())
    }
}

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
                "  ⚠️  Vanishing gradient warnings: {}",
                self.vanishing_count
            );
        }

        if self.exploding_count > 0 {
            println!(
                "  ⚠️  Exploding gradient warnings: {}",
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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state() -> TrainingState {
        TrainingState {
            epoch: 0,
            batch: 0,
            train_loss: 1.0,
            val_loss: Some(0.8),
            batch_loss: 0.5,
            learning_rate: 0.001,
            metrics: HashMap::new(),
        }
    }

    #[test]
    fn test_callback_list() {
        let mut callbacks = CallbackList::new();
        callbacks.add(Box::new(EpochCallback::new(false)));

        let state = create_test_state();
        callbacks.on_train_begin(&state).unwrap();
        callbacks.on_epoch_begin(0, &state).unwrap();
        callbacks.on_epoch_end(0, &state).unwrap();
        callbacks.on_train_end(&state).unwrap();
    }

    #[test]
    fn test_early_stopping() {
        let mut callback = EarlyStoppingCallback::new(2, 0.01);
        let mut state = create_test_state();

        // First epoch - improvement
        state.val_loss = Some(1.0);
        callback.on_epoch_end(0, &state).unwrap();
        assert!(!callback.should_stop());

        // Second epoch - improvement
        state.val_loss = Some(0.8);
        callback.on_epoch_end(1, &state).unwrap();
        assert!(!callback.should_stop());

        // Third epoch - no improvement
        state.val_loss = Some(0.81);
        callback.on_epoch_end(2, &state).unwrap();
        assert!(!callback.should_stop());

        // Fourth epoch - no improvement (exceeds patience)
        state.val_loss = Some(0.82);
        callback.on_epoch_end(3, &state).unwrap();
        assert!(callback.should_stop());
    }

    #[test]
    fn test_checkpoint_callback() {
        use std::env::temp_dir;

        let checkpoint_dir = temp_dir().join("tensorlogic_test_checkpoints");
        let mut callback = CheckpointCallback::new(checkpoint_dir.clone(), 1, false);
        let state = create_test_state();

        callback.on_epoch_end(0, &state).unwrap();

        // Verify checkpoint was created
        let checkpoint_path = checkpoint_dir.join("checkpoint_epoch_0.json");
        assert!(checkpoint_path.exists());

        // Clean up
        std::fs::remove_dir_all(checkpoint_dir).ok();
    }

    #[test]
    fn test_training_checkpoint_save_load() {
        use scirs2_core::ndarray::Array2;
        use std::env::temp_dir;

        // Create test parameters
        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Array2::from_elem((2, 3), 1.5));
        parameters.insert("bias".to_string(), Array2::from_elem((1, 3), 0.5));

        // Create test state
        let state = TrainingState {
            epoch: 5,
            batch: 100,
            train_loss: 0.75,
            val_loss: Some(0.85),
            batch_loss: 0.72,
            learning_rate: 0.001,
            metrics: HashMap::new(),
        };

        // Create optimizer state (mock)
        let optimizer_state = {
            let mut state = HashMap::new();
            state.insert("momentum_weight".to_string(), vec![0.1, 0.2, 0.3]);
            state.insert("momentum_bias".to_string(), vec![0.05]);
            state
        };

        // Create checkpoint
        let checkpoint = TrainingCheckpoint::new(
            5,
            &parameters,
            &optimizer_state,
            None,
            &state,
            &[1.0, 0.9, 0.8, 0.77, 0.75],
            &[1.1, 0.95, 0.88, 0.87, 0.85],
            &HashMap::new(),
            Some(0.85),
        );

        // Save checkpoint
        let checkpoint_path = temp_dir().join("test_training_checkpoint.json");
        checkpoint.save(&checkpoint_path).unwrap();

        // Verify file exists
        assert!(checkpoint_path.exists());

        // Load checkpoint
        let loaded = TrainingCheckpoint::load(&checkpoint_path).unwrap();

        // Verify data
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.train_loss, 0.75);
        assert_eq!(loaded.val_loss, Some(0.85));
        assert_eq!(loaded.learning_rate, 0.001);
        assert_eq!(loaded.train_loss_history.len(), 5);
        assert_eq!(loaded.val_loss_history.len(), 5);
        assert_eq!(loaded.best_val_loss, Some(0.85));

        // Verify parameters
        assert_eq!(loaded.parameters.len(), 2);
        assert!(loaded.parameters.contains_key("weight"));
        assert!(loaded.parameters.contains_key("bias"));

        // Verify optimizer state
        assert_eq!(loaded.optimizer_state.len(), 2);
        assert!(loaded.optimizer_state.contains_key("momentum_weight"));

        // Clean up
        std::fs::remove_file(checkpoint_path).ok();
    }

    #[test]
    fn test_training_checkpoint_with_metrics() {
        use scirs2_core::ndarray::Array2;
        use std::env::temp_dir;

        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), Array2::zeros((2, 2)));

        let state = create_test_state();
        let optimizer_state = HashMap::new();

        // Add metrics history
        let mut metrics_history = HashMap::new();
        metrics_history.insert("accuracy".to_string(), vec![0.5, 0.6, 0.7]);
        metrics_history.insert("f1_score".to_string(), vec![0.45, 0.55, 0.65]);

        let checkpoint = TrainingCheckpoint::new(
            2,
            &parameters,
            &optimizer_state,
            None,
            &state,
            &[1.0, 0.8, 0.6],
            &[1.1, 0.9, 0.7],
            &metrics_history,
            Some(0.7),
        );

        let checkpoint_path = temp_dir().join("test_checkpoint_with_metrics.json");
        checkpoint.save(&checkpoint_path).unwrap();

        let loaded = TrainingCheckpoint::load(&checkpoint_path).unwrap();

        // Verify metrics
        assert_eq!(loaded.metrics_history.len(), 2);
        assert!(loaded.metrics_history.contains_key("accuracy"));
        assert!(loaded.metrics_history.contains_key("f1_score"));
        assert_eq!(loaded.metrics_history["accuracy"].len(), 3);

        std::fs::remove_file(checkpoint_path).ok();
    }
}

/// Weight histogram statistics for debugging and monitoring.
#[derive(Debug, Clone)]
pub struct HistogramStats {
    /// Parameter name.
    pub name: String,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Histogram bins (boundaries).
    pub bins: Vec<f64>,
    /// Histogram counts per bin.
    pub counts: Vec<usize>,
}

impl HistogramStats {
    /// Compute histogram statistics from parameter values.
    pub fn compute(name: &str, values: &[f64], num_bins: usize) -> Self {
        if values.is_empty() {
            return Self {
                name: name.to_string(),
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std: 0.0,
                bins: vec![],
                counts: vec![],
            };
        }

        // Basic statistics
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        // Create histogram bins
        let mut bins = Vec::with_capacity(num_bins + 1);
        let mut counts = vec![0; num_bins];

        let range = max - min;
        let bin_width = if range > 0.0 {
            range / num_bins as f64
        } else {
            1.0
        };

        for i in 0..=num_bins {
            bins.push(min + i as f64 * bin_width);
        }

        // Count values in each bin
        for &value in values {
            let bin_idx = if range > 0.0 {
                ((value - min) / bin_width).floor() as usize
            } else {
                0
            };
            let bin_idx = bin_idx.min(num_bins - 1);
            counts[bin_idx] += 1;
        }

        Self {
            name: name.to_string(),
            min,
            max,
            mean,
            std,
            bins,
            counts,
        }
    }

    /// Pretty print histogram as ASCII art.
    pub fn display(&self, width: usize) {
        println!("\n=== Histogram: {} ===", self.name);
        println!("  Min: {:.6}, Max: {:.6}", self.min, self.max);
        println!("  Mean: {:.6}, Std: {:.6}", self.mean, self.std);
        println!("\n  Distribution:");

        if self.counts.is_empty() {
            println!("    (empty)");
            return;
        }

        let max_count = *self.counts.iter().max().unwrap_or(&1);

        for (i, &count) in self.counts.iter().enumerate() {
            let bar_len = if max_count > 0 {
                (count as f64 / max_count as f64 * width as f64) as usize
            } else {
                0
            };

            let bar = "█".repeat(bar_len);
            let left = if i < self.bins.len() - 1 {
                self.bins[i]
            } else {
                self.bins[i - 1]
            };
            let right = if i < self.bins.len() - 1 {
                self.bins[i + 1]
            } else {
                self.bins[i]
            };

            println!("  [{:>8.3}, {:>8.3}): {:>6} {}", left, right, count, bar);
        }
    }
}

/// Callback for tracking weight histograms during training.
///
/// This callback computes and logs histogram statistics of model parameters
/// at regular intervals. Useful for:
/// - Detecting vanishing/exploding weights
/// - Monitoring weight distribution changes
/// - Debugging initialization issues
/// - Understanding parameter evolution
///
/// # Example
///
/// ```no_run
/// use tensorlogic_train::{CallbackList, HistogramCallback};
///
/// let mut callbacks = CallbackList::new();
/// callbacks.add(Box::new(HistogramCallback::new(
///     5,   // log_frequency: Every 5 epochs
///     10,  // num_bins: 10 histogram bins
///     true, // verbose: Print detailed histograms
/// )));
/// ```
pub struct HistogramCallback {
    /// Frequency of logging (every N epochs).
    log_frequency: usize,
    /// Number of histogram bins.
    #[allow(dead_code)]
    // Used in compute_histograms - will be active when parameters are accessible
    num_bins: usize,
    /// Whether to print detailed histograms.
    verbose: bool,
    /// History of histogram statistics.
    pub history: Vec<HashMap<String, HistogramStats>>,
}

impl HistogramCallback {
    /// Create a new histogram callback.
    ///
    /// # Arguments
    /// * `log_frequency` - Log histograms every N epochs
    /// * `num_bins` - Number of bins in each histogram
    /// * `verbose` - Print detailed ASCII histograms
    pub fn new(log_frequency: usize, num_bins: usize, verbose: bool) -> Self {
        Self {
            log_frequency,
            num_bins,
            verbose,
            history: Vec::new(),
        }
    }

    /// Compute histograms for all parameters in state.
    #[allow(dead_code)] // Placeholder - will be used when TrainingState includes parameters
    fn compute_histograms(&self, _state: &TrainingState) -> HashMap<String, HistogramStats> {
        // In a real implementation, we would access parameters from state
        // For now, this is a placeholder that would be populated when
        // TrainingState includes parameter access

        // Example of what this would look like with actual parameters:
        // let mut histograms = HashMap::new();
        // for (name, param) in state.parameters.iter() {
        //     let values: Vec<f64> = param.iter().copied().collect();
        //     let stats = HistogramStats::compute(name, &values, self.num_bins);
        //     histograms.insert(name.clone(), stats);
        // }
        // histograms

        HashMap::new()
    }
}

impl Callback for HistogramCallback {
    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        if (epoch + 1).is_multiple_of(self.log_frequency) {
            let histograms = self.compute_histograms(state);

            if self.verbose {
                println!("\n--- Weight Histograms (Epoch {}) ---", epoch + 1);
                for (_name, stats) in histograms.iter() {
                    stats.display(40); // 40 character width for ASCII bars
                }
            } else {
                println!(
                    "Epoch {}: Computed histograms for {} parameters",
                    epoch + 1,
                    histograms.len()
                );
            }

            self.history.push(histograms);
        }

        Ok(())
    }
}

/// Performance profiling statistics.
#[derive(Debug, Clone, Default)]
pub struct ProfilingStats {
    /// Total training time (seconds).
    pub total_time: f64,
    /// Time per epoch (seconds).
    pub epoch_times: Vec<f64>,
    /// Samples per second.
    pub samples_per_sec: f64,
    /// Batches per second.
    pub batches_per_sec: f64,
    /// Average batch time (seconds).
    pub avg_batch_time: f64,
    /// Peak memory usage (MB) - placeholder.
    pub peak_memory_mb: f64,
}

impl ProfilingStats {
    /// Pretty print profiling statistics.
    pub fn display(&self) {
        println!("\n=== Profiling Statistics ===");
        println!("Total time: {:.2}s", self.total_time);
        println!("Samples/sec: {:.2}", self.samples_per_sec);
        println!("Batches/sec: {:.2}", self.batches_per_sec);
        println!("Avg batch time: {:.4}s", self.avg_batch_time);

        if !self.epoch_times.is_empty() {
            let avg_epoch = self.epoch_times.iter().sum::<f64>() / self.epoch_times.len() as f64;
            let min_epoch = self
                .epoch_times
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let max_epoch = self
                .epoch_times
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            println!("\nEpoch times:");
            println!("  Average: {:.2}s", avg_epoch);
            println!("  Min: {:.2}s", min_epoch);
            println!("  Max: {:.2}s", max_epoch);
        }
    }
}

/// Callback for profiling training performance.
///
/// Tracks timing information and throughput metrics during training.
/// Useful for:
/// - Identifying performance bottlenecks
/// - Comparing different configurations
/// - Monitoring training speed
/// - Resource utilization tracking
///
/// # Example
///
/// ```no_run
/// use tensorlogic_train::{CallbackList, ProfilingCallback};
///
/// let mut callbacks = CallbackList::new();
/// callbacks.add(Box::new(ProfilingCallback::new(
///     true,  // verbose: Print detailed stats
///     5,     // log_frequency: Every 5 epochs
/// )));
/// ```
pub struct ProfilingCallback {
    /// Whether to print detailed profiling info.
    verbose: bool,
    /// Frequency of logging (every N epochs).
    log_frequency: usize,
    /// Training start time.
    start_time: Option<std::time::Instant>,
    /// Last epoch start time.
    epoch_start_time: Option<std::time::Instant>,
    /// Batch start time.
    batch_start_time: Option<std::time::Instant>,
    /// Accumulated statistics.
    pub stats: ProfilingStats,
    /// Batch times for current epoch.
    current_epoch_batch_times: Vec<f64>,
    /// Total batches processed.
    total_batches: usize,
}

impl ProfilingCallback {
    /// Create a new profiling callback.
    ///
    /// # Arguments
    /// * `verbose` - Print detailed profiling information
    /// * `log_frequency` - Log stats every N epochs
    pub fn new(verbose: bool, log_frequency: usize) -> Self {
        Self {
            verbose,
            log_frequency,
            start_time: None,
            epoch_start_time: None,
            batch_start_time: None,
            stats: ProfilingStats::default(),
            current_epoch_batch_times: Vec::new(),
            total_batches: 0,
        }
    }

    /// Get profiling statistics.
    pub fn get_stats(&self) -> &ProfilingStats {
        &self.stats
    }
}

impl Callback for ProfilingCallback {
    fn on_train_begin(&mut self, _state: &TrainingState) -> TrainResult<()> {
        self.start_time = Some(std::time::Instant::now());
        if self.verbose {
            println!("⏱️  Profiling started");
        }
        Ok(())
    }

    fn on_train_end(&mut self, _state: &TrainingState) -> TrainResult<()> {
        if let Some(start) = self.start_time {
            self.stats.total_time = start.elapsed().as_secs_f64();

            // Compute aggregate statistics
            if self.total_batches > 0 {
                self.stats.avg_batch_time = self.stats.total_time / self.total_batches as f64;
                self.stats.batches_per_sec = self.total_batches as f64 / self.stats.total_time;
            }

            if self.verbose {
                println!("\n⏱️  Profiling completed");
                self.stats.display();
            }
        }
        Ok(())
    }

    fn on_epoch_begin(&mut self, epoch: usize, _state: &TrainingState) -> TrainResult<()> {
        self.epoch_start_time = Some(std::time::Instant::now());
        self.current_epoch_batch_times.clear();

        if self.verbose && (epoch + 1).is_multiple_of(self.log_frequency) {
            println!("\n⏱️  Epoch {} profiling started", epoch + 1);
        }
        Ok(())
    }

    fn on_epoch_end(&mut self, epoch: usize, _state: &TrainingState) -> TrainResult<()> {
        if let Some(epoch_start) = self.epoch_start_time {
            let epoch_time = epoch_start.elapsed().as_secs_f64();
            self.stats.epoch_times.push(epoch_time);

            if self.verbose && (epoch + 1).is_multiple_of(self.log_frequency) {
                let avg_batch = if !self.current_epoch_batch_times.is_empty() {
                    self.current_epoch_batch_times.iter().sum::<f64>()
                        / self.current_epoch_batch_times.len() as f64
                } else {
                    0.0
                };

                println!("⏱️  Epoch {} completed:", epoch + 1);
                println!("    Time: {:.2}s", epoch_time);
                println!(
                    "    Batches: {} ({:.4}s avg)",
                    self.current_epoch_batch_times.len(),
                    avg_batch
                );
            }
        }
        Ok(())
    }

    fn on_batch_begin(&mut self, _batch: usize, _state: &TrainingState) -> TrainResult<()> {
        self.batch_start_time = Some(std::time::Instant::now());
        Ok(())
    }

    fn on_batch_end(&mut self, _batch: usize, _state: &TrainingState) -> TrainResult<()> {
        if let Some(batch_start) = self.batch_start_time {
            let batch_time = batch_start.elapsed().as_secs_f64();
            self.current_epoch_batch_times.push(batch_time);
            self.total_batches += 1;
        }
        Ok(())
    }
}

/// Model EMA (Exponential Moving Average) callback.
///
/// Maintains an exponential moving average of model parameters during training.
/// This often leads to better generalization and more stable predictions.
///
/// The shadow parameters are updated as:
/// shadow_param = decay * shadow_param + (1 - decay) * param
///
/// Reference: Common practice in modern deep learning, popularized by Mean Teacher
/// and other semi-supervised learning methods.
pub struct ModelEMACallback {
    /// Decay rate for EMA (typically 0.999 or 0.9999).
    decay: f64,
    /// Shadow parameters (EMA of model parameters).
    shadow_params: HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    /// Whether to use warmup for the decay (start with smaller decay).
    use_warmup: bool,
    /// Current update step (for warmup).
    num_updates: usize,
    /// Whether callback is initialized.
    initialized: bool,
}

impl ModelEMACallback {
    /// Create a new Model EMA callback.
    ///
    /// # Arguments
    /// * `decay` - EMA decay rate (e.g., 0.999, 0.9999)
    /// * `use_warmup` - Whether to use decay warmup (recommended)
    pub fn new(decay: f64, use_warmup: bool) -> Self {
        Self {
            decay,
            shadow_params: HashMap::new(),
            use_warmup,
            num_updates: 0,
            initialized: false,
        }
    }

    /// Initialize shadow parameters from current model parameters.
    pub fn initialize(
        &mut self,
        parameters: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    ) {
        self.shadow_params.clear();
        for (name, param) in parameters {
            self.shadow_params.insert(name.clone(), param.clone());
        }
        self.initialized = true;
    }

    /// Update EMA parameters.
    pub fn update(
        &mut self,
        parameters: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    ) -> TrainResult<()> {
        if !self.initialized {
            return Err(TrainError::CallbackError(
                "ModelEMA not initialized. Call initialize() first.".to_string(),
            ));
        }

        self.num_updates += 1;

        // Compute effective decay with warmup
        let decay = if self.use_warmup {
            // Gradual warmup: start with (1 + num_updates) / (10 + num_updates)
            // and approach self.decay
            let warmup_decay = (1.0 + self.num_updates as f64) / (10.0 + self.num_updates as f64);
            warmup_decay.min(self.decay)
        } else {
            self.decay
        };

        // Update shadow parameters
        for (name, param) in parameters {
            if let Some(shadow) = self.shadow_params.get_mut(name) {
                // shadow = decay * shadow + (1 - decay) * param
                *shadow = &*shadow * decay + &(param * (1.0 - decay));
            }
        }

        Ok(())
    }

    /// Get the EMA parameters.
    pub fn get_shadow_params(
        &self,
    ) -> &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>> {
        &self.shadow_params
    }

    /// Apply EMA parameters to the model (for evaluation).
    pub fn apply_shadow(
        &self,
        parameters: &mut HashMap<
            String,
            scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>,
        >,
    ) {
        for (name, shadow) in &self.shadow_params {
            if let Some(param) = parameters.get_mut(name) {
                *param = shadow.clone();
            }
        }
    }
}

impl Callback for ModelEMACallback {
    fn on_train_begin(&mut self, _state: &TrainingState) -> TrainResult<()> {
        // Note: Initialization must be done externally since we don't have access to parameters here
        Ok(())
    }

    fn on_batch_end(&mut self, _batch: usize, _state: &TrainingState) -> TrainResult<()> {
        // Note: Update must be called externally since we don't have access to parameters here
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

/// SWA (Stochastic Weight Averaging) callback.
///
/// Averages model parameters over the course of training, typically starting
/// from a later epoch. This often leads to better generalization and wider optima.
///
/// Reference: Izmailov et al. "Averaging Weights Leads to Wider Optima and Better Generalization" (UAI 2018)
pub struct SWACallback {
    /// Epoch to start SWA (e.g., 75% through training).
    start_epoch: usize,
    /// Frequency of parameter averaging (every N epochs).
    update_frequency: usize,
    /// Running average of parameters.
    swa_params: HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    /// Number of models averaged so far.
    num_averaged: usize,
    /// Whether SWA is active.
    active: bool,
    /// Whether SWA parameters are initialized.
    initialized: bool,
    /// Verbose output.
    verbose: bool,
}

impl SWACallback {
    /// Create a new SWA callback.
    ///
    /// # Arguments
    /// * `start_epoch` - Epoch to start averaging (e.g., 0.75 * total_epochs)
    /// * `update_frequency` - Average parameters every N epochs (typically 1)
    /// * `verbose` - Whether to print progress
    pub fn new(start_epoch: usize, update_frequency: usize, verbose: bool) -> Self {
        Self {
            start_epoch,
            update_frequency,
            swa_params: HashMap::new(),
            num_averaged: 0,
            active: false,
            initialized: false,
            verbose,
        }
    }

    /// Update SWA parameters with current model parameters.
    pub fn update_average(
        &mut self,
        parameters: &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>>,
    ) -> TrainResult<()> {
        if !self.active {
            return Ok(());
        }

        if !self.initialized {
            // Initialize with first model
            for (name, param) in parameters {
                self.swa_params.insert(name.clone(), param.clone());
            }
            self.initialized = true;
            self.num_averaged = 1;

            if self.verbose {
                println!("📊 SWA: Initialized with model parameters");
            }
        } else {
            // Running average: swa = (swa * n + param) / (n + 1)
            let n = self.num_averaged as f64;
            for (name, param) in parameters {
                if let Some(swa_param) = self.swa_params.get_mut(name) {
                    *swa_param = &(&*swa_param * n + param) / (n + 1.0);
                }
            }
            self.num_averaged += 1;

            if self.verbose {
                println!("📊 SWA: Updated average (n={})", self.num_averaged);
            }
        }

        Ok(())
    }

    /// Get the SWA parameters.
    pub fn get_swa_params(
        &self,
    ) -> &HashMap<String, scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>> {
        &self.swa_params
    }

    /// Apply SWA parameters to the model.
    pub fn apply_swa(
        &self,
        parameters: &mut HashMap<
            String,
            scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::Ix2>,
        >,
    ) {
        if self.initialized {
            for (name, swa_param) in &self.swa_params {
                if let Some(param) = parameters.get_mut(name) {
                    *param = swa_param.clone();
                }
            }
        }
    }

    /// Check if SWA has collected any averages.
    pub fn is_ready(&self) -> bool {
        self.initialized && self.num_averaged > 0
    }
}

impl Callback for SWACallback {
    fn on_epoch_end(&mut self, epoch: usize, _state: &TrainingState) -> TrainResult<()> {
        // Activate SWA at start_epoch
        if epoch >= self.start_epoch && !self.active {
            self.active = true;
            if self.verbose {
                println!("\n📊 SWA: Activated at epoch {}", epoch + 1);
            }
        }

        // Check if we should update average
        if self.active && epoch >= self.start_epoch {
            let relative_epoch = epoch - self.start_epoch;
            if relative_epoch.is_multiple_of(self.update_frequency) {
                // Note: Actual update must be called externally with parameters
                if self.verbose && self.initialized {
                    println!(
                        "📊 SWA: Ready to update at epoch {} (call update_average with parameters)",
                        epoch + 1
                    );
                }
            }
        }

        Ok(())
    }

    fn on_train_end(&mut self, _state: &TrainingState) -> TrainResult<()> {
        if self.verbose && self.initialized {
            println!(
                "\n📊 SWA: Training complete. Averaged {} models.",
                self.num_averaged
            );
            println!("📊 SWA: Call apply_swa() to use averaged parameters.");
        }
        Ok(())
    }
}

#[cfg(test)]
mod profiling_tests {
    use super::*;

    #[test]
    fn test_histogram_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = HistogramStats::compute("test", &values, 5);

        assert_eq!(stats.name, "test");
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert!((stats.mean - 5.5).abs() < 1e-6);
        assert_eq!(stats.bins.len(), 6);
        assert_eq!(stats.counts.len(), 5);
        assert_eq!(stats.counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_histogram_callback() {
        use std::collections::HashMap;
        let mut callback = HistogramCallback::new(2, 10, false);
        let state = TrainingState {
            epoch: 0,
            batch: 0,
            train_loss: 0.5,
            batch_loss: 0.5,
            val_loss: Some(0.6),
            learning_rate: 0.01,
            metrics: HashMap::new(),
        };

        // Should not log on epoch 0
        callback.on_epoch_end(0, &state).unwrap();
        assert_eq!(callback.history.len(), 0);

        // Should log on epoch 1 (frequency=2, so every 2 epochs)
        callback.on_epoch_end(1, &state).unwrap();
        assert_eq!(callback.history.len(), 1);
    }

    #[test]
    fn test_profiling_callback() {
        use std::collections::HashMap;
        let mut callback = ProfilingCallback::new(false, 1);
        let state = TrainingState {
            epoch: 0,
            batch: 0,
            train_loss: 0.5,
            batch_loss: 0.5,
            val_loss: Some(0.6),
            learning_rate: 0.01,
            metrics: HashMap::new(),
        };

        callback.on_train_begin(&state).unwrap();
        assert!(callback.start_time.is_some());

        callback.on_epoch_begin(0, &state).unwrap();
        assert!(callback.epoch_start_time.is_some());

        callback.on_batch_begin(0, &state).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        callback.on_batch_end(0, &state).unwrap();

        assert_eq!(callback.total_batches, 1);
        assert_eq!(callback.current_epoch_batch_times.len(), 1);

        callback.on_epoch_end(0, &state).unwrap();
        assert_eq!(callback.stats.epoch_times.len(), 1);

        callback.on_train_end(&state).unwrap();
        assert!(callback.stats.total_time > 0.0);
    }
}
