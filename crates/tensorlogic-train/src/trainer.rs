//! Main training loop implementation.

use crate::{
    extract_batch, BatchConfig, BatchIterator, CallbackList, Loss, LrScheduler, MetricTracker,
    Optimizer, TrainResult,
};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};
use std::collections::HashMap;

/// Training state passed to callbacks.
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch number.
    pub epoch: usize,
    /// Current batch number within epoch.
    pub batch: usize,
    /// Training loss for current epoch.
    pub train_loss: f64,
    /// Validation loss (if validation is performed).
    pub val_loss: Option<f64>,
    /// Loss for current batch.
    pub batch_loss: f64,
    /// Current learning rate.
    pub learning_rate: f64,
    /// Additional metrics.
    pub metrics: HashMap<String, f64>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            batch: 0,
            train_loss: 0.0,
            val_loss: None,
            batch_loss: 0.0,
            learning_rate: 0.001,
            metrics: HashMap::new(),
        }
    }
}

/// Configuration for training.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Number of epochs to train.
    pub num_epochs: usize,
    /// Batch configuration.
    pub batch_config: BatchConfig,
    /// Whether to validate after each epoch.
    pub validate_every_epoch: bool,
    /// Frequency of logging (every N batches).
    pub log_frequency: usize,
    /// Whether to use learning rate scheduler.
    pub use_scheduler: bool,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            num_epochs: 10,
            batch_config: BatchConfig::default(),
            validate_every_epoch: true,
            log_frequency: 100,
            use_scheduler: false,
        }
    }
}

/// Main trainer for model training.
pub struct Trainer {
    /// Configuration.
    config: TrainerConfig,
    /// Loss function.
    loss_fn: Box<dyn Loss>,
    /// Optimizer.
    optimizer: Box<dyn Optimizer>,
    /// Optional learning rate scheduler.
    scheduler: Option<Box<dyn LrScheduler>>,
    /// Callbacks.
    callbacks: CallbackList,
    /// Metric tracker.
    metrics: MetricTracker,
    /// Training state.
    state: TrainingState,
}

impl Trainer {
    /// Create a new trainer.
    pub fn new(
        config: TrainerConfig,
        loss_fn: Box<dyn Loss>,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        Self {
            config,
            loss_fn,
            optimizer,
            scheduler: None,
            callbacks: CallbackList::new(),
            metrics: MetricTracker::new(),
            state: TrainingState::default(),
        }
    }

    /// Set learning rate scheduler.
    pub fn with_scheduler(mut self, scheduler: Box<dyn LrScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Set callbacks.
    pub fn with_callbacks(mut self, callbacks: CallbackList) -> Self {
        self.callbacks = callbacks;
        self
    }

    /// Set metrics.
    pub fn with_metrics(mut self, metrics: MetricTracker) -> Self {
        self.metrics = metrics;
        self
    }

    /// Train the model.
    pub fn train(
        &mut self,
        train_data: &ArrayView<f64, Ix2>,
        train_targets: &ArrayView<f64, Ix2>,
        val_data: Option<&ArrayView<f64, Ix2>>,
        val_targets: Option<&ArrayView<f64, Ix2>>,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<TrainingHistory> {
        let mut history = TrainingHistory::new();

        // Initialize state
        self.state.learning_rate = self.optimizer.get_lr();

        // Call on_train_begin
        self.callbacks.on_train_begin(&self.state)?;

        // Training loop
        for epoch in 0..self.config.num_epochs {
            self.state.epoch = epoch;

            // Call on_epoch_begin
            self.callbacks.on_epoch_begin(epoch, &self.state)?;

            // Train one epoch
            let epoch_loss = self.train_epoch(train_data, train_targets, parameters)?;

            self.state.train_loss = epoch_loss;
            history.train_loss.push(epoch_loss);

            // Validation
            if self.config.validate_every_epoch {
                if let (Some(val_data), Some(val_targets)) = (val_data, val_targets) {
                    let val_loss = self.validate(val_data, val_targets, parameters)?;
                    self.state.val_loss = Some(val_loss);
                    history.val_loss.push(val_loss);

                    // Compute metrics
                    let predictions = self.forward(val_data, parameters)?;
                    let metrics = self.metrics.compute_all(&predictions.view(), val_targets)?;
                    self.state.metrics = metrics.clone();

                    for (name, value) in metrics {
                        history.metrics.entry(name).or_default().push(value);
                    }

                    // Call on_validation_end
                    self.callbacks.on_validation_end(&self.state)?;
                }
            }

            // Update learning rate
            if self.config.use_scheduler {
                if let Some(scheduler) = &mut self.scheduler {
                    scheduler.step(&mut *self.optimizer);
                    self.state.learning_rate = self.optimizer.get_lr();
                }
            }

            // Call on_epoch_end
            self.callbacks.on_epoch_end(epoch, &self.state)?;

            // Check for early stopping
            if self.callbacks.should_stop() {
                println!("Early stopping triggered at epoch {}", epoch);
                break;
            }
        }

        // Call on_train_end
        self.callbacks.on_train_end(&self.state)?;

        Ok(history)
    }

    /// Train for one epoch.
    fn train_epoch(
        &mut self,
        train_data: &ArrayView<f64, Ix2>,
        train_targets: &ArrayView<f64, Ix2>,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<f64> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        let mut batch_iter =
            BatchIterator::new(train_data.nrows(), self.config.batch_config.clone());

        while let Some(batch_indices) = batch_iter.next_batch() {
            self.state.batch = num_batches;

            // Call on_batch_begin
            self.callbacks.on_batch_begin(num_batches, &self.state)?;

            // Extract batch
            let batch_data = extract_batch(train_data, &batch_indices)?;
            let batch_targets = extract_batch(train_targets, &batch_indices)?;

            // Forward pass
            let predictions = self.forward(&batch_data.view(), parameters)?;

            // Compute loss
            let loss = self
                .loss_fn
                .compute(&predictions.view(), &batch_targets.view())?;
            self.state.batch_loss = loss;
            total_loss += loss;

            // Compute gradients
            let loss_grad = self
                .loss_fn
                .gradient(&predictions.view(), &batch_targets.view())?;

            // Backward pass (simplified - in real implementation would use autodiff)
            let gradients = self.backward(&batch_data.view(), &loss_grad.view(), parameters)?;

            // Update parameters
            self.optimizer.step(parameters, &gradients)?;

            // Call on_batch_end
            self.callbacks.on_batch_end(num_batches, &self.state)?;

            num_batches += 1;

            // Logging
            if num_batches % self.config.log_frequency == 0 {
                log::debug!("Batch {}: loss={:.6}", num_batches, loss);
            }
        }

        Ok(total_loss / num_batches as f64)
    }

    /// Validate the model.
    fn validate(
        &mut self,
        val_data: &ArrayView<f64, Ix2>,
        val_targets: &ArrayView<f64, Ix2>,
        parameters: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<f64> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        let mut batch_iter = BatchIterator::new(val_data.nrows(), self.config.batch_config.clone());

        while let Some(batch_indices) = batch_iter.next_batch() {
            let batch_data = extract_batch(val_data, &batch_indices)?;
            let batch_targets = extract_batch(val_targets, &batch_indices)?;

            let predictions = self.forward(&batch_data.view(), parameters)?;
            let loss = self
                .loss_fn
                .compute(&predictions.view(), &batch_targets.view())?;

            total_loss += loss;
            num_batches += 1;
        }

        Ok(total_loss / num_batches as f64)
    }

    /// Forward pass (placeholder - actual implementation depends on model).
    fn forward(
        &self,
        data: &ArrayView<f64, Ix2>,
        _parameters: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<Array<f64, Ix2>> {
        // This is a placeholder implementation
        // In a real scenario, this would depend on the model architecture
        // For now, return input as output
        Ok(data.to_owned())
    }

    /// Backward pass (placeholder - actual implementation would use autodiff).
    fn backward(
        &self,
        _data: &ArrayView<f64, Ix2>,
        _loss_grad: &ArrayView<f64, Ix2>,
        parameters: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>> {
        // This is a placeholder implementation
        // In a real scenario, this would use automatic differentiation
        let mut gradients = HashMap::new();

        for (name, param) in parameters {
            // Simple gradient (placeholder)
            gradients.insert(name.clone(), Array::zeros(param.raw_dim()));
        }

        Ok(gradients)
    }

    /// Get current training state.
    pub fn get_state(&self) -> &TrainingState {
        &self.state
    }

    /// Save a complete training checkpoint.
    ///
    /// This saves all state needed to resume training, including:
    /// - Model parameters
    /// - Optimizer state
    /// - Scheduler state (if present)
    /// - Training history
    /// - Current epoch and losses
    pub fn save_checkpoint(
        &self,
        path: &std::path::PathBuf,
        parameters: &HashMap<String, Array<f64, Ix2>>,
        history: &TrainingHistory,
        best_val_loss: Option<f64>,
    ) -> TrainResult<()> {
        use crate::TrainingCheckpoint;

        // Get optimizer state
        let optimizer_state = self.optimizer.state_dict();

        // Get scheduler state if present
        let scheduler_state = self.scheduler.as_ref().map(|s| s.state_dict());

        // Create checkpoint
        let checkpoint = TrainingCheckpoint::new(
            self.state.epoch,
            parameters,
            &optimizer_state,
            scheduler_state,
            &self.state,
            &history.train_loss,
            &history.val_loss,
            &history.metrics,
            best_val_loss,
        );

        // Save to file
        checkpoint.save(path)?;

        println!("Training checkpoint saved to {:?}", path);
        Ok(())
    }

    /// Resume training from a checkpoint.
    ///
    /// This restores all training state including parameters, optimizer state,
    /// and history. Training will resume from the saved epoch.
    ///
    /// Returns the restored parameters, history, and starting epoch.
    #[allow(clippy::type_complexity)]
    pub fn load_checkpoint(
        &mut self,
        path: &std::path::PathBuf,
    ) -> TrainResult<(HashMap<String, Array<f64, Ix2>>, TrainingHistory, usize)> {
        use crate::TrainingCheckpoint;
        use scirs2_core::ndarray::Array;

        // Load checkpoint
        let checkpoint = TrainingCheckpoint::load(path)?;

        println!(
            "Loading checkpoint from epoch {} (val_loss: {:?})",
            checkpoint.epoch, checkpoint.val_loss
        );

        // Restore parameters
        let mut parameters = HashMap::new();
        for (name, values) in checkpoint.parameters {
            // Note: We need to know the shape to reconstruct the array
            // For now, we'll create a dummy shape. In practice, this would need
            // to be handled by the model's load_state_dict method
            let len = values.len();
            let array = Array::from_vec(values);
            // This is a limitation - we need shape information
            // In real usage, the model should handle this via its load_state_dict
            parameters.insert(
                name,
                array.into_shape_with_order((1, len)).map_err(|e| {
                    crate::TrainError::CheckpointError(format!(
                        "Failed to reshape parameter: {}",
                        e
                    ))
                })?,
            );
        }

        // Restore optimizer state
        self.optimizer.load_state_dict(checkpoint.optimizer_state);

        // Restore scheduler state
        if let (Some(scheduler), Some(scheduler_state)) =
            (self.scheduler.as_mut(), checkpoint.scheduler_state.as_ref())
        {
            scheduler.load_state_dict(scheduler_state)?;
        }

        // Restore training history
        let history = TrainingHistory {
            train_loss: checkpoint.train_loss_history,
            val_loss: checkpoint.val_loss_history,
            metrics: checkpoint.metrics_history,
        };

        // Restore training state
        self.state.epoch = checkpoint.epoch;
        self.state.train_loss = checkpoint.train_loss;
        self.state.val_loss = checkpoint.val_loss;
        self.state.learning_rate = checkpoint.learning_rate;

        println!(
            "Checkpoint loaded successfully. Resuming from epoch {}",
            checkpoint.epoch + 1
        );

        Ok((parameters, history, checkpoint.epoch))
    }

    /// Train the model starting from a checkpoint.
    ///
    /// This is a convenience method that loads a checkpoint and continues training.
    #[allow(clippy::type_complexity)]
    pub fn train_from_checkpoint(
        &mut self,
        checkpoint_path: &std::path::PathBuf,
        train_data: &ArrayView<f64, Ix2>,
        train_targets: &ArrayView<f64, Ix2>,
        val_data: Option<&ArrayView<f64, Ix2>>,
        val_targets: Option<&ArrayView<f64, Ix2>>,
    ) -> TrainResult<(HashMap<String, Array<f64, Ix2>>, TrainingHistory)> {
        // Load checkpoint
        let (mut parameters, mut history, start_epoch) = self.load_checkpoint(checkpoint_path)?;

        // Adjust config to continue from checkpoint epoch
        let remaining_epochs = self.config.num_epochs.saturating_sub(start_epoch + 1);
        let original_num_epochs = self.config.num_epochs;
        self.config.num_epochs = remaining_epochs;

        println!(
            "Resuming training: {} epochs completed, {} epochs remaining",
            start_epoch + 1,
            remaining_epochs
        );

        // Continue training
        let continued_history = self.train(
            train_data,
            train_targets,
            val_data,
            val_targets,
            &mut parameters,
        )?;

        // Restore original config
        self.config.num_epochs = original_num_epochs;

        // Merge histories
        history.train_loss.extend(continued_history.train_loss);
        history.val_loss.extend(continued_history.val_loss);
        for (metric_name, values) in continued_history.metrics {
            history
                .metrics
                .entry(metric_name)
                .or_default()
                .extend(values);
        }

        Ok((parameters, history))
    }
}

/// Training history containing losses and metrics.
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Training loss per epoch.
    pub train_loss: Vec<f64>,
    /// Validation loss per epoch.
    pub val_loss: Vec<f64>,
    /// Metrics per epoch.
    pub metrics: HashMap<String, Vec<f64>>,
}

impl TrainingHistory {
    /// Create a new training history.
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            metrics: HashMap::new(),
        }
    }

    /// Get best validation loss and corresponding epoch.
    pub fn best_val_loss(&self) -> Option<(usize, f64)> {
        self.val_loss
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &loss)| (idx, loss))
    }

    /// Get metric history.
    pub fn get_metric_history(&self, metric_name: &str) -> Option<&Vec<f64>> {
        self.metrics.get(metric_name)
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MseLoss, OptimizerConfig, SgdOptimizer};

    #[test]
    fn test_trainer_creation() {
        let config = TrainerConfig {
            num_epochs: 5,
            ..Default::default()
        };

        let loss = Box::new(MseLoss);
        let optimizer = Box::new(SgdOptimizer::new(OptimizerConfig::default()));

        let trainer = Trainer::new(config, loss, optimizer);
        assert_eq!(trainer.config.num_epochs, 5);
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();
        history.train_loss.push(1.0);
        history.train_loss.push(0.8);
        history.train_loss.push(0.6);

        history.val_loss.push(1.2);
        history.val_loss.push(0.9);
        history.val_loss.push(0.7);

        let (best_epoch, best_loss) = history.best_val_loss().unwrap();
        assert_eq!(best_epoch, 2);
        assert_eq!(best_loss, 0.7);
    }

    #[test]
    fn test_training_state() {
        let state = TrainingState {
            epoch: 5,
            batch: 100,
            train_loss: 0.5,
            val_loss: Some(0.6),
            batch_loss: 0.4,
            learning_rate: 0.001,
            metrics: HashMap::new(),
        };

        assert_eq!(state.epoch, 5);
        assert_eq!(state.batch, 100);
        assert!((state.train_loss - 0.5).abs() < 1e-6);
    }
}
