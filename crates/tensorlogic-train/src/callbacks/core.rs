//! Core callback infrastructure for training.

use crate::{TrainResult, TrainingState};

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
}
