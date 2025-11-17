//! Early stopping and learning rate reduction callbacks.

use crate::callbacks::core::Callback;
use crate::{TrainResult, TrainingState};

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
}
