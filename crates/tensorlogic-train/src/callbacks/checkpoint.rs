//! Checkpoint callbacks for saving and loading training state.

use crate::callbacks::core::Callback;
use crate::{TrainError, TrainResult, TrainingState};
use std::collections::HashMap;
use std::path::PathBuf;

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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use std::env::temp_dir;

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
    fn test_checkpoint_callback() {
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
