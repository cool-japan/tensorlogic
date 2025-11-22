//! Checkpoint callbacks for saving and loading training state.

use crate::callbacks::core::Callback;
use crate::{TrainError, TrainResult, TrainingState};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

/// Compression method for checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckpointCompression {
    /// No compression (plain JSON).
    #[default]
    None,
    /// Gzip compression (good balance of speed and ratio).
    Gzip,
    /// Fast gzip compression (faster but lower ratio).
    GzipFast,
    /// Best gzip compression (slower but better ratio).
    GzipBest,
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
        self.save_with_compression(path, CheckpointCompression::None)
    }

    /// Save checkpoint to a file with compression.
    ///
    /// # Arguments
    /// * `path` - Path to save the checkpoint
    /// * `compression` - Compression method to use
    ///
    /// # Example
    /// ```no_run
    /// use tensorlogic_train::TrainingCheckpoint;
    /// use tensorlogic_train::CheckpointCompression;
    /// use std::path::PathBuf;
    ///
    /// // Assuming you have a checkpoint...
    /// # let checkpoint: TrainingCheckpoint = unimplemented!();
    ///
    /// // Save with gzip compression
    /// checkpoint.save_with_compression(
    ///     &PathBuf::from("/tmp/checkpoint.json.gz"),
    ///     CheckpointCompression::Gzip
    /// ).unwrap();
    /// ```
    pub fn save_with_compression(
        &self,
        path: &PathBuf,
        compression: CheckpointCompression,
    ) -> TrainResult<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to serialize checkpoint: {}", e))
        })?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                TrainError::CheckpointError(format!("Failed to create checkpoint directory: {}", e))
            })?;
        }

        match compression {
            CheckpointCompression::None => {
                std::fs::write(path, json).map_err(|e| {
                    TrainError::CheckpointError(format!("Failed to write checkpoint: {}", e))
                })?;
            }
            CheckpointCompression::Gzip => {
                let file = File::create(path).map_err(|e| {
                    TrainError::CheckpointError(format!("Failed to create checkpoint file: {}", e))
                })?;
                let mut encoder = GzEncoder::new(file, Compression::default());
                encoder.write_all(json.as_bytes()).map_err(|e| {
                    TrainError::CheckpointError(format!("Failed to compress checkpoint: {}", e))
                })?;
                encoder.finish().map_err(|e| {
                    TrainError::CheckpointError(format!("Failed to finish compression: {}", e))
                })?;
            }
            CheckpointCompression::GzipFast => {
                let file = File::create(path).map_err(|e| {
                    TrainError::CheckpointError(format!("Failed to create checkpoint file: {}", e))
                })?;
                let mut encoder = GzEncoder::new(file, Compression::fast());
                encoder.write_all(json.as_bytes()).map_err(|e| {
                    TrainError::CheckpointError(format!("Failed to compress checkpoint: {}", e))
                })?;
                encoder.finish().map_err(|e| {
                    TrainError::CheckpointError(format!("Failed to finish compression: {}", e))
                })?;
            }
            CheckpointCompression::GzipBest => {
                let file = File::create(path).map_err(|e| {
                    TrainError::CheckpointError(format!("Failed to create checkpoint file: {}", e))
                })?;
                let mut encoder = GzEncoder::new(file, Compression::best());
                encoder.write_all(json.as_bytes()).map_err(|e| {
                    TrainError::CheckpointError(format!("Failed to compress checkpoint: {}", e))
                })?;
                encoder.finish().map_err(|e| {
                    TrainError::CheckpointError(format!("Failed to finish compression: {}", e))
                })?;
            }
        }

        Ok(())
    }

    /// Load checkpoint from a file.
    pub fn load(path: &PathBuf) -> TrainResult<Self> {
        // Auto-detect compression based on file extension
        if path.to_string_lossy().ends_with(".gz") {
            Self::load_compressed(path)
        } else {
            Self::load_uncompressed(path)
        }
    }

    /// Load uncompressed checkpoint from a file.
    fn load_uncompressed(path: &PathBuf) -> TrainResult<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to read checkpoint: {}", e))
        })?;

        let checkpoint: Self = serde_json::from_str(&json).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to deserialize checkpoint: {}", e))
        })?;

        Ok(checkpoint)
    }

    /// Load compressed checkpoint from a file.
    pub fn load_compressed(path: &PathBuf) -> TrainResult<Self> {
        let file = File::open(path).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to open checkpoint file: {}", e))
        })?;

        let mut decoder = GzDecoder::new(file);
        let mut json = String::new();
        decoder.read_to_string(&mut json).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to decompress checkpoint: {}", e))
        })?;

        let checkpoint: Self = serde_json::from_str(&json).map_err(|e| {
            TrainError::CheckpointError(format!("Failed to deserialize checkpoint: {}", e))
        })?;

        Ok(checkpoint)
    }

    /// Get the size of the checkpoint in bytes (estimated).
    pub fn estimated_size(&self) -> usize {
        // Rough estimate: parameters + optimizer_state + histories
        let param_size: usize = self
            .parameters
            .values()
            .map(|v| v.len() * std::mem::size_of::<f64>())
            .sum();
        let optimizer_size: usize = self
            .optimizer_state
            .values()
            .map(|v| v.len() * std::mem::size_of::<f64>())
            .sum();
        let history_size = (self.train_loss_history.len() + self.val_loss_history.len())
            * std::mem::size_of::<f64>();

        param_size + optimizer_size + history_size
    }
}

/// Checkpoint metadata for tracking saved checkpoints.
#[derive(Debug, Clone, PartialEq)]
struct CheckpointMetadata {
    /// Epoch number.
    epoch: usize,
    /// Validation loss (if available).
    val_loss: Option<f64>,
    /// File path.
    path: PathBuf,
}

/// Callback for model checkpointing with auto-cleanup.
pub struct CheckpointCallback {
    /// Directory to save checkpoints.
    pub checkpoint_dir: PathBuf,
    /// Frequency of checkpointing (every N epochs).
    pub save_frequency: usize,
    /// Whether to save only the best model.
    pub save_best_only: bool,
    /// Maximum number of checkpoints to keep (None = keep all).
    pub keep_top_k: Option<usize>,
    /// Best validation loss seen so far.
    best_val_loss: Option<f64>,
    /// Metadata of saved checkpoints for cleanup.
    saved_checkpoints: Vec<CheckpointMetadata>,
}

impl CheckpointCallback {
    /// Create a new checkpoint callback.
    pub fn new(checkpoint_dir: PathBuf, save_frequency: usize, save_best_only: bool) -> Self {
        Self {
            checkpoint_dir,
            save_frequency,
            save_best_only,
            keep_top_k: None,
            best_val_loss: None,
            saved_checkpoints: Vec::new(),
        }
    }

    /// Create a new checkpoint callback with auto-cleanup.
    ///
    /// This will automatically delete old checkpoints when the number exceeds `keep_top_k`,
    /// keeping only the checkpoints with the best (lowest) validation loss.
    ///
    /// # Arguments
    /// * `checkpoint_dir` - Directory to save checkpoints
    /// * `save_frequency` - Save every N epochs
    /// * `save_best_only` - Only save when validation loss improves
    /// * `keep_top_k` - Maximum number of checkpoints to keep (keeps best by validation loss)
    ///
    /// # Example
    /// ```no_run
    /// use tensorlogic_train::CheckpointCallback;
    /// use std::path::PathBuf;
    ///
    /// // Keep only the top 5 best checkpoints
    /// let callback = CheckpointCallback::with_cleanup(
    ///     PathBuf::from("/tmp/checkpoints"),
    ///     1,    // save every epoch
    ///     false, // save all, not just best
    ///     5     // keep top 5
    /// );
    /// ```
    pub fn with_cleanup(
        checkpoint_dir: PathBuf,
        save_frequency: usize,
        save_best_only: bool,
        keep_top_k: usize,
    ) -> Self {
        Self {
            checkpoint_dir,
            save_frequency,
            save_best_only,
            keep_top_k: Some(keep_top_k),
            best_val_loss: None,
            saved_checkpoints: Vec::new(),
        }
    }

    /// Get the number of saved checkpoints being tracked.
    pub fn num_saved_checkpoints(&self) -> usize {
        self.saved_checkpoints.len()
    }

    /// Manually cleanup checkpoints, keeping only the top-k best.
    ///
    /// This can be called manually to trigger cleanup if you've changed the
    /// `keep_top_k` setting.
    pub fn cleanup_checkpoints(&mut self) -> TrainResult<usize> {
        let keep_top_k = match self.keep_top_k {
            Some(k) => k,
            None => return Ok(0), // No cleanup needed
        };

        if self.saved_checkpoints.len() <= keep_top_k {
            return Ok(0); // Don't need to clean up yet
        }

        // Sort by validation loss (ascending - best first)
        // For checkpoints without val_loss, prefer more recent epochs
        self.saved_checkpoints.sort_by(|a, b| {
            match (a.val_loss, b.val_loss) {
                (Some(a_loss), Some(b_loss)) => a_loss
                    .partial_cmp(&b_loss)
                    .unwrap_or(std::cmp::Ordering::Equal),
                (Some(_), None) => std::cmp::Ordering::Less, // Prefer checkpoints with val_loss
                (None, Some(_)) => std::cmp::Ordering::Greater, // Prefer checkpoints with val_loss
                (None, None) => b.epoch.cmp(&a.epoch),       // Prefer newer epochs (descending)
            }
        });

        // Remove checkpoints beyond top-k
        let to_remove: Vec<CheckpointMetadata> =
            self.saved_checkpoints.drain(keep_top_k..).collect();

        let mut deleted_count = 0;
        for checkpoint in to_remove {
            if let Err(e) = std::fs::remove_file(&checkpoint.path) {
                eprintln!(
                    "Warning: Failed to delete checkpoint {:?}: {}",
                    checkpoint.path, e
                );
            } else {
                deleted_count += 1;
            }
        }

        Ok(deleted_count)
    }

    /// Save checkpoint to disk (legacy simple format).
    fn save_checkpoint(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
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

        // Track checkpoint metadata
        let metadata = CheckpointMetadata {
            epoch,
            val_loss: state.val_loss,
            path: checkpoint_path.clone(),
        };
        self.saved_checkpoints.push(metadata);

        // Auto-cleanup if needed
        if self.keep_top_k.is_some() {
            let deleted = self.cleanup_checkpoints()?;
            if deleted > 0 {
                println!(
                    "Checkpoint saved to {:?} (deleted {} old checkpoints)",
                    checkpoint_path, deleted
                );
            } else {
                println!("Checkpoint saved to {:?}", checkpoint_path);
            }
        } else {
            println!("Checkpoint saved to {:?}", checkpoint_path);
        }

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

    #[test]
    fn test_checkpoint_compression_gzip() {
        let mut parameters = HashMap::new();
        parameters.insert("weights".to_string(), Array2::from_elem((100, 100), 1.5));

        let state = create_test_state();
        let optimizer_state = HashMap::new();

        let checkpoint = TrainingCheckpoint::new(
            10,
            &parameters,
            &optimizer_state,
            None,
            &state,
            &vec![1.0; 100],
            &vec![0.9; 100],
            &HashMap::new(),
            Some(0.5),
        );

        // Save with gzip compression
        let compressed_path = temp_dir().join("test_checkpoint_compressed.json.gz");
        checkpoint
            .save_with_compression(&compressed_path, CheckpointCompression::Gzip)
            .unwrap();

        // Verify compressed file exists
        assert!(compressed_path.exists());

        // Load compressed checkpoint
        let loaded = TrainingCheckpoint::load(&compressed_path).unwrap();

        // Verify data
        assert_eq!(loaded.epoch, 10);
        assert_eq!(loaded.parameters.len(), 1);
        assert_eq!(loaded.parameters["weights"].len(), 10000); // 100x100

        // Compare file sizes
        let uncompressed_path = temp_dir().join("test_checkpoint_uncompressed.json");
        checkpoint.save(&uncompressed_path).unwrap();

        let compressed_size = std::fs::metadata(&compressed_path).unwrap().len();
        let uncompressed_size = std::fs::metadata(&uncompressed_path).unwrap().len();

        // Compressed should be smaller
        assert!(
            compressed_size < uncompressed_size,
            "Compressed size {} should be less than uncompressed size {}",
            compressed_size,
            uncompressed_size
        );

        // Clean up
        std::fs::remove_file(compressed_path).ok();
        std::fs::remove_file(uncompressed_path).ok();
    }

    #[test]
    fn test_checkpoint_compression_fast_vs_best() {
        let mut parameters = HashMap::new();
        parameters.insert("weights".to_string(), Array2::from_elem((50, 50), 2.0));

        let state = create_test_state();
        let optimizer_state = HashMap::new();

        let checkpoint = TrainingCheckpoint::new(
            5,
            &parameters,
            &optimizer_state,
            None,
            &state,
            &vec![1.0; 50],
            &vec![0.8; 50],
            &HashMap::new(),
            None,
        );

        // Save with fast compression
        let fast_path = temp_dir().join("test_checkpoint_fast.json.gz");
        checkpoint
            .save_with_compression(&fast_path, CheckpointCompression::GzipFast)
            .unwrap();

        // Save with best compression
        let best_path = temp_dir().join("test_checkpoint_best.json.gz");
        checkpoint
            .save_with_compression(&best_path, CheckpointCompression::GzipBest)
            .unwrap();

        // Both should be loadable
        let loaded_fast = TrainingCheckpoint::load(&fast_path).unwrap();
        let loaded_best = TrainingCheckpoint::load(&best_path).unwrap();

        assert_eq!(loaded_fast.epoch, 5);
        assert_eq!(loaded_best.epoch, 5);
        assert_eq!(
            loaded_fast.parameters["weights"],
            loaded_best.parameters["weights"]
        );

        // Clean up
        std::fs::remove_file(fast_path).ok();
        std::fs::remove_file(best_path).ok();
    }

    #[test]
    fn test_checkpoint_estimated_size() {
        let mut parameters = HashMap::new();
        parameters.insert("w1".to_string(), Array2::from_elem((10, 10), 1.0));
        parameters.insert("w2".to_string(), Array2::from_elem((5, 5), 1.0));

        let state = create_test_state();
        let optimizer_state = HashMap::new();

        let train_loss_history: [f64; 10] = [1.0; 10];
        let val_loss_history: [f64; 10] = [0.9; 10];
        let checkpoint = TrainingCheckpoint::new(
            1,
            &parameters,
            &optimizer_state,
            None,
            &state,
            &train_loss_history,
            &val_loss_history,
            &HashMap::new(),
            None,
        );

        let size = checkpoint.estimated_size();
        // 100 + 25 = 125 parameters * 8 bytes + 20 history entries * 8 bytes
        assert!(size > 0);
        assert_eq!(
            size,
            (100 + 25) * std::mem::size_of::<f64>() + 20 * std::mem::size_of::<f64>()
        );
    }

    #[test]
    fn test_checkpoint_auto_detect_compression() {
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), Array2::from_elem((5, 5), 1.0));

        let state = create_test_state();

        let checkpoint = TrainingCheckpoint::new(
            1,
            &parameters,
            &HashMap::new(),
            None,
            &state,
            &[1.0],
            &[0.9],
            &HashMap::new(),
            None,
        );

        // Save uncompressed
        let uncompressed_path = temp_dir().join("test_auto_detect.json");
        checkpoint.save(&uncompressed_path).unwrap();

        // Save compressed
        let compressed_path = temp_dir().join("test_auto_detect.json.gz");
        checkpoint
            .save_with_compression(&compressed_path, CheckpointCompression::Gzip)
            .unwrap();

        // Load both using auto-detection
        let loaded_uncompressed = TrainingCheckpoint::load(&uncompressed_path).unwrap();
        let loaded_compressed = TrainingCheckpoint::load(&compressed_path).unwrap();

        assert_eq!(loaded_uncompressed.epoch, loaded_compressed.epoch);
        assert_eq!(loaded_uncompressed.parameters, loaded_compressed.parameters);

        // Clean up
        std::fs::remove_file(uncompressed_path).ok();
        std::fs::remove_file(compressed_path).ok();
    }

    #[test]
    fn test_checkpoint_auto_cleanup() {
        let checkpoint_dir = temp_dir().join("tensorlogic_test_auto_cleanup");
        std::fs::create_dir_all(&checkpoint_dir).ok();

        // Create callback with keep_top_k = 3
        let mut callback = CheckpointCallback::with_cleanup(checkpoint_dir.clone(), 1, false, 3);

        // Save 5 checkpoints with different validation losses
        let val_losses = [0.9, 0.7, 0.8, 0.6, 0.5]; // Best is 0.5, then 0.6, then 0.7

        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            let mut state = create_test_state();
            state.val_loss = Some(val_loss);
            callback.save_checkpoint(epoch, &state).unwrap();
        }

        // Should only have 3 checkpoints remaining (top 3 best)
        assert_eq!(callback.num_saved_checkpoints(), 3);

        // Verify the best 3 checkpoints exist
        assert!(checkpoint_dir.join("checkpoint_epoch_4.json").exists()); // val_loss = 0.5
        assert!(checkpoint_dir.join("checkpoint_epoch_3.json").exists()); // val_loss = 0.6
        assert!(checkpoint_dir.join("checkpoint_epoch_1.json").exists()); // val_loss = 0.7

        // Verify the worst 2 were deleted
        assert!(!checkpoint_dir.join("checkpoint_epoch_0.json").exists()); // val_loss = 0.9
        assert!(!checkpoint_dir.join("checkpoint_epoch_2.json").exists()); // val_loss = 0.8

        // Clean up
        std::fs::remove_dir_all(checkpoint_dir).ok();
    }

    #[test]
    fn test_checkpoint_no_cleanup_when_disabled() {
        let checkpoint_dir = temp_dir().join("tensorlogic_test_no_cleanup");
        std::fs::create_dir_all(&checkpoint_dir).ok();

        // Create callback without cleanup (keep_top_k = None)
        let mut callback = CheckpointCallback::new(checkpoint_dir.clone(), 1, false);

        // Save 5 checkpoints
        for epoch in 0..5 {
            let state = create_test_state();
            callback.save_checkpoint(epoch, &state).unwrap();
        }

        // All 5 checkpoints should still exist
        for epoch in 0..5 {
            let path = checkpoint_dir.join(format!("checkpoint_epoch_{}.json", epoch));
            assert!(path.exists(), "Checkpoint {} should exist", epoch);
        }

        // Clean up
        std::fs::remove_dir_all(checkpoint_dir).ok();
    }

    #[test]
    fn test_checkpoint_manual_cleanup() {
        let checkpoint_dir = temp_dir().join("tensorlogic_test_manual_cleanup");
        std::fs::create_dir_all(&checkpoint_dir).ok();

        // Create callback with keep_top_k = 2
        let mut callback = CheckpointCallback::with_cleanup(checkpoint_dir.clone(), 1, false, 2);

        // Save 4 checkpoints
        let val_losses = [0.8, 0.6, 0.9, 0.5];
        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            let mut state = create_test_state();
            state.val_loss = Some(val_loss);
            callback.save_checkpoint(epoch, &state).unwrap();
        }

        // Should have only top 2
        assert_eq!(callback.num_saved_checkpoints(), 2);

        // Manually trigger cleanup (should do nothing since we're already at top-2)
        let deleted = callback.cleanup_checkpoints().unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(callback.num_saved_checkpoints(), 2);

        // Clean up
        std::fs::remove_dir_all(checkpoint_dir).ok();
    }

    #[test]
    fn test_checkpoint_cleanup_without_val_loss() {
        let checkpoint_dir = temp_dir().join("tensorlogic_test_cleanup_no_val_loss");
        std::fs::create_dir_all(&checkpoint_dir).ok();

        // Create callback with keep_top_k = 2
        let mut callback = CheckpointCallback::with_cleanup(checkpoint_dir.clone(), 1, false, 2);

        // Save 4 checkpoints without validation loss
        for epoch in 0..4 {
            let mut state = create_test_state();
            state.val_loss = None; // No validation loss
            callback.save_checkpoint(epoch, &state).unwrap();
        }

        // Should keep top 2 (most recent by epoch)
        assert_eq!(callback.num_saved_checkpoints(), 2);

        // Verify most recent 2 epochs exist
        assert!(checkpoint_dir.join("checkpoint_epoch_3.json").exists());
        assert!(checkpoint_dir.join("checkpoint_epoch_2.json").exists());

        // Clean up
        std::fs::remove_dir_all(checkpoint_dir).ok();
    }

    #[test]
    fn test_checkpoint_with_save_best_only_and_cleanup() {
        let checkpoint_dir = temp_dir().join("tensorlogic_test_best_and_cleanup");
        std::fs::create_dir_all(&checkpoint_dir).ok();

        // Create callback with both save_best_only and keep_top_k
        let mut callback = CheckpointCallback::with_cleanup(checkpoint_dir.clone(), 1, true, 2);

        // Try to save checkpoints with improving and non-improving losses
        let val_losses = [0.9, 0.7, 0.8, 0.6]; // 0.9 -> 0.7 (save), 0.8 (skip), 0.6 (save)

        for (epoch, &val_loss) in val_losses.iter().enumerate() {
            let mut state = create_test_state();
            state.val_loss = Some(val_loss);
            callback.on_epoch_end(epoch, &state).unwrap();
        }

        // Should only have saved the improving checkpoints (0.9, 0.7, 0.6), then cleaned up to top-2
        assert!(callback.num_saved_checkpoints() <= 2);

        // Clean up
        std::fs::remove_dir_all(checkpoint_dir).ok();
    }
}
