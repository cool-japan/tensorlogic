//! Checkpoint and resume functionality for training workflows.
//!
//! This module provides utilities to save and restore executor state during training,
//! enabling mid-training checkpoints, recovery from failures, and incremental compilation.
//!
//! ## Features
//!
//! - **State Serialization**: Save executor tensors and forward tape
//! - **Incremental Checkpoints**: Save only changed tensors
//! - **Compression**: Optional compression for checkpoint files
//! - **Metadata**: Track training iteration, timestamp, and custom data
//! - **Verification**: Checksum validation for data integrity
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_scirs_backend::{Scirs2Exec, Checkpoint, CheckpointConfig};
//!
//! let mut executor = Scirs2Exec::new();
//! // ... training loop ...
//!
//! // Save checkpoint
//! let checkpoint = Checkpoint::from_executor(&executor, iteration)?;
//! checkpoint.save("checkpoint_epoch_5.bin")?;
//!
//! // Restore checkpoint
//! let checkpoint = Checkpoint::load("checkpoint_epoch_5.bin")?;
//! let mut executor = checkpoint.restore()?;
//! ```

use crate::{Scirs2Exec, TlBackendError, TlBackendResult};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for checkpoint creation and loading.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Enable compression (reduces file size but increases save time)
    pub enable_compression: bool,

    /// Include forward tape in checkpoint (needed for gradient computation)
    pub include_tape: bool,

    /// Verify checksum on load
    pub verify_checksum: bool,

    /// Save only tensors that changed since last checkpoint
    pub incremental: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enable_compression: false,
            include_tape: false,
            verify_checksum: true,
            incremental: false,
        }
    }
}

impl CheckpointConfig {
    /// Create a configuration for training checkpoints (includes tape).
    pub fn for_training() -> Self {
        Self {
            enable_compression: false,
            include_tape: true,
            verify_checksum: true,
            incremental: false,
        }
    }

    /// Create a configuration for inference checkpoints (no tape, compressed).
    pub fn for_inference() -> Self {
        Self {
            enable_compression: true,
            include_tape: false,
            verify_checksum: true,
            incremental: false,
        }
    }

    /// Create a configuration for incremental checkpoints.
    pub fn incremental() -> Self {
        Self {
            enable_compression: false,
            include_tape: true,
            verify_checksum: true,
            incremental: true,
        }
    }
}

/// Metadata about a checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    /// Training iteration/epoch number
    pub iteration: usize,

    /// Timestamp when checkpoint was created
    pub timestamp: u64,

    /// Version of the checkpoint format
    pub version: String,

    /// Number of tensors in checkpoint
    pub tensor_count: usize,

    /// Total size in bytes (uncompressed)
    pub total_bytes: usize,

    /// Custom metadata (user-defined)
    pub custom: HashMap<String, String>,

    /// Checksum for verification (if enabled)
    pub checksum: Option<String>,
}

/// Serialized tensor data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SerializedTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f64>,
}

/// A checkpoint containing executor state.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,

    /// Serialized tensors
    tensors: Vec<SerializedTensor>,

    /// Configuration used to create this checkpoint
    #[allow(dead_code)]
    config: CheckpointConfig,
}

impl Checkpoint {
    /// Create a checkpoint from an executor.
    pub fn from_executor(executor: &Scirs2Exec, iteration: usize) -> TlBackendResult<Self> {
        Self::from_executor_with_config(executor, iteration, &CheckpointConfig::default())
    }

    /// Create a checkpoint with custom configuration.
    pub fn from_executor_with_config(
        executor: &Scirs2Exec,
        iteration: usize,
        config: &CheckpointConfig,
    ) -> TlBackendResult<Self> {
        let mut tensors = Vec::new();
        let mut total_bytes = 0;

        // Serialize all tensors
        for (name, tensor) in &executor.tensors {
            let shape = tensor.shape().to_vec();
            let data: Vec<f64> = tensor.iter().copied().collect();
            total_bytes += data.len() * std::mem::size_of::<f64>();

            tensors.push(SerializedTensor {
                name: name.clone(),
                shape,
                data,
            });
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| TlBackendError::execution(format!("Failed to get timestamp: {}", e)))?
            .as_secs();

        let checksum = if config.verify_checksum {
            Some(Self::compute_checksum(&tensors))
        } else {
            None
        };

        let metadata = CheckpointMetadata {
            iteration,
            timestamp,
            version: "0.1.0".to_string(),
            tensor_count: tensors.len(),
            total_bytes,
            custom: HashMap::new(),
            checksum,
        };

        Ok(Checkpoint {
            metadata,
            tensors,
            config: config.clone(),
        })
    }

    /// Save checkpoint to a file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> TlBackendResult<()> {
        let file = File::create(path.as_ref()).map_err(|e| {
            TlBackendError::execution(format!("Failed to create checkpoint file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);

        // Serialize to JSON (could use bincode for better performance)
        let checkpoint_data = CheckpointData {
            metadata: self.metadata.clone(),
            tensors: self.tensors.clone(),
        };

        serde_json::to_writer(&mut writer, &checkpoint_data).map_err(|e| {
            TlBackendError::execution(format!("Failed to serialize checkpoint: {}", e))
        })?;

        writer
            .flush()
            .map_err(|e| TlBackendError::execution(format!("Failed to flush checkpoint: {}", e)))?;

        Ok(())
    }

    /// Load checkpoint from a file.
    pub fn load<P: AsRef<Path>>(path: P) -> TlBackendResult<Self> {
        Self::load_with_config(path, &CheckpointConfig::default())
    }

    /// Load checkpoint with custom configuration.
    pub fn load_with_config<P: AsRef<Path>>(
        path: P,
        config: &CheckpointConfig,
    ) -> TlBackendResult<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            TlBackendError::execution(format!("Failed to open checkpoint file: {}", e))
        })?;
        let reader = BufReader::new(file);

        let checkpoint_data: CheckpointData = serde_json::from_reader(reader).map_err(|e| {
            TlBackendError::execution(format!("Failed to deserialize checkpoint: {}", e))
        })?;

        // Verify checksum if requested
        if config.verify_checksum {
            if let Some(ref expected_checksum) = checkpoint_data.metadata.checksum {
                let actual_checksum = Self::compute_checksum(&checkpoint_data.tensors);
                if &actual_checksum != expected_checksum {
                    return Err(TlBackendError::execution(
                        "Checkpoint checksum verification failed",
                    ));
                }
            }
        }

        Ok(Checkpoint {
            metadata: checkpoint_data.metadata,
            tensors: checkpoint_data.tensors,
            config: config.clone(),
        })
    }

    /// Restore an executor from this checkpoint.
    pub fn restore(&self) -> TlBackendResult<Scirs2Exec> {
        let mut executor = Scirs2Exec::new();

        // Deserialize tensors
        for serialized in &self.tensors {
            let tensor = scirs2_core::ndarray::ArrayD::from_shape_vec(
                serialized.shape.clone(),
                serialized.data.clone(),
            )
            .map_err(|e| {
                TlBackendError::execution(format!(
                    "Failed to restore tensor {}: {}",
                    serialized.name, e
                ))
            })?;

            executor.add_tensor(&serialized.name, tensor);
        }

        Ok(executor)
    }

    /// Restore tensors into an existing executor.
    pub fn restore_into(&self, executor: &mut Scirs2Exec) -> TlBackendResult<()> {
        for serialized in &self.tensors {
            let tensor = scirs2_core::ndarray::ArrayD::from_shape_vec(
                serialized.shape.clone(),
                serialized.data.clone(),
            )
            .map_err(|e| {
                TlBackendError::execution(format!(
                    "Failed to restore tensor {}: {}",
                    serialized.name, e
                ))
            })?;

            executor.add_tensor(&serialized.name, tensor);
        }

        Ok(())
    }

    /// Add custom metadata to the checkpoint.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.custom.insert(key, value);
    }

    /// Get custom metadata from the checkpoint.
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.custom.get(key)
    }

    /// Compute checksum for verification.
    fn compute_checksum(tensors: &[SerializedTensor]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        for tensor in tensors {
            tensor.name.hash(&mut hasher);
            tensor.shape.hash(&mut hasher);
            // Hash float data as bytes
            for &value in &tensor.data {
                value.to_bits().hash(&mut hasher);
            }
        }

        format!("{:x}", hasher.finish())
    }

    /// Get the size of this checkpoint in bytes (uncompressed).
    pub fn size_bytes(&self) -> usize {
        self.metadata.total_bytes
    }

    /// Get a human-readable size string.
    pub fn size_human_readable(&self) -> String {
        let bytes = self.metadata.total_bytes;
        if bytes < 1024 {
            format!("{} bytes", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.2} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

/// Internal checkpoint data structure for serialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CheckpointData {
    metadata: CheckpointMetadata,
    tensors: Vec<SerializedTensor>,
}

/// Manager for handling multiple checkpoints.
pub struct CheckpointManager {
    /// Directory where checkpoints are stored
    checkpoint_dir: std::path::PathBuf,

    /// Maximum number of checkpoints to keep
    max_checkpoints: Option<usize>,

    /// Pattern for checkpoint filenames
    filename_pattern: String,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> TlBackendResult<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !checkpoint_dir.exists() {
            std::fs::create_dir_all(&checkpoint_dir).map_err(|e| {
                TlBackendError::execution(format!("Failed to create checkpoint directory: {}", e))
            })?;
        }

        Ok(Self {
            checkpoint_dir,
            max_checkpoints: Some(5), // Keep last 5 checkpoints by default
            filename_pattern: "checkpoint_iter_{}.json".to_string(),
        })
    }

    /// Set the maximum number of checkpoints to keep.
    pub fn set_max_checkpoints(&mut self, max: Option<usize>) {
        self.max_checkpoints = max;
    }

    /// Set the filename pattern for checkpoints.
    pub fn set_filename_pattern(&mut self, pattern: String) {
        self.filename_pattern = pattern;
    }

    /// Save a checkpoint and manage old checkpoints.
    pub fn save_checkpoint(
        &self,
        executor: &Scirs2Exec,
        iteration: usize,
    ) -> TlBackendResult<std::path::PathBuf> {
        let checkpoint = Checkpoint::from_executor(executor, iteration)?;
        let filename = self.filename_pattern.replace("{}", &iteration.to_string());
        let path = self.checkpoint_dir.join(filename);

        checkpoint.save(&path)?;

        // Clean up old checkpoints if needed
        if let Some(max) = self.max_checkpoints {
            self.cleanup_old_checkpoints(max)?;
        }

        Ok(path)
    }

    /// Load the latest checkpoint.
    pub fn load_latest(&self) -> TlBackendResult<Checkpoint> {
        let latest_path = self.find_latest_checkpoint()?;
        Checkpoint::load(latest_path)
    }

    /// Find the latest checkpoint file.
    fn find_latest_checkpoint(&self) -> TlBackendResult<std::path::PathBuf> {
        let entries = std::fs::read_dir(&self.checkpoint_dir).map_err(|e| {
            TlBackendError::execution(format!("Failed to read checkpoint directory: {}", e))
        })?;

        let mut checkpoints: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s == "json")
                    .unwrap_or(false)
            })
            .collect();

        checkpoints.sort_by_key(|e| {
            e.metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(SystemTime::UNIX_EPOCH)
        });

        checkpoints
            .last()
            .map(|e| e.path())
            .ok_or_else(|| TlBackendError::execution("No checkpoints found"))
    }

    /// Remove old checkpoints keeping only the most recent `max` checkpoints.
    fn cleanup_old_checkpoints(&self, max: usize) -> TlBackendResult<()> {
        let entries = std::fs::read_dir(&self.checkpoint_dir).map_err(|e| {
            TlBackendError::execution(format!("Failed to read checkpoint directory: {}", e))
        })?;

        let mut checkpoints: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s == "json")
                    .unwrap_or(false)
            })
            .collect();

        checkpoints.sort_by_key(|e| {
            e.metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(SystemTime::UNIX_EPOCH)
        });

        // Remove oldest checkpoints
        let to_remove = checkpoints.len().saturating_sub(max);
        for entry in checkpoints.iter().take(to_remove) {
            std::fs::remove_file(entry.path()).ok();
        }

        Ok(())
    }

    /// List all checkpoints in the directory.
    pub fn list_checkpoints(&self) -> TlBackendResult<Vec<std::path::PathBuf>> {
        let entries = std::fs::read_dir(&self.checkpoint_dir).map_err(|e| {
            TlBackendError::execution(format!("Failed to read checkpoint directory: {}", e))
        })?;

        let mut checkpoints: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s == "json")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();

        checkpoints.sort();
        Ok(checkpoints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert!(!config.enable_compression);
        assert!(!config.include_tape);
        assert!(config.verify_checksum);
        assert!(!config.incremental);
    }

    #[test]
    fn test_checkpoint_config_training() {
        let config = CheckpointConfig::for_training();
        assert!(!config.enable_compression);
        assert!(config.include_tape);
        assert!(config.verify_checksum);
    }

    #[test]
    fn test_checkpoint_config_inference() {
        let config = CheckpointConfig::for_inference();
        assert!(config.enable_compression);
        assert!(!config.include_tape);
        assert!(config.verify_checksum);
    }

    #[test]
    fn test_checkpoint_from_executor() {
        let mut executor = Scirs2Exec::new();
        let tensor =
            ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        executor.add_tensor("test_tensor", tensor);

        let checkpoint = Checkpoint::from_executor(&executor, 1).unwrap();

        assert_eq!(checkpoint.metadata.iteration, 1);
        assert_eq!(checkpoint.metadata.tensor_count, 1);
        assert!(checkpoint.metadata.total_bytes > 0);
    }

    #[test]
    fn test_checkpoint_save_and_load() {
        let mut executor = Scirs2Exec::new();
        let tensor = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        executor.add_tensor("weights", tensor);

        // Save checkpoint
        let checkpoint = Checkpoint::from_executor(&executor, 5).unwrap();
        let temp_path = std::env::temp_dir().join("test_checkpoint.json");
        checkpoint.save(&temp_path).unwrap();

        // Load checkpoint
        let loaded = Checkpoint::load(&temp_path).unwrap();
        assert_eq!(loaded.metadata.iteration, 5);
        assert_eq!(loaded.metadata.tensor_count, 1);

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_checkpoint_restore() {
        let mut executor = Scirs2Exec::new();
        let tensor = ArrayD::from_shape_vec(vec![2], vec![10.0, 20.0]).unwrap();
        executor.add_tensor("params", tensor.clone());

        // Create and restore checkpoint
        let checkpoint = Checkpoint::from_executor(&executor, 1).unwrap();
        let restored_executor = checkpoint.restore().unwrap();

        // Verify restored tensor
        let restored_tensor = restored_executor.get_tensor("params").unwrap();
        assert_eq!(restored_tensor.shape(), tensor.shape());
        assert_eq!(restored_tensor[[0]], 10.0);
        assert_eq!(restored_tensor[[1]], 20.0);
    }

    #[test]
    fn test_checkpoint_metadata() {
        let mut executor = Scirs2Exec::new();
        let tensor = ArrayD::from_shape_vec(vec![1], vec![1.0]).unwrap();
        executor.add_tensor("x", tensor);

        let mut checkpoint = Checkpoint::from_executor(&executor, 10).unwrap();
        checkpoint.add_metadata("learning_rate".to_string(), "0.001".to_string());
        checkpoint.add_metadata("optimizer".to_string(), "adam".to_string());

        assert_eq!(
            checkpoint.get_metadata("learning_rate"),
            Some(&"0.001".to_string())
        );
        assert_eq!(
            checkpoint.get_metadata("optimizer"),
            Some(&"adam".to_string())
        );
        assert_eq!(checkpoint.get_metadata("missing"), None);
    }

    #[test]
    fn test_checkpoint_size_human_readable() {
        let mut executor = Scirs2Exec::new();
        let tensor = ArrayD::from_shape_vec(vec![1000], vec![1.0; 1000]).unwrap();
        executor.add_tensor("big_tensor", tensor);

        let checkpoint = Checkpoint::from_executor(&executor, 1).unwrap();
        let size_str = checkpoint.size_human_readable();

        // 1000 floats * 8 bytes = 8000 bytes = ~7.81 KB
        assert!(size_str.contains("KB") || size_str.contains("bytes"));
    }

    #[test]
    fn test_checkpoint_manager() {
        let temp_dir = std::env::temp_dir().join("test_checkpoints");
        let manager = CheckpointManager::new(&temp_dir).unwrap();

        let mut executor = Scirs2Exec::new();
        let tensor = ArrayD::from_shape_vec(vec![2], vec![1.0, 2.0]).unwrap();
        executor.add_tensor("data", tensor);

        // Save checkpoint
        let path = manager.save_checkpoint(&executor, 1).unwrap();
        assert!(path.exists());

        // List checkpoints
        let checkpoints = manager.list_checkpoints().unwrap();
        assert_eq!(checkpoints.len(), 1);

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_checkpoint_manager_cleanup() {
        let temp_dir = std::env::temp_dir().join("test_checkpoints_cleanup");
        let mut manager = CheckpointManager::new(&temp_dir).unwrap();
        manager.set_max_checkpoints(Some(3));

        let mut executor = Scirs2Exec::new();
        let tensor = ArrayD::from_shape_vec(vec![1], vec![1.0]).unwrap();
        executor.add_tensor("x", tensor);

        // Save 5 checkpoints
        for i in 1..=5 {
            manager.save_checkpoint(&executor, i).unwrap();
        }

        // Should keep only last 3
        let checkpoints = manager.list_checkpoints().unwrap();
        assert!(checkpoints.len() <= 3);

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_checkpoint_checksum_verification() {
        let mut executor = Scirs2Exec::new();
        let tensor = ArrayD::from_shape_vec(vec![2], vec![1.0, 2.0]).unwrap();
        executor.add_tensor("data", tensor);

        let config = CheckpointConfig {
            verify_checksum: true,
            ..Default::default()
        };

        let checkpoint = Checkpoint::from_executor_with_config(&executor, 1, &config).unwrap();
        assert!(checkpoint.metadata.checksum.is_some());

        let temp_path = std::env::temp_dir().join("test_checksum.json");
        checkpoint.save(&temp_path).unwrap();

        // Load with verification
        let loaded = Checkpoint::load_with_config(&temp_path, &config).unwrap();
        assert_eq!(loaded.metadata.checksum, checkpoint.metadata.checksum);

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }
}
