//! Gradient checkpointing for memory-efficient training.
//!
//! Gradient checkpointing trades compute for memory by recomputing activations
//! during the backward pass instead of storing them. This allows training much
//! larger models or using larger batch sizes.
//!
//! ## How It Works
//!
//! Without checkpointing:
//! ```text
//! Forward: x → layer1 → layer2 → layer3 → loss
//!          ↓     ↓       ↓       ↓
//!        store  store   store   store (memory)
//! ```
//!
//! With checkpointing:
//! ```text
//! Forward: x → layer1 → [checkpoint] → layer2 → [checkpoint] → layer3 → loss
//!          ↓                             ↓                       ↓
//!        store                         store                   store
//!
//! Backward: Recompute layer1 and layer2 activations as needed
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use tensorlogic_trustformers::{CheckpointConfig, CheckpointStrategy};
//!
//! // Checkpoint every 2 layers
//! let config = CheckpointConfig::uniform(2);
//!
//! // Checkpoint specific layers
//! let config = CheckpointConfig::selective(vec![0, 3, 6, 9]);
//!
//! // Dynamic checkpointing (more frequent in deeper layers)
//! let config = CheckpointConfig::dynamic(12, 0.3); // 30% memory target
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrustformerError};

/// Gradient checkpointing configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Checkpointing strategy
    pub strategy: CheckpointStrategy,
    /// Whether to checkpoint attention
    pub checkpoint_attention: bool,
    /// Whether to checkpoint feed-forward
    pub checkpoint_ffn: bool,
    /// Minimum layers between checkpoints
    pub min_checkpoint_interval: usize,
}

/// Strategy for placing gradient checkpoints
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CheckpointStrategy {
    /// No checkpointing
    None,
    /// Checkpoint every N layers uniformly
    Uniform { interval: usize },
    /// Checkpoint specific layer indices
    Selective { layers: Vec<usize> },
    /// Dynamic checkpointing based on memory budget
    Dynamic {
        /// Total number of layers
        num_layers: usize,
        /// Target memory fraction (0.0 - 1.0)
        memory_fraction: f64,
    },
}

impl CheckpointConfig {
    /// Create a uniform checkpointing configuration
    ///
    /// # Arguments
    /// * `interval` - Checkpoint every N layers (e.g., 2 means checkpoint layers 0, 2, 4, ...)
    pub fn uniform(interval: usize) -> Self {
        Self {
            strategy: CheckpointStrategy::Uniform { interval },
            checkpoint_attention: true,
            checkpoint_ffn: true,
            min_checkpoint_interval: 1,
        }
    }

    /// Create a selective checkpointing configuration
    ///
    /// # Arguments
    /// * `layers` - Specific layer indices to checkpoint
    pub fn selective(layers: Vec<usize>) -> Self {
        Self {
            strategy: CheckpointStrategy::Selective { layers },
            checkpoint_attention: true,
            checkpoint_ffn: true,
            min_checkpoint_interval: 1,
        }
    }

    /// Create a dynamic checkpointing configuration
    ///
    /// # Arguments
    /// * `num_layers` - Total number of layers in the model
    /// * `memory_fraction` - Target memory usage as fraction of full storage (0.0 - 1.0)
    pub fn dynamic(num_layers: usize, memory_fraction: f64) -> Result<Self> {
        if num_layers == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "num_layers must be > 0".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&memory_fraction) {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: format!(
                    "memory_fraction must be in [0.0, 1.0], got {}",
                    memory_fraction
                ),
            });
        }

        Ok(Self {
            strategy: CheckpointStrategy::Dynamic {
                num_layers,
                memory_fraction,
            },
            checkpoint_attention: true,
            checkpoint_ffn: true,
            min_checkpoint_interval: 1,
        })
    }

    /// Disable checkpointing
    pub fn none() -> Self {
        Self {
            strategy: CheckpointStrategy::None,
            checkpoint_attention: false,
            checkpoint_ffn: false,
            min_checkpoint_interval: 1,
        }
    }

    /// Set whether to checkpoint attention sublayers
    pub fn with_checkpoint_attention(mut self, checkpoint: bool) -> Self {
        self.checkpoint_attention = checkpoint;
        self
    }

    /// Set whether to checkpoint feed-forward sublayers
    pub fn with_checkpoint_ffn(mut self, checkpoint: bool) -> Self {
        self.checkpoint_ffn = checkpoint;
        self
    }

    /// Set minimum interval between checkpoints
    pub fn with_min_interval(mut self, interval: usize) -> Self {
        self.min_checkpoint_interval = interval;
        self
    }

    /// Check if a specific layer should be checkpointed
    pub fn should_checkpoint(&self, layer_idx: usize) -> bool {
        match &self.strategy {
            CheckpointStrategy::None => false,
            CheckpointStrategy::Uniform { interval } => {
                *interval > 0 && layer_idx.is_multiple_of(*interval)
            }
            CheckpointStrategy::Selective { layers } => layers.contains(&layer_idx),
            CheckpointStrategy::Dynamic {
                num_layers,
                memory_fraction,
            } => {
                // Calculate optimal checkpoint interval for target memory fraction
                // Memory without checkpointing: O(n * d^2) for n layers
                // Memory with checkpointing every k layers: O(k * d^2)
                // Target: k * d^2 = memory_fraction * n * d^2
                // Therefore: k = memory_fraction * n

                if *num_layers == 0 {
                    return false;
                }

                let target_interval = (*memory_fraction * *num_layers as f64).max(1.0) as usize;
                let interval = target_interval.max(self.min_checkpoint_interval);
                interval > 0 && layer_idx.is_multiple_of(interval)
            }
        }
    }

    /// Calculate expected memory savings
    ///
    /// Returns the fraction of activation memory saved (0.0 - 1.0)
    pub fn memory_savings(&self, num_layers: usize) -> f64 {
        if num_layers == 0 {
            return 0.0;
        }

        match &self.strategy {
            CheckpointStrategy::None => 0.0,
            CheckpointStrategy::Uniform { interval } => {
                let interval_val = *interval;
                if interval_val == 0 || interval_val >= num_layers {
                    return 0.0;
                }
                // We store activations at checkpoint boundaries only
                let num_checkpoints = num_layers.div_ceil(interval_val);
                1.0 - (num_checkpoints as f64 / num_layers as f64)
            }
            CheckpointStrategy::Selective { layers } => {
                if layers.is_empty() {
                    return 0.0;
                }
                1.0 - (layers.len() as f64 / num_layers as f64)
            }
            CheckpointStrategy::Dynamic {
                memory_fraction, ..
            } => {
                // Dynamic strategy aims to use memory_fraction of full storage
                1.0 - memory_fraction
            }
        }
    }

    /// Calculate expected compute overhead
    ///
    /// Returns the multiplicative factor for compute (1.0 = no overhead)
    pub fn compute_overhead(&self, num_layers: usize) -> f64 {
        if num_layers == 0 {
            return 1.0;
        }

        match &self.strategy {
            CheckpointStrategy::None => 1.0,
            CheckpointStrategy::Uniform { interval } => {
                if *interval == 0 || *interval >= num_layers {
                    return 1.0;
                }
                // We recompute layers between checkpoints during backward pass
                // Each layer is computed once in forward, and segments are recomputed in backward
                // Overhead ≈ 1 + (average segment length / 2)
                1.0 + (*interval as f64 / 2.0) / num_layers as f64
            }
            CheckpointStrategy::Selective { layers } => {
                if layers.is_empty() {
                    return 1.0;
                }
                // Average interval between checkpoints
                let avg_interval = num_layers as f64 / layers.len() as f64;
                1.0 + (avg_interval / 2.0) / num_layers as f64
            }
            CheckpointStrategy::Dynamic {
                memory_fraction, ..
            } => {
                // Compute overhead scales with memory savings
                1.0 + (1.0 - memory_fraction) * 0.3 // ~30% overhead for full checkpointing
            }
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        match &self.strategy {
            CheckpointStrategy::None => Ok(()),
            CheckpointStrategy::Uniform { interval } => {
                if *interval == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "checkpoint interval must be > 0".to_string(),
                    });
                }
                Ok(())
            }
            CheckpointStrategy::Selective { layers } => {
                // Check for duplicates
                let mut sorted = layers.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if sorted.len() != layers.len() {
                    return Err(TrustformerError::InvalidDimension {
                        expected: sorted.len(),
                        got: layers.len(),
                        context: "duplicate layer indices in selective checkpointing".to_string(),
                    });
                }
                Ok(())
            }
            CheckpointStrategy::Dynamic {
                num_layers,
                memory_fraction,
            } => {
                if *num_layers == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "num_layers must be > 0".to_string(),
                    });
                }
                if !(0.0..=1.0).contains(memory_fraction) {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: format!(
                            "memory_fraction must be in [0.0, 1.0], got {}",
                            memory_fraction
                        ),
                    });
                }
                Ok(())
            }
        }
    }

    /// Get human-readable summary
    pub fn summary(&self) -> String {
        match &self.strategy {
            CheckpointStrategy::None => "No checkpointing".to_string(),
            CheckpointStrategy::Uniform { interval } => {
                format!("Uniform checkpointing every {} layers", interval)
            }
            CheckpointStrategy::Selective { layers } => {
                format!("Selective checkpointing at {} layers", layers.len())
            }
            CheckpointStrategy::Dynamic {
                num_layers,
                memory_fraction,
            } => {
                format!(
                    "Dynamic checkpointing ({} layers, {:.1}% memory target)",
                    num_layers,
                    memory_fraction * 100.0
                )
            }
        }
    }
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self::none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_checkpointing() {
        let config = CheckpointConfig::uniform(2);
        assert!(config.should_checkpoint(0));
        assert!(!config.should_checkpoint(1));
        assert!(config.should_checkpoint(2));
        assert!(!config.should_checkpoint(3));
        assert!(config.should_checkpoint(4));
    }

    #[test]
    fn test_selective_checkpointing() {
        let config = CheckpointConfig::selective(vec![0, 3, 7]);
        assert!(config.should_checkpoint(0));
        assert!(!config.should_checkpoint(1));
        assert!(!config.should_checkpoint(2));
        assert!(config.should_checkpoint(3));
        assert!(!config.should_checkpoint(6));
        assert!(config.should_checkpoint(7));
    }

    #[test]
    fn test_dynamic_checkpointing() {
        let config = CheckpointConfig::dynamic(12, 0.3).unwrap();
        // With 12 layers and 30% memory target, checkpoint every ~4 layers
        assert!(config.validate().is_ok());

        // Check some layers
        let checkpointed_count = (0..12).filter(|&i| config.should_checkpoint(i)).count();
        assert!(checkpointed_count > 0);
        assert!(checkpointed_count < 12);
    }

    #[test]
    fn test_no_checkpointing() {
        let config = CheckpointConfig::none();
        assert!(!config.should_checkpoint(0));
        assert!(!config.should_checkpoint(5));
        assert!(!config.should_checkpoint(10));
    }

    #[test]
    fn test_memory_savings_uniform() {
        let config = CheckpointConfig::uniform(3);
        let savings = config.memory_savings(12);
        // With interval 3, we checkpoint layers 0, 3, 6, 9 (4 checkpoints out of 12)
        assert!((savings - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_savings_selective() {
        let config = CheckpointConfig::selective(vec![0, 6]);
        let savings = config.memory_savings(12);
        // 2 checkpoints out of 12 layers
        assert!((savings - 10.0 / 12.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_overhead() {
        let config = CheckpointConfig::uniform(2);
        let overhead = config.compute_overhead(12);
        assert!(overhead >= 1.0);
        assert!(overhead < 2.0); // Should be modest overhead
    }

    #[test]
    fn test_invalid_dynamic_memory_fraction() {
        let result = CheckpointConfig::dynamic(12, 1.5);
        assert!(result.is_err());

        let result = CheckpointConfig::dynamic(12, -0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let config = CheckpointConfig::uniform(2)
            .with_checkpoint_attention(false)
            .with_checkpoint_ffn(true)
            .with_min_interval(2);

        assert!(!config.checkpoint_attention);
        assert!(config.checkpoint_ffn);
        assert_eq!(config.min_checkpoint_interval, 2);
    }

    #[test]
    fn test_validate_uniform() {
        let config = CheckpointConfig::uniform(2);
        assert!(config.validate().is_ok());

        let config = CheckpointConfig::uniform(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_selective_duplicates() {
        let config = CheckpointConfig::selective(vec![0, 3, 3, 7]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_summary() {
        let config = CheckpointConfig::uniform(2);
        assert!(config.summary().contains("every 2 layers"));

        let config = CheckpointConfig::selective(vec![0, 3, 7]);
        assert!(config.summary().contains("3 layers"));

        let config = CheckpointConfig::dynamic(12, 0.3).unwrap();
        assert!(config.summary().contains("30.0%"));
    }

    #[test]
    fn test_default() {
        let config = CheckpointConfig::default();
        assert_eq!(config.strategy, CheckpointStrategy::None);
        assert!(!config.should_checkpoint(0));
    }

    #[test]
    fn test_zero_interval_uniform() {
        let config = CheckpointConfig::uniform(0);
        assert!(!config.should_checkpoint(0));
        assert!(!config.should_checkpoint(1));
    }

    #[test]
    fn test_dynamic_zero_layers() {
        let result = CheckpointConfig::dynamic(0, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_savings_edge_cases() {
        // No layers
        let config = CheckpointConfig::uniform(2);
        assert_eq!(config.memory_savings(0), 0.0);

        // Interval >= num_layers
        let config = CheckpointConfig::uniform(20);
        assert_eq!(config.memory_savings(10), 0.0);

        // Empty selective
        let config = CheckpointConfig::selective(vec![]);
        assert_eq!(config.memory_savings(10), 0.0);
    }
}
