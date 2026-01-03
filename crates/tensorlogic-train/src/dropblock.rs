//! DropBlock regularization for convolutional networks.
//!
//! Implements DropBlock, a structured form of dropout that drops contiguous regions
//! rather than independent random units. This is particularly effective for convolutional
//! neural networks where spatial correlation means standard dropout is less effective.
//!
//! # References
//!
//! - Ghiasi, G., Lin, T. Y., & Le, Q. V. (2018).
//!   "DropBlock: A regularization method for convolutional networks". NeurIPS 2018.
//!   <https://arxiv.org/abs/1810.12890>
//!
//! - Used in:
//!   - ResNets (ImageNet)
//!   - AmoebaNet
//!   - EfficientNet variants
//!   - Modern CNNs in general
//!
//! # Key Concepts
//!
//! **DropBlock vs Dropout**:
//! - Dropout: Randomly zeros individual units/pixels
//! - DropBlock: Randomly zeros contiguous blocks/regions
//!
//! **Why DropBlock works better for CNNs**:
//! - Convolutional layers have spatial correlation
//! - Dropping individual pixels allows network to use nearby activations
//! - Dropping blocks forces network to learn more robust features
//!
//! **Block size**: Typically 7x7 or 5x5 for images
//! **Drop probability**: Should be scheduled (linear increase during training)
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_train::DropBlock;
//! use scirs2_core::ndarray::Array2;
//! use scirs2_core::random::{StdRng, SeedableRng};
//!
//! // Create DropBlock with block_size=3, drop_prob=0.1
//! let dropblock = DropBlock::new(3, 0.1).unwrap();
//!
//! let mut rng = StdRng::seed_from_u64(42);
//! let activations = Array2::ones((10, 10));
//!
//! // Training: drop blocks
//! let dropped = dropblock.apply(&activations.view(), true, &mut rng).unwrap();
//!
//! // Inference: no dropping
//! let output = dropblock.apply(&activations.view(), false, &mut rng).unwrap();
//! assert_eq!(output, activations);
//! ```

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::{Rng, StdRng};

/// DropBlock regularization.
///
/// Drops contiguous blocks of activations in convolutional feature maps.
/// This is more effective than standard dropout for CNNs because it forces
/// the network to learn more distributed representations.
#[derive(Debug, Clone)]
pub struct DropBlock {
    /// Size of the block to drop (e.g., 7 for 7x7 blocks)
    pub block_size: usize,

    /// Probability that a block center will be chosen for dropping
    pub drop_prob: f64,

    /// Keep probability (1 - drop_prob)
    keep_prob: f64,
}

impl DropBlock {
    /// Create a new DropBlock regularizer.
    ///
    /// # Arguments
    /// * `block_size` - Size of the block to drop (must be odd and >= 1)
    /// * `drop_prob` - Probability of dropping a block (0.0 to 1.0)
    ///
    /// # Returns
    /// A new DropBlock instance or an error if parameters are invalid.
    ///
    /// # Example
    /// ```rust
    /// use tensorlogic_train::DropBlock;
    ///
    /// let dropblock = DropBlock::new(7, 0.1).unwrap();
    /// ```
    pub fn new(block_size: usize, drop_prob: f64) -> TrainResult<Self> {
        if block_size == 0 {
            return Err(TrainError::InvalidParameter(
                "block_size must be at least 1".to_string(),
            ));
        }

        if block_size.is_multiple_of(2) {
            return Err(TrainError::InvalidParameter(
                "block_size must be odd".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&drop_prob) {
            return Err(TrainError::InvalidParameter(
                "drop_prob must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(Self {
            block_size,
            drop_prob,
            keep_prob: 1.0 - drop_prob,
        })
    }

    /// Set the drop probability (useful for scheduling).
    ///
    /// # Arguments
    /// * `drop_prob` - New drop probability (0.0 to 1.0)
    pub fn set_drop_prob(&mut self, drop_prob: f64) -> TrainResult<()> {
        if !(0.0..=1.0).contains(&drop_prob) {
            return Err(TrainError::InvalidParameter(
                "drop_prob must be between 0.0 and 1.0".to_string(),
            ));
        }

        self.drop_prob = drop_prob;
        self.keep_prob = 1.0 - drop_prob;
        Ok(())
    }

    /// Apply DropBlock to activations.
    ///
    /// # Arguments
    /// * `activations` - Input activation map (height × width)
    /// * `training` - Whether in training mode (drops blocks) or inference mode (no dropping)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Activation map with blocks dropped (if training) or unchanged (if inference)
    ///
    /// # Algorithm
    /// 1. Sample Bernoulli mask for each position (potential block centers)
    /// 2. Expand each selected center to a block of size block_size × block_size
    /// 3. Zero out the selected blocks
    /// 4. Normalize by keep_prob to maintain expected value
    pub fn apply(
        &self,
        activations: &ArrayView2<f64>,
        training: bool,
        rng: &mut StdRng,
    ) -> TrainResult<Array2<f64>> {
        if !training || self.drop_prob == 0.0 {
            return Ok(activations.to_owned());
        }

        let (height, width) = activations.dim();

        if height < self.block_size || width < self.block_size {
            return Err(TrainError::InvalidParameter(format!(
                "Activation map size ({}x{}) is smaller than block_size ({})",
                height, width, self.block_size
            )));
        }

        // Compute gamma (adjusted drop probability accounting for block size)
        // This ensures the expected fraction of dropped units matches drop_prob
        let gamma = self.drop_prob * (height * width) as f64
            / ((height - self.block_size + 1) * (width - self.block_size + 1)) as f64
            / (self.block_size * self.block_size) as f64;

        // Sample block centers using Bernoulli(gamma)
        let mut mask = Array2::ones((height, width));
        let half_block = self.block_size / 2;

        for i in 0..height {
            for j in 0..width {
                if rng.random::<f64>() < gamma {
                    // This position is a block center - zero out the block
                    let i_start = i.saturating_sub(half_block);
                    let i_end = (i + half_block + 1).min(height);
                    let j_start = j.saturating_sub(half_block);
                    let j_end = (j + half_block + 1).min(width);

                    for ii in i_start..i_end {
                        for jj in j_start..j_end {
                            mask[[ii, jj]] = 0.0;
                        }
                    }
                }
            }
        }

        // Apply mask and normalize
        let mut output = activations.to_owned();
        let count_kept = mask.iter().filter(|&&x| x == 1.0).count();
        let normalization_factor = if count_kept > 0 {
            (height * width) as f64 / count_kept as f64
        } else {
            1.0
        };

        for i in 0..height {
            for j in 0..width {
                output[[i, j]] *= mask[[i, j]] * normalization_factor;
            }
        }

        Ok(output)
    }
}

/// Linear DropBlock scheduler.
///
/// Linearly increases drop probability from 0 to target value over training.
/// This is the recommended scheduling strategy from the paper.
#[derive(Debug, Clone)]
pub struct LinearDropBlockScheduler {
    /// Target (maximum) drop probability
    pub drop_prob_target: f64,

    /// Total number of steps to reach target
    pub total_steps: usize,
}

impl LinearDropBlockScheduler {
    /// Create a new linear scheduler.
    ///
    /// # Arguments
    /// * `drop_prob_target` - Final drop probability to reach
    /// * `total_steps` - Number of training steps to linearly increase over
    pub fn new(drop_prob_target: f64, total_steps: usize) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&drop_prob_target) {
            return Err(TrainError::InvalidParameter(
                "drop_prob_target must be between 0.0 and 1.0".to_string(),
            ));
        }

        if total_steps == 0 {
            return Err(TrainError::InvalidParameter(
                "total_steps must be at least 1".to_string(),
            ));
        }

        Ok(Self {
            drop_prob_target,
            total_steps,
        })
    }

    /// Get drop probability for current step.
    ///
    /// # Arguments
    /// * `current_step` - Current training step (0-indexed)
    ///
    /// # Returns
    /// Drop probability linearly interpolated from 0 to target
    pub fn get_drop_prob(&self, current_step: usize) -> f64 {
        if current_step >= self.total_steps {
            return self.drop_prob_target;
        }

        let progress = current_step as f64 / self.total_steps as f64;
        self.drop_prob_target * progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    #[test]
    fn test_dropblock_creation() {
        let db = DropBlock::new(7, 0.1).unwrap();
        assert_eq!(db.block_size, 7);
        assert_eq!(db.drop_prob, 0.1);
        assert_eq!(db.keep_prob, 0.9);
    }

    #[test]
    fn test_dropblock_invalid_params() {
        // Zero block size
        assert!(DropBlock::new(0, 0.1).is_err());

        // Even block size
        assert!(DropBlock::new(4, 0.1).is_err());

        // Invalid drop probability
        assert!(DropBlock::new(7, -0.1).is_err());
        assert!(DropBlock::new(7, 1.5).is_err());
    }

    #[test]
    fn test_dropblock_set_drop_prob() {
        let mut db = DropBlock::new(7, 0.1).unwrap();

        db.set_drop_prob(0.2).unwrap();
        assert_eq!(db.drop_prob, 0.2);
        assert_eq!(db.keep_prob, 0.8);

        // Invalid probability
        assert!(db.set_drop_prob(1.5).is_err());
    }

    #[test]
    fn test_dropblock_inference_mode() {
        let db = DropBlock::new(3, 0.5).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let activations = Array2::ones((10, 10));
        let output = db.apply(&activations.view(), false, &mut rng).unwrap();

        // In inference mode, output should be unchanged
        assert_eq!(output, activations);
    }

    #[test]
    fn test_dropblock_zero_prob() {
        let db = DropBlock::new(3, 0.0).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let activations = Array2::ones((10, 10));
        let output = db.apply(&activations.view(), true, &mut rng).unwrap();

        // With zero probability, no blocks should be dropped
        assert_eq!(output, activations);
    }

    #[test]
    fn test_dropblock_training_mode() {
        let db = DropBlock::new(3, 0.3).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let activations = Array2::ones((20, 20));
        let output = db.apply(&activations.view(), true, &mut rng).unwrap();

        // Shape should be preserved
        assert_eq!(output.shape(), activations.shape());

        // Some values should be zero (dropped blocks)
        let zeros_count = output.iter().filter(|&&x| x == 0.0).count();
        assert!(zeros_count > 0, "Expected some blocks to be dropped");

        // Not all values should be zero
        assert!(zeros_count < 400, "Not all values should be dropped");
    }

    #[test]
    fn test_dropblock_small_activation_map() {
        let db = DropBlock::new(7, 0.1).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        // Activation map smaller than block size
        let activations = Array2::ones((5, 5));
        let result = db.apply(&activations.view(), true, &mut rng);

        assert!(result.is_err());
    }

    #[test]
    fn test_linear_scheduler_creation() {
        let scheduler = LinearDropBlockScheduler::new(0.1, 1000).unwrap();
        assert_eq!(scheduler.drop_prob_target, 0.1);
        assert_eq!(scheduler.total_steps, 1000);
    }

    #[test]
    fn test_linear_scheduler_invalid_params() {
        // Invalid target probability
        assert!(LinearDropBlockScheduler::new(-0.1, 1000).is_err());
        assert!(LinearDropBlockScheduler::new(1.5, 1000).is_err());

        // Zero steps
        assert!(LinearDropBlockScheduler::new(0.1, 0).is_err());
    }

    #[test]
    fn test_linear_scheduler_interpolation() {
        let scheduler = LinearDropBlockScheduler::new(0.1, 100).unwrap();

        // At step 0
        assert_eq!(scheduler.get_drop_prob(0), 0.0);

        // At step 50 (halfway)
        let mid_prob = scheduler.get_drop_prob(50);
        assert!((mid_prob - 0.05).abs() < 1e-10);

        // At step 100 (end)
        assert_eq!(scheduler.get_drop_prob(100), 0.1);

        // Beyond total steps
        assert_eq!(scheduler.get_drop_prob(150), 0.1);
    }

    #[test]
    fn test_dropblock_with_scheduler() {
        let mut db = DropBlock::new(3, 0.0).unwrap();
        let scheduler = LinearDropBlockScheduler::new(0.2, 100).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let activations = Array2::ones((20, 20));

        // Simulate training with scheduler
        for step in [0, 50, 100] {
            let drop_prob = scheduler.get_drop_prob(step);
            db.set_drop_prob(drop_prob).unwrap();

            let output = db.apply(&activations.view(), true, &mut rng).unwrap();
            assert_eq!(output.shape(), activations.shape());
        }
    }

    #[test]
    fn test_dropblock_normalization() {
        let db = DropBlock::new(3, 0.1).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let activations = Array2::from_elem((20, 20), 1.0);
        let output = db.apply(&activations.view(), true, &mut rng).unwrap();

        // The sum of output should be close to sum of input (due to normalization)
        // This ensures expected value is preserved
        let input_sum = activations.sum();
        let output_sum = output.sum();

        // Allow some variance due to randomness, but should be reasonably close
        let relative_diff = (output_sum - input_sum).abs() / input_sum;
        assert!(
            relative_diff < 0.5,
            "Normalization should preserve approximate expected value"
        );
    }
}
