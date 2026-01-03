//! Stochastic Depth (DropPath) for deep networks.
//!
//! Implements Drop-Path regularization, a technique that randomly drops entire
//! residual paths during training. This is particularly effective for very deep
//! networks and is widely used in Vision Transformers and ResNets.
//!
//! # References
//!
//! - Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016).
//!   "Deep Networks with Stochastic Depth". ECCV 2016.
//!   <https://arxiv.org/abs/1603.09382>
//!
//! - Widely used in:
//!   - Vision Transformers (ViT, DeiT, Swin)
//!   - EfficientNet
//!   - ResNets and variants
//!
//! # Key Concepts
//!
//! **DropPath vs Dropout**:
//! - Dropout: Randomly zeros individual neurons
//! - DropPath: Randomly zeros entire paths/blocks
//!
//! **Usage**: Applied to residual connections:
//! ```text
//! output = x + DropPath(F(x))
//! ```
//!
//! **Linear Scheduling**: Drop probability increases with depth:
//! ```text
//! drop_prob(layer_i) = drop_prob_min + (drop_prob_max - drop_prob_min) * i / (L-1)
//! ```

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::{Rng, StdRng};

/// DropPath (Stochastic Depth) regularization.
///
/// Randomly drops entire paths in residual networks during training.
/// At test time, paths are always kept but scaled by (1 - drop_prob).
///
/// # Example
///
/// ```rust
/// use tensorlogic_train::DropPath;
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::random::{StdRng, SeedableRng};
///
/// let drop_path = DropPath::new(0.1).unwrap(); // 10% drop probability
/// let mut rng = StdRng::seed_from_u64(42);
///
/// let residual = Array2::ones((4, 8));
///
/// // Training mode: randomly drop
/// let output = drop_path.apply(&residual.view(), true, &mut rng).unwrap();
///
/// // Inference mode: scale by keep probability
/// let output_test = drop_path.apply(&residual.view(), false, &mut rng).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DropPath {
    /// Probability of dropping the path (0.0 to 1.0).
    pub drop_prob: f64,
    /// Keep probability (1.0 - drop_prob).
    keep_prob: f64,
}

impl DropPath {
    /// Create a new DropPath regularizer.
    ///
    /// # Arguments
    ///
    /// * `drop_prob` - Probability of dropping the path (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// A new DropPath instance or error if drop_prob is invalid.
    pub fn new(drop_prob: f64) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&drop_prob) {
            return Err(TrainError::InvalidParameter(
                "drop_prob must be in [0, 1]".to_string(),
            ));
        }

        Ok(Self {
            drop_prob,
            keep_prob: 1.0 - drop_prob,
        })
    }

    /// Apply DropPath to a residual path.
    ///
    /// # Arguments
    ///
    /// * `path` - The residual path to potentially drop
    /// * `training` - Whether in training mode (drop) or inference mode (scale)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Transformed path (either dropped or scaled)
    pub fn apply(
        &self,
        path: &ArrayView2<f64>,
        training: bool,
        rng: &mut StdRng,
    ) -> TrainResult<Array2<f64>> {
        if !training || self.drop_prob == 0.0 {
            // Inference mode or no dropout: just return the path
            return Ok(path.to_owned());
        }

        if self.drop_prob == 1.0 {
            // Always drop: return zeros
            return Ok(Array2::zeros(path.raw_dim()));
        }

        // Training mode: randomly drop the entire path
        let should_drop = rng.random::<f64>() < self.drop_prob;

        if should_drop {
            // Drop the path (return zeros)
            Ok(Array2::zeros(path.raw_dim()))
        } else {
            // Keep the path but scale by 1/keep_prob to maintain expected value
            // This is the "inverted dropout" technique
            Ok(path.mapv(|x| x / self.keep_prob))
        }
    }

    /// Apply DropPath to a batch of paths.
    ///
    /// Each sample in the batch is independently dropped with probability drop_prob.
    ///
    /// # Arguments
    ///
    /// * `paths` - Batch of residual paths (batch_size × features)
    /// * `training` - Whether in training mode
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Batch with randomly dropped paths
    pub fn apply_batch(
        &self,
        paths: &ArrayView2<f64>,
        training: bool,
        rng: &mut StdRng,
    ) -> TrainResult<Array2<f64>> {
        if !training || self.drop_prob == 0.0 {
            return Ok(paths.to_owned());
        }

        let (batch_size, _) = paths.dim();
        let mut output = paths.to_owned();

        // Independently drop each sample in batch
        for i in 0..batch_size {
            let should_drop = rng.random::<f64>() < self.drop_prob;
            if should_drop {
                // Zero out this sample
                for j in 0..output.ncols() {
                    output[[i, j]] = 0.0;
                }
            } else {
                // Scale by 1/keep_prob
                for j in 0..output.ncols() {
                    output[[i, j]] /= self.keep_prob;
                }
            }
        }

        Ok(output)
    }

    /// Get the keep probability.
    pub fn keep_probability(&self) -> f64 {
        self.keep_prob
    }

    /// Set new drop probability.
    pub fn set_drop_prob(&mut self, drop_prob: f64) -> TrainResult<()> {
        if !(0.0..=1.0).contains(&drop_prob) {
            return Err(TrainError::InvalidParameter(
                "drop_prob must be in [0, 1]".to_string(),
            ));
        }

        self.drop_prob = drop_prob;
        self.keep_prob = 1.0 - drop_prob;
        Ok(())
    }
}

/// Linear stochastic depth scheduler.
///
/// Linearly increases drop probability from min to max across network depth.
/// This is the standard approach used in most deep networks:
/// - Shallow layers: low drop probability (more stable)
/// - Deep layers: high drop probability (more regularization)
///
/// # Example
///
/// ```no_run
/// use tensorlogic_train::LinearStochasticDepth;
///
/// // 10 layers, drop_prob from 0.0 to 0.3
/// let scheduler = LinearStochasticDepth::new(10, 0.0, 0.3).unwrap();
///
/// // Get drop probability for layer 5
/// let drop_prob_5 = scheduler.get_drop_prob(5);
/// assert!((drop_prob_5 - 0.15).abs() < 1e-6); // Halfway point
/// ```
#[derive(Debug, Clone)]
pub struct LinearStochasticDepth {
    /// Total number of layers/blocks.
    pub num_layers: usize,
    /// Minimum drop probability (first layer).
    pub drop_prob_min: f64,
    /// Maximum drop probability (last layer).
    pub drop_prob_max: f64,
}

impl LinearStochasticDepth {
    /// Create a new linear stochastic depth scheduler.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Total number of layers in the network
    /// * `drop_prob_min` - Drop probability for first layer (usually 0.0)
    /// * `drop_prob_max` - Drop probability for last layer (e.g., 0.1-0.5)
    ///
    /// # Returns
    ///
    /// A new scheduler or error if parameters are invalid.
    pub fn new(num_layers: usize, drop_prob_min: f64, drop_prob_max: f64) -> TrainResult<Self> {
        if num_layers == 0 {
            return Err(TrainError::InvalidParameter(
                "num_layers must be > 0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&drop_prob_min) || !(0.0..=1.0).contains(&drop_prob_max) {
            return Err(TrainError::InvalidParameter(
                "drop probabilities must be in [0, 1]".to_string(),
            ));
        }

        if drop_prob_min > drop_prob_max {
            return Err(TrainError::InvalidParameter(
                "drop_prob_min must be <= drop_prob_max".to_string(),
            ));
        }

        Ok(Self {
            num_layers,
            drop_prob_min,
            drop_prob_max,
        })
    }

    /// Get drop probability for a specific layer.
    ///
    /// Uses linear interpolation:
    /// ```text
    /// drop_prob(i) = drop_prob_min + (drop_prob_max - drop_prob_min) * i / (L-1)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index (0 to num_layers-1)
    ///
    /// # Returns
    ///
    /// Drop probability for this layer.
    pub fn get_drop_prob(&self, layer_idx: usize) -> f64 {
        if layer_idx >= self.num_layers {
            return self.drop_prob_max;
        }

        if self.num_layers == 1 {
            return self.drop_prob_min;
        }

        // Linear interpolation
        let ratio = layer_idx as f64 / (self.num_layers - 1) as f64;
        self.drop_prob_min + (self.drop_prob_max - self.drop_prob_min) * ratio
    }

    /// Create DropPath instances for all layers.
    ///
    /// # Returns
    ///
    /// Vector of DropPath instances with linearly increasing drop probabilities.
    pub fn create_drop_paths(&self) -> TrainResult<Vec<DropPath>> {
        let mut drop_paths = Vec::with_capacity(self.num_layers);

        for i in 0..self.num_layers {
            let drop_prob = self.get_drop_prob(i);
            drop_paths.push(DropPath::new(drop_prob)?);
        }

        Ok(drop_paths)
    }
}

/// Exponential stochastic depth scheduler.
///
/// Exponentially increases drop probability across network depth.
/// Provides more aggressive regularization in deeper layers.
#[derive(Debug, Clone)]
pub struct ExponentialStochasticDepth {
    /// Total number of layers/blocks.
    pub num_layers: usize,
    /// Drop probability for first layer.
    pub drop_prob_min: f64,
    /// Drop probability for last layer.
    pub drop_prob_max: f64,
}

impl ExponentialStochasticDepth {
    /// Create a new exponential stochastic depth scheduler.
    pub fn new(num_layers: usize, drop_prob_min: f64, drop_prob_max: f64) -> TrainResult<Self> {
        if num_layers == 0 {
            return Err(TrainError::InvalidParameter(
                "num_layers must be > 0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&drop_prob_min) || !(0.0..=1.0).contains(&drop_prob_max) {
            return Err(TrainError::InvalidParameter(
                "drop probabilities must be in [0, 1]".to_string(),
            ));
        }

        if drop_prob_min > drop_prob_max {
            return Err(TrainError::InvalidParameter(
                "drop_prob_min must be <= drop_prob_max".to_string(),
            ));
        }

        Ok(Self {
            num_layers,
            drop_prob_min,
            drop_prob_max,
        })
    }

    /// Get drop probability for a specific layer using exponential interpolation.
    pub fn get_drop_prob(&self, layer_idx: usize) -> f64 {
        if layer_idx >= self.num_layers {
            return self.drop_prob_max;
        }

        if self.num_layers == 1 {
            return self.drop_prob_min;
        }

        // Exponential interpolation: use power of 2 for smooth curve
        let ratio = layer_idx as f64 / (self.num_layers - 1) as f64;
        let exp_ratio = ratio * ratio; // Quadratic for exponential effect

        self.drop_prob_min + (self.drop_prob_max - self.drop_prob_min) * exp_ratio
    }

    /// Create DropPath instances for all layers.
    pub fn create_drop_paths(&self) -> TrainResult<Vec<DropPath>> {
        let mut drop_paths = Vec::with_capacity(self.num_layers);

        for i in 0..self.num_layers {
            let drop_prob = self.get_drop_prob(i);
            drop_paths.push(DropPath::new(drop_prob)?);
        }

        Ok(drop_paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::SeedableRng;

    fn create_test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_drop_path_creation() {
        let dp = DropPath::new(0.2).unwrap();
        assert_eq!(dp.drop_prob, 0.2);
        assert!((dp.keep_prob - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_drop_path_invalid_prob() {
        assert!(DropPath::new(-0.1).is_err());
        assert!(DropPath::new(1.5).is_err());
    }

    #[test]
    fn test_drop_path_zero_prob() {
        let dp = DropPath::new(0.0).unwrap();
        let mut rng = create_test_rng();

        let path = array![[1.0, 2.0], [3.0, 4.0]];

        // With 0% drop prob, path should be unchanged
        let output = dp.apply(&path.view(), true, &mut rng).unwrap();
        assert_eq!(output, path);
    }

    #[test]
    fn test_drop_path_full_prob() {
        let dp = DropPath::new(1.0).unwrap();
        let mut rng = create_test_rng();

        let path = array![[1.0, 2.0], [3.0, 4.0]];

        // With 100% drop prob, should return zeros
        let output = dp.apply(&path.view(), true, &mut rng).unwrap();
        assert_eq!(output, Array2::<f64>::zeros((2, 2)));
    }

    #[test]
    fn test_drop_path_inference_mode() {
        let dp = DropPath::new(0.5).unwrap();
        let mut rng = create_test_rng();

        let path = array![[1.0, 2.0], [3.0, 4.0]];

        // In inference mode (training=false), path should be unchanged
        let output = dp.apply(&path.view(), false, &mut rng).unwrap();
        assert_eq!(output, path);
    }

    #[test]
    fn test_drop_path_training_mode() {
        let dp = DropPath::new(0.5).unwrap();
        let mut rng = create_test_rng();

        let path = array![[1.0, 2.0]];

        // Run multiple times to check stochastic behavior
        let mut dropped_count = 0;
        let mut kept_count = 0;

        for _ in 0..100 {
            let output = dp.apply(&path.view(), true, &mut rng).unwrap();

            if output[[0, 0]] == 0.0 {
                dropped_count += 1;
            } else {
                kept_count += 1;
                // When kept, should be scaled by 1/keep_prob = 2.0
                assert!((output[[0, 0]] - 2.0).abs() < 1e-10);
            }
        }

        // With 50% drop prob, should drop roughly half the time
        assert!(dropped_count > 30 && dropped_count < 70);
        assert!(kept_count > 30 && kept_count < 70);
    }

    #[test]
    fn test_drop_path_batch() {
        let dp = DropPath::new(0.5).unwrap();
        let mut rng = create_test_rng();

        let paths = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let output = dp.apply_batch(&paths.view(), true, &mut rng).unwrap();

        // Shape should be preserved
        assert_eq!(output.shape(), paths.shape());

        // Some rows should be dropped (all zeros), others kept and scaled
        let mut dropped_rows = 0;
        for i in 0..output.nrows() {
            if output[[i, 0]] == 0.0 && output[[i, 1]] == 0.0 {
                dropped_rows += 1;
            }
        }

        // With 50% drop prob and 4 rows, expect ~2 dropped
        assert!(dropped_rows > 0);
    }

    #[test]
    fn test_drop_path_set_prob() {
        let mut dp = DropPath::new(0.2).unwrap();
        assert_eq!(dp.drop_prob, 0.2);

        dp.set_drop_prob(0.5).unwrap();
        assert_eq!(dp.drop_prob, 0.5);
        assert!((dp.keep_prob - 0.5).abs() < 1e-10);

        // Invalid probability
        assert!(dp.set_drop_prob(1.5).is_err());
    }

    #[test]
    fn test_linear_stochastic_depth_creation() {
        let scheduler = LinearStochasticDepth::new(10, 0.0, 0.5).unwrap();
        assert_eq!(scheduler.num_layers, 10);
        assert_eq!(scheduler.drop_prob_min, 0.0);
        assert_eq!(scheduler.drop_prob_max, 0.5);
    }

    #[test]
    fn test_linear_stochastic_depth_invalid() {
        assert!(LinearStochasticDepth::new(0, 0.0, 0.5).is_err());
        assert!(LinearStochasticDepth::new(10, -0.1, 0.5).is_err());
        assert!(LinearStochasticDepth::new(10, 0.0, 1.5).is_err());
        assert!(LinearStochasticDepth::new(10, 0.6, 0.3).is_err());
    }

    #[test]
    fn test_linear_stochastic_depth_interpolation() {
        let scheduler = LinearStochasticDepth::new(10, 0.0, 0.9).unwrap();

        // First layer: min drop prob
        assert!((scheduler.get_drop_prob(0) - 0.0).abs() < 1e-10);

        // Middle layer: halfway
        assert!((scheduler.get_drop_prob(5) - 0.5).abs() < 1e-6);

        // Last layer: max drop prob
        assert!((scheduler.get_drop_prob(9) - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_linear_stochastic_depth_create_paths() {
        let scheduler = LinearStochasticDepth::new(5, 0.0, 0.4).unwrap();
        let paths = scheduler.create_drop_paths().unwrap();

        assert_eq!(paths.len(), 5);

        // Check drop probabilities increase linearly
        assert!((paths[0].drop_prob - 0.0).abs() < 1e-10);
        assert!((paths[2].drop_prob - 0.2).abs() < 1e-10);
        assert!((paths[4].drop_prob - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_stochastic_depth() {
        let scheduler = ExponentialStochasticDepth::new(10, 0.0, 0.8).unwrap();

        // First layer: min drop prob
        assert!((scheduler.get_drop_prob(0) - 0.0).abs() < 1e-10);

        // Last layer: max drop prob
        assert!((scheduler.get_drop_prob(9) - 0.8).abs() < 1e-10);

        // Middle layers should have exponentially increasing drop prob
        let mid_prob = scheduler.get_drop_prob(5);
        let linear_mid = 0.4; // What it would be with linear

        // Exponential should be less than linear in first half
        assert!(mid_prob < linear_mid + 0.1);
    }

    #[test]
    fn test_exponential_create_paths() {
        let scheduler = ExponentialStochasticDepth::new(5, 0.0, 0.4).unwrap();
        let paths = scheduler.create_drop_paths().unwrap();

        assert_eq!(paths.len(), 5);

        // Verify increasing drop probabilities
        for i in 0..paths.len() - 1 {
            assert!(paths[i].drop_prob <= paths[i + 1].drop_prob);
        }
    }
}
