//! Data augmentation techniques for training.
//!
//! This module provides various data augmentation strategies to improve model generalization:
//! - Noise augmentation (Gaussian)
//! - Scale augmentation (random scaling)
//! - Rotation augmentation (placeholder for future)
//! - Mixup augmentation (interpolation between samples)

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Array2, ArrayView2};
use scirs2_core::random::{Rng, StdRng};

/// Trait for data augmentation strategies.
pub trait DataAugmenter {
    /// Augment the given data.
    ///
    /// # Arguments
    /// * `data` - Input data to augment
    /// * `rng` - Random number generator for stochastic augmentation
    ///
    /// # Returns
    /// Augmented data with the same shape as input
    fn augment(&self, data: &ArrayView2<f64>, rng: &mut StdRng) -> TrainResult<Array2<f64>>;
}

/// No augmentation (identity transformation).
///
/// Useful for testing or as a placeholder.
#[derive(Debug, Clone, Default)]
pub struct NoAugmentation;

impl DataAugmenter for NoAugmentation {
    fn augment(&self, data: &ArrayView2<f64>, _rng: &mut StdRng) -> TrainResult<Array2<f64>> {
        Ok(data.to_owned())
    }
}

/// Gaussian noise augmentation.
///
/// Adds random Gaussian noise to the input data: x' = x + N(0, σ²)
#[derive(Debug, Clone)]
pub struct NoiseAugmenter {
    /// Standard deviation of the Gaussian noise.
    pub std_dev: f64,
}

impl NoiseAugmenter {
    /// Create a new noise augmenter.
    ///
    /// # Arguments
    /// * `std_dev` - Standard deviation of the noise
    pub fn new(std_dev: f64) -> TrainResult<Self> {
        if std_dev < 0.0 {
            return Err(TrainError::InvalidParameter(
                "std_dev must be non-negative".to_string(),
            ));
        }
        Ok(Self { std_dev })
    }
}

impl Default for NoiseAugmenter {
    fn default() -> Self {
        Self { std_dev: 0.01 }
    }
}

impl DataAugmenter for NoiseAugmenter {
    fn augment(&self, data: &ArrayView2<f64>, rng: &mut StdRng) -> TrainResult<Array2<f64>> {
        let mut augmented = data.to_owned();

        // Add Gaussian noise using Box-Muller transform
        for value in augmented.iter_mut() {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();

            // Box-Muller transform
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let noise = z0 * self.std_dev;

            *value += noise;
        }

        Ok(augmented)
    }
}

/// Scale augmentation.
///
/// Randomly scales the input data by a factor in [1 - scale_range, 1 + scale_range].
#[derive(Debug, Clone)]
pub struct ScaleAugmenter {
    /// Range of scaling factor (e.g., 0.1 means scale in [0.9, 1.1]).
    pub scale_range: f64,
}

impl ScaleAugmenter {
    /// Create a new scale augmenter.
    ///
    /// # Arguments
    /// * `scale_range` - Range of scaling factor
    pub fn new(scale_range: f64) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&scale_range) {
            return Err(TrainError::InvalidParameter(
                "scale_range must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self { scale_range })
    }
}

impl Default for ScaleAugmenter {
    fn default() -> Self {
        Self { scale_range: 0.1 }
    }
}

impl DataAugmenter for ScaleAugmenter {
    fn augment(&self, data: &ArrayView2<f64>, rng: &mut StdRng) -> TrainResult<Array2<f64>> {
        // Generate random scale factor
        let scale = 1.0 + (rng.random::<f64>() * 2.0 - 1.0) * self.scale_range;

        let augmented = data.mapv(|x| x * scale);
        Ok(augmented)
    }
}

/// Rotation augmentation (placeholder for future implementation).
///
/// For 2D images, this would apply random rotations.
/// Currently implements a simplified version for tabular data.
#[derive(Debug, Clone)]
pub struct RotationAugmenter {
    /// Maximum rotation angle in radians.
    pub max_angle: f64,
}

impl RotationAugmenter {
    /// Create a new rotation augmenter.
    ///
    /// # Arguments
    /// * `max_angle` - Maximum rotation angle in radians
    pub fn new(max_angle: f64) -> Self {
        Self { max_angle }
    }
}

impl Default for RotationAugmenter {
    fn default() -> Self {
        Self {
            max_angle: std::f64::consts::PI / 18.0, // 10 degrees
        }
    }
}

impl DataAugmenter for RotationAugmenter {
    fn augment(&self, data: &ArrayView2<f64>, rng: &mut StdRng) -> TrainResult<Array2<f64>> {
        // For now, this is a placeholder that returns a simple transformation
        // Future: implement proper 2D rotation for image data
        let angle = (rng.random::<f64>() * 2.0 - 1.0) * self.max_angle;

        // Apply a simple rotation-inspired transformation
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let augmented = data.mapv(|x| x * cos_a + x * sin_a * 0.1);
        Ok(augmented)
    }
}

/// Mixup augmentation.
///
/// Creates new training samples by linearly interpolating between pairs of samples:
/// x' = λ * x₁ + (1 - λ) * x₂, y' = λ * y₁ + (1 - λ) * y₂
///
/// Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
#[derive(Debug, Clone)]
pub struct MixupAugmenter {
    /// Alpha parameter for Beta distribution (controls mixing strength).
    pub alpha: f64,
}

impl MixupAugmenter {
    /// Create a new mixup augmenter.
    ///
    /// # Arguments
    /// * `alpha` - Alpha parameter for Beta distribution
    pub fn new(alpha: f64) -> TrainResult<Self> {
        if alpha <= 0.0 {
            return Err(TrainError::InvalidParameter(
                "alpha must be positive".to_string(),
            ));
        }
        Ok(Self { alpha })
    }

    /// Apply mixup to a batch of data.
    ///
    /// # Arguments
    /// * `data` - Input data batch [N, features]
    /// * `labels` - Corresponding labels [N, classes]
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Tuple of (augmented_data, augmented_labels)
    pub fn augment_batch(
        &self,
        data: &ArrayView2<f64>,
        labels: &ArrayView2<f64>,
        rng: &mut StdRng,
    ) -> TrainResult<(Array2<f64>, Array2<f64>)> {
        if data.nrows() != labels.nrows() {
            return Err(TrainError::InvalidParameter(
                "data and labels must have same number of rows".to_string(),
            ));
        }

        let n = data.nrows();
        let mut augmented_data = Array::zeros(data.raw_dim());
        let mut augmented_labels = Array::zeros(labels.raw_dim());

        // Create random permutation
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        for i in 0..n {
            let j = indices[i];

            // Sample mixing coefficient from Beta distribution
            // Simplified: use uniform distribution as approximation
            let lambda = self.sample_beta(rng);

            // Mix data: x' = λ * x_i + (1 - λ) * x_j
            for k in 0..data.ncols() {
                augmented_data[[i, k]] = lambda * data[[i, k]] + (1.0 - lambda) * data[[j, k]];
            }

            // Mix labels: y' = λ * y_i + (1 - λ) * y_j
            for k in 0..labels.ncols() {
                augmented_labels[[i, k]] =
                    lambda * labels[[i, k]] + (1.0 - lambda) * labels[[j, k]];
            }
        }

        Ok((augmented_data, augmented_labels))
    }

    /// Sample from Beta(alpha, alpha) distribution.
    ///
    /// Simplified implementation using uniform distribution when alpha is close to 1.
    fn sample_beta(&self, rng: &mut StdRng) -> f64 {
        if self.alpha < 0.5 {
            // For small alpha, prefer values near 0 or 1
            if rng.random::<f64>() < 0.5 {
                rng.random::<f64>().powf(2.0)
            } else {
                1.0 - rng.random::<f64>().powf(2.0)
            }
        } else {
            // For alpha >= 0.5, approximate with uniform
            rng.random::<f64>()
        }
    }
}

impl Default for MixupAugmenter {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl DataAugmenter for MixupAugmenter {
    fn augment(&self, data: &ArrayView2<f64>, _rng: &mut StdRng) -> TrainResult<Array2<f64>> {
        // For single-sample augmentation, mix with itself (no-op)
        // In practice, mixup should be used with augment_batch
        Ok(data.to_owned())
    }
}

/// Composite augmenter that applies multiple augmentations sequentially.
#[derive(Clone, Default)]
pub struct CompositeAugmenter {
    augmenters: Vec<Box<dyn AugmenterClone>>,
}

impl std::fmt::Debug for CompositeAugmenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeAugmenter")
            .field("num_augmenters", &self.augmenters.len())
            .finish()
    }
}

/// Helper trait for cloning boxed augmenters.
trait AugmenterClone: DataAugmenter {
    fn clone_box(&self) -> Box<dyn AugmenterClone>;
}

impl<T: DataAugmenter + Clone + 'static> AugmenterClone for T {
    fn clone_box(&self) -> Box<dyn AugmenterClone> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn AugmenterClone> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl DataAugmenter for Box<dyn AugmenterClone> {
    fn augment(&self, data: &ArrayView2<f64>, rng: &mut StdRng) -> TrainResult<Array2<f64>> {
        (**self).augment(data, rng)
    }
}

impl CompositeAugmenter {
    /// Create a new composite augmenter.
    pub fn new() -> Self {
        Self {
            augmenters: Vec::new(),
        }
    }

    /// Add an augmenter to the pipeline.
    pub fn add<A: DataAugmenter + Clone + 'static>(&mut self, augmenter: A) {
        self.augmenters.push(Box::new(augmenter));
    }

    /// Get the number of augmenters.
    pub fn len(&self) -> usize {
        self.augmenters.len()
    }

    /// Check if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.augmenters.is_empty()
    }
}

impl DataAugmenter for CompositeAugmenter {
    fn augment(&self, data: &ArrayView2<f64>, rng: &mut StdRng) -> TrainResult<Array2<f64>> {
        let mut result = data.to_owned();

        for augmenter in &self.augmenters {
            result = augmenter.augment(&result.view(), rng)?;
        }

        Ok(result)
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
    fn test_no_augmentation() {
        let augmenter = NoAugmentation;
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let mut rng = create_test_rng();

        let augmented = augmenter.augment(&data.view(), &mut rng).unwrap();
        assert_eq!(augmented, data);
    }

    #[test]
    fn test_noise_augmenter() {
        let augmenter = NoiseAugmenter::new(0.1).unwrap();
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let mut rng = create_test_rng();

        let augmented = augmenter.augment(&data.view(), &mut rng).unwrap();

        // Shape should be preserved
        assert_eq!(augmented.shape(), data.shape());

        // Values should be different (with high probability)
        assert_ne!(augmented[[0, 0]], data[[0, 0]]);

        // But should be close to original values
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                let diff = (augmented[[i, j]] - data[[i, j]]).abs();
                assert!(diff < 1.0); // Within reasonable noise range
            }
        }
    }

    #[test]
    fn test_noise_augmenter_invalid() {
        let result = NoiseAugmenter::new(-0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_scale_augmenter() {
        let augmenter = ScaleAugmenter::new(0.2).unwrap();
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let mut rng = create_test_rng();

        let augmented = augmenter.augment(&data.view(), &mut rng).unwrap();

        // Shape should be preserved
        assert_eq!(augmented.shape(), data.shape());

        // All values should be scaled by the same factor
        let scale = augmented[[0, 0]] / data[[0, 0]];
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                let computed_scale = augmented[[i, j]] / data[[i, j]];
                assert!((computed_scale - scale).abs() < 1e-10);
            }
        }

        // Scale should be within range [0.8, 1.2]
        assert!((0.8..=1.2).contains(&scale));
    }

    #[test]
    fn test_scale_augmenter_invalid() {
        assert!(ScaleAugmenter::new(-0.1).is_err());
        assert!(ScaleAugmenter::new(1.5).is_err());
    }

    #[test]
    fn test_rotation_augmenter() {
        let augmenter = RotationAugmenter::default();
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let mut rng = create_test_rng();

        let augmented = augmenter.augment(&data.view(), &mut rng).unwrap();

        // Shape should be preserved
        assert_eq!(augmented.shape(), data.shape());
    }

    #[test]
    fn test_mixup_augmenter_batch() {
        let augmenter = MixupAugmenter::new(1.0).unwrap();
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let mut rng = create_test_rng();

        let (aug_data, aug_labels) = augmenter
            .augment_batch(&data.view(), &labels.view(), &mut rng)
            .unwrap();

        // Shapes should be preserved
        assert_eq!(aug_data.shape(), data.shape());
        assert_eq!(aug_labels.shape(), labels.shape());

        // Values should be interpolations (between min and max of original)
        let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        for &val in aug_data.iter() {
            assert!(val >= data_min && val <= data_max);
        }
    }

    #[test]
    fn test_mixup_invalid_alpha() {
        let result = MixupAugmenter::new(0.0);
        assert!(result.is_err());

        let result = MixupAugmenter::new(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_mixup_mismatched_shapes() {
        let augmenter = MixupAugmenter::default();
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array![[1.0, 0.0]]; // Wrong shape
        let mut rng = create_test_rng();

        let result = augmenter.augment_batch(&data.view(), &labels.view(), &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_composite_augmenter() {
        let mut composite = CompositeAugmenter::new();
        composite.add(NoiseAugmenter::new(0.01).unwrap());
        composite.add(ScaleAugmenter::new(0.1).unwrap());

        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let mut rng = create_test_rng();

        let augmented = composite.augment(&data.view(), &mut rng).unwrap();

        // Shape should be preserved
        assert_eq!(augmented.shape(), data.shape());

        // Values should be different due to augmentation
        assert_ne!(augmented[[0, 0]], data[[0, 0]]);
    }

    #[test]
    fn test_composite_empty() {
        let composite = CompositeAugmenter::new();
        assert!(composite.is_empty());
        assert_eq!(composite.len(), 0);

        let data = array![[1.0, 2.0]];
        let mut rng = create_test_rng();

        let augmented = composite.augment(&data.view(), &mut rng).unwrap();
        assert_eq!(augmented, data);
    }

    #[test]
    fn test_composite_multiple() {
        let mut composite = CompositeAugmenter::new();
        composite.add(NoAugmentation);
        composite.add(ScaleAugmenter::default());
        composite.add(NoiseAugmenter::default());

        assert_eq!(composite.len(), 3);
        assert!(!composite.is_empty());
    }
}
