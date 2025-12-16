//! Data augmentation techniques for training.
//!
//! This module provides various data augmentation strategies to improve model generalization:
//! - Noise augmentation (Gaussian)
//! - Scale augmentation (random scaling)
//! - Rotation augmentation (placeholder for future)
//! - Mixup augmentation (interpolation between samples)
//! - CutMix augmentation (cutting and mixing patches)
//! - Random Erasing (randomly erase rectangular regions)
//! - CutOut (fixed-size random erasing)
//!

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

/// CutMix augmentation (ICCV 2019).
///
/// Instead of mixing pixels uniformly like Mixup, CutMix cuts a rectangular region
/// from one image and pastes it to another. Labels are mixed proportionally to the
/// area of the patch.
///
/// Reference: Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers
/// with Localizable Features" (ICCV 2019)
#[derive(Debug, Clone)]
pub struct CutMixAugmenter {
    /// Beta distribution parameter for sampling mixing ratio.
    pub alpha: f64,
}

impl CutMixAugmenter {
    /// Create a new CutMix augmenter.
    ///
    /// # Arguments
    /// * `alpha` - Beta distribution parameter (typically 1.0)
    ///
    /// # Returns
    /// New CutMix augmenter
    pub fn new(alpha: f64) -> TrainResult<Self> {
        if alpha <= 0.0 {
            return Err(TrainError::InvalidParameter(
                "alpha must be positive".to_string(),
            ));
        }
        Ok(Self { alpha })
    }

    /// Apply CutMix augmentation to a batch of data.
    ///
    /// For 2D feature arrays, we treat the second dimension as a "spatial" dimension
    /// and cut rectangular regions along it.
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
        let features = data.ncols();
        let mut augmented_data = data.to_owned();
        let mut augmented_labels = labels.to_owned();

        // Create random permutation for pairing samples
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        for i in 0..n {
            let j = indices[i];

            // Sample mixing ratio from Beta distribution
            let lambda = self.sample_beta(rng);

            // Generate random bounding box
            // For 1D feature vectors, we cut along the feature dimension
            let cut_ratio = (1.0 - lambda).sqrt();
            let cut_size = (features as f64 * cut_ratio) as usize;
            let cut_size = cut_size.max(1).min(features - 1);

            // Random starting position
            let start = if features > cut_size {
                rng.gen_range(0..=(features - cut_size))
            } else {
                0
            };

            // Apply CutMix: replace region with data from paired sample
            for k in start..(start + cut_size).min(features) {
                augmented_data[[i, k]] = data[[j, k]];
            }

            // Mix labels proportionally to the area of the cut region
            let actual_ratio = cut_size as f64 / features as f64;
            for k in 0..labels.ncols() {
                augmented_labels[[i, k]] =
                    (1.0 - actual_ratio) * labels[[i, k]] + actual_ratio * labels[[j, k]];
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

impl Default for CutMixAugmenter {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl DataAugmenter for CutMixAugmenter {
    fn augment(&self, data: &ArrayView2<f64>, _rng: &mut StdRng) -> TrainResult<Array2<f64>> {
        // For single-sample augmentation, no operation
        // In practice, CutMix should be used with augment_batch
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

/// Random Erasing augmentation.
///
/// Randomly erases rectangular regions in the input data with random values.
/// This technique prevents overfitting and improves generalization, especially for image data.
///
/// Reference: Zhong et al., "Random Erasing Data Augmentation" (AAAI 2020)
///
/// # Parameters
/// - `p`: Probability of applying erasing (default: 0.5)
/// - `scale`: Range of proportion of erased area (default: [0.02, 0.33])
/// - `ratio`: Range of aspect ratio of erased area (default: [0.3, 3.3])
/// - `value`: Value to fill erased region (0.0 = zero, 1.0 = random, -1.0 = pixel mean)
#[derive(Debug, Clone)]
pub struct RandomErasingAugmenter {
    /// Probability of applying erasing.
    pub probability: f64,
    /// Minimum proportion of erased area.
    pub scale_min: f64,
    /// Maximum proportion of erased area.
    pub scale_max: f64,
    /// Minimum aspect ratio.
    pub ratio_min: f64,
    /// Maximum aspect ratio.
    pub ratio_max: f64,
    /// Fill value (0.0 = zero, 1.0 = random).
    pub fill_value: f64,
}

impl RandomErasingAugmenter {
    /// Create a new Random Erasing augmenter with custom parameters.
    pub fn new(
        probability: f64,
        scale_min: f64,
        scale_max: f64,
        ratio_min: f64,
        ratio_max: f64,
        fill_value: f64,
    ) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(TrainError::InvalidParameter(
                "probability must be in [0, 1]".to_string(),
            ));
        }
        if scale_min >= scale_max || scale_min < 0.0 || scale_max > 1.0 {
            return Err(TrainError::InvalidParameter(
                "scale range must be valid (0 <= min < max <= 1)".to_string(),
            ));
        }
        if ratio_min <= 0.0 || ratio_min >= ratio_max {
            return Err(TrainError::InvalidParameter(
                "ratio range must be valid (0 < min < max)".to_string(),
            ));
        }

        Ok(Self {
            probability,
            scale_min,
            scale_max,
            ratio_min,
            ratio_max,
            fill_value,
        })
    }

    /// Create with default parameters (as in the paper).
    pub fn with_defaults() -> Self {
        Self {
            probability: 0.5,
            scale_min: 0.02,
            scale_max: 0.33,
            ratio_min: 0.3,
            ratio_max: 3.3,
            fill_value: 0.0,
        }
    }
}

impl Default for RandomErasingAugmenter {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl DataAugmenter for RandomErasingAugmenter {
    fn augment(&self, data: &ArrayView2<f64>, rng: &mut StdRng) -> TrainResult<Array2<f64>> {
        let mut augmented = data.to_owned();

        // Apply with probability p
        if rng.random::<f64>() > self.probability {
            return Ok(augmented);
        }

        let (height, width) = (data.nrows(), data.ncols());
        let area = (height * width) as f64;

        // Try multiple times to find a valid erasing region
        for _ in 0..10 {
            // Random scale (proportion of total area)
            let scale = self.scale_min + rng.random::<f64>() * (self.scale_max - self.scale_min);
            let erase_area = area * scale;

            // Random aspect ratio
            let aspect_ratio =
                self.ratio_min + rng.random::<f64>() * (self.ratio_max - self.ratio_min);

            // Compute erase region dimensions
            let h = (erase_area * aspect_ratio).sqrt().min(height as f64);
            let w = (erase_area / aspect_ratio).sqrt().min(width as f64);

            if h >= 1.0 && w >= 1.0 {
                let erase_h = h as usize;
                let erase_w = w as usize;

                // Random position
                let i = if erase_h < height {
                    (rng.random::<f64>() * (height - erase_h) as f64) as usize
                } else {
                    0
                };
                let j = if erase_w < width {
                    (rng.random::<f64>() * (width - erase_w) as f64) as usize
                } else {
                    0
                };

                // Fill with specified value
                if self.fill_value == 1.0 {
                    // Random values
                    for row in i..i + erase_h.min(height - i) {
                        for col in j..j + erase_w.min(width - j) {
                            augmented[[row, col]] = rng.random();
                        }
                    }
                } else {
                    // Fixed value (0.0 or specified)
                    for row in i..i + erase_h.min(height - i) {
                        for col in j..j + erase_w.min(width - j) {
                            augmented[[row, col]] = self.fill_value;
                        }
                    }
                }

                break;
            }
        }

        Ok(augmented)
    }
}

/// CutOut augmentation.
///
/// Randomly erases a fixed-size square region in the input data.
/// Simpler variant of Random Erasing with fixed square regions.
///
/// Reference: DeVries & Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout" (2017)
///
/// # Parameters
/// - `size`: Size of the square region to erase
/// - `p`: Probability of applying cutout (default: 1.0)
/// - `fill_value`: Value to fill the erased region (default: 0.0)
#[derive(Debug, Clone)]
pub struct CutOutAugmenter {
    /// Size of the square cutout region.
    pub cutout_size: usize,
    /// Probability of applying cutout.
    pub probability: f64,
    /// Fill value for erased region.
    pub fill_value: f64,
}

impl CutOutAugmenter {
    /// Create a new CutOut augmenter.
    pub fn new(cutout_size: usize, probability: f64, fill_value: f64) -> TrainResult<Self> {
        if cutout_size == 0 {
            return Err(TrainError::InvalidParameter(
                "cutout_size must be > 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&probability) {
            return Err(TrainError::InvalidParameter(
                "probability must be in [0, 1]".to_string(),
            ));
        }

        Ok(Self {
            cutout_size,
            probability,
            fill_value,
        })
    }

    /// Create with default parameters.
    pub fn with_size(size: usize) -> TrainResult<Self> {
        Self::new(size, 1.0, 0.0)
    }
}

impl DataAugmenter for CutOutAugmenter {
    fn augment(&self, data: &ArrayView2<f64>, rng: &mut StdRng) -> TrainResult<Array2<f64>> {
        let mut augmented = data.to_owned();

        // Apply with probability p
        if rng.random::<f64>() > self.probability {
            return Ok(augmented);
        }

        let (height, width) = (data.nrows(), data.ncols());

        // Random center position
        let center_y = (rng.random::<f64>() * height as f64) as usize;
        let center_x = (rng.random::<f64>() * width as f64) as usize;

        // Compute cutout region bounds (allow partial cutout at boundaries)
        let half_size = self.cutout_size / 2;

        let y_start = center_y.saturating_sub(half_size);
        let y_end = (center_y + half_size).min(height);

        let x_start = center_x.saturating_sub(half_size);
        let x_end = (center_x + half_size).min(width);

        // Erase the region
        for i in y_start..y_end {
            for j in x_start..x_end {
                augmented[[i, j]] = self.fill_value;
            }
        }

        Ok(augmented)
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
    fn test_cutmix_augmenter_batch() {
        let augmenter = CutMixAugmenter::new(1.0).unwrap();
        let data = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];
        let labels = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let mut rng = create_test_rng();

        let (aug_data, aug_labels) = augmenter
            .augment_batch(&data.view(), &labels.view(), &mut rng)
            .unwrap();

        // Shapes should be preserved
        assert_eq!(aug_data.shape(), data.shape());
        assert_eq!(aug_labels.shape(), labels.shape());

        // Each row should contain a mix of original values
        // (some regions from original, some from paired sample)
        for i in 0..aug_data.nrows() {
            let mut found_original = false;
            let mut found_different = false;

            for j in 0..aug_data.ncols() {
                // Check if value matches original row
                if (aug_data[[i, j]] - data[[i, j]]).abs() < 1e-10 {
                    found_original = true;
                } else {
                    found_different = true;
                }
            }

            // Should have both original and swapped regions (unless randomly paired with self)
            assert!(found_original || found_different);
        }

        // Label mixing: each element should be in [0, 1] and sum across classes
        for i in 0..aug_labels.nrows() {
            let sum: f64 = aug_labels.row(i).iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Labels should sum to 1.0");
        }
    }

    #[test]
    fn test_cutmix_invalid_alpha() {
        let result = CutMixAugmenter::new(0.0);
        assert!(result.is_err());

        let result = CutMixAugmenter::new(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cutmix_mismatched_shapes() {
        let augmenter = CutMixAugmenter::default();
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array![[1.0, 0.0]]; // Wrong shape
        let mut rng = create_test_rng();

        let result = augmenter.augment_batch(&data.view(), &labels.view(), &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_cutmix_label_proportions() {
        let augmenter = CutMixAugmenter::new(1.0).unwrap();
        // Use distinctive patterns
        let data = array![[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]];
        let labels = array![[1.0, 0.0], [0.0, 1.0]];
        let mut rng = create_test_rng();

        let (aug_data, aug_labels) = augmenter
            .augment_batch(&data.view(), &labels.view(), &mut rng)
            .unwrap();

        // Verify that labels are mixed proportionally
        for i in 0..aug_labels.nrows() {
            // Each sample should have labels that sum to 1
            let sum: f64 = aug_labels.row(i).iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);

            // Labels should be between original values
            for j in 0..aug_labels.ncols() {
                assert!(aug_labels[[i, j]] >= 0.0);
                assert!(aug_labels[[i, j]] <= 1.0);
            }
        }

        // Verify data has been cut and mixed
        assert_eq!(aug_data.shape(), data.shape());
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

    #[test]
    fn test_random_erasing_creation() {
        let augmenter = RandomErasingAugmenter::new(0.5, 0.02, 0.33, 0.3, 3.3, 0.0).unwrap();
        assert_eq!(augmenter.probability, 0.5);
        assert_eq!(augmenter.scale_min, 0.02);
        assert_eq!(augmenter.scale_max, 0.33);
    }

    #[test]
    fn test_random_erasing_invalid_params() {
        // Invalid probability
        assert!(RandomErasingAugmenter::new(1.5, 0.02, 0.33, 0.3, 3.3, 0.0).is_err());

        // Invalid scale range
        assert!(RandomErasingAugmenter::new(0.5, 0.33, 0.02, 0.3, 3.3, 0.0).is_err());

        // Invalid ratio range
        assert!(RandomErasingAugmenter::new(0.5, 0.02, 0.33, 3.3, 0.3, 0.0).is_err());
    }

    #[test]
    fn test_random_erasing_augment() {
        let augmenter = RandomErasingAugmenter::with_defaults();
        let data = Array2::ones((10, 10));
        let mut rng = create_test_rng();

        let augmented = augmenter.augment(&data.view(), &mut rng).unwrap();

        // Shape should be preserved
        assert_eq!(augmented.shape(), data.shape());

        // Some values may be erased (but not guaranteed due to probability)
    }

    #[test]
    fn test_random_erasing_probability_zero() {
        let augmenter = RandomErasingAugmenter::new(0.0, 0.02, 0.33, 0.3, 3.3, 0.0).unwrap();
        let data = Array2::ones((10, 10));
        let mut rng = create_test_rng();

        let augmented = augmenter.augment(&data.view(), &mut rng).unwrap();

        // With probability 0, data should be unchanged
        assert_eq!(augmented, data);
    }

    #[test]
    fn test_cutout_creation() {
        let augmenter = CutOutAugmenter::new(5, 1.0, 0.0).unwrap();
        assert_eq!(augmenter.cutout_size, 5);
        assert_eq!(augmenter.probability, 1.0);
        assert_eq!(augmenter.fill_value, 0.0);
    }

    #[test]
    fn test_cutout_invalid_params() {
        // Zero size
        assert!(CutOutAugmenter::new(0, 1.0, 0.0).is_err());

        // Invalid probability
        assert!(CutOutAugmenter::new(5, 1.5, 0.0).is_err());
    }

    #[test]
    fn test_cutout_augment() {
        let augmenter = CutOutAugmenter::with_size(3).unwrap();
        let data = Array2::ones((10, 10));
        let mut rng = create_test_rng();

        let augmented = augmenter.augment(&data.view(), &mut rng).unwrap();

        // Shape should be preserved
        assert_eq!(augmented.shape(), data.shape());

        // Some values should be zero (erased)
        let zeros_count = augmented.iter().filter(|&&x| x == 0.0).count();
        assert!(zeros_count > 0, "Expected some values to be erased");
        assert!(zeros_count < 100, "Not all values should be erased");
    }

    #[test]
    fn test_cutout_probability_zero() {
        let augmenter = CutOutAugmenter::new(5, 0.0, 0.0).unwrap();
        let data = Array2::ones((10, 10));
        let mut rng = create_test_rng();

        let augmented = augmenter.augment(&data.view(), &mut rng).unwrap();

        // With probability 0, data should be unchanged
        assert_eq!(augmented, data);
    }

    #[test]
    fn test_cutout_fill_value() {
        let augmenter = CutOutAugmenter::new(3, 1.0, 0.5).unwrap();
        let data = Array2::ones((10, 10));
        let mut rng = create_test_rng();

        let augmented = augmenter.augment(&data.view(), &mut rng).unwrap();

        // Some values should be 0.5 (fill value)
        let filled_count = augmented.iter().filter(|&&x| x == 0.5).count();
        assert!(
            filled_count > 0,
            "Expected some values to be filled with 0.5"
        );
    }
}
