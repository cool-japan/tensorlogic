//! Label smoothing regularization for improved generalization.
//!
//! Label smoothing is a regularization technique that prevents the model from becoming
//! overconfident by smoothing the target distribution.

use crate::{Loss, TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

/// Label smoothing cross-entropy loss.
///
/// Based on "Rethinking the Inception Architecture for Computer Vision" (Szegedy et al., 2016).
/// Replaces hard 0/1 labels with smoothed distribution:
/// - True class: 1 - epsilon
/// - Other classes: epsilon / (num_classes - 1)
#[derive(Debug, Clone)]
pub struct LabelSmoothingLoss {
    /// Smoothing parameter (typically 0.1).
    pub epsilon: f64,
    /// Number of classes.
    pub num_classes: usize,
}

impl LabelSmoothingLoss {
    /// Create a new label smoothing loss.
    ///
    /// # Arguments
    /// * `epsilon` - Smoothing parameter (0 = no smoothing, higher = more smoothing)
    /// * `num_classes` - Number of classes
    pub fn new(epsilon: f64, num_classes: usize) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&epsilon) {
            return Err(TrainError::ConfigError(
                "Epsilon must be between 0 and 1".to_string(),
            ));
        }

        if num_classes == 0 {
            return Err(TrainError::ConfigError(
                "Number of classes must be positive".to_string(),
            ));
        }

        Ok(Self {
            epsilon,
            num_classes,
        })
    }

    /// Apply label smoothing to targets.
    ///
    /// # Arguments
    /// * `targets` - One-hot encoded targets
    ///
    /// # Returns
    /// Smoothed targets
    pub fn smooth_labels(&self, targets: &ArrayView<f64, Ix2>) -> Array<f64, Ix2> {
        if targets.ncols() != self.num_classes {
            // If mismatch, return original (will error in loss computation)
            return targets.to_owned();
        }

        let mut smoothed = Array::zeros(targets.raw_dim());

        let true_confidence = 1.0 - self.epsilon;
        let other_confidence = self.epsilon / (self.num_classes - 1) as f64;

        for i in 0..targets.nrows() {
            for j in 0..targets.ncols() {
                if targets[[i, j]] > 0.5 {
                    // True class
                    smoothed[[i, j]] = true_confidence;
                } else {
                    // Other classes
                    smoothed[[i, j]] = other_confidence;
                }
            }
        }

        smoothed
    }
}

impl Loss for LabelSmoothingLoss {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        if predictions.ncols() != self.num_classes {
            return Err(TrainError::LossError(format!(
                "Number of classes mismatch: expected {}, got {}",
                self.num_classes,
                predictions.ncols()
            )));
        }

        // Apply label smoothing
        let smoothed_targets = self.smooth_labels(targets);

        // Compute cross-entropy with smoothed labels
        let mut total_loss = 0.0;
        let n_samples = predictions.nrows();

        for i in 0..n_samples {
            // Softmax normalization
            let max_pred = predictions
                .row(i)
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let exp_preds: Vec<f64> = predictions
                .row(i)
                .iter()
                .map(|&x| (x - max_pred).exp())
                .collect();

            let sum_exp: f64 = exp_preds.iter().sum();

            // Cross-entropy
            for j in 0..predictions.ncols() {
                let prob = exp_preds[j] / sum_exp;
                let target = smoothed_targets[[i, j]];

                if target > 1e-8 {
                    total_loss -= target * (prob + 1e-8).ln();
                }
            }
        }

        Ok(total_loss / n_samples as f64)
    }

    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        if predictions.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        if predictions.ncols() != self.num_classes {
            return Err(TrainError::LossError(format!(
                "Number of classes mismatch: expected {}, got {}",
                self.num_classes,
                predictions.ncols()
            )));
        }

        // Apply label smoothing
        let smoothed_targets = self.smooth_labels(targets);

        let n_samples = predictions.nrows();
        let mut grad = Array::zeros(predictions.raw_dim());

        for i in 0..n_samples {
            // Softmax normalization
            let max_pred = predictions
                .row(i)
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let exp_preds: Vec<f64> = predictions
                .row(i)
                .iter()
                .map(|&x| (x - max_pred).exp())
                .collect();

            let sum_exp: f64 = exp_preds.iter().sum();

            // Gradient: softmax(predictions) - smoothed_targets
            for j in 0..predictions.ncols() {
                let prob = exp_preds[j] / sum_exp;
                let target = smoothed_targets[[i, j]];
                grad[[i, j]] = (prob - target) / n_samples as f64;
            }
        }

        Ok(grad)
    }

    fn name(&self) -> &str {
        "label_smoothing"
    }
}

/// Mixup data augmentation that mixes training examples and their labels.
///
/// Based on "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018).
#[derive(Debug)]
pub struct MixupLoss {
    /// Alpha parameter for Beta distribution (typically 1.0).
    pub alpha: f64,
    /// Base loss function.
    pub base_loss: Box<dyn Loss>,
}

impl MixupLoss {
    /// Create a new Mixup loss.
    ///
    /// # Arguments
    /// * `alpha` - Beta distribution parameter (higher = more mixing)
    /// * `base_loss` - Underlying loss function
    pub fn new(alpha: f64, base_loss: Box<dyn Loss>) -> TrainResult<Self> {
        if alpha <= 0.0 {
            return Err(TrainError::ConfigError(
                "Alpha must be positive".to_string(),
            ));
        }

        Ok(Self { alpha, base_loss })
    }

    /// Compute mixup loss.
    ///
    /// Note: In practice, mixup is applied during data loading, not in the loss function.
    /// This implementation assumes pre-mixed inputs and targets.
    pub fn compute_mixup(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        mixed_targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        // With pre-mixed targets, just use the base loss
        self.base_loss.compute(predictions, mixed_targets)
    }

    /// Mix two batches of data with random lambda.
    ///
    /// # Arguments
    /// * `data1` - First batch of data
    /// * `data2` - Second batch of data
    /// * `lambda` - Mixing coefficient (0 to 1)
    ///
    /// # Returns
    /// Mixed data
    pub fn mix_data(
        data1: &ArrayView<f64, Ix2>,
        data2: &ArrayView<f64, Ix2>,
        lambda: f64,
    ) -> TrainResult<Array<f64, Ix2>> {
        if data1.shape() != data2.shape() {
            return Err(TrainError::LossError(
                "Data shapes must match for mixing".to_string(),
            ));
        }

        let mixed = data1 * lambda + data2 * (1.0 - lambda);
        Ok(mixed.to_owned())
    }

    /// Sample mixing coefficient from Beta distribution.
    ///
    /// In practice, use a proper random number generator.
    /// This is a simplified implementation for demonstration.
    #[allow(dead_code)]
    fn sample_lambda(&self) -> f64 {
        // Simplified: return midpoint
        // In real implementation, sample from Beta(alpha, alpha)
        0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    fn test_label_smoothing_creation() {
        let loss = LabelSmoothingLoss::new(0.1, 10);
        assert!(loss.is_ok());

        let loss = loss.unwrap();
        assert_eq!(loss.epsilon, 0.1);
        assert_eq!(loss.num_classes, 10);
    }

    #[test]
    fn test_label_smoothing_invalid_epsilon() {
        assert!(LabelSmoothingLoss::new(-0.1, 10).is_err());
        assert!(LabelSmoothingLoss::new(1.5, 10).is_err());
    }

    #[test]
    fn test_label_smoothing_smooth_labels() {
        let loss = LabelSmoothingLoss::new(0.1, 3).unwrap();

        // One-hot encoded: class 1 is true
        let targets = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let smoothed = loss.smooth_labels(&targets.view());

        // True class should have 1 - epsilon = 0.9
        assert!((smoothed[[0, 1]] - 0.9).abs() < 1e-6);

        // Other classes should have epsilon / (num_classes - 1) = 0.1 / 2 = 0.05
        assert!((smoothed[[0, 0]] - 0.05).abs() < 1e-6);
        assert!((smoothed[[0, 2]] - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_label_smoothing_loss_compute() {
        let loss = LabelSmoothingLoss::new(0.1, 3).unwrap();

        let predictions = array![[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]];
        let targets = array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let result = loss.compute(&predictions.view(), &targets.view());
        assert!(result.is_ok());

        let loss_value = result.unwrap();
        assert!(loss_value > 0.0);
        assert!(loss_value.is_finite());
    }

    #[test]
    fn test_mixup_loss_creation() {
        use crate::MseLoss;

        let loss = MixupLoss::new(1.0, Box::new(MseLoss));
        assert!(loss.is_ok());

        assert!(MixupLoss::new(0.0, Box::new(MseLoss)).is_err());
        assert!(MixupLoss::new(-1.0, Box::new(MseLoss)).is_err());
    }

    #[test]
    fn test_mixup_mix_data() {
        let data1 = array![[1.0, 2.0], [3.0, 4.0]];
        let data2 = array![[5.0, 6.0], [7.0, 8.0]];

        let mixed = MixupLoss::mix_data(&data1.view(), &data2.view(), 0.5).unwrap();

        // With lambda=0.5, should be average
        assert!((mixed[[0, 0]] - 3.0).abs() < 1e-6);
        assert!((mixed[[0, 1]] - 4.0).abs() < 1e-6);
        assert!((mixed[[1, 0]] - 5.0).abs() < 1e-6);
        assert!((mixed[[1, 1]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_mixup_mix_data_lambda_extremes() {
        let data1 = array![[1.0, 2.0]];
        let data2 = array![[5.0, 6.0]];

        // Lambda = 1.0 should return data1
        let mixed = MixupLoss::mix_data(&data1.view(), &data2.view(), 1.0).unwrap();
        assert!((mixed[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((mixed[[0, 1]] - 2.0).abs() < 1e-6);

        // Lambda = 0.0 should return data2
        let mixed = MixupLoss::mix_data(&data1.view(), &data2.view(), 0.0).unwrap();
        assert!((mixed[[0, 0]] - 5.0).abs() < 1e-6);
        assert!((mixed[[0, 1]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_label_smoothing_zero_epsilon() {
        // With epsilon = 0, should behave like regular one-hot
        let loss = LabelSmoothingLoss::new(0.0, 3).unwrap();

        let targets = array![[0.0, 1.0, 0.0]];
        let smoothed = loss.smooth_labels(&targets.view());

        assert!((smoothed[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((smoothed[[0, 1]] - 1.0).abs() < 1e-6);
        assert!((smoothed[[0, 2]] - 0.0).abs() < 1e-6);
    }
}
