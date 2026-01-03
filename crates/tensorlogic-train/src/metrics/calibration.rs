//! Calibration metrics.

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{ArrayView, Ix2};

use super::Metric;

/// Expected Calibration Error (ECE) metric.
///
/// Measures the difference between predicted probabilities and actual accuracy.
/// ECE divides predictions into bins and computes the average difference between
/// confidence and accuracy across bins, weighted by bin frequency.
///
/// Lower ECE indicates better calibration.
///
/// Reference: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
#[derive(Debug, Clone)]
pub struct ExpectedCalibrationError {
    /// Number of bins for calibration
    pub num_bins: usize,
}

impl Default for ExpectedCalibrationError {
    fn default() -> Self {
        Self { num_bins: 10 }
    }
}

impl ExpectedCalibrationError {
    /// Create with custom number of bins.
    pub fn new(num_bins: usize) -> Self {
        Self { num_bins }
    }
}

impl Metric for ExpectedCalibrationError {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.shape() != targets.shape() {
            return Err(TrainError::MetricsError(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let n_samples = predictions.nrows();
        if n_samples == 0 {
            return Ok(0.0);
        }

        // Initialize bins
        let mut bin_counts = vec![0usize; self.num_bins];
        let mut bin_confidences = vec![0.0; self.num_bins];
        let mut bin_accuracies = vec![0.0; self.num_bins];

        for i in 0..n_samples {
            // Get predicted class and confidence
            let mut pred_class = 0;
            let mut max_confidence = predictions[[i, 0]];
            for j in 1..predictions.ncols() {
                if predictions[[i, j]] > max_confidence {
                    max_confidence = predictions[[i, j]];
                    pred_class = j;
                }
            }

            // Get true class
            let mut true_class = 0;
            let mut max_target = targets[[i, 0]];
            for j in 1..targets.ncols() {
                if targets[[i, j]] > max_target {
                    max_target = targets[[i, j]];
                    true_class = j;
                }
            }

            // Determine bin index
            let bin_idx =
                ((max_confidence * self.num_bins as f64).floor() as usize).min(self.num_bins - 1);

            // Update bin statistics
            bin_counts[bin_idx] += 1;
            bin_confidences[bin_idx] += max_confidence;
            if pred_class == true_class {
                bin_accuracies[bin_idx] += 1.0;
            }
        }

        // Compute ECE
        let mut ece = 0.0;
        for i in 0..self.num_bins {
            if bin_counts[i] > 0 {
                let bin_confidence = bin_confidences[i] / bin_counts[i] as f64;
                let bin_accuracy = bin_accuracies[i] / bin_counts[i] as f64;
                let weight = bin_counts[i] as f64 / n_samples as f64;

                ece += weight * (bin_confidence - bin_accuracy).abs();
            }
        }

        Ok(ece)
    }

    fn name(&self) -> &str {
        "expected_calibration_error"
    }
}

/// Maximum Calibration Error (MCE) metric.
///
/// Measures the worst-case calibration error across all bins.
/// MCE is the maximum absolute difference between confidence and accuracy
/// in any bin.
///
/// Lower MCE indicates better calibration.
///
/// Reference: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
#[derive(Debug, Clone)]
pub struct MaximumCalibrationError {
    /// Number of bins for calibration
    pub num_bins: usize,
}

impl Default for MaximumCalibrationError {
    fn default() -> Self {
        Self { num_bins: 10 }
    }
}

impl MaximumCalibrationError {
    /// Create with custom number of bins.
    pub fn new(num_bins: usize) -> Self {
        Self { num_bins }
    }
}

impl Metric for MaximumCalibrationError {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.shape() != targets.shape() {
            return Err(TrainError::MetricsError(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let n_samples = predictions.nrows();
        if n_samples == 0 {
            return Ok(0.0);
        }

        // Initialize bins
        let mut bin_counts = vec![0usize; self.num_bins];
        let mut bin_confidences = vec![0.0; self.num_bins];
        let mut bin_accuracies = vec![0.0; self.num_bins];

        for i in 0..n_samples {
            // Get predicted class and confidence
            let mut pred_class = 0;
            let mut max_confidence = predictions[[i, 0]];
            for j in 1..predictions.ncols() {
                if predictions[[i, j]] > max_confidence {
                    max_confidence = predictions[[i, j]];
                    pred_class = j;
                }
            }

            // Get true class
            let mut true_class = 0;
            let mut max_target = targets[[i, 0]];
            for j in 1..targets.ncols() {
                if targets[[i, j]] > max_target {
                    max_target = targets[[i, j]];
                    true_class = j;
                }
            }

            // Determine bin index
            let bin_idx =
                ((max_confidence * self.num_bins as f64).floor() as usize).min(self.num_bins - 1);

            // Update bin statistics
            bin_counts[bin_idx] += 1;
            bin_confidences[bin_idx] += max_confidence;
            if pred_class == true_class {
                bin_accuracies[bin_idx] += 1.0;
            }
        }

        // Compute MCE (maximum calibration error)
        let mut mce: f64 = 0.0;
        for i in 0..self.num_bins {
            if bin_counts[i] > 0 {
                let bin_confidence = bin_confidences[i] / bin_counts[i] as f64;
                let bin_accuracy = bin_accuracies[i] / bin_counts[i] as f64;
                let calibration_error = (bin_confidence - bin_accuracy).abs();

                mce = mce.max(calibration_error);
            }
        }

        Ok(mce)
    }

    fn name(&self) -> &str {
        "maximum_calibration_error"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_expected_calibration_error_perfect() {
        let metric = ExpectedCalibrationError::default();

        // Perfect calibration: confidence matches accuracy
        // All predictions at 100% confidence and all correct
        let predictions = array![[0.95, 0.05], [0.05, 0.95], [0.95, 0.05], [0.05, 0.95]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

        let ece = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Should be very small for perfectly calibrated predictions
        assert!(ece < 0.1);
    }

    #[test]
    fn test_expected_calibration_error_poor() {
        let metric = ExpectedCalibrationError::default();

        // Poor calibration: high confidence but wrong predictions
        let predictions = array![[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]];
        let targets = array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];

        let ece = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Should be high for poorly calibrated predictions
        assert!(ece > 0.5);
    }

    #[test]
    fn test_expected_calibration_error_custom_bins() {
        let metric = ExpectedCalibrationError::new(5); // Use 5 bins instead of 10

        let predictions = array![[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]];
        let targets = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];

        let ece = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        assert!((0.0..=1.0).contains(&ece));
    }

    #[test]
    fn test_maximum_calibration_error_perfect() {
        let metric = MaximumCalibrationError::default();

        // Perfect calibration
        let predictions = array![[0.95, 0.05], [0.05, 0.95], [0.95, 0.05], [0.05, 0.95]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

        let mce = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Should be small for well-calibrated predictions
        assert!(mce < 0.15);
    }

    #[test]
    fn test_maximum_calibration_error_poor() {
        let metric = MaximumCalibrationError::default();

        // One bin with very poor calibration
        let predictions = array![[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]];
        let targets = array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];

        let mce = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // MCE should capture the worst bin
        assert!(mce > 0.5);
    }

    #[test]
    fn test_calibration_metrics_empty() {
        let ece_metric = ExpectedCalibrationError::default();
        let mce_metric = MaximumCalibrationError::default();

        use scirs2_core::ndarray::Array;
        let empty_predictions: Array<f64, _> = Array::zeros((0, 2));
        let empty_targets: Array<f64, _> = Array::zeros((0, 2));

        let ece = ece_metric
            .compute(&empty_predictions.view(), &empty_targets.view())
            .unwrap();
        let mce = mce_metric
            .compute(&empty_predictions.view(), &empty_targets.view())
            .unwrap();

        assert_eq!(ece, 0.0);
        assert_eq!(mce, 0.0);
    }

    #[test]
    fn test_calibration_metrics_shape_mismatch() {
        let metric = ExpectedCalibrationError::default();

        let predictions = array![[0.9, 0.1], [0.8, 0.2]];
        let targets = array![[1.0, 0.0, 0.0]]; // Wrong shape

        let result = metric.compute(&predictions.view(), &targets.view());
        assert!(result.is_err());
    }
}
