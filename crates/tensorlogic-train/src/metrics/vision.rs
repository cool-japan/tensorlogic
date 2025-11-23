//! Computer vision metrics.

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{ArrayView, Ix2};

use super::Metric;

/// Intersection over Union (IoU) metric for segmentation tasks.
///
/// IoU = (Intersection) / (Union) = TP / (TP + FP + FN)
///
/// Also known as Jaccard Index, this is a key metric for:
/// - Semantic segmentation
/// - Instance segmentation
/// - Object detection (bounding box overlap)
#[derive(Debug, Clone)]
pub struct IoU {
    /// Threshold for converting predictions to binary
    pub threshold: f64,
    /// Small epsilon to avoid division by zero
    pub epsilon: f64,
}

impl Default for IoU {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            epsilon: 1e-7,
        }
    }
}

impl IoU {
    /// Create a new IoU metric with custom threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            epsilon: 1e-7,
        }
    }
}

impl Metric for IoU {
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

        let mut intersection = 0.0;
        let mut union = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = if predictions[[i, j]] >= self.threshold {
                    1.0
                } else {
                    0.0
                };
                let target = targets[[i, j]];

                intersection += pred * target;
                union += (pred + target - pred * target).max(0.0);
            }
        }

        Ok(intersection / (union + self.epsilon))
    }

    fn name(&self) -> &str {
        "iou"
    }
}

/// Mean Intersection over Union (mIoU) metric for multi-class segmentation.
///
/// Computes IoU for each class separately and returns the mean.
/// This is the standard evaluation metric for semantic segmentation.
#[derive(Debug, Clone)]
pub struct MeanIoU {
    /// Threshold for converting predictions to binary
    pub threshold: f64,
    /// Small epsilon to avoid division by zero
    pub epsilon: f64,
}

impl Default for MeanIoU {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            epsilon: 1e-7,
        }
    }
}

impl Metric for MeanIoU {
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

        let num_classes = predictions.ncols();
        let mut class_ious = Vec::new();

        // Compute IoU for each class
        for class_idx in 0..num_classes {
            let mut intersection = 0.0;
            let mut union = 0.0;

            for i in 0..predictions.nrows() {
                let pred = if predictions[[i, class_idx]] >= self.threshold {
                    1.0
                } else {
                    0.0
                };
                let target = targets[[i, class_idx]];

                intersection += pred * target;
                union += (pred + target - pred * target).max(0.0);
            }

            if union > self.epsilon {
                class_ious.push(intersection / union);
            }
        }

        if class_ious.is_empty() {
            return Ok(0.0);
        }

        Ok(class_ious.iter().sum::<f64>() / class_ious.len() as f64)
    }

    fn name(&self) -> &str {
        "mean_iou"
    }
}

/// Dice Coefficient metric (F1 Score variant for segmentation).
///
/// Dice = 2 * (Intersection) / (|A| + |B|) = 2TP / (2TP + FP + FN)
///
/// Often used in medical image segmentation.
/// Range: [0, 1] where 1 is perfect overlap.
#[derive(Debug, Clone)]
pub struct DiceCoefficient {
    /// Threshold for converting predictions to binary
    pub threshold: f64,
    /// Small epsilon to avoid division by zero
    pub epsilon: f64,
}

impl Default for DiceCoefficient {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            epsilon: 1e-7,
        }
    }
}

impl Metric for DiceCoefficient {
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

        let mut intersection = 0.0;
        let mut pred_sum = 0.0;
        let mut target_sum = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = if predictions[[i, j]] >= self.threshold {
                    1.0
                } else {
                    0.0
                };
                let target = targets[[i, j]];

                intersection += pred * target;
                pred_sum += pred;
                target_sum += target;
            }
        }

        Ok((2.0 * intersection) / (pred_sum + target_sum + self.epsilon))
    }

    fn name(&self) -> &str {
        "dice_coefficient"
    }
}

/// Mean Average Precision (mAP) metric for object detection and retrieval.
///
/// Computes the average precision (AP) for each class and returns the mean.
/// This is a simplified version for multi-label classification scenarios.
///
/// For true object detection mAP with IoU thresholds, use specialized computer vision libraries.
#[derive(Debug, Clone)]
pub struct MeanAveragePrecision {
    /// Number of recall points to sample for AP calculation
    pub num_recall_points: usize,
}

impl Default for MeanAveragePrecision {
    fn default() -> Self {
        Self {
            num_recall_points: 11, // Standard 11-point interpolation
        }
    }
}

impl MeanAveragePrecision {
    /// Create with custom number of recall points.
    pub fn new(num_recall_points: usize) -> Self {
        Self { num_recall_points }
    }

    /// Compute Average Precision for a single class.
    fn compute_ap(&self, predictions: &[f64], targets: &[bool]) -> f64 {
        if predictions.is_empty() || targets.is_empty() {
            return 0.0;
        }

        // Sort by prediction scores (descending)
        let mut paired: Vec<(f64, bool)> = predictions
            .iter()
            .copied()
            .zip(targets.iter().copied())
            .collect();
        paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_positives = targets.iter().filter(|&&t| t).count() as f64;
        if total_positives == 0.0 {
            return 0.0;
        }

        let mut true_positives = 0.0;
        let mut false_positives = 0.0;
        let mut precisions = Vec::new();
        let mut recalls = Vec::new();

        for (_, target) in paired {
            if target {
                true_positives += 1.0;
            } else {
                false_positives += 1.0;
            }

            let precision = true_positives / (true_positives + false_positives);
            let recall = true_positives / total_positives;

            precisions.push(precision);
            recalls.push(recall);
        }

        // Interpolate precision at standard recall levels
        let mut ap = 0.0;
        for i in 0..self.num_recall_points {
            let recall_level = i as f64 / (self.num_recall_points - 1) as f64;

            // Find max precision at recall >= recall_level
            let max_precision = recalls
                .iter()
                .enumerate()
                .filter(|(_, &r)| r >= recall_level)
                .map(|(i, _)| precisions[i])
                .fold(0.0, f64::max);

            ap += max_precision;
        }

        ap / self.num_recall_points as f64
    }
}

impl Metric for MeanAveragePrecision {
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

        let num_classes = predictions.ncols();
        let mut aps = Vec::new();

        // Compute AP for each class
        for class_idx in 0..num_classes {
            let mut class_preds = Vec::new();
            let mut class_targets = Vec::new();

            for i in 0..predictions.nrows() {
                class_preds.push(predictions[[i, class_idx]]);
                class_targets.push(targets[[i, class_idx]] > 0.5);
            }

            let ap = self.compute_ap(&class_preds, &class_targets);
            aps.push(ap);
        }

        if aps.is_empty() {
            return Ok(0.0);
        }

        Ok(aps.iter().sum::<f64>() / aps.len() as f64)
    }

    fn name(&self) -> &str {
        "mean_average_precision"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_iou() {
        let metric = IoU::default();

        // Perfect overlap
        let predictions = array![[0.9, 0.1], [0.8, 0.2], [0.9, 0.1]];
        let targets = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];

        let iou = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((iou - 1.0).abs() < 1e-6);

        // Partial overlap
        let predictions = array![[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]];
        let targets = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];

        let iou = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((0.0..=1.0).contains(&iou));
        assert!(iou < 1.0);
    }

    #[test]
    fn test_mean_iou() {
        let metric = MeanIoU::default();

        // Perfect multi-class segmentation
        let predictions = array![[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]];
        let targets = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let miou = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((miou - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dice_coefficient() {
        let metric = DiceCoefficient::default();

        // Perfect overlap
        let predictions = array![[0.9, 0.1], [0.8, 0.2], [0.9, 0.1]];
        let targets = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];

        let dice = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((dice - 1.0).abs() < 1e-6);

        // No overlap
        let predictions = array![[0.1, 0.9], [0.2, 0.8]];
        let targets = array![[1.0, 0.0], [1.0, 0.0]];

        let dice = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!(dice < 0.1);
    }

    #[test]
    fn test_mean_average_precision() {
        let metric = MeanAveragePrecision::default();

        // Perfect ranking
        let predictions = array![[0.9, 0.8], [0.8, 0.7], [0.3, 0.2], [0.2, 0.1]];
        let targets = array![[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        let map = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((map - 1.0).abs() < 1e-6);

        // Random ranking
        let predictions = array![[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

        let map = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((0.0..=1.0).contains(&map));
    }

    #[test]
    fn test_iou_custom_threshold() {
        let metric = IoU::new(0.7);

        let predictions = array![[0.8, 0.2], [0.6, 0.4]]; // Second one below threshold
        let targets = array![[1.0, 0.0], [1.0, 0.0]];

        let iou = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((0.0..=1.0).contains(&iou));
        assert!(iou < 1.0); // Should be less than 1 due to threshold
    }

    #[test]
    fn test_mean_average_precision_custom_points() {
        let metric = MeanAveragePrecision::new(5); // 5-point interpolation

        let predictions = array![[0.9], [0.8], [0.3], [0.2]];
        let targets = array![[1.0], [1.0], [0.0], [0.0]];

        let map = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((0.0..=1.0).contains(&map));
    }
}
