//! Basic classification metrics.

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{ArrayView, Ix2};

use super::Metric;

/// Accuracy metric for classification.
#[derive(Debug, Clone)]
pub struct Accuracy {
    /// Threshold for binary classification.
    pub threshold: f64,
}

impl Default for Accuracy {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl Metric for Accuracy {
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

        let mut correct = 0;
        let total = predictions.nrows();

        for i in 0..total {
            // Find predicted class (argmax)
            let mut pred_class = 0;
            let mut max_pred = predictions[[i, 0]];
            for j in 1..predictions.ncols() {
                if predictions[[i, j]] > max_pred {
                    max_pred = predictions[[i, j]];
                    pred_class = j;
                }
            }

            // Find true class (argmax)
            let mut true_class = 0;
            let mut max_true = targets[[i, 0]];
            for j in 1..targets.ncols() {
                if targets[[i, j]] > max_true {
                    max_true = targets[[i, j]];
                    true_class = j;
                }
            }

            if pred_class == true_class {
                correct += 1;
            }
        }

        Ok(correct as f64 / total as f64)
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}

/// Precision metric for classification.
#[derive(Debug, Clone, Default)]
pub struct Precision {
    /// Class to compute precision for (None = macro average).
    pub class_id: Option<usize>,
}

impl Metric for Precision {
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
        let mut true_positives = vec![0; num_classes];
        let mut predicted_positives = vec![0; num_classes];

        for i in 0..predictions.nrows() {
            // Find predicted class
            let mut pred_class = 0;
            let mut max_pred = predictions[[i, 0]];
            for j in 1..num_classes {
                if predictions[[i, j]] > max_pred {
                    max_pred = predictions[[i, j]];
                    pred_class = j;
                }
            }

            // Find true class
            let mut true_class = 0;
            let mut max_true = targets[[i, 0]];
            for j in 1..num_classes {
                if targets[[i, j]] > max_true {
                    max_true = targets[[i, j]];
                    true_class = j;
                }
            }

            predicted_positives[pred_class] += 1;
            if pred_class == true_class {
                true_positives[pred_class] += 1;
            }
        }

        if let Some(class_id) = self.class_id {
            // Precision for specific class
            if predicted_positives[class_id] == 0 {
                Ok(0.0)
            } else {
                Ok(true_positives[class_id] as f64 / predicted_positives[class_id] as f64)
            }
        } else {
            // Macro-averaged precision
            let mut total_precision = 0.0;
            let mut valid_classes = 0;

            for class_id in 0..num_classes {
                if predicted_positives[class_id] > 0 {
                    total_precision +=
                        true_positives[class_id] as f64 / predicted_positives[class_id] as f64;
                    valid_classes += 1;
                }
            }

            if valid_classes == 0 {
                Ok(0.0)
            } else {
                Ok(total_precision / valid_classes as f64)
            }
        }
    }

    fn name(&self) -> &str {
        "precision"
    }
}

/// Recall metric for classification.
#[derive(Debug, Clone, Default)]
pub struct Recall {
    /// Class to compute recall for (None = macro average).
    pub class_id: Option<usize>,
}

impl Metric for Recall {
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
        let mut true_positives = vec![0; num_classes];
        let mut actual_positives = vec![0; num_classes];

        for i in 0..predictions.nrows() {
            // Find predicted class
            let mut pred_class = 0;
            let mut max_pred = predictions[[i, 0]];
            for j in 1..num_classes {
                if predictions[[i, j]] > max_pred {
                    max_pred = predictions[[i, j]];
                    pred_class = j;
                }
            }

            // Find true class
            let mut true_class = 0;
            let mut max_true = targets[[i, 0]];
            for j in 1..num_classes {
                if targets[[i, j]] > max_true {
                    max_true = targets[[i, j]];
                    true_class = j;
                }
            }

            actual_positives[true_class] += 1;
            if pred_class == true_class {
                true_positives[pred_class] += 1;
            }
        }

        if let Some(class_id) = self.class_id {
            // Recall for specific class
            if actual_positives[class_id] == 0 {
                Ok(0.0)
            } else {
                Ok(true_positives[class_id] as f64 / actual_positives[class_id] as f64)
            }
        } else {
            // Macro-averaged recall
            let mut total_recall = 0.0;
            let mut valid_classes = 0;

            for class_id in 0..num_classes {
                if actual_positives[class_id] > 0 {
                    total_recall +=
                        true_positives[class_id] as f64 / actual_positives[class_id] as f64;
                    valid_classes += 1;
                }
            }

            if valid_classes == 0 {
                Ok(0.0)
            } else {
                Ok(total_recall / valid_classes as f64)
            }
        }
    }

    fn name(&self) -> &str {
        "recall"
    }
}

/// F1 score metric for classification.
#[derive(Debug, Clone, Default)]
pub struct F1Score {
    /// Class to compute F1 for (None = macro average).
    pub class_id: Option<usize>,
}

impl Metric for F1Score {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        let precision = Precision {
            class_id: self.class_id,
        }
        .compute(predictions, targets)?;
        let recall = Recall {
            class_id: self.class_id,
        }
        .compute(predictions, targets)?;

        if precision + recall == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * precision * recall / (precision + recall))
        }
    }

    fn name(&self) -> &str {
        "f1_score"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_accuracy() {
        let metric = Accuracy::default();

        // Perfect predictions
        let predictions = array![[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];

        let accuracy = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert_eq!(accuracy, 1.0);

        // Partial correct
        let predictions = array![[0.9, 0.1], [0.8, 0.2], [0.8, 0.2]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];

        let accuracy = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((accuracy - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_precision() {
        let metric = Precision::default();

        let predictions = array![[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];

        let precision = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((0.0..=1.0).contains(&precision));
    }

    #[test]
    fn test_recall() {
        let metric = Recall::default();

        let predictions = array![[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];

        let recall = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((0.0..=1.0).contains(&recall));
    }

    #[test]
    fn test_f1_score() {
        let metric = F1Score::default();

        let predictions = array![[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];

        let f1 = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((0.0..=1.0).contains(&f1));
    }
}
