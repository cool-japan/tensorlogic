//! Metrics for evaluating model performance.

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{ArrayView, Ix2};
use std::collections::HashMap;

/// Trait for metrics.
pub trait Metric {
    /// Compute metric value.
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64>;

    /// Get metric name.
    fn name(&self) -> &str;

    /// Reset metric state (for stateful metrics).
    fn reset(&mut self) {}
}

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

/// Metric tracker for managing multiple metrics.
pub struct MetricTracker {
    /// Metrics to track.
    metrics: Vec<Box<dyn Metric>>,
    /// History of metric values.
    history: HashMap<String, Vec<f64>>,
}

impl MetricTracker {
    /// Create a new metric tracker.
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            history: HashMap::new(),
        }
    }

    /// Add a metric to track.
    pub fn add(&mut self, metric: Box<dyn Metric>) {
        let name = metric.name().to_string();
        self.history.insert(name, Vec::new());
        self.metrics.push(metric);
    }

    /// Compute all metrics.
    pub fn compute_all(
        &mut self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for metric in &self.metrics {
            let value = metric.compute(predictions, targets)?;
            let name = metric.name().to_string();

            results.insert(name.clone(), value);

            if let Some(history) = self.history.get_mut(&name) {
                history.push(value);
            }
        }

        Ok(results)
    }

    /// Get history for a specific metric.
    pub fn get_history(&self, metric_name: &str) -> Option<&Vec<f64>> {
        self.history.get(metric_name)
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        for metric in &mut self.metrics {
            metric.reset();
        }
    }

    /// Clear history.
    pub fn clear_history(&mut self) {
        for history in self.history.values_mut() {
            history.clear();
        }
    }
}

impl Default for MetricTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Confusion matrix for multi-class classification.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Number of classes.
    num_classes: usize,
    /// Confusion matrix (rows=true labels, cols=predicted labels).
    matrix: Vec<Vec<usize>>,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix.
    ///
    /// # Arguments
    /// * `num_classes` - Number of classes
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            matrix: vec![vec![0; num_classes]; num_classes],
        }
    }

    /// Compute confusion matrix from predictions and targets.
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (one-hot or class probabilities)
    /// * `targets` - True labels (one-hot encoded)
    ///
    /// # Returns
    /// Confusion matrix
    pub fn compute(
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Self> {
        if predictions.shape() != targets.shape() {
            return Err(TrainError::MetricsError(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let num_classes = predictions.ncols();
        let mut matrix = vec![vec![0; num_classes]; num_classes];

        for i in 0..predictions.nrows() {
            // Find predicted class (argmax)
            let mut pred_class = 0;
            let mut max_pred = predictions[[i, 0]];
            for j in 1..num_classes {
                if predictions[[i, j]] > max_pred {
                    max_pred = predictions[[i, j]];
                    pred_class = j;
                }
            }

            // Find true class (argmax)
            let mut true_class = 0;
            let mut max_true = targets[[i, 0]];
            for j in 1..num_classes {
                if targets[[i, j]] > max_true {
                    max_true = targets[[i, j]];
                    true_class = j;
                }
            }

            matrix[true_class][pred_class] += 1;
        }

        Ok(Self {
            num_classes,
            matrix,
        })
    }

    /// Get the confusion matrix.
    pub fn matrix(&self) -> &Vec<Vec<usize>> {
        &self.matrix
    }

    /// Get value at (true_class, pred_class).
    pub fn get(&self, true_class: usize, pred_class: usize) -> usize {
        self.matrix[true_class][pred_class]
    }

    /// Compute per-class precision.
    pub fn precision_per_class(&self) -> Vec<f64> {
        let mut precisions = Vec::with_capacity(self.num_classes);

        for pred_class in 0..self.num_classes {
            let mut predicted_positive = 0;
            let mut true_positive = 0;

            for true_class in 0..self.num_classes {
                predicted_positive += self.matrix[true_class][pred_class];
                if true_class == pred_class {
                    true_positive += self.matrix[true_class][pred_class];
                }
            }

            let precision = if predicted_positive == 0 {
                0.0
            } else {
                true_positive as f64 / predicted_positive as f64
            };
            precisions.push(precision);
        }

        precisions
    }

    /// Compute per-class recall.
    pub fn recall_per_class(&self) -> Vec<f64> {
        let mut recalls = Vec::with_capacity(self.num_classes);

        for true_class in 0..self.num_classes {
            let mut actual_positive = 0;
            let mut true_positive = 0;

            for pred_class in 0..self.num_classes {
                actual_positive += self.matrix[true_class][pred_class];
                if true_class == pred_class {
                    true_positive += self.matrix[true_class][pred_class];
                }
            }

            let recall = if actual_positive == 0 {
                0.0
            } else {
                true_positive as f64 / actual_positive as f64
            };
            recalls.push(recall);
        }

        recalls
    }

    /// Compute per-class F1 scores.
    pub fn f1_per_class(&self) -> Vec<f64> {
        let precisions = self.precision_per_class();
        let recalls = self.recall_per_class();

        precisions
            .iter()
            .zip(recalls.iter())
            .map(|(p, r)| {
                if p + r == 0.0 {
                    0.0
                } else {
                    2.0 * p * r / (p + r)
                }
            })
            .collect()
    }

    /// Compute overall accuracy.
    pub fn accuracy(&self) -> f64 {
        let mut correct = 0;
        let mut total = 0;

        for i in 0..self.num_classes {
            for j in 0..self.num_classes {
                total += self.matrix[i][j];
                if i == j {
                    correct += self.matrix[i][j];
                }
            }
        }

        if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        }
    }

    /// Get total number of predictions.
    pub fn total_predictions(&self) -> usize {
        let mut total = 0;
        for i in 0..self.num_classes {
            for j in 0..self.num_classes {
                total += self.matrix[i][j];
            }
        }
        total
    }
}

impl std::fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Confusion Matrix:")?;
        write!(f, "     ")?;

        for j in 0..self.num_classes {
            write!(f, "{:5}", j)?;
        }
        writeln!(f)?;

        for i in 0..self.num_classes {
            write!(f, "{:3}| ", i)?;
            for j in 0..self.num_classes {
                write!(f, "{:5}", self.matrix[i][j])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/// ROC curve and AUC computation utilities.
#[derive(Debug, Clone)]
pub struct RocCurve {
    /// False positive rates.
    pub fpr: Vec<f64>,
    /// True positive rates.
    pub tpr: Vec<f64>,
    /// Thresholds.
    pub thresholds: Vec<f64>,
}

impl RocCurve {
    /// Compute ROC curve for binary classification.
    ///
    /// # Arguments
    /// * `predictions` - Predicted probabilities for positive class
    /// * `targets` - True binary labels (0 or 1)
    ///
    /// # Returns
    /// ROC curve with FPR, TPR, and thresholds
    pub fn compute(predictions: &[f64], targets: &[bool]) -> TrainResult<Self> {
        if predictions.len() != targets.len() {
            return Err(TrainError::MetricsError(format!(
                "Length mismatch: predictions {} vs targets {}",
                predictions.len(),
                targets.len()
            )));
        }

        // Create sorted indices by prediction score (descending)
        let mut indices: Vec<usize> = (0..predictions.len()).collect();
        indices.sort_by(|&a, &b| {
            predictions[b]
                .partial_cmp(&predictions[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut fpr = Vec::new();
        let mut tpr = Vec::new();
        let mut thresholds = Vec::new();

        let num_positive = targets.iter().filter(|&&x| x).count();
        let num_negative = targets.len() - num_positive;

        let mut true_positives = 0;
        let mut false_positives = 0;

        // Start with all predictions as negative
        fpr.push(0.0);
        tpr.push(0.0);
        thresholds.push(f64::INFINITY);

        for &idx in &indices {
            if targets[idx] {
                true_positives += 1;
            } else {
                false_positives += 1;
            }

            let fpr_val = if num_negative == 0 {
                0.0
            } else {
                false_positives as f64 / num_negative as f64
            };
            let tpr_val = if num_positive == 0 {
                0.0
            } else {
                true_positives as f64 / num_positive as f64
            };

            fpr.push(fpr_val);
            tpr.push(tpr_val);
            thresholds.push(predictions[idx]);
        }

        Ok(Self {
            fpr,
            tpr,
            thresholds,
        })
    }

    /// Compute area under the ROC curve (AUC) using trapezoidal rule.
    pub fn auc(&self) -> f64 {
        let mut auc = 0.0;

        for i in 1..self.fpr.len() {
            let width = self.fpr[i] - self.fpr[i - 1];
            let height = (self.tpr[i] + self.tpr[i - 1]) / 2.0;
            auc += width * height;
        }

        auc
    }
}

/// Per-class metrics report.
#[derive(Debug, Clone)]
pub struct PerClassMetrics {
    /// Precision per class.
    pub precision: Vec<f64>,
    /// Recall per class.
    pub recall: Vec<f64>,
    /// F1 score per class.
    pub f1_score: Vec<f64>,
    /// Support (number of samples) per class.
    pub support: Vec<usize>,
}

impl PerClassMetrics {
    /// Compute per-class metrics from predictions and targets.
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (one-hot or class probabilities)
    /// * `targets` - True labels (one-hot encoded)
    ///
    /// # Returns
    /// Per-class metrics report
    pub fn compute(
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Self> {
        let confusion_matrix = ConfusionMatrix::compute(predictions, targets)?;

        let precision = confusion_matrix.precision_per_class();
        let recall = confusion_matrix.recall_per_class();
        let f1_score = confusion_matrix.f1_per_class();

        // Compute support (number of samples per class)
        let num_classes = targets.ncols();
        let mut support = vec![0; num_classes];

        for i in 0..targets.nrows() {
            // Find true class
            let mut true_class = 0;
            let mut max_true = targets[[i, 0]];
            for j in 1..num_classes {
                if targets[[i, j]] > max_true {
                    max_true = targets[[i, j]];
                    true_class = j;
                }
            }
            support[true_class] += 1;
        }

        Ok(Self {
            precision,
            recall,
            f1_score,
            support,
        })
    }
}

impl std::fmt::Display for PerClassMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Per-Class Metrics:")?;
        writeln!(f, "Class  Precision  Recall  F1-Score  Support")?;
        writeln!(f, "-----  ---------  ------  --------  -------")?;

        for i in 0..self.precision.len() {
            writeln!(
                f,
                "{:5}  {:9.4}  {:6.4}  {:8.4}  {:7}",
                i, self.precision[i], self.recall[i], self.f1_score[i], self.support[i]
            )?;
        }

        // Compute macro averages
        let macro_precision: f64 = self.precision.iter().sum::<f64>() / self.precision.len() as f64;
        let macro_recall: f64 = self.recall.iter().sum::<f64>() / self.recall.len() as f64;
        let macro_f1: f64 = self.f1_score.iter().sum::<f64>() / self.f1_score.len() as f64;
        let total_support: usize = self.support.iter().sum();

        writeln!(f, "-----  ---------  ------  --------  -------")?;
        writeln!(
            f,
            "Macro  {:9.4}  {:6.4}  {:8.4}  {:7}",
            macro_precision, macro_recall, macro_f1, total_support
        )?;

        Ok(())
    }
}

/// Matthews Correlation Coefficient (MCC) metric.
/// Ranges from -1 to +1, where +1 is perfect prediction, 0 is random, -1 is total disagreement.
/// Particularly useful for imbalanced datasets.
#[derive(Debug, Clone, Default)]
pub struct MatthewsCorrelationCoefficient;

impl Metric for MatthewsCorrelationCoefficient {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        let confusion_matrix = ConfusionMatrix::compute(predictions, targets)?;
        let num_classes = confusion_matrix.num_classes;

        // For binary classification, use the standard MCC formula
        if num_classes == 2 {
            let tp = confusion_matrix.matrix[1][1] as f64;
            let tn = confusion_matrix.matrix[0][0] as f64;
            let fp = confusion_matrix.matrix[0][1] as f64;
            let fn_val = confusion_matrix.matrix[1][0] as f64;

            let numerator = (tp * tn) - (fp * fn_val);
            let denominator = ((tp + fp) * (tp + fn_val) * (tn + fp) * (tn + fn_val)).sqrt();

            if denominator == 0.0 {
                Ok(0.0)
            } else {
                Ok(numerator / denominator)
            }
        } else {
            // Multi-class MCC formula
            let mut s = 0.0;
            let mut c = 0.0;
            let t = confusion_matrix.total_predictions() as f64;

            // Compute column sums
            let mut p_k = vec![0.0; num_classes];
            let mut t_k = vec![0.0; num_classes];

            for k in 0..num_classes {
                for l in 0..num_classes {
                    p_k[k] += confusion_matrix.matrix[l][k] as f64;
                    t_k[k] += confusion_matrix.matrix[k][l] as f64;
                }
            }

            // Compute trace (correct predictions)
            for k in 0..num_classes {
                c += confusion_matrix.matrix[k][k] as f64;
            }

            // Compute sum of products
            for k in 0..num_classes {
                s += p_k[k] * t_k[k];
            }

            let numerator = (t * c) - s;
            let denominator_1 = ((t * t) - s).sqrt();
            let mut sum_p_sq = 0.0;
            let mut sum_t_sq = 0.0;
            for k in 0..num_classes {
                sum_p_sq += p_k[k] * p_k[k];
                sum_t_sq += t_k[k] * t_k[k];
            }
            let denominator_2 = ((t * t) - sum_p_sq).sqrt();
            let denominator_3 = ((t * t) - sum_t_sq).sqrt();

            let denominator = denominator_1 * denominator_2 * denominator_3;

            if denominator == 0.0 {
                Ok(0.0)
            } else {
                Ok(numerator / denominator)
            }
        }
    }

    fn name(&self) -> &str {
        "mcc"
    }
}

/// Cohen's Kappa statistic.
/// Measures inter-rater agreement, accounting for chance agreement.
/// Ranges from -1 to +1, where 1 is perfect agreement, 0 is random chance.
#[derive(Debug, Clone, Default)]
pub struct CohensKappa;

impl Metric for CohensKappa {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        let confusion_matrix = ConfusionMatrix::compute(predictions, targets)?;
        let num_classes = confusion_matrix.num_classes;
        let total = confusion_matrix.total_predictions() as f64;

        // Observed agreement (accuracy)
        let mut observed = 0.0;
        for i in 0..num_classes {
            observed += confusion_matrix.matrix[i][i] as f64;
        }
        observed /= total;

        // Expected agreement by chance
        let mut expected = 0.0;
        for i in 0..num_classes {
            let row_sum: f64 = (0..num_classes)
                .map(|j| confusion_matrix.matrix[i][j] as f64)
                .sum();
            let col_sum: f64 = (0..num_classes)
                .map(|j| confusion_matrix.matrix[j][i] as f64)
                .sum();
            expected += (row_sum / total) * (col_sum / total);
        }

        if expected >= 1.0 {
            Ok(0.0)
        } else {
            Ok((observed - expected) / (1.0 - expected))
        }
    }

    fn name(&self) -> &str {
        "cohens_kappa"
    }
}

/// Top-K accuracy metric.
/// Measures whether the correct class is in the top K predictions.
#[derive(Debug, Clone)]
pub struct TopKAccuracy {
    /// Number of top predictions to consider.
    pub k: usize,
}

impl Default for TopKAccuracy {
    fn default() -> Self {
        Self { k: 5 }
    }
}

impl TopKAccuracy {
    /// Create a new Top-K accuracy metric.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Metric for TopKAccuracy {
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
        if self.k > num_classes {
            return Err(TrainError::MetricsError(format!(
                "K ({}) cannot be greater than number of classes ({})",
                self.k, num_classes
            )));
        }

        let mut correct = 0;
        let total = predictions.nrows();

        for i in 0..total {
            // Find true class
            let mut true_class = 0;
            let mut max_true = targets[[i, 0]];
            for j in 1..num_classes {
                if targets[[i, j]] > max_true {
                    max_true = targets[[i, j]];
                    true_class = j;
                }
            }

            // Get top K predictions
            let mut indices: Vec<usize> = (0..num_classes).collect();
            indices.sort_by(|&a, &b| {
                predictions[[i, b]]
                    .partial_cmp(&predictions[[i, a]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Check if true class is in top K
            if indices[..self.k].contains(&true_class) {
                correct += 1;
            }
        }

        Ok(correct as f64 / total as f64)
    }

    fn name(&self) -> &str {
        "top_k_accuracy"
    }
}

/// Balanced accuracy metric.
/// Average of recall per class, useful for imbalanced datasets.
#[derive(Debug, Clone, Default)]
pub struct BalancedAccuracy;

impl Metric for BalancedAccuracy {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        let confusion_matrix = ConfusionMatrix::compute(predictions, targets)?;
        let recalls = confusion_matrix.recall_per_class();

        // Balanced accuracy is the average of recall per class
        let sum: f64 = recalls.iter().sum();
        Ok(sum / recalls.len() as f64)
    }

    fn name(&self) -> &str {
        "balanced_accuracy"
    }
}

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

/// Normalized Discounted Cumulative Gain (NDCG) metric for ranking.
///
/// NDCG measures the quality of ranking by comparing the predicted order
/// with the ideal order. It accounts for position: items ranked higher
/// contribute more to the score.
///
/// # Formula
/// DCG@k = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1)
/// NDCG@k = DCG@k / IDCG@k
///
/// where IDCG is the DCG of the ideal ranking.
///
/// # Use Cases
/// - Recommendation systems
/// - Search engine ranking
/// - Information retrieval
/// - Learning to rank
///
/// Reference: Järvelin & Kekäläinen "Cumulated gain-based evaluation of IR techniques" (ACM TOIS 2002)
#[derive(Debug, Clone)]
pub struct NormalizedDiscountedCumulativeGain {
    /// Number of top results to consider (k).
    pub k: usize,
}

impl Default for NormalizedDiscountedCumulativeGain {
    fn default() -> Self {
        Self { k: 10 }
    }
}

impl NormalizedDiscountedCumulativeGain {
    /// Create NDCG metric with custom k value.
    ///
    /// # Arguments
    /// * `k` - Number of top results to consider
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compute DCG (Discounted Cumulative Gain) for a single ranking.
    ///
    /// # Arguments
    /// * `relevances` - Relevance scores in the predicted order
    /// * `k` - Number of positions to consider
    fn compute_dcg(relevances: &[f64], k: usize) -> f64 {
        let k = k.min(relevances.len());
        let mut dcg = 0.0;

        for (i, &rel) in relevances.iter().take(k).enumerate() {
            let position = (i + 2) as f64; // i+2 because positions start at 1 and log₂(1) = 0
            dcg += (2.0_f64.powf(rel) - 1.0) / position.log2();
        }

        dcg
    }

    /// Compute IDCG (Ideal DCG) by sorting relevances in descending order.
    fn compute_idcg(relevances: &[f64], k: usize) -> f64 {
        let mut sorted_rel = relevances.to_vec();
        sorted_rel.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Self::compute_dcg(&sorted_rel, k)
    }
}

impl Metric for NormalizedDiscountedCumulativeGain {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.shape() != targets.shape() {
            return Err(crate::TrainError::MetricsError(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let n_samples = predictions.nrows();
        if n_samples == 0 {
            return Ok(0.0);
        }

        let mut ndcg_sum = 0.0;

        for i in 0..n_samples {
            // Get predicted scores and true relevances for this sample
            let pred_scores: Vec<f64> = predictions.row(i).iter().copied().collect();
            let true_relevances: Vec<f64> = targets.row(i).iter().copied().collect();

            // Create indices and sort by predicted scores (descending)
            let mut indices: Vec<usize> = (0..pred_scores.len()).collect();
            indices.sort_by(|&a, &b| {
                pred_scores[b]
                    .partial_cmp(&pred_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Reorder relevances according to predicted ranking
            let ranked_relevances: Vec<f64> =
                indices.iter().map(|&idx| true_relevances[idx]).collect();

            // Compute DCG for this ranking
            let dcg = Self::compute_dcg(&ranked_relevances, self.k);

            // Compute IDCG (ideal ranking)
            let idcg = Self::compute_idcg(&true_relevances, self.k);

            // Compute NDCG (handle division by zero)
            let ndcg = if idcg > 1e-12 { dcg / idcg } else { 0.0 };

            ndcg_sum += ndcg;
        }

        Ok(ndcg_sum / n_samples as f64)
    }

    fn name(&self) -> &str {
        "ndcg"
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

    #[test]
    fn test_metric_tracker() {
        let mut tracker = MetricTracker::new();
        tracker.add(Box::new(Accuracy::default()));
        tracker.add(Box::new(F1Score::default()));

        let predictions = array![[0.9, 0.1], [0.2, 0.8]];
        let targets = array![[1.0, 0.0], [0.0, 1.0]];

        let results = tracker
            .compute_all(&predictions.view(), &targets.view())
            .unwrap();
        assert!(results.contains_key("accuracy"));
        assert!(results.contains_key("f1_score"));

        let history = tracker.get_history("accuracy").unwrap();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = array![
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.8, 0.1, 0.1]
        ];
        let targets = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ];

        let cm = ConfusionMatrix::compute(&predictions.view(), &targets.view()).unwrap();

        assert_eq!(cm.get(0, 0), 2); // Class 0 correctly predicted
        assert_eq!(cm.get(1, 1), 1); // Class 1 correctly predicted
        assert_eq!(cm.get(2, 2), 1); // Class 2 correctly predicted
        assert_eq!(cm.accuracy(), 1.0);
    }

    #[test]
    fn test_confusion_matrix_per_class_metrics() {
        let predictions = array![[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

        let cm = ConfusionMatrix::compute(&predictions.view(), &targets.view()).unwrap();

        let precision = cm.precision_per_class();
        let recall = cm.recall_per_class();
        let f1 = cm.f1_per_class();

        assert_eq!(precision.len(), 2);
        assert_eq!(recall.len(), 2);
        assert_eq!(f1.len(), 2);

        // All predictions correct
        assert_eq!(precision[0], 1.0);
        assert_eq!(precision[1], 1.0);
        assert_eq!(recall[0], 1.0);
        assert_eq!(recall[1], 1.0);
    }

    #[test]
    fn test_roc_curve() {
        let predictions = vec![0.9, 0.8, 0.4, 0.3, 0.1];
        let targets = vec![true, true, false, true, false];

        let roc = RocCurve::compute(&predictions, &targets).unwrap();

        assert!(!roc.fpr.is_empty());
        assert!(!roc.tpr.is_empty());
        assert!(!roc.thresholds.is_empty());
        assert_eq!(roc.fpr.len(), roc.tpr.len());

        let auc = roc.auc();
        assert!((0.0..=1.0).contains(&auc));
    }

    #[test]
    fn test_roc_auc_perfect() {
        let predictions = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![true, true, false, false];

        let roc = RocCurve::compute(&predictions, &targets).unwrap();
        let auc = roc.auc();

        // Perfect classification should have AUC = 1.0
        assert!((auc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_per_class_metrics() {
        let predictions = array![
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.8, 0.1, 0.1]
        ];
        let targets = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ];

        let metrics = PerClassMetrics::compute(&predictions.view(), &targets.view()).unwrap();

        assert_eq!(metrics.precision.len(), 3);
        assert_eq!(metrics.recall.len(), 3);
        assert_eq!(metrics.f1_score.len(), 3);
        assert_eq!(metrics.support.len(), 3);

        // Check support counts
        assert_eq!(metrics.support[0], 2);
        assert_eq!(metrics.support[1], 1);
        assert_eq!(metrics.support[2], 1);
    }

    #[test]
    fn test_matthews_correlation_coefficient() {
        let metric = MatthewsCorrelationCoefficient;

        // Perfect binary classification
        let predictions = array![[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

        let mcc = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((mcc - 1.0).abs() < 1e-6);

        // Random classification
        let predictions = array![[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

        let mcc = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!(mcc.abs() < 0.1);
    }

    #[test]
    fn test_cohens_kappa() {
        let metric = CohensKappa;

        // Perfect agreement
        let predictions = array![[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

        let kappa = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((kappa - 1.0).abs() < 1e-6);

        // Random agreement
        let predictions = array![[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

        let kappa = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((-1.0..=1.0).contains(&kappa));
    }

    #[test]
    fn test_top_k_accuracy() {
        let metric = TopKAccuracy::new(2);

        // Test with 3 classes
        let predictions = array![
            [0.7, 0.2, 0.1], // Correct class is 0, top-2 includes it
            [0.1, 0.6, 0.3], // Correct class is 1, top-2 includes it
            [0.3, 0.4, 0.3], // Correct class is 2, top-2 includes it (1, 0)
        ];
        let targets = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let top_k = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((0.0..=1.0).contains(&top_k));
        assert!(top_k >= 0.66); // At least 2/3 should be in top-2
    }

    #[test]
    fn test_balanced_accuracy() {
        let metric = BalancedAccuracy;

        // Perfect classification
        let predictions = array![[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]];
        let targets = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];

        let balanced_acc = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((balanced_acc - 1.0).abs() < 1e-6);

        // Imbalanced but perfect
        let predictions = array![[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9]];
        let targets = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        let balanced_acc = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((balanced_acc - 1.0).abs() < 1e-6);
    }

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

    #[test]
    fn test_ndcg_perfect_ranking() {
        let metric = NormalizedDiscountedCumulativeGain::new(5);

        // Perfect ranking: predicted order matches true relevance order
        let predictions = array![
            [5.0, 4.0, 3.0, 2.0, 1.0], // Pred scores: highest to lowest
        ];
        let targets = array![
            [5.0, 4.0, 3.0, 2.0, 1.0], // True relevances: match pred order
        ];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Perfect ranking should give NDCG = 1.0
        assert!(
            (ndcg - 1.0).abs() < 1e-6,
            "Perfect ranking should have NDCG ≈ 1.0, got {}",
            ndcg
        );
    }

    #[test]
    fn test_ndcg_worst_ranking() {
        let metric = NormalizedDiscountedCumulativeGain::new(5);

        // Worst ranking: predicted order is reverse of true relevance
        let predictions = array![
            [1.0, 2.0, 3.0, 4.0, 5.0], // Pred scores: lowest to highest
        ];
        let targets = array![
            [5.0, 4.0, 3.0, 2.0, 1.0], // True relevances: highest to lowest
        ];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Worst ranking should give low NDCG
        assert!(
            ndcg < 0.8,
            "Worst ranking should have low NDCG, got {}",
            ndcg
        );
    }

    #[test]
    fn test_ndcg_partial_match() {
        let metric = NormalizedDiscountedCumulativeGain::new(3);

        // Partial match: some items ranked correctly
        let predictions = array![
            [4.0, 5.0, 2.0, 3.0, 1.0], // Pred order: [1, 0, 3, 2, 4]
        ];
        let targets = array![
            [3.0, 5.0, 1.0, 2.0, 0.0], // True relevances
        ];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Should be between 0 and 1
        assert!(
            (0.0..=1.0).contains(&ndcg),
            "NDCG should be in [0, 1], got {}",
            ndcg
        );

        // Should be reasonably high since highest relevance (5.0) is predicted correctly
        assert!(
            ndcg > 0.7,
            "NDCG should be > 0.7 for this ranking, got {}",
            ndcg
        );
    }

    #[test]
    fn test_ndcg_multiple_samples() {
        let metric = NormalizedDiscountedCumulativeGain::new(3);

        // Two samples: one perfect, one reversed
        let predictions = array![[5.0, 4.0, 3.0, 2.0], [2.0, 3.0, 4.0, 5.0],];
        let targets = array![[5.0, 4.0, 3.0, 2.0], [5.0, 4.0, 3.0, 2.0],];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Average of perfect (1.0) and poor ranking
        assert!((0.0..=1.0).contains(&ndcg));
        assert!(ndcg > 0.4 && ndcg < 0.9); // Should be somewhere in between
    }

    #[test]
    fn test_ndcg_different_k_values() {
        let metric_k3 = NormalizedDiscountedCumulativeGain::new(3);
        let metric_k5 = NormalizedDiscountedCumulativeGain::new(5);

        let predictions = array![[5.0, 4.0, 3.0, 1.0, 2.0]];
        let targets = array![[5.0, 4.0, 3.0, 2.0, 1.0]];

        let ndcg_k3 = metric_k3
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        let ndcg_k5 = metric_k5
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // k=3 should be perfect (top 3 are correct)
        assert!((ndcg_k3 - 1.0).abs() < 1e-6);

        // k=5 should be lower (last 2 are swapped)
        assert!(ndcg_k5 < ndcg_k3);
        assert!(ndcg_k5 > 0.9); // Still very good
    }

    #[test]
    fn test_ndcg_zero_relevances() {
        let metric = NormalizedDiscountedCumulativeGain::new(5);

        // All zero relevances
        let predictions = array![[1.0, 2.0, 3.0]];
        let targets = array![[0.0, 0.0, 0.0]];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Should handle gracefully (IDCG = 0)
        assert!(ndcg.is_finite());
        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn test_ndcg_empty_input() {
        let metric = NormalizedDiscountedCumulativeGain::default();

        use scirs2_core::ndarray::Array;
        let empty_predictions: Array<f64, _> = Array::zeros((0, 5));
        let empty_targets: Array<f64, _> = Array::zeros((0, 5));

        let ndcg = metric
            .compute(&empty_predictions.view(), &empty_targets.view())
            .unwrap();

        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn test_ndcg_shape_mismatch() {
        let metric = NormalizedDiscountedCumulativeGain::default();

        let predictions = array![[1.0, 2.0, 3.0]];
        let targets = array![[1.0, 2.0]]; // Different shape

        let result = metric.compute(&predictions.view(), &targets.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_ndcg_binary_relevance() {
        let metric = NormalizedDiscountedCumulativeGain::new(5);

        // Binary relevance (0 or 1)
        let predictions = array![[0.9, 0.7, 0.5, 0.3, 0.1]];
        let targets = array![[1.0, 1.0, 0.0, 1.0, 0.0]];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Should be in valid range
        assert!((0.0..=1.0).contains(&ndcg));

        // Top 2 are relevant, so should have decent NDCG
        assert!(
            ndcg > 0.6,
            "Should have decent NDCG with top-2 relevant, got {}",
            ndcg
        );
    }
}
