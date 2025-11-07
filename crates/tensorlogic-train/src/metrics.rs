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
}
