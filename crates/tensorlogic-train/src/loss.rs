//! Loss functions for training.
//!
//! Provides both standard ML loss functions and logical constraint-based losses.

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};
use std::fmt::Debug;

/// Configuration for loss functions.
#[derive(Debug, Clone)]
pub struct LossConfig {
    /// Weight for supervised loss component.
    pub supervised_weight: f64,
    /// Weight for constraint violation loss component.
    pub constraint_weight: f64,
    /// Weight for rule satisfaction loss component.
    pub rule_weight: f64,
    /// Temperature for soft constraint penalties.
    pub temperature: f64,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            supervised_weight: 1.0,
            constraint_weight: 1.0,
            rule_weight: 1.0,
            temperature: 1.0,
        }
    }
}

/// Trait for loss functions.
pub trait Loss: Debug {
    /// Compute loss value.
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64>;

    /// Compute loss gradient with respect to predictions.
    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>>;

    /// Get the name of the loss function.
    fn name(&self) -> &str {
        "unknown"
    }
}

/// Cross-entropy loss for classification.
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    /// Epsilon for numerical stability.
    pub epsilon: f64,
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self { epsilon: 1e-10 }
    }
}

impl Loss for CrossEntropyLoss {
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

        let n = predictions.nrows() as f64;
        let mut total_loss = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]]
                    .max(self.epsilon)
                    .min(1.0 - self.epsilon);
                let target = targets[[i, j]];
                total_loss -= target * pred.ln();
            }
        }

        Ok(total_loss / n)
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

        let n = predictions.nrows() as f64;
        let mut grad = Array::zeros(predictions.raw_dim());

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]]
                    .max(self.epsilon)
                    .min(1.0 - self.epsilon);
                let target = targets[[i, j]];
                grad[[i, j]] = -(target / pred) / n;
            }
        }

        Ok(grad)
    }
}

/// Mean squared error loss for regression.
#[derive(Debug, Clone, Default)]
pub struct MseLoss;

impl Loss for MseLoss {
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

        let n = predictions.len() as f64;
        let mut total_loss = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let diff = predictions[[i, j]] - targets[[i, j]];
                total_loss += diff * diff;
            }
        }

        Ok(total_loss / n)
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

        let n = predictions.len() as f64;
        let mut grad = Array::zeros(predictions.raw_dim());

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                grad[[i, j]] = 2.0 * (predictions[[i, j]] - targets[[i, j]]) / n;
            }
        }

        Ok(grad)
    }
}

/// Logical loss combining multiple objectives.
#[derive(Debug)]
pub struct LogicalLoss {
    /// Configuration.
    pub config: LossConfig,
    /// Supervised loss component.
    pub supervised_loss: Box<dyn Loss>,
    /// Rule satisfaction components.
    pub rule_losses: Vec<Box<dyn Loss>>,
    /// Constraint violation components.
    pub constraint_losses: Vec<Box<dyn Loss>>,
}

impl LogicalLoss {
    /// Create a new logical loss.
    pub fn new(
        config: LossConfig,
        supervised_loss: Box<dyn Loss>,
        rule_losses: Vec<Box<dyn Loss>>,
        constraint_losses: Vec<Box<dyn Loss>>,
    ) -> Self {
        Self {
            config,
            supervised_loss,
            rule_losses,
            constraint_losses,
        }
    }

    /// Compute total loss with all components.
    pub fn compute_total(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
        rule_values: &[ArrayView<f64, Ix2>],
        constraint_values: &[ArrayView<f64, Ix2>],
    ) -> TrainResult<f64> {
        let mut total = 0.0;

        // Supervised loss
        let supervised = self.supervised_loss.compute(predictions, targets)?;
        total += self.config.supervised_weight * supervised;

        // Rule satisfaction losses
        if !rule_values.is_empty() && !self.rule_losses.is_empty() {
            let expected_true = Array::ones((rule_values[0].nrows(), rule_values[0].ncols()));
            let expected_true_view = expected_true.view();

            for (rule_val, rule_loss) in rule_values.iter().zip(self.rule_losses.iter()) {
                let rule_loss_val = rule_loss.compute(rule_val, &expected_true_view)?;
                total += self.config.rule_weight * rule_loss_val;
            }
        }

        // Constraint violation losses
        if !constraint_values.is_empty() && !self.constraint_losses.is_empty() {
            let expected_zero =
                Array::zeros((constraint_values[0].nrows(), constraint_values[0].ncols()));
            let expected_zero_view = expected_zero.view();

            for (constraint_val, constraint_loss) in
                constraint_values.iter().zip(self.constraint_losses.iter())
            {
                let constraint_loss_val =
                    constraint_loss.compute(constraint_val, &expected_zero_view)?;
                total += self.config.constraint_weight * constraint_loss_val;
            }
        }

        Ok(total)
    }
}

/// Rule satisfaction loss - measures how well rules are satisfied.
#[derive(Debug, Clone)]
pub struct RuleSatisfactionLoss {
    /// Temperature for soft satisfaction.
    pub temperature: f64,
}

impl Default for RuleSatisfactionLoss {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

impl Loss for RuleSatisfactionLoss {
    fn compute(
        &self,
        rule_values: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if rule_values.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: rule_values {:?} vs targets {:?}",
                rule_values.shape(),
                targets.shape()
            )));
        }

        let n = rule_values.len() as f64;
        let mut total_loss = 0.0;

        // Penalize deviations from expected rule satisfaction (typically 1.0)
        for i in 0..rule_values.nrows() {
            for j in 0..rule_values.ncols() {
                let diff = targets[[i, j]] - rule_values[[i, j]];
                // Soft penalty with temperature
                total_loss += (diff / self.temperature).powi(2);
            }
        }

        Ok(total_loss / n)
    }

    fn gradient(
        &self,
        rule_values: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        if rule_values.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: rule_values {:?} vs targets {:?}",
                rule_values.shape(),
                targets.shape()
            )));
        }

        let n = rule_values.len() as f64;
        let mut grad = Array::zeros(rule_values.raw_dim());

        for i in 0..rule_values.nrows() {
            for j in 0..rule_values.ncols() {
                let diff = targets[[i, j]] - rule_values[[i, j]];
                grad[[i, j]] = -2.0 * diff / (self.temperature * self.temperature * n);
            }
        }

        Ok(grad)
    }
}

/// Constraint violation loss - penalizes constraint violations.
#[derive(Debug, Clone)]
pub struct ConstraintViolationLoss {
    /// Penalty weight for violations.
    pub penalty_weight: f64,
}

impl Default for ConstraintViolationLoss {
    fn default() -> Self {
        Self {
            penalty_weight: 10.0,
        }
    }
}

impl Loss for ConstraintViolationLoss {
    fn compute(
        &self,
        constraint_values: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if constraint_values.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: constraint_values {:?} vs targets {:?}",
                constraint_values.shape(),
                targets.shape()
            )));
        }

        let n = constraint_values.len() as f64;
        let mut total_loss = 0.0;

        // Penalize any positive violation (constraint_values should be <= 0)
        for i in 0..constraint_values.nrows() {
            for j in 0..constraint_values.ncols() {
                let violation = (constraint_values[[i, j]] - targets[[i, j]]).max(0.0);
                total_loss += self.penalty_weight * violation * violation;
            }
        }

        Ok(total_loss / n)
    }

    fn gradient(
        &self,
        constraint_values: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        if constraint_values.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: constraint_values {:?} vs targets {:?}",
                constraint_values.shape(),
                targets.shape()
            )));
        }

        let n = constraint_values.len() as f64;
        let mut grad = Array::zeros(constraint_values.raw_dim());

        for i in 0..constraint_values.nrows() {
            for j in 0..constraint_values.ncols() {
                let violation = constraint_values[[i, j]] - targets[[i, j]];
                if violation > 0.0 {
                    grad[[i, j]] = 2.0 * self.penalty_weight * violation / n;
                }
            }
        }

        Ok(grad)
    }
}

/// Focal loss for addressing class imbalance.
/// Reference: Lin et al., "Focal Loss for Dense Object Detection"
#[derive(Debug, Clone)]
pub struct FocalLoss {
    /// Alpha weighting factor for positive class (range: [0, 1]).
    pub alpha: f64,
    /// Gamma focusing parameter (typically 2.0).
    pub gamma: f64,
    /// Epsilon for numerical stability.
    pub epsilon: f64,
}

impl Default for FocalLoss {
    fn default() -> Self {
        Self {
            alpha: 0.25,
            gamma: 2.0,
            epsilon: 1e-10,
        }
    }
}

impl Loss for FocalLoss {
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

        let n = predictions.nrows() as f64;
        let mut total_loss = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]]
                    .max(self.epsilon)
                    .min(1.0 - self.epsilon);
                let target = targets[[i, j]];

                // Focal loss: -alpha * (1 - p)^gamma * log(p) for positive class
                //             -(1 - alpha) * p^gamma * log(1 - p) for negative class
                if target > 0.5 {
                    // Positive class
                    let focal_weight = (1.0 - pred).powf(self.gamma);
                    total_loss -= self.alpha * focal_weight * pred.ln();
                } else {
                    // Negative class
                    let focal_weight = pred.powf(self.gamma);
                    total_loss -= (1.0 - self.alpha) * focal_weight * (1.0 - pred).ln();
                }
            }
        }

        Ok(total_loss / n)
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

        let n = predictions.nrows() as f64;
        let mut grad = Array::zeros(predictions.raw_dim());

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]]
                    .max(self.epsilon)
                    .min(1.0 - self.epsilon);
                let target = targets[[i, j]];

                if target > 0.5 {
                    // Positive class gradient
                    let focal_weight = (1.0 - pred).powf(self.gamma);
                    let d_focal = self.gamma * (1.0 - pred).powf(self.gamma - 1.0);
                    grad[[i, j]] = -self.alpha * (focal_weight / pred - d_focal * pred.ln()) / n;
                } else {
                    // Negative class gradient
                    let focal_weight = pred.powf(self.gamma);
                    let d_focal = self.gamma * pred.powf(self.gamma - 1.0);
                    grad[[i, j]] = -(1.0 - self.alpha)
                        * (d_focal * (1.0 - pred).ln() - focal_weight / (1.0 - pred))
                        / n;
                }
            }
        }

        Ok(grad)
    }
}

/// Huber loss for robust regression.
#[derive(Debug, Clone)]
pub struct HuberLoss {
    /// Delta threshold for switching between L1 and L2.
    pub delta: f64,
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl Loss for HuberLoss {
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

        let n = predictions.len() as f64;
        let mut total_loss = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let diff = (predictions[[i, j]] - targets[[i, j]]).abs();
                if diff <= self.delta {
                    // Quadratic for small errors
                    total_loss += 0.5 * diff * diff;
                } else {
                    // Linear for large errors
                    total_loss += self.delta * (diff - 0.5 * self.delta);
                }
            }
        }

        Ok(total_loss / n)
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

        let n = predictions.len() as f64;
        let mut grad = Array::zeros(predictions.raw_dim());

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let diff = predictions[[i, j]] - targets[[i, j]];
                let abs_diff = diff.abs();

                if abs_diff <= self.delta {
                    grad[[i, j]] = diff / n;
                } else {
                    grad[[i, j]] = self.delta * diff.signum() / n;
                }
            }
        }

        Ok(grad)
    }
}

/// Binary cross-entropy with logits loss (numerically stable).
#[derive(Debug, Clone, Default)]
pub struct BCEWithLogitsLoss;

impl Loss for BCEWithLogitsLoss {
    fn compute(
        &self,
        logits: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if logits.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: logits {:?} vs targets {:?}",
                logits.shape(),
                targets.shape()
            )));
        }

        let n = logits.len() as f64;
        let mut total_loss = 0.0;

        for i in 0..logits.nrows() {
            for j in 0..logits.ncols() {
                let logit = logits[[i, j]];
                let target = targets[[i, j]];

                // Numerically stable BCE: max(x, 0) - x * z + log(1 + exp(-|x|))
                // where x = logit, z = target
                let max_val = logit.max(0.0);
                total_loss += max_val - logit * target + (1.0 + (-logit.abs()).exp()).ln();
            }
        }

        Ok(total_loss / n)
    }

    fn gradient(
        &self,
        logits: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        if logits.shape() != targets.shape() {
            return Err(TrainError::LossError(format!(
                "Shape mismatch: logits {:?} vs targets {:?}",
                logits.shape(),
                targets.shape()
            )));
        }

        let n = logits.len() as f64;
        let mut grad = Array::zeros(logits.raw_dim());

        for i in 0..logits.nrows() {
            for j in 0..logits.ncols() {
                let logit = logits[[i, j]];
                let target = targets[[i, j]];

                // Gradient: sigmoid(logit) - target
                let sigmoid = 1.0 / (1.0 + (-logit).exp());
                grad[[i, j]] = (sigmoid - target) / n;
            }
        }

        Ok(grad)
    }
}

/// Dice loss for segmentation tasks.
#[derive(Debug, Clone)]
pub struct DiceLoss {
    /// Smoothing factor to avoid division by zero.
    pub smooth: f64,
}

impl Default for DiceLoss {
    fn default() -> Self {
        Self { smooth: 1.0 }
    }
}

impl Loss for DiceLoss {
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

        let mut intersection = 0.0;
        let mut pred_sum = 0.0;
        let mut target_sum = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]];
                let target = targets[[i, j]];

                intersection += pred * target;
                pred_sum += pred;
                target_sum += target;
            }
        }

        // Dice coefficient: 2 * |X ∩ Y| / (|X| + |Y|)
        // Dice loss: 1 - Dice coefficient
        let dice_coef = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth);
        Ok(1.0 - dice_coef)
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

        let mut intersection = 0.0;
        let mut pred_sum = 0.0;
        let mut target_sum = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                intersection += predictions[[i, j]] * targets[[i, j]];
                pred_sum += predictions[[i, j]];
                target_sum += targets[[i, j]];
            }
        }

        let denominator = pred_sum + target_sum + self.smooth;
        let numerator = 2.0 * intersection + self.smooth;

        let mut grad = Array::zeros(predictions.raw_dim());

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let target = targets[[i, j]];
                // Gradient of Dice loss w.r.t. predictions
                grad[[i, j]] =
                    -2.0 * (target * denominator - numerator) / (denominator * denominator);
            }
        }

        Ok(grad)
    }
}

/// Tversky loss (generalization of Dice loss).
/// Useful for handling class imbalance in segmentation.
#[derive(Debug, Clone)]
pub struct TverskyLoss {
    /// Alpha parameter (weight for false positives).
    pub alpha: f64,
    /// Beta parameter (weight for false negatives).
    pub beta: f64,
    /// Smoothing factor.
    pub smooth: f64,
}

impl Default for TverskyLoss {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
            smooth: 1.0,
        }
    }
}

impl Loss for TverskyLoss {
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

        let mut true_pos = 0.0;
        let mut false_pos = 0.0;
        let mut false_neg = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]];
                let target = targets[[i, j]];

                true_pos += pred * target;
                false_pos += pred * (1.0 - target);
                false_neg += (1.0 - pred) * target;
            }
        }

        // Tversky index: TP / (TP + alpha * FP + beta * FN)
        let tversky_index = (true_pos + self.smooth)
            / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth);

        Ok(1.0 - tversky_index)
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

        let mut true_pos = 0.0;
        let mut false_pos = 0.0;
        let mut false_neg = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]];
                let target = targets[[i, j]];

                true_pos += pred * target;
                false_pos += pred * (1.0 - target);
                false_neg += (1.0 - pred) * target;
            }
        }

        let denominator = true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth;
        let numerator = true_pos + self.smooth;

        let mut grad = Array::zeros(predictions.raw_dim());

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let target = targets[[i, j]];

                // Gradient of Tversky loss
                let d_tp = target;
                let d_fp = self.alpha * (1.0 - target);
                let d_fn = -self.beta * target;

                grad[[i, j]] = -(d_tp * denominator - numerator * (d_tp + d_fp + d_fn))
                    / (denominator * denominator);
            }
        }

        Ok(grad)
    }
}

/// Contrastive loss for metric learning.
/// Used to learn embeddings where similar pairs are close and dissimilar pairs are far apart.
#[derive(Debug, Clone)]
pub struct ContrastiveLoss {
    /// Margin for dissimilar pairs.
    pub margin: f64,
}

impl Default for ContrastiveLoss {
    fn default() -> Self {
        Self { margin: 1.0 }
    }
}

impl Loss for ContrastiveLoss {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.ncols() != 2 || targets.ncols() != 1 {
            return Err(TrainError::LossError(format!(
                "ContrastiveLoss expects predictions shape [N, 2] (distances) and targets shape [N, 1] (labels), got {:?} and {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let mut total_loss = 0.0;
        let n = predictions.nrows() as f64;

        for i in 0..predictions.nrows() {
            let distance = predictions[[i, 0]];
            let label = targets[[i, 0]];

            if label > 0.5 {
                // Similar pair: minimize distance
                total_loss += distance * distance;
            } else {
                // Dissimilar pair: maximize distance up to margin
                total_loss += (self.margin - distance).max(0.0).powi(2);
            }
        }

        Ok(total_loss / n)
    }

    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        let mut grad = Array::zeros(predictions.raw_dim());
        let n = predictions.nrows() as f64;

        for i in 0..predictions.nrows() {
            let distance = predictions[[i, 0]];
            let label = targets[[i, 0]];

            if label > 0.5 {
                // Similar pair gradient
                grad[[i, 0]] = 2.0 * distance / n;
            } else {
                // Dissimilar pair gradient
                if distance < self.margin {
                    grad[[i, 0]] = -2.0 * (self.margin - distance) / n;
                }
            }
        }

        Ok(grad)
    }
}

/// Triplet loss for metric learning.
/// Learns embeddings where anchor-positive distance < anchor-negative distance + margin.
#[derive(Debug, Clone)]
pub struct TripletLoss {
    /// Margin between positive and negative distances.
    pub margin: f64,
}

impl Default for TripletLoss {
    fn default() -> Self {
        Self { margin: 1.0 }
    }
}

impl Loss for TripletLoss {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        _targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.ncols() != 2 {
            return Err(TrainError::LossError(format!(
                "TripletLoss expects predictions shape [N, 2] (pos_dist, neg_dist), got {:?}",
                predictions.shape()
            )));
        }

        let mut total_loss = 0.0;
        let n = predictions.nrows() as f64;

        for i in 0..predictions.nrows() {
            let pos_distance = predictions[[i, 0]];
            let neg_distance = predictions[[i, 1]];

            // Loss = max(0, pos_dist - neg_dist + margin)
            let loss = (pos_distance - neg_distance + self.margin).max(0.0);
            total_loss += loss;
        }

        Ok(total_loss / n)
    }

    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        _targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        let mut grad = Array::zeros(predictions.raw_dim());
        let n = predictions.nrows() as f64;

        for i in 0..predictions.nrows() {
            let pos_distance = predictions[[i, 0]];
            let neg_distance = predictions[[i, 1]];

            if pos_distance - neg_distance + self.margin > 0.0 {
                // Gradient w.r.t. positive distance
                grad[[i, 0]] = 1.0 / n;
                // Gradient w.r.t. negative distance
                grad[[i, 1]] = -1.0 / n;
            }
        }

        Ok(grad)
    }
}

/// Hinge loss for maximum-margin classification (SVM-style).
#[derive(Debug, Clone)]
pub struct HingeLoss {
    /// Margin for classification.
    pub margin: f64,
}

impl Default for HingeLoss {
    fn default() -> Self {
        Self { margin: 1.0 }
    }
}

impl Loss for HingeLoss {
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

        let mut total_loss = 0.0;
        let n = predictions.nrows() as f64;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]];
                let target = targets[[i, j]];

                // targets should be +1 or -1
                let loss = (self.margin - target * pred).max(0.0);
                total_loss += loss;
            }
        }

        Ok(total_loss / n)
    }

    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        let mut grad = Array::zeros(predictions.raw_dim());
        let n = predictions.nrows() as f64;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]];
                let target = targets[[i, j]];

                if self.margin - target * pred > 0.0 {
                    grad[[i, j]] = -target / n;
                }
            }
        }

        Ok(grad)
    }
}

/// Kullback-Leibler Divergence loss.
/// Measures how one probability distribution diverges from a reference distribution.
#[derive(Debug, Clone)]
pub struct KLDivergenceLoss {
    /// Epsilon for numerical stability.
    pub epsilon: f64,
}

impl Default for KLDivergenceLoss {
    fn default() -> Self {
        Self { epsilon: 1e-10 }
    }
}

impl Loss for KLDivergenceLoss {
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

        let mut total_loss = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]].max(self.epsilon);
                let target = targets[[i, j]].max(self.epsilon);

                // KL(target || pred) = sum(target * log(target / pred))
                total_loss += target * (target / pred).ln();
            }
        }

        Ok(total_loss)
    }

    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        let mut grad = Array::zeros(predictions.raw_dim());

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]].max(self.epsilon);
                let target = targets[[i, j]].max(self.epsilon);

                // Gradient of KL divergence w.r.t. predictions
                grad[[i, j]] = -target / pred;
            }
        }

        Ok(grad)
    }
}

/// Poly Loss - Polynomial Expansion of Cross-Entropy Loss.
///
/// Paper: "PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions" (Leng et al., 2022)
/// <https://arxiv.org/abs/2204.12511>
///
/// PolyLoss adds polynomial terms to cross-entropy to provide better gradient flow
/// for well-classified examples. It helps with:
/// - Label noise robustness
/// - Improved generalization
/// - Better handling of class imbalance
///
/// The loss is defined as:
/// L_poly = CE + ε₁(1 - p_t) + ε₂(1 - p_t)² + ... + εⱼ(1 - p_t)^j
///
/// where p_t is the predicted probability of the target class, and εⱼ are polynomial coefficients.
/// In practice, Poly-1 (j=1) is most commonly used.
#[derive(Debug, Clone)]
pub struct PolyLoss {
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Polynomial coefficient (typically between 0.5 and 2.0)
    pub poly_coeff: f64,
}

impl Default for PolyLoss {
    fn default() -> Self {
        Self {
            epsilon: 1e-10,
            poly_coeff: 1.0, // Poly-1 Loss
        }
    }
}

impl PolyLoss {
    /// Create a new Poly Loss with custom coefficient.
    pub fn new(poly_coeff: f64) -> Self {
        Self {
            epsilon: 1e-10,
            poly_coeff,
        }
    }
}

impl Loss for PolyLoss {
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

        let n = predictions.nrows() as f64;
        let mut total_loss = 0.0;

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]]
                    .max(self.epsilon)
                    .min(1.0 - self.epsilon);
                let target = targets[[i, j]];

                // Cross-entropy term
                let ce = -target * pred.ln();

                // Poly term: ε * (1 - p_t) where p_t is probability of target class
                // For multi-class, we use the predicted probability at the target position
                let poly_term = if target > 0.5 {
                    // Target is 1, so p_t = pred
                    self.poly_coeff * (1.0 - pred)
                } else {
                    // Target is 0, so p_t = 1 - pred
                    self.poly_coeff * pred
                };

                total_loss += ce + poly_term;
            }
        }

        Ok(total_loss / n)
    }

    fn gradient(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<Array<f64, Ix2>> {
        let n = predictions.nrows() as f64;
        let mut grad = Array::zeros(predictions.raw_dim());

        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                let pred = predictions[[i, j]]
                    .max(self.epsilon)
                    .min(1.0 - self.epsilon);
                let target = targets[[i, j]];

                // Gradient of cross-entropy: -target / pred
                let ce_grad = -target / pred;

                // Gradient of poly term
                let poly_grad = if target > 0.5 {
                    // d/dp [ε * (1 - p)] = -ε
                    -self.poly_coeff
                } else {
                    // d/dp [ε * p] = ε
                    self.poly_coeff
                };

                grad[[i, j]] = (ce_grad + poly_grad) / n;
            }
        }

        Ok(grad)
    }

    fn name(&self) -> &str {
        "poly_loss"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cross_entropy_loss() {
        let loss = CrossEntropyLoss::default();
        let predictions = array![[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]];
        let targets = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val > 0.0);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());
    }

    #[test]
    fn test_mse_loss() {
        let loss = MseLoss;
        let predictions = array![[1.0, 2.0], [3.0, 4.0]];
        let targets = array![[1.5, 2.5], [3.5, 4.5]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!((loss_val - 0.25).abs() < 1e-6);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());
    }

    #[test]
    fn test_rule_satisfaction_loss() {
        let loss = RuleSatisfactionLoss::default();
        let rule_values = array![[0.9, 0.8], [0.95, 0.85]];
        let targets = array![[1.0, 1.0], [1.0, 1.0]];

        let loss_val = loss.compute(&rule_values.view(), &targets.view()).unwrap();
        assert!(loss_val > 0.0);

        let grad = loss.gradient(&rule_values.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), rule_values.shape());
    }

    #[test]
    fn test_constraint_violation_loss() {
        let loss = ConstraintViolationLoss::default();
        let constraint_values = array![[0.1, -0.1], [0.2, -0.2]];
        let targets = array![[0.0, 0.0], [0.0, 0.0]];

        let loss_val = loss
            .compute(&constraint_values.view(), &targets.view())
            .unwrap();
        assert!(loss_val > 0.0);

        let grad = loss
            .gradient(&constraint_values.view(), &targets.view())
            .unwrap();
        assert_eq!(grad.shape(), constraint_values.shape());
    }

    #[test]
    fn test_focal_loss() {
        let loss = FocalLoss::default();
        let predictions = array![[0.9, 0.1], [0.2, 0.8]];
        let targets = array![[1.0, 0.0], [0.0, 1.0]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val >= 0.0);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());
    }

    #[test]
    fn test_huber_loss() {
        let loss = HuberLoss::default();
        let predictions = array![[1.0, 3.0], [2.0, 5.0]];
        let targets = array![[1.5, 2.0], [2.5, 4.0]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val > 0.0);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());
    }

    #[test]
    fn test_bce_with_logits_loss() {
        let loss = BCEWithLogitsLoss;
        let logits = array![[0.5, -0.5], [1.0, -1.0]];
        let targets = array![[1.0, 0.0], [1.0, 0.0]];

        let loss_val = loss.compute(&logits.view(), &targets.view()).unwrap();
        assert!(loss_val >= 0.0);

        let grad = loss.gradient(&logits.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), logits.shape());
    }

    #[test]
    fn test_dice_loss() {
        let loss = DiceLoss::default();
        let predictions = array![[0.9, 0.1], [0.8, 0.2]];
        let targets = array![[1.0, 0.0], [1.0, 0.0]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val >= 0.0);
        assert!(loss_val <= 1.0);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());
    }

    #[test]
    fn test_tversky_loss() {
        let loss = TverskyLoss::default();
        let predictions = array![[0.9, 0.1], [0.8, 0.2]];
        let targets = array![[1.0, 0.0], [1.0, 0.0]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val >= 0.0);
        assert!(loss_val <= 1.0);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());
    }

    #[test]
    fn test_contrastive_loss() {
        let loss = ContrastiveLoss::default();
        // Predictions: [N, 2] where first column is distance, second is unused
        // Targets: [N, 1] where 1.0 = similar pair, 0.0 = dissimilar pair
        let predictions = array![[0.5, 0.0], [1.5, 0.0], [0.2, 0.0]];
        let targets = array![[1.0], [0.0], [1.0]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val >= 0.0);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());

        // For similar pair (label=1), gradient should push distance down
        assert!(grad[[0, 0]] > 0.0);
        // For dissimilar pair beyond margin, gradient should be 0
        assert_eq!(grad[[1, 0]], 0.0);
    }

    #[test]
    fn test_triplet_loss() {
        let loss = TripletLoss::default();
        // Predictions: [N, 2] where columns are (positive_distance, negative_distance)
        let predictions = array![[0.5, 2.0], [1.0, 0.5], [0.3, 1.5]];
        let targets = array![[0.0], [0.0], [0.0]]; // Not used but required for interface

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val >= 0.0);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());

        // First triplet: pos_dist < neg_dist - margin, so no gradient
        assert_eq!(grad[[0, 0]], 0.0);
        assert_eq!(grad[[0, 1]], 0.0);

        // Second triplet: pos_dist > neg_dist, so should have gradient
        assert!(grad[[1, 0]] > 0.0);
        assert!(grad[[1, 1]] < 0.0);
    }

    #[test]
    fn test_hinge_loss() {
        let loss = HingeLoss::default();
        // Predictions are raw scores, targets should be +1 or -1
        let predictions = array![[0.5, -0.5], [2.0, -2.0]];
        let targets = array![[1.0, -1.0], [1.0, -1.0]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val >= 0.0);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());

        // For correct predictions with large margin, gradient should be 0
        assert_eq!(grad[[1, 0]], 0.0);
        assert_eq!(grad[[1, 1]], 0.0);
    }

    #[test]
    fn test_kl_divergence_loss() {
        let loss = KLDivergenceLoss::default();
        // Both predictions and targets should be probability distributions
        let predictions = array![[0.6, 0.4], [0.7, 0.3]];
        let targets = array![[0.5, 0.5], [0.8, 0.2]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val >= 0.0);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());

        // KL divergence should be 0 when distributions are identical
        let identical_preds = array![[0.5, 0.5]];
        let identical_targets = array![[0.5, 0.5]];
        let identical_loss = loss
            .compute(&identical_preds.view(), &identical_targets.view())
            .unwrap();
        assert!(identical_loss.abs() < 1e-6);
    }

    #[test]
    fn test_poly_loss() {
        let loss = PolyLoss::default();
        let predictions = array![[0.9, 0.1], [0.2, 0.8]];
        let targets = array![[1.0, 0.0], [0.0, 1.0]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val > 0.0);

        let grad = loss.gradient(&predictions.view(), &targets.view()).unwrap();
        assert_eq!(grad.shape(), predictions.shape());

        // Poly loss should be greater than standard cross-entropy for well-classified examples
        let ce_loss = CrossEntropyLoss::default();
        let ce_val = ce_loss
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // With default poly_coeff = 1.0, poly loss includes additional penalty term
        assert!(loss_val >= ce_val);
    }

    #[test]
    fn test_poly_loss_custom_coefficient() {
        let loss = PolyLoss::new(2.0);
        let predictions = array![[0.8, 0.2]];
        let targets = array![[1.0, 0.0]];

        let loss_val = loss.compute(&predictions.view(), &targets.view()).unwrap();
        assert!(loss_val > 0.0);

        // Higher coefficient should result in larger poly term
        let loss_low_coeff = PolyLoss::new(0.5);
        let loss_val_low = loss_low_coeff
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        assert!(loss_val > loss_val_low);
    }
}
