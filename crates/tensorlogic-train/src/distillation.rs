//! Knowledge distillation utilities for model compression and transfer learning.
//!
//! This module provides utilities for knowledge distillation, where a smaller "student"
//! model learns from a larger "teacher" model's outputs.

use crate::{Loss, TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2};

/// Knowledge distillation loss that combines student predictions with teacher soft targets.
///
/// Based on "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015).
pub struct DistillationLoss {
    /// Temperature for softening probabilities (higher = softer).
    pub temperature: f64,
    /// Weight for distillation loss (1 - alpha for hard target loss).
    pub alpha: f64,
    /// Base loss function for hard targets.
    pub hard_loss: Box<dyn Loss>,
}

impl DistillationLoss {
    /// Create a new distillation loss.
    ///
    /// # Arguments
    /// * `temperature` - Temperature for softening (typically 2.0-5.0)
    /// * `alpha` - Weight for soft targets (typically 0.5-0.9)
    /// * `hard_loss` - Loss function for hard targets
    pub fn new(temperature: f64, alpha: f64, hard_loss: Box<dyn Loss>) -> TrainResult<Self> {
        if temperature <= 0.0 {
            return Err(TrainError::ConfigError(
                "Temperature must be positive".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&alpha) {
            return Err(TrainError::ConfigError(
                "Alpha must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            temperature,
            alpha,
            hard_loss,
        })
    }

    /// Compute distillation loss combining soft and hard targets.
    ///
    /// # Arguments
    /// * `student_logits` - Raw student model outputs (before softmax)
    /// * `teacher_logits` - Raw teacher model outputs (before softmax)
    /// * `hard_targets` - True labels (one-hot encoded)
    ///
    /// # Returns
    /// Combined distillation loss
    pub fn compute_distillation(
        &self,
        student_logits: &ArrayView<f64, Ix2>,
        teacher_logits: &ArrayView<f64, Ix2>,
        hard_targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if student_logits.shape() != teacher_logits.shape() {
            return Err(TrainError::LossError(format!(
                "Student and teacher logits must have same shape: {:?} vs {:?}",
                student_logits.shape(),
                teacher_logits.shape()
            )));
        }

        // Soft targets loss (KL divergence of softened distributions)
        let soft_loss =
            self.compute_kl_divergence_with_temperature(student_logits, teacher_logits)?;

        // Hard targets loss
        let hard_loss = self.hard_loss.compute(student_logits, hard_targets)?;

        // Combine with weighting
        // Note: soft loss is scaled by T^2 as per original paper
        let t_squared = self.temperature * self.temperature;
        let combined_loss = self.alpha * soft_loss * t_squared + (1.0 - self.alpha) * hard_loss;

        Ok(combined_loss)
    }

    /// Compute KL divergence between temperature-scaled distributions.
    fn compute_kl_divergence_with_temperature(
        &self,
        student_logits: &ArrayView<f64, Ix2>,
        teacher_logits: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        let t = self.temperature;

        let mut total_loss = 0.0;
        let n_samples = student_logits.nrows();

        for i in 0..n_samples {
            // Scale by temperature and apply softmax
            let student_probs = self.softmax_with_temperature(&student_logits.row(i), t);
            let teacher_probs = self.softmax_with_temperature(&teacher_logits.row(i), t);

            // KL divergence: sum(teacher * log(teacher / student))
            for j in 0..student_probs.len() {
                if teacher_probs[j] > 1e-8 {
                    let ratio = teacher_probs[j] / (student_probs[j] + 1e-8);
                    total_loss += teacher_probs[j] * ratio.ln();
                }
            }
        }

        Ok(total_loss / n_samples as f64)
    }

    /// Apply softmax with temperature scaling.
    fn softmax_with_temperature(
        &self,
        logits: &ArrayView<f64, scirs2_core::ndarray::Ix1>,
        temperature: f64,
    ) -> Vec<f64> {
        let scaled: Vec<f64> = logits.iter().map(|&x| x / temperature).collect();

        let max_val = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();

        exp_vals.iter().map(|&x| x / sum).collect()
    }
}

/// Feature-based distillation that matches intermediate layer representations.
pub struct FeatureDistillationLoss {
    /// Weight for each feature layer.
    pub layer_weights: Vec<f64>,
    /// Distance metric (2.0 for L2, 1.0 for L1).
    pub p_norm: f64,
}

impl FeatureDistillationLoss {
    /// Create a new feature distillation loss.
    ///
    /// # Arguments
    /// * `layer_weights` - Weights for each intermediate layer
    /// * `p_norm` - Norm to use for distance (1.0 or 2.0)
    pub fn new(layer_weights: Vec<f64>, p_norm: f64) -> TrainResult<Self> {
        if layer_weights.is_empty() {
            return Err(TrainError::ConfigError(
                "Must specify at least one layer weight".to_string(),
            ));
        }

        if p_norm != 1.0 && p_norm != 2.0 {
            return Err(TrainError::ConfigError(
                "p_norm must be 1.0 or 2.0".to_string(),
            ));
        }

        Ok(Self {
            layer_weights,
            p_norm,
        })
    }

    /// Compute feature matching loss for intermediate representations.
    ///
    /// # Arguments
    /// * `student_features` - Student model's intermediate features
    /// * `teacher_features` - Teacher model's intermediate features
    ///
    /// # Returns
    /// Weighted sum of feature matching losses
    pub fn compute_feature_loss(
        &self,
        student_features: &[ArrayView<f64, Ix2>],
        teacher_features: &[ArrayView<f64, Ix2>],
    ) -> TrainResult<f64> {
        if student_features.len() != teacher_features.len() {
            return Err(TrainError::LossError(
                "Number of student and teacher feature layers must match".to_string(),
            ));
        }

        if student_features.len() != self.layer_weights.len() {
            return Err(TrainError::LossError(format!(
                "Number of layers ({}) must match number of weights ({})",
                student_features.len(),
                self.layer_weights.len()
            )));
        }

        let mut total_loss = 0.0;

        for (i, (student_feat, teacher_feat)) in student_features
            .iter()
            .zip(teacher_features.iter())
            .enumerate()
        {
            if student_feat.shape() != teacher_feat.shape() {
                return Err(TrainError::LossError(format!(
                    "Layer {} shape mismatch: {:?} vs {:?}",
                    i,
                    student_feat.shape(),
                    teacher_feat.shape()
                )));
            }

            // Compute distance
            let mut layer_loss = 0.0;
            for (&s, &t) in student_feat.iter().zip(teacher_feat.iter()) {
                let diff = (s - t).abs();
                layer_loss += if self.p_norm == 2.0 {
                    diff * diff
                } else {
                    diff
                };
            }

            // Normalize by number of elements
            let n_elements = student_feat.len() as f64;
            layer_loss /= n_elements;

            // Apply layer weight
            total_loss += self.layer_weights[i] * layer_loss;
        }

        Ok(total_loss)
    }
}

/// Attention transfer for distillation based on attention maps.
pub struct AttentionTransferLoss {
    /// Beta parameter for attention map normalization.
    pub beta: f64,
}

impl AttentionTransferLoss {
    /// Create a new attention transfer loss.
    ///
    /// # Arguments
    /// * `beta` - Power for attention map normalization (typically 2.0)
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }

    /// Compute attention transfer loss.
    ///
    /// # Arguments
    /// * `student_attention` - Student attention maps
    /// * `teacher_attention` - Teacher attention maps
    ///
    /// # Returns
    /// Attention transfer loss
    pub fn compute_attention_loss(
        &self,
        student_attention: &ArrayView<f64, Ix2>,
        teacher_attention: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if student_attention.shape() != teacher_attention.shape() {
            return Err(TrainError::LossError(format!(
                "Attention maps must have same shape: {:?} vs {:?}",
                student_attention.shape(),
                teacher_attention.shape()
            )));
        }

        // Normalize attention maps
        let student_norm = self.normalize_attention(student_attention);
        let teacher_norm = self.normalize_attention(teacher_attention);

        // Compute L2 distance
        let mut loss = 0.0;
        for (s, t) in student_norm.iter().zip(teacher_norm.iter()) {
            let diff = s - t;
            loss += diff * diff;
        }

        let n_elements = student_norm.len() as f64;
        Ok(loss / n_elements)
    }

    /// Normalize attention map using beta-power normalization.
    fn normalize_attention(&self, attention: &ArrayView<f64, Ix2>) -> Array<f64, Ix2> {
        let mut normalized = attention.mapv(|x| x.abs().powf(self.beta));

        // Normalize each sample
        for mut row in normalized.rows_mut() {
            let sum: f64 = row.iter().sum();
            if sum > 1e-8 {
                row.mapv_inplace(|x| x / sum);
            }
        }

        normalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CrossEntropyLoss;
    use scirs2_core::array;

    #[test]
    fn test_distillation_loss_creation() {
        let loss = DistillationLoss::new(3.0, 0.7, Box::new(CrossEntropyLoss::default()));
        assert!(loss.is_ok());

        let loss = loss.unwrap();
        assert_eq!(loss.temperature, 3.0);
        assert_eq!(loss.alpha, 0.7);
    }

    #[test]
    fn test_distillation_invalid_temperature() {
        let result = DistillationLoss::new(0.0, 0.5, Box::new(CrossEntropyLoss::default()));
        assert!(result.is_err());

        let result = DistillationLoss::new(-1.0, 0.5, Box::new(CrossEntropyLoss::default()));
        assert!(result.is_err());
    }

    #[test]
    fn test_distillation_invalid_alpha() {
        let result = DistillationLoss::new(3.0, -0.1, Box::new(CrossEntropyLoss::default()));
        assert!(result.is_err());

        let result = DistillationLoss::new(3.0, 1.1, Box::new(CrossEntropyLoss::default()));
        assert!(result.is_err());
    }

    #[test]
    fn test_distillation_compute() {
        let loss = DistillationLoss::new(2.0, 0.5, Box::new(CrossEntropyLoss::default())).unwrap();

        let student_logits = array![[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]];
        let teacher_logits = array![[1.2, 1.8, 0.6], [0.6, 1.1, 1.9]];
        let hard_targets = array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let result = loss.compute_distillation(
            &student_logits.view(),
            &teacher_logits.view(),
            &hard_targets.view(),
        );

        assert!(result.is_ok());
        let loss_value = result.unwrap();
        assert!(loss_value > 0.0);
        assert!(loss_value.is_finite());
    }

    #[test]
    fn test_feature_distillation_loss() {
        let loss = FeatureDistillationLoss::new(vec![0.5, 0.3, 0.2], 2.0).unwrap();

        let s1 = array![[1.0, 2.0], [3.0, 4.0]];
        let s2 = array![[0.5, 1.5], [2.5, 3.5]];
        let s3 = array![[0.1, 0.2], [0.3, 0.4]];
        let student_features = vec![s1.view(), s2.view(), s3.view()];

        let t1 = array![[1.1, 2.1], [3.1, 4.1]];
        let t2 = array![[0.6, 1.6], [2.6, 3.6]];
        let t3 = array![[0.2, 0.3], [0.4, 0.5]];
        let teacher_features = vec![t1.view(), t2.view(), t3.view()];

        let result = loss.compute_feature_loss(&student_features, &teacher_features);
        assert!(result.is_ok());

        let loss_value = result.unwrap();
        assert!(loss_value > 0.0);
        assert!(loss_value < 1.0); // Should be small for similar features
    }

    #[test]
    fn test_attention_transfer_loss() {
        let loss = AttentionTransferLoss::new(2.0);

        let student_attention = array![[0.3, 0.5, 0.2], [0.4, 0.4, 0.2]];
        let teacher_attention = array![[0.35, 0.45, 0.2], [0.35, 0.45, 0.2]];

        let result =
            loss.compute_attention_loss(&student_attention.view(), &teacher_attention.view());
        assert!(result.is_ok());

        let loss_value = result.unwrap();
        assert!(loss_value >= 0.0);
        assert!(loss_value.is_finite());
    }

    #[test]
    fn test_feature_distillation_shape_mismatch() {
        let loss = FeatureDistillationLoss::new(vec![1.0], 2.0).unwrap();

        let s1 = array![[1.0, 2.0]];
        let student_features = vec![s1.view()];

        let t1 = array![[1.0, 2.0, 3.0]];
        let teacher_features = vec![t1.view()];

        let result = loss.compute_feature_loss(&student_features, &teacher_features);
        assert!(result.is_err());
    }
}
