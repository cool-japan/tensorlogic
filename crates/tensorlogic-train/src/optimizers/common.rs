//! Common optimizer utilities and traits.

use crate::TrainResult;
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// Compute the global L2 norm of all gradients.
///
/// # Arguments
/// * `gradients` - Gradients for all parameters
///
/// # Returns
/// The L2 norm of all gradients combined
pub fn compute_gradient_norm(gradients: &HashMap<String, Array<f64, Ix2>>) -> f64 {
    let mut total_norm_sq = 0.0;

    for grad in gradients.values() {
        for &g in grad.iter() {
            total_norm_sq += g * g;
        }
    }

    total_norm_sq.sqrt()
}

/// Gradient clipping mode.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GradClipMode {
    /// Clip by value (element-wise).
    Value,
    /// Clip by global L2 norm.
    Norm,
}

/// Configuration for optimizers.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Learning rate.
    pub learning_rate: f64,
    /// Momentum (for SGD).
    pub momentum: f64,
    /// Beta1 (for Adam/AdamW).
    pub beta1: f64,
    /// Beta2 (for Adam/AdamW).
    pub beta2: f64,
    /// Epsilon for numerical stability.
    pub epsilon: f64,
    /// Weight decay (for AdamW).
    pub weight_decay: f64,
    /// Gradient clipping threshold (None = no clipping).
    pub grad_clip: Option<f64>,
    /// Gradient clipping mode.
    pub grad_clip_mode: GradClipMode,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            grad_clip: None,
            grad_clip_mode: GradClipMode::Value,
        }
    }
}

/// Trait for optimizers.
pub trait Optimizer {
    /// Update parameters with computed gradients.
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()>;

    /// Zero all gradients.
    fn zero_grad(&mut self);

    /// Get current learning rate.
    fn get_lr(&self) -> f64;

    /// Set learning rate.
    fn set_lr(&mut self, lr: f64);

    /// Get optimizer state for checkpointing.
    fn state_dict(&self) -> HashMap<String, Vec<f64>>;

    /// Load optimizer state from checkpoint.
    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>);
}
