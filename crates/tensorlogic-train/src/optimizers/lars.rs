//! LARS optimizer (Layer-wise Adaptive Rate Scaling).
//!
//! LARS scales the learning rate for each layer based on the ratio of the parameter norm
//! to the gradient norm. This is particularly effective for large batch training.
//!
//! Reference: You et al. "Large Batch Training of Convolutional Networks" (2017)

use super::common::{compute_gradient_norm, GradClipMode, Optimizer, OptimizerConfig};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// LARS optimizer (Layer-wise Adaptive Rate Scaling).
///
/// LARS scales the learning rate for each layer based on the ratio of the parameter norm
/// to the gradient norm. This is particularly effective for large batch training.
///
/// Reference: You et al. "Large Batch Training of Convolutional Networks" (2017)
#[derive(Debug)]
pub struct LarsOptimizer {
    config: OptimizerConfig,
    /// Momentum buffers for each parameter.
    velocity: HashMap<String, Array<f64, Ix2>>,
    /// Trust coefficient for layer-wise LR adaptation (typically 0.001).
    trust_coef: f64,
    /// Whether to apply LARS to bias parameters.
    exclude_bias: bool,
}

impl LarsOptimizer {
    /// Create a new LARS optimizer.
    ///
    /// # Arguments
    /// * `config` - Optimizer configuration
    /// * `trust_coef` - Trust coefficient for adaptive LR (default: 0.001)
    /// * `exclude_bias` - Whether to exclude bias from LARS adaptation (default: true)
    pub fn new(config: OptimizerConfig, trust_coef: f64, exclude_bias: bool) -> Self {
        Self {
            config,
            velocity: HashMap::new(),
            trust_coef,
            exclude_bias,
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    let total_norm = compute_gradient_norm(gradients);
                    if total_norm > clip_value {
                        let scale = clip_value / total_norm;
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|g| g * scale);
                        }
                    }
                }
            }
        }
    }

    /// Compute layer-wise adaptive learning rate.
    fn compute_adaptive_lr(
        &self,
        param: &Array<f64, Ix2>,
        grad: &Array<f64, Ix2>,
        name: &str,
    ) -> f64 {
        if self.exclude_bias && (name.contains("bias") || name.contains("b")) {
            return self.config.learning_rate;
        }
        let param_norm: f64 = param.iter().map(|&p| p * p).sum::<f64>().sqrt();
        let grad_norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
        if param_norm == 0.0 || grad_norm == 0.0 {
            return self.config.learning_rate;
        }
        let local_lr = self.trust_coef * param_norm / grad_norm;
        self.config.learning_rate * local_lr
    }
}

impl Optimizer for LarsOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);
        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;
            let adaptive_lr = self.compute_adaptive_lr(param, grad, name);
            let mut effective_grad = grad.clone();
            if self.config.weight_decay > 0.0 {
                effective_grad += &(&*param * self.config.weight_decay);
            }
            if !self.velocity.contains_key(name) {
                self.velocity
                    .insert(name.clone(), Array::zeros(param.raw_dim()));
            }
            let velocity = self.velocity.get_mut(name).unwrap();
            velocity.mapv_inplace(|v| self.config.momentum * v);
            *velocity = &*velocity + &(effective_grad * adaptive_lr);
            *param = &*param - &*velocity;
        }
        Ok(())
    }

    fn zero_grad(&mut self) {}

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        state.insert("trust_coef".to_string(), vec![self.trust_coef]);
        state.insert(
            "exclude_bias".to_string(),
            vec![if self.exclude_bias { 1.0 } else { 0.0 }],
        );
        for (name, velocity) in &self.velocity {
            state.insert(
                format!("velocity_{}", name),
                velocity.iter().copied().collect(),
            );
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(trust) = state.get("trust_coef") {
            self.trust_coef = trust[0];
        }
        if let Some(exclude) = state.get("exclude_bias") {
            self.exclude_bias = exclude[0] > 0.5;
        }
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("velocity_") {
                if let Some(velocity) = self.velocity.get(name) {
                    let shape = velocity.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.velocity.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_lars_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            momentum: 0.9,
            weight_decay: 0.0001,
            ..Default::default()
        };
        let mut optimizer = LarsOptimizer::new(config, 0.001, true);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);
        optimizer.step(&mut params, &grads).unwrap();
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[1, 1]] < 4.0);
        let state = optimizer.state_dict();
        assert!(state.contains_key("trust_coef"));
        assert!(state.contains_key("exclude_bias"));
        assert!(state.contains_key("velocity_w"));
    }

    #[test]
    fn test_lars_bias_exclusion() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            momentum: 0.9,
            ..Default::default()
        };
        let mut optimizer = LarsOptimizer::new(config.clone(), 0.001, true);
        let mut params = HashMap::new();
        params.insert("weights".to_string(), array![[1.0, 2.0]]);
        params.insert("bias".to_string(), array![[1.0, 2.0]]);
        let mut grads = HashMap::new();
        grads.insert("weights".to_string(), array![[0.1, 0.1]]);
        grads.insert("bias".to_string(), array![[0.1, 0.1]]);
        optimizer.step(&mut params, &grads).unwrap();
        let weights = params.get("weights").unwrap();
        let bias = params.get("bias").unwrap();
        assert!(weights[[0, 0]] < 1.0);
        assert!(bias[[0, 0]] < 1.0);
    }
}
