//! Adagrad optimizer (Adaptive Gradient).
//!
//! Adagrad adapts the learning rate for each parameter based on the historical
//! sum of squared gradients, giving frequently occurring features lower learning rates.
//!
//! Reference: Duchi et al., "Adaptive Subgradient Methods for Online Learning
//! and Stochastic Optimization", JMLR 2011

use super::common::{compute_gradient_norm, GradClipMode, Optimizer, OptimizerConfig};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// Adagrad optimizer (Adaptive Gradient).
#[derive(Debug)]
pub struct AdagradOptimizer {
    config: OptimizerConfig,
    /// Accumulated sum of squared gradients.
    sum_squared_grads: HashMap<String, Array<f64, Ix2>>,
}

impl AdagradOptimizer {
    /// Create a new Adagrad optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            sum_squared_grads: HashMap::new(),
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
}

impl Optimizer for AdagradOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);
        let lr = self.config.learning_rate;
        let eps = self.config.epsilon;
        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;
            if !self.sum_squared_grads.contains_key(name) {
                self.sum_squared_grads
                    .insert(name.clone(), Array::zeros(param.raw_dim()));
            }
            let sum_sq = self.sum_squared_grads.get_mut(name).unwrap();
            let grad_squared = grad.mapv(|g| g * g);
            *sum_sq = &*sum_sq + &grad_squared;
            let update = grad / &sum_sq.mapv(|s| s.sqrt() + eps);
            *param = &*param - &(update * lr);
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
        for (name, sum_sq) in &self.sum_squared_grads {
            state.insert(
                format!("sum_squared_grads_{}", name),
                sum_sq.iter().copied().collect(),
            );
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("sum_squared_grads_") {
                if let Some(sum_sq) = self.sum_squared_grads.get(name) {
                    let shape = sum_sq.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.sum_squared_grads.insert(name.to_string(), arr);
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
    fn test_adagrad_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = AdagradOptimizer::new(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2]]);
        optimizer.step(&mut params, &grads).unwrap();
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[0, 1]] < 2.0);
    }
}
