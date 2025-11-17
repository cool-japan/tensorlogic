//! RMSprop optimizer (Root Mean Square Propagation).
//!
//! RMSprop divides the learning rate by an exponentially decaying average
//! of squared gradients. It's effective for non-stationary objectives.
//!
//! Reference: Tieleman & Hinton, "Lecture 6.5-rmsprop", COURSERA: Neural networks for machine learning

use super::common::{compute_gradient_norm, GradClipMode, Optimizer, OptimizerConfig};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// RMSprop optimizer (Root Mean Square Propagation).
#[derive(Debug)]
pub struct RMSpropOptimizer {
    config: OptimizerConfig,
    /// Moving average of squared gradients.
    v: HashMap<String, Array<f64, Ix2>>,
}

impl RMSpropOptimizer {
    /// Create a new RMSprop optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            v: HashMap::new(),
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

impl Optimizer for RMSpropOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);
        let lr = self.config.learning_rate;
        let alpha = self.config.beta2;
        let eps = self.config.epsilon;
        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;
            if !self.v.contains_key(name) {
                self.v.insert(name.clone(), Array::zeros(param.raw_dim()));
            }
            let v = self.v.get_mut(name).unwrap();
            let grad_squared = grad.mapv(|g| g * g);
            *v = &*v * alpha + &(grad_squared * (1.0 - alpha));
            let update = grad / &v.mapv(|v_val| v_val.sqrt() + eps);
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
        for (name, v_val) in &self.v {
            state.insert(format!("v_{}", name), v_val.iter().copied().collect());
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("v_") {
                if let Some(v) = self.v.get(name) {
                    let shape = v.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.v.insert(name.to_string(), arr);
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
    fn test_rmsprop_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let mut optimizer = RMSpropOptimizer::new(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);
        optimizer.step(&mut params, &grads).unwrap();
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
    }
}
