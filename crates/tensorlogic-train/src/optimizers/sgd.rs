//! SGD optimizer with momentum.

use super::common::{compute_gradient_norm, GradClipMode, Optimizer, OptimizerConfig};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// SGD optimizer with momentum.
#[derive(Debug)]
pub struct SgdOptimizer {
    config: OptimizerConfig,
    /// Momentum buffers for each parameter.
    velocity: HashMap<String, Array<f64, Ix2>>,
}

impl SgdOptimizer {
    /// Create a new SGD optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            velocity: HashMap::new(),
        }
    }

    /// Apply gradient clipping if configured.
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    // Clip by value (element-wise)
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|g| g.max(-clip_value).min(clip_value));
                    }
                }
                GradClipMode::Norm => {
                    // Clip by global L2 norm
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

impl Optimizer for SgdOptimizer {
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

            // Initialize velocity if not present
            if !self.velocity.contains_key(name) {
                self.velocity
                    .insert(name.clone(), Array::zeros(param.raw_dim()));
            }

            let velocity = self.velocity.get_mut(name).unwrap();

            // Update velocity: v = momentum * v + lr * grad
            velocity.mapv_inplace(|v| self.config.momentum * v);
            *velocity = &*velocity + &(grad * self.config.learning_rate);

            // Update parameter: param = param - velocity
            *param = &*param - &*velocity;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Gradients are managed externally, nothing to do here
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        for (name, velocity) in &self.velocity {
            state.insert(
                format!("velocity_{}", name),
                velocity.iter().copied().collect(),
            );
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("velocity_") {
                // Reconstruct array from values (assumes correct shape)
                if let Some(velocity) = self.velocity.get(name) {
                    let shape = velocity.raw_dim();
                    if let Ok(new_velocity) = Array::from_shape_vec(shape, values) {
                        self.velocity.insert(name.to_string(), new_velocity);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::array;

    #[test]
    fn test_sgd_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            momentum: 0.9,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1]]);

        optimizer.step(&mut params, &grads).unwrap();

        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0); // Should decrease
        assert!(w[[0, 1]] < 2.0);

        // Test state dict
        let state = optimizer.state_dict();
        assert!(state.contains_key("velocity_w"));
    }

    #[test]
    fn test_gradient_clipping() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            grad_clip: Some(0.05),
            grad_clip_mode: GradClipMode::Value,
            ..Default::default()
        };
        let mut optimizer = SgdOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[1.0]]); // Large gradient

        optimizer.step(&mut params, &grads).unwrap();

        // Gradient should be clipped to 0.05, so update should be small
        let w = params.get("w").unwrap();
        assert!((w[[0, 0]] - 1.0).abs() < 0.1); // Small change due to clipping
    }
}
