//! LAMB optimizer (Layer-wise Adaptive Moments optimizer for Batch training).
//!
//! LAMB is designed for large batch training and uses layer-wise adaptation
//! of the learning rate based on the ratio of parameter and update norms.
//!
//! Reference: You et al., "Large Batch Optimization for Deep Learning:
//! Training BERT in 76 minutes", ICLR 2020

use super::common::{compute_gradient_norm, GradClipMode, Optimizer, OptimizerConfig};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// LAMB optimizer (Layer-wise Adaptive Moments optimizer for Batch training).
/// Designed for large batch training, uses layer-wise adaptation.
#[derive(Debug)]
pub struct LambOptimizer {
    config: OptimizerConfig,
    /// First moment estimates.
    m: HashMap<String, Array<f64, Ix2>>,
    /// Second moment estimates.
    v: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl LambOptimizer {
    /// Create a new LAMB optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
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

    /// Compute L2 norm of an array.
    fn compute_norm(arr: &Array<f64, Ix2>) -> f64 {
        arr.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

impl Optimizer for LambOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);
        self.t += 1;
        let lr = self.config.learning_rate;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.epsilon;
        let weight_decay = self.config.weight_decay;
        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.v.insert(name.clone(), Array::zeros(param.raw_dim()));
            }
            let m = self.m.get_mut(name).unwrap();
            let v = self.v.get_mut(name).unwrap();
            *m = &*m * beta1 + &(grad * (1.0 - beta1));
            let grad_squared = grad.mapv(|g| g * g);
            *v = &*v * beta2 + &(grad_squared * (1.0 - beta2));
            let m_hat = &*m / (1.0 - beta1.powi(self.t as i32));
            let v_hat = &*v / (1.0 - beta2.powi(self.t as i32));
            let adam_step = &m_hat / &v_hat.mapv(|v_val| v_val.sqrt() + eps);
            let update = &adam_step + &param.mapv(|p| p * weight_decay);
            let param_norm = Self::compute_norm(param);
            let update_norm = Self::compute_norm(&update);
            let trust_ratio = if param_norm > 0.0 && update_norm > 0.0 {
                param_norm / update_norm
            } else {
                1.0
            };
            *param = &*param - &(update * (lr * trust_ratio));
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
        state.insert("t".to_string(), vec![self.t as f64]);
        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, v_val) in &self.v {
            state.insert(format!("v_{}", name), v_val.iter().copied().collect());
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(t_vals) = state.get("t") {
            self.t = t_vals[0] as usize;
        }
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m) = self.m.get(name) {
                    let shape = m.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.m.insert(name.to_string(), arr);
                    }
                }
            } else if let Some(name) = key.strip_prefix("v_") {
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
    fn test_lamb_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            weight_decay: 0.01,
            ..Default::default()
        };
        let mut optimizer = LambOptimizer::new(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);
        optimizer.step(&mut params, &grads).unwrap();
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
    }
}
