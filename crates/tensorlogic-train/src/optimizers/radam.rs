//! RAdam optimizer (Rectified Adam) with variance warmup (ICLR 2020).
//!
//! RAdam addresses the bad convergence problem of Adam in the early stages
//! by rectifying the variance of the adaptive learning rate. It provides
//! a variance warmup mechanism that stabilizes training.
//!
//! Reference: Liu et al. "On the Variance of the Adaptive Learning Rate and Beyond" (ICLR 2020)

use super::common::{compute_gradient_norm, GradClipMode, Optimizer, OptimizerConfig};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// RAdam optimizer (Rectified Adam) with variance warmup (ICLR 2020).
///
/// RAdam addresses the bad convergence problem of Adam in the early stages
/// by rectifying the variance of the adaptive learning rate. It provides
/// a variance warmup mechanism that stabilizes training.
///
/// Reference: Liu et al. "On the Variance of the Adaptive Learning Rate and Beyond" (ICLR 2020)
#[derive(Debug)]
pub struct RAdamOptimizer {
    config: OptimizerConfig,
    /// First moment estimates.
    m: HashMap<String, Array<f64, Ix2>>,
    /// Second moment estimates.
    v: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl RAdamOptimizer {
    /// Create a new RAdam optimizer.
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

    /// Compute the variance rectification term.
    fn compute_rectification(&self) -> (bool, f64) {
        let beta2 = self.config.beta2;
        let t = self.t as f64;
        let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
        let rho_t = rho_inf - 2.0 * t * beta2.powf(t) / (1.0 - beta2.powf(t));
        if rho_t > 5.0 {
            let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t);
            (true, rect.sqrt())
        } else {
            (false, 0.0)
        }
    }
}

impl Optimizer for RAdamOptimizer {
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
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let (use_adaptive, rect) = self.compute_rectification();
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
            let m_hat = &*m / bias_correction1;
            if use_adaptive {
                let bias_correction2 = 1.0 - beta2.powi(self.t as i32);
                let v_hat = &*v / bias_correction2;
                let update = m_hat / (v_hat.mapv(|val| val.sqrt()) + eps);
                *param = &*param - &(update * (lr * rect));
            } else {
                *param = &*param - &(m_hat * lr);
            }
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
        if let Some(t_val) = state.get("t") {
            self.t = t_val[0] as usize;
        }
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m_array) = self.m.get(name) {
                    let shape = m_array.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.m.insert(name.to_string(), arr);
                    }
                }
            } else if let Some(name) = key.strip_prefix("v_") {
                if let Some(v_array) = self.v.get(name) {
                    let shape = v_array.raw_dim();
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
    fn test_radam_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let mut optimizer = RAdamOptimizer::new(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1], [0.1, 0.1]]);
        for _ in 0..10 {
            optimizer.step(&mut params, &grads).unwrap();
        }
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[0, 1]] < 2.0);
        let state = optimizer.state_dict();
        assert!(state.contains_key("t"));
        assert!(state.contains_key("m_w"));
        assert!(state.contains_key("v_w"));
    }
}
