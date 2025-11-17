//! AdaMax optimizer (variant of Adam with infinity norm).
//!
//! AdaMax uses the infinity norm of gradients instead of L2 norm, making it
//! more robust to large gradients and outliers.
//!
//! Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015

use super::common::{compute_gradient_norm, GradClipMode, Optimizer, OptimizerConfig};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// AdaMax optimizer (variant of Adam with infinity norm).
///
/// Uses the infinity norm of gradients instead of L2 norm, making it more robust
/// to large gradients and outliers.
///
/// Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
#[derive(Debug)]
pub struct AdaMaxOptimizer {
    config: OptimizerConfig,
    /// First moment estimates (exponential moving average of gradients).
    m: HashMap<String, Array<f64, Ix2>>,
    /// Exponentially weighted infinity norm.
    u: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl AdaMaxOptimizer {
    /// Create a new AdaMax optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            u: HashMap::new(),
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
}

impl Optimizer for AdaMaxOptimizer {
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
        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.u.insert(name.clone(), Array::zeros(param.raw_dim()));
            }
            let m = self.m.get_mut(name).unwrap();
            let u = self.u.get_mut(name).unwrap();
            *m = &*m * beta1 + &(grad * (1.0 - beta1));
            for i in 0..u.nrows() {
                for j in 0..u.ncols() {
                    u[[i, j]] = (beta2 * u[[i, j]]).max(grad[[i, j]].abs());
                }
            }
            let bias_correction = 1.0 - beta1.powi(self.t as i32);
            let lr_t = lr / bias_correction;
            for i in 0..param.nrows() {
                for j in 0..param.ncols() {
                    let update = lr_t * m[[i, j]] / (u[[i, j]] + self.config.epsilon);
                    param[[i, j]] -= update;
                }
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
        for (name, u_val) in &self.u {
            state.insert(format!("u_{}", name), u_val.iter().copied().collect());
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
            } else if let Some(name) = key.strip_prefix("u_") {
                if let Some(u) = self.u.get(name) {
                    let shape = u.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.u.insert(name.to_string(), arr);
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
    fn test_adamax_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.002,
            ..Default::default()
        };
        let mut optimizer = AdaMaxOptimizer::new(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2], [0.3, 0.4]]);
        for _ in 0..3 {
            optimizer.step(&mut params, &grads).unwrap();
        }
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[0, 1]] < 2.0);
        assert!(w[[1, 0]] < 3.0);
        assert!(w[[1, 1]] < 4.0);
        let state = optimizer.state_dict();
        assert!(state.contains_key("t"));
        assert!(state.contains_key("m_w"));
        assert!(state.contains_key("u_w"));
    }
}
