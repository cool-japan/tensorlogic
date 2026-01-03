//! AdaBelief optimizer (NeurIPS 2020).
//!
//! AdaBelief adapts the step size according to the "belief" in the gradient direction.
//! It uses the variance of gradients (belief) to adapt the learning rate, which can
//! achieve faster convergence and better generalization than Adam/AdamW.
//!
//! Reference: Zhuang et al. "AdaBelief Optimizer: Adapting Stepsizes by the Belief
//! in Observed Gradients" (NeurIPS 2020)

use super::common::{compute_gradient_norm, GradClipMode, Optimizer, OptimizerConfig};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// AdaBelief optimizer (NeurIPS 2020).
///
/// AdaBelief adapts the step size according to the "belief" in the gradient direction.
/// It uses the variance of gradients (belief) to adapt the learning rate, which can
/// achieve faster convergence and better generalization than Adam/AdamW.
///
/// Reference: Zhuang et al. "AdaBelief Optimizer: Adapting Stepsizes by the Belief
/// in Observed Gradients" (NeurIPS 2020)
#[derive(Debug)]
pub struct AdaBeliefOptimizer {
    config: OptimizerConfig,
    /// First moment estimates (exponential moving average of gradients).
    m: HashMap<String, Array<f64, Ix2>>,
    /// Second moment estimates (variance of gradients).
    s: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter.
    t: usize,
}

impl AdaBeliefOptimizer {
    /// Create a new AdaBelief optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            s: HashMap::new(),
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

impl Optimizer for AdaBeliefOptimizer {
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
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32);
        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.s.insert(name.clone(), Array::zeros(param.raw_dim()));
            }
            let m = self.m.get_mut(name).unwrap();
            let s = self.s.get_mut(name).unwrap();
            *m = &*m * beta1 + &(grad * (1.0 - beta1));
            let grad_diff = grad - &*m;
            let grad_diff_squared = grad_diff.mapv(|g| g * g);
            *s = &*s * beta2 + &(grad_diff_squared * (1.0 - beta2));
            let m_hat = &*m / bias_correction1;
            let s_hat = &*s / bias_correction2;
            if weight_decay > 0.0 {
                param.mapv_inplace(|p| p * (1.0 - lr * weight_decay));
            }
            let update = m_hat / (s_hat.mapv(|v| v.sqrt()) + eps);
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
        state.insert("t".to_string(), vec![self.t as f64]);
        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, s_val) in &self.s {
            state.insert(format!("s_{}", name), s_val.iter().copied().collect());
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
            } else if let Some(name) = key.strip_prefix("s_") {
                if let Some(s_array) = self.s.get(name) {
                    let shape = s_array.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.s.insert(name.to_string(), arr);
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
    fn test_adabelief_optimizer() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            weight_decay: 0.01,
            ..Default::default()
        };
        let mut optimizer = AdaBeliefOptimizer::new(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2], [0.3, 0.4]]);
        for _ in 0..5 {
            optimizer.step(&mut params, &grads).unwrap();
        }
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[1, 1]] < 4.0);
        let state = optimizer.state_dict();
        assert!(state.contains_key("t"));
        assert!(state.contains_key("m_w"));
        assert!(state.contains_key("s_w"));
    }
}
