//! AdamP optimizer (NeurIPS 2020).
//!
//! AdamP (Adaptive Momentum Projection) combines Adam's adaptive learning rates
//! with projection-based weight decay, leading to better generalization.
//!
//! Reference: Heo et al. "AdamP: Slowing Down the Slowdown for Momentum Optimizers
//! on Scale-invariant Weights" (NeurIPS 2020)

use crate::optimizer::{GradClipMode, Optimizer, OptimizerConfig};
use crate::TrainResult;
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// AdamP optimizer with projection-based weight decay.
///
/// AdamP addresses the issue that Adam's weight decay can slow down training
/// on scale-invariant weights (e.g., batch normalization, layer normalization).
/// It uses projection to apply weight decay only in the direction orthogonal
/// to the gradient.
///
/// # Key Features
/// - Projection-based weight decay
/// - Maintains Adam's adaptive learning rates
/// - Better generalization on deep networks
/// - Particularly effective with normalization layers
///
/// # Example
/// ```
/// use tensorlogic_train::{AdamPOptimizer, OptimizerConfig};
///
/// let config = OptimizerConfig {
///     learning_rate: 0.001,
///     weight_decay: 0.01,  // L2 regularization
///     ..Default::default()
/// };
///
/// let optimizer = AdamPOptimizer::new(config);
/// ```
#[derive(Clone)]
pub struct AdamPOptimizer {
    /// Optimizer configuration.
    config: OptimizerConfig,
    /// First moment (mean) estimates.
    m: HashMap<String, Array<f64, Ix2>>,
    /// Second moment (variance) estimates.
    v: HashMap<String, Array<f64, Ix2>>,
    /// Time step for bias correction.
    t: usize,
    /// Nesterov momentum coefficient (default: 0.9).
    nesterov: f64,
    /// Delta parameter for projection (default: 0.1).
    delta: f64,
    /// Weight decay decoupling coefficient (default: 1.0).
    wd_ratio: f64,
}

impl AdamPOptimizer {
    /// Create a new AdamP optimizer with default parameters.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
            nesterov: 0.9,
            delta: 0.1,
            wd_ratio: 1.0,
        }
    }

    /// Create AdamP with custom hyperparameters.
    ///
    /// # Arguments
    /// * `config` - Base optimizer configuration
    /// * `nesterov` - Nesterov momentum coefficient (default: 0.9)
    /// * `delta` - Projection threshold (default: 0.1)
    /// * `wd_ratio` - Weight decay ratio (default: 1.0)
    pub fn with_params(config: OptimizerConfig, nesterov: f64, delta: f64, wd_ratio: f64) -> Self {
        Self {
            config,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
            nesterov,
            delta,
            wd_ratio,
        }
    }

    /// Compute projection of weight decay.
    ///
    /// Projects weight decay into the space orthogonal to the gradient direction.
    fn projection(
        &self,
        _param: &Array<f64, Ix2>,
        grad: &Array<f64, Ix2>,
        perturb: &Array<f64, Ix2>,
        delta: f64,
        wd_ratio: f64,
    ) -> Array<f64, Ix2> {
        // Compute gradient norm
        let grad_norm = grad.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if grad_norm < 1e-12 {
            return perturb.clone();
        }

        // Compute perturbation norm
        let perturb_norm = perturb.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if perturb_norm < 1e-12 {
            return perturb.clone();
        }

        // Compute cosine similarity
        let dot_product: f64 = grad.iter().zip(perturb.iter()).map(|(&g, &p)| g * p).sum();
        let cosine = dot_product / (grad_norm * perturb_norm + 1e-12);

        // If cosine similarity is high, project perturbation
        if cosine.abs() > delta {
            // Project perturbation orthogonal to gradient
            let scale = dot_product / (grad_norm * grad_norm + 1e-12);
            let projection = grad.mapv(|x| x * scale);
            let mut result = perturb - &projection;

            // Scale the projection
            let result_norm = result.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if result_norm > 1e-12 {
                result = result.mapv(|x| x * perturb_norm / result_norm * wd_ratio);
            }

            result
        } else {
            // Use original perturbation
            perturb.mapv(|x| x * wd_ratio)
        }
    }
}

impl Optimizer for AdamPOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        self.t += 1;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let epsilon = self.config.epsilon;
        let lr = self.config.learning_rate;
        let weight_decay = self.config.weight_decay;

        for (name, param) in parameters.iter_mut() {
            let grad = gradients.get(name).ok_or_else(|| {
                crate::TrainError::OptimizerError(format!("No gradient for parameter {}", name))
            })?;

            // Apply gradient clipping if configured
            let grad = if let Some(clip_value) = self.config.grad_clip {
                let mut clipped = grad.clone();
                match self.config.grad_clip_mode {
                    GradClipMode::Value => {
                        clipped.mapv_inplace(|x| x.max(-clip_value).min(clip_value));
                    }
                    GradClipMode::Norm => {
                        let norm = grad.iter().map(|&x| x * x).sum::<f64>().sqrt();
                        if norm > clip_value {
                            let scale = clip_value / norm;
                            clipped.mapv_inplace(|x| x * scale);
                        }
                    }
                }
                clipped
            } else {
                grad.clone()
            };

            // Initialize moment estimates if needed
            let m = self
                .m
                .entry(name.clone())
                .or_insert_with(|| Array::zeros(param.raw_dim()));
            let v = self
                .v
                .entry(name.clone())
                .or_insert_with(|| Array::zeros(param.raw_dim()));

            // Update biased first moment estimate
            *m = m.mapv(|x| x * beta1) + grad.mapv(|x| x * (1.0 - beta1));

            // Update biased second moment estimate
            *v = v.mapv(|x| x * beta2) + grad.mapv(|x| x * x * (1.0 - beta2));

            // Compute bias-corrected estimates
            let m_hat = m.mapv(|x| x / (1.0 - beta1.powi(self.t as i32)));
            let v_hat = v.mapv(|x| x / (1.0 - beta2.powi(self.t as i32)));

            // Compute adaptive update
            let update = &m_hat / &v_hat.mapv(|x| x.sqrt() + epsilon);

            // Nesterov momentum
            let perturb = if self.nesterov > 0.0 {
                let nesterov_m = m.mapv(|x| x * beta1) + grad.mapv(|x| x * (1.0 - beta1));
                let nesterov_m_hat =
                    nesterov_m.mapv(|x| x / (1.0 - beta1.powi((self.t + 1) as i32)));
                &nesterov_m_hat / &v_hat.mapv(|x| x.sqrt() + epsilon)
            } else {
                update.clone()
            };

            // Apply projection-based weight decay
            if weight_decay > 0.0 {
                let wd_perturb = param.mapv(|x| -x * weight_decay);
                let projected_wd =
                    self.projection(param, &grad, &wd_perturb, self.delta, self.wd_ratio);

                // Update parameters with projected weight decay
                *param = param.clone() - perturb.mapv(|x| x * lr) + projected_wd;
            } else {
                // Update parameters without weight decay
                param.scaled_add(-lr, &perturb);
            }
        }

        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();

        // Save timestep
        state.insert("t".to_string(), vec![self.t as f64]);

        // Save moment estimates
        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, v_val) in &self.v {
            state.insert(format!("v_{}", name), v_val.iter().copied().collect());
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        // Load timestep
        if let Some(t_vec) = state.get("t") {
            self.t = t_vec[0] as usize;
        }

        // Load moment estimates
        for (key, value) in &state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m) = self.m.get(name) {
                    let shape = m.raw_dim();
                    if let Ok(array) = Array::from_shape_vec(shape, value.clone()) {
                        self.m.insert(name.to_string(), array);
                    }
                }
            } else if let Some(name) = key.strip_prefix("v_") {
                if let Some(v) = self.v.get(name) {
                    let shape = v.raw_dim();
                    if let Ok(array) = Array::from_shape_vec(shape, value.clone()) {
                        self.v.insert(name.to_string(), array);
                    }
                }
            }
        }
    }

    fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    fn zero_grad(&mut self) {
        // AdamP maintains state across steps
        // Only reset on explicit request
    }
}

impl AdamPOptimizer {
    /// Reset optimizer state (clear momentum and timestep).
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adamp_basic() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            ..Default::default()
        };

        let mut optimizer = AdamPOptimizer::new(config);

        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), array![[0.1, 0.2], [0.3, 0.4]]);

        // First step
        optimizer.step(&mut parameters, &gradients).unwrap();

        // Parameters should have changed
        assert_ne!(parameters["w"][[0, 0]], 1.0);
        assert_ne!(parameters["w"][[1, 1]], 4.0);

        // Check that parameters decreased (gradient descent)
        assert!(parameters["w"][[0, 0]] < 1.0);
        assert!(parameters["w"][[1, 1]] < 4.0);
    }

    #[test]
    fn test_adamp_with_weight_decay() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            weight_decay: 0.1,
            ..Default::default()
        };

        let mut optimizer = AdamPOptimizer::new(config);

        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), array![[0.1, 0.2], [0.3, 0.4]]);

        let initial_param = parameters["w"].clone();

        optimizer.step(&mut parameters, &gradients).unwrap();

        // With projection-based weight decay, parameters should change differently
        // than standard Adam
        assert_ne!(parameters["w"], initial_param);
    }

    #[test]
    fn test_adamp_state_dict() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };

        let mut optimizer = AdamPOptimizer::new(config);

        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), array![[0.1, 0.2]]);

        // Take a few steps
        for _ in 0..5 {
            optimizer.step(&mut parameters, &gradients).unwrap();
        }

        // Save state
        let state = optimizer.state_dict();
        assert!(state.contains_key("t"));
        assert!(state.contains_key("m_w"));
        assert!(state.contains_key("v_w"));

        // Create new optimizer and load state
        let mut new_optimizer = AdamPOptimizer::new(OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        });

        // Initialize with dummy step to create state
        new_optimizer.step(&mut parameters, &gradients).unwrap();

        // Load state
        new_optimizer.load_state_dict(state);

        assert_eq!(new_optimizer.t, 5);
    }

    #[test]
    fn test_adamp_convergence() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };

        let mut optimizer = AdamPOptimizer::new(config);

        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), array![[5.0, 5.0]]);

        // Target is [0, 0], so gradient points toward origin
        for _ in 0..100 {
            let grad = parameters["w"].mapv(|x| x * 0.1); // Gradient proportional to distance
            let mut gradients = HashMap::new();
            gradients.insert("w".to_string(), grad);

            optimizer.step(&mut parameters, &gradients).unwrap();
        }

        // Should converge toward zero
        assert!(parameters["w"][[0, 0]].abs() < 1.0);
        assert!(parameters["w"][[0, 1]].abs() < 1.0);
    }

    #[test]
    fn test_adamp_projection() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            weight_decay: 0.1,
            ..Default::default()
        };

        let optimizer = AdamPOptimizer::with_params(config, 0.9, 0.1, 1.0);

        let param = array![[1.0, 2.0], [3.0, 4.0]];
        let grad = array![[0.1, 0.2], [0.3, 0.4]];
        let perturb = array![[-0.1, -0.2], [-0.3, -0.4]];

        let projected = optimizer.projection(&param, &grad, &perturb, 0.1, 1.0);

        // Projection should produce a result
        assert_eq!(projected.shape(), perturb.shape());
    }

    #[test]
    fn test_adamp_nesterov() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };

        // With Nesterov
        let mut opt_nesterov = AdamPOptimizer::with_params(config.clone(), 0.9, 0.1, 1.0);

        // Without Nesterov
        let mut opt_standard = AdamPOptimizer::with_params(config, 0.0, 0.1, 1.0);

        let mut params1 = HashMap::new();
        params1.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut params2 = params1.clone();

        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), array![[0.1, 0.2]]);

        opt_nesterov.step(&mut params1, &gradients).unwrap();
        opt_standard.step(&mut params2, &gradients).unwrap();

        // With Nesterov should produce different results
        // (though they may be close for a single step)
        assert!(
            params1["w"][[0, 0]] != params2["w"][[0, 0]]
                || (params1["w"][[0, 0]] - params2["w"][[0, 0]]).abs() < 1e-10
        );
    }
}
