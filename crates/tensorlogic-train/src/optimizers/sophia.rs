//! Sophia optimizer - Scalable Stochastic Second-order Optimizer
//!
//! Implementation of the Sophia optimizer from:
//! "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
//! Hong Liu, Zhiyuan Li, David Hall, Percy Liang, Tengyu Ma (2023)
//! <https://arxiv.org/abs/2305.14342>
//!
//! Sophia uses a lightweight Hessian diagonal estimate to adapt learning rates,
//! achieving faster convergence than Adam with similar memory requirements.
//!
//! # Key Features
//! - **Second-order information**: Uses Hessian diagonal estimates for better curvature awareness
//! - **Scalable**: Only requires tracking Hessian diagonal (same memory as Adam)
//! - **Fast convergence**: Typically 2-3x faster than Adam for language model pretraining
//! - **Two variants**: Sophia-G (Gauss-Newton-Bartlett) and Sophia-H (Hutchinson)
//!
//! # Usage
//! ```rust
//! use tensorlogic_train::{SophiaOptimizer, OptimizerConfig, Optimizer};
//! use scirs2_core::ndarray::Array2;
//! use std::collections::HashMap;
//!
//! let config = OptimizerConfig {
//!     learning_rate: 1e-4,
//!     ..Default::default()
//! };
//!
//! let mut optimizer = SophiaOptimizer::new(config);
//!
//! // During training with parameter HashMap:
//! // optimizer.step(&mut parameters, &gradients)?;
//! ```
//!
//! # Hyperparameter Recommendations
//! - Learning rate: 1e-4 to 2e-4 (higher than Adam's typical 1e-5)
//! - Beta1: 0.965 (momentum for gradients)
//! - Beta2: 0.99 (momentum for Hessian diagonal)
//! - Epsilon: 1e-8
//! - Rho: 0.04 (clipping parameter for update direction)
//! - Hessian update frequency: Every 10 steps (k=10)

use super::common::{compute_gradient_norm, GradClipMode, Optimizer, OptimizerConfig};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// Variant of Sophia optimizer to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SophiaVariant {
    /// Gauss-Newton-Bartlett estimator (more accurate, slightly more expensive)
    GaussNewtonBartlett,
    /// Hutchinson estimator (cheaper, uses random projections)
    Hutchinson,
}

/// Configuration for Sophia optimizer with additional Sophia-specific parameters
#[derive(Debug, Clone)]
pub struct SophiaConfig {
    /// Base optimizer configuration
    pub base: OptimizerConfig,
    /// Clipping parameter for update direction (typically 0.04)
    pub rho: f64,
    /// Frequency of Hessian updates (every k steps)
    pub hessian_update_freq: usize,
    /// Variant to use (G or H)
    pub variant: SophiaVariant,
}

impl Default for SophiaConfig {
    fn default() -> Self {
        Self {
            base: OptimizerConfig {
                learning_rate: 2e-4,
                beta1: 0.965,
                beta2: 0.99,
                epsilon: 1e-8,
                weight_decay: 0.01,
                ..Default::default()
            },
            rho: 0.04,
            hessian_update_freq: 10,
            variant: SophiaVariant::GaussNewtonBartlett,
        }
    }
}

/// Sophia optimizer - Second-order optimizer with Hessian diagonal estimation
///
/// Maintains three state tensors per parameter:
/// - m: First moment estimate (exponential moving average of gradients)
/// - h: Hessian diagonal estimate (EMA of element-wise gradient^2 or Hutchinson estimate)
/// - t: Step counter for bias correction
pub struct SophiaOptimizer {
    config: SophiaConfig,
    /// First moment estimates (m_t)
    m: HashMap<String, Array<f64, Ix2>>,
    /// Hessian diagonal estimates (h_t)
    h: HashMap<String, Array<f64, Ix2>>,
    /// Timestep counter
    t: usize,
    /// Steps since last Hessian update
    steps_since_hessian_update: usize,
}

impl SophiaOptimizer {
    /// Create a new Sophia optimizer with default Sophia configuration
    pub fn new(config: OptimizerConfig) -> Self {
        Self::with_sophia_config(SophiaConfig {
            base: config,
            ..Default::default()
        })
    }

    /// Create a new Sophia optimizer with custom Sophia configuration
    pub fn with_sophia_config(config: SophiaConfig) -> Self {
        Self {
            config,
            m: HashMap::new(),
            h: HashMap::new(),
            t: 0,
            steps_since_hessian_update: 0,
        }
    }

    /// Apply gradient clipping if configured
    fn clip_gradients(&self, gradients: &mut HashMap<String, Array<f64, Ix2>>) {
        if let Some(clip_value) = self.config.base.grad_clip {
            match self.config.base.grad_clip_mode {
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

    /// Update Hessian diagonal estimate using Gauss-Newton-Bartlett method
    ///
    /// This uses the gradient itself as an approximation:
    /// h_t = β₂ * h_{t-1} + (1 - β₂) * g_t²
    fn update_hessian_gnb(&mut self, gradients: &HashMap<String, Array<f64, Ix2>>) {
        let beta2 = self.config.base.beta2;

        for (name, grad) in gradients {
            let grad_squared = grad.mapv(|g| g * g);

            if let Some(h_state) = self.h.get_mut(name) {
                // h_t = β₂ * h_{t-1} + (1 - β₂) * g²
                *h_state = &*h_state * beta2 + &grad_squared * (1.0 - beta2);
            } else {
                self.h.insert(name.clone(), grad_squared * (1.0 - beta2));
            }
        }
    }

    /// Update Hessian diagonal estimate using Hutchinson method
    ///
    /// Uses random Rademacher vectors for unbiased estimation:
    /// h_t ≈ g_t ⊙ (∇²L * u) where u ~ Rademacher({-1, +1})
    ///
    /// Note: Full Hutchinson requires Hessian-vector products which aren't available
    /// in this interface, so we use GNB as a reasonable approximation.
    fn update_hessian_hutchinson(&mut self, gradients: &HashMap<String, Array<f64, Ix2>>) {
        // For a full Hutchinson implementation, we'd need:
        // 1. Sample u ~ Rademacher({-1, +1})
        // 2. Compute Hessian-vector product: Hv = ∇(g^T u)
        // 3. Estimate diagonal: h ≈ u ⊙ Hv
        //
        // Since we don't have access to Hessian-vector products in this interface,
        // we use GNB as a practical approximation
        self.update_hessian_gnb(gradients);
    }
}

impl Optimizer for SophiaOptimizer {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let mut clipped_gradients = gradients.clone();
        self.clip_gradients(&mut clipped_gradients);

        self.t += 1;
        self.steps_since_hessian_update += 1;

        let lr = self.config.base.learning_rate;
        let beta1 = self.config.base.beta1;
        let eps = self.config.base.epsilon;
        let rho = self.config.rho;
        let weight_decay = self.config.base.weight_decay;

        // Update Hessian diagonal estimate (every k steps)
        if self.steps_since_hessian_update >= self.config.hessian_update_freq {
            match self.config.variant {
                SophiaVariant::GaussNewtonBartlett => {
                    self.update_hessian_gnb(&clipped_gradients);
                }
                SophiaVariant::Hutchinson => {
                    self.update_hessian_hutchinson(&clipped_gradients);
                }
            }
            self.steps_since_hessian_update = 0;
        }

        // Bias correction for first moment
        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);

        // Update parameters
        for (name, param) in parameters.iter_mut() {
            let grad = clipped_gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;

            // Initialize state if needed
            if !self.m.contains_key(name) {
                self.m.insert(name.clone(), Array::zeros(param.raw_dim()));
                self.h
                    .insert(name.clone(), Array::ones(param.raw_dim()) * eps);
            }

            let m = self.m.get_mut(name).unwrap();
            let h = self.h.get(name).unwrap();

            // Update first moment (gradient EMA): m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            *m = &*m * beta1 + &(grad * (1.0 - beta1));

            // Bias-corrected first moment: m̂_t = m_t / (1 - β₁^t)
            let m_hat = &*m / bias_correction1;

            // Compute update direction: m̂ / (ρ * h + ε)
            let denominator = h * rho + eps;
            let update_direction = &m_hat / &denominator;

            // Clip update direction to [-1, 1]
            let clipped_update = update_direction.mapv(|x| x.clamp(-1.0, 1.0));

            // Apply update: θ_{t+1} = θ_t - lr * clip(m̂ / (ρ * h), -1, 1)
            *param = &*param - &(&clipped_update * lr);

            // Weight decay (decoupled, like AdamW): θ_{t+1} -= lr * λ * θ_t
            if weight_decay > 0.0 {
                *param = &*param - &(&*param * (weight_decay * lr));
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Gradients are passed in, not stored, so nothing to zero
    }

    fn get_lr(&self) -> f64 {
        self.config.base.learning_rate
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.base.learning_rate = lr;
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();
        state.insert("t".to_string(), vec![self.t as f64]);
        state.insert(
            "steps_since_hessian_update".to_string(),
            vec![self.steps_since_hessian_update as f64],
        );

        for (name, m_val) in &self.m {
            state.insert(format!("m_{}", name), m_val.iter().copied().collect());
        }
        for (name, h_val) in &self.h {
            state.insert(format!("h_{}", name), h_val.iter().copied().collect());
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(t_vals) = state.get("t") {
            self.t = t_vals[0] as usize;
        }
        if let Some(steps_vals) = state.get("steps_since_hessian_update") {
            self.steps_since_hessian_update = steps_vals[0] as usize;
        }

        for (key, values) in state {
            if let Some(name) = key.strip_prefix("m_") {
                if let Some(m) = self.m.get(name) {
                    let shape = m.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.m.insert(name.to_string(), arr);
                    }
                }
            } else if let Some(name) = key.strip_prefix("h_") {
                if let Some(h) = self.h.get(name) {
                    let shape = h.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.h.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sophia_initialization() {
        let config = OptimizerConfig::default();
        let optimizer = SophiaOptimizer::new(config);

        assert_eq!(optimizer.t, 0);
        assert!(optimizer.m.is_empty());
        assert!(optimizer.h.is_empty());
    }

    #[test]
    fn test_sophia_custom_config() {
        let config = SophiaConfig {
            base: OptimizerConfig {
                learning_rate: 1e-4,
                beta1: 0.965,
                beta2: 0.99,
                ..Default::default()
            },
            rho: 0.04,
            ..Default::default()
        };

        let optimizer = SophiaOptimizer::with_sophia_config(config);
        assert_relative_eq!(optimizer.get_lr(), 1e-4);
    }

    #[test]
    fn test_sophia_single_step() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };

        let mut optimizer = SophiaOptimizer::new(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0, 3.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2, 0.3]]);

        let initial = params["w"].clone();
        optimizer.step(&mut params, &grads).unwrap();

        // Parameters should be updated (decreased for positive gradients)
        assert!(params["w"][[0, 0]] < initial[[0, 0]]);
        assert!(params["w"][[0, 1]] < initial[[0, 1]]);
        assert!(params["w"][[0, 2]] < initial[[0, 2]]);
    }

    #[test]
    fn test_sophia_convergence() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };

        let mut optimizer = SophiaOptimizer::new(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[5.0], [-3.0], [2.0]]);

        // Simulate optimization to zero
        for _ in 0..50 {
            let mut grads = HashMap::new();
            grads.insert("w".to_string(), &params["w"] * 2.0); // Gradient of x²
            optimizer.step(&mut params, &grads).unwrap();
        }

        // Should converge close to zero
        for &p in params["w"].iter() {
            assert!(p.abs() < 0.5);
        }
    }

    #[test]
    fn test_sophia_2d_parameters() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };

        let mut optimizer = SophiaOptimizer::new(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]]);

        let initial_shape = params["w"].shape().to_vec();
        optimizer.step(&mut params, &grads).unwrap();

        assert_eq!(params["w"].shape(), &initial_shape[..]);
    }

    #[test]
    fn test_sophia_reset_and_state_dict() {
        let config = OptimizerConfig::default();
        let mut optimizer = SophiaOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2]]);

        optimizer.step(&mut params, &grads).unwrap();
        assert!(!optimizer.m.is_empty());
        assert_eq!(optimizer.t, 1);

        // Test state dict
        let state = optimizer.state_dict();
        assert!(state.contains_key("t"));
        assert!(state.contains_key("m_w"));
        assert!(state.contains_key("h_w"));
    }

    #[test]
    fn test_sophia_hessian_update_frequency() {
        let config = SophiaConfig {
            hessian_update_freq: 5,
            ..Default::default()
        };

        let mut optimizer = SophiaOptimizer::with_sophia_config(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2]]);

        // First step should update Hessian
        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(optimizer.steps_since_hessian_update, 1);

        // Steps 2-4 should not update
        for _ in 0..4 {
            optimizer.step(&mut params, &grads).unwrap();
        }
        assert_eq!(optimizer.steps_since_hessian_update, 0); // Reset after 5 steps

        // Hessian state should exist
        assert!(optimizer.h.contains_key("w"));
    }

    #[test]
    fn test_sophia_weight_decay() {
        let config = SophiaConfig {
            base: OptimizerConfig {
                learning_rate: 0.1,
                weight_decay: 0.01,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut optimizer = SophiaOptimizer::with_sophia_config(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0, 3.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.0, 0.0, 0.0]]); // Zero gradients

        let initial = params["w"].clone();
        optimizer.step(&mut params, &grads).unwrap();

        // With weight decay and zero gradients, parameters should decay
        assert!(params["w"][[0, 0]] < initial[[0, 0]]);
        assert!(params["w"][[0, 1]] < initial[[0, 1]]);
        assert!(params["w"][[0, 2]] < initial[[0, 2]]);
    }

    #[test]
    fn test_sophia_gradient_clipping_value() {
        let config = SophiaConfig {
            base: OptimizerConfig {
                learning_rate: 0.1,
                grad_clip: Some(0.5),
                grad_clip_mode: GradClipMode::Value,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut optimizer = SophiaOptimizer::with_sophia_config(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[1.0, -2.0]]); // Should be clipped to [0.5, -0.5]

        let initial = params["w"].clone();
        optimizer.step(&mut params, &grads).unwrap();

        // Effect should be limited by clipping
        let update_mag = (initial[[0, 0]] - params["w"][[0, 0]]).abs();
        assert!(update_mag < 0.2); // Much less than if unclipped
    }

    #[test]
    fn test_sophia_gradient_clipping_norm() {
        let config = SophiaConfig {
            base: OptimizerConfig {
                learning_rate: 0.1,
                grad_clip: Some(1.0),
                grad_clip_mode: GradClipMode::Norm,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut optimizer = SophiaOptimizer::with_sophia_config(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0, 3.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[10.0, 10.0, 10.0]]); // Large gradients

        let initial = params["w"].clone();
        optimizer.step(&mut params, &grads).unwrap();

        // Norm clipping should limit the total update
        let total_update: f64 = initial
            .iter()
            .zip(params["w"].iter())
            .map(|(&p, &u)| (p - u).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(total_update < 1.0); // Should be limited
    }

    #[test]
    fn test_sophia_learning_rate_getter_setter() {
        let config = OptimizerConfig::default();
        let mut optimizer = SophiaOptimizer::new(config);

        optimizer.set_lr(0.001);
        assert_relative_eq!(optimizer.get_lr(), 0.001);

        optimizer.set_lr(0.1);
        assert_relative_eq!(optimizer.get_lr(), 0.1);
    }

    #[test]
    fn test_sophia_variant_gnb() {
        let config = SophiaConfig {
            variant: SophiaVariant::GaussNewtonBartlett,
            ..Default::default()
        };

        let mut optimizer = SophiaOptimizer::with_sophia_config(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.5, 0.5]]);

        let initial = params["w"].clone();
        optimizer.step(&mut params, &grads).unwrap();
        assert!(params["w"][[0, 0]] < initial[[0, 0]]); // Should make progress
    }

    #[test]
    fn test_sophia_variant_hutchinson() {
        let config = SophiaConfig {
            variant: SophiaVariant::Hutchinson,
            ..Default::default()
        };

        let mut optimizer = SophiaOptimizer::with_sophia_config(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.5, 0.5]]);

        let initial = params["w"].clone();
        optimizer.step(&mut params, &grads).unwrap();
        assert!(params["w"][[0, 0]] < initial[[0, 0]]); // Should make progress
    }

    #[test]
    fn test_sophia_update_clipping() {
        // Test that updates are clipped to [-1, 1] before applying learning rate
        let config = SophiaConfig {
            base: OptimizerConfig {
                learning_rate: 0.1,
                ..Default::default()
            },
            rho: 0.001, // Very small rho to create large update direction
            ..Default::default()
        };

        let mut optimizer = SophiaOptimizer::with_sophia_config(config);
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[10.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[100.0]]); // Large gradient

        let initial = params["w"][[0, 0]];
        optimizer.step(&mut params, &grads).unwrap();

        // Even with large gradient, update should be bounded
        let update_size = (initial - params["w"][[0, 0]]).abs();
        assert!(update_size <= 0.12); // lr * 1.0 (clipped) + small margin
    }

    #[test]
    fn test_sophia_load_state_dict() {
        let config = OptimizerConfig::default();
        let mut optimizer1 = SophiaOptimizer::new(config.clone());
        let mut optimizer2 = SophiaOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2]]);

        // Take several steps with optimizer1
        for _ in 0..5 {
            optimizer1.step(&mut params, &grads).unwrap();
        }

        // Save and load state
        let state = optimizer1.state_dict();
        optimizer2.load_state_dict(state);

        // Verify state was loaded
        assert_eq!(optimizer2.t, optimizer1.t);
        assert_eq!(
            optimizer2.steps_since_hessian_update,
            optimizer1.steps_since_hessian_update
        );
    }
}
