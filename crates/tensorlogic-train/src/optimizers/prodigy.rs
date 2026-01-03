//! Prodigy optimizer - Auto-tuning learning rate optimizer (2024)
//!
//! Reference: Mishchenko, K., & Defazio, A. (2024).
//! "Prodigyopt: An Adaptive Learning Rate Method That Requires No Manual Tuning"
//! <https://arxiv.org/abs/2306.06101>
//!
//! Key innovation: Automatically estimates the learning rate scale (D) without manual tuning.
//! Uses distance from initialization to estimate appropriate step size.
//!
//! Benefits:
//! - No manual LR tuning required
//! - Works across different problem scales
//! - Adaptive to problem difficulty
//! - Combines benefits of Adam and D-Adaptation
//!
//! Usage:
//! ```rust
//! use tensorlogic_train::{ProdigyConfig, ProdigyOptimizer};
//!
//! // Create Prodigy optimizer with defaults
//! let config = ProdigyConfig::default()
//!     .with_d0(1e-6)      // Initial D estimate (small value)
//!     .with_d_coef(1.0)   // Coefficient for D adaptation
//!     .with_beta1(0.9)    // First moment decay
//!     .with_beta2(0.999); // Second moment decay
//!
//! let mut optimizer = ProdigyOptimizer::new(config);
//!
//! // Prodigy automatically adapts the learning rate!
//! // No need to manually tune LR or use schedulers
//! ```

use crate::error::TrainResult;
use crate::optimizer::{GradClipMode, Optimizer};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Prodigy optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProdigyConfig {
    /// Initial D estimate (default: 1e-6)
    /// D represents the distance scale from initialization
    pub d0: f64,

    /// Coefficient for D adaptation (default: 1.0)
    /// Controls how aggressively D is adapted
    pub d_coef: f64,

    /// Learning rate (default: 1.0)
    /// Note: Prodigy is relatively insensitive to this value
    pub lr: f64,

    /// First moment decay rate (default: 0.9)
    pub beta1: f64,

    /// Second moment decay rate (default: 0.999)
    pub beta2: f64,

    /// Small constant for numerical stability (default: 1e-8)
    pub eps: f64,

    /// Weight decay coefficient (default: 0.0)
    /// Applied as decoupled weight decay (like AdamW)
    pub weight_decay: f64,

    /// Gradient clipping threshold (optional)
    pub grad_clip: Option<f64>,

    /// Gradient clipping mode
    pub grad_clip_mode: GradClipMode,

    /// Whether to use bias correction (default: true)
    pub bias_correction: bool,

    /// Growth rate for D (default: infinity, meaning no limit)
    pub d_growth_rate: f64,
}

impl Default for ProdigyConfig {
    fn default() -> Self {
        Self {
            d0: 1e-6,
            d_coef: 1.0,
            lr: 1.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            grad_clip: None,
            grad_clip_mode: GradClipMode::Norm,
            bias_correction: true,
            d_growth_rate: f64::INFINITY,
        }
    }
}

impl ProdigyConfig {
    /// Create new Prodigy config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set initial D estimate
    pub fn with_d0(mut self, d0: f64) -> Self {
        self.d0 = d0;
        self
    }

    /// Set D coefficient
    pub fn with_d_coef(mut self, d_coef: f64) -> Self {
        self.d_coef = d_coef;
        self
    }

    /// Set learning rate
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set beta1 (first moment decay)
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (second moment decay)
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set gradient clipping
    pub fn with_grad_clip(mut self, grad_clip: f64) -> Self {
        self.grad_clip = Some(grad_clip);
        self
    }

    /// Set gradient clipping mode
    pub fn with_grad_clip_mode(mut self, mode: GradClipMode) -> Self {
        self.grad_clip_mode = mode;
        self
    }

    /// Enable or disable bias correction
    pub fn with_bias_correction(mut self, bias_correction: bool) -> Self {
        self.bias_correction = bias_correction;
        self
    }

    /// Set D growth rate limit
    pub fn with_d_growth_rate(mut self, rate: f64) -> Self {
        self.d_growth_rate = rate;
        self
    }
}

/// Prodigy optimizer
///
/// Auto-tuning adaptive learning rate optimizer that estimates the distance scale D
/// from initialization to automatically set appropriate step sizes.
///
/// Key features:
/// - No manual LR tuning needed
/// - Adapts to problem scale automatically
/// - Combines Adam-style updates with D-Adaptation
/// - Maintains first and second moment estimates
pub struct ProdigyOptimizer {
    config: ProdigyConfig,
    /// First moment estimates (momentum)
    first_moments: HashMap<String, Array2<f64>>,
    /// Second moment estimates (variance)
    second_moments: HashMap<String, Array2<f64>>,
    /// Initial parameters (for distance computation)
    initial_params: HashMap<String, Array2<f64>>,
    /// Current step count
    step: usize,
    /// Estimated distance scale D
    d: f64,
    /// Sum of gradient norms (for D estimation)
    sum_grad_norm: f64,
}

impl ProdigyOptimizer {
    /// Create new Prodigy optimizer with given config
    pub fn new(config: ProdigyConfig) -> Self {
        Self {
            config,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            initial_params: HashMap::new(),
            step: 0,
            d: 0.0, // Will be initialized to d0 on first step
            sum_grad_norm: 0.0,
        }
    }

    /// Get current D estimate
    pub fn get_d(&self) -> f64 {
        self.d
    }

    /// Get current step count
    pub fn get_step(&self) -> usize {
        self.step
    }

    /// Compute parameter distance from initialization
    fn compute_distance(&self, parameters: &HashMap<String, Array2<f64>>) -> f64 {
        let mut distance_sq = 0.0;

        for (name, param) in parameters {
            if let Some(init_param) = self.initial_params.get(name) {
                let diff = param - init_param;
                distance_sq += diff.mapv(|x| x * x).sum();
            }
        }

        distance_sq.sqrt()
    }

    /// Update D estimate based on gradients and parameters
    fn update_d(&mut self, parameters: &HashMap<String, Array2<f64>>, grad_norm: f64) {
        // Initialize D on first step
        if self.step == 1 {
            self.d = self.config.d0;
            return;
        }

        // Accumulate gradient norms
        self.sum_grad_norm += grad_norm;

        // Compute parameter distance from initialization
        let param_distance = self.compute_distance(parameters);

        // Estimate D based on ratio of distance to accumulated gradient norm
        if self.sum_grad_norm > 0.0 {
            let d_estimate = self.config.d_coef * param_distance / self.sum_grad_norm;

            // Apply growth rate limit if specified
            if self.config.d_growth_rate.is_finite() {
                let max_d = self.d * (1.0 + self.config.d_growth_rate);
                self.d = d_estimate.min(max_d).max(self.config.d0);
            } else {
                self.d = d_estimate.max(self.config.d0);
            }
        }
    }

    /// Compute total gradient norm
    fn compute_gradient_norm(&self, gradients: &HashMap<String, Array2<f64>>) -> f64 {
        let mut norm_sq = 0.0;
        for grad in gradients.values() {
            norm_sq += grad.mapv(|x| x * x).sum();
        }
        norm_sq.sqrt()
    }

    /// Apply gradient clipping
    fn clip_gradients(
        &self,
        gradients: &mut HashMap<String, Array2<f64>>,
        _rng: Option<&mut StdRng>,
    ) -> TrainResult<()> {
        if let Some(max_val) = self.config.grad_clip {
            match self.config.grad_clip_mode {
                GradClipMode::Value => {
                    // Clip by value
                    for grad in gradients.values_mut() {
                        grad.mapv_inplace(|x| x.max(-max_val).min(max_val));
                    }
                }
                GradClipMode::Norm => {
                    // Clip by norm
                    let total_norm = self.compute_gradient_norm(gradients);
                    if total_norm > max_val {
                        let scale = max_val / (total_norm + self.config.eps);
                        for grad in gradients.values_mut() {
                            grad.mapv_inplace(|x| x * scale);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Optimizer for ProdigyOptimizer {
    fn zero_grad(&mut self) {
        // Prodigy doesn't maintain gradients separately, so this is a no-op
    }

    fn get_lr(&self) -> f64 {
        self.config.lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array2<f64>>,
        gradients: &HashMap<String, Array2<f64>>,
    ) -> TrainResult<()> {
        // Increment step counter
        self.step += 1;

        // Save initial parameters on first step
        if self.step == 1 {
            for (name, param) in parameters.iter() {
                self.initial_params.insert(name.clone(), param.clone());
            }
        }

        // Clone gradients if clipping is needed
        let gradients = if self.config.grad_clip.is_some() {
            let mut clipped = HashMap::new();
            for (name, grad) in gradients.iter() {
                clipped.insert(name.clone(), grad.clone());
            }
            self.clip_gradients(&mut clipped, None)?;
            clipped
        } else {
            gradients.clone()
        };

        // Compute gradient norm for D estimation
        let grad_norm = self.compute_gradient_norm(&gradients);

        // Update D estimate
        self.update_d(parameters, grad_norm);

        // Compute effective learning rate
        let effective_lr = self.config.lr * self.d;

        // Bias correction factors (if enabled)
        let bias_correction1 = if self.config.bias_correction {
            1.0 - self.config.beta1.powi(self.step as i32)
        } else {
            1.0
        };
        let bias_correction2 = if self.config.bias_correction {
            1.0 - self.config.beta2.powi(self.step as i32)
        } else {
            1.0
        };

        // Update parameters
        for (name, param) in parameters.iter_mut() {
            let grad = match gradients.get(name) {
                Some(g) => g,
                None => continue,
            };

            // Initialize moments if needed
            let m = self
                .first_moments
                .entry(name.clone())
                .or_insert_with(|| Array2::zeros(grad.raw_dim()));
            let v = self
                .second_moments
                .entry(name.clone())
                .or_insert_with(|| Array2::zeros(grad.raw_dim()));

            // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            *m = &*m * self.config.beta1 + grad * (1.0 - self.config.beta1);

            // Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            let grad_sq = grad.mapv(|x| x * x);
            *v = &*v * self.config.beta2 + &grad_sq * (1.0 - self.config.beta2);

            // Compute bias-corrected moments
            let m_hat = m.mapv(|x| x / bias_correction1);
            let v_hat = v.mapv(|x| x / bias_correction2);

            // Compute update: delta = lr * D * m_hat / (sqrt(v_hat) + eps)
            let update = &m_hat / &v_hat.mapv(|x| x.sqrt() + self.config.eps);

            // Apply weight decay (decoupled, like AdamW)
            if self.config.weight_decay > 0.0 {
                param.mapv_inplace(|x| x * (1.0 - effective_lr * self.config.weight_decay));
            }

            // Update parameters: theta_t = theta_{t-1} - lr * D * update
            *param = &*param - &update * effective_lr;
        }

        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();

        // Save scalar values
        state.insert("step".to_string(), vec![self.step as f64]);
        state.insert("d".to_string(), vec![self.d]);
        state.insert("sum_grad_norm".to_string(), vec![self.sum_grad_norm]);

        // Save config values
        state.insert("d0".to_string(), vec![self.config.d0]);
        state.insert("d_coef".to_string(), vec![self.config.d_coef]);
        state.insert("lr".to_string(), vec![self.config.lr]);
        state.insert("beta1".to_string(), vec![self.config.beta1]);
        state.insert("beta2".to_string(), vec![self.config.beta2]);
        state.insert("eps".to_string(), vec![self.config.eps]);
        state.insert("weight_decay".to_string(), vec![self.config.weight_decay]);

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        // Load scalar values
        if let Some(v) = state.get("step") {
            if !v.is_empty() {
                self.step = v[0] as usize;
            }
        }
        if let Some(v) = state.get("d") {
            if !v.is_empty() {
                self.d = v[0];
            }
        }
        if let Some(v) = state.get("sum_grad_norm") {
            if !v.is_empty() {
                self.sum_grad_norm = v[0];
            }
        }

        // Load config values
        if let Some(v) = state.get("d0") {
            if !v.is_empty() {
                self.config.d0 = v[0];
            }
        }
        if let Some(v) = state.get("d_coef") {
            if !v.is_empty() {
                self.config.d_coef = v[0];
            }
        }
        if let Some(v) = state.get("lr") {
            if !v.is_empty() {
                self.config.lr = v[0];
            }
        }
        if let Some(v) = state.get("beta1") {
            if !v.is_empty() {
                self.config.beta1 = v[0];
            }
        }
        if let Some(v) = state.get("beta2") {
            if !v.is_empty() {
                self.config.beta2 = v[0];
            }
        }
        if let Some(v) = state.get("eps") {
            if !v.is_empty() {
                self.config.eps = v[0];
            }
        }
        if let Some(v) = state.get("weight_decay") {
            if !v.is_empty() {
                self.config.weight_decay = v[0];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prodigy_config_default() {
        let config = ProdigyConfig::default();
        assert_eq!(config.d0, 1e-6);
        assert_eq!(config.d_coef, 1.0);
        assert_eq!(config.lr, 1.0);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.eps, 1e-8);
        assert_eq!(config.weight_decay, 0.0);
    }

    #[test]
    fn test_prodigy_config_builder() {
        let config = ProdigyConfig::default()
            .with_d0(1e-5)
            .with_d_coef(2.0)
            .with_lr(0.5)
            .with_beta1(0.95)
            .with_beta2(0.9999)
            .with_eps(1e-7)
            .with_weight_decay(0.01)
            .with_grad_clip(1.0)
            .with_bias_correction(false)
            .with_d_growth_rate(0.1);

        assert_eq!(config.d0, 1e-5);
        assert_eq!(config.d_coef, 2.0);
        assert_eq!(config.lr, 0.5);
        assert_eq!(config.beta1, 0.95);
        assert_eq!(config.beta2, 0.9999);
        assert_eq!(config.eps, 1e-7);
        assert_eq!(config.weight_decay, 0.01);
        assert_eq!(config.grad_clip, Some(1.0));
        assert!(!config.bias_correction);
        assert_eq!(config.d_growth_rate, 0.1);
    }

    #[test]
    fn test_prodigy_initialization() {
        let config = ProdigyConfig::default();
        let optimizer = ProdigyOptimizer::new(config);

        assert_eq!(optimizer.get_step(), 0);
        assert_eq!(optimizer.get_d(), 0.0);
    }

    #[test]
    fn test_prodigy_first_step() {
        let config = ProdigyConfig::default().with_d0(1e-6);
        let mut optimizer = ProdigyOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array2::from_elem((2, 2), 1.0));

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Array2::from_elem((2, 2), 0.1));

        optimizer.step(&mut params, &grads).unwrap();

        assert_eq!(optimizer.get_step(), 1);
        assert_eq!(optimizer.get_d(), 1e-6); // D initialized to d0
    }

    #[test]
    fn test_prodigy_d_adaptation() {
        let config = ProdigyConfig::default().with_d0(1e-6).with_d_coef(1.0);
        let mut optimizer = ProdigyOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array2::from_elem((2, 2), 1.0));

        // First step
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Array2::from_elem((2, 2), 0.1));
        optimizer.step(&mut params, &grads).unwrap();

        let d_after_step1 = optimizer.get_d();
        assert_eq!(d_after_step1, 1e-6);

        // Second step - D should adapt
        optimizer.step(&mut params, &grads).unwrap();

        let d_after_step2 = optimizer.get_d();
        assert!(d_after_step2 >= 1e-6); // D should increase or stay same
    }

    #[test]
    fn test_prodigy_parameter_update() {
        let config = ProdigyConfig::default();
        let mut optimizer = ProdigyOptimizer::new(config);

        let mut params = HashMap::new();
        let initial_value = 1.0;
        params.insert("w".to_string(), Array2::from_elem((2, 2), initial_value));

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Array2::from_elem((2, 2), 0.5));

        optimizer.step(&mut params, &grads).unwrap();

        // Parameters should be updated (decreased since gradient is positive)
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < initial_value);
    }

    #[test]
    fn test_prodigy_weight_decay() {
        let config = ProdigyConfig::default().with_weight_decay(0.01);
        let mut optimizer = ProdigyOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array2::from_elem((2, 2), 1.0));

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Array2::from_elem((2, 2), 0.0));

        // With weight decay and zero gradient, parameters should decay
        let initial_sum: f64 = params.get("w").unwrap().sum();
        optimizer.step(&mut params, &grads).unwrap();
        let final_sum: f64 = params.get("w").unwrap().sum();

        assert!(final_sum < initial_sum);
    }

    #[test]
    fn test_prodigy_gradient_clipping_by_norm() {
        let config = ProdigyConfig::default().with_grad_clip(0.1);
        let mut optimizer = ProdigyOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array2::from_elem((2, 2), 1.0));

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Array2::from_elem((2, 2), 10.0)); // Large gradient

        // Should not panic, gradients should be clipped
        optimizer.step(&mut params, &grads).unwrap();

        // Parameters should still be updated, but with clipped gradients
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
    }

    #[test]
    fn test_prodigy_state_dict() {
        let config = ProdigyConfig::default();
        let mut optimizer = ProdigyOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array2::from_elem((2, 2), 1.0));

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Array2::from_elem((2, 2), 0.1));

        // Take a few steps
        for _ in 0..3 {
            optimizer.step(&mut params, &grads).unwrap();
        }

        let state = optimizer.state_dict();
        assert!(state.contains_key("step"));
        assert!(state.contains_key("d"));
        assert!(state.contains_key("sum_grad_norm"));
    }

    #[test]
    fn test_prodigy_load_state_dict() {
        let config = ProdigyConfig::default();
        let mut optimizer1 = ProdigyOptimizer::new(config.clone());

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array2::from_elem((2, 2), 1.0));

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Array2::from_elem((2, 2), 0.1));

        // Take a few steps with optimizer1
        for _ in 0..3 {
            optimizer1.step(&mut params, &grads).unwrap();
        }

        let state = optimizer1.state_dict();

        // Create new optimizer and load state
        let mut optimizer2 = ProdigyOptimizer::new(config);
        optimizer2.load_state_dict(state);

        assert_eq!(optimizer1.get_step(), optimizer2.get_step());
        assert_eq!(optimizer1.get_d(), optimizer2.get_d());
    }

    #[test]
    fn test_prodigy_bias_correction() {
        let config_with_bc = ProdigyConfig::default().with_bias_correction(true);
        let config_without_bc = ProdigyConfig::default().with_bias_correction(false);

        let mut opt_with_bc = ProdigyOptimizer::new(config_with_bc);
        let mut opt_without_bc = ProdigyOptimizer::new(config_without_bc);

        let mut params1 = HashMap::new();
        params1.insert("w".to_string(), Array2::from_elem((2, 2), 1.0));

        let mut params2 = params1.clone();

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Array2::from_elem((2, 2), 0.1));

        opt_with_bc.step(&mut params1, &grads).unwrap();
        opt_without_bc.step(&mut params2, &grads).unwrap();

        // Results should be different due to bias correction
        let w1 = params1.get("w").unwrap();
        let w2 = params2.get("w").unwrap();

        // They should be different (bias correction affects the updates)
        let diff = (w1[[0, 0]] - w2[[0, 0]]).abs();
        assert!(diff > 1e-10);
    }

    #[test]
    fn test_prodigy_d_growth_rate_limit() {
        let config = ProdigyConfig::default()
            .with_d0(1e-6)
            .with_d_growth_rate(0.1); // 10% max growth per step

        let mut optimizer = ProdigyOptimizer::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array2::from_elem((2, 2), 1.0));

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Array2::from_elem((2, 2), 1.0)); // Large gradient

        // First step
        optimizer.step(&mut params, &grads).unwrap();
        let d1 = optimizer.get_d();

        // Second step
        optimizer.step(&mut params, &grads).unwrap();
        let d2 = optimizer.get_d();

        // D should not grow more than 10% per step
        if d2 > d1 {
            let growth_ratio = d2 / d1;
            assert!(growth_ratio <= 1.11); // Allow small numerical error
        }
    }
}
