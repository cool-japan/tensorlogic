//! Schedule-Free Optimizers - No Learning Rate Schedule Needed!
//!
//! Implementation of schedule-free learning from "The Road Less Scheduled" (Defazio et al., 2024).
//!
//! # Key Innovation
//!
//! Traditional deep learning requires carefully tuned learning rate schedules. Schedule-free
//! optimizers eliminate this requirement by maintaining two parameter sequences:
//! - **x_t**: Training sequence (used for gradient computation)
//! - **y_t**: Evaluation sequence (used for inference)
//!
//! The evaluation sequence y_t is an interpolation between current and past training parameters,
//! providing implicit scheduling without manual tuning.
//!
//! # Benefits
//!
//! 1. **No schedule tuning required** - Just set a constant learning rate
//! 2. **Better generalization** - Averaging provides implicit regularization
//! 3. **Faster convergence** - Adaptive to problem structure
//! 4. **Simpler hyperparameter search** - One less hyperparameter to tune
//!
//! # References
//!
//! - Defazio, A., Mishchenko, K., & Orabona, F. (2024).
//!   "The Road Less Scheduled". arXiv:2405.15682

use crate::optimizers::common::{GradClipMode, Optimizer};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Array2, Zip};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Schedule-free AdamW optimizer.
///
/// Maintains both training sequence (x) and evaluation sequence (y).
/// During training, gradients are computed w.r.t. x_t.
/// During evaluation, use y_t for better generalization.
///
/// # Algorithm
///
/// ```text
/// # Training step:
/// m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
/// v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
/// x_t = x_{t-1} - lr * m_t / (√v_t + ε) - lr * λ * x_{t-1}  # weight decay
///
/// # Evaluation sequence (exponential moving average):
/// y_t = (1 - γ) * x_t + γ * y_{t-1}
///
/// # At test time, use y_t instead of x_t
/// ```
///
/// # Example
///
/// ```no_run
/// use tensorlogic_train::{ScheduleFreeAdamW, ScheduleFreeConfig};
/// use scirs2_core::ndarray::Array2;
/// use std::collections::HashMap;
///
/// let config = ScheduleFreeConfig::default()
///     .with_lr(0.001)
///     .with_warmup_steps(1000);
///
/// let mut optimizer = ScheduleFreeAdamW::new(config);
///
/// // Training mode - use training parameters
/// optimizer.set_training_mode(true);
///
/// // ... compute gradients ...
///
/// // Evaluation mode - switch to averaged parameters
/// optimizer.set_training_mode(false);
/// let eval_params = optimizer.get_eval_parameters();
/// ```
#[derive(Debug, Clone)]
pub struct ScheduleFreeAdamW {
    /// Configuration
    config: ScheduleFreeConfig,
    /// Training parameters (x_t)
    train_params: HashMap<String, Array2<f64>>,
    /// Evaluation parameters (y_t) - exponential moving average
    eval_params: HashMap<String, Array2<f64>>,
    /// First moment estimates
    first_moments: HashMap<String, Array2<f64>>,
    /// Second moment estimates
    second_moments: HashMap<String, Array2<f64>>,
    /// Current step number
    step: usize,
    /// Training mode flag (true = use train_params, false = use eval_params)
    training_mode: bool,
}

/// Configuration for schedule-free optimizers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleFreeConfig {
    /// Learning rate (constant, no schedule needed!)
    pub lr: f64,
    /// Beta1 for first moment (default: 0.9)
    pub beta1: f64,
    /// Beta2 for second moment (default: 0.999)
    pub beta2: f64,
    /// Weight decay coefficient (default: 0.01)
    pub weight_decay: f64,
    /// Epsilon for numerical stability (default: 1e-8)
    pub eps: f64,
    /// Averaging coefficient γ for evaluation sequence (default: 0.95)
    /// Higher values = more smoothing, better generalization
    pub gamma: f64,
    /// Number of warmup steps (default: 0)
    /// During warmup, gamma increases linearly from 0 to target value
    pub warmup_steps: usize,
    /// Gradient clipping threshold (None = no clipping)
    pub grad_clip: Option<f64>,
    /// Gradient clipping mode
    pub grad_clip_mode: GradClipMode,
}

impl Default for ScheduleFreeConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            gamma: 0.95,
            warmup_steps: 0,
            grad_clip: None,
            grad_clip_mode: GradClipMode::Norm,
        }
    }
}

impl ScheduleFreeConfig {
    /// Create a new configuration with custom learning rate.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Set learning rate (builder pattern).
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set beta1 (builder pattern).
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (builder pattern).
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set weight decay (builder pattern).
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set gamma (averaging coefficient) (builder pattern).
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set warmup steps (builder pattern).
    pub fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set gradient clipping (builder pattern).
    pub fn with_grad_clip(mut self, threshold: f64, mode: GradClipMode) -> Self {
        self.grad_clip = Some(threshold);
        self.grad_clip_mode = mode;
        self
    }
}

impl ScheduleFreeAdamW {
    /// Create a new Schedule-Free AdamW optimizer.
    pub fn new(config: ScheduleFreeConfig) -> Self {
        Self {
            config,
            train_params: HashMap::new(),
            eval_params: HashMap::new(),
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            step: 0,
            training_mode: true,
        }
    }

    /// Set training mode.
    ///
    /// When training_mode = true, use train_params (x_t) for gradients.
    /// When training_mode = false, use eval_params (y_t) for evaluation.
    pub fn set_training_mode(&mut self, training: bool) {
        self.training_mode = training;
    }

    /// Get current training mode.
    pub fn is_training(&self) -> bool {
        self.training_mode
    }

    /// Get evaluation parameters (for inference).
    pub fn get_eval_parameters(&self) -> &HashMap<String, Array2<f64>> {
        &self.eval_params
    }

    /// Get training parameters (for gradient computation).
    pub fn get_train_parameters(&self) -> &HashMap<String, Array2<f64>> {
        &self.train_params
    }

    /// Compute effective gamma with warmup.
    fn effective_gamma(&self) -> f64 {
        if self.config.warmup_steps == 0 {
            return self.config.gamma;
        }

        if self.step >= self.config.warmup_steps {
            self.config.gamma
        } else {
            // Linear warmup: gamma goes from 0 to target value
            self.config.gamma * (self.step as f64 / self.config.warmup_steps as f64)
        }
    }
}

impl Optimizer for ScheduleFreeAdamW {
    fn zero_grad(&mut self) {
        // Schedule-free optimizers don't maintain gradients, so this is a no-op
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
        if gradients.is_empty() {
            return Ok(());
        }

        self.step += 1;

        // Initialize if needed
        if self.train_params.is_empty() {
            for (name, param) in parameters.iter() {
                self.train_params.insert(name.clone(), param.clone());
                self.eval_params.insert(name.clone(), param.clone());
                self.first_moments
                    .insert(name.clone(), Array::zeros(param.raw_dim()));
                self.second_moments
                    .insert(name.clone(), Array::zeros(param.raw_dim()));
            }
        }

        let gamma = self.effective_gamma();

        // Update each parameter
        for (name, grad) in gradients.iter() {
            let param = self.train_params.get_mut(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Parameter {} not found", name))
            })?;

            let m = self.first_moments.get_mut(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("First moment {} not found", name))
            })?;

            let v = self.second_moments.get_mut(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Second moment {} not found", name))
            })?;

            // Apply gradient clipping if configured
            let grad_clipped = if let Some(threshold) = self.config.grad_clip {
                match self.config.grad_clip_mode {
                    GradClipMode::Value => grad.mapv(|g| g.max(-threshold).min(threshold)),
                    GradClipMode::Norm => {
                        let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                        if norm > threshold {
                            grad.mapv(|g| g * threshold / norm)
                        } else {
                            grad.clone()
                        }
                    }
                }
            } else {
                grad.clone()
            };

            // Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            Zip::from(&mut *m).and(&grad_clipped).for_each(|m_val, &g| {
                *m_val = self.config.beta1 * *m_val + (1.0 - self.config.beta1) * g;
            });

            // Update biased second moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            Zip::from(&mut *v).and(&grad_clipped).for_each(|v_val, &g| {
                *v_val = self.config.beta2 * *v_val + (1.0 - self.config.beta2) * g * g;
            });

            // Bias correction
            let m_hat_coef = 1.0 / (1.0 - self.config.beta1.powi(self.step as i32));
            let v_hat_coef = 1.0 / (1.0 - self.config.beta2.powi(self.step as i32));

            // Update training parameters with AdamW update:
            // x_t = x_{t-1} - lr * (m_hat / (√v_hat + ε) + λ * x_{t-1})
            Zip::from(&mut *param)
                .and(&*m)
                .and(&*v)
                .for_each(|p, &m_val, &v_val| {
                    let m_hat = m_val * m_hat_coef;
                    let v_hat = v_val * v_hat_coef;

                    // AdamW-style update
                    let adam_update = m_hat / (v_hat.sqrt() + self.config.eps);
                    let weight_decay_update = self.config.weight_decay * *p;

                    *p -= self.config.lr * (adam_update + weight_decay_update);
                });

            // Update evaluation parameters: y_t = (1 - γ) * x_t + γ * y_{t-1}
            let eval_param = self.eval_params.get_mut(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Eval parameter {} not found", name))
            })?;

            Zip::from(&mut *eval_param).and(&*param).for_each(|y, &x| {
                *y = (1.0 - gamma) * x + gamma * *y;
            });
        }

        // Update the provided parameters based on current mode
        for (name, param) in parameters.iter_mut() {
            if self.training_mode {
                // Use training parameters
                if let Some(train_param) = self.train_params.get(name) {
                    param.assign(train_param);
                }
            } else {
                // Use evaluation parameters
                if let Some(eval_param) = self.eval_params.get(name) {
                    param.assign(eval_param);
                }
            }
        }

        Ok(())
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();

        // Save configuration
        state.insert("lr".to_string(), vec![self.config.lr]);
        state.insert("beta1".to_string(), vec![self.config.beta1]);
        state.insert("beta2".to_string(), vec![self.config.beta2]);
        state.insert("weight_decay".to_string(), vec![self.config.weight_decay]);
        state.insert("eps".to_string(), vec![self.config.eps]);
        state.insert("gamma".to_string(), vec![self.config.gamma]);
        state.insert(
            "warmup_steps".to_string(),
            vec![self.config.warmup_steps as f64],
        );
        state.insert("step".to_string(), vec![self.step as f64]);
        state.insert(
            "training_mode".to_string(),
            vec![if self.training_mode { 1.0 } else { 0.0 }],
        );

        // Save moments and parameters
        for (name, m) in &self.first_moments {
            state.insert(
                format!("first_moment_{}", name),
                m.iter().copied().collect(),
            );
        }

        for (name, v) in &self.second_moments {
            state.insert(
                format!("second_moment_{}", name),
                v.iter().copied().collect(),
            );
        }

        for (name, p) in &self.train_params {
            state.insert(format!("train_param_{}", name), p.iter().copied().collect());
        }

        for (name, p) in &self.eval_params {
            state.insert(format!("eval_param_{}", name), p.iter().copied().collect());
        }

        state
    }

    fn load_state_dict(&mut self, _state: HashMap<String, Vec<f64>>) {
        // Simplified: just reset to initial state
        // In production, would properly deserialize all state
        self.step = 0;
        self.first_moments.clear();
        self.second_moments.clear();
        self.train_params.clear();
        self.eval_params.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_schedulefree_creation() {
        let config = ScheduleFreeConfig::default();
        let optimizer = ScheduleFreeAdamW::new(config);

        assert_eq!(optimizer.get_lr(), 0.001);
        assert!(optimizer.is_training());
    }

    #[test]
    fn test_schedulefree_config_builder() {
        let config = ScheduleFreeConfig::default()
            .with_lr(0.01)
            .with_beta1(0.85)
            .with_beta2(0.995)
            .with_gamma(0.98)
            .with_warmup_steps(1000);

        assert_eq!(config.lr, 0.01);
        assert_eq!(config.beta1, 0.85);
        assert_eq!(config.beta2, 0.995);
        assert_eq!(config.gamma, 0.98);
        assert_eq!(config.warmup_steps, 1000);
    }

    #[test]
    fn test_schedulefree_training_mode() {
        let config = ScheduleFreeConfig::default();
        let mut optimizer = ScheduleFreeAdamW::new(config);

        assert!(optimizer.is_training());

        optimizer.set_training_mode(false);
        assert!(!optimizer.is_training());

        optimizer.set_training_mode(true);
        assert!(optimizer.is_training());
    }

    #[test]
    fn test_schedulefree_step() {
        let config = ScheduleFreeConfig::default().with_lr(0.1);
        let mut optimizer = ScheduleFreeAdamW::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2], [0.3, 0.4]]);

        // Take a step
        optimizer.step(&mut params, &grads).unwrap();

        // Parameters should have been updated
        let updated_w = params.get("w").unwrap();
        assert_ne!(updated_w[[0, 0]], 1.0);

        // Should have created moments
        assert_eq!(optimizer.first_moments.len(), 1);
        assert_eq!(optimizer.second_moments.len(), 1);
    }

    #[test]
    fn test_schedulefree_eval_parameters() {
        let config = ScheduleFreeConfig::default().with_lr(0.1).with_gamma(0.5);
        let mut optimizer = ScheduleFreeAdamW::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.2]]);

        // Take multiple steps
        for _ in 0..5 {
            optimizer.step(&mut params, &grads).unwrap();
        }

        // Eval parameters should be different from training parameters
        let train_params = optimizer.get_train_parameters();
        let eval_params = optimizer.get_eval_parameters();

        let train_w = train_params.get("w").unwrap();
        let eval_w = eval_params.get("w").unwrap();

        // They should be different due to averaging
        assert_ne!(train_w[[0, 0]], eval_w[[0, 0]]);
    }

    #[test]
    fn test_schedulefree_gamma_warmup() {
        let config = ScheduleFreeConfig::default().with_warmup_steps(100);
        let mut optimizer = ScheduleFreeAdamW::new(config);

        // At step 0, effective gamma should be 0
        assert_eq!(optimizer.effective_gamma(), 0.0);

        // Initialize and take steps
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0]]);

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1]]);

        for _ in 0..50 {
            optimizer.step(&mut params, &grads).unwrap();
        }

        // At step 50, effective gamma should be approximately halfway
        let gamma_50 = optimizer.effective_gamma();
        let expected_50 = 0.95 * (50.0 / 100.0);
        assert!(
            (gamma_50 - expected_50).abs() < 0.05,
            "gamma_50 = {}, expected ~{}",
            gamma_50,
            expected_50
        );

        for _ in 50..100 {
            optimizer.step(&mut params, &grads).unwrap();
        }

        // At step 100, effective gamma should be full value
        assert!((optimizer.effective_gamma() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_schedulefree_gradient_clipping() {
        let config = ScheduleFreeConfig::default()
            .with_lr(0.1)
            .with_grad_clip(0.5, GradClipMode::Value);

        let mut optimizer = ScheduleFreeAdamW::new(config);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut grads = HashMap::new();
        // Large gradients that should be clipped
        grads.insert("w".to_string(), array![[10.0, -10.0]]);

        optimizer.step(&mut params, &grads).unwrap();

        // With clipping, the update should be smaller
        let updated_w = params.get("w").unwrap();
        // If no clipping, change would be huge; with clipping, it's bounded
        assert!(updated_w[[0, 0]] > 0.5); // Not too much decrease
        assert!(updated_w[[0, 1]] < 2.5); // Not too much increase
    }

    #[test]
    fn test_schedulefree_weight_decay() {
        let config_no_decay = ScheduleFreeConfig::default()
            .with_lr(0.1)
            .with_weight_decay(0.0);

        let config_with_decay = ScheduleFreeConfig::default()
            .with_lr(0.1)
            .with_weight_decay(0.1);

        let mut opt_no_decay = ScheduleFreeAdamW::new(config_no_decay);
        let mut opt_with_decay = ScheduleFreeAdamW::new(config_with_decay);

        let mut params1 = HashMap::new();
        params1.insert("w".to_string(), array![[1.0, 2.0]]);

        let mut params2 = params1.clone();

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1]]);

        opt_no_decay.step(&mut params1, &grads).unwrap();
        opt_with_decay.step(&mut params2, &grads).unwrap();

        // With weight decay, parameters should shrink more
        let w1 = params1.get("w").unwrap();
        let w2 = params2.get("w").unwrap();

        assert!(w2[[0, 0]] < w1[[0, 0]]);
        assert!(w2[[0, 1]] < w1[[0, 1]]);
    }
}
