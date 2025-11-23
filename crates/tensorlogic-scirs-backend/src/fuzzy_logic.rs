//! Temperature-controlled fuzzy logic operations.
//!
//! This module provides soft, differentiable versions of logical operations
//! with configurable temperature parameters. Lower temperatures approximate
//! hard Boolean logic, while higher temperatures give smoother gradients.
//!
//! # Fuzzy Logic Families
//!
//! - **Gödel (min/max)**: Traditional fuzzy logic
//! - **Product**: Probabilistic interpretation
//! - **Łukasiewicz**: Bounded arithmetic
//! - **Soft (temperature-controlled)**: Differentiable approximations
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_scirs_backend::fuzzy_logic::{FuzzyConfig, FuzzyLogic, soft_and, soft_or};
//! use scirs2_core::ndarray::ArrayD;
//!
//! let config = FuzzyConfig::soft(0.1);  // Temperature = 0.1
//!
//! let a = ArrayD::from_elem(vec![3], 0.8);
//! let b = ArrayD::from_elem(vec![3], 0.6);
//!
//! // Soft AND (approximates min as temperature -> 0)
//! let and_result = soft_and(&a, &b, config.temperature);
//!
//! // Soft OR (approximates max as temperature -> 0)
//! let or_result = soft_or(&a, &b, config.temperature);
//! ```

use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};
use std::f64::consts::E;
use thiserror::Error;

/// Errors for fuzzy logic operations.
#[derive(Debug, Error)]
pub enum FuzzyError {
    #[error("Shape mismatch: {0:?} vs {1:?}")]
    ShapeMismatch(Vec<usize>, Vec<usize>),

    #[error("Invalid temperature: {0} (must be positive)")]
    InvalidTemperature(f64),

    #[error("Invalid value: {0} (must be in [0, 1])")]
    InvalidValue(f64),
}

/// Result type for fuzzy operations.
pub type FuzzyResult<T> = Result<T, FuzzyError>;

/// Fuzzy logic family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuzzyFamily {
    /// Gödel fuzzy logic: AND = min, OR = max
    Godel,
    /// Product fuzzy logic: AND = product, OR = probabilistic sum
    Product,
    /// Łukasiewicz fuzzy logic: bounded arithmetic
    Lukasiewicz,
    /// Soft (differentiable) approximations with temperature
    Soft,
}

impl Default for FuzzyFamily {
    fn default() -> Self {
        FuzzyFamily::Soft
    }
}

/// Configuration for fuzzy logic operations.
#[derive(Debug, Clone)]
pub struct FuzzyConfig {
    /// The fuzzy logic family to use
    pub family: FuzzyFamily,
    /// Temperature parameter for soft operations (lower = sharper)
    pub temperature: f64,
    /// Whether to clamp outputs to [0, 1]
    pub clamp_output: bool,
    /// Epsilon for numerical stability
    pub epsilon: f64,
}

impl Default for FuzzyConfig {
    fn default() -> Self {
        FuzzyConfig {
            family: FuzzyFamily::Soft,
            temperature: 1.0,
            clamp_output: true,
            epsilon: 1e-10,
        }
    }
}

impl FuzzyConfig {
    /// Create a Gödel (min/max) fuzzy logic configuration.
    pub fn godel() -> Self {
        FuzzyConfig {
            family: FuzzyFamily::Godel,
            temperature: 1.0,
            clamp_output: true,
            epsilon: 1e-10,
        }
    }

    /// Create a Product fuzzy logic configuration.
    pub fn product() -> Self {
        FuzzyConfig {
            family: FuzzyFamily::Product,
            temperature: 1.0,
            clamp_output: true,
            epsilon: 1e-10,
        }
    }

    /// Create a Łukasiewicz fuzzy logic configuration.
    pub fn lukasiewicz() -> Self {
        FuzzyConfig {
            family: FuzzyFamily::Lukasiewicz,
            temperature: 1.0,
            clamp_output: true,
            epsilon: 1e-10,
        }
    }

    /// Create a soft (temperature-controlled) fuzzy logic configuration.
    pub fn soft(temperature: f64) -> Self {
        FuzzyConfig {
            family: FuzzyFamily::Soft,
            temperature,
            clamp_output: true,
            epsilon: 1e-10,
        }
    }

    /// Set the temperature parameter.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set whether to clamp outputs to [0, 1].
    pub fn with_clamping(mut self, clamp: bool) -> Self {
        self.clamp_output = clamp;
        self
    }

    /// Set the epsilon for numerical stability.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> FuzzyResult<()> {
        if self.temperature <= 0.0 {
            return Err(FuzzyError::InvalidTemperature(self.temperature));
        }
        Ok(())
    }
}

/// Soft minimum (approximates min as temperature -> 0).
///
/// Uses log-sum-exp trick: softmin(a, b) = -T * log(exp(-a/T) + exp(-b/T))
pub fn soft_min(a: f64, b: f64, temperature: f64) -> f64 {
    if temperature <= 1e-10 {
        return a.min(b);
    }

    // Use log-sum-exp trick for numerical stability
    let min_val = a.min(b);
    let exp_a = ((-a + min_val) / temperature).exp();
    let exp_b = ((-b + min_val) / temperature).exp();

    -temperature * (exp_a + exp_b).ln() + min_val
}

/// Soft maximum (approximates max as temperature -> 0).
///
/// Uses log-sum-exp trick: softmax(a, b) = T * log(exp(a/T) + exp(b/T))
pub fn soft_max(a: f64, b: f64, temperature: f64) -> f64 {
    if temperature <= 1e-10 {
        return a.max(b);
    }

    // Use log-sum-exp trick for numerical stability
    let max_val = a.max(b);
    let exp_a = ((a - max_val) / temperature).exp();
    let exp_b = ((b - max_val) / temperature).exp();

    temperature * (exp_a + exp_b).ln() + max_val
}

/// Soft AND operation (element-wise).
///
/// Approximates min(a, b) with configurable temperature.
pub fn soft_and(a: &ArrayD<f64>, b: &ArrayD<f64>, temperature: f64) -> ArrayD<f64> {
    Zip::from(a).and(b).map_collect(|&x, &y| soft_min(x, y, temperature))
}

/// Soft OR operation (element-wise).
///
/// Approximates max(a, b) with configurable temperature.
pub fn soft_or(a: &ArrayD<f64>, b: &ArrayD<f64>, temperature: f64) -> ArrayD<f64> {
    Zip::from(a).and(b).map_collect(|&x, &y| soft_max(x, y, temperature))
}

/// Soft NOT operation.
///
/// Standard negation: NOT(x) = 1 - x (same across all families).
pub fn soft_not(x: &ArrayD<f64>) -> ArrayD<f64> {
    x.mapv(|v| 1.0 - v)
}

/// Soft implication with temperature control.
///
/// Several variants:
/// - Gödel: a → b = 1 if a <= b, else b
/// - Product: a → b = min(1, b/a) if a > 0, else 1
/// - Łukasiewicz: a → b = min(1, 1 - a + b)
/// - Soft: Uses softmax for differentiability
pub fn soft_imply(a: &ArrayD<f64>, b: &ArrayD<f64>, config: &FuzzyConfig) -> ArrayD<f64> {
    match config.family {
        FuzzyFamily::Godel => Zip::from(a)
            .and(b)
            .map_collect(|&x, &y| if x <= y { 1.0 } else { y }),
        FuzzyFamily::Product => Zip::from(a).and(b).map_collect(|&x, &y| {
            if x <= config.epsilon {
                1.0
            } else {
                (y / x).min(1.0)
            }
        }),
        FuzzyFamily::Lukasiewicz => {
            Zip::from(a).and(b).map_collect(|&x, &y| (1.0 - x + y).min(1.0))
        }
        FuzzyFamily::Soft => {
            // Soft version using ReLU-like approximation: max(0, b - a) scaled
            Zip::from(a).and(b).map_collect(|&x, &y| {
                let diff = y - x;
                let sigmoid_diff = 1.0 / (1.0 + (-diff / config.temperature).exp());
                // Blend between 1 (when b >= a) and b (when b < a)
                sigmoid_diff + (1.0 - sigmoid_diff) * y.max(0.0).min(1.0)
            })
        }
    }
}

/// Fuzzy logic executor for tensor operations.
#[derive(Debug, Clone)]
pub struct FuzzyLogic {
    config: FuzzyConfig,
}

impl FuzzyLogic {
    /// Create a new fuzzy logic executor with the given configuration.
    pub fn new(config: FuzzyConfig) -> FuzzyResult<Self> {
        config.validate()?;
        Ok(FuzzyLogic { config })
    }

    /// Create with default (soft) configuration.
    pub fn default_soft() -> Self {
        FuzzyLogic {
            config: FuzzyConfig::default(),
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &FuzzyConfig {
        &self.config
    }

    /// Set the temperature (for soft family).
    pub fn set_temperature(&mut self, temperature: f64) -> FuzzyResult<()> {
        if temperature <= 0.0 {
            return Err(FuzzyError::InvalidTemperature(temperature));
        }
        self.config.temperature = temperature;
        Ok(())
    }

    /// Clamp tensor to [0, 1] if configured.
    fn maybe_clamp(&self, x: ArrayD<f64>) -> ArrayD<f64> {
        if self.config.clamp_output {
            x.mapv(|v| v.clamp(0.0, 1.0))
        } else {
            x
        }
    }

    /// Fuzzy AND operation.
    pub fn and(&self, a: &ArrayD<f64>, b: &ArrayD<f64>) -> FuzzyResult<ArrayD<f64>> {
        if a.shape() != b.shape() {
            return Err(FuzzyError::ShapeMismatch(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        let result = match self.config.family {
            FuzzyFamily::Godel => Zip::from(a).and(b).map_collect(|&x, &y| x.min(y)),
            FuzzyFamily::Product => Zip::from(a).and(b).map_collect(|&x, &y| x * y),
            FuzzyFamily::Lukasiewicz => {
                Zip::from(a).and(b).map_collect(|&x, &y| (x + y - 1.0).max(0.0))
            }
            FuzzyFamily::Soft => soft_and(a, b, self.config.temperature),
        };

        Ok(self.maybe_clamp(result))
    }

    /// Fuzzy OR operation.
    pub fn or(&self, a: &ArrayD<f64>, b: &ArrayD<f64>) -> FuzzyResult<ArrayD<f64>> {
        if a.shape() != b.shape() {
            return Err(FuzzyError::ShapeMismatch(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        let result = match self.config.family {
            FuzzyFamily::Godel => Zip::from(a).and(b).map_collect(|&x, &y| x.max(y)),
            FuzzyFamily::Product => Zip::from(a).and(b).map_collect(|&x, &y| x + y - x * y),
            FuzzyFamily::Lukasiewicz => {
                Zip::from(a).and(b).map_collect(|&x, &y| (x + y).min(1.0))
            }
            FuzzyFamily::Soft => soft_or(a, b, self.config.temperature),
        };

        Ok(self.maybe_clamp(result))
    }

    /// Fuzzy NOT operation.
    pub fn not(&self, x: &ArrayD<f64>) -> ArrayD<f64> {
        let result = soft_not(x);
        self.maybe_clamp(result)
    }

    /// Fuzzy implication (a → b).
    pub fn imply(&self, a: &ArrayD<f64>, b: &ArrayD<f64>) -> FuzzyResult<ArrayD<f64>> {
        if a.shape() != b.shape() {
            return Err(FuzzyError::ShapeMismatch(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        let result = soft_imply(a, b, &self.config);
        Ok(self.maybe_clamp(result))
    }

    /// Fuzzy equivalence (a ↔ b).
    pub fn equiv(&self, a: &ArrayD<f64>, b: &ArrayD<f64>) -> FuzzyResult<ArrayD<f64>> {
        // a ↔ b = (a → b) ∧ (b → a)
        let a_implies_b = self.imply(a, b)?;
        let b_implies_a = self.imply(b, a)?;
        self.and(&a_implies_b, &b_implies_a)
    }

    /// Fuzzy XOR operation.
    pub fn xor(&self, a: &ArrayD<f64>, b: &ArrayD<f64>) -> FuzzyResult<ArrayD<f64>> {
        if a.shape() != b.shape() {
            return Err(FuzzyError::ShapeMismatch(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        // XOR = (a ∧ ¬b) ∨ (¬a ∧ b)
        let not_a = self.not(a);
        let not_b = self.not(b);
        let a_and_not_b = self.and(a, &not_b)?;
        let not_a_and_b = self.and(&not_a, b)?;
        self.or(&a_and_not_b, &not_a_and_b)
    }

    /// Fuzzy NAND operation.
    pub fn nand(&self, a: &ArrayD<f64>, b: &ArrayD<f64>) -> FuzzyResult<ArrayD<f64>> {
        let and_result = self.and(a, b)?;
        Ok(self.not(&and_result))
    }

    /// Fuzzy NOR operation.
    pub fn nor(&self, a: &ArrayD<f64>, b: &ArrayD<f64>) -> FuzzyResult<ArrayD<f64>> {
        let or_result = self.or(a, b)?;
        Ok(self.not(&or_result))
    }

    /// Multi-argument AND (conjunction).
    pub fn and_many(&self, tensors: &[&ArrayD<f64>]) -> FuzzyResult<ArrayD<f64>> {
        if tensors.is_empty() {
            return Err(FuzzyError::InvalidValue(0.0));
        }

        let mut result = tensors[0].clone();
        for tensor in &tensors[1..] {
            result = self.and(&result, tensor)?;
        }
        Ok(result)
    }

    /// Multi-argument OR (disjunction).
    pub fn or_many(&self, tensors: &[&ArrayD<f64>]) -> FuzzyResult<ArrayD<f64>> {
        if tensors.is_empty() {
            return Err(FuzzyError::InvalidValue(0.0));
        }

        let mut result = tensors[0].clone();
        for tensor in &tensors[1..] {
            result = self.or(&result, tensor)?;
        }
        Ok(result)
    }
}

/// Gradient computation for soft AND with respect to first input.
pub fn soft_and_grad_a(a: &ArrayD<f64>, b: &ArrayD<f64>, grad: &ArrayD<f64>, temperature: f64) -> ArrayD<f64> {
    if temperature <= 1e-10 {
        // Hard min gradient
        return Zip::from(a)
            .and(b)
            .and(grad)
            .map_collect(|&x, &y, &g| if x <= y { g } else { 0.0 });
    }

    // Soft gradient using softmin derivative
    Zip::from(a).and(b).and(grad).map_collect(|&x, &y, &g| {
        let min_val = x.min(y);
        let exp_a = ((-x + min_val) / temperature).exp();
        let exp_b = ((-y + min_val) / temperature).exp();
        let sum_exp = exp_a + exp_b;
        g * exp_a / sum_exp
    })
}

/// Gradient computation for soft AND with respect to second input.
pub fn soft_and_grad_b(a: &ArrayD<f64>, b: &ArrayD<f64>, grad: &ArrayD<f64>, temperature: f64) -> ArrayD<f64> {
    if temperature <= 1e-10 {
        return Zip::from(a)
            .and(b)
            .and(grad)
            .map_collect(|&x, &y, &g| if y < x { g } else { 0.0 });
    }

    Zip::from(a).and(b).and(grad).map_collect(|&x, &y, &g| {
        let min_val = x.min(y);
        let exp_a = ((-x + min_val) / temperature).exp();
        let exp_b = ((-y + min_val) / temperature).exp();
        let sum_exp = exp_a + exp_b;
        g * exp_b / sum_exp
    })
}

/// Gradient computation for soft OR with respect to first input.
pub fn soft_or_grad_a(a: &ArrayD<f64>, b: &ArrayD<f64>, grad: &ArrayD<f64>, temperature: f64) -> ArrayD<f64> {
    if temperature <= 1e-10 {
        return Zip::from(a)
            .and(b)
            .and(grad)
            .map_collect(|&x, &y, &g| if x >= y { g } else { 0.0 });
    }

    Zip::from(a).and(b).and(grad).map_collect(|&x, &y, &g| {
        let max_val = x.max(y);
        let exp_a = ((x - max_val) / temperature).exp();
        let exp_b = ((y - max_val) / temperature).exp();
        let sum_exp = exp_a + exp_b;
        g * exp_a / sum_exp
    })
}

/// Gradient computation for soft OR with respect to second input.
pub fn soft_or_grad_b(a: &ArrayD<f64>, b: &ArrayD<f64>, grad: &ArrayD<f64>, temperature: f64) -> ArrayD<f64> {
    if temperature <= 1e-10 {
        return Zip::from(a)
            .and(b)
            .and(grad)
            .map_collect(|&x, &y, &g| if y > x { g } else { 0.0 });
    }

    Zip::from(a).and(b).and(grad).map_collect(|&x, &y, &g| {
        let max_val = x.max(y);
        let exp_a = ((x - max_val) / temperature).exp();
        let exp_b = ((y - max_val) / temperature).exp();
        let sum_exp = exp_a + exp_b;
        g * exp_b / sum_exp
    })
}

/// Temperature annealing schedules for fuzzy logic training.
#[derive(Debug, Clone, Copy)]
pub enum AnnealingSchedule {
    /// Constant temperature
    Constant,
    /// Linear decay: T(t) = T_max - (T_max - T_min) * t / T_steps
    Linear {
        t_max: f64,
        t_min: f64,
        steps: usize,
    },
    /// Exponential decay: T(t) = T_max * decay^t
    Exponential { t_max: f64, decay: f64 },
    /// Cosine annealing: T(t) = T_min + 0.5 * (T_max - T_min) * (1 + cos(pi * t / T_steps))
    Cosine {
        t_max: f64,
        t_min: f64,
        steps: usize,
    },
}

impl AnnealingSchedule {
    /// Get temperature at step t.
    pub fn temperature(&self, base_temp: f64, step: usize) -> f64 {
        match *self {
            AnnealingSchedule::Constant => base_temp,
            AnnealingSchedule::Linear { t_max, t_min, steps } => {
                if step >= steps {
                    t_min
                } else {
                    t_max - (t_max - t_min) * (step as f64 / steps as f64)
                }
            }
            AnnealingSchedule::Exponential { t_max, decay } => t_max * decay.powi(step as i32),
            AnnealingSchedule::Cosine { t_max, t_min, steps } => {
                if step >= steps {
                    t_min
                } else {
                    let progress = step as f64 / steps as f64;
                    t_min + 0.5 * (t_max - t_min) * (1.0 + (std::f64::consts::PI * progress).cos())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(values: Vec<f64>) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap()
    }

    #[test]
    fn test_soft_min() {
        // Low temperature should approximate hard min
        assert!((soft_min(0.3, 0.7, 0.001) - 0.3).abs() < 0.01);
        assert!((soft_min(0.8, 0.2, 0.001) - 0.2).abs() < 0.01);

        // Higher temperature gives softer result
        let soft_result = soft_min(0.3, 0.7, 1.0);
        assert!(soft_result < 0.5); // Should be less than average
        assert!(soft_result > 0.3); // But greater than hard min
    }

    #[test]
    fn test_soft_max() {
        // Low temperature should approximate hard max
        assert!((soft_max(0.3, 0.7, 0.001) - 0.7).abs() < 0.01);
        assert!((soft_max(0.8, 0.2, 0.001) - 0.8).abs() < 0.01);

        // Higher temperature gives softer result
        let soft_result = soft_max(0.3, 0.7, 1.0);
        assert!(soft_result > 0.5); // Should be more than average
        assert!(soft_result < 0.7); // But less than hard max
    }

    #[test]
    fn test_godel_and() {
        let fuzzy = FuzzyLogic::new(FuzzyConfig::godel()).unwrap();
        let a = arr(vec![0.3, 0.8, 0.5]);
        let b = arr(vec![0.7, 0.2, 0.5]);

        let result = fuzzy.and(&a, &b).unwrap();

        assert!((result[[0]] - 0.3).abs() < 1e-10);
        assert!((result[[1]] - 0.2).abs() < 1e-10);
        assert!((result[[2]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_godel_or() {
        let fuzzy = FuzzyLogic::new(FuzzyConfig::godel()).unwrap();
        let a = arr(vec![0.3, 0.8, 0.5]);
        let b = arr(vec![0.7, 0.2, 0.5]);

        let result = fuzzy.or(&a, &b).unwrap();

        assert!((result[[0]] - 0.7).abs() < 1e-10);
        assert!((result[[1]] - 0.8).abs() < 1e-10);
        assert!((result[[2]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_product_and() {
        let fuzzy = FuzzyLogic::new(FuzzyConfig::product()).unwrap();
        let a = arr(vec![0.5, 0.8]);
        let b = arr(vec![0.6, 0.5]);

        let result = fuzzy.and(&a, &b).unwrap();

        assert!((result[[0]] - 0.3).abs() < 1e-10); // 0.5 * 0.6
        assert!((result[[1]] - 0.4).abs() < 1e-10); // 0.8 * 0.5
    }

    #[test]
    fn test_product_or() {
        let fuzzy = FuzzyLogic::new(FuzzyConfig::product()).unwrap();
        let a = arr(vec![0.5, 0.8]);
        let b = arr(vec![0.6, 0.5]);

        let result = fuzzy.or(&a, &b).unwrap();

        // a + b - ab
        assert!((result[[0]] - 0.8).abs() < 1e-10); // 0.5 + 0.6 - 0.3
        assert!((result[[1]] - 0.9).abs() < 1e-10); // 0.8 + 0.5 - 0.4
    }

    #[test]
    fn test_lukasiewicz_and() {
        let fuzzy = FuzzyLogic::new(FuzzyConfig::lukasiewicz()).unwrap();
        let a = arr(vec![0.8, 0.3]);
        let b = arr(vec![0.7, 0.4]);

        let result = fuzzy.and(&a, &b).unwrap();

        // max(0, a + b - 1)
        assert!((result[[0]] - 0.5).abs() < 1e-10); // max(0, 0.8 + 0.7 - 1) = 0.5
        assert!((result[[1]] - 0.0).abs() < 1e-10); // max(0, 0.3 + 0.4 - 1) = 0
    }

    #[test]
    fn test_lukasiewicz_or() {
        let fuzzy = FuzzyLogic::new(FuzzyConfig::lukasiewicz()).unwrap();
        let a = arr(vec![0.8, 0.3]);
        let b = arr(vec![0.7, 0.4]);

        let result = fuzzy.or(&a, &b).unwrap();

        // min(1, a + b)
        assert!((result[[0]] - 1.0).abs() < 1e-10); // min(1, 0.8 + 0.7) = 1
        assert!((result[[1]] - 0.7).abs() < 1e-10); // min(1, 0.3 + 0.4) = 0.7
    }

    #[test]
    fn test_soft_and_tensor() {
        let a = arr(vec![0.3, 0.8]);
        let b = arr(vec![0.7, 0.2]);

        let result = soft_and(&a, &b, 0.1);

        // Should be close to min but slightly higher
        assert!(result[[0]] < 0.4);
        assert!(result[[0]] > 0.25);
        assert!(result[[1]] < 0.3);
        assert!(result[[1]] > 0.15);
    }

    #[test]
    fn test_fuzzy_not() {
        let fuzzy = FuzzyLogic::default_soft();
        let x = arr(vec![0.0, 0.3, 0.5, 0.7, 1.0]);

        let result = fuzzy.not(&x);

        assert!((result[[0]] - 1.0).abs() < 1e-10);
        assert!((result[[1]] - 0.7).abs() < 1e-10);
        assert!((result[[2]] - 0.5).abs() < 1e-10);
        assert!((result[[3]] - 0.3).abs() < 1e-10);
        assert!((result[[4]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fuzzy_xor() {
        let fuzzy = FuzzyLogic::new(FuzzyConfig::product()).unwrap();
        let a = arr(vec![0.0, 1.0, 0.0, 1.0]);
        let b = arr(vec![0.0, 0.0, 1.0, 1.0]);

        let result = fuzzy.xor(&a, &b).unwrap();

        // XOR truth table (fuzzy approximation)
        assert!(result[[0]] < 0.1); // 0 XOR 0 = 0
        assert!(result[[1]] > 0.9); // 1 XOR 0 = 1
        assert!(result[[2]] > 0.9); // 0 XOR 1 = 1
        assert!(result[[3]] < 0.1); // 1 XOR 1 = 0
    }

    #[test]
    fn test_fuzzy_nand_nor() {
        let fuzzy = FuzzyLogic::new(FuzzyConfig::godel()).unwrap();
        let a = arr(vec![0.8, 0.3]);
        let b = arr(vec![0.6, 0.4]);

        let nand_result = fuzzy.nand(&a, &b).unwrap();
        let nor_result = fuzzy.nor(&a, &b).unwrap();

        // NAND = NOT(AND)
        assert!((nand_result[[0]] - 0.4).abs() < 1e-10); // 1 - min(0.8, 0.6) = 0.4
        assert!((nand_result[[1]] - 0.7).abs() < 1e-10); // 1 - min(0.3, 0.4) = 0.7

        // NOR = NOT(OR)
        assert!((nor_result[[0]] - 0.2).abs() < 1e-10); // 1 - max(0.8, 0.6) = 0.2
        assert!((nor_result[[1]] - 0.6).abs() < 1e-10); // 1 - max(0.3, 0.4) = 0.6
    }

    #[test]
    fn test_and_many() {
        let fuzzy = FuzzyLogic::new(FuzzyConfig::godel()).unwrap();
        let a = arr(vec![0.8, 0.5]);
        let b = arr(vec![0.6, 0.7]);
        let c = arr(vec![0.4, 0.3]);

        let result = fuzzy.and_many(&[&a, &b, &c]).unwrap();

        assert!((result[[0]] - 0.4).abs() < 1e-10); // min(0.8, 0.6, 0.4)
        assert!((result[[1]] - 0.3).abs() < 1e-10); // min(0.5, 0.7, 0.3)
    }

    #[test]
    fn test_or_many() {
        let fuzzy = FuzzyLogic::new(FuzzyConfig::godel()).unwrap();
        let a = arr(vec![0.2, 0.5]);
        let b = arr(vec![0.6, 0.3]);
        let c = arr(vec![0.4, 0.7]);

        let result = fuzzy.or_many(&[&a, &b, &c]).unwrap();

        assert!((result[[0]] - 0.6).abs() < 1e-10); // max(0.2, 0.6, 0.4)
        assert!((result[[1]] - 0.7).abs() < 1e-10); // max(0.5, 0.3, 0.7)
    }

    #[test]
    fn test_soft_and_gradient() {
        let a = arr(vec![0.3, 0.8]);
        let b = arr(vec![0.7, 0.2]);
        let grad = arr(vec![1.0, 1.0]);

        let grad_a = soft_and_grad_a(&a, &b, &grad, 0.1);
        let grad_b = soft_and_grad_b(&a, &b, &grad, 0.1);

        // For soft min, gradient should flow more to the smaller input
        assert!(grad_a[[0]] > grad_b[[0]]); // a[0] < b[0], so grad_a should be larger
        assert!(grad_a[[1]] < grad_b[[1]]); // a[1] > b[1], so grad_b should be larger
    }

    #[test]
    fn test_annealing_constant() {
        let schedule = AnnealingSchedule::Constant;
        assert!((schedule.temperature(1.0, 0) - 1.0).abs() < 1e-10);
        assert!((schedule.temperature(1.0, 100) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_annealing_linear() {
        let schedule = AnnealingSchedule::Linear {
            t_max: 1.0,
            t_min: 0.1,
            steps: 100,
        };

        assert!((schedule.temperature(1.0, 0) - 1.0).abs() < 1e-10);
        assert!((schedule.temperature(1.0, 50) - 0.55).abs() < 1e-10);
        assert!((schedule.temperature(1.0, 100) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_annealing_exponential() {
        let schedule = AnnealingSchedule::Exponential {
            t_max: 1.0,
            decay: 0.9,
        };

        assert!((schedule.temperature(1.0, 0) - 1.0).abs() < 1e-10);
        assert!((schedule.temperature(1.0, 1) - 0.9).abs() < 1e-10);
        assert!((schedule.temperature(1.0, 2) - 0.81).abs() < 1e-10);
    }

    #[test]
    fn test_annealing_cosine() {
        let schedule = AnnealingSchedule::Cosine {
            t_max: 1.0,
            t_min: 0.0,
            steps: 100,
        };

        assert!((schedule.temperature(1.0, 0) - 1.0).abs() < 1e-10);
        assert!((schedule.temperature(1.0, 50) - 0.5).abs() < 0.01);
        assert!((schedule.temperature(1.0, 100) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_validation() {
        assert!(FuzzyConfig::soft(-1.0).validate().is_err());
        assert!(FuzzyConfig::soft(0.0).validate().is_err());
        assert!(FuzzyConfig::soft(0.1).validate().is_ok());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let fuzzy = FuzzyLogic::default_soft();
        let a = arr(vec![0.5, 0.5]);
        let b = arr(vec![0.5, 0.5, 0.5]);

        assert!(fuzzy.and(&a, &b).is_err());
        assert!(fuzzy.or(&a, &b).is_err());
    }
}
