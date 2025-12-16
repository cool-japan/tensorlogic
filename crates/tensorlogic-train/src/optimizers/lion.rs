//! Lion (EvoLved Sign Momentum) optimizer.
//!
//! Paper: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
//! <https://arxiv.org/abs/2302.06675>
//!
//! Lion is a simple yet effective optimizer that:
//! - Uses only the sign of momentum for updates
//! - Has only 2 hyperparameters (learning rate and betas)
//! - Requires less memory than Adam (no second moment)
//! - Often achieves better performance with larger batch sizes

use crate::error::TrainResult;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// Lion optimizer configuration.
#[derive(Debug, Clone)]
pub struct LionConfig {
    /// Learning rate (default: 1e-4, typically 3-10x smaller than Adam)
    pub learning_rate: f64,
    /// Momentum coefficient for update direction (default: 0.9)
    pub beta1: f64,
    /// Momentum coefficient for state update (default: 0.99)
    pub beta2: f64,
    /// Weight decay coefficient (default: 0.0)
    pub weight_decay: f64,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

/// Lion optimizer.
///
/// Update rule:
/// 1. c_t = β1 * m_{t-1} + (1 - β1) * g_t  (interpolation)
/// 2. θ_t = θ_{t-1} - lr * (sign(c_t) + λ * θ_{t-1})  (parameter update with weight decay)
/// 3. m_t = β2 * m_{t-1} + (1 - β2) * g_t  (momentum update)
///
/// Key differences from Adam:
/// - Uses sign(momentum) instead of normalized gradients
/// - Only tracks first moment (momentum), no second moment
/// - Typically requires smaller learning rates than Adam
/// - More memory efficient
pub struct LionOptimizer {
    config: LionConfig,
    /// Momentum buffers (first moment)
    momentum: HashMap<String, Array1<f64>>,
}

impl LionOptimizer {
    /// Create a new Lion optimizer.
    pub fn new(config: LionConfig) -> TrainResult<Self> {
        if config.learning_rate <= 0.0 {
            return Err(crate::error::TrainError::ConfigError(
                "Learning rate must be positive".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&config.beta1) {
            return Err(crate::error::TrainError::ConfigError(
                "beta1 must be in [0, 1)".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&config.beta2) {
            return Err(crate::error::TrainError::ConfigError(
                "beta2 must be in [0, 1)".to_string(),
            ));
        }
        if config.weight_decay < 0.0 {
            return Err(crate::error::TrainError::ConfigError(
                "weight_decay must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            config,
            momentum: HashMap::new(),
        })
    }

    /// Perform a single optimization step.
    pub fn step(
        &mut self,
        params: &mut HashMap<String, Array1<f64>>,
        gradients: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<()> {
        for (name, param) in params.iter_mut() {
            if let Some(grad) = gradients.get(name) {
                // Initialize momentum if needed
                let momentum = self
                    .momentum
                    .entry(name.clone())
                    .or_insert_with(|| Array1::zeros(param.len()));

                // Step 1: Interpolate for update direction
                // c_t = β1 * m_{t-1} + (1 - β1) * g_t
                let update_direction = momentum.mapv(|m| m * self.config.beta1)
                    + grad.mapv(|g| g * (1.0 - self.config.beta1));

                // Step 2: Parameter update using sign of update direction
                // θ_t = θ_{t-1} - lr * (sign(c_t) + λ * θ_{t-1})
                for i in 0..param.len() {
                    let sign_update = if update_direction[i] > 0.0 {
                        1.0
                    } else if update_direction[i] < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };

                    let update = sign_update + self.config.weight_decay * param[i];
                    param[i] -= self.config.learning_rate * update;
                }

                // Step 3: Update momentum
                // m_t = β2 * m_{t-1} + (1 - β2) * g_t
                *momentum = momentum.mapv(|m| m * self.config.beta2)
                    + grad.mapv(|g| g * (1.0 - self.config.beta2));
            }
        }

        Ok(())
    }

    /// Get the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.config.learning_rate
    }

    /// Set the learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    /// Get optimizer state for checkpointing.
    pub fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        self.momentum
            .iter()
            .map(|(k, v)| (format!("momentum.{}", k), v.to_vec()))
            .collect()
    }

    /// Load optimizer state from checkpoint.
    pub fn load_state_dict(&mut self, state: &HashMap<String, Vec<f64>>) -> TrainResult<()> {
        for (key, value) in state {
            if let Some(param_name) = key.strip_prefix("momentum.") {
                self.momentum
                    .insert(param_name.to_string(), Array1::from_vec(value.clone()));
            }
        }
        Ok(())
    }

    /// Reset optimizer state.
    pub fn reset(&mut self) {
        self.momentum.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use std::collections::HashMap;

    #[test]
    fn test_lion_optimizer() {
        let config = LionConfig::default();
        let mut optimizer = LionOptimizer::new(config).unwrap();

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));

        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), Array1::from_vec(vec![0.1, 0.2, 0.3]));

        // Perform optimization step
        optimizer.step(&mut params, &gradients).unwrap();

        // Parameters should have changed
        let w = params.get("w").unwrap();
        assert!(w[0] < 1.0);
        assert!(w[1] < 2.0);
        assert!(w[2] < 3.0);
    }

    #[test]
    fn test_lion_with_weight_decay() {
        let config = LionConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.01,
        };
        let mut optimizer = LionOptimizer::new(config).unwrap();

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array1::from_vec(vec![1.0, 1.0]));

        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), Array1::from_vec(vec![0.1, 0.1]));

        let initial_w = params.get("w").unwrap()[0];

        optimizer.step(&mut params, &gradients).unwrap();

        let updated_w = params.get("w").unwrap()[0];
        // With weight decay, the update should be larger
        assert!(updated_w < initial_w);
    }

    #[test]
    fn test_lion_sign_based_update() {
        let config = LionConfig {
            learning_rate: 1e-2,
            beta1: 0.0, // No momentum for clearer test
            beta2: 0.0,
            weight_decay: 0.0,
        };
        let mut optimizer = LionOptimizer::new(config).unwrap();

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array1::from_vec(vec![1.0, 1.0, 1.0]));

        let mut gradients = HashMap::new();
        gradients.insert(
            "w".to_string(),
            Array1::from_vec(vec![0.1, 1.0, 100.0]), // Different magnitudes
        );

        optimizer.step(&mut params, &gradients).unwrap();

        let w = params.get("w").unwrap();
        // All updates should be the same magnitude (sign-based)
        let delta0 = 1.0 - w[0];
        let delta1 = 1.0 - w[1];
        let delta2 = 1.0 - w[2];

        assert!((delta0 - delta1).abs() < 1e-10);
        assert!((delta1 - delta2).abs() < 1e-10);
    }

    #[test]
    fn test_lion_state_dict() {
        let config = LionConfig::default();
        let mut optimizer = LionOptimizer::new(config).unwrap();

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array1::from_vec(vec![1.0, 2.0]));

        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), Array1::from_vec(vec![0.1, 0.2]));

        optimizer.step(&mut params, &gradients).unwrap();

        // Save state
        let state = optimizer.state_dict();
        assert!(state.contains_key("momentum.w"));

        // Create new optimizer and load state
        let mut optimizer2 = LionOptimizer::new(LionConfig::default()).unwrap();
        optimizer2.load_state_dict(&state).unwrap();

        // States should match
        assert_eq!(
            optimizer.momentum.get("w").unwrap().to_vec(),
            optimizer2.momentum.get("w").unwrap().to_vec()
        );
    }

    #[test]
    fn test_lion_lr_schedule() {
        let config = LionConfig::default();
        let mut optimizer = LionOptimizer::new(config).unwrap();

        assert!((optimizer.get_lr() - 1e-4).abs() < 1e-10);

        optimizer.set_lr(1e-3);
        assert!((optimizer.get_lr() - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn test_lion_invalid_config() {
        let config = LionConfig {
            learning_rate: -1.0,
            ..Default::default()
        };
        assert!(LionOptimizer::new(config).is_err());

        let config = LionConfig {
            beta1: 1.5,
            ..Default::default()
        };
        assert!(LionOptimizer::new(config).is_err());

        let config = LionConfig {
            beta2: -0.1,
            ..Default::default()
        };
        assert!(LionOptimizer::new(config).is_err());

        let config = LionConfig {
            weight_decay: -0.1,
            ..Default::default()
        };
        assert!(LionOptimizer::new(config).is_err());
    }

    #[test]
    fn test_lion_reset() {
        let config = LionConfig::default();
        let mut optimizer = LionOptimizer::new(config).unwrap();

        let mut params = HashMap::new();
        params.insert("w".to_string(), Array1::from_vec(vec![1.0]));

        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), Array1::from_vec(vec![0.1]));

        optimizer.step(&mut params, &gradients).unwrap();
        assert!(!optimizer.momentum.is_empty());

        optimizer.reset();
        assert!(optimizer.momentum.is_empty());
    }
}
