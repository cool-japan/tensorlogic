//! SAM optimizer (Sharpness Aware Minimization).
//!
//! SAM seeks parameters that lie in neighborhoods having uniformly low loss,
//! improving model generalization. It requires two forward-backward passes per step:
//! one to compute the adversarial perturbation, and one to compute the actual gradient.
//!
//! Reference: Foret et al. "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021)
//!
//! Note: This is a wrapper optimizer. SAM requires special handling in the training loop
//! to perform two gradient computations per step. The typical usage is:
//! 1. Compute gradients at current parameters
//! 2. Compute adversarial perturbation
//! 3. Compute gradients at perturbed parameters
//! 4. Update with the perturbed gradients

use super::common::{compute_gradient_norm, Optimizer};
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// SAM optimizer (Sharpness Aware Minimization).
///
/// SAM seeks parameters that lie in neighborhoods having uniformly low loss,
/// improving model generalization. It requires two forward-backward passes per step:
/// one to compute the adversarial perturbation, and one to compute the actual gradient.
///
/// Reference: Foret et al. "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021)
///
/// Note: This is a wrapper optimizer. SAM requires special handling in the training loop
/// to perform two gradient computations per step. The typical usage is:
/// 1. Compute gradients at current parameters
/// 2. Compute adversarial perturbation
/// 3. Compute gradients at perturbed parameters
/// 4. Update with the perturbed gradients
#[derive(Debug)]
pub struct SamOptimizer<O: Optimizer> {
    /// Base optimizer (e.g., SGD, Adam).
    base_optimizer: O,
    /// Perturbation radius (rho).
    rho: f64,
    /// Stored perturbations for each parameter.
    perturbations: HashMap<String, Array<f64, Ix2>>,
}

impl<O: Optimizer> SamOptimizer<O> {
    /// Create a new SAM optimizer.
    ///
    /// # Arguments
    /// * `base_optimizer` - The base optimizer to use (SGD, Adam, etc.)
    /// * `rho` - Perturbation radius (typically 0.05)
    pub fn new(base_optimizer: O, rho: f64) -> TrainResult<Self> {
        if rho <= 0.0 {
            return Err(TrainError::OptimizerError(
                "SAM rho must be positive".to_string(),
            ));
        }
        Ok(Self {
            base_optimizer,
            rho,
            perturbations: HashMap::new(),
        })
    }

    /// Compute adversarial perturbations.
    ///
    /// This should be called with the first set of gradients to compute
    /// the perturbation direction.
    pub fn first_step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        let grad_norm = compute_gradient_norm(gradients);
        if grad_norm == 0.0 {
            return Ok(());
        }
        for (name, param) in parameters.iter_mut() {
            let grad = gradients.get(name).ok_or_else(|| {
                TrainError::OptimizerError(format!("Missing gradient for parameter: {}", name))
            })?;
            let perturbation = grad.mapv(|g| self.rho * g / grad_norm);
            *param = &*param + &perturbation;
            self.perturbations.insert(name.clone(), perturbation);
        }
        Ok(())
    }

    /// Perform the actual optimization step.
    ///
    /// This should be called with the second set of gradients (computed at the perturbed parameters).
    /// It will remove the perturbations and update the parameters using the base optimizer.
    pub fn second_step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        for (name, param) in parameters.iter_mut() {
            if let Some(perturbation) = self.perturbations.get(name) {
                *param = &*param - perturbation;
            }
        }
        self.perturbations.clear();
        self.base_optimizer.step(parameters, gradients)
    }
}

impl<O: Optimizer> Optimizer for SamOptimizer<O> {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        self.second_step(parameters, gradients)
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
    }

    fn get_lr(&self) -> f64 {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f64) {
        self.base_optimizer.set_lr(lr);
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = self.base_optimizer.state_dict();
        state.insert("rho".to_string(), vec![self.rho]);
        for (name, perturbation) in &self.perturbations {
            state.insert(
                format!("perturbation_{}", name),
                perturbation.iter().copied().collect(),
            );
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        if let Some(rho_val) = state.get("rho") {
            self.rho = rho_val[0];
        }
        self.base_optimizer.load_state_dict(state.clone());
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("perturbation_") {
                if let Some(pert) = self.perturbations.get(name) {
                    let shape = pert.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.perturbations.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::common::OptimizerConfig;
    use super::super::sgd::SgdOptimizer;
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sam_optimizer() {
        let inner_config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let inner_optimizer = SgdOptimizer::new(inner_config);
        let mut optimizer = SamOptimizer::new(inner_optimizer, 0.05).unwrap();
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1]]);
        let original_w = params.get("w").unwrap().clone();
        optimizer.first_step(&mut params, &grads).unwrap();
        let perturbed_w = params.get("w").unwrap();
        assert_ne!(perturbed_w[[0, 0]], original_w[[0, 0]]);
        optimizer.second_step(&mut params, &grads).unwrap();
        let final_w = params.get("w").unwrap();
        assert!(final_w[[0, 0]] < original_w[[0, 0]]);
        let state = optimizer.state_dict();
        assert!(state.contains_key("rho"));
    }

    #[test]
    fn test_sam_invalid_rho() {
        let inner_optimizer = SgdOptimizer::new(OptimizerConfig::default());
        let result = SamOptimizer::new(inner_optimizer, 0.0);
        assert!(result.is_err());
        let inner_optimizer = SgdOptimizer::new(OptimizerConfig::default());
        let result = SamOptimizer::new(inner_optimizer, -0.1);
        assert!(result.is_err());
    }
}
