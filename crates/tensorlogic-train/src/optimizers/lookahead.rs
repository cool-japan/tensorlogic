//! Lookahead optimizer (wrapper that uses slow and fast weights).
//!
//! Lookahead maintains two sets of weights: fast weights updated by an inner optimizer,
//! and slow weights that are periodically updated as an exponential moving average.
//!
//! Reference: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019

use super::common::Optimizer;
use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// Lookahead optimizer (wrapper that uses slow and fast weights).
///
/// Maintains two sets of weights: fast weights updated by an inner optimizer,
/// and slow weights that are periodically updated as an exponential moving average.
///
/// Reference: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019
#[derive(Debug)]
pub struct LookaheadOptimizer<O: Optimizer> {
    /// Inner optimizer for fast weights.
    inner_optimizer: O,
    /// Slow weights (maintained separately).
    slow_weights: HashMap<String, Array<f64, Ix2>>,
    /// Interpolation coefficient (typically 0.5).
    alpha: f64,
    /// Number of inner optimizer steps before synchronization.
    k: usize,
    /// Current step counter.
    step_counter: usize,
}

impl<O: Optimizer> LookaheadOptimizer<O> {
    /// Create a new Lookahead optimizer.
    ///
    /// # Arguments
    /// * `inner_optimizer` - The inner optimizer (e.g., Adam, SGD)
    /// * `alpha` - Interpolation coefficient for slow weight update (typically 0.5)
    /// * `k` - Number of fast updates before slow weight synchronization (typically 5-10)
    pub fn new(inner_optimizer: O, alpha: f64, k: usize) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(TrainError::InvalidParameter(
                "alpha must be in [0, 1]".to_string(),
            ));
        }
        if k == 0 {
            return Err(TrainError::InvalidParameter(
                "k must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            inner_optimizer,
            slow_weights: HashMap::new(),
            alpha,
            k,
            step_counter: 0,
        })
    }

    /// Initialize slow weights from current parameters.
    fn initialize_slow_weights(&mut self, parameters: &HashMap<String, Array<f64, Ix2>>) {
        if self.slow_weights.is_empty() {
            for (name, param) in parameters {
                self.slow_weights.insert(name.clone(), param.clone());
            }
        }
    }

    /// Synchronize slow weights with fast weights.
    fn synchronize_weights(&mut self, parameters: &mut HashMap<String, Array<f64, Ix2>>) {
        for (name, param) in parameters.iter_mut() {
            if let Some(slow_weight) = self.slow_weights.get_mut(name) {
                *slow_weight = &*slow_weight + &((&*param - &*slow_weight) * self.alpha);
                *param = slow_weight.clone();
            }
        }
    }
}

impl<O: Optimizer> Optimizer for LookaheadOptimizer<O> {
    fn step(
        &mut self,
        parameters: &mut HashMap<String, Array<f64, Ix2>>,
        gradients: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<()> {
        self.initialize_slow_weights(parameters);
        self.inner_optimizer.step(parameters, gradients)?;
        self.step_counter += 1;
        if self.step_counter.is_multiple_of(self.k) {
            self.synchronize_weights(parameters);
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.inner_optimizer.zero_grad();
    }

    fn get_lr(&self) -> f64 {
        self.inner_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f64) {
        self.inner_optimizer.set_lr(lr);
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = self.inner_optimizer.state_dict();
        state.insert("step_counter".to_string(), vec![self.step_counter as f64]);
        state.insert("alpha".to_string(), vec![self.alpha]);
        state.insert("k".to_string(), vec![self.k as f64]);
        for (name, slow_weight) in &self.slow_weights {
            state.insert(
                format!("slow_{}", name),
                slow_weight.iter().copied().collect(),
            );
        }
        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        self.inner_optimizer.load_state_dict(state.clone());
        if let Some(counter) = state.get("step_counter") {
            self.step_counter = counter[0] as usize;
        }
        if let Some(alpha_val) = state.get("alpha") {
            self.alpha = alpha_val[0];
        }
        if let Some(k_val) = state.get("k") {
            self.k = k_val[0] as usize;
        }
        for (key, values) in state {
            if let Some(name) = key.strip_prefix("slow_") {
                if let Some(slow_weight) = self.slow_weights.get(name) {
                    let shape = slow_weight.raw_dim();
                    if let Ok(arr) = Array::from_shape_vec(shape, values) {
                        self.slow_weights.insert(name.to_string(), arr);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::adam::AdamOptimizer;
    use super::super::common::OptimizerConfig;
    use super::super::sgd::SgdOptimizer;
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_lookahead_optimizer() {
        let inner_config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let inner_optimizer = AdamOptimizer::new(inner_config);
        let mut optimizer = LookaheadOptimizer::new(inner_optimizer, 0.5, 5).unwrap();
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1, 0.1]]);
        for _ in 0..10 {
            optimizer.step(&mut params, &grads).unwrap();
        }
        let w = params.get("w").unwrap();
        assert!(w[[0, 0]] < 1.0);
        assert!(w[[0, 1]] < 2.0);
        assert_eq!(optimizer.get_lr(), 0.01);
        optimizer.set_lr(0.02);
        assert_eq!(optimizer.get_lr(), 0.02);
    }

    #[test]
    fn test_lookahead_invalid_alpha() {
        let inner_optimizer = AdamOptimizer::new(OptimizerConfig::default());
        let result = LookaheadOptimizer::new(inner_optimizer, 1.5, 5);
        assert!(result.is_err());
        let inner_optimizer = AdamOptimizer::new(OptimizerConfig::default());
        let result = LookaheadOptimizer::new(inner_optimizer, -0.1, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_lookahead_invalid_k() {
        let inner_optimizer = AdamOptimizer::new(OptimizerConfig::default());
        let result = LookaheadOptimizer::new(inner_optimizer, 0.5, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_lookahead_synchronization() {
        let inner_config = OptimizerConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let inner_optimizer = SgdOptimizer::new(inner_config);
        let mut optimizer = LookaheadOptimizer::new(inner_optimizer, 0.5, 3).unwrap();
        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0]]);
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), array![[0.1]]);
        let initial_w = params.get("w").unwrap()[[0, 0]];
        for _ in 0..3 {
            optimizer.step(&mut params, &grads).unwrap();
        }
        let w_after_sync = params.get("w").unwrap()[[0, 0]];
        assert_ne!(w_after_sync, initial_w);
        assert!(w_after_sync < initial_w);
    }
}
