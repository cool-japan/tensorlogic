//! Regularization techniques for training.
//!
//! This module provides various regularization strategies to prevent overfitting:
//! - L1 regularization (Lasso): Encourages sparsity
//! - L2 regularization (Ridge): Prevents large weights
//! - Composite regularization: Combines multiple regularizers

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

/// Trait for regularization strategies.
pub trait Regularizer {
    /// Compute the regularization penalty for given parameters.
    ///
    /// # Arguments
    /// * `parameters` - Model parameters to regularize
    ///
    /// # Returns
    /// The regularization penalty value
    fn compute_penalty(&self, parameters: &HashMap<String, Array<f64, Ix2>>) -> TrainResult<f64>;

    /// Compute the gradient of the regularization penalty.
    ///
    /// # Arguments
    /// * `parameters` - Model parameters
    ///
    /// # Returns
    /// Gradients of the regularization penalty for each parameter
    fn compute_gradient(
        &self,
        parameters: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>>;
}

/// L1 regularization (Lasso).
///
/// Adds penalty proportional to the absolute value of weights: λ * Σ|w|
/// Encourages sparsity by driving some weights to exactly zero.
#[derive(Debug, Clone)]
pub struct L1Regularization {
    /// Regularization strength (lambda).
    pub lambda: f64,
}

impl L1Regularization {
    /// Create a new L1 regularizer.
    ///
    /// # Arguments
    /// * `lambda` - Regularization strength
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Default for L1Regularization {
    fn default() -> Self {
        Self { lambda: 0.01 }
    }
}

impl Regularizer for L1Regularization {
    fn compute_penalty(&self, parameters: &HashMap<String, Array<f64, Ix2>>) -> TrainResult<f64> {
        let mut penalty = 0.0;

        for param in parameters.values() {
            for &value in param.iter() {
                penalty += value.abs();
            }
        }

        Ok(self.lambda * penalty)
    }

    fn compute_gradient(
        &self,
        parameters: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>> {
        let mut gradients = HashMap::new();

        for (name, param) in parameters {
            // Gradient of L1: λ * sign(w)
            let grad = param.mapv(|w| self.lambda * w.signum());
            gradients.insert(name.clone(), grad);
        }

        Ok(gradients)
    }
}

/// L2 regularization (Ridge / Weight Decay).
///
/// Adds penalty proportional to the square of weights: λ * Σw²
/// Prevents weights from becoming too large.
#[derive(Debug, Clone)]
pub struct L2Regularization {
    /// Regularization strength (lambda).
    pub lambda: f64,
}

impl L2Regularization {
    /// Create a new L2 regularizer.
    ///
    /// # Arguments
    /// * `lambda` - Regularization strength
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Default for L2Regularization {
    fn default() -> Self {
        Self { lambda: 0.01 }
    }
}

impl Regularizer for L2Regularization {
    fn compute_penalty(&self, parameters: &HashMap<String, Array<f64, Ix2>>) -> TrainResult<f64> {
        let mut penalty = 0.0;

        for param in parameters.values() {
            for &value in param.iter() {
                penalty += value * value;
            }
        }

        Ok(0.5 * self.lambda * penalty)
    }

    fn compute_gradient(
        &self,
        parameters: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>> {
        let mut gradients = HashMap::new();

        for (name, param) in parameters {
            // Gradient of L2: λ * w
            let grad = param.mapv(|w| self.lambda * w);
            gradients.insert(name.clone(), grad);
        }

        Ok(gradients)
    }
}

/// Elastic Net regularization (combination of L1 and L2).
///
/// Combines L1 and L2 penalties: l1_ratio * L1 + (1 - l1_ratio) * L2
#[derive(Debug, Clone)]
pub struct ElasticNetRegularization {
    /// Overall regularization strength.
    pub lambda: f64,
    /// Balance between L1 and L2 (0.0 = pure L2, 1.0 = pure L1).
    pub l1_ratio: f64,
}

impl ElasticNetRegularization {
    /// Create a new Elastic Net regularizer.
    ///
    /// # Arguments
    /// * `lambda` - Overall regularization strength
    /// * `l1_ratio` - Balance between L1 and L2 (should be in [0, 1])
    pub fn new(lambda: f64, l1_ratio: f64) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(TrainError::InvalidParameter(
                "l1_ratio must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(Self { lambda, l1_ratio })
    }
}

impl Default for ElasticNetRegularization {
    fn default() -> Self {
        Self {
            lambda: 0.01,
            l1_ratio: 0.5,
        }
    }
}

impl Regularizer for ElasticNetRegularization {
    fn compute_penalty(&self, parameters: &HashMap<String, Array<f64, Ix2>>) -> TrainResult<f64> {
        let mut l1_penalty = 0.0;
        let mut l2_penalty = 0.0;

        for param in parameters.values() {
            for &value in param.iter() {
                l1_penalty += value.abs();
                l2_penalty += value * value;
            }
        }

        let penalty =
            self.lambda * (self.l1_ratio * l1_penalty + (1.0 - self.l1_ratio) * 0.5 * l2_penalty);

        Ok(penalty)
    }

    fn compute_gradient(
        &self,
        parameters: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>> {
        let mut gradients = HashMap::new();

        for (name, param) in parameters {
            // Gradient: λ * (l1_ratio * sign(w) + (1 - l1_ratio) * w)
            let grad = param
                .mapv(|w| self.lambda * (self.l1_ratio * w.signum() + (1.0 - self.l1_ratio) * w));
            gradients.insert(name.clone(), grad);
        }

        Ok(gradients)
    }
}

/// Composite regularization that combines multiple regularizers.
///
/// Useful for applying different regularization strategies simultaneously.
#[derive(Clone)]
pub struct CompositeRegularization {
    regularizers: Vec<Box<dyn RegularizerClone>>,
}

impl std::fmt::Debug for CompositeRegularization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeRegularization")
            .field("num_regularizers", &self.regularizers.len())
            .finish()
    }
}

/// Helper trait for cloning boxed regularizers.
trait RegularizerClone: Regularizer {
    fn clone_box(&self) -> Box<dyn RegularizerClone>;
}

impl<T: Regularizer + Clone + 'static> RegularizerClone for T {
    fn clone_box(&self) -> Box<dyn RegularizerClone> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn RegularizerClone> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl Regularizer for Box<dyn RegularizerClone> {
    fn compute_penalty(&self, parameters: &HashMap<String, Array<f64, Ix2>>) -> TrainResult<f64> {
        (**self).compute_penalty(parameters)
    }

    fn compute_gradient(
        &self,
        parameters: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>> {
        (**self).compute_gradient(parameters)
    }
}

impl CompositeRegularization {
    /// Create a new composite regularizer.
    pub fn new() -> Self {
        Self {
            regularizers: Vec::new(),
        }
    }

    /// Add a regularizer to the composite.
    ///
    /// # Arguments
    /// * `regularizer` - Regularizer to add
    pub fn add<R: Regularizer + Clone + 'static>(&mut self, regularizer: R) {
        self.regularizers.push(Box::new(regularizer));
    }

    /// Get the number of regularizers in the composite.
    pub fn len(&self) -> usize {
        self.regularizers.len()
    }

    /// Check if the composite is empty.
    pub fn is_empty(&self) -> bool {
        self.regularizers.is_empty()
    }
}

impl Default for CompositeRegularization {
    fn default() -> Self {
        Self::new()
    }
}

impl Regularizer for CompositeRegularization {
    fn compute_penalty(&self, parameters: &HashMap<String, Array<f64, Ix2>>) -> TrainResult<f64> {
        let mut total_penalty = 0.0;

        for regularizer in &self.regularizers {
            total_penalty += regularizer.compute_penalty(parameters)?;
        }

        Ok(total_penalty)
    }

    fn compute_gradient(
        &self,
        parameters: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>> {
        let mut total_gradients: HashMap<String, Array<f64, Ix2>> = HashMap::new();

        // Initialize with zeros
        for (name, param) in parameters {
            total_gradients.insert(name.clone(), Array::zeros(param.raw_dim()));
        }

        // Accumulate gradients from all regularizers
        for regularizer in &self.regularizers {
            let grads = regularizer.compute_gradient(parameters)?;

            for (name, grad) in grads {
                if let Some(total_grad) = total_gradients.get_mut(&name) {
                    *total_grad = &*total_grad + &grad;
                }
            }
        }

        Ok(total_gradients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_l1_regularization() {
        let regularizer = L1Regularization::new(0.1);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, -2.0], [3.0, -4.0]]);

        let penalty = regularizer.compute_penalty(&params).unwrap();
        // Expected: 0.1 * (1 + 2 + 3 + 4) = 1.0
        assert!((penalty - 1.0).abs() < 1e-6);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        let grad_w = gradients.get("w").unwrap();

        // Gradient should be λ * sign(w)
        assert_eq!(grad_w[[0, 0]], 0.1); // sign(1.0) = 1.0
        assert_eq!(grad_w[[0, 1]], -0.1); // sign(-2.0) = -1.0
        assert_eq!(grad_w[[1, 0]], 0.1); // sign(3.0) = 1.0
        assert_eq!(grad_w[[1, 1]], -0.1); // sign(-4.0) = -1.0
    }

    #[test]
    fn test_l2_regularization() {
        let regularizer = L2Regularization::new(0.1);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let penalty = regularizer.compute_penalty(&params).unwrap();
        // Expected: 0.5 * 0.1 * (1 + 4 + 9 + 16) = 1.5
        assert!((penalty - 1.5).abs() < 1e-6);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        let grad_w = gradients.get("w").unwrap();

        // Gradient should be λ * w
        assert!((grad_w[[0, 0]] - 0.1).abs() < 1e-10); // 0.1 * 1.0
        assert!((grad_w[[0, 1]] - 0.2).abs() < 1e-10); // 0.1 * 2.0
        assert!((grad_w[[1, 0]] - 0.3).abs() < 1e-10); // 0.1 * 3.0
        assert!((grad_w[[1, 1]] - 0.4).abs() < 1e-10); // 0.1 * 4.0
    }

    #[test]
    fn test_elastic_net_regularization() {
        let regularizer = ElasticNetRegularization::new(0.1, 0.5).unwrap();

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let penalty = regularizer.compute_penalty(&params).unwrap();
        assert!(penalty > 0.0);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        let grad_w = gradients.get("w").unwrap();
        assert_eq!(grad_w.shape(), &[1, 2]);
    }

    #[test]
    fn test_elastic_net_invalid_ratio() {
        let result = ElasticNetRegularization::new(0.1, 1.5);
        assert!(result.is_err());

        let result = ElasticNetRegularization::new(0.1, -0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_composite_regularization() {
        let mut composite = CompositeRegularization::new();
        composite.add(L1Regularization::new(0.1));
        composite.add(L2Regularization::new(0.1));

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0]]);

        let penalty = composite.compute_penalty(&params).unwrap();
        // L1: 0.1 * (1 + 2) = 0.3
        // L2: 0.5 * 0.1 * (1 + 4) = 0.25
        // Total: 0.55
        assert!((penalty - 0.55).abs() < 1e-6);

        let gradients = composite.compute_gradient(&params).unwrap();
        let grad_w = gradients.get("w").unwrap();
        assert_eq!(grad_w.shape(), &[1, 2]);

        // Gradient should combine both L1 and L2
        // For w[0,0] = 1.0: L1 grad = 0.1, L2 grad = 0.1, total = 0.2
        assert!((grad_w[[0, 0]] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_composite_empty() {
        let composite = CompositeRegularization::new();
        assert!(composite.is_empty());
        assert_eq!(composite.len(), 0);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0]]);

        let penalty = composite.compute_penalty(&params).unwrap();
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn test_multiple_parameters() {
        let regularizer = L2Regularization::new(0.1);

        let mut params = HashMap::new();
        params.insert("w1".to_string(), array![[1.0, 2.0]]);
        params.insert("w2".to_string(), array![[3.0]]);

        let penalty = regularizer.compute_penalty(&params).unwrap();
        // Expected: 0.5 * 0.1 * (1 + 4 + 9) = 0.7
        assert!((penalty - 0.7).abs() < 1e-6);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        assert_eq!(gradients.len(), 2);
        assert!(gradients.contains_key("w1"));
        assert!(gradients.contains_key("w2"));
    }

    #[test]
    fn test_zero_lambda() {
        let regularizer = L1Regularization::new(0.0);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[100.0, 200.0]]);

        let penalty = regularizer.compute_penalty(&params).unwrap();
        assert_eq!(penalty, 0.0);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        let grad_w = gradients.get("w").unwrap();
        assert_eq!(grad_w[[0, 0]], 0.0);
        assert_eq!(grad_w[[0, 1]], 0.0);
    }
}
