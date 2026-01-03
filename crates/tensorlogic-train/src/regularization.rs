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

/// Spectral Normalization regularizer.
///
/// Normalizes weight matrices by their spectral norm (largest singular value).
/// Useful for stabilizing GAN training and improving generalization.
///
/// # References
/// - Miyato et al. (2018): "Spectral Normalization for Generative Adversarial Networks"
#[derive(Debug, Clone)]
pub struct SpectralNormalization {
    /// Target spectral norm (usually 1.0)
    pub target_norm: f64,
    /// Strength of the regularization
    pub lambda: f64,
    /// Number of power iterations for spectral norm estimation
    pub power_iterations: usize,
}

impl SpectralNormalization {
    /// Create a new spectral normalization regularizer.
    pub fn new(lambda: f64, target_norm: f64, power_iterations: usize) -> Self {
        Self {
            lambda,
            target_norm,
            power_iterations,
        }
    }

    /// Estimate spectral norm using power iteration.
    fn estimate_spectral_norm(&self, matrix: &Array<f64, Ix2>) -> f64 {
        if matrix.is_empty() {
            return 0.0;
        }

        let (nrows, ncols) = matrix.dim();
        if nrows == 0 || ncols == 0 {
            return 0.0;
        }

        // Initialize random vector
        let mut v = Array::from_elem((ncols,), 1.0 / (ncols as f64).sqrt());

        // Power iteration to find dominant singular value
        for _ in 0..self.power_iterations {
            // u = W * v
            let u = matrix.dot(&v);
            let u_norm = u.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if u_norm < 1e-10 {
                break;
            }
            let u = u / u_norm;

            // v = W^T * u
            v = matrix.t().dot(&u);
            let v_norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if v_norm < 1e-10 {
                break;
            }
            v /= v_norm;
        }

        // σ = ||W * v||
        let final_u = matrix.dot(&v);
        final_u.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

impl Default for SpectralNormalization {
    fn default() -> Self {
        Self {
            target_norm: 1.0,
            lambda: 0.01,
            power_iterations: 1,
        }
    }
}

impl Regularizer for SpectralNormalization {
    fn compute_penalty(&self, parameters: &HashMap<String, Array<f64, Ix2>>) -> TrainResult<f64> {
        let mut penalty = 0.0;

        for param in parameters.values() {
            let spectral_norm = self.estimate_spectral_norm(param);
            // Penalty for deviation from target norm
            penalty += (spectral_norm - self.target_norm).powi(2);
        }

        Ok(self.lambda * penalty)
    }

    fn compute_gradient(
        &self,
        parameters: &HashMap<String, Array<f64, Ix2>>,
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>> {
        let mut gradients = HashMap::new();

        for (name, param) in parameters {
            let spectral_norm = self.estimate_spectral_norm(param);
            if spectral_norm < 1e-10 {
                gradients.insert(name.clone(), Array::zeros(param.dim()));
                continue;
            }

            // Approximate gradient: ∇||W||_2 ≈ W / ||W||_F
            let frobenius_norm = param.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if frobenius_norm < 1e-10 {
                gradients.insert(name.clone(), Array::zeros(param.dim()));
                continue;
            }

            let scale = 2.0 * self.lambda * (spectral_norm - self.target_norm) / frobenius_norm;
            let grad = param.mapv(|w| scale * w);
            gradients.insert(name.clone(), grad);
        }

        Ok(gradients)
    }
}

/// MaxNorm constraint regularizer.
///
/// Constrains the norm of weight vectors to a maximum value.
/// Useful for preventing exploding gradients and improving stability.
///
/// # References
/// - Hinton et al.: "Improving neural networks by preventing co-adaptation"
#[derive(Debug, Clone)]
pub struct MaxNormRegularization {
    /// Maximum allowed norm
    pub max_norm: f64,
    /// Regularization strength
    pub lambda: f64,
    /// Axis along which to compute norms (0 for rows, 1 for columns)
    pub axis: usize,
}

impl MaxNormRegularization {
    /// Create a new max norm regularizer.
    pub fn new(max_norm: f64, lambda: f64, axis: usize) -> Self {
        Self {
            max_norm,
            lambda,
            axis,
        }
    }
}

impl Default for MaxNormRegularization {
    fn default() -> Self {
        Self {
            max_norm: 2.0,
            lambda: 0.01,
            axis: 0,
        }
    }
}

impl Regularizer for MaxNormRegularization {
    fn compute_penalty(&self, parameters: &HashMap<String, Array<f64, Ix2>>) -> TrainResult<f64> {
        let mut penalty = 0.0;

        for param in parameters.values() {
            let axis_len = if self.axis == 0 {
                param.nrows()
            } else {
                param.ncols()
            };

            for i in 0..axis_len {
                let row_or_col = if self.axis == 0 {
                    param.row(i)
                } else {
                    param.column(i)
                };

                let norm = row_or_col.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm > self.max_norm {
                    penalty += (norm - self.max_norm).powi(2);
                }
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
            let mut grad = Array::zeros(param.dim());

            let axis_len = if self.axis == 0 {
                param.nrows()
            } else {
                param.ncols()
            };

            for i in 0..axis_len {
                let row_or_col = if self.axis == 0 {
                    param.row(i)
                } else {
                    param.column(i)
                };

                let norm = row_or_col.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm > self.max_norm {
                    let scale = 2.0 * self.lambda * (norm - self.max_norm) / (norm + 1e-10);

                    for (j, &val) in row_or_col.iter().enumerate() {
                        if self.axis == 0 {
                            grad[[i, j]] = scale * val;
                        } else {
                            grad[[j, i]] = scale * val;
                        }
                    }
                }
            }

            gradients.insert(name.clone(), grad);
        }

        Ok(gradients)
    }
}

/// Orthogonal regularization.
///
/// Encourages weight matrices to be orthogonal: W^T * W ≈ I
/// Helps prevent internal covariate shift and improves gradient flow.
///
/// # References
/// - Brock et al. (2017): "Neural Photo Editing with Introspective Adversarial Networks"
#[derive(Debug, Clone)]
pub struct OrthogonalRegularization {
    /// Regularization strength
    pub lambda: f64,
}

impl OrthogonalRegularization {
    /// Create a new orthogonal regularizer.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Default for OrthogonalRegularization {
    fn default() -> Self {
        Self { lambda: 0.01 }
    }
}

impl Regularizer for OrthogonalRegularization {
    fn compute_penalty(&self, parameters: &HashMap<String, Array<f64, Ix2>>) -> TrainResult<f64> {
        let mut penalty = 0.0;

        for param in parameters.values() {
            // Compute W^T * W
            let wt_w = param.t().dot(param);

            // Compute ||W^T * W - I||_F^2
            let (n, _) = wt_w.dim();
            for i in 0..n {
                for j in 0..n {
                    let target = if i == j { 1.0 } else { 0.0 };
                    let diff = wt_w[[i, j]] - target;
                    penalty += diff * diff;
                }
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
            // W^T * W
            let wt_w = param.t().dot(param);

            // Create identity matrix
            let (n, _) = wt_w.dim();
            let mut identity = Array::zeros((n, n));
            for i in 0..n {
                identity[[i, i]] = 1.0;
            }

            // Gradient: 2 * λ * W * (W^T * W - I)
            let diff = &wt_w - &identity;
            let grad = param.dot(&diff) * (2.0 * self.lambda);

            gradients.insert(name.clone(), grad);
        }

        Ok(gradients)
    }
}

/// Group Lasso regularization.
///
/// Encourages group-wise sparsity by penalizing the L2 norm of groups.
/// Useful when features have natural groupings.
///
/// # References
/// - Yuan & Lin (2006): "Model selection and estimation in regression with grouped variables"
#[derive(Debug, Clone)]
pub struct GroupLassoRegularization {
    /// Regularization strength
    pub lambda: f64,
    /// Group size (number of consecutive parameters per group)
    pub group_size: usize,
}

impl GroupLassoRegularization {
    /// Create a new group lasso regularizer.
    pub fn new(lambda: f64, group_size: usize) -> Self {
        Self { lambda, group_size }
    }
}

impl Default for GroupLassoRegularization {
    fn default() -> Self {
        Self {
            lambda: 0.01,
            group_size: 10,
        }
    }
}

impl Regularizer for GroupLassoRegularization {
    fn compute_penalty(&self, parameters: &HashMap<String, Array<f64, Ix2>>) -> TrainResult<f64> {
        let mut penalty = 0.0;

        for param in parameters.values() {
            // Flatten to 1D
            let flat: Vec<f64> = param.iter().copied().collect();

            // Compute group norms
            for group in flat.chunks(self.group_size) {
                let group_norm = group.iter().map(|&x| x * x).sum::<f64>().sqrt();
                penalty += group_norm;
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
            let mut grad_flat = Vec::new();
            let flat: Vec<f64> = param.iter().copied().collect();

            for group in flat.chunks(self.group_size) {
                let group_norm = group.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if group_norm > 1e-10 {
                    let scale = self.lambda / group_norm;
                    grad_flat.extend(group.iter().map(|&x| scale * x));
                } else {
                    grad_flat.extend(vec![0.0; group.len()]);
                }
            }

            // Reshape back to original shape
            let grad = Array::from_shape_vec(param.dim(), grad_flat).map_err(|e| {
                TrainError::ModelError(format!("Failed to reshape gradient: {}", e))
            })?;
            gradients.insert(name.clone(), grad);
        }

        Ok(gradients)
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

    #[test]
    fn test_spectral_normalization() {
        let regularizer = SpectralNormalization::new(0.1, 1.0, 5);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[2.0, 0.0], [0.0, 1.0]]);

        let penalty = regularizer.compute_penalty(&params).unwrap();
        // Spectral norm of [[2,0],[0,1]] is 2.0
        // Penalty = 0.1 * (2.0 - 1.0)^2 = 0.1
        assert!((penalty - 0.1).abs() < 0.01);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        assert!(gradients.contains_key("w"));
    }

    #[test]
    fn test_max_norm_regularization() {
        let regularizer = MaxNormRegularization::new(1.0, 0.1, 0);

        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            array![[3.0, 4.0], [0.1, 0.1]], // First row has norm 5.0 > 1.0
        );

        let penalty = regularizer.compute_penalty(&params).unwrap();
        // First row: norm = 5.0, exceeds max_norm = 1.0
        // Penalty = 0.1 * (5.0 - 1.0)^2 = 1.6
        assert!((penalty - 1.6).abs() < 0.1);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        let grad_w = gradients.get("w").unwrap();
        // First row should have non-zero gradient
        assert!(grad_w[[0, 0]].abs() > 0.0);
        // Second row should have zero gradient (norm below max_norm)
        assert!(grad_w[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_orthogonal_regularization() {
        let regularizer = OrthogonalRegularization::new(0.1);

        let mut params = HashMap::new();
        // Identity matrix should have zero penalty
        params.insert("w".to_string(), array![[1.0, 0.0], [0.0, 1.0]]);

        let penalty = regularizer.compute_penalty(&params).unwrap();
        assert!(penalty.abs() < 1e-10);

        // Non-orthogonal matrix should have non-zero penalty
        params.insert("w".to_string(), array![[1.0, 1.0], [1.0, 1.0]]);
        let penalty = regularizer.compute_penalty(&params).unwrap();
        assert!(penalty > 0.0);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        assert!(gradients.contains_key("w"));
    }

    #[test]
    fn test_group_lasso_regularization() {
        let regularizer = GroupLassoRegularization::new(0.1, 2);

        let mut params = HashMap::new();
        params.insert(
            "w".to_string(),
            array![[1.0, 2.0], [3.0, 4.0]], // Flatten to [1,2,3,4], groups [1,2] and [3,4]
        );

        let penalty = regularizer.compute_penalty(&params).unwrap();
        // Group 1: sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.236
        // Group 2: sqrt(3^2 + 4^2) = sqrt(25) = 5.0
        // Total: 0.1 * (2.236 + 5.0) ≈ 0.7236
        assert!((penalty - 0.7236).abs() < 0.01);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        let grad_w = gradients.get("w").unwrap();
        assert_eq!(grad_w.dim(), (2, 2));
    }

    #[test]
    fn test_spectral_normalization_zero_matrix() {
        let regularizer = SpectralNormalization::new(0.1, 1.0, 5);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[0.0, 0.0], [0.0, 0.0]]);

        let penalty = regularizer.compute_penalty(&params).unwrap();
        // Spectral norm of zero matrix is 0
        // Penalty = 0.1 * (0 - 1.0)^2 = 0.1
        assert!((penalty - 0.1).abs() < 0.01);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        let grad_w = gradients.get("w").unwrap();
        // Gradient should be zero for zero matrix
        assert!(grad_w.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_max_norm_no_violation() {
        let regularizer = MaxNormRegularization::new(10.0, 0.1, 0);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);

        let penalty = regularizer.compute_penalty(&params).unwrap();
        // All norms are below 10.0, so no penalty
        assert!(penalty.abs() < 1e-10);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        let grad_w = gradients.get("w").unwrap();
        // All gradients should be zero
        assert!(grad_w.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_orthogonal_non_square() {
        let regularizer = OrthogonalRegularization::new(0.1);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);

        // Non-square matrix: W^T * W will be 3x3
        let penalty = regularizer.compute_penalty(&params).unwrap();
        assert!(penalty > 0.0); // Should have some penalty

        let gradients = regularizer.compute_gradient(&params).unwrap();
        assert!(gradients.contains_key("w"));
    }

    #[test]
    fn test_group_lasso_single_group() {
        let regularizer = GroupLassoRegularization::new(0.1, 4);

        let mut params = HashMap::new();
        params.insert("w".to_string(), array![[3.0, 4.0]]);

        let penalty = regularizer.compute_penalty(&params).unwrap();
        // Single group: sqrt(3^2 + 4^2) = 5.0
        // Penalty = 0.1 * 5.0 = 0.5
        assert!((penalty - 0.5).abs() < 0.01);

        let gradients = regularizer.compute_gradient(&params).unwrap();
        let grad_w = gradients.get("w").unwrap();
        assert_eq!(grad_w.dim(), (1, 2));
    }
}
