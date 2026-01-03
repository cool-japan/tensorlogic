//! Composite kernels for combining multiple kernel functions.
//!
//! This module provides ways to combine existing kernels through:
//! - Weighted sum (convex combinations)
//! - Product (multiplicative combinations)
//! - Kernel alignment (meta-learning)

use std::sync::Arc;

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// Weighted sum of multiple kernels: K(x,y) = Σ_i w_i * K_i(x,y)
///
/// Combines multiple kernels using weighted averaging.
/// Weights should sum to 1.0 for proper normalization.
///
/// # Example
///
/// ```rust
/// use tensorlogic_sklears_kernels::{
///     LinearKernel, RbfKernel, RbfKernelConfig,
///     WeightedSumKernel, Kernel
/// };
///
/// let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
/// let rbf = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;
///
/// let weights = vec![0.7, 0.3];
/// let composite = WeightedSumKernel::new(vec![linear, rbf], weights).unwrap();
///
/// let x = vec![1.0, 2.0, 3.0];
/// let y = vec![4.0, 5.0, 6.0];
/// let sim = composite.compute(&x, &y).unwrap();
/// // sim = 0.7 * linear(x,y) + 0.3 * rbf(x,y)
/// ```
pub struct WeightedSumKernel {
    /// Component kernels
    kernels: Vec<Arc<dyn Kernel>>,
    /// Weights for each kernel
    weights: Vec<f64>,
    /// Whether weights are normalized
    normalized: bool,
}

impl WeightedSumKernel {
    /// Create a new weighted sum kernel
    pub fn new(kernels: Vec<Box<dyn Kernel>>, weights: Vec<f64>) -> Result<Self> {
        if kernels.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "kernels".to_string(),
                value: "empty".to_string(),
                reason: "at least one kernel required".to_string(),
            });
        }

        if kernels.len() != weights.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![kernels.len()],
                got: vec![weights.len()],
                context: "weighted sum kernel".to_string(),
            });
        }

        // Check weights are non-negative
        if weights.iter().any(|&w| w < 0.0) {
            return Err(KernelError::InvalidParameter {
                parameter: "weights".to_string(),
                value: format!("{:?}", weights),
                reason: "all weights must be non-negative".to_string(),
            });
        }

        let weight_sum: f64 = weights.iter().sum();
        if weight_sum <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "weights".to_string(),
                value: format!("{:?}", weights),
                reason: "weights must sum to a positive value".to_string(),
            });
        }

        // Convert Box to Arc for shared ownership
        let kernels: Vec<Arc<dyn Kernel>> = kernels.into_iter().map(Arc::from).collect();

        Ok(Self {
            kernels,
            weights,
            normalized: false,
        })
    }

    /// Create with normalized weights (sum to 1.0)
    pub fn new_normalized(kernels: Vec<Box<dyn Kernel>>, mut weights: Vec<f64>) -> Result<Self> {
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "weights".to_string(),
                value: format!("{:?}", weights),
                reason: "weights must sum to a positive value".to_string(),
            });
        }

        // Normalize weights
        for w in &mut weights {
            *w /= weight_sum;
        }

        let mut kernel = Self::new(kernels, weights)?;
        kernel.normalized = true;
        Ok(kernel)
    }

    /// Create with uniform weights
    pub fn uniform(kernels: Vec<Box<dyn Kernel>>) -> Result<Self> {
        let n = kernels.len();
        if n == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "kernels".to_string(),
                value: "empty".to_string(),
                reason: "at least one kernel required".to_string(),
            });
        }

        let weights = vec![1.0 / n as f64; n];
        Self::new_normalized(kernels, weights)
    }
}

impl Kernel for WeightedSumKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let mut result = 0.0;

        for (kernel, &weight) in self.kernels.iter().zip(self.weights.iter()) {
            let value = kernel.compute(x, y)?;
            result += weight * value;
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "WeightedSum"
    }

    fn is_psd(&self) -> bool {
        // Weighted sum of PSD kernels is PSD if weights are non-negative
        self.weights.iter().all(|&w| w >= 0.0) && self.kernels.iter().all(|k| k.is_psd())
    }
}

/// Product of multiple kernels: K(x,y) = Π_i K_i(x,y)
///
/// Combines kernels through multiplication.
/// The resulting kernel corresponds to the tensor product of feature spaces.
///
/// # Example
///
/// ```rust
/// use tensorlogic_sklears_kernels::{
///     LinearKernel, CosineKernel,
///     ProductKernel, Kernel
/// };
///
/// let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
/// let cosine = Box::new(CosineKernel::new()) as Box<dyn Kernel>;
///
/// let product = ProductKernel::new(vec![linear, cosine]).unwrap();
///
/// let x = vec![1.0, 2.0, 3.0];
/// let y = vec![4.0, 5.0, 6.0];
/// let sim = product.compute(&x, &y).unwrap();
/// // sim = linear(x,y) * cosine(x,y)
/// ```
pub struct ProductKernel {
    /// Component kernels
    kernels: Vec<Arc<dyn Kernel>>,
}

impl ProductKernel {
    /// Create a new product kernel
    pub fn new(kernels: Vec<Box<dyn Kernel>>) -> Result<Self> {
        if kernels.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "kernels".to_string(),
                value: "empty".to_string(),
                reason: "at least one kernel required".to_string(),
            });
        }

        // Convert Box to Arc for shared ownership
        let kernels: Vec<Arc<dyn Kernel>> = kernels.into_iter().map(Arc::from).collect();

        Ok(Self { kernels })
    }
}

impl Kernel for ProductKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let mut result = 1.0;

        for kernel in &self.kernels {
            let value = kernel.compute(x, y)?;
            result *= value;
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "Product"
    }

    fn is_psd(&self) -> bool {
        // Product of PSD kernels is PSD
        self.kernels.iter().all(|k| k.is_psd())
    }
}

/// Kernel alignment computation for measuring similarity between kernels.
///
/// Kernel alignment measures how well two kernels agree on a dataset.
/// It's useful for kernel selection and meta-learning.
///
/// # Formula
///
/// ```text
/// A(K1, K2) = <K1, K2>_F / (||K1||_F * ||K2||_F)
/// ```
///
/// Where `<·,·>_F` is the Frobenius inner product and `||·||_F` is the Frobenius norm.
pub struct KernelAlignment;

impl KernelAlignment {
    /// Compute centered kernel alignment between two kernel matrices
    ///
    /// # Arguments
    /// * `k1` - First kernel matrix
    /// * `k2` - Second kernel matrix
    ///
    /// # Returns
    /// Alignment score in range [-1, 1], where 1 means perfect alignment
    pub fn compute_alignment(k1: &[Vec<f64>], k2: &[Vec<f64>]) -> Result<f64> {
        if k1.is_empty() || k2.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "kernel_matrices".to_string(),
                value: "empty".to_string(),
                reason: "kernel matrices cannot be empty".to_string(),
            });
        }

        let n1 = k1.len();
        let n2 = k2.len();

        if n1 != n2 {
            return Err(KernelError::DimensionMismatch {
                expected: vec![n1, n1],
                got: vec![n2, n2],
                context: "kernel alignment".to_string(),
            });
        }

        // Check square matrices
        for (i, row) in k1.iter().enumerate() {
            if row.len() != n1 {
                return Err(KernelError::DimensionMismatch {
                    expected: vec![n1],
                    got: vec![row.len()],
                    context: format!("k1 row {}", i),
                });
            }
        }

        for (i, row) in k2.iter().enumerate() {
            if row.len() != n2 {
                return Err(KernelError::DimensionMismatch {
                    expected: vec![n2],
                    got: vec![row.len()],
                    context: format!("k2 row {}", i),
                });
            }
        }

        // Center the kernel matrices
        let k1_centered = Self::center_kernel_matrix(k1);
        let k2_centered = Self::center_kernel_matrix(k2);

        // Compute Frobenius inner product
        let mut inner_product = 0.0;
        for i in 0..n1 {
            for j in 0..n1 {
                inner_product += k1_centered[i][j] * k2_centered[i][j];
            }
        }

        // Compute Frobenius norms
        let norm1 = Self::frobenius_norm(&k1_centered);
        let norm2 = Self::frobenius_norm(&k2_centered);

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }

        Ok(inner_product / (norm1 * norm2))
    }

    /// Center a kernel matrix
    #[allow(clippy::needless_range_loop)]
    fn center_kernel_matrix(k: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = k.len();
        let mut centered = vec![vec![0.0; n]; n];

        // Compute row and column means
        let mut row_means = vec![0.0; n];
        let mut col_means = vec![0.0; n];
        let mut total_mean = 0.0;

        for i in 0..n {
            for j in 0..n {
                row_means[i] += k[i][j];
                col_means[j] += k[i][j];
                total_mean += k[i][j];
            }
        }

        for mean in &mut row_means {
            *mean /= n as f64;
        }
        for mean in &mut col_means {
            *mean /= n as f64;
        }
        total_mean /= (n * n) as f64;

        // Center the matrix
        for i in 0..n {
            for j in 0..n {
                centered[i][j] = k[i][j] - row_means[i] - col_means[j] + total_mean;
            }
        }

        centered
    }

    /// Compute Frobenius norm of a matrix
    fn frobenius_norm(k: &[Vec<f64>]) -> f64 {
        let mut sum_sq = 0.0;
        for row in k {
            for &val in row {
                sum_sq += val * val;
            }
        }
        sum_sq.sqrt()
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::tensor_kernels::{CosineKernel, LinearKernel, RbfKernel};
    use crate::types::RbfKernelConfig;

    #[test]
    fn test_weighted_sum_kernel() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let rbf = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;

        let weights = vec![0.7, 0.3];
        let kernel = WeightedSumKernel::new(vec![linear, rbf], weights).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        assert!(result > 0.0);
        assert_eq!(kernel.name(), "WeightedSum");
    }

    #[test]
    fn test_weighted_sum_normalized() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let cosine = Box::new(CosineKernel::new()) as Box<dyn Kernel>;

        let weights = vec![2.0, 3.0]; // Will be normalized to [0.4, 0.6]
        let kernel = WeightedSumKernel::new_normalized(vec![linear, cosine], weights).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        assert!(result > 0.0);
    }

    #[test]
    fn test_weighted_sum_uniform() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let cosine = Box::new(CosineKernel::new()) as Box<dyn Kernel>;
        let rbf = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;

        let kernel = WeightedSumKernel::uniform(vec![linear, cosine, rbf]).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        assert!(result > 0.0);
    }

    #[test]
    fn test_weighted_sum_empty_kernels() {
        let result = WeightedSumKernel::new(vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_sum_dimension_mismatch() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let result = WeightedSumKernel::new(vec![linear], vec![0.5, 0.5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_sum_negative_weights() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let result = WeightedSumKernel::new(vec![linear], vec![-0.5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_product_kernel() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let cosine = Box::new(CosineKernel::new()) as Box<dyn Kernel>;

        let kernel = ProductKernel::new(vec![linear, cosine]).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = kernel.compute(&x, &y).unwrap();
        assert!(result > 0.0);
        assert_eq!(kernel.name(), "Product");
    }

    #[test]
    fn test_product_kernel_empty() {
        let result = ProductKernel::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_product_psd_property() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let rbf = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;

        let kernel = ProductKernel::new(vec![linear, rbf]).unwrap();
        assert!(kernel.is_psd());
    }

    #[test]
    fn test_kernel_alignment() {
        // Create two similar kernel matrices
        let k1 = vec![
            vec![1.0, 0.8, 0.6],
            vec![0.8, 1.0, 0.7],
            vec![0.6, 0.7, 1.0],
        ];

        let k2 = vec![
            vec![1.0, 0.75, 0.55],
            vec![0.75, 1.0, 0.65],
            vec![0.55, 0.65, 1.0],
        ];

        let alignment = KernelAlignment::compute_alignment(&k1, &k2).unwrap();

        // Similar matrices should have high alignment
        assert!(alignment > 0.9);
        assert!(alignment <= 1.0);
    }

    #[test]
    fn test_kernel_alignment_identity() {
        let k = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];

        let alignment = KernelAlignment::compute_alignment(&k, &k).unwrap();

        // A kernel should have perfect alignment with itself
        assert!((alignment - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_alignment_dimension_mismatch() {
        let k1 = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let k2 = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];

        let result = KernelAlignment::compute_alignment(&k1, &k2);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_sum_kernel_matrix() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let cosine = Box::new(CosineKernel::new()) as Box<dyn Kernel>;

        let kernel = WeightedSumKernel::uniform(vec![linear, cosine]).unwrap();

        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let matrix = kernel.compute_matrix(&inputs).unwrap();
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-10);
            }
        }
    }
}
