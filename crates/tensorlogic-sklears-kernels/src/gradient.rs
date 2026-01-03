#![allow(clippy::needless_range_loop)]
//! Matrix-level gradient computation for kernel hyperparameter optimization.
//!
//! This module provides utilities for computing gradients of kernel matrices
//! with respect to hyperparameters, essential for Gaussian Process optimization.
//!
//! ## Overview
//!
//! When optimizing kernel hyperparameters (e.g., via gradient descent or L-BFGS),
//! we need to compute the gradient of the kernel matrix K with respect to each
//! hyperparameter θ: dK/dθ (an N×N matrix of partial derivatives).
//!
//! ## Supported Kernels
//!
//! - **RBF/Gaussian**: dK/dγ (gamma) and dK/dl (length scale)
//! - **Polynomial**: dK/dc (constant) and dK/dd (degree, continuous)
//! - **Matérn**: dK/dl (length scale) for nu = 0.5, 1.5, 2.5
//! - **Laplacian**: dK/dγ (gamma) and dK/dσ (sigma)
//! - **Rational Quadratic**: dK/dl (length scale) and dK/dα (alpha)
//!
//! ## Example
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{RbfKernel, RbfKernelConfig};
//! use tensorlogic_sklears_kernels::gradient::{KernelGradientMatrix, compute_rbf_gradient_matrix};
//!
//! let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
//! let data = vec![
//!     vec![1.0, 2.0],
//!     vec![3.0, 4.0],
//!     vec![5.0, 6.0],
//! ];
//!
//! let grad_result = compute_rbf_gradient_matrix(&kernel, &data).unwrap();
//! // grad_result.kernel_matrix is the N×N kernel matrix K
//! // grad_result.gradient_gamma is dK/dγ (N×N matrix)
//! // grad_result.gradient_length_scale is dK/dl (N×N matrix)
//! ```

use crate::error::{KernelError, Result};
use crate::tensor_kernels::{
    LaplacianKernel, MaternKernel, PolynomialKernel, RationalQuadraticKernel, RbfKernel,
};

/// Result of computing kernel matrix with gradients.
#[derive(Debug, Clone)]
pub struct KernelGradientMatrix {
    /// The kernel matrix K (N×N)
    pub kernel_matrix: Vec<Vec<f64>>,
    /// Gradient matrices for each hyperparameter
    pub gradients: Vec<GradientComponent>,
}

/// A single gradient component with its name and matrix.
#[derive(Debug, Clone)]
pub struct GradientComponent {
    /// Name of the hyperparameter (e.g., "gamma", "length_scale")
    pub name: String,
    /// The gradient matrix dK/dθ (N×N)
    pub matrix: Vec<Vec<f64>>,
}

impl KernelGradientMatrix {
    /// Create a new kernel gradient matrix result.
    pub fn new(kernel_matrix: Vec<Vec<f64>>, gradients: Vec<GradientComponent>) -> Self {
        Self {
            kernel_matrix,
            gradients,
        }
    }

    /// Get the number of samples (N).
    pub fn n_samples(&self) -> usize {
        self.kernel_matrix.len()
    }

    /// Get a gradient by name.
    pub fn get_gradient(&self, name: &str) -> Option<&Vec<Vec<f64>>> {
        self.gradients
            .iter()
            .find(|g| g.name == name)
            .map(|g| &g.matrix)
    }

    /// Get all gradient names.
    pub fn gradient_names(&self) -> Vec<&str> {
        self.gradients.iter().map(|g| g.name.as_str()).collect()
    }
}

/// Result of RBF gradient computation with specific gradient accessors.
#[derive(Debug, Clone)]
pub struct RbfGradientResult {
    /// The kernel matrix K
    pub kernel_matrix: Vec<Vec<f64>>,
    /// Gradient w.r.t. gamma: dK/dγ
    pub gradient_gamma: Vec<Vec<f64>>,
    /// Gradient w.r.t. length scale: dK/dl (where l = 1/sqrt(2γ))
    pub gradient_length_scale: Vec<Vec<f64>>,
}

/// Compute the kernel matrix and gradients for an RBF kernel.
///
/// Returns the kernel matrix K and gradients dK/dγ and dK/dl.
///
/// # Arguments
/// * `kernel` - The RBF kernel
/// * `data` - Vector of N samples, each of dimension D
///
/// # Example
/// ```rust
/// use tensorlogic_sklears_kernels::{RbfKernel, RbfKernelConfig};
/// use tensorlogic_sklears_kernels::gradient::compute_rbf_gradient_matrix;
///
/// let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
/// let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// let result = compute_rbf_gradient_matrix(&kernel, &data).unwrap();
/// ```
pub fn compute_rbf_gradient_matrix(
    kernel: &RbfKernel,
    data: &[Vec<f64>],
) -> Result<RbfGradientResult> {
    let n = data.len();
    if n == 0 {
        return Err(KernelError::ComputationError("Empty data".to_string()));
    }

    let mut kernel_matrix = vec![vec![0.0; n]; n];
    let mut gradient_gamma = vec![vec![0.0; n]; n];
    let mut gradient_length_scale = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            let (k, grad_gamma) = kernel.compute_with_gradient(&data[i], &data[j])?;
            let (_, grad_ls) = kernel.compute_with_length_scale_gradient(&data[i], &data[j])?;

            kernel_matrix[i][j] = k;
            kernel_matrix[j][i] = k;

            gradient_gamma[i][j] = grad_gamma;
            gradient_gamma[j][i] = grad_gamma;

            gradient_length_scale[i][j] = grad_ls;
            gradient_length_scale[j][i] = grad_ls;
        }
    }

    Ok(RbfGradientResult {
        kernel_matrix,
        gradient_gamma,
        gradient_length_scale,
    })
}

/// Result of Polynomial gradient computation.
#[derive(Debug, Clone)]
pub struct PolynomialGradientResult {
    /// The kernel matrix K
    pub kernel_matrix: Vec<Vec<f64>>,
    /// Gradient w.r.t. constant: dK/dc
    pub gradient_constant: Vec<Vec<f64>>,
    /// Gradient w.r.t. degree (continuous): dK/dd
    pub gradient_degree: Vec<Vec<f64>>,
}

/// Compute the kernel matrix and gradients for a Polynomial kernel.
pub fn compute_polynomial_gradient_matrix(
    kernel: &PolynomialKernel,
    data: &[Vec<f64>],
) -> Result<PolynomialGradientResult> {
    let n = data.len();
    if n == 0 {
        return Err(KernelError::ComputationError("Empty data".to_string()));
    }

    let mut kernel_matrix = vec![vec![0.0; n]; n];
    let mut gradient_constant = vec![vec![0.0; n]; n];
    let mut gradient_degree = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            let (k, grad_c, grad_d) = kernel.compute_with_all_gradients(&data[i], &data[j])?;

            kernel_matrix[i][j] = k;
            kernel_matrix[j][i] = k;

            gradient_constant[i][j] = grad_c;
            gradient_constant[j][i] = grad_c;

            // Handle NaN for degree gradient (when base is negative with odd power)
            let grad_d_safe = if grad_d.is_nan() { 0.0 } else { grad_d };
            gradient_degree[i][j] = grad_d_safe;
            gradient_degree[j][i] = grad_d_safe;
        }
    }

    Ok(PolynomialGradientResult {
        kernel_matrix,
        gradient_constant,
        gradient_degree,
    })
}

/// Result of Matérn gradient computation.
#[derive(Debug, Clone)]
pub struct MaternGradientResult {
    /// The kernel matrix K
    pub kernel_matrix: Vec<Vec<f64>>,
    /// Gradient w.r.t. length scale: dK/dl
    pub gradient_length_scale: Vec<Vec<f64>>,
}

/// Compute the kernel matrix and gradients for a Matérn kernel.
pub fn compute_matern_gradient_matrix(
    kernel: &MaternKernel,
    data: &[Vec<f64>],
) -> Result<MaternGradientResult> {
    let n = data.len();
    if n == 0 {
        return Err(KernelError::ComputationError("Empty data".to_string()));
    }

    let mut kernel_matrix = vec![vec![0.0; n]; n];
    let mut gradient_length_scale = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            let (k, grad_l) = kernel.compute_with_length_scale_gradient(&data[i], &data[j])?;

            kernel_matrix[i][j] = k;
            kernel_matrix[j][i] = k;

            gradient_length_scale[i][j] = grad_l;
            gradient_length_scale[j][i] = grad_l;
        }
    }

    Ok(MaternGradientResult {
        kernel_matrix,
        gradient_length_scale,
    })
}

/// Result of Laplacian gradient computation.
#[derive(Debug, Clone)]
pub struct LaplacianGradientResult {
    /// The kernel matrix K
    pub kernel_matrix: Vec<Vec<f64>>,
    /// Gradient w.r.t. gamma: dK/dγ
    pub gradient_gamma: Vec<Vec<f64>>,
    /// Gradient w.r.t. sigma: dK/dσ
    pub gradient_sigma: Vec<Vec<f64>>,
}

/// Compute the kernel matrix and gradients for a Laplacian kernel.
pub fn compute_laplacian_gradient_matrix(
    kernel: &LaplacianKernel,
    data: &[Vec<f64>],
) -> Result<LaplacianGradientResult> {
    let n = data.len();
    if n == 0 {
        return Err(KernelError::ComputationError("Empty data".to_string()));
    }

    let mut kernel_matrix = vec![vec![0.0; n]; n];
    let mut gradient_gamma = vec![vec![0.0; n]; n];
    let mut gradient_sigma = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            let (k, grad_g) = kernel.compute_with_gradient(&data[i], &data[j])?;
            let (_, grad_s) = kernel.compute_with_sigma_gradient(&data[i], &data[j])?;

            kernel_matrix[i][j] = k;
            kernel_matrix[j][i] = k;

            gradient_gamma[i][j] = grad_g;
            gradient_gamma[j][i] = grad_g;

            gradient_sigma[i][j] = grad_s;
            gradient_sigma[j][i] = grad_s;
        }
    }

    Ok(LaplacianGradientResult {
        kernel_matrix,
        gradient_gamma,
        gradient_sigma,
    })
}

/// Result of Rational Quadratic gradient computation.
#[derive(Debug, Clone)]
pub struct RationalQuadraticGradientResult {
    /// The kernel matrix K
    pub kernel_matrix: Vec<Vec<f64>>,
    /// Gradient w.r.t. length scale: dK/dl
    pub gradient_length_scale: Vec<Vec<f64>>,
    /// Gradient w.r.t. alpha: dK/dα
    pub gradient_alpha: Vec<Vec<f64>>,
}

/// Compute the kernel matrix and gradients for a Rational Quadratic kernel.
pub fn compute_rational_quadratic_gradient_matrix(
    kernel: &RationalQuadraticKernel,
    data: &[Vec<f64>],
) -> Result<RationalQuadraticGradientResult> {
    let n = data.len();
    if n == 0 {
        return Err(KernelError::ComputationError("Empty data".to_string()));
    }

    let mut kernel_matrix = vec![vec![0.0; n]; n];
    let mut gradient_length_scale = vec![vec![0.0; n]; n];
    let mut gradient_alpha = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            let (k, grad_l, grad_a) = kernel.compute_with_all_gradients(&data[i], &data[j])?;

            kernel_matrix[i][j] = k;
            kernel_matrix[j][i] = k;

            gradient_length_scale[i][j] = grad_l;
            gradient_length_scale[j][i] = grad_l;

            gradient_alpha[i][j] = grad_a;
            gradient_alpha[j][i] = grad_a;
        }
    }

    Ok(RationalQuadraticGradientResult {
        kernel_matrix,
        gradient_length_scale,
        gradient_alpha,
    })
}

/// Generic function to compute gradient matrix for any kernel with gradient support.
///
/// This function creates a KernelGradientMatrix from a kernel and its gradient computation.
pub fn compute_generic_gradient_matrix<F>(
    data: &[Vec<f64>],
    kernel_fn: F,
    gradient_names: Vec<String>,
) -> Result<KernelGradientMatrix>
where
    F: Fn(&[f64], &[f64]) -> Result<(f64, Vec<f64>)>,
{
    let n = data.len();
    if n == 0 {
        return Err(KernelError::ComputationError("Empty data".to_string()));
    }

    let n_params = gradient_names.len();
    let mut kernel_matrix = vec![vec![0.0; n]; n];
    let mut gradient_matrices: Vec<Vec<Vec<f64>>> =
        (0..n_params).map(|_| vec![vec![0.0; n]; n]).collect();

    for i in 0..n {
        for j in i..n {
            let (k, grads) = kernel_fn(&data[i], &data[j])?;

            if grads.len() != n_params {
                return Err(KernelError::ComputationError(format!(
                    "Expected {} gradients, got {}",
                    n_params,
                    grads.len()
                )));
            }

            kernel_matrix[i][j] = k;
            kernel_matrix[j][i] = k;

            for (p, grad) in grads.iter().enumerate() {
                gradient_matrices[p][i][j] = *grad;
                gradient_matrices[p][j][i] = *grad;
            }
        }
    }

    let gradients = gradient_names
        .into_iter()
        .zip(gradient_matrices)
        .map(|(name, matrix)| GradientComponent { name, matrix })
        .collect();

    Ok(KernelGradientMatrix::new(kernel_matrix, gradients))
}

/// Compute the trace of the product of two matrices: tr(A * B).
///
/// Useful for computing gradients in GP log marginal likelihood:
/// d/dθ log p(y|X,θ) = 0.5 * tr((α α^T - K^{-1}) * dK/dθ)
///
/// # Arguments
/// * `a` - First matrix (N×N)
/// * `b` - Second matrix (N×N)
///
/// # Returns
/// The trace of A * B
pub fn trace_product(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<f64> {
    let n = a.len();
    if n == 0 {
        return Ok(0.0);
    }
    if b.len() != n {
        return Err(KernelError::DimensionMismatch {
            expected: vec![n],
            got: vec![b.len()],
            context: "trace_product matrix dimensions".to_string(),
        });
    }

    // tr(A * B) = sum_i (A * B)_{ii} = sum_i sum_j A_{ij} * B_{ji}
    let mut trace = 0.0;
    for i in 0..n {
        if a[i].len() != n || b[i].len() != n {
            return Err(KernelError::DimensionMismatch {
                expected: vec![n],
                got: vec![a[i].len()],
                context: "trace_product row dimension".to_string(),
            });
        }
        for j in 0..n {
            trace += a[i][j] * b[j][i];
        }
    }

    Ok(trace)
}

/// Compute the Frobenius norm of a matrix: ||A||_F = sqrt(sum_{i,j} A_{ij}^2).
///
/// Useful for gradient magnitude analysis and regularization.
pub fn frobenius_norm(matrix: &[Vec<f64>]) -> f64 {
    let sum_sq: f64 = matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|x| x * x)
        .sum();
    sum_sq.sqrt()
}

/// Check if a gradient matrix is symmetric (as it should be for symmetric kernels).
pub fn is_symmetric(matrix: &[Vec<f64>], tolerance: f64) -> bool {
    let n = matrix.len();
    for i in 0..n {
        for j in i + 1..n {
            if (matrix[i][j] - matrix[j][i]).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RbfKernelConfig;

    #[test]
    fn test_rbf_gradient_matrix() {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let result = compute_rbf_gradient_matrix(&kernel, &data).unwrap();

        // Check dimensions
        assert_eq!(result.kernel_matrix.len(), 3);
        assert_eq!(result.gradient_gamma.len(), 3);
        assert_eq!(result.gradient_length_scale.len(), 3);

        // Check symmetry
        assert!(is_symmetric(&result.kernel_matrix, 1e-10));
        assert!(is_symmetric(&result.gradient_gamma, 1e-10));
        assert!(is_symmetric(&result.gradient_length_scale, 1e-10));

        // Check diagonal is 1.0 for kernel matrix
        for i in 0..3 {
            assert!((result.kernel_matrix[i][i] - 1.0).abs() < 1e-10);
        }

        // Check diagonal gradient is 0 (same point)
        for i in 0..3 {
            assert!(result.gradient_gamma[i][i].abs() < 1e-10);
        }
    }

    #[test]
    fn test_polynomial_gradient_matrix() {
        let kernel = PolynomialKernel::new(2, 1.0).unwrap();
        let data = vec![vec![1.0, 2.0], vec![2.0, 3.0]];

        let result = compute_polynomial_gradient_matrix(&kernel, &data).unwrap();

        assert_eq!(result.kernel_matrix.len(), 2);
        assert!(is_symmetric(&result.kernel_matrix, 1e-10));
        assert!(is_symmetric(&result.gradient_constant, 1e-10));
    }

    #[test]
    fn test_matern_gradient_matrix() {
        let kernel = MaternKernel::nu_3_2(1.0).unwrap();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let result = compute_matern_gradient_matrix(&kernel, &data).unwrap();

        assert_eq!(result.kernel_matrix.len(), 2);
        assert!(is_symmetric(&result.kernel_matrix, 1e-10));
        assert!(is_symmetric(&result.gradient_length_scale, 1e-10));
    }

    #[test]
    fn test_laplacian_gradient_matrix() {
        let kernel = LaplacianKernel::new(0.5).unwrap();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let result = compute_laplacian_gradient_matrix(&kernel, &data).unwrap();

        assert_eq!(result.kernel_matrix.len(), 2);
        assert!(is_symmetric(&result.kernel_matrix, 1e-10));
        assert!(is_symmetric(&result.gradient_gamma, 1e-10));
        assert!(is_symmetric(&result.gradient_sigma, 1e-10));
    }

    #[test]
    fn test_rational_quadratic_gradient_matrix() {
        let kernel = RationalQuadraticKernel::new(1.0, 2.0).unwrap();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let result = compute_rational_quadratic_gradient_matrix(&kernel, &data).unwrap();

        assert_eq!(result.kernel_matrix.len(), 2);
        assert!(is_symmetric(&result.kernel_matrix, 1e-10));
        assert!(is_symmetric(&result.gradient_length_scale, 1e-10));
        assert!(is_symmetric(&result.gradient_alpha, 1e-10));
    }

    #[test]
    fn test_trace_product() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let trace = trace_product(&a, &b).unwrap();

        // A * B = [[19, 22], [43, 50]]
        // trace = 19 + 50 = 69
        assert!((trace - 69.0).abs() < 1e-10);
    }

    #[test]
    fn test_frobenius_norm() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let norm = frobenius_norm(&matrix);

        // ||A||_F = sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let expected = 30.0_f64.sqrt();
        assert!((norm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_is_symmetric() {
        let symmetric = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let asymmetric = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        assert!(is_symmetric(&symmetric, 1e-10));
        assert!(!is_symmetric(&asymmetric, 1e-10));
    }

    #[test]
    fn test_kernel_gradient_matrix_accessors() {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let rbf_result = compute_rbf_gradient_matrix(&kernel, &data).unwrap();

        // Convert to generic format
        let gradients = vec![
            GradientComponent {
                name: "gamma".to_string(),
                matrix: rbf_result.gradient_gamma,
            },
            GradientComponent {
                name: "length_scale".to_string(),
                matrix: rbf_result.gradient_length_scale,
            },
        ];
        let result = KernelGradientMatrix::new(rbf_result.kernel_matrix, gradients);

        assert_eq!(result.n_samples(), 2);
        assert_eq!(result.gradient_names(), vec!["gamma", "length_scale"]);
        assert!(result.get_gradient("gamma").is_some());
        assert!(result.get_gradient("length_scale").is_some());
        assert!(result.get_gradient("nonexistent").is_none());
    }

    #[test]
    fn test_empty_data() {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let data: Vec<Vec<f64>> = vec![];

        let result = compute_rbf_gradient_matrix(&kernel, &data);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_point() {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let data = vec![vec![1.0, 2.0]];

        let result = compute_rbf_gradient_matrix(&kernel, &data).unwrap();

        // Single point: K = [[1.0]], gradients = [[0.0]]
        assert_eq!(result.kernel_matrix.len(), 1);
        assert!((result.kernel_matrix[0][0] - 1.0).abs() < 1e-10);
        assert!(result.gradient_gamma[0][0].abs() < 1e-10);
    }

    #[test]
    fn test_gradient_consistency_with_element_wise() {
        // Verify matrix gradients match element-wise computation
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let result = compute_rbf_gradient_matrix(&kernel, &data).unwrap();

        // Check each element
        for i in 0..2 {
            for j in 0..2 {
                let (k, grad_g) = kernel.compute_with_gradient(&data[i], &data[j]).unwrap();
                assert!(
                    (result.kernel_matrix[i][j] - k).abs() < 1e-10,
                    "K[{},{}] mismatch",
                    i,
                    j
                );
                assert!(
                    (result.gradient_gamma[i][j] - grad_g).abs() < 1e-10,
                    "dK/dγ[{},{}] mismatch",
                    i,
                    j
                );
            }
        }
    }
}
