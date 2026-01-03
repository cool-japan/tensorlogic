//! Kernel transformation utilities for preprocessing and normalization.
//!
//! This module provides utilities for transforming kernel matrices, including:
//! - Kernel normalization (normalize to unit diagonal)
//! - Kernel centering (for kernel PCA)
//! - Kernel standardization
//!
//! These transformations are essential for many kernel-based algorithms.

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// Normalize a kernel matrix to have unit diagonal entries.
///
/// Normalized kernel: K_norm(x,y) = K(x,y) / sqrt(K(x,x) * K(y,y))
///
/// This ensures all diagonal entries equal 1.0, which is useful for
/// algorithms that assume normalized kernels.
///
/// # Arguments
/// * `kernel_matrix` - Input kernel matrix (must be square)
///
/// # Returns
/// * Normalized kernel matrix
///
/// # Examples
/// ```
/// use tensorlogic_sklears_kernels::kernel_transform::normalize_kernel_matrix;
///
/// let K = vec![
///     vec![4.0, 2.0, 1.0],
///     vec![2.0, 9.0, 3.0],
///     vec![1.0, 3.0, 16.0],
/// ];
///
/// let K_norm = normalize_kernel_matrix(&K).unwrap();
///
/// // All diagonal entries should be 1.0
/// assert!((K_norm[0][0] - 1.0).abs() < 1e-10);
/// assert!((K_norm[1][1] - 1.0).abs() < 1e-10);
/// assert!((K_norm[2][2] - 1.0).abs() < 1e-10);
/// ```
pub fn normalize_kernel_matrix(kernel_matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = kernel_matrix.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Verify square matrix
    for row in kernel_matrix {
        if row.len() != n {
            return Err(KernelError::ComputationError(
                "Kernel matrix must be square".to_string(),
            ));
        }
    }

    // Extract diagonal elements
    let diagonal: Vec<f64> = (0..n).map(|i| kernel_matrix[i][i]).collect();

    // Check for non-positive diagonal elements
    if diagonal.iter().any(|&d| d <= 0.0) {
        return Err(KernelError::ComputationError(
            "Kernel matrix has non-positive diagonal elements".to_string(),
        ));
    }

    // Compute normalization factors
    let sqrt_diag: Vec<f64> = diagonal.iter().map(|&d| d.sqrt()).collect();

    // Normalize: K_norm[i,j] = K[i,j] / (sqrt(K[i,i]) * sqrt(K[j,j]))
    let mut normalized = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            normalized[i][j] = kernel_matrix[i][j] / (sqrt_diag[i] * sqrt_diag[j]);
        }
    }

    Ok(normalized)
}

/// Center a kernel matrix by removing the mean in feature space.
///
/// Centered kernel: K_c = (I - 1/n * 11^T) K (I - 1/n * 11^T)
///
/// This transformation is required for kernel PCA to ensure the
/// data is centered in feature space.
///
/// # Arguments
/// * `kernel_matrix` - Input kernel matrix (must be square)
///
/// # Returns
/// * Centered kernel matrix
///
/// # Examples
/// ```
/// use tensorlogic_sklears_kernels::kernel_transform::center_kernel_matrix;
///
/// let K = vec![
///     vec![1.0, 0.8, 0.6],
///     vec![0.8, 1.0, 0.7],
///     vec![0.6, 0.7, 1.0],
/// ];
///
/// let K_centered = center_kernel_matrix(&K).unwrap();
///
/// // Row and column means should be approximately zero
/// let row_mean: f64 = K_centered[0].iter().sum::<f64>() / 3.0;
/// assert!(row_mean.abs() < 1e-10);
/// ```
pub fn center_kernel_matrix(kernel_matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = kernel_matrix.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Verify square matrix
    for row in kernel_matrix {
        if row.len() != n {
            return Err(KernelError::ComputationError(
                "Kernel matrix must be square".to_string(),
            ));
        }
    }

    // Compute row means
    let row_means: Vec<f64> = kernel_matrix
        .iter()
        .map(|row| row.iter().sum::<f64>() / n as f64)
        .collect();

    // Compute column means (for symmetric matrices, same as row means, but compute anyway)
    let col_means: Vec<f64> = (0..n)
        .map(|j| kernel_matrix.iter().map(|row| row[j]).sum::<f64>() / n as f64)
        .collect();

    // Compute grand mean
    let grand_mean = row_means.iter().sum::<f64>() / n as f64;

    // Center: K_c[i,j] = K[i,j] - row_mean[i] - col_mean[j] + grand_mean
    let mut centered = vec![vec![0.0; n]; n];
    #[allow(clippy::needless_range_loop)] // Nested loops needed for matrix indexing
    for i in 0..n {
        for j in 0..n {
            centered[i][j] = kernel_matrix[i][j] - row_means[i] - col_means[j] + grand_mean;
        }
    }

    Ok(centered)
}

/// Standardize a kernel matrix (normalize then center).
///
/// This combines normalization and centering in one operation,
/// which is useful for many kernel-based algorithms.
///
/// # Arguments
/// * `kernel_matrix` - Input kernel matrix (must be square)
///
/// # Returns
/// * Standardized kernel matrix
///
/// # Examples
/// ```
/// use tensorlogic_sklears_kernels::kernel_transform::standardize_kernel_matrix;
///
/// let K = vec![
///     vec![4.0, 2.0, 1.0],
///     vec![2.0, 9.0, 3.0],
///     vec![1.0, 3.0, 16.0],
/// ];
///
/// let K_std = standardize_kernel_matrix(&K).unwrap();
/// ```
pub fn standardize_kernel_matrix(kernel_matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let normalized = normalize_kernel_matrix(kernel_matrix)?;
    center_kernel_matrix(&normalized)
}

/// Wrapper that creates a normalized version of any kernel.
///
/// The normalized kernel computes K_norm(x,y) = K(x,y) / sqrt(K(x,x) * K(y,y))
/// ensuring that K_norm(x,x) = 1.0 for all x.
pub struct NormalizedKernel {
    /// Base kernel
    base_kernel: Box<dyn Kernel>,
    /// Cache for diagonal values K(x,x) (thread-safe)
    diagonal_cache: std::sync::Mutex<std::collections::HashMap<u64, f64>>,
}

impl NormalizedKernel {
    /// Create a new normalized kernel wrapper
    ///
    /// # Examples
    /// ```
    /// use tensorlogic_sklears_kernels::{LinearKernel, NormalizedKernel, Kernel};
    ///
    /// let linear = Box::new(LinearKernel::new());
    /// let normalized = NormalizedKernel::new(linear);
    ///
    /// let x = vec![1.0, 2.0, 3.0];
    /// let y = vec![4.0, 5.0, 6.0];
    /// let sim = normalized.compute(&x, &y).unwrap();
    ///
    /// // Self-similarity should be 1.0
    /// let self_sim = normalized.compute(&x, &x).unwrap();
    /// assert!((self_sim - 1.0).abs() < 1e-10);
    /// ```
    pub fn new(base_kernel: Box<dyn Kernel>) -> Self {
        Self {
            base_kernel,
            diagonal_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Hash a vector for caching (simple hash for demonstration)
    fn hash_vector(x: &[f64]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for &val in x {
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Get diagonal value K(x,x) with caching
    fn get_diagonal(&self, x: &[f64]) -> Result<f64> {
        let hash = Self::hash_vector(x);

        // Check cache
        {
            let cache = self.diagonal_cache.lock().unwrap();
            if let Some(&cached) = cache.get(&hash) {
                return Ok(cached);
            }
        }

        // Compute and cache
        let value = self.base_kernel.compute(x, x)?;
        let mut cache = self.diagonal_cache.lock().unwrap();
        cache.insert(hash, value);
        Ok(value)
    }
}

impl Kernel for NormalizedKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let k_xy = self.base_kernel.compute(x, y)?;
        let k_xx = self.get_diagonal(x)?;
        let k_yy = self.get_diagonal(y)?;

        if k_xx <= 0.0 || k_yy <= 0.0 {
            return Err(KernelError::ComputationError(
                "Kernel diagonal elements must be positive for normalization".to_string(),
            ));
        }

        Ok(k_xy / (k_xx * k_yy).sqrt())
    }

    fn name(&self) -> &str {
        "Normalized"
    }
}

#[cfg(test)]
#[allow(non_snake_case, clippy::needless_range_loop)] // Allow K for kernel matrices, range loops for 2D matrix access
mod tests {
    use super::*;
    use crate::{LinearKernel, RbfKernel, RbfKernelConfig};

    #[test]
    fn test_normalize_kernel_matrix_basic() {
        let K = vec![
            vec![4.0, 2.0, 1.0],
            vec![2.0, 9.0, 3.0],
            vec![1.0, 3.0, 16.0],
        ];

        let K_norm = normalize_kernel_matrix(&K).unwrap();

        // Check diagonal is all 1.0
        assert!((K_norm[0][0] - 1.0).abs() < 1e-10);
        assert!((K_norm[1][1] - 1.0).abs() < 1e-10);
        assert!((K_norm[2][2] - 1.0).abs() < 1e-10);

        // Check symmetry preserved
        assert!((K_norm[0][1] - K_norm[1][0]).abs() < 1e-10);
        assert!((K_norm[0][2] - K_norm[2][0]).abs() < 1e-10);
        assert!((K_norm[1][2] - K_norm[2][1]).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_kernel_matrix_correctness() {
        let K = vec![vec![4.0, 2.0], vec![2.0, 9.0]];

        let K_norm = normalize_kernel_matrix(&K).unwrap();

        // K_norm[0][1] = K[0][1] / sqrt(K[0][0] * K[1][1])
        //              = 2.0 / sqrt(4.0 * 9.0)
        //              = 2.0 / 6.0 = 1/3
        assert!((K_norm[0][1] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_kernel_matrix_empty() {
        let K: Vec<Vec<f64>> = Vec::new();
        let K_norm = normalize_kernel_matrix(&K).unwrap();
        assert!(K_norm.is_empty());
    }

    #[test]
    fn test_normalize_kernel_matrix_non_square() {
        let K = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]];

        let result = normalize_kernel_matrix(&K);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_kernel_matrix_negative_diagonal() {
        let K = vec![vec![-1.0, 2.0], vec![2.0, 4.0]];

        let result = normalize_kernel_matrix(&K);
        assert!(result.is_err());
    }

    #[test]
    fn test_center_kernel_matrix_basic() {
        let K = vec![
            vec![1.0, 0.8, 0.6],
            vec![0.8, 1.0, 0.7],
            vec![0.6, 0.7, 1.0],
        ];

        let K_centered = center_kernel_matrix(&K).unwrap();

        // Check row sums are approximately zero
        for row in &K_centered {
            let row_sum: f64 = row.iter().sum();
            assert!(row_sum.abs() < 1e-10);
        }

        // Check column sums are approximately zero
        for j in 0..3 {
            let col_sum: f64 = (0..3).map(|i| K_centered[i][j]).sum();
            assert!(col_sum.abs() < 1e-10);
        }

        // Check grand sum is approximately zero
        let grand_sum: f64 = K_centered.iter().map(|row| row.iter().sum::<f64>()).sum();
        assert!(grand_sum.abs() < 1e-9);
    }

    #[test]
    fn test_center_kernel_matrix_empty() {
        let K: Vec<Vec<f64>> = Vec::new();
        let K_centered = center_kernel_matrix(&K).unwrap();
        assert!(K_centered.is_empty());
    }

    #[test]
    fn test_center_kernel_matrix_non_square() {
        let K = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]];

        let result = center_kernel_matrix(&K);
        assert!(result.is_err());
    }

    #[test]
    fn test_standardize_kernel_matrix() {
        let K = vec![
            vec![4.0, 2.0, 1.0],
            vec![2.0, 9.0, 3.0],
            vec![1.0, 3.0, 16.0],
        ];

        let K_std = standardize_kernel_matrix(&K).unwrap();

        // After standardization, row/column sums should be close to zero
        for row in &K_std {
            let row_sum: f64 = row.iter().sum();
            assert!(row_sum.abs() < 1e-9);
        }
    }

    #[test]
    fn test_normalized_kernel_wrapper() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let normalized = NormalizedKernel::new(linear);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        // Self-similarity should be 1.0
        let self_sim = normalized.compute(&x, &x).unwrap();
        assert!((self_sim - 1.0).abs() < 1e-10);

        // Compute normalized similarity
        let sim = normalized.compute(&x, &y).unwrap();
        assert!((-1.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_normalized_kernel_rbf() {
        let rbf = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;
        let normalized = NormalizedKernel::new(rbf);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];

        // Self-similarity should be 1.0
        let self_sim_x = normalized.compute(&x, &x).unwrap();
        let self_sim_y = normalized.compute(&y, &y).unwrap();
        assert!((self_sim_x - 1.0).abs() < 1e-10);
        assert!((self_sim_y - 1.0).abs() < 1e-10);

        // Cross-similarity should be in (0, 1)
        let sim = normalized.compute(&x, &y).unwrap();
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_normalized_kernel_symmetry() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let normalized = NormalizedKernel::new(linear);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let sim_xy = normalized.compute(&x, &y).unwrap();
        let sim_yx = normalized.compute(&y, &x).unwrap();

        assert!((sim_xy - sim_yx).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_kernel_caching() {
        let linear = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let normalized = NormalizedKernel::new(linear);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        // Multiple calls should use cache
        let sim1 = normalized.compute(&x, &y).unwrap();
        let sim2 = normalized.compute(&x, &y).unwrap();
        let sim3 = normalized.compute(&x, &y).unwrap();

        assert!((sim1 - sim2).abs() < 1e-10);
        assert!((sim2 - sim3).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_then_center_vs_standardize() {
        let K = vec![
            vec![4.0, 2.0, 1.0],
            vec![2.0, 9.0, 3.0],
            vec![1.0, 3.0, 16.0],
        ];

        // Method 1: Normalize then center
        let K_norm = normalize_kernel_matrix(&K).unwrap();
        let K_norm_cent = center_kernel_matrix(&K_norm).unwrap();

        // Method 2: Use standardize
        let K_std = standardize_kernel_matrix(&K).unwrap();

        // Should be identical
        for i in 0..3 {
            for j in 0..3 {
                assert!((K_norm_cent[i][j] - K_std[i][j]).abs() < 1e-10);
            }
        }
    }
}
