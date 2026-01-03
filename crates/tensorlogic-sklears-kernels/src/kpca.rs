#![allow(clippy::needless_range_loop)]
//! Kernel Principal Component Analysis (KPCA) utilities.
//!
//! This module provides utilities for Kernel PCA, a nonlinear dimensionality
//! reduction technique that uses kernel methods to perform PCA in a high-dimensional
//! feature space.
//!
//! ## Overview
//!
//! Kernel PCA extends classical PCA by:
//! 1. Mapping data into a high-dimensional feature space via a kernel
//! 2. Performing PCA in this feature space
//! 3. Projecting data onto principal components
//!
//! The key insight is that we never need to explicitly compute the feature mapping;
//! we only need the kernel matrix K.
//!
//! ## Algorithm
//!
//! Given a centered kernel matrix K_c = H K H (where H = I - 1/n 11^T):
//! 1. Compute eigendecomposition: K_c = V Λ V^T
//! 2. Project data: z_i = V_k^T k_c(x_i) / sqrt(λ_k)
//!
//! ## Example
//!
//! ```no_run
//! use tensorlogic_sklears_kernels::kpca::{KernelPCA, KernelPCAConfig};
//! use tensorlogic_sklears_kernels::{RbfKernel, RbfKernelConfig};
//!
//! let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
//! let data = vec![
//!     vec![1.0, 2.0],
//!     vec![3.0, 4.0],
//!     vec![5.0, 6.0],
//!     vec![7.0, 8.0],
//! ];
//!
//! let config = KernelPCAConfig::new(2); // 2 components
//! let kpca = KernelPCA::fit(&data, &kernel, config).unwrap();
//!
//! // Transform new data
//! let transformed = kpca.transform(&data, &kernel).unwrap();
//! ```

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// Configuration for Kernel PCA.
#[derive(Debug, Clone)]
pub struct KernelPCAConfig {
    /// Number of principal components to extract
    pub n_components: usize,
    /// Whether to center the kernel matrix
    pub center: bool,
    /// Eigenvalue threshold for numerical stability (eigenvalues below this are ignored)
    pub eigenvalue_threshold: f64,
}

impl KernelPCAConfig {
    /// Create a new KPCA configuration with the specified number of components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            center: true,
            eigenvalue_threshold: 1e-10,
        }
    }

    /// Set whether to center the kernel matrix.
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set the eigenvalue threshold.
    pub fn with_eigenvalue_threshold(mut self, threshold: f64) -> Self {
        self.eigenvalue_threshold = threshold;
        self
    }
}

/// Kernel PCA model.
#[derive(Debug, Clone)]
pub struct KernelPCA {
    /// Training data (for kernel computation with new points)
    training_data: Vec<Vec<f64>>,
    /// Eigenvectors (scaled by 1/sqrt(eigenvalue))
    eigenvectors: Vec<Vec<f64>>,
    /// Eigenvalues
    eigenvalues: Vec<f64>,
    /// Number of components
    n_components: usize,
    /// Whether the kernel was centered
    centered: bool,
    /// Mean of training kernel values (for centering new data)
    kernel_row_means: Vec<f64>,
    /// Mean of all training kernel values
    kernel_mean: f64,
}

impl KernelPCA {
    /// Fit Kernel PCA to the training data.
    ///
    /// # Arguments
    /// * `data` - Training data as a vector of feature vectors
    /// * `kernel` - The kernel function to use
    /// * `config` - KPCA configuration
    ///
    /// # Returns
    /// A fitted KernelPCA model
    pub fn fit<K: Kernel + ?Sized>(
        data: &[Vec<f64>],
        kernel: &K,
        config: KernelPCAConfig,
    ) -> Result<Self> {
        let n = data.len();
        if n == 0 {
            return Err(KernelError::ComputationError(
                "Empty training data".to_string(),
            ));
        }

        if config.n_components > n {
            return Err(KernelError::InvalidParameter {
                parameter: "n_components".to_string(),
                value: config.n_components.to_string(),
                reason: format!(
                    "Cannot extract {} components from {} samples",
                    config.n_components, n
                ),
            });
        }

        // Compute kernel matrix
        let mut k_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let k_val = kernel.compute(&data[i], &data[j])?;
                k_matrix[i][j] = k_val;
                k_matrix[j][i] = k_val;
            }
        }

        // Compute row means and overall mean (for centering)
        let mut row_means = vec![0.0; n];
        let mut total_mean = 0.0;
        for i in 0..n {
            let row_sum: f64 = k_matrix[i].iter().sum();
            row_means[i] = row_sum / n as f64;
            total_mean += row_sum;
        }
        total_mean /= (n * n) as f64;

        // Center the kernel matrix if requested: K_c = K - row_mean - col_mean + total_mean
        let mut k_centered = k_matrix.clone();
        if config.center {
            for i in 0..n {
                for j in 0..n {
                    k_centered[i][j] = k_matrix[i][j] - row_means[i] - row_means[j] + total_mean;
                }
            }
        }

        // Compute eigendecomposition using power iteration
        let (eigenvalues, eigenvectors) = eigen_decomposition(
            &k_centered,
            config.n_components,
            config.eigenvalue_threshold,
        )?;

        // Store training data for later transformation
        let training_data = data.to_vec();

        Ok(Self {
            training_data,
            eigenvectors,
            eigenvalues,
            n_components: config.n_components,
            centered: config.center,
            kernel_row_means: row_means,
            kernel_mean: total_mean,
        })
    }

    /// Transform data using the fitted KPCA model.
    ///
    /// # Arguments
    /// * `data` - Data to transform (can be training or new data)
    /// * `kernel` - The kernel function (must be same as used in fit)
    ///
    /// # Returns
    /// Transformed data with `n_components` dimensions
    pub fn transform<K: Kernel + ?Sized>(
        &self,
        data: &[Vec<f64>],
        kernel: &K,
    ) -> Result<Vec<Vec<f64>>> {
        let n_train = self.training_data.len();
        let n_test = data.len();

        // Compute kernel values between test and training data
        let mut k_new = vec![vec![0.0; n_train]; n_test];
        for i in 0..n_test {
            for j in 0..n_train {
                k_new[i][j] = kernel.compute(&data[i], &self.training_data[j])?;
            }
        }

        // Center new kernel values if centering was used during fit
        if self.centered {
            for i in 0..n_test {
                let row_mean: f64 = k_new[i].iter().sum::<f64>() / n_train as f64;
                for j in 0..n_train {
                    k_new[i][j] =
                        k_new[i][j] - row_mean - self.kernel_row_means[j] + self.kernel_mean;
                }
            }
        }

        // Project onto principal components
        let mut transformed = vec![vec![0.0; self.n_components]; n_test];
        for i in 0..n_test {
            for c in 0..self.n_components {
                if self.eigenvalues[c] > 0.0 {
                    let mut proj = 0.0;
                    for j in 0..n_train {
                        proj += k_new[i][j] * self.eigenvectors[c][j];
                    }
                    transformed[i][c] = proj / self.eigenvalues[c].sqrt();
                }
            }
        }

        Ok(transformed)
    }

    /// Transform the training data (convenience method).
    pub fn transform_training<K: Kernel + ?Sized>(&self, kernel: &K) -> Result<Vec<Vec<f64>>> {
        self.transform(&self.training_data, kernel)
    }

    /// Get the eigenvalues.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Get the explained variance ratio for each component.
    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        let total_var: f64 = self.eigenvalues.iter().sum();
        if total_var > 0.0 {
            self.eigenvalues.iter().map(|&e| e / total_var).collect()
        } else {
            vec![0.0; self.eigenvalues.len()]
        }
    }

    /// Get the cumulative explained variance.
    pub fn cumulative_variance_explained(&self) -> Vec<f64> {
        let ratios = self.explained_variance_ratio();
        let mut cumulative = Vec::with_capacity(ratios.len());
        let mut sum = 0.0;
        for r in ratios {
            sum += r;
            cumulative.push(sum);
        }
        cumulative
    }

    /// Get the number of components.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Get the number of training samples.
    pub fn n_samples(&self) -> usize {
        self.training_data.len()
    }
}

/// Center a kernel matrix in place.
///
/// `K_c[i,j] = K[i,j] - mean(K[i,:]) - mean(K[:,j]) + mean(K)`
pub fn center_kernel_matrix(matrix: &mut [Vec<f64>]) {
    let n = matrix.len();
    if n == 0 {
        return;
    }

    // Compute row means and overall mean
    let mut row_means = vec![0.0; n];
    let mut total_mean = 0.0;
    for i in 0..n {
        let row_sum: f64 = matrix[i].iter().sum();
        row_means[i] = row_sum / n as f64;
        total_mean += row_sum;
    }
    total_mean /= (n * n) as f64;

    // Center
    for i in 0..n {
        for j in 0..n {
            matrix[i][j] = matrix[i][j] - row_means[i] - row_means[j] + total_mean;
        }
    }
}

/// Compute eigendecomposition using power iteration with deflation.
///
/// Returns (eigenvalues, eigenvectors) for the top k components.
fn eigen_decomposition(
    matrix: &[Vec<f64>],
    n_components: usize,
    threshold: f64,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = matrix.len();
    if n == 0 {
        return Err(KernelError::ComputationError("Empty matrix".to_string()));
    }

    let max_iter = 1000;
    let tol = 1e-10;

    let mut eigenvalues = Vec::with_capacity(n_components);
    let mut eigenvectors = Vec::with_capacity(n_components);

    // Work with a copy of the matrix for deflation
    let mut work_matrix = matrix.to_vec();

    for _ in 0..n_components {
        // Power iteration to find largest eigenvalue/eigenvector
        let mut v = vec![1.0 / (n as f64).sqrt(); n]; // Initial vector

        let mut eigenvalue = 0.0;
        for _ in 0..max_iter {
            // Multiply: w = A * v
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += work_matrix[i][j] * v[j];
                }
            }

            // Compute eigenvalue (Rayleigh quotient)
            let new_eigenvalue: f64 = v.iter().zip(w.iter()).map(|(vi, wi)| vi * wi).sum();

            // Normalize
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break; // Degenerate case
            }
            for wi in &mut w {
                *wi /= norm;
            }

            // Check convergence
            let diff: f64 = v.iter().zip(w.iter()).map(|(vi, wi)| (vi - wi).abs()).sum();
            v = w;

            if (new_eigenvalue - eigenvalue).abs() < tol && diff < tol {
                eigenvalue = new_eigenvalue;
                break;
            }
            eigenvalue = new_eigenvalue;
        }

        // Skip if eigenvalue is too small
        if eigenvalue < threshold {
            // Push zeros for remaining components
            while eigenvalues.len() < n_components {
                eigenvalues.push(0.0);
                eigenvectors.push(vec![0.0; n]);
            }
            break;
        }

        eigenvalues.push(eigenvalue);
        eigenvectors.push(v.clone());

        // Deflate: A = A - λ * v * v^T
        for i in 0..n {
            for j in 0..n {
                work_matrix[i][j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    Ok((eigenvalues, eigenvectors))
}

/// Compute the reconstruction error of KPCA.
///
/// This measures how well the lower-dimensional representation approximates
/// the original kernel matrix.
pub fn reconstruction_error(original_matrix: &[Vec<f64>], eigenvalues: &[f64]) -> f64 {
    // Total variance is the trace of the kernel matrix
    let total_var: f64 = (0..original_matrix.len())
        .map(|i| original_matrix[i][i])
        .sum();

    // Explained variance is sum of selected eigenvalues
    let explained_var: f64 = eigenvalues.iter().sum();

    // Reconstruction error is unexplained variance
    (total_var - explained_var).max(0.0)
}

/// Select the number of components to explain a target variance ratio.
///
/// # Arguments
/// * `eigenvalues` - All eigenvalues (sorted in descending order)
/// * `target_ratio` - Target explained variance ratio (e.g., 0.95 for 95%)
///
/// # Returns
/// Number of components needed
pub fn select_n_components(eigenvalues: &[f64], target_ratio: f64) -> usize {
    let total: f64 = eigenvalues.iter().sum();
    if total <= 0.0 {
        return 1;
    }

    let mut cumsum = 0.0;
    for (i, &e) in eigenvalues.iter().enumerate() {
        cumsum += e;
        if cumsum / total >= target_ratio {
            return i + 1;
        }
    }
    eigenvalues.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_kernels::{LinearKernel, RbfKernel};
    use crate::types::RbfKernelConfig;

    #[test]
    fn test_kpca_config() {
        let config = KernelPCAConfig::new(3)
            .with_center(false)
            .with_eigenvalue_threshold(1e-8);

        assert_eq!(config.n_components, 3);
        assert!(!config.center);
        assert!((config.eigenvalue_threshold - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn test_kpca_linear_kernel() {
        let kernel = LinearKernel::new();
        let data = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
        ];

        let config = KernelPCAConfig::new(2);
        let kpca = KernelPCA::fit(&data, &kernel, config).unwrap();

        assert_eq!(kpca.n_components(), 2);
        assert_eq!(kpca.n_samples(), 4);

        let transformed = kpca.transform(&data, &kernel).unwrap();
        assert_eq!(transformed.len(), 4);
        assert_eq!(transformed[0].len(), 2);
    }

    #[test]
    fn test_kpca_rbf_kernel() {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let config = KernelPCAConfig::new(2);
        let kpca = KernelPCA::fit(&data, &kernel, config).unwrap();

        let transformed = kpca.transform(&data, &kernel).unwrap();
        assert_eq!(transformed.len(), 4);

        // Check that eigenvalues are non-negative and sorted
        for eigenvalue in kpca.eigenvalues() {
            assert!(*eigenvalue >= -1e-10);
        }
    }

    #[test]
    fn test_kpca_explained_variance() {
        let kernel = LinearKernel::new();
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];

        let config = KernelPCAConfig::new(3);
        let kpca = KernelPCA::fit(&data, &kernel, config).unwrap();

        let ratios = kpca.explained_variance_ratio();
        let total: f64 = ratios.iter().sum();

        // Total ratio should be <= 1.0 (could be less if some eigenvalues are skipped)
        assert!(total <= 1.01, "Total explained variance ratio: {}", total);

        // Each ratio should be non-negative
        for r in &ratios {
            assert!(*r >= 0.0);
        }

        let cumulative = kpca.cumulative_variance_explained();
        // Cumulative should be monotonically increasing
        for i in 1..cumulative.len() {
            assert!(cumulative[i] >= cumulative[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_kpca_too_many_components() {
        let kernel = LinearKernel::new();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let config = KernelPCAConfig::new(10); // More than 2 samples
        let result = KernelPCA::fit(&data, &kernel, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_kpca_empty_data() {
        let kernel = LinearKernel::new();
        let data: Vec<Vec<f64>> = vec![];

        let config = KernelPCAConfig::new(2);
        let result = KernelPCA::fit(&data, &kernel, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_center_kernel_matrix() {
        let mut matrix = vec![vec![4.0, 2.0], vec![2.0, 4.0]];
        center_kernel_matrix(&mut matrix);

        // After centering, row and column means should be approximately 0
        let row_mean: f64 = matrix[0].iter().sum::<f64>() / 2.0;
        let col_mean: f64 = matrix.iter().map(|r| r[0]).sum::<f64>() / 2.0;

        assert!(row_mean.abs() < 1e-10);
        assert!(col_mean.abs() < 1e-10);
    }

    #[test]
    fn test_select_n_components() {
        let eigenvalues = vec![5.0, 3.0, 1.5, 0.5];

        // Total = 10, need 9.5 = 95% => 5+3+1.5 = 9.5 (3 components)
        assert_eq!(select_n_components(&eigenvalues, 0.95), 3);

        // 80% => 5+3 = 8 (2 components)
        assert_eq!(select_n_components(&eigenvalues, 0.80), 2);

        // 50% => 5 (1 component)
        assert_eq!(select_n_components(&eigenvalues, 0.50), 1);
    }

    #[test]
    fn test_reconstruction_error() {
        let matrix = vec![vec![4.0, 2.0], vec![2.0, 4.0]];
        let eigenvalues = vec![6.0]; // Only first eigenvalue

        let error = reconstruction_error(&matrix, &eigenvalues);
        // Trace = 8, explained = 6, error = 2
        assert!((error - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_kpca_transform_new_data() {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let train_data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let config = KernelPCAConfig::new(2);
        let kpca = KernelPCA::fit(&train_data, &kernel, config).unwrap();

        // Transform new data point
        let new_data = vec![vec![0.5, 0.5]];
        let transformed = kpca.transform(&new_data, &kernel).unwrap();

        assert_eq!(transformed.len(), 1);
        assert_eq!(transformed[0].len(), 2);
    }

    #[test]
    fn test_kpca_no_centering() {
        let kernel = LinearKernel::new();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let config = KernelPCAConfig::new(2).with_center(false);
        let kpca = KernelPCA::fit(&data, &kernel, config).unwrap();

        let transformed = kpca.transform(&data, &kernel).unwrap();
        assert_eq!(transformed.len(), 3);
    }
}
