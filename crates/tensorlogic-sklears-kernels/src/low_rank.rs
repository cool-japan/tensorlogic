//! Low-rank kernel matrix approximations
//!
//! This module provides methods for approximating large kernel matrices
//! using low-rank representations, which significantly reduce memory
//! and computational costs.
//!
//! ## Nyström Method
//!
//! The Nyström method approximates a kernel matrix K (n×n) using a subset
//! of m landmark points, producing an approximation with O(nm) instead of O(n²) complexity.
//!
//! ## References
//!
//! - Williams & Seeger (2001): "Using the Nyström Method to Speed Up Kernel Machines"
//! - Kumar et al. (2012): "Sampling Methods for the Nyström Method"

use crate::error::{KernelError, Result};
use crate::types::Kernel;
use serde::{Deserialize, Serialize};

/// Configuration for Nyström approximation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NystromConfig {
    /// Number of landmark points to use
    pub num_landmarks: usize,
    /// Sampling method for selecting landmarks
    pub sampling: SamplingMethod,
    /// Regularization parameter for numerical stability
    pub regularization: f64,
}

/// Sampling method for selecting landmark points
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingMethod {
    /// Uniform random sampling
    Uniform,
    /// Sample first n points (deterministic)
    First,
    /// K-means++ style sampling (diverse landmarks)
    KMeansPlusPlus,
}

impl NystromConfig {
    /// Create a new configuration
    pub fn new(num_landmarks: usize) -> Result<Self> {
        if num_landmarks == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "num_landmarks".to_string(),
                value: num_landmarks.to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            num_landmarks,
            sampling: SamplingMethod::Uniform,
            regularization: 1e-6,
        })
    }

    /// Set sampling method
    pub fn with_sampling(mut self, sampling: SamplingMethod) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set regularization parameter
    pub fn with_regularization(mut self, regularization: f64) -> Result<Self> {
        if regularization < 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "regularization".to_string(),
                value: regularization.to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        self.regularization = regularization;
        Ok(self)
    }
}

/// Low-rank Nyström approximation of a kernel matrix
///
/// Stores the approximation K ≈ C * W_inv * C^T where:
/// - C is the n×m matrix of kernel values between all points and landmarks
/// - W_inv is the m×m inverse of the kernel matrix on landmarks
pub struct NystromApproximation {
    /// Kernel values between all points and landmarks (n×m)
    c_matrix: Vec<Vec<f64>>,
    /// Pseudo-inverse of landmark kernel matrix (m×m)
    w_inv: Vec<Vec<f64>>,
    /// Indices of landmark points
    landmark_indices: Vec<usize>,
}

impl NystromApproximation {
    /// Create a low-rank approximation using the Nyström method
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset (n samples)
    /// * `kernel` - The kernel function
    /// * `config` - Configuration for the approximation
    ///
    /// # Returns
    ///
    /// A Nyström approximation that can efficiently compute approximate kernel values
    pub fn fit(data: &[Vec<f64>], kernel: &dyn Kernel, config: NystromConfig) -> Result<Self> {
        let n = data.len();

        if config.num_landmarks > n {
            return Err(KernelError::InvalidParameter {
                parameter: "num_landmarks".to_string(),
                value: config.num_landmarks.to_string(),
                reason: format!("cannot exceed dataset size ({})", n),
            });
        }

        // Select landmark points
        let landmark_indices = Self::select_landmarks(data, kernel, &config)?;

        // Compute C matrix (n×m): kernel values between all points and landmarks
        let mut c_matrix = vec![vec![0.0; config.num_landmarks]; n];
        for i in 0..n {
            for (j, &landmark_idx) in landmark_indices.iter().enumerate() {
                c_matrix[i][j] = kernel.compute(&data[i], &data[landmark_idx])?;
            }
        }

        // Compute W matrix (m×m): kernel matrix on landmarks
        let mut w_matrix = vec![vec![0.0; config.num_landmarks]; config.num_landmarks];
        for i in 0..config.num_landmarks {
            for j in 0..config.num_landmarks {
                w_matrix[i][j] =
                    kernel.compute(&data[landmark_indices[i]], &data[landmark_indices[j]])?;
            }
        }

        // Add regularization to diagonal
        #[allow(clippy::needless_range_loop)]
        for i in 0..config.num_landmarks {
            w_matrix[i][i] += config.regularization;
        }

        // Compute pseudo-inverse of W
        let w_inv = Self::pseudo_inverse(&w_matrix)?;

        Ok(Self {
            c_matrix,
            w_inv,
            landmark_indices,
        })
    }

    /// Select landmark points based on sampling method
    fn select_landmarks(
        data: &[Vec<f64>],
        kernel: &dyn Kernel,
        config: &NystromConfig,
    ) -> Result<Vec<usize>> {
        match config.sampling {
            SamplingMethod::First => {
                // Simply take first m points
                Ok((0..config.num_landmarks).collect())
            }
            SamplingMethod::Uniform => {
                // Uniform random sampling (deterministic for reproducibility)
                let step = data.len() / config.num_landmarks;
                Ok((0..config.num_landmarks).map(|i| i * step).collect())
            }
            SamplingMethod::KMeansPlusPlus => {
                // K-means++ style sampling for diversity
                Self::kmeans_plusplus_sampling(data, kernel, config.num_landmarks)
            }
        }
    }

    /// K-means++ style sampling for diverse landmark selection
    fn kmeans_plusplus_sampling(
        data: &[Vec<f64>],
        kernel: &dyn Kernel,
        num_landmarks: usize,
    ) -> Result<Vec<usize>> {
        let n = data.len();
        let mut landmarks = Vec::with_capacity(num_landmarks);
        let mut min_distances = vec![f64::INFINITY; n];

        // Select first landmark (first point for determinism)
        landmarks.push(0);

        // Select remaining landmarks based on distance from existing landmarks
        for _ in 1..num_landmarks {
            // Update minimum distances to nearest landmark
            let last_landmark = *landmarks.last().unwrap();
            for i in 0..n {
                if landmarks.contains(&i) {
                    continue;
                }

                // Compute kernel-based distance (1 - kernel_similarity)
                let similarity = kernel.compute(&data[i], &data[last_landmark])?;
                let distance = 1.0 - similarity;
                min_distances[i] = min_distances[i].min(distance);
            }

            // Select point with maximum distance to nearest landmark
            let mut max_dist = 0.0;
            let mut best_idx = 0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                if !landmarks.contains(&i) && min_distances[i] > max_dist {
                    max_dist = min_distances[i];
                    best_idx = i;
                }
            }

            landmarks.push(best_idx);
        }

        Ok(landmarks)
    }

    /// Compute pseudo-inverse using simple iterative method
    ///
    /// For production use, this should use a proper linear algebra library.
    /// This is a simplified implementation for demonstration.
    fn pseudo_inverse(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = matrix.len();
        if n == 0 {
            return Err(KernelError::ComputationError(
                "Cannot invert empty matrix".to_string(),
            ));
        }

        // For small matrices, use Gauss-Jordan elimination
        let mut augmented = vec![vec![0.0; 2 * n]; n];

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[i][j] = matrix[i][j];
            }
            augmented[i][n + i] = 1.0;
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[k][i].abs() > augmented[max_row][i].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                augmented.swap(i, max_row);
            }

            // Check for singular matrix
            if augmented[i][i].abs() < 1e-10 {
                return Err(KernelError::ComputationError(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Scale pivot row
            let pivot = augmented[i][i];
            for j in 0..(2 * n) {
                augmented[i][j] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[k][i];
                    for j in 0..(2 * n) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }

        // Extract inverse from right half
        let mut inverse = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inverse[i][j] = augmented[i][n + j];
            }
        }

        Ok(inverse)
    }

    /// Approximate kernel value between two points
    ///
    /// Uses the approximation: K(x_i, x_j) ≈ C_i * W_inv * C_j^T
    pub fn approximate(&self, i: usize, j: usize) -> Result<f64> {
        if i >= self.c_matrix.len() || j >= self.c_matrix.len() {
            return Err(KernelError::ComputationError(format!(
                "Indices out of bounds: i={}, j={}",
                i, j
            )));
        }

        // Compute C_i * W_inv * C_j^T
        let m = self.w_inv.len();
        let mut result = 0.0;

        for k in 0..m {
            for idx in 0..m {
                result += self.c_matrix[i][k] * self.w_inv[k][idx] * self.c_matrix[j][idx];
            }
        }

        Ok(result)
    }

    /// Get the full approximate kernel matrix
    pub fn get_approximate_matrix(&self) -> Result<Vec<Vec<f64>>> {
        let n = self.c_matrix.len();
        let mut matrix = vec![vec![0.0; n]; n];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = self.approximate(i, j)?;
            }
        }

        Ok(matrix)
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.c_matrix.len()
    }

    /// Get the number of landmarks
    pub fn num_landmarks(&self) -> usize {
        self.landmark_indices.len()
    }

    /// Get landmark indices
    pub fn landmark_indices(&self) -> &[usize] {
        &self.landmark_indices
    }

    /// Compute approximation error (Frobenius norm) compared to exact kernel matrix
    pub fn approximation_error(&self, exact_matrix: &[Vec<f64>]) -> Result<f64> {
        let approx_matrix = self.get_approximate_matrix()?;
        let n = exact_matrix.len();

        if approx_matrix.len() != n || approx_matrix[0].len() != n {
            return Err(KernelError::DimensionMismatch {
                expected: vec![n, n],
                got: vec![approx_matrix.len(), approx_matrix[0].len()],
                context: "approximation error computation".to_string(),
            });
        }

        let mut error = 0.0;
        for i in 0..n {
            for j in 0..n {
                let diff = exact_matrix[i][j] - approx_matrix[i][j];
                error += diff * diff;
            }
        }

        Ok(error.sqrt())
    }

    /// Get compression ratio (original size / approximation size)
    pub fn compression_ratio(&self) -> f64 {
        let n = self.num_samples() as f64;
        let m = self.num_landmarks() as f64;
        (n * n) / (n * m + m * m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearKernel;

    fn generate_test_data(n: usize, dim: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..dim).map(|j| ((i * dim + j) as f64).sin()).collect())
            .collect()
    }

    #[test]
    fn test_nystrom_config() {
        let config = NystromConfig::new(10).unwrap();
        assert_eq!(config.num_landmarks, 10);
        assert_eq!(config.sampling, SamplingMethod::Uniform);
    }

    #[test]
    fn test_nystrom_config_invalid() {
        let result = NystromConfig::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_nystrom_approximation_basic() {
        let data = generate_test_data(50, 10);
        let kernel = LinearKernel::new();
        let config = NystromConfig::new(10).unwrap();

        let approx = NystromApproximation::fit(&data, &kernel, config).unwrap();

        assert_eq!(approx.num_samples(), 50);
        assert_eq!(approx.num_landmarks(), 10);
    }

    #[test]
    fn test_nystrom_approximation_value() {
        let data = generate_test_data(20, 5);
        let kernel = LinearKernel::new();
        let config = NystromConfig::new(5).unwrap();

        let approx = NystromApproximation::fit(&data, &kernel, config).unwrap();

        // Approximate kernel value should be reasonable
        let value = approx.approximate(0, 1).unwrap();
        assert!(value.is_finite());
    }

    #[test]
    fn test_nystrom_sampling_methods() {
        let data = generate_test_data(30, 5);
        let kernel = LinearKernel::new();

        // First sampling
        let config1 = NystromConfig::new(10)
            .unwrap()
            .with_sampling(SamplingMethod::First);
        let approx1 = NystromApproximation::fit(&data, &kernel, config1).unwrap();
        assert_eq!(approx1.landmark_indices()[0], 0);

        // Uniform sampling
        let config2 = NystromConfig::new(10)
            .unwrap()
            .with_sampling(SamplingMethod::Uniform);
        let approx2 = NystromApproximation::fit(&data, &kernel, config2).unwrap();
        assert_eq!(approx2.num_landmarks(), 10);

        // K-means++ sampling
        let config3 = NystromConfig::new(10)
            .unwrap()
            .with_sampling(SamplingMethod::KMeansPlusPlus);
        let approx3 = NystromApproximation::fit(&data, &kernel, config3).unwrap();
        assert_eq!(approx3.num_landmarks(), 10);
    }

    #[test]
    fn test_nystrom_compression_ratio() {
        let data = generate_test_data(100, 5);
        let kernel = LinearKernel::new();
        let config = NystromConfig::new(20).unwrap();

        let approx = NystromApproximation::fit(&data, &kernel, config).unwrap();

        let ratio = approx.compression_ratio();
        // Should have significant compression
        assert!(ratio > 3.0);
    }

    #[test]
    fn test_nystrom_approximation_error() {
        let data = generate_test_data(30, 5);
        let kernel = LinearKernel::new();

        // Compute exact kernel matrix
        let exact = kernel.compute_matrix(&data).unwrap();

        // Compute approximation with many landmarks (should be accurate)
        let config = NystromConfig::new(20).unwrap();
        let approx = NystromApproximation::fit(&data, &kernel, config).unwrap();

        let error = approx.approximation_error(&exact).unwrap();
        // Error should be relatively small with many landmarks
        assert!(error < 10.0);
    }

    #[test]
    fn test_nystrom_too_many_landmarks() {
        let data = generate_test_data(10, 5);
        let kernel = LinearKernel::new();
        let config = NystromConfig::new(20).unwrap(); // More landmarks than data points

        let result = NystromApproximation::fit(&data, &kernel, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_nystrom_regularization() {
        let data = generate_test_data(20, 5);
        let kernel = LinearKernel::new();
        let config = NystromConfig::new(5)
            .unwrap()
            .with_regularization(1e-4)
            .unwrap();

        let approx = NystromApproximation::fit(&data, &kernel, config).unwrap();
        assert!(approx.approximate(0, 1).is_ok());
    }

    #[test]
    fn test_nystrom_invalid_regularization() {
        let result = NystromConfig::new(10).unwrap().with_regularization(-0.1);
        assert!(result.is_err());
    }
}
