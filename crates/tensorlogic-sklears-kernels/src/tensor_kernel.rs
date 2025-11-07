//! Tensor-based kernels operating on feature vectors.
//!
//! These kernels work directly on tensor/vector representations,
//! complementing the logic-derived kernels.

use crate::error::{KernelError, Result};
use crate::types::{Kernel, RbfKernelConfig};

/// Linear kernel: K(x, y) = x · y
///
/// The simplest kernel, equivalent to inner product in feature space.
pub struct LinearKernel;

impl LinearKernel {
    /// Create a new linear kernel
    pub fn new() -> Self {
        Self
    }

    /// Compute dot product
    fn dot_product(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
}

impl Default for LinearKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel for LinearKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "linear kernel".to_string(),
            });
        }

        Ok(Self::dot_product(x, y))
    }

    fn name(&self) -> &str {
        "Linear"
    }
}

/// Polynomial kernel: K(x, y) = (x · y + c)^d
///
/// Captures polynomial relationships up to degree d.
pub struct PolynomialKernel {
    /// Polynomial degree
    degree: u32,
    /// Constant term
    constant: f64,
}

impl PolynomialKernel {
    /// Create a new polynomial kernel
    pub fn new(degree: u32, constant: f64) -> Result<Self> {
        if degree == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "degree".to_string(),
                value: degree.to_string(),
                reason: "degree must be >= 1".to_string(),
            });
        }

        Ok(Self { degree, constant })
    }

    /// Compute dot product
    fn dot_product(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
}

impl Kernel for PolynomialKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "polynomial kernel".to_string(),
            });
        }

        let dot = Self::dot_product(x, y);
        let result = (dot + self.constant).powi(self.degree as i32);
        Ok(result)
    }

    fn name(&self) -> &str {
        "Polynomial"
    }
}

/// RBF (Radial Basis Function) kernel: K(x, y) = exp(-gamma * ||x - y||^2)
///
/// Also known as Gaussian kernel. Maps to infinite-dimensional space.
pub struct RbfKernel {
    /// Configuration
    config: RbfKernelConfig,
}

impl RbfKernel {
    /// Create a new RBF kernel
    pub fn new(config: RbfKernelConfig) -> Result<Self> {
        if config.gamma <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "gamma".to_string(),
                value: config.gamma.to_string(),
                reason: "gamma must be positive".to_string(),
            });
        }

        Ok(Self { config })
    }

    /// Compute squared Euclidean distance
    fn squared_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
    }
}

impl Kernel for RbfKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "RBF kernel".to_string(),
            });
        }

        let sq_dist = Self::squared_distance(x, y);
        let result = (-self.config.gamma * sq_dist).exp();
        Ok(result)
    }

    fn name(&self) -> &str {
        "RBF"
    }
}

/// Cosine similarity kernel: K(x, y) = (x · y) / (||x|| * ||y||)
///
/// Measures angle between vectors, normalized to [-1, 1].
pub struct CosineKernel;

impl CosineKernel {
    /// Create a new cosine kernel
    pub fn new() -> Self {
        Self
    }

    /// Compute dot product
    fn dot_product(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    /// Compute L2 norm
    fn norm(x: &[f64]) -> f64 {
        x.iter().map(|a| a * a).sum::<f64>().sqrt()
    }
}

impl Default for CosineKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel for CosineKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "cosine kernel".to_string(),
            });
        }

        let dot = Self::dot_product(x, y);
        let norm_x = Self::norm(x);
        let norm_y = Self::norm(y);

        if norm_x == 0.0 || norm_y == 0.0 {
            return Ok(0.0);
        }

        Ok(dot / (norm_x * norm_y))
    }

    fn name(&self) -> &str {
        "Cosine"
    }
}

/// Laplacian kernel: K(x, y) = exp(-gamma * ||x - y||_1)
///
/// Similar to RBF but uses L1 (Manhattan) distance instead of L2.
/// More robust to outliers than RBF kernel.
#[derive(Debug, Clone)]
pub struct LaplacianKernel {
    /// Bandwidth parameter (gamma)
    gamma: f64,
}

impl LaplacianKernel {
    /// Create a new Laplacian kernel
    ///
    /// # Arguments
    /// * `gamma` - Bandwidth parameter (must be positive)
    ///
    /// # Examples
    /// ```
    /// use tensorlogic_sklears_kernels::{LaplacianKernel, Kernel};
    ///
    /// let kernel = LaplacianKernel::new(0.5).unwrap();
    /// let x = vec![1.0, 2.0, 3.0];
    /// let y = vec![1.5, 2.5, 3.5];
    /// let sim = kernel.compute(&x, &y).unwrap();
    /// ```
    pub fn new(gamma: f64) -> Result<Self> {
        if gamma <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "gamma".to_string(),
                value: gamma.to_string(),
                reason: "gamma must be positive".to_string(),
            });
        }
        Ok(Self { gamma })
    }

    /// Create from bandwidth parameter sigma: gamma = 1 / sigma
    pub fn from_sigma(sigma: f64) -> Result<Self> {
        if sigma <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "sigma".to_string(),
                value: sigma.to_string(),
                reason: "sigma must be positive".to_string(),
            });
        }
        Self::new(1.0 / sigma)
    }

    /// Compute L1 (Manhattan) distance
    fn l1_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum()
    }
}

impl Kernel for LaplacianKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Laplacian kernel".to_string(),
            });
        }

        let l1_dist = Self::l1_distance(x, y);
        let result = (-self.gamma * l1_dist).exp();
        Ok(result)
    }

    fn name(&self) -> &str {
        "Laplacian"
    }
}

/// Sigmoid (Tanh) kernel: K(x, y) = tanh(alpha * (x · y) + c)
///
/// Neural network inspired kernel. Note: Not guaranteed to be positive semi-definite
/// for all parameter values, but can be useful in practice.
#[derive(Debug, Clone)]
pub struct SigmoidKernel {
    /// Scale parameter (alpha)
    alpha: f64,
    /// Offset parameter (c)
    offset: f64,
}

impl SigmoidKernel {
    /// Create a new sigmoid kernel
    ///
    /// # Arguments
    /// * `alpha` - Scale parameter (typically small, e.g., 0.01)
    /// * `offset` - Offset parameter (typically negative, e.g., -1.0)
    ///
    /// # Examples
    /// ```
    /// use tensorlogic_sklears_kernels::{SigmoidKernel, Kernel};
    ///
    /// let kernel = SigmoidKernel::new(0.01, -1.0).unwrap();
    /// let x = vec![1.0, 2.0, 3.0];
    /// let y = vec![4.0, 5.0, 6.0];
    /// let sim = kernel.compute(&x, &y).unwrap();
    /// ```
    pub fn new(alpha: f64, offset: f64) -> Result<Self> {
        if !alpha.is_finite() {
            return Err(KernelError::InvalidParameter {
                parameter: "alpha".to_string(),
                value: alpha.to_string(),
                reason: "alpha must be finite".to_string(),
            });
        }
        if !offset.is_finite() {
            return Err(KernelError::InvalidParameter {
                parameter: "offset".to_string(),
                value: offset.to_string(),
                reason: "offset must be finite".to_string(),
            });
        }
        Ok(Self { alpha, offset })
    }

    /// Compute dot product
    fn dot_product(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
}

impl Kernel for SigmoidKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "sigmoid kernel".to_string(),
            });
        }

        let dot = Self::dot_product(x, y);
        let result = (self.alpha * dot + self.offset).tanh();
        Ok(result)
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn is_psd(&self) -> bool {
        // Sigmoid kernel is not guaranteed to be PSD for all parameter values
        false
    }
}

/// Chi-squared kernel: K(x, y) = exp(-gamma * Σ((x_i - y_i)² / (x_i + y_i)))
///
/// Especially effective for histogram data and computer vision applications.
/// Handles non-negative features naturally.
#[derive(Debug, Clone)]
pub struct ChiSquaredKernel {
    /// Bandwidth parameter (gamma)
    gamma: f64,
}

impl ChiSquaredKernel {
    /// Create a new chi-squared kernel
    ///
    /// # Arguments
    /// * `gamma` - Bandwidth parameter (must be positive, typically 1.0)
    ///
    /// # Examples
    /// ```
    /// use tensorlogic_sklears_kernels::{ChiSquaredKernel, Kernel};
    ///
    /// let kernel = ChiSquaredKernel::new(1.0).unwrap();
    /// // Histogram data (normalized)
    /// let hist1 = vec![0.2, 0.3, 0.5];
    /// let hist2 = vec![0.25, 0.35, 0.4];
    /// let sim = kernel.compute(&hist1, &hist2).unwrap();
    /// ```
    pub fn new(gamma: f64) -> Result<Self> {
        if gamma <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "gamma".to_string(),
                value: gamma.to_string(),
                reason: "gamma must be positive".to_string(),
            });
        }
        Ok(Self { gamma })
    }

    /// Compute chi-squared distance between histograms
    ///
    /// Uses the formula: Σ((x_i - y_i)² / (x_i + y_i + epsilon))
    /// Small epsilon prevents division by zero
    fn chi_squared_distance(x: &[f64], y: &[f64]) -> f64 {
        const EPSILON: f64 = 1e-10;
        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| {
                let sum = xi + yi + EPSILON;
                let diff = xi - yi;
                diff * diff / sum
            })
            .sum()
    }
}

impl Kernel for ChiSquaredKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "chi-squared kernel".to_string(),
            });
        }

        let chi_sq_dist = Self::chi_squared_distance(x, y);
        let result = (-self.gamma * chi_sq_dist).exp();
        Ok(result)
    }

    fn name(&self) -> &str {
        "ChiSquared"
    }
}

/// Histogram Intersection kernel: K(x, y) = Σ min(x_i, y_i)
///
/// Measures overlap between histograms. Particularly effective for
/// image classification and object recognition with histogram features.
/// Requires non-negative input features.
#[derive(Debug, Clone)]
pub struct HistogramIntersectionKernel;

impl HistogramIntersectionKernel {
    /// Create a new histogram intersection kernel
    ///
    /// # Examples
    /// ```
    /// use tensorlogic_sklears_kernels::{HistogramIntersectionKernel, Kernel};
    ///
    /// let kernel = HistogramIntersectionKernel::new();
    /// // Histogram data (normalized)
    /// let hist1 = vec![0.2, 0.3, 0.5];
    /// let hist2 = vec![0.25, 0.35, 0.4];
    /// let sim = kernel.compute(&hist1, &hist2).unwrap();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Compute histogram intersection
    fn intersection(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(xi, yi)| xi.min(*yi)).sum()
    }
}

impl Default for HistogramIntersectionKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel for HistogramIntersectionKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "histogram intersection kernel".to_string(),
            });
        }

        // Check for negative values (histograms should be non-negative)
        if x.iter().any(|&v| v < 0.0) || y.iter().any(|&v| v < 0.0) {
            return Err(KernelError::ComputationError(
                "Histogram features must be non-negative".to_string(),
            ));
        }

        let result = Self::intersection(x, y);
        Ok(result)
    }

    fn name(&self) -> &str {
        "HistogramIntersection"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_kernel() {
        let kernel = LinearKernel::new();
        assert_eq!(kernel.name(), "Linear");

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_linear_kernel_dimension_mismatch() {
        let kernel = LinearKernel::new();
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = kernel.compute(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_polynomial_kernel() {
        let kernel = PolynomialKernel::new(2, 1.0).unwrap();
        assert_eq!(kernel.name(), "Polynomial");

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        // dot = 1*3 + 2*4 = 11
        // (11 + 1)^2 = 144
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 144.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_kernel_degree_zero() {
        let result = PolynomialKernel::new(0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rbf_kernel() {
        let config = RbfKernelConfig::new(0.5);
        let kernel = RbfKernel::new(config).unwrap();
        assert_eq!(kernel.name(), "RBF");

        // Same vectors should give similarity 1.0
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // Different vectors should give similarity < 1.0
        let y = vec![2.0, 3.0, 4.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!(result < 1.0);
        assert!(result > 0.0);
    }

    #[test]
    fn test_rbf_kernel_invalid_gamma() {
        let config = RbfKernelConfig::new(-0.5);
        let result = RbfKernel::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_cosine_kernel() {
        let kernel = CosineKernel::new();
        assert_eq!(kernel.name(), "Cosine");

        // Parallel vectors (same direction)
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // Orthogonal vectors
        let x = vec![1.0, 0.0];
        let y = vec![0.0, 1.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!(result.abs() < 1e-10);

        // Opposite vectors
        let x = vec![1.0, 2.0];
        let y = vec![-1.0, -2.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_kernel_zero_vector() {
        let kernel = CosineKernel::new();
        let x = vec![0.0, 0.0];
        let y = vec![1.0, 2.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!(result.abs() < 1e-10);
    }

    // ===== Laplacian Kernel Tests =====

    #[test]
    fn test_laplacian_kernel_basic() {
        let kernel = LaplacianKernel::new(0.5).unwrap();
        assert_eq!(kernel.name(), "Laplacian");

        // Same vectors should give similarity 1.0
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // Different vectors should give similarity < 1.0
        let y = vec![2.0, 3.0, 4.0];
        let result = kernel.compute(&x, &y).unwrap();
        // L1 distance = |1-2| + |2-3| + |3-4| = 3
        // exp(-0.5 * 3) = exp(-1.5) ≈ 0.2231
        assert!(result < 1.0);
        assert!(result > 0.0);
        assert!((result - 0.5_f64.mul_add(-3.0, 0.0).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_kernel_from_sigma() {
        let kernel = LaplacianKernel::from_sigma(2.0).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_kernel_invalid_gamma() {
        let result = LaplacianKernel::new(-0.5);
        assert!(result.is_err());
        let result = LaplacianKernel::new(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_laplacian_kernel_invalid_sigma() {
        let result = LaplacianKernel::from_sigma(-1.0);
        assert!(result.is_err());
        let result = LaplacianKernel::from_sigma(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_laplacian_kernel_dimension_mismatch() {
        let kernel = LaplacianKernel::new(0.5).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = kernel.compute(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_laplacian_kernel_outlier_robustness() {
        // Laplacian should be more robust to outliers than RBF
        let laplacian = LaplacianKernel::new(0.5).unwrap();
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 10.0]; // Outlier in last dimension

        let lap_sim = laplacian.compute(&x, &y).unwrap();
        let rbf_sim = rbf.compute(&x, &y).unwrap();

        // Laplacian should be less affected by the outlier
        assert!(lap_sim > rbf_sim);
    }

    // ===== Sigmoid Kernel Tests =====

    #[test]
    fn test_sigmoid_kernel_basic() {
        let kernel = SigmoidKernel::new(0.01, -1.0).unwrap();
        assert_eq!(kernel.name(), "Sigmoid");
        assert!(!kernel.is_psd()); // Sigmoid is not guaranteed PSD

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let result = kernel.compute(&x, &y).unwrap();

        // Result should be in [-1, 1] due to tanh
        assert!(result >= -1.0);
        assert!(result <= 1.0);
    }

    #[test]
    fn test_sigmoid_kernel_range() {
        let kernel = SigmoidKernel::new(0.01, -1.0).unwrap();

        // Test various vector pairs
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let result = kernel.compute(&x, &y).unwrap();

        // tanh always returns values in [-1, 1]
        assert!(result >= -1.0);
        assert!(result <= 1.0);
    }

    #[test]
    fn test_sigmoid_kernel_invalid_parameters() {
        let result = SigmoidKernel::new(f64::INFINITY, -1.0);
        assert!(result.is_err());

        let result = SigmoidKernel::new(0.01, f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_sigmoid_kernel_dimension_mismatch() {
        let kernel = SigmoidKernel::new(0.01, -1.0).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = kernel.compute(&x, &y);
        assert!(result.is_err());
    }

    // ===== Chi-Squared Kernel Tests =====

    #[test]
    fn test_chi_squared_kernel_basic() {
        let kernel = ChiSquaredKernel::new(1.0).unwrap();
        assert_eq!(kernel.name(), "ChiSquared");

        // Same histograms should give similarity 1.0
        let hist1 = vec![0.2, 0.3, 0.5];
        let hist2 = vec![0.2, 0.3, 0.5];
        let result = kernel.compute(&hist1, &hist2).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_kernel_histograms() {
        let kernel = ChiSquaredKernel::new(1.0).unwrap();

        // Similar histograms
        let hist1 = vec![0.2, 0.3, 0.5];
        let hist2 = vec![0.25, 0.35, 0.4];
        let result = kernel.compute(&hist1, &hist2).unwrap();

        // Should be high similarity (close to 1.0)
        assert!(result > 0.8);
        assert!(result < 1.0);
    }

    #[test]
    fn test_chi_squared_kernel_invalid_gamma() {
        let result = ChiSquaredKernel::new(-1.0);
        assert!(result.is_err());

        let result = ChiSquaredKernel::new(0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_chi_squared_kernel_dimension_mismatch() {
        let kernel = ChiSquaredKernel::new(1.0).unwrap();
        let x = vec![0.2, 0.3];
        let y = vec![0.2, 0.3, 0.5];
        let result = kernel.compute(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_chi_squared_kernel_different_gammas() {
        let kernel1 = ChiSquaredKernel::new(0.5).unwrap();
        let kernel2 = ChiSquaredKernel::new(2.0).unwrap();

        let hist1 = vec![0.2, 0.3, 0.5];
        let hist2 = vec![0.25, 0.35, 0.4];

        let result1 = kernel1.compute(&hist1, &hist2).unwrap();
        let result2 = kernel2.compute(&hist1, &hist2).unwrap();

        // Smaller gamma should give higher similarity
        assert!(result1 > result2);
    }

    // ===== Histogram Intersection Kernel Tests =====

    #[test]
    fn test_histogram_intersection_basic() {
        let kernel = HistogramIntersectionKernel::new();
        assert_eq!(kernel.name(), "HistogramIntersection");

        // Identical normalized histograms
        let hist1 = vec![0.2, 0.3, 0.5];
        let hist2 = vec![0.2, 0.3, 0.5];
        let result = kernel.compute(&hist1, &hist2).unwrap();
        assert!((result - 1.0).abs() < 1e-10); // Sum = 1.0
    }

    #[test]
    fn test_histogram_intersection_partial_overlap() {
        let kernel = HistogramIntersectionKernel::new();

        let hist1 = vec![0.5, 0.3, 0.2];
        let hist2 = vec![0.3, 0.4, 0.3];

        // Intersection = min(0.5,0.3) + min(0.3,0.4) + min(0.2,0.3)
        //               = 0.3 + 0.3 + 0.2 = 0.8
        let result = kernel.compute(&hist1, &hist2).unwrap();
        assert!((result - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_histogram_intersection_no_overlap() {
        let kernel = HistogramIntersectionKernel::new();

        let hist1 = vec![1.0, 0.0, 0.0];
        let hist2 = vec![0.0, 1.0, 0.0];

        // No overlap
        let result = kernel.compute(&hist1, &hist2).unwrap();
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_histogram_intersection_negative_values() {
        let kernel = HistogramIntersectionKernel::new();

        let hist1 = vec![0.5, -0.3, 0.2];
        let hist2 = vec![0.3, 0.4, 0.3];

        // Should error on negative values
        let result = kernel.compute(&hist1, &hist2);
        assert!(result.is_err());
    }

    #[test]
    fn test_histogram_intersection_dimension_mismatch() {
        let kernel = HistogramIntersectionKernel::new();
        let x = vec![0.2, 0.3];
        let y = vec![0.2, 0.3, 0.5];
        let result = kernel.compute(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_histogram_intersection_unnormalized() {
        let kernel = HistogramIntersectionKernel::new();

        // Unnormalized histograms (raw counts)
        let hist1 = vec![10.0, 20.0, 30.0];
        let hist2 = vec![15.0, 25.0, 20.0];

        // Intersection = min(10,15) + min(20,25) + min(30,20)
        //               = 10 + 20 + 20 = 50
        let result = kernel.compute(&hist1, &hist2).unwrap();
        assert!((result - 50.0).abs() < 1e-10);
    }

    // ===== Cross-kernel comparison tests =====

    #[test]
    fn test_rbf_vs_laplacian() {
        let rbf = RbfKernel::new(RbfKernelConfig::new(1.0)).unwrap();
        let lap = LaplacianKernel::new(1.0).unwrap();

        let x = vec![0.0, 0.0, 0.0];
        let y = vec![1.0, 1.0, 1.0];

        let rbf_sim = rbf.compute(&x, &y).unwrap();
        let lap_sim = lap.compute(&x, &y).unwrap();

        // Both should give values between 0 and 1
        assert!((0.0..=1.0).contains(&rbf_sim));
        assert!((0.0..=1.0).contains(&lap_sim));

        // RBF: exp(-gamma * ||x-y||^2) = exp(-1.0 * 3) = exp(-3)
        // Laplacian: exp(-gamma * ||x-y||_1) = exp(-1.0 * 3) = exp(-3)
        // For this specific case with equal gamma, they should be equal
        assert!((rbf_sim - lap_sim).abs() < 1e-10);

        // Test with different distance pattern where RBF decays faster
        let x2 = vec![0.0, 0.0];
        let y2 = vec![2.0, 0.0];
        // L2^2 distance = 4, L1 distance = 2
        // RBF: exp(-1.0 * 4) = exp(-4)
        // Laplacian: exp(-1.0 * 2) = exp(-2)
        let rbf_sim2 = rbf.compute(&x2, &y2).unwrap();
        let lap_sim2 = lap.compute(&x2, &y2).unwrap();
        // RBF should decay faster (lower similarity) due to squared distance
        assert!(rbf_sim2 < lap_sim2);
    }

    #[test]
    fn test_all_new_kernels_symmetry() {
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(LaplacianKernel::new(0.5).unwrap()),
            Box::new(SigmoidKernel::new(0.01, -1.0).unwrap()),
            Box::new(ChiSquaredKernel::new(1.0).unwrap()),
            Box::new(HistogramIntersectionKernel::new()),
        ];

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        for kernel in kernels {
            let k_xy = kernel.compute(&x, &y).unwrap();
            let k_yx = kernel.compute(&y, &x).unwrap();
            assert!(
                (k_xy - k_yx).abs() < 1e-10,
                "Kernel {} not symmetric",
                kernel.name()
            );
        }
    }

    #[test]
    fn test_all_new_kernels_self_similarity() {
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(LaplacianKernel::new(0.5).unwrap()),
            Box::new(ChiSquaredKernel::new(1.0).unwrap()),
        ];

        let x = vec![1.0, 2.0, 3.0];

        for kernel in kernels {
            let k_xx = kernel.compute(&x, &x).unwrap();
            // For distance-based kernels, self-similarity should be 1.0
            assert!(
                (k_xx - 1.0).abs() < 1e-10,
                "Kernel {} self-similarity not 1.0: got {}",
                kernel.name(),
                k_xx
            );
        }
    }
}
