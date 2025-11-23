//! Tensor-based kernels operating on feature vectors.
//!
//! These kernels work directly on tensor/vector representations,
//! complementing the logic-derived kernels.

use crate::error::{KernelError, Result};
use crate::types::{Kernel, RbfKernelConfig};

/// Linear kernel: K(x, y) = x · y
///
/// The simplest kernel, equivalent to inner product in feature space.
#[derive(Clone)]
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
#[derive(Clone)]
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

    /// Get the degree parameter
    pub fn degree(&self) -> usize {
        self.degree as usize
    }

    /// Get the constant parameter
    pub fn constant(&self) -> f64 {
        self.constant
    }

    /// Compute dot product
    fn dot_product(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
}

impl PolynomialKernel {
    /// Compute the kernel value and gradient with respect to the constant term.
    ///
    /// Returns (K(x,y), dK/dc) where c is the constant term.
    ///
    /// # Example
    /// ```rust
    /// use tensorlogic_sklears_kernels::PolynomialKernel;
    ///
    /// let kernel = PolynomialKernel::new(2, 1.0).unwrap();
    /// let x = vec![1.0, 2.0];
    /// let y = vec![3.0, 4.0];
    /// let (value, grad_c) = kernel.compute_with_constant_gradient(&x, &y).unwrap();
    /// ```
    pub fn compute_with_constant_gradient(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "polynomial kernel gradient".to_string(),
            });
        }

        let dot = Self::dot_product(x, y);
        let base = dot + self.constant;

        if base.abs() < 1e-10 && self.degree > 1 {
            // Near zero base with degree > 1, gradient is zero
            return Ok((0.0, 0.0));
        }

        let k = base.powi(self.degree as i32);
        // dK/dc = d * (dot + c)^(d-1)
        let grad_c = (self.degree as f64) * base.powi(self.degree as i32 - 1);

        Ok((k, grad_c))
    }

    /// Compute the kernel value and all gradients (w.r.t. constant and degree).
    ///
    /// Returns (K(x,y), dK/dc, dK/dd) where:
    /// - dK/dc is the gradient w.r.t. constant term
    /// - dK/dd is the gradient w.r.t. degree (treating degree as continuous)
    ///
    /// Note: dK/dd = K * ln(dot + c), useful for continuous degree optimization.
    pub fn compute_with_all_gradients(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64, f64)> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "polynomial kernel all gradients".to_string(),
            });
        }

        let dot = Self::dot_product(x, y);
        let base = dot + self.constant;

        if base.abs() < 1e-10 && self.degree > 1 {
            return Ok((0.0, 0.0, 0.0));
        }

        let k = base.powi(self.degree as i32);

        // dK/dc = d * (dot + c)^(d-1)
        let grad_c = (self.degree as f64) * base.powi(self.degree as i32 - 1);

        // dK/dd = K * ln(dot + c), treating degree as continuous
        let grad_d = if base > 0.0 {
            k * base.ln()
        } else if base < 0.0 && self.degree.is_multiple_of(2) {
            // Even power of negative number
            k * base.abs().ln()
        } else {
            // Negative base with odd power - gradient undefined for continuous degree
            f64::NAN
        };

        Ok((k, grad_c, grad_d))
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
#[derive(Clone)]
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

    /// Get the configuration
    pub fn config(&self) -> &RbfKernelConfig {
        &self.config
    }

    /// Get the gamma parameter
    pub fn gamma(&self) -> f64 {
        self.config.gamma
    }

    /// Compute squared Euclidean distance
    fn squared_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
    }

    /// Compute the kernel value and gradient with respect to gamma.
    ///
    /// Returns (K(x,y), dK/d_gamma) for hyperparameter optimization.
    ///
    /// # Example
    /// ```rust
    /// use tensorlogic_sklears_kernels::{RbfKernel, RbfKernelConfig};
    ///
    /// let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
    /// let x = vec![1.0, 2.0];
    /// let y = vec![3.0, 4.0];
    /// let (value, grad) = kernel.compute_with_gradient(&x, &y).unwrap();
    /// ```
    pub fn compute_with_gradient(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "RBF kernel gradient".to_string(),
            });
        }

        let sq_dist = Self::squared_distance(x, y);
        let k = (-self.config.gamma * sq_dist).exp();

        // dK/d_gamma = -sq_dist * exp(-gamma * sq_dist) = -sq_dist * K
        let grad_gamma = -sq_dist * k;

        Ok((k, grad_gamma))
    }

    /// Compute the kernel gradient with respect to length scale (sigma = 1/sqrt(2*gamma)).
    ///
    /// This is useful when parameterizing in terms of length scale instead of gamma.
    pub fn compute_with_length_scale_gradient(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "RBF kernel length scale gradient".to_string(),
            });
        }

        let sq_dist = Self::squared_distance(x, y);
        let k = (-self.config.gamma * sq_dist).exp();

        // If gamma = 1/(2*sigma^2), then dK/d_sigma = sq_dist * gamma^{3/2} * sqrt(2) * K
        // Simplified: dK/d_sigma = sq_dist / sigma^3 * K where sigma = 1/sqrt(2*gamma)
        let sigma = 1.0 / (2.0 * self.config.gamma).sqrt();
        let grad_sigma = sq_dist / (sigma * sigma * sigma) * k;

        Ok((k, grad_sigma))
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
#[derive(Clone)]
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

    /// Get the gamma parameter
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Compute L1 (Manhattan) distance
    fn l1_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum()
    }

    /// Compute the kernel value and gradient with respect to gamma.
    ///
    /// Returns (K(x,y), dK/d_gamma) for hyperparameter optimization.
    ///
    /// # Example
    /// ```rust
    /// use tensorlogic_sklears_kernels::LaplacianKernel;
    ///
    /// let kernel = LaplacianKernel::new(0.5).unwrap();
    /// let x = vec![1.0, 2.0];
    /// let y = vec![3.0, 4.0];
    /// let (value, grad) = kernel.compute_with_gradient(&x, &y).unwrap();
    /// ```
    pub fn compute_with_gradient(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Laplacian kernel gradient".to_string(),
            });
        }

        let l1_dist = Self::l1_distance(x, y);
        let k = (-self.gamma * l1_dist).exp();

        // dK/d_gamma = -l1_dist * exp(-gamma * l1_dist) = -l1_dist * K
        let grad_gamma = -l1_dist * k;

        Ok((k, grad_gamma))
    }

    /// Compute the kernel gradient with respect to sigma (where gamma = 1/sigma).
    ///
    /// Returns (K(x,y), dK/d_sigma) for length-scale parameterization.
    pub fn compute_with_sigma_gradient(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Laplacian kernel sigma gradient".to_string(),
            });
        }

        let l1_dist = Self::l1_distance(x, y);
        let k = (-self.gamma * l1_dist).exp();

        // If gamma = 1/sigma, then dK/d_sigma = l1_dist/sigma² * K = l1_dist * gamma² * K
        let grad_sigma = l1_dist * self.gamma * self.gamma * k;

        Ok((k, grad_sigma))
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

    /// Get the alpha parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the offset parameter
    pub fn offset(&self) -> f64 {
        self.offset
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

    /// Get the gamma parameter
    pub fn gamma(&self) -> f64 {
        self.gamma
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

/// Matérn kernel: K(x, y) = σ² * (2^(1-ν) / Γ(ν)) * (√(2ν) * r / l)^ν * K_ν(√(2ν) * r / l)
///
/// A generalization of the RBF kernel with an additional smoothness parameter nu.
/// Widely used in Gaussian Process regression. Special cases:
/// - nu = 1/2: Exponential kernel (same as Laplacian)
/// - nu = 3/2: Once differentiable
/// - nu = 5/2: Twice differentiable
/// - nu → ∞: RBF kernel
#[derive(Debug, Clone)]
pub struct MaternKernel {
    /// Length scale parameter
    length_scale: f64,
    /// Smoothness parameter (nu)
    /// Common values: 0.5 (exponential), 1.5, 2.5
    nu: f64,
}

impl MaternKernel {
    /// Create a new Matérn kernel
    ///
    /// # Arguments
    /// * `length_scale` - Controls the length scale of the kernel (must be positive)
    /// * `nu` - Smoothness parameter (must be positive, common: 0.5, 1.5, 2.5)
    ///
    /// # Examples
    /// ```
    /// use tensorlogic_sklears_kernels::{MaternKernel, Kernel};
    ///
    /// // nu = 1.5 gives once-differentiable functions
    /// let kernel = MaternKernel::new(1.0, 1.5).unwrap();
    /// let x = vec![1.0, 2.0, 3.0];
    /// let y = vec![1.5, 2.5, 3.5];
    /// let sim = kernel.compute(&x, &y).unwrap();
    /// ```
    pub fn new(length_scale: f64, nu: f64) -> Result<Self> {
        if length_scale <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "length_scale".to_string(),
                value: length_scale.to_string(),
                reason: "length_scale must be positive".to_string(),
            });
        }
        if nu <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "nu".to_string(),
                value: nu.to_string(),
                reason: "nu must be positive".to_string(),
            });
        }
        Ok(Self { length_scale, nu })
    }

    /// Create Matérn kernel with nu = 1/2 (exponential kernel)
    pub fn exponential(length_scale: f64) -> Result<Self> {
        Self::new(length_scale, 0.5)
    }

    /// Create Matérn kernel with nu = 3/2 (once differentiable)
    pub fn nu_3_2(length_scale: f64) -> Result<Self> {
        Self::new(length_scale, 1.5)
    }

    /// Create Matérn kernel with nu = 5/2 (twice differentiable)
    pub fn nu_5_2(length_scale: f64) -> Result<Self> {
        Self::new(length_scale, 2.5)
    }

    /// Get the length scale parameter
    pub fn length_scale(&self) -> f64 {
        self.length_scale
    }

    /// Get the smoothness parameter nu
    pub fn nu(&self) -> f64 {
        self.nu
    }

    /// Compute Euclidean distance
    fn euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute the kernel value and gradient with respect to length_scale.
    ///
    /// Returns (K(x,y), dK/dl) where l is the length_scale parameter.
    /// Only supports nu = 0.5, 1.5, 2.5 (the most common cases).
    ///
    /// # Example
    /// ```rust
    /// use tensorlogic_sklears_kernels::MaternKernel;
    ///
    /// let kernel = MaternKernel::nu_3_2(1.0).unwrap();
    /// let x = vec![1.0, 2.0];
    /// let y = vec![3.0, 4.0];
    /// let (value, grad_l) = kernel.compute_with_length_scale_gradient(&x, &y).unwrap();
    /// ```
    pub fn compute_with_length_scale_gradient(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Matérn kernel length scale gradient".to_string(),
            });
        }

        let dist = Self::euclidean_distance(x, y);

        // Handle same point case
        if dist < 1e-10 {
            return Ok((1.0, 0.0)); // K=1, gradient=0 at same point
        }

        let l = self.length_scale;
        let sqrt_2nu = (2.0 * self.nu).sqrt();
        let scaled_dist = sqrt_2nu * dist / l;

        // Use closed-form expressions for common nu values
        if (self.nu - 0.5).abs() < 1e-10 {
            // nu = 1/2: K = exp(-r/l)
            // dK/dl = (r/l²) * K = scaled_dist/l * K
            let k = (-scaled_dist).exp();
            let grad_l = (dist / (l * l)) * k;
            Ok((k, grad_l))
        } else if (self.nu - 1.5).abs() < 1e-10 {
            // nu = 3/2: K = (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
            // Let z = sqrt(3)*r/l
            // K = (1 + z) * exp(-z)
            // dK/dl = (z²/l) * exp(-z) = (3*r²/l³) * exp(-z)
            let sqrt3 = 3.0_f64.sqrt();
            let z = sqrt3 * dist / l;
            let exp_neg_z = (-z).exp();
            let k = (1.0 + z) * exp_neg_z;
            let grad_l = (z * z / l) * exp_neg_z;
            Ok((k, grad_l))
        } else if (self.nu - 2.5).abs() < 1e-10 {
            // nu = 5/2: K = (1 + sqrt(5)*r/l + 5*r²/(3*l²)) * exp(-sqrt(5)*r/l)
            // Let z = sqrt(5)*r/l
            // K = (1 + z + z²/3) * exp(-z)
            // dK/dl = (z²/l) * (1 + z) / 3 * exp(-z)
            let sqrt5 = 5.0_f64.sqrt();
            let z = sqrt5 * dist / l;
            let exp_neg_z = (-z).exp();
            let k = (1.0 + z + z * z / 3.0) * exp_neg_z;
            // dK/dl = z²*(1+z)/(3*l) * exp(-z)
            let grad_l = (z * z * (1.0 + z) / (3.0 * l)) * exp_neg_z;
            Ok((k, grad_l))
        } else {
            // General case: use finite differences approximation
            let k = (1.0 + scaled_dist) * (-scaled_dist).exp();
            let eps = 1e-6;
            let l_plus = l + eps;
            let scaled_dist_plus = sqrt_2nu * dist / l_plus;
            let k_plus = (1.0 + scaled_dist_plus) * (-scaled_dist_plus).exp();
            let grad_l = (k_plus - k) / eps;
            Ok((k, grad_l))
        }
    }
}

impl Kernel for MaternKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Matérn kernel".to_string(),
            });
        }

        let dist = Self::euclidean_distance(x, y);

        // Handle same point case
        if dist < 1e-10 {
            return Ok(1.0);
        }

        // Scaled distance
        let scaled_dist = (2.0 * self.nu).sqrt() * dist / self.length_scale;

        // Use closed-form expressions for common nu values
        let result = if (self.nu - 0.5).abs() < 1e-10 {
            // nu = 1/2: exponential kernel
            (-scaled_dist).exp()
        } else if (self.nu - 1.5).abs() < 1e-10 {
            // nu = 3/2: once differentiable
            (1.0 + scaled_dist) * (-scaled_dist).exp()
        } else if (self.nu - 2.5).abs() < 1e-10 {
            // nu = 5/2: twice differentiable
            (1.0 + scaled_dist + scaled_dist * scaled_dist / 3.0) * (-scaled_dist).exp()
        } else {
            // General case: use approximation for other nu values
            // For simplicity, use nu=1.5 as fallback
            (1.0 + scaled_dist) * (-scaled_dist).exp()
        };

        Ok(result)
    }

    fn name(&self) -> &str {
        "Matérn"
    }
}

/// Rational Quadratic kernel: K(x, y) = (1 + ||x-y||² / (2 * alpha * l²))^(-alpha)
///
/// Can be seen as a scale mixture of RBF kernels with different length scales.
/// As alpha → ∞, this kernel becomes equivalent to the RBF kernel.
/// Useful when data exhibits multiple characteristic length scales.
#[derive(Debug, Clone)]
pub struct RationalQuadraticKernel {
    /// Length scale parameter
    length_scale: f64,
    /// Scale mixture parameter (alpha)
    /// Controls the relative weighting of large vs small scale variations
    alpha: f64,
}

impl RationalQuadraticKernel {
    /// Create a new Rational Quadratic kernel
    ///
    /// # Arguments
    /// * `length_scale` - Controls the length scale (must be positive)
    /// * `alpha` - Scale mixture parameter (must be positive, typically > 1)
    ///
    /// # Examples
    /// ```
    /// use tensorlogic_sklears_kernels::{RationalQuadraticKernel, Kernel};
    ///
    /// let kernel = RationalQuadraticKernel::new(1.0, 2.0).unwrap();
    /// let x = vec![1.0, 2.0, 3.0];
    /// let y = vec![1.5, 2.5, 3.5];
    /// let sim = kernel.compute(&x, &y).unwrap();
    /// ```
    pub fn new(length_scale: f64, alpha: f64) -> Result<Self> {
        if length_scale <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "length_scale".to_string(),
                value: length_scale.to_string(),
                reason: "length_scale must be positive".to_string(),
            });
        }
        if alpha <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "alpha".to_string(),
                value: alpha.to_string(),
                reason: "alpha must be positive".to_string(),
            });
        }
        Ok(Self {
            length_scale,
            alpha,
        })
    }

    /// Get the length scale parameter
    pub fn length_scale(&self) -> f64 {
        self.length_scale
    }

    /// Get the alpha parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Compute squared Euclidean distance
    fn squared_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
    }

    /// Compute the kernel value and gradient with respect to length_scale.
    ///
    /// Returns (K(x,y), dK/dl) where l is the length_scale parameter.
    ///
    /// # Example
    /// ```rust
    /// use tensorlogic_sklears_kernels::RationalQuadraticKernel;
    ///
    /// let kernel = RationalQuadraticKernel::new(1.0, 2.0).unwrap();
    /// let x = vec![1.0, 2.0];
    /// let y = vec![3.0, 4.0];
    /// let (value, grad_l) = kernel.compute_with_length_scale_gradient(&x, &y).unwrap();
    /// ```
    pub fn compute_with_length_scale_gradient(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Rational Quadratic kernel length scale gradient".to_string(),
            });
        }

        let sq_dist = Self::squared_distance(x, y);
        let l_sq = self.length_scale * self.length_scale;
        let term = 1.0 + sq_dist / (2.0 * self.alpha * l_sq);
        let k = term.powf(-self.alpha);

        // dK/dl = K * alpha * sq_dist / (alpha * l³ * term) = K * sq_dist / (l³ * term)
        // Simplified: dK/dl = K * sq_dist / (l * (2 * alpha * l² + sq_dist))
        let denom = self.length_scale * (2.0 * self.alpha * l_sq + sq_dist);
        let grad_l = if denom.abs() > 1e-10 {
            k * sq_dist / denom
        } else {
            0.0
        };

        Ok((k, grad_l))
    }

    /// Compute the kernel value and gradient with respect to alpha.
    ///
    /// Returns (K(x,y), dK/d_alpha) where alpha is the scale mixture parameter.
    pub fn compute_with_alpha_gradient(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Rational Quadratic kernel alpha gradient".to_string(),
            });
        }

        let sq_dist = Self::squared_distance(x, y);
        let l_sq = self.length_scale * self.length_scale;
        let u = sq_dist / (2.0 * self.alpha * l_sq);
        let term = 1.0 + u;
        let k = term.powf(-self.alpha);

        // dK/d_alpha = K * (u/term - ln(term))
        let grad_alpha = k * (u / term - term.ln());

        Ok((k, grad_alpha))
    }

    /// Compute the kernel value and all gradients (length_scale and alpha).
    ///
    /// Returns (K(x,y), dK/dl, dK/d_alpha).
    pub fn compute_with_all_gradients(&self, x: &[f64], y: &[f64]) -> Result<(f64, f64, f64)> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Rational Quadratic kernel all gradients".to_string(),
            });
        }

        let sq_dist = Self::squared_distance(x, y);
        let l_sq = self.length_scale * self.length_scale;
        let u = sq_dist / (2.0 * self.alpha * l_sq);
        let term = 1.0 + u;
        let k = term.powf(-self.alpha);

        // dK/dl
        let denom = self.length_scale * (2.0 * self.alpha * l_sq + sq_dist);
        let grad_l = if denom.abs() > 1e-10 {
            k * sq_dist / denom
        } else {
            0.0
        };

        // dK/d_alpha
        let grad_alpha = k * (u / term - term.ln());

        Ok((k, grad_l, grad_alpha))
    }
}

impl Kernel for RationalQuadraticKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Rational Quadratic kernel".to_string(),
            });
        }

        let sq_dist = Self::squared_distance(x, y);
        let term = 1.0 + sq_dist / (2.0 * self.alpha * self.length_scale * self.length_scale);
        let result = term.powf(-self.alpha);
        Ok(result)
    }

    fn name(&self) -> &str {
        "RationalQuadratic"
    }
}

/// Periodic kernel: K(x, y) = exp(-2 * sin²(π * ||x-y|| / period) / l²)
///
/// Captures periodic patterns in data. Useful for modeling seasonal effects,
/// oscillatory behavior, and other repeating patterns.
#[derive(Debug, Clone)]
pub struct PeriodicKernel {
    /// Period of the periodic pattern
    period: f64,
    /// Length scale parameter
    length_scale: f64,
}

impl PeriodicKernel {
    /// Create a new Periodic kernel
    ///
    /// # Arguments
    /// * `period` - Period of the repeating pattern (must be positive)
    /// * `length_scale` - Controls smoothness within each period (must be positive)
    ///
    /// # Examples
    /// ```
    /// use tensorlogic_sklears_kernels::{PeriodicKernel, Kernel};
    ///
    /// // Model data with period = 24 (e.g., hours in a day)
    /// let kernel = PeriodicKernel::new(24.0, 1.0).unwrap();
    /// let x = vec![1.0, 2.0];
    /// let y = vec![25.0, 26.0];  // One period later
    /// let sim = kernel.compute(&x, &y).unwrap();
    /// // Similarity is high because points are one period apart
    /// ```
    pub fn new(period: f64, length_scale: f64) -> Result<Self> {
        if period <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "period".to_string(),
                value: period.to_string(),
                reason: "period must be positive".to_string(),
            });
        }
        if length_scale <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "length_scale".to_string(),
                value: length_scale.to_string(),
                reason: "length_scale must be positive".to_string(),
            });
        }
        Ok(Self {
            period,
            length_scale,
        })
    }

    /// Get the period parameter
    pub fn period(&self) -> f64 {
        self.period
    }

    /// Get the length scale parameter
    pub fn length_scale(&self) -> f64 {
        self.length_scale
    }

    /// Compute Euclidean distance
    fn euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

impl Kernel for PeriodicKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Periodic kernel".to_string(),
            });
        }

        let dist = Self::euclidean_distance(x, y);
        let sin_term = (std::f64::consts::PI * dist / self.period).sin();
        let result = (-2.0 * sin_term * sin_term / (self.length_scale * self.length_scale)).exp();
        Ok(result)
    }

    fn name(&self) -> &str {
        "Periodic"
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

    // ============================================================================
    // Tests for advanced kernels (Matérn, Rational Quadratic, Periodic)
    // ============================================================================

    #[test]
    fn test_matern_kernel_nu_half() {
        // nu = 0.5 should give exponential kernel (same as Laplacian with adjusted gamma)
        let kernel = MaternKernel::exponential(1.0).unwrap();
        assert_eq!(kernel.name(), "Matérn");
        assert!((kernel.nu() - 0.5).abs() < 1e-10);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.5, 2.5, 3.5];

        let sim = kernel.compute(&x, &y).unwrap();
        assert!(sim > 0.0 && sim <= 1.0);

        // Self-similarity should be 1.0
        let self_sim = kernel.compute(&x, &x).unwrap();
        assert!((self_sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matern_kernel_nu_3_2() {
        let kernel = MaternKernel::nu_3_2(1.0).unwrap();
        assert!((kernel.nu() - 1.5).abs() < 1e-10);

        let x = vec![0.0, 0.0];
        let y = vec![1.0, 0.0];

        let sim = kernel.compute(&x, &y).unwrap();
        assert!(sim > 0.0 && sim < 1.0);

        // Closer points should have higher similarity
        let y_close = vec![0.1, 0.0];
        let sim_close = kernel.compute(&x, &y_close).unwrap();
        assert!(sim_close > sim);
    }

    #[test]
    fn test_matern_kernel_nu_5_2() {
        let kernel = MaternKernel::nu_5_2(1.0).unwrap();
        assert!((kernel.nu() - 2.5).abs() < 1e-10);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matern_kernel_invalid_parameters() {
        // Invalid length scale
        assert!(MaternKernel::new(0.0, 1.5).is_err());
        assert!(MaternKernel::new(-1.0, 1.5).is_err());

        // Invalid nu
        assert!(MaternKernel::new(1.0, 0.0).is_err());
        assert!(MaternKernel::new(1.0, -1.0).is_err());
    }

    #[test]
    fn test_matern_kernel_dimension_mismatch() {
        let kernel = MaternKernel::nu_3_2(1.0).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];

        assert!(kernel.compute(&x, &y).is_err());
    }

    #[test]
    fn test_matern_kernel_symmetry() {
        let kernel = MaternKernel::nu_3_2(1.0).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let k_xy = kernel.compute(&x, &y).unwrap();
        let k_yx = kernel.compute(&y, &x).unwrap();
        assert!((k_xy - k_yx).abs() < 1e-10);
    }

    #[test]
    fn test_matern_smoothness_ordering() {
        // With same length scale, nu=0.5 should be less smooth than nu=2.5
        let kernel_rough = MaternKernel::exponential(1.0).unwrap();
        let kernel_smooth = MaternKernel::nu_5_2(1.0).unwrap();

        let x = vec![0.0];
        let y = vec![0.5];

        let sim_rough = kernel_rough.compute(&x, &y).unwrap();
        let sim_smooth = kernel_smooth.compute(&x, &y).unwrap();

        // Smoother kernel decays slower at short distances
        assert!(sim_smooth > sim_rough);
    }

    #[test]
    fn test_rational_quadratic_kernel_basic() {
        let kernel = RationalQuadraticKernel::new(1.0, 2.0).unwrap();
        assert_eq!(kernel.name(), "RationalQuadratic");

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.5, 2.5, 3.5];

        let sim = kernel.compute(&x, &y).unwrap();
        assert!(sim > 0.0 && sim <= 1.0);

        // Self-similarity
        let self_sim = kernel.compute(&x, &x).unwrap();
        assert!((self_sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rational_quadratic_rbf_limit() {
        // As alpha increases, RQ should approach RBF behavior
        let x = vec![0.0, 0.0];
        let y = vec![1.0, 0.0];

        let rq_small = RationalQuadraticKernel::new(1.0, 1.0).unwrap();
        let rq_large = RationalQuadraticKernel::new(1.0, 100.0).unwrap();

        let sim_small = rq_small.compute(&x, &y).unwrap();
        let sim_large = rq_large.compute(&x, &y).unwrap();

        // Large alpha should give smaller similarity (more like RBF)
        assert!(sim_large < sim_small);
    }

    #[test]
    fn test_rational_quadratic_invalid_parameters() {
        assert!(RationalQuadraticKernel::new(0.0, 2.0).is_err());
        assert!(RationalQuadraticKernel::new(-1.0, 2.0).is_err());
        assert!(RationalQuadraticKernel::new(1.0, 0.0).is_err());
        assert!(RationalQuadraticKernel::new(1.0, -1.0).is_err());
    }

    #[test]
    fn test_rational_quadratic_symmetry() {
        let kernel = RationalQuadraticKernel::new(1.0, 2.0).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let k_xy = kernel.compute(&x, &y).unwrap();
        let k_yx = kernel.compute(&y, &x).unwrap();
        assert!((k_xy - k_yx).abs() < 1e-10);
    }

    #[test]
    fn test_periodic_kernel_basic() {
        let kernel = PeriodicKernel::new(10.0, 1.0).unwrap();
        assert_eq!(kernel.name(), "Periodic");
        assert!((kernel.period() - 10.0).abs() < 1e-10);

        let x = vec![1.0];
        let y = vec![2.0];

        let sim = kernel.compute(&x, &y).unwrap();
        assert!(sim > 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_periodic_kernel_periodicity() {
        let period = 10.0;
        let kernel = PeriodicKernel::new(period, 1.0).unwrap();

        let x = vec![0.0];
        let y1 = vec![5.0];
        let y2 = vec![5.0 + period]; // One period later
        let y3 = vec![5.0 + 2.0 * period]; // Two periods later

        let sim1 = kernel.compute(&x, &y1).unwrap();
        let sim2 = kernel.compute(&x, &y2).unwrap();
        let sim3 = kernel.compute(&x, &y3).unwrap();

        // Similarity should be nearly identical at periodic intervals
        assert!((sim1 - sim2).abs() < 1e-8);
        assert!((sim1 - sim3).abs() < 1e-8);
    }

    #[test]
    fn test_periodic_kernel_exact_period() {
        let period = 24.0; // e.g., hours in a day
        let kernel = PeriodicKernel::new(period, 1.0).unwrap();

        // Use 1D data for testing periodicity (periodic kernel works best in 1D)
        let x = vec![1.0];
        let y = vec![1.0 + period]; // Exactly one period later

        // Points exactly one period apart should have very high similarity
        let sim = kernel.compute(&x, &y).unwrap();
        assert!(sim > 0.99, "Periodic similarity at exact period: {}", sim);
    }

    #[test]
    fn test_periodic_kernel_invalid_parameters() {
        assert!(PeriodicKernel::new(0.0, 1.0).is_err());
        assert!(PeriodicKernel::new(-1.0, 1.0).is_err());
        assert!(PeriodicKernel::new(10.0, 0.0).is_err());
        assert!(PeriodicKernel::new(10.0, -1.0).is_err());
    }

    #[test]
    fn test_periodic_kernel_symmetry() {
        let kernel = PeriodicKernel::new(10.0, 1.0).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let k_xy = kernel.compute(&x, &y).unwrap();
        let k_yx = kernel.compute(&y, &x).unwrap();
        assert!((k_xy - k_yx).abs() < 1e-10);
    }

    #[test]
    fn test_periodic_kernel_length_scale_effect() {
        let period = 10.0;
        let kernel_smooth = PeriodicKernel::new(period, 2.0).unwrap();
        let kernel_rough = PeriodicKernel::new(period, 0.5).unwrap();

        let x = vec![0.0];
        let y = vec![1.0]; // Small displacement

        let sim_smooth = kernel_smooth.compute(&x, &y).unwrap();
        let sim_rough = kernel_rough.compute(&x, &y).unwrap();

        // Larger length scale should give smoother transitions (higher similarity)
        assert!(sim_smooth > sim_rough);
    }

    #[test]
    fn test_advanced_kernels_dimension_mismatch() {
        let matern = MaternKernel::nu_3_2(1.0).unwrap();
        let rq = RationalQuadraticKernel::new(1.0, 2.0).unwrap();
        let periodic = PeriodicKernel::new(10.0, 1.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];

        assert!(matern.compute(&x, &y).is_err());
        assert!(rq.compute(&x, &y).is_err());
        assert!(periodic.compute(&x, &y).is_err());
    }

    // ============================================================================
    // Tests for kernel gradient methods
    // ============================================================================

    #[test]
    fn test_rbf_kernel_gradient_basic() {
        let config = RbfKernelConfig::new(0.5);
        let kernel = RbfKernel::new(config).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad) = kernel.compute_with_gradient(&x, &y).unwrap();

        // Verify kernel value matches standard compute
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);

        // Gradient should be negative (kernel decreases as gamma increases for distant points)
        assert!(grad < 0.0);
    }

    #[test]
    fn test_rbf_kernel_gradient_same_point() {
        let config = RbfKernelConfig::new(0.5);
        let kernel = RbfKernel::new(config).unwrap();

        let x = vec![1.0, 2.0, 3.0];

        let (k, grad) = kernel.compute_with_gradient(&x, &x).unwrap();

        // Same point: K = 1.0, gradient = 0 (no distance)
        assert!((k - 1.0).abs() < 1e-10);
        assert!(grad.abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_gradient_numerical_check() {
        let gamma = 0.5;
        let kernel = RbfKernel::new(RbfKernelConfig::new(gamma)).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad) = kernel.compute_with_gradient(&x, &y).unwrap();

        // Numerical gradient check
        let eps = 1e-6;
        let kernel_plus = RbfKernel::new(RbfKernelConfig::new(gamma + eps)).unwrap();
        let k_plus = kernel_plus.compute(&x, &y).unwrap();
        let numerical_grad = (k_plus - k) / eps;

        assert!(
            (grad - numerical_grad).abs() < 1e-4,
            "Analytical: {}, Numerical: {}",
            grad,
            numerical_grad
        );
    }

    #[test]
    fn test_rbf_kernel_length_scale_gradient() {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad_sigma) = kernel.compute_with_length_scale_gradient(&x, &y).unwrap();

        // Kernel value should match
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);

        // Gradient w.r.t. sigma should be positive (kernel increases as length scale increases)
        assert!(grad_sigma > 0.0);
    }

    #[test]
    fn test_polynomial_kernel_constant_gradient() {
        let kernel = PolynomialKernel::new(2, 1.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad_c) = kernel.compute_with_constant_gradient(&x, &y).unwrap();

        // Verify kernel value: (dot + c)^d = (1*3 + 2*4 + 1)^2 = (11 + 1)^2 = 144
        assert!((k - 144.0).abs() < 1e-10);

        // dK/dc = d * (dot + c)^(d-1) = 2 * 12 = 24
        assert!((grad_c - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_kernel_all_gradients() {
        let kernel = PolynomialKernel::new(3, 1.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![2.0, 3.0];

        let (k, grad_c, grad_d) = kernel.compute_with_all_gradients(&x, &y).unwrap();

        // dot = 1*2 + 2*3 = 8, base = 9
        // K = 9^3 = 729
        assert!((k - 729.0).abs() < 1e-8);

        // dK/dc = 3 * 9^2 = 243
        assert!((grad_c - 243.0).abs() < 1e-8);

        // dK/dd = K * ln(base) = 729 * ln(9)
        let expected_grad_d = 729.0 * 9.0_f64.ln();
        assert!(
            (grad_d - expected_grad_d).abs() < 1e-6,
            "Expected: {}, Got: {}",
            expected_grad_d,
            grad_d
        );
    }

    #[test]
    fn test_polynomial_kernel_gradient_numerical_check() {
        let c = 1.0;
        let kernel = PolynomialKernel::new(2, c).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad_c) = kernel.compute_with_constant_gradient(&x, &y).unwrap();

        // Numerical gradient check
        let eps = 1e-6;
        let kernel_plus = PolynomialKernel::new(2, c + eps).unwrap();
        let k_plus = kernel_plus.compute(&x, &y).unwrap();
        let numerical_grad = (k_plus - k) / eps;

        assert!(
            (grad_c - numerical_grad).abs() < 1e-3,
            "Analytical: {}, Numerical: {}",
            grad_c,
            numerical_grad
        );
    }

    #[test]
    fn test_laplacian_kernel_gradient() {
        let kernel = LaplacianKernel::new(0.5).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad) = kernel.compute_with_gradient(&x, &y).unwrap();

        // Verify kernel value matches
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);

        // Gradient should be negative
        assert!(grad < 0.0);
    }

    #[test]
    fn test_laplacian_kernel_gradient_same_point() {
        let kernel = LaplacianKernel::new(0.5).unwrap();

        let x = vec![1.0, 2.0, 3.0];

        let (k, grad) = kernel.compute_with_gradient(&x, &x).unwrap();

        // Same point: K = 1.0, gradient = 0
        assert!((k - 1.0).abs() < 1e-10);
        assert!(grad.abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_kernel_sigma_gradient() {
        let kernel = LaplacianKernel::new(0.5).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad_sigma) = kernel.compute_with_sigma_gradient(&x, &y).unwrap();

        // Kernel value should match
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);

        // Gradient w.r.t. sigma should be positive
        assert!(grad_sigma > 0.0);
    }

    #[test]
    fn test_matern_kernel_gradient_nu_half() {
        let kernel = MaternKernel::exponential(1.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad_l) = kernel.compute_with_length_scale_gradient(&x, &y).unwrap();

        // Verify kernel value matches
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10, "K mismatch: {} vs {}", k, k_std);

        // Gradient w.r.t. length scale should be positive (kernel increases as l increases)
        assert!(grad_l > 0.0);
    }

    #[test]
    fn test_matern_kernel_gradient_nu_3_2() {
        let kernel = MaternKernel::nu_3_2(1.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad_l) = kernel.compute_with_length_scale_gradient(&x, &y).unwrap();

        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10, "K mismatch: {} vs {}", k, k_std);

        assert!(grad_l > 0.0, "Gradient should be positive, got: {}", grad_l);
    }

    #[test]
    fn test_matern_kernel_gradient_nu_5_2() {
        let kernel = MaternKernel::nu_5_2(1.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad_l) = kernel.compute_with_length_scale_gradient(&x, &y).unwrap();

        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10, "K mismatch: {} vs {}", k, k_std);

        assert!(grad_l > 0.0);
    }

    #[test]
    fn test_matern_kernel_gradient_same_point() {
        let kernel = MaternKernel::nu_3_2(1.0).unwrap();

        let x = vec![1.0, 2.0, 3.0];

        let (k, grad_l) = kernel.compute_with_length_scale_gradient(&x, &x).unwrap();

        // Same point: K = 1.0, gradient = 0
        assert!((k - 1.0).abs() < 1e-10);
        assert!(grad_l.abs() < 1e-10);
    }

    #[test]
    fn test_rational_quadratic_kernel_length_scale_gradient() {
        let kernel = RationalQuadraticKernel::new(1.0, 2.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad_l) = kernel.compute_with_length_scale_gradient(&x, &y).unwrap();

        // Verify kernel value matches
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);

        // Gradient w.r.t. length scale should be positive
        assert!(grad_l > 0.0);
    }

    #[test]
    fn test_rational_quadratic_kernel_alpha_gradient() {
        let kernel = RationalQuadraticKernel::new(1.0, 2.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad_alpha) = kernel.compute_with_alpha_gradient(&x, &y).unwrap();

        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);

        // Gradient w.r.t. alpha: should be negative (kernel decreases as alpha increases)
        assert!(grad_alpha < 0.0);
    }

    #[test]
    fn test_rational_quadratic_kernel_all_gradients() {
        let kernel = RationalQuadraticKernel::new(1.0, 2.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let (k, grad_l, grad_alpha) = kernel.compute_with_all_gradients(&x, &y).unwrap();

        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);

        // Verify consistency with individual gradient methods
        let (_, grad_l_single) = kernel.compute_with_length_scale_gradient(&x, &y).unwrap();
        let (_, grad_alpha_single) = kernel.compute_with_alpha_gradient(&x, &y).unwrap();

        assert!((grad_l - grad_l_single).abs() < 1e-10);
        assert!((grad_alpha - grad_alpha_single).abs() < 1e-10);
    }

    #[test]
    fn test_rational_quadratic_kernel_gradient_same_point() {
        let kernel = RationalQuadraticKernel::new(1.0, 2.0).unwrap();

        let x = vec![1.0, 2.0, 3.0];

        let (k, grad_l, grad_alpha) = kernel.compute_with_all_gradients(&x, &x).unwrap();

        // Same point: K = 1.0, gradients = 0
        assert!((k - 1.0).abs() < 1e-10);
        assert!(grad_l.abs() < 1e-10);
        assert!(grad_alpha.abs() < 1e-10);
    }

    #[test]
    fn test_kernel_gradient_dimension_mismatch() {
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let poly = PolynomialKernel::new(2, 1.0).unwrap();
        let lap = LaplacianKernel::new(0.5).unwrap();
        let matern = MaternKernel::nu_3_2(1.0).unwrap();
        let rq = RationalQuadraticKernel::new(1.0, 2.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];

        assert!(rbf.compute_with_gradient(&x, &y).is_err());
        assert!(poly.compute_with_constant_gradient(&x, &y).is_err());
        assert!(lap.compute_with_gradient(&x, &y).is_err());
        assert!(matern.compute_with_length_scale_gradient(&x, &y).is_err());
        assert!(rq.compute_with_length_scale_gradient(&x, &y).is_err());
    }
}
