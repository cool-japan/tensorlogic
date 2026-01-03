//! Automatic Relevance Determination (ARD) kernels.
//!
//! ARD kernels learn a separate length scale for each input dimension,
//! automatically determining the relevance of each feature. This is
//! particularly useful for high-dimensional data where some features
//! may be more important than others.
//!
//! ## Key Features
//!
//! - **Per-dimension length scales**: Each feature has its own length scale
//! - **Feature selection**: Irrelevant features get large length scales (effectively ignored)
//! - **Gradient support**: For hyperparameter optimization via gradient descent
//!
//! ## Example
//!
//! ```rust
//! use tensorlogic_sklears_kernels::ard_kernel::{ArdRbfKernel, ArdMaternKernel};
//! use tensorlogic_sklears_kernels::Kernel;
//!
//! // Create ARD RBF kernel with 3 features, each with its own length scale
//! let length_scales = vec![1.0, 2.0, 0.5]; // Different relevance per dimension
//! let kernel = ArdRbfKernel::new(length_scales.clone()).unwrap();
//!
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![1.5, 2.5, 3.5];
//! let sim = kernel.compute(&x, &y).unwrap();
//! ```

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// ARD (Automatic Relevance Determination) RBF kernel.
///
/// K(x, y) = σ² * exp(-0.5 * Σ((x_i - y_i)² / l_i²))
///
/// Each dimension has its own length scale `l_i`, allowing the kernel
/// to automatically weight features by their relevance.
#[derive(Debug, Clone)]
pub struct ArdRbfKernel {
    /// Per-dimension length scales
    length_scales: Vec<f64>,
    /// Signal variance (output scale)
    variance: f64,
}

impl ArdRbfKernel {
    /// Create a new ARD RBF kernel with per-dimension length scales.
    ///
    /// # Arguments
    /// * `length_scales` - Length scale for each dimension (all must be positive)
    ///
    /// # Example
    /// ```rust
    /// use tensorlogic_sklears_kernels::ard_kernel::ArdRbfKernel;
    ///
    /// let kernel = ArdRbfKernel::new(vec![1.0, 2.0, 0.5]).unwrap();
    /// ```
    pub fn new(length_scales: Vec<f64>) -> Result<Self> {
        Self::with_variance(length_scales, 1.0)
    }

    /// Create ARD RBF kernel with custom signal variance.
    ///
    /// # Arguments
    /// * `length_scales` - Per-dimension length scales
    /// * `variance` - Signal variance (output scale, must be positive)
    pub fn with_variance(length_scales: Vec<f64>, variance: f64) -> Result<Self> {
        if length_scales.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "length_scales".to_string(),
                value: "[]".to_string(),
                reason: "must have at least one dimension".to_string(),
            });
        }

        for (i, &ls) in length_scales.iter().enumerate() {
            if ls <= 0.0 {
                return Err(KernelError::InvalidParameter {
                    parameter: format!("length_scales[{}]", i),
                    value: ls.to_string(),
                    reason: "all length scales must be positive".to_string(),
                });
            }
        }

        if variance <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "variance".to_string(),
                value: variance.to_string(),
                reason: "variance must be positive".to_string(),
            });
        }

        Ok(Self {
            length_scales,
            variance,
        })
    }

    /// Get the length scales.
    pub fn length_scales(&self) -> &[f64] {
        &self.length_scales
    }

    /// Get the signal variance.
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.length_scales.len()
    }

    /// Compute the kernel gradient with respect to hyperparameters.
    ///
    /// Returns gradients for:
    /// 1. Each length scale (one per dimension)
    /// 2. The signal variance
    ///
    /// This is useful for hyperparameter optimization via gradient descent.
    pub fn compute_gradient(&self, x: &[f64], y: &[f64]) -> Result<KernelGradient> {
        if x.len() != self.length_scales.len() || y.len() != self.length_scales.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.length_scales.len()],
                got: vec![x.len(), y.len()],
                context: "ARD RBF kernel gradient".to_string(),
            });
        }

        // Compute scaled squared differences
        let mut sum_scaled_sq = 0.0;
        let mut scaled_sq_diffs = Vec::with_capacity(self.length_scales.len());

        for i in 0..self.length_scales.len() {
            let diff = x[i] - y[i];
            let ls = self.length_scales[i];
            let scaled_sq = diff * diff / (ls * ls);
            scaled_sq_diffs.push(scaled_sq);
            sum_scaled_sq += scaled_sq;
        }

        let exp_term = (-0.5 * sum_scaled_sq).exp();
        let k_value = self.variance * exp_term;

        // Gradient w.r.t. each length scale: dk/dl_i = k * (x_i - y_i)² / l_i³
        let grad_length_scales: Vec<f64> = scaled_sq_diffs
            .iter()
            .enumerate()
            .map(|(i, &sq_diff)| {
                let ls = self.length_scales[i];
                k_value * sq_diff / ls
            })
            .collect();

        // Gradient w.r.t. variance: dk/dσ² = exp_term
        let grad_variance = exp_term;

        Ok(KernelGradient {
            value: k_value,
            grad_length_scales,
            grad_variance,
        })
    }
}

impl Kernel for ArdRbfKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != self.length_scales.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.length_scales.len()],
                got: vec![x.len()],
                context: "ARD RBF kernel".to_string(),
            });
        }
        if y.len() != self.length_scales.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.length_scales.len()],
                got: vec![y.len()],
                context: "ARD RBF kernel".to_string(),
            });
        }

        let mut sum_scaled_sq = 0.0;
        for i in 0..self.length_scales.len() {
            let diff = x[i] - y[i];
            let ls = self.length_scales[i];
            sum_scaled_sq += (diff * diff) / (ls * ls);
        }

        Ok(self.variance * (-0.5 * sum_scaled_sq).exp())
    }

    fn name(&self) -> &str {
        "ARD-RBF"
    }
}

/// ARD Matérn kernel with per-dimension length scales.
///
/// Supports Matérn with nu = 0.5 (exponential), 1.5, and 2.5.
#[derive(Debug, Clone)]
pub struct ArdMaternKernel {
    /// Per-dimension length scales
    length_scales: Vec<f64>,
    /// Signal variance (output scale)
    variance: f64,
    /// Smoothness parameter (0.5, 1.5, or 2.5)
    nu: f64,
}

impl ArdMaternKernel {
    /// Create a new ARD Matérn kernel.
    ///
    /// # Arguments
    /// * `length_scales` - Per-dimension length scales
    /// * `nu` - Smoothness parameter (must be 0.5, 1.5, or 2.5)
    pub fn new(length_scales: Vec<f64>, nu: f64) -> Result<Self> {
        Self::with_variance(length_scales, nu, 1.0)
    }

    /// Create ARD Matérn kernel with custom variance.
    pub fn with_variance(length_scales: Vec<f64>, nu: f64, variance: f64) -> Result<Self> {
        if length_scales.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "length_scales".to_string(),
                value: "[]".to_string(),
                reason: "must have at least one dimension".to_string(),
            });
        }

        for (i, &ls) in length_scales.iter().enumerate() {
            if ls <= 0.0 {
                return Err(KernelError::InvalidParameter {
                    parameter: format!("length_scales[{}]", i),
                    value: ls.to_string(),
                    reason: "all length scales must be positive".to_string(),
                });
            }
        }

        if !((nu - 0.5).abs() < 1e-10 || (nu - 1.5).abs() < 1e-10 || (nu - 2.5).abs() < 1e-10) {
            return Err(KernelError::InvalidParameter {
                parameter: "nu".to_string(),
                value: nu.to_string(),
                reason: "nu must be 0.5, 1.5, or 2.5".to_string(),
            });
        }

        if variance <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "variance".to_string(),
                value: variance.to_string(),
                reason: "variance must be positive".to_string(),
            });
        }

        Ok(Self {
            length_scales,
            variance,
            nu,
        })
    }

    /// Create ARD Matérn 1/2 kernel (exponential).
    pub fn exponential(length_scales: Vec<f64>) -> Result<Self> {
        Self::new(length_scales, 0.5)
    }

    /// Create ARD Matérn 3/2 kernel.
    pub fn nu_3_2(length_scales: Vec<f64>) -> Result<Self> {
        Self::new(length_scales, 1.5)
    }

    /// Create ARD Matérn 5/2 kernel.
    pub fn nu_5_2(length_scales: Vec<f64>) -> Result<Self> {
        Self::new(length_scales, 2.5)
    }

    /// Get the length scales.
    pub fn length_scales(&self) -> &[f64] {
        &self.length_scales
    }

    /// Get the signal variance.
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Get the smoothness parameter nu.
    pub fn nu(&self) -> f64 {
        self.nu
    }

    /// Compute scaled Euclidean distance using ARD length scales.
    fn scaled_distance(&self, x: &[f64], y: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.length_scales.len() {
            let diff = x[i] - y[i];
            let ls = self.length_scales[i];
            sum += (diff * diff) / (ls * ls);
        }
        sum.sqrt()
    }
}

impl Kernel for ArdMaternKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != self.length_scales.len() || y.len() != self.length_scales.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.length_scales.len()],
                got: vec![x.len(), y.len()],
                context: "ARD Matérn kernel".to_string(),
            });
        }

        let r = self.scaled_distance(x, y);

        // Handle same point
        if r < 1e-10 {
            return Ok(self.variance);
        }

        let sqrt_2nu = (2.0 * self.nu).sqrt();
        let scaled_r = sqrt_2nu * r;

        let result = if (self.nu - 0.5).abs() < 1e-10 {
            // Matérn 1/2 (exponential)
            (-scaled_r).exp()
        } else if (self.nu - 1.5).abs() < 1e-10 {
            // Matérn 3/2
            (1.0 + scaled_r) * (-scaled_r).exp()
        } else {
            // Matérn 5/2
            (1.0 + scaled_r + scaled_r * scaled_r / 3.0) * (-scaled_r).exp()
        };

        Ok(self.variance * result)
    }

    fn name(&self) -> &str {
        "ARD-Matérn"
    }
}

/// ARD Rational Quadratic kernel.
///
/// K(x, y) = σ² * (1 + Σ((x_i - y_i)² / (2 * α * l_i²)))^(-α)
#[derive(Debug, Clone)]
pub struct ArdRationalQuadraticKernel {
    /// Per-dimension length scales
    length_scales: Vec<f64>,
    /// Signal variance
    variance: f64,
    /// Scale mixture parameter
    alpha: f64,
}

impl ArdRationalQuadraticKernel {
    /// Create a new ARD Rational Quadratic kernel.
    pub fn new(length_scales: Vec<f64>, alpha: f64) -> Result<Self> {
        Self::with_variance(length_scales, alpha, 1.0)
    }

    /// Create with custom variance.
    pub fn with_variance(length_scales: Vec<f64>, alpha: f64, variance: f64) -> Result<Self> {
        if length_scales.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "length_scales".to_string(),
                value: "[]".to_string(),
                reason: "must have at least one dimension".to_string(),
            });
        }

        for (i, &ls) in length_scales.iter().enumerate() {
            if ls <= 0.0 {
                return Err(KernelError::InvalidParameter {
                    parameter: format!("length_scales[{}]", i),
                    value: ls.to_string(),
                    reason: "all length scales must be positive".to_string(),
                });
            }
        }

        if alpha <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "alpha".to_string(),
                value: alpha.to_string(),
                reason: "alpha must be positive".to_string(),
            });
        }

        if variance <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "variance".to_string(),
                value: variance.to_string(),
                reason: "variance must be positive".to_string(),
            });
        }

        Ok(Self {
            length_scales,
            variance,
            alpha,
        })
    }

    /// Get the length scales.
    pub fn length_scales(&self) -> &[f64] {
        &self.length_scales
    }

    /// Get the variance.
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Get the alpha parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl Kernel for ArdRationalQuadraticKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != self.length_scales.len() || y.len() != self.length_scales.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.length_scales.len()],
                got: vec![x.len(), y.len()],
                context: "ARD Rational Quadratic kernel".to_string(),
            });
        }

        let mut sum_scaled_sq = 0.0;
        for i in 0..self.length_scales.len() {
            let diff = x[i] - y[i];
            let ls = self.length_scales[i];
            sum_scaled_sq += (diff * diff) / (ls * ls);
        }

        let term = 1.0 + sum_scaled_sq / (2.0 * self.alpha);
        Ok(self.variance * term.powf(-self.alpha))
    }

    fn name(&self) -> &str {
        "ARD-RationalQuadratic"
    }
}

/// Gradient information for kernel hyperparameter optimization.
#[derive(Debug, Clone)]
pub struct KernelGradient {
    /// The kernel value K(x, y)
    pub value: f64,
    /// Gradient with respect to each length scale
    pub grad_length_scales: Vec<f64>,
    /// Gradient with respect to the signal variance
    pub grad_variance: f64,
}

/// Utility kernel: White Noise kernel for observation noise modeling.
///
/// K(x, y) = σ² if x == y, else 0
///
/// Used to model i.i.d. observation noise in Gaussian Processes.
#[derive(Debug, Clone)]
pub struct WhiteNoiseKernel {
    /// Noise variance
    variance: f64,
}

impl WhiteNoiseKernel {
    /// Create a new white noise kernel.
    ///
    /// # Arguments
    /// * `variance` - Noise variance (must be positive)
    pub fn new(variance: f64) -> Result<Self> {
        if variance <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "variance".to_string(),
                value: variance.to_string(),
                reason: "variance must be positive".to_string(),
            });
        }
        Ok(Self { variance })
    }

    /// Get the noise variance.
    pub fn variance(&self) -> f64 {
        self.variance
    }
}

impl Kernel for WhiteNoiseKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "White Noise kernel".to_string(),
            });
        }

        // Check if x and y are the same point (within tolerance)
        let is_same = x.iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < 1e-10);

        if is_same {
            Ok(self.variance)
        } else {
            Ok(0.0)
        }
    }

    fn name(&self) -> &str {
        "WhiteNoise"
    }
}

/// Constant kernel: K(x, y) = σ²
///
/// Produces constant predictions. Useful as a building block in composite kernels.
#[derive(Debug, Clone)]
pub struct ConstantKernel {
    /// Constant value (variance)
    variance: f64,
}

impl ConstantKernel {
    /// Create a new constant kernel.
    pub fn new(variance: f64) -> Result<Self> {
        if variance <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "variance".to_string(),
                value: variance.to_string(),
                reason: "variance must be positive".to_string(),
            });
        }
        Ok(Self { variance })
    }

    /// Get the variance.
    pub fn variance(&self) -> f64 {
        self.variance
    }
}

impl Kernel for ConstantKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Constant kernel".to_string(),
            });
        }
        Ok(self.variance)
    }

    fn name(&self) -> &str {
        "Constant"
    }
}

/// Dot Product kernel (Linear kernel with variance and shift).
///
/// K(x, y) = σ² + σ_b² + x · y
///
/// Useful for Bayesian linear regression. The offset parameter allows
/// the linear model to have a non-zero mean.
#[derive(Debug, Clone)]
pub struct DotProductKernel {
    /// Signal variance (scaling)
    variance: f64,
    /// Offset variance (bias)
    variance_bias: f64,
}

impl DotProductKernel {
    /// Create a new dot product kernel.
    pub fn new(variance: f64, variance_bias: f64) -> Result<Self> {
        if variance < 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "variance".to_string(),
                value: variance.to_string(),
                reason: "variance must be non-negative".to_string(),
            });
        }
        if variance_bias < 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "variance_bias".to_string(),
                value: variance_bias.to_string(),
                reason: "variance_bias must be non-negative".to_string(),
            });
        }
        Ok(Self {
            variance,
            variance_bias,
        })
    }

    /// Create a simple dot product kernel (variance=1, no bias).
    pub fn simple() -> Self {
        Self {
            variance: 1.0,
            variance_bias: 0.0,
        }
    }

    /// Get the variance.
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Get the bias variance.
    pub fn variance_bias(&self) -> f64 {
        self.variance_bias
    }
}

impl Kernel for DotProductKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Dot Product kernel".to_string(),
            });
        }

        let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        Ok(self.variance_bias + self.variance * dot)
    }

    fn name(&self) -> &str {
        "DotProduct"
    }
}

/// Scaled kernel wrapper that multiplies a kernel by a variance parameter.
///
/// K_scaled(x, y) = σ² * K(x, y)
///
/// This is useful for controlling the output scale of any kernel.
#[derive(Debug, Clone)]
pub struct ScaledKernel<K: Kernel> {
    /// The base kernel
    kernel: K,
    /// The scaling factor (variance)
    variance: f64,
}

impl<K: Kernel> ScaledKernel<K> {
    /// Create a scaled kernel.
    pub fn new(kernel: K, variance: f64) -> Result<Self> {
        if variance <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "variance".to_string(),
                value: variance.to_string(),
                reason: "variance must be positive".to_string(),
            });
        }
        Ok(Self { kernel, variance })
    }

    /// Get the base kernel.
    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    /// Get the variance.
    pub fn variance(&self) -> f64 {
        self.variance
    }
}

impl<K: Kernel> Kernel for ScaledKernel<K> {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let base_value = self.kernel.compute(x, y)?;
        Ok(self.variance * base_value)
    }

    fn name(&self) -> &str {
        "Scaled"
    }

    fn is_psd(&self) -> bool {
        self.kernel.is_psd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== ARD RBF Kernel Tests =====

    #[test]
    fn test_ard_rbf_kernel_basic() {
        let kernel = ArdRbfKernel::new(vec![1.0, 1.0, 1.0]).unwrap();
        assert_eq!(kernel.name(), "ARD-RBF");
        assert_eq!(kernel.ndim(), 3);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        // Self-similarity should be variance (1.0)
        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ard_rbf_kernel_different_length_scales() {
        // High length scale in dimension 0 makes that dimension less important
        let kernel = ArdRbfKernel::new(vec![10.0, 1.0, 1.0]).unwrap();

        let x = vec![0.0, 0.0, 0.0];
        let y1 = vec![1.0, 0.0, 0.0]; // Difference in dim 0 (large length scale)
        let y2 = vec![0.0, 1.0, 0.0]; // Difference in dim 1 (small length scale)

        let sim1 = kernel.compute(&x, &y1).unwrap();
        let sim2 = kernel.compute(&x, &y2).unwrap();

        // y1 should be MORE similar because dim 0 has large length scale (less relevant)
        assert!(sim1 > sim2);
    }

    #[test]
    fn test_ard_rbf_kernel_with_variance() {
        let kernel = ArdRbfKernel::with_variance(vec![1.0, 1.0], 2.0).unwrap();
        assert!((kernel.variance() - 2.0).abs() < 1e-10);

        let x = vec![0.0, 0.0];
        let sim = kernel.compute(&x, &x).unwrap();
        assert!((sim - 2.0).abs() < 1e-10); // Self-similarity = variance
    }

    #[test]
    fn test_ard_rbf_kernel_gradient() {
        let kernel = ArdRbfKernel::new(vec![1.0, 2.0]).unwrap();
        let x = vec![0.0, 0.0];
        let y = vec![1.0, 1.0];

        let grad = kernel.compute_gradient(&x, &y).unwrap();

        // Check that value matches compute
        let value = kernel.compute(&x, &y).unwrap();
        assert!((grad.value - value).abs() < 1e-10);

        // Gradients should have correct dimensions
        assert_eq!(grad.grad_length_scales.len(), 2);
    }

    #[test]
    fn test_ard_rbf_kernel_invalid_empty() {
        let result = ArdRbfKernel::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ard_rbf_kernel_invalid_negative() {
        let result = ArdRbfKernel::new(vec![1.0, -1.0, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ard_rbf_kernel_invalid_variance() {
        let result = ArdRbfKernel::with_variance(vec![1.0], 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ard_rbf_kernel_dimension_mismatch() {
        let kernel = ArdRbfKernel::new(vec![1.0, 1.0]).unwrap();
        let x = vec![1.0, 2.0, 3.0]; // 3 dims
        let y = vec![1.0, 2.0]; // 2 dims

        assert!(kernel.compute(&x, &y).is_err());
    }

    #[test]
    fn test_ard_rbf_kernel_symmetry() {
        let kernel = ArdRbfKernel::new(vec![1.0, 2.0, 0.5]).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let k_xy = kernel.compute(&x, &y).unwrap();
        let k_yx = kernel.compute(&y, &x).unwrap();
        assert!((k_xy - k_yx).abs() < 1e-10);
    }

    // ===== ARD Matérn Kernel Tests =====

    #[test]
    fn test_ard_matern_kernel_nu_3_2() {
        let kernel = ArdMaternKernel::nu_3_2(vec![1.0, 1.0]).unwrap();
        assert_eq!(kernel.name(), "ARD-Matérn");
        assert!((kernel.nu() - 1.5).abs() < 1e-10);

        let x = vec![0.0, 0.0];
        let sim = kernel.compute(&x, &x).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ard_matern_kernel_nu_5_2() {
        let kernel = ArdMaternKernel::nu_5_2(vec![1.0, 2.0]).unwrap();
        assert!((kernel.nu() - 2.5).abs() < 1e-10);

        let x = vec![0.0, 0.0];
        let y = vec![0.5, 0.5];
        let sim = kernel.compute(&x, &y).unwrap();
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_ard_matern_kernel_exponential() {
        let kernel = ArdMaternKernel::exponential(vec![1.0]).unwrap();
        assert!((kernel.nu() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ard_matern_kernel_invalid_nu() {
        // Only 0.5, 1.5, 2.5 are supported
        let result = ArdMaternKernel::new(vec![1.0], 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ard_matern_kernel_different_length_scales() {
        let kernel = ArdMaternKernel::nu_3_2(vec![10.0, 1.0]).unwrap();

        let x = vec![0.0, 0.0];
        let y1 = vec![1.0, 0.0];
        let y2 = vec![0.0, 1.0];

        let sim1 = kernel.compute(&x, &y1).unwrap();
        let sim2 = kernel.compute(&x, &y2).unwrap();

        // Larger length scale in dim 0 makes it less relevant
        assert!(sim1 > sim2);
    }

    // ===== ARD Rational Quadratic Kernel Tests =====

    #[test]
    fn test_ard_rq_kernel_basic() {
        let kernel = ArdRationalQuadraticKernel::new(vec![1.0, 1.0], 2.0).unwrap();
        assert_eq!(kernel.name(), "ARD-RationalQuadratic");

        let x = vec![0.0, 0.0];
        let sim = kernel.compute(&x, &x).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ard_rq_kernel_with_variance() {
        let kernel = ArdRationalQuadraticKernel::with_variance(vec![1.0], 2.0, 3.0).unwrap();
        assert!((kernel.variance() - 3.0).abs() < 1e-10);

        let x = vec![0.0];
        let sim = kernel.compute(&x, &x).unwrap();
        assert!((sim - 3.0).abs() < 1e-10);
    }

    // ===== White Noise Kernel Tests =====

    #[test]
    fn test_white_noise_kernel_same_point() {
        let kernel = WhiteNoiseKernel::new(0.1).unwrap();
        assert_eq!(kernel.name(), "WhiteNoise");

        let x = vec![1.0, 2.0, 3.0];
        let sim = kernel.compute(&x, &x).unwrap();
        assert!((sim - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_white_noise_kernel_different_points() {
        let kernel = WhiteNoiseKernel::new(0.1).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.1]; // Slightly different
        let sim = kernel.compute(&x, &y).unwrap();
        assert!(sim.abs() < 1e-10); // Should be 0
    }

    #[test]
    fn test_white_noise_kernel_invalid() {
        let result = WhiteNoiseKernel::new(0.0);
        assert!(result.is_err());

        let result = WhiteNoiseKernel::new(-1.0);
        assert!(result.is_err());
    }

    // ===== Constant Kernel Tests =====

    #[test]
    fn test_constant_kernel() {
        let kernel = ConstantKernel::new(2.5).unwrap();
        assert_eq!(kernel.name(), "Constant");

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_constant_kernel_invalid() {
        assert!(ConstantKernel::new(0.0).is_err());
        assert!(ConstantKernel::new(-1.0).is_err());
    }

    // ===== Dot Product Kernel Tests =====

    #[test]
    fn test_dot_product_kernel_simple() {
        let kernel = DotProductKernel::simple();
        assert_eq!(kernel.name(), "DotProduct");

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        // dot = 1*4 + 2*5 + 3*6 = 32
        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_kernel_with_bias() {
        let kernel = DotProductKernel::new(1.0, 5.0).unwrap();

        let x = vec![1.0, 0.0];
        let y = vec![0.0, 1.0]; // Orthogonal

        // dot = 0, result = bias + variance * dot = 5 + 0 = 5
        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_kernel_with_variance() {
        let kernel = DotProductKernel::new(2.0, 0.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        // dot = 11, result = 2 * 11 = 22
        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 22.0).abs() < 1e-10);
    }

    // ===== Scaled Kernel Tests =====

    #[test]
    fn test_scaled_kernel() {
        use crate::tensor_kernels::LinearKernel;

        let base = LinearKernel::new();
        let scaled = ScaledKernel::new(base, 2.0).unwrap();
        assert_eq!(scaled.name(), "Scaled");

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        // Linear: dot = 32, scaled: 2 * 32 = 64
        let sim = scaled.compute(&x, &y).unwrap();
        assert!((sim - 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaled_kernel_invalid() {
        use crate::tensor_kernels::LinearKernel;

        let base = LinearKernel::new();
        let result = ScaledKernel::new(base, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_kernel_psd() {
        use crate::tensor_kernels::LinearKernel;

        let base = LinearKernel::new();
        let scaled = ScaledKernel::new(base, 2.0).unwrap();
        assert!(scaled.is_psd());
    }

    // ===== Integration Tests =====

    #[test]
    fn test_ard_kernels_symmetry() {
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(ArdRbfKernel::new(vec![1.0, 2.0]).unwrap()),
            Box::new(ArdMaternKernel::nu_3_2(vec![1.0, 2.0]).unwrap()),
            Box::new(ArdRationalQuadraticKernel::new(vec![1.0, 2.0], 2.0).unwrap()),
        ];

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        for kernel in kernels {
            let k_xy = kernel.compute(&x, &y).unwrap();
            let k_yx = kernel.compute(&y, &x).unwrap();
            assert!(
                (k_xy - k_yx).abs() < 1e-10,
                "{} not symmetric",
                kernel.name()
            );
        }
    }

    #[test]
    fn test_utility_kernels_symmetry() {
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(WhiteNoiseKernel::new(0.1).unwrap()),
            Box::new(ConstantKernel::new(1.0).unwrap()),
            Box::new(DotProductKernel::simple()),
        ];

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        for kernel in kernels {
            let k_xy = kernel.compute(&x, &y).unwrap();
            let k_yx = kernel.compute(&y, &x).unwrap();
            assert!(
                (k_xy - k_yx).abs() < 1e-10,
                "{} not symmetric",
                kernel.name()
            );
        }
    }
}
