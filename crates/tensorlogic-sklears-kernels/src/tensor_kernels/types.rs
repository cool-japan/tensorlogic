//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::RbfKernelConfig;

/// Rational Quadratic kernel: K(x, y) = (1 + ||x-y||² / (2 * alpha * l²))^(-alpha)
///
/// Can be seen as a scale mixture of RBF kernels with different length scales.
/// As alpha → ∞, this kernel becomes equivalent to the RBF kernel.
/// Useful when data exhibits multiple characteristic length scales.
#[derive(Debug, Clone)]
pub struct RationalQuadraticKernel {
    /// Length scale parameter
    pub(super) length_scale: f64,
    /// Scale mixture parameter (alpha)
    /// Controls the relative weighting of large vs small scale variations
    pub(super) alpha: f64,
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
    pub(super) fn squared_distance(x: &[f64], y: &[f64]) -> f64 {
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
        let denom = self.length_scale * (2.0 * self.alpha * l_sq + sq_dist);
        let grad_l = if denom.abs() > 1e-10 {
            k * sq_dist / denom
        } else {
            0.0
        };
        let grad_alpha = k * (u / term - term.ln());
        Ok((k, grad_l, grad_alpha))
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
    pub(super) length_scale: f64,
    /// Smoothness parameter (nu)
    /// Common values: 0.5 (exponential), 1.5, 2.5
    pub(super) nu: f64,
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
    pub(super) fn euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
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
        if dist < 1e-10 {
            return Ok((1.0, 0.0));
        }
        let l = self.length_scale;
        let sqrt_2nu = (2.0 * self.nu).sqrt();
        let scaled_dist = sqrt_2nu * dist / l;
        if (self.nu - 0.5).abs() < 1e-10 {
            let k = (-scaled_dist).exp();
            let grad_l = (dist / (l * l)) * k;
            Ok((k, grad_l))
        } else if (self.nu - 1.5).abs() < 1e-10 {
            let sqrt3 = 3.0_f64.sqrt();
            let z = sqrt3 * dist / l;
            let exp_neg_z = (-z).exp();
            let k = (1.0 + z) * exp_neg_z;
            let grad_l = (z * z / l) * exp_neg_z;
            Ok((k, grad_l))
        } else if (self.nu - 2.5).abs() < 1e-10 {
            let sqrt5 = 5.0_f64.sqrt();
            let z = sqrt5 * dist / l;
            let exp_neg_z = (-z).exp();
            let k = (1.0 + z + z * z / 3.0) * exp_neg_z;
            let grad_l = (z * z * (1.0 + z) / (3.0 * l)) * exp_neg_z;
            Ok((k, grad_l))
        } else {
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
    pub(super) fn dot_product(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
    /// Compute L2 norm
    pub(super) fn norm(x: &[f64]) -> f64 {
        x.iter().map(|a| a * a).sum::<f64>().sqrt()
    }
}
/// Laplacian kernel: K(x, y) = exp(-gamma * ||x - y||_1)
///
/// Similar to RBF but uses L1 (Manhattan) distance instead of L2.
/// More robust to outliers than RBF kernel.
#[derive(Debug, Clone)]
pub struct LaplacianKernel {
    /// Bandwidth parameter (gamma)
    pub(super) gamma: f64,
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
    pub(super) fn l1_distance(x: &[f64], y: &[f64]) -> f64 {
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
        let grad_sigma = l1_dist * self.gamma * self.gamma * k;
        Ok((k, grad_sigma))
    }
}
/// RBF (Radial Basis Function) kernel: K(x, y) = exp(-gamma * ||x - y||^2)
///
/// Also known as Gaussian kernel. Maps to infinite-dimensional space.
#[derive(Clone)]
pub struct RbfKernel {
    /// Configuration
    pub(super) config: RbfKernelConfig,
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
    pub(super) fn squared_distance(x: &[f64], y: &[f64]) -> f64 {
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
        let sigma = 1.0 / (2.0 * self.config.gamma).sqrt();
        let grad_sigma = sq_dist / (sigma * sigma * sigma) * k;
        Ok((k, grad_sigma))
    }
}
/// Sigmoid (Tanh) kernel: K(x, y) = tanh(alpha * (x · y) + c)
///
/// Neural network inspired kernel. Note: Not guaranteed to be positive semi-definite
/// for all parameter values, but can be useful in practice.
#[derive(Debug, Clone)]
pub struct SigmoidKernel {
    /// Scale parameter (alpha)
    pub(super) alpha: f64,
    /// Offset parameter (c)
    pub(super) offset: f64,
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
    pub(super) fn dot_product(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
}
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
    pub(super) fn dot_product(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
}
/// Polynomial kernel: K(x, y) = (x · y + c)^d
///
/// Captures polynomial relationships up to degree d.
#[derive(Clone)]
pub struct PolynomialKernel {
    /// Polynomial degree
    pub(super) degree: u32,
    /// Constant term
    pub(super) constant: f64,
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
    pub(super) fn dot_product(x: &[f64], y: &[f64]) -> f64 {
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
            return Ok((0.0, 0.0));
        }
        let k = base.powi(self.degree as i32);
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
        let grad_c = (self.degree as f64) * base.powi(self.degree as i32 - 1);
        let grad_d = if base > 0.0 {
            k * base.ln()
        } else if base < 0.0 && self.degree.is_multiple_of(2) {
            k * base.abs().ln()
        } else {
            f64::NAN
        };
        Ok((k, grad_c, grad_d))
    }
}
/// Periodic kernel: K(x, y) = exp(-2 * sin²(π * ||x-y|| / period) / l²)
///
/// Captures periodic patterns in data. Useful for modeling seasonal effects,
/// oscillatory behavior, and other repeating patterns.
#[derive(Debug, Clone)]
pub struct PeriodicKernel {
    /// Period of the periodic pattern
    pub(super) period: f64,
    /// Length scale parameter
    pub(super) length_scale: f64,
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
    pub(super) fn euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}
/// Chi-squared kernel: K(x, y) = exp(-gamma * Σ((x_i - y_i)² / (x_i + y_i)))
///
/// Especially effective for histogram data and computer vision applications.
/// Handles non-negative features naturally.
#[derive(Debug, Clone)]
pub struct ChiSquaredKernel {
    /// Bandwidth parameter (gamma)
    pub(super) gamma: f64,
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
    pub(super) fn chi_squared_distance(x: &[f64], y: &[f64]) -> f64 {
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
    pub(super) fn intersection(x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(xi, yi)| xi.min(*yi)).sum()
    }
}
