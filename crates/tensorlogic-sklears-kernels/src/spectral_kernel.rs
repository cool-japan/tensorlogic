// Allow needless_range_loop for spectral computations which are clearer with indexed loops
#![allow(clippy::needless_range_loop)]

//! Spectral Mixture kernels for discovering latent periodic components.
//!
//! The Spectral Mixture (SM) kernel models a signal as a mixture of
//! Gaussian components in the spectral (frequency) domain. This allows
//! it to automatically discover multiple periodic patterns in data.
//!
//! ## Key Features
//!
//! - **Automatic pattern discovery**: Learns multiple periodic components
//! - **Flexible modeling**: Can approximate any stationary kernel
//! - **Interpretable**: Each component has clear frequency and length scale
//!
//! ## Reference
//!
//! Wilson, A. G., & Adams, R. P. (2013). "Gaussian Process Kernels for
//! Pattern Discovery and Extrapolation." ICML.

use crate::error::{KernelError, Result};
use crate::types::Kernel;
use std::f64::consts::PI;

/// A single spectral component with weight, mean frequency, and variance.
#[derive(Debug, Clone)]
pub struct SpectralComponent {
    /// Weight (mixture proportion)
    pub weight: f64,
    /// Mean frequency (per-dimension)
    pub mean: Vec<f64>,
    /// Variance (per-dimension, controls bandwidth)
    pub variance: Vec<f64>,
}

impl SpectralComponent {
    /// Create a new spectral component.
    ///
    /// # Arguments
    /// * `weight` - Mixture weight (must be positive)
    /// * `mean` - Mean frequency for each dimension
    /// * `variance` - Variance for each dimension (must be positive)
    pub fn new(weight: f64, mean: Vec<f64>, variance: Vec<f64>) -> Result<Self> {
        if weight <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "weight".to_string(),
                value: weight.to_string(),
                reason: "weight must be positive".to_string(),
            });
        }

        if mean.len() != variance.len() {
            return Err(KernelError::InvalidParameter {
                parameter: "mean/variance".to_string(),
                value: format!(
                    "mean.len()={}, variance.len()={}",
                    mean.len(),
                    variance.len()
                ),
                reason: "mean and variance must have same length".to_string(),
            });
        }

        if mean.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "mean".to_string(),
                value: "[]".to_string(),
                reason: "must have at least one dimension".to_string(),
            });
        }

        for (i, &v) in variance.iter().enumerate() {
            if v <= 0.0 {
                return Err(KernelError::InvalidParameter {
                    parameter: format!("variance[{}]", i),
                    value: v.to_string(),
                    reason: "variance must be positive".to_string(),
                });
            }
        }

        Ok(Self {
            weight,
            mean,
            variance,
        })
    }

    /// Create a 1D spectral component.
    pub fn new_1d(weight: f64, mean: f64, variance: f64) -> Result<Self> {
        Self::new(weight, vec![mean], vec![variance])
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.mean.len()
    }
}

/// Spectral Mixture (SM) kernel.
///
/// K(x, y) = Σ_q w_q * exp(-2π² * Σ_d (x_d - y_d)² * v_q,d) * cos(2π * Σ_d (x_d - y_d) * μ_q,d)
///
/// where:
/// - w_q is the weight of component q
/// - μ_q,d is the mean frequency of component q in dimension d
/// - v_q,d is the variance of component q in dimension d
///
/// This kernel can approximate any stationary covariance function arbitrarily
/// well as the number of components increases.
#[derive(Debug, Clone)]
pub struct SpectralMixtureKernel {
    /// Spectral components
    components: Vec<SpectralComponent>,
    /// Number of input dimensions
    ndim: usize,
}

impl SpectralMixtureKernel {
    /// Create a new Spectral Mixture kernel.
    ///
    /// # Arguments
    /// * `components` - List of spectral components
    ///
    /// # Example
    /// ```rust
    /// use tensorlogic_sklears_kernels::spectral_kernel::{SpectralMixtureKernel, SpectralComponent};
    /// use tensorlogic_sklears_kernels::Kernel;
    ///
    /// // Create a kernel with two periodic components
    /// let components = vec![
    ///     SpectralComponent::new_1d(1.0, 0.1, 0.01).unwrap(),  // Low frequency
    ///     SpectralComponent::new_1d(0.5, 1.0, 0.1).unwrap(),   // High frequency
    /// ];
    /// let kernel = SpectralMixtureKernel::new(components).unwrap();
    /// ```
    pub fn new(components: Vec<SpectralComponent>) -> Result<Self> {
        if components.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "components".to_string(),
                value: "[]".to_string(),
                reason: "must have at least one component".to_string(),
            });
        }

        let ndim = components[0].ndim();
        for (i, comp) in components.iter().enumerate() {
            if comp.ndim() != ndim {
                return Err(KernelError::InvalidParameter {
                    parameter: format!("components[{}]", i),
                    value: format!("ndim={}", comp.ndim()),
                    reason: format!("all components must have {} dimensions", ndim),
                });
            }
        }

        Ok(Self { components, ndim })
    }

    /// Create a simple 1D spectral mixture kernel with given frequencies.
    ///
    /// # Arguments
    /// * `frequencies` - List of (weight, mean_frequency, variance) tuples
    pub fn new_1d(frequencies: Vec<(f64, f64, f64)>) -> Result<Self> {
        let components: Result<Vec<_>> = frequencies
            .into_iter()
            .map(|(w, m, v)| SpectralComponent::new_1d(w, m, v))
            .collect();
        Self::new(components?)
    }

    /// Get the components.
    pub fn components(&self) -> &[SpectralComponent] {
        &self.components
    }

    /// Get the number of components.
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Compute the contribution of a single component.
    fn compute_component(&self, comp: &SpectralComponent, tau: &[f64]) -> f64 {
        let mut exp_term = 0.0;
        let mut cos_term = 0.0;

        for d in 0..self.ndim {
            let tau_d = tau[d];
            exp_term += tau_d * tau_d * comp.variance[d];
            cos_term += tau_d * comp.mean[d];
        }

        // K_q = w_q * exp(-2π² * exp_term) * cos(2π * cos_term)
        comp.weight * (-2.0 * PI * PI * exp_term).exp() * (2.0 * PI * cos_term).cos()
    }
}

impl Kernel for SpectralMixtureKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != self.ndim {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.ndim],
                got: vec![x.len()],
                context: "Spectral Mixture kernel".to_string(),
            });
        }
        if y.len() != self.ndim {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.ndim],
                got: vec![y.len()],
                context: "Spectral Mixture kernel".to_string(),
            });
        }

        // Compute tau = x - y
        let tau: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();

        // Sum over all components
        let mut result = 0.0;
        for comp in &self.components {
            result += self.compute_component(comp, &tau);
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "SpectralMixture"
    }
}

/// Exponential Sine Squared kernel (also known as Periodic kernel).
///
/// K(x, y) = exp(-2 * sin²(π * |x - y| / period) / l²)
///
/// This is equivalent to the ExpSineSquared kernel in scikit-learn.
/// It models functions that repeat exactly over a specified period.
#[derive(Debug, Clone)]
pub struct ExpSineSquaredKernel {
    /// Period of the periodic pattern
    period: f64,
    /// Length scale (controls smoothness within period)
    length_scale: f64,
}

impl ExpSineSquaredKernel {
    /// Create a new Exponential Sine Squared kernel.
    ///
    /// # Arguments
    /// * `period` - Period of the pattern (must be positive)
    /// * `length_scale` - Length scale parameter (must be positive)
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

    /// Get the period.
    pub fn period(&self) -> f64 {
        self.period
    }

    /// Get the length scale.
    pub fn length_scale(&self) -> f64 {
        self.length_scale
    }
}

impl Kernel for ExpSineSquaredKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "ExpSineSquared kernel".to_string(),
            });
        }

        // Compute Euclidean distance
        let dist: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        let sin_term = (PI * dist / self.period).sin();
        let result = (-2.0 * sin_term * sin_term / (self.length_scale * self.length_scale)).exp();

        Ok(result)
    }

    fn name(&self) -> &str {
        "ExpSineSquared"
    }
}

/// Locally Periodic kernel: RBF × Periodic
///
/// K(x, y) = k_rbf(x, y) * k_periodic(x, y)
///
/// Models functions that are periodic but whose amplitude varies smoothly.
/// The RBF component controls the locality (how quickly periodicity decays),
/// while the periodic component captures the repetitive structure.
#[derive(Debug, Clone)]
pub struct LocallyPeriodicKernel {
    /// Period of the periodic component
    period: f64,
    /// Length scale for the periodic component
    periodic_length_scale: f64,
    /// Length scale for the RBF component (controls locality)
    rbf_length_scale: f64,
}

impl LocallyPeriodicKernel {
    /// Create a new Locally Periodic kernel.
    ///
    /// # Arguments
    /// * `period` - Period of the repetitive pattern
    /// * `periodic_length_scale` - Length scale within each period
    /// * `rbf_length_scale` - How quickly the periodic pattern decays
    pub fn new(period: f64, periodic_length_scale: f64, rbf_length_scale: f64) -> Result<Self> {
        if period <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "period".to_string(),
                value: period.to_string(),
                reason: "period must be positive".to_string(),
            });
        }
        if periodic_length_scale <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "periodic_length_scale".to_string(),
                value: periodic_length_scale.to_string(),
                reason: "periodic_length_scale must be positive".to_string(),
            });
        }
        if rbf_length_scale <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "rbf_length_scale".to_string(),
                value: rbf_length_scale.to_string(),
                reason: "rbf_length_scale must be positive".to_string(),
            });
        }
        Ok(Self {
            period,
            periodic_length_scale,
            rbf_length_scale,
        })
    }

    /// Get the period.
    pub fn period(&self) -> f64 {
        self.period
    }

    /// Get the periodic length scale.
    pub fn periodic_length_scale(&self) -> f64 {
        self.periodic_length_scale
    }

    /// Get the RBF length scale.
    pub fn rbf_length_scale(&self) -> f64 {
        self.rbf_length_scale
    }
}

impl Kernel for LocallyPeriodicKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Locally Periodic kernel".to_string(),
            });
        }

        // Compute squared distance and distance
        let sq_dist: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
        let dist = sq_dist.sqrt();

        // RBF component: exp(-0.5 * d² / l²)
        let rbf = (-0.5 * sq_dist / (self.rbf_length_scale * self.rbf_length_scale)).exp();

        // Periodic component: exp(-2 * sin²(π * d / p) / l²)
        let sin_term = (PI * dist / self.period).sin();
        let periodic = (-2.0 * sin_term * sin_term
            / (self.periodic_length_scale * self.periodic_length_scale))
            .exp();

        Ok(rbf * periodic)
    }

    fn name(&self) -> &str {
        "LocallyPeriodic"
    }
}

/// Product of RBF and Linear kernels.
///
/// K(x, y) = k_rbf(x, y) * k_linear(x, y)
///
/// Models functions with smoothly varying linear trends.
#[derive(Debug, Clone)]
pub struct RbfLinearKernel {
    /// RBF length scale
    length_scale: f64,
    /// Linear kernel variance
    variance: f64,
}

impl RbfLinearKernel {
    /// Create a new RBF × Linear kernel.
    pub fn new(length_scale: f64, variance: f64) -> Result<Self> {
        if length_scale <= 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "length_scale".to_string(),
                value: length_scale.to_string(),
                reason: "length_scale must be positive".to_string(),
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
            length_scale,
            variance,
        })
    }

    /// Get the length scale.
    pub fn length_scale(&self) -> f64 {
        self.length_scale
    }

    /// Get the variance.
    pub fn variance(&self) -> f64 {
        self.variance
    }
}

impl Kernel for RbfLinearKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "RBF-Linear kernel".to_string(),
            });
        }

        // Squared distance
        let sq_dist: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum();

        // RBF component
        let rbf = (-0.5 * sq_dist / (self.length_scale * self.length_scale)).exp();

        // Linear component (dot product)
        let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let linear = self.variance * dot;

        Ok(rbf * linear)
    }

    fn name(&self) -> &str {
        "RBF-Linear"
    }

    fn is_psd(&self) -> bool {
        // Product of PSD kernels is PSD
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Spectral Component Tests =====

    #[test]
    fn test_spectral_component_1d() {
        let comp = SpectralComponent::new_1d(1.0, 0.5, 0.1).unwrap();
        assert!((comp.weight - 1.0).abs() < 1e-10);
        assert_eq!(comp.ndim(), 1);
    }

    #[test]
    fn test_spectral_component_multidim() {
        let comp = SpectralComponent::new(1.0, vec![0.1, 0.2], vec![0.01, 0.02]).unwrap();
        assert_eq!(comp.ndim(), 2);
    }

    #[test]
    fn test_spectral_component_invalid_weight() {
        assert!(SpectralComponent::new_1d(0.0, 0.5, 0.1).is_err());
        assert!(SpectralComponent::new_1d(-1.0, 0.5, 0.1).is_err());
    }

    #[test]
    fn test_spectral_component_invalid_variance() {
        assert!(SpectralComponent::new_1d(1.0, 0.5, 0.0).is_err());
        assert!(SpectralComponent::new_1d(1.0, 0.5, -0.1).is_err());
    }

    #[test]
    fn test_spectral_component_mismatched_dims() {
        assert!(SpectralComponent::new(1.0, vec![0.1, 0.2], vec![0.01]).is_err());
    }

    // ===== Spectral Mixture Kernel Tests =====

    #[test]
    fn test_spectral_mixture_kernel_single_component() {
        let components = vec![SpectralComponent::new_1d(1.0, 0.0, 0.1).unwrap()];
        let kernel = SpectralMixtureKernel::new(components).unwrap();
        assert_eq!(kernel.name(), "SpectralMixture");
        assert_eq!(kernel.num_components(), 1);

        let x = vec![0.0];
        let y = vec![0.0];
        let sim = kernel.compute(&x, &y).unwrap();
        // At same point with mean=0: cos(0) = 1, exp(0) = 1
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_mixture_kernel_multiple_components() {
        let components = vec![
            SpectralComponent::new_1d(0.5, 0.1, 0.01).unwrap(),
            SpectralComponent::new_1d(0.5, 1.0, 0.1).unwrap(),
        ];
        let kernel = SpectralMixtureKernel::new(components).unwrap();
        assert_eq!(kernel.num_components(), 2);

        let x = vec![0.0];
        let y = vec![0.0];
        let sim = kernel.compute(&x, &y).unwrap();
        // At same point: should be sum of weights = 1.0
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_mixture_kernel_1d_convenience() {
        let kernel =
            SpectralMixtureKernel::new_1d(vec![(1.0, 0.5, 0.1), (0.5, 1.0, 0.05)]).unwrap();
        assert_eq!(kernel.num_components(), 2);
        assert_eq!(kernel.ndim(), 1);
    }

    #[test]
    fn test_spectral_mixture_kernel_periodicity() {
        // Single component with specific frequency should show periodicity
        // Note: SM kernel has exponential decay * cos, so values won't be exactly 1
        // at period boundaries, but the cosine component peaks at period multiples
        let freq = 0.25; // Period = 1/freq = 4
                         // Use very small variance to minimize exponential decay
        let components = vec![SpectralComponent::new_1d(1.0, freq, 0.0001).unwrap()];
        let kernel = SpectralMixtureKernel::new(components).unwrap();

        let x = vec![0.0];
        let y_period = vec![4.0]; // One period - cosine term = 1
        let y_half = vec![2.0]; // Half period - cosine term = -1

        let sim_period = kernel.compute(&x, &y_period).unwrap();
        let sim_half = kernel.compute(&x, &y_half).unwrap();

        // At exact period, cosine = 1, so value should be positive and near decay term
        // At half period, cosine = -1, so value should be negative or lower
        assert!(
            sim_period > sim_half,
            "Period value {} should exceed half-period value {}",
            sim_period,
            sim_half
        );
        // Period value should be reasonably high (accounting for some decay)
        assert!(
            sim_period > 0.5,
            "Period value {} should be > 0.5",
            sim_period
        );
    }

    #[test]
    fn test_spectral_mixture_kernel_symmetry() {
        let components = vec![SpectralComponent::new_1d(1.0, 0.5, 0.1).unwrap()];
        let kernel = SpectralMixtureKernel::new(components).unwrap();

        let x = vec![1.0];
        let y = vec![2.0];

        let k_xy = kernel.compute(&x, &y).unwrap();
        let k_yx = kernel.compute(&y, &x).unwrap();
        assert!((k_xy - k_yx).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_mixture_kernel_empty_components() {
        let result = SpectralMixtureKernel::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_spectral_mixture_kernel_dimension_mismatch() {
        let components = vec![SpectralComponent::new_1d(1.0, 0.5, 0.1).unwrap()];
        let kernel = SpectralMixtureKernel::new(components).unwrap();

        let x = vec![0.0, 0.0]; // 2D
        let y = vec![0.0]; // 1D

        assert!(kernel.compute(&x, &y).is_err());
    }

    // ===== Exponential Sine Squared Kernel Tests =====

    #[test]
    fn test_exp_sine_squared_kernel_basic() {
        let kernel = ExpSineSquaredKernel::new(10.0, 1.0).unwrap();
        assert_eq!(kernel.name(), "ExpSineSquared");

        let x = vec![0.0];
        let y = vec![0.0];
        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_sine_squared_kernel_periodicity() {
        let period = 10.0;
        let kernel = ExpSineSquaredKernel::new(period, 1.0).unwrap();

        let x = vec![0.0];
        let y1 = vec![period]; // One period
        let y2 = vec![2.0 * period]; // Two periods

        let sim1 = kernel.compute(&x, &y1).unwrap();
        let sim2 = kernel.compute(&x, &y2).unwrap();

        // At exact period multiples, similarity should be very high
        assert!(sim1 > 0.99);
        assert!(sim2 > 0.99);
    }

    #[test]
    fn test_exp_sine_squared_kernel_invalid() {
        assert!(ExpSineSquaredKernel::new(0.0, 1.0).is_err());
        assert!(ExpSineSquaredKernel::new(10.0, 0.0).is_err());
    }

    // ===== Locally Periodic Kernel Tests =====

    #[test]
    fn test_locally_periodic_kernel_basic() {
        let kernel = LocallyPeriodicKernel::new(10.0, 1.0, 100.0).unwrap();
        assert_eq!(kernel.name(), "LocallyPeriodic");

        let x = vec![0.0];
        let sim = kernel.compute(&x, &x).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_locally_periodic_kernel_decay() {
        // With small RBF length scale, periodicity should decay quickly
        let kernel = LocallyPeriodicKernel::new(10.0, 1.0, 5.0).unwrap();

        let x = vec![0.0];
        let y_near = vec![10.0]; // One period, near
        let y_far = vec![100.0]; // Ten periods, far

        let sim_near = kernel.compute(&x, &y_near).unwrap();
        let sim_far = kernel.compute(&x, &y_far).unwrap();

        // Far point should have much lower similarity due to RBF decay
        assert!(sim_near > sim_far);
    }

    #[test]
    fn test_locally_periodic_kernel_invalid() {
        assert!(LocallyPeriodicKernel::new(0.0, 1.0, 1.0).is_err());
        assert!(LocallyPeriodicKernel::new(10.0, 0.0, 1.0).is_err());
        assert!(LocallyPeriodicKernel::new(10.0, 1.0, 0.0).is_err());
    }

    // ===== RBF-Linear Kernel Tests =====

    #[test]
    fn test_rbf_linear_kernel_basic() {
        let kernel = RbfLinearKernel::new(1.0, 1.0).unwrap();
        assert_eq!(kernel.name(), "RBF-Linear");
        assert!(kernel.is_psd());

        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];

        let sim = kernel.compute(&x, &y).unwrap();
        // dot(x,x) = 5, rbf(x,x) = 1, so result = 5
        assert!((sim - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_linear_kernel_symmetry() {
        let kernel = RbfLinearKernel::new(1.0, 1.0).unwrap();

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let k_xy = kernel.compute(&x, &y).unwrap();
        let k_yx = kernel.compute(&y, &x).unwrap();
        assert!((k_xy - k_yx).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_linear_kernel_invalid() {
        assert!(RbfLinearKernel::new(0.0, 1.0).is_err());
        assert!(RbfLinearKernel::new(1.0, 0.0).is_err());
    }

    // ===== Integration Tests =====

    #[test]
    fn test_spectral_kernels_symmetry() {
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(
                SpectralMixtureKernel::new(vec![SpectralComponent::new_1d(1.0, 0.5, 0.1).unwrap()])
                    .unwrap(),
            ),
            Box::new(ExpSineSquaredKernel::new(10.0, 1.0).unwrap()),
            Box::new(LocallyPeriodicKernel::new(10.0, 1.0, 10.0).unwrap()),
            Box::new(RbfLinearKernel::new(1.0, 1.0).unwrap()),
        ];

        let x = vec![1.0];
        let y = vec![2.0];

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
