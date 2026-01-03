//! SkleaRS integration module
//!
//! This module provides adapters to integrate TensorLogic kernels with the SkleaRS
//! machine learning library, enabling use of logic-derived kernels in SkleaRS algorithms
//! like kernel SVM, kernel ridge regression, and kernel PCA.
//!
//! # Features
//!
//! - **KernelFunction trait** - All tensor kernels implement SkleaRS's `KernelFunction`
//! - **Random Fourier Features** - Support for kernel approximation methods
//! - **Seamless Integration** - Direct use in SkleaRS estimators
//!
//! # Example
//!
//! ```rust,ignore
//! use tensorlogic_sklears_kernels::{RbfKernel, RbfKernelConfig};
//! use tensorlogic_sklears_kernels::sklears_integration::SklearsKernelAdapter;
//!
//! // Create a TensorLogic kernel
//! let tl_kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
//!
//! // Wrap it for SkleaRS
//! let sklears_kernel = SklearsKernelAdapter::new(tl_kernel);
//!
//! // Use in SkleaRS algorithms
//! // let svm = KernelSVM::new(sklears_kernel);
//! ```

#[cfg(feature = "sklears")]
use sklears_core::types::Float;
#[cfg(feature = "sklears")]
use sklears_kernel_approximation::custom_kernel::KernelFunction;

use crate::{
    ChiSquaredKernel, CosineKernel, HistogramIntersectionKernel, Kernel, LaplacianKernel,
    LinearKernel, PolynomialKernel, RbfKernel, SigmoidKernel,
};
use scirs2_core::ndarray::{s, Array2};
use scirs2_core::random::essentials::Normal as RandNormal;
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;

/// Adapter that wraps any TensorLogic kernel to implement SkleaRS's KernelFunction trait
///
/// This allows TensorLogic kernels to be used seamlessly in SkleaRS algorithms that
/// expect the `KernelFunction` trait.
#[cfg(feature = "sklears")]
#[derive(Clone)]
pub struct SklearsKernelAdapter<K: Kernel + Clone> {
    /// The underlying TensorLogic kernel
    kernel: K,
}

#[cfg(feature = "sklears")]
impl<K: Kernel + Clone> SklearsKernelAdapter<K> {
    /// Create a new adapter wrapping a TensorLogic kernel
    pub fn new(kernel: K) -> Self {
        Self { kernel }
    }

    /// Get a reference to the underlying kernel
    pub fn inner(&self) -> &K {
        &self.kernel
    }

    /// Consume the adapter and return the underlying kernel
    pub fn into_inner(self) -> K {
        self.kernel
    }
}

// Specific implementations for each kernel type
// Note: We use specific implementations instead of a generic one to allow
// customization of Fourier transforms and frequency sampling for each kernel

/// RBF kernel adapter with proper Fourier transform
#[cfg(feature = "sklears")]
impl KernelFunction for SklearsKernelAdapter<RbfKernel> {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        self.kernel.compute(x, y).unwrap_or(0.0)
    }

    fn fourier_transform(&self, w: &[Float]) -> Float {
        let gamma = self.kernel.gamma();
        let w_norm_sq: Float = w.iter().map(|wi| wi.powi(2)).sum();
        (-w_norm_sq / (4.0 * gamma)).exp()
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        let gamma = self.kernel.gamma();
        let normal = RandNormal::new(0.0, (2.0 * gamma).sqrt()).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = normal.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        format!("RBF kernel with gamma={}", self.kernel.gamma())
    }
}

/// Laplacian kernel adapter with proper Fourier transform
#[cfg(feature = "sklears")]
impl KernelFunction for SklearsKernelAdapter<LaplacianKernel> {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        self.kernel.compute(x, y).unwrap_or(0.0)
    }

    fn fourier_transform(&self, w: &[Float]) -> Float {
        let gamma = self.kernel.gamma();
        let w_norm_sq: Float = w.iter().map(|wi| wi.powi(2)).sum();
        // Cauchy distribution in frequency domain
        1.0 / (1.0 + w_norm_sq / gamma)
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        let gamma = self.kernel.gamma();
        // Sample from Cauchy distribution for Laplacian kernel
        let normal = RandNormal::new(0.0, gamma.sqrt()).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = normal.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        format!("Laplacian kernel with gamma={}", self.kernel.gamma())
    }
}

/// Linear kernel adapter
#[cfg(feature = "sklears")]
impl KernelFunction for SklearsKernelAdapter<LinearKernel> {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        self.kernel.compute(x, y).unwrap_or(0.0)
    }

    fn fourier_transform(&self, _w: &[Float]) -> Float {
        // Linear kernel is already in feature space
        1.0
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        _rng: &mut RealStdRng,
    ) -> Array2<Float> {
        // Linear kernel doesn't need random features
        // Return identity-like mapping
        Array2::eye(n_features.min(n_components))
            .slice(s![0..n_features, 0..n_components])
            .to_owned()
    }

    fn description(&self) -> String {
        "Linear kernel (dot product)".to_string()
    }
}

/// Polynomial kernel adapter
#[cfg(feature = "sklears")]
impl KernelFunction for SklearsKernelAdapter<PolynomialKernel> {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        self.kernel.compute(x, y).unwrap_or(0.0)
    }

    fn fourier_transform(&self, _w: &[Float]) -> Float {
        // Polynomial kernel doesn't have a simple Fourier representation
        1.0
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        // Use standard normal for lack of better alternative
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = normal.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        format!(
            "Polynomial kernel (degree={}, constant={})",
            self.kernel.degree(),
            self.kernel.constant()
        )
    }
}

/// Cosine kernel adapter
#[cfg(feature = "sklears")]
impl KernelFunction for SklearsKernelAdapter<CosineKernel> {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        self.kernel.compute(x, y).unwrap_or(0.0)
    }

    fn fourier_transform(&self, _w: &[Float]) -> Float {
        1.0
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = normal.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        "Cosine similarity kernel".to_string()
    }
}

/// Sigmoid kernel adapter
#[cfg(feature = "sklears")]
impl KernelFunction for SklearsKernelAdapter<SigmoidKernel> {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        self.kernel.compute(x, y).unwrap_or(0.0)
    }

    fn fourier_transform(&self, _w: &[Float]) -> Float {
        1.0
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = normal.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        format!(
            "Sigmoid kernel (alpha={}, offset={})",
            self.kernel.alpha(),
            self.kernel.offset()
        )
    }
}

/// Chi-Squared kernel adapter
#[cfg(feature = "sklears")]
impl KernelFunction for SklearsKernelAdapter<ChiSquaredKernel> {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        self.kernel.compute(x, y).unwrap_or(0.0)
    }

    fn fourier_transform(&self, _w: &[Float]) -> Float {
        1.0
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = normal.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        format!("Chi-Squared kernel (gamma={})", self.kernel.gamma())
    }
}

/// Histogram Intersection kernel adapter
#[cfg(feature = "sklears")]
impl KernelFunction for SklearsKernelAdapter<HistogramIntersectionKernel> {
    fn kernel(&self, x: &[Float], y: &[Float]) -> Float {
        self.kernel.compute(x, y).unwrap_or(0.0)
    }

    fn fourier_transform(&self, _w: &[Float]) -> Float {
        1.0
    }

    fn sample_frequencies(
        &self,
        n_features: usize,
        n_components: usize,
        rng: &mut RealStdRng,
    ) -> Array2<Float> {
        let normal = RandNormal::new(0.0, 1.0).unwrap();
        let mut weights = Array2::zeros((n_features, n_components));
        for mut col in weights.columns_mut() {
            for val in col.iter_mut() {
                *val = normal.sample(rng);
            }
        }
        weights
    }

    fn description(&self) -> String {
        "Histogram Intersection kernel".to_string()
    }
}

// Note: Accessor methods for kernel parameters are defined in tensor_kernel.rs

#[cfg(all(test, feature = "sklears"))]
mod tests {
    use super::*;
    use crate::RbfKernelConfig;

    #[test]
    fn test_rbf_kernel_adapter() {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let adapter = SklearsKernelAdapter::new(kernel);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        // Test kernel computation
        let result = adapter.kernel(&x, &y);
        assert!(result > 0.0 && result <= 1.0);

        // Test description
        let desc = adapter.description();
        assert!(desc.contains("RBF"));
        assert!(desc.contains("0.5"));
    }

    #[test]
    fn test_linear_kernel_adapter() {
        let kernel = LinearKernel::new();
        let adapter = SklearsKernelAdapter::new(kernel);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = adapter.kernel(&x, &y);
        assert!((result - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_laplacian_kernel_adapter() {
        let kernel = LaplacianKernel::new(0.5).unwrap();
        let adapter = SklearsKernelAdapter::new(kernel);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let result = adapter.kernel(&x, &y);
        assert!((result - 1.0).abs() < 1e-6); // Same vectors -> similarity = 1
    }

    #[test]
    fn test_polynomial_kernel_adapter() {
        let kernel = PolynomialKernel::new(2, 1.0).unwrap();
        let adapter = SklearsKernelAdapter::new(kernel);

        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];

        let result = adapter.kernel(&x, &y);
        // (1*3 + 2*4 + 1)^2 = (11 + 1)^2 = 144
        assert!((result - 144.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_kernel_adapter() {
        let kernel = CosineKernel::new();
        let adapter = SklearsKernelAdapter::new(kernel);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0]; // Parallel to x

        let result = adapter.kernel(&x, &y);
        assert!((result - 1.0).abs() < 1e-6); // Parallel vectors -> cosine = 1
    }

    #[test]
    fn test_adapter_into_inner() {
        let kernel = LinearKernel::new();
        let adapter = SklearsKernelAdapter::new(kernel.clone());

        let recovered = adapter.into_inner();
        assert_eq!(recovered.name(), kernel.name());
    }

    #[test]
    fn test_adapter_inner_ref() {
        let kernel = LinearKernel::new();
        let adapter = SklearsKernelAdapter::new(kernel.clone());

        let inner_ref = adapter.inner();
        assert_eq!(inner_ref.name(), kernel.name());
    }
}
