//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use crate::tensor_kernels::types::*;
    use crate::{Kernel, RbfKernelConfig};
    #[test]
    fn test_linear_kernel() {
        let kernel = LinearKernel::new();
        assert_eq!(kernel.name(), "Linear");
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 32.0).abs() < 1e-10);
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
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
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
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
        let x = vec![1.0, 0.0];
        let y = vec![0.0, 1.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!(result.abs() < 1e-10);
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
    #[test]
    fn test_laplacian_kernel_basic() {
        let kernel = LaplacianKernel::new(0.5).unwrap();
        assert_eq!(kernel.name(), "Laplacian");
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
        let y = vec![2.0, 3.0, 4.0];
        let result = kernel.compute(&x, &y).unwrap();
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
        let laplacian = LaplacianKernel::new(0.5).unwrap();
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 10.0];
        let lap_sim = laplacian.compute(&x, &y).unwrap();
        let rbf_sim = rbf.compute(&x, &y).unwrap();
        assert!(lap_sim > rbf_sim);
    }
    #[test]
    fn test_sigmoid_kernel_basic() {
        let kernel = SigmoidKernel::new(0.01, -1.0).unwrap();
        assert_eq!(kernel.name(), "Sigmoid");
        assert!(!kernel.is_psd());
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let result = kernel.compute(&x, &y).unwrap();
        assert!(result >= -1.0);
        assert!(result <= 1.0);
    }
    #[test]
    fn test_sigmoid_kernel_range() {
        let kernel = SigmoidKernel::new(0.01, -1.0).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let result = kernel.compute(&x, &y).unwrap();
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
    #[test]
    fn test_chi_squared_kernel_basic() {
        let kernel = ChiSquaredKernel::new(1.0).unwrap();
        assert_eq!(kernel.name(), "ChiSquared");
        let hist1 = vec![0.2, 0.3, 0.5];
        let hist2 = vec![0.2, 0.3, 0.5];
        let result = kernel.compute(&hist1, &hist2).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_chi_squared_kernel_histograms() {
        let kernel = ChiSquaredKernel::new(1.0).unwrap();
        let hist1 = vec![0.2, 0.3, 0.5];
        let hist2 = vec![0.25, 0.35, 0.4];
        let result = kernel.compute(&hist1, &hist2).unwrap();
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
        assert!(result1 > result2);
    }
    #[test]
    fn test_histogram_intersection_basic() {
        let kernel = HistogramIntersectionKernel::new();
        assert_eq!(kernel.name(), "HistogramIntersection");
        let hist1 = vec![0.2, 0.3, 0.5];
        let hist2 = vec![0.2, 0.3, 0.5];
        let result = kernel.compute(&hist1, &hist2).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_histogram_intersection_partial_overlap() {
        let kernel = HistogramIntersectionKernel::new();
        let hist1 = vec![0.5, 0.3, 0.2];
        let hist2 = vec![0.3, 0.4, 0.3];
        let result = kernel.compute(&hist1, &hist2).unwrap();
        assert!((result - 0.8).abs() < 1e-10);
    }
    #[test]
    fn test_histogram_intersection_no_overlap() {
        let kernel = HistogramIntersectionKernel::new();
        let hist1 = vec![1.0, 0.0, 0.0];
        let hist2 = vec![0.0, 1.0, 0.0];
        let result = kernel.compute(&hist1, &hist2).unwrap();
        assert!(result.abs() < 1e-10);
    }
    #[test]
    fn test_histogram_intersection_negative_values() {
        let kernel = HistogramIntersectionKernel::new();
        let hist1 = vec![0.5, -0.3, 0.2];
        let hist2 = vec![0.3, 0.4, 0.3];
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
        let hist1 = vec![10.0, 20.0, 30.0];
        let hist2 = vec![15.0, 25.0, 20.0];
        let result = kernel.compute(&hist1, &hist2).unwrap();
        assert!((result - 50.0).abs() < 1e-10);
    }
    #[test]
    fn test_rbf_vs_laplacian() {
        let rbf = RbfKernel::new(RbfKernelConfig::new(1.0)).unwrap();
        let lap = LaplacianKernel::new(1.0).unwrap();
        let x = vec![0.0, 0.0, 0.0];
        let y = vec![1.0, 1.0, 1.0];
        let rbf_sim = rbf.compute(&x, &y).unwrap();
        let lap_sim = lap.compute(&x, &y).unwrap();
        assert!((0.0..=1.0).contains(&rbf_sim));
        assert!((0.0..=1.0).contains(&lap_sim));
        assert!((rbf_sim - lap_sim).abs() < 1e-10);
        let x2 = vec![0.0, 0.0];
        let y2 = vec![2.0, 0.0];
        let rbf_sim2 = rbf.compute(&x2, &y2).unwrap();
        let lap_sim2 = lap.compute(&x2, &y2).unwrap();
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
            assert!(
                (k_xx - 1.0).abs() < 1e-10,
                "Kernel {} self-similarity not 1.0: got {}",
                kernel.name(),
                k_xx
            );
        }
    }
    #[test]
    fn test_matern_kernel_nu_half() {
        let kernel = MaternKernel::exponential(1.0).unwrap();
        assert_eq!(kernel.name(), "Matérn");
        assert!((kernel.nu() - 0.5).abs() < 1e-10);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.5, 2.5, 3.5];
        let sim = kernel.compute(&x, &y).unwrap();
        assert!(sim > 0.0 && sim <= 1.0);
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
        assert!(MaternKernel::new(0.0, 1.5).is_err());
        assert!(MaternKernel::new(-1.0, 1.5).is_err());
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
        let kernel_rough = MaternKernel::exponential(1.0).unwrap();
        let kernel_smooth = MaternKernel::nu_5_2(1.0).unwrap();
        let x = vec![0.0];
        let y = vec![0.5];
        let sim_rough = kernel_rough.compute(&x, &y).unwrap();
        let sim_smooth = kernel_smooth.compute(&x, &y).unwrap();
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
        let self_sim = kernel.compute(&x, &x).unwrap();
        assert!((self_sim - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_rational_quadratic_rbf_limit() {
        let x = vec![0.0, 0.0];
        let y = vec![1.0, 0.0];
        let rq_small = RationalQuadraticKernel::new(1.0, 1.0).unwrap();
        let rq_large = RationalQuadraticKernel::new(1.0, 100.0).unwrap();
        let sim_small = rq_small.compute(&x, &y).unwrap();
        let sim_large = rq_large.compute(&x, &y).unwrap();
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
        let y2 = vec![5.0 + period];
        let y3 = vec![5.0 + 2.0 * period];
        let sim1 = kernel.compute(&x, &y1).unwrap();
        let sim2 = kernel.compute(&x, &y2).unwrap();
        let sim3 = kernel.compute(&x, &y3).unwrap();
        assert!((sim1 - sim2).abs() < 1e-8);
        assert!((sim1 - sim3).abs() < 1e-8);
    }
    #[test]
    fn test_periodic_kernel_exact_period() {
        let period = 24.0;
        let kernel = PeriodicKernel::new(period, 1.0).unwrap();
        let x = vec![1.0];
        let y = vec![1.0 + period];
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
        let y = vec![1.0];
        let sim_smooth = kernel_smooth.compute(&x, &y).unwrap();
        let sim_rough = kernel_rough.compute(&x, &y).unwrap();
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
    #[test]
    fn test_rbf_kernel_gradient_basic() {
        let config = RbfKernelConfig::new(0.5);
        let kernel = RbfKernel::new(config).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let (k, grad) = kernel.compute_with_gradient(&x, &y).unwrap();
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);
        assert!(grad < 0.0);
    }
    #[test]
    fn test_rbf_kernel_gradient_same_point() {
        let config = RbfKernelConfig::new(0.5);
        let kernel = RbfKernel::new(config).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let (k, grad) = kernel.compute_with_gradient(&x, &x).unwrap();
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
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);
        assert!(grad_sigma > 0.0);
    }
    #[test]
    fn test_polynomial_kernel_constant_gradient() {
        let kernel = PolynomialKernel::new(2, 1.0).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let (k, grad_c) = kernel.compute_with_constant_gradient(&x, &y).unwrap();
        assert!((k - 144.0).abs() < 1e-10);
        assert!((grad_c - 24.0).abs() < 1e-10);
    }
    #[test]
    fn test_polynomial_kernel_all_gradients() {
        let kernel = PolynomialKernel::new(3, 1.0).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![2.0, 3.0];
        let (k, grad_c, grad_d) = kernel.compute_with_all_gradients(&x, &y).unwrap();
        assert!((k - 729.0).abs() < 1e-8);
        assert!((grad_c - 243.0).abs() < 1e-8);
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
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);
        assert!(grad < 0.0);
    }
    #[test]
    fn test_laplacian_kernel_gradient_same_point() {
        let kernel = LaplacianKernel::new(0.5).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let (k, grad) = kernel.compute_with_gradient(&x, &x).unwrap();
        assert!((k - 1.0).abs() < 1e-10);
        assert!(grad.abs() < 1e-10);
    }
    #[test]
    fn test_laplacian_kernel_sigma_gradient() {
        let kernel = LaplacianKernel::new(0.5).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let (k, grad_sigma) = kernel.compute_with_sigma_gradient(&x, &y).unwrap();
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);
        assert!(grad_sigma > 0.0);
    }
    #[test]
    fn test_matern_kernel_gradient_nu_half() {
        let kernel = MaternKernel::exponential(1.0).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let (k, grad_l) = kernel.compute_with_length_scale_gradient(&x, &y).unwrap();
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10, "K mismatch: {} vs {}", k, k_std);
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
        assert!((k - 1.0).abs() < 1e-10);
        assert!(grad_l.abs() < 1e-10);
    }
    #[test]
    fn test_rational_quadratic_kernel_length_scale_gradient() {
        let kernel = RationalQuadraticKernel::new(1.0, 2.0).unwrap();
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let (k, grad_l) = kernel.compute_with_length_scale_gradient(&x, &y).unwrap();
        let k_std = kernel.compute(&x, &y).unwrap();
        assert!((k - k_std).abs() < 1e-10);
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
