//! Gradient checking utilities for validating autodiff implementations.
//!
//! This module provides numerical gradient checking to verify that
//! automatic differentiation implementations are correct.
//!
//! # Example
//!
//! ```ignore
//! use tensorlogic_infer::gradcheck::{check_gradients, GradCheckConfig};
//!
//! let config = GradCheckConfig::default();
//! let result = check_gradients(
//!     &mut executor,
//!     &graph,
//!     &inputs,
//!     config
//! )?;
//!
//! assert!(result.max_error < 1e-5);
//! ```

use std::collections::HashMap;

/// Configuration for gradient checking
#[derive(Debug, Clone)]
pub struct GradCheckConfig {
    /// Epsilon for numerical differentiation
    pub epsilon: f64,
    /// Relative tolerance for comparing gradients
    pub rel_tolerance: f64,
    /// Absolute tolerance for comparing gradients
    pub abs_tolerance: f64,
    /// Whether to print detailed errors
    pub verbose: bool,
    /// Maximum number of errors to report
    pub max_errors_to_report: usize,
}

impl Default for GradCheckConfig {
    fn default() -> Self {
        GradCheckConfig {
            epsilon: 1e-5,
            rel_tolerance: 1e-3,
            abs_tolerance: 1e-5,
            verbose: false,
            max_errors_to_report: 10,
        }
    }
}

impl GradCheckConfig {
    /// Create a strict configuration with tighter tolerances
    pub fn strict() -> Self {
        GradCheckConfig {
            epsilon: 1e-6,
            rel_tolerance: 1e-4,
            abs_tolerance: 1e-6,
            verbose: true,
            max_errors_to_report: 10,
        }
    }

    /// Create a relaxed configuration with looser tolerances
    pub fn relaxed() -> Self {
        GradCheckConfig {
            epsilon: 1e-4,
            rel_tolerance: 1e-2,
            abs_tolerance: 1e-4,
            verbose: false,
            max_errors_to_report: 10,
        }
    }

    /// Enable verbose error reporting
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set epsilon for numerical differentiation
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set relative tolerance
    pub fn with_rel_tolerance(mut self, tolerance: f64) -> Self {
        self.rel_tolerance = tolerance;
        self
    }

    /// Set absolute tolerance
    pub fn with_abs_tolerance(mut self, tolerance: f64) -> Self {
        self.abs_tolerance = tolerance;
        self
    }
}

/// Result of gradient checking
#[derive(Debug, Clone)]
pub struct GradCheckResult {
    /// Number of parameters checked
    pub num_params: usize,
    /// Number of mismatches found
    pub num_errors: usize,
    /// Maximum absolute error
    pub max_error: f64,
    /// Maximum relative error
    pub max_rel_error: f64,
    /// Average absolute error
    pub avg_error: f64,
    /// Whether all gradients passed the check
    pub passed: bool,
    /// Detailed error information
    pub errors: Vec<GradientError>,
}

impl GradCheckResult {
    /// Create a new result
    pub fn new(num_params: usize) -> Self {
        GradCheckResult {
            num_params,
            num_errors: 0,
            max_error: 0.0,
            max_rel_error: 0.0,
            avg_error: 0.0,
            passed: true,
            errors: Vec::new(),
        }
    }

    /// Add an error to the result
    pub fn add_error(&mut self, error: GradientError) {
        self.num_errors += 1;
        self.max_error = self.max_error.max(error.abs_error);
        self.max_rel_error = self.max_rel_error.max(error.rel_error);
        self.passed = false;
        self.errors.push(error);
    }

    /// Finalize the result by computing averages
    pub fn finalize(mut self) -> Self {
        if !self.errors.is_empty() {
            let total_error: f64 = self.errors.iter().map(|e| e.abs_error).sum();
            self.avg_error = total_error / self.errors.len() as f64;
        }
        self
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        format!(
            "Gradient Check: {} params, {} errors, max_error={:.2e}, max_rel_error={:.2e}, avg_error={:.2e}, {}",
            self.num_params,
            self.num_errors,
            self.max_error,
            self.max_rel_error,
            self.avg_error,
            if self.passed { "PASSED" } else { "FAILED" }
        )
    }

    /// Print detailed error report
    pub fn print_errors(&self, max_to_print: usize) {
        if self.errors.is_empty() {
            println!("✓ All gradients passed!");
            return;
        }

        println!("\n✗ Gradient errors found:");
        for (i, error) in self.errors.iter().take(max_to_print).enumerate() {
            println!(
                "  [{}] Param {}: analytical={:.6e}, numerical={:.6e}, abs_err={:.2e}, rel_err={:.2e}",
                i + 1,
                error.param_id,
                error.analytical_grad,
                error.numerical_grad,
                error.abs_error,
                error.rel_error
            );
        }

        if self.errors.len() > max_to_print {
            println!("  ... and {} more errors", self.errors.len() - max_to_print);
        }
    }
}

/// Information about a gradient error
#[derive(Debug, Clone)]
pub struct GradientError {
    /// Parameter identifier
    pub param_id: String,
    /// Index in the flattened parameter vector
    pub index: usize,
    /// Analytical gradient from autodiff
    pub analytical_grad: f64,
    /// Numerical gradient from finite differences
    pub numerical_grad: f64,
    /// Absolute error
    pub abs_error: f64,
    /// Relative error
    pub rel_error: f64,
}

impl GradientError {
    /// Create a new gradient error
    pub fn new(param_id: String, index: usize, analytical: f64, numerical: f64) -> Self {
        let abs_error = (analytical - numerical).abs();
        let rel_error = if numerical.abs() > 1e-10 {
            abs_error / numerical.abs()
        } else {
            abs_error
        };

        GradientError {
            param_id,
            index,
            analytical_grad: analytical,
            numerical_grad: numerical,
            abs_error,
            rel_error,
        }
    }

    /// Check if this error exceeds tolerances
    pub fn exceeds_tolerance(&self, config: &GradCheckConfig) -> bool {
        self.abs_error > config.abs_tolerance && self.rel_error > config.rel_tolerance
    }
}

/// Compute numerical gradient using central differences
///
/// For a function f(x), the numerical gradient is approximated as:
/// df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)
pub fn numerical_gradient_central(
    forward_fn: impl Fn(&[f64]) -> f64,
    x: &[f64],
    epsilon: f64,
) -> Vec<f64> {
    let mut grad = vec![0.0; x.len()];

    for i in 0..x.len() {
        // Compute f(x + ε)
        let mut x_plus = x.to_vec();
        x_plus[i] += epsilon;
        let f_plus = forward_fn(&x_plus);

        // Compute f(x - ε)
        let mut x_minus = x.to_vec();
        x_minus[i] -= epsilon;
        let f_minus = forward_fn(&x_minus);

        // Central difference
        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);
    }

    grad
}

/// Compute numerical gradient using forward differences
///
/// For a function f(x), the numerical gradient is approximated as:
/// df/dx ≈ (f(x + ε) - f(x)) / ε
pub fn numerical_gradient_forward(
    forward_fn: impl Fn(&[f64]) -> f64,
    x: &[f64],
    f_x: f64,
    epsilon: f64,
) -> Vec<f64> {
    let mut grad = vec![0.0; x.len()];

    for i in 0..x.len() {
        // Compute f(x + ε)
        let mut x_plus = x.to_vec();
        x_plus[i] += epsilon;
        let f_plus = forward_fn(&x_plus);

        // Forward difference
        grad[i] = (f_plus - f_x) / epsilon;
    }

    grad
}

/// Compare two gradients and return detailed comparison
pub fn compare_gradients(
    param_id: String,
    analytical: &[f64],
    numerical: &[f64],
    config: &GradCheckConfig,
) -> Vec<GradientError> {
    assert_eq!(analytical.len(), numerical.len());

    let mut errors = Vec::new();

    for (i, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let error = GradientError::new(param_id.clone(), i, a, n);

        if error.exceeds_tolerance(config) {
            errors.push(error);
        }
    }

    errors
}

/// Gradient checker for multi-parameter functions
pub struct GradientChecker {
    config: GradCheckConfig,
    results: HashMap<String, GradCheckResult>,
}

impl GradientChecker {
    /// Create a new gradient checker with the given configuration
    pub fn new(config: GradCheckConfig) -> Self {
        GradientChecker {
            config,
            results: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(GradCheckConfig::default())
    }

    /// Check gradients for a single parameter
    pub fn check_parameter(
        &mut self,
        param_id: String,
        forward_fn: impl Fn(&[f64]) -> f64,
        x: &[f64],
        analytical_grad: &[f64],
    ) -> GradCheckResult {
        // Compute numerical gradient
        let numerical_grad = numerical_gradient_central(&forward_fn, x, self.config.epsilon);

        // Compare gradients
        let errors = compare_gradients(
            param_id.clone(),
            analytical_grad,
            &numerical_grad,
            &self.config,
        );

        // Build result
        let mut result = GradCheckResult::new(x.len());
        for error in errors {
            result.add_error(error);
        }
        let result = result.finalize();

        if self.config.verbose {
            println!("Checking parameter '{}':", param_id);
            println!("  {}", result.summary());
            if !result.passed {
                result.print_errors(self.config.max_errors_to_report);
            }
        }

        self.results.insert(param_id, result.clone());
        result
    }

    /// Get results for all checked parameters
    pub fn results(&self) -> &HashMap<String, GradCheckResult> {
        &self.results
    }

    /// Check if all parameters passed
    pub fn all_passed(&self) -> bool {
        self.results.values().all(|r| r.passed)
    }

    /// Get total number of errors across all parameters
    pub fn total_errors(&self) -> usize {
        self.results.values().map(|r| r.num_errors).sum()
    }

    /// Print summary of all checks
    pub fn print_summary(&self) {
        println!("\n=== Gradient Check Summary ===");
        for (param_id, result) in &self.results {
            println!("{}: {}", param_id, result.summary());
        }
        println!(
            "\nTotal: {} parameters, {} errors",
            self.results.len(),
            self.total_errors()
        );

        if self.all_passed() {
            println!("✓ All gradient checks PASSED");
        } else {
            println!("✗ Some gradient checks FAILED");
        }
    }
}

/// Quick gradient check for a single function
pub fn quick_check(
    forward_fn: impl Fn(&[f64]) -> f64,
    x: &[f64],
    analytical_grad: &[f64],
) -> Result<(), String> {
    let config = GradCheckConfig::default();
    let numerical = numerical_gradient_central(&forward_fn, x, config.epsilon);

    let errors = compare_gradients(
        "quick_check".to_string(),
        analytical_grad,
        &numerical,
        &config,
    );

    if errors.is_empty() {
        Ok(())
    } else {
        let mut result = GradCheckResult::new(x.len());
        for error in errors {
            result.add_error(error);
        }
        Err(result.finalize().summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_check_config_default() {
        let config = GradCheckConfig::default();
        assert!(config.epsilon > 0.0);
        assert!(config.rel_tolerance > 0.0);
        assert!(config.abs_tolerance > 0.0);
    }

    #[test]
    fn test_grad_check_config_strict() {
        let strict = GradCheckConfig::strict();
        let default = GradCheckConfig::default();
        assert!(strict.epsilon <= default.epsilon);
        assert!(strict.rel_tolerance <= default.rel_tolerance);
    }

    #[test]
    fn test_grad_check_config_builder() {
        let config = GradCheckConfig::default()
            .with_epsilon(1e-4)
            .with_verbose(true)
            .with_rel_tolerance(1e-2);

        assert_eq!(config.epsilon, 1e-4);
        assert!(config.verbose);
        assert_eq!(config.rel_tolerance, 1e-2);
    }

    #[test]
    fn test_numerical_gradient_simple() {
        // f(x) = x^2, df/dx = 2x
        let f = |x: &[f64]| x[0] * x[0];
        let x = vec![3.0];
        let grad = numerical_gradient_central(f, &x, 1e-5);

        // Should be close to 2*3 = 6
        assert!((grad[0] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_numerical_gradient_multivariate() {
        // f(x, y) = x^2 + y^2, df/dx = 2x, df/dy = 2y
        let f = |xy: &[f64]| xy[0] * xy[0] + xy[1] * xy[1];
        let xy = vec![3.0, 4.0];
        let grad = numerical_gradient_central(f, &xy, 1e-5);

        assert!((grad[0] - 6.0).abs() < 1e-4);
        assert!((grad[1] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_gradient_error_creation() {
        let error = GradientError::new("param1".to_string(), 0, 1.0, 1.01);

        assert_eq!(error.param_id, "param1");
        assert_eq!(error.index, 0);
        assert_eq!(error.analytical_grad, 1.0);
        assert_eq!(error.numerical_grad, 1.01);
        assert!(error.abs_error > 0.0);
        assert!(error.rel_error > 0.0);
    }

    #[test]
    fn test_gradient_error_exceeds_tolerance() {
        let config = GradCheckConfig::default();

        // Large error
        let error1 = GradientError::new("p1".to_string(), 0, 1.0, 2.0);
        assert!(error1.exceeds_tolerance(&config));

        // Small error
        let error2 = GradientError::new("p2".to_string(), 0, 1.0, 1.0000001);
        assert!(!error2.exceeds_tolerance(&config));
    }

    #[test]
    fn test_grad_check_result() {
        let mut result = GradCheckResult::new(10);
        assert!(result.passed);
        assert_eq!(result.num_errors, 0);

        result.add_error(GradientError::new("p1".to_string(), 0, 1.0, 2.0));
        assert!(!result.passed);
        assert_eq!(result.num_errors, 1);

        let final_result = result.finalize();
        assert!(final_result.avg_error > 0.0);
    }

    #[test]
    fn test_compare_gradients() {
        let config = GradCheckConfig::default();

        // Perfect match
        let analytical = vec![1.0, 2.0, 3.0];
        let numerical = vec![1.0, 2.0, 3.0];
        let errors = compare_gradients("test".to_string(), &analytical, &numerical, &config);
        assert_eq!(errors.len(), 0);

        // With errors
        let numerical2 = vec![1.0, 2.5, 3.0];
        let errors2 = compare_gradients("test".to_string(), &analytical, &numerical2, &config);
        assert!(!errors2.is_empty());
    }

    #[test]
    fn test_gradient_checker() {
        let mut checker = GradientChecker::new(GradCheckConfig::default());

        // Check a simple function: f(x) = x^2
        let f = |x: &[f64]| x[0] * x[0];
        let x = vec![3.0];
        let analytical = vec![6.0]; // df/dx = 2x = 6

        let result = checker.check_parameter("x".to_string(), f, &x, &analytical);
        assert!(result.passed);
        assert!(checker.all_passed());
    }

    #[test]
    fn test_quick_check() {
        // Correct gradient
        let f = |x: &[f64]| x[0] * x[0];
        let x = vec![3.0];
        let grad = vec![6.0];
        assert!(quick_check(f, &x, &grad).is_ok());

        // Incorrect gradient
        let bad_grad = vec![7.0];
        assert!(quick_check(f, &x, &bad_grad).is_err());
    }

    #[test]
    fn test_forward_gradient() {
        let f = |x: &[f64]| x[0] * x[0];
        let x = vec![3.0];
        let f_x = f(&x);
        let grad = numerical_gradient_forward(f, &x, f_x, 1e-5);

        // Should be close to 6.0, but less accurate than central
        assert!((grad[0] - 6.0).abs() < 1e-3);
    }
}
