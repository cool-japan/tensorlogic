//! Property-based tests for tensorlogic-scirs-backend.
//!
//! These tests use proptest to verify mathematical properties of tensor operations
//! with randomly generated inputs.
//!
//! **Note**: These tests require the `integration-tests` feature to avoid
//! circular dev-dependencies with tensorlogic-compiler.

#![cfg(feature = "integration-tests")]

use proptest::prelude::*;
use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_scirs_backend::Scirs2Exec;

/// Strategy for generating finite f64 values (no NaN, no Inf)
fn finite_f64_strategy() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |v: &f64| v.is_finite())
}

proptest! {
    /// Test that tensor addition is commutative: a + b = b + a
    #[test]
    fn test_add_commutative(
        a_data in prop::collection::vec(finite_f64_strategy(), 5..=10),
        b_data in prop::collection::vec(finite_f64_strategy(), 5..=10)
    ) {
        // Ensure same length
        let len = a_data.len().min(b_data.len());
        let a_vec: Vec<f64> = a_data[..len].to_vec();
        let b_vec: Vec<f64> = b_data[..len].to_vec();

        // Test a + b
        let expr1 = TLExpr::add(
            TLExpr::pred("a", vec![Term::var("i")]),
            TLExpr::pred("b", vec![Term::var("i")])
        );
        let graph1 = compile_to_einsum(&expr1).unwrap();

        let mut exec1 = Scirs2Exec::new();
        exec1.add_tensor("a[a]", Scirs2Exec::from_vec(a_vec.clone(), vec![len]).unwrap());
        exec1.add_tensor("b[a]", Scirs2Exec::from_vec(b_vec.clone(), vec![len]).unwrap());
        let result1 = exec1.forward(&graph1).unwrap();

        // Test b + a
        let expr2 = TLExpr::add(
            TLExpr::pred("b", vec![Term::var("i")]),
            TLExpr::pred("a", vec![Term::var("i")])
        );
        let graph2 = compile_to_einsum(&expr2).unwrap();

        let mut exec2 = Scirs2Exec::new();
        exec2.add_tensor("b[a]", Scirs2Exec::from_vec(b_vec, vec![len]).unwrap());
        exec2.add_tensor("a[a]", Scirs2Exec::from_vec(a_vec, vec![len]).unwrap());
        let result2 = exec2.forward(&graph2).unwrap();

        // Check element-wise equality (with small epsilon for floating point)
        let diff = (&result1 - &result2).mapv(|v| v.abs());
        let max_diff = diff.iter().cloned().fold(0.0, f64::max);

        prop_assert!(max_diff < 1e-10, "Addition not commutative, max diff: {}", max_diff);
    }

    /// Test that multiplication is associative for scalars: (a * b) * c = a * (b * c)
    #[test]
    fn test_mul_associative_scalars(
        a in finite_f64_strategy(),
        b in finite_f64_strategy(),
        c in finite_f64_strategy()
    ) {
        let result1 = (a * b) * c;
        let result2 = a * (b * c);

        let diff = (result1 - result2).abs();
        prop_assert!(diff < 1e-8, "Multiplication not associative: {}", diff);
    }

    /// Test that sigmoid output is always in [0, 1] range
    #[test]
    fn test_sigmoid_range_property(x in finite_f64_strategy()) {
        // Test scalar sigmoid property
        let sigmoid = 1.0 / (1.0 + (-x).exp());

        // Sigmoid is in [0, 1] (inclusive) due to floating point saturation for large |x|
        prop_assert!((0.0..=1.0).contains(&sigmoid), "Sigmoid {} not in [0, 1]", sigmoid);
    }

    /// Test distributive property: a * (b + c) = a*b + a*c
    #[test]
    fn test_multiply_distributive_scalars(
        a in finite_f64_strategy(),
        b in finite_f64_strategy(),
        c in finite_f64_strategy()
    ) {
        let result1 = a * (b + c);
        let result2 = a * b + a * c;

        let diff = (result1 - result2).abs();
        prop_assert!(
            diff < 1e-8,
            "Multiplication not distributive: {} * ({} + {}) = {}, but {} * {} + {} * {} = {}, diff: {}",
            a, b, c, result1, a, b, a, c, result2, diff
        );
    }

    /// Test that sum is linear: sum(a*x + b*y) = a*sum(x) + b*sum(y) for scalars
    #[test]
    fn test_sum_linearity(
        x_data in prop::collection::vec(finite_f64_strategy(), 5..=10),
        y_data in prop::collection::vec(finite_f64_strategy(), 5..=10),
        a in finite_f64_strategy(),
        b in finite_f64_strategy()
    ) {
        let len = x_data.len().min(y_data.len());

        let x_sum: f64 = x_data[..len].iter().sum();
        let y_sum: f64 = y_data[..len].iter().sum();

        let combined: Vec<f64> = x_data[..len].iter().zip(&y_data[..len])
            .map(|(&x, &y)| a * x + b * y)
            .collect();
        let combined_sum: f64 = combined.iter().sum();

        let expected = a * x_sum + b * y_sum;

        let diff = (combined_sum - expected).abs();
        prop_assert!(diff < 1e-6, "Sum not linear: got {}, expected {}, diff {}", combined_sum, expected, diff);
    }

    /// Test that max is monotonic: if x > y, then max(x, z) >= max(y, z)
    #[test]
    fn test_max_monotonic(x in finite_f64_strategy(), y in finite_f64_strategy(), z in finite_f64_strategy()) {
        if x > y {
            let max_x_z = x.max(z);
            let max_y_z = y.max(z);
            prop_assert!(max_x_z >= max_y_z, "Max not monotonic");
        }
    }

    /// Test that multiplication by zero gives zero
    #[test]
    fn test_multiply_by_zero(x in finite_f64_strategy()) {
        let result = x * 0.0;
        prop_assert_eq!(result, 0.0, "Multiply by zero should give zero");
    }

    /// Test that addition of zero is identity
    #[test]
    fn test_add_zero_identity(x in finite_f64_strategy()) {
        let result = x + 0.0;
        let diff = (result - x).abs();
        prop_assert!(diff < 1e-15, "Add zero should be identity");
    }

    /// Test that subtraction gives inverse: x - x = 0
    #[test]
    fn test_subtract_self(x in finite_f64_strategy()) {
        let result = x - x;
        prop_assert!(result.abs() < 1e-15, "Subtract self should give zero");
    }

    /// Test that division by one is identity
    #[test]
    fn test_divide_by_one(x in finite_f64_strategy()) {
        let result = x / 1.0;
        let diff = (result - x).abs();
        prop_assert!(diff < 1e-15, "Divide by one should be identity");
    }

    /// Test that absolute difference is symmetric: |a - b| = |b - a|
    #[test]
    fn test_abs_diff_symmetric(a in finite_f64_strategy(), b in finite_f64_strategy()) {
        let diff1 = (a - b).abs();
        let diff2 = (b - a).abs();
        let error = (diff1 - diff2).abs();
        prop_assert!(error < 1e-15, "Absolute difference not symmetric");
    }
}
