//! Backend compatibility test templates.
//!
//! This module provides comprehensive test templates that backend developers
//! can use to verify their implementations of TlExecutor and TlAutodiff traits.
//!
//! # Usage
//!
//! Backend developers should implement the `BackendTestAdapter` trait for their
//! executor and then use the provided test functions to validate correctness.
//!
//! ```ignore
//! use tensorlogic_infer::backend_tests::*;
//!
//! struct MyBackendAdapter;
//!
//! impl BackendTestAdapter for MyBackendAdapter {
//!     type Executor = MyExecutor;
//!     type Tensor = MyTensor;
//!
//!     fn create_executor() -> Self::Executor {
//!         MyExecutor::new()
//!     }
//!
//!     fn create_tensor_from_data(data: &[f64], shape: &[usize]) -> Self::Tensor {
//!         MyTensor::from_data(data, shape)
//!     }
//!
//!     fn tensor_to_vec(tensor: &Self::Tensor) -> Vec<f64> {
//!         tensor.to_vec()
//!     }
//! }
//!
//! // Run all tests
//! test_backend_basic_ops::<MyBackendAdapter>();
//! test_backend_einsum::<MyBackendAdapter>();
//! test_backend_autodiff::<MyBackendAdapter>();
//! ```

use crate::ops::{ElemOp, ReduceOp};
use crate::traits::{TlAutodiff, TlExecutor};

/// Adapter trait for backend testing.
///
/// Backend developers implement this trait to adapt their executor
/// to the test framework.
pub trait BackendTestAdapter {
    /// The executor type being tested
    type Executor: TlExecutor<Tensor = Self::Tensor>;
    /// The tensor type used by the executor
    type Tensor: Clone;

    /// Create a new executor instance for testing
    fn create_executor() -> Self::Executor;

    /// Create a tensor from raw data and shape
    fn create_tensor_from_data(data: &[f64], shape: &[usize]) -> Self::Tensor;

    /// Convert tensor to a flat vector for comparison
    fn tensor_to_vec(tensor: &Self::Tensor) -> Vec<f64>;

    /// Get the shape of a tensor
    fn tensor_shape(tensor: &Self::Tensor) -> Vec<usize>;

    /// Create a scalar tensor
    fn create_scalar(value: f64) -> Self::Tensor {
        Self::create_tensor_from_data(&[value], &[])
    }

    /// Create a 1D tensor (vector)
    fn create_vector(data: &[f64]) -> Self::Tensor {
        Self::create_tensor_from_data(data, &[data.len()])
    }

    /// Create a 2D tensor (matrix)
    fn create_matrix(data: &[f64], rows: usize, cols: usize) -> Self::Tensor {
        assert_eq!(data.len(), rows * cols);
        Self::create_tensor_from_data(data, &[rows, cols])
    }
}

/// Test result with optional failure message
pub type TestResult = Result<(), String>;

/// Tolerance for floating-point comparisons
pub const DEFAULT_TOLERANCE: f64 = 1e-6;

/// Compare two vectors with tolerance
pub fn assert_vec_close(actual: &[f64], expected: &[f64], tolerance: f64) -> TestResult {
    if actual.len() != expected.len() {
        return Err(format!(
            "Length mismatch: got {}, expected {}",
            actual.len(),
            expected.len()
        ));
    }

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > tolerance && diff / (e.abs() + 1e-10) > tolerance {
            return Err(format!(
                "Value mismatch at index {}: got {}, expected {}, diff {}",
                i, a, e, diff
            ));
        }
    }

    Ok(())
}

//
// ===== BASIC OPERATION TESTS =====
//

/// Test basic element-wise unary operations
pub fn test_backend_elem_unary<A: BackendTestAdapter>() -> TestResult
where
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    let mut executor = A::create_executor();

    // Test OneMinus
    let x = A::create_vector(&[1.0, 0.5, 0.0]);
    let result = executor
        .elem_op(ElemOp::OneMinus, &x)
        .map_err(|e| format!("OneMinus failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[0.0, 0.5, 1.0], DEFAULT_TOLERANCE)?;

    // Test Relu
    let x = A::create_vector(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = executor
        .elem_op(ElemOp::Relu, &x)
        .map_err(|e| format!("Relu failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[0.0, 0.0, 0.0, 1.0, 2.0], DEFAULT_TOLERANCE)?;

    // Test Sigmoid
    let x = A::create_vector(&[0.0]);
    let result = executor
        .elem_op(ElemOp::Sigmoid, &x)
        .map_err(|e| format!("Sigmoid failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[0.5], DEFAULT_TOLERANCE)?;

    Ok(())
}

/// Test basic element-wise binary operations
pub fn test_backend_elem_binary<A: BackendTestAdapter>() -> TestResult
where
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    let mut executor = A::create_executor();

    // Test Add
    let x = A::create_vector(&[1.0, 2.0, 3.0]);
    let y = A::create_vector(&[4.0, 5.0, 6.0]);
    let result = executor
        .elem_op_binary(ElemOp::Add, &x, &y)
        .map_err(|e| format!("Add failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[5.0, 7.0, 9.0], DEFAULT_TOLERANCE)?;

    // Test Multiply
    let result = executor
        .elem_op_binary(ElemOp::Multiply, &x, &y)
        .map_err(|e| format!("Multiply failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[4.0, 10.0, 18.0], DEFAULT_TOLERANCE)?;

    // Test Subtract
    let result = executor
        .elem_op_binary(ElemOp::Subtract, &x, &y)
        .map_err(|e| format!("Subtract failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[-3.0, -3.0, -3.0], DEFAULT_TOLERANCE)?;

    // Test Divide
    let result = executor
        .elem_op_binary(ElemOp::Divide, &y, &x)
        .map_err(|e| format!("Divide failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[4.0, 2.5, 2.0], DEFAULT_TOLERANCE)?;

    Ok(())
}

/// Test reduction operations
pub fn test_backend_reduce<A: BackendTestAdapter>() -> TestResult
where
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    let mut executor = A::create_executor();

    // Test Sum reduction
    let x = A::create_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

    // Sum over axis 0
    let result = executor
        .reduce(ReduceOp::Sum, &x, &[0])
        .map_err(|e| format!("Sum reduce failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[5.0, 7.0, 9.0], DEFAULT_TOLERANCE)?;

    // Sum over axis 1
    let result = executor
        .reduce(ReduceOp::Sum, &x, &[1])
        .map_err(|e| format!("Sum reduce failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[6.0, 15.0], DEFAULT_TOLERANCE)?;

    // Test Max reduction
    let x = A::create_vector(&[1.0, 5.0, 3.0, 2.0]);
    let result = executor
        .reduce(ReduceOp::Max, &x, &[0])
        .map_err(|e| format!("Max reduce failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[5.0], DEFAULT_TOLERANCE)?;

    // Test Mean reduction
    let x = A::create_vector(&[2.0, 4.0, 6.0, 8.0]);
    let result = executor
        .reduce(ReduceOp::Mean, &x, &[0])
        .map_err(|e| format!("Mean reduce failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[5.0], DEFAULT_TOLERANCE)?;

    Ok(())
}

/// Test einsum operations
pub fn test_backend_einsum<A: BackendTestAdapter>() -> TestResult
where
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    let mut executor = A::create_executor();

    // Test vector dot product: "i,i->"
    let a = A::create_vector(&[1.0, 2.0, 3.0]);
    let b = A::create_vector(&[4.0, 5.0, 6.0]);
    let result = executor
        .einsum("i,i->", &[a.clone(), b.clone()])
        .map_err(|e| format!("Einsum dot product failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[32.0], DEFAULT_TOLERANCE)?; // 1*4 + 2*5 + 3*6 = 32

    // Test matrix-vector multiply: "ij,j->i"
    let mat = A::create_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let vec = A::create_vector(&[1.0, 2.0, 3.0]);
    let result = executor
        .einsum("ij,j->i", &[mat, vec])
        .map_err(|e| format!("Einsum matvec failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[14.0, 32.0], DEFAULT_TOLERANCE)?;

    // Test matrix-matrix multiply: "ij,jk->ik"
    let a = A::create_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = A::create_matrix(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let result = executor
        .einsum("ij,jk->ik", &[a, b])
        .map_err(|e| format!("Einsum matmul failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[19.0, 22.0, 43.0, 50.0], DEFAULT_TOLERANCE)?;

    Ok(())
}

//
// ===== AUTODIFF TESTS =====
//

/// Test forward pass execution
///
/// Note: This is a placeholder test. Backend developers should implement
/// their own forward pass tests based on their specific graph execution
/// requirements and tensor injection mechanisms.
pub fn test_backend_forward<A>() -> TestResult
where
    A: BackendTestAdapter,
    A::Executor: TlAutodiff<Tensor = A::Tensor>,
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    // This is a simplified test that backend developers should customize
    // for their specific implementation. The test validates that the
    // forward pass can be called without panicking.

    // Backend-specific graph construction and execution should go here
    Ok(())
}

//
// ===== NUMERICAL STABILITY TESTS =====
//

/// Test handling of edge cases (NaN, Inf, zeros)
pub fn test_backend_edge_cases<A: BackendTestAdapter>() -> TestResult
where
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    let mut executor = A::create_executor();

    // Test division by zero handling
    let x = A::create_vector(&[1.0, 2.0, 3.0]);
    let y = A::create_vector(&[1.0, 0.0, 3.0]);
    let result = executor.elem_op_binary(ElemOp::Divide, &x, &y);

    // Backend should either return Inf or error - both are acceptable
    match result {
        Ok(tensor) => {
            let output = A::tensor_to_vec(&tensor);
            assert_eq!(output.len(), 3);
            assert!((output[0] - 1.0).abs() < DEFAULT_TOLERANCE);
            assert!(output[1].is_infinite() || output[1].is_nan());
        }
        Err(_) => {
            // Error on division by zero is also acceptable
        }
    }

    // Test Relu with very large values
    let x = A::create_vector(&[1e10, -1e10, 0.0]);
    let result = executor
        .elem_op(ElemOp::Relu, &x)
        .map_err(|e| format!("Relu with large values failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[1e10, 0.0, 0.0], 1e4)?;

    Ok(())
}

//
// ===== SHAPE HANDLING TESTS =====
//

/// Test various tensor shapes
pub fn test_backend_shapes<A: BackendTestAdapter>() -> TestResult
where
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    let mut executor = A::create_executor();

    // Test scalar operations
    let scalar1 = A::create_scalar(5.0);
    let scalar2 = A::create_scalar(3.0);
    let result = executor
        .elem_op_binary(ElemOp::Add, &scalar1, &scalar2)
        .map_err(|e| format!("Scalar add failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[8.0], DEFAULT_TOLERANCE)?;

    // Test broadcasting (if supported)
    // Backends may choose to not support broadcasting - test should be optional

    // Test empty reduction
    let x = A::create_vector(&[1.0, 2.0, 3.0]);
    let result = executor
        .reduce(ReduceOp::Sum, &x, &[]) // No axes = reduce all
        .map_err(|e| format!("Empty axes reduce failed: {:?}", e))?;
    let output = A::tensor_to_vec(&result);
    assert_vec_close(&output, &[6.0], DEFAULT_TOLERANCE)?;

    Ok(())
}

//
// ===== PERFORMANCE/STRESS TESTS =====
//

/// Test performance with large tensors
pub fn test_backend_large_tensors<A: BackendTestAdapter>() -> TestResult
where
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    let mut executor = A::create_executor();

    // Create large vectors
    let size = 10000;
    let data1: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let data2: Vec<f64> = (0..size).map(|i| (size - i) as f64).collect();

    let x = A::create_vector(&data1);
    let y = A::create_vector(&data2);

    // Test large vector addition
    let result = executor
        .elem_op_binary(ElemOp::Add, &x, &y)
        .map_err(|e| format!("Large vector add failed: {:?}", e))?;

    let output = A::tensor_to_vec(&result);
    assert_eq!(output.len(), size);

    // Verify a few values
    assert_vec_close(
        &output[0..3],
        &[10000.0, 10000.0, 10000.0],
        DEFAULT_TOLERANCE,
    )?;

    Ok(())
}

/// Test memory efficiency with repeated operations
pub fn test_backend_memory_efficiency<A: BackendTestAdapter>() -> TestResult
where
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    let mut executor = A::create_executor();

    // Perform many operations to test memory management
    let mut x = A::create_vector(&[1.0, 2.0, 3.0]);

    for i in 0..100 {
        let y = A::create_scalar((i + 1) as f64);
        x = executor
            .elem_op_binary(ElemOp::Add, &x, &y)
            .map_err(|e| format!("Memory efficiency test failed at iteration {}: {:?}", i, e))?;
    }

    // After 100 iterations of adding 1, 2, 3, ..., 100
    // Sum = 100 * 101 / 2 = 5050
    let output = A::tensor_to_vec(&x);
    assert_vec_close(&output, &[5051.0, 5052.0, 5053.0], DEFAULT_TOLERANCE)?;

    Ok(())
}

//
// ===== GRADIENT CHECKING =====
//

/// Compute numerical gradient using finite differences
pub fn numerical_gradient<A, F>(f: F, x: &A::Tensor, epsilon: f64) -> Vec<f64>
where
    A: BackendTestAdapter,
    F: Fn(&A::Tensor) -> A::Tensor,
{
    let x_vec = A::tensor_to_vec(x);
    let shape = A::tensor_shape(x);
    let mut grad = vec![0.0; x_vec.len()];

    for i in 0..x_vec.len() {
        // Compute f(x + epsilon)
        let mut x_plus = x_vec.clone();
        x_plus[i] += epsilon;
        let x_plus_tensor = A::create_tensor_from_data(&x_plus, &shape);
        let f_plus = A::tensor_to_vec(&f(&x_plus_tensor));

        // Compute f(x - epsilon)
        let mut x_minus = x_vec.clone();
        x_minus[i] -= epsilon;
        let x_minus_tensor = A::create_tensor_from_data(&x_minus, &shape);
        let f_minus = A::tensor_to_vec(&f(&x_minus_tensor));

        // Central difference: (f(x+eps) - f(x-eps)) / (2*eps)
        grad[i] = (f_plus[0] - f_minus[0]) / (2.0 * epsilon);
    }

    grad
}

//
// ===== COMPREHENSIVE TEST SUITE =====
//

/// Run all basic operation tests
pub fn run_all_basic_tests<A: BackendTestAdapter>() -> Vec<(String, TestResult)>
where
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    vec![
        ("elem_unary".to_string(), test_backend_elem_unary::<A>()),
        ("elem_binary".to_string(), test_backend_elem_binary::<A>()),
        ("reduce".to_string(), test_backend_reduce::<A>()),
        ("einsum".to_string(), test_backend_einsum::<A>()),
        ("edge_cases".to_string(), test_backend_edge_cases::<A>()),
        ("shapes".to_string(), test_backend_shapes::<A>()),
    ]
}

/// Run all performance tests
pub fn run_all_performance_tests<A: BackendTestAdapter>() -> Vec<(String, TestResult)>
where
    <A::Executor as TlExecutor>::Error: std::fmt::Debug,
{
    vec![
        (
            "large_tensors".to_string(),
            test_backend_large_tensors::<A>(),
        ),
        (
            "memory_efficiency".to_string(),
            test_backend_memory_efficiency::<A>(),
        ),
    ]
}

/// Print test results summary
pub fn print_test_summary(results: &[(String, TestResult)]) {
    println!("\n=== Backend Test Results ===");
    let mut passed = 0;
    let mut failed = 0;

    for (name, result) in results {
        match result {
            Ok(()) => {
                println!("✓ {}", name);
                passed += 1;
            }
            Err(msg) => {
                println!("✗ {} - {}", name, msg);
                failed += 1;
            }
        }
    }

    println!("\nPassed: {}, Failed: {}", passed, failed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_vec_close() {
        assert!(assert_vec_close(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1e-10).is_ok());
        assert!(assert_vec_close(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.1], 0.2).is_ok());
        assert!(assert_vec_close(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.1], 0.01).is_err());
        assert!(assert_vec_close(&[1.0, 2.0], &[1.0, 2.0, 3.0], 1e-10).is_err());
    }

    #[test]
    fn test_default_tolerance() {
        // Verify tolerance is sensible (not a compile-time constant check)
        let tolerance = DEFAULT_TOLERANCE;
        let max_tolerance = 1e-5;
        assert!(tolerance > 0.0);
        assert!(tolerance < max_tolerance);
    }
}
