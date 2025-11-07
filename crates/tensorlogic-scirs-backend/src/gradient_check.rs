//! Numeric gradient checking utilities for verifying analytical gradients.
//!
//! This module provides tools to verify the correctness of analytical gradient
//! computation by comparing against numeric gradients computed via finite differences.

use crate::{Scirs2Exec, Scirs2Tensor};
use scirs2_core::ndarray::ArrayD;
use tensorlogic_infer::{ExecutorError, TlAutodiff};
use tensorlogic_ir::EinsumGraph;

/// Configuration for gradient checking
#[derive(Clone, Copy, Debug)]
pub struct GradientCheckConfig {
    /// Epsilon for finite difference computation
    pub epsilon: f64,
    /// Relative tolerance for gradient comparison
    pub rtol: f64,
    /// Absolute tolerance for gradient comparison
    pub atol: f64,
}

impl Default for GradientCheckConfig {
    fn default() -> Self {
        GradientCheckConfig {
            epsilon: 1e-5,
            rtol: 1e-3,
            atol: 1e-5,
        }
    }
}

/// Result of gradient checking for a single tensor
#[derive(Debug)]
pub struct GradientCheckResult {
    /// Name of the tensor being checked
    pub tensor_name: String,
    /// Maximum absolute difference between analytical and numeric gradients
    pub max_abs_diff: f64,
    /// Maximum relative difference
    pub max_rel_diff: f64,
    /// Whether the gradient check passed
    pub passed: bool,
    /// Number of elements checked
    pub num_elements: usize,
}

impl GradientCheckResult {
    /// Check if gradients match within tolerance
    pub fn is_close(&self, config: &GradientCheckConfig) -> bool {
        self.max_abs_diff < config.atol || self.max_rel_diff < config.rtol
    }
}

/// Compute numeric gradient for a specific input tensor using finite differences
pub fn compute_numeric_gradient(
    graph: &EinsumGraph,
    executor: &mut Scirs2Exec,
    tensor_name: &str,
    config: &GradientCheckConfig,
) -> Result<Scirs2Tensor, ExecutorError> {
    // Get the input tensor
    let input_tensor = executor
        .tensors
        .get(tensor_name)
        .ok_or_else(|| ExecutorError::TensorNotFound(tensor_name.to_string()))?
        .clone();

    let shape = input_tensor.shape();
    let mut numeric_grad = ArrayD::zeros(shape);

    // Compute numeric gradient for each element using central differences
    for idx in 0..input_tensor.len() {
        // Create perturbed tensors
        let mut tensor_plus = input_tensor.clone();
        let mut tensor_minus = input_tensor.clone();

        // Apply perturbation
        let flat_plus = tensor_plus.as_slice_mut().unwrap();
        let flat_minus = tensor_minus.as_slice_mut().unwrap();
        flat_plus[idx] += config.epsilon;
        flat_minus[idx] -= config.epsilon;

        // Compute forward pass with perturbed inputs
        executor.add_tensor(tensor_name, tensor_plus);
        let output_plus = executor.forward(graph)?;

        executor.add_tensor(tensor_name, tensor_minus);
        let output_minus = executor.forward(graph)?;

        // Compute numeric gradient: (f(x+ε) - f(x-ε)) / (2ε)
        let grad_value = (output_plus.sum() - output_minus.sum()) / (2.0 * config.epsilon);

        // Store in numeric gradient array
        let flat_grad = numeric_grad.as_slice_mut().unwrap();
        flat_grad[idx] = grad_value;
    }

    // Restore original input tensor
    executor.add_tensor(tensor_name, input_tensor);

    Ok(numeric_grad)
}

/// Compare analytical and numeric gradients
pub fn compare_gradients(
    analytical: &Scirs2Tensor,
    numeric: &Scirs2Tensor,
    tensor_name: &str,
    config: &GradientCheckConfig,
) -> GradientCheckResult {
    assert_eq!(
        analytical.shape(),
        numeric.shape(),
        "Gradient shapes must match"
    );

    let mut max_abs_diff: f64 = 0.0;
    let mut max_rel_diff: f64 = 0.0;
    let num_elements = analytical.len();

    // Compare each element
    for (a, n) in analytical.iter().zip(numeric.iter()) {
        let abs_diff = (a - n).abs();
        let rel_diff = if n.abs() > 1e-10 {
            abs_diff / n.abs()
        } else {
            abs_diff
        };

        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
    }

    let passed = max_abs_diff < config.atol || max_rel_diff < config.rtol;

    GradientCheckResult {
        tensor_name: tensor_name.to_string(),
        max_abs_diff,
        max_rel_diff,
        passed,
        num_elements,
    }
}

/// Check gradients for all input tensors in the graph
pub fn check_gradients(
    graph: &EinsumGraph,
    executor: &mut Scirs2Exec,
    config: Option<GradientCheckConfig>,
) -> Result<Vec<GradientCheckResult>, ExecutorError> {
    let config = config.unwrap_or_default();
    let mut results = Vec::new();

    // Perform forward pass to compute output
    let output = executor.forward(graph)?;

    // Compute analytical gradients via backward pass
    let loss_grad = Scirs2Tensor::ones(output.raw_dim());
    let analytical_tape = executor.backward(graph, &loss_grad)?;

    // Check gradient for each input tensor
    for (idx, tensor_name) in graph.tensors.iter().enumerate() {
        // Skip if this is an intermediate or output tensor (not an input)
        if executor.tensors.contains_key(tensor_name) {
            // Compute numeric gradient
            let numeric_grad = compute_numeric_gradient(graph, executor, tensor_name, &config)?;

            // Get analytical gradient from tape
            if let Some(Some(analytical_grad)) = analytical_tape.tensors.get(idx) {
                let result =
                    compare_gradients(analytical_grad, &numeric_grad, tensor_name, &config);
                results.push(result);
            }
        }
    }

    Ok(results)
}

#[cfg(all(test, feature = "integration-tests"))]
mod tests {
    use super::*;
    use tensorlogic_compiler::compile_to_einsum;
    use tensorlogic_ir::{TLExpr, Term};

    #[test]
    fn test_gradient_check_add() {
        // Create a simple addition expression: x + y
        let x = TLExpr::pred("x", vec![Term::var("i"), Term::var("j")]);
        let y = TLExpr::pred("y", vec![Term::var("i"), Term::var("j")]);
        let expr = TLExpr::add(x, y);

        // Compile to graph
        let graph = compile_to_einsum(&expr).unwrap();

        // Setup executor with input tensors
        let mut executor = Scirs2Exec::new();
        let x_tensor = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y_tensor = Scirs2Exec::from_vec(vec![0.5, 0.5, 1.0, 1.0], vec![2, 2]).unwrap();

        executor.add_tensor(graph.tensors[0].clone(), x_tensor);
        executor.add_tensor(graph.tensors[1].clone(), y_tensor);

        // Check gradients
        let results = check_gradients(&graph, &mut executor, None).unwrap();

        // Verify all gradients passed
        for result in results {
            println!(
                "Tensor: {}, Max abs diff: {:.6e}, Max rel diff: {:.6e}, Passed: {}",
                result.tensor_name, result.max_abs_diff, result.max_rel_diff, result.passed
            );
            assert!(
                result.passed,
                "Gradient check failed for {}",
                result.tensor_name
            );
        }
    }

    #[test]
    fn test_gradient_check_multiply() {
        // Create multiplication expression: x * y
        let x = TLExpr::pred("x", vec![Term::var("i"), Term::var("j")]);
        let y = TLExpr::pred("y", vec![Term::var("i"), Term::var("j")]);
        let expr = TLExpr::mul(x, y);

        let graph = compile_to_einsum(&expr).unwrap();

        let mut executor = Scirs2Exec::new();
        let x_tensor = Scirs2Exec::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let y_tensor = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        executor.add_tensor(graph.tensors[0].clone(), x_tensor);
        executor.add_tensor(graph.tensors[1].clone(), y_tensor);

        let results = check_gradients(&graph, &mut executor, None).unwrap();

        for result in results {
            println!(
                "Tensor: {}, Max abs diff: {:.6e}, Max rel diff: {:.6e}",
                result.tensor_name, result.max_abs_diff, result.max_rel_diff
            );
            assert!(
                result.passed,
                "Gradient check failed for {}",
                result.tensor_name
            );
        }
    }

    #[test]
    fn test_gradient_check_divide() {
        // Create division expression: x / y
        let x = TLExpr::pred("x", vec![Term::var("i"), Term::var("j")]);
        let y = TLExpr::pred("y", vec![Term::var("i"), Term::var("j")]);
        let expr = TLExpr::div(x, y);

        let graph = compile_to_einsum(&expr).unwrap();

        let mut executor = Scirs2Exec::new();
        let x_tensor = Scirs2Exec::from_vec(vec![6.0, 8.0, 10.0, 12.0], vec![2, 2]).unwrap();
        let y_tensor = Scirs2Exec::from_vec(vec![2.0, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();

        executor.add_tensor(graph.tensors[0].clone(), x_tensor);
        executor.add_tensor(graph.tensors[1].clone(), y_tensor);

        let results = check_gradients(&graph, &mut executor, None).unwrap();

        for result in results {
            println!(
                "Tensor: {}, Max abs diff: {:.6e}, Max rel diff: {:.6e}",
                result.tensor_name, result.max_abs_diff, result.max_rel_diff
            );
            assert!(
                result.passed,
                "Gradient check failed for {}",
                result.tensor_name
            );
        }
    }
}
