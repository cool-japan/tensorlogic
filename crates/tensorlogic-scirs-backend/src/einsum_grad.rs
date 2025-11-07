//! Einsum gradient computation.
//!
//! This module provides proper gradient computation for einsum operations
//! by determining the correct contraction pattern to backpropagate gradients.

use crate::{Scirs2Exec, Scirs2Tensor};
use tensorlogic_infer::{ExecutorError, TlExecutor};

/// Parse an einsum specification into input specs and output spec
/// Example: "ij,jk->ik" returns (["ij", "jk"], "ik")
fn parse_einsum_spec(spec: &str) -> Result<(Vec<String>, String), ExecutorError> {
    let parts: Vec<&str> = spec.split("->").collect();
    if parts.len() != 2 {
        return Err(ExecutorError::InvalidEinsumSpec(format!(
            "Einsum spec must contain exactly one '->': {}",
            spec
        )));
    }

    let input_specs: Vec<String> = parts[0].split(',').map(|s| s.trim().to_string()).collect();
    let output_spec = parts[1].trim().to_string();

    Ok((input_specs, output_spec))
}

/// Compute the einsum spec for the gradient of a specific input
///
/// For einsum operation: output = einsum(spec, [A, B, ...])
/// To compute grad_A, we need to contract output_grad with all other inputs
///
/// Algorithm:
/// 1. Parse the original spec to get input indices and output indices
/// 2. For the target input, find which indices it has
/// 3. Build a new spec that contracts output_grad with other inputs to produce target input shape
fn compute_gradient_spec(
    _original_spec: &str,
    target_input_idx: usize,
    input_specs: &[String],
    output_spec: &str,
) -> Result<String, ExecutorError> {
    if target_input_idx >= input_specs.len() {
        return Err(ExecutorError::InvalidEinsumSpec(format!(
            "Target input index {} out of bounds (total inputs: {})",
            target_input_idx,
            input_specs.len()
        )));
    }

    let target_spec = &input_specs[target_input_idx];

    // Build new spec: contract output_grad with all other inputs
    let mut new_input_specs = vec![output_spec.to_string()];

    // Add all inputs except the target
    for (idx, spec) in input_specs.iter().enumerate() {
        if idx != target_input_idx {
            new_input_specs.push(spec.clone());
        }
    }

    // The output should have the same indices as the target input
    let new_output_spec = target_spec.clone();

    // Build the final spec
    let inputs_str = new_input_specs.join(",");
    let grad_spec = format!("{}->{}", inputs_str, new_output_spec);

    Ok(grad_spec)
}

/// Compute einsum gradients for all inputs
///
/// Given an einsum operation and the gradient of the output,
/// compute the gradient for each input tensor.
pub fn compute_einsum_gradients(
    spec: &str,
    inputs: &[Scirs2Tensor],
    output_grad: &Scirs2Tensor,
    executor: &mut Scirs2Exec,
) -> Result<Vec<Scirs2Tensor>, ExecutorError> {
    let (input_specs, output_spec) = parse_einsum_spec(spec)?;

    if inputs.len() != input_specs.len() {
        return Err(ExecutorError::InvalidEinsumSpec(format!(
            "Number of inputs ({}) doesn't match spec ({})",
            inputs.len(),
            input_specs.len()
        )));
    }

    let mut gradients = Vec::new();

    // Compute gradient for each input
    for target_idx in 0..inputs.len() {
        let grad_spec = compute_gradient_spec(spec, target_idx, &input_specs, &output_spec)?;

        // Build input list for gradient computation: [output_grad, other inputs]
        let mut grad_inputs = vec![output_grad.clone()];
        for (idx, input) in inputs.iter().enumerate() {
            if idx != target_idx {
                grad_inputs.push(input.clone());
            }
        }

        // Compute the gradient via einsum
        let grad = executor.einsum(&grad_spec, &grad_inputs)?;
        gradients.push(grad);
    }

    Ok(gradients)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_einsum_spec() {
        let (inputs, output) = parse_einsum_spec("ij,jk->ik").unwrap();
        assert_eq!(inputs, vec!["ij", "jk"]);
        assert_eq!(output, "ik");

        let (inputs, output) = parse_einsum_spec("abc,bcd,cde->ae").unwrap();
        assert_eq!(inputs, vec!["abc", "bcd", "cde"]);
        assert_eq!(output, "ae");
    }

    #[test]
    fn test_compute_gradient_spec_matmul() {
        let input_specs = vec!["ij".to_string(), "jk".to_string()];
        let output_spec = "ik";

        // Gradient for first input (A in A @ B)
        let grad_spec_0 = compute_gradient_spec("ij,jk->ik", 0, &input_specs, output_spec).unwrap();
        // grad_A = grad_output @ B^T, which is "ik,jk->ij"
        assert_eq!(grad_spec_0, "ik,jk->ij");

        // Gradient for second input (B in A @ B)
        let grad_spec_1 = compute_gradient_spec("ij,jk->ik", 1, &input_specs, output_spec).unwrap();
        // grad_B = A^T @ grad_output, which is "ik,ij->jk"
        assert_eq!(grad_spec_1, "ik,ij->jk");
    }

    #[test]
    fn test_einsum_gradient_matmul() {
        let mut executor = Scirs2Exec::new();

        // Matrix multiplication: C = A @ B
        // A: 2x3, B: 3x4, C: 2x4
        let a = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Scirs2Exec::from_vec(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![3, 4],
        )
        .unwrap();

        // Forward pass
        let _c = executor
            .einsum("ij,jk->ik", &[a.clone(), b.clone()])
            .unwrap();

        // Backward pass with ones gradient
        let grad_c = Scirs2Exec::ones(vec![2, 4]);
        let grads = compute_einsum_gradients("ij,jk->ik", &[a, b], &grad_c, &mut executor).unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 3]); // grad_A shape
        assert_eq!(grads[1].shape(), &[3, 4]); // grad_B shape
    }

    #[test]
    fn test_einsum_gradient_elementwise() {
        let mut executor = Scirs2Exec::new();

        // Element-wise multiplication with explicit indices: C = A * B
        let a = Scirs2Exec::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let b = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // Forward: element-wise multiply (sum over no indices)
        let _c = executor
            .einsum("ij,ij->ij", &[a.clone(), b.clone()])
            .unwrap();

        // Backward
        let grad_c = Scirs2Exec::ones(vec![2, 2]);
        let grads =
            compute_einsum_gradients("ij,ij->ij", &[a.clone(), b.clone()], &grad_c, &mut executor)
                .unwrap();

        // grad_A = grad_C * B
        assert_eq!(grads[0].shape(), &[2, 2]);
        // Check first element: grad_A[0,0] should equal B[0,0] = 1.0
        assert!((grads[0][[0, 0]] - b[[0, 0]]).abs() < 1e-10);

        // grad_B = grad_C * A
        assert_eq!(grads[1].shape(), &[2, 2]);
        // Check first element: grad_B[0,0] should equal A[0,0] = 2.0
        assert!((grads[1][[0, 0]] - a[[0, 0]]).abs() < 1e-10);
    }
}
