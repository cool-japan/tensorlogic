//! Unit tests for SciRS2 backend.

use tensorlogic_infer::{ElemOp, ReduceOp, TlExecutor};

use crate::Scirs2Exec;

#[test]
fn test_scirs2_tensor_creation() {
    let tensor = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.len(), 4);
}

#[test]
fn test_scirs2_elem_ops() {
    let mut exec = Scirs2Exec::new();
    let tensor = Scirs2Exec::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).unwrap();

    let relu = exec.elem_op(ElemOp::Relu, &tensor).unwrap();
    assert_eq!(relu[[0]], 0.0);
    assert_eq!(relu[[1]], 0.0);
    assert_eq!(relu[[2]], 1.0);
    assert_eq!(relu[[3]], 2.0);

    let one_minus = exec.elem_op(ElemOp::OneMinus, &tensor).unwrap();
    assert_eq!(one_minus[[0]], 2.0);
    assert_eq!(one_minus[[1]], 1.0);
    assert_eq!(one_minus[[2]], 0.0);
    assert_eq!(one_minus[[3]], -1.0);
}

#[test]
fn test_scirs2_binary_ops() {
    let mut exec = Scirs2Exec::new();
    let a = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Scirs2Exec::from_vec(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]).unwrap();

    let mult = exec.elem_op_binary(ElemOp::Multiply, &a, &b).unwrap();
    assert_eq!(mult[[0, 0]], 2.0);
    assert_eq!(mult[[0, 1]], 4.0);
    assert_eq!(mult[[1, 0]], 6.0);
    assert_eq!(mult[[1, 1]], 8.0);

    let sub = exec.elem_op_binary(ElemOp::Subtract, &a, &b).unwrap();
    assert_eq!(sub[[0, 0]], -1.0);
    assert_eq!(sub[[0, 1]], 0.0);
    assert_eq!(sub[[1, 0]], 1.0);
    assert_eq!(sub[[1, 1]], 2.0);
}

#[test]
fn test_scirs2_reduce_ops() {
    let mut exec = Scirs2Exec::new();
    let tensor = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

    let sum_axis0 = exec.reduce(ReduceOp::Sum, &tensor, &[0]).unwrap();
    assert_eq!(sum_axis0.shape(), &[3]);

    let sum_axis1 = exec.reduce(ReduceOp::Sum, &tensor, &[1]).unwrap();
    assert_eq!(sum_axis1.shape(), &[2]);

    let max_axis0 = exec.reduce(ReduceOp::Max, &tensor, &[0]).unwrap();
    assert_eq!(max_axis0.shape(), &[3]);
}

#[test]
fn test_scirs2_einsum_matmul() {
    let mut exec = Scirs2Exec::new();

    let a = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Scirs2Exec::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    let result = exec.einsum("ij,jk->ik", &[a, b]).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_backward_pass_basic() {
    use tensorlogic_infer::TlAutodiff;
    use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

    let mut exec = Scirs2Exec::new();

    // Create a simple graph: unary operation (ReLU)
    let mut graph = EinsumGraph::new();
    graph.tensors.push("input[a]".to_string());
    graph.tensors.push("output[a]".to_string());
    graph.nodes.push(EinsumNode {
        op: OpType::ElemUnary {
            op: "relu".to_string(),
        },
        inputs: vec![0],
        outputs: vec![1],
        metadata: None,
    });
    graph.outputs.push(1);

    // Add input tensor
    let input = Scirs2Exec::from_vec(vec![-1.0, 2.0, -3.0, 4.0], vec![4]).unwrap();
    exec.add_tensor("input[a]".to_string(), input);

    // Forward pass
    let output = exec.forward(&graph).unwrap();
    assert_eq!(output.shape(), &[4]);

    // Backward pass with ones gradient
    let loss_grad = Scirs2Exec::ones(vec![4]);
    let tape = exec.backward(&graph, &loss_grad).unwrap();

    // Should have gradients
    assert!(!tape.is_empty());
}

#[test]
fn test_backward_pass_binary() {
    use tensorlogic_infer::TlAutodiff;
    use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

    let mut exec = Scirs2Exec::new();

    // Create a graph: binary operation (subtract)
    let mut graph = EinsumGraph::new();
    graph.tensors.push("a[i]".to_string());
    graph.tensors.push("b[i]".to_string());
    graph.tensors.push("output[i]".to_string());
    graph.nodes.push(EinsumNode {
        op: OpType::ElemBinary {
            op: "subtract".to_string(),
        },
        inputs: vec![0, 1],
        outputs: vec![2],
        metadata: None,
    });
    graph.outputs.push(2);

    // Add input tensors
    let a = Scirs2Exec::from_vec(vec![5.0, 6.0], vec![2]).unwrap();
    let b = Scirs2Exec::from_vec(vec![3.0, 2.0], vec![2]).unwrap();
    exec.add_tensor("a[i]".to_string(), a);
    exec.add_tensor("b[i]".to_string(), b);

    // Forward pass
    let output = exec.forward(&graph).unwrap();
    assert_eq!(output.shape(), &[2]);

    // Backward pass
    let loss_grad = Scirs2Exec::ones(vec![2]);
    let tape = exec.backward(&graph, &loss_grad).unwrap();

    // Should have gradients for both inputs
    assert!(!tape.is_empty());
}
