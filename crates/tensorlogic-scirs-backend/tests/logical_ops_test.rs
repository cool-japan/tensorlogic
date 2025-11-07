//! Integration tests for extended logical operations.

use scirs2_core::ndarray::{ArrayD, IxDyn};
use tensorlogic_infer::{ElemOp, ReduceOp, TlExecutor};
use tensorlogic_scirs_backend::Scirs2Exec;

fn create_tensor(data: Vec<f64>, shape: Vec<usize>) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap()
}

fn assert_tensor_eq(actual: &[f64], expected: &[f64], epsilon: f64) {
    assert_eq!(actual.len(), expected.len(), "Tensor length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < epsilon,
            "Mismatch at index {}: expected {}, got {} (diff: {})",
            i,
            e,
            a,
            (a - e).abs()
        );
    }
}

#[test]
fn test_or_max_operation() {
    let mut executor = Scirs2Exec::new();

    let x = create_tensor(vec![0.3, 0.7, 0.1, 0.9], vec![4]);
    let y = create_tensor(vec![0.5, 0.2, 0.8, 0.4], vec![4]);

    let result = executor.elem_op_binary(ElemOp::OrMax, &x, &y).unwrap();

    // OR(max) should return max(x, y)
    let expected = vec![0.5, 0.7, 0.8, 0.9];
    assert_tensor_eq(result.as_slice().unwrap(), &expected, 1e-10);
}

#[test]
fn test_or_prob_sum_operation() {
    let mut executor = Scirs2Exec::new();

    let x = create_tensor(vec![0.3, 0.7, 0.0, 1.0], vec![4]);
    let y = create_tensor(vec![0.5, 0.2, 0.0, 0.0], vec![4]);

    let result = executor.elem_op_binary(ElemOp::OrProbSum, &x, &y).unwrap();

    // OR(prob): a + b - ab
    // 0.3 + 0.5 - 0.15 = 0.65
    // 0.7 + 0.2 - 0.14 = 0.76
    // 0.0 + 0.0 - 0.0 = 0.0
    // 1.0 + 0.0 - 0.0 = 1.0
    let expected = vec![0.65, 0.76, 0.0, 1.0];
    assert_tensor_eq(result.as_slice().unwrap(), &expected, 1e-10);
}

#[test]
fn test_nand_operation() {
    let mut executor = Scirs2Exec::new();

    let x = create_tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4]);
    let y = create_tensor(vec![0.0, 1.0, 0.0, 1.0], vec![4]);

    let result = executor.elem_op_binary(ElemOp::Nand, &x, &y).unwrap();

    // NAND: 1 - (a * b)
    // 1 - 0 = 1.0
    // 1 - 0 = 1.0
    // 1 - 0 = 1.0
    // 1 - 1 = 0.0
    let expected = vec![1.0, 1.0, 1.0, 0.0];
    assert_tensor_eq(result.as_slice().unwrap(), &expected, 1e-10);
}

#[test]
fn test_nor_operation() {
    let mut executor = Scirs2Exec::new();

    let x = create_tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4]);
    let y = create_tensor(vec![0.0, 1.0, 0.0, 1.0], vec![4]);

    let result = executor.elem_op_binary(ElemOp::Nor, &x, &y).unwrap();

    // NOR: 1 - max(a, b)
    // 1 - 0 = 1.0
    // 1 - 1 = 0.0
    // 1 - 1 = 0.0
    // 1 - 1 = 0.0
    let expected = vec![1.0, 0.0, 0.0, 0.0];
    assert_tensor_eq(result.as_slice().unwrap(), &expected, 1e-10);
}

#[test]
fn test_xor_operation() {
    let mut executor = Scirs2Exec::new();

    let x = create_tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4]);
    let y = create_tensor(vec![0.0, 1.0, 0.0, 1.0], vec![4]);

    let result = executor.elem_op_binary(ElemOp::Xor, &x, &y).unwrap();

    // XOR (soft): a + b - 2ab
    // 0 + 0 - 0 = 0.0
    // 0 + 1 - 0 = 1.0
    // 1 + 0 - 0 = 1.0
    // 1 + 1 - 2 = 0.0
    let expected = vec![0.0, 1.0, 1.0, 0.0];
    assert_tensor_eq(result.as_slice().unwrap(), &expected, 1e-10);
}

#[test]
fn test_product_reduction() {
    let mut executor = Scirs2Exec::new();

    let x = create_tensor(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);

    let result = executor.reduce(ReduceOp::Product, &x, &[0]).unwrap();

    // Product over axis 0: [2*4, 3*5] = [8, 15]
    let expected = vec![8.0, 15.0];
    assert_tensor_eq(result.as_slice().unwrap(), &expected, 1e-10);
}

#[test]
fn test_product_reduction_all_axes() {
    let mut executor = Scirs2Exec::new();

    let x = create_tensor(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);

    let result = executor.reduce(ReduceOp::Product, &x, &[0, 1]).unwrap();

    // Product of all elements: 2*3*4*5 = 120
    let value = result.iter().next().unwrap();
    assert!(
        (*value - 120.0).abs() < 1e-10,
        "Expected 120.0, got {}",
        value
    );
}
