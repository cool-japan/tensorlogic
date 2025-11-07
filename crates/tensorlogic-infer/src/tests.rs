//! Unit tests for dummy tensor and executor.

use crate::dummy_executor::DummyExecutor;
use crate::dummy_tensor::DummyTensor;
use crate::ops::{ElemOp, ReduceOp};
use crate::traits::TlExecutor;

#[test]
fn test_dummy_tensor_creation() {
    let t = DummyTensor::new("test", vec![2, 3]);
    assert_eq!(t.shape, vec![2, 3]);
    assert_eq!(t.size(), 6);
    assert_eq!(t.data.len(), 6);

    let t2 = DummyTensor::ones("ones", vec![3, 2]);
    assert!(t2.data.iter().all(|&x| x == 1.0));
}

#[test]
fn test_dummy_executor_basic() {
    let mut exec = DummyExecutor::new();

    let t1 = DummyTensor::with_data("A", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let t2 = DummyTensor::with_data("B", vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    let result = exec.elem_op_binary(ElemOp::Multiply, &t1, &t2).unwrap();
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn test_elem_ops() {
    let mut exec = DummyExecutor::new();
    let t = DummyTensor::with_data("test", vec![4], vec![-1.0, 0.0, 1.0, 2.0]);

    let relu = exec.elem_op(ElemOp::Relu, &t).unwrap();
    assert_eq!(relu.data, vec![0.0, 0.0, 1.0, 2.0]);

    let one_minus = exec.elem_op(ElemOp::OneMinus, &t).unwrap();
    assert_eq!(one_minus.data, vec![2.0, 1.0, 0.0, -1.0]);
}

#[test]
fn test_reduce_ops() {
    let mut exec = DummyExecutor::new();
    let t = DummyTensor::with_data("test", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let sum = exec.reduce(ReduceOp::Sum, &t, &[0]).unwrap();
    assert_eq!(sum.shape, vec![3]);

    let max = exec.reduce(ReduceOp::Max, &t, &[1]).unwrap();
    assert_eq!(max.shape, vec![2]);
}

#[test]
fn test_einsum_basic() {
    let mut exec = DummyExecutor::new();
    let t1 = DummyTensor::ones("A", vec![2, 3]);
    let t2 = DummyTensor::ones("B", vec![3, 2]);

    let result = exec.einsum("ij,jk->ik", &[t1, t2]).unwrap();
    assert_eq!(result.shape, vec![2, 3]);
}
