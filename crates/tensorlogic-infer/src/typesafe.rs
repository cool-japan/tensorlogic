//! Type-safe tensor wrappers with compile-time shape checking.
//!
//! This module provides strongly-typed tensor wrappers that encode
//! shape information in the type system for compile-time verification.

use std::marker::PhantomData;

/// Type-level natural number for compile-time dimensions
pub trait Nat {
    fn to_usize() -> usize;
}

/// Zero dimension
pub struct Z;
impl Nat for Z {
    fn to_usize() -> usize {
        0
    }
}

/// Successor of a natural number
pub struct S<N: Nat>(PhantomData<N>);
impl<N: Nat> Nat for S<N> {
    fn to_usize() -> usize {
        N::to_usize() + 1
    }
}

/// Type aliases for common dimensions
pub type D1 = S<Z>;
pub type D2 = S<D1>;
pub type D3 = S<D2>;
pub type D4 = S<D3>;
pub type D5 = S<D4>;
pub type D6 = S<D5>;

/// Type-level dimension size
pub trait DimSize {
    fn size() -> usize;
}

/// Dynamic dimension size (runtime)
pub struct Dyn;
impl DimSize for Dyn {
    fn size() -> usize {
        0 // Runtime determined
    }
}

/// Static dimension size
pub struct Static<const N: usize>;
impl<const N: usize> DimSize for Static<N> {
    fn size() -> usize {
        N
    }
}

/// Type-safe tensor with compile-time rank
pub struct TypedTensor<T, R: Nat> {
    inner: T,
    shape: Vec<usize>,
    _rank: PhantomData<R>,
}

impl<T, R: Nat> TypedTensor<T, R> {
    /// Create a typed tensor with shape validation
    pub fn new(inner: T, shape: Vec<usize>) -> Result<Self, String> {
        if shape.len() != R::to_usize() {
            return Err(format!(
                "Shape length {} does not match rank {}",
                shape.len(),
                R::to_usize()
            ));
        }

        Ok(TypedTensor {
            inner,
            shape,
            _rank: PhantomData,
        })
    }

    /// Create without validation (unsafe)
    pub fn new_unchecked(inner: T, shape: Vec<usize>) -> Self {
        TypedTensor {
            inner,
            shape,
            _rank: PhantomData,
        }
    }

    /// Get inner tensor
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get mutable inner tensor
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Consume and get inner tensor
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get rank (compile-time known)
    pub fn rank() -> usize {
        R::to_usize()
    }

    /// Check if shape matches expected
    pub fn validate_shape(&self, expected: &[usize]) -> bool {
        self.shape == expected
    }
}

/// Scalar (rank 0)
pub type Scalar<T> = TypedTensor<T, Z>;

/// Vector (rank 1)
pub type Vector<T> = TypedTensor<T, D1>;

/// Matrix (rank 2)
pub type Matrix<T> = TypedTensor<T, D2>;

/// 3D Tensor (rank 3)
pub type Tensor3D<T> = TypedTensor<T, D3>;

/// 4D Tensor (rank 4)
pub type Tensor4D<T> = TypedTensor<T, D4>;

/// Type-safe tensor with both rank and shape
pub struct ShapedTensor<T, R: Nat, S: DimSize> {
    inner: T,
    _rank: PhantomData<R>,
    _shape: PhantomData<S>,
}

impl<T, R: Nat, S: DimSize> ShapedTensor<T, R, S> {
    pub fn new(inner: T) -> Self {
        ShapedTensor {
            inner,
            _rank: PhantomData,
            _shape: PhantomData,
        }
    }

    pub fn inner(&self) -> &T {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    pub fn into_inner(self) -> T {
        self.inner
    }

    pub fn rank() -> usize {
        R::to_usize()
    }

    pub fn size() -> usize {
        S::size()
    }
}

/// Trait for type-safe tensor operations
pub trait TypedTensorOps<T, R: Nat> {
    /// Element-wise addition (same shape)
    fn add(&self, other: &TypedTensor<T, R>) -> TypedTensor<T, R>;

    /// Element-wise multiplication (same shape)
    fn mul(&self, other: &TypedTensor<T, R>) -> TypedTensor<T, R>;

    /// Scalar multiplication
    fn scale(&self, scalar: f64) -> TypedTensor<T, R>;
}

/// Matrix operations (rank 2 specific)
pub trait MatrixOps<T> {
    /// Matrix multiplication (M x N) * (N x K) -> (M x K)
    fn matmul(&self, other: &Matrix<T>) -> Result<Matrix<T>, String>;

    /// Transpose (M x N) -> (N x M)
    fn transpose(&self) -> Matrix<T>;
}

/// Type-safe einsum specification
pub struct EinsumSpec<Input, Output> {
    spec_string: String,
    _input: PhantomData<Input>,
    _output: PhantomData<Output>,
}

impl<Input, Output> EinsumSpec<Input, Output> {
    pub fn new(spec: String) -> Self {
        EinsumSpec {
            spec_string: spec,
            _input: PhantomData,
            _output: PhantomData,
        }
    }

    pub fn spec(&self) -> &str {
        &self.spec_string
    }
}

/// Typed input container for execution
pub struct TypedInputs<T> {
    tensors: Vec<T>,
}

impl<T> TypedInputs<T> {
    pub fn new() -> Self {
        TypedInputs {
            tensors: Vec::new(),
        }
    }

    pub fn with(mut self, tensor: T) -> Self {
        self.tensors.push(tensor);
        self
    }

    pub fn tensors(&self) -> &[T] {
        &self.tensors
    }

    pub fn into_vec(self) -> Vec<T> {
        self.tensors
    }
}

impl<T> Default for TypedInputs<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Typed output container from execution
pub struct TypedOutputs<T> {
    tensors: Vec<T>,
}

impl<T> TypedOutputs<T> {
    pub fn new(tensors: Vec<T>) -> Self {
        TypedOutputs { tensors }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.tensors.get(index)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.tensors
    }
}

/// Shape constraint for compile-time checking
pub trait ShapeConstraint<R: Nat> {
    fn check_shape(shape: &[usize]) -> bool;
}

/// Fixed shape constraint
pub struct FixedShape<const N: usize>;

impl<const N: usize, R: Nat> ShapeConstraint<R> for FixedShape<N> {
    fn check_shape(shape: &[usize]) -> bool {
        shape.len() == R::to_usize() && shape.iter().all(|&d| d == N)
    }
}

/// Broadcasting-compatible shape constraint
pub struct BroadcastShape;

impl<R: Nat> ShapeConstraint<R> for BroadcastShape {
    fn check_shape(shape: &[usize]) -> bool {
        shape.len() == R::to_usize()
    }
}

/// Type-safe batch of tensors
pub struct TypedBatch<T, R: Nat> {
    tensors: Vec<TypedTensor<T, R>>,
}

impl<T, R: Nat> TypedBatch<T, R> {
    pub fn new() -> Self {
        TypedBatch {
            tensors: Vec::new(),
        }
    }

    pub fn with(mut self, tensor: TypedTensor<T, R>) -> Self {
        self.tensors.push(tensor);
        self
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&TypedTensor<T, R>> {
        self.tensors.get(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = &TypedTensor<T, R>> {
        self.tensors.iter()
    }
}

impl<T, R: Nat> Default for TypedBatch<T, R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for type-safe tensor construction
pub struct TensorBuilder<T> {
    inner: Option<T>,
    shape: Vec<usize>,
}

impl<T> TensorBuilder<T> {
    pub fn new(inner: T) -> Self {
        TensorBuilder {
            inner: Some(inner),
            shape: Vec::new(),
        }
    }

    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = shape;
        self
    }

    pub fn build_scalar(self) -> Result<Scalar<T>, String> {
        let inner = self.inner.ok_or("Missing inner tensor")?;
        if !self.shape.is_empty() {
            return Err("Scalar must have empty shape".to_string());
        }
        Scalar::new(inner, vec![])
    }

    pub fn build_vector(self) -> Result<Vector<T>, String> {
        let inner = self.inner.ok_or("Missing inner tensor")?;
        if self.shape.len() != 1 {
            return Err("Vector must have rank 1".to_string());
        }
        Vector::new(inner, self.shape)
    }

    pub fn build_matrix(self) -> Result<Matrix<T>, String> {
        let inner = self.inner.ok_or("Missing inner tensor")?;
        if self.shape.len() != 2 {
            return Err("Matrix must have rank 2".to_string());
        }
        Matrix::new(inner, self.shape)
    }

    pub fn build<R: Nat>(self) -> Result<TypedTensor<T, R>, String> {
        let inner = self.inner.ok_or("Missing inner tensor")?;
        TypedTensor::new(inner, self.shape)
    }
}

/// Type-safe dimension information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim<const N: usize>;

impl<const N: usize> Dim<N> {
    pub const fn size() -> usize {
        N
    }

    pub fn matches(actual: usize) -> bool {
        actual == N
    }
}

/// Helper for dimension arithmetic (marker trait)
pub trait DimOp {
    // Marker trait for dimension operations
    // Actual operations would require unstable const generics features
}

/// Dimension multiplication (placeholder)
pub struct DimMul<A, B>(PhantomData<(A, B)>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nat_types() {
        assert_eq!(Z::to_usize(), 0);
        assert_eq!(D1::to_usize(), 1);
        assert_eq!(D2::to_usize(), 2);
        assert_eq!(D3::to_usize(), 3);
        assert_eq!(D4::to_usize(), 4);
    }

    #[test]
    fn test_dim_size() {
        assert_eq!(Static::<10>::size(), 10);
        assert_eq!(Static::<256>::size(), 256);
        assert_eq!(Dyn::size(), 0);
    }

    #[test]
    fn test_typed_tensor_creation() {
        let tensor: Vector<f64> = TypedTensor::new(1.0, vec![10]).unwrap();
        assert_eq!(tensor.shape(), &[10]);
        assert_eq!(Vector::<f64>::rank(), 1);

        let matrix: Matrix<f64> = TypedTensor::new(2.0, vec![10, 20]).unwrap();
        assert_eq!(matrix.shape(), &[10, 20]);
        assert_eq!(Matrix::<f64>::rank(), 2);
    }

    #[test]
    fn test_typed_tensor_validation() {
        let result: Result<Vector<f64>, _> = TypedTensor::new(1.0, vec![10, 20]);
        assert!(result.is_err()); // Wrong rank

        let result: Result<Matrix<f64>, _> = TypedTensor::new(2.0, vec![10]);
        assert!(result.is_err()); // Wrong rank
    }

    #[test]
    fn test_typed_tensor_inner() {
        let tensor: Vector<i32> = TypedTensor::new(42, vec![5]).unwrap();
        assert_eq!(*tensor.inner(), 42);

        let inner = tensor.into_inner();
        assert_eq!(inner, 42);
    }

    #[test]
    fn test_shaped_tensor() {
        let tensor: ShapedTensor<f64, D2, Static<10>> = ShapedTensor::new(2.5);
        assert_eq!(ShapedTensor::<f64, D2, Static<10>>::rank(), 2);
        assert_eq!(ShapedTensor::<f64, D2, Static<10>>::size(), 10);
        assert_eq!(*tensor.inner(), 2.5);
    }

    #[test]
    fn test_typed_inputs() {
        let inputs: TypedInputs<i32> = TypedInputs::new().with(1).with(2).with(3);

        assert_eq!(inputs.tensors().len(), 3);
        assert_eq!(inputs.tensors(), &[1, 2, 3]);
    }

    #[test]
    fn test_typed_outputs() {
        let outputs: TypedOutputs<i32> = TypedOutputs::new(vec![1, 2, 3]);

        assert_eq!(outputs.len(), 3);
        assert!(!outputs.is_empty());
        assert_eq!(outputs.get(0), Some(&1));
        assert_eq!(outputs.get(1), Some(&2));
        assert_eq!(outputs.get(2), Some(&3));
        assert_eq!(outputs.get(3), None);
    }

    #[test]
    fn test_einsum_spec() {
        let spec: EinsumSpec<(Matrix<f64>, Matrix<f64>), Matrix<f64>> =
            EinsumSpec::new("ij,jk->ik".to_string());
        assert_eq!(spec.spec(), "ij,jk->ik");
    }

    #[test]
    fn test_typed_batch() {
        let mut batch: TypedBatch<i32, D1> = TypedBatch::new();
        assert!(batch.is_empty());

        let tensor1: Vector<i32> = TypedTensor::new(1, vec![5]).unwrap();
        let tensor2: Vector<i32> = TypedTensor::new(2, vec![5]).unwrap();

        batch = batch.with(tensor1).with(tensor2);

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());

        let first = batch.get(0).unwrap();
        assert_eq!(*first.inner(), 1);
    }

    #[test]
    fn test_tensor_builder() {
        let scalar: Scalar<f64> = TensorBuilder::new(2.5)
            .with_shape(vec![])
            .build_scalar()
            .unwrap();
        assert_eq!(*scalar.inner(), 2.5);

        let vector: Vector<f64> = TensorBuilder::new(2.71)
            .with_shape(vec![10])
            .build_vector()
            .unwrap();
        assert_eq!(vector.shape(), &[10]);

        let matrix: Matrix<f64> = TensorBuilder::new(1.41)
            .with_shape(vec![3, 4])
            .build_matrix()
            .unwrap();
        assert_eq!(matrix.shape(), &[3, 4]);
    }

    #[test]
    fn test_tensor_builder_errors() {
        let result = TensorBuilder::new(1.0).with_shape(vec![10]).build_scalar();
        assert!(result.is_err()); // Scalar can't have shape

        let result = TensorBuilder::new(1.0)
            .with_shape(vec![10, 20])
            .build_vector();
        assert!(result.is_err()); // Vector must be rank 1

        let result = TensorBuilder::new(1.0).with_shape(vec![10]).build_matrix();
        assert!(result.is_err()); // Matrix must be rank 2
    }

    #[test]
    fn test_dim() {
        assert_eq!(Dim::<10>::size(), 10);
        assert_eq!(Dim::<256>::size(), 256);

        assert!(Dim::<10>::matches(10));
        assert!(!Dim::<10>::matches(20));
    }

    #[test]
    fn test_shape_validation() {
        let tensor: Vector<i32> = TypedTensor::new(42, vec![10]).unwrap();
        assert!(tensor.validate_shape(&[10]));
        assert!(!tensor.validate_shape(&[20]));
        assert!(!tensor.validate_shape(&[10, 10]));
    }
}
