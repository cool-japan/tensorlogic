//! Eager mode automatic differentiation.
//!
//! This module provides eager execution with automatic differentiation,
//! similar to PyTorch's autograd or TensorFlow's eager execution.
//!
//! Unlike `TlAutodiff` which requires a full `EinsumGraph`, eager mode
//! computes gradients by building a dynamic computation graph as operations
//! are executed.
//!
//! # Example
//!
//! ```ignore
//! use tensorlogic_infer::eager::{Variable, TlEagerAutodiff};
//!
//! // Create variables
//! let x = Variable::new(tensor_x, true); // requires_grad = true
//! let y = Variable::new(tensor_y, true);
//!
//! // Execute operations eagerly
//! let z = executor.eager_add(&x, &y)?;
//! let loss = executor.eager_reduce_sum(&z)?;
//!
//! // Compute gradients
//! let grads = executor.eager_backward(&loss)?;
//! ```

use crate::ops::{ElemOp, ReduceOp};
use crate::traits::TlExecutor;
use std::collections::HashMap;

/// A variable in the eager execution graph.
///
/// Wraps a tensor and tracks whether gradients should be computed for it.
#[derive(Debug, Clone)]
pub struct Variable<T> {
    /// The tensor data
    pub tensor: T,
    /// Whether this variable requires gradient computation
    pub requires_grad: bool,
    /// Unique ID for gradient tracking
    pub id: usize,
}

impl<T> Variable<T> {
    /// Create a new variable.
    pub fn new(tensor: T, requires_grad: bool) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Variable {
            tensor,
            requires_grad,
            id,
        }
    }

    /// Create a constant (no gradient).
    pub fn constant(tensor: T) -> Self {
        Self::new(tensor, false)
    }

    /// Get a reference to the tensor.
    pub fn tensor(&self) -> &T {
        &self.tensor
    }

    /// Check if this variable requires gradients.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

/// Gradient storage for a variable.
#[derive(Debug, Clone)]
pub struct VariableGrad<T> {
    /// The gradient tensor
    pub grad: T,
    /// Whether the gradient has been computed
    pub computed: bool,
}

impl<T> VariableGrad<T> {
    /// Create a new gradient container.
    pub fn new(grad: T) -> Self {
        VariableGrad {
            grad,
            computed: true,
        }
    }

    /// Create an uncomputed gradient placeholder.
    pub fn placeholder(grad: T) -> Self {
        VariableGrad {
            grad,
            computed: false,
        }
    }
}

/// Tape for recording eager operations and their gradients.
///
/// The tape stores the computation graph as operations are executed,
/// enabling backward pass for gradient computation.
#[derive(Debug)]
pub struct EagerTape<T> {
    /// Map from variable ID to gradient
    gradients: HashMap<usize, VariableGrad<T>>,
    /// Operations recorded on the tape
    operations: Vec<EagerOp<T>>,
}

impl<T> EagerTape<T> {
    /// Create a new empty tape.
    pub fn new() -> Self {
        EagerTape {
            gradients: HashMap::new(),
            operations: Vec::new(),
        }
    }

    /// Record an operation on the tape.
    pub fn record_op(&mut self, op: EagerOp<T>) {
        self.operations.push(op);
    }

    /// Set gradient for a variable.
    pub fn set_gradient(&mut self, var_id: usize, grad: VariableGrad<T>) {
        self.gradients.insert(var_id, grad);
    }

    /// Get gradient for a variable.
    pub fn get_gradient(&self, var_id: usize) -> Option<&VariableGrad<T>> {
        self.gradients.get(&var_id)
    }

    /// Get all gradients.
    pub fn gradients(&self) -> &HashMap<usize, VariableGrad<T>> {
        &self.gradients
    }

    /// Get all operations.
    pub fn operations(&self) -> &[EagerOp<T>] {
        &self.operations
    }

    /// Clear the tape.
    pub fn clear(&mut self) {
        self.gradients.clear();
        self.operations.clear();
    }

    /// Number of operations recorded.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if tape is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

impl<T> Default for EagerTape<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// An operation recorded in the eager execution tape.
#[derive(Debug, Clone)]
pub enum EagerOp<T> {
    /// Element-wise unary operation
    ElemUnary {
        op: ElemOp,
        input: Variable<T>,
        output: Variable<T>,
    },
    /// Element-wise binary operation
    ElemBinary {
        op: ElemOp,
        left: Variable<T>,
        right: Variable<T>,
        output: Variable<T>,
    },
    /// Reduction operation
    Reduce {
        op: ReduceOp,
        input: Variable<T>,
        axes: Vec<usize>,
        output: Variable<T>,
    },
    /// Einsum operation
    Einsum {
        spec: String,
        inputs: Vec<Variable<T>>,
        output: Variable<T>,
    },
}

/// Trait for eager execution with automatic differentiation.
///
/// This trait extends `TlExecutor` with eager autodiff capabilities,
/// allowing gradient computation without building a full graph upfront.
pub trait TlEagerAutodiff: TlExecutor {
    /// Execute element-wise unary operation eagerly.
    ///
    /// Returns a new variable containing the result. If the input requires
    /// gradients, the operation is recorded on the tape.
    fn eager_elem_op(
        &mut self,
        op: ElemOp,
        x: &Variable<Self::Tensor>,
    ) -> Result<Variable<Self::Tensor>, Self::Error>;

    /// Execute element-wise binary operation eagerly.
    fn eager_elem_op_binary(
        &mut self,
        op: ElemOp,
        x: &Variable<Self::Tensor>,
        y: &Variable<Self::Tensor>,
    ) -> Result<Variable<Self::Tensor>, Self::Error>;

    /// Execute reduction operation eagerly.
    fn eager_reduce(
        &mut self,
        op: ReduceOp,
        x: &Variable<Self::Tensor>,
        axes: &[usize],
    ) -> Result<Variable<Self::Tensor>, Self::Error>;

    /// Execute einsum operation eagerly.
    fn eager_einsum(
        &mut self,
        spec: &str,
        inputs: &[Variable<Self::Tensor>],
    ) -> Result<Variable<Self::Tensor>, Self::Error>;

    /// Compute gradients for all variables with respect to the output.
    ///
    /// This performs backpropagation through the recorded operations
    /// to compute gradients.
    fn eager_backward(
        &mut self,
        output: &Variable<Self::Tensor>,
    ) -> Result<EagerTape<Self::Tensor>, Self::Error>;

    /// Create a new empty tape for recording operations.
    fn create_tape(&self) -> EagerTape<Self::Tensor> {
        EagerTape::new()
    }
}

/// Convenience methods for common operations.
pub trait EagerOps: TlEagerAutodiff {
    /// Add two variables.
    fn eager_add(
        &mut self,
        x: &Variable<Self::Tensor>,
        y: &Variable<Self::Tensor>,
    ) -> Result<Variable<Self::Tensor>, Self::Error> {
        self.eager_elem_op_binary(ElemOp::Add, x, y)
    }

    /// Multiply two variables.
    fn eager_mul(
        &mut self,
        x: &Variable<Self::Tensor>,
        y: &Variable<Self::Tensor>,
    ) -> Result<Variable<Self::Tensor>, Self::Error> {
        self.eager_elem_op_binary(ElemOp::Multiply, x, y)
    }

    /// Subtract two variables.
    fn eager_sub(
        &mut self,
        x: &Variable<Self::Tensor>,
        y: &Variable<Self::Tensor>,
    ) -> Result<Variable<Self::Tensor>, Self::Error> {
        self.eager_elem_op_binary(ElemOp::Subtract, x, y)
    }

    /// Apply Relu activation.
    fn eager_relu(
        &mut self,
        x: &Variable<Self::Tensor>,
    ) -> Result<Variable<Self::Tensor>, Self::Error> {
        self.eager_elem_op(ElemOp::Relu, x)
    }

    /// Apply sigmoid activation.
    fn eager_sigmoid(
        &mut self,
        x: &Variable<Self::Tensor>,
    ) -> Result<Variable<Self::Tensor>, Self::Error> {
        self.eager_elem_op(ElemOp::Sigmoid, x)
    }

    /// Apply one-minus operation (1 - x).
    fn eager_one_minus(
        &mut self,
        x: &Variable<Self::Tensor>,
    ) -> Result<Variable<Self::Tensor>, Self::Error> {
        self.eager_elem_op(ElemOp::OneMinus, x)
    }

    /// Sum reduction along axes.
    fn eager_sum(
        &mut self,
        x: &Variable<Self::Tensor>,
        axes: &[usize],
    ) -> Result<Variable<Self::Tensor>, Self::Error> {
        self.eager_reduce(ReduceOp::Sum, x, axes)
    }

    /// Mean reduction along axes.
    fn eager_mean(
        &mut self,
        x: &Variable<Self::Tensor>,
        axes: &[usize],
    ) -> Result<Variable<Self::Tensor>, Self::Error> {
        self.eager_reduce(ReduceOp::Mean, x, axes)
    }

    /// Max reduction along axes.
    fn eager_max(
        &mut self,
        x: &Variable<Self::Tensor>,
        axes: &[usize],
    ) -> Result<Variable<Self::Tensor>, Self::Error> {
        self.eager_reduce(ReduceOp::Max, x, axes)
    }
}

/// Automatic implementation of EagerOps for any type implementing TlEagerAutodiff
impl<T: TlEagerAutodiff> EagerOps for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_creation() {
        let tensor = vec![1.0, 2.0, 3.0];
        let var = Variable::new(tensor.clone(), true);

        assert_eq!(var.tensor, tensor);
        assert!(var.requires_grad);
        // ID is assigned sequentially starting from 0
        // Just verify it's a valid ID (any value is fine)
    }

    #[test]
    fn test_variable_constant() {
        let tensor = vec![1.0, 2.0, 3.0];
        let var = Variable::constant(tensor.clone());

        assert_eq!(var.tensor, tensor);
        assert!(!var.requires_grad);
    }

    #[test]
    fn test_variable_unique_ids() {
        let var1 = Variable::new(vec![1.0], true);
        let var2 = Variable::new(vec![2.0], true);

        assert_ne!(var1.id, var2.id);
    }

    #[test]
    fn test_eager_tape_creation() {
        let tape: EagerTape<Vec<f64>> = EagerTape::new();

        assert!(tape.is_empty());
        assert_eq!(tape.len(), 0);
        assert_eq!(tape.gradients().len(), 0);
    }

    #[test]
    fn test_eager_tape_set_gradient() {
        let mut tape = EagerTape::new();
        let grad = VariableGrad::new(vec![1.0, 2.0, 3.0]);

        tape.set_gradient(1, grad);

        assert!(tape.get_gradient(1).is_some());
        assert!(tape.get_gradient(2).is_none());
    }

    #[test]
    fn test_eager_tape_clear() {
        let mut tape = EagerTape::new();
        tape.set_gradient(1, VariableGrad::new(vec![1.0]));

        assert!(!tape.is_empty() || !tape.gradients().is_empty());

        tape.clear();

        assert!(tape.is_empty());
        assert_eq!(tape.gradients().len(), 0);
    }

    #[test]
    fn test_variable_grad_creation() {
        let grad = VariableGrad::new(vec![1.0, 2.0]);

        assert!(grad.computed);
        assert_eq!(grad.grad, vec![1.0, 2.0]);
    }

    #[test]
    fn test_variable_grad_placeholder() {
        let grad = VariableGrad::placeholder(vec![0.0]);

        assert!(!grad.computed);
    }

    #[test]
    fn test_eager_op_variants() {
        let var1 = Variable::new(vec![1.0], true);
        let var2 = Variable::new(vec![2.0], true);
        let var3 = Variable::new(vec![3.0], true);

        // Test ElemUnary variant
        let _op1 = EagerOp::ElemUnary {
            op: ElemOp::OneMinus,
            input: var1.clone(),
            output: var3.clone(),
        };

        // Test ElemBinary variant
        let _op2 = EagerOp::ElemBinary {
            op: ElemOp::Add,
            left: var1.clone(),
            right: var2.clone(),
            output: var3.clone(),
        };

        // Test Reduce variant
        let _op3 = EagerOp::Reduce {
            op: ReduceOp::Sum,
            input: var1.clone(),
            axes: vec![0],
            output: var3.clone(),
        };

        // Test Einsum variant
        let _op4 = EagerOp::Einsum {
            spec: "ij,jk->ik".to_string(),
            inputs: vec![var1.clone(), var2.clone()],
            output: var3.clone(),
        };
    }

    #[test]
    fn test_tape_record_op() {
        let mut tape = EagerTape::new();
        let var1 = Variable::new(vec![1.0], true);
        let var2 = Variable::new(vec![2.0], true);

        let op = EagerOp::ElemBinary {
            op: ElemOp::Add,
            left: var1,
            right: var2.clone(),
            output: var2,
        };

        tape.record_op(op);

        assert_eq!(tape.len(), 1);
        assert!(!tape.is_empty());
    }

    #[test]
    fn test_variable_methods() {
        let tensor = vec![1.0, 2.0, 3.0];
        let var = Variable::new(tensor.clone(), true);

        assert_eq!(var.tensor(), &tensor);
        assert!(var.requires_grad());
    }

    #[test]
    fn test_tape_default() {
        let tape: EagerTape<Vec<f64>> = EagerTape::default();

        assert!(tape.is_empty());
    }
}
