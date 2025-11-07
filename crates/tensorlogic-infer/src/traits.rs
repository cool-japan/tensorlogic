//! Core execution traits for TensorLogic engines.

use tensorlogic_ir::EinsumGraph;

use crate::ops::{ElemOp, ReduceOp};

/// Core tensor execution interface.
///
/// Implementations provide the fundamental tensor operations required
/// for executing compiled TensorLogic programs.
pub trait TlExecutor {
    type Tensor;
    type Error;

    /// Execute an einsum operation on input tensors.
    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Result<Self::Tensor, Self::Error>;

    /// Apply an element-wise unary operation.
    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    /// Apply an element-wise binary operation.
    fn elem_op_binary(
        &mut self,
        op: ElemOp,
        x: &Self::Tensor,
        y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Reduce a tensor along specified axes.
    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &Self::Tensor,
        axes: &[usize],
    ) -> Result<Self::Tensor, Self::Error>;
}

/// Automatic differentiation interface.
///
/// Extends `TlExecutor` with forward/backward pass capabilities for training.
pub trait TlAutodiff: TlExecutor {
    type Tape;

    /// Execute forward pass on an EinsumGraph.
    fn forward(&mut self, graph: &EinsumGraph) -> Result<Self::Tensor, Self::Error>;

    /// Execute backward pass to compute gradients.
    fn backward(
        &mut self,
        graph: &EinsumGraph,
        loss: &Self::Tensor,
    ) -> Result<Self::Tape, Self::Error>;
}
