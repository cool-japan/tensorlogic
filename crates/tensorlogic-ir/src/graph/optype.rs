//! Operation types for tensor computations.

use serde::{Deserialize, Serialize};

/// Operation types supported in the tensor graph
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OpType {
    /// Einstein summation (tensor contraction)
    Einsum { spec: String },
    /// Element-wise unary operation
    ElemUnary { op: String },
    /// Element-wise binary operation
    ElemBinary { op: String },
    /// Reduction operation
    Reduce { op: String, axes: Vec<usize> },
}
