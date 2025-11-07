//! Error types for the IR.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum IrError {
    #[error("Einsum spec cannot be empty")]
    EmptyEinsumSpec,
    #[error("Invalid einsum spec '{spec}': {reason}")]
    InvalidEinsumSpec { spec: String, reason: String },
    #[error("Input tensor index {index} out of bounds (max: {max})")]
    TensorIndexOutOfBounds { index: usize, max: usize },
    #[error("Output index {index} out of bounds (max: {max})")]
    OutputIndexOutOfBounds { index: usize, max: usize },
    #[error("Node {node}: {message}")]
    NodeValidation { node: usize, message: String },
    #[error("Predicate {name} not found in signature registry")]
    PredicateNotFound { name: String },
    #[error("Predicate {name} arity mismatch: expected {expected}, got {actual}")]
    ArityMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error(
        "Predicate {name} type mismatch at argument {arg_index}: expected {expected}, got {actual}"
    )]
    TypeMismatch {
        name: String,
        arg_index: usize,
        expected: String,
        actual: String,
    },
    #[error("Unbound variable {var} in expression")]
    UnboundVariable { var: String },
    #[error("Variable {var} used with inconsistent types: {type1} and {type2}")]
    InconsistentTypes {
        var: String,
        type1: String,
        type2: String,
    },
    #[error("Domain {name} not found in registry")]
    DomainNotFound { name: String },
    #[error("Domain {name} already exists in registry")]
    DomainAlreadyExists { name: String },
    #[error("Domain incompatibility: {domain1} and {domain2} are not compatible")]
    DomainIncompatible { domain1: String, domain2: String },
    #[error("Variable {var} domain mismatch: expected {expected}, got {actual}")]
    VariableDomainMismatch {
        var: String,
        expected: String,
        actual: String,
    },
    #[error("Aggregation operation {op} not supported")]
    UnsupportedAggregation { op: String },
    #[error("Graph contains a cycle and cannot be topologically sorted")]
    CyclicGraph,
}
