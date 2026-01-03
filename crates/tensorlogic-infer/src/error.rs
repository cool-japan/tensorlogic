//! Error types for execution engines.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecutorError {
    #[error("Tensor '{0}' not found")]
    TensorNotFound(String),

    #[error("Invalid einsum spec: {0}")]
    InvalidEinsumSpec(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Invalid axis: {axis} for tensor with rank {rank}")]
    InvalidAxis { axis: usize, rank: usize },

    #[error("Graph validation failed: {0}")]
    GraphValidationError(String),

    #[error("Node {0} has invalid dependencies")]
    InvalidDependencies(usize),

    #[error("Profiling error: {0}")]
    ProfilingError(String),

    #[error("Batch execution error: {0}")]
    BatchExecutionError(String),

    #[error("Memory allocation failed: {0}")]
    MemoryError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Type error: expected {expected}, got {actual}")]
    TypeError { expected: String, actual: String },

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Empty input: {0}")]
    EmptyInput(String),

    #[error("Backend capability not supported: {0}")]
    CapabilityNotSupported(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}
