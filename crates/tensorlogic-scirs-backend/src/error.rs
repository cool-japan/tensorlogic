//! Comprehensive error types for tensorlogic-scirs-backend.
//!
//! This module provides detailed error types for all failure modes in the SciRS2 backend,
//! including shape mismatches, invalid operations, device errors, and numerical issues.

use std::fmt;
use thiserror::Error;

/// Main error type for SciRS2 backend operations
#[derive(Error, Debug)]
pub enum TlBackendError {
    /// Shape mismatch between tensors or operations
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(ShapeMismatchError),

    /// Invalid einsum specification
    #[error("Invalid einsum spec: {0}")]
    InvalidEinsumSpec(String),

    /// Tensor not found in storage
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    /// Invalid operation or operation parameters
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Device-related errors (GPU unavailable, memory, etc.)
    #[error("Device error: {0}")]
    DeviceError(DeviceError),

    /// Out of memory errors
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Numerical stability issues (NaN, Inf, overflow)
    #[error("Numerical error: {0}")]
    NumericalError(NumericalError),

    /// Gradient computation errors
    #[error("Gradient error: {0}")]
    GradientError(String),

    /// Graph structure errors (cycles, missing nodes, etc.)
    #[error("Graph error: {0}")]
    GraphError(String),

    /// Execution errors (runtime failures)
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Unsupported feature or operation
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Internal errors (should not happen)
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Detailed shape mismatch error with context
#[derive(Debug, Clone)]
pub struct ShapeMismatchError {
    /// Description of the operation that failed
    pub operation: String,
    /// Expected shape(s)
    pub expected: Vec<Vec<usize>>,
    /// Actual shape(s) that were provided
    pub actual: Vec<Vec<usize>>,
    /// Additional context
    pub context: Option<String>,
}

impl fmt::Display for ShapeMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Shape mismatch in {}: expected {:?}, got {:?}",
            self.operation, self.expected, self.actual
        )?;
        if let Some(ctx) = &self.context {
            write!(f, " ({})", ctx)?;
        }
        Ok(())
    }
}

impl ShapeMismatchError {
    /// Create a new shape mismatch error
    pub fn new(
        operation: impl Into<String>,
        expected: Vec<Vec<usize>>,
        actual: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            operation: operation.into(),
            expected,
            actual,
            context: None,
        }
    }

    /// Add context to the error
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Device-related errors
#[derive(Error, Debug, Clone)]
pub enum DeviceError {
    /// GPU is not available
    #[error("GPU not available: {0}")]
    GpuUnavailable(String),

    /// Device memory allocation failed
    #[error("Device memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Device synchronization failed
    #[error("Device synchronization failed: {0}")]
    SyncFailed(String),

    /// Unsupported device type
    #[error("Unsupported device: {0}")]
    UnsupportedDevice(String),
}

/// Numerical stability and correctness errors
#[derive(Debug, Clone)]
pub struct NumericalError {
    /// Type of numerical issue
    pub kind: NumericalErrorKind,
    /// Location where the error occurred
    pub location: String,
    /// Values that caused the error (if available)
    pub values: Option<Vec<f64>>,
}

/// Types of numerical errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalErrorKind {
    /// Not-a-Number detected
    NaN,
    /// Infinity detected
    Infinity,
    /// Overflow in computation
    Overflow,
    /// Underflow in computation
    Underflow,
    /// Division by zero
    DivisionByZero,
    /// Loss of precision
    PrecisionLoss,
}

impl fmt::Display for NumericalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} detected in {}", self.kind, self.location)?;
        if let Some(vals) = &self.values {
            write!(f, " (values: {:?})", vals)?;
        }
        Ok(())
    }
}

impl NumericalError {
    /// Create a new numerical error
    pub fn new(kind: NumericalErrorKind, location: impl Into<String>) -> Self {
        Self {
            kind,
            location: location.into(),
            values: None,
        }
    }

    /// Add values that caused the error
    pub fn with_values(mut self, values: Vec<f64>) -> Self {
        self.values = Some(values);
        self
    }
}

/// Result type using TlBackendError
pub type TlBackendResult<T> = Result<T, TlBackendError>;

/// Helper functions for creating common errors
impl TlBackendError {
    /// Create a shape mismatch error
    pub fn shape_mismatch(
        operation: impl Into<String>,
        expected: Vec<Vec<usize>>,
        actual: Vec<Vec<usize>>,
    ) -> Self {
        TlBackendError::ShapeMismatch(ShapeMismatchError::new(operation, expected, actual))
    }

    /// Create an invalid einsum spec error
    pub fn invalid_einsum(spec: impl Into<String>) -> Self {
        TlBackendError::InvalidEinsumSpec(spec.into())
    }

    /// Create a tensor not found error
    pub fn tensor_not_found(name: impl Into<String>) -> Self {
        TlBackendError::TensorNotFound(name.into())
    }

    /// Create an invalid operation error
    pub fn invalid_operation(msg: impl Into<String>) -> Self {
        TlBackendError::InvalidOperation(msg.into())
    }

    /// Create a numerical error
    pub fn numerical(kind: NumericalErrorKind, location: impl Into<String>) -> Self {
        TlBackendError::NumericalError(NumericalError::new(kind, location))
    }

    /// Create a GPU unavailable error
    pub fn gpu_unavailable(msg: impl Into<String>) -> Self {
        TlBackendError::DeviceError(DeviceError::GpuUnavailable(msg.into()))
    }

    /// Create an unsupported feature error
    pub fn unsupported(msg: impl Into<String>) -> Self {
        TlBackendError::Unsupported(msg.into())
    }

    /// Create an execution error
    pub fn execution(msg: impl Into<String>) -> Self {
        TlBackendError::ExecutionError(msg.into())
    }

    /// Create a gradient error
    pub fn gradient(msg: impl Into<String>) -> Self {
        TlBackendError::GradientError(msg.into())
    }
}

/// Check if a value is numerically valid (not NaN or Inf)
pub fn validate_numeric_value(value: f64, location: &str) -> TlBackendResult<()> {
    if value.is_nan() {
        Err(TlBackendError::numerical(NumericalErrorKind::NaN, location))
    } else if value.is_infinite() {
        Err(TlBackendError::numerical(
            NumericalErrorKind::Infinity,
            location,
        ))
    } else {
        Ok(())
    }
}

/// Check if all values in a slice are numerically valid
pub fn validate_numeric_values(values: &[f64], location: &str) -> TlBackendResult<()> {
    for &value in values.iter() {
        if value.is_nan() {
            return Err(TlBackendError::NumericalError(
                NumericalError::new(NumericalErrorKind::NaN, location).with_values(vec![value]),
            ));
        }
        if value.is_infinite() {
            return Err(TlBackendError::NumericalError(
                NumericalError::new(NumericalErrorKind::Infinity, location)
                    .with_values(vec![value]),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_mismatch_error() {
        let err = TlBackendError::shape_mismatch(
            "matmul",
            vec![vec![2, 3], vec![3, 4]],
            vec![vec![2, 3], vec![2, 4]],
        );
        assert!(matches!(err, TlBackendError::ShapeMismatch(_)));
        assert!(err.to_string().contains("matmul"));
    }

    #[test]
    fn test_numerical_error() {
        let err = TlBackendError::numerical(NumericalErrorKind::NaN, "relu operation");
        assert!(matches!(err, TlBackendError::NumericalError(_)));
        assert!(err.to_string().contains("NaN"));
    }

    #[test]
    fn test_validate_numeric_value() {
        // Valid values
        assert!(validate_numeric_value(0.0, "test").is_ok());
        assert!(validate_numeric_value(1.5, "test").is_ok());
        assert!(validate_numeric_value(-10.0, "test").is_ok());

        // Invalid values
        assert!(validate_numeric_value(f64::NAN, "test").is_err());
        assert!(validate_numeric_value(f64::INFINITY, "test").is_err());
        assert!(validate_numeric_value(f64::NEG_INFINITY, "test").is_err());
    }

    #[test]
    fn test_validate_numeric_values() {
        // Valid values
        let valid = vec![0.0, 1.0, -1.0, 100.0];
        assert!(validate_numeric_values(&valid, "test").is_ok());

        // Invalid values
        let invalid_nan = vec![0.0, f64::NAN, 1.0];
        assert!(validate_numeric_values(&invalid_nan, "test").is_err());

        let invalid_inf = vec![0.0, 1.0, f64::INFINITY];
        assert!(validate_numeric_values(&invalid_inf, "test").is_err());
    }

    #[test]
    fn test_error_display() {
        let err = TlBackendError::invalid_einsum("abc,def->xyz");
        assert_eq!(err.to_string(), "Invalid einsum spec: abc,def->xyz");

        let err = TlBackendError::tensor_not_found("tensor_x");
        assert_eq!(err.to_string(), "Tensor not found: tensor_x");
    }

    #[test]
    fn test_device_error() {
        let err = TlBackendError::gpu_unavailable("CUDA not installed");
        assert!(matches!(err, TlBackendError::DeviceError(_)));
        assert!(err.to_string().contains("GPU not available"));
    }

    #[test]
    fn test_shape_mismatch_with_context() {
        let mut err = ShapeMismatchError::new("einsum", vec![vec![2, 3]], vec![vec![3, 4]]);
        err = err.with_context("input tensor 'x'");
        let err_str = err.to_string();
        assert!(err_str.contains("einsum"));
        assert!(err_str.contains("input tensor 'x'"));
    }
}
