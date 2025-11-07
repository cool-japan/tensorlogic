//! Error types for tensorlogic-sklears-kernels.

use std::fmt;

/// Errors that can occur in kernel operations.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelError {
    /// Mismatched dimensions between inputs
    DimensionMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
        context: String,
    },
    /// Invalid kernel parameter
    InvalidParameter {
        parameter: String,
        value: String,
        reason: String,
    },
    /// Kernel computation failed
    ComputationError(String),
    /// Invalid TLExpr for kernel construction
    InvalidExpression(String),
    /// Incompatible kernel types for composition
    IncompatibleKernels {
        kernel_a: String,
        kernel_b: String,
        reason: String,
    },
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                expected,
                got,
                context,
            } => write!(
                f,
                "Dimension mismatch in {}: expected {:?}, got {:?}",
                context, expected, got
            ),
            Self::InvalidParameter {
                parameter,
                value,
                reason,
            } => write!(
                f,
                "Invalid parameter '{}' = '{}': {}",
                parameter, value, reason
            ),
            Self::ComputationError(msg) => write!(f, "Kernel computation error: {}", msg),
            Self::InvalidExpression(msg) => write!(f, "Invalid expression for kernel: {}", msg),
            Self::IncompatibleKernels {
                kernel_a,
                kernel_b,
                reason,
            } => write!(
                f,
                "Incompatible kernels '{}' and '{}': {}",
                kernel_a, kernel_b, reason
            ),
        }
    }
}

impl std::error::Error for KernelError {}

/// Convert IrError to KernelError
impl From<tensorlogic_ir::IrError> for KernelError {
    fn from(err: tensorlogic_ir::IrError) -> Self {
        KernelError::InvalidExpression(err.to_string())
    }
}

/// Result type for kernel operations
pub type Result<T> = std::result::Result<T, KernelError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_display() {
        let err = KernelError::DimensionMismatch {
            expected: vec![10, 20],
            got: vec![10, 30],
            context: "kernel matrix".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("10, 20"));
        assert!(msg.contains("10, 30"));
    }

    #[test]
    fn test_invalid_parameter_display() {
        let err = KernelError::InvalidParameter {
            parameter: "gamma".to_string(),
            value: "-1.0".to_string(),
            reason: "must be positive".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("gamma"));
        assert!(msg.contains("-1.0"));
        assert!(msg.contains("positive"));
    }
}
