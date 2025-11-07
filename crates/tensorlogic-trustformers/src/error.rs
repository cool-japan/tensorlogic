//! Error types for tensorlogic-trustformers.

use std::fmt;

/// Errors that can occur in transformer operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TrustformerError {
    /// Invalid dimension configuration
    InvalidDimension {
        expected: usize,
        got: usize,
        context: String,
    },
    /// Head count doesn't divide model dimension evenly
    InvalidHeadCount { d_model: usize, n_heads: usize },
    /// Invalid attention mask shape
    InvalidMaskShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Missing required parameter
    MissingParameter(String),
    /// Compilation error when building einsum graph
    CompilationError(String),
}

impl fmt::Display for TrustformerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension {
                expected,
                got,
                context,
            } => write!(
                f,
                "Invalid dimension in {}: expected {}, got {}",
                context, expected, got
            ),
            Self::InvalidHeadCount { d_model, n_heads } => write!(
                f,
                "d_model ({}) must be divisible by n_heads ({})",
                d_model, n_heads
            ),
            Self::InvalidMaskShape { expected, got } => write!(
                f,
                "Invalid mask shape: expected {:?}, got {:?}",
                expected, got
            ),
            Self::MissingParameter(param) => write!(f, "Missing required parameter: {}", param),
            Self::CompilationError(msg) => write!(f, "Compilation error: {}", msg),
        }
    }
}

impl std::error::Error for TrustformerError {}

/// Convert IrError to TrustformerError (for ? operator)
impl From<tensorlogic_ir::IrError> for TrustformerError {
    fn from(err: tensorlogic_ir::IrError) -> Self {
        TrustformerError::CompilationError(err.to_string())
    }
}

/// Result type for transformer operations
pub type Result<T> = std::result::Result<T, TrustformerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TrustformerError::InvalidDimension {
            expected: 512,
            got: 256,
            context: "attention".to_string(),
        };
        assert!(err.to_string().contains("512"));
        assert!(err.to_string().contains("256"));
    }

    #[test]
    fn test_invalid_head_count() {
        let err = TrustformerError::InvalidHeadCount {
            d_model: 512,
            n_heads: 7,
        };
        assert!(err.to_string().contains("512"));
        assert!(err.to_string().contains("7"));
    }
}
