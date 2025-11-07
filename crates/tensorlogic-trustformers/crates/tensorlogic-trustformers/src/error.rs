//! Error types for the TensorLogic TrustformeRS crate.

use thiserror::Error;

/// Result type for TrustformeRS operations.
pub type TrustformersResult<T> = Result<T, TrustformersError>;

/// Error types for transformer component operations.
#[derive(Error, Debug)]
pub enum TrustformersError {
    /// Invalid configuration parameter
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Dimension mismatch in tensor operations
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Invalid attention pattern specification
    #[error("Invalid attention pattern: {0}")]
    InvalidAttentionPattern(String),

    /// Invalid position encoding specification
    #[error("Invalid position encoding: {0}")]
    InvalidPositionEncoding(String),

    /// Number of heads does not divide hidden size
    #[error("Hidden size {hidden_size} must be divisible by number of heads {num_heads}")]
    InvalidHeadDimension { hidden_size: usize, num_heads: usize },

    /// Sequence length exceeds maximum
    #[error("Sequence length {actual} exceeds maximum {max}")]
    SequenceTooLong { actual: usize, max: usize },

    /// IR compilation error
    #[error("IR compilation error: {0}")]
    IrError(#[from] tensorlogic_ir::error::IrError),

    /// Inference error
    #[error("Inference error: {0}")]
    InferError(String),

    /// Generic error for other cases
    #[error("TrustformeRS error: {0}")]
    Other(String),
}

impl TrustformersError {
    /// Create an InvalidConfig error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create a DimensionMismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create an InvalidAttentionPattern error
    pub fn invalid_attention_pattern(msg: impl Into<String>) -> Self {
        Self::InvalidAttentionPattern(msg.into())
    }

    /// Create an InvalidPositionEncoding error
    pub fn invalid_position_encoding(msg: impl Into<String>) -> Self {
        Self::InvalidPositionEncoding(msg.into())
    }

    /// Create an InvalidHeadDimension error
    pub fn invalid_head_dimension(hidden_size: usize, num_heads: usize) -> Self {
        Self::InvalidHeadDimension {
            hidden_size,
            num_heads,
        }
    }

    /// Create a SequenceTooLong error
    pub fn sequence_too_long(actual: usize, max: usize) -> Self {
        Self::SequenceTooLong { actual, max }
    }

    /// Create an InferError
    pub fn infer_error(msg: impl Into<String>) -> Self {
        Self::InferError(msg.into())
    }

    /// Create a generic Other error
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = TrustformersError::invalid_config("bad value");
        assert!(matches!(err, TrustformersError::InvalidConfig(_)));

        let err = TrustformersError::dimension_mismatch(10, 5);
        assert!(matches!(err, TrustformersError::DimensionMismatch { .. }));

        let err = TrustformersError::invalid_head_dimension(768, 11);
        assert!(matches!(err, TrustformersError::InvalidHeadDimension { .. }));
    }

    #[test]
    fn test_error_display() {
        let err = TrustformersError::invalid_config("test message");
        assert_eq!(err.to_string(), "Invalid configuration: test message");

        let err = TrustformersError::dimension_mismatch(10, 5);
        assert_eq!(err.to_string(), "Dimension mismatch: expected 10, got 5");

        let err = TrustformersError::invalid_head_dimension(768, 11);
        assert_eq!(
            err.to_string(),
            "Hidden size 768 must be divisible by number of heads 11"
        );
    }
}
