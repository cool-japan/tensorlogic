//! Error types for PGM operations.

use std::fmt;

/// Errors that can occur in PGM operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PgmError {
    /// Variable not found in factor graph
    VariableNotFound(String),
    /// Factor not found
    FactorNotFound(String),
    /// Dimension mismatch in tensor operations
    DimensionMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Invalid probability distribution
    InvalidDistribution(String),
    /// Message passing convergence failure
    ConvergenceFailure(String),
    /// Invalid factor graph structure
    InvalidGraph(String),
}

impl fmt::Display for PgmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VariableNotFound(name) => write!(f, "Variable not found: {}", name),
            Self::FactorNotFound(name) => write!(f, "Factor not found: {}", name),
            Self::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, got {:?}",
                    expected, got
                )
            }
            Self::InvalidDistribution(msg) => write!(f, "Invalid distribution: {}", msg),
            Self::ConvergenceFailure(msg) => write!(f, "Convergence failure: {}", msg),
            Self::InvalidGraph(msg) => write!(f, "Invalid graph: {}", msg),
        }
    }
}

impl std::error::Error for PgmError {}

/// Result type for PGM operations.
pub type Result<T> = std::result::Result<T, PgmError>;
