//! Error types for training operations.

use thiserror::Error;

/// Errors that can occur during training.
#[derive(Error, Debug)]
pub enum TrainError {
    /// Error in loss computation.
    #[error("Loss computation error: {0}")]
    LossError(String),

    /// Error in optimizer operation.
    #[error("Optimizer error: {0}")]
    OptimizerError(String),

    /// Error in batch processing.
    #[error("Batch processing error: {0}")]
    BatchError(String),

    /// Error in callback execution.
    #[error("Callback error: {0}")]
    CallbackError(String),

    /// Error in metrics computation.
    #[error("Metrics error: {0}")]
    MetricsError(String),

    /// Error in checkpoint save/load.
    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    /// Error with invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Error in model operations.
    #[error("Model error: {0}")]
    ModelError(String),

    /// Error in configuration.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Error from tensorlogic-infer.
    #[error("Executor error: {0}")]
    ExecutorError(#[from] tensorlogic_infer::ExecutorError),

    /// Generic error.
    #[error("{0}")]
    Other(String),
}

/// Result type for training operations.
pub type TrainResult<T> = Result<T, TrainError>;
