//! Training callbacks for monitoring and controlling training.

pub mod advanced;
pub mod checkpoint;
pub mod core;
pub mod early_stopping;
pub mod gradient;
pub mod histogram;
pub mod lr_finder;
pub mod profiling;

// Re-export core types
pub use core::{BatchCallback, Callback, CallbackList, EpochCallback, ValidationCallback};

// Re-export checkpoint types
pub use checkpoint::{CheckpointCallback, CheckpointCompression, TrainingCheckpoint};

// Re-export early stopping types
pub use early_stopping::{EarlyStoppingCallback, ReduceLrOnPlateauCallback};

// Re-export lr_finder types
pub use lr_finder::LearningRateFinder;

// Re-export gradient types
pub use gradient::{
    GradientAccumulationCallback, GradientAccumulationStats, GradientMonitor,
    GradientScalingStrategy, GradientSummary,
};

// Re-export histogram types
pub use histogram::{HistogramCallback, HistogramStats};

// Re-export profiling types
pub use profiling::{ProfilingCallback, ProfilingStats};

// Re-export advanced types
pub use advanced::{ModelEMACallback, SWACallback};
