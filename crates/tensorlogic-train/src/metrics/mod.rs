//! Metrics for evaluating model performance.

use crate::TrainResult;
use scirs2_core::ndarray::{ArrayView, Ix2};

/// Trait for metrics.
pub trait Metric {
    /// Compute metric value.
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64>;

    /// Get metric name.
    fn name(&self) -> &str;

    /// Reset metric state (for stateful metrics).
    fn reset(&mut self) {}
}

// Module declarations
mod advanced;
mod basic;
mod calibration;
mod ranking;
mod tracker;
mod vision;

// Re-exports - Basic metrics
pub use basic::{Accuracy, F1Score, Precision, Recall};

// Re-exports - Advanced metrics
pub use advanced::{
    BalancedAccuracy, CohensKappa, ConfusionMatrix, MatthewsCorrelationCoefficient,
    PerClassMetrics, RocCurve,
};

// Re-exports - Ranking metrics
pub use ranking::{NormalizedDiscountedCumulativeGain, TopKAccuracy};

// Re-exports - Vision metrics
pub use vision::{DiceCoefficient, IoU, MeanAveragePrecision, MeanIoU};

// Re-exports - Calibration metrics
pub use calibration::{ExpectedCalibrationError, MaximumCalibrationError};

// Re-exports - Tracker
pub use tracker::MetricTracker;
