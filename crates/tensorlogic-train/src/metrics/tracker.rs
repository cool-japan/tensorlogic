//! Metric tracker for managing multiple metrics.

use crate::TrainResult;
use scirs2_core::ndarray::{ArrayView, Ix2};
use std::collections::HashMap;

use super::Metric;

/// Metric tracker for managing multiple metrics.
pub struct MetricTracker {
    /// Metrics to track.
    metrics: Vec<Box<dyn Metric>>,
    /// History of metric values.
    history: HashMap<String, Vec<f64>>,
}

impl MetricTracker {
    /// Create a new metric tracker.
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            history: HashMap::new(),
        }
    }

    /// Add a metric to track.
    pub fn add(&mut self, metric: Box<dyn Metric>) {
        let name = metric.name().to_string();
        self.history.insert(name, Vec::new());
        self.metrics.push(metric);
    }

    /// Compute all metrics.
    pub fn compute_all(
        &mut self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for metric in &self.metrics {
            let value = metric.compute(predictions, targets)?;
            let name = metric.name().to_string();

            results.insert(name.clone(), value);

            if let Some(history) = self.history.get_mut(&name) {
                history.push(value);
            }
        }

        Ok(results)
    }

    /// Get history for a specific metric.
    pub fn get_history(&self, metric_name: &str) -> Option<&Vec<f64>> {
        self.history.get(metric_name)
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        for metric in &mut self.metrics {
            metric.reset();
        }
    }

    /// Clear history.
    pub fn clear_history(&mut self) {
        for history in self.history.values_mut() {
            history.clear();
        }
    }
}

impl Default for MetricTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{Accuracy, F1Score};
    use scirs2_core::ndarray::array;

    #[test]
    fn test_metric_tracker() {
        let mut tracker = MetricTracker::new();
        tracker.add(Box::new(Accuracy::default()));
        tracker.add(Box::new(F1Score::default()));

        let predictions = array![[0.9, 0.1], [0.2, 0.8]];
        let targets = array![[1.0, 0.0], [0.0, 1.0]];

        let results = tracker
            .compute_all(&predictions.view(), &targets.view())
            .unwrap();
        assert!(results.contains_key("accuracy"));
        assert!(results.contains_key("f1_score"));

        let history = tracker.get_history("accuracy").unwrap();
        assert_eq!(history.len(), 1);
    }
}
