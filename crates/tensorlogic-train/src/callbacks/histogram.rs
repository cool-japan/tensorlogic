//! Histogram monitoring callbacks for weight distributions.

use crate::callbacks::core::Callback;
use crate::{TrainResult, TrainingState};
use std::collections::HashMap;

/// Weight histogram statistics for debugging and monitoring.
#[derive(Debug, Clone)]
pub struct HistogramStats {
    /// Parameter name.
    pub name: String,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Histogram bins (boundaries).
    pub bins: Vec<f64>,
    /// Histogram counts per bin.
    pub counts: Vec<usize>,
}

impl HistogramStats {
    /// Compute histogram statistics from parameter values.
    pub fn compute(name: &str, values: &[f64], num_bins: usize) -> Self {
        if values.is_empty() {
            return Self {
                name: name.to_string(),
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std: 0.0,
                bins: vec![],
                counts: vec![],
            };
        }

        // Basic statistics
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        // Create histogram bins
        let mut bins = Vec::with_capacity(num_bins + 1);
        let mut counts = vec![0; num_bins];

        let range = max - min;
        let bin_width = if range > 0.0 {
            range / num_bins as f64
        } else {
            1.0
        };

        for i in 0..=num_bins {
            bins.push(min + i as f64 * bin_width);
        }

        // Count values in each bin
        for &value in values {
            let bin_idx = if range > 0.0 {
                ((value - min) / bin_width).floor() as usize
            } else {
                0
            };
            let bin_idx = bin_idx.min(num_bins - 1);
            counts[bin_idx] += 1;
        }

        Self {
            name: name.to_string(),
            min,
            max,
            mean,
            std,
            bins,
            counts,
        }
    }

    /// Pretty print histogram as ASCII art.
    pub fn display(&self, width: usize) {
        println!("\n=== Histogram: {} ===", self.name);
        println!("  Min: {:.6}, Max: {:.6}", self.min, self.max);
        println!("  Mean: {:.6}, Std: {:.6}", self.mean, self.std);
        println!("\n  Distribution:");

        if self.counts.is_empty() {
            println!("    (empty)");
            return;
        }

        let max_count = *self.counts.iter().max().unwrap_or(&1);

        for (i, &count) in self.counts.iter().enumerate() {
            let bar_len = if max_count > 0 {
                (count as f64 / max_count as f64 * width as f64) as usize
            } else {
                0
            };

            let bar = "█".repeat(bar_len);
            let left = if i < self.bins.len() - 1 {
                self.bins[i]
            } else {
                self.bins[i - 1]
            };
            let right = if i < self.bins.len() - 1 {
                self.bins[i + 1]
            } else {
                self.bins[i]
            };

            println!("  [{:>8.3}, {:>8.3}): {:>6} {}", left, right, count, bar);
        }
    }
}

/// Callback for tracking weight histograms during training.
///
/// This callback computes and logs histogram statistics of model parameters
/// at regular intervals. Useful for:
/// - Detecting vanishing/exploding weights
/// - Monitoring weight distribution changes
/// - Debugging initialization issues
/// - Understanding parameter evolution
///
/// # Example
///
/// ```no_run
/// use tensorlogic_train::{CallbackList, HistogramCallback};
///
/// let mut callbacks = CallbackList::new();
/// callbacks.add(Box::new(HistogramCallback::new(
///     5,   // log_frequency: Every 5 epochs
///     10,  // num_bins: 10 histogram bins
///     true, // verbose: Print detailed histograms
/// )));
/// ```
pub struct HistogramCallback {
    /// Frequency of logging (every N epochs).
    log_frequency: usize,
    /// Number of histogram bins.
    #[allow(dead_code)]
    // Used in compute_histograms - will be active when parameters are accessible
    num_bins: usize,
    /// Whether to print detailed histograms.
    verbose: bool,
    /// History of histogram statistics.
    pub history: Vec<HashMap<String, HistogramStats>>,
}

impl HistogramCallback {
    /// Create a new histogram callback.
    ///
    /// # Arguments
    /// * `log_frequency` - Log histograms every N epochs
    /// * `num_bins` - Number of bins in each histogram
    /// * `verbose` - Print detailed ASCII histograms
    pub fn new(log_frequency: usize, num_bins: usize, verbose: bool) -> Self {
        Self {
            log_frequency,
            num_bins,
            verbose,
            history: Vec::new(),
        }
    }

    /// Compute histograms for all parameters in state.
    #[allow(dead_code)] // Placeholder - will be used when TrainingState includes parameters
    fn compute_histograms(&self, _state: &TrainingState) -> HashMap<String, HistogramStats> {
        // In a real implementation, we would access parameters from state
        // For now, this is a placeholder that would be populated when
        // TrainingState includes parameter access

        // Example of what this would look like with actual parameters:
        // let mut histograms = HashMap::new();
        // for (name, param) in state.parameters.iter() {
        //     let values: Vec<f64> = param.iter().copied().collect();
        //     let stats = HistogramStats::compute(name, &values, self.num_bins);
        //     histograms.insert(name.clone(), stats);
        // }
        // histograms

        HashMap::new()
    }
}

impl Callback for HistogramCallback {
    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        if (epoch + 1).is_multiple_of(self.log_frequency) {
            let histograms = self.compute_histograms(state);

            if self.verbose {
                println!("\n--- Weight Histograms (Epoch {}) ---", epoch + 1);
                for (_name, stats) in histograms.iter() {
                    stats.display(40); // 40 character width for ASCII bars
                }
            } else {
                println!(
                    "Epoch {}: Computed histograms for {} parameters",
                    epoch + 1,
                    histograms.len()
                );
            }

            self.history.push(histograms);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = HistogramStats::compute("test", &values, 5);

        assert_eq!(stats.name, "test");
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert!((stats.mean - 5.5).abs() < 1e-6);
        assert_eq!(stats.bins.len(), 6);
        assert_eq!(stats.counts.len(), 5);
        assert_eq!(stats.counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_histogram_callback() {
        let mut callback = HistogramCallback::new(2, 10, false);
        let state = TrainingState {
            epoch: 0,
            batch: 0,
            train_loss: 0.5,
            batch_loss: 0.5,
            val_loss: Some(0.6),
            learning_rate: 0.01,
            metrics: HashMap::new(),
        };

        // Should not log on epoch 0
        callback.on_epoch_end(0, &state).unwrap();
        assert_eq!(callback.history.len(), 0);

        // Should log on epoch 1 (frequency=2, so every 2 epochs)
        callback.on_epoch_end(1, &state).unwrap();
        assert_eq!(callback.history.len(), 1);
    }
}
