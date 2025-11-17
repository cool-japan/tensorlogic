//! Performance profiling callbacks for training.

use crate::callbacks::core::Callback;
use crate::{TrainResult, TrainingState};

/// Performance profiling statistics.
#[derive(Debug, Clone, Default)]
pub struct ProfilingStats {
    /// Total training time (seconds).
    pub total_time: f64,
    /// Time per epoch (seconds).
    pub epoch_times: Vec<f64>,
    /// Samples per second.
    pub samples_per_sec: f64,
    /// Batches per second.
    pub batches_per_sec: f64,
    /// Average batch time (seconds).
    pub avg_batch_time: f64,
    /// Peak memory usage (MB) - placeholder.
    pub peak_memory_mb: f64,
}

impl ProfilingStats {
    /// Pretty print profiling statistics.
    pub fn display(&self) {
        println!("\n=== Profiling Statistics ===");
        println!("Total time: {:.2}s", self.total_time);
        println!("Samples/sec: {:.2}", self.samples_per_sec);
        println!("Batches/sec: {:.2}", self.batches_per_sec);
        println!("Avg batch time: {:.4}s", self.avg_batch_time);

        if !self.epoch_times.is_empty() {
            let avg_epoch = self.epoch_times.iter().sum::<f64>() / self.epoch_times.len() as f64;
            let min_epoch = self
                .epoch_times
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let max_epoch = self
                .epoch_times
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            println!("\nEpoch times:");
            println!("  Average: {:.2}s", avg_epoch);
            println!("  Min: {:.2}s", min_epoch);
            println!("  Max: {:.2}s", max_epoch);
        }
    }
}

/// Callback for profiling training performance.
///
/// Tracks timing information and throughput metrics during training.
/// Useful for:
/// - Identifying performance bottlenecks
/// - Comparing different configurations
/// - Monitoring training speed
/// - Resource utilization tracking
///
/// # Example
///
/// ```no_run
/// use tensorlogic_train::{CallbackList, ProfilingCallback};
///
/// let mut callbacks = CallbackList::new();
/// callbacks.add(Box::new(ProfilingCallback::new(
///     true,  // verbose: Print detailed stats
///     5,     // log_frequency: Every 5 epochs
/// )));
/// ```
pub struct ProfilingCallback {
    /// Whether to print detailed profiling info.
    verbose: bool,
    /// Frequency of logging (every N epochs).
    log_frequency: usize,
    /// Training start time.
    start_time: Option<std::time::Instant>,
    /// Last epoch start time.
    epoch_start_time: Option<std::time::Instant>,
    /// Batch start time.
    batch_start_time: Option<std::time::Instant>,
    /// Accumulated statistics.
    pub stats: ProfilingStats,
    /// Batch times for current epoch.
    current_epoch_batch_times: Vec<f64>,
    /// Total batches processed.
    total_batches: usize,
}

impl ProfilingCallback {
    /// Create a new profiling callback.
    ///
    /// # Arguments
    /// * `verbose` - Print detailed profiling information
    /// * `log_frequency` - Log stats every N epochs
    pub fn new(verbose: bool, log_frequency: usize) -> Self {
        Self {
            verbose,
            log_frequency,
            start_time: None,
            epoch_start_time: None,
            batch_start_time: None,
            stats: ProfilingStats::default(),
            current_epoch_batch_times: Vec::new(),
            total_batches: 0,
        }
    }

    /// Get profiling statistics.
    pub fn get_stats(&self) -> &ProfilingStats {
        &self.stats
    }
}

impl Callback for ProfilingCallback {
    fn on_train_begin(&mut self, _state: &TrainingState) -> TrainResult<()> {
        self.start_time = Some(std::time::Instant::now());
        if self.verbose {
            println!("Profiling started");
        }
        Ok(())
    }

    fn on_train_end(&mut self, _state: &TrainingState) -> TrainResult<()> {
        if let Some(start) = self.start_time {
            self.stats.total_time = start.elapsed().as_secs_f64();

            // Compute aggregate statistics
            if self.total_batches > 0 {
                self.stats.avg_batch_time = self.stats.total_time / self.total_batches as f64;
                self.stats.batches_per_sec = self.total_batches as f64 / self.stats.total_time;
            }

            if self.verbose {
                println!("\nProfiling completed");
                self.stats.display();
            }
        }
        Ok(())
    }

    fn on_epoch_begin(&mut self, epoch: usize, _state: &TrainingState) -> TrainResult<()> {
        self.epoch_start_time = Some(std::time::Instant::now());
        self.current_epoch_batch_times.clear();

        if self.verbose && (epoch + 1).is_multiple_of(self.log_frequency) {
            println!("\nEpoch {} profiling started", epoch + 1);
        }
        Ok(())
    }

    fn on_epoch_end(&mut self, epoch: usize, _state: &TrainingState) -> TrainResult<()> {
        if let Some(epoch_start) = self.epoch_start_time {
            let epoch_time = epoch_start.elapsed().as_secs_f64();
            self.stats.epoch_times.push(epoch_time);

            if self.verbose && (epoch + 1).is_multiple_of(self.log_frequency) {
                let avg_batch = if !self.current_epoch_batch_times.is_empty() {
                    self.current_epoch_batch_times.iter().sum::<f64>()
                        / self.current_epoch_batch_times.len() as f64
                } else {
                    0.0
                };

                println!("Epoch {} completed:", epoch + 1);
                println!("    Time: {:.2}s", epoch_time);
                println!(
                    "    Batches: {} ({:.4}s avg)",
                    self.current_epoch_batch_times.len(),
                    avg_batch
                );
            }
        }
        Ok(())
    }

    fn on_batch_begin(&mut self, _batch: usize, _state: &TrainingState) -> TrainResult<()> {
        self.batch_start_time = Some(std::time::Instant::now());
        Ok(())
    }

    fn on_batch_end(&mut self, _batch: usize, _state: &TrainingState) -> TrainResult<()> {
        if let Some(batch_start) = self.batch_start_time {
            let batch_time = batch_start.elapsed().as_secs_f64();
            self.current_epoch_batch_times.push(batch_time);
            self.total_batches += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_profiling_callback() {
        let mut callback = ProfilingCallback::new(false, 1);
        let state = TrainingState {
            epoch: 0,
            batch: 0,
            train_loss: 0.5,
            batch_loss: 0.5,
            val_loss: Some(0.6),
            learning_rate: 0.01,
            metrics: HashMap::new(),
        };

        callback.on_train_begin(&state).unwrap();
        assert!(callback.start_time.is_some());

        callback.on_epoch_begin(0, &state).unwrap();
        assert!(callback.epoch_start_time.is_some());

        callback.on_batch_begin(0, &state).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        callback.on_batch_end(0, &state).unwrap();

        assert_eq!(callback.total_batches, 1);
        assert_eq!(callback.current_epoch_batch_times.len(), 1);

        callback.on_epoch_end(0, &state).unwrap();
        assert_eq!(callback.stats.epoch_times.len(), 1);

        callback.on_train_end(&state).unwrap();
        assert!(callback.stats.total_time > 0.0);
    }
}
