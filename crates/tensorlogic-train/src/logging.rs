//! Logging infrastructure for training.
//!
//! This module provides various logging backends to track training progress:
//! - Console logging (stdout/stderr)
//! - File logging (write to file)
//! - TensorBoard logging (placeholder for future integration)
//! - Metrics logging and aggregation

use crate::{TrainError, TrainResult};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Trait for logging backends.
pub trait LoggingBackend {
    /// Log a scalar metric.
    ///
    /// # Arguments
    /// * `name` - Name of the metric
    /// * `value` - Value of the metric
    /// * `step` - Training step/epoch number
    fn log_scalar(&mut self, name: &str, value: f64, step: usize) -> TrainResult<()>;

    /// Log a text message.
    ///
    /// # Arguments
    /// * `message` - Text message to log
    fn log_text(&mut self, message: &str) -> TrainResult<()>;

    /// Flush any buffered logs.
    fn flush(&mut self) -> TrainResult<()>;
}

/// Console logger that outputs to stdout.
///
/// Simple logger for debugging and development.
#[derive(Debug, Clone, Default)]
pub struct ConsoleLogger {
    /// Whether to include timestamps.
    pub include_timestamp: bool,
}

impl ConsoleLogger {
    /// Create a new console logger.
    pub fn new() -> Self {
        Self {
            include_timestamp: true,
        }
    }

    /// Create a console logger without timestamps.
    pub fn without_timestamp() -> Self {
        Self {
            include_timestamp: false,
        }
    }

    fn format_timestamp(&self) -> String {
        if self.include_timestamp {
            let now = std::time::SystemTime::now();
            match now.duration_since(std::time::UNIX_EPOCH) {
                Ok(duration) => format!("[{:.3}] ", duration.as_secs_f64()),
                Err(_) => String::new(),
            }
        } else {
            String::new()
        }
    }
}

impl LoggingBackend for ConsoleLogger {
    fn log_scalar(&mut self, name: &str, value: f64, step: usize) -> TrainResult<()> {
        println!(
            "{}Step {}: {} = {:.6}",
            self.format_timestamp(),
            step,
            name,
            value
        );
        Ok(())
    }

    fn log_text(&mut self, message: &str) -> TrainResult<()> {
        println!("{}{}", self.format_timestamp(), message);
        Ok(())
    }

    fn flush(&mut self) -> TrainResult<()> {
        use std::io::stdout;
        stdout()
            .flush()
            .map_err(|e| TrainError::Other(format!("Failed to flush stdout: {}", e)))?;
        Ok(())
    }
}

/// File logger that writes logs to a file.
///
/// Useful for persistent logging and later analysis.
#[derive(Debug)]
pub struct FileLogger {
    file: File,
    path: PathBuf,
}

impl FileLogger {
    /// Create a new file logger.
    ///
    /// # Arguments
    /// * `path` - Path to the log file (will be created or appended)
    pub fn new<P: AsRef<Path>>(path: P) -> TrainResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| TrainError::Other(format!("Failed to open log file {:?}: {}", path, e)))?;

        Ok(Self { file, path })
    }

    /// Create a new file logger, truncating the file if it exists.
    ///
    /// # Arguments
    /// * `path` - Path to the log file
    pub fn new_truncate<P: AsRef<Path>>(path: P) -> TrainResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| TrainError::Other(format!("Failed to open log file {:?}: {}", path, e)))?;

        Ok(Self { file, path })
    }

    /// Get the path to the log file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl LoggingBackend for FileLogger {
    fn log_scalar(&mut self, name: &str, value: f64, step: usize) -> TrainResult<()> {
        writeln!(self.file, "Step {}: {} = {:.6}", step, name, value)
            .map_err(|e| TrainError::Other(format!("Failed to write to log file: {}", e)))?;
        Ok(())
    }

    fn log_text(&mut self, message: &str) -> TrainResult<()> {
        writeln!(self.file, "{}", message)
            .map_err(|e| TrainError::Other(format!("Failed to write to log file: {}", e)))?;
        Ok(())
    }

    fn flush(&mut self) -> TrainResult<()> {
        self.file
            .flush()
            .map_err(|e| TrainError::Other(format!("Failed to flush log file: {}", e)))?;
        Ok(())
    }
}

/// TensorBoard logger (placeholder for future implementation).
///
/// Will integrate with TensorBoard for visualization.
#[derive(Debug, Clone)]
pub struct TensorBoardLogger {
    log_dir: PathBuf,
}

impl TensorBoardLogger {
    /// Create a new TensorBoard logger.
    ///
    /// # Arguments
    /// * `log_dir` - Directory for TensorBoard logs
    pub fn new<P: AsRef<Path>>(log_dir: P) -> TrainResult<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&log_dir).map_err(|e| {
            TrainError::Other(format!(
                "Failed to create log directory {:?}: {}",
                log_dir, e
            ))
        })?;

        Ok(Self { log_dir })
    }

    /// Get the log directory.
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }
}

impl LoggingBackend for TensorBoardLogger {
    fn log_scalar(&mut self, name: &str, value: f64, step: usize) -> TrainResult<()> {
        // Placeholder: in the future, this would write TensorBoard event files
        log::debug!("TensorBoard: Step {}: {} = {:.6}", step, name, value);
        Ok(())
    }

    fn log_text(&mut self, message: &str) -> TrainResult<()> {
        log::debug!("TensorBoard: {}", message);
        Ok(())
    }

    fn flush(&mut self) -> TrainResult<()> {
        // Placeholder
        Ok(())
    }
}

/// Metrics logger that aggregates and logs training metrics.
///
/// Collects metrics and logs them using multiple backends.
#[derive(Debug)]
pub struct MetricsLogger {
    backends: Vec<Box<dyn LoggingBackendClone>>,
    current_step: usize,
    accumulated_metrics: HashMap<String, Vec<f64>>,
}

/// Helper trait for cloning boxed logging backends.
trait LoggingBackendClone: LoggingBackend + std::fmt::Debug {
    fn clone_box(&self) -> Box<dyn LoggingBackendClone>;
}

impl<T: LoggingBackend + Clone + std::fmt::Debug + 'static> LoggingBackendClone for T {
    fn clone_box(&self) -> Box<dyn LoggingBackendClone> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn LoggingBackendClone> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl LoggingBackend for Box<dyn LoggingBackendClone> {
    fn log_scalar(&mut self, name: &str, value: f64, step: usize) -> TrainResult<()> {
        (**self).log_scalar(name, value, step)
    }

    fn log_text(&mut self, message: &str) -> TrainResult<()> {
        (**self).log_text(message)
    }

    fn flush(&mut self) -> TrainResult<()> {
        (**self).flush()
    }
}

impl MetricsLogger {
    /// Create a new metrics logger.
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            current_step: 0,
            accumulated_metrics: HashMap::new(),
        }
    }

    /// Add a logging backend.
    ///
    /// # Arguments
    /// * `backend` - Backend to add
    pub fn add_backend<B: LoggingBackend + Clone + std::fmt::Debug + 'static>(
        &mut self,
        backend: B,
    ) {
        self.backends.push(Box::new(backend));
    }

    /// Log a scalar metric to all backends.
    ///
    /// # Arguments
    /// * `name` - Metric name
    /// * `value` - Metric value
    pub fn log_metric(&mut self, name: &str, value: f64) -> TrainResult<()> {
        for backend in &mut self.backends {
            backend.log_scalar(name, value, self.current_step)?;
        }
        Ok(())
    }

    /// Accumulate a metric value (for averaging over batch).
    ///
    /// # Arguments
    /// * `name` - Metric name
    /// * `value` - Metric value
    pub fn accumulate_metric(&mut self, name: &str, value: f64) {
        self.accumulated_metrics
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    /// Log accumulated metrics (average) and clear accumulation.
    pub fn log_accumulated_metrics(&mut self) -> TrainResult<()> {
        // Collect metrics to log before clearing
        let metrics_to_log: Vec<(String, f64)> = self
            .accumulated_metrics
            .iter()
            .filter(|(_, values)| !values.is_empty())
            .map(|(name, values)| {
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                (name.clone(), avg)
            })
            .collect();

        // Log all metrics
        for (name, avg) in metrics_to_log {
            self.log_metric(&name, avg)?;
        }

        // Clear accumulation
        self.accumulated_metrics.clear();
        Ok(())
    }

    /// Log a text message to all backends.
    ///
    /// # Arguments
    /// * `message` - Text message
    pub fn log_message(&mut self, message: &str) -> TrainResult<()> {
        for backend in &mut self.backends {
            backend.log_text(message)?;
        }
        Ok(())
    }

    /// Increment the step counter.
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Set the current step.
    ///
    /// # Arguments
    /// * `step` - Step number
    pub fn set_step(&mut self, step: usize) {
        self.current_step = step;
    }

    /// Get the current step.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Flush all backends.
    pub fn flush(&mut self) -> TrainResult<()> {
        for backend in &mut self.backends {
            backend.flush()?;
        }
        Ok(())
    }

    /// Get the number of backends.
    pub fn num_backends(&self) -> usize {
        self.backends.len()
    }
}

impl Default for MetricsLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn test_console_logger() {
        let mut logger = ConsoleLogger::new();

        // These should not fail
        logger.log_scalar("loss", 0.5, 1).unwrap();
        logger.log_text("Test message").unwrap();
        logger.flush().unwrap();
    }

    #[test]
    fn test_console_logger_without_timestamp() {
        let mut logger = ConsoleLogger::without_timestamp();

        logger.log_scalar("accuracy", 0.95, 10).unwrap();
        logger.log_text("Another test").unwrap();
    }

    #[test]
    fn test_file_logger() {
        let temp_dir = env::temp_dir();
        let log_path = temp_dir.join("test_training.log");

        // Clean up if file exists
        let _ = fs::remove_file(&log_path);

        let mut logger = FileLogger::new(&log_path).unwrap();

        logger.log_scalar("loss", 0.5, 1).unwrap();
        logger.log_scalar("accuracy", 0.9, 1).unwrap();
        logger.log_text("Training started").unwrap();
        logger.flush().unwrap();

        // Verify file was created
        assert!(log_path.exists());

        // Read and verify contents
        let contents = fs::read_to_string(&log_path).unwrap();
        assert!(contents.contains("loss = 0.500000"));
        assert!(contents.contains("accuracy = 0.900000"));
        assert!(contents.contains("Training started"));

        // Clean up
        fs::remove_file(&log_path).unwrap();
    }

    #[test]
    fn test_file_logger_truncate() {
        let temp_dir = env::temp_dir();
        let log_path = temp_dir.join("test_training_truncate.log");

        // Create file with some content
        {
            let mut logger = FileLogger::new(&log_path).unwrap();
            logger.log_text("Old content").unwrap();
            logger.flush().unwrap();
        }

        // Truncate and write new content
        {
            let mut logger = FileLogger::new_truncate(&log_path).unwrap();
            logger.log_text("New content").unwrap();
            logger.flush().unwrap();
        }

        // Verify old content is gone
        let contents = fs::read_to_string(&log_path).unwrap();
        assert!(!contents.contains("Old content"));
        assert!(contents.contains("New content"));

        // Clean up
        fs::remove_file(&log_path).unwrap();
    }

    #[test]
    fn test_tensorboard_logger() {
        let temp_dir = env::temp_dir();
        let tb_dir = temp_dir.join("test_tensorboard");

        // Clean up if directory exists
        let _ = fs::remove_dir_all(&tb_dir);

        let mut logger = TensorBoardLogger::new(&tb_dir).unwrap();

        // Directory should be created
        assert!(tb_dir.exists());

        logger.log_scalar("loss", 0.5, 1).unwrap();
        logger.log_text("Test message").unwrap();
        logger.flush().unwrap();

        // Clean up
        fs::remove_dir_all(&tb_dir).unwrap();
    }

    #[test]
    fn test_metrics_logger() {
        let mut logger = MetricsLogger::new();
        assert_eq!(logger.num_backends(), 0);

        logger.add_backend(ConsoleLogger::without_timestamp());
        assert_eq!(logger.num_backends(), 1);

        logger.log_metric("loss", 0.5).unwrap();
        logger.log_message("Epoch 1").unwrap();

        assert_eq!(logger.current_step(), 0);
        logger.step();
        assert_eq!(logger.current_step(), 1);

        logger.set_step(10);
        assert_eq!(logger.current_step(), 10);

        logger.flush().unwrap();
    }

    #[test]
    fn test_metrics_logger_accumulation() {
        let mut logger = MetricsLogger::new();
        logger.add_backend(ConsoleLogger::without_timestamp());

        // Accumulate multiple values
        logger.accumulate_metric("batch_loss", 0.5);
        logger.accumulate_metric("batch_loss", 0.4);
        logger.accumulate_metric("batch_loss", 0.6);

        // Log accumulated (should be average: 0.5)
        logger.log_accumulated_metrics().unwrap();

        // Accumulation should be cleared
        logger.log_accumulated_metrics().unwrap(); // Should not fail even if empty
    }

    #[test]
    fn test_metrics_logger_multiple_backends() {
        let mut logger = MetricsLogger::new();
        logger.add_backend(ConsoleLogger::without_timestamp());
        logger.add_backend(ConsoleLogger::new());

        assert_eq!(logger.num_backends(), 2);

        logger.log_metric("loss", 0.5).unwrap();
        logger.flush().unwrap();
    }

    #[test]
    fn test_metrics_logger_empty_accumulation() {
        let mut logger = MetricsLogger::new();
        logger.add_backend(ConsoleLogger::without_timestamp());

        // Log without accumulating anything
        logger.log_accumulated_metrics().unwrap();
    }

    #[test]
    fn test_file_logger_path() {
        let temp_dir = env::temp_dir();
        let log_path = temp_dir.join("test_path.log");
        let _ = fs::remove_file(&log_path);

        let logger = FileLogger::new(&log_path).unwrap();
        assert_eq!(logger.path(), log_path.as_path());

        // Clean up
        fs::remove_file(&log_path).unwrap();
    }

    #[test]
    fn test_tensorboard_logger_log_dir() {
        let temp_dir = env::temp_dir();
        let tb_dir = temp_dir.join("test_tb_path");
        let _ = fs::remove_dir_all(&tb_dir);

        let logger = TensorBoardLogger::new(&tb_dir).unwrap();
        assert_eq!(logger.log_dir(), tb_dir.as_path());

        // Clean up
        fs::remove_dir_all(&tb_dir).unwrap();
    }
}
