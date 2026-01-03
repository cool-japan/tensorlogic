//! Logging infrastructure for training.
//!
//! This module provides various logging backends to track training progress:
//! - Console logging (stdout/stderr)
//! - File logging (write to file)
//! - TensorBoard logging (real event file writing)
//! - CSV logging (for easy analysis)
//! - JSONL logging (machine-readable format)
//! - Metrics logging and aggregation

use crate::{TrainError, TrainResult};
use byteorder::{LittleEndian, WriteBytesExt};
use chrono::Utc;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
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

/// TensorBoard logger that writes real event files.
///
/// Writes TensorBoard event files in the tfevents format,
/// which can be visualized using TensorBoard.
///
/// # Example
/// ```no_run
/// use tensorlogic_train::TensorBoardLogger;
/// use tensorlogic_train::LoggingBackend;
///
/// let mut logger = TensorBoardLogger::new("./logs/run1").unwrap();
/// logger.log_scalar("loss", 0.5, 1).unwrap();
/// logger.log_scalar("accuracy", 0.95, 1).unwrap();
/// logger.flush().unwrap();
/// ```
#[derive(Debug)]
pub struct TensorBoardLogger {
    log_dir: PathBuf,
    writer: BufWriter<File>,
    file_path: PathBuf,
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

        // Create event file with TensorBoard naming convention
        let timestamp = Utc::now().timestamp();
        let hostname = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "localhost".to_string());
        let filename = format!("events.out.tfevents.{}.{}", timestamp, hostname);
        let file_path = log_dir.join(&filename);

        let file = File::create(&file_path).map_err(|e| {
            TrainError::Other(format!(
                "Failed to create event file {:?}: {}",
                file_path, e
            ))
        })?;

        let mut logger = Self {
            log_dir,
            writer: BufWriter::new(file),
            file_path,
        };

        // Write initial file_version event
        logger.write_file_version()?;

        Ok(logger)
    }

    /// Get the log directory.
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }

    /// Get the event file path.
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// Write the file version event (required by TensorBoard).
    fn write_file_version(&mut self) -> TrainResult<()> {
        let wall_time = Utc::now().timestamp_micros() as f64 / 1_000_000.0;

        // Create file_version event
        let event = TensorBoardEvent {
            wall_time,
            step: 0,
            value: TensorBoardValue::FileVersion("brain.Event:2".to_string()),
        };

        self.write_event(&event)
    }

    /// Write a TensorBoard event.
    fn write_event(&mut self, event: &TensorBoardEvent) -> TrainResult<()> {
        let data = event.to_bytes();

        // TensorBoard record format:
        // uint64 length
        // uint32 masked_crc32_of_length
        // byte   data[length]
        // uint32 masked_crc32_of_data

        let length = data.len() as u64;
        let length_bytes = length.to_le_bytes();
        let length_crc = masked_crc32(&length_bytes);
        let data_crc = masked_crc32(&data);

        self.writer
            .write_u64::<LittleEndian>(length)
            .map_err(|e| TrainError::Other(format!("Failed to write event length: {}", e)))?;
        self.writer
            .write_u32::<LittleEndian>(length_crc)
            .map_err(|e| TrainError::Other(format!("Failed to write length CRC: {}", e)))?;
        self.writer
            .write_all(&data)
            .map_err(|e| TrainError::Other(format!("Failed to write event data: {}", e)))?;
        self.writer
            .write_u32::<LittleEndian>(data_crc)
            .map_err(|e| TrainError::Other(format!("Failed to write data CRC: {}", e)))?;

        Ok(())
    }

    /// Log a histogram (weight distributions).
    pub fn log_histogram(&mut self, tag: &str, values: &[f64], step: usize) -> TrainResult<()> {
        let wall_time = Utc::now().timestamp_micros() as f64 / 1_000_000.0;

        let event = TensorBoardEvent {
            wall_time,
            step: step as i64,
            value: TensorBoardValue::Histogram {
                tag: tag.to_string(),
                values: values.to_vec(),
            },
        };

        self.write_event(&event)
    }
}

impl LoggingBackend for TensorBoardLogger {
    fn log_scalar(&mut self, name: &str, value: f64, step: usize) -> TrainResult<()> {
        let wall_time = Utc::now().timestamp_micros() as f64 / 1_000_000.0;

        let event = TensorBoardEvent {
            wall_time,
            step: step as i64,
            value: TensorBoardValue::Scalar {
                tag: name.to_string(),
                value,
            },
        };

        self.write_event(&event)
    }

    fn log_text(&mut self, message: &str) -> TrainResult<()> {
        let wall_time = Utc::now().timestamp_micros() as f64 / 1_000_000.0;

        let event = TensorBoardEvent {
            wall_time,
            step: 0,
            value: TensorBoardValue::Text {
                tag: "text".to_string(),
                content: message.to_string(),
            },
        };

        self.write_event(&event)
    }

    fn flush(&mut self) -> TrainResult<()> {
        self.writer
            .flush()
            .map_err(|e| TrainError::Other(format!("Failed to flush TensorBoard writer: {}", e)))?;
        Ok(())
    }
}

/// TensorBoard event structure.
#[derive(Debug)]
struct TensorBoardEvent {
    wall_time: f64,
    step: i64,
    value: TensorBoardValue,
}

/// TensorBoard value types.
#[derive(Debug)]
enum TensorBoardValue {
    FileVersion(String),
    Scalar { tag: String, value: f64 },
    Histogram { tag: String, values: Vec<f64> },
    Text { tag: String, content: String },
}

impl TensorBoardEvent {
    /// Convert event to bytes (simplified protobuf-like format).
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write wall_time (field 1, double)
        bytes.push(0x09); // field 1, wire type 1 (64-bit)
        bytes.extend_from_slice(&self.wall_time.to_le_bytes());

        // Write step (field 2, int64)
        bytes.push(0x10); // field 2, wire type 0 (varint)
        write_varint(&mut bytes, self.step as u64);

        match &self.value {
            TensorBoardValue::FileVersion(version) => {
                // Write file_version (field 3, string)
                bytes.push(0x1a); // field 3, wire type 2 (length-delimited)
                write_varint(&mut bytes, version.len() as u64);
                bytes.extend_from_slice(version.as_bytes());
            }
            TensorBoardValue::Scalar { tag, value } => {
                // Write summary (field 5)
                let summary_bytes = encode_scalar_summary(tag, *value);
                bytes.push(0x2a); // field 5, wire type 2
                write_varint(&mut bytes, summary_bytes.len() as u64);
                bytes.extend_from_slice(&summary_bytes);
            }
            TensorBoardValue::Histogram { tag, values } => {
                // Write summary with histogram
                let summary_bytes = encode_histogram_summary(tag, values);
                bytes.push(0x2a);
                write_varint(&mut bytes, summary_bytes.len() as u64);
                bytes.extend_from_slice(&summary_bytes);
            }
            TensorBoardValue::Text { tag, content } => {
                // Write summary with text
                let summary_bytes = encode_text_summary(tag, content);
                bytes.push(0x2a);
                write_varint(&mut bytes, summary_bytes.len() as u64);
                bytes.extend_from_slice(&summary_bytes);
            }
        }

        bytes
    }
}

/// Encode a scalar summary.
fn encode_scalar_summary(tag: &str, value: f64) -> Vec<u8> {
    let mut bytes = Vec::new();

    // Summary message contains repeated Value
    // Value: tag (field 1), simple_value (field 2)
    let mut value_bytes = Vec::new();

    // tag (field 1, string)
    value_bytes.push(0x0a);
    write_varint(&mut value_bytes, tag.len() as u64);
    value_bytes.extend_from_slice(tag.as_bytes());

    // simple_value (field 2, float)
    value_bytes.push(0x15); // field 2, wire type 5 (32-bit)
    value_bytes.extend_from_slice(&(value as f32).to_le_bytes());

    // Wrap in Summary.value (field 1, repeated)
    bytes.push(0x0a);
    write_varint(&mut bytes, value_bytes.len() as u64);
    bytes.extend_from_slice(&value_bytes);

    bytes
}

/// Encode a histogram summary.
fn encode_histogram_summary(tag: &str, values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::new();

    // Compute histogram statistics
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().sum();
    let sum_squares: f64 = values.iter().map(|x| x * x).sum();

    let mut value_bytes = Vec::new();

    // tag
    value_bytes.push(0x0a);
    write_varint(&mut value_bytes, tag.len() as u64);
    value_bytes.extend_from_slice(tag.as_bytes());

    // histo (field 4) - simplified histogram encoding
    let mut histo_bytes = Vec::new();

    // min (field 1)
    histo_bytes.push(0x09);
    histo_bytes.extend_from_slice(&min.to_le_bytes());
    // max (field 2)
    histo_bytes.push(0x11);
    histo_bytes.extend_from_slice(&max.to_le_bytes());
    // num (field 3)
    histo_bytes.push(0x18);
    write_varint(&mut histo_bytes, values.len() as u64);
    // sum (field 4)
    histo_bytes.push(0x21);
    histo_bytes.extend_from_slice(&sum.to_le_bytes());
    // sum_squares (field 5)
    histo_bytes.push(0x29);
    histo_bytes.extend_from_slice(&sum_squares.to_le_bytes());

    value_bytes.push(0x22); // field 4, wire type 2
    write_varint(&mut value_bytes, histo_bytes.len() as u64);
    value_bytes.extend_from_slice(&histo_bytes);

    bytes.push(0x0a);
    write_varint(&mut bytes, value_bytes.len() as u64);
    bytes.extend_from_slice(&value_bytes);

    bytes
}

/// Encode a text summary.
fn encode_text_summary(tag: &str, content: &str) -> Vec<u8> {
    let mut bytes = Vec::new();

    let mut value_bytes = Vec::new();

    // tag
    value_bytes.push(0x0a);
    write_varint(&mut value_bytes, tag.len() as u64);
    value_bytes.extend_from_slice(tag.as_bytes());

    // tensor (field 8) for text
    let mut tensor_bytes = Vec::new();
    // dtype = DT_STRING (7)
    tensor_bytes.push(0x08);
    write_varint(&mut tensor_bytes, 7);
    // string_val (field 8)
    tensor_bytes.push(0x42);
    write_varint(&mut tensor_bytes, content.len() as u64);
    tensor_bytes.extend_from_slice(content.as_bytes());

    value_bytes.push(0x42); // field 8, wire type 2
    write_varint(&mut value_bytes, tensor_bytes.len() as u64);
    value_bytes.extend_from_slice(&tensor_bytes);

    bytes.push(0x0a);
    write_varint(&mut bytes, value_bytes.len() as u64);
    bytes.extend_from_slice(&value_bytes);

    bytes
}

/// Write a varint to the buffer.
fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// Compute masked CRC32.
fn masked_crc32(data: &[u8]) -> u32 {
    let crc = crc32fast::hash(data);
    crc.rotate_right(15).wrapping_add(0xa282ead8)
}

/// CSV logger for easy data analysis.
///
/// Writes metrics to a CSV file that can be imported into spreadsheets
/// or analyzed with pandas/numpy.
///
/// # Example
/// ```no_run
/// use tensorlogic_train::CsvLogger;
/// use tensorlogic_train::LoggingBackend;
///
/// let mut logger = CsvLogger::new("/tmp/metrics.csv").unwrap();
/// logger.log_scalar("loss", 0.5, 1).unwrap();
/// logger.log_scalar("accuracy", 0.95, 1).unwrap();
/// logger.flush().unwrap();
/// ```
#[derive(Debug)]
pub struct CsvLogger {
    writer: BufWriter<File>,
    path: PathBuf,
    header_written: bool,
}

impl CsvLogger {
    /// Create a new CSV logger.
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    pub fn new<P: AsRef<Path>>(path: P) -> TrainResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path).map_err(|e| {
            TrainError::Other(format!("Failed to create CSV file {:?}: {}", path, e))
        })?;

        let mut logger = Self {
            writer: BufWriter::new(file),
            path,
            header_written: false,
        };

        // Write header
        writeln!(logger.writer, "step,metric,value,timestamp")
            .map_err(|e| TrainError::Other(format!("Failed to write CSV header: {}", e)))?;
        logger.header_written = true;

        Ok(logger)
    }

    /// Get the path to the CSV file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl LoggingBackend for CsvLogger {
    fn log_scalar(&mut self, name: &str, value: f64, step: usize) -> TrainResult<()> {
        let timestamp = Utc::now().to_rfc3339();
        writeln!(self.writer, "{},{},{:.6},{}", step, name, value, timestamp)
            .map_err(|e| TrainError::Other(format!("Failed to write to CSV: {}", e)))?;
        Ok(())
    }

    fn log_text(&mut self, message: &str) -> TrainResult<()> {
        let timestamp = Utc::now().to_rfc3339();
        // Escape message for CSV
        let escaped = message.replace('"', "\"\"");
        writeln!(self.writer, "0,text,\"{}\",{}", escaped, timestamp)
            .map_err(|e| TrainError::Other(format!("Failed to write to CSV: {}", e)))?;
        Ok(())
    }

    fn flush(&mut self) -> TrainResult<()> {
        self.writer
            .flush()
            .map_err(|e| TrainError::Other(format!("Failed to flush CSV writer: {}", e)))?;
        Ok(())
    }
}

impl Clone for CsvLogger {
    fn clone(&self) -> Self {
        // Create a new logger pointing to the same file in append mode
        Self::new(&self.path).expect("Failed to clone CsvLogger")
    }
}

/// JSONL (JSON Lines) logger for machine-readable output.
///
/// Writes each metric as a JSON object on its own line,
/// making it easy to parse and process programmatically.
///
/// # Example
/// ```no_run
/// use tensorlogic_train::JsonlLogger;
/// use tensorlogic_train::LoggingBackend;
///
/// let mut logger = JsonlLogger::new("/tmp/metrics.jsonl").unwrap();
/// logger.log_scalar("loss", 0.5, 1).unwrap();
/// logger.log_scalar("accuracy", 0.95, 1).unwrap();
/// logger.flush().unwrap();
/// ```
#[derive(Debug)]
pub struct JsonlLogger {
    writer: BufWriter<File>,
    path: PathBuf,
}

impl JsonlLogger {
    /// Create a new JSONL logger.
    ///
    /// # Arguments
    /// * `path` - Path to the JSONL file
    pub fn new<P: AsRef<Path>>(path: P) -> TrainResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path).map_err(|e| {
            TrainError::Other(format!("Failed to create JSONL file {:?}: {}", path, e))
        })?;

        Ok(Self {
            writer: BufWriter::new(file),
            path,
        })
    }

    /// Get the path to the JSONL file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl LoggingBackend for JsonlLogger {
    fn log_scalar(&mut self, name: &str, value: f64, step: usize) -> TrainResult<()> {
        let timestamp = Utc::now().to_rfc3339();
        let json = format!(
            r#"{{"type":"scalar","step":{},"metric":"{}","value":{},"timestamp":"{}"}}"#,
            step, name, value, timestamp
        );
        writeln!(self.writer, "{}", json)
            .map_err(|e| TrainError::Other(format!("Failed to write to JSONL: {}", e)))?;
        Ok(())
    }

    fn log_text(&mut self, message: &str) -> TrainResult<()> {
        let timestamp = Utc::now().to_rfc3339();
        // Escape message for JSON
        let escaped = message.replace('\\', "\\\\").replace('"', "\\\"");
        let json = format!(
            r#"{{"type":"text","step":0,"message":"{}","timestamp":"{}"}}"#,
            escaped, timestamp
        );
        writeln!(self.writer, "{}", json)
            .map_err(|e| TrainError::Other(format!("Failed to write to JSONL: {}", e)))?;
        Ok(())
    }

    fn flush(&mut self) -> TrainResult<()> {
        self.writer
            .flush()
            .map_err(|e| TrainError::Other(format!("Failed to flush JSONL writer: {}", e)))?;
        Ok(())
    }
}

impl Clone for JsonlLogger {
    fn clone(&self) -> Self {
        // Create a new logger pointing to the same file in append mode
        Self::new(&self.path).expect("Failed to clone JsonlLogger")
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
        let tb_dir = temp_dir.join("test_tensorboard_real");

        // Clean up if directory exists
        let _ = fs::remove_dir_all(&tb_dir);

        let mut logger = TensorBoardLogger::new(&tb_dir).unwrap();

        // Directory should be created
        assert!(tb_dir.exists());

        // Log some scalars
        logger.log_scalar("loss", 0.5, 1).unwrap();
        logger.log_scalar("accuracy", 0.95, 1).unwrap();
        logger.log_text("Test message").unwrap();

        // Log a histogram
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        logger.log_histogram("weights", &values, 1).unwrap();

        logger.flush().unwrap();

        // Verify event file was created
        let event_file = logger.file_path();
        assert!(event_file.exists());
        assert!(event_file.to_string_lossy().contains("tfevents"));

        // Clean up
        fs::remove_dir_all(&tb_dir).unwrap();
    }

    #[test]
    fn test_csv_logger() {
        let temp_dir = env::temp_dir();
        let csv_path = temp_dir.join("test_metrics.csv");

        // Clean up if file exists
        let _ = fs::remove_file(&csv_path);

        let mut logger = CsvLogger::new(&csv_path).unwrap();

        logger.log_scalar("loss", 0.5, 1).unwrap();
        logger.log_scalar("accuracy", 0.95, 2).unwrap();
        logger.log_text("Training started").unwrap();
        logger.flush().unwrap();

        // Verify file was created
        assert!(csv_path.exists());

        // Read and verify contents
        let contents = fs::read_to_string(&csv_path).unwrap();
        assert!(contents.contains("step,metric,value,timestamp")); // Header
        assert!(contents.contains("1,loss,0.500000"));
        assert!(contents.contains("2,accuracy,0.950000"));
        assert!(contents.contains("Training started"));

        // Clean up
        fs::remove_file(&csv_path).unwrap();
    }

    #[test]
    fn test_jsonl_logger() {
        let temp_dir = env::temp_dir();
        let jsonl_path = temp_dir.join("test_metrics.jsonl");

        // Clean up if file exists
        let _ = fs::remove_file(&jsonl_path);

        let mut logger = JsonlLogger::new(&jsonl_path).unwrap();

        logger.log_scalar("loss", 0.5, 1).unwrap();
        logger.log_scalar("accuracy", 0.95, 2).unwrap();
        logger.log_text("Training started").unwrap();
        logger.flush().unwrap();

        // Verify file was created
        assert!(jsonl_path.exists());

        // Read and verify contents
        let contents = fs::read_to_string(&jsonl_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 3);

        // Verify first line is valid JSON
        assert!(lines[0].contains("\"type\":\"scalar\""));
        assert!(lines[0].contains("\"metric\":\"loss\""));
        assert!(lines[0].contains("\"value\":0.5"));

        // Verify second line
        assert!(lines[1].contains("\"metric\":\"accuracy\""));
        assert!(lines[1].contains("\"value\":0.95"));

        // Verify text message
        assert!(lines[2].contains("\"type\":\"text\""));
        assert!(lines[2].contains("Training started"));

        // Clean up
        fs::remove_file(&jsonl_path).unwrap();
    }

    #[test]
    fn test_csv_logger_path() {
        let temp_dir = env::temp_dir();
        let csv_path = temp_dir.join("test_csv_path.csv");
        let _ = fs::remove_file(&csv_path);

        let logger = CsvLogger::new(&csv_path).unwrap();
        assert_eq!(logger.path(), csv_path.as_path());

        // Clean up
        fs::remove_file(&csv_path).unwrap();
    }

    #[test]
    fn test_jsonl_logger_path() {
        let temp_dir = env::temp_dir();
        let jsonl_path = temp_dir.join("test_jsonl_path.jsonl");
        let _ = fs::remove_file(&jsonl_path);

        let logger = JsonlLogger::new(&jsonl_path).unwrap();
        assert_eq!(logger.path(), jsonl_path.as_path());

        // Clean up
        fs::remove_file(&jsonl_path).unwrap();
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
