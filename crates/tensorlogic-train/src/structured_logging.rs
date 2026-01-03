//! Structured logging support using the `tracing` crate.
//!
//! This module provides integration with the `tracing` ecosystem for
//! structured, context-aware logging. It's especially useful for:
//! - Production debugging with structured data
//! - Distributed tracing across training components
//! - Performance profiling and diagnostics
//! - Integration with observability platforms
//!
//! # Features
//!
//! This module is only available when the `structured-logging` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! tensorlogic-train = { version = "0.1", features = ["structured-logging"] }
//! ```
//!
//! # Examples
//!
//! ```no_run
//! use tensorlogic_train::structured_logging::{TracingLogger, LogFormat, LogLevel};
//!
//! // Initialize with JSON format for production
//! let logger = TracingLogger::builder()
//!     .with_format(LogFormat::Json)
//!     .with_level(LogLevel::Info)
//!     .build()
//!     .expect("Failed to initialize logger");
//!
//! // Use tracing macros for structured logging
//! tracing::info!(epoch = 5, loss = 0.25, "Training progress");
//! tracing::warn!(gradient_norm = 1e6, "Large gradient detected");
//! ```

#[cfg(feature = "structured-logging")]
use crate::{TrainError, TrainResult};

#[cfg(feature = "structured-logging")]
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

/// Log output format.
#[cfg(feature = "structured-logging")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-readable format with colors (for development).
    Pretty,
    /// Compact format without colors (for production).
    Compact,
    /// JSON format (for machine parsing and log aggregation).
    Json,
}

/// Log level filter.
#[cfg(feature = "structured-logging")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Show all logs (trace level).
    Trace,
    /// Show debug and higher.
    Debug,
    /// Show info and higher (default for production).
    Info,
    /// Show warnings and errors only.
    Warn,
    /// Show only errors.
    Error,
}

#[cfg(feature = "structured-logging")]
impl LogLevel {
    fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

/// Configuration builder for structured logging.
#[cfg(feature = "structured-logging")]
#[derive(Debug, Clone)]
pub struct TracingLoggerBuilder {
    format: LogFormat,
    level: LogLevel,
    env_filter: Option<String>,
    with_targets: bool,
    with_file_location: bool,
    with_thread_ids: bool,
    with_span_events: bool,
}

#[cfg(feature = "structured-logging")]
impl Default for TracingLoggerBuilder {
    fn default() -> Self {
        Self {
            format: LogFormat::Pretty,
            level: LogLevel::Info,
            env_filter: None,
            with_targets: true,
            with_file_location: false,
            with_thread_ids: false,
            with_span_events: false,
        }
    }
}

#[cfg(feature = "structured-logging")]
impl TracingLoggerBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the output format.
    pub fn with_format(mut self, format: LogFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the log level filter.
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.level = level;
        self
    }

    /// Set a custom environment filter (overrides level setting).
    ///
    /// # Examples
    ///
    /// ```
    /// # use tensorlogic_train::structured_logging::TracingLoggerBuilder;
    /// let builder = TracingLoggerBuilder::new()
    ///     .with_env_filter("tensorlogic=debug,scirs2=info");
    /// ```
    pub fn with_env_filter(mut self, filter: impl Into<String>) -> Self {
        self.env_filter = Some(filter.into());
        self
    }

    /// Include target names in logs (module paths).
    pub fn with_targets(mut self, enabled: bool) -> Self {
        self.with_targets = enabled;
        self
    }

    /// Include file locations (file:line) in logs.
    pub fn with_file_location(mut self, enabled: bool) -> Self {
        self.with_file_location = enabled;
        self
    }

    /// Include thread IDs in logs.
    pub fn with_thread_ids(mut self, enabled: bool) -> Self {
        self.with_thread_ids = enabled;
        self
    }

    /// Enable span lifecycle events (enter/exit).
    pub fn with_span_events(mut self, enabled: bool) -> Self {
        self.with_span_events = enabled;
        self
    }

    /// Build and initialize the logger.
    ///
    /// This must be called only once per application.
    /// Subsequent calls will return an error.
    pub fn build(self) -> TrainResult<TracingLogger> {
        // Create environment filter
        let env_filter = if let Some(custom_filter) = self.env_filter {
            EnvFilter::try_new(custom_filter)
                .map_err(|e| TrainError::Other(format!("Invalid env filter: {}", e)))?
        } else {
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(self.level.as_str()))
        };

        // Configure span events
        let span_events = if self.with_span_events {
            FmtSpan::NEW | FmtSpan::CLOSE
        } else {
            FmtSpan::NONE
        };

        // Build the subscriber based on format
        match self.format {
            LogFormat::Pretty => {
                let layer = fmt::layer()
                    .with_target(self.with_targets)
                    .with_file(self.with_file_location)
                    .with_line_number(self.with_file_location)
                    .with_thread_ids(self.with_thread_ids)
                    .with_span_events(span_events)
                    .pretty();

                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(layer)
                    .try_init()
                    .map_err(|e| {
                        TrainError::Other(format!("Failed to initialize tracing: {}", e))
                    })?;
            }
            LogFormat::Compact => {
                let layer = fmt::layer()
                    .with_target(self.with_targets)
                    .with_file(self.with_file_location)
                    .with_line_number(self.with_file_location)
                    .with_thread_ids(self.with_thread_ids)
                    .with_span_events(span_events)
                    .with_ansi(false)
                    .compact();

                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(layer)
                    .try_init()
                    .map_err(|e| {
                        TrainError::Other(format!("Failed to initialize tracing: {}", e))
                    })?;
            }
            LogFormat::Json => {
                let layer = fmt::layer()
                    .with_target(self.with_targets)
                    .with_file(self.with_file_location)
                    .with_line_number(self.with_file_location)
                    .with_thread_ids(self.with_thread_ids)
                    .with_span_events(span_events)
                    .json();

                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(layer)
                    .try_init()
                    .map_err(|e| {
                        TrainError::Other(format!("Failed to initialize tracing: {}", e))
                    })?;
            }
        }

        Ok(TracingLogger {
            _format: self.format,
        })
    }
}

/// Structured logger using the `tracing` ecosystem.
///
/// This logger provides structured, context-aware logging with
/// support for multiple output formats and advanced features like
/// distributed tracing, performance profiling, and integration
/// with observability platforms.
///
/// # Examples
///
/// ```no_run
/// use tensorlogic_train::structured_logging::{TracingLogger, LogFormat};
///
/// // Initialize logger
/// let logger = TracingLogger::builder()
///     .with_format(LogFormat::Json)
///     .build()
///     .expect("Failed to initialize logger");
///
/// // Use tracing macros
/// tracing::info!(model = "resnet50", "Starting training");
/// tracing::debug!(batch_size = 32, "Processing batch");
/// ```
#[cfg(feature = "structured-logging")]
#[derive(Debug)]
pub struct TracingLogger {
    _format: LogFormat,
}

#[cfg(feature = "structured-logging")]
impl TracingLogger {
    /// Create a new logger builder.
    pub fn builder() -> TracingLoggerBuilder {
        TracingLoggerBuilder::new()
    }

    /// Initialize with default settings (pretty format, info level).
    ///
    /// This is a convenience method equivalent to:
    /// ```no_run
    /// # use tensorlogic_train::structured_logging::TracingLogger;
    /// TracingLogger::builder().build()
    /// # ;
    /// ```
    pub fn init() -> TrainResult<Self> {
        Self::builder().build()
    }

    /// Initialize for production (JSON format, info level).
    pub fn init_production() -> TrainResult<Self> {
        Self::builder()
            .with_format(LogFormat::Json)
            .with_level(LogLevel::Info)
            .with_targets(false)
            .build()
    }

    /// Initialize for development (pretty format, debug level).
    pub fn init_development() -> TrainResult<Self> {
        Self::builder()
            .with_format(LogFormat::Pretty)
            .with_level(LogLevel::Debug)
            .with_file_location(true)
            .build()
    }
}

/// Training-specific structured logging utilities.
#[cfg(feature = "structured-logging")]
pub mod training {
    /// Log training epoch completion.
    #[macro_export]
    macro_rules! log_epoch {
        ($epoch:expr, $loss:expr, $($key:ident = $value:expr),* $(,)?) => {
            tracing::info!(
                epoch = $epoch,
                loss = $loss,
                $($key = $value,)*
                "Epoch completed"
            );
        };
    }

    /// Log batch processing.
    #[macro_export]
    macro_rules! log_batch {
        ($batch:expr, $loss:expr, $($key:ident = $value:expr),* $(,)?) => {
            tracing::debug!(
                batch = $batch,
                loss = $loss,
                $($key = $value,)*
                "Batch processed"
            );
        };
    }

    /// Log gradient statistics.
    #[macro_export]
    macro_rules! log_gradients {
        ($norm:expr, $($key:ident = $value:expr),* $(,)?) => {
            tracing::trace!(
                gradient_norm = $norm,
                $($key = $value,)*
                "Gradient statistics"
            );
        };
    }

    /// Create a training span for scoped logging.
    #[macro_export]
    macro_rules! training_span {
        ($name:expr, $($key:ident = $value:expr),* $(,)?) => {
            tracing::info_span!($name, $($key = $value,)*)
        };
    }
}

#[cfg(all(test, feature = "structured-logging"))]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = TracingLoggerBuilder::new();
        assert_eq!(builder.format, LogFormat::Pretty);
        assert_eq!(builder.level, LogLevel::Info);
    }

    #[test]
    fn test_builder_configuration() {
        let builder = TracingLoggerBuilder::new()
            .with_format(LogFormat::Json)
            .with_level(LogLevel::Debug)
            .with_targets(false)
            .with_file_location(true)
            .with_thread_ids(true)
            .with_span_events(true);

        assert_eq!(builder.format, LogFormat::Json);
        assert_eq!(builder.level, LogLevel::Debug);
        assert!(!builder.with_targets);
        assert!(builder.with_file_location);
        assert!(builder.with_thread_ids);
        assert!(builder.with_span_events);
    }

    #[test]
    fn test_log_level_as_str() {
        assert_eq!(LogLevel::Trace.as_str(), "trace");
        assert_eq!(LogLevel::Debug.as_str(), "debug");
        assert_eq!(LogLevel::Info.as_str(), "info");
        assert_eq!(LogLevel::Warn.as_str(), "warn");
        assert_eq!(LogLevel::Error.as_str(), "error");
    }

    #[test]
    fn test_custom_env_filter() {
        let builder = TracingLoggerBuilder::new().with_env_filter("tensorlogic=debug,scirs2=info");

        assert!(builder.env_filter.is_some());
        assert_eq!(builder.env_filter.unwrap(), "tensorlogic=debug,scirs2=info");
    }
}
