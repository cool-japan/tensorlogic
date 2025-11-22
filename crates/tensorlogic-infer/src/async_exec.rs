//! Asynchronous execution traits for concurrent tensor operations.
//!
//! This module provides async/await-based execution interfaces for
//! non-blocking tensor computations and streaming operations.
//!
//! Note: Async support requires the "async" feature flag.

#[cfg(feature = "async")]
use std::collections::HashMap;
#[cfg(feature = "async")]
use std::future::Future;
#[cfg(feature = "async")]
use std::pin::Pin;

#[cfg(feature = "async")]
use tensorlogic_ir::EinsumGraph;

#[cfg(feature = "async")]
use crate::batch::BatchResult;
#[cfg(feature = "async")]
use crate::streaming::{StreamResult, StreamingConfig};

/// Type alias for pinned boxed futures
#[cfg(feature = "async")]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Asynchronous executor trait for non-blocking execution
///
/// This trait enables concurrent execution of multiple graphs without blocking,
/// making it suitable for high-throughput inference servers and streaming applications.
#[cfg(feature = "async")]
pub trait TlAsyncExecutor {
    type Tensor: Send;
    type Error: Send;

    /// Execute a graph asynchronously
    fn execute_async<'a>(
        &'a mut self,
        graph: &'a EinsumGraph,
        inputs: &'a HashMap<String, Self::Tensor>,
    ) -> BoxFuture<'a, Result<Vec<Self::Tensor>, Self::Error>>;

    /// Check if executor is ready (non-blocking)
    fn is_ready(&self) -> bool {
        true
    }

    /// Wait until executor is ready
    fn wait_ready(&mut self) -> BoxFuture<'_, ()>
    where
        Self: Send,
    {
        Box::pin(async move {
            while !self.is_ready() {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        })
    }
}

/// Asynchronous batch executor
#[cfg(feature = "async")]
pub trait TlAsyncBatchExecutor: TlAsyncExecutor {
    /// Execute a batch asynchronously
    fn execute_batch_async<'a>(
        &'a mut self,
        graph: &'a EinsumGraph,
        batch_inputs: Vec<HashMap<String, Self::Tensor>>,
    ) -> BoxFuture<'a, Result<BatchResult<Self::Tensor>, Self::Error>>;
}

/// Type alias for async stream results
#[cfg(feature = "async")]
pub type AsyncStreamResults<T, E> = Vec<Result<StreamResult<T>, E>>;

/// Asynchronous streaming executor
#[cfg(feature = "async")]
pub trait TlAsyncStreamExecutor: TlAsyncExecutor {
    /// Execute stream asynchronously with chunking
    fn execute_stream_async<'a>(
        &'a mut self,
        graph: &'a EinsumGraph,
        input_stream: Vec<Vec<Vec<Self::Tensor>>>,
        config: &'a StreamingConfig,
    ) -> BoxFuture<'a, AsyncStreamResults<Self::Tensor, Self::Error>>;
}

/// Errors specific to async execution
#[derive(Debug, Clone)]
pub enum AsyncExecutionError<E> {
    /// Execution timed out
    Timeout { elapsed_ms: u64 },
    /// Executor is busy (overloaded)
    ExecutorBusy { queue_size: usize },
    /// Cancellation requested
    Cancelled,
    /// Underlying executor error
    ExecutorError(E),
    /// Future was dropped before completion
    Dropped,
}

impl<E: std::fmt::Display> std::fmt::Display for AsyncExecutionError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout { elapsed_ms } => {
                write!(f, "Execution timed out after {}ms", elapsed_ms)
            }
            Self::ExecutorBusy { queue_size } => {
                write!(
                    f,
                    "Executor is busy (queue size: {}), try again later",
                    queue_size
                )
            }
            Self::Cancelled => write!(f, "Execution was cancelled"),
            Self::ExecutorError(e) => write!(f, "Executor error: {}", e),
            Self::Dropped => write!(f, "Future was dropped before completion"),
        }
    }
}

impl<E: std::error::Error> std::error::Error for AsyncExecutionError<E> {}

/// Async execution handle for tracking and cancellation
#[cfg(feature = "async")]
pub struct AsyncExecutionHandle {
    execution_id: String,
    started_at: std::time::Instant,
    cancel_token: tokio::sync::mpsc::Sender<()>,
}

#[cfg(feature = "async")]
impl AsyncExecutionHandle {
    /// Create a new execution handle
    pub fn new(execution_id: String) -> (Self, tokio::sync::mpsc::Receiver<()>) {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        (
            AsyncExecutionHandle {
                execution_id,
                started_at: std::time::Instant::now(),
                cancel_token: tx,
            },
            rx,
        )
    }

    /// Get execution ID
    pub fn execution_id(&self) -> &str {
        &self.execution_id
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }

    /// Request cancellation
    pub async fn cancel(&self) -> Result<(), AsyncExecutionError<std::io::Error>> {
        self.cancel_token
            .send(())
            .await
            .map_err(|_| AsyncExecutionError::Cancelled)
    }
}

/// Async executor pool for load balancing
#[cfg(feature = "async")]
pub struct AsyncExecutorPool<E: TlAsyncExecutor> {
    executors: Vec<E>,
    next_index: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "async")]
impl<E: TlAsyncExecutor> AsyncExecutorPool<E> {
    /// Create a new executor pool
    pub fn new(executors: Vec<E>) -> Self {
        AsyncExecutorPool {
            executors,
            next_index: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get number of executors in pool
    pub fn size(&self) -> usize {
        self.executors.len()
    }

    /// Get next executor index (round-robin)
    pub fn get_next_index(&self) -> usize {
        self.next_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.executors.len()
    }

    /// Get least loaded executor index
    pub fn get_least_loaded_index(&self) -> usize {
        // Simple implementation: return first ready executor
        // In production, track actual load per executor
        for (idx, executor) in self.executors.iter().enumerate() {
            if executor.is_ready() {
                return idx;
            }
        }
        0
    }

    /// Execute on any available executor
    pub async fn execute_any<'a>(
        &'a mut self,
        graph: &'a EinsumGraph,
        inputs: &'a HashMap<String, E::Tensor>,
    ) -> Result<Vec<E::Tensor>, E::Error> {
        let index = self.get_least_loaded_index();
        self.executors[index].execute_async(graph, inputs).await
    }
}

/// Configuration for async execution
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// Maximum number of concurrent executions
    pub max_concurrent: usize,
    /// Timeout for each execution (milliseconds)
    pub timeout_ms: Option<u64>,
    /// Enable automatic retry on transient failures
    pub enable_retry: bool,
    /// Maximum number of retries
    pub max_retries: usize,
    /// Backoff strategy for retries
    pub backoff_ms: u64,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        AsyncConfig {
            max_concurrent: 4,
            timeout_ms: None,
            enable_retry: false,
            max_retries: 3,
            backoff_ms: 100,
        }
    }
}

impl AsyncConfig {
    /// Create a new async configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum concurrent executions
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Enable retry
    pub fn with_retry(mut self, max_retries: usize, backoff_ms: u64) -> Self {
        self.enable_retry = true;
        self.max_retries = max_retries;
        self.backoff_ms = backoff_ms;
        self
    }
}

/// Async execution statistics
#[derive(Debug, Clone, Default)]
pub struct AsyncStats {
    /// Total executions started
    pub total_executions: usize,
    /// Successful completions
    pub successful: usize,
    /// Failed executions
    pub failed: usize,
    /// Timed out executions
    pub timeouts: usize,
    /// Cancelled executions
    pub cancelled: usize,
    /// Average execution time (milliseconds)
    pub avg_execution_time_ms: f64,
    /// Peak concurrent executions
    pub peak_concurrent: usize,
}

impl AsyncStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.successful as f64 / self.total_executions as f64
        }
    }

    /// Summary report
    pub fn summary(&self) -> String {
        format!(
            "Async Execution Stats:\n\
             - Total: {}\n\
             - Successful: {} ({:.1}%)\n\
             - Failed: {}\n\
             - Timeouts: {}\n\
             - Cancelled: {}\n\
             - Avg time: {:.2}ms\n\
             - Peak concurrent: {}",
            self.total_executions,
            self.successful,
            self.success_rate() * 100.0,
            self.failed,
            self.timeouts,
            self.cancelled,
            self.avg_execution_time_ms,
            self.peak_concurrent
        )
    }
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;

    #[test]
    fn test_async_config() {
        let config = AsyncConfig::new()
            .with_max_concurrent(8)
            .with_timeout(5000)
            .with_retry(3, 200);

        assert_eq!(config.max_concurrent, 8);
        assert_eq!(config.timeout_ms, Some(5000));
        assert!(config.enable_retry);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.backoff_ms, 200);
    }

    #[test]
    fn test_async_stats() {
        let mut stats = AsyncStats::new();
        stats.total_executions = 100;
        stats.successful = 95;
        stats.failed = 3;
        stats.timeouts = 2;

        assert_eq!(stats.success_rate(), 0.95);
        assert!(stats.summary().contains("95.0%"));
    }

    #[test]
    fn test_async_error_display() {
        let err = AsyncExecutionError::<String>::Timeout { elapsed_ms: 5000 };
        assert_eq!(err.to_string(), "Execution timed out after 5000ms");

        let err2 = AsyncExecutionError::<String>::ExecutorBusy { queue_size: 10 };
        assert!(err2.to_string().contains("queue size: 10"));
    }

    #[tokio::test]
    async fn test_execution_handle() {
        let (handle, mut rx) = AsyncExecutionHandle::new("test-123".to_string());
        assert_eq!(handle.execution_id(), "test-123");
        assert!(handle.elapsed().as_millis() < 100);

        // Test cancellation
        handle.cancel().await.unwrap();
        assert!(rx.recv().await.is_some());
    }
}
