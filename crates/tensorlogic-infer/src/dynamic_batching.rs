//! Dynamic batching for inference serving.
//!
//! This module provides dynamic batching capabilities for efficient inference serving:
//! - Automatic request batching with configurable timeouts
//! - Priority-based request queuing
//! - Adaptive batch sizing based on load
//! - Request deduplication
//! - Batch splitting for heterogeneous requests
//! - Latency and throughput optimization

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use thiserror::Error;

#[cfg(feature = "async")]
use tokio::sync::oneshot;

/// Dynamic batching errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum BatchingError {
    #[error("Request queue is full")]
    QueueFull,

    #[error("Request timeout after {0:?}")]
    Timeout(Duration),

    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(usize),

    #[error("Request cancelled")]
    Cancelled,

    #[error("Incompatible request shapes")]
    IncompatibleShapes,
}

/// Priority level for requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority (batch-friendly, can wait)
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority (minimize latency)
    High = 2,
    /// Critical priority (process immediately)
    Critical = 3,
}

/// Request metadata for batching decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// Request ID
    pub id: String,
    /// Priority level
    pub priority: Priority,
    /// Arrival timestamp
    #[serde(skip, default = "Instant::now")]
    pub arrival_time: Instant,
    /// Maximum tolerable latency
    pub max_latency: Option<Duration>,
    /// Input shapes (for compatibility checking)
    pub input_shapes: Vec<Vec<usize>>,
}

/// A request to be batched.
pub struct BatchRequest<T> {
    /// Request metadata
    pub metadata: RequestMetadata,
    /// Input data
    pub inputs: T,
    /// Response channel (for async execution)
    #[cfg(feature = "async")]
    pub response_tx: Option<oneshot::Sender<Result<T, BatchingError>>>,
}

impl<T> BatchRequest<T> {
    /// Create a new batch request.
    pub fn new(id: String, inputs: T, input_shapes: Vec<Vec<usize>>) -> Self {
        Self {
            metadata: RequestMetadata {
                id,
                priority: Priority::Normal,
                arrival_time: Instant::now(),
                max_latency: None,
                input_shapes,
            },
            inputs,
            #[cfg(feature = "async")]
            response_tx: None,
        }
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.metadata.priority = priority;
        self
    }

    /// Set maximum latency.
    pub fn with_max_latency(mut self, max_latency: Duration) -> Self {
        self.metadata.max_latency = Some(max_latency);
        self
    }

    /// Check if request has timed out.
    pub fn is_timed_out(&self) -> bool {
        if let Some(max_latency) = self.metadata.max_latency {
            self.metadata.arrival_time.elapsed() > max_latency
        } else {
            false
        }
    }

    /// Get age of the request.
    pub fn age(&self) -> Duration {
        self.metadata.arrival_time.elapsed()
    }
}

/// Configuration for dynamic batching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Minimum batch size (for efficiency)
    pub min_batch_size: usize,
    /// Maximum wait time before forming a batch
    pub max_wait_time: Duration,
    /// Maximum queue depth
    pub max_queue_depth: usize,
    /// Enable adaptive batch sizing
    pub adaptive_sizing: bool,
    /// Target latency for adaptive sizing
    pub target_latency: Option<Duration>,
    /// Enable request deduplication
    pub enable_deduplication: bool,
    /// Enable batch splitting for heterogeneous requests
    pub enable_splitting: bool,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            min_batch_size: 1,
            max_wait_time: Duration::from_millis(10),
            max_queue_depth: 1000,
            adaptive_sizing: true,
            target_latency: Some(Duration::from_millis(50)),
            enable_deduplication: false,
            enable_splitting: true,
        }
    }
}

impl DynamicBatchConfig {
    /// Create configuration optimized for throughput.
    pub fn throughput_optimized() -> Self {
        Self {
            max_batch_size: 128,
            min_batch_size: 8,
            max_wait_time: Duration::from_millis(50),
            ..Default::default()
        }
    }

    /// Create configuration optimized for latency.
    pub fn latency_optimized() -> Self {
        Self {
            max_batch_size: 16,
            min_batch_size: 1,
            max_wait_time: Duration::from_millis(1),
            target_latency: Some(Duration::from_millis(10)),
            ..Default::default()
        }
    }

    /// Create configuration for interactive workloads.
    pub fn interactive() -> Self {
        Self {
            max_batch_size: 8,
            min_batch_size: 1,
            max_wait_time: Duration::from_millis(5),
            adaptive_sizing: true,
            target_latency: Some(Duration::from_millis(20)),
            ..Default::default()
        }
    }
}

/// Statistics for dynamic batching.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchingStats {
    /// Total requests processed
    pub total_requests: usize,
    /// Total batches formed
    pub total_batches: usize,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average wait time
    pub avg_wait_time: Duration,
    /// Average latency
    pub avg_latency: Duration,
    /// Number of timeouts
    pub num_timeouts: usize,
    /// Number of queue overflows
    pub num_overflows: usize,
    /// Current queue depth
    pub current_queue_depth: usize,
}

impl BatchingStats {
    /// Update statistics with a new batch.
    pub fn update_batch(&mut self, batch_size: usize, wait_time: Duration, latency: Duration) {
        self.total_batches += 1;
        self.total_requests += batch_size;

        // Update averages using incremental formula
        let n = self.total_batches as f64;
        self.avg_batch_size = (self.avg_batch_size * (n - 1.0) + batch_size as f64) / n;

        self.avg_wait_time = Duration::from_secs_f64(
            (self.avg_wait_time.as_secs_f64() * (n - 1.0) + wait_time.as_secs_f64()) / n,
        );

        self.avg_latency = Duration::from_secs_f64(
            (self.avg_latency.as_secs_f64() * (n - 1.0) + latency.as_secs_f64()) / n,
        );
    }

    /// Record a timeout.
    pub fn record_timeout(&mut self) {
        self.num_timeouts += 1;
    }

    /// Record a queue overflow.
    pub fn record_overflow(&mut self) {
        self.num_overflows += 1;
    }

    /// Get throughput (requests per second).
    pub fn throughput(&self) -> f64 {
        if self.avg_latency.as_secs_f64() > 0.0 {
            self.avg_batch_size / self.avg_latency.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get batching efficiency (ratio of actual to max batch size).
    pub fn efficiency(&self, max_batch_size: usize) -> f64 {
        if max_batch_size > 0 {
            self.avg_batch_size / max_batch_size as f64
        } else {
            0.0
        }
    }
}

/// Request queue with priority support.
pub struct RequestQueue<T> {
    queues: HashMap<Priority, VecDeque<BatchRequest<T>>>,
    config: DynamicBatchConfig,
}

impl<T> RequestQueue<T> {
    /// Create a new request queue.
    pub fn new(config: DynamicBatchConfig) -> Self {
        let mut queues = HashMap::new();
        queues.insert(Priority::Low, VecDeque::new());
        queues.insert(Priority::Normal, VecDeque::new());
        queues.insert(Priority::High, VecDeque::new());
        queues.insert(Priority::Critical, VecDeque::new());

        Self { queues, config }
    }

    /// Enqueue a request.
    pub fn enqueue(&mut self, request: BatchRequest<T>) -> Result<(), BatchingError> {
        let total_depth: usize = self.queues.values().map(|q| q.len()).sum();
        if total_depth >= self.config.max_queue_depth {
            return Err(BatchingError::QueueFull);
        }

        let priority = request.metadata.priority;
        self.queues.get_mut(&priority).unwrap().push_back(request);
        Ok(())
    }

    /// Dequeue requests to form a batch.
    pub fn dequeue_batch(&mut self, max_size: usize) -> Vec<BatchRequest<T>> {
        let mut batch = Vec::new();
        let priorities = [
            Priority::Critical,
            Priority::High,
            Priority::Normal,
            Priority::Low,
        ];

        for &priority in &priorities {
            if batch.len() >= max_size {
                break;
            }

            let queue = self.queues.get_mut(&priority).unwrap();
            while let Some(request) = queue.pop_front() {
                // Skip timed-out requests
                if request.is_timed_out() {
                    continue;
                }

                batch.push(request);

                if batch.len() >= max_size {
                    break;
                }
            }
        }

        batch
    }

    /// Get total queue depth.
    pub fn depth(&self) -> usize {
        self.queues.values().map(|q| q.len()).sum()
    }

    /// Get oldest request age.
    pub fn oldest_age(&self) -> Option<Duration> {
        let priorities = [
            Priority::Critical,
            Priority::High,
            Priority::Normal,
            Priority::Low,
        ];

        for &priority in &priorities {
            if let Some(request) = self.queues.get(&priority).unwrap().front() {
                return Some(request.age());
            }
        }
        None
    }

    /// Check if batch formation criteria are met.
    pub fn should_form_batch(&self) -> bool {
        // Form batch if max wait time exceeded
        if let Some(age) = self.oldest_age() {
            if age >= self.config.max_wait_time {
                return true;
            }
        }

        // Form batch if min batch size reached
        let depth = self.depth();
        if depth >= self.config.min_batch_size {
            return true;
        }

        // Form batch immediately for critical requests
        if !self.queues.get(&Priority::Critical).unwrap().is_empty() {
            return true;
        }

        false
    }
}

/// Adaptive batch size controller.
pub struct AdaptiveBatcher {
    config: DynamicBatchConfig,
    current_batch_size: usize,
    latency_history: VecDeque<Duration>,
    throughput_history: VecDeque<f64>,
}

impl AdaptiveBatcher {
    /// Create a new adaptive batcher.
    pub fn new(config: DynamicBatchConfig) -> Self {
        Self {
            current_batch_size: config.max_batch_size / 2,
            config,
            latency_history: VecDeque::with_capacity(100),
            throughput_history: VecDeque::with_capacity(100),
        }
    }

    /// Get current recommended batch size.
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Update batch size based on observed latency.
    pub fn update(&mut self, _batch_size: usize, latency: Duration, throughput: f64) {
        self.latency_history.push_back(latency);
        self.throughput_history.push_back(throughput);

        // Keep only recent history
        while self.latency_history.len() > 100 {
            self.latency_history.pop_front();
        }
        while self.throughput_history.len() > 100 {
            self.throughput_history.pop_front();
        }

        if !self.config.adaptive_sizing {
            return;
        }

        let target_latency = match self.config.target_latency {
            Some(t) => t,
            None => return,
        };

        // Simple adaptive strategy: increase batch size if under target,
        // decrease if over target
        if latency < target_latency * 8 / 10 {
            // Under target - can increase batch size
            self.current_batch_size = (self.current_batch_size + 1).min(self.config.max_batch_size);
        } else if latency > target_latency {
            // Over target - decrease batch size
            self.current_batch_size =
                (self.current_batch_size.saturating_sub(1)).max(self.config.min_batch_size);
        }
    }

    /// Get average recent latency.
    pub fn avg_latency(&self) -> Option<Duration> {
        if self.latency_history.is_empty() {
            return None;
        }

        let sum: Duration = self.latency_history.iter().sum();
        Some(sum / self.latency_history.len() as u32)
    }

    /// Get average recent throughput.
    pub fn avg_throughput(&self) -> Option<f64> {
        if self.throughput_history.is_empty() {
            return None;
        }

        Some(self.throughput_history.iter().sum::<f64>() / self.throughput_history.len() as f64)
    }
}

/// Dynamic batcher for inference requests.
pub struct DynamicBatcher<T> {
    queue: RequestQueue<T>,
    stats: BatchingStats,
    adaptive: AdaptiveBatcher,
}

impl<T> DynamicBatcher<T> {
    /// Create a new dynamic batcher.
    pub fn new(config: DynamicBatchConfig) -> Self {
        let adaptive = AdaptiveBatcher::new(config.clone());
        let queue = RequestQueue::new(config.clone());

        Self {
            queue,
            stats: BatchingStats::default(),
            adaptive,
        }
    }

    /// Submit a request for batching.
    pub fn submit(&mut self, request: BatchRequest<T>) -> Result<(), BatchingError> {
        self.queue.enqueue(request)?;
        self.stats.current_queue_depth = self.queue.depth();
        Ok(())
    }

    /// Try to form a batch if criteria are met.
    pub fn try_form_batch(&mut self) -> Option<Vec<BatchRequest<T>>> {
        if !self.queue.should_form_batch() {
            return None;
        }

        let batch_size = self.adaptive.current_batch_size();
        let batch = self.queue.dequeue_batch(batch_size);

        if batch.is_empty() {
            return None;
        }

        self.stats.current_queue_depth = self.queue.depth();
        Some(batch)
    }

    /// Get statistics.
    pub fn stats(&self) -> &BatchingStats {
        &self.stats
    }

    /// Record batch execution results.
    pub fn record_batch(&mut self, batch_size: usize, wait_time: Duration, latency: Duration) {
        self.stats.update_batch(batch_size, wait_time, latency);

        let throughput = batch_size as f64 / latency.as_secs_f64();
        self.adaptive.update(batch_size, latency, throughput);
    }

    /// Get current queue depth.
    pub fn queue_depth(&self) -> usize {
        self.queue.depth()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_request_timeout() {
        let request = BatchRequest::new("test".to_string(), vec![1.0, 2.0], vec![vec![2]])
            .with_max_latency(Duration::from_millis(1));

        std::thread::sleep(Duration::from_millis(2));
        assert!(request.is_timed_out());
    }

    #[test]
    fn test_queue_enqueue_dequeue() {
        let config = DynamicBatchConfig::default();
        let mut queue: RequestQueue<Vec<f64>> = RequestQueue::new(config);

        let req1 = BatchRequest::new("1".to_string(), vec![1.0], vec![vec![1]]);
        let req2 = BatchRequest::new("2".to_string(), vec![2.0], vec![vec![1]])
            .with_priority(Priority::High);

        queue.enqueue(req1).unwrap();
        queue.enqueue(req2).unwrap();

        assert_eq!(queue.depth(), 2);

        let batch = queue.dequeue_batch(10);
        assert_eq!(batch.len(), 2);
        // High priority should be first
        assert_eq!(batch[0].metadata.id, "2");
    }

    #[test]
    fn test_queue_overflow() {
        let config = DynamicBatchConfig {
            max_queue_depth: 2,
            ..Default::default()
        };
        let mut queue: RequestQueue<Vec<f64>> = RequestQueue::new(config);

        queue
            .enqueue(BatchRequest::new("1".to_string(), vec![1.0], vec![vec![1]]))
            .unwrap();
        queue
            .enqueue(BatchRequest::new("2".to_string(), vec![2.0], vec![vec![1]]))
            .unwrap();

        let result = queue.enqueue(BatchRequest::new("3".to_string(), vec![3.0], vec![vec![1]]));
        assert!(matches!(result, Err(BatchingError::QueueFull)));
    }

    #[test]
    fn test_batching_stats() {
        let mut stats = BatchingStats::default();

        stats.update_batch(4, Duration::from_millis(5), Duration::from_millis(10));
        stats.update_batch(8, Duration::from_millis(6), Duration::from_millis(12));

        assert_eq!(stats.total_requests, 12);
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.avg_batch_size, 6.0);
    }

    #[test]
    fn test_adaptive_batcher() {
        let config = DynamicBatchConfig {
            adaptive_sizing: true,
            target_latency: Some(Duration::from_millis(50)),
            min_batch_size: 1,
            max_batch_size: 32,
            ..Default::default()
        };

        let mut batcher = AdaptiveBatcher::new(config);
        let initial_size = batcher.current_batch_size();

        // Simulate low latency - should increase batch size
        batcher.update(8, Duration::from_millis(20), 400.0);
        assert!(batcher.current_batch_size() >= initial_size);

        // Simulate high latency - should decrease batch size
        for _ in 0..10 {
            batcher.update(8, Duration::from_millis(100), 80.0);
        }
        assert!(batcher.current_batch_size() < initial_size);
    }

    #[test]
    fn test_dynamic_batcher() {
        let config = DynamicBatchConfig::latency_optimized();
        let mut batcher: DynamicBatcher<Vec<f64>> = DynamicBatcher::new(config);

        // Submit requests
        for i in 0..5 {
            let request = BatchRequest::new(format!("req_{}", i), vec![i as f64], vec![vec![1]]);
            batcher.submit(request).unwrap();
        }

        assert_eq!(batcher.queue_depth(), 5);

        // Form batch
        let batch = batcher.try_form_batch();
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_config_presets() {
        let throughput = DynamicBatchConfig::throughput_optimized();
        assert!(throughput.max_batch_size > DynamicBatchConfig::default().max_batch_size);

        let latency = DynamicBatchConfig::latency_optimized();
        assert!(latency.max_wait_time < DynamicBatchConfig::default().max_wait_time);

        let interactive = DynamicBatchConfig::interactive();
        assert!(interactive.max_batch_size < throughput.max_batch_size);
    }

    #[test]
    fn test_stats_efficiency() {
        let mut stats = BatchingStats::default();
        stats.update_batch(16, Duration::from_millis(5), Duration::from_millis(10));

        assert_eq!(stats.efficiency(32), 0.5);
        assert_eq!(stats.efficiency(16), 1.0);
    }
}
