//! Error recovery and fault tolerance for execution.
//!
//! This module provides mechanisms for handling failures gracefully:
//! - Partial results on execution failure
//! - Checkpoint/restart capabilities
//! - Graceful degradation strategies

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tensorlogic_ir::EinsumGraph;

/// Recovery strategy for handling execution failures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Fail immediately on any error
    FailFast,
    /// Continue execution with partial results
    ContinuePartial,
    /// Retry failed operations with exponential backoff
    RetryWithBackoff { max_retries: usize },
    /// Degrade gracefully by skipping non-critical operations
    GracefulDegradation,
}

/// Configuration for error recovery
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    pub strategy: RecoveryStrategy,
    pub checkpoint_interval: Option<usize>,
    pub max_failures: Option<usize>,
    pub timeout: Option<Duration>,
}

impl RecoveryConfig {
    pub fn fail_fast() -> Self {
        RecoveryConfig {
            strategy: RecoveryStrategy::FailFast,
            checkpoint_interval: None,
            max_failures: None,
            timeout: None,
        }
    }

    pub fn partial_results() -> Self {
        RecoveryConfig {
            strategy: RecoveryStrategy::ContinuePartial,
            checkpoint_interval: Some(10),
            max_failures: Some(5),
            timeout: None,
        }
    }

    pub fn retry(max_retries: usize) -> Self {
        RecoveryConfig {
            strategy: RecoveryStrategy::RetryWithBackoff { max_retries },
            checkpoint_interval: Some(5),
            max_failures: None,
            timeout: Some(Duration::from_secs(300)), // 5 minutes
        }
    }

    pub fn graceful() -> Self {
        RecoveryConfig {
            strategy: RecoveryStrategy::GracefulDegradation,
            checkpoint_interval: Some(10),
            max_failures: Some(10),
            timeout: Some(Duration::from_secs(600)), // 10 minutes
        }
    }

    pub fn with_checkpointing(mut self, interval: usize) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_max_failures(mut self, max: usize) -> Self {
        self.max_failures = Some(max);
        self
    }
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self::partial_results()
    }
}

/// Result of execution with recovery information
#[derive(Debug, Clone)]
pub struct RecoveryResult<T> {
    /// Successfully computed outputs
    pub outputs: Vec<T>,
    /// Indices of failed operations
    pub failures: Vec<FailureInfo>,
    /// Total number of operations attempted
    pub total_operations: usize,
    /// Whether execution completed successfully
    pub success: bool,
    /// Recovery metadata
    pub metadata: RecoveryMetadata,
}

impl<T> RecoveryResult<T> {
    pub fn success(outputs: Vec<T>) -> Self {
        let total = outputs.len();
        RecoveryResult {
            outputs,
            failures: Vec::new(),
            total_operations: total,
            success: true,
            metadata: RecoveryMetadata::default(),
        }
    }

    pub fn partial(
        outputs: Vec<T>,
        failures: Vec<FailureInfo>,
        total_operations: usize,
        metadata: RecoveryMetadata,
    ) -> Self {
        RecoveryResult {
            outputs,
            failures,
            total_operations,
            success: false,
            metadata,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            return 0.0;
        }
        (self.outputs.len() as f64) / (self.total_operations as f64)
    }

    pub fn failure_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }

    pub fn has_failures(&self) -> bool {
        !self.failures.is_empty()
    }
}

/// Information about a failed operation
#[derive(Debug, Clone)]
pub struct FailureInfo {
    pub operation_id: usize,
    pub error: String,
    pub retry_count: usize,
    pub timestamp: Instant,
}

impl FailureInfo {
    pub fn new(operation_id: usize, error: String) -> Self {
        FailureInfo {
            operation_id,
            error,
            retry_count: 0,
            timestamp: Instant::now(),
        }
    }

    pub fn with_retries(mut self, count: usize) -> Self {
        self.retry_count = count;
        self
    }
}

/// Metadata about recovery process
#[derive(Debug, Clone)]
pub struct RecoveryMetadata {
    pub total_retries: usize,
    pub checkpoints_created: usize,
    pub execution_time: Duration,
    pub recovery_strategy_used: RecoveryStrategy,
}

impl RecoveryMetadata {
    pub fn new(strategy: RecoveryStrategy) -> Self {
        RecoveryMetadata {
            total_retries: 0,
            checkpoints_created: 0,
            execution_time: Duration::default(),
            recovery_strategy_used: strategy,
        }
    }
}

impl Default for RecoveryMetadata {
    fn default() -> Self {
        Self::new(RecoveryStrategy::FailFast)
    }
}

/// Checkpoint for saving execution state
#[derive(Debug, Clone)]
pub struct Checkpoint<T> {
    pub checkpoint_id: usize,
    pub operation_index: usize,
    pub partial_results: Vec<T>,
    pub timestamp: Instant,
}

impl<T: Clone> Checkpoint<T> {
    pub fn new(checkpoint_id: usize, operation_index: usize, partial_results: Vec<T>) -> Self {
        Checkpoint {
            checkpoint_id,
            operation_index,
            partial_results,
            timestamp: Instant::now(),
        }
    }

    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

/// Manager for checkpoints during execution
pub struct CheckpointManager<T> {
    checkpoints: Vec<Checkpoint<T>>,
    max_checkpoints: usize,
}

impl<T: Clone> CheckpointManager<T> {
    pub fn new(max_checkpoints: usize) -> Self {
        CheckpointManager {
            checkpoints: Vec::new(),
            max_checkpoints,
        }
    }

    pub fn create_checkpoint(&mut self, operation_index: usize, partial_results: Vec<T>) -> usize {
        let checkpoint_id = self.checkpoints.len();
        let checkpoint = Checkpoint::new(checkpoint_id, operation_index, partial_results);

        self.checkpoints.push(checkpoint);

        // Evict oldest checkpoint if we exceed max
        if self.checkpoints.len() > self.max_checkpoints {
            self.checkpoints.remove(0);
        }

        checkpoint_id
    }

    pub fn restore_checkpoint(&self, checkpoint_id: usize) -> Option<&Checkpoint<T>> {
        self.checkpoints.get(checkpoint_id)
    }

    pub fn latest_checkpoint(&self) -> Option<&Checkpoint<T>> {
        self.checkpoints.last()
    }

    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
}

// Note: No Default impl because it requires T: Clone for new()

/// Trait for executors with recovery capabilities
pub trait TlRecoverableExecutor {
    type Tensor;
    type Error;

    /// Execute graph with recovery configuration
    fn execute_with_recovery(
        &mut self,
        graph: &EinsumGraph,
        inputs: Vec<Self::Tensor>,
        config: &RecoveryConfig,
    ) -> Result<RecoveryResult<Self::Tensor>, Self::Error>;

    /// Create a checkpoint of current execution state
    fn create_checkpoint(&mut self, operation_index: usize) -> Result<usize, Self::Error>;

    /// Restore from a checkpoint
    fn restore_checkpoint(&mut self, checkpoint_id: usize) -> Result<(), Self::Error>;

    /// Get recovery statistics
    fn recovery_stats(&self) -> RecoveryStats;
}

/// Statistics about recovery operations
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    pub total_recoveries: usize,
    pub successful_recoveries: usize,
    pub failed_recoveries: usize,
    pub total_retries: usize,
    pub total_checkpoints: usize,
}

impl RecoveryStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_recovery(&mut self, success: bool) {
        self.total_recoveries += 1;
        if success {
            self.successful_recoveries += 1;
        } else {
            self.failed_recoveries += 1;
        }
    }

    pub fn record_retry(&mut self) {
        self.total_retries += 1;
    }

    pub fn record_checkpoint(&mut self) {
        self.total_checkpoints += 1;
    }

    pub fn recovery_rate(&self) -> f64 {
        if self.total_recoveries == 0 {
            return 0.0;
        }
        (self.successful_recoveries as f64) / (self.total_recoveries as f64)
    }
}

/// Retry policy with exponential backoff
pub struct RetryPolicy {
    max_retries: usize,
    base_delay_ms: u64,
    max_delay_ms: u64,
    backoff_multiplier: f64,
}

impl RetryPolicy {
    pub fn new(max_retries: usize) -> Self {
        RetryPolicy {
            max_retries,
            base_delay_ms: 100,
            max_delay_ms: 10_000,
            backoff_multiplier: 2.0,
        }
    }

    pub fn exponential(max_retries: usize, base_delay_ms: u64) -> Self {
        RetryPolicy {
            max_retries,
            base_delay_ms,
            max_delay_ms: 60_000, // 1 minute max
            backoff_multiplier: 2.0,
        }
    }

    pub fn calculate_delay(&self, retry_count: usize) -> Duration {
        if retry_count >= self.max_retries {
            return Duration::from_millis(self.max_delay_ms);
        }

        let delay_ms =
            (self.base_delay_ms as f64) * self.backoff_multiplier.powi(retry_count as i32);
        let delay_ms = delay_ms.min(self.max_delay_ms as f64) as u64;

        Duration::from_millis(delay_ms)
    }

    pub fn should_retry(&self, retry_count: usize) -> bool {
        retry_count < self.max_retries
    }

    pub fn max_retries(&self) -> usize {
        self.max_retries
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(3)
    }
}

/// Degradation policy for graceful degradation
#[derive(Debug, Clone)]
pub struct DegradationPolicy {
    /// Operations that can be skipped without critical failure
    pub skippable_operations: Vec<usize>,
    /// Fallback strategies for failed operations
    pub fallback_strategies: HashMap<usize, FallbackStrategy>,
}

impl DegradationPolicy {
    pub fn new() -> Self {
        DegradationPolicy {
            skippable_operations: Vec::new(),
            fallback_strategies: HashMap::new(),
        }
    }

    pub fn mark_skippable(mut self, operation_id: usize) -> Self {
        self.skippable_operations.push(operation_id);
        self
    }

    pub fn with_fallback(mut self, operation_id: usize, strategy: FallbackStrategy) -> Self {
        self.fallback_strategies.insert(operation_id, strategy);
        self
    }

    pub fn can_skip(&self, operation_id: usize) -> bool {
        self.skippable_operations.contains(&operation_id)
    }

    pub fn get_fallback(&self, operation_id: usize) -> Option<&FallbackStrategy> {
        self.fallback_strategies.get(&operation_id)
    }
}

impl Default for DegradationPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Fallback strategy for failed operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackStrategy {
    /// Skip the operation entirely
    Skip,
    /// Use a default/zero value
    UseDefault,
    /// Use result from a previous successful execution
    UseCached,
    /// Use a simpler approximation
    UseApproximation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_config() {
        let config = RecoveryConfig::partial_results()
            .with_checkpointing(20)
            .with_max_failures(3);

        assert_eq!(config.strategy, RecoveryStrategy::ContinuePartial);
        assert_eq!(config.checkpoint_interval, Some(20));
        assert_eq!(config.max_failures, Some(3));
    }

    #[test]
    fn test_recovery_config_retry() {
        let config = RecoveryConfig::retry(5);
        assert_eq!(
            config.strategy,
            RecoveryStrategy::RetryWithBackoff { max_retries: 5 }
        );
        assert!(config.timeout.is_some());
    }

    #[test]
    fn test_recovery_result_success() {
        let result: RecoveryResult<i32> = RecoveryResult::success(vec![1, 2, 3]);
        assert!(result.success);
        assert_eq!(result.success_rate(), 1.0);
        assert_eq!(result.failure_rate(), 0.0);
        assert!(!result.has_failures());
    }

    #[test]
    fn test_recovery_result_partial() {
        let failures = vec![FailureInfo::new(2, "Error".to_string())];
        let metadata = RecoveryMetadata::new(RecoveryStrategy::ContinuePartial);
        let result: RecoveryResult<i32> =
            RecoveryResult::partial(vec![1, 2], failures, 3, metadata);

        assert!(!result.success);
        assert_eq!(result.success_rate(), 2.0 / 3.0);
        assert!(result.has_failures());
        assert_eq!(result.failures.len(), 1);
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut manager: CheckpointManager<i32> = CheckpointManager::new(3);

        let id1 = manager.create_checkpoint(0, vec![1, 2, 3]);
        let _id2 = manager.create_checkpoint(1, vec![4, 5, 6]);
        let _id3 = manager.create_checkpoint(2, vec![7, 8, 9]);

        assert_eq!(manager.num_checkpoints(), 3);

        let checkpoint = manager.restore_checkpoint(id1).unwrap();
        assert_eq!(checkpoint.checkpoint_id, 0);
        assert_eq!(checkpoint.partial_results, vec![1, 2, 3]);

        // Add one more, should evict the oldest
        manager.create_checkpoint(3, vec![10, 11, 12]);
        assert_eq!(manager.num_checkpoints(), 3);
    }

    #[test]
    fn test_checkpoint_manager_latest() {
        let mut manager: CheckpointManager<i32> = CheckpointManager::new(5);

        manager.create_checkpoint(0, vec![1]);
        manager.create_checkpoint(1, vec![2]);
        manager.create_checkpoint(2, vec![3]);

        let latest = manager.latest_checkpoint().unwrap();
        assert_eq!(latest.checkpoint_id, 2);
        assert_eq!(latest.partial_results, vec![3]);
    }

    #[test]
    fn test_recovery_stats() {
        let mut stats = RecoveryStats::new();

        stats.record_recovery(true);
        stats.record_recovery(true);
        stats.record_recovery(false);
        stats.record_retry();
        stats.record_retry();
        stats.record_checkpoint();

        assert_eq!(stats.total_recoveries, 3);
        assert_eq!(stats.successful_recoveries, 2);
        assert_eq!(stats.failed_recoveries, 1);
        assert_eq!(stats.total_retries, 2);
        assert_eq!(stats.total_checkpoints, 1);
        assert!((stats.recovery_rate() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_retry_policy() {
        let policy = RetryPolicy::new(3);

        assert!(policy.should_retry(0));
        assert!(policy.should_retry(2));
        assert!(!policy.should_retry(3));
        assert!(!policy.should_retry(4));

        let delay1 = policy.calculate_delay(0);
        let delay2 = policy.calculate_delay(1);
        let delay3 = policy.calculate_delay(2);

        // Exponential backoff
        assert!(delay2 > delay1);
        assert!(delay3 > delay2);
    }

    #[test]
    fn test_retry_policy_exponential() {
        let policy = RetryPolicy::exponential(5, 50);

        let delay0 = policy.calculate_delay(0);
        let delay1 = policy.calculate_delay(1);
        let delay2 = policy.calculate_delay(2);

        assert_eq!(delay0.as_millis(), 50);
        assert_eq!(delay1.as_millis(), 100);
        assert_eq!(delay2.as_millis(), 200);
    }

    #[test]
    fn test_degradation_policy() {
        let policy = DegradationPolicy::new()
            .mark_skippable(1)
            .mark_skippable(3)
            .with_fallback(2, FallbackStrategy::UseDefault);

        assert!(policy.can_skip(1));
        assert!(!policy.can_skip(2));
        assert!(policy.can_skip(3));

        let fallback = policy.get_fallback(2);
        assert_eq!(fallback, Some(&FallbackStrategy::UseDefault));
        assert!(policy.get_fallback(1).is_none());
    }

    #[test]
    fn test_failure_info() {
        let info = FailureInfo::new(5, "Test error".to_string()).with_retries(3);

        assert_eq!(info.operation_id, 5);
        assert_eq!(info.error, "Test error");
        assert_eq!(info.retry_count, 3);
    }

    #[test]
    fn test_checkpoint_age() {
        let checkpoint: Checkpoint<i32> = Checkpoint::new(0, 0, vec![1, 2, 3]);
        std::thread::sleep(Duration::from_millis(10));
        let age = checkpoint.age();
        assert!(age >= Duration::from_millis(10));
    }
}
