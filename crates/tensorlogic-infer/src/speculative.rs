//! Speculative execution for computation graphs.
//!
//! This module implements speculative execution techniques:
//! - **Branch prediction**: Predict conditional branches and execute speculatively
//! - **Prefetching**: Pre-execute likely future operations
//! - **Rollback mechanisms**: Discard incorrect speculative results
//! - **Confidence scoring**: Track prediction accuracy
//! - **Adaptive strategies**: Learn from prediction success/failure
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_infer::{SpeculativeExecutor, PredictionStrategy, RollbackPolicy};
//!
//! // Create speculative executor
//! let executor = SpeculativeExecutor::new()
//!     .with_strategy(PredictionStrategy::HistoryBased)
//!     .with_rollback_policy(RollbackPolicy::Immediate)
//!     .with_confidence_threshold(0.7);
//!
//! // Execute with speculation
//! let result = executor.execute_speculative(&graph, &inputs)?;
//!
//! // Check speculation stats
//! let stats = executor.get_stats();
//! println!("Speculation success rate: {:.1}%", stats.success_rate * 100.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

/// Speculative execution errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SpeculativeError {
    #[error("Speculation failed: {0}")]
    SpeculationFailed(String),

    #[error("Rollback failed: {0}")]
    RollbackFailed(String),

    #[error("Invalid prediction: {0}")]
    InvalidPrediction(String),

    #[error("Checkpoint not found: {0}")]
    CheckpointNotFound(String),
}

/// Node ID in the computation graph.
pub type NodeId = String;

/// Prediction strategy for speculative execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionStrategy {
    /// Always predict most frequent branch
    MostFrequent,
    /// Use recent history to predict
    HistoryBased,
    /// Use static analysis and heuristics
    Static,
    /// Adaptive strategy that learns over time
    Adaptive,
    /// Always speculate on true branch
    AlwaysTrue,
    /// Never speculate (conservative)
    Never,
}

/// Rollback policy when speculation is incorrect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RollbackPolicy {
    /// Immediately rollback on misprediction
    Immediate,
    /// Continue speculation and rollback later
    Lazy,
    /// Checkpoint-based rollback
    Checkpoint,
}

/// Branch outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchOutcome {
    True,
    False,
    Unknown,
}

/// Speculative task representing work done speculatively.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeTask {
    pub task_id: u64,
    pub node_id: NodeId,
    pub predicted_branch: BranchOutcome,
    pub confidence: f64,
    pub started_at: u64, // timestamp in microseconds
    pub completed: bool,
    pub correct: Option<bool>, // None if not yet validated
}

/// Branch history entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BranchHistory {
    node_id: NodeId,
    outcomes: VecDeque<BranchOutcome>,
    max_history: usize,
}

impl BranchHistory {
    fn new(node_id: NodeId, max_history: usize) -> Self {
        Self {
            node_id,
            outcomes: VecDeque::new(),
            max_history,
        }
    }

    fn add_outcome(&mut self, outcome: BranchOutcome) {
        if self.outcomes.len() >= self.max_history {
            self.outcomes.pop_front();
        }
        self.outcomes.push_back(outcome);
    }

    fn predict(&self) -> (BranchOutcome, f64) {
        if self.outcomes.is_empty() {
            return (BranchOutcome::Unknown, 0.0);
        }

        let true_count = self
            .outcomes
            .iter()
            .filter(|&&o| o == BranchOutcome::True)
            .count();
        let false_count = self
            .outcomes
            .iter()
            .filter(|&&o| o == BranchOutcome::False)
            .count();
        let total = true_count + false_count;

        if total == 0 {
            return (BranchOutcome::Unknown, 0.0);
        }

        if true_count > false_count {
            (BranchOutcome::True, true_count as f64 / total as f64)
        } else {
            (BranchOutcome::False, false_count as f64 / total as f64)
        }
    }
}

/// Speculation statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationStats {
    pub total_speculations: usize,
    pub correct_speculations: usize,
    pub incorrect_speculations: usize,
    pub rollbacks: usize,
    pub success_rate: f64,
    pub average_confidence: f64,
    pub time_saved_us: f64,
    pub time_wasted_us: f64,
}

/// Checkpoint for rollback.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checkpoint {
    checkpoint_id: u64,
    node_id: NodeId,
    timestamp: u64,
    // In real implementation, this would store actual state
}

/// Speculative executor.
pub struct SpeculativeExecutor {
    strategy: PredictionStrategy,
    rollback_policy: RollbackPolicy,
    confidence_threshold: f64,
    max_speculation_depth: usize,
    branch_history: HashMap<NodeId, BranchHistory>,
    active_tasks: HashMap<u64, SpeculativeTask>,
    checkpoints: HashMap<u64, Checkpoint>,
    next_task_id: u64,
    next_checkpoint_id: u64,
    stats: SpeculationStats,
    history_length: usize,
}

impl SpeculativeExecutor {
    /// Create a new speculative executor with default settings.
    pub fn new() -> Self {
        Self {
            strategy: PredictionStrategy::HistoryBased,
            rollback_policy: RollbackPolicy::Immediate,
            confidence_threshold: 0.6,
            max_speculation_depth: 3,
            branch_history: HashMap::new(),
            active_tasks: HashMap::new(),
            checkpoints: HashMap::new(),
            next_task_id: 0,
            next_checkpoint_id: 0,
            stats: SpeculationStats {
                total_speculations: 0,
                correct_speculations: 0,
                incorrect_speculations: 0,
                rollbacks: 0,
                success_rate: 0.0,
                average_confidence: 0.0,
                time_saved_us: 0.0,
                time_wasted_us: 0.0,
            },
            history_length: 10,
        }
    }

    /// Set prediction strategy.
    pub fn with_strategy(mut self, strategy: PredictionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set rollback policy.
    pub fn with_rollback_policy(mut self, policy: RollbackPolicy) -> Self {
        self.rollback_policy = policy;
        self
    }

    /// Set confidence threshold for speculation.
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set maximum speculation depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_speculation_depth = depth;
        self
    }

    /// Predict branch outcome for a node.
    pub fn predict_branch(&self, node_id: &NodeId) -> (BranchOutcome, f64) {
        match self.strategy {
            PredictionStrategy::Never => (BranchOutcome::Unknown, 0.0),
            PredictionStrategy::AlwaysTrue => (BranchOutcome::True, 1.0),
            PredictionStrategy::MostFrequent => {
                if let Some(history) = self.branch_history.get(node_id) {
                    history.predict()
                } else {
                    (BranchOutcome::True, 0.5) // Default to true with low confidence
                }
            }
            PredictionStrategy::HistoryBased => {
                if let Some(history) = self.branch_history.get(node_id) {
                    history.predict()
                } else {
                    (BranchOutcome::Unknown, 0.0)
                }
            }
            PredictionStrategy::Static | PredictionStrategy::Adaptive => {
                // Simplified: use history if available
                if let Some(history) = self.branch_history.get(node_id) {
                    history.predict()
                } else {
                    (BranchOutcome::True, 0.5)
                }
            }
        }
    }

    /// Start speculative execution for a branch.
    pub fn speculate(&mut self, node_id: NodeId) -> Result<u64, SpeculativeError> {
        let (predicted_branch, confidence) = self.predict_branch(&node_id);

        // Only speculate if confidence exceeds threshold
        if confidence < self.confidence_threshold {
            return Err(SpeculativeError::SpeculationFailed(format!(
                "Confidence {} below threshold {}",
                confidence, self.confidence_threshold
            )));
        }

        // Check speculation depth
        let active_count = self.active_tasks.values().filter(|t| !t.completed).count();

        if active_count >= self.max_speculation_depth {
            return Err(SpeculativeError::SpeculationFailed(format!(
                "Maximum speculation depth {} reached",
                self.max_speculation_depth
            )));
        }

        // Create speculative task
        let task_id = self.next_task_id;
        self.next_task_id += 1;

        let task = SpeculativeTask {
            task_id,
            node_id: node_id.clone(),
            predicted_branch,
            confidence,
            started_at: 0, // Would be real timestamp
            completed: false,
            correct: None,
        };

        self.active_tasks.insert(task_id, task);
        self.stats.total_speculations += 1;

        Ok(task_id)
    }

    /// Validate speculative execution result.
    pub fn validate(
        &mut self,
        task_id: u64,
        actual_branch: BranchOutcome,
    ) -> Result<bool, SpeculativeError> {
        let task = self.active_tasks.get_mut(&task_id).ok_or_else(|| {
            SpeculativeError::InvalidPrediction(format!("Task {} not found", task_id))
        })?;

        let correct = task.predicted_branch == actual_branch;
        task.correct = Some(correct);
        task.completed = true;

        // Update history
        let history = self
            .branch_history
            .entry(task.node_id.clone())
            .or_insert_with(|| BranchHistory::new(task.node_id.clone(), self.history_length));
        history.add_outcome(actual_branch);

        // Update stats
        if correct {
            self.stats.correct_speculations += 1;
        } else {
            self.stats.incorrect_speculations += 1;
            // Perform rollback if needed
            self.rollback(task_id)?;
        }

        self.update_stats();

        Ok(correct)
    }

    /// Rollback speculative execution.
    fn rollback(&mut self, task_id: u64) -> Result<(), SpeculativeError> {
        match self.rollback_policy {
            RollbackPolicy::Immediate => {
                // Immediately discard speculative work
                self.active_tasks.remove(&task_id);
                self.stats.rollbacks += 1;
                Ok(())
            }
            RollbackPolicy::Lazy => {
                // Mark for later cleanup
                if let Some(task) = self.active_tasks.get_mut(&task_id) {
                    task.completed = true;
                }
                self.stats.rollbacks += 1;
                Ok(())
            }
            RollbackPolicy::Checkpoint => {
                // Restore from checkpoint
                self.restore_checkpoint(task_id)?;
                self.stats.rollbacks += 1;
                Ok(())
            }
        }
    }

    /// Create checkpoint before speculation.
    pub fn create_checkpoint(&mut self, node_id: NodeId) -> u64 {
        let checkpoint_id = self.next_checkpoint_id;
        self.next_checkpoint_id += 1;

        let checkpoint = Checkpoint {
            checkpoint_id,
            node_id,
            timestamp: 0, // Would be real timestamp
        };

        self.checkpoints.insert(checkpoint_id, checkpoint);
        checkpoint_id
    }

    /// Restore from checkpoint.
    fn restore_checkpoint(&mut self, task_id: u64) -> Result<(), SpeculativeError> {
        // Find and restore checkpoint
        let _task = self.active_tasks.get(&task_id).ok_or_else(|| {
            SpeculativeError::CheckpointNotFound(format!("No task found for id: {}", task_id))
        })?;

        // In real implementation, would restore actual state
        self.active_tasks.remove(&task_id);
        Ok(())
    }

    /// Update speculation statistics.
    fn update_stats(&mut self) {
        let total = (self.stats.correct_speculations + self.stats.incorrect_speculations) as f64;
        if total > 0.0 {
            self.stats.success_rate = self.stats.correct_speculations as f64 / total;
        }

        let confidence_sum: f64 = self.active_tasks.values().map(|t| t.confidence).sum();
        let task_count = self.active_tasks.len() as f64;
        if task_count > 0.0 {
            self.stats.average_confidence = confidence_sum / task_count;
        }
    }

    /// Get speculation statistics.
    pub fn get_stats(&self) -> &SpeculationStats {
        &self.stats
    }

    /// Clear completed speculative tasks.
    pub fn cleanup(&mut self) {
        self.active_tasks.retain(|_, task| !task.completed);
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpeculationStats {
            total_speculations: 0,
            correct_speculations: 0,
            incorrect_speculations: 0,
            rollbacks: 0,
            success_rate: 0.0,
            average_confidence: 0.0,
            time_saved_us: 0.0,
            time_wasted_us: 0.0,
        };
    }

    /// Get active speculation count.
    pub fn active_speculation_count(&self) -> usize {
        self.active_tasks.values().filter(|t| !t.completed).count()
    }

    /// Check if should speculate based on current state.
    pub fn should_speculate(&self, node_id: &NodeId) -> bool {
        let (_, confidence) = self.predict_branch(node_id);
        confidence >= self.confidence_threshold
            && self.active_speculation_count() < self.max_speculation_depth
    }
}

impl Default for SpeculativeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_executor_creation() {
        let executor = SpeculativeExecutor::new();
        assert_eq!(executor.strategy, PredictionStrategy::HistoryBased);
        assert_eq!(executor.rollback_policy, RollbackPolicy::Immediate);
        assert_eq!(executor.confidence_threshold, 0.6);
    }

    #[test]
    fn test_builder_pattern() {
        let executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::Adaptive)
            .with_rollback_policy(RollbackPolicy::Checkpoint)
            .with_confidence_threshold(0.8)
            .with_max_depth(5);

        assert_eq!(executor.strategy, PredictionStrategy::Adaptive);
        assert_eq!(executor.rollback_policy, RollbackPolicy::Checkpoint);
        assert_eq!(executor.confidence_threshold, 0.8);
        assert_eq!(executor.max_speculation_depth, 5);
    }

    #[test]
    fn test_always_true_prediction() {
        let executor = SpeculativeExecutor::new().with_strategy(PredictionStrategy::AlwaysTrue);

        let (outcome, confidence) = executor.predict_branch(&"test".to_string());
        assert_eq!(outcome, BranchOutcome::True);
        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn test_never_speculation() {
        let executor = SpeculativeExecutor::new().with_strategy(PredictionStrategy::Never);

        let (outcome, confidence) = executor.predict_branch(&"test".to_string());
        assert_eq!(outcome, BranchOutcome::Unknown);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_speculation_below_threshold() {
        let mut executor = SpeculativeExecutor::new().with_confidence_threshold(0.9);

        let result = executor.speculate("test".to_string());
        assert!(result.is_err()); // Should fail due to low confidence
    }

    #[test]
    fn test_successful_speculation() {
        let mut executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::AlwaysTrue)
            .with_confidence_threshold(0.5);

        let task_id = executor.speculate("test".to_string()).unwrap();
        assert_eq!(executor.stats.total_speculations, 1);
        assert!(executor.active_tasks.contains_key(&task_id));
    }

    #[test]
    fn test_correct_validation() {
        let mut executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::AlwaysTrue)
            .with_confidence_threshold(0.5);

        let task_id = executor.speculate("test".to_string()).unwrap();
        let correct = executor.validate(task_id, BranchOutcome::True).unwrap();

        assert!(correct);
        assert_eq!(executor.stats.correct_speculations, 1);
        assert_eq!(executor.stats.incorrect_speculations, 0);
    }

    #[test]
    fn test_incorrect_validation() {
        let mut executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::AlwaysTrue)
            .with_confidence_threshold(0.5);

        let task_id = executor.speculate("test".to_string()).unwrap();
        let correct = executor.validate(task_id, BranchOutcome::False).unwrap();

        assert!(!correct);
        assert_eq!(executor.stats.correct_speculations, 0);
        assert_eq!(executor.stats.incorrect_speculations, 1);
        assert_eq!(executor.stats.rollbacks, 1);
    }

    #[test]
    fn test_history_based_prediction() {
        let mut executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::AlwaysTrue) // Start with AlwaysTrue to build history
            .with_confidence_threshold(0.5);

        // Build history with mostly true outcomes
        for _ in 0..8 {
            let task_id = executor.speculate("node1".to_string()).unwrap();
            executor.validate(task_id, BranchOutcome::True).unwrap();
        }

        for _ in 0..2 {
            let task_id = executor.speculate("node1".to_string()).unwrap();
            executor.validate(task_id, BranchOutcome::False).unwrap();
        }

        // Switch to history-based after building history
        executor.strategy = PredictionStrategy::HistoryBased;

        // Should predict True with high confidence
        let (outcome, confidence) = executor.predict_branch(&"node1".to_string());
        assert_eq!(outcome, BranchOutcome::True);
        assert!(confidence > 0.7);
    }

    #[test]
    fn test_max_speculation_depth() {
        let mut executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::AlwaysTrue)
            .with_confidence_threshold(0.5)
            .with_max_depth(2);

        executor.speculate("node1".to_string()).unwrap();
        executor.speculate("node2".to_string()).unwrap();

        // Third speculation should fail
        let result = executor.speculate("node3".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_creation() {
        let mut executor = SpeculativeExecutor::new();
        let checkpoint_id = executor.create_checkpoint("node1".to_string());

        assert!(executor.checkpoints.contains_key(&checkpoint_id));
    }

    #[test]
    fn test_cleanup() {
        let mut executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::AlwaysTrue)
            .with_confidence_threshold(0.5);

        let task_id = executor.speculate("test".to_string()).unwrap();
        executor.validate(task_id, BranchOutcome::True).unwrap();

        assert!(executor.active_tasks.contains_key(&task_id));
        executor.cleanup();
        assert!(!executor.active_tasks.contains_key(&task_id));
    }

    #[test]
    fn test_success_rate_calculation() {
        let mut executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::AlwaysTrue)
            .with_confidence_threshold(0.5);

        // 3 correct, 1 incorrect = 75% success rate
        for _ in 0..3 {
            let task_id = executor.speculate("test".to_string()).unwrap();
            executor.validate(task_id, BranchOutcome::True).unwrap();
        }

        let task_id = executor.speculate("test".to_string()).unwrap();
        executor.validate(task_id, BranchOutcome::False).unwrap();

        assert!((executor.stats.success_rate - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_reset_stats() {
        let mut executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::AlwaysTrue)
            .with_confidence_threshold(0.5);

        let task_id = executor.speculate("test".to_string()).unwrap();
        executor.validate(task_id, BranchOutcome::True).unwrap();

        assert_eq!(executor.stats.total_speculations, 1);

        executor.reset_stats();
        assert_eq!(executor.stats.total_speculations, 0);
        assert_eq!(executor.stats.correct_speculations, 0);
    }

    #[test]
    fn test_should_speculate() {
        let mut executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::AlwaysTrue)
            .with_confidence_threshold(0.5);

        assert!(executor.should_speculate(&"test".to_string()));

        // Fill up speculation depth
        for i in 0..executor.max_speculation_depth {
            executor.speculate(format!("node{}", i)).unwrap();
        }

        assert!(!executor.should_speculate(&"test".to_string()));
    }

    #[test]
    fn test_active_speculation_count() {
        let mut executor = SpeculativeExecutor::new()
            .with_strategy(PredictionStrategy::AlwaysTrue)
            .with_confidence_threshold(0.5);

        assert_eq!(executor.active_speculation_count(), 0);

        executor.speculate("node1".to_string()).unwrap();
        assert_eq!(executor.active_speculation_count(), 1);

        executor.speculate("node2".to_string()).unwrap();
        assert_eq!(executor.active_speculation_count(), 2);
    }

    #[test]
    fn test_different_rollback_policies() {
        let strategies = vec![
            RollbackPolicy::Immediate,
            RollbackPolicy::Lazy,
            RollbackPolicy::Checkpoint,
        ];

        for policy in strategies {
            let mut executor = SpeculativeExecutor::new()
                .with_strategy(PredictionStrategy::AlwaysTrue)
                .with_rollback_policy(policy)
                .with_confidence_threshold(0.5);

            let task_id = executor.speculate("test".to_string()).unwrap();
            executor.validate(task_id, BranchOutcome::False).unwrap();

            assert_eq!(executor.stats.rollbacks, 1);
        }
    }
}
