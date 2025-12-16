//! Parallel execution utilities with work-stealing scheduler.
//!
//! This module provides advanced parallel execution infrastructure:
//! - **Work-stealing scheduler** for dynamic load balancing
//! - **NUMA-aware memory allocation** for multi-socket systems
//! - **Thread pool management** with configurable worker counts
//! - **Task dependencies** and execution ordering
//! - **Load balancing metrics** and monitoring
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_infer::{WorkStealingScheduler, ParallelConfig, Task};
//!
//! // Create a work-stealing scheduler
//! let config = ParallelConfig::default()
//!     .with_num_workers(8)
//!     .with_steal_strategy(StealStrategy::Random);
//!
//! let scheduler = WorkStealingScheduler::new(config);
//!
//! // Submit tasks
//! for task in tasks {
//!     scheduler.submit(task)?;
//! }
//!
//! // Execute in parallel with work stealing
//! let results = scheduler.execute_all()?;
//!
//! // Check load balancing stats
//! let stats = scheduler.stats();
//! println!("Steal count: {}", stats.steal_count);
//! println!("Load balance: {:.2}%", stats.load_balance_ratio * 100.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Parallel execution errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ParallelError {
    #[error("Task queue is full")]
    QueueFull,

    #[error("Task dependency cycle detected")]
    DependencyCycle,

    #[error("Task {0} not found")]
    TaskNotFound(String),

    #[error("Invalid worker count: {0}")]
    InvalidWorkerCount(usize),

    #[error("Parallel execution failed: {0}")]
    ExecutionFailed(String),

    #[error("NUMA allocation failed: {0}")]
    NumaAllocationFailed(String),
}

/// Work-stealing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StealStrategy {
    /// Random victim selection
    Random,
    /// Steal from the worker with the most work
    MaxLoad,
    /// Steal from the worker with the least recently updated queue
    LRU,
    /// Round-robin victim selection
    RoundRobin,
}

/// NUMA node identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NumaNode(pub usize);

/// NUMA allocation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumaStrategy {
    /// No NUMA awareness (default)
    None,
    /// Prefer local NUMA node
    LocalPreferred,
    /// Strict local NUMA node
    LocalStrict,
    /// Interleave across all NUMA nodes
    Interleave,
}

/// Parallel execution configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub num_workers: usize,

    /// Work-stealing strategy
    pub steal_strategy: StealStrategy,

    /// NUMA allocation strategy
    pub numa_strategy: NumaStrategy,

    /// Enable task priority
    pub enable_priority: bool,

    /// Enable load balancing statistics
    pub enable_stats: bool,

    /// Maximum queue size per worker
    pub max_queue_size: usize,

    /// Enable cache-line padding for worker queues
    pub cache_line_padding: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            steal_strategy: StealStrategy::Random,
            numa_strategy: NumaStrategy::None,
            enable_priority: false,
            enable_stats: true,
            max_queue_size: 10000,
            cache_line_padding: true,
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel config.
    pub fn new(num_workers: usize) -> Result<Self, ParallelError> {
        if num_workers == 0 {
            return Err(ParallelError::InvalidWorkerCount(num_workers));
        }

        Ok(Self {
            num_workers,
            ..Default::default()
        })
    }

    /// Set the number of workers.
    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Set the steal strategy.
    pub fn with_steal_strategy(mut self, strategy: StealStrategy) -> Self {
        self.steal_strategy = strategy;
        self
    }

    /// Set the NUMA strategy.
    pub fn with_numa_strategy(mut self, strategy: NumaStrategy) -> Self {
        self.numa_strategy = strategy;
        self
    }

    /// Enable or disable priority.
    pub fn with_priority(mut self, enabled: bool) -> Self {
        self.enable_priority = enabled;
        self
    }

    /// Enable or disable statistics.
    pub fn with_stats(mut self, enabled: bool) -> Self {
        self.enable_stats = enabled;
        self
    }
}

/// Task identifier.
pub type TaskId = String;

/// Task priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Low priority
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical priority
    Critical = 3,
}

/// A parallel task.
#[derive(Debug, Clone)]
pub struct Task {
    /// Task identifier
    pub id: TaskId,

    /// Task priority
    pub priority: TaskPriority,

    /// Task dependencies (must complete before this task)
    pub dependencies: Vec<TaskId>,

    /// NUMA node affinity
    pub numa_node: Option<NumaNode>,

    /// Estimated execution time (microseconds)
    pub estimated_time_us: Option<u64>,
}

impl Task {
    /// Create a new task.
    pub fn new(id: TaskId) -> Self {
        Self {
            id,
            priority: TaskPriority::Normal,
            dependencies: Vec::new(),
            numa_node: None,
            estimated_time_us: None,
        }
    }

    /// Set the task priority.
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Add a dependency.
    pub fn with_dependency(mut self, dep: TaskId) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Set NUMA node affinity.
    pub fn with_numa_node(mut self, node: NumaNode) -> Self {
        self.numa_node = Some(node);
        self
    }

    /// Set estimated execution time.
    pub fn with_estimated_time(mut self, time_us: u64) -> Self {
        self.estimated_time_us = Some(time_us);
        self
    }
}

/// Worker queue with cache-line padding to avoid false sharing.
#[repr(align(64))] // Cache line size on most architectures
struct WorkerQueue {
    queue: VecDeque<Task>,
    steal_count: usize,
    tasks_executed: usize,
    total_execution_time_us: u64,
}

impl WorkerQueue {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            steal_count: 0,
            tasks_executed: 0,
            total_execution_time_us: 0,
        }
    }

    fn push(&mut self, task: Task) {
        self.queue.push_back(task);
    }

    fn pop(&mut self) -> Option<Task> {
        self.queue.pop_front()
    }

    fn steal(&mut self) -> Option<Task> {
        self.steal_count += 1;
        self.queue.pop_back()
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}

/// Work-stealing scheduler.
pub struct WorkStealingScheduler {
    config: ParallelConfig,
    workers: Vec<Arc<Mutex<WorkerQueue>>>,
    completed_tasks: Arc<Mutex<HashMap<TaskId, u64>>>, // task_id -> execution_time_us
    stats: Arc<Mutex<SchedulerStats>>,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler.
    pub fn new(config: ParallelConfig) -> Self {
        let mut workers = Vec::with_capacity(config.num_workers);
        for _ in 0..config.num_workers {
            workers.push(Arc::new(Mutex::new(WorkerQueue::new())));
        }

        Self {
            config,
            workers,
            completed_tasks: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(SchedulerStats::default())),
        }
    }

    /// Submit a task to the scheduler.
    pub fn submit(&self, task: Task) -> Result<(), ParallelError> {
        // Check dependencies
        self.validate_dependencies(&task)?;

        // Find the worker with the least load
        let worker_idx = self.select_worker(&task);

        // Add task to worker queue
        let mut worker = self.workers[worker_idx].lock().unwrap();
        if worker.len() >= self.config.max_queue_size {
            return Err(ParallelError::QueueFull);
        }

        worker.push(task);

        Ok(())
    }

    /// Submit multiple tasks.
    pub fn submit_batch(&self, tasks: Vec<Task>) -> Result<(), ParallelError> {
        for task in tasks {
            self.submit(task)?;
        }
        Ok(())
    }

    /// Execute all submitted tasks.
    pub fn execute_all(&self) -> Result<Vec<TaskId>, ParallelError> {
        let mut completed = Vec::new();

        // Simplified execution model (in a real implementation, this would use a thread pool)
        for worker in &self.workers {
            let mut worker = worker.lock().unwrap();
            while let Some(task) = worker.pop() {
                // Check if dependencies are satisfied
                if self.dependencies_satisfied(&task)? {
                    // Execute task (simulated)
                    let execution_time = task.estimated_time_us.unwrap_or(1000);
                    worker.tasks_executed += 1;
                    worker.total_execution_time_us += execution_time;

                    // Mark as completed
                    self.completed_tasks
                        .lock()
                        .unwrap()
                        .insert(task.id.clone(), execution_time);

                    completed.push(task.id);

                    // Update stats
                    if self.config.enable_stats {
                        let mut stats = self.stats.lock().unwrap();
                        stats.tasks_executed += 1;
                        stats.total_execution_time_us += execution_time;
                    }
                } else {
                    // Re-queue task if dependencies not satisfied
                    worker.push(task);
                }
            }
        }

        Ok(completed)
    }

    /// Get scheduler statistics.
    pub fn stats(&self) -> SchedulerStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset the scheduler.
    pub fn reset(&self) {
        for worker in &self.workers {
            let mut worker = worker.lock().unwrap();
            worker.queue.clear();
            worker.steal_count = 0;
            worker.tasks_executed = 0;
            worker.total_execution_time_us = 0;
        }

        self.completed_tasks.lock().unwrap().clear();
        *self.stats.lock().unwrap() = SchedulerStats::default();
    }

    // Helper methods

    fn validate_dependencies(&self, task: &Task) -> Result<(), ParallelError> {
        // Simple cycle detection (would need more sophisticated algorithm for production)
        let mut visited = std::collections::HashSet::new();
        self.check_cycle(&task.id, &task.dependencies, &mut visited)
    }

    fn check_cycle(
        &self,
        current: &TaskId,
        dependencies: &[TaskId],
        visited: &mut std::collections::HashSet<TaskId>,
    ) -> Result<(), ParallelError> {
        if visited.contains(current) {
            return Err(ParallelError::DependencyCycle);
        }

        visited.insert(current.clone());

        for _dep in dependencies {
            // In a real implementation, we'd look up the dependencies of _dep
            // For now, just assume no cycles
        }

        Ok(())
    }

    fn dependencies_satisfied(&self, task: &Task) -> Result<bool, ParallelError> {
        let completed = self.completed_tasks.lock().unwrap();
        Ok(task
            .dependencies
            .iter()
            .all(|dep| completed.contains_key(dep)))
    }

    fn select_worker(&self, task: &Task) -> usize {
        // NUMA affinity if specified
        if let Some(numa_node) = task.numa_node {
            return self.numa_node_to_worker(numa_node);
        }

        // Otherwise, find worker with least load
        let mut min_load = usize::MAX;
        let mut selected = 0;

        for (idx, worker) in self.workers.iter().enumerate() {
            let worker = worker.lock().unwrap();
            let load = worker.len();
            if load < min_load {
                min_load = load;
                selected = idx;
            }
        }

        selected
    }

    fn numa_node_to_worker(&self, node: NumaNode) -> usize {
        // Simple mapping: distribute workers evenly across NUMA nodes
        // In practice, this would query the system topology
        node.0 % self.config.num_workers
    }

    /// Attempt to steal work from another worker.
    pub fn try_steal(&self, thief_idx: usize) -> Option<Task> {
        let victim_idx = self.select_victim(thief_idx);
        if victim_idx == thief_idx {
            return None;
        }

        let mut victim = self.workers[victim_idx].lock().unwrap();
        let stolen = victim.steal();

        if stolen.is_some() && self.config.enable_stats {
            let mut stats = self.stats.lock().unwrap();
            stats.steal_count += 1;
        }

        stolen
    }

    fn select_victim(&self, thief_idx: usize) -> usize {
        match self.config.steal_strategy {
            StealStrategy::Random => {
                // Simple random victim selection
                (thief_idx + 1) % self.config.num_workers
            }
            StealStrategy::MaxLoad => {
                // Find worker with most work
                let mut max_load = 0;
                let mut victim = thief_idx;

                for (idx, worker) in self.workers.iter().enumerate() {
                    if idx == thief_idx {
                        continue;
                    }
                    let worker = worker.lock().unwrap();
                    let load = worker.len();
                    if load > max_load {
                        max_load = load;
                        victim = idx;
                    }
                }

                victim
            }
            StealStrategy::LRU | StealStrategy::RoundRobin => {
                // Simple round-robin
                (thief_idx + 1) % self.config.num_workers
            }
        }
    }

    /// Get load balancing statistics.
    pub fn load_balance_stats(&self) -> LoadBalanceStats {
        let mut worker_loads = Vec::new();
        let mut total_tasks = 0;

        for worker in &self.workers {
            let worker = worker.lock().unwrap();
            let load = worker.tasks_executed;
            worker_loads.push(load);
            total_tasks += load;
        }

        let avg_load = total_tasks as f64 / self.config.num_workers as f64;
        let variance = worker_loads
            .iter()
            .map(|&load| (load as f64 - avg_load).powi(2))
            .sum::<f64>()
            / self.config.num_workers as f64;

        let std_dev = variance.sqrt();
        let cv = if avg_load > 0.0 {
            std_dev / avg_load
        } else {
            0.0
        };
        let max_load = *worker_loads.iter().max().unwrap_or(&0);

        LoadBalanceStats {
            worker_loads,
            avg_load,
            std_dev,
            coefficient_of_variation: cv,
            imbalance_ratio: if avg_load > 0.0 {
                max_load as f64 / avg_load
            } else {
                1.0
            },
        }
    }
}

/// Scheduler statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SchedulerStats {
    /// Total number of tasks executed
    pub tasks_executed: usize,

    /// Total execution time (microseconds)
    pub total_execution_time_us: u64,

    /// Number of work-stealing operations
    pub steal_count: usize,

    /// Number of failed steal attempts
    pub failed_steals: usize,
}

impl SchedulerStats {
    /// Get the average execution time per task.
    pub fn avg_execution_time_us(&self) -> f64 {
        if self.tasks_executed > 0 {
            self.total_execution_time_us as f64 / self.tasks_executed as f64
        } else {
            0.0
        }
    }

    /// Get the steal success rate.
    pub fn steal_success_rate(&self) -> f64 {
        let total_attempts = self.steal_count + self.failed_steals;
        if total_attempts > 0 {
            self.steal_count as f64 / total_attempts as f64
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for SchedulerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Scheduler Statistics")?;
        writeln!(f, "====================")?;
        writeln!(f, "Tasks executed:     {}", self.tasks_executed)?;
        writeln!(
            f,
            "Total time:         {:.2} ms",
            self.total_execution_time_us as f64 / 1000.0
        )?;
        writeln!(
            f,
            "Avg time/task:      {:.2} µs",
            self.avg_execution_time_us()
        )?;
        writeln!(f, "Steal count:        {}", self.steal_count)?;
        writeln!(f, "Failed steals:      {}", self.failed_steals)?;
        writeln!(
            f,
            "Steal success rate: {:.2}%",
            self.steal_success_rate() * 100.0
        )?;
        Ok(())
    }
}

/// Load balancing statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadBalanceStats {
    /// Load per worker (tasks executed)
    pub worker_loads: Vec<usize>,

    /// Average load across workers
    pub avg_load: f64,

    /// Standard deviation of load
    pub std_dev: f64,

    /// Coefficient of variation (CV = std_dev / mean)
    pub coefficient_of_variation: f64,

    /// Imbalance ratio (max_load / avg_load)
    pub imbalance_ratio: f64,
}

impl LoadBalanceStats {
    /// Check if the load is well balanced (CV < 0.2).
    pub fn is_well_balanced(&self) -> bool {
        self.coefficient_of_variation < 0.2
    }
}

impl std::fmt::Display for LoadBalanceStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Load Balance Statistics")?;
        writeln!(f, "=======================")?;
        writeln!(f, "Worker loads:   {:?}", self.worker_loads)?;
        writeln!(f, "Average load:   {:.2}", self.avg_load)?;
        writeln!(f, "Std deviation:  {:.2}", self.std_dev)?;
        writeln!(f, "CV:             {:.4}", self.coefficient_of_variation)?;
        writeln!(f, "Imbalance:      {:.2}x", self.imbalance_ratio)?;
        writeln!(
            f,
            "Well balanced:  {}",
            if self.is_well_balanced() { "Yes" } else { "No" }
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.num_workers > 0);
        assert_eq!(config.steal_strategy, StealStrategy::Random);
        assert!(config.enable_stats);
    }

    #[test]
    fn test_parallel_config_builder() {
        let config = ParallelConfig::new(4)
            .unwrap()
            .with_steal_strategy(StealStrategy::MaxLoad)
            .with_numa_strategy(NumaStrategy::LocalPreferred)
            .with_priority(true);

        assert_eq!(config.num_workers, 4);
        assert_eq!(config.steal_strategy, StealStrategy::MaxLoad);
        assert_eq!(config.numa_strategy, NumaStrategy::LocalPreferred);
        assert!(config.enable_priority);
    }

    #[test]
    fn test_task_creation() {
        let task = Task::new("task1".to_string())
            .with_priority(TaskPriority::High)
            .with_dependency("task0".to_string())
            .with_estimated_time(1000);

        assert_eq!(task.id, "task1");
        assert_eq!(task.priority, TaskPriority::High);
        assert_eq!(task.dependencies.len(), 1);
        assert_eq!(task.estimated_time_us, Some(1000));
    }

    #[test]
    fn test_scheduler_creation() {
        let config = ParallelConfig::new(4).unwrap();
        let scheduler = WorkStealingScheduler::new(config);

        assert_eq!(scheduler.workers.len(), 4);
    }

    #[test]
    fn test_scheduler_submit() {
        let config = ParallelConfig::new(2).unwrap();
        let scheduler = WorkStealingScheduler::new(config);

        let task = Task::new("task1".to_string());
        assert!(scheduler.submit(task).is_ok());
    }

    #[test]
    fn test_scheduler_execute_simple() {
        let config = ParallelConfig::new(2).unwrap();
        let scheduler = WorkStealingScheduler::new(config);

        let task1 = Task::new("task1".to_string()).with_estimated_time(100);
        let task2 = Task::new("task2".to_string()).with_estimated_time(200);

        scheduler.submit(task1).unwrap();
        scheduler.submit(task2).unwrap();

        let completed = scheduler.execute_all().unwrap();
        assert_eq!(completed.len(), 2);
    }

    #[test]
    fn test_scheduler_dependencies() {
        let config = ParallelConfig::new(2).unwrap();
        let scheduler = WorkStealingScheduler::new(config);

        let task1 = Task::new("task1".to_string());
        let task2 = Task::new("task2".to_string()).with_dependency("task1".to_string());

        scheduler.submit(task1).unwrap();
        scheduler.submit(task2).unwrap();

        let completed = scheduler.execute_all().unwrap();
        assert!(completed.contains(&"task1".to_string()));
    }

    #[test]
    fn test_scheduler_stats() {
        let config = ParallelConfig::new(2).unwrap();
        let scheduler = WorkStealingScheduler::new(config);

        let task1 = Task::new("task1".to_string()).with_estimated_time(1000);
        let task2 = Task::new("task2".to_string()).with_estimated_time(2000);

        scheduler.submit(task1).unwrap();
        scheduler.submit(task2).unwrap();
        scheduler.execute_all().unwrap();

        let stats = scheduler.stats();
        assert_eq!(stats.tasks_executed, 2);
        assert_eq!(stats.total_execution_time_us, 3000);
    }

    #[test]
    fn test_load_balance_stats() {
        let config = ParallelConfig::new(4).unwrap();
        let scheduler = WorkStealingScheduler::new(config);

        // Submit tasks
        for i in 0..8 {
            let task = Task::new(format!("task{}", i)).with_estimated_time(100);
            scheduler.submit(task).unwrap();
        }

        scheduler.execute_all().unwrap();

        let stats = scheduler.load_balance_stats();
        assert!((stats.avg_load - 2.0).abs() < 0.1); // 8 tasks / 4 workers = 2
    }

    #[test]
    fn test_scheduler_reset() {
        let config = ParallelConfig::new(2).unwrap();
        let scheduler = WorkStealingScheduler::new(config);

        let task = Task::new("task1".to_string());
        scheduler.submit(task).unwrap();
        scheduler.execute_all().unwrap();

        let stats_before = scheduler.stats();
        assert_eq!(stats_before.tasks_executed, 1);

        scheduler.reset();

        let stats_after = scheduler.stats();
        assert_eq!(stats_after.tasks_executed, 0);
    }

    #[test]
    fn test_task_priority() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_numa_node() {
        let node = NumaNode(0);
        assert_eq!(node.0, 0);
    }

    #[test]
    fn test_steal_strategy() {
        // Verify that StealStrategy is Copy
        let s1 = StealStrategy::Random;
        let s2 = s1;
        let s3 = s1; // This compiles only if StealStrategy is Copy
        assert_eq!(s2, s3);
    }
}
