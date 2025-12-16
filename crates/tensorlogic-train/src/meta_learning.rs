//! Meta-learning algorithms for learning to learn.
//!
//! This module implements meta-learning algorithms that learn model initializations
//! or update rules that enable rapid adaptation to new tasks with minimal data.
//!
//! # Overview
//!
//! Meta-learning (or "learning to learn") aims to improve a model's ability to
//! quickly adapt to new tasks by learning across multiple related tasks. This module
//! provides implementations of state-of-the-art meta-learning algorithms:
//!
//! - **MAML** (Model-Agnostic Meta-Learning): Learns an initialization that can
//!   quickly adapt via a few gradient steps
//! - **Reptile**: Simpler first-order approximation that directly moves toward
//!   task-specific parameters
//! - **Task sampling**: Infrastructure for episodic meta-learning
//!
//! # Key Concepts
//!
//! - **Meta-training**: Outer loop that updates the meta-parameters
//! - **Task adaptation**: Inner loop that adapts to specific tasks
//! - **Support set**: Training data for task adaptation
//! - **Query set**: Validation data for meta-objective
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_train::{MAMLConfig, ReptileConfig, MetaLearner};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure MAML
//! let maml_config = MAMLConfig {
//!     inner_steps: 5,
//!     inner_lr: 0.01,
//!     outer_lr: 0.001,
//!     first_order: false,
//! };
//!
//! // Configure Reptile
//! let reptile_config = ReptileConfig {
//!     inner_steps: 5,
//!     inner_lr: 0.01,
//!     outer_lr: 0.001,
//! };
//! # Ok(())
//! # }
//! ```

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// MAML (Model-Agnostic Meta-Learning) configuration.
///
/// MAML learns an initialization θ* such that a small number of gradient steps
/// on a new task yields good performance.
#[derive(Debug, Clone)]
pub struct MAMLConfig {
    /// Number of gradient steps for task adaptation (inner loop).
    pub inner_steps: usize,
    /// Learning rate for task adaptation (inner loop).
    pub inner_lr: f64,
    /// Learning rate for meta-update (outer loop).
    pub outer_lr: f64,
    /// Use first-order approximation (ignores second derivatives).
    pub first_order: bool,
}

impl Default for MAMLConfig {
    fn default() -> Self {
        Self {
            inner_steps: 5,
            inner_lr: 0.01,
            outer_lr: 0.001,
            first_order: false,
        }
    }
}

/// Reptile algorithm configuration.
///
/// Reptile is a simpler first-order alternative to MAML that repeatedly:
/// 1. Samples a task
/// 2. Trains on it to get task-specific parameters
/// 3. Moves the meta-parameters toward the task-specific parameters
#[derive(Debug, Clone)]
pub struct ReptileConfig {
    /// Number of gradient steps for task adaptation.
    pub inner_steps: usize,
    /// Learning rate for task adaptation.
    pub inner_lr: f64,
    /// Learning rate for meta-update (interpolation weight).
    pub outer_lr: f64,
}

impl Default for ReptileConfig {
    fn default() -> Self {
        Self {
            inner_steps: 10,
            inner_lr: 0.01,
            outer_lr: 0.1,
        }
    }
}

/// Meta-learning task representation.
///
/// Each task consists of a support set (for adaptation) and
/// a query set (for evaluation).
#[derive(Debug, Clone)]
pub struct MetaTask {
    /// Support set features (for training/adaptation).
    pub support_x: Array2<f64>,
    /// Support set labels.
    pub support_y: Array2<f64>,
    /// Query set features (for evaluation).
    pub query_x: Array2<f64>,
    /// Query set labels.
    pub query_y: Array2<f64>,
}

impl MetaTask {
    /// Create a new meta-learning task.
    pub fn new(
        support_x: Array2<f64>,
        support_y: Array2<f64>,
        query_x: Array2<f64>,
        query_y: Array2<f64>,
    ) -> TrainResult<Self> {
        if support_x.nrows() != support_y.nrows() {
            return Err(TrainError::InvalidParameter(format!(
                "Support X rows ({}) must match support Y rows ({})",
                support_x.nrows(),
                support_y.nrows()
            )));
        }

        if query_x.nrows() != query_y.nrows() {
            return Err(TrainError::InvalidParameter(format!(
                "Query X rows ({}) must match query Y rows ({})",
                query_x.nrows(),
                query_y.nrows()
            )));
        }

        Ok(Self {
            support_x,
            support_y,
            query_x,
            query_y,
        })
    }

    /// Get support set size.
    pub fn support_size(&self) -> usize {
        self.support_x.nrows()
    }

    /// Get query set size.
    pub fn query_size(&self) -> usize {
        self.query_x.nrows()
    }
}

/// Meta-learner trait for different meta-learning algorithms.
pub trait MetaLearner {
    /// Perform one step of meta-training on a batch of tasks.
    ///
    /// # Arguments
    /// * `tasks` - Batch of meta-learning tasks
    /// * `parameters` - Current meta-parameters
    ///
    /// # Returns
    /// Updated meta-parameters and meta-loss
    fn meta_step(
        &self,
        tasks: &[MetaTask],
        parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<(HashMap<String, Array1<f64>>, f64)>;

    /// Adapt parameters to a specific task (inner loop).
    ///
    /// # Arguments
    /// * `task` - Task to adapt to
    /// * `parameters` - Initial parameters
    ///
    /// # Returns
    /// Task-adapted parameters
    fn adapt(
        &self,
        task: &MetaTask,
        parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<HashMap<String, Array1<f64>>>;
}

/// MAML (Model-Agnostic Meta-Learning) implementation.
///
/// MAML optimizes for a model initialization that can quickly adapt
/// to new tasks through a few gradient steps.
#[derive(Debug, Clone)]
pub struct MAML {
    config: MAMLConfig,
}

impl MAML {
    /// Create a new MAML meta-learner.
    pub fn new(config: MAMLConfig) -> Self {
        Self { config }
    }
}

impl Default for MAML {
    fn default() -> Self {
        Self::new(MAMLConfig::default())
    }
}

impl MetaLearner for MAML {
    fn meta_step(
        &self,
        tasks: &[MetaTask],
        parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<(HashMap<String, Array1<f64>>, f64)> {
        let mut meta_gradients: HashMap<String, Array1<f64>> = HashMap::new();
        let mut total_loss = 0.0;

        // Initialize meta-gradients to zero
        for (name, param) in parameters {
            meta_gradients.insert(name.clone(), Array1::zeros(param.len()));
        }

        // For each task in the batch
        for task in tasks {
            // 1. Adapt to the task (inner loop)
            let adapted_params = self.adapt(task, parameters)?;

            // 2. Compute loss on query set with adapted parameters
            // This is a placeholder - in practice, you'd use your model here
            let query_loss = self.compute_query_loss(task, &adapted_params)?;
            total_loss += query_loss;

            // 3. Compute gradients of query loss w.r.t. meta-parameters
            // In full MAML, we'd backprop through the adaptation process
            // For first-order MAML, we use adapted params directly
            let task_gradients = if self.config.first_order {
                self.compute_first_order_gradients(task, &adapted_params)?
            } else {
                self.compute_second_order_gradients(task, parameters, &adapted_params)?
            };

            // 4. Accumulate gradients
            for (name, grad) in task_gradients {
                if let Some(meta_grad) = meta_gradients.get_mut(&name) {
                    *meta_grad = meta_grad.clone() + grad;
                }
            }
        }

        // Average gradients and loss
        let n_tasks = tasks.len() as f64;
        for grad in meta_gradients.values_mut() {
            *grad = grad.mapv(|x| x / n_tasks);
        }
        total_loss /= n_tasks;

        // Meta-update (SGD step)
        let mut updated_params = HashMap::new();
        for (name, param) in parameters {
            if let Some(grad) = meta_gradients.get(name) {
                let updated = param - &grad.mapv(|g| g * self.config.outer_lr);
                updated_params.insert(name.clone(), updated);
            }
        }

        Ok((updated_params, total_loss))
    }

    fn adapt(
        &self,
        task: &MetaTask,
        parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<HashMap<String, Array1<f64>>> {
        let mut adapted_params = parameters.clone();

        // Perform inner loop updates
        for _ in 0..self.config.inner_steps {
            // Compute loss and gradients on support set
            let gradients = self.compute_support_gradients(task, &adapted_params)?;

            // SGD update
            for (name, param) in &mut adapted_params {
                if let Some(grad) = gradients.get(name) {
                    *param = param.clone() - &grad.mapv(|g| g * self.config.inner_lr);
                }
            }
        }

        Ok(adapted_params)
    }
}

impl MAML {
    /// Compute gradients on support set (placeholder).
    fn compute_support_gradients(
        &self,
        task: &MetaTask,
        _parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<HashMap<String, Array1<f64>>> {
        // This is a simplified placeholder
        // In practice, you'd compute actual gradients through your model
        let mut gradients = HashMap::new();
        gradients.insert("weights".to_string(), Array1::zeros(task.support_x.ncols()));
        Ok(gradients)
    }

    /// Compute loss on query set (placeholder).
    fn compute_query_loss(
        &self,
        task: &MetaTask,
        _parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<f64> {
        // This is a simplified placeholder
        // In practice, you'd compute actual loss through your model
        Ok(task.query_size() as f64 * 0.1)
    }

    /// Compute first-order gradients (placeholder).
    fn compute_first_order_gradients(
        &self,
        task: &MetaTask,
        _parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<HashMap<String, Array1<f64>>> {
        // This is a simplified placeholder
        let mut gradients = HashMap::new();
        gradients.insert("weights".to_string(), Array1::zeros(task.query_x.ncols()));
        Ok(gradients)
    }

    /// Compute second-order gradients through adaptation (placeholder).
    fn compute_second_order_gradients(
        &self,
        task: &MetaTask,
        _meta_params: &HashMap<String, Array1<f64>>,
        _adapted_params: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<HashMap<String, Array1<f64>>> {
        // This is a simplified placeholder
        // In full MAML, we'd backprop through the inner loop
        let mut gradients = HashMap::new();
        gradients.insert("weights".to_string(), Array1::zeros(task.query_x.ncols()));
        Ok(gradients)
    }
}

/// Reptile meta-learning algorithm.
///
/// Reptile is a simpler first-order algorithm that:
/// 1. Samples a task
/// 2. Trains on it via SGD to get φ
/// 3. Updates θ ← θ + ε(φ - θ)
#[derive(Debug, Clone)]
pub struct Reptile {
    config: ReptileConfig,
}

impl Reptile {
    /// Create a new Reptile meta-learner.
    pub fn new(config: ReptileConfig) -> Self {
        Self { config }
    }
}

impl Default for Reptile {
    fn default() -> Self {
        Self::new(ReptileConfig::default())
    }
}

impl MetaLearner for Reptile {
    fn meta_step(
        &self,
        tasks: &[MetaTask],
        parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<(HashMap<String, Array1<f64>>, f64)> {
        let mut total_loss = 0.0;
        let mut accumulated_delta: HashMap<String, Array1<f64>> = HashMap::new();

        // Initialize accumulated delta to zero
        for (name, param) in parameters {
            accumulated_delta.insert(name.clone(), Array1::zeros(param.len()));
        }

        // For each task in the batch
        for task in tasks {
            // 1. Adapt to the task (train on support set)
            let task_params = self.adapt(task, parameters)?;

            // 2. Compute task loss (for monitoring)
            let task_loss = self.compute_task_loss(task, &task_params)?;
            total_loss += task_loss;

            // 3. Compute direction: φ - θ
            for (name, param) in parameters {
                if let Some(task_param) = task_params.get(name) {
                    let delta = task_param - param;
                    if let Some(acc_delta) = accumulated_delta.get_mut(name) {
                        *acc_delta = acc_delta.clone() + delta;
                    }
                }
            }
        }

        // Average delta and loss
        let n_tasks = tasks.len() as f64;
        for delta in accumulated_delta.values_mut() {
            *delta = delta.mapv(|x| x / n_tasks);
        }
        total_loss /= n_tasks;

        // Meta-update: θ ← θ + ε * average_delta
        let mut updated_params = HashMap::new();
        for (name, param) in parameters {
            if let Some(delta) = accumulated_delta.get(name) {
                let updated = param + &delta.mapv(|d| d * self.config.outer_lr);
                updated_params.insert(name.clone(), updated);
            }
        }

        Ok((updated_params, total_loss))
    }

    fn adapt(
        &self,
        task: &MetaTask,
        parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<HashMap<String, Array1<f64>>> {
        let mut task_params = parameters.clone();

        // Perform SGD steps on support set
        for _ in 0..self.config.inner_steps {
            // Compute loss and gradients on support set
            let gradients = self.compute_support_gradients(task, &task_params)?;

            // SGD update
            for (name, param) in &mut task_params {
                if let Some(grad) = gradients.get(name) {
                    *param = param.clone() - &grad.mapv(|g| g * self.config.inner_lr);
                }
            }
        }

        Ok(task_params)
    }
}

impl Reptile {
    /// Compute gradients on support set (placeholder).
    fn compute_support_gradients(
        &self,
        task: &MetaTask,
        _parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<HashMap<String, Array1<f64>>> {
        // This is a simplified placeholder
        let mut gradients = HashMap::new();
        gradients.insert("weights".to_string(), Array1::zeros(task.support_x.ncols()));
        Ok(gradients)
    }

    /// Compute task loss (placeholder).
    fn compute_task_loss(
        &self,
        task: &MetaTask,
        _parameters: &HashMap<String, Array1<f64>>,
    ) -> TrainResult<f64> {
        // This is a simplified placeholder
        Ok(task.query_size() as f64 * 0.1)
    }
}

/// Meta-learning statistics tracker.
#[derive(Debug, Clone, Default)]
pub struct MetaStats {
    /// Meta-training losses over time.
    pub meta_losses: Vec<f64>,
    /// Task adaptation losses.
    pub task_losses: Vec<Vec<f64>>,
    /// Number of meta-iterations completed.
    pub iterations: usize,
}

impl MetaStats {
    /// Create a new statistics tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a meta-training step.
    pub fn record_meta_step(&mut self, meta_loss: f64) {
        self.meta_losses.push(meta_loss);
        self.iterations += 1;
    }

    /// Record task adaptation.
    pub fn record_task_adaptation(&mut self, task_id: usize, losses: Vec<f64>) {
        while self.task_losses.len() <= task_id {
            self.task_losses.push(Vec::new());
        }
        self.task_losses[task_id] = losses;
    }

    /// Get average meta-loss over last N steps.
    pub fn avg_meta_loss(&self, last_n: usize) -> f64 {
        if self.meta_losses.is_empty() {
            return 0.0;
        }

        let n = last_n.min(self.meta_losses.len());
        let start = self.meta_losses.len() - n;
        self.meta_losses[start..].iter().sum::<f64>() / n as f64
    }

    /// Check if meta-training is improving (loss decreasing).
    pub fn is_improving(&self, window: usize) -> bool {
        if self.meta_losses.len() < window * 2 {
            return false;
        }

        let recent = self.avg_meta_loss(window);
        let previous = {
            let start = self.meta_losses.len() - window * 2;
            let end = self.meta_losses.len() - window;
            self.meta_losses[start..end].iter().sum::<f64>() / window as f64
        };

        recent < previous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maml_config_default() {
        let config = MAMLConfig::default();
        assert_eq!(config.inner_steps, 5);
        assert_eq!(config.inner_lr, 0.01);
        assert_eq!(config.outer_lr, 0.001);
        assert!(!config.first_order);
    }

    #[test]
    fn test_reptile_config_default() {
        let config = ReptileConfig::default();
        assert_eq!(config.inner_steps, 10);
        assert_eq!(config.inner_lr, 0.01);
        assert_eq!(config.outer_lr, 0.1);
    }

    #[test]
    fn test_meta_task_creation() {
        let support_x = Array2::zeros((5, 10));
        let support_y = Array2::zeros((5, 2));
        let query_x = Array2::zeros((15, 10));
        let query_y = Array2::zeros((15, 2));

        let task = MetaTask::new(support_x, support_y, query_x, query_y).unwrap();
        assert_eq!(task.support_size(), 5);
        assert_eq!(task.query_size(), 15);
    }

    #[test]
    fn test_meta_task_validation() {
        let support_x = Array2::zeros((5, 10));
        let support_y = Array2::zeros((4, 2)); // Mismatch!
        let query_x = Array2::zeros((15, 10));
        let query_y = Array2::zeros((15, 2));

        let result = MetaTask::new(support_x, support_y, query_x, query_y);
        assert!(result.is_err());
    }

    #[test]
    fn test_maml_creation() {
        let config = MAMLConfig::default();
        let maml = MAML::new(config);
        assert_eq!(maml.config.inner_steps, 5);
    }

    #[test]
    fn test_maml_default() {
        let maml = MAML::default();
        assert_eq!(maml.config.inner_steps, 5);
    }

    #[test]
    fn test_reptile_creation() {
        let config = ReptileConfig::default();
        let reptile = Reptile::new(config);
        assert_eq!(reptile.config.inner_steps, 10);
    }

    #[test]
    fn test_reptile_default() {
        let reptile = Reptile::default();
        assert_eq!(reptile.config.inner_steps, 10);
    }

    #[test]
    fn test_maml_adapt() {
        let maml = MAML::default();

        let task = create_dummy_task();
        let mut params = HashMap::new();
        params.insert("weights".to_string(), Array1::zeros(10));

        let adapted = maml.adapt(&task, &params).unwrap();
        assert!(adapted.contains_key("weights"));
    }

    #[test]
    fn test_reptile_adapt() {
        let reptile = Reptile::default();

        let task = create_dummy_task();
        let mut params = HashMap::new();
        params.insert("weights".to_string(), Array1::zeros(10));

        let adapted = reptile.adapt(&task, &params).unwrap();
        assert!(adapted.contains_key("weights"));
    }

    #[test]
    fn test_maml_meta_step() {
        let maml = MAML::default();

        let tasks = vec![create_dummy_task(), create_dummy_task()];
        let mut params = HashMap::new();
        params.insert("weights".to_string(), Array1::zeros(10));

        let (updated_params, loss) = maml.meta_step(&tasks, &params).unwrap();
        assert!(updated_params.contains_key("weights"));
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_reptile_meta_step() {
        let reptile = Reptile::default();

        let tasks = vec![create_dummy_task(), create_dummy_task()];
        let mut params = HashMap::new();
        params.insert("weights".to_string(), Array1::zeros(10));

        let (updated_params, loss) = reptile.meta_step(&tasks, &params).unwrap();
        assert!(updated_params.contains_key("weights"));
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_meta_stats() {
        let mut stats = MetaStats::new();

        stats.record_meta_step(1.0);
        stats.record_meta_step(0.8);
        stats.record_meta_step(0.6);

        assert_eq!(stats.iterations, 3);
        assert_eq!(stats.meta_losses.len(), 3);
        assert_eq!(stats.avg_meta_loss(2), 0.7);
    }

    #[test]
    fn test_meta_stats_improvement() {
        let mut stats = MetaStats::new();

        // Add decreasing losses
        for i in 0..20 {
            stats.record_meta_step(1.0 - i as f64 * 0.01);
        }

        assert!(stats.is_improving(5));
    }

    #[test]
    fn test_meta_stats_task_adaptation() {
        let mut stats = MetaStats::new();

        stats.record_task_adaptation(0, vec![1.0, 0.8, 0.6]);
        stats.record_task_adaptation(1, vec![1.2, 0.9, 0.7]);

        assert_eq!(stats.task_losses.len(), 2);
        assert_eq!(stats.task_losses[0].len(), 3);
    }

    // Helper function
    fn create_dummy_task() -> MetaTask {
        let support_x = Array2::zeros((5, 10));
        let support_y = Array2::zeros((5, 2));
        let query_x = Array2::zeros((15, 10));
        let query_y = Array2::zeros((15, 2));
        MetaTask::new(support_x, support_y, query_x, query_y).unwrap()
    }
}
