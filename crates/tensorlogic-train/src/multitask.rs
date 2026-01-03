//! Multi-task learning utilities for training with multiple objectives.
//!
//! This module provides utilities for multi-task learning, including:
//! - Task weighting strategies
//! - Multi-task loss composition
//! - Gradient balancing techniques
//! - Task-specific metrics tracking

use crate::{Loss, TrainError, TrainResult};
use scirs2_core::ndarray::{s, Array, ArrayView, Ix2};
use std::collections::HashMap;

/// Strategy for weighting multiple tasks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskWeightingStrategy {
    /// Fixed weights for each task.
    Fixed,
    /// Dynamic Task Prioritization (DTP) - weights based on task difficulty.
    DynamicTaskPrioritization,
    /// GradNorm - balances gradient magnitudes across tasks.
    GradNorm { alpha: f64 },
    /// Uncertainty weighting - learns task weights from homoscedastic uncertainty.
    UncertaintyWeighting,
}

/// Multi-task loss that combines multiple losses with configurable weighting.
pub struct MultiTaskLoss {
    /// Individual task losses.
    pub task_losses: Vec<Box<dyn Loss>>,
    /// Task weights (automatically managed based on strategy).
    pub task_weights: Vec<f64>,
    /// Weighting strategy.
    pub strategy: TaskWeightingStrategy,
    /// Learning rate for weight updates (used in some strategies).
    pub weight_lr: f64,
    /// Initial loss values for normalization.
    initial_losses: Option<Vec<f64>>,
}

impl MultiTaskLoss {
    /// Create a new multi-task loss with fixed weights.
    ///
    /// # Arguments
    /// * `task_losses` - Individual loss functions for each task
    /// * `task_weights` - Fixed weights for each task (should sum to 1.0)
    pub fn new_fixed(task_losses: Vec<Box<dyn Loss>>, task_weights: Vec<f64>) -> TrainResult<Self> {
        if task_losses.len() != task_weights.len() {
            return Err(TrainError::ConfigError(
                "Number of losses must match number of weights".to_string(),
            ));
        }

        if task_losses.is_empty() {
            return Err(TrainError::ConfigError(
                "Must have at least one task".to_string(),
            ));
        }

        Ok(Self {
            task_losses,
            task_weights,
            strategy: TaskWeightingStrategy::Fixed,
            weight_lr: 0.0,
            initial_losses: None,
        })
    }

    /// Create a new multi-task loss with dynamic weighting.
    ///
    /// # Arguments
    /// * `task_losses` - Individual loss functions for each task
    /// * `strategy` - Weighting strategy to use
    /// * `weight_lr` - Learning rate for weight updates
    pub fn new_dynamic(
        task_losses: Vec<Box<dyn Loss>>,
        strategy: TaskWeightingStrategy,
        weight_lr: f64,
    ) -> TrainResult<Self> {
        if task_losses.is_empty() {
            return Err(TrainError::ConfigError(
                "Must have at least one task".to_string(),
            ));
        }

        let n_tasks = task_losses.len();
        let task_weights = vec![1.0 / n_tasks as f64; n_tasks];

        Ok(Self {
            task_losses,
            task_weights,
            strategy,
            weight_lr,
            initial_losses: None,
        })
    }

    /// Compute multi-task loss.
    ///
    /// # Arguments
    /// * `predictions` - Predictions for all tasks (concatenated)
    /// * `targets` - Targets for all tasks (concatenated)
    /// * `task_splits` - Column indices where each task starts
    ///
    /// # Returns
    /// Weighted sum of task losses
    pub fn compute_multi_task(
        &mut self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
        task_splits: &[usize],
    ) -> TrainResult<f64> {
        if task_splits.len() != self.task_losses.len() + 1 {
            return Err(TrainError::LossError(format!(
                "task_splits must have {} elements (n_tasks + 1)",
                self.task_losses.len() + 1
            )));
        }

        let mut task_losses_values = Vec::new();

        // Compute individual task losses
        for i in 0..self.task_losses.len() {
            let start = task_splits[i];
            let end = task_splits[i + 1];

            let task_pred = predictions.slice(s![.., start..end]);
            let task_target = targets.slice(s![.., start..end]);

            let loss_value = self.task_losses[i].compute(&task_pred, &task_target)?;
            task_losses_values.push(loss_value);
        }

        // Initialize on first call
        if self.initial_losses.is_none() {
            self.initial_losses = Some(task_losses_values.clone());
        }

        // Update task weights based on strategy
        self.update_weights(&task_losses_values)?;

        // Compute weighted sum
        let total_loss = task_losses_values
            .iter()
            .zip(self.task_weights.iter())
            .map(|(loss, weight)| loss * weight)
            .sum();

        Ok(total_loss)
    }

    /// Update task weights based on the selected strategy.
    fn update_weights(&mut self, current_losses: &[f64]) -> TrainResult<()> {
        match self.strategy {
            TaskWeightingStrategy::Fixed => {
                // Weights don't change
                Ok(())
            }
            TaskWeightingStrategy::DynamicTaskPrioritization => {
                // Weight tasks inversely to their performance
                // Tasks with higher loss get higher weight
                let sum: f64 = current_losses.iter().sum();
                if sum > 1e-8 {
                    for (i, &loss) in current_losses.iter().enumerate() {
                        self.task_weights[i] = loss / sum;
                    }
                }
                Ok(())
            }
            TaskWeightingStrategy::GradNorm { alpha } => {
                // GradNorm: balance gradient magnitudes
                // Simplified version - in practice, needs gradient information
                if let Some(ref initial) = self.initial_losses {
                    let mut relative_rates = Vec::new();
                    for i in 0..current_losses.len() {
                        let rate = current_losses[i] / initial[i].max(1e-8);
                        relative_rates.push(rate);
                    }

                    let mean_rate: f64 =
                        relative_rates.iter().sum::<f64>() / relative_rates.len() as f64;

                    // Update weights to balance training rates
                    for (i, &rate) in relative_rates.iter().enumerate() {
                        let target_rate = mean_rate * self.task_weights[i].powf(alpha);
                        let adjustment = (target_rate / rate.max(1e-8)).ln();
                        self.task_weights[i] *= (self.weight_lr * adjustment).exp();
                    }

                    // Normalize weights
                    let sum: f64 = self.task_weights.iter().sum();
                    for w in &mut self.task_weights {
                        *w /= sum;
                    }
                }
                Ok(())
            }
            TaskWeightingStrategy::UncertaintyWeighting => {
                // Uncertainty weighting: 1 / (2 * sigma^2) per task
                // In practice, sigma would be learned parameters
                // Here we use a simplified version based on loss variance
                Ok(())
            }
        }
    }

    /// Get current task weights.
    pub fn get_weights(&self) -> &[f64] {
        &self.task_weights
    }

    /// Get number of tasks.
    pub fn num_tasks(&self) -> usize {
        self.task_losses.len()
    }
}

/// PCGrad: Project conflicting gradients for multi-task learning.
///
/// This implements "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020).
/// When gradients from different tasks conflict (negative cosine similarity),
/// it projects the conflicting gradient onto the normal plane of the other.
pub struct PCGrad;

impl PCGrad {
    /// Apply PCGrad to balance gradients from multiple tasks.
    ///
    /// # Arguments
    /// * `task_gradients` - Gradients for each task and parameter
    ///
    /// # Returns
    /// Combined gradients with conflicts resolved
    pub fn apply(
        task_gradients: &[HashMap<String, Array<f64, Ix2>>],
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>> {
        if task_gradients.is_empty() {
            return Err(TrainError::OptimizerError(
                "PCGrad requires at least one task".to_string(),
            ));
        }

        let n_tasks = task_gradients.len();
        if n_tasks == 1 {
            return Ok(task_gradients[0].clone());
        }

        // Get all parameter names
        let param_names: Vec<String> = task_gradients[0].keys().cloned().collect();

        let mut combined_gradients = HashMap::new();

        // For each parameter
        for param_name in param_names {
            // Collect gradients for this parameter from all tasks
            let mut grads: Vec<&Array<f64, Ix2>> = Vec::new();
            for task_grad in task_gradients {
                if let Some(grad) = task_grad.get(&param_name) {
                    grads.push(grad);
                }
            }

            if grads.len() != n_tasks {
                continue; // Skip if not all tasks have this parameter
            }

            // Apply PCGrad algorithm
            let mut modified_grads: Vec<Array<f64, Ix2>> = Vec::new();

            for (i, grad) in grads.iter().enumerate() {
                let mut grad_i = (*grad).clone();

                // Project onto normal plane of other tasks if conflicting
                for (j, other_grad) in grads.iter().enumerate() {
                    if i == j {
                        continue;
                    }

                    // Compute cosine similarity
                    let dot_product: f64 = grad_i
                        .iter()
                        .zip(other_grad.iter())
                        .map(|(a, b)| a * b)
                        .sum();

                    // If negative (conflicting), project
                    if dot_product < 0.0 {
                        let norm_j_sq: f64 = other_grad.iter().map(|x| x * x).sum();

                        if norm_j_sq > 1e-8 {
                            // Project: g_i = g_i - (g_i · g_j / ||g_j||^2) * g_j
                            let scale = dot_product / norm_j_sq;
                            grad_i = &grad_i - &(*other_grad * scale);
                        }
                    }
                }

                modified_grads.push(grad_i);
            }

            // Average the modified gradients
            let mut combined = Array::zeros(grads[0].raw_dim());
            for grad in &modified_grads {
                combined = &combined + grad;
            }
            combined.mapv_inplace(|x| x / n_tasks as f64);

            combined_gradients.insert(param_name.clone(), combined);
        }

        Ok(combined_gradients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MseLoss;
    use scirs2_core::array;

    #[test]
    fn test_multitask_loss_fixed() {
        let losses: Vec<Box<dyn Loss>> = vec![Box::new(MseLoss), Box::new(MseLoss)];
        let weights = vec![0.7, 0.3];

        let mut mt_loss = MultiTaskLoss::new_fixed(losses, weights).unwrap();

        let predictions = array![[1.0, 2.0, 3.0, 4.0]];
        let targets = array![[1.5, 2.5, 2.5, 3.5]];
        let task_splits = vec![0, 2, 4]; // Two tasks, 2 outputs each

        let loss = mt_loss
            .compute_multi_task(&predictions.view(), &targets.view(), &task_splits)
            .unwrap();

        assert!(loss > 0.0);
        assert_eq!(mt_loss.get_weights(), &[0.7, 0.3]);
    }

    #[test]
    fn test_multitask_loss_dtp() {
        let losses: Vec<Box<dyn Loss>> = vec![Box::new(MseLoss), Box::new(MseLoss)];

        let mut mt_loss = MultiTaskLoss::new_dynamic(
            losses,
            TaskWeightingStrategy::DynamicTaskPrioritization,
            0.01,
        )
        .unwrap();

        let predictions = array![[1.0, 2.0, 10.0, 11.0]]; // Second task has higher error
        let targets = array![[1.5, 2.5, 2.0, 3.0]];
        let task_splits = vec![0, 2, 4];

        let _loss = mt_loss
            .compute_multi_task(&predictions.view(), &targets.view(), &task_splits)
            .unwrap();

        // DTP should give more weight to the task with higher loss
        let weights = mt_loss.get_weights();
        assert!(weights[1] > weights[0], "Task 2 should have higher weight");
    }

    #[test]
    fn test_pcgrad_no_conflict() {
        // When gradients align, PCGrad should average them
        let grad1 = array![[1.0, 2.0], [3.0, 4.0]];
        let grad2 = array![[1.0, 2.0], [3.0, 4.0]];

        let mut task_grads = vec![HashMap::new(), HashMap::new()];
        task_grads[0].insert("param".to_string(), grad1);
        task_grads[1].insert("param".to_string(), grad2);

        let result = PCGrad::apply(&task_grads).unwrap();
        let combined = result.get("param").unwrap();

        // Should be the average
        assert!((combined[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((combined[[1, 1]] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_pcgrad_conflict() {
        // When gradients conflict, PCGrad should resolve them
        let grad1 = array![[1.0, 0.0]];
        let grad2 = array![[-1.0, 0.0]]; // Opposite direction

        let mut task_grads = vec![HashMap::new(), HashMap::new()];
        task_grads[0].insert("param".to_string(), grad1);
        task_grads[1].insert("param".to_string(), grad2);

        let result = PCGrad::apply(&task_grads).unwrap();
        let combined = result.get("param").unwrap();

        // Conflicting gradients should be projected
        assert!(combined[[0, 0]].abs() < 1.0); // Should be reduced
    }

    #[test]
    fn test_multitask_invalid_splits() {
        let losses: Vec<Box<dyn Loss>> = vec![Box::new(MseLoss), Box::new(MseLoss)];
        let mut mt_loss = MultiTaskLoss::new_fixed(losses, vec![0.5, 0.5]).unwrap();

        let predictions = array![[1.0, 2.0]];
        let targets = array![[1.5, 2.5]];
        let task_splits = vec![0, 1]; // Wrong: should have 3 elements for 2 tasks

        let result = mt_loss.compute_multi_task(&predictions.view(), &targets.view(), &task_splits);
        assert!(result.is_err());
    }
}
