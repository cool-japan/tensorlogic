//! Curriculum learning strategies for progressive training.
//!
//! This module provides various curriculum learning strategies that gradually increase
//! training difficulty:
//! - Sample-level curriculum (difficulty scoring)
//! - Competence-based pacing (adaptive difficulty)
//! - Self-paced learning
//! - Task-level curriculum (multi-task progressive training)

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;

/// Trait for curriculum learning strategies.
pub trait CurriculumStrategy {
    /// Get the subset of samples to use for the current training step.
    ///
    /// # Arguments
    /// * `epoch` - Current training epoch
    /// * `total_epochs` - Total number of training epochs
    /// * `difficulties` - Difficulty scores for each sample `[N]`
    ///
    /// # Returns
    /// Indices of samples to include in training at this stage
    fn select_samples(
        &self,
        epoch: usize,
        total_epochs: usize,
        difficulties: &ArrayView1<f64>,
    ) -> TrainResult<Vec<usize>>;

    /// Compute difficulty scores for training samples.
    ///
    /// # Arguments
    /// * `data` - Training data `[N, features]`
    /// * `labels` - Training labels `[N, classes]`
    /// * `predictions` - Model predictions `[N, classes]` (optional, for adaptive strategies)
    ///
    /// # Returns
    /// Difficulty score for each sample `[N]` (higher = more difficult)
    fn compute_difficulty(
        &self,
        data: &Array2<f64>,
        labels: &Array2<f64>,
        predictions: Option<&Array2<f64>>,
    ) -> TrainResult<Array1<f64>>;
}

/// Linear curriculum: gradually increase the percentage of samples used.
///
/// Starts with a small percentage of easiest samples and linearly increases
/// to use all samples by the end of training.
#[derive(Debug, Clone)]
pub struct LinearCurriculum {
    /// Initial percentage of samples to use (0.0 to 1.0).
    pub start_percentage: f64,
    /// Whether to sort by difficulty (true) or use all samples (false).
    pub sort_by_difficulty: bool,
}

impl LinearCurriculum {
    /// Create a new linear curriculum.
    ///
    /// # Arguments
    /// * `start_percentage` - Initial percentage of samples (e.g., 0.1 for 10%)
    pub fn new(start_percentage: f64) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&start_percentage) {
            return Err(TrainError::InvalidParameter(
                "start_percentage must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self {
            start_percentage,
            sort_by_difficulty: true,
        })
    }

    /// Disable sorting by difficulty (use random subset instead).
    pub fn without_sorting(mut self) -> Self {
        self.sort_by_difficulty = false;
        self
    }
}

impl Default for LinearCurriculum {
    fn default() -> Self {
        Self {
            start_percentage: 0.2,
            sort_by_difficulty: true,
        }
    }
}

impl CurriculumStrategy for LinearCurriculum {
    fn select_samples(
        &self,
        epoch: usize,
        total_epochs: usize,
        difficulties: &ArrayView1<f64>,
    ) -> TrainResult<Vec<usize>> {
        let n = difficulties.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Compute current percentage (linear interpolation)
        let progress = if total_epochs > 1 {
            epoch as f64 / (total_epochs - 1) as f64
        } else {
            1.0
        };
        let current_percentage = self.start_percentage + (1.0 - self.start_percentage) * progress;
        let num_samples = ((n as f64 * current_percentage).ceil() as usize).min(n);

        if !self.sort_by_difficulty {
            // Return first num_samples indices
            return Ok((0..num_samples).collect());
        }

        // Sort by difficulty (ascending) and select easiest samples
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            difficulties[a]
                .partial_cmp(&difficulties[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(indices.into_iter().take(num_samples).collect())
    }

    fn compute_difficulty(
        &self,
        _data: &Array2<f64>,
        _labels: &Array2<f64>,
        predictions: Option<&Array2<f64>>,
    ) -> TrainResult<Array1<f64>> {
        // Default: use prediction entropy as difficulty
        // If no predictions provided, use zeros (all equal difficulty)
        if let Some(preds) = predictions {
            let n = preds.nrows();
            let mut difficulties = Array1::zeros(n);

            for i in 0..n {
                let pred = preds.row(i);
                // Compute entropy: -Σ p_i log(p_i)
                let mut entropy = 0.0;
                for &p in pred.iter() {
                    if p > 1e-10 {
                        entropy -= p * p.ln();
                    }
                }
                difficulties[i] = entropy;
            }

            Ok(difficulties)
        } else {
            // No predictions provided, assume all equal difficulty
            Ok(Array1::zeros(_labels.nrows()))
        }
    }
}

/// Exponential curriculum: exponentially increase sample percentage.
///
/// Uses an exponential schedule to quickly ramp up the number of training samples.
#[derive(Debug, Clone)]
pub struct ExponentialCurriculum {
    /// Initial percentage of samples.
    pub start_percentage: f64,
    /// Exponential growth rate (higher = faster growth).
    pub growth_rate: f64,
}

impl ExponentialCurriculum {
    /// Create a new exponential curriculum.
    ///
    /// # Arguments
    /// * `start_percentage` - Initial percentage of samples
    /// * `growth_rate` - Growth rate (e.g., 2.0 for doubling)
    pub fn new(start_percentage: f64, growth_rate: f64) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&start_percentage) {
            return Err(TrainError::InvalidParameter(
                "start_percentage must be in [0, 1]".to_string(),
            ));
        }
        if growth_rate <= 0.0 {
            return Err(TrainError::InvalidParameter(
                "growth_rate must be positive".to_string(),
            ));
        }
        Ok(Self {
            start_percentage,
            growth_rate,
        })
    }
}

impl Default for ExponentialCurriculum {
    fn default() -> Self {
        Self {
            start_percentage: 0.1,
            growth_rate: 2.0,
        }
    }
}

impl CurriculumStrategy for ExponentialCurriculum {
    fn select_samples(
        &self,
        epoch: usize,
        total_epochs: usize,
        difficulties: &ArrayView1<f64>,
    ) -> TrainResult<Vec<usize>> {
        let n = difficulties.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Exponential growth: p(t) = start * exp(growth * t)
        let progress = if total_epochs > 1 {
            epoch as f64 / (total_epochs - 1) as f64
        } else {
            1.0
        };
        let current_percentage =
            (self.start_percentage * (self.growth_rate * progress).exp()).min(1.0);
        let num_samples = ((n as f64 * current_percentage).ceil() as usize).min(n);

        // Sort by difficulty and select easiest samples
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            difficulties[a]
                .partial_cmp(&difficulties[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(indices.into_iter().take(num_samples).collect())
    }

    fn compute_difficulty(
        &self,
        _data: &Array2<f64>,
        _labels: &Array2<f64>,
        predictions: Option<&Array2<f64>>,
    ) -> TrainResult<Array1<f64>> {
        // Same as LinearCurriculum
        if let Some(preds) = predictions {
            let n = preds.nrows();
            let mut difficulties = Array1::zeros(n);

            for i in 0..n {
                let pred = preds.row(i);
                let mut entropy = 0.0;
                for &p in pred.iter() {
                    if p > 1e-10 {
                        entropy -= p * p.ln();
                    }
                }
                difficulties[i] = entropy;
            }

            Ok(difficulties)
        } else {
            Ok(Array1::zeros(_labels.nrows()))
        }
    }
}

/// Self-paced learning: model determines its own learning pace.
///
/// Adaptively selects samples based on current model performance,
/// prioritizing samples the model is ready to learn from.
#[derive(Debug, Clone)]
pub struct SelfPacedCurriculum {
    /// Age parameter controlling pace (higher = more aggressive).
    pub lambda: f64,
    /// Threshold for sample selection.
    pub threshold: f64,
}

impl SelfPacedCurriculum {
    /// Create a new self-paced curriculum.
    ///
    /// # Arguments
    /// * `lambda` - Age parameter (controls learning pace)
    /// * `threshold` - Selection threshold
    pub fn new(lambda: f64, threshold: f64) -> TrainResult<Self> {
        if lambda <= 0.0 {
            return Err(TrainError::InvalidParameter(
                "lambda must be positive".to_string(),
            ));
        }
        Ok(Self { lambda, threshold })
    }
}

impl Default for SelfPacedCurriculum {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            threshold: 0.5,
        }
    }
}

impl CurriculumStrategy for SelfPacedCurriculum {
    fn select_samples(
        &self,
        _epoch: usize,
        _total_epochs: usize,
        difficulties: &ArrayView1<f64>,
    ) -> TrainResult<Vec<usize>> {
        // Select samples with difficulty below threshold
        let indices: Vec<usize> = difficulties
            .iter()
            .enumerate()
            .filter(|(_, &d)| d < self.threshold)
            .map(|(i, _)| i)
            .collect();

        Ok(indices)
    }

    fn compute_difficulty(
        &self,
        _data: &Array2<f64>,
        labels: &Array2<f64>,
        predictions: Option<&Array2<f64>>,
    ) -> TrainResult<Array1<f64>> {
        if let Some(preds) = predictions {
            let n = preds.nrows();
            let mut difficulties = Array1::zeros(n);

            for i in 0..n {
                // Compute loss for each sample (cross-entropy)
                let pred = preds.row(i);
                let label = labels.row(i);

                let mut loss = 0.0;
                for j in 0..pred.len() {
                    let p = pred[j].clamp(1e-10, 1.0 - 1e-10);
                    loss -= label[j] * p.ln();
                }

                // Weight by self-pacing parameter
                difficulties[i] = loss * self.lambda;
            }

            Ok(difficulties)
        } else {
            Err(TrainError::InvalidParameter(
                "SelfPacedCurriculum requires predictions for difficulty computation".to_string(),
            ))
        }
    }
}

/// Competence-based curriculum: adapts to model's current competence level.
///
/// Gradually increases difficulty based on model's mastery of easier samples.
#[derive(Debug, Clone)]
pub struct CompetenceCurriculum {
    /// Initial competence level (0.0 to 1.0).
    pub initial_competence: f64,
    /// Competence growth rate per epoch.
    pub growth_rate: f64,
    /// Maximum competence level.
    pub max_competence: f64,
}

impl CompetenceCurriculum {
    /// Create a new competence-based curriculum.
    ///
    /// # Arguments
    /// * `initial_competence` - Starting competence level
    /// * `growth_rate` - How fast competence grows per epoch
    pub fn new(initial_competence: f64, growth_rate: f64) -> TrainResult<Self> {
        if !(0.0..=1.0).contains(&initial_competence) {
            return Err(TrainError::InvalidParameter(
                "initial_competence must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self {
            initial_competence,
            growth_rate,
            max_competence: 1.0,
        })
    }
}

impl Default for CompetenceCurriculum {
    fn default() -> Self {
        Self {
            initial_competence: 0.3,
            growth_rate: 0.05,
            max_competence: 1.0,
        }
    }
}

impl CurriculumStrategy for CompetenceCurriculum {
    fn select_samples(
        &self,
        epoch: usize,
        _total_epochs: usize,
        difficulties: &ArrayView1<f64>,
    ) -> TrainResult<Vec<usize>> {
        // Current competence level
        let competence =
            (self.initial_competence + self.growth_rate * epoch as f64).min(self.max_competence);

        // Select samples with difficulty <= competence
        let indices: Vec<usize> = difficulties
            .iter()
            .enumerate()
            .filter(|(_, &d)| d <= competence)
            .map(|(i, _)| i)
            .collect();

        Ok(indices)
    }

    fn compute_difficulty(
        &self,
        _data: &Array2<f64>,
        _labels: &Array2<f64>,
        predictions: Option<&Array2<f64>>,
    ) -> TrainResult<Array1<f64>> {
        // Normalize difficulties to [0, 1]
        if let Some(preds) = predictions {
            let n = preds.nrows();
            let mut difficulties = Array1::zeros(n);

            for i in 0..n {
                let pred = preds.row(i);
                let mut entropy = 0.0;
                for &p in pred.iter() {
                    if p > 1e-10 {
                        entropy -= p * p.ln();
                    }
                }
                difficulties[i] = entropy;
            }

            // Normalize to [0, 1]
            let max_difficulty = difficulties.iter().cloned().fold(0.0f64, f64::max);
            if max_difficulty > 0.0 {
                difficulties.mapv_inplace(|d| d / max_difficulty);
            }

            Ok(difficulties)
        } else {
            Ok(Array1::zeros(_labels.nrows()))
        }
    }
}

/// Task-level curriculum for multi-task learning.
///
/// Progressively introduces different tasks during training.
#[derive(Debug, Clone)]
pub struct TaskCurriculum {
    /// Task schedule: (start_epoch, task_id) pairs.
    task_schedule: Vec<(usize, usize)>,
}

impl TaskCurriculum {
    /// Create a new task curriculum.
    ///
    /// # Arguments
    /// * `schedule` - Task introduction schedule [(epoch, task_id)]
    pub fn new(schedule: Vec<(usize, usize)>) -> Self {
        let mut sorted_schedule = schedule;
        sorted_schedule.sort_by_key(|(epoch, _)| *epoch);
        Self {
            task_schedule: sorted_schedule,
        }
    }

    /// Get active tasks for the current epoch.
    ///
    /// # Arguments
    /// * `epoch` - Current training epoch
    ///
    /// # Returns
    /// Set of active task IDs
    pub fn get_active_tasks(&self, epoch: usize) -> Vec<usize> {
        self.task_schedule
            .iter()
            .filter(|(start_epoch, _)| *start_epoch <= epoch)
            .map(|(_, task_id)| *task_id)
            .collect()
    }
}

impl Default for TaskCurriculum {
    fn default() -> Self {
        // Default: single task from epoch 0
        Self {
            task_schedule: vec![(0, 0)],
        }
    }
}

/// Manager for curriculum learning that tracks training progress.
pub struct CurriculumManager {
    strategy: Box<dyn CurriculumStrategyClone>,
    difficulty_cache: HashMap<String, Array1<f64>>,
    current_epoch: usize,
}

impl std::fmt::Debug for CurriculumManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CurriculumManager")
            .field("current_epoch", &self.current_epoch)
            .field("num_cached_difficulties", &self.difficulty_cache.len())
            .finish()
    }
}

/// Helper trait for cloning curriculum strategies.
trait CurriculumStrategyClone: CurriculumStrategy {
    fn clone_box(&self) -> Box<dyn CurriculumStrategyClone>;
}

impl<T: CurriculumStrategy + Clone + 'static> CurriculumStrategyClone for T {
    fn clone_box(&self) -> Box<dyn CurriculumStrategyClone> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CurriculumStrategyClone> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl CurriculumStrategy for Box<dyn CurriculumStrategyClone> {
    fn select_samples(
        &self,
        epoch: usize,
        total_epochs: usize,
        difficulties: &ArrayView1<f64>,
    ) -> TrainResult<Vec<usize>> {
        (**self).select_samples(epoch, total_epochs, difficulties)
    }

    fn compute_difficulty(
        &self,
        data: &Array2<f64>,
        labels: &Array2<f64>,
        predictions: Option<&Array2<f64>>,
    ) -> TrainResult<Array1<f64>> {
        (**self).compute_difficulty(data, labels, predictions)
    }
}

impl CurriculumManager {
    /// Create a new curriculum manager.
    ///
    /// # Arguments
    /// * `strategy` - Curriculum learning strategy
    pub fn new<S: CurriculumStrategy + Clone + 'static>(strategy: S) -> Self {
        Self {
            strategy: Box::new(strategy),
            difficulty_cache: HashMap::new(),
            current_epoch: 0,
        }
    }

    /// Update the current epoch.
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    /// Compute and cache difficulty scores.
    ///
    /// # Arguments
    /// * `key` - Cache key (e.g., "train", "val")
    /// * `data` - Training data
    /// * `labels` - Training labels
    /// * `predictions` - Optional model predictions
    pub fn compute_difficulty(
        &mut self,
        key: &str,
        data: &Array2<f64>,
        labels: &Array2<f64>,
        predictions: Option<&Array2<f64>>,
    ) -> TrainResult<()> {
        let difficulties = self
            .strategy
            .compute_difficulty(data, labels, predictions)?;
        self.difficulty_cache.insert(key.to_string(), difficulties);
        Ok(())
    }

    /// Get selected sample indices for training.
    ///
    /// # Arguments
    /// * `key` - Cache key for difficulty scores
    /// * `total_epochs` - Total number of training epochs
    ///
    /// # Returns
    /// Indices of samples to use for current epoch
    pub fn get_selected_samples(&self, key: &str, total_epochs: usize) -> TrainResult<Vec<usize>> {
        let difficulties = self.difficulty_cache.get(key).ok_or_else(|| {
            TrainError::InvalidParameter(format!("No difficulty scores cached for key: {}", key))
        })?;

        self.strategy
            .select_samples(self.current_epoch, total_epochs, &difficulties.view())
    }

    /// Clear the difficulty cache.
    pub fn clear_cache(&mut self) {
        self.difficulty_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_linear_curriculum() {
        let curriculum = LinearCurriculum::new(0.2).unwrap();
        let difficulties = array![0.1, 0.5, 0.3, 0.9, 0.2];

        // At epoch 0, should select 20% of samples (1 sample)
        let selected = curriculum
            .select_samples(0, 10, &difficulties.view())
            .unwrap();
        assert_eq!(selected.len(), 1);

        // At epoch 9 (last epoch), should select all samples
        let selected = curriculum
            .select_samples(9, 10, &difficulties.view())
            .unwrap();
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_linear_curriculum_invalid() {
        assert!(LinearCurriculum::new(-0.1).is_err());
        assert!(LinearCurriculum::new(1.5).is_err());
    }

    #[test]
    fn test_exponential_curriculum() {
        let curriculum = ExponentialCurriculum::new(0.1, 2.0).unwrap();
        let difficulties = array![0.1, 0.5, 0.3, 0.9, 0.2];

        let selected = curriculum
            .select_samples(0, 10, &difficulties.view())
            .unwrap();
        assert!(!selected.is_empty());

        let selected = curriculum
            .select_samples(9, 10, &difficulties.view())
            .unwrap();
        // Should select most/all samples at the end (exponential growth may round differently)
        assert!(selected.len() >= 4);
    }

    #[test]
    fn test_self_paced_curriculum() {
        let curriculum = SelfPacedCurriculum::new(1.0, 0.5).unwrap();
        let difficulties = array![0.1, 0.6, 0.3, 0.9, 0.2];

        // Should select samples with difficulty < 0.5
        let selected = curriculum
            .select_samples(0, 10, &difficulties.view())
            .unwrap();
        assert_eq!(selected.len(), 3); // indices 0, 2, 4
    }

    #[test]
    fn test_competence_curriculum() {
        let curriculum = CompetenceCurriculum::new(0.3, 0.1).unwrap();
        let difficulties = array![0.1, 0.5, 0.3, 0.9, 0.2];

        // At epoch 0, competence = 0.3, should select difficulties <= 0.3
        let selected = curriculum
            .select_samples(0, 10, &difficulties.view())
            .unwrap();
        assert_eq!(selected.len(), 3); // indices 0, 2, 4

        // At epoch 5, competence = 0.8, should select more samples
        let selected = curriculum
            .select_samples(5, 10, &difficulties.view())
            .unwrap();
        assert!(selected.len() >= 3);
    }

    #[test]
    fn test_task_curriculum() {
        let curriculum = TaskCurriculum::new(vec![(0, 0), (5, 1), (10, 2)]);

        let tasks = curriculum.get_active_tasks(0);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0], 0);

        let tasks = curriculum.get_active_tasks(7);
        assert_eq!(tasks.len(), 2);
        assert!(tasks.contains(&0));
        assert!(tasks.contains(&1));

        let tasks = curriculum.get_active_tasks(15);
        assert_eq!(tasks.len(), 3);
    }

    #[test]
    fn test_difficulty_computation() {
        let curriculum = LinearCurriculum::default();

        // Test with predictions
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array![[1.0, 0.0], [0.0, 1.0]];
        let predictions = array![[0.8, 0.2], [0.3, 0.7]];

        let difficulties = curriculum
            .compute_difficulty(&data, &labels, Some(&predictions))
            .unwrap();
        assert_eq!(difficulties.len(), 2);
        assert!(difficulties.iter().all(|&d| d >= 0.0));

        // Test without predictions
        let difficulties = curriculum.compute_difficulty(&data, &labels, None).unwrap();
        assert_eq!(difficulties.len(), 2);
        assert!(difficulties.iter().all(|&d| d == 0.0));
    }

    #[test]
    fn test_curriculum_manager() {
        let mut manager = CurriculumManager::new(LinearCurriculum::default());

        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let predictions = array![[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]];

        // Compute difficulties
        manager
            .compute_difficulty("train", &data, &labels, Some(&predictions))
            .unwrap();

        // Get selected samples
        manager.set_epoch(0);
        let selected = manager.get_selected_samples("train", 10).unwrap();
        assert!(!selected.is_empty());

        // Clear cache
        manager.clear_cache();
    }

    #[test]
    fn test_curriculum_manager_missing_key() {
        let manager = CurriculumManager::new(LinearCurriculum::default());
        let result = manager.get_selected_samples("nonexistent", 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_curriculum_without_sorting() {
        let curriculum = LinearCurriculum::new(0.5).unwrap().without_sorting();
        let difficulties = array![0.9, 0.1, 0.5, 0.3, 0.7];

        // Should not sort by difficulty
        let selected = curriculum
            .select_samples(0, 10, &difficulties.view())
            .unwrap();
        assert_eq!(selected.len(), 3); // 50% of 5 samples, rounded up
    }

    #[test]
    fn test_empty_difficulties() {
        let curriculum = LinearCurriculum::default();
        let difficulties = Array1::<f64>::zeros(0);

        let selected = curriculum
            .select_samples(0, 10, &difficulties.view())
            .unwrap();
        assert_eq!(selected.len(), 0);
    }
}
