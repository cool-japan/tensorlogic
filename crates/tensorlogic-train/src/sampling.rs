//! Advanced sampling strategies for training.
//!
//! This module provides sophisticated sampling techniques to improve training efficiency:
//! - **Importance sampling**: Weight samples by their importance for learning
//! - **Hard negative mining**: Focus on difficult negative examples
//! - **Focal sampling**: Emphasize hard-to-classify examples
//! - **Class-balanced sampling**: Handle imbalanced datasets
//! - **Curriculum sampling**: Gradually increase sample difficulty
//!
//! # Examples
//!
//! ## Hard Negative Mining
//! ```rust
//! use tensorlogic_train::{HardNegativeMiner, MiningStrategy};
//! use scirs2_core::ndarray::Array1;
//!
//! let losses = Array1::from_vec(vec![0.1, 0.9, 0.3, 0.8, 0.2]);
//! let labels = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
//!
//! let miner = HardNegativeMiner::new(MiningStrategy::TopK(2), 0.0);
//! let selected = miner.select_samples(&losses, &labels).unwrap();
//! ```
//!
//! ## Importance Sampling
//! ```rust
//! use tensorlogic_train::ImportanceSampler;
//! use scirs2_core::ndarray::Array1;
//!
//! let scores = Array1::from_vec(vec![0.1, 0.5, 0.9, 0.3]);
//! let sampler = ImportanceSampler::new(2, 42);
//! let selected = sampler.sample(&scores).unwrap();
//! ```

use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{TrainError, TrainResult};

/// Strategy for mining hard examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MiningStrategy {
    /// Select top-K samples with highest loss
    TopK(usize),
    /// Select samples above a loss threshold
    Threshold(f64),
    /// Select top percentage of samples
    TopPercentage(f64),
    /// Select samples using focal weighting (emphasize hard examples)
    Focal { gamma: f64, num_samples: usize },
}

/// Hard negative mining for handling imbalanced datasets.
///
/// Focuses training on difficult negative examples to improve classifier discrimination.
///
/// # References
/// - Shrivastava et al. (2016): "Training Region-based Object Detectors with Online Hard Example Mining"
#[derive(Debug, Clone)]
pub struct HardNegativeMiner {
    /// Mining strategy to use
    pub strategy: MiningStrategy,
    /// Ratio of positives to negatives to maintain
    pub pos_neg_ratio: f64,
}

impl HardNegativeMiner {
    /// Create a new hard negative miner.
    pub fn new(strategy: MiningStrategy, pos_neg_ratio: f64) -> Self {
        Self {
            strategy,
            pos_neg_ratio,
        }
    }

    /// Select hard negative samples based on loss values.
    ///
    /// # Arguments
    /// * `losses` - Per-sample loss values
    /// * `labels` - True labels (1.0 for positive, 0.0 for negative)
    ///
    /// # Returns
    /// Indices of selected samples
    pub fn select_samples(
        &self,
        losses: &Array1<f64>,
        labels: &Array1<f64>,
    ) -> TrainResult<Vec<usize>> {
        if losses.len() != labels.len() {
            return Err(TrainError::InvalidParameter(
                "Losses and labels must have same length".to_string(),
            ));
        }

        // Separate positive and negative indices
        let mut pos_indices = Vec::new();
        let mut neg_indices = Vec::new();

        for (idx, &label) in labels.iter().enumerate() {
            if label > 0.5 {
                pos_indices.push(idx);
            } else {
                neg_indices.push(idx);
            }
        }

        // Select all positives
        let mut selected = pos_indices.clone();

        // Select hard negatives based on strategy
        let num_negatives = if self.pos_neg_ratio > 0.0 {
            (pos_indices.len() as f64 * self.pos_neg_ratio) as usize
        } else {
            match &self.strategy {
                MiningStrategy::TopK(k) => *k,
                MiningStrategy::TopPercentage(p) => (neg_indices.len() as f64 * p) as usize,
                MiningStrategy::Focal { num_samples, .. } => *num_samples,
                MiningStrategy::Threshold(_) => neg_indices.len(),
            }
        };

        let hard_negatives = self.select_hard_negatives(losses, &neg_indices, num_negatives)?;
        selected.extend(hard_negatives);

        Ok(selected)
    }

    /// Select hard negative examples.
    fn select_hard_negatives(
        &self,
        losses: &Array1<f64>,
        neg_indices: &[usize],
        num_samples: usize,
    ) -> TrainResult<Vec<usize>> {
        if neg_indices.is_empty() {
            return Ok(Vec::new());
        }

        match &self.strategy {
            MiningStrategy::TopK(_) | MiningStrategy::TopPercentage(_) => {
                // Sort negatives by loss (descending)
                let mut neg_with_loss: Vec<(usize, f64)> =
                    neg_indices.iter().map(|&idx| (idx, losses[idx])).collect();
                neg_with_loss.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let k = num_samples.min(neg_with_loss.len());
                Ok(neg_with_loss.iter().take(k).map(|(idx, _)| *idx).collect())
            }
            MiningStrategy::Threshold(threshold) => {
                // Select all negatives above threshold
                Ok(neg_indices
                    .iter()
                    .filter(|&&idx| losses[idx] > *threshold)
                    .copied()
                    .collect())
            }
            MiningStrategy::Focal { gamma, .. } => {
                // Use focal weighting: (1 - p)^gamma
                let mut neg_with_weight: Vec<(usize, f64)> = neg_indices
                    .iter()
                    .map(|&idx| {
                        let loss = losses[idx];
                        let p = (-loss).exp(); // Approximate probability
                        let weight = (1.0 - p).powf(*gamma);
                        (idx, weight)
                    })
                    .collect();
                neg_with_weight.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let k = num_samples.min(neg_with_weight.len());
                Ok(neg_with_weight
                    .iter()
                    .take(k)
                    .map(|(idx, _)| *idx)
                    .collect())
            }
        }
    }
}

/// Importance sampling based on sample scores.
///
/// Samples examples with probability proportional to their importance scores.
/// Useful for focusing on informative examples.
#[derive(Debug, Clone)]
pub struct ImportanceSampler {
    /// Number of samples to draw
    pub num_samples: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl ImportanceSampler {
    /// Create a new importance sampler.
    pub fn new(num_samples: usize, seed: u64) -> Self {
        Self { num_samples, seed }
    }

    /// Sample indices based on importance scores.
    ///
    /// # Arguments
    /// * `scores` - Importance scores for each sample (higher = more important)
    ///
    /// # Returns
    /// Sampled indices
    pub fn sample(&self, scores: &Array1<f64>) -> TrainResult<Vec<usize>> {
        if scores.is_empty() {
            return Ok(Vec::new());
        }

        // Normalize scores to probabilities
        let total: f64 = scores.iter().sum();
        if total <= 0.0 {
            return Err(TrainError::InvalidParameter(
                "Importance scores must be positive".to_string(),
            ));
        }

        let probabilities: Vec<f64> = scores.iter().map(|&s| s / total).collect();

        // Compute cumulative probabilities
        let mut cumulative = Vec::with_capacity(probabilities.len());
        let mut sum = 0.0;
        for &p in &probabilities {
            sum += p;
            cumulative.push(sum);
        }

        // Sample using linear congruential generator (simple, deterministic)
        let mut selected = Vec::new();
        let mut rng_state = self.seed;

        for _ in 0..self.num_samples {
            // Generate random number in [0, 1)
            rng_state = (rng_state.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
            let rand = (rng_state as f64) / (0x7fffffff as f64);

            // Binary search for the sample
            match cumulative.binary_search_by(|&p| {
                if p < rand {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            }) {
                Ok(idx) => selected.push(idx),
                Err(idx) => selected.push(idx.min(cumulative.len() - 1)),
            }
        }

        Ok(selected)
    }

    /// Sample with replacement allowed.
    pub fn sample_with_replacement(&self, scores: &Array1<f64>) -> TrainResult<Vec<usize>> {
        self.sample(scores)
    }

    /// Sample without replacement (unique indices).
    pub fn sample_without_replacement(&self, scores: &Array1<f64>) -> TrainResult<Vec<usize>> {
        let mut samples = self.sample(scores)?;
        samples.sort_unstable();
        samples.dedup();
        Ok(samples)
    }
}

/// Focal sampling strategy.
///
/// Emphasizes hard-to-classify examples using focal loss weighting.
///
/// # References
/// - Lin et al. (2017): "Focal Loss for Dense Object Detection"
#[derive(Debug, Clone)]
pub struct FocalSampler {
    /// Focusing parameter (higher = more focus on hard examples)
    pub gamma: f64,
    /// Number of samples to select
    pub num_samples: usize,
}

impl FocalSampler {
    /// Create a new focal sampler.
    pub fn new(gamma: f64, num_samples: usize) -> Self {
        Self { gamma, num_samples }
    }

    /// Select samples using focal weighting.
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (probabilities)
    /// * `labels` - True labels
    ///
    /// # Returns
    /// Indices of selected samples
    pub fn select_samples(
        &self,
        predictions: &Array1<f64>,
        labels: &Array1<f64>,
    ) -> TrainResult<Vec<usize>> {
        if predictions.len() != labels.len() {
            return Err(TrainError::InvalidParameter(
                "Predictions and labels must have same length".to_string(),
            ));
        }

        // Compute focal weights: (1 - p_t)^gamma
        let mut weights = Vec::with_capacity(predictions.len());
        for (&pred, &label) in predictions.iter().zip(labels.iter()) {
            let p_t = if label > 0.5 { pred } else { 1.0 - pred };
            let weight = (1.0 - p_t).powf(self.gamma);
            weights.push(weight);
        }

        // Select top samples by weight
        let mut indexed_weights: Vec<(usize, f64)> = weights.into_iter().enumerate().collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = self.num_samples.min(indexed_weights.len());
        Ok(indexed_weights
            .iter()
            .take(k)
            .map(|(idx, _)| *idx)
            .collect())
    }
}

/// Class-balanced sampling for imbalanced datasets.
///
/// Ensures equal representation of all classes during training.
#[derive(Debug, Clone)]
pub struct ClassBalancedSampler {
    /// Number of samples per class
    pub samples_per_class: usize,
    /// Random seed
    pub seed: u64,
}

impl ClassBalancedSampler {
    /// Create a new class-balanced sampler.
    pub fn new(samples_per_class: usize, seed: u64) -> Self {
        Self {
            samples_per_class,
            seed,
        }
    }

    /// Sample balanced batches from each class.
    ///
    /// # Arguments
    /// * `labels` - Class labels
    ///
    /// # Returns
    /// Sampled indices
    pub fn sample(&self, labels: &Array1<f64>) -> TrainResult<Vec<usize>> {
        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();

        for (idx, &label) in labels.iter().enumerate() {
            let class = label.round() as i32;
            class_indices.entry(class).or_default().push(idx);
        }

        if class_indices.is_empty() {
            return Ok(Vec::new());
        }

        // Sample from each class
        let mut selected = Vec::new();
        let mut rng_state = self.seed;

        for (_, indices) in class_indices.iter() {
            let num_to_sample = self.samples_per_class.min(indices.len());

            // Fisher-Yates shuffle and take first k
            let mut shuffled = indices.clone();
            for i in 0..num_to_sample {
                rng_state = (rng_state.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
                let j = i + ((rng_state as usize) % (shuffled.len() - i));
                shuffled.swap(i, j);
            }

            selected.extend_from_slice(&shuffled[..num_to_sample]);
        }

        Ok(selected)
    }

    /// Compute class weights for weighted sampling.
    pub fn compute_class_weights(&self, labels: &Array1<f64>) -> TrainResult<HashMap<i32, f64>> {
        let mut class_counts: HashMap<i32, usize> = HashMap::new();

        for &label in labels.iter() {
            let class = label.round() as i32;
            *class_counts.entry(class).or_insert(0) += 1;
        }

        let total = labels.len() as f64;
        let num_classes = class_counts.len() as f64;

        // Inverse frequency weighting
        let weights: HashMap<i32, f64> = class_counts
            .into_iter()
            .map(|(class, count)| {
                let weight = total / (num_classes * count as f64);
                (class, weight)
            })
            .collect();

        Ok(weights)
    }
}

/// Curriculum sampling for progressive difficulty.
///
/// Gradually introduces harder examples as training progresses.
#[derive(Debug, Clone)]
pub struct CurriculumSampler {
    /// Current training progress (0.0 to 1.0)
    pub progress: f64,
    /// Difficulty scores for each sample
    pub difficulty_scores: Array1<f64>,
    /// Number of samples to select
    pub num_samples: usize,
}

impl CurriculumSampler {
    /// Create a new curriculum sampler.
    pub fn new(difficulty_scores: Array1<f64>, num_samples: usize) -> Self {
        Self {
            progress: 0.0,
            difficulty_scores,
            num_samples,
        }
    }

    /// Update training progress.
    pub fn update_progress(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 1.0);
    }

    /// Select samples based on current curriculum stage.
    ///
    /// # Returns
    /// Indices of samples appropriate for current training stage
    pub fn select_samples(&self) -> TrainResult<Vec<usize>> {
        // Difficulty threshold increases with progress
        let max_difficulty = self.progress;

        // Select samples below difficulty threshold
        let mut candidates: Vec<usize> = self
            .difficulty_scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score <= max_difficulty)
            .map(|(idx, _)| idx)
            .collect();

        // If not enough samples, gradually include harder ones
        if candidates.len() < self.num_samples {
            let mut all_sorted: Vec<(usize, f64)> = self
                .difficulty_scores
                .iter()
                .enumerate()
                .map(|(idx, &score)| (idx, score))
                .collect();
            all_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            candidates = all_sorted
                .iter()
                .take(self.num_samples)
                .map(|(idx, _)| *idx)
                .collect();
        }

        // Randomly sample if we have too many
        if candidates.len() > self.num_samples {
            candidates.truncate(self.num_samples);
        }

        Ok(candidates)
    }
}

/// Online hard example mining during training.
///
/// Dynamically identifies and focuses on hard examples within each batch.
#[derive(Debug, Clone)]
pub struct OnlineHardExampleMiner {
    /// Mining strategy
    pub strategy: MiningStrategy,
    /// Keep easy examples ratio (for stability)
    pub keep_easy_ratio: f64,
}

impl OnlineHardExampleMiner {
    /// Create a new online hard example miner.
    pub fn new(strategy: MiningStrategy, keep_easy_ratio: f64) -> Self {
        Self {
            strategy,
            keep_easy_ratio,
        }
    }

    /// Mine hard examples from a batch.
    ///
    /// # Arguments
    /// * `losses` - Per-sample losses in the batch
    ///
    /// # Returns
    /// Indices of samples to keep for gradient update
    pub fn mine_batch(&self, losses: &Array1<f64>) -> TrainResult<Vec<usize>> {
        if losses.is_empty() {
            return Ok(Vec::new());
        }

        // Sort by loss
        let mut indexed_losses: Vec<(usize, f64)> = losses.iter().copied().enumerate().collect();
        indexed_losses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let total_samples = losses.len();
        let num_hard = match &self.strategy {
            MiningStrategy::TopK(k) => (*k).min(total_samples),
            MiningStrategy::TopPercentage(p) => (total_samples as f64 * p) as usize,
            MiningStrategy::Threshold(t) => {
                indexed_losses.iter().filter(|(_, loss)| *loss > *t).count()
            }
            MiningStrategy::Focal { num_samples, .. } => (*num_samples).min(total_samples),
        };

        // Keep some easy examples for stability
        let num_easy = (total_samples as f64 * self.keep_easy_ratio) as usize;

        // Take hard examples from the front, easy examples from the back
        let mut selected = Vec::new();
        selected.extend(indexed_losses.iter().take(num_hard).map(|(idx, _)| *idx));
        if num_easy > 0 {
            selected.extend(
                indexed_losses
                    .iter()
                    .skip(total_samples - num_easy)
                    .map(|(idx, _)| *idx),
            );
        }

        Ok(selected)
    }
}

/// Batch reweighting based on sample importance.
///
/// Computes weights for each sample in a batch to emphasize important examples.
#[derive(Debug, Clone)]
pub struct BatchReweighter {
    /// Reweighting strategy
    pub strategy: ReweightingStrategy,
}

/// Strategy for reweighting samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReweightingStrategy {
    /// Uniform weights (no reweighting)
    Uniform,
    /// Inverse loss weighting
    InverseLoss { epsilon: f64 },
    /// Focal loss weighting
    Focal { gamma: f64 },
    /// Gradient norm based
    GradientNorm { epsilon: f64 },
}

impl BatchReweighter {
    /// Create a new batch reweighter.
    pub fn new(strategy: ReweightingStrategy) -> Self {
        Self { strategy }
    }

    /// Compute sample weights for a batch.
    ///
    /// # Arguments
    /// * `losses` - Per-sample losses
    ///
    /// # Returns
    /// Weight for each sample
    pub fn compute_weights(&self, losses: &Array1<f64>) -> TrainResult<Array1<f64>> {
        match &self.strategy {
            ReweightingStrategy::Uniform => Ok(Array1::ones(losses.len())),
            ReweightingStrategy::InverseLoss { epsilon } => {
                let weights = losses.mapv(|loss| 1.0 / (loss + epsilon));
                // Normalize
                let sum: f64 = weights.sum();
                Ok(weights * (losses.len() as f64 / sum))
            }
            ReweightingStrategy::Focal { gamma } => {
                // Weight = (1 - p)^gamma where p = exp(-loss)
                let weights = losses.mapv(|loss| {
                    let p = (-loss).exp().min(0.9999);
                    (1.0 - p).powf(*gamma)
                });
                // Normalize
                let sum: f64 = weights.sum();
                Ok(weights * (losses.len() as f64 / sum))
            }
            ReweightingStrategy::GradientNorm { epsilon } => {
                // Approximate gradient norm from loss
                let weights = losses.mapv(|loss| loss.sqrt() + epsilon);
                let sum: f64 = weights.sum();
                Ok(weights * (losses.len() as f64 / sum))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_negative_miner_topk() {
        let losses = Array1::from_vec(vec![0.1, 0.9, 0.3, 0.8, 0.2, 0.7]);
        let labels = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

        let miner = HardNegativeMiner::new(MiningStrategy::TopK(2), 0.0);
        let selected = miner.select_samples(&losses, &labels).unwrap();

        // Should include all positives (0, 2, 4) and top 2 negatives (1, 3)
        assert!(selected.contains(&0));
        assert!(selected.contains(&2));
        assert!(selected.contains(&4));
        assert!(selected.contains(&1)); // Loss 0.9
        assert!(selected.contains(&3)); // Loss 0.8
    }

    #[test]
    fn test_hard_negative_miner_threshold() {
        let losses = Array1::from_vec(vec![0.1, 0.9, 0.3, 0.8, 0.2]);
        let labels = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 0.0]);

        let miner = HardNegativeMiner::new(MiningStrategy::Threshold(0.5), 0.0);
        let selected = miner.select_samples(&losses, &labels).unwrap();

        // Should include all positives and negatives with loss > 0.5
        assert!(selected.contains(&0)); // Positive
        assert!(selected.contains(&2)); // Positive
        assert!(selected.contains(&1)); // Negative, loss 0.9 > 0.5
        assert!(selected.contains(&3)); // Negative, loss 0.8 > 0.5
        assert!(!selected.contains(&4)); // Negative, loss 0.2 < 0.5
    }

    #[test]
    fn test_importance_sampler() {
        let scores = Array1::from_vec(vec![0.1, 0.5, 0.9, 0.3]);
        let sampler = ImportanceSampler::new(3, 42);

        let selected = sampler.sample(&scores).unwrap();
        assert_eq!(selected.len(), 3);

        // Higher scored items should be more likely
        // With seed 42, we should get deterministic results
        assert!(selected.len() <= 4);
    }

    #[test]
    fn test_importance_sampler_without_replacement() {
        let scores = Array1::from_vec(vec![0.1, 0.5, 0.9, 0.3]);
        let sampler = ImportanceSampler::new(5, 42);

        let selected = sampler.sample_without_replacement(&scores).unwrap();

        // Should have unique indices
        let mut sorted = selected.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), selected.len());
    }

    #[test]
    fn test_focal_sampler() {
        let predictions = Array1::from_vec(vec![0.9, 0.1, 0.5, 0.8, 0.3]);
        let labels = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0]);

        let sampler = FocalSampler::new(2.0, 3);
        let selected = sampler.select_samples(&predictions, &labels).unwrap();

        assert_eq!(selected.len(), 3);
        // Should select hard examples (where prediction is far from label)
        assert!(selected.contains(&2)); // pred=0.5, label=1.0 (hard)
    }

    #[test]
    fn test_class_balanced_sampler() {
        let labels = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0]);
        let sampler = ClassBalancedSampler::new(2, 42);

        let selected = sampler.sample(&labels).unwrap();

        // Should sample up to 2 from each class
        // Class 0: 2, Class 1: 2, Class 2: 1 (only 1 available) = 5 total
        assert_eq!(selected.len(), 5);

        // Verify we got samples from each class
        let selected_labels: Vec<f64> = selected.iter().map(|&idx| labels[idx]).collect();
        assert!(selected_labels.contains(&0.0));
        assert!(selected_labels.contains(&1.0));
        assert!(selected_labels.contains(&2.0));
    }

    #[test]
    fn test_class_balanced_weights() {
        let labels = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0]);
        let sampler = ClassBalancedSampler::new(2, 42);

        let weights = sampler.compute_class_weights(&labels).unwrap();

        // Class 0: 3 samples, weight = 6/(3*3) = 0.667
        // Class 1: 2 samples, weight = 6/(3*2) = 1.0
        // Class 2: 1 sample, weight = 6/(3*1) = 2.0
        assert!((weights[&0] - 0.667).abs() < 0.01);
        assert!((weights[&1] - 1.0).abs() < 0.01);
        assert!((weights[&2] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_curriculum_sampler() {
        let difficulty = Array1::from_vec(vec![0.1, 0.3, 0.5, 0.7, 0.9]);
        let mut sampler = CurriculumSampler::new(difficulty, 3);

        // At 0% progress, should only select easiest samples
        sampler.update_progress(0.0);
        let selected = sampler.select_samples().unwrap();
        assert!(!selected.is_empty());

        // At 50% progress, should include medium difficulty
        sampler.update_progress(0.5);
        let selected = sampler.select_samples().unwrap();
        assert!(selected.len() >= 3);

        // At 100% progress, should include all samples
        sampler.update_progress(1.0);
        let selected = sampler.select_samples().unwrap();
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_online_hard_example_miner() {
        let losses = Array1::from_vec(vec![0.1, 0.9, 0.3, 0.8, 0.2]);
        let miner = OnlineHardExampleMiner::new(MiningStrategy::TopK(2), 0.2);

        let selected = miner.mine_batch(&losses).unwrap();

        // Should keep top 2 hard (1, 3) and some easy
        assert!(selected.len() >= 2);
        assert!(selected.contains(&1)); // Highest loss
        assert!(selected.contains(&3)); // Second highest
    }

    #[test]
    fn test_batch_reweighter_uniform() {
        let losses = Array1::from_vec(vec![0.1, 0.5, 0.9]);
        let reweighter = BatchReweighter::new(ReweightingStrategy::Uniform);

        let weights = reweighter.compute_weights(&losses).unwrap();

        assert_eq!(weights.len(), 3);
        assert!((weights[0] - 1.0).abs() < 1e-10);
        assert!((weights[1] - 1.0).abs() < 1e-10);
        assert!((weights[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_reweighter_inverse_loss() {
        let losses = Array1::from_vec(vec![0.1, 0.5, 0.9]);
        let reweighter = BatchReweighter::new(ReweightingStrategy::InverseLoss { epsilon: 0.01 });

        let weights = reweighter.compute_weights(&losses).unwrap();

        // Lower loss should have higher weight (inverse)
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);

        // Weights should sum to number of samples (normalized)
        let sum: f64 = weights.sum();
        assert!((sum - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_reweighter_focal() {
        let losses = Array1::from_vec(vec![0.1, 0.5, 0.9]);
        let reweighter = BatchReweighter::new(ReweightingStrategy::Focal { gamma: 2.0 });

        let weights = reweighter.compute_weights(&losses).unwrap();

        // Higher loss should have higher weight (focal emphasizes hard)
        assert!(weights[2] > weights[1]);
        assert!(weights[1] > weights[0]);

        // Weights should sum to number of samples
        let sum: f64 = weights.sum();
        assert!((sum - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_hard_negative_miner_pos_neg_ratio() {
        let losses = Array1::from_vec(vec![0.1, 0.9, 0.3, 0.8, 0.2, 0.7]);
        let labels = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

        // 3 positives, ratio 1.0 means 3 negatives
        let miner = HardNegativeMiner::new(MiningStrategy::TopK(10), 1.0);
        let selected = miner.select_samples(&losses, &labels).unwrap();

        let num_pos = selected.iter().filter(|&&idx| labels[idx] > 0.5).count();
        let num_neg = selected.iter().filter(|&&idx| labels[idx] < 0.5).count();

        assert_eq!(num_pos, 3);
        assert_eq!(num_neg, 3); // Should select 3 negatives (ratio 1:1)
    }

    #[test]
    fn test_curriculum_sampler_progress_bounds() {
        let difficulty = Array1::from_vec(vec![0.1, 0.5, 0.9]);
        let mut sampler = CurriculumSampler::new(difficulty, 2);

        // Test progress clamping
        sampler.update_progress(-0.5);
        assert_eq!(sampler.progress, 0.0);

        sampler.update_progress(1.5);
        assert_eq!(sampler.progress, 1.0);
    }
}
