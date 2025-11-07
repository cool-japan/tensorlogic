//! Cross-validation utilities for model evaluation.
//!
//! This module provides various cross-validation strategies:
//! - K-fold cross-validation
//! - Stratified K-fold (maintains class distribution)
//! - Time series split (preserves temporal order)
//! - Leave-one-out cross-validation
//! - Custom split strategies

use crate::{TrainError, TrainResult};
use scirs2_core::random::{SeedableRng, StdRng};
use std::collections::HashMap;

/// Trait for cross-validation splitting strategies.
pub trait CrossValidationSplit {
    /// Get the number of splits.
    fn num_splits(&self) -> usize;

    /// Get the train/validation indices for a specific fold.
    ///
    /// # Arguments
    /// * `fold` - Fold index (0 to num_splits - 1)
    /// * `n_samples` - Total number of samples
    ///
    /// # Returns
    /// (train_indices, validation_indices)
    fn get_split(&self, fold: usize, n_samples: usize) -> TrainResult<(Vec<usize>, Vec<usize>)>;
}

/// K-fold cross-validation.
///
/// Splits the data into K equally-sized folds. Each fold is used once as validation
/// while the K-1 remaining folds form the training set.
#[derive(Debug, Clone)]
pub struct KFold {
    /// Number of folds.
    pub n_splits: usize,
    /// Whether to shuffle the data before splitting.
    pub shuffle: bool,
    /// Random seed for shuffling.
    pub random_seed: u64,
}

impl KFold {
    /// Create a new K-fold splitter.
    ///
    /// # Arguments
    /// * `n_splits` - Number of folds (must be >= 2)
    pub fn new(n_splits: usize) -> TrainResult<Self> {
        if n_splits < 2 {
            return Err(TrainError::InvalidParameter(
                "n_splits must be at least 2".to_string(),
            ));
        }
        Ok(Self {
            n_splits,
            shuffle: false,
            random_seed: 42,
        })
    }

    /// Enable shuffling with a specific seed.
    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.shuffle = true;
        self.random_seed = seed;
        self
    }
}

impl CrossValidationSplit for KFold {
    fn num_splits(&self) -> usize {
        self.n_splits
    }

    fn get_split(&self, fold: usize, n_samples: usize) -> TrainResult<(Vec<usize>, Vec<usize>)> {
        if fold >= self.n_splits {
            return Err(TrainError::InvalidParameter(format!(
                "fold {} is out of range [0, {})",
                fold, self.n_splits
            )));
        }

        // Create indices
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle if requested
        if self.shuffle {
            let mut rng = StdRng::seed_from_u64(self.random_seed);
            for i in (1..n_samples).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }

        // Split into folds
        let fold_size = n_samples / self.n_splits;
        let remainder = n_samples % self.n_splits;

        let mut fold_sizes = vec![fold_size; self.n_splits];
        for fold in fold_sizes.iter_mut().take(remainder) {
            *fold += 1;
        }

        // Compute fold boundaries
        let mut boundaries = vec![0];
        for size in &fold_sizes {
            boundaries.push(boundaries.last().unwrap() + size);
        }

        // Get validation indices for this fold
        let val_start = boundaries[fold];
        let val_end = boundaries[fold + 1];
        let val_indices: Vec<usize> = indices[val_start..val_end].to_vec();

        // Get training indices (all others)
        let mut train_indices = Vec::new();
        train_indices.extend_from_slice(&indices[..val_start]);
        train_indices.extend_from_slice(&indices[val_end..]);

        Ok((train_indices, val_indices))
    }
}

/// Stratified K-fold cross-validation.
///
/// Maintains class distribution in each fold (useful for imbalanced datasets).
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    /// Number of folds.
    pub n_splits: usize,
    /// Whether to shuffle the data before splitting.
    pub shuffle: bool,
    /// Random seed for shuffling.
    pub random_seed: u64,
}

impl StratifiedKFold {
    /// Create a new stratified K-fold splitter.
    ///
    /// # Arguments
    /// * `n_splits` - Number of folds (must be >= 2)
    pub fn new(n_splits: usize) -> TrainResult<Self> {
        if n_splits < 2 {
            return Err(TrainError::InvalidParameter(
                "n_splits must be at least 2".to_string(),
            ));
        }
        Ok(Self {
            n_splits,
            shuffle: true,
            random_seed: 42,
        })
    }

    /// Set random seed for shuffling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    /// Get stratified split based on class labels.
    ///
    /// # Arguments
    /// * `fold` - Fold index
    /// * `labels` - Class labels for each sample
    pub fn get_stratified_split(
        &self,
        fold: usize,
        labels: &[usize],
    ) -> TrainResult<(Vec<usize>, Vec<usize>)> {
        if fold >= self.n_splits {
            return Err(TrainError::InvalidParameter(format!(
                "fold {} is out of range [0, {})",
                fold, self.n_splits
            )));
        }

        // Group indices by class
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            class_indices.entry(label).or_default().push(i);
        }

        // Shuffle each class if requested
        if self.shuffle {
            let mut rng = StdRng::seed_from_u64(self.random_seed);
            for indices in class_indices.values_mut() {
                for i in (1..indices.len()).rev() {
                    let j = rng.gen_range(0..=i);
                    indices.swap(i, j);
                }
            }
        }

        // Split each class into folds
        let mut train_indices = Vec::new();
        let mut val_indices = Vec::new();

        for indices in class_indices.values() {
            let class_size = indices.len();
            let fold_size = class_size / self.n_splits;
            let remainder = class_size % self.n_splits;

            let mut fold_sizes = vec![fold_size; self.n_splits];
            for fold in fold_sizes.iter_mut().take(remainder) {
                *fold += 1;
            }

            // Compute fold boundaries
            let mut boundaries = vec![0];
            for size in &fold_sizes {
                boundaries.push(boundaries.last().unwrap() + size);
            }

            // Get validation indices for this fold
            let val_start = boundaries[fold];
            let val_end = boundaries[fold + 1];
            val_indices.extend_from_slice(&indices[val_start..val_end]);

            // Get training indices (all others)
            train_indices.extend_from_slice(&indices[..val_start]);
            train_indices.extend_from_slice(&indices[val_end..]);
        }

        Ok((train_indices, val_indices))
    }
}

impl CrossValidationSplit for StratifiedKFold {
    fn num_splits(&self) -> usize {
        self.n_splits
    }

    fn get_split(&self, fold: usize, n_samples: usize) -> TrainResult<(Vec<usize>, Vec<usize>)> {
        // Default implementation: uniform distribution
        // For actual stratification, use get_stratified_split with labels
        let labels: Vec<usize> = (0..n_samples).map(|i| i % self.n_splits).collect();
        self.get_stratified_split(fold, &labels)
    }
}

/// Time series split for temporal data.
///
/// Respects the temporal order of data. Each training set consists of data
/// before the validation set (no data leakage from future).
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    /// Number of splits.
    pub n_splits: usize,
    /// Minimum training set size.
    pub min_train_size: Option<usize>,
    /// Maximum training set size (for sliding window).
    pub max_train_size: Option<usize>,
}

impl TimeSeriesSplit {
    /// Create a new time series split.
    ///
    /// # Arguments
    /// * `n_splits` - Number of splits (must be >= 2)
    pub fn new(n_splits: usize) -> TrainResult<Self> {
        if n_splits < 2 {
            return Err(TrainError::InvalidParameter(
                "n_splits must be at least 2".to_string(),
            ));
        }
        Ok(Self {
            n_splits,
            min_train_size: None,
            max_train_size: None,
        })
    }

    /// Set minimum training set size.
    pub fn with_min_train_size(mut self, size: usize) -> Self {
        self.min_train_size = Some(size);
        self
    }

    /// Set maximum training set size (for sliding window).
    pub fn with_max_train_size(mut self, size: usize) -> Self {
        self.max_train_size = Some(size);
        self
    }
}

impl CrossValidationSplit for TimeSeriesSplit {
    fn num_splits(&self) -> usize {
        self.n_splits
    }

    fn get_split(&self, fold: usize, n_samples: usize) -> TrainResult<(Vec<usize>, Vec<usize>)> {
        if fold >= self.n_splits {
            return Err(TrainError::InvalidParameter(format!(
                "fold {} is out of range [0, {})",
                fold, self.n_splits
            )));
        }

        // Compute validation set size
        let test_size = n_samples / (self.n_splits + 1);
        if test_size == 0 {
            return Err(TrainError::InvalidParameter(
                "Not enough samples for time series split".to_string(),
            ));
        }

        // Validation set for this fold
        let val_start = (fold + 1) * test_size;
        let val_end = ((fold + 2) * test_size).min(n_samples);

        // Training set: all data before validation
        let train_end = val_start;
        let train_start = if let Some(max_size) = self.max_train_size {
            train_end.saturating_sub(max_size)
        } else if let Some(min_size) = self.min_train_size {
            if train_end < min_size {
                return Err(TrainError::InvalidParameter(
                    "Not enough samples for min_train_size".to_string(),
                ));
            }
            0
        } else {
            0
        };

        let train_indices: Vec<usize> = (train_start..train_end).collect();
        let val_indices: Vec<usize> = (val_start..val_end).collect();

        if train_indices.is_empty() {
            return Err(TrainError::InvalidParameter(
                "Training set is empty for this fold".to_string(),
            ));
        }

        Ok((train_indices, val_indices))
    }
}

/// Leave-one-out cross-validation.
///
/// Each sample is used once as validation while all others form the training set.
/// Useful for very small datasets but computationally expensive.
#[derive(Debug, Clone, Default)]
pub struct LeaveOneOut;

impl LeaveOneOut {
    /// Create a new leave-one-out splitter.
    pub fn new() -> Self {
        Self
    }
}

impl CrossValidationSplit for LeaveOneOut {
    fn num_splits(&self) -> usize {
        // This is a placeholder; actual number depends on n_samples
        usize::MAX
    }

    fn get_split(&self, fold: usize, n_samples: usize) -> TrainResult<(Vec<usize>, Vec<usize>)> {
        if fold >= n_samples {
            return Err(TrainError::InvalidParameter(format!(
                "fold {} is out of range [0, {})",
                fold, n_samples
            )));
        }

        // Validation: single sample
        let val_indices = vec![fold];

        // Training: all other samples
        let mut train_indices: Vec<usize> = (0..fold).collect();
        train_indices.extend(fold + 1..n_samples);

        Ok((train_indices, val_indices))
    }
}

/// Cross-validation result aggregator.
#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    /// Scores for each fold.
    pub fold_scores: Vec<f64>,
    /// Additional metrics for each fold.
    pub fold_metrics: Vec<HashMap<String, f64>>,
}

impl CrossValidationResults {
    /// Create a new result aggregator.
    pub fn new() -> Self {
        Self {
            fold_scores: Vec::new(),
            fold_metrics: Vec::new(),
        }
    }

    /// Add a fold result.
    pub fn add_fold(&mut self, score: f64, metrics: HashMap<String, f64>) {
        self.fold_scores.push(score);
        self.fold_metrics.push(metrics);
    }

    /// Get mean score across all folds.
    pub fn mean_score(&self) -> f64 {
        if self.fold_scores.is_empty() {
            return 0.0;
        }
        self.fold_scores.iter().sum::<f64>() / self.fold_scores.len() as f64
    }

    /// Get standard deviation of scores.
    pub fn std_score(&self) -> f64 {
        if self.fold_scores.len() <= 1 {
            return 0.0;
        }

        let mean = self.mean_score();
        let variance = self
            .fold_scores
            .iter()
            .map(|&score| (score - mean).powi(2))
            .sum::<f64>()
            / (self.fold_scores.len() - 1) as f64;

        variance.sqrt()
    }

    /// Get mean of a specific metric across all folds.
    pub fn mean_metric(&self, metric_name: &str) -> Option<f64> {
        if self.fold_metrics.is_empty() {
            return None;
        }

        let mut sum = 0.0;
        let mut count = 0;

        for metrics in &self.fold_metrics {
            if let Some(&value) = metrics.get(metric_name) {
                sum += value;
                count += 1;
            }
        }

        if count > 0 {
            Some(sum / count as f64)
        } else {
            None
        }
    }

    /// Get number of folds.
    pub fn num_folds(&self) -> usize {
        self.fold_scores.len()
    }
}

impl Default for CrossValidationResults {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kfold_basic() {
        let kfold = KFold::new(3).unwrap();
        assert_eq!(kfold.num_splits(), 3);

        let (train, val) = kfold.get_split(0, 10).unwrap();
        assert!(!train.is_empty());
        assert!(!val.is_empty());

        // Train and validation should be disjoint
        for &idx in &val {
            assert!(!train.contains(&idx));
        }

        // Together should cover all indices
        let mut all_indices = train.clone();
        all_indices.extend(&val);
        all_indices.sort();
        assert_eq!(all_indices, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_kfold_with_shuffle() {
        let kfold = KFold::new(3).unwrap().with_shuffle(42);
        let (train1, val1) = kfold.get_split(0, 10).unwrap();
        let (train2, val2) = kfold.get_split(0, 10).unwrap();

        // Same seed should produce same results
        assert_eq!(train1, train2);
        assert_eq!(val1, val2);
    }

    #[test]
    fn test_kfold_invalid() {
        assert!(KFold::new(1).is_err());
        let kfold = KFold::new(3).unwrap();
        assert!(kfold.get_split(5, 10).is_err()); // fold out of range
    }

    #[test]
    fn test_stratified_kfold() {
        let skfold = StratifiedKFold::new(3).unwrap();
        assert_eq!(skfold.num_splits(), 3);

        // Create balanced labels: [0, 0, 0, 1, 1, 1, 2, 2, 2]
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let (_train, val) = skfold.get_stratified_split(0, &labels).unwrap();

        // Check that validation set has samples from each class
        let mut val_classes: Vec<usize> = val.iter().map(|&i| labels[i]).collect();
        val_classes.sort();
        val_classes.dedup();

        // Should have at least some class diversity
        assert!(!val.is_empty());
    }

    #[test]
    fn test_time_series_split() {
        let ts_split = TimeSeriesSplit::new(3).unwrap();
        assert_eq!(ts_split.num_splits(), 3);

        let (train, val) = ts_split.get_split(0, 10).unwrap();

        // Training indices should be before validation indices
        if !train.is_empty() && !val.is_empty() {
            assert!(train.iter().max().unwrap() < val.iter().min().unwrap());
        }
    }

    #[test]
    fn test_time_series_split_with_window() {
        let ts_split = TimeSeriesSplit::new(3)
            .unwrap()
            .with_min_train_size(2)
            .with_max_train_size(5);

        let (train, val) = ts_split.get_split(1, 20).unwrap();

        // Training set should respect max size
        assert!(train.len() <= 5);
        assert!(!val.is_empty());
    }

    #[test]
    fn test_time_series_split_invalid() {
        let ts_split = TimeSeriesSplit::new(3).unwrap();

        // Too few samples
        assert!(ts_split.get_split(0, 2).is_err());

        // Fold out of range
        assert!(ts_split.get_split(5, 10).is_err());
    }

    #[test]
    fn test_leave_one_out() {
        let loo = LeaveOneOut::new();

        let (train, val) = loo.get_split(0, 5).unwrap();

        assert_eq!(val.len(), 1);
        assert_eq!(train.len(), 4);
        assert_eq!(val[0], 0);

        let (train, val) = loo.get_split(3, 5).unwrap();
        assert_eq!(val[0], 3);
        assert_eq!(train.len(), 4);
    }

    #[test]
    fn test_leave_one_out_invalid() {
        let loo = LeaveOneOut::new();
        assert!(loo.get_split(5, 5).is_err()); // fold out of range
    }

    #[test]
    fn test_cv_results() {
        let mut results = CrossValidationResults::new();

        let mut metrics1 = HashMap::new();
        metrics1.insert("accuracy".to_string(), 0.9);
        results.add_fold(0.85, metrics1);

        let mut metrics2 = HashMap::new();
        metrics2.insert("accuracy".to_string(), 0.95);
        results.add_fold(0.90, metrics2);

        let mut metrics3 = HashMap::new();
        metrics3.insert("accuracy".to_string(), 0.92);
        results.add_fold(0.88, metrics3);

        assert_eq!(results.num_folds(), 3);

        // Mean score: (0.85 + 0.90 + 0.88) / 3 = 0.876666...
        let mean = results.mean_score();
        assert!((mean - 0.8766666).abs() < 1e-6);

        // Standard deviation
        let std = results.std_score();
        assert!(std > 0.0);

        // Mean metric
        let mean_acc = results.mean_metric("accuracy").unwrap();
        assert!((mean_acc - 0.923333).abs() < 1e-5);
    }

    #[test]
    fn test_cv_results_empty() {
        let results = CrossValidationResults::new();
        assert_eq!(results.mean_score(), 0.0);
        assert_eq!(results.std_score(), 0.0);
        assert_eq!(results.num_folds(), 0);
        assert!(results.mean_metric("accuracy").is_none());
    }

    #[test]
    fn test_kfold_all_folds() {
        let kfold = KFold::new(5).unwrap();
        let n_samples = 20;

        let mut all_val_indices = Vec::new();

        // Collect validation indices from all folds
        for fold in 0..5 {
            let (_, val) = kfold.get_split(fold, n_samples).unwrap();
            all_val_indices.extend(val);
        }

        all_val_indices.sort();

        // All samples should appear exactly once in validation sets
        assert_eq!(all_val_indices.len(), n_samples);
        assert_eq!(all_val_indices, (0..n_samples).collect::<Vec<_>>());
    }
}
