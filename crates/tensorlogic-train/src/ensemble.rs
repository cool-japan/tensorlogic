//! Model ensembling utilities for combining multiple models.
//!
//! This module provides various ensemble strategies:
//! - Voting ensembles (hard and soft voting)
//! - Averaging ensembles (simple and weighted)
//! - Stacking ensembles (meta-learner)
//! - Bagging utilities

use crate::{Model, TrainError, TrainResult};
use scirs2_core::ndarray::Array2;

/// Trait for ensemble methods.
pub trait Ensemble {
    /// Predict using the ensemble.
    ///
    /// # Arguments
    /// * `input` - Input data [batch_size, features]
    ///
    /// # Returns
    /// Ensemble predictions [batch_size, num_classes]
    fn predict(&self, input: &Array2<f64>) -> TrainResult<Array2<f64>>;

    /// Get the number of models in the ensemble.
    fn num_models(&self) -> usize;
}

/// Voting ensemble for classification.
///
/// Combines predictions from multiple models using voting:
/// - Hard voting: Majority vote (class with most votes wins)
/// - Soft voting: Average predicted probabilities
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VotingMode {
    /// Hard voting (majority vote).
    Hard,
    /// Soft voting (average probabilities).
    Soft,
}

/// Voting ensemble configuration.
#[derive(Debug)]
pub struct VotingEnsemble<M: Model> {
    /// Base models in the ensemble.
    models: Vec<M>,
    /// Voting mode (hard or soft).
    mode: VotingMode,
    /// Model weights (for weighted voting).
    weights: Option<Vec<f64>>,
}

impl<M: Model> VotingEnsemble<M> {
    /// Create a new voting ensemble.
    ///
    /// # Arguments
    /// * `models` - Base models to ensemble
    /// * `mode` - Voting mode (hard or soft)
    pub fn new(models: Vec<M>, mode: VotingMode) -> TrainResult<Self> {
        if models.is_empty() {
            return Err(TrainError::InvalidParameter(
                "Ensemble must have at least one model".to_string(),
            ));
        }
        Ok(Self {
            models,
            mode,
            weights: None,
        })
    }

    /// Set model weights for weighted voting.
    ///
    /// # Arguments
    /// * `weights` - Weight for each model (must sum to 1.0)
    pub fn with_weights(mut self, weights: Vec<f64>) -> TrainResult<Self> {
        if weights.len() != self.models.len() {
            return Err(TrainError::InvalidParameter(
                "Number of weights must match number of models".to_string(),
            ));
        }

        let sum: f64 = weights.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(TrainError::InvalidParameter(
                "Weights must sum to 1.0".to_string(),
            ));
        }

        self.weights = Some(weights);
        Ok(self)
    }

    /// Get voting mode.
    pub fn mode(&self) -> VotingMode {
        self.mode
    }
}

impl<M: Model> Ensemble for VotingEnsemble<M> {
    fn predict(&self, input: &Array2<f64>) -> TrainResult<Array2<f64>> {
        let batch_size = input.nrows();

        // Collect predictions from all models
        let mut all_predictions = Vec::with_capacity(self.models.len());
        for model in &self.models {
            let pred = model.forward(&input.view())?;
            all_predictions.push(pred);
        }

        // Get output shape from first prediction
        let num_classes = all_predictions[0].ncols();
        let mut ensemble_pred = Array2::zeros((batch_size, num_classes));

        match self.mode {
            VotingMode::Hard => {
                // Hard voting: count votes for each class
                for i in 0..batch_size {
                    let mut votes = vec![0.0; num_classes];

                    for (model_idx, pred) in all_predictions.iter().enumerate() {
                        // Get predicted class (argmax)
                        let row = pred.row(i);
                        let class_idx = row
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);

                        let weight = self.weights.as_ref().map(|w| w[model_idx]).unwrap_or(1.0);
                        votes[class_idx] += weight;
                    }

                    // Convert votes to one-hot prediction
                    let max_votes = votes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let winning_class = votes
                        .iter()
                        .position(|&v| (v - max_votes).abs() < 1e-10)
                        .unwrap();

                    ensemble_pred[[i, winning_class]] = 1.0;
                }
            }
            VotingMode::Soft => {
                // Soft voting: average probabilities
                for i in 0..batch_size {
                    for j in 0..num_classes {
                        let mut weighted_sum = 0.0;

                        for (model_idx, pred) in all_predictions.iter().enumerate() {
                            let weight = self.weights.as_ref().map(|w| w[model_idx]).unwrap_or(1.0);
                            weighted_sum += pred[[i, j]] * weight;
                        }

                        let normalizer = if self.weights.is_some() {
                            1.0 // Weights already sum to 1.0
                        } else {
                            self.models.len() as f64
                        };

                        ensemble_pred[[i, j]] = weighted_sum / normalizer;
                    }
                }
            }
        }

        Ok(ensemble_pred)
    }

    fn num_models(&self) -> usize {
        self.models.len()
    }
}

/// Averaging ensemble for regression.
///
/// Combines predictions by averaging (simple or weighted).
#[derive(Debug)]
pub struct AveragingEnsemble<M: Model> {
    /// Base models in the ensemble.
    models: Vec<M>,
    /// Model weights (for weighted averaging).
    weights: Option<Vec<f64>>,
}

impl<M: Model> AveragingEnsemble<M> {
    /// Create a new averaging ensemble.
    ///
    /// # Arguments
    /// * `models` - Base models to ensemble
    pub fn new(models: Vec<M>) -> TrainResult<Self> {
        if models.is_empty() {
            return Err(TrainError::InvalidParameter(
                "Ensemble must have at least one model".to_string(),
            ));
        }
        Ok(Self {
            models,
            weights: None,
        })
    }

    /// Set model weights for weighted averaging.
    ///
    /// # Arguments
    /// * `weights` - Weight for each model
    pub fn with_weights(mut self, weights: Vec<f64>) -> TrainResult<Self> {
        if weights.len() != self.models.len() {
            return Err(TrainError::InvalidParameter(
                "Number of weights must match number of models".to_string(),
            ));
        }

        // Normalize weights
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            return Err(TrainError::InvalidParameter(
                "Weights must sum to a positive value".to_string(),
            ));
        }

        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / sum).collect();
        self.weights = Some(normalized_weights);
        Ok(self)
    }
}

impl<M: Model> Ensemble for AveragingEnsemble<M> {
    fn predict(&self, input: &Array2<f64>) -> TrainResult<Array2<f64>> {
        // Collect predictions from all models
        let mut all_predictions = Vec::with_capacity(self.models.len());
        for model in &self.models {
            let pred = model.forward(&input.view())?;
            all_predictions.push(pred);
        }

        // Average predictions
        let shape = all_predictions[0].raw_dim();
        let mut ensemble_pred = Array2::zeros(shape);

        for (model_idx, pred) in all_predictions.iter().enumerate() {
            let weight = self.weights.as_ref().map(|w| w[model_idx]).unwrap_or(1.0);

            for i in 0..pred.nrows() {
                for j in 0..pred.ncols() {
                    ensemble_pred[[i, j]] += pred[[i, j]] * weight;
                }
            }
        }

        // Normalize if using uniform weights
        if self.weights.is_none() {
            ensemble_pred /= self.models.len() as f64;
        }

        Ok(ensemble_pred)
    }

    fn num_models(&self) -> usize {
        self.models.len()
    }
}

/// Stacking ensemble with a meta-learner.
///
/// Uses base models' predictions as features for a meta-model.
#[derive(Debug)]
pub struct StackingEnsemble<M: Model, Meta: Model> {
    /// Base models (first level).
    base_models: Vec<M>,
    /// Meta-model (second level).
    meta_model: Meta,
}

impl<M: Model, Meta: Model> StackingEnsemble<M, Meta> {
    /// Create a new stacking ensemble.
    ///
    /// # Arguments
    /// * `base_models` - First-level base models
    /// * `meta_model` - Second-level meta-learner
    pub fn new(base_models: Vec<M>, meta_model: Meta) -> TrainResult<Self> {
        if base_models.is_empty() {
            return Err(TrainError::InvalidParameter(
                "Ensemble must have at least one base model".to_string(),
            ));
        }
        Ok(Self {
            base_models,
            meta_model,
        })
    }

    /// Generate meta-features from base model predictions.
    ///
    /// # Arguments
    /// * `input` - Input data
    ///
    /// # Returns
    /// Meta-features [batch_size, num_base_models * num_classes]
    pub fn generate_meta_features(&self, input: &Array2<f64>) -> TrainResult<Array2<f64>> {
        let batch_size = input.nrows();

        // Collect predictions from all base models
        let mut all_predictions = Vec::with_capacity(self.base_models.len());
        for model in &self.base_models {
            let pred = model.forward(&input.view())?;
            all_predictions.push(pred);
        }

        // Concatenate predictions horizontally to form meta-features
        let num_features_per_model = all_predictions[0].ncols();
        let total_features = self.base_models.len() * num_features_per_model;

        let mut meta_features = Array2::zeros((batch_size, total_features));

        for (model_idx, pred) in all_predictions.iter().enumerate() {
            let start_col = model_idx * num_features_per_model;

            for i in 0..batch_size {
                for j in 0..num_features_per_model {
                    meta_features[[i, start_col + j]] = pred[[i, j]];
                }
            }
        }

        Ok(meta_features)
    }
}

impl<M: Model, Meta: Model> Ensemble for StackingEnsemble<M, Meta> {
    fn predict(&self, input: &Array2<f64>) -> TrainResult<Array2<f64>> {
        // Generate meta-features from base models
        let meta_features = self.generate_meta_features(input)?;

        // Make final prediction with meta-model
        self.meta_model.forward(&meta_features.view())
    }

    fn num_models(&self) -> usize {
        self.base_models.len() + 1 // base models + meta model
    }
}

/// Bagging (Bootstrap Aggregating) utilities.
///
/// Generates bootstrap samples for training ensemble members.
#[derive(Debug)]
pub struct BaggingHelper {
    /// Number of bootstrap samples.
    pub n_estimators: usize,
    /// Random seed for reproducibility.
    pub random_seed: u64,
}

impl BaggingHelper {
    /// Create a new bagging helper.
    ///
    /// # Arguments
    /// * `n_estimators` - Number of bootstrap samples
    /// * `random_seed` - Random seed
    pub fn new(n_estimators: usize, random_seed: u64) -> TrainResult<Self> {
        if n_estimators == 0 {
            return Err(TrainError::InvalidParameter(
                "n_estimators must be positive".to_string(),
            ));
        }
        Ok(Self {
            n_estimators,
            random_seed,
        })
    }

    /// Generate bootstrap sample indices.
    ///
    /// # Arguments
    /// * `n_samples` - Total number of samples
    /// * `estimator_idx` - Index of the estimator (for seeding)
    ///
    /// # Returns
    /// Bootstrap sample indices (with replacement)
    pub fn generate_bootstrap_indices(&self, n_samples: usize, estimator_idx: usize) -> Vec<usize> {
        #[allow(unused_imports)]
        use scirs2_core::random::{Rng, SeedableRng, StdRng};

        let seed = self.random_seed.wrapping_add(estimator_idx as u64);
        let mut rng = StdRng::seed_from_u64(seed);

        (0..n_samples)
            .map(|_| rng.gen_range(0..n_samples))
            .collect()
    }

    /// Get out-of-bag (OOB) indices for an estimator.
    ///
    /// # Arguments
    /// * `n_samples` - Total number of samples
    /// * `bootstrap_indices` - Bootstrap sample indices
    ///
    /// # Returns
    /// OOB sample indices (not in bootstrap sample)
    pub fn get_oob_indices(&self, n_samples: usize, bootstrap_indices: &[usize]) -> Vec<usize> {
        let bootstrap_set: std::collections::HashSet<usize> =
            bootstrap_indices.iter().cloned().collect();

        (0..n_samples)
            .filter(|idx| !bootstrap_set.contains(idx))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearModel;
    use scirs2_core::ndarray::array;

    fn create_test_model() -> LinearModel {
        // Create a 2-input, 2-output linear model
        LinearModel::new(2, 2)
    }

    #[test]
    fn test_voting_ensemble_hard() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let ensemble = VotingEnsemble::new(vec![model1, model2], VotingMode::Hard).unwrap();

        assert_eq!(ensemble.num_models(), 2);
        assert_eq!(ensemble.mode(), VotingMode::Hard);

        let input = array![[1.0, 0.0], [0.0, 1.0]];
        let pred = ensemble.predict(&input).unwrap();

        assert_eq!(pred.shape(), &[2, 2]);
    }

    #[test]
    fn test_voting_ensemble_soft() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let ensemble = VotingEnsemble::new(vec![model1, model2], VotingMode::Soft).unwrap();

        let input = array![[1.0, 0.0]];
        let pred = ensemble.predict(&input).unwrap();

        assert_eq!(pred.shape(), &[1, 2]);
    }

    #[test]
    fn test_voting_ensemble_with_weights() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let ensemble = VotingEnsemble::new(vec![model1, model2], VotingMode::Soft)
            .unwrap()
            .with_weights(vec![0.7, 0.3])
            .unwrap();

        let input = array![[1.0, 0.0]];
        let pred = ensemble.predict(&input).unwrap();

        assert_eq!(pred.shape(), &[1, 2]);
    }

    #[test]
    fn test_voting_ensemble_invalid_weights() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let ensemble = VotingEnsemble::new(vec![model1, model2], VotingMode::Soft).unwrap();

        // Wrong number of weights
        let result = ensemble.with_weights(vec![0.5]);
        assert!(result.is_err());

        // Weights don't sum to 1.0
        let model3 = create_test_model();
        let model4 = create_test_model();
        let ensemble2 = VotingEnsemble::new(vec![model3, model4], VotingMode::Soft).unwrap();
        let result = ensemble2.with_weights(vec![0.5, 0.6]);
        assert!(result.is_err());
    }

    #[test]
    fn test_averaging_ensemble() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let ensemble = AveragingEnsemble::new(vec![model1, model2]).unwrap();

        assert_eq!(ensemble.num_models(), 2);

        let input = array![[1.0, 0.0], [0.0, 1.0]];
        let pred = ensemble.predict(&input).unwrap();

        assert_eq!(pred.shape(), &[2, 2]);
    }

    #[test]
    fn test_averaging_ensemble_with_weights() {
        let model1 = create_test_model();
        let model2 = create_test_model();

        let ensemble = AveragingEnsemble::new(vec![model1, model2])
            .unwrap()
            .with_weights(vec![2.0, 1.0])
            .unwrap();

        let input = array![[1.0, 0.0]];
        let pred = ensemble.predict(&input).unwrap();

        assert_eq!(pred.shape(), &[1, 2]);
    }

    #[test]
    fn test_stacking_ensemble() {
        let base1 = create_test_model(); // 2 inputs, 2 outputs
        let base2 = create_test_model(); // 2 inputs, 2 outputs
        let meta = LinearModel::new(4, 2); // 4 inputs (2 base models × 2 outputs), 2 outputs

        let ensemble = StackingEnsemble::new(vec![base1, base2], meta).unwrap();

        assert_eq!(ensemble.num_models(), 3); // 2 base + 1 meta

        let input = array![[1.0, 0.0]];
        let pred = ensemble.predict(&input).unwrap();

        // Meta-model takes concatenated predictions as input
        assert_eq!(pred.nrows(), 1);
    }

    #[test]
    fn test_stacking_meta_features() {
        let base1 = create_test_model();
        let base2 = create_test_model();
        let meta = create_test_model();

        let ensemble = StackingEnsemble::new(vec![base1, base2], meta).unwrap();

        let input = array![[1.0, 0.0]];
        let meta_features = ensemble.generate_meta_features(&input).unwrap();

        // Should concatenate predictions from 2 base models
        // Each base model outputs 2 features, so total = 2 * 2 = 4
        assert_eq!(meta_features.shape(), &[1, 4]);
    }

    #[test]
    fn test_bagging_helper() {
        let helper = BaggingHelper::new(10, 42).unwrap();

        let indices = helper.generate_bootstrap_indices(100, 0);
        assert_eq!(indices.len(), 100);

        // All indices should be in valid range
        assert!(indices.iter().all(|&i| i < 100));

        // OOB indices should be the complement
        let oob = helper.get_oob_indices(100, &indices);
        assert!(!oob.is_empty());

        for &idx in &oob {
            assert!(!indices.contains(&idx));
        }
    }

    #[test]
    fn test_bagging_helper_different_seeds() {
        let helper = BaggingHelper::new(10, 42).unwrap();

        let indices1 = helper.generate_bootstrap_indices(50, 0);
        let indices2 = helper.generate_bootstrap_indices(50, 1);

        // Different seeds should produce different samples
        assert_ne!(indices1, indices2);
    }

    #[test]
    fn test_bagging_helper_invalid() {
        assert!(BaggingHelper::new(0, 42).is_err());
    }

    #[test]
    fn test_ensemble_empty_models() {
        let result = VotingEnsemble::<LinearModel>::new(vec![], VotingMode::Hard);
        assert!(result.is_err());

        let result = AveragingEnsemble::<LinearModel>::new(vec![]);
        assert!(result.is_err());
    }
}
