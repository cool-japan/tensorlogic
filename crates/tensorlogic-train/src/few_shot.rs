//! Few-shot learning utilities for learning from limited examples.
//!
//! This module provides infrastructure for few-shot learning scenarios where
//! models must learn to generalize from only a small number of labeled examples.
//!
//! # Overview
//!
//! Few-shot learning addresses the challenge of learning new tasks with minimal
//! training data. This module implements:
//!
//! - **Episode sampling**: N-way K-shot task generation
//! - **Prototypical networks**: Learn metric space for prototype-based classification
//! - **Matching networks**: Attention-based matching between query and support sets
//! - **Support set management**: Efficient storage and retrieval of support examples
//! - **Distance metrics**: Various similarity functions for few-shot learning
//!
//! # Key Concepts
//!
//! - **Support set**: Small set of labeled examples used for adaptation
//! - **Query set**: Examples to classify based on the support set
//! - **N-way K-shot**: Task with N classes, K examples per class
//! - **Episode**: Single few-shot task instance during training
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_train::{
//!     EpisodeSampler, PrototypicalDistance, DistanceMetric, ShotType
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a 5-way 1-shot episode sampler
//! let sampler = EpisodeSampler::new(5, ShotType::OneShot, 15);
//!
//! // Use prototypical distance for classification
//! let distance = PrototypicalDistance::euclidean();
//!
//! # Ok(())
//! # }
//! ```

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};

/// Type of shot configuration for few-shot learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotType {
    /// One example per class (1-shot).
    OneShot,
    /// Few examples per class (typically 5-shot).
    FewShot(usize),
    /// Custom number of examples per class.
    Custom(usize),
}

impl ShotType {
    /// Get the number of shots.
    pub fn k(&self) -> usize {
        match self {
            ShotType::OneShot => 1,
            ShotType::FewShot(k) => *k,
            ShotType::Custom(k) => *k,
        }
    }
}

/// Distance metric for few-shot learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm).
    Euclidean,
    /// Cosine similarity (normalized dot product).
    Cosine,
    /// Manhattan distance (L1 norm).
    Manhattan,
    /// Squared Euclidean distance.
    SquaredEuclidean,
}

impl DistanceMetric {
    /// Compute distance between two vectors.
    pub fn compute(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        match self {
            DistanceMetric::Euclidean => {
                let diff = a.to_owned() - b.to_owned();
                diff.dot(&diff).sqrt()
            }
            DistanceMetric::Cosine => {
                let dot = a.dot(b);
                let norm_a = a.dot(a).sqrt();
                let norm_b = b.dot(b).sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    1.0 - (dot / (norm_a * norm_b))
                }
            }
            DistanceMetric::Manhattan => {
                let diff = a.to_owned() - b.to_owned();
                diff.iter().map(|x| x.abs()).sum()
            }
            DistanceMetric::SquaredEuclidean => {
                let diff = a.to_owned() - b.to_owned();
                diff.dot(&diff)
            }
        }
    }
}

/// Support set for few-shot learning.
///
/// Contains labeled examples used for classification or regression.
#[derive(Debug, Clone)]
pub struct SupportSet {
    /// Feature vectors for support examples.
    pub features: Array2<f64>,
    /// Labels for support examples (class indices).
    pub labels: Array1<usize>,
    /// Number of classes.
    pub num_classes: usize,
}

impl SupportSet {
    /// Create a new support set.
    ///
    /// # Arguments
    /// * `features` - Feature matrix (n_examples × n_features)
    /// * `labels` - Class labels (n_examples,)
    ///
    /// # Returns
    /// New support set
    pub fn new(features: Array2<f64>, labels: Array1<usize>) -> TrainResult<Self> {
        if features.nrows() != labels.len() {
            return Err(TrainError::InvalidParameter(format!(
                "Feature rows ({}) must match label count ({})",
                features.nrows(),
                labels.len()
            )));
        }

        let num_classes = labels.iter().max().copied().unwrap_or(0) + 1;

        Ok(Self {
            features,
            labels,
            num_classes,
        })
    }

    /// Get examples for a specific class.
    pub fn get_class_examples(&self, class_id: usize) -> Array2<f64> {
        let indices: Vec<usize> = self
            .labels
            .iter()
            .enumerate()
            .filter(|(_, &label)| label == class_id)
            .map(|(idx, _)| idx)
            .collect();

        if indices.is_empty() {
            return Array2::zeros((0, self.features.ncols()));
        }

        let mut result = Array2::zeros((indices.len(), self.features.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            result.row_mut(i).assign(&self.features.row(idx));
        }
        result
    }

    /// Get number of support examples.
    pub fn size(&self) -> usize {
        self.features.nrows()
    }
}

/// Prototypical distance calculator for few-shot learning.
///
/// Computes distances between query examples and class prototypes.
/// Prototypes are computed as the mean of support examples for each class.
#[derive(Debug, Clone)]
pub struct PrototypicalDistance {
    /// Distance metric to use.
    metric: DistanceMetric,
    /// Class prototypes (computed from support set).
    prototypes: Option<Array2<f64>>,
}

impl PrototypicalDistance {
    /// Create with Euclidean distance.
    pub fn euclidean() -> Self {
        Self {
            metric: DistanceMetric::Euclidean,
            prototypes: None,
        }
    }

    /// Create with cosine distance.
    pub fn cosine() -> Self {
        Self {
            metric: DistanceMetric::Cosine,
            prototypes: None,
        }
    }

    /// Create with custom distance metric.
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            metric,
            prototypes: None,
        }
    }

    /// Compute prototypes from support set.
    ///
    /// # Arguments
    /// * `support` - Support set with labeled examples
    pub fn compute_prototypes(&mut self, support: &SupportSet) {
        let mut prototypes = Array2::zeros((support.num_classes, support.features.ncols()));

        for class_id in 0..support.num_classes {
            let class_examples = support.get_class_examples(class_id);
            if class_examples.nrows() > 0 {
                let prototype = class_examples.mean_axis(Axis(0)).unwrap();
                prototypes.row_mut(class_id).assign(&prototype);
            }
        }

        self.prototypes = Some(prototypes);
    }

    /// Compute distances from query to all prototypes.
    ///
    /// # Arguments
    /// * `query` - Query feature vector
    ///
    /// # Returns
    /// Distance to each prototype (class)
    pub fn compute_distances(&self, query: &ArrayView1<f64>) -> TrainResult<Array1<f64>> {
        let prototypes = self
            .prototypes
            .as_ref()
            .ok_or_else(|| TrainError::Other("Prototypes not computed".to_string()))?;

        let mut distances = Array1::zeros(prototypes.nrows());
        for (i, prototype) in prototypes.axis_iter(Axis(0)).enumerate() {
            distances[i] = self.metric.compute(query, &prototype);
        }

        Ok(distances)
    }

    /// Predict class for query example.
    ///
    /// # Arguments
    /// * `query` - Query feature vector
    ///
    /// # Returns
    /// Predicted class (nearest prototype)
    pub fn predict(&self, query: &ArrayView1<f64>) -> TrainResult<usize> {
        let distances = self.compute_distances(query)?;

        // Find minimum distance
        let mut min_idx = 0;
        let mut min_dist = distances[0];
        for (i, &dist) in distances.iter().enumerate() {
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        Ok(min_idx)
    }

    /// Predict probabilities using softmax over negative distances.
    ///
    /// # Arguments
    /// * `query` - Query feature vector
    /// * `temperature` - Temperature for softmax (default 1.0)
    ///
    /// # Returns
    /// Probability distribution over classes
    pub fn predict_proba(
        &self,
        query: &ArrayView1<f64>,
        temperature: f64,
    ) -> TrainResult<Array1<f64>> {
        let distances = self.compute_distances(query)?;

        // Convert to logits (negative distances)
        let logits = distances.mapv(|d| -d / temperature);

        // Softmax
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits = logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        let probs = exp_logits.mapv(|x| x / sum_exp);

        Ok(probs)
    }
}

/// Episode sampler for N-way K-shot tasks.
///
/// Generates episodes for episodic training in few-shot learning.
#[derive(Debug, Clone)]
pub struct EpisodeSampler {
    /// Number of classes per episode (N-way).
    n_way: usize,
    /// Number of shots per class (K-shot).
    shot_type: ShotType,
    /// Number of query examples per class.
    n_query: usize,
}

impl EpisodeSampler {
    /// Create a new episode sampler.
    ///
    /// # Arguments
    /// * `n_way` - Number of classes per episode
    /// * `shot_type` - Number of shots per class
    /// * `n_query` - Number of query examples per class
    pub fn new(n_way: usize, shot_type: ShotType, n_query: usize) -> Self {
        Self {
            n_way,
            shot_type,
            n_query,
        }
    }

    /// Get total support examples per episode.
    pub fn support_size(&self) -> usize {
        self.n_way * self.shot_type.k()
    }

    /// Get total query examples per episode.
    pub fn query_size(&self) -> usize {
        self.n_way * self.n_query
    }

    /// Get episode description.
    pub fn description(&self) -> String {
        format!(
            "{}-way {}-shot (query: {} per class)",
            self.n_way,
            self.shot_type.k(),
            self.n_query
        )
    }
}

/// Matching network for few-shot learning.
///
/// Uses attention mechanism to match query examples to support examples.
#[derive(Debug, Clone)]
pub struct MatchingNetwork {
    /// Distance metric for similarity.
    metric: DistanceMetric,
    /// Support set.
    support: Option<SupportSet>,
}

impl MatchingNetwork {
    /// Create a new matching network.
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            metric,
            support: None,
        }
    }

    /// Set support set.
    pub fn set_support(&mut self, support: SupportSet) {
        self.support = Some(support);
    }

    /// Compute attention weights between query and all support examples.
    ///
    /// # Arguments
    /// * `query` - Query feature vector
    ///
    /// # Returns
    /// Attention weights for each support example
    pub fn compute_attention(&self, query: &ArrayView1<f64>) -> TrainResult<Array1<f64>> {
        let support = self
            .support
            .as_ref()
            .ok_or_else(|| TrainError::Other("Support set not set".to_string()))?;

        let n_support = support.size();
        let mut similarities = Array1::zeros(n_support);

        // Compute similarities
        for i in 0..n_support {
            let support_example = support.features.row(i);
            similarities[i] = -self.metric.compute(query, &support_example);
        }

        // Softmax to get attention weights
        let max_sim = similarities
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sims = similarities.mapv(|x| (x - max_sim).exp());
        let sum_exp = exp_sims.sum();
        let weights = exp_sims.mapv(|x| x / sum_exp);

        Ok(weights)
    }

    /// Predict class using attention-weighted voting.
    ///
    /// # Arguments
    /// * `query` - Query feature vector
    ///
    /// # Returns
    /// Predicted class probabilities
    pub fn predict_proba(&self, query: &ArrayView1<f64>) -> TrainResult<Array1<f64>> {
        let support = self
            .support
            .as_ref()
            .ok_or_else(|| TrainError::Other("Support set not set".to_string()))?;

        let attention = self.compute_attention(query)?;
        let mut class_probs = Array1::zeros(support.num_classes);

        // Weighted voting
        for (i, &weight) in attention.iter().enumerate() {
            let label = support.labels[i];
            class_probs[label] += weight;
        }

        Ok(class_probs)
    }

    /// Predict class label.
    pub fn predict(&self, query: &ArrayView1<f64>) -> TrainResult<usize> {
        let probs = self.predict_proba(query)?;
        let mut max_idx = 0;
        let mut max_prob = probs[0];
        for (i, &prob) in probs.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                max_idx = i;
            }
        }
        Ok(max_idx)
    }
}

/// Few-shot accuracy evaluator.
#[derive(Debug, Clone, Default)]
pub struct FewShotAccuracy {
    correct: usize,
    total: usize,
}

impl FewShotAccuracy {
    /// Create a new accuracy tracker.
    pub fn new() -> Self {
        Self {
            correct: 0,
            total: 0,
        }
    }

    /// Update with prediction.
    pub fn update(&mut self, predicted: usize, actual: usize) {
        self.total += 1;
        if predicted == actual {
            self.correct += 1;
        }
    }

    /// Get current accuracy.
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    /// Reset counters.
    pub fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    /// Get counts.
    pub fn counts(&self) -> (usize, usize) {
        (self.correct, self.total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_shot_type() {
        assert_eq!(ShotType::OneShot.k(), 1);
        assert_eq!(ShotType::FewShot(5).k(), 5);
        assert_eq!(ShotType::Custom(10).k(), 10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let dist = DistanceMetric::Euclidean.compute(&a.view(), &b.view());
        assert_relative_eq!(dist, 5.196152, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_distance() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);

        let dist = DistanceMetric::Cosine.compute(&a.view(), &b.view());
        assert_relative_eq!(dist, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_support_set_creation() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let support = SupportSet::new(features, labels).unwrap();
        assert_eq!(support.size(), 4);
        assert_eq!(support.num_classes, 2);
    }

    #[test]
    fn test_support_set_get_class_examples() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let support = SupportSet::new(features, labels).unwrap();
        let class_0 = support.get_class_examples(0);

        assert_eq!(class_0.nrows(), 2);
        assert_eq!(class_0[[0, 0]], 1.0);
        assert_eq!(class_0[[1, 0]], 3.0);
    }

    #[test]
    fn test_prototypical_distance() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let support = SupportSet::new(features, labels).unwrap();

        let mut proto = PrototypicalDistance::euclidean();
        proto.compute_prototypes(&support);

        let query = Array1::from_vec(vec![2.0, 3.0]);
        let prediction = proto.predict(&query.view()).unwrap();

        assert_eq!(prediction, 0); // Closer to class 0
    }

    #[test]
    fn test_prototypical_predict_proba() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let support = SupportSet::new(features, labels).unwrap();

        let mut proto = PrototypicalDistance::euclidean();
        proto.compute_prototypes(&support);

        let query = Array1::from_vec(vec![2.0, 3.0]);
        let probs = proto.predict_proba(&query.view(), 1.0).unwrap();

        assert_eq!(probs.len(), 2);
        assert!(probs[0] > probs[1]); // Higher probability for class 0
        assert_relative_eq!(probs.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_episode_sampler() {
        let sampler = EpisodeSampler::new(5, ShotType::OneShot, 15);

        assert_eq!(sampler.support_size(), 5); // 5 classes × 1 shot
        assert_eq!(sampler.query_size(), 75); // 5 classes × 15 queries
        assert!(sampler.description().contains("5-way"));
    }

    #[test]
    fn test_matching_network() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let support = SupportSet::new(features, labels).unwrap();

        let mut matcher = MatchingNetwork::new(DistanceMetric::Euclidean);
        matcher.set_support(support);

        let query = Array1::from_vec(vec![2.0, 3.0]);
        let prediction = matcher.predict(&query.view()).unwrap();

        assert_eq!(prediction, 0); // Should predict class 0
    }

    #[test]
    fn test_matching_network_attention() {
        let features =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let support = SupportSet::new(features, labels).unwrap();

        let mut matcher = MatchingNetwork::new(DistanceMetric::Euclidean);
        matcher.set_support(support);

        let query = Array1::from_vec(vec![2.0, 3.0]);
        let attention = matcher.compute_attention(&query.view()).unwrap();

        assert_eq!(attention.len(), 4);
        assert_relative_eq!(attention.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_few_shot_accuracy() {
        let mut acc = FewShotAccuracy::new();

        acc.update(0, 0); // Correct
        acc.update(1, 1); // Correct
        acc.update(1, 0); // Wrong

        assert_eq!(acc.accuracy(), 2.0 / 3.0);
        assert_eq!(acc.counts(), (2, 3));

        acc.reset();
        assert_eq!(acc.accuracy(), 0.0);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let dist = DistanceMetric::Manhattan.compute(&a.view(), &b.view());
        assert_eq!(dist, 9.0);
    }

    #[test]
    fn test_squared_euclidean_distance() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![4.0, 6.0]);

        let dist = DistanceMetric::SquaredEuclidean.compute(&a.view(), &b.view());
        assert_eq!(dist, 25.0); // (3^2 + 4^2) = 9 + 16 = 25
    }
}
