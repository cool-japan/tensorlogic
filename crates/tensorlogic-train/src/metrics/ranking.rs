//! Ranking metrics.

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{ArrayView, Ix2};

use super::Metric;

/// Top-K accuracy metric.
/// Measures whether the correct class is in the top K predictions.
#[derive(Debug, Clone)]
pub struct TopKAccuracy {
    /// Number of top predictions to consider.
    pub k: usize,
}

impl Default for TopKAccuracy {
    fn default() -> Self {
        Self { k: 5 }
    }
}

impl TopKAccuracy {
    /// Create a new Top-K accuracy metric.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Metric for TopKAccuracy {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.shape() != targets.shape() {
            return Err(TrainError::MetricsError(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let num_classes = predictions.ncols();
        if self.k > num_classes {
            return Err(TrainError::MetricsError(format!(
                "K ({}) cannot be greater than number of classes ({})",
                self.k, num_classes
            )));
        }

        let mut correct = 0;
        let total = predictions.nrows();

        for i in 0..total {
            // Find true class
            let mut true_class = 0;
            let mut max_true = targets[[i, 0]];
            for j in 1..num_classes {
                if targets[[i, j]] > max_true {
                    max_true = targets[[i, j]];
                    true_class = j;
                }
            }

            // Get top K predictions
            let mut indices: Vec<usize> = (0..num_classes).collect();
            indices.sort_by(|&a, &b| {
                predictions[[i, b]]
                    .partial_cmp(&predictions[[i, a]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Check if true class is in top K
            if indices[..self.k].contains(&true_class) {
                correct += 1;
            }
        }

        Ok(correct as f64 / total as f64)
    }

    fn name(&self) -> &str {
        "top_k_accuracy"
    }
}

/// Normalized Discounted Cumulative Gain (NDCG) metric for ranking.
///
/// NDCG measures the quality of ranking by comparing the predicted order
/// with the ideal order. It accounts for position: items ranked higher
/// contribute more to the score.
///
/// # Formula
/// DCG@k = Σᵢ₌₁ᵏ (2^relᵢ - 1) / log₂(i + 1)
/// NDCG@k = DCG@k / IDCG@k
///
/// where IDCG is the DCG of the ideal ranking.
///
/// # Use Cases
/// - Recommendation systems
/// - Search engine ranking
/// - Information retrieval
/// - Learning to rank
///
/// Reference: Järvelin & Kekäläinen "Cumulated gain-based evaluation of IR techniques" (ACM TOIS 2002)
#[derive(Debug, Clone)]
pub struct NormalizedDiscountedCumulativeGain {
    /// Number of top results to consider (k).
    pub k: usize,
}

impl Default for NormalizedDiscountedCumulativeGain {
    fn default() -> Self {
        Self { k: 10 }
    }
}

impl NormalizedDiscountedCumulativeGain {
    /// Create NDCG metric with custom k value.
    ///
    /// # Arguments
    /// * `k` - Number of top results to consider
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compute DCG (Discounted Cumulative Gain) for a single ranking.
    ///
    /// # Arguments
    /// * `relevances` - Relevance scores in the predicted order
    /// * `k` - Number of positions to consider
    fn compute_dcg(relevances: &[f64], k: usize) -> f64 {
        let k = k.min(relevances.len());
        let mut dcg = 0.0;

        for (i, &rel) in relevances.iter().take(k).enumerate() {
            let position = (i + 2) as f64; // i+2 because positions start at 1 and log₂(1) = 0
            dcg += (2.0_f64.powf(rel) - 1.0) / position.log2();
        }

        dcg
    }

    /// Compute IDCG (Ideal DCG) by sorting relevances in descending order.
    fn compute_idcg(relevances: &[f64], k: usize) -> f64 {
        let mut sorted_rel = relevances.to_vec();
        sorted_rel.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Self::compute_dcg(&sorted_rel, k)
    }
}

impl Metric for NormalizedDiscountedCumulativeGain {
    fn compute(
        &self,
        predictions: &ArrayView<f64, Ix2>,
        targets: &ArrayView<f64, Ix2>,
    ) -> TrainResult<f64> {
        if predictions.shape() != targets.shape() {
            return Err(TrainError::MetricsError(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let n_samples = predictions.nrows();
        if n_samples == 0 {
            return Ok(0.0);
        }

        let mut ndcg_sum = 0.0;

        for i in 0..n_samples {
            // Get predicted scores and true relevances for this sample
            let pred_scores: Vec<f64> = predictions.row(i).iter().copied().collect();
            let true_relevances: Vec<f64> = targets.row(i).iter().copied().collect();

            // Create indices and sort by predicted scores (descending)
            let mut indices: Vec<usize> = (0..pred_scores.len()).collect();
            indices.sort_by(|&a, &b| {
                pred_scores[b]
                    .partial_cmp(&pred_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Reorder relevances according to predicted ranking
            let ranked_relevances: Vec<f64> =
                indices.iter().map(|&idx| true_relevances[idx]).collect();

            // Compute DCG for this ranking
            let dcg = Self::compute_dcg(&ranked_relevances, self.k);

            // Compute IDCG (ideal ranking)
            let idcg = Self::compute_idcg(&true_relevances, self.k);

            // Compute NDCG (handle division by zero)
            let ndcg = if idcg > 1e-12 { dcg / idcg } else { 0.0 };

            ndcg_sum += ndcg;
        }

        Ok(ndcg_sum / n_samples as f64)
    }

    fn name(&self) -> &str {
        "ndcg"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_top_k_accuracy() {
        let metric = TopKAccuracy::new(2);

        // Test with 3 classes
        let predictions = array![
            [0.7, 0.2, 0.1], // Correct class is 0, top-2 includes it
            [0.1, 0.6, 0.3], // Correct class is 1, top-2 includes it
            [0.3, 0.4, 0.3], // Correct class is 2, top-2 includes it (1, 0)
        ];
        let targets = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let top_k = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        assert!((0.0..=1.0).contains(&top_k));
        assert!(top_k >= 0.66); // At least 2/3 should be in top-2
    }

    #[test]
    fn test_ndcg_perfect_ranking() {
        let metric = NormalizedDiscountedCumulativeGain::new(5);

        // Perfect ranking: predicted order matches true relevance order
        let predictions = array![
            [5.0, 4.0, 3.0, 2.0, 1.0], // Pred scores: highest to lowest
        ];
        let targets = array![
            [5.0, 4.0, 3.0, 2.0, 1.0], // True relevances: match pred order
        ];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Perfect ranking should give NDCG = 1.0
        assert!(
            (ndcg - 1.0).abs() < 1e-6,
            "Perfect ranking should have NDCG ≈ 1.0, got {}",
            ndcg
        );
    }

    #[test]
    fn test_ndcg_worst_ranking() {
        let metric = NormalizedDiscountedCumulativeGain::new(5);

        // Worst ranking: predicted order is reverse of true relevance
        let predictions = array![
            [1.0, 2.0, 3.0, 4.0, 5.0], // Pred scores: lowest to highest
        ];
        let targets = array![
            [5.0, 4.0, 3.0, 2.0, 1.0], // True relevances: highest to lowest
        ];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Worst ranking should give low NDCG
        assert!(
            ndcg < 0.8,
            "Worst ranking should have low NDCG, got {}",
            ndcg
        );
    }

    #[test]
    fn test_ndcg_partial_match() {
        let metric = NormalizedDiscountedCumulativeGain::new(3);

        // Partial match: some items ranked correctly
        let predictions = array![
            [4.0, 5.0, 2.0, 3.0, 1.0], // Pred order: [1, 0, 3, 2, 4]
        ];
        let targets = array![
            [3.0, 5.0, 1.0, 2.0, 0.0], // True relevances
        ];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Should be between 0 and 1
        assert!(
            (0.0..=1.0).contains(&ndcg),
            "NDCG should be in [0, 1], got {}",
            ndcg
        );

        // Should be reasonably high since highest relevance (5.0) is predicted correctly
        assert!(
            ndcg > 0.7,
            "NDCG should be > 0.7 for this ranking, got {}",
            ndcg
        );
    }

    #[test]
    fn test_ndcg_multiple_samples() {
        let metric = NormalizedDiscountedCumulativeGain::new(3);

        // Two samples: one perfect, one reversed
        let predictions = array![[5.0, 4.0, 3.0, 2.0], [2.0, 3.0, 4.0, 5.0],];
        let targets = array![[5.0, 4.0, 3.0, 2.0], [5.0, 4.0, 3.0, 2.0],];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Average of perfect (1.0) and poor ranking
        assert!((0.0..=1.0).contains(&ndcg));
        assert!(ndcg > 0.4 && ndcg < 0.9); // Should be somewhere in between
    }

    #[test]
    fn test_ndcg_different_k_values() {
        let metric_k3 = NormalizedDiscountedCumulativeGain::new(3);
        let metric_k5 = NormalizedDiscountedCumulativeGain::new(5);

        let predictions = array![[5.0, 4.0, 3.0, 1.0, 2.0]];
        let targets = array![[5.0, 4.0, 3.0, 2.0, 1.0]];

        let ndcg_k3 = metric_k3
            .compute(&predictions.view(), &targets.view())
            .unwrap();
        let ndcg_k5 = metric_k5
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // k=3 should be perfect (top 3 are correct)
        assert!((ndcg_k3 - 1.0).abs() < 1e-6);

        // k=5 should be lower (last 2 are swapped)
        assert!(ndcg_k5 < ndcg_k3);
        assert!(ndcg_k5 > 0.9); // Still very good
    }

    #[test]
    fn test_ndcg_zero_relevances() {
        let metric = NormalizedDiscountedCumulativeGain::new(5);

        // All zero relevances
        let predictions = array![[1.0, 2.0, 3.0]];
        let targets = array![[0.0, 0.0, 0.0]];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Should handle gracefully (IDCG = 0)
        assert!(ndcg.is_finite());
        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn test_ndcg_empty_input() {
        let metric = NormalizedDiscountedCumulativeGain::default();

        use scirs2_core::ndarray::Array;
        let empty_predictions: Array<f64, _> = Array::zeros((0, 5));
        let empty_targets: Array<f64, _> = Array::zeros((0, 5));

        let ndcg = metric
            .compute(&empty_predictions.view(), &empty_targets.view())
            .unwrap();

        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn test_ndcg_shape_mismatch() {
        let metric = NormalizedDiscountedCumulativeGain::default();

        let predictions = array![[1.0, 2.0, 3.0]];
        let targets = array![[1.0, 2.0]]; // Different shape

        let result = metric.compute(&predictions.view(), &targets.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_ndcg_binary_relevance() {
        let metric = NormalizedDiscountedCumulativeGain::new(5);

        // Binary relevance (0 or 1)
        let predictions = array![[0.9, 0.7, 0.5, 0.3, 0.1]];
        let targets = array![[1.0, 1.0, 0.0, 1.0, 0.0]];

        let ndcg = metric
            .compute(&predictions.view(), &targets.view())
            .unwrap();

        // Should be in valid range
        assert!((0.0..=1.0).contains(&ndcg));

        // Top 2 are relevant, so should have decent NDCG
        assert!(
            ndcg > 0.6,
            "Should have decent NDCG with top-2 relevant, got {}",
            ndcg
        );
    }
}
