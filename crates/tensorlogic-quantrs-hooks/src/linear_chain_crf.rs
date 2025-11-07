//! Linear-chain Conditional Random Fields (CRFs).
//!
//! Linear-chain CRFs are a special case of CRFs where the output variables form a chain.
//! This structure enables efficient inference and learning using dynamic programming.
//!
//! # Applications
//!
//! - Sequence labeling (POS tagging, NER, etc.)
//! - Speech recognition
//! - Bioinformatics (protein sequence analysis)
//!
//! # Algorithm
//!
//! Given input sequence x = (x₁, ..., xₙ) and output sequence y = (y₁, ..., yₙ):
//!
//! ```text
//! P(y|x) = (1/Z(x)) × exp(Σᵢ Σₖ λₖ fₖ(yᵢ₋₁, yᵢ, x, i))
//! ```
//!
//! Where:
//! - fₖ are feature functions
//! - λₖ are learned weights
//! - Z(x) is the partition function
//!
//! # References
//!
//! - Lafferty et al., "Conditional Random Fields: Probabilistic Models for Segmenting
//!   and Labeling Sequence Data" (2001)

use crate::{Factor, FactorGraph, PgmError, Result};
use scirs2_core::ndarray::{Array1, Array2};

/// Feature function for linear-chain CRF.
///
/// Features can be:
/// - Transition features: depend on (yᵢ₋₁, yᵢ, x, i)
/// - Emission features: depend on (yᵢ, x, i)
pub trait FeatureFunction: Send + Sync {
    /// Compute feature value for a transition.
    ///
    /// # Arguments
    /// * `prev_label` - Previous output label (None for first position)
    /// * `curr_label` - Current output label
    /// * `input_sequence` - Input sequence
    /// * `position` - Current position in sequence
    fn compute(
        &self,
        prev_label: Option<usize>,
        curr_label: usize,
        input_sequence: &[usize],
        position: usize,
    ) -> f64;

    /// Get feature name/description.
    fn name(&self) -> &str;
}

/// Linear-chain CRF for sequence labeling.
///
/// This specialized CRF structure enables efficient inference using
/// the forward-backward algorithm and Viterbi decoding.
pub struct LinearChainCRF {
    /// Number of states (labels)
    num_states: usize,
    /// Feature functions with their weights
    features: Vec<(Box<dyn FeatureFunction>, f64)>,
    /// Transition weights matrix: [from_state, to_state]
    transition_weights: Option<Array2<f64>>,
    /// Emission weights matrix: [state, observation]
    emission_weights: Option<Array2<f64>>,
}

impl LinearChainCRF {
    /// Create a new linear-chain CRF.
    pub fn new(num_states: usize) -> Self {
        Self {
            num_states,
            features: Vec::new(),
            transition_weights: None,
            emission_weights: None,
        }
    }

    /// Add a feature function with its weight.
    pub fn add_feature(&mut self, feature: Box<dyn FeatureFunction>, weight: f64) {
        self.features.push((feature, weight));
    }

    /// Set transition weights directly.
    ///
    /// This is useful when you have pre-trained weights.
    pub fn set_transition_weights(&mut self, weights: Array2<f64>) -> Result<()> {
        if weights.shape() != [self.num_states, self.num_states] {
            return Err(PgmError::DimensionMismatch {
                expected: vec![self.num_states, self.num_states],
                got: weights.shape().to_vec(),
            });
        }
        self.transition_weights = Some(weights);
        Ok(())
    }

    /// Set emission weights directly.
    pub fn set_emission_weights(&mut self, weights: Array2<f64>) -> Result<()> {
        if weights.shape()[0] != self.num_states {
            return Err(PgmError::DimensionMismatch {
                expected: vec![self.num_states, weights.shape()[1]],
                got: weights.shape().to_vec(),
            });
        }
        self.emission_weights = Some(weights);
        Ok(())
    }

    /// Compute feature scores for a sequence.
    fn compute_feature_scores(&self, input_sequence: &[usize], position: usize) -> Array2<f64> {
        let mut scores = Array2::zeros((self.num_states, self.num_states));

        // Transition features (from prev_state to curr_state)
        for prev_state in 0..self.num_states {
            for curr_state in 0..self.num_states {
                let mut score = 0.0;

                // Compute weighted feature sum
                for (feature, weight) in &self.features {
                    let feat_val =
                        feature.compute(Some(prev_state), curr_state, input_sequence, position);
                    score += weight * feat_val;
                }

                scores[[prev_state, curr_state]] = score;
            }
        }

        scores
    }

    /// Compute emission scores for a position.
    fn compute_emission_scores(&self, input_sequence: &[usize], position: usize) -> Array1<f64> {
        let mut scores = Array1::zeros(self.num_states);

        for state in 0..self.num_states {
            let mut score = 0.0;

            // Emission features
            for (feature, weight) in &self.features {
                let feat_val = feature.compute(None, state, input_sequence, position);
                score += weight * feat_val;
            }

            // Add pre-trained emission weights if available
            if let Some(ref emission_weights) = self.emission_weights {
                if position < input_sequence.len() {
                    let obs = input_sequence[position];
                    if obs < emission_weights.shape()[1] {
                        score += emission_weights[[state, obs]];
                    }
                }
            }

            scores[state] = score;
        }

        scores
    }

    /// Viterbi algorithm for finding the most likely label sequence.
    ///
    /// Returns the optimal label sequence and its score.
    pub fn viterbi(&self, input_sequence: &[usize]) -> Result<(Vec<usize>, f64)> {
        if input_sequence.is_empty() {
            return Err(PgmError::InvalidGraph("Empty input sequence".to_string()));
        }

        let seq_len = input_sequence.len();

        // Viterbi table: [position, state] -> max score
        let mut viterbi_table = Array2::zeros((seq_len, self.num_states));

        // Backpointer table: [position, state] -> previous state
        let mut backpointers = Array2::zeros((seq_len, self.num_states));

        // Initialize first position
        let emission_scores = self.compute_emission_scores(input_sequence, 0);
        for state in 0..self.num_states {
            viterbi_table[[0, state]] = emission_scores[state];
        }

        // Forward pass
        for t in 1..seq_len {
            let emission_scores = self.compute_emission_scores(input_sequence, t);
            let transition_scores = if let Some(ref weights) = self.transition_weights {
                weights.clone()
            } else {
                self.compute_feature_scores(input_sequence, t)
            };

            for curr_state in 0..self.num_states {
                let mut max_score = f64::NEG_INFINITY;
                let mut best_prev_state = 0;

                for prev_state in 0..self.num_states {
                    let score = viterbi_table[[t - 1, prev_state]]
                        + transition_scores[[prev_state, curr_state]]
                        + emission_scores[curr_state];

                    if score > max_score {
                        max_score = score;
                        best_prev_state = prev_state;
                    }
                }

                viterbi_table[[t, curr_state]] = max_score;
                backpointers[[t, curr_state]] = best_prev_state as f64;
            }
        }

        // Find best final state
        let mut best_final_state = 0;
        let mut best_final_score = f64::NEG_INFINITY;
        for state in 0..self.num_states {
            let score = viterbi_table[[seq_len - 1, state]];
            if score > best_final_score {
                best_final_score = score;
                best_final_state = state;
            }
        }

        // Backward pass to reconstruct path
        let mut path = vec![0; seq_len];
        path[seq_len - 1] = best_final_state;

        for t in (1..seq_len).rev() {
            path[t - 1] = backpointers[[t, path[t]]] as usize;
        }

        Ok((path, best_final_score))
    }

    /// Forward algorithm for computing marginal probabilities.
    ///
    /// Returns forward probabilities: α[t, s] = P(y₁...yₜ = s, x₁...xₜ)
    pub fn forward(&self, input_sequence: &[usize]) -> Result<Array2<f64>> {
        if input_sequence.is_empty() {
            return Err(PgmError::InvalidGraph("Empty input sequence".to_string()));
        }

        let seq_len = input_sequence.len();
        let mut alpha = Array2::zeros((seq_len, self.num_states));

        // Initialize first position
        let emission_scores = self.compute_emission_scores(input_sequence, 0);
        for state in 0..self.num_states {
            alpha[[0, state]] = emission_scores[state].exp();
        }

        // Normalize initial position
        let init_sum: f64 = alpha.row(0).sum();
        if init_sum > 0.0 {
            for state in 0..self.num_states {
                alpha[[0, state]] /= init_sum;
            }
        }

        // Forward pass
        for t in 1..seq_len {
            let emission_scores = self.compute_emission_scores(input_sequence, t);
            let transition_scores = if let Some(ref weights) = self.transition_weights {
                weights.clone()
            } else {
                self.compute_feature_scores(input_sequence, t)
            };

            for curr_state in 0..self.num_states {
                let mut sum = 0.0;

                for prev_state in 0..self.num_states {
                    sum += alpha[[t - 1, prev_state]]
                        * (transition_scores[[prev_state, curr_state]]
                            + emission_scores[curr_state])
                            .exp();
                }

                alpha[[t, curr_state]] = sum;
            }

            // Normalize to prevent underflow
            let row_sum: f64 = alpha.row(t).sum();
            if row_sum > 0.0 {
                for state in 0..self.num_states {
                    alpha[[t, state]] /= row_sum;
                }
            }
        }

        Ok(alpha)
    }

    /// Backward algorithm for computing marginal probabilities.
    ///
    /// Returns backward probabilities: β[t, s] = P(yₜ₊₁...yₙ | yₜ = s, xₜ₊₁...xₙ)
    pub fn backward(&self, input_sequence: &[usize]) -> Result<Array2<f64>> {
        if input_sequence.is_empty() {
            return Err(PgmError::InvalidGraph("Empty input sequence".to_string()));
        }

        let seq_len = input_sequence.len();
        let mut beta = Array2::zeros((seq_len, self.num_states));

        // Initialize last position
        for state in 0..self.num_states {
            beta[[seq_len - 1, state]] = 1.0;
        }

        // Backward pass
        for t in (0..seq_len - 1).rev() {
            let emission_scores = self.compute_emission_scores(input_sequence, t + 1);
            let transition_scores = if let Some(ref weights) = self.transition_weights {
                weights.clone()
            } else {
                self.compute_feature_scores(input_sequence, t + 1)
            };

            for curr_state in 0..self.num_states {
                let mut sum = 0.0;

                for next_state in 0..self.num_states {
                    sum += beta[[t + 1, next_state]]
                        * (transition_scores[[curr_state, next_state]]
                            + emission_scores[next_state])
                            .exp();
                }

                beta[[t, curr_state]] = sum;
            }

            // Normalize to prevent overflow
            let row_sum: f64 = beta.row(t).sum();
            if row_sum > 0.0 {
                for state in 0..self.num_states {
                    beta[[t, state]] /= row_sum;
                }
            }
        }

        Ok(beta)
    }

    /// Compute marginal probabilities for each position.
    ///
    /// Returns: P(yₜ = s | x) for all t and s
    pub fn marginals(&self, input_sequence: &[usize]) -> Result<Array2<f64>> {
        let alpha = self.forward(input_sequence)?;
        let beta = self.backward(input_sequence)?;

        let seq_len = input_sequence.len();
        let mut marginals = Array2::zeros((seq_len, self.num_states));

        for t in 0..seq_len {
            for state in 0..self.num_states {
                marginals[[t, state]] = alpha[[t, state]] * beta[[t, state]];
            }

            // Normalize
            let row_sum: f64 = marginals.row(t).sum();
            if row_sum > 0.0 {
                for state in 0..self.num_states {
                    marginals[[t, state]] /= row_sum;
                }
            }
        }

        Ok(marginals)
    }

    /// Convert to factor graph representation.
    pub fn to_factor_graph(&self, input_sequence: &[usize]) -> Result<FactorGraph> {
        let mut graph = FactorGraph::new();
        let seq_len = input_sequence.len();

        // Add variables for each position
        for t in 0..seq_len {
            graph.add_variable_with_card(format!("y_{}", t), "Label".to_string(), self.num_states);
        }

        // Add emission factors
        for t in 0..seq_len {
            let emission_scores = self.compute_emission_scores(input_sequence, t);
            let emission_potentials = emission_scores.mapv(|x| x.exp());

            let factor = Factor::new(
                format!("emission_{}", t),
                vec![format!("y_{}", t)],
                emission_potentials.into_dyn(),
            )?;

            graph.add_factor(factor)?;
        }

        // Add transition factors
        for t in 1..seq_len {
            let transition_scores = if let Some(ref weights) = self.transition_weights {
                weights.clone()
            } else {
                self.compute_feature_scores(input_sequence, t)
            };

            let transition_potentials = transition_scores.mapv(|x| x.exp());

            let factor = Factor::new(
                format!("transition_{}", t),
                vec![format!("y_{}", t - 1), format!("y_{}", t)],
                transition_potentials.into_dyn(),
            )?;

            graph.add_factor(factor)?;
        }

        Ok(graph)
    }
}

/// Simple identity feature: always returns 1.0
pub struct IdentityFeature {
    name: String,
}

impl IdentityFeature {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl FeatureFunction for IdentityFeature {
    fn compute(
        &self,
        _prev_label: Option<usize>,
        _curr_label: usize,
        _input_sequence: &[usize],
        _position: usize,
    ) -> f64 {
        1.0
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Transition feature: fires when transitioning from one state to another.
pub struct TransitionFeature {
    from_state: usize,
    to_state: usize,
    name: String,
}

impl TransitionFeature {
    pub fn new(from_state: usize, to_state: usize) -> Self {
        Self {
            from_state,
            to_state,
            name: format!("transition_{}_{}", from_state, to_state),
        }
    }
}

impl FeatureFunction for TransitionFeature {
    fn compute(
        &self,
        prev_label: Option<usize>,
        curr_label: usize,
        _input_sequence: &[usize],
        _position: usize,
    ) -> f64 {
        if let Some(prev) = prev_label {
            if prev == self.from_state && curr_label == self.to_state {
                return 1.0;
            }
        }
        0.0
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Emission feature: fires when a specific label is paired with a specific observation.
pub struct EmissionFeature {
    state: usize,
    observation: usize,
    name: String,
}

impl EmissionFeature {
    pub fn new(state: usize, observation: usize) -> Self {
        Self {
            state,
            observation,
            name: format!("emission_{}_{}", state, observation),
        }
    }
}

impl FeatureFunction for EmissionFeature {
    fn compute(
        &self,
        _prev_label: Option<usize>,
        curr_label: usize,
        input_sequence: &[usize],
        position: usize,
    ) -> f64 {
        if curr_label == self.state
            && position < input_sequence.len()
            && input_sequence[position] == self.observation
        {
            return 1.0;
        }
        0.0
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_linear_chain_crf_creation() {
        let crf = LinearChainCRF::new(3);
        assert_eq!(crf.num_states, 3);
        assert_eq!(crf.features.len(), 0);
    }

    #[test]
    fn test_add_feature() {
        let mut crf = LinearChainCRF::new(2);
        let feature = Box::new(IdentityFeature::new("test".to_string()));
        crf.add_feature(feature, 1.0);
        assert_eq!(crf.features.len(), 1);
    }

    #[test]
    fn test_viterbi_simple() {
        let mut crf = LinearChainCRF::new(2);

        // Set simple transition weights favoring 0->0 and 1->1
        let transition_weights = Array::from_shape_vec(
            vec![2, 2],
            vec![1.0, -1.0, -1.0, 1.0], // Prefer staying in same state
        )
        .unwrap()
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .unwrap();
        crf.set_transition_weights(transition_weights).unwrap();

        // Simple input sequence
        let input_sequence = vec![0, 0, 0];

        // Run Viterbi
        let (path, _score) = crf.viterbi(&input_sequence).unwrap();

        assert_eq!(path.len(), 3);
        // With positive weights on diagonal, should prefer staying in state 0
    }

    #[test]
    fn test_forward_backward() {
        let mut crf = LinearChainCRF::new(2);

        // Set uniform transition weights
        let transition_weights = Array::from_shape_vec(vec![2, 2], vec![0.0, 0.0, 0.0, 0.0])
            .unwrap()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .unwrap();
        crf.set_transition_weights(transition_weights).unwrap();

        let input_sequence = vec![0, 1];

        // Run forward
        let alpha = crf.forward(&input_sequence).unwrap();
        assert_eq!(alpha.shape(), &[2, 2]);

        // Run backward
        let beta = crf.backward(&input_sequence).unwrap();
        assert_eq!(beta.shape(), &[2, 2]);

        // Check normalization
        for t in 0..2 {
            let sum: f64 = alpha.row(t).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_marginals() {
        let mut crf = LinearChainCRF::new(2);

        let transition_weights = Array::from_shape_vec(vec![2, 2], vec![0.0, 0.0, 0.0, 0.0])
            .unwrap()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .unwrap();
        crf.set_transition_weights(transition_weights).unwrap();

        let input_sequence = vec![0, 1];

        let marginals = crf.marginals(&input_sequence).unwrap();

        assert_eq!(marginals.shape(), &[2, 2]);

        // Each row should sum to 1
        for t in 0..2 {
            let sum: f64 = marginals.row(t).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_transition_feature() {
        let feature = TransitionFeature::new(0, 1);

        // Should fire when transitioning from 0 to 1
        let val = feature.compute(Some(0), 1, &[0, 1], 1);
        assert_abs_diff_eq!(val, 1.0, epsilon = 1e-10);

        // Should not fire for other transitions
        let val = feature.compute(Some(0), 0, &[0, 1], 1);
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_emission_feature() {
        let feature = EmissionFeature::new(0, 5);

        // Should fire when state=0 and observation=5
        let val = feature.compute(None, 0, &[5, 3], 0);
        assert_abs_diff_eq!(val, 1.0, epsilon = 1e-10);

        // Should not fire for different observation
        let val = feature.compute(None, 0, &[3, 5], 0);
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);

        // Should not fire for different state
        let val = feature.compute(None, 1, &[5, 3], 0);
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_factor_graph() {
        let mut crf = LinearChainCRF::new(2);

        let transition_weights = Array::from_shape_vec(vec![2, 2], vec![0.5, 0.5, 0.5, 0.5])
            .unwrap()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .unwrap();
        crf.set_transition_weights(transition_weights).unwrap();

        let input_sequence = vec![0, 1, 0];

        let graph = crf.to_factor_graph(&input_sequence).unwrap();

        // Should have 3 variables (one per position)
        assert_eq!(graph.num_variables(), 3);

        // Should have 3 emission factors + 2 transition factors = 5 total
        assert_eq!(graph.num_factors(), 5);
    }
}
