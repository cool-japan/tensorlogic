//! Parameter learning algorithms for probabilistic graphical models.
//!
//! This module provides algorithms for learning model parameters from data:
//! - **Maximum Likelihood Estimation (MLE)**: For fully observed data
//! - **Expectation-Maximization (EM)**: For partially observed data
//! - **Bayesian Estimation**: With Dirichlet priors
//! - **Baum-Welch Algorithm**: Specialized EM for Hidden Markov Models
//!
//! # Overview
//!
//! Parameter learning is the process of estimating the parameters (probabilities)
//! of a probabilistic model from observed data. This module supports both:
//! - **Complete data**: All variables are observed (use MLE)
//! - **Incomplete data**: Some variables are hidden (use EM)
//!
//! # Examples
//!
//! ```ignore
//! // Learn HMM parameters from observed sequences
//! let mut hmm = HiddenMarkovModel::new(2, 2);
//! let learner = BaumWelchLearner::new(100, 1e-4);
//! learner.learn(&mut hmm, &observation_sequences)?;
//! ```

use crate::error::{PgmError, Result};
use crate::sampling::Assignment;
use scirs2_core::ndarray::{Array1, Array2, ArrayD};
use std::collections::HashMap;

/// Simple HMM representation for parameter learning.
///
/// This is a standalone representation with explicit parameter matrices,
/// designed for efficient parameter learning algorithms like Baum-Welch.
#[derive(Debug, Clone)]
pub struct SimpleHMM {
    /// Number of hidden states
    pub num_states: usize,
    /// Number of observable symbols
    pub num_observations: usize,
    /// Initial state distribution π: `[num_states]`
    pub initial_distribution: Array1<f64>,
    /// Transition probabilities A: [from_state, to_state]
    pub transition_probabilities: Array2<f64>,
    /// Emission probabilities B: [state, observation]
    pub emission_probabilities: Array2<f64>,
}

impl SimpleHMM {
    /// Create a new SimpleHMM with uniform initialization.
    pub fn new(num_states: usize, num_observations: usize) -> Self {
        let initial_distribution = Array1::from_elem(num_states, 1.0 / num_states as f64);

        let transition_probabilities =
            Array2::from_elem((num_states, num_states), 1.0 / num_states as f64);

        let emission_probabilities = Array2::from_elem(
            (num_states, num_observations),
            1.0 / num_observations as f64,
        );

        Self {
            num_states,
            num_observations,
            initial_distribution,
            transition_probabilities,
            emission_probabilities,
        }
    }

    /// Create an HMM with random initialization.
    pub fn new_random(num_states: usize, num_observations: usize) -> Self {
        use scirs2_core::random::thread_rng;

        let mut rng = thread_rng();
        let mut hmm = Self::new(num_states, num_observations);

        // Randomize initial distribution
        let mut init_sum = 0.0;
        for i in 0..num_states {
            hmm.initial_distribution[i] = rng.random::<f64>();
            init_sum += hmm.initial_distribution[i];
        }
        hmm.initial_distribution /= init_sum;

        // Randomize transition probabilities
        for i in 0..num_states {
            let mut trans_sum = 0.0;
            for j in 0..num_states {
                hmm.transition_probabilities[[i, j]] = rng.random::<f64>();
                trans_sum += hmm.transition_probabilities[[i, j]];
            }
            for j in 0..num_states {
                hmm.transition_probabilities[[i, j]] /= trans_sum;
            }
        }

        // Randomize emission probabilities
        for i in 0..num_states {
            let mut emission_sum = 0.0;
            for j in 0..num_observations {
                hmm.emission_probabilities[[i, j]] = rng.random::<f64>();
                emission_sum += hmm.emission_probabilities[[i, j]];
            }
            for j in 0..num_observations {
                hmm.emission_probabilities[[i, j]] /= emission_sum;
            }
        }

        hmm
    }
}

/// Maximum Likelihood Estimator for discrete distributions.
///
/// Estimates parameters by counting frequencies in complete data.
#[derive(Debug, Clone)]
pub struct MaximumLikelihoodEstimator {
    /// Use Laplace smoothing (add-one smoothing)
    pub use_laplace: bool,
    /// Pseudocount for Laplace smoothing
    pub pseudocount: f64,
}

impl MaximumLikelihoodEstimator {
    /// Create a new MLE estimator.
    pub fn new() -> Self {
        Self {
            use_laplace: false,
            pseudocount: 1.0,
        }
    }

    /// Create an MLE estimator with Laplace smoothing.
    pub fn with_laplace(pseudocount: f64) -> Self {
        Self {
            use_laplace: true,
            pseudocount,
        }
    }

    /// Estimate parameters for a single variable from data.
    ///
    /// # Arguments
    ///
    /// * `variable` - Variable name
    /// * `cardinality` - Number of possible values
    /// * `data` - Observed assignments
    ///
    /// # Returns
    ///
    /// Estimated probability distribution P(variable)
    pub fn estimate_marginal(
        &self,
        variable: &str,
        cardinality: usize,
        data: &[Assignment],
    ) -> Result<ArrayD<f64>> {
        let pseudocount = if self.use_laplace {
            self.pseudocount
        } else {
            0.0
        };
        let mut counts = vec![pseudocount; cardinality];

        // Count occurrences
        for assignment in data {
            if let Some(&value) = assignment.get(variable) {
                if value < cardinality {
                    counts[value] += 1.0;
                }
            }
        }

        // Normalize to probabilities
        let total: f64 = counts.iter().sum();
        if total == 0.0 {
            return Err(PgmError::InvalidDistribution(
                "No data for variable".to_string(),
            ));
        }

        let probs: Vec<f64> = counts.iter().map(|&c| c / total).collect();

        ArrayD::from_shape_vec(vec![cardinality], probs)
            .map_err(|e| PgmError::InvalidGraph(format!("Array creation failed: {}", e)))
    }

    /// Estimate conditional probability table P(child | parents) from data.
    ///
    /// # Arguments
    ///
    /// * `child` - Child variable name
    /// * `parents` - Parent variable names
    /// * `cardinalities` - Cardinalities for [child, parent1, parent2, ...]
    /// * `data` - Observed assignments
    pub fn estimate_conditional(
        &self,
        child: &str,
        parents: &[String],
        cardinalities: &[usize],
        data: &[Assignment],
    ) -> Result<ArrayD<f64>> {
        if cardinalities.is_empty() {
            return Err(PgmError::InvalidGraph(
                "Cardinalities must not be empty".to_string(),
            ));
        }

        let pseudocount = if self.use_laplace {
            self.pseudocount
        } else {
            0.0
        };

        let child_card = cardinalities[0];
        let parent_cards = &cardinalities[1..];

        // Calculate total number of parent configurations
        let num_parent_configs: usize = parent_cards.iter().product();

        // Initialize counts: [parent_config][child_value]
        let mut counts = vec![vec![pseudocount; child_card]; num_parent_configs];

        // Count co-occurrences
        for assignment in data {
            if let Some(&child_val) = assignment.get(child) {
                // Compute parent configuration index
                let mut parent_config = 0;
                let mut multiplier = 1;

                for (i, parent) in parents.iter().enumerate() {
                    if let Some(&parent_val) = assignment.get(parent) {
                        parent_config += parent_val * multiplier;
                        multiplier *= parent_cards[i];
                    } else {
                        continue; // Skip if parent value missing
                    }
                }

                if parent_config < num_parent_configs && child_val < child_card {
                    counts[parent_config][child_val] += 1.0;
                }
            }
        }

        // Normalize each parent configuration
        let mut probs = Vec::new();
        for config_counts in counts {
            let total: f64 = config_counts.iter().sum();
            if total > 0.0 {
                for count in config_counts {
                    probs.push(count / total);
                }
            } else {
                // Uniform distribution if no data
                for _ in 0..child_card {
                    probs.push(1.0 / child_card as f64);
                }
            }
        }

        // Shape: [parent1_card, parent2_card, ..., child_card]
        ArrayD::from_shape_vec(cardinalities.to_vec(), probs)
            .map_err(|e| PgmError::InvalidGraph(format!("Array creation failed: {}", e)))
    }
}

impl Default for MaximumLikelihoodEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Bayesian parameter estimator with Dirichlet priors.
///
/// Uses conjugate Dirichlet priors for robust parameter estimation.
#[derive(Debug, Clone)]
pub struct BayesianEstimator {
    /// Dirichlet prior hyperparameters (equivalent sample size)
    pub prior_strength: f64,
}

impl BayesianEstimator {
    /// Create a new Bayesian estimator.
    ///
    /// # Arguments
    ///
    /// * `prior_strength` - Strength of the prior (equivalent sample size)
    pub fn new(prior_strength: f64) -> Self {
        Self { prior_strength }
    }

    /// Estimate parameters with Dirichlet prior.
    pub fn estimate_marginal(
        &self,
        variable: &str,
        cardinality: usize,
        data: &[Assignment],
    ) -> Result<ArrayD<f64>> {
        // Dirichlet(α, α, ..., α) prior
        let alpha = self.prior_strength / cardinality as f64;
        let mut counts = vec![alpha; cardinality];

        // Add data counts
        for assignment in data {
            if let Some(&value) = assignment.get(variable) {
                if value < cardinality {
                    counts[value] += 1.0;
                }
            }
        }

        // Posterior mean of Dirichlet
        let total: f64 = counts.iter().sum();
        let probs: Vec<f64> = counts.iter().map(|&c| c / total).collect();

        ArrayD::from_shape_vec(vec![cardinality], probs)
            .map_err(|e| PgmError::InvalidGraph(format!("Array creation failed: {}", e)))
    }
}

/// Baum-Welch algorithm for learning HMM parameters.
///
/// This is a specialized EM algorithm for Hidden Markov Models that learns:
/// - Initial state distribution
/// - Transition probabilities
/// - Emission probabilities
///
/// from sequences of observations (even when hidden states are not observed).
#[derive(Debug, Clone)]
pub struct BaumWelchLearner {
    /// Maximum number of EM iterations
    pub max_iterations: usize,
    /// Convergence tolerance (change in log-likelihood)
    pub tolerance: f64,
    /// Whether to print progress
    pub verbose: bool,
}

impl BaumWelchLearner {
    /// Create a new Baum-Welch learner.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            verbose: false,
        }
    }

    /// Create a verbose learner that prints progress.
    pub fn with_verbose(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            verbose: true,
        }
    }

    /// Learn HMM parameters from observation sequences.
    ///
    /// # Arguments
    ///
    /// * `hmm` - HMM model to update (will be modified in place)
    /// * `observation_sequences` - Multiple observation sequences
    ///
    /// # Returns
    ///
    /// Final log-likelihood
    #[allow(clippy::needless_range_loop)]
    pub fn learn(&self, hmm: &mut SimpleHMM, observation_sequences: &[Vec<usize>]) -> Result<f64> {
        let num_states = hmm.num_states;
        let num_observations = hmm.num_observations;

        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for iteration in 0..self.max_iterations {
            // E-step: Compute expected counts
            let mut initial_counts = vec![0.0; num_states];
            let mut transition_counts = vec![vec![0.0; num_states]; num_states];
            let mut emission_counts = vec![vec![0.0; num_observations]; num_states];

            let mut total_log_likelihood = 0.0;

            for sequence in observation_sequences {
                let (alpha, beta, log_likelihood) = self.forward_backward(hmm, sequence)?;
                total_log_likelihood += log_likelihood;

                let seq_len = sequence.len();

                // Expected counts for initial state
                for (s, count) in initial_counts.iter_mut().enumerate().take(num_states) {
                    let gamma_0 = self.compute_gamma(&alpha, &beta, 0, s, log_likelihood);
                    *count += gamma_0;
                }

                // Expected counts for transitions and emissions
                for t in 0..(seq_len - 1) {
                    for s1 in 0..num_states {
                        let gamma_t = self.compute_gamma(&alpha, &beta, t, s1, log_likelihood);

                        // Emission count
                        emission_counts[s1][sequence[t]] += gamma_t;

                        // Transition counts
                        for s2 in 0..num_states {
                            let xi = self.compute_xi(
                                hmm,
                                &alpha,
                                &beta,
                                t,
                                s1,
                                s2,
                                sequence[t + 1],
                                log_likelihood,
                            );
                            transition_counts[s1][s2] += xi;
                        }
                    }
                }

                // Last time step emission
                for (s, counts) in emission_counts.iter_mut().enumerate().take(num_states) {
                    let gamma_last =
                        self.compute_gamma(&alpha, &beta, seq_len - 1, s, log_likelihood);
                    counts[sequence[seq_len - 1]] += gamma_last;
                }
            }

            // M-step: Update parameters
            self.update_parameters(hmm, &initial_counts, &transition_counts, &emission_counts)?;

            // Check convergence
            let avg_log_likelihood = total_log_likelihood / observation_sequences.len() as f64;

            if self.verbose {
                println!(
                    "Iteration {}: log-likelihood = {:.4}",
                    iteration, avg_log_likelihood
                );
            }

            if (avg_log_likelihood - prev_log_likelihood).abs() < self.tolerance {
                if self.verbose {
                    println!("Converged after {} iterations", iteration + 1);
                }
                return Ok(avg_log_likelihood);
            }

            prev_log_likelihood = avg_log_likelihood;
        }

        if self.verbose {
            println!("Maximum iterations reached");
        }

        Ok(prev_log_likelihood)
    }

    /// Forward-backward algorithm.
    #[allow(clippy::type_complexity, clippy::needless_range_loop)]
    fn forward_backward(
        &self,
        hmm: &SimpleHMM,
        sequence: &[usize],
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, f64)> {
        let num_states = hmm.num_states;
        let seq_len = sequence.len();

        // Forward pass
        let mut alpha = vec![vec![0.0; num_states]; seq_len];

        // Initialize
        for s in 0..num_states {
            alpha[0][s] =
                hmm.initial_distribution[[s]] * hmm.emission_probabilities[[s, sequence[0]]];
        }

        // Forward recursion
        for t in 1..seq_len {
            for s2 in 0..num_states {
                let mut sum = 0.0;
                for s1 in 0..num_states {
                    sum += alpha[t - 1][s1] * hmm.transition_probabilities[[s1, s2]];
                }
                alpha[t][s2] = sum * hmm.emission_probabilities[[s2, sequence[t]]];
            }
        }

        // Backward pass
        let mut beta = vec![vec![0.0; num_states]; seq_len];

        // Initialize
        for s in 0..num_states {
            beta[seq_len - 1][s] = 1.0;
        }

        // Backward recursion
        for t in (0..(seq_len - 1)).rev() {
            for s1 in 0..num_states {
                let mut sum = 0.0;
                for s2 in 0..num_states {
                    sum += hmm.transition_probabilities[[s1, s2]]
                        * hmm.emission_probabilities[[s2, sequence[t + 1]]]
                        * beta[t + 1][s2];
                }
                beta[t][s1] = sum;
            }
        }

        // Compute log-likelihood
        let log_likelihood: f64 = alpha[seq_len - 1].iter().sum::<f64>().ln();

        Ok((alpha, beta, log_likelihood))
    }

    /// Compute gamma (state occupation probability).
    fn compute_gamma(
        &self,
        alpha: &[Vec<f64>],
        beta: &[Vec<f64>],
        t: usize,
        s: usize,
        log_likelihood: f64,
    ) -> f64 {
        (alpha[t][s] * beta[t][s]) / log_likelihood.exp()
    }

    /// Compute xi (state transition probability).
    #[allow(clippy::too_many_arguments)]
    fn compute_xi(
        &self,
        hmm: &SimpleHMM,
        alpha: &[Vec<f64>],
        beta: &[Vec<f64>],
        t: usize,
        s1: usize,
        s2: usize,
        next_obs: usize,
        log_likelihood: f64,
    ) -> f64 {
        let numerator = alpha[t][s1]
            * hmm.transition_probabilities[[s1, s2]]
            * hmm.emission_probabilities[[s2, next_obs]]
            * beta[t + 1][s2];

        numerator / log_likelihood.exp()
    }

    /// Update HMM parameters (M-step).
    fn update_parameters(
        &self,
        hmm: &mut SimpleHMM,
        initial_counts: &[f64],
        transition_counts: &[Vec<f64>],
        emission_counts: &[Vec<f64>],
    ) -> Result<()> {
        let num_states = hmm.num_states;
        let num_observations = hmm.num_observations;

        // Update initial distribution
        let initial_sum: f64 = initial_counts.iter().sum();
        if initial_sum > 0.0 {
            for (s, &count) in initial_counts.iter().enumerate().take(num_states) {
                hmm.initial_distribution[[s]] = count / initial_sum;
            }
        }

        // Update transition probabilities
        for (s1, trans_counts) in transition_counts.iter().enumerate().take(num_states) {
            let trans_sum: f64 = trans_counts.iter().sum();
            if trans_sum > 0.0 {
                for (s2, &count) in trans_counts.iter().enumerate().take(num_states) {
                    hmm.transition_probabilities[[s1, s2]] = count / trans_sum;
                }
            }
        }

        // Update emission probabilities
        for (s, emis_counts) in emission_counts.iter().enumerate().take(num_states) {
            let emission_sum: f64 = emis_counts.iter().sum();
            if emission_sum > 0.0 {
                for (o, &count) in emis_counts.iter().enumerate().take(num_observations) {
                    hmm.emission_probabilities[[s, o]] = count / emission_sum;
                }
            }
        }

        Ok(())
    }
}

/// Utilities for parameter learning.
pub mod utils {
    use super::*;

    /// Count variable occurrences in data.
    pub fn count_occurrences(variable: &str, data: &[Assignment]) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();

        for assignment in data {
            if let Some(&value) = assignment.get(variable) {
                *counts.entry(value).or_insert(0) += 1;
            }
        }

        counts
    }

    /// Count co-occurrences of two variables.
    pub fn count_joint_occurrences(
        var1: &str,
        var2: &str,
        data: &[Assignment],
    ) -> HashMap<(usize, usize), usize> {
        let mut counts = HashMap::new();

        for assignment in data {
            if let (Some(&v1), Some(&v2)) = (assignment.get(var1), assignment.get(var2)) {
                *counts.entry((v1, v2)).or_insert(0) += 1;
            }
        }

        counts
    }

    /// Compute empirical distribution from counts.
    pub fn counts_to_distribution(counts: &HashMap<usize, usize>, cardinality: usize) -> Vec<f64> {
        let total: usize = counts.values().sum();
        let mut probs = vec![0.0; cardinality];

        for (&value, &count) in counts {
            if value < cardinality && total > 0 {
                probs[value] = count as f64 / total as f64;
            }
        }

        probs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mle_marginal() {
        let estimator = MaximumLikelihoodEstimator::new();

        let mut data = Vec::new();
        for _ in 0..7 {
            let mut assignment = HashMap::new();
            assignment.insert("X".to_string(), 0);
            data.push(assignment);
        }
        for _ in 0..3 {
            let mut assignment = HashMap::new();
            assignment.insert("X".to_string(), 1);
            data.push(assignment);
        }

        let probs = estimator.estimate_marginal("X", 2, &data).unwrap();

        assert!((probs[[0]] - 0.7).abs() < 1e-6);
        assert!((probs[[1]] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_mle_with_laplace() {
        let estimator = MaximumLikelihoodEstimator::with_laplace(1.0);

        let mut data = Vec::new();
        for _ in 0..8 {
            let mut assignment = HashMap::new();
            assignment.insert("X".to_string(), 0);
            data.push(assignment);
        }
        // No observations of X=1

        let probs = estimator.estimate_marginal("X", 2, &data).unwrap();

        // With Laplace: (8+1)/(8+1+0+1) = 9/10 = 0.9
        assert!((probs[[0]] - 0.9).abs() < 1e-6);
        assert!((probs[[1]] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_bayesian_estimator() {
        let estimator = BayesianEstimator::new(2.0);

        let mut data = Vec::new();
        for _ in 0..8 {
            let mut assignment = HashMap::new();
            assignment.insert("X".to_string(), 0);
            data.push(assignment);
        }

        let probs = estimator.estimate_marginal("X", 2, &data).unwrap();

        // Prior: Dirichlet(1, 1), Data: 8, 0
        // Posterior: (8+1, 0+1) / (8+1+0+1) = (9, 1) / 10
        assert!((probs[[0]] - 0.9).abs() < 1e-6);
        assert!((probs[[1]] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_count_occurrences() {
        let mut data = Vec::new();
        for i in 0..10 {
            let mut assignment = HashMap::new();
            assignment.insert("X".to_string(), i % 3);
            data.push(assignment);
        }

        let counts = utils::count_occurrences("X", &data);

        assert_eq!(counts.get(&0), Some(&4)); // 0, 3, 6, 9
        assert_eq!(counts.get(&1), Some(&3)); // 1, 4, 7
        assert_eq!(counts.get(&2), Some(&3)); // 2, 5, 8
    }

    #[test]
    fn test_counts_to_distribution() {
        let mut counts = HashMap::new();
        counts.insert(0, 7);
        counts.insert(1, 3);

        let probs = utils::counts_to_distribution(&counts, 2);

        assert!((probs[0] - 0.7).abs() < 1e-6);
        assert!((probs[1] - 0.3).abs() < 1e-6);
    }
}
