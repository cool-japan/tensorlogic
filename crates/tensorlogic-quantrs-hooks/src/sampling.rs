//! Sampling-based inference methods for PGM.
//!
//! This module provides MCMC and other sampling algorithms for approximate inference.

use scirs2_core::ndarray::ArrayD;
use scirs2_core::random::{thread_rng, Rng};
use std::collections::HashMap;

use crate::error::{PgmError, Result};
use crate::graph::FactorGraph;

/// Assignment of values to variables.
pub type Assignment = HashMap<String, usize>;

/// Gibbs sampling for approximate inference.
///
/// Uses Markov Chain Monte Carlo to sample from the joint distribution.
pub struct GibbsSampler {
    /// Number of burn-in samples to discard
    pub burn_in: usize,
    /// Number of samples to collect
    pub num_samples: usize,
    /// Thinning interval (keep every N-th sample)
    pub thinning: usize,
}

impl Default for GibbsSampler {
    fn default() -> Self {
        Self {
            burn_in: 100,
            num_samples: 1000,
            thinning: 1,
        }
    }
}

impl GibbsSampler {
    /// Create with custom parameters.
    pub fn new(burn_in: usize, num_samples: usize, thinning: usize) -> Self {
        Self {
            burn_in,
            num_samples,
            thinning,
        }
    }

    /// Run Gibbs sampling to approximate marginals.
    pub fn run(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        // Initialize random assignment
        let mut current_assignment = self.initialize_assignment(graph)?;

        // Burn-in phase
        for _ in 0..self.burn_in {
            self.gibbs_step(graph, &mut current_assignment)?;
        }

        // Collect samples
        let mut samples = Vec::new();
        for i in 0..self.num_samples * self.thinning {
            self.gibbs_step(graph, &mut current_assignment)?;

            // Keep sample if it's at thinning interval
            if i % self.thinning == 0 {
                samples.push(current_assignment.clone());
            }
        }

        // Compute empirical marginals from samples
        self.compute_empirical_marginals(graph, &samples)
    }

    /// Initialize random assignment for all variables.
    fn initialize_assignment(&self, graph: &FactorGraph) -> Result<Assignment> {
        let mut rng = thread_rng();
        let mut assignment = Assignment::new();

        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let random_value = rng.gen_range(0..var_node.cardinality);
                assignment.insert(var_name.clone(), random_value);
            }
        }

        Ok(assignment)
    }

    /// Perform one Gibbs sampling step (resample all variables).
    fn gibbs_step(&self, graph: &FactorGraph, assignment: &mut Assignment) -> Result<()> {
        // Resample each variable conditioned on others
        for var_name in graph.variable_names() {
            self.resample_variable(graph, var_name, assignment)?;
        }

        Ok(())
    }

    /// Resample a single variable given current assignment of others.
    fn resample_variable(
        &self,
        graph: &FactorGraph,
        var_name: &str,
        assignment: &mut Assignment,
    ) -> Result<()> {
        let var_node = graph
            .get_variable(var_name)
            .ok_or_else(|| PgmError::VariableNotFound(var_name.to_string()))?;

        // Compute conditional distribution P(X | others)
        let mut conditional_probs = vec![0.0; var_node.cardinality];

        for (value, prob) in conditional_probs
            .iter_mut()
            .enumerate()
            .take(var_node.cardinality)
        {
            assignment.insert(var_name.to_string(), value);
            *prob = self.compute_joint_probability(graph, assignment)?;
        }

        // Normalize
        let sum: f64 = conditional_probs.iter().sum();
        if sum > 0.0 {
            for prob in &mut conditional_probs {
                *prob /= sum;
            }
        } else {
            // Fallback to uniform if all zero
            let uniform_prob = 1.0 / var_node.cardinality as f64;
            conditional_probs = vec![uniform_prob; var_node.cardinality];
        }

        // Sample from conditional distribution
        let sampled_value = self.sample_from_distribution(&conditional_probs);
        assignment.insert(var_name.to_string(), sampled_value);

        Ok(())
    }

    /// Compute joint probability for a full assignment.
    fn compute_joint_probability(
        &self,
        graph: &FactorGraph,
        assignment: &Assignment,
    ) -> Result<f64> {
        let mut prob = 1.0;

        for factor_id in graph.factor_ids() {
            if let Some(factor) = graph.get_factor(factor_id) {
                // Build index for this factor
                let mut indices = Vec::new();
                for var in &factor.variables {
                    if let Some(&value) = assignment.get(var) {
                        indices.push(value);
                    } else {
                        return Err(PgmError::VariableNotFound(var.clone()));
                    }
                }

                prob *= factor.values[indices.as_slice()];
            }
        }

        Ok(prob)
    }

    /// Sample from a discrete probability distribution.
    fn sample_from_distribution(&self, probs: &[f64]) -> usize {
        let mut rng = thread_rng();
        let u: f64 = rng.random();

        let mut cumulative = 0.0;
        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if u < cumulative {
                return idx;
            }
        }

        // Fallback to last index
        probs.len() - 1
    }

    /// Compute empirical marginals from collected samples.
    fn compute_empirical_marginals(
        &self,
        graph: &FactorGraph,
        samples: &[Assignment],
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut marginals = HashMap::new();

        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let mut counts = vec![0; var_node.cardinality];

                // Count occurrences
                for sample in samples {
                    if let Some(&value) = sample.get(var_name) {
                        counts[value] += 1;
                    }
                }

                // Normalize to probabilities
                let total = samples.len() as f64;
                let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();

                marginals.insert(
                    var_name.clone(),
                    ArrayD::from_shape_vec(vec![var_node.cardinality], probs)?,
                );
            }
        }

        Ok(marginals)
    }

    /// Get all samples (for analysis).
    pub fn get_samples(&self, graph: &FactorGraph) -> Result<Vec<Assignment>> {
        let mut current_assignment = self.initialize_assignment(graph)?;

        // Burn-in
        for _ in 0..self.burn_in {
            self.gibbs_step(graph, &mut current_assignment)?;
        }

        // Collect samples
        let mut samples = Vec::new();
        for i in 0..self.num_samples * self.thinning {
            self.gibbs_step(graph, &mut current_assignment)?;

            if i % self.thinning == 0 {
                samples.push(current_assignment.clone());
            }
        }

        Ok(samples)
    }
}

impl From<scirs2_core::ndarray::ShapeError> for PgmError {
    fn from(err: scirs2_core::ndarray::ShapeError) -> Self {
        PgmError::InvalidDistribution(format!("Shape error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gibbs_sampler_single_variable() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let sampler = GibbsSampler::new(10, 100, 1);
        let result = sampler.run(&graph);
        assert!(result.is_ok());

        let marginals = result.unwrap();
        assert!(marginals.contains_key("x"));

        // Should be approximately uniform
        let dist = &marginals["x"];
        let sum: f64 = dist.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gibbs_sampler_multiple_variables() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let sampler = GibbsSampler::new(20, 100, 1);
        let result = sampler.run(&graph);
        assert!(result.is_ok());

        let marginals = result.unwrap();
        assert_eq!(marginals.len(), 2);
    }

    #[test]
    fn test_sample_collection() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let sampler = GibbsSampler::new(10, 50, 1);
        let samples = sampler.get_samples(&graph);
        assert!(samples.is_ok());
        assert_eq!(samples.unwrap().len(), 50);
    }

    #[test]
    fn test_gibbs_with_thinning() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let sampler = GibbsSampler::new(10, 50, 5);
        let samples = sampler.get_samples(&graph);
        assert!(samples.is_ok());
        assert_eq!(samples.unwrap().len(), 50);
    }
}
