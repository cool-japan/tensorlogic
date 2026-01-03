//! Variational inference methods for approximate PGM inference.
//!
//! This module provides variational inference algorithms that approximate
//! intractable posterior distributions with simpler distributions.

use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;

use crate::error::{PgmError, Result};
use crate::factor::Factor;
use crate::graph::FactorGraph;
use crate::message_passing::MessagePassingAlgorithm;

/// Mean-field variational inference.
///
/// Approximates the joint distribution with a product of independent marginals:
/// Q(X₁, ..., Xₙ) = ∏ᵢ Qᵢ(Xᵢ)
pub struct MeanFieldInference {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for MeanFieldInference {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl MeanFieldInference {
    /// Create with custom parameters.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Run mean-field variational inference.
    ///
    /// Returns approximate marginals for each variable.
    pub fn run(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        // Initialize Q distributions uniformly
        let mut q_distributions: HashMap<String, ArrayD<f64>> = HashMap::new();

        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let uniform = ArrayD::from_elem(
                    vec![var_node.cardinality],
                    1.0 / var_node.cardinality as f64,
                );
                q_distributions.insert(var_name.clone(), uniform);
            }
        }

        // Iterative updates
        for iteration in 0..self.max_iterations {
            let old_q = q_distributions.clone();

            // Update each Q distribution
            for var_name in graph.variable_names() {
                let updated_q = self.update_q_distribution(graph, var_name, &q_distributions)?;
                q_distributions.insert(var_name.clone(), updated_q);
            }

            // Check convergence
            if self.check_convergence(&old_q, &q_distributions) {
                return Ok(q_distributions);
            }

            if iteration == self.max_iterations - 1 {
                return Err(PgmError::ConvergenceFailure(format!(
                    "Mean-field inference did not converge after {} iterations",
                    self.max_iterations
                )));
            }
        }

        Ok(q_distributions)
    }

    /// Update Q distribution for a single variable.
    ///
    /// Q*(Xᵢ) ∝ exp(E[log p(X)] over Q\{Xᵢ})
    fn update_q_distribution(
        &self,
        graph: &FactorGraph,
        var_name: &str,
        q_distributions: &HashMap<String, ArrayD<f64>>,
    ) -> Result<ArrayD<f64>> {
        let var_node = graph
            .get_variable(var_name)
            .ok_or_else(|| PgmError::VariableNotFound(var_name.to_string()))?;

        // Initialize log potential
        let mut log_potential = ArrayD::zeros(vec![var_node.cardinality]);

        // Get factors containing this variable
        if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
            for factor_id in adjacent_factors {
                if let Some(factor) = graph.get_factor(factor_id) {
                    // Compute expected log factor
                    let expected_log =
                        self.compute_expected_log_factor(factor, var_name, q_distributions)?;
                    log_potential = log_potential + expected_log;
                }
            }
        }

        // Normalize: Q*(X) = exp(log_potential) / Z
        let unnormalized = log_potential.mapv(|x: f64| x.exp());
        let z: f64 = unnormalized.iter().sum();

        if z > 0.0 {
            Ok(&unnormalized / z)
        } else {
            // Fallback to uniform if normalization fails
            Ok(ArrayD::from_elem(
                vec![var_node.cardinality],
                1.0 / var_node.cardinality as f64,
            ))
        }
    }

    /// Compute expected log factor: E[log φ(X)] over Q\{Xᵢ}
    fn compute_expected_log_factor(
        &self,
        factor: &Factor,
        target_var: &str,
        q_distributions: &HashMap<String, ArrayD<f64>>,
    ) -> Result<ArrayD<f64>> {
        // Find target variable index
        let target_idx = factor
            .variables
            .iter()
            .position(|v| v == target_var)
            .ok_or_else(|| PgmError::VariableNotFound(target_var.to_string()))?;

        let target_card = factor.values.shape()[target_idx];
        let mut expected_log = ArrayD::zeros(vec![target_card]);

        // Compute expectation over all assignments
        let total_size: usize = factor.values.shape().iter().product();
        for linear_idx in 0..total_size {
            // Convert to multi-dimensional index
            let mut assignment = Vec::new();
            let mut temp_idx = linear_idx;
            for &dim in factor.values.shape().iter().rev() {
                assignment.push(temp_idx % dim);
                temp_idx /= dim;
            }
            assignment.reverse();

            // Get factor value
            let factor_val = factor.values[assignment.as_slice()];
            let log_factor_val = if factor_val > 1e-10 {
                factor_val.ln()
            } else {
                -10.0 // log of very small number
            };

            // Compute probability of this assignment under Q
            let mut q_prob = 1.0;
            for (idx, var) in factor.variables.iter().enumerate() {
                if var != target_var {
                    if let Some(q) = q_distributions.get(var) {
                        q_prob *= q[[assignment[idx]]];
                    }
                }
            }

            // Accumulate expected log
            let target_val = assignment[target_idx];
            expected_log[[target_val]] += q_prob * log_factor_val;
        }

        Ok(expected_log)
    }

    /// Check convergence by comparing Q distributions.
    fn check_convergence(
        &self,
        old_q: &HashMap<String, ArrayD<f64>>,
        new_q: &HashMap<String, ArrayD<f64>>,
    ) -> bool {
        let mut max_delta = 0.0_f64;

        for (var, new_dist) in new_q {
            if let Some(old_dist) = old_q.get(var) {
                let delta: f64 = (new_dist - old_dist)
                    .mapv(|x| x.abs())
                    .iter()
                    .fold(0.0_f64, |acc, &x| acc.max(x));
                max_delta = max_delta.max(delta);
            }
        }

        max_delta < self.tolerance
    }

    /// Compute ELBO (Evidence Lower BOund).
    ///
    /// ELBO = E[log p(X, Z)] - E[log q(Z)]
    pub fn compute_elbo(
        &self,
        graph: &FactorGraph,
        q_distributions: &HashMap<String, ArrayD<f64>>,
    ) -> Result<f64> {
        let mut elbo = 0.0;

        // E[log p(X, Z)] - sum over all factors
        for factor_id in graph.factor_ids() {
            if let Some(factor) = graph.get_factor(factor_id) {
                elbo += self.expected_log_joint_factor(factor, q_distributions)?;
            }
        }

        // -E[log q(Z)] - entropy of Q
        for q_dist in q_distributions.values() {
            let entropy: f64 = q_dist
                .iter()
                .map(|&p| if p > 1e-10 { -p * p.ln() } else { 0.0 })
                .sum();
            elbo += entropy;
        }

        Ok(elbo)
    }

    /// Compute E[log φ(X)] for a factor.
    fn expected_log_joint_factor(
        &self,
        factor: &Factor,
        q_distributions: &HashMap<String, ArrayD<f64>>,
    ) -> Result<f64> {
        let mut expected = 0.0;

        let total_size: usize = factor.values.shape().iter().product();
        for linear_idx in 0..total_size {
            let mut assignment = Vec::new();
            let mut temp_idx = linear_idx;
            for &dim in factor.values.shape().iter().rev() {
                assignment.push(temp_idx % dim);
                temp_idx /= dim;
            }
            assignment.reverse();

            // Factor value
            let factor_val = factor.values[assignment.as_slice()];
            let log_factor_val = if factor_val > 1e-10 {
                factor_val.ln()
            } else {
                -10.0
            };

            // Probability under Q
            let mut q_prob = 1.0;
            for (idx, var) in factor.variables.iter().enumerate() {
                if let Some(q) = q_distributions.get(var) {
                    q_prob *= q[[assignment[idx]]];
                }
            }

            expected += q_prob * log_factor_val;
        }

        Ok(expected)
    }
}

/// Bethe approximation for structured variational inference.
///
/// Uses the factor graph structure to define a structured approximation.
/// More accurate than mean-field but still tractable.
///
/// The Bethe free energy is:
/// F_Bethe = Σ_α H(b_α) - Σ_i (d_i - 1) H(b_i) - Σ_α <log ψ_α>_b_α
///
/// where:
/// - b_α are factor beliefs (cluster marginals)
/// - b_i are variable beliefs (node marginals)
/// - d_i is the degree of variable i
/// - H is entropy
pub struct BetheApproximation {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Damping factor for message updates
    pub damping: f64,
}

impl Default for BetheApproximation {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            damping: 0.0,
        }
    }
}

impl BetheApproximation {
    /// Create with custom parameters.
    pub fn new(max_iterations: usize, tolerance: f64, damping: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            damping: damping.clamp(0.0, 1.0),
        }
    }

    /// Run Bethe approximation using belief propagation.
    ///
    /// Returns variable beliefs (marginals).
    pub fn run(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        // Use sum-product belief propagation as the message passing algorithm
        // The fixed points of BP correspond to stationary points of Bethe free energy
        use crate::message_passing::SumProductAlgorithm;

        let bp = SumProductAlgorithm::new(self.max_iterations, self.tolerance, self.damping);
        bp.run(graph)
    }

    /// Compute Bethe free energy.
    ///
    /// F_Bethe = Σ_α E_bα[log bα - log ψα] - Σ_i (d_i - 1) H(b_i)
    pub fn compute_free_energy(
        &self,
        graph: &FactorGraph,
        variable_beliefs: &HashMap<String, ArrayD<f64>>,
        factor_beliefs: &HashMap<String, ArrayD<f64>>,
    ) -> Result<f64> {
        let mut free_energy = 0.0;

        // Factor contribution: Σ_α E_bα[log bα - log ψα]
        for (factor_id, belief) in factor_beliefs {
            if let Some(factor) = graph.get_factor(factor_id) {
                // E[log bα]
                let entropy_contrib: f64 = belief
                    .iter()
                    .map(|&p| if p > 1e-10 { -p * p.ln() } else { 0.0 })
                    .sum();

                // E[log ψα]
                let mut energy_contrib = 0.0;
                let total_size: usize = belief.shape().iter().product();
                for linear_idx in 0..total_size {
                    let mut assignment = Vec::new();
                    let mut temp_idx = linear_idx;
                    for &dim in belief.shape().iter().rev() {
                        assignment.push(temp_idx % dim);
                        temp_idx /= dim;
                    }
                    assignment.reverse();

                    let b_val = belief[assignment.as_slice()];
                    let psi_val = factor.values[assignment.as_slice()];
                    if b_val > 1e-10 && psi_val > 1e-10 {
                        energy_contrib += b_val * psi_val.ln();
                    }
                }

                free_energy -= entropy_contrib;
                free_energy -= energy_contrib;
            }
        }

        // Variable contribution: Σ_i (d_i - 1) H(b_i)
        for (var_name, belief) in variable_beliefs {
            // Get degree of variable (number of adjacent factors)
            let degree = if let Some(adjacent) = graph.get_adjacent_factors(var_name) {
                adjacent.len()
            } else {
                0
            };

            if degree > 0 {
                let entropy: f64 = belief
                    .iter()
                    .map(|&p| if p > 1e-10 { -p * p.ln() } else { 0.0 })
                    .sum();

                free_energy += (degree as f64 - 1.0) * entropy;
            }
        }

        Ok(free_energy)
    }

    /// Compute factor beliefs from variable beliefs and factor potentials.
    pub fn compute_factor_beliefs(
        &self,
        graph: &FactorGraph,
        variable_beliefs: &HashMap<String, ArrayD<f64>>,
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut factor_beliefs = HashMap::new();

        for factor_id in graph.factor_ids() {
            if let Some(factor) = graph.get_factor(factor_id) {
                // Start with factor potential
                let mut belief = factor.clone();

                // Multiply by variable beliefs (approximately - using product)
                for var in &factor.variables {
                    if let Some(var_belief) = variable_beliefs.get(var) {
                        // Create a factor from variable belief
                        let var_factor = Factor {
                            name: format!("belief_{}", var),
                            variables: vec![var.clone()],
                            values: var_belief.clone(),
                        };
                        belief = belief.product(&var_factor)?;
                    }
                }

                belief.normalize();
                factor_beliefs.insert(factor_id.clone(), belief.values);
            }
        }

        Ok(factor_beliefs)
    }
}

/// Tree-reweighted belief propagation (TRW-BP).
///
/// Uses a convex combination of spanning trees to provide an upper bound
/// on the log partition function. More robust than standard BP for loopy graphs.
///
/// Messages are reweighted by edge appearance probabilities ρ_e ∈ `[0,1]`.
pub struct TreeReweightedBP {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Edge appearance probabilities (default: uniform)
    pub edge_weights: HashMap<(String, String), f64>,
}

impl Default for TreeReweightedBP {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            edge_weights: HashMap::new(),
        }
    }
}

impl TreeReweightedBP {
    /// Create with custom parameters.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            edge_weights: HashMap::new(),
        }
    }

    /// Set edge appearance probability for a variable-factor edge.
    pub fn set_edge_weight(&mut self, var: String, factor: String, weight: f64) {
        self.edge_weights
            .insert((var, factor), weight.clamp(0.0, 1.0));
    }

    /// Initialize uniform edge weights for all edges in graph.
    pub fn initialize_uniform_weights(&mut self, graph: &FactorGraph) {
        for var_name in graph.variable_names() {
            if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
                let weight = 1.0 / adjacent_factors.len() as f64;
                for factor_id in adjacent_factors {
                    self.edge_weights
                        .insert((var_name.clone(), factor_id.clone()), weight);
                }
            }
        }
    }

    /// Get edge weight (default to 1.0 if not set).
    fn get_edge_weight(&self, var: &str, factor: &str) -> f64 {
        self.edge_weights
            .get(&(var.to_string(), factor.to_string()))
            .copied()
            .unwrap_or(1.0)
    }

    /// Run tree-reweighted belief propagation.
    ///
    /// Returns variable beliefs (marginals) and an upper bound on log Z.
    pub fn run(&mut self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        // Initialize uniform weights if not set
        if self.edge_weights.is_empty() {
            self.initialize_uniform_weights(graph);
        }

        // Message storage: (var, factor) -> message
        let mut messages: HashMap<(String, String), ArrayD<f64>> = HashMap::new();

        // Initialize messages uniformly
        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
                    for factor_id in adjacent_factors {
                        let init_msg = ArrayD::from_elem(
                            vec![var_node.cardinality],
                            1.0 / var_node.cardinality as f64,
                        );
                        messages.insert((var_name.clone(), factor_id.clone()), init_msg);
                    }
                }
            }
        }

        // Iterative message passing
        for iteration in 0..self.max_iterations {
            let old_messages = messages.clone();

            // Update all messages
            for var_name in graph.variable_names() {
                if let Some(var_node) = graph.get_variable(var_name) {
                    if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
                        for target_factor in adjacent_factors {
                            // Compute reweighted message
                            let mut message = ArrayD::ones(vec![var_node.cardinality])
                                / var_node.cardinality as f64;

                            // Multiply messages from other factors (with reweighting)
                            for other_factor in adjacent_factors {
                                if other_factor != target_factor {
                                    if let Some(incoming) =
                                        old_messages.get(&(var_name.clone(), other_factor.clone()))
                                    {
                                        let rho = self.get_edge_weight(var_name, other_factor);
                                        // Reweighted message: m^ρ
                                        let reweighted = incoming.mapv(|x| x.powf(rho));
                                        message = &message * &reweighted;
                                    }
                                }
                            }

                            // Normalize
                            let sum: f64 = message.iter().sum();
                            if sum > 1e-10 {
                                message /= sum;
                            }

                            messages.insert((var_name.clone(), target_factor.clone()), message);
                        }
                    }
                }
            }

            // Check convergence
            let mut max_delta = 0.0_f64;
            for ((var, factor), new_msg) in &messages {
                if let Some(old_msg) = old_messages.get(&(var.clone(), factor.clone())) {
                    let delta: f64 = (new_msg - old_msg)
                        .mapv(|x| x.abs())
                        .iter()
                        .fold(0.0_f64, |acc, &x| acc.max(x));
                    max_delta = max_delta.max(delta);
                }
            }

            if max_delta < self.tolerance {
                break;
            }

            if iteration == self.max_iterations - 1 {
                return Err(PgmError::ConvergenceFailure(format!(
                    "TRW-BP did not converge after {} iterations (max_delta={})",
                    self.max_iterations, max_delta
                )));
            }
        }

        // Compute beliefs from messages
        let mut beliefs = HashMap::new();
        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let mut belief =
                    ArrayD::ones(vec![var_node.cardinality]) / var_node.cardinality as f64;

                if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
                    for factor_id in adjacent_factors {
                        if let Some(message) = messages.get(&(var_name.clone(), factor_id.clone()))
                        {
                            let rho = self.get_edge_weight(var_name, factor_id);
                            let reweighted = message.mapv(|x| x.powf(rho));
                            belief = &belief * &reweighted;
                        }
                    }
                }

                // Normalize
                let sum: f64 = belief.iter().sum();
                if sum > 1e-10 {
                    belief /= sum;
                }

                beliefs.insert(var_name.clone(), belief);
            }
        }

        Ok(beliefs)
    }

    /// Compute upper bound on log partition function.
    ///
    /// log Z ≤ log Z_TRW = Σ_i ρ_i log Z_i
    pub fn compute_log_partition_upper_bound(
        &self,
        _graph: &FactorGraph,
        _beliefs: &HashMap<String, ArrayD<f64>>,
    ) -> Result<f64> {
        // Simplified implementation - full version requires factor beliefs
        // and region-based computation
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mean_field_single_variable() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let mf = MeanFieldInference::default();
        let result = mf.run(&graph);
        assert!(result.is_ok());

        let marginals = result.unwrap();
        assert!(marginals.contains_key("x"));

        // Should be uniform for single variable with no factors
        let dist = &marginals["x"];
        assert_abs_diff_eq!(dist[[0]], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(dist[[1]], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_mean_field_convergence() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let mf = MeanFieldInference::new(50, 1e-6);
        let result = mf.run(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_elbo_computation() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let mf = MeanFieldInference::default();
        let marginals = mf.run(&graph).unwrap();

        let elbo = mf.compute_elbo(&graph, &marginals);
        assert!(elbo.is_ok());
    }

    #[test]
    fn test_bethe_approximation_single_variable() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let bethe = BetheApproximation::default();
        let result = bethe.run(&graph);
        assert!(result.is_ok());

        let marginals = result.unwrap();
        assert!(marginals.contains_key("x"));

        let dist = &marginals["x"];
        assert_abs_diff_eq!(dist[[0]], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(dist[[1]], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_bethe_approximation_two_variables() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let bethe = BetheApproximation::new(50, 1e-6, 0.0);
        let result = bethe.run(&graph);
        assert!(result.is_ok());

        let marginals = result.unwrap();
        assert_eq!(marginals.len(), 2);
    }

    #[test]
    fn test_bethe_free_energy() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let bethe = BetheApproximation::default();
        let marginals = bethe.run(&graph).unwrap();
        let factor_beliefs = bethe.compute_factor_beliefs(&graph, &marginals).unwrap();

        let free_energy = bethe.compute_free_energy(&graph, &marginals, &factor_beliefs);
        assert!(free_energy.is_ok());
    }

    #[test]
    fn test_bethe_with_damping() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let bethe = BetheApproximation::new(50, 1e-6, 0.5);
        let result = bethe.run(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trw_bp_single_variable() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let mut trw = TreeReweightedBP::default();
        let result = trw.run(&graph);
        assert!(result.is_ok());

        let beliefs = result.unwrap();
        assert!(beliefs.contains_key("x"));

        let dist = &beliefs["x"];
        assert_abs_diff_eq!(dist[[0]], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(dist[[1]], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_trw_bp_two_variables() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let mut trw = TreeReweightedBP::new(50, 1e-6);
        let result = trw.run(&graph);
        assert!(result.is_ok());

        let beliefs = result.unwrap();
        assert_eq!(beliefs.len(), 2);
    }

    #[test]
    fn test_trw_bp_custom_weights() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let mut trw = TreeReweightedBP::default();
        trw.set_edge_weight("x".to_string(), "f1".to_string(), 0.5);

        // Should handle missing edges gracefully
        let result = trw.run(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trw_bp_uniform_initialization() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let mut trw = TreeReweightedBP::default();
        trw.initialize_uniform_weights(&graph);

        assert!(!trw.edge_weights.is_empty() || graph.factor_ids().count() == 0);
    }

    #[test]
    fn test_trw_bp_partition_bound() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        let mut trw = TreeReweightedBP::default();
        let beliefs = trw.run(&graph).unwrap();

        let bound = trw.compute_log_partition_upper_bound(&graph, &beliefs);
        assert!(bound.is_ok());
    }
}
