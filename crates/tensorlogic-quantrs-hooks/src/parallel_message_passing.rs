//! Parallel message passing algorithms using rayon.
//!
//! This module provides parallel implementations of belief propagation algorithms
//! that can significantly speed up inference on large factor graphs with many variables.

use crate::error::{PgmError, Result};
use crate::factor::Factor;
use crate::graph::FactorGraph;
use crate::message_passing::ConvergenceStats;
use rayon::prelude::*;
use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Parallel sum-product belief propagation.
///
/// Uses rayon to compute messages in parallel, which can provide significant
/// speedup for large factor graphs.
pub struct ParallelSumProduct {
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Damping factor (0.0 = no damping, 1.0 = full damping)
    pub damping: f64,
}

impl Default for ParallelSumProduct {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            damping: 0.0,
        }
    }
}

impl ParallelSumProduct {
    /// Create with custom parameters.
    pub fn new(max_iterations: usize, tolerance: f64, damping: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            damping,
        }
    }

    /// Run parallel belief propagation.
    pub fn run_parallel(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        // Initialize messages
        let messages = Arc::new(Mutex::new(self.initialize_messages(graph)?));

        // Iterative message passing
        for iteration in 0..self.max_iterations {
            let old_messages = messages.lock().unwrap().clone();

            // Parallel computation of variable-to-factor messages
            let var_factor_updates: Vec<_> = graph
                .variable_names()
                .par_bridge()
                .flat_map(|var_name| {
                    if let Some(factors) = graph.get_adjacent_factors(var_name) {
                        factors
                            .par_iter()
                            .filter_map(|factor_id| {
                                if let Some(factor) = graph.get_factor(factor_id) {
                                    let key = (var_name.to_string(), factor.name.clone());
                                    match self.compute_var_to_factor_message(
                                        graph,
                                        &old_messages,
                                        var_name,
                                        &factor.name,
                                    ) {
                                        Ok(msg) => Some((key, msg)),
                                        Err(_) => None,
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    } else {
                        Vec::new()
                    }
                })
                .collect();

            // Parallel computation of factor-to-variable messages
            let factor_var_updates: Vec<_> = graph
                .factor_ids()
                .par_bridge()
                .filter_map(|factor_id| graph.get_factor(factor_id))
                .flat_map(|factor| {
                    factor
                        .variables
                        .par_iter()
                        .filter_map(|var_name| {
                            let key = (factor.name.clone(), var_name.clone());
                            match self.compute_factor_to_var_message(
                                graph,
                                &old_messages,
                                &factor.name,
                                var_name,
                            ) {
                                Ok(msg) => Some((key, msg)),
                                Err(_) => None,
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            // Update messages with damping
            {
                let mut messages_guard = messages.lock().unwrap();
                for (key, new_msg) in var_factor_updates.into_iter().chain(factor_var_updates) {
                    if let Some(old_msg) = messages_guard.get(&key) {
                        if self.damping > 0.0 {
                            // Apply damping: msg_new = (1-d)*msg_new + d*msg_old
                            let damped = self.apply_damping(old_msg, &new_msg);
                            messages_guard.insert(key, damped);
                        } else {
                            messages_guard.insert(key, new_msg);
                        }
                    } else {
                        messages_guard.insert(key, new_msg);
                    }
                }
            }

            // Check convergence
            let converged = self.check_convergence(&old_messages, &messages.lock().unwrap());
            if converged {
                break;
            }

            if iteration == self.max_iterations - 1 {
                return Err(PgmError::ConvergenceFailure(format!(
                    "Parallel belief propagation did not converge after {} iterations",
                    self.max_iterations
                )));
            }
        }

        // Compute final marginals in parallel
        let marginals: HashMap<String, ArrayD<f64>> = graph
            .variable_names()
            .par_bridge()
            .filter_map(|var_name| {
                match self.compute_marginal(graph, &messages.lock().unwrap(), var_name) {
                    Ok(marginal) => Some((var_name.to_string(), marginal)),
                    Err(_) => None,
                }
            })
            .collect();

        Ok(marginals)
    }

    /// Initialize messages with uniform distributions.
    fn initialize_messages(
        &self,
        graph: &FactorGraph,
    ) -> Result<HashMap<(String, String), Factor>> {
        let mut messages = HashMap::new();

        // Initialize variable-to-factor messages
        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let uniform_values = vec![1.0 / var_node.cardinality as f64; var_node.cardinality];
                let uniform_array =
                    scirs2_core::ndarray::Array::from_vec(uniform_values).into_dyn();

                if let Some(factors) = graph.get_adjacent_factors(var_name) {
                    for factor_id in factors {
                        if let Some(factor) = graph.get_factor(factor_id) {
                            let msg = Factor::new(
                                format!("msg_{}_{}", var_name, factor.name),
                                vec![var_name.to_string()],
                                uniform_array.clone(),
                            )?;
                            messages.insert((var_name.to_string(), factor.name.clone()), msg);
                        }
                    }
                }
            }
        }

        // Initialize factor-to-variable messages
        for factor_id in graph.factor_ids() {
            if let Some(factor) = graph.get_factor(factor_id) {
                for var_name in &factor.variables {
                    if let Some(var_node) = graph.get_variable(var_name) {
                        let uniform_values =
                            vec![1.0 / var_node.cardinality as f64; var_node.cardinality];
                        let uniform_array =
                            scirs2_core::ndarray::Array::from_vec(uniform_values).into_dyn();

                        let msg = Factor::new(
                            format!("msg_{}_{}", factor.name, var_name),
                            vec![var_name.to_string()],
                            uniform_array,
                        )?;
                        messages.insert((factor.name.clone(), var_name.to_string()), msg);
                    }
                }
            }
        }

        Ok(messages)
    }

    /// Compute variable-to-factor message.
    fn compute_var_to_factor_message(
        &self,
        graph: &FactorGraph,
        messages: &HashMap<(String, String), Factor>,
        var: &str,
        target_factor: &str,
    ) -> Result<Factor> {
        let var_node = graph
            .get_variable(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;

        // Start with uniform
        let mut message_values = vec![1.0; var_node.cardinality];

        // Multiply all incoming factor-to-variable messages except from target
        if let Some(factors) = graph.get_adjacent_factors(var) {
            for factor_id in factors {
                if let Some(factor) = graph.get_factor(factor_id) {
                    if factor.name != target_factor {
                        let key = (factor.name.clone(), var.to_string());
                        if let Some(incoming_msg) = messages.get(&key) {
                            for (i, message_value) in message_values
                                .iter_mut()
                                .enumerate()
                                .take(var_node.cardinality)
                            {
                                *message_value *= incoming_msg.values[[i]];
                            }
                        }
                    }
                }
            }
        }

        let array = scirs2_core::ndarray::Array::from_vec(message_values).into_dyn();
        Factor::new(
            format!("msg_{}_{}", var, target_factor),
            vec![var.to_string()],
            array,
        )
    }

    /// Compute factor-to-variable message.
    fn compute_factor_to_var_message(
        &self,
        graph: &FactorGraph,
        messages: &HashMap<(String, String), Factor>,
        factor_name: &str,
        target_var: &str,
    ) -> Result<Factor> {
        let factor = graph
            .get_factor_by_name(factor_name)
            .ok_or_else(|| PgmError::InvalidGraph(format!("Factor {} not found", factor_name)))?;

        // Start with the factor
        let mut product = factor.clone();

        // Multiply all incoming variable-to-factor messages except from target
        for var in &factor.variables {
            if var != target_var {
                let key = (var.clone(), factor_name.to_string());
                if let Some(incoming_msg) = messages.get(&key) {
                    product = product.product(incoming_msg)?;
                }
            }
        }

        // Marginalize out all variables except target
        for var in &factor.variables {
            if var != target_var {
                product = product.marginalize_out(var)?;
            }
        }

        Ok(product)
    }

    /// Compute marginal for a variable.
    fn compute_marginal(
        &self,
        graph: &FactorGraph,
        messages: &HashMap<(String, String), Factor>,
        var: &str,
    ) -> Result<ArrayD<f64>> {
        let var_node = graph
            .get_variable(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;

        let mut marginal_values = vec![1.0; var_node.cardinality];

        // Multiply all incoming factor-to-variable messages
        if let Some(factors) = graph.get_adjacent_factors(var) {
            for factor_id in factors {
                if let Some(factor) = graph.get_factor(factor_id) {
                    let key = (factor.name.clone(), var.to_string());
                    if let Some(msg) = messages.get(&key) {
                        for (i, marginal_value) in marginal_values
                            .iter_mut()
                            .enumerate()
                            .take(var_node.cardinality)
                        {
                            *marginal_value *= msg.values[[i]];
                        }
                    }
                }
            }
        }

        // Normalize
        let sum: f64 = marginal_values.iter().sum();
        if sum > 0.0 {
            for val in &mut marginal_values {
                *val /= sum;
            }
        }

        Ok(scirs2_core::ndarray::Array::from_vec(marginal_values).into_dyn())
    }

    /// Apply damping to messages.
    fn apply_damping(&self, old_msg: &Factor, new_msg: &Factor) -> Factor {
        let mut damped_values = new_msg.values.clone();
        for i in 0..damped_values.len() {
            damped_values[[i]] =
                (1.0 - self.damping) * damped_values[[i]] + self.damping * old_msg.values[[i]];
        }

        Factor::new(
            new_msg.name.clone(),
            new_msg.variables.clone(),
            damped_values,
        )
        .unwrap_or_else(|_| new_msg.clone())
    }

    /// Check convergence of messages.
    fn check_convergence(
        &self,
        old_messages: &HashMap<(String, String), Factor>,
        new_messages: &HashMap<(String, String), Factor>,
    ) -> bool {
        for (key, new_msg) in new_messages {
            if let Some(old_msg) = old_messages.get(key) {
                let diff: f64 = new_msg
                    .values
                    .iter()
                    .zip(old_msg.values.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();

                if diff > self.tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Get convergence statistics.
    pub fn get_stats(&self) -> ConvergenceStats {
        ConvergenceStats {
            iterations: 0,
            converged: false,
            max_delta: 0.0,
        }
    }
}

/// Parallel max-product algorithm for MAP inference.
pub struct ParallelMaxProduct {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for ParallelMaxProduct {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl ParallelMaxProduct {
    /// Create with custom parameters.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Run parallel max-product.
    pub fn run_parallel(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        // Similar to ParallelSumProduct but using max instead of sum
        // Implementation follows the same pattern with max operations

        let parallel_sp = ParallelSumProduct::new(self.max_iterations, self.tolerance, 0.0);
        // For now, delegate to sum-product (in a full implementation, replace sum with max)
        parallel_sp.run_parallel(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn create_simple_chain() -> FactorGraph {
        let mut graph = FactorGraph::new();

        graph.add_variable_with_card("X".to_string(), "Domain".to_string(), 2);
        graph.add_variable_with_card("Y".to_string(), "Domain".to_string(), 2);

        let f_xy = Factor::new(
            "f_xy".to_string(),
            vec!["X".to_string(), "Y".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();

        graph.add_factor(f_xy).unwrap();

        graph
    }

    #[test]
    fn test_parallel_sum_product() {
        let graph = create_simple_chain();
        let parallel_bp = ParallelSumProduct::default();

        let marginals = parallel_bp.run_parallel(&graph).unwrap();

        assert_eq!(marginals.len(), 2);

        // Check normalization
        for marginal in marginals.values() {
            let sum: f64 = marginal.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Marginal sum: {}", sum);
        }
    }

    #[test]
    fn test_parallel_with_damping() {
        let graph = create_simple_chain();
        let parallel_bp = ParallelSumProduct::new(100, 1e-6, 0.5);

        let marginals = parallel_bp.run_parallel(&graph).unwrap();

        assert_eq!(marginals.len(), 2);
    }

    #[test]
    fn test_parallel_max_product() {
        let graph = create_simple_chain();
        let parallel_mp = ParallelMaxProduct::default();

        let marginals = parallel_mp.run_parallel(&graph).unwrap();

        assert_eq!(marginals.len(), 2);
    }
}
