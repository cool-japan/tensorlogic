//! Message passing algorithms for PGM inference.

use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;

use crate::error::{PgmError, Result};
use crate::factor::Factor;
use crate::graph::FactorGraph;

/// Trait for message passing algorithms.
pub trait MessagePassingAlgorithm: Send + Sync {
    /// Run the algorithm on a factor graph.
    fn run(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>>;

    /// Get algorithm name.
    fn name(&self) -> &str;
}

/// Message storage for belief propagation.
#[derive(Clone, Debug)]
struct MessageStore {
    /// Messages from variables to factors: (var, factor) -> message
    var_to_factor: HashMap<(String, String), Factor>,
    /// Messages from factors to variables: (factor, var) -> message
    factor_to_var: HashMap<(String, String), Factor>,
}

impl MessageStore {
    fn new() -> Self {
        Self {
            var_to_factor: HashMap::new(),
            factor_to_var: HashMap::new(),
        }
    }

    fn get_var_to_factor(&self, var: &str, factor: &str) -> Option<&Factor> {
        self.var_to_factor
            .get(&(var.to_string(), factor.to_string()))
    }

    fn set_var_to_factor(&mut self, var: String, factor: String, message: Factor) {
        self.var_to_factor.insert((var, factor), message);
    }

    fn get_factor_to_var(&self, factor: &str, var: &str) -> Option<&Factor> {
        self.factor_to_var
            .get(&(factor.to_string(), var.to_string()))
    }

    fn set_factor_to_var(&mut self, factor: String, var: String, message: Factor) {
        self.factor_to_var.insert((factor, var), message);
    }
}

/// Convergence statistics for belief propagation.
#[derive(Clone, Debug)]
pub struct ConvergenceStats {
    /// Number of iterations performed
    pub iterations: usize,
    /// Maximum message difference in last iteration
    pub max_delta: f64,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// Sum-product algorithm (belief propagation).
///
/// Computes exact marginal probabilities for tree-structured graphs.
/// For loopy graphs, runs loopy belief propagation with optional damping.
pub struct SumProductAlgorithm {
    /// Maximum iterations for loopy graphs
    pub max_iterations: usize,
    /// Convergence threshold
    pub tolerance: f64,
    /// Damping factor (0.0 = no damping, 1.0 = full damping)
    pub damping: f64,
}

impl Default for SumProductAlgorithm {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            damping: 0.0,
        }
    }
}

impl SumProductAlgorithm {
    /// Create with custom parameters.
    pub fn new(max_iterations: usize, tolerance: f64, damping: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            damping: damping.clamp(0.0, 1.0),
        }
    }

    /// Compute variable-to-factor message.
    ///
    /// μ(x→f) = ∏_{g∈N(x)\f} μ(g→x)
    fn compute_var_to_factor_message(
        &self,
        graph: &FactorGraph,
        messages: &MessageStore,
        var: &str,
        target_factor: &str,
    ) -> Result<Factor> {
        let var_node = graph
            .get_variable(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;

        // Get all factors connected to this variable except target
        let adjacent_factors = graph
            .get_adjacent_factors(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;

        let other_factors: Vec<&String> = adjacent_factors
            .iter()
            .filter(|&f| f != target_factor)
            .collect();

        // Start with uniform message
        let mut message = Factor::uniform(
            format!("msg_{}_{}", var, target_factor),
            vec![var.to_string()],
            var_node.cardinality,
        );

        // Multiply incoming messages from other factors
        for &factor_id in &other_factors {
            if let Some(incoming) = messages.get_factor_to_var(factor_id, var) {
                message = message.product(incoming)?;
            }
        }

        // Normalize
        message.normalize();

        Ok(message)
    }

    /// Compute factor-to-variable message.
    ///
    /// μ(f→x) = ∑_{~x} [φ(x) ∏_{y∈N(f)\x} μ(y→f)]
    fn compute_factor_to_var_message(
        &self,
        graph: &FactorGraph,
        messages: &MessageStore,
        factor_id: &str,
        target_var: &str,
    ) -> Result<Factor> {
        let factor = graph
            .get_factor(factor_id)
            .ok_or_else(|| PgmError::FactorNotFound(factor_id.to_string()))?;

        // Start with factor itself
        let mut message = factor.clone();

        // Get variables in factor except target
        let other_vars: Vec<&String> = factor
            .variables
            .iter()
            .filter(|&v| v != target_var)
            .collect();

        // Multiply incoming messages from other variables
        for &var in &other_vars {
            if let Some(incoming) = messages.get_var_to_factor(var, factor_id) {
                message = message.product(incoming)?;
            }
        }

        // Marginalize out all variables except target
        for &var in &other_vars {
            message = message.marginalize_out(var)?;
        }

        // Normalize
        message.normalize();

        Ok(message)
    }

    /// Compute beliefs (marginals) from messages.
    fn compute_beliefs(
        &self,
        graph: &FactorGraph,
        messages: &MessageStore,
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut beliefs = HashMap::new();

        // For each variable, multiply all incoming factor messages
        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let mut belief = Factor::uniform(
                    format!("belief_{}", var_name),
                    vec![var_name.clone()],
                    var_node.cardinality,
                );

                // Get all adjacent factors
                if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
                    for factor_id in adjacent_factors {
                        if let Some(message) = messages.get_factor_to_var(factor_id, var_name) {
                            belief = belief.product(message)?;
                        }
                    }
                }

                belief.normalize();
                beliefs.insert(var_name.clone(), belief.values);
            }
        }

        Ok(beliefs)
    }

    /// Check convergence by comparing message differences.
    fn check_convergence(
        &self,
        old_messages: &MessageStore,
        new_messages: &MessageStore,
    ) -> (bool, f64) {
        let mut max_delta: f64 = 0.0;

        // Check factor-to-var messages
        for ((factor, var), new_msg) in &new_messages.factor_to_var {
            if let Some(old_msg) = old_messages.get_factor_to_var(factor, var) {
                let delta: f64 = (&new_msg.values - &old_msg.values)
                    .mapv(|x| x.abs())
                    .iter()
                    .fold(0.0_f64, |acc, &x| acc.max(x));
                max_delta = max_delta.max(delta);
            }
        }

        (max_delta < self.tolerance, max_delta)
    }

    /// Apply damping to messages.
    fn apply_damping(&self, old_msg: &Factor, new_msg: &Factor) -> Result<Factor> {
        if self.damping == 0.0 {
            return Ok(new_msg.clone());
        }

        // Damped message = (1 - λ) * new + λ * old
        let damped_values = &new_msg.values * (1.0 - self.damping) + &old_msg.values * self.damping;

        Ok(Factor {
            name: new_msg.name.clone(),
            variables: new_msg.variables.clone(),
            values: damped_values,
        })
    }
}

impl MessagePassingAlgorithm for SumProductAlgorithm {
    fn run(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut messages = MessageStore::new();

        // Initialize all messages to uniform
        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
                    for factor_id in adjacent_factors {
                        let init_msg = Factor::uniform(
                            format!("init_{}_{}", var_name, factor_id),
                            vec![var_name.clone()],
                            var_node.cardinality,
                        );
                        messages.set_var_to_factor(var_name.clone(), factor_id.clone(), init_msg);
                    }
                }
            }
        }

        // Iterative message passing
        for iteration in 0..self.max_iterations {
            let old_messages = messages.clone();

            // Update all variable-to-factor messages
            for var_name in graph.variable_names() {
                if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
                    for factor_id in adjacent_factors {
                        let new_msg = self
                            .compute_var_to_factor_message(graph, &messages, var_name, factor_id)?;
                        messages.set_var_to_factor(var_name.clone(), factor_id.clone(), new_msg);
                    }
                }
            }

            // Update all factor-to-variable messages
            for factor_id in graph.factor_ids() {
                if let Some(adjacent_vars) = graph.get_adjacent_variables(factor_id) {
                    for var in adjacent_vars {
                        let new_msg =
                            self.compute_factor_to_var_message(graph, &messages, factor_id, var)?;

                        // Apply damping if enabled
                        let damped_msg =
                            if let Some(old_msg) = old_messages.get_factor_to_var(factor_id, var) {
                                self.apply_damping(old_msg, &new_msg)?
                            } else {
                                new_msg
                            };

                        messages.set_factor_to_var(factor_id.clone(), var.clone(), damped_msg);
                    }
                }
            }

            // Check convergence
            let (converged, max_delta) = self.check_convergence(&old_messages, &messages);

            if converged {
                // Compute and return beliefs
                return self.compute_beliefs(graph, &messages);
            }

            // Prevent infinite loop
            if iteration == self.max_iterations - 1 {
                return Err(PgmError::ConvergenceFailure(format!(
                    "Failed to converge after {} iterations (max_delta={})",
                    self.max_iterations, max_delta
                )));
            }
        }

        // Compute beliefs even if not converged
        self.compute_beliefs(graph, &messages)
    }

    fn name(&self) -> &str {
        "SumProduct"
    }
}

/// Max-product algorithm (MAP inference).
///
/// Computes the most likely assignment to all variables.
pub struct MaxProductAlgorithm {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub tolerance: f64,
}

impl Default for MaxProductAlgorithm {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl MaxProductAlgorithm {
    /// Create with custom parameters.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl MaxProductAlgorithm {
    /// Compute variable-to-factor message (same as sum-product).
    fn compute_var_to_factor_message(
        &self,
        graph: &FactorGraph,
        messages: &MessageStore,
        var: &str,
        target_factor: &str,
    ) -> Result<Factor> {
        let var_node = graph
            .get_variable(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;

        let adjacent_factors = graph
            .get_adjacent_factors(var)
            .ok_or_else(|| PgmError::VariableNotFound(var.to_string()))?;

        let other_factors: Vec<&String> = adjacent_factors
            .iter()
            .filter(|&f| f != target_factor)
            .collect();

        let mut message = Factor::uniform(
            format!("msg_{}_{}", var, target_factor),
            vec![var.to_string()],
            var_node.cardinality,
        );

        for &factor_id in &other_factors {
            if let Some(incoming) = messages.get_factor_to_var(factor_id, var) {
                message = message.product(incoming)?;
            }
        }

        message.normalize();
        Ok(message)
    }

    /// Compute factor-to-variable message using MAX instead of SUM.
    fn compute_factor_to_var_message(
        &self,
        graph: &FactorGraph,
        messages: &MessageStore,
        factor_id: &str,
        target_var: &str,
    ) -> Result<Factor> {
        let factor = graph
            .get_factor(factor_id)
            .ok_or_else(|| PgmError::FactorNotFound(factor_id.to_string()))?;

        let mut message = factor.clone();

        let other_vars: Vec<&String> = factor
            .variables
            .iter()
            .filter(|&v| v != target_var)
            .collect();

        for &var in &other_vars {
            if let Some(incoming) = messages.get_var_to_factor(var, factor_id) {
                message = message.product(incoming)?;
            }
        }

        // Use MAX instead of SUM for marginalization
        for &var in &other_vars {
            message = message.maximize_out(var)?;
        }

        message.normalize();
        Ok(message)
    }

    /// Compute beliefs using max-product messages.
    fn compute_beliefs(
        &self,
        graph: &FactorGraph,
        messages: &MessageStore,
    ) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut beliefs = HashMap::new();

        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                let mut belief = Factor::uniform(
                    format!("belief_{}", var_name),
                    vec![var_name.clone()],
                    var_node.cardinality,
                );

                if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
                    for factor_id in adjacent_factors {
                        if let Some(message) = messages.get_factor_to_var(factor_id, var_name) {
                            belief = belief.product(message)?;
                        }
                    }
                }

                belief.normalize();
                beliefs.insert(var_name.clone(), belief.values);
            }
        }

        Ok(beliefs)
    }
}

impl MessagePassingAlgorithm for MaxProductAlgorithm {
    fn run(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut messages = MessageStore::new();

        // Initialize messages
        for var_name in graph.variable_names() {
            if let Some(var_node) = graph.get_variable(var_name) {
                if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
                    for factor_id in adjacent_factors {
                        let init_msg = Factor::uniform(
                            format!("init_{}_{}", var_name, factor_id),
                            vec![var_name.clone()],
                            var_node.cardinality,
                        );
                        messages.set_var_to_factor(var_name.clone(), factor_id.clone(), init_msg);
                    }
                }
            }
        }

        // Iterative message passing
        for _iteration in 0..self.max_iterations {
            let _old_messages = messages.clone();

            // Update variable-to-factor messages
            for var_name in graph.variable_names() {
                if let Some(adjacent_factors) = graph.get_adjacent_factors(var_name) {
                    for factor_id in adjacent_factors {
                        let new_msg = self
                            .compute_var_to_factor_message(graph, &messages, var_name, factor_id)?;
                        messages.set_var_to_factor(var_name.clone(), factor_id.clone(), new_msg);
                    }
                }
            }

            // Update factor-to-variable messages
            for factor_id in graph.factor_ids() {
                if let Some(adjacent_vars) = graph.get_adjacent_variables(factor_id) {
                    for var in adjacent_vars {
                        let new_msg =
                            self.compute_factor_to_var_message(graph, &messages, factor_id, var)?;
                        messages.set_factor_to_var(factor_id.clone(), var.clone(), new_msg);
                    }
                }
            }
        }

        self.compute_beliefs(graph, &messages)
    }

    fn name(&self) -> &str {
        "MaxProduct"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::FactorGraph;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sum_product_algorithm() {
        let algorithm = SumProductAlgorithm::default();
        assert_eq!(algorithm.name(), "SumProduct");

        let mut graph = FactorGraph::new();
        graph.add_variable("var_0".to_string(), "D1".to_string());

        let result = algorithm.run(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_max_product_algorithm() {
        let algorithm = MaxProductAlgorithm::default();
        assert_eq!(algorithm.name(), "MaxProduct");

        let mut graph = FactorGraph::new();
        graph.add_variable("var_0".to_string(), "D1".to_string());

        let result = algorithm.run(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_message_store() {
        let mut store = MessageStore::new();
        let msg = Factor::uniform("test".to_string(), vec!["x".to_string()], 2);

        store.set_var_to_factor("x".to_string(), "f1".to_string(), msg.clone());
        assert!(store.get_var_to_factor("x", "f1").is_some());

        store.set_factor_to_var("f1".to_string(), "x".to_string(), msg.clone());
        assert!(store.get_factor_to_var("f1", "x").is_some());
    }

    #[test]
    fn test_sum_product_with_damping() {
        let algorithm = SumProductAlgorithm::new(50, 1e-5, 0.5);
        assert_eq!(algorithm.damping, 0.5);

        let mut graph = FactorGraph::new();
        graph.add_variable("var_0".to_string(), "D1".to_string());

        let result = algorithm.run(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_belief_normalization() {
        let mut graph = FactorGraph::new();
        graph.add_variable("var_0".to_string(), "D1".to_string());

        let algorithm = SumProductAlgorithm::default();
        let beliefs = algorithm.run(&graph).unwrap();

        if let Some(belief) = beliefs.get("var_0") {
            let sum: f64 = belief.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }
}
