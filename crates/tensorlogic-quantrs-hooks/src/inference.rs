//! High-level inference operations.

use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;

use crate::error::Result;
use crate::graph::FactorGraph;
use crate::message_passing::MessagePassingAlgorithm;

/// Query for marginal probability P(X).
#[derive(Clone, Debug)]
pub struct MarginalizationQuery {
    /// Variable to query
    pub variable: String,
}

/// Query for conditional probability P(X | Y = y).
#[derive(Clone, Debug)]
pub struct ConditionalQuery {
    /// Query variables
    pub query_vars: Vec<String>,
    /// Evidence: variable -> value
    pub evidence: HashMap<String, usize>,
}

/// Inference engine for PGM queries.
pub struct InferenceEngine {
    /// Factor graph
    graph: FactorGraph,
    /// Message passing algorithm
    algorithm: Box<dyn MessagePassingAlgorithm>,
}

impl InferenceEngine {
    /// Create a new inference engine.
    pub fn new(graph: FactorGraph, algorithm: Box<dyn MessagePassingAlgorithm>) -> Self {
        Self { graph, algorithm }
    }

    /// Compute marginal probability for a variable.
    pub fn marginalize(&self, query: &MarginalizationQuery) -> Result<ArrayD<f64>> {
        let marginals = self.algorithm.run(&self.graph)?;
        marginals
            .get(&query.variable)
            .cloned()
            .ok_or_else(|| crate::error::PgmError::VariableNotFound(query.variable.clone()))
    }

    /// Compute conditional probability.
    pub fn conditional(&self, query: &ConditionalQuery) -> Result<HashMap<String, ArrayD<f64>>> {
        // Run inference with evidence
        let marginals = self.algorithm.run(&self.graph)?;

        // Filter to query variables
        let mut result = HashMap::new();
        for var in &query.query_vars {
            if let Some(marginal) = marginals.get(var) {
                result.insert(var.clone(), marginal.clone());
            }
        }

        Ok(result)
    }

    /// Compute joint probability over all variables.
    ///
    /// This computes P(X₁, X₂, ..., Xₙ) = ∏ᵢ φᵢ(Xᵢ)
    pub fn joint(&self) -> Result<ArrayD<f64>> {
        use crate::factor::Factor;

        // Collect all variables
        let all_vars: Vec<String> = self.graph.variable_names().cloned().collect();

        if all_vars.is_empty() {
            return Err(crate::error::PgmError::InvalidGraph(
                "No variables in graph".to_string(),
            ));
        }

        // Start with first factor or uniform distribution
        let mut joint_factor: Option<Factor> = None;

        for factor_id in self.graph.factor_ids() {
            if let Some(factor) = self.graph.get_factor(factor_id) {
                joint_factor = if let Some(existing) = joint_factor {
                    Some(existing.product(factor)?)
                } else {
                    Some(factor.clone())
                };
            }
        }

        // If no factors, return uniform distribution
        if let Some(mut joint) = joint_factor {
            joint.normalize();
            Ok(joint.values)
        } else {
            // No factors - return uniform
            let shape: Vec<usize> = all_vars
                .iter()
                .filter_map(|v| self.graph.get_variable(v))
                .map(|n| n.cardinality)
                .collect();
            let size: usize = shape.iter().product();
            Ok(ArrayD::from_elem(shape, 1.0 / size as f64))
        }
    }

    /// Get the factor graph.
    pub fn graph(&self) -> &FactorGraph {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message_passing::SumProductAlgorithm;

    #[test]
    fn test_inference_engine() {
        let mut graph = FactorGraph::new();
        graph.add_variable("x".to_string(), "D1".to_string());

        let algorithm = Box::new(SumProductAlgorithm::default());
        let engine = InferenceEngine::new(graph, algorithm);

        let query = MarginalizationQuery {
            variable: "x".to_string(),
        };

        let result = engine.marginalize(&query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_conditional_query() {
        let mut graph = FactorGraph::new();
        graph.add_variable("x".to_string(), "D1".to_string());
        graph.add_variable("y".to_string(), "D2".to_string());

        let algorithm = Box::new(SumProductAlgorithm::default());
        let engine = InferenceEngine::new(graph, algorithm);

        let query = ConditionalQuery {
            query_vars: vec!["x".to_string()],
            evidence: HashMap::new(),
        };

        let result = engine.conditional(&query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_joint_probability() {
        let mut graph = FactorGraph::new();
        graph.add_variable("var_0".to_string(), "D1".to_string());

        let algorithm = Box::new(SumProductAlgorithm::default());
        let engine = InferenceEngine::new(graph, algorithm);

        let joint = engine.joint();
        assert!(joint.is_ok());

        // Should be normalized
        let sum: f64 = joint.unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_joint_with_multiple_variables() {
        let mut graph = FactorGraph::new();
        graph.add_variable("var_0".to_string(), "D1".to_string());
        graph.add_variable("var_1".to_string(), "D2".to_string());

        let algorithm = Box::new(SumProductAlgorithm::default());
        let engine = InferenceEngine::new(graph, algorithm);

        let joint = engine.joint();
        assert!(joint.is_ok());

        // Should be normalized
        let sum: f64 = joint.unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
