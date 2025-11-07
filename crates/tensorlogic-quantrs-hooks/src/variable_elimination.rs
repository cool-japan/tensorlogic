//! Variable Elimination algorithm for exact inference.
//!
//! Variable Elimination is a classic exact inference algorithm that eliminates
//! variables one by one from the factor graph. The complexity depends on the
//! elimination order.

use scirs2_core::ndarray::ArrayD;
use std::collections::{HashMap, HashSet};

use crate::error::{PgmError, Result};
use crate::factor::Factor;
use crate::graph::FactorGraph;

/// Variable elimination algorithm for exact inference.
///
/// Computes marginal probabilities by eliminating variables in a specific order.
pub struct VariableElimination {
    /// Elimination order (if None, uses min-degree heuristic)
    pub elimination_order: Option<Vec<String>>,
}

impl Default for VariableElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl VariableElimination {
    /// Create a new variable elimination algorithm.
    pub fn new() -> Self {
        Self {
            elimination_order: None,
        }
    }

    /// Create with a specific elimination order.
    pub fn with_order(order: Vec<String>) -> Self {
        Self {
            elimination_order: Some(order),
        }
    }

    /// Compute marginal for a single query variable.
    pub fn marginalize(&self, graph: &FactorGraph, query_var: &str) -> Result<ArrayD<f64>> {
        // Check if query variable exists
        let query_node = graph
            .get_variable(query_var)
            .ok_or_else(|| PgmError::VariableNotFound(query_var.to_string()))?;

        // Get all factors as a working set
        let mut factors: Vec<Factor> = graph
            .factor_ids()
            .filter_map(|id| graph.get_factor(id).cloned())
            .collect();

        // If no factors, return uniform distribution
        if factors.is_empty() {
            let uniform = ArrayD::from_elem(
                vec![query_node.cardinality],
                1.0 / query_node.cardinality as f64,
            );
            return Ok(uniform);
        }

        // Determine elimination order
        let all_vars: HashSet<String> = graph.variable_names().cloned().collect();
        let vars_to_eliminate: Vec<String> =
            all_vars.into_iter().filter(|v| v != query_var).collect();

        let order = if let Some(ref custom_order) = self.elimination_order {
            custom_order
                .iter()
                .filter(|v| vars_to_eliminate.contains(v))
                .cloned()
                .collect()
        } else {
            self.compute_elimination_order(graph, &vars_to_eliminate)?
        };

        // Eliminate variables one by one
        for var in &order {
            factors = self.eliminate_variable(&factors, var)?;
        }

        // Multiply remaining factors and marginalize to query variable
        let mut result = self.multiply_all_factors(&factors)?;

        // If result contains more than just the query variable, marginalize others
        let vars_to_remove: Vec<String> = result
            .variables
            .iter()
            .filter(|v| *v != query_var)
            .cloned()
            .collect();

        for var in vars_to_remove {
            result = result.marginalize_out(&var)?;
        }

        // Normalize
        result.normalize();

        Ok(result.values)
    }

    /// Compute marginals for all variables.
    pub fn marginalize_all(&self, graph: &FactorGraph) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut marginals = HashMap::new();

        for var_name in graph.variable_names() {
            let marginal = self.marginalize(graph, var_name)?;
            marginals.insert(var_name.clone(), marginal);
        }

        Ok(marginals)
    }

    /// Eliminate a single variable from a set of factors.
    fn eliminate_variable(&self, factors: &[Factor], var: &str) -> Result<Vec<Factor>> {
        // Find all factors containing this variable
        let (containing, not_containing): (Vec<Factor>, Vec<Factor>) = factors
            .iter()
            .cloned()
            .partition(|f| f.variables.contains(&var.to_string()));

        if containing.is_empty() {
            // Variable not in any factor, nothing to eliminate
            return Ok(factors.to_vec());
        }

        // Multiply all factors containing the variable
        let mut product = containing[0].clone();
        for factor in &containing[1..] {
            product = product.product(factor)?;
        }

        // Marginalize out the variable
        let marginalized = product.marginalize_out(var)?;

        // Combine with factors that didn't contain the variable
        let mut result = not_containing;
        if !marginalized.variables.is_empty() {
            result.push(marginalized);
        }

        Ok(result)
    }

    /// Multiply all factors together.
    fn multiply_all_factors(&self, factors: &[Factor]) -> Result<Factor> {
        if factors.is_empty() {
            return Err(PgmError::InvalidGraph("No factors to multiply".to_string()));
        }

        let mut result = factors[0].clone();
        for factor in &factors[1..] {
            result = result.product(factor)?;
        }

        Ok(result)
    }

    /// Compute elimination order using min-degree heuristic.
    ///
    /// Chooses variables in order of fewest connections (smallest induced clique size).
    fn compute_elimination_order(
        &self,
        graph: &FactorGraph,
        vars: &[String],
    ) -> Result<Vec<String>> {
        // Simple heuristic: eliminate variables in the order they appear
        // More sophisticated: use min-degree, min-fill, or max-cardinality search
        let mut order = vars.to_vec();

        // Sort by number of factors containing each variable
        order.sort_by_key(|v| {
            graph
                .get_adjacent_factors(v)
                .map(|factors| factors.len())
                .unwrap_or(0)
        });

        Ok(order)
    }

    /// Compute joint probability for a specific assignment.
    pub fn joint_probability(
        &self,
        graph: &FactorGraph,
        assignment: &HashMap<String, usize>,
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

    /// Compute MAP (Maximum A Posteriori) assignment using variable elimination.
    pub fn map(&self, graph: &FactorGraph) -> Result<HashMap<String, usize>> {
        // Get all factors
        let mut factors: Vec<Factor> = graph
            .factor_ids()
            .filter_map(|id| graph.get_factor(id).cloned())
            .collect();

        // Get elimination order (all variables)
        let all_vars: Vec<String> = graph.variable_names().cloned().collect();
        let order = if let Some(ref custom_order) = self.elimination_order {
            custom_order.clone()
        } else {
            self.compute_elimination_order(graph, &all_vars)?
        };

        let mut assignment = HashMap::new();

        // Eliminate variables using MAX instead of SUM
        for var in order.iter().rev() {
            // Find factors containing this variable
            let (containing, not_containing): (Vec<Factor>, Vec<Factor>) = factors
                .iter()
                .cloned()
                .partition(|f| f.variables.contains(&var.to_string()));

            if containing.is_empty() {
                continue;
            }

            // Multiply factors
            let mut product = containing[0].clone();
            for factor in &containing[1..] {
                product = product.product(factor)?;
            }

            // Find max value for this variable
            let var_node = graph
                .get_variable(var)
                .ok_or_else(|| PgmError::VariableNotFound(var.clone()))?;

            let mut max_val = f64::NEG_INFINITY;
            let mut max_idx = 0;

            for val in 0..var_node.cardinality {
                let reduced = product.reduce(var, val)?;
                let prob: f64 = reduced.values.iter().product();

                if prob > max_val {
                    max_val = prob;
                    max_idx = val;
                }
            }

            assignment.insert(var.clone(), max_idx);

            // Reduce factor to this assignment
            let reduced = product.reduce(var, max_idx)?;
            factors = not_containing;
            if !reduced.variables.is_empty() {
                factors.push(reduced);
            }
        }

        Ok(assignment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_variable_elimination_single_variable() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

        // Add uniform factor
        let factor = Factor::uniform("P(x)".to_string(), vec!["x".to_string()], 2);
        graph.add_factor(factor).unwrap();

        let ve = VariableElimination::new();
        let marginal = ve.marginalize(&graph, "x").unwrap();

        assert_eq!(marginal.len(), 2);
        let sum: f64 = marginal.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_variable_elimination_chain() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        // P(x)
        let px_values = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        let px = Factor::new("P(x)".to_string(), vec!["x".to_string()], px_values).unwrap();
        graph.add_factor(px).unwrap();

        // P(y|x)
        let pyx_values = Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])
            .unwrap()
            .into_dyn();
        let pyx = Factor::new(
            "P(y|x)".to_string(),
            vec!["x".to_string(), "y".to_string()],
            pyx_values,
        )
        .unwrap();
        graph.add_factor(pyx).unwrap();

        let ve = VariableElimination::new();
        let marginal_y = ve.marginalize(&graph, "y").unwrap();

        assert_eq!(marginal_y.len(), 2);
        let sum: f64 = marginal_y.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_marginalize_all() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let ve = VariableElimination::new();
        let marginals = ve.marginalize_all(&graph).unwrap();

        assert_eq!(marginals.len(), 2);
        assert!(marginals.contains_key("x"));
        assert!(marginals.contains_key("y"));
    }

    #[test]
    fn test_joint_probability() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        // Add factors
        let px_values = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
            .unwrap()
            .into_dyn();
        let px = Factor::new("P(x)".to_string(), vec!["x".to_string()], px_values).unwrap();
        graph.add_factor(px).unwrap();

        let pyx_values = Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])
            .unwrap()
            .into_dyn();
        let pyx = Factor::new(
            "P(y|x)".to_string(),
            vec!["x".to_string(), "y".to_string()],
            pyx_values,
        )
        .unwrap();
        graph.add_factor(pyx).unwrap();

        let mut assignment = HashMap::new();
        assignment.insert("x".to_string(), 0);
        assignment.insert("y".to_string(), 1);

        let ve = VariableElimination::new();
        let prob = ve.joint_probability(&graph, &assignment).unwrap();

        // P(x=0, y=1) = P(x=0) * P(y=1|x=0) = 0.6 * 0.1 = 0.06
        assert_abs_diff_eq!(prob, 0.06, epsilon = 1e-6);
    }

    #[test]
    fn test_custom_elimination_order() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("z".to_string(), "Binary".to_string(), 2);

        let order = vec!["x".to_string(), "y".to_string()];
        let ve = VariableElimination::with_order(order);

        let marginal = ve.marginalize(&graph, "z").unwrap();
        assert_eq!(marginal.len(), 2);
    }

    #[test]
    fn test_map_inference() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        // Add biased factors
        let px_values = Array::from_shape_vec(vec![2], vec![0.3, 0.7])
            .unwrap()
            .into_dyn();
        let px = Factor::new("P(x)".to_string(), vec!["x".to_string()], px_values).unwrap();
        graph.add_factor(px).unwrap();

        let pyx_values = Array::from_shape_vec(vec![2, 2], vec![0.8, 0.2, 0.1, 0.9])
            .unwrap()
            .into_dyn();
        let pyx = Factor::new(
            "P(y|x)".to_string(),
            vec!["x".to_string(), "y".to_string()],
            pyx_values,
        )
        .unwrap();
        graph.add_factor(pyx).unwrap();

        let ve = VariableElimination::new();
        let map_assignment = ve.map(&graph).unwrap();

        assert!(map_assignment.contains_key("x"));
        assert!(map_assignment.contains_key("y"));
    }
}
