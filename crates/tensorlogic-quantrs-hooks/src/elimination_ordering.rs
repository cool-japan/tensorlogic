//! Elimination ordering heuristics for variable elimination.
//!
//! Different heuristics can produce significantly different elimination orders,
//! which affects the computational cost of variable elimination. This module
//! provides several classic ordering heuristics.

use crate::error::{PgmError, Result};
use crate::graph::FactorGraph;
use std::collections::{HashMap, HashSet};

/// Strategy for computing variable elimination ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EliminationStrategy {
    /// Min-degree: Choose variable with fewest neighbors
    #[default]
    MinDegree,
    /// Min-fill: Choose variable that introduces fewest new edges
    MinFill,
    /// Weighted min-fill: Min-fill weighted by factor sizes
    WeightedMinFill,
    /// Min-width: Minimize the width of the induced tree
    MinWidth,
    /// Max-cardinality search: Greedy algorithm that tends to produce good orderings
    MaxCardinalitySearch,
}

/// Compute elimination ordering for variable elimination.
pub struct EliminationOrdering {
    strategy: EliminationStrategy,
}

impl Default for EliminationOrdering {
    fn default() -> Self {
        Self::new(EliminationStrategy::default())
    }
}

impl EliminationOrdering {
    /// Create with a specific strategy.
    pub fn new(strategy: EliminationStrategy) -> Self {
        Self { strategy }
    }

    /// Compute elimination order for the given variables.
    pub fn compute_order(&self, graph: &FactorGraph, vars: &[String]) -> Result<Vec<String>> {
        match self.strategy {
            EliminationStrategy::MinDegree => self.min_degree_order(graph, vars),
            EliminationStrategy::MinFill => self.min_fill_order(graph, vars),
            EliminationStrategy::WeightedMinFill => self.weighted_min_fill_order(graph, vars),
            EliminationStrategy::MinWidth => self.min_width_order(graph, vars),
            EliminationStrategy::MaxCardinalitySearch => self.max_cardinality_search(graph, vars),
        }
    }

    /// Min-degree heuristic: Choose variable with fewest neighbors.
    ///
    /// This is a simple and fast heuristic that works well in many cases.
    fn min_degree_order(&self, graph: &FactorGraph, vars: &[String]) -> Result<Vec<String>> {
        let mut remaining: HashSet<String> = vars.iter().cloned().collect();
        let mut order = Vec::new();

        // Build initial adjacency graph
        let mut adjacency = self.build_adjacency_graph(graph, &remaining)?;

        while !remaining.is_empty() {
            // Find variable with minimum degree
            let min_var = remaining
                .iter()
                .min_by_key(|v| adjacency.get(*v).map(|s| s.len()).unwrap_or(0))
                .ok_or_else(|| PgmError::InvalidGraph("No variables to eliminate".to_string()))?
                .clone();

            order.push(min_var.clone());
            remaining.remove(&min_var);

            // Update adjacency after elimination
            self.update_adjacency_after_elimination(&mut adjacency, &min_var);
        }

        Ok(order)
    }

    /// Min-fill heuristic: Choose variable that introduces fewest new edges.
    ///
    /// When a variable is eliminated, its neighbors become fully connected.
    /// This heuristic minimizes the number of new edges (fill) created.
    fn min_fill_order(&self, graph: &FactorGraph, vars: &[String]) -> Result<Vec<String>> {
        let mut remaining: HashSet<String> = vars.iter().cloned().collect();
        let mut order = Vec::new();

        // Build initial adjacency graph
        let mut adjacency = self.build_adjacency_graph(graph, &remaining)?;

        while !remaining.is_empty() {
            // Find variable that introduces minimum fill
            let min_var = remaining
                .iter()
                .min_by_key(|v| self.compute_fill(&adjacency, v))
                .ok_or_else(|| PgmError::InvalidGraph("No variables to eliminate".to_string()))?
                .clone();

            order.push(min_var.clone());
            remaining.remove(&min_var);

            // Update adjacency after elimination
            self.update_adjacency_after_elimination(&mut adjacency, &min_var);
        }

        Ok(order)
    }

    /// Weighted min-fill: Min-fill weighted by factor sizes.
    ///
    /// Similar to min-fill, but weights the fill by the product of factor sizes.
    /// This tries to minimize the computational cost more directly.
    fn weighted_min_fill_order(&self, graph: &FactorGraph, vars: &[String]) -> Result<Vec<String>> {
        let mut remaining: HashSet<String> = vars.iter().cloned().collect();
        let mut order = Vec::new();

        // Build initial adjacency graph with weights
        let mut adjacency = self.build_adjacency_graph(graph, &remaining)?;
        let weights = self.compute_variable_weights(graph, vars)?;

        while !remaining.is_empty() {
            // Find variable that introduces minimum weighted fill
            let min_var = remaining
                .iter()
                .min_by_key(|v| {
                    let fill = self.compute_fill(&adjacency, v);
                    let weight = weights.get(*v).copied().unwrap_or(1);
                    fill * weight
                })
                .ok_or_else(|| PgmError::InvalidGraph("No variables to eliminate".to_string()))?
                .clone();

            order.push(min_var.clone());
            remaining.remove(&min_var);

            // Update adjacency after elimination
            self.update_adjacency_after_elimination(&mut adjacency, &min_var);
        }

        Ok(order)
    }

    /// Min-width heuristic: Minimize the width of the induced tree.
    ///
    /// Width is the size of the largest clique created during elimination.
    fn min_width_order(&self, graph: &FactorGraph, vars: &[String]) -> Result<Vec<String>> {
        let mut remaining: HashSet<String> = vars.iter().cloned().collect();
        let mut order = Vec::new();

        // Build initial adjacency graph
        let mut adjacency = self.build_adjacency_graph(graph, &remaining)?;

        while !remaining.is_empty() {
            // Find variable that minimizes induced width
            let min_var = remaining
                .iter()
                .min_by_key(|v| {
                    let neighbors = adjacency.get(*v).map(|s| s.len()).unwrap_or(0);
                    neighbors
                })
                .ok_or_else(|| PgmError::InvalidGraph("No variables to eliminate".to_string()))?
                .clone();

            order.push(min_var.clone());
            remaining.remove(&min_var);

            // Update adjacency after elimination
            self.update_adjacency_after_elimination(&mut adjacency, &min_var);
        }

        Ok(order)
    }

    /// Max-cardinality search: Greedy algorithm that produces good orderings.
    ///
    /// This algorithm iteratively selects variables with maximum cardinality
    /// (number of already-selected neighbors).
    fn max_cardinality_search(&self, graph: &FactorGraph, vars: &[String]) -> Result<Vec<String>> {
        let mut remaining: HashSet<String> = vars.iter().cloned().collect();
        let mut order = Vec::new();
        let mut cardinality: HashMap<String, usize> = HashMap::new();

        // Initialize cardinality to 0
        for var in vars {
            cardinality.insert(var.clone(), 0);
        }

        // Build adjacency graph
        let adjacency = self.build_adjacency_graph(graph, &remaining)?;

        while !remaining.is_empty() {
            // Find variable with maximum cardinality
            let max_var = remaining
                .iter()
                .max_by_key(|v| cardinality.get(*v).copied().unwrap_or(0))
                .ok_or_else(|| PgmError::InvalidGraph("No variables to eliminate".to_string()))?
                .clone();

            order.push(max_var.clone());
            remaining.remove(&max_var);

            // Update cardinality of neighbors
            if let Some(neighbors) = adjacency.get(&max_var) {
                for neighbor in neighbors {
                    if remaining.contains(neighbor) {
                        *cardinality.entry(neighbor.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        Ok(order)
    }

    /// Build adjacency graph from factor graph.
    fn build_adjacency_graph(
        &self,
        graph: &FactorGraph,
        vars: &HashSet<String>,
    ) -> Result<HashMap<String, HashSet<String>>> {
        let mut adjacency: HashMap<String, HashSet<String>> = HashMap::new();

        // Initialize empty sets
        for var in vars {
            adjacency.insert(var.clone(), HashSet::new());
        }

        // Add edges based on factors
        for factor_id in graph.factor_ids() {
            if let Some(factor) = graph.get_factor(factor_id) {
                let factor_vars: Vec<String> = factor
                    .variables
                    .iter()
                    .filter(|v| vars.contains(*v))
                    .cloned()
                    .collect();

                // Connect all pairs of variables in the factor
                for i in 0..factor_vars.len() {
                    for j in (i + 1)..factor_vars.len() {
                        let v1 = &factor_vars[i];
                        let v2 = &factor_vars[j];

                        adjacency.entry(v1.clone()).or_default().insert(v2.clone());
                        adjacency.entry(v2.clone()).or_default().insert(v1.clone());
                    }
                }
            }
        }

        Ok(adjacency)
    }

    /// Compute fill for eliminating a variable.
    ///
    /// Fill is the number of new edges that would be created.
    fn compute_fill(&self, adjacency: &HashMap<String, HashSet<String>>, var: &str) -> usize {
        let neighbors = match adjacency.get(var) {
            Some(n) => n,
            None => return 0,
        };

        if neighbors.is_empty() {
            return 0;
        }

        // Count pairs of neighbors that are not already connected
        let mut fill = 0;
        let neighbors_vec: Vec<_> = neighbors.iter().collect();

        for i in 0..neighbors_vec.len() {
            for j in (i + 1)..neighbors_vec.len() {
                let v1 = neighbors_vec[i];
                let v2 = neighbors_vec[j];

                // Check if edge exists
                if let Some(adj_v1) = adjacency.get(v1) {
                    if !adj_v1.contains(v2) {
                        fill += 1;
                    }
                }
            }
        }

        fill
    }

    /// Update adjacency graph after eliminating a variable.
    fn update_adjacency_after_elimination(
        &self,
        adjacency: &mut HashMap<String, HashSet<String>>,
        var: &str,
    ) {
        let neighbors = match adjacency.remove(var) {
            Some(n) => n,
            None => return,
        };

        // Remove var from all neighbor lists
        for neighbor in &neighbors {
            if let Some(adj) = adjacency.get_mut(neighbor) {
                adj.remove(var);
            }
        }

        // Connect all pairs of neighbors (create fill edges)
        let neighbors_vec: Vec<_> = neighbors.iter().cloned().collect();
        for i in 0..neighbors_vec.len() {
            for j in (i + 1)..neighbors_vec.len() {
                let v1 = &neighbors_vec[i];
                let v2 = &neighbors_vec[j];

                // Add edge v1 <-> v2
                if let Some(adj_v1) = adjacency.get_mut(v1) {
                    adj_v1.insert(v2.clone());
                }
                if let Some(adj_v2) = adjacency.get_mut(v2) {
                    adj_v2.insert(v1.clone());
                }
            }
        }
    }

    /// Compute weights for variables based on factor sizes.
    fn compute_variable_weights(
        &self,
        graph: &FactorGraph,
        vars: &[String],
    ) -> Result<HashMap<String, usize>> {
        let mut weights = HashMap::new();

        for var in vars {
            let mut weight = 1;

            if let Some(factors) = graph.get_adjacent_factors(var) {
                for factor_id in factors {
                    if let Some(factor) = graph.get_factor(factor_id) {
                        // Weight by product of variable cardinalities
                        for factor_var in &factor.variables {
                            if let Some(var_node) = graph.get_variable(factor_var) {
                                weight *= var_node.cardinality;
                            }
                        }
                    }
                }
            }

            weights.insert(var.clone(), weight);
        }

        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Factor;
    use scirs2_core::ndarray::Array;

    fn create_test_graph() -> FactorGraph {
        let mut graph = FactorGraph::new();

        // Create a simple chain: X - Y - Z
        graph.add_variable_with_card("X".to_string(), "Domain".to_string(), 2);
        graph.add_variable_with_card("Y".to_string(), "Domain".to_string(), 2);
        graph.add_variable_with_card("Z".to_string(), "Domain".to_string(), 2);

        let f_xy = Factor::new(
            "f_xy".to_string(),
            vec!["X".to_string(), "Y".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();

        let f_yz = Factor::new(
            "f_yz".to_string(),
            vec!["Y".to_string(), "Z".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.5, 0.6, 0.7, 0.8])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();

        graph.add_factor(f_xy).unwrap();
        graph.add_factor(f_yz).unwrap();

        graph
    }

    #[test]
    fn test_min_degree_ordering() {
        let graph = create_test_graph();
        let vars = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];

        let ordering = EliminationOrdering::new(EliminationStrategy::MinDegree);
        let order = ordering.compute_order(&graph, &vars).unwrap();

        assert_eq!(order.len(), 3);
        // X and Z have degree 1, Y has degree 2
        assert!(order[0] == "X" || order[0] == "Z");
    }

    #[test]
    fn test_min_fill_ordering() {
        let graph = create_test_graph();
        let vars = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];

        let ordering = EliminationOrdering::new(EliminationStrategy::MinFill);
        let order = ordering.compute_order(&graph, &vars).unwrap();

        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_weighted_min_fill_ordering() {
        let graph = create_test_graph();
        let vars = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];

        let ordering = EliminationOrdering::new(EliminationStrategy::WeightedMinFill);
        let order = ordering.compute_order(&graph, &vars).unwrap();

        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_max_cardinality_search() {
        let graph = create_test_graph();
        let vars = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];

        let ordering = EliminationOrdering::new(EliminationStrategy::MaxCardinalitySearch);
        let order = ordering.compute_order(&graph, &vars).unwrap();

        assert_eq!(order.len(), 3);
    }
}
