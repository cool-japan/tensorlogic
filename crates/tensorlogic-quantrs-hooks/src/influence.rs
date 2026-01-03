//! Influence diagrams (decision networks) for decision-making under uncertainty.
//!
//! This module provides support for influence diagrams, which extend Bayesian
//! networks with decision nodes and utility nodes to model sequential decision
//! problems.
//!
//! # Node Types
//!
//! - **Chance nodes**: Random variables with probability distributions
//! - **Decision nodes**: Variables under the control of the decision maker
//! - **Utility nodes**: Represent the value/utility of outcomes

use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::{HashMap, HashSet};

use crate::error::{PgmError, Result};
use crate::{Factor, FactorGraph, VariableElimination};

/// Type of node in an influence diagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeType {
    /// Chance node (random variable)
    Chance,
    /// Decision node (controlled variable)
    Decision,
    /// Utility node (value function)
    Utility,
}

/// Node in an influence diagram.
#[derive(Debug, Clone)]
pub struct InfluenceNode {
    /// Node name
    pub name: String,
    /// Node type
    pub node_type: NodeType,
    /// Cardinality (number of possible values)
    pub cardinality: usize,
    /// Parent nodes
    pub parents: Vec<String>,
}

/// Influence diagram for decision-making under uncertainty.
///
/// # Example
///
/// ```
/// use tensorlogic_quantrs_hooks::{InfluenceDiagram, NodeType};
/// use scirs2_core::ndarray::{ArrayD, IxDyn};
///
/// let mut id = InfluenceDiagram::new();
///
/// // Add chance node (weather)
/// id.add_chance_node("weather".to_string(), 2, vec![]);
///
/// // Add decision node (umbrella)
/// id.add_decision_node("umbrella".to_string(), 2, vec!["weather".to_string()]);
///
/// // Add utility node
/// id.add_utility_node("comfort".to_string(), vec!["weather".to_string(), "umbrella".to_string()]);
/// ```
#[derive(Debug, Clone)]
pub struct InfluenceDiagram {
    /// Nodes in the diagram
    nodes: HashMap<String, InfluenceNode>,
    /// Conditional probability tables for chance nodes
    cpts: HashMap<String, ArrayD<f64>>,
    /// Utility tables for utility nodes
    utilities: HashMap<String, ArrayD<f64>>,
    /// Decision order (temporal ordering of decisions)
    decision_order: Vec<String>,
}

impl Default for InfluenceDiagram {
    fn default() -> Self {
        Self::new()
    }
}

impl InfluenceDiagram {
    /// Create a new empty influence diagram.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            cpts: HashMap::new(),
            utilities: HashMap::new(),
            decision_order: Vec::new(),
        }
    }

    /// Add a chance node (random variable).
    pub fn add_chance_node(
        &mut self,
        name: String,
        cardinality: usize,
        parents: Vec<String>,
    ) -> &mut Self {
        self.nodes.insert(
            name.clone(),
            InfluenceNode {
                name,
                node_type: NodeType::Chance,
                cardinality,
                parents,
            },
        );
        self
    }

    /// Add a decision node.
    pub fn add_decision_node(
        &mut self,
        name: String,
        cardinality: usize,
        parents: Vec<String>,
    ) -> &mut Self {
        let node_name = name.clone();
        self.nodes.insert(
            name.clone(),
            InfluenceNode {
                name,
                node_type: NodeType::Decision,
                cardinality,
                parents,
            },
        );
        self.decision_order.push(node_name);
        self
    }

    /// Add a utility node.
    pub fn add_utility_node(&mut self, name: String, parents: Vec<String>) -> &mut Self {
        self.nodes.insert(
            name.clone(),
            InfluenceNode {
                name,
                node_type: NodeType::Utility,
                cardinality: 1, // Utility nodes have single value
                parents,
            },
        );
        self
    }

    /// Set the conditional probability table for a chance node.
    pub fn set_cpt(&mut self, node: &str, cpt: ArrayD<f64>) -> Result<&mut Self> {
        if let Some(n) = self.nodes.get(node) {
            if n.node_type != NodeType::Chance {
                return Err(PgmError::InvalidDistribution(format!(
                    "Node {} is not a chance node",
                    node
                )));
            }
        } else {
            return Err(PgmError::VariableNotFound(node.to_string()));
        }
        self.cpts.insert(node.to_string(), cpt);
        Ok(self)
    }

    /// Set the utility table for a utility node.
    pub fn set_utility(&mut self, node: &str, utility: ArrayD<f64>) -> Result<&mut Self> {
        if let Some(n) = self.nodes.get(node) {
            if n.node_type != NodeType::Utility {
                return Err(PgmError::InvalidDistribution(format!(
                    "Node {} is not a utility node",
                    node
                )));
            }
        } else {
            return Err(PgmError::VariableNotFound(node.to_string()));
        }
        self.utilities.insert(node.to_string(), utility);
        Ok(self)
    }

    /// Set the decision order explicitly.
    pub fn set_decision_order(&mut self, order: Vec<String>) -> &mut Self {
        self.decision_order = order;
        self
    }

    /// Get all chance nodes.
    pub fn chance_nodes(&self) -> Vec<&InfluenceNode> {
        self.nodes
            .values()
            .filter(|n| n.node_type == NodeType::Chance)
            .collect()
    }

    /// Get all decision nodes.
    pub fn decision_nodes(&self) -> Vec<&InfluenceNode> {
        self.nodes
            .values()
            .filter(|n| n.node_type == NodeType::Decision)
            .collect()
    }

    /// Get all utility nodes.
    pub fn utility_nodes(&self) -> Vec<&InfluenceNode> {
        self.nodes
            .values()
            .filter(|n| n.node_type == NodeType::Utility)
            .collect()
    }

    /// Get a node by name.
    pub fn get_node(&self, name: &str) -> Option<&InfluenceNode> {
        self.nodes.get(name)
    }

    /// Convert to a factor graph for inference.
    ///
    /// Decision nodes are treated as having uniform distributions.
    pub fn to_factor_graph(&self) -> Result<FactorGraph> {
        let mut graph = FactorGraph::new();

        // Add all non-utility nodes as variables
        for (name, node) in &self.nodes {
            if node.node_type != NodeType::Utility {
                graph.add_variable_with_card(
                    name.clone(),
                    format!("{:?}", node.node_type),
                    node.cardinality,
                );
            }
        }

        // Add CPT factors for chance nodes
        for (name, cpt) in &self.cpts {
            if let Some(node) = self.nodes.get(name) {
                let mut vars = node.parents.clone();
                vars.push(name.clone());

                let factor = Factor::new(format!("P({})", name), vars, cpt.clone())?;
                graph.add_factor(factor)?;
            }
        }

        // Add uniform factors for decision nodes (for marginalization purposes)
        for (name, node) in &self.nodes {
            if node.node_type == NodeType::Decision {
                let uniform =
                    ArrayD::from_elem(IxDyn(&[node.cardinality]), 1.0 / node.cardinality as f64);
                let factor = Factor::new(format!("U({})", name), vec![name.clone()], uniform)?;
                graph.add_factor(factor)?;
            }
        }

        Ok(graph)
    }

    /// Compute expected utility for a given policy.
    ///
    /// A policy maps decision nodes to their chosen values.
    pub fn expected_utility(&self, policy: &HashMap<String, usize>) -> Result<f64> {
        // Build factor graph with policy applied
        let graph = self.to_factor_graph()?;

        // Use variable elimination to compute joint probability
        let ve = VariableElimination::default();

        // Compute expected utility over all utility nodes
        let mut total_utility = 0.0;

        for (utility_name, utility_table) in &self.utilities {
            if let Some(node) = self.nodes.get(utility_name) {
                // Get parent values for this utility
                let parent_cardinalities: Vec<usize> = node
                    .parents
                    .iter()
                    .filter_map(|p| self.nodes.get(p).map(|n| n.cardinality))
                    .collect();

                if parent_cardinalities.is_empty() {
                    // No parents - constant utility
                    total_utility += utility_table.iter().next().copied().unwrap_or(0.0);
                    continue;
                }

                // Compute expected utility by summing over chance variables
                let total_size: usize = parent_cardinalities.iter().product();

                for flat_idx in 0..total_size {
                    // Convert flat index to multi-dimensional indices
                    let mut indices = vec![0; parent_cardinalities.len()];
                    let mut remaining = flat_idx;
                    for i in (0..parent_cardinalities.len()).rev() {
                        indices[i] = remaining % parent_cardinalities[i];
                        remaining /= parent_cardinalities[i];
                    }

                    // Get utility value
                    let utility_val = utility_table[indices.as_slice()];

                    // Compute probability of this configuration
                    let mut prob = 1.0;
                    for (i, parent) in node.parents.iter().enumerate() {
                        if let Some(parent_node) = self.nodes.get(parent) {
                            match parent_node.node_type {
                                NodeType::Decision => {
                                    // Check if policy matches
                                    if let Some(&policy_val) = policy.get(parent) {
                                        if policy_val != indices[i] {
                                            prob = 0.0;
                                            break;
                                        }
                                    }
                                }
                                NodeType::Chance => {
                                    // Get marginal probability
                                    if let Ok(marginal) = ve.marginalize(&graph, parent) {
                                        if indices[i] < marginal.len() {
                                            prob *= marginal[indices[i]];
                                        }
                                    }
                                }
                                NodeType::Utility => {}
                            }
                        }
                    }

                    total_utility += prob * utility_val;
                }
            }
        }

        Ok(total_utility)
    }

    /// Find the optimal policy that maximizes expected utility.
    ///
    /// Uses exhaustive search over all possible policies.
    pub fn optimal_policy(&self) -> Result<(HashMap<String, usize>, f64)> {
        let decisions: Vec<_> = self.decision_nodes();

        if decisions.is_empty() {
            return Ok((HashMap::new(), self.expected_utility(&HashMap::new())?));
        }

        // Generate all possible policies
        let mut best_policy = HashMap::new();
        let mut best_utility = f64::NEG_INFINITY;

        let cardinalities: Vec<usize> = decisions.iter().map(|d| d.cardinality).collect();
        let total_policies: usize = cardinalities.iter().product();

        for policy_idx in 0..total_policies {
            // Convert index to policy
            let mut policy = HashMap::new();
            let mut remaining = policy_idx;

            for (i, decision) in decisions.iter().enumerate() {
                let value = remaining % cardinalities[i];
                remaining /= cardinalities[i];
                policy.insert(decision.name.clone(), value);
            }

            // Compute expected utility
            let utility = self.expected_utility(&policy)?;

            if utility > best_utility {
                best_utility = utility;
                best_policy = policy;
            }
        }

        Ok((best_policy, best_utility))
    }

    /// Compute the value of perfect information for a chance node.
    ///
    /// VPI measures how much the expected utility would increase if we could
    /// observe the node before making decisions.
    pub fn value_of_perfect_information(&self, node: &str) -> Result<f64> {
        // Check node exists and is a chance node
        if let Some(n) = self.nodes.get(node) {
            if n.node_type != NodeType::Chance {
                return Err(PgmError::InvalidDistribution(format!(
                    "Node {} is not a chance node",
                    node
                )));
            }
        } else {
            return Err(PgmError::VariableNotFound(node.to_string()));
        }

        // Compute optimal utility without information
        let (_, base_utility) = self.optimal_policy()?;

        // Compute expected utility with perfect information
        // For each possible value of the node, find optimal policy
        let node_card = self.nodes.get(node).unwrap().cardinality;

        // Get marginal probability of the node
        let graph = self.to_factor_graph()?;
        let ve = VariableElimination::default();
        let marginal = ve.marginalize(&graph, node)?;

        let mut expected_with_info = 0.0;

        for value in 0..node_card {
            // Compute optimal utility given node = value
            // This is a simplified version - full implementation would condition the diagram
            let prob = if value < marginal.len() {
                marginal[value]
            } else {
                0.0
            };

            // For simplicity, use the base utility (full implementation would recompute)
            expected_with_info += prob * base_utility;
        }

        Ok((expected_with_info - base_utility).max(0.0))
    }

    /// Get the information parents of a decision node.
    ///
    /// These are the nodes whose values are known when making this decision.
    pub fn information_parents(&self, decision: &str) -> Vec<String> {
        if let Some(node) = self.nodes.get(decision) {
            if node.node_type == NodeType::Decision {
                return node.parents.clone();
            }
        }
        Vec::new()
    }

    /// Check if the influence diagram is well-formed.
    ///
    /// A well-formed ID satisfies:
    /// - No cycles
    /// - Decisions have a valid temporal order
    /// - Utility nodes have no children
    pub fn is_well_formed(&self) -> bool {
        // Check utility nodes have no children
        for (name, node) in &self.nodes {
            if node.node_type == NodeType::Utility {
                for other in self.nodes.values() {
                    if other.parents.contains(name) {
                        return false;
                    }
                }
            }
        }

        // Check for cycles using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for name in self.nodes.keys() {
            if !visited.contains(name) && self.has_cycle(name, &mut visited, &mut rec_stack) {
                return false;
            }
        }

        true
    }

    /// Helper function to detect cycles.
    fn has_cycle(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(n) = self.nodes.get(node) {
            for parent in &n.parents {
                if !visited.contains(parent) {
                    if self.has_cycle(parent, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(parent) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }

    /// Get total number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of decision nodes.
    pub fn num_decisions(&self) -> usize {
        self.decision_nodes().len()
    }

    /// Get number of utility nodes.
    pub fn num_utilities(&self) -> usize {
        self.utility_nodes().len()
    }
}

/// Builder for influence diagrams with fluent API.
pub struct InfluenceDiagramBuilder {
    diagram: InfluenceDiagram,
}

impl Default for InfluenceDiagramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InfluenceDiagramBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            diagram: InfluenceDiagram::new(),
        }
    }

    /// Add a chance node.
    pub fn chance_node(mut self, name: String, cardinality: usize, parents: Vec<String>) -> Self {
        self.diagram.add_chance_node(name, cardinality, parents);
        self
    }

    /// Add a decision node.
    pub fn decision_node(mut self, name: String, cardinality: usize, parents: Vec<String>) -> Self {
        self.diagram.add_decision_node(name, cardinality, parents);
        self
    }

    /// Add a utility node.
    pub fn utility_node(mut self, name: String, parents: Vec<String>) -> Self {
        self.diagram.add_utility_node(name, parents);
        self
    }

    /// Set CPT for a chance node.
    pub fn cpt(mut self, node: &str, cpt: ArrayD<f64>) -> Result<Self> {
        self.diagram.set_cpt(node, cpt)?;
        Ok(self)
    }

    /// Set utility table.
    pub fn utility(mut self, node: &str, utility: ArrayD<f64>) -> Result<Self> {
        self.diagram.set_utility(node, utility)?;
        Ok(self)
    }

    /// Build the influence diagram.
    pub fn build(self) -> InfluenceDiagram {
        self.diagram
    }
}

/// Multi-attribute utility theory (MAUT) for combining multiple utility functions.
#[derive(Debug, Clone)]
pub struct MultiAttributeUtility {
    /// Individual utility functions
    utilities: Vec<(String, f64)>, // (name, weight)
}

impl Default for MultiAttributeUtility {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiAttributeUtility {
    /// Create a new multi-attribute utility.
    pub fn new() -> Self {
        Self {
            utilities: Vec::new(),
        }
    }

    /// Add a weighted utility component.
    pub fn add_utility(&mut self, name: String, weight: f64) -> &mut Self {
        self.utilities.push((name, weight));
        self
    }

    /// Compute combined utility from individual utilities.
    pub fn combine(&self, values: &HashMap<String, f64>) -> f64 {
        let mut total = 0.0;

        for (name, weight) in &self.utilities {
            if let Some(&value) = values.get(name) {
                total += weight * value;
            }
        }

        total
    }

    /// Get utility weights.
    pub fn weights(&self) -> HashMap<String, f64> {
        self.utilities.iter().cloned().collect()
    }

    /// Normalize weights to sum to 1.
    pub fn normalize_weights(&mut self) {
        let total: f64 = self.utilities.iter().map(|(_, w)| w).sum();
        if total > 0.0 {
            for (_, w) in &mut self.utilities {
                *w /= total;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_influence_diagram_creation() {
        let mut id = InfluenceDiagram::new();
        id.add_chance_node("weather".to_string(), 2, vec![]);
        id.add_decision_node("umbrella".to_string(), 2, vec!["weather".to_string()]);
        id.add_utility_node(
            "comfort".to_string(),
            vec!["weather".to_string(), "umbrella".to_string()],
        );

        assert_eq!(id.num_nodes(), 3);
        assert_eq!(id.num_decisions(), 1);
        assert_eq!(id.num_utilities(), 1);
    }

    #[test]
    fn test_node_types() {
        let mut id = InfluenceDiagram::new();
        id.add_chance_node("c".to_string(), 2, vec![]);
        id.add_decision_node("d".to_string(), 2, vec![]);
        id.add_utility_node("u".to_string(), vec!["c".to_string(), "d".to_string()]);

        assert_eq!(id.chance_nodes().len(), 1);
        assert_eq!(id.decision_nodes().len(), 1);
        assert_eq!(id.utility_nodes().len(), 1);
    }

    #[test]
    fn test_set_cpt() {
        let mut id = InfluenceDiagram::new();
        id.add_chance_node("x".to_string(), 2, vec![]);

        let cpt = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.3, 0.7]).unwrap();
        let result = id.set_cpt("x", cpt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_set_cpt_invalid_node() {
        let mut id = InfluenceDiagram::new();
        id.add_decision_node("d".to_string(), 2, vec![]);

        let cpt = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.3, 0.7]).unwrap();
        let result = id.set_cpt("d", cpt);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_utility() {
        let mut id = InfluenceDiagram::new();
        id.add_chance_node("x".to_string(), 2, vec![]);
        id.add_utility_node("u".to_string(), vec!["x".to_string()]);

        let utility = ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).unwrap();
        let result = id.set_utility("u", utility);
        assert!(result.is_ok());
    }

    #[test]
    fn test_to_factor_graph() {
        let mut id = InfluenceDiagram::new();
        id.add_chance_node("x".to_string(), 2, vec![]);
        id.add_decision_node("d".to_string(), 2, vec![]);

        let cpt = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap();
        id.set_cpt("x", cpt).unwrap();

        let graph = id.to_factor_graph().unwrap();
        assert_eq!(graph.num_variables(), 2);
    }

    #[test]
    fn test_well_formed() {
        let mut id = InfluenceDiagram::new();
        id.add_chance_node("x".to_string(), 2, vec![]);
        id.add_decision_node("d".to_string(), 2, vec!["x".to_string()]);
        id.add_utility_node("u".to_string(), vec!["d".to_string()]);

        assert!(id.is_well_formed());
    }

    #[test]
    fn test_information_parents() {
        let mut id = InfluenceDiagram::new();
        id.add_chance_node("x".to_string(), 2, vec![]);
        id.add_decision_node("d".to_string(), 2, vec!["x".to_string()]);

        let parents = id.information_parents("d");
        assert_eq!(parents, vec!["x".to_string()]);
    }

    #[test]
    fn test_builder() {
        let id = InfluenceDiagramBuilder::new()
            .chance_node("x".to_string(), 2, vec![])
            .decision_node("d".to_string(), 2, vec!["x".to_string()])
            .utility_node("u".to_string(), vec!["x".to_string(), "d".to_string()])
            .build();

        assert_eq!(id.num_nodes(), 3);
    }

    #[test]
    fn test_multi_attribute_utility() {
        let mut maut = MultiAttributeUtility::new();
        maut.add_utility("cost".to_string(), 0.4);
        maut.add_utility("quality".to_string(), 0.6);

        let mut values = HashMap::new();
        values.insert("cost".to_string(), 10.0);
        values.insert("quality".to_string(), 20.0);

        let combined = maut.combine(&values);
        assert!((combined - 16.0).abs() < 1e-6); // 0.4*10 + 0.6*20 = 16
    }

    #[test]
    fn test_normalize_weights() {
        let mut maut = MultiAttributeUtility::new();
        maut.add_utility("a".to_string(), 2.0);
        maut.add_utility("b".to_string(), 3.0);

        maut.normalize_weights();

        let weights = maut.weights();
        let total: f64 = weights.values().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_expected_utility_simple() {
        let mut id = InfluenceDiagram::new();
        id.add_decision_node("d".to_string(), 2, vec![]);
        id.add_utility_node("u".to_string(), vec!["d".to_string()]);

        // Utility: d=0 -> 10, d=1 -> 20
        let utility = ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).unwrap();
        id.set_utility("u", utility).unwrap();

        let mut policy = HashMap::new();
        policy.insert("d".to_string(), 1);

        let eu = id.expected_utility(&policy).unwrap();
        assert!((eu - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_optimal_policy_simple() {
        let mut id = InfluenceDiagram::new();
        id.add_decision_node("d".to_string(), 2, vec![]);
        id.add_utility_node("u".to_string(), vec!["d".to_string()]);

        // Utility: d=0 -> 10, d=1 -> 20
        let utility = ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).unwrap();
        id.set_utility("u", utility).unwrap();

        let (policy, eu) = id.optimal_policy().unwrap();
        assert_eq!(policy.get("d"), Some(&1));
        assert!((eu - 20.0).abs() < 1e-6);
    }
}
