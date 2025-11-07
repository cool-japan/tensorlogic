//! Factor graph representation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{PgmError, Result};
use crate::factor::Factor;

/// Variable node in a factor graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariableNode {
    /// Variable name
    pub name: String,
    /// Domain of the variable
    pub domain: String,
    /// Cardinality (number of possible values)
    pub cardinality: usize,
}

/// Factor node in a factor graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FactorNode {
    /// Factor ID
    pub id: String,
    /// Connected variable names
    pub variables: Vec<String>,
}

/// Factor graph representation for PGM.
#[derive(Clone, Debug)]
pub struct FactorGraph {
    /// Variable nodes
    variables: HashMap<String, VariableNode>,
    /// Factor nodes
    factors: HashMap<String, Factor>,
    /// Adjacency: variable -> connected factors
    var_to_factors: HashMap<String, Vec<String>>,
    /// Adjacency: factor -> connected variables
    factor_to_vars: HashMap<String, Vec<String>>,
}

impl FactorGraph {
    /// Create a new empty factor graph.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            factors: HashMap::new(),
            var_to_factors: HashMap::new(),
            factor_to_vars: HashMap::new(),
        }
    }

    /// Add a variable to the graph.
    pub fn add_variable(&mut self, name: String, domain: String) {
        let node = VariableNode {
            name: name.clone(),
            domain,
            cardinality: 2, // Default binary
        };
        self.variables.insert(name.clone(), node);
        self.var_to_factors.entry(name).or_default();
    }

    /// Add a variable with specific cardinality.
    pub fn add_variable_with_card(&mut self, name: String, domain: String, cardinality: usize) {
        let node = VariableNode {
            name: name.clone(),
            domain,
            cardinality,
        };
        self.variables.insert(name.clone(), node);
        self.var_to_factors.entry(name).or_default();
    }

    /// Add a factor to the graph.
    pub fn add_factor(&mut self, factor: Factor) -> Result<()> {
        let factor_id = factor.name.clone();

        // Ensure all variables exist
        for var in &factor.variables {
            if !self.variables.contains_key(var) {
                return Err(PgmError::VariableNotFound(var.clone()));
            }
        }

        // Update adjacency lists
        for var in &factor.variables {
            self.var_to_factors
                .entry(var.clone())
                .or_default()
                .push(factor_id.clone());
        }
        self.factor_to_vars
            .insert(factor_id.clone(), factor.variables.clone());

        self.factors.insert(factor_id, factor);
        Ok(())
    }

    /// Add a factor from predicate name and variables.
    pub fn add_factor_from_predicate(&mut self, name: &str, var_names: &[String]) -> Result<()> {
        // Create uniform factor
        let factor = Factor::uniform(name.to_string(), var_names.to_vec(), 2);
        self.add_factor(factor)
    }

    /// Get variable node.
    pub fn get_variable(&self, name: &str) -> Option<&VariableNode> {
        self.variables.get(name)
    }

    /// Get factor.
    pub fn get_factor(&self, id: &str) -> Option<&Factor> {
        self.factors.get(id)
    }

    /// Get factors connected to a variable.
    pub fn get_adjacent_factors(&self, var: &str) -> Option<&Vec<String>> {
        self.var_to_factors.get(var)
    }

    /// Get variables connected to a factor.
    pub fn get_adjacent_variables(&self, factor_id: &str) -> Option<&Vec<String>> {
        self.factor_to_vars.get(factor_id)
    }

    /// Get number of variables.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Get number of factors.
    pub fn num_factors(&self) -> usize {
        self.factors.len()
    }

    /// Check if graph is empty.
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty() && self.factors.is_empty()
    }

    /// Get all variable names.
    pub fn variable_names(&self) -> impl Iterator<Item = &String> {
        self.variables.keys()
    }

    /// Get all factor IDs.
    pub fn factor_ids(&self) -> impl Iterator<Item = &String> {
        self.factors.keys()
    }

    /// Get all variables as an iterator.
    pub fn variables(&self) -> impl Iterator<Item = (&String, &VariableNode)> {
        self.variables.iter()
    }

    /// Get all factors as an iterator.
    pub fn factors(&self) -> impl Iterator<Item = &Factor> {
        self.factors.values()
    }

    /// Get all factors as a vector (for external use).
    pub fn get_all_factors(&self) -> Vec<&Factor> {
        self.factors.values().collect()
    }
}

impl Default for FactorGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = FactorGraph::new();
        assert!(graph.is_empty());
    }

    #[test]
    fn test_add_variables() {
        let mut graph = FactorGraph::new();
        graph.add_variable("x".to_string(), "D1".to_string());
        graph.add_variable("y".to_string(), "D2".to_string());

        assert_eq!(graph.num_variables(), 2);
        assert!(graph.get_variable("x").is_some());
    }

    #[test]
    fn test_add_factor() {
        let mut graph = FactorGraph::new();
        graph.add_variable("x".to_string(), "D1".to_string());
        graph.add_variable("y".to_string(), "D2".to_string());

        let result = graph.add_factor_from_predicate("P", &["x".to_string(), "y".to_string()]);
        assert!(result.is_ok());
        assert_eq!(graph.num_factors(), 1);
    }

    #[test]
    fn test_adjacency() {
        let mut graph = FactorGraph::new();
        graph.add_variable("x".to_string(), "D1".to_string());
        graph.add_variable("y".to_string(), "D2".to_string());
        graph
            .add_factor_from_predicate("P", &["x".to_string(), "y".to_string()])
            .unwrap();

        let adjacent = graph.get_adjacent_factors("x");
        assert!(adjacent.is_some());
        assert_eq!(adjacent.unwrap().len(), 1);
    }
}
