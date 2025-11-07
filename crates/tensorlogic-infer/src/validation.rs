//! Graph validation utilities for ensuring well-formed execution graphs.

use std::collections::HashSet;

use tensorlogic_ir::{EinsumGraph, OpType};

use crate::error::ExecutorError;

/// Validation result with detailed diagnostics
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn new() -> Self {
        ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: impl Into<String>) {
        self.is_valid = false;
        self.errors.push(error.into());
    }

    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    pub fn merge(&mut self, other: ValidationResult) {
        self.is_valid &= other.is_valid;
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }

    pub fn summary(&self) -> String {
        let mut summary = String::new();
        if self.is_valid {
            summary.push_str("✓ Graph is valid\n");
        } else {
            summary.push_str("✗ Graph validation failed\n");
        }

        if !self.errors.is_empty() {
            summary.push_str("\nErrors:\n");
            for error in &self.errors {
                summary.push_str(&format!("  - {}\n", error));
            }
        }

        if !self.warnings.is_empty() {
            summary.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                summary.push_str(&format!("  - {}\n", warning));
            }
        }

        summary
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph validator
pub struct GraphValidator;

impl GraphValidator {
    pub fn new() -> Self {
        GraphValidator
    }

    /// Validate a complete execution graph
    pub fn validate(&self, graph: &EinsumGraph) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Check if graph is empty
        if graph.nodes.is_empty() {
            result.add_warning("Graph has no computation nodes");
        }

        // Validate tensor indices
        self.validate_tensor_indices(graph, &mut result);

        // Validate node dependencies
        self.validate_dependencies(graph, &mut result);

        // Validate operations
        self.validate_operations(graph, &mut result);

        // Check for cycles (DAG property)
        self.validate_dag(graph, &mut result);

        result
    }

    fn validate_tensor_indices(&self, graph: &EinsumGraph, result: &mut ValidationResult) {
        let num_tensors = graph.tensors.len();

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &input_idx in &node.inputs {
                // Input indices should be either:
                // 1. Input tensors (< num_tensors)
                // 2. Outputs from previous nodes (>= num_tensors && < num_tensors + node_idx)
                let max_valid_idx = num_tensors + node_idx;

                if input_idx >= max_valid_idx {
                    result.add_error(format!(
                        "Node {} references invalid tensor index {} (max valid: {})",
                        node_idx, input_idx, max_valid_idx
                    ));
                }
            }
        }
    }

    fn validate_dependencies(&self, graph: &EinsumGraph, result: &mut ValidationResult) {
        let num_tensors = graph.tensors.len();

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // Check that all dependencies come from earlier in the graph
            for &input_idx in &node.inputs {
                if input_idx >= num_tensors {
                    let dep_node_idx = input_idx - num_tensors;
                    if dep_node_idx >= node_idx {
                        result.add_error(format!(
                            "Node {} has forward dependency on node {}",
                            node_idx, dep_node_idx
                        ));
                    }
                }
            }
        }
    }

    fn validate_operations(&self, graph: &EinsumGraph, result: &mut ValidationResult) {
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            match &node.op {
                OpType::Einsum { spec } => {
                    if spec.is_empty() {
                        result.add_error(format!("Node {} has empty einsum spec", node_idx));
                    }
                    if node.inputs.is_empty() {
                        result.add_error(format!("Node {} einsum has no inputs", node_idx));
                    }
                }
                OpType::ElemUnary { op: _ } => {
                    if node.inputs.len() != 1 {
                        result.add_error(format!(
                            "Node {} unary operation requires exactly 1 input, got {}",
                            node_idx,
                            node.inputs.len()
                        ));
                    }
                }
                OpType::ElemBinary { op: _ } => {
                    if node.inputs.len() != 2 {
                        result.add_error(format!(
                            "Node {} binary operation requires exactly 2 inputs, got {}",
                            node_idx,
                            node.inputs.len()
                        ));
                    }
                }
                OpType::Reduce { op: _, axes } => {
                    if node.inputs.len() != 1 {
                        result.add_error(format!(
                            "Node {} reduce operation requires exactly 1 input, got {}",
                            node_idx,
                            node.inputs.len()
                        ));
                    }
                    if axes.is_empty() {
                        result.add_warning(format!(
                            "Node {} reduce operation has no axes (identity operation)",
                            node_idx
                        ));
                    }
                }
            }
        }
    }

    fn validate_dag(&self, graph: &EinsumGraph, result: &mut ValidationResult) {
        // Build adjacency list
        let num_nodes = graph.nodes.len();
        let num_tensors = graph.tensors.len();

        // Detect cycles using DFS
        let mut visited = vec![false; num_nodes];
        let mut rec_stack = vec![false; num_nodes];

        for node_idx in 0..num_nodes {
            if !visited[node_idx]
                && has_cycle_helper(node_idx, graph, num_tensors, &mut visited, &mut rec_stack)
            {
                result.add_error("Graph contains a cycle (not a DAG)");
                break;
            }
        }
    }

    /// Quick validation that returns an error if graph is invalid
    pub fn validate_or_error(&self, graph: &EinsumGraph) -> Result<(), ExecutorError> {
        let result = self.validate(graph);
        if result.is_valid {
            Ok(())
        } else {
            Err(ExecutorError::GraphValidationError(
                result.errors.join("; "),
            ))
        }
    }

    /// Check if graph has any unreachable nodes
    pub fn find_unreachable_nodes(&self, graph: &EinsumGraph) -> HashSet<usize> {
        let num_nodes = graph.nodes.len();
        let num_tensors = graph.tensors.len();

        let mut reachable = HashSet::new();

        // Work backwards from the last node
        if num_nodes > 0 {
            let mut to_visit = vec![num_nodes - 1];
            while let Some(node_idx) = to_visit.pop() {
                if reachable.insert(node_idx) {
                    // Add dependencies to visit list
                    for &input_idx in &graph.nodes[node_idx].inputs {
                        if input_idx >= num_tensors {
                            let dep_node_idx = input_idx - num_tensors;
                            if !reachable.contains(&dep_node_idx) {
                                to_visit.push(dep_node_idx);
                            }
                        }
                    }
                }
            }
        }

        // Return nodes that are not reachable
        (0..num_nodes)
            .filter(|idx| !reachable.contains(idx))
            .collect()
    }
}

// Helper function to detect cycles (separate to avoid clippy recursion warning)
#[allow(clippy::only_used_in_recursion)]
fn has_cycle_helper(
    node_idx: usize,
    graph: &EinsumGraph,
    num_tensors: usize,
    visited: &mut [bool],
    rec_stack: &mut [bool],
) -> bool {
    visited[node_idx] = true;
    rec_stack[node_idx] = true;

    // Check all dependencies
    for &input_idx in &graph.nodes[node_idx].inputs {
        if input_idx >= num_tensors {
            let dep_node_idx = input_idx - num_tensors;
            // Bounds check to prevent panic on invalid indices
            if dep_node_idx >= visited.len() {
                continue; // Skip invalid indices - they'll be caught by validate_tensor_indices
            }
            if !visited[dep_node_idx] {
                if has_cycle_helper(dep_node_idx, graph, num_tensors, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack[dep_node_idx] {
                return true;
            }
        }
    }

    rec_stack[node_idx] = false;
    false
}

impl Default for GraphValidator {
    fn default() -> Self {
        Self::new()
    }
}
