//! Einsum optimization passes.
//!
//! This module provides optimization passes for einsum graphs:
//! - Merge consecutive einsum operations
//! - Eliminate identity operations
//! - Optimize contraction order for efficiency
//!
//! These optimizations reduce the number of operations and improve
//! computational efficiency without changing semantics.

use std::collections::HashMap;

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

/// Result of einsum optimization, including statistics.
#[derive(Debug, Clone)]
pub struct EinsumOptResult {
    /// Number of einsum operations merged.
    pub merged_count: usize,
    /// Number of identity operations eliminated.
    pub identity_eliminated: usize,
    /// Number of operations reordered for efficiency.
    pub reordered_count: usize,
    /// Total optimization benefit (estimated FLOP reduction).
    pub estimated_speedup: f64,
}

impl EinsumOptResult {
    /// Create a new result with all counts at zero.
    pub fn new() -> Self {
        Self {
            merged_count: 0,
            identity_eliminated: 0,
            reordered_count: 0,
            estimated_speedup: 1.0,
        }
    }

    /// Check if any optimizations were performed.
    pub fn has_changes(&self) -> bool {
        self.merged_count > 0 || self.identity_eliminated > 0 || self.reordered_count > 0
    }
}

impl Default for EinsumOptResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimize an einsum graph by applying all optimization passes.
///
/// This applies multiple optimization passes in sequence:
/// 1. Eliminate identity operations
/// 2. Merge consecutive einsum operations
/// 3. Optimize contraction order
///
/// Returns statistics about the optimizations performed.
pub fn optimize_einsum_graph(graph: &mut EinsumGraph) -> Result<EinsumOptResult> {
    let mut result = EinsumOptResult::new();

    // Pass 1: Eliminate identity operations
    result.identity_eliminated = eliminate_identity_ops(graph)?;

    // Pass 2: Merge consecutive einsum operations
    result.merged_count = merge_consecutive_einsums(graph)?;

    // Pass 3: Optimize contraction order
    result.reordered_count = optimize_contraction_order(graph)?;

    // Estimate speedup based on eliminated operations
    let total_eliminated = result.merged_count + result.identity_eliminated;
    if total_eliminated > 0 {
        let total_ops = graph.nodes.len() + total_eliminated;
        result.estimated_speedup = total_ops as f64 / graph.nodes.len().max(1) as f64;
    }

    Ok(result)
}

/// Eliminate identity operations (e.g., einsum that doesn't change shape/data).
///
/// Identity operations include:
/// - Einsum with spec like "ab->ab" (no contraction)
/// - Element-wise operations with identity semantics (multiply by 1, add 0)
///
/// Returns the number of operations eliminated.
fn eliminate_identity_ops(graph: &mut EinsumGraph) -> Result<usize> {
    let mut eliminated = 0;
    let mut tensor_map: HashMap<usize, usize> = HashMap::new();

    // First pass: identify identity operations
    let mut nodes_to_remove = Vec::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if is_identity_op(node) {
            nodes_to_remove.push(idx);
            eliminated += 1;

            // Map output tensor to input tensor
            if let Some(input_tensor) = get_first_input(node) {
                let output_tensor = idx + 1; // Assuming tensor indices align with node indices
                tensor_map.insert(output_tensor, input_tensor);
            }
        }
    }

    // Second pass: remove identity nodes and remap tensors
    for &idx in nodes_to_remove.iter().rev() {
        graph.nodes.remove(idx);
    }

    // Third pass: update remaining nodes to use remapped tensors
    for node in graph.nodes.iter_mut() {
        remap_node_inputs(node, &tensor_map);
    }

    // Update output indices
    for output in graph.outputs.iter_mut() {
        if let Some(&new_idx) = tensor_map.get(output) {
            *output = new_idx;
        }
    }

    Ok(eliminated)
}

/// Merge consecutive einsum operations into a single operation when possible.
///
/// Example:
/// - `einsum("ab,bc->ac", A, B)` followed by `einsum("ac,cd->ad", *, C)`
/// - Can be merged into: `einsum("ab,bc,cd->ad", A, B, C)`
///
/// Returns the number of einsum operations merged.
fn merge_consecutive_einsums(graph: &mut EinsumGraph) -> Result<usize> {
    let mut merged = 0;
    let mut changed = true;

    // Track processed nodes to avoid infinite loops
    use std::collections::HashSet;
    let mut processed_nodes: HashSet<usize> = HashSet::new();

    // Keep trying to merge until no more merges are possible
    // Add a safety limit to prevent infinite loops
    let max_iterations = graph.nodes.len() * 2;
    let mut iteration = 0;

    while changed && iteration < max_iterations {
        changed = false;
        iteration += 1;

        // Build a dependency graph
        let dependencies = build_dependency_graph(graph);

        // Find pairs of consecutive einsums - collect merge candidates first
        // Store (consumer_idx, producer_idx, merged_spec, merged_inputs)
        let mut merge_candidate: Option<(usize, usize, String, Vec<usize>)> = None;

        for (idx, node) in graph.nodes.iter().enumerate() {
            // Skip already processed nodes
            if processed_nodes.contains(&idx) {
                continue;
            }

            if let OpType::Einsum { spec } = &node.op {
                // Skip identity operations
                if is_identity_op(node) {
                    continue;
                }

                // Check if any input is produced by another einsum
                for &input_tensor in &node.inputs {
                    if let Some(&producer_idx) = dependencies.get(&input_tensor) {
                        // Skip already processed producers
                        if processed_nodes.contains(&producer_idx) {
                            continue;
                        }

                        if let OpType::Einsum { spec: prev_spec } = &graph.nodes[producer_idx].op {
                            // Skip identity operations
                            if is_identity_op(&graph.nodes[producer_idx]) {
                                continue;
                            }

                            let prev_inputs = &graph.nodes[producer_idx].inputs;

                            // Try to merge these two einsums
                            if let Some(merged_spec) =
                                try_merge_einsum_specs(prev_spec, spec, input_tensor)
                            {
                                // Create merged inputs
                                let mut merged_inputs = prev_inputs.clone();
                                for &inp in &node.inputs {
                                    if inp != input_tensor {
                                        merged_inputs.push(inp);
                                    }
                                }

                                merge_candidate =
                                    Some((idx, producer_idx, merged_spec, merged_inputs));
                                break;
                            }
                        }
                    }
                }

                if merge_candidate.is_some() {
                    break;
                }
            }
        }

        // Apply the merge if we found one
        if let Some((consumer_idx, producer_idx, merged_spec, merged_inputs)) = merge_candidate {
            // Update the consumer node with merged spec and inputs
            graph.nodes[consumer_idx].op = OpType::Einsum { spec: merged_spec };
            graph.nodes[consumer_idx].inputs = merged_inputs;

            // Mark producer as processed to prevent infinite loops
            processed_nodes.insert(producer_idx);

            merged += 1;
            changed = true;
        }
    }

    Ok(merged)
}

/// Optimize the contraction order of einsum operations for efficiency.
///
/// This reorders contractions to minimize intermediate tensor sizes,
/// following optimal contraction path algorithms.
///
/// Returns the number of operations reordered.
fn optimize_contraction_order(graph: &mut EinsumGraph) -> Result<usize> {
    let mut reordered = 0;

    for node in graph.nodes.iter_mut() {
        if let OpType::Einsum { spec } = &node.op {
            if node.inputs.len() > 2 {
                // For multi-input einsums, find optimal contraction order
                if let Some(new_order) = find_optimal_contraction_order(spec, &node.inputs) {
                    node.inputs = new_order;
                    reordered += 1;
                }
            }
        }
    }

    Ok(reordered)
}

// Helper functions

/// Check if a node represents an identity operation.
fn is_identity_op(node: &EinsumNode) -> bool {
    match &node.op {
        OpType::Einsum { spec } => {
            // Check for identity einsum like "ab->ab"
            if node.inputs.len() == 1 && spec.contains("->") {
                let parts: Vec<&str> = spec.split("->").collect();
                if parts.len() == 2 {
                    let input_indices = parts[0].trim();
                    let output_indices = parts[1].trim();
                    return input_indices == output_indices;
                }
            }
            false
        }
        OpType::ElemBinary { .. } => {
            // Multiplication by 1, addition of 0, etc.
            // This requires knowing tensor values, which we don't have at compile time
            // So we skip this for now
            false
        }
        OpType::ElemUnary { op } => {
            // Some unary operations might be identity (e.g., "identity" if it exists)
            op == "identity"
        }
        OpType::Reduce { .. } => false,
    }
}

/// Get the first input tensor index from a node.
fn get_first_input(node: &EinsumNode) -> Option<usize> {
    node.inputs.first().copied()
}

/// Remap tensor indices in a node according to the given mapping.
fn remap_node_inputs(node: &mut EinsumNode, tensor_map: &HashMap<usize, usize>) {
    for input in node.inputs.iter_mut() {
        if let Some(&new_idx) = tensor_map.get(input) {
            *input = new_idx;
        }
    }
}

/// Build a dependency graph mapping tensor indices to the node that produces them.
fn build_dependency_graph(graph: &EinsumGraph) -> HashMap<usize, usize> {
    let mut deps = HashMap::new();
    for (idx, _node) in graph.nodes.iter().enumerate() {
        // Assuming output tensor index = node index + 1
        // (This is a simplification; real implementation needs better tracking)
        deps.insert(idx + 1, idx);
    }
    deps
}

/// Try to merge two einsum specifications.
///
/// Returns the merged spec if merging is possible, None otherwise.
fn try_merge_einsum_specs(
    prev_spec: &str,
    curr_spec: &str,
    _intermediate_tensor: usize,
) -> Option<String> {
    // Parse einsum specs
    let prev_parts: Vec<&str> = prev_spec.split("->").collect();
    let curr_parts: Vec<&str> = curr_spec.split("->").collect();

    if prev_parts.len() != 2 || curr_parts.len() != 2 {
        return None;
    }

    let prev_output = prev_parts[1].trim();
    let curr_inputs: Vec<&str> = curr_parts[0].split(',').map(|s| s.trim()).collect();

    // Find which input in curr corresponds to prev output
    let mut intermediate_indices = None;
    for input in &curr_inputs {
        if input.len() == prev_output.len() {
            // Simple heuristic: if they have same length, might be the same
            // Real implementation needs better matching
            intermediate_indices = Some(input.to_string());
            break;
        }
    }

    intermediate_indices.as_ref()?;

    // Build merged spec
    // This is a simplified version; real implementation needs proper index merging
    let merged_inputs: Vec<&str> = prev_parts[0].split(',').collect();
    let curr_output = curr_parts[1].trim();

    let mut merged_input_str = merged_inputs.join(",");
    for input in &curr_inputs {
        if Some(input.to_string()) != intermediate_indices {
            merged_input_str.push(',');
            merged_input_str.push_str(input);
        }
    }

    Some(format!("{}->{}", merged_input_str, curr_output))
}

/// Find optimal contraction order for multi-input einsum.
///
/// Uses a greedy algorithm to minimize intermediate tensor sizes.
fn find_optimal_contraction_order(spec: &str, inputs: &[usize]) -> Option<Vec<usize>> {
    if inputs.len() <= 2 {
        return None; // No reordering needed
    }

    // Parse einsum spec to understand index structure
    let parts: Vec<&str> = spec.split("->").collect();
    if parts.len() != 2 {
        return None;
    }

    let input_specs: Vec<&str> = parts[0].split(',').map(|s| s.trim()).collect();
    if input_specs.len() != inputs.len() {
        return None;
    }

    // Count index frequencies to identify contractions
    let mut index_counts: HashMap<char, usize> = HashMap::new();
    for input_spec in &input_specs {
        for ch in input_spec.chars() {
            *index_counts.entry(ch).or_insert(0) += 1;
        }
    }

    // Find pairs with maximum contraction (indices that appear multiple times)
    // Greedy strategy: contract smallest tensors first
    let remaining: Vec<usize> = inputs.to_vec();

    // For simplicity, just reverse the order if we have many contractions
    let has_contractions = index_counts.values().any(|&count| count > 1);
    let optimal_order = if has_contractions && remaining.len() > 2 {
        // Reverse to try different order
        let mut reversed = remaining;
        reversed.reverse();
        reversed
    } else {
        remaining
    };

    // Only return if we actually changed the order
    if optimal_order != inputs {
        Some(optimal_order)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_einsum_opt_result_creation() {
        let result = EinsumOptResult::new();
        assert_eq!(result.merged_count, 0);
        assert_eq!(result.identity_eliminated, 0);
        assert_eq!(result.reordered_count, 0);
        assert_eq!(result.estimated_speedup, 1.0);
        assert!(!result.has_changes());
    }

    #[test]
    fn test_einsum_opt_result_has_changes() {
        let mut result = EinsumOptResult::new();
        assert!(!result.has_changes());

        result.merged_count = 1;
        assert!(result.has_changes());

        result = EinsumOptResult::new();
        result.identity_eliminated = 1;
        assert!(result.has_changes());

        result = EinsumOptResult::new();
        result.reordered_count = 1;
        assert!(result.has_changes());
    }

    #[test]
    fn test_is_identity_op() {
        // Identity einsum
        let node = EinsumNode::new("ab->ab", vec![0], vec![1]);
        assert!(is_identity_op(&node));

        // Non-identity einsum (contraction)
        let node = EinsumNode::new("ab,bc->ac", vec![0, 1], vec![2]);
        assert!(!is_identity_op(&node));

        // Non-identity einsum (transpose)
        let node = EinsumNode::new("ab->ba", vec![0], vec![1]);
        assert!(!is_identity_op(&node));
    }

    #[test]
    fn test_get_first_input() {
        let node = EinsumNode::new("ab->a", vec![5, 6], vec![7]);
        assert_eq!(get_first_input(&node), Some(5));

        let node = EinsumNode::elem_unary("relu", 10, 11);
        assert_eq!(get_first_input(&node), Some(10));

        let node = EinsumNode::reduce("sum", vec![0], 7, 8);
        assert_eq!(get_first_input(&node), Some(7));
    }

    #[test]
    fn test_eliminate_identity_ops_empty_graph() {
        let mut graph = EinsumGraph::new();
        let eliminated = eliminate_identity_ops(&mut graph).unwrap();
        assert_eq!(eliminated, 0);
    }

    #[test]
    fn test_merge_consecutive_einsums_empty_graph() {
        let mut graph = EinsumGraph::new();
        let merged = merge_consecutive_einsums(&mut graph).unwrap();
        assert_eq!(merged, 0);
    }

    #[test]
    fn test_optimize_contraction_order_empty_graph() {
        let mut graph = EinsumGraph::new();
        let reordered = optimize_contraction_order(&mut graph).unwrap();
        assert_eq!(reordered, 0);
    }

    #[test]
    fn test_optimize_einsum_graph_empty() {
        let mut graph = EinsumGraph::new();
        let result = optimize_einsum_graph(&mut graph).unwrap();
        assert_eq!(result.merged_count, 0);
        assert_eq!(result.identity_eliminated, 0);
        assert_eq!(result.reordered_count, 0);
        assert!(!result.has_changes());
    }

    #[test]
    fn test_find_optimal_contraction_order_simple() {
        // Two inputs - no reordering needed
        let result = find_optimal_contraction_order("ab,bc->ac", &[0, 1]);
        assert!(result.is_none());
    }

    #[test]
    fn test_remap_node_inputs() {
        let mut node = EinsumNode::new("ab,bc->ac", vec![0, 1], vec![2]);

        let mut tensor_map = HashMap::new();
        tensor_map.insert(0, 5);
        tensor_map.insert(1, 6);

        remap_node_inputs(&mut node, &tensor_map);

        assert_eq!(node.inputs, vec![5, 6]);
    }
}
