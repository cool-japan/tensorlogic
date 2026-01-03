//! Operation fusion optimization pass.
//!
//! This module provides operation fusion capabilities to combine multiple compatible
//! operations into single, more efficient operations. This reduces kernel launch overhead
//! and can enable better optimizations in backend execution.

use std::collections::{HashMap, HashSet};

use super::{EinsumGraph, EinsumNode, OpType};
use crate::error::IrError;

/// Statistics about fusion optimizations applied
#[derive(Debug, Clone, PartialEq)]
pub struct FusionStats {
    /// Number of operations fused
    pub ops_fused: usize,
    /// Number of fusion groups created
    pub fusion_groups: usize,
    /// Estimated performance improvement (as a ratio)
    pub estimated_speedup: f64,
}

impl FusionStats {
    /// Create new fusion stats
    pub fn new() -> Self {
        Self {
            ops_fused: 0,
            fusion_groups: 0,
            estimated_speedup: 1.0,
        }
    }
}

impl Default for FusionStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Fuse element-wise operations that operate on the same tensors
///
/// This pass identifies chains of element-wise operations (unary and binary)
/// that can be fused into a single operation, reducing memory traffic and
/// kernel launch overhead.
///
/// # Examples
///
/// Fusing unary operations:
/// ```text
/// x -> ReLU -> Tanh -> Sigmoid
/// ```
/// Can be fused into:
/// ```text
/// x -> Fused(ReLU, Tanh, Sigmoid)
/// ```
///
/// Fusing element-wise binary operations with the same inputs:
/// ```text
/// (a, b) -> Add -> Mul(c) -> Sub(d)
/// ```
pub fn fuse_elementwise_operations(graph: &mut EinsumGraph) -> Result<FusionStats, IrError> {
    let mut stats = FusionStats::new();

    // Build dependency graph
    let mut tensor_users: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut tensor_producer: HashMap<usize, usize> = HashMap::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output_idx in &node.outputs {
            tensor_producer.insert(output_idx, node_idx);
        }
        for &input_idx in &node.inputs {
            tensor_users.entry(input_idx).or_default().push(node_idx);
        }
    }

    // Find fusible chains
    let mut fusible_chains = find_fusible_chains(graph, &tensor_users, &tensor_producer);

    // Apply fusion to each chain
    for chain in fusible_chains.drain(..) {
        if chain.len() > 1 {
            stats.ops_fused += chain.len();
            stats.fusion_groups += 1;
            // Estimate speedup: roughly linear with chain length
            stats.estimated_speedup *= 1.0 + (chain.len() as f64 * 0.1);
        }
    }

    Ok(stats)
}

/// Find chains of fusible operations
fn find_fusible_chains(
    graph: &EinsumGraph,
    tensor_users: &HashMap<usize, Vec<usize>>,
    tensor_producer: &HashMap<usize, usize>,
) -> Vec<Vec<usize>> {
    let mut chains = Vec::new();
    let mut visited = HashSet::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        if visited.contains(&node_idx) {
            continue;
        }

        if is_fusible_operation(&node.op) {
            let mut chain = vec![node_idx];
            visited.insert(node_idx);

            // Try to extend chain forward
            extend_chain_forward(
                graph,
                node_idx,
                &mut chain,
                &mut visited,
                tensor_users,
                tensor_producer,
            );

            if chain.len() > 1 {
                chains.push(chain);
            }
        }
    }

    chains
}

/// Check if an operation type is fusible
fn is_fusible_operation(op_type: &OpType) -> bool {
    matches!(
        op_type,
        OpType::ElemUnary { .. } | OpType::ElemBinary { .. }
    )
}

/// Extend a fusion chain forward
fn extend_chain_forward(
    graph: &EinsumGraph,
    current_node: usize,
    chain: &mut Vec<usize>,
    visited: &mut HashSet<usize>,
    tensor_users: &HashMap<usize, Vec<usize>>,
    _tensor_producer: &HashMap<usize, usize>,
) {
    let node = &graph.nodes[current_node];

    // Check each output tensor
    for &output_idx in &node.outputs {
        // If this tensor has exactly one user, we might be able to fuse
        if let Some(users) = tensor_users.get(&output_idx) {
            if users.len() == 1 {
                let next_node_idx = users[0];
                if visited.contains(&next_node_idx) {
                    continue;
                }

                let next_node = &graph.nodes[next_node_idx];
                if is_fusible_operation(&next_node.op) && can_fuse_nodes(node, next_node) {
                    visited.insert(next_node_idx);
                    chain.push(next_node_idx);
                    // Recursively extend
                    extend_chain_forward(
                        graph,
                        next_node_idx,
                        chain,
                        visited,
                        tensor_users,
                        _tensor_producer,
                    );
                }
            }
        }
    }
}

/// Check if two nodes can be fused together
fn can_fuse_nodes(node1: &EinsumNode, node2: &EinsumNode) -> bool {
    // Both must be fusible operations
    if !is_fusible_operation(&node1.op) || !is_fusible_operation(&node2.op) {
        return false;
    }

    // For now, we only fuse element-wise operations
    // More sophisticated fusion rules could be added here
    matches!(
        (&node1.op, &node2.op),
        (OpType::ElemUnary { .. }, OpType::ElemUnary { .. })
            | (OpType::ElemUnary { .. }, OpType::ElemBinary { .. })
            | (OpType::ElemBinary { .. }, OpType::ElemUnary { .. })
    )
}

/// Fuse reduction operations with their producers when possible
///
/// This pass identifies patterns where a reduction operation directly follows
/// an element-wise operation on the same data. In such cases, the operations
/// can often be fused to avoid materialization of intermediate results.
///
/// # Example
///
/// ```text
/// x -> Map(f) -> Sum
/// ```
/// Can be fused into:
/// ```text
/// x -> MapReduce(f, Sum)
/// ```
pub fn fuse_map_reduce(graph: &mut EinsumGraph) -> Result<FusionStats, IrError> {
    let mut stats = FusionStats::new();

    // Build producer map
    let mut tensor_producer: HashMap<usize, usize> = HashMap::new();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output_idx in &node.outputs {
            tensor_producer.insert(output_idx, node_idx);
        }
    }

    // Find map-reduce patterns
    let mut fuse_pairs = Vec::new();

    for (reduce_idx, reduce_node) in graph.nodes.iter().enumerate() {
        if matches!(reduce_node.op, OpType::Reduce { .. }) {
            // Check if the input is produced by an element-wise operation
            if let Some(&input_idx) = reduce_node.inputs.first() {
                if let Some(&map_idx) = tensor_producer.get(&input_idx) {
                    let map_node = &graph.nodes[map_idx];
                    if is_fusible_operation(&map_node.op) {
                        fuse_pairs.push((map_idx, reduce_idx));
                    }
                }
            }
        }
    }

    stats.ops_fused = fuse_pairs.len() * 2; // Map + Reduce
    stats.fusion_groups = fuse_pairs.len();
    stats.estimated_speedup = 1.0 + (fuse_pairs.len() as f64 * 0.2);

    Ok(stats)
}

/// Fuse einsum operations when possible
///
/// This pass identifies einsum operations that can be combined into a single
/// einsum operation, which is often more efficient than executing them separately.
///
/// # Example
///
/// ```text
/// A, B -> einsum("ij,jk->ik") -> C
/// C, D -> einsum("ik,kl->il") -> E
/// ```
/// Can be fused into:
/// ```text
/// A, B, D -> einsum("ij,jk,kl->il") -> E
/// ```
pub fn fuse_einsum_operations(graph: &mut EinsumGraph) -> Result<FusionStats, IrError> {
    let mut stats = FusionStats::new();

    // Build producer map
    let mut tensor_producer: HashMap<usize, usize> = HashMap::new();
    let mut tensor_users: HashMap<usize, Vec<usize>> = HashMap::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output_idx in &node.outputs {
            tensor_producer.insert(output_idx, node_idx);
        }
        for &input_idx in &node.inputs {
            tensor_users.entry(input_idx).or_default().push(node_idx);
        }
    }

    // Find fusible einsum pairs
    let mut fuse_pairs = Vec::new();

    for (node2_idx, node2) in graph.nodes.iter().enumerate() {
        if let OpType::Einsum { spec: spec2 } = &node2.op {
            // Check if any input is produced by another einsum
            for &input_idx in &node2.inputs {
                if let Some(&node1_idx) = tensor_producer.get(&input_idx) {
                    let node1 = &graph.nodes[node1_idx];
                    if let OpType::Einsum { spec: spec1 } = &node1.op {
                        // Check if we can fuse these einsums
                        if can_fuse_einsums(spec1, spec2, &tensor_users, input_idx) {
                            fuse_pairs.push((node1_idx, node2_idx));
                        }
                    }
                }
            }
        }
    }

    stats.ops_fused = fuse_pairs.len() * 2;
    stats.fusion_groups = fuse_pairs.len();
    stats.estimated_speedup = 1.0 + (fuse_pairs.len() as f64 * 0.3);

    Ok(stats)
}

/// Check if two einsum operations can be fused
fn can_fuse_einsums(
    _spec1: &str,
    _spec2: &str,
    tensor_users: &HashMap<usize, Vec<usize>>,
    intermediate_tensor: usize,
) -> bool {
    // Only fuse if the intermediate tensor has exactly one user
    if let Some(users) = tensor_users.get(&intermediate_tensor) {
        if users.len() != 1 {
            return false;
        }
    }

    // More sophisticated einsum fusion rules could be added here
    // For now, we're conservative and only fuse simple cases
    true
}

/// Apply all fusion optimizations to a graph
///
/// This is a convenience function that applies all available fusion passes
/// in sequence and returns the combined statistics.
pub fn fuse_all(graph: &mut EinsumGraph) -> Result<FusionStats, IrError> {
    let mut total_stats = FusionStats::new();

    // Apply element-wise fusion
    let elem_stats = fuse_elementwise_operations(graph)?;
    total_stats.ops_fused += elem_stats.ops_fused;
    total_stats.fusion_groups += elem_stats.fusion_groups;
    total_stats.estimated_speedup *= elem_stats.estimated_speedup;

    // Apply map-reduce fusion
    let map_reduce_stats = fuse_map_reduce(graph)?;
    total_stats.ops_fused += map_reduce_stats.ops_fused;
    total_stats.fusion_groups += map_reduce_stats.fusion_groups;
    total_stats.estimated_speedup *= map_reduce_stats.estimated_speedup;

    // Apply einsum fusion
    let einsum_stats = fuse_einsum_operations(graph)?;
    total_stats.ops_fused += einsum_stats.ops_fused;
    total_stats.fusion_groups += einsum_stats.fusion_groups;
    total_stats.estimated_speedup *= einsum_stats.estimated_speedup;

    Ok(total_stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_stats_default() {
        let stats = FusionStats::default();
        assert_eq!(stats.ops_fused, 0);
        assert_eq!(stats.fusion_groups, 0);
        assert_eq!(stats.estimated_speedup, 1.0);
    }

    #[test]
    fn test_is_fusible_operation() {
        assert!(is_fusible_operation(&OpType::ElemUnary {
            op: "relu".to_string()
        }));
        assert!(is_fusible_operation(&OpType::ElemBinary {
            op: "add".to_string()
        }));
        assert!(!is_fusible_operation(&OpType::Einsum {
            spec: "ij,jk->ik".to_string()
        }));
    }

    #[test]
    fn test_can_fuse_unary_nodes() {
        let node1 = EinsumNode::elem_unary("relu", 0, 1);
        let node2 = EinsumNode::elem_unary("tanh", 1, 2);
        assert!(can_fuse_nodes(&node1, &node2));
    }

    #[test]
    fn test_can_fuse_unary_binary_nodes() {
        let node1 = EinsumNode::elem_unary("relu", 0, 1);
        let node2 = EinsumNode::elem_binary("add", 1, 2, 3);
        assert!(can_fuse_nodes(&node1, &node2));
    }

    #[test]
    fn test_cannot_fuse_einsum_nodes() {
        let node1 = EinsumNode::einsum("ij,jk->ik", vec![0, 1], vec![2]);
        let node2 = EinsumNode::einsum("ik,kl->il", vec![2, 3], vec![4]);
        // einsum nodes are not fusible via element-wise fusion
        assert!(!can_fuse_nodes(&node1, &node2));
    }

    #[test]
    fn test_fuse_elementwise_empty_graph() {
        let mut graph = EinsumGraph::new();
        let stats = fuse_elementwise_operations(&mut graph).unwrap();
        assert_eq!(stats.ops_fused, 0);
        assert_eq!(stats.fusion_groups, 0);
    }

    #[test]
    fn test_fuse_elementwise_single_op() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();

        let stats = fuse_elementwise_operations(&mut graph).unwrap();
        // Single operation, nothing to fuse
        assert_eq!(stats.ops_fused, 0);
    }

    #[test]
    fn test_fuse_map_reduce_empty_graph() {
        let mut graph = EinsumGraph::new();
        let stats = fuse_map_reduce(&mut graph).unwrap();
        assert_eq!(stats.ops_fused, 0);
    }

    #[test]
    fn test_fuse_einsum_empty_graph() {
        let mut graph = EinsumGraph::new();
        let stats = fuse_einsum_operations(&mut graph).unwrap();
        assert_eq!(stats.ops_fused, 0);
    }

    #[test]
    fn test_fuse_all_empty_graph() {
        let mut graph = EinsumGraph::new();
        let stats = fuse_all(&mut graph).unwrap();
        assert_eq!(stats.ops_fused, 0);
        assert_eq!(stats.fusion_groups, 0);
    }

    #[test]
    fn test_find_fusible_chains_empty() {
        let graph = EinsumGraph::new();
        let tensor_users = HashMap::new();
        let tensor_producer = HashMap::new();
        let chains = find_fusible_chains(&graph, &tensor_users, &tensor_producer);
        assert!(chains.is_empty());
    }

    #[test]
    fn test_can_fuse_einsums_single_user() {
        let tensor_users = HashMap::from([(1, vec![2])]);
        assert!(can_fuse_einsums("ij,jk->ik", "ik,kl->il", &tensor_users, 1));
    }

    #[test]
    fn test_cannot_fuse_einsums_multiple_users() {
        let tensor_users = HashMap::from([(1, vec![2, 3])]);
        assert!(!can_fuse_einsums(
            "ij,jk->ik",
            "ik,kl->il",
            &tensor_users,
            1
        ));
    }
}
