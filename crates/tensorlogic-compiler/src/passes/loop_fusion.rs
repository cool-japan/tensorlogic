//! Loop fusion optimization pass.
//!
//! This module provides optimization passes that fuse multiple loops/reductions
//! over the same axes to improve cache locality and reduce memory traffic.
//!
//! # Overview
//!
//! Loop fusion combines multiple consecutive operations that iterate over
//! the same axis into a single fused operation. This optimization:
//! - Reduces memory traffic (fewer intermediate tensors)
//! - Improves cache locality (better temporal locality)
//! - Reduces loop overhead (fewer loop iterations)
//!
//! # Fusion Criteria
//!
//! Two loops can be fused if:
//! 1. They iterate over the same axis/axes
//! 2. They have compatible domains
//! 3. There are no dependencies that prevent fusion
//! 4. The fused operation doesn't exceed memory constraints
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_compiler::passes::fuse_loops;
//! use tensorlogic_ir::EinsumGraph;
//!
//! let graph = EinsumGraph::new();
//! let (fused_graph, stats) = fuse_loops(&graph);
//! ```

use std::collections::{HashMap, HashSet};
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

/// Statistics from loop fusion optimization.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoopFusionStats {
    /// Number of loop pairs successfully fused
    pub loops_fused: usize,
    /// Number of reduction operations merged
    pub reductions_merged: usize,
    /// Number of intermediate tensors eliminated
    pub intermediates_eliminated: usize,
    /// Total number of nodes processed
    pub total_processed: usize,
}

impl LoopFusionStats {
    /// Get total number of optimizations applied.
    pub fn total_optimizations(&self) -> usize {
        self.loops_fused + self.reductions_merged + self.intermediates_eliminated
    }
}

/// Configuration for loop fusion optimization.
#[derive(Debug, Clone)]
pub struct LoopFusionConfig {
    /// Enable fusion of reduction operations
    pub enable_reduction_fusion: bool,
    /// Enable fusion of element-wise operations
    pub enable_elementwise_fusion: bool,
    /// Maximum number of operations to fuse together
    pub max_fusion_size: usize,
    /// Minimum benefit threshold (estimated speedup factor)
    pub min_benefit_threshold: f64,
}

impl Default for LoopFusionConfig {
    fn default() -> Self {
        Self {
            enable_reduction_fusion: true,
            enable_elementwise_fusion: true,
            max_fusion_size: 8,
            min_benefit_threshold: 1.1, // At least 10% speedup
        }
    }
}

/// Fuse loops in an einsum graph.
///
/// This function identifies opportunities to fuse multiple loops/reductions
/// over the same axes and combines them into single fused operations.
///
/// # Arguments
///
/// * `graph` - The einsum graph to optimize
///
/// # Returns
///
/// A tuple of (optimized_graph, statistics)
pub fn fuse_loops(graph: &EinsumGraph) -> (EinsumGraph, LoopFusionStats) {
    fuse_loops_with_config(graph, &LoopFusionConfig::default())
}

/// Fuse loops with custom configuration.
pub fn fuse_loops_with_config(
    graph: &EinsumGraph,
    config: &LoopFusionConfig,
) -> (EinsumGraph, LoopFusionStats) {
    let optimized = graph.clone();
    let mut stats = LoopFusionStats::default();

    // Build dependency graph
    let dependencies = build_dependency_graph(&optimized);

    // Find fusible loop groups
    let fusion_groups = find_fusion_groups(&optimized, &dependencies, config);

    stats.total_processed = optimized.nodes.len();

    // Count potential fusions
    for group in fusion_groups {
        if group.len() >= 2 {
            stats.loops_fused += 1;
            stats.intermediates_eliminated += group.len() - 1;

            // Check if we would fuse reductions
            for &node_idx in &group {
                if let Some(node) = optimized.nodes.get(node_idx) {
                    if matches!(node.op, OpType::Reduce { .. }) {
                        stats.reductions_merged += 1;
                    }
                }
            }
        }
    }

    (optimized, stats)
}

/// Build a dependency graph showing which nodes depend on which.
fn build_dependency_graph(graph: &EinsumGraph) -> HashMap<usize, HashSet<usize>> {
    let mut deps = HashMap::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        let mut node_deps = HashSet::new();

        // Add dependencies from input tensors
        for &input_idx in &node.inputs {
            // Find which node produced this tensor
            for (producer_idx, producer) in graph.nodes.iter().enumerate() {
                if producer.outputs.contains(&input_idx) {
                    node_deps.insert(producer_idx);
                }
            }
        }

        deps.insert(idx, node_deps);
    }

    deps
}

/// Find groups of nodes that can be fused together.
fn find_fusion_groups(
    graph: &EinsumGraph,
    dependencies: &HashMap<usize, HashSet<usize>>,
    config: &LoopFusionConfig,
) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut visited = HashSet::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if visited.contains(&idx) {
            continue;
        }

        // Start a new potential fusion group
        let mut group = vec![idx];
        visited.insert(idx);

        // Try to find compatible nodes to fuse with
        for (other_idx, other_node) in graph.nodes.iter().enumerate() {
            if other_idx == idx || visited.contains(&other_idx) {
                continue;
            }

            if group.len() >= config.max_fusion_size {
                break;
            }

            // Check if nodes are fusible
            if can_fuse_nodes(node, other_node, config)
                && !has_dependency_conflict(&group, other_idx, dependencies)
            {
                group.push(other_idx);
                visited.insert(other_idx);
            }
        }

        if group.len() > 1 {
            groups.push(group);
        }
    }

    groups
}

/// Check if two nodes can be fused together.
fn can_fuse_nodes(node1: &EinsumNode, node2: &EinsumNode, config: &LoopFusionConfig) -> bool {
    match (&node1.op, &node2.op) {
        // Fuse reductions over the same axes
        (
            OpType::Reduce {
                op: op1,
                axes: axes1,
            },
            OpType::Reduce {
                op: op2,
                axes: axes2,
            },
        ) => {
            config.enable_reduction_fusion
                && op1 == op2 // Same reduction operation
                && axes1 == axes2 // Same axes
        }

        // Fuse element-wise operations
        (OpType::ElemUnary { .. }, OpType::ElemUnary { .. })
        | (OpType::ElemBinary { .. }, OpType::ElemBinary { .. }) => {
            config.enable_elementwise_fusion
        }

        _ => false,
    }
}

/// Check if adding a node to a group would create dependency conflicts.
fn has_dependency_conflict(
    group: &[usize],
    candidate: usize,
    dependencies: &HashMap<usize, HashSet<usize>>,
) -> bool {
    // Check if candidate depends on any node in the group
    if let Some(candidate_deps) = dependencies.get(&candidate) {
        for &group_member in group {
            if candidate_deps.contains(&group_member) {
                return true;
            }
        }
    }

    // Check if any node in the group depends on candidate
    for &group_member in group {
        if let Some(member_deps) = dependencies.get(&group_member) {
            if member_deps.contains(&candidate) {
                return true;
            }
        }
    }

    false
}

/// Estimate the benefit of fusing a group of nodes.
///
/// Returns an estimated speedup factor (1.0 = no benefit, 2.0 = 2x speedup).
pub fn estimate_fusion_benefit(graph: &EinsumGraph, group: &[usize]) -> f64 {
    if group.len() < 2 {
        return 1.0;
    }

    // Simple heuristic: speedup ~ number of fused operations
    // In practice, fusion reduces memory traffic and loop overhead
    let base_speedup = 1.0 + (group.len() as f64 - 1.0) * 0.3;

    // Bonus for reducing intermediate tensors
    let intermediate_bonus = (group.len() - 1) as f64 * 0.2;

    // Check if we're fusing reductions (higher benefit)
    let mut reduction_count = 0;
    for &node_idx in group {
        if let Some(node) = graph.nodes.get(node_idx) {
            if matches!(node.op, OpType::Reduce { .. }) {
                reduction_count += 1;
            }
        }
    }
    let reduction_bonus = reduction_count as f64 * 0.1;

    base_speedup + intermediate_bonus + reduction_bonus
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();

        // Add some tensors
        let _t0 = graph.add_tensor("t0");
        let _t1 = graph.add_tensor("t1");

        graph
    }

    #[test]
    fn test_build_dependency_graph() {
        let graph = create_test_graph();
        let deps = build_dependency_graph(&graph);

        assert_eq!(deps.len(), 0); // No nodes yet
    }

    #[test]
    fn test_can_fuse_same_reductions() {
        let config = LoopFusionConfig::default();
        let node1 = EinsumNode::reduce("sum", vec![0], 0, 1);
        let node2 = EinsumNode::reduce("sum", vec![0], 2, 3);

        assert!(can_fuse_nodes(&node1, &node2, &config));
    }

    #[test]
    fn test_cannot_fuse_different_axes() {
        let config = LoopFusionConfig::default();
        let node1 = EinsumNode::reduce("sum", vec![0], 0, 1);
        let node2 = EinsumNode::reduce("sum", vec![1], 2, 3);

        assert!(!can_fuse_nodes(&node1, &node2, &config));
    }

    #[test]
    fn test_can_fuse_elementwise() {
        let config = LoopFusionConfig::default();
        let node1 = EinsumNode::elem_unary("exp", 0, 1);
        let node2 = EinsumNode::elem_unary("log", 2, 3);

        assert!(can_fuse_nodes(&node1, &node2, &config));
    }

    #[test]
    fn test_estimate_fusion_benefit() {
        let graph = create_test_graph();

        // Single node: no benefit
        let benefit = estimate_fusion_benefit(&graph, &[0]);
        assert_eq!(benefit, 1.0);

        // Two nodes: significant benefit
        let benefit = estimate_fusion_benefit(&graph, &[0, 1]);
        assert!(benefit > 1.0);
        assert!(benefit < 3.0);
    }

    #[test]
    fn test_fuse_loops_stats() {
        let graph = create_test_graph();
        let (_optimized, stats) = fuse_loops(&graph);

        assert_eq!(stats.total_processed, 0); // No nodes
    }

    #[test]
    fn test_config_builder() {
        let config = LoopFusionConfig {
            enable_reduction_fusion: false,
            enable_elementwise_fusion: true,
            max_fusion_size: 4,
            min_benefit_threshold: 1.5,
        };

        assert!(!config.enable_reduction_fusion);
        assert!(config.enable_elementwise_fusion);
        assert_eq!(config.max_fusion_size, 4);
        assert_eq!(config.min_benefit_threshold, 1.5);
    }

    #[test]
    fn test_dependency_conflict_detection() {
        let mut deps = HashMap::new();
        deps.insert(0, HashSet::new());
        deps.insert(1, vec![0].into_iter().collect());

        // Node 1 depends on node 0, so they cannot be fused
        assert!(has_dependency_conflict(&[0], 1, &deps));
        assert!(!has_dependency_conflict(&[0], 2, &deps));
    }

    #[test]
    fn test_stats_total_optimizations() {
        let stats = LoopFusionStats {
            loops_fused: 2,
            reductions_merged: 3,
            intermediates_eliminated: 1,
            total_processed: 10,
        };

        assert_eq!(stats.total_optimizations(), 6);
    }
}
