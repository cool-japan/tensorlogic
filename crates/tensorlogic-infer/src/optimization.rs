//! Graph optimization and fusion detection utilities.
//!
//! This module provides utilities for analyzing and optimizing EinsumGraph structures:
//! - Fusion opportunities detection (combining adjacent operations)
//! - Dead node elimination
//! - Redundant computation detection
//! - Operation reordering for better cache locality

use std::collections::{HashMap, HashSet};
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

/// Fusion opportunity between two nodes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionOpportunity {
    pub producer_idx: usize,
    pub consumer_idx: usize,
    pub fusion_type: FusionType,
    pub estimated_speedup: u32, // Percentage improvement
}

/// Type of fusion that can be applied
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionType {
    /// Element-wise operations can be fused
    ElementWise,
    /// Reduction followed by element-wise
    ReductionElementWise,
    /// Multiple reductions on same input
    MultiReduction,
    /// Einsum operations with compatible specs
    EinsumChain,
}

/// Optimization pass result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub fusion_opportunities: Vec<FusionOpportunity>,
    pub dead_nodes: Vec<usize>,
    pub redundant_computations: Vec<(usize, usize)>, // Pairs of equivalent nodes
    pub estimated_improvement: f64,                  // Overall estimated improvement percentage
}

impl OptimizationResult {
    pub fn new() -> Self {
        OptimizationResult {
            fusion_opportunities: Vec::new(),
            dead_nodes: Vec::new(),
            redundant_computations: Vec::new(),
            estimated_improvement: 0.0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.fusion_opportunities.is_empty()
            && self.dead_nodes.is_empty()
            && self.redundant_computations.is_empty()
    }

    pub fn total_opportunities(&self) -> usize {
        self.fusion_opportunities.len() + self.dead_nodes.len() + self.redundant_computations.len()
    }
}

impl Default for OptimizationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph optimizer for detecting optimization opportunities
pub struct GraphOptimizer {
    enable_fusion: bool,
    enable_dead_node_elimination: bool,
    enable_redundancy_detection: bool,
    min_fusion_benefit: u32,
}

impl GraphOptimizer {
    pub fn new() -> Self {
        GraphOptimizer {
            enable_fusion: true,
            enable_dead_node_elimination: true,
            enable_redundancy_detection: true,
            min_fusion_benefit: 10, // Minimum 10% improvement
        }
    }

    pub fn with_fusion(mut self, enabled: bool) -> Self {
        self.enable_fusion = enabled;
        self
    }

    pub fn with_dead_node_elimination(mut self, enabled: bool) -> Self {
        self.enable_dead_node_elimination = enabled;
        self
    }

    pub fn with_redundancy_detection(mut self, enabled: bool) -> Self {
        self.enable_redundancy_detection = enabled;
        self
    }

    pub fn with_min_fusion_benefit(mut self, min_benefit: u32) -> Self {
        self.min_fusion_benefit = min_benefit;
        self
    }

    /// Analyze graph and detect optimization opportunities
    pub fn analyze(&self, graph: &EinsumGraph) -> OptimizationResult {
        let mut result = OptimizationResult::new();

        // Build dependency information
        let tensor_producers = self.build_producer_map(graph);
        let tensor_consumers = self.build_consumer_map(graph);

        // Detect fusion opportunities
        if self.enable_fusion {
            result.fusion_opportunities =
                self.detect_fusion_opportunities(graph, &tensor_producers, &tensor_consumers);
        }

        // Detect dead nodes
        if self.enable_dead_node_elimination {
            result.dead_nodes = self.detect_dead_nodes(graph, &tensor_consumers);
        }

        // Detect redundant computations
        if self.enable_redundancy_detection {
            result.redundant_computations = self.detect_redundant_computations(graph);
        }

        // Estimate overall improvement
        result.estimated_improvement = self.estimate_improvement(&result);

        result
    }

    /// Build map of which node produces each tensor
    fn build_producer_map(&self, graph: &EinsumGraph) -> HashMap<usize, usize> {
        let mut producers = HashMap::new();
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &output_idx in &node.outputs {
                producers.insert(output_idx, node_idx);
            }
        }
        producers
    }

    /// Build map of which nodes consume each tensor
    fn build_consumer_map(&self, graph: &EinsumGraph) -> HashMap<usize, Vec<usize>> {
        let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &input_idx in &node.inputs {
                consumers.entry(input_idx).or_default().push(node_idx);
            }
        }
        consumers
    }

    /// Detect fusion opportunities between adjacent nodes
    fn detect_fusion_opportunities(
        &self,
        graph: &EinsumGraph,
        tensor_producers: &HashMap<usize, usize>,
        tensor_consumers: &HashMap<usize, Vec<usize>>,
    ) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // Check each input to see if it comes from a fusible producer
            for &input_idx in &node.inputs {
                if let Some(&producer_idx) = tensor_producers.get(&input_idx) {
                    // Check if this tensor is only used by current node (enables fusion)
                    let is_single_use = tensor_consumers
                        .get(&input_idx)
                        .map(|consumers| consumers.len() == 1)
                        .unwrap_or(false);

                    if is_single_use {
                        if let Some(fusion_type) = self.can_fuse(&graph.nodes[producer_idx], node) {
                            let estimated_speedup = self.estimate_fusion_speedup(fusion_type);
                            if estimated_speedup >= self.min_fusion_benefit {
                                opportunities.push(FusionOpportunity {
                                    producer_idx,
                                    consumer_idx: node_idx,
                                    fusion_type,
                                    estimated_speedup,
                                });
                            }
                        }
                    }
                }
            }
        }

        opportunities
    }

    /// Check if two nodes can be fused
    fn can_fuse(&self, producer: &EinsumNode, consumer: &EinsumNode) -> Option<FusionType> {
        match (&producer.op, &consumer.op) {
            // Element-wise + Element-wise (unary or binary)
            (OpType::ElemUnary { .. }, OpType::ElemUnary { .. })
            | (OpType::ElemUnary { .. }, OpType::ElemBinary { .. })
            | (OpType::ElemBinary { .. }, OpType::ElemUnary { .. })
            | (OpType::ElemBinary { .. }, OpType::ElemBinary { .. }) => {
                Some(FusionType::ElementWise)
            }

            // Reduction + Element-wise
            (OpType::Reduce { .. }, OpType::ElemUnary { .. })
            | (OpType::Reduce { .. }, OpType::ElemBinary { .. }) => {
                Some(FusionType::ReductionElementWise)
            }

            // Einsum chain (simplified detection)
            (OpType::Einsum { .. }, OpType::Einsum { .. }) => Some(FusionType::EinsumChain),

            _ => None,
        }
    }

    /// Estimate speedup from fusion
    fn estimate_fusion_speedup(&self, fusion_type: FusionType) -> u32 {
        match fusion_type {
            FusionType::ElementWise => 40, // High benefit: eliminates memory round-trip
            FusionType::ReductionElementWise => 25, // Moderate benefit
            FusionType::MultiReduction => 30, // Good benefit: shared input loading
            FusionType::EinsumChain => 20, // Lower benefit: complex to fuse
        }
    }

    /// Detect nodes whose outputs are never used
    fn detect_dead_nodes(
        &self,
        graph: &EinsumGraph,
        tensor_consumers: &HashMap<usize, Vec<usize>>,
    ) -> Vec<usize> {
        let mut dead_nodes = Vec::new();

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // A node is dead if none of its outputs are consumed
            let all_outputs_unused = node.outputs.iter().all(|&output_idx| {
                tensor_consumers
                    .get(&output_idx)
                    .map(|consumers| consumers.is_empty())
                    .unwrap_or(true)
            });

            if all_outputs_unused {
                dead_nodes.push(node_idx);
            }
        }

        dead_nodes
    }

    /// Detect redundant computations (nodes with identical inputs and operations)
    fn detect_redundant_computations(&self, graph: &EinsumGraph) -> Vec<(usize, usize)> {
        let mut redundant_pairs = Vec::new();
        let mut seen: HashMap<String, Vec<usize>> = HashMap::new();

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // Create signature for this node (op + sorted inputs)
            let mut signature = format!("{:?}", node.op);
            let mut sorted_inputs = node.inputs.clone();
            sorted_inputs.sort_unstable();
            signature.push_str(&format!("{:?}", sorted_inputs));

            // Check if we've seen this signature before
            if let Some(previous_nodes) = seen.get(&signature) {
                for &prev_idx in previous_nodes {
                    redundant_pairs.push((prev_idx, node_idx));
                }
            }

            seen.entry(signature).or_default().push(node_idx);
        }

        redundant_pairs
    }

    /// Estimate overall improvement percentage
    fn estimate_improvement(&self, result: &OptimizationResult) -> f64 {
        let mut total_improvement = 0.0;

        // Add fusion benefits
        for fusion in &result.fusion_opportunities {
            total_improvement += fusion.estimated_speedup as f64;
        }

        // Add dead node elimination (assume 5% per dead node)
        total_improvement += result.dead_nodes.len() as f64 * 5.0;

        // Add redundancy elimination (assume 10% per redundant pair)
        total_improvement += result.redundant_computations.len() as f64 * 10.0;

        total_improvement
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fusion planner for actually applying fusion transformations
pub struct FusionPlanner {
    max_fusion_depth: usize,
}

impl FusionPlanner {
    pub fn new() -> Self {
        FusionPlanner {
            max_fusion_depth: 3,
        }
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_fusion_depth = depth;
        self
    }

    /// Plan which fusions to apply (considering dependencies and depth limits)
    pub fn plan_fusions(&self, opportunities: &[FusionOpportunity]) -> Vec<FusionOpportunity> {
        let mut planned = Vec::new();
        let mut fused_nodes = HashSet::new();

        // Sort by estimated speedup (highest first)
        let mut sorted_ops = opportunities.to_vec();
        sorted_ops.sort_by(|a, b| b.estimated_speedup.cmp(&a.estimated_speedup));

        for fusion in sorted_ops {
            // Skip if either node already part of a fusion
            if fused_nodes.contains(&fusion.producer_idx)
                || fused_nodes.contains(&fusion.consumer_idx)
            {
                continue;
            }

            // Check depth limit (simplified: just count current chain length)
            if planned.len() >= self.max_fusion_depth {
                break;
            }

            planned.push(fusion.clone());
            fused_nodes.insert(fusion.producer_idx);
            fused_nodes.insert(fusion.consumer_idx);
        }

        planned
    }

    /// Validate that planned fusions don't conflict
    pub fn validate_plan(&self, plan: &[FusionOpportunity]) -> bool {
        let mut used_nodes = HashSet::new();

        for fusion in plan {
            if used_nodes.contains(&fusion.producer_idx)
                || used_nodes.contains(&fusion.consumer_idx)
            {
                return false;
            }
            used_nodes.insert(fusion.producer_idx);
            used_nodes.insert(fusion.consumer_idx);
        }

        true
    }
}

impl Default for FusionPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();

        // Add input tensors
        graph.tensors.push("x".to_string()); // tensor 0
        graph.tensors.push("y".to_string()); // tensor 1

        // Node 0: Einsum ab,bc->ac
        graph.tensors.push("t2".to_string()); // tensor 2
        graph.nodes.push(EinsumNode {
            inputs: vec![0, 1],
            outputs: vec![2],
            op: OpType::Einsum {
                spec: "ab,bc->ac".into(),
            },
            metadata: None,
        });

        // Node 1: Element-wise operation on tensor 2
        graph.tensors.push("t3".to_string()); // tensor 3
        graph.nodes.push(EinsumNode {
            inputs: vec![2],
            outputs: vec![3],
            op: OpType::ElemUnary { op: "add".into() },
            metadata: None,
        });

        // Node 2: Another element-wise on tensor 3 (fusible with node 1)
        graph.tensors.push("t4".to_string()); // tensor 4
        graph.nodes.push(EinsumNode {
            inputs: vec![3],
            outputs: vec![4],
            op: OpType::ElemUnary { op: "mul".into() },
            metadata: None,
        });

        graph
    }

    fn create_graph_with_dead_node() -> EinsumGraph {
        let mut graph = create_test_graph();

        // Add a dead node whose output is never used
        graph.tensors.push("t5".to_string()); // tensor 5 (never consumed)
        graph.nodes.push(EinsumNode {
            inputs: vec![0],
            outputs: vec![5],
            op: OpType::ElemUnary { op: "add".into() },
            metadata: None,
        });

        graph
    }

    fn create_graph_with_redundancy() -> EinsumGraph {
        let mut graph = EinsumGraph::new();

        // Input tensors
        graph.tensors.push("x".to_string()); // tensor 0
        graph.tensors.push("y".to_string()); // tensor 1

        // Node 0: Add tensors 0 and 1
        graph.tensors.push("t2".to_string()); // tensor 2
        graph.nodes.push(EinsumNode {
            inputs: vec![0, 1],
            outputs: vec![2],
            op: OpType::ElemBinary { op: "add".into() },
            metadata: None,
        });

        // Node 1: Duplicate of node 0 (redundant)
        graph.tensors.push("t3".to_string()); // tensor 3
        graph.nodes.push(EinsumNode {
            inputs: vec![0, 1],
            outputs: vec![3],
            op: OpType::ElemBinary { op: "add".into() },
            metadata: None,
        });

        graph
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = GraphOptimizer::new();
        assert!(optimizer.enable_fusion);
        assert!(optimizer.enable_dead_node_elimination);
        assert!(optimizer.enable_redundancy_detection);
        assert_eq!(optimizer.min_fusion_benefit, 10);
    }

    #[test]
    fn test_optimizer_builder() {
        let optimizer = GraphOptimizer::new()
            .with_fusion(false)
            .with_dead_node_elimination(false)
            .with_min_fusion_benefit(20);

        assert!(!optimizer.enable_fusion);
        assert!(!optimizer.enable_dead_node_elimination);
        assert_eq!(optimizer.min_fusion_benefit, 20);
    }

    #[test]
    fn test_producer_map() {
        let graph = create_test_graph();
        let optimizer = GraphOptimizer::new();
        let producers = optimizer.build_producer_map(&graph);

        assert_eq!(producers.get(&2), Some(&0)); // Node 0 produces tensor 2
        assert_eq!(producers.get(&3), Some(&1)); // Node 1 produces tensor 3
        assert_eq!(producers.get(&4), Some(&2)); // Node 2 produces tensor 4
    }

    #[test]
    fn test_consumer_map() {
        let graph = create_test_graph();
        let optimizer = GraphOptimizer::new();
        let consumers = optimizer.build_consumer_map(&graph);

        assert_eq!(consumers.get(&0), Some(&vec![0])); // Tensor 0 consumed by node 0
        assert_eq!(consumers.get(&2), Some(&vec![1])); // Tensor 2 consumed by node 1
        assert_eq!(consumers.get(&3), Some(&vec![2])); // Tensor 3 consumed by node 2
    }

    #[test]
    fn test_fusion_detection() {
        let graph = create_test_graph();
        let optimizer = GraphOptimizer::new();
        let result = optimizer.analyze(&graph);

        // Should detect fusion opportunity between nodes 1 and 2 (element-wise chain)
        assert!(!result.fusion_opportunities.is_empty());
        let fusion = &result.fusion_opportunities[0];
        assert_eq!(fusion.fusion_type, FusionType::ElementWise);
        assert!(fusion.estimated_speedup >= 10);
    }

    #[test]
    fn test_dead_node_detection() {
        let graph = create_graph_with_dead_node();
        let optimizer = GraphOptimizer::new();
        let result = optimizer.analyze(&graph);

        // Should detect node 3 as dead (output never consumed)
        assert!(!result.dead_nodes.is_empty());
        assert!(result.dead_nodes.contains(&3));
    }

    #[test]
    fn test_redundancy_detection() {
        let graph = create_graph_with_redundancy();
        let optimizer = GraphOptimizer::new();
        let result = optimizer.analyze(&graph);

        // Should detect nodes 0 and 1 as redundant
        assert!(!result.redundant_computations.is_empty());
        assert_eq!(result.redundant_computations[0], (0, 1));
    }

    #[test]
    fn test_optimization_result_empty() {
        let result = OptimizationResult::new();
        assert!(result.is_empty());
        assert_eq!(result.total_opportunities(), 0);
    }

    #[test]
    fn test_optimization_result_nonempty() {
        let mut result = OptimizationResult::new();
        result.fusion_opportunities.push(FusionOpportunity {
            producer_idx: 0,
            consumer_idx: 1,
            fusion_type: FusionType::ElementWise,
            estimated_speedup: 40,
        });
        result.dead_nodes.push(2);

        assert!(!result.is_empty());
        assert_eq!(result.total_opportunities(), 2);
    }

    #[test]
    fn test_can_fuse_elementwise() {
        let optimizer = GraphOptimizer::new();

        let producer = EinsumNode {
            inputs: vec![0],
            outputs: vec![1],
            op: OpType::ElemUnary { op: "add".into() },
            metadata: None,
        };

        let consumer = EinsumNode {
            inputs: vec![1],
            outputs: vec![2],
            op: OpType::ElemUnary { op: "mul".into() },
            metadata: None,
        };

        let fusion_type = optimizer.can_fuse(&producer, &consumer);
        assert_eq!(fusion_type, Some(FusionType::ElementWise));
    }

    #[test]
    fn test_fusion_planner_creation() {
        let planner = FusionPlanner::new();
        assert_eq!(planner.max_fusion_depth, 3);
    }

    #[test]
    fn test_fusion_planner_with_depth() {
        let planner = FusionPlanner::new().with_max_depth(5);
        assert_eq!(planner.max_fusion_depth, 5);
    }

    #[test]
    fn test_fusion_planning() {
        let opportunities = vec![
            FusionOpportunity {
                producer_idx: 0,
                consumer_idx: 1,
                fusion_type: FusionType::ElementWise,
                estimated_speedup: 40,
            },
            FusionOpportunity {
                producer_idx: 2,
                consumer_idx: 3,
                fusion_type: FusionType::ReductionElementWise,
                estimated_speedup: 25,
            },
        ];

        let planner = FusionPlanner::new();
        let plan = planner.plan_fusions(&opportunities);

        assert_eq!(plan.len(), 2);
        assert!(planner.validate_plan(&plan));
    }

    #[test]
    fn test_fusion_planning_with_conflicts() {
        let opportunities = vec![
            FusionOpportunity {
                producer_idx: 0,
                consumer_idx: 1,
                fusion_type: FusionType::ElementWise,
                estimated_speedup: 40,
            },
            FusionOpportunity {
                producer_idx: 1, // Conflicts with previous consumer
                consumer_idx: 2,
                fusion_type: FusionType::ElementWise,
                estimated_speedup: 35,
            },
        ];

        let planner = FusionPlanner::new();
        let plan = planner.plan_fusions(&opportunities);

        // Should only include the first fusion (higher speedup)
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].producer_idx, 0);
    }

    #[test]
    fn test_estimate_improvement() {
        let optimizer = GraphOptimizer::new();
        let mut result = OptimizationResult::new();

        result.fusion_opportunities.push(FusionOpportunity {
            producer_idx: 0,
            consumer_idx: 1,
            fusion_type: FusionType::ElementWise,
            estimated_speedup: 40,
        });
        result.dead_nodes.push(2);
        result.redundant_computations.push((3, 4));

        let improvement = optimizer.estimate_improvement(&result);
        assert!(improvement > 0.0);
        assert_eq!(improvement, 40.0 + 5.0 + 10.0); // fusion + dead + redundant
    }

    #[test]
    fn test_disabled_optimizations() {
        let graph = create_graph_with_dead_node();
        let optimizer = GraphOptimizer::new()
            .with_fusion(false)
            .with_dead_node_elimination(false)
            .with_redundancy_detection(false);

        let result = optimizer.analyze(&graph);

        assert!(result.fusion_opportunities.is_empty());
        assert!(result.dead_nodes.is_empty());
        assert!(result.redundant_computations.is_empty());
    }
}
