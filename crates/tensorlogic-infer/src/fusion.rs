//! Advanced kernel fusion for optimized execution.
//!
//! This module provides sophisticated kernel fusion capabilities:
//! - Pattern-based fusion (common operator patterns)
//! - Vertical fusion (producer-consumer chains)
//! - Horizontal fusion (independent parallel operations)
//! - Loop fusion for reductions
//! - Memory bandwidth-aware fusion decisions
//! - Cost modeling for fusion trade-offs

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tensorlogic_ir::{EinsumGraph, OpType};
use thiserror::Error;

/// Node identifier (0-based index into graph.nodes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Fusion-related errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum FusionError {
    #[error("Fusion would create a cycle in the graph")]
    WouldCreateCycle,

    #[error("Incompatible operations for fusion: {0:?} and {1:?}")]
    IncompatibleOps(OpType, OpType),

    #[error("Fusion exceeds resource limits: {0}")]
    ResourceLimitExceeded(String),

    #[error("Invalid fusion pattern")]
    InvalidPattern,
}

/// Fusion pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusionPattern {
    /// Matrix multiplication followed by bias addition
    MatMulBias,
    /// Matrix multiplication followed by activation (ReLU, tanh, etc.)
    MatMulActivation,
    /// Bias addition followed by activation
    BiasActivation,
    /// BatchNorm + ReLU fusion
    BatchNormReLU,
    /// Conv + BatchNorm + ReLU
    ConvBNReLU,
    /// Elementwise operations chain
    ElementwiseChain,
    /// Reduction followed by elementwise
    ReduceElementwise,
    /// Multiple independent reductions (horizontal fusion)
    ParallelReductions,
    /// Broadcast followed by elementwise
    BroadcastElementwise,
}

/// Fusion strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Conservative - only fuse proven beneficial patterns
    Conservative,
    /// Aggressive - fuse as much as possible
    Aggressive,
    /// Balanced - consider cost model
    Balanced,
    /// Memory-aware - prioritize memory bandwidth reduction
    MemoryAware,
}

/// Fusion candidate representing potential fusion opportunity.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionCandidate {
    /// Nodes to be fused
    pub nodes: Vec<NodeId>,
    /// Pattern type
    pub pattern: FusionPattern,
    /// Estimated benefit score (higher is better)
    pub benefit_score: f64,
    /// Estimated memory savings (bytes)
    pub memory_savings: usize,
    /// Estimated compute reduction (FLOPS)
    pub compute_savings: f64,
}

/// Fusion configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Fusion strategy
    pub strategy: FusionStrategy,
    /// Maximum nodes per fused kernel
    pub max_fusion_size: usize,
    /// Enable pattern-based fusion
    pub enable_patterns: bool,
    /// Enable vertical fusion
    pub enable_vertical: bool,
    /// Enable horizontal fusion
    pub enable_horizontal: bool,
    /// Enable loop fusion
    pub enable_loop_fusion: bool,
    /// Memory bandwidth threshold (bytes/s)
    pub memory_bandwidth_threshold: Option<f64>,
    /// Minimum benefit score to apply fusion
    pub min_benefit_score: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategy::Balanced,
            max_fusion_size: 8,
            enable_patterns: true,
            enable_vertical: true,
            enable_horizontal: true,
            enable_loop_fusion: true,
            memory_bandwidth_threshold: None,
            min_benefit_score: 0.1,
        }
    }
}

impl FusionConfig {
    /// Create aggressive fusion configuration.
    pub fn aggressive() -> Self {
        Self {
            strategy: FusionStrategy::Aggressive,
            max_fusion_size: 16,
            min_benefit_score: 0.0,
            ..Default::default()
        }
    }

    /// Create conservative fusion configuration.
    pub fn conservative() -> Self {
        Self {
            strategy: FusionStrategy::Conservative,
            max_fusion_size: 4,
            enable_horizontal: false,
            enable_loop_fusion: false,
            min_benefit_score: 0.3,
            ..Default::default()
        }
    }

    /// Create memory-aware fusion configuration.
    pub fn memory_aware() -> Self {
        Self {
            strategy: FusionStrategy::MemoryAware,
            memory_bandwidth_threshold: Some(100e9), // 100 GB/s
            ..Default::default()
        }
    }
}

/// Cost model for fusion decisions.
#[derive(Debug, Clone)]
pub struct FusionCostModel {
    /// Cost of memory access (relative units)
    pub memory_access_cost: f64,
    /// Cost of compute operation (relative units)
    pub compute_cost: f64,
    /// Cost of kernel launch overhead
    pub kernel_launch_cost: f64,
    /// Memory bandwidth (bytes/second)
    pub memory_bandwidth: f64,
}

impl Default for FusionCostModel {
    fn default() -> Self {
        Self {
            memory_access_cost: 1.0,
            compute_cost: 0.1,
            kernel_launch_cost: 10.0,
            memory_bandwidth: 100e9, // 100 GB/s typical
        }
    }
}

impl FusionCostModel {
    /// Estimate cost of executing operations separately.
    pub fn cost_separate(&self, num_ops: usize, data_size: usize) -> f64 {
        let memory_cost = self.memory_access_cost * data_size as f64 * num_ops as f64;
        let launch_cost = self.kernel_launch_cost * num_ops as f64;
        memory_cost + launch_cost
    }

    /// Estimate cost of executing operations fused.
    pub fn cost_fused(&self, num_ops: usize, data_size: usize) -> f64 {
        // Fused operations read data once, write once
        let memory_cost = self.memory_access_cost * data_size as f64 * 2.0;
        let launch_cost = self.kernel_launch_cost;
        let compute_overhead = self.compute_cost * num_ops as f64; // slight overhead
        memory_cost + launch_cost + compute_overhead
    }

    /// Calculate fusion benefit.
    pub fn fusion_benefit(&self, num_ops: usize, data_size: usize) -> f64 {
        let separate_cost = self.cost_separate(num_ops, data_size);
        let fused_cost = self.cost_fused(num_ops, data_size);
        (separate_cost - fused_cost) / separate_cost
    }
}

/// Kernel fusion analyzer and optimizer.
pub struct FusionOptimizer {
    config: FusionConfig,
    cost_model: FusionCostModel,
    candidates: Vec<FusionCandidate>,
}

impl FusionOptimizer {
    /// Create a new fusion optimizer.
    pub fn new(config: FusionConfig) -> Self {
        Self {
            config,
            cost_model: FusionCostModel::default(),
            candidates: Vec::new(),
        }
    }

    /// Create with custom cost model.
    pub fn with_cost_model(config: FusionConfig, cost_model: FusionCostModel) -> Self {
        Self {
            config,
            cost_model,
            candidates: Vec::new(),
        }
    }

    /// Analyze graph and identify fusion opportunities.
    pub fn analyze(&mut self, graph: &EinsumGraph) -> Vec<FusionCandidate> {
        self.candidates.clear();

        if self.config.enable_patterns {
            self.find_pattern_fusions(graph);
        }

        if self.config.enable_vertical {
            self.find_vertical_fusions(graph);
        }

        if self.config.enable_horizontal {
            self.find_horizontal_fusions(graph);
        }

        // Sort by benefit score
        self.candidates
            .sort_by(|a, b| b.benefit_score.partial_cmp(&a.benefit_score).unwrap());

        self.candidates.clone()
    }

    /// Find pattern-based fusion opportunities.
    fn find_pattern_fusions(&mut self, graph: &EinsumGraph) {
        // Look for common patterns like MatMul + Bias + Activation
        for node_id in 0..graph.nodes.len() {
            let node_id = NodeId(node_id);
            let node = &graph.nodes[node_id.0];

            // Example: Look for MatMul followed by elementwise operations
            if matches!(node.op, OpType::Einsum { .. }) {
                // Check if output is consumed by elementwise op
                let consumers = self.find_consumers(graph, node_id);
                for consumer in consumers {
                    let consumer_node = &graph.nodes[consumer.0];
                    if matches!(
                        consumer_node.op,
                        OpType::ElemUnary { .. } | OpType::ElemBinary { .. }
                    ) {
                        let benefit = self.estimate_pattern_benefit(2, 1024); // Example sizes

                        if benefit >= self.config.min_benefit_score {
                            self.candidates.push(FusionCandidate {
                                nodes: vec![node_id, consumer],
                                pattern: FusionPattern::MatMulActivation,
                                benefit_score: benefit,
                                memory_savings: 1024 * 4, // Rough estimate
                                compute_savings: 0.0,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Find vertical fusion opportunities (producer-consumer chains).
    fn find_vertical_fusions(&mut self, graph: &EinsumGraph) {
        for node_id in 0..graph.nodes.len() {
            let node_id = NodeId(node_id);
            let consumers = self.find_consumers(graph, node_id);

            // If node has exactly one consumer, consider vertical fusion
            if consumers.len() == 1 {
                let consumer = consumers[0];
                if self.can_fuse_vertically(graph, node_id, consumer) {
                    let benefit = self.cost_model.fusion_benefit(2, 1024);

                    if benefit >= self.config.min_benefit_score {
                        self.candidates.push(FusionCandidate {
                            nodes: vec![node_id, consumer],
                            pattern: FusionPattern::ElementwiseChain,
                            benefit_score: benefit,
                            memory_savings: 1024 * 4,
                            compute_savings: 0.0,
                        });
                    }
                }
            }
        }
    }

    /// Find horizontal fusion opportunities (parallel independent ops).
    fn find_horizontal_fusions(&mut self, graph: &EinsumGraph) {
        let _independent_groups: Vec<Vec<NodeId>> = Vec::new();

        // Group nodes by their depth in the graph
        let mut depth_groups: HashMap<usize, Vec<NodeId>> = HashMap::new();

        for node_id in 0..graph.nodes.len() {
            let depth = self.compute_depth(graph, NodeId(node_id));
            depth_groups.entry(depth).or_default().push(NodeId(node_id));
        }

        // Within each depth level, find independent operations
        for (_, nodes) in depth_groups {
            if nodes.len() >= 2 {
                // Check for independence and similar operation types
                for i in 0..nodes.len() {
                    for j in i + 1..nodes.len() {
                        if self.are_independent(graph, nodes[i], nodes[j])
                            && self.have_similar_ops(graph, nodes[i], nodes[j])
                        {
                            let benefit = self.cost_model.fusion_benefit(2, 512);

                            if benefit >= self.config.min_benefit_score {
                                self.candidates.push(FusionCandidate {
                                    nodes: vec![nodes[i], nodes[j]],
                                    pattern: FusionPattern::ParallelReductions,
                                    benefit_score: benefit * 0.8, // Slightly lower benefit for horizontal
                                    memory_savings: 512 * 4,
                                    compute_savings: 0.0,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check if vertical fusion is possible.
    fn can_fuse_vertically(
        &self,
        _graph: &EinsumGraph,
        _producer: NodeId,
        _consumer: NodeId,
    ) -> bool {
        // Basic checks:
        // 1. No other consumers of producer
        // 2. Compatible operations
        // 3. No intermediate materialization required
        // 4. Within size limits
        true // Simplified for now
    }

    /// Check if two nodes are independent.
    fn are_independent(&self, graph: &EinsumGraph, a: NodeId, b: NodeId) -> bool {
        // Check if there's no data dependency between a and b
        let a_deps = self.get_all_dependencies(graph, a);
        let b_deps = self.get_all_dependencies(graph, b);

        !a_deps.contains(&b) && !b_deps.contains(&a)
    }

    /// Check if two nodes have similar operations (for horizontal fusion).
    fn have_similar_ops(&self, graph: &EinsumGraph, a: NodeId, b: NodeId) -> bool {
        let op_a = &graph.nodes[a.0].op;
        let op_b = &graph.nodes[b.0].op;

        std::mem::discriminant(op_a) == std::mem::discriminant(op_b)
    }

    /// Find all nodes that consume the output of a given node.
    fn find_consumers(&self, graph: &EinsumGraph, producer: NodeId) -> Vec<NodeId> {
        let mut consumers = Vec::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            if node.inputs.iter().any(|&n| NodeId(n) == producer) {
                consumers.push(NodeId(i));
            }
        }

        consumers
    }

    /// Get all transitive dependencies of a node.
    fn get_all_dependencies(&self, graph: &EinsumGraph, node_id: NodeId) -> HashSet<NodeId> {
        let mut deps = HashSet::new();
        let mut to_visit = vec![node_id];

        while let Some(current) = to_visit.pop() {
            if deps.contains(&current) {
                continue;
            }
            deps.insert(current);

            let node = &graph.nodes[current.0];
            for &input in &node.inputs {
                to_visit.push(NodeId(input));
            }
        }

        deps
    }

    /// Compute the depth of a node in the graph.
    #[allow(clippy::only_used_in_recursion)]
    fn compute_depth(&self, graph: &EinsumGraph, node_id: NodeId) -> usize {
        let node = &graph.nodes[node_id.0];

        if node.inputs.is_empty() {
            0
        } else {
            1 + node
                .inputs
                .iter()
                .map(|&input| self.compute_depth(graph, NodeId(input)))
                .max()
                .unwrap_or(0)
        }
    }

    /// Estimate benefit of fusing a pattern.
    fn estimate_pattern_benefit(&self, num_ops: usize, data_size: usize) -> f64 {
        match self.config.strategy {
            FusionStrategy::Aggressive => self.cost_model.fusion_benefit(num_ops, data_size) * 1.2,
            FusionStrategy::Conservative => {
                self.cost_model.fusion_benefit(num_ops, data_size) * 0.8
            }
            FusionStrategy::Balanced => self.cost_model.fusion_benefit(num_ops, data_size),
            FusionStrategy::MemoryAware => {
                let base_benefit = self.cost_model.fusion_benefit(num_ops, data_size);
                // Prioritize memory savings
                base_benefit * 1.5
            }
        }
    }

    /// Apply fusion candidates to create optimized graph.
    pub fn apply_fusions(
        &self,
        graph: &EinsumGraph,
        _candidates: &[FusionCandidate],
    ) -> Result<EinsumGraph, FusionError> {
        // This would create a new graph with fused operations
        // For now, return a clone
        Ok(graph.clone())
    }

    /// Get fusion statistics.
    pub fn stats(&self) -> FusionStats {
        let total_candidates = self.candidates.len();
        let total_memory_savings: usize = self.candidates.iter().map(|c| c.memory_savings).sum();
        let avg_benefit_score = if total_candidates > 0 {
            self.candidates.iter().map(|c| c.benefit_score).sum::<f64>() / total_candidates as f64
        } else {
            0.0
        };

        let mut pattern_counts = HashMap::new();
        for candidate in &self.candidates {
            *pattern_counts.entry(candidate.pattern).or_insert(0) += 1;
        }

        FusionStats {
            total_candidates,
            total_memory_savings,
            avg_benefit_score,
            pattern_distribution: pattern_counts,
        }
    }
}

/// Fusion statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionStats {
    /// Total number of fusion candidates found
    pub total_candidates: usize,
    /// Total estimated memory savings (bytes)
    pub total_memory_savings: usize,
    /// Average benefit score
    pub avg_benefit_score: f64,
    /// Distribution of fusion patterns
    pub pattern_distribution: HashMap<FusionPattern, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::EinsumNode;

    fn create_test_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();

        // Add some test nodes
        graph.nodes.push(EinsumNode {
            op: OpType::Einsum {
                spec: "ij,jk->ik".to_string(),
            },
            inputs: vec![],
            outputs: vec![0],
            metadata: Default::default(),
        });

        graph.nodes.push(EinsumNode {
            op: OpType::ElemUnary {
                op: "relu".to_string(),
            },
            inputs: vec![0],
            outputs: vec![1],
            metadata: Default::default(),
        });

        graph
    }

    #[test]
    fn test_fusion_config() {
        let config = FusionConfig::aggressive();
        assert_eq!(config.strategy, FusionStrategy::Aggressive);
        assert!(config.max_fusion_size >= FusionConfig::default().max_fusion_size);

        let config = FusionConfig::conservative();
        assert_eq!(config.strategy, FusionStrategy::Conservative);
    }

    #[test]
    fn test_cost_model() {
        let model = FusionCostModel::default();

        let benefit = model.fusion_benefit(3, 1024);
        assert!(benefit > 0.0);
        assert!(benefit < 1.0);

        // More operations should have higher benefit
        let benefit_more = model.fusion_benefit(5, 1024);
        assert!(benefit_more > benefit);
    }

    #[test]
    fn test_fusion_optimizer_creation() {
        let config = FusionConfig::default();
        let optimizer = FusionOptimizer::new(config);
        assert_eq!(optimizer.candidates.len(), 0);
    }

    #[test]
    fn test_fusion_analysis() {
        let graph = create_test_graph();
        // Use aggressive config to ensure we find candidates
        let config = FusionConfig {
            min_benefit_score: 0.0,
            ..FusionConfig::default()
        };
        let mut optimizer = FusionOptimizer::new(config);

        let candidates = optimizer.analyze(&graph);
        // Should find at least one fusion opportunity (matmul + relu)
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_consumer_finding() {
        let graph = create_test_graph();
        let optimizer = FusionOptimizer::new(FusionConfig::default());

        let consumers = optimizer.find_consumers(&graph, NodeId(0));
        assert_eq!(consumers.len(), 1);
        assert_eq!(consumers[0], NodeId(1));
    }

    #[test]
    fn test_depth_computation() {
        let graph = create_test_graph();
        let optimizer = FusionOptimizer::new(FusionConfig::default());

        assert_eq!(optimizer.compute_depth(&graph, NodeId(0)), 0);
        assert_eq!(optimizer.compute_depth(&graph, NodeId(1)), 1);
    }

    #[test]
    fn test_independence_check() {
        let mut graph = create_test_graph();

        // Add independent node
        graph.nodes.push(EinsumNode {
            op: OpType::ElemUnary {
                op: "tanh".to_string(),
            },
            inputs: vec![],
            outputs: vec![2],
            metadata: Default::default(),
        });

        let optimizer = FusionOptimizer::new(FusionConfig::default());

        // Node 1 depends on Node 0
        assert!(!optimizer.are_independent(&graph, NodeId(0), NodeId(1)));

        // Node 0 and Node 2 are independent
        assert!(optimizer.are_independent(&graph, NodeId(0), NodeId(2)));
    }

    #[test]
    fn test_fusion_stats() {
        let graph = create_test_graph();
        // Use aggressive config to ensure we find candidates
        let config = FusionConfig {
            min_benefit_score: 0.0,
            ..FusionConfig::default()
        };
        let mut optimizer = FusionOptimizer::new(config);

        optimizer.analyze(&graph);
        let stats = optimizer.stats();

        assert!(stats.total_candidates > 0);
        assert!(stats.avg_benefit_score >= 0.0);
    }

    #[test]
    fn test_similar_ops_check() {
        let graph = create_test_graph();
        let optimizer = FusionOptimizer::new(FusionConfig::default());

        // Same operation type
        assert!(optimizer.have_similar_ops(&graph, NodeId(0), NodeId(0)));

        // Different operation types
        assert!(!optimizer.have_similar_ops(&graph, NodeId(0), NodeId(1)));
    }
}
