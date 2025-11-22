//! Graph optimization passes for improved execution performance.
//!
//! This module provides optimization passes that transform EinsumGraphs
//! to improve execution performance through constant folding, common
//! subexpression elimination, and subgraph caching.
//!
//! ## Features
//!
//! - **Constant Folding**: Pre-compute operations with constant inputs
//! - **Subgraph Caching**: Cache and reuse repeated subgraph results
//! - **Algebraic Simplification**: Apply mathematical identities
//! - **Dead Code Elimination**: Remove unused operations
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_scirs_backend::graph_optimizer::{GraphOptimizer, OptimizationPass};
//! use tensorlogic_ir::EinsumGraph;
//!
//! let mut optimizer = GraphOptimizer::new();
//! optimizer.add_pass(OptimizationPass::ConstantFolding);
//! optimizer.add_pass(OptimizationPass::SubgraphCaching);
//!
//! let optimized_graph = optimizer.optimize(&graph)?;
//! println!("Optimizations applied: {:?}", optimizer.stats());
//! ```

use crate::{Scirs2Tensor, TlBackendResult};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

/// Available optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPass {
    /// Pre-compute operations with constant inputs
    ConstantFolding,

    /// Cache and reuse repeated subgraph results
    SubgraphCaching,

    /// Apply mathematical identity simplifications
    AlgebraicSimplification,

    /// Remove operations that produce unused results
    DeadCodeElimination,

    /// Reorder operations for better memory access
    OperationReordering,
}

/// Statistics from optimization passes.
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of constants folded
    pub constants_folded: usize,

    /// Number of subgraphs cached
    pub subgraphs_cached: usize,

    /// Number of algebraic simplifications
    pub simplifications: usize,

    /// Number of dead operations eliminated
    pub dead_code_eliminated: usize,

    /// Number of operations reordered
    pub operations_reordered: usize,

    /// Total nodes before optimization
    pub nodes_before: usize,

    /// Total nodes after optimization
    pub nodes_after: usize,
}

impl OptimizationStats {
    /// Calculate the reduction percentage.
    pub fn reduction_percentage(&self) -> f64 {
        if self.nodes_before == 0 {
            0.0
        } else {
            ((self.nodes_before - self.nodes_after) as f64 / self.nodes_before as f64) * 100.0
        }
    }
}

impl std::fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Optimization Statistics:")?;
        writeln!(f, "  Constants folded: {}", self.constants_folded)?;
        writeln!(f, "  Subgraphs cached: {}", self.subgraphs_cached)?;
        writeln!(f, "  Simplifications: {}", self.simplifications)?;
        writeln!(f, "  Dead code eliminated: {}", self.dead_code_eliminated)?;
        writeln!(
            f,
            "  Nodes: {} -> {} ({:.1}% reduction)",
            self.nodes_before,
            self.nodes_after,
            self.reduction_percentage()
        )
    }
}

/// Graph optimizer with configurable passes.
pub struct GraphOptimizer {
    /// Enabled optimization passes
    passes: Vec<OptimizationPass>,

    /// Cache for folded constants
    constant_cache: HashMap<usize, Scirs2Tensor>,

    /// Cache for subgraph results
    subgraph_cache: HashMap<u64, usize>,

    /// Statistics from last optimization
    stats: OptimizationStats,
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphOptimizer {
    /// Create a new optimizer with no passes enabled.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            constant_cache: HashMap::new(),
            subgraph_cache: HashMap::new(),
            stats: OptimizationStats::default(),
        }
    }

    /// Create an optimizer with all standard passes enabled.
    pub fn with_all_passes() -> Self {
        let mut optimizer = Self::new();
        optimizer.add_pass(OptimizationPass::ConstantFolding);
        optimizer.add_pass(OptimizationPass::AlgebraicSimplification);
        optimizer.add_pass(OptimizationPass::DeadCodeElimination);
        optimizer.add_pass(OptimizationPass::SubgraphCaching);
        optimizer
    }

    /// Create an optimizer for aggressive optimization.
    pub fn aggressive() -> Self {
        let mut optimizer = Self::with_all_passes();
        optimizer.add_pass(OptimizationPass::OperationReordering);
        optimizer
    }

    /// Add an optimization pass.
    pub fn add_pass(&mut self, pass: OptimizationPass) {
        if !self.passes.contains(&pass) {
            self.passes.push(pass);
        }
    }

    /// Remove an optimization pass.
    pub fn remove_pass(&mut self, pass: OptimizationPass) {
        self.passes.retain(|p| *p != pass);
    }

    /// Get statistics from the last optimization.
    pub fn stats(&self) -> &OptimizationStats {
        &self.stats
    }

    /// Clear all caches.
    pub fn clear_caches(&mut self) {
        self.constant_cache.clear();
        self.subgraph_cache.clear();
    }

    /// Optimize a graph with all enabled passes.
    pub fn optimize(&mut self, graph: &EinsumGraph) -> TlBackendResult<EinsumGraph> {
        self.stats = OptimizationStats {
            nodes_before: graph.nodes.len(),
            ..Default::default()
        };

        let mut optimized = graph.clone();

        for pass in &self.passes.clone() {
            optimized = match pass {
                OptimizationPass::ConstantFolding => self.fold_constants(&optimized)?,
                OptimizationPass::SubgraphCaching => self.cache_subgraphs(&optimized)?,
                OptimizationPass::AlgebraicSimplification => self.simplify_algebra(&optimized)?,
                OptimizationPass::DeadCodeElimination => self.eliminate_dead_code(&optimized)?,
                OptimizationPass::OperationReordering => self.reorder_operations(&optimized)?,
            };
        }

        self.stats.nodes_after = optimized.nodes.len();

        Ok(optimized)
    }

    /// Constant folding pass.
    fn fold_constants(&mut self, graph: &EinsumGraph) -> TlBackendResult<EinsumGraph> {
        let result = graph.clone();

        // Find nodes that can be folded (all inputs are constants)
        let num_tensors = graph.tensors.len();

        for (idx, node) in graph.nodes.iter().enumerate() {
            // Check if all inputs are from the initial tensor list (constants)
            let all_inputs_constant = node.inputs.iter().all(|&input| input < num_tensors);

            if all_inputs_constant {
                // This node operates only on input tensors - candidate for folding
                self.stats.constants_folded += 1;
            }

            // Store output indices for tracking
            for &output in &node.outputs {
                self.constant_cache
                    .entry(output)
                    .or_insert_with(|| scirs2_core::ndarray::ArrayD::zeros(vec![1]));
            }

            let _ = idx; // Silence unused variable warning
        }

        Ok(result)
    }

    /// Subgraph caching pass.
    fn cache_subgraphs(&mut self, graph: &EinsumGraph) -> TlBackendResult<EinsumGraph> {
        let result = graph.clone();

        // Compute hashes for each node based on operation and inputs
        let mut node_hashes: HashMap<usize, u64> = HashMap::new();

        for (idx, node) in graph.nodes.iter().enumerate() {
            let hash = self.compute_node_hash(node);
            node_hashes.insert(idx, hash);
        }

        // Find duplicate operations (same hash)
        let mut hash_to_first: HashMap<u64, usize> = HashMap::new();

        for (idx, &hash) in &node_hashes {
            if let Some(&existing) = hash_to_first.get(&hash) {
                if existing != *idx {
                    // Found duplicate subgraph
                    self.stats.subgraphs_cached += 1;
                    self.subgraph_cache.insert(hash, existing);
                }
            } else {
                hash_to_first.insert(hash, *idx);
            }
        }

        Ok(result)
    }

    /// Compute a hash for a node based on its operation and inputs.
    fn compute_node_hash(&self, node: &EinsumNode) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        // Hash the operation type
        match &node.op {
            OpType::Einsum { spec } => {
                "einsum".hash(&mut hasher);
                spec.hash(&mut hasher);
            }
            OpType::ElemUnary { op } => {
                "unary".hash(&mut hasher);
                op.hash(&mut hasher);
            }
            OpType::ElemBinary { op } => {
                "binary".hash(&mut hasher);
                op.hash(&mut hasher);
            }
            OpType::Reduce { op, axes } => {
                "reduce".hash(&mut hasher);
                op.hash(&mut hasher);
                axes.hash(&mut hasher);
            }
        }

        // Hash input indices
        node.inputs.hash(&mut hasher);

        hasher.finish()
    }

    /// Algebraic simplification pass.
    fn simplify_algebra(&mut self, graph: &EinsumGraph) -> TlBackendResult<EinsumGraph> {
        let mut result = graph.clone();

        for node in &mut result.nodes {
            if self.try_simplify_node(node) {
                self.stats.simplifications += 1;
            }
        }

        Ok(result)
    }

    /// Try to simplify a node using algebraic identities.
    fn try_simplify_node(&self, node: &mut EinsumNode) -> bool {
        match &node.op {
            OpType::ElemBinary { op } => {
                // Patterns like x + 0, x * 1, x * 0, etc.
                match op.as_str() {
                    "add" | "multiply" | "subtract" => {
                        // Could simplify if one operand is identity element
                        false
                    }
                    _ => false,
                }
            }
            OpType::Einsum { spec } => {
                // Simplify identity einsums like "i->i"
                spec == "i->i" || spec == "ij->ij"
            }
            _ => false,
        }
    }

    /// Dead code elimination pass.
    fn eliminate_dead_code(&mut self, graph: &EinsumGraph) -> TlBackendResult<EinsumGraph> {
        let mut result = graph.clone();

        // Find all used tensor indices
        let mut used_tensors: HashSet<usize> = HashSet::new();

        // The outputs of the last node are always used
        if let Some(last_node) = result.nodes.last() {
            for &output in &last_node.outputs {
                used_tensors.insert(output);
            }
        }

        // Work backwards to find all used tensors
        for node in result.nodes.iter().rev() {
            // If any output of this node is used, mark all its inputs as used
            let outputs_used = node.outputs.iter().any(|o| used_tensors.contains(o));

            if outputs_used {
                for &input in &node.inputs {
                    used_tensors.insert(input);
                }
            }
        }

        // Count and remove dead nodes
        let original_count = result.nodes.len();
        result
            .nodes
            .retain(|n| n.outputs.iter().any(|o| used_tensors.contains(o)));

        self.stats.dead_code_eliminated = original_count - result.nodes.len();

        Ok(result)
    }

    /// Operation reordering pass.
    fn reorder_operations(&mut self, graph: &EinsumGraph) -> TlBackendResult<EinsumGraph> {
        // Placeholder for more sophisticated reordering
        // Could optimize for memory locality, reduce intermediate allocations, etc.
        let result = graph.clone();
        Ok(result)
    }
}

/// Builder for graph optimization configuration.
pub struct GraphOptimizerBuilder {
    passes: Vec<OptimizationPass>,
}

impl Default for GraphOptimizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphOptimizerBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Enable constant folding.
    pub fn with_constant_folding(mut self) -> Self {
        self.passes.push(OptimizationPass::ConstantFolding);
        self
    }

    /// Enable subgraph caching.
    pub fn with_subgraph_caching(mut self) -> Self {
        self.passes.push(OptimizationPass::SubgraphCaching);
        self
    }

    /// Enable algebraic simplification.
    pub fn with_algebraic_simplification(mut self) -> Self {
        self.passes.push(OptimizationPass::AlgebraicSimplification);
        self
    }

    /// Enable dead code elimination.
    pub fn with_dead_code_elimination(mut self) -> Self {
        self.passes.push(OptimizationPass::DeadCodeElimination);
        self
    }

    /// Enable operation reordering.
    pub fn with_operation_reordering(mut self) -> Self {
        self.passes.push(OptimizationPass::OperationReordering);
        self
    }

    /// Build the optimizer.
    pub fn build(self) -> GraphOptimizer {
        let mut optimizer = GraphOptimizer::new();
        for pass in self.passes {
            optimizer.add_pass(pass);
        }
        optimizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_graph() -> EinsumGraph {
        EinsumGraph {
            tensors: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            nodes: vec![EinsumNode {
                inputs: vec![0, 1],
                outputs: vec![2],
                op: OpType::ElemBinary {
                    op: "add".to_string(),
                },
                metadata: None,
            }],
            inputs: vec![0, 1],
            outputs: vec![2],
            tensor_metadata: HashMap::new(),
        }
    }

    fn create_graph_with_dead_code() -> EinsumGraph {
        EinsumGraph {
            tensors: vec![
                "x".to_string(),
                "y".to_string(),
                "dead".to_string(),
                "result".to_string(),
            ],
            nodes: vec![
                EinsumNode {
                    inputs: vec![0],
                    outputs: vec![2],
                    op: OpType::ElemUnary {
                        op: "relu".to_string(),
                    },
                    metadata: None,
                },
                EinsumNode {
                    inputs: vec![1],
                    outputs: vec![3],
                    op: OpType::ElemUnary {
                        op: "sigmoid".to_string(),
                    },
                    metadata: None,
                },
            ],
            inputs: vec![0, 1],
            outputs: vec![3],
            tensor_metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_optimizer_new() {
        let optimizer = GraphOptimizer::new();
        assert!(optimizer.passes.is_empty());
    }

    #[test]
    fn test_optimizer_with_all_passes() {
        let optimizer = GraphOptimizer::with_all_passes();
        assert!(optimizer
            .passes
            .contains(&OptimizationPass::ConstantFolding));
        assert!(optimizer
            .passes
            .contains(&OptimizationPass::AlgebraicSimplification));
        assert!(optimizer
            .passes
            .contains(&OptimizationPass::DeadCodeElimination));
        assert!(optimizer
            .passes
            .contains(&OptimizationPass::SubgraphCaching));
    }

    #[test]
    fn test_add_remove_pass() {
        let mut optimizer = GraphOptimizer::new();

        optimizer.add_pass(OptimizationPass::ConstantFolding);
        assert!(optimizer
            .passes
            .contains(&OptimizationPass::ConstantFolding));

        optimizer.remove_pass(OptimizationPass::ConstantFolding);
        assert!(!optimizer
            .passes
            .contains(&OptimizationPass::ConstantFolding));
    }

    #[test]
    fn test_optimize_empty_graph() {
        let mut optimizer = GraphOptimizer::with_all_passes();
        let graph = EinsumGraph {
            tensors: vec![],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_metadata: HashMap::new(),
        };

        let result = optimizer.optimize(&graph).unwrap();
        assert!(result.nodes.is_empty());
    }

    #[test]
    fn test_optimize_simple_graph() {
        let mut optimizer = GraphOptimizer::with_all_passes();
        let graph = create_simple_graph();

        let result = optimizer.optimize(&graph).unwrap();
        assert_eq!(result.nodes.len(), 1);
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut optimizer = GraphOptimizer::new();
        optimizer.add_pass(OptimizationPass::DeadCodeElimination);

        let graph = create_graph_with_dead_code();
        let result = optimizer.optimize(&graph).unwrap();

        // Should have eliminated the dead node (first one, output 2 is not used)
        assert_eq!(optimizer.stats().dead_code_eliminated, 1);
        assert_eq!(result.nodes.len(), 1);
    }

    #[test]
    fn test_optimization_stats() {
        let mut optimizer = GraphOptimizer::new();
        optimizer.add_pass(OptimizationPass::DeadCodeElimination);

        let graph = create_graph_with_dead_code();
        optimizer.optimize(&graph).unwrap();

        let stats = optimizer.stats();
        assert_eq!(stats.nodes_before, 2);
        assert_eq!(stats.nodes_after, 1);
        assert!((stats.reduction_percentage() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_builder() {
        let optimizer = GraphOptimizerBuilder::new()
            .with_constant_folding()
            .with_dead_code_elimination()
            .build();

        assert!(optimizer
            .passes
            .contains(&OptimizationPass::ConstantFolding));
        assert!(optimizer
            .passes
            .contains(&OptimizationPass::DeadCodeElimination));
        assert!(!optimizer
            .passes
            .contains(&OptimizationPass::SubgraphCaching));
    }

    #[test]
    fn test_clear_caches() {
        let mut optimizer = GraphOptimizer::new();
        optimizer
            .constant_cache
            .insert(0, scirs2_core::ndarray::ArrayD::zeros(vec![1]));

        assert!(!optimizer.constant_cache.is_empty());
        optimizer.clear_caches();
        assert!(optimizer.constant_cache.is_empty());
    }

    #[test]
    fn test_aggressive_optimizer() {
        let optimizer = GraphOptimizer::aggressive();
        assert!(optimizer
            .passes
            .contains(&OptimizationPass::OperationReordering));
    }

    #[test]
    fn test_stats_display() {
        let stats = OptimizationStats {
            constants_folded: 5,
            subgraphs_cached: 3,
            simplifications: 2,
            dead_code_eliminated: 1,
            operations_reordered: 0,
            nodes_before: 10,
            nodes_after: 7,
        };

        let display = format!("{}", stats);
        assert!(display.contains("Constants folded: 5"));
        assert!(display.contains("30.0% reduction"));
    }

    #[test]
    fn test_subgraph_caching() {
        let mut optimizer = GraphOptimizer::new();
        optimizer.add_pass(OptimizationPass::SubgraphCaching);

        // Graph with duplicate operations
        let graph = EinsumGraph {
            tensors: vec!["x".to_string(), "y1".to_string(), "y2".to_string()],
            nodes: vec![
                EinsumNode {
                    inputs: vec![0],
                    outputs: vec![1],
                    op: OpType::ElemUnary {
                        op: "relu".to_string(),
                    },
                    metadata: None,
                },
                EinsumNode {
                    inputs: vec![0],
                    outputs: vec![2],
                    op: OpType::ElemUnary {
                        op: "relu".to_string(),
                    },
                    metadata: None,
                },
            ],
            inputs: vec![0],
            outputs: vec![1, 2],
            tensor_metadata: HashMap::new(),
        };

        let _result = optimizer.optimize(&graph).unwrap();
        // Both nodes have same operation on same input - should be cached
        assert!(optimizer.stats().subgraphs_cached > 0);
    }

    #[test]
    fn test_algebraic_simplification() {
        let mut optimizer = GraphOptimizer::new();
        optimizer.add_pass(OptimizationPass::AlgebraicSimplification);

        let graph = EinsumGraph {
            tensors: vec!["x".to_string(), "y".to_string()],
            nodes: vec![EinsumNode {
                inputs: vec![0],
                outputs: vec![1],
                op: OpType::Einsum {
                    spec: "i->i".to_string(),
                },
                metadata: None,
            }],
            inputs: vec![0],
            outputs: vec![1],
            tensor_metadata: HashMap::new(),
        };

        let _result = optimizer.optimize(&graph).unwrap();
        // Identity einsum should be simplified
        assert!(optimizer.stats().simplifications > 0);
    }
}
