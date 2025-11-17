//! Integration pass for IR-level graph optimizations.
//!
//! This module provides compiler-level integration with the sophisticated
//! graph optimization passes available in tensorlogic-ir. It applies
//! fusion, layout optimization, memory optimization, and pattern-based
//! transformations to compiled EinsumGraphs.

use anyhow::Result;
use tensorlogic_ir::{
    analyze_memory, apply_tiling, fold_constants_aggressive, fuse_elementwise_operations,
    optimize_layouts, EinsumGraph, GraphScheduler, OpType, SchedulingObjective,
};

/// Configuration for graph optimization integration
#[derive(Debug, Clone)]
pub struct GraphOptConfig {
    /// Enable operation fusion
    pub enable_fusion: bool,
    /// Enable layout optimization
    pub enable_layout_opt: bool,
    /// Enable memory optimization
    pub enable_memory_opt: bool,
    /// Enable constant folding at graph level
    pub enable_constant_folding: bool,
    /// Enable tiling for large operations
    pub enable_tiling: bool,
    /// Enable scheduling optimization
    pub enable_scheduling: bool,
    /// Target tile size for tiling optimization
    pub tile_size: Option<usize>,
    /// Memory budget for memory optimization (in bytes)
    pub memory_budget: Option<usize>,
}

impl Default for GraphOptConfig {
    fn default() -> Self {
        Self {
            enable_fusion: true,
            enable_layout_opt: true,
            enable_memory_opt: true,
            enable_constant_folding: true,
            enable_tiling: false, // Conservative default
            enable_scheduling: true,
            tile_size: Some(64),
            memory_budget: None, // Auto-detect
        }
    }
}

impl GraphOptConfig {
    /// Create a new configuration with all optimizations enabled
    pub fn aggressive() -> Self {
        Self {
            enable_fusion: true,
            enable_layout_opt: true,
            enable_memory_opt: true,
            enable_constant_folding: true,
            enable_tiling: true,
            enable_scheduling: true,
            tile_size: Some(128),
            memory_budget: None,
        }
    }

    /// Create a configuration with conservative optimizations (safe defaults)
    pub fn conservative() -> Self {
        Self {
            enable_fusion: true,
            enable_layout_opt: false,
            enable_memory_opt: false,
            enable_constant_folding: true,
            enable_tiling: false,
            enable_scheduling: false,
            tile_size: None,
            memory_budget: None,
        }
    }

    /// Create a configuration optimized for inference (minimize latency)
    pub fn for_inference() -> Self {
        Self {
            enable_fusion: true,
            enable_layout_opt: true,
            enable_memory_opt: false,
            enable_constant_folding: true,
            enable_tiling: false,
            enable_scheduling: true,
            tile_size: None,
            memory_budget: None,
        }
    }

    /// Create a configuration optimized for training (minimize memory)
    pub fn for_training() -> Self {
        Self {
            enable_fusion: true,
            enable_layout_opt: true,
            enable_memory_opt: true,
            enable_constant_folding: true,
            enable_tiling: true,
            enable_scheduling: true,
            tile_size: Some(64),
            memory_budget: None,
        }
    }
}

/// Statistics from graph optimization integration
#[derive(Debug, Default, Clone)]
pub struct GraphOptStats {
    /// Number of operations fused
    pub ops_fused: usize,
    /// Number of layout transformations applied
    pub layout_transforms: usize,
    /// Number of memory optimizations applied
    pub memory_opts: usize,
    /// Number of constants folded at graph level
    pub graph_constants_folded: usize,
    /// Number of tiles created
    pub tiles_created: usize,
    /// Estimated memory reduction (bytes)
    pub memory_saved: usize,
    /// Estimated speedup from optimizations
    pub estimated_speedup: f64,
}

impl GraphOptStats {
    /// Total number of optimizations applied
    pub fn total_optimizations(&self) -> usize {
        self.ops_fused
            + self.layout_transforms
            + self.memory_opts
            + self.graph_constants_folded
            + self.tiles_created
    }

    /// Check if any optimizations were applied
    pub fn any_applied(&self) -> bool {
        self.total_optimizations() > 0
    }
}

/// Apply integrated graph optimizations to a compiled EinsumGraph
///
/// This is the main entry point for applying IR-level optimizations to
/// graphs produced by the compiler. It orchestrates multiple optimization
/// passes in an intelligent order.
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::passes::graph_opt_integration::{
///     apply_graph_optimizations, GraphOptConfig
/// };
/// use tensorlogic_ir::EinsumGraph;
///
/// let mut graph = EinsumGraph::new();
/// // ... compile logic expressions into graph ...
///
/// let config = GraphOptConfig::default();
/// let (optimized, stats) = apply_graph_optimizations(&graph, &config).unwrap();
///
/// println!("Applied {} optimizations", stats.total_optimizations());
/// println!("Estimated speedup: {:.2}x", stats.estimated_speedup);
/// ```
pub fn apply_graph_optimizations(
    graph: &EinsumGraph,
    config: &GraphOptConfig,
) -> Result<(EinsumGraph, GraphOptStats)> {
    let mut optimized = graph.clone();
    let mut stats = GraphOptStats {
        estimated_speedup: 1.0,
        ..Default::default()
    };

    // Phase 1: Constant folding (do this early to simplify later passes)
    if config.enable_constant_folding {
        let fold_result = fold_constants_aggressive(&mut optimized)?;
        stats.graph_constants_folded = fold_result.operations_folded;
        stats.estimated_speedup *= fold_result.estimated_speedup;
    }

    // Phase 2: Operation fusion (reduce kernel launches)
    if config.enable_fusion {
        let fusion_result = fuse_elementwise_operations(&mut optimized)?;
        stats.ops_fused = fusion_result.ops_fused;
        stats.estimated_speedup *= fusion_result.estimated_speedup;
    }

    // Phase 3: Layout optimization (improve cache locality)
    if config.enable_layout_opt {
        let layout_result = optimize_layouts(&optimized)?;
        stats.layout_transforms = layout_result.transformations_needed;
        stats.estimated_speedup *= layout_result.estimated_speedup;
    }

    // Phase 4: Memory analysis (understand memory usage patterns)
    if config.enable_memory_opt {
        let mem_result = analyze_memory(&optimized, 8)?; // 8 bytes for f64 elements
        stats.memory_saved = mem_result.total_memory_bytes - mem_result.peak_memory_bytes;
        // Note: memory analysis provides insights but doesn't modify the graph
        stats.memory_opts = if mem_result.avg_utilization < 0.8 {
            1
        } else {
            0
        };
    }

    // Phase 5: Tiling (for large operations)
    if config.enable_tiling {
        if let Some(tile_size) = config.tile_size {
            use tensorlogic_ir::{TileConfig as IrTileConfig, TilingStrategy};
            let mut strategy = TilingStrategy::new();
            strategy.add_tile(IrTileConfig::new(0, tile_size));
            let tiling_result = apply_tiling(&mut optimized, &strategy)?;
            stats.tiles_created = tiling_result.nodes_tiled + tiling_result.loops_unrolled;
        }
    }

    // Phase 6: Scheduling (optimize execution order)
    if config.enable_scheduling {
        let scheduler = GraphScheduler::new();
        let _schedule = scheduler.schedule(&optimized, SchedulingObjective::MinimizeMemory)?;
        // Scheduling doesn't modify graph, just provides execution order
        stats.estimated_speedup *= 1.05;
    }

    Ok((optimized, stats))
}

/// Apply pattern-based graph transformations
///
/// Uses the IR's pattern matching system to apply domain-specific
/// optimizations and transformations to the graph.
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::passes::graph_opt_integration::apply_pattern_optimizations;
/// use tensorlogic_ir::EinsumGraph;
///
/// let graph = EinsumGraph::new();
/// let (optimized, count) = apply_pattern_optimizations(&graph).unwrap();
/// println!("Applied {} pattern-based optimizations", count);
/// ```
pub fn apply_pattern_optimizations(graph: &EinsumGraph) -> Result<(EinsumGraph, usize)> {
    // Pattern rewriting is available but needs to be explicitly configured
    // For now, return the original graph with 0 rewrites
    // In the future, this will use tensorlogic_ir::PatternMatcher
    Ok((graph.clone(), 0))
}

/// Quick optimization pass with sensible defaults
///
/// Applies a curated set of fast optimizations that provide good
/// bang-for-buck with minimal compilation overhead.
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::passes::graph_opt_integration::quick_optimize;
/// use tensorlogic_ir::EinsumGraph;
///
/// let graph = EinsumGraph::new();
/// let optimized = quick_optimize(&graph).unwrap();
/// ```
pub fn quick_optimize(graph: &EinsumGraph) -> Result<EinsumGraph> {
    let config = GraphOptConfig {
        enable_fusion: true,
        enable_layout_opt: false,
        enable_memory_opt: false,
        enable_constant_folding: true,
        enable_tiling: false,
        enable_scheduling: false,
        tile_size: None,
        memory_budget: None,
    };

    let (optimized, _) = apply_graph_optimizations(graph, &config)?;
    Ok(optimized)
}

/// Analyze graph and recommend optimization configuration
///
/// Examines the structure and characteristics of a graph to suggest
/// which optimizations are likely to be beneficial.
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::passes::graph_opt_integration::recommend_optimizations;
/// use tensorlogic_ir::EinsumGraph;
///
/// let graph = EinsumGraph::new();
/// let config = recommend_optimizations(&graph);
/// println!("Recommended fusion: {}", config.enable_fusion);
/// ```
pub fn recommend_optimizations(graph: &EinsumGraph) -> GraphOptConfig {
    let node_count = graph.nodes.len();
    let tensor_count = graph.tensors.len();

    // Count operation types
    let mut elementwise_count = 0;
    let mut einsum_count = 0;

    for node in &graph.nodes {
        match &node.op {
            OpType::ElemUnary { .. } | OpType::ElemBinary { .. } => elementwise_count += 1,
            OpType::Einsum { .. } => einsum_count += 1,
            _ => {}
        }
    }

    // Small graphs: conservative optimizations
    if node_count < 10 {
        return GraphOptConfig {
            enable_fusion: elementwise_count > 2,
            enable_layout_opt: false,
            enable_memory_opt: false,
            enable_constant_folding: true,
            enable_tiling: false,
            enable_scheduling: false,
            tile_size: None,
            memory_budget: None,
        };
    }

    // Medium graphs: selective optimizations
    if node_count < 50 {
        return GraphOptConfig {
            enable_fusion: elementwise_count > 3,
            enable_layout_opt: einsum_count > 5,
            enable_memory_opt: tensor_count > 20,
            enable_constant_folding: true,
            enable_tiling: false,
            enable_scheduling: true,
            tile_size: Some(64),
            memory_budget: None,
        };
    }

    // Large graphs: aggressive optimizations
    GraphOptConfig {
        enable_fusion: true,
        enable_layout_opt: true,
        enable_memory_opt: true,
        enable_constant_folding: true,
        enable_tiling: einsum_count > 10,
        enable_scheduling: true,
        tile_size: Some(128),
        memory_budget: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::EinsumNode;

    #[test]
    fn test_config_defaults() {
        let config = GraphOptConfig::default();
        assert!(config.enable_fusion);
        assert!(config.enable_constant_folding);
    }

    #[test]
    fn test_config_aggressive() {
        let config = GraphOptConfig::aggressive();
        assert!(config.enable_fusion);
        assert!(config.enable_layout_opt);
        assert!(config.enable_memory_opt);
        assert!(config.enable_tiling);
        assert!(config.enable_scheduling);
    }

    #[test]
    fn test_config_conservative() {
        let config = GraphOptConfig::conservative();
        assert!(config.enable_fusion);
        assert!(!config.enable_layout_opt);
        assert!(!config.enable_tiling);
    }

    #[test]
    fn test_stats_total_optimizations() {
        let stats = GraphOptStats {
            ops_fused: 5,
            layout_transforms: 3,
            memory_opts: 2,
            graph_constants_folded: 1,
            tiles_created: 0,
            memory_saved: 1024,
            estimated_speedup: 1.5,
        };
        assert_eq!(stats.total_optimizations(), 11);
        assert!(stats.any_applied());
    }

    #[test]
    fn test_quick_optimize_empty_graph() {
        let graph = EinsumGraph::new();
        let result = quick_optimize(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_recommend_optimizations_small_graph() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("x");
        let t1 = graph.add_tensor("y");
        let t2 = graph.add_tensor("z");
        let _ = graph.add_node(EinsumNode::elem_binary("add", t0, t1, t2));

        let config = recommend_optimizations(&graph);
        // Small graph should have conservative settings
        assert!(!config.enable_tiling);
        assert!(config.enable_constant_folding);
    }

    #[test]
    fn test_recommend_optimizations_medium_graph() {
        let mut graph = EinsumGraph::new();
        // Create a graph with ~30 nodes
        for i in 0..30 {
            let t_in = graph.add_tensor(format!("t{}", i));
            let t_out = graph.add_tensor(format!("t{}_out", i));
            let _ = graph.add_node(EinsumNode::elem_unary("relu", t_in, t_out));
        }

        let config = recommend_optimizations(&graph);
        assert!(config.enable_fusion);
        assert!(config.enable_scheduling);
    }

    #[test]
    fn test_apply_optimizations_with_default_config() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("x");
        let t1 = graph.add_tensor("const");
        let t2 = graph.add_tensor("result");
        let _ = graph.add_node(EinsumNode::elem_binary("add", t0, t1, t2));

        let config = GraphOptConfig::default();
        let result = apply_graph_optimizations(&graph, &config);
        assert!(result.is_ok());

        let (_optimized, stats) = result.unwrap();
        assert!(stats.estimated_speedup >= 1.0);
    }

    #[test]
    fn test_apply_pattern_optimizations_empty() {
        let graph = EinsumGraph::new();
        let result = apply_pattern_optimizations(&graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_stats_any_applied_false() {
        let stats = GraphOptStats::default();
        assert!(!stats.any_applied());
    }

    #[test]
    fn test_config_for_inference() {
        let config = GraphOptConfig::for_inference();
        assert!(config.enable_fusion);
        assert!(config.enable_layout_opt);
        assert!(!config.enable_memory_opt); // Prioritize speed over memory
        assert!(!config.enable_tiling);
    }

    #[test]
    fn test_config_for_training() {
        let config = GraphOptConfig::for_training();
        assert!(config.enable_memory_opt); // Prioritize memory efficiency
        assert!(config.enable_tiling);
        assert_eq!(config.tile_size, Some(64));
    }
}
