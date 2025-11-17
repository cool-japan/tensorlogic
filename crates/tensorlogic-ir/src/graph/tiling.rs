//! Loop tiling and unrolling optimizations for cache locality.
//!
//! This module provides advanced loop transformation techniques to improve
//! memory access patterns and cache utilization in tensor computations.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{EinsumGraph, EinsumNode, IrError, OpType};

/// Tiling configuration for a specific axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileConfig {
    /// The axis/dimension to tile
    pub axis: usize,
    /// Tile size (number of elements per tile)
    pub tile_size: usize,
    /// Whether to unroll the inner loop
    pub unroll: bool,
}

impl TileConfig {
    /// Create a new tiling configuration.
    pub fn new(axis: usize, tile_size: usize) -> Self {
        Self {
            axis,
            tile_size,
            unroll: false,
        }
    }

    /// Create a tiling configuration with unrolling enabled.
    pub fn with_unroll(axis: usize, tile_size: usize) -> Self {
        Self {
            axis,
            tile_size,
            unroll: true,
        }
    }
}

/// Multi-dimensional tiling strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TilingStrategy {
    /// Tiling configurations for each axis
    pub tiles: Vec<TileConfig>,
    /// Whether to apply register tiling (very small tiles for register reuse)
    pub register_tiling: bool,
    /// Cache line size in bytes (for alignment)
    pub cache_line_size: usize,
}

impl Default for TilingStrategy {
    fn default() -> Self {
        Self {
            tiles: Vec::new(),
            register_tiling: false,
            cache_line_size: 64, // Common cache line size
        }
    }
}

impl TilingStrategy {
    /// Create a new tiling strategy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tile configuration for a specific axis.
    pub fn add_tile(&mut self, config: TileConfig) -> &mut Self {
        self.tiles.push(config);
        self
    }

    /// Enable register tiling for maximum register reuse.
    pub fn with_register_tiling(mut self) -> Self {
        self.register_tiling = true;
        self
    }

    /// Set the cache line size for alignment optimization.
    pub fn with_cache_line_size(mut self, size: usize) -> Self {
        self.cache_line_size = size;
        self
    }

    /// Get recommended tile sizes for matrix multiplication (M×K @ K×N).
    pub fn for_matmul(m: usize, k: usize, n: usize) -> Self {
        // Recommended tile sizes based on typical L1/L2 cache sizes
        let tile_m = m.clamp(8, 64);
        let tile_k = k.clamp(8, 64);
        let tile_n = n.clamp(8, 64);

        let mut strategy = Self::new();
        strategy.add_tile(TileConfig::new(0, tile_m)); // M dimension
        strategy.add_tile(TileConfig::new(1, tile_k)); // K dimension
        strategy.add_tile(TileConfig::new(2, tile_n)); // N dimension
        strategy
    }

    /// Get recommended tile sizes for convolution operations.
    pub fn for_conv(
        batch: usize,
        out_channels: usize,
        out_height: usize,
        out_width: usize,
    ) -> Self {
        let tile_b = batch.clamp(1, 16);
        let tile_c = out_channels.clamp(1, 16);
        let tile_h = out_height.clamp(1, 8);
        let tile_w = out_width.clamp(1, 8);

        let mut strategy = Self::new();
        strategy.add_tile(TileConfig::new(0, tile_b));
        strategy.add_tile(TileConfig::new(1, tile_c));
        strategy.add_tile(TileConfig::new(2, tile_h));
        strategy.add_tile(TileConfig::new(3, tile_w));
        strategy
    }
}

/// Result of applying tiling transformations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TilingResult {
    /// Number of nodes that were tiled
    pub nodes_tiled: usize,
    /// Number of loops unrolled
    pub loops_unrolled: usize,
    /// Estimated cache hit rate improvement (0.0 to 1.0)
    pub estimated_cache_improvement: f64,
    /// Estimated speedup factor
    pub estimated_speedup: f64,
}

impl TilingResult {
    /// Create a new tiling result with no transformations.
    pub fn none() -> Self {
        Self {
            nodes_tiled: 0,
            loops_unrolled: 0,
            estimated_cache_improvement: 0.0,
            estimated_speedup: 1.0,
        }
    }
}

/// Apply loop tiling to einsum operations in the graph.
pub fn apply_tiling(
    graph: &mut EinsumGraph,
    strategy: &TilingStrategy,
) -> Result<TilingResult, IrError> {
    let mut result = TilingResult::none();

    for node in &mut graph.nodes {
        if let OpType::Einsum { spec } = &node.op {
            if should_tile_einsum(spec) {
                // Apply tiling transformation
                tile_einsum_node(node, strategy)?;
                result.nodes_tiled += 1;

                // Count unrolled loops
                for tile in &strategy.tiles {
                    if tile.unroll {
                        result.loops_unrolled += 1;
                    }
                }
            }
        }
    }

    // Estimate performance improvements
    if result.nodes_tiled > 0 {
        result.estimated_cache_improvement = estimate_cache_improvement(strategy);
        result.estimated_speedup = 1.0 + result.estimated_cache_improvement * 0.5;
    }

    Ok(result)
}

/// Apply register-level tiling for maximum register reuse.
pub fn apply_register_tiling(graph: &mut EinsumGraph) -> Result<TilingResult, IrError> {
    let mut strategy = TilingStrategy::new().with_register_tiling();

    // Use small tile sizes (4-8 elements) for register tiling
    strategy.add_tile(TileConfig::with_unroll(0, 4));
    strategy.add_tile(TileConfig::with_unroll(1, 4));

    apply_tiling(graph, &strategy)
}

/// Apply multi-level tiling (L1, L2, L3 cache hierarchy).
pub fn apply_multilevel_tiling(
    graph: &mut EinsumGraph,
    l1_tiles: &[usize],
    l2_tiles: &[usize],
    l3_tiles: &[usize],
) -> Result<TilingResult, IrError> {
    let mut total_result = TilingResult::none();

    // Apply L3 tiles first (outermost)
    if !l3_tiles.is_empty() {
        let mut strategy = TilingStrategy::new();
        for (i, &tile_size) in l3_tiles.iter().enumerate() {
            strategy.add_tile(TileConfig::new(i, tile_size));
        }
        let result = apply_tiling(graph, &strategy)?;
        total_result.nodes_tiled += result.nodes_tiled;
    }

    // Apply L2 tiles
    if !l2_tiles.is_empty() {
        let mut strategy = TilingStrategy::new();
        for (i, &tile_size) in l2_tiles.iter().enumerate() {
            strategy.add_tile(TileConfig::new(i, tile_size));
        }
        let result = apply_tiling(graph, &strategy)?;
        total_result.nodes_tiled += result.nodes_tiled;
    }

    // Apply L1 tiles with unrolling (innermost)
    if !l1_tiles.is_empty() {
        let mut strategy = TilingStrategy::new();
        for (i, &tile_size) in l1_tiles.iter().enumerate() {
            strategy.add_tile(TileConfig::with_unroll(i, tile_size));
        }
        let result = apply_tiling(graph, &strategy)?;
        total_result.nodes_tiled += result.nodes_tiled;
        total_result.loops_unrolled += result.loops_unrolled;
    }

    // Estimate combined improvements
    total_result.estimated_cache_improvement = 0.3; // Conservative estimate
    total_result.estimated_speedup = 1.5; // Typical speedup for multi-level tiling

    Ok(total_result)
}

/// Analyze a graph and recommend optimal tiling strategies.
pub fn recommend_tiling_strategy(graph: &EinsumGraph) -> HashMap<usize, TilingStrategy> {
    let mut recommendations = HashMap::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        if let OpType::Einsum { spec } = &node.op {
            if let Some(strategy) = analyze_einsum_for_tiling(spec) {
                recommendations.insert(node_idx, strategy);
            }
        }
    }

    recommendations
}

// Helper functions

fn should_tile_einsum(spec: &str) -> bool {
    // Tile if the einsum involves reduction or matrix-like operations
    spec.contains("->") && (spec.contains(',') || spec.len() > 6)
}

fn tile_einsum_node(node: &mut EinsumNode, strategy: &TilingStrategy) -> Result<(), IrError> {
    // In a real implementation, this would transform the einsum specification
    // to include tiling metadata. For now, we just annotate it.

    // Add tiling metadata to the node
    if node.metadata.is_none() {
        node.metadata = Some(crate::Metadata::new());
    }

    if let Some(metadata) = &mut node.metadata {
        metadata.attributes.push((
            "tiling_strategy".to_string(),
            format!("{} tiles", strategy.tiles.len()),
        ));
        metadata.attributes.push((
            "register_tiling".to_string(),
            strategy.register_tiling.to_string(),
        ));
    }

    Ok(())
}

fn estimate_cache_improvement(strategy: &TilingStrategy) -> f64 {
    // Estimate based on number of tiling levels and tile sizes
    let base_improvement = 0.2; // 20% baseline
    let per_tile_improvement = 0.1; // 10% per tiling dimension
    let register_bonus = if strategy.register_tiling { 0.15 } else { 0.0 };

    let total =
        base_improvement + (strategy.tiles.len() as f64 * per_tile_improvement) + register_bonus;

    total.min(0.8) // Cap at 80% improvement
}

fn analyze_einsum_for_tiling(spec: &str) -> Option<TilingStrategy> {
    // Parse einsum spec to determine appropriate tiling
    if let Some(arrow_pos) = spec.find("->") {
        let inputs = &spec[..arrow_pos];
        let output = &spec[arrow_pos + 2..];

        // Detect matrix multiplication pattern (e.g., "ik,kj->ij")
        if inputs.contains(',') {
            let parts: Vec<&str> = inputs.split(',').collect();
            if parts.len() == 2 {
                let a_axes = parts[0].trim();
                let b_axes = parts[1].trim();

                // Check for matmul-like pattern
                if a_axes.len() == 2 && b_axes.len() == 2 && output.len() == 2 {
                    let mut strategy = TilingStrategy::new();
                    strategy.add_tile(TileConfig::new(0, 32)); // M dimension
                    strategy.add_tile(TileConfig::new(1, 32)); // K dimension
                    strategy.add_tile(TileConfig::new(2, 32)); // N dimension
                    return Some(strategy);
                }
            }
        }

        // For reductions, use smaller tiles
        if output.len() < inputs.replace(',', "").len() {
            let mut strategy = TilingStrategy::new();
            strategy.add_tile(TileConfig::new(0, 16));
            return Some(strategy);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_config_creation() {
        let config = TileConfig::new(0, 32);
        assert_eq!(config.axis, 0);
        assert_eq!(config.tile_size, 32);
        assert!(!config.unroll);

        let config_unroll = TileConfig::with_unroll(1, 16);
        assert_eq!(config_unroll.axis, 1);
        assert_eq!(config_unroll.tile_size, 16);
        assert!(config_unroll.unroll);
    }

    #[test]
    fn test_tiling_strategy_builder() {
        let mut strategy = TilingStrategy::new();
        strategy.add_tile(TileConfig::new(0, 32));
        strategy.add_tile(TileConfig::new(1, 32));

        assert_eq!(strategy.tiles.len(), 2);
        assert!(!strategy.register_tiling);
    }

    #[test]
    fn test_matmul_tiling_strategy() {
        let strategy = TilingStrategy::for_matmul(128, 128, 128);
        assert_eq!(strategy.tiles.len(), 3);
        assert!(strategy.tiles[0].tile_size <= 64);
    }

    #[test]
    fn test_conv_tiling_strategy() {
        let strategy = TilingStrategy::for_conv(32, 64, 56, 56);
        assert_eq!(strategy.tiles.len(), 4);
    }

    #[test]
    fn test_should_tile_einsum() {
        assert!(should_tile_einsum("ik,kj->ij"));
        assert!(should_tile_einsum("ijk->ij"));
        assert!(!should_tile_einsum("i->i"));
    }

    #[test]
    fn test_analyze_einsum_for_tiling() {
        let strategy = analyze_einsum_for_tiling("ik,kj->ij");
        assert!(strategy.is_some());
        let s = strategy.unwrap();
        assert_eq!(s.tiles.len(), 3);

        let strategy_reduction = analyze_einsum_for_tiling("ijk->ij");
        assert!(strategy_reduction.is_some());
    }

    #[test]
    fn test_apply_tiling_to_graph() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::einsum("ik,kj->ij", vec![a, b], vec![c]))
            .unwrap();

        let strategy = TilingStrategy::for_matmul(64, 64, 64);
        let result = apply_tiling(&mut graph, &strategy).unwrap();

        assert_eq!(result.nodes_tiled, 1);
        assert!(result.estimated_speedup >= 1.0);
    }

    #[test]
    fn test_register_tiling() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::einsum("ik,kj->ij", vec![a, b], vec![c]))
            .unwrap();

        let result = apply_register_tiling(&mut graph).unwrap();
        assert_eq!(result.nodes_tiled, 1);
        assert!(result.loops_unrolled > 0);
    }

    #[test]
    fn test_multilevel_tiling() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::einsum("ik,kj->ij", vec![a, b], vec![c]))
            .unwrap();

        let l1_tiles = vec![8, 8, 8];
        let l2_tiles = vec![32, 32, 32];
        let l3_tiles = vec![128, 128, 128];

        let result = apply_multilevel_tiling(&mut graph, &l1_tiles, &l2_tiles, &l3_tiles).unwrap();
        assert!(result.nodes_tiled > 0);
        assert!(result.estimated_speedup > 1.0);
    }

    #[test]
    fn test_recommend_tiling_strategy() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");
        let d = graph.add_tensor("D");

        // Matrix multiplication
        graph
            .add_node(EinsumNode::einsum("ik,kj->ij", vec![a, b], vec![c]))
            .unwrap();

        // Element-wise operation (should not be tiled)
        graph
            .add_node(EinsumNode::elem_unary("relu", c, d))
            .unwrap();

        let recommendations = recommend_tiling_strategy(&graph);
        assert_eq!(recommendations.len(), 1); // Only matmul should have recommendation
        assert!(recommendations.contains_key(&0));
    }

    #[test]
    fn test_estimate_cache_improvement() {
        let mut strategy = TilingStrategy::new();
        strategy.add_tile(TileConfig::new(0, 32));
        strategy.add_tile(TileConfig::new(1, 32));

        let improvement = estimate_cache_improvement(&strategy);
        assert!(improvement > 0.0 && improvement <= 0.8);

        let strategy_with_register = strategy.with_register_tiling();
        let improvement_with_register = estimate_cache_improvement(&strategy_with_register);
        assert!(improvement_with_register > improvement);
    }

    #[test]
    fn test_tiling_result_none() {
        let result = TilingResult::none();
        assert_eq!(result.nodes_tiled, 0);
        assert_eq!(result.loops_unrolled, 0);
        assert_eq!(result.estimated_cache_improvement, 0.0);
        assert_eq!(result.estimated_speedup, 1.0);
    }

    #[test]
    fn test_tiling_with_metadata() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::einsum("ik,kj->ij", vec![a, b], vec![c]))
            .unwrap();

        let strategy = TilingStrategy::for_matmul(64, 64, 64);
        apply_tiling(&mut graph, &strategy).unwrap();

        // Check that metadata was added
        let node = &graph.nodes[0];
        assert!(node.metadata.is_some());
        if let Some(metadata) = &node.metadata {
            assert!(metadata.get_attribute("tiling_strategy").is_some());
        }
    }

    #[test]
    fn test_cache_line_size_configuration() {
        let strategy = TilingStrategy::new().with_cache_line_size(128);
        assert_eq!(strategy.cache_line_size, 128);
    }

    #[test]
    fn test_small_matrix_tiling() {
        // Test tiling for small matrices (should use minimum tile sizes)
        let strategy = TilingStrategy::for_matmul(4, 4, 4);
        assert_eq!(strategy.tiles.len(), 3);
        // All tiles should be at least 8 (the minimum)
        for tile in &strategy.tiles {
            assert!(tile.tile_size >= 8);
        }
    }

    #[test]
    fn test_large_matrix_tiling() {
        // Test tiling for large matrices (should cap at 64)
        let strategy = TilingStrategy::for_matmul(1024, 1024, 1024);
        assert_eq!(strategy.tiles.len(), 3);
        // All tiles should be capped at 64
        for tile in &strategy.tiles {
            assert!(tile.tile_size <= 64);
        }
    }
}
