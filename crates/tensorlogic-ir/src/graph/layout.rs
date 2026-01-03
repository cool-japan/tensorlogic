//! Tensor layout and stride optimization.
//!
//! This module provides optimizations for tensor memory layouts to improve
//! cache utilization and memory access patterns.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{EinsumGraph, IrError};

/// Memory layout strategy for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum LayoutStrategy {
    /// Row-major order (C-style, default for most systems)
    #[default]
    RowMajor,
    /// Column-major order (Fortran-style, good for column operations)
    ColumnMajor,
    /// Blocked layout for cache-friendly access
    Blocked { block_size: usize },
    /// Tiled layout with specific tile dimensions
    Tiled {
        tile_height: usize,
        tile_width: usize,
    },
    /// Z-order (Morton) curve for locality preservation
    ZOrder,
    /// Hilbert curve for even better locality
    Hilbert,
}

impl LayoutStrategy {
    /// Get the recommended strategy for a given operation pattern.
    pub fn for_operation(op: &str) -> Self {
        match op {
            "matmul" | "einsum" => Self::Blocked { block_size: 32 },
            "transpose" => Self::ColumnMajor,
            "conv2d" => Self::Tiled {
                tile_height: 8,
                tile_width: 8,
            },
            "scan" | "reduce" => Self::RowMajor,
            _ => Self::default(),
        }
    }

    /// Check if this layout benefits from vectorization.
    pub fn supports_vectorization(&self) -> bool {
        matches!(
            self,
            Self::RowMajor | Self::Blocked { .. } | Self::Tiled { .. }
        )
    }

    /// Check if this layout preserves spatial locality.
    pub fn preserves_locality(&self) -> bool {
        matches!(
            self,
            Self::Blocked { .. } | Self::Tiled { .. } | Self::ZOrder | Self::Hilbert
        )
    }
}

/// Stride pattern for a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StridePattern {
    /// Strides for each dimension (in elements)
    pub strides: Vec<usize>,
    /// Whether strides are contiguous
    pub is_contiguous: bool,
    /// Alignment in bytes (0 means no specific alignment)
    pub alignment: usize,
}

impl StridePattern {
    /// Create a row-major stride pattern for given dimensions.
    pub fn row_major(dims: &[usize]) -> Self {
        let mut strides = vec![1];
        for i in (0..dims.len() - 1).rev() {
            strides.insert(0, strides[0] * dims[i + 1]);
        }

        Self {
            strides,
            is_contiguous: true,
            alignment: 0,
        }
    }

    /// Create a column-major stride pattern for given dimensions.
    pub fn column_major(dims: &[usize]) -> Self {
        let mut strides = vec![1];
        for i in 0..dims.len() - 1 {
            strides.push(strides[i] * dims[i]);
        }

        Self {
            strides,
            is_contiguous: true,
            alignment: 0,
        }
    }

    /// Create a custom stride pattern.
    pub fn custom(strides: Vec<usize>) -> Self {
        let is_contiguous = is_contiguous_strides(&strides);
        Self {
            strides,
            is_contiguous,
            alignment: 0,
        }
    }

    /// Set the alignment requirement.
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    /// Check if the stride pattern allows efficient vectorization.
    pub fn is_vectorizable(&self) -> bool {
        self.is_contiguous && self.strides.last().copied().unwrap_or(0) == 1
    }

    /// Estimate memory access cost (lower is better).
    pub fn access_cost(&self) -> f64 {
        if self.is_contiguous {
            1.0
        } else {
            // Non-contiguous access is more expensive
            1.5 + (self.strides.len() as f64 * 0.1)
        }
    }
}

/// Layout configuration for a tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorLayout {
    /// Tensor index
    pub tensor_idx: usize,
    /// Layout strategy
    pub strategy: LayoutStrategy,
    /// Stride pattern
    pub strides: StridePattern,
    /// Whether this layout can be transformed
    pub is_mutable: bool,
}

impl TensorLayout {
    /// Create a new tensor layout.
    pub fn new(tensor_idx: usize, strategy: LayoutStrategy, dims: &[usize]) -> Self {
        let strides = match strategy {
            LayoutStrategy::RowMajor => StridePattern::row_major(dims),
            LayoutStrategy::ColumnMajor => StridePattern::column_major(dims),
            _ => StridePattern::row_major(dims), // Default to row-major
        };

        Self {
            tensor_idx,
            strategy,
            strides,
            is_mutable: true,
        }
    }

    /// Estimate the memory access efficiency (0.0 to 1.0, higher is better).
    pub fn access_efficiency(&self) -> f64 {
        let base_efficiency = if self.strides.is_contiguous { 0.9 } else { 0.5 };

        let locality_bonus: f64 = if self.strategy.preserves_locality() {
            0.1
        } else {
            0.0
        };

        (base_efficiency + locality_bonus).min(1.0f64)
    }
}

/// Result of layout optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutOptimizationResult {
    /// Optimized layouts for each tensor
    pub layouts: HashMap<usize, TensorLayout>,
    /// Number of layout transformations required
    pub transformations_needed: usize,
    /// Estimated memory access improvement (0.0 to 1.0)
    pub estimated_improvement: f64,
    /// Estimated speedup from better layouts
    pub estimated_speedup: f64,
}

impl LayoutOptimizationResult {
    /// Create a result with no optimizations.
    pub fn none() -> Self {
        Self {
            layouts: HashMap::new(),
            transformations_needed: 0,
            estimated_improvement: 0.0,
            estimated_speedup: 1.0,
        }
    }

    /// Get the layout for a tensor.
    pub fn get_layout(&self, tensor_idx: usize) -> Option<&TensorLayout> {
        self.layouts.get(&tensor_idx)
    }
}

/// Optimize tensor layouts for a graph.
pub fn optimize_layouts(graph: &EinsumGraph) -> Result<LayoutOptimizationResult, IrError> {
    let mut result = LayoutOptimizationResult::none();

    // Analyze each tensor and choose optimal layout
    for (tensor_idx, tensor_name) in graph.tensors.iter().enumerate() {
        // Infer dimensions from tensor name or metadata
        let dims = infer_dimensions(tensor_name, graph, tensor_idx);

        // Analyze usage pattern to determine best layout
        let strategy = analyze_usage_pattern(graph, tensor_idx);

        let layout = TensorLayout::new(tensor_idx, strategy, &dims);
        result.layouts.insert(tensor_idx, layout);
    }

    // Count needed transformations
    result.transformations_needed = count_layout_conversions(&result.layouts);

    // Estimate improvements
    let avg_efficiency: f64 = result
        .layouts
        .values()
        .map(|l| l.access_efficiency())
        .sum::<f64>()
        / result.layouts.len().max(1) as f64;

    result.estimated_improvement = (avg_efficiency - 0.7).max(0.0);
    result.estimated_speedup = 1.0 + result.estimated_improvement * 0.3;

    Ok(result)
}

/// Apply the recommended layouts to a graph.
pub fn apply_layouts(
    graph: &mut EinsumGraph,
    layouts: &HashMap<usize, TensorLayout>,
) -> Result<(), IrError> {
    // Add layout metadata to tensors
    for (tensor_idx, layout) in layouts {
        if *tensor_idx < graph.tensors.len() {
            let mut metadata = graph
                .get_tensor_metadata(*tensor_idx)
                .cloned()
                .unwrap_or_else(crate::Metadata::new);

            metadata
                .attributes
                .push(("layout".to_string(), format!("{:?}", layout.strategy)));
            metadata.attributes.push((
                "is_contiguous".to_string(),
                layout.strides.is_contiguous.to_string(),
            ));

            graph.add_tensor_metadata(*tensor_idx, metadata);
        }
    }

    Ok(())
}

/// Find opportunities for layout fusion (avoiding layout conversions).
pub fn find_layout_fusion_opportunities(
    layouts: &HashMap<usize, TensorLayout>,
) -> Vec<(usize, usize)> {
    let mut opportunities = Vec::new();

    // Find pairs of tensors that would benefit from the same layout
    let tensor_indices: Vec<_> = layouts.keys().copied().collect();

    for i in 0..tensor_indices.len() {
        for j in (i + 1)..tensor_indices.len() {
            let idx1 = tensor_indices[i];
            let idx2 = tensor_indices[j];

            if let (Some(layout1), Some(layout2)) = (layouts.get(&idx1), layouts.get(&idx2)) {
                if layout1.strategy != layout2.strategy && layout1.is_mutable && layout2.is_mutable
                {
                    opportunities.push((idx1, idx2));
                }
            }
        }
    }

    opportunities
}

// Helper functions

fn infer_dimensions(_tensor_name: &str, _graph: &EinsumGraph, _tensor_idx: usize) -> Vec<usize> {
    // Try to infer dimensions from tensor name
    // For now, return a default 2D shape
    // In a real implementation, this would use shape inference
    vec![64, 64]
}

fn analyze_usage_pattern(graph: &EinsumGraph, tensor_idx: usize) -> LayoutStrategy {
    // Count how tensor is used
    let mut read_patterns = Vec::new();

    for node in &graph.nodes {
        if node.inputs.contains(&tensor_idx) {
            // Analyze how it's accessed
            let pattern = match &node.op {
                crate::OpType::Einsum { spec } => analyze_einsum_pattern(spec),
                crate::OpType::Reduce { .. } => "reduce",
                crate::OpType::ElemUnary { .. } => "scan",
                crate::OpType::ElemBinary { .. } => "scan",
            };
            read_patterns.push(pattern);
        }
    }

    // Choose best layout based on dominant pattern
    if read_patterns.contains(&"matmul") {
        LayoutStrategy::Blocked { block_size: 32 }
    } else if read_patterns.contains(&"transpose") {
        LayoutStrategy::ColumnMajor
    } else if read_patterns.contains(&"conv") {
        LayoutStrategy::Tiled {
            tile_height: 8,
            tile_width: 8,
        }
    } else {
        LayoutStrategy::RowMajor
    }
}

fn analyze_einsum_pattern(spec: &str) -> &'static str {
    if spec.contains(',') {
        "matmul"
    } else if spec.contains("->") {
        let parts: Vec<&str> = spec.split("->").collect();
        if parts.len() == 2 && parts[0].len() > parts[1].len() {
            "reduce"
        } else {
            "scan"
        }
    } else {
        "scan"
    }
}

fn count_layout_conversions(layouts: &HashMap<usize, TensorLayout>) -> usize {
    // Count tensors that need non-default layout
    layouts
        .values()
        .filter(|l| l.strategy != LayoutStrategy::RowMajor)
        .count()
}

fn is_contiguous_strides(strides: &[usize]) -> bool {
    if strides.is_empty() {
        return true;
    }

    // Check if strides form a contiguous pattern
    let mut prev = strides[strides.len() - 1];
    if prev != 1 {
        return false;
    }

    for &stride in strides.iter().rev().skip(1) {
        if stride <= prev {
            return false;
        }
        // Check if the ratio is reasonable (not a huge gap)
        // For contiguous arrays, dimension sizes are typically < 10000
        let ratio = stride / prev;
        if ratio == 0 || ratio > 10000 {
            return false;
        }
        // Also check that stride is exactly divisible by prev
        if stride % prev != 0 {
            return false;
        }
        prev = stride;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_strategy_default() {
        assert_eq!(LayoutStrategy::default(), LayoutStrategy::RowMajor);
    }

    #[test]
    fn test_layout_strategy_for_operation() {
        let matmul_layout = LayoutStrategy::for_operation("matmul");
        assert!(matches!(matmul_layout, LayoutStrategy::Blocked { .. }));

        let transpose_layout = LayoutStrategy::for_operation("transpose");
        assert_eq!(transpose_layout, LayoutStrategy::ColumnMajor);

        let conv_layout = LayoutStrategy::for_operation("conv2d");
        assert!(matches!(conv_layout, LayoutStrategy::Tiled { .. }));
    }

    #[test]
    fn test_layout_strategy_vectorization() {
        assert!(LayoutStrategy::RowMajor.supports_vectorization());
        assert!(LayoutStrategy::Blocked { block_size: 32 }.supports_vectorization());
        assert!(!LayoutStrategy::ZOrder.supports_vectorization());
    }

    #[test]
    fn test_layout_strategy_locality() {
        assert!(LayoutStrategy::Blocked { block_size: 32 }.preserves_locality());
        assert!(LayoutStrategy::ZOrder.preserves_locality());
        assert!(LayoutStrategy::Hilbert.preserves_locality());
        assert!(!LayoutStrategy::RowMajor.preserves_locality());
    }

    #[test]
    fn test_stride_pattern_row_major() {
        let dims = vec![4, 8, 16];
        let pattern = StridePattern::row_major(&dims);

        assert_eq!(pattern.strides, vec![128, 16, 1]);
        assert!(pattern.is_contiguous);
        assert!(pattern.is_vectorizable());
    }

    #[test]
    fn test_stride_pattern_column_major() {
        let dims = vec![4, 8, 16];
        let pattern = StridePattern::column_major(&dims);

        assert_eq!(pattern.strides, vec![1, 4, 32]);
        assert!(pattern.is_contiguous);
    }

    #[test]
    fn test_stride_pattern_custom() {
        let strides = vec![64, 8, 1];
        let pattern = StridePattern::custom(strides.clone());

        assert_eq!(pattern.strides, strides);
        assert!(pattern.is_contiguous);
    }

    #[test]
    fn test_stride_pattern_non_contiguous() {
        let strides = vec![100, 10, 2]; // Non-contiguous
        let pattern = StridePattern::custom(strides);

        assert!(!pattern.is_contiguous);
        assert!(!pattern.is_vectorizable());
    }

    #[test]
    fn test_stride_pattern_with_alignment() {
        let pattern = StridePattern::row_major(&[4, 8]).with_alignment(64);
        assert_eq!(pattern.alignment, 64);
    }

    #[test]
    fn test_stride_pattern_access_cost() {
        let contiguous = StridePattern::row_major(&[4, 8]);
        let non_contiguous = StridePattern::custom(vec![100, 10, 2]);

        assert!(contiguous.access_cost() < non_contiguous.access_cost());
    }

    #[test]
    fn test_tensor_layout_creation() {
        let layout = TensorLayout::new(0, LayoutStrategy::RowMajor, &[4, 8]);

        assert_eq!(layout.tensor_idx, 0);
        assert_eq!(layout.strategy, LayoutStrategy::RowMajor);
        assert!(layout.is_mutable);
        assert!(layout.strides.is_contiguous);
    }

    #[test]
    fn test_tensor_layout_access_efficiency() {
        let row_major = TensorLayout::new(0, LayoutStrategy::RowMajor, &[4, 8]);
        let blocked = TensorLayout::new(0, LayoutStrategy::Blocked { block_size: 32 }, &[4, 8]);

        let row_efficiency = row_major.access_efficiency();
        let blocked_efficiency = blocked.access_efficiency();

        assert!(row_efficiency > 0.0 && row_efficiency <= 1.0);
        assert!(blocked_efficiency > row_efficiency); // Blocked should be more efficient
    }

    #[test]
    fn test_layout_optimization_result_none() {
        let result = LayoutOptimizationResult::none();
        assert!(result.layouts.is_empty());
        assert_eq!(result.transformations_needed, 0);
        assert_eq!(result.estimated_improvement, 0.0);
        assert_eq!(result.estimated_speedup, 1.0);
    }

    #[test]
    fn test_optimize_layouts_empty_graph() {
        let graph = EinsumGraph::new();
        let result = optimize_layouts(&graph).unwrap();
        assert!(result.layouts.is_empty());
    }

    #[test]
    fn test_optimize_layouts_simple_graph() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph
            .add_node(crate::EinsumNode::einsum("ik,kj->ij", vec![a, b], vec![c]))
            .unwrap();

        let result = optimize_layouts(&graph).unwrap();
        assert_eq!(result.layouts.len(), 3);
        assert!(result.estimated_speedup >= 1.0);
    }

    #[test]
    fn test_apply_layouts() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");

        let mut layouts = HashMap::new();
        layouts.insert(
            a,
            TensorLayout::new(a, LayoutStrategy::Blocked { block_size: 32 }, &[64, 64]),
        );

        apply_layouts(&mut graph, &layouts).unwrap();

        // Check that metadata was added
        let metadata = graph.get_tensor_metadata(a);
        assert!(metadata.is_some());
    }

    #[test]
    fn test_find_layout_fusion_opportunities() {
        let mut layouts = HashMap::new();

        layouts.insert(0, TensorLayout::new(0, LayoutStrategy::RowMajor, &[4, 8]));
        layouts.insert(
            1,
            TensorLayout::new(1, LayoutStrategy::ColumnMajor, &[4, 8]),
        );
        layouts.insert(2, TensorLayout::new(2, LayoutStrategy::RowMajor, &[4, 8]));

        let opportunities = find_layout_fusion_opportunities(&layouts);
        assert!(!opportunities.is_empty());
    }

    #[test]
    fn test_analyze_einsum_pattern() {
        assert_eq!(analyze_einsum_pattern("ik,kj->ij"), "matmul");
        assert_eq!(analyze_einsum_pattern("ijk->ij"), "reduce");
        assert_eq!(analyze_einsum_pattern("ij->ij"), "scan");
    }

    #[test]
    fn test_is_contiguous_strides() {
        assert!(is_contiguous_strides(&[8, 4, 1]));
        assert!(is_contiguous_strides(&[1]));
        assert!(is_contiguous_strides(&[]));
        assert!(is_contiguous_strides(&[8, 2, 1])); // Valid: dims [?, 4, 2]
        assert!(!is_contiguous_strides(&[8, 4, 2])); // Doesn't end with 1
        assert!(!is_contiguous_strides(&[9, 2, 1])); // Not divisible: 9 % 2 != 0
    }

    #[test]
    fn test_count_layout_conversions() {
        let mut layouts = HashMap::new();

        layouts.insert(0, TensorLayout::new(0, LayoutStrategy::RowMajor, &[4, 8]));
        layouts.insert(
            1,
            TensorLayout::new(1, LayoutStrategy::ColumnMajor, &[4, 8]),
        );
        layouts.insert(
            2,
            TensorLayout::new(2, LayoutStrategy::Blocked { block_size: 32 }, &[4, 8]),
        );

        let conversions = count_layout_conversions(&layouts);
        assert_eq!(conversions, 2); // column-major and blocked need conversion
    }

    #[test]
    fn test_layout_optimization_with_metadata() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");

        // Add metadata to suggest layout
        let metadata = crate::Metadata::new().with_attribute("preferred_layout", "blocked");
        graph.add_tensor_metadata(a, metadata);

        let result = optimize_layouts(&graph).unwrap();
        assert!(result.get_layout(a).is_some());
        assert!(result.get_layout(b).is_some());
    }
}
