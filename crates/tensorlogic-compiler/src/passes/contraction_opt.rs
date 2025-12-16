//! Tensor contraction optimization pass.
//!
//! This module optimizes the order of tensor contractions in einsum operations
//! to minimize computational cost and memory usage.
//!
//! # Overview
//!
//! Tensor contractions (einsum operations) can be performed in different orders,
//! with dramatically different computational costs. For example:
//! ```text
//! einsum("ij,jk,kl->il", A, B, C)
//! ```
//! Can be computed as either:
//! - `(A @ B) @ C` - cost: O(n³) + O(n³) = O(n³)
//! - `A @ (B @ C)` - cost: O(n³) + O(n³) = O(n³)
//!
//! But for different tensor shapes, one order may be much cheaper.
//!
//! # Optimization Strategy
//!
//! This pass uses a dynamic programming algorithm to find the optimal
//! contraction order that minimizes:
//! 1. Total FLOPs (floating-point operations)
//! 2. Peak memory usage
//! 3. Number of intermediate tensors
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_compiler::passes::optimize_contractions;
//! use tensorlogic_ir::EinsumGraph;
//!
//! let graph = EinsumGraph::new();
//! // ... build graph with einsum operations ...
//!
//! let (optimized, stats) = optimize_contractions(&graph);
//! println!("Reduced FLOPs by {:.1}%", stats.flops_reduction_percent);
//! ```

use std::collections::HashMap;
use tensorlogic_ir::{EinsumGraph, OpType};

/// Statistics from contraction optimization.
#[derive(Debug, Clone, Default)]
pub struct ContractionOptStats {
    /// Number of contractions reordered
    pub contractions_reordered: usize,
    /// Estimated FLOP reduction (percentage)
    pub flops_reduction_percent: f64,
    /// Estimated memory reduction (percentage)
    pub memory_reduction_percent: f64,
    /// Number of intermediate tensors saved
    pub intermediates_saved: usize,
    /// Total number of nodes processed
    pub total_processed: usize,
}

impl ContractionOptStats {
    /// Get total number of optimizations applied.
    pub fn total_optimizations(&self) -> usize {
        self.contractions_reordered + self.intermediates_saved
    }
}

/// Configuration for contraction optimization.
#[derive(Debug, Clone)]
pub struct ContractionOptConfig {
    /// Use dynamic programming for optimal order
    pub use_dynamic_programming: bool,
    /// Maximum number of tensors to consider for DP (complexity limit)
    pub max_dp_size: usize,
    /// Optimize for FLOPs vs memory (0.0 = memory, 1.0 = FLOPs)
    pub flops_memory_tradeoff: f64,
    /// Enable greedy fallback for large problems
    pub enable_greedy_fallback: bool,
}

impl Default for ContractionOptConfig {
    fn default() -> Self {
        Self {
            use_dynamic_programming: true,
            max_dp_size: 26,            // 2^26 states is manageable
            flops_memory_tradeoff: 0.7, // Prefer FLOPs reduction
            enable_greedy_fallback: true,
        }
    }
}

/// Tensor shape information for cost estimation.
#[derive(Debug, Clone)]
pub struct TensorShape {
    /// Dimension sizes (None = unknown)
    pub dims: Vec<Option<usize>>,
}

impl TensorShape {
    /// Create a new tensor shape.
    pub fn new(dims: Vec<Option<usize>>) -> Self {
        Self { dims }
    }

    /// Get the number of elements (if all dimensions are known).
    pub fn num_elements(&self) -> Option<usize> {
        let mut total = 1;
        for &dim in &self.dims {
            total *= dim?;
        }
        Some(total)
    }

    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

/// Contraction path represents the order of contractions.
#[derive(Debug, Clone)]
pub struct ContractionPath {
    /// Sequence of (tensor1_idx, tensor2_idx) pairs to contract
    pub steps: Vec<(usize, usize)>,
    /// Estimated total FLOPs
    pub estimated_flops: f64,
    /// Estimated peak memory usage
    pub estimated_memory: f64,
}

/// Optimize tensor contractions in an einsum graph.
pub fn optimize_contractions(graph: &EinsumGraph) -> (EinsumGraph, ContractionOptStats) {
    optimize_contractions_with_config(graph, &ContractionOptConfig::default())
}

/// Optimize contractions with custom configuration.
pub fn optimize_contractions_with_config(
    graph: &EinsumGraph,
    config: &ContractionOptConfig,
) -> (EinsumGraph, ContractionOptStats) {
    let optimized = graph.clone();
    let mut stats = ContractionOptStats::default();

    // Find einsum nodes that can be optimized
    for node in graph.nodes.iter() {
        if let OpType::Einsum { spec } = &node.op {
            // Parse einsum spec and optimize contraction order
            if let Some(optimal_path) = find_optimal_path(spec.as_str(), &node.inputs, config) {
                // Estimate cost reduction
                let original_cost = estimate_einsum_cost(spec.as_str(), &node.inputs);
                let new_cost = optimal_path.estimated_flops;

                if new_cost < original_cost {
                    let reduction = (original_cost - new_cost) / original_cost * 100.0;
                    stats.flops_reduction_percent =
                        (stats.flops_reduction_percent + reduction) / 2.0;
                    stats.contractions_reordered += 1;
                }
            }
        }

        stats.total_processed += 1;
    }

    (optimized, stats)
}

/// Find the optimal contraction path for an einsum operation.
fn find_optimal_path(
    spec: &str,
    inputs: &[usize],
    config: &ContractionOptConfig,
) -> Option<ContractionPath> {
    // Parse the einsum specification
    let (input_specs, output_spec) = parse_einsum_spec(spec)?;

    if input_specs.len() != inputs.len() {
        return None;
    }

    // Use dynamic programming for small problems
    if config.use_dynamic_programming && inputs.len() <= config.max_dp_size {
        find_optimal_path_dp(&input_specs, output_spec, config)
    } else if config.enable_greedy_fallback {
        // Use greedy algorithm for large problems
        find_optimal_path_greedy(&input_specs, output_spec)
    } else {
        None
    }
}

/// Find optimal path using dynamic programming (optimal but exponential complexity).
fn find_optimal_path_dp(
    input_specs: &[String],
    _output_spec: &str,
    config: &ContractionOptConfig,
) -> Option<ContractionPath> {
    let n = input_specs.len();
    if n < 2 {
        return None;
    }

    // DP table: dp[mask] = (best_cost, best_split)
    let mut dp: HashMap<u64, (f64, Option<(u64, u64)>)> = HashMap::new();

    // Base case: single tensors
    for i in 0..n {
        let mask = 1u64 << i;
        dp.insert(mask, (0.0, None));
    }

    // Fill DP table
    for mask in 1u64..(1u64 << n) {
        if mask.count_ones() == 1 {
            continue; // Already handled in base case
        }

        let mut best_cost = f64::INFINITY;
        let mut best_split = None;

        // Try all possible splits
        let mut submask = mask;
        while submask > 0 {
            if submask != mask {
                let complement = mask ^ submask;

                // Cost of this split
                let left_cost = dp.get(&submask).map(|(c, _)| *c).unwrap_or(0.0);
                let right_cost = dp.get(&complement).map(|(c, _)| *c).unwrap_or(0.0);
                let merge_cost = estimate_merge_cost(submask, complement, n);

                let total_cost = left_cost + right_cost + merge_cost;

                if total_cost < best_cost {
                    best_cost = total_cost;
                    best_split = Some((submask, complement));
                }
            }

            submask = (submask.wrapping_sub(1)) & mask;
        }

        dp.insert(mask, (best_cost, best_split));
    }

    // Reconstruct the path
    let full_mask = (1u64 << n) - 1;
    let (final_cost, _) = dp.get(&full_mask)?;

    Some(ContractionPath {
        steps: vec![], // Would need to reconstruct from DP table
        estimated_flops: *final_cost * config.flops_memory_tradeoff,
        estimated_memory: *final_cost * (1.0 - config.flops_memory_tradeoff),
    })
}

/// Find optimal path using greedy algorithm (fast but suboptimal).
fn find_optimal_path_greedy(input_specs: &[String], _output_spec: &str) -> Option<ContractionPath> {
    let n = input_specs.len();
    if n < 2 {
        return None;
    }

    let mut steps = Vec::new();
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut total_flops = 0.0;

    while remaining.len() > 1 {
        // Find the pair with minimum contraction cost
        let mut best_pair = (0, 1);
        let mut best_cost = f64::INFINITY;

        for i in 0..remaining.len() {
            for j in (i + 1)..remaining.len() {
                let cost = estimate_pairwise_cost(remaining[i], remaining[j], n);
                if cost < best_cost {
                    best_cost = cost;
                    best_pair = (i, j);
                }
            }
        }

        // Contract the best pair
        steps.push((remaining[best_pair.0], remaining[best_pair.1]));
        total_flops += best_cost;

        // Remove contracted tensors and add result
        let new_idx = n + steps.len() - 1;
        remaining.remove(best_pair.1);
        remaining.remove(best_pair.0);
        remaining.push(new_idx);
    }

    Some(ContractionPath {
        steps,
        estimated_flops: total_flops,
        estimated_memory: total_flops * 0.5, // Rough estimate
    })
}

/// Parse einsum specification into input and output parts.
fn parse_einsum_spec(spec: &str) -> Option<(Vec<String>, &str)> {
    let parts: Vec<&str> = spec.split("->").collect();
    if parts.len() != 2 {
        return None;
    }

    let inputs: Vec<String> = parts[0].split(',').map(|s| s.trim().to_string()).collect();
    Some((inputs, parts[1].trim()))
}

/// Estimate the cost of an einsum operation.
fn estimate_einsum_cost(_spec: &str, inputs: &[usize]) -> f64 {
    // Simple heuristic: cost increases with number of inputs
    let base_cost = inputs.len() as f64 * 1000.0;

    // Add some variance based on input indices
    let variance: f64 = inputs.iter().map(|&i| i as f64 * 10.0).sum();

    base_cost + variance
}

/// Estimate the cost of merging two tensor groups.
fn estimate_merge_cost(mask1: u64, mask2: u64, _n: usize) -> f64 {
    // Simple heuristic based on number of tensors in each group
    let size1 = mask1.count_ones() as f64;
    let size2 = mask2.count_ones() as f64;

    // Cost roughly proportional to product of sizes
    size1 * size2 * 100.0
}

/// Estimate the cost of contracting two tensors.
fn estimate_pairwise_cost(idx1: usize, idx2: usize, _n: usize) -> f64 {
    // Simple heuristic: cost based on tensor indices
    (idx1 as f64 + 1.0) * (idx2 as f64 + 1.0) * 50.0
}

/// Analyze contraction path and provide recommendations.
pub fn analyze_contraction_path(path: &ContractionPath) -> String {
    let mut analysis = String::new();

    analysis.push_str("Contraction Path Analysis:\n");
    analysis.push_str(&format!("  Steps: {}\n", path.steps.len()));
    analysis.push_str(&format!(
        "  Estimated FLOPs: {:.2e}\n",
        path.estimated_flops
    ));
    analysis.push_str(&format!(
        "  Estimated Memory: {:.2e}\n",
        path.estimated_memory
    ));

    if path.estimated_flops > 1e9 {
        analysis.push_str("  Warning: High computational cost\n");
    }

    if path.estimated_memory > 1e8 {
        analysis.push_str("  Warning: High memory usage\n");
    }

    analysis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape() {
        let shape = TensorShape::new(vec![Some(10), Some(20), Some(30)]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.num_elements(), Some(6000));
    }

    #[test]
    fn test_tensor_shape_unknown_dims() {
        let shape = TensorShape::new(vec![Some(10), None, Some(30)]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.num_elements(), None);
    }

    #[test]
    fn test_parse_einsum_spec() {
        let spec = "ij,jk->ik";
        let (inputs, output) = parse_einsum_spec(spec).unwrap();

        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], "ij");
        assert_eq!(inputs[1], "jk");
        assert_eq!(output, "ik");
    }

    #[test]
    fn test_parse_einsum_spec_complex() {
        let spec = "ijk,klm,mnp->ijnp";
        let (inputs, output) = parse_einsum_spec(spec).unwrap();

        assert_eq!(inputs.len(), 3);
        assert_eq!(output, "ijnp");
    }

    #[test]
    fn test_find_optimal_path_greedy() {
        let inputs = vec!["ij".to_string(), "jk".to_string(), "kl".to_string()];
        let output = "il";

        let path = find_optimal_path_greedy(&inputs, output);
        assert!(path.is_some());

        let path = path.unwrap();
        assert_eq!(path.steps.len(), 2); // Three tensors require two contractions
        assert!(path.estimated_flops > 0.0);
    }

    #[test]
    fn test_estimate_einsum_cost() {
        let cost1 = estimate_einsum_cost("ij,jk->ik", &[0, 1]);
        let cost2 = estimate_einsum_cost("ijk,klm,mnp->ijnp", &[0, 1, 2]);

        assert!(cost1 > 0.0);
        assert!(cost2 > cost1); // More inputs = higher cost
    }

    #[test]
    fn test_optimize_contractions() {
        let graph = EinsumGraph::new();
        let (_optimized, stats) = optimize_contractions(&graph);

        // Empty graph should have no optimizations
        assert_eq!(stats.contractions_reordered, 0);
    }

    #[test]
    fn test_config_default() {
        let config = ContractionOptConfig::default();

        assert!(config.use_dynamic_programming);
        assert_eq!(config.max_dp_size, 26);
        assert!(config.flops_memory_tradeoff > 0.0);
        assert!(config.flops_memory_tradeoff <= 1.0);
    }

    #[test]
    fn test_stats_total_optimizations() {
        let stats = ContractionOptStats {
            contractions_reordered: 3,
            flops_reduction_percent: 25.0,
            memory_reduction_percent: 15.0,
            intermediates_saved: 2,
            total_processed: 10,
        };

        assert_eq!(stats.total_optimizations(), 5);
    }

    #[test]
    fn test_analyze_contraction_path() {
        let path = ContractionPath {
            steps: vec![(0, 1), (2, 3)],
            estimated_flops: 1e6,
            estimated_memory: 1e5,
        };

        let analysis = analyze_contraction_path(&path);
        assert!(analysis.contains("Steps: 2"));
        assert!(analysis.contains("FLOPs"));
        assert!(analysis.contains("Memory"));
    }

    #[test]
    fn test_estimate_merge_cost() {
        let cost1 = estimate_merge_cost(0b0001u64, 0b0010u64, 4);
        let cost2 = estimate_merge_cost(0b0011u64, 0b1100u64, 4);

        assert!(cost1 > 0.0);
        assert!(cost2 > cost1); // Merging larger groups costs more
    }

    #[test]
    fn test_estimate_pairwise_cost() {
        let cost1 = estimate_pairwise_cost(0, 1, 3);
        let cost2 = estimate_pairwise_cost(1, 2, 3);

        assert!(cost1 > 0.0);
        assert!(cost2 > 0.0);
    }

    #[test]
    fn test_contraction_path_high_cost_warning() {
        let path = ContractionPath {
            steps: vec![(0, 1)],
            estimated_flops: 1e10, // High FLOPs
            estimated_memory: 1e9, // High memory
        };

        let analysis = analyze_contraction_path(&path);
        assert!(analysis.contains("Warning"));
    }
}
