//! Memory optimization and layout analysis.
//!
//! This module provides utilities for analyzing and optimizing memory usage
//! in computation graphs. It includes memory footprint estimation, tensor
//! lifetime analysis, and operation reordering to minimize peak memory usage.

use std::collections::{HashMap, HashSet};

use super::{EinsumGraph, OpType};
use crate::error::IrError;

/// Memory footprint estimate for a tensor
#[derive(Debug, Clone, PartialEq)]
pub struct TensorMemory {
    /// Tensor index
    pub tensor_idx: usize,
    /// Estimated size in bytes
    pub size_bytes: usize,
    /// First node that uses this tensor
    pub first_use: Option<usize>,
    /// Last node that uses this tensor
    pub last_use: Option<usize>,
}

/// Memory optimization analysis result
#[derive(Debug, Clone)]
pub struct MemoryAnalysis {
    /// Memory information for each tensor
    pub tensors: Vec<TensorMemory>,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Total memory allocated across all tensors
    pub total_memory_bytes: usize,
    /// Average memory utilization (0.0 to 1.0)
    pub avg_utilization: f64,
    /// Suggested operation execution order for minimal peak memory
    pub optimal_schedule: Vec<usize>,
}

impl MemoryAnalysis {
    /// Create new empty analysis
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            peak_memory_bytes: 0,
            total_memory_bytes: 0,
            avg_utilization: 0.0,
            optimal_schedule: Vec::new(),
        }
    }

    /// Get memory waste (difference between peak and average)
    pub fn memory_waste_ratio(&self) -> f64 {
        if self.peak_memory_bytes == 0 {
            return 0.0;
        }
        let avg_memory = self.total_memory_bytes as f64 * self.avg_utilization;
        (self.peak_memory_bytes as f64 - avg_memory) / self.peak_memory_bytes as f64
    }
}

impl Default for MemoryAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze memory usage patterns in a computation graph
///
/// This function performs a comprehensive analysis of memory usage including:
/// - Tensor lifetime analysis (first use to last use)
/// - Peak memory estimation
/// - Memory utilization statistics
///
/// # Example
///
/// ```rust
/// use tensorlogic_ir::{EinsumGraph, analyze_memory};
///
/// let mut graph = EinsumGraph::new();
/// // Build your graph...
///
/// let analysis = analyze_memory(&graph, 8).unwrap();
/// println!("Peak memory: {} bytes", analysis.peak_memory_bytes);
/// println!("Memory waste ratio: {:.2}%", analysis.memory_waste_ratio() * 100.0);
/// ```
pub fn analyze_memory(
    graph: &EinsumGraph,
    element_size_bytes: usize,
) -> Result<MemoryAnalysis, IrError> {
    if graph.nodes.is_empty() {
        return Ok(MemoryAnalysis::new());
    }

    // Analyze tensor lifetimes
    let tensor_lifetimes = analyze_tensor_lifetimes(graph);

    // Estimate tensor sizes (simplified: assume all tensors are same size)
    let mut tensor_memories = Vec::new();
    for (tensor_idx, (first_use, last_use)) in tensor_lifetimes.iter().enumerate() {
        // Simplified size estimation
        let size_bytes = estimate_tensor_size(graph, tensor_idx, element_size_bytes);
        tensor_memories.push(TensorMemory {
            tensor_idx,
            size_bytes,
            first_use: *first_use,
            last_use: *last_use,
        });
    }

    // Compute peak memory usage
    let peak_memory_bytes = compute_peak_memory(graph, &tensor_memories);

    // Compute total memory
    let total_memory_bytes = tensor_memories.iter().map(|t| t.size_bytes).sum();

    // Estimate average utilization
    let avg_utilization = if graph.nodes.is_empty() {
        0.0
    } else {
        // Average number of live tensors at each step
        let total_live: usize = (0..graph.nodes.len())
            .map(|step| count_live_tensors_at_step(step, &tensor_memories))
            .sum();
        let avg_live = total_live as f64 / graph.nodes.len() as f64;
        let avg_memory = avg_live * (total_memory_bytes as f64 / tensor_memories.len() as f64);
        if peak_memory_bytes > 0 {
            avg_memory / peak_memory_bytes as f64
        } else {
            0.0
        }
    };

    // Generate optimal schedule
    let optimal_schedule = generate_memory_optimal_schedule(graph, &tensor_memories)?;

    Ok(MemoryAnalysis {
        tensors: tensor_memories,
        peak_memory_bytes,
        total_memory_bytes,
        avg_utilization,
        optimal_schedule,
    })
}

/// Analyze when each tensor is first and last used
fn analyze_tensor_lifetimes(graph: &EinsumGraph) -> Vec<(Option<usize>, Option<usize>)> {
    let mut lifetimes = vec![(None, None); graph.tensors.len()];

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        // Update first/last use for inputs
        for &input_idx in &node.inputs {
            if input_idx < lifetimes.len() {
                let (ref mut first, ref mut last) = lifetimes[input_idx];
                *first = Some(first.map_or(node_idx, |f: usize| f.min(node_idx)));
                *last = Some(last.map_or(node_idx, |l: usize| l.max(node_idx)));
            }
        }

        // Update first/last use for outputs
        for &output_idx in &node.outputs {
            if output_idx < lifetimes.len() {
                let (ref mut first, ref mut last) = lifetimes[output_idx];
                *first = Some(first.map_or(node_idx, |f: usize| f.min(node_idx)));
                *last = Some(last.map_or(node_idx, |l: usize| l.max(node_idx)));
            }
        }
    }

    lifetimes
}

/// Estimate the size of a tensor in bytes (simplified)
fn estimate_tensor_size(
    _graph: &EinsumGraph,
    _tensor_idx: usize,
    element_size_bytes: usize,
) -> usize {
    // Simplified: assume 1000 elements per tensor
    // In practice, this would use shape information
    1000 * element_size_bytes
}

/// Compute peak memory usage across all execution steps
fn compute_peak_memory(graph: &EinsumGraph, tensors: &[TensorMemory]) -> usize {
    let mut peak = 0;

    for step in 0..graph.nodes.len() {
        let live_memory: usize = tensors
            .iter()
            .filter(|t| is_tensor_live_at_step(t, step))
            .map(|t| t.size_bytes)
            .sum();
        peak = peak.max(live_memory);
    }

    peak
}

/// Check if a tensor is live at a given execution step
fn is_tensor_live_at_step(tensor: &TensorMemory, step: usize) -> bool {
    match (tensor.first_use, tensor.last_use) {
        (Some(first), Some(last)) => step >= first && step <= last,
        _ => false,
    }
}

/// Count how many tensors are live at a given step
fn count_live_tensors_at_step(step: usize, tensors: &[TensorMemory]) -> usize {
    tensors
        .iter()
        .filter(|t| is_tensor_live_at_step(t, step))
        .count()
}

/// Generate an execution schedule that minimizes peak memory usage
///
/// This uses a greedy algorithm that prioritizes operations that:
/// 1. Free the most memory (last use of large tensors)
/// 2. Have dependencies satisfied
fn generate_memory_optimal_schedule(
    graph: &EinsumGraph,
    _tensors: &[TensorMemory],
) -> Result<Vec<usize>, IrError> {
    // Build dependency graph
    let dependencies = build_dependencies(graph);

    // Topological sort with memory-aware ordering
    let schedule = topological_sort_memory_aware(graph, &dependencies);

    Ok(schedule)
}

/// Build dependency map for the graph
fn build_dependencies(graph: &EinsumGraph) -> HashMap<usize, Vec<usize>> {
    let mut dependencies: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut tensor_producer: HashMap<usize, usize> = HashMap::new();

    // Map each tensor to its producer node
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output_idx in &node.outputs {
            tensor_producer.insert(output_idx, node_idx);
        }
    }

    // Build dependencies
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        let mut deps = Vec::new();
        for &input_idx in &node.inputs {
            if let Some(&producer) = tensor_producer.get(&input_idx) {
                if producer != node_idx {
                    deps.push(producer);
                }
            }
        }
        dependencies.insert(node_idx, deps);
    }

    dependencies
}

/// Topological sort with memory-aware heuristics
fn topological_sort_memory_aware(
    graph: &EinsumGraph,
    dependencies: &HashMap<usize, Vec<usize>>,
) -> Vec<usize> {
    let mut schedule = Vec::new();
    let mut scheduled = HashSet::new();
    let mut in_degree = vec![0; graph.nodes.len()];

    // Calculate in-degrees
    for deps in dependencies.values() {
        for &dep in deps {
            if dep < in_degree.len() {
                in_degree[dep] += 1;
            }
        }
    }

    // Process nodes in order
    while schedule.len() < graph.nodes.len() {
        // Find all ready nodes (in-degree 0)
        let ready: Vec<usize> = (0..graph.nodes.len())
            .filter(|&i| !scheduled.contains(&i) && in_degree[i] == 0)
            .collect();

        if ready.is_empty() {
            break; // No more nodes can be scheduled (possible cycle)
        }

        // Select node with best memory characteristics
        let next = select_next_node_memory_aware(graph, &ready);
        schedule.push(next);
        scheduled.insert(next);

        // Update in-degrees
        if let Some(deps) = dependencies.get(&next) {
            for &dep in deps {
                if dep < in_degree.len() {
                    let current_degree: usize = in_degree[dep];
                    in_degree[dep] = current_degree.saturating_sub(1);
                }
            }
        }
    }

    schedule
}

/// Select next node to schedule based on memory characteristics
fn select_next_node_memory_aware(graph: &EinsumGraph, candidates: &[usize]) -> usize {
    // Simplified: prefer operations that free memory (have fewer outputs)
    candidates
        .iter()
        .min_by_key(|&&idx| {
            graph
                .nodes
                .get(idx)
                .map(|n| n.outputs.len())
                .unwrap_or(usize::MAX)
        })
        .copied()
        .unwrap_or(0)
}

/// Estimate memory savings from in-place operations
///
/// Identifies opportunities where operations could reuse input buffers
/// for outputs (in-place operations) to reduce memory footprint.
pub fn analyze_inplace_opportunities(graph: &EinsumGraph) -> Result<Vec<usize>, IrError> {
    let mut inplace_candidates = Vec::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        if can_be_inplace(&node.op) && has_single_input_use(graph, node_idx) {
            inplace_candidates.push(node_idx);
        }
    }

    Ok(inplace_candidates)
}

/// Check if an operation can be performed in-place
fn can_be_inplace(op_type: &OpType) -> bool {
    // Element-wise unary operations can typically be done in-place
    matches!(op_type, OpType::ElemUnary { .. })
}

/// Check if a node's input tensor is used only by this node
fn has_single_input_use(graph: &EinsumGraph, node_idx: usize) -> bool {
    let node = &graph.nodes[node_idx];
    if node.inputs.is_empty() {
        return false;
    }

    let input_tensor = node.inputs[0];

    // Count how many nodes use this tensor
    let use_count = graph
        .nodes
        .iter()
        .filter(|n| n.inputs.contains(&input_tensor))
        .count();

    use_count == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EinsumNode;

    #[test]
    fn test_memory_analysis_default() {
        let analysis = MemoryAnalysis::default();
        assert_eq!(analysis.peak_memory_bytes, 0);
        assert_eq!(analysis.total_memory_bytes, 0);
    }

    #[test]
    fn test_analyze_empty_graph() {
        let graph = EinsumGraph::new();
        let analysis = analyze_memory(&graph, 8).unwrap();
        assert_eq!(analysis.peak_memory_bytes, 0);
        assert_eq!(analysis.tensors.len(), 0);
    }

    #[test]
    fn test_analyze_single_node() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();

        let analysis = analyze_memory(&graph, 8).unwrap();
        assert!(analysis.peak_memory_bytes > 0);
        assert_eq!(analysis.tensors.len(), 2);
    }

    #[test]
    fn test_tensor_lifetime_single_use() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();

        let lifetimes = analyze_tensor_lifetimes(&graph);
        assert_eq!(lifetimes[a], (Some(0), Some(0)));
        assert_eq!(lifetimes[b], (Some(0), Some(0)));
    }

    #[test]
    fn test_tensor_lifetime_multiple_uses() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("tanh", b, c))
            .unwrap();

        let lifetimes = analyze_tensor_lifetimes(&graph);
        assert_eq!(lifetimes[b], (Some(0), Some(1)));
    }

    #[test]
    fn test_estimate_tensor_size() {
        let graph = EinsumGraph::new();
        let size = estimate_tensor_size(&graph, 0, 8);
        assert_eq!(size, 8000); // 1000 elements * 8 bytes
    }

    #[test]
    fn test_is_tensor_live_at_step() {
        let tensor = TensorMemory {
            tensor_idx: 0,
            size_bytes: 1000,
            first_use: Some(2),
            last_use: Some(5),
        };

        assert!(!is_tensor_live_at_step(&tensor, 0));
        assert!(!is_tensor_live_at_step(&tensor, 1));
        assert!(is_tensor_live_at_step(&tensor, 2));
        assert!(is_tensor_live_at_step(&tensor, 3));
        assert!(is_tensor_live_at_step(&tensor, 5));
        assert!(!is_tensor_live_at_step(&tensor, 6));
    }

    #[test]
    fn test_memory_waste_ratio_zero_peak() {
        let analysis = MemoryAnalysis {
            peak_memory_bytes: 0,
            total_memory_bytes: 1000,
            avg_utilization: 0.5,
            ..Default::default()
        };
        assert_eq!(analysis.memory_waste_ratio(), 0.0);
    }

    #[test]
    fn test_can_be_inplace() {
        assert!(can_be_inplace(&OpType::ElemUnary {
            op: "relu".to_string()
        }));
        assert!(!can_be_inplace(&OpType::Einsum {
            spec: "ij,jk->ik".to_string()
        }));
    }

    #[test]
    fn test_analyze_inplace_opportunities_empty() {
        let graph = EinsumGraph::new();
        let candidates = analyze_inplace_opportunities(&graph).unwrap();
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_analyze_inplace_single_use() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();

        let candidates = analyze_inplace_opportunities(&graph).unwrap();
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_build_dependencies() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("tanh", b, c))
            .unwrap();

        let deps = build_dependencies(&graph);
        assert_eq!(deps.get(&0).unwrap().len(), 0); // Node 0 has no dependencies
        assert_eq!(deps.get(&1).unwrap(), &vec![0]); // Node 1 depends on node 0
    }

    #[test]
    fn test_topological_sort_simple() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");

        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();

        let deps = build_dependencies(&graph);
        let schedule = topological_sort_memory_aware(&graph, &deps);
        assert_eq!(schedule, vec![0]);
    }
}
