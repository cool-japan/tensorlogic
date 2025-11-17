//! Parallelization analysis for identifying independent subgraphs.
//!
//! This module provides utilities for analyzing computational graphs to identify
//! opportunities for parallel execution. It finds groups of operations that have
//! no dependencies on each other and can safely execute concurrently.

use std::collections::{HashMap, HashSet, VecDeque};

use super::EinsumGraph;
use crate::error::IrError;

/// A group of nodes that can execute in parallel
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelGroup {
    /// Indices of nodes in this parallel group
    pub nodes: Vec<usize>,
    /// Estimated computation cost of this group
    pub estimated_cost: f64,
    /// Level in the execution schedule (for visualization)
    pub level: usize,
}

/// Analysis result containing parallel execution opportunities
#[derive(Debug, Clone)]
pub struct ParallelizationAnalysis {
    /// Groups of nodes that can execute in parallel at each level
    pub parallel_groups: Vec<ParallelGroup>,
    /// Maximum parallelism (largest group size)
    pub max_parallelism: usize,
    /// Average parallelism across all levels
    pub avg_parallelism: f64,
    /// Critical path length (longest dependency chain)
    pub critical_path_length: usize,
    /// Nodes on the critical path
    pub critical_path: Vec<usize>,
    /// Estimated parallel speedup (compared to sequential execution)
    pub estimated_speedup: f64,
}

impl ParallelizationAnalysis {
    /// Create a new empty analysis
    pub fn new() -> Self {
        Self {
            parallel_groups: Vec::new(),
            max_parallelism: 0,
            avg_parallelism: 0.0,
            critical_path_length: 0,
            critical_path: Vec::new(),
            estimated_speedup: 1.0,
        }
    }

    /// Check if the graph has any parallelism opportunities
    pub fn has_parallelism(&self) -> bool {
        self.max_parallelism > 1
    }

    /// Get total number of nodes across all parallel groups
    pub fn total_nodes(&self) -> usize {
        self.parallel_groups.iter().map(|g| g.nodes.len()).sum()
    }
}

impl Default for ParallelizationAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze graph for parallel execution opportunities
///
/// This function performs a topological analysis of the computation graph
/// to identify sets of operations that can execute in parallel. It uses
/// level-based scheduling to group independent operations.
///
/// # Returns
///
/// Returns a `ParallelizationAnalysis` containing:
/// - Groups of parallelizable operations at each level
/// - Critical path analysis
/// - Estimated speedup from parallelization
///
/// # Example
///
/// ```rust
/// use tensorlogic_ir::{EinsumGraph, analyze_parallelization};
///
/// let mut graph = EinsumGraph::new();
/// // Build your graph...
///
/// let analysis = analyze_parallelization(&graph).unwrap();
/// if analysis.has_parallelism() {
///     println!("Max parallelism: {}", analysis.max_parallelism);
///     println!("Estimated speedup: {:.2}x", analysis.estimated_speedup);
/// }
/// ```
pub fn analyze_parallelization(graph: &EinsumGraph) -> Result<ParallelizationAnalysis, IrError> {
    if graph.nodes.is_empty() {
        return Ok(ParallelizationAnalysis::new());
    }

    // Build dependency information
    let (dependencies, dependents) = build_dependency_graph(graph);

    // Compute node levels using topological sort
    let node_levels = compute_node_levels(graph, &dependencies);

    // Group nodes by level
    let mut level_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node_idx, &level) in node_levels.iter().enumerate() {
        level_groups.entry(level).or_default().push(node_idx);
    }

    // Create parallel groups
    let mut parallel_groups = Vec::new();
    let max_level = node_levels.iter().max().copied().unwrap_or(0);

    for level in 0..=max_level {
        if let Some(nodes) = level_groups.get(&level) {
            let estimated_cost = estimate_group_cost(graph, nodes);
            parallel_groups.push(ParallelGroup {
                nodes: nodes.clone(),
                estimated_cost,
                level,
            });
        }
    }

    // Compute statistics
    let max_parallelism = parallel_groups
        .iter()
        .map(|g| g.nodes.len())
        .max()
        .unwrap_or(0);

    let total_nodes: usize = parallel_groups.iter().map(|g| g.nodes.len()).sum();
    let avg_parallelism = if !parallel_groups.is_empty() {
        total_nodes as f64 / parallel_groups.len() as f64
    } else {
        0.0
    };

    // Find critical path
    let (critical_path, critical_path_length) =
        find_critical_path(graph, &node_levels, &dependents);

    // Estimate speedup (simplified model)
    let sequential_cost: f64 = (0..graph.nodes.len())
        .map(|i| estimate_node_cost(graph, i))
        .sum();
    let parallel_cost: f64 = parallel_groups.iter().map(|g| g.estimated_cost).sum();
    let estimated_speedup = if parallel_cost > 0.0 {
        sequential_cost / parallel_cost
    } else {
        1.0
    };

    Ok(ParallelizationAnalysis {
        parallel_groups,
        max_parallelism,
        avg_parallelism,
        critical_path_length,
        critical_path,
        estimated_speedup,
    })
}

/// Build dependency graph (forward and backward)
fn build_dependency_graph(
    graph: &EinsumGraph,
) -> (HashMap<usize, Vec<usize>>, HashMap<usize, Vec<usize>>) {
    let mut dependencies: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();

    // Build tensor-to-producer mapping
    let mut tensor_producer: HashMap<usize, usize> = HashMap::new();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output_idx in &node.outputs {
            tensor_producer.insert(output_idx, node_idx);
        }
    }

    // Build dependency relationships
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        let mut node_deps = Vec::new();
        for &input_idx in &node.inputs {
            if let Some(&producer_idx) = tensor_producer.get(&input_idx) {
                if producer_idx != node_idx {
                    node_deps.push(producer_idx);
                    dependents.entry(producer_idx).or_default().push(node_idx);
                }
            }
        }
        dependencies.insert(node_idx, node_deps);
    }

    (dependencies, dependents)
}

/// Compute the execution level for each node using topological sort
fn compute_node_levels(
    graph: &EinsumGraph,
    dependencies: &HashMap<usize, Vec<usize>>,
) -> Vec<usize> {
    let mut levels = vec![0; graph.nodes.len()];
    let mut in_degree = vec![0; graph.nodes.len()];

    // Calculate in-degrees (count how many nodes each node depends on)
    for (node_idx, deps) in dependencies.iter() {
        in_degree[*node_idx] = deps.len();
    }

    // Find all nodes with no dependencies (level 0)
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (node_idx, &degree) in in_degree.iter().enumerate() {
        if degree == 0 && node_idx < graph.nodes.len() {
            queue.push_back(node_idx);
            levels[node_idx] = 0;
        }
    }

    // Build reverse dependency map (who depends on me?)
    let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();
    for (node_idx, deps) in dependencies.iter() {
        for &dep in deps {
            dependents.entry(dep).or_default().push(*node_idx);
        }
    }

    // BFS to assign levels
    let mut visited = HashSet::new();
    while let Some(node_idx) = queue.pop_front() {
        if visited.contains(&node_idx) {
            continue;
        }
        visited.insert(node_idx);

        let current_level = levels[node_idx];

        // Update nodes that depend on this one
        if let Some(deps) = dependents.get(&node_idx) {
            for &dep_idx in deps {
                if dep_idx < graph.nodes.len() {
                    levels[dep_idx] = levels[dep_idx].max(current_level + 1);
                    queue.push_back(dep_idx);
                }
            }
        }
    }

    levels
}

/// Estimate computational cost for a group of nodes
fn estimate_group_cost(graph: &EinsumGraph, nodes: &[usize]) -> f64 {
    nodes
        .iter()
        .map(|&idx| estimate_node_cost(graph, idx))
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0)
}

/// Estimate computational cost for a single node (simplified)
fn estimate_node_cost(_graph: &EinsumGraph, _node_idx: usize) -> f64 {
    // Simplified cost model - in practice, this would analyze the operation type
    // and tensor sizes to estimate FLOPs and memory traffic
    1.0
}

/// Find the critical path in the computation graph
fn find_critical_path(
    graph: &EinsumGraph,
    node_levels: &[usize],
    _dependents: &HashMap<usize, Vec<usize>>,
) -> (Vec<usize>, usize) {
    let max_level = node_levels.iter().max().copied().unwrap_or(0);

    // Find nodes at maximum level
    let end_nodes: Vec<usize> = node_levels
        .iter()
        .enumerate()
        .filter(|(_, &level)| level == max_level)
        .map(|(idx, _)| idx)
        .collect();

    if end_nodes.is_empty() {
        return (Vec::new(), 0);
    }

    // Backtrack from end node to find critical path
    let mut path = Vec::new();
    let mut current = end_nodes[0];
    path.push(current);

    while node_levels[current] > 0 {
        // Find predecessor with highest level
        let predecessors = get_predecessors(graph, current);
        if let Some(&pred) = predecessors
            .iter()
            .max_by_key(|&&idx| node_levels.get(idx).copied().unwrap_or(0))
        {
            path.push(pred);
            current = pred;
        } else {
            break;
        }
    }

    path.reverse();
    let length = path.len();
    (path, length)
}

/// Get predecessor nodes for a given node
fn get_predecessors(graph: &EinsumGraph, node_idx: usize) -> Vec<usize> {
    let mut predecessors = Vec::new();

    // Build tensor-to-producer mapping
    let mut tensor_producer: HashMap<usize, usize> = HashMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        for &output in &node.outputs {
            tensor_producer.insert(output, idx);
        }
    }

    // Find nodes that produce inputs for this node
    if let Some(node) = graph.nodes.get(node_idx) {
        for &input in &node.inputs {
            if let Some(&producer) = tensor_producer.get(&input) {
                predecessors.push(producer);
            }
        }
    }

    predecessors
}

/// Partition graph into independent subgraphs for parallel execution
///
/// This function divides the computation graph into the largest possible
/// independent subgraphs that can execute in parallel without any
/// data dependencies between them.
pub fn partition_independent_subgraphs(graph: &EinsumGraph) -> Result<Vec<Vec<usize>>, IrError> {
    if graph.nodes.is_empty() {
        return Ok(Vec::new());
    }

    let (dependencies, dependents) = build_dependency_graph(graph);
    let mut visited = HashSet::new();
    let mut subgraphs = Vec::new();

    for node_idx in 0..graph.nodes.len() {
        if visited.contains(&node_idx) {
            continue;
        }

        let mut subgraph = Vec::new();
        let mut stack = vec![node_idx];

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);
            subgraph.push(current);

            // Add dependencies and dependents
            if let Some(deps) = dependencies.get(&current) {
                stack.extend(deps.iter().copied());
            }
            if let Some(deps) = dependents.get(&current) {
                stack.extend(deps.iter().copied());
            }
        }

        subgraphs.push(subgraph);
    }

    Ok(subgraphs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EinsumNode;

    #[test]
    fn test_parallelization_analysis_default() {
        let analysis = ParallelizationAnalysis::default();
        assert_eq!(analysis.max_parallelism, 0);
        assert!(!analysis.has_parallelism());
    }

    #[test]
    fn test_analyze_empty_graph() {
        let graph = EinsumGraph::new();
        let analysis = analyze_parallelization(&graph).unwrap();
        assert_eq!(analysis.max_parallelism, 0);
        assert_eq!(analysis.total_nodes(), 0);
    }

    #[test]
    fn test_analyze_single_node() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();

        let analysis = analyze_parallelization(&graph).unwrap();
        assert_eq!(analysis.max_parallelism, 1);
        assert_eq!(analysis.total_nodes(), 1);
    }

    #[test]
    fn test_analyze_parallel_nodes() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");
        let d = graph.add_tensor("D");

        // Two independent operations
        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("tanh", c, d))
            .unwrap();

        let analysis = analyze_parallelization(&graph).unwrap();
        assert_eq!(analysis.max_parallelism, 2);
        assert!(analysis.has_parallelism());
    }

    #[test]
    fn test_analyze_sequential_nodes() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        // Sequential operations
        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("tanh", b, c))
            .unwrap();

        let analysis = analyze_parallelization(&graph).unwrap();
        assert_eq!(analysis.critical_path_length, 2);
    }

    #[test]
    fn test_partition_empty_graph() {
        let graph = EinsumGraph::new();
        let subgraphs = partition_independent_subgraphs(&graph).unwrap();
        assert!(subgraphs.is_empty());
    }

    #[test]
    fn test_partition_single_node() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();

        let subgraphs = partition_independent_subgraphs(&graph).unwrap();
        assert_eq!(subgraphs.len(), 1);
        assert_eq!(subgraphs[0].len(), 1);
    }

    #[test]
    fn test_partition_independent_nodes() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");
        let d = graph.add_tensor("D");

        // Two truly independent operations (no shared tensors)
        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("tanh", c, d))
            .unwrap();

        let subgraphs = partition_independent_subgraphs(&graph).unwrap();
        // Should have 2 independent subgraphs
        assert_eq!(subgraphs.len(), 2);
    }

    #[test]
    fn test_estimate_node_cost() {
        let graph = EinsumGraph::new();
        let cost = estimate_node_cost(&graph, 0);
        assert_eq!(cost, 1.0);
    }

    #[test]
    fn test_estimate_group_cost() {
        let graph = EinsumGraph::new();
        let cost = estimate_group_cost(&graph, &[0, 1, 2]);
        assert_eq!(cost, 1.0); // Max of individual costs
    }

    #[test]
    fn test_parallel_group_creation() {
        let group = ParallelGroup {
            nodes: vec![0, 1, 2],
            estimated_cost: 3.5,
            level: 1,
        };
        assert_eq!(group.nodes.len(), 3);
        assert_eq!(group.estimated_cost, 3.5);
        assert_eq!(group.level, 1);
    }
}
