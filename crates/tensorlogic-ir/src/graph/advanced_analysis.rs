//! Advanced graph analysis utilities.
//!
//! This module provides sophisticated analysis tools for EinsumGraphs,
//! including critical path analysis, memory estimation, and operation scheduling.

use std::collections::{HashMap, HashSet, VecDeque};

use super::{EinsumGraph, OpType};

/// Memory footprint estimation for a graph.
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)] // Public API, used by library consumers
pub struct MemoryFootprint {
    /// Total number of tensors
    pub tensor_count: usize,
    /// Number of intermediate tensors (not inputs or outputs)
    pub intermediate_count: usize,
    /// Peak live tensors (max tensors alive simultaneously)
    pub peak_live_tensors: usize,
    /// Estimated operations by type
    pub op_counts: HashMap<String, usize>,
}

impl MemoryFootprint {
    /// Estimate memory footprint of a graph.
    #[allow(dead_code)] // Public API method
    pub fn estimate(graph: &EinsumGraph) -> Self {
        let tensor_count = graph.tensors.len();
        let intermediate_count = tensor_count - graph.inputs.len() - graph.outputs.len();

        // Estimate peak live tensors using liveness analysis
        let liveness = compute_liveness(graph);
        let peak_live_tensors = liveness
            .values()
            .map(|live_set| live_set.len())
            .max()
            .unwrap_or(0);

        // Count operations by type
        let mut op_counts = HashMap::new();
        for node in &graph.nodes {
            let op_name = match &node.op {
                OpType::Einsum { .. } => "einsum",
                OpType::ElemUnary { .. } => "elem_unary",
                OpType::ElemBinary { .. } => "elem_binary",
                OpType::Reduce { .. } => "reduce",
            };
            *op_counts.entry(op_name.to_string()).or_insert(0) += 1;
        }

        Self {
            tensor_count,
            intermediate_count,
            peak_live_tensors,
            op_counts,
        }
    }
}

/// Critical path analysis results.
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)] // Public API, used by library consumers
pub struct CriticalPath {
    /// Node indices on the critical path
    pub path: Vec<usize>,
    /// Estimated cost of the critical path
    pub cost: usize,
    /// Maximum parallelism (width of the graph)
    pub max_parallelism: usize,
}

impl CriticalPath {
    /// Compute the critical path through a graph.
    #[allow(dead_code)] // Public API method
    pub fn analyze(graph: &EinsumGraph) -> Self {
        let (distances, predecessors) = compute_distances(graph);

        // Find the output tensor with maximum distance
        let (critical_output, max_distance) = graph
            .outputs
            .iter()
            .filter_map(|&output_idx| distances.get(&output_idx).map(|&dist| (output_idx, dist)))
            .max_by_key(|(_, dist)| *dist)
            .unwrap_or((0, 0));

        // Reconstruct the path
        let mut path = Vec::new();
        let mut current = critical_output;
        while let Some(&pred) = predecessors.get(&current) {
            if let Some(node_idx) = find_producing_node(graph, current) {
                path.push(node_idx);
            }
            current = pred;
        }
        path.reverse();

        // Compute maximum parallelism (max nodes at same level)
        let max_parallelism = compute_max_parallelism(graph, &distances);

        Self {
            path,
            cost: max_distance,
            max_parallelism,
        }
    }
}

/// Execution schedule for a graph.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Public API, used by library consumers
pub struct ExecutionSchedule {
    /// Nodes grouped by execution level (can execute in parallel)
    pub levels: Vec<Vec<usize>>,
    /// Total number of levels (sequential steps required)
    pub depth: usize,
}

impl ExecutionSchedule {
    /// Compute an execution schedule for a graph using topological sorting.
    #[allow(dead_code)] // Public API method
    pub fn compute(graph: &EinsumGraph) -> Self {
        let mut levels = Vec::new();
        let mut in_degree = vec![0; graph.nodes.len()];
        let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();

        // Build adjacency list and compute in-degrees
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &input_tensor in &node.inputs {
                if let Some(producer_idx) = find_producing_node(graph, input_tensor) {
                    adj_list.entry(producer_idx).or_default().push(node_idx);
                    in_degree[node_idx] += 1;
                }
            }
        }

        // Find all nodes with in-degree 0 (can start immediately)
        let mut queue: VecDeque<usize> = (0..graph.nodes.len())
            .filter(|&i| in_degree[i] == 0)
            .collect();

        while !queue.is_empty() {
            let level_size = queue.len();
            let mut current_level = Vec::new();

            for _ in 0..level_size {
                if let Some(node_idx) = queue.pop_front() {
                    current_level.push(node_idx);

                    // Update successors
                    if let Some(successors) = adj_list.get(&node_idx) {
                        for &succ in successors {
                            in_degree[succ] -= 1;
                            if in_degree[succ] == 0 {
                                queue.push_back(succ);
                            }
                        }
                    }
                }
            }

            if !current_level.is_empty() {
                levels.push(current_level);
            }
        }

        let depth = levels.len();
        Self { levels, depth }
    }

    /// Get the maximum parallelism (max nodes in any level).
    #[allow(dead_code)] // Public API method
    pub fn max_parallelism(&self) -> usize {
        self.levels
            .iter()
            .map(|level| level.len())
            .max()
            .unwrap_or(0)
    }
}

/// Data flow analysis results.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Public API, used by library consumers
pub struct DataFlowAnalysis {
    /// Tensors that are read multiple times
    pub reused_tensors: HashSet<usize>,
    /// Tensors that are written multiple times (potential issue)
    pub overwritten_tensors: HashSet<usize>,
    /// Fan-out: number of consumers per tensor
    pub fan_out: HashMap<usize, usize>,
}

impl DataFlowAnalysis {
    /// Analyze data flow in a graph.
    #[allow(dead_code)] // Public API method
    pub fn analyze(graph: &EinsumGraph) -> Self {
        let mut read_count: HashMap<usize, usize> = HashMap::new();
        let mut write_count: HashMap<usize, usize> = HashMap::new();

        // Count reads and writes
        for node in &graph.nodes {
            for &input in &node.inputs {
                *read_count.entry(input).or_insert(0) += 1;
            }
            for &output in &node.outputs {
                *write_count.entry(output).or_insert(0) += 1;
            }
        }

        let reused_tensors = read_count
            .iter()
            .filter(|(_, &count)| count > 1)
            .map(|(&tensor, _)| tensor)
            .collect();

        let overwritten_tensors = write_count
            .iter()
            .filter(|(_, &count)| count > 1)
            .map(|(&tensor, _)| tensor)
            .collect();

        let fan_out = read_count;

        Self {
            reused_tensors,
            overwritten_tensors,
            fan_out,
        }
    }
}

// Helper functions

#[allow(dead_code)] // Used by public API methods
fn compute_liveness(graph: &EinsumGraph) -> HashMap<usize, HashSet<usize>> {
    let mut liveness: HashMap<usize, HashSet<usize>> = HashMap::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        let mut live_set = HashSet::new();

        // Add inputs as live
        for &input in &node.inputs {
            live_set.insert(input);
        }

        // Add outputs as live
        for &output in &node.outputs {
            live_set.insert(output);
        }

        liveness.insert(node_idx, live_set);
    }

    liveness
}

#[allow(dead_code)] // Used by public API methods
fn compute_distances(graph: &EinsumGraph) -> (HashMap<usize, usize>, HashMap<usize, usize>) {
    let mut distances: HashMap<usize, usize> = HashMap::new();
    let mut predecessors: HashMap<usize, usize> = HashMap::new();

    // Initialize input tensors with distance 0
    for &input_idx in &graph.inputs {
        distances.insert(input_idx, 0);
    }

    // Process nodes in topological order
    let schedule = ExecutionSchedule::compute(graph);
    for level in &schedule.levels {
        for &node_idx in level {
            let node = &graph.nodes[node_idx];

            // Compute maximum distance from inputs
            let max_input_distance = node
                .inputs
                .iter()
                .filter_map(|&input| distances.get(&input))
                .max()
                .copied()
                .unwrap_or(0);

            // Update output distances
            for &output in &node.outputs {
                let new_distance = max_input_distance + 1;
                if new_distance > *distances.get(&output).unwrap_or(&0) {
                    distances.insert(output, new_distance);
                    if let Some(&input) = node.inputs.first() {
                        predecessors.insert(output, input);
                    }
                }
            }
        }
    }

    (distances, predecessors)
}

#[allow(dead_code)] // Used by public API methods
fn find_producing_node(graph: &EinsumGraph, tensor_idx: usize) -> Option<usize> {
    graph
        .nodes
        .iter()
        .position(|node| node.outputs.contains(&tensor_idx))
}

#[allow(dead_code)] // Used by public API methods
fn compute_max_parallelism(graph: &EinsumGraph, distances: &HashMap<usize, usize>) -> usize {
    let mut level_counts: HashMap<usize, usize> = HashMap::new();

    for node in &graph.nodes {
        // Find the maximum distance of inputs
        let level = node
            .inputs
            .iter()
            .filter_map(|&input| distances.get(&input))
            .max()
            .copied()
            .unwrap_or(0);

        *level_counts.entry(level).or_insert(0) += 1;
    }

    level_counts.values().copied().max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EinsumNode;

    fn create_simple_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph.add_input(a).unwrap();
        graph.add_input(b).unwrap();
        graph
            .add_node(EinsumNode::einsum("i,j->ij", vec![a, b], vec![c]))
            .unwrap();
        graph.add_output(c).unwrap();

        graph
    }

    fn create_chain_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("t0");
        let t1 = graph.add_tensor("t1");
        let t2 = graph.add_tensor("t2");
        let t3 = graph.add_tensor("t3");

        graph.add_input(t0).unwrap();
        graph
            .add_node(EinsumNode::einsum("i->i", vec![t0], vec![t1]))
            .unwrap();
        graph
            .add_node(EinsumNode::einsum("i->i", vec![t1], vec![t2]))
            .unwrap();
        graph
            .add_node(EinsumNode::einsum("i->i", vec![t2], vec![t3]))
            .unwrap();
        graph.add_output(t3).unwrap();

        graph
    }

    #[test]
    fn test_memory_footprint_simple() {
        let graph = create_simple_graph();
        let footprint = MemoryFootprint::estimate(&graph);

        assert_eq!(footprint.tensor_count, 3);
        assert_eq!(footprint.intermediate_count, 0); // C is output
        assert!(footprint.op_counts.contains_key("einsum"));
    }

    #[test]
    fn test_critical_path_simple() {
        let graph = create_simple_graph();
        let critical = CriticalPath::analyze(&graph);

        assert_eq!(critical.path.len(), 1); // One operation
        assert!(critical.cost > 0);
    }

    #[test]
    fn test_critical_path_chain() {
        let graph = create_chain_graph();
        let critical = CriticalPath::analyze(&graph);

        assert_eq!(critical.path.len(), 3); // Three operations in chain
        assert_eq!(critical.cost, 3); // Distance of 3
    }

    #[test]
    fn test_execution_schedule_simple() {
        let graph = create_simple_graph();
        let schedule = ExecutionSchedule::compute(&graph);

        assert_eq!(schedule.depth, 1); // All can execute in one level
        assert_eq!(schedule.max_parallelism(), 1);
    }

    #[test]
    fn test_execution_schedule_chain() {
        let graph = create_chain_graph();
        let schedule = ExecutionSchedule::compute(&graph);

        assert_eq!(schedule.depth, 3); // Sequential execution
        assert_eq!(schedule.max_parallelism(), 1); // No parallelism in chain
    }

    #[test]
    fn test_data_flow_analysis() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph.add_input(a).unwrap();
        // Use 'a' twice
        graph
            .add_node(EinsumNode::einsum("i->i", vec![a], vec![b]))
            .unwrap();
        graph
            .add_node(EinsumNode::einsum("i->i", vec![a], vec![c]))
            .unwrap();
        graph.add_output(b).unwrap();
        graph.add_output(c).unwrap();

        let analysis = DataFlowAnalysis::analyze(&graph);

        assert!(analysis.reused_tensors.contains(&a)); // 'a' is reused
        assert_eq!(*analysis.fan_out.get(&a).unwrap(), 2); // 'a' has fan-out of 2
    }

    #[test]
    fn test_data_flow_no_reuse() {
        let graph = create_simple_graph();
        let analysis = DataFlowAnalysis::analyze(&graph);

        // In simple graph, no tensor is reused
        assert!(analysis.reused_tensors.is_empty());
    }
}
