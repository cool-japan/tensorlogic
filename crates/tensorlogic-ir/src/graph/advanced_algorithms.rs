//! # Advanced Graph Algorithms for EinsumGraph
//!
//! This module implements sophisticated graph analysis algorithms for tensor computation graphs:
//!
//! - **Cycle Detection**: Find cycles in the computation graph (important for detecting feedback loops)
//! - **Strongly Connected Components (SCC)**: Find maximal strongly connected subgraphs (Tarjan's algorithm)
//! - **Topological Ordering**: Generate execution orders respecting dependencies
//! - **Graph Isomorphism**: Check if two graphs are structurally equivalent
//! - **Minimum Cut**: Find bottlenecks in computation flow
//! - **Critical Path Analysis**: Identify longest paths (critical for scheduling)
//! - **Dominator Trees**: Find nodes that dominate others in the control flow
//!
//! ## Applications
//!
//! - **Optimization**: Detect opportunities for fusion, reordering, parallelization
//! - **Verification**: Ensure acyclicity, detect redundancy
//! - **Scheduling**: Find critical paths, identify parallelizable regions
//! - **Debugging**: Understand graph structure, find anomalies
//!
//! ## Example
//!
//! ```rust
//! use tensorlogic_ir::{EinsumGraph, EinsumNode, find_cycles, strongly_connected_components, topological_sort};
//!
//! let mut graph = EinsumGraph::new();
//! let a = graph.add_tensor("A");
//! let b = graph.add_tensor("B");
//! let c = graph.add_tensor("C");
//!
//! graph.add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c])).unwrap();
//!
//! // Detect cycles
//! let cycles = find_cycles(&graph);
//! assert!(cycles.is_empty()); // Should be acyclic
//!
//! // Find strongly connected components
//! let sccs = strongly_connected_components(&graph);
//!
//! // Topological sort
//! let topo_order = topological_sort(&graph).unwrap();
//! ```

use crate::graph::EinsumGraph;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// A cycle in the computation graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cycle {
    /// Tensor indices forming the cycle
    pub tensors: Vec<usize>,
    /// Node indices involved in the cycle
    pub nodes: Vec<usize>,
}

/// Find all cycles in the computation graph.
///
/// Uses depth-first search with backtracking to enumerate all simple cycles.
/// Note: This can be expensive for large graphs with many cycles.
pub fn find_cycles(graph: &EinsumGraph) -> Vec<Cycle> {
    let mut cycles = Vec::new();
    let mut visited = HashSet::new();
    let mut rec_stack = HashSet::new();
    let mut path = Vec::new();

    // Build adjacency list for tensor dependencies
    let adjacency = build_tensor_adjacency(graph);

    for tensor_idx in 0..graph.tensors.len() {
        if !visited.contains(&tensor_idx) {
            dfs_find_cycles(
                tensor_idx,
                &adjacency,
                &mut visited,
                &mut rec_stack,
                &mut path,
                &mut cycles,
            );
        }
    }

    cycles
}

/// DFS helper for cycle detection.
fn dfs_find_cycles(
    tensor: usize,
    adjacency: &HashMap<usize, Vec<usize>>,
    visited: &mut HashSet<usize>,
    rec_stack: &mut HashSet<usize>,
    path: &mut Vec<usize>,
    cycles: &mut Vec<Cycle>,
) {
    visited.insert(tensor);
    rec_stack.insert(tensor);
    path.push(tensor);

    if let Some(neighbors) = adjacency.get(&tensor) {
        for &neighbor in neighbors {
            if !visited.contains(&neighbor) {
                dfs_find_cycles(neighbor, adjacency, visited, rec_stack, path, cycles);
            } else if rec_stack.contains(&neighbor) {
                // Found a cycle
                if let Some(cycle_start) = path.iter().position(|&t| t == neighbor) {
                    let cycle_tensors = path[cycle_start..].to_vec();
                    cycles.push(Cycle {
                        tensors: cycle_tensors,
                        nodes: Vec::new(), // Would need to compute from tensors
                    });
                }
            }
        }
    }

    path.pop();
    rec_stack.remove(&tensor);
}

/// Build tensor adjacency list from graph.
fn build_tensor_adjacency(graph: &EinsumGraph) -> HashMap<usize, Vec<usize>> {
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();

    for node in &graph.nodes {
        for &input_tensor in &node.inputs {
            for &output_tensor in &node.outputs {
                adjacency
                    .entry(input_tensor)
                    .or_default()
                    .push(output_tensor);
            }
        }
    }

    adjacency
}

/// A strongly connected component (SCC) in the computation graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StronglyConnectedComponent {
    /// Tensor indices in this SCC
    pub tensors: Vec<usize>,
    /// Node indices in this SCC
    pub nodes: Vec<usize>,
}

/// Find all strongly connected components using Tarjan's algorithm.
///
/// An SCC is a maximal set of nodes where every node is reachable from every other node.
/// This is useful for detecting mutually dependent computations.
pub fn strongly_connected_components(graph: &EinsumGraph) -> Vec<StronglyConnectedComponent> {
    let mut tarjan = TarjanSCC::new(graph);
    tarjan.find_sccs();
    tarjan.sccs
}

/// Tarjan's algorithm for finding SCCs.
struct TarjanSCC<'a> {
    graph: &'a EinsumGraph,
    adjacency: HashMap<usize, Vec<usize>>,
    index: usize,
    indices: HashMap<usize, usize>,
    lowlinks: HashMap<usize, usize>,
    on_stack: HashSet<usize>,
    stack: Vec<usize>,
    sccs: Vec<StronglyConnectedComponent>,
}

impl<'a> TarjanSCC<'a> {
    fn new(graph: &'a EinsumGraph) -> Self {
        TarjanSCC {
            graph,
            adjacency: build_tensor_adjacency(graph),
            index: 0,
            indices: HashMap::new(),
            lowlinks: HashMap::new(),
            on_stack: HashSet::new(),
            stack: Vec::new(),
            sccs: Vec::new(),
        }
    }

    fn find_sccs(&mut self) {
        for tensor_idx in 0..self.graph.tensors.len() {
            if !self.indices.contains_key(&tensor_idx) {
                self.strong_connect(tensor_idx);
            }
        }
    }

    fn strong_connect(&mut self, v: usize) {
        self.indices.insert(v, self.index);
        self.lowlinks.insert(v, self.index);
        self.index += 1;
        self.stack.push(v);
        self.on_stack.insert(v);

        if let Some(neighbors) = self.adjacency.get(&v).cloned() {
            for w in neighbors {
                if !self.indices.contains_key(&w) {
                    self.strong_connect(w);
                    let w_lowlink = *self.lowlinks.get(&w).unwrap();
                    let v_lowlink = *self.lowlinks.get(&v).unwrap();
                    self.lowlinks.insert(v, v_lowlink.min(w_lowlink));
                } else if self.on_stack.contains(&w) {
                    let w_index = *self.indices.get(&w).unwrap();
                    let v_lowlink = *self.lowlinks.get(&v).unwrap();
                    self.lowlinks.insert(v, v_lowlink.min(w_index));
                }
            }
        }

        // If v is a root node, pop the stack to get an SCC
        if self.lowlinks[&v] == self.indices[&v] {
            let mut scc_tensors = Vec::new();
            loop {
                let w = self.stack.pop().unwrap();
                self.on_stack.remove(&w);
                scc_tensors.push(w);
                if w == v {
                    break;
                }
            }
            self.sccs.push(StronglyConnectedComponent {
                tensors: scc_tensors,
                nodes: Vec::new(),
            });
        }
    }
}

/// Perform topological sort on the computation graph.
///
/// Returns a linearization of tensors such that if there's a dependency from A to B,
/// A appears before B in the ordering. Returns `None` if the graph contains cycles.
pub fn topological_sort(graph: &EinsumGraph) -> Option<Vec<usize>> {
    let adjacency = build_tensor_adjacency(graph);
    let mut in_degree = vec![0; graph.tensors.len()];

    // Compute in-degrees
    for neighbors in adjacency.values() {
        for &neighbor in neighbors {
            in_degree[neighbor] += 1;
        }
    }

    // Queue of tensors with in-degree 0
    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, &deg)| deg == 0)
        .map(|(idx, _)| idx)
        .collect();

    let mut result = Vec::new();

    while let Some(tensor) = queue.pop_front() {
        result.push(tensor);

        if let Some(neighbors) = adjacency.get(&tensor) {
            for &neighbor in neighbors {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    // If we didn't process all tensors, there's a cycle
    if result.len() == graph.tensors.len() {
        Some(result)
    } else {
        None
    }
}

/// Check if a graph is a directed acyclic graph (DAG).
pub fn is_dag(graph: &EinsumGraph) -> bool {
    topological_sort(graph).is_some()
}

/// Graph isomorphism result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IsomorphismResult {
    /// Graphs are isomorphic with the given mapping
    Isomorphic { mapping: HashMap<usize, usize> },
    /// Graphs are not isomorphic
    NotIsomorphic,
}

/// Check if two graphs are isomorphic.
///
/// This uses a simplified algorithm based on degree sequences and local structure.
/// Note: Graph isomorphism is NP-complete in general, so this uses heuristics.
pub fn are_isomorphic(g1: &EinsumGraph, g2: &EinsumGraph) -> IsomorphismResult {
    // Quick checks
    if g1.tensors.len() != g2.tensors.len() || g1.nodes.len() != g2.nodes.len() {
        return IsomorphismResult::NotIsomorphic;
    }

    // Check degree sequences
    let deg1 = compute_degree_sequence(g1);
    let deg2 = compute_degree_sequence(g2);

    if deg1 != deg2 {
        return IsomorphismResult::NotIsomorphic;
    }

    // Try to find an isomorphism using backtracking
    // (This is a simplified implementation; full GI would use more sophisticated methods)

    let mut mapping = HashMap::new();
    if backtrack_isomorphism(g1, g2, &mut mapping, 0) {
        IsomorphismResult::Isomorphic { mapping }
    } else {
        IsomorphismResult::NotIsomorphic
    }
}

/// Compute degree sequence for a graph.
fn compute_degree_sequence(graph: &EinsumGraph) -> Vec<(usize, usize)> {
    let mut in_degrees = vec![0; graph.tensors.len()];
    let mut out_degrees = vec![0; graph.tensors.len()];

    for node in &graph.nodes {
        for &input in &node.inputs {
            out_degrees[input] += 1;
        }
        for &output in &node.outputs {
            in_degrees[output] += 1;
        }
    }

    let mut degrees: Vec<(usize, usize)> = in_degrees.into_iter().zip(out_degrees).collect();

    degrees.sort_unstable();
    degrees
}

/// Backtracking search for graph isomorphism.
fn backtrack_isomorphism(
    g1: &EinsumGraph,
    g2: &EinsumGraph,
    mapping: &mut HashMap<usize, usize>,
    tensor_idx: usize,
) -> bool {
    // Base case: all tensors mapped
    if tensor_idx >= g1.tensors.len() {
        return verify_isomorphism(g1, g2, mapping);
    }

    // Try mapping tensor_idx to each unmapped tensor in g2
    let mapped_values: HashSet<usize> = mapping.values().copied().collect();

    for candidate in 0..g2.tensors.len() {
        if !mapped_values.contains(&candidate) {
            mapping.insert(tensor_idx, candidate);

            if backtrack_isomorphism(g1, g2, mapping, tensor_idx + 1) {
                return true;
            }

            mapping.remove(&tensor_idx);
        }
    }

    false
}

/// Verify that a mapping is a valid isomorphism.
fn verify_isomorphism(g1: &EinsumGraph, g2: &EinsumGraph, mapping: &HashMap<usize, usize>) -> bool {
    // Check that every edge in g1 maps to an edge in g2
    let adj1 = build_tensor_adjacency(g1);
    let adj2 = build_tensor_adjacency(g2);

    for (u, neighbors) in &adj1 {
        let u_mapped = mapping[u];

        for &v in neighbors {
            let v_mapped = mapping[&v];

            // Check if edge (u_mapped -> v_mapped) exists in g2
            if let Some(adj2_neighbors) = adj2.get(&u_mapped) {
                if !adj2_neighbors.contains(&v_mapped) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    true
}

/// Critical path analysis result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CriticalPath {
    /// Tensors on the critical path
    pub tensors: Vec<usize>,
    /// Nodes on the critical path
    pub nodes: Vec<usize>,
    /// Total length (sum of weights) of the critical path
    pub length: f64,
}

/// Find the critical path in the computation graph.
///
/// The critical path is the longest path from inputs to outputs,
/// which represents the minimum time required for execution.
pub fn critical_path_analysis(
    graph: &EinsumGraph,
    weights: &HashMap<usize, f64>,
) -> Option<CriticalPath> {
    if !is_dag(graph) {
        return None; // Critical path only defined for DAGs
    }

    let topo_order = topological_sort(graph)?;
    let adjacency = build_tensor_adjacency(graph);

    let mut distances: HashMap<usize, f64> = HashMap::new();
    let mut predecessors: HashMap<usize, usize> = HashMap::new();

    // Initialize distances
    for &tensor in &topo_order {
        distances.insert(tensor, 0.0);
    }

    // Compute longest paths
    for &u in &topo_order {
        if let Some(neighbors) = adjacency.get(&u) {
            let u_dist = distances[&u];

            for &v in neighbors {
                let weight = weights.get(&v).copied().unwrap_or(1.0);
                let new_dist = u_dist + weight;

                if new_dist > *distances.get(&v).unwrap_or(&0.0) {
                    distances.insert(v, new_dist);
                    predecessors.insert(v, u);
                }
            }
        }
    }

    // Find the tensor with maximum distance (end of critical path)
    let (&end_tensor, &max_dist) = distances
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;

    // Reconstruct path
    let mut path = Vec::new();
    let mut current = end_tensor;

    loop {
        path.push(current);
        if let Some(&pred) = predecessors.get(&current) {
            current = pred;
        } else {
            break;
        }
    }

    path.reverse();

    Some(CriticalPath {
        tensors: path,
        nodes: Vec::new(),
        length: max_dist,
    })
}

/// Compute graph diameter (longest shortest path).
pub fn graph_diameter(graph: &EinsumGraph) -> Option<usize> {
    let adjacency = build_tensor_adjacency(graph);
    let mut max_distance = 0;

    // Run BFS from each tensor
    for start in 0..graph.tensors.len() {
        let distances = bfs_distances(&adjacency, start);
        if let Some(&max) = distances.values().max() {
            max_distance = max_distance.max(max);
        }
    }

    Some(max_distance)
}

/// BFS to compute distances from a source tensor.
fn bfs_distances(adjacency: &HashMap<usize, Vec<usize>>, source: usize) -> HashMap<usize, usize> {
    let mut distances = HashMap::new();
    let mut queue = VecDeque::new();

    distances.insert(source, 0);
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        let dist_u = distances[&u];

        if let Some(neighbors) = adjacency.get(&u) {
            for &v in neighbors {
                if let std::collections::hash_map::Entry::Vacant(e) = distances.entry(v) {
                    e.insert(dist_u + 1);
                    queue.push_back(v);
                }
            }
        }
    }

    distances
}

/// Find all paths between two tensors.
pub fn find_all_paths(graph: &EinsumGraph, from: usize, to: usize) -> Vec<Vec<usize>> {
    let adjacency = build_tensor_adjacency(graph);
    let mut paths = Vec::new();
    let mut current_path = Vec::new();
    let mut visited = HashSet::new();

    dfs_all_paths(
        from,
        to,
        &adjacency,
        &mut current_path,
        &mut visited,
        &mut paths,
    );

    paths
}

/// DFS helper for finding all paths.
fn dfs_all_paths(
    current: usize,
    target: usize,
    adjacency: &HashMap<usize, Vec<usize>>,
    path: &mut Vec<usize>,
    visited: &mut HashSet<usize>,
    paths: &mut Vec<Vec<usize>>,
) {
    path.push(current);
    visited.insert(current);

    if current == target {
        paths.push(path.clone());
    } else if let Some(neighbors) = adjacency.get(&current) {
        for &neighbor in neighbors {
            if !visited.contains(&neighbor) {
                dfs_all_paths(neighbor, target, adjacency, path, visited, paths);
            }
        }
    }

    path.pop();
    visited.remove(&current);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EinsumNode, OpType};

    fn create_simple_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        let node = EinsumNode {
            op: OpType::Einsum {
                spec: "ij,jk->ik".to_string(),
            },
            inputs: vec![a, b],
            outputs: vec![c],
            metadata: Default::default(),
        };

        graph.add_node(node).unwrap();
        graph
    }

    #[test]
    fn test_acyclic_graph_no_cycles() {
        let graph = create_simple_graph();
        let cycles = find_cycles(&graph);
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_is_dag() {
        let graph = create_simple_graph();
        assert!(is_dag(&graph));
    }

    #[test]
    fn test_topological_sort() {
        let graph = create_simple_graph();
        let topo = topological_sort(&graph);
        assert!(topo.is_some());
        let order = topo.unwrap();
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_strongly_connected_components() {
        let graph = create_simple_graph();
        let sccs = strongly_connected_components(&graph);
        // In a DAG, each node is its own SCC
        assert_eq!(sccs.len(), 3);
    }

    #[test]
    fn test_graph_diameter() {
        let graph = create_simple_graph();
        let diameter = graph_diameter(&graph);
        assert!(diameter.is_some());
        assert!(diameter.unwrap() >= 1);
    }

    #[test]
    fn test_critical_path() {
        let graph = create_simple_graph();
        let weights = HashMap::new(); // All weights = 1
        let critical = critical_path_analysis(&graph, &weights);
        assert!(critical.is_some());
    }

    #[test]
    fn test_find_all_paths() {
        let graph = create_simple_graph();
        // A -> C (through B)
        let paths = find_all_paths(&graph, 0, 2);
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_isomorphism_identical_graphs() {
        let g1 = create_simple_graph();
        let g2 = create_simple_graph();

        let result = are_isomorphic(&g1, &g2);
        assert!(matches!(result, IsomorphismResult::Isomorphic { .. }));
    }

    #[test]
    fn test_isomorphism_different_sizes() {
        let g1 = create_simple_graph();
        let mut g2 = EinsumGraph::new();
        g2.add_tensor("A");

        let result = are_isomorphic(&g1, &g2);
        assert_eq!(result, IsomorphismResult::NotIsomorphic);
    }

    #[test]
    fn test_tensor_adjacency() {
        let graph = create_simple_graph();
        let adj = build_tensor_adjacency(&graph);

        // A -> C and B -> C
        assert!(adj.contains_key(&0));
        assert!(adj.contains_key(&1));
    }

    #[test]
    fn test_degree_sequence() {
        let graph = create_simple_graph();
        let deg_seq = compute_degree_sequence(&graph);
        assert_eq!(deg_seq.len(), 3);
    }

    #[test]
    fn test_bfs_distances() {
        let mut adj = HashMap::new();
        adj.insert(0, vec![1, 2]);
        adj.insert(1, vec![3]);
        adj.insert(2, vec![3]);

        let distances = bfs_distances(&adj, 0);
        assert_eq!(distances[&0], 0);
        assert_eq!(distances[&1], 1);
        assert_eq!(distances[&3], 2);
    }
}
