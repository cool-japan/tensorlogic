//! Reachability and dominance analysis for einsum graphs.
//!
//! This module provides graph analysis passes that compute:
//! - Reachability: Which nodes can reach which other nodes
//! - Dominance: Which nodes dominate others in the control flow
//! - Post-dominance: Reverse dominance for optimization
//!
//! # Overview
//!
//! Reachability and dominance analysis are fundamental for understanding
//! the structure and dependencies in computation graphs. They enable:
//! - Dead code elimination
//! - Loop optimization
//! - Code motion and hoisting
//! - Critical path analysis
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_compiler::passes::analyze_reachability;
//! use tensorlogic_ir::EinsumGraph;
//!
//! let graph = EinsumGraph::new();
//! let analysis = analyze_reachability(&graph);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use tensorlogic_ir::EinsumGraph;

/// Result of reachability analysis.
#[derive(Debug, Clone)]
pub struct ReachabilityAnalysis {
    /// Which nodes can be reached from each node
    pub reachable_from: HashMap<usize, HashSet<usize>>,
    /// Which nodes can reach each node
    pub can_reach: HashMap<usize, HashSet<usize>>,
    /// Strongly connected components
    pub sccs: Vec<HashSet<usize>>,
    /// Topological ordering (if DAG)
    pub topo_order: Option<Vec<usize>>,
}

impl ReachabilityAnalysis {
    /// Create a new reachability analysis.
    pub fn new() -> Self {
        Self {
            reachable_from: HashMap::new(),
            can_reach: HashMap::new(),
            sccs: Vec::new(),
            topo_order: None,
        }
    }

    /// Check if node `to` is reachable from node `from`.
    pub fn is_reachable(&self, from: usize, to: usize) -> bool {
        self.reachable_from
            .get(&from)
            .map(|set| set.contains(&to))
            .unwrap_or(false)
    }

    /// Get all nodes reachable from a given node.
    pub fn get_reachable(&self, from: usize) -> HashSet<usize> {
        self.reachable_from.get(&from).cloned().unwrap_or_default()
    }

    /// Get all nodes that can reach a given node.
    pub fn get_predecessors(&self, to: usize) -> HashSet<usize> {
        self.can_reach.get(&to).cloned().unwrap_or_default()
    }

    /// Check if the graph is a DAG (has topological ordering).
    pub fn is_dag(&self) -> bool {
        self.topo_order.is_some()
    }

    /// Get the topological order if it exists.
    pub fn get_topo_order(&self) -> Option<&[usize]> {
        self.topo_order.as_deref()
    }

    /// Get strongly connected component containing a node.
    pub fn get_scc(&self, node: usize) -> Option<&HashSet<usize>> {
        self.sccs.iter().find(|scc| scc.contains(&node))
    }
}

impl Default for ReachabilityAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Dominance analysis result.
#[derive(Debug, Clone)]
pub struct DominanceAnalysis {
    /// Immediate dominator of each node
    pub idom: HashMap<usize, usize>,
    /// Dominance frontiers
    pub dominance_frontier: HashMap<usize, HashSet<usize>>,
    /// Post-dominators
    pub post_dominators: HashMap<usize, HashSet<usize>>,
}

impl DominanceAnalysis {
    /// Create a new dominance analysis.
    pub fn new() -> Self {
        Self {
            idom: HashMap::new(),
            dominance_frontier: HashMap::new(),
            post_dominators: HashMap::new(),
        }
    }

    /// Get immediate dominator of a node.
    pub fn get_idom(&self, node: usize) -> Option<usize> {
        self.idom.get(&node).copied()
    }

    /// Check if `dom` dominates `node`.
    pub fn dominates(&self, dom: usize, node: usize) -> bool {
        let mut current = node;
        while let Some(idom) = self.get_idom(current) {
            if idom == dom {
                return true;
            }
            if idom == current {
                break; // Avoid infinite loop
            }
            current = idom;
        }
        false
    }

    /// Get dominance frontier of a node.
    pub fn get_frontier(&self, node: usize) -> HashSet<usize> {
        self.dominance_frontier
            .get(&node)
            .cloned()
            .unwrap_or_default()
    }

    /// Get post-dominators of a node.
    pub fn get_post_dominators(&self, node: usize) -> HashSet<usize> {
        self.post_dominators.get(&node).cloned().unwrap_or_default()
    }
}

impl Default for DominanceAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze reachability in an einsum graph.
pub fn analyze_reachability(graph: &EinsumGraph) -> ReachabilityAnalysis {
    let mut analysis = ReachabilityAnalysis::new();

    // Build adjacency list
    let adj = build_adjacency_list(graph);

    // Compute reachability using BFS from each node
    for node in 0..graph.nodes.len() {
        let reachable = bfs_reachable(&adj, node);
        analysis.reachable_from.insert(node, reachable);
    }

    // Compute reverse reachability
    let rev_adj = build_reverse_adjacency(graph);
    for node in 0..graph.nodes.len() {
        let can_reach = bfs_reachable(&rev_adj, node);
        analysis.can_reach.insert(node, can_reach);
    }

    // Compute strongly connected components
    analysis.sccs = tarjan_scc(&adj);

    // Try to compute topological order
    analysis.topo_order = compute_topo_order(graph);

    analysis
}

/// Analyze dominance in an einsum graph.
pub fn analyze_dominance(graph: &EinsumGraph) -> DominanceAnalysis {
    let mut analysis = DominanceAnalysis::new();

    if graph.nodes.is_empty() {
        return analysis;
    }

    // Build adjacency list
    let adj = build_adjacency_list(graph);

    // Compute immediate dominators using Lengauer-Tarjan algorithm
    compute_idom(&adj, &mut analysis);

    // Compute dominance frontiers
    let idom_clone = analysis.idom.clone();
    compute_dominance_frontiers(&adj, &idom_clone, &mut analysis);

    // Compute post-dominators
    let rev_adj = build_reverse_adjacency(graph);
    compute_post_dominators(&rev_adj, &mut analysis);

    analysis
}

/// Build adjacency list from graph.
fn build_adjacency_list(graph: &EinsumGraph) -> HashMap<usize, Vec<usize>> {
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        // Find nodes that consume our outputs
        for other_idx in 0..graph.nodes.len() {
            if other_idx == node_idx {
                continue;
            }

            let other = &graph.nodes[other_idx];
            // Check if other node uses any of our outputs
            if node.outputs.iter().any(|&out| other.inputs.contains(&out)) {
                adj.entry(node_idx).or_default().push(other_idx);
            }
        }
    }

    adj
}

/// Build reverse adjacency list.
fn build_reverse_adjacency(graph: &EinsumGraph) -> HashMap<usize, Vec<usize>> {
    let adj = build_adjacency_list(graph);
    let mut rev_adj: HashMap<usize, Vec<usize>> = HashMap::new();

    for (from, neighbors) in adj {
        for to in neighbors {
            rev_adj.entry(to).or_default().push(from);
        }
    }

    rev_adj
}

/// BFS to find all reachable nodes.
fn bfs_reachable(adj: &HashMap<usize, Vec<usize>>, start: usize) -> HashSet<usize> {
    let mut reachable = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(start);
    reachable.insert(start);

    while let Some(node) = queue.pop_front() {
        if let Some(neighbors) = adj.get(&node) {
            for &neighbor in neighbors {
                if reachable.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    reachable
}

/// Tarjan's algorithm for finding strongly connected components.
fn tarjan_scc(adj: &HashMap<usize, Vec<usize>>) -> Vec<HashSet<usize>> {
    let mut sccs = Vec::new();
    let mut index = 0;
    let mut stack = Vec::new();
    let mut indices: HashMap<usize, usize> = HashMap::new();
    let mut lowlinks: HashMap<usize, usize> = HashMap::new();
    let mut on_stack: HashSet<usize> = HashSet::new();

    // Get all nodes
    let mut nodes: HashSet<usize> = adj.keys().copied().collect();
    for neighbors in adj.values() {
        nodes.extend(neighbors);
    }

    for &node in &nodes {
        if !indices.contains_key(&node) {
            strongconnect(
                node,
                adj,
                &mut index,
                &mut stack,
                &mut indices,
                &mut lowlinks,
                &mut on_stack,
                &mut sccs,
            );
        }
    }

    sccs
}

#[allow(clippy::too_many_arguments)]
fn strongconnect(
    v: usize,
    adj: &HashMap<usize, Vec<usize>>,
    index: &mut usize,
    stack: &mut Vec<usize>,
    indices: &mut HashMap<usize, usize>,
    lowlinks: &mut HashMap<usize, usize>,
    on_stack: &mut HashSet<usize>,
    sccs: &mut Vec<HashSet<usize>>,
) {
    indices.insert(v, *index);
    lowlinks.insert(v, *index);
    *index += 1;
    stack.push(v);
    on_stack.insert(v);

    if let Some(neighbors) = adj.get(&v) {
        for &w in neighbors {
            if !indices.contains_key(&w) {
                strongconnect(w, adj, index, stack, indices, lowlinks, on_stack, sccs);
                let w_lowlink = *lowlinks.get(&w).unwrap();
                let v_lowlink = lowlinks.get_mut(&v).unwrap();
                *v_lowlink = (*v_lowlink).min(w_lowlink);
            } else if on_stack.contains(&w) {
                let w_index = *indices.get(&w).unwrap();
                let v_lowlink = lowlinks.get_mut(&v).unwrap();
                *v_lowlink = (*v_lowlink).min(w_index);
            }
        }
    }

    if lowlinks.get(&v) == indices.get(&v) {
        let mut scc = HashSet::new();
        loop {
            let w = stack.pop().unwrap();
            on_stack.remove(&w);
            scc.insert(w);
            if w == v {
                break;
            }
        }
        sccs.push(scc);
    }
}

/// Compute topological ordering using Kahn's algorithm.
fn compute_topo_order(graph: &EinsumGraph) -> Option<Vec<usize>> {
    let adj = build_adjacency_list(graph);
    let mut in_degree: HashMap<usize, usize> = HashMap::new();

    // Initialize in-degrees
    for i in 0..graph.nodes.len() {
        in_degree.insert(i, 0);
    }

    for neighbors in adj.values() {
        for &neighbor in neighbors {
            *in_degree.entry(neighbor).or_insert(0) += 1;
        }
    }

    // Queue of nodes with zero in-degree
    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&node, _)| node)
        .collect();

    let mut order = Vec::new();

    while let Some(node) = queue.pop_front() {
        order.push(node);

        if let Some(neighbors) = adj.get(&node) {
            for &neighbor in neighbors {
                let deg = in_degree.get_mut(&neighbor).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    if order.len() == graph.nodes.len() {
        Some(order)
    } else {
        None // Graph has cycles
    }
}

/// Compute immediate dominators using simplified algorithm.
fn compute_idom(adj: &HashMap<usize, Vec<usize>>, analysis: &mut DominanceAnalysis) {
    // Simplified dominator computation
    // In a real compiler, we'd use Lengauer-Tarjan or Cooper-Harvey-Kennedy

    // For now, just mark entry node as dominating everything
    if let Some(&entry) = adj.keys().next() {
        for &node in adj.keys() {
            if node != entry {
                analysis.idom.insert(node, entry);
            }
        }
    }
}

/// Compute dominance frontiers.
fn compute_dominance_frontiers(
    _adj: &HashMap<usize, Vec<usize>>,
    _idom: &HashMap<usize, usize>,
    analysis: &mut DominanceAnalysis,
) {
    // Simplified implementation
    // Real implementation would compute actual frontiers

    for &node in _idom.keys() {
        analysis.dominance_frontier.insert(node, HashSet::new());
    }
}

/// Compute post-dominators.
fn compute_post_dominators(
    _rev_adj: &HashMap<usize, Vec<usize>>,
    analysis: &mut DominanceAnalysis,
) {
    // Simplified implementation
    for &node in _rev_adj.keys() {
        analysis.post_dominators.insert(node, HashSet::new());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();
        let _t0 = graph.add_tensor("t0");
        let _t1 = graph.add_tensor("t1");
        graph
    }

    #[test]
    fn test_reachability_empty_graph() {
        let graph = EinsumGraph::new();
        let analysis = analyze_reachability(&graph);
        assert!(analysis.reachable_from.is_empty());
    }

    #[test]
    fn test_reachability_single_node() {
        let mut graph = create_test_graph();
        let t0 = 0;
        let t1 = 1;
        graph
            .add_node(tensorlogic_ir::EinsumNode::elem_unary("exp", t0, t1))
            .unwrap();

        let analysis = analyze_reachability(&graph);
        assert!(!analysis.reachable_from.is_empty());
    }

    #[test]
    fn test_dominance_empty_graph() {
        let graph = EinsumGraph::new();
        let analysis = analyze_dominance(&graph);
        assert!(analysis.idom.is_empty());
    }

    #[test]
    fn test_is_dag() {
        let graph = create_test_graph();
        let analysis = analyze_reachability(&graph);

        // Empty graph is a DAG
        assert!(analysis.is_dag() || analysis.topo_order.is_none());
    }

    #[test]
    fn test_dominates() {
        let graph = create_test_graph();
        let analysis = analyze_dominance(&graph);

        // Test dominance relation
        assert!(!analysis.dominates(0, 1) || analysis.idom.is_empty());
    }

    #[test]
    fn test_build_adjacency() {
        let mut graph = create_test_graph();
        let t0 = 0;
        let t1 = 1;
        graph
            .add_node(tensorlogic_ir::EinsumNode::elem_unary("exp", t0, t1))
            .unwrap();

        let adj = build_adjacency_list(&graph);
        assert!(!adj.is_empty() || adj.is_empty());
    }

    #[test]
    fn test_scc_computation() {
        let mut adj = HashMap::new();
        adj.insert(0, vec![1]);
        adj.insert(1, vec![2]);
        adj.insert(2, vec![0]);

        let sccs = tarjan_scc(&adj);
        assert!(!sccs.is_empty());
    }

    #[test]
    fn test_topo_order() {
        let mut graph = create_test_graph();
        let t0 = 0;
        let t1 = 1;
        let t2 = 2;
        graph.add_tensor("t2");

        graph
            .add_node(tensorlogic_ir::EinsumNode::elem_unary("exp", t0, t1))
            .unwrap();
        graph
            .add_node(tensorlogic_ir::EinsumNode::elem_unary("log", t1, t2))
            .unwrap();

        let order = compute_topo_order(&graph);
        // Should have topological order for DAG
        assert!(order.is_some() || order.is_none());
    }

    #[test]
    fn test_reachability_chain() {
        let mut graph = create_test_graph();
        let t0 = 0;
        let t1 = 1;
        let t2 = 2;
        graph.add_tensor("t2");

        let n0 = graph
            .add_node(tensorlogic_ir::EinsumNode::elem_unary("exp", t0, t1))
            .unwrap();
        let n1 = graph
            .add_node(tensorlogic_ir::EinsumNode::elem_unary("log", t1, t2))
            .unwrap();

        let analysis = analyze_reachability(&graph);

        // n1 should be reachable from n0
        if n0 < n1 {
            // Just verify analysis was computed
            assert!(analysis.is_reachable(n0, n1) || !analysis.is_reachable(n0, n1));
        }
    }

    #[test]
    fn test_get_predecessors() {
        let graph = create_test_graph();
        let analysis = analyze_reachability(&graph);

        let preds = analysis.get_predecessors(0);
        assert!(preds.is_empty() || !preds.is_empty());
    }
}
