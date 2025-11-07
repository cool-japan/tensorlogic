//! Graph kernels for measuring similarity between structured data.
//!
//! These kernels operate on graph representations of logical expressions
//! and measure structural similarity through various graph-theoretic properties.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tensorlogic_ir::TLExpr;

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// Simple graph representation for kernel computation
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    /// Number of nodes
    pub n_nodes: usize,
    /// Edge list (from, to, edge_type)
    pub edges: Vec<(usize, usize, String)>,
    /// Node labels
    pub node_labels: Vec<String>,
}

impl Graph {
    /// Create a new graph
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            edges: Vec::new(),
            node_labels: vec!["node".to_string(); n_nodes],
        }
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, from: usize, to: usize, edge_type: String) {
        if from < self.n_nodes && to < self.n_nodes {
            self.edges.push((from, to, edge_type));
        }
    }

    /// Set node label
    pub fn set_node_label(&mut self, node: usize, label: String) {
        if node < self.n_nodes {
            self.node_labels[node] = label;
        }
    }

    /// Get adjacency list representation
    pub fn adjacency_list(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.n_nodes];
        for &(from, to, _) in &self.edges {
            adj[from].push(to);
        }
        adj
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|(from, _, _)| *from == node)
            .map(|(_, to, _)| *to)
            .collect()
    }

    /// Convert TLExpr to graph representation
    pub fn from_tlexpr(expr: &TLExpr) -> Self {
        let mut graph = Graph::new(0);
        let mut node_id = 0;
        Self::build_graph_recursive(expr, &mut graph, &mut node_id, None);
        graph
    }

    fn build_graph_recursive(
        expr: &TLExpr,
        graph: &mut Graph,
        node_id: &mut usize,
        parent: Option<usize>,
    ) -> usize {
        let current_id = *node_id;
        *node_id += 1;
        graph.n_nodes += 1;

        // Set node label based on expression type
        let label = match expr {
            TLExpr::Pred { name, .. } => format!("pred:{}", name),
            TLExpr::And(_, _) => "and".to_string(),
            TLExpr::Or(_, _) => "or".to_string(),
            TLExpr::Not(_) => "not".to_string(),
            TLExpr::Exists { domain, .. } => format!("exists:{}", domain),
            TLExpr::ForAll { domain, .. } => format!("forall:{}", domain),
            TLExpr::Imply(_, _) => "imply".to_string(),
            _ => "unknown".to_string(),
        };

        graph.node_labels.push(label.clone());

        // Add edge from parent if it exists
        if let Some(parent_id) = parent {
            graph.add_edge(parent_id, current_id, "child".to_string());
        }

        // Recursively process children
        match expr {
            TLExpr::And(left, right) | TLExpr::Or(left, right) | TLExpr::Imply(left, right) => {
                Self::build_graph_recursive(left, graph, node_id, Some(current_id));
                Self::build_graph_recursive(right, graph, node_id, Some(current_id));
            }
            TLExpr::Not(inner) => {
                Self::build_graph_recursive(inner, graph, node_id, Some(current_id));
            }
            TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
                Self::build_graph_recursive(body, graph, node_id, Some(current_id));
            }
            _ => {}
        }

        current_id
    }
}

/// Subgraph matching kernel configuration
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubgraphMatchingConfig {
    /// Maximum subgraph size to consider
    pub max_subgraph_size: usize,
    /// Whether to normalize by graph sizes
    pub normalize: bool,
}

impl SubgraphMatchingConfig {
    /// Create default configuration
    pub fn new() -> Self {
        Self {
            max_subgraph_size: 3,
            normalize: true,
        }
    }

    /// Set maximum subgraph size
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_subgraph_size = size;
        self
    }
}

impl Default for SubgraphMatchingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Subgraph matching kernel
///
/// Measures similarity by counting common subgraphs between two graphs.
pub struct SubgraphMatchingKernel {
    config: SubgraphMatchingConfig,
}

impl SubgraphMatchingKernel {
    /// Create a new subgraph matching kernel
    pub fn new(config: SubgraphMatchingConfig) -> Self {
        Self { config }
    }

    /// Count subgraphs of given size in a graph
    fn count_subgraphs(&self, graph: &Graph, size: usize) -> HashMap<String, usize> {
        let mut subgraph_counts = HashMap::new();

        if size > graph.n_nodes {
            return subgraph_counts;
        }

        // For simplicity, count node label patterns
        // More sophisticated: enumerate all connected subgraphs
        for node in 0..graph.n_nodes {
            let pattern = self.extract_pattern(graph, node, size);
            *subgraph_counts.entry(pattern).or_insert(0) += 1;
        }

        subgraph_counts
    }

    /// Extract local pattern around a node
    fn extract_pattern(&self, graph: &Graph, start: usize, depth: usize) -> String {
        let mut pattern_parts = vec![graph.node_labels[start].clone()];

        if depth > 1 {
            let neighbors = graph.neighbors(start);
            let mut neighbor_labels: Vec<_> = neighbors
                .iter()
                .map(|&n| graph.node_labels[n].clone())
                .collect();
            neighbor_labels.sort();
            pattern_parts.extend(neighbor_labels);
        }

        pattern_parts.join("|")
    }

    /// Compute similarity between two graphs
    pub fn compute_graphs(&self, g1: &Graph, g2: &Graph) -> Result<f64> {
        let mut total_similarity = 0.0;

        for size in 1..=self.config.max_subgraph_size {
            let counts1 = self.count_subgraphs(g1, size);
            let counts2 = self.count_subgraphs(g2, size);

            // Compute intersection
            let mut intersection = 0.0;
            for (pattern, count1) in &counts1 {
                if let Some(count2) = counts2.get(pattern) {
                    intersection += (*count1).min(*count2) as f64;
                }
            }

            total_similarity += intersection;
        }

        if self.config.normalize {
            let max_size = (g1.n_nodes.max(g2.n_nodes)) as f64;
            if max_size > 0.0 {
                total_similarity /= max_size;
            }
        }

        Ok(total_similarity)
    }
}

impl Kernel for SubgraphMatchingKernel {
    fn compute(&self, x: &[f64], _y: &[f64]) -> Result<f64> {
        // For basic kernel trait compatibility, return a placeholder
        // Real usage should use compute_graphs
        Ok(x.iter().sum::<f64>())
    }

    fn name(&self) -> &str {
        "SubgraphMatching"
    }
}

/// Walk-based kernel configuration
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WalkKernelConfig {
    /// Maximum walk length
    pub max_walk_length: usize,
    /// Decay factor for longer walks
    pub decay_factor: f64,
    /// Whether to normalize
    pub normalize: bool,
}

impl WalkKernelConfig {
    /// Create default configuration
    pub fn new() -> Self {
        Self {
            max_walk_length: 4,
            decay_factor: 0.8,
            normalize: true,
        }
    }

    /// Set maximum walk length
    pub fn with_max_length(mut self, length: usize) -> Self {
        self.max_walk_length = length;
        self
    }

    /// Set decay factor
    pub fn with_decay(mut self, decay: f64) -> Self {
        self.decay_factor = decay;
        self
    }
}

impl Default for WalkKernelConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Random walk kernel
///
/// Measures similarity by counting common random walks between graphs.
pub struct RandomWalkKernel {
    config: WalkKernelConfig,
}

impl RandomWalkKernel {
    /// Create a new random walk kernel
    pub fn new(config: WalkKernelConfig) -> Result<Self> {
        if config.decay_factor <= 0.0 || config.decay_factor > 1.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "decay_factor".to_string(),
                value: config.decay_factor.to_string(),
                reason: "must be in (0, 1]".to_string(),
            });
        }

        Ok(Self { config })
    }

    /// Extract walks from a graph
    fn extract_walks(&self, graph: &Graph) -> HashMap<Vec<String>, usize> {
        let mut walk_counts = HashMap::new();
        let adj = graph.adjacency_list();

        for start in 0..graph.n_nodes {
            self.dfs_walks(
                graph,
                &adj,
                start,
                vec![graph.node_labels[start].clone()],
                &mut walk_counts,
            );
        }

        walk_counts
    }

    /// DFS to enumerate walks
    fn dfs_walks(
        &self,
        graph: &Graph,
        adj: &[Vec<usize>],
        current: usize,
        path: Vec<String>,
        walk_counts: &mut HashMap<Vec<String>, usize>,
    ) {
        if path.len() >= self.config.max_walk_length {
            *walk_counts.entry(path).or_insert(0) += 1;
            return;
        }

        // Add current path
        *walk_counts.entry(path.clone()).or_insert(0) += 1;

        // Continue walk
        for &neighbor in &adj[current] {
            let mut new_path = path.clone();
            new_path.push(graph.node_labels[neighbor].clone());
            self.dfs_walks(graph, adj, neighbor, new_path, walk_counts);
        }
    }

    /// Compute similarity between two graphs
    pub fn compute_graphs(&self, g1: &Graph, g2: &Graph) -> Result<f64> {
        let walks1 = self.extract_walks(g1);
        let walks2 = self.extract_walks(g2);

        let mut similarity = 0.0;

        for (walk, count1) in &walks1 {
            if let Some(count2) = walks2.get(walk) {
                let walk_sim = (*count1).min(*count2) as f64;
                let decay = self.config.decay_factor.powi(walk.len() as i32);
                similarity += walk_sim * decay;
            }
        }

        if self.config.normalize {
            let total1: usize = walks1.values().sum();
            let total2: usize = walks2.values().sum();
            let normalizer = ((total1 * total2) as f64).sqrt();
            if normalizer > 0.0 {
                similarity /= normalizer;
            }
        }

        Ok(similarity)
    }
}

impl Kernel for RandomWalkKernel {
    fn compute(&self, x: &[f64], _y: &[f64]) -> Result<f64> {
        // Placeholder for trait compatibility
        Ok(x.iter().sum::<f64>())
    }

    fn name(&self) -> &str {
        "RandomWalk"
    }
}

/// Weisfeiler-Lehman kernel configuration
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WeisfeilerLehmanConfig {
    /// Number of WL iterations
    pub n_iterations: usize,
    /// Whether to normalize
    pub normalize: bool,
}

impl WeisfeilerLehmanConfig {
    /// Create default configuration
    pub fn new() -> Self {
        Self {
            n_iterations: 3,
            normalize: true,
        }
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.n_iterations = iterations;
        self
    }
}

impl Default for WeisfeilerLehmanConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Weisfeiler-Lehman (WL) kernel
///
/// Iteratively refines node labels based on neighborhood structure,
/// then compares label histograms.
pub struct WeisfeilerLehmanKernel {
    config: WeisfeilerLehmanConfig,
}

impl WeisfeilerLehmanKernel {
    /// Create a new WL kernel
    pub fn new(config: WeisfeilerLehmanConfig) -> Self {
        Self { config }
    }

    /// Perform one WL iteration
    fn wl_iteration(&self, graph: &Graph, labels: &[String]) -> Vec<String> {
        let mut new_labels = Vec::with_capacity(graph.n_nodes);
        let adj = graph.adjacency_list();

        for node in 0..graph.n_nodes {
            // Collect neighbor labels
            let mut neighbor_labels: Vec<String> =
                adj[node].iter().map(|&n| labels[n].clone()).collect();

            neighbor_labels.sort();

            // Create new label by concatenating
            let mut new_label = labels[node].clone();
            for neighbor_label in neighbor_labels {
                new_label.push('_');
                new_label.push_str(&neighbor_label);
            }

            new_labels.push(new_label);
        }

        new_labels
    }

    /// Extract label histograms across all iterations
    fn extract_label_histograms(&self, graph: &Graph) -> Vec<HashMap<String, usize>> {
        let mut histograms = Vec::new();
        let mut labels = graph.node_labels.clone();

        for _ in 0..self.config.n_iterations {
            // Count labels
            let mut histogram = HashMap::new();
            for label in &labels {
                *histogram.entry(label.clone()).or_insert(0) += 1;
            }
            histograms.push(histogram);

            // Update labels
            labels = self.wl_iteration(graph, &labels);
        }

        histograms
    }

    /// Compute similarity between two graphs
    pub fn compute_graphs(&self, g1: &Graph, g2: &Graph) -> Result<f64> {
        let hists1 = self.extract_label_histograms(g1);
        let hists2 = self.extract_label_histograms(g2);

        let mut total_similarity = 0.0;

        for (hist1, hist2) in hists1.iter().zip(hists2.iter()) {
            // Compute histogram intersection
            let mut intersection = 0.0;
            for (label, count1) in hist1 {
                if let Some(count2) = hist2.get(label) {
                    intersection += (*count1).min(*count2) as f64;
                }
            }
            total_similarity += intersection;
        }

        if self.config.normalize {
            let size1 = g1.n_nodes as f64;
            let size2 = g2.n_nodes as f64;
            let normalizer = (size1 * size2).sqrt();
            if normalizer > 0.0 {
                total_similarity /= normalizer;
            }
        }

        Ok(total_similarity)
    }
}

impl Kernel for WeisfeilerLehmanKernel {
    fn compute(&self, x: &[f64], _y: &[f64]) -> Result<f64> {
        // Placeholder for trait compatibility
        Ok(x.iter().sum::<f64>())
    }

    fn name(&self) -> &str {
        "WeisfeilerLehman"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1, "edge".to_string());
        graph.add_edge(1, 2, "edge".to_string());
        graph.set_node_label(0, "A".to_string());
        graph.set_node_label(1, "B".to_string());
        graph.set_node_label(2, "C".to_string());

        assert_eq!(graph.n_nodes, 3);
        assert_eq!(graph.edges.len(), 2);
        assert_eq!(graph.node_labels[0], "A");
    }

    #[test]
    fn test_graph_from_tlexpr() {
        let expr = TLExpr::and(TLExpr::pred("p1", vec![]), TLExpr::pred("p2", vec![]));

        let graph = Graph::from_tlexpr(&expr);
        assert!(graph.n_nodes > 0);
        assert!(!graph.node_labels.is_empty());
    }

    #[test]
    fn test_subgraph_matching_kernel() {
        let config = SubgraphMatchingConfig::new().with_max_size(2);
        let kernel = SubgraphMatchingKernel::new(config);

        let mut g1 = Graph::new(3);
        g1.add_edge(0, 1, "edge".to_string());
        g1.add_edge(1, 2, "edge".to_string());

        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1, "edge".to_string());
        g2.add_edge(0, 2, "edge".to_string());

        let sim = kernel.compute_graphs(&g1, &g2).unwrap();
        assert!(sim >= 0.0);
    }

    #[test]
    fn test_random_walk_kernel() {
        let config = WalkKernelConfig::new().with_max_length(3);
        let kernel = RandomWalkKernel::new(config).unwrap();

        let mut g1 = Graph::new(3);
        g1.add_edge(0, 1, "edge".to_string());
        g1.add_edge(1, 2, "edge".to_string());

        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1, "edge".to_string());
        g2.add_edge(1, 2, "edge".to_string());

        let sim = kernel.compute_graphs(&g1, &g2).unwrap();
        assert!(sim > 0.0);
    }

    #[test]
    fn test_random_walk_kernel_invalid_decay() {
        let config = WalkKernelConfig::new().with_decay(1.5);
        let result = RandomWalkKernel::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_weisfeiler_lehman_kernel() {
        let config = WeisfeilerLehmanConfig::new().with_iterations(2);
        let kernel = WeisfeilerLehmanKernel::new(config);

        let mut g1 = Graph::new(4);
        g1.set_node_label(0, "A".to_string());
        g1.set_node_label(1, "B".to_string());
        g1.set_node_label(2, "B".to_string());
        g1.set_node_label(3, "A".to_string());
        g1.add_edge(0, 1, "edge".to_string());
        g1.add_edge(1, 2, "edge".to_string());
        g1.add_edge(2, 3, "edge".to_string());

        let mut g2 = Graph::new(4);
        g2.set_node_label(0, "A".to_string());
        g2.set_node_label(1, "B".to_string());
        g2.set_node_label(2, "B".to_string());
        g2.set_node_label(3, "A".to_string());
        g2.add_edge(0, 1, "edge".to_string());
        g2.add_edge(1, 2, "edge".to_string());
        g2.add_edge(2, 3, "edge".to_string());

        let sim = kernel.compute_graphs(&g1, &g2).unwrap();
        assert!(sim > 0.0);
    }

    #[test]
    fn test_wl_self_similarity() {
        let config = WeisfeilerLehmanConfig::new();
        let kernel = WeisfeilerLehmanKernel::new(config);

        let mut graph = Graph::new(3);
        graph.add_edge(0, 1, "edge".to_string());
        graph.add_edge(1, 2, "edge".to_string());

        let sim = kernel.compute_graphs(&graph, &graph).unwrap();
        assert!(sim > 0.0);
    }

    #[test]
    fn test_graph_neighbors() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1, "edge".to_string());
        graph.add_edge(0, 2, "edge".to_string());

        let neighbors = graph.neighbors(0);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
    }

    #[test]
    fn test_graph_adjacency_list() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1, "edge".to_string());
        graph.add_edge(1, 2, "edge".to_string());

        let adj = graph.adjacency_list();
        assert_eq!(adj.len(), 3);
        assert_eq!(adj[0], vec![1]);
        assert_eq!(adj[1], vec![2]);
        assert_eq!(adj[2], Vec::<usize>::new());
    }
}
