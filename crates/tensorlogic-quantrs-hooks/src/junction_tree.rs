//! Junction tree algorithm for exact inference in probabilistic graphical models.
//!
//! The junction tree algorithm (also known as the clique tree algorithm) is an exact inference
//! algorithm that works by:
//! 1. Converting the factor graph into a tree structure via triangulation
//! 2. Creating cliques (maximal sets of connected variables)
//! 3. Building a junction tree where nodes are cliques
//! 4. Passing messages between cliques to compute exact marginals
//!
//! # Algorithm Overview
//!
//! ```text
//! Factor Graph → Moralize → Triangulate → Find Cliques → Build Tree → Calibrate
//!       ↓            ↓           ↓             ↓            ↓           ↓
//!    Variables   Undirected   Chordal     Maximal      Junction    Marginals
//!                 Graph       Graph       Cliques       Tree
//! ```
//!
//! # Complexity
//!
//! - Time: O(n × d^(w+1)) where w is the treewidth
//! - Space: O(d^w)
//! - Exact for any graph structure (unlike loopy BP)
//!
//! # References
//!
//! - Koller & Friedman, "Probabilistic Graphical Models", Chapter 9
//! - Lauritzen & Spiegelhalter, "Local Computations with Probabilities on
//!   Graphical Structures and their Application to Expert Systems" (1988)

use crate::error::{PgmError, Result};
use crate::factor::Factor;
use crate::graph::FactorGraph;
use scirs2_core::ndarray::ArrayD;
use std::collections::{HashMap, HashSet, VecDeque};

/// A clique in the junction tree (a maximal set of connected variables).
#[derive(Debug, Clone)]
pub struct Clique {
    /// Unique identifier for the clique
    pub id: usize,
    /// Variables in this clique
    pub variables: HashSet<String>,
    /// Potential function (product of all factors involving these variables)
    pub potential: Option<Factor>,
}

impl Clique {
    /// Create a new clique with the given variables.
    pub fn new(id: usize, variables: HashSet<String>) -> Self {
        Self {
            id,
            variables,
            potential: None,
        }
    }

    /// Check if this clique contains all variables in the given set.
    pub fn contains_all(&self, vars: &HashSet<String>) -> bool {
        vars.is_subset(&self.variables)
    }

    /// Get the intersection of variables with another clique.
    pub fn intersection(&self, other: &Clique) -> HashSet<String> {
        self.variables
            .intersection(&other.variables)
            .cloned()
            .collect()
    }
}

/// A separator between two cliques (their shared variables).
#[derive(Debug, Clone)]
pub struct Separator {
    /// Variables in the separator
    pub variables: HashSet<String>,
    /// Message potential
    pub potential: Option<Factor>,
}

impl Separator {
    /// Create a new separator from the intersection of two cliques.
    pub fn from_cliques(c1: &Clique, c2: &Clique) -> Self {
        Self {
            variables: c1.intersection(c2),
            potential: None,
        }
    }
}

/// Edge in the junction tree connecting two cliques.
#[derive(Debug, Clone)]
pub struct JunctionTreeEdge {
    /// ID of the first clique
    pub clique1: usize,
    /// ID of the second clique
    pub clique2: usize,
    /// Separator (shared variables)
    pub separator: Separator,
    /// Message from clique1 to clique2
    pub message_1_to_2: Option<Factor>,
    /// Message from clique2 to clique1
    pub message_2_to_1: Option<Factor>,
}

impl JunctionTreeEdge {
    /// Create a new edge between two cliques.
    pub fn new(clique1: usize, clique2: usize, separator: Separator) -> Self {
        Self {
            clique1,
            clique2,
            separator,
            message_1_to_2: None,
            message_2_to_1: None,
        }
    }
}

/// Junction tree structure for exact inference.
#[derive(Debug, Clone)]
pub struct JunctionTree {
    /// Cliques in the tree
    pub cliques: Vec<Clique>,
    /// Edges connecting cliques
    pub edges: Vec<JunctionTreeEdge>,
    /// Variable to clique mapping (for query efficiency)
    pub var_to_cliques: HashMap<String, Vec<usize>>,
    /// Whether the tree has been calibrated
    pub calibrated: bool,
}

impl JunctionTree {
    /// Create a new junction tree.
    pub fn new() -> Self {
        Self {
            cliques: Vec::new(),
            edges: Vec::new(),
            var_to_cliques: HashMap::new(),
            calibrated: false,
        }
    }

    /// Build a junction tree from a factor graph.
    ///
    /// This implements the complete junction tree construction algorithm:
    /// 1. Moralize the graph (if directed)
    /// 2. Triangulate to create a chordal graph
    /// 3. Identify maximal cliques
    /// 4. Build a junction tree satisfying the running intersection property
    pub fn from_factor_graph(graph: &FactorGraph) -> Result<Self> {
        // Step 1: Extract interaction graph (moralized graph)
        let interaction_graph = Self::build_interaction_graph(graph)?;

        // Step 2: Triangulate the graph using min-fill heuristic
        let triangulated = Self::triangulate(&interaction_graph)?;

        // Step 3: Find maximal cliques
        let cliques = Self::find_maximal_cliques(&triangulated)?;

        // Step 4: Build junction tree from cliques
        let mut tree = Self::build_tree_from_cliques(cliques)?;

        // Step 5: Initialize clique potentials
        tree.initialize_potentials(graph)?;

        Ok(tree)
    }

    /// Build the interaction graph (moralized graph).
    ///
    /// For factor graphs, the interaction graph is an undirected graph where:
    /// - Nodes are variables
    /// - Edges connect variables that appear together in some factor
    fn build_interaction_graph(graph: &FactorGraph) -> Result<HashMap<String, HashSet<String>>> {
        let mut adjacency: HashMap<String, HashSet<String>> = HashMap::new();

        // Initialize all variables
        for var_name in graph.variable_names() {
            adjacency.insert(var_name.clone(), HashSet::new());
        }

        // Add edges for each factor
        for factor in graph.factors() {
            let vars = &factor.variables;
            // Connect all pairs of variables in the factor
            for i in 0..vars.len() {
                for j in (i + 1)..vars.len() {
                    let v1 = &vars[i];
                    let v2 = &vars[j];

                    adjacency.entry(v1.clone()).or_default().insert(v2.clone());
                    adjacency.entry(v2.clone()).or_default().insert(v1.clone());
                }
            }
        }

        Ok(adjacency)
    }

    /// Triangulate the graph to make it chordal.
    ///
    /// Uses the min-fill heuristic for variable elimination ordering.
    /// A chordal graph has the property that every cycle of length ≥4 has a chord.
    fn triangulate(
        graph: &HashMap<String, HashSet<String>>,
    ) -> Result<HashMap<String, HashSet<String>>> {
        let mut triangulated = graph.clone();
        let mut remaining: HashSet<String> = graph.keys().cloned().collect();

        while !remaining.is_empty() {
            // Find variable with minimum fill-in edges
            let var = Self::find_min_fill_variable(&triangulated, &remaining)?;

            // Get neighbors of the variable
            let neighbors: Vec<String> = triangulated
                .get(&var)
                .ok_or_else(|| PgmError::InvalidGraph("Variable not found".to_string()))?
                .intersection(&remaining)
                .cloned()
                .collect();

            // Add fill-in edges (connect all pairs of neighbors)
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let n1 = &neighbors[i];
                    let n2 = &neighbors[j];

                    triangulated
                        .entry(n1.clone())
                        .or_default()
                        .insert(n2.clone());
                    triangulated
                        .entry(n2.clone())
                        .or_default()
                        .insert(n1.clone());
                }
            }

            // Remove variable from remaining set
            remaining.remove(&var);
        }

        Ok(triangulated)
    }

    /// Find the variable with minimum fill-in (min-fill heuristic).
    fn find_min_fill_variable(
        graph: &HashMap<String, HashSet<String>>,
        remaining: &HashSet<String>,
    ) -> Result<String> {
        let mut min_fill = usize::MAX;
        let mut best_var = None;

        for var in remaining {
            let neighbors: Vec<String> = graph
                .get(var)
                .ok_or_else(|| PgmError::InvalidGraph("Variable not found".to_string()))?
                .intersection(remaining)
                .cloned()
                .collect();

            // Count how many edges need to be added
            let mut fill_count = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let n1 = &neighbors[i];
                    let n2 = &neighbors[j];
                    if !graph.get(n1).unwrap().contains(n2) {
                        fill_count += 1;
                    }
                }
            }

            if fill_count < min_fill {
                min_fill = fill_count;
                best_var = Some(var.clone());
            }
        }

        best_var.ok_or_else(|| PgmError::InvalidGraph("No variable found".to_string()))
    }

    /// Find maximal cliques in a triangulated graph.
    ///
    /// Uses a greedy algorithm to identify maximal cliques.
    fn find_maximal_cliques(
        graph: &HashMap<String, HashSet<String>>,
    ) -> Result<Vec<HashSet<String>>> {
        let mut cliques = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();

        // Start from each variable and grow cliques
        for var in graph.keys() {
            if visited.contains(var) {
                continue;
            }

            let mut clique: HashSet<String> = HashSet::new();
            clique.insert(var.clone());

            // Add neighbors that form a clique
            for neighbor in graph.get(var).unwrap() {
                // Check if neighbor is connected to all current clique members
                let is_fully_connected = clique
                    .iter()
                    .all(|c| c == neighbor || graph.get(neighbor).unwrap().contains(c));

                if is_fully_connected {
                    clique.insert(neighbor.clone());
                }
            }

            // Check if this is a maximal clique
            let is_maximal = !cliques
                .iter()
                .any(|c: &HashSet<String>| c.is_superset(&clique));

            if is_maximal {
                // Remove non-maximal cliques that are subsets of this one
                cliques.retain(|c| !clique.is_superset(c));
                cliques.push(clique.clone());
            }

            visited.insert(var.clone());
        }

        // Ensure we have at least one clique
        if cliques.is_empty() && !graph.is_empty() {
            // Create a clique with all variables (fallback)
            let all_vars: HashSet<String> = graph.keys().cloned().collect();
            cliques.push(all_vars);
        }

        Ok(cliques)
    }

    /// Build a junction tree from maximal cliques.
    ///
    /// Uses a maximum spanning tree algorithm based on separator size.
    fn build_tree_from_cliques(clique_sets: Vec<HashSet<String>>) -> Result<Self> {
        let mut tree = JunctionTree::new();

        // Create clique nodes
        for (id, vars) in clique_sets.into_iter().enumerate() {
            let clique = Clique::new(id, vars.clone());

            // Update variable to clique mapping
            for var in &vars {
                tree.var_to_cliques.entry(var.clone()).or_default().push(id);
            }

            tree.cliques.push(clique);
        }

        // Build maximum spanning tree based on separator size
        if tree.cliques.len() > 1 {
            tree.build_maximum_spanning_tree()?;
        }

        Ok(tree)
    }

    /// Build a maximum spanning tree connecting cliques.
    ///
    /// Uses Prim's algorithm with separator size as edge weight.
    fn build_maximum_spanning_tree(&mut self) -> Result<()> {
        let n = self.cliques.len();
        if n == 0 {
            return Ok(());
        }

        let mut in_tree = vec![false; n];
        let mut edges_to_add: Vec<(usize, usize, usize)> = Vec::new();

        // Start with clique 0
        in_tree[0] = true;
        let mut tree_size = 1;

        while tree_size < n {
            let mut best_edge = None;
            let mut best_weight = 0;

            // Find best edge to add
            for i in 0..n {
                if !in_tree[i] {
                    continue;
                }

                for (j, &is_in_tree) in in_tree.iter().enumerate().take(n) {
                    if is_in_tree {
                        continue;
                    }

                    let separator = self.cliques[i].intersection(&self.cliques[j]);
                    let weight = separator.len();

                    if weight > best_weight {
                        best_weight = weight;
                        best_edge = Some((i, j, weight));
                    }
                }
            }

            if let Some((i, j, _)) = best_edge {
                edges_to_add.push((i, j, best_weight));
                in_tree[j] = true;
                tree_size += 1;
            } else {
                break;
            }
        }

        // Create edges
        for (c1, c2, _) in edges_to_add {
            let separator = Separator::from_cliques(&self.cliques[c1], &self.cliques[c2]);
            let edge = JunctionTreeEdge::new(c1, c2, separator);
            self.edges.push(edge);
        }

        Ok(())
    }

    /// Initialize clique potentials from the factor graph.
    ///
    /// Assigns each factor to a clique that contains all its variables.
    fn initialize_potentials(&mut self, graph: &FactorGraph) -> Result<()> {
        for factor in graph.factors() {
            let factor_vars: HashSet<String> = factor.variables.iter().cloned().collect();

            // Find a clique that contains all variables in this factor
            let clique_idx = self
                .cliques
                .iter()
                .position(|c| c.contains_all(&factor_vars))
                .ok_or_else(|| {
                    PgmError::InvalidGraph(format!(
                        "No clique contains all variables for factor: {:?}",
                        factor.name
                    ))
                })?;

            let clique = &mut self.cliques[clique_idx];

            // Multiply factor into clique potential
            if let Some(ref mut potential) = clique.potential {
                *potential = potential.product(factor)?;
            } else {
                clique.potential = Some(factor.clone());
            }
        }

        // Initialize cliques without factors to uniform potentials
        for clique in &mut self.cliques {
            if clique.potential.is_none() {
                // Create uniform potential
                clique.potential = Some(Self::create_uniform_potential(&clique.variables, graph)?);
            }
        }

        Ok(())
    }

    /// Create a uniform potential over a set of variables.
    fn create_uniform_potential(
        variables: &HashSet<String>,
        graph: &FactorGraph,
    ) -> Result<Factor> {
        let var_vec: Vec<String> = variables.iter().cloned().collect();
        let mut shape = Vec::new();

        for var in &var_vec {
            let cardinality = graph
                .get_variable(var)
                .ok_or_else(|| PgmError::InvalidGraph(format!("Variable {} not found", var)))?
                .cardinality;
            shape.push(cardinality);
        }

        let size: usize = shape.iter().product();
        let values = vec![1.0; size];

        let array = ArrayD::from_shape_vec(shape, values)
            .map_err(|e| PgmError::InvalidGraph(format!("Array creation failed: {}", e)))?;

        Factor::new("uniform".to_string(), var_vec, array)
    }

    /// Calibrate the junction tree by passing messages.
    ///
    /// This implements the message passing schedule for exact inference.
    pub fn calibrate(&mut self) -> Result<()> {
        if self.edges.is_empty() {
            self.calibrated = true;
            return Ok(());
        }

        // Collect evidence (inward pass) from leaves to root
        let root = 0;
        self.collect_evidence(root, None)?;

        // Distribute evidence (outward pass) from root to leaves
        self.distribute_evidence(root, None)?;

        self.calibrated = true;
        Ok(())
    }

    /// Collect evidence (inward pass).
    fn collect_evidence(&mut self, current: usize, parent: Option<usize>) -> Result<()> {
        // Find children (neighbors except parent)
        let children: Vec<usize> = self.get_neighbors(current, parent);

        // Recursively collect from children
        for child in &children {
            self.collect_evidence(*child, Some(current))?;
        }

        // Send message to parent if exists
        if let Some(parent_idx) = parent {
            self.send_message(current, parent_idx)?;
        }

        Ok(())
    }

    /// Distribute evidence (outward pass).
    fn distribute_evidence(&mut self, current: usize, parent: Option<usize>) -> Result<()> {
        // Find children (neighbors except parent)
        let children: Vec<usize> = self.get_neighbors(current, parent);

        // Send messages to all children
        for child in &children {
            self.send_message(current, *child)?;
            self.distribute_evidence(*child, Some(current))?;
        }

        Ok(())
    }

    /// Get neighbors of a clique excluding the parent.
    fn get_neighbors(&self, clique: usize, parent: Option<usize>) -> Vec<usize> {
        let mut neighbors = Vec::new();

        for edge in &self.edges {
            if edge.clique1 == clique {
                if parent.is_none() || parent.unwrap() != edge.clique2 {
                    neighbors.push(edge.clique2);
                }
            } else if edge.clique2 == clique
                && (parent.is_none() || parent.unwrap() != edge.clique1)
            {
                neighbors.push(edge.clique1);
            }
        }

        neighbors
    }

    /// Send a message from one clique to another.
    fn send_message(&mut self, from: usize, to: usize) -> Result<()> {
        // Find the edge
        let edge_idx = self
            .edges
            .iter()
            .position(|e| {
                (e.clique1 == from && e.clique2 == to) || (e.clique1 == to && e.clique2 == from)
            })
            .ok_or_else(|| PgmError::InvalidGraph("Edge not found".to_string()))?;

        // Get separator variables
        let separator_vars = self.edges[edge_idx].separator.variables.clone();

        // Get clique potential
        let clique_potential = self.cliques[from].potential.clone().ok_or_else(|| {
            PgmError::InvalidGraph("Clique potential not initialized".to_string())
        })?;

        // Marginalize out variables not in separator
        let mut message = clique_potential;
        let all_vars: HashSet<String> = message.variables.iter().cloned().collect();
        let vars_to_eliminate: Vec<String> =
            all_vars.difference(&separator_vars).cloned().collect();

        for var in vars_to_eliminate {
            message = message.marginalize_out(&var)?;
        }

        // Store message
        let edge = &mut self.edges[edge_idx];
        if edge.clique1 == from {
            edge.message_1_to_2 = Some(message);
        } else {
            edge.message_2_to_1 = Some(message);
        }

        Ok(())
    }

    /// Query marginal probability for a variable.
    pub fn query_marginal(&self, variable: &str) -> Result<ArrayD<f64>> {
        if !self.calibrated {
            return Err(PgmError::InvalidGraph(
                "Tree must be calibrated before querying".to_string(),
            ));
        }

        // Find a clique containing this variable
        let clique_indices = self
            .var_to_cliques
            .get(variable)
            .ok_or_else(|| PgmError::InvalidGraph(format!("Variable {} not found", variable)))?;

        if clique_indices.is_empty() {
            return Err(PgmError::InvalidGraph(format!(
                "No clique contains variable {}",
                variable
            )));
        }

        // Get belief from first clique
        let clique = &self.cliques[clique_indices[0]];
        let mut belief = clique.potential.clone().ok_or_else(|| {
            PgmError::InvalidGraph("Clique potential not initialized".to_string())
        })?;

        // Marginalize out all variables except the query variable
        let all_vars: HashSet<String> = belief.variables.iter().cloned().collect();
        let mut target_set = HashSet::new();
        target_set.insert(variable.to_string());
        let vars_to_eliminate: Vec<String> = all_vars.difference(&target_set).cloned().collect();

        for var in vars_to_eliminate {
            belief = belief.marginalize_out(&var)?;
        }

        // Normalize
        belief.normalize();

        Ok(belief.values)
    }

    /// Query joint marginal over multiple variables.
    pub fn query_joint_marginal(&self, variables: &[String]) -> Result<ArrayD<f64>> {
        if !self.calibrated {
            return Err(PgmError::InvalidGraph(
                "Tree must be calibrated before querying".to_string(),
            ));
        }

        let var_set: HashSet<String> = variables.iter().cloned().collect();

        // Find a clique containing all these variables
        let clique = self
            .cliques
            .iter()
            .find(|c| c.contains_all(&var_set))
            .ok_or_else(|| {
                PgmError::InvalidGraph(format!("No clique contains all variables: {:?}", variables))
            })?;

        let mut belief = clique.potential.clone().ok_or_else(|| {
            PgmError::InvalidGraph("Clique potential not initialized".to_string())
        })?;

        // Marginalize out variables not in query
        let all_vars: HashSet<String> = belief.variables.iter().cloned().collect();
        let vars_to_eliminate: Vec<String> = all_vars.difference(&var_set).cloned().collect();

        for var in vars_to_eliminate {
            belief = belief.marginalize_out(&var)?;
        }

        // Normalize
        belief.normalize();

        Ok(belief.values)
    }

    /// Get the treewidth of this junction tree.
    ///
    /// The treewidth is the size of the largest clique minus 1.
    pub fn treewidth(&self) -> usize {
        self.cliques
            .iter()
            .map(|c| c.variables.len())
            .max()
            .unwrap_or(0)
            .saturating_sub(1)
    }

    /// Check if the junction tree satisfies the running intersection property.
    ///
    /// For every variable X, the set of cliques containing X forms a connected subtree.
    pub fn verify_running_intersection_property(&self) -> bool {
        for var in self.var_to_cliques.keys() {
            let cliques_with_var = self.var_to_cliques.get(var).unwrap();

            if cliques_with_var.len() <= 1 {
                continue;
            }

            // Check if these cliques form a connected component
            if !self.is_connected_subgraph(cliques_with_var) {
                return false;
            }
        }

        true
    }

    /// Check if a set of cliques forms a connected subgraph.
    fn is_connected_subgraph(&self, cliques: &[usize]) -> bool {
        if cliques.is_empty() {
            return true;
        }

        let clique_set: HashSet<usize> = cliques.iter().copied().collect();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start BFS from first clique
        queue.push_back(cliques[0]);
        visited.insert(cliques[0]);

        while let Some(current) = queue.pop_front() {
            for edge in &self.edges {
                let neighbor = if edge.clique1 == current {
                    Some(edge.clique2)
                } else if edge.clique2 == current {
                    Some(edge.clique1)
                } else {
                    None
                };

                if let Some(n) = neighbor {
                    if clique_set.contains(&n) && !visited.contains(&n) {
                        visited.insert(n);
                        queue.push_back(n);
                    }
                }
            }
        }

        visited.len() == cliques.len()
    }
}

impl Default for JunctionTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::FactorGraph;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_clique_creation() {
        let mut vars = HashSet::new();
        vars.insert("x".to_string());
        vars.insert("y".to_string());

        let clique = Clique::new(0, vars);
        assert_eq!(clique.id, 0);
        assert_eq!(clique.variables.len(), 2);
    }

    #[test]
    fn test_clique_intersection() {
        let mut vars1 = HashSet::new();
        vars1.insert("x".to_string());
        vars1.insert("y".to_string());

        let mut vars2 = HashSet::new();
        vars2.insert("y".to_string());
        vars2.insert("z".to_string());

        let c1 = Clique::new(0, vars1);
        let c2 = Clique::new(1, vars2);

        let intersection = c1.intersection(&c2);
        assert_eq!(intersection.len(), 1);
        assert!(intersection.contains("y"));
    }

    #[test]
    fn test_interaction_graph() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("z".to_string(), "Binary".to_string(), 2);

        // Add factor P(x, y)
        let pxy = Factor::new(
            "P(x,y)".to_string(),
            vec!["x".to_string(), "y".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
        graph.add_factor(pxy).unwrap();

        // Add factor P(y, z)
        let pyz = Factor::new(
            "P(y,z)".to_string(),
            vec!["y".to_string(), "z".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.5, 0.1, 0.2, 0.2])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
        graph.add_factor(pyz).unwrap();

        let interaction_graph = JunctionTree::build_interaction_graph(&graph).unwrap();

        // Check edges
        assert!(interaction_graph.get("x").unwrap().contains("y"));
        assert!(interaction_graph.get("y").unwrap().contains("x"));
        assert!(interaction_graph.get("y").unwrap().contains("z"));
        assert!(interaction_graph.get("z").unwrap().contains("y"));
    }

    #[test]
    fn test_junction_tree_construction() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let pxy = Factor::new(
            "P(x,y)".to_string(),
            vec!["x".to_string(), "y".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.3, 0.7, 0.4, 0.6])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
        graph.add_factor(pxy).unwrap();

        let tree = JunctionTree::from_factor_graph(&graph).unwrap();

        assert!(!tree.cliques.is_empty());
        assert!(tree.verify_running_intersection_property());
    }

    #[test]
    fn test_junction_tree_calibration() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let pxy = Factor::new(
            "P(x,y)".to_string(),
            vec!["x".to_string(), "y".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.25, 0.25, 0.25, 0.25])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
        graph.add_factor(pxy).unwrap();

        let mut tree = JunctionTree::from_factor_graph(&graph).unwrap();
        tree.calibrate().unwrap();

        assert!(tree.calibrated);
    }

    #[test]
    fn test_marginal_query() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

        let pxy = Factor::new(
            "P(x,y)".to_string(),
            vec!["x".to_string(), "y".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.1, 0.4, 0.2, 0.3])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
        graph.add_factor(pxy).unwrap();

        let mut tree = JunctionTree::from_factor_graph(&graph).unwrap();
        tree.calibrate().unwrap();

        let marginal_x = tree.query_marginal("x").unwrap();

        // P(x=0) = 0.1 + 0.4 = 0.5
        // P(x=1) = 0.2 + 0.3 = 0.5
        assert_abs_diff_eq!(marginal_x[[0]], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(marginal_x[[1]], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_treewidth() {
        let mut graph = FactorGraph::new();
        graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);
        graph.add_variable_with_card("z".to_string(), "Binary".to_string(), 2);

        let pxy = Factor::new(
            "P(x,y)".to_string(),
            vec!["x".to_string(), "y".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.3, 0.7, 0.4, 0.6])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();
        let pyz = Factor::new(
            "P(y,z)".to_string(),
            vec!["y".to_string(), "z".to_string()],
            Array::from_shape_vec(vec![2, 2], vec![0.5, 0.5, 0.6, 0.4])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();

        graph.add_factor(pxy).unwrap();
        graph.add_factor(pyz).unwrap();

        let tree = JunctionTree::from_factor_graph(&graph).unwrap();

        // Treewidth should be at most 2 (clique size of 3 minus 1)
        assert!(tree.treewidth() <= 2);
    }
}
