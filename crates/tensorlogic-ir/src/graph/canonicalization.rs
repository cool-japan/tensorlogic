//! Graph canonicalization for deterministic comparison and hashing.
//!
//! This module provides functionality to convert computation graphs into a canonical form,
//! which is useful for:
//! - Graph equality comparison
//! - Duplicate graph detection
//! - Common subexpression elimination
//! - Graph hashing and caching
//!
//! # Algorithm
//!
//! The canonicalization process:
//! 1. Compute topological ordering of tensors
//! 2. Assign canonical names (t0, t1, t2, ...) based on order
//! 3. Sort nodes in execution order
//! 4. Normalize inputs and outputs
//!
//! # Examples
//!
//! ```
//! use tensorlogic_ir::{EinsumGraph, EinsumNode};
//!
//! let mut graph = EinsumGraph::new();
//! let a = graph.add_tensor("foo");
//! let b = graph.add_tensor("bar");
//! let c = graph.add_tensor("baz");
//! graph.add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c])).unwrap();
//! graph.add_output(c).unwrap();
//!
//! let canonical = tensorlogic_ir::canonicalize_graph(&graph).unwrap();
//! // Tensors are renamed to t0, t1, t2
//! assert_eq!(canonical.tensors, vec!["t0", "t1", "t2"]);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use super::{EinsumGraph, EinsumNode};
use crate::error::IrError;

/// Canonicalize a computation graph.
///
/// This function converts a graph into a canonical form where:
/// - Tensors are renamed to t0, t1, t2, ... in topological order
/// - Nodes are sorted in execution order
/// - Inputs and outputs are sorted consistently
///
/// The resulting graph is semantically equivalent to the original but has a
/// normalized structure that facilitates comparison and hashing.
pub fn canonicalize_graph(graph: &EinsumGraph) -> Result<EinsumGraph, IrError> {
    // Empty graph is already canonical
    if graph.is_empty() {
        return Ok(graph.clone());
    }

    // Validate the input graph
    graph.validate()?;

    // Step 1: Compute topological order of tensors
    let tensor_order = topological_sort_tensors(graph)?;

    // Step 2: Create mapping from old indices to new canonical indices
    let mut tensor_mapping = HashMap::new();
    for (new_idx, &old_idx) in tensor_order.iter().enumerate() {
        tensor_mapping.insert(old_idx, new_idx);
    }

    // Step 3: Build canonical graph
    let mut canonical = EinsumGraph::new();

    // Add tensors with canonical names
    for i in 0..tensor_order.len() {
        canonical.add_tensor(format!("t{}", i));
    }

    // Step 4: Remap and add nodes in topological order
    let sorted_nodes = topological_sort_nodes(graph)?;
    for node_idx in sorted_nodes {
        let old_node = &graph.nodes[node_idx];
        let new_node = remap_node(old_node, &tensor_mapping);
        canonical.add_node(new_node)?;
    }

    // Step 5: Remap and sort inputs
    let mut new_inputs: Vec<usize> = graph
        .inputs
        .iter()
        .map(|&idx| *tensor_mapping.get(&idx).unwrap())
        .collect();
    new_inputs.sort_unstable();
    canonical.inputs = new_inputs;

    // Step 6: Remap and sort outputs
    let mut new_outputs: Vec<usize> = graph
        .outputs
        .iter()
        .map(|&idx| *tensor_mapping.get(&idx).unwrap())
        .collect();
    new_outputs.sort_unstable();
    canonical.outputs = new_outputs;

    Ok(canonical)
}

/// Compute topological ordering of tensors.
///
/// Returns a vector of tensor indices in topological order, where:
/// - Input tensors come first
/// - Intermediate tensors are ordered by their producers
/// - Unused tensors come last
fn topological_sort_tensors(graph: &EinsumGraph) -> Result<Vec<usize>, IrError> {
    let num_tensors = graph.tensors.len();

    // Track tensor dependencies: which tensors are used to produce each tensor
    let mut producers: HashMap<usize, usize> = HashMap::new(); // tensor -> node that produces it
    let mut dependencies: HashMap<usize, Vec<usize>> = HashMap::new(); // tensor -> input tensors

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output_tensor in &node.outputs {
            producers.insert(output_tensor, node_idx);
            dependencies.insert(output_tensor, node.inputs.clone());
        }
    }

    // Tensors with no producers are inputs or constants
    let mut result = Vec::new();
    let mut visited = HashSet::new();
    let mut processing = HashSet::new();

    // Helper function for DFS traversal
    fn visit(
        tensor_idx: usize,
        dependencies: &HashMap<usize, Vec<usize>>,
        visited: &mut HashSet<usize>,
        processing: &mut HashSet<usize>,
        result: &mut Vec<usize>,
    ) -> Result<(), IrError> {
        if visited.contains(&tensor_idx) {
            return Ok(());
        }
        if processing.contains(&tensor_idx) {
            return Err(IrError::CyclicGraph);
        }

        processing.insert(tensor_idx);

        // Visit dependencies first
        if let Some(deps) = dependencies.get(&tensor_idx) {
            for &dep in deps {
                visit(dep, dependencies, visited, processing, result)?;
            }
        }

        processing.remove(&tensor_idx);
        visited.insert(tensor_idx);
        result.push(tensor_idx);

        Ok(())
    }

    // Process all tensors
    for tensor_idx in 0..num_tensors {
        if !visited.contains(&tensor_idx) {
            visit(
                tensor_idx,
                &dependencies,
                &mut visited,
                &mut processing,
                &mut result,
            )?;
        }
    }

    Ok(result)
}

/// Compute topological ordering of nodes.
///
/// Returns a vector of node indices in execution order.
fn topological_sort_nodes(graph: &EinsumGraph) -> Result<Vec<usize>, IrError> {
    let num_nodes = graph.nodes.len();

    // Build dependency graph: which nodes must execute before others
    let mut in_degree = vec![0; num_nodes];
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];

    // Track which node produces each tensor
    let mut tensor_producers: HashMap<usize, usize> = HashMap::new();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output_tensor in &node.outputs {
            tensor_producers.insert(output_tensor, node_idx);
        }
    }

    // Build edges: if node B uses a tensor produced by node A, then A -> B
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &input_tensor in &node.inputs {
            if let Some(&producer_idx) = tensor_producers.get(&input_tensor) {
                if producer_idx != node_idx {
                    adjacency[producer_idx].push(node_idx);
                    in_degree[node_idx] += 1;
                }
            }
        }
    }

    // Kahn's algorithm for topological sort
    let mut queue = VecDeque::new();
    for (idx, &degree) in in_degree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(idx);
        }
    }

    let mut result = Vec::new();
    while let Some(node_idx) = queue.pop_front() {
        result.push(node_idx);

        for &neighbor in &adjacency[node_idx] {
            in_degree[neighbor] -= 1;
            if in_degree[neighbor] == 0 {
                queue.push_back(neighbor);
            }
        }
    }

    if result.len() != num_nodes {
        return Err(IrError::CyclicGraph);
    }

    Ok(result)
}

/// Remap a node's tensor indices using the provided mapping.
fn remap_node(node: &EinsumNode, tensor_mapping: &HashMap<usize, usize>) -> EinsumNode {
    let new_inputs = node
        .inputs
        .iter()
        .map(|&idx| *tensor_mapping.get(&idx).unwrap())
        .collect();
    let new_outputs = node
        .outputs
        .iter()
        .map(|&idx| *tensor_mapping.get(&idx).unwrap())
        .collect();

    EinsumNode {
        op: node.op.clone(),
        inputs: new_inputs,
        outputs: new_outputs,
        metadata: node.metadata.clone(),
    }
}

/// Check if two graphs are canonically equivalent.
///
/// This is more efficient than canonicalizing both graphs and comparing,
/// as it can short-circuit on basic structural differences.
pub fn are_graphs_equivalent(g1: &EinsumGraph, g2: &EinsumGraph) -> bool {
    // Quick structural checks
    if g1.tensors.len() != g2.tensors.len()
        || g1.nodes.len() != g2.nodes.len()
        || g1.inputs.len() != g2.inputs.len()
        || g1.outputs.len() != g2.outputs.len()
    {
        return false;
    }

    // Canonicalize and compare
    match (canonicalize_graph(g1), canonicalize_graph(g2)) {
        (Ok(c1), Ok(c2)) => c1 == c2,
        _ => false,
    }
}

/// Compute a hash of a graph in canonical form.
///
/// This can be used for efficient graph deduplication and caching.
pub fn canonical_hash(graph: &EinsumGraph) -> Result<u64, IrError> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let canonical = canonicalize_graph(graph)?;

    let mut hasher = DefaultHasher::new();

    // Hash the structure
    canonical.tensors.len().hash(&mut hasher);
    canonical.nodes.len().hash(&mut hasher);
    canonical.inputs.len().hash(&mut hasher);
    canonical.outputs.len().hash(&mut hasher);

    // Hash tensor names (should all be t0, t1, t2, ... but hash anyway)
    for tensor in &canonical.tensors {
        tensor.hash(&mut hasher);
    }

    // Hash nodes
    for node in &canonical.nodes {
        // Hash operation type
        match &node.op {
            super::OpType::Einsum { spec } => {
                "einsum".hash(&mut hasher);
                spec.hash(&mut hasher);
            }
            super::OpType::ElemUnary { op } => {
                "elem_unary".hash(&mut hasher);
                op.hash(&mut hasher);
            }
            super::OpType::ElemBinary { op } => {
                "elem_binary".hash(&mut hasher);
                op.hash(&mut hasher);
            }
            super::OpType::Reduce { op, axes } => {
                "reduce".hash(&mut hasher);
                op.hash(&mut hasher);
                axes.hash(&mut hasher);
            }
        }

        // Hash inputs and outputs
        node.inputs.hash(&mut hasher);
        node.outputs.hash(&mut hasher);
    }

    // Hash inputs and outputs
    canonical.inputs.hash(&mut hasher);
    canonical.outputs.hash(&mut hasher);

    Ok(hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph_canonicalization() {
        let graph = EinsumGraph::new();
        let canonical = canonicalize_graph(&graph).unwrap();
        assert!(canonical.is_empty());
    }

    #[test]
    fn test_simple_graph_canonicalization() {
        // Build a simple graph: A @ B = C
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("matrix_A");
        let b = graph.add_tensor("matrix_B");
        let c = graph.add_tensor("result");

        graph
            .add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c]))
            .unwrap();
        graph.add_output(c).unwrap();

        let canonical = canonicalize_graph(&graph).unwrap();

        // Check tensor names are canonical
        assert_eq!(canonical.tensors, vec!["t0", "t1", "t2"]);

        // Check structure is preserved
        assert_eq!(canonical.nodes.len(), 1);
        assert_eq!(canonical.outputs.len(), 1);
    }

    #[test]
    fn test_tensor_reordering() {
        // Build two graphs with different tensor orderings but same computation
        let mut g1 = EinsumGraph::new();
        let a1 = g1.add_tensor("A");
        let b1 = g1.add_tensor("B");
        let c1 = g1.add_tensor("C");
        g1.add_node(EinsumNode::elem_binary("mul", a1, b1, c1))
            .unwrap();
        g1.add_output(c1).unwrap();

        let mut g2 = EinsumGraph::new();
        let x2 = g2.add_tensor("X");
        let y2 = g2.add_tensor("Y");
        let z2 = g2.add_tensor("Z");
        g2.add_node(EinsumNode::elem_binary("mul", x2, y2, z2))
            .unwrap();
        g2.add_output(z2).unwrap();

        // Both should canonicalize to the same structure
        let c1 = canonicalize_graph(&g1).unwrap();
        let c2 = canonicalize_graph(&g2).unwrap();

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_graph_equivalence() {
        let mut g1 = EinsumGraph::new();
        let a = g1.add_tensor("foo");
        let b = g1.add_tensor("bar");
        g1.add_node(EinsumNode::elem_unary("neg", a, b)).unwrap();

        let mut g2 = EinsumGraph::new();
        let x = g2.add_tensor("different");
        let y = g2.add_tensor("names");
        g2.add_node(EinsumNode::elem_unary("neg", x, y)).unwrap();

        assert!(are_graphs_equivalent(&g1, &g2));
    }

    #[test]
    fn test_non_equivalent_graphs() {
        let mut g1 = EinsumGraph::new();
        let a = g1.add_tensor("A");
        let b = g1.add_tensor("B");
        g1.add_node(EinsumNode::elem_unary("neg", a, b)).unwrap();

        let mut g2 = EinsumGraph::new();
        let x = g2.add_tensor("X");
        let y = g2.add_tensor("Y");
        g2.add_node(EinsumNode::elem_unary("sqrt", x, y)).unwrap();

        assert!(!are_graphs_equivalent(&g1, &g2));
    }

    #[test]
    fn test_canonical_hash_consistency() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        graph
            .add_node(EinsumNode::elem_binary("add", a, a, b))
            .unwrap();

        let hash1 = canonical_hash(&graph).unwrap();
        let hash2 = canonical_hash(&graph).unwrap();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_equivalent_graphs_same_hash() {
        let mut g1 = EinsumGraph::new();
        let a1 = g1.add_tensor("foo");
        let b1 = g1.add_tensor("bar");
        g1.add_node(EinsumNode::elem_unary("exp", a1, b1)).unwrap();

        let mut g2 = EinsumGraph::new();
        let a2 = g2.add_tensor("different");
        let b2 = g2.add_tensor("names");
        g2.add_node(EinsumNode::elem_unary("exp", a2, b2)).unwrap();

        let hash1 = canonical_hash(&g1).unwrap();
        let hash2 = canonical_hash(&g2).unwrap();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_complex_graph_canonicalization() {
        // Build a multi-node graph
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("input1");
        let b = graph.add_tensor("input2");
        let c = graph.add_tensor("intermediate1");
        let d = graph.add_tensor("intermediate2");
        let e = graph.add_tensor("output");

        graph
            .add_node(EinsumNode::elem_binary("mul", a, b, c))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("sqrt", c, d))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_binary("add", d, a, e))
            .unwrap();
        graph.add_output(e).unwrap();

        let canonical = canonicalize_graph(&graph).unwrap();

        // Verify canonicalization worked
        assert_eq!(canonical.tensors.len(), 5);
        assert_eq!(canonical.nodes.len(), 3);

        // Verify all tensor names are canonical
        for (i, name) in canonical.tensors.iter().enumerate() {
            assert_eq!(name, &format!("t{}", i));
        }
    }

    #[test]
    fn test_topological_sort_simple() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        // A -> B -> C
        graph.add_node(EinsumNode::elem_unary("op1", a, b)).unwrap();
        graph.add_node(EinsumNode::elem_unary("op2", b, c)).unwrap();

        let node_order = topological_sort_nodes(&graph).unwrap();

        // First node should come before second node
        assert_eq!(node_order, vec![0, 1]);
    }

    #[test]
    fn test_inputs_outputs_preservation() {
        let mut graph = EinsumGraph::new();
        let in1 = graph.add_tensor("input1");
        let in2 = graph.add_tensor("input2");
        let out1 = graph.add_tensor("output1");
        let out2 = graph.add_tensor("output2");

        graph.inputs = vec![in1, in2];
        graph.outputs = vec![out1, out2];

        graph
            .add_node(EinsumNode::elem_unary("op1", in1, out1))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("op2", in2, out2))
            .unwrap();

        let canonical = canonicalize_graph(&graph).unwrap();

        // Inputs and outputs should be preserved (but sorted)
        assert_eq!(canonical.inputs.len(), 2);
        assert_eq!(canonical.outputs.len(), 2);

        // They should be sorted
        let mut sorted_inputs = canonical.inputs.clone();
        sorted_inputs.sort_unstable();
        assert_eq!(canonical.inputs, sorted_inputs);

        let mut sorted_outputs = canonical.outputs.clone();
        sorted_outputs.sort_unstable();
        assert_eq!(canonical.outputs, sorted_outputs);
    }
}
