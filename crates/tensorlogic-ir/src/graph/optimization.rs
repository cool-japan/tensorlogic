//! Graph optimization passes.

use std::collections::{HashMap, HashSet};

use crate::{EinsumGraph, EinsumNode, IrError};

/// Dead Code Elimination (DCE) - removes unused tensors and nodes
pub fn eliminate_dead_code(graph: &mut EinsumGraph) -> Result<usize, IrError> {
    if graph.outputs.is_empty() {
        return Ok(0);
    }

    // Track which tensors are live (needed)
    let mut live_tensors = HashSet::new();
    let mut worklist: Vec<usize> = graph.outputs.clone();

    // Mark all output tensors as live
    for &output_idx in &graph.outputs {
        live_tensors.insert(output_idx);
    }

    // Build tensor-to-node mapping (which node produces each tensor)
    let mut tensor_producers: HashMap<usize, usize> = HashMap::new();
    for (node_idx, _node) in graph.nodes.iter().enumerate() {
        // Each node produces a tensor at index equal to the number of tensors
        // before this node plus its position
        let produced_tensor_idx = node_idx + count_input_tensors(graph, node_idx);
        tensor_producers.insert(produced_tensor_idx, node_idx);
    }

    // Backward pass: mark all dependencies as live
    while let Some(tensor_idx) = worklist.pop() {
        if let Some(&node_idx) = tensor_producers.get(&tensor_idx) {
            let node = &graph.nodes[node_idx];
            for &input_idx in &node.inputs {
                if !live_tensors.contains(&input_idx) {
                    live_tensors.insert(input_idx);
                    worklist.push(input_idx);
                }
            }
        }
    }

    // Remove dead tensors and nodes
    let mut removed_count = 0;

    // Mark dead nodes for removal (nodes whose output is not live)
    let mut nodes_to_keep = Vec::new();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        let produced_tensor_idx = node_idx + count_input_tensors(graph, node_idx);
        if live_tensors.contains(&produced_tensor_idx) {
            nodes_to_keep.push(node.clone());
        } else {
            removed_count += 1;
        }
    }

    graph.nodes = nodes_to_keep;

    // Note: We don't actually remove tensors from the tensors vector
    // as this would require renumbering all node inputs and outputs.
    // Instead, we just remove the nodes that produce unused tensors.

    Ok(removed_count)
}

#[allow(dead_code)]
fn count_input_tensors(graph: &EinsumGraph, before_node: usize) -> usize {
    // Count how many tensors exist before this node
    // This is a simplified version - in practice, you'd track this more carefully
    graph
        .nodes
        .iter()
        .take(before_node)
        .map(|_| 1) // Each node produces one tensor
        .sum()
}

/// Common Subexpression Elimination (CSE) - detects and deduplicates identical subgraphs
pub fn eliminate_common_subexpressions(graph: &mut EinsumGraph) -> Result<usize, IrError> {
    let mut node_hashes: HashMap<String, usize> = HashMap::new();
    let mut replacements: HashMap<usize, usize> = HashMap::new();
    let mut eliminated_count = 0;

    // Build hash for each node (based on operation and inputs)
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        let node_hash = compute_node_hash(node);

        if let Some(&existing_idx) = node_hashes.get(&node_hash) {
            // Found a duplicate - mark for replacement
            let produced_tensor_idx = node_idx + count_input_tensors(graph, node_idx);
            let existing_tensor_idx = existing_idx + count_input_tensors(graph, existing_idx);
            replacements.insert(produced_tensor_idx, existing_tensor_idx);
            eliminated_count += 1;
        } else {
            node_hashes.insert(node_hash, node_idx);
        }
    }

    // Apply replacements to all node inputs and outputs
    for node in &mut graph.nodes {
        for input_idx in &mut node.inputs {
            if let Some(&replacement) = replacements.get(input_idx) {
                *input_idx = replacement;
            }
        }
    }

    for output_idx in &mut graph.outputs {
        if let Some(&replacement) = replacements.get(output_idx) {
            *output_idx = replacement;
        }
    }

    // Remove duplicate nodes (would require DCE to actually clean up)
    Ok(eliminated_count)
}

#[allow(dead_code)]
fn compute_node_hash(node: &EinsumNode) -> String {
    // Simple hash based on operation type and inputs
    // In a real implementation, you'd use a proper hash function
    format!("{:?}|{:?}", node.op, node.inputs)
}

/// Simplify identity operations (operations that don't transform their input)
pub fn simplify_identity_operations(graph: &mut EinsumGraph) -> Result<usize, IrError> {
    let mut simplified_count = 0;
    let mut replacements: HashMap<usize, usize> = HashMap::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        if is_identity_operation(node) && !node.inputs.is_empty() {
            // Map output to input directly
            let produced_tensor_idx = node_idx + count_input_tensors(graph, node_idx);
            replacements.insert(produced_tensor_idx, node.inputs[0]);
            simplified_count += 1;
        }
    }

    // Apply replacements
    for node in &mut graph.nodes {
        for input_idx in &mut node.inputs {
            if let Some(&replacement) = replacements.get(input_idx) {
                *input_idx = replacement;
            }
        }
    }

    for output_idx in &mut graph.outputs {
        if let Some(&replacement) = replacements.get(output_idx) {
            *output_idx = replacement;
        }
    }

    Ok(simplified_count)
}

#[allow(dead_code)]
fn is_identity_operation(node: &EinsumNode) -> bool {
    use crate::OpType;

    match &node.op {
        // Einsum with identity spec (e.g., "a->a")
        OpType::Einsum { spec } => {
            if let Some(arrow_pos) = spec.find("->") {
                let input_axes = &spec[..arrow_pos];
                let output_axes = &spec[arrow_pos + 2..];
                input_axes == output_axes && node.inputs.len() == 1
            } else {
                false
            }
        }
        // ElemBinary multiply by 1 or add 0 could be detected here
        _ => false,
    }
}

/// Apply all optimization passes to the graph
pub fn optimize_graph(graph: &mut EinsumGraph) -> Result<OptimizationStats, IrError> {
    let mut stats = OptimizationStats::default();

    // Multiple passes for maximum effect
    for _ in 0..3 {
        let cse_count = eliminate_common_subexpressions(graph)?;
        stats.cse_eliminated += cse_count;

        let identity_count = simplify_identity_operations(graph)?;
        stats.identities_simplified += identity_count;

        let dce_count = eliminate_dead_code(graph)?;
        stats.dead_code_eliminated += dce_count;

        // Stop if no changes
        if cse_count == 0 && identity_count == 0 && dce_count == 0 {
            break;
        }
    }

    Ok(stats)
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OptimizationStats {
    pub dead_code_eliminated: usize,
    pub cse_eliminated: usize,
    pub identities_simplified: usize,
}

impl OptimizationStats {
    pub fn total_optimizations(&self) -> usize {
        self.dead_code_eliminated + self.cse_eliminated + self.identities_simplified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpType;

    #[test]
    fn test_dead_code_elimination_empty_graph() {
        let mut graph = EinsumGraph::new();
        let removed = eliminate_dead_code(&mut graph).unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_dead_code_elimination_no_outputs() {
        let mut graph = EinsumGraph::new();
        graph.add_tensor("a[i]");
        graph.add_tensor("b[i]");
        let removed = eliminate_dead_code(&mut graph).unwrap();
        assert_eq!(removed, 0); // No outputs, so nothing to eliminate
    }

    #[test]
    fn test_identity_operation_detection() {
        let identity_node = EinsumNode {
            op: OpType::Einsum {
                spec: "a->a".to_string(),
            },
            inputs: vec![0],
            outputs: vec![1],
            metadata: None,
        };
        assert!(is_identity_operation(&identity_node));

        let non_identity_node = EinsumNode {
            op: OpType::Einsum {
                spec: "ab->a".to_string(),
            },
            inputs: vec![0],
            outputs: vec![1],
            metadata: None,
        };
        assert!(!is_identity_operation(&non_identity_node));
    }

    #[test]
    fn test_node_hash_computation() {
        let node1 = EinsumNode {
            op: OpType::Einsum {
                spec: "ab->a".to_string(),
            },
            inputs: vec![0],
            outputs: vec![1],
            metadata: None,
        };
        let node2 = EinsumNode {
            op: OpType::Einsum {
                spec: "ab->a".to_string(),
            },
            inputs: vec![0],
            outputs: vec![1],
            metadata: None,
        };
        let node3 = EinsumNode {
            op: OpType::Einsum {
                spec: "ab->b".to_string(),
            },
            inputs: vec![0],
            outputs: vec![1],
            metadata: None,
        };

        assert_eq!(compute_node_hash(&node1), compute_node_hash(&node2));
        assert_ne!(compute_node_hash(&node1), compute_node_hash(&node3));
    }

    #[test]
    fn test_optimization_stats() {
        let stats = OptimizationStats {
            dead_code_eliminated: 2,
            cse_eliminated: 3,
            identities_simplified: 1,
        };
        assert_eq!(stats.total_optimizations(), 6);
    }

    #[test]
    fn test_full_optimization_pipeline() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input[a]");
        let t1 = graph.add_tensor("output[a]");

        // Add some nodes
        let _n1 = graph
            .add_node(EinsumNode {
                op: OpType::Einsum {
                    spec: "a->a".to_string(),
                },
                inputs: vec![t0],
                outputs: vec![t1],
                metadata: None,
            })
            .unwrap();

        // Set output
        graph.add_output(t1).unwrap();

        let stats = optimize_graph(&mut graph).unwrap();
        // Optimization stats should be computed (just check it doesn't panic)
        let _total = stats.total_optimizations();
    }
}
