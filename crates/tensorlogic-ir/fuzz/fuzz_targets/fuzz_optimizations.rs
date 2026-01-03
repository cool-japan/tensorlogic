#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_ir::{
    eliminate_common_subexpressions, eliminate_dead_code, optimize_graph,
    simplify_identity_operations, EinsumGraph, EinsumNode,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }

    let mut graph = EinsumGraph::new();
    let num_nodes = (data[0] % 20) as usize + 1; // 1-20 nodes

    // Build a graph with potential optimization opportunities
    let mut tensor_indices = Vec::new();

    // Create initial input tensors
    for i in 0..3 {
        let idx = graph.add_tensor(format!("input{}", i));
        tensor_indices.push(idx);
        let _ = graph.add_input(idx);
    }

    // Build nodes based on fuzz input
    for i in 0..num_nodes.min(data.len() / 5) {
        let idx = i * 5;
        if idx + 4 >= data.len() {
            break;
        }

        let op_byte = data[idx];
        let output_tensor = graph.add_tensor(format!("t{}", i));

        let node = match op_byte % 8 {
            0 => {
                // Identity operation (can be eliminated)
                if !tensor_indices.is_empty() {
                    let input = tensor_indices[data[idx + 1] as usize % tensor_indices.len()];
                    EinsumNode::einsum("a->a", vec![input], vec![output_tensor])
                } else {
                    continue;
                }
            }
            1 => {
                // Negation (potential for double-negation elimination)
                if !tensor_indices.is_empty() {
                    let input = tensor_indices[data[idx + 1] as usize % tensor_indices.len()];
                    EinsumNode::elem_unary("neg", input, output_tensor)
                } else {
                    continue;
                }
            }
            2 => {
                // Multiplication (potential for CSE if duplicated)
                if tensor_indices.len() >= 2 {
                    let left = tensor_indices[data[idx + 1] as usize % tensor_indices.len()];
                    let right = tensor_indices[data[idx + 2] as usize % tensor_indices.len()];
                    EinsumNode::elem_binary("mul", left, right, output_tensor)
                } else {
                    continue;
                }
            }
            3 => {
                // Addition (potential for CSE if duplicated)
                if tensor_indices.len() >= 2 {
                    let left = tensor_indices[data[idx + 1] as usize % tensor_indices.len()];
                    let right = tensor_indices[data[idx + 2] as usize % tensor_indices.len()];
                    EinsumNode::elem_binary("add", left, right, output_tensor)
                } else {
                    continue;
                }
            }
            4 => {
                // Einsum operation
                if tensor_indices.len() >= 2 {
                    let t1 = tensor_indices[data[idx + 1] as usize % tensor_indices.len()];
                    let t2 = tensor_indices[data[idx + 2] as usize % tensor_indices.len()];
                    EinsumNode::einsum("ij,jk->ik", vec![t1, t2], vec![output_tensor])
                } else {
                    continue;
                }
            }
            5 => {
                // Reduction (potential for fusion)
                if !tensor_indices.is_empty() {
                    let input = tensor_indices[data[idx + 1] as usize % tensor_indices.len()];
                    let axis = (data[idx + 2] % 2) as usize;
                    EinsumNode::reduce("sum", vec![axis], input, output_tensor)
                } else {
                    continue;
                }
            }
            6 => {
                // Max reduction
                if !tensor_indices.is_empty() {
                    let input = tensor_indices[data[idx + 1] as usize % tensor_indices.len()];
                    EinsumNode::reduce("max", vec![0], input, output_tensor)
                } else {
                    continue;
                }
            }
            _ => {
                // Duplicate a previous operation (for CSE testing)
                if graph.nodes.is_empty() {
                    continue;
                }
                let prev_node = &graph.nodes[data[idx + 1] as usize % graph.nodes.len()];
                EinsumNode {
                    op: prev_node.op.clone(),
                    inputs: prev_node.inputs.clone(),
                    outputs: vec![output_tensor],
                    metadata: None,
                }
            }
        };

        if graph.add_node(node).is_ok() {
            tensor_indices.push(output_tensor);
        }
    }

    // Mark last few tensors as outputs
    if !tensor_indices.is_empty() {
        let num_outputs = (data[1] % 3) as usize + 1;
        for i in 0..num_outputs.min(tensor_indices.len()) {
            let output_idx = tensor_indices[tensor_indices.len() - 1 - i];
            let _ = graph.add_output(output_idx);
        }
    }

    // Clone graph for multiple optimization passes
    let mut graph_dce = graph.clone();
    let mut graph_cse = graph.clone();
    let mut graph_identity = graph.clone();
    let mut graph_full = graph.clone();

    // Test individual optimization passes (shouldn't panic)
    let _ = eliminate_dead_code(&mut graph_dce);
    let _ = eliminate_common_subexpressions(&mut graph_cse);
    let _ = simplify_identity_operations(&mut graph_identity);

    // Test full optimization pipeline
    if let Ok(stats) = optimize_graph(&mut graph_full) {
        // Verify stats
        let _ = stats.total_optimizations();
        let _ = format!("{:?}", stats);

        // Verify optimized graph is still valid
        let _ = graph_full.validate();
        let _ = graph_full.is_empty();
    }

    // Test serialization of optimized graph
    let _ = serde_json::to_string(&graph_full);

    drop(graph);
    drop(graph_dce);
    drop(graph_cse);
    drop(graph_identity);
    drop(graph_full);
});
