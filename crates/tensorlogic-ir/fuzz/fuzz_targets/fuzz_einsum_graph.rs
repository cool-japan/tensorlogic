#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_ir::{EinsumGraph, EinsumNode};

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let num_operations = (data[0] % 10) as usize + 1; // 1-10 operations
    let mut graph = EinsumGraph::new();

    // Create some input tensors
    let t0 = graph.add_tensor("input0");
    let t1 = graph.add_tensor("input1");
    let t2 = graph.add_tensor("input2");

    // Mark inputs
    let _ = graph.add_input(t0);
    let _ = graph.add_input(t1);

    let mut last_tensor = t2;

    // Create nodes with various operation types based on fuzz input
    for i in 0..num_operations.min(data.len() / 3) {
        let idx = i * 3;
        if idx + 2 >= data.len() {
            break;
        }

        let op_type_byte = data[idx];
        let output_tensor = graph.add_tensor(format!("t{}", i + 3));

        let node = match op_type_byte % 5 {
            0 => {
                // Einsum: matrix multiplication
                EinsumNode::einsum("ij,jk->ik", vec![t0, t1], vec![output_tensor])
            }
            1 => {
                // Element-wise unary: negation
                EinsumNode::elem_unary("neg", last_tensor, output_tensor)
            }
            2 => {
                // Element-wise binary: multiplication
                EinsumNode::elem_binary("mul", t0, last_tensor, output_tensor)
            }
            3 => {
                // Reduction: sum over axis 0
                EinsumNode::reduce("sum", vec![0], last_tensor, output_tensor)
            }
            _ => {
                // Element-wise binary: addition
                EinsumNode::elem_binary("add", t1, last_tensor, output_tensor)
            }
        };

        if graph.add_node(node).is_ok() {
            last_tensor = output_tensor;
        }
    }

    // Mark the last tensor as output
    let _ = graph.add_output(last_tensor);

    // Test graph operations (shouldn't panic)
    let _ = graph.validate();
    let _ = graph.is_empty();

    // Test serialization (JSON only - no bincode per COOLJAPAN policy)
    if let Ok(serialized) = serde_json::to_string(&graph) {
        let _: Result<EinsumGraph, _> = serde_json::from_str(&serialized);
    }

    // Test debug formatting
    let _ = format!("{:?}", graph);

    // Test iteration over nodes
    for node in &graph.nodes {
        let _ = node.primary_output();
        let _ = node.operation_description();
        let _ = node.parse_einsum_spec();
    }

    // Test tensor metadata operations
    for i in 0..graph.tensors.len() {
        let _ = graph.get_tensor_metadata(i);
    }

    drop(graph);
});
