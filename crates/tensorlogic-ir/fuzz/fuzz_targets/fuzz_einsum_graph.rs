#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_ir::{EinsumGraph, EinsumNode, ElemOp, OpType, ReduceOp};

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let num_nodes = (data[0] % 10) as usize + 1; // 1-10 nodes
    let mut graph = EinsumGraph::new();

    // Create nodes with various operation types
    for i in 0..num_nodes.min(data.len() / 3) {
        let idx = i * 3;
        if idx + 2 >= data.len() {
            break;
        }

        let op_type_byte = data[idx];
        let node_id = format!("node_{}", i);

        let op_type = match op_type_byte % 5 {
            0 => OpType::Einsum {
                spec: "ij,jk->ik".to_string(),
                inputs: vec!["input0".to_string(), "input1".to_string()],
            },
            1 => OpType::ElemUnary {
                op: ElemOp::Neg,
                input: "input0".to_string(),
            },
            2 => OpType::ElemBinary {
                op: tensorlogic_ir::BinOp::Mul,
                lhs: "input0".to_string(),
                rhs: "input1".to_string(),
            },
            3 => OpType::Reduce {
                op: ReduceOp::Sum,
                input: "input0".to_string(),
                axes: vec![0],
            },
            _ => OpType::Constant {
                value: 1.0,
                shape: vec![2, 2],
            },
        };

        let node = EinsumNode {
            id: node_id.clone(),
            op_type,
            output_axes: vec!["i".to_string(), "j".to_string()],
            metadata: None,
        };

        graph.add_node(node);
    }

    // Test graph operations
    let _ = graph.num_nodes();
    let _ = graph.output_nodes();

    // Test serialization
    if let Ok(serialized) = serde_json::to_string(&graph) {
        let _: Result<EinsumGraph, _> = serde_json::from_str(&serialized);
    }

    // Test bincode serialization
    if let Ok(encoded) = bincode::encode_to_vec(&graph, bincode::config::standard()) {
        let _: Result<(EinsumGraph, _), _> =
            bincode::decode_from_slice(&encoded, bincode::config::standard());
    }

    // Test debug formatting
    let _ = format!("{:?}", graph);

    // Test topological properties (shouldn't panic)
    for node_id in graph.output_nodes() {
        let _ = graph.get_node(&node_id);
    }

    drop(graph);
});
