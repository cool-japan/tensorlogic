#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType, optimize};

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }

    let mut graph = EinsumGraph::new();
    let num_nodes = (data[0] % 20) as usize + 1; // 1-20 nodes

    // Build a more complex graph with potential optimization opportunities
    for i in 0..num_nodes.min(data.len() / 5) {
        let idx = i * 5;
        if idx + 4 >= data.len() {
            break;
        }

        let node_id = format!("n{}", i);
        let op_byte = data[idx];

        let op_type = match op_byte % 8 {
            0 => {
                // Identity operation (can be eliminated)
                OpType::ElemUnary {
                    op: tensorlogic_ir::ElemOp::Identity,
                    input: format!("n{}", i.saturating_sub(1)),
                }
            }
            1 => {
                // Constant (potential for folding)
                OpType::Constant {
                    value: (data[idx + 1] as f64) / 255.0,
                    shape: vec![
                        (data[idx + 2] % 5 + 1) as usize,
                        (data[idx + 3] % 5 + 1) as usize,
                    ],
                }
            }
            2 => {
                // Negation (potential for double-negation elimination)
                OpType::ElemUnary {
                    op: tensorlogic_ir::ElemOp::Neg,
                    input: format!("n{}", i.saturating_sub(1)),
                }
            }
            3 => {
                // Multiplication (potential for CSE)
                OpType::ElemBinary {
                    op: tensorlogic_ir::BinOp::Mul,
                    lhs: format!("n{}", (i / 2).saturating_sub(1)),
                    rhs: format!("n{}", (i / 2)),
                }
            }
            4 => {
                // Addition (potential for CSE)
                OpType::ElemBinary {
                    op: tensorlogic_ir::BinOp::Add,
                    lhs: format!("n{}", (i / 2).saturating_sub(1)),
                    rhs: format!("n{}", (i / 2)),
                }
            }
            5 => {
                // Einsum operation
                OpType::Einsum {
                    spec: "ij,jk->ik".to_string(),
                    inputs: vec![
                        format!("n{}", (i / 2).saturating_sub(1)),
                        format!("n{}", i.saturating_sub(1)),
                    ],
                }
            }
            6 => {
                // Reduction (potential for fusion)
                OpType::Reduce {
                    op: tensorlogic_ir::ReduceOp::Sum,
                    input: format!("n{}", i.saturating_sub(1)),
                    axes: vec![(data[idx + 1] % 2) as usize],
                }
            }
            _ => {
                // Max reduction
                OpType::Reduce {
                    op: tensorlogic_ir::ReduceOp::Max,
                    input: format!("n{}", i.saturating_sub(1)),
                    axes: vec![0],
                }
            }
        };

        let node = EinsumNode {
            id: node_id,
            op_type,
            output_axes: vec!["i".to_string(), "j".to_string()],
            metadata: None,
        };

        graph.add_node(node);
    }

    // Test optimization passes (shouldn't panic)
    if let Ok(optimized) = optimize::dead_code_elimination(&graph) {
        // Verify optimized graph is valid
        let _ = optimized.num_nodes();
        let _ = optimized.output_nodes();

        // Test serialization of optimized graph
        let _ = serde_json::to_string(&optimized);
    }

    if let Ok(optimized) = optimize::common_subexpression_elimination(&graph) {
        let _ = optimized.num_nodes();
        let _ = optimized.output_nodes();
    }

    // Test full optimization pipeline
    if let Ok(pipeline_result) = optimize::optimize_graph(&graph, &optimize::OptimizationConfig::aggressive()) {
        let _ = pipeline_result.num_nodes();
        let _ = format!("{:?}", pipeline_result.stats);
    }

    drop(graph);
});
