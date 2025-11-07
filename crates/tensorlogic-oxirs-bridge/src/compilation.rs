//! Rule compilation utilities.

use anyhow::{bail, Result};
use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_ir::{EinsumGraph, TLExpr};

/// Compile multiple TLExpr rules into a single execution plan
///
/// Currently performs simple graph merging. Future optimizations:
/// - Merge compatible rules
/// - Deduplicate shared subexpressions
/// - Optimize tensor ordering
pub fn compile_rules(rules: &[TLExpr]) -> Result<EinsumGraph> {
    if rules.is_empty() {
        bail!("No rules to compile");
    }

    // Compile each rule separately
    let graphs: Result<Vec<_>> = rules.iter().map(compile_to_einsum).collect();
    let graphs = graphs?;

    // Merge graphs (simple concatenation for now)
    let mut merged = EinsumGraph::new();
    let mut tensor_offset = 0;

    for graph in graphs {
        for tensor in graph.tensors {
            merged.add_tensor(tensor);
        }
        for node in graph.nodes {
            merged.add_node(node)?;
        }
        for output in graph.outputs {
            merged.outputs.push(output + tensor_offset);
        }
        tensor_offset = merged.tensors.len();
    }

    Ok(merged)
}
