//! Compilation of conditional expressions and constants.

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr};

use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

/// Compile if-then-else: condition ? then_branch : else_branch
///
/// Compiles to: condition * then_branch + (1 - condition) * else_branch
/// This assumes condition is a soft (probabilistic) value in [0, 1].
pub(crate) fn compile_if_then_else(
    condition: &TLExpr,
    then_branch: &TLExpr,
    else_branch: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Compile all three branches
    let cond_state = compile_expr(condition, ctx, graph)?;
    let then_state = compile_expr(then_branch, ctx, graph)?;
    let else_state = compile_expr(else_branch, ctx, graph)?;

    let axes = cond_state.axes.clone();

    // Create (1 - condition)
    let one_minus_name = ctx.fresh_temp();
    let one_minus_idx = graph.add_tensor(one_minus_name);
    let one_minus_node = EinsumNode::elem_unary("oneminus", cond_state.tensor_idx, one_minus_idx);
    graph.add_node(one_minus_node)?;

    // Create condition * then_branch
    let then_weighted_name = ctx.fresh_temp();
    let then_weighted_idx = graph.add_tensor(then_weighted_name);
    let then_weighted_node = EinsumNode::elem_binary(
        "multiply",
        cond_state.tensor_idx,
        then_state.tensor_idx,
        then_weighted_idx,
    );
    graph.add_node(then_weighted_node)?;

    // Create (1 - condition) * else_branch
    let else_weighted_name = ctx.fresh_temp();
    let else_weighted_idx = graph.add_tensor(else_weighted_name);
    let else_weighted_node = EinsumNode::elem_binary(
        "multiply",
        one_minus_idx,
        else_state.tensor_idx,
        else_weighted_idx,
    );
    graph.add_node(else_weighted_node)?;

    // Create final result: then_weighted + else_weighted
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);
    let result_node =
        EinsumNode::elem_binary("add", then_weighted_idx, else_weighted_idx, result_idx);
    graph.add_node(result_node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile a constant value
pub(crate) fn compile_constant(
    value: f64,
    _ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Create a scalar constant tensor
    let tensor_name = format!("const_{}", value);
    let tensor_idx = graph.add_tensor(&tensor_name);

    // Add metadata indicating this is a constant with a specific value
    // The backend will need to initialize this tensor with the constant value

    Ok(CompileState {
        tensor_idx,
        axes: String::new(), // Constants are scalars with no axes
    })
}
