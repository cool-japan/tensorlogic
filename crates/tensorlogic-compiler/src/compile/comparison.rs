//! Compilation of comparison operations (Eq, Lt, Gt, Lte, Gte).

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr};

use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

/// Compile equality: a == b
pub(crate) fn compile_eq(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let left_state = compile_expr(left, ctx, graph)?;
    let right_state = compile_expr(right, ctx, graph)?;

    let axes = left_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise binary comparison node
    let node = EinsumNode::elem_binary(
        "eq",
        left_state.tensor_idx,
        right_state.tensor_idx,
        result_idx,
    );

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile less than: a < b
pub(crate) fn compile_lt(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let left_state = compile_expr(left, ctx, graph)?;
    let right_state = compile_expr(right, ctx, graph)?;

    let axes = left_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise binary comparison node
    let node = EinsumNode::elem_binary(
        "lt",
        left_state.tensor_idx,
        right_state.tensor_idx,
        result_idx,
    );

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile greater than: a > b
pub(crate) fn compile_gt(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let left_state = compile_expr(left, ctx, graph)?;
    let right_state = compile_expr(right, ctx, graph)?;

    let axes = left_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise binary comparison node
    let node = EinsumNode::elem_binary(
        "gt",
        left_state.tensor_idx,
        right_state.tensor_idx,
        result_idx,
    );

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile less than or equal: a <= b
pub(crate) fn compile_lte(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let left_state = compile_expr(left, ctx, graph)?;
    let right_state = compile_expr(right, ctx, graph)?;

    let axes = left_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise binary comparison node
    let node = EinsumNode::elem_binary(
        "lte",
        left_state.tensor_idx,
        right_state.tensor_idx,
        result_idx,
    );

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile greater than or equal: a >= b
pub(crate) fn compile_gte(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let left_state = compile_expr(left, ctx, graph)?;
    let right_state = compile_expr(right, ctx, graph)?;

    let axes = left_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise binary comparison node
    let node = EinsumNode::elem_binary(
        "gte",
        left_state.tensor_idx,
        right_state.tensor_idx,
        result_idx,
    );

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}
