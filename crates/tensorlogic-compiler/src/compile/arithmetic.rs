//! Compilation of arithmetic and mathematical operations.

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr};

use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

/// Compile addition: a + b
pub(crate) fn compile_add(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let left_state = compile_expr(left, ctx, graph)?;
    let right_state = compile_expr(right, ctx, graph)?;

    // For arithmetic, we assume element-wise operations, so axes should match
    let axes = left_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise binary operation node
    let node = EinsumNode::elem_binary(
        "add",
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

/// Compile subtraction: a - b
pub(crate) fn compile_sub(
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

    // Create element-wise binary operation node
    let node = EinsumNode::elem_binary(
        "subtract",
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

/// Compile multiplication: a * b
pub(crate) fn compile_mul(
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

    // Create element-wise binary operation node
    let node = EinsumNode::elem_binary(
        "multiply",
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

/// Compile division: a / b
pub(crate) fn compile_div(
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

    // Create element-wise binary operation node
    let node = EinsumNode::elem_binary(
        "divide",
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

/// Compile power: a ^ b
pub(crate) fn compile_pow(
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

    // Create element-wise binary operation node
    let node = EinsumNode::elem_binary(
        "power",
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

/// Compile modulo: a % b
pub(crate) fn compile_mod(
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

    // Create element-wise binary operation node
    let node = EinsumNode::elem_binary(
        "modulo",
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

/// Compile minimum: min(a, b)
pub(crate) fn compile_min_binary(
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

    // Create element-wise binary operation node
    let node = EinsumNode::elem_binary(
        "minimum",
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

/// Compile maximum: max(a, b)
pub(crate) fn compile_max_binary(
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

    // Create element-wise binary operation node
    let node = EinsumNode::elem_binary(
        "maximum",
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

/// Compile absolute value: abs(x)
pub(crate) fn compile_abs(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    let axes = inner_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise unary operation node
    let node = EinsumNode::elem_unary("abs", inner_state.tensor_idx, result_idx);

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile floor: floor(x)
pub(crate) fn compile_floor(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    let axes = inner_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise unary operation node
    let node = EinsumNode::elem_unary("floor", inner_state.tensor_idx, result_idx);

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile ceil: ceil(x)
pub(crate) fn compile_ceil(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    let axes = inner_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise unary operation node
    let node = EinsumNode::elem_unary("ceil", inner_state.tensor_idx, result_idx);

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile round: round(x)
pub(crate) fn compile_round(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    let axes = inner_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise unary operation node
    let node = EinsumNode::elem_unary("round", inner_state.tensor_idx, result_idx);

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile square root: sqrt(x)
pub(crate) fn compile_sqrt(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    let axes = inner_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise unary operation node
    let node = EinsumNode::elem_unary("sqrt", inner_state.tensor_idx, result_idx);

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile exponential: exp(x)
pub(crate) fn compile_exp(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    let axes = inner_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise unary operation node
    let node = EinsumNode::elem_unary("exp", inner_state.tensor_idx, result_idx);

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile natural logarithm: log(x)
pub(crate) fn compile_log(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    let axes = inner_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise unary operation node
    let node = EinsumNode::elem_unary("log", inner_state.tensor_idx, result_idx);

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile sine: sin(x)
pub(crate) fn compile_sin(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    let axes = inner_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise unary operation node
    let node = EinsumNode::elem_unary("sin", inner_state.tensor_idx, result_idx);

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile cosine: cos(x)
pub(crate) fn compile_cos(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    let axes = inner_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise unary operation node
    let node = EinsumNode::elem_unary("cos", inner_state.tensor_idx, result_idx);

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}

/// Compile tangent: tan(x)
pub(crate) fn compile_tan(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    let axes = inner_state.axes.clone();

    // Create result tensor first
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create element-wise unary operation node
    let node = EinsumNode::elem_unary("tan", inner_state.tensor_idx, result_idx);

    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes,
    })
}
