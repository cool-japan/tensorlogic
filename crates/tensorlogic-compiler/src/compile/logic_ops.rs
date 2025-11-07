//! Compilation of logical operations: AND, OR, NOT.

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr};

use crate::context::{CompileState, CompilerContext};

use super::{compile_expr, strategy_mapping};

pub(crate) fn compile_and(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    use crate::config::AndStrategy;

    let left_state = compile_expr(left, ctx, graph)?;
    let right_state = compile_expr(right, ctx, graph)?;

    // Compute output axes: union of left and right axes (preserving order)
    let mut output_axes = String::new();
    let mut seen = std::collections::HashSet::new();

    // Add all axes from left
    for c in left_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }

    // Add axes from right that aren't in left (these are new free variables)
    for c in right_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }

    // OPTIMIZATION: For Product/ProductTNorm strategy with non-empty axes,
    // use einsum directly (1 operation instead of 3)
    if matches!(
        ctx.config.and_strategy,
        AndStrategy::Product | AndStrategy::ProductTNorm
    ) && !left_state.axes.is_empty()
        && !right_state.axes.is_empty()
    {
        // Build einsum spec: "left,right->output"
        let spec = format!("{},{}->{}", left_state.axes, right_state.axes, output_axes);

        let result_name = ctx.fresh_temp();
        let result_idx = graph.add_tensor(result_name);

        let node = EinsumNode::new(
            spec,
            vec![left_state.tensor_idx, right_state.tensor_idx],
            vec![result_idx],
        );
        graph.add_node(node)?;

        return Ok(CompileState {
            tensor_idx: result_idx,
            axes: output_axes,
        });
    }

    // For other strategies or scalars, use broadcasting + strategy operation
    let mut left_state = left_state;
    let mut right_state = right_state;

    // If axes differ, broadcast them to the same shape first
    if !left_state.axes.is_empty()
        && !right_state.axes.is_empty()
        && left_state.axes != right_state.axes
    {
        // Broadcast left if needed
        if left_state.axes != output_axes {
            let broadcast_spec = format!("{}->{}", left_state.axes, output_axes);
            let broadcast_name = ctx.fresh_temp();
            let broadcast_idx = graph.add_tensor(broadcast_name);
            let broadcast_node = EinsumNode::new(
                broadcast_spec,
                vec![left_state.tensor_idx],
                vec![broadcast_idx],
            );
            graph.add_node(broadcast_node)?;
            left_state = CompileState {
                tensor_idx: broadcast_idx,
                axes: output_axes.clone(),
            };
        }

        // Broadcast right if needed
        if right_state.axes != output_axes {
            let broadcast_spec = format!("{}->{}", right_state.axes, output_axes);
            let broadcast_name = ctx.fresh_temp();
            let broadcast_idx = graph.add_tensor(broadcast_name);
            let broadcast_node = EinsumNode::new(
                broadcast_spec,
                vec![right_state.tensor_idx],
                vec![broadcast_idx],
            );
            graph.add_node(broadcast_node)?;
            right_state = CompileState {
                tensor_idx: broadcast_idx,
                axes: output_axes.clone(),
            };
        }
    }

    // Now apply the AND strategy
    let result_idx = strategy_mapping::compile_and_with_strategy(
        left_state.tensor_idx,
        right_state.tensor_idx,
        ctx,
        graph,
    )?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

pub(crate) fn compile_or(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let mut left_state = compile_expr(left, ctx, graph)?;
    let mut right_state = compile_expr(right, ctx, graph)?;

    // If either operand is a scalar (empty axes), elem_binary will handle broadcasting
    // No need to do axis alignment via einsum
    if !left_state.axes.is_empty()
        && !right_state.axes.is_empty()
        && left_state.axes != right_state.axes
    {
        // Both operands have axes but they differ - align them using einsum broadcasting
        // Compute the union of all axes, maintaining consistent ordering
        let mut output_axes = String::new();
        let mut seen = std::collections::HashSet::new();

        // Add all axes from left first
        for c in left_state.axes.chars() {
            if seen.insert(c) {
                output_axes.push(c);
            }
        }

        // Add axes from right that aren't in left
        for c in right_state.axes.chars() {
            if seen.insert(c) {
                output_axes.push(c);
            }
        }

        // Broadcast left to the common shape if needed
        if left_state.axes != output_axes {
            let broadcast_spec = format!("{}->{}", left_state.axes, output_axes);
            let broadcast_name = ctx.fresh_temp();
            let broadcast_idx = graph.add_tensor(broadcast_name);
            let broadcast_node = EinsumNode::new(
                broadcast_spec,
                vec![left_state.tensor_idx],
                vec![broadcast_idx],
            );
            graph.add_node(broadcast_node)?;

            left_state = CompileState {
                tensor_idx: broadcast_idx,
                axes: output_axes.clone(),
            };
        }

        // Broadcast right to the common shape if needed
        if right_state.axes != output_axes {
            let broadcast_spec = format!("{}->{}", right_state.axes, output_axes);
            let broadcast_name = ctx.fresh_temp();
            let broadcast_idx = graph.add_tensor(broadcast_name);
            let broadcast_node = EinsumNode::new(
                broadcast_spec,
                vec![right_state.tensor_idx],
                vec![broadcast_idx],
            );
            graph.add_node(broadcast_node)?;

            right_state = CompileState {
                tensor_idx: broadcast_idx,
                axes: output_axes.clone(),
            };
        }
    }

    // Compute output axes (same logic as AND)
    let mut output_axes = String::new();
    let mut seen = std::collections::HashSet::new();

    for c in left_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }

    for c in right_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }

    // Now apply the OR strategy
    let result_idx = strategy_mapping::compile_or_with_strategy(
        left_state.tensor_idx,
        right_state.tensor_idx,
        ctx,
        graph,
    )?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

pub(crate) fn compile_not(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let inner_state = compile_expr(inner, ctx, graph)?;

    // Apply the NOT strategy
    let result_idx =
        strategy_mapping::compile_not_with_strategy(inner_state.tensor_idx, ctx, graph)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: inner_state.axes,
    })
}
