//! Compilation of implication (→).

use anyhow::{bail, Result};
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr};

use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

pub(crate) fn compile_imply(
    premise: &TLExpr,
    conclusion: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let mut premise_state = compile_expr(premise, ctx, graph)?;
    let mut conclusion_state = compile_expr(conclusion, ctx, graph)?;

    // Align axes between premise and conclusion
    // Strategy:
    // 1. If premise has extra axes → marginalize them out (∀ quantification)
    // 2. If conclusion has extra axes → broadcast premise to match (independent axes)
    // 3. Result has union of all axes from both operands

    if premise_state.axes != conclusion_state.axes {
        // Find axes in premise not in conclusion (need to marginalize)
        let premise_extra: Vec<char> = premise_state
            .axes
            .chars()
            .filter(|c| !conclusion_state.axes.contains(*c))
            .collect();

        // Find axes in conclusion not in premise (need to broadcast)
        let conclusion_extra: Vec<char> = conclusion_state
            .axes
            .chars()
            .filter(|c| !premise_state.axes.contains(*c))
            .collect();

        // Marginalize extra premise axes
        if !premise_extra.is_empty() {
            let axes_to_reduce: Vec<usize> = premise_extra
                .iter()
                .map(|c| premise_state.axes.find(*c).unwrap())
                .collect();

            let reduce_name = ctx.fresh_temp();
            let reduce_idx = graph.add_tensor(reduce_name);
            let reduce_node =
                EinsumNode::reduce("sum", axes_to_reduce, premise_state.tensor_idx, reduce_idx);
            graph.add_node(reduce_node)?;

            let new_axes: String = premise_state
                .axes
                .chars()
                .filter(|c| !premise_extra.contains(c))
                .collect();

            premise_state = CompileState {
                tensor_idx: reduce_idx,
                axes: new_axes,
            };
        }

        // Broadcast premise to include conclusion's extra axes
        if !conclusion_extra.is_empty() {
            // Use einsum to broadcast: "ab->abc" where c is the extra axis
            let mut output_axes = premise_state.axes.clone();
            for &c in &conclusion_extra {
                output_axes.push(c);
            }

            let broadcast_spec = format!("{}->{}", premise_state.axes, output_axes);
            let broadcast_name = ctx.fresh_temp();
            let broadcast_idx = graph.add_tensor(broadcast_name);
            let broadcast_node = EinsumNode::new(
                broadcast_spec,
                vec![premise_state.tensor_idx],
                vec![broadcast_idx],
            );
            graph.add_node(broadcast_node)?;

            premise_state = CompileState {
                tensor_idx: broadcast_idx,
                axes: output_axes,
            };
        }

        // Similarly broadcast conclusion if needed
        let premise_remaining: Vec<char> = premise_state
            .axes
            .chars()
            .filter(|c| !conclusion_state.axes.contains(*c))
            .collect();

        if !premise_remaining.is_empty() {
            let mut output_axes = conclusion_state.axes.clone();
            for &c in &premise_remaining {
                output_axes.push(c);
            }

            let broadcast_spec = format!("{}->{}", conclusion_state.axes, output_axes);
            let broadcast_name = ctx.fresh_temp();
            let broadcast_idx = graph.add_tensor(broadcast_name);
            let broadcast_node = EinsumNode::new(
                broadcast_spec,
                vec![conclusion_state.tensor_idx],
                vec![broadcast_idx],
            );
            graph.add_node(broadcast_node)?;

            conclusion_state = CompileState {
                tensor_idx: broadcast_idx,
                axes: output_axes,
            };
        }

        // Final check: axes should now be aligned
        if premise_state.axes != conclusion_state.axes {
            bail!(
                "IMPLY operands have incompatible axes after alignment: '{}' vs '{}'",
                premise_state.axes,
                conclusion_state.axes
            );
        }
    }

    // Implication: a → b == ReLU(b - a)
    // Step 1: Subtract (conclusion - premise)
    let subtract_name = ctx.fresh_temp();
    let subtract_idx = graph.add_tensor(subtract_name);
    let subtract_node = EinsumNode::elem_binary(
        "subtract",
        conclusion_state.tensor_idx,
        premise_state.tensor_idx,
        subtract_idx,
    );
    graph.add_node(subtract_node)?;

    // Step 2: Apply ReLU
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);
    let relu_node = EinsumNode::elem_unary("relu", subtract_idx, result_idx);
    graph.add_node(relu_node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: premise_state.axes,
    })
}
