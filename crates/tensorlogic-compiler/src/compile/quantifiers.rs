//! Compilation of quantifiers: EXISTS, FORALL, and aggregation operations.

use anyhow::Result;
use tensorlogic_ir::{AggregateOp, EinsumGraph, EinsumNode, TLExpr};

use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

pub(crate) fn compile_exists(
    var: &str,
    domain: &str,
    body: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    ctx.bind_var(var, domain)?;
    let axis = ctx.assign_axis(var);

    let body_state = compile_expr(body, ctx, graph)?;

    let output_axes: String = body_state.axes.chars().filter(|&c| c != axis).collect();

    let spec = format!("{}->{}", body_state.axes, output_axes);
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);
    let node = EinsumNode::new(spec, vec![body_state.tensor_idx], vec![result_idx]);
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

pub(crate) fn compile_forall(
    var: &str,
    domain: &str,
    body: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let negated_body = TLExpr::negate(body.clone());
    let exists_not = TLExpr::exists(var, domain, negated_body);
    let forall_expr = TLExpr::negate(exists_not);

    compile_expr(&forall_expr, ctx, graph)
}

/// Compile aggregate operations (Count, Sum, Average, Max, Min, Product).
pub(crate) fn compile_aggregate(
    op: &AggregateOp,
    var: &str,
    domain: &str,
    body: &TLExpr,
    _group_by: &Option<Vec<String>>,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    ctx.bind_var(var, domain)?;
    let axis = ctx.assign_axis(var);

    let body_state = compile_expr(body, ctx, graph)?;

    // Determine output axes (all axes except the reduction axis)
    let output_axes: String = body_state.axes.chars().filter(|&c| c != axis).collect();

    // Map aggregate operation to reduction operation string
    let reduce_op = match op {
        AggregateOp::Count => "count",
        AggregateOp::Sum => "sum",
        AggregateOp::Average => "mean",
        AggregateOp::Max => "max",
        AggregateOp::Min => "min",
        AggregateOp::Product => "prod",
        AggregateOp::Any => "any",
        AggregateOp::All => "all",
    };

    // Create reduction node
    let spec = format!("{}({}->{})", reduce_op, body_state.axes, output_axes);
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);
    let node = EinsumNode::new(spec, vec![body_state.tensor_idx], vec![result_idx]);
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Compile soft existential quantifier with temperature parameter.
///
/// Soft exists uses log-sum-exp for smooth, differentiable aggregation:
/// - Low temperature (→0): approaches hard max (standard exists)
/// - High temperature: smoother aggregation
pub(crate) fn compile_soft_exists(
    var: &str,
    domain: &str,
    body: &TLExpr,
    temperature: f64,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    ctx.bind_var(var, domain)?;
    let axis = ctx.assign_axis(var);

    let body_state = compile_expr(body, ctx, graph)?;

    let output_axes_vec: Vec<char> = body_state.axes.chars().filter(|&c| c != axis).collect();
    let output_axes: String = output_axes_vec.iter().collect();

    // Implementation using log-sum-exp:
    // logsumexp(x) = log(sum(exp(x)))
    // For numerical stability: logsumexp(x) = log(sum(exp(x - max(x)))) + max(x)

    if temperature.abs() < 1e-6 {
        // Temperature ≈ 0: use hard max (standard exists)
        let result_name = ctx.fresh_temp();
        let result_idx = graph.add_tensor(result_name);
        let node = EinsumNode::reduce(
            "max",
            vec![axis as u8 as usize],
            body_state.tensor_idx,
            result_idx,
        );
        graph.add_node(node)?;

        return Ok(CompileState {
            tensor_idx: result_idx,
            axes: output_axes,
        });
    }

    // Step 1: Scale by temperature: x / T
    let temp_name = format!("const_{}", temperature);
    let temp_idx = if !graph.tensors.contains(&temp_name) {
        graph.add_tensor(temp_name.clone())
    } else {
        graph.tensors.iter().position(|t| t == &temp_name).unwrap()
    };

    let scaled_idx = graph.add_tensor(format!("scaled_{}", graph.tensors.len()));
    let node = EinsumNode::elem_binary("divide", body_state.tensor_idx, temp_idx, scaled_idx);
    graph.add_node(node)?;

    // Step 2: Find max for numerical stability
    let max_idx = graph.add_tensor(format!("soft_exists_max_{}", graph.tensors.len()));
    let node = EinsumNode::reduce("max", vec![axis as u8 as usize], scaled_idx, max_idx);
    graph.add_node(node)?;

    // Step 3: Subtract max: (x/T - max)
    let centered_idx = graph.add_tensor(format!("centered_{}", graph.tensors.len()));
    // Broadcasting: if output_axes is empty (scalar max), subtract directly; otherwise need einsum
    if output_axes.is_empty() {
        // max is scalar, can use elem_binary
        let node = EinsumNode::elem_binary("subtract", scaled_idx, max_idx, centered_idx);
        graph.add_node(node)?;
    } else {
        // Need broadcasting via einsum
        let sub_spec = format!("{},{}->{}", body_state.axes, output_axes, body_state.axes);
        let node = EinsumNode::new(sub_spec, vec![scaled_idx, max_idx], vec![centered_idx]);
        graph.add_node(node)?;
    }

    // Step 4: exp(centered)
    let exp_idx = graph.add_tensor(format!("exp_{}", graph.tensors.len()));
    let node = EinsumNode::elem_unary("exp", centered_idx, exp_idx);
    graph.add_node(node)?;

    // Step 5: sum(exp(centered))
    let sum_idx = graph.add_tensor(format!("sum_exp_{}", graph.tensors.len()));
    let node = EinsumNode::reduce("sum", vec![axis as u8 as usize], exp_idx, sum_idx);
    graph.add_node(node)?;

    // Step 6: log(sum)
    let log_idx = graph.add_tensor(format!("log_sum_{}", graph.tensors.len()));
    let node = EinsumNode::elem_unary("log", sum_idx, log_idx);
    graph.add_node(node)?;

    // Step 7: log(sum) + max
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);
    let node = EinsumNode::elem_binary("add", log_idx, max_idx, result_idx);
    graph.add_node(node)?;

    // Step 8: Multiply by temperature to get final result
    let final_idx = graph.add_tensor(format!("soft_exists_final_{}", graph.tensors.len()));
    let node = EinsumNode::elem_binary("multiply", result_idx, temp_idx, final_idx);
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: final_idx,
        axes: output_axes,
    })
}

/// Compile soft universal quantifier with temperature parameter.
///
/// Soft forall is implemented as the dual of soft exists:
/// SoftForAll(x, P(x), T) = -SoftExists(x, -P(x), T)
pub(crate) fn compile_soft_forall(
    var: &str,
    domain: &str,
    body: &TLExpr,
    temperature: f64,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Compile body
    ctx.bind_var(var, domain)?;
    let axis = ctx.assign_axis(var);

    let body_state = compile_expr(body, ctx, graph)?;

    // Step 1: Negate body: -P(x)
    let neg_idx = graph.add_tensor(format!("soft_forall_neg_{}", graph.tensors.len()));
    let node = EinsumNode::elem_unary("negate", body_state.tensor_idx, neg_idx);
    graph.add_node(node)?;

    let output_axes_vec: Vec<char> = body_state.axes.chars().filter(|&c| c != axis).collect();
    let output_axes: String = output_axes_vec.iter().collect();

    // Step 2: Apply log-sum-exp to -P(x) (same as soft_exists implementation)
    if temperature.abs() < 1e-6 {
        // Temperature ≈ 0: use hard min (via -max(-x))
        let max_idx = graph.add_tensor(format!("soft_forall_max_{}", graph.tensors.len()));
        let node = EinsumNode::reduce("max", vec![axis as u8 as usize], neg_idx, max_idx);
        graph.add_node(node)?;

        // Negate max to get min
        let result_name = ctx.fresh_temp();
        let result_idx = graph.add_tensor(result_name);
        let node = EinsumNode::elem_unary("negate", max_idx, result_idx);
        graph.add_node(node)?;

        return Ok(CompileState {
            tensor_idx: result_idx,
            axes: output_axes,
        });
    }

    // Apply log-sum-exp on negated values
    let temp_name = format!("const_{}", temperature);
    let temp_idx = if !graph.tensors.contains(&temp_name) {
        graph.add_tensor(temp_name.clone())
    } else {
        graph.tensors.iter().position(|t| t == &temp_name).unwrap()
    };

    let scaled_idx = graph.add_tensor(format!("scaled_{}", graph.tensors.len()));
    let node = EinsumNode::elem_binary("divide", neg_idx, temp_idx, scaled_idx);
    graph.add_node(node)?;

    let max_idx = graph.add_tensor(format!("soft_forall_max_{}", graph.tensors.len()));
    let node = EinsumNode::reduce("max", vec![axis as u8 as usize], scaled_idx, max_idx);
    graph.add_node(node)?;

    let centered_idx = graph.add_tensor(format!("centered_{}", graph.tensors.len()));
    if output_axes.is_empty() {
        let node = EinsumNode::elem_binary("subtract", scaled_idx, max_idx, centered_idx);
        graph.add_node(node)?;
    } else {
        let sub_spec = format!("{},{}->{}", body_state.axes, output_axes, body_state.axes);
        let node = EinsumNode::new(sub_spec, vec![scaled_idx, max_idx], vec![centered_idx]);
        graph.add_node(node)?;
    }

    let exp_idx = graph.add_tensor(format!("exp_{}", graph.tensors.len()));
    let node = EinsumNode::elem_unary("exp", centered_idx, exp_idx);
    graph.add_node(node)?;

    let sum_idx = graph.add_tensor(format!("sum_exp_{}", graph.tensors.len()));
    let node = EinsumNode::reduce("sum", vec![axis as u8 as usize], exp_idx, sum_idx);
    graph.add_node(node)?;

    let log_idx = graph.add_tensor(format!("log_sum_{}", graph.tensors.len()));
    let node = EinsumNode::elem_unary("log", sum_idx, log_idx);
    graph.add_node(node)?;

    let lse_idx = graph.add_tensor(format!("lse_{}", graph.tensors.len()));
    let node = EinsumNode::elem_binary("add", log_idx, max_idx, lse_idx);
    graph.add_node(node)?;

    let scaled_lse_idx = graph.add_tensor(format!("scaled_lse_{}", graph.tensors.len()));
    let node = EinsumNode::elem_binary("multiply", lse_idx, temp_idx, scaled_lse_idx);
    graph.add_node(node)?;

    // Step 3: Negate result: -lse(-P(x)/T) = SoftForAll(P(x))
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);
    let node = EinsumNode::elem_unary("negate", scaled_lse_idx, result_idx);
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}
