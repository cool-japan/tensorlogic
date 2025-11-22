//! Counting quantifier compilation.
//!
//! Implements compilation for counting quantifiers that specify
//! exact or minimum counts of satisfying elements:
//! - CountingExists: ∃≥k x. P(x) - at least k elements satisfy P
//! - CountingForAll: ∀≥k x. P(x) - at least k elements satisfy P
//! - ExactCount: ∃=k x. P(x) - exactly k elements satisfy P
//! - Majority: Majority x. P(x) - more than half satisfy P

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr};

use crate::compile::compile_expr;
use crate::context::{CompileState, CompilerContext};

/// Compile CountingExists: ∃≥k x. P(x)
/// Strategy: sum(P(x)) >= k encoded as sigmoid(sum(P(x)) - k + 0.5)
pub(crate) fn compile_counting_exists(
    var: &str,
    domain: &str,
    body: &TLExpr,
    min_count: usize,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Bind variable and assign axis
    ctx.bind_var(var, domain)?;
    let axis = ctx.assign_axis(var);

    // Compile the body
    let body_state = compile_expr(body, ctx, graph)?;

    // Build output axes (without quantified axis)
    let output_axes: String = body_state.axes.chars().filter(|&c| c != axis).collect();

    // Create sum reduction over quantified axis
    let sum_spec = format!("sum({}->{})", body_state.axes, output_axes);
    let sum_name = ctx.fresh_temp();
    let sum_idx = graph.add_tensor(sum_name);
    let sum_node = EinsumNode::new(sum_spec, vec![body_state.tensor_idx], vec![sum_idx]);
    graph.add_node(sum_node)?;

    // Create constant for threshold (min_count - 0.5)
    let threshold = (min_count as f64) - 0.5;
    let threshold_name = format!("const_{}", threshold);
    let threshold_idx = graph.add_tensor(&threshold_name);
    // Note: Backend will initialize this as a constant with value `threshold`

    // Subtract: sum - threshold
    let diff_spec = format!("subtract({},{}->{})", output_axes, "", output_axes);
    let diff_name = ctx.fresh_temp();
    let diff_idx = graph.add_tensor(diff_name);
    let diff_node = EinsumNode::new(diff_spec, vec![sum_idx, threshold_idx], vec![diff_idx]);
    graph.add_node(diff_node)?;

    // Apply sigmoid for smooth threshold
    let sigmoid_spec = format!("sigmoid({}->{})", output_axes, output_axes);
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);
    let sigmoid_node = EinsumNode::new(sigmoid_spec, vec![diff_idx], vec![result_idx]);
    graph.add_node(sigmoid_node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Compile CountingForAll: ∀≥k x. P(x)
/// Same semantics as CountingExists for "at least k"
pub(crate) fn compile_counting_forall(
    var: &str,
    domain: &str,
    body: &TLExpr,
    min_count: usize,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    compile_counting_exists(var, domain, body, min_count, ctx, graph)
}

/// Compile ExactCount: ∃=k x. P(x)
/// Strategy: exp(-|sum(P(x)) - k|^2) for exact matching
pub(crate) fn compile_exact_count(
    var: &str,
    domain: &str,
    body: &TLExpr,
    count: usize,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Bind variable and assign axis
    ctx.bind_var(var, domain)?;
    let axis = ctx.assign_axis(var);

    // Compile body
    let body_state = compile_expr(body, ctx, graph)?;

    // Build output axes
    let output_axes: String = body_state.axes.chars().filter(|&c| c != axis).collect();

    // Sum reduction
    let sum_spec = format!("sum({}->{})", body_state.axes, output_axes);
    let sum_name = ctx.fresh_temp();
    let sum_idx = graph.add_tensor(sum_name);
    let sum_node = EinsumNode::new(sum_spec, vec![body_state.tensor_idx], vec![sum_idx]);
    graph.add_node(sum_node)?;

    // Create constant for target count
    let target = count as f64;
    let target_name = format!("const_{}", target);
    let target_idx = graph.add_tensor(&target_name);
    // Note: Backend will initialize this as a constant with value `target`

    // Subtract: sum - target
    let diff_spec = format!("subtract({},{}->{})", output_axes, "", output_axes);
    let diff_name = ctx.fresh_temp();
    let diff_idx = graph.add_tensor(diff_name);
    let diff_node = EinsumNode::new(diff_spec, vec![sum_idx, target_idx], vec![diff_idx]);
    graph.add_node(diff_node)?;

    // Square: diff * diff
    let sq_spec = format!("multiply({},{}->{})", output_axes, output_axes, output_axes);
    let sq_name = ctx.fresh_temp();
    let sq_idx = graph.add_tensor(sq_name);
    let sq_node = EinsumNode::new(sq_spec, vec![diff_idx, diff_idx], vec![sq_idx]);
    graph.add_node(sq_node)?;

    // Negate
    let neg_spec = format!("negate({}->{})", output_axes, output_axes);
    let neg_name = ctx.fresh_temp();
    let neg_idx = graph.add_tensor(neg_name);
    let neg_node = EinsumNode::new(neg_spec, vec![sq_idx], vec![neg_idx]);
    graph.add_node(neg_node)?;

    // Exponential: exp(-diff^2)
    let exp_spec = format!("exp({}->{})", output_axes, output_axes);
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);
    let exp_node = EinsumNode::new(exp_spec, vec![neg_idx], vec![result_idx]);
    graph.add_node(exp_node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Compile Majority: Majority x. P(x)
/// Strategy: sum(P(x)) > |domain|/2 encoded as sigmoid(sum - |domain|/2)
pub(crate) fn compile_majority(
    var: &str,
    domain: &str,
    body: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Get domain size
    let domain_info = ctx
        .domains
        .get(domain)
        .ok_or_else(|| anyhow::anyhow!("Domain '{}' not found", domain))?;
    let domain_size = domain_info.cardinality;

    // Bind variable and assign axis
    ctx.bind_var(var, domain)?;
    let axis = ctx.assign_axis(var);

    // Compile body
    let body_state = compile_expr(body, ctx, graph)?;

    // Build output axes
    let output_axes: String = body_state.axes.chars().filter(|&c| c != axis).collect();

    // Sum reduction
    let sum_spec = format!("sum({}->{})", body_state.axes, output_axes);
    let sum_name = ctx.fresh_temp();
    let sum_idx = graph.add_tensor(sum_name);
    let sum_node = EinsumNode::new(sum_spec, vec![body_state.tensor_idx], vec![sum_idx]);
    graph.add_node(sum_node)?;

    // Create constant for half domain size
    let half_size = (domain_size as f64) / 2.0;
    let half_name = format!("const_{}", half_size);
    let half_idx = graph.add_tensor(&half_name);
    // Note: Backend will initialize this as a constant with value `half_size`

    // Subtract: sum - half
    let diff_spec = format!("subtract({},{}->{})", output_axes, "", output_axes);
    let diff_name = ctx.fresh_temp();
    let diff_idx = graph.add_tensor(diff_name);
    let diff_node = EinsumNode::new(diff_spec, vec![sum_idx, half_idx], vec![diff_idx]);
    graph.add_node(diff_node)?;

    // Apply sigmoid
    let sigmoid_spec = format!("sigmoid({}->{})", output_axes, output_axes);
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);
    let sigmoid_node = EinsumNode::new(sigmoid_spec, vec![diff_idx], vec![result_idx]);
    graph.add_node(sigmoid_node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_counting_exists_compilation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 10);

        let body = TLExpr::pred("happy", vec![Term::var("x")]);
        let mut graph = EinsumGraph::default();

        let result = compile_counting_exists("x", "Person", &body, 3, &mut ctx, &mut graph);
        assert!(result.is_ok());

        let state = result.unwrap();
        // Should have no axes (quantified variable removed)
        assert!(state.axes.is_empty());
        // Should have multiple nodes
        assert!(graph.nodes.len() >= 3);
    }

    #[test]
    fn test_exact_count_compilation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Item", 5);

        let body = TLExpr::pred("selected", vec![Term::var("i")]);
        let mut graph = EinsumGraph::default();

        let result = compile_exact_count("i", "Item", &body, 2, &mut ctx, &mut graph);
        assert!(result.is_ok());

        let state = result.unwrap();
        assert!(state.axes.is_empty());
        assert!(graph.nodes.len() >= 5);
    }

    #[test]
    fn test_majority_compilation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Voter", 100);

        let body = TLExpr::pred("votes_yes", vec![Term::var("v")]);
        let mut graph = EinsumGraph::default();

        let result = compile_majority("v", "Voter", &body, &mut ctx, &mut graph);
        assert!(result.is_ok());

        let state = result.unwrap();
        assert!(state.axes.is_empty());
        assert!(graph.nodes.len() >= 3);
    }

    #[test]
    fn test_counting_forall_compilation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Student", 20);

        let body = TLExpr::pred("passed", vec![Term::var("s")]);
        let mut graph = EinsumGraph::default();

        let result = compile_counting_forall("s", "Student", &body, 15, &mut ctx, &mut graph);
        assert!(result.is_ok());

        let state = result.unwrap();
        assert!(state.axes.is_empty());
    }
}
