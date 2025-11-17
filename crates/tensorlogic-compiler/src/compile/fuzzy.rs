//! Fuzzy logic operator compilation.
//!
//! This module implements compilation of fuzzy logic operators including:
//! - T-norms (fuzzy AND)
//! - T-conorms (fuzzy OR)
//! - Fuzzy negations
//! - Fuzzy implications
//!
//! All operators are compiled using correct EinsumNode API patterns.

use anyhow::{bail, Result};
use tensorlogic_ir::{
    EinsumGraph, EinsumNode, FuzzyImplicationKind, FuzzyNegationKind, TCoNormKind, TLExpr,
    TNormKind,
};

use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

/// Helper function to get or create a constant tensor.
fn get_or_create_const(value: f64, graph: &mut EinsumGraph) -> usize {
    let const_name = format!("const_{}", value);
    if let Some(pos) = graph.tensors.iter().position(|t| t == &const_name) {
        pos
    } else {
        graph.add_tensor(const_name)
    }
}

/// Compile a T-norm (fuzzy AND) operation.
///
/// T-norms are generalizations of logical AND to the fuzzy domain [0,1].
/// Common t-norms include minimum, product, and Łukasiewicz.
pub(crate) fn compile_tnorm(
    kind: TNormKind,
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let left_state = compile_expr(left, ctx, graph)?;
    let right_state = compile_expr(right, ctx, graph)?;

    let left_idx = left_state.tensor_idx;
    let right_idx = right_state.tensor_idx;

    // Determine output axes: union of left and right axes
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

    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    match kind {
        TNormKind::Minimum => {
            // min(a, b): use element-wise min operation
            let node = EinsumNode::elem_binary("min", left_idx, right_idx, result_idx);
            graph.add_node(node)?;
        }
        TNormKind::Product => {
            // a * b: use element-wise multiplication
            let node = EinsumNode::elem_binary("multiply", left_idx, right_idx, result_idx);
            graph.add_node(node)?;
        }
        TNormKind::Lukasiewicz => {
            // max(0, a + b - 1)
            // Step 1: a + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", left_idx, right_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Step 2: (a + b) - 1
            let one_idx = get_or_create_const(1.0, graph);
            let sub_name = ctx.fresh_temp();
            let sub_idx = graph.add_tensor(sub_name);
            let sub_node = EinsumNode::elem_binary("subtract", sum_idx, one_idx, sub_idx);
            graph.add_node(sub_node)?;

            // Step 3: max(0, (a + b) - 1) using ReLU
            let node = EinsumNode::elem_unary("relu", sub_idx, result_idx);
            graph.add_node(node)?;
        }
        TNormKind::Drastic => {
            // Drastic t-norm: T(a,b) = { b if a=1, a if b=1, 0 otherwise }
            bail!("Drastic t-norm requires conditional logic that is complex to implement efficiently. Use Minimum or Product t-norm instead.")
        }
        TNormKind::NilpotentMinimum => {
            // Nilpotent minimum: T(a,b) = { min(a,b) if a+b>1, 0 otherwise }
            // Implementation: if(a+b>1, min(a,b), 0)

            // Step 1: a + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", left_idx, right_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Step 2: min(a, b)
            let min_name = ctx.fresh_temp();
            let min_idx = graph.add_tensor(min_name);
            let min_node = EinsumNode::elem_binary("min", left_idx, right_idx, min_idx);
            graph.add_node(min_node)?;

            // Step 3: a+b > 1 (condition as soft indicator)
            let one_idx = get_or_create_const(1.0, graph);
            let cond_name = ctx.fresh_temp();
            let cond_idx = graph.add_tensor(cond_name);
            let cond_node = EinsumNode::elem_binary("greater_than", sum_idx, one_idx, cond_idx);
            graph.add_node(cond_node)?;

            // Step 4: cond * min(a,b)
            let node = EinsumNode::elem_binary("multiply", cond_idx, min_idx, result_idx);
            graph.add_node(node)?;
        }
        TNormKind::Hamacher => {
            // Hamacher product: T(a,b) = ab/(a+b-ab) for a,b > 0

            // Step 1: a * b (numerator)
            let prod_name = ctx.fresh_temp();
            let prod_idx = graph.add_tensor(prod_name);
            let prod_node = EinsumNode::elem_binary("multiply", left_idx, right_idx, prod_idx);
            graph.add_node(prod_node)?;

            // Step 2: a + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", left_idx, right_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Step 3: (a + b) - (a * b) (denominator)
            let denom_name = ctx.fresh_temp();
            let denom_idx = graph.add_tensor(denom_name);
            let denom_node = EinsumNode::elem_binary("subtract", sum_idx, prod_idx, denom_idx);
            graph.add_node(denom_node)?;

            // Step 4: ab / (a + b - ab)
            let node = EinsumNode::elem_binary("divide", prod_idx, denom_idx, result_idx);
            graph.add_node(node)?;
        }
    }

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Compile a T-conorm (fuzzy OR) operation.
///
/// T-conorms are duals of t-norms, generalizing logical OR to [0,1].
pub(crate) fn compile_tconorm(
    kind: TCoNormKind,
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let left_state = compile_expr(left, ctx, graph)?;
    let right_state = compile_expr(right, ctx, graph)?;

    let left_idx = left_state.tensor_idx;
    let right_idx = right_state.tensor_idx;

    // Output axes: union of left and right
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

    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    match kind {
        TCoNormKind::Maximum => {
            // max(a, b)
            let node = EinsumNode::elem_binary("max", left_idx, right_idx, result_idx);
            graph.add_node(node)?;
        }
        TCoNormKind::ProbabilisticSum => {
            // a + b - a*b
            // Step 1: a + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", left_idx, right_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Step 2: a * b
            let prod_name = ctx.fresh_temp();
            let prod_idx = graph.add_tensor(prod_name);
            let prod_node = EinsumNode::elem_binary("multiply", left_idx, right_idx, prod_idx);
            graph.add_node(prod_node)?;

            // Step 3: (a + b) - (a * b)
            let node = EinsumNode::elem_binary("subtract", sum_idx, prod_idx, result_idx);
            graph.add_node(node)?;
        }
        TCoNormKind::BoundedSum => {
            // min(1, a + b)
            // Step 1: a + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", left_idx, right_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Step 2: min(1, sum)
            let one_idx = get_or_create_const(1.0, graph);
            let node = EinsumNode::elem_binary("min", one_idx, sum_idx, result_idx);
            graph.add_node(node)?;
        }
        TCoNormKind::Drastic => {
            bail!("Drastic t-conorm requires complex conditional logic. Use Maximum or ProbabilisticSum instead.")
        }
        TCoNormKind::NilpotentMaximum => {
            // Nilpotent maximum: S(a,b) = { max(a,b) if a+b<1, 1 otherwise }

            // Step 1: a + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", left_idx, right_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Step 2: max(a, b)
            let max_name = ctx.fresh_temp();
            let max_idx = graph.add_tensor(max_name);
            let max_node = EinsumNode::elem_binary("max", left_idx, right_idx, max_idx);
            graph.add_node(max_node)?;

            // Step 3: a+b < 1 (condition)
            let one_idx = get_or_create_const(1.0, graph);
            let cond_name = ctx.fresh_temp();
            let cond_idx = graph.add_tensor(cond_name);
            let cond_node = EinsumNode::elem_binary("less_than", sum_idx, one_idx, cond_idx);
            graph.add_node(cond_node)?;

            // Step 4: (1 - cond)
            let one_minus_cond_name = ctx.fresh_temp();
            let one_minus_cond_idx = graph.add_tensor(one_minus_cond_name);
            let one_minus_node =
                EinsumNode::elem_binary("subtract", one_idx, cond_idx, one_minus_cond_idx);
            graph.add_node(one_minus_node)?;

            // Step 5: cond * max(a,b)
            let term1_name = ctx.fresh_temp();
            let term1_idx = graph.add_tensor(term1_name);
            let term1_node = EinsumNode::elem_binary("multiply", cond_idx, max_idx, term1_idx);
            graph.add_node(term1_node)?;

            // Step 6: term1 + (1-cond)
            let node = EinsumNode::elem_binary("add", term1_idx, one_minus_cond_idx, result_idx);
            graph.add_node(node)?;
        }
        TCoNormKind::Hamacher => {
            // Hamacher sum: S(a,b) = (a + b - 2ab) / (1 - ab)

            // Step 1: a * b
            let prod_name = ctx.fresh_temp();
            let prod_idx = graph.add_tensor(prod_name);
            let prod_node = EinsumNode::elem_binary("multiply", left_idx, right_idx, prod_idx);
            graph.add_node(prod_node)?;

            // Step 2: 2 * ab
            let two_idx = get_or_create_const(2.0, graph);
            let two_prod_name = ctx.fresh_temp();
            let two_prod_idx = graph.add_tensor(two_prod_name);
            let two_prod_node =
                EinsumNode::elem_binary("multiply", two_idx, prod_idx, two_prod_idx);
            graph.add_node(two_prod_node)?;

            // Step 3: a + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", left_idx, right_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Step 4: (a + b) - 2ab (numerator)
            let numer_name = ctx.fresh_temp();
            let numer_idx = graph.add_tensor(numer_name);
            let numer_node = EinsumNode::elem_binary("subtract", sum_idx, two_prod_idx, numer_idx);
            graph.add_node(numer_node)?;

            // Step 5: 1 - ab (denominator)
            let one_idx = get_or_create_const(1.0, graph);
            let denom_name = ctx.fresh_temp();
            let denom_idx = graph.add_tensor(denom_name);
            let denom_node = EinsumNode::elem_binary("subtract", one_idx, prod_idx, denom_idx);
            graph.add_node(denom_node)?;

            // Step 6: numerator / denominator
            let node = EinsumNode::elem_binary("divide", numer_idx, denom_idx, result_idx);
            graph.add_node(node)?;
        }
    }

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Compile a fuzzy negation operation.
pub(crate) fn compile_fuzzy_not(
    kind: FuzzyNegationKind,
    expr: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let state = compile_expr(expr, ctx, graph)?;
    let input_idx = state.tensor_idx;

    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    match kind {
        FuzzyNegationKind::Standard => {
            // 1 - a
            let node = EinsumNode::elem_unary("one_minus", input_idx, result_idx);
            graph.add_node(node)?;
        }
        FuzzyNegationKind::Sugeno { lambda } => {
            // N(a) = (1-a)/(1+λa) for λ > -1
            let lambda_f64 = lambda as f64 / 100.0;

            // Step 1: 1 - a (numerator)
            let one_idx = get_or_create_const(1.0, graph);
            let numer_name = ctx.fresh_temp();
            let numer_idx = graph.add_tensor(numer_name);
            let numer_node = EinsumNode::elem_binary("subtract", one_idx, input_idx, numer_idx);
            graph.add_node(numer_node)?;

            // Step 2: λ * a
            let lambda_idx = get_or_create_const(lambda_f64, graph);
            let lambda_a_name = ctx.fresh_temp();
            let lambda_a_idx = graph.add_tensor(lambda_a_name);
            let lambda_a_node =
                EinsumNode::elem_binary("multiply", lambda_idx, input_idx, lambda_a_idx);
            graph.add_node(lambda_a_node)?;

            // Step 3: 1 + λa (denominator)
            let denom_name = ctx.fresh_temp();
            let denom_idx = graph.add_tensor(denom_name);
            let denom_node = EinsumNode::elem_binary("add", one_idx, lambda_a_idx, denom_idx);
            graph.add_node(denom_node)?;

            // Step 4: (1-a) / (1+λa)
            let node = EinsumNode::elem_binary("divide", numer_idx, denom_idx, result_idx);
            graph.add_node(node)?;
        }
        FuzzyNegationKind::Yager { w } => {
            // N(a) = (1 - a^w)^(1/w) for w > 0
            let w_f64 = w as f64 / 10.0;

            // Step 1: a^w
            let w_idx = get_or_create_const(w_f64, graph);
            let pow_name = ctx.fresh_temp();
            let pow_idx = graph.add_tensor(pow_name);
            let pow_node = EinsumNode::elem_binary("power", input_idx, w_idx, pow_idx);
            graph.add_node(pow_node)?;

            // Step 2: 1 - a^w
            let one_idx = get_or_create_const(1.0, graph);
            let diff_name = ctx.fresh_temp();
            let diff_idx = graph.add_tensor(diff_name);
            let diff_node = EinsumNode::elem_binary("subtract", one_idx, pow_idx, diff_idx);
            graph.add_node(diff_node)?;

            // Step 3: (1 - a^w)^(1/w)
            let inv_w_idx = get_or_create_const(1.0 / w_f64, graph);
            let node = EinsumNode::elem_binary("power", diff_idx, inv_w_idx, result_idx);
            graph.add_node(node)?;
        }
    }

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: state.axes,
    })
}

/// Compile a fuzzy implication operation.
pub(crate) fn compile_fuzzy_implication(
    kind: FuzzyImplicationKind,
    premise: &TLExpr,
    conclusion: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let premise_state = compile_expr(premise, ctx, graph)?;
    let conclusion_state = compile_expr(conclusion, ctx, graph)?;

    let premise_idx = premise_state.tensor_idx;
    let conclusion_idx = conclusion_state.tensor_idx;

    // Output axes: union of premise and conclusion
    let mut output_axes = String::new();
    let mut seen = std::collections::HashSet::new();

    for c in premise_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }
    for c in conclusion_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }

    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    match kind {
        FuzzyImplicationKind::Godel => {
            // I(a,b) = { 1 if a≤b, b otherwise }
            // Soft approximation: (a <= b) * 1 + (a > b) * b

            // Step 1: a <= b (condition)
            let cond_name = ctx.fresh_temp();
            let cond_idx = graph.add_tensor(cond_name);
            let cond_node =
                EinsumNode::elem_binary("less_than_equal", premise_idx, conclusion_idx, cond_idx);
            graph.add_node(cond_node)?;

            // Step 2: 1 - cond
            let one_idx = get_or_create_const(1.0, graph);
            let not_cond_name = ctx.fresh_temp();
            let not_cond_idx = graph.add_tensor(not_cond_name);
            let not_cond_node =
                EinsumNode::elem_binary("subtract", one_idx, cond_idx, not_cond_idx);
            graph.add_node(not_cond_node)?;

            // Step 3: (1-cond) * b
            let term2_name = ctx.fresh_temp();
            let term2_idx = graph.add_tensor(term2_name);
            let term2_node =
                EinsumNode::elem_binary("multiply", not_cond_idx, conclusion_idx, term2_idx);
            graph.add_node(term2_node)?;

            // Step 4: cond + term2
            let node = EinsumNode::elem_binary("add", cond_idx, term2_idx, result_idx);
            graph.add_node(node)?;
        }
        FuzzyImplicationKind::Lukasiewicz => {
            // I(a,b) = min(1, 1-a+b)

            // Step 1: 1 - a
            let one_idx = get_or_create_const(1.0, graph);
            let not_premise_name = ctx.fresh_temp();
            let not_premise_idx = graph.add_tensor(not_premise_name);
            let not_premise_node =
                EinsumNode::elem_binary("subtract", one_idx, premise_idx, not_premise_idx);
            graph.add_node(not_premise_node)?;

            // Step 2: (1-a) + b
            let sum_name = ctx.fresh_temp();
            let sum_idx = graph.add_tensor(sum_name);
            let sum_node = EinsumNode::elem_binary("add", not_premise_idx, conclusion_idx, sum_idx);
            graph.add_node(sum_node)?;

            // Step 3: min(1, sum)
            let node = EinsumNode::elem_binary("min", one_idx, sum_idx, result_idx);
            graph.add_node(node)?;
        }
        FuzzyImplicationKind::Reichenbach => {
            // I(a,b) = 1 - a + ab

            // Step 1: 1 - a
            let one_idx = get_or_create_const(1.0, graph);
            let not_premise_name = ctx.fresh_temp();
            let not_premise_idx = graph.add_tensor(not_premise_name);
            let not_premise_node =
                EinsumNode::elem_binary("subtract", one_idx, premise_idx, not_premise_idx);
            graph.add_node(not_premise_node)?;

            // Step 2: a * b
            let prod_name = ctx.fresh_temp();
            let prod_idx = graph.add_tensor(prod_name);
            let prod_node =
                EinsumNode::elem_binary("multiply", premise_idx, conclusion_idx, prod_idx);
            graph.add_node(prod_node)?;

            // Step 3: (1-a) + ab
            let node = EinsumNode::elem_binary("add", not_premise_idx, prod_idx, result_idx);
            graph.add_node(node)?;
        }
        FuzzyImplicationKind::KleeneDienes => {
            // I(a,b) = max(1-a, b)

            // Step 1: 1 - a
            let one_idx = get_or_create_const(1.0, graph);
            let not_premise_name = ctx.fresh_temp();
            let not_premise_idx = graph.add_tensor(not_premise_name);
            let not_premise_node =
                EinsumNode::elem_binary("subtract", one_idx, premise_idx, not_premise_idx);
            graph.add_node(not_premise_node)?;

            // Step 2: max(1-a, b)
            let node = EinsumNode::elem_binary("max", not_premise_idx, conclusion_idx, result_idx);
            graph.add_node(node)?;
        }
        FuzzyImplicationKind::Rescher => {
            // I(a,b) = { 1 if a≤b, 0 otherwise }
            // This is a crisp implication: (a <= b) as indicator
            let node =
                EinsumNode::elem_binary("less_than_equal", premise_idx, conclusion_idx, result_idx);
            graph.add_node(node)?;
        }
        FuzzyImplicationKind::Goguen => {
            // I(a,b) = { 1 if a≤b, b/a otherwise }

            // Step 1: a <= b
            let cond_name = ctx.fresh_temp();
            let cond_idx = graph.add_tensor(cond_name);
            let cond_node =
                EinsumNode::elem_binary("less_than_equal", premise_idx, conclusion_idx, cond_idx);
            graph.add_node(cond_node)?;

            // Step 2: b / a
            let div_name = ctx.fresh_temp();
            let div_idx = graph.add_tensor(div_name);
            let div_node = EinsumNode::elem_binary("divide", conclusion_idx, premise_idx, div_idx);
            graph.add_node(div_node)?;

            // Step 3: 1 - cond
            let one_idx = get_or_create_const(1.0, graph);
            let not_cond_name = ctx.fresh_temp();
            let not_cond_idx = graph.add_tensor(not_cond_name);
            let not_cond_node =
                EinsumNode::elem_binary("subtract", one_idx, cond_idx, not_cond_idx);
            graph.add_node(not_cond_node)?;

            // Step 4: (1-cond) * (b/a)
            let term2_name = ctx.fresh_temp();
            let term2_idx = graph.add_tensor(term2_name);
            let term2_node = EinsumNode::elem_binary("multiply", not_cond_idx, div_idx, term2_idx);
            graph.add_node(term2_node)?;

            // Step 5: cond + term2
            let node = EinsumNode::elem_binary("add", cond_idx, term2_idx, result_idx);
            graph.add_node(node)?;
        }
    }

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompilerContext;
    use tensorlogic_ir::{TLExpr, Term};

    #[test]
    fn test_compile_tnorm_minimum() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        ctx.bind_var("x", "D").unwrap();
        ctx.assign_axis("x");

        let mut graph = EinsumGraph::new();

        let left = TLExpr::pred("P", vec![Term::var("x")]);
        let right = TLExpr::pred("Q", vec![Term::var("x")]);

        let result = compile_tnorm(TNormKind::Minimum, &left, &right, &mut ctx, &mut graph);
        assert!(result.is_ok());
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_compile_tnorm_product() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        ctx.bind_var("x", "D").unwrap();
        ctx.assign_axis("x");

        let mut graph = EinsumGraph::new();

        let left = TLExpr::pred("P", vec![Term::var("x")]);
        let right = TLExpr::pred("Q", vec![Term::var("x")]);

        let result = compile_tnorm(TNormKind::Product, &left, &right, &mut ctx, &mut graph);
        assert!(result.is_ok());
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_compile_tnorm_lukasiewicz() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        ctx.bind_var("x", "D").unwrap();
        ctx.assign_axis("x");

        let mut graph = EinsumGraph::new();

        let left = TLExpr::pred("P", vec![Term::var("x")]);
        let right = TLExpr::pred("Q", vec![Term::var("x")]);

        let result = compile_tnorm(TNormKind::Lukasiewicz, &left, &right, &mut ctx, &mut graph);
        assert!(result.is_ok());
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_compile_tconorm_maximum() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        ctx.bind_var("x", "D").unwrap();
        ctx.assign_axis("x");

        let mut graph = EinsumGraph::new();

        let left = TLExpr::pred("P", vec![Term::var("x")]);
        let right = TLExpr::pred("Q", vec![Term::var("x")]);

        let result = compile_tconorm(TCoNormKind::Maximum, &left, &right, &mut ctx, &mut graph);
        assert!(result.is_ok());
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_compile_fuzzy_not_standard() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        ctx.bind_var("x", "D").unwrap();
        ctx.assign_axis("x");

        let mut graph = EinsumGraph::new();

        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        let result = compile_fuzzy_not(FuzzyNegationKind::Standard, &expr, &mut ctx, &mut graph);
        assert!(result.is_ok());
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_compile_fuzzy_implication_lukasiewicz() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        ctx.bind_var("x", "D").unwrap();
        ctx.assign_axis("x");

        let mut graph = EinsumGraph::new();

        let premise = TLExpr::pred("P", vec![Term::var("x")]);
        let conclusion = TLExpr::pred("Q", vec![Term::var("x")]);

        let result = compile_fuzzy_implication(
            FuzzyImplicationKind::Lukasiewicz,
            &premise,
            &conclusion,
            &mut ctx,
            &mut graph,
        );
        assert!(result.is_ok());
        assert!(graph.validate().is_ok());
    }
}
