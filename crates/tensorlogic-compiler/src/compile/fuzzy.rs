//! Fuzzy logic operator compilation.
//!
//! This module implements compilation of fuzzy logic operators including:
//! - T-norms (fuzzy AND)
//! - T-conorms (fuzzy OR)
//! - Fuzzy negations
//! - Fuzzy implications

use anyhow::{bail, Result};
use tensorlogic_ir::{
    EinsumGraph, EinsumNode, FuzzyImplicationKind, FuzzyNegationKind, TCoNormKind, TLExpr,
    TNormKind,
};

use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

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
    let mut output_axes = left_state.axes.clone();
    for axis in &right_state.axes {
        if !output_axes.contains(axis) {
            output_axes.push(*axis);
        }
    }
    output_axes.sort_unstable();

    // Create einsum spec for broadcasting
    let left_spec = left_state
        .axes
        .iter()
        .map(|a| format!("{}", ('a' as u8 + *a as u8) as char))
        .collect::<Vec<_>>()
        .join("");
    let right_spec = right_state
        .axes
        .iter()
        .map(|a| format!("{}", ('a' as u8 + *a as u8) as char))
        .collect::<Vec<_>>()
        .join("");
    let out_spec = output_axes
        .iter()
        .map(|a| format!("{}", ('a' as u8 + *a as u8) as char))
        .collect::<Vec<_>>()
        .join("");

    let result_idx = graph.add_tensor(&format!("tnorm_{}_{}", kind_name(kind), graph.tensors.len()));

    match kind {
        TNormKind::Minimum => {
            // min(a, b): use element-wise min operation
            let einsum_spec = format!("{},{}->{}", left_spec, right_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "min",
                vec![left_idx, right_idx],
                vec![result_idx],
                Some(einsum_spec),
            ))?;
        }
        TNormKind::Product => {
            // a * b: use element-wise multiplication
            let einsum_spec = format!("{},{}->{}", left_spec, right_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![left_idx, right_idx],
                vec![result_idx],
                Some(einsum_spec),
            ))?;
        }
        TNormKind::Lukasiewicz => {
            // max(0, a + b - 1)
            // Step 1: a + b
            let sum_idx = graph.add_tensor(&format!("lukasiewicz_sum_{}", graph.tensors.len()));
            let einsum_spec = format!("{},{}->{}", left_spec, right_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![left_idx, right_idx],
                vec![sum_idx],
                Some(einsum_spec),
            ))?;

            // Step 2: (a + b) - 1
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let sub_idx = graph.add_tensor(&format!("lukasiewicz_sub_{}", graph.tensors.len()));
            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![sum_idx, one_idx],
                vec![sub_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 3: max(0, (a + b) - 1)
            let zero_idx = graph.add_tensor(&format!("constant_zero_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(0.0, vec![], vec![zero_idx]))?;

            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "max",
                vec![zero_idx, sub_idx],
                vec![result_idx],
                Some(broadcast_spec.to_string()),
            ))?;
        }
        TNormKind::Drastic => {
            // Drastic t-norm: T(a,b) = { b if a=1, a if b=1, 0 otherwise }
            // Implementation: if(abs(a-1)<ε, b, if(abs(b-1)<ε, a, 0))
            bail!("Drastic t-norm requires conditional logic that is complex to implement efficiently. Use Minimum or Product t-norm instead.")
        }
        TNormKind::NilpotentMinimum => {
            // Nilpotent minimum: T(a,b) = { min(a,b) if a+b>1, 0 otherwise }
            // Implementation: if(a+b>1, min(a,b), 0)

            // Step 1: a + b
            let sum_idx = graph.add_tensor(&format!("nilpotent_sum_{}", graph.tensors.len()));
            let einsum_spec = format!("{},{}->{}", left_spec, right_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![left_idx, right_idx],
                vec![sum_idx],
                Some(einsum_spec.clone()),
            ))?;

            // Step 2: min(a, b)
            let min_idx = graph.add_tensor(&format!("nilpotent_min_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "min",
                vec![left_idx, right_idx],
                vec![min_idx],
                Some(einsum_spec),
            ))?;

            // Step 3: a+b > 1 (condition as soft indicator)
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let cond_idx = graph.add_tensor(&format!("nilpotent_cond_{}", graph.tensors.len()));
            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "gt",
                vec![sum_idx, one_idx],
                vec![cond_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 4: cond * min(a,b) + (1-cond) * 0 = cond * min(a,b)
            let out_einsum = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![cond_idx, min_idx],
                vec![result_idx],
                Some(out_einsum.to_string()),
            ))?;
        }
        TNormKind::Hamacher => {
            // Hamacher product: T(a,b) = ab/(a+b-ab) for a,b > 0
            // For numerical stability, we handle division carefully

            // Step 1: a * b (numerator)
            let prod_idx = graph.add_tensor(&format!("hamacher_prod_{}", graph.tensors.len()));
            let einsum_spec = format!("{},{}->{}", left_spec, right_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![left_idx, right_idx],
                vec![prod_idx],
                Some(einsum_spec.clone()),
            ))?;

            // Step 2: a + b
            let sum_idx = graph.add_tensor(&format!("hamacher_sum_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![left_idx, right_idx],
                vec![sum_idx],
                Some(einsum_spec),
            ))?;

            // Step 3: (a + b) - (a * b) (denominator)
            let denom_idx = graph.add_tensor(&format!("hamacher_denom_{}", graph.tensors.len()));
            let denom_einsum = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![sum_idx, prod_idx],
                vec![denom_idx],
                Some(denom_einsum.to_string()),
            ))?;

            // Step 4: ab / (a + b - ab)
            let div_einsum = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "div",
                vec![prod_idx, denom_idx],
                vec![result_idx],
                Some(div_einsum.to_string()),
            ))?;
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
    let mut output_axes = left_state.axes.clone();
    for axis in &right_state.axes {
        if !output_axes.contains(axis) {
            output_axes.push(*axis);
        }
    }
    output_axes.sort_unstable();

    let left_spec = left_state
        .axes
        .iter()
        .map(|a| format!("{}", ('a' as u8 + *a as u8) as char))
        .collect::<Vec<_>>()
        .join("");
    let right_spec = right_state
        .axes
        .iter()
        .map(|a| format!("{}", ('a' as u8 + *a as u8) as char))
        .collect::<Vec<_>>()
        .join("");
    let out_spec = output_axes
        .iter()
        .map(|a| format!("{}", ('a' as u8 + *a as u8) as char))
        .collect::<Vec<_>>()
        .join("");

    let result_idx = graph.add_tensor(&format!("tconorm_{}_{}", tconorm_name(kind), graph.tensors.len()));

    match kind {
        TCoNormKind::Maximum => {
            // max(a, b)
            let einsum_spec = format!("{},{}->{}", left_spec, right_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "max",
                vec![left_idx, right_idx],
                vec![result_idx],
                Some(einsum_spec),
            ))?;
        }
        TCoNormKind::ProbabilisticSum => {
            // a + b - a*b
            // Step 1: a + b
            let sum_idx = graph.add_tensor(&format!("probsum_sum_{}", graph.tensors.len()));
            let einsum_spec = format!("{},{}->{}", left_spec, right_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![left_idx, right_idx],
                vec![sum_idx],
                Some(einsum_spec.clone()),
            ))?;

            // Step 2: a * b
            let prod_idx = graph.add_tensor(&format!("probsum_prod_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![left_idx, right_idx],
                vec![prod_idx],
                Some(einsum_spec),
            ))?;

            // Step 3: (a + b) - (a * b)
            let sub_einsum = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![sum_idx, prod_idx],
                vec![result_idx],
                Some(sub_einsum.to_string()),
            ))?;
        }
        TCoNormKind::BoundedSum => {
            // min(1, a + b) = min(1, a + b)
            // Step 1: a + b
            let sum_idx = graph.add_tensor(&format!("bounded_sum_{}", graph.tensors.len()));
            let einsum_spec = format!("{},{}->{}", left_spec, right_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![left_idx, right_idx],
                vec![sum_idx],
                Some(einsum_spec),
            ))?;

            // Step 2: min(1, sum)
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "min",
                vec![one_idx, sum_idx],
                vec![result_idx],
                Some(broadcast_spec.to_string()),
            ))?;
        }
        TCoNormKind::Drastic => {
            bail!("Drastic t-conorm requires complex conditional logic. Use Maximum or ProbabilisticSum instead.")
        }
        TCoNormKind::NilpotentMaximum => {
            // Nilpotent maximum: S(a,b) = { max(a,b) if a+b<1, 1 otherwise }

            // Step 1: a + b
            let sum_idx = graph.add_tensor(&format!("nilpotent_max_sum_{}", graph.tensors.len()));
            let einsum_spec = format!("{},{}->{}", left_spec, right_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![left_idx, right_idx],
                vec![sum_idx],
                Some(einsum_spec.clone()),
            ))?;

            // Step 2: max(a, b)
            let max_idx = graph.add_tensor(&format!("nilpotent_max_max_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "max",
                vec![left_idx, right_idx],
                vec![max_idx],
                Some(einsum_spec),
            ))?;

            // Step 3: a+b < 1 (condition)
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let cond_idx = graph.add_tensor(&format!("nilpotent_max_cond_{}", graph.tensors.len()));
            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "lt",
                vec![sum_idx, one_idx],
                vec![cond_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 4: cond * max(a,b) + (1-cond) * 1
            // First: (1 - cond)
            let one_minus_cond_idx = graph.add_tensor(&format!("one_minus_cond_{}", graph.tensors.len()));
            let sub_einsum = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![one_idx, cond_idx],
                vec![one_minus_cond_idx],
                Some(sub_einsum.to_string()),
            ))?;

            // cond * max(a,b)
            let term1_idx = graph.add_tensor(&format!("nilpotent_max_term1_{}", graph.tensors.len()));
            let mul_einsum = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![cond_idx, max_idx],
                vec![term1_idx],
                Some(mul_einsum.to_string()),
            ))?;

            // Final: term1 + (1-cond)
            let add_einsum = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![term1_idx, one_minus_cond_idx],
                vec![result_idx],
                Some(add_einsum.to_string()),
            ))?;
        }
        TCoNormKind::Hamacher => {
            // Hamacher sum: dual of Hamacher product
            // S(a,b) = (a + b - 2ab) / (1 - ab)

            // Step 1: a * b
            let prod_idx = graph.add_tensor(&format!("hamacher_s_prod_{}", graph.tensors.len()));
            let einsum_spec = format!("{},{}->{}", left_spec, right_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![left_idx, right_idx],
                vec![prod_idx],
                Some(einsum_spec.clone()),
            ))?;

            // Step 2: 2 * ab
            let two_idx = graph.add_tensor(&format!("constant_two_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(2.0, vec![], vec![two_idx]))?;

            let two_prod_idx = graph.add_tensor(&format!("hamacher_s_2prod_{}", graph.tensors.len()));
            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![two_idx, prod_idx],
                vec![two_prod_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 3: a + b
            let sum_idx = graph.add_tensor(&format!("hamacher_s_sum_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![left_idx, right_idx],
                vec![sum_idx],
                Some(einsum_spec),
            ))?;

            // Step 4: (a + b) - 2ab (numerator)
            let numer_idx = graph.add_tensor(&format!("hamacher_s_numer_{}", graph.tensors.len()));
            let sub_einsum = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![sum_idx, two_prod_idx],
                vec![numer_idx],
                Some(sub_einsum.to_string()),
            ))?;

            // Step 5: 1 - ab (denominator)
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let denom_idx = graph.add_tensor(&format!("hamacher_s_denom_{}", graph.tensors.len()));
            let denom_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![one_idx, prod_idx],
                vec![denom_idx],
                Some(denom_spec.to_string()),
            ))?;

            // Step 6: numerator / denominator
            let div_einsum = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "div",
                vec![numer_idx, denom_idx],
                vec![result_idx],
                Some(div_einsum.to_string()),
            ))?;
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

    let out_spec = state
        .axes
        .iter()
        .map(|a| format!("{}", ('a' as u8 + *a as u8) as char))
        .collect::<Vec<_>>()
        .join("");

    let result_idx = graph.add_tensor(&format!("fuzzy_not_{}", graph.tensors.len()));

    match kind {
        FuzzyNegationKind::Standard => {
            // 1 - a
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![one_idx, input_idx],
                vec![result_idx],
                Some(broadcast_spec.to_string()),
            ))?;
        }
        FuzzyNegationKind::Sugeno { lambda } => {
            // N(a) = (1-a)/(1+λa) for λ > -1
            // Convert lambda from i32 to f64 (stored as lambda/100)
            let lambda_f64 = lambda as f64 / 100.0;

            // Step 1: 1 - a (numerator)
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let numer_idx = graph.add_tensor(&format!("sugeno_numer_{}", graph.tensors.len()));
            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![one_idx, input_idx],
                vec![numer_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 2: λ * a
            let lambda_idx = graph.add_tensor(&format!("lambda_const_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(lambda_f64, vec![], vec![lambda_idx]))?;

            let lambda_a_idx = graph.add_tensor(&format!("sugeno_lambda_a_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![lambda_idx, input_idx],
                vec![lambda_a_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 3: 1 + λa (denominator)
            let denom_idx = graph.add_tensor(&format!("sugeno_denom_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![one_idx, lambda_a_idx],
                vec![denom_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 4: (1-a) / (1+λa)
            let div_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "div",
                vec![numer_idx, denom_idx],
                vec![result_idx],
                Some(div_spec.to_string()),
            ))?;
        }
        FuzzyNegationKind::Yager { w } => {
            // N(a) = (1 - a^w)^(1/w) for w > 0
            // Convert w from u32 to f64 (stored as w/10)
            let w_f64 = w as f64 / 10.0;

            // Step 1: a^w
            let w_idx = graph.add_tensor(&format!("w_const_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(w_f64, vec![], vec![w_idx]))?;

            let pow_idx = graph.add_tensor(&format!("yager_pow_{}", graph.tensors.len()));
            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "pow",
                vec![input_idx, w_idx],
                vec![pow_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 2: 1 - a^w
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let diff_idx = graph.add_tensor(&format!("yager_diff_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![one_idx, pow_idx],
                vec![diff_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 3: (1 - a^w)^(1/w)
            let inv_w_idx = graph.add_tensor(&format!("inv_w_const_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0 / w_f64, vec![], vec![inv_w_idx]))?;

            graph.add_node(EinsumNode::element_wise(
                "pow",
                vec![diff_idx, inv_w_idx],
                vec![result_idx],
                Some(broadcast_spec.to_string()),
            ))?;
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
    let mut output_axes = premise_state.axes.clone();
    for axis in &conclusion_state.axes {
        if !output_axes.contains(axis) {
            output_axes.push(*axis);
        }
    }
    output_axes.sort_unstable();

    let premise_spec = premise_state
        .axes
        .iter()
        .map(|a| format!("{}", ('a' as u8 + *a as u8) as char))
        .collect::<Vec<_>>()
        .join("");
    let conclusion_spec = conclusion_state
        .axes
        .iter()
        .map(|a| format!("{}", ('a' as u8 + *a as u8) as char))
        .collect::<Vec<_>>()
        .join("");
    let out_spec = output_axes
        .iter()
        .map(|a| format!("{}", ('a' as u8 + *a as u8) as char))
        .collect::<Vec<_>>()
        .join("");

    let result_idx = graph.add_tensor(&format!("fuzzy_impl_{}_{}", fuzzy_impl_name(kind), graph.tensors.len()));

    match kind {
        FuzzyImplicationKind::Godel => {
            // I(a,b) = { 1 if a≤b, b otherwise }
            // Approximation: max(1-a+b, b) with clipping
            // Better: use conditional (a <= b) * 1 + (a > b) * b

            // Step 1: a <= b (condition)
            let einsum_spec = format!("{},{}->{}", premise_spec, conclusion_spec, out_spec);
            let cond_idx = graph.add_tensor(&format!("godel_cond_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "lte",
                vec![premise_idx, conclusion_idx],
                vec![cond_idx],
                Some(einsum_spec),
            ))?;

            // Step 2: 1 - cond
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let not_cond_idx = graph.add_tensor(&format!("godel_not_cond_{}", graph.tensors.len()));
            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![one_idx, cond_idx],
                vec![not_cond_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 3: cond * 1 + (1-cond) * b
            let term2_idx = graph.add_tensor(&format!("godel_term2_{}", graph.tensors.len()));
            let mul_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, conclusion_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![not_cond_idx, conclusion_idx],
                vec![term2_idx],
                Some(mul_spec.to_string()),
            ))?;

            let add_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![cond_idx, term2_idx],
                vec![result_idx],
                Some(add_spec.to_string()),
            ))?;
        }
        FuzzyImplicationKind::Lukasiewicz => {
            // I(a,b) = min(1, 1-a+b)

            // Step 1: 1 - a
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let not_premise_idx = graph.add_tensor(&format!("lukasiewicz_not_premise_{}", graph.tensors.len()));
            let broadcast_spec = if premise_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", premise_spec, premise_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![one_idx, premise_idx],
                vec![not_premise_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 2: (1-a) + b
            let einsum_spec = format!("{},{}->{}", premise_spec, conclusion_spec, out_spec);
            let sum_idx = graph.add_tensor(&format!("lukasiewicz_sum_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![not_premise_idx, conclusion_idx],
                vec![sum_idx],
                Some(einsum_spec),
            ))?;

            // Step 3: min(1, sum)
            let min_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "min",
                vec![one_idx, sum_idx],
                vec![result_idx],
                Some(min_spec.to_string()),
            ))?;
        }
        FuzzyImplicationKind::Reichenbach => {
            // I(a,b) = 1 - a + ab

            // Step 1: 1 - a
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let not_premise_idx = graph.add_tensor(&format!("reichenbach_not_premise_{}", graph.tensors.len()));
            let broadcast_spec = if premise_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", premise_spec, premise_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![one_idx, premise_idx],
                vec![not_premise_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 2: a * b
            let einsum_spec = format!("{},{}->{}", premise_spec, conclusion_spec, out_spec);
            let prod_idx = graph.add_tensor(&format!("reichenbach_prod_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![premise_idx, conclusion_idx],
                vec![prod_idx],
                Some(einsum_spec),
            ))?;

            // Step 3: (1-a) + ab
            let add_spec1 = if out_spec.is_empty() {
                ",->"
            } else if premise_spec.is_empty() {
                &format!("{}->{}", out_spec, out_spec)
            } else {
                &format!("{},{}->{}", premise_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![not_premise_idx, prod_idx],
                vec![result_idx],
                Some(add_spec1.to_string()),
            ))?;
        }
        FuzzyImplicationKind::KleeneDienes => {
            // I(a,b) = max(1-a, b)

            // Step 1: 1 - a
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let not_premise_idx = graph.add_tensor(&format!("kleene_not_premise_{}", graph.tensors.len()));
            let broadcast_spec = if premise_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", premise_spec, premise_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![one_idx, premise_idx],
                vec![not_premise_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            // Step 2: max(1-a, b)
            let einsum_spec = format!("{},{}->{}", premise_spec, conclusion_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "max",
                vec![not_premise_idx, conclusion_idx],
                vec![result_idx],
                Some(einsum_spec),
            ))?;
        }
        FuzzyImplicationKind::Rescher => {
            // I(a,b) = { 1 if a≤b, 0 otherwise }
            // This is a crisp implication: (a <= b) as indicator
            let einsum_spec = format!("{},{}->{}", premise_spec, conclusion_spec, out_spec);
            graph.add_node(EinsumNode::element_wise(
                "lte",
                vec![premise_idx, conclusion_idx],
                vec![result_idx],
                Some(einsum_spec),
            ))?;
        }
        FuzzyImplicationKind::Goguen => {
            // I(a,b) = { 1 if a≤b, b/a otherwise }

            // Step 1: a <= b
            let einsum_spec = format!("{},{}->{}", premise_spec, conclusion_spec, out_spec);
            let cond_idx = graph.add_tensor(&format!("goguen_cond_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "lte",
                vec![premise_idx, conclusion_idx],
                vec![cond_idx],
                Some(einsum_spec.clone()),
            ))?;

            // Step 2: b / a
            let div_idx = graph.add_tensor(&format!("goguen_div_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::element_wise(
                "div",
                vec![conclusion_idx, premise_idx],
                vec![div_idx],
                Some(einsum_spec),
            ))?;

            // Step 3: cond * 1 + (1-cond) * (b/a)
            let one_idx = graph.add_tensor(&format!("constant_one_{}", graph.tensors.len()));
            graph.add_node(EinsumNode::constant(1.0, vec![], vec![one_idx]))?;

            let not_cond_idx = graph.add_tensor(&format!("goguen_not_cond_{}", graph.tensors.len()));
            let broadcast_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},->{}", out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "sub",
                vec![one_idx, cond_idx],
                vec![not_cond_idx],
                Some(broadcast_spec.to_string()),
            ))?;

            let term2_idx = graph.add_tensor(&format!("goguen_term2_{}", graph.tensors.len()));
            let mul_spec = if out_spec.is_empty() {
                ",->"
            } else {
                &format!("{},{}->{}", out_spec, out_spec, out_spec)
            };
            graph.add_node(EinsumNode::element_wise(
                "mul",
                vec![not_cond_idx, div_idx],
                vec![term2_idx],
                Some(mul_spec.to_string()),
            ))?;

            graph.add_node(EinsumNode::element_wise(
                "add",
                vec![cond_idx, term2_idx],
                vec![result_idx],
                Some(mul_spec.to_string()),
            ))?;
        }
    }

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

// Helper functions for generating operation names
fn kind_name(kind: TNormKind) -> &'static str {
    match kind {
        TNormKind::Minimum => "min",
        TNormKind::Product => "prod",
        TNormKind::Lukasiewicz => "lukasiewicz",
        TNormKind::Drastic => "drastic",
        TNormKind::NilpotentMinimum => "nilpotent_min",
        TNormKind::Hamacher => "hamacher",
    }
}

fn tconorm_name(kind: TCoNormKind) -> &'static str {
    match kind {
        TCoNormKind::Maximum => "max",
        TCoNormKind::ProbabilisticSum => "prob_sum",
        TCoNormKind::BoundedSum => "bounded_sum",
        TCoNormKind::Drastic => "drastic",
        TCoNormKind::NilpotentMaximum => "nilpotent_max",
        TCoNormKind::Hamacher => "hamacher",
    }
}

fn fuzzy_impl_name(kind: FuzzyImplicationKind) -> &'static str {
    match kind {
        FuzzyImplicationKind::Godel => "godel",
        FuzzyImplicationKind::Lukasiewicz => "lukasiewicz",
        FuzzyImplicationKind::Reichenbach => "reichenbach",
        FuzzyImplicationKind::KleeneDienes => "kleene_dienes",
        FuzzyImplicationKind::Rescher => "rescher",
        FuzzyImplicationKind::Goguen => "goguen",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{Term, TLExpr};

    #[test]
    fn test_compile_tnorm_minimum() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        let mut graph = EinsumGraph::new();

        let left = TLExpr::pred("P", vec![Term::var("x")]);
        let right = TLExpr::pred("Q", vec![Term::var("x")]);

        let result = compile_tnorm(TNormKind::Minimum, &left, &right, &mut ctx, &mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_tnorm_product() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        let mut graph = EinsumGraph::new();

        let left = TLExpr::pred("P", vec![Term::var("x")]);
        let right = TLExpr::pred("Q", vec![Term::var("x")]);

        let result = compile_tnorm(TNormKind::Product, &left, &right, &mut ctx, &mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_tconorm_maximum() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        let mut graph = EinsumGraph::new();

        let left = TLExpr::pred("P", vec![Term::var("x")]);
        let right = TLExpr::pred("Q", vec![Term::var("x")]);

        let result = compile_tconorm(TCoNormKind::Maximum, &left, &right, &mut ctx, &mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_fuzzy_not_standard() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        let mut graph = EinsumGraph::new();

        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        let result = compile_fuzzy_not(FuzzyNegationKind::Standard, &expr, &mut ctx, &mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_fuzzy_implication_lukasiewicz() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
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
    }
}
