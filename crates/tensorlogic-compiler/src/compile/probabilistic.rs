//! Probabilistic operator compilation.
//!
//! This module implements compilation of probabilistic operators including:
//! - Weighted rules (soft constraints with confidence weights)
//! - Probabilistic choice (stochastic selection between alternatives)

use anyhow::{bail, Result};
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr};

use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

/// Compile a weighted rule with confidence/probability weight.
///
/// A weighted rule multiplies the compiled expression by its weight,
/// allowing soft constraints and probabilistic reasoning.
///
/// For example, a rule with weight 0.8 indicates 80% confidence.
pub(crate) fn compile_weighted_rule(
    weight: f64,
    rule: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Compile the inner rule
    let rule_state = compile_expr(rule, ctx, graph)?;

    // Create constant tensor for weight
    let weight_name = format!("const_{}", weight);
    let weight_idx = if !graph.tensors.contains(&weight_name) {
        graph.add_tensor(weight_name.clone())
    } else {
        graph
            .tensors
            .iter()
            .position(|t| t == &weight_name)
            .unwrap()
    };

    // Multiply rule result by weight
    let result_idx = graph.add_tensor(format!("weighted_rule_{}", graph.tensors.len()));
    let node = EinsumNode::elem_binary("multiply", rule_state.tensor_idx, weight_idx, result_idx);
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: rule_state.axes,
    })
}

/// Compile probabilistic choice between alternatives with given probabilities.
///
/// This implements a soft mixture model where each alternative is weighted by its probability:
/// result = sum_i (p_i * expr_i)
///
/// The probabilities should sum to 1.0, but this is not enforced at compile time.
pub(crate) fn compile_probabilistic_choice(
    alternatives: &[(f64, TLExpr)],
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    if alternatives.is_empty() {
        bail!("ProbabilisticChoice requires at least one alternative");
    }

    // Validate that probabilities are non-negative
    for (prob, _) in alternatives {
        if *prob < 0.0 {
            bail!(
                "Probabilities in ProbabilisticChoice must be non-negative, got {}",
                prob
            );
        }
    }

    // Compile all alternatives
    let mut compiled_alternatives = Vec::new();
    let mut output_axes = String::new();
    let mut seen = std::collections::HashSet::new();

    for (prob, expr) in alternatives {
        let state = compile_expr(expr, ctx, graph)?;

        // Update axes union
        for c in state.axes.chars() {
            if seen.insert(c) {
                output_axes.push(c);
            }
        }

        compiled_alternatives.push((*prob, state));
    }

    // If only one alternative, just weight it
    if compiled_alternatives.len() == 1 {
        let (prob, state) = &compiled_alternatives[0];

        let prob_name = format!("const_{}", prob);
        let prob_idx = if !graph.tensors.contains(&prob_name) {
            graph.add_tensor(prob_name.clone())
        } else {
            graph.tensors.iter().position(|t| t == &prob_name).unwrap()
        };

        let result_idx = graph.add_tensor(format!("prob_choice_{}", graph.tensors.len()));
        let node = EinsumNode::elem_binary("multiply", state.tensor_idx, prob_idx, result_idx);
        graph.add_node(node)?;

        return Ok(CompileState {
            tensor_idx: result_idx,
            axes: state.axes.clone(),
        });
    }

    // Compute weighted sum: sum_i (p_i * expr_i)
    // First alternative: prob_0 * expr_0
    let (prob_0, state_0) = &compiled_alternatives[0];
    let prob_0_name = format!("const_{}", prob_0);
    let prob_0_idx = if !graph.tensors.contains(&prob_0_name) {
        graph.add_tensor(prob_0_name.clone())
    } else {
        graph
            .tensors
            .iter()
            .position(|t| t == &prob_0_name)
            .unwrap()
    };

    let weighted_0_idx = graph.add_tensor(format!("weighted_0_{}", graph.tensors.len()));
    let node = EinsumNode::elem_binary("multiply", state_0.tensor_idx, prob_0_idx, weighted_0_idx);
    graph.add_node(node)?;

    // Accumulate sum with broadcasting for different axes
    let mut accum_idx = weighted_0_idx;
    let mut accum_axes = state_0.axes.clone();

    for (i, (prob_i, state_i)) in compiled_alternatives.iter().skip(1).enumerate() {
        let prob_i_name = format!("const_{}", prob_i);
        let prob_i_idx = if !graph.tensors.contains(&prob_i_name) {
            graph.add_tensor(prob_i_name.clone())
        } else {
            graph
                .tensors
                .iter()
                .position(|t| t == &prob_i_name)
                .unwrap()
        };

        let weighted_i_idx =
            graph.add_tensor(format!("weighted_{}_{}", i + 1, graph.tensors.len()));
        let weighted_node =
            EinsumNode::elem_binary("multiply", state_i.tensor_idx, prob_i_idx, weighted_i_idx);
        graph.add_node(weighted_node)?;

        // Add to accumulator with broadcasting if needed
        let new_accum_idx = graph.add_tensor(format!("accum_{}_{}", i + 1, graph.tensors.len()));

        // If axes match or one is scalar, use elem_binary
        if accum_axes == state_i.axes || accum_axes.is_empty() || state_i.axes.is_empty() {
            let add_node = EinsumNode::elem_binary("add", accum_idx, weighted_i_idx, new_accum_idx);
            graph.add_node(add_node)?;
            // Update accum_axes to union
            if accum_axes.is_empty() {
                accum_axes = state_i.axes.clone();
            } else if !state_i.axes.is_empty() {
                // Union of axes
                let mut seen_in_accum = std::collections::HashSet::new();
                for c in accum_axes.chars() {
                    seen_in_accum.insert(c);
                }
                for c in state_i.axes.chars() {
                    if !seen_in_accum.contains(&c) {
                        accum_axes.push(c);
                    }
                }
            }
        } else {
            // Need broadcasting via einsum
            let add_spec = format!("{},{}->{}", accum_axes, state_i.axes, output_axes);
            let add_node = EinsumNode::new(
                add_spec,
                vec![accum_idx, weighted_i_idx],
                vec![new_accum_idx],
            );
            graph.add_node(add_node)?;
            accum_axes = output_axes.clone();
        }

        accum_idx = new_accum_idx;
    }

    Ok(CompileState {
        tensor_idx: accum_idx,
        axes: output_axes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TLExpr, Term};

    #[test]
    fn test_compile_weighted_rule() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        let mut graph = EinsumGraph::new();

        let rule = TLExpr::pred("P", vec![Term::var("x")]);

        let result = compile_weighted_rule(0.8, &rule, &mut ctx, &mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_probabilistic_choice_single() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        let mut graph = EinsumGraph::new();

        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        let alternatives = vec![(1.0, expr)];

        let result = compile_probabilistic_choice(&alternatives, &mut ctx, &mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_probabilistic_choice_multiple() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);
        let mut graph = EinsumGraph::new();

        let expr1 = TLExpr::pred("P", vec![Term::var("x")]);
        let expr2 = TLExpr::pred("Q", vec![Term::var("x")]);
        let alternatives = vec![(0.6, expr1), (0.4, expr2)];

        let result = compile_probabilistic_choice(&alternatives, &mut ctx, &mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_probabilistic_choice_empty_fails() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let alternatives = vec![];

        let result = compile_probabilistic_choice(&alternatives, &mut ctx, &mut graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_probabilistic_choice_negative_prob_fails() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        let alternatives = vec![(-0.5, expr)];

        let result = compile_probabilistic_choice(&alternatives, &mut ctx, &mut graph);
        assert!(result.is_err());
    }
}
