//! Constraint programming operations compilation.
//!
//! This module implements compilation of constraint programming operators
//! commonly used in combinatorial optimization, scheduling, and planning problems.
//!
//! # Constraints
//!
//! ## AllDifferent
//!
//! The AllDifferent constraint ensures that all variables take distinct values.
//! It's fundamental for problems like:
//! - N-Queens problem
//! - Sudoku
//! - Task scheduling
//! - Graph coloring
//!
//! ### Tensor Compilation
//!
//! For variables x₁, x₂, ..., xₙ over a discrete domain, we compile to:
//! ```text
//! AllDifferent(x₁, ..., xₙ) = ∏_{i<j} Inequality(xᵢ, xⱼ)
//! ```
//!
//! Where Inequality can be:
//! - Hard: 1 - δ(xᵢ, xⱼ) where δ is Kronecker delta
//! - Soft: sigmoid(|xᵢ - xⱼ|) for continuous relaxation
//!
//! ## GlobalCardinality
//!
//! The GlobalCardinality constraint bounds how many times each value appears.
//! Useful for:
//! - Resource allocation (limited resources)
//! - Load balancing
//! - Fair assignment
//!
//! ### Tensor Compilation
//!
//! For values v₁, ..., vₘ with min/max occurrences:
//! ```text
//! GlobalCardinality = ∧ᵥ (minᵥ ≤ count(v) ≤ maxᵥ)
//! ```
//!
//! Where count(v) = ∑ᵢ δ(xᵢ, v) counts occurrences of value v.

use anyhow::{bail, Result};
use tensorlogic_ir::{EinsumGraph, TLExpr};

use crate::compile::compile_expr;
use crate::context::{CompileState, CompilerContext};

/// Compile AllDifferent constraint
///
/// Ensures all variables in the list take different values.
///
/// # Tensor Compilation Strategy
///
/// For a list of n variable names, we:
/// 1. Look up each variable's tensor representation
/// 2. Create pairwise inequality constraints
/// 3. Combine with AND (product or min depending on strategy)
///
/// For discrete domains, this checks:
/// ```text
/// ∏_{i<j} (1 - Equal(var[i], var[j]))
/// ```
///
/// For continuous domains (soft constraint):
/// ```text
/// ∏_{i<j} sigmoid(scale * |var[i] - var[j]|)
/// ```
///
/// # Example
///
/// ```text
/// AllDifferent([x, y, z]) over domain {1, 2, 3}
/// Compiles to: (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z)
/// ```
pub(crate) fn compile_all_different(
    variables: &[String],
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    if variables.is_empty() {
        bail!("AllDifferent constraint requires at least one variable");
    }

    if variables.len() == 1 {
        // Trivially satisfied for single variable
        let tensor_name = "const_1.0";
        let tensor_idx = graph.add_tensor(tensor_name);
        return Ok(CompileState {
            tensor_idx,
            axes: String::new(),
        });
    }

    // For each pair of variables, create inequality constraint
    let mut constraints = Vec::new();

    for i in 0..variables.len() {
        for j in (i + 1)..variables.len() {
            let var_i = &variables[i];
            let var_j = &variables[j];

            // Create expressions for the variables
            // We'll use predicates that represent the variable values
            let expr_i = TLExpr::pred(var_i, vec![]);
            let expr_j = TLExpr::pred(var_j, vec![]);

            // Inequality: NOT(Equal(var_i, var_j))
            // Which compiles to: 1 - Equal(var_i, var_j)
            let inequality = TLExpr::negate(TLExpr::Eq(Box::new(expr_i), Box::new(expr_j)));

            constraints.push(inequality);
        }
    }

    // Combine all pairwise constraints with AND
    let result_expr = constraints.into_iter().reduce(TLExpr::and).unwrap();

    // Compile the combined constraint
    compile_expr(&result_expr, ctx, graph)
}

/// Compile GlobalCardinality constraint
///
/// Ensures that each value appears within specified bounds across variables.
///
/// # Parameters
///
/// - `variables`: List of variable names to constrain
/// - `values`: List of TLExpr representing possible values
/// - `min_occurrences`: Minimum times each value must appear
/// - `max_occurrences`: Maximum times each value can appear
///
/// # Tensor Compilation Strategy
///
/// For each value v with bounds [min, max]:
/// 1. Count occurrences: count(v) = ∑ᵢ Equal(varᵢ, v)
/// 2. Check bounds: (count(v) ≥ min) ∧ (count(v) ≤ max)
/// 3. Combine all value constraints with AND
///
/// The inequality constraints are compiled as:
/// ```text
/// count ≥ min: sigmoid(scale * (count - min + 0.5))
/// count ≤ max: sigmoid(scale * (max - count + 0.5))
/// ```
///
/// # Example
///
/// ```text
/// Variables: [x, y, z] over {1, 2, 3}
/// Values: [1, 2, 3]
/// Min: [1, 0, 1]  // Value 1 appears ≥1 time, value 2 ≥0 times, value 3 ≥1 time
/// Max: [2, 2, 1]  // Value 1 appears ≤2 times, value 2 ≤2 times, value 3 ≤1 time
/// ```
pub(crate) fn compile_global_cardinality(
    variables: &[String],
    values: &[TLExpr],
    min_occurrences: &[usize],
    max_occurrences: &[usize],
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    if variables.is_empty() {
        bail!("GlobalCardinality constraint requires at least one variable");
    }

    if values.len() != min_occurrences.len() || values.len() != max_occurrences.len() {
        bail!(
            "GlobalCardinality: values, min_occurrences, and max_occurrences must have same length"
        );
    }

    // Check that min ≤ max for all values
    for (i, (min, max)) in min_occurrences
        .iter()
        .zip(max_occurrences.iter())
        .enumerate()
    {
        if min > max {
            bail!(
                "GlobalCardinality: min_occurrences[{}] ({}) > max_occurrences[{}] ({})",
                i,
                min,
                i,
                max
            );
        }
    }

    let mut value_constraints = Vec::new();

    // For each value, create cardinality constraints
    for (idx, value_expr) in values.iter().enumerate() {
        let min = min_occurrences[idx];
        let max = max_occurrences[idx];

        // Count occurrences of this value across all variables
        let mut occurrence_indicators = Vec::new();

        for var_name in variables {
            // Create expression for the variable
            let var_expr = TLExpr::pred(var_name, vec![]);

            // Check if variable equals this value
            let equals = TLExpr::Eq(Box::new(var_expr), Box::new(value_expr.clone()));

            occurrence_indicators.push(equals);
        }

        // Sum all indicators to get count
        // count = indicator₁ + indicator₂ + ... + indicatorₙ
        let count_expr = occurrence_indicators
            .into_iter()
            .reduce(|acc, indicator| TLExpr::Add(Box::new(acc), Box::new(indicator)))
            .unwrap();

        // Create bounds constraints
        // count ≥ min
        let min_constraint = if min > 0 {
            Some(TLExpr::Gte(
                Box::new(count_expr.clone()),
                Box::new(TLExpr::Constant(min as f64)),
            ))
        } else {
            None // No constraint if min is 0
        };

        // count ≤ max
        let max_constraint = if max < variables.len() {
            Some(TLExpr::Lte(
                Box::new(count_expr),
                Box::new(TLExpr::Constant(max as f64)),
            ))
        } else {
            None // No constraint if max is >= number of variables
        };

        // Combine min and max constraints
        match (min_constraint, max_constraint) {
            (Some(min_c), Some(max_c)) => {
                value_constraints.push(TLExpr::and(min_c, max_c));
            }
            (Some(c), None) | (None, Some(c)) => {
                value_constraints.push(c);
            }
            (None, None) => {
                // No constraint for this value
            }
        }
    }

    if value_constraints.is_empty() {
        // All constraints are trivially satisfied
        let tensor_name = "const_1.0";
        let tensor_idx = graph.add_tensor(tensor_name);
        return Ok(CompileState {
            tensor_idx,
            axes: String::new(),
        });
    }

    // Combine all value constraints with AND
    let combined_constraint = value_constraints.into_iter().reduce(TLExpr::and).unwrap();

    // Compile the combined constraint
    compile_expr(&combined_constraint, ctx, graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_different_single_variable() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let variables = vec!["x".to_string()];

        let result = compile_all_different(&variables, &mut ctx, &mut graph).unwrap();

        // Single variable is trivially different from itself
        assert!(result.axes.is_empty()); // Should be a scalar
    }

    #[test]
    fn test_all_different_two_variables() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let variables = vec!["x".to_string(), "y".to_string()];

        let _result = compile_all_different(&variables, &mut ctx, &mut graph).unwrap();

        // Should create inequality constraint
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_all_different_three_variables() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let variables = vec!["x".to_string(), "y".to_string(), "z".to_string()];

        let _result = compile_all_different(&variables, &mut ctx, &mut graph).unwrap();

        // Should create 3 pairwise inequality constraints
        // (x≠y) ∧ (y≠z) ∧ (x≠z)
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_all_different_empty_fails() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let variables: Vec<String> = vec![];

        let result = compile_all_different(&variables, &mut ctx, &mut graph);

        assert!(result.is_err());
    }

    #[test]
    fn test_global_cardinality_simple() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let variables = vec!["x".to_string(), "y".to_string(), "z".to_string()];

        let values = vec![TLExpr::Constant(1.0), TLExpr::Constant(2.0)];

        let min_occurrences = vec![1, 1]; // Each value appears at least once
        let max_occurrences = vec![2, 2]; // Each value appears at most twice

        let _result = compile_global_cardinality(
            &variables,
            &values,
            &min_occurrences,
            &max_occurrences,
            &mut ctx,
            &mut graph,
        )
        .unwrap();

        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_global_cardinality_no_constraints() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let variables = vec!["x".to_string(), "y".to_string()];
        let values = vec![TLExpr::Constant(1.0)];

        // No effective constraints (min=0, max=2 which is >= number of variables)
        let min_occurrences = vec![0];
        let max_occurrences = vec![2];

        let result = compile_global_cardinality(
            &variables,
            &values,
            &min_occurrences,
            &max_occurrences,
            &mut ctx,
            &mut graph,
        )
        .unwrap();

        // Should return trivially satisfied constraint
        assert!(result.axes.is_empty());
    }

    #[test]
    fn test_global_cardinality_invalid_bounds() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let variables = vec!["x".to_string()];
        let values = vec![TLExpr::Constant(1.0)];

        // min > max is invalid
        let min_occurrences = vec![2];
        let max_occurrences = vec![1];

        let result = compile_global_cardinality(
            &variables,
            &values,
            &min_occurrences,
            &max_occurrences,
            &mut ctx,
            &mut graph,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_global_cardinality_mismatched_lengths() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let variables = vec!["x".to_string()];
        let values = vec![TLExpr::Constant(1.0), TLExpr::Constant(2.0)];

        // Mismatched lengths
        let min_occurrences = vec![1]; // Only 1 element
        let max_occurrences = vec![1, 1]; // 2 elements

        let result = compile_global_cardinality(
            &variables,
            &values,
            &min_occurrences,
            &max_occurrences,
            &mut ctx,
            &mut graph,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_global_cardinality_empty_variables_fails() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let variables: Vec<String> = vec![];
        let values = vec![TLExpr::Constant(1.0)];
        let min_occurrences = vec![0];
        let max_occurrences = vec![1];

        let result = compile_global_cardinality(
            &variables,
            &values,
            &min_occurrences,
            &max_occurrences,
            &mut ctx,
            &mut graph,
        );

        assert!(result.is_err());
    }
}
