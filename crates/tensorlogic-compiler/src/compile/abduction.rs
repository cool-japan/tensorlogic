//! Abductive reasoning operator compilation.
//!
//! This module implements compilation of abductive reasoning operators for
//! explanation generation and hypothesis selection.
//!
//! # Background
//!
//! Abductive reasoning is a form of logical inference that seeks the best explanation
//! for observed facts:
//!
//! - **Given**: A theory T and observations O
//! - **Find**: A hypothesis H such that T ∪ H ⊨ O
//! - **Optimize**: Minimize the cost of H
//!
//! # Operators
//!
//! ## Abducible
//!
//! An `Abducible {name, cost}` is a literal that can be assumed true to explain observations.
//! The cost parameter represents how expensive it is to assume this literal (lower is better).
//!
//! ## Explain
//!
//! `Explain {formula}` marks a formula as needing explanation. The system should find
//! the minimal-cost set of abducible assumptions that make the formula true.
//!
//! # Tensor Compilation Strategy
//!
//! ## Abducible Literals
//!
//! An abducible literal is represented as:
//! - A binary decision variable (0 or 1)
//! - With an associated cost tensor
//!
//! ```text
//! Abducible("H", cost) → decision_var("H") with cost_weight(cost)
//! ```
//!
//! ## Explanation Search
//!
//! The `Explain` operator compiles to an optimization problem:
//!
//! 1. **Decision variables**: Binary tensors for each abducible
//! 2. **Constraint**: The formula must be satisfied
//! 3. **Objective**: Minimize total cost = Σ (abducible_i * cost_i)
//!
//! Since EinsumGraph is a static computation graph, we can't perform dynamic search.
//! Instead, we compile to:
//! - A soft constraint that encourages the formula to be true
//! - Cost penalties for using abducibles
//! - The backend can then use gradient descent or other optimization to find solutions
//!
//! ## Soft Abduction Compilation
//!
//! For differentiable backends:
//!
//! ```text
//! explain(φ) with abducibles {H₁(c₁), H₂(c₂), ...}
//! ```
//!
//! Compiles to:
//!
//! ```text
//! satisfaction_loss = (1 - φ)²
//! cost_loss = Σ (Hᵢ * cᵢ)
//! total_loss = satisfaction_loss + λ * cost_loss
//! ```
//!
//! Where:
//! - `satisfaction_loss`: Penalizes when φ is not satisfied
//! - `cost_loss`: Penalizes using expensive abducibles
//! - `λ`: Trade-off parameter (default: 1.0)
//!
//! # Examples
//!
//! ## Medical Diagnosis
//!
//! ```text
//! Explain(Fever(patient) ∧ Cough(patient))
//!
//! Abducibles:
//!   - Flu(patient), cost=0.3
//!   - Cold(patient), cost=0.2
//!   - COVID(patient), cost=0.5
//! ```
//!
//! The system finds the minimal-cost explanation (likely Cold or Flu).
//!
//! ## Robot Planning
//!
//! ```text
//! Explain(At(robot, goal))
//!
//! Abducibles:
//!   - Move(robot, A, goal), cost=distance(A, goal)
//!   - Teleport(robot, goal), cost=10.0
//! ```
//!
//! # Compilation Modes
//!
//! ## Mode 1: Soft Optimization (Default)
//!
//! Compiles to a differentiable loss that can be minimized via gradient descent.
//! Suitable for neural-symbolic integration.
//!
//! ## Mode 2: Hard Constraint
//!
//! Compiles to discrete constraints for SAT/CSP solvers.
//! Requires backend support for discrete optimization.
//!
//! # Limitations
//!
//! - No dynamic hypothesis search (compile-time encoding only)
//! - Cannot enumerate all explanations (only finds one via optimization)
//! - Cost model is simplified (linear combination)
//! - No probabilistic abduction (could be added via probabilistic operators)
//!
//! # Future Work
//!
//! - Support multiple explanation enumeration
//! - Integrate with probabilistic reasoning for Bayesian abduction
//! - Add preference-based abduction (preference orderings, not just costs)
//! - Implement answer set programming (ASP) style abduction
//! - Support abductive logic programming with unification

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

use crate::compile::compile_expr;
use crate::context::{CompileState, CompilerContext};

/// Default trade-off parameter between satisfying the formula and minimizing cost.
const DEFAULT_LAMBDA: f64 = 1.0;

/// Compile an Abducible literal.
///
/// An abducible is a hypothesis that can be assumed true to explain observations.
/// It's represented as a decision variable (0 or 1) with an associated cost.
///
/// # Parameters
///
/// - `name`: The name of the abducible literal (e.g., "has_flu", "is_raining")
/// - `_cost`: The cost of assuming this literal (lower is better)
/// - `ctx`: Compiler context
/// - `graph`: The einsum graph
///
/// # Returns
///
/// A CompileState representing the decision variable for this abducible.
///
/// # Example
///
/// ```text
/// Abducible("has_flu", 0.3) → decision_var[0.3 weighted]
/// ```
pub(crate) fn compile_abducible(
    name: &str,
    cost: f64,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Create a tensor for this abducible decision variable
    let abducible_name = format!("abducible_{}", name);
    let abducible_idx = graph.add_tensor(abducible_name.clone());

    // Store the cost in proper metadata using the tensor_metadata HashMap
    use tensorlogic_ir::Metadata;
    let metadata = Metadata::new()
        .with_name(format!("Abducible: {}", name))
        .with_attribute("abducible_cost", cost.to_string());
    graph.tensor_metadata.insert(abducible_idx, metadata);

    // Register this abducible in context for tracking
    register_abducible(ctx, name, cost, abducible_idx)?;

    // Return a scalar decision variable (no axes)
    Ok(CompileState {
        tensor_idx: abducible_idx,
        axes: String::new(),
    })
}

/// Compile an Explain operator.
///
/// The explain operator marks a formula as needing explanation via abductive reasoning.
/// It compiles to an optimization objective that:
/// 1. Encourages the formula to be satisfied
/// 2. Minimizes the cost of abducibles used
///
/// # Strategy
///
/// We compile to a soft optimization problem:
///
/// ```text
/// satisfaction_term = formula
/// cost_term = Σ (abducible_i * cost_i)
/// result = satisfaction_term - λ * cost_term
/// ```
///
/// Where we want to maximize `result` (satisfy formula while minimizing cost).
///
/// # Parameters
///
/// - `formula`: The formula to explain
/// - `ctx`: Compiler context
/// - `graph`: The einsum graph
///
/// # Returns
///
/// A CompileState representing the explanation objective.
///
/// # Example
///
/// ```text
/// Explain(Fever ∧ Cough) with abducibles {Flu(0.3), Cold(0.2)}
/// ```
///
/// Compiles to an objective that balances satisfying "Fever ∧ Cough" with
/// minimizing the cost of assuming Flu or Cold.
pub(crate) fn compile_explain(
    formula: &tensorlogic_ir::TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // First, compile the formula to be explained
    let formula_result = compile_expr(formula, ctx, graph)?;

    // Get all registered abducibles and their costs
    let abducibles = get_registered_abducibles(ctx, graph)?;

    if abducibles.is_empty() {
        // No abducibles available - just return the formula as-is
        // (This means explanation degenerates to just checking the formula)
        return Ok(formula_result);
    }

    // Compute the total cost term: Σ (abducible_i * cost_i)
    let cost_term_idx = compute_total_cost(ctx, graph, &abducibles)?;

    // Create the explanation objective: satisfaction - λ * cost
    // We want to maximize this (satisfy the formula while minimizing cost)

    // Multiply cost term by lambda
    let lambda_tensor = create_constant_tensor(DEFAULT_LAMBDA, ctx, graph)?;
    let weighted_cost_name = ctx.fresh_temp();
    let weighted_cost_idx = graph.add_tensor(weighted_cost_name);

    let mul_node = EinsumNode {
        op: OpType::ElemBinary {
            op: "mul".to_string(),
        },
        inputs: vec![cost_term_idx, lambda_tensor],
        outputs: vec![weighted_cost_idx],
        metadata: None,
    };
    graph.add_node(mul_node)?;

    // Subtract weighted cost from formula satisfaction
    // result = formula - weighted_cost
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create subtraction node
    let sub_node = EinsumNode {
        op: OpType::ElemBinary {
            op: "sub".to_string(),
        },
        inputs: vec![formula_result.tensor_idx, weighted_cost_idx],
        outputs: vec![result_idx],
        metadata: None, // Could add Metadata::new() if needed
    };
    graph.add_node(sub_node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: formula_result.axes,
    })
}

/// Register an abducible in the compiler context.
///
/// Stores the abducible's tensor index in the context's let_bindings map
/// with the prefix "abd_" for easy retrieval. The cost is stored in the
/// tensor's metadata in the graph.
fn register_abducible(
    ctx: &mut CompilerContext,
    name: &str,
    _cost: f64,
    tensor_idx: usize,
) -> Result<()> {
    // Store abducibles in let_bindings with "abd_" prefix
    // The cost is stored in graph.tensor_metadata via the "abducible_cost" attribute
    let key = format!("abd_{}", name);
    ctx.let_bindings.insert(key, tensor_idx);

    Ok(())
}

/// Get all registered abducibles from the compiler context.
///
/// Returns a vector of (name, cost, tensor_idx) tuples.
fn get_registered_abducibles(
    ctx: &CompilerContext,
    graph: &EinsumGraph,
) -> Result<Vec<(String, f64, usize)>> {
    let mut abducibles = Vec::new();

    for (key, &tensor_idx) in &ctx.let_bindings {
        if let Some(name) = key.strip_prefix("abd_") {
            // Extract cost from tensor metadata
            let cost = if let Some(metadata) = graph.tensor_metadata.get(&tensor_idx) {
                // Try to get the cost attribute
                if let Some(cost_str) = metadata.get_attribute("abducible_cost") {
                    cost_str.parse::<f64>().unwrap_or(1.0) // Default to 1.0 if parsing fails
                } else {
                    1.0 // Default cost if attribute not found
                }
            } else {
                1.0 // Default cost if no metadata
            };

            abducibles.push((name.to_string(), cost, tensor_idx));
        }
    }

    Ok(abducibles)
}

/// Compute the total cost of all abducibles: Σ (abducible_i * cost_i)
fn compute_total_cost(
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
    abducibles: &[(String, f64, usize)],
) -> Result<usize> {
    if abducibles.is_empty() {
        // Return a zero tensor
        return create_constant_tensor(0.0, ctx, graph);
    }

    // Start with the first abducible * cost
    let (_, cost_0, tensor_idx_0) = abducibles[0];
    let cost_0_tensor = create_constant_tensor(cost_0, ctx, graph)?;

    let accumulator_name = ctx.fresh_temp();
    let mut accumulator_idx = graph.add_tensor(accumulator_name);

    // First weighted abducible
    let mul_node_0 = EinsumNode {
        op: OpType::ElemBinary {
            op: "mul".to_string(),
        },
        inputs: vec![tensor_idx_0, cost_0_tensor],
        outputs: vec![accumulator_idx],
        metadata: None,
    };
    graph.add_node(mul_node_0)?;

    // Accumulate the rest
    for (_, cost_i, tensor_idx_i) in abducibles.iter().skip(1) {
        let cost_i_tensor = create_constant_tensor(*cost_i, ctx, graph)?;

        // Multiply abducible by its cost
        let weighted_name = ctx.fresh_temp();
        let weighted_idx = graph.add_tensor(weighted_name);

        let mul_node = EinsumNode {
            op: OpType::ElemBinary {
                op: "mul".to_string(),
            },
            inputs: vec![*tensor_idx_i, cost_i_tensor],
            outputs: vec![weighted_idx],
            metadata: None,
        };
        graph.add_node(mul_node)?;

        // Add to accumulator
        let new_accumulator_name = ctx.fresh_temp();
        let new_accumulator_idx = graph.add_tensor(new_accumulator_name);

        let add_node = EinsumNode {
            op: OpType::ElemBinary {
                op: "add".to_string(),
            },
            inputs: vec![accumulator_idx, weighted_idx],
            outputs: vec![new_accumulator_idx],
            metadata: None,
        };
        graph.add_node(add_node)?;

        accumulator_idx = new_accumulator_idx;
    }

    Ok(accumulator_idx)
}

/// Create a constant scalar tensor with the given value.
fn create_constant_tensor(
    value: f64,
    _ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<usize> {
    let const_name = format!("const_{}", value);
    let const_idx = graph.add_tensor(const_name.clone());

    // Mark as constant in metadata
    let metadata = format!("constant:{}", value);
    graph
        .tensors
        .get_mut(const_idx)
        .unwrap()
        .push_str(&format!("#{}", metadata));

    Ok(const_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TLExpr, Term};

    #[test]
    fn test_abducible_compilation() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let result = compile_abducible("has_flu", 0.3, &mut ctx, &mut graph).unwrap();

        // Should have created a tensor
        assert!(!graph.tensors.is_empty());
        // Should be a scalar (no axes)
        assert!(result.axes.is_empty());
        // Tensor name should contain "abducible"
        assert!(graph.tensors[result.tensor_idx].contains("abducible"));
    }

    #[test]
    fn test_explain_without_abducibles() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // Explain(Safe) with no abducibles
        let safe = TLExpr::pred("Safe", vec![]);

        let _result = compile_explain(&safe, &mut ctx, &mut graph).unwrap();

        // Should compile successfully (degenerates to just the formula)
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_explain_with_single_abducible() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // First register an abducible
        compile_abducible("has_flu", 0.3, &mut ctx, &mut graph).unwrap();

        // Then explain a formula
        let fever = TLExpr::pred("Fever", vec![]);

        let _result = compile_explain(&fever, &mut ctx, &mut graph).unwrap();

        // Should have created nodes for the explanation objective
        assert!(!graph.nodes.is_empty());
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_explain_with_multiple_abducibles() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // Register multiple abducibles
        compile_abducible("has_flu", 0.3, &mut ctx, &mut graph).unwrap();
        compile_abducible("has_cold", 0.2, &mut ctx, &mut graph).unwrap();
        compile_abducible("has_covid", 0.5, &mut ctx, &mut graph).unwrap();

        // Explain a complex formula
        let fever = TLExpr::pred("Fever", vec![]);
        let cough = TLExpr::pred("Cough", vec![]);
        let symptoms = TLExpr::and(fever, cough);

        let _result = compile_explain(&symptoms, &mut ctx, &mut graph).unwrap();

        // Should have created a complex graph
        assert!(graph.nodes.len() >= 3);
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_abducible_with_zero_cost() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let result = compile_abducible("free_assumption", 0.0, &mut ctx, &mut graph).unwrap();

        // Zero cost abducibles should still compile
        assert!(!graph.tensors.is_empty());
        assert!(graph.tensors[result.tensor_idx].contains("abducible"));
    }

    #[test]
    fn test_abducible_with_high_cost() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let result =
            compile_abducible("expensive_hypothesis", 100.0, &mut ctx, &mut graph).unwrap();

        // High cost abducibles should still compile
        assert!(!graph.tensors.is_empty());
        assert!(graph.tensors[result.tensor_idx].contains("abducible"));
    }

    #[test]
    fn test_multiple_explain_calls() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // Register abducibles
        compile_abducible("H1", 1.0, &mut ctx, &mut graph).unwrap();
        compile_abducible("H2", 2.0, &mut ctx, &mut graph).unwrap();

        // Multiple explain calls
        let formula1 = TLExpr::pred("P", vec![]);
        let formula2 = TLExpr::pred("Q", vec![]);

        let _result1 = compile_explain(&formula1, &mut ctx, &mut graph).unwrap();
        let _result2 = compile_explain(&formula2, &mut ctx, &mut graph).unwrap();

        // Both should compile successfully
        assert!(graph.nodes.len() >= 2);
    }

    #[test]
    fn test_explain_with_free_variables() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 10);
        let mut graph = EinsumGraph::new();

        // Register abducibles
        compile_abducible("knows_someone", 1.0, &mut ctx, &mut graph).unwrap();

        // Explain(Knows(x, y))
        let knows = TLExpr::pred("Knows", vec![Term::var("x"), Term::var("y")]);
        ctx.bind_var("x", "Person").unwrap();
        ctx.bind_var("y", "Person").unwrap();

        let result = compile_explain(&knows, &mut ctx, &mut graph).unwrap();

        // Should preserve axes from formula
        assert!(!result.axes.is_empty());
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_nested_explain_not_recommended() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        compile_abducible("H", 1.0, &mut ctx, &mut graph).unwrap();

        // Nested explain: Explain(Explain(P))
        // This is semantically unusual but should compile
        let p = TLExpr::pred("P", vec![]);
        let inner_explain = TLExpr::Explain {
            formula: Box::new(p),
        };

        let _result = compile_explain(&inner_explain, &mut ctx, &mut graph);

        // Should compile (though the semantics are questionable)
        // This test just checks it doesn't crash
    }

    #[test]
    fn test_abducible_name_uniqueness() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let result1 = compile_abducible("H", 1.0, &mut ctx, &mut graph).unwrap();
        let result2 = compile_abducible("H", 1.0, &mut ctx, &mut graph).unwrap();

        // Second abducible with same name should create a different tensor
        // (or reuse the same one - implementation dependent)
        // For now, we create new tensors each time
        assert!(!graph.tensors.is_empty());
        // Indices should be different
        assert_ne!(result1.tensor_idx, result2.tensor_idx);
    }

    #[test]
    fn test_abducible_cost_metadata_storage() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // Create abducibles with different costs
        let result1 = compile_abducible("cheap", 0.5, &mut ctx, &mut graph).unwrap();
        let result2 = compile_abducible("expensive", 10.0, &mut ctx, &mut graph).unwrap();
        let result3 = compile_abducible("moderate", 2.5, &mut ctx, &mut graph).unwrap();

        // Verify costs are stored in metadata
        let meta1 = graph.tensor_metadata.get(&result1.tensor_idx).unwrap();
        assert_eq!(meta1.get_attribute("abducible_cost"), Some("0.5"));

        let meta2 = graph.tensor_metadata.get(&result2.tensor_idx).unwrap();
        assert_eq!(meta2.get_attribute("abducible_cost"), Some("10"));

        let meta3 = graph.tensor_metadata.get(&result3.tensor_idx).unwrap();
        assert_eq!(meta3.get_attribute("abducible_cost"), Some("2.5"));
    }

    #[test]
    fn test_get_registered_abducibles_extracts_costs() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // Create abducibles with known costs
        compile_abducible("H1", 1.0, &mut ctx, &mut graph).unwrap();
        compile_abducible("H2", 2.5, &mut ctx, &mut graph).unwrap();
        compile_abducible("H3", 0.3, &mut ctx, &mut graph).unwrap();

        // Get registered abducibles
        let abducibles = get_registered_abducibles(&ctx, &graph).unwrap();

        // Should have 3 abducibles
        assert_eq!(abducibles.len(), 3);

        // Verify costs are correctly extracted
        // Note: order might vary so we check all possible names
        for (name, cost, _idx) in &abducibles {
            match name.as_str() {
                "H1" => assert_eq!(*cost, 1.0),
                "H2" => assert_eq!(*cost, 2.5),
                "H3" => assert_eq!(*cost, 0.3),
                _ => panic!("Unexpected abducible name: {}", name),
            }
        }
    }

    #[test]
    fn test_explain_uses_correct_costs() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // Register abducibles with specific costs
        compile_abducible("H1", 1.0, &mut ctx, &mut graph).unwrap();
        compile_abducible("H2", 5.0, &mut ctx, &mut graph).unwrap();

        // Create a formula to explain
        let formula = TLExpr::pred("Safe", vec![]);

        // Compile the explanation
        compile_explain(&formula, &mut ctx, &mut graph).unwrap();

        // The graph should have computed cost terms
        // We can't easily verify the exact computation here,
        // but we can check that nodes were created
        assert!(!graph.nodes.is_empty());

        // Verify that the abducibles still have their correct costs in metadata
        let abducibles = get_registered_abducibles(&ctx, &graph).unwrap();
        assert_eq!(abducibles.len(), 2);

        for (name, cost, _) in &abducibles {
            if name == "H1" {
                assert_eq!(*cost, 1.0);
            } else if name == "H2" {
                assert_eq!(*cost, 5.0);
            }
        }
    }
}
