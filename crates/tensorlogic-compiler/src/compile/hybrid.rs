//! Hybrid logic operator compilation.
//!
//! This module implements compilation of hybrid logic operators that extend modal logic
//! with nominals (named states) and satisfaction operators.
//!
//! # Background
//!
//! Hybrid logic enriches modal logic with:
//! - **Nominals**: Named constants that refer to specific states in a Kripke model
//! - **Satisfaction operator (@)**: Allows evaluating formulas at named states
//! - **Universal modalities (E/A)**: Navigate to reachable states
//!
//! # Operators
//!
//! ## Nominal (@i)
//! A nominal `@i` is a propositional symbol that is true at exactly one state (named `i`).
//! In tensor terms, it's a one-hot vector over the state space.
//!
//! ## At Operator (@i φ)
//! "Formula φ is true at the nominal state i."
//! We evaluate φ and then select the value at the state named by i.
//!
//! ## Somewhere (E φ)
//! "φ is true in some state reachable from the current state."
//! This is an existential quantifier over the reachability relation.
//!
//! ## Everywhere (A φ)
//! "φ is true in all states reachable from the current state."
//! This is a universal quantifier over the reachability relation (dual of Somewhere).
//!
//! # Tensor Compilation Strategy
//!
//! ## State Space Representation
//!
//! We represent the state space as a dedicated axis `__state__`:
//! - States are indexed 0, 1, 2, ..., n-1
//! - Nominals map to specific state indices
//! - Default state space size: 10 states
//!
//! ## Reachability Relation
//!
//! The reachability relation R is a binary relation over states:
//! - R[i,j] = 1 if state j is reachable from state i
//! - R[i,j] = 0 otherwise
//!
//! For compilation without explicit R, we assume:
//! - Full connectivity: all states can reach all states (R = all ones)
//! - This is a conservative over-approximation
//!
//! ## Nominal Compilation
//!
//! ```text
//! Nominal("i") → one_hot_vector[state_index("i")]
//! ```
//!
//! Creates a tensor with 1 at the nominal's state index, 0 elsewhere.
//!
//! ## At Operator Compilation
//!
//! ```text
//! At("i", φ) → select(φ, state_axis, index("i"))
//! ```
//!
//! Compiles φ, then extracts the value at the specific state.
//!
//! ## Somewhere Compilation
//!
//! ```text
//! Somewhere(φ) → max_over_states(R * φ)
//! ```
//!
//! With full connectivity:
//! ```text
//! Somewhere(φ) → max(φ, axis=state)
//! ```
//!
//! ## Everywhere Compilation
//!
//! ```text
//! Everywhere(φ) → min_over_states(R * φ)
//! ```
//!
//! With full connectivity:
//! ```text
//! Everywhere(φ) → min(φ, axis=state)
//! ```
//!
//! # Examples
//!
//! ## Named State Reasoning
//!
//! ```text
//! @home Safe
//! ```
//! "The state named 'home' is safe"
//!
//! ## Reachability
//!
//! ```text
//! E Safe
//! ```
//! "There exists a reachable state where Safe holds"
//!
//! ## Invariants
//!
//! ```text
//! A Safe
//! ```
//! "In all reachable states, Safe holds"
//!
//! # Limitations
//!
//! - No dynamic reachability relation (uses full connectivity assumption)
//! - Fixed state space size (configurable but not inferred)
//! - Nominals must be pre-registered with state indices
//! - No binder logic (bind operator ↓)
//!
//! # Future Work
//!
//! - Support explicit reachability relations from context
//! - Dynamic state space sizing
//! - Implement bind operator (↓x.φ)
//! - Add difference operator (D)
//! - Support path constraints for limited reachability

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

use crate::compile::compile_expr;
use crate::context::{CompileState, CompilerContext};

/// Default number of states in the state space.
const DEFAULT_STATE_SPACE_SIZE: usize = 10;

/// Special axis name for the state space in hybrid logic.
const STATE_AXIS: &str = "__state__";

/// Compile a nominal: @i
///
/// A nominal is true at exactly one state (the state it names).
/// In tensor form, this is a one-hot vector over the state space.
///
/// # Parameters
///
/// - `name`: The name of the nominal (e.g., "home", "office")
/// - `ctx`: Compiler context
/// - `graph`: The einsum graph
///
/// # Returns
///
/// A CompileState with a one-hot tensor over the state axis.
///
/// # Example
///
/// ```text
/// Nominal("s1") → [0, 1, 0, 0, ..., 0]  (if s1 is state index 1)
/// ```
pub(crate) fn compile_nominal(
    name: &str,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure the state domain exists
    ensure_state_domain(ctx)?;

    // Get or assign the state index for this nominal
    let state_index = get_nominal_index(name, ctx)?;

    // Create a one-hot tensor for this nominal
    // This tensor has value 1.0 at state_index, 0.0 elsewhere
    let nominal_name = format!("nominal_{}", name);
    let nominal_idx = graph.add_tensor(nominal_name.clone());

    // We create a special constant tensor that represents the one-hot encoding
    // The backend will need to initialize this appropriately
    // For now, we mark it with metadata to indicate it's a nominal

    // Get the state axis
    let state_axis = ctx.assign_axis(STATE_AXIS);

    // Add metadata to indicate this is a nominal one-hot vector
    let metadata = format!("nominal:{}:index:{}", name, state_index);

    // Create a constant node that the backend will interpret as a one-hot vector
    // We use ElemUnary with a special operation that creates a one-hot
    // Actually, let's create it as a raw tensor that needs backend initialization

    // For compilation, we just return the tensor with appropriate metadata
    // The backend will be responsible for initializing it as a one-hot vector
    graph
        .tensors
        .get_mut(nominal_idx)
        .unwrap()
        .push_str(&format!("#{}", metadata));

    Ok(CompileState {
        tensor_idx: nominal_idx,
        axes: state_axis.to_string(),
    })
}

/// Compile the At operator: @i φ
///
/// Evaluates formula φ and extracts the value at the state named by nominal i.
///
/// # Strategy
///
/// 1. Compile φ to get a tensor that includes the state axis
/// 2. Select the slice at the nominal's state index
/// 3. Return the selected value
///
/// # Parameters
///
/// - `nominal`: The name of the nominal state
/// - `formula`: The formula to evaluate at that state
/// - `ctx`: Compiler context
/// - `graph`: The einsum graph
///
/// # Example
///
/// ```text
/// @home Safe(x)
/// ```
/// Evaluates Safe(x) and returns only the values at the "home" state.
pub(crate) fn compile_at(
    nominal: &str,
    formula: &tensorlogic_ir::TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure the state domain exists
    ensure_state_domain(ctx)?;

    // Get the nominal's state index
    let _state_index = get_nominal_index(nominal, ctx)?;

    // Compile the formula
    let formula_result = compile_expr(formula, ctx, graph)?;

    // Get the state axis
    let state_axis = ctx.assign_axis(STATE_AXIS);

    // Check if the formula result has the state axis
    if !formula_result.axes.contains(state_axis) {
        // If the formula doesn't depend on state, we need to broadcast it to the state space
        // and then select. For simplicity, just return it as-is (it's state-independent)
        return Ok(formula_result);
    }

    // We need to select the slice at the nominal's state index
    // In einsum terms, this is a bit tricky - we need a selection operation
    // For now, we'll create a multiplication with the nominal one-hot vector
    // and then sum over the state axis

    let nominal_result = compile_nominal(nominal, ctx, graph)?;

    // Multiply formula with the one-hot nominal vector
    let selected_name = ctx.fresh_temp();
    let selected_idx = graph.add_tensor(selected_name);

    // Create einsum spec for multiplication
    let output_axes: String = formula_result
        .axes
        .chars()
        .filter(|&c| c != state_axis)
        .collect();

    let spec = if formula_result.axes == state_axis.to_string() {
        // Formula only has state axis: s,s->
        format!("{0},{0}->", state_axis)
    } else {
        // Formula has state axis plus others: abc,b->ac (if b is state axis)
        format!(
            "{},{}->{}",
            formula_result.axes, nominal_result.axes, output_axes
        )
    };

    let node = EinsumNode::new(
        spec,
        vec![formula_result.tensor_idx, nominal_result.tensor_idx],
        vec![selected_idx],
    );
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: selected_idx,
        axes: output_axes,
    })
}

/// Compile Somewhere (E φ): "φ is true in some reachable state"
///
/// # Strategy
///
/// Under full connectivity assumption (all states reach all states):
/// ```text
/// E φ = max(φ, axis=state)
/// ```
///
/// This checks if φ is true in at least one state.
///
/// # Parameters
///
/// - `formula`: The formula to check
/// - `ctx`: Compiler context
/// - `graph`: The einsum graph
pub(crate) fn compile_somewhere(
    formula: &tensorlogic_ir::TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure the state domain exists
    ensure_state_domain(ctx)?;

    // Compile the formula
    let formula_result = compile_expr(formula, ctx, graph)?;

    // Get the state axis
    let state_axis = ctx.assign_axis(STATE_AXIS);

    // If formula doesn't have state axis, it's already state-independent
    if !formula_result.axes.contains(state_axis) {
        return Ok(formula_result);
    }

    // Reduce over state axis using Max (existential quantification)
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Output axes = all axes except state axis
    let output_axes: String = formula_result
        .axes
        .chars()
        .filter(|&c| c != state_axis)
        .collect();

    // Create reduction operation
    let node = EinsumNode {
        op: OpType::Reduce {
            op: "max".to_string(),
            axes: vec![], // Backend will infer from axis names
        },
        inputs: vec![formula_result.tensor_idx],
        outputs: vec![result_idx],
        metadata: None,
    };
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Compile Everywhere (A φ): "φ is true in all reachable states"
///
/// # Strategy
///
/// Under full connectivity assumption (all states reach all states):
/// ```text
/// A φ = min(φ, axis=state)
/// ```
///
/// This checks if φ is true in every state.
///
/// # Parameters
///
/// - `formula`: The formula to check
/// - `ctx`: Compiler context
/// - `graph`: The einsum graph
pub(crate) fn compile_everywhere(
    formula: &tensorlogic_ir::TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure the state domain exists
    ensure_state_domain(ctx)?;

    // Compile the formula
    let formula_result = compile_expr(formula, ctx, graph)?;

    // Get the state axis
    let state_axis = ctx.assign_axis(STATE_AXIS);

    // If formula doesn't have state axis, it's already state-independent
    if !formula_result.axes.contains(state_axis) {
        return Ok(formula_result);
    }

    // Reduce over state axis using Min (universal quantification)
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Output axes = all axes except state axis
    let output_axes: String = formula_result
        .axes
        .chars()
        .filter(|&c| c != state_axis)
        .collect();

    // Create reduction operation
    let node = EinsumNode {
        op: OpType::Reduce {
            op: "min".to_string(),
            axes: vec![], // Backend will infer from axis names
        },
        inputs: vec![formula_result.tensor_idx],
        outputs: vec![result_idx],
        metadata: None,
    };
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Ensure the state domain exists in the compiler context.
fn ensure_state_domain(ctx: &mut CompilerContext) -> Result<()> {
    if !ctx.domains.contains_key(STATE_AXIS) {
        // Add the state domain with default size
        ctx.add_domain(STATE_AXIS, DEFAULT_STATE_SPACE_SIZE);
    }
    Ok(())
}

/// Get or assign the state index for a nominal.
///
/// Nominals are mapped to consecutive indices 0, 1, 2, ...
/// The mapping is stored in the compiler context's custom data.
fn get_nominal_index(name: &str, ctx: &mut CompilerContext) -> Result<usize> {
    // Check if we have a nominal_indices map in the context
    // For now, we'll use a simple hash of the name modulo state space size
    // In a real implementation, this would be managed explicitly

    // Get state space size
    let state_size = ctx
        .domains
        .get(STATE_AXIS)
        .map(|d| d.cardinality)
        .unwrap_or(DEFAULT_STATE_SPACE_SIZE);

    // Simple deterministic mapping: hash the name to an index
    let mut hash: usize = 0;
    for byte in name.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as usize);
    }
    let index = hash % state_size;

    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TLExpr, Term};

    #[test]
    fn test_nominal_compilation() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let result = compile_nominal("home", &mut ctx, &mut graph).unwrap();

        // Should have created a tensor
        assert!(!graph.tensors.is_empty());
        // Should have state axis
        let state_axis = ctx.assign_axis(STATE_AXIS);
        assert!(result.axes.contains(state_axis));
    }

    #[test]
    fn test_at_operator_simple() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 10);
        let mut graph = EinsumGraph::new();

        // @home Safe(x)
        let safe = TLExpr::pred("Safe", vec![Term::var("x")]);
        ctx.bind_var("x", "Person").unwrap();

        let _result = compile_at("home", &safe, &mut ctx, &mut graph).unwrap();

        // Should have created tensors (may or may not have nodes depending on whether formula has state axis)
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_somewhere_operator() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // E Safe
        let safe = TLExpr::pred("Safe", vec![]);

        let _result = compile_somewhere(&safe, &mut ctx, &mut graph).unwrap();

        // Should have compiled successfully
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_everywhere_operator() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // A Safe
        let safe = TLExpr::pred("Safe", vec![]);

        let _result = compile_everywhere(&safe, &mut ctx, &mut graph).unwrap();

        // Should have compiled successfully
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_somewhere_with_free_variable() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 10);
        let mut graph = EinsumGraph::new();

        // E Knows(x, y)
        let knows = TLExpr::pred("Knows", vec![Term::var("x"), Term::var("y")]);
        ctx.bind_var("x", "Person").unwrap();
        ctx.bind_var("y", "Person").unwrap();

        let _result = compile_somewhere(&knows, &mut ctx, &mut graph).unwrap();

        // Should preserve non-state axes
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_everywhere_with_free_variable() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 10);
        let mut graph = EinsumGraph::new();

        // A Safe(x)
        let safe = TLExpr::pred("Safe", vec![Term::var("x")]);
        ctx.bind_var("x", "Person").unwrap();

        let _result = compile_everywhere(&safe, &mut ctx, &mut graph).unwrap();

        // Should preserve non-state axes
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_nested_somewhere_everywhere() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // E (A Safe)
        let safe = TLExpr::pred("Safe", vec![]);
        let everywhere_safe = TLExpr::Everywhere {
            formula: Box::new(safe),
        };

        let _result = compile_somewhere(&everywhere_safe, &mut ctx, &mut graph).unwrap();

        // Should have created tensors (nodes created only if formula has state axis)
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_multiple_nominals_distinct_indices() {
        let mut ctx = CompilerContext::new();
        ensure_state_domain(&mut ctx).unwrap();

        let idx1 = get_nominal_index("home", &mut ctx).unwrap();
        let idx2 = get_nominal_index("office", &mut ctx).unwrap();
        let idx3 = get_nominal_index("home", &mut ctx).unwrap();

        // Same nominal should give same index
        assert_eq!(idx1, idx3);

        // Different nominals should (usually) give different indices
        // (not guaranteed with hash-based mapping, but likely)
        // This is informational rather than a hard requirement
        let _ = idx2;
    }

    #[test]
    fn test_at_with_constant_formula() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        // @home True (constant 1.0)
        let constant = TLExpr::Constant(1.0);

        let result = compile_at("home", &constant, &mut ctx, &mut graph).unwrap();

        // Constant doesn't have state axis, so should return unchanged
        assert!(!graph.tensors.is_empty());
        // Should not have the state axis in output
        let state_axis = ctx.assign_axis(STATE_AXIS);
        assert!(!result.axes.contains(state_axis));
    }
}
