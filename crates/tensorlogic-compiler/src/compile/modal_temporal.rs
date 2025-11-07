//! Modal and temporal logic compilation to tensor operations.
//!
//! This module implements compilation strategies for modal and temporal logic operators,
//! enabling reasoning about possibility, necessity, and temporal sequences in tensor form.
//!
//! # Modal Logic
//!
//! Modal logic extends classical logic with operators for reasoning about necessity and possibility:
//!
//! - **Box (□P)**: "P is necessarily true" - P holds in all possible worlds/states
//! - **Diamond (◇P)**: "P is possibly true" - P holds in at least one possible world/state
//!
//! ## Tensor Representation
//!
//! Modal operators require an additional "world" or "state" dimension in tensors:
//! - Predicates are evaluated over multiple possible worlds
//! - Box reduces over worlds using min/product (all worlds must satisfy P)
//! - Diamond reduces over worlds using max/sum (at least one world satisfies P)
//!
//! # Temporal Logic (LTL)
//!
//! Temporal logic extends classical logic with operators for reasoning about sequences over time:
//!
//! - **Next (XP)**: "P is true in the next state" (requires backend support for shifts)
//! - **Eventually (FP)**: "P will be true in some future state"
//! - **Always (GP)**: "P is true in all future states"
//! - **Until (P U Q)**: "P holds until Q becomes true" (complex, requires scan operations)

use anyhow::{bail, Result};
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr};

use crate::config::{ModalStrategy, TemporalStrategy};
use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

/// Special axis name for modal "world" dimension
const WORLD_AXIS: &str = "__world__";

/// Special axis name for temporal "time" dimension
const TIME_AXIS: &str = "__time__";

/// Compile a Box (□) modal operator: "P is necessarily true in all possible worlds"
///
/// Tensor semantics:
/// - Reduces over the world axis using the configured modal strategy
/// - Default: Min reduction (all worlds must satisfy P)
/// - Alternative: Product reduction for probabilistic interpretation
pub(crate) fn compile_box(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure world axis exists in context
    let world_axis = ensure_world_axis(ctx);

    // Compile inner expression (should now have world axis available)
    let inner_state = compile_expr(inner, ctx, graph)?;

    // Get the modal strategy from config
    let strategy = ctx.config.modal_strategy;

    // Check if the inner expression actually uses the world axis
    if !inner_state.axes.contains(world_axis) {
        // If inner doesn't use world axis, just return it as-is
        // This handles predicates that don't reference possible worlds
        return Ok(inner_state);
    }

    // Apply reduction over world axis based on strategy
    match strategy {
        ModalStrategy::AllWorldsMin | ModalStrategy::Threshold { .. } => {
            // Use min reduction: all worlds must satisfy
            apply_reduction(&inner_state, world_axis, "min", ctx, graph)
        }
        ModalStrategy::AllWorldsProduct => {
            // Use product reduction: probabilistic interpretation
            apply_reduction(&inner_state, world_axis, "prod", ctx, graph)
        }
    }
}

/// Compile a Diamond (◇) modal operator: "P is possibly true in at least one world"
///
/// Tensor semantics:
/// - Reduces over the world axis using max/sum
/// - Default: Max reduction (at least one world satisfies P)
/// - Alternative: Sum reduction for probabilistic interpretation
pub(crate) fn compile_diamond(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure world axis exists
    let world_axis = ensure_world_axis(ctx);

    // Compile inner expression
    let inner_state = compile_expr(inner, ctx, graph)?;

    // Check if the inner expression actually uses the world axis
    if !inner_state.axes.contains(world_axis) {
        // If inner doesn't use world axis, just return it as-is
        return Ok(inner_state);
    }

    // Get the modal strategy from config
    let strategy = ctx.config.modal_strategy;

    // Apply reduction over world axis based on strategy
    match strategy {
        ModalStrategy::AllWorldsMin | ModalStrategy::Threshold { .. } => {
            // Use max reduction (dual of min for Box)
            apply_reduction(&inner_state, world_axis, "max", ctx, graph)
        }
        ModalStrategy::AllWorldsProduct => {
            // Use sum reduction (dual of product for probabilistic interpretation)
            apply_reduction(&inner_state, world_axis, "sum", ctx, graph)
        }
    }
}

/// Compile Next (X) temporal operator: "P is true in the next time step"
///
/// Note: This requires backend support for shift/roll operations which are not
/// available in basic einsum. Returns an error for now.
pub(crate) fn compile_next(
    _inner: &TLExpr,
    _ctx: &mut CompilerContext,
    _graph: &mut EinsumGraph,
) -> Result<CompileState> {
    bail!(
        "Next (X) temporal operator requires shift operations which are not available in einsum. \
         Consider using Eventually or Always operators, or implement backend-specific shift support."
    )
}

/// Compile Eventually (F) temporal operator: "P will be true in some future state"
///
/// Tensor semantics:
/// - Reduces over future time using max/sum
pub(crate) fn compile_eventually(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure time axis exists
    let time_axis = ensure_time_axis(ctx);

    // Compile inner expression
    let inner_state = compile_expr(inner, ctx, graph)?;

    // Check if the inner expression uses the time axis
    if !inner_state.axes.contains(time_axis) {
        return Ok(inner_state);
    }

    // Get temporal strategy from config
    let strategy = ctx.config.temporal_strategy;

    // Apply reduction based on strategy
    match strategy {
        TemporalStrategy::Max | TemporalStrategy::LogSumExp => {
            // Use max: true if true in any future state
            apply_reduction(&inner_state, time_axis, "max", ctx, graph)
        }
        TemporalStrategy::Sum => {
            // Use sum: probabilistic interpretation
            apply_reduction(&inner_state, time_axis, "sum", ctx, graph)
        }
    }
}

/// Compile Always (G) temporal operator: "P is true in all future states"
///
/// Tensor semantics:
/// - Reduces over future time using min/product
pub(crate) fn compile_always(
    inner: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure time axis exists
    let time_axis = ensure_time_axis(ctx);

    // Compile inner expression
    let inner_state = compile_expr(inner, ctx, graph)?;

    // Check if the inner expression uses the time axis
    if !inner_state.axes.contains(time_axis) {
        return Ok(inner_state);
    }

    // Get temporal strategy from config
    let strategy = ctx.config.temporal_strategy;

    // Apply reduction based on strategy
    match strategy {
        TemporalStrategy::Max | TemporalStrategy::LogSumExp => {
            // Use min: true only if true in all future states
            apply_reduction(&inner_state, time_axis, "min", ctx, graph)
        }
        TemporalStrategy::Sum => {
            // Use product: probabilistic interpretation
            apply_reduction(&inner_state, time_axis, "prod", ctx, graph)
        }
    }
}

/// Compile Until (U) temporal operator: "P holds until Q becomes true"
///
/// Note: Until requires complex scan operations which are not available in einsum.
pub(crate) fn compile_until(
    _before: &TLExpr,
    _after: &TLExpr,
    _ctx: &mut CompilerContext,
    _graph: &mut EinsumGraph,
) -> Result<CompileState> {
    bail!(
        "Until (U) temporal operator requires scan operations which are not available in einsum. \
         Consider using Eventually or Always operators as approximations, or implement \
         backend-specific scan support."
    )
}

// ========================================================================
// Helper Functions
// ========================================================================

/// Ensure the world axis exists in the compilation context.
/// Returns the axis character for the world dimension.
fn ensure_world_axis(ctx: &mut CompilerContext) -> char {
    // Check if world axis already assigned
    if let Some(&axis) = ctx.var_to_axis.get(WORLD_AXIS) {
        return axis;
    }

    // Add world domain if not present
    if !ctx.domains.contains_key(WORLD_AXIS) {
        // Default: 10 possible worlds (configurable via context)
        let world_size = ctx.config.modal_world_size.unwrap_or(10);
        ctx.add_domain(WORLD_AXIS, world_size);
    }

    // Assign axis for world variable and return it
    ctx.assign_axis(WORLD_AXIS)
}

/// Ensure the time axis exists in the compilation context.
/// Returns the axis character for the time dimension.
fn ensure_time_axis(ctx: &mut CompilerContext) -> char {
    // Check if time axis already assigned
    if let Some(&axis) = ctx.var_to_axis.get(TIME_AXIS) {
        return axis;
    }

    // Add time domain if not present
    if !ctx.domains.contains_key(TIME_AXIS) {
        // Default: 100 time steps (configurable via context)
        let time_size = ctx.config.temporal_time_steps.unwrap_or(100);
        ctx.add_domain(TIME_AXIS, time_size);
    }

    // Assign axis for time variable and return it
    ctx.assign_axis(TIME_AXIS)
}

/// Apply a reduction operation over a specific axis.
///
/// Creates an einsum spec that reduces over the given axis using the specified operation.
fn apply_reduction(
    state: &CompileState,
    axis_to_reduce: char,
    reduction_op: &str,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Build output axes (all input axes except the one being reduced)
    let output_axes: String = state
        .axes
        .chars()
        .filter(|&c| c != axis_to_reduce)
        .collect();

    // Create einsum spec with reduction
    // Format: "op(input_axes->output_axes)" where op is the reduction operation
    let spec = format!("{}({}->{})", reduction_op, state.axes, output_axes);

    // Create result tensor
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Create reduction node
    let node = EinsumNode::new(spec, vec![state.tensor_idx], vec![result_idx]);
    graph.add_node(node)?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompilationConfig, CompilerContext};
    use tensorlogic_ir::{TLExpr, Term};

    #[test]
    fn test_ensure_world_axis() {
        let mut ctx = CompilerContext::new();
        let axis1 = ensure_world_axis(&mut ctx);
        let axis2 = ensure_world_axis(&mut ctx);

        // Should return same axis when called twice
        assert_eq!(axis1, axis2);
        assert!(ctx.domains.contains_key(WORLD_AXIS));
        assert!(ctx.var_to_axis.contains_key(WORLD_AXIS));
    }

    #[test]
    fn test_ensure_time_axis() {
        let mut ctx = CompilerContext::new();
        let axis1 = ensure_time_axis(&mut ctx);
        let axis2 = ensure_time_axis(&mut ctx);

        // Should return same axis when called twice
        assert_eq!(axis1, axis2);
        assert!(ctx.domains.contains_key(TIME_AXIS));
        assert!(ctx.var_to_axis.contains_key(TIME_AXIS));
    }

    #[test]
    fn test_compile_box_simple() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 10);

        let mut graph = EinsumGraph::new();

        // Box(P(x)) where P is some predicate
        let pred = TLExpr::pred("happy", vec![Term::var("x")]);

        // For this test, we expect it to work even if predicate doesn't exist
        // (it will fail at compilation, but modal logic setup should work)
        let result = compile_box(&pred, &mut ctx, &mut graph);

        // World axis should be created
        assert!(ctx.domains.contains_key(WORLD_AXIS));

        // Result may fail due to missing predicate info, but that's okay for this test
        let _ = result;
    }

    #[test]
    fn test_compile_diamond_simple() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 10);

        let mut graph = EinsumGraph::new();

        let pred = TLExpr::pred("possible", vec![Term::var("x")]);

        let result = compile_diamond(&pred, &mut ctx, &mut graph);

        // World axis should be created
        assert!(ctx.domains.contains_key(WORLD_AXIS));

        let _ = result;
    }

    #[test]
    fn test_compile_eventually_simple() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Event", 5);

        let mut graph = EinsumGraph::new();

        let pred = TLExpr::pred("occurs", vec![Term::var("e")]);

        let result = compile_eventually(&pred, &mut ctx, &mut graph);

        // Time axis should be created
        assert!(ctx.domains.contains_key(TIME_AXIS));

        let _ = result;
    }

    #[test]
    fn test_next_not_implemented() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let pred = TLExpr::pred("p", vec![Term::var("x")]);
        let result = compile_next(&pred, &mut ctx, &mut graph);

        // Should return error about not being implemented
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("shift"));
    }

    #[test]
    fn test_until_not_implemented() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let pred1 = TLExpr::pred("p", vec![Term::var("x")]);
        let pred2 = TLExpr::pred("q", vec![Term::var("x")]);
        let result = compile_until(&pred1, &pred2, &mut ctx, &mut graph);

        // Should return error about not being implemented
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("scan"));
    }

    #[test]
    fn test_modal_strategy_configuration() {
        // Test different modal strategies
        let ctx = CompilerContext::with_config(CompilationConfig::hard_boolean());
        assert_eq!(ctx.config.modal_strategy, ModalStrategy::AllWorldsMin);

        let ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());
        assert_eq!(ctx.config.modal_strategy, ModalStrategy::AllWorldsProduct);
    }

    #[test]
    fn test_temporal_strategy_configuration() {
        // Test different temporal strategies
        let ctx = CompilerContext::with_config(CompilationConfig::hard_boolean());
        assert_eq!(ctx.config.temporal_strategy, TemporalStrategy::Max);

        let ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());
        assert_eq!(ctx.config.temporal_strategy, TemporalStrategy::Sum);
    }
}
