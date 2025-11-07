//! Compilation of Let bindings for local variable definitions.

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, TLExpr};

use crate::context::{CompileState, CompilerContext};

use super::compile_expr;

/// Compile let binding: let var = value in body
///
/// This compiles a let expression by:
/// 1. Compiling the value expression
/// 2. Binding the variable name to the value's tensor
/// 3. Compiling the body expression (which can reference the variable)
/// 4. Restoring the previous variable binding (if any)
pub(crate) fn compile_let(
    var: &str,
    value: &TLExpr,
    body: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Compile the value expression
    let value_state = compile_expr(value, ctx, graph)?;

    // Save the previous binding for this variable (if any)
    let prev_binding = ctx.var_to_domain.get(var).cloned();
    let prev_axis = ctx.var_to_axis.get(var).copied();

    // Bind the variable to the value's tensor state
    // For let bindings, we treat the variable as bound to the same domain/axis as the value
    if let Some(domain) = infer_domain_from_axes(&value_state.axes, ctx) {
        ctx.var_to_domain.insert(var.to_string(), domain);
    }

    // If the value has axes, bind the variable to the first axis
    // (this is a simplification; more complex bindings might need different logic)
    if let Some(first_axis) = value_state.axes.chars().next() {
        ctx.var_to_axis.insert(var.to_string(), first_axis);
    }

    // Store the tensor binding
    ctx.let_bindings
        .insert(var.to_string(), value_state.tensor_idx);

    // Compile the body expression (which can reference the variable)
    let body_state = compile_expr(body, ctx, graph)?;

    // Restore the previous binding
    ctx.let_bindings.remove(var);
    match prev_binding {
        Some(domain) => {
            ctx.var_to_domain.insert(var.to_string(), domain);
        }
        None => {
            ctx.var_to_domain.remove(var);
        }
    }
    match prev_axis {
        Some(axis) => {
            ctx.var_to_axis.insert(var.to_string(), axis);
        }
        None => {
            ctx.var_to_axis.remove(var);
        }
    }

    Ok(body_state)
}

/// Infer domain from axes by looking up the first axis in the context
fn infer_domain_from_axes(axes: &str, ctx: &CompilerContext) -> Option<String> {
    if axes.is_empty() {
        return None;
    }

    // Get the first axis character
    let first_axis = axes.chars().next()?;

    // Find a variable that maps to this axis
    for (var, &var_axis) in &ctx.var_to_axis {
        if var_axis == first_axis {
            if let Some(domain) = ctx.var_to_domain.get(var) {
                return Some(domain.clone());
            }
        }
    }

    None
}
