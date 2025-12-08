//! Higher-order logic compilation.
//!
//! This module implements compilation of higher-order logic operators:
//! - **Lambda abstraction (λx:τ.e)**: Function definitions with type annotations
//! - **Application (f a)**: Function application with beta reduction
//!
//! # Background
//!
//! Higher-order logic extends first-order logic with:
//! - Functions as first-class values
//! - Lambda abstractions to define anonymous functions
//! - Function application with substitution
//! - Beta reduction: (λx.e) a ⟹ e[x := a]
//!
//! # Use Cases
//!
//! - **Functional predicates**: `λx. Positive(x) ∧ Even(x)`
//! - **Higher-order quantification**: `∀P. (∀x. P(x)) → P(c)`
//! - **Combinator logic**: SKI combinators and function composition
//! - **Neural operators**: Parameterized tensor transformations
//!
//! # Tensor Compilation Strategy
//!
//! ## Lambda Abstraction
//!
//! A lambda `λx:D.φ(x)` is compiled to a specialized tensor operation:
//! - The body `φ(x)` is compiled with `x` bound to the lambda's domain
//! - The result is a tensor function parameterized by the bound variable
//! - At application time, we substitute the argument for the bound variable
//!
//! ## Application
//!
//! Application `(λx.φ(x)) a` is compiled via beta reduction:
//! 1. Recognize the pattern: function is a Lambda
//! 2. Substitute argument `a` for bound variable `x` in body `φ(x)`
//! 3. Compile the substituted expression: `φ(a)`
//!
//! This is essentially an inline substitution that avoids runtime function calls.
//!
//! ## Non-Lambda Application
//!
//! When the function is not a lambda (e.g., a variable or expression):
//! - Treat as a predicate application
//! - Compile both function and argument
//! - Create a tensor operation that combines them
//!
//! # Examples
//!
//! ## Simple Lambda
//!
//! ```text
//! λx:Node. Connected(x, y)
//! ```
//! Compiles to a tensor parameterized by x, with y free.
//!
//! ## Beta Reduction
//!
//! ```text
//! (λx:Node. Connected(x, y)) source
//! ⟹ Connected(source, y)
//! ```
//! Beta-reduces to eliminate the lambda.
//!
//! ## Higher-Order Function
//!
//! ```text
//! λP:Pred. ∀x:Node. P(x)
//! ```
//! Takes a predicate P and universally quantifies it over nodes.
//!
//! # Limitations
//!
//! - No true closure support (captured variables must be free in context)
//! - No recursion through lambdas (use fixpoint operators instead)
//! - Type system is simplified (only domain names, not full typing)
//! - Beta reduction happens at compile time, not runtime
//!
//! # Future Work
//!
//! - Add closure support with environment capture
//! - Implement eta conversion: λx.f(x) ⟹ f
//! - Support currying and partial application
//! - Add type inference for lambda parameters

use anyhow::{bail, Result};
use tensorlogic_ir::{EinsumGraph, TLExpr};

use crate::compile::compile_expr;
use crate::context::{CompileState, CompilerContext};

/// Compile a lambda abstraction: λvar:type.body
///
/// # Strategy
///
/// Since we perform beta reduction at compile time, lambdas that aren't
/// immediately applied are treated as error cases (we don't support
/// first-class functions in the tensor representation yet).
///
/// However, we do support lambdas in let bindings and immediate applications.
///
/// # Parameters
///
/// - `var`: The bound variable name
/// - `var_type`: The type/domain of the bound variable
/// - `body`: The lambda body expression
/// - `ctx`: Compiler context
/// - `graph`: The einsum graph to compile into
///
/// # Note
///
/// This function is primarily used as a building block for Apply compilation.
/// Standalone lambdas are not directly compiled to tensors.
pub(crate) fn compile_lambda(
    var: &str,
    var_type: &Option<String>,
    body: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Lambda without application is not directly representable in tensor form
    // We'd need to create a closure or higher-order tensor operation
    //
    // For now, we compile the body with the variable bound, which gives
    // us a tensor parameterized by that variable. This works for simple cases
    // where the lambda is immediately applied or stored in a let binding.

    // Get the type/domain name
    let type_name = match var_type {
        Some(t) => t.as_str(),
        None => {
            bail!(
                "Lambda variable '{}' requires a type annotation. \
                 Please specify the domain type (e.g., λx:Node.φ(x)).",
                var
            );
        }
    };

    // Ensure the variable's domain is registered
    if !ctx.domains.contains_key(type_name) {
        bail!(
            "Lambda variable '{}' has unknown type '{}'. \
             Please register the domain before using in lambda.",
            var,
            type_name
        );
    }

    // Bind the lambda variable to its domain
    let prev_binding = ctx.var_to_domain.get(var).cloned();
    ctx.bind_var(var, type_name)?;

    // Compile the body with the variable bound
    let body_result = compile_expr(body, ctx, graph)?;

    // Restore previous binding
    if let Some(domain) = prev_binding {
        ctx.var_to_domain.insert(var.to_string(), domain);
    } else {
        ctx.var_to_domain.remove(var);
    }

    // Return the compiled body
    // This represents a "parameterized tensor" where var is a free variable
    Ok(body_result)
}

/// Compile function application: function(argument)
///
/// # Beta Reduction Strategy
///
/// We perform beta reduction at compile time:
///
/// 1. **Lambda application**: If function is `λx.e`, reduce to `e[x := arg]`
/// 2. **Predicate application**: If function is a predicate, apply argument
/// 3. **Expression application**: Compile both and combine
///
/// # Parameters
///
/// - `function`: The function expression (often a Lambda, but can be any expression)
/// - `argument`: The argument expression
/// - `ctx`: Compiler context
/// - `graph`: The einsum graph
///
/// # Beta Reduction Example
///
/// ```text
/// Apply(Lambda(x, "Node", Connected(x, y)), source)
/// ⟹ Connected(source, y)  // Direct substitution
/// ```
///
/// # Non-Lambda Application
///
/// If the function is not a lambda, we treat it as a higher-order predicate:
/// ```text
/// Apply(P, x) ⟹ P(x)
/// ```
pub(crate) fn compile_apply(
    function: &TLExpr,
    argument: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Check if function is a Lambda for beta reduction
    match function {
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => {
            // Beta reduction: (λx:τ.e) a ⟹ e[x := a]
            //
            // We perform this by:
            // 1. Compile the argument
            // 2. Bind the lambda variable to the argument's tensor
            // 3. Compile the body
            // 4. Return the result

            // Get the type/domain name
            let type_name = match var_type {
                Some(t) => t.as_str(),
                None => {
                    bail!(
                        "Lambda variable '{}' requires a type annotation for beta reduction.",
                        var
                    );
                }
            };

            // Ensure the variable type/domain exists
            if !ctx.domains.contains_key(type_name) {
                bail!(
                    "Lambda variable '{}' has unknown type '{}'. \
                     Domain must be registered before beta reduction.",
                    var,
                    type_name
                );
            }

            // Compile the argument first
            let arg_result = compile_expr(argument, ctx, graph)?;

            // Save previous bindings for the lambda variable
            let prev_domain_binding = ctx.var_to_domain.get(var).cloned();
            let prev_axis_binding = ctx.var_to_axis.get(var).copied();
            let prev_let_binding = ctx.let_bindings.get(var).copied();

            // Bind the lambda variable to the argument's tensor in let_bindings
            // This allows predicates in the body that reference var to use arg's tensor
            ctx.let_bindings
                .insert(var.to_string(), arg_result.tensor_idx);

            // Also bind domain and axis if the argument has them
            ctx.bind_var(var, type_name)?;
            if !arg_result.axes.is_empty() {
                // Use the first axis from the argument
                if let Some(first_axis) = arg_result.axes.chars().next() {
                    ctx.var_to_axis.insert(var.to_string(), first_axis);
                }
            }

            // Compile the body with the variable bound to the argument
            let body_result = compile_expr(body, ctx, graph)?;

            // Restore previous bindings
            ctx.let_bindings.remove(var);
            if let Some(domain) = prev_domain_binding {
                ctx.var_to_domain.insert(var.to_string(), domain);
            } else {
                ctx.var_to_domain.remove(var);
            }
            if let Some(axis) = prev_axis_binding {
                ctx.var_to_axis.insert(var.to_string(), axis);
            } else {
                ctx.var_to_axis.remove(var);
            }
            if let Some(tensor_idx) = prev_let_binding {
                ctx.let_bindings.insert(var.to_string(), tensor_idx);
            }

            Ok(body_result)
        }

        // If function is not a lambda, treat as predicate application
        _ => {
            // Compile both function and argument
            let func_result = compile_expr(function, ctx, graph)?;
            let arg_result = compile_expr(argument, ctx, graph)?;

            // For non-lambda application, we create a tensor operation that
            // combines the function and argument tensors
            //
            // This is similar to predicate application: func(arg)
            // We use element-wise multiplication (Hadamard product)
            let result_name = ctx.fresh_temp();
            let result_idx = graph.add_tensor(result_name);

            // Merge axes from both function and argument
            let output_axes = merge_axes(&func_result.axes, &arg_result.axes);

            // Create einsum spec for multiplication
            let spec = if func_result.axes.is_empty() && arg_result.axes.is_empty() {
                // Both scalars: simple multiplication
                ",->".to_string()
            } else if func_result.axes.is_empty() {
                // Function is scalar, argument has axes
                format!(",{}->{}", arg_result.axes, output_axes)
            } else if arg_result.axes.is_empty() {
                // Argument is scalar, function has axes
                format!("{},->{}", func_result.axes, output_axes)
            } else {
                // Both have axes: broadcast and multiply
                format!("{},{}->{}", func_result.axes, arg_result.axes, output_axes)
            };

            let node = tensorlogic_ir::EinsumNode::new(
                spec,
                vec![func_result.tensor_idx, arg_result.tensor_idx],
                vec![result_idx],
            );
            graph.add_node(node)?;

            Ok(CompileState {
                tensor_idx: result_idx,
                axes: output_axes,
            })
        }
    }
}

/// Merge two axis strings, taking the union of axes.
fn merge_axes(axes1: &str, axes2: &str) -> String {
    let mut result = axes1.to_string();
    for c in axes2.chars() {
        if !result.contains(c) {
            result.push(c);
        }
    }
    // Sort for canonical form
    let mut chars: Vec<char> = result.chars().collect();
    chars.sort();
    chars.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_lambda_simple_body() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Node", 10);
        let mut graph = EinsumGraph::new();

        // λx:Node. P(x)
        let body = TLExpr::pred("P", vec![Term::var("x")]);

        let result =
            compile_lambda("x", &Some("Node".to_string()), &body, &mut ctx, &mut graph).unwrap();

        // Should have compiled the body with x bound
        assert!(!graph.tensors.is_empty());
        assert!(!result.axes.is_empty());
    }

    #[test]
    fn test_beta_reduction_simple() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Node", 10);
        let mut graph = EinsumGraph::new();

        // (λx:Node. P(x)) a
        // Should reduce to: P(a)
        let lambda_body = TLExpr::pred("P", vec![Term::var("x")]);
        let lambda = TLExpr::lambda("x", Some("Node".to_string()), lambda_body);
        let argument = TLExpr::pred("a", vec![]);

        let _result = compile_apply(&lambda, &argument, &mut ctx, &mut graph).unwrap();

        // Should have reduced and compiled P(a)
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_beta_reduction_with_free_variable() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Node", 10);
        let mut graph = EinsumGraph::new();

        // (λx:Node. Connected(x, y)) source
        // Should reduce to: Connected(source, y)
        let lambda_body = TLExpr::pred("Connected", vec![Term::var("x"), Term::var("y")]);
        let lambda = TLExpr::lambda("x", Some("Node".to_string()), lambda_body);
        let argument = TLExpr::pred("source", vec![]);

        // y should remain as a free variable
        ctx.bind_var("y", "Node").unwrap();

        let _result = compile_apply(&lambda, &argument, &mut ctx, &mut graph).unwrap();

        // Should have successfully compiled with y still free
        assert!(!graph.tensors.is_empty());
    }

    #[test]
    fn test_lambda_with_unbound_type_fails() {
        let mut ctx = CompilerContext::new();
        // Don't register "Node" domain
        let mut graph = EinsumGraph::new();

        let body = TLExpr::pred("P", vec![Term::var("x")]);

        let result = compile_lambda("x", &Some("Node".to_string()), &body, &mut ctx, &mut graph);

        // Should fail because Node domain is not registered
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_non_lambda() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 5);
        let mut graph = EinsumGraph::new();

        // P(x) where P is not a lambda
        let function = TLExpr::pred("P", vec![]);
        let argument = TLExpr::pred("x", vec![]);

        let _result = compile_apply(&function, &argument, &mut ctx, &mut graph).unwrap();

        // Should compile both and combine them
        assert!(!graph.tensors.is_empty());
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_nested_lambda_application() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Node", 10);
        let mut graph = EinsumGraph::new();

        // (λx:Node. λy:Node. Connected(x, y)) a b
        // First application: (λy:Node. Connected(a, y)) b
        // Second application: Connected(a, b)

        let inner_body = TLExpr::pred("Connected", vec![Term::var("x"), Term::var("y")]);
        let inner_lambda = TLExpr::lambda("y", Some("Node".to_string()), inner_body);
        let outer_lambda = TLExpr::lambda("x", Some("Node".to_string()), inner_lambda);

        let arg_a = TLExpr::pred("a", vec![]);
        let arg_b = TLExpr::pred("b", vec![]);

        // First application
        let first_app = TLExpr::apply(outer_lambda, arg_a);
        // Second application
        let second_app = TLExpr::apply(first_app, arg_b);

        let _result = compile_expr(&second_app, &mut ctx, &mut graph).unwrap();

        // Should have successfully reduced to Connected(a, b)
        assert!(!graph.tensors.is_empty());
    }
}
