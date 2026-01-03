//! Set theory operations compilation.
//!
//! This module implements compilation of set-theoretic operations into tensor
//! computations. Sets are represented as characteristic functions (indicator tensors)
//! over domains, where 1 indicates membership and 0 indicates non-membership.
//!
//! # Representation
//!
//! A set S ⊆ Domain is represented as a tensor χ_S : Domain → {0,1} where:
//! - χ_S(x) = 1 if x ∈ S
//! - χ_S(x) = 0 if x ∉ S
//!
//! This characteristic function representation allows efficient tensor-based
//! set operations using element-wise operations and reductions.
//!
//! # Operations
//!
//! | Set Operation | Tensor Equivalent | Notes |
//! |--------------|-------------------|-------|
//! | x ∈ S | index(χ_S, x) | Element lookup |
//! | A ∪ B | max(χ_A, χ_B) | Or element-wise OR for Boolean |
//! | A ∩ B | min(χ_A, χ_B) | Or element-wise AND for Boolean |
//! | A \ B | χ_A * (1 - χ_B) | Element-wise masking |
//! | \|S\| | sum(χ_S) | Count of 1s in characteristic function |
//! | ∅ | zeros(domain_size) | All zeros tensor |
//! | {x : D \| P(x)} | P(x) for x in D | Predicate as characteristic function |

use anyhow::{bail, Result};
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr};

use crate::compile::compile_expr;
use crate::context::{CompileState, CompilerContext};

/// Compile set membership: elem ∈ set
///
/// For a set represented as a characteristic function χ_S over domain D,
/// membership elem ∈ S is compiled as χ_S[elem], which is the indicator
/// value at that element.
///
/// If elem is a constant, this becomes a tensor indexing operation.
/// If elem is a variable, this becomes an element-wise predicate check.
///
/// # Example
///
/// ```text
/// Let S = {x : Person | age(x) > 18}
/// Then: alice ∈ S compiles to age(alice) > 18
/// ```
pub(crate) fn compile_set_membership(
    element: &TLExpr,
    set: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Compile the set expression (should yield a characteristic function)
    let set_state = compile_expr(set, ctx, graph)?;

    // Compile the element expression
    let elem_state = compile_expr(element, ctx, graph)?;

    // Set membership is the element-wise product: elem_indicator * set_characteristic
    // This effectively performs an AND operation to check if the element is in the set

    // Compute output axes: union of both operands' axes
    let mut output_axes = String::new();
    let mut seen = std::collections::HashSet::new();

    for c in elem_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }

    for c in set_state.axes.chars() {
        if seen.insert(c) {
            output_axes.push(c);
        }
    }

    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Use einsum for multiplication with broadcasting
    let spec = if output_axes.is_empty() {
        ",->"
    } else if elem_state.axes == set_state.axes {
        // Same axes - direct element-wise multiplication
        graph.add_node(EinsumNode::elem_binary(
            "multiply",
            elem_state.tensor_idx,
            set_state.tensor_idx,
            result_idx,
        ))?;

        return Ok(CompileState {
            tensor_idx: result_idx,
            axes: output_axes,
        });
    } else {
        // Different axes - use einsum for broadcasting
        &format!("{},{}->{}", elem_state.axes, set_state.axes, output_axes)
    };

    graph.add_node(EinsumNode::einsum(
        spec,
        vec![elem_state.tensor_idx, set_state.tensor_idx],
        vec![result_idx],
    ))?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Compile set union: A ∪ B
///
/// Union of two sets represented as characteristic functions:
/// χ_{A∪B}(x) = max(χ_A(x), χ_B(x))
pub(crate) fn compile_set_union(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let mut left_state = compile_expr(left, ctx, graph)?;
    let mut right_state = compile_expr(right, ctx, graph)?;

    // Compute output axes: union of both sets' axes
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

    // Broadcast operands to match output_axes if needed
    if !left_state.axes.is_empty() && left_state.axes != output_axes {
        let broadcast_spec = format!("{}->{}", left_state.axes, output_axes);
        let broadcast_name = ctx.fresh_temp();
        let broadcast_idx = graph.add_tensor(broadcast_name);
        graph.add_node(EinsumNode::einsum(
            broadcast_spec,
            vec![left_state.tensor_idx],
            vec![broadcast_idx],
        ))?;
        left_state = CompileState {
            tensor_idx: broadcast_idx,
            axes: output_axes.clone(),
        };
    }

    if !right_state.axes.is_empty() && right_state.axes != output_axes {
        let broadcast_spec = format!("{}->{}", right_state.axes, output_axes);
        let broadcast_name = ctx.fresh_temp();
        let broadcast_idx = graph.add_tensor(broadcast_name);
        graph.add_node(EinsumNode::einsum(
            broadcast_spec,
            vec![right_state.tensor_idx],
            vec![broadcast_idx],
        ))?;
        right_state = CompileState {
            tensor_idx: broadcast_idx,
            axes: output_axes.clone(),
        };
    }

    // Apply element-wise max
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    graph.add_node(EinsumNode::elem_binary(
        "max",
        left_state.tensor_idx,
        right_state.tensor_idx,
        result_idx,
    ))?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Compile set intersection: A ∩ B
///
/// Intersection of two sets represented as characteristic functions:
/// χ_{A∩B}(x) = min(χ_A(x), χ_B(x))
pub(crate) fn compile_set_intersection(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let mut left_state = compile_expr(left, ctx, graph)?;
    let mut right_state = compile_expr(right, ctx, graph)?;

    // Compute output axes: union of both sets' axes
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

    // Broadcast operands to match output_axes if needed
    if !left_state.axes.is_empty() && left_state.axes != output_axes {
        let broadcast_spec = format!("{}->{}", left_state.axes, output_axes);
        let broadcast_name = ctx.fresh_temp();
        let broadcast_idx = graph.add_tensor(broadcast_name);
        graph.add_node(EinsumNode::einsum(
            broadcast_spec,
            vec![left_state.tensor_idx],
            vec![broadcast_idx],
        ))?;
        left_state = CompileState {
            tensor_idx: broadcast_idx,
            axes: output_axes.clone(),
        };
    }

    if !right_state.axes.is_empty() && right_state.axes != output_axes {
        let broadcast_spec = format!("{}->{}", right_state.axes, output_axes);
        let broadcast_name = ctx.fresh_temp();
        let broadcast_idx = graph.add_tensor(broadcast_name);
        graph.add_node(EinsumNode::einsum(
            broadcast_spec,
            vec![right_state.tensor_idx],
            vec![broadcast_idx],
        ))?;
        right_state = CompileState {
            tensor_idx: broadcast_idx,
            axes: output_axes.clone(),
        };
    }

    // Apply element-wise min
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    graph.add_node(EinsumNode::elem_binary(
        "min",
        left_state.tensor_idx,
        right_state.tensor_idx,
        result_idx,
    ))?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Compile set difference: A \ B
///
/// Set difference (relative complement) of two sets:
/// χ_{A\B}(x) = χ_A(x) * (1 - χ_B(x))
pub(crate) fn compile_set_difference(
    left: &TLExpr,
    right: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let mut left_state = compile_expr(left, ctx, graph)?;
    let mut right_state = compile_expr(right, ctx, graph)?;

    // Compute output axes
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

    // Broadcast operands if needed
    if !left_state.axes.is_empty() && left_state.axes != output_axes {
        let broadcast_spec = format!("{}->{}", left_state.axes, output_axes);
        let broadcast_name = ctx.fresh_temp();
        let broadcast_idx = graph.add_tensor(broadcast_name);
        graph.add_node(EinsumNode::einsum(
            broadcast_spec,
            vec![left_state.tensor_idx],
            vec![broadcast_idx],
        ))?;
        left_state = CompileState {
            tensor_idx: broadcast_idx,
            axes: output_axes.clone(),
        };
    }

    if !right_state.axes.is_empty() && right_state.axes != output_axes {
        let broadcast_spec = format!("{}->{}", right_state.axes, output_axes);
        let broadcast_name = ctx.fresh_temp();
        let broadcast_idx = graph.add_tensor(broadcast_name);
        graph.add_node(EinsumNode::einsum(
            broadcast_spec,
            vec![right_state.tensor_idx],
            vec![broadcast_idx],
        ))?;
        right_state = CompileState {
            tensor_idx: broadcast_idx,
            axes: output_axes.clone(),
        };
    }

    // Compute (1 - B)
    let not_right_name = ctx.fresh_temp();
    let not_right_idx = graph.add_tensor(not_right_name);
    graph.add_node(EinsumNode::elem_unary(
        "one_minus",
        right_state.tensor_idx,
        not_right_idx,
    ))?;

    // Multiply A * (1 - B)
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);
    graph.add_node(EinsumNode::elem_binary(
        "multiply",
        left_state.tensor_idx,
        not_right_idx,
        result_idx,
    ))?;

    Ok(CompileState {
        tensor_idx: result_idx,
        axes: output_axes,
    })
}

/// Compile set cardinality: |S|
///
/// Cardinality of a set represented as a characteristic function:
/// |S| = sum(χ_S)
pub(crate) fn compile_set_cardinality(
    set: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    let set_state = compile_expr(set, ctx, graph)?;

    // Cardinality is the sum of the characteristic function
    let result_name = ctx.fresh_temp();
    let result_idx = graph.add_tensor(result_name);

    // Extract axes to reduce
    let axes_to_reduce: Vec<usize> = set_state
        .axes
        .chars()
        .map(|c| (c as u8 - b'a') as usize)
        .collect();

    graph.add_node(EinsumNode::reduce(
        "sum",
        axes_to_reduce,
        set_state.tensor_idx,
        result_idx,
    ))?;

    // Result is a scalar (no axes)
    Ok(CompileState {
        tensor_idx: result_idx,
        axes: String::new(),
    })
}

/// Compile empty set: ∅
///
/// The empty set is represented as a constant zero scalar.
pub(crate) fn compile_empty_set(
    _ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Empty set is a constant zero tensor
    let tensor_name = "const_0.0";
    let tensor_idx = graph.add_tensor(tensor_name);

    // The backend will initialize this as a zero tensor
    // No node is needed - just the tensor definition

    // Empty set has no axes (it's a scalar)
    Ok(CompileState {
        tensor_idx,
        axes: String::new(),
    })
}

/// Compile set comprehension: { var : domain | condition }
///
/// Set comprehension creates a set by filtering elements from a domain
/// based on a condition. The characteristic function is the condition
/// predicate evaluated over the domain.
pub(crate) fn compile_set_comprehension(
    var: &str,
    domain: &str,
    condition: &TLExpr,
    ctx: &mut CompilerContext,
    graph: &mut EinsumGraph,
) -> Result<CompileState> {
    // Ensure the domain exists
    if !ctx.domains.contains_key(domain) {
        bail!(
            "Domain '{}' not found for set comprehension variable '{}'",
            domain,
            var
        );
    }

    // Bind the variable to the domain
    ctx.bind_var(var, domain)?;

    // Compile the condition (this is the characteristic function)
    let cond_state = compile_expr(condition, ctx, graph)?;

    // The result is the condition tensor itself (it's the characteristic function)
    Ok(cond_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{TLExpr, Term};

    #[test]
    fn test_empty_set_compilation() {
        let mut ctx = CompilerContext::new();
        let mut graph = EinsumGraph::new();

        let result = compile_empty_set(&mut ctx, &mut graph).unwrap();

        // Empty set should be a scalar (no axes)
        assert!(result.axes.is_empty());
    }

    #[test]
    fn test_set_comprehension_compilation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        let mut graph = EinsumGraph::new();

        // { x : Person | P(x) }
        let condition = TLExpr::pred("P", vec![Term::var("x")]);

        let result =
            compile_set_comprehension("x", "Person", &condition, &mut ctx, &mut graph).unwrap();

        // Should have axes for the variable
        assert!(!result.axes.is_empty());
    }

    #[test]
    fn test_set_union_compilation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        let mut graph = EinsumGraph::new();

        // Create two set comprehensions
        let set_a = TLExpr::SetComprehension {
            var: "x".to_string(),
            domain: "Person".to_string(),
            condition: Box::new(TLExpr::pred("A", vec![Term::var("x")])),
        };

        let set_b = TLExpr::SetComprehension {
            var: "x".to_string(),
            domain: "Person".to_string(),
            condition: Box::new(TLExpr::pred("B", vec![Term::var("x")])),
        };

        let result = compile_set_union(&set_a, &set_b, &mut ctx, &mut graph).unwrap();

        // Should have axes
        assert!(!result.axes.is_empty());
    }

    #[test]
    fn test_set_intersection_compilation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        let mut graph = EinsumGraph::new();

        let set_a = TLExpr::SetComprehension {
            var: "x".to_string(),
            domain: "Person".to_string(),
            condition: Box::new(TLExpr::pred("A", vec![Term::var("x")])),
        };

        let set_b = TLExpr::SetComprehension {
            var: "x".to_string(),
            domain: "Person".to_string(),
            condition: Box::new(TLExpr::pred("B", vec![Term::var("x")])),
        };

        let result = compile_set_intersection(&set_a, &set_b, &mut ctx, &mut graph).unwrap();

        assert!(!result.axes.is_empty());
    }

    #[test]
    fn test_set_difference_compilation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        let mut graph = EinsumGraph::new();

        let set_a = TLExpr::SetComprehension {
            var: "x".to_string(),
            domain: "Person".to_string(),
            condition: Box::new(TLExpr::pred("A", vec![Term::var("x")])),
        };

        let set_b = TLExpr::SetComprehension {
            var: "x".to_string(),
            domain: "Person".to_string(),
            condition: Box::new(TLExpr::pred("B", vec![Term::var("x")])),
        };

        let result = compile_set_difference(&set_a, &set_b, &mut ctx, &mut graph).unwrap();

        assert!(!result.axes.is_empty());
    }

    #[test]
    fn test_set_cardinality_compilation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        let mut graph = EinsumGraph::new();

        let set_expr = TLExpr::SetComprehension {
            var: "x".to_string(),
            domain: "Person".to_string(),
            condition: Box::new(TLExpr::pred("Adult", vec![Term::var("x")])),
        };

        let result = compile_set_cardinality(&set_expr, &mut ctx, &mut graph).unwrap();

        // Cardinality should be a scalar (no axes)
        assert!(result.axes.is_empty());
    }

    #[test]
    fn test_set_membership_compilation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        let mut graph = EinsumGraph::new();

        // Create a set
        let set_expr = TLExpr::SetComprehension {
            var: "x".to_string(),
            domain: "Person".to_string(),
            condition: Box::new(TLExpr::pred("Adult", vec![Term::var("x")])),
        };

        // Element to check
        let elem = TLExpr::pred("IsAlice", vec![Term::var("y")]);

        let result = compile_set_membership(&elem, &set_expr, &mut ctx, &mut graph).unwrap();

        // Should have axes
        assert!(!result.axes.is_empty());
    }
}
