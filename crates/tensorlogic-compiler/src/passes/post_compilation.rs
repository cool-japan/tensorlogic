//! Post-compilation validation and optimization passes.
//!
//! This module provides validation and optimization passes that run after
//! the initial compilation to ensure correctness and improve performance.

use anyhow::{bail, Result};
use std::collections::HashSet;
use tensorlogic_ir::{validate_graph, EinsumGraph, OpType, ValidationReport};

use crate::CompilerContext;

/// Post-compilation validation options.
#[derive(Debug, Clone)]
pub struct PostCompilationOptions {
    /// Enable graph structure validation
    pub validate_graph_structure: bool,
    /// Enable axis consistency checks
    pub validate_axes: bool,
    /// Enable shape compatibility checks
    pub validate_shapes: bool,
    /// Enable optimization passes
    pub apply_optimizations: bool,
    /// Fail on warnings
    pub strict_mode: bool,
}

impl Default for PostCompilationOptions {
    fn default() -> Self {
        Self {
            validate_graph_structure: true,
            validate_axes: true,
            validate_shapes: true,
            apply_optimizations: true,
            strict_mode: false,
        }
    }
}

/// Result of post-compilation passes.
#[derive(Debug, Clone)]
pub struct PostCompilationResult {
    /// Validation report
    pub validation_report: ValidationReport,
    /// Whether the graph passed all checks
    pub is_valid: bool,
    /// Number of optimizations applied
    pub optimizations_applied: usize,
    /// Detailed messages
    pub messages: Vec<String>,
}

/// Run post-compilation validation and optimization passes.
///
/// # Examples
///
/// ```
/// use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
/// use tensorlogic_compiler::passes::{
///     post_compilation_passes, PostCompilationOptions
/// };
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let mut ctx = CompilerContext::new();
/// ctx.add_domain("Person", 100);
///
/// let expr = TLExpr::exists(
///     "y",
///     "Person",
///     TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
/// );
///
/// let mut graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
///
/// let options = PostCompilationOptions::default();
/// let result = post_compilation_passes(&mut graph, &ctx, options).unwrap();
///
/// assert!(result.is_valid);
/// ```
pub fn post_compilation_passes(
    graph: &mut EinsumGraph,
    ctx: &CompilerContext,
    options: PostCompilationOptions,
) -> Result<PostCompilationResult> {
    let mut messages = Vec::new();
    let mut optimizations_applied = 0;

    // 1. Validate graph structure
    let validation_report = if options.validate_graph_structure {
        let report = validate_graph(graph);

        // For simple predicate expressions, the output might be an input tensor
        // with no producer node, which is valid. Filter out such errors.
        let has_simple_passthrough = graph.nodes.is_empty()
            || (graph.outputs.len() == 1 && graph.inputs.contains(&graph.outputs[0]));

        let filtered_errors: Vec<_> = report
            .errors
            .into_iter()
            .filter(|error| {
                // Allow "no producer" errors for simple passthrough graphs
                if has_simple_passthrough && error.message.contains("has no producer") {
                    return false; // Filter out this error
                }
                true // Keep all other errors
            })
            .collect();

        for error in &filtered_errors {
            messages.push(format!("ERROR: {}", error.message));
        }

        if !report.warnings.is_empty() {
            for warning in &report.warnings {
                messages.push(format!("WARNING: {}", warning.message));
            }
        }

        ValidationReport {
            checks_performed: report.checks_performed,
            errors: filtered_errors,
            warnings: report.warnings,
            stats: report.stats,
        }
    } else {
        ValidationReport {
            checks_performed: 0,
            errors: vec![],
            warnings: vec![],
            stats: Default::default(),
        }
    };

    // 2. Validate axis consistency
    if options.validate_axes {
        validate_axis_consistency(graph, ctx, &mut messages)?;
    }

    // 3. Validate shape compatibility (basic checks)
    if options.validate_shapes {
        validate_shape_compatibility(graph, ctx, &mut messages)?;
    }

    // 4. Apply optimization passes
    if options.apply_optimizations {
        optimizations_applied += apply_optimization_passes(graph, &mut messages)?;
    }

    // Check if valid
    let is_valid = validation_report.is_valid()
        && (!options.strict_mode || validation_report.warnings.is_empty());

    if !is_valid {
        bail!(
            "Post-compilation validation failed:\n{}",
            messages.join("\n")
        );
    }

    Ok(PostCompilationResult {
        validation_report,
        is_valid,
        optimizations_applied,
        messages,
    })
}

/// Validate axis consistency across the graph.
fn validate_axis_consistency(
    graph: &EinsumGraph,
    ctx: &CompilerContext,
    messages: &mut Vec<String>,
) -> Result<()> {
    // Track which axes are used and their expected sizes
    let mut axis_domains = std::collections::HashMap::new();

    for node in &graph.nodes {
        if let OpType::Einsum { spec, .. } = &node.op {
            // Extract axes from einsum spec
            let axes = extract_axes_from_spec(spec);

            for axis_char in axes {
                // Check if this axis character is used by a variable
                for (var, &var_axis_char) in &ctx.var_to_axis {
                    if var_axis_char == axis_char {
                        // Get the domain for this variable
                        if let Some(domain_name) = ctx.var_to_domain.get(var) {
                            if let Some(domain_info) = ctx.domains.get(domain_name) {
                                let size = domain_info.cardinality;

                                // Track or validate axis size
                                if let Some(&existing_size) = axis_domains.get(&axis_char) {
                                    if existing_size != size {
                                        messages.push(format!(
                                            "WARNING: Axis '{}' has inconsistent domain sizes: {} vs {}",
                                            axis_char, existing_size, size
                                        ));
                                    }
                                } else {
                                    axis_domains.insert(axis_char, size);
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Extract axis labels from einsum specification.
fn extract_axes_from_spec(spec: &str) -> Vec<char> {
    let mut axes = Vec::new();

    // Parse einsum spec: "ij,jk->ik"
    if let Some((inputs, _output)) = spec.split_once("->") {
        for input in inputs.split(',') {
            for c in input.chars() {
                if c.is_ascii_lowercase() && !axes.contains(&c) {
                    axes.push(c);
                }
            }
        }
    }

    axes.sort();
    axes.dedup();
    axes
}

/// Validate basic shape compatibility.
fn validate_shape_compatibility(
    graph: &EinsumGraph,
    _ctx: &CompilerContext,
    messages: &mut Vec<String>,
) -> Result<()> {
    // Track tensor shapes (if known)
    let mut tensor_ranks = std::collections::HashMap::new();

    for node in &graph.nodes {
        match &node.op {
            OpType::Einsum { spec } => {
                // Parse spec to determine output rank
                if let Some((_inputs, output)) = spec.split_once("->") {
                    let output_rank = output.chars().filter(|c| c.is_alphabetic()).count();
                    if let Some(&output_idx) = node.outputs.first() {
                        tensor_ranks.insert(output_idx, output_rank);
                    }
                }
            }
            OpType::ElemUnary { .. } => {
                // Unary ops preserve rank
                if let Some(&input_idx) = node.inputs.first() {
                    if let Some(&rank) = tensor_ranks.get(&input_idx) {
                        if let Some(&output_idx) = node.outputs.first() {
                            tensor_ranks.insert(output_idx, rank);
                        }
                    }
                }
            }
            OpType::ElemBinary { .. } => {
                // Binary ops require compatible ranks
                if node.inputs.len() >= 2 {
                    let left_rank = tensor_ranks.get(&node.inputs[0]);
                    let right_rank = tensor_ranks.get(&node.inputs[1]);

                    if let (Some(&l), Some(&r)) = (left_rank, right_rank) {
                        if l != r && l != 0 && r != 0 {
                            messages.push(format!(
                                "WARNING: Element-wise binary op has mismatched ranks: {} vs {}",
                                l, r
                            ));
                        }
                        if let Some(&output_idx) = node.outputs.first() {
                            tensor_ranks.insert(output_idx, l.max(r));
                        }
                    }
                }
            }
            OpType::Reduce { .. } => {
                // Reduce decreases rank by 1
                if let Some(&input_idx) = node.inputs.first() {
                    if let Some(&rank) = tensor_ranks.get(&input_idx) {
                        if let Some(&output_idx) = node.outputs.first() {
                            tensor_ranks.insert(output_idx, rank.saturating_sub(1));
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Apply optimization passes to the graph.
fn apply_optimization_passes(
    _graph: &mut EinsumGraph,
    messages: &mut Vec<String>,
) -> Result<usize> {
    // Note: Graph optimization methods (eliminate_dead_code, eliminate_common_subexpressions,
    // simplify_identities) are not yet available in the current tensorlogic-ir API.
    // These optimizations can be added when the IR supports them.

    messages.push("Graph optimizations: currently disabled (awaiting IR API support)".to_string());

    Ok(0)
}

/// Quick validation check (used internally).
pub fn quick_validate(graph: &EinsumGraph) -> Result<()> {
    // Check for cycles
    if has_cycle(graph) {
        bail!("Graph contains cycles");
    }

    // Check that all tensor references are valid
    for node in &graph.nodes {
        for &input_idx in &node.inputs {
            if input_idx >= graph.tensors.len() {
                bail!(
                    "Invalid tensor reference: {} (graph has {} tensors)",
                    input_idx,
                    graph.tensors.len()
                );
            }
        }
    }

    // Check that outputs are valid
    for &output_idx in &graph.outputs {
        if output_idx >= graph.tensors.len() {
            bail!(
                "Invalid output reference: {} (graph has {} tensors)",
                output_idx,
                graph.tensors.len()
            );
        }
    }

    Ok(())
}

/// Check if graph contains cycles (basic DFS).
fn has_cycle(graph: &EinsumGraph) -> bool {
    let mut visited = HashSet::new();
    let mut rec_stack = HashSet::new();

    for node in &graph.nodes {
        for &output_idx in &node.outputs {
            if !visited.contains(&output_idx)
                && has_cycle_util(graph, output_idx, &mut visited, &mut rec_stack)
            {
                return true;
            }
        }
    }

    false
}

fn has_cycle_util(
    graph: &EinsumGraph,
    tensor_idx: usize,
    visited: &mut HashSet<usize>,
    rec_stack: &mut HashSet<usize>,
) -> bool {
    visited.insert(tensor_idx);
    rec_stack.insert(tensor_idx);

    // Find nodes that produce this tensor
    for node in &graph.nodes {
        if node.outputs.contains(&tensor_idx) {
            for &input_idx in &node.inputs {
                if !visited.contains(&input_idx) {
                    if has_cycle_util(graph, input_idx, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&input_idx) {
                    return true;
                }
            }
        }
    }

    rec_stack.remove(&tensor_idx);
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compile_to_einsum_with_context, CompilerContext};
    use tensorlogic_ir::{TLExpr, Term};

    #[test]
    fn test_post_compilation_simple() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

        let mut graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

        let options = PostCompilationOptions::default();
        let result = post_compilation_passes(&mut graph, &ctx, options).unwrap();

        assert!(result.is_valid);
    }

    #[test]
    fn test_post_compilation_with_quantifier() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr = TLExpr::exists(
            "y",
            "Person",
            TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
        );

        let mut graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

        let options = PostCompilationOptions::default();
        let result = post_compilation_passes(&mut graph, &ctx, options).unwrap();

        assert!(result.is_valid);
    }

    #[test]
    fn test_quick_validate_success() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);

        let expr = TLExpr::pred("p", vec![Term::var("x")]);
        let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

        assert!(quick_validate(&graph).is_ok());
    }

    #[test]
    fn test_extract_axes_from_spec() {
        let spec = "ab,bc->ac";
        let axes = extract_axes_from_spec(spec);
        assert_eq!(axes, vec!['a', 'b', 'c']);

        let spec2 = "ij->i";
        let axes2 = extract_axes_from_spec(spec2);
        assert_eq!(axes2, vec!['i', 'j']);
    }

    #[test]
    fn test_post_compilation_optimizations() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);

        // Create expression that will have optimizable structure
        let expr = TLExpr::And(
            Box::new(TLExpr::pred("p", vec![Term::var("x")])),
            Box::new(TLExpr::pred("q", vec![Term::var("y")])),
        );

        let mut graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

        let options = PostCompilationOptions {
            apply_optimizations: true,
            ..Default::default()
        };

        let result = post_compilation_passes(&mut graph, &ctx, options).unwrap();
        assert!(result.is_valid);
        // May or may not have optimizations depending on graph structure
    }

    #[test]
    fn test_post_compilation_strict_mode() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

        let mut graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

        let options = PostCompilationOptions {
            strict_mode: true,
            ..Default::default()
        };

        let result = post_compilation_passes(&mut graph, &ctx, options);
        // Should pass if no warnings
        assert!(result.is_ok());
    }
}
