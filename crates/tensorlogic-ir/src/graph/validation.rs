//! Graph validation utilities for post-compilation checks.
//!
//! This module provides comprehensive validation for `EinsumGraph` instances,
//! checking for common errors and structural issues that may occur during compilation.

use crate::graph::{EinsumGraph, OpType};
use std::collections::{HashMap, HashSet};

/// Result of graph validation with detailed diagnostics.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Total number of validation checks performed
    pub checks_performed: usize,
    /// List of errors found
    pub errors: Vec<ValidationError>,
    /// List of warnings (non-fatal issues)
    pub warnings: Vec<ValidationWarning>,
    /// Graph statistics
    pub stats: GraphValidationStats,
}

/// Validation error with severity and context.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub kind: ValidationErrorKind,
    pub message: String,
    pub node_index: Option<usize>,
    pub tensor_index: Option<usize>,
}

/// Types of validation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationErrorKind {
    /// Tensor index out of bounds
    TensorOutOfBounds,
    /// Undefined tensor referenced
    UndefinedTensor,
    /// Tensor is never produced (no node writes to it)
    UnproducedTensor,
    /// Output tensor has no producer
    OutputWithoutProducer,
    /// Cyclic dependency detected
    CyclicDependency,
    /// Empty einsum specification
    EmptyEinsumSpec,
    /// Invalid einsum specification syntax
    InvalidEinsumSpec,
    /// Node has no outputs
    NoOutputs,
    /// Duplicate output (two nodes write to same tensor)
    DuplicateOutput,
}

/// Validation warning (non-fatal issue).
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub kind: ValidationWarningKind,
    pub message: String,
    pub tensor_index: Option<usize>,
    pub node_index: Option<usize>,
}

/// Types of validation warnings.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationWarningKind {
    /// Tensor is produced but never consumed
    UnusedTensor,
    /// Input tensor is never used
    UnusedInput,
    /// Tensor has unnamed or generated name
    GeneratedTensorName,
    /// Large number of operations (may be slow)
    LargeGraph,
    /// Deep operation nesting (may cause stack issues)
    DeepNesting,
}

/// Statistics about the validated graph.
#[derive(Debug, Clone, Default)]
pub struct GraphValidationStats {
    pub total_tensors: usize,
    pub total_nodes: usize,
    pub input_tensors: usize,
    pub output_tensors: usize,
    pub unused_tensors: usize,
    pub max_operation_depth: usize,
    pub einsum_operations: usize,
    pub elem_unary_operations: usize,
    pub elem_binary_operations: usize,
    pub reduce_operations: usize,
}

impl ValidationReport {
    /// Check if validation passed (no errors).
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Check if there are any issues (errors or warnings).
    pub fn has_issues(&self) -> bool {
        !self.errors.is_empty() || !self.warnings.is_empty()
    }

    /// Get a summary string of the validation results.
    pub fn summary(&self) -> String {
        format!(
            "Validation: {} errors, {} warnings ({} checks)",
            self.errors.len(),
            self.warnings.len(),
            self.checks_performed
        )
    }
}

/// Validate an `EinsumGraph` with comprehensive checks.
///
/// # Example
///
/// ```
/// use tensorlogic_ir::{EinsumGraph, EinsumNode, validate_graph};
///
/// let mut graph = EinsumGraph::new();
/// let t0 = graph.add_tensor("input".to_string());
/// let t1 = graph.add_tensor("output".to_string());
/// graph.inputs = vec![t0];
/// graph.outputs = vec![t1];
///
/// let node = EinsumNode::elem_unary("relu", t0, t1);
/// graph.add_node(node).unwrap();
///
/// let report = validate_graph(&graph);
/// assert!(report.is_valid());
/// ```
pub fn validate_graph(graph: &EinsumGraph) -> ValidationReport {
    let mut report = ValidationReport {
        checks_performed: 0,
        errors: Vec::new(),
        warnings: Vec::new(),
        stats: GraphValidationStats::default(),
    };

    // Collect statistics
    report.stats.total_tensors = graph.tensors.len();
    report.stats.total_nodes = graph.nodes.len();
    report.stats.input_tensors = graph.inputs.len();
    report.stats.output_tensors = graph.outputs.len();

    // Check 1: Tensor index bounds
    report.checks_performed += 1;
    check_tensor_bounds(graph, &mut report);

    // Check 2: Producer analysis (which tensors are written to)
    report.checks_performed += 1;
    let producers = analyze_producers(graph, &mut report);

    // Check 3: Consumer analysis (which tensors are read from)
    report.checks_performed += 1;
    let consumers = analyze_consumers(graph, &mut report);

    // Check 4: Output tensors have producers
    report.checks_performed += 1;
    check_output_producers(graph, &producers, &mut report);

    // Check 5: Check for unused tensors
    report.checks_performed += 1;
    check_unused_tensors(graph, &producers, &consumers, &mut report);

    // Check 6: Einsum specification validity
    report.checks_performed += 1;
    check_einsum_specs(graph, &mut report);

    // Check 7: Check for cycles
    report.checks_performed += 1;
    check_cycles(graph, &mut report);

    // Check 8: Nodes have outputs
    report.checks_performed += 1;
    check_node_outputs(graph, &mut report);

    // Check 9: Count operation types
    report.checks_performed += 1;
    count_operations(graph, &mut report);

    // Check 10: Check graph size warnings
    report.checks_performed += 1;
    check_graph_size(graph, &mut report);

    report
}

/// Check that all tensor indices are within bounds.
fn check_tensor_bounds(graph: &EinsumGraph, report: &mut ValidationReport) {
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &input in &node.inputs {
            if input >= graph.tensors.len() {
                report.errors.push(ValidationError {
                    kind: ValidationErrorKind::TensorOutOfBounds,
                    message: format!(
                        "Input tensor {} is out of bounds (max: {})",
                        input,
                        graph.tensors.len() - 1
                    ),
                    node_index: Some(node_idx),
                    tensor_index: Some(input),
                });
            }
        }

        for &output in &node.outputs {
            if output >= graph.tensors.len() {
                report.errors.push(ValidationError {
                    kind: ValidationErrorKind::TensorOutOfBounds,
                    message: format!(
                        "Output tensor {} is out of bounds (max: {})",
                        output,
                        graph.tensors.len() - 1
                    ),
                    node_index: Some(node_idx),
                    tensor_index: Some(output),
                });
            }
        }
    }
}

/// Analyze which nodes produce which tensors.
fn analyze_producers(graph: &EinsumGraph, report: &mut ValidationReport) -> HashMap<usize, usize> {
    let mut producers = HashMap::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output in &node.outputs {
            if let Some(existing_producer) = producers.insert(output, node_idx) {
                report.errors.push(ValidationError {
                    kind: ValidationErrorKind::DuplicateOutput,
                    message: format!(
                        "Tensor {} is produced by multiple nodes: {} and {}",
                        output, existing_producer, node_idx
                    ),
                    node_index: Some(node_idx),
                    tensor_index: Some(output),
                });
            }
        }
    }

    producers
}

/// Analyze which nodes consume which tensors.
fn analyze_consumers(
    graph: &EinsumGraph,
    _report: &mut ValidationReport,
) -> HashMap<usize, Vec<usize>> {
    let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &input in &node.inputs {
            consumers.entry(input).or_default().push(node_idx);
        }
    }

    consumers
}

/// Check that output tensors have producers.
fn check_output_producers(
    graph: &EinsumGraph,
    producers: &HashMap<usize, usize>,
    report: &mut ValidationReport,
) {
    for &output_idx in &graph.outputs {
        if output_idx >= graph.tensors.len() {
            continue; // Already reported in bounds check
        }

        if !producers.contains_key(&output_idx) && !graph.inputs.contains(&output_idx) {
            report.errors.push(ValidationError {
                kind: ValidationErrorKind::OutputWithoutProducer,
                message: format!(
                    "Output tensor {} '{}' has no producer",
                    output_idx, graph.tensors[output_idx]
                ),
                node_index: None,
                tensor_index: Some(output_idx),
            });
        }
    }
}

/// Check for unused tensors.
fn check_unused_tensors(
    graph: &EinsumGraph,
    producers: &HashMap<usize, usize>,
    consumers: &HashMap<usize, Vec<usize>>,
    report: &mut ValidationReport,
) {
    for (tensor_idx, tensor_name) in graph.tensors.iter().enumerate() {
        let is_input = graph.inputs.contains(&tensor_idx);
        let is_output = graph.outputs.contains(&tensor_idx);
        let has_producer = producers.contains_key(&tensor_idx);
        let has_consumers = consumers.contains_key(&tensor_idx);

        // Tensor is produced but never used (and not an output)
        if has_producer && !has_consumers && !is_output {
            report.warnings.push(ValidationWarning {
                kind: ValidationWarningKind::UnusedTensor,
                message: format!(
                    "Tensor {} '{}' is produced but never consumed",
                    tensor_idx, tensor_name
                ),
                tensor_index: Some(tensor_idx),
                node_index: None,
            });
            report.stats.unused_tensors += 1;
        }

        // Input tensor is never used
        if is_input && !has_consumers {
            report.warnings.push(ValidationWarning {
                kind: ValidationWarningKind::UnusedInput,
                message: format!(
                    "Input tensor {} '{}' is never consumed",
                    tensor_idx, tensor_name
                ),
                tensor_index: Some(tensor_idx),
                node_index: None,
            });
        }

        // Check for generated names (e.g., "temp_0", "t_123")
        if tensor_name.starts_with("temp_")
            || tensor_name.starts_with("t_")
            || tensor_name.starts_with("_")
        {
            report.warnings.push(ValidationWarning {
                kind: ValidationWarningKind::GeneratedTensorName,
                message: format!("Tensor {} has generated name '{}'", tensor_idx, tensor_name),
                tensor_index: Some(tensor_idx),
                node_index: None,
            });
        }
    }
}

/// Check einsum specifications for validity.
fn check_einsum_specs(graph: &EinsumGraph, report: &mut ValidationReport) {
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        if let OpType::Einsum { spec } = &node.op {
            if spec.is_empty() {
                report.errors.push(ValidationError {
                    kind: ValidationErrorKind::EmptyEinsumSpec,
                    message: "Einsum operation has empty specification".to_string(),
                    node_index: Some(node_idx),
                    tensor_index: None,
                });
            }

            // Basic syntax check: should contain "->"
            if !spec.contains("->") {
                report.errors.push(ValidationError {
                    kind: ValidationErrorKind::InvalidEinsumSpec,
                    message: format!("Einsum specification '{}' is invalid (missing '->')", spec),
                    node_index: Some(node_idx),
                    tensor_index: None,
                });
            }
        }
    }
}

/// Check for cyclic dependencies in the graph.
fn check_cycles(graph: &EinsumGraph, report: &mut ValidationReport) {
    // Build dependency map: which tensors does each node depend on
    let mut visited = HashSet::new();
    let mut rec_stack = HashSet::new();

    for node_idx in 0..graph.nodes.len() {
        if !visited.contains(&node_idx)
            && has_cycle_dfs(node_idx, graph, &mut visited, &mut rec_stack)
        {
            report.errors.push(ValidationError {
                kind: ValidationErrorKind::CyclicDependency,
                message: format!("Cyclic dependency detected involving node {}", node_idx),
                node_index: Some(node_idx),
                tensor_index: None,
            });
        }
    }
}

/// DFS helper for cycle detection.
fn has_cycle_dfs(
    node_idx: usize,
    graph: &EinsumGraph,
    visited: &mut HashSet<usize>,
    rec_stack: &mut HashSet<usize>,
) -> bool {
    visited.insert(node_idx);
    rec_stack.insert(node_idx);

    let node = &graph.nodes[node_idx];

    // Find nodes that depend on this node's outputs
    for &output in &node.outputs {
        for (next_node_idx, next_node) in graph.nodes.iter().enumerate() {
            if next_node.inputs.contains(&output) {
                if !visited.contains(&next_node_idx) {
                    if has_cycle_dfs(next_node_idx, graph, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&next_node_idx) {
                    return true;
                }
            }
        }
    }

    rec_stack.remove(&node_idx);
    false
}

/// Check that all nodes produce outputs.
fn check_node_outputs(graph: &EinsumGraph, report: &mut ValidationReport) {
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        if node.outputs.is_empty() {
            report.errors.push(ValidationError {
                kind: ValidationErrorKind::NoOutputs,
                message: format!("Node {} has no outputs", node_idx),
                node_index: Some(node_idx),
                tensor_index: None,
            });
        }
    }
}

/// Count operation types for statistics.
fn count_operations(graph: &EinsumGraph, report: &mut ValidationReport) {
    for node in &graph.nodes {
        match &node.op {
            OpType::Einsum { .. } => report.stats.einsum_operations += 1,
            OpType::ElemUnary { .. } => report.stats.elem_unary_operations += 1,
            OpType::ElemBinary { .. } => report.stats.elem_binary_operations += 1,
            OpType::Reduce { .. } => report.stats.reduce_operations += 1,
        }
    }
}

/// Check for large graphs that may have performance issues.
fn check_graph_size(graph: &EinsumGraph, report: &mut ValidationReport) {
    if graph.nodes.len() > 1000 {
        report.warnings.push(ValidationWarning {
            kind: ValidationWarningKind::LargeGraph,
            message: format!(
                "Graph has {} operations (may be slow to execute)",
                graph.nodes.len()
            ),
            tensor_index: None,
            node_index: None,
        });
    }

    if graph.tensors.len() > 10000 {
        report.warnings.push(ValidationWarning {
            kind: ValidationWarningKind::LargeGraph,
            message: format!(
                "Graph has {} tensors (may use significant memory)",
                graph.tensors.len()
            ),
            tensor_index: None,
            node_index: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EinsumGraph, EinsumNode};

    #[test]
    fn test_validate_empty_graph() {
        let graph = EinsumGraph::new();
        let report = validate_graph(&graph);
        assert!(report.is_valid());
        assert_eq!(report.errors.len(), 0);
    }

    #[test]
    fn test_validate_simple_graph() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input".to_string());
        let t1 = graph.add_tensor("output".to_string());
        graph.inputs = vec![t0];
        graph.outputs = vec![t1];

        let node = EinsumNode::elem_unary("relu", t0, t1);
        graph.add_node(node).unwrap();

        let report = validate_graph(&graph);
        assert!(report.is_valid());
        assert_eq!(report.stats.total_tensors, 2);
        assert_eq!(report.stats.total_nodes, 1);
    }

    #[test]
    fn test_detect_tensor_out_of_bounds() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input".to_string());
        graph.add_tensor("output".to_string());

        // Create node with invalid tensor index
        let bad_node = EinsumNode::elem_unary("relu", t0, 999);
        graph.nodes.push(bad_node);

        let report = validate_graph(&graph);
        assert!(!report.is_valid());
        assert_eq!(report.errors.len(), 1);
        assert_eq!(
            report.errors[0].kind,
            ValidationErrorKind::TensorOutOfBounds
        );
    }

    #[test]
    fn test_detect_unused_tensor() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input".to_string());
        let t1 = graph.add_tensor("intermediate".to_string());
        let t2 = graph.add_tensor("output".to_string());
        graph.inputs = vec![t0];
        graph.outputs = vec![t2];

        // t1 is produced but never used
        graph
            .add_node(EinsumNode::elem_unary("relu", t0, t1))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("sigmoid", t0, t2))
            .unwrap();

        let report = validate_graph(&graph);
        assert!(report.is_valid()); // No errors, just warnings
        assert_eq!(report.warnings.len(), 1);
        assert_eq!(report.warnings[0].kind, ValidationWarningKind::UnusedTensor);
    }

    #[test]
    fn test_detect_output_without_producer() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input".to_string());
        let t1 = graph.add_tensor("output".to_string());
        graph.inputs = vec![t0];
        graph.outputs = vec![t1]; // t1 is output but no node produces it

        let report = validate_graph(&graph);
        assert!(!report.is_valid());
        assert_eq!(report.errors.len(), 1);
        assert_eq!(
            report.errors[0].kind,
            ValidationErrorKind::OutputWithoutProducer
        );
    }

    #[test]
    fn test_detect_empty_einsum_spec() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input".to_string());
        let t1 = graph.add_tensor("output".to_string());

        let bad_node = EinsumNode::einsum("", vec![t0], vec![t1]);
        graph.nodes.push(bad_node);

        let report = validate_graph(&graph);
        assert!(!report.is_valid());
        assert!(report
            .errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::EmptyEinsumSpec));
    }

    #[test]
    fn test_detect_invalid_einsum_spec() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("input".to_string());
        let t1 = graph.add_tensor("output".to_string());

        let bad_node = EinsumNode::einsum("ijk", vec![t0], vec![t1]); // Missing "->"
        graph.nodes.push(bad_node);

        let report = validate_graph(&graph);
        assert!(!report.is_valid());
        assert!(report
            .errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::InvalidEinsumSpec));
    }

    #[test]
    fn test_statistics_collection() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("a".to_string());
        let t1 = graph.add_tensor("b".to_string());
        let t2 = graph.add_tensor("c".to_string());
        let t3 = graph.add_tensor("d".to_string());

        graph
            .add_node(EinsumNode::elem_unary("relu", t0, t1))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_binary("add", t1, t2, t3))
            .unwrap();

        let report = validate_graph(&graph);
        assert_eq!(report.stats.elem_unary_operations, 1);
        assert_eq!(report.stats.elem_binary_operations, 1);
        assert_eq!(report.stats.total_nodes, 2);
        assert_eq!(report.stats.total_tensors, 4);
    }
}
