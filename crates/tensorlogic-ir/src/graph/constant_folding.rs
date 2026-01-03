//! Constant propagation and folding optimizations.
//!
//! This module identifies constant tensors and folds constant computations
//! at compile time to reduce runtime overhead.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{EinsumGraph, EinsumNode, IrError, OpType};

/// Information about a constant tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstantInfo {
    /// Tensor index
    pub tensor_idx: usize,
    /// Whether the value is known at compile time
    pub is_compile_time_constant: bool,
    /// Whether the value is an identity (e.g., 1 for multiplication)
    pub is_identity: bool,
    /// Whether the value is zero/absorbing
    pub is_zero: bool,
}

/// Result of constant propagation analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstantPropagationResult {
    /// Set of constant tensor indices
    pub constant_tensors: HashSet<usize>,
    /// Detailed information about each constant
    pub constant_info: HashMap<usize, ConstantInfo>,
    /// Number of operations that can be constant-folded
    pub foldable_operations: usize,
    /// Estimated speedup from constant folding
    pub estimated_speedup: f64,
}

impl ConstantPropagationResult {
    /// Create a result with no constants.
    pub fn none() -> Self {
        Self {
            constant_tensors: HashSet::new(),
            constant_info: HashMap::new(),
            foldable_operations: 0,
            estimated_speedup: 1.0,
        }
    }

    /// Check if a tensor is constant.
    pub fn is_constant(&self, tensor_idx: usize) -> bool {
        self.constant_tensors.contains(&tensor_idx)
    }

    /// Get detailed information about a constant tensor.
    pub fn get_info(&self, tensor_idx: usize) -> Option<&ConstantInfo> {
        self.constant_info.get(&tensor_idx)
    }
}

/// Statistics from constant folding transformation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoldingStats {
    /// Number of operations folded
    pub operations_folded: usize,
    /// Number of operations simplified (e.g., x * 1 → x)
    pub operations_simplified: usize,
    /// Number of operations eliminated (e.g., x * 0 → 0)
    pub operations_eliminated: usize,
    /// Estimated speedup
    pub estimated_speedup: f64,
}

impl FoldingStats {
    /// Create statistics with no transformations.
    pub fn none() -> Self {
        Self {
            operations_folded: 0,
            operations_simplified: 0,
            operations_eliminated: 0,
            estimated_speedup: 1.0,
        }
    }

    /// Total number of transformations.
    pub fn total_transformations(&self) -> usize {
        self.operations_folded + self.operations_simplified + self.operations_eliminated
    }
}

/// Analyze constant propagation opportunities in a graph.
pub fn analyze_constants(graph: &EinsumGraph) -> Result<ConstantPropagationResult, IrError> {
    let mut result = ConstantPropagationResult::none();

    // Start with input tensors marked as potentially constant
    let _constant_candidates: HashSet<usize> = graph.inputs.iter().copied().collect();

    // Identify tensors that are compile-time constants based on metadata
    for (tensor_idx, metadata) in &graph.tensor_metadata {
        if is_compile_time_constant(metadata) {
            result.constant_tensors.insert(*tensor_idx);
            result.constant_info.insert(
                *tensor_idx,
                ConstantInfo {
                    tensor_idx: *tensor_idx,
                    is_compile_time_constant: true,
                    is_identity: is_identity_value(metadata),
                    is_zero: is_zero_value(metadata),
                },
            );
        }
    }

    // Propagate constant information through the graph
    let mut changed = true;
    while changed {
        changed = false;

        for node in graph.nodes.iter() {
            // Check if all inputs are constants
            let all_inputs_constant = node
                .inputs
                .iter()
                .all(|&idx| result.constant_tensors.contains(&idx));

            if all_inputs_constant && !node.inputs.is_empty() {
                // This operation can be constant-folded
                for &output_idx in &node.outputs {
                    if !result.constant_tensors.contains(&output_idx) {
                        result.constant_tensors.insert(output_idx);
                        result.constant_info.insert(
                            output_idx,
                            ConstantInfo {
                                tensor_idx: output_idx,
                                is_compile_time_constant: true,
                                is_identity: false,
                                is_zero: false,
                            },
                        );
                        result.foldable_operations += 1;
                        changed = true;
                    }
                }
            }
        }
    }

    // Estimate speedup
    if result.foldable_operations > 0 {
        let total_ops = graph.nodes.len();
        let folding_ratio = result.foldable_operations as f64 / total_ops.max(1) as f64;
        result.estimated_speedup = 1.0 + folding_ratio * 0.3; // Conservative estimate
    }

    Ok(result)
}

/// Apply constant folding transformations to a graph.
pub fn apply_constant_folding(
    graph: &mut EinsumGraph,
    constants: &ConstantPropagationResult,
) -> Result<FoldingStats, IrError> {
    let mut stats = FoldingStats::none();
    let mut replacements: HashMap<usize, usize> = HashMap::new();

    // First pass: identify algebraic simplifications
    for node in graph.nodes.iter() {
        if let Some(simplified_output) = try_simplify_operation(node, constants) {
            // Record the simplification
            if !node.outputs.is_empty() {
                replacements.insert(node.outputs[0], simplified_output);
                stats.operations_simplified += 1;
            }
        } else if try_eliminate_operation(node, constants) {
            stats.operations_eliminated += 1;
        } else if constants.is_constant(node.outputs.first().copied().unwrap_or(usize::MAX)) {
            stats.operations_folded += 1;
        }
    }

    // Second pass: apply replacements
    for node in &mut graph.nodes {
        for input_idx in &mut node.inputs {
            if let Some(&replacement) = replacements.get(input_idx) {
                *input_idx = replacement;
            }
        }
    }

    // Update outputs
    for output_idx in &mut graph.outputs {
        if let Some(&replacement) = replacements.get(output_idx) {
            *output_idx = replacement;
        }
    }

    // Estimate speedup
    if stats.total_transformations() > 0 {
        let total_ops = graph.nodes.len().max(1);
        let optimization_ratio = stats.total_transformations() as f64 / total_ops as f64;
        stats.estimated_speedup = 1.0 + optimization_ratio * 0.4;
    }

    Ok(stats)
}

/// Perform aggressive constant folding with all available optimizations.
pub fn fold_constants_aggressive(graph: &mut EinsumGraph) -> Result<FoldingStats, IrError> {
    let mut total_stats = FoldingStats::none();

    // Multiple passes for maximum effect
    for _ in 0..3 {
        let constants = analyze_constants(graph)?;
        let stats = apply_constant_folding(graph, &constants)?;

        total_stats.operations_folded += stats.operations_folded;
        total_stats.operations_simplified += stats.operations_simplified;
        total_stats.operations_eliminated += stats.operations_eliminated;

        // Stop if no changes
        if stats.total_transformations() == 0 {
            break;
        }
    }

    // Update final speedup estimate
    if total_stats.total_transformations() > 0 {
        let total_ops = graph.nodes.len().max(1);
        let optimization_ratio = total_stats.total_transformations() as f64 / total_ops as f64;
        total_stats.estimated_speedup = 1.0 + optimization_ratio * 0.5;
    }

    Ok(total_stats)
}

/// Identify constant subgraphs that can be pre-computed.
pub fn identify_constant_subgraphs(graph: &EinsumGraph) -> Result<Vec<Vec<usize>>, IrError> {
    let constants = analyze_constants(graph)?;
    let mut subgraphs = Vec::new();
    let mut visited = HashSet::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        if visited.contains(&node_idx) {
            continue;
        }

        // Check if all inputs are constants
        let all_constant = node.inputs.iter().all(|&idx| constants.is_constant(idx));

        if all_constant && !node.inputs.is_empty() {
            // Find connected constant subgraph
            let mut subgraph = vec![node_idx];
            visited.insert(node_idx);

            // Expand to include dependent constant operations
            let mut changed = true;
            while changed {
                changed = false;
                for (idx, n) in graph.nodes.iter().enumerate() {
                    if visited.contains(&idx) {
                        continue;
                    }

                    let depends_on_subgraph = n.inputs.iter().any(|&input_idx| {
                        graph.nodes.iter().enumerate().any(|(sub_idx, sub_node)| {
                            subgraph.contains(&sub_idx) && sub_node.outputs.contains(&input_idx)
                        })
                    });

                    if depends_on_subgraph {
                        subgraph.push(idx);
                        visited.insert(idx);
                        changed = true;
                    }
                }
            }

            if !subgraph.is_empty() {
                subgraphs.push(subgraph);
            }
        }
    }

    Ok(subgraphs)
}

// Helper functions

fn is_compile_time_constant(metadata: &crate::Metadata) -> bool {
    metadata
        .get_attribute("constant")
        .map(|v| v == "true")
        .unwrap_or(false)
}

fn is_identity_value(metadata: &crate::Metadata) -> bool {
    metadata
        .get_attribute("identity")
        .map(|v| v == "true")
        .unwrap_or(false)
}

fn is_zero_value(metadata: &crate::Metadata) -> bool {
    metadata
        .get_attribute("zero")
        .map(|v| v == "true")
        .unwrap_or(false)
}

fn try_simplify_operation(
    node: &EinsumNode,
    constants: &ConstantPropagationResult,
) -> Option<usize> {
    if let OpType::ElemBinary { op } = &node.op {
        if node.inputs.len() == 2 {
            let left = node.inputs[0];
            let right = node.inputs[1];

            // Simplify x + 0 → x or 0 + x → x
            if op == "add" {
                if constants.get_info(right).is_some_and(|info| info.is_zero) {
                    return Some(left);
                }
                if constants.get_info(left).is_some_and(|info| info.is_zero) {
                    return Some(right);
                }
            }

            // Simplify x * 1 → x or 1 * x → x
            if op == "mul" {
                if constants
                    .get_info(right)
                    .is_some_and(|info| info.is_identity)
                {
                    return Some(left);
                }
                if constants
                    .get_info(left)
                    .is_some_and(|info| info.is_identity)
                {
                    return Some(right);
                }
            }
        }
    }

    None
}

fn try_eliminate_operation(node: &EinsumNode, constants: &ConstantPropagationResult) -> bool {
    if let OpType::ElemBinary { op } = &node.op {
        if node.inputs.len() == 2 {
            let left = node.inputs[0];
            let right = node.inputs[1];

            // Eliminate x * 0 or 0 * x (result is always 0)
            if op == "mul" {
                return constants.get_info(left).is_some_and(|info| info.is_zero)
                    || constants.get_info(right).is_some_and(|info| info.is_zero);
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Metadata;

    fn create_constant_metadata() -> Metadata {
        Metadata::new().with_attribute("constant", "true")
    }

    fn create_zero_metadata() -> Metadata {
        Metadata::new()
            .with_attribute("constant", "true")
            .with_attribute("zero", "true")
    }

    fn create_identity_metadata() -> Metadata {
        Metadata::new()
            .with_attribute("constant", "true")
            .with_attribute("identity", "true")
    }

    #[test]
    fn test_constant_info() {
        let info = ConstantInfo {
            tensor_idx: 0,
            is_compile_time_constant: true,
            is_identity: false,
            is_zero: false,
        };

        assert_eq!(info.tensor_idx, 0);
        assert!(info.is_compile_time_constant);
        assert!(!info.is_identity);
        assert!(!info.is_zero);
    }

    #[test]
    fn test_constant_propagation_result_none() {
        let result = ConstantPropagationResult::none();
        assert!(result.constant_tensors.is_empty());
        assert!(result.constant_info.is_empty());
        assert_eq!(result.foldable_operations, 0);
        assert_eq!(result.estimated_speedup, 1.0);
    }

    #[test]
    fn test_folding_stats_none() {
        let stats = FoldingStats::none();
        assert_eq!(stats.operations_folded, 0);
        assert_eq!(stats.operations_simplified, 0);
        assert_eq!(stats.operations_eliminated, 0);
        assert_eq!(stats.total_transformations(), 0);
    }

    #[test]
    fn test_analyze_constants_empty_graph() {
        let graph = EinsumGraph::new();
        let result = analyze_constants(&graph).unwrap();
        assert!(result.constant_tensors.is_empty());
    }

    #[test]
    fn test_analyze_constants_with_metadata() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor_with_metadata("A", create_constant_metadata());
        let b = graph.add_tensor("B");

        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();

        let result = analyze_constants(&graph).unwrap();
        assert!(result.is_constant(a));
        assert!(result.is_constant(b)); // Propagated from a
        assert_eq!(result.foldable_operations, 1);
    }

    #[test]
    fn test_simplify_add_zero() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("x");
        let zero = graph.add_tensor_with_metadata("zero", create_zero_metadata());
        let result = graph.add_tensor("result");

        let node = EinsumNode::elem_binary("add", x, zero, result);

        let mut const_result = ConstantPropagationResult::none();
        const_result.constant_tensors.insert(zero);
        const_result.constant_info.insert(
            zero,
            ConstantInfo {
                tensor_idx: zero,
                is_compile_time_constant: true,
                is_identity: false,
                is_zero: true,
            },
        );

        let simplified = try_simplify_operation(&node, &const_result);
        assert_eq!(simplified, Some(x));
    }

    #[test]
    fn test_simplify_mul_one() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("x");
        let one = graph.add_tensor_with_metadata("one", create_identity_metadata());
        let result = graph.add_tensor("result");

        let node = EinsumNode::elem_binary("mul", x, one, result);

        let mut const_result = ConstantPropagationResult::none();
        const_result.constant_tensors.insert(one);
        const_result.constant_info.insert(
            one,
            ConstantInfo {
                tensor_idx: one,
                is_compile_time_constant: true,
                is_identity: true,
                is_zero: false,
            },
        );

        let simplified = try_simplify_operation(&node, &const_result);
        assert_eq!(simplified, Some(x));
    }

    #[test]
    fn test_eliminate_mul_zero() {
        let mut graph = EinsumGraph::new();
        let x = graph.add_tensor("x");
        let zero = graph.add_tensor_with_metadata("zero", create_zero_metadata());
        let result = graph.add_tensor("result");

        let node = EinsumNode::elem_binary("mul", x, zero, result);

        let mut const_result = ConstantPropagationResult::none();
        const_result.constant_tensors.insert(zero);
        const_result.constant_info.insert(
            zero,
            ConstantInfo {
                tensor_idx: zero,
                is_compile_time_constant: true,
                is_identity: false,
                is_zero: true,
            },
        );

        let should_eliminate = try_eliminate_operation(&node, &const_result);
        assert!(should_eliminate);
    }

    #[test]
    fn test_apply_constant_folding() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor_with_metadata("A", create_constant_metadata());
        let b = graph.add_tensor_with_metadata("B", create_constant_metadata());
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::elem_binary("add", a, b, c))
            .unwrap();

        let constants = analyze_constants(&graph).unwrap();
        let stats = apply_constant_folding(&mut graph, &constants).unwrap();

        assert!(stats.operations_folded > 0 || stats.total_transformations() > 0);
    }

    #[test]
    fn test_fold_constants_aggressive() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor_with_metadata("A", create_constant_metadata());
        let b = graph.add_tensor_with_metadata("B", create_constant_metadata());
        let c = graph.add_tensor("C");
        let d = graph.add_tensor("D");

        graph
            .add_node(EinsumNode::elem_binary("add", a, b, c))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("relu", c, d))
            .unwrap();

        let stats = fold_constants_aggressive(&mut graph).unwrap();
        assert!(stats.operations_folded >= 1);
    }

    #[test]
    fn test_identify_constant_subgraphs() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor_with_metadata("A", create_constant_metadata());
        let b = graph.add_tensor_with_metadata("B", create_constant_metadata());
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::elem_binary("add", a, b, c))
            .unwrap();

        let subgraphs = identify_constant_subgraphs(&graph).unwrap();
        assert!(!subgraphs.is_empty());
    }

    #[test]
    fn test_is_constant_metadata_helpers() {
        let const_metadata = create_constant_metadata();
        assert!(is_compile_time_constant(&const_metadata));

        let zero_metadata = create_zero_metadata();
        assert!(is_compile_time_constant(&zero_metadata));
        assert!(is_zero_value(&zero_metadata));

        let identity_metadata = create_identity_metadata();
        assert!(is_compile_time_constant(&identity_metadata));
        assert!(is_identity_value(&identity_metadata));
    }

    #[test]
    fn test_constant_propagation_through_chain() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor_with_metadata("A", create_constant_metadata());
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");
        let d = graph.add_tensor("D");

        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("relu", b, c))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("relu", c, d))
            .unwrap();

        let result = analyze_constants(&graph).unwrap();

        assert!(result.is_constant(a));
        assert!(result.is_constant(b));
        assert!(result.is_constant(c));
        assert!(result.is_constant(d));
        assert_eq!(result.foldable_operations, 3);
    }

    #[test]
    fn test_mixed_constant_and_variable_graph() {
        let mut graph = EinsumGraph::new();
        let const_a = graph.add_tensor_with_metadata("const_A", create_constant_metadata());
        let var_x = graph.add_tensor("var_X");
        let result = graph.add_tensor("result");

        graph
            .add_node(EinsumNode::elem_binary("add", const_a, var_x, result))
            .unwrap();

        let analysis = analyze_constants(&graph).unwrap();

        assert!(analysis.is_constant(const_a));
        assert!(!analysis.is_constant(var_x));
        assert!(!analysis.is_constant(result)); // Result is not constant (depends on variable)
    }

    #[test]
    fn test_folding_stats_total_transformations() {
        let stats = FoldingStats {
            operations_folded: 2,
            operations_simplified: 3,
            operations_eliminated: 1,
            estimated_speedup: 1.5,
        };

        assert_eq!(stats.total_transformations(), 6);
    }

    #[test]
    fn test_speedup_estimation() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor_with_metadata("A", create_constant_metadata());
        let b = graph.add_tensor("B");

        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();

        let result = analyze_constants(&graph).unwrap();
        assert!(result.estimated_speedup > 1.0);
    }
}
