//! Cost model annotations for EinsumGraphs.
//!
//! This module provides infrastructure for annotating graphs with cost estimates,
//! which can be used for optimization, scheduling, and execution planning.
//!
//! # Cost Components
//!
//! - **Computational cost**: FLOPs required for the operation
//! - **Memory cost**: Bytes allocated for intermediate tensors
//! - **Communication cost**: Data transfer between devices/nodes
//! - **I/O cost**: Disk or network I/O operations

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{EinsumGraph, EinsumNode, OpType};

/// Cost annotation for a single operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationCost {
    /// Estimated computational cost (FLOPs)
    pub compute_flops: f64,
    /// Estimated memory footprint (bytes)
    pub memory_bytes: f64,
    /// Estimated communication cost (bytes transferred)
    pub communication_bytes: f64,
    /// Estimated I/O cost (bytes read/written)
    pub io_bytes: f64,
    /// Estimated latency (milliseconds)
    pub latency_ms: f64,
    /// Custom cost metrics
    #[serde(default)]
    pub custom: HashMap<String, f64>,
}

impl Default for OperationCost {
    fn default() -> Self {
        Self {
            compute_flops: 0.0,
            memory_bytes: 0.0,
            communication_bytes: 0.0,
            io_bytes: 0.0,
            latency_ms: 0.0,
            custom: HashMap::new(),
        }
    }
}

impl OperationCost {
    /// Create a new operation cost with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an operation cost with only computational cost.
    pub fn compute_only(flops: f64) -> Self {
        Self {
            compute_flops: flops,
            ..Default::default()
        }
    }

    /// Create an operation cost with computational and memory cost.
    pub fn compute_and_memory(flops: f64, memory_bytes: f64) -> Self {
        Self {
            compute_flops: flops,
            memory_bytes,
            ..Default::default()
        }
    }

    /// Add a custom cost metric.
    pub fn with_custom(mut self, key: impl Into<String>, value: f64) -> Self {
        self.custom.insert(key.into(), value);
        self
    }

    /// Combine two costs (for sequential operations).
    pub fn add(&self, other: &OperationCost) -> OperationCost {
        OperationCost {
            compute_flops: self.compute_flops + other.compute_flops,
            memory_bytes: self.memory_bytes.max(other.memory_bytes), // Peak memory
            communication_bytes: self.communication_bytes + other.communication_bytes,
            io_bytes: self.io_bytes + other.io_bytes,
            latency_ms: self.latency_ms + other.latency_ms,
            custom: {
                let mut merged = self.custom.clone();
                for (k, v) in &other.custom {
                    *merged.entry(k.clone()).or_insert(0.0) += v;
                }
                merged
            },
        }
    }

    /// Get the maximum cost (for parallel operations).
    pub fn max(&self, other: &OperationCost) -> OperationCost {
        OperationCost {
            compute_flops: self.compute_flops.max(other.compute_flops),
            memory_bytes: self.memory_bytes + other.memory_bytes, // Total memory
            communication_bytes: self.communication_bytes.max(other.communication_bytes),
            io_bytes: self.io_bytes.max(other.io_bytes),
            latency_ms: self.latency_ms.max(other.latency_ms),
            custom: {
                let mut merged = self.custom.clone();
                for (k, v) in &other.custom {
                    let entry = merged.entry(k.clone()).or_insert(0.0);
                    *entry = entry.max(*v);
                }
                merged
            },
        }
    }
}

/// Cost model for an entire graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphCostModel {
    /// Cost annotations per node (indexed by node index)
    pub node_costs: HashMap<usize, OperationCost>,
    /// Total estimated cost
    pub total_cost: OperationCost,
    /// Cost model metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl GraphCostModel {
    /// Create a new empty cost model.
    pub fn new() -> Self {
        Self {
            node_costs: HashMap::new(),
            total_cost: OperationCost::default(),
            metadata: HashMap::new(),
        }
    }

    /// Add a cost annotation for a node.
    pub fn set_node_cost(&mut self, node_idx: usize, cost: OperationCost) {
        self.node_costs.insert(node_idx, cost);
    }

    /// Get the cost annotation for a node.
    pub fn get_node_cost(&self, node_idx: usize) -> Option<&OperationCost> {
        self.node_costs.get(&node_idx)
    }

    /// Compute the total cost based on node costs and graph structure.
    pub fn compute_total_cost(&mut self, graph: &EinsumGraph) {
        self.total_cost = estimate_graph_cost(graph, self);
    }

    /// Add metadata to the cost model.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get a summary of the cost model.
    pub fn summary(&self) -> CostSummary {
        CostSummary {
            total_flops: self.total_cost.compute_flops,
            total_memory_bytes: self.total_cost.memory_bytes,
            total_communication_bytes: self.total_cost.communication_bytes,
            total_io_bytes: self.total_cost.io_bytes,
            total_latency_ms: self.total_cost.latency_ms,
            node_count: self.node_costs.len(),
        }
    }
}

impl Default for GraphCostModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of graph costs.
#[derive(Clone, Debug, PartialEq)]
pub struct CostSummary {
    /// Total computational cost (FLOPs)
    pub total_flops: f64,
    /// Total memory footprint (bytes)
    pub total_memory_bytes: f64,
    /// Total communication cost (bytes)
    pub total_communication_bytes: f64,
    /// Total I/O cost (bytes)
    pub total_io_bytes: f64,
    /// Total estimated latency (milliseconds)
    pub total_latency_ms: f64,
    /// Number of nodes with cost annotations
    pub node_count: usize,
}

/// Estimate the cost of a graph operation.
///
/// This is a simple heuristic-based estimator. For production use,
/// you should provide custom cost estimates based on profiling.
pub fn estimate_operation_cost(
    node: &EinsumNode,
    _tensor_sizes: &HashMap<usize, Vec<usize>>,
) -> OperationCost {
    match &node.op {
        OpType::Einsum { spec } => {
            // Estimate FLOPs for einsum based on the spec
            // This is a rough estimate - in practice, you'd parse the spec
            let inputs_len = node.inputs.len() as f64;
            let outputs_len = node.outputs.len() as f64;

            // Rough estimate: assume matrix multiply-like complexity
            let estimated_flops = 1000.0 * inputs_len * outputs_len;
            let estimated_memory = 100.0 * (inputs_len + outputs_len);

            OperationCost::compute_and_memory(estimated_flops, estimated_memory)
                .with_custom("spec_complexity", spec.len() as f64)
        }
        OpType::ElemUnary { .. } => {
            // Element-wise unary operations are typically cheap
            OperationCost::compute_and_memory(100.0, 50.0)
        }
        OpType::ElemBinary { .. } => {
            // Element-wise binary operations
            OperationCost::compute_and_memory(200.0, 100.0)
        }
        OpType::Reduce { .. } => {
            // Reductions require O(n) operations
            OperationCost::compute_and_memory(500.0, 75.0)
        }
    }
}

/// Estimate the total cost of a graph given per-node costs.
pub fn estimate_graph_cost(graph: &EinsumGraph, cost_model: &GraphCostModel) -> OperationCost {
    let mut total = OperationCost::default();

    // Simple sequential cost model (assumes nodes execute sequentially)
    // For a more sophisticated model, use the critical path or parallel schedule
    for (idx, _node) in graph.nodes.iter().enumerate() {
        if let Some(node_cost) = cost_model.get_node_cost(idx) {
            total = total.add(node_cost);
        }
    }

    total
}

/// Auto-annotate a graph with estimated costs.
///
/// This uses heuristic estimates for each operation type.
/// For production use, provide custom cost estimates based on profiling.
pub fn auto_annotate_costs(graph: &EinsumGraph) -> GraphCostModel {
    let mut cost_model = GraphCostModel::new();
    let tensor_sizes = HashMap::new(); // Would be populated from actual tensor metadata

    for (idx, node) in graph.nodes.iter().enumerate() {
        let cost = estimate_operation_cost(node, &tensor_sizes);
        cost_model.set_node_cost(idx, cost);
    }

    cost_model.compute_total_cost(graph);
    cost_model
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EinsumNode;

    #[test]
    fn test_operation_cost_creation() {
        let cost = OperationCost::compute_only(1000.0);
        assert_eq!(cost.compute_flops, 1000.0);
        assert_eq!(cost.memory_bytes, 0.0);
    }

    #[test]
    fn test_operation_cost_add() {
        let cost1 = OperationCost::compute_and_memory(1000.0, 500.0);
        let cost2 = OperationCost::compute_and_memory(2000.0, 300.0);

        let total = cost1.add(&cost2);
        assert_eq!(total.compute_flops, 3000.0);
        assert_eq!(total.memory_bytes, 500.0); // Max of the two
    }

    #[test]
    fn test_operation_cost_max() {
        let cost1 = OperationCost::compute_and_memory(1000.0, 500.0);
        let cost2 = OperationCost::compute_and_memory(2000.0, 300.0);

        let max_cost = cost1.max(&cost2);
        assert_eq!(max_cost.compute_flops, 2000.0);
        assert_eq!(max_cost.memory_bytes, 800.0); // Sum for parallel
    }

    #[test]
    fn test_cost_model_creation() {
        let mut model = GraphCostModel::new();
        let cost = OperationCost::compute_only(1000.0);

        model.set_node_cost(0, cost.clone());
        assert_eq!(model.get_node_cost(0), Some(&cost));
    }

    #[test]
    fn test_estimate_einsum_cost() {
        let node = EinsumNode::einsum("ik,kj->ij", vec![0, 1], vec![2]);
        let tensor_sizes = HashMap::new();

        let cost = estimate_operation_cost(&node, &tensor_sizes);
        assert!(cost.compute_flops > 0.0);
        assert!(cost.memory_bytes > 0.0);
    }

    #[test]
    fn test_auto_annotate_costs() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph.add_input(a).unwrap();
        graph.add_input(b).unwrap();
        graph
            .add_node(EinsumNode::einsum("i,j->ij", vec![a, b], vec![c]))
            .unwrap();
        graph.add_output(c).unwrap();

        let cost_model = auto_annotate_costs(&graph);
        assert_eq!(cost_model.node_costs.len(), 1);
        assert!(cost_model.total_cost.compute_flops > 0.0);
    }

    #[test]
    fn test_cost_summary() {
        let mut model = GraphCostModel::new();
        model.set_node_cost(0, OperationCost::compute_and_memory(1000.0, 500.0));
        model.set_node_cost(1, OperationCost::compute_and_memory(2000.0, 300.0));

        let summary = model.summary();
        assert_eq!(summary.node_count, 2);
    }

    #[test]
    fn test_custom_cost_metrics() {
        let cost = OperationCost::new()
            .with_custom("custom_metric", 42.0)
            .with_custom("another_metric", 100.0);

        assert_eq!(cost.custom.get("custom_metric"), Some(&42.0));
        assert_eq!(cost.custom.get("another_metric"), Some(&100.0));
    }

    #[test]
    fn test_cost_model_metadata() {
        let model = GraphCostModel::new()
            .with_metadata("device", "GPU")
            .with_metadata("precision", "fp32");

        assert_eq!(model.metadata.get("device"), Some(&"GPU".to_string()));
        assert_eq!(model.metadata.get("precision"), Some(&"fp32".to_string()));
    }
}
