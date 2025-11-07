//! Graph transformation and manipulation utilities.

use std::collections::{HashMap, HashSet};

use super::{EinsumGraph, EinsumNode};
use crate::error::IrError;

/// Visitor trait for traversing graph nodes.
pub trait GraphVisitor {
    /// Visit a node in the graph.
    fn visit_node(&mut self, node_idx: usize, node: &EinsumNode, graph: &EinsumGraph);

    /// Called before visiting all nodes.
    fn start(&mut self, _graph: &EinsumGraph) {}

    /// Called after visiting all nodes.
    fn finish(&mut self, _graph: &EinsumGraph) {}
}

/// Mutable visitor trait for transforming graph nodes.
pub trait GraphMutVisitor {
    /// Visit and potentially modify a node.
    fn visit_node_mut(
        &mut self,
        node_idx: usize,
        node: &mut EinsumNode,
        graph: &EinsumGraph,
    ) -> Result<(), IrError>;
}

impl EinsumGraph {
    /// Extract a subgraph containing only the specified nodes and their dependencies.
    pub fn extract_subgraph(&self, node_indices: &[usize]) -> Result<EinsumGraph, IrError> {
        // Validate node indices
        for &idx in node_indices {
            if idx >= self.nodes.len() {
                return Err(IrError::NodeValidation {
                    node: idx,
                    message: format!("Node index {} out of bounds", idx),
                });
            }
        }

        // Collect all nodes reachable from the specified nodes (via dependencies)
        let mut reachable_nodes = HashSet::new();
        for &idx in node_indices {
            self.collect_dependencies(idx, &mut reachable_nodes);
        }

        // Build index mapping for tensors
        let mut tensor_map = HashMap::new();
        let mut new_graph = EinsumGraph::new();

        // Collect all tensors used by reachable nodes (both inputs and outputs)
        let mut used_tensors = HashSet::new();
        for &node_idx in &reachable_nodes {
            let node = &self.nodes[node_idx];
            for &input_idx in &node.inputs {
                used_tensors.insert(input_idx);
            }
            for &output_idx in &node.outputs {
                used_tensors.insert(output_idx);
            }
        }

        // Add tensors to new graph
        for &tensor_idx in &used_tensors {
            let new_idx = new_graph.add_tensor(&self.tensors[tensor_idx]);
            tensor_map.insert(tensor_idx, new_idx);
        }

        // Add nodes with remapped tensor indices
        for &node_idx in &reachable_nodes {
            let old_node = &self.nodes[node_idx];
            let new_node = old_node.remap_tensors(&tensor_map)?;
            new_graph.add_node(new_node)?;
        }

        // Set outputs (if any of the original outputs are in the subgraph)
        for &out_idx in &self.outputs {
            if let Some(&new_idx) = tensor_map.get(&out_idx) {
                new_graph.add_output(new_idx)?;
            }
        }

        Ok(new_graph)
    }

    /// Collect all nodes that this node depends on (recursively).
    fn collect_dependencies(&self, node_idx: usize, visited: &mut HashSet<usize>) {
        if visited.contains(&node_idx) {
            return;
        }
        visited.insert(node_idx);

        let node = &self.nodes[node_idx];

        // Find nodes that produce the input tensors for this node
        for &input_tensor in &node.inputs {
            // Find which node produces this input tensor
            for (idx, other_node) in self.nodes.iter().enumerate() {
                if idx < node_idx && other_node.produces(input_tensor) {
                    self.collect_dependencies(idx, visited);
                }
            }
        }
    }

    /// Merge another graph into this one.
    ///
    /// Returns a mapping from old tensor indices to new tensor indices.
    pub fn merge(&mut self, other: &EinsumGraph) -> Result<HashMap<usize, usize>, IrError> {
        let mut tensor_map = HashMap::new();

        // Try to reuse existing tensors with the same name
        for (old_idx, tensor_name) in other.tensors.iter().enumerate() {
            if let Some(existing_idx) = self.tensors.iter().position(|t| t == tensor_name) {
                tensor_map.insert(old_idx, existing_idx);
            } else {
                let new_idx = self.add_tensor(tensor_name);
                tensor_map.insert(old_idx, new_idx);
            }
        }

        // Add nodes with remapped tensor indices
        for node in &other.nodes {
            let new_node = node.remap_tensors(&tensor_map)?;
            self.add_node(new_node)?;
        }

        // Add outputs
        for &out_idx in &other.outputs {
            if let Some(&new_idx) = tensor_map.get(&out_idx) {
                if !self.outputs.contains(&new_idx) {
                    self.add_output(new_idx)?;
                }
            }
        }

        Ok(tensor_map)
    }

    /// Visit all nodes in the graph using a visitor.
    pub fn visit<V: GraphVisitor>(&self, visitor: &mut V) {
        visitor.start(self);
        for (idx, node) in self.nodes.iter().enumerate() {
            visitor.visit_node(idx, node, self);
        }
        visitor.finish(self);
    }

    /// Visit all nodes mutably using a mutable visitor.
    pub fn visit_mut<V: GraphMutVisitor>(&mut self, visitor: &mut V) -> Result<(), IrError> {
        // We need to clone the graph for the visitor to see the original structure
        let graph_clone = self.clone();

        for idx in 0..self.nodes.len() {
            visitor.visit_node_mut(idx, &mut self.nodes[idx], &graph_clone)?;
        }

        Ok(())
    }

    /// Apply a rewrite rule to all nodes in the graph.
    ///
    /// The rule function takes a node and returns an optional replacement node.
    pub fn apply_rewrite<F>(&mut self, mut rule: F) -> Result<usize, IrError>
    where
        F: FnMut(&EinsumNode) -> Option<EinsumNode>,
    {
        let mut rewrites = 0;

        for node in &mut self.nodes {
            if let Some(new_node) = rule(node) {
                *node = new_node;
                rewrites += 1;
            }
        }

        Ok(rewrites)
    }

    /// Get all nodes that depend on a specific tensor (consume it as input).
    pub fn tensor_consumers(&self, tensor_idx: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.inputs.contains(&tensor_idx))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get the node that produces a specific tensor.
    ///
    /// Note: In the current graph model, tensors can be produced by at most one node
    /// or be external inputs. This returns nodes that might output to this tensor
    /// based on graph topology.
    pub fn tensor_producer(&self, tensor_idx: usize) -> Option<usize> {
        // A simple heuristic: find nodes that come before uses of this tensor
        let consumers = self.tensor_consumers(tensor_idx);
        if consumers.is_empty() {
            return None;
        }

        let min_consumer = consumers.iter().min().copied()?;

        // Find the latest node before min_consumer
        if min_consumer > 0 {
            Some(min_consumer - 1)
        } else {
            None
        }
    }

    /// Check if there's a path from node_from to node_to based on node ordering.
    pub fn has_path(&self, node_from: usize, node_to: usize) -> bool {
        // Simple topological ordering: lower indices come before higher indices
        node_from <= node_to
    }

    /// Get dependency chain for a node (all nodes it depends on).
    pub fn dependencies(&self, node_idx: usize) -> HashSet<usize> {
        let mut deps = HashSet::new();
        self.collect_dependencies(node_idx, &mut deps);
        deps.remove(&node_idx); // Remove self
        deps
    }

    /// Get number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

impl EinsumNode {
    /// Remap tensor indices using the provided mapping.
    pub(crate) fn remap_tensors(
        &self,
        tensor_map: &HashMap<usize, usize>,
    ) -> Result<Self, IrError> {
        let inputs: Vec<usize> = self
            .inputs
            .iter()
            .map(|&idx| {
                tensor_map
                    .get(&idx)
                    .copied()
                    .ok_or_else(|| IrError::NodeValidation {
                        node: 0,
                        message: format!("Input tensor {} not in mapping", idx),
                    })
            })
            .collect::<Result<_, _>>()?;

        let outputs: Vec<usize> = self
            .outputs
            .iter()
            .map(|&idx| {
                tensor_map
                    .get(&idx)
                    .copied()
                    .ok_or_else(|| IrError::NodeValidation {
                        node: 0,
                        message: format!("Output tensor {} not in mapping", idx),
                    })
            })
            .collect::<Result<_, _>>()?;

        Ok(EinsumNode {
            inputs,
            outputs,
            op: self.op.clone(),
            metadata: self.metadata.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpType;

    fn create_test_graph() -> EinsumGraph {
        let mut g = EinsumGraph::new();

        // Tensors: t0, t1, t2, t3, t4, t5, t6 (inputs + intermediate + outputs)
        let t0 = g.add_tensor("t0");
        let t1 = g.add_tensor("t1");
        let t2 = g.add_tensor("t2");
        let _t3 = g.add_tensor("t3");
        let t4 = g.add_tensor("t4"); // output of node 0
        let t5 = g.add_tensor("t5"); // output of node 1
        let t6 = g.add_tensor("t6"); // output of node 2

        // Node 0: uses t0, produces t4
        g.add_node(EinsumNode {
            inputs: vec![t0],
            outputs: vec![t4],
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            metadata: None,
        })
        .unwrap();

        // Node 1: uses t1, produces t5
        g.add_node(EinsumNode {
            inputs: vec![t1],
            outputs: vec![t5],
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            metadata: None,
        })
        .unwrap();

        // Node 2: uses t2, produces t6
        g.add_node(EinsumNode {
            inputs: vec![t2],
            outputs: vec![t6],
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            metadata: None,
        })
        .unwrap();

        g.add_output(t6).unwrap();

        g
    }

    #[test]
    fn test_extract_subgraph() {
        let graph = create_test_graph();

        // Extract nodes 0 and 1
        let subgraph = graph.extract_subgraph(&[0, 1]).unwrap();

        assert_eq!(subgraph.nodes.len(), 2);
        assert!(subgraph.tensors.len() >= 2);
    }

    #[test]
    fn test_merge_graphs() {
        let mut g1 = EinsumGraph::new();
        let t0 = g1.add_tensor("shared");
        let t1 = g1.add_tensor("out1");
        g1.add_node(EinsumNode {
            inputs: vec![t0],
            outputs: vec![t1],
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            metadata: None,
        })
        .unwrap();

        let mut g2 = EinsumGraph::new();
        let t0_2 = g2.add_tensor("shared");
        let t1_2 = g2.add_tensor("out2");
        g2.add_node(EinsumNode {
            inputs: vec![t0_2],
            outputs: vec![t1_2],
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            metadata: None,
        })
        .unwrap();

        let tensor_map = g1.merge(&g2).unwrap();

        // Should reuse "shared" tensor
        assert_eq!(tensor_map[&0], 0); // "shared" mapped to same index
        assert_eq!(g1.nodes.len(), 2);
    }

    #[test]
    fn test_tensor_consumers() {
        let graph = create_test_graph();

        let consumers = graph.tensor_consumers(1); // t1
        assert_eq!(consumers.len(), 1);
        assert_eq!(consumers[0], 1); // Node 1 consumes t1
    }

    #[test]
    fn test_has_path() {
        let graph = create_test_graph();

        assert!(graph.has_path(0, 2)); // 0 -> 2 (via ordering)
        assert!(graph.has_path(0, 0)); // Same node
        assert!(!graph.has_path(2, 0)); // No backward path
    }

    #[test]
    fn test_visitor_pattern() {
        let graph = create_test_graph();

        struct CountingVisitor {
            count: usize,
        }

        impl GraphVisitor for CountingVisitor {
            fn visit_node(&mut self, _idx: usize, _node: &EinsumNode, _graph: &EinsumGraph) {
                self.count += 1;
            }
        }

        let mut visitor = CountingVisitor { count: 0 };
        graph.visit(&mut visitor);

        assert_eq!(visitor.count, 3);
    }

    #[test]
    fn test_apply_rewrite() {
        let mut graph = create_test_graph();

        // Replace all einsum operations with a different spec
        let rewrites = graph
            .apply_rewrite(|node| {
                if matches!(node.op, OpType::Einsum { .. }) {
                    Some(EinsumNode {
                        inputs: node.inputs.clone(),
                        outputs: node.outputs.clone(),
                        op: OpType::Einsum {
                            spec: "new->spec".to_string(),
                        },
                        metadata: None,
                    })
                } else {
                    None
                }
            })
            .unwrap();

        assert_eq!(rewrites, 3);

        for node in &graph.nodes {
            if let OpType::Einsum { spec } = &node.op {
                assert_eq!(spec, "new->spec");
            }
        }
    }

    #[test]
    fn test_node_count() {
        let graph = create_test_graph();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.tensor_count(), 7); // t0-t6 (3 inputs + 1 unused + 3 outputs)
    }

    #[test]
    fn test_dependencies() {
        // Create a graph with actual dependencies
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("t0");
        let t1 = graph.add_tensor("t1"); // output of node 0
        let t2 = graph.add_tensor("t2"); // output of node 1

        // Node 0: produces t1 from t0
        graph
            .add_node(EinsumNode {
                inputs: vec![t0],
                outputs: vec![t1],
                op: OpType::Einsum {
                    spec: "i->i".to_string(),
                },
                metadata: None,
            })
            .unwrap();

        // Node 1: produces t2 from t1 (depends on node 0)
        graph
            .add_node(EinsumNode {
                inputs: vec![t1],
                outputs: vec![t2],
                op: OpType::Einsum {
                    spec: "i->i".to_string(),
                },
                metadata: None,
            })
            .unwrap();

        let deps = graph.dependencies(1);
        // Node 1 depends on node 0 (which produces t1)
        assert!(deps.contains(&0));
        assert_eq!(deps.len(), 1);
    }
}
