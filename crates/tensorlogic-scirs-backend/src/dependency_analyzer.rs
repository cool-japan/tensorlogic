//! Dependency analysis for parallel execution of EinsumGraph operations.
//!
//! This module provides tools to analyze dependencies between operations in an
//! EinsumGraph, enabling parallel execution of independent operations.
//!
//! ## Key Concepts
//!
//! - **Dependencies**: An operation depends on another if it uses the output tensor
//! - **Independent operations**: Operations with no data dependencies can run in parallel
//! - **Execution levels**: Groups of operations that can execute concurrently
//!
//! ## Example
//!
//! ```text
//! Graph:
//!   Op0: A = input[a]
//!   Op1: B = input[b]
//!   Op2: C = Op0 + Op1  (depends on Op0, Op1)
//!   Op3: D = Op0 * 2    (depends on Op0)
//!
//! Dependency Analysis:
//!   Level 0: [Op0, Op1]  (independent, can run in parallel)
//!   Level 1: [Op2, Op3]  (both depend on Level 0, can run in parallel)
//! ```

use std::collections::HashMap;
use tensorlogic_ir::EinsumGraph;

/// Dependency information for a single operation.
#[derive(Debug, Clone)]
pub struct OperationDependency {
    /// Index of this operation in the graph
    pub node_index: usize,
    /// Indices of operations this one depends on (reads their outputs)
    pub depends_on: Vec<usize>,
    /// Indices of operations that depend on this one (read its output)
    pub dependents: Vec<usize>,
    /// Execution level (0 = can execute immediately, higher = needs previous levels)
    pub execution_level: usize,
}

/// Result of dependency analysis on an EinsumGraph.
#[derive(Debug, Clone)]
pub struct DependencyAnalysis {
    /// Per-operation dependency information
    pub operations: Vec<OperationDependency>,
    /// Operations grouped by execution level (level 0 first, then 1, etc.)
    pub execution_levels: Vec<Vec<usize>>,
    /// Total number of execution levels
    pub num_levels: usize,
    /// Maximum parallelism (max ops in any single level)
    pub max_parallelism: usize,
}

impl DependencyAnalysis {
    /// Analyzes dependencies in an EinsumGraph.
    ///
    /// This performs topological analysis to determine:
    /// 1. Which operations depend on which other operations
    /// 2. Which operations can be executed in parallel
    /// 3. The minimum number of sequential execution stages required
    ///
    /// # Arguments
    /// * `graph` - The EinsumGraph to analyze
    ///
    /// # Returns
    /// Dependency analysis result with execution levels
    pub fn analyze(graph: &EinsumGraph) -> Self {
        let num_ops = graph.nodes.len();
        let mut operations = Vec::with_capacity(num_ops);

        // Step 1: Build dependency map (which operations depend on which)
        // Map tensor index -> node index that produces it
        let mut tensor_producers: HashMap<usize, usize> = HashMap::new();

        // Mark input tensors as "produced" before any operations (no producer node)
        // We use usize::MAX as a special marker for graph inputs
        for &input_idx in &graph.inputs {
            tensor_producers.insert(input_idx, usize::MAX);
        }

        // Build initial dependency structures
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            let mut depends_on = Vec::new();

            // Check which operations produce the input tensors for this node
            for &input_tensor_idx in &node.inputs {
                if let Some(&producer_idx) = tensor_producers.get(&input_tensor_idx) {
                    if producer_idx != usize::MAX {
                        depends_on.push(producer_idx);
                    }
                    // If producer_idx == usize::MAX, it's a graph input, no dependency
                }
            }

            // Remove duplicates and sort for consistency
            depends_on.sort_unstable();
            depends_on.dedup();

            operations.push(OperationDependency {
                node_index: node_idx,
                depends_on,
                dependents: Vec::new(),
                execution_level: 0, // Will be computed later
            });

            // Register this operation as the producer of its output tensors
            for &output_tensor_idx in &node.outputs {
                tensor_producers.insert(output_tensor_idx, node_idx);
            }
        }

        // Step 2: Build reverse dependencies (dependents)
        for idx in 0..num_ops {
            let deps = operations[idx].depends_on.clone();
            for &dep_idx in &deps {
                operations[dep_idx].dependents.push(idx);
            }
        }

        // Step 3: Compute execution levels using topological sort
        let mut execution_levels: Vec<Vec<usize>> = Vec::new();
        let mut level_assigned = vec![false; num_ops];
        let mut current_level = 0;

        loop {
            // Find all operations that can execute at this level
            let mut level_ops = Vec::new();

            for idx in 0..num_ops {
                if level_assigned[idx] {
                    continue;
                }

                // Check if all dependencies are satisfied (assigned to earlier levels)
                let all_deps_satisfied = operations[idx]
                    .depends_on
                    .iter()
                    .all(|&dep_idx| level_assigned[dep_idx]);

                if all_deps_satisfied {
                    level_ops.push(idx);
                }
            }

            if level_ops.is_empty() {
                break; // No more operations to assign
            }

            // Assign execution level to these operations
            for &idx in &level_ops {
                operations[idx].execution_level = current_level;
                level_assigned[idx] = true;
            }

            execution_levels.push(level_ops);
            current_level += 1;
        }

        // Check for unassigned operations (indicates a cycle, which shouldn't happen)
        let unassigned: Vec<usize> = (0..num_ops).filter(|&idx| !level_assigned[idx]).collect();

        if !unassigned.is_empty() {
            panic!(
                "Cyclic dependency detected in graph! Unassigned operations: {:?}",
                unassigned
            );
        }

        let max_parallelism = execution_levels
            .iter()
            .map(|level| level.len())
            .max()
            .unwrap_or(0);

        Self {
            operations,
            execution_levels,
            num_levels: current_level,
            max_parallelism,
        }
    }

    /// Returns true if the graph has opportunities for parallel execution.
    pub fn has_parallelism(&self) -> bool {
        self.max_parallelism > 1
    }

    /// Returns the expected speedup from parallelization (simplified estimate).
    ///
    /// This is a rough estimate assuming:
    /// - All operations have similar cost
    /// - Perfect parallelization within levels (wall-clock time = 1 per level)
    /// - No overhead from thread management
    ///
    /// Real speedup will be lower due to overhead and work imbalance.
    pub fn estimated_speedup(&self) -> f64 {
        if self.operations.is_empty() {
            return 1.0;
        }

        let sequential_cost = self.operations.len() as f64;
        // Parallel cost is just the number of levels (wall-clock time)
        // since operations within a level can execute concurrently
        let parallel_cost = self.num_levels as f64;

        sequential_cost / parallel_cost
    }

    /// Gets the operations at a specific execution level.
    pub fn get_level(&self, level: usize) -> Option<&[usize]> {
        self.execution_levels.get(level).map(|v| v.as_slice())
    }

    /// Returns statistics about the dependency structure.
    pub fn stats(&self) -> DependencyStats {
        let total_ops = self.operations.len();
        let independent_ops = self
            .operations
            .iter()
            .filter(|op| op.depends_on.is_empty())
            .count();

        let avg_dependencies = if total_ops > 0 {
            self.operations
                .iter()
                .map(|op| op.depends_on.len())
                .sum::<usize>() as f64
                / total_ops as f64
        } else {
            0.0
        };

        let avg_dependents = if total_ops > 0 {
            self.operations
                .iter()
                .map(|op| op.dependents.len())
                .sum::<usize>() as f64
                / total_ops as f64
        } else {
            0.0
        };

        DependencyStats {
            total_operations: total_ops,
            independent_operations: independent_ops,
            num_levels: self.num_levels,
            max_parallelism: self.max_parallelism,
            avg_dependencies,
            avg_dependents,
            estimated_speedup: self.estimated_speedup(),
        }
    }
}

/// Statistics about graph dependencies.
#[derive(Debug, Clone)]
pub struct DependencyStats {
    /// Total number of operations in the graph
    pub total_operations: usize,
    /// Number of operations with no dependencies
    pub independent_operations: usize,
    /// Number of execution levels required
    pub num_levels: usize,
    /// Maximum number of operations that can run in parallel
    pub max_parallelism: usize,
    /// Average number of dependencies per operation
    pub avg_dependencies: f64,
    /// Average number of operations depending on each operation
    pub avg_dependents: f64,
    /// Estimated speedup from parallelization
    pub estimated_speedup: f64,
}

impl std::fmt::Display for DependencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DependencyStats {{ total_ops: {}, independent: {}, levels: {}, max_parallel: {}, \
             avg_deps: {:.2}, avg_dependents: {:.2}, estimated_speedup: {:.2}x }}",
            self.total_operations,
            self.independent_operations,
            self.num_levels,
            self.max_parallelism,
            self.avg_dependencies,
            self.avg_dependents,
            self.estimated_speedup
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{EinsumNode, OpType};

    fn create_test_graph() -> EinsumGraph {
        // Create a simple graph:
        // Tensors: 0=a, 1=b, 2=c, 3=d, 4=e
        // Op0: c = a + b (inputs: 0, 1 -> output: 2)
        // Op1: d = a * const (inputs: 0 -> output: 3)
        // Op2: e = c + d (inputs: 2, 3 -> output: 4, depends on Op0, Op1)

        let mut graph = EinsumGraph::new();

        // Add tensors
        let a_idx = graph.add_tensor("a"); // 0
        let b_idx = graph.add_tensor("b"); // 1
        let c_idx = graph.add_tensor("c"); // 2
        let d_idx = graph.add_tensor("d"); // 3
        let e_idx = graph.add_tensor("e"); // 4

        // Mark inputs
        graph.add_input(a_idx).unwrap();
        graph.add_input(b_idx).unwrap();

        // Op0: c = a + b
        graph
            .add_node(EinsumNode {
                op: OpType::ElemBinary {
                    op: "add".to_string(),
                },
                inputs: vec![a_idx, b_idx],
                outputs: vec![c_idx],
                metadata: None,
            })
            .unwrap();

        // Op1: d = a * const (simplified)
        graph
            .add_node(EinsumNode {
                op: OpType::ElemBinary {
                    op: "multiply".to_string(),
                },
                inputs: vec![a_idx], // Simplified: just a as input
                outputs: vec![d_idx],
                metadata: None,
            })
            .unwrap();

        // Op2: e = c + d
        graph
            .add_node(EinsumNode {
                op: OpType::ElemBinary {
                    op: "add".to_string(),
                },
                inputs: vec![c_idx, d_idx],
                outputs: vec![e_idx],
                metadata: None,
            })
            .unwrap();

        // Mark output
        graph.add_output(e_idx).unwrap();

        graph
    }

    #[test]
    fn test_dependency_analysis_basic() {
        let graph = create_test_graph();
        let analysis = DependencyAnalysis::analyze(&graph);

        assert_eq!(analysis.operations.len(), 3);
        assert_eq!(analysis.num_levels, 2);

        // Op0 and Op1 should be at level 0 (independent)
        assert_eq!(analysis.operations[0].execution_level, 0);
        assert_eq!(analysis.operations[1].execution_level, 0);

        // Op2 should be at level 1 (depends on Op0 and Op1)
        assert_eq!(analysis.operations[2].execution_level, 1);

        // Check dependencies
        assert_eq!(analysis.operations[0].depends_on, Vec::<usize>::new());
        assert_eq!(analysis.operations[1].depends_on, Vec::<usize>::new());
        assert_eq!(analysis.operations[2].depends_on, vec![0_usize, 1_usize]);
    }

    #[test]
    fn test_execution_levels() {
        let graph = create_test_graph();
        let analysis = DependencyAnalysis::analyze(&graph);

        assert_eq!(analysis.execution_levels.len(), 2);
        assert_eq!(analysis.execution_levels[0].len(), 2); // Op0 and Op1
        assert_eq!(analysis.execution_levels[1].len(), 1); // Op2

        assert!(analysis.execution_levels[0].contains(&0));
        assert!(analysis.execution_levels[0].contains(&1));
        assert!(analysis.execution_levels[1].contains(&2));
    }

    #[test]
    fn test_has_parallelism() {
        let graph = create_test_graph();
        let analysis = DependencyAnalysis::analyze(&graph);

        assert!(analysis.has_parallelism());
        assert_eq!(analysis.max_parallelism, 2);
    }

    #[test]
    fn test_estimated_speedup() {
        let graph = create_test_graph();
        let analysis = DependencyAnalysis::analyze(&graph);

        // 3 operations, 2 levels -> speedup = 3/2 = 1.5
        let speedup = analysis.estimated_speedup();
        assert!((speedup - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_dependency_stats() {
        let graph = create_test_graph();
        let analysis = DependencyAnalysis::analyze(&graph);
        let stats = analysis.stats();

        assert_eq!(stats.total_operations, 3);
        assert_eq!(stats.independent_operations, 2);
        assert_eq!(stats.num_levels, 2);
        assert_eq!(stats.max_parallelism, 2);
        assert!((stats.avg_dependencies - 0.666).abs() < 0.01); // (0+0+2)/3
    }

    #[test]
    fn test_sequential_graph() {
        // Create a sequential graph: Op0 -> Op1 -> Op2
        // Tensors: 0=a, 1=b, 2=c, 3=d
        let mut graph = EinsumGraph::new();

        let a_idx = graph.add_tensor("a"); // 0
        let b_idx = graph.add_tensor("b"); // 1
        let c_idx = graph.add_tensor("c"); // 2
        let d_idx = graph.add_tensor("d"); // 3

        graph.add_input(a_idx).unwrap();

        // Op0: b = relu(a)
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "relu".to_string(),
                },
                inputs: vec![a_idx],
                outputs: vec![b_idx],
                metadata: None,
            })
            .unwrap();

        // Op1: c = sigmoid(b)
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "sigmoid".to_string(),
                },
                inputs: vec![b_idx],
                outputs: vec![c_idx],
                metadata: None,
            })
            .unwrap();

        // Op2: d = 1 - c
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "oneminus".to_string(),
                },
                inputs: vec![c_idx],
                outputs: vec![d_idx],
                metadata: None,
            })
            .unwrap();

        graph.add_output(d_idx).unwrap();

        let analysis = DependencyAnalysis::analyze(&graph);

        assert_eq!(analysis.num_levels, 3);
        assert_eq!(analysis.max_parallelism, 1);
        assert!(!analysis.has_parallelism());
        assert!((analysis.estimated_speedup() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_get_level() {
        let graph = create_test_graph();
        let analysis = DependencyAnalysis::analyze(&graph);

        let level0 = analysis.get_level(0).unwrap();
        assert_eq!(level0.len(), 2);

        let level1 = analysis.get_level(1).unwrap();
        assert_eq!(level1.len(), 1);

        assert!(analysis.get_level(2).is_none());
    }

    #[test]
    fn test_dependents() {
        let graph = create_test_graph();
        let analysis = DependencyAnalysis::analyze(&graph);

        // Op0 is depended on by Op2
        assert_eq!(analysis.operations[0].dependents, vec![2_usize]);

        // Op1 is depended on by Op2
        assert_eq!(analysis.operations[1].dependents, vec![2_usize]);

        // Op2 has no dependents
        assert_eq!(analysis.operations[2].dependents, Vec::<usize>::new());
    }
}
