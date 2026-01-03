//! # Example 15: Advanced Graph Algorithms
//!
//! This example demonstrates sophisticated graph analysis algorithms for tensor computation graphs.
//! These algorithms are essential for optimization, scheduling, and verification of tensor programs.
//!
//! ## What You'll Learn
//!
//! - Detecting cycles in computation graphs
//! - Finding strongly connected components (Tarjan's algorithm)
//! - Topological sorting for execution ordering
//! - Checking for directed acyclic graphs (DAGs)
//! - Graph isomorphism testing
//! - Critical path analysis for scheduling
//! - Computing graph diameter
//! - Finding all paths between nodes
//!
//! ## Key Concepts
//!
//! - **Cycle**: A path from a node back to itself
//! - **SCC**: Maximal strongly connected subgraph
//! - **Topological Sort**: Linear ordering respecting dependencies
//! - **Critical Path**: Longest path (determines minimum execution time)
//! - **Graph Diameter**: Maximum shortest path length

use std::collections::HashMap;
use tensorlogic_ir::{
    are_isomorphic, critical_path_analysis, find_all_paths, find_cycles, graph_diameter, is_dag,
    strongly_connected_components, topological_sort, EinsumGraph, EinsumNode, OpType,
};

fn main() {
    println!("=== Advanced Graph Algorithms Examples ===\n");

    // Example 1: Cycle Detection
    example_1_cycle_detection();

    // Example 2: Strongly Connected Components
    example_2_scc();

    // Example 3: Topological Sorting
    example_3_topological_sort();

    // Example 4: DAG Verification
    example_4_dag_verification();

    // Example 5: Graph Isomorphism
    example_5_isomorphism();

    // Example 6: Critical Path Analysis
    example_6_critical_path();

    // Example 7: Graph Diameter
    example_7_diameter();

    // Example 8: All Paths Finding
    example_8_all_paths();
}

fn example_1_cycle_detection() {
    println!("Example 1: Cycle Detection");
    println!("Find cycles in the computation graph\n");

    // Create an acyclic graph: A -> B -> C
    let mut graph = EinsumGraph::new();
    let a = graph.add_tensor("A");
    let b = graph.add_tensor("B");
    let c = graph.add_tensor("C");

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![a],
            outputs: vec![b],
            metadata: Default::default(),
        })
        .unwrap();

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![b],
            outputs: vec![c],
            metadata: Default::default(),
        })
        .unwrap();

    let cycles = find_cycles(&graph);

    println!("  Graph: A → B → C");
    println!("  Number of cycles: {}", cycles.len());
    println!("  Is acyclic: {}\n", cycles.is_empty());
}

fn example_2_scc() {
    println!("Example 2: Strongly Connected Components (Tarjan's Algorithm)");
    println!("Find maximal strongly connected subgraphs\n");

    // Create a graph with multiple tensors
    let mut graph = EinsumGraph::new();
    let a = graph.add_tensor("A");
    let b = graph.add_tensor("B");
    let c = graph.add_tensor("C");
    let d = graph.add_tensor("D");

    // A -> B -> C
    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![a],
            outputs: vec![b],
            metadata: Default::default(),
        })
        .unwrap();

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![b],
            outputs: vec![c],
            metadata: Default::default(),
        })
        .unwrap();

    // D is isolated
    let _ = d;

    let sccs = strongly_connected_components(&graph);

    println!("  Graph has {} tensors", graph.tensors.len());
    println!("  Number of SCCs: {}", sccs.len());

    for (i, scc) in sccs.iter().enumerate() {
        println!("  SCC {}: {} tensors", i + 1, scc.tensors.len());
        for &tensor_idx in &scc.tensors {
            println!("    - {}", graph.tensors[tensor_idx]);
        }
    }
    println!();
}

fn example_3_topological_sort() {
    println!("Example 3: Topological Sorting");
    println!("Find a valid execution order for the graph\n");

    let mut graph = EinsumGraph::new();
    let a = graph.add_tensor("Input_A");
    let b = graph.add_tensor("Input_B");
    let c = graph.add_tensor("Intermediate_C");
    let d = graph.add_tensor("Output_D");

    // A, B -> C -> D
    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i,j->ij".to_string(),
            },
            inputs: vec![a, b],
            outputs: vec![c],
            metadata: Default::default(),
        })
        .unwrap();

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "ij->ij".to_string(),
            },
            inputs: vec![c],
            outputs: vec![d],
            metadata: Default::default(),
        })
        .unwrap();

    println!("  Graph: Input_A, Input_B → Intermediate_C → Output_D");

    match topological_sort(&graph) {
        Some(order) => {
            println!("  ✓ Topological ordering found:");
            for idx in order {
                println!("    {} (index {})", graph.tensors[idx], idx);
            }
        }
        None => println!("  ✗ Graph contains cycles, no topological ordering exists"),
    }
    println!();
}

fn example_4_dag_verification() {
    println!("Example 4: Directed Acyclic Graph (DAG) Verification");
    println!("Check if the graph is acyclic (required for valid computation)\n");

    // Create a simple DAG
    let mut graph = EinsumGraph::new();
    let a = graph.add_tensor("A");
    let b = graph.add_tensor("B");

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![a],
            outputs: vec![b],
            metadata: Default::default(),
        })
        .unwrap();

    let is_acyclic = is_dag(&graph);

    println!("  Graph: A → B");
    println!("  Is DAG: {}", is_acyclic);
    println!("  Can be executed: {}\n", is_acyclic);
}

fn example_5_isomorphism() {
    println!("Example 5: Graph Isomorphism Testing");
    println!("Check if two graphs have the same structure\n");

    // Create first graph: A -> B
    let mut g1 = EinsumGraph::new();
    let a1 = g1.add_tensor("A");
    let b1 = g1.add_tensor("B");
    g1.add_node(EinsumNode {
        op: OpType::Einsum {
            spec: "i->i".to_string(),
        },
        inputs: vec![a1],
        outputs: vec![b1],
        metadata: Default::default(),
    })
    .unwrap();

    // Create second graph: X -> Y (same structure, different names)
    let mut g2 = EinsumGraph::new();
    let x2 = g2.add_tensor("X");
    let y2 = g2.add_tensor("Y");
    g2.add_node(EinsumNode {
        op: OpType::Einsum {
            spec: "i->i".to_string(),
        },
        inputs: vec![x2],
        outputs: vec![y2],
        metadata: Default::default(),
    })
    .unwrap();

    println!("  Graph 1: A → B");
    println!("  Graph 2: X → Y");

    match are_isomorphic(&g1, &g2) {
        tensorlogic_ir::IsomorphismResult::Isomorphic { mapping } => {
            println!("  ✓ Graphs are isomorphic!");
            println!("  Tensor mapping:");
            for (from, to) in mapping {
                println!("    {} → {}", g1.tensors[from], g2.tensors[to]);
            }
        }
        tensorlogic_ir::IsomorphismResult::NotIsomorphic => {
            println!("  ✗ Graphs are not isomorphic");
        }
    }
    println!();
}

fn example_6_critical_path() {
    println!("Example 6: Critical Path Analysis");
    println!("Find the longest path (determines minimum execution time)\n");

    let mut graph = EinsumGraph::new();
    let a = graph.add_tensor("Input");
    let b = graph.add_tensor("Intermediate1");
    let c = graph.add_tensor("Intermediate2");
    let d = graph.add_tensor("Output");

    // Create a diamond pattern
    // A -> B -> D
    // A -> C -> D
    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![a],
            outputs: vec![b],
            metadata: Default::default(),
        })
        .unwrap();

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![a],
            outputs: vec![c],
            metadata: Default::default(),
        })
        .unwrap();

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i,i->i".to_string(),
            },
            inputs: vec![b, c],
            outputs: vec![d],
            metadata: Default::default(),
        })
        .unwrap();

    // Assign weights (execution costs)
    let mut weights = HashMap::new();
    weights.insert(b, 2.0); // Heavy computation
    weights.insert(c, 1.0); // Light computation
    weights.insert(d, 1.0);

    println!("  Graph: Diamond pattern with different path costs");

    match critical_path_analysis(&graph, &weights) {
        Some(path) => {
            println!("  ✓ Critical path found:");
            println!("    Length: {} time units", path.length);
            println!("    Path:");
            for &tensor_idx in &path.tensors {
                println!("      - {}", graph.tensors[tensor_idx]);
            }
            println!("  This path determines the minimum execution time");
        }
        None => println!("  ✗ No critical path (graph may contain cycles)"),
    }
    println!();
}

fn example_7_diameter() {
    println!("Example 7: Graph Diameter");
    println!("Compute the longest shortest path (graph diameter)\n");

    let mut graph = EinsumGraph::new();
    let a = graph.add_tensor("A");
    let b = graph.add_tensor("B");
    let c = graph.add_tensor("C");
    let d = graph.add_tensor("D");

    // Linear chain: A -> B -> C -> D
    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![a],
            outputs: vec![b],
            metadata: Default::default(),
        })
        .unwrap();

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![b],
            outputs: vec![c],
            metadata: Default::default(),
        })
        .unwrap();

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![c],
            outputs: vec![d],
            metadata: Default::default(),
        })
        .unwrap();

    println!("  Graph: A → B → C → D (linear chain)");

    match graph_diameter(&graph) {
        Some(diameter) => {
            println!("  Diameter: {} edges", diameter);
            println!("  This is the maximum distance between any two tensors");
        }
        None => println!("  Could not compute diameter"),
    }
    println!();
}

fn example_8_all_paths() {
    println!("Example 8: Finding All Paths Between Nodes");
    println!("Enumerate all possible execution paths\n");

    let mut graph = EinsumGraph::new();
    let a = graph.add_tensor("Start");
    let b = graph.add_tensor("Path1");
    let c = graph.add_tensor("Path2");
    let d = graph.add_tensor("End");

    // Diamond: Start -> {Path1, Path2} -> End
    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![a],
            outputs: vec![b],
            metadata: Default::default(),
        })
        .unwrap();

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i->i".to_string(),
            },
            inputs: vec![a],
            outputs: vec![c],
            metadata: Default::default(),
        })
        .unwrap();

    graph
        .add_node(EinsumNode {
            op: OpType::Einsum {
                spec: "i,i->i".to_string(),
            },
            inputs: vec![b, c],
            outputs: vec![d],
            metadata: Default::default(),
        })
        .unwrap();

    let paths = find_all_paths(&graph, a, d);

    println!("  Graph: Diamond pattern (Start → End)");
    println!("  Number of paths from Start to End: {}", paths.len());

    for (i, path) in paths.iter().enumerate() {
        println!("  Path {}:", i + 1);
        for &tensor_idx in path {
            println!("    - {}", graph.tensors[tensor_idx]);
        }
    }
    println!();
}
