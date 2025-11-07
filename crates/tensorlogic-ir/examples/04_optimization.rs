//! Graph Optimization Pipeline
//!
//! This example demonstrates the graph optimization capabilities including
//! dead code elimination, common subexpression elimination, and simplification.
//!
//! NOTE: This example is currently disabled because the optimization API is not yet
//! publicly exposed. It will be enabled in a future release when the optimization
//! module is stabilized and made part of the public API.

// Commented out until optimization API is made public

/*
use tensorlogic_ir::{EinsumGraph, EinsumNode, IrError, OpType};

fn main() -> Result<(), IrError> {
    println!("=== TensorLogic IR: Graph Optimization ===\n");

    // 1. Dead Code Elimination
    println!("1. Dead Code Elimination:");

    let mut graph1 = EinsumGraph::new();

    // Add useful computation
    let input = graph1.add_tensor("input");
    let useful = graph1.add_tensor("useful");
    graph1.add_node(EinsumNode {
        inputs: vec![input],
        op: OpType::ElemUnary {
            op: "relu".to_string(),
        },
    })?;

    // Add dead computation (not connected to output)
    let dead_tensor = graph1.add_tensor("dead_tensor");
    graph1.add_node(EinsumNode {
        inputs: vec![dead_tensor],
        op: OpType::ElemUnary {
            op: "sigmoid".to_string(),
        },
    })?;

    graph1.add_output(useful)?;

    println!("   Before DCE:");
    println!("   - Tensors: {}", graph1.tensors.len());
    println!("   - Nodes: {}", graph1.nodes.len());

    let stats = graph1.eliminate_dead_code()?;

    println!("   After DCE:");
    println!("   - Tensors: {}", graph1.tensors.len());
    println!("   - Nodes: {}", graph1.nodes.len());
    println!("   - Dead nodes removed: {}", stats.dead_nodes_removed);

    // 2. Common Subexpression Elimination
    println!("\n2. Common Subexpression Elimination:");

    let mut graph2 = EinsumGraph::new();

    let input_a = graph2.add_tensor("a");
    let input_b = graph2.add_tensor("b");

    // First computation: a + b
    let sum1 = graph2.add_tensor("sum1");
    graph2.add_node(EinsumNode {
        inputs: vec![input_a, input_b],
        op: OpType::ElemBinary {
            op: "add".to_string(),
        },
    })?;

    // Duplicate computation: a + b (same operation, same inputs)
    let sum2 = graph2.add_tensor("sum2");
    graph2.add_node(EinsumNode {
        inputs: vec![input_a, input_b],
        op: OpType::ElemBinary {
            op: "add".to_string(),
        },
    })?;

    // Use both results
    let final_result = graph2.add_tensor("result");
    graph2.add_node(EinsumNode {
        inputs: vec![sum1, sum2],
        op: OpType::ElemBinary {
            op: "mul".to_string(),
        },
    })?;

    graph2.add_output(final_result)?;

    println!("   Before CSE:");
    println!("   - Nodes: {} (includes duplicate a+b)", graph2.nodes.len());

    let stats = graph2.common_subexpression_elimination()?;

    println!("   After CSE:");
    println!("   - Nodes: {} (duplicate removed)", graph2.nodes.len());
    println!("   - CSE count: {}", stats.cse_count);

    // 3. Identity Operation Simplification
    println!("\n3. Identity Operation Simplification:");

    let mut graph3 = EinsumGraph::new();

    let input = graph3.add_tensor("input");

    // Add identity operation (multiply by 1)
    let one_tensor = graph3.add_tensor("one");
    let after_identity = graph3.add_tensor("after_identity");
    graph3.add_node(EinsumNode {
        inputs: vec![input, one_tensor],
        op: OpType::ElemBinary {
            op: "mul_by_one".to_string(),
        },
    })?;

    // Useful operation
    let output = graph3.add_tensor("output");
    graph3.add_node(EinsumNode {
        inputs: vec![after_identity],
        op: OpType::ElemUnary {
            op: "relu".to_string(),
        },
    })?;

    graph3.add_output(output)?;

    println!("   Before simplification:");
    println!("   - Nodes: {}", graph3.nodes.len());

    let stats = graph3.simplify()?;

    println!("   After simplification:");
    println!("   - Nodes: {}", graph3.nodes.len());
    println!("   - Simplifications: {}", stats.simplifications);

    // 4. Full Optimization Pipeline
    println!("\n4. Full Optimization Pipeline:");

    let mut graph4 = EinsumGraph::new();

    // Build a graph with multiple optimization opportunities
    let a = graph4.add_tensor("a");
    let b = graph4.add_tensor("b");

    // Duplicate computation 1: a * b
    let prod1 = graph4.add_tensor("prod1");
    graph4.add_node(EinsumNode {
        inputs: vec![a, b],
        op: OpType::ElemBinary {
            op: "mul".to_string(),
        },
    })?;

    // Duplicate computation 2: a * b (same as above)
    let prod2 = graph4.add_tensor("prod2");
    graph4.add_node(EinsumNode {
        inputs: vec![a, b],
        op: OpType::ElemBinary {
            op: "mul".to_string(),
        },
    })?;

    // Use first product
    let result1 = graph4.add_tensor("result1");
    graph4.add_node(EinsumNode {
        inputs: vec![prod1],
        op: OpType::ElemUnary {
            op: "relu".to_string(),
        },
    })?;

    // Dead computation (not used)
    let dead = graph4.add_tensor("dead");
    graph4.add_node(EinsumNode {
        inputs: vec![prod2],
        op: OpType::ElemUnary {
            op: "sigmoid".to_string(),
        },
    })?;

    graph4.add_output(result1)?;

    println!("   Before optimization:");
    println!("   - Tensors: {}", graph4.tensors.len());
    println!("   - Nodes: {}", graph4.nodes.len());
    println!("   - Issues: duplicate computation + dead code");

    // Run full optimization pipeline
    let stats = graph4.optimize()?;

    println!("   After optimization:");
    println!("   - Tensors: {}", graph4.tensors.len());
    println!("   - Nodes: {}", graph4.nodes.len());
    println!("   - Dead nodes removed: {}", stats.dead_nodes_removed);
    println!("   - CSE count: {}", stats.cse_count);
    println!("   - Simplifications: {}", stats.simplifications);
    println!("   - Total passes: {}", stats.passes);

    // 5. Optimization on Complex Graph
    println!("\n5. Optimization on Complex Graph:");

    let mut graph5 = EinsumGraph::new();

    // Build a more complex computation graph
    let x = graph5.add_tensor("x");
    let y = graph5.add_tensor("y");
    let z = graph5.add_tensor("z");

    // Branch 1: (x + y) * z
    let sum_xy = graph5.add_tensor("sum_xy");
    graph5.add_node(EinsumNode {
        inputs: vec![x, y],
        op: OpType::ElemBinary {
            op: "add".to_string(),
        },
    })?;

    let prod1 = graph5.add_tensor("prod1");
    graph5.add_node(EinsumNode {
        inputs: vec![sum_xy, z],
        op: OpType::ElemBinary {
            op: "mul".to_string(),
        },
    })?;

    // Branch 2: (x + y) * 2  [duplicate x + y computation]
    let sum_xy_dup = graph5.add_tensor("sum_xy_dup");
    graph5.add_node(EinsumNode {
        inputs: vec![x, y],
        op: OpType::ElemBinary {
            op: "add".to_string(),
        },
    })?;

    let two = graph5.add_tensor("two");
    let prod2 = graph5.add_tensor("prod2");
    graph5.add_node(EinsumNode {
        inputs: vec![sum_xy_dup, two],
        op: OpType::ElemBinary {
            op: "mul".to_string(),
        },
    })?;

    // Combine branches
    let final_result = graph5.add_tensor("final");
    graph5.add_node(EinsumNode {
        inputs: vec![prod1, prod2],
        op: OpType::ElemBinary {
            op: "add".to_string(),
        },
    })?;

    // Dead branch (unused)
    let dead1 = graph5.add_tensor("dead1");
    graph5.add_node(EinsumNode {
        inputs: vec![x],
        op: OpType::ElemUnary {
            op: "exp".to_string(),
        },
    })?;

    let dead2 = graph5.add_tensor("dead2");
    graph5.add_node(EinsumNode {
        inputs: vec![dead1],
        op: OpType::ElemUnary {
            op: "log".to_string(),
        },
    })?;

    graph5.add_output(final_result)?;

    println!("   Complex graph before optimization:");
    println!("   - Tensors: {}", graph5.tensors.len());
    println!("   - Nodes: {}", graph5.nodes.len());

    let stats = graph5.optimize()?;

    println!("   After optimization:");
    println!("   - Tensors: {}", graph5.tensors.len());
    println!("   - Nodes: {}", graph5.nodes.len());
    println!("   - Optimizations applied:");
    println!("     * Dead nodes removed: {}", stats.dead_nodes_removed);
    println!("     * Common subexpressions: {}", stats.cse_count);
    println!("     * Simplifications: {}", stats.simplifications);

    // 6. Iterative Optimization
    println!("\n6. Iterative Optimization:");

    let mut graph6 = EinsumGraph::new();

    // Build graph that benefits from multiple passes
    let a = graph6.add_tensor("a");
    let b = graph6.add_tensor("b");
    let c = graph6.add_tensor("c");

    // Stage 1: a + b
    let sum1 = graph6.add_tensor("sum1");
    graph6.add_node(EinsumNode {
        inputs: vec![a, b],
        op: OpType::ElemBinary {
            op: "add".to_string(),
        },
    })?;

    // Stage 2: (a + b) + c
    let sum2 = graph6.add_tensor("sum2");
    graph6.add_node(EinsumNode {
        inputs: vec![sum1, c],
        op: OpType::ElemBinary {
            op: "add".to_string(),
        },
    })?;

    // Duplicate of stage 1
    let sum1_dup = graph6.add_tensor("sum1_dup");
    graph6.add_node(EinsumNode {
        inputs: vec![a, b],
        op: OpType::ElemBinary {
            op: "add".to_string(),
        },
    })?;

    // Using duplicate
    let result = graph6.add_tensor("result");
    graph6.add_node(EinsumNode {
        inputs: vec![sum2, sum1_dup],
        op: OpType::ElemBinary {
            op: "mul".to_string(),
        },
    })?;

    graph6.add_output(result)?;

    println!("   Before multi-pass optimization:");
    println!("   - Nodes: {}", graph6.nodes.len());

    // Run optimization with multiple passes
    let stats = graph6.optimize()?;

    println!("   After multi-pass optimization:");
    println!("   - Nodes: {}", graph6.nodes.len());
    println!("   - Total passes: {}", stats.passes);
    println!("   - Benefits from iterative application");

    // 7. Validation After Optimization
    println!("\n7. Validation After Optimization:");

    println!("   Validating all optimized graphs...");

    let graphs = vec![&graph1, &graph2, &graph3, &graph4, &graph5, &graph6];

    for (i, graph) in graphs.iter().enumerate() {
        match graph.validate() {
            Ok(_) => println!("   ✓ Graph {} is valid after optimization", i + 1),
            Err(e) => println!("   ✗ Graph {} validation error: {:?}", i + 1, e),
        }
    }

    println!("\n=== Example Complete ===");

    Ok(())
}
*/

fn main() {
    println!("=== TensorLogic IR: Graph Optimization ===\n");
    println!("This example is currently disabled.");
    println!("The optimization API is not yet publicly exposed.");
    println!("It will be enabled in a future release when the optimization module is stabilized.");
}
