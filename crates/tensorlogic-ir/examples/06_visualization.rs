//! Visualization and DOT Export
//!
//! This example demonstrates how to visualize computation graphs using DOT format
//! and generate human-readable representations of expressions and graphs.

use std::env;
use std::fs;
use tensorlogic_ir::{
    export_to_dot, export_to_dot_with_options, pretty_print_expr, pretty_print_graph,
    DotExportOptions, EinsumGraph, EinsumNode, IrError, TLExpr, Term,
};

fn main() -> Result<(), IrError> {
    println!("=== TensorLogic IR: Visualization ===\n");

    // 1. Pretty Printing Expressions
    println!("1. Pretty Printing Expressions:");

    let expr = TLExpr::forall(
        "x",
        "Person",
        TLExpr::imply(
            TLExpr::and(
                TLExpr::pred("Person", vec![Term::var("x")]),
                TLExpr::pred("Wise", vec![Term::var("x")]),
            ),
            TLExpr::pred("Respected", vec![Term::var("x")]),
        ),
    );

    println!("   Debug format: {:?}", expr);
    println!("\n   Pretty format:");
    println!("{}", pretty_print_expr(&expr));

    // 2. Pretty Printing Nested Expressions
    println!("\n2. Pretty Printing Nested Expressions:");

    let nested = TLExpr::exists(
        "x",
        "Person",
        TLExpr::forall(
            "y",
            "City",
            TLExpr::imply(
                TLExpr::pred("livesIn", vec![Term::var("x"), Term::var("y")]),
                TLExpr::and(
                    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
                    TLExpr::pred("visits", vec![Term::var("x"), Term::var("y")]),
                ),
            ),
        ),
    );

    println!("{}", pretty_print_expr(&nested));

    // 3. Pretty Printing Arithmetic Expressions
    println!("\n3. Pretty Printing Arithmetic Expressions:");

    let arithmetic = TLExpr::if_then_else(
        TLExpr::gt(
            TLExpr::add(
                TLExpr::pred("score", vec![Term::var("x")]),
                TLExpr::constant(10.0),
            ),
            TLExpr::constant(90.0),
        ),
        TLExpr::constant(1.0), // Pass
        TLExpr::constant(0.0), // Fail
    );

    println!("{}", pretty_print_expr(&arithmetic));

    // 4. Building a Graph to Visualize
    println!("\n4. Building a Computation Graph:");

    let mut graph = EinsumGraph::new();

    // Input layer
    let input = graph.add_tensor("input");
    let weights1 = graph.add_tensor("weights1");

    // Hidden layer
    let hidden = graph.add_tensor("hidden");
    graph.add_node(EinsumNode::einsum(
        "bi,ij->bj",
        vec![input, weights1],
        vec![hidden],
    ))?;

    // Activation
    let activated = graph.add_tensor("activated");
    graph.add_node(EinsumNode::elem_unary("relu", hidden, activated))?;

    // Output layer
    let weights2 = graph.add_tensor("weights2");
    let output = graph.add_tensor("output");
    graph.add_node(EinsumNode::einsum(
        "bi,ij->bj",
        vec![activated, weights2],
        vec![output],
    ))?;

    graph.add_output(output)?;

    println!("   Created 2-layer neural network graph");
    println!("   - Tensors: {}", graph.tensors.len());
    println!("   - Nodes: {}", graph.nodes.len());

    // 5. Pretty Printing Graph
    println!("\n5. Pretty Printing Graph:");

    pretty_print_graph(&graph);

    // 6. DOT Export (Basic)
    println!("\n6. DOT Export (Basic):");

    let dot_basic = export_to_dot(&graph);
    println!("   DOT format (basic):");
    println!("{}", dot_basic);

    // 7. DOT Export with Options
    println!("\n7. DOT Export with Custom Options:");

    let options = DotExportOptions {
        show_tensor_ids: true,
        show_node_ids: true,
        show_metadata: true,
        cluster_by_operation: false,
        horizontal_layout: false,
        show_shapes: false,
        highlight_tensors: vec![],
        highlight_nodes: vec![],
    };

    let dot_custom = export_to_dot_with_options(&graph, &options);
    println!("   DOT format (with options):");
    println!("{}", dot_custom);

    // 8. Saving DOT Files
    println!("\n8. Saving DOT Files:");

    let temp_dir = env::temp_dir();
    let dot_path = temp_dir.join("tensorlogic_graph.dot");

    fs::write(&dot_path, &dot_custom).expect("Failed to write DOT file");
    println!("   ✓ Saved DOT file to: {:?}", dot_path);
    println!("   To visualize: dot -Tpng {:?} -o graph.png", dot_path);

    // 9. Complex Graph Visualization
    println!("\n9. Complex Graph Visualization:");

    let mut complex_graph = EinsumGraph::new();

    // Multi-head attention pattern
    let query = complex_graph.add_tensor("query");
    let key = complex_graph.add_tensor("key");
    let value = complex_graph.add_tensor("value");

    // Q @ K^T
    let attention_scores = complex_graph.add_tensor("attention_scores");
    complex_graph.add_node(EinsumNode::einsum(
        "bhqd,bhkd->bhqk",
        vec![query, key],
        vec![attention_scores],
    ))?;

    // Softmax (approximated as normalize)
    let attention_weights = complex_graph.add_tensor("attention_weights");
    complex_graph.add_node(EinsumNode::elem_unary(
        "softmax",
        attention_scores,
        attention_weights,
    ))?;

    // Attention @ V
    let attention_output = complex_graph.add_tensor("attention_output");
    complex_graph.add_node(EinsumNode::einsum(
        "bhqk,bhkd->bhqd",
        vec![attention_weights, value],
        vec![attention_output],
    ))?;

    complex_graph.add_output(attention_output)?;

    println!("   Multi-head attention graph:");
    pretty_print_graph(&complex_graph);

    let attention_options = DotExportOptions {
        show_tensor_ids: true,
        show_node_ids: true,
        show_metadata: true,
        cluster_by_operation: true,
        horizontal_layout: true, // Horizontal for attention
        show_shapes: false,
        highlight_tensors: vec![],
        highlight_nodes: vec![],
    };

    let attention_dot = export_to_dot_with_options(&complex_graph, &attention_options);
    let attention_path = temp_dir.join("attention_graph.dot");
    fs::write(&attention_path, &attention_dot).expect("Failed to write attention graph DOT file");
    println!("   ✓ Saved attention graph DOT to: {:?}", attention_path);

    // 10. Graph Statistics Visualization
    println!("\n10. Graph Statistics:");

    use tensorlogic_ir::{ExprStats, GraphStats};

    // Expression statistics
    let expr_stats = ExprStats::compute(&expr);
    println!("   Expression statistics:");
    println!("   - Node count: {}", expr_stats.node_count);
    println!("   - Max depth: {}", expr_stats.max_depth);
    println!("   - Free variable count: {}", expr_stats.free_var_count);
    println!("   - Predicates: {}", expr_stats.predicate_count);
    println!("   - Quantifiers: {}", expr_stats.quantifier_count);
    println!("   - Logical ops: {}", expr_stats.logical_op_count);

    // Graph statistics
    let graph_stats = GraphStats::compute(&complex_graph);
    println!("\n   Graph statistics:");
    println!("   - Tensors: {}", graph_stats.tensor_count);
    println!("   - Nodes: {}", graph_stats.node_count);
    println!("   - Einsum ops: {}", graph_stats.einsum_count);
    println!("   - Elem unary ops: {}", graph_stats.elem_unary_count);
    println!("   - Elem binary ops: {}", graph_stats.elem_binary_count);
    println!("   - Reduce ops: {}", graph_stats.reduce_count);

    // 11. Comparison Visualization
    println!("\n11. Graph Comparison:");

    use tensorlogic_ir::diff_graphs;

    let mut graph_v1 = EinsumGraph::new();
    let a = graph_v1.add_tensor("a");
    let b = graph_v1.add_tensor("b");
    let out = graph_v1.add_tensor("out");
    graph_v1.add_node(EinsumNode::elem_binary("add", a, b, out))?;
    graph_v1.add_output(out)?;

    let mut graph_v2 = graph_v1.clone();
    // Add extra operation
    let _c = graph_v2.add_tensor("c");
    let out2 = graph_v2.add_tensor("out2");
    graph_v2.add_node(EinsumNode::elem_unary("relu", out, out2))?;

    let diffs = diff_graphs(&graph_v1, &graph_v2);
    println!("   Graph differences:");
    println!("   - Left only tensors: {:?}", diffs.left_only_tensors);
    println!("   - Right only tensors: {:?}", diffs.right_only_tensors);
    println!("   - Left only nodes: {}", diffs.left_only_nodes);
    println!("   - Right only nodes: {}", diffs.right_only_nodes);
    println!("   - Node differences: {}", diffs.node_differences.len());
    println!(
        "   - Output differences: {}",
        diffs.output_differences.len()
    );

    // 12. Visualization Commands
    println!("\n12. Generating Visualizations:");

    println!("\n   To generate PNG images from DOT files:");
    println!("   $ dot -Tpng {:?} -o mlp.png", dot_path);
    println!("   $ dot -Tpng {:?} -o attention.png", attention_path);

    println!("\n   To generate SVG (scalable):");
    println!("   $ dot -Tsvg {:?} -o mlp.svg", dot_path);

    println!("\n   To generate PDF:");
    println!("   $ dot -Tpdf {:?} -o mlp.pdf", dot_path);

    println!("\n   Interactive visualization:");
    println!("   $ xdot {:?}", dot_path);

    // Clean up
    fs::remove_file(&dot_path).ok();
    fs::remove_file(&attention_path).ok();
    println!("\n   ✓ Cleaned up temporary files");

    println!("\n=== Example Complete ===");
    println!("\nNote: Install Graphviz to visualize DOT files:");
    println!("  macOS: brew install graphviz");
    println!("  Ubuntu: sudo apt-get install graphviz");
    println!("  Windows: choco install graphviz");

    Ok(())
}
