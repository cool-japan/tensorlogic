//! Graph Export Example
//!
//! This example demonstrates exporting EinsumGraph to various ML interchange formats:
//! - ONNX (Open Neural Network Exchange) text representation
//! - TorchScript (PyTorch JIT) text representation
//!
//! These exports enable interoperability with other ML frameworks and tools.

use tensorlogic_ir::{
    export_to_onnx_text, export_to_onnx_text_with_options, export_to_torchscript_text,
    export_to_torchscript_text_with_options, EinsumGraph, EinsumNode, OnnxExportOptions,
    TorchScriptExportOptions,
};

fn main() {
    println!("=== TensorLogic IR Graph Export Demo ===\n");

    // Build a simple matrix multiplication graph
    let graph = build_matmul_graph();

    println!("Graph Structure:");
    println!("  Tensors: {:?}", graph.tensors);
    println!("  Nodes: {} operations", graph.nodes.len());
    println!("  Outputs: {:?}\n", graph.outputs);

    // Export to ONNX
    println!("--- ONNX Export ---");
    match export_to_onnx_text(&graph) {
        Ok(onnx) => {
            println!("{}", onnx);
        }
        Err(e) => eprintln!("ONNX export failed: {}", e),
    }

    println!("\n--- TorchScript Export ---");
    match export_to_torchscript_text(&graph) {
        Ok(script) => {
            println!("{}", script);
        }
        Err(e) => eprintln!("TorchScript export failed: {}", e),
    }

    // Demonstrate custom export options
    println!("\n--- Custom Export Options ---");

    let onnx_options = OnnxExportOptions {
        opset_version: 14,
        producer_name: "MyCustomProducer".to_string(),
        model_version: 2,
        ..Default::default()
    };

    println!("ONNX with custom options (opset=14, custom producer):");
    match export_to_onnx_text_with_options(&graph, &onnx_options) {
        Ok(onnx) => {
            // Print just the header to show custom options
            for line in onnx.lines().take(10) {
                println!("{}", line);
            }
            println!("...");
        }
        Err(e) => eprintln!("Custom ONNX export failed: {}", e),
    }

    let torch_options = TorchScriptExportOptions {
        include_comments: false,
        ..Default::default()
    };

    println!("\nTorchScript without comments:");
    match export_to_torchscript_text_with_options(&graph, &torch_options) {
        Ok(script) => {
            println!("{}", script);
        }
        Err(e) => eprintln!("Custom TorchScript export failed: {}", e),
    }

    // Build a more complex graph with various operations
    println!("\n\n=== Complex Graph Example ===\n");
    let complex_graph = build_complex_graph();

    println!("Complex Graph Structure:");
    println!("  Tensors: {:?}", complex_graph.tensors);
    println!("  Nodes: {} operations", complex_graph.nodes.len());
    println!();

    println!("--- Complex Graph ONNX Export ---");
    match export_to_onnx_text(&complex_graph) {
        Ok(onnx) => {
            // Print only the operations part
            let mut in_ops = false;
            for line in onnx.lines() {
                if line.contains("# Operations") {
                    in_ops = true;
                }
                if in_ops {
                    println!("{}", line);
                    if line.contains("# Outputs") {
                        break;
                    }
                }
            }
        }
        Err(e) => eprintln!("Complex ONNX export failed: {}", e),
    }

    println!("\n--- Complex Graph TorchScript Export ---");
    match export_to_torchscript_text(&complex_graph) {
        Ok(script) => {
            // Print only the forward method
            let mut in_forward = false;
            for line in script.lines() {
                if line.contains("def forward") {
                    in_forward = true;
                }
                if in_forward {
                    println!("{}", line);
                }
            }
        }
        Err(e) => eprintln!("Complex TorchScript export failed: {}", e),
    }
}

/// Build a simple matrix multiplication graph: Z = X @ W
fn build_matmul_graph() -> EinsumGraph {
    let mut graph = EinsumGraph::new();

    let x = graph.add_tensor("X");
    let w = graph.add_tensor("W");
    let z = graph.add_tensor("Z");

    // Matrix multiplication using einsum: Z = X @ W
    graph
        .add_node(EinsumNode::einsum("ij,jk->ik", vec![x, w], vec![z]))
        .expect("Failed to add node");

    graph.add_output(z).expect("Failed to add output");

    graph
}

/// Build a more complex graph with various operations
fn build_complex_graph() -> EinsumGraph {
    let mut graph = EinsumGraph::new();

    // Input tensors
    let x = graph.add_tensor("X");
    let w1 = graph.add_tensor("W1");
    let b1 = graph.add_tensor("B1");

    // Intermediate tensors
    let h1 = graph.add_tensor("H1");
    let h2 = graph.add_tensor("H2");
    let h3 = graph.add_tensor("H3");

    // Output tensors
    let y = graph.add_tensor("Y");

    // Layer 1: Linear transformation (matmul)
    graph
        .add_node(EinsumNode::einsum("bi,ij->bj", vec![x, w1], vec![h1]))
        .expect("Failed to add matmul node");

    // Add bias
    graph
        .add_node(EinsumNode::elem_binary("add", h1, b1, h2))
        .expect("Failed to add bias node");

    // Activation function (ReLU)
    graph
        .add_node(EinsumNode::elem_unary("relu", h2, h3))
        .expect("Failed to add relu node");

    // Global average pooling (reduce mean)
    graph
        .add_node(EinsumNode::reduce("mean", vec![1], h3, y))
        .expect("Failed to add reduce node");

    graph.add_output(y).expect("Failed to add output");

    graph
}
