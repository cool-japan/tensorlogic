//! Simplified error recovery demonstration.
//!
//! This example shows basic error handling concepts.
//! Full error recovery features are planned for future releases.
//!
//! Run with: cargo run --example recovery_demo

use tensorlogic_infer::{DummyExecutor, DummyTensor, ExecutorError};
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TensorLogic Error Recovery Demo ===\n");
    println!("NOTE: This is a simplified demonstration.");
    println!("Full recovery features are under development.\n");

    // Create a sample computation graph
    let graph = create_sample_graph();
    println!(
        "Created computation graph with {} nodes, {} tensors\n",
        graph.nodes.len(),
        graph.tensors.len()
    );

    // Demo basic error handling
    println!("--- Demo: Error Handling Patterns ---");
    demo_error_handling(&graph)?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn create_sample_graph() -> EinsumGraph {
    let mut graph = EinsumGraph::new();

    // Create tensors
    let x = graph.add_tensor("x");
    let weights = graph.add_tensor("weights");
    let matmul_out = graph.add_tensor("matmul_out");
    let final_out = graph.add_tensor("final_out");

    // Einsum operation (matrix multiplication)
    let einsum_node = EinsumNode {
        op: OpType::Einsum {
            spec: "ij,jk->ik".to_string(),
        },
        inputs: vec![x, weights],
        outputs: vec![matmul_out],
        metadata: None,
    };
    graph.nodes.push(einsum_node);

    // Activation
    let relu_node = EinsumNode {
        op: OpType::ElemUnary {
            op: "relu".to_string(),
        },
        inputs: vec![matmul_out],
        outputs: vec![final_out],
        metadata: None,
    };
    graph.nodes.push(relu_node);

    // Mark final output
    graph.add_output(final_out).ok();

    graph
}

fn demo_error_handling(graph: &EinsumGraph) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing error handling patterns...\n");

    // Test 1: Successful graph validation
    println!("Test 1: Graph validation (should succeed)");
    match graph.validate() {
        Ok(_) => println!("  ✓ Graph is valid"),
        Err(e) => println!("  ✗ Validation error: {}", e),
    }

    // Test 2: Invalid tensor reference
    println!("\nTest 2: Checking tensor reference validation");
    let mut invalid_graph = EinsumGraph::new();
    let invalid_node = EinsumNode {
        op: OpType::ElemUnary {
            op: "relu".to_string(),
        },
        inputs: vec![999], // Invalid tensor index
        outputs: vec![0],
        metadata: None,
    };
    invalid_graph.nodes.push(invalid_node);

    match invalid_graph.validate() {
        Ok(_) => println!("  ✗ Unexpectedly succeeded"),
        Err(e) => println!("  ✓ Correctly caught validation error: {}", e),
    }

    // Test 3: Executor error types
    println!("\nTest 3: Executor error types demonstration");
    let _executor = DummyExecutor::new();

    // Create sample tensors
    let _x = DummyTensor::new("x", vec![64, 128]);
    let _weights = DummyTensor::new("weights", vec![128, 256]);

    // Demonstrate error type matching
    let example_error = ExecutorError::TensorNotFound("missing_tensor".to_string());
    match example_error {
        ExecutorError::TensorNotFound(name) => {
            println!("  ✓ TensorNotFound error for: {}", name)
        }
        _ => println!("  ✗ Unexpected error type"),
    }

    println!("\nError handling patterns demonstrated!");
    println!("\nAvailable ExecutorError types:");
    println!("  - TensorNotFound");
    println!("  - InvalidEinsumSpec");
    println!("  - ShapeMismatch");
    println!("  - UnsupportedOperation");
    println!("  - InvalidAxis");
    println!("  - GraphValidationError");
    println!("  - InvalidDependencies");

    Ok(())
}
