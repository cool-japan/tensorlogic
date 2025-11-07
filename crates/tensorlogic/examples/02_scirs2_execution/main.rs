//! Example demonstrating end-to-end logic compilation and execution with SciRS2 backend.
//!
//! This example shows:
//! 1. Defining a logic rule (Parent → Ancestor)
//! 2. Compiling to einsum graph
//! 3. Creating input tensors with SciRS2
//! 4. Executing the graph
//! 5. Inspecting outputs

use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_infer::{TlAutodiff, TlExecutor};
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_scirs_backend::Scirs2Exec;

fn main() -> anyhow::Result<()> {
    println!("=== Tensorlogic + SciRS2: End-to-End Execution ===\n");

    // 1. Define logic rule: Parent(x, y) → Ancestor(x, y)
    let parent = TLExpr::pred("Parent", vec![Term::var("x"), Term::var("y")]);
    let ancestor = TLExpr::pred("Ancestor", vec![Term::var("x"), Term::var("y")]);
    let rule = TLExpr::imply(parent.clone(), ancestor.clone());

    println!("Logic Rule: Parent(x, y) → Ancestor(x, y)");
    println!("Interpretation: If Parent(x,y) holds, then Ancestor(x,y) should hold\n");

    // 2. Compile to einsum graph
    println!("Compiling to tensor graph...");
    let graph = compile_to_einsum(&rule)?;
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());

    for (idx, tensor_name) in graph.tensors.iter().enumerate() {
        println!("  Tensor {}: {}", idx, tensor_name);
    }

    for (idx, node) in graph.nodes.iter().enumerate() {
        println!("  Node {}: {:?} (inputs: {:?})", idx, node.op, node.inputs);
    }
    println!();

    // 3. Create SciRS2 executor and input tensors
    println!("Creating SciRS2 executor...");
    let mut executor = Scirs2Exec::new();

    // Create a 3x3 Parent relation tensor (3 people, 3 people)
    // Parent[i,j] = 1.0 if person i is parent of person j
    let parent_data = vec![
        0.0, 1.0, 1.0, // Person 0 is parent of 1, 2
        0.0, 0.0, 0.0, // Person 1 has no children
        0.0, 0.0, 0.0, // Person 2 has no children
    ];

    let parent_tensor = Scirs2Exec::from_vec(parent_data, vec![3, 3])?;
    println!("Parent tensor shape: {:?}", parent_tensor.shape());
    println!("Parent tensor:\n{:?}\n", parent_tensor);

    // Create expected Ancestor tensor (also 3x3, same as Parent for this simple case)
    let ancestor_data = vec![
        0.0, 1.0, 1.0, // Person 0 is ancestor of 1, 2
        0.0, 0.0, 0.0, // Person 1 has no descendants
        0.0, 0.0, 0.0, // Person 2 has no descendants
    ];

    let ancestor_tensor = Scirs2Exec::from_vec(ancestor_data, vec![3, 3])?;
    println!(
        "Expected Ancestor tensor shape: {:?}",
        ancestor_tensor.shape()
    );
    println!("Expected Ancestor tensor:\n{:?}\n", ancestor_tensor);

    // Add tensors to executor storage using dynamic names from compiled graph
    executor.add_tensor(graph.tensors[0].clone(), parent_tensor);
    executor.add_tensor(graph.tensors[1].clone(), ancestor_tensor);

    // 4. Execute the graph
    println!("Executing forward pass...");
    let output = executor.forward(&graph)?;
    println!("Output tensor shape: {:?}", output.shape());
    println!("Output tensor (constraint violation):\n{:?}\n", output);

    // 5. Interpret results
    println!("=== Interpretation ===");
    println!("The output tensor represents constraint violations (via ReLU(Ancestor - Parent)).");
    println!("  - Values near 0: rule satisfied (Parent implies Ancestor)");
    println!("  - Positive values: violations (Parent holds but Ancestor doesn't)\n");

    // Example 2: Test element-wise operations
    println!("=== Testing SciRS2 Element-wise Operations ===\n");

    let test_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let test_tensor = Scirs2Exec::from_vec(test_data, vec![5])?;

    println!("Input: {:?}", test_tensor);

    let relu_result = executor.elem_op(tensorlogic_infer::ElemOp::Relu, &test_tensor)?;
    println!("ReLU:  {:?}", relu_result);

    let sigmoid_result = executor.elem_op(tensorlogic_infer::ElemOp::Sigmoid, &test_tensor)?;
    println!("Sigmoid: {:?}", sigmoid_result);

    let one_minus_result = executor.elem_op(tensorlogic_infer::ElemOp::OneMinus, &test_tensor)?;
    println!("OneMinus: {:?}\n", one_minus_result);

    // Example 3: Test reduction operations
    println!("=== Testing SciRS2 Reduction Operations ===\n");

    let matrix_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let matrix = Scirs2Exec::from_vec(matrix_data, vec![2, 3])?;

    println!("Input matrix (2x3):\n{:?}\n", matrix);

    let sum_axis0 = executor.reduce(tensorlogic_infer::ReduceOp::Sum, &matrix, &[0])?;
    println!("Sum over axis 0: {:?}", sum_axis0);

    let sum_axis1 = executor.reduce(tensorlogic_infer::ReduceOp::Sum, &matrix, &[1])?;
    println!("Sum over axis 1: {:?}", sum_axis1);

    let max_axis0 = executor.reduce(tensorlogic_infer::ReduceOp::Max, &matrix, &[0])?;
    println!("Max over axis 0: {:?}", max_axis0);

    let mean_axis1 = executor.reduce(tensorlogic_infer::ReduceOp::Mean, &matrix, &[1])?;
    println!("Mean over axis 1: {:?}\n", mean_axis1);

    println!("=== Execution Complete ===");
    println!("\nThis demonstrates the full pipeline:");
    println!("  Logic → IR → Einsum Graph → SciRS2 Execution → Results");

    Ok(())
}
