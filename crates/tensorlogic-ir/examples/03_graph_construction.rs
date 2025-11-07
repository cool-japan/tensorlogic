//! Computation Graph Construction
//!
//! This example demonstrates how to build tensor computation graphs (EinsumGraph)
//! that represent the compiled form of logical expressions.

use tensorlogic_ir::{EinsumGraph, EinsumNode, IrError};

fn main() -> Result<(), IrError> {
    println!("=== TensorLogic IR: Graph Construction ===\n");

    // 1. Basic Graph with Single Operation
    println!("1. Basic Graph with Single Operation:");

    let mut graph1 = EinsumGraph::new();

    // Add tensors
    let input_a = graph1.add_tensor("input_a");
    let output = graph1.add_tensor("output");

    // Add ReLU activation node
    graph1.add_node(EinsumNode::elem_unary("relu", input_a, output))?;

    graph1.add_output(output)?;

    println!("   Graph with 1 node (ReLU activation)");
    println!("   Tensors: {:?}", graph1.tensors);
    println!("   Nodes: {} operation(s)", graph1.nodes.len());
    println!("   Outputs: {:?}", graph1.outputs);

    // Validate the graph
    match graph1.validate() {
        Ok(_) => println!("   ✓ Graph is valid"),
        Err(e) => println!("   ✗ Validation error: {:?}", e),
    }

    // 2. Matrix Multiplication Graph
    println!("\n2. Matrix Multiplication Graph:");

    let mut graph2 = EinsumGraph::new();

    // Create tensors for matrix multiplication: C = A @ B
    let mat_a = graph2.add_tensor("matrix_a");
    let mat_b = graph2.add_tensor("matrix_b");
    let mat_c = graph2.add_tensor("matrix_c");

    // Einsum specification: "ik,kj->ij" (matrix multiply)
    graph2.add_node(EinsumNode::einsum(
        "ik,kj->ij",
        vec![mat_a, mat_b],
        vec![mat_c],
    ))?;

    graph2.add_output(mat_c)?;

    println!("   Matrix multiplication: C = A @ B");
    println!("   Einsum spec: ik,kj->ij");
    println!("   Tensors: {:?}", graph2.tensors);

    match graph2.validate() {
        Ok(_) => println!("   ✓ Graph is valid"),
        Err(e) => println!("   ✗ Error: {:?}", e),
    }

    // 3. Multi-Stage Computation
    println!("\n3. Multi-Stage Computation:");

    let mut graph3 = EinsumGraph::new();

    // Stage 1: Matrix multiply
    let input_a = graph3.add_tensor("input_a");
    let input_b = graph3.add_tensor("input_b");
    let intermediate = graph3.add_tensor("intermediate");

    graph3.add_node(EinsumNode::einsum(
        "ik,kj->ij",
        vec![input_a, input_b],
        vec![intermediate],
    ))?;

    // Stage 2: Add bias
    let bias = graph3.add_tensor("bias");
    let after_bias = graph3.add_tensor("after_bias");

    graph3.add_node(EinsumNode::elem_binary(
        "add",
        intermediate,
        bias,
        after_bias,
    ))?;

    // Stage 3: Apply ReLU
    let output = graph3.add_tensor("output");

    graph3.add_node(EinsumNode::elem_unary("relu", after_bias, output))?;

    graph3.add_output(output)?;

    println!("   3-stage computation:");
    println!("   1. Matrix multiply: A @ B");
    println!("   2. Add bias: result + bias");
    println!("   3. Activation: ReLU(result)");
    println!("   Total nodes: {}", graph3.nodes.len());

    match graph3.validate() {
        Ok(_) => println!("   ✓ Graph is valid"),
        Err(e) => println!("   ✗ Error: {:?}", e),
    }

    // 4. Reduction Operations
    println!("\n4. Reduction Operations:");

    let mut graph4 = EinsumGraph::new();

    let tensor_3d = graph4.add_tensor("tensor_3d"); // Shape: (batch, seq, hidden)
    let reduced = graph4.add_tensor("reduced"); // Shape: (batch, hidden)

    // Sum reduction along axis 1 (sequence dimension)
    graph4.add_node(EinsumNode::reduce("sum", vec![1], tensor_3d, reduced))?;

    graph4.add_output(reduced)?;

    println!("   Reduce sum along axis 1");
    println!("   Input: (batch, seq, hidden) -> Output: (batch, hidden)");

    match graph4.validate() {
        Ok(_) => println!("   ✓ Graph is valid"),
        Err(e) => println!("   ✗ Error: {:?}", e),
    }

    // 5. Element-wise Operations
    println!("\n5. Element-wise Binary Operations:");

    let mut graph5 = EinsumGraph::new();

    let tensor_a = graph5.add_tensor("a");
    let tensor_b = graph5.add_tensor("b");
    let product = graph5.add_tensor("product");
    let sum_tensor = graph5.add_tensor("sum");

    // Hadamard product (element-wise multiplication)
    graph5.add_node(EinsumNode::elem_binary("mul", tensor_a, tensor_b, product))?;

    // Element-wise addition
    let tensor_c = graph5.add_tensor("c");
    graph5.add_node(EinsumNode::elem_binary(
        "add", product, tensor_c, sum_tensor,
    ))?;

    graph5.add_output(sum_tensor)?;

    println!("   (A ⊙ B) + C  (⊙ = Hadamard product)");
    println!("   Operations: mul, add");

    match graph5.validate() {
        Ok(_) => println!("   ✓ Graph is valid"),
        Err(e) => println!("   ✗ Error: {:?}", e),
    }

    // 6. Complex Einsum Specifications
    println!("\n6. Complex Einsum Specifications:");

    let mut graph6 = EinsumGraph::new();

    // Batch matrix multiplication: "bik,bkj->bij"
    let batch_a = graph6.add_tensor("batch_a"); // (batch, i, k)
    let batch_b = graph6.add_tensor("batch_b"); // (batch, k, j)
    let batch_c = graph6.add_tensor("batch_c"); // (batch, i, j)

    graph6.add_node(EinsumNode::einsum(
        "bik,bkj->bij",
        vec![batch_a, batch_b],
        vec![batch_c],
    ))?;

    println!("   Batch matrix multiply: bik,bkj->bij");

    // Bilinear form: "bi,ij,bj->b"
    let vec_a = graph6.add_tensor("vec_a"); // (batch, i)
    let matrix_m = graph6.add_tensor("matrix_m"); // (i, j)
    let vec_b = graph6.add_tensor("vec_b"); // (batch, j)
    let scalar_out = graph6.add_tensor("scalar_out"); // (batch,)

    graph6.add_node(EinsumNode::einsum(
        "bi,ij,bj->b",
        vec![vec_a, matrix_m, vec_b],
        vec![scalar_out],
    ))?;

    println!("   Bilinear form: bi,ij,bj->b");

    graph6.add_output(batch_c)?;
    graph6.add_output(scalar_out)?;

    match graph6.validate() {
        Ok(_) => println!("   ✓ Graph is valid with multiple outputs"),
        Err(e) => println!("   ✗ Error: {:?}", e),
    }

    // 7. Multiple Outputs
    println!("\n7. Graph with Multiple Outputs:");

    let mut graph7 = EinsumGraph::new();

    let input = graph7.add_tensor("input");

    // Output 1: squared values
    let squared = graph7.add_tensor("squared");
    graph7.add_node(EinsumNode::elem_binary("mul", input, input, squared))?;

    // Output 2: negated values
    let negated = graph7.add_tensor("negated");
    graph7.add_node(EinsumNode::elem_unary("neg", input, negated))?;

    // Output 3: exponential
    let exp_out = graph7.add_tensor("exp");
    graph7.add_node(EinsumNode::elem_unary("exp", input, exp_out))?;

    graph7.add_output(squared)?;
    graph7.add_output(negated)?;
    graph7.add_output(exp_out)?;

    println!("   Three outputs from same input:");
    println!("   1. squared = input * input");
    println!("   2. negated = -input");
    println!("   3. exp = exp(input)");

    match graph7.validate() {
        Ok(_) => println!("   ✓ Graph with {} outputs is valid", graph7.outputs.len()),
        Err(e) => println!("   ✗ Error: {:?}", e),
    }

    // 8. Graph Statistics
    println!("\n8. Graph Statistics:");

    println!("   Graph 3 (multi-stage) stats:");
    println!("   - Tensors: {}", graph3.tensors.len());
    println!("   - Nodes: {}", graph3.nodes.len());
    println!("   - Outputs: {}", graph3.outputs.len());

    println!("\n   Graph 6 (complex einsum) stats:");
    println!("   - Tensors: {}", graph6.tensors.len());
    println!("   - Nodes: {}", graph6.nodes.len());
    println!("   - Outputs: {}", graph6.outputs.len());

    // 9. Graph Cloning and Serialization
    println!("\n9. Graph Cloning:");

    let cloned_graph = graph3.clone();
    println!("   Original graph: {} nodes", graph3.nodes.len());
    println!("   Cloned graph: {} nodes", cloned_graph.nodes.len());
    println!("   ✓ Graphs can be cloned for independent manipulation");

    println!("\n=== Example Complete ===");

    Ok(())
}
