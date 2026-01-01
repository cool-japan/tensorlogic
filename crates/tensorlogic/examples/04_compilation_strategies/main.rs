//! Example 04: Compilation Strategies Comparison
//!
//! This example demonstrates the different compilation strategies available in TensorLogic
//! and shows how they affect the semantics of logical operations.
//!
//! Strategies demonstrated:
//! - soft_differentiable: For neural network training (default)
//! - hard_boolean: Discrete Boolean logic
//! - fuzzy_godel: Gödel fuzzy logic
//! - fuzzy_product: Product fuzzy logic
//! - fuzzy_lukasiewicz: Łukasiewicz fuzzy logic
//! - probabilistic: Probabilistic interpretation

use tensorlogic_compiler::{compile_to_einsum_with_config, CompilationConfig};
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_scirs_backend::{Scirs2Exec, Scirs2Tensor};

fn main() {
    let sep = "=".repeat(70);
    println!("{}", sep);
    println!("TensorLogic - Compilation Strategies Comparison");
    println!("{}", sep);
    println!();

    // Create a simple logical expression: P(x) AND Q(x)
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let and_expr = TLExpr::and(p.clone(), q.clone());

    // Test data: 5 different cases
    // P values: [0.0, 0.2, 0.5, 0.8, 1.0]
    // Q values: [1.0, 0.8, 0.5, 0.2, 0.0]
    let p_data = Scirs2Exec::from_vec(vec![0.0, 0.2, 0.5, 0.8, 1.0], vec![5]).unwrap();
    let q_data = Scirs2Exec::from_vec(vec![1.0, 0.8, 0.5, 0.2, 0.0], vec![5]).unwrap();

    println!("Test Data:");
    println!("  P(x): {:?}", p_data.as_slice().unwrap());
    println!("  Q(x): {:?}", q_data.as_slice().unwrap());
    println!();

    // Test each compilation strategy
    test_strategy(
        "Soft Differentiable",
        &and_expr,
        &p_data,
        &q_data,
        CompilationConfig::soft_differentiable(),
    );

    test_strategy(
        "Hard Boolean",
        &and_expr,
        &p_data,
        &q_data,
        CompilationConfig::hard_boolean(),
    );

    test_strategy(
        "Fuzzy Gödel",
        &and_expr,
        &p_data,
        &q_data,
        CompilationConfig::fuzzy_godel(),
    );

    test_strategy(
        "Fuzzy Product",
        &and_expr,
        &p_data,
        &q_data,
        CompilationConfig::fuzzy_product(),
    );

    test_strategy(
        "Fuzzy Łukasiewicz",
        &and_expr,
        &p_data,
        &q_data,
        CompilationConfig::fuzzy_lukasiewicz(),
    );

    test_strategy(
        "Probabilistic",
        &and_expr,
        &p_data,
        &q_data,
        CompilationConfig::probabilistic(),
    );

    println!();
    let sep = "=".repeat(70);
    println!("{}", sep);
    println!("Analysis:");
    println!("{}", sep);
    println!();
    println!("1. Soft Differentiable (Product):");
    println!("   - Smooth, differentiable everywhere");
    println!("   - Good for gradient-based optimization");
    println!("   - Penalizes low values more");
    println!();
    println!("2. Hard Boolean (Min):");
    println!("   - Exact Boolean semantics");
    println!("   - Not differentiable at certain points");
    println!("   - Returns minimum of inputs");
    println!();
    println!("3. Fuzzy Gödel (Min):");
    println!("   - Same as Hard Boolean for AND");
    println!("   - Part of Gödel fuzzy logic system");
    println!();
    println!("4. Fuzzy Product (Product):");
    println!("   - Similar to Soft Differentiable");
    println!("   - Standard product t-norm");
    println!();
    println!("5. Fuzzy Łukasiewicz (max(0, a+b-1)):");
    println!("   - Different from other strategies");
    println!("   - More conservative (lower values)");
    println!("   - Good for certain applications");
    println!();
    println!("6. Probabilistic (Product, assumes independence):");
    println!("   - Treats values as probabilities");
    println!("   - P(A AND B) = P(A) * P(B) if independent");
    println!();
    let sep = "=".repeat(70);
    println!("{}", sep);
    println!();
    println!("Note: This example demonstrates how different compilation strategies");
    println!("affect the semantics of logical operations. Each strategy compiles");
    println!("the same logical expression to different tensor operations.");
    println!();
    let sep = "=".repeat(70);
    println!("{}", sep);
    println!("Example Complete!");
    println!("{}", sep);
}

fn test_strategy(
    name: &str,
    expr: &TLExpr,
    p_data: &Scirs2Tensor,
    q_data: &Scirs2Tensor,
    config: CompilationConfig,
) {
    // Compile with the specified configuration
    let graph = compile_to_einsum_with_config(expr, &config).unwrap();

    let mut executor = Scirs2Exec::new();

    // Add input tensors
    if !graph.tensors.is_empty() {
        executor.add_tensor(graph.tensors[0].clone(), p_data.clone());
    }
    if graph.tensors.len() > 1 {
        executor.add_tensor(graph.tensors[1].clone(), q_data.clone());
    }

    // Execute
    let result = executor.forward(&graph).unwrap();

    println!("{} Strategy:", name);
    println!("  Result: {:?}", result.as_slice().unwrap());
    println!();
}
