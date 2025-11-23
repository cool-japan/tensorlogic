//! Example: Basic Compilation
//!
//! Demonstrates how to use the tensorlogic-cli library to parse and compile
//! logical expressions programmatically.
//!
//! Run with:
//! ```bash
//! cargo run --example 01_basic_compilation
//! ```

use tensorlogic_cli::{parser, CompilationContext};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationConfig};

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic CLI: Basic Compilation Example ===\n");

    // Example 1: Simple predicate
    println!("Example 1: Simple Predicate");
    println!("Expression: knows(alice, bob)");
    let expr1 = parser::parse_expression("knows(alice, bob)")?;
    println!("Parsed: {:?}\n", expr1);

    // Example 2: Logical AND
    println!("Example 2: Logical AND");
    println!("Expression: knows(x, y) AND likes(y, z)");
    let expr2 = parser::parse_expression("knows(x, y) AND likes(y, z)")?;
    println!("Parsed: {:?}\n", expr2);

    // Compile with default configuration
    let config = CompilationConfig::soft_differentiable();
    let mut ctx = CompilationContext::with_config(config);
    ctx.add_domain("D", 100);

    let graph = compile_to_einsum_with_context(&expr2, &mut ctx)?;
    println!("Compiled Graph:");
    println!("  Tensors: {}", graph.tensors.len());
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Inputs: {}", graph.inputs.len());
    println!("  Outputs: {}", graph.outputs.len());
    println!();

    // Example 3: Logical OR
    println!("Example 3: Logical OR");
    println!("Expression: person(x) OR robot(x)");
    let expr3 = parser::parse_expression("person(x) OR robot(x)")?;
    let mut ctx3 = CompilationContext::with_config(CompilationConfig::soft_differentiable());
    ctx3.add_domain("D", 100);
    let graph3 = compile_to_einsum_with_context(&expr3, &mut ctx3)?;
    println!(
        "Compiled: {} tensors, {} nodes\n",
        graph3.tensors.len(),
        graph3.nodes.len()
    );

    // Example 4: Negation
    println!("Example 4: Negation");
    println!("Expression: NOT mortal(x)");
    let expr4 = parser::parse_expression("NOT mortal(x)")?;
    let mut ctx4 = CompilationContext::with_config(CompilationConfig::soft_differentiable());
    ctx4.add_domain("D", 100);
    let graph4 = compile_to_einsum_with_context(&expr4, &mut ctx4)?;
    println!(
        "Compiled: {} tensors, {} nodes\n",
        graph4.tensors.len(),
        graph4.nodes.len()
    );

    // Example 5: Implication
    println!("Example 5: Implication");
    println!("Expression: knows(x, y) -> likes(x, y)");
    let expr5 = parser::parse_expression("knows(x, y) -> likes(x, y)")?;
    let mut ctx5 = CompilationContext::with_config(CompilationConfig::soft_differentiable());
    ctx5.add_domain("D", 100);
    let graph5 = compile_to_einsum_with_context(&expr5, &mut ctx5)?;
    println!(
        "Compiled: {} tensors, {} nodes\n",
        graph5.tensors.len(),
        graph5.nodes.len()
    );

    // Example 6: Complex nested expression
    println!("Example 6: Complex Nested Expression");
    println!("Expression: (knows(x, y) AND likes(y, z)) OR friend(x, z)");
    let expr6 = parser::parse_expression("(knows(x, y) AND likes(y, z)) OR friend(x, z)")?;
    let mut ctx6 = CompilationContext::with_config(CompilationConfig::soft_differentiable());
    ctx6.add_domain("D", 100);
    let graph6 = compile_to_einsum_with_context(&expr6, &mut ctx6)?;
    println!(
        "Compiled: {} tensors, {} nodes\n",
        graph6.tensors.len(),
        graph6.nodes.len()
    );

    println!("=== Compilation Examples Complete ===");
    Ok(())
}
