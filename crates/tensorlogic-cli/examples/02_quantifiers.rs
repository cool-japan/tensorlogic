//! Example: Quantifiers
//!
//! Demonstrates existential (EXISTS) and universal (FORALL) quantifiers.
//!
//! Run with:
//! ```bash
//! cargo run --example 02_quantifiers
//! ```

use tensorlogic_cli::{parser, CompilationContext};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationConfig};

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic CLI: Quantifier Examples ===\n");

    // Create compilation context with custom domains
    let config = CompilationConfig::soft_differentiable();
    let mut ctx = CompilationContext::with_config(config);
    ctx.add_domain("Person", 100);
    ctx.add_domain("City", 50);
    ctx.add_domain("D", 100);

    // Example 1: Existential quantifier
    println!("Example 1: Existential Quantifier");
    println!("Expression: EXISTS x IN Person. knows(x, alice)");
    println!("Meaning: There exists at least one person who knows alice");
    let expr1 = parser::parse_expression("EXISTS x IN Person. knows(x, alice)")?;
    let mut ctx1 = ctx.clone();
    let graph1 = compile_to_einsum_with_context(&expr1, &mut ctx1)?;
    println!(
        "Compiled: {} tensors, {} nodes",
        graph1.tensors.len(),
        graph1.nodes.len()
    );
    println!("This compiles to a sum reduction over the Person domain\n");

    // Example 2: Universal quantifier
    println!("Example 2: Universal Quantifier");
    println!("Expression: FORALL x IN Person. mortal(x)");
    println!("Meaning: All persons are mortal");
    let expr2 = parser::parse_expression("FORALL x IN Person. mortal(x)")?;
    let mut ctx2 = ctx.clone();
    let graph2 = compile_to_einsum_with_context(&expr2, &mut ctx2)?;
    println!(
        "Compiled: {} tensors, {} nodes",
        graph2.tensors.len(),
        graph2.nodes.len()
    );
    println!("FORALL is dual to EXISTS: FORALL x. P(x) = NOT EXISTS x. NOT P(x)\n");

    // Example 3: Nested quantifiers
    println!("Example 3: Nested Quantifiers");
    println!("Expression: EXISTS x IN Person. FORALL y IN Person. knows(x, y)");
    println!("Meaning: There exists someone who knows everyone");
    let expr3 = parser::parse_expression("EXISTS x IN Person. FORALL y IN Person. knows(x, y)")?;
    let mut ctx3 = ctx.clone();
    let graph3 = compile_to_einsum_with_context(&expr3, &mut ctx3)?;
    println!(
        "Compiled: {} tensors, {} nodes",
        graph3.tensors.len(),
        graph3.nodes.len()
    );
    println!("Nested quantifiers create multiple reduction operations\n");

    // Example 4: Quantifier with conjunction
    println!("Example 4: Quantifier with Conjunction");
    println!("Expression: EXISTS x IN Person. (knows(x, bob) AND likes(x, pizza))");
    println!("Meaning: There exists someone who knows bob AND likes pizza");
    let expr4 =
        parser::parse_expression("EXISTS x IN Person. (knows(x, bob) AND likes(x, pizza))")?;
    let mut ctx4 = ctx.clone();
    let graph4 = compile_to_einsum_with_context(&expr4, &mut ctx4)?;
    println!(
        "Compiled: {} tensors, {} nodes",
        graph4.tensors.len(),
        graph4.nodes.len()
    );
    println!();

    // Example 5: Multiple domains
    println!("Example 5: Multiple Domains");
    println!("Expression: EXISTS p IN Person. EXISTS c IN City. lives_in(p, c)");
    println!("Meaning: There exists a person who lives in some city");
    let expr5 = parser::parse_expression("EXISTS p IN Person. EXISTS c IN City. lives_in(p, c)")?;
    let mut ctx5 = ctx.clone();
    let graph5 = compile_to_einsum_with_context(&expr5, &mut ctx5)?;
    println!(
        "Compiled: {} tensors, {} nodes",
        graph5.tensors.len(),
        graph5.nodes.len()
    );
    println!();

    // Example 6: Quantified implication
    println!("Example 6: Quantified Implication");
    println!("Expression: FORALL x IN Person. (human(x) -> mortal(x))");
    println!("Meaning: All humans are mortal");
    let expr6 = parser::parse_expression("FORALL x IN Person. (human(x) -> mortal(x))")?;
    let mut ctx6 = ctx.clone();
    let graph6 = compile_to_einsum_with_context(&expr6, &mut ctx6)?;
    println!(
        "Compiled: {} tensors, {} nodes",
        graph6.tensors.len(),
        graph6.nodes.len()
    );
    println!();

    println!("=== Quantifier Examples Complete ===");
    Ok(())
}
