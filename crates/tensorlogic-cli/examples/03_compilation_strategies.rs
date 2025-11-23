//! Example: Compilation Strategies
//!
//! Demonstrates different compilation strategies and their effects.
//!
//! Run with:
//! ```bash
//! cargo run --example 03_compilation_strategies
//! ```

use tensorlogic_cli::{parser, CompilationContext};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationConfig};

fn compile_with_strategy(
    expr_str: &str,
    config: CompilationConfig,
    strategy_name: &str,
) -> anyhow::Result<()> {
    let expr = parser::parse_expression(expr_str)?;
    let mut ctx = CompilationContext::with_config(config);
    ctx.add_domain("D", 100);

    let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;

    println!("  Strategy: {}", strategy_name);
    println!(
        "  Tensors: {}, Nodes: {}",
        graph.tensors.len(),
        graph.nodes.len()
    );
    println!();

    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic CLI: Compilation Strategies ===\n");

    let expr_str = "p(x) AND q(y)";
    println!("Expression: {}\n", expr_str);

    // Strategy 1: Soft Differentiable (default)
    println!("Strategy 1: Soft Differentiable");
    println!("Best for: Neural network training with smooth gradients");
    println!("AND: Element-wise product");
    println!("OR: Probabilistic sum (1 - (1-a)(1-b))");
    println!("NOT: Complement (1 - x)");
    compile_with_strategy(
        expr_str,
        CompilationConfig::soft_differentiable(),
        "soft_differentiable",
    )?;

    // Strategy 2: Hard Boolean
    println!("Strategy 2: Hard Boolean");
    println!("Best for: Discrete Boolean logic");
    println!("AND: Minimum (min)");
    println!("OR: Maximum (max)");
    println!("NOT: Complement (1 - x)");
    compile_with_strategy(expr_str, CompilationConfig::hard_boolean(), "hard_boolean")?;

    // Strategy 3: Fuzzy Gödel
    println!("Strategy 3: Fuzzy Gödel");
    println!("Best for: Gödel fuzzy logic (min/max operations)");
    println!("AND: Minimum (min)");
    println!("OR: Maximum (max)");
    println!("NOT: Complement (1 - x)");
    compile_with_strategy(expr_str, CompilationConfig::fuzzy_godel(), "fuzzy_godel")?;

    // Strategy 4: Fuzzy Product
    println!("Strategy 4: Fuzzy Product");
    println!("Best for: Product fuzzy logic (probabilistic)");
    println!("AND: Product (a * b)");
    println!("OR: Probabilistic sum");
    println!("NOT: Complement (1 - x)");
    compile_with_strategy(
        expr_str,
        CompilationConfig::fuzzy_product(),
        "fuzzy_product",
    )?;

    // Strategy 5: Fuzzy Łukasiewicz
    println!("Strategy 5: Fuzzy Łukasiewicz");
    println!("Best for: Łukasiewicz fuzzy logic (bounded)");
    println!("AND: max(0, a + b - 1)");
    println!("OR: min(1, a + b)");
    println!("NOT: Complement (1 - x)");
    compile_with_strategy(
        expr_str,
        CompilationConfig::fuzzy_lukasiewicz(),
        "fuzzy_lukasiewicz",
    )?;

    // Strategy 6: Probabilistic
    println!("Strategy 6: Probabilistic");
    println!("Best for: Probabilistic interpretation");
    println!("AND: Product (independent events)");
    println!("OR: Probabilistic sum");
    println!("NOT: Complement (1 - x)");
    compile_with_strategy(
        expr_str,
        CompilationConfig::probabilistic(),
        "probabilistic",
    )?;

    // Demonstrate strategy effects on complex expressions
    println!("=== Complex Expression Comparison ===\n");
    let complex_expr = "p(x) AND q(y) OR r(z)";
    println!("Expression: {}\n", complex_expr);

    println!("With soft_differentiable:");
    compile_with_strategy(
        complex_expr,
        CompilationConfig::soft_differentiable(),
        "soft_differentiable",
    )?;

    println!("With hard_boolean:");
    compile_with_strategy(
        complex_expr,
        CompilationConfig::hard_boolean(),
        "hard_boolean",
    )?;

    println!("=== Strategy Examples Complete ===");
    Ok(())
}
