//! Example 10: Modal and Temporal Logic Compilation
//!
//! This example demonstrates the compilation of modal and temporal logic operators,
//! showcasing how TensorLogic can reason about possibility, necessity, and temporal sequences.
//!
//! # Modal Logic
//!
//! Modal logic extends classical logic with operators for reasoning about "possible worlds":
//! - Box (□): Necessity - "P is true in all possible worlds"
//! - Diamond (◇): Possibility - "P is true in at least one possible world"
//!
//! # Temporal Logic (LTL)
//!
//! Temporal logic extends classical logic with operators for reasoning about time:
//! - Eventually (F): "P will be true in some future state"
//! - Always (G): "P is true in all future states"
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example 10_modal_temporal_logic
//! ```

use tensorlogic_compiler::{
    compile_to_einsum_with_context, CompilationConfig, CompilerContext, ModalStrategy,
    TemporalStrategy,
};
use tensorlogic_ir::{OpType, TLExpr, Term};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Modal and Temporal Logic Compilation Example             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Example 1: Modal Logic - Necessity (Box □)
    modal_necessity_example();

    // Example 2: Modal Logic - Possibility (Diamond ◇)
    modal_possibility_example();

    // Example 3: Temporal Logic - Eventually (F)
    temporal_eventually_example();

    // Example 4: Temporal Logic - Always (G)
    temporal_always_example();

    // Example 5: Combined Modal and Temporal
    combined_modal_temporal_example();

    // Example 6: Different Compilation Strategies
    strategy_comparison_example();

    println!("\n✅ All examples completed successfully!");
}

/// Example 1: Modal Necessity - "All possible worlds satisfy a condition"
fn modal_necessity_example() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 1: Modal Necessity (Box □)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 50);

    // □(happy(x)) - "In all possible worlds, x is happy"
    // This compiles to: reduce over world axis using min/product
    let expr = TLExpr::Box(Box::new(TLExpr::pred("happy", vec![Term::var("x")])));

    println!("Expression: □(happy(x))");
    println!("Meaning: In all possible worlds, person x is happy");
    println!("Tensor operation: min/product reduction over world axis\n");

    match compile_to_einsum_with_context(&expr, &mut ctx) {
        Ok(graph) => {
            println!("✓ Compilation successful!");
            println!("  Graph nodes: {}", graph.nodes.len());
            println!("  Graph tensors: {}", graph.tensors.len());
            println!(
                "  World axis created: {}",
                ctx.domains.contains_key("__world__")
            );
            println!("  Modal strategy: {:?}", ctx.config.modal_strategy);
        }
        Err(e) => println!("✗ Compilation failed: {}", e),
    }

    println!();
}

/// Example 2: Modal Possibility - "At least one possible world satisfies a condition"
fn modal_possibility_example() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 2: Modal Possibility (Diamond ◇)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 50);

    // ◇(∃y. knows(x, y)) - "It's possible that x knows someone"
    // Outer: possibility (max/sum over worlds)
    // Inner: existential quantifier (sum over people)
    let expr = TLExpr::Diamond(Box::new(TLExpr::exists(
        "y",
        "Person",
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
    )));

    println!("Expression: ◇(∃y. knows(x, y))");
    println!("Meaning: It's possible that person x knows someone");
    println!("Tensor operation: max/sum reduction over world axis, then sum over y\n");

    match compile_to_einsum_with_context(&expr, &mut ctx) {
        Ok(graph) => {
            println!("✓ Compilation successful!");
            println!("  Graph nodes: {}", graph.nodes.len());
            println!(
                "  World axis created: {}",
                ctx.domains.contains_key("__world__")
            );

            // Show the world axis details
            if let Some(domain) = ctx.domains.get("__world__") {
                println!("  World domain size: {}", domain.cardinality);
            }
        }
        Err(e) => println!("✗ Compilation failed: {}", e),
    }

    println!();
}

/// Example 3: Temporal Eventually - "Something will happen in the future"
fn temporal_eventually_example() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 3: Temporal Eventually (F)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Task", 20);

    // F(completed(t)) - "Task t will eventually be completed"
    let expr = TLExpr::Eventually(Box::new(TLExpr::pred("completed", vec![Term::var("t")])));

    println!("Expression: F(completed(t))");
    println!("Meaning: Task t will eventually be completed");
    println!("Tensor operation: max/sum reduction over future time steps\n");

    match compile_to_einsum_with_context(&expr, &mut ctx) {
        Ok(graph) => {
            println!("✓ Compilation successful!");
            println!("  Graph nodes: {}", graph.nodes.len());
            println!(
                "  Time axis created: {}",
                ctx.domains.contains_key("__time__")
            );
            println!("  Temporal strategy: {:?}", ctx.config.temporal_strategy);

            // Show the time axis details
            if let Some(domain) = ctx.domains.get("__time__") {
                println!("  Time domain size: {}", domain.cardinality);
            }
        }
        Err(e) => println!("✗ Compilation failed: {}", e),
    }

    println!();
}

/// Example 4: Temporal Always - "Something is always true"
fn temporal_always_example() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 4: Temporal Always (G)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut ctx = CompilerContext::new();
    ctx.add_domain("System", 10);

    // G(safe(s)) - "System s is always safe"
    // This compiles to: reduce over time axis using min/product
    let expr = TLExpr::Always(Box::new(TLExpr::pred("safe", vec![Term::var("s")])));

    println!("Expression: G(safe(s))");
    println!("Meaning: System s is always safe (in all future states)");
    println!("Tensor operation: min/product reduction over time axis\n");

    match compile_to_einsum_with_context(&expr, &mut ctx) {
        Ok(graph) => {
            println!("✓ Compilation successful!");
            println!("  Graph nodes: {}", graph.nodes.len());
            println!(
                "  Time axis created: {}",
                ctx.domains.contains_key("__time__")
            );
        }
        Err(e) => println!("✗ Compilation failed: {}", e),
    }

    println!();
}

/// Example 5: Combined Modal and Temporal Logic
fn combined_modal_temporal_example() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 5: Combined Modal and Temporal Logic");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Agent", 15);

    // □(F(goal_achieved(a))) - "In all possible worlds, agent a will eventually achieve its goal"
    // Outer: necessity over worlds
    // Inner: eventually over time
    let expr = TLExpr::Box(Box::new(TLExpr::Eventually(Box::new(TLExpr::pred(
        "goal_achieved",
        vec![Term::var("a")],
    )))));

    println!("Expression: □(F(goal_achieved(a)))");
    println!("Meaning: In all possible worlds, agent a will eventually achieve its goal");
    println!("This combines:");
    println!("  - Modal reasoning: necessity across possible worlds");
    println!("  - Temporal reasoning: eventually in the future");
    println!("Tensor operations:");
    println!("  1. Max/sum reduction over time (Eventually)");
    println!("  2. Min/product reduction over worlds (Box)\n");

    match compile_to_einsum_with_context(&expr, &mut ctx) {
        Ok(graph) => {
            println!("✓ Compilation successful!");
            println!("  Graph nodes: {}", graph.nodes.len());
            println!("  Both world and time axes created:");
            println!(
                "    - World axis: {}",
                ctx.domains.contains_key("__world__")
            );
            println!("    - Time axis: {}", ctx.domains.contains_key("__time__"));

            // Count reduction operations
            let reduction_count = graph
                .nodes
                .iter()
                .filter(|node| {
                    matches!(&node.op,
                    OpType::Einsum { spec } if
                        spec.contains("min(") || spec.contains("max(") ||
                        spec.contains("sum(") || spec.contains("prod("))
                })
                .count();
            println!("  Reduction operations: {}", reduction_count);
        }
        Err(e) => println!("✗ Compilation failed: {}", e),
    }

    println!();
}

/// Example 6: Comparing Different Compilation Strategies
fn strategy_comparison_example() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 6: Strategy Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let expr = TLExpr::Diamond(Box::new(TLExpr::Eventually(Box::new(TLExpr::pred(
        "event_occurs",
        vec![Term::var("e")],
    )))));

    println!("Expression: ◇(F(event_occurs(e)))");
    println!("Meaning: It's possible that event e will eventually occur\n");

    // Strategy 1: Hard Boolean (min/max)
    println!("─────────────────────────────────────────────────────────────");
    println!("Strategy 1: Hard Boolean (crisp logic)");
    println!("─────────────────────────────────────────────────────────────");
    let mut ctx1 = CompilerContext::with_config(CompilationConfig::hard_boolean());
    ctx1.add_domain("Event", 10);

    match compile_to_einsum_with_context(&expr, &mut ctx1) {
        Ok(graph) => {
            println!("✓ Modal strategy: {:?}", ctx1.config.modal_strategy);
            println!("✓ Temporal strategy: {:?}", ctx1.config.temporal_strategy);
            println!("✓ Operations: max (Eventually) + max (Diamond)");
            println!("✓ Graph nodes: {}", graph.nodes.len());
        }
        Err(e) => println!("✗ Failed: {}", e),
    }

    println!();

    // Strategy 2: Soft Differentiable (product/sum)
    println!("─────────────────────────────────────────────────────────────");
    println!("Strategy 2: Soft Differentiable (neural-friendly)");
    println!("─────────────────────────────────────────────────────────────");
    let mut ctx2 = CompilerContext::with_config(CompilationConfig::soft_differentiable());
    ctx2.add_domain("Event", 10);

    match compile_to_einsum_with_context(&expr, &mut ctx2) {
        Ok(graph) => {
            println!("✓ Modal strategy: {:?}", ctx2.config.modal_strategy);
            println!("✓ Temporal strategy: {:?}", ctx2.config.temporal_strategy);
            println!("✓ Operations: sum (Eventually) + sum (Diamond)");
            println!("✓ Graph nodes: {}", graph.nodes.len());
        }
        Err(e) => println!("✗ Failed: {}", e),
    }

    println!();

    // Strategy 3: Custom Configuration
    println!("─────────────────────────────────────────────────────────────");
    println!("Strategy 3: Custom (threshold-based)");
    println!("─────────────────────────────────────────────────────────────");
    let config = CompilationConfig::custom()
        .modal_strategy(ModalStrategy::Threshold { threshold: 0.7 })
        .temporal_strategy(TemporalStrategy::Max)
        .build();

    let mut ctx3 = CompilerContext::with_config(config);
    ctx3.add_domain("Event", 10);

    match compile_to_einsum_with_context(&expr, &mut ctx3) {
        Ok(graph) => {
            println!("✓ Modal strategy: {:?}", ctx3.config.modal_strategy);
            println!("✓ Temporal strategy: {:?}", ctx3.config.temporal_strategy);
            println!("✓ Operations: threshold-based satisfaction checking");
            println!("✓ Graph nodes: {}", graph.nodes.len());
        }
        Err(e) => println!("✗ Failed: {}", e),
    }

    println!();
}
