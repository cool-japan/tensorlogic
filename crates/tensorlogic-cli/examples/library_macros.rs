//! Example demonstrating macro system integration
//!
//! This example shows:
//! - Defining custom macros programmatically
//! - Using built-in macros
//! - Expanding macros in expressions
//! - Compiling macro-expanded expressions
//!
//! Run with: cargo run --example library_macros

use tensorlogic_cli::{
    macros::{MacroDef, MacroRegistry},
    parser, CompilationContext,
};
use tensorlogic_compiler::compile_to_einsum_with_context;

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic Library Mode - Macros Example ===\n");

    // 1. Create macro registry with built-ins
    println!("1. Setting up macro registry...");
    let mut macros = MacroRegistry::with_builtins();
    println!("   Loaded {} built-in macros", macros.len());

    // List built-in macros
    println!("\n   Built-in macros:");
    for macro_def in macros.list() {
        println!(
            "     - {}({}) = {}",
            macro_def.name,
            macro_def.params.join(", "),
            macro_def.body
        );
    }

    // 2. Define custom macros
    println!("\n2. Defining custom macros...");

    // Define a "friend of friend" macro
    let fof_macro = MacroDef::new(
        "friendOfFriend".to_string(),
        vec!["x".to_string(), "z".to_string()],
        "EXISTS y. (friend(x, y) AND friend(y, z))".to_string(),
    );
    macros.define(fof_macro)?;
    println!("   Defined: friendOfFriend(x, z)");

    // Define an "acquaintance" macro using existing macros
    let acq_macro = MacroDef::new(
        "acquaintance".to_string(),
        vec!["x".to_string(), "y".to_string()],
        "friend(x, y) OR friendOfFriend(x, y)".to_string(),
    );
    macros.define(acq_macro)?;
    println!("   Defined: acquaintance(x, y)");

    // 3. Use macros in expressions
    println!("\n3. Expanding macros in expressions...");

    let expression = "acquaintance(Alice, Bob)";
    println!("   Original: {}", expression);

    let expanded = macros.expand_all(expression)?;
    println!("   Expanded: {}", expanded);

    // 4. Compile the expanded expression
    println!("\n4. Compiling expanded expression...");

    let expr = parser::parse_expression(&expanded)?;
    println!("   Parsed successfully");

    let mut context = CompilationContext::new();
    context.add_domain("Person", 50);

    let graph = compile_to_einsum_with_context(&expr, &mut context)?;
    println!("   Compiled successfully!");
    println!(
        "   Graph has {} tensors and {} nodes",
        graph.tensors.len(),
        graph.nodes.len()
    );

    // 5. Demonstrate transitive closure using built-in macro
    println!("\n5. Using built-in 'transitive' macro...");

    let transitive_expr = "transitive(knows, Alice, Charlie)";
    println!("   Expression: {}", transitive_expr);

    let trans_expanded = macros.expand_all(transitive_expr)?;
    println!("   Expanded: {}", trans_expanded);

    let trans_expr = parser::parse_expression(&trans_expanded)?;
    let trans_graph = compile_to_einsum_with_context(&trans_expr, &mut context)?;
    println!(
        "   Compiled: {} tensors, {} nodes",
        trans_graph.tensors.len(),
        trans_graph.nodes.len()
    );

    println!("\n=== Macros example completed successfully! ===");

    Ok(())
}
