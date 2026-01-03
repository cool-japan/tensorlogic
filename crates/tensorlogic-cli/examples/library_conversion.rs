//! Example demonstrating format conversion utilities
//!
//! This example shows:
//! - Converting expressions between different formats
//! - Pretty-printing expressions
//! - JSON and YAML serialization
//! - Expression normalization
//!
//! Run with: cargo run --example library_conversion

use tensorlogic_cli::{conversion, parser};

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic Library Mode - Format Conversion Example ===\n");

    // 1. Parse an expression
    println!("1. Original expression:");
    let expression = "EXISTS x IN Person. (knows(x, Alice) AND (smart(x) OR creative(x)))";
    println!("   {}\n", expression);

    let expr = parser::parse_expression(expression)?;

    // 2. Format expression (non-pretty)
    println!("2. Compact format:");
    let compact = conversion::format_expression(&expr, false);
    println!("   {}\n", compact);

    // 3. Format expression (pretty)
    println!("3. Pretty format:");
    let pretty = conversion::format_expression(&expr, true);
    println!("{}\n", pretty);

    // 4. Convert to JSON
    println!("4. JSON serialization:");
    let json = serde_json::to_string_pretty(&expr)?;
    println!("{}\n", json);

    // 5. Convert to YAML
    println!("5. YAML serialization:");
    let yaml = serde_yaml::to_string(&expr)?;
    println!("{}", yaml);

    // 6. Round-trip test
    println!("6. Round-trip conversion test:");
    println!("   Original -> JSON -> Expression -> Pretty");

    // Expression to JSON
    let json_str = serde_json::to_string(&expr)?;
    println!("   ✓ Serialized to JSON ({} bytes)", json_str.len());

    // JSON back to expression
    let expr_from_json: tensorlogic_cli::TLExpr = serde_json::from_str(&json_str)?;
    println!("   ✓ Deserialized from JSON");

    // Format pretty
    let final_pretty = conversion::format_expression(&expr_from_json, true);
    println!("   ✓ Formatted:\n{}", final_pretty);

    // 7. Demonstrate various expression types
    println!("\n7. Formatting different expression types:");

    let examples = vec![
        ("Predicate", "person(Alice, age, 30)"),
        ("Conjunction", "happy(x) AND healthy(x)"),
        ("Disjunction", "smart(x) OR creative(x) OR hardworking(x)"),
        ("Negation", "NOT lazy(x)"),
        ("Implication", "student(x) -> studies(x)"),
        ("Quantifier", "FORALL x IN Student. studies(x)"),
        ("Arithmetic", "age(x) + 5 > 18"),
        ("Conditional", "IF rich(x) THEN happy(x) ELSE sad(x)"),
    ];

    for (name, expr_str) in examples {
        match parser::parse_expression(expr_str) {
            Ok(e) => {
                let formatted = conversion::format_expression(&e, false);
                println!("   {}: {}", name, formatted);
            }
            Err(e) => {
                println!("   {}: Error - {}", name, e);
            }
        }
    }

    println!("\n=== Format conversion example completed successfully! ===");

    Ok(())
}
