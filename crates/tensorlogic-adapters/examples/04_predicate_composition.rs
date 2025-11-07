//! Predicate composition example.
//!
//! This example demonstrates how to define predicates in terms of other
//! predicates using composition.

use tensorlogic_adapters::{CompositePredicate, CompositeRegistry, PredicateBody};

fn main() -> anyhow::Result<()> {
    println!("=== Predicate Composition Example ===\n");

    let mut registry = CompositeRegistry::new();

    // Define a simple composite predicate: friend(x, y)
    println!("Defining composite predicates:\n");

    let friend = CompositePredicate::new(
        "friend",
        vec!["x".to_string(), "y".to_string()],
        PredicateBody::And(vec![
            PredicateBody::Reference {
                name: "knows".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
            PredicateBody::Reference {
                name: "trusts".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
        ]),
    )
    .with_description("A friend is someone you know and trust");

    println!("1. friend(x, y) := knows(x, y) AND trusts(x, y)");
    registry.register(friend)?;

    // Define a disjunctive predicate: connected(x, y)
    let connected = CompositePredicate::new(
        "connected",
        vec!["x".to_string(), "y".to_string()],
        PredicateBody::Or(vec![
            PredicateBody::Reference {
                name: "colleague".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
            PredicateBody::Reference {
                name: "friend".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
        ]),
    )
    .with_description("Connected if colleague or friend");

    println!("2. connected(x, y) := colleague(x, y) OR friend(x, y)");
    registry.register(connected)?;

    // Define a negation predicate: not_enemy(x, y)
    let not_enemy = CompositePredicate::new(
        "not_enemy",
        vec!["x".to_string(), "y".to_string()],
        PredicateBody::Not(Box::new(PredicateBody::Reference {
            name: "enemy".to_string(),
            args: vec!["x".to_string(), "y".to_string()],
        })),
    )
    .with_description("Not an enemy");

    println!("3. not_enemy(x, y) := NOT enemy(x, y)\n");
    registry.register(not_enemy)?;

    // List all registered predicates
    println!("Registered predicates:");
    for name in registry.list_predicates() {
        if let Some(pred) = registry.get(&name) {
            println!("  - {} (arity: {})", pred.name, pred.arity());
            if let Some(desc) = &pred.description {
                println!("    Description: {}", desc);
            }
        }
    }
    println!();

    // Expand predicates with concrete arguments
    println!("Expanding predicates:\n");

    println!("Expanding friend(alice, bob):");
    let expanded = registry.expand("friend", &["alice".to_string(), "bob".to_string()])?;
    println!("  Result: {:?}\n", expanded);

    println!("Expanding connected(charlie, david):");
    let expanded = registry.expand("connected", &["charlie".to_string(), "david".to_string()])?;
    println!("  Result: {:?}\n", expanded);

    // Validate composite predicates
    println!("Validation:");
    for name in registry.list_predicates() {
        if let Some(pred) = registry.get(&name) {
            match pred.validate() {
                Ok(_) => println!("  ✓ {} is well-formed", pred.name),
                Err(e) => println!("  ✗ {} has errors: {}", pred.name, e),
            }
        }
    }

    Ok(())
}
