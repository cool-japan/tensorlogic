//! Example: Fast predicate lookup with SignatureMatcher
//!
//! This example demonstrates how to use the SignatureMatcher for efficient
//! predicate lookups by arity, signature, and domain patterns.

use tensorlogic_adapters::{DomainInfo, PredicateInfo, SignatureMatcher, SymbolTable};

fn main() {
    println!("=== SignatureMatcher Example ===\n");

    // Create a symbol table with various predicates
    let mut table = SymbolTable::new();

    // Add domains
    println!("Adding domains...");
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table.add_domain(DomainInfo::new("Location", 50)).unwrap();
    table.add_domain(DomainInfo::new("Object", 200)).unwrap();
    table.add_domain(DomainInfo::new("Time", 1000)).unwrap();

    // Add predicates with different arities
    println!("Adding predicates...\n");

    // Unary predicates (arity 1)
    table
        .add_predicate(PredicateInfo::new("person", vec!["Person".to_string()]))
        .unwrap();

    table
        .add_predicate(PredicateInfo::new("location", vec!["Location".to_string()]))
        .unwrap();

    // Binary predicates (arity 2)
    table
        .add_predicate(PredicateInfo::new(
            "knows",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();

    table
        .add_predicate(PredicateInfo::new(
            "at",
            vec!["Person".to_string(), "Location".to_string()],
        ))
        .unwrap();

    table
        .add_predicate(PredicateInfo::new(
            "has",
            vec!["Person".to_string(), "Object".to_string()],
        ))
        .unwrap();

    // Ternary predicates (arity 3)
    table
        .add_predicate(PredicateInfo::new(
            "gave",
            vec![
                "Person".to_string(),
                "Object".to_string(),
                "Person".to_string(),
            ],
        ))
        .unwrap();

    table
        .add_predicate(PredicateInfo::new(
            "met",
            vec![
                "Person".to_string(),
                "Person".to_string(),
                "Location".to_string(),
            ],
        ))
        .unwrap();

    // Quaternary predicates (arity 4)
    table
        .add_predicate(PredicateInfo::new(
            "transaction",
            vec![
                "Person".to_string(),
                "Object".to_string(),
                "Person".to_string(),
                "Time".to_string(),
            ],
        ))
        .unwrap();

    // Build the signature matcher
    println!("Building SignatureMatcher...");
    let matcher = SignatureMatcher::from_predicates(table.predicates.values());

    let stats = matcher.stats();
    println!("✓ Indexed {} predicates", stats.total_predicates);
    println!("  - {} unique arities", stats.unique_arities);
    println!("  - {} unique signatures", stats.unique_signatures);
    println!(
        "  - Avg predicates per signature: {:.2}\n",
        stats.avg_index_size()
    );

    // Example 1: Find predicates by arity
    println!("=== Example 1: Find by Arity ===");
    for arity in 1..=4 {
        let predicates = matcher.find_by_arity(arity);
        if !predicates.is_empty() {
            println!("Arity {}: {:?}", arity, predicates);
        }
    }
    println!();

    // Example 2: Find predicates by exact signature
    println!("=== Example 2: Find by Exact Signature ===");
    let signature = vec!["Person".to_string(), "Location".to_string()];
    let matches = matcher.find_by_signature(&signature);
    println!("Signature {:?}: {:?}", signature, matches);

    let signature2 = vec!["Person".to_string(), "Person".to_string()];
    let matches2 = matcher.find_by_signature(&signature2);
    println!("Signature {:?}: {:?}", signature2, matches2);
    println!();

    // Example 3: Get predicate details
    println!("=== Example 3: Predicate Details ===");
    if let Some(pred) = matcher.get_predicate("gave") {
        println!("Predicate: {}", pred.name);
        println!("  Arity: {}", pred.arg_domains.len());
        println!("  Signature: {:?}", pred.arg_domains);
        if let Some(desc) = &pred.description {
            println!("  Description: {}", desc);
        }
    }
    println!();

    // Example 4: Performance comparison
    println!("=== Example 4: Performance Benefits ===");
    println!("SignatureMatcher provides O(1) lookups instead of O(n) scans:");
    println!("  - Direct hash table access for arity lookups");
    println!("  - Pre-computed signature indices");
    println!("  - Multiple index strategies for different query patterns");
    println!();

    // Example 5: Use case - type checking
    println!("=== Example 5: Type Checking Use Case ===");
    println!("When compiling expressions, quickly check if a predicate exists:");

    let required_signature = vec!["Person".to_string(), "Object".to_string()];
    let available = matcher.find_by_signature(&required_signature);

    if !available.is_empty() {
        println!(
            "✓ Found predicates with signature {:?}: {:?}",
            required_signature, available
        );
    } else {
        println!(
            "✗ No predicates found with signature {:?}",
            required_signature
        );
    }
    println!();

    // Example 6: Dynamic predicate discovery
    println!("=== Example 6: Discover All Binary Relations ===");
    let binary_predicates = matcher.find_by_arity(2);
    println!("Found {} binary predicates:", binary_predicates.len());
    for pred_name in binary_predicates {
        if let Some(pred) = matcher.get_predicate(&pred_name) {
            println!("  - {}: {:?}", pred.name, pred.arg_domains);
        }
    }

    println!("\n=== Example Complete ===");
}
