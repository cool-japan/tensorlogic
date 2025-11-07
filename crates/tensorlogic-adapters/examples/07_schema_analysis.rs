//! Example: Schema analysis and recommendations
//!
//! This example demonstrates how to use SchemaAnalyzer and SchemaStatistics
//! to analyze schemas, detect issues, and get recommendations.

use tensorlogic_adapters::{
    DomainInfo, PredicateInfo, SchemaAnalyzer, SchemaStatistics, SymbolTable,
};

fn main() {
    println!("=== Schema Analysis Example ===\n");

    // Example 1: Analyze a well-designed schema
    println!("=== Example 1: Well-Designed Schema ===");
    analyze_good_schema();
    println!();

    // Example 2: Analyze a schema with issues
    println!("=== Example 2: Schema with Issues ===");
    analyze_problematic_schema();
    println!();

    // Example 3: Compare schemas
    println!("=== Example 3: Schema Comparison ===");
    compare_schemas();
}

fn analyze_good_schema() {
    let mut table = SymbolTable::new();

    // Well-balanced domains
    table
        .add_domain(DomainInfo::new("Person", 100).with_description("Human entities"))
        .unwrap();
    table
        .add_domain(DomainInfo::new("Location", 50).with_description("Physical places"))
        .unwrap();
    table
        .add_domain(DomainInfo::new("Event", 200).with_description("Temporal events"))
        .unwrap();

    // Appropriate predicates
    table
        .add_predicate(PredicateInfo::new(
            "attends",
            vec!["Person".to_string(), "Event".to_string()],
        ))
        .unwrap();

    table
        .add_predicate(PredicateInfo::new(
            "held_at",
            vec!["Event".to_string(), "Location".to_string()],
        ))
        .unwrap();

    table
        .add_predicate(PredicateInfo::new(
            "knows",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();

    // Compute statistics
    let stats = SchemaStatistics::compute(&table);
    println!("Schema Statistics:");
    println!("  Domains: {}", stats.domain_count);
    println!("  Predicates: {}", stats.predicate_count);
    println!("  Total cardinality: {}", stats.total_cardinality);
    println!("  Avg cardinality: {:.2}", stats.avg_cardinality);
    println!("  Complexity score: {:.2}", stats.complexity_score());
    println!();

    // Domain usage
    println!("Domain Usage:");
    for (domain, count) in stats.most_used_domains(10) {
        println!("  {} (used {} times)", domain, count);
    }
    println!();

    // Analyze for issues
    let recommendations = SchemaAnalyzer::analyze(&table);
    if recommendations.issues.is_empty() {
        println!("✓ No issues detected!");
    } else {
        println!("Issues found:");
        for issue in &recommendations.issues {
            println!(
                "  [{}] {}",
                match issue.severity() {
                    1 => "INFO",
                    2 => "WARN",
                    3 => "ERROR",
                    _ => "UNKNOWN",
                },
                issue.description()
            );
        }
    }

    if !recommendations.suggestions.is_empty() {
        println!("\nSuggestions:");
        for suggestion in &recommendations.suggestions {
            println!("  • {}", suggestion);
        }
    }
}

fn analyze_problematic_schema() {
    let mut table = SymbolTable::new();

    // Issue 1: Zero cardinality domain
    table.add_domain(DomainInfo::new("EmptyDomain", 0)).unwrap();

    // Issue 2: Very high cardinality
    table
        .add_domain(DomainInfo::new("HugeDomain", 1_000_000))
        .unwrap();

    // Issue 3: Unused domain
    table
        .add_domain(DomainInfo::new("UnusedDomain", 50))
        .unwrap();

    // Issue 4: High arity predicate
    table.add_domain(DomainInfo::new("A", 10)).unwrap();
    table.add_domain(DomainInfo::new("B", 10)).unwrap();
    table.add_domain(DomainInfo::new("C", 10)).unwrap();

    table
        .add_predicate(PredicateInfo::new(
            "complex_relation",
            vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
            ],
        ))
        .unwrap();

    // Normal predicate
    table
        .add_predicate(PredicateInfo::new(
            "normal",
            vec!["A".to_string(), "B".to_string()],
        ))
        .unwrap();

    // Compute statistics
    let stats = SchemaStatistics::compute(&table);
    println!("Schema Statistics:");
    println!("  Domains: {}", stats.domain_count);
    println!("  Predicates: {}", stats.predicate_count);
    println!("  Max cardinality: {}", stats.max_cardinality);
    println!("  Min cardinality: {}", stats.min_cardinality);
    println!("  Complexity score: {:.2}", stats.complexity_score());
    println!();

    // Unused domains
    if !stats.unused_domains.is_empty() {
        println!("Unused domains: {:?}", stats.unused_domains);
        println!();
    }

    // Analyze for issues
    let recommendations = SchemaAnalyzer::analyze(&table);
    println!("Issues detected: {}", recommendations.issues.len());
    for issue in &recommendations.issues {
        let severity_label = match issue.severity() {
            1 => "INFO",
            2 => "WARN",
            3 => "ERROR",
            _ => "UNKNOWN",
        };
        println!("  [{}] {}", severity_label, issue.description());
    }

    println!();
    println!("Suggestions:");
    for suggestion in &recommendations.suggestions {
        println!("  • {}", suggestion);
    }
}

fn compare_schemas() {
    // Create two schemas with different characteristics
    let mut schema1 = SymbolTable::new();
    schema1.add_domain(DomainInfo::new("Person", 100)).unwrap();
    schema1.add_domain(DomainInfo::new("Location", 50)).unwrap();
    schema1
        .add_predicate(PredicateInfo::new(
            "at",
            vec!["Person".to_string(), "Location".to_string()],
        ))
        .unwrap();

    let mut schema2 = SymbolTable::new();
    schema2.add_domain(DomainInfo::new("Entity", 500)).unwrap();
    schema2
        .add_domain(DomainInfo::new("Relation", 1000))
        .unwrap();
    schema2
        .add_domain(DomainInfo::new("Attribute", 100))
        .unwrap();
    schema2
        .add_predicate(PredicateInfo::new(
            "has_relation",
            vec![
                "Entity".to_string(),
                "Relation".to_string(),
                "Entity".to_string(),
            ],
        ))
        .unwrap();
    schema2
        .add_predicate(PredicateInfo::new(
            "has_attribute",
            vec!["Entity".to_string(), "Attribute".to_string()],
        ))
        .unwrap();

    let stats1 = SchemaStatistics::compute(&schema1);
    let stats2 = SchemaStatistics::compute(&schema2);

    println!("Schema 1:");
    println!(
        "  Domains: {}, Predicates: {}",
        stats1.domain_count, stats1.predicate_count
    );
    println!("  Total cardinality: {}", stats1.total_cardinality);
    println!("  Complexity: {:.2}", stats1.complexity_score());
    println!();

    println!("Schema 2:");
    println!(
        "  Domains: {}, Predicates: {}",
        stats2.domain_count, stats2.predicate_count
    );
    println!("  Total cardinality: {}", stats2.total_cardinality);
    println!("  Complexity: {:.2}", stats2.complexity_score());
    println!();

    println!("Comparison:");
    println!(
        "  Schema 2 has {} more domains",
        stats2.domain_count - stats1.domain_count
    );
    println!(
        "  Schema 2 has {}x higher cardinality",
        stats2.total_cardinality / stats1.total_cardinality.max(1)
    );
    println!(
        "  Schema 2 is {:.1}x more complex",
        stats2.complexity_score() / stats1.complexity_score().max(0.1)
    );

    println!("\n=== Example Complete ===");
}
