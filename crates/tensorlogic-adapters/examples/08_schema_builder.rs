//! Example: Fluent schema construction with SchemaBuilder
//!
//! This example demonstrates how to use SchemaBuilder for easy
//! programmatic schema construction with a fluent API.

use tensorlogic_adapters::{Metadata, Provenance, SchemaBuilder};

fn main() {
    println!("=== SchemaBuilder Example ===\n");

    // Example 1: Basic schema construction
    println!("=== Example 1: Basic Schema ===");
    basic_schema();
    println!();

    // Example 2: Schema with metadata
    println!("=== Example 2: Schema with Metadata ===");
    schema_with_metadata();
    println!();

    // Example 3: Complex knowledge graph schema
    println!("=== Example 3: Knowledge Graph Schema ===");
    knowledge_graph_schema();
    println!();

    // Example 4: Validated schema
    println!("=== Example 4: Schema Validation ===");
    validated_schema();
}

fn basic_schema() {
    let table = SchemaBuilder::new()
        .domain("Person", 100)
        .domain("Location", 50)
        .domain("Event", 200)
        .predicate("attends", vec!["Person", "Event"])
        .predicate("held_at", vec!["Event", "Location"])
        .predicate("knows", vec!["Person", "Person"])
        .variable("x", "Person")
        .variable("y", "Event")
        .build()
        .unwrap();

    println!("✓ Created schema with:");
    println!("  {} domains", table.domains.len());
    println!("  {} predicates", table.predicates.len());
    println!("  {} variables", table.variables.len());

    println!("\nDomains:");
    for (name, domain) in &table.domains {
        println!("  - {}: cardinality {}", name, domain.cardinality);
    }

    println!("\nPredicates:");
    for (name, pred) in &table.predicates {
        println!("  - {}: {:?}", name, pred.arg_domains);
    }
}

fn schema_with_metadata() {
    // Create metadata for domains
    let mut person_meta = Metadata::new();
    person_meta.add_tag("core");
    person_meta.add_tag("entity");
    person_meta.set_attribute("namespace", "org.example");
    person_meta.provenance =
        Some(Provenance::new("Alice", "2025-01-15T10:30:00Z").with_source("schema.yaml", Some(42)));

    let table = SchemaBuilder::new()
        .domain_with_metadata("Person", 100, person_meta)
        .domain_with_desc("Location", 50, "Physical places")
        .domain_with_desc("Organization", 75, "Companies and institutions")
        .predicate_with_desc(
            "works_at",
            vec!["Person", "Organization"],
            "Employment relationship",
        )
        .predicate_with_desc(
            "located_in",
            vec!["Organization", "Location"],
            "Organization location",
        )
        .build()
        .unwrap();

    println!("✓ Created schema with metadata");

    // Check metadata
    if let Some(person) = table.get_domain("Person") {
        println!("\nPerson domain:");
        if let Some(meta) = &person.metadata {
            println!("  Tags: {:?}", meta.tags);
            println!("  Attributes: {:?}", meta.attributes);
            if let Some(prov) = &meta.provenance {
                println!("  Created by: {}", prov.created_by);
                println!("  Created at: {}", prov.created_at);
            }
        }
    }

    // Check descriptions
    if let Some(location) = table.get_domain("Location") {
        if let Some(desc) = &location.description {
            println!("\nLocation: {}", desc);
        }
    }
}

fn knowledge_graph_schema() {
    let table = SchemaBuilder::new()
        // Core entity types
        .domain("Entity", 1000)
        .domain("Person", 500)
        .domain("Organization", 200)
        .domain("Document", 1000)
        .domain("Topic", 100)
        // Properties
        .domain("String", 10000)
        .domain("Date", 365)
        // Type hierarchy (would require hierarchy support)
        .subtype("Person", "Entity")
        .subtype("Organization", "Entity")
        .subtype("Document", "Entity")
        // Relationships
        .predicate("works_for", vec!["Person", "Organization"])
        .predicate("authored", vec!["Person", "Document"])
        .predicate("mentions", vec!["Document", "Entity"])
        .predicate("about", vec!["Document", "Topic"])
        .predicate("related_to", vec!["Entity", "Entity"])
        // Attributes
        .predicate("has_name", vec!["Entity", "String"])
        .predicate("founded", vec!["Organization", "Date"])
        .predicate("born", vec!["Person", "Date"])
        .build()
        .unwrap();

    println!("✓ Created knowledge graph schema");
    println!("\nStatistics:");
    println!("  Entity domains: {}", table.domains.len());
    println!("  Relationships: {}", table.predicates.len());

    // Calculate total possible facts
    let mut total_possible: usize = 0;
    for pred in table.predicates.values() {
        let mut pred_size: usize = 1;
        for domain_name in &pred.arg_domains {
            if let Some(domain) = table.get_domain(domain_name) {
                pred_size *= domain.cardinality;
            }
        }
        total_possible += pred_size;
    }
    println!("  Possible facts: ~{}", total_possible);

    println!("\nSample predicates:");
    for (name, pred) in table.predicates.iter().take(5) {
        println!("  - {}: {:?}", name, pred.arg_domains);
    }
}

fn validated_schema() {
    // Example of a valid schema
    println!("Valid schema:");
    let result = SchemaBuilder::new()
        .domain("Person", 100)
        .domain("Location", 50)
        .predicate("at", vec!["Person", "Location"])
        .build_and_validate();

    match result {
        Ok(table) => {
            println!("  ✓ Schema validated successfully");
            println!(
                "    {} domains, {} predicates",
                table.domains.len(),
                table.predicates.len()
            );
        }
        Err(e) => {
            println!("  ✗ Validation failed: {}", e);
        }
    }

    // Example of invalid schema (missing domain)
    println!("\nInvalid schema (missing domain):");
    let result = SchemaBuilder::new()
        .predicate("at", vec!["Person", "UndefinedDomain"])
        .build();

    match result {
        Ok(_) => println!("  ✗ Should have failed but succeeded"),
        Err(e) => println!("  ✓ Caught error as expected: {}", e),
    }

    // Example of valid but problematic schema
    println!("\nProblematic schema (unused domain):");
    let result = SchemaBuilder::new()
        .domain("UsedDomain", 100)
        .domain("UnusedDomain", 50)
        .predicate("test", vec!["UsedDomain"])
        .build_and_validate();

    match result {
        Ok(table) => {
            // Use SchemaAnalyzer to detect issues
            use tensorlogic_adapters::SchemaAnalyzer;
            let recommendations = SchemaAnalyzer::analyze(&table);

            if recommendations.issues.is_empty() {
                println!("  ✓ No issues found");
            } else {
                println!("  ⚠ Issues detected:");
                for issue in &recommendations.issues {
                    println!("    - {}", issue.description());
                }
            }
        }
        Err(e) => {
            println!("  ✗ Unexpected error: {}", e);
        }
    }

    println!("\n=== Example Complete ===");
}
