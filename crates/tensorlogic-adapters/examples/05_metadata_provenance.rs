//! Metadata and provenance tracking example.
//!
//! This example demonstrates how to attach rich metadata, provenance
//! information, and documentation to domains and predicates.

use tensorlogic_adapters::{
    Documentation, DomainInfo, Example, Metadata, PredicateInfo, Provenance, SymbolTable,
    TagRegistry,
};

fn main() -> anyhow::Result<()> {
    println!("=== Metadata and Provenance Example ===\n");

    let mut table = SymbolTable::new();

    // Create a domain with rich metadata
    println!("Creating domain with metadata...\n");

    let mut meta = Metadata::new();

    // Add provenance information
    meta.provenance = Some(
        Provenance::new("Alice", "2025-01-15T10:30:00Z")
            .with_source("schema.yaml", Some(42))
            .with_notes("Imported from legacy system during migration"),
    );

    println!("Provenance:");
    if let Some(prov) = &meta.provenance {
        println!("{}", prov);
    }

    // Add tags for categorization
    meta.add_tag("experimental");
    meta.add_tag("person");
    meta.add_tag("reasoning");

    println!("\nTags:");
    for tag in &meta.tags {
        println!("  - {}", tag);
    }

    // Add custom attributes
    meta.set_attribute("complexity", "O(n^2)");
    meta.set_attribute("author", "Alice");
    meta.set_attribute("version", "1.0.0");

    println!("\nAttributes:");
    for (key, value) in &meta.attributes {
        println!("  - {}: {}", key, value);
    }

    // Add documentation
    let mut doc = Documentation::new("Domain for persons in the knowledge base");
    doc.description = Some("This domain represents human entities in the system. \
                          Each person has unique properties and can participate in various relationships.".to_string());

    let example1 = Example::new("Creating a person", "let person = Person::new(\"alice\");")
        .with_output("Person { id: \"alice\", ... }");
    doc.add_example(example1);

    doc.add_note("This domain supports reasoning about human entities");
    doc.add_note("Use with caution in privacy-sensitive contexts");
    doc.add_see_also("Agent");
    doc.add_see_also("Organization");

    meta.set_documentation(doc);

    println!("\nDocumentation:");
    if let Some(doc) = &meta.documentation {
        println!("  Summary: {}", doc.summary);
        if let Some(desc) = &doc.description {
            println!("  Description: {}", desc);
        }
        println!("  Examples: {}", doc.examples.len());
        println!("  Notes: {}", doc.notes.len());
        println!("  See also: {:?}", doc.see_also);
    }

    // Add version history
    meta.add_version("1.0.0", "2025-01-15T10:00:00Z", "Alice", "Initial version");
    meta.add_version("1.1.0", "2025-01-20T15:30:00Z", "Bob", "Added constraints");
    meta.add_version(
        "1.2.0",
        "2025-01-25T09:15:00Z",
        "Alice",
        "Enhanced documentation",
    );

    println!("\nVersion History:");
    for version in &meta.version_history {
        println!(
            "  - {} ({}) by {}: {}",
            version.version, version.timestamp, version.author, version.changes
        );
    }

    // Create domain with metadata
    let person_domain = DomainInfo::new("Person", 100).with_metadata(meta);

    table.add_domain(person_domain)?;

    println!("\n=== Tag Registry ===\n");

    let registry = TagRegistry::standard();

    println!("Standard tag categories:");
    if let Some(domain_cat) = registry.get_category("domain") {
        println!("  - Domain tags: {:?}", domain_cat.tags);
    }
    if let Some(status_cat) = registry.get_category("status") {
        println!("  - Status tags: {:?}", status_cat.tags);
    }
    if let Some(app_cat) = registry.get_category("application") {
        println!("  - Application tags: {:?}", app_cat.tags);
    }

    println!("\nTag categorization:");
    let tags = vec!["experimental", "person", "reasoning"];
    for tag in tags {
        if let Some(category) = registry.find_category_for_tag(tag) {
            println!("  - '{}' belongs to category '{}'", tag, category);
        } else {
            println!("  - '{}' is uncategorized", tag);
        }
    }

    println!("\n=== Predicate with Metadata ===\n");

    let mut pred_meta = Metadata::new();
    pred_meta.provenance = Some(Provenance::new("Bob", "2025-01-16T14:00:00Z"));
    pred_meta.add_tag("reasoning");
    pred_meta.add_tag("stable");

    let knows = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
        .with_description("Indicates that one person knows another")
        .with_metadata(pred_meta);

    table.add_predicate(knows)?;

    println!("Added predicate 'knows' with metadata");

    // Query metadata
    if let Some(pred) = table.get_predicate("knows") {
        if let Some(meta) = &pred.metadata {
            println!("\nPredicate metadata:");
            if let Some(prov) = &meta.provenance {
                println!("  Created by: {}", prov.created_by);
            }
            println!("  Tags: {:?}", meta.tags);
        }
    }

    Ok(())
}
