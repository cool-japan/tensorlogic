//! Basic symbol table usage example.
//!
//! This example demonstrates how to create and use a symbol table
//! for managing domains, predicates, and variables.

use tensorlogic_adapters::{DomainInfo, PredicateInfo, SymbolTable};

fn main() -> anyhow::Result<()> {
    println!("=== Symbol Table Basics ===\n");

    // Create a new symbol table
    let mut table = SymbolTable::new();

    // Add domains with cardinalities
    println!("Adding domains...");
    table.add_domain(DomainInfo::new("Person", 100))?;
    table.add_domain(DomainInfo::new("City", 50))?;
    table.add_domain(DomainInfo::new("Company", 30))?;

    println!("  - Person (cardinality: 100)");
    println!("  - City (cardinality: 50)");
    println!("  - Company (cardinality: 30)\n");

    // Add predicates with their domain signatures
    println!("Adding predicates...");

    let knows = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
        .with_description("Indicates that one person knows another");
    table.add_predicate(knows)?;
    println!("  - knows(Person, Person)");

    let lives_in = PredicateInfo::new("lives_in", vec!["Person".to_string(), "City".to_string()])
        .with_description("Indicates where a person lives");
    table.add_predicate(lives_in)?;
    println!("  - lives_in(Person, City)");

    let works_at = PredicateInfo::new(
        "works_at",
        vec!["Person".to_string(), "Company".to_string()],
    )
    .with_description("Indicates where a person works");
    table.add_predicate(works_at)?;
    println!("  - works_at(Person, Company)\n");

    // Bind variables to domains
    println!("Binding variables...");
    table.bind_variable("x", "Person")?;
    table.bind_variable("y", "Person")?;
    table.bind_variable("c", "City")?;
    println!("  - x: Person");
    println!("  - y: Person");
    println!("  - c: City\n");

    // Query the symbol table
    println!("Querying symbol table...");
    if let Some(domain) = table.get_domain("Person") {
        println!(
            "  - Found domain 'Person' with cardinality {}",
            domain.cardinality
        );
    }

    if let Some(predicate) = table.get_predicate("knows") {
        println!("  - Found predicate 'knows' with arity {}", predicate.arity);
    }

    if let Some(domain) = table.get_variable_domain("x") {
        println!("  - Variable 'x' is bound to domain '{}'", domain);
    }

    println!("\n=== Export to JSON ===\n");
    let json = table.to_json()?;
    println!("Symbol table serialized to JSON ({} bytes)", json.len());

    println!("\n=== Export to YAML ===\n");
    let yaml = table.to_yaml()?;
    println!("Symbol table serialized to YAML ({} bytes)", yaml.len());

    Ok(())
}
