//! Basic RDF schema analysis example
//!
//! This example demonstrates how to:
//! 1. Load an RDF schema in Turtle format
//! 2. Analyze classes and properties
//! 3. Convert to TensorLogic SymbolTable
//!
//! Run with: cargo run --example 01_basic_schema_analysis -p tensorlogic-oxirs-bridge

use anyhow::Result;
use tensorlogic_oxirs_bridge::SchemaAnalyzer;

fn main() -> Result<()> {
    println!("=== Basic RDF Schema Analysis ===\n");

    // Define a simple RDF schema using FOAF (Friend of a Friend) vocabulary
    let foaf_schema = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .

        # Class definitions
        foaf:Person a rdfs:Class ;
            rdfs:label "Person" ;
            rdfs:comment "A person" ;
            rdfs:subClassOf foaf:Agent .

        foaf:Agent a rdfs:Class ;
            rdfs:label "Agent" ;
            rdfs:comment "An agent (person, group, software or physical artifact)" .

        foaf:Organization a rdfs:Class ;
            rdfs:label "Organization" ;
            rdfs:comment "An organization" ;
            rdfs:subClassOf foaf:Agent .

        foaf:Document a rdfs:Class ;
            rdfs:label "Document" ;
            rdfs:comment "A document" .

        # Property definitions
        foaf:name a rdf:Property ;
            rdfs:label "name" ;
            rdfs:comment "A name for some thing" ;
            rdfs:domain foaf:Agent ;
            rdfs:range rdfs:Literal .

        foaf:knows a rdf:Property ;
            rdfs:label "knows" ;
            rdfs:comment "A person known by this person" ;
            rdfs:domain foaf:Person ;
            rdfs:range foaf:Person .

        foaf:member a rdf:Property ;
            rdfs:label "member" ;
            rdfs:comment "Indicates a member of a Group" ;
            rdfs:domain foaf:Organization ;
            rdfs:range foaf:Agent .

        foaf:homepage a rdf:Property ;
            rdfs:label "homepage" ;
            rdfs:comment "A homepage for some thing" ;
            rdfs:domain rdfs:Resource ;
            rdfs:range foaf:Document .
    "#;

    // Step 1: Create analyzer and load schema
    println!("Step 1: Loading RDF schema...");
    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(foaf_schema)?;
    println!("✓ Schema loaded\n");

    // Step 2: Analyze schema to extract classes and properties
    println!("Step 2: Analyzing schema...");
    analyzer.analyze()?;
    println!("✓ Schema analyzed\n");

    // Step 3: Display extracted classes
    println!("Step 3: Extracted Classes ({})", analyzer.classes.len());
    println!("{}", "=".repeat(50));
    for (iri, class_info) in &analyzer.classes {
        println!("Class: {}", class_info.label.as_ref().unwrap_or(iri));
        println!("  IRI: {}", iri);
        if let Some(comment) = &class_info.comment {
            println!("  Comment: {}", comment);
        }
        if !class_info.subclass_of.is_empty() {
            println!("  Subclass of: {:?}", class_info.subclass_of);
        }
        println!();
    }

    // Step 4: Display extracted properties
    println!(
        "Step 4: Extracted Properties ({})",
        analyzer.properties.len()
    );
    println!("{}", "=".repeat(50));
    for (iri, prop_info) in &analyzer.properties {
        println!("Property: {}", prop_info.label.as_ref().unwrap_or(iri));
        println!("  IRI: {}", iri);
        if let Some(comment) = &prop_info.comment {
            println!("  Comment: {}", comment);
        }
        if !prop_info.domain.is_empty() {
            println!("  Domain: {:?}", prop_info.domain);
        }
        if !prop_info.range.is_empty() {
            println!("  Range: {:?}", prop_info.range);
        }
        println!();
    }

    // Step 5: Convert to TensorLogic SymbolTable
    println!("Step 5: Converting to TensorLogic SymbolTable...");
    let symbol_table = analyzer.to_symbol_table()?;
    println!("✓ Conversion complete\n");

    // Step 6: Display SymbolTable contents
    println!("Step 6: SymbolTable Summary");
    println!("{}", "=".repeat(50));
    println!("Domains (classes): {}", symbol_table.domains.len());
    for (name, domain) in &symbol_table.domains {
        println!("  - {} (cardinality: {})", name, domain.cardinality);
    }
    println!(
        "\nPredicates (properties): {}",
        symbol_table.predicates.len()
    );
    for (name, predicate) in &symbol_table.predicates {
        println!("  - {} (arity: {})", name, predicate.arg_domains.len());
        println!("    Args: {:?}", predicate.arg_domains);
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
