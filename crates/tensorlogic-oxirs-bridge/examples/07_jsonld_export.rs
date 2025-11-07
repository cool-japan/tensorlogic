//! JSON-LD export example
//!
//! This example demonstrates how to export RDF schema to JSON-LD format.
//! JSON-LD is a JSON-based serialization for Linked Data that is both
//! human-readable and machine-processable.
//!
//! Features demonstrated:
//! 1. Load RDF schema in Turtle format
//! 2. Export to JSON-LD with default context
//! 3. Export to JSON-LD with custom context
//! 4. Parse JSON-LD output to verify structure
//!
//! Run with: cargo run --example 07_jsonld_export -p tensorlogic-oxirs-bridge

use anyhow::Result;
use tensorlogic_oxirs_bridge::{JsonLdContext, SchemaAnalyzer};

fn main() -> Result<()> {
    println!("=== JSON-LD Export Example ===\n");

    // Step 1: Load a sample RDF schema
    println!("Step 1: Loading RDF Schema");
    println!("{}", "=".repeat(70));

    let rdf_schema = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .

        # Classes
        foaf:Person a rdfs:Class ;
            rdfs:label "Person" ;
            rdfs:comment "A person" .

        foaf:Agent a rdfs:Class ;
            rdfs:label "Agent" ;
            rdfs:comment "An agent (person, group, software or physical artifact)" .

        foaf:Organization a rdfs:Class ;
            rdfs:label "Organization" ;
            rdfs:comment "An organization" ;
            rdfs:subClassOf foaf:Agent .

        ex:Employee a rdfs:Class ;
            rdfs:label "Employee" ;
            rdfs:comment "An employee of an organization" ;
            rdfs:subClassOf foaf:Person .

        # Properties
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

        ex:worksFor a rdf:Property ;
            rdfs:label "works for" ;
            rdfs:comment "The organization a person works for" ;
            rdfs:domain ex:Employee ;
            rdfs:range foaf:Organization .

        ex:salary a rdf:Property ;
            rdfs:label "salary" ;
            rdfs:comment "The salary of an employee" ;
            rdfs:domain ex:Employee ;
            rdfs:range rdfs:Literal .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(rdf_schema)?;
    analyzer.analyze()?;

    println!(
        "✓ Loaded schema with {} classes and {} properties",
        analyzer.classes.len(),
        analyzer.properties.len()
    );

    // Step 2: Export to JSON-LD with default context
    println!("\n\nStep 2: Export to JSON-LD (Default Context)");
    println!("{}", "=".repeat(70));

    let jsonld_default = analyzer.to_jsonld()?;
    println!("\n{}", jsonld_default);

    // Verify JSON structure
    let parsed: serde_json::Value = serde_json::from_str(&jsonld_default)?;
    println!("\n✓ JSON-LD is valid JSON");
    println!(
        "  @context keys: {:?}",
        parsed["@context"]
            .as_object()
            .map(|o| o.keys().collect::<Vec<_>>())
            .unwrap_or_default()
    );
    println!(
        "  @graph items: {}",
        parsed["@graph"].as_array().map(|a| a.len()).unwrap_or(0)
    );

    // Step 3: Export to JSON-LD with custom context
    println!("\n\nStep 3: Export to JSON-LD (Custom Context)");
    println!("{}", "=".repeat(70));

    let mut custom_context = JsonLdContext::new();
    custom_context.add_prefix("ex".to_string(), "http://example.org/".to_string());
    custom_context.add_prefix("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());

    let jsonld_custom = analyzer.to_jsonld_with_context(custom_context)?;
    println!("\n{}", jsonld_custom);

    // Step 4: Demonstrate namespace compaction
    println!("\n\nStep 4: Namespace Compaction");
    println!("{}", "=".repeat(70));

    let parsed_custom: serde_json::Value = serde_json::from_str(&jsonld_custom)?;

    println!("\nCompacted IRIs in @graph:");
    if let Some(graph) = parsed_custom["@graph"].as_array() {
        for (i, item) in graph.iter().take(5).enumerate() {
            if let Some(id) = item["@id"].as_str() {
                if let Some(item_type) = item["@type"].as_str() {
                    println!("  [{}] {} (type: {})", i + 1, id, item_type);
                }
            }
        }
        if graph.len() > 5 {
            println!("  ... and {} more items", graph.len() - 5);
        }
    }

    // Step 5: Show context compaction benefits
    println!("\n\nStep 5: Context Compaction Benefits");
    println!("{}", "=".repeat(70));

    println!("\nDefault context (auto-detected):");
    if let Some(ctx) = parsed["@context"].as_object() {
        for (k, v) in ctx.iter().take(5) {
            println!("  \"{}\": \"{}\"", k, v);
        }
    }

    println!("\nCustom context (user-defined):");
    if let Some(ctx) = parsed_custom["@context"].as_object() {
        for (k, v) in ctx {
            println!("  \"{}\": \"{}\"", k, v);
        }
    }

    // Step 6: Demonstrate JSON-LD features
    println!("\n\nStep 6: JSON-LD Features");
    println!("{}", "=".repeat(70));

    println!("\n✓ JSON-LD provides several advantages:");
    println!("  1. Valid JSON - can be parsed by any JSON parser");
    println!("  2. Context-based - namespace prefixes reduce verbosity");
    println!("  3. Type information - @type specifies RDF types");
    println!("  4. Linked Data - @id provides unique identifiers");
    println!("  5. Web-friendly - integrates with REST APIs and JavaScript");

    println!("\n✓ Use cases:");
    println!("  - REST API responses for Linked Data");
    println!("  - JavaScript/TypeScript applications");
    println!("  - NoSQL databases (MongoDB, CouchDB)");
    println!("  - Web components and microservices");
    println!("  - Schema.org markup in HTML");

    // Step 7: File size comparison
    println!("\n\nStep 7: Format Comparison");
    println!("{}", "=".repeat(70));

    let ntriples = analyzer.to_ntriples();

    println!("\nFile sizes (bytes):");
    println!("  JSON-LD (default): {}", jsonld_default.len());
    println!("  JSON-LD (custom):  {}", jsonld_custom.len());
    println!("  N-Triples:         {}", ntriples.len());

    println!("\nReadability comparison:");
    println!("  JSON-LD:   ✓ Human-readable with context");
    println!("  N-Triples: ✓ Simple line-based format");

    println!("\nParsing complexity:");
    println!("  JSON-LD:   ✓ Standard JSON parser + context processing");
    println!("  N-Triples: ✓ Simple line-by-line parsing");

    println!("\n=== Example Complete ===");
    println!("\nNext steps:");
    println!("  - Use JSON-LD in REST APIs for semantic data");
    println!("  - Integrate with JavaScript frameworks");
    println!("  - Store in document databases");
    println!("  - Consume with JSON-LD processing libraries");

    Ok(())
}
