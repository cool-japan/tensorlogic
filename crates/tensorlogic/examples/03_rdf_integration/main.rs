//! Example: RDF Schema → TensorLogic Integration
//!
//! This example demonstrates the full pipeline:
//! 1. Load RDF schema from Turtle
//! 2. Analyze schema and convert to SymbolTable
//! 3. Create TLExpr rules based on schema
//! 4. Compile rules to EinsumGraph
//! 5. Execute using SciRS2 backend
//! 6. Track provenance with RDF*

use anyhow::Result;
use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_oxirs_bridge::{ProvenanceTracker, SchemaAnalyzer};
use tensorlogic_scirs_backend::Scirs2Exec;

fn main() -> Result<()> {
    println!("=== RDF Schema → TensorLogic Integration ===\n");

    // Step 1: Define RDF schema in Turtle
    println!("Step 1: Loading RDF schema...");
    let schema = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .

        foaf:Person rdf:type rdfs:Class ;
                    rdfs:label "Person" ;
                    rdfs:comment "A person" .

        foaf:Organization rdf:type rdfs:Class ;
                         rdfs:label "Organization" ;
                         rdfs:comment "An organization" .

        foaf:knows rdf:type rdf:Property ;
                   rdfs:label "knows" ;
                   rdfs:comment "A person known by this person" ;
                   rdfs:domain foaf:Person ;
                   rdfs:range foaf:Person .

        foaf:member rdf:type rdf:Property ;
                    rdfs:label "member" ;
                    rdfs:comment "Indicates a member of an organization" ;
                    rdfs:domain foaf:Person ;
                    rdfs:range foaf:Organization .
    "#;

    // Step 2: Analyze schema
    println!("Step 2: Analyzing RDF schema...");
    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(schema)?;
    analyzer.analyze()?;

    println!("  Found {} classes:", analyzer.classes.len());
    for (iri, class_info) in &analyzer.classes {
        println!(
            "    - {} ({})",
            SchemaAnalyzer::iri_to_name(iri),
            class_info.label.as_ref().unwrap_or(&"no label".to_string())
        );
    }

    println!("  Found {} properties:", analyzer.properties.len());
    for (iri, prop_info) in &analyzer.properties {
        println!(
            "    - {} ({})",
            SchemaAnalyzer::iri_to_name(iri),
            prop_info.label.as_ref().unwrap_or(&"no label".to_string())
        );
    }

    // Step 3: Convert to SymbolTable
    println!("\nStep 3: Converting to SymbolTable...");
    let symbol_table = analyzer.to_symbol_table()?;
    println!(
        "  Domains: {:?}",
        symbol_table.domains.keys().collect::<Vec<_>>()
    );
    println!(
        "  Predicates: {:?}",
        symbol_table.predicates.keys().collect::<Vec<_>>()
    );

    // Step 4: Create TensorLogic rules
    println!("\nStep 4: Creating TensorLogic rules...");

    // Rule 1: Simple predicate - persons who know someone
    // ∃y. knows(x, y)
    let rule1 = TLExpr::Exists {
        var: "y".to_string(),
        domain: "Person".to_string(),
        body: Box::new(TLExpr::Pred {
            name: "knows".to_string(),
            args: vec![Term::Var("x".to_string()), Term::Var("y".to_string())],
        }),
    };

    println!("  Rule 1: Exists query");
    println!("    ∃y. knows(x, y) - Find all persons who know someone");

    // Step 5: Compile to EinsumGraph
    println!("\nStep 5: Compiling rules to tensor operations...");
    let graph = compile_to_einsum(&rule1)?;
    println!(
        "  Generated {} tensors, {} operations",
        graph.tensors.len(),
        graph.nodes.len()
    );

    // Step 6: Set up provenance tracking
    println!("\nStep 6: Setting up provenance tracking...");
    let mut provenance = ProvenanceTracker::new();

    // Track entities
    provenance.track_entity("http://xmlns.com/foaf/0.1/Person".to_string(), 0);
    provenance.track_entity("http://xmlns.com/foaf/0.1/knows".to_string(), 1);

    // Track rule
    provenance.track_shape(
        "http://example.org/shapes#KnowsSomeone".to_string(),
        "∃y. knows(x, y)".to_string(),
        0,
    );

    println!("  Tracked {} entities", provenance.entity_to_tensor.len());

    // Step 7: Export provenance as RDF*
    println!("\nStep 7: Exporting provenance as RDF*...");
    let rdf_star = provenance.to_rdf_star();
    println!("  Generated {} RDF* statements:", rdf_star.len());
    for stmt in &rdf_star {
        println!("    {}", stmt);
    }

    // Step 8: Export provenance as JSON
    println!("\nStep 8: Exporting provenance as JSON...");
    let json = provenance.to_json()?;
    println!("  JSON (first 200 chars): {}", &json[..json.len().min(200)]);

    // Step 9: Execute with SciRS2 backend (demonstration)
    println!("\nStep 9: Executing with SciRS2 backend...");
    let _executor = Scirs2Exec::new();
    println!("  Executor initialized (ready for execution)");

    // Note: Full execution would require input tensors
    println!("  (Full execution requires concrete input data)");

    println!("\n=== Integration Complete ===");
    println!("\nSummary:");
    println!(
        "  ✓ RDF schema parsed ({} classes, {} properties)",
        analyzer.classes.len(),
        analyzer.properties.len()
    );
    println!(
        "  ✓ SymbolTable created ({} domains, {} predicates)",
        symbol_table.domains.len(),
        symbol_table.predicates.len()
    );
    println!("  ✓ Rules compiled to tensor operations");
    println!("  ✓ Provenance tracking enabled");
    println!("  ✓ Backend ready for execution");

    Ok(())
}
