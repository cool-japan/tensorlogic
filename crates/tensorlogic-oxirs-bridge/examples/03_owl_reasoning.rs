//! OWL reasoning and inference example
//!
//! This example demonstrates how to:
//! 1. Load an OWL ontology with class hierarchies
//! 2. Extract OWL class expressions and property characteristics
//! 3. Perform RDFS/OWL inference
//! 4. Query inferred relationships
//!
//! Run with: cargo run --example 03_owl_reasoning -p tensorlogic-oxirs-bridge

use anyhow::Result;
use tensorlogic_oxirs_bridge::SchemaAnalyzer;

fn main() -> Result<()> {
    println!("=== OWL Reasoning and Inference ===\n");

    // Define an OWL ontology with class and property hierarchies
    let owl_ontology = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix ex: <http://example.org/> .

        # Class hierarchy
        ex:LivingThing a rdfs:Class ;
            rdfs:label "Living Thing" ;
            rdfs:comment "Any living organism" .

        ex:Animal a rdfs:Class ;
            rdfs:label "Animal" ;
            rdfs:subClassOf ex:LivingThing .

        ex:Plant a rdfs:Class ;
            rdfs:label "Plant" ;
            rdfs:subClassOf ex:LivingThing .

        ex:Mammal a rdfs:Class ;
            rdfs:label "Mammal" ;
            rdfs:subClassOf ex:Animal .

        ex:Bird a rdfs:Class ;
            rdfs:label "Bird" ;
            rdfs:subClassOf ex:Animal .

        ex:Dog a rdfs:Class ;
            rdfs:label "Dog" ;
            rdfs:subClassOf ex:Mammal .

        ex:Cat a rdfs:Class ;
            rdfs:label "Cat" ;
            rdfs:subClassOf ex:Mammal .

        ex:Eagle a rdfs:Class ;
            rdfs:label "Eagle" ;
            rdfs:subClassOf ex:Bird .

        # OWL class expressions
        ex:Pet a owl:Class ;
            rdfs:label "Pet" ;
            owl:unionOf ( ex:Dog ex:Cat ) .

        ex:WildAnimal a owl:Class ;
            rdfs:label "Wild Animal" ;
            owl:complementOf ex:Pet .

        # Property hierarchy
        ex:relatedTo a rdf:Property ;
            rdfs:label "related to" ;
            rdfs:domain ex:LivingThing ;
            rdfs:range ex:LivingThing .

        ex:partOf a rdf:Property ;
            rdfs:label "part of" ;
            rdfs:subPropertyOf ex:relatedTo ;
            rdf:type owl:TransitiveProperty .

        ex:contains a rdf:Property ;
            rdfs:label "contains" ;
            rdfs:subPropertyOf ex:relatedTo ;
            owl:inverseOf ex:partOf .

        ex:hasParent a rdf:Property ;
            rdfs:label "has parent" ;
            rdfs:domain ex:Animal ;
            rdfs:range ex:Animal ;
            rdfs:subPropertyOf ex:relatedTo .

        ex:hasChild a rdf:Property ;
            rdfs:label "has child" ;
            owl:inverseOf ex:hasParent .

        # OWL property characteristics
        ex:sameAs a rdf:Property ;
            rdf:type owl:SymmetricProperty ;
            rdf:type owl:TransitiveProperty .

        ex:marriedTo a rdf:Property ;
            rdf:type owl:SymmetricProperty ;
            rdf:type owl:FunctionalProperty .

        ex:hasSocialSecurityNumber a rdf:Property ;
            rdf:type owl:InverseFunctionalProperty .
    "#;

    // Step 1: Load and analyze the ontology
    println!("Step 1: Loading OWL ontology...");
    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(owl_ontology)?;
    analyzer.analyze()?;
    println!("✓ Ontology loaded and analyzed");
    println!("  - Classes: {}", analyzer.classes.len());
    println!("  - Properties: {}", analyzer.properties.len());
    println!();

    // Step 2: Display class hierarchy
    println!("Step 2: Class Hierarchy");
    println!("{}", "=".repeat(60));
    for (iri, class_info) in &analyzer.classes {
        let name = SchemaAnalyzer::iri_to_name(iri);
        println!("{}", name);
        if !class_info.subclass_of.is_empty() {
            for parent in &class_info.subclass_of {
                let parent_name = SchemaAnalyzer::iri_to_name(parent);
                println!("  └─ subClassOf: {}", parent_name);
            }
        }
    }
    println!();

    // Step 3: Display property hierarchy
    println!("Step 3: Property Hierarchy");
    println!("{}", "=".repeat(60));
    for (iri, prop_info) in &analyzer.properties {
        let name = SchemaAnalyzer::iri_to_name(iri);
        println!("{}", name);
        if !prop_info.domain.is_empty() {
            println!(
                "  Domain: {:?}",
                prop_info
                    .domain
                    .iter()
                    .map(|d| SchemaAnalyzer::iri_to_name(d))
                    .collect::<Vec<_>>()
            );
        }
        if !prop_info.range.is_empty() {
            println!(
                "  Range: {:?}",
                prop_info
                    .range
                    .iter()
                    .map(|r| SchemaAnalyzer::iri_to_name(r))
                    .collect::<Vec<_>>()
            );
        }
    }
    println!();

    // Step 4: Create inference engine and perform reasoning
    println!("Step 4: RDFS/OWL Inference");
    println!("{}", "=".repeat(60));
    let mut inference_engine = analyzer.create_inference_engine();
    inference_engine.materialize()?;

    // Test subclass queries
    println!("\nSubclass Reasoning:");
    let test_pairs = vec![
        ("ex:Dog", "ex:Mammal"),
        ("ex:Dog", "ex:Animal"),
        ("ex:Dog", "ex:LivingThing"),
        ("ex:Cat", "ex:Animal"),
        ("ex:Eagle", "ex:Bird"),
        ("ex:Dog", "ex:Bird"),
    ];

    for (subclass, superclass) in test_pairs {
        let is_sub = inference_engine.is_subclass_of(subclass, superclass);
        let status = if is_sub { "✓ TRUE " } else { "✗ FALSE" };
        println!(
            "  {} {} ⊆ {}",
            status,
            SchemaAnalyzer::iri_to_name(subclass),
            SchemaAnalyzer::iri_to_name(superclass)
        );
    }

    // Get all superclasses for a given class
    println!("\nAll Superclasses of 'Dog':");
    let dog_supers = inference_engine.get_all_superclasses("ex:Dog");
    for sup in &dog_supers {
        println!("  - {}", SchemaAnalyzer::iri_to_name(sup));
    }

    // Test subproperty queries
    println!("\nSubproperty Reasoning:");
    let prop_test_pairs = vec![
        ("ex:hasParent", "ex:relatedTo"),
        ("ex:partOf", "ex:relatedTo"),
        ("ex:hasChild", "ex:relatedTo"),
    ];

    for (subprop, superprop) in prop_test_pairs {
        let is_sub = inference_engine.is_subproperty_of(subprop, superprop);
        let status = if is_sub { "✓ TRUE " } else { "✗ FALSE" };
        println!(
            "  {} {} ⊑ {}",
            status,
            SchemaAnalyzer::iri_to_name(subprop),
            SchemaAnalyzer::iri_to_name(superprop)
        );
    }

    // Get all superproperties for a given property
    println!("\nAll Superproperties of 'hasParent':");
    let parent_supers = inference_engine.get_all_superproperties("ex:hasParent");
    for sup in &parent_supers {
        println!("  - {}", SchemaAnalyzer::iri_to_name(sup));
    }

    // Step 5: Inference statistics
    println!("\n\nStep 5: Inference Statistics");
    println!("{}", "=".repeat(60));
    let stats = inference_engine.get_inference_stats();
    println!("Original triples: {}", stats.original_triples);
    println!("Inferred triples: {}", stats.inferred_triples);
    println!("Total triples: {}", stats.total_triples);
    println!("Subclass relationships: {}", stats.subclass_relations);
    println!("Subproperty relationships: {}", stats.subproperty_relations);

    // Step 6: Complete graph access
    println!("\n\nStep 6: Materialized Knowledge");
    println!("{}", "=".repeat(60));
    let complete_graph = inference_engine.get_complete_graph();
    println!(
        "Complete graph contains {} triples (original + inferred)",
        complete_graph.len()
    );
    println!(
        "\nInferred triples only: {}",
        inference_engine.get_inferred_triples().len()
    );

    println!("\n=== Example Complete ===");
    println!("\nKey Insights:");
    println!("  - Transitive reasoning: Dog → Mammal → Animal → LivingThing");
    println!("  - Property inheritance: hasParent inherits domain/range from relatedTo");
    println!("  - OWL characteristics: Symmetric, Transitive, Functional properties");

    Ok(())
}
