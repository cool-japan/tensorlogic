//! End-to-end SHACL validation pipeline example
//!
//! This example demonstrates:
//! 1. Loading RDF schema
//! 2. Parsing SHACL constraints
//! 3. Converting constraints to TensorLogic rules
//! 4. Generating validation reports

use tensorlogic_oxirs_bridge::{
    SchemaAnalyzer, ShaclConverter, ShaclValidator, ValidationResult, ValidationSeverity,
};

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic OxiRS Bridge: SHACL Validation Pipeline ===\n");

    // Step 1: Load RDF Schema
    println!("Step 1: Loading RDF Schema...");
    let schema_turtle = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .

        ex:Person rdf:type rdfs:Class ;
            rdfs:label "Person" ;
            rdfs:comment "A human being" .

        ex:String rdf:type rdfs:Class ;
            rdfs:label "String" .

        ex:Integer rdf:type rdfs:Class ;
            rdfs:label "Integer" .

        ex:name rdf:type rdf:Property ;
            rdfs:label "name" ;
            rdfs:domain ex:Person ;
            rdfs:range ex:String .

        ex:email rdf:type rdf:Property ;
            rdfs:label "email" ;
            rdfs:domain ex:Person ;
            rdfs:range ex:String .

        ex:age rdf:type rdf:Property ;
            rdfs:label "age" ;
            rdfs:domain ex:Person ;
            rdfs:range ex:Integer .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(schema_turtle)?;
    analyzer.analyze()?;

    println!("  ✓ Loaded {} classes", analyzer.classes.len());
    println!("  ✓ Loaded {} properties\n", analyzer.properties.len());

    // Step 2: Convert to SymbolTable
    println!("Step 2: Converting to TensorLogic SymbolTable...");
    let symbol_table = analyzer.to_symbol_table()?;
    println!(
        "  ✓ Created symbol table with {} domains\n",
        symbol_table.domains.len()
    );

    // Step 3: Parse SHACL Constraints
    println!("Step 3: Parsing SHACL Constraints...");
    let shacl_turtle = r#"
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
                sh:datatype xsd:string ;
            ] ;
            sh:property [
                sh:path ex:email ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
                sh:pattern "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$" ;
            ] ;
            sh:property [
                sh:path ex:age ;
                sh:datatype xsd:integer ;
                sh:minInclusive 0 ;
                sh:maxInclusive 150 ;
            ] .
    "#;

    let converter = ShaclConverter::new(symbol_table.clone());
    let rules = converter.convert_to_rules(shacl_turtle)?;

    println!(
        "  ✓ Generated {} TensorLogic rules from SHACL constraints\n",
        rules.len()
    );

    // Display the generated rules
    println!("Generated Rules:");
    for (i, rule) in rules.iter().enumerate() {
        println!("  Rule {}: {:?}", i + 1, rule);
    }
    println!();

    // Step 4: Validate Data (Mock)
    println!("Step 4: Validating Data...");
    let validator = ShaclValidator::new();

    // Simulate validation results
    let mut report = validator.validate_mock(
        "http://example.org/PersonShape",
        &["http://example.org/person/1", "http://example.org/person/2"],
        &[
            ("minCount:name".to_string(), true),
            ("minCount:email".to_string(), false), // Violation!
            ("pattern:email".to_string(), true),
            ("datatype:age".to_string(), true),
            ("minInclusive:age".to_string(), false), // Violation!
        ],
    );

    // Add specific validation results
    report.add_result(
        ValidationResult::new(
            "http://example.org/person/3",
            "http://example.org/PersonShape",
            "http://www.w3.org/ns/shacl#MaxCountConstraintComponent",
            "Person has 2 names but should have at most 1",
        )
        .with_path("http://example.org/name")
        .with_value("John Doe, Jane Doe"),
    );

    report.add_result(
        ValidationResult::new(
            "http://example.org/person/4",
            "http://example.org/PersonShape",
            "http://www.w3.org/ns/shacl#PatternConstraintComponent",
            "Email does not match pattern",
        )
        .with_path("http://example.org/email")
        .with_value("invalid-email")
        .with_severity(ValidationSeverity::Warning),
    );

    report.set_statistics(1, rules.len());

    println!("\n{}", report.summary());
    println!("\nValidation Results:");
    for (i, result) in report.results.iter().enumerate() {
        println!(
            "  {}. [{}] {} - {}",
            i + 1,
            match result.severity {
                ValidationSeverity::Violation => "VIOLATION",
                ValidationSeverity::Warning => "WARNING",
                ValidationSeverity::Info => "INFO",
            },
            result.focus_node,
            result.message
        );
    }

    // Step 5: Export Results
    println!("\nStep 5: Exporting Validation Report...");

    // Export as Turtle
    println!("\n--- Validation Report (Turtle) ---");
    println!("{}", report.to_turtle());

    // Export as JSON
    println!("\n--- Validation Report (JSON) ---");
    println!("{}", report.to_json()?);

    println!("\n=== Pipeline Complete ===");

    Ok(())
}
