//! Complete validation pipeline example
//!
//! This example demonstrates a full end-to-end validation workflow:
//! 1. Load RDF schema
//! 2. Parse SHACL constraints
//! 3. Convert to TensorLogic rules
//! 4. Validate data (simulated)
//! 5. Generate validation reports
//! 6. Export results in multiple formats
//!
//! Run with: cargo run --example 06_validation_pipeline -p tensorlogic-oxirs-bridge

use anyhow::Result;
use tensorlogic_oxirs_bridge::{
    shacl::validation::{ShaclValidator, ValidationReport, ValidationResult, ValidationSeverity},
    SchemaAnalyzer, ShaclConverter,
};

fn main() -> Result<()> {
    println!("=== Complete Validation Pipeline ===\n");

    // Step 1: Load RDF schema
    println!("Step 1: Loading RDF Schema");
    println!("{}", "=".repeat(70));

    let rdf_schema = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .

        ex:Person a rdfs:Class ;
            rdfs:label "Person" ;
            rdfs:comment "A human being" .

        ex:Organization a rdfs:Class ;
            rdfs:label "Organization" ;
            rdfs:comment "An organized group" .

        ex:name a rdf:Property ;
            rdfs:label "name" ;
            rdfs:domain ex:Person ;
            rdfs:range rdfs:Literal .

        ex:email a rdf:Property ;
            rdfs:label "email" ;
            rdfs:domain ex:Person ;
            rdfs:range rdfs:Literal .

        ex:age a rdf:Property ;
            rdfs:label "age" ;
            rdfs:domain ex:Person ;
            rdfs:range rdfs:Literal .

        ex:worksFor a rdf:Property ;
            rdfs:label "works for" ;
            rdfs:domain ex:Person ;
            rdfs:range ex:Organization .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(rdf_schema)?;
    analyzer.analyze()?;

    let symbol_table = analyzer.to_symbol_table()?;
    println!(
        "✓ Loaded schema with {} domains and {} predicates",
        symbol_table.domains.len(),
        symbol_table.predicates.len()
    );

    // Step 2: Parse SHACL constraints
    println!("\n\nStep 2: Parsing SHACL Constraints");
    println!("{}", "=".repeat(70));

    let shacl_shapes = r#"
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
                sh:minLength 2 ;
                sh:maxLength 100 ;
            ] ;
            sh:property [
                sh:path ex:email ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
                sh:pattern "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$" ;
            ] ;
            sh:property [
                sh:path ex:age ;
                sh:minCount 1 ;
                sh:datatype xsd:integer ;
                sh:minInclusive 0 ;
                sh:maxInclusive 150 ;
            ] .
    "#;

    let converter = ShaclConverter::new(symbol_table);
    let rules = converter.convert_to_rules(shacl_shapes)?;
    println!(
        "✓ Generated {} TensorLogic rules from SHACL constraints",
        rules.len()
    );

    // Display some rules
    println!("\nSample rules:");
    for (i, rule) in rules.iter().take(3).enumerate() {
        let rule_str = format!("{:?}", rule);
        let truncated = if rule_str.len() > 60 {
            format!("{}...", &rule_str[..60])
        } else {
            rule_str
        };
        println!("  [{}] {}", i + 1, truncated);
    }

    // Step 3: Validate simulated data
    println!("\n\nStep 3: Validating Data Instances");
    println!("{}", "=".repeat(70));

    let validator = ShaclValidator::new();
    let mut report = ValidationReport::new();

    // Simulate validation of different data instances
    println!("\nValidating test instances...\n");

    // Instance 1: Alice - Valid
    println!("Instance 1: Alice (http://example.org/person/alice)");
    let alice_results = validate_person(
        &validator,
        "http://example.org/person/alice",
        "http://example.org/PersonShape",
        Some("Alice"),
        Some("alice@example.com"),
        Some(30),
    );
    for result in alice_results {
        report.add_result(result);
    }
    println!("  ✓ All constraints satisfied\n");

    // Instance 2: Bob - Missing email
    println!("Instance 2: Bob (http://example.org/person/bob)");
    let bob_results = validate_person(
        &validator,
        "http://example.org/person/bob",
        "http://example.org/PersonShape",
        Some("Bob"),
        None, // Missing email
        Some(25),
    );
    for result in bob_results {
        report.add_result(result);
    }
    println!("  ✗ Missing required email property\n");

    // Instance 3: Carol - Invalid age
    println!("Instance 3: Carol (http://example.org/person/carol)");
    let carol_results = validate_person(
        &validator,
        "http://example.org/person/carol",
        "http://example.org/PersonShape",
        Some("Carol"),
        Some("carol@example.com"),
        Some(200), // Invalid age
    );
    for result in carol_results {
        report.add_result(result);
    }
    println!("  ✗ Age exceeds maximum value (150)\n");

    // Instance 4: Dave - Invalid email format
    println!("Instance 4: Dave (http://example.org/person/dave)");
    let dave_results = validate_person(
        &validator,
        "http://example.org/person/dave",
        "http://example.org/PersonShape",
        Some("Dave"),
        Some("not-an-email"), // Invalid format
        Some(35),
    );
    for result in dave_results {
        report.add_result(result);
    }
    println!("  ✗ Email doesn't match required pattern\n");

    // Instance 5: Eve - Missing name
    println!("Instance 5: Eve (http://example.org/person/eve)");
    let eve_results = validate_person(
        &validator,
        "http://example.org/person/eve",
        "http://example.org/PersonShape",
        None, // Missing name
        Some("eve@example.com"),
        Some(28),
    );
    for result in eve_results {
        report.add_result(result);
    }
    println!("  ✗ Missing required name property\n");

    // Step 4: Generate validation report summary
    println!("\nStep 4: Validation Report Summary");
    println!("{}", "=".repeat(70));
    println!("{}", report.summary());
    println!("\nDetailed statistics:");
    println!("  Checked shapes: {}", report.statistics.total_shapes);
    println!(
        "  Checked constraints: {}",
        report.statistics.total_constraints
    );
    println!("  Violations: {}", report.statistics.violations);
    println!("  Warnings: {}", report.statistics.warnings);
    println!("  Infos: {}", report.statistics.infos);

    // Step 5: Export as Turtle (SHACL-compliant RDF)
    println!("\n\nStep 5: Export as Turtle (SHACL RDF)");
    println!("{}", "=".repeat(70));

    let turtle_export = report.to_turtle();
    println!("\n{}", turtle_export);

    // Step 6: Export as JSON
    println!("\n\nStep 6: Export as JSON");
    println!("{}", "=".repeat(70));

    let json_export = report.to_json()?;
    println!("\n{}", json_export);

    // Step 7: Filter results by severity
    println!("\n\nStep 7: Results Grouped by Severity");
    println!("{}", "=".repeat(70));

    let violations: Vec<_> = report
        .results
        .iter()
        .filter(|r| matches!(r.severity, ValidationSeverity::Violation))
        .collect();
    let warnings: Vec<_> = report
        .results
        .iter()
        .filter(|r| matches!(r.severity, ValidationSeverity::Warning))
        .collect();

    println!("\nViolations ({}):", violations.len());
    for (i, result) in violations.iter().enumerate() {
        println!("  [{}] {}", i + 1, result.focus_node);
        println!("      {}", result.message);
        if let Some(path) = &result.result_path {
            println!("      Path: {}", path);
        }
    }

    if !warnings.is_empty() {
        println!("\nWarnings ({}):", warnings.len());
        for (i, result) in warnings.iter().enumerate() {
            println!("  [{}] {}", i + 1, result.focus_node);
            println!("      {}", result.message);
        }
    }

    println!("\n=== Validation Pipeline Complete ===");
    println!("\nWorkflow Summary:");
    println!("  1. ✓ Loaded RDF schema");
    println!("  2. ✓ Parsed SHACL constraints");
    println!("  3. ✓ Converted to TensorLogic rules");
    println!("  4. ✓ Validated 5 test instances");
    println!("  5. ✓ Generated validation report");
    println!("  6. ✓ Exported in Turtle and JSON formats");
    println!("\nProduction Usage:");
    println!("  - Integrate with tensorlogic-compiler for rule compilation");
    println!("  - Use tensorlogic-scirs-backend for tensor execution");
    println!("  - Scale to large datasets with batch processing");

    Ok(())
}

/// Helper function to validate a person instance
fn validate_person(
    validator: &ShaclValidator,
    focus_node: &str,
    source_shape: &str,
    name: Option<&str>,
    email: Option<&str>,
    age: Option<i32>,
) -> Vec<ValidationResult> {
    let mut results = Vec::new();

    // Validate name
    if let Some(violation) = validator.validate_min_count(
        focus_node,
        "http://example.org/name",
        1,
        if name.is_some() { 1 } else { 0 },
    ) {
        results.push(violation.with_path("http://example.org/name"));
    }

    if let Some(name_val) = name {
        if name_val.len() < 2 {
            results.push(
                ValidationResult::new(
                    focus_node,
                    source_shape,
                    "http://www.w3.org/ns/shacl#MinLengthConstraintComponent",
                    "Name is too short (minimum 2 characters)",
                )
                .with_path("http://example.org/name"),
            );
        }
    }

    // Validate email
    if let Some(violation) = validator.validate_min_count(
        focus_node,
        "http://example.org/email",
        1,
        if email.is_some() { 1 } else { 0 },
    ) {
        results.push(violation.with_path("http://example.org/email"));
    }

    if let Some(email_val) = email {
        if let Some(violation) = validator.validate_pattern(
            focus_node,
            "http://example.org/email",
            email_val,
            "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
        ) {
            results.push(violation.with_path("http://example.org/email"));
        }
    }

    // Validate age
    if let Some(age_val) = age {
        if age_val < 0 {
            results.push(
                ValidationResult::new(
                    focus_node,
                    source_shape,
                    "http://www.w3.org/ns/shacl#MinInclusiveConstraintComponent",
                    format!("Age {} is below minimum value 0", age_val),
                )
                .with_path("http://example.org/age"),
            );
        }
        if age_val > 150 {
            results.push(
                ValidationResult::new(
                    focus_node,
                    source_shape,
                    "http://www.w3.org/ns/shacl#MaxInclusiveConstraintComponent",
                    format!("Age {} exceeds maximum value 150", age_val),
                )
                .with_path("http://example.org/age"),
            );
        }
    } else if let Some(violation) =
        validator.validate_min_count(focus_node, "http://example.org/age", 1, 0)
    {
        results.push(violation.with_path("http://example.org/age"));
    }

    results
}
