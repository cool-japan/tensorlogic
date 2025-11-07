//! SHACL constraints to TensorLogic rules example
//!
//! This example demonstrates how to:
//! 1. Define SHACL shapes with various constraints
//! 2. Parse SHACL constraints
//! 3. Convert constraints to TensorLogic rules
//! 4. Inspect the generated logical rules
//!
//! Run with: cargo run --example 02_shacl_constraints -p tensorlogic-oxirs-bridge

use anyhow::Result;
use tensorlogic_adapters::SymbolTable;
use tensorlogic_oxirs_bridge::ShaclConverter;

fn main() -> Result<()> {
    println!("=== SHACL Constraints to TensorLogic Rules ===\n");

    // Define a comprehensive SHACL shape with multiple constraint types
    let shacl_shapes = r#"
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

        # Person shape with comprehensive validation rules
        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;

            # Name must exist and have reasonable length
            sh:property [
                sh:path ex:name ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
                sh:datatype xsd:string ;
                sh:minLength 2 ;
                sh:maxLength 100 ;
            ] ;

            # Email must exist, be unique, and match pattern
            sh:property [
                sh:path ex:email ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
                sh:datatype xsd:string ;
                sh:pattern "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$" ;
            ] ;

            # Age must be in valid range
            sh:property [
                sh:path ex:age ;
                sh:datatype xsd:integer ;
                sh:minInclusive 0 ;
                sh:maxInclusive 150 ;
            ] ;

            # Status must be one of the allowed values
            sh:property [
                sh:path ex:status ;
                sh:in ( "active" "inactive" "pending" ) ;
            ] ;

            # Address must conform to AddressShape
            sh:property [
                sh:path ex:address ;
                sh:node ex:AddressShape ;
            ] .

        # Address shape
        ex:AddressShape a sh:NodeShape ;
            sh:targetClass ex:Address ;
            sh:property [
                sh:path ex:street ;
                sh:minCount 1 ;
            ] ;
            sh:property [
                sh:path ex:postalCode ;
                sh:minCount 1 ;
                sh:pattern "^[0-9]{5}(-[0-9]{4})?$" ;
            ] .

        # Product shape with logical constraints
        ex:ProductShape a sh:NodeShape ;
            sh:targetClass ex:Product ;

            # Must have either digital or physical properties (XOR)
            sh:property [
                sh:path ex:format ;
                sh:xone ( ex:DigitalFormat ex:PhysicalFormat ) ;
            ] ;

            # Price constraint using NOT
            sh:property [
                sh:path ex:price ;
                sh:not ex:NegativePriceShape ;
            ] .

        ex:NegativePriceShape a sh:NodeShape ;
            sh:property [
                sh:path ex:value ;
                sh:maxInclusive -0.01 ;
            ] .
    "#;

    // Step 1: Create SHACL converter
    println!("Step 1: Creating SHACL converter...");
    let symbol_table = SymbolTable::new();
    let converter = ShaclConverter::new(symbol_table);
    println!("✓ Converter created\n");

    // Step 2: Parse and convert SHACL shapes to TensorLogic rules
    println!("Step 2: Converting SHACL shapes to TensorLogic rules...");
    let rules = converter.convert_to_rules(shacl_shapes)?;
    println!("✓ Generated {} rules\n", rules.len());

    // Step 3: Display generated rules by category
    println!("Step 3: Generated TensorLogic Rules");
    println!("{}", "=".repeat(70));

    let mut cardinality_rules = Vec::new();
    let mut type_rules = Vec::new();
    let mut value_rules = Vec::new();
    let mut pattern_rules = Vec::new();
    let mut logical_rules = Vec::new();
    let mut other_rules = Vec::new();

    for (i, rule) in rules.iter().enumerate() {
        let rule_str = format!("{:?}", rule);

        if rule_str.contains("Exists") {
            cardinality_rules.push((i, rule_str));
        } else if rule_str.contains("hasType") || rule_str.contains("hasDatatype") {
            type_rules.push((i, rule_str));
        } else if rule_str.contains("greaterOrEqual")
            || rule_str.contains("lessOrEqual")
            || rule_str.contains("lengthAtLeast")
            || rule_str.contains("lengthAtMost")
        {
            value_rules.push((i, rule_str));
        } else if rule_str.contains("matchesPattern") || rule_str.contains("equals") {
            pattern_rules.push((i, rule_str));
        } else if rule_str.contains("distinct")
            || rule_str.contains("nodeConformsTo")
            || rule_str.contains("conformsTo")
            || rule_str.contains("Not")
        {
            logical_rules.push((i, rule_str));
        } else {
            other_rules.push((i, rule_str));
        }
    }

    if !cardinality_rules.is_empty() {
        println!("\n📊 Cardinality Rules ({}):", cardinality_rules.len());
        for (i, rule) in &cardinality_rules {
            println!("  [{}] {}", i, truncate_rule(rule, 65));
        }
    }

    if !type_rules.is_empty() {
        println!("\n🏷️  Type Constraint Rules ({}):", type_rules.len());
        for (i, rule) in &type_rules {
            println!("  [{}] {}", i, truncate_rule(rule, 65));
        }
    }

    if !value_rules.is_empty() {
        println!("\n📏 Value Range Rules ({}):", value_rules.len());
        for (i, rule) in &value_rules {
            println!("  [{}] {}", i, truncate_rule(rule, 65));
        }
    }

    if !pattern_rules.is_empty() {
        println!("\n🔍 Pattern Matching Rules ({}):", pattern_rules.len());
        for (i, rule) in &pattern_rules {
            println!("  [{}] {}", i, truncate_rule(rule, 65));
        }
    }

    if !logical_rules.is_empty() {
        println!("\n🧩 Logical Constraint Rules ({}):", logical_rules.len());
        for (i, rule) in &logical_rules {
            println!("  [{}] {}", i, truncate_rule(rule, 65));
        }
    }

    if !other_rules.is_empty() {
        println!("\n📋 Other Rules ({}):", other_rules.len());
        for (i, rule) in &other_rules {
            println!("  [{}] {}", i, truncate_rule(rule, 65));
        }
    }

    // Step 4: Show detailed examples of specific rule types
    println!("\n\nStep 4: Detailed Rule Examples");
    println!("{}", "=".repeat(70));

    if !rules.is_empty() {
        println!("\nExample 1 - Existence Constraint (minCount):");
        if let Some((_, rule)) = cardinality_rules.first() {
            println!("  {}", rule);
            println!("  → Ensures the property has at least one value");
        }
    }

    if rules.len() > 1 {
        println!("\nExample 2 - Type Constraint (datatype):");
        if let Some((_, rule)) = type_rules.first() {
            println!("  {}", rule);
            println!("  → Validates that values have the correct datatype");
        }
    }

    if rules.len() > 2 {
        println!("\nExample 3 - Range Constraint (minInclusive/maxInclusive):");
        if let Some((_, rule)) = value_rules.first() {
            println!("  {}", rule);
            println!("  → Ensures numeric values are within valid range");
        }
    }

    println!("\n=== Example Complete ===");
    println!("\nNext Steps:");
    println!("  1. Compile these rules using tensorlogic-compiler");
    println!("  2. Execute with tensorlogic-scirs-backend");
    println!("  3. Generate validation reports with ShaclValidator");

    Ok(())
}

/// Helper function to truncate long rules for display
fn truncate_rule(rule: &str, max_len: usize) -> String {
    if rule.len() <= max_len {
        rule.to_string()
    } else {
        format!("{}...", &rule[..max_len])
    }
}
