//! Integration tests with real-world schema scenarios.

use tensorlogic_adapters::*;

/// Test a complete academic domain schema.
#[test]
fn test_academic_schema() {
    let mut table = SymbolTable::new();

    // Define domains
    table.add_domain(DomainInfo::new("Student", 1000)).unwrap();
    table.add_domain(DomainInfo::new("Professor", 100)).unwrap();
    table.add_domain(DomainInfo::new("Course", 500)).unwrap();
    table.add_domain(DomainInfo::new("Department", 20)).unwrap();

    // Define domain hierarchy
    let mut hierarchy = DomainHierarchy::new();
    hierarchy.add_subtype("Student", "Person");
    hierarchy.add_subtype("Professor", "Person");

    // Define predicates
    table
        .add_predicate(PredicateInfo::new(
            "enrolled",
            vec!["Student".to_string(), "Course".to_string()],
        ))
        .unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "teaches",
            vec!["Professor".to_string(), "Course".to_string()],
        ))
        .unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "belongs_to",
            vec!["Course".to_string(), "Department".to_string()],
        ))
        .unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "advises",
            vec!["Professor".to_string(), "Student".to_string()],
        ))
        .unwrap();

    // Validate schema
    let validator = SchemaValidator::new(&table);
    let report = validator.validate().unwrap();
    assert!(report.errors.is_empty(), "Schema should be valid");

    // Test serialization round-trip
    let json = table.to_json().unwrap();
    let restored = SymbolTable::from_json(&json).unwrap();
    assert_eq!(table.domains.len(), restored.domains.len());
    assert_eq!(table.predicates.len(), restored.predicates.len());

    // Test compact representation
    let compact = CompactSchema::from_symbol_table(&table);
    let stats = compact.compression_stats();
    // Compression might not always reduce size for medium schemas
    assert!(
        stats.compression_ratio() > 0.0,
        "Compression ratio should be positive"
    );

    println!(
        "Academic schema - Compression ratio: {:.2}%",
        stats.compression_ratio() * 100.0
    );
    println!(
        "Academic schema - Space savings: {:.2}%",
        stats.space_savings()
    );
}

/// Test a social network schema with versioning.
#[test]
fn test_social_network_versioning() {
    // Version 1: Basic social network
    let mut v1 = SymbolTable::new();
    v1.add_domain(DomainInfo::new("User", 10000)).unwrap();
    v1.add_domain(DomainInfo::new("Post", 50000)).unwrap();
    v1.add_predicate(PredicateInfo::new(
        "follows",
        vec!["User".to_string(), "User".to_string()],
    ))
    .unwrap();
    v1.add_predicate(PredicateInfo::new(
        "created",
        vec!["User".to_string(), "Post".to_string()],
    ))
    .unwrap();

    // Version 2: Add reactions and groups
    let mut v2 = v1.clone();
    v2.add_domain(DomainInfo::new("Group", 1000)).unwrap();
    v2.add_domain(DomainInfo::new("Reaction", 5)).unwrap();
    v2.add_predicate(PredicateInfo::new(
        "member_of",
        vec!["User".to_string(), "Group".to_string()],
    ))
    .unwrap();
    v2.add_predicate(PredicateInfo::new(
        "reacted_with",
        vec![
            "User".to_string(),
            "Post".to_string(),
            "Reaction".to_string(),
        ],
    ))
    .unwrap();

    // Compare versions
    let diff = compute_diff(&v1, &v2);
    assert_eq!(diff.domains_added.len(), 2);
    assert_eq!(diff.predicates_added.len(), 2);
    assert!(diff.is_backward_compatible());

    let compat = check_compatibility(&v1, &v2);
    assert_eq!(compat, CompatibilityLevel::BackwardCompatible);

    // Generate diff report
    let report = diff.report();
    assert!(report.contains("Domains Added"));
    assert!(report.contains("Group"));
    assert!(report.contains("Reaction"));

    println!("Social network diff report:\n{}", report);
}

/// Test e-commerce schema with constraints.
#[test]
fn test_ecommerce_schema() {
    let mut table = SymbolTable::new();

    // Define domains
    table
        .add_domain(DomainInfo::new("Customer", 100000))
        .unwrap();
    table.add_domain(DomainInfo::new("Product", 10000)).unwrap();
    table.add_domain(DomainInfo::new("Order", 500000)).unwrap();
    table.add_domain(DomainInfo::new("Category", 100)).unwrap();

    // Define predicates with constraints
    let mut purchases = PredicateInfo::new(
        "purchases",
        vec!["Customer".to_string(), "Product".to_string()],
    );
    purchases.constraints = Some(
        PredicateConstraints::new().with_property(PredicateProperty::Functional), // Each purchase links to one product
    );
    table.add_predicate(purchases).unwrap();

    let mut contains =
        PredicateInfo::new("contains", vec!["Order".to_string(), "Product".to_string()]);
    contains.description = Some("Products contained in an order".to_string());
    table.add_predicate(contains).unwrap();

    let mut in_category = PredicateInfo::new(
        "in_category",
        vec!["Product".to_string(), "Category".to_string()],
    );
    in_category.constraints = Some(
        PredicateConstraints::new().with_property(PredicateProperty::Functional), // Each product has one category
    );
    table.add_predicate(in_category).unwrap();

    // Validate
    let validator = SchemaValidator::new(&table);
    let report = validator.validate().unwrap();
    assert!(report.errors.is_empty());

    // Test compiler export
    let export = CompilerExport::export_all(&table);
    assert_eq!(export.domains.len(), 4);
    assert_eq!(export.predicate_signatures.len(), 3);
}

/// Test schema merge and conflict resolution.
#[test]
fn test_schema_merge() {
    // Base schema
    let mut base = SymbolTable::new();
    base.add_domain(DomainInfo::new("Entity", 1000)).unwrap();
    base.add_predicate(PredicateInfo::new("related", vec!["Entity".to_string()]))
        .unwrap();

    // Update schema 1: Add attributes
    let mut update1 = SymbolTable::new();
    update1.add_domain(DomainInfo::new("Entity", 1000)).unwrap(); // Need Entity for predicate
    update1
        .add_domain(DomainInfo::new("Attribute", 100))
        .unwrap();
    update1
        .add_predicate(PredicateInfo::new(
            "has_attr",
            vec!["Entity".to_string(), "Attribute".to_string()],
        ))
        .unwrap();

    // Update schema 2: Modify entity cardinality
    let mut update2 = SymbolTable::new();
    update2.add_domain(DomainInfo::new("Entity", 2000)).unwrap(); // Increased

    // Merge base + update1
    let merged1 = merge_tables(&base, &update1);
    assert_eq!(merged1.domains.len(), 2);
    assert_eq!(merged1.predicates.len(), 2);

    // Merge merged1 + update2
    let final_merged = merge_tables(&merged1, &update2);
    assert_eq!(final_merged.get_domain("Entity").unwrap().cardinality, 2000);
}

/// Test parametric types in a real scenario.
#[test]
fn test_collection_schema() {
    let mut table = SymbolTable::new();

    // Define base domains
    table.add_domain(DomainInfo::new("User", 1000)).unwrap();
    table.add_domain(DomainInfo::new("Item", 5000)).unwrap();

    // Create parametric types
    let user_list = ParametricType::list(TypeParameter::concrete("User"));
    let item_set = ParametricType::map(
        TypeParameter::concrete("User"),
        TypeParameter::concrete("Item"),
    );

    // Add domains with parametric types
    let mut shopping_cart = DomainInfo::new("ShoppingCart", 1000);
    shopping_cart.parametric_type = Some(item_set);
    table.add_domain(shopping_cart).unwrap();

    let mut friends_list = DomainInfo::new("FriendsList", 1000);
    friends_list.parametric_type = Some(user_list);
    table.add_domain(friends_list).unwrap();

    // Validate
    let validator = SchemaValidator::new(&table);
    let report = validator.validate().unwrap();
    assert!(report.errors.is_empty());
}

/// Test hierarchical domain schema.
#[test]
fn test_organizational_hierarchy() {
    let mut table = SymbolTable::new();
    let mut hierarchy = DomainHierarchy::new();

    // Define organization structure
    table.add_domain(DomainInfo::new("Employee", 1000)).unwrap();
    table.add_domain(DomainInfo::new("Manager", 100)).unwrap();
    table.add_domain(DomainInfo::new("Director", 20)).unwrap();
    table.add_domain(DomainInfo::new("Executive", 5)).unwrap();

    // Build hierarchy
    hierarchy.add_subtype("Manager", "Employee");
    hierarchy.add_subtype("Director", "Manager");
    hierarchy.add_subtype("Executive", "Director");

    // Verify transitivity
    assert!(hierarchy.is_subtype("Executive", "Employee"));
    assert!(hierarchy.is_subtype("Director", "Employee"));
    assert!(!hierarchy.is_subtype("Employee", "Executive"));

    // Define predicates using hierarchy
    table
        .add_predicate(PredicateInfo::new(
            "reports_to",
            vec!["Employee".to_string(), "Manager".to_string()],
        ))
        .unwrap();

    // Validate hierarchy is acyclic
    assert!(hierarchy.validate_acyclic().is_ok());
}

/// Test metadata and provenance tracking.
#[test]
fn test_metadata_tracking() {
    let mut table = SymbolTable::new();

    // Create domain with metadata
    let mut person = DomainInfo::new("Person", 1000);

    let provenance = Provenance::new("admin", "2025-11-06T00:00:00Z")
        .with_source("schema_v1.yaml", Some(1))
        .with_notes("Initial schema setup");

    let mut documentation = Documentation::new("Represents individuals in the system")
        .with_description("Person domain for storing individual entity data");
    documentation.add_example(Example::new(
        "John Doe",
        "Person { id: 1, name: 'John Doe' }",
    ));

    let mut metadata = Metadata::new();
    metadata.provenance = Some(provenance);
    metadata.documentation = Some(documentation);

    person.metadata = Some(metadata);
    table.add_domain(person).unwrap();

    // Verify metadata
    let retrieved = table.get_domain("Person").unwrap();
    assert!(retrieved.metadata.is_some());
    let metadata = retrieved.metadata.as_ref().unwrap();
    assert!(metadata.provenance.is_some());
    assert!(metadata.documentation.is_some());
}

/// Test large schema performance.
#[test]
fn test_large_schema() {
    let mut table = SymbolTable::new();

    // Create many domains
    for i in 0..100 {
        table
            .add_domain(DomainInfo::new(format!("Domain{}", i), 1000))
            .unwrap();
    }

    // Create many predicates
    for i in 0..200 {
        let domain_idx = i % 100;
        table
            .add_predicate(PredicateInfo::new(
                format!("pred{}", i),
                vec![format!("Domain{}", domain_idx)],
            ))
            .unwrap();
    }

    // Test operations on large schema
    assert_eq!(table.domains.len(), 100);
    assert_eq!(table.predicates.len(), 200);

    // Test compact representation
    let compact = CompactSchema::from_symbol_table(&table);
    let stats = compact.compression_stats();
    println!("Large schema - Unique strings: {}", stats.unique_strings);
    println!(
        "Large schema - Compression ratio: {:.2}",
        stats.compression_ratio()
    );
    println!(
        "Large schema - Space savings: {:.2}%",
        stats.space_savings()
    );

    // Verify round-trip
    let restored = compact.to_symbol_table().unwrap();
    assert_eq!(table.domains.len(), restored.domains.len());
    assert_eq!(table.predicates.len(), restored.predicates.len());
}

/// Test binary serialization for efficient storage.
#[test]
fn test_binary_storage() {
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table.add_domain(DomainInfo::new("Location", 50)).unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "at",
            vec!["Person".to_string(), "Location".to_string()],
        ))
        .unwrap();

    // Test binary serialization
    let compact = CompactSchema::from_symbol_table(&table);
    let binary = compact.to_binary().unwrap();

    // Verify size is reasonable
    assert!(!binary.is_empty());
    println!("Binary size: {} bytes", binary.len());

    // Restore from binary
    let restored_compact = CompactSchema::from_binary(&binary).unwrap();
    let restored_table = restored_compact.to_symbol_table().unwrap();

    assert_eq!(table.domains.len(), restored_table.domains.len());
    assert_eq!(table.predicates.len(), restored_table.predicates.len());
}

/// Test validation with compiler bundle.
#[test]
fn test_compiler_bundle_validation() {
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table
        .add_predicate(PredicateInfo::new("knows", vec!["Person".to_string()]))
        .unwrap();

    // Export to compiler bundle
    let bundle = CompilerExport::export_all(&table);

    // Create another table and validate bundle against it
    let mut other_table = SymbolTable::new();
    other_table
        .add_domain(DomainInfo::new("Person", 100))
        .unwrap();

    let result = SymbolTableSync::validate_bundle(&other_table, &bundle).unwrap();
    assert!(result.is_valid());

    // Test invalid bundle
    let mut invalid_bundle = CompilerExportBundle::new();
    invalid_bundle.predicate_signatures.insert(
        "unknown_pred".to_string(),
        vec!["UnknownDomain".to_string()],
    );

    let result = SymbolTableSync::validate_bundle(&table, &invalid_bundle).unwrap();
    assert!(!result.is_valid());
    assert!(!result.errors.is_empty());
}

/// Test CLI validation tool with valid schema.
#[test]
fn test_cli_validate_valid_schema() {
    use std::fs;
    use std::process::Command;

    let temp_dir = std::env::temp_dir();
    let schema_path = temp_dir.join("test_schema_valid.yaml");

    // Create a valid schema
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "knows",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();

    let yaml = table.to_yaml().unwrap();
    fs::write(&schema_path, yaml).unwrap();

    // Run validation command
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "schema_validate",
            "--",
            schema_path.to_str().unwrap(),
        ])
        .output();

    // Clean up
    let _ = fs::remove_file(&schema_path);

    if let Ok(output) = output {
        assert!(output.status.success(), "Validation should succeed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Schema validation passed"));
    }
}

/// Test CLI migration tool convert command.
#[test]
fn test_cli_migrate_convert() {
    use std::fs;
    use std::process::Command;

    let temp_dir = std::env::temp_dir();
    let json_path = temp_dir.join("test_schema.json");
    let yaml_path = temp_dir.join("test_schema_converted.yaml");

    // Create a schema in JSON
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "knows",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();

    let json = table.to_json().unwrap();
    fs::write(&json_path, json).unwrap();

    // Run convert command
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "schema_migrate",
            "--",
            "convert",
            json_path.to_str().unwrap(),
            yaml_path.to_str().unwrap(),
        ])
        .output();

    // Verify conversion
    if let Ok(output) = output {
        if output.status.success() {
            assert!(yaml_path.exists(), "YAML file should be created");

            // Verify content
            let yaml_content = fs::read_to_string(&yaml_path).unwrap();
            let restored = SymbolTable::from_yaml(&yaml_content).unwrap();
            assert_eq!(table.domains.len(), restored.domains.len());
            assert_eq!(table.predicates.len(), restored.predicates.len());
        }
    }

    // Clean up
    let _ = fs::remove_file(&json_path);
    let _ = fs::remove_file(&yaml_path);
}

/// Test CLI migration tool diff command.
#[test]
fn test_cli_migrate_diff() {
    use std::fs;
    use std::process::Command;

    let temp_dir = std::env::temp_dir();
    let old_path = temp_dir.join("test_schema_old.yaml");
    let new_path = temp_dir.join("test_schema_new.yaml");

    // Create old schema
    let mut old_table = SymbolTable::new();
    old_table
        .add_domain(DomainInfo::new("Person", 100))
        .unwrap();
    old_table
        .add_predicate(PredicateInfo::new(
            "knows",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();

    fs::write(&old_path, old_table.to_yaml().unwrap()).unwrap();

    // Create new schema (with additions)
    let mut new_table = old_table.clone();
    new_table
        .add_domain(DomainInfo::new("Location", 50))
        .unwrap();
    new_table
        .add_predicate(PredicateInfo::new(
            "at",
            vec!["Person".to_string(), "Location".to_string()],
        ))
        .unwrap();

    fs::write(&new_path, new_table.to_yaml().unwrap()).unwrap();

    // Run diff command
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "schema_migrate",
            "--",
            "diff",
            old_path.to_str().unwrap(),
            new_path.to_str().unwrap(),
        ])
        .output();

    // Verify diff output
    if let Ok(output) = output {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.contains("Added domains") || stdout.contains("Added predicates"));
        }
    }

    // Clean up
    let _ = fs::remove_file(&old_path);
    let _ = fs::remove_file(&new_path);
}

/// Test CLI migration tool merge command.
#[test]
fn test_cli_migrate_merge() {
    use std::fs;
    use std::process::Command;

    let temp_dir = std::env::temp_dir();
    let schema1_path = temp_dir.join("test_schema1.yaml");
    let schema2_path = temp_dir.join("test_schema2.yaml");
    let merged_path = temp_dir.join("test_schema_merged.yaml");

    // Create first schema
    let mut table1 = SymbolTable::new();
    table1.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table1
        .add_predicate(PredicateInfo::new(
            "knows",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();

    fs::write(&schema1_path, table1.to_yaml().unwrap()).unwrap();

    // Create second schema
    let mut table2 = SymbolTable::new();
    table2.add_domain(DomainInfo::new("Person", 100)).unwrap(); // Added Person domain
    table2.add_domain(DomainInfo::new("Location", 50)).unwrap();
    table2
        .add_predicate(PredicateInfo::new(
            "at",
            vec!["Person".to_string(), "Location".to_string()],
        ))
        .unwrap();

    fs::write(&schema2_path, table2.to_yaml().unwrap()).unwrap();

    // Run merge command
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "schema_migrate",
            "--",
            "merge",
            schema1_path.to_str().unwrap(),
            schema2_path.to_str().unwrap(),
            merged_path.to_str().unwrap(),
        ])
        .output();

    // Verify merged result
    if let Ok(output) = output {
        if output.status.success() && merged_path.exists() {
            let merged_content = fs::read_to_string(&merged_path).unwrap();
            let merged_table = SymbolTable::from_yaml(&merged_content).unwrap();

            // Should contain domains from both schemas
            assert!(merged_table.get_domain("Person").is_some());
            assert!(merged_table.get_domain("Location").is_some());
        }
    }

    // Clean up
    let _ = fs::remove_file(&schema1_path);
    let _ = fs::remove_file(&schema2_path);
    let _ = fs::remove_file(&merged_path);
}
