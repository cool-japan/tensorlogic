//! Example 09: Product Domains for Cross-Domain Reasoning
//!
//! This example demonstrates how to use product domains to create
//! composite types from multiple base domains, enabling cross-domain
//! reasoning and relational predicates.
//!
//! Run with: cargo run --example 09_product_domains

use anyhow::Result;
use tensorlogic_adapters::{
    DomainInfo, PredicateInfo, ProductDomain, ProductDomainExt, SchemaBuilder, SymbolTable,
};

fn main() -> Result<()> {
    println!("=== Product Domains Example ===\n");

    // Example 1: Basic Binary Product
    println!("1. Basic Binary Product (Person × Location)");
    println!("{}", "=".repeat(50));

    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 1000))?;
    table.add_domain(DomainInfo::new("Location", 500))?;

    // Create a product domain
    let person_at_location = ProductDomain::binary("Person", "Location");
    println!("Product: {}", person_at_location);
    println!("Arity: {}", person_at_location.arity());
    println!("Cardinality: {}", person_at_location.cardinality(&table)?);

    // Add to symbol table
    table.add_product_domain("PersonAtLocation", person_at_location)?;
    let domain = table.get_domain("PersonAtLocation").unwrap();
    println!("Added to table with cardinality: {}\n", domain.cardinality);

    // Example 2: Ternary Product
    println!("2. Ternary Product (Person × Location × Time)");
    println!("{}", "=".repeat(50));

    table.add_domain(DomainInfo::new("Time", 24))?; // 24 hours

    let person_at_location_time = ProductDomain::ternary("Person", "Location", "Time");
    println!("Product: {}", person_at_location_time);
    println!(
        "Cardinality: {} (1000 × 500 × 24)",
        person_at_location_time.cardinality(&table)?
    );

    table.add_product_domain("PersonAtLocationTime", person_at_location_time)?;
    println!("✓ Added PersonAtLocationTime domain\n");

    // Example 3: Projection from Products
    println!("3. Projection from Product Domains");
    println!("{}", "=".repeat(50));

    let product = ProductDomain::new(vec![
        "Person".to_string(),
        "Location".to_string(),
        "Time".to_string(),
    ]);

    // Project to individual components
    for i in 0..product.arity() {
        if let Some(component) = product.project(i) {
            println!("π{}: {}", i, component);
        }
    }
    println!();

    // Example 4: Slicing Products
    println!("4. Slicing Product Domains");
    println!("{}", "=".repeat(50));

    table.add_domain(DomainInfo::new("Action", 10))?;

    let full_product = ProductDomain::new(vec![
        "Person".to_string(),
        "Location".to_string(),
        "Time".to_string(),
        "Action".to_string(),
    ]);
    println!("Full product: {}", full_product);

    // Get middle two components (Location × Time)
    let slice = full_product.slice(1, 3)?;
    println!("Slice [1:3]: {}", slice);
    println!("Slice cardinality: {}\n", slice.cardinality(&table)?);

    // Example 5: Predicates over Product Domains
    println!("5. Predicates over Product Domains");
    println!("{}", "=".repeat(50));

    // Define predicates using product domains
    table.add_predicate(PredicateInfo::new(
        "at",
        vec!["Person".to_string(), "Location".to_string()],
    ))?;

    table.add_predicate(PredicateInfo::new(
        "scheduled",
        vec![
            "Person".to_string(),
            "Location".to_string(),
            "Time".to_string(),
        ],
    ))?;

    println!("✓ Defined predicate: at(Person, Location)");
    println!("✓ Defined predicate: scheduled(Person, Location, Time)");
    println!();

    // Example 6: Nested Products
    println!("6. Nested Product Domains");
    println!("{}", "=".repeat(50));

    // Create (Person × Location)
    let pl_product = ProductDomain::binary("Person", "Location");
    table.add_product_domain("PL", pl_product)?;

    // Create (PL × Time) - nested product
    let nested = ProductDomain::binary("PL", "Time");
    println!("Nested product: {}", nested);
    println!(
        "Cardinality: {} (same as Person × Location × Time)",
        nested.cardinality(&table)?
    );
    println!();

    // Example 7: Dynamic Product Extension
    println!("7. Dynamic Product Extension");
    println!("{}", "=".repeat(50));

    let mut dynamic_product = ProductDomain::binary("Person", "Location");
    println!("Initial: {}", dynamic_product);

    // Extend with additional components
    dynamic_product.extend(vec!["Time".to_string(), "Action".to_string()]);
    println!("Extended: {}", dynamic_product);
    println!("New arity: {}\n", dynamic_product.arity());

    // Example 8: Real-World Use Case - Event Tracking
    println!("8. Real-World Use Case: Event Tracking System");
    println!("{}", "=".repeat(50));

    let event_table = SchemaBuilder::new()
        .domain("User", 10000)
        .domain("Resource", 5000)
        .domain("Operation", 20)
        .domain("Timestamp", 86400) // Seconds in a day
        .domain("Location", 100)
        .build()?;

    // Define event predicates using products
    let access_event = ProductDomain::new(vec![
        "User".to_string(),
        "Resource".to_string(),
        "Operation".to_string(),
        "Timestamp".to_string(),
    ]);

    let geo_event = ProductDomain::new(vec![
        "User".to_string(),
        "Location".to_string(),
        "Timestamp".to_string(),
    ]);

    println!("Access Event Schema:");
    println!("  Product: {}", access_event);
    println!(
        "  Cardinality: {} events possible",
        access_event.cardinality(&event_table)?
    );

    println!("\nGeo-Location Event Schema:");
    println!("  Product: {}", geo_event);
    println!(
        "  Cardinality: {} events possible",
        geo_event.cardinality(&event_table)?
    );
    println!();

    // Example 9: Validation and Error Handling
    println!("9. Validation and Error Handling");
    println!("{}", "=".repeat(50));

    let mut test_table = SymbolTable::new();
    test_table.add_domain(DomainInfo::new("A", 10))?;

    // Try to create product with unknown domain
    let invalid_product = ProductDomain::binary("A", "Unknown");
    match invalid_product.validate(&test_table) {
        Ok(_) => println!("✗ Should have failed validation"),
        Err(e) => println!("✓ Validation caught error: {}", e),
    }

    // Valid product
    test_table.add_domain(DomainInfo::new("B", 20))?;
    let valid_product = ProductDomain::binary("A", "B");
    match valid_product.validate(&test_table) {
        Ok(_) => println!("✓ Valid product passed validation"),
        Err(e) => println!("✗ Unexpected error: {}", e),
    }
    println!();

    // Example 10: Performance Considerations
    println!("10. Performance Considerations");
    println!("{}", "=".repeat(50));

    let perf_table = SchemaBuilder::new()
        .domain("D1", 100)
        .domain("D2", 100)
        .domain("D3", 100)
        .domain("D4", 100)
        .build()?;

    let large_product = ProductDomain::new(vec![
        "D1".to_string(),
        "D2".to_string(),
        "D3".to_string(),
        "D4".to_string(),
    ]);

    let cardinality = large_product.cardinality(&perf_table)?;
    println!("Product: {}", large_product);
    println!("Cardinality: {} (100^4)", cardinality);
    println!(
        "Memory overhead: ~{} bytes (just component names)",
        large_product.components().len() * 8
    );
    println!();

    println!("=== Summary ===");
    println!("✓ Created binary, ternary, and n-ary products");
    println!("✓ Demonstrated projection and slicing");
    println!("✓ Used products with predicates");
    println!("✓ Showed nested products");
    println!("✓ Dynamic extension of products");
    println!("✓ Real-world event tracking use case");
    println!("✓ Validation and error handling");
    println!("✓ Performance analysis");

    Ok(())
}
