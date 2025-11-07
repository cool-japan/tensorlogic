//! Example 10: Computed Domains for Virtual Types
//!
//! This example demonstrates how to use computed domains to create
//! virtual domains derived from operations on existing domains,
//! enabling lazy evaluation and complex domain transformations.
//!
//! Run with: cargo run --example 10_computed_domains

use anyhow::Result;
use tensorlogic_adapters::{
    ComputedDomain, ComputedDomainRegistry, DomainComputation, DomainInfo, SchemaBuilder,
};

fn main() -> Result<()> {
    println!("=== Computed Domains Example ===\n");

    // Example 1: Filter Operation
    println!("1. Filter: Create Subset Domains");
    println!("{}", "=".repeat(50));

    let mut table = SchemaBuilder::new()
        .domain("Person", 1000)
        .predicate("is_adult", vec!["Person"])
        .predicate("is_student", vec!["Person"])
        .build()?;

    // Define Adults as filtered subset of Person
    let adults = ComputedDomain::new(
        "Adults",
        DomainComputation::Filter {
            base: "Person".to_string(),
            predicate: "is_adult".to_string(),
        },
    )
    .with_cardinality_estimate(750); // Estimate 75% are adults

    println!("Computed domain: {}", adults);
    println!("Validation: {:?}", adults.validate(&table));

    let (lower, upper) = adults.cardinality_bounds(&table)?;
    println!("Cardinality bounds: [{}, {}]", lower, upper);
    println!("Estimate: {:?}\n", adults.cardinality_estimate());

    // Example 2: Union Operation
    println!("2. Union: Combine Multiple Domains");
    println!("{}", "=".repeat(50));

    table.add_domain(DomainInfo::new("Organization", 500))?;
    table.add_domain(DomainInfo::new("Robot", 100))?;

    let agents = ComputedDomain::new(
        "Agents",
        DomainComputation::Union {
            domains: vec![
                "Person".to_string(),
                "Organization".to_string(),
                "Robot".to_string(),
            ],
        },
    );

    println!("Computed domain: {}", agents);
    let (lower, upper) = agents.cardinality_bounds(&table)?;
    println!("Cardinality bounds: [{}, {}]", lower, upper);
    println!("  Lower bound = max(1000, 500, 100) = {}", lower);
    println!("  Upper bound = 1000 + 500 + 100 = {}\n", upper);

    // Example 3: Intersection Operation
    println!("3. Intersection: Find Common Elements");
    println!("{}", "=".repeat(50));

    table.add_domain(DomainInfo::new("Employee", 800))?;
    table.add_domain(DomainInfo::new("Manager", 200))?;

    let employee_managers = ComputedDomain::new(
        "EmployeeManagers",
        DomainComputation::Intersection {
            domains: vec!["Employee".to_string(), "Manager".to_string()],
        },
    );

    println!("Computed domain: {}", employee_managers);
    let (lower, upper) = employee_managers.cardinality_bounds(&table)?;
    println!("Cardinality bounds: [{}, {}]", lower, upper);
    println!("  Upper bound = min(800, 200) = {}\n", upper);

    // Example 4: Difference Operation
    println!("4. Difference: Set Subtraction");
    println!("{}", "=".repeat(50));

    let non_managers = ComputedDomain::new(
        "NonManagers",
        DomainComputation::Difference {
            base: "Employee".to_string(),
            subtract: "Manager".to_string(),
        },
    );

    println!("Computed domain: {}", non_managers);
    let (lower, upper) = non_managers.cardinality_bounds(&table)?;
    println!("Cardinality bounds: [{}, {}]", lower, upper);
    println!("  Upper bound = 800 - 200 = {}\n", upper);

    // Example 5: Product Operation
    println!("5. Product: Cartesian Product via Computation");
    println!("{}", "=".repeat(50));

    table.add_domain(DomainInfo::new("Location", 50))?;

    let person_location_pairs = ComputedDomain::new(
        "PersonLocationPairs",
        DomainComputation::Product {
            domains: vec!["Person".to_string(), "Location".to_string()],
        },
    );

    println!("Computed domain: {}", person_location_pairs);
    let (lower, _upper) = person_location_pairs.cardinality_bounds(&table)?;
    println!("Cardinality: {} (1000 × 50)\n", lower);

    // Example 6: PowerSet Operation
    println!("6. PowerSet: All Subsets");
    println!("{}", "=".repeat(50));

    table.add_domain(DomainInfo::new("SmallSet", 5))?;

    let power_set = ComputedDomain::new(
        "PowerSetOfSmallSet",
        DomainComputation::PowerSet {
            base: "SmallSet".to_string(),
        },
    );

    println!("Computed domain: {}", power_set);
    let (lower, _upper) = power_set.cardinality_bounds(&table)?;
    println!("Cardinality: {} (2^5)\n", lower);

    // Example 7: Projection Operation
    println!("7. Projection: Extract Component");
    println!("{}", "=".repeat(50));

    table.add_domain(DomainInfo::new("PersonLocationProduct", 50000))?;

    let projection = ComputedDomain::new(
        "ProjectedPerson",
        DomainComputation::Projection {
            product: "PersonLocationProduct".to_string(),
            index: 0,
        },
    );

    println!("Computed domain: {}", projection);
    println!(
        "Cardinality bounds: {:?}\n",
        projection.cardinality_bounds(&table)?
    );

    // Example 8: Custom Computation
    println!("8. Custom: User-Defined Computation");
    println!("{}", "=".repeat(50));

    let custom = ComputedDomain::new(
        "ActiveUsers",
        DomainComputation::Custom {
            description: "Users active in last 30 days".to_string(),
            formula: "SELECT user_id FROM activity WHERE timestamp > NOW() - 30d".to_string(),
        },
    )
    .with_cardinality_estimate(8500);

    println!("Computed domain: {}", custom);
    println!("Estimate: {:?}\n", custom.cardinality_estimate());

    // Example 9: ComputedDomainRegistry
    println!("9. Managing Computed Domains with Registry");
    println!("{}", "=".repeat(50));

    let mut registry = ComputedDomainRegistry::new();

    // Register multiple computed domains
    registry.register(adults)?;
    registry.register(agents)?;
    registry.register(non_managers)?;

    println!("Registered {} computed domains", registry.len());
    for domain in registry.list() {
        println!("  - {}", domain.name());
    }
    println!();

    // Validate all at once
    match registry.validate_all(&table) {
        Ok(_) => println!("✓ All computed domains validated successfully"),
        Err(errors) => {
            println!("✗ Validation errors:");
            for err in errors {
                println!("  - {}", err);
            }
        }
    }
    println!();

    // Example 10: Real-World Use Case - Access Control
    println!("10. Real-World Use Case: Role-Based Access Control");
    println!("{}", "=".repeat(50));

    let _rbac_table = SchemaBuilder::new()
        .domain("User", 10000)
        .domain("Resource", 5000)
        .domain("Role", 10)
        .predicate("has_role", vec!["User", "Role"])
        .predicate("can_access", vec!["Role", "Resource"])
        .build()?;

    let mut rbac_registry = ComputedDomainRegistry::new();

    // Define role-specific user domains
    let admins = ComputedDomain::new(
        "Admins",
        DomainComputation::Filter {
            base: "User".to_string(),
            predicate: "is_admin".to_string(),
        },
    )
    .with_cardinality_estimate(100);

    let editors = ComputedDomain::new(
        "Editors",
        DomainComputation::Filter {
            base: "User".to_string(),
            predicate: "is_editor".to_string(),
        },
    )
    .with_cardinality_estimate(500);

    let viewers = ComputedDomain::new(
        "Viewers",
        DomainComputation::Filter {
            base: "User".to_string(),
            predicate: "is_viewer".to_string(),
        },
    )
    .with_cardinality_estimate(9400);

    // Privileged users (Admins ∪ Editors) - not registered for simplicity
    let _privileged = ComputedDomain::new(
        "PrivilegedUsers",
        DomainComputation::Union {
            domains: vec!["Admins".to_string(), "Editors".to_string()],
        },
    );

    rbac_registry.register(admins)?;
    rbac_registry.register(editors)?;
    rbac_registry.register(viewers)?;

    println!("RBAC Domain Hierarchy:");
    println!("  Users (10000)");
    println!("  ├── Admins (~100)");
    println!("  ├── Editors (~500)");
    println!("  └── Viewers (~9400)");
    println!("\nComputed Domains:");
    for domain in rbac_registry.list() {
        if let Some(est) = domain.cardinality_estimate() {
            println!("  - {}: ~{} users", domain.name(), est);
        }
    }
    println!();

    // Example 11: Chaining Computations
    println!("11. Chaining Computed Domains");
    println!("{}", "=".repeat(50));

    let _chain_table = SchemaBuilder::new()
        .domain("AllPeople", 1000)
        .predicate("is_adult", vec!["AllPeople"])
        .predicate("is_employed", vec!["AllPeople"])
        .build()?;

    let mut chain_registry = ComputedDomainRegistry::new();

    // Step 1: Filter adults
    let adults_chain = ComputedDomain::new(
        "Adults",
        DomainComputation::Filter {
            base: "AllPeople".to_string(),
            predicate: "is_adult".to_string(),
        },
    )
    .with_cardinality_estimate(800);

    chain_registry.register(adults_chain)?;

    // Step 2: Filter employed adults (if we could reference Adults)
    println!("Domain: AllPeople (1000)");
    println!("  → Filter(is_adult) → Adults (~800)");
    println!("  → Filter(is_employed) → EmployedAdults (~600)");
    println!("Chaining enables progressive refinement\n");

    println!("=== Summary ===");
    println!("✓ Filter operation for subsets");
    println!("✓ Union for combining domains");
    println!("✓ Intersection for common elements");
    println!("✓ Difference for set subtraction");
    println!("✓ Product for Cartesian products");
    println!("✓ PowerSet for all subsets");
    println!("✓ Projection from products");
    println!("✓ Custom computations");
    println!("✓ ComputedDomainRegistry management");
    println!("✓ Real-world RBAC use case");
    println!("✓ Chained computations");

    Ok(())
}
