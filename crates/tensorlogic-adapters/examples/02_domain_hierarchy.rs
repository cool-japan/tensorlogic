//! Domain hierarchy and subtyping example.
//!
//! This example demonstrates how to define and use domain hierarchies
//! with subtype relationships.

use tensorlogic_adapters::{DomainHierarchy, DomainInfo, SymbolTable};

fn main() -> anyhow::Result<()> {
    println!("=== Domain Hierarchy Example ===\n");

    // Create a domain hierarchy
    let mut hierarchy = DomainHierarchy::new();

    // Define a type hierarchy for agents
    println!("Building agent hierarchy:");
    println!("  Agent");
    println!("    ├── Person");
    println!("    │   ├── Student");
    println!("    │   └── Teacher");
    println!("    └── Organization\n");

    hierarchy.add_subtype("Student", "Person");
    hierarchy.add_subtype("Teacher", "Person");
    hierarchy.add_subtype("Person", "Agent");
    hierarchy.add_subtype("Organization", "Agent");

    // Test subtype relationships
    println!("Testing subtype relationships:");
    println!(
        "  - Is Student a subtype of Person? {}",
        hierarchy.is_subtype("Student", "Person")
    );
    println!(
        "  - Is Student a subtype of Agent? {} (transitive)",
        hierarchy.is_subtype("Student", "Agent")
    );
    println!(
        "  - Is Student a subtype of Teacher? {}",
        hierarchy.is_subtype("Student", "Teacher")
    );
    println!(
        "  - Is Organization a subtype of Agent? {}\n",
        hierarchy.is_subtype("Organization", "Agent")
    );

    // Find common supertypes
    println!("Finding common supertypes:");
    if let Some(common) = hierarchy.least_common_supertype("Student", "Teacher") {
        println!("  - LCS(Student, Teacher) = {}", common);
    }
    if let Some(common) = hierarchy.least_common_supertype("Student", "Organization") {
        println!("  - LCS(Student, Organization) = {}", common);
    }
    println!();

    // Query ancestors and descendants
    println!("Querying hierarchy:");
    let ancestors = hierarchy.get_ancestors("Student");
    println!("  - Ancestors of Student: {:?}", ancestors);

    let descendants = hierarchy.get_descendants("Person");
    println!("  - Descendants of Person: {:?}", descendants);
    println!();

    // Validate the hierarchy
    println!("Validating hierarchy...");
    match hierarchy.validate_acyclic() {
        Ok(_) => println!("  ✓ Hierarchy is valid (no cycles)\n"),
        Err(e) => println!("  ✗ Hierarchy validation failed: {}\n", e),
    }

    // Integration with symbol table
    println!("=== Integration with Symbol Table ===\n");
    let mut table = SymbolTable::new();

    // Add domains to symbol table
    table.add_domain(DomainInfo::new("Agent", 200))?;
    table.add_domain(DomainInfo::new("Person", 150))?;
    table.add_domain(DomainInfo::new("Student", 50))?;
    table.add_domain(DomainInfo::new("Teacher", 30))?;
    table.add_domain(DomainInfo::new("Organization", 40))?;

    println!("Symbol table created with {} domains", table.domains.len());

    // Use hierarchy for type checking
    println!("\nType checking with hierarchy:");
    if hierarchy.is_subtype("Student", "Person") {
        println!("  ✓ Student can be used where Person is expected");
    }
    if hierarchy.is_subtype("Student", "Agent") {
        println!("  ✓ Student can be used where Agent is expected");
    }

    Ok(())
}
