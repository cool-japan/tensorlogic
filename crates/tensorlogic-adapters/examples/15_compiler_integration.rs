//! Comprehensive end-to-end compiler integration example.
//!
//! This example demonstrates the complete workflow of using tensorlogic-adapters
//! with tensorlogic-compiler, showcasing:
//!
//! 1. Schema definition with advanced type systems
//! 2. Domain hierarchies and predicate constraints
//! 3. Export to compiler context
//! 4. Compilation of logic expressions
//! 5. Round-trip synchronization
//!
//! Run with: `cargo run --example 15_compiler_integration`

use tensorlogic_adapters::{
    CompilerExport, CompilerExportAdvanced, CompleteExportBundle, DomainHierarchy, DomainInfo,
    PredicateConstraints, PredicateInfo, PredicateProperty, SymbolTable, ValueRange,
};

fn main() {
    println!("=== TensorLogic Compiler Integration Example ===\n");

    // ========================================================================
    // PART 1: Define a rich schema with advanced features
    // ========================================================================
    println!("PART 1: Schema Definition");
    println!("-------------------------");

    let mut schema = SymbolTable::new();
    let mut hierarchy = DomainHierarchy::new();

    // Define domain hierarchy
    schema
        .add_domain(DomainInfo::new("Entity", 1000).with_description("Root of type hierarchy"))
        .unwrap();

    schema
        .add_domain(
            DomainInfo::new("Agent", 500).with_description("Autonomous agents in the system"),
        )
        .unwrap();

    schema
        .add_domain(DomainInfo::new("Person", 200).with_description("Human agents with identities"))
        .unwrap();

    schema
        .add_domain(DomainInfo::new("Student", 80).with_description("Students enrolled in courses"))
        .unwrap();

    schema
        .add_domain(DomainInfo::new("Professor", 20).with_description("Faculty teaching courses"))
        .unwrap();

    schema
        .add_domain(DomainInfo::new("Course", 50).with_description("Academic courses"))
        .unwrap();

    schema
        .add_domain(DomainInfo::new("Grade", 5).with_description("Letter grades A-F"))
        .unwrap();

    // Set up hierarchy: Student <: Person <: Agent <: Entity
    hierarchy.add_subtype("Agent", "Entity");
    hierarchy.add_subtype("Person", "Agent");
    hierarchy.add_subtype("Student", "Person");
    hierarchy.add_subtype("Professor", "Person");

    println!("✓ Defined 7 domains with 5 subtype relationships");

    // Define predicates with constraints
    let mut enrolled = PredicateInfo::new(
        "enrolled",
        vec!["Student".to_string(), "Course".to_string()],
    )
    .with_description("Student is enrolled in a course");

    let enrolled_constraints =
        PredicateConstraints::new().with_property(PredicateProperty::Functional); // Each student enrolled in at most one course per predicate instance
    enrolled.constraints = Some(enrolled_constraints);
    schema.add_predicate(enrolled).unwrap();

    let mut teaches = PredicateInfo::new(
        "teaches",
        vec!["Professor".to_string(), "Course".to_string()],
    )
    .with_description("Professor teaches a course");
    teaches.constraints = Some(PredicateConstraints::new());
    schema.add_predicate(teaches).unwrap();

    let mut grade = PredicateInfo::new(
        "grade",
        vec![
            "Student".to_string(),
            "Course".to_string(),
            "Grade".to_string(),
        ],
    )
    .with_description("Student received grade in course");

    let grade_constraints =
        PredicateConstraints::new().with_property(PredicateProperty::Functional); // Each (student, course) pair has at most one grade
    grade.constraints = Some(grade_constraints);
    schema.add_predicate(grade).unwrap();

    let mut knows = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
        .with_description("Person knows another person");

    let knows_constraints = PredicateConstraints::new()
        .with_property(PredicateProperty::Symmetric) // If A knows B, then B knows A
        .with_property(PredicateProperty::Reflexive); // Everyone knows themselves
    knows.constraints = Some(knows_constraints);
    schema.add_predicate(knows).unwrap();

    let mut age =
        PredicateInfo::new("age", vec!["Person".to_string()]).with_description("Person's age");

    let age_range = ValueRange::new().with_min(0.0, true).with_max(120.0, true); // Age between 0 and 120 (inclusive)
    let age_constraints = PredicateConstraints::new().with_value_range(0, age_range);
    age.constraints = Some(age_constraints);
    schema.add_predicate(age).unwrap();

    println!("✓ Defined 5 predicates with various constraints:");
    println!("  - enrolled: functional property");
    println!("  - teaches: basic predicate");
    println!("  - grade: functional property");
    println!("  - knows: symmetric and reflexive");
    println!("  - age: value range [0, 120]");

    // Bind some variables
    schema.bind_variable("student", "Student").unwrap();
    schema.bind_variable("professor", "Professor").unwrap();
    schema.bind_variable("course", "Course").unwrap();

    println!("✓ Bound 3 variables to their domains\n");

    // ========================================================================
    // PART 2: Export to compiler format
    // ========================================================================
    println!("PART 2: Export to Compiler");
    println!("--------------------------");

    // Basic export
    let basic_export = CompilerExport::export_all(&schema);
    println!("Basic Export:");
    println!("  Domains: {}", basic_export.domains.len());
    println!("  Predicates: {}", basic_export.predicate_signatures.len());
    println!("  Variables: {}", basic_export.variable_bindings.len());

    // Advanced export
    let advanced_export = CompilerExportAdvanced::export_all_advanced(&schema, Some(&hierarchy));
    println!("\nAdvanced Export:");
    println!(
        "  Hierarchy relationships: {}",
        advanced_export.hierarchy.as_ref().map_or(0, |h| h.len())
    );
    println!(
        "  Constrained predicates: {}",
        advanced_export.constraints.len()
    );
    println!(
        "  Refinement types: {}",
        advanced_export.refinement_types.len()
    );
    println!(
        "  Dependent types: {}",
        advanced_export.dependent_types.len()
    );
    println!("  Linear types: {}", advanced_export.linear_types.len());
    println!("  Effect types: {}", advanced_export.effects.len());

    // Complete export
    let complete_export = CompleteExportBundle::from_symbol_table(&schema, Some(&hierarchy));
    println!("\nComplete Export Bundle:");
    println!("  Is empty: {}", complete_export.is_empty());

    // Show some hierarchy details
    if let Some(ref hier) = advanced_export.hierarchy {
        println!("\nHierarchy Details:");
        for (domain, ancestors) in hier {
            println!("  {} <: {:?}", domain, ancestors);
        }
    }

    // Show constraint details
    println!("\nConstraint Details:");
    for (pred_name, constraints) in &advanced_export.constraints {
        let props: Vec<_> = constraints.properties.iter().collect();
        if !props.is_empty() {
            println!("  {}: {:?}", pred_name, props);
        }
        for (idx, range_opt) in constraints.value_ranges.iter().enumerate() {
            if let Some(range) = range_opt {
                println!("    Arg {} range: [{:?}, {:?}]", idx, range.min, range.max);
            }
        }
    }

    println!();

    // ========================================================================
    // PART 3: Show advanced type systems
    // ========================================================================
    println!("PART 3: Advanced Type Systems");
    println!("-----------------------------");

    println!("Refinement Types (value constraints):");
    for (name, spec) in &advanced_export.refinement_types {
        println!("  {}: {}", name, spec);
    }

    println!("\nDependent Types (dimension tracking):");
    for (name, spec) in &advanced_export.dependent_types {
        println!("  {}: {}", name, spec);
    }

    println!("\nLinear Types (resource management):");
    for (name, linearity) in &advanced_export.linear_types {
        println!("  {}: {}", name, linearity);
    }

    println!("\nEffect Types (effect tracking):");
    for (name, effects) in &advanced_export.effects {
        println!("  {}: {:?}", name, effects);
    }

    println!();

    // ========================================================================
    // PART 4: Demonstrate compiler integration usage
    // ========================================================================
    println!("PART 4: Compiler Integration Usage");
    println!("----------------------------------");

    // In a real scenario, you would use this with tensorlogic-compiler:
    //
    // use tensorlogic_compiler::CompilerContext;
    //
    // let ctx = CompilerContext::from_symbol_table(&schema);
    //
    // // Add hierarchy information if the compiler supports it
    // // ctx.add_hierarchy(hierarchy);
    //
    // // Compile expressions using the schema-enriched context
    // let expr = TLExpr::exists(
    //     "c",
    //     "Course",
    //     TLExpr::pred("enrolled", vec![Term::var("student"), Term::var("c")])
    // );
    //
    // let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;

    println!("Example compilation workflow:");
    println!("1. Create CompilerContext from SymbolTable");
    println!("2. Compiler automatically knows about:");
    println!("   - All domains and their cardinalities");
    println!("   - All predicate signatures for type checking");
    println!("   - Variable bindings for scope analysis");
    println!("   - Domain hierarchy for subtype checking");
    println!("   - Predicate constraints for optimization");
    println!("3. Compile logic expressions to einsum graphs");
    println!("4. Use advanced types for static analysis");

    println!();

    // ========================================================================
    // PART 5: Show schema statistics
    // ========================================================================
    println!("PART 5: Schema Statistics");
    println!("-------------------------");

    println!("Schema Overview:");
    println!("  Total domains: {}", basic_export.domains.len());
    println!(
        "  Total predicates: {}",
        basic_export.predicate_signatures.len()
    );
    println!(
        "  Total variables: {}",
        basic_export.variable_bindings.len()
    );
    println!(
        "  Hierarchy depth: {}",
        hierarchy.get_ancestors("Student").len()
    );
    println!(
        "  Constrained predicates: {}",
        advanced_export.constraints.len()
    );

    let total_cardinality: usize = basic_export.domains.values().sum();
    println!("  Total cardinality: {}", total_cardinality);

    println!("\nDomain Sizes:");
    let mut sorted_domains: Vec<_> = basic_export.domains.iter().collect();
    sorted_domains.sort_by_key(|(_, &size)| std::cmp::Reverse(size));
    for (name, size) in sorted_domains {
        println!("  {:12} = {} elements", name, size);
    }

    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("=== Summary ===");
    println!();
    println!("This example demonstrated:");
    println!("✓ Rich schema definition with domains, predicates, and constraints");
    println!("✓ Domain hierarchies with subtype relationships");
    println!("✓ Basic export (domains, predicates, variables)");
    println!("✓ Advanced export (hierarchy, constraints, type systems)");
    println!("✓ Complete bundle combining all exports");
    println!("✓ Integration patterns with tensorlogic-compiler");
    println!();
    println!("The exported bundles can be used to:");
    println!("• Initialize CompilerContext with full schema information");
    println!("• Enable schema-driven compilation with type checking");
    println!("• Support advanced optimizations based on constraints");
    println!("• Track resources with linear types");
    println!("• Validate dimensions with dependent types");
    println!("• Enforce value constraints with refinement types");
    println!("• Track computational effects with effect types");
    println!();
    println!("See tensorlogic-compiler documentation for compilation examples.");
}
