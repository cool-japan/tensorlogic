//! GraphQL Schema Generation Example.
//!
//! This example demonstrates how to generate GraphQL schemas from TensorLogic
//! symbol tables, enabling type-safe API development.
//!
//! Run with: `cargo run --example 16_graphql_codegen`

use tensorlogic_adapters::{DomainInfo, GraphQLCodegen, PredicateInfo, SymbolTable};

fn main() {
    println!("=== TensorLogic GraphQL Schema Generation ===\n");

    // Create a symbol table for an academic domain
    let mut schema = SymbolTable::new();

    // Define domains
    schema
        .add_domain(
            DomainInfo::new("Student", 1000)
                .with_description("Students enrolled in the university"),
        )
        .unwrap();

    schema
        .add_domain(DomainInfo::new("Professor", 100).with_description("Faculty members"))
        .unwrap();

    schema
        .add_domain(DomainInfo::new("Course", 200).with_description("Available courses"))
        .unwrap();

    schema
        .add_domain(DomainInfo::new("Grade", 5).with_description("Letter grades A through F"))
        .unwrap();

    println!("✓ Defined 4 domains");

    // Define predicates
    let enrolled = PredicateInfo::new(
        "enrolled",
        vec!["Student".to_string(), "Course".to_string()],
    )
    .with_description("Student is enrolled in a course");
    schema.add_predicate(enrolled).unwrap();

    let teaches = PredicateInfo::new(
        "teaches",
        vec!["Professor".to_string(), "Course".to_string()],
    )
    .with_description("Professor teaches a course");
    schema.add_predicate(teaches).unwrap();

    let grade_predicate = PredicateInfo::new(
        "course_grade",
        vec![
            "Student".to_string(),
            "Course".to_string(),
            "Grade".to_string(),
        ],
    )
    .with_description("Grade received by student in course");
    schema.add_predicate(grade_predicate).unwrap();

    let advises = PredicateInfo::new(
        "advises",
        vec!["Professor".to_string(), "Student".to_string()],
    )
    .with_description("Professor advises student");
    schema.add_predicate(advises).unwrap();

    println!("✓ Defined 4 predicates\n");

    // Generate GraphQL schema with queries only
    println!("Generating GraphQL schema with queries...\n");
    let codegen_queries = GraphQLCodegen::new("AcademicSchema")
        .with_queries(true)
        .with_mutations(false);
    let graphql_queries = codegen_queries.generate(&schema);

    println!("{}", "=".repeat(60));
    println!("GraphQL Schema (Queries Only):");
    println!("{}", "=".repeat(60));
    println!("{}", graphql_queries);

    // Generate GraphQL schema with both queries and mutations
    println!("\n{}", "=".repeat(60));
    println!("Generating GraphQL schema with queries and mutations...");
    println!("{}", "=".repeat(60));
    println!();

    let codegen_full = GraphQLCodegen::new("AcademicSchema")
        .with_queries(true)
        .with_mutations(true);
    let graphql_full = codegen_full.generate(&schema);

    // Show just the mutation section
    if let Some(mutation_start) = graphql_full.find("# Mutation Operations") {
        if let Some(schema_def_start) = graphql_full.find("# Schema Definition") {
            let mutation_section = &graphql_full[mutation_start..schema_def_start];
            println!("Mutation Section:");
            println!("{}", mutation_section);
        }
    }

    // Generate schema without descriptions (more compact)
    println!("\n{}", "=".repeat(60));
    println!("Compact GraphQL Schema (No Descriptions):");
    println!("{}", "=".repeat(60));
    println!();

    let codegen_compact = GraphQLCodegen::new("AcademicSchema")
        .with_descriptions(false)
        .with_queries(true)
        .with_mutations(false);
    let graphql_compact = codegen_compact.generate(&schema);

    // Show just the type definitions
    if let Some(domain_start) = graphql_compact.find("# Domain Types") {
        if let Some(query_start) = graphql_compact.find("# Query Operations") {
            let types_section = &graphql_compact[domain_start..query_start];
            println!("{}", types_section);
        }
    }

    // Statistics
    println!("\n{}", "=".repeat(60));
    println!("Statistics:");
    println!("{}", "=".repeat(60));
    println!("Generated schema types:");
    println!("  - 4 domain types (Student, Professor, Course, Grade)");
    println!("  - 4 predicate types (Enrolled, Teaches, CourseGrade, Advises)");
    println!("  - 1 Query type with {} query fields", 4 * 2 + 4); // domains * 2 + predicates
    println!("  - 1 Mutation type with {} mutation fields", 4 * 2); // predicates * 2
    println!();
    println!("Schema characteristics:");
    println!("  - Total lines: {}", graphql_full.lines().count());
    println!("  - With descriptions: {} bytes", graphql_full.len());
    println!("  - Without descriptions: {} bytes", graphql_compact.len());
    println!(
        "  - Compression ratio: {:.1}%",
        (1.0 - graphql_compact.len() as f64 / graphql_full.len() as f64) * 100.0
    );

    println!("\n{}", "=".repeat(60));
    println!("Use Cases:");
    println!("{}", "=".repeat(60));
    println!("✓ API Development: Use generated schema for GraphQL API server");
    println!("✓ Type Safety: Ensure consistency between logic rules and API");
    println!("✓ Documentation: Auto-generate API documentation");
    println!("✓ Code Generation: Generate client code from schema");
    println!("✓ Validation: Validate GraphQL queries against schema");
    println!();

    println!("The generated GraphQL schema can be:");
    println!("  1. Used with Apollo Server, GraphQL Yoga, or similar frameworks");
    println!("  2. Loaded into GraphQL IDE tools for query development");
    println!("  3. Used for client code generation (TypeScript, Rust, etc.)");
    println!("  4. Validated with GraphQL schema validation tools");
    println!();
}
