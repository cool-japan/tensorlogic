//! Example 17: Complete Code Generation Suite
//!
//! This example demonstrates the multi-target code generation capabilities,
//! showing how to generate Rust, GraphQL, TypeScript, and Python code from
//! a single TensorLogic schema.

use tensorlogic_adapters::{
    DomainInfo, GraphQLCodegen, PredicateInfo, PythonCodegen, RustCodegen, SymbolTable,
    TypeScriptCodegen,
};

fn main() {
    println!("=== TensorLogic Code Generation Suite ===\n");

    // Create a sample schema for a university system
    let mut table = SymbolTable::new();

    // Add domains
    table
        .add_domain(
            DomainInfo::new("Student", 500)
                .with_description("A student enrolled in the university"),
        )
        .unwrap();
    table
        .add_domain(
            DomainInfo::new("Course", 100).with_description("A course offered by the university"),
        )
        .unwrap();
    table
        .add_domain(DomainInfo::new("Professor", 50).with_description("A university professor"))
        .unwrap();
    table
        .add_domain(DomainInfo::new("Grade", 5).with_description("Letter grades A-F"))
        .unwrap();

    // Add predicates
    table
        .add_predicate(
            PredicateInfo::new(
                "enrolled",
                vec!["Student".to_string(), "Course".to_string()],
            )
            .with_description("Student is enrolled in a course"),
        )
        .unwrap();
    table
        .add_predicate(
            PredicateInfo::new(
                "teaches",
                vec!["Professor".to_string(), "Course".to_string()],
            )
            .with_description("Professor teaches a course"),
        )
        .unwrap();
    table
        .add_predicate(
            PredicateInfo::new(
                "grade",
                vec![
                    "Student".to_string(),
                    "Course".to_string(),
                    "Grade".to_string(),
                ],
            )
            .with_description("Student received a grade in a course"),
        )
        .unwrap();

    println!("Schema created with:");
    println!("  - {} domains", table.domains.len());
    println!("  - {} predicates\n", table.predicates.len());

    // 1. Generate Rust code
    println!("=== 1. Rust Code Generation ===\n");
    let rust_codegen = RustCodegen::new("university_schema");
    let rust_code = rust_codegen.generate(&table);
    println!("Generated Rust code ({} lines):", rust_code.lines().count());
    println!("{}\n", &rust_code[..rust_code.len().min(500)]);
    println!("... (truncated)\n");

    // 2. Generate GraphQL schema
    println!("=== 2. GraphQL Schema Generation ===\n");
    let graphql_codegen = GraphQLCodegen::new("UniversitySchema")
        .with_queries(true)
        .with_mutations(true);
    let graphql_schema = graphql_codegen.generate(&table);
    println!(
        "Generated GraphQL schema ({} lines):",
        graphql_schema.lines().count()
    );
    println!("{}\n", &graphql_schema[..graphql_schema.len().min(500)]);
    println!("... (truncated)\n");

    // 3. Generate TypeScript definitions
    println!("=== 3. TypeScript Code Generation ===\n");
    let ts_codegen = TypeScriptCodegen::new("university_schema")
        .with_validators(true)
        .with_jsdoc(true);
    let ts_code = ts_codegen.generate(&table);
    println!(
        "Generated TypeScript code ({} lines):",
        ts_code.lines().count()
    );
    println!("{}\n", &ts_code[..ts_code.len().min(500)]);
    println!("... (truncated)\n");

    // 4. Generate Python type stubs
    println!("=== 4. Python Type Stubs Generation ===\n");
    let py_stub_codegen = PythonCodegen::new("university_schema").with_dataclasses(true);
    let py_stubs = py_stub_codegen.generate(&table);
    println!(
        "Generated Python stubs ({} lines):",
        py_stubs.lines().count()
    );
    println!("{}\n", &py_stubs[..py_stubs.len().min(500)]);
    println!("... (truncated)\n");

    // 5. Generate PyO3 bindings
    println!("=== 5. PyO3 Bindings Generation ===\n");
    let pyo3_codegen = PythonCodegen::new("university_schema").with_pyo3(true);
    let pyo3_code = pyo3_codegen.generate(&table);
    println!("Generated PyO3 code ({} lines):", pyo3_code.lines().count());
    println!("{}\n", &pyo3_code[..pyo3_code.len().min(500)]);
    println!("... (truncated)\n");

    // Summary
    println!("=== Summary ===");
    println!("Successfully generated code for 5 different targets:");
    println!("  ✓ Rust: {} lines", rust_code.lines().count());
    println!("  ✓ GraphQL: {} lines", graphql_schema.lines().count());
    println!("  ✓ TypeScript: {} lines", ts_code.lines().count());
    println!("  ✓ Python Stubs: {} lines", py_stubs.lines().count());
    println!("  ✓ PyO3 Bindings: {} lines", pyo3_code.lines().count());
    println!(
        "\nTotal generated: {} lines from a single schema definition!",
        rust_code.lines().count()
            + graphql_schema.lines().count()
            + ts_code.lines().count()
            + py_stubs.lines().count()
            + pyo3_code.lines().count()
    );
}
