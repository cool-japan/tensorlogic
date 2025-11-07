//! Integration tests for tensorlogic-adapters with compiler.
//!
//! Tests schema-driven compilation where domain and predicate information
//! comes from a SymbolTable rather than being manually registered.

use tensorlogic_adapters::{DomainInfo, PredicateInfo, SymbolTable};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

#[test]
fn test_schema_driven_compilation_basic() {
    // Create a symbol table with domains and predicates
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table.add_domain(DomainInfo::new("City", 50)).unwrap();

    table
        .add_predicate(PredicateInfo::new(
            "lives_in",
            vec!["Person".to_string(), "City".to_string()],
        ))
        .unwrap();

    // Create compiler context from symbol table
    let mut ctx = CompilerContext::from_symbol_table(&table);

    // Compile an expression using the imported schema
    let expr = TLExpr::pred("lives_in", vec![Term::var("x"), Term::var("y")]);

    let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

    // Verify compilation succeeded
    // For simple predicates, we get a tensor but may not have nodes
    assert!(!graph.tensors.is_empty());

    // Verify domains were imported
    assert_eq!(ctx.domains.len(), 2);
    assert!(ctx.domains.contains_key("Person"));
    assert!(ctx.domains.contains_key("City"));

    let person = ctx.domains.get("Person").unwrap();
    assert_eq!(person.cardinality, 100);
}

#[test]
fn test_schema_with_quantifiers() {
    // Create symbol table
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();

    table
        .add_predicate(PredicateInfo::new(
            "knows",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();

    // Create context from table
    let mut ctx = CompilerContext::from_symbol_table(&table);

    // Compile: ∃y. knows(x, y)
    // "Find all persons x who know someone"
    let expr = TLExpr::exists(
        "y",
        "Person",
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
    );

    let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

    // Verify compilation succeeded
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());

    // Verify domain was used
    assert!(ctx.domains.contains_key("Person"));
}

#[test]
fn test_schema_with_metadata() {
    // Create symbol table with rich metadata
    let mut table = SymbolTable::new();

    let domain =
        DomainInfo::new("Person", 100).with_description("All persons in the social network");

    table.add_domain(domain).unwrap();

    let predicate = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
        .with_description("Binary relationship indicating familiarity");

    table.add_predicate(predicate).unwrap();

    // Create context from table
    let ctx = CompilerContext::from_symbol_table(&table);

    // Verify metadata was preserved
    let person = ctx.domains.get("Person").unwrap();
    assert!(person.description.is_some());
    assert_eq!(
        person.description.as_ref().unwrap(),
        "All persons in the social network"
    );
}

#[test]
fn test_schema_with_variable_bindings() {
    // Create symbol table with pre-bound variables
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();

    table
        .add_predicate(PredicateInfo::new("p", vec!["Person".to_string()]))
        .unwrap();

    // Pre-bind a variable
    table.bind_variable("x", "Person").unwrap();

    // Create context from table
    let ctx = CompilerContext::from_symbol_table(&table);

    // Verify variable binding was imported
    assert_eq!(ctx.var_to_domain.len(), 1);
    assert_eq!(ctx.var_to_domain.get("x"), Some(&"Person".to_string()));
}

#[test]
fn test_schema_driven_with_config() {
    use tensorlogic_compiler::CompilationConfig;

    // Create symbol table
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();

    // Create context with custom configuration
    let ctx = CompilerContext::from_symbol_table_with_config(
        &table,
        CompilationConfig::fuzzy_lukasiewicz(),
    );

    // Verify both schema and config were applied
    assert_eq!(ctx.domains.len(), 1);
    assert!(matches!(
        ctx.config.and_strategy,
        tensorlogic_compiler::AndStrategy::Lukasiewicz
    ));
}

#[test]
fn test_manual_domain_addition_after_import() {
    // Create symbol table
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();

    // Create context from table
    let mut ctx = CompilerContext::from_symbol_table(&table);

    // Manually add more domains
    ctx.add_domain("City", 50);

    // Verify both imported and manual domains exist
    assert_eq!(ctx.domains.len(), 2);
    assert!(ctx.domains.contains_key("Person"));
    assert!(ctx.domains.contains_key("City"));

    assert_eq!(ctx.domains.get("Person").unwrap().cardinality, 100);
    assert_eq!(ctx.domains.get("City").unwrap().cardinality, 50);
}

#[test]
fn test_add_domain_info_with_full_metadata() {
    let mut ctx = CompilerContext::new();

    // Add domain with elements using the constructor
    let domain = DomainInfo::with_elements(
        "Person",
        vec![
            "alice".to_string(),
            "bob".to_string(),
            "charlie".to_string(),
        ],
    )
    .with_description("All persons in the system");

    ctx.add_domain_info(domain);

    // Verify metadata was preserved
    let person = ctx.domains.get("Person").unwrap();
    assert!(person.description.is_some());
    assert!(person.elements.is_some());
    assert_eq!(person.elements.as_ref().unwrap().len(), 3);
    assert_eq!(person.cardinality, 3); // Should match element count
}
