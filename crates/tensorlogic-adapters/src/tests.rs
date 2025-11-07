//! Unit tests for adapters.

use tensorlogic_ir::{TLExpr, Term};

use crate::{
    AxisMetadata, DomainHierarchy, DomainInfo, DomainMask, PredicateConstraints, PredicateInfo,
    PredicateProperty, SchemaValidator, SymbolTable, ValueRange,
};

#[test]
fn test_domain_info() {
    let domain = DomainInfo::with_elements(
        "Person",
        vec!["Alice".into(), "Bob".into(), "Charlie".into()],
    );

    assert_eq!(domain.cardinality, 3);
    assert!(domain.has_element("Alice"));
    assert!(!domain.has_element("Dave"));
    assert_eq!(domain.get_index("Bob"), Some(1));
}

#[test]
fn test_predicate_info() {
    let pred = PredicateInfo::new("Parent", vec!["Person".into(), "Person".into()]);

    assert_eq!(pred.arity, 2);
    assert!(pred
        .validate_args(&[Term::var("x"), Term::var("y")])
        .is_ok());
    assert!(pred.validate_args(&[Term::var("x")]).is_err());
}

#[test]
fn test_predicate_with_constraints() {
    let constraints = PredicateConstraints::new()
        .with_property(PredicateProperty::Symmetric)
        .with_property(PredicateProperty::Transitive);

    let pred = PredicateInfo::new("Friend", vec!["Person".into(), "Person".into()])
        .with_constraints(constraints);

    assert_eq!(pred.arity, 2);
    assert!(pred.constraints.is_some());
    let c = pred.constraints.unwrap();
    assert!(c.is_symmetric());
    assert!(c.is_transitive());
}

#[test]
fn test_axis_metadata() {
    let mut meta = AxisMetadata::new();

    let axis_x = meta.assign("x", "Person");
    let axis_y = meta.assign("y", "City");

    assert_eq!(axis_x, 0);
    assert_eq!(axis_y, 1);
    assert_eq!(meta.get_domain(0), Some("Person"));
    assert_eq!(meta.get_char(0), Some('a'));
    assert_eq!(meta.get_char(1), Some('b'));

    let spec = meta.build_spec(&["x".into(), "y".into()]);
    assert_eq!(spec, "ab");
}

#[test]
fn test_symbol_table() {
    let mut table = SymbolTable::new();

    table.add_domain(DomainInfo::new("Person", 10)).unwrap();
    table.add_domain(DomainInfo::new("City", 5)).unwrap();

    table
        .add_predicate(PredicateInfo::new(
            "LivesIn",
            vec!["Person".into(), "City".into()],
        ))
        .unwrap();

    assert_eq!(table.domains.len(), 2);
    assert_eq!(table.predicates.len(), 1);

    table.bind_variable("x", "Person").unwrap();
    assert_eq!(table.get_variable_domain("x"), Some("Person"));
}

#[test]
fn test_domain_mask() {
    let domain = DomainInfo::with_elements(
        "Person",
        vec![
            "Alice".into(),
            "Bob".into(),
            "Charlie".into(),
            "Dave".into(),
        ],
    );

    let mut mask = DomainMask::new("Person");
    mask.include("Alice").include("Bob");

    assert!(mask.is_allowed("Alice"));
    assert!(mask.is_allowed("Bob"));
    assert!(!mask.is_allowed("Charlie"));

    let indices = mask.apply_to_indices(&domain);
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn test_symbol_table_json() {
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 10)).unwrap();

    let json = table.to_json().unwrap();
    let restored = SymbolTable::from_json(&json).unwrap();

    assert_eq!(restored.domains.len(), 1);
    assert_eq!(restored.domains.get("Person").unwrap().cardinality, 10);
}

#[test]
fn test_symbol_table_yaml() {
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 10)).unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "Parent",
            vec!["Person".into(), "Person".into()],
        ))
        .unwrap();

    let yaml = table.to_yaml().unwrap();
    let restored = SymbolTable::from_yaml(&yaml).unwrap();

    assert_eq!(restored.domains.len(), 1);
    assert_eq!(restored.predicates.len(), 1);
    assert_eq!(restored.domains.get("Person").unwrap().cardinality, 10);
}

#[test]
fn test_infer_from_expr() {
    let mut table = SymbolTable::new();

    let pred = TLExpr::pred("Parent", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::exists("x", "Person", pred);

    table.infer_from_expr(&expr).unwrap();

    assert!(table.domains.contains_key("Person"));
    assert!(table.predicates.contains_key("Parent"));
    assert_eq!(table.get_variable_domain("x"), Some("Person"));
}

#[test]
fn test_domain_hierarchy() {
    let mut hierarchy = DomainHierarchy::new();
    hierarchy.add_subtype("Student", "Person");
    hierarchy.add_subtype("Teacher", "Person");
    hierarchy.add_subtype("Person", "Agent");

    assert!(hierarchy.is_subtype("Student", "Person"));
    assert!(hierarchy.is_subtype("Student", "Agent"));
    assert!(hierarchy.is_subtype("Teacher", "Person"));
    assert!(!hierarchy.is_subtype("Student", "Teacher"));

    let ancestors = hierarchy.get_ancestors("Student");
    assert_eq!(ancestors.len(), 2);
    assert!(ancestors.contains(&"Person".to_string()));
    assert!(ancestors.contains(&"Agent".to_string()));
}

#[test]
fn test_schema_validation_complete() {
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 10)).unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "Parent",
            vec!["Person".into(), "Person".into()],
        ))
        .unwrap();

    let validator = SchemaValidator::new(&table);
    let report = validator.validate().unwrap();

    assert!(report.is_valid());
}

#[test]
fn test_schema_validation_with_hierarchy() {
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 10)).unwrap();
    table.add_domain(DomainInfo::new("Student", 5)).unwrap();

    let mut hierarchy = DomainHierarchy::new();
    hierarchy.add_subtype("Student", "Person");

    let validator = SchemaValidator::new(&table).with_hierarchy(&hierarchy);
    let report = validator.validate().unwrap();

    assert!(report.is_valid());
}

#[test]
fn test_value_range_constraints() {
    let range = ValueRange::new().with_min(0.0, true).with_max(1.0, true);

    assert!(range.contains(0.0));
    assert!(range.contains(0.5));
    assert!(range.contains(1.0));
    assert!(!range.contains(-0.1));
    assert!(!range.contains(1.1));
}

#[test]
fn test_predicate_properties() {
    let constraints = PredicateConstraints::new()
        .with_property(PredicateProperty::Symmetric)
        .with_property(PredicateProperty::Reflexive);

    assert!(constraints.has_property(&PredicateProperty::Symmetric));
    assert!(constraints.has_property(&PredicateProperty::Reflexive));
    assert!(!constraints.has_property(&PredicateProperty::Transitive));
}

#[test]
fn test_infer_arithmetic_expr() {
    let mut table = SymbolTable::new();

    // Test with arithmetic operations
    let pred1 = TLExpr::pred("Score", vec![Term::var("x")]);
    let pred2 = TLExpr::pred("Bonus", vec![Term::var("x")]);
    let expr = TLExpr::add(pred1, pred2);

    table.infer_from_expr(&expr).unwrap();

    assert!(table.predicates.contains_key("Score"));
    assert!(table.predicates.contains_key("Bonus"));
}

#[test]
fn test_infer_comparison_expr() {
    let mut table = SymbolTable::new();

    // Test with comparison operations
    let pred1 = TLExpr::pred("Age", vec![Term::var("x")]);
    let pred2 = TLExpr::constant(18.0);
    let expr = TLExpr::gte(pred1, pred2);

    table.infer_from_expr(&expr).unwrap();

    assert!(table.predicates.contains_key("Age"));
}

#[test]
fn test_infer_conditional_expr() {
    let mut table = SymbolTable::new();

    // Test with if-then-else
    let cond = TLExpr::pred("IsAdult", vec![Term::var("x")]);
    let then_branch = TLExpr::pred("CanVote", vec![Term::var("x")]);
    let else_branch = TLExpr::pred("CannotVote", vec![Term::var("x")]);
    let expr = TLExpr::if_then_else(cond, then_branch, else_branch);

    table.infer_from_expr(&expr).unwrap();

    assert!(table.predicates.contains_key("IsAdult"));
    assert!(table.predicates.contains_key("CanVote"));
    assert!(table.predicates.contains_key("CannotVote"));
}
