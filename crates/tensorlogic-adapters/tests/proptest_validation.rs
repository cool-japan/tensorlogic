//! Property-based tests for tensorlogic-adapters.
//!
//! These tests use proptest to verify invariants across randomly generated inputs.

use proptest::prelude::*;
use tensorlogic_adapters::{
    DomainHierarchy, DomainInfo, PredicateInfo, SchemaValidator, StringInterner, SymbolTable,
};

// Strategy for generating valid domain names
fn domain_name_strategy() -> impl Strategy<Value = String> {
    "[A-Z][a-zA-Z0-9_]{0,20}"
        .prop_map(|s| s.to_string())
        .prop_filter("Not empty", |s| !s.is_empty())
}

// Strategy for generating domain cardinalities
fn cardinality_strategy() -> impl Strategy<Value = usize> {
    1usize..1000
}

// Strategy for generating DomainInfo
fn domain_info_strategy() -> impl Strategy<Value = DomainInfo> {
    (domain_name_strategy(), cardinality_strategy())
        .prop_map(|(name, card)| DomainInfo::new(name, card))
}

// Strategy for generating PredicateInfo
fn predicate_info_strategy(domains: Vec<String>) -> impl Strategy<Value = PredicateInfo> {
    let domains_clone = domains.clone();
    (
        "[a-z][a-zA-Z0-9_]{0,20}",
        prop::collection::vec(
            prop::sample::select(domains_clone),
            1..5, // Arity between 1 and 4
        ),
    )
        .prop_map(|(name, arg_domains)| PredicateInfo::new(name.to_string(), arg_domains))
}

// Strategy for generating a complete SymbolTable
fn symbol_table_strategy() -> impl Strategy<Value = SymbolTable> {
    prop::collection::vec(domain_info_strategy(), 1..10).prop_flat_map(|domains| {
        let domain_names: Vec<String> = domains.iter().map(|d| d.name.clone()).collect();
        let domain_names_clone = domain_names.clone();

        (
            Just(domains),
            prop::collection::vec(predicate_info_strategy(domain_names_clone), 0..20),
        )
            .prop_map(|(domains, predicates)| {
                let mut table = SymbolTable::new();
                for domain in domains {
                    table.add_domain(domain).unwrap();
                }
                for predicate in predicates {
                    let _ = table.add_predicate(predicate); // May fail if references unknown domain
                }
                table
            })
    })
}

proptest! {
    /// Test that JSON serialization round-trips correctly
    #[test]
    fn test_json_roundtrip(table in symbol_table_strategy()) {
        let json = table.to_json().unwrap();
        let deserialized = SymbolTable::from_json(&json).unwrap();

        // Check domains match
        prop_assert_eq!(table.domains.len(), deserialized.domains.len());
        for (name, domain) in &table.domains {
            let deser_domain = deserialized.get_domain(name).unwrap();
            prop_assert_eq!(&domain.name, &deser_domain.name);
            prop_assert_eq!(domain.cardinality, deser_domain.cardinality);
        }

        // Check predicates match
        prop_assert_eq!(table.predicates.len(), deserialized.predicates.len());
        for (name, pred) in &table.predicates {
            let deser_pred = deserialized.get_predicate(name).unwrap();
            prop_assert_eq!(&pred.name, &deser_pred.name);
            prop_assert_eq!(&pred.arg_domains, &deser_pred.arg_domains);
        }
    }

    /// Test that YAML serialization round-trips correctly
    #[test]
    fn test_yaml_roundtrip(table in symbol_table_strategy()) {
        let yaml = table.to_yaml().unwrap();
        let deserialized = SymbolTable::from_yaml(&yaml).unwrap();

        // Check domains match
        prop_assert_eq!(table.domains.len(), deserialized.domains.len());

        // Check predicates match
        prop_assert_eq!(table.predicates.len(), deserialized.predicates.len());
    }

    /// Test that adding a domain always succeeds
    #[test]
    fn test_add_domain_always_succeeds(domain in domain_info_strategy()) {
        let mut table = SymbolTable::new();
        prop_assert!(table.add_domain(domain.clone()).is_ok());
        prop_assert!(table.get_domain(&domain.name).is_some());
    }

    /// Test that domain lookup is consistent
    #[test]
    fn test_domain_lookup_consistency(domains in prop::collection::vec(domain_info_strategy(), 1..20)) {
        let mut table = SymbolTable::new();

        // Use a map to track the last version of each domain (simulating replacement behavior)
        let mut domain_map = std::collections::HashMap::new();
        for domain in &domains {
            table.add_domain(domain.clone()).unwrap();
            domain_map.insert(domain.name.clone(), domain.clone());
        }

        // Verify lookups match the last inserted version of each domain
        for (name, domain) in domain_map.iter() {
            let retrieved = table.get_domain(name);
            prop_assert!(retrieved.is_some());
            let retrieved = retrieved.unwrap();
            prop_assert_eq!(&retrieved.name, &domain.name);
            prop_assert_eq!(retrieved.cardinality, domain.cardinality);
        }
    }

    /// Test that predicate arity is preserved
    #[test]
    fn test_predicate_arity_preserved(
        domains in prop::collection::vec(domain_info_strategy(), 1..10),
        arity in 1usize..5
    ) {
        let mut table = SymbolTable::new();
        for domain in &domains {
            table.add_domain(domain.clone()).unwrap();
        }

        let domain_names: Vec<String> = domains.iter().map(|d| d.name.clone()).collect();
        let arg_domains = vec![domain_names[0].clone(); arity];
        let pred = PredicateInfo::new("test_pred", arg_domains.clone());

        table.add_predicate(pred).unwrap();
        let retrieved = table.get_predicate("test_pred").unwrap();
        prop_assert_eq!(retrieved.arg_domains.len(), arity);
    }

    /// Test that schema validator never panics
    #[test]
    fn test_validator_never_panics(table in symbol_table_strategy()) {
        let validator = SchemaValidator::new(&table);
        let _report = validator.validate(); // Should never panic
    }

    /// Test that string interner maintains consistency
    #[test]
    fn test_string_interner_consistency(strings in prop::collection::vec("[a-zA-Z]{1,20}", 1..100)) {
        let mut interner = StringInterner::new();
        let mut id_map = std::collections::HashMap::new();

        for s in &strings {
            let id = interner.intern(s);
            id_map.entry(s.clone()).or_insert(id);
        }

        // Check that same string always gets same ID
        for (s, expected_id) in &id_map {
            let id = interner.intern(s);
            prop_assert_eq!(id, *expected_id);
        }

        // Check that all IDs resolve correctly
        for (s, id) in &id_map {
            let resolved = interner.resolve(*id);
            prop_assert_eq!(resolved, Some(s.as_str()));
        }
    }

    /// Test that domain hierarchy is always acyclic
    #[test]
    fn test_hierarchy_acyclic(edges in prop::collection::vec(
        (domain_name_strategy(), domain_name_strategy()),
        1..20
    )) {
        let mut hierarchy = DomainHierarchy::new();

        for (subtype, supertype) in edges {
            if subtype != supertype {
                hierarchy.add_subtype(&subtype, &supertype);
            }
        }

        // Verify no cycles exist by checking that no domain is its own ancestor
        for domain in hierarchy.all_domains() {
            let ancestors = hierarchy.get_ancestors(&domain);
            prop_assert!(!ancestors.contains(&domain));
        }
    }

    /// Test that hierarchy transitivity holds
    #[test]
    fn test_hierarchy_transitivity(
        a in domain_name_strategy(),
        b in domain_name_strategy(),
        c in domain_name_strategy()
    ) {
        if a == b || b == c || a == c {
            return Ok(());
        }

        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype(&a, &b);
        hierarchy.add_subtype(&b, &c);

        // a <: b and b <: c implies a <: c (transitivity)
        prop_assert!(hierarchy.is_subtype(&a, &c));
    }

    /// Test that memory stats are consistent
    #[test]
    fn test_memory_stats_consistent(strings in prop::collection::vec("[a-zA-Z]{5,20}", 10..100)) {
        let mut interner = StringInterner::new();
        for s in &strings {
            interner.intern(s);
        }

        let stats = interner.memory_usage();
        let unique_strings: std::collections::HashSet<_> = strings.into_iter().collect();

        prop_assert_eq!(stats.string_count, unique_strings.len());
        prop_assert!(stats.total_bytes() > 0);
    }

    /// Test that variable binding respects domains
    #[test]
    fn test_variable_binding_domain_check(
        domains in prop::collection::vec(domain_info_strategy(), 1..10),
        var_name in "[a-z][a-z0-9]{0,10}"
    ) {
        let mut table = SymbolTable::new();
        for domain in &domains {
            table.add_domain(domain.clone()).unwrap();
        }

        // Binding to existing domain should succeed
        if let Some(domain) = domains.first() {
            prop_assert!(table.bind_variable(&var_name, &domain.name).is_ok());
            prop_assert_eq!(table.get_variable_domain(&var_name), Some(domain.name.as_str()));
        }

        // Binding to non-existent domain should fail
        prop_assert!(table.bind_variable("x", "NonExistentDomain").is_err());
    }
}

// Additional deterministic tests for edge cases
#[cfg(test)]
mod deterministic_tests {
    use super::*;

    #[test]
    fn test_empty_symbol_table_serialization() {
        let table = SymbolTable::new();
        let json = table.to_json().unwrap();
        let deserialized = SymbolTable::from_json(&json).unwrap();
        assert_eq!(table.domains.len(), deserialized.domains.len());
    }

    #[test]
    fn test_string_interner_empty() {
        let interner = StringInterner::new();
        assert_eq!(interner.len(), 0);
        assert!(interner.is_empty());
    }

    #[test]
    fn test_hierarchy_no_cycles() {
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("A", "B");
        hierarchy.add_subtype("B", "C");

        // Verify validation passes for acyclic hierarchy
        assert!(hierarchy.validate_acyclic().is_ok());
    }

    #[test]
    fn test_validation_report_empty_table() {
        let table = SymbolTable::new();
        let validator = SchemaValidator::new(&table);
        let report = validator.validate().unwrap();
        // Empty table should have no errors but may have warnings
        assert!(report.errors.is_empty());
    }
}
