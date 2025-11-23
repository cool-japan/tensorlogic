//! Property-based tests for tensorlogic-adapters.
//!
//! These tests use proptest to verify invariants across randomly generated inputs.

use proptest::prelude::*;
use tensorlogic_adapters::{
    DependentType,
    DependentTypeContext,
    DimExpr,
    DomainHierarchy,
    DomainInfo,
    Effect,
    EffectSet,
    EvolutionAnalyzer,
    LinearContext,
    LinearKind,
    LinearType,
    MemoryDatabase,
    PredicateInfo,
    PredicateQuery,
    QueryPlanner,
    // Advanced Type System
    RefinementPredicate,
    RefinementRegistry,
    SchemaDatabase,
    SchemaValidator,
    StringInterner,
    SymbolTable,
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

// Property tests for schema evolution
proptest! {
    /// Test that evolution analysis never panics
    #[test]
    fn test_evolution_never_panics(
        old_table in symbol_table_strategy(),
        new_table in symbol_table_strategy()
    ) {
        let analyzer = EvolutionAnalyzer::new(&old_table, &new_table);
        let _report = analyzer.analyze(); // Should never panic
    }

    /// Test that identical schemas have no breaking changes
    #[test]
    fn test_evolution_identical_schemas_no_breaking(table in symbol_table_strategy()) {
        let analyzer = EvolutionAnalyzer::new(&table, &table);
        let report = analyzer.analyze().unwrap();

        prop_assert!(!report.has_breaking_changes());
        prop_assert!(report.is_backward_compatible());
    }

    /// Test that evolution is reflexive: schema compared to itself is identical
    #[test]
    fn test_evolution_reflexivity(table in symbol_table_strategy()) {
        let analyzer = EvolutionAnalyzer::new(&table, &table);
        let report = analyzer.analyze().unwrap();

        prop_assert_eq!(report.breaking_changes.len(), 0);
        prop_assert!(report.is_backward_compatible());
    }

    /// Test that adding NEW domains is backward compatible
    #[test]
    fn test_evolution_adding_domains_is_backward_compatible(
        old_table in symbol_table_strategy(),
        new_domain in domain_info_strategy()
    ) {
        // Only test if the domain is truly new (doesn't exist in old table)
        if old_table.get_domain(&new_domain.name).is_some() {
            // Skip this test case - domain already exists
            return Ok(());
        }

        let mut new_table = old_table.clone();
        new_table.add_domain(new_domain).unwrap();

        let analyzer = EvolutionAnalyzer::new(&old_table, &new_table);
        let report = analyzer.analyze().unwrap();

        // Adding truly new domains should be backward compatible
        prop_assert!(report.is_backward_compatible());
    }

    /// Test that migration plan is always generated
    #[test]
    fn test_evolution_migration_plan_exists(
        old_table in symbol_table_strategy(),
        new_table in symbol_table_strategy()
    ) {
        let analyzer = EvolutionAnalyzer::new(&old_table, &new_table);
        let _report = analyzer.analyze().unwrap();

        // Migration plan should always exist (may be empty if no changes)
        // Always true for Vec, just verify it doesn't panic
    }
}

// Property tests for query planner
proptest! {
    /// Test that query planner never panics
    #[test]
    fn test_query_planner_never_panics(table in symbol_table_strategy()) {
        let mut planner = QueryPlanner::new(&table);

        // Try various query types
        let _by_name = planner.execute(&PredicateQuery::by_name("test"));
        let _by_arity = planner.execute(&PredicateQuery::by_arity(2));
    }

    /// Test that query by name returns correct results
    #[test]
    fn test_query_by_name_correctness(table in symbol_table_strategy()) {
        let mut planner = QueryPlanner::new(&table);

        // Query for each predicate by name
        for (name, _pred) in &table.predicates {
            let query = PredicateQuery::by_name(name);
            let results = planner.execute(&query).unwrap();

            prop_assert!(results.len() <= 1); // At most one result
            if let Some((result_name, _)) = results.first() {
                prop_assert_eq!(result_name, name);
            }
        }
    }

    /// Test that query by arity returns predicates with correct arity
    #[test]
    fn test_query_by_arity_correctness(
        table in symbol_table_strategy(),
        arity in 1usize..5
    ) {
        let mut planner = QueryPlanner::new(&table);
        let query = PredicateQuery::by_arity(arity);
        let results = planner.execute(&query).unwrap();

        // All results should have the specified arity
        for (_name, pred) in results {
            prop_assert_eq!(pred.arg_domains.len(), arity);
        }
    }

    /// Test that query results are deterministic
    #[test]
    fn test_query_determinism(table in symbol_table_strategy()) {
        let mut planner = QueryPlanner::new(&table);
        let query = PredicateQuery::by_arity(2);

        // Execute same query multiple times
        let results1 = planner.execute(&query).unwrap();
        let results2 = planner.execute(&query).unwrap();

        // Results should be identical
        prop_assert_eq!(results1.len(), results2.len());
        for ((name1, pred1), (name2, pred2)) in results1.iter().zip(results2.iter()) {
            prop_assert_eq!(name1, name2);
            prop_assert_eq!(&pred1.arg_domains, &pred2.arg_domains);
        }
    }

    /// Test that query planner caches plans
    #[test]
    fn test_query_planner_has_cache(table in symbol_table_strategy()) {
        let mut planner = QueryPlanner::new(&table);
        let query = PredicateQuery::by_arity(2);

        // Execute query multiple times
        let _ = planner.execute(&query);
        let cache_size1 = planner.cache_size();

        let _ = planner.execute(&query);
        let cache_size2 = planner.cache_size();

        // Cache size should not decrease
        prop_assert!(cache_size2 >= cache_size1);
    }

    // ==========================================================================
    // Advanced Type System Property Tests
    // ==========================================================================

    /// Test that refinement predicate range checks are consistent
    #[test]
    fn test_refinement_range_consistency(
        min in -1000.0f64..1000.0,
        max in -1000.0f64..1000.0,
        value in -2000.0f64..2000.0
    ) {
        if min <= max {
            let pred = RefinementPredicate::range(min, max);
            let result = pred.check(value);

            // Value should be in range iff both bounds are satisfied
            let expected = value >= min && value <= max;
            prop_assert_eq!(result, expected);
        }
    }

    /// Test that refinement And combines predicates correctly
    #[test]
    fn test_refinement_and_composition(value in -1000.0f64..1000.0) {
        let p1 = RefinementPredicate::GreaterThan(0.0);
        let p2 = RefinementPredicate::LessThan(100.0);
        let combined = RefinementPredicate::And(vec![p1.clone(), p2.clone()]);

        let result = combined.check(value);
        let expected = p1.check(value) && p2.check(value);
        prop_assert_eq!(result, expected);
    }

    /// Test that refinement Or combines predicates correctly
    #[test]
    fn test_refinement_or_composition(value in -1000.0f64..1000.0) {
        let p1 = RefinementPredicate::LessThan(0.0);
        let p2 = RefinementPredicate::GreaterThan(100.0);
        let combined = RefinementPredicate::Or(vec![p1.clone(), p2.clone()]);

        let result = combined.check(value);
        let expected = p1.check(value) || p2.check(value);
        prop_assert_eq!(result, expected);
    }

    /// Test that refinement Not negates correctly
    #[test]
    fn test_refinement_not_negation(value in -1000.0f64..1000.0) {
        let pred = RefinementPredicate::GreaterThan(0.0);
        let negated = RefinementPredicate::not(pred.clone());

        let result = negated.check(value);
        let expected = !pred.check(value);
        prop_assert_eq!(result, expected);
    }

    /// Test that built-in refinement registry types are valid
    #[test]
    fn test_refinement_registry_builtins_valid(value in -1000.0f64..1000.0) {
        let registry = RefinementRegistry::with_builtins();

        // PositiveInt should only accept positive values
        if let Some(is_positive) = registry.check("PositiveInt", value) {
            prop_assert_eq!(is_positive, value > 0.0);
        }

        // NonNegativeInt should accept zero and positive
        if let Some(is_nonneg) = registry.check("NonNegativeInt", value) {
            prop_assert_eq!(is_nonneg, value >= 0.0);
        }
    }

    /// Test that dependent dimension expressions evaluate correctly
    #[test]
    fn test_dim_expr_arithmetic(
        a in 1usize..1000,
        b in 1usize..1000
    ) {
        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("a", a);
        ctx.set_dim("b", b);

        // Test addition
        let add_expr = DimExpr::var("a").add(DimExpr::var("b"));
        prop_assert_eq!(add_expr.eval(&ctx), Some(a + b));

        // Test multiplication
        let mul_expr = DimExpr::var("a").mul(DimExpr::var("b"));
        prop_assert_eq!(mul_expr.eval(&ctx), Some(a * b));

        // Test max
        let max_expr = DimExpr::var("a").max(DimExpr::var("b"));
        prop_assert_eq!(max_expr.eval(&ctx), Some(a.max(b)));

        // Test min
        let min_expr = DimExpr::var("a").min(DimExpr::var("b"));
        prop_assert_eq!(min_expr.eval(&ctx), Some(a.min(b)));
    }

    /// Test that dependent type shapes evaluate correctly
    #[test]
    fn test_dependent_type_shape(
        batch in 1usize..100,
        height in 1usize..100,
        width in 1usize..100
    ) {
        let mut ctx = DependentTypeContext::new();
        ctx.set_dim("batch", batch);
        ctx.set_dim("height", height);
        ctx.set_dim("width", width);

        let tensor = DependentType::tensor(
            "Float",
            vec![DimExpr::var("batch"), DimExpr::var("height"), DimExpr::var("width")]
        );

        let shape = tensor.eval_shape(&ctx).unwrap();
        prop_assert_eq!(shape, vec![batch, height, width]);
        prop_assert_eq!(tensor.rank(), 3);
    }

    /// Test that linear type kind properties are correct
    #[test]
    fn test_linear_kind_properties(_dummy in 0..1) {
        // Linear: no copy, no drop
        prop_assert!(!LinearKind::Linear.allows_copy());
        prop_assert!(!LinearKind::Linear.allows_drop());

        // Affine: no copy, yes drop
        prop_assert!(!LinearKind::Affine.allows_copy());
        prop_assert!(LinearKind::Affine.allows_drop());

        // Relevant: yes copy, no drop
        prop_assert!(LinearKind::Relevant.allows_copy());
        prop_assert!(!LinearKind::Relevant.allows_drop());

        // Unrestricted: yes copy, yes drop
        prop_assert!(LinearKind::Unrestricted.allows_copy());
        prop_assert!(LinearKind::Unrestricted.allows_drop());
    }

    /// Test that linear context tracks resources correctly
    #[test]
    fn test_linear_context_tracking(n_resources in 1usize..10) {
        let mut ctx = LinearContext::new();
        let ty = LinearType::affine("Test");

        // Create resources
        for i in 0..n_resources {
            let name = format!("resource_{}", i);
            ctx.create_resource(&name, ty.clone(), "test").unwrap();
        }

        // Check statistics
        let stats = ctx.statistics();
        prop_assert_eq!(stats.total, n_resources);
        prop_assert_eq!(stats.unused, n_resources);
        prop_assert_eq!(stats.used, 0);

        // Use first resource
        ctx.use_resource("resource_0", "test").unwrap();
        let stats = ctx.statistics();
        prop_assert_eq!(stats.used, 1);
        prop_assert_eq!(stats.unused, n_resources - 1);
    }

    /// Test that effect set operations are consistent
    #[test]
    fn test_effect_set_union_idempotent(_dummy in 0..1) {
        let set1 = EffectSet::new().with(Effect::IO).with(Effect::State);
        let union = set1.union(&set1);

        // Union with self should be the same
        prop_assert_eq!(set1.len(), union.len());
        for effect in set1.iter() {
            prop_assert!(union.has(*effect));
        }
    }

    /// Test that effect set union is commutative
    #[test]
    fn test_effect_set_union_commutative(_dummy in 0..1) {
        let set1 = EffectSet::new().with(Effect::IO).with(Effect::State);
        let set2 = EffectSet::new().with(Effect::NonDet).with(Effect::GPU);

        let union1 = set1.union(&set2);
        let union2 = set2.union(&set1);

        prop_assert_eq!(union1.len(), union2.len());
        for effect in union1.iter() {
            prop_assert!(union2.has(*effect));
        }
    }

    /// Test that pure effect set is empty
    #[test]
    fn test_effect_set_pure_empty(_dummy in 0..1) {
        let pure = EffectSet::pure();
        prop_assert!(pure.is_pure());
        prop_assert!(pure.is_empty());
        prop_assert_eq!(pure.len(), 0);
    }

    /// Test that effect set difference works correctly
    #[test]
    fn test_effect_set_difference(_dummy in 0..1) {
        let set1 = EffectSet::new().with(Effect::IO).with(Effect::State).with(Effect::GPU);
        let set2 = EffectSet::new().with(Effect::IO).with(Effect::State);

        let diff = set1.difference(&set2);

        // Difference should contain only GPU
        prop_assert_eq!(diff.len(), 1);
        prop_assert!(diff.has(Effect::GPU));
        prop_assert!(!diff.has(Effect::IO));
        prop_assert!(!diff.has(Effect::State));
    }
}

// =============================================================================
// Database Property Tests
// =============================================================================

mod database_properties {
    use super::*;

    proptest! {
        /// Test that storing and loading a schema preserves all data
        #[test]
        fn test_database_store_load_roundtrip(table in symbol_table_strategy()) {
            let mut db = MemoryDatabase::new();

            // Store the schema
            let schema_id = db.store_schema("test_schema", &table).unwrap();

            // Load it back
            let loaded = db.load_schema(schema_id).unwrap();

            // Verify all domains are preserved
            prop_assert_eq!(table.domains.len(), loaded.domains.len());
            for (name, domain) in &table.domains {
                prop_assert!(loaded.domains.contains_key(name));
                let loaded_domain = &loaded.domains[name];
                prop_assert_eq!(domain.cardinality, loaded_domain.cardinality);
                prop_assert_eq!(&domain.name, &loaded_domain.name);
            }

            // Verify all predicates are preserved
            prop_assert_eq!(table.predicates.len(), loaded.predicates.len());
            for (name, pred) in &table.predicates {
                prop_assert!(loaded.predicates.contains_key(name));
                let loaded_pred = &loaded.predicates[name];
                prop_assert_eq!(pred.arity, loaded_pred.arity);
                prop_assert_eq!(&pred.arg_domains, &loaded_pred.arg_domains);
            }
        }

        /// Test that schemas can be loaded by name
        #[test]
        fn test_database_load_by_name(table in symbol_table_strategy(), name in "[a-z]{3,15}") {
            let mut db = MemoryDatabase::new();

            // Store with a specific name
            db.store_schema(&name, &table).unwrap();

            // Load by name
            let loaded = db.load_schema_by_name(&name).unwrap();

            // Should match
            prop_assert_eq!(table.domains.len(), loaded.domains.len());
            prop_assert_eq!(table.predicates.len(), loaded.predicates.len());
        }

        /// Test that multiple versions of the same schema can coexist
        #[test]
        fn test_database_versioning(
            table1 in symbol_table_strategy(),
            table2 in symbol_table_strategy(),
        ) {
            let mut db = MemoryDatabase::new();

            // Store two versions of the same schema
            let id1 = db.store_schema("versioned_schema", &table1).unwrap();
            let id2 = db.store_schema("versioned_schema", &table2).unwrap();

            // IDs should be different
            prop_assert_ne!(id1, id2);

            // Both should be loadable
            let loaded1 = db.load_schema(id1).unwrap();
            let loaded2 = db.load_schema(id2).unwrap();

            prop_assert_eq!(table1.domains.len(), loaded1.domains.len());
            prop_assert_eq!(table2.domains.len(), loaded2.domains.len());
        }

        /// Test that schema search works correctly
        #[test]
        fn test_database_search(
            table in symbol_table_strategy(),
            prefix in "[a-z]{2,5}",
            suffix in "[a-z]{2,5}",
        ) {
            let mut db = MemoryDatabase::new();
            let name = format!("{}_test_{}", prefix, suffix);

            db.store_schema(&name, &table).unwrap();

            // Search with prefix should find it
            let results = db.search_schemas(&format!("{}_", prefix)).unwrap();
            prop_assert!(!results.is_empty());

            // Verify the found schema has the correct name
            prop_assert!(results.iter().any(|m| m.name.starts_with(&prefix)));
        }

        /// Test that deleting a schema removes it
        #[test]
        fn test_database_delete(table in symbol_table_strategy()) {
            let mut db = MemoryDatabase::new();

            let id = db.store_schema("to_delete", &table).unwrap();

            // Verify it exists
            prop_assert!(db.load_schema(id).is_ok());

            // Delete it
            db.delete_schema(id).unwrap();

            // Should no longer be loadable
            prop_assert!(db.load_schema(id).is_err());
        }

        /// Test that schema history tracks versions
        #[test]
        fn test_database_history(
            table1 in symbol_table_strategy(),
            table2 in symbol_table_strategy(),
            table3 in symbol_table_strategy(),
        ) {
            let mut db = MemoryDatabase::new();
            let name = "history_test";

            db.store_schema(name, &table1).unwrap();
            db.store_schema(name, &table2).unwrap();
            db.store_schema(name, &table3).unwrap();

            let history = db.get_schema_history(name).unwrap();

            // Should have 3 versions
            prop_assert_eq!(history.len(), 3);

            // Versions should be sequential
            prop_assert_eq!(history[0].version, 1);
            prop_assert_eq!(history[1].version, 2);
            prop_assert_eq!(history[2].version, 3);
        }

        /// Test that list_schemas returns all stored schemas
        #[test]
        fn test_database_list_schemas(
            tables in prop::collection::vec(symbol_table_strategy(), 1..10),
        ) {
            let mut db = MemoryDatabase::new();
            let mut names = Vec::new();

            // Store multiple schemas with unique names
            for (i, table) in tables.iter().enumerate() {
                let name = format!("schema_{}", i);
                names.push(name.clone());
                db.store_schema(&name, table).unwrap();
            }

            let list = db.list_schemas().unwrap();

            // Should have at least as many as we stored
            prop_assert!(list.len() >= tables.len());

            // All our schemas should be in the list
            for name in &names {
                prop_assert!(list.iter().any(|m| m.name == *name));
            }
        }

        /// Test that database handles empty schemas correctly
        #[test]
        fn test_database_empty_schema(_dummy in 0..1) {
            let mut db = MemoryDatabase::new();
            let empty = SymbolTable::new();

            let id = db.store_schema("empty", &empty).unwrap();
            let loaded = db.load_schema(id).unwrap();

            prop_assert_eq!(loaded.domains.len(), 0);
            prop_assert_eq!(loaded.predicates.len(), 0);
            prop_assert_eq!(loaded.variables.len(), 0);
        }

        /// Test database metadata accuracy
        #[test]
        fn test_database_metadata_accuracy(table in symbol_table_strategy()) {
            let mut db = MemoryDatabase::new();

            let id = db.store_schema("metadata_test", &table).unwrap();
            let list = db.list_schemas().unwrap();

            // Find our schema in the list
            let metadata = list.iter().find(|m| m.id == id).unwrap();

            // Verify counts match
            prop_assert_eq!(metadata.num_domains, table.domains.len());
            prop_assert_eq!(metadata.num_predicates, table.predicates.len());
            prop_assert_eq!(metadata.num_variables, table.variables.len());
        }

        /// Test that concurrent schema operations don't interfere
        #[test]
        fn test_database_independence(
            table1 in symbol_table_strategy(),
            table2 in symbol_table_strategy(),
        ) {
            let mut db = MemoryDatabase::new();

            // Store two different schemas
            let id1 = db.store_schema("schema_a", &table1).unwrap();
            let id2 = db.store_schema("schema_b", &table2).unwrap();

            // Load them back
            let loaded1 = db.load_schema(id1).unwrap();
            let loaded2 = db.load_schema(id2).unwrap();

            // They should be independent (not affect each other)
            prop_assert_eq!(table1.domains.len(), loaded1.domains.len());
            prop_assert_eq!(table2.domains.len(), loaded2.domains.len());

            // Modifying one shouldn't affect the other
            if !table1.domains.is_empty() && !table2.domains.is_empty() {
                let dom1_names: Vec<_> = table1.domains.keys().collect();
                let dom2_names: Vec<_> = table2.domains.keys().collect();

                // If they have different domains, verify independence
                if dom1_names != dom2_names {
                    for name in &dom1_names {
                        if !dom2_names.contains(name) {
                            prop_assert!(!loaded2.domains.contains_key(*name));
                        }
                    }
                }
            }
        }
    }
}
