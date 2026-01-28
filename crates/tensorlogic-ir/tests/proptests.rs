//! Property-based tests for TensorLogic IR
//!
//! These tests use proptest to validate invariants and properties
//! that should hold for all possible IR structures.

use proptest::prelude::*;
use tensorlogic_ir::{
    DomainInfo, DomainRegistry, DomainType, EinsumGraph, EinsumNode, TLExpr, Term,
};

// ===== Strategies for generating test data =====

/// Generate random variable names
fn arb_var_name() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9]*".prop_map(|s| s.to_string())
}

/// Generate random constant names
fn arb_const_name() -> impl Strategy<Value = String> {
    "[A-Z][A-Za-z0-9]*".prop_map(|s| s.to_string())
}

/// Generate random predicate names
fn arb_pred_name() -> impl Strategy<Value = String> {
    "[A-Z][a-z]+".prop_map(|s| s.to_string())
}

/// Generate random Terms (variables or constants)
fn arb_term() -> impl Strategy<Value = Term> {
    prop_oneof![
        arb_var_name().prop_map(Term::var),
        arb_const_name().prop_map(Term::constant),
    ]
}

/// Generate random TLExpr with bounded depth
fn arb_expr(depth: u32) -> impl Strategy<Value = TLExpr> {
    let leaf = prop_oneof![
        (arb_pred_name(), prop::collection::vec(arb_term(), 0..=3))
            .prop_map(|(name, args)| TLExpr::pred(name, args)),
        any::<f64>().prop_map(TLExpr::constant),
    ];

    leaf.prop_recursive(depth, 256, 10, move |inner| {
        prop_oneof![
            inner.clone().prop_map(TLExpr::negate),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| TLExpr::and(a, b)),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| TLExpr::or(a, b)),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| TLExpr::imply(a, b)),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| TLExpr::add(a, b)),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| TLExpr::mul(a, b)),
        ]
    })
}

// ===== Property Tests =====

proptest! {
    #[test]
    fn prop_and_free_vars_is_union(e1 in arb_expr(3), e2 in arb_expr(3)) {
        let and_expr = TLExpr::and(e1.clone(), e2.clone());
        let and_vars = and_expr.free_vars();
        let e1_vars = e1.free_vars();
        let e2_vars = e2.free_vars();

        // Every var in e1 or e2 should be in the AND expression
        for v in &e1_vars {
            prop_assert!(and_vars.contains(v), "Variable {} from e1 not in AND", v);
        }
        for v in &e2_vars {
            prop_assert!(and_vars.contains(v), "Variable {} from e2 not in AND", v);
        }

        // Every var in AND should be in e1 or e2
        for v in &and_vars {
            prop_assert!(
                e1_vars.contains(v) || e2_vars.contains(v),
                "Variable {} in AND but not in e1 or e2", v
            );
        }
    }

    /// Property: Free variables of (P ∨ Q) = free_vars(P) ∪ free_vars(Q)
    #[test]
    fn prop_or_free_vars_is_union(e1 in arb_expr(3), e2 in arb_expr(3)) {
        let or_expr = TLExpr::or(e1.clone(), e2.clone());
        let or_vars = or_expr.free_vars();
        let e1_vars = e1.free_vars();
        let e2_vars = e2.free_vars();

        for v in &e1_vars {
            prop_assert!(or_vars.contains(v));
        }
        for v in &e2_vars {
            prop_assert!(or_vars.contains(v));
        }
    }

    /// Property: Free variables of ¬P = free_vars(P)
    #[test]
    fn prop_not_preserves_free_vars(e in arb_expr(4)) {
        let not_expr = TLExpr::negate(e.clone());
        prop_assert_eq!(not_expr.free_vars(), e.free_vars());
    }

    /// Property: Serialization roundtrip preserves equality
    /// Note: JSON serialization may lose precision for very large f64 values
    #[test]
    #[ignore]
    fn prop_json_roundtrip(e in arb_expr(4)) {
        let json = serde_json::to_string(&e).expect("serialization failed");
        let decoded: TLExpr = serde_json::from_str(&json).expect("deserialization failed");
        prop_assert_eq!(e, decoded);
    }

    /// Property: Binary serialization roundtrip preserves equality
    #[test]
    fn prop_binary_roundtrip(e in arb_expr(4)) {
        let binary = oxicode::serde::encode_to_vec(&e, oxicode::config::standard()).expect("serialization failed");
        let (decoded, _): (TLExpr, usize) = oxicode::serde::decode_from_slice(&binary, oxicode::config::standard()).expect("deserialization failed");
        prop_assert_eq!(e, decoded);
    }

    /// Property: All predicates in (P ∧ Q) = predicates(P) ∪ predicates(Q)
    #[test]
    fn prop_and_predicates_is_union(e1 in arb_expr(3), e2 in arb_expr(3)) {
        let and_expr = TLExpr::and(e1.clone(), e2.clone());
        let and_preds = and_expr.all_predicates();
        let e1_preds = e1.all_predicates();
        let e2_preds = e2.all_predicates();

        for (name, arity) in &e1_preds {
            prop_assert!(and_preds.contains_key(name), "Predicate {} from e1 not in AND", name);
            prop_assert_eq!(and_preds.get(name), Some(arity));
        }
        for (name, arity) in &e2_preds {
            prop_assert!(and_preds.contains_key(name), "Predicate {} from e2 not in AND", name);
            prop_assert_eq!(and_preds.get(name), Some(arity));
        }
    }

    /// Property: Cloned expressions are equal
    #[test]
    fn prop_clone_equality(e in arb_expr(4)) {
        let cloned = e.clone();
        prop_assert_eq!(e, cloned);
    }

    /// Property: Domain validation succeeds for registered domains
    #[test]
    fn prop_domain_validation_with_registry(
        domain_name in "[A-Z][a-z]+",
        var_name in arb_var_name()
    ) {
        let mut registry = DomainRegistry::new();
        registry.register(DomainInfo::finite(&domain_name, 100)).expect("register failed");

        let expr = TLExpr::exists(
            &var_name,
            &domain_name,
            TLExpr::pred("P", vec![Term::var(&var_name)])
        );

        prop_assert!(expr.validate_domains(&registry).is_ok());
    }

    /// Property: Adding tensors increases graph size
    #[test]
    fn prop_graph_add_tensor_increases_size(
        tensor_names in prop::collection::vec("[a-z]+", 1..10)
    ) {
        let mut graph = EinsumGraph::new();
        for (i, name) in tensor_names.iter().enumerate() {
            let idx = graph.add_tensor(name);
            prop_assert_eq!(idx, i);
            prop_assert_eq!(graph.tensors.len(), i + 1);
        }
    }

    /// Property: Constants have no free variables
    #[test]
    fn prop_constant_no_free_vars(value in any::<f64>()) {
        let expr = TLExpr::constant(value);
        prop_assert!(expr.free_vars().is_empty());
    }

    /// Property: Predicate with only constants has no free variables
    #[test]
    fn prop_const_predicate_no_free_vars(
        pred_name in arb_pred_name(),
        const_names in prop::collection::vec(arb_const_name(), 1..4)
    ) {
        let terms: Vec<Term> = const_names.into_iter().map(Term::constant).collect();
        let expr = TLExpr::pred(pred_name, terms);
        prop_assert!(expr.free_vars().is_empty());
    }

    /// Property: Double negation can be applied
    #[test]
    fn prop_double_negation_structure(e in arb_expr(3)) {
        let not_e = TLExpr::negate(e.clone());
        let not_not_e = TLExpr::negate(not_e);

        // Double negation has same free vars as original
        prop_assert_eq!(not_not_e.free_vars(), e.free_vars());
    }

    /// Property: Implication preserves free variables
    #[test]
    fn prop_imply_free_vars_union(e1 in arb_expr(3), e2 in arb_expr(3)) {
        let imply_expr = TLExpr::imply(e1.clone(), e2.clone());
        let imply_vars = imply_expr.free_vars();
        let e1_vars = e1.free_vars();
        let e2_vars = e2.free_vars();

        for v in &e1_vars {
            prop_assert!(imply_vars.contains(v));
        }
        for v in &e2_vars {
            prop_assert!(imply_vars.contains(v));
        }
    }

    /// Property: Arithmetic operations preserve free variables
    #[test]
    fn prop_arithmetic_free_vars_union(e1 in arb_expr(2), e2 in arb_expr(2)) {
        let add_expr = TLExpr::add(e1.clone(), e2.clone());
        let mul_expr = TLExpr::mul(e1.clone(), e2.clone());

        let e1_vars = e1.free_vars();
        let e2_vars = e2.free_vars();

        // Check ADD
        for v in &e1_vars {
            prop_assert!(add_expr.free_vars().contains(v));
        }
        for v in &e2_vars {
            prop_assert!(add_expr.free_vars().contains(v));
        }

        // Check MUL
        for v in &e1_vars {
            prop_assert!(mul_expr.free_vars().contains(v));
        }
        for v in &e2_vars {
            prop_assert!(mul_expr.free_vars().contains(v));
        }
    }

    /// Property: Comparison operations preserve free variables
    #[test]
    fn prop_comparison_free_vars_union(e1 in arb_expr(2), e2 in arb_expr(2)) {
        let lt_expr = TLExpr::lt(e1.clone(), e2.clone());
        let gt_expr = TLExpr::gt(e1.clone(), e2.clone());

        let e1_vars = e1.free_vars();
        let e2_vars = e2.free_vars();

        for v in &e1_vars {
            prop_assert!(lt_expr.free_vars().contains(v));
            prop_assert!(gt_expr.free_vars().contains(v));
        }
        for v in &e2_vars {
            prop_assert!(lt_expr.free_vars().contains(v));
            prop_assert!(gt_expr.free_vars().contains(v));
        }
    }

    /// Property: Domain types can be created and queried
    #[test]
    fn prop_domain_type_operations(size in 1usize..1000) {
        let finite_domain = DomainInfo::finite("TestDomain", size);
        prop_assert_eq!(finite_domain.domain_type, DomainType::Finite);
        prop_assert_eq!(finite_domain.name, "TestDomain");
        prop_assert_eq!(finite_domain.size, Some(size));
    }

    /// Property: Term equality is reflexive
    #[test]
    fn prop_term_equality_reflexive(name in "[a-z]+") {
        let t1 = Term::var(&name);
        let t2 = Term::var(&name);
        prop_assert_eq!(t1, t2);
    }
}

// ===== Additional deterministic property tests =====

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_graph_validation_well_formed() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("a");
        let b = graph.add_tensor("b");
        let c = graph.add_tensor("c");

        graph
            .add_node(EinsumNode::elem_binary("add", a, b, c))
            .expect("add node failed");
        graph.add_output(c).expect("add output failed");

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_empty_graph_valid() {
        let graph = EinsumGraph::new();
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_builtin_domains() {
        let registry = DomainRegistry::with_builtins();

        assert!(registry.get("Bool").is_some());
        assert!(registry.get("Int").is_some());
        assert!(registry.get("Real").is_some());
        assert!(registry.get("Nat").is_some());
        assert!(registry.get("Probability").is_some());
    }

    #[test]
    fn test_graph_clone_equality() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("a");
        let b = graph.add_tensor("b");
        let c = graph.add_tensor("c");

        graph
            .add_node(EinsumNode::elem_binary("add", a, b, c))
            .expect("add node failed");
        graph.add_output(c).expect("add output failed");

        let cloned = graph.clone();
        assert_eq!(graph, cloned);
    }

    #[test]
    fn test_exists_quantifier_binds_variable() {
        let expr = TLExpr::exists("x", "Domain", TLExpr::pred("P", vec![Term::var("x")]));

        let free_vars = expr.free_vars();
        assert!(!free_vars.contains("x"), "Exists should bind variable x");
    }

    #[test]
    fn test_forall_quantifier_binds_variable() {
        let expr = TLExpr::forall("x", "Domain", TLExpr::pred("P", vec![Term::var("x")]));

        let free_vars = expr.free_vars();
        assert!(!free_vars.contains("x"), "ForAll should bind variable x");
    }

    #[test]
    fn test_nested_quantifiers_bind_correctly() {
        let expr = TLExpr::exists(
            "x",
            "D1",
            TLExpr::forall(
                "y",
                "D2",
                TLExpr::pred("R", vec![Term::var("x"), Term::var("y"), Term::var("z")]),
            ),
        );

        let free_vars = expr.free_vars();
        assert!(!free_vars.contains("x"));
        assert!(!free_vars.contains("y"));
        assert!(free_vars.contains("z"), "z should remain free");
    }

    #[test]
    fn test_if_then_else_preserves_free_vars() {
        let cond = TLExpr::pred("C", vec![Term::var("x")]);
        let then_branch = TLExpr::pred("T", vec![Term::var("y")]);
        let else_branch = TLExpr::pred("E", vec![Term::var("z")]);

        let ite = TLExpr::if_then_else(cond, then_branch, else_branch);

        let free_vars = ite.free_vars();
        assert!(free_vars.contains("x"));
        assert!(free_vars.contains("y"));
        assert!(free_vars.contains("z"));
    }

    #[test]
    fn test_score_preserves_free_vars() {
        let expr = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
        let scored = TLExpr::score(expr.clone());

        assert_eq!(scored.free_vars(), expr.free_vars());
    }

    #[test]
    fn test_graph_multi_output() {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("a");
        let b = graph.add_tensor("b");
        let c = graph.add_tensor("c");

        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .expect("add node 1");
        graph
            .add_node(EinsumNode::elem_unary("sigmoid", a, c))
            .expect("add node 2");

        graph.add_output(b).expect("add output 1");
        graph.add_output(c).expect("add output 2");

        assert_eq!(graph.outputs.len(), 2);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_domain_compatibility() {
        let mut registry = DomainRegistry::new();
        registry
            .register(DomainInfo::finite("A", 10))
            .expect("register A");
        registry
            .register(DomainInfo::finite("B", 20))
            .expect("register B");

        // Same domain is compatible with itself
        assert!(registry.are_compatible("A", "A").unwrap());
        assert!(registry.are_compatible("B", "B").unwrap());
    }

    #[test]
    fn test_arity_validation_catches_mismatch() {
        // P(x, y) ∧ P(z) - should fail arity validation
        let p1 = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
        let p2 = TLExpr::pred("P", vec![Term::var("z")]);
        let expr = TLExpr::and(p1, p2);

        assert!(
            expr.validate_arity().is_err(),
            "Should detect arity mismatch"
        );
    }

    #[test]
    fn test_arity_validation_accepts_consistent() {
        // P(x, y) ∧ P(a, b) - should pass arity validation
        let p1 = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
        let p2 = TLExpr::pred("P", vec![Term::var("a"), Term::var("b")]);
        let expr = TLExpr::and(p1, p2);

        assert!(
            expr.validate_arity().is_ok(),
            "Should accept consistent arity"
        );
    }
}

// ===== Strategy for generating logical expressions (no arithmetic) =====

fn arb_logical_expr(depth: u32) -> impl Strategy<Value = TLExpr> {
    let leaf = (arb_pred_name(), prop::collection::vec(arb_term(), 0..=2))
        .prop_map(|(name, args)| TLExpr::pred(name, args));

    leaf.prop_recursive(depth, 128, 8, move |inner| {
        prop_oneof![
            inner.clone().prop_map(TLExpr::negate),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| TLExpr::and(a, b)),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| TLExpr::or(a, b)),
            (inner.clone(), inner).prop_map(|(a, b)| TLExpr::imply(a, b)),
        ]
    })
}

// ===== Property Tests for Normal Forms =====

proptest! {
    /// Property: Converting to NNF and back should preserve semantics
    #[test]
    fn prop_nnf_preserves_structure(expr in arb_logical_expr(4)) {
        use tensorlogic_ir::to_nnf;

        let nnf = to_nnf(&expr);

        // NNF should have same predicates
        let orig_preds = expr.all_predicates();
        let nnf_preds = nnf.all_predicates();

        // All predicates in original should be in NNF
        for name in orig_preds.keys() {
            prop_assert!(nnf_preds.contains_key(name),
                "Predicate {} lost in NNF conversion", name);
        }
    }

    /// Property: Converting to CNF multiple times should be idempotent
    #[test]
    fn prop_cnf_idempotent(expr in arb_logical_expr(2)) {
        use tensorlogic_ir::{to_cnf, is_cnf};

        let cnf1 = to_cnf(&expr);
        let cnf2 = to_cnf(&cnf1);

        // Second conversion should not change structure significantly
        prop_assert!(is_cnf(&cnf1), "First conversion not in CNF");
        prop_assert!(is_cnf(&cnf2), "Second conversion not in CNF");

        // Should have same number of predicates
        prop_assert_eq!(
            cnf1.all_predicates().len(),
            cnf2.all_predicates().len(),
            "CNF not idempotent - predicate count changed"
        );
    }

    /// Property: Converting to DNF multiple times should be idempotent
    #[test]
    fn prop_dnf_idempotent(expr in arb_logical_expr(2)) {
        use tensorlogic_ir::{to_dnf, is_dnf};

        let dnf1 = to_dnf(&expr);
        let dnf2 = to_dnf(&dnf1);

        prop_assert!(is_dnf(&dnf1), "First conversion not in DNF");
        prop_assert!(is_dnf(&dnf2), "Second conversion not in DNF");

        prop_assert_eq!(
            dnf1.all_predicates().len(),
            dnf2.all_predicates().len(),
            "DNF not idempotent - predicate count changed"
        );
    }

    /// Property: CNF result should always satisfy is_cnf predicate
    #[test]
    fn prop_to_cnf_produces_cnf(expr in arb_logical_expr(2)) {
        use tensorlogic_ir::{to_cnf, is_cnf};

        let cnf = to_cnf(&expr);
        prop_assert!(is_cnf(&cnf), "to_cnf() did not produce valid CNF");
    }

    /// Property: DNF result should always satisfy is_dnf predicate
    #[test]
    fn prop_to_dnf_produces_dnf(expr in arb_logical_expr(2)) {
        use tensorlogic_ir::{to_dnf, is_dnf};

        let dnf = to_dnf(&expr);
        prop_assert!(is_dnf(&dnf), "to_dnf() did not produce valid DNF");
    }
}

// ===== Property Tests for Modal/Temporal Logic =====

proptest! {
    /// Property: Free variables in Box(P) = free_vars(P)
    #[test]
    fn prop_box_preserves_free_vars(expr in arb_expr(3)) {
        let boxed = TLExpr::modal_box(expr.clone());
        let orig_vars = expr.free_vars();
        let boxed_vars = boxed.free_vars();

        prop_assert_eq!(orig_vars, boxed_vars,
            "Box operator changed free variables");
    }

    /// Property: Free variables in Diamond(P) = free_vars(P)
    #[test]
    fn prop_diamond_preserves_free_vars(expr in arb_expr(3)) {
        let diamond = TLExpr::modal_diamond(expr.clone());
        let orig_vars = expr.free_vars();
        let diamond_vars = diamond.free_vars();

        prop_assert_eq!(orig_vars, diamond_vars,
            "Diamond operator changed free variables");
    }

    /// Property: Free variables in Next(P) = free_vars(P)
    #[test]
    fn prop_next_preserves_free_vars(expr in arb_expr(3)) {
        let next = TLExpr::next(expr.clone());
        prop_assert_eq!(expr.free_vars(), next.free_vars(),
            "Next operator changed free variables");
    }

    /// Property: Free variables in Eventually(P) = free_vars(P)
    #[test]
    fn prop_eventually_preserves_free_vars(expr in arb_expr(3)) {
        let eventually = TLExpr::eventually(expr.clone());
        prop_assert_eq!(expr.free_vars(), eventually.free_vars(),
            "Eventually operator changed free variables");
    }

    /// Property: Free variables in Always(P) = free_vars(P)
    #[test]
    fn prop_always_preserves_free_vars(expr in arb_expr(3)) {
        let always = TLExpr::always(expr.clone());
        prop_assert_eq!(expr.free_vars(), always.free_vars(),
            "Always operator changed free variables");
    }

    /// Property: Free variables in Until(P, Q) = free_vars(P) ∪ free_vars(Q)
    #[test]
    fn prop_until_combines_free_vars(e1 in arb_expr(3), e2 in arb_expr(3)) {
        let until = TLExpr::until(e1.clone(), e2.clone());
        let until_vars = until.free_vars();
        let e1_vars = e1.free_vars();
        let e2_vars = e2.free_vars();

        for v in &e1_vars {
            prop_assert!(until_vars.contains(v),
                "Variable {} from first arg not in Until", v);
        }
        for v in &e2_vars {
            prop_assert!(until_vars.contains(v),
                "Variable {} from second arg not in Until", v);
        }
    }

    /// Property: Modal/temporal operators preserve predicates
    #[test]
    fn prop_modal_temporal_preserve_predicates(expr in arb_expr(3)) {
        let orig_preds = expr.all_predicates();

        let operators = vec![
            TLExpr::modal_box(expr.clone()),
            TLExpr::modal_diamond(expr.clone()),
            TLExpr::next(expr.clone()),
            TLExpr::eventually(expr.clone()),
            TLExpr::always(expr.clone()),
        ];

        for op_expr in operators {
            let op_preds = op_expr.all_predicates();
            prop_assert_eq!(orig_preds.len(), op_preds.len(),
                "Modal/temporal operator changed predicate count");

            for (name, arity) in &orig_preds {
                prop_assert_eq!(
                    op_preds.get(name),
                    Some(arity),
                    "Predicate {} lost or changed in modal/temporal operator", name
                );
            }
        }
    }
}

// ===== Property Tests for Graph Canonicalization =====

proptest! {
    /// Property: Canonicalizing a graph twice should produce identical results
    #[test]
    fn prop_canonicalize_idempotent(
        tensor_count in 3_usize..6
    ) {
        use tensorlogic_ir::{EinsumGraph, EinsumNode, canonicalize_graph};

        let mut graph = EinsumGraph::new();
        for i in 0..tensor_count {
            graph.add_tensor(format!("t{}", i));
        }

        // Add a simple node (input tensors 0,1 -> output tensor 2)
        let _ = graph.add_node(EinsumNode::elem_binary("add", 0, 1, 2));

        let canon1 = canonicalize_graph(&graph).unwrap();
        let canon2 = canonicalize_graph(&canon1).unwrap();

        prop_assert_eq!(canon1, canon2, "Canonicalization not idempotent");
    }

    /// Property: Equivalent graphs should have the same canonical hash
    #[test]
    fn prop_equivalent_graphs_same_hash(
        tensor_count in 2_usize..6
    ) {
        use tensorlogic_ir::{EinsumGraph, EinsumNode, canonical_hash};

        // Create two structurally identical graphs with different names
        let mut g1 = EinsumGraph::new();
        let mut g2 = EinsumGraph::new();

        for i in 0..tensor_count {
            g1.add_tensor(format!("tensor_{}", i));
            g2.add_tensor(format!("T{}", i));
        }

        // Add identical structure to both
        let _ = g1.add_node(EinsumNode::elem_unary("neg", 0, 1));
        let _ = g2.add_node(EinsumNode::elem_unary("neg", 0, 1));

        let hash1 = canonical_hash(&g1).unwrap();
        let hash2 = canonical_hash(&g2).unwrap();

        prop_assert_eq!(hash1, hash2,
            "Structurally equivalent graphs have different canonical hashes");
    }
}
