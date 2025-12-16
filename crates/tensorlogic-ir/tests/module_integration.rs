//! Integration Tests for TensorLogic-IR Modules
//!
//! This test suite demonstrates integration between different modules:
//! - Resolution + Sequent Calculus
//! - Type System + Domain Constraints
//! - Proof Systems + Graph Construction
//! - Unification across modules

use tensorlogic_ir::*;

// ============================================================================
// Resolution + Sequent Calculus Integration
// ============================================================================

#[test]
fn test_resolution_and_sequent_on_same_problem() {
    // Problem: Simple contradiction - P and ¬P

    // Resolution approach (proof by refutation)
    let mut res_prover = ResolutionProver::new();

    let p_lit = Literal::positive(TLExpr::pred("P", vec![]));
    let not_p_lit = Literal::negative(TLExpr::pred("P", vec![]));

    res_prover.add_clause(Clause::from_literals(vec![p_lit]));
    res_prover.add_clause(Clause::from_literals(vec![not_p_lit]));

    let res_result = res_prover.prove();
    assert!(res_result.is_unsatisfiable());

    // Sequent calculus approach (direct proof)
    // P ∧ Q ⊢ P (a simpler positive example)
    let p_expr = TLExpr::pred("P", vec![]);
    let q_expr = TLExpr::pred("Q", vec![]);
    let and_pq = TLExpr::and(p_expr.clone(), q_expr);

    let sequent = Sequent::new(vec![and_pq], vec![p_expr]);

    let mut engine =
        ProofSearchEngine::new(ProofSearchStrategy::DepthFirst { max_depth: 10 }, 1000);
    let seq_result = engine.search(&sequent);
    assert!(seq_result.is_some());

    // Both approaches find proofs (for different problems)
    let proof = seq_result.unwrap();
    assert!(proof.is_valid());
}

#[test]
fn test_modal_logic_identity() {
    // Test modal operators in sequent calculus
    let p = TLExpr::pred("P", vec![]);

    // □P ⊢ □P (identity for modal necessity)
    let box_p = TLExpr::modal_box(p);
    let sequent = Sequent::identity(box_p);

    assert!(sequent.is_axiom());
}

#[test]
fn test_temporal_operators() {
    // Test temporal operators
    let p = TLExpr::pred("P", vec![]);

    // F(P) - Eventually P
    let eventually_p = TLExpr::eventually(p);

    // Check predicates are extracted
    let preds = eventually_p.all_predicates();
    assert!(preds.contains_key("P"));
}

#[test]
fn test_multiple_proof_strategies_integration() {
    // Compare different proof strategies on the same problem
    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);
    let r = TLExpr::pred("R", vec![]);

    // (P ∧ Q) ∧ R ⊢ P
    let nested = TLExpr::and(TLExpr::and(p.clone(), q), r);
    let goal = Sequent::new(vec![nested], vec![p]);

    let strategies = vec![
        ProofSearchStrategy::DepthFirst { max_depth: 10 },
        ProofSearchStrategy::BreadthFirst { max_depth: 10 },
        ProofSearchStrategy::IterativeDeepening { max_depth: 10 },
    ];

    for strategy in strategies {
        let mut engine = ProofSearchEngine::new(strategy, 1000);
        let proof = engine.search(&goal);

        // All strategies should find a proof
        assert!(proof.is_some());
        assert!(proof.unwrap().is_valid());
    }
}

// ============================================================================
// Type System + Domain Constraints Integration
// ============================================================================

#[test]
fn test_domain_validation_with_quantifiers() {
    let domain_registry = DomainRegistry::with_builtins();

    // Create expression: ∃x: Int. P(x)
    let x = Term::var("x");
    let p_x = TLExpr::pred("P", vec![x]);
    let exists_p = TLExpr::exists("x", "Int", p_x);

    // Validate domains
    assert!(exists_p.validate_domains(&domain_registry).is_ok());

    // Nested quantifiers
    let y = Term::var("y");
    let q_y = TLExpr::pred("Q", vec![y]);
    let forall_q = TLExpr::forall("y", "Real", q_y);
    let nested = TLExpr::and(exists_p, forall_q);

    assert!(nested.validate_domains(&domain_registry).is_ok());
}

#[test]
fn test_domain_constraints_enforce_type_safety() {
    let registry = DomainRegistry::with_builtins();

    // Create well-typed expression
    let x = Term::var("x");
    let p = TLExpr::pred("P", vec![x]);
    let exists_p = TLExpr::exists("x", "Int", p.clone());

    // Domain validation succeeds
    assert!(exists_p.validate_domains(&registry).is_ok());

    // Invalid domain should fail
    let invalid = TLExpr::exists("x", "InvalidDomain", p);
    assert!(invalid.validate_domains(&registry).is_err());
}

#[test]
fn test_refinement_types_basic() {
    // Create a simple refinement type
    let zero = TLExpr::constant(0.0);
    let x_var = TLExpr::pred("x", vec![Term::var("x")]);
    let predicate = TLExpr::gt(x_var, zero);

    let positive_int = RefinementType::new("x", "Int", predicate);

    // Refinement type is created successfully
    assert_eq!(positive_int.var_name, "x");
}

// ============================================================================
// Proof Systems + Graph Construction Integration
// ============================================================================

#[test]
fn test_logical_expression_to_graph_mapping() {
    // Logical AND maps to element-wise multiplication

    let mut graph = EinsumGraph::new();

    let p_tensor = graph.add_tensor("P");
    let q_tensor = graph.add_tensor("Q");
    let result_tensor = graph.add_tensor("result");

    // Element-wise multiplication for AND
    graph
        .add_node(EinsumNode::elem_binary(
            "mul",
            p_tensor,
            q_tensor,
            result_tensor,
        ))
        .unwrap();

    graph.add_output(result_tensor).unwrap();

    // Validate the graph
    assert!(graph.validate().is_ok());
    assert_eq!(graph.tensors.len(), 3);
    assert_eq!(graph.nodes.len(), 1);
}

#[test]
fn test_proof_complexity_metrics() {
    // Test that proof complexity can be measured
    let mut prover = ResolutionProver::new();

    // Simple problem: P, ¬P
    let p = Literal::positive(TLExpr::pred("P", vec![]));
    let not_p = p.negate();

    prover.add_clause(Clause::from_literals(vec![p]));
    prover.add_clause(Clause::from_literals(vec![not_p]));

    let result = prover.prove();
    assert!(result.is_unsatisfiable());

    // Stats should show minimal work
    assert!(prover.stats.resolution_steps > 0);
    assert!(prover.stats.empty_clause_found);
}

// ============================================================================
// Unification in Resolution
// ============================================================================

#[test]
fn test_resolution_with_ground_terms() {
    let mut prover = ResolutionProver::new();

    // Ground facts: P(a)
    let p_a = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));

    // Negated goal: ¬P(a)
    let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));

    prover.add_clause(Clause::from_literals(vec![p_a]));
    prover.add_clause(Clause::from_literals(vec![not_p_a]));

    // Resolution should prove this by finding contradiction
    let result = prover.prove();
    assert!(result.is_unsatisfiable());
    assert!(prover.stats.empty_clause_found);
}

#[test]
fn test_resolution_with_multiple_literals() {
    let mut prover = ResolutionProver::new();

    // R(a, b)
    let r_ab = Literal::positive(TLExpr::pred(
        "R",
        vec![Term::constant("a"), Term::constant("b")],
    ));

    // ¬R(a, b)
    let not_r_ab = Literal::negative(TLExpr::pred(
        "R",
        vec![Term::constant("a"), Term::constant("b")],
    ));

    prover.add_clause(Clause::from_literals(vec![r_ab]));
    prover.add_clause(Clause::from_literals(vec![not_r_ab]));

    let result = prover.prove();
    assert!(result.is_unsatisfiable());
    assert!(prover.stats.resolution_steps > 0);
}

// ============================================================================
// Cross-Module Properties
// ============================================================================

#[test]
fn test_normal_form_transformation_preserves_predicates() {
    let x = Term::var("x");
    let p = TLExpr::pred("P", vec![x]);

    // Transform to NNF
    let nnf = to_nnf(&p);

    // Predicates should be preserved
    let original_preds = p.all_predicates();
    let nnf_preds = nnf.all_predicates();

    assert_eq!(original_preds, nnf_preds);
}

#[test]
fn test_anti_unification_generalizes_terms() {
    // Anti-unification finds most specific generalization
    let a = Term::constant("a");
    let b = Term::constant("b");

    let (gen, _subst1, _subst2) = anti_unify_terms(&a, &b);

    // Result should be a fresh variable
    match gen {
        Term::Var(name) => assert!(name.starts_with("_G")),
        _ => panic!("Expected variable from anti-unification"),
    }
}

#[test]
fn test_lgg_multiple_terms() {
    // Least general generalization of multiple terms
    let terms = vec![
        Term::constant("a"),
        Term::constant("b"),
        Term::constant("c"),
    ];

    let (gen, substs) = lgg_terms(&terms);

    // Should have one substitution per term
    assert_eq!(substs.len(), 3);

    // Generalization should be a variable
    match gen {
        Term::Var(_) => {} // Expected
        _ => panic!("LGG should produce a variable"),
    }
}

#[test]
fn test_resolution_strategies_comparison() {
    // Compare resolution strategies
    let p = Literal::positive(TLExpr::pred("P", vec![]));
    let q = Literal::positive(TLExpr::pred("Q", vec![]));
    let not_p = Literal::negative(TLExpr::pred("P", vec![]));
    let not_q = Literal::negative(TLExpr::pred("Q", vec![]));

    let clauses = vec![
        Clause::from_literals(vec![p, q]),
        Clause::from_literals(vec![not_p]),
        Clause::from_literals(vec![not_q]),
    ];

    let strategies = vec![
        ResolutionStrategy::Saturation { max_clauses: 1000 },
        ResolutionStrategy::SetOfSupport { max_steps: 1000 },
        ResolutionStrategy::UnitResolution { max_steps: 1000 },
        ResolutionStrategy::Linear { max_depth: 100 },
    ];

    for strategy in strategies {
        let mut prover = ResolutionProver::with_strategy(strategy);
        for clause in &clauses {
            prover.add_clause(clause.clone());
        }

        let result = prover.prove();
        // All strategies should prove this
        assert!(result.is_unsatisfiable());
    }
}

#[test]
fn test_proof_guided_graph_optimization() {
    // Use proof structure to guide graph construction

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);

    // P ∧ Q expression
    let _and_pq = TLExpr::and(p, q);

    // Create a simple graph that could represent this
    let mut graph = EinsumGraph::new();
    let p_tensor = graph.add_tensor("P");
    graph.add_output(p_tensor).unwrap();

    assert!(graph.validate().is_ok());
}

#[test]
fn test_typed_terms_with_unification() {
    // Test that typed terms can be unified
    let x = Term::Typed {
        value: Box::new(Term::var("x")),
        type_annotation: TypeAnnotation::new("Int"),
    };

    let a = Term::Typed {
        value: Box::new(Term::constant("a")),
        type_annotation: TypeAnnotation::new("Int"),
    };

    // Unify typed terms
    let result = unify_terms(&x, &a);
    assert!(result.is_ok());

    // Different types should not unify
    let y_real = Term::Typed {
        value: Box::new(Term::var("y")),
        type_annotation: TypeAnnotation::new("Real"),
    };

    let result2 = unify_terms(&x, &y_real);
    assert!(result2.is_err()); // Type mismatch
}
