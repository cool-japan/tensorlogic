//! Integration tests for TLExpr to PGM conversion and inference.
//!
//! These tests demonstrate end-to-end workflows from logical expressions
//! to probabilistic inference.

use approx::assert_abs_diff_eq;
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_quantrs_hooks::{
    expr_to_factor_graph, InferenceEngine, MarginalizationQuery, MessagePassingAlgorithm,
    ParallelSumProduct, SumProductAlgorithm, VariableElimination,
};

/// Test basic predicate conversion to factor graph.
#[test]
fn test_single_predicate_conversion() {
    let expr = TLExpr::pred("P", vec![Term::var("x")]);
    let graph = expr_to_factor_graph(&expr).unwrap();

    assert_eq!(graph.num_variables(), 1);
    assert_eq!(graph.num_factors(), 1);
}

/// Test conjunction (AND) conversion.
#[test]
fn test_conjunction_conversion() {
    let expr = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("y")]),
    );

    let graph = expr_to_factor_graph(&expr).unwrap();

    assert_eq!(graph.num_variables(), 2);
    assert_eq!(graph.num_factors(), 2);
}

/// Test existential quantification.
#[test]
fn test_existential_quantification() {
    let expr = TLExpr::exists("x", "Domain", TLExpr::pred("P", vec![Term::var("x")]));

    let graph = expr_to_factor_graph(&expr).unwrap();

    assert_eq!(graph.num_variables(), 1);
    assert!(graph.get_variable("x").is_some());
}

/// Test nested logical expressions.
#[test]
fn test_nested_expressions() {
    // (P(x) ∧ Q(x)) ∧ R(y)
    let inner = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("x")]),
    );
    let expr = TLExpr::and(inner, TLExpr::pred("R", vec![Term::var("y")]));

    let graph = expr_to_factor_graph(&expr).unwrap();

    assert_eq!(graph.num_variables(), 2); // x, y
    assert_eq!(graph.num_factors(), 3); // P, Q, R
}

/// Test end-to-end inference from TLExpr.
#[test]
fn test_end_to_end_inference() {
    // Create a simple logical expression: P(x) ∧ Q(x, y)
    let expr = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("x"), Term::var("y")]),
    );

    // Convert to factor graph
    let graph = expr_to_factor_graph(&expr).unwrap();

    // Run inference
    let algorithm = SumProductAlgorithm::default();
    let marginals = algorithm.run(&graph).unwrap();

    // Check that we got marginals for all variables
    assert!(marginals.contains_key("x"));
    assert!(marginals.contains_key("y"));

    // Check normalization
    for marginal in marginals.values() {
        let sum: f64 = marginal.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

/// Test parallel inference from TLExpr.
#[test]
fn test_parallel_inference_from_tlexpr() {
    let expr = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("y")]),
    );

    let graph = expr_to_factor_graph(&expr).unwrap();

    // Run parallel inference
    let parallel_bp = ParallelSumProduct::default();
    let marginals = parallel_bp.run_parallel(&graph).unwrap();

    assert_eq!(marginals.len(), 2);

    for marginal in marginals.values() {
        let sum: f64 = marginal.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

/// Test variable elimination from TLExpr.
#[test]
fn test_variable_elimination_from_tlexpr() {
    let expr = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("x"), Term::var("y")]),
    );

    let graph = expr_to_factor_graph(&expr).unwrap();

    // Run variable elimination
    let ve = VariableElimination::new();
    let marginal_x = ve.marginalize(&graph, "x").unwrap();

    // Check result
    assert_eq!(marginal_x.len(), 2); // Binary variable
    let sum: f64 = marginal_x.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
}

/// Test inference engine with TLExpr.
#[test]
fn test_inference_engine_with_tlexpr() {
    let expr = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("y")]),
    );

    let graph = expr_to_factor_graph(&expr).unwrap();

    // Use inference engine
    let algorithm = Box::new(SumProductAlgorithm::default());
    let engine = InferenceEngine::new(graph, algorithm);

    let query = MarginalizationQuery {
        variable: "x".to_string(),
    };

    let marginal = engine.marginalize(&query).unwrap();

    let sum: f64 = marginal.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
}

/// Test implication conversion.
#[test]
fn test_implication_conversion() {
    // P(x) → Q(x)
    let expr = TLExpr::imply(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("x")]),
    );

    let graph = expr_to_factor_graph(&expr).unwrap();

    // Both predicates should be in the graph
    assert!(graph.num_factors() >= 2);
    assert!(graph.get_variable("x").is_some());
}

/// Test universal quantification.
#[test]
fn test_universal_quantification() {
    // ∀x. P(x)
    let expr = TLExpr::forall("x", "Domain", TLExpr::pred("P", vec![Term::var("x")]));

    let graph = expr_to_factor_graph(&expr).unwrap();

    assert!(graph.get_variable("x").is_some());
}

/// Test negation in expressions.
#[test]
fn test_negation_conversion() {
    // ¬P(x)
    let expr = TLExpr::negate(TLExpr::pred("P", vec![Term::var("x")]));

    let graph = expr_to_factor_graph(&expr).unwrap();

    assert!(graph.get_variable("x").is_some());
}

/// Test complex nested quantifiers.
#[test]
fn test_nested_quantifiers() {
    // ∃x. ∀y. P(x, y)
    let inner = TLExpr::forall(
        "y",
        "Domain",
        TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]),
    );
    let expr = TLExpr::exists("x", "Domain", inner);

    let graph = expr_to_factor_graph(&expr).unwrap();

    assert!(graph.get_variable("x").is_some());
    assert!(graph.get_variable("y").is_some());
}

/// Test multiple predicates with shared variables.
#[test]
fn test_shared_variables() {
    // P(x) ∧ Q(x) ∧ R(x, y)
    let expr = TLExpr::and(
        TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("x")]),
        ),
        TLExpr::pred("R", vec![Term::var("x"), Term::var("y")]),
    );

    let graph = expr_to_factor_graph(&expr).unwrap();

    assert_eq!(graph.num_variables(), 2); // x, y
    assert_eq!(graph.num_factors(), 3); // P, Q, R

    // x should be connected to all three factors
    if let Some(factors) = graph.get_adjacent_factors("x") {
        assert_eq!(factors.len(), 3);
    } else {
        panic!("x should be connected to factors");
    }
}

/// Test probabilistic reasoning from logical rules.
#[test]
fn test_probabilistic_reasoning() {
    // Simple rule: P(x) ∧ Q(x, y)
    let expr = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("x"), Term::var("y")]),
    );

    let graph = expr_to_factor_graph(&expr).unwrap();

    // Run inference with both serial and parallel algorithms
    let serial_bp = SumProductAlgorithm::default();
    let serial_marginals = serial_bp.run(&graph).unwrap();

    let parallel_bp = ParallelSumProduct::default();
    let parallel_marginals = parallel_bp.run_parallel(&graph).unwrap();

    // Results should be approximately equal
    for var in ["x", "y"] {
        let serial_m = &serial_marginals[var];
        let parallel_m = &parallel_marginals[var];

        for i in 0..serial_m.len() {
            assert_abs_diff_eq!(serial_m[[i]], parallel_m[[i]], epsilon = 1e-5);
        }
    }
}
