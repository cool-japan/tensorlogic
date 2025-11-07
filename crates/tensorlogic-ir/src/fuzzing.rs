//! Fuzzing and stress testing infrastructure for the IR.
//!
//! This module provides tools for testing the robustness of the TensorLogic IR through:
//! - Stress testing with deeply nested expressions
//! - Edge case testing (empty, large, boundary values)
//! - Invariant checking across operations
//! - Malformed input handling
//!
//! Unlike traditional fuzzing which uses random generation, this module focuses on
//! deterministic stress tests and edge cases that can reliably reproduce issues.

use std::collections::HashSet;
use std::panic::AssertUnwindSafe;

use crate::{EinsumGraph, EinsumNode, TLExpr, Term};

/// Fuzz testing statistics.
#[derive(Debug, Clone, Default)]
pub struct FuzzStats {
    /// Number of tests run
    pub tests_run: usize,
    /// Number of tests that passed
    pub tests_passed: usize,
    /// Number of tests that failed
    pub tests_failed: usize,
    /// Number of panics caught
    pub panics_caught: usize,
    /// Unique error messages
    pub unique_errors: HashSet<String>,
}

impl FuzzStats {
    /// Create new stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful test.
    pub fn record_success(&mut self) {
        self.tests_run += 1;
        self.tests_passed += 1;
    }

    /// Record a failure.
    pub fn record_failure(&mut self, error: impl Into<String>) {
        self.tests_run += 1;
        self.tests_failed += 1;
        self.unique_errors.insert(error.into());
    }

    /// Record a panic.
    pub fn record_panic(&mut self) {
        self.tests_run += 1;
        self.panics_caught += 1;
    }

    /// Get success rate.
    pub fn success_rate(&self) -> f64 {
        if self.tests_run == 0 {
            return 1.0;
        }
        self.tests_passed as f64 / self.tests_run as f64
    }

    /// Print summary.
    pub fn summary(&self) -> String {
        format!(
            "Fuzz Stats:\n\
             - Tests run: {}\n\
             - Passed: {}\n\
             - Failed: {}\n\
             - Panics: {}\n\
             - Unique errors: {}\n\
             - Success rate: {:.2}%",
            self.tests_run,
            self.tests_passed,
            self.tests_failed,
            self.panics_caught,
            self.unique_errors.len(),
            self.success_rate() * 100.0
        )
    }
}

/// Test expression operations for robustness.
///
/// This function applies various operations to an expression and checks
/// that they don't panic and maintain invariants.
pub fn fuzz_expression_operations(expr: &TLExpr) -> FuzzStats {
    let mut stats = FuzzStats::new();

    // Test free_vars
    if std::panic::catch_unwind(|| expr.free_vars()).is_ok() {
        stats.record_success();
    } else {
        stats.record_panic();
    }

    // Test all_predicates
    if std::panic::catch_unwind(|| expr.all_predicates()).is_ok() {
        stats.record_success();
    } else {
        stats.record_panic();
    }

    // Test clone + equality
    if std::panic::catch_unwind(|| {
        let cloned = expr.clone();
        assert!(cloned == *expr);
    })
    .is_ok()
    {
        stats.record_success();
    } else {
        stats.record_panic();
    }

    // Test Debug formatting
    if std::panic::catch_unwind(|| format!("{:?}", expr)).is_ok() {
        stats.record_success();
    } else {
        stats.record_panic();
    }

    // Test serialization (serde is always enabled)
    if std::panic::catch_unwind(|| {
        let json = serde_json::to_string(expr).unwrap();
        let _deserialized: TLExpr = serde_json::from_str(&json).unwrap();
    })
    .is_ok()
    {
        stats.record_success();
    } else {
        stats.record_panic();
    }

    stats
}

/// Test graph validation for robustness.
pub fn fuzz_graph_validation(graph: &EinsumGraph) -> FuzzStats {
    let mut stats = FuzzStats::new();

    // Test validation
    if std::panic::catch_unwind(|| graph.validate()).is_ok() {
        stats.record_success();
    } else {
        stats.record_panic();
    }

    // Test clone
    if std::panic::catch_unwind(|| {
        let _cloned = graph.clone();
    })
    .is_ok()
    {
        stats.record_success();
    } else {
        stats.record_panic();
    }

    // Test Debug formatting
    if std::panic::catch_unwind(|| format!("{:?}", graph)).is_ok() {
        stats.record_success();
    } else {
        stats.record_panic();
    }

    stats
}

/// Create a deeply nested expression for stress testing.
///
/// Creates an expression of the form: ¬¬¬...¬P (depth negations)
pub fn create_deep_negation(depth: usize) -> TLExpr {
    let mut expr = TLExpr::pred("P", vec![Term::var("x")]);
    for _ in 0..depth {
        expr = TLExpr::negate(expr);
    }
    expr
}

/// Create a wide AND expression for stress testing.
///
/// Creates: P1 ∧ P2 ∧ P3 ∧ ... ∧ Pn
pub fn create_wide_and(width: usize) -> TLExpr {
    if width == 0 {
        return TLExpr::constant(1.0);
    }

    let mut expr = TLExpr::pred(format!("P{}", 0), vec![Term::var("x")]);
    for i in 1..width {
        expr = TLExpr::and(expr, TLExpr::pred(format!("P{}", i), vec![Term::var("x")]));
    }
    expr
}

/// Create a wide OR expression for stress testing.
pub fn create_wide_or(width: usize) -> TLExpr {
    if width == 0 {
        return TLExpr::constant(0.0);
    }

    let mut expr = TLExpr::pred(format!("P{}", 0), vec![Term::var("x")]);
    for i in 1..width {
        expr = TLExpr::or(expr, TLExpr::pred(format!("P{}", i), vec![Term::var("x")]));
    }
    expr
}

/// Create nested quantifiers for stress testing.
pub fn create_nested_quantifiers(depth: usize) -> TLExpr {
    let mut expr = TLExpr::pred("P", vec![Term::var(format!("x{}", depth))]);
    for i in (0..depth).rev() {
        let var = format!("x{}", i);
        if i % 2 == 0 {
            expr = TLExpr::exists(var, "Entity", expr);
        } else {
            expr = TLExpr::forall(var, "Entity", expr);
        }
    }
    expr
}

/// Test edge cases for expressions.
pub fn test_expression_edge_cases() -> FuzzStats {
    let mut stats = FuzzStats::new();

    let test_cases = vec![
        // Empty predicate name
        ("empty_name", TLExpr::pred("", vec![])),
        // Zero-arity predicate
        ("zero_arity", TLExpr::pred("P", vec![])),
        // Large arity predicate
        (
            "large_arity",
            TLExpr::pred(
                "P",
                (0..100).map(|i| Term::var(format!("x{}", i))).collect(),
            ),
        ),
        // Extreme constants
        ("max_float", TLExpr::constant(f64::MAX)),
        ("min_float", TLExpr::constant(f64::MIN)),
        ("inf", TLExpr::constant(f64::INFINITY)),
        ("neg_inf", TLExpr::constant(f64::NEG_INFINITY)),
        ("nan", TLExpr::constant(f64::NAN)),
        // Deep nesting
        ("deep_negation", create_deep_negation(100)),
        // Wide expressions
        ("wide_and", create_wide_and(100)),
        ("wide_or", create_wide_or(100)),
        // Nested quantifiers
        ("nested_quantifiers", create_nested_quantifiers(20)),
    ];

    for (name, expr) in test_cases {
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = expr.free_vars();
            let _ = expr.all_predicates();
            let _ = expr.clone();
        }));

        if result.is_ok() {
            stats.record_success();
        } else {
            stats.record_failure(format!("Edge case '{}' caused panic", name));
        }
    }

    stats
}

/// Test edge cases for graphs.
pub fn test_graph_edge_cases() -> FuzzStats {
    let mut stats = FuzzStats::new();

    // Empty graph
    let empty_graph = EinsumGraph::new();
    if std::panic::catch_unwind(AssertUnwindSafe(|| empty_graph.validate())).is_ok() {
        stats.record_success();
    } else {
        stats.record_failure("empty graph validation panicked");
    }

    // Graph with single tensor, no operations
    let mut single_tensor_graph = EinsumGraph::new();
    let t1 = single_tensor_graph.add_tensor("t1");
    single_tensor_graph.add_output(t1).ok();
    if std::panic::catch_unwind(AssertUnwindSafe(|| single_tensor_graph.validate())).is_ok() {
        stats.record_success();
    } else {
        stats.record_failure("single tensor graph validation panicked");
    }

    // Graph with many tensors
    let mut many_tensors_graph = EinsumGraph::new();
    let mut tensors = Vec::new();
    for i in 0..1000 {
        tensors.push(many_tensors_graph.add_tensor(format!("t{}", i)));
    }
    if !tensors.is_empty() {
        many_tensors_graph.add_output(tensors[0]).ok();
    }
    if std::panic::catch_unwind(AssertUnwindSafe(|| many_tensors_graph.validate())).is_ok() {
        stats.record_success();
    } else {
        stats.record_failure("many tensors graph validation panicked");
    }

    // Graph with out-of-bounds tensor reference
    let mut invalid_graph = EinsumGraph::new();
    let t1 = invalid_graph.add_tensor("t1");
    // Try to add node with invalid tensor ID (999 doesn't exist)
    let result = invalid_graph.add_node(EinsumNode::elem_binary("add", 999, t1, t1));
    if result.is_err() {
        stats.record_success(); // Should fail gracefully
    } else {
        stats.record_failure("invalid tensor reference not caught");
    }

    stats
}

/// Check invariants for an expression.
///
/// Returns true if all invariants hold.
pub fn check_expression_invariants(expr: &TLExpr) -> bool {
    // Invariant 1: Free vars of cloned expression should be the same
    let free_vars1 = expr.free_vars();
    let cloned = expr.clone();
    let free_vars2 = cloned.free_vars();
    if free_vars1 != free_vars2 {
        return false;
    }

    // Invariant 2: Predicates should be consistent across clones
    let preds1 = expr.all_predicates();
    let preds2 = cloned.all_predicates();
    if preds1 != preds2 {
        return false;
    }

    // Invariant 3: Equality should be reflexive
    #[allow(clippy::eq_op)]
    if expr != expr {
        return false;
    }

    // Invariant 4: Clone should be equal to original
    if expr != &cloned {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzz_expression_operations() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        let stats = fuzz_expression_operations(&expr);
        assert!(stats.success_rate() > 0.9);
        assert_eq!(stats.panics_caught, 0);
    }

    #[test]
    fn test_fuzz_graph_validation() {
        let mut graph = EinsumGraph::new();
        let t1 = graph.add_tensor("t1");
        graph.add_output(t1).ok();

        let stats = fuzz_graph_validation(&graph);
        assert!(stats.success_rate() > 0.9);
        assert_eq!(stats.panics_caught, 0);
    }

    #[test]
    fn test_deep_negation() {
        let expr = create_deep_negation(50);
        let stats = fuzz_expression_operations(&expr);
        assert_eq!(stats.panics_caught, 0);
    }

    #[test]
    fn test_wide_expressions() {
        let and_expr = create_wide_and(50);
        let or_expr = create_wide_or(50);

        let and_stats = fuzz_expression_operations(&and_expr);
        let or_stats = fuzz_expression_operations(&or_expr);

        assert_eq!(and_stats.panics_caught, 0);
        assert_eq!(or_stats.panics_caught, 0);
    }

    #[test]
    fn test_nested_quantifiers() {
        let expr = create_nested_quantifiers(10);
        let stats = fuzz_expression_operations(&expr);
        assert_eq!(stats.panics_caught, 0);
    }

    #[test]
    fn test_expression_invariants() {
        let expr = TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("y")]),
        );

        assert!(check_expression_invariants(&expr));
    }

    #[test]
    fn test_stress_edge_cases_compile() {
        // Just test that the edge case functions compile and run
        let _ = test_expression_edge_cases();
        let _ = test_graph_edge_cases();
        // Success if we get here without panicking
    }
}
