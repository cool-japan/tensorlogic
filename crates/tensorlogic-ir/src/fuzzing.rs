//! Fuzzing and stress testing infrastructure for the IR.
//!
//! This module provides tools for testing the robustness of the TensorLogic IR through:
//! - Stress testing with deeply nested expressions
//! - Edge case testing (empty, large, boundary values)
//! - Invariant checking across operations
//! - Malformed input handling
//! - Random expression generation for property-based testing
//! - Mutation-based fuzzing
//!
//! The module supports both deterministic stress tests and random property-based testing
//! through proptest integration.

use std::collections::HashSet;
use std::panic::AssertUnwindSafe;

use crate::{EinsumGraph, EinsumNode, TLExpr, Term};

/// Configuration for random expression generation.
#[derive(Debug, Clone)]
pub struct ExprGenConfig {
    /// Maximum depth of nested expressions
    pub max_depth: usize,
    /// Maximum number of arguments for predicates
    pub max_arity: usize,
    /// Maximum number of variables to use
    pub max_vars: usize,
    /// Maximum number of predicates to use
    pub max_predicates: usize,
    /// Probability of generating a quantifier (0.0 - 1.0)
    pub quantifier_probability: f64,
    /// Probability of generating an arithmetic operation
    pub arithmetic_probability: f64,
    /// Domains to use for quantifiers
    pub domains: Vec<String>,
}

impl Default for ExprGenConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_arity: 3,
            max_vars: 10,
            max_predicates: 5,
            quantifier_probability: 0.2,
            arithmetic_probability: 0.1,
            domains: vec!["Entity".to_string(), "Int".to_string(), "Bool".to_string()],
        }
    }
}

impl ExprGenConfig {
    /// Create a minimal config for quick tests
    pub fn minimal() -> Self {
        Self {
            max_depth: 2,
            max_arity: 2,
            max_vars: 3,
            max_predicates: 2,
            quantifier_probability: 0.1,
            arithmetic_probability: 0.05,
            domains: vec!["Entity".to_string()],
        }
    }

    /// Create a stress test config with deep nesting
    pub fn stress() -> Self {
        Self {
            max_depth: 10,
            max_arity: 5,
            max_vars: 20,
            max_predicates: 10,
            quantifier_probability: 0.3,
            arithmetic_probability: 0.2,
            domains: vec![
                "Entity".to_string(),
                "Int".to_string(),
                "Bool".to_string(),
                "Real".to_string(),
            ],
        }
    }
}

/// Simple deterministic random number generator for reproducible tests.
/// Uses a linear congruential generator.
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG with a seed
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next random u64
    pub fn next_u64(&mut self) -> u64 {
        // LCG parameters (same as glibc)
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    /// Generate a random number in range [0, max)
    pub fn gen_range(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }

    /// Generate a random f64 in [0, 1)
    pub fn gen_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Generate a random bool with given probability of true
    pub fn gen_bool(&mut self, probability: f64) -> bool {
        self.gen_f64() < probability
    }

    /// Choose a random element from a slice
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            Some(&items[self.gen_range(items.len())])
        }
    }
}

/// Random expression generator.
pub struct ExprGenerator {
    config: ExprGenConfig,
    rng: SimpleRng,
    var_names: Vec<String>,
    pred_names: Vec<String>,
}

impl ExprGenerator {
    /// Create a new generator with config and seed
    pub fn new(config: ExprGenConfig, seed: u64) -> Self {
        let var_names: Vec<String> = (0..config.max_vars).map(|i| format!("x{}", i)).collect();
        let pred_names: Vec<String> = (0..config.max_predicates)
            .map(|i| format!("P{}", i))
            .collect();

        Self {
            config,
            rng: SimpleRng::new(seed),
            var_names,
            pred_names,
        }
    }

    /// Generate a random variable term
    pub fn gen_var(&mut self) -> Term {
        let name = self.rng.choose(&self.var_names).unwrap().clone();
        Term::var(name)
    }

    /// Generate a random constant term
    pub fn gen_const(&mut self) -> Term {
        let value = format!("c{}", self.rng.gen_range(100));
        Term::constant(value)
    }

    /// Generate a random term
    pub fn gen_term(&mut self) -> Term {
        if self.rng.gen_bool(0.7) {
            self.gen_var()
        } else {
            self.gen_const()
        }
    }

    /// Generate a random predicate expression
    pub fn gen_predicate(&mut self) -> TLExpr {
        let name = self.rng.choose(&self.pred_names).unwrap().clone();
        let arity = self.rng.gen_range(self.config.max_arity) + 1;
        let args: Vec<Term> = (0..arity).map(|_| self.gen_term()).collect();
        TLExpr::pred(name, args)
    }

    /// Generate a random expression with given depth limit
    pub fn gen_expr(&mut self, depth: usize) -> TLExpr {
        if depth == 0 {
            // Base case: generate atomic expression
            if self.rng.gen_bool(0.8) {
                self.gen_predicate()
            } else {
                TLExpr::constant(self.rng.gen_f64())
            }
        } else {
            // Choose expression type
            let choice = self.rng.gen_range(10);
            match choice {
                0 => self.gen_predicate(),
                1 => TLExpr::negate(self.gen_expr(depth - 1)),
                2 => TLExpr::and(self.gen_expr(depth - 1), self.gen_expr(depth - 1)),
                3 => TLExpr::or(self.gen_expr(depth - 1), self.gen_expr(depth - 1)),
                4 => TLExpr::imply(self.gen_expr(depth - 1), self.gen_expr(depth - 1)),
                5 if self.rng.gen_bool(self.config.quantifier_probability) => {
                    let var = self.rng.choose(&self.var_names).unwrap().clone();
                    let domain = self.rng.choose(&self.config.domains).unwrap().clone();
                    TLExpr::exists(var, domain, self.gen_expr(depth - 1))
                }
                6 if self.rng.gen_bool(self.config.quantifier_probability) => {
                    let var = self.rng.choose(&self.var_names).unwrap().clone();
                    let domain = self.rng.choose(&self.config.domains).unwrap().clone();
                    TLExpr::forall(var, domain, self.gen_expr(depth - 1))
                }
                7 if self.rng.gen_bool(self.config.arithmetic_probability) => {
                    TLExpr::add(self.gen_expr(depth - 1), self.gen_expr(depth - 1))
                }
                8 if self.rng.gen_bool(self.config.arithmetic_probability) => {
                    TLExpr::mul(self.gen_expr(depth - 1), self.gen_expr(depth - 1))
                }
                _ => self.gen_predicate(),
            }
        }
    }

    /// Generate a random expression with default depth
    pub fn gen(&mut self) -> TLExpr {
        let depth = self.rng.gen_range(self.config.max_depth) + 1;
        self.gen_expr(depth)
    }
}

/// Mutation operations for expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationKind {
    /// Negate the expression
    Negate,
    /// Wrap in existential quantifier
    WrapExists,
    /// Wrap in universal quantifier
    WrapForall,
    /// Add conjunction with random expression
    AndWith,
    /// Add disjunction with random expression
    OrWith,
    /// Replace a subexpression
    ReplaceSubexpr,
    /// Duplicate expression (wrap in AND with self)
    Duplicate,
}

/// Mutate an expression
pub fn mutate_expr(expr: &TLExpr, mutation: MutationKind, rng: &mut SimpleRng) -> TLExpr {
    match mutation {
        MutationKind::Negate => TLExpr::negate(expr.clone()),
        MutationKind::WrapExists => {
            let var = format!("mut_x{}", rng.gen_range(100));
            TLExpr::exists(var, "Entity", expr.clone())
        }
        MutationKind::WrapForall => {
            let var = format!("mut_x{}", rng.gen_range(100));
            TLExpr::forall(var, "Entity", expr.clone())
        }
        MutationKind::AndWith => {
            let mut gen = ExprGenerator::new(ExprGenConfig::minimal(), rng.next_u64());
            TLExpr::and(expr.clone(), gen.gen_predicate())
        }
        MutationKind::OrWith => {
            let mut gen = ExprGenerator::new(ExprGenConfig::minimal(), rng.next_u64());
            TLExpr::or(expr.clone(), gen.gen_predicate())
        }
        MutationKind::ReplaceSubexpr => {
            // For simplicity, just wrap in a new operation
            let mut gen = ExprGenerator::new(ExprGenConfig::minimal(), rng.next_u64());
            if rng.gen_bool(0.5) {
                TLExpr::and(expr.clone(), gen.gen_predicate())
            } else {
                TLExpr::or(expr.clone(), gen.gen_predicate())
            }
        }
        MutationKind::Duplicate => TLExpr::and(expr.clone(), expr.clone()),
    }
}

/// Apply a random mutation to an expression
pub fn random_mutation(expr: &TLExpr, rng: &mut SimpleRng) -> TLExpr {
    let mutations = [
        MutationKind::Negate,
        MutationKind::WrapExists,
        MutationKind::WrapForall,
        MutationKind::AndWith,
        MutationKind::OrWith,
        MutationKind::Duplicate,
    ];
    let mutation = *rng.choose(&mutations).unwrap();
    mutate_expr(expr, mutation, rng)
}

/// Apply multiple random mutations to an expression
pub fn multi_mutate(expr: &TLExpr, num_mutations: usize, rng: &mut SimpleRng) -> TLExpr {
    let mut result = expr.clone();
    for _ in 0..num_mutations {
        result = random_mutation(&result, rng);
    }
    result
}

/// Configuration for graph generation
#[derive(Debug, Clone)]
pub struct GraphGenConfig {
    /// Maximum number of tensors
    pub max_tensors: usize,
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// Probability of creating an einsum operation
    pub einsum_probability: f64,
}

impl Default for GraphGenConfig {
    fn default() -> Self {
        Self {
            max_tensors: 10,
            max_nodes: 5,
            einsum_probability: 0.3,
        }
    }
}

/// Generate a random graph for testing
pub fn gen_random_graph(config: &GraphGenConfig, rng: &mut SimpleRng) -> EinsumGraph {
    let mut graph = EinsumGraph::new();

    // Add random tensors
    let num_tensors = rng.gen_range(config.max_tensors) + 1;
    let mut tensors = Vec::new();
    for i in 0..num_tensors {
        tensors.push(graph.add_tensor(format!("t{}", i)));
    }

    // Add random nodes
    let num_nodes = rng.gen_range(config.max_nodes);
    for _ in 0..num_nodes {
        if tensors.len() < 2 {
            break;
        }

        // Pick random operation
        if rng.gen_bool(config.einsum_probability) && tensors.len() >= 2 {
            // Einsum operation
            let idx1 = rng.gen_range(tensors.len());
            let idx2 = rng.gen_range(tensors.len());
            if idx1 != idx2 {
                let out = graph.add_tensor(format!("out_{}", graph.tensor_count()));
                let _ = graph.add_node(EinsumNode::einsum(
                    "ij,jk->ik",
                    vec![tensors[idx1], tensors[idx2]],
                    vec![out],
                ));
                tensors.push(out);
            }
        } else if !tensors.is_empty() {
            // Element-wise unary operation
            let idx = rng.gen_range(tensors.len());
            let out = graph.add_tensor(format!("out_{}", graph.tensor_count()));
            let ops = ["neg", "exp", "log", "relu"];
            let op = *rng.choose(&ops).unwrap();
            let _ = graph.add_node(EinsumNode::elem_unary(op, tensors[idx], out));
            tensors.push(out);
        }
    }

    // Set output
    if !tensors.is_empty() {
        let output_idx = rng.gen_range(tensors.len());
        let _ = graph.add_output(tensors[output_idx]);
    }

    graph
}

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

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);

        // Test determinism
        let vals: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();

        let mut rng2 = SimpleRng::new(42);
        let vals2: Vec<u64> = (0..5).map(|_| rng2.next_u64()).collect();

        assert_eq!(vals, vals2, "RNG should be deterministic with same seed");
    }

    #[test]
    fn test_rng_gen_range() {
        let mut rng = SimpleRng::new(123);

        // Test that gen_range produces values in range
        for _ in 0..100 {
            let val = rng.gen_range(10);
            assert!(val < 10, "gen_range should produce values < max");
        }
    }

    #[test]
    fn test_expr_generator_basic() {
        let config = ExprGenConfig::minimal();
        let mut gen = ExprGenerator::new(config, 42);

        // Generate several expressions
        for _ in 0..10 {
            let expr = gen.gen();
            // Just verify no panic and operations work
            let _ = expr.free_vars();
            let _ = expr.all_predicates();
        }
    }

    #[test]
    fn test_expr_generator_deterministic() {
        let config = ExprGenConfig::minimal();

        let mut gen1 = ExprGenerator::new(config.clone(), 42);
        let expr1 = gen1.gen();

        let mut gen2 = ExprGenerator::new(config, 42);
        let expr2 = gen2.gen();

        assert_eq!(expr1, expr2, "Same seed should produce same expression");
    }

    #[test]
    fn test_expr_generator_stress() {
        let config = ExprGenConfig::stress();
        let mut gen = ExprGenerator::new(config, 12345);

        // Generate complex expressions
        for _ in 0..5 {
            let expr = gen.gen();
            let stats = fuzz_expression_operations(&expr);
            assert_eq!(
                stats.panics_caught, 0,
                "Stress-generated expressions should not panic"
            );
        }
    }

    #[test]
    fn test_mutation_negate() {
        let mut rng = SimpleRng::new(42);
        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        let mutated = mutate_expr(&expr, MutationKind::Negate, &mut rng);

        match mutated {
            TLExpr::Not { .. } => {} // Expected
            _ => panic!("Negate should produce Not expression"),
        }
    }

    #[test]
    fn test_mutation_wrap_quantifiers() {
        let mut rng = SimpleRng::new(42);
        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        let exists = mutate_expr(&expr, MutationKind::WrapExists, &mut rng);
        match exists {
            TLExpr::Exists { .. } => {}
            _ => panic!("WrapExists should produce Exists expression"),
        }

        let forall = mutate_expr(&expr, MutationKind::WrapForall, &mut rng);
        match forall {
            TLExpr::ForAll { .. } => {}
            _ => panic!("WrapForall should produce ForAll expression"),
        }
    }

    #[test]
    fn test_random_mutation() {
        let mut rng = SimpleRng::new(42);
        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        // Apply random mutations
        for _ in 0..10 {
            let mutated = random_mutation(&expr, &mut rng);
            // Just verify it produces valid expressions
            let stats = fuzz_expression_operations(&mutated);
            assert_eq!(stats.panics_caught, 0);
        }
    }

    #[test]
    fn test_multi_mutate() {
        let mut rng = SimpleRng::new(42);
        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        // Apply multiple mutations
        let mutated = multi_mutate(&expr, 5, &mut rng);

        // Should still be a valid expression
        let stats = fuzz_expression_operations(&mutated);
        assert_eq!(stats.panics_caught, 0);
    }

    #[test]
    fn test_gen_random_graph() {
        let config = GraphGenConfig::default();
        let mut rng = SimpleRng::new(42);

        // Generate several graphs
        for _ in 0..10 {
            let graph = gen_random_graph(&config, &mut rng);
            let stats = fuzz_graph_validation(&graph);
            assert_eq!(stats.panics_caught, 0);
        }
    }

    #[test]
    fn test_graph_gen_config_variations() {
        let mut rng = SimpleRng::new(42);

        // Small graph
        let small_config = GraphGenConfig {
            max_tensors: 3,
            max_nodes: 2,
            einsum_probability: 0.5,
        };
        let small_graph = gen_random_graph(&small_config, &mut rng);
        assert!(small_graph.tensor_count() <= 10); // Including generated outputs

        // Larger graph
        let large_config = GraphGenConfig {
            max_tensors: 20,
            max_nodes: 10,
            einsum_probability: 0.5,
        };
        let large_graph = gen_random_graph(&large_config, &mut rng);
        let _ = large_graph.validate();
    }

    #[test]
    fn test_expr_gen_config_presets() {
        // Test minimal config
        let minimal = ExprGenConfig::minimal();
        assert_eq!(minimal.max_depth, 2);
        assert_eq!(minimal.max_arity, 2);

        // Test stress config
        let stress = ExprGenConfig::stress();
        assert_eq!(stress.max_depth, 10);
        assert_eq!(stress.max_arity, 5);
    }

    #[test]
    fn test_generated_expressions_have_valid_free_vars() {
        let config = ExprGenConfig::default();
        let mut gen = ExprGenerator::new(config, 99);

        for _ in 0..20 {
            let expr = gen.gen();
            let free_vars = expr.free_vars();

            // All free vars should be from our variable pool
            for var in &free_vars {
                let is_valid = var.starts_with('x') || var.starts_with("mut_x");
                assert!(is_valid, "Free var '{}' should be from generator pool", var);
            }
        }
    }

    #[test]
    fn test_generated_predicates_have_valid_names() {
        let config = ExprGenConfig::default();
        let mut gen = ExprGenerator::new(config, 77);

        for _ in 0..20 {
            let expr = gen.gen();
            let predicates = expr.all_predicates();

            // All predicates should be from our predicate pool
            for pred in predicates.keys() {
                assert!(
                    pred.starts_with('P'),
                    "Predicate '{}' should be from generator pool",
                    pred
                );
            }
        }
    }

    #[test]
    fn test_mutation_preserves_expression_validity() {
        let mut rng = SimpleRng::new(555);
        let config = ExprGenConfig::default();
        let mut gen = ExprGenerator::new(config, 666);

        // Generate expressions and mutate them
        for _ in 0..10 {
            let expr = gen.gen();

            // Apply each mutation type
            for mutation in [
                MutationKind::Negate,
                MutationKind::WrapExists,
                MutationKind::WrapForall,
                MutationKind::AndWith,
                MutationKind::OrWith,
                MutationKind::Duplicate,
            ] {
                let mutated = mutate_expr(&expr, mutation, &mut rng);

                // Verify invariants still hold
                assert!(
                    check_expression_invariants(&mutated),
                    "Mutation {:?} broke invariants",
                    mutation
                );
            }
        }
    }

    #[test]
    fn test_fuzz_many_random_expressions() {
        let config = ExprGenConfig::default();
        let mut gen = ExprGenerator::new(config, 12345);
        let mut total_stats = FuzzStats::new();

        // Generate and fuzz 100 expressions
        for _ in 0..100 {
            let expr = gen.gen();
            let stats = fuzz_expression_operations(&expr);
            total_stats.tests_run += stats.tests_run;
            total_stats.tests_passed += stats.tests_passed;
            total_stats.tests_failed += stats.tests_failed;
            total_stats.panics_caught += stats.panics_caught;
        }

        assert_eq!(
            total_stats.panics_caught, 0,
            "Random expressions should not cause panics"
        );
        assert!(
            total_stats.success_rate() > 0.95,
            "Success rate should be > 95%"
        );
    }

    #[test]
    fn test_rng_choose() {
        let mut rng = SimpleRng::new(42);
        let items = vec!["a", "b", "c", "d"];

        // Choose many times
        let mut seen = HashSet::new();
        for _ in 0..100 {
            let item = rng.choose(&items).unwrap();
            seen.insert(*item);
        }

        // Should eventually see all items (probabilistically)
        assert!(seen.len() >= 2, "Should see multiple distinct items");
    }

    #[test]
    fn test_rng_gen_bool() {
        let mut rng = SimpleRng::new(42);

        // With probability 0, should always return false
        for _ in 0..10 {
            assert!(!rng.gen_bool(0.0));
        }

        // With probability 1, should always return true
        for _ in 0..10 {
            assert!(rng.gen_bool(1.0));
        }
    }

    #[test]
    fn test_expr_generator_specific_depth() {
        let config = ExprGenConfig::default();
        let mut gen = ExprGenerator::new(config, 42);

        // Generate expression with depth 0 (should be atomic)
        let atomic = gen.gen_expr(0);
        match atomic {
            TLExpr::Pred { .. } | TLExpr::Constant { .. } => {}
            _ => panic!("Depth 0 should produce atomic expression"),
        }
    }

    #[test]
    fn test_graph_output_is_valid() {
        let config = GraphGenConfig::default();
        let mut rng = SimpleRng::new(42);

        for _ in 0..10 {
            let graph = gen_random_graph(&config, &mut rng);

            // Should have at least one output (if tensors exist)
            if graph.tensor_count() > 0 {
                // Validate should catch invalid outputs
                let _ = graph.validate();
            }
        }
    }
}
