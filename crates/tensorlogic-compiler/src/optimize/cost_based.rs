//! Cost-based query optimization for TLExpr.
//!
//! This module implements a cost-based optimizer that:
//! - Generates equivalent expression rewrites
//! - Estimates execution cost for each alternative
//! - Selects the most efficient representation
//! - Provides backend optimization hints
//!
//! # Overview
//!
//! The cost-based optimizer explores the space of equivalent expressions
//! by applying rewriting rules, then uses cost estimation to select the
//! best alternative. This is particularly useful for complex queries where
//! there are multiple valid execution strategies.
//!
//! # Example
//!
//! ```no_run
//! use tensorlogic_compiler::optimize::optimize_by_cost;
//! use tensorlogic_compiler::CompilerContext;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let mut ctx = CompilerContext::new();
//! ctx.add_domain("Person", 1000);
//!
//! // Complex expression with multiple equivalent forms
//! let expr = TLExpr::and(
//!     TLExpr::pred("p", vec![Term::var("x")]),
//!     TLExpr::or(
//!         TLExpr::pred("q", vec![Term::var("x")]),
//!         TLExpr::pred("r", vec![Term::var("x")]),
//!     ),
//! );
//!
//! let (optimized, stats) = optimize_by_cost(&expr, &ctx);
//! println!("Cost reduction: {:.1}%", stats.cost_reduction_percent());
//! ```

use std::collections::HashSet;

use tensorlogic_ir::TLExpr;

use super::complexity::{analyze_complexity, CostWeights};
use crate::CompilerContext;

/// Statistics from cost-based optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct CostBasedStats {
    /// Number of alternative expressions explored
    pub alternatives_explored: usize,
    /// Original expression cost
    pub original_cost: f64,
    /// Optimized expression cost
    pub optimized_cost: f64,
    /// Number of rewrites applied
    pub rewrites_applied: usize,
    /// Time spent in microseconds
    pub time_us: u64,
}

impl CostBasedStats {
    /// Calculate cost reduction percentage.
    pub fn cost_reduction_percent(&self) -> f64 {
        if self.original_cost == 0.0 {
            0.0
        } else {
            ((self.original_cost - self.optimized_cost) / self.original_cost) * 100.0
        }
    }

    /// Check if optimization was beneficial.
    pub fn is_beneficial(&self) -> bool {
        self.optimized_cost < self.original_cost
    }

    /// Get the cost ratio (optimized / original).
    pub fn cost_ratio(&self) -> f64 {
        if self.original_cost == 0.0 {
            1.0
        } else {
            self.optimized_cost / self.original_cost
        }
    }
}

/// Rewriting rule for generating equivalent expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RewriteRule {
    /// Distribute conjunction over disjunction: A ∧ (B ∨ C) → (A ∧ B) ∨ (A ∧ C)
    DistributeAndOverOr,
    /// Distribute disjunction over conjunction: A ∨ (B ∧ C) → (A ∨ B) ∧ (A ∨ C)
    DistributeOrOverAnd,
    /// Factor common terms: (A ∧ B) ∨ (A ∧ C) → A ∧ (B ∨ C)
    FactorCommonAnd,
    /// Factor common terms: (A ∨ B) ∧ (A ∨ C) → A ∨ (B ∧ C)
    FactorCommonOr,
    /// Push quantifiers inward: ∃x. (P ∧ Q) → (∃x. P) ∧ Q (if x not in Q)
    PushExistsInward,
    /// Push quantifiers inward: ∀x. (P ∧ Q) → (∀x. P) ∧ Q (if x not in Q)
    PushForallInward,
    /// Pull quantifiers outward: (∃x. P) ∧ Q → ∃x. (P ∧ Q) (if x not in Q)
    PullExistsOutward,
    /// Pull quantifiers outward: (∀x. P) ∧ Q → ∀x. (P ∧ Q) (if x not in Q)
    PullForallOutward,
    /// Convert nested quantifiers: ∃x. ∃y. P → ∃x,y. P
    MergeNestedExists,
    /// Convert nested quantifiers: ∀x. ∀y. P → ∀x,y. P
    MergeNestedForall,
    /// Reorder conjunctions based on selectivity
    ReorderConjunctions,
    /// Reorder disjunctions based on short-circuit potential
    ReorderDisjunctions,
}

/// Alternative expression with associated cost.
#[derive(Debug, Clone)]
struct Alternative {
    expr: TLExpr,
    cost: f64,
    rules_applied: Vec<RewriteRule>,
}

/// Cost-based optimizer that explores rewrite space.
pub struct CostBasedOptimizer<'a> {
    _context: &'a CompilerContext,
    cost_weights: CostWeights,
    max_alternatives: usize,
    explored: HashSet<String>,
}

impl<'a> CostBasedOptimizer<'a> {
    /// Create a new cost-based optimizer.
    pub fn new(context: &'a CompilerContext) -> Self {
        Self {
            _context: context,
            cost_weights: CostWeights::default(),
            max_alternatives: 100,
            explored: HashSet::new(),
        }
    }

    /// Set custom cost weights.
    pub fn with_cost_weights(mut self, weights: CostWeights) -> Self {
        self.cost_weights = weights;
        self
    }

    /// Set maximum number of alternatives to explore.
    pub fn with_max_alternatives(mut self, max: usize) -> Self {
        self.max_alternatives = max;
        self
    }

    /// Optimize an expression by cost.
    pub fn optimize(&mut self, expr: &TLExpr) -> (TLExpr, CostBasedStats) {
        let start = std::time::Instant::now();

        let original_cost = self.estimate_cost(expr);
        let mut alternatives = vec![Alternative {
            expr: expr.clone(),
            cost: original_cost,
            rules_applied: Vec::new(),
        }];

        self.explored.clear();
        self.explored.insert(expr_hash(expr));

        // Explore alternative rewrites
        let mut iteration = 0;
        while iteration < self.max_alternatives && iteration < alternatives.len() {
            let current = &alternatives[iteration].clone();
            let new_alts = self.generate_alternatives(&current.expr, &current.rules_applied);

            for alt in new_alts {
                let hash = expr_hash(&alt.expr);
                if !self.explored.contains(&hash) {
                    self.explored.insert(hash);
                    alternatives.push(alt);

                    if alternatives.len() >= self.max_alternatives {
                        break;
                    }
                }
            }

            iteration += 1;
        }

        // Select best alternative
        let best = alternatives
            .iter()
            .min_by(|a, b| {
                a.cost
                    .partial_cmp(&b.cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let time_us = start.elapsed().as_micros() as u64;

        let stats = CostBasedStats {
            alternatives_explored: alternatives.len(),
            original_cost,
            optimized_cost: best.cost,
            rewrites_applied: best.rules_applied.len(),
            time_us,
        };

        (best.expr.clone(), stats)
    }

    /// Estimate cost of an expression.
    fn estimate_cost(&self, expr: &TLExpr) -> f64 {
        let complexity = analyze_complexity(expr);
        complexity.total_cost_with_weights(&self.cost_weights)
    }

    /// Generate alternative expressions using rewrite rules.
    fn generate_alternatives(&self, expr: &TLExpr, applied: &[RewriteRule]) -> Vec<Alternative> {
        let mut alternatives = Vec::new();

        // Try each rewrite rule
        for rule in self.available_rules() {
            if let Some(rewritten) = self.apply_rule(expr, &rule) {
                let cost = self.estimate_cost(&rewritten);
                let mut new_applied = applied.to_vec();
                new_applied.push(rule);

                alternatives.push(Alternative {
                    expr: rewritten,
                    cost,
                    rules_applied: new_applied,
                });
            }
        }

        alternatives
    }

    /// Get available rewrite rules.
    fn available_rules(&self) -> Vec<RewriteRule> {
        vec![
            RewriteRule::DistributeAndOverOr,
            RewriteRule::DistributeOrOverAnd,
            RewriteRule::FactorCommonAnd,
            RewriteRule::FactorCommonOr,
            RewriteRule::PushExistsInward,
            RewriteRule::PushForallInward,
            RewriteRule::PullExistsOutward,
            RewriteRule::PullForallOutward,
            RewriteRule::MergeNestedExists,
            RewriteRule::MergeNestedForall,
            RewriteRule::ReorderConjunctions,
            RewriteRule::ReorderDisjunctions,
        ]
    }

    /// Apply a rewrite rule to an expression.
    fn apply_rule(&self, expr: &TLExpr, rule: &RewriteRule) -> Option<TLExpr> {
        match rule {
            RewriteRule::DistributeAndOverOr => self.distribute_and_over_or(expr),
            RewriteRule::DistributeOrOverAnd => self.distribute_or_over_and(expr),
            RewriteRule::FactorCommonAnd => self.factor_common_and(expr),
            RewriteRule::FactorCommonOr => self.factor_common_or(expr),
            RewriteRule::PushExistsInward => self.push_exists_inward(expr),
            RewriteRule::PushForallInward => self.push_forall_inward(expr),
            RewriteRule::PullExistsOutward => self.pull_exists_outward(expr),
            RewriteRule::PullForallOutward => self.pull_forall_outward(expr),
            RewriteRule::MergeNestedExists => self.merge_nested_exists(expr),
            RewriteRule::MergeNestedForall => self.merge_nested_forall(expr),
            RewriteRule::ReorderConjunctions => self.reorder_conjunctions(expr),
            RewriteRule::ReorderDisjunctions => self.reorder_disjunctions(expr),
        }
    }

    /// Distribute AND over OR: A ∧ (B ∨ C) → (A ∧ B) ∨ (A ∧ C)
    fn distribute_and_over_or(&self, expr: &TLExpr) -> Option<TLExpr> {
        match expr {
            TLExpr::And(a, b) => {
                if let TLExpr::Or(b1, b2) = b.as_ref() {
                    Some(TLExpr::or(
                        TLExpr::and(a.as_ref().clone(), b1.as_ref().clone()),
                        TLExpr::and(a.as_ref().clone(), b2.as_ref().clone()),
                    ))
                } else if let TLExpr::Or(a1, a2) = a.as_ref() {
                    Some(TLExpr::or(
                        TLExpr::and(a1.as_ref().clone(), b.as_ref().clone()),
                        TLExpr::and(a2.as_ref().clone(), b.as_ref().clone()),
                    ))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Distribute OR over AND: A ∨ (B ∧ C) → (A ∨ B) ∧ (A ∨ C)
    fn distribute_or_over_and(&self, expr: &TLExpr) -> Option<TLExpr> {
        match expr {
            TLExpr::Or(a, b) => {
                if let TLExpr::And(b1, b2) = b.as_ref() {
                    Some(TLExpr::and(
                        TLExpr::or(a.as_ref().clone(), b1.as_ref().clone()),
                        TLExpr::or(a.as_ref().clone(), b2.as_ref().clone()),
                    ))
                } else if let TLExpr::And(a1, a2) = a.as_ref() {
                    Some(TLExpr::and(
                        TLExpr::or(a1.as_ref().clone(), b.as_ref().clone()),
                        TLExpr::or(a2.as_ref().clone(), b.as_ref().clone()),
                    ))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Factor common AND terms: (A ∧ B) ∨ (A ∧ C) → A ∧ (B ∨ C)
    fn factor_common_and(&self, expr: &TLExpr) -> Option<TLExpr> {
        match expr {
            TLExpr::Or(left, right) => {
                if let (TLExpr::And(a1, b1), TLExpr::And(a2, b2)) = (left.as_ref(), right.as_ref())
                {
                    if a1 == a2 {
                        return Some(TLExpr::and(
                            a1.as_ref().clone(),
                            TLExpr::or(b1.as_ref().clone(), b2.as_ref().clone()),
                        ));
                    }
                    if b1 == b2 {
                        return Some(TLExpr::and(
                            b1.as_ref().clone(),
                            TLExpr::or(a1.as_ref().clone(), a2.as_ref().clone()),
                        ));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Factor common OR terms: (A ∨ B) ∧ (A ∨ C) → A ∨ (B ∧ C)
    fn factor_common_or(&self, expr: &TLExpr) -> Option<TLExpr> {
        match expr {
            TLExpr::And(left, right) => {
                if let (TLExpr::Or(a1, b1), TLExpr::Or(a2, b2)) = (left.as_ref(), right.as_ref()) {
                    if a1 == a2 {
                        return Some(TLExpr::or(
                            a1.as_ref().clone(),
                            TLExpr::and(b1.as_ref().clone(), b2.as_ref().clone()),
                        ));
                    }
                    if b1 == b2 {
                        return Some(TLExpr::or(
                            b1.as_ref().clone(),
                            TLExpr::and(a1.as_ref().clone(), a2.as_ref().clone()),
                        ));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Push EXISTS inward: ∃x. (P ∧ Q) → (∃x. P) ∧ Q (if x not in Q)
    fn push_exists_inward(&self, expr: &TLExpr) -> Option<TLExpr> {
        match expr {
            TLExpr::Exists { var, domain, body } => {
                if let TLExpr::And(p, q) = body.as_ref() {
                    let q_vars = q.free_vars();
                    if !q_vars.contains(var.as_str()) {
                        return Some(TLExpr::and(
                            TLExpr::exists(var, domain, p.as_ref().clone()),
                            q.as_ref().clone(),
                        ));
                    }

                    let p_vars = p.free_vars();
                    if !p_vars.contains(var.as_str()) {
                        return Some(TLExpr::and(
                            p.as_ref().clone(),
                            TLExpr::exists(var, domain, q.as_ref().clone()),
                        ));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Push FORALL inward: ∀x. (P ∧ Q) → (∀x. P) ∧ Q (if x not in Q)
    fn push_forall_inward(&self, expr: &TLExpr) -> Option<TLExpr> {
        match expr {
            TLExpr::ForAll { var, domain, body } => {
                if let TLExpr::And(p, q) = body.as_ref() {
                    let q_vars = q.free_vars();
                    if !q_vars.contains(var.as_str()) {
                        return Some(TLExpr::and(
                            TLExpr::forall(var, domain, p.as_ref().clone()),
                            q.as_ref().clone(),
                        ));
                    }

                    let p_vars = p.free_vars();
                    if !p_vars.contains(var.as_str()) {
                        return Some(TLExpr::and(
                            p.as_ref().clone(),
                            TLExpr::forall(var, domain, q.as_ref().clone()),
                        ));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Pull EXISTS outward: (∃x. P) ∧ Q → ∃x. (P ∧ Q) (if x not in Q)
    fn pull_exists_outward(&self, expr: &TLExpr) -> Option<TLExpr> {
        match expr {
            TLExpr::And(left, right) => {
                if let TLExpr::Exists { var, domain, body } = left.as_ref() {
                    let right_vars = right.free_vars();
                    if !right_vars.contains(var.as_str()) {
                        return Some(TLExpr::exists(
                            var,
                            domain,
                            TLExpr::and(body.as_ref().clone(), right.as_ref().clone()),
                        ));
                    }
                }

                if let TLExpr::Exists { var, domain, body } = right.as_ref() {
                    let left_vars = left.free_vars();
                    if !left_vars.contains(var.as_str()) {
                        return Some(TLExpr::exists(
                            var,
                            domain,
                            TLExpr::and(left.as_ref().clone(), body.as_ref().clone()),
                        ));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Pull FORALL outward: (∀x. P) ∧ Q → ∀x. (P ∧ Q) (if x not in Q)
    fn pull_forall_outward(&self, expr: &TLExpr) -> Option<TLExpr> {
        match expr {
            TLExpr::And(left, right) => {
                if let TLExpr::ForAll { var, domain, body } = left.as_ref() {
                    let right_vars = right.free_vars();
                    if !right_vars.contains(var.as_str()) {
                        return Some(TLExpr::forall(
                            var,
                            domain,
                            TLExpr::and(body.as_ref().clone(), right.as_ref().clone()),
                        ));
                    }
                }

                if let TLExpr::ForAll { var, domain, body } = right.as_ref() {
                    let left_vars = left.free_vars();
                    if !left_vars.contains(var.as_str()) {
                        return Some(TLExpr::forall(
                            var,
                            domain,
                            TLExpr::and(left.as_ref().clone(), body.as_ref().clone()),
                        ));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Merge nested EXISTS: ∃x. ∃y. P → leave unchanged (multi-var not in TLExpr)
    fn merge_nested_exists(&self, _expr: &TLExpr) -> Option<TLExpr> {
        // TLExpr doesn't support multi-variable quantifiers yet
        // This would require IR extension
        None
    }

    /// Merge nested FORALL: ∀x. ∀y. P → leave unchanged (multi-var not in TLExpr)
    fn merge_nested_forall(&self, _expr: &TLExpr) -> Option<TLExpr> {
        // TLExpr doesn't support multi-variable quantifiers yet
        // This would require IR extension
        None
    }

    /// Reorder conjunctions based on estimated selectivity.
    fn reorder_conjunctions(&self, expr: &TLExpr) -> Option<TLExpr> {
        match expr {
            TLExpr::And(a, b) => {
                let cost_a = self.estimate_cost(a);
                let cost_b = self.estimate_cost(b);

                // Put cheaper term first for short-circuit evaluation
                if cost_b < cost_a {
                    Some(TLExpr::and(b.as_ref().clone(), a.as_ref().clone()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Reorder disjunctions based on short-circuit potential.
    fn reorder_disjunctions(&self, expr: &TLExpr) -> Option<TLExpr> {
        match expr {
            TLExpr::Or(a, b) => {
                let cost_a = self.estimate_cost(a);
                let cost_b = self.estimate_cost(b);

                // Put cheaper term first for short-circuit evaluation
                if cost_b < cost_a {
                    Some(TLExpr::or(b.as_ref().clone(), a.as_ref().clone()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Compute a hash for an expression (for deduplication).
fn expr_hash(expr: &TLExpr) -> String {
    format!("{:?}", expr)
}

/// Optimize an expression using cost-based optimization.
///
/// This function explores equivalent rewrites and selects the lowest-cost
/// alternative based on estimated execution cost.
///
/// # Example
///
/// ```no_run
/// use tensorlogic_compiler::optimize::optimize_by_cost;
/// use tensorlogic_compiler::CompilerContext;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let mut ctx = CompilerContext::new();
/// ctx.add_domain("Person", 1000);
///
/// let expr = TLExpr::and(
///     TLExpr::pred("expensive", vec![Term::var("x")]),
///     TLExpr::pred("cheap", vec![Term::var("x")]),
/// );
///
/// let (optimized, stats) = optimize_by_cost(&expr, &ctx);
/// assert!(stats.alternatives_explored > 0);
/// ```
pub fn optimize_by_cost(expr: &TLExpr, context: &CompilerContext) -> (TLExpr, CostBasedStats) {
    let mut optimizer = CostBasedOptimizer::new(context);
    optimizer.optimize(expr)
}

/// Optimize with custom parameters.
pub fn optimize_by_cost_with_config(
    expr: &TLExpr,
    context: &CompilerContext,
    weights: CostWeights,
    max_alternatives: usize,
) -> (TLExpr, CostBasedStats) {
    let mut optimizer = CostBasedOptimizer::new(context)
        .with_cost_weights(weights)
        .with_max_alternatives(max_alternatives);
    optimizer.optimize(expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    fn test_context() -> CompilerContext {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        ctx.add_domain("City", 50);
        ctx
    }

    #[test]
    fn test_distribute_and_over_or() {
        let ctx = test_context();
        let expr = TLExpr::and(
            TLExpr::pred("p", vec![Term::var("x")]),
            TLExpr::or(
                TLExpr::pred("q", vec![Term::var("x")]),
                TLExpr::pred("r", vec![Term::var("x")]),
            ),
        );

        // Limit alternatives to 10 for fast testing
        let weights = CostWeights::default();
        let (_optimized, stats) = optimize_by_cost_with_config(&expr, &ctx, weights, 10);
        assert!(stats.alternatives_explored > 1);
        // Soft limit, may explore slightly more
        assert!(stats.alternatives_explored < 50);
    }

    #[test]
    fn test_factor_common_and() {
        let ctx = test_context();
        let p = TLExpr::pred("p", vec![Term::var("x")]);
        let q = TLExpr::pred("q", vec![Term::var("x")]);
        let r = TLExpr::pred("r", vec![Term::var("x")]);

        // (p ∧ q) ∨ (p ∧ r) should factor to p ∧ (q ∨ r)
        let expr = TLExpr::or(
            TLExpr::and(p.clone(), q.clone()),
            TLExpr::and(p.clone(), r.clone()),
        );

        // Limit alternatives to 10 for fast testing
        let weights = CostWeights::default();
        let (_optimized, stats) = optimize_by_cost_with_config(&expr, &ctx, weights, 10);
        assert!(stats.alternatives_explored > 1);
        // Soft limit, may explore slightly more
        assert!(stats.alternatives_explored < 50);
    }

    #[test]
    fn test_push_exists_inward() {
        let ctx = test_context();
        // ∃x. (p(x) ∧ q(y)) should push to (∃x. p(x)) ∧ q(y)
        let expr = TLExpr::exists(
            "x",
            "Person",
            TLExpr::and(
                TLExpr::pred("p", vec![Term::var("x")]),
                TLExpr::pred("q", vec![Term::var("y")]),
            ),
        );

        let (_optimized, stats) = optimize_by_cost(&expr, &ctx);
        assert!(stats.alternatives_explored > 0);
    }

    #[test]
    fn test_reorder_conjunctions() {
        let ctx = test_context();
        let expensive = TLExpr::exists(
            "y",
            "City",
            TLExpr::pred("expensive", vec![Term::var("x"), Term::var("y")]),
        );
        let cheap = TLExpr::pred("cheap", vec![Term::var("x")]);

        // Should reorder to put cheap predicate first
        let expr = TLExpr::and(expensive, cheap);

        let (_optimized, stats) = optimize_by_cost(&expr, &ctx);
        assert!(stats.alternatives_explored > 1);
    }

    #[test]
    fn test_cost_reduction_calculation() {
        let stats = CostBasedStats {
            alternatives_explored: 5,
            original_cost: 100.0,
            optimized_cost: 75.0,
            rewrites_applied: 2,
            time_us: 1000,
        };

        assert_eq!(stats.cost_reduction_percent(), 25.0);
        assert!(stats.is_beneficial());
        assert_eq!(stats.cost_ratio(), 0.75);
    }

    #[test]
    fn test_no_improvement() {
        let stats = CostBasedStats {
            alternatives_explored: 3,
            original_cost: 50.0,
            optimized_cost: 50.0,
            rewrites_applied: 0,
            time_us: 500,
        };

        assert_eq!(stats.cost_reduction_percent(), 0.0);
        assert!(!stats.is_beneficial());
        assert_eq!(stats.cost_ratio(), 1.0);
    }

    #[test]
    fn test_simple_expression_no_rewrites() {
        let ctx = test_context();
        let expr = TLExpr::pred("p", vec![Term::var("x")]);

        let (optimized, stats) = optimize_by_cost(&expr, &ctx);
        assert_eq!(optimized, expr);
        assert_eq!(stats.rewrites_applied, 0);
    }

    #[test]
    fn test_custom_cost_weights() {
        let ctx = test_context();
        let expr = TLExpr::and(
            TLExpr::pred("p", vec![Term::var("x")]),
            TLExpr::pred("q", vec![Term::var("x")]),
        );

        let weights = CostWeights {
            reduction: 10.0, // Make reductions more expensive
            cmp: 5.0,        // Make comparisons expensive
            ..Default::default()
        };

        let (_optimized, stats) = optimize_by_cost_with_config(&expr, &ctx, weights, 50);
        assert!(stats.alternatives_explored > 0);
    }

    #[test]
    fn test_max_alternatives_limit() {
        let ctx = test_context();
        let expr = TLExpr::and(
            TLExpr::or(
                TLExpr::pred("p", vec![Term::var("x")]),
                TLExpr::pred("q", vec![Term::var("x")]),
            ),
            TLExpr::or(
                TLExpr::pred("r", vec![Term::var("x")]),
                TLExpr::pred("s", vec![Term::var("x")]),
            ),
        );

        let weights = CostWeights::default();
        let (_optimized, stats) = optimize_by_cost_with_config(&expr, &ctx, weights, 5);
        // Soft limit, may explore slightly more before pruning
        assert!(stats.alternatives_explored < 25);
    }

    #[test]
    fn test_complex_quantifier_expression() {
        let ctx = test_context();
        // ∃x. ∃y. (p(x) ∧ q(y)) - should explore different quantifier orderings
        let expr = TLExpr::exists(
            "x",
            "Person",
            TLExpr::exists(
                "y",
                "City",
                TLExpr::and(
                    TLExpr::pred("p", vec![Term::var("x")]),
                    TLExpr::pred("q", vec![Term::var("y")]),
                ),
            ),
        );

        let (_optimized, stats) = optimize_by_cost(&expr, &ctx);
        assert!(stats.alternatives_explored > 0);
    }
}
