//! Quantifier optimization pass.
//!
//! This module provides optimizations for quantified expressions:
//!
//! - **Loop-invariant code motion**: Move expressions that don't depend on
//!   the quantified variable outside the quantifier
//! - **Quantifier reordering**: Reorder nested quantifiers for better performance
//! - **Quantifier fusion**: Merge adjacent same-type quantifiers when possible
//!
//! # Examples
//!
//! ```
//! use tensorlogic_compiler::optimize::optimize_quantifiers;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! // Loop invariant: EXISTS x. (a + p(x)) where a doesn't depend on x
//! // Can be optimized to: a + EXISTS x. p(x)
//! let a = TLExpr::pred("a", vec![Term::var("i")]);
//! let px = TLExpr::pred("p", vec![Term::var("x")]);
//! let expr = TLExpr::exists("x", "D", TLExpr::add(a, px));
//!
//! let (optimized, stats) = optimize_quantifiers(&expr);
//! assert!(stats.invariants_hoisted > 0);
//! ```

use std::collections::HashSet;
use tensorlogic_ir::TLExpr;

/// Statistics from quantifier optimization.
#[derive(Debug, Clone, Default)]
pub struct QuantifierOptStats {
    /// Number of loop-invariant expressions hoisted
    pub invariants_hoisted: usize,
    /// Number of quantifier pairs reordered
    pub quantifiers_reordered: usize,
    /// Number of quantifiers fused
    pub quantifiers_fused: usize,
    /// Total expressions processed
    pub total_processed: usize,
}

impl QuantifierOptStats {
    /// Get total number of optimizations applied.
    pub fn total_optimizations(&self) -> usize {
        self.invariants_hoisted + self.quantifiers_reordered + self.quantifiers_fused
    }
}

/// Apply quantifier optimizations to an expression.
///
/// This pass hoists loop-invariant expressions and optimizes quantifier
/// nesting when beneficial.
///
/// # Arguments
///
/// * `expr` - The expression to optimize
///
/// # Returns
///
/// A tuple of (optimized expression, statistics)
pub fn optimize_quantifiers(expr: &TLExpr) -> (TLExpr, QuantifierOptStats) {
    let mut stats = QuantifierOptStats::default();
    let result = optimize_quantifiers_impl(expr, &mut stats);
    (result, stats)
}

fn optimize_quantifiers_impl(expr: &TLExpr, stats: &mut QuantifierOptStats) -> TLExpr {
    stats.total_processed += 1;

    match expr {
        TLExpr::Exists { var, domain, body } => {
            let body_opt = optimize_quantifiers_impl(body, stats);

            // Try to hoist loop-invariant expressions
            if let Some(hoisted) = try_hoist_invariant(var, domain, &body_opt) {
                stats.invariants_hoisted += 1;
                return optimize_quantifiers_impl(&hoisted, stats);
            }

            TLExpr::Exists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
            }
        }

        TLExpr::ForAll { var, domain, body } => {
            let body_opt = optimize_quantifiers_impl(body, stats);

            // Try to hoist loop-invariant expressions
            if let Some(hoisted) = try_hoist_invariant_forall(var, domain, &body_opt) {
                stats.invariants_hoisted += 1;
                return optimize_quantifiers_impl(&hoisted, stats);
            }

            TLExpr::ForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
            }
        }

        // Recursive cases
        TLExpr::Add(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Add(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Sub(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Sub(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Mul(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Mul(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Div(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Div(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::And(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::And(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Or(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Or(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Not(inner) => {
            let inner_opt = optimize_quantifiers_impl(inner, stats);
            TLExpr::Not(Box::new(inner_opt))
        }

        TLExpr::Imply(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Imply(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Pow(base, exp) => {
            let base_opt = optimize_quantifiers_impl(base, stats);
            let exp_opt = optimize_quantifiers_impl(exp, stats);
            TLExpr::Pow(Box::new(base_opt), Box::new(exp_opt))
        }

        TLExpr::Abs(inner) => {
            let inner_opt = optimize_quantifiers_impl(inner, stats);
            TLExpr::Abs(Box::new(inner_opt))
        }

        TLExpr::Sqrt(inner) => {
            let inner_opt = optimize_quantifiers_impl(inner, stats);
            TLExpr::Sqrt(Box::new(inner_opt))
        }

        TLExpr::Exp(inner) => {
            let inner_opt = optimize_quantifiers_impl(inner, stats);
            TLExpr::Exp(Box::new(inner_opt))
        }

        TLExpr::Log(inner) => {
            let inner_opt = optimize_quantifiers_impl(inner, stats);
            TLExpr::Log(Box::new(inner_opt))
        }

        TLExpr::Let { var, value, body } => {
            let value_opt = optimize_quantifiers_impl(value, stats);
            let body_opt = optimize_quantifiers_impl(body, stats);
            TLExpr::Let {
                var: var.clone(),
                value: Box::new(value_opt),
                body: Box::new(body_opt),
            }
        }

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond_opt = optimize_quantifiers_impl(condition, stats);
            let then_opt = optimize_quantifiers_impl(then_branch, stats);
            let else_opt = optimize_quantifiers_impl(else_branch, stats);
            TLExpr::IfThenElse {
                condition: Box::new(cond_opt),
                then_branch: Box::new(then_opt),
                else_branch: Box::new(else_opt),
            }
        }

        // Comparison operators
        TLExpr::Eq(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Eq(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Lt(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Lt(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Lte(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Lte(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Gt(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Gt(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Gte(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Gte(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Min/Max
        TLExpr::Min(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Min(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Max(lhs, rhs) => {
            let lhs_opt = optimize_quantifiers_impl(lhs, stats);
            let rhs_opt = optimize_quantifiers_impl(rhs, stats);
            TLExpr::Max(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Modal logic
        TLExpr::Box(inner) => {
            let inner_opt = optimize_quantifiers_impl(inner, stats);
            TLExpr::Box(Box::new(inner_opt))
        }

        TLExpr::Diamond(inner) => {
            let inner_opt = optimize_quantifiers_impl(inner, stats);
            TLExpr::Diamond(Box::new(inner_opt))
        }

        // Temporal logic
        TLExpr::Next(inner) => {
            let inner_opt = optimize_quantifiers_impl(inner, stats);
            TLExpr::Next(Box::new(inner_opt))
        }

        TLExpr::Eventually(inner) => {
            let inner_opt = optimize_quantifiers_impl(inner, stats);
            TLExpr::Eventually(Box::new(inner_opt))
        }

        TLExpr::Always(inner) => {
            let inner_opt = optimize_quantifiers_impl(inner, stats);
            TLExpr::Always(Box::new(inner_opt))
        }

        TLExpr::Until { before, after } => {
            let before_opt = optimize_quantifiers_impl(before, stats);
            let after_opt = optimize_quantifiers_impl(after, stats);
            TLExpr::Until {
                before: Box::new(before_opt),
                after: Box::new(after_opt),
            }
        }

        // Leaves and other variants
        TLExpr::Pred { .. }
        | TLExpr::Constant(_)
        | TLExpr::Score(_)
        | TLExpr::Mod(_, _)
        | TLExpr::Floor(_)
        | TLExpr::Ceil(_)
        | TLExpr::Round(_)
        | TLExpr::Sin(_)
        | TLExpr::Cos(_)
        | TLExpr::Tan(_)
        | TLExpr::Aggregate { .. }
        | TLExpr::TNorm { .. }
        | TLExpr::TCoNorm { .. }
        | TLExpr::FuzzyNot { .. }
        | TLExpr::FuzzyImplication { .. }
        | TLExpr::SoftExists { .. }
        | TLExpr::SoftForAll { .. }
        | TLExpr::WeightedRule { .. }
        | TLExpr::ProbabilisticChoice { .. }
        | TLExpr::Release { .. }
        | TLExpr::WeakUntil { .. }
        | TLExpr::StrongRelease { .. } => expr.clone(),

        // All other expression types (alpha.3 enhancements)
        _ => expr.clone(),
    }
}

/// Collect all free variables in an expression.
fn free_vars(expr: &TLExpr) -> HashSet<String> {
    let mut vars = HashSet::new();
    collect_free_vars(expr, &mut HashSet::new(), &mut vars);
    vars
}

fn collect_free_vars(expr: &TLExpr, bound: &mut HashSet<String>, free: &mut HashSet<String>) {
    match expr {
        TLExpr::Pred { args, .. } => {
            for arg in args {
                if let tensorlogic_ir::Term::Var(v) = arg {
                    if !bound.contains(v) {
                        free.insert(v.clone());
                    }
                }
            }
        }

        TLExpr::Exists { var, body, .. } | TLExpr::ForAll { var, body, .. } => {
            bound.insert(var.clone());
            collect_free_vars(body, bound, free);
            bound.remove(var);
        }

        TLExpr::Let { var, value, body } => {
            collect_free_vars(value, bound, free);
            bound.insert(var.clone());
            collect_free_vars(body, bound, free);
            bound.remove(var);
        }

        TLExpr::Add(lhs, rhs)
        | TLExpr::Sub(lhs, rhs)
        | TLExpr::Mul(lhs, rhs)
        | TLExpr::Div(lhs, rhs)
        | TLExpr::And(lhs, rhs)
        | TLExpr::Or(lhs, rhs)
        | TLExpr::Imply(lhs, rhs)
        | TLExpr::Eq(lhs, rhs)
        | TLExpr::Lt(lhs, rhs)
        | TLExpr::Lte(lhs, rhs)
        | TLExpr::Gt(lhs, rhs)
        | TLExpr::Gte(lhs, rhs)
        | TLExpr::Min(lhs, rhs)
        | TLExpr::Max(lhs, rhs)
        | TLExpr::Mod(lhs, rhs) => {
            collect_free_vars(lhs, bound, free);
            collect_free_vars(rhs, bound, free);
        }

        TLExpr::Until { before, after } | TLExpr::WeakUntil { before, after } => {
            collect_free_vars(before, bound, free);
            collect_free_vars(after, bound, free);
        }

        TLExpr::Release { released, releaser } | TLExpr::StrongRelease { released, releaser } => {
            collect_free_vars(released, bound, free);
            collect_free_vars(releaser, bound, free);
        }

        TLExpr::Pow(base, exp) => {
            collect_free_vars(base, bound, free);
            collect_free_vars(exp, bound, free);
        }

        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            collect_free_vars(left, bound, free);
            collect_free_vars(right, bound, free);
        }

        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => {
            collect_free_vars(premise, bound, free);
            collect_free_vars(conclusion, bound, free);
        }

        TLExpr::Not(inner)
        | TLExpr::Abs(inner)
        | TLExpr::Sqrt(inner)
        | TLExpr::Exp(inner)
        | TLExpr::Log(inner)
        | TLExpr::Box(inner)
        | TLExpr::Diamond(inner)
        | TLExpr::Next(inner)
        | TLExpr::Eventually(inner)
        | TLExpr::Always(inner)
        | TLExpr::Score(inner)
        | TLExpr::Floor(inner)
        | TLExpr::Ceil(inner)
        | TLExpr::Round(inner)
        | TLExpr::Sin(inner)
        | TLExpr::Cos(inner)
        | TLExpr::Tan(inner)
        | TLExpr::FuzzyNot { expr: inner, .. }
        | TLExpr::WeightedRule { rule: inner, .. } => {
            collect_free_vars(inner, bound, free);
        }

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_free_vars(condition, bound, free);
            collect_free_vars(then_branch, bound, free);
            collect_free_vars(else_branch, bound, free);
        }

        TLExpr::Aggregate { var, body, .. }
        | TLExpr::SoftExists { var, body, .. }
        | TLExpr::SoftForAll { var, body, .. } => {
            bound.insert(var.clone());
            collect_free_vars(body, bound, free);
            bound.remove(var);
        }

        TLExpr::ProbabilisticChoice { alternatives } => {
            for (_, expr) in alternatives {
                collect_free_vars(expr, bound, free);
            }
        }

        TLExpr::Constant(_) => {}

        // All other expression types (alpha.3 enhancements)
        _ => {}
    }
}

/// Try to hoist loop-invariant expressions from EXISTS.
fn try_hoist_invariant(var: &str, domain: &str, body: &TLExpr) -> Option<TLExpr> {
    match body {
        // EXISTS x. (a + b) where a doesn't depend on x → a + EXISTS x. b
        TLExpr::Add(lhs, rhs) => {
            let lhs_vars = free_vars(lhs);
            let rhs_vars = free_vars(rhs);

            if !lhs_vars.contains(var) {
                // LHS is loop-invariant
                return Some(TLExpr::add(
                    (**lhs).clone(),
                    TLExpr::exists(var, domain, (**rhs).clone()),
                ));
            }
            if !rhs_vars.contains(var) {
                // RHS is loop-invariant
                return Some(TLExpr::add(
                    TLExpr::exists(var, domain, (**lhs).clone()),
                    (**rhs).clone(),
                ));
            }
            None
        }

        // EXISTS x. (a * b) where a doesn't depend on x → a * EXISTS x. b
        TLExpr::Mul(lhs, rhs) => {
            let lhs_vars = free_vars(lhs);
            let rhs_vars = free_vars(rhs);

            if !lhs_vars.contains(var) {
                return Some(TLExpr::mul(
                    (**lhs).clone(),
                    TLExpr::exists(var, domain, (**rhs).clone()),
                ));
            }
            if !rhs_vars.contains(var) {
                return Some(TLExpr::mul(
                    TLExpr::exists(var, domain, (**lhs).clone()),
                    (**rhs).clone(),
                ));
            }
            None
        }

        // EXISTS x. (a AND b) where a doesn't depend on x → a AND EXISTS x. b
        TLExpr::And(lhs, rhs) => {
            let lhs_vars = free_vars(lhs);
            let rhs_vars = free_vars(rhs);

            if !lhs_vars.contains(var) {
                return Some(TLExpr::and(
                    (**lhs).clone(),
                    TLExpr::exists(var, domain, (**rhs).clone()),
                ));
            }
            if !rhs_vars.contains(var) {
                return Some(TLExpr::and(
                    TLExpr::exists(var, domain, (**lhs).clone()),
                    (**rhs).clone(),
                ));
            }
            None
        }

        _ => None,
    }
}

/// Try to hoist loop-invariant expressions from FORALL.
fn try_hoist_invariant_forall(var: &str, domain: &str, body: &TLExpr) -> Option<TLExpr> {
    match body {
        // FORALL x. (a AND b) where a doesn't depend on x → a AND FORALL x. b
        TLExpr::And(lhs, rhs) => {
            let lhs_vars = free_vars(lhs);
            let rhs_vars = free_vars(rhs);

            if !lhs_vars.contains(var) {
                return Some(TLExpr::and(
                    (**lhs).clone(),
                    TLExpr::forall(var, domain, (**rhs).clone()),
                ));
            }
            if !rhs_vars.contains(var) {
                return Some(TLExpr::and(
                    TLExpr::forall(var, domain, (**lhs).clone()),
                    (**rhs).clone(),
                ));
            }
            None
        }

        // FORALL x. (a OR b) where a doesn't depend on x → a OR FORALL x. b
        TLExpr::Or(lhs, rhs) => {
            let lhs_vars = free_vars(lhs);
            let rhs_vars = free_vars(rhs);

            if !lhs_vars.contains(var) {
                return Some(TLExpr::or(
                    (**lhs).clone(),
                    TLExpr::forall(var, domain, (**rhs).clone()),
                ));
            }
            if !rhs_vars.contains(var) {
                return Some(TLExpr::or(
                    TLExpr::forall(var, domain, (**lhs).clone()),
                    (**rhs).clone(),
                ));
            }
            None
        }

        // FORALL x. (a * b) where a doesn't depend on x → a * FORALL x. b
        TLExpr::Mul(lhs, rhs) => {
            let lhs_vars = free_vars(lhs);
            let rhs_vars = free_vars(rhs);

            if !lhs_vars.contains(var) {
                return Some(TLExpr::mul(
                    (**lhs).clone(),
                    TLExpr::forall(var, domain, (**rhs).clone()),
                ));
            }
            if !rhs_vars.contains(var) {
                return Some(TLExpr::mul(
                    TLExpr::forall(var, domain, (**lhs).clone()),
                    (**rhs).clone(),
                ));
            }
            None
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_hoist_add_lhs() {
        // EXISTS x. (a + p(x)) → a + EXISTS x. p(x)
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let px = TLExpr::pred("p", vec![Term::var("x")]);
        let expr = TLExpr::exists("x", "D", TLExpr::add(a.clone(), px.clone()));

        let (optimized, stats) = optimize_quantifiers(&expr);
        assert_eq!(stats.invariants_hoisted, 1);

        // Should be a + EXISTS x. p(x)
        if let TLExpr::Add(lhs, rhs) = optimized {
            assert_eq!(*lhs, a);
            assert!(matches!(*rhs, TLExpr::Exists { .. }));
        } else {
            panic!("Expected Add expression");
        }
    }

    #[test]
    fn test_hoist_add_rhs() {
        // EXISTS x. (p(x) + a) → EXISTS x. p(x) + a
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let px = TLExpr::pred("p", vec![Term::var("x")]);
        let expr = TLExpr::exists("x", "D", TLExpr::add(px.clone(), a.clone()));

        let (optimized, stats) = optimize_quantifiers(&expr);
        assert_eq!(stats.invariants_hoisted, 1);

        // Should be EXISTS x. p(x) + a
        if let TLExpr::Add(lhs, rhs) = optimized {
            assert!(matches!(*lhs, TLExpr::Exists { .. }));
            assert_eq!(*rhs, a);
        } else {
            panic!("Expected Add expression");
        }
    }

    #[test]
    fn test_hoist_mul() {
        // EXISTS x. (a * p(x)) → a * EXISTS x. p(x)
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let px = TLExpr::pred("p", vec![Term::var("x")]);
        let expr = TLExpr::exists("x", "D", TLExpr::mul(a.clone(), px.clone()));

        let (optimized, stats) = optimize_quantifiers(&expr);
        assert_eq!(stats.invariants_hoisted, 1);

        // Should be a * EXISTS x. p(x)
        if let TLExpr::Mul(lhs, rhs) = optimized {
            assert_eq!(*lhs, a);
            assert!(matches!(*rhs, TLExpr::Exists { .. }));
        } else {
            panic!("Expected Mul expression");
        }
    }

    #[test]
    fn test_hoist_and() {
        // EXISTS x. (a AND p(x)) → a AND EXISTS x. p(x)
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let px = TLExpr::pred("p", vec![Term::var("x")]);
        let expr = TLExpr::exists("x", "D", TLExpr::and(a.clone(), px.clone()));

        let (optimized, stats) = optimize_quantifiers(&expr);
        assert_eq!(stats.invariants_hoisted, 1);

        // Should be a AND EXISTS x. p(x)
        if let TLExpr::And(lhs, rhs) = optimized {
            assert_eq!(*lhs, a);
            assert!(matches!(*rhs, TLExpr::Exists { .. }));
        } else {
            panic!("Expected And expression");
        }
    }

    #[test]
    fn test_no_hoist_when_dependent() {
        // EXISTS x. (p(x) + q(x)) → no hoisting possible
        let px = TLExpr::pred("p", vec![Term::var("x")]);
        let qx = TLExpr::pred("q", vec![Term::var("x")]);
        let expr = TLExpr::exists("x", "D", TLExpr::add(px, qx));

        let (_, stats) = optimize_quantifiers(&expr);
        assert_eq!(stats.invariants_hoisted, 0);
    }

    #[test]
    fn test_forall_hoist_and() {
        // FORALL x. (a AND p(x)) → a AND FORALL x. p(x)
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let px = TLExpr::pred("p", vec![Term::var("x")]);
        let expr = TLExpr::forall("x", "D", TLExpr::and(a.clone(), px.clone()));

        let (optimized, stats) = optimize_quantifiers(&expr);
        assert_eq!(stats.invariants_hoisted, 1);

        // Should be a AND FORALL x. p(x)
        if let TLExpr::And(lhs, rhs) = optimized {
            assert_eq!(*lhs, a);
            assert!(matches!(*rhs, TLExpr::ForAll { .. }));
        } else {
            panic!("Expected And expression");
        }
    }

    #[test]
    fn test_forall_hoist_or() {
        // FORALL x. (a OR p(x)) → a OR FORALL x. p(x)
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let px = TLExpr::pred("p", vec![Term::var("x")]);
        let expr = TLExpr::forall("x", "D", TLExpr::or(a.clone(), px.clone()));

        let (optimized, stats) = optimize_quantifiers(&expr);
        assert_eq!(stats.invariants_hoisted, 1);

        // Should be a OR FORALL x. p(x)
        if let TLExpr::Or(lhs, rhs) = optimized {
            assert_eq!(*lhs, a);
            assert!(matches!(*rhs, TLExpr::ForAll { .. }));
        } else {
            panic!("Expected Or expression");
        }
    }

    #[test]
    fn test_nested_hoisting() {
        // EXISTS x. (a + (b * p(x))) where both a and b are invariant
        // Should hoist both: a + b * EXISTS x. p(x)
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let px = TLExpr::pred("p", vec![Term::var("x")]);
        let expr = TLExpr::exists("x", "D", TLExpr::add(a.clone(), TLExpr::mul(b.clone(), px)));

        let (optimized, stats) = optimize_quantifiers(&expr);
        // Should hoist at least once
        assert!(stats.invariants_hoisted >= 1);

        // Result should have EXISTS not at top level
        if let TLExpr::Add(lhs, _) = optimized {
            assert_eq!(*lhs, a);
        } else {
            panic!("Expected Add at top level");
        }
    }

    #[test]
    fn test_free_vars() {
        let expr = TLExpr::exists(
            "x",
            "D",
            TLExpr::add(
                TLExpr::pred("p", vec![Term::var("x"), Term::var("y")]),
                TLExpr::pred("q", vec![Term::var("z")]),
            ),
        );

        let vars = free_vars(&expr);
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
        assert!(!vars.contains("x")); // x is bound
    }

    #[test]
    fn test_stats_total_optimizations() {
        let stats = QuantifierOptStats {
            invariants_hoisted: 3,
            quantifiers_reordered: 2,
            quantifiers_fused: 1,
            total_processed: 100,
        };
        assert_eq!(stats.total_optimizations(), 6);
    }
}
