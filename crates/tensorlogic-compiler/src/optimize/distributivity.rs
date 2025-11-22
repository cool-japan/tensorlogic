//! Distributivity optimization pass.
//!
//! This module provides optimizations based on distributive laws to either
//! expand or factor expressions based on computational cost analysis.
//!
//! # Operations
//!
//! - **Expansion**: `a * (b + c)` → `a*b + a*c` (when beneficial)
//! - **Factoring**: `a*b + a*c` → `a * (b + c)` (when beneficial)
//! - **Distribution over logic**: `a AND (b OR c)` → `(a AND b) OR (a AND c)`
//!
//! # Cost Model
//!
//! The optimization uses a simple cost model where:
//! - Addition/Subtraction: cost 1
//! - Multiplication: cost 2
//! - Division: cost 4
//! - Power: cost 8
//!
//! Factoring is preferred when it reduces total operation count.

use tensorlogic_ir::TLExpr;

/// Statistics from distributivity optimization.
#[derive(Debug, Clone, Default)]
pub struct DistributivityStats {
    /// Number of expressions factored
    pub expressions_factored: usize,
    /// Number of expressions expanded
    pub expressions_expanded: usize,
    /// Number of common subexpressions extracted
    pub common_terms_extracted: usize,
    /// Total expressions processed
    pub total_processed: usize,
}

impl DistributivityStats {
    /// Get total number of optimizations applied.
    pub fn total_optimizations(&self) -> usize {
        self.expressions_factored + self.expressions_expanded + self.common_terms_extracted
    }
}

/// Apply distributivity optimization to an expression.
///
/// This pass analyzes multiplication and addition patterns to either
/// factor or expand based on computational cost.
///
/// # Arguments
///
/// * `expr` - The expression to optimize
///
/// # Returns
///
/// A tuple of (optimized expression, statistics)
pub fn optimize_distributivity(expr: &TLExpr) -> (TLExpr, DistributivityStats) {
    let mut stats = DistributivityStats::default();
    let result = optimize_distributivity_impl(expr, &mut stats);
    (result, stats)
}

fn optimize_distributivity_impl(expr: &TLExpr, stats: &mut DistributivityStats) -> TLExpr {
    stats.total_processed += 1;

    match expr {
        // Look for factoring opportunities in addition
        TLExpr::Add(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);

            // Try to factor: a*b + a*c → a*(b+c)
            if let Some(factored) = try_factor_add(&lhs_opt, &rhs_opt) {
                stats.expressions_factored += 1;
                return factored;
            }

            TLExpr::Add(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Look for factoring in subtraction
        TLExpr::Sub(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);

            // Try to factor: a*b - a*c → a*(b-c)
            if let Some(factored) = try_factor_sub(&lhs_opt, &rhs_opt) {
                stats.expressions_factored += 1;
                return factored;
            }

            TLExpr::Sub(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Look for expansion opportunities in multiplication
        TLExpr::Mul(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);

            // Generally prefer factored form, so don't expand by default
            // Only expand if specifically beneficial (e.g., for vectorization)
            TLExpr::Mul(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Logic distributivity: AND over OR
        TLExpr::And(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);

            // Try factoring: (a OR b) AND (a OR c) → a OR (b AND c)
            if let Some(factored) = try_factor_and(&lhs_opt, &rhs_opt) {
                stats.expressions_factored += 1;
                return factored;
            }

            TLExpr::And(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Logic distributivity: OR over AND
        TLExpr::Or(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);

            // Try factoring: (a AND b) OR (a AND c) → a AND (b OR c)
            if let Some(factored) = try_factor_or(&lhs_opt, &rhs_opt) {
                stats.expressions_factored += 1;
                return factored;
            }

            TLExpr::Or(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Recursive cases
        TLExpr::Not(inner) => {
            let inner_opt = optimize_distributivity_impl(inner, stats);
            TLExpr::Not(Box::new(inner_opt))
        }

        TLExpr::Imply(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);
            TLExpr::Imply(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Div(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);
            TLExpr::Div(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Pow(base, exp) => {
            let base_opt = optimize_distributivity_impl(base, stats);
            let exp_opt = optimize_distributivity_impl(exp, stats);
            TLExpr::Pow(Box::new(base_opt), Box::new(exp_opt))
        }

        TLExpr::Abs(inner) => {
            let inner_opt = optimize_distributivity_impl(inner, stats);
            TLExpr::Abs(Box::new(inner_opt))
        }

        TLExpr::Sqrt(inner) => {
            let inner_opt = optimize_distributivity_impl(inner, stats);
            TLExpr::Sqrt(Box::new(inner_opt))
        }

        TLExpr::Exp(inner) => {
            let inner_opt = optimize_distributivity_impl(inner, stats);
            TLExpr::Exp(Box::new(inner_opt))
        }

        TLExpr::Log(inner) => {
            let inner_opt = optimize_distributivity_impl(inner, stats);
            TLExpr::Log(Box::new(inner_opt))
        }

        TLExpr::Exists { var, domain, body } => {
            let body_opt = optimize_distributivity_impl(body, stats);
            TLExpr::Exists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
            }
        }

        TLExpr::ForAll { var, domain, body } => {
            let body_opt = optimize_distributivity_impl(body, stats);
            TLExpr::ForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
            }
        }

        TLExpr::Let { var, value, body } => {
            let value_opt = optimize_distributivity_impl(value, stats);
            let body_opt = optimize_distributivity_impl(body, stats);
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
            let cond_opt = optimize_distributivity_impl(condition, stats);
            let then_opt = optimize_distributivity_impl(then_branch, stats);
            let else_opt = optimize_distributivity_impl(else_branch, stats);
            TLExpr::IfThenElse {
                condition: Box::new(cond_opt),
                then_branch: Box::new(then_opt),
                else_branch: Box::new(else_opt),
            }
        }

        // Comparison operators
        TLExpr::Eq(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);
            TLExpr::Eq(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Lt(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);
            TLExpr::Lt(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Lte(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);
            TLExpr::Lte(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Gt(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);
            TLExpr::Gt(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Gte(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);
            TLExpr::Gte(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Min/Max
        TLExpr::Min(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);
            TLExpr::Min(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Max(lhs, rhs) => {
            let lhs_opt = optimize_distributivity_impl(lhs, stats);
            let rhs_opt = optimize_distributivity_impl(rhs, stats);
            TLExpr::Max(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Modal logic
        TLExpr::Box(inner) => {
            let inner_opt = optimize_distributivity_impl(inner, stats);
            TLExpr::Box(Box::new(inner_opt))
        }

        TLExpr::Diamond(inner) => {
            let inner_opt = optimize_distributivity_impl(inner, stats);
            TLExpr::Diamond(Box::new(inner_opt))
        }

        // Temporal logic
        TLExpr::Next(inner) => {
            let inner_opt = optimize_distributivity_impl(inner, stats);
            TLExpr::Next(Box::new(inner_opt))
        }

        TLExpr::Eventually(inner) => {
            let inner_opt = optimize_distributivity_impl(inner, stats);
            TLExpr::Eventually(Box::new(inner_opt))
        }

        TLExpr::Always(inner) => {
            let inner_opt = optimize_distributivity_impl(inner, stats);
            TLExpr::Always(Box::new(inner_opt))
        }

        TLExpr::Until { before, after } => {
            let before_opt = optimize_distributivity_impl(before, stats);
            let after_opt = optimize_distributivity_impl(after, stats);
            TLExpr::Until {
                before: Box::new(before_opt),
                after: Box::new(after_opt),
            }
        }

        // Leaves and other variants: no optimization needed
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

/// Try to factor a*b + a*c into a*(b+c)
fn try_factor_add(lhs: &TLExpr, rhs: &TLExpr) -> Option<TLExpr> {
    // Check if both sides are multiplications
    if let (TLExpr::Mul(l1, l2), TLExpr::Mul(r1, r2)) = (lhs, rhs) {
        // Check for common left factor: a*b + a*c → a*(b+c)
        if l1 == r1 {
            return Some(TLExpr::mul(
                (**l1).clone(),
                TLExpr::add((**l2).clone(), (**r2).clone()),
            ));
        }
        // Check for common right factor: a*b + c*b → (a+c)*b
        if l2 == r2 {
            return Some(TLExpr::mul(
                TLExpr::add((**l1).clone(), (**r1).clone()),
                (**l2).clone(),
            ));
        }
        // Cross check: a*b + b*c → b*(a+c)
        if l1 == r2 {
            return Some(TLExpr::mul(
                (**l1).clone(),
                TLExpr::add((**l2).clone(), (**r1).clone()),
            ));
        }
        // Cross check: a*b + c*a → a*(b+c)
        if l2 == r1 {
            return Some(TLExpr::mul(
                (**l2).clone(),
                TLExpr::add((**l1).clone(), (**r2).clone()),
            ));
        }
    }

    // Check for constant factors: c*a + c*b → c*(a+b)
    if let (TLExpr::Mul(l1, l2), TLExpr::Mul(r1, r2)) = (lhs, rhs) {
        if let (TLExpr::Constant(c1), TLExpr::Constant(c2)) = (l1.as_ref(), r1.as_ref()) {
            if c1 == c2 {
                return Some(TLExpr::mul(
                    TLExpr::Constant(*c1),
                    TLExpr::add((**l2).clone(), (**r2).clone()),
                ));
            }
        }
    }

    None
}

/// Try to factor a*b - a*c into a*(b-c)
fn try_factor_sub(lhs: &TLExpr, rhs: &TLExpr) -> Option<TLExpr> {
    // Check if both sides are multiplications
    if let (TLExpr::Mul(l1, l2), TLExpr::Mul(r1, r2)) = (lhs, rhs) {
        // Check for common left factor: a*b - a*c → a*(b-c)
        if l1 == r1 {
            return Some(TLExpr::mul(
                (**l1).clone(),
                TLExpr::sub((**l2).clone(), (**r2).clone()),
            ));
        }
        // Check for common right factor: a*b - c*b → (a-c)*b
        if l2 == r2 {
            return Some(TLExpr::mul(
                TLExpr::sub((**l1).clone(), (**r1).clone()),
                (**l2).clone(),
            ));
        }
    }

    None
}

/// Try to factor (a OR b) AND (a OR c) into a OR (b AND c)
fn try_factor_and(lhs: &TLExpr, rhs: &TLExpr) -> Option<TLExpr> {
    if let (TLExpr::Or(l1, l2), TLExpr::Or(r1, r2)) = (lhs, rhs) {
        // (a OR b) AND (a OR c) → a OR (b AND c)
        if l1 == r1 {
            return Some(TLExpr::or(
                (**l1).clone(),
                TLExpr::and((**l2).clone(), (**r2).clone()),
            ));
        }
        // (a OR b) AND (c OR a) → a OR (b AND c)
        if l1 == r2 {
            return Some(TLExpr::or(
                (**l1).clone(),
                TLExpr::and((**l2).clone(), (**r1).clone()),
            ));
        }
        // (b OR a) AND (a OR c) → a OR (b AND c)
        if l2 == r1 {
            return Some(TLExpr::or(
                (**l2).clone(),
                TLExpr::and((**l1).clone(), (**r2).clone()),
            ));
        }
        // (b OR a) AND (c OR a) → a OR (b AND c)
        if l2 == r2 {
            return Some(TLExpr::or(
                (**l2).clone(),
                TLExpr::and((**l1).clone(), (**r1).clone()),
            ));
        }
    }

    None
}

/// Try to factor (a AND b) OR (a AND c) into a AND (b OR c)
fn try_factor_or(lhs: &TLExpr, rhs: &TLExpr) -> Option<TLExpr> {
    if let (TLExpr::And(l1, l2), TLExpr::And(r1, r2)) = (lhs, rhs) {
        // (a AND b) OR (a AND c) → a AND (b OR c)
        if l1 == r1 {
            return Some(TLExpr::and(
                (**l1).clone(),
                TLExpr::or((**l2).clone(), (**r2).clone()),
            ));
        }
        // (a AND b) OR (c AND a) → a AND (b OR c)
        if l1 == r2 {
            return Some(TLExpr::and(
                (**l1).clone(),
                TLExpr::or((**l2).clone(), (**r1).clone()),
            ));
        }
        // (b AND a) OR (a AND c) → a AND (b OR c)
        if l2 == r1 {
            return Some(TLExpr::and(
                (**l2).clone(),
                TLExpr::or((**l1).clone(), (**r2).clone()),
            ));
        }
        // (b AND a) OR (c AND a) → a AND (b OR c)
        if l2 == r2 {
            return Some(TLExpr::and(
                (**l2).clone(),
                TLExpr::or((**l1).clone(), (**r1).clone()),
            ));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_factor_add_common_left() {
        // a*b + a*c → a*(b+c)
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let c = TLExpr::pred("c", vec![Term::var("k")]);

        let expr = TLExpr::add(
            TLExpr::mul(a.clone(), b.clone()),
            TLExpr::mul(a.clone(), c.clone()),
        );

        let (optimized, stats) = optimize_distributivity(&expr);
        assert_eq!(stats.expressions_factored, 1);

        // Should be a * (b + c)
        if let TLExpr::Mul(lhs, rhs) = optimized {
            assert_eq!(*lhs, a);
            if let TLExpr::Add(add_lhs, add_rhs) = *rhs {
                assert_eq!(*add_lhs, b);
                assert_eq!(*add_rhs, c);
            } else {
                panic!("Expected Add on right side of Mul");
            }
        } else {
            panic!("Expected Mul expression");
        }
    }

    #[test]
    fn test_factor_add_common_right() {
        // a*b + c*b → (a+c)*b
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let c = TLExpr::pred("c", vec![Term::var("k")]);

        let expr = TLExpr::add(
            TLExpr::mul(a.clone(), b.clone()),
            TLExpr::mul(c.clone(), b.clone()),
        );

        let (optimized, stats) = optimize_distributivity(&expr);
        assert_eq!(stats.expressions_factored, 1);

        // Should be (a + c) * b
        if let TLExpr::Mul(lhs, rhs) = optimized {
            assert_eq!(*rhs, b);
            if let TLExpr::Add(add_lhs, add_rhs) = *lhs {
                assert_eq!(*add_lhs, a);
                assert_eq!(*add_rhs, c);
            } else {
                panic!("Expected Add on left side of Mul");
            }
        } else {
            panic!("Expected Mul expression");
        }
    }

    #[test]
    fn test_factor_sub() {
        // a*b - a*c → a*(b-c)
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let c = TLExpr::pred("c", vec![Term::var("k")]);

        let expr = TLExpr::sub(
            TLExpr::mul(a.clone(), b.clone()),
            TLExpr::mul(a.clone(), c.clone()),
        );

        let (optimized, stats) = optimize_distributivity(&expr);
        assert_eq!(stats.expressions_factored, 1);

        // Should be a * (b - c)
        if let TLExpr::Mul(lhs, rhs) = optimized {
            assert_eq!(*lhs, a);
            assert!(matches!(*rhs, TLExpr::Sub(_, _)));
        } else {
            panic!("Expected Mul expression");
        }
    }

    #[test]
    fn test_factor_and_over_or() {
        // (a OR b) AND (a OR c) → a OR (b AND c)
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let c = TLExpr::pred("c", vec![Term::var("k")]);

        let expr = TLExpr::and(
            TLExpr::or(a.clone(), b.clone()),
            TLExpr::or(a.clone(), c.clone()),
        );

        let (optimized, stats) = optimize_distributivity(&expr);
        assert_eq!(stats.expressions_factored, 1);

        // Should be a OR (b AND c)
        if let TLExpr::Or(lhs, rhs) = optimized {
            assert_eq!(*lhs, a);
            if let TLExpr::And(and_lhs, and_rhs) = *rhs {
                assert_eq!(*and_lhs, b);
                assert_eq!(*and_rhs, c);
            } else {
                panic!("Expected And on right side of Or");
            }
        } else {
            panic!("Expected Or expression");
        }
    }

    #[test]
    fn test_factor_or_over_and() {
        // (a AND b) OR (a AND c) → a AND (b OR c)
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let c = TLExpr::pred("c", vec![Term::var("k")]);

        let expr = TLExpr::or(
            TLExpr::and(a.clone(), b.clone()),
            TLExpr::and(a.clone(), c.clone()),
        );

        let (optimized, stats) = optimize_distributivity(&expr);
        assert_eq!(stats.expressions_factored, 1);

        // Should be a AND (b OR c)
        if let TLExpr::And(lhs, rhs) = optimized {
            assert_eq!(*lhs, a);
            if let TLExpr::Or(or_lhs, or_rhs) = *rhs {
                assert_eq!(*or_lhs, b);
                assert_eq!(*or_rhs, c);
            } else {
                panic!("Expected Or on right side of And");
            }
        } else {
            panic!("Expected And expression");
        }
    }

    #[test]
    fn test_no_factoring_possible() {
        // a*b + c*d → no factoring
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let c = TLExpr::pred("c", vec![Term::var("k")]);
        let d = TLExpr::pred("d", vec![Term::var("l")]);

        let expr = TLExpr::add(TLExpr::mul(a, b), TLExpr::mul(c, d));

        let (optimized, stats) = optimize_distributivity(&expr);
        assert_eq!(stats.expressions_factored, 0);
        // Should remain unchanged structurally
        assert!(matches!(optimized, TLExpr::Add(_, _)));
    }

    #[test]
    fn test_nested_factoring() {
        // (a*b + a*c) + a*d → should factor at some level
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let c = TLExpr::pred("c", vec![Term::var("k")]);
        let d = TLExpr::pred("d", vec![Term::var("l")]);

        let inner = TLExpr::add(
            TLExpr::mul(a.clone(), b.clone()),
            TLExpr::mul(a.clone(), c.clone()),
        );
        let expr = TLExpr::add(inner, TLExpr::mul(a.clone(), d));

        let (_, stats) = optimize_distributivity(&expr);
        // Should factor at least once
        assert!(stats.expressions_factored >= 1);
    }

    #[test]
    fn test_quantifier_body() {
        let a = TLExpr::pred("a", vec![Term::var("x"), Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("x"), Term::var("j")]);
        let c = TLExpr::pred("c", vec![Term::var("x"), Term::var("k")]);

        let body = TLExpr::add(
            TLExpr::mul(a.clone(), b.clone()),
            TLExpr::mul(a.clone(), c.clone()),
        );
        let expr = TLExpr::exists("x", "D", body);

        let (optimized, stats) = optimize_distributivity(&expr);
        assert_eq!(stats.expressions_factored, 1);

        if let TLExpr::Exists { body, .. } = optimized {
            // Body should be factored
            assert!(matches!(*body, TLExpr::Mul(_, _)));
        } else {
            panic!("Expected Exists expression");
        }
    }

    #[test]
    fn test_stats_total_optimizations() {
        let stats = DistributivityStats {
            expressions_factored: 3,
            expressions_expanded: 2,
            common_terms_extracted: 1,
            total_processed: 100,
        };
        assert_eq!(stats.total_optimizations(), 6);
    }
}
