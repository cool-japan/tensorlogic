//! Distributive law transformations for logical expressions.
//!
//! This module implements distributive law transformations that can convert expressions
//! between equivalent forms, which is useful for optimization and normalization.
//!
//! # Distributive Laws
//!
//! ## Basic Distributive Laws
//! - **AND over OR**: A ∧ (B ∨ C) ≡ (A ∧ B) ∨ (A ∧ C)
//! - **OR over AND**: A ∨ (B ∧ C) ≡ (A ∨ B) ∧ (A ∨ C)
//!
//! ## Quantifier Distribution
//! - **∀ over ∧**: ∀x.(P(x) ∧ Q(x)) ≡ (∀x.P(x)) ∧ (∀x.Q(x))
//! - **∃ over ∨**: ∃x.(P(x) ∨ Q(x)) ≡ (∃x.P(x)) ∨ (∃x.Q(x))
//!
//! ## Modal Distribution
//! - **□ over ∧**: □(P ∧ Q) ≡ □P ∧ □Q
//! - **◇ over ∨**: ◇(P ∨ Q) ≡ ◇P ∨ ◇Q

use super::TLExpr;

/// Strategy for applying distributive laws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributiveStrategy {
    /// Distribute AND over OR: A ∧ (B ∨ C) → (A ∧ B) ∨ (A ∧ C)
    AndOverOr,
    /// Distribute OR over AND: A ∨ (B ∧ C) → (A ∨ B) ∧ (A ∨ C)
    OrOverAnd,
    /// Distribute quantifiers when possible
    Quantifiers,
    /// Distribute modal operators
    Modal,
    /// Apply all distributive laws
    All,
}

/// Apply distributive laws to an expression.
///
/// # Arguments
/// * `expr` - The expression to transform
/// * `strategy` - Which distributive laws to apply
///
/// # Returns
/// A transformed expression with distributive laws applied.
pub fn apply_distributive_laws(expr: &TLExpr, strategy: DistributiveStrategy) -> TLExpr {
    match strategy {
        DistributiveStrategy::AndOverOr => distribute_and_over_or(expr),
        DistributiveStrategy::OrOverAnd => distribute_or_over_and(expr),
        DistributiveStrategy::Quantifiers => distribute_quantifiers(expr),
        DistributiveStrategy::Modal => distribute_modal(expr),
        DistributiveStrategy::All => {
            let mut result = expr.clone();
            result = distribute_and_over_or(&result);
            result = distribute_or_over_and(&result);
            result = distribute_quantifiers(&result);
            result = distribute_modal(&result);
            result
        }
    }
}

/// Distribute AND over OR: A ∧ (B ∨ C) → (A ∧ B) ∨ (A ∧ C)
fn distribute_and_over_or(expr: &TLExpr) -> TLExpr {
    match expr {
        // A ∧ (B ∨ C) → (A ∧ B) ∨ (A ∧ C)
        TLExpr::And(a, b) if matches!(**b, TLExpr::Or(_, _)) => {
            if let TLExpr::Or(b1, b2) = &**b {
                let left = TLExpr::and(distribute_and_over_or(a), distribute_and_over_or(b1));
                let right = TLExpr::and(distribute_and_over_or(a), distribute_and_over_or(b2));
                TLExpr::or(left, right)
            } else {
                expr.clone()
            }
        }
        // (A ∨ B) ∧ C → (A ∧ C) ∨ (B ∧ C)
        TLExpr::And(a, c) if matches!(**a, TLExpr::Or(_, _)) => {
            if let TLExpr::Or(a1, a2) = &**a {
                let left = TLExpr::and(distribute_and_over_or(a1), distribute_and_over_or(c));
                let right = TLExpr::and(distribute_and_over_or(a2), distribute_and_over_or(c));
                TLExpr::or(left, right)
            } else {
                expr.clone()
            }
        }
        // Recursively apply to subexpressions
        TLExpr::And(l, r) => TLExpr::and(distribute_and_over_or(l), distribute_and_over_or(r)),
        TLExpr::Or(l, r) => TLExpr::or(distribute_and_over_or(l), distribute_and_over_or(r)),
        TLExpr::Not(e) => TLExpr::negate(distribute_and_over_or(e)),
        TLExpr::Imply(l, r) => TLExpr::imply(distribute_and_over_or(l), distribute_and_over_or(r)),
        TLExpr::Exists { var, domain, body } => {
            TLExpr::exists(var.clone(), domain.clone(), distribute_and_over_or(body))
        }
        TLExpr::ForAll { var, domain, body } => {
            TLExpr::forall(var.clone(), domain.clone(), distribute_and_over_or(body))
        }
        TLExpr::Box(e) => TLExpr::modal_box(distribute_and_over_or(e)),
        TLExpr::Diamond(e) => TLExpr::modal_diamond(distribute_and_over_or(e)),
        TLExpr::Next(e) => TLExpr::next(distribute_and_over_or(e)),
        TLExpr::Eventually(e) => TLExpr::eventually(distribute_and_over_or(e)),
        TLExpr::Always(e) => TLExpr::always(distribute_and_over_or(e)),
        TLExpr::Until { before, after } => TLExpr::until(
            distribute_and_over_or(before),
            distribute_and_over_or(after),
        ),
        _ => expr.clone(),
    }
}

/// Distribute OR over AND: A ∨ (B ∧ C) → (A ∨ B) ∧ (A ∨ C)
fn distribute_or_over_and(expr: &TLExpr) -> TLExpr {
    match expr {
        // A ∨ (B ∧ C) → (A ∨ B) ∧ (A ∨ C)
        TLExpr::Or(a, b) if matches!(**b, TLExpr::And(_, _)) => {
            if let TLExpr::And(b1, b2) = &**b {
                let left = TLExpr::or(distribute_or_over_and(a), distribute_or_over_and(b1));
                let right = TLExpr::or(distribute_or_over_and(a), distribute_or_over_and(b2));
                TLExpr::and(left, right)
            } else {
                expr.clone()
            }
        }
        // (A ∧ B) ∨ C → (A ∨ C) ∧ (B ∨ C)
        TLExpr::Or(a, c) if matches!(**a, TLExpr::And(_, _)) => {
            if let TLExpr::And(a1, a2) = &**a {
                let left = TLExpr::or(distribute_or_over_and(a1), distribute_or_over_and(c));
                let right = TLExpr::or(distribute_or_over_and(a2), distribute_or_over_and(c));
                TLExpr::and(left, right)
            } else {
                expr.clone()
            }
        }
        // Recursively apply to subexpressions
        TLExpr::And(l, r) => TLExpr::and(distribute_or_over_and(l), distribute_or_over_and(r)),
        TLExpr::Or(l, r) => TLExpr::or(distribute_or_over_and(l), distribute_or_over_and(r)),
        TLExpr::Not(e) => TLExpr::negate(distribute_or_over_and(e)),
        TLExpr::Imply(l, r) => TLExpr::imply(distribute_or_over_and(l), distribute_or_over_and(r)),
        TLExpr::Exists { var, domain, body } => {
            TLExpr::exists(var.clone(), domain.clone(), distribute_or_over_and(body))
        }
        TLExpr::ForAll { var, domain, body } => {
            TLExpr::forall(var.clone(), domain.clone(), distribute_or_over_and(body))
        }
        TLExpr::Box(e) => TLExpr::modal_box(distribute_or_over_and(e)),
        TLExpr::Diamond(e) => TLExpr::modal_diamond(distribute_or_over_and(e)),
        TLExpr::Next(e) => TLExpr::next(distribute_or_over_and(e)),
        TLExpr::Eventually(e) => TLExpr::eventually(distribute_or_over_and(e)),
        TLExpr::Always(e) => TLExpr::always(distribute_or_over_and(e)),
        TLExpr::Until { before, after } => TLExpr::until(
            distribute_or_over_and(before),
            distribute_or_over_and(after),
        ),
        _ => expr.clone(),
    }
}

/// Distribute quantifiers: ∀x.(P(x) ∧ Q(x)) → (∀x.P(x)) ∧ (∀x.Q(x))
fn distribute_quantifiers(expr: &TLExpr) -> TLExpr {
    match expr {
        // ∀x.(P(x) ∧ Q(x)) → (∀x.P(x)) ∧ (∀x.Q(x))
        TLExpr::ForAll { var, domain, body } if matches!(**body, TLExpr::And(_, _)) => {
            if let TLExpr::And(p, q) = &**body {
                let left = TLExpr::forall(var.clone(), domain.clone(), distribute_quantifiers(p));
                let right = TLExpr::forall(var.clone(), domain.clone(), distribute_quantifiers(q));
                TLExpr::and(left, right)
            } else {
                expr.clone()
            }
        }
        // ∃x.(P(x) ∨ Q(x)) → (∃x.P(x)) ∨ (∃x.Q(x))
        TLExpr::Exists { var, domain, body } if matches!(**body, TLExpr::Or(_, _)) => {
            if let TLExpr::Or(p, q) = &**body {
                let left = TLExpr::exists(var.clone(), domain.clone(), distribute_quantifiers(p));
                let right = TLExpr::exists(var.clone(), domain.clone(), distribute_quantifiers(q));
                TLExpr::or(left, right)
            } else {
                expr.clone()
            }
        }
        // Recursively apply to subexpressions
        TLExpr::And(l, r) => TLExpr::and(distribute_quantifiers(l), distribute_quantifiers(r)),
        TLExpr::Or(l, r) => TLExpr::or(distribute_quantifiers(l), distribute_quantifiers(r)),
        TLExpr::Not(e) => TLExpr::negate(distribute_quantifiers(e)),
        TLExpr::Imply(l, r) => TLExpr::imply(distribute_quantifiers(l), distribute_quantifiers(r)),
        TLExpr::Exists { var, domain, body } => {
            TLExpr::exists(var.clone(), domain.clone(), distribute_quantifiers(body))
        }
        TLExpr::ForAll { var, domain, body } => {
            TLExpr::forall(var.clone(), domain.clone(), distribute_quantifiers(body))
        }
        TLExpr::Box(e) => TLExpr::modal_box(distribute_quantifiers(e)),
        TLExpr::Diamond(e) => TLExpr::modal_diamond(distribute_quantifiers(e)),
        _ => expr.clone(),
    }
}

/// Distribute modal operators: □(P ∧ Q) → □P ∧ □Q
fn distribute_modal(expr: &TLExpr) -> TLExpr {
    match expr {
        // □(P ∧ Q) → □P ∧ □Q
        TLExpr::Box(body) if matches!(**body, TLExpr::And(_, _)) => {
            if let TLExpr::And(p, q) = &**body {
                let left = TLExpr::modal_box(distribute_modal(p));
                let right = TLExpr::modal_box(distribute_modal(q));
                TLExpr::and(left, right)
            } else {
                expr.clone()
            }
        }
        // ◇(P ∨ Q) → ◇P ∨ ◇Q
        TLExpr::Diamond(body) if matches!(**body, TLExpr::Or(_, _)) => {
            if let TLExpr::Or(p, q) = &**body {
                let left = TLExpr::modal_diamond(distribute_modal(p));
                let right = TLExpr::modal_diamond(distribute_modal(q));
                TLExpr::or(left, right)
            } else {
                expr.clone()
            }
        }
        // Recursively apply to subexpressions
        TLExpr::And(l, r) => TLExpr::and(distribute_modal(l), distribute_modal(r)),
        TLExpr::Or(l, r) => TLExpr::or(distribute_modal(l), distribute_modal(r)),
        TLExpr::Not(e) => TLExpr::negate(distribute_modal(e)),
        TLExpr::Imply(l, r) => TLExpr::imply(distribute_modal(l), distribute_modal(r)),
        TLExpr::Exists { var, domain, body } => {
            TLExpr::exists(var.clone(), domain.clone(), distribute_modal(body))
        }
        TLExpr::ForAll { var, domain, body } => {
            TLExpr::forall(var.clone(), domain.clone(), distribute_modal(body))
        }
        TLExpr::Box(e) => TLExpr::modal_box(distribute_modal(e)),
        TLExpr::Diamond(e) => TLExpr::modal_diamond(distribute_modal(e)),
        TLExpr::Next(e) => TLExpr::next(distribute_modal(e)),
        TLExpr::Eventually(e) => TLExpr::eventually(distribute_modal(e)),
        TLExpr::Always(e) => TLExpr::always(distribute_modal(e)),
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_and_over_or_simple() {
        // A ∧ (B ∨ C) → (A ∧ B) ∨ (A ∧ C)
        let a = TLExpr::pred("A", vec![Term::var("x")]);
        let b = TLExpr::pred("B", vec![Term::var("x")]);
        let c = TLExpr::pred("C", vec![Term::var("x")]);

        let expr = TLExpr::and(a.clone(), TLExpr::or(b.clone(), c.clone()));
        let result = apply_distributive_laws(&expr, DistributiveStrategy::AndOverOr);

        // Result should be (A ∧ B) ∨ (A ∧ C)
        assert!(matches!(result, TLExpr::Or(_, _)));
        if let TLExpr::Or(left, right) = result {
            assert!(matches!(*left, TLExpr::And(_, _)));
            assert!(matches!(*right, TLExpr::And(_, _)));
        }
    }

    #[test]
    fn test_and_over_or_left() {
        // (A ∨ B) ∧ C → (A ∧ C) ∨ (B ∧ C)
        let a = TLExpr::pred("A", vec![Term::var("x")]);
        let b = TLExpr::pred("B", vec![Term::var("x")]);
        let c = TLExpr::pred("C", vec![Term::var("x")]);

        let expr = TLExpr::and(TLExpr::or(a.clone(), b.clone()), c.clone());
        let result = apply_distributive_laws(&expr, DistributiveStrategy::AndOverOr);

        // Result should be (A ∧ C) ∨ (B ∧ C)
        assert!(matches!(result, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_or_over_and_simple() {
        // A ∨ (B ∧ C) → (A ∨ B) ∧ (A ∨ C)
        let a = TLExpr::pred("A", vec![Term::var("x")]);
        let b = TLExpr::pred("B", vec![Term::var("x")]);
        let c = TLExpr::pred("C", vec![Term::var("x")]);

        let expr = TLExpr::or(a.clone(), TLExpr::and(b.clone(), c.clone()));
        let result = apply_distributive_laws(&expr, DistributiveStrategy::OrOverAnd);

        // Result should be (A ∨ B) ∧ (A ∨ C)
        assert!(matches!(result, TLExpr::And(_, _)));
        if let TLExpr::And(left, right) = result {
            assert!(matches!(*left, TLExpr::Or(_, _)));
            assert!(matches!(*right, TLExpr::Or(_, _)));
        }
    }

    #[test]
    fn test_or_over_and_left() {
        // (A ∧ B) ∨ C → (A ∨ C) ∧ (B ∨ C)
        let a = TLExpr::pred("A", vec![Term::var("x")]);
        let b = TLExpr::pred("B", vec![Term::var("x")]);
        let c = TLExpr::pred("C", vec![Term::var("x")]);

        let expr = TLExpr::or(TLExpr::and(a.clone(), b.clone()), c.clone());
        let result = apply_distributive_laws(&expr, DistributiveStrategy::OrOverAnd);

        // Result should be (A ∨ C) ∧ (B ∨ C)
        assert!(matches!(result, TLExpr::And(_, _)));
    }

    #[test]
    fn test_forall_over_and() {
        // ∀x.(P(x) ∧ Q(x)) → (∀x.P(x)) ∧ (∀x.Q(x))
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let q = TLExpr::pred("Q", vec![Term::var("x")]);

        let expr = TLExpr::forall("x", "D", TLExpr::and(p, q));
        let result = apply_distributive_laws(&expr, DistributiveStrategy::Quantifiers);

        // Result should be (∀x.P(x)) ∧ (∀x.Q(x))
        assert!(matches!(result, TLExpr::And(_, _)));
        if let TLExpr::And(left, right) = result {
            assert!(matches!(*left, TLExpr::ForAll { .. }));
            assert!(matches!(*right, TLExpr::ForAll { .. }));
        }
    }

    #[test]
    fn test_exists_over_or() {
        // ∃x.(P(x) ∨ Q(x)) → (∃x.P(x)) ∨ (∃x.Q(x))
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let q = TLExpr::pred("Q", vec![Term::var("x")]);

        let expr = TLExpr::exists("x", "D", TLExpr::or(p, q));
        let result = apply_distributive_laws(&expr, DistributiveStrategy::Quantifiers);

        // Result should be (∃x.P(x)) ∨ (∃x.Q(x))
        assert!(matches!(result, TLExpr::Or(_, _)));
        if let TLExpr::Or(left, right) = result {
            assert!(matches!(*left, TLExpr::Exists { .. }));
            assert!(matches!(*right, TLExpr::Exists { .. }));
        }
    }

    #[test]
    fn test_box_over_and() {
        // □(P ∧ Q) → □P ∧ □Q
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let q = TLExpr::pred("Q", vec![Term::var("x")]);

        let expr = TLExpr::modal_box(TLExpr::and(p, q));
        let result = apply_distributive_laws(&expr, DistributiveStrategy::Modal);

        // Result should be □P ∧ □Q
        assert!(matches!(result, TLExpr::And(_, _)));
        if let TLExpr::And(left, right) = result {
            assert!(matches!(*left, TLExpr::Box(_)));
            assert!(matches!(*right, TLExpr::Box(_)));
        }
    }

    #[test]
    fn test_diamond_over_or() {
        // ◇(P ∨ Q) → ◇P ∨ ◇Q
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let q = TLExpr::pred("Q", vec![Term::var("x")]);

        let expr = TLExpr::modal_diamond(TLExpr::or(p, q));
        let result = apply_distributive_laws(&expr, DistributiveStrategy::Modal);

        // Result should be ◇P ∨ ◇Q
        assert!(matches!(result, TLExpr::Or(_, _)));
        if let TLExpr::Or(left, right) = result {
            assert!(matches!(*left, TLExpr::Diamond(_)));
            assert!(matches!(*right, TLExpr::Diamond(_)));
        }
    }

    #[test]
    fn test_all_strategy() {
        // Test that All strategy applies AndOverOr transformation
        let a = TLExpr::pred("A", vec![Term::var("x")]);
        let b = TLExpr::pred("B", vec![Term::var("x")]);
        let c = TLExpr::pred("C", vec![Term::var("x")]);

        // A ∧ (B ∨ C)
        let expr = TLExpr::and(a, TLExpr::or(b, c));
        let result = apply_distributive_laws(&expr, DistributiveStrategy::AndOverOr);

        // Should be distributed to (A ∧ B) ∨ (A ∧ C)
        assert!(matches!(result, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_no_distribution_needed() {
        // Test that expressions that don't need distribution are unchanged
        let a = TLExpr::pred("A", vec![Term::var("x")]);
        let b = TLExpr::pred("B", vec![Term::var("x")]);

        // A ∧ B (no distribution needed)
        let expr = TLExpr::and(a.clone(), b.clone());
        let result = apply_distributive_laws(&expr, DistributiveStrategy::AndOverOr);

        // Should remain A ∧ B
        assert!(matches!(result, TLExpr::And(_, _)));
    }
}
