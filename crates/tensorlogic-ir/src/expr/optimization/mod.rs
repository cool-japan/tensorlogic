//! Expression-level optimizations.
//!
//! This module provides various optimization passes for `TLExpr`:
//! - **Constant folding**: Evaluate constant expressions at compile time
//! - **Algebraic simplification**: Apply algebraic identities (e.g., x + 0 = x)
//! - **Constant propagation**: Substitute variables bound in Let expressions
//!
//! The main entry point is [`optimize_expr`], which applies multiple passes
//! iteratively until a fixed point is reached.

mod algebraic;
mod constant_folding;
mod propagation;
pub(crate) mod substitution;

// Re-export public functions
pub use algebraic::algebraic_simplify;
pub use constant_folding::constant_fold;
pub use propagation::propagate_constants;

use crate::expr::TLExpr;

/// Apply multiple optimization passes in sequence
///
/// This function applies constant propagation, constant folding, and algebraic
/// simplification iteratively until no more changes occur or a maximum number
/// of iterations is reached.
///
/// # Example
///
/// ```
/// use tensorlogic_ir::TLExpr;
/// use tensorlogic_ir::optimize_expr;
///
/// // (2 + 3) * 1 should become 5
/// let expr = TLExpr::mul(
///     TLExpr::add(TLExpr::constant(2.0), TLExpr::constant(3.0)),
///     TLExpr::constant(1.0),
/// );
/// let optimized = optimize_expr(&expr);
/// assert_eq!(optimized, TLExpr::Constant(5.0));
/// ```
pub fn optimize_expr(expr: &TLExpr) -> TLExpr {
    // Apply optimizations iteratively until no more changes occur
    // This handles nested Let bindings and cascading optimizations
    let mut current = expr.clone();
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 10; // Prevent infinite loops

    loop {
        let propagated = propagate_constants(&current);
        let folded = constant_fold(&propagated);
        let simplified = algebraic_simplify(&folded);

        // If no change occurred, we're done
        if simplified == current || iterations >= MAX_ITERATIONS {
            return simplified;
        }

        current = simplified;
        iterations += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_fold_addition() {
        let expr = TLExpr::add(TLExpr::constant(2.0), TLExpr::constant(3.0));
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(5.0));
    }

    #[test]
    fn test_constant_fold_multiplication() {
        let expr = TLExpr::mul(TLExpr::constant(4.0), TLExpr::constant(5.0));
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(20.0));
    }

    #[test]
    fn test_constant_fold_nested() {
        // (2 + 3) * 4 = 20
        let expr = TLExpr::mul(
            TLExpr::add(TLExpr::constant(2.0), TLExpr::constant(3.0)),
            TLExpr::constant(4.0),
        );
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(20.0));
    }

    #[test]
    fn test_algebraic_simplify_add_zero() {
        let expr = TLExpr::add(TLExpr::constant(5.0), TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(5.0));
    }

    #[test]
    fn test_algebraic_simplify_mul_one() {
        let expr = TLExpr::mul(TLExpr::constant(7.0), TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(7.0));
    }

    #[test]
    fn test_algebraic_simplify_mul_zero() {
        let expr = TLExpr::mul(TLExpr::constant(7.0), TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_algebraic_simplify_double_negation() {
        let expr = TLExpr::negate(TLExpr::negate(TLExpr::constant(5.0)));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(5.0));
    }

    #[test]
    fn test_optimize_expr_combined() {
        // (2 + 3) * 1 should become 5
        let expr = TLExpr::mul(
            TLExpr::add(TLExpr::constant(2.0), TLExpr::constant(3.0)),
            TLExpr::constant(1.0),
        );
        let optimized = optimize_expr(&expr);
        assert_eq!(optimized, TLExpr::Constant(5.0));
    }

    #[test]
    fn test_propagate_constants_let_binding() {
        // let x = 5 in x + 3 should become 8
        let expr = TLExpr::let_binding(
            "x".to_string(),
            TLExpr::constant(5.0),
            TLExpr::add(TLExpr::pred("x", vec![]), TLExpr::constant(3.0)),
        );
        let propagated = propagate_constants(&expr);
        // After propagation: 5 + 3
        let folded = constant_fold(&propagated);
        assert_eq!(folded, TLExpr::Constant(8.0));
    }

    #[test]
    fn test_propagate_constants_nested_let() {
        // let x = 2 in let y = x + 1 in y * 3 should become 9
        let expr = TLExpr::let_binding(
            "x".to_string(),
            TLExpr::constant(2.0),
            TLExpr::let_binding(
                "y".to_string(),
                TLExpr::add(TLExpr::pred("x", vec![]), TLExpr::constant(1.0)),
                TLExpr::mul(TLExpr::pred("y", vec![]), TLExpr::constant(3.0)),
            ),
        );
        let optimized = optimize_expr(&expr);
        assert_eq!(optimized, TLExpr::Constant(9.0));
    }

    #[test]
    fn test_algebraic_simplify_and_true() {
        let expr = TLExpr::and(TLExpr::pred("P", vec![]), TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::pred("P", vec![]));
    }

    #[test]
    fn test_algebraic_simplify_and_false() {
        let expr = TLExpr::and(TLExpr::pred("P", vec![]), TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_algebraic_simplify_or_false() {
        let expr = TLExpr::or(TLExpr::pred("P", vec![]), TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::pred("P", vec![]));
    }

    #[test]
    fn test_algebraic_simplify_or_true() {
        let expr = TLExpr::or(TLExpr::pred("P", vec![]), TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_algebraic_simplify_implies_true_antecedent() {
        let expr = TLExpr::imply(TLExpr::constant(1.0), TLExpr::pred("Q", vec![]));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::pred("Q", vec![]));
    }

    #[test]
    fn test_algebraic_simplify_implies_false_antecedent() {
        let expr = TLExpr::imply(TLExpr::constant(0.0), TLExpr::pred("Q", vec![]));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_algebraic_simplify_implies_true_consequent() {
        let expr = TLExpr::imply(TLExpr::pred("P", vec![]), TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_algebraic_simplify_implies_false_consequent() {
        let expr = TLExpr::imply(TLExpr::pred("P", vec![]), TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        // P → FALSE = ¬P
        matches!(simplified, TLExpr::Not(_));
    }

    #[test]
    fn test_algebraic_simplify_same_comparison() {
        // x = x should become TRUE (1.0)
        let x = TLExpr::pred("x", vec![]);
        let expr = TLExpr::eq(x.clone(), x);
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_algebraic_simplify_comparison_lt_same() {
        // x < x should become FALSE (0.0)
        let x = TLExpr::pred("x", vec![]);
        let expr = TLExpr::lt(x.clone(), x);
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_algebraic_simplify_comparison_lte_same() {
        // x <= x should become TRUE (1.0)
        let x = TLExpr::pred("x", vec![]);
        let expr = TLExpr::lte(x.clone(), x);
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_algebraic_simplify_division_same_constant() {
        // 5.0 / 5.0 should become 1.0
        let expr = TLExpr::div(TLExpr::constant(5.0), TLExpr::constant(5.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_modal_simplify_box_true() {
        let expr = TLExpr::modal_box(TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_modal_simplify_box_false() {
        let expr = TLExpr::modal_box(TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_modal_simplify_diamond_true() {
        let expr = TLExpr::modal_diamond(TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_modal_simplify_diamond_false() {
        let expr = TLExpr::modal_diamond(TLExpr::constant(0.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_temporal_simplify_next_true() {
        let expr = TLExpr::next(TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_temporal_simplify_eventually_true() {
        let expr = TLExpr::eventually(TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_temporal_simplify_always_true() {
        let expr = TLExpr::always(TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_temporal_simplify_eventually_idempotent() {
        // F(F(P)) = F(P)
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::eventually(TLExpr::eventually(p.clone()));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::eventually(p));
    }

    #[test]
    fn test_temporal_simplify_always_idempotent() {
        // G(G(P)) = G(P)
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::always(TLExpr::always(p.clone()));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::always(p));
    }

    #[test]
    fn test_temporal_simplify_until_true() {
        let expr = TLExpr::until(TLExpr::pred("P", vec![]), TLExpr::constant(1.0));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_temporal_simplify_until_false_left() {
        // FALSE U P = F(P)
        let p = TLExpr::pred("P", vec![]);
        let expr = TLExpr::until(TLExpr::constant(0.0), p.clone());
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::eventually(p));
    }

    #[test]
    fn test_algebraic_simplify_absorption_and_or() {
        // A ∧ (A ∨ B) = A
        let a = TLExpr::pred("A", vec![]);
        let b = TLExpr::pred("B", vec![]);
        let expr = TLExpr::and(a.clone(), TLExpr::or(a.clone(), b));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, a);
    }

    #[test]
    fn test_algebraic_simplify_absorption_or_and() {
        // A ∨ (A ∧ B) = A
        let a = TLExpr::pred("A", vec![]);
        let b = TLExpr::pred("B", vec![]);
        let expr = TLExpr::or(a.clone(), TLExpr::and(a.clone(), b));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, a);
    }

    #[test]
    fn test_algebraic_simplify_idempotence_and() {
        // A ∧ A = A
        let a = TLExpr::pred("A", vec![]);
        let expr = TLExpr::and(a.clone(), a.clone());
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, a);
    }

    #[test]
    fn test_algebraic_simplify_idempotence_or() {
        // A ∨ A = A
        let a = TLExpr::pred("A", vec![]);
        let expr = TLExpr::or(a.clone(), a.clone());
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, a);
    }

    #[test]
    fn test_algebraic_simplify_complement_and() {
        // A ∧ ¬A = FALSE
        let a = TLExpr::pred("A", vec![]);
        let expr = TLExpr::and(a.clone(), TLExpr::negate(a));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_algebraic_simplify_complement_or() {
        // A ∨ ¬A = TRUE
        let a = TLExpr::pred("A", vec![]);
        let expr = TLExpr::or(a.clone(), TLExpr::negate(a));
        let simplified = algebraic_simplify(&expr);
        assert_eq!(simplified, TLExpr::Constant(1.0));
    }
}
