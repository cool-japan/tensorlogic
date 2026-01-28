//! Algebraic simplification optimization pass.
//!
//! This module implements algebraic simplifications based on mathematical identities
//! and properties, such as x + 0 = x, x * 1 = x, x * 0 = 0, etc.

use tensorlogic_ir::TLExpr;

/// Statistics from algebraic simplification.
#[derive(Debug, Default, Clone)]
pub struct AlgebraicSimplificationStats {
    /// Number of identity operations eliminated (e.g., x + 0, x * 1)
    pub identities_eliminated: usize,
    /// Number of annihilation operations eliminated (e.g., x * 0)
    pub annihilations_applied: usize,
    /// Number of idempotent operations simplified (e.g., min(x, x) = x)
    pub idempotent_simplified: usize,
    /// Total expressions processed
    pub total_processed: usize,
}

/// Simplify an expression using algebraic identities.
///
/// This pass applies mathematical identities to simplify expressions:
/// - Identity: x + 0 = x, x * 1 = x, x - 0 = x, x / 1 = x
/// - Annihilation: x * 0 = 0, 0 / x = 0
/// - Idempotent: min(x, x) = x, max(x, x) = x
/// - Power identities: x^0 = 1, x^1 = x, 1^x = 1
///
/// # Examples
///
/// ```
/// use tensorlogic_compiler::optimize::simplify_algebraic;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// // x + 0 => x
/// let x = TLExpr::pred("x", vec![Term::var("i")]);
/// let expr = TLExpr::Add(Box::new(x.clone()), Box::new(TLExpr::Constant(0.0)));
///
/// let (simplified, stats) = simplify_algebraic(&expr);
/// assert!(matches!(simplified, TLExpr::Pred { .. }));
/// assert_eq!(stats.identities_eliminated, 1);
/// ```
pub fn simplify_algebraic(expr: &TLExpr) -> (TLExpr, AlgebraicSimplificationStats) {
    let mut stats = AlgebraicSimplificationStats::default();
    let result = simplify_algebraic_impl(expr, &mut stats);
    (result, stats)
}

fn simplify_algebraic_impl(expr: &TLExpr, stats: &mut AlgebraicSimplificationStats) -> TLExpr {
    stats.total_processed += 1;

    match expr {
        // Addition: x + 0 = x, 0 + x = x
        TLExpr::Add(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);

            if is_zero(&right_simp) {
                stats.identities_eliminated += 1;
                left_simp
            } else if is_zero(&left_simp) {
                stats.identities_eliminated += 1;
                right_simp
            } else {
                TLExpr::Add(Box::new(left_simp), Box::new(right_simp))
            }
        }

        // Subtraction: x - 0 = x
        TLExpr::Sub(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);

            if is_zero(&right_simp) {
                stats.identities_eliminated += 1;
                left_simp
            } else {
                TLExpr::Sub(Box::new(left_simp), Box::new(right_simp))
            }
        }

        // Multiplication: x * 1 = x, 1 * x = x, x * 0 = 0, 0 * x = 0
        TLExpr::Mul(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);

            if is_zero(&left_simp) || is_zero(&right_simp) {
                stats.annihilations_applied += 1;
                TLExpr::Constant(0.0)
            } else if is_one(&right_simp) {
                stats.identities_eliminated += 1;
                left_simp
            } else if is_one(&left_simp) {
                stats.identities_eliminated += 1;
                right_simp
            } else {
                TLExpr::Mul(Box::new(left_simp), Box::new(right_simp))
            }
        }

        // Division: x / 1 = x, 0 / x = 0
        TLExpr::Div(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);

            if is_one(&right_simp) {
                stats.identities_eliminated += 1;
                left_simp
            } else if is_zero(&left_simp) {
                stats.annihilations_applied += 1;
                TLExpr::Constant(0.0)
            } else {
                TLExpr::Div(Box::new(left_simp), Box::new(right_simp))
            }
        }

        // Power: x^0 = 1, x^1 = x, 1^x = 1, 0^x = 0 (for x > 0)
        TLExpr::Pow(base, exponent) => {
            let base_simp = simplify_algebraic_impl(base, stats);
            let exp_simp = simplify_algebraic_impl(exponent, stats);

            if is_zero(&exp_simp) {
                stats.identities_eliminated += 1;
                TLExpr::Constant(1.0)
            } else if is_one(&exp_simp) {
                stats.identities_eliminated += 1;
                base_simp
            } else if is_one(&base_simp) {
                stats.annihilations_applied += 1;
                TLExpr::Constant(1.0)
            } else if is_zero(&base_simp) {
                stats.annihilations_applied += 1;
                TLExpr::Constant(0.0)
            } else {
                TLExpr::Pow(Box::new(base_simp), Box::new(exp_simp))
            }
        }

        // Min/Max: min(x, x) = x, max(x, x) = x (idempotent)
        TLExpr::Min(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);

            if expressions_equal(&left_simp, &right_simp) {
                stats.idempotent_simplified += 1;
                left_simp
            } else {
                TLExpr::Min(Box::new(left_simp), Box::new(right_simp))
            }
        }
        TLExpr::Max(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);

            if expressions_equal(&left_simp, &right_simp) {
                stats.idempotent_simplified += 1;
                left_simp
            } else {
                TLExpr::Max(Box::new(left_simp), Box::new(right_simp))
            }
        }

        // Unary operations: abs(abs(x)) = abs(x) is handled by simplification
        TLExpr::Abs(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            // abs(abs(x)) = abs(x)
            if matches!(&inner_simp, TLExpr::Abs(_)) {
                stats.idempotent_simplified += 1;
                inner_simp
            } else {
                TLExpr::Abs(Box::new(inner_simp))
            }
        }

        // Other unary operations
        TLExpr::Floor(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::Floor(Box::new(inner_simp))
        }
        TLExpr::Ceil(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::Ceil(Box::new(inner_simp))
        }
        TLExpr::Round(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::Round(Box::new(inner_simp))
        }
        TLExpr::Sqrt(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::Sqrt(Box::new(inner_simp))
        }
        TLExpr::Exp(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::Exp(Box::new(inner_simp))
        }
        TLExpr::Log(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::Log(Box::new(inner_simp))
        }
        TLExpr::Sin(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::Sin(Box::new(inner_simp))
        }
        TLExpr::Cos(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::Cos(Box::new(inner_simp))
        }
        TLExpr::Tan(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::Tan(Box::new(inner_simp))
        }

        // Modulo
        TLExpr::Mod(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::Mod(Box::new(left_simp), Box::new(right_simp))
        }

        // Logical operations
        TLExpr::And(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::And(Box::new(left_simp), Box::new(right_simp))
        }
        TLExpr::Or(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::Or(Box::new(left_simp), Box::new(right_simp))
        }
        TLExpr::Not(inner) => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::Not(Box::new(inner_simp))
        }
        TLExpr::Imply(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::Imply(Box::new(left_simp), Box::new(right_simp))
        }

        // Comparison operations
        TLExpr::Eq(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::Eq(Box::new(left_simp), Box::new(right_simp))
        }
        TLExpr::Lt(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::Lt(Box::new(left_simp), Box::new(right_simp))
        }
        TLExpr::Gt(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::Gt(Box::new(left_simp), Box::new(right_simp))
        }
        TLExpr::Lte(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::Lte(Box::new(left_simp), Box::new(right_simp))
        }
        TLExpr::Gte(left, right) => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::Gte(Box::new(left_simp), Box::new(right_simp))
        }

        // Quantifiers and other constructs
        TLExpr::Exists { var, domain, body } => {
            let body_simp = simplify_algebraic_impl(body, stats);
            TLExpr::Exists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_simp),
            }
        }
        TLExpr::ForAll { var, domain, body } => {
            let body_simp = simplify_algebraic_impl(body, stats);
            TLExpr::ForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_simp),
            }
        }
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            let body_simp = simplify_algebraic_impl(body, stats);
            TLExpr::Aggregate {
                op: op.clone(),
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_simp),
                group_by: group_by.clone(),
            }
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond_simp = simplify_algebraic_impl(condition, stats);
            let then_simp = simplify_algebraic_impl(then_branch, stats);
            let else_simp = simplify_algebraic_impl(else_branch, stats);
            TLExpr::IfThenElse {
                condition: Box::new(cond_simp),
                then_branch: Box::new(then_simp),
                else_branch: Box::new(else_simp),
            }
        }
        TLExpr::Let { var, value, body } => {
            let value_simp = simplify_algebraic_impl(value, stats);
            let body_simp = simplify_algebraic_impl(body, stats);
            TLExpr::Let {
                var: var.clone(),
                value: Box::new(value_simp),
                body: Box::new(body_simp),
            }
        }

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::TNorm {
                kind: *kind,
                left: Box::new(left_simp),
                right: Box::new(right_simp),
            }
        }
        TLExpr::TCoNorm { kind, left, right } => {
            let left_simp = simplify_algebraic_impl(left, stats);
            let right_simp = simplify_algebraic_impl(right, stats);
            TLExpr::TCoNorm {
                kind: *kind,
                left: Box::new(left_simp),
                right: Box::new(right_simp),
            }
        }
        TLExpr::FuzzyNot { kind, expr: inner } => {
            let inner_simp = simplify_algebraic_impl(inner, stats);
            TLExpr::FuzzyNot {
                kind: *kind,
                expr: Box::new(inner_simp),
            }
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => {
            let premise_simp = simplify_algebraic_impl(premise, stats);
            let conclusion_simp = simplify_algebraic_impl(conclusion, stats);
            TLExpr::FuzzyImplication {
                kind: *kind,
                premise: Box::new(premise_simp),
                conclusion: Box::new(conclusion_simp),
            }
        }
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => {
            let body_simp = simplify_algebraic_impl(body, stats);
            TLExpr::SoftExists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_simp),
                temperature: *temperature,
            }
        }
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => {
            let body_simp = simplify_algebraic_impl(body, stats);
            TLExpr::SoftForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_simp),
                temperature: *temperature,
            }
        }
        TLExpr::WeightedRule { weight, rule } => {
            let rule_simp = simplify_algebraic_impl(rule, stats);
            TLExpr::WeightedRule {
                weight: *weight,
                rule: Box::new(rule_simp),
            }
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            let alts_simp: Vec<_> = alternatives
                .iter()
                .map(|(w, e)| (*w, simplify_algebraic_impl(e, stats)))
                .collect();
            TLExpr::ProbabilisticChoice {
                alternatives: alts_simp,
            }
        }

        // Modal/temporal logic operators - not yet implemented, pass through with recursion
        TLExpr::Box(inner) => TLExpr::Box(Box::new(simplify_algebraic_impl(inner, stats))),
        TLExpr::Diamond(inner) => TLExpr::Diamond(Box::new(simplify_algebraic_impl(inner, stats))),
        TLExpr::Next(inner) => TLExpr::Next(Box::new(simplify_algebraic_impl(inner, stats))),
        TLExpr::Eventually(inner) => {
            TLExpr::Eventually(Box::new(simplify_algebraic_impl(inner, stats)))
        }
        TLExpr::Always(inner) => TLExpr::Always(Box::new(simplify_algebraic_impl(inner, stats))),
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(simplify_algebraic_impl(before, stats)),
            after: Box::new(simplify_algebraic_impl(after, stats)),
        },
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(simplify_algebraic_impl(released, stats)),
            releaser: Box::new(simplify_algebraic_impl(releaser, stats)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(simplify_algebraic_impl(before, stats)),
            after: Box::new(simplify_algebraic_impl(after, stats)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(simplify_algebraic_impl(released, stats)),
            releaser: Box::new(simplify_algebraic_impl(releaser, stats)),
        },

        // Base cases
        TLExpr::Pred { .. } | TLExpr::Constant(_) | TLExpr::Score(_) => expr.clone(),
        // All other expression types (enhancements) - no algebraic simplification
        _ => expr.clone(),
    }
}

/// Check if an expression is constant zero
fn is_zero(expr: &TLExpr) -> bool {
    matches!(expr, TLExpr::Constant(x) if x.abs() < f64::EPSILON)
}

/// Check if an expression is constant one
fn is_one(expr: &TLExpr) -> bool {
    matches!(expr, TLExpr::Constant(x) if (x - 1.0).abs() < f64::EPSILON)
}

/// Check if two expressions are structurally equal (for idempotent simplification)
fn expressions_equal(a: &TLExpr, b: &TLExpr) -> bool {
    match (a, b) {
        (TLExpr::Constant(x), TLExpr::Constant(y)) => (x - y).abs() < f64::EPSILON,
        (TLExpr::Pred { name: n1, args: a1 }, TLExpr::Pred { name: n2, args: a2 }) => {
            n1 == n2 && a1 == a2
        }
        _ => false, // Conservative: only check simple cases
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_addition_identity() {
        // x + 0 = x
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::Add(Box::new(x.clone()), Box::new(TLExpr::Constant(0.0)));

        let (result, stats) = simplify_algebraic(&expr);
        assert!(matches!(result, TLExpr::Pred { .. }));
        assert_eq!(stats.identities_eliminated, 1);
    }

    #[test]
    fn test_multiplication_identity() {
        // x * 1 = x
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::Mul(Box::new(x.clone()), Box::new(TLExpr::Constant(1.0)));

        let (result, stats) = simplify_algebraic(&expr);
        assert!(matches!(result, TLExpr::Pred { .. }));
        assert_eq!(stats.identities_eliminated, 1);
    }

    #[test]
    fn test_multiplication_annihilation() {
        // x * 0 = 0
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::Mul(Box::new(x), Box::new(TLExpr::Constant(0.0)));

        let (result, stats) = simplify_algebraic(&expr);
        assert!(matches!(result, TLExpr::Constant(0.0)));
        assert_eq!(stats.annihilations_applied, 1);
    }

    #[test]
    fn test_power_identities() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);

        // x^0 = 1
        let expr1 = TLExpr::Pow(Box::new(x.clone()), Box::new(TLExpr::Constant(0.0)));
        let (result1, stats1) = simplify_algebraic(&expr1);
        assert!(matches!(result1, TLExpr::Constant(1.0)));
        assert_eq!(stats1.identities_eliminated, 1);

        // x^1 = x
        let expr2 = TLExpr::Pow(Box::new(x), Box::new(TLExpr::Constant(1.0)));
        let (result2, stats2) = simplify_algebraic(&expr2);
        assert!(matches!(result2, TLExpr::Pred { .. }));
        assert_eq!(stats2.identities_eliminated, 1);
    }

    #[test]
    fn test_idempotent_min_max() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);

        // min(x, x) = x
        let expr1 = TLExpr::Min(Box::new(x.clone()), Box::new(x.clone()));
        let (result1, stats1) = simplify_algebraic(&expr1);
        assert!(matches!(result1, TLExpr::Pred { .. }));
        assert_eq!(stats1.idempotent_simplified, 1);

        // max(x, x) = x
        let expr2 = TLExpr::Max(Box::new(x.clone()), Box::new(x));
        let (result2, stats2) = simplify_algebraic(&expr2);
        assert!(matches!(result2, TLExpr::Pred { .. }));
        assert_eq!(stats2.idempotent_simplified, 1);
    }

    #[test]
    fn test_nested_simplification() {
        // (x + 0) * 1 = x
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let add = TLExpr::Add(Box::new(x), Box::new(TLExpr::Constant(0.0)));
        let expr = TLExpr::Mul(Box::new(add), Box::new(TLExpr::Constant(1.0)));

        let (result, stats) = simplify_algebraic(&expr);
        assert!(matches!(result, TLExpr::Pred { .. }));
        assert_eq!(stats.identities_eliminated, 2);
    }
}
