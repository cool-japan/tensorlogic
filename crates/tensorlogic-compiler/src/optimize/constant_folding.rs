//! Constant folding optimization pass.
//!
//! This module implements compile-time evaluation of constant expressions,
//! reducing them to single constant values where possible.

use tensorlogic_ir::TLExpr;

/// Statistics from constant folding optimization.
#[derive(Debug, Default, Clone)]
pub struct ConstantFoldingStats {
    /// Number of binary operations folded
    pub binary_ops_folded: usize,
    /// Number of unary operations folded
    pub unary_ops_folded: usize,
    /// Total expressions processed
    pub total_processed: usize,
}

/// Optimize an expression by folding constant subexpressions.
///
/// This pass evaluates constant expressions at compile time, replacing them
/// with their computed values. This can significantly reduce runtime computation
/// for expressions involving constants.
///
/// # Examples
///
/// ```
/// use tensorlogic_compiler::optimize::fold_constants;
/// use tensorlogic_ir::TLExpr;
///
/// // 2.0 + 3.0 => 5.0
/// let expr = TLExpr::Add(
///     Box::new(TLExpr::Constant(2.0)),
///     Box::new(TLExpr::Constant(3.0)),
/// );
///
/// let (optimized, stats) = fold_constants(&expr);
/// assert!(matches!(optimized, TLExpr::Constant(5.0)));
/// assert_eq!(stats.binary_ops_folded, 1);
/// ```
pub fn fold_constants(expr: &TLExpr) -> (TLExpr, ConstantFoldingStats) {
    let mut stats = ConstantFoldingStats::default();
    let result = fold_constants_impl(expr, &mut stats);
    (result, stats)
}

fn fold_constants_impl(expr: &TLExpr, stats: &mut ConstantFoldingStats) -> TLExpr {
    stats.total_processed += 1;

    match expr {
        #[allow(unreachable_patterns)] // Binary arithmetic operations
        TLExpr::Add(left, right) => fold_binary_op(
            left,
            right,
            stats,
            |a, b| a + b,
            |l, r| TLExpr::Add(Box::new(l), Box::new(r)),
        ),
        TLExpr::Sub(left, right) => fold_binary_op(
            left,
            right,
            stats,
            |a, b| a - b,
            |l, r| TLExpr::Sub(Box::new(l), Box::new(r)),
        ),
        TLExpr::Mul(left, right) => fold_binary_op(
            left,
            right,
            stats,
            |a, b| a * b,
            |l, r| TLExpr::Mul(Box::new(l), Box::new(r)),
        ),
        TLExpr::Div(left, right) => fold_binary_op(
            left,
            right,
            stats,
            |a, b| {
                if b.abs() < f64::EPSILON {
                    f64::NAN // Division by zero
                } else {
                    a / b
                }
            },
            |l, r| TLExpr::Div(Box::new(l), Box::new(r)),
        ),
        TLExpr::Pow(left, right) => fold_binary_op(
            left,
            right,
            stats,
            |a, b| a.powf(b),
            |l, r| TLExpr::Pow(Box::new(l), Box::new(r)),
        ),
        TLExpr::Mod(left, right) => fold_binary_op(
            left,
            right,
            stats,
            |a, b| a % b,
            |l, r| TLExpr::Mod(Box::new(l), Box::new(r)),
        ),
        TLExpr::Min(left, right) => fold_binary_op(
            left,
            right,
            stats,
            |a, b| a.min(b),
            |l, r| TLExpr::Min(Box::new(l), Box::new(r)),
        ),
        TLExpr::Max(left, right) => fold_binary_op(
            left,
            right,
            stats,
            |a, b| a.max(b),
            |l, r| TLExpr::Max(Box::new(l), Box::new(r)),
        ),

        // Unary mathematical operations
        TLExpr::Abs(inner) => {
            fold_unary_op(inner, stats, |x| x.abs(), |i| TLExpr::Abs(Box::new(i)))
        }
        TLExpr::Floor(inner) => {
            fold_unary_op(inner, stats, |x| x.floor(), |i| TLExpr::Floor(Box::new(i)))
        }
        TLExpr::Ceil(inner) => {
            fold_unary_op(inner, stats, |x| x.ceil(), |i| TLExpr::Ceil(Box::new(i)))
        }
        TLExpr::Round(inner) => {
            fold_unary_op(inner, stats, |x| x.round(), |i| TLExpr::Round(Box::new(i)))
        }
        TLExpr::Sqrt(inner) => {
            fold_unary_op(inner, stats, |x| x.sqrt(), |i| TLExpr::Sqrt(Box::new(i)))
        }
        TLExpr::Exp(inner) => {
            fold_unary_op(inner, stats, |x| x.exp(), |i| TLExpr::Exp(Box::new(i)))
        }
        TLExpr::Log(inner) => fold_unary_op(inner, stats, |x| x.ln(), |i| TLExpr::Log(Box::new(i))),
        TLExpr::Sin(inner) => {
            fold_unary_op(inner, stats, |x| x.sin(), |i| TLExpr::Sin(Box::new(i)))
        }
        TLExpr::Cos(inner) => {
            fold_unary_op(inner, stats, |x| x.cos(), |i| TLExpr::Cos(Box::new(i)))
        }
        TLExpr::Tan(inner) => {
            fold_unary_op(inner, stats, |x| x.tan(), |i| TLExpr::Tan(Box::new(i)))
        }

        // Logical operations (can't fold without knowing tensor values)
        TLExpr::And(left, right) => {
            let left_opt = fold_constants_impl(left, stats);
            let right_opt = fold_constants_impl(right, stats);
            TLExpr::And(Box::new(left_opt), Box::new(right_opt))
        }
        TLExpr::Or(left, right) => {
            let left_opt = fold_constants_impl(left, stats);
            let right_opt = fold_constants_impl(right, stats);
            TLExpr::Or(Box::new(left_opt), Box::new(right_opt))
        }
        TLExpr::Not(inner) => {
            let inner_opt = fold_constants_impl(inner, stats);
            TLExpr::Not(Box::new(inner_opt))
        }
        TLExpr::Imply(left, right) => {
            let left_opt = fold_constants_impl(left, stats);
            let right_opt = fold_constants_impl(right, stats);
            TLExpr::Imply(Box::new(left_opt), Box::new(right_opt))
        }

        // Comparison operations
        TLExpr::Eq(left, right) => {
            let left_opt = fold_constants_impl(left, stats);
            let right_opt = fold_constants_impl(right, stats);
            TLExpr::Eq(Box::new(left_opt), Box::new(right_opt))
        }
        TLExpr::Lt(left, right) => {
            let left_opt = fold_constants_impl(left, stats);
            let right_opt = fold_constants_impl(right, stats);
            TLExpr::Lt(Box::new(left_opt), Box::new(right_opt))
        }
        TLExpr::Gt(left, right) => {
            let left_opt = fold_constants_impl(left, stats);
            let right_opt = fold_constants_impl(right, stats);
            TLExpr::Gt(Box::new(left_opt), Box::new(right_opt))
        }
        TLExpr::Lte(left, right) => {
            let left_opt = fold_constants_impl(left, stats);
            let right_opt = fold_constants_impl(right, stats);
            TLExpr::Lte(Box::new(left_opt), Box::new(right_opt))
        }
        TLExpr::Gte(left, right) => {
            let left_opt = fold_constants_impl(left, stats);
            let right_opt = fold_constants_impl(right, stats);
            TLExpr::Gte(Box::new(left_opt), Box::new(right_opt))
        }

        // Quantifiers and other constructs
        TLExpr::Exists { var, domain, body } => {
            let body_opt = fold_constants_impl(body, stats);
            TLExpr::Exists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
            }
        }
        TLExpr::ForAll { var, domain, body } => {
            let body_opt = fold_constants_impl(body, stats);
            TLExpr::ForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
            }
        }
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            let body_opt = fold_constants_impl(body, stats);
            TLExpr::Aggregate {
                op: op.clone(),
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
                group_by: group_by.clone(),
            }
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond_opt = fold_constants_impl(condition, stats);
            let then_opt = fold_constants_impl(then_branch, stats);
            let else_opt = fold_constants_impl(else_branch, stats);
            TLExpr::IfThenElse {
                condition: Box::new(cond_opt),
                then_branch: Box::new(then_opt),
                else_branch: Box::new(else_opt),
            }
        }
        TLExpr::Let { var, value, body } => {
            let value_opt = fold_constants_impl(value, stats);
            let body_opt = fold_constants_impl(body, stats);
            TLExpr::Let {
                var: var.clone(),
                value: Box::new(value_opt),
                body: Box::new(body_opt),
            }
        }

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => {
            let left_opt = fold_constants_impl(left, stats);
            let right_opt = fold_constants_impl(right, stats);
            TLExpr::TNorm {
                kind: *kind,
                left: Box::new(left_opt),
                right: Box::new(right_opt),
            }
        }
        TLExpr::TCoNorm { kind, left, right } => {
            let left_opt = fold_constants_impl(left, stats);
            let right_opt = fold_constants_impl(right, stats);
            TLExpr::TCoNorm {
                kind: *kind,
                left: Box::new(left_opt),
                right: Box::new(right_opt),
            }
        }
        TLExpr::FuzzyNot { kind, expr: inner } => {
            let inner_opt = fold_constants_impl(inner, stats);
            TLExpr::FuzzyNot {
                kind: *kind,
                expr: Box::new(inner_opt),
            }
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => {
            let premise_opt = fold_constants_impl(premise, stats);
            let conclusion_opt = fold_constants_impl(conclusion, stats);
            TLExpr::FuzzyImplication {
                kind: *kind,
                premise: Box::new(premise_opt),
                conclusion: Box::new(conclusion_opt),
            }
        }
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => {
            let body_opt = fold_constants_impl(body, stats);
            TLExpr::SoftExists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
                temperature: *temperature,
            }
        }
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => {
            let body_opt = fold_constants_impl(body, stats);
            TLExpr::SoftForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
                temperature: *temperature,
            }
        }
        TLExpr::WeightedRule { weight, rule } => {
            let rule_opt = fold_constants_impl(rule, stats);
            TLExpr::WeightedRule {
                weight: *weight,
                rule: Box::new(rule_opt),
            }
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            let alts_opt: Vec<_> = alternatives
                .iter()
                .map(|(w, e)| (*w, fold_constants_impl(e, stats)))
                .collect();
            TLExpr::ProbabilisticChoice {
                alternatives: alts_opt,
            }
        }

        // Modal/temporal logic operators - not yet implemented, pass through with recursion
        TLExpr::Box(inner) => TLExpr::Box(Box::new(fold_constants_impl(inner, stats))),
        TLExpr::Diamond(inner) => TLExpr::Diamond(Box::new(fold_constants_impl(inner, stats))),
        TLExpr::Next(inner) => TLExpr::Next(Box::new(fold_constants_impl(inner, stats))),
        TLExpr::Eventually(inner) => {
            TLExpr::Eventually(Box::new(fold_constants_impl(inner, stats)))
        }
        TLExpr::Always(inner) => TLExpr::Always(Box::new(fold_constants_impl(inner, stats))),
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(fold_constants_impl(before, stats)),
            after: Box::new(fold_constants_impl(after, stats)),
        },
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(fold_constants_impl(released, stats)),
            releaser: Box::new(fold_constants_impl(releaser, stats)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(fold_constants_impl(before, stats)),
            after: Box::new(fold_constants_impl(after, stats)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(fold_constants_impl(released, stats)),
            releaser: Box::new(fold_constants_impl(releaser, stats)),
        },

        // Base cases
        TLExpr::Pred { .. } | TLExpr::Constant(_) | TLExpr::Score(_) => expr.clone(),
        // All other expression types (alpha.3 enhancements) - no constant folding
        _ => expr.clone(),
    }
}

/// Helper function to fold binary operations on constants
fn fold_binary_op<F, C>(
    left: &TLExpr,
    right: &TLExpr,
    stats: &mut ConstantFoldingStats,
    op: F,
    constructor: C,
) -> TLExpr
where
    F: Fn(f64, f64) -> f64,
    C: Fn(TLExpr, TLExpr) -> TLExpr,
{
    let left_opt = fold_constants_impl(left, stats);
    let right_opt = fold_constants_impl(right, stats);

    if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&left_opt, &right_opt) {
        stats.binary_ops_folded += 1;
        TLExpr::Constant(op(*a, *b))
    } else {
        constructor(left_opt, right_opt)
    }
}

/// Helper function to fold unary operations on constants
fn fold_unary_op<F, C>(
    inner: &TLExpr,
    stats: &mut ConstantFoldingStats,
    op: F,
    constructor: C,
) -> TLExpr
where
    F: Fn(f64) -> f64,
    C: Fn(TLExpr) -> TLExpr,
{
    let inner_opt = fold_constants_impl(inner, stats);

    if let TLExpr::Constant(x) = inner_opt {
        stats.unary_ops_folded += 1;
        TLExpr::Constant(op(x))
    } else {
        constructor(inner_opt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_binary_arithmetic() {
        // 2.0 + 3.0 = 5.0
        let expr = TLExpr::Add(
            Box::new(TLExpr::Constant(2.0)),
            Box::new(TLExpr::Constant(3.0)),
        );
        let (result, stats) = fold_constants(&expr);
        assert!(matches!(result, TLExpr::Constant(x) if (x - 5.0).abs() < f64::EPSILON));
        assert_eq!(stats.binary_ops_folded, 1);
    }

    #[test]
    fn test_fold_nested_arithmetic() {
        // (2.0 + 3.0) * 4.0 = 5.0 * 4.0 = 20.0
        let expr = TLExpr::Mul(
            Box::new(TLExpr::Add(
                Box::new(TLExpr::Constant(2.0)),
                Box::new(TLExpr::Constant(3.0)),
            )),
            Box::new(TLExpr::Constant(4.0)),
        );
        let (result, stats) = fold_constants(&expr);
        assert!(matches!(result, TLExpr::Constant(x) if (x - 20.0).abs() < f64::EPSILON));
        assert_eq!(stats.binary_ops_folded, 2);
    }

    #[test]
    fn test_fold_unary_operations() {
        // sqrt(16.0) = 4.0
        let expr = TLExpr::Sqrt(Box::new(TLExpr::Constant(16.0)));
        let (result, stats) = fold_constants(&expr);
        assert!(matches!(result, TLExpr::Constant(x) if (x - 4.0).abs() < f64::EPSILON));
        assert_eq!(stats.unary_ops_folded, 1);
    }

    #[test]
    fn test_fold_trigonometry() {
        // sin(0.0) = 0.0
        let expr = TLExpr::Sin(Box::new(TLExpr::Constant(0.0)));
        let (result, stats) = fold_constants(&expr);
        assert!(matches!(result, TLExpr::Constant(x) if x.abs() < f64::EPSILON));
        assert_eq!(stats.unary_ops_folded, 1);
    }

    #[test]
    fn test_no_fold_with_variables() {
        use tensorlogic_ir::Term;

        // x + 2.0 (cannot fold because of variable)
        let expr = TLExpr::Add(
            Box::new(TLExpr::pred("x", vec![Term::var("i")])),
            Box::new(TLExpr::Constant(2.0)),
        );
        let (result, stats) = fold_constants(&expr);
        assert!(matches!(result, TLExpr::Add(..)));
        assert_eq!(stats.binary_ops_folded, 0);
    }
}
