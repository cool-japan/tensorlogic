//! Strength reduction optimization pass.
//!
//! This module provides optimizations that replace expensive operations with
//! cheaper equivalents. Examples include:
//!
//! - `x^2` → `x * x` (avoid power function overhead)
//! - `x^0` → `1` (eliminate unnecessary computation)
//! - `x^1` → `x` (eliminate identity operation)
//! - `exp(0)` → `1` (constant evaluation)
//! - `log(1)` → `0` (constant evaluation)
//! - `sqrt(x*x)` → `abs(x)` (eliminate redundant sqrt)
//!
//! # Examples
//!
//! ```
//! use tensorlogic_compiler::optimize::reduce_strength;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! // x^2 → x * x
//! let x = TLExpr::pred("x", vec![Term::var("i")]);
//! let expr = TLExpr::pow(x, TLExpr::Constant(2.0));
//! let (optimized, stats) = reduce_strength(&expr);
//! assert!(stats.power_reductions > 0);
//! ```

use tensorlogic_ir::TLExpr;

/// Statistics from strength reduction optimization.
#[derive(Debug, Clone, Default)]
pub struct StrengthReductionStats {
    /// Number of power operations reduced (e.g., x^2 → x*x)
    pub power_reductions: usize,
    /// Number of operations eliminated (e.g., x^0 → 1)
    pub operations_eliminated: usize,
    /// Number of special function optimizations (e.g., exp(0) → 1)
    pub special_function_optimizations: usize,
    /// Total expressions processed
    pub total_processed: usize,
}

impl StrengthReductionStats {
    /// Get total number of optimizations applied.
    pub fn total_optimizations(&self) -> usize {
        self.power_reductions + self.operations_eliminated + self.special_function_optimizations
    }
}

/// Apply strength reduction optimization to an expression.
///
/// This pass replaces expensive operations with cheaper equivalents.
///
/// # Arguments
///
/// * `expr` - The expression to optimize
///
/// # Returns
///
/// A tuple of (optimized expression, statistics)
pub fn reduce_strength(expr: &TLExpr) -> (TLExpr, StrengthReductionStats) {
    let mut stats = StrengthReductionStats::default();
    let result = reduce_strength_impl(expr, &mut stats);
    (result, stats)
}

fn reduce_strength_impl(expr: &TLExpr, stats: &mut StrengthReductionStats) -> TLExpr {
    stats.total_processed += 1;

    match expr {
        // Power optimizations
        TLExpr::Pow(base, exp) => {
            let base_opt = reduce_strength_impl(base, stats);
            let exp_opt = reduce_strength_impl(exp, stats);

            // Check for constant exponents
            if let TLExpr::Constant(n) = &exp_opt {
                // x^0 → 1
                if *n == 0.0 {
                    stats.operations_eliminated += 1;
                    return TLExpr::Constant(1.0);
                }
                // x^1 → x
                if *n == 1.0 {
                    stats.operations_eliminated += 1;
                    return base_opt;
                }
                // x^2 → x * x (avoid power function overhead)
                if *n == 2.0 {
                    stats.power_reductions += 1;
                    return TLExpr::mul(base_opt.clone(), base_opt);
                }
                // x^3 → x * x * x
                if *n == 3.0 {
                    stats.power_reductions += 1;
                    return TLExpr::mul(base_opt.clone(), TLExpr::mul(base_opt.clone(), base_opt));
                }
                // x^(-1) → 1 / x
                if *n == -1.0 {
                    stats.power_reductions += 1;
                    return TLExpr::div(TLExpr::Constant(1.0), base_opt);
                }
                // x^0.5 → sqrt(x)
                if *n == 0.5 {
                    stats.power_reductions += 1;
                    return TLExpr::sqrt(base_opt);
                }
            }

            TLExpr::Pow(Box::new(base_opt), Box::new(exp_opt))
        }

        // Exponential optimizations
        TLExpr::Exp(inner) => {
            let inner_opt = reduce_strength_impl(inner, stats);

            // exp(0) → 1
            if let TLExpr::Constant(n) = &inner_opt {
                if *n == 0.0 {
                    stats.special_function_optimizations += 1;
                    return TLExpr::Constant(1.0);
                }
                // exp(1) → e (approximate)
                if *n == 1.0 {
                    stats.special_function_optimizations += 1;
                    return TLExpr::Constant(std::f64::consts::E);
                }
            }

            // exp(log(x)) → x
            if let TLExpr::Log(log_inner) = &inner_opt {
                stats.special_function_optimizations += 1;
                return (**log_inner).clone();
            }

            TLExpr::Exp(Box::new(inner_opt))
        }

        // Logarithm optimizations
        TLExpr::Log(inner) => {
            let inner_opt = reduce_strength_impl(inner, stats);

            // log(1) → 0
            if let TLExpr::Constant(n) = &inner_opt {
                if *n == 1.0 {
                    stats.special_function_optimizations += 1;
                    return TLExpr::Constant(0.0);
                }
                // log(e) → 1
                if (*n - std::f64::consts::E).abs() < 1e-10 {
                    stats.special_function_optimizations += 1;
                    return TLExpr::Constant(1.0);
                }
            }

            // log(exp(x)) → x
            if let TLExpr::Exp(exp_inner) = &inner_opt {
                stats.special_function_optimizations += 1;
                return (**exp_inner).clone();
            }

            // log(x^n) → n * log(x)
            if let TLExpr::Pow(base, exp) = &inner_opt {
                if let TLExpr::Constant(_) = exp.as_ref() {
                    stats.special_function_optimizations += 1;
                    return TLExpr::mul((**exp).clone(), TLExpr::log((**base).clone()));
                }
            }

            TLExpr::Log(Box::new(inner_opt))
        }

        // Square root optimizations
        TLExpr::Sqrt(inner) => {
            let inner_opt = reduce_strength_impl(inner, stats);

            // sqrt(0) → 0
            if let TLExpr::Constant(n) = &inner_opt {
                if *n == 0.0 {
                    stats.special_function_optimizations += 1;
                    return TLExpr::Constant(0.0);
                }
                // sqrt(1) → 1
                if *n == 1.0 {
                    stats.special_function_optimizations += 1;
                    return TLExpr::Constant(1.0);
                }
                // sqrt(4) → 2
                if *n == 4.0 {
                    stats.special_function_optimizations += 1;
                    return TLExpr::Constant(2.0);
                }
            }

            // sqrt(x^2) → abs(x) (conceptually; we use x for now)
            if let TLExpr::Pow(base, exp) = &inner_opt {
                if let TLExpr::Constant(n) = exp.as_ref() {
                    if *n == 2.0 {
                        stats.special_function_optimizations += 1;
                        return TLExpr::abs((**base).clone());
                    }
                }
            }

            // sqrt(x * x) → abs(x)
            if let TLExpr::Mul(lhs, rhs) = &inner_opt {
                if lhs == rhs {
                    stats.special_function_optimizations += 1;
                    return TLExpr::abs((**lhs).clone());
                }
            }

            TLExpr::Sqrt(Box::new(inner_opt))
        }

        // Absolute value optimizations
        TLExpr::Abs(inner) => {
            let inner_opt = reduce_strength_impl(inner, stats);

            // abs(constant) → |constant|
            if let TLExpr::Constant(n) = &inner_opt {
                stats.special_function_optimizations += 1;
                return TLExpr::Constant(n.abs());
            }

            // abs(abs(x)) → abs(x)
            if let TLExpr::Abs(_) = &inner_opt {
                stats.special_function_optimizations += 1;
                return inner_opt;
            }

            TLExpr::Abs(Box::new(inner_opt))
        }

        // Division optimizations
        TLExpr::Div(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);

            // x / 1 → x (already handled in algebraic, but good to have here)
            if let TLExpr::Constant(n) = &rhs_opt {
                if *n == 1.0 {
                    stats.operations_eliminated += 1;
                    return lhs_opt;
                }
                // 0 / x → 0
                if let TLExpr::Constant(m) = &lhs_opt {
                    if *m == 0.0 {
                        stats.operations_eliminated += 1;
                        return TLExpr::Constant(0.0);
                    }
                }
                // x / 2 → x * 0.5 (multiplication is often faster)
                if *n == 2.0 {
                    stats.power_reductions += 1;
                    return TLExpr::mul(lhs_opt, TLExpr::Constant(0.5));
                }
                // x / 4 → x * 0.25
                if *n == 4.0 {
                    stats.power_reductions += 1;
                    return TLExpr::mul(lhs_opt, TLExpr::Constant(0.25));
                }
            }

            TLExpr::Div(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Multiplication optimizations for powers
        TLExpr::Mul(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);

            // exp(a) * exp(b) → exp(a + b)
            if let (TLExpr::Exp(a), TLExpr::Exp(b)) = (&lhs_opt, &rhs_opt) {
                stats.special_function_optimizations += 1;
                return TLExpr::exp(TLExpr::add((**a).clone(), (**b).clone()));
            }

            TLExpr::Mul(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Addition for exp/log patterns
        TLExpr::Add(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);

            // log(a) + log(b) → log(a * b)
            if let (TLExpr::Log(a), TLExpr::Log(b)) = (&lhs_opt, &rhs_opt) {
                stats.special_function_optimizations += 1;
                return TLExpr::log(TLExpr::mul((**a).clone(), (**b).clone()));
            }

            TLExpr::Add(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Subtraction
        TLExpr::Sub(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);

            // log(a) - log(b) → log(a / b)
            if let (TLExpr::Log(a), TLExpr::Log(b)) = (&lhs_opt, &rhs_opt) {
                stats.special_function_optimizations += 1;
                return TLExpr::log(TLExpr::div((**a).clone(), (**b).clone()));
            }

            TLExpr::Sub(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Recursive cases for compound expressions
        TLExpr::And(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);
            TLExpr::And(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Or(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);
            TLExpr::Or(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Not(inner) => {
            let inner_opt = reduce_strength_impl(inner, stats);
            TLExpr::Not(Box::new(inner_opt))
        }

        TLExpr::Imply(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);
            TLExpr::Imply(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Exists { var, domain, body } => {
            let body_opt = reduce_strength_impl(body, stats);
            TLExpr::Exists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
            }
        }

        TLExpr::ForAll { var, domain, body } => {
            let body_opt = reduce_strength_impl(body, stats);
            TLExpr::ForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
            }
        }

        TLExpr::Let { var, value, body } => {
            let value_opt = reduce_strength_impl(value, stats);
            let body_opt = reduce_strength_impl(body, stats);
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
            let cond_opt = reduce_strength_impl(condition, stats);
            let then_opt = reduce_strength_impl(then_branch, stats);
            let else_opt = reduce_strength_impl(else_branch, stats);
            TLExpr::IfThenElse {
                condition: Box::new(cond_opt),
                then_branch: Box::new(then_opt),
                else_branch: Box::new(else_opt),
            }
        }

        // Comparison operators
        TLExpr::Eq(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);
            TLExpr::Eq(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Lt(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);
            TLExpr::Lt(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Lte(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);
            TLExpr::Lte(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Gt(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);
            TLExpr::Gt(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Gte(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);
            TLExpr::Gte(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Min/Max
        TLExpr::Min(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);
            TLExpr::Min(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Max(lhs, rhs) => {
            let lhs_opt = reduce_strength_impl(lhs, stats);
            let rhs_opt = reduce_strength_impl(rhs, stats);
            TLExpr::Max(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Modal logic
        TLExpr::Box(inner) => {
            let inner_opt = reduce_strength_impl(inner, stats);
            TLExpr::Box(Box::new(inner_opt))
        }

        TLExpr::Diamond(inner) => {
            let inner_opt = reduce_strength_impl(inner, stats);
            TLExpr::Diamond(Box::new(inner_opt))
        }

        // Temporal logic
        TLExpr::Next(inner) => {
            let inner_opt = reduce_strength_impl(inner, stats);
            TLExpr::Next(Box::new(inner_opt))
        }

        TLExpr::Eventually(inner) => {
            let inner_opt = reduce_strength_impl(inner, stats);
            TLExpr::Eventually(Box::new(inner_opt))
        }

        TLExpr::Always(inner) => {
            let inner_opt = reduce_strength_impl(inner, stats);
            TLExpr::Always(Box::new(inner_opt))
        }

        TLExpr::Until { before, after } => {
            let before_opt = reduce_strength_impl(before, stats);
            let after_opt = reduce_strength_impl(after, stats);
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

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_power_reduction_x_squared() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::pow(x.clone(), TLExpr::Constant(2.0));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.power_reductions, 1);
        // Should be x * x
        if let TLExpr::Mul(lhs, rhs) = optimized {
            assert_eq!(*lhs, x);
            assert_eq!(*rhs, x);
        } else {
            panic!("Expected Mul expression");
        }
    }

    #[test]
    fn test_power_reduction_x_zero() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::pow(x, TLExpr::Constant(0.0));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.operations_eliminated, 1);
        assert_eq!(optimized, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_power_reduction_x_one() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::pow(x.clone(), TLExpr::Constant(1.0));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.operations_eliminated, 1);
        assert_eq!(optimized, x);
    }

    #[test]
    fn test_power_reduction_x_half() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::pow(x.clone(), TLExpr::Constant(0.5));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.power_reductions, 1);
        assert!(matches!(optimized, TLExpr::Sqrt(_)));
    }

    #[test]
    fn test_exp_zero() {
        let expr = TLExpr::exp(TLExpr::Constant(0.0));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.special_function_optimizations, 1);
        assert_eq!(optimized, TLExpr::Constant(1.0));
    }

    #[test]
    fn test_log_one() {
        let expr = TLExpr::log(TLExpr::Constant(1.0));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.special_function_optimizations, 1);
        assert_eq!(optimized, TLExpr::Constant(0.0));
    }

    #[test]
    fn test_exp_log_inverse() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::exp(TLExpr::log(x.clone()));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.special_function_optimizations, 1);
        assert_eq!(optimized, x);
    }

    #[test]
    fn test_log_exp_inverse() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::log(TLExpr::exp(x.clone()));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.special_function_optimizations, 1);
        assert_eq!(optimized, x);
    }

    #[test]
    fn test_sqrt_x_squared() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::sqrt(TLExpr::pow(x.clone(), TLExpr::Constant(2.0)));
        let (optimized, stats) = reduce_strength(&expr);

        // sqrt(x^2) should become abs(x)
        assert!(stats.special_function_optimizations > 0 || stats.power_reductions > 0);
        assert!(matches!(optimized, TLExpr::Abs(_)));
    }

    #[test]
    fn test_sqrt_x_times_x() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::sqrt(TLExpr::mul(x.clone(), x.clone()));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.special_function_optimizations, 1);
        assert!(matches!(optimized, TLExpr::Abs(_)));
    }

    #[test]
    fn test_abs_abs() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::abs(TLExpr::abs(x.clone()));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.special_function_optimizations, 1);
        // Should be abs(x), not abs(abs(x))
        if let TLExpr::Abs(inner) = optimized {
            assert_eq!(*inner, x);
        } else {
            panic!("Expected Abs expression");
        }
    }

    #[test]
    fn test_division_by_two() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::div(x.clone(), TLExpr::Constant(2.0));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.power_reductions, 1);
        // Should be x * 0.5
        if let TLExpr::Mul(lhs, rhs) = optimized {
            assert_eq!(*lhs, x);
            assert_eq!(*rhs, TLExpr::Constant(0.5));
        } else {
            panic!("Expected Mul expression");
        }
    }

    #[test]
    fn test_exp_product() {
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let expr = TLExpr::mul(TLExpr::exp(a.clone()), TLExpr::exp(b.clone()));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.special_function_optimizations, 1);
        // Should be exp(a + b)
        if let TLExpr::Exp(inner) = optimized {
            if let TLExpr::Add(lhs, rhs) = *inner {
                assert_eq!(*lhs, a);
                assert_eq!(*rhs, b);
            } else {
                panic!("Expected Add inside Exp");
            }
        } else {
            panic!("Expected Exp expression");
        }
    }

    #[test]
    fn test_log_sum() {
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let expr = TLExpr::add(TLExpr::log(a.clone()), TLExpr::log(b.clone()));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.special_function_optimizations, 1);
        // Should be log(a * b)
        if let TLExpr::Log(inner) = optimized {
            if let TLExpr::Mul(lhs, rhs) = *inner {
                assert_eq!(*lhs, a);
                assert_eq!(*rhs, b);
            } else {
                panic!("Expected Mul inside Log");
            }
        } else {
            panic!("Expected Log expression");
        }
    }

    #[test]
    fn test_log_difference() {
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("j")]);
        let expr = TLExpr::sub(TLExpr::log(a.clone()), TLExpr::log(b.clone()));
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.special_function_optimizations, 1);
        // Should be log(a / b)
        if let TLExpr::Log(inner) = optimized {
            if let TLExpr::Div(lhs, rhs) = *inner {
                assert_eq!(*lhs, a);
                assert_eq!(*rhs, b);
            } else {
                panic!("Expected Div inside Log");
            }
        } else {
            panic!("Expected Log expression");
        }
    }

    #[test]
    fn test_nested_optimization() {
        // exp(log(x^2)) should reduce to x^2 → then x * x
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::exp(TLExpr::log(TLExpr::pow(x.clone(), TLExpr::Constant(2.0))));
        let (optimized, stats) = reduce_strength(&expr);

        // Multiple optimizations: exp(log(..)) → .., x^2 → x*x
        assert!(stats.total_optimizations() >= 2);
        // Final result should be x * x
        if let TLExpr::Mul(lhs, rhs) = optimized {
            assert_eq!(*lhs, x);
            assert_eq!(*rhs, x);
        } else {
            panic!("Expected Mul expression, got {:?}", optimized);
        }
    }

    #[test]
    fn test_quantifier_body_optimization() {
        let x = TLExpr::pred("x", vec![Term::var("y")]);
        let body = TLExpr::pow(x.clone(), TLExpr::Constant(2.0));
        let expr = TLExpr::exists("y", "D", body);
        let (optimized, stats) = reduce_strength(&expr);

        assert_eq!(stats.power_reductions, 1);
        if let TLExpr::Exists { body, .. } = optimized {
            assert!(matches!(*body, TLExpr::Mul(_, _)));
        } else {
            panic!("Expected Exists expression");
        }
    }

    #[test]
    fn test_stats_total_optimizations() {
        let stats = StrengthReductionStats {
            power_reductions: 3,
            operations_eliminated: 2,
            special_function_optimizations: 5,
            total_processed: 100,
        };
        assert_eq!(stats.total_optimizations(), 10);
    }
}
