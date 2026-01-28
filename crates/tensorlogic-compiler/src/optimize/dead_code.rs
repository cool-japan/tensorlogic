//! Dead code elimination optimization pass.
//!
//! This module provides optimizations that remove unreachable or redundant
//! code from TLExpr expressions. Examples include:
//!
//! - `if true then A else B` → `A`
//! - `if false then A else B` → `B`
//! - `AND(false, x)` → `false` (short-circuit)
//! - `OR(true, x)` → `true` (short-circuit)
//! - `EXISTS x. constant` → `constant` (if x is not free in constant)
//! - Remove unused subexpressions that don't affect the result
//!
//! # Examples
//!
//! ```
//! use tensorlogic_compiler::optimize::eliminate_dead_code;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! // if true then A else B → A
//! let a = TLExpr::pred("a", vec![Term::var("i")]);
//! let b = TLExpr::pred("b", vec![Term::var("i")]);
//! let expr = TLExpr::IfThenElse {
//!     condition: Box::new(TLExpr::Constant(1.0)),
//!     then_branch: Box::new(a.clone()),
//!     else_branch: Box::new(b),
//! };
//! let (optimized, stats) = eliminate_dead_code(&expr);
//! assert!(stats.branches_eliminated > 0);
//! ```

use std::collections::HashSet;
use tensorlogic_ir::TLExpr;

/// Statistics from dead code elimination.
#[derive(Debug, Clone, Default)]
pub struct DeadCodeStats {
    /// Number of conditional branches eliminated
    pub branches_eliminated: usize,
    /// Number of short-circuit evaluations applied
    pub short_circuits: usize,
    /// Number of unused quantifiers removed
    pub unused_quantifiers_removed: usize,
    /// Number of identity expressions simplified
    pub identity_simplifications: usize,
    /// Total expressions processed
    pub total_processed: usize,
}

impl DeadCodeStats {
    /// Get total number of optimizations applied.
    pub fn total_optimizations(&self) -> usize {
        self.branches_eliminated
            + self.short_circuits
            + self.unused_quantifiers_removed
            + self.identity_simplifications
    }
}

/// Apply dead code elimination to an expression.
///
/// This pass removes unreachable code and simplifies expressions
/// that have known outcomes.
///
/// # Arguments
///
/// * `expr` - The expression to optimize
///
/// # Returns
///
/// A tuple of (optimized expression, statistics)
pub fn eliminate_dead_code(expr: &TLExpr) -> (TLExpr, DeadCodeStats) {
    let mut stats = DeadCodeStats::default();
    let result = eliminate_dead_code_impl(expr, &mut stats);
    (result, stats)
}

fn eliminate_dead_code_impl(expr: &TLExpr, stats: &mut DeadCodeStats) -> TLExpr {
    stats.total_processed += 1;

    match expr {
        // Conditional elimination
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond_opt = eliminate_dead_code_impl(condition, stats);
            let then_opt = eliminate_dead_code_impl(then_branch, stats);
            let else_opt = eliminate_dead_code_impl(else_branch, stats);

            // Check for constant conditions
            if let TLExpr::Constant(c) = &cond_opt {
                stats.branches_eliminated += 1;
                // Non-zero is truthy
                return if *c != 0.0 { then_opt } else { else_opt };
            }

            TLExpr::IfThenElse {
                condition: Box::new(cond_opt),
                then_branch: Box::new(then_opt),
                else_branch: Box::new(else_opt),
            }
        }

        // AND short-circuit: AND(false, x) → false, AND(x, false) → false
        TLExpr::And(lhs, rhs) => {
            let lhs_opt = eliminate_dead_code_impl(lhs, stats);
            let rhs_opt = eliminate_dead_code_impl(rhs, stats);

            // Check for constant false
            if let TLExpr::Constant(c) = &lhs_opt {
                if *c == 0.0 {
                    stats.short_circuits += 1;
                    return TLExpr::Constant(0.0);
                }
            }
            if let TLExpr::Constant(c) = &rhs_opt {
                if *c == 0.0 {
                    stats.short_circuits += 1;
                    return TLExpr::Constant(0.0);
                }
            }

            // AND(true, x) → x
            if let TLExpr::Constant(c) = &lhs_opt {
                if *c != 0.0 {
                    stats.identity_simplifications += 1;
                    return rhs_opt;
                }
            }
            // AND(x, true) → x
            if let TLExpr::Constant(c) = &rhs_opt {
                if *c != 0.0 {
                    stats.identity_simplifications += 1;
                    return lhs_opt;
                }
            }

            TLExpr::And(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // OR short-circuit: OR(true, x) → true, OR(x, true) → true
        TLExpr::Or(lhs, rhs) => {
            let lhs_opt = eliminate_dead_code_impl(lhs, stats);
            let rhs_opt = eliminate_dead_code_impl(rhs, stats);

            // Check for constant true
            if let TLExpr::Constant(c) = &lhs_opt {
                if *c != 0.0 {
                    stats.short_circuits += 1;
                    return TLExpr::Constant(1.0);
                }
            }
            if let TLExpr::Constant(c) = &rhs_opt {
                if *c != 0.0 {
                    stats.short_circuits += 1;
                    return TLExpr::Constant(1.0);
                }
            }

            // OR(false, x) → x
            if let TLExpr::Constant(c) = &lhs_opt {
                if *c == 0.0 {
                    stats.identity_simplifications += 1;
                    return rhs_opt;
                }
            }
            // OR(x, false) → x
            if let TLExpr::Constant(c) = &rhs_opt {
                if *c == 0.0 {
                    stats.identity_simplifications += 1;
                    return lhs_opt;
                }
            }

            TLExpr::Or(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Imply with constant conditions
        TLExpr::Imply(lhs, rhs) => {
            let lhs_opt = eliminate_dead_code_impl(lhs, stats);
            let rhs_opt = eliminate_dead_code_impl(rhs, stats);

            // false → x is always true
            if let TLExpr::Constant(c) = &lhs_opt {
                if *c == 0.0 {
                    stats.short_circuits += 1;
                    return TLExpr::Constant(1.0);
                }
            }
            // x → true is always true
            if let TLExpr::Constant(c) = &rhs_opt {
                if *c != 0.0 {
                    stats.short_circuits += 1;
                    return TLExpr::Constant(1.0);
                }
            }
            // true → x is just x
            if let TLExpr::Constant(c) = &lhs_opt {
                if *c != 0.0 {
                    stats.identity_simplifications += 1;
                    return rhs_opt;
                }
            }

            TLExpr::Imply(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // EXISTS with unused variable
        TLExpr::Exists { var, domain, body } => {
            let body_opt = eliminate_dead_code_impl(body, stats);

            // If the variable is not free in the body, remove the quantifier
            let free_vars = collect_free_vars(&body_opt);
            if !free_vars.contains(var.as_str()) {
                stats.unused_quantifiers_removed += 1;
                return body_opt;
            }

            TLExpr::Exists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
            }
        }

        // FORALL with unused variable
        TLExpr::ForAll { var, domain, body } => {
            let body_opt = eliminate_dead_code_impl(body, stats);

            // If the variable is not free in the body, remove the quantifier
            let free_vars = collect_free_vars(&body_opt);
            if !free_vars.contains(var.as_str()) {
                stats.unused_quantifiers_removed += 1;
                return body_opt;
            }

            TLExpr::ForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(body_opt),
            }
        }

        // Multiplication by zero
        TLExpr::Mul(lhs, rhs) => {
            let lhs_opt = eliminate_dead_code_impl(lhs, stats);
            let rhs_opt = eliminate_dead_code_impl(rhs, stats);

            // 0 * x = 0, x * 0 = 0
            if matches!(&lhs_opt, TLExpr::Constant(c) if *c == 0.0) {
                stats.short_circuits += 1;
                return TLExpr::Constant(0.0);
            }
            if matches!(&rhs_opt, TLExpr::Constant(c) if *c == 0.0) {
                stats.short_circuits += 1;
                return TLExpr::Constant(0.0);
            }

            TLExpr::Mul(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // NOT with constant
        TLExpr::Not(inner) => {
            let inner_opt = eliminate_dead_code_impl(inner, stats);

            // NOT(true) → false, NOT(false) → true
            if let TLExpr::Constant(c) = &inner_opt {
                stats.identity_simplifications += 1;
                return TLExpr::Constant(if *c == 0.0 { 1.0 } else { 0.0 });
            }

            TLExpr::Not(Box::new(inner_opt))
        }

        // Min/Max with same operand
        TLExpr::Min(lhs, rhs) => {
            let lhs_opt = eliminate_dead_code_impl(lhs, stats);
            let rhs_opt = eliminate_dead_code_impl(rhs, stats);

            // min(x, x) = x
            if exprs_equal(&lhs_opt, &rhs_opt) {
                stats.identity_simplifications += 1;
                return lhs_opt;
            }

            TLExpr::Min(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        TLExpr::Max(lhs, rhs) => {
            let lhs_opt = eliminate_dead_code_impl(lhs, stats);
            let rhs_opt = eliminate_dead_code_impl(rhs, stats);

            // max(x, x) = x
            if exprs_equal(&lhs_opt, &rhs_opt) {
                stats.identity_simplifications += 1;
                return lhs_opt;
            }

            TLExpr::Max(Box::new(lhs_opt), Box::new(rhs_opt))
        }

        // Recursive cases - binary operations
        TLExpr::Add(lhs, rhs) => TLExpr::Add(
            Box::new(eliminate_dead_code_impl(lhs, stats)),
            Box::new(eliminate_dead_code_impl(rhs, stats)),
        ),
        TLExpr::Sub(lhs, rhs) => TLExpr::Sub(
            Box::new(eliminate_dead_code_impl(lhs, stats)),
            Box::new(eliminate_dead_code_impl(rhs, stats)),
        ),
        TLExpr::Div(lhs, rhs) => TLExpr::Div(
            Box::new(eliminate_dead_code_impl(lhs, stats)),
            Box::new(eliminate_dead_code_impl(rhs, stats)),
        ),
        TLExpr::Pow(base, exp) => TLExpr::Pow(
            Box::new(eliminate_dead_code_impl(base, stats)),
            Box::new(eliminate_dead_code_impl(exp, stats)),
        ),
        TLExpr::Mod(lhs, rhs) => TLExpr::Mod(
            Box::new(eliminate_dead_code_impl(lhs, stats)),
            Box::new(eliminate_dead_code_impl(rhs, stats)),
        ),

        // Comparison operations
        TLExpr::Eq(lhs, rhs) => TLExpr::Eq(
            Box::new(eliminate_dead_code_impl(lhs, stats)),
            Box::new(eliminate_dead_code_impl(rhs, stats)),
        ),
        TLExpr::Lt(lhs, rhs) => TLExpr::Lt(
            Box::new(eliminate_dead_code_impl(lhs, stats)),
            Box::new(eliminate_dead_code_impl(rhs, stats)),
        ),
        TLExpr::Lte(lhs, rhs) => TLExpr::Lte(
            Box::new(eliminate_dead_code_impl(lhs, stats)),
            Box::new(eliminate_dead_code_impl(rhs, stats)),
        ),
        TLExpr::Gt(lhs, rhs) => TLExpr::Gt(
            Box::new(eliminate_dead_code_impl(lhs, stats)),
            Box::new(eliminate_dead_code_impl(rhs, stats)),
        ),
        TLExpr::Gte(lhs, rhs) => TLExpr::Gte(
            Box::new(eliminate_dead_code_impl(lhs, stats)),
            Box::new(eliminate_dead_code_impl(rhs, stats)),
        ),

        // Unary operations
        TLExpr::Exp(inner) => TLExpr::Exp(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Log(inner) => TLExpr::Log(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Sqrt(inner) => TLExpr::Sqrt(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Abs(inner) => TLExpr::Abs(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Sin(inner) => TLExpr::Sin(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Cos(inner) => TLExpr::Cos(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Tan(inner) => TLExpr::Tan(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Floor(inner) => TLExpr::Floor(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Ceil(inner) => TLExpr::Ceil(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Round(inner) => TLExpr::Round(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Score(inner) => TLExpr::Score(Box::new(eliminate_dead_code_impl(inner, stats))),

        // Modal operators
        TLExpr::Box(inner) => TLExpr::Box(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Diamond(inner) => TLExpr::Diamond(Box::new(eliminate_dead_code_impl(inner, stats))),

        // Temporal operators
        TLExpr::Next(inner) => TLExpr::Next(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Eventually(inner) => {
            TLExpr::Eventually(Box::new(eliminate_dead_code_impl(inner, stats)))
        }
        TLExpr::Always(inner) => TLExpr::Always(Box::new(eliminate_dead_code_impl(inner, stats))),
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(eliminate_dead_code_impl(before, stats)),
            after: Box::new(eliminate_dead_code_impl(after, stats)),
        },
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(eliminate_dead_code_impl(released, stats)),
            releaser: Box::new(eliminate_dead_code_impl(releaser, stats)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(eliminate_dead_code_impl(before, stats)),
            after: Box::new(eliminate_dead_code_impl(after, stats)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(eliminate_dead_code_impl(released, stats)),
            releaser: Box::new(eliminate_dead_code_impl(releaser, stats)),
        },

        // Fuzzy operators
        TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
            kind: *kind,
            left: Box::new(eliminate_dead_code_impl(left, stats)),
            right: Box::new(eliminate_dead_code_impl(right, stats)),
        },
        TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
            kind: *kind,
            left: Box::new(eliminate_dead_code_impl(left, stats)),
            right: Box::new(eliminate_dead_code_impl(right, stats)),
        },
        TLExpr::FuzzyNot { kind, expr } => TLExpr::FuzzyNot {
            kind: *kind,
            expr: Box::new(eliminate_dead_code_impl(expr, stats)),
        },
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::FuzzyImplication {
            kind: *kind,
            premise: Box::new(eliminate_dead_code_impl(premise, stats)),
            conclusion: Box::new(eliminate_dead_code_impl(conclusion, stats)),
        },

        // Probabilistic operators
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::SoftExists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(eliminate_dead_code_impl(body, stats)),
            temperature: *temperature,
        },
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::SoftForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(eliminate_dead_code_impl(body, stats)),
            temperature: *temperature,
        },
        TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
            weight: *weight,
            rule: Box::new(eliminate_dead_code_impl(rule, stats)),
        },
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::ProbabilisticChoice {
            alternatives: alternatives
                .iter()
                .map(|(prob, e)| (*prob, eliminate_dead_code_impl(e, stats)))
                .collect(),
        },

        // Leaf nodes - no recursion needed
        TLExpr::Pred { .. } | TLExpr::Constant(_) => expr.clone(),

        // Aggregate operations
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => TLExpr::Aggregate {
            op: op.clone(),
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(eliminate_dead_code_impl(body, stats)),
            group_by: group_by.clone(),
        },

        // Let binding
        TLExpr::Let { var, value, body } => TLExpr::Let {
            var: var.clone(),
            value: Box::new(eliminate_dead_code_impl(value, stats)),
            body: Box::new(eliminate_dead_code_impl(body, stats)),
        },

        // All other expression types (enhancements)
        _ => expr.clone(),
    }
}

/// Collect free variables in an expression.
fn collect_free_vars(expr: &TLExpr) -> HashSet<String> {
    let mut vars = HashSet::new();
    collect_free_vars_impl(expr, &mut vars, &HashSet::new());
    vars
}

fn collect_free_vars_impl(
    expr: &TLExpr,
    free_vars: &mut HashSet<String>,
    bound_vars: &HashSet<String>,
) {
    match expr {
        TLExpr::Pred { args, .. } => {
            for arg in args {
                if let tensorlogic_ir::Term::Var(v) = arg {
                    if !bound_vars.contains(v) {
                        free_vars.insert(v.clone());
                    }
                }
            }
        }

        TLExpr::Exists { var, body, .. }
        | TLExpr::ForAll { var, body, .. }
        | TLExpr::SoftExists { var, body, .. }
        | TLExpr::SoftForAll { var, body, .. } => {
            let mut new_bound = bound_vars.clone();
            new_bound.insert(var.clone());
            collect_free_vars_impl(body, free_vars, &new_bound);
        }

        TLExpr::Aggregate { var, body, .. } => {
            let mut new_bound = bound_vars.clone();
            new_bound.insert(var.clone());
            collect_free_vars_impl(body, free_vars, &new_bound);
        }

        TLExpr::Let { var, value, body } => {
            collect_free_vars_impl(value, free_vars, bound_vars);
            let mut new_bound = bound_vars.clone();
            new_bound.insert(var.clone());
            collect_free_vars_impl(body, free_vars, &new_bound);
        }

        // Binary operations
        TLExpr::And(lhs, rhs)
        | TLExpr::Or(lhs, rhs)
        | TLExpr::Imply(lhs, rhs)
        | TLExpr::Add(lhs, rhs)
        | TLExpr::Sub(lhs, rhs)
        | TLExpr::Mul(lhs, rhs)
        | TLExpr::Div(lhs, rhs)
        | TLExpr::Pow(lhs, rhs)
        | TLExpr::Mod(lhs, rhs)
        | TLExpr::Min(lhs, rhs)
        | TLExpr::Max(lhs, rhs)
        | TLExpr::Eq(lhs, rhs)
        | TLExpr::Lt(lhs, rhs)
        | TLExpr::Lte(lhs, rhs)
        | TLExpr::Gt(lhs, rhs)
        | TLExpr::Gte(lhs, rhs) => {
            collect_free_vars_impl(lhs, free_vars, bound_vars);
            collect_free_vars_impl(rhs, free_vars, bound_vars);
        }

        TLExpr::Until { before, after }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => {
            collect_free_vars_impl(before, free_vars, bound_vars);
            collect_free_vars_impl(after, free_vars, bound_vars);
        }

        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            collect_free_vars_impl(left, free_vars, bound_vars);
            collect_free_vars_impl(right, free_vars, bound_vars);
        }

        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => {
            collect_free_vars_impl(premise, free_vars, bound_vars);
            collect_free_vars_impl(conclusion, free_vars, bound_vars);
        }

        // Unary operations
        TLExpr::Not(inner)
        | TLExpr::Exp(inner)
        | TLExpr::Log(inner)
        | TLExpr::Sqrt(inner)
        | TLExpr::Abs(inner)
        | TLExpr::Sin(inner)
        | TLExpr::Cos(inner)
        | TLExpr::Tan(inner)
        | TLExpr::Floor(inner)
        | TLExpr::Ceil(inner)
        | TLExpr::Round(inner)
        | TLExpr::Score(inner)
        | TLExpr::Box(inner)
        | TLExpr::Diamond(inner)
        | TLExpr::Next(inner)
        | TLExpr::Eventually(inner)
        | TLExpr::Always(inner) => {
            collect_free_vars_impl(inner, free_vars, bound_vars);
        }

        TLExpr::FuzzyNot { expr, .. } => {
            collect_free_vars_impl(expr, free_vars, bound_vars);
        }

        TLExpr::WeightedRule { rule, .. } => {
            collect_free_vars_impl(rule, free_vars, bound_vars);
        }

        TLExpr::ProbabilisticChoice { alternatives } => {
            for (_, e) in alternatives {
                collect_free_vars_impl(e, free_vars, bound_vars);
            }
        }

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_free_vars_impl(condition, free_vars, bound_vars);
            collect_free_vars_impl(then_branch, free_vars, bound_vars);
            collect_free_vars_impl(else_branch, free_vars, bound_vars);
        }

        // Leaves
        TLExpr::Constant(_) => {}

        // All other expression types (enhancements)
        _ => {}
    }
}

/// Check if two expressions are structurally equal.
fn exprs_equal(a: &TLExpr, b: &TLExpr) -> bool {
    match (a, b) {
        (TLExpr::Constant(c1), TLExpr::Constant(c2)) => (c1 - c2).abs() < 1e-10,
        (TLExpr::Pred { name: n1, args: a1 }, TLExpr::Pred { name: n2, args: a2 }) => {
            n1 == n2 && a1 == a2
        }
        (TLExpr::Add(l1, r1), TLExpr::Add(l2, r2))
        | (TLExpr::Sub(l1, r1), TLExpr::Sub(l2, r2))
        | (TLExpr::Mul(l1, r1), TLExpr::Mul(l2, r2))
        | (TLExpr::Div(l1, r1), TLExpr::Div(l2, r2))
        | (TLExpr::And(l1, r1), TLExpr::And(l2, r2))
        | (TLExpr::Or(l1, r1), TLExpr::Or(l2, r2)) => exprs_equal(l1, l2) && exprs_equal(r1, r2),
        (TLExpr::Not(e1), TLExpr::Not(e2))
        | (TLExpr::Exp(e1), TLExpr::Exp(e2))
        | (TLExpr::Log(e1), TLExpr::Log(e2))
        | (TLExpr::Sqrt(e1), TLExpr::Sqrt(e2))
        | (TLExpr::Abs(e1), TLExpr::Abs(e2)) => exprs_equal(e1, e2),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_if_true_elimination() {
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("i")]);
        let expr = TLExpr::IfThenElse {
            condition: Box::new(TLExpr::Constant(1.0)),
            then_branch: Box::new(a.clone()),
            else_branch: Box::new(b),
        };
        let (optimized, stats) = eliminate_dead_code(&expr);
        assert_eq!(stats.branches_eliminated, 1);
        assert!(matches!(optimized, TLExpr::Pred { name, .. } if name == "a"));
    }

    #[test]
    fn test_if_false_elimination() {
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("i")]);
        let expr = TLExpr::IfThenElse {
            condition: Box::new(TLExpr::Constant(0.0)),
            then_branch: Box::new(a),
            else_branch: Box::new(b.clone()),
        };
        let (optimized, stats) = eliminate_dead_code(&expr);
        assert_eq!(stats.branches_eliminated, 1);
        assert!(matches!(optimized, TLExpr::Pred { name, .. } if name == "b"));
    }

    #[test]
    fn test_and_short_circuit_false() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::and(TLExpr::Constant(0.0), x);
        let (optimized, stats) = eliminate_dead_code(&expr);
        assert_eq!(stats.short_circuits, 1);
        assert!(matches!(optimized, TLExpr::Constant(c) if c == 0.0));
    }

    #[test]
    fn test_or_short_circuit_true() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::or(TLExpr::Constant(1.0), x);
        let (optimized, stats) = eliminate_dead_code(&expr);
        assert_eq!(stats.short_circuits, 1);
        assert!(matches!(optimized, TLExpr::Constant(c) if c == 1.0));
    }

    #[test]
    fn test_unused_exists_quantifier() {
        let const_expr = TLExpr::Constant(5.0);
        let expr = TLExpr::Exists {
            var: "x".to_string(),
            domain: "D".to_string(),
            body: Box::new(const_expr),
        };
        let (optimized, stats) = eliminate_dead_code(&expr);
        assert_eq!(stats.unused_quantifiers_removed, 1);
        assert!(matches!(optimized, TLExpr::Constant(c) if c == 5.0));
    }

    #[test]
    fn test_used_exists_quantifier() {
        let p_x = TLExpr::pred("p", vec![Term::var("x")]);
        let expr = TLExpr::Exists {
            var: "x".to_string(),
            domain: "D".to_string(),
            body: Box::new(p_x),
        };
        let (optimized, stats) = eliminate_dead_code(&expr);
        assert_eq!(stats.unused_quantifiers_removed, 0);
        assert!(matches!(optimized, TLExpr::Exists { .. }));
    }

    #[test]
    fn test_mul_by_zero() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::mul(x, TLExpr::Constant(0.0));
        let (optimized, stats) = eliminate_dead_code(&expr);
        assert_eq!(stats.short_circuits, 1);
        assert!(matches!(optimized, TLExpr::Constant(c) if c == 0.0));
    }

    #[test]
    fn test_min_same_operands() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::Min(Box::new(x.clone()), Box::new(x));
        let (optimized, stats) = eliminate_dead_code(&expr);
        assert_eq!(stats.identity_simplifications, 1);
        assert!(matches!(optimized, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_not_constant() {
        let expr = TLExpr::Not(Box::new(TLExpr::Constant(1.0)));
        let (optimized, stats) = eliminate_dead_code(&expr);
        assert_eq!(stats.identity_simplifications, 1);
        assert!(matches!(optimized, TLExpr::Constant(c) if c == 0.0));
    }

    #[test]
    fn test_imply_false_antecedent() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::Imply(Box::new(TLExpr::Constant(0.0)), Box::new(x));
        let (optimized, stats) = eliminate_dead_code(&expr);
        assert_eq!(stats.short_circuits, 1);
        assert!(matches!(optimized, TLExpr::Constant(c) if c == 1.0));
    }
}
