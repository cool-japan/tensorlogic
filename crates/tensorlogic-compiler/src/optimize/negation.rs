//! Negation optimization pass.
//!
//! This module optimizes negation operations in logical expressions:
//! - Eliminate double negations: NOT(NOT(x)) → x
//! - Apply De Morgan's laws: NOT(AND(x, y)) → OR(NOT(x), NOT(y))
//! - Push negations through quantifiers: NOT(EXISTS x. P(x)) → FORALL x. NOT(P(x))

use tensorlogic_ir::TLExpr;

/// Statistics from negation optimization
#[derive(Debug, Clone, Default)]
pub struct NegationOptStats {
    /// Double negations eliminated
    pub double_negations_eliminated: usize,
    /// De Morgan's laws applied
    pub demorgans_applied: usize,
    /// Quantifier negations pushed
    pub quantifier_negations_pushed: usize,
}

impl std::fmt::Display for NegationOptStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NegationOptStats {{ double_negations: {}, demorgans: {}, quantifier_pushes: {} }}",
            self.double_negations_eliminated,
            self.demorgans_applied,
            self.quantifier_negations_pushed
        )
    }
}

/// Optimize negations in an expression
pub fn optimize_negations(expr: &TLExpr) -> (TLExpr, NegationOptStats) {
    let mut stats = NegationOptStats::default();
    let optimized = optimize_negations_impl(expr, &mut stats);
    (optimized, stats)
}

fn optimize_negations_impl(expr: &TLExpr, stats: &mut NegationOptStats) -> TLExpr {
    match expr {
        #[allow(unreachable_patterns)] // Double negation elimination: NOT(NOT(x)) → x
        TLExpr::Not(inner) => {
            if let TLExpr::Not(inner_inner) = inner.as_ref() {
                stats.double_negations_eliminated += 1;
                optimize_negations_impl(inner_inner, stats)
            } else {
                // Apply De Morgan's laws
                match inner.as_ref() {
                    // NOT(AND(x, y)) → OR(NOT(x), NOT(y))
                    TLExpr::And(a, b) => {
                        stats.demorgans_applied += 1;
                        let not_a =
                            optimize_negations_impl(&TLExpr::negate(a.as_ref().clone()), stats);
                        let not_b =
                            optimize_negations_impl(&TLExpr::negate(b.as_ref().clone()), stats);
                        TLExpr::or(not_a, not_b)
                    }
                    // NOT(OR(x, y)) → AND(NOT(x), NOT(y))
                    TLExpr::Or(a, b) => {
                        stats.demorgans_applied += 1;
                        let not_a =
                            optimize_negations_impl(&TLExpr::negate(a.as_ref().clone()), stats);
                        let not_b =
                            optimize_negations_impl(&TLExpr::negate(b.as_ref().clone()), stats);
                        TLExpr::and(not_a, not_b)
                    }
                    // NOT(EXISTS x. P(x)) → FORALL x. NOT(P(x))
                    TLExpr::Exists { var, domain, body } => {
                        stats.quantifier_negations_pushed += 1;
                        let not_body =
                            optimize_negations_impl(&TLExpr::negate(body.as_ref().clone()), stats);
                        TLExpr::forall(var.clone(), domain.clone(), not_body)
                    }
                    // NOT(FORALL x. P(x)) → EXISTS x. NOT(P(x))
                    TLExpr::ForAll { var, domain, body } => {
                        stats.quantifier_negations_pushed += 1;
                        let not_body =
                            optimize_negations_impl(&TLExpr::negate(body.as_ref().clone()), stats);
                        TLExpr::exists(var.clone(), domain.clone(), not_body)
                    }
                    // Keep negation for other expressions, but optimize inner
                    _ => TLExpr::negate(optimize_negations_impl(inner, stats)),
                }
            }
        }
        // Recurse into subexpressions
        TLExpr::And(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::and(opt_a, opt_b)
        }
        TLExpr::Or(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::or(opt_a, opt_b)
        }
        TLExpr::Imply(premise, conclusion) => {
            let opt_premise = optimize_negations_impl(premise, stats);
            let opt_conclusion = optimize_negations_impl(conclusion, stats);
            TLExpr::imply(opt_premise, opt_conclusion)
        }
        TLExpr::Exists { var, domain, body } => {
            let opt_body = optimize_negations_impl(body, stats);
            TLExpr::exists(var.clone(), domain.clone(), opt_body)
        }
        TLExpr::ForAll { var, domain, body } => {
            let opt_body = optimize_negations_impl(body, stats);
            TLExpr::forall(var.clone(), domain.clone(), opt_body)
        }
        TLExpr::Add(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::add(opt_a, opt_b)
        }
        TLExpr::Sub(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::sub(opt_a, opt_b)
        }
        TLExpr::Mul(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::mul(opt_a, opt_b)
        }
        TLExpr::Div(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::div(opt_a, opt_b)
        }
        TLExpr::Eq(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::eq(opt_a, opt_b)
        }
        TLExpr::Lt(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::lt(opt_a, opt_b)
        }
        TLExpr::Gt(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::gt(opt_a, opt_b)
        }
        TLExpr::Lte(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::lte(opt_a, opt_b)
        }
        TLExpr::Gte(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::gte(opt_a, opt_b)
        }
        TLExpr::Pow(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::pow(opt_a, opt_b)
        }
        TLExpr::Mod(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::modulo(opt_a, opt_b)
        }
        TLExpr::Min(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::min(opt_a, opt_b)
        }
        TLExpr::Max(a, b) => {
            let opt_a = optimize_negations_impl(a, stats);
            let opt_b = optimize_negations_impl(b, stats);
            TLExpr::max(opt_a, opt_b)
        }
        TLExpr::Abs(inner) => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::abs(opt_inner)
        }
        TLExpr::Floor(inner) => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::floor(opt_inner)
        }
        TLExpr::Ceil(inner) => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::ceil(opt_inner)
        }
        TLExpr::Round(inner) => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::round(opt_inner)
        }
        TLExpr::Sqrt(inner) => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::sqrt(opt_inner)
        }
        TLExpr::Exp(inner) => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::exp(opt_inner)
        }
        TLExpr::Log(inner) => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::log(opt_inner)
        }
        TLExpr::Sin(inner) => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::sin(opt_inner)
        }
        TLExpr::Cos(inner) => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::cos(opt_inner)
        }
        TLExpr::Tan(inner) => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::tan(opt_inner)
        }
        TLExpr::Let { var, value, body } => {
            let opt_value = optimize_negations_impl(value, stats);
            let opt_body = optimize_negations_impl(body, stats);
            TLExpr::let_binding(var, opt_value, opt_body)
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            let opt_condition = optimize_negations_impl(condition, stats);
            let opt_then = optimize_negations_impl(then_branch, stats);
            let opt_else = optimize_negations_impl(else_branch, stats);
            TLExpr::if_then_else(opt_condition, opt_then, opt_else)
        }
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            let opt_body = optimize_negations_impl(body, stats);
            TLExpr::Aggregate {
                op: op.clone(),
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(opt_body),
                group_by: group_by.clone(),
            }
        }
        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => {
            let opt_left = optimize_negations_impl(left, stats);
            let opt_right = optimize_negations_impl(right, stats);
            TLExpr::TNorm {
                kind: *kind,
                left: Box::new(opt_left),
                right: Box::new(opt_right),
            }
        }
        TLExpr::TCoNorm { kind, left, right } => {
            let opt_left = optimize_negations_impl(left, stats);
            let opt_right = optimize_negations_impl(right, stats);
            TLExpr::TCoNorm {
                kind: *kind,
                left: Box::new(opt_left),
                right: Box::new(opt_right),
            }
        }
        TLExpr::FuzzyNot { kind, expr: inner } => {
            let opt_inner = optimize_negations_impl(inner, stats);
            TLExpr::FuzzyNot {
                kind: *kind,
                expr: Box::new(opt_inner),
            }
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => {
            let opt_premise = optimize_negations_impl(premise, stats);
            let opt_conclusion = optimize_negations_impl(conclusion, stats);
            TLExpr::FuzzyImplication {
                kind: *kind,
                premise: Box::new(opt_premise),
                conclusion: Box::new(opt_conclusion),
            }
        }
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => {
            let opt_body = optimize_negations_impl(body, stats);
            TLExpr::SoftExists {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(opt_body),
                temperature: *temperature,
            }
        }
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => {
            let opt_body = optimize_negations_impl(body, stats);
            TLExpr::SoftForAll {
                var: var.clone(),
                domain: domain.clone(),
                body: Box::new(opt_body),
                temperature: *temperature,
            }
        }
        TLExpr::WeightedRule { weight, rule } => {
            let opt_rule = optimize_negations_impl(rule, stats);
            TLExpr::WeightedRule {
                weight: *weight,
                rule: Box::new(opt_rule),
            }
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            let opt_alts: Vec<_> = alternatives
                .iter()
                .map(|(w, e)| (*w, optimize_negations_impl(e, stats)))
                .collect();
            TLExpr::ProbabilisticChoice {
                alternatives: opt_alts,
            }
        }

        // Leaf expressions - return as-is
        // Modal/temporal logic operators - not yet implemented, pass through with recursion
        TLExpr::Box(inner) => TLExpr::Box(Box::new(optimize_negations_impl(inner, stats))),
        TLExpr::Diamond(inner) => TLExpr::Diamond(Box::new(optimize_negations_impl(inner, stats))),
        TLExpr::Next(inner) => TLExpr::Next(Box::new(optimize_negations_impl(inner, stats))),
        TLExpr::Eventually(inner) => {
            TLExpr::Eventually(Box::new(optimize_negations_impl(inner, stats)))
        }
        TLExpr::Always(inner) => TLExpr::Always(Box::new(optimize_negations_impl(inner, stats))),
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(optimize_negations_impl(before, stats)),
            after: Box::new(optimize_negations_impl(after, stats)),
        },
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(optimize_negations_impl(released, stats)),
            releaser: Box::new(optimize_negations_impl(releaser, stats)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(optimize_negations_impl(before, stats)),
            after: Box::new(optimize_negations_impl(after, stats)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(optimize_negations_impl(released, stats)),
            releaser: Box::new(optimize_negations_impl(releaser, stats)),
        },

        TLExpr::Pred { .. } | TLExpr::Score { .. } | TLExpr::Constant(_) => expr.clone(),
        // All other expression types (enhancements) - no negation optimization
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_double_negation_elimination() {
        let x = TLExpr::pred("p", vec![Term::var("x")]);
        let not_not_x = TLExpr::negate(TLExpr::negate(x.clone()));

        let (optimized, stats) = optimize_negations(&not_not_x);

        assert_eq!(stats.double_negations_eliminated, 1);
        assert_eq!(optimized, x);
    }

    #[test]
    fn test_triple_negation() {
        let x = TLExpr::pred("p", vec![Term::var("x")]);
        let not_not_not_x = TLExpr::negate(TLExpr::negate(TLExpr::negate(x.clone())));

        let (optimized, stats) = optimize_negations(&not_not_not_x);

        assert_eq!(stats.double_negations_eliminated, 1);
        // Should result in NOT(x)
        matches!(optimized, TLExpr::Not(_));
    }

    #[test]
    fn test_demorgan_and() {
        // NOT(AND(p, q)) → OR(NOT(p), NOT(q))
        let p = TLExpr::pred("p", vec![Term::var("x")]);
        let q = TLExpr::pred("q", vec![Term::var("x")]);
        let not_and = TLExpr::negate(TLExpr::and(p.clone(), q.clone()));

        let (optimized, stats) = optimize_negations(&not_and);

        assert_eq!(stats.demorgans_applied, 1);
        // Should be OR(NOT(p), NOT(q))
        matches!(optimized, TLExpr::Or(_, _));
    }

    #[test]
    fn test_demorgan_or() {
        // NOT(OR(p, q)) → AND(NOT(p), NOT(q))
        let p = TLExpr::pred("p", vec![Term::var("x")]);
        let q = TLExpr::pred("q", vec![Term::var("x")]);
        let not_or = TLExpr::negate(TLExpr::or(p.clone(), q.clone()));

        let (optimized, stats) = optimize_negations(&not_or);

        assert_eq!(stats.demorgans_applied, 1);
        // Should be AND(NOT(p), NOT(q))
        matches!(optimized, TLExpr::And(_, _));
    }

    #[test]
    fn test_quantifier_negation_exists() {
        // NOT(EXISTS x. P(x)) → FORALL x. NOT(P(x))
        let p = TLExpr::pred("p", vec![Term::var("x")]);
        let exists = TLExpr::exists("x", "Domain", p);
        let not_exists = TLExpr::negate(exists);

        let (optimized, stats) = optimize_negations(&not_exists);

        assert_eq!(stats.quantifier_negations_pushed, 1);
        // Should be FORALL
        matches!(optimized, TLExpr::ForAll { .. });
    }

    #[test]
    fn test_quantifier_negation_forall() {
        // NOT(FORALL x. P(x)) → EXISTS x. NOT(P(x))
        let p = TLExpr::pred("p", vec![Term::var("x")]);
        let forall = TLExpr::forall("x", "Domain", p);
        let not_forall = TLExpr::negate(forall);

        let (optimized, stats) = optimize_negations(&not_forall);

        assert_eq!(stats.quantifier_negations_pushed, 1);
        // Should be EXISTS
        matches!(optimized, TLExpr::Exists { .. });
    }

    #[test]
    fn test_complex_nested_negation() {
        // NOT(AND(NOT(p), NOT(q))) should become OR(p, q)
        let p = TLExpr::pred("p", vec![Term::var("x")]);
        let q = TLExpr::pred("q", vec![Term::var("x")]);
        let expr = TLExpr::negate(TLExpr::and(
            TLExpr::negate(p.clone()),
            TLExpr::negate(q.clone()),
        ));

        let (_optimized, stats) = optimize_negations(&expr);

        // Should apply De Morgan's law and eliminate double negations
        assert!(stats.demorgans_applied >= 1);
        assert!(stats.double_negations_eliminated >= 2);
    }

    #[test]
    fn test_no_optimization_needed() {
        let x = TLExpr::pred("p", vec![Term::var("x")]);
        let y = TLExpr::pred("q", vec![Term::var("x")]);
        let expr = TLExpr::and(x, y);

        let (optimized, stats) = optimize_negations(&expr);

        assert_eq!(stats.double_negations_eliminated, 0);
        assert_eq!(stats.demorgans_applied, 0);
        assert_eq!(optimized, expr);
    }
}
