//! Temporal logic equivalence optimizations.
//!
//! This module implements optimization passes based on temporal logic equivalences,
//! including duality laws and derived operators.

use super::TLExpr;

/// Apply temporal logic equivalences to simplify an expression.
///
/// This function applies standard temporal logic (LTL) equivalences:
/// - **Duality**: FP ≡ ¬G¬P and GP ≡ ¬F¬P
/// - **Derived operators**: FP ≡ true U P
/// - **Derived operators**: GP ≡ false R P
/// - **Idempotence**: FFP ≡ FP and GGP ≡ GP
/// - **Absorption**: GFP ≡ FGP ≡ FP (for certain cases)
pub fn apply_temporal_equivalences(expr: &TLExpr) -> TLExpr {
    match expr {
        // Duality: ¬G¬P → FP
        TLExpr::Not(e) => {
            let inner = apply_temporal_equivalences(e);
            if let TLExpr::Always(always_inner) = &inner {
                if let TLExpr::Not(not_inner) = &**always_inner {
                    return TLExpr::Eventually(not_inner.clone());
                }
            }
            // Duality: ¬F¬P → GP
            if let TLExpr::Eventually(eventually_inner) = &inner {
                if let TLExpr::Not(not_inner) = &**eventually_inner {
                    return TLExpr::Always(not_inner.clone());
                }
            }
            TLExpr::Not(Box::new(inner))
        }

        // Idempotence: FFP → FP
        TLExpr::Eventually(e) => {
            let inner = apply_temporal_equivalences(e);
            if let TLExpr::Eventually(inner_inner) = &inner {
                return TLExpr::Eventually(inner_inner.clone());
            }
            TLExpr::Eventually(Box::new(inner))
        }

        // Idempotence: GGP → GP
        TLExpr::Always(e) => {
            let inner = apply_temporal_equivalences(e);
            if let TLExpr::Always(inner_inner) = &inner {
                return TLExpr::Always(inner_inner.clone());
            }
            TLExpr::Always(Box::new(inner))
        }

        // Until optimizations
        TLExpr::Until { before, after } => {
            let before_opt = apply_temporal_equivalences(before);
            let after_opt = apply_temporal_equivalences(after);

            // true U P → FP (eventually P)
            if matches!(before_opt, TLExpr::Constant(v) if v >= 1.0) {
                return TLExpr::Eventually(Box::new(after_opt));
            }

            TLExpr::until(before_opt, after_opt)
        }

        // Release optimizations
        TLExpr::Release { released, releaser } => {
            let released_opt = apply_temporal_equivalences(released);
            let releaser_opt = apply_temporal_equivalences(releaser);

            // false R P → GP (always P)
            if matches!(released_opt, TLExpr::Constant(v) if v <= 0.0) {
                return TLExpr::Always(Box::new(releaser_opt));
            }

            TLExpr::release(released_opt, releaser_opt)
        }

        // Weak until
        TLExpr::WeakUntil { before, after } => {
            let before_opt = apply_temporal_equivalences(before);
            let after_opt = apply_temporal_equivalences(after);
            TLExpr::weak_until(before_opt, after_opt)
        }

        // Strong release
        TLExpr::StrongRelease { released, releaser } => {
            let released_opt = apply_temporal_equivalences(released);
            let releaser_opt = apply_temporal_equivalences(releaser);
            TLExpr::strong_release(released_opt, releaser_opt)
        }

        // Next
        TLExpr::Next(e) => TLExpr::next(apply_temporal_equivalences(e)),

        // Binary operators
        TLExpr::And(l, r) => TLExpr::and(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Or(l, r) => TLExpr::or(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Imply(l, r) => TLExpr::imply(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),

        // Arithmetic
        TLExpr::Add(l, r) => TLExpr::add(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Sub(l, r) => TLExpr::sub(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Mul(l, r) => TLExpr::mul(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Div(l, r) => TLExpr::div(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Pow(l, r) => TLExpr::pow(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Mod(l, r) => TLExpr::modulo(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Min(l, r) => TLExpr::min(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Max(l, r) => TLExpr::max(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),

        // Comparison
        TLExpr::Eq(l, r) => TLExpr::eq(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Lt(l, r) => TLExpr::lt(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Gt(l, r) => TLExpr::gt(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Lte(l, r) => TLExpr::lte(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),
        TLExpr::Gte(l, r) => TLExpr::gte(
            apply_temporal_equivalences(l),
            apply_temporal_equivalences(r),
        ),

        // Unary operators (Not is handled above for duality)
        TLExpr::Score(e) => TLExpr::score(apply_temporal_equivalences(e)),
        TLExpr::Abs(e) => TLExpr::abs(apply_temporal_equivalences(e)),
        TLExpr::Floor(e) => TLExpr::floor(apply_temporal_equivalences(e)),
        TLExpr::Ceil(e) => TLExpr::ceil(apply_temporal_equivalences(e)),
        TLExpr::Round(e) => TLExpr::round(apply_temporal_equivalences(e)),
        TLExpr::Sqrt(e) => TLExpr::sqrt(apply_temporal_equivalences(e)),
        TLExpr::Exp(e) => TLExpr::exp(apply_temporal_equivalences(e)),
        TLExpr::Log(e) => TLExpr::log(apply_temporal_equivalences(e)),
        TLExpr::Sin(e) => TLExpr::sin(apply_temporal_equivalences(e)),
        TLExpr::Cos(e) => TLExpr::cos(apply_temporal_equivalences(e)),
        TLExpr::Tan(e) => TLExpr::tan(apply_temporal_equivalences(e)),

        // Modal operators (just recurse)
        TLExpr::Box(e) => TLExpr::modal_box(apply_temporal_equivalences(e)),
        TLExpr::Diamond(e) => TLExpr::modal_diamond(apply_temporal_equivalences(e)),

        // Quantifiers
        TLExpr::Exists { var, domain, body } => TLExpr::exists(
            var.clone(),
            domain.clone(),
            apply_temporal_equivalences(body),
        ),
        TLExpr::ForAll { var, domain, body } => TLExpr::forall(
            var.clone(),
            domain.clone(),
            apply_temporal_equivalences(body),
        ),
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::soft_exists(
            var.clone(),
            domain.clone(),
            apply_temporal_equivalences(body),
            *temperature,
        ),
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::soft_forall(
            var.clone(),
            domain.clone(),
            apply_temporal_equivalences(body),
            *temperature,
        ),

        // Aggregation
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            if let Some(group_vars) = group_by {
                TLExpr::aggregate_with_group_by(
                    op.clone(),
                    var.clone(),
                    domain.clone(),
                    apply_temporal_equivalences(body),
                    group_vars.clone(),
                )
            } else {
                TLExpr::aggregate(
                    op.clone(),
                    var.clone(),
                    domain.clone(),
                    apply_temporal_equivalences(body),
                )
            }
        }

        // Control flow
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::if_then_else(
            apply_temporal_equivalences(condition),
            apply_temporal_equivalences(then_branch),
            apply_temporal_equivalences(else_branch),
        ),
        TLExpr::Let { var, value, body } => TLExpr::let_binding(
            var.clone(),
            apply_temporal_equivalences(value),
            apply_temporal_equivalences(body),
        ),

        // Fuzzy logic
        TLExpr::TNorm { kind, left, right } => TLExpr::tnorm(
            *kind,
            apply_temporal_equivalences(left),
            apply_temporal_equivalences(right),
        ),
        TLExpr::TCoNorm { kind, left, right } => TLExpr::tconorm(
            *kind,
            apply_temporal_equivalences(left),
            apply_temporal_equivalences(right),
        ),
        TLExpr::FuzzyNot { kind, expr } => {
            TLExpr::fuzzy_not(*kind, apply_temporal_equivalences(expr))
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::fuzzy_imply(
            *kind,
            apply_temporal_equivalences(premise),
            apply_temporal_equivalences(conclusion),
        ),

        // Probabilistic
        TLExpr::WeightedRule { weight, rule } => {
            TLExpr::weighted_rule(*weight, apply_temporal_equivalences(rule))
        }
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::probabilistic_choice(
            alternatives
                .iter()
                .map(|(p, e)| (*p, apply_temporal_equivalences(e)))
                .collect(),
        ),

        // Alpha.3 enhancements - just recurse, no temporal-specific optimizations
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => TLExpr::lambda(
            var.clone(),
            var_type.clone(),
            apply_temporal_equivalences(body),
        ),
        TLExpr::Apply { function, argument } => TLExpr::apply(
            apply_temporal_equivalences(function),
            apply_temporal_equivalences(argument),
        ),
        TLExpr::SetMembership { element, set } => TLExpr::set_membership(
            apply_temporal_equivalences(element),
            apply_temporal_equivalences(set),
        ),
        TLExpr::SetUnion { left, right } => TLExpr::set_union(
            apply_temporal_equivalences(left),
            apply_temporal_equivalences(right),
        ),
        TLExpr::SetIntersection { left, right } => TLExpr::set_intersection(
            apply_temporal_equivalences(left),
            apply_temporal_equivalences(right),
        ),
        TLExpr::SetDifference { left, right } => TLExpr::set_difference(
            apply_temporal_equivalences(left),
            apply_temporal_equivalences(right),
        ),
        TLExpr::SetCardinality { set } => TLExpr::set_cardinality(apply_temporal_equivalences(set)),
        TLExpr::EmptySet => expr.clone(),
        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => TLExpr::set_comprehension(
            var.clone(),
            domain.clone(),
            apply_temporal_equivalences(condition),
        ),
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => TLExpr::counting_exists(
            var.clone(),
            domain.clone(),
            apply_temporal_equivalences(body),
            *min_count,
        ),
        TLExpr::CountingForAll {
            var,
            domain,
            body,
            min_count,
        } => TLExpr::counting_forall(
            var.clone(),
            domain.clone(),
            apply_temporal_equivalences(body),
            *min_count,
        ),
        TLExpr::ExactCount {
            var,
            domain,
            body,
            count,
        } => TLExpr::exact_count(
            var.clone(),
            domain.clone(),
            apply_temporal_equivalences(body),
            *count,
        ),
        TLExpr::Majority { var, domain, body } => TLExpr::majority(
            var.clone(),
            domain.clone(),
            apply_temporal_equivalences(body),
        ),
        TLExpr::LeastFixpoint { var, body } => {
            TLExpr::least_fixpoint(var.clone(), apply_temporal_equivalences(body))
        }
        TLExpr::GreatestFixpoint { var, body } => {
            TLExpr::greatest_fixpoint(var.clone(), apply_temporal_equivalences(body))
        }
        TLExpr::Nominal { .. } => expr.clone(),
        TLExpr::At { nominal, formula } => {
            TLExpr::at(nominal.clone(), apply_temporal_equivalences(formula))
        }
        TLExpr::Somewhere { formula } => TLExpr::somewhere(apply_temporal_equivalences(formula)),
        TLExpr::Everywhere { formula } => TLExpr::everywhere(apply_temporal_equivalences(formula)),
        TLExpr::AllDifferent { .. } => expr.clone(),
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => TLExpr::global_cardinality(
            variables.clone(),
            values.iter().map(apply_temporal_equivalences).collect(),
            min_occurrences.clone(),
            max_occurrences.clone(),
        ),
        TLExpr::Abducible { .. } => expr.clone(),
        TLExpr::Explain { formula } => TLExpr::explain(apply_temporal_equivalences(formula)),

        // Leaves
        TLExpr::Pred { .. } | TLExpr::Constant(_) => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_temporal_duality_eventually_to_always() {
        // ¬G¬P → FP
        let expr = TLExpr::negate(TLExpr::always(TLExpr::negate(TLExpr::pred(
            "P",
            vec![Term::var("x")],
        ))));

        let result = apply_temporal_equivalences(&expr);
        assert!(matches!(result, TLExpr::Eventually(_)));
    }

    #[test]
    fn test_temporal_duality_always_to_eventually() {
        // ¬F¬P → GP
        let expr = TLExpr::negate(TLExpr::eventually(TLExpr::negate(TLExpr::pred(
            "P",
            vec![Term::var("x")],
        ))));

        let result = apply_temporal_equivalences(&expr);
        assert!(matches!(result, TLExpr::Always(_)));
    }

    #[test]
    fn test_temporal_idempotence_eventually() {
        // FFP → FP
        let expr = TLExpr::eventually(TLExpr::eventually(TLExpr::pred("P", vec![Term::var("x")])));

        let result = apply_temporal_equivalences(&expr);
        // Should be simplified to a single Eventually
        if let TLExpr::Eventually(inner) = result {
            assert!(matches!(*inner, TLExpr::Pred { .. }));
        } else {
            panic!("Expected Eventually, got {:?}", result);
        }
    }

    #[test]
    fn test_temporal_idempotence_always() {
        // GGP → GP
        let expr = TLExpr::always(TLExpr::always(TLExpr::pred("P", vec![Term::var("x")])));

        let result = apply_temporal_equivalences(&expr);
        // Should be simplified to a single Always
        if let TLExpr::Always(inner) = result {
            assert!(matches!(*inner, TLExpr::Pred { .. }));
        } else {
            panic!("Expected Always, got {:?}", result);
        }
    }

    #[test]
    fn test_until_true_to_eventually() {
        // true U P → FP
        let expr = TLExpr::until(
            TLExpr::constant(1.0),
            TLExpr::pred("P", vec![Term::var("x")]),
        );

        let result = apply_temporal_equivalences(&expr);
        assert!(matches!(result, TLExpr::Eventually(_)));
    }

    #[test]
    fn test_release_false_to_always() {
        // false R P → GP
        let expr = TLExpr::release(
            TLExpr::constant(0.0),
            TLExpr::pred("P", vec![Term::var("x")]),
        );

        let result = apply_temporal_equivalences(&expr);
        assert!(matches!(result, TLExpr::Always(_)));
    }
}
