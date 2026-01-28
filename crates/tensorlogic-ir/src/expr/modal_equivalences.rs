//! Modal logic equivalence optimizations.
//!
//! This module implements optimization passes based on modal logic equivalences,
//! including duality laws and distribution laws.

use super::TLExpr;

/// Apply modal logic equivalences to simplify an expression.
///
/// This function applies standard modal logic equivalences:
/// - **Duality**: ◇P ≡ ¬□¬P and □P ≡ ¬◇¬P
/// - **Distribution**: □(P ∧ Q) ≡ □P ∧ □Q
/// - **Distribution**: ◇(P ∨ Q) ≡ ◇P ∨ ◇Q
/// - **Idempotence**: □□P ≡ □P and ◇◇P ≡ ◇P
pub fn apply_modal_equivalences(expr: &TLExpr) -> TLExpr {
    match expr {
        // Duality: ¬□¬P → ◇P
        TLExpr::Not(e) => {
            let inner = apply_modal_equivalences(e);
            if let TLExpr::Box(box_inner) = &inner {
                if let TLExpr::Not(not_inner) = &**box_inner {
                    return TLExpr::Diamond(not_inner.clone());
                }
            }
            // Duality: ¬◇¬P → □P
            if let TLExpr::Diamond(diamond_inner) = &inner {
                if let TLExpr::Not(not_inner) = &**diamond_inner {
                    return TLExpr::Box(not_inner.clone());
                }
            }
            TLExpr::Not(Box::new(inner))
        }

        // Idempotence: □□P → □P
        TLExpr::Box(e) => {
            let inner = apply_modal_equivalences(e);
            if let TLExpr::Box(inner_inner) = &inner {
                return TLExpr::Box(inner_inner.clone());
            }
            // Distribution: □(P ∧ Q) → □P ∧ □Q (optional optimization)
            if let TLExpr::And(l, r) = &inner {
                return TLExpr::and(TLExpr::Box(l.clone()), TLExpr::Box(r.clone()));
            }
            TLExpr::Box(Box::new(inner))
        }

        // Idempotence: ◇◇P → ◇P
        TLExpr::Diamond(e) => {
            let inner = apply_modal_equivalences(e);
            if let TLExpr::Diamond(inner_inner) = &inner {
                return TLExpr::Diamond(inner_inner.clone());
            }
            // Distribution: ◇(P ∨ Q) → ◇P ∨ ◇Q (optional optimization)
            if let TLExpr::Or(l, r) = &inner {
                return TLExpr::or(TLExpr::Diamond(l.clone()), TLExpr::Diamond(r.clone()));
            }
            TLExpr::Diamond(Box::new(inner))
        }

        // Binary operators
        TLExpr::And(l, r) => TLExpr::and(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Or(l, r) => TLExpr::or(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Imply(l, r) => {
            TLExpr::imply(apply_modal_equivalences(l), apply_modal_equivalences(r))
        }

        // Arithmetic
        TLExpr::Add(l, r) => TLExpr::add(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Sub(l, r) => TLExpr::sub(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Mul(l, r) => TLExpr::mul(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Div(l, r) => TLExpr::div(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Pow(l, r) => TLExpr::pow(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Mod(l, r) => {
            TLExpr::modulo(apply_modal_equivalences(l), apply_modal_equivalences(r))
        }
        TLExpr::Min(l, r) => TLExpr::min(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Max(l, r) => TLExpr::max(apply_modal_equivalences(l), apply_modal_equivalences(r)),

        // Comparison
        TLExpr::Eq(l, r) => TLExpr::eq(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Lt(l, r) => TLExpr::lt(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Gt(l, r) => TLExpr::gt(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Lte(l, r) => TLExpr::lte(apply_modal_equivalences(l), apply_modal_equivalences(r)),
        TLExpr::Gte(l, r) => TLExpr::gte(apply_modal_equivalences(l), apply_modal_equivalences(r)),

        // Unary operators
        TLExpr::Score(e) => TLExpr::score(apply_modal_equivalences(e)),
        TLExpr::Abs(e) => TLExpr::abs(apply_modal_equivalences(e)),
        TLExpr::Floor(e) => TLExpr::floor(apply_modal_equivalences(e)),
        TLExpr::Ceil(e) => TLExpr::ceil(apply_modal_equivalences(e)),
        TLExpr::Round(e) => TLExpr::round(apply_modal_equivalences(e)),
        TLExpr::Sqrt(e) => TLExpr::sqrt(apply_modal_equivalences(e)),
        TLExpr::Exp(e) => TLExpr::exp(apply_modal_equivalences(e)),
        TLExpr::Log(e) => TLExpr::log(apply_modal_equivalences(e)),
        TLExpr::Sin(e) => TLExpr::sin(apply_modal_equivalences(e)),
        TLExpr::Cos(e) => TLExpr::cos(apply_modal_equivalences(e)),
        TLExpr::Tan(e) => TLExpr::tan(apply_modal_equivalences(e)),

        // Temporal operators (just recurse)
        TLExpr::Next(e) => TLExpr::next(apply_modal_equivalences(e)),
        TLExpr::Eventually(e) => TLExpr::eventually(apply_modal_equivalences(e)),
        TLExpr::Always(e) => TLExpr::always(apply_modal_equivalences(e)),
        TLExpr::Until { before, after } => TLExpr::until(
            apply_modal_equivalences(before),
            apply_modal_equivalences(after),
        ),
        TLExpr::Release { released, releaser } => TLExpr::release(
            apply_modal_equivalences(released),
            apply_modal_equivalences(releaser),
        ),
        TLExpr::WeakUntil { before, after } => TLExpr::weak_until(
            apply_modal_equivalences(before),
            apply_modal_equivalences(after),
        ),
        TLExpr::StrongRelease { released, releaser } => TLExpr::strong_release(
            apply_modal_equivalences(released),
            apply_modal_equivalences(releaser),
        ),

        // Quantifiers
        TLExpr::Exists { var, domain, body } => {
            TLExpr::exists(var.clone(), domain.clone(), apply_modal_equivalences(body))
        }
        TLExpr::ForAll { var, domain, body } => {
            TLExpr::forall(var.clone(), domain.clone(), apply_modal_equivalences(body))
        }
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::soft_exists(
            var.clone(),
            domain.clone(),
            apply_modal_equivalences(body),
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
            apply_modal_equivalences(body),
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
                    apply_modal_equivalences(body),
                    group_vars.clone(),
                )
            } else {
                TLExpr::aggregate(
                    op.clone(),
                    var.clone(),
                    domain.clone(),
                    apply_modal_equivalences(body),
                )
            }
        }

        // Control flow
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::if_then_else(
            apply_modal_equivalences(condition),
            apply_modal_equivalences(then_branch),
            apply_modal_equivalences(else_branch),
        ),
        TLExpr::Let { var, value, body } => TLExpr::let_binding(
            var.clone(),
            apply_modal_equivalences(value),
            apply_modal_equivalences(body),
        ),

        // Fuzzy logic
        TLExpr::TNorm { kind, left, right } => TLExpr::tnorm(
            *kind,
            apply_modal_equivalences(left),
            apply_modal_equivalences(right),
        ),
        TLExpr::TCoNorm { kind, left, right } => TLExpr::tconorm(
            *kind,
            apply_modal_equivalences(left),
            apply_modal_equivalences(right),
        ),
        TLExpr::FuzzyNot { kind, expr } => TLExpr::fuzzy_not(*kind, apply_modal_equivalences(expr)),
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::fuzzy_imply(
            *kind,
            apply_modal_equivalences(premise),
            apply_modal_equivalences(conclusion),
        ),

        // Probabilistic
        TLExpr::WeightedRule { weight, rule } => {
            TLExpr::weighted_rule(*weight, apply_modal_equivalences(rule))
        }
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::probabilistic_choice(
            alternatives
                .iter()
                .map(|(p, e)| (*p, apply_modal_equivalences(e)))
                .collect(),
        ),

        // Beta.1 enhancements - just recurse, no modal-specific optimizations
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => TLExpr::lambda(
            var.clone(),
            var_type.clone(),
            apply_modal_equivalences(body),
        ),
        TLExpr::Apply { function, argument } => TLExpr::apply(
            apply_modal_equivalences(function),
            apply_modal_equivalences(argument),
        ),
        TLExpr::SetMembership { element, set } => TLExpr::set_membership(
            apply_modal_equivalences(element),
            apply_modal_equivalences(set),
        ),
        TLExpr::SetUnion { left, right } => TLExpr::set_union(
            apply_modal_equivalences(left),
            apply_modal_equivalences(right),
        ),
        TLExpr::SetIntersection { left, right } => TLExpr::set_intersection(
            apply_modal_equivalences(left),
            apply_modal_equivalences(right),
        ),
        TLExpr::SetDifference { left, right } => TLExpr::set_difference(
            apply_modal_equivalences(left),
            apply_modal_equivalences(right),
        ),
        TLExpr::SetCardinality { set } => TLExpr::set_cardinality(apply_modal_equivalences(set)),
        TLExpr::EmptySet => expr.clone(),
        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => TLExpr::set_comprehension(
            var.clone(),
            domain.clone(),
            apply_modal_equivalences(condition),
        ),
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => TLExpr::counting_exists(
            var.clone(),
            domain.clone(),
            apply_modal_equivalences(body),
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
            apply_modal_equivalences(body),
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
            apply_modal_equivalences(body),
            *count,
        ),
        TLExpr::Majority { var, domain, body } => {
            TLExpr::majority(var.clone(), domain.clone(), apply_modal_equivalences(body))
        }
        TLExpr::LeastFixpoint { var, body } => {
            TLExpr::least_fixpoint(var.clone(), apply_modal_equivalences(body))
        }
        TLExpr::GreatestFixpoint { var, body } => {
            TLExpr::greatest_fixpoint(var.clone(), apply_modal_equivalences(body))
        }
        TLExpr::Nominal { .. } => expr.clone(),
        TLExpr::At { nominal, formula } => {
            TLExpr::at(nominal.clone(), apply_modal_equivalences(formula))
        }
        TLExpr::Somewhere { formula } => TLExpr::somewhere(apply_modal_equivalences(formula)),
        TLExpr::Everywhere { formula } => TLExpr::everywhere(apply_modal_equivalences(formula)),
        TLExpr::AllDifferent { .. } => expr.clone(),
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => TLExpr::global_cardinality(
            variables.clone(),
            values.iter().map(apply_modal_equivalences).collect(),
            min_occurrences.clone(),
            max_occurrences.clone(),
        ),
        TLExpr::Abducible { .. } => expr.clone(),
        TLExpr::Explain { formula } => TLExpr::explain(apply_modal_equivalences(formula)),

        // Leaves
        TLExpr::Pred { .. } | TLExpr::Constant(_) => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_modal_duality_diamond_to_box() {
        // ¬□¬P → ◇P
        let expr = TLExpr::negate(TLExpr::modal_box(TLExpr::negate(TLExpr::pred(
            "P",
            vec![Term::var("x")],
        ))));

        let result = apply_modal_equivalences(&expr);
        assert!(matches!(result, TLExpr::Diamond(_)));
    }

    #[test]
    fn test_modal_duality_box_to_diamond() {
        // ¬◇¬P → □P
        let expr = TLExpr::negate(TLExpr::modal_diamond(TLExpr::negate(TLExpr::pred(
            "P",
            vec![Term::var("x")],
        ))));

        let result = apply_modal_equivalences(&expr);
        assert!(matches!(result, TLExpr::Box(_)));
    }

    #[test]
    fn test_modal_idempotence_box() {
        // □□P → □P
        let expr = TLExpr::modal_box(TLExpr::modal_box(TLExpr::pred("P", vec![Term::var("x")])));

        let result = apply_modal_equivalences(&expr);
        // Should be simplified to a single Box
        if let TLExpr::Box(inner) = result {
            assert!(matches!(*inner, TLExpr::Pred { .. }));
        } else {
            panic!("Expected Box, got {:?}", result);
        }
    }

    #[test]
    fn test_modal_idempotence_diamond() {
        // ◇◇P → ◇P
        let expr = TLExpr::modal_diamond(TLExpr::modal_diamond(TLExpr::pred(
            "P",
            vec![Term::var("x")],
        )));

        let result = apply_modal_equivalences(&expr);
        // Should be simplified to a single Diamond
        if let TLExpr::Diamond(inner) = result {
            assert!(matches!(*inner, TLExpr::Pred { .. }));
        } else {
            panic!("Expected Diamond, got {:?}", result);
        }
    }

    #[test]
    fn test_modal_distribution_box_and() {
        // □(P ∧ Q) → □P ∧ □Q
        let expr = TLExpr::modal_box(TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("x")]),
        ));

        let result = apply_modal_equivalences(&expr);
        // Should be distributed
        if let TLExpr::And(l, r) = result {
            assert!(matches!(*l, TLExpr::Box(_)));
            assert!(matches!(*r, TLExpr::Box(_)));
        } else {
            panic!("Expected And of two Boxes, got {:?}", result);
        }
    }

    #[test]
    fn test_modal_distribution_diamond_or() {
        // ◇(P ∨ Q) → ◇P ∨ ◇Q
        let expr = TLExpr::modal_diamond(TLExpr::or(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("x")]),
        ));

        let result = apply_modal_equivalences(&expr);
        // Should be distributed
        if let TLExpr::Or(l, r) = result {
            assert!(matches!(*l, TLExpr::Diamond(_)));
            assert!(matches!(*r, TLExpr::Diamond(_)));
        } else {
            panic!("Expected Or of two Diamonds, got {:?}", result);
        }
    }
}
