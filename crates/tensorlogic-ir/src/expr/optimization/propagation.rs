//! Constant propagation for Let bindings.
//!
//! This module implements constant propagation, which substitutes variables
//! bound in Let expressions with their values throughout the expression tree.

use super::substitution::substitute;
use crate::expr::TLExpr;

pub fn propagate_constants(expr: &TLExpr) -> TLExpr {
    match expr {
        // If the Let binding value is a constant, substitute it into the body
        TLExpr::Let { var, value, body } => {
            let optimized_value = propagate_constants(value);
            let optimized_body = propagate_constants(body);

            // If the value is constant, substitute it
            if matches!(optimized_value, TLExpr::Constant(_)) {
                substitute(&optimized_body, var, &optimized_value)
            } else {
                TLExpr::Let {
                    var: var.clone(),
                    value: Box::new(optimized_value),
                    body: Box::new(optimized_body),
                }
            }
        }

        // Recursively propagate in other expressions
        TLExpr::And(l, r) => TLExpr::And(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Or(l, r) => TLExpr::Or(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Imply(l, r) => TLExpr::Imply(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Add(l, r) => TLExpr::Add(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Sub(l, r) => TLExpr::Sub(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Mul(l, r) => TLExpr::Mul(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Div(l, r) => TLExpr::Div(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Pow(l, r) => TLExpr::Pow(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Mod(l, r) => TLExpr::Mod(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Min(l, r) => TLExpr::Min(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Max(l, r) => TLExpr::Max(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Eq(l, r) => TLExpr::Eq(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Lt(l, r) => TLExpr::Lt(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Gt(l, r) => TLExpr::Gt(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Lte(l, r) => TLExpr::Lte(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Gte(l, r) => TLExpr::Gte(
            Box::new(propagate_constants(l)),
            Box::new(propagate_constants(r)),
        ),
        TLExpr::Not(e) => TLExpr::Not(Box::new(propagate_constants(e))),
        TLExpr::Score(e) => TLExpr::Score(Box::new(propagate_constants(e))),
        TLExpr::Abs(e) => TLExpr::Abs(Box::new(propagate_constants(e))),
        TLExpr::Floor(e) => TLExpr::Floor(Box::new(propagate_constants(e))),
        TLExpr::Ceil(e) => TLExpr::Ceil(Box::new(propagate_constants(e))),
        TLExpr::Round(e) => TLExpr::Round(Box::new(propagate_constants(e))),
        TLExpr::Sqrt(e) => TLExpr::Sqrt(Box::new(propagate_constants(e))),
        TLExpr::Exp(e) => TLExpr::Exp(Box::new(propagate_constants(e))),
        TLExpr::Log(e) => TLExpr::Log(Box::new(propagate_constants(e))),
        TLExpr::Sin(e) => TLExpr::Sin(Box::new(propagate_constants(e))),
        TLExpr::Cos(e) => TLExpr::Cos(Box::new(propagate_constants(e))),
        TLExpr::Tan(e) => TLExpr::Tan(Box::new(propagate_constants(e))),
        TLExpr::Box(e) => TLExpr::Box(Box::new(propagate_constants(e))),
        TLExpr::Diamond(e) => TLExpr::Diamond(Box::new(propagate_constants(e))),
        TLExpr::Next(e) => TLExpr::Next(Box::new(propagate_constants(e))),
        TLExpr::Eventually(e) => TLExpr::Eventually(Box::new(propagate_constants(e))),
        TLExpr::Always(e) => TLExpr::Always(Box::new(propagate_constants(e))),
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(propagate_constants(before)),
            after: Box::new(propagate_constants(after)),
        },

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
            kind: *kind,
            left: Box::new(propagate_constants(left)),
            right: Box::new(propagate_constants(right)),
        },
        TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
            kind: *kind,
            left: Box::new(propagate_constants(left)),
            right: Box::new(propagate_constants(right)),
        },
        TLExpr::FuzzyNot { kind, expr } => TLExpr::FuzzyNot {
            kind: *kind,
            expr: Box::new(propagate_constants(expr)),
        },
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::FuzzyImplication {
            kind: *kind,
            premise: Box::new(propagate_constants(premise)),
            conclusion: Box::new(propagate_constants(conclusion)),
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
            body: Box::new(propagate_constants(body)),
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
            body: Box::new(propagate_constants(body)),
            temperature: *temperature,
        },
        TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
            weight: *weight,
            rule: Box::new(propagate_constants(rule)),
        },
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::ProbabilisticChoice {
            alternatives: alternatives
                .iter()
                .map(|(p, e)| (*p, propagate_constants(e)))
                .collect(),
        },

        // Extended temporal logic
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(propagate_constants(released)),
            releaser: Box::new(propagate_constants(releaser)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(propagate_constants(before)),
            after: Box::new(propagate_constants(after)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(propagate_constants(released)),
            releaser: Box::new(propagate_constants(releaser)),
        },

        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(propagate_constants(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(propagate_constants(body)),
        },
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
            body: Box::new(propagate_constants(body)),
            group_by: group_by.clone(),
        },
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::IfThenElse {
            condition: Box::new(propagate_constants(condition)),
            then_branch: Box::new(propagate_constants(then_branch)),
            else_branch: Box::new(propagate_constants(else_branch)),
        },

        // Alpha.3 enhancements
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => TLExpr::lambda(var.clone(), var_type.clone(), propagate_constants(body)),
        TLExpr::Apply { function, argument } => {
            TLExpr::apply(propagate_constants(function), propagate_constants(argument))
        }
        TLExpr::SetMembership { element, set } => {
            TLExpr::set_membership(propagate_constants(element), propagate_constants(set))
        }
        TLExpr::SetUnion { left, right } => {
            TLExpr::set_union(propagate_constants(left), propagate_constants(right))
        }
        TLExpr::SetIntersection { left, right } => {
            TLExpr::set_intersection(propagate_constants(left), propagate_constants(right))
        }
        TLExpr::SetDifference { left, right } => {
            TLExpr::set_difference(propagate_constants(left), propagate_constants(right))
        }
        TLExpr::SetCardinality { set } => TLExpr::set_cardinality(propagate_constants(set)),
        TLExpr::EmptySet => expr.clone(),
        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => TLExpr::set_comprehension(var.clone(), domain.clone(), propagate_constants(condition)),
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => TLExpr::counting_exists(
            var.clone(),
            domain.clone(),
            propagate_constants(body),
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
            propagate_constants(body),
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
            propagate_constants(body),
            *count,
        ),
        TLExpr::Majority { var, domain, body } => {
            TLExpr::majority(var.clone(), domain.clone(), propagate_constants(body))
        }
        TLExpr::LeastFixpoint { var, body } => {
            TLExpr::least_fixpoint(var.clone(), propagate_constants(body))
        }
        TLExpr::GreatestFixpoint { var, body } => {
            TLExpr::greatest_fixpoint(var.clone(), propagate_constants(body))
        }
        TLExpr::Nominal { .. } => expr.clone(),
        TLExpr::At { nominal, formula } => {
            TLExpr::at(nominal.clone(), propagate_constants(formula))
        }
        TLExpr::Somewhere { formula } => TLExpr::somewhere(propagate_constants(formula)),
        TLExpr::Everywhere { formula } => TLExpr::everywhere(propagate_constants(formula)),
        TLExpr::AllDifferent { .. } => expr.clone(),
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => TLExpr::global_cardinality(
            variables.clone(),
            values.iter().map(propagate_constants).collect(),
            min_occurrences.clone(),
            max_occurrences.clone(),
        ),
        TLExpr::Abducible { .. } => expr.clone(),
        TLExpr::Explain { formula } => TLExpr::explain(propagate_constants(formula)),

        TLExpr::Pred { .. } | TLExpr::Constant(_) => expr.clone(),
    }
}
