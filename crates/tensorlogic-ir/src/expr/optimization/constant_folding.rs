//! Constant folding: evaluate constant expressions at compile time.
//!
//! This module implements constant folding optimizations that evaluate
//! expressions with constant operands at compile time, reducing runtime overhead.

use crate::expr::TLExpr;

/// Constant folding: evaluate constant expressions at compile time
pub fn constant_fold(expr: &TLExpr) -> TLExpr {
    match expr {
        // Binary arithmetic operations on constants
        TLExpr::Add(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv + rv);
            }
            TLExpr::Add(Box::new(left), Box::new(right))
        }
        TLExpr::Sub(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv - rv);
            }
            TLExpr::Sub(Box::new(left), Box::new(right))
        }
        TLExpr::Mul(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv * rv);
            }
            TLExpr::Mul(Box::new(left), Box::new(right))
        }
        TLExpr::Div(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                if *rv != 0.0 {
                    return TLExpr::Constant(lv / rv);
                }
            }
            TLExpr::Div(Box::new(left), Box::new(right))
        }
        TLExpr::Pow(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv.powf(*rv));
            }
            TLExpr::Pow(Box::new(left), Box::new(right))
        }
        TLExpr::Mod(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv % rv);
            }
            TLExpr::Mod(Box::new(left), Box::new(right))
        }
        TLExpr::Min(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv.min(*rv));
            }
            TLExpr::Min(Box::new(left), Box::new(right))
        }
        TLExpr::Max(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(lv.max(*rv));
            }
            TLExpr::Max(Box::new(left), Box::new(right))
        }

        // Unary mathematical operations on constants
        TLExpr::Abs(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.abs());
            }
            TLExpr::Abs(Box::new(inner))
        }
        TLExpr::Floor(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.floor());
            }
            TLExpr::Floor(Box::new(inner))
        }
        TLExpr::Ceil(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.ceil());
            }
            TLExpr::Ceil(Box::new(inner))
        }
        TLExpr::Round(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.round());
            }
            TLExpr::Round(Box::new(inner))
        }
        TLExpr::Sqrt(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                if *v >= 0.0 {
                    return TLExpr::Constant(v.sqrt());
                }
            }
            TLExpr::Sqrt(Box::new(inner))
        }
        TLExpr::Exp(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.exp());
            }
            TLExpr::Exp(Box::new(inner))
        }
        TLExpr::Log(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                if *v > 0.0 {
                    return TLExpr::Constant(v.ln());
                }
            }
            TLExpr::Log(Box::new(inner))
        }
        TLExpr::Sin(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.sin());
            }
            TLExpr::Sin(Box::new(inner))
        }
        TLExpr::Cos(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.cos());
            }
            TLExpr::Cos(Box::new(inner))
        }
        TLExpr::Tan(e) => {
            let inner = constant_fold(e);
            if let TLExpr::Constant(v) = &inner {
                return TLExpr::Constant(v.tan());
            }
            TLExpr::Tan(Box::new(inner))
        }

        // Comparison operations on constants
        TLExpr::Eq(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(if (lv - rv).abs() < f64::EPSILON {
                    1.0
                } else {
                    0.0
                });
            }
            TLExpr::Eq(Box::new(left), Box::new(right))
        }
        TLExpr::Lt(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(if lv < rv { 1.0 } else { 0.0 });
            }
            TLExpr::Lt(Box::new(left), Box::new(right))
        }
        TLExpr::Gt(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(if lv > rv { 1.0 } else { 0.0 });
            }
            TLExpr::Gt(Box::new(left), Box::new(right))
        }
        TLExpr::Lte(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(if lv <= rv { 1.0 } else { 0.0 });
            }
            TLExpr::Lte(Box::new(left), Box::new(right))
        }
        TLExpr::Gte(l, r) => {
            let left = constant_fold(l);
            let right = constant_fold(r);
            if let (TLExpr::Constant(lv), TLExpr::Constant(rv)) = (&left, &right) {
                return TLExpr::Constant(if lv >= rv { 1.0 } else { 0.0 });
            }
            TLExpr::Gte(Box::new(left), Box::new(right))
        }

        // Logical connectives - recursively fold subexpressions
        TLExpr::And(l, r) => TLExpr::And(Box::new(constant_fold(l)), Box::new(constant_fold(r))),
        TLExpr::Or(l, r) => TLExpr::Or(Box::new(constant_fold(l)), Box::new(constant_fold(r))),
        TLExpr::Not(e) => TLExpr::Not(Box::new(constant_fold(e))),
        TLExpr::Imply(l, r) => {
            TLExpr::Imply(Box::new(constant_fold(l)), Box::new(constant_fold(r)))
        }

        // Quantifiers - fold the body
        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(constant_fold(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(constant_fold(body)),
        },

        // Score operator
        TLExpr::Score(e) => TLExpr::Score(Box::new(constant_fold(e))),

        // Aggregation
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
            body: Box::new(constant_fold(body)),
            group_by: group_by.clone(),
        },

        // Modal logic operators
        TLExpr::Box(e) => TLExpr::Box(Box::new(constant_fold(e))),
        TLExpr::Diamond(e) => TLExpr::Diamond(Box::new(constant_fold(e))),

        // Temporal logic operators
        TLExpr::Next(e) => TLExpr::Next(Box::new(constant_fold(e))),
        TLExpr::Eventually(e) => TLExpr::Eventually(Box::new(constant_fold(e))),
        TLExpr::Always(e) => TLExpr::Always(Box::new(constant_fold(e))),
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(constant_fold(before)),
            after: Box::new(constant_fold(after)),
        },

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
            kind: *kind,
            left: Box::new(constant_fold(left)),
            right: Box::new(constant_fold(right)),
        },
        TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
            kind: *kind,
            left: Box::new(constant_fold(left)),
            right: Box::new(constant_fold(right)),
        },
        TLExpr::FuzzyNot { kind, expr } => TLExpr::FuzzyNot {
            kind: *kind,
            expr: Box::new(constant_fold(expr)),
        },
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::FuzzyImplication {
            kind: *kind,
            premise: Box::new(constant_fold(premise)),
            conclusion: Box::new(constant_fold(conclusion)),
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
            body: Box::new(constant_fold(body)),
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
            body: Box::new(constant_fold(body)),
            temperature: *temperature,
        },
        TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
            weight: *weight,
            rule: Box::new(constant_fold(rule)),
        },
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::ProbabilisticChoice {
            alternatives: alternatives
                .iter()
                .map(|(p, e)| (*p, constant_fold(e)))
                .collect(),
        },

        // Extended temporal logic
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(constant_fold(released)),
            releaser: Box::new(constant_fold(releaser)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(constant_fold(before)),
            after: Box::new(constant_fold(after)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(constant_fold(released)),
            releaser: Box::new(constant_fold(releaser)),
        },

        // Conditional expressions
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::IfThenElse {
            condition: Box::new(constant_fold(condition)),
            then_branch: Box::new(constant_fold(then_branch)),
            else_branch: Box::new(constant_fold(else_branch)),
        },
        TLExpr::Let { var, value, body } => TLExpr::Let {
            var: var.clone(),
            value: Box::new(constant_fold(value)),
            body: Box::new(constant_fold(body)),
        },

        // Alpha.3 enhancements: recurse into subexpressions (minimal optimization for now)
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => TLExpr::lambda(var.clone(), var_type.clone(), constant_fold(body)),
        TLExpr::Apply { function, argument } => {
            TLExpr::apply(constant_fold(function), constant_fold(argument))
        }
        TLExpr::SetMembership { element, set } => {
            TLExpr::set_membership(constant_fold(element), constant_fold(set))
        }
        TLExpr::SetUnion { left, right } => {
            TLExpr::set_union(constant_fold(left), constant_fold(right))
        }
        TLExpr::SetIntersection { left, right } => {
            TLExpr::set_intersection(constant_fold(left), constant_fold(right))
        }
        TLExpr::SetDifference { left, right } => {
            TLExpr::set_difference(constant_fold(left), constant_fold(right))
        }
        TLExpr::SetCardinality { set } => TLExpr::set_cardinality(constant_fold(set)),
        TLExpr::EmptySet => expr.clone(),
        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => TLExpr::set_comprehension(var.clone(), domain.clone(), constant_fold(condition)),
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => TLExpr::counting_exists(var.clone(), domain.clone(), constant_fold(body), *min_count),
        TLExpr::CountingForAll {
            var,
            domain,
            body,
            min_count,
        } => TLExpr::counting_forall(var.clone(), domain.clone(), constant_fold(body), *min_count),
        TLExpr::ExactCount {
            var,
            domain,
            body,
            count,
        } => TLExpr::exact_count(var.clone(), domain.clone(), constant_fold(body), *count),
        TLExpr::Majority { var, domain, body } => {
            TLExpr::majority(var.clone(), domain.clone(), constant_fold(body))
        }
        TLExpr::LeastFixpoint { var, body } => {
            TLExpr::least_fixpoint(var.clone(), constant_fold(body))
        }
        TLExpr::GreatestFixpoint { var, body } => {
            TLExpr::greatest_fixpoint(var.clone(), constant_fold(body))
        }
        TLExpr::Nominal { .. } => expr.clone(),
        TLExpr::At { nominal, formula } => TLExpr::at(nominal.clone(), constant_fold(formula)),
        TLExpr::Somewhere { formula } => TLExpr::somewhere(constant_fold(formula)),
        TLExpr::Everywhere { formula } => TLExpr::everywhere(constant_fold(formula)),
        TLExpr::AllDifferent { .. } => expr.clone(),
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => TLExpr::global_cardinality(
            variables.clone(),
            values.iter().map(constant_fold).collect(),
            min_occurrences.clone(),
            max_occurrences.clone(),
        ),
        TLExpr::Abducible { .. } => expr.clone(),
        TLExpr::Explain { formula } => TLExpr::explain(constant_fold(formula)),

        // Leaves - no folding needed
        TLExpr::Pred { .. } | TLExpr::Constant(_) => expr.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_fold_addition() {
        let expr = TLExpr::Add(
            Box::new(TLExpr::Constant(2.0)),
            Box::new(TLExpr::Constant(3.0)),
        );
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(5.0));
    }

    #[test]
    fn test_constant_fold_multiplication() {
        let expr = TLExpr::Mul(
            Box::new(TLExpr::Constant(4.0)),
            Box::new(TLExpr::Constant(5.0)),
        );
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(20.0));
    }

    #[test]
    fn test_constant_fold_nested() {
        // (2 + 3) * 4 = 20
        let expr = TLExpr::Mul(
            Box::new(TLExpr::Add(
                Box::new(TLExpr::Constant(2.0)),
                Box::new(TLExpr::Constant(3.0)),
            )),
            Box::new(TLExpr::Constant(4.0)),
        );
        let folded = constant_fold(&expr);
        assert_eq!(folded, TLExpr::Constant(20.0));
    }

    #[test]
    fn test_constant_fold_division_zero() {
        let expr = TLExpr::Div(
            Box::new(TLExpr::Constant(5.0)),
            Box::new(TLExpr::Constant(0.0)),
        );
        let folded = constant_fold(&expr);
        // Should not fold division by zero
        matches!(folded, TLExpr::Div(_, _));
    }

    #[test]
    fn test_constant_fold_sqrt_negative() {
        let expr = TLExpr::Sqrt(Box::new(TLExpr::Constant(-4.0)));
        let folded = constant_fold(&expr);
        // Should not fold sqrt of negative
        matches!(folded, TLExpr::Sqrt(_));
    }
}
