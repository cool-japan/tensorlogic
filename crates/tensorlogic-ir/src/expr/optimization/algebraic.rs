//! Algebraic simplification rules for logical and arithmetic expressions.
//!
//! This module implements algebraic identities and simplification rules that
//! transform expressions into simpler equivalent forms without changing semantics.

use crate::expr::TLExpr;

pub fn algebraic_simplify(expr: &TLExpr) -> TLExpr {
    match expr {
        // Addition identities
        TLExpr::Add(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x + 0 = x
            if let TLExpr::Constant(0.0) = right {
                return left;
            }
            // 0 + x = x
            if let TLExpr::Constant(0.0) = left {
                return right;
            }

            TLExpr::Add(Box::new(left), Box::new(right))
        }

        // Subtraction identities
        TLExpr::Sub(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x - 0 = x
            if let TLExpr::Constant(0.0) = right {
                return left;
            }
            // x - x = 0 (simplified form comparison)
            if left == right {
                return TLExpr::Constant(0.0);
            }

            TLExpr::Sub(Box::new(left), Box::new(right))
        }

        // Multiplication identities
        TLExpr::Mul(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x * 0 = 0
            if let TLExpr::Constant(0.0) = right {
                return TLExpr::Constant(0.0);
            }
            if let TLExpr::Constant(0.0) = left {
                return TLExpr::Constant(0.0);
            }

            // x * 1 = x
            if let TLExpr::Constant(1.0) = right {
                return left;
            }
            // 1 * x = x
            if let TLExpr::Constant(1.0) = left {
                return right;
            }

            TLExpr::Mul(Box::new(left), Box::new(right))
        }

        // Division identities
        TLExpr::Div(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x / 1 = x
            if let TLExpr::Constant(1.0) = right {
                return left;
            }

            // 0 / x = 0 (assuming x != 0)
            if let TLExpr::Constant(0.0) = left {
                if let TLExpr::Constant(rv) = right {
                    if rv != 0.0 {
                        return TLExpr::Constant(0.0);
                    }
                }
            }

            // x / x = 1 (assuming x != 0)
            // Only apply for constants to avoid division by zero issues
            if left == right {
                if let TLExpr::Constant(v) = left {
                    if v != 0.0 {
                        return TLExpr::Constant(1.0);
                    }
                }
            }

            TLExpr::Div(Box::new(left), Box::new(right))
        }

        // Power identities
        TLExpr::Pow(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x ^ 0 = 1
            if let TLExpr::Constant(0.0) = right {
                return TLExpr::Constant(1.0);
            }
            // x ^ 1 = x
            if let TLExpr::Constant(1.0) = right {
                return left;
            }
            // 0 ^ x = 0 (for x > 0)
            if let TLExpr::Constant(0.0) = left {
                if let TLExpr::Constant(rv) = right {
                    if rv > 0.0 {
                        return TLExpr::Constant(0.0);
                    }
                }
            }
            // 1 ^ x = 1
            if let TLExpr::Constant(1.0) = left {
                return TLExpr::Constant(1.0);
            }

            TLExpr::Pow(Box::new(left), Box::new(right))
        }

        // Double negation: NOT(NOT(x)) = x
        TLExpr::Not(e) => {
            let inner = algebraic_simplify(e);
            if let TLExpr::Not(inner_inner) = &inner {
                return *inner_inner.clone();
            }
            TLExpr::Not(Box::new(inner))
        }

        // Recursively simplify other operations
        TLExpr::Mod(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);
            TLExpr::Mod(Box::new(left), Box::new(right))
        }
        TLExpr::Min(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);
            TLExpr::Min(Box::new(left), Box::new(right))
        }
        TLExpr::Max(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);
            TLExpr::Max(Box::new(left), Box::new(right))
        }
        TLExpr::Abs(e) => TLExpr::Abs(Box::new(algebraic_simplify(e))),
        TLExpr::Floor(e) => TLExpr::Floor(Box::new(algebraic_simplify(e))),
        TLExpr::Ceil(e) => TLExpr::Ceil(Box::new(algebraic_simplify(e))),
        TLExpr::Round(e) => TLExpr::Round(Box::new(algebraic_simplify(e))),
        TLExpr::Sqrt(e) => TLExpr::Sqrt(Box::new(algebraic_simplify(e))),
        // Modal logic simplifications
        TLExpr::Box(e) => {
            let inner = algebraic_simplify(e);

            // □(TRUE) = TRUE, □(FALSE) = FALSE
            if let TLExpr::Constant(v) = inner {
                return TLExpr::Constant(v);
            }

            TLExpr::Box(Box::new(inner))
        }
        TLExpr::Diamond(e) => {
            let inner = algebraic_simplify(e);

            // ◇(TRUE) = TRUE, ◇(FALSE) = FALSE
            if let TLExpr::Constant(v) = inner {
                return TLExpr::Constant(v);
            }

            TLExpr::Diamond(Box::new(inner))
        }

        // Temporal logic simplifications
        TLExpr::Next(e) => {
            let inner = algebraic_simplify(e);

            // X(TRUE) = TRUE, X(FALSE) = FALSE
            if let TLExpr::Constant(v) = inner {
                return TLExpr::Constant(v);
            }

            TLExpr::Next(Box::new(inner))
        }
        TLExpr::Eventually(e) => {
            let inner = algebraic_simplify(e);

            // F(TRUE) = TRUE, F(FALSE) = FALSE
            if let TLExpr::Constant(v) = inner {
                return TLExpr::Constant(v);
            }

            // Idempotence: F(F(P)) = F(P)
            if let TLExpr::Eventually(inner_inner) = &inner {
                return TLExpr::Eventually(inner_inner.clone());
            }

            TLExpr::Eventually(Box::new(inner))
        }
        TLExpr::Always(e) => {
            let inner = algebraic_simplify(e);

            // G(TRUE) = TRUE, G(FALSE) = FALSE
            if let TLExpr::Constant(v) = inner {
                return TLExpr::Constant(v);
            }

            // Idempotence: G(G(P)) = G(P)
            if let TLExpr::Always(inner_inner) = &inner {
                return TLExpr::Always(inner_inner.clone());
            }

            TLExpr::Always(Box::new(inner))
        }
        TLExpr::Until { before, after } => {
            let before_simplified = algebraic_simplify(before);
            let after_simplified = algebraic_simplify(after);

            // P U TRUE = TRUE (after becomes immediately true)
            if let TLExpr::Constant(1.0) = after_simplified {
                return TLExpr::Constant(1.0);
            }

            // FALSE U P = F(P) (before is never true, so we just wait for after)
            if let TLExpr::Constant(0.0) = before_simplified {
                return TLExpr::Eventually(Box::new(after_simplified));
            }

            TLExpr::Until {
                before: Box::new(before_simplified),
                after: Box::new(after_simplified),
            }
        }

        // Fuzzy logic operators - pass through with recursive simplification
        TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
            kind: *kind,
            left: Box::new(algebraic_simplify(left)),
            right: Box::new(algebraic_simplify(right)),
        },
        TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
            kind: *kind,
            left: Box::new(algebraic_simplify(left)),
            right: Box::new(algebraic_simplify(right)),
        },
        TLExpr::FuzzyNot { kind, expr } => TLExpr::FuzzyNot {
            kind: *kind,
            expr: Box::new(algebraic_simplify(expr)),
        },
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::FuzzyImplication {
            kind: *kind,
            premise: Box::new(algebraic_simplify(premise)),
            conclusion: Box::new(algebraic_simplify(conclusion)),
        },

        // Probabilistic operators - pass through
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::SoftExists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(algebraic_simplify(body)),
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
            body: Box::new(algebraic_simplify(body)),
            temperature: *temperature,
        },
        TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
            weight: *weight,
            rule: Box::new(algebraic_simplify(rule)),
        },
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::ProbabilisticChoice {
            alternatives: alternatives
                .iter()
                .map(|(p, e)| (*p, algebraic_simplify(e)))
                .collect(),
        },

        // Extended temporal logic - pass through
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(algebraic_simplify(released)),
            releaser: Box::new(algebraic_simplify(releaser)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(algebraic_simplify(before)),
            after: Box::new(algebraic_simplify(after)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(algebraic_simplify(released)),
            releaser: Box::new(algebraic_simplify(releaser)),
        },

        TLExpr::Exp(e) => TLExpr::Exp(Box::new(algebraic_simplify(e))),
        TLExpr::Log(e) => TLExpr::Log(Box::new(algebraic_simplify(e))),
        TLExpr::Sin(e) => TLExpr::Sin(Box::new(algebraic_simplify(e))),
        TLExpr::Cos(e) => TLExpr::Cos(Box::new(algebraic_simplify(e))),
        TLExpr::Tan(e) => TLExpr::Tan(Box::new(algebraic_simplify(e))),
        // EQ simplifications
        TLExpr::Eq(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x = x → TRUE
            if left == right {
                return TLExpr::Constant(1.0);
            }

            TLExpr::Eq(Box::new(left), Box::new(right))
        }

        // LT simplifications
        TLExpr::Lt(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x < x → FALSE
            if left == right {
                return TLExpr::Constant(0.0);
            }

            TLExpr::Lt(Box::new(left), Box::new(right))
        }

        // GT simplifications
        TLExpr::Gt(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x > x → FALSE
            if left == right {
                return TLExpr::Constant(0.0);
            }

            TLExpr::Gt(Box::new(left), Box::new(right))
        }

        // LTE simplifications
        TLExpr::Lte(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x <= x → TRUE
            if left == right {
                return TLExpr::Constant(1.0);
            }

            TLExpr::Lte(Box::new(left), Box::new(right))
        }

        // GTE simplifications
        TLExpr::Gte(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // x >= x → TRUE
            if left == right {
                return TLExpr::Constant(1.0);
            }

            TLExpr::Gte(Box::new(left), Box::new(right))
        }
        // AND logical laws
        TLExpr::And(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // Idempotence: A ∧ A = A
            if left == right {
                return left;
            }

            // Identity: A ∧ TRUE = A, TRUE ∧ A = A
            if let TLExpr::Constant(1.0) = right {
                return left;
            }
            if let TLExpr::Constant(1.0) = left {
                return right;
            }

            // Annihilation: A ∧ FALSE = FALSE, FALSE ∧ A = FALSE
            if let TLExpr::Constant(0.0) = right {
                return TLExpr::Constant(0.0);
            }
            if let TLExpr::Constant(0.0) = left {
                return TLExpr::Constant(0.0);
            }

            // Complement: A ∧ ¬A = FALSE
            if let TLExpr::Not(inner) = &right {
                if **inner == left {
                    return TLExpr::Constant(0.0);
                }
            }
            if let TLExpr::Not(inner) = &left {
                if **inner == right {
                    return TLExpr::Constant(0.0);
                }
            }

            // Absorption: A ∧ (A ∨ B) = A
            if let TLExpr::Or(or_left, _or_right) = &right {
                if **or_left == left {
                    return left;
                }
            }
            if let TLExpr::Or(or_left, _or_right) = &left {
                if **or_left == right {
                    return right;
                }
            }

            TLExpr::And(Box::new(left), Box::new(right))
        }

        // OR logical laws
        TLExpr::Or(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // Idempotence: A ∨ A = A
            if left == right {
                return left;
            }

            // Annihilation: A ∨ TRUE = TRUE, TRUE ∨ A = TRUE
            if let TLExpr::Constant(1.0) = right {
                return TLExpr::Constant(1.0);
            }
            if let TLExpr::Constant(1.0) = left {
                return TLExpr::Constant(1.0);
            }

            // Identity: A ∨ FALSE = A, FALSE ∨ A = A
            if let TLExpr::Constant(0.0) = right {
                return left;
            }
            if let TLExpr::Constant(0.0) = left {
                return right;
            }

            // Complement: A ∨ ¬A = TRUE
            if let TLExpr::Not(inner) = &right {
                if **inner == left {
                    return TLExpr::Constant(1.0);
                }
            }
            if let TLExpr::Not(inner) = &left {
                if **inner == right {
                    return TLExpr::Constant(1.0);
                }
            }

            // Absorption: A ∨ (A ∧ B) = A
            if let TLExpr::And(and_left, _and_right) = &right {
                if **and_left == left {
                    return left;
                }
            }
            if let TLExpr::And(and_left, _and_right) = &left {
                if **and_left == right {
                    return right;
                }
            }

            TLExpr::Or(Box::new(left), Box::new(right))
        }

        // IMPLY simplifications
        TLExpr::Imply(l, r) => {
            let left = algebraic_simplify(l);
            let right = algebraic_simplify(r);

            // TRUE → P = P
            if let TLExpr::Constant(1.0) = left {
                return right;
            }

            // FALSE → P = TRUE
            if let TLExpr::Constant(0.0) = left {
                return TLExpr::Constant(1.0);
            }

            // P → TRUE = TRUE
            if let TLExpr::Constant(1.0) = right {
                return TLExpr::Constant(1.0);
            }

            // P → FALSE = ¬P
            if let TLExpr::Constant(0.0) = right {
                return TLExpr::negate(left);
            }

            // P → P = TRUE
            if left == right {
                return TLExpr::Constant(1.0);
            }

            TLExpr::Imply(Box::new(left), Box::new(right))
        }
        TLExpr::Score(e) => TLExpr::Score(Box::new(algebraic_simplify(e))),
        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(algebraic_simplify(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(algebraic_simplify(body)),
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
            body: Box::new(algebraic_simplify(body)),
            group_by: group_by.clone(),
        },
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::IfThenElse {
            condition: Box::new(algebraic_simplify(condition)),
            then_branch: Box::new(algebraic_simplify(then_branch)),
            else_branch: Box::new(algebraic_simplify(else_branch)),
        },
        TLExpr::Let { var, value, body } => TLExpr::Let {
            var: var.clone(),
            value: Box::new(algebraic_simplify(value)),
            body: Box::new(algebraic_simplify(body)),
        },

        // Alpha.3 enhancements
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => TLExpr::lambda(var.clone(), var_type.clone(), algebraic_simplify(body)),
        TLExpr::Apply { function, argument } => {
            TLExpr::apply(algebraic_simplify(function), algebraic_simplify(argument))
        }
        TLExpr::SetMembership { element, set } => {
            TLExpr::set_membership(algebraic_simplify(element), algebraic_simplify(set))
        }
        TLExpr::SetUnion { left, right } => {
            TLExpr::set_union(algebraic_simplify(left), algebraic_simplify(right))
        }
        TLExpr::SetIntersection { left, right } => {
            TLExpr::set_intersection(algebraic_simplify(left), algebraic_simplify(right))
        }
        TLExpr::SetDifference { left, right } => {
            TLExpr::set_difference(algebraic_simplify(left), algebraic_simplify(right))
        }
        TLExpr::SetCardinality { set } => TLExpr::set_cardinality(algebraic_simplify(set)),
        TLExpr::EmptySet => expr.clone(),
        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => TLExpr::set_comprehension(var.clone(), domain.clone(), algebraic_simplify(condition)),
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => TLExpr::counting_exists(
            var.clone(),
            domain.clone(),
            algebraic_simplify(body),
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
            algebraic_simplify(body),
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
            algebraic_simplify(body),
            *count,
        ),
        TLExpr::Majority { var, domain, body } => {
            TLExpr::majority(var.clone(), domain.clone(), algebraic_simplify(body))
        }
        TLExpr::LeastFixpoint { var, body } => {
            TLExpr::least_fixpoint(var.clone(), algebraic_simplify(body))
        }
        TLExpr::GreatestFixpoint { var, body } => {
            TLExpr::greatest_fixpoint(var.clone(), algebraic_simplify(body))
        }
        TLExpr::Nominal { .. } => expr.clone(),
        TLExpr::At { nominal, formula } => TLExpr::at(nominal.clone(), algebraic_simplify(formula)),
        TLExpr::Somewhere { formula } => TLExpr::somewhere(algebraic_simplify(formula)),
        TLExpr::Everywhere { formula } => TLExpr::everywhere(algebraic_simplify(formula)),
        TLExpr::AllDifferent { .. } => expr.clone(),
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => TLExpr::global_cardinality(
            variables.clone(),
            values.iter().map(algebraic_simplify).collect(),
            min_occurrences.clone(),
            max_occurrences.clone(),
        ),
        TLExpr::Abducible { .. } => expr.clone(),
        TLExpr::Explain { formula } => TLExpr::explain(algebraic_simplify(formula)),

        // Leaves
        TLExpr::Pred { .. } | TLExpr::Constant(_) => expr.clone(),
    }
}
