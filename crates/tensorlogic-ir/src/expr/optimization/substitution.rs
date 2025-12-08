//! Variable substitution for expression optimization.
//!
//! This module provides utilities for replacing variables with their bound values
//! in expressions, which is essential for optimizing Let bindings and other constructs.

use crate::expr::TLExpr;

/// Substitute a variable with a value in an expression
pub(crate) fn substitute(expr: &TLExpr, var: &str, value: &TLExpr) -> TLExpr {
    match expr {
        // If we find a predicate matching the variable name with no args, substitute
        TLExpr::Pred { name, args } if name == var && args.is_empty() => value.clone(),

        // For predicates with args or different names, keep them
        TLExpr::Pred { .. } => expr.clone(),

        // Recursively substitute in binary operations
        TLExpr::And(l, r) => TLExpr::And(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Or(l, r) => TLExpr::Or(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Imply(l, r) => TLExpr::Imply(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Add(l, r) => TLExpr::Add(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Sub(l, r) => TLExpr::Sub(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Mul(l, r) => TLExpr::Mul(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Div(l, r) => TLExpr::Div(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Pow(l, r) => TLExpr::Pow(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Mod(l, r) => TLExpr::Mod(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Min(l, r) => TLExpr::Min(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Max(l, r) => TLExpr::Max(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Eq(l, r) => TLExpr::Eq(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Lt(l, r) => TLExpr::Lt(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Gt(l, r) => TLExpr::Gt(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Lte(l, r) => TLExpr::Lte(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),
        TLExpr::Gte(l, r) => TLExpr::Gte(
            Box::new(substitute(l, var, value)),
            Box::new(substitute(r, var, value)),
        ),

        // Recursively substitute in unary operations
        TLExpr::Not(e) => TLExpr::Not(Box::new(substitute(e, var, value))),
        TLExpr::Box(e) => TLExpr::Box(Box::new(substitute(e, var, value))),
        TLExpr::Diamond(e) => TLExpr::Diamond(Box::new(substitute(e, var, value))),
        TLExpr::Next(e) => TLExpr::Next(Box::new(substitute(e, var, value))),
        TLExpr::Eventually(e) => TLExpr::Eventually(Box::new(substitute(e, var, value))),
        TLExpr::Always(e) => TLExpr::Always(Box::new(substitute(e, var, value))),
        TLExpr::Until { before, after } => TLExpr::Until {
            before: Box::new(substitute(before, var, value)),
            after: Box::new(substitute(after, var, value)),
        },

        // Fuzzy logic operators
        TLExpr::TNorm { kind, left, right } => TLExpr::TNorm {
            kind: *kind,
            left: Box::new(substitute(left, var, value)),
            right: Box::new(substitute(right, var, value)),
        },
        TLExpr::TCoNorm { kind, left, right } => TLExpr::TCoNorm {
            kind: *kind,
            left: Box::new(substitute(left, var, value)),
            right: Box::new(substitute(right, var, value)),
        },
        TLExpr::FuzzyNot { kind, expr } => TLExpr::FuzzyNot {
            kind: *kind,
            expr: Box::new(substitute(expr, var, value)),
        },
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::FuzzyImplication {
            kind: *kind,
            premise: Box::new(substitute(premise, var, value)),
            conclusion: Box::new(substitute(conclusion, var, value)),
        },

        // Probabilistic operators
        TLExpr::SoftExists {
            var: v,
            domain,
            body,
            temperature,
        } => TLExpr::SoftExists {
            var: v.clone(),
            domain: domain.clone(),
            body: Box::new(if v == var {
                (**body).clone()
            } else {
                substitute(body, var, value)
            }),
            temperature: *temperature,
        },
        TLExpr::SoftForAll {
            var: v,
            domain,
            body,
            temperature,
        } => TLExpr::SoftForAll {
            var: v.clone(),
            domain: domain.clone(),
            body: Box::new(if v == var {
                (**body).clone()
            } else {
                substitute(body, var, value)
            }),
            temperature: *temperature,
        },
        TLExpr::WeightedRule { weight, rule } => TLExpr::WeightedRule {
            weight: *weight,
            rule: Box::new(substitute(rule, var, value)),
        },
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::ProbabilisticChoice {
            alternatives: alternatives
                .iter()
                .map(|(p, e)| (*p, substitute(e, var, value)))
                .collect(),
        },

        // Extended temporal logic
        TLExpr::Release { released, releaser } => TLExpr::Release {
            released: Box::new(substitute(released, var, value)),
            releaser: Box::new(substitute(releaser, var, value)),
        },
        TLExpr::WeakUntil { before, after } => TLExpr::WeakUntil {
            before: Box::new(substitute(before, var, value)),
            after: Box::new(substitute(after, var, value)),
        },
        TLExpr::StrongRelease { released, releaser } => TLExpr::StrongRelease {
            released: Box::new(substitute(released, var, value)),
            releaser: Box::new(substitute(releaser, var, value)),
        },

        TLExpr::Score(e) => TLExpr::Score(Box::new(substitute(e, var, value))),
        TLExpr::Abs(e) => TLExpr::Abs(Box::new(substitute(e, var, value))),
        TLExpr::Floor(e) => TLExpr::Floor(Box::new(substitute(e, var, value))),
        TLExpr::Ceil(e) => TLExpr::Ceil(Box::new(substitute(e, var, value))),
        TLExpr::Round(e) => TLExpr::Round(Box::new(substitute(e, var, value))),
        TLExpr::Sqrt(e) => TLExpr::Sqrt(Box::new(substitute(e, var, value))),
        TLExpr::Exp(e) => TLExpr::Exp(Box::new(substitute(e, var, value))),
        TLExpr::Log(e) => TLExpr::Log(Box::new(substitute(e, var, value))),
        TLExpr::Sin(e) => TLExpr::Sin(Box::new(substitute(e, var, value))),
        TLExpr::Cos(e) => TLExpr::Cos(Box::new(substitute(e, var, value))),
        TLExpr::Tan(e) => TLExpr::Tan(Box::new(substitute(e, var, value))),

        // For quantifiers and aggregates, don't substitute if the variable shadows
        TLExpr::Exists {
            var: qvar,
            domain,
            body,
        } => {
            if qvar == var {
                expr.clone() // Variable is shadowed, don't substitute
            } else {
                TLExpr::Exists {
                    var: qvar.clone(),
                    domain: domain.clone(),
                    body: Box::new(substitute(body, var, value)),
                }
            }
        }
        TLExpr::ForAll {
            var: qvar,
            domain,
            body,
        } => {
            if qvar == var {
                expr.clone() // Variable is shadowed, don't substitute
            } else {
                TLExpr::ForAll {
                    var: qvar.clone(),
                    domain: domain.clone(),
                    body: Box::new(substitute(body, var, value)),
                }
            }
        }
        TLExpr::Aggregate {
            op,
            var: avar,
            domain,
            body,
            group_by,
        } => {
            if avar == var {
                expr.clone() // Variable is shadowed, don't substitute
            } else {
                TLExpr::Aggregate {
                    op: op.clone(),
                    var: avar.clone(),
                    domain: domain.clone(),
                    body: Box::new(substitute(body, var, value)),
                    group_by: group_by.clone(),
                }
            }
        }

        // For Let bindings, handle shadowing and substitute recursively
        TLExpr::Let {
            var: lvar,
            value: lvalue,
            body,
        } => {
            let new_value = substitute(lvalue, var, value);
            if lvar == var {
                // Variable is shadowed in body, don't substitute there
                TLExpr::Let {
                    var: lvar.clone(),
                    value: Box::new(new_value),
                    body: body.clone(),
                }
            } else {
                TLExpr::Let {
                    var: lvar.clone(),
                    value: Box::new(new_value),
                    body: Box::new(substitute(body, var, value)),
                }
            }
        }

        // For if-then-else, substitute in all branches
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::IfThenElse {
            condition: Box::new(substitute(condition, var, value)),
            then_branch: Box::new(substitute(then_branch, var, value)),
            else_branch: Box::new(substitute(else_branch, var, value)),
        },

        // Alpha.3 enhancements: recurse with substitution
        TLExpr::Lambda {
            var: lvar,
            var_type,
            body,
        } => {
            if lvar == var {
                expr.clone() // Variable shadowed, don't substitute in body
            } else {
                TLExpr::lambda(lvar.clone(), var_type.clone(), substitute(body, var, value))
            }
        }
        TLExpr::Apply { function, argument } => TLExpr::apply(
            substitute(function, var, value),
            substitute(argument, var, value),
        ),
        TLExpr::SetMembership { element, set } => {
            TLExpr::set_membership(substitute(element, var, value), substitute(set, var, value))
        }
        TLExpr::SetUnion { left, right } => {
            TLExpr::set_union(substitute(left, var, value), substitute(right, var, value))
        }
        TLExpr::SetIntersection { left, right } => {
            TLExpr::set_intersection(substitute(left, var, value), substitute(right, var, value))
        }
        TLExpr::SetDifference { left, right } => {
            TLExpr::set_difference(substitute(left, var, value), substitute(right, var, value))
        }
        TLExpr::SetCardinality { set } => TLExpr::set_cardinality(substitute(set, var, value)),
        TLExpr::EmptySet => expr.clone(),
        TLExpr::SetComprehension {
            var: svar,
            domain,
            condition,
        } => {
            if svar == var {
                expr.clone() // Variable shadowed
            } else {
                TLExpr::set_comprehension(
                    svar.clone(),
                    domain.clone(),
                    substitute(condition, var, value),
                )
            }
        }
        TLExpr::CountingExists {
            var: qvar,
            domain,
            body,
            min_count,
        } => {
            if qvar == var {
                expr.clone()
            } else {
                TLExpr::counting_exists(
                    qvar.clone(),
                    domain.clone(),
                    substitute(body, var, value),
                    *min_count,
                )
            }
        }
        TLExpr::CountingForAll {
            var: qvar,
            domain,
            body,
            min_count,
        } => {
            if qvar == var {
                expr.clone()
            } else {
                TLExpr::counting_forall(
                    qvar.clone(),
                    domain.clone(),
                    substitute(body, var, value),
                    *min_count,
                )
            }
        }
        TLExpr::ExactCount {
            var: qvar,
            domain,
            body,
            count,
        } => {
            if qvar == var {
                expr.clone()
            } else {
                TLExpr::exact_count(
                    qvar.clone(),
                    domain.clone(),
                    substitute(body, var, value),
                    *count,
                )
            }
        }
        TLExpr::Majority {
            var: qvar,
            domain,
            body,
        } => {
            if qvar == var {
                expr.clone()
            } else {
                TLExpr::majority(qvar.clone(), domain.clone(), substitute(body, var, value))
            }
        }
        TLExpr::LeastFixpoint { var: fvar, body } => {
            if fvar == var {
                expr.clone()
            } else {
                TLExpr::least_fixpoint(fvar.clone(), substitute(body, var, value))
            }
        }
        TLExpr::GreatestFixpoint { var: fvar, body } => {
            if fvar == var {
                expr.clone()
            } else {
                TLExpr::greatest_fixpoint(fvar.clone(), substitute(body, var, value))
            }
        }
        TLExpr::Nominal { .. } => expr.clone(),
        TLExpr::At { nominal, formula } => {
            TLExpr::at(nominal.clone(), substitute(formula, var, value))
        }
        TLExpr::Somewhere { formula } => TLExpr::somewhere(substitute(formula, var, value)),
        TLExpr::Everywhere { formula } => TLExpr::everywhere(substitute(formula, var, value)),
        TLExpr::AllDifferent { .. } => expr.clone(), // No substitution in variable names
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => TLExpr::global_cardinality(
            variables.clone(),
            values.iter().map(|v| substitute(v, var, value)).collect(),
            min_occurrences.clone(),
            max_occurrences.clone(),
        ),
        TLExpr::Abducible { .. } => expr.clone(),
        TLExpr::Explain { formula } => TLExpr::explain(substitute(formula, var, value)),

        // Constants remain unchanged
        TLExpr::Constant(_) => expr.clone(),
    }
}
