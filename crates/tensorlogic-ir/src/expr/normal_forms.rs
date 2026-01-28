//! Normal form transformations for logical expressions.
//!
//! This module provides conversions to standard logical normal forms:
//! - **CNF (Conjunctive Normal Form)**: A conjunction of disjunctions (AND of ORs)
//! - **DNF (Disjunctive Normal Form)**: A disjunction of conjunctions (OR of ANDs)
//!
//! # Algorithm
//!
//! The transformation follows these steps:
//! 1. Eliminate implications: `A → B` becomes `¬A ∨ B`
//! 2. Push negations inward using De Morgan's laws
//! 3. Distribute operators (CNF: OR over AND, DNF: AND over OR)
//! 4. Flatten nested operators of the same type
//!
//! # Limitations
//!
//! - Quantifiers, arithmetic operations, and aggregations are treated as atomic predicates
//! - Normal form transformation can lead to exponential blowup in expression size
//! - For complex expressions, consider using `to_nnf` (Negation Normal Form) first

use super::TLExpr;

/// Convert an expression to Negation Normal Form (NNF).
///
/// NNF is a normalized form where:
/// - Negations are pushed inward to appear only before predicates
/// - Implications are eliminated
/// - Only AND, OR, NOT (before predicates), predicates, and quantifiers remain
///
/// This is an intermediate step for CNF/DNF conversion.
///
/// # Examples
///
/// ```
/// use tensorlogic_ir::{TLExpr, Term};
///
/// // ¬(A ∧ B) becomes (¬A ∨ ¬B)
/// let a = TLExpr::pred("A", vec![]);
/// let b = TLExpr::pred("B", vec![]);
/// let expr = TLExpr::negate(TLExpr::and(a.clone(), b.clone()));
/// let nnf = tensorlogic_ir::to_nnf(&expr);
/// // Result: OR(NOT(A), NOT(B))
/// ```
pub fn to_nnf(expr: &TLExpr) -> TLExpr {
    match expr {
        // Base cases - already in NNF
        TLExpr::Pred { .. } => expr.clone(),
        TLExpr::Constant(_) => expr.clone(),

        // Logical operators
        TLExpr::And(l, r) => TLExpr::and(to_nnf(l), to_nnf(r)),
        TLExpr::Or(l, r) => TLExpr::or(to_nnf(l), to_nnf(r)),

        // Implication elimination: A → B becomes ¬A ∨ B
        TLExpr::Imply(l, r) => {
            let not_l = push_negation_inward(&TLExpr::negate((**l).clone()));
            let r_nnf = to_nnf(r);
            TLExpr::or(not_l, r_nnf)
        }

        // Push negation inward
        TLExpr::Not(inner) => push_negation_inward(&TLExpr::Not(Box::new((**inner).clone()))),

        // Quantifiers: convert body to NNF
        TLExpr::Exists { var, domain, body } => TLExpr::exists(var, domain, to_nnf(body)),
        TLExpr::ForAll { var, domain, body } => TLExpr::forall(var, domain, to_nnf(body)),

        // Score: convert inner expression
        TLExpr::Score(inner) => TLExpr::score(to_nnf(inner)),

        // Modal and temporal logic operators: convert body to NNF
        TLExpr::Box(inner) => TLExpr::modal_box(to_nnf(inner)),
        TLExpr::Diamond(inner) => TLExpr::modal_diamond(to_nnf(inner)),
        TLExpr::Next(inner) => TLExpr::next(to_nnf(inner)),
        TLExpr::Eventually(inner) => TLExpr::eventually(to_nnf(inner)),
        TLExpr::Always(inner) => TLExpr::always(to_nnf(inner)),
        TLExpr::Until { before, after } => TLExpr::until(to_nnf(before), to_nnf(after)),

        // Arithmetic, comparison, and other operations are treated as atomic
        // (they don't have logical structure to normalize)
        TLExpr::Add(l, r) => TLExpr::add(to_nnf(l), to_nnf(r)),
        TLExpr::Sub(l, r) => TLExpr::sub(to_nnf(l), to_nnf(r)),
        TLExpr::Mul(l, r) => TLExpr::mul(to_nnf(l), to_nnf(r)),
        TLExpr::Div(l, r) => TLExpr::div(to_nnf(l), to_nnf(r)),
        TLExpr::Pow(l, r) => TLExpr::pow(to_nnf(l), to_nnf(r)),
        TLExpr::Mod(l, r) => TLExpr::modulo(to_nnf(l), to_nnf(r)),
        TLExpr::Min(l, r) => TLExpr::min(to_nnf(l), to_nnf(r)),
        TLExpr::Max(l, r) => TLExpr::max(to_nnf(l), to_nnf(r)),

        // Unary operations
        TLExpr::Abs(e) => TLExpr::abs(to_nnf(e)),
        TLExpr::Floor(e) => TLExpr::floor(to_nnf(e)),
        TLExpr::Ceil(e) => TLExpr::ceil(to_nnf(e)),
        TLExpr::Round(e) => TLExpr::round(to_nnf(e)),
        TLExpr::Sqrt(e) => TLExpr::sqrt(to_nnf(e)),
        TLExpr::Exp(e) => TLExpr::exp(to_nnf(e)),
        TLExpr::Log(e) => TLExpr::log(to_nnf(e)),
        TLExpr::Sin(e) => TLExpr::sin(to_nnf(e)),
        TLExpr::Cos(e) => TLExpr::cos(to_nnf(e)),
        TLExpr::Tan(e) => TLExpr::tan(to_nnf(e)),

        // Comparisons
        TLExpr::Eq(l, r) => TLExpr::eq(to_nnf(l), to_nnf(r)),
        TLExpr::Lt(l, r) => TLExpr::lt(to_nnf(l), to_nnf(r)),
        TLExpr::Gt(l, r) => TLExpr::gt(to_nnf(l), to_nnf(r)),
        TLExpr::Lte(l, r) => TLExpr::lte(to_nnf(l), to_nnf(r)),
        TLExpr::Gte(l, r) => TLExpr::gte(to_nnf(l), to_nnf(r)),

        // Conditionals
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => TLExpr::if_then_else(to_nnf(condition), to_nnf(then_branch), to_nnf(else_branch)),

        // Aggregations and Let bindings
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
            body: Box::new(to_nnf(body)),
            group_by: group_by.clone(),
        },
        TLExpr::Let { var, value, body } => TLExpr::let_binding(var, to_nnf(value), to_nnf(body)),

        // Fuzzy logic operators: pass through (treat as atomic for boolean structure)
        TLExpr::TNorm { kind, left, right } => TLExpr::tnorm(*kind, to_nnf(left), to_nnf(right)),
        TLExpr::TCoNorm { kind, left, right } => {
            TLExpr::tconorm(*kind, to_nnf(left), to_nnf(right))
        }
        TLExpr::FuzzyNot { kind, expr } => TLExpr::fuzzy_not(*kind, to_nnf(expr)),
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => TLExpr::fuzzy_imply(*kind, to_nnf(premise), to_nnf(conclusion)),

        // Probabilistic operators: pass through
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::soft_exists(var, domain, to_nnf(body), *temperature),
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => TLExpr::soft_forall(var, domain, to_nnf(body), *temperature),
        TLExpr::WeightedRule { weight, rule } => TLExpr::weighted_rule(*weight, to_nnf(rule)),
        TLExpr::ProbabilisticChoice { alternatives } => TLExpr::probabilistic_choice(
            alternatives.iter().map(|(p, e)| (*p, to_nnf(e))).collect(),
        ),

        // Extended temporal logic: pass through
        TLExpr::Release { released, releaser } => {
            TLExpr::release(to_nnf(released), to_nnf(releaser))
        }
        TLExpr::WeakUntil { before, after } => TLExpr::weak_until(to_nnf(before), to_nnf(after)),
        TLExpr::StrongRelease { released, releaser } => {
            TLExpr::strong_release(to_nnf(released), to_nnf(releaser))
        }

        // Beta.1 enhancements: pass through (treat as atomic for NNF purposes)
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => TLExpr::lambda(var.clone(), var_type.clone(), to_nnf(body)),
        TLExpr::Apply { function, argument } => TLExpr::apply(to_nnf(function), to_nnf(argument)),
        TLExpr::SetMembership { element, set } => {
            TLExpr::set_membership(to_nnf(element), to_nnf(set))
        }
        TLExpr::SetUnion { left, right } => TLExpr::set_union(to_nnf(left), to_nnf(right)),
        TLExpr::SetIntersection { left, right } => {
            TLExpr::set_intersection(to_nnf(left), to_nnf(right))
        }
        TLExpr::SetDifference { left, right } => {
            TLExpr::set_difference(to_nnf(left), to_nnf(right))
        }
        TLExpr::SetCardinality { set } => TLExpr::set_cardinality(to_nnf(set)),
        TLExpr::EmptySet => expr.clone(),
        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => TLExpr::set_comprehension(var.clone(), domain.clone(), to_nnf(condition)),
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => TLExpr::counting_exists(var.clone(), domain.clone(), to_nnf(body), *min_count),
        TLExpr::CountingForAll {
            var,
            domain,
            body,
            min_count,
        } => TLExpr::counting_forall(var.clone(), domain.clone(), to_nnf(body), *min_count),
        TLExpr::ExactCount {
            var,
            domain,
            body,
            count,
        } => TLExpr::exact_count(var.clone(), domain.clone(), to_nnf(body), *count),
        TLExpr::Majority { var, domain, body } => {
            TLExpr::majority(var.clone(), domain.clone(), to_nnf(body))
        }
        TLExpr::LeastFixpoint { var, body } => TLExpr::least_fixpoint(var.clone(), to_nnf(body)),
        TLExpr::GreatestFixpoint { var, body } => {
            TLExpr::greatest_fixpoint(var.clone(), to_nnf(body))
        }
        TLExpr::Nominal { .. } => expr.clone(),
        TLExpr::At { nominal, formula } => TLExpr::at(nominal.clone(), to_nnf(formula)),
        TLExpr::Somewhere { formula } => TLExpr::somewhere(to_nnf(formula)),
        TLExpr::Everywhere { formula } => TLExpr::everywhere(to_nnf(formula)),
        TLExpr::AllDifferent { .. } => expr.clone(),
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => TLExpr::global_cardinality(
            variables.clone(),
            values.iter().map(to_nnf).collect(),
            min_occurrences.clone(),
            max_occurrences.clone(),
        ),
        TLExpr::Abducible { .. } => expr.clone(),
        TLExpr::Explain { formula } => TLExpr::explain(to_nnf(formula)),
    }
}

/// Push negation inward using De Morgan's laws.
fn push_negation_inward(expr: &TLExpr) -> TLExpr {
    match expr {
        // Double negation elimination: ¬¬A = A
        TLExpr::Not(inner) => match &**inner {
            TLExpr::Not(inner2) => to_nnf(inner2),
            // De Morgan's laws
            TLExpr::And(l, r) => {
                // ¬(A ∧ B) = ¬A ∨ ¬B
                let not_l = push_negation_inward(&TLExpr::negate((**l).clone()));
                let not_r = push_negation_inward(&TLExpr::negate((**r).clone()));
                TLExpr::or(not_l, not_r)
            }
            TLExpr::Or(l, r) => {
                // ¬(A ∨ B) = ¬A ∧ ¬B
                let not_l = push_negation_inward(&TLExpr::negate((**l).clone()));
                let not_r = push_negation_inward(&TLExpr::negate((**r).clone()));
                TLExpr::and(not_l, not_r)
            }
            // ¬(A → B) = A ∧ ¬B
            TLExpr::Imply(l, r) => {
                let l_nnf = to_nnf(l);
                let not_r = push_negation_inward(&TLExpr::negate((**r).clone()));
                TLExpr::and(l_nnf, not_r)
            }
            // Quantifier negation: ¬∃x.P(x) = ∀x.¬P(x), ¬∀x.P(x) = ∃x.¬P(x)
            TLExpr::Exists { var, domain, body } => {
                let not_body = push_negation_inward(&TLExpr::negate((**body).clone()));
                TLExpr::forall(var, domain, not_body)
            }
            TLExpr::ForAll { var, domain, body } => {
                let not_body = push_negation_inward(&TLExpr::negate((**body).clone()));
                TLExpr::exists(var, domain, not_body)
            }
            // For atomic predicates and other expressions, keep the negation
            _ => expr.clone(),
        },
        // Non-negation: convert to NNF
        _ => to_nnf(expr),
    }
}

/// Convert an expression to Conjunctive Normal Form (CNF).
///
/// CNF is a conjunction of disjunctions: `(A ∨ B ∨ C) ∧ (D ∨ E) ∧ ...`
///
/// **Warning**: CNF conversion can lead to exponential blowup in expression size.
/// For expressions with many nested operators, consider limiting the depth or
/// using approximation techniques.
///
/// # Examples
///
/// ```
/// use tensorlogic_ir::{TLExpr, Term};
///
/// // (A ∧ B) ∨ C becomes (A ∨ C) ∧ (B ∨ C)
/// let a = TLExpr::pred("A", vec![]);
/// let b = TLExpr::pred("B", vec![]);
/// let c = TLExpr::pred("C", vec![]);
/// let expr = TLExpr::or(TLExpr::and(a, b), c);
/// let cnf = tensorlogic_ir::to_cnf(&expr);
/// // Result: AND(OR(A, C), OR(B, C))
/// ```
pub fn to_cnf(expr: &TLExpr) -> TLExpr {
    let nnf = to_nnf(expr);
    cnf_distribute(&nnf)
}

/// Distribute OR over AND to achieve CNF.
fn cnf_distribute(expr: &TLExpr) -> TLExpr {
    match expr {
        TLExpr::And(l, r) => {
            let l_cnf = cnf_distribute(l);
            let r_cnf = cnf_distribute(r);
            TLExpr::and(l_cnf, r_cnf)
        }
        TLExpr::Or(l, r) => {
            let l_cnf = cnf_distribute(l);
            let r_cnf = cnf_distribute(r);
            distribute_or_over_and(&l_cnf, &r_cnf)
        }
        TLExpr::Exists { var, domain, body } => TLExpr::exists(var, domain, cnf_distribute(body)),
        TLExpr::ForAll { var, domain, body } => TLExpr::forall(var, domain, cnf_distribute(body)),
        _ => expr.clone(),
    }
}

/// Helper function to distribute OR over AND.
/// (A ∧ B) ∨ C = (A ∨ C) ∧ (B ∨ C)
/// A ∨ (B ∧ C) = (A ∨ B) ∧ (A ∨ C)
fn distribute_or_over_and(left: &TLExpr, right: &TLExpr) -> TLExpr {
    match (left, right) {
        // (A ∧ B) ∨ C = (A ∨ C) ∧ (B ∨ C)
        (TLExpr::And(a, b), c) => {
            let left_part = distribute_or_over_and(a, c);
            let right_part = distribute_or_over_and(b, c);
            TLExpr::and(left_part, right_part)
        }
        // A ∨ (B ∧ C) = (A ∨ B) ∧ (A ∨ C)
        (a, TLExpr::And(b, c)) => {
            let left_part = distribute_or_over_and(a, b);
            let right_part = distribute_or_over_and(a, c);
            TLExpr::and(left_part, right_part)
        }
        // Base case: A ∨ B (both are not ANDs)
        (a, b) => TLExpr::or(a.clone(), b.clone()),
    }
}

/// Convert an expression to Disjunctive Normal Form (DNF).
///
/// DNF is a disjunction of conjunctions: `(A ∧ B ∧ C) ∨ (D ∧ E) ∨ ...`
///
/// **Warning**: DNF conversion can lead to exponential blowup in expression size.
/// For expressions with many nested operators, consider limiting the depth or
/// using approximation techniques.
///
/// # Examples
///
/// ```
/// use tensorlogic_ir::{TLExpr, Term};
///
/// // (A ∨ B) ∧ C becomes (A ∧ C) ∨ (B ∧ C)
/// let a = TLExpr::pred("A", vec![]);
/// let b = TLExpr::pred("B", vec![]);
/// let c = TLExpr::pred("C", vec![]);
/// let expr = TLExpr::and(TLExpr::or(a, b), c);
/// let dnf = tensorlogic_ir::to_dnf(&expr);
/// // Result: OR(AND(A, C), AND(B, C))
/// ```
pub fn to_dnf(expr: &TLExpr) -> TLExpr {
    let nnf = to_nnf(expr);
    dnf_distribute(&nnf)
}

/// Distribute AND over OR to achieve DNF.
fn dnf_distribute(expr: &TLExpr) -> TLExpr {
    match expr {
        TLExpr::Or(l, r) => {
            let l_dnf = dnf_distribute(l);
            let r_dnf = dnf_distribute(r);
            TLExpr::or(l_dnf, r_dnf)
        }
        TLExpr::And(l, r) => {
            let l_dnf = dnf_distribute(l);
            let r_dnf = dnf_distribute(r);
            distribute_and_over_or(&l_dnf, &r_dnf)
        }
        TLExpr::Exists { var, domain, body } => TLExpr::exists(var, domain, dnf_distribute(body)),
        TLExpr::ForAll { var, domain, body } => TLExpr::forall(var, domain, dnf_distribute(body)),
        _ => expr.clone(),
    }
}

/// Helper function to distribute AND over OR.
/// (A ∨ B) ∧ C = (A ∧ C) ∨ (B ∧ C)
/// A ∧ (B ∨ C) = (A ∧ B) ∨ (A ∧ C)
fn distribute_and_over_or(left: &TLExpr, right: &TLExpr) -> TLExpr {
    match (left, right) {
        // (A ∨ B) ∧ C = (A ∧ C) ∨ (B ∧ C)
        (TLExpr::Or(a, b), c) => {
            let left_part = distribute_and_over_or(a, c);
            let right_part = distribute_and_over_or(b, c);
            TLExpr::or(left_part, right_part)
        }
        // A ∧ (B ∨ C) = (A ∧ B) ∨ (A ∧ C)
        (a, TLExpr::Or(b, c)) => {
            let left_part = distribute_and_over_or(a, b);
            let right_part = distribute_and_over_or(a, c);
            TLExpr::or(left_part, right_part)
        }
        // Base case: A ∧ B (both are not ORs)
        (a, b) => TLExpr::and(a.clone(), b.clone()),
    }
}

/// Check if an expression is in Conjunctive Normal Form (CNF).
///
/// An expression is in CNF if it's a conjunction of disjunctions where
/// negations appear only before predicates.
pub fn is_cnf(expr: &TLExpr) -> bool {
    match expr {
        TLExpr::And(l, r) => is_cnf(l) && is_cnf(r),
        TLExpr::Or(l, r) => is_cnf_clause(l) && is_cnf_clause(r),
        TLExpr::Not(inner) => is_literal(inner),
        TLExpr::Pred { .. } => true,
        TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => is_cnf(body),
        TLExpr::Constant(_) => true,
        _ => false, // Arithmetic, comparisons, etc. are treated as atomic
    }
}

/// Check if an expression is a CNF clause (disjunction of literals).
fn is_cnf_clause(expr: &TLExpr) -> bool {
    match expr {
        TLExpr::Or(l, r) => is_cnf_clause(l) && is_cnf_clause(r),
        TLExpr::Not(inner) => is_literal(inner),
        TLExpr::Pred { .. } => true,
        TLExpr::Constant(_) => true,
        _ => false,
    }
}

/// Check if an expression is in Disjunctive Normal Form (DNF).
///
/// An expression is in DNF if it's a disjunction of conjunctions where
/// negations appear only before predicates.
pub fn is_dnf(expr: &TLExpr) -> bool {
    match expr {
        TLExpr::Or(l, r) => is_dnf(l) && is_dnf(r),
        TLExpr::And(l, r) => is_dnf_clause(l) && is_dnf_clause(r),
        TLExpr::Not(inner) => is_literal(inner),
        TLExpr::Pred { .. } => true,
        TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => is_dnf(body),
        TLExpr::Constant(_) => true,
        _ => false,
    }
}

/// Check if an expression is a DNF clause (conjunction of literals).
fn is_dnf_clause(expr: &TLExpr) -> bool {
    match expr {
        TLExpr::And(l, r) => is_dnf_clause(l) && is_dnf_clause(r),
        TLExpr::Not(inner) => is_literal(inner),
        TLExpr::Pred { .. } => true,
        TLExpr::Constant(_) => true,
        _ => false,
    }
}

/// Check if an expression is a literal (predicate or constant).
fn is_literal(expr: &TLExpr) -> bool {
    matches!(expr, TLExpr::Pred { .. } | TLExpr::Constant(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    fn pred_a() -> TLExpr {
        TLExpr::pred("A", vec![])
    }

    fn pred_b() -> TLExpr {
        TLExpr::pred("B", vec![])
    }

    fn pred_c() -> TLExpr {
        TLExpr::pred("C", vec![])
    }

    #[test]
    fn test_nnf_double_negation() {
        // ¬¬A = A
        let expr = TLExpr::negate(TLExpr::negate(pred_a()));
        let nnf = to_nnf(&expr);
        assert_eq!(nnf, pred_a());
    }

    #[test]
    fn test_nnf_de_morgan_and() {
        // ¬(A ∧ B) = ¬A ∨ ¬B
        let expr = TLExpr::negate(TLExpr::and(pred_a(), pred_b()));
        let nnf = to_nnf(&expr);
        assert!(matches!(nnf, TLExpr::Or(_, _)));
        if let TLExpr::Or(l, r) = nnf {
            assert!(matches!(*l, TLExpr::Not(_)));
            assert!(matches!(*r, TLExpr::Not(_)));
        }
    }

    #[test]
    fn test_nnf_de_morgan_or() {
        // ¬(A ∨ B) = ¬A ∧ ¬B
        let expr = TLExpr::negate(TLExpr::or(pred_a(), pred_b()));
        let nnf = to_nnf(&expr);
        assert!(matches!(nnf, TLExpr::And(_, _)));
        if let TLExpr::And(l, r) = nnf {
            assert!(matches!(*l, TLExpr::Not(_)));
            assert!(matches!(*r, TLExpr::Not(_)));
        }
    }

    #[test]
    fn test_nnf_implication() {
        // A → B = ¬A ∨ B
        let expr = TLExpr::imply(pred_a(), pred_b());
        let nnf = to_nnf(&expr);
        assert!(matches!(nnf, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_nnf_quantifier_negation() {
        // ¬∃x.P(x) = ∀x.¬P(x)
        let exists_expr = TLExpr::exists("x", "D", pred_a());
        let neg_exists = TLExpr::negate(exists_expr);
        let nnf = to_nnf(&neg_exists);
        assert!(matches!(nnf, TLExpr::ForAll { .. }));

        // ¬∀x.P(x) = ∃x.¬P(x)
        let forall_expr = TLExpr::forall("x", "D", pred_a());
        let neg_forall = TLExpr::negate(forall_expr);
        let nnf = to_nnf(&neg_forall);
        assert!(matches!(nnf, TLExpr::Exists { .. }));
    }

    #[test]
    fn test_cnf_simple_distribution() {
        // (A ∧ B) ∨ C = (A ∨ C) ∧ (B ∨ C)
        let expr = TLExpr::or(TLExpr::and(pred_a(), pred_b()), pred_c());
        let cnf = to_cnf(&expr);

        // Verify it's in CNF form
        assert!(is_cnf(&cnf));
        // Should have top-level AND
        assert!(matches!(cnf, TLExpr::And(_, _)));
    }

    #[test]
    fn test_cnf_already_in_cnf() {
        // (A ∨ B) ∧ C is already in CNF
        let expr = TLExpr::and(TLExpr::or(pred_a(), pred_b()), pred_c());
        let cnf = to_cnf(&expr);
        assert!(is_cnf(&cnf));
    }

    #[test]
    fn test_dnf_simple_distribution() {
        // (A ∨ B) ∧ C = (A ∧ C) ∨ (B ∧ C)
        let expr = TLExpr::and(TLExpr::or(pred_a(), pred_b()), pred_c());
        let dnf = to_dnf(&expr);

        // Verify it's in DNF form
        assert!(is_dnf(&dnf));
        // Should have top-level OR
        assert!(matches!(dnf, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_dnf_already_in_dnf() {
        // (A ∧ B) ∨ C is already in DNF
        let expr = TLExpr::or(TLExpr::and(pred_a(), pred_b()), pred_c());
        let dnf = to_dnf(&expr);
        assert!(is_dnf(&dnf));
    }

    #[test]
    fn test_is_cnf() {
        // (A ∨ B) ∧ (C ∨ ¬D) is CNF
        let d = TLExpr::pred("D", vec![]);
        let expr = TLExpr::and(
            TLExpr::or(pred_a(), pred_b()),
            TLExpr::or(pred_c(), TLExpr::negate(d)),
        );
        assert!(is_cnf(&expr));

        // (A ∧ B) ∨ C is not CNF
        let not_cnf = TLExpr::or(TLExpr::and(pred_a(), pred_b()), pred_c());
        assert!(!is_cnf(&not_cnf));
    }

    #[test]
    fn test_is_dnf() {
        // (A ∧ B) ∨ (C ∧ ¬D) is DNF
        let d = TLExpr::pred("D", vec![]);
        let expr = TLExpr::or(
            TLExpr::and(pred_a(), pred_b()),
            TLExpr::and(pred_c(), TLExpr::negate(d)),
        );
        assert!(is_dnf(&expr));

        // (A ∨ B) ∧ C is not DNF
        let not_dnf = TLExpr::and(TLExpr::or(pred_a(), pred_b()), pred_c());
        assert!(!is_dnf(&not_dnf));
    }

    #[test]
    fn test_complex_cnf_conversion() {
        // ((A ∨ B) ∧ C) ∨ D should convert to CNF
        let expr = TLExpr::or(
            TLExpr::and(TLExpr::or(pred_a(), pred_b()), pred_c()),
            TLExpr::pred("D", vec![]),
        );
        let cnf = to_cnf(&expr);
        assert!(is_cnf(&cnf));
    }

    #[test]
    fn test_complex_dnf_conversion() {
        // ((A ∧ B) ∨ C) ∧ D should convert to DNF
        let expr = TLExpr::and(
            TLExpr::or(TLExpr::and(pred_a(), pred_b()), pred_c()),
            TLExpr::pred("D", vec![]),
        );
        let dnf = to_dnf(&expr);
        assert!(is_dnf(&dnf));
    }

    #[test]
    fn test_cnf_with_negations() {
        // ¬A ∨ (B ∧ ¬C) should convert to (¬A ∨ B) ∧ (¬A ∨ ¬C)
        let expr = TLExpr::or(
            TLExpr::negate(pred_a()),
            TLExpr::and(pred_b(), TLExpr::negate(pred_c())),
        );
        let cnf = to_cnf(&expr);
        assert!(is_cnf(&cnf));
    }

    #[test]
    fn test_dnf_with_negations() {
        // ¬A ∧ (B ∨ ¬C) should convert to (¬A ∧ B) ∨ (¬A ∧ ¬C)
        let expr = TLExpr::and(
            TLExpr::negate(pred_a()),
            TLExpr::or(pred_b(), TLExpr::negate(pred_c())),
        );
        let dnf = to_dnf(&expr);
        assert!(is_dnf(&dnf));
    }

    #[test]
    fn test_cnf_with_quantifiers() {
        // ∀x. (A(x) ∨ B(x)) should preserve quantifier
        let pred_ax = TLExpr::pred("A", vec![Term::var("x")]);
        let pred_bx = TLExpr::pred("B", vec![Term::var("x")]);
        let expr = TLExpr::forall("x", "D", TLExpr::or(pred_ax, pred_bx));
        let cnf = to_cnf(&expr);
        assert!(matches!(cnf, TLExpr::ForAll { .. }));
    }

    #[test]
    fn test_literal_identification() {
        assert!(is_literal(&pred_a()));
        assert!(is_literal(&TLExpr::constant(42.0)));
        assert!(!is_literal(&TLExpr::and(pred_a(), pred_b())));
        assert!(!is_literal(&TLExpr::negate(pred_a())));
    }
}
