//! Expression simplification and normalization
//!
//! This module provides logical expression simplification to optimize expressions
//! before compilation. It applies standard logical equivalences and normalization rules.

#![allow(dead_code)]

use tensorlogic_ir::{TLExpr, Term};

/// Simplify a logical expression using standard equivalences
///
/// Applies the following transformations:
/// - Constant folding: Add(Constant(a), Constant(b)) => Constant(a + b)
/// - Identity laws: AND(x, true) => x, OR(x, false) => x
/// - Annihilation laws: AND(x, false) => false, OR(x, true) => true
/// - Double negation elimination: NOT(NOT(x)) => x
/// - Idempotent laws: AND(x, x) => x, OR(x, x) => x
/// - Absorption laws: AND(x, OR(x, y)) => x
/// - De Morgan's laws: NOT(AND(x, y)) => OR(NOT(x), NOT(y))
pub fn simplify_expression(expr: &TLExpr) -> TLExpr {
    let expr = apply_constant_folding(expr);
    let expr = apply_identity_laws(&expr);
    let expr = apply_double_negation(&expr);
    let expr = apply_idempotent_laws(&expr);
    let expr = apply_absorption_laws(&expr);

    apply_de_morgan(&expr)
}

/// Fold constant expressions at compile time
///
/// Evaluates arithmetic operations on constants:
/// - Add(Constant(a), Constant(b)) => Constant(a + b)
/// - Mul(Constant(a), Constant(b)) => Constant(a * b)
/// - etc.
fn apply_constant_folding(expr: &TLExpr) -> TLExpr {
    match expr {
        // Arithmetic binary operations
        TLExpr::Add(left, right) => {
            let left_folded = apply_constant_folding(left);
            let right_folded = apply_constant_folding(right);
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&left_folded, &right_folded) {
                TLExpr::Constant(a + b)
            } else {
                TLExpr::Add(Box::new(left_folded), Box::new(right_folded))
            }
        }
        TLExpr::Sub(left, right) => {
            let left_folded = apply_constant_folding(left);
            let right_folded = apply_constant_folding(right);
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&left_folded, &right_folded) {
                TLExpr::Constant(a - b)
            } else {
                TLExpr::Sub(Box::new(left_folded), Box::new(right_folded))
            }
        }
        TLExpr::Mul(left, right) => {
            let left_folded = apply_constant_folding(left);
            let right_folded = apply_constant_folding(right);
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&left_folded, &right_folded) {
                TLExpr::Constant(a * b)
            } else {
                TLExpr::Mul(Box::new(left_folded), Box::new(right_folded))
            }
        }
        TLExpr::Div(left, right) => {
            let left_folded = apply_constant_folding(left);
            let right_folded = apply_constant_folding(right);
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&left_folded, &right_folded) {
                if *b != 0.0 {
                    TLExpr::Constant(a / b)
                } else {
                    TLExpr::Div(Box::new(left_folded), Box::new(right_folded))
                }
            } else {
                TLExpr::Div(Box::new(left_folded), Box::new(right_folded))
            }
        }
        TLExpr::Pow(left, right) => {
            let left_folded = apply_constant_folding(left);
            let right_folded = apply_constant_folding(right);
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&left_folded, &right_folded) {
                TLExpr::Constant(a.powf(*b))
            } else {
                TLExpr::Pow(Box::new(left_folded), Box::new(right_folded))
            }
        }
        TLExpr::Min(left, right) => {
            let left_folded = apply_constant_folding(left);
            let right_folded = apply_constant_folding(right);
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&left_folded, &right_folded) {
                TLExpr::Constant(a.min(*b))
            } else {
                TLExpr::Min(Box::new(left_folded), Box::new(right_folded))
            }
        }
        TLExpr::Max(left, right) => {
            let left_folded = apply_constant_folding(left);
            let right_folded = apply_constant_folding(right);
            if let (TLExpr::Constant(a), TLExpr::Constant(b)) = (&left_folded, &right_folded) {
                TLExpr::Constant(a.max(*b))
            } else {
                TLExpr::Max(Box::new(left_folded), Box::new(right_folded))
            }
        }
        // Unary mathematical operations
        TLExpr::Abs(inner) => {
            let inner_folded = apply_constant_folding(inner);
            if let TLExpr::Constant(a) = inner_folded {
                TLExpr::Constant(a.abs())
            } else {
                TLExpr::Abs(Box::new(inner_folded))
            }
        }
        TLExpr::Sqrt(inner) => {
            let inner_folded = apply_constant_folding(inner);
            if let TLExpr::Constant(a) = inner_folded {
                TLExpr::Constant(a.sqrt())
            } else {
                TLExpr::Sqrt(Box::new(inner_folded))
            }
        }
        TLExpr::Exp(inner) => {
            let inner_folded = apply_constant_folding(inner);
            if let TLExpr::Constant(a) = inner_folded {
                TLExpr::Constant(a.exp())
            } else {
                TLExpr::Exp(Box::new(inner_folded))
            }
        }
        TLExpr::Log(inner) => {
            let inner_folded = apply_constant_folding(inner);
            if let TLExpr::Constant(a) = inner_folded {
                TLExpr::Constant(a.ln())
            } else {
                TLExpr::Log(Box::new(inner_folded))
            }
        }
        // Logical operations (recurse)
        TLExpr::And(left, right) => TLExpr::And(
            Box::new(apply_constant_folding(left)),
            Box::new(apply_constant_folding(right)),
        ),
        TLExpr::Or(left, right) => TLExpr::Or(
            Box::new(apply_constant_folding(left)),
            Box::new(apply_constant_folding(right)),
        ),
        TLExpr::Not(inner) => TLExpr::Not(Box::new(apply_constant_folding(inner))),
        TLExpr::Imply(left, right) => TLExpr::Imply(
            Box::new(apply_constant_folding(left)),
            Box::new(apply_constant_folding(right)),
        ),
        _ => expr.clone(),
    }
}

/// Apply identity and annihilation laws for logical operations
///
/// Identity laws:
/// - AND(x, Constant(1.0)) => x (AND with true)
/// - OR(x, Constant(0.0)) => x (OR with false)
///
/// Annihilation laws:
/// - AND(x, Constant(0.0)) => Constant(0.0) (AND with false)
/// - OR(x, Constant(1.0)) => Constant(1.0) (OR with true)
fn apply_identity_laws(expr: &TLExpr) -> TLExpr {
    match expr {
        TLExpr::And(left, right) => {
            let left_simplified = apply_identity_laws(left);
            let right_simplified = apply_identity_laws(right);

            // AND(x, true) => x
            if let TLExpr::Constant(c) = &right_simplified {
                if (*c - 1.0).abs() < 1e-10 {
                    return left_simplified;
                }
                // AND(x, false) => false
                if c.abs() < 1e-10 {
                    return TLExpr::Constant(0.0);
                }
            }
            // AND(true, x) => x
            if let TLExpr::Constant(c) = &left_simplified {
                if (*c - 1.0).abs() < 1e-10 {
                    return right_simplified;
                }
                // AND(false, x) => false
                if c.abs() < 1e-10 {
                    return TLExpr::Constant(0.0);
                }
            }

            TLExpr::And(Box::new(left_simplified), Box::new(right_simplified))
        }
        TLExpr::Or(left, right) => {
            let left_simplified = apply_identity_laws(left);
            let right_simplified = apply_identity_laws(right);

            // OR(x, false) => x
            if let TLExpr::Constant(c) = &right_simplified {
                if c.abs() < 1e-10 {
                    return left_simplified;
                }
                // OR(x, true) => true
                if (*c - 1.0).abs() < 1e-10 {
                    return TLExpr::Constant(1.0);
                }
            }
            // OR(false, x) => x
            if let TLExpr::Constant(c) = &left_simplified {
                if c.abs() < 1e-10 {
                    return right_simplified;
                }
                // OR(true, x) => true
                if (*c - 1.0).abs() < 1e-10 {
                    return TLExpr::Constant(1.0);
                }
            }

            TLExpr::Or(Box::new(left_simplified), Box::new(right_simplified))
        }
        TLExpr::Not(inner) => TLExpr::Not(Box::new(apply_identity_laws(inner))),
        TLExpr::Imply(left, right) => TLExpr::Imply(
            Box::new(apply_identity_laws(left)),
            Box::new(apply_identity_laws(right)),
        ),
        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(apply_identity_laws(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(apply_identity_laws(body)),
        },
        _ => expr.clone(),
    }
}

/// Remove double negations: NOT(NOT(x)) => x
fn apply_double_negation(expr: &TLExpr) -> TLExpr {
    match expr {
        TLExpr::Not(inner) => {
            if let TLExpr::Not(inner_inner) = &**inner {
                apply_double_negation(inner_inner)
            } else {
                TLExpr::Not(Box::new(apply_double_negation(inner)))
            }
        }
        TLExpr::And(left, right) => TLExpr::And(
            Box::new(apply_double_negation(left)),
            Box::new(apply_double_negation(right)),
        ),
        TLExpr::Or(left, right) => TLExpr::Or(
            Box::new(apply_double_negation(left)),
            Box::new(apply_double_negation(right)),
        ),
        TLExpr::Imply(left, right) => TLExpr::Imply(
            Box::new(apply_double_negation(left)),
            Box::new(apply_double_negation(right)),
        ),
        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(apply_double_negation(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(apply_double_negation(body)),
        },
        _ => expr.clone(),
    }
}

/// Apply idempotent laws: AND(x, x) => x, OR(x, x) => x
fn apply_idempotent_laws(expr: &TLExpr) -> TLExpr {
    match expr {
        TLExpr::And(left, right) => {
            let left_simplified = apply_idempotent_laws(left);
            let right_simplified = apply_idempotent_laws(right);

            // AND(x, x) => x
            if expressions_equal(&left_simplified, &right_simplified) {
                left_simplified
            } else {
                TLExpr::And(Box::new(left_simplified), Box::new(right_simplified))
            }
        }
        TLExpr::Or(left, right) => {
            let left_simplified = apply_idempotent_laws(left);
            let right_simplified = apply_idempotent_laws(right);

            // OR(x, x) => x
            if expressions_equal(&left_simplified, &right_simplified) {
                left_simplified
            } else {
                TLExpr::Or(Box::new(left_simplified), Box::new(right_simplified))
            }
        }
        TLExpr::Not(inner) => TLExpr::Not(Box::new(apply_idempotent_laws(inner))),
        TLExpr::Imply(left, right) => TLExpr::Imply(
            Box::new(apply_idempotent_laws(left)),
            Box::new(apply_idempotent_laws(right)),
        ),
        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(apply_idempotent_laws(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(apply_idempotent_laws(body)),
        },
        _ => expr.clone(),
    }
}

/// Apply absorption laws: AND(x, OR(x, y)) => x, OR(x, AND(x, y)) => x
fn apply_absorption_laws(expr: &TLExpr) -> TLExpr {
    match expr {
        TLExpr::And(left, right) => {
            let left_simplified = apply_absorption_laws(left);
            let right_simplified = apply_absorption_laws(right);

            // AND(x, OR(x, y)) => x
            if let TLExpr::Or(or_left, _or_right) = &right_simplified {
                if expressions_equal(&left_simplified, or_left) {
                    return left_simplified;
                }
            }
            // AND(OR(x, y), x) => x
            if let TLExpr::Or(or_left, _or_right) = &left_simplified {
                if expressions_equal(&right_simplified, or_left) {
                    return right_simplified;
                }
            }

            TLExpr::And(Box::new(left_simplified), Box::new(right_simplified))
        }
        TLExpr::Or(left, right) => {
            let left_simplified = apply_absorption_laws(left);
            let right_simplified = apply_absorption_laws(right);

            // OR(x, AND(x, y)) => x
            if let TLExpr::And(and_left, _and_right) = &right_simplified {
                if expressions_equal(&left_simplified, and_left) {
                    return left_simplified;
                }
            }
            // OR(AND(x, y), x) => x
            if let TLExpr::And(and_left, _and_right) = &left_simplified {
                if expressions_equal(&right_simplified, and_left) {
                    return right_simplified;
                }
            }

            TLExpr::Or(Box::new(left_simplified), Box::new(right_simplified))
        }
        TLExpr::Not(inner) => TLExpr::Not(Box::new(apply_absorption_laws(inner))),
        TLExpr::Imply(left, right) => TLExpr::Imply(
            Box::new(apply_absorption_laws(left)),
            Box::new(apply_absorption_laws(right)),
        ),
        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(apply_absorption_laws(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(apply_absorption_laws(body)),
        },
        _ => expr.clone(),
    }
}

/// Apply De Morgan's laws: NOT(AND(x, y)) => OR(NOT(x), NOT(y))
fn apply_de_morgan(expr: &TLExpr) -> TLExpr {
    match expr {
        TLExpr::Not(inner) => match &**inner {
            TLExpr::And(left, right) => {
                // NOT(AND(x, y)) => OR(NOT(x), NOT(y))
                TLExpr::Or(
                    Box::new(apply_de_morgan(&TLExpr::Not(left.clone()))),
                    Box::new(apply_de_morgan(&TLExpr::Not(right.clone()))),
                )
            }
            TLExpr::Or(left, right) => {
                // NOT(OR(x, y)) => AND(NOT(x), NOT(y))
                TLExpr::And(
                    Box::new(apply_de_morgan(&TLExpr::Not(left.clone()))),
                    Box::new(apply_de_morgan(&TLExpr::Not(right.clone()))),
                )
            }
            _ => TLExpr::Not(Box::new(apply_de_morgan(inner))),
        },
        TLExpr::And(left, right) => TLExpr::And(
            Box::new(apply_de_morgan(left)),
            Box::new(apply_de_morgan(right)),
        ),
        TLExpr::Or(left, right) => TLExpr::Or(
            Box::new(apply_de_morgan(left)),
            Box::new(apply_de_morgan(right)),
        ),
        TLExpr::Imply(left, right) => TLExpr::Imply(
            Box::new(apply_de_morgan(left)),
            Box::new(apply_de_morgan(right)),
        ),
        TLExpr::Exists { var, domain, body } => TLExpr::Exists {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(apply_de_morgan(body)),
        },
        TLExpr::ForAll { var, domain, body } => TLExpr::ForAll {
            var: var.clone(),
            domain: domain.clone(),
            body: Box::new(apply_de_morgan(body)),
        },
        _ => expr.clone(),
    }
}

/// Check if two expressions are structurally equal
fn expressions_equal(left: &TLExpr, right: &TLExpr) -> bool {
    match (left, right) {
        (TLExpr::Pred { name: n1, args: a1 }, TLExpr::Pred { name: n2, args: a2 }) => {
            n1 == n2 && terms_equal_vec(a1, a2)
        }
        (TLExpr::And(l1, r1), TLExpr::And(l2, r2)) => {
            expressions_equal(l1, l2) && expressions_equal(r1, r2)
        }
        (TLExpr::Or(l1, r1), TLExpr::Or(l2, r2)) => {
            expressions_equal(l1, l2) && expressions_equal(r1, r2)
        }
        (TLExpr::Not(e1), TLExpr::Not(e2)) => expressions_equal(e1, e2),
        (TLExpr::Imply(l1, r1), TLExpr::Imply(l2, r2)) => {
            expressions_equal(l1, l2) && expressions_equal(r1, r2)
        }
        (
            TLExpr::Exists {
                var: v1,
                domain: d1,
                body: b1,
            },
            TLExpr::Exists {
                var: v2,
                domain: d2,
                body: b2,
            },
        ) => v1 == v2 && d1 == d2 && expressions_equal(b1, b2),
        (
            TLExpr::ForAll {
                var: v1,
                domain: d1,
                body: b1,
            },
            TLExpr::ForAll {
                var: v2,
                domain: d2,
                body: b2,
            },
        ) => v1 == v2 && d1 == d2 && expressions_equal(b1, b2),
        _ => false,
    }
}

/// Check if two term vectors are equal
fn terms_equal_vec(terms1: &[Term], terms2: &[Term]) -> bool {
    if terms1.len() != terms2.len() {
        return false;
    }
    terms1.iter().zip(terms2.iter()).all(|(t1, t2)| t1 == t2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_negation() {
        let expr = TLExpr::Not(Box::new(TLExpr::Not(Box::new(TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        }))));

        let simplified = apply_double_negation(&expr);

        assert!(matches!(simplified, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_idempotent_and() {
        let pred = TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        };

        let expr = TLExpr::And(Box::new(pred.clone()), Box::new(pred));

        let simplified = apply_idempotent_laws(&expr);

        assert!(matches!(simplified, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_simplify_complex_expression() {
        // NOT(NOT(P(x)))
        let pred = TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        };

        let double_neg = TLExpr::Not(Box::new(TLExpr::Not(Box::new(pred))));

        let simplified = simplify_expression(&double_neg);

        assert!(matches!(simplified, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_de_morgan_and() {
        let p = TLExpr::Pred {
            name: "P".to_string(),
            args: vec![],
        };
        let q = TLExpr::Pred {
            name: "Q".to_string(),
            args: vec![],
        };

        // NOT(AND(P, Q))
        let expr = TLExpr::Not(Box::new(TLExpr::And(Box::new(p), Box::new(q))));

        let simplified = apply_de_morgan(&expr);

        // Should become OR(NOT(P), NOT(Q))
        assert!(matches!(simplified, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_constant_folding_add() {
        // Add(Constant(2.0), Constant(3.0)) => Constant(5.0)
        let expr = TLExpr::Add(
            Box::new(TLExpr::Constant(2.0)),
            Box::new(TLExpr::Constant(3.0)),
        );

        let simplified = apply_constant_folding(&expr);

        assert!(matches!(simplified, TLExpr::Constant(c) if (c - 5.0).abs() < 1e-10));
    }

    #[test]
    fn test_constant_folding_mul() {
        // Mul(Constant(4.0), Constant(5.0)) => Constant(20.0)
        let expr = TLExpr::Mul(
            Box::new(TLExpr::Constant(4.0)),
            Box::new(TLExpr::Constant(5.0)),
        );

        let simplified = apply_constant_folding(&expr);

        assert!(matches!(simplified, TLExpr::Constant(c) if (c - 20.0).abs() < 1e-10));
    }

    #[test]
    fn test_constant_folding_sqrt() {
        // Sqrt(Constant(16.0)) => Constant(4.0)
        let expr = TLExpr::Sqrt(Box::new(TLExpr::Constant(16.0)));

        let simplified = apply_constant_folding(&expr);

        assert!(matches!(simplified, TLExpr::Constant(c) if (c - 4.0).abs() < 1e-10));
    }

    #[test]
    fn test_constant_folding_nested() {
        // Add(Mul(2.0, 3.0), Constant(4.0)) => Add(Constant(6.0), Constant(4.0)) => Constant(10.0)
        let expr = TLExpr::Add(
            Box::new(TLExpr::Mul(
                Box::new(TLExpr::Constant(2.0)),
                Box::new(TLExpr::Constant(3.0)),
            )),
            Box::new(TLExpr::Constant(4.0)),
        );

        let simplified = apply_constant_folding(&expr);

        assert!(matches!(simplified, TLExpr::Constant(c) if (c - 10.0).abs() < 1e-10));
    }

    #[test]
    fn test_identity_law_and_true() {
        let pred = TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        };

        // AND(P(x), Constant(1.0)) => P(x)
        let expr = TLExpr::And(Box::new(pred.clone()), Box::new(TLExpr::Constant(1.0)));

        let simplified = apply_identity_laws(&expr);

        assert!(matches!(simplified, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_identity_law_and_false() {
        let pred = TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        };

        // AND(P(x), Constant(0.0)) => Constant(0.0)
        let expr = TLExpr::And(Box::new(pred), Box::new(TLExpr::Constant(0.0)));

        let simplified = apply_identity_laws(&expr);

        assert!(matches!(simplified, TLExpr::Constant(c) if c.abs() < 1e-10));
    }

    #[test]
    fn test_identity_law_or_false() {
        let pred = TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        };

        // OR(P(x), Constant(0.0)) => P(x)
        let expr = TLExpr::Or(Box::new(pred.clone()), Box::new(TLExpr::Constant(0.0)));

        let simplified = apply_identity_laws(&expr);

        assert!(matches!(simplified, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_identity_law_or_true() {
        let pred = TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        };

        // OR(P(x), Constant(1.0)) => Constant(1.0)
        let expr = TLExpr::Or(Box::new(pred), Box::new(TLExpr::Constant(1.0)));

        let simplified = apply_identity_laws(&expr);

        assert!(matches!(simplified, TLExpr::Constant(c) if (c - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_combined_simplification() {
        // AND(NOT(NOT(P(x))), Constant(1.0)) => P(x)
        let pred = TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        };

        let double_neg = TLExpr::Not(Box::new(TLExpr::Not(Box::new(pred))));
        let expr = TLExpr::And(Box::new(double_neg), Box::new(TLExpr::Constant(1.0)));

        let simplified = simplify_expression(&expr);

        assert!(matches!(simplified, TLExpr::Pred { .. }));
    }
}
