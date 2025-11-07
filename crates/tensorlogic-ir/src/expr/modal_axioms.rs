//! Modal logic axiom systems and verification.
//!
//! This module implements well-known modal logic axiom systems and provides
//! tools for verifying modal expressions conform to specific axioms.
//!
//! # Modal Logic Systems
//!
//! - **K (Basic Modal Logic)**: The weakest normal modal logic
//!   - Axiom K: □(p → q) → (□p → □q) (Distribution axiom)
//!   - Necessitation: If ⊢ p, then ⊢ □p
//!
//! - **T (Reflexive)**: K + T axiom
//!   - Axiom T: □p → p (Reflexivity - what is necessary is true)
//!
//! - **S4 (Transitive)**: K + T + 4 axiom
//!   - Axiom 4: □p → □□p (Transitivity - necessity of necessity)
//!
//! - **S5 (Euclidean)**: K + T + 5 axiom
//!   - Axiom 5: ◇p → □◇p (Euclidean property - possibility implies necessary possibility)
//!   - Equivalent to: S4 + Axiom B (p → □◇p)
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_ir::{TLExpr, ModalSystem, verify_axiom_k, verify_axiom_t};
//!
//! // Check if an expression satisfies axiom K
//! let expr = TLExpr::imply(
//!     TLExpr::modal_box(TLExpr::imply(
//!         TLExpr::pred("p", vec![]),
//!         TLExpr::pred("q", vec![])
//!     )),
//!     TLExpr::imply(
//!         TLExpr::modal_box(TLExpr::pred("p", vec![])),
//!         TLExpr::modal_box(TLExpr::pred("q", vec![]))
//!     )
//! );
//!
//! assert!(verify_axiom_k(&expr));
//! ```

use super::TLExpr;

/// Modal logic axiom systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModalSystem {
    /// K - Basic modal logic (weakest normal modal logic)
    K,
    /// T - Reflexive modal logic (K + reflexivity)
    T,
    /// S4 - Transitive modal logic (T + transitivity)
    S4,
    /// S5 - Euclidean modal logic (S4 + symmetry/Euclidean property)
    S5,
    /// D - Deontic logic (K + seriality: □p → ◇p)
    D,
    /// B - Symmetric modal logic (T + symmetry: p → □◇p)
    B,
}

impl ModalSystem {
    /// Get all axioms that this system satisfies.
    pub fn axioms(&self) -> Vec<&'static str> {
        match self {
            ModalSystem::K => vec!["K"],
            ModalSystem::T => vec!["K", "T"],
            ModalSystem::S4 => vec!["K", "T", "4"],
            ModalSystem::S5 => vec!["K", "T", "4", "5"],
            ModalSystem::D => vec!["K", "D"],
            ModalSystem::B => vec!["K", "T", "B"],
        }
    }

    /// Check if this system includes a specific axiom.
    pub fn has_axiom(&self, axiom: &str) -> bool {
        self.axioms().contains(&axiom)
    }

    /// Get a description of this modal system.
    pub fn description(&self) -> &'static str {
        match self {
            ModalSystem::K => "Basic modal logic - distribution of necessity over implication",
            ModalSystem::T => "Reflexive logic - what is necessary is true",
            ModalSystem::S4 => "Transitive logic - necessity of necessity",
            ModalSystem::S5 => "Euclidean logic - full modal equivalence",
            ModalSystem::D => "Deontic logic - consistency (obligation implies permission)",
            ModalSystem::B => "Symmetric logic - truth implies necessary possibility",
        }
    }
}

/// Verify that an expression conforms to Axiom K: □(p → q) → (□p → □q)
///
/// The K axiom states that necessity distributes over implication.
pub fn verify_axiom_k(expr: &TLExpr) -> bool {
    matches!(
        expr,
        TLExpr::Imply(box1, box2) if matches!(
            (&**box1, &**box2),
            (
                TLExpr::Box(impl1),
                TLExpr::Imply(box_p, box_q)
            ) if matches!(
                (&**impl1, &**box_p, &**box_q),
                (
                    TLExpr::Imply(p1, q1),
                    TLExpr::Box(p2),
                    TLExpr::Box(q2)
                ) if p1 == p2 && q1 == q2
            )
        )
    )
}

/// Verify that an expression conforms to Axiom T: □p → p
///
/// The T axiom states that what is necessary is true (reflexivity).
pub fn verify_axiom_t(expr: &TLExpr) -> bool {
    matches!(
        expr,
        TLExpr::Imply(box_p, p) if matches!(
            (&**box_p, &**p),
            (TLExpr::Box(inner), outer) if **inner == *outer
        )
    )
}

/// Verify that an expression conforms to Axiom 4: □p → □□p
///
/// The 4 axiom states that necessity is transitive.
pub fn verify_axiom_4(expr: &TLExpr) -> bool {
    matches!(
        expr,
        TLExpr::Imply(box_p, box_box_p) if matches!(
            (&**box_p, &**box_box_p),
            (TLExpr::Box(p1), TLExpr::Box(box_p2)) if matches!(
                &**box_p2,
                TLExpr::Box(p2) if p1 == p2
            )
        )
    )
}

/// Verify that an expression conforms to Axiom 5: ◇p → □◇p
///
/// The 5 axiom states the Euclidean property.
pub fn verify_axiom_5(expr: &TLExpr) -> bool {
    matches!(
        expr,
        TLExpr::Imply(dia_p, box_dia_p) if matches!(
            (&**dia_p, &**box_dia_p),
            (TLExpr::Diamond(p1), TLExpr::Box(dia_p2)) if matches!(
                &**dia_p2,
                TLExpr::Diamond(p2) if p1 == p2
            )
        )
    )
}

/// Verify that an expression conforms to Axiom D: □p → ◇p
///
/// The D axiom states seriality (consistency in deontic logic).
pub fn verify_axiom_d(expr: &TLExpr) -> bool {
    matches!(
        expr,
        TLExpr::Imply(box_p, dia_p) if matches!(
            (&**box_p, &**dia_p),
            (TLExpr::Box(p1), TLExpr::Diamond(p2)) if p1 == p2
        )
    )
}

/// Verify that an expression conforms to Axiom B: p → □◇p
///
/// The B axiom states symmetry.
pub fn verify_axiom_b(expr: &TLExpr) -> bool {
    matches!(
        expr,
        TLExpr::Imply(p, box_dia_p) if matches!(
            &**box_dia_p,
            TLExpr::Box(dia_p) if matches!(
                &**dia_p,
                TLExpr::Diamond(p2) if **p == **p2
            )
        )
    )
}

/// Apply axiom K to simplify modal expressions.
///
/// Transforms: □(p → q) ∧ □p into □q (modus ponens in modal logic)
pub fn apply_axiom_k(expr: &TLExpr) -> Option<TLExpr> {
    if let TLExpr::And(left, right) = expr {
        // Check for pattern: □(p → q) ∧ □p
        if let (TLExpr::Box(impl_expr), TLExpr::Box(p_expr)) = (&**left, &**right) {
            if let TLExpr::Imply(p1, q) = &**impl_expr {
                if p1 == p_expr {
                    // We have □(p → q) ∧ □p, so we can derive □q
                    return Some(TLExpr::modal_box((**q).clone()));
                }
            }
        }
        // Try the other order: □p ∧ □(p → q)
        if let (TLExpr::Box(p_expr), TLExpr::Box(impl_expr)) = (&**left, &**right) {
            if let TLExpr::Imply(p1, q) = &**impl_expr {
                if p1 == p_expr {
                    return Some(TLExpr::modal_box((**q).clone()));
                }
            }
        }
    }
    None
}

/// Apply axiom T to simplify modal expressions.
///
/// Transforms: □p into p (when reflexivity holds)
pub fn apply_axiom_t(expr: &TLExpr) -> Option<TLExpr> {
    if let TLExpr::Box(inner) = expr {
        Some((**inner).clone())
    } else {
        None
    }
}

/// Apply axiom 4 to transform modal expressions.
///
/// Transforms: □p into □□p (adds modal depth)
#[allow(dead_code)]
pub fn apply_axiom_4_forward(expr: &TLExpr) -> Option<TLExpr> {
    if let TLExpr::Box(_inner) = expr {
        Some(TLExpr::modal_box(expr.clone()))
    } else {
        None
    }
}

/// Apply axiom 4 in reverse to simplify.
///
/// Transforms: □□p into □p (reduces modal depth)
#[allow(dead_code)]
pub fn apply_axiom_4_backward(expr: &TLExpr) -> Option<TLExpr> {
    if let TLExpr::Box(inner) = expr {
        if let TLExpr::Box(_) = &**inner {
            return Some((**inner).clone());
        }
    }
    None
}

/// Apply axiom 5 to normalize modal expressions.
///
/// Transforms: ◇p into □◇p
#[allow(dead_code)]
pub fn apply_axiom_5(expr: &TLExpr) -> Option<TLExpr> {
    if let TLExpr::Diamond(_) = expr {
        Some(TLExpr::modal_box(expr.clone()))
    } else {
        None
    }
}

/// Normalize a modal expression according to S5 axioms.
///
/// In S5, any sequence of modal operators can be reduced to a single operator.
/// This function reduces nested modalities.
pub fn normalize_s5(expr: &TLExpr) -> TLExpr {
    match expr {
        // □□p = □p in S5
        TLExpr::Box(inner) if matches!(**inner, TLExpr::Box(_)) => normalize_s5(inner),
        // ◇◇p = ◇p in S5
        TLExpr::Diamond(inner) if matches!(**inner, TLExpr::Diamond(_)) => normalize_s5(inner),
        // □◇p = ◇p in S5
        TLExpr::Box(inner) if matches!(**inner, TLExpr::Diamond(_)) => normalize_s5(inner),
        // ◇□p = □p in S5
        TLExpr::Diamond(inner) if matches!(**inner, TLExpr::Box(_)) => normalize_s5(inner),
        // Recurse into subexpressions
        TLExpr::Box(inner) => TLExpr::modal_box(normalize_s5(inner)),
        TLExpr::Diamond(inner) => TLExpr::modal_diamond(normalize_s5(inner)),
        TLExpr::And(l, r) => TLExpr::and(normalize_s5(l), normalize_s5(r)),
        TLExpr::Or(l, r) => TLExpr::or(normalize_s5(l), normalize_s5(r)),
        TLExpr::Not(e) => TLExpr::negate(normalize_s5(e)),
        TLExpr::Imply(l, r) => TLExpr::imply(normalize_s5(l), normalize_s5(r)),
        _ => expr.clone(),
    }
}

/// Calculate the modal depth of an expression.
///
/// Returns the maximum nesting level of modal operators.
pub fn modal_depth(expr: &TLExpr) -> usize {
    match expr {
        TLExpr::Box(inner) | TLExpr::Diamond(inner) => 1 + modal_depth(inner),
        TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
            modal_depth(l).max(modal_depth(r))
        }
        TLExpr::Not(e) => modal_depth(e),
        _ => 0,
    }
}

/// Check if an expression is modal-free (contains no modal operators).
pub fn is_modal_free(expr: &TLExpr) -> bool {
    modal_depth(expr) == 0
}

/// Extract all modal subformulas from an expression.
pub fn extract_modal_subformulas(expr: &TLExpr) -> Vec<TLExpr> {
    let mut formulas = Vec::new();
    extract_modal_subformulas_rec(expr, &mut formulas);
    formulas
}

fn extract_modal_subformulas_rec(expr: &TLExpr, acc: &mut Vec<TLExpr>) {
    match expr {
        TLExpr::Box(_) | TLExpr::Diamond(_) => {
            acc.push(expr.clone());
            // Also recurse into inner expression
            if let TLExpr::Box(inner) | TLExpr::Diamond(inner) = expr {
                extract_modal_subformulas_rec(inner, acc);
            }
        }
        TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
            extract_modal_subformulas_rec(l, acc);
            extract_modal_subformulas_rec(r, acc);
        }
        TLExpr::Not(e) => extract_modal_subformulas_rec(e, acc),
        _ => {}
    }
}

/// Verify that an expression is a theorem in a given modal system.
///
/// This is a simplified verification that checks for basic axiom patterns.
/// Full verification would require a complete proof system.
pub fn is_theorem_in_system(expr: &TLExpr, system: ModalSystem) -> bool {
    // Check if expression matches any axiom in the system
    let axioms = system.axioms();

    (axioms.contains(&"K") && verify_axiom_k(expr))
        || (axioms.contains(&"T") && verify_axiom_t(expr))
        || (axioms.contains(&"4") && verify_axiom_4(expr))
        || (axioms.contains(&"5") && verify_axiom_5(expr))
        || (axioms.contains(&"D") && verify_axiom_d(expr))
        || (axioms.contains(&"B") && verify_axiom_b(expr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modal_system_axioms() {
        assert_eq!(ModalSystem::K.axioms(), vec!["K"]);
        assert_eq!(ModalSystem::T.axioms(), vec!["K", "T"]);
        assert_eq!(ModalSystem::S4.axioms(), vec!["K", "T", "4"]);
        assert_eq!(ModalSystem::S5.axioms(), vec!["K", "T", "4", "5"]);
    }

    #[test]
    fn test_verify_axiom_k() {
        // □(p → q) → (□p → □q)
        let p = TLExpr::pred("p", vec![]);
        let q = TLExpr::pred("q", vec![]);

        let expr = TLExpr::imply(
            TLExpr::modal_box(TLExpr::imply(p.clone(), q.clone())),
            TLExpr::imply(TLExpr::modal_box(p), TLExpr::modal_box(q)),
        );

        assert!(verify_axiom_k(&expr));
    }

    #[test]
    fn test_verify_axiom_t() {
        // □p → p
        let p = TLExpr::pred("p", vec![]);
        let expr = TLExpr::imply(TLExpr::modal_box(p.clone()), p);

        assert!(verify_axiom_t(&expr));
    }

    #[test]
    fn test_verify_axiom_4() {
        // □p → □□p
        let p = TLExpr::pred("p", vec![]);
        let expr = TLExpr::imply(
            TLExpr::modal_box(p.clone()),
            TLExpr::modal_box(TLExpr::modal_box(p)),
        );

        assert!(verify_axiom_4(&expr));
    }

    #[test]
    fn test_verify_axiom_5() {
        // ◇p → □◇p
        let p = TLExpr::pred("p", vec![]);
        let expr = TLExpr::imply(
            TLExpr::modal_diamond(p.clone()),
            TLExpr::modal_box(TLExpr::modal_diamond(p)),
        );

        assert!(verify_axiom_5(&expr));
    }

    #[test]
    fn test_apply_axiom_k() {
        // □(p → q) ∧ □p should yield □q
        let p = TLExpr::pred("p", vec![]);
        let q = TLExpr::pred("q", vec![]);

        let expr = TLExpr::and(
            TLExpr::modal_box(TLExpr::imply(p.clone(), q.clone())),
            TLExpr::modal_box(p),
        );

        let result = apply_axiom_k(&expr).unwrap();
        assert_eq!(result, TLExpr::modal_box(q));
    }

    #[test]
    fn test_apply_axiom_t() {
        let p = TLExpr::pred("p", vec![]);
        let box_p = TLExpr::modal_box(p.clone());

        let result = apply_axiom_t(&box_p).unwrap();
        assert_eq!(result, p);
    }

    #[test]
    fn test_normalize_s5() {
        // □□p should become □p
        let p = TLExpr::pred("p", vec![]);
        let expr = TLExpr::modal_box(TLExpr::modal_box(p.clone()));

        let normalized = normalize_s5(&expr);
        assert_eq!(normalized, TLExpr::modal_box(p.clone()));

        // □◇p should become ◇p
        let expr2 = TLExpr::modal_box(TLExpr::modal_diamond(p.clone()));
        let normalized2 = normalize_s5(&expr2);
        assert_eq!(normalized2, TLExpr::modal_diamond(p));
    }

    #[test]
    fn test_modal_depth() {
        let p = TLExpr::pred("p", vec![]);
        assert_eq!(modal_depth(&p), 0);

        let box_p = TLExpr::modal_box(p.clone());
        assert_eq!(modal_depth(&box_p), 1);

        let box_box_p = TLExpr::modal_box(TLExpr::modal_box(p));
        assert_eq!(modal_depth(&box_box_p), 2);
    }

    #[test]
    fn test_is_modal_free() {
        let p = TLExpr::pred("p", vec![]);
        assert!(is_modal_free(&p));

        let box_p = TLExpr::modal_box(p);
        assert!(!is_modal_free(&box_p));
    }

    #[test]
    fn test_extract_modal_subformulas() {
        let p = TLExpr::pred("p", vec![]);
        let q = TLExpr::pred("q", vec![]);

        // □p ∧ ◇q
        let expr = TLExpr::and(TLExpr::modal_box(p.clone()), TLExpr::modal_diamond(q));

        let subformulas = extract_modal_subformulas(&expr);
        assert_eq!(subformulas.len(), 2);
    }

    #[test]
    fn test_is_theorem_in_system() {
        // Axiom T: □p → p
        let p = TLExpr::pred("p", vec![]);
        let axiom_t = TLExpr::imply(TLExpr::modal_box(p.clone()), p);

        assert!(is_theorem_in_system(&axiom_t, ModalSystem::T));
        assert!(is_theorem_in_system(&axiom_t, ModalSystem::S4));
        assert!(is_theorem_in_system(&axiom_t, ModalSystem::S5));
        assert!(!is_theorem_in_system(&axiom_t, ModalSystem::K)); // K doesn't have axiom T
    }
}
