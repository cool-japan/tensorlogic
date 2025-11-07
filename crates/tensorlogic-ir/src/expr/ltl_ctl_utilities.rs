//! LTL/CTL temporal logic utilities and model checking support.
//!
//! This module provides advanced temporal logic functionality including:
//! - Formula classification (safety, liveness, fairness properties)
//! - Temporal pattern recognition and analysis
//! - Model checking preparation and utilities
//! - Extended temporal equivalences beyond basic optimizations
//! - Temporal formula complexity analysis

use super::TLExpr;
use std::collections::HashSet;

/// Classification of temporal formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalClass {
    /// Safety property: "something bad never happens"
    Safety,
    /// Liveness property: "something good eventually happens"
    Liveness,
    /// Fairness property: "if requested infinitely often, granted infinitely often"
    Fairness,
    /// Persistence property: "eventually always true"
    Persistence,
    /// Recurrence property: "infinitely often true"
    Recurrence,
    /// Mixed: combination of multiple property types
    Mixed,
}

/// Temporal pattern types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TemporalPattern {
    /// Always P: safety property
    AlwaysP,
    /// Eventually P: liveness property
    EventuallyP,
    /// Eventually Always P: persistence
    EventuallyAlwaysP,
    /// Always Eventually P: recurrence
    AlwaysEventuallyP,
    /// P Until Q: reachability
    PUntilQ,
    /// Always (P implies Eventually Q): response property
    AlwaysImpliesEventually,
    /// Always (P implies Next Q): immediate response
    AlwaysImpliesNext,
    /// Nested temporal operators
    Complex,
}

/// Temporal logic complexity metrics.
#[derive(Debug, Clone, Default)]
pub struct TemporalComplexity {
    /// Maximum nesting depth of temporal operators
    pub temporal_depth: usize,
    /// Number of temporal operators
    pub temporal_op_count: usize,
    /// Number of Until operators (most expensive)
    pub until_count: usize,
    /// Number of Release operators
    pub release_count: usize,
    /// Number of Next operators (least expensive)
    pub next_count: usize,
    /// Whether formula contains fairness constraints
    pub has_fairness: bool,
}

/// Classify a temporal formula.
///
/// Determines whether the formula expresses a safety, liveness, fairness,
/// or other temporal property.
pub fn classify_temporal_formula(expr: &TLExpr) -> TemporalClass {
    match expr {
        // Eventually Always P is persistence (check before general Eventually)
        TLExpr::Eventually(e) if matches!(**e, TLExpr::Always(_)) => TemporalClass::Persistence,

        // Eventually P is a liveness property
        TLExpr::Eventually(_) => TemporalClass::Liveness,

        // Always Eventually P is recurrence (check before general Always)
        TLExpr::Always(e) if matches!(**e, TLExpr::Eventually(_)) => TemporalClass::Recurrence,

        // Fairness: Always (P -> Eventually Q) (check before general Always)
        TLExpr::Always(e) if matches!(&**e, TLExpr::Imply(_, then_branch) if matches!(**then_branch, TLExpr::Eventually(_))) => {
            TemporalClass::Fairness
        }

        // Always P is a safety property (general case)
        TLExpr::Always(_) => TemporalClass::Safety,

        // Conjunction of temporal properties
        TLExpr::And(left, right) => {
            let left_class = classify_temporal_formula(left);
            let right_class = classify_temporal_formula(right);
            if left_class == right_class {
                left_class
            } else {
                TemporalClass::Mixed
            }
        }

        _ => TemporalClass::Mixed,
    }
}

/// Identify temporal pattern in a formula.
pub fn identify_temporal_pattern(expr: &TLExpr) -> TemporalPattern {
    match expr {
        TLExpr::Always(e) if !is_temporal(e) => TemporalPattern::AlwaysP,

        TLExpr::Eventually(e) if !is_temporal(e) => TemporalPattern::EventuallyP,

        TLExpr::Eventually(e) if matches!(**e, TLExpr::Always(_)) => {
            TemporalPattern::EventuallyAlwaysP
        }

        TLExpr::Always(e) if matches!(**e, TLExpr::Eventually(_)) => {
            TemporalPattern::AlwaysEventuallyP
        }

        TLExpr::Until { .. } => TemporalPattern::PUntilQ,

        TLExpr::Always(e) => {
            if let TLExpr::Imply(_, then_branch) = &**e {
                if matches!(**then_branch, TLExpr::Eventually(_)) {
                    return TemporalPattern::AlwaysImpliesEventually;
                } else if matches!(**then_branch, TLExpr::Next(_)) {
                    return TemporalPattern::AlwaysImpliesNext;
                }
            }
            TemporalPattern::Complex
        }

        _ => TemporalPattern::Complex,
    }
}

/// Check if an expression contains temporal operators.
pub fn is_temporal(expr: &TLExpr) -> bool {
    match expr {
        TLExpr::Next(_)
        | TLExpr::Eventually(_)
        | TLExpr::Always(_)
        | TLExpr::Until { .. }
        | TLExpr::Release { .. }
        | TLExpr::WeakUntil { .. }
        | TLExpr::StrongRelease { .. } => true,

        TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
            is_temporal(l) || is_temporal(r)
        }

        TLExpr::Not(e) | TLExpr::Score(e) => is_temporal(e),

        TLExpr::Exists { body, .. }
        | TLExpr::ForAll { body, .. }
        | TLExpr::SoftExists { body, .. }
        | TLExpr::SoftForAll { body, .. } => is_temporal(body),

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => is_temporal(condition) || is_temporal(then_branch) || is_temporal(else_branch),

        _ => false,
    }
}

/// Compute temporal complexity metrics for a formula.
pub fn compute_temporal_complexity(expr: &TLExpr) -> TemporalComplexity {
    fn compute_rec(expr: &TLExpr, depth: usize) -> TemporalComplexity {
        let mut metrics = TemporalComplexity {
            temporal_depth: depth,
            ..Default::default()
        };

        match expr {
            TLExpr::Next(e) => {
                metrics.next_count = 1;
                metrics.temporal_op_count = 1;
                let inner = compute_rec(e, depth + 1);
                metrics.merge(inner);
            }

            TLExpr::Eventually(e) => {
                metrics.temporal_op_count = 1;
                let inner = compute_rec(e, depth + 1);
                metrics.merge(inner);
            }

            TLExpr::Always(e) => {
                metrics.temporal_op_count = 1;
                // Check for fairness pattern: Always (P -> Eventually Q)
                if let TLExpr::Imply(_, then_br) = &**e {
                    if matches!(**then_br, TLExpr::Eventually(_)) {
                        metrics.has_fairness = true;
                    }
                }
                let inner = compute_rec(e, depth + 1);
                metrics.merge(inner);
            }

            TLExpr::Until { before, after } => {
                metrics.until_count = 1;
                metrics.temporal_op_count = 1;
                let before_metrics = compute_rec(before, depth + 1);
                let after_metrics = compute_rec(after, depth + 1);
                metrics.merge(before_metrics);
                metrics.merge(after_metrics);
            }

            TLExpr::Release { released, releaser } => {
                metrics.release_count = 1;
                metrics.temporal_op_count = 1;
                let released_metrics = compute_rec(released, depth + 1);
                let releaser_metrics = compute_rec(releaser, depth + 1);
                metrics.merge(released_metrics);
                metrics.merge(releaser_metrics);
            }

            TLExpr::WeakUntil { before, after } => {
                metrics.until_count = 1;
                metrics.temporal_op_count = 1;
                let before_metrics = compute_rec(before, depth);
                let after_metrics = compute_rec(after, depth);
                metrics.merge(before_metrics);
                metrics.merge(after_metrics);
            }

            TLExpr::StrongRelease { released, releaser } => {
                metrics.release_count = 1;
                metrics.temporal_op_count = 1;
                let released_metrics = compute_rec(released, depth);
                let releaser_metrics = compute_rec(releaser, depth);
                metrics.merge(released_metrics);
                metrics.merge(releaser_metrics);
            }

            TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
                let left_metrics = compute_rec(l, depth);
                let right_metrics = compute_rec(r, depth);
                metrics.merge(left_metrics);
                metrics.merge(right_metrics);
            }

            TLExpr::Not(e) | TLExpr::Score(e) => {
                let inner = compute_rec(e, depth);
                metrics.merge(inner);
            }

            _ => {}
        }

        metrics
    }

    compute_rec(expr, 0)
}

impl TemporalComplexity {
    fn merge(&mut self, other: TemporalComplexity) {
        self.temporal_depth = self.temporal_depth.max(other.temporal_depth);
        self.temporal_op_count += other.temporal_op_count;
        self.until_count += other.until_count;
        self.release_count += other.release_count;
        self.next_count += other.next_count;
        self.has_fairness = self.has_fairness || other.has_fairness;
    }
}

/// Extract all temporal subformulas from an expression.
pub fn extract_temporal_subformulas(expr: &TLExpr) -> Vec<TLExpr> {
    let mut result = Vec::new();
    extract_temporal_rec(expr, &mut result);
    result
}

fn extract_temporal_rec(expr: &TLExpr, result: &mut Vec<TLExpr>) {
    match expr {
        TLExpr::Next(_)
        | TLExpr::Eventually(_)
        | TLExpr::Always(_)
        | TLExpr::Until { .. }
        | TLExpr::Release { .. }
        | TLExpr::WeakUntil { .. }
        | TLExpr::StrongRelease { .. } => {
            result.push(expr.clone());
        }
        _ => {}
    }

    // Recurse into children
    match expr {
        TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
            extract_temporal_rec(l, result);
            extract_temporal_rec(r, result);
        }
        TLExpr::Not(e)
        | TLExpr::Score(e)
        | TLExpr::Next(e)
        | TLExpr::Eventually(e)
        | TLExpr::Always(e) => {
            extract_temporal_rec(e, result);
        }
        TLExpr::Until { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => {
            extract_temporal_rec(before, result);
            extract_temporal_rec(after, result);
        }
        _ => {}
    }
}

/// Convert temporal formula to safety-progress decomposition.
///
/// Every LTL formula can be expressed as the conjunction of a safety
/// property and a liveness property.
pub fn decompose_safety_liveness(expr: &TLExpr) -> (Option<TLExpr>, Option<TLExpr>) {
    match expr {
        // Always P is pure safety
        TLExpr::Always(e) if !has_liveness(e) => (Some(expr.clone()), None),

        // Eventually P is pure liveness
        TLExpr::Eventually(e) if !has_safety(e) => (None, Some(expr.clone())),

        // Conjunction: decompose both sides
        TLExpr::And(left, right) => {
            let (left_safety, left_liveness) = decompose_safety_liveness(left);
            let (right_safety, right_liveness) = decompose_safety_liveness(right);

            let safety = match (left_safety, right_safety) {
                (Some(l), Some(r)) => Some(TLExpr::and(l, r)),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            };

            let liveness = match (left_liveness, right_liveness) {
                (Some(l), Some(r)) => Some(TLExpr::and(l, r)),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            };

            (safety, liveness)
        }

        // Default: treat as mixed
        _ => (Some(expr.clone()), None),
    }
}

/// Check if formula contains liveness properties.
fn has_liveness(expr: &TLExpr) -> bool {
    match expr {
        TLExpr::Eventually(_) => true,
        TLExpr::Always(e) if matches!(**e, TLExpr::Eventually(_)) => true,
        TLExpr::Until { .. } => true,
        TLExpr::And(l, r) | TLExpr::Or(l, r) => has_liveness(l) || has_liveness(r),
        TLExpr::Not(e) => has_liveness(e),
        _ => false,
    }
}

/// Check if formula contains safety properties.
fn has_safety(expr: &TLExpr) -> bool {
    match expr {
        TLExpr::Always(e) if !matches!(**e, TLExpr::Eventually(_)) => true,
        TLExpr::And(l, r) | TLExpr::Or(l, r) => has_safety(l) || has_safety(r),
        TLExpr::Not(e) => has_safety(e),
        _ => false,
    }
}

/// Check if formula is in negation normal form for temporal operators.
///
/// In temporal NNF, negations only appear directly before predicates.
pub fn is_temporal_nnf(expr: &TLExpr) -> bool {
    match expr {
        TLExpr::Not(e) => {
            // Negation should only be in front of predicates or constants
            matches!(**e, TLExpr::Pred { .. } | TLExpr::Constant(_))
        }

        TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
            is_temporal_nnf(l) && is_temporal_nnf(r)
        }

        TLExpr::Next(e) | TLExpr::Eventually(e) | TLExpr::Always(e) => is_temporal_nnf(e),

        TLExpr::Until { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => is_temporal_nnf(before) && is_temporal_nnf(after),

        _ => true,
    }
}

/// Apply additional LTL equivalences beyond basic optimizations.
///
/// These are more advanced equivalences useful for model checking:
/// - Absorption laws: GFP ∧ FGP ≡ FGP
/// - Distributive laws: F(P ∨ Q) ≡ FP ∨ FQ
/// - Expansion laws: FP ≡ P ∨ XFP
pub fn apply_advanced_ltl_equivalences(expr: &TLExpr) -> TLExpr {
    match expr {
        // F(P ∨ Q) → FP ∨ FQ (distributive)
        TLExpr::Eventually(e) => {
            if let TLExpr::Or(left, right) = &**e {
                return TLExpr::or(
                    TLExpr::eventually(apply_advanced_ltl_equivalences(left)),
                    TLExpr::eventually(apply_advanced_ltl_equivalences(right)),
                );
            }
            TLExpr::eventually(apply_advanced_ltl_equivalences(e))
        }

        // G(P ∧ Q) → GP ∧ GQ (distributive)
        TLExpr::Always(e) => {
            if let TLExpr::And(left, right) = &**e {
                return TLExpr::and(
                    TLExpr::always(apply_advanced_ltl_equivalences(left)),
                    TLExpr::always(apply_advanced_ltl_equivalences(right)),
                );
            }
            TLExpr::always(apply_advanced_ltl_equivalences(e))
        }

        // Absorption: GFP ∧ FGP → FGP
        TLExpr::And(left, right) => {
            // Check for GFP ∧ FGP pattern
            let is_gfp_left =
                matches!(left.as_ref(), TLExpr::Always(e) if matches!(**e, TLExpr::Eventually(_)));
            let is_fgp_right =
                matches!(right.as_ref(), TLExpr::Eventually(e) if matches!(**e, TLExpr::Always(_)));

            if is_gfp_left && is_fgp_right {
                return apply_advanced_ltl_equivalences(right);
            }

            // Check reverse
            let is_fgp_left =
                matches!(left.as_ref(), TLExpr::Eventually(e) if matches!(**e, TLExpr::Always(_)));
            let is_gfp_right =
                matches!(right.as_ref(), TLExpr::Always(e) if matches!(**e, TLExpr::Eventually(_)));

            if is_fgp_left && is_gfp_right {
                return apply_advanced_ltl_equivalences(left);
            }

            TLExpr::and(
                apply_advanced_ltl_equivalences(left),
                apply_advanced_ltl_equivalences(right),
            )
        }

        // Recurse for other operators
        TLExpr::Or(l, r) => TLExpr::or(
            apply_advanced_ltl_equivalences(l),
            apply_advanced_ltl_equivalences(r),
        ),

        TLExpr::Imply(l, r) => TLExpr::imply(
            apply_advanced_ltl_equivalences(l),
            apply_advanced_ltl_equivalences(r),
        ),

        TLExpr::Not(e) => TLExpr::negate(apply_advanced_ltl_equivalences(e)),

        TLExpr::Next(e) => TLExpr::next(apply_advanced_ltl_equivalences(e)),

        TLExpr::Until { before, after } => TLExpr::until(
            apply_advanced_ltl_equivalences(before),
            apply_advanced_ltl_equivalences(after),
        ),

        _ => expr.clone(),
    }
}

/// Extract state predicates (non-temporal atomic propositions).
///
/// Useful for model checking to identify the atomic propositions
/// that need to be evaluated in each state.
pub fn extract_state_predicates(expr: &TLExpr) -> HashSet<String> {
    let mut predicates = HashSet::new();
    extract_state_predicates_rec(expr, &mut predicates);
    predicates
}

fn extract_state_predicates_rec(expr: &TLExpr, predicates: &mut HashSet<String>) {
    match expr {
        TLExpr::Pred { name, .. } => {
            predicates.insert(name.clone());
        }

        TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
            extract_state_predicates_rec(l, predicates);
            extract_state_predicates_rec(r, predicates);
        }

        TLExpr::Not(e)
        | TLExpr::Score(e)
        | TLExpr::Next(e)
        | TLExpr::Eventually(e)
        | TLExpr::Always(e) => {
            extract_state_predicates_rec(e, predicates);
        }

        TLExpr::Until { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        } => {
            extract_state_predicates_rec(before, predicates);
            extract_state_predicates_rec(after, predicates);
        }

        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_classify_safety_property() {
        let expr = TLExpr::always(TLExpr::pred("safe", vec![Term::var("x")]));
        assert_eq!(classify_temporal_formula(&expr), TemporalClass::Safety);
    }

    #[test]
    fn test_classify_liveness_property() {
        let expr = TLExpr::eventually(TLExpr::pred("goal", vec![Term::var("x")]));
        assert_eq!(classify_temporal_formula(&expr), TemporalClass::Liveness);
    }

    #[test]
    fn test_classify_persistence() {
        let expr = TLExpr::eventually(TLExpr::always(TLExpr::pred("stable", vec![])));
        assert_eq!(classify_temporal_formula(&expr), TemporalClass::Persistence);
    }

    #[test]
    fn test_classify_fairness() {
        let request = TLExpr::pred("request", vec![]);
        let grant = TLExpr::pred("grant", vec![]);
        let expr = TLExpr::always(TLExpr::imply(request, TLExpr::eventually(grant)));
        assert_eq!(classify_temporal_formula(&expr), TemporalClass::Fairness);
    }

    #[test]
    fn test_identify_pattern_always() {
        let expr = TLExpr::always(TLExpr::pred("P", vec![]));
        assert_eq!(identify_temporal_pattern(&expr), TemporalPattern::AlwaysP);
    }

    #[test]
    fn test_identify_pattern_eventually_always() {
        let expr = TLExpr::eventually(TLExpr::always(TLExpr::pred("P", vec![])));
        assert_eq!(
            identify_temporal_pattern(&expr),
            TemporalPattern::EventuallyAlwaysP
        );
    }

    #[test]
    fn test_is_temporal() {
        let temporal = TLExpr::next(TLExpr::pred("P", vec![]));
        assert!(is_temporal(&temporal));

        let non_temporal = TLExpr::pred("P", vec![]);
        assert!(!is_temporal(&non_temporal));
    }

    #[test]
    fn test_temporal_complexity() {
        let expr = TLExpr::until(
            TLExpr::pred("P", vec![]),
            TLExpr::eventually(TLExpr::pred("Q", vec![])),
        );

        let metrics = compute_temporal_complexity(&expr);
        assert_eq!(metrics.until_count, 1);
        assert_eq!(metrics.temporal_op_count, 2);
        assert!(metrics.temporal_depth >= 1);
    }

    #[test]
    fn test_extract_temporal_subformulas() {
        let expr = TLExpr::and(
            TLExpr::always(TLExpr::pred("P", vec![])),
            TLExpr::eventually(TLExpr::pred("Q", vec![])),
        );

        let subformulas = extract_temporal_subformulas(&expr);
        assert_eq!(subformulas.len(), 2);
    }

    #[test]
    fn test_decompose_pure_safety() {
        let expr = TLExpr::always(TLExpr::pred("safe", vec![]));
        let (safety, liveness) = decompose_safety_liveness(&expr);
        assert!(safety.is_some());
        assert!(liveness.is_none());
    }

    #[test]
    fn test_decompose_pure_liveness() {
        let expr = TLExpr::eventually(TLExpr::pred("goal", vec![]));
        let (safety, liveness) = decompose_safety_liveness(&expr);
        assert!(safety.is_none());
        assert!(liveness.is_some());
    }

    #[test]
    fn test_is_temporal_nnf() {
        // Positive case: negation only in front of predicate
        let nnf = TLExpr::and(
            TLExpr::pred("P", vec![]),
            TLExpr::negate(TLExpr::pred("Q", vec![])),
        );
        assert!(is_temporal_nnf(&nnf));

        // Negative case: negation in front of And
        let not_nnf = TLExpr::negate(TLExpr::and(
            TLExpr::pred("P", vec![]),
            TLExpr::pred("Q", vec![]),
        ));
        assert!(!is_temporal_nnf(&not_nnf));
    }

    #[test]
    fn test_distributive_eventually() {
        let expr = TLExpr::eventually(TLExpr::or(
            TLExpr::pred("P", vec![]),
            TLExpr::pred("Q", vec![]),
        ));

        let result = apply_advanced_ltl_equivalences(&expr);
        // Should distribute: F(P ∨ Q) → FP ∨ FQ
        assert!(matches!(result, TLExpr::Or(_, _)));
    }

    #[test]
    fn test_distributive_always() {
        let expr = TLExpr::always(TLExpr::and(
            TLExpr::pred("P", vec![]),
            TLExpr::pred("Q", vec![]),
        ));

        let result = apply_advanced_ltl_equivalences(&expr);
        // Should distribute: G(P ∧ Q) → GP ∧ GQ
        assert!(matches!(result, TLExpr::And(_, _)));
    }

    #[test]
    fn test_extract_state_predicates() {
        let expr = TLExpr::and(
            TLExpr::eventually(TLExpr::pred("P", vec![])),
            TLExpr::always(TLExpr::pred("Q", vec![])),
        );

        let predicates = extract_state_predicates(&expr);
        assert_eq!(predicates.len(), 2);
        assert!(predicates.contains("P"));
        assert!(predicates.contains("Q"));
    }

    #[test]
    fn test_fairness_detection() {
        let request = TLExpr::pred("req", vec![]);
        let grant = TLExpr::pred("grant", vec![]);
        let fairness = TLExpr::always(TLExpr::imply(request, TLExpr::eventually(grant)));

        let metrics = compute_temporal_complexity(&fairness);
        assert!(metrics.has_fairness);
    }
}
