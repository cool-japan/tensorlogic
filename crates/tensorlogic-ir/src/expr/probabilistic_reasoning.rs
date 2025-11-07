//! Probabilistic reasoning and probability bounds propagation.
//!
//! This module implements probabilistic inference and uncertainty quantification including:
//! - Probability interval arithmetic (Fréchet bounds)
//! - Imprecise probabilities (lower/upper bounds)
//! - Credal sets and convex sets of probability distributions
//! - Probabilistic semantics for weighted rules
//! - Probability propagation through logical connectives
//!
//! # Applications
//! - Markov Logic Networks (MLNs)
//! - Probabilistic Logic Programs
//! - Bayesian inference with interval probabilities
//! - Uncertainty quantification under incomplete information

use super::TLExpr;
use std::collections::HashMap;

/// Probability interval representing imprecise probabilities.
///
/// Represents the set [lower, upper] of possible probability values.
/// Follows the theory of imprecise probabilities and credal sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityInterval {
    /// Lower probability bound (must be in [0, 1])
    pub lower: f64,
    /// Upper probability bound (must be in [0, 1] and >= lower)
    pub upper: f64,
}

impl ProbabilityInterval {
    /// Create a new probability interval.
    ///
    /// Returns None if bounds are invalid (not in [0,1] or lower > upper).
    pub fn new(lower: f64, upper: f64) -> Option<Self> {
        if lower < 0.0 || upper > 1.0 || lower > upper {
            None
        } else {
            Some(Self { lower, upper })
        }
    }

    /// Create a precise probability (point interval).
    pub fn precise(prob: f64) -> Option<Self> {
        Self::new(prob, prob)
    }

    /// Create a vacuous interval [0, 1] (complete ignorance).
    pub fn vacuous() -> Self {
        Self {
            lower: 0.0,
            upper: 1.0,
        }
    }

    /// Width of the interval (measure of imprecision).
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Check if this is a precise probability.
    pub fn is_precise(&self) -> bool {
        (self.upper - self.lower).abs() < 1e-10
    }

    /// Check if the interval is vacuous (completely imprecise).
    pub fn is_vacuous(&self) -> bool {
        self.lower == 0.0 && self.upper == 1.0
    }

    /// Complement: P(¬A) given P(A).
    pub fn complement(&self) -> Self {
        Self {
            lower: 1.0 - self.upper,
            upper: 1.0 - self.lower,
        }
    }

    /// Conjunction bounds: P(A ∧ B) given P(A) and P(B).
    ///
    /// Uses Fréchet bounds: max(0, P(A) + P(B) - 1) ≤ P(A ∧ B) ≤ min(P(A), P(B))
    pub fn and(&self, other: &Self) -> Self {
        let lower = (self.lower + other.lower - 1.0).max(0.0);
        let upper = self.upper.min(other.upper);
        Self { lower, upper }
    }

    /// Disjunction bounds: P(A ∨ B) given P(A) and P(B).
    ///
    /// Uses Fréchet bounds: max(P(A), P(B)) ≤ P(A ∨ B) ≤ min(1, P(A) + P(B))
    pub fn or(&self, other: &Self) -> Self {
        let lower = self.lower.max(other.lower);
        let upper = (self.upper + other.upper).min(1.0);
        Self { lower, upper }
    }

    /// Implication bounds: P(A → B) given P(A) and P(B).
    ///
    /// A → B ≡ ¬A ∨ B, so use complement and disjunction.
    pub fn implies(&self, other: &Self) -> Self {
        self.complement().or(other)
    }

    /// Conditional probability bounds: P(B|A) given P(A) and P(A ∧ B).
    ///
    /// If P(A) > 0, returns P(A ∧ B) / P(A).
    /// Uses interval division: [a,b] / [c,d] = [a/d, b/c] for positive intervals.
    pub fn conditional(&self, joint: &Self) -> Option<Self> {
        if self.upper == 0.0 {
            // Cannot condition on zero probability event
            None
        } else if self.lower == 0.0 {
            // Lower bound might be zero, use conservative bounds
            Some(Self {
                lower: 0.0,
                upper: joint.upper / self.lower.max(1e-10),
            })
        } else {
            Some(Self {
                lower: joint.lower / self.upper,
                upper: joint.upper / self.lower,
            })
        }
    }

    /// Intersection of two probability intervals.
    ///
    /// Returns None if intervals don't overlap.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        if lower <= upper {
            Some(Self { lower, upper })
        } else {
            None
        }
    }

    /// Convex combination of two intervals.
    ///
    /// Useful for averaging or mixing probability assessments.
    pub fn convex_combine(&self, other: &Self, weight: f64) -> Option<Self> {
        if !(0.0..=1.0).contains(&weight) {
            return None;
        }
        Some(Self {
            lower: self.lower * weight + other.lower * (1.0 - weight),
            upper: self.upper * weight + other.upper * (1.0 - weight),
        })
    }
}

/// Credal set: convex set of probability distributions.
///
/// Represented by extreme points (vertices) of the credal set.
#[derive(Debug, Clone)]
pub struct CredalSet {
    /// Extreme probability distributions (each sums to 1)
    extreme_points: Vec<HashMap<String, f64>>,
}

impl CredalSet {
    /// Create a credal set from extreme points.
    pub fn new(extreme_points: Vec<HashMap<String, f64>>) -> Self {
        Self { extreme_points }
    }

    /// Create a precise credal set (single distribution).
    pub fn precise(distribution: HashMap<String, f64>) -> Self {
        Self {
            extreme_points: vec![distribution],
        }
    }

    /// Get lower probability bound for an event.
    pub fn lower_prob(&self, event: &str) -> f64 {
        self.extreme_points
            .iter()
            .filter_map(|dist| dist.get(event).copied())
            .fold(f64::INFINITY, f64::min)
    }

    /// Get upper probability bound for an event.
    pub fn upper_prob(&self, event: &str) -> f64 {
        self.extreme_points
            .iter()
            .filter_map(|dist| dist.get(event).copied())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get probability interval for an event.
    pub fn prob_interval(&self, event: &str) -> ProbabilityInterval {
        ProbabilityInterval {
            lower: self.lower_prob(event),
            upper: self.upper_prob(event),
        }
    }

    /// Number of extreme points in the credal set.
    pub fn size(&self) -> usize {
        self.extreme_points.len()
    }

    /// Check if credal set is precise (single distribution).
    pub fn is_precise(&self) -> bool {
        self.extreme_points.len() == 1
    }
}

/// Propagate probability intervals through a logical expression.
///
/// Given probability assignments to atomic predicates, computes
/// probability bounds for the compound expression.
pub fn propagate_probabilities(
    expr: &TLExpr,
    prob_map: &HashMap<String, ProbabilityInterval>,
) -> ProbabilityInterval {
    match expr {
        TLExpr::Pred { name, .. } => prob_map
            .get(name)
            .copied()
            .unwrap_or_else(ProbabilityInterval::vacuous),

        TLExpr::Constant(v) => {
            if *v >= 1.0 {
                ProbabilityInterval::precise(1.0).unwrap()
            } else if *v <= 0.0 {
                ProbabilityInterval::precise(0.0).unwrap()
            } else {
                ProbabilityInterval::vacuous()
            }
        }

        TLExpr::And(left, right) => {
            let left_prob = propagate_probabilities(left, prob_map);
            let right_prob = propagate_probabilities(right, prob_map);
            left_prob.and(&right_prob)
        }

        TLExpr::Or(left, right) => {
            let left_prob = propagate_probabilities(left, prob_map);
            let right_prob = propagate_probabilities(right, prob_map);
            left_prob.or(&right_prob)
        }

        TLExpr::Not(inner) => {
            let inner_prob = propagate_probabilities(inner, prob_map);
            inner_prob.complement()
        }

        TLExpr::Imply(premise, conclusion) => {
            let premise_prob = propagate_probabilities(premise, prob_map);
            let conclusion_prob = propagate_probabilities(conclusion, prob_map);
            premise_prob.implies(&conclusion_prob)
        }

        // For weighted rules, the weight represents confidence
        TLExpr::WeightedRule { weight, rule } => {
            let rule_prob = propagate_probabilities(rule, prob_map);
            // Weight modulates the probability bounds
            ProbabilityInterval {
                lower: rule_prob.lower * weight,
                upper: rule_prob.upper * weight,
            }
        }

        // For probabilistic choice, compute expected bounds
        TLExpr::ProbabilisticChoice { alternatives } => {
            let mut lower_sum = 0.0;
            let mut upper_sum = 0.0;
            let mut total_weight = 0.0;

            for (prob, expr) in alternatives {
                let expr_interval = propagate_probabilities(expr, prob_map);
                lower_sum += prob * expr_interval.lower;
                upper_sum += prob * expr_interval.upper;
                total_weight += prob;
            }

            // Normalize if weights don't sum to 1
            if total_weight > 0.0 && (total_weight - 1.0).abs() > 1e-10 {
                lower_sum /= total_weight;
                upper_sum /= total_weight;
            }

            ProbabilityInterval {
                lower: lower_sum.clamp(0.0, 1.0),
                upper: upper_sum.clamp(0.0, 1.0),
            }
        }

        // Default: vacuous interval (no information)
        _ => ProbabilityInterval::vacuous(),
    }
}

/// Compute tightest probability bounds for an expression using optimization.
///
/// This uses linear programming to find the tightest possible bounds
/// given constraints. For now, uses a simple iterative tightening approach.
pub fn compute_tight_bounds(
    expr: &TLExpr,
    prob_map: &HashMap<String, ProbabilityInterval>,
) -> ProbabilityInterval {
    // Start with Fréchet bounds
    let mut current = propagate_probabilities(expr, prob_map);

    // Iteratively tighten bounds by considering dependencies
    // For simplicity, we do 3 iterations (could be made configurable)
    for _ in 0..3 {
        current = tighten_iteration(expr, prob_map, &current);
    }

    current
}

fn tighten_iteration(
    expr: &TLExpr,
    prob_map: &HashMap<String, ProbabilityInterval>,
    current: &ProbabilityInterval,
) -> ProbabilityInterval {
    match expr {
        TLExpr::And(left, right) => {
            let left_prob = compute_tight_bounds(left, prob_map);
            let right_prob = compute_tight_bounds(right, prob_map);

            // Tighten using independence assumption if possible
            let mut result = left_prob.and(&right_prob);

            // Additional tightening: if we know the result bounds, constrain components
            if let Some(intersection) = result.intersect(current) {
                result = intersection;
            }

            result
        }

        TLExpr::Or(left, right) => {
            let left_prob = compute_tight_bounds(left, prob_map);
            let right_prob = compute_tight_bounds(right, prob_map);

            let mut result = left_prob.or(&right_prob);

            if let Some(intersection) = result.intersect(current) {
                result = intersection;
            }

            result
        }

        _ => propagate_probabilities(expr, prob_map),
    }
}

/// Extract probabilistic semantics from weighted rules.
///
/// Converts weighted rules into probability distributions over possible worlds.
pub fn extract_probabilistic_semantics(expr: &TLExpr) -> Vec<(f64, TLExpr)> {
    let mut weighted_rules = Vec::new();
    extract_weighted_rec(expr, &mut weighted_rules);
    weighted_rules
}

fn extract_weighted_rec(expr: &TLExpr, result: &mut Vec<(f64, TLExpr)>) {
    match expr {
        TLExpr::WeightedRule { weight, rule } => {
            result.push((*weight, (**rule).clone()));
            extract_weighted_rec(rule, result);
        }

        TLExpr::ProbabilisticChoice { alternatives } => {
            for (prob, expr) in alternatives {
                result.push((*prob, expr.clone()));
                extract_weighted_rec(expr, result);
            }
        }

        TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
            extract_weighted_rec(l, result);
            extract_weighted_rec(r, result);
        }

        TLExpr::Not(e) => extract_weighted_rec(e, result),

        _ => {}
    }
}

/// Compute probability of an expression under a Markov Logic Network (MLN) semantics.
///
/// MLN uses weighted rules where weight w corresponds to log-odds ratio.
/// P(world) ∝ exp(∑ w_i * n_i) where n_i is number of groundings satisfied.
pub fn mln_probability(
    _expr: &TLExpr,
    weights: &[(f64, TLExpr)],
    evidence: &HashMap<String, bool>,
) -> f64 {
    // Simplified MLN: compute unnormalized probability
    let mut total_weight = 0.0;

    for (weight, rule) in weights {
        if evaluates_true(rule, evidence) {
            total_weight += weight;
        }
    }

    // Logistic function to get probability
    1.0 / (1.0 + (-total_weight).exp())
}

/// Simple boolean evaluation for ground facts.
fn evaluates_true(expr: &TLExpr, evidence: &HashMap<String, bool>) -> bool {
    match expr {
        TLExpr::Pred { name, .. } => evidence.get(name).copied().unwrap_or(false),

        TLExpr::And(l, r) => evaluates_true(l, evidence) && evaluates_true(r, evidence),

        TLExpr::Or(l, r) => evaluates_true(l, evidence) || evaluates_true(r, evidence),

        TLExpr::Not(e) => !evaluates_true(e, evidence),

        TLExpr::Imply(l, r) => !evaluates_true(l, evidence) || evaluates_true(r, evidence),

        TLExpr::Constant(v) => *v >= 1.0,

        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_interval_creation() {
        let interval = ProbabilityInterval::new(0.3, 0.7).unwrap();
        assert!((interval.lower - 0.3).abs() < 1e-10);
        assert!((interval.upper - 0.7).abs() < 1e-10);
        assert!((interval.width() - 0.4).abs() < 1e-10);

        // Invalid intervals
        assert!(ProbabilityInterval::new(-0.1, 0.5).is_none());
        assert!(ProbabilityInterval::new(0.8, 0.5).is_none());
        assert!(ProbabilityInterval::new(0.5, 1.5).is_none());
    }

    #[test]
    fn test_precise_probability() {
        let precise = ProbabilityInterval::precise(0.5).unwrap();
        assert!(precise.is_precise());
        assert_eq!(precise.width(), 0.0);
    }

    #[test]
    fn test_vacuous_interval() {
        let vacuous = ProbabilityInterval::vacuous();
        assert!(vacuous.is_vacuous());
        assert_eq!(vacuous.width(), 1.0);
    }

    #[test]
    fn test_complement() {
        let interval = ProbabilityInterval::new(0.3, 0.7).unwrap();
        let complement = interval.complement();
        assert!((complement.lower - 0.3).abs() < 1e-10);
        assert!((complement.upper - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_frechet_and() {
        let p_a = ProbabilityInterval::new(0.4, 0.6).unwrap();
        let p_b = ProbabilityInterval::new(0.5, 0.8).unwrap();
        let p_and = p_a.and(&p_b);

        // Lower: max(0, 0.4 + 0.5 - 1) = 0.0
        assert_eq!(p_and.lower, 0.0);
        // Upper: min(0.6, 0.8) = 0.6
        assert_eq!(p_and.upper, 0.6);
    }

    #[test]
    fn test_frechet_or() {
        let p_a = ProbabilityInterval::new(0.4, 0.6).unwrap();
        let p_b = ProbabilityInterval::new(0.5, 0.8).unwrap();
        let p_or = p_a.or(&p_b);

        // Lower: max(0.4, 0.5) = 0.5
        assert_eq!(p_or.lower, 0.5);
        // Upper: min(1, 0.6 + 0.8) = 1.0
        assert_eq!(p_or.upper, 1.0);
    }

    #[test]
    fn test_implication_bounds() {
        let p_a = ProbabilityInterval::new(0.3, 0.5).unwrap();
        let p_b = ProbabilityInterval::new(0.6, 0.9).unwrap();
        let p_implies = p_a.implies(&p_b);

        // A -> B ≡ ¬A ∨ B
        let not_a = p_a.complement();
        let expected = not_a.or(&p_b);

        assert_eq!(p_implies.lower, expected.lower);
        assert_eq!(p_implies.upper, expected.upper);
    }

    #[test]
    fn test_conditional_probability() {
        let p_a = ProbabilityInterval::new(0.4, 0.6).unwrap();
        let p_a_and_b = ProbabilityInterval::new(0.2, 0.3).unwrap();

        let p_b_given_a = p_a.conditional(&p_a_and_b).unwrap();

        // P(B|A) = P(A ∧ B) / P(A)
        // Lower: 0.2 / 0.6 = 0.333...
        // Upper: 0.3 / 0.4 = 0.75
        assert!((p_b_given_a.lower - 0.333).abs() < 0.01);
        assert!((p_b_given_a.upper - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_interval_intersection() {
        let i1 = ProbabilityInterval::new(0.2, 0.7).unwrap();
        let i2 = ProbabilityInterval::new(0.5, 0.9).unwrap();

        let intersection = i1.intersect(&i2).unwrap();
        assert_eq!(intersection.lower, 0.5);
        assert_eq!(intersection.upper, 0.7);

        // No intersection
        let i3 = ProbabilityInterval::new(0.1, 0.3).unwrap();
        let i4 = ProbabilityInterval::new(0.6, 0.9).unwrap();
        assert!(i3.intersect(&i4).is_none());
    }

    #[test]
    fn test_convex_combination() {
        let i1 = ProbabilityInterval::new(0.2, 0.4).unwrap();
        let i2 = ProbabilityInterval::new(0.6, 0.8).unwrap();

        let combo = i1.convex_combine(&i2, 0.5).unwrap();
        assert!((combo.lower - 0.4).abs() < 1e-10); // 0.2 * 0.5 + 0.6 * 0.5
        assert!((combo.upper - 0.6).abs() < 1e-10); // 0.4 * 0.5 + 0.8 * 0.5
    }

    #[test]
    fn test_propagate_probabilities_and() {
        let mut prob_map = HashMap::new();
        prob_map.insert("P".to_string(), ProbabilityInterval::new(0.4, 0.6).unwrap());
        prob_map.insert("Q".to_string(), ProbabilityInterval::new(0.5, 0.8).unwrap());

        let expr = TLExpr::and(TLExpr::pred("P", vec![]), TLExpr::pred("Q", vec![]));

        let result = propagate_probabilities(&expr, &prob_map);
        assert_eq!(result.lower, 0.0);
        assert_eq!(result.upper, 0.6);
    }

    #[test]
    fn test_propagate_probabilities_or() {
        let mut prob_map = HashMap::new();
        prob_map.insert("P".to_string(), ProbabilityInterval::new(0.4, 0.6).unwrap());
        prob_map.insert("Q".to_string(), ProbabilityInterval::new(0.5, 0.8).unwrap());

        let expr = TLExpr::or(TLExpr::pred("P", vec![]), TLExpr::pred("Q", vec![]));

        let result = propagate_probabilities(&expr, &prob_map);
        assert_eq!(result.lower, 0.5);
        assert_eq!(result.upper, 1.0);
    }

    #[test]
    fn test_propagate_probabilities_not() {
        let mut prob_map = HashMap::new();
        prob_map.insert("P".to_string(), ProbabilityInterval::new(0.3, 0.7).unwrap());

        let expr = TLExpr::negate(TLExpr::pred("P", vec![]));

        let result = propagate_probabilities(&expr, &prob_map);
        assert!((result.lower - 0.3).abs() < 1e-10);
        assert!((result.upper - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_rule_propagation() {
        let mut prob_map = HashMap::new();
        prob_map.insert("P".to_string(), ProbabilityInterval::new(0.5, 0.8).unwrap());

        let expr = TLExpr::weighted_rule(0.5, TLExpr::pred("P", vec![]));

        let result = propagate_probabilities(&expr, &prob_map);
        assert_eq!(result.lower, 0.25); // 0.5 * 0.5
        assert_eq!(result.upper, 0.4); // 0.5 * 0.8
    }

    #[test]
    fn test_probabilistic_choice() {
        let mut prob_map = HashMap::new();
        prob_map.insert("P".to_string(), ProbabilityInterval::precise(0.6).unwrap());
        prob_map.insert("Q".to_string(), ProbabilityInterval::precise(0.4).unwrap());

        let expr = TLExpr::probabilistic_choice(vec![
            (0.5, TLExpr::pred("P", vec![])),
            (0.5, TLExpr::pred("Q", vec![])),
        ]);

        let result = propagate_probabilities(&expr, &prob_map);
        // Expected: 0.5 * 0.6 + 0.5 * 0.4 = 0.5
        assert_eq!(result.lower, 0.5);
        assert_eq!(result.upper, 0.5);
    }

    #[test]
    fn test_credal_set() {
        let mut dist1 = HashMap::new();
        dist1.insert("A".to_string(), 0.3);
        dist1.insert("B".to_string(), 0.7);

        let mut dist2 = HashMap::new();
        dist2.insert("A".to_string(), 0.6);
        dist2.insert("B".to_string(), 0.4);

        let credal = CredalSet::new(vec![dist1, dist2]);

        assert_eq!(credal.lower_prob("A"), 0.3);
        assert_eq!(credal.upper_prob("A"), 0.6);
        assert!(!credal.is_precise());
    }

    #[test]
    fn test_mln_probability() {
        let rule = TLExpr::pred("P", vec![]);
        let weights = vec![(2.0, rule.clone())];

        let mut evidence = HashMap::new();
        evidence.insert("P".to_string(), true);

        let prob = mln_probability(&rule, &weights, &evidence);
        // exp(2) / (1 + exp(2)) ≈ 0.88
        assert!((prob - 0.88).abs() < 0.01);
    }
}
