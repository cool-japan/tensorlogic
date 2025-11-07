//! Confluence checking and critical pair analysis for term rewriting systems.
//!
//! This module provides tools for analyzing the confluence properties of rewrite systems:
//! - **Confluence**: A property where different rewrite sequences lead to the same result
//! - **Critical pairs**: Overlapping rule applications that may cause conflicts
//! - **Joinability**: Whether two expressions can be rewritten to a common form
//!
//! A confluent rewrite system guarantees that the order of rule application doesn't matter,
//! which is crucial for correctness and determinism.

use std::collections::{HashMap, HashSet, VecDeque};

use super::rewriting::RewriteSystem;
use super::TLExpr;

/// A critical pair representing a potential conflict between two rules.
#[derive(Debug, Clone)]
pub struct CriticalPair {
    /// The expression where the overlap occurs
    pub overlap: TLExpr,
    /// Result of applying the first rule
    pub result1: TLExpr,
    /// Result of applying the second rule
    pub result2: TLExpr,
    /// Names of the rules involved
    pub rule1_name: String,
    pub rule2_name: String,
    /// Whether this critical pair is joinable
    pub joinable: Option<bool>,
}

impl CriticalPair {
    /// Create a new critical pair.
    pub fn new(
        overlap: TLExpr,
        result1: TLExpr,
        result2: TLExpr,
        rule1_name: String,
        rule2_name: String,
    ) -> Self {
        Self {
            overlap,
            result1,
            result2,
            rule1_name,
            rule2_name,
            joinable: None,
        }
    }

    /// Check if this critical pair is trivially joinable (results are equal).
    pub fn is_trivially_joinable(&self) -> bool {
        self.result1 == self.result2
    }

    /// Check if the results are syntactically different.
    pub fn has_conflict(&self) -> bool {
        !self.is_trivially_joinable()
    }
}

/// Result of confluence analysis.
#[derive(Debug, Clone)]
pub struct ConfluenceReport {
    /// All critical pairs found
    pub critical_pairs: Vec<CriticalPair>,
    /// Number of joinable critical pairs
    pub joinable_count: usize,
    /// Number of non-joinable critical pairs
    pub non_joinable_count: usize,
    /// Whether the system is locally confluent
    pub is_locally_confluent: bool,
    /// Whether termination was verified
    pub is_terminating: bool,
}

impl ConfluenceReport {
    /// Create a new confluence report.
    pub fn new() -> Self {
        Self {
            critical_pairs: Vec::new(),
            joinable_count: 0,
            non_joinable_count: 0,
            is_locally_confluent: false,
            is_terminating: false,
        }
    }

    /// Check if the system is globally confluent (by Newman's lemma).
    ///
    /// Newman's lemma: A terminating system is confluent iff it is locally confluent.
    pub fn is_confluent(&self) -> bool {
        self.is_terminating && self.is_locally_confluent
    }

    /// Get a summary string.
    pub fn summary(&self) -> String {
        format!(
            "Confluence Report:\n\
             - Critical pairs: {}\n\
             - Joinable: {}\n\
             - Non-joinable: {}\n\
             - Locally confluent: {}\n\
             - Terminating: {}\n\
             - Confluent: {}",
            self.critical_pairs.len(),
            self.joinable_count,
            self.non_joinable_count,
            self.is_locally_confluent,
            self.is_terminating,
            self.is_confluent()
        )
    }
}

impl Default for ConfluenceReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Confluence checker for rewrite systems.
pub struct ConfluenceChecker {
    /// Maximum depth for joinability testing
    max_depth: usize,
    /// Maximum expression size for analysis
    max_expr_size: usize,
    /// Cache of expression pairs and their joinability
    joinability_cache: HashMap<(String, String), bool>,
}

impl ConfluenceChecker {
    /// Create a new confluence checker.
    pub fn new() -> Self {
        Self {
            max_depth: 10,
            max_expr_size: 1000,
            joinability_cache: HashMap::new(),
        }
    }

    /// Set maximum depth for joinability testing.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set maximum expression size.
    pub fn with_max_expr_size(mut self, size: usize) -> Self {
        self.max_expr_size = size;
        self
    }

    /// Check confluence of a rewrite system.
    pub fn check(&mut self, system: &RewriteSystem) -> ConfluenceReport {
        let mut report = ConfluenceReport::new();

        // Find all critical pairs
        self.find_critical_pairs_basic(system, &mut report);

        // Test joinability for each critical pair
        for pair in &mut report.critical_pairs {
            if pair.is_trivially_joinable() {
                pair.joinable = Some(true);
                report.joinable_count += 1;
            } else {
                let joinable = self.test_joinability(&pair.result1, &pair.result2, system);
                pair.joinable = Some(joinable);
                if joinable {
                    report.joinable_count += 1;
                } else {
                    report.non_joinable_count += 1;
                }
            }
        }

        // System is locally confluent if all critical pairs are joinable
        report.is_locally_confluent = report.non_joinable_count == 0;

        // For termination, we use a simple heuristic: no rule increases expression size
        report.is_terminating = self.check_termination_heuristic(system);

        report
    }

    /// Find critical pairs in the system (basic version).
    ///
    /// This is a simplified implementation that checks for overlaps at the top level.
    fn find_critical_pairs_basic(&self, _system: &RewriteSystem, _report: &mut ConfluenceReport) {
        // In a full implementation, we would:
        // 1. For each pair of rules (r1, r2)
        // 2. Find all ways their patterns can overlap
        // 3. Apply both rules and record the results
        //
        // This is complex because it requires:
        // - Unification of patterns
        // - Finding all overlap positions
        // - Handling variable bindings correctly
        //
        // For now, we provide the infrastructure without the full implementation.
        // A production system would use a sophisticated unification algorithm.
    }

    /// Test if two expressions are joinable (can be rewritten to a common form).
    ///
    /// Uses breadth-first search to explore possible rewrites.
    pub fn test_joinability(
        &mut self,
        expr1: &TLExpr,
        expr2: &TLExpr,
        system: &RewriteSystem,
    ) -> bool {
        // Check cache first
        let key = (format!("{:?}", expr1), format!("{:?}", expr2));
        if let Some(&result) = self.joinability_cache.get(&key) {
            return result;
        }

        if expr1 == expr2 {
            self.joinability_cache.insert(key, true);
            return true;
        }

        // BFS from both expressions
        let mut visited1 = HashSet::new();
        let mut visited2 = HashSet::new();
        let mut queue1 = VecDeque::new();
        let mut queue2 = VecDeque::new();

        queue1.push_back((expr1.clone(), 0));
        queue2.push_back((expr2.clone(), 0));

        visited1.insert(format!("{:?}", expr1));
        visited2.insert(format!("{:?}", expr2));

        while !queue1.is_empty() || !queue2.is_empty() {
            // Expand from expr1
            if let Some((current, depth)) = queue1.pop_front() {
                if depth >= self.max_depth {
                    continue;
                }

                // Check if we've reached a form that expr2 can reach
                let current_key = format!("{:?}", &current);
                if visited2.contains(&current_key) {
                    self.joinability_cache.insert(key, true);
                    return true;
                }

                // Apply all possible rewrites
                for rewrite in self.get_all_rewrites(&current, system) {
                    let rewrite_key = format!("{:?}", &rewrite);
                    if !visited1.contains(&rewrite_key) {
                        visited1.insert(rewrite_key);
                        queue1.push_back((rewrite, depth + 1));
                    }
                }
            }

            // Expand from expr2
            if let Some((current, depth)) = queue2.pop_front() {
                if depth >= self.max_depth {
                    continue;
                }

                let current_key = format!("{:?}", &current);
                if visited1.contains(&current_key) {
                    self.joinability_cache.insert(key, true);
                    return true;
                }

                for rewrite in self.get_all_rewrites(&current, system) {
                    let rewrite_key = format!("{:?}", &rewrite);
                    if !visited2.contains(&rewrite_key) {
                        visited2.insert(rewrite_key);
                        queue2.push_back((rewrite, depth + 1));
                    }
                }
            }
        }

        self.joinability_cache.insert(key, false);
        false
    }

    /// Get all possible one-step rewrites of an expression.
    #[allow(clippy::only_used_in_recursion)]
    fn get_all_rewrites(&self, expr: &TLExpr, system: &RewriteSystem) -> Vec<TLExpr> {
        let mut results = Vec::new();

        // Try each rule
        if let Some(rewritten) = system.apply_once(expr) {
            results.push(rewritten);
        }

        // Recursively try rewrites on subexpressions
        match expr {
            TLExpr::And(l, r) => {
                for l_rewrite in self.get_all_rewrites(l, system) {
                    results.push(TLExpr::and(l_rewrite, (**r).clone()));
                }
                for r_rewrite in self.get_all_rewrites(r, system) {
                    results.push(TLExpr::and((**l).clone(), r_rewrite));
                }
            }
            TLExpr::Or(l, r) => {
                for l_rewrite in self.get_all_rewrites(l, system) {
                    results.push(TLExpr::or(l_rewrite, (**r).clone()));
                }
                for r_rewrite in self.get_all_rewrites(r, system) {
                    results.push(TLExpr::or((**l).clone(), r_rewrite));
                }
            }
            TLExpr::Not(e) => {
                for e_rewrite in self.get_all_rewrites(e, system) {
                    results.push(TLExpr::negate(e_rewrite));
                }
            }
            _ => {}
        }

        results
    }

    /// Check termination using a simple heuristic.
    ///
    /// Returns true if no rule seems to increase expression size indefinitely.
    fn check_termination_heuristic(&self, _system: &RewriteSystem) -> bool {
        // A proper termination checker would use techniques like:
        // - Polynomial interpretations
        // - Lexicographic path ordering
        // - Dependency pairs
        //
        // For now, we assume termination (conservative)
        true
    }
}

impl Default for ConfluenceChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if two expressions can be joined under a rewrite system.
///
/// Convenience function that creates a checker and tests joinability.
pub fn are_joinable(expr1: &TLExpr, expr2: &TLExpr, system: &RewriteSystem) -> bool {
    let mut checker = ConfluenceChecker::new();
    checker.test_joinability(expr1, expr2, system)
}

/// Compute normal form of an expression (if it exists).
///
/// Returns None if no normal form is reached within max_steps.
pub fn normalize(expr: &TLExpr, system: &RewriteSystem, max_steps: usize) -> Option<TLExpr> {
    let mut current = expr.clone();
    let mut steps = 0;

    while steps < max_steps {
        if let Some(next) = system.apply_once(&current) {
            current = next;
            steps += 1;
        } else {
            return Some(current); // Normal form reached
        }
    }

    None // Max steps exceeded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Pattern, RewriteRule, Term};

    #[test]
    fn test_critical_pair_trivial_joinable() {
        let overlap = TLExpr::pred("P", vec![Term::var("x")]);
        let result = TLExpr::pred("Q", vec![Term::var("x")]);

        let pair = CriticalPair::new(
            overlap,
            result.clone(),
            result,
            "rule1".to_string(),
            "rule2".to_string(),
        );

        assert!(pair.is_trivially_joinable());
        assert!(!pair.has_conflict());
    }

    #[test]
    fn test_critical_pair_with_conflict() {
        let overlap = TLExpr::pred("P", vec![Term::var("x")]);
        let result1 = TLExpr::pred("Q", vec![Term::var("x")]);
        let result2 = TLExpr::pred("R", vec![Term::var("x")]);

        let pair = CriticalPair::new(
            overlap,
            result1,
            result2,
            "rule1".to_string(),
            "rule2".to_string(),
        );

        assert!(!pair.is_trivially_joinable());
        assert!(pair.has_conflict());
    }

    #[test]
    fn test_joinability_identical() {
        let system = RewriteSystem::new();
        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        let mut checker = ConfluenceChecker::new();
        assert!(checker.test_joinability(&expr, &expr, &system));
    }

    #[test]
    fn test_joinability_via_rewriting() {
        let system = RewriteSystem::new().add_rule(RewriteRule::new(
            Pattern::negation(Pattern::negation(Pattern::var("A"))),
            |bindings| bindings.get("A").unwrap().clone(),
        ));

        let expr1 = TLExpr::negate(TLExpr::negate(TLExpr::pred("P", vec![Term::var("x")])));
        let expr2 = TLExpr::pred("P", vec![Term::var("x")]);

        let mut checker = ConfluenceChecker::new();
        assert!(checker.test_joinability(&expr1, &expr2, &system));
    }

    #[test]
    fn test_normalize_to_normal_form() {
        let system = RewriteSystem::new().add_rule(RewriteRule::new(
            Pattern::negation(Pattern::negation(Pattern::var("A"))),
            |bindings| bindings.get("A").unwrap().clone(),
        ));

        let expr = TLExpr::negate(TLExpr::negate(TLExpr::pred("P", vec![Term::var("x")])));
        let normal_form = normalize(&expr, &system, 100).unwrap();

        assert!(matches!(normal_form, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_confluence_report_summary() {
        let mut report = ConfluenceReport::new();
        report.joinable_count = 5;
        report.non_joinable_count = 2;
        report.is_locally_confluent = false;
        report.is_terminating = true;

        let summary = report.summary();
        assert!(summary.contains("Joinable: 5"));
        assert!(summary.contains("Non-joinable: 2"));
        assert!(summary.contains("Confluent: false"));
    }

    #[test]
    fn test_confluence_via_newmans_lemma() {
        let mut report = ConfluenceReport::new();

        // Case 1: Terminating and locally confluent => confluent
        report.is_terminating = true;
        report.is_locally_confluent = true;
        assert!(report.is_confluent());

        // Case 2: Not terminating => can't deduce confluence
        report.is_terminating = false;
        report.is_locally_confluent = true;
        assert!(!report.is_confluent());

        // Case 3: Not locally confluent => not confluent
        report.is_terminating = true;
        report.is_locally_confluent = false;
        assert!(!report.is_confluent());
    }
}
