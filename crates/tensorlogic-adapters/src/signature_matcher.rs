//! Optimized predicate signature matching.
//!
//! This module provides fast lookup and matching of predicates based on their
//! signatures (arity and domain types). It uses indexed data structures for
//! O(1) lookups instead of linear scans.

use std::collections::HashMap;

use crate::PredicateInfo;

/// Indexed structure for fast predicate signature matching.
///
/// This provides O(1) lookup of predicates by:
/// - Arity (number of arguments)
/// - Exact signature (ordered domain types)
/// - Domain patterns (unordered domain types)
///
/// # Example
///
/// ```rust
/// use tensorlogic_adapters::{PredicateInfo, SignatureMatcher};
///
/// let mut matcher = SignatureMatcher::new();
///
/// let knows = PredicateInfo::new(
///     "knows",
///     vec!["Person".to_string(), "Person".to_string()]
/// );
/// matcher.add_predicate(&knows);
///
/// // Find by arity
/// let arity_2 = matcher.find_by_arity(2);
/// assert_eq!(arity_2.len(), 1);
///
/// // Find by exact signature
/// let signature = vec!["Person".to_string(), "Person".to_string()];
/// let matches = matcher.find_by_signature(&signature);
/// assert_eq!(matches.len(), 1);
/// assert_eq!(matches[0], "knows");
/// ```
#[derive(Clone, Debug, Default)]
pub struct SignatureMatcher {
    /// Index by arity: arity -> [predicate_names]
    by_arity: HashMap<usize, Vec<String>>,
    /// Index by exact signature: signature -> [predicate_names]
    by_signature: HashMap<Vec<String>, Vec<String>>,
    /// Index by sorted signature (for unordered matching): sorted_sig -> [predicate_names]
    by_domain_set: HashMap<Vec<String>, Vec<String>>,
    /// Full predicate information
    predicates: HashMap<String, PredicateInfo>,
}

impl SignatureMatcher {
    /// Create a new signature matcher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a predicate to the matcher indices.
    pub fn add_predicate(&mut self, pred: &PredicateInfo) {
        let name = pred.name.clone();
        let arity = pred.arg_domains.len();
        let signature = pred.arg_domains.clone();

        // Index by arity
        self.by_arity.entry(arity).or_default().push(name.clone());

        // Index by exact signature
        self.by_signature
            .entry(signature.clone())
            .or_default()
            .push(name.clone());

        // Index by sorted domain set (for unordered matching)
        let mut sorted_sig = signature.clone();
        sorted_sig.sort();
        self.by_domain_set
            .entry(sorted_sig)
            .or_default()
            .push(name.clone());

        // Store full predicate info
        self.predicates.insert(name, pred.clone());
    }

    /// Remove a predicate from all indices.
    pub fn remove_predicate(&mut self, name: &str) {
        if let Some(pred) = self.predicates.remove(name) {
            let arity = pred.arg_domains.len();
            let signature = pred.arg_domains.clone();

            // Remove from arity index
            if let Some(names) = self.by_arity.get_mut(&arity) {
                names.retain(|n| n != name);
                if names.is_empty() {
                    self.by_arity.remove(&arity);
                }
            }

            // Remove from signature index
            if let Some(names) = self.by_signature.get_mut(&signature) {
                names.retain(|n| n != name);
                if names.is_empty() {
                    self.by_signature.remove(&signature);
                }
            }

            // Remove from domain set index
            let mut sorted_sig = signature;
            sorted_sig.sort();
            if let Some(names) = self.by_domain_set.get_mut(&sorted_sig) {
                names.retain(|n| n != name);
                if names.is_empty() {
                    self.by_domain_set.remove(&sorted_sig);
                }
            }
        }
    }

    /// Find all predicates with the given arity.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{PredicateInfo, SignatureMatcher};
    ///
    /// let mut matcher = SignatureMatcher::new();
    /// matcher.add_predicate(&PredicateInfo::new("knows", vec!["Person".into(), "Person".into()]));
    /// matcher.add_predicate(&PredicateInfo::new("age", vec!["Person".into()]));
    ///
    /// let unary = matcher.find_by_arity(1);
    /// assert_eq!(unary.len(), 1);
    /// assert!(unary.contains(&"age".to_string()));
    /// ```
    pub fn find_by_arity(&self, arity: usize) -> Vec<String> {
        self.by_arity.get(&arity).cloned().unwrap_or_default()
    }

    /// Find all predicates with the exact signature (ordered domain types).
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{PredicateInfo, SignatureMatcher};
    ///
    /// let mut matcher = SignatureMatcher::new();
    /// matcher.add_predicate(&PredicateInfo::new("at", vec!["Person".into(), "Location".into()]));
    ///
    /// let sig = vec!["Person".to_string(), "Location".to_string()];
    /// let matches = matcher.find_by_signature(&sig);
    /// assert_eq!(matches, vec!["at"]);
    /// ```
    pub fn find_by_signature(&self, signature: &[String]) -> Vec<String> {
        self.by_signature
            .get(signature)
            .cloned()
            .unwrap_or_default()
    }

    /// Find all predicates with the given domain types (unordered).
    ///
    /// This is useful for finding predicates that operate on a set of domains
    /// regardless of argument order.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{PredicateInfo, SignatureMatcher};
    ///
    /// let mut matcher = SignatureMatcher::new();
    /// matcher.add_predicate(&PredicateInfo::new("knows", vec!["Person".into(), "Person".into()]));
    ///
    /// let domains = vec!["Person".to_string()];
    /// let matches = matcher.find_by_domain_set(&domains);
    /// // "knows" has signature [Person, Person], which when deduplicated matches [Person]
    /// // Note: This requires exact match of sorted signature
    /// ```
    pub fn find_by_domain_set(&self, domains: &[String]) -> Vec<String> {
        let mut sorted = domains.to_vec();
        sorted.sort();
        self.by_domain_set.get(&sorted).cloned().unwrap_or_default()
    }

    /// Get full predicate information by name.
    pub fn get_predicate(&self, name: &str) -> Option<&PredicateInfo> {
        self.predicates.get(name)
    }

    /// Check if a predicate exists.
    pub fn contains(&self, name: &str) -> bool {
        self.predicates.contains_key(name)
    }

    /// Get the total number of predicates indexed.
    pub fn len(&self) -> usize {
        self.predicates.len()
    }

    /// Check if the matcher is empty.
    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }

    /// Get all predicate names.
    pub fn predicate_names(&self) -> Vec<String> {
        self.predicates.keys().cloned().collect()
    }

    /// Clear all indices.
    pub fn clear(&mut self) {
        self.by_arity.clear();
        self.by_signature.clear();
        self.by_domain_set.clear();
        self.predicates.clear();
    }

    /// Get statistics about the index sizes.
    pub fn stats(&self) -> MatcherStats {
        MatcherStats {
            total_predicates: self.predicates.len(),
            unique_arities: self.by_arity.len(),
            unique_signatures: self.by_signature.len(),
            unique_domain_sets: self.by_domain_set.len(),
        }
    }

    /// Build a matcher from a collection of predicates.
    pub fn from_predicates<'a>(predicates: impl IntoIterator<Item = &'a PredicateInfo>) -> Self {
        let mut matcher = Self::new();
        for pred in predicates {
            matcher.add_predicate(pred);
        }
        matcher
    }
}

/// Statistics about the signature matcher indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatcherStats {
    /// Total number of predicates indexed.
    pub total_predicates: usize,
    /// Number of unique arities.
    pub unique_arities: usize,
    /// Number of unique exact signatures.
    pub unique_signatures: usize,
    /// Number of unique domain sets (unordered).
    pub unique_domain_sets: usize,
}

impl MatcherStats {
    /// Calculate the average index size.
    pub fn avg_index_size(&self) -> f64 {
        if self.unique_signatures == 0 {
            0.0
        } else {
            self.total_predicates as f64 / self.unique_signatures as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_find_by_arity() {
        let mut matcher = SignatureMatcher::new();

        let knows = PredicateInfo::new("knows", vec!["Person".into(), "Person".into()]);
        let age = PredicateInfo::new("age", vec!["Person".into()]);

        matcher.add_predicate(&knows);
        matcher.add_predicate(&age);

        let unary = matcher.find_by_arity(1);
        assert_eq!(unary.len(), 1);
        assert!(unary.contains(&"age".to_string()));

        let binary = matcher.find_by_arity(2);
        assert_eq!(binary.len(), 1);
        assert!(binary.contains(&"knows".to_string()));
    }

    #[test]
    fn test_find_by_exact_signature() {
        let mut matcher = SignatureMatcher::new();

        let at = PredicateInfo::new("at", vec!["Person".into(), "Location".into()]);
        matcher.add_predicate(&at);

        let sig = vec!["Person".to_string(), "Location".to_string()];
        let matches = matcher.find_by_signature(&sig);
        assert_eq!(matches, vec!["at"]);

        // Different order should not match
        let sig_reversed = vec!["Location".to_string(), "Person".to_string()];
        let no_matches = matcher.find_by_signature(&sig_reversed);
        assert!(no_matches.is_empty());
    }

    #[test]
    fn test_remove_predicate() {
        let mut matcher = SignatureMatcher::new();

        let knows = PredicateInfo::new("knows", vec!["Person".into(), "Person".into()]);
        matcher.add_predicate(&knows);

        assert_eq!(matcher.len(), 1);
        assert!(matcher.contains("knows"));

        matcher.remove_predicate("knows");
        assert_eq!(matcher.len(), 0);
        assert!(!matcher.contains("knows"));

        // Should be empty
        assert!(matcher.find_by_arity(2).is_empty());
    }

    #[test]
    fn test_multiple_predicates_same_signature() {
        let mut matcher = SignatureMatcher::new();

        let p1 = PredicateInfo::new("pred1", vec!["Person".into(), "Person".into()]);
        let p2 = PredicateInfo::new("pred2", vec!["Person".into(), "Person".into()]);

        matcher.add_predicate(&p1);
        matcher.add_predicate(&p2);

        let sig = vec!["Person".to_string(), "Person".to_string()];
        let matches = matcher.find_by_signature(&sig);
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&"pred1".to_string()));
        assert!(matches.contains(&"pred2".to_string()));
    }

    #[test]
    fn test_from_predicates() {
        let preds = vec![
            PredicateInfo::new("knows", vec!["Person".into(), "Person".into()]),
            PredicateInfo::new("age", vec!["Person".into()]),
        ];

        let matcher = SignatureMatcher::from_predicates(&preds);
        assert_eq!(matcher.len(), 2);
        assert!(matcher.contains("knows"));
        assert!(matcher.contains("age"));
    }

    #[test]
    fn test_stats() {
        let mut matcher = SignatureMatcher::new();

        matcher.add_predicate(&PredicateInfo::new("p1", vec!["A".into(), "B".into()]));
        matcher.add_predicate(&PredicateInfo::new("p2", vec!["A".into(), "B".into()]));
        matcher.add_predicate(&PredicateInfo::new("p3", vec!["C".into()]));

        let stats = matcher.stats();
        assert_eq!(stats.total_predicates, 3);
        assert_eq!(stats.unique_arities, 2); // arity 1 and 2
        assert_eq!(stats.unique_signatures, 2); // [A,B] and [C]
    }

    #[test]
    fn test_clear() {
        let mut matcher = SignatureMatcher::new();
        matcher.add_predicate(&PredicateInfo::new("p1", vec!["A".into()]));

        assert_eq!(matcher.len(), 1);
        matcher.clear();
        assert_eq!(matcher.len(), 0);
        assert!(matcher.is_empty());
    }
}
