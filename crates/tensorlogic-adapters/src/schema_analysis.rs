//! Schema analysis and statistics.
//!
//! This module provides tools for analyzing symbol tables and generating
//! insights about schema structure, complexity, and usage patterns.

use std::collections::{HashMap, HashSet};

use crate::SymbolTable;

/// Comprehensive statistics about a schema.
#[derive(Clone, Debug)]
pub struct SchemaStatistics {
    /// Number of domains defined.
    pub domain_count: usize,
    /// Number of predicates defined.
    pub predicate_count: usize,
    /// Number of variables bound.
    pub variable_count: usize,
    /// Total cardinality across all domains.
    pub total_cardinality: usize,
    /// Average domain cardinality.
    pub avg_cardinality: f64,
    /// Maximum domain cardinality.
    pub max_cardinality: usize,
    /// Minimum domain cardinality.
    pub min_cardinality: usize,
    /// Distribution of predicate arities.
    pub arity_distribution: HashMap<usize, usize>,
    /// Most common domain types in predicates.
    pub domain_usage_frequency: HashMap<String, usize>,
    /// Domains that are never used in predicates.
    pub unused_domains: Vec<String>,
    /// Predicates grouped by arity.
    pub predicates_by_arity: HashMap<usize, Vec<String>>,
}

impl SchemaStatistics {
    /// Compute statistics for a symbol table.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo, SchemaStatistics};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_predicate(PredicateInfo::new(
    ///     "knows",
    ///     vec!["Person".to_string(), "Person".to_string()]
    /// )).unwrap();
    ///
    /// let stats = SchemaStatistics::compute(&table);
    /// assert_eq!(stats.domain_count, 1);
    /// assert_eq!(stats.predicate_count, 1);
    /// ```
    pub fn compute(table: &SymbolTable) -> Self {
        let domain_count = table.domains.len();
        let predicate_count = table.predicates.len();
        let variable_count = table.variables.len();

        // Compute cardinality statistics
        let cardinalities: Vec<usize> = table.domains.values().map(|d| d.cardinality).collect();
        let total_cardinality: usize = cardinalities.iter().sum();
        let avg_cardinality = if domain_count > 0 {
            total_cardinality as f64 / domain_count as f64
        } else {
            0.0
        };
        let max_cardinality = cardinalities.iter().copied().max().unwrap_or(0);
        let min_cardinality = cardinalities.iter().copied().min().unwrap_or(0);

        // Compute arity distribution
        let mut arity_distribution = HashMap::new();
        let mut predicates_by_arity: HashMap<usize, Vec<String>> = HashMap::new();
        for (name, pred) in &table.predicates {
            let arity = pred.arg_domains.len();
            *arity_distribution.entry(arity).or_insert(0) += 1;
            predicates_by_arity
                .entry(arity)
                .or_default()
                .push(name.clone());
        }

        // Compute domain usage frequency
        let mut domain_usage_frequency = HashMap::new();
        for pred in table.predicates.values() {
            for domain in &pred.arg_domains {
                *domain_usage_frequency.entry(domain.clone()).or_insert(0) += 1;
            }
        }

        // Find unused domains
        let used_domains: HashSet<_> = domain_usage_frequency.keys().cloned().collect();
        let unused_domains: Vec<String> = table
            .domains
            .keys()
            .filter(|d| !used_domains.contains(*d))
            .cloned()
            .collect();

        Self {
            domain_count,
            predicate_count,
            variable_count,
            total_cardinality,
            avg_cardinality,
            max_cardinality,
            min_cardinality,
            arity_distribution,
            domain_usage_frequency,
            unused_domains,
            predicates_by_arity,
        }
    }

    /// Get the most frequently used domains.
    pub fn most_used_domains(&self, n: usize) -> Vec<(String, usize)> {
        let mut usage: Vec<_> = self.domain_usage_frequency.iter().collect();
        usage.sort_by(|a, b| b.1.cmp(a.1));
        usage
            .into_iter()
            .take(n)
            .map(|(d, &count)| (d.clone(), count))
            .collect()
    }

    /// Get the least frequently used domains (excluding unused).
    pub fn least_used_domains(&self, n: usize) -> Vec<(String, usize)> {
        let mut usage: Vec<_> = self.domain_usage_frequency.iter().collect();
        usage.sort_by(|a, b| a.1.cmp(b.1));
        usage
            .into_iter()
            .take(n)
            .map(|(d, &count)| (d.clone(), count))
            .collect()
    }

    /// Calculate schema complexity score.
    ///
    /// This is a heuristic metric based on:
    /// - Number of domains
    /// - Number of predicates
    /// - Arity distribution
    /// - Domain usage patterns
    pub fn complexity_score(&self) -> f64 {
        let domain_factor = self.domain_count as f64;
        let predicate_factor = self.predicate_count as f64;
        let arity_diversity = self.arity_distribution.len() as f64;
        let usage_variance = self.compute_usage_variance();

        // Weighted combination
        domain_factor * 0.2 + predicate_factor * 0.3 + arity_diversity * 0.2 + usage_variance * 0.3
    }

    fn compute_usage_variance(&self) -> f64 {
        if self.domain_usage_frequency.is_empty() {
            return 0.0;
        }

        let counts: Vec<f64> = self
            .domain_usage_frequency
            .values()
            .map(|&c| c as f64)
            .collect();
        let mean = counts.iter().sum::<f64>() / counts.len() as f64;
        let variance = counts.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / counts.len() as f64;
        variance.sqrt()
    }
}

/// Schema recommendations based on analysis.
#[derive(Clone, Debug)]
pub struct SchemaRecommendations {
    /// Issues found in the schema.
    pub issues: Vec<SchemaIssue>,
    /// Suggestions for improvement.
    pub suggestions: Vec<String>,
}

/// Types of schema issues that can be detected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchemaIssue {
    /// Domain is never used in any predicate.
    UnusedDomain(String),
    /// Domain has zero cardinality.
    ZeroCardinalityDomain(String),
    /// Very high cardinality that might cause performance issues.
    HighCardinalityDomain(String, usize),
    /// Predicate with unusually high arity.
    HighArityPredicate(String, usize),
    /// No predicates defined.
    NoPredicates,
    /// No domains defined.
    NoDomains,
}

impl SchemaIssue {
    /// Get a human-readable description of the issue.
    pub fn description(&self) -> String {
        match self {
            Self::UnusedDomain(name) => format!("Domain '{}' is defined but never used", name),
            Self::ZeroCardinalityDomain(name) => {
                format!("Domain '{}' has zero cardinality", name)
            }
            Self::HighCardinalityDomain(name, card) => {
                format!(
                    "Domain '{}' has very high cardinality ({}), which may impact performance",
                    name, card
                )
            }
            Self::HighArityPredicate(name, arity) => {
                format!(
                    "Predicate '{}' has high arity ({}), consider decomposition",
                    name, arity
                )
            }
            Self::NoPredicates => "Schema has no predicates defined".to_string(),
            Self::NoDomains => "Schema has no domains defined".to_string(),
        }
    }

    /// Get the severity level (1=info, 2=warning, 3=error).
    pub fn severity(&self) -> u8 {
        match self {
            Self::UnusedDomain(_) => 1,
            Self::ZeroCardinalityDomain(_) => 2,
            Self::HighCardinalityDomain(_, _) => 1,
            Self::HighArityPredicate(_, _) => 1,
            Self::NoPredicates => 2,
            Self::NoDomains => 3,
        }
    }
}

/// Analyzer for generating schema recommendations.
pub struct SchemaAnalyzer;

impl SchemaAnalyzer {
    /// Analyze a symbol table and generate recommendations.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, SchemaAnalyzer};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 0)).unwrap();
    ///
    /// let recommendations = SchemaAnalyzer::analyze(&table);
    /// assert!(!recommendations.issues.is_empty());
    /// ```
    pub fn analyze(table: &SymbolTable) -> SchemaRecommendations {
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();

        // Check for no domains
        if table.domains.is_empty() {
            issues.push(SchemaIssue::NoDomains);
            suggestions.push("Define at least one domain for your schema".to_string());
            return SchemaRecommendations {
                issues,
                suggestions,
            };
        }

        // Check for no predicates
        if table.predicates.is_empty() {
            issues.push(SchemaIssue::NoPredicates);
            suggestions.push("Define predicates to enable reasoning over your domains".to_string());
        }

        // Analyze domains
        let stats = SchemaStatistics::compute(table);
        const HIGH_CARDINALITY_THRESHOLD: usize = 100_000;

        for (name, domain) in &table.domains {
            // Check for zero cardinality
            if domain.cardinality == 0 {
                issues.push(SchemaIssue::ZeroCardinalityDomain(name.clone()));
            }

            // Check for high cardinality
            if domain.cardinality > HIGH_CARDINALITY_THRESHOLD {
                issues.push(SchemaIssue::HighCardinalityDomain(
                    name.clone(),
                    domain.cardinality,
                ));
            }

            // Check for unused domains
            if stats.unused_domains.contains(name) {
                issues.push(SchemaIssue::UnusedDomain(name.clone()));
                suggestions.push(format!(
                    "Consider removing unused domain '{}' or defining predicates that use it",
                    name
                ));
            }
        }

        // Analyze predicates
        const HIGH_ARITY_THRESHOLD: usize = 5;
        for (name, pred) in &table.predicates {
            if pred.arg_domains.len() > HIGH_ARITY_THRESHOLD {
                issues.push(SchemaIssue::HighArityPredicate(
                    name.clone(),
                    pred.arg_domains.len(),
                ));
                suggestions.push(format!(
                    "Consider decomposing high-arity predicate '{}' into smaller predicates",
                    name
                ));
            }
        }

        // General suggestions
        if stats.domain_count > 0 && stats.predicate_count == 0 {
            suggestions
                .push("Add predicates to establish relationships between your domains".to_string());
        }

        if stats.variable_count == 0 && stats.predicate_count > 0 {
            suggestions
                .push("Consider binding variables to enable quantification in rules".to_string());
        }

        SchemaRecommendations {
            issues,
            suggestions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DomainInfo, PredicateInfo};

    #[test]
    fn test_statistics_empty_table() {
        let table = SymbolTable::new();
        let stats = SchemaStatistics::compute(&table);

        assert_eq!(stats.domain_count, 0);
        assert_eq!(stats.predicate_count, 0);
        assert_eq!(stats.variable_count, 0);
    }

    #[test]
    fn test_statistics_with_data() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Location", 50)).unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "knows",
                vec!["Person".into(), "Person".into()],
            ))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "at",
                vec!["Person".into(), "Location".into()],
            ))
            .unwrap();

        let stats = SchemaStatistics::compute(&table);

        assert_eq!(stats.domain_count, 2);
        assert_eq!(stats.predicate_count, 2);
        assert_eq!(stats.total_cardinality, 150);
        assert_eq!(stats.avg_cardinality, 75.0);
        assert_eq!(stats.max_cardinality, 100);
        assert_eq!(stats.min_cardinality, 50);

        // Check domain usage
        assert_eq!(stats.domain_usage_frequency.get("Person"), Some(&3));
        assert_eq!(stats.domain_usage_frequency.get("Location"), Some(&1));
        assert!(stats.unused_domains.is_empty());
    }

    #[test]
    fn test_unused_domains() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Unused", 50)).unwrap();
        table
            .add_predicate(PredicateInfo::new("age", vec!["Person".into()]))
            .unwrap();

        let stats = SchemaStatistics::compute(&table);
        assert_eq!(stats.unused_domains, vec!["Unused"]);
    }

    #[test]
    fn test_arity_distribution() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("D", 10)).unwrap();
        table
            .add_predicate(PredicateInfo::new("p1", vec!["D".into()]))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new("p2", vec!["D".into(), "D".into()]))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new("p3", vec!["D".into(), "D".into()]))
            .unwrap();

        let stats = SchemaStatistics::compute(&table);
        assert_eq!(stats.arity_distribution.get(&1), Some(&1));
        assert_eq!(stats.arity_distribution.get(&2), Some(&2));
    }

    #[test]
    fn test_analyzer_no_domains() {
        let table = SymbolTable::new();
        let recs = SchemaAnalyzer::analyze(&table);

        assert!(!recs.issues.is_empty());
        assert!(recs.issues.contains(&SchemaIssue::NoDomains));
    }

    #[test]
    fn test_analyzer_zero_cardinality() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 0)).unwrap();

        let recs = SchemaAnalyzer::analyze(&table);
        assert!(recs
            .issues
            .contains(&SchemaIssue::ZeroCardinalityDomain("Person".to_string())));
    }

    #[test]
    fn test_analyzer_unused_domain() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Used", 10)).unwrap();
        table.add_domain(DomainInfo::new("Unused", 10)).unwrap();
        table
            .add_predicate(PredicateInfo::new("p", vec!["Used".into()]))
            .unwrap();

        let recs = SchemaAnalyzer::analyze(&table);
        assert!(recs
            .issues
            .contains(&SchemaIssue::UnusedDomain("Unused".to_string())));
    }

    #[test]
    fn test_analyzer_high_arity() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("D", 10)).unwrap();
        let args = vec!["D".to_string(); 10]; // 10-arity predicate
        table
            .add_predicate(PredicateInfo::new("complex", args))
            .unwrap();

        let recs = SchemaAnalyzer::analyze(&table);
        assert!(recs
            .issues
            .iter()
            .any(|i| matches!(i, SchemaIssue::HighArityPredicate(_, _))));
    }

    #[test]
    fn test_complexity_score() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table
            .add_predicate(PredicateInfo::new("p", vec!["Person".into()]))
            .unwrap();

        let stats = SchemaStatistics::compute(&table);
        let score = stats.complexity_score();
        assert!(score > 0.0);
    }

    #[test]
    fn test_most_used_domains() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("A", 10)).unwrap();
        table.add_domain(DomainInfo::new("B", 10)).unwrap();
        table
            .add_predicate(PredicateInfo::new("p1", vec!["A".into(), "A".into()]))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new("p2", vec!["B".into()]))
            .unwrap();

        let stats = SchemaStatistics::compute(&table);
        let most_used = stats.most_used_domains(1);
        assert_eq!(most_used[0].0, "A");
        assert_eq!(most_used[0].1, 2);
    }
}
