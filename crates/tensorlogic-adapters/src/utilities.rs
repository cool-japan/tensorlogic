//! Advanced utility functions for tensorlogic-adapters.
//!
//! This module provides helpful utility functions for common operations
//! on symbol tables, domains, predicates, and related structures.

use crate::{DomainInfo, PredicateInfo, SchemaStatistics, SymbolTable, ValidationReport};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Batch operations for efficient bulk processing.
pub struct BatchOperations;

impl BatchOperations {
    /// Add multiple domains at once with validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, BatchOperations};
    ///
    /// let mut table = SymbolTable::new();
    /// let domains = vec![
    ///     DomainInfo::new("Person", 100),
    ///     DomainInfo::new("Organization", 50),
    /// ];
    ///
    /// let result = BatchOperations::add_domains(&mut table, domains);
    /// assert!(result.is_ok());
    /// assert_eq!(table.domains.len(), 2);
    /// ```
    pub fn add_domains(table: &mut SymbolTable, domains: Vec<DomainInfo>) -> Result<()> {
        for domain in domains {
            table.add_domain(domain)?;
        }
        Ok(())
    }

    /// Add multiple predicates at once with validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo, BatchOperations};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let predicates = vec![
    ///     PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()]),
    ///     PredicateInfo::new("age", vec!["Person".to_string()]),
    /// ];
    ///
    /// let result = BatchOperations::add_predicates(&mut table, predicates);
    /// assert!(result.is_ok());
    /// assert_eq!(table.predicates.len(), 2);
    /// ```
    pub fn add_predicates(table: &mut SymbolTable, predicates: Vec<PredicateInfo>) -> Result<()> {
        for predicate in predicates {
            table.add_predicate(predicate)?;
        }
        Ok(())
    }

    /// Bind multiple variables at once.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, BatchOperations};
    /// use std::collections::HashMap;
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let mut bindings = HashMap::new();
    /// bindings.insert("x".to_string(), "Person".to_string());
    /// bindings.insert("y".to_string(), "Person".to_string());
    ///
    /// let result = BatchOperations::bind_variables(&mut table, bindings);
    /// assert!(result.is_ok());
    /// assert_eq!(table.variables.len(), 2);
    /// ```
    pub fn bind_variables(
        table: &mut SymbolTable,
        bindings: HashMap<String, String>,
    ) -> Result<()> {
        for (var, domain) in bindings {
            table.bind_variable(var, domain)?;
        }
        Ok(())
    }
}

/// Conversion utilities for different formats.
pub struct ConversionUtils;

impl ConversionUtils {
    /// Convert a symbol table to a compact summary string.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, ConversionUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let summary = ConversionUtils::to_summary(&table);
    /// assert!(summary.contains("Domains: 1"));
    /// ```
    pub fn to_summary(table: &SymbolTable) -> String {
        format!(
            "SymbolTable Summary:\n  Domains: {}\n  Predicates: {}\n  Variables: {}",
            table.domains.len(),
            table.predicates.len(),
            table.variables.len()
        )
    }

    /// Extract domain names as a sorted vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, ConversionUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_domain(DomainInfo::new("Organization", 50)).unwrap();
    ///
    /// let names = ConversionUtils::extract_domain_names(&table);
    /// assert_eq!(names, vec!["Organization", "Person"]);
    /// ```
    pub fn extract_domain_names(table: &SymbolTable) -> Vec<String> {
        let mut names: Vec<String> = table.domains.keys().cloned().collect();
        names.sort();
        names
    }

    /// Extract predicate names as a sorted vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo, ConversionUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_predicate(PredicateInfo::new("knows", vec!["Person".to_string()])).unwrap();
    ///
    /// let names = ConversionUtils::extract_predicate_names(&table);
    /// assert_eq!(names, vec!["knows"]);
    /// ```
    pub fn extract_predicate_names(table: &SymbolTable) -> Vec<String> {
        let mut names: Vec<String> = table.predicates.keys().cloned().collect();
        names.sort();
        names
    }
}

/// Query utilities for advanced filtering and searching.
pub struct QueryUtils;

impl QueryUtils {
    /// Find all predicates that use a specific domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo, QueryUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_predicate(PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])).unwrap();
    ///
    /// let predicates = QueryUtils::find_predicates_using_domain(&table, "Person");
    /// assert_eq!(predicates.len(), 1);
    /// assert_eq!(predicates[0].name, "knows");
    /// ```
    pub fn find_predicates_using_domain(
        table: &SymbolTable,
        domain_name: &str,
    ) -> Vec<PredicateInfo> {
        table
            .predicates
            .values()
            .filter(|p| p.arg_domains.contains(&domain_name.to_string()))
            .cloned()
            .collect()
    }

    /// Find all domains that are never used by any predicate.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, QueryUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_domain(DomainInfo::new("Unused", 10)).unwrap();
    ///
    /// let unused = QueryUtils::find_unused_domains(&table);
    /// assert_eq!(unused.len(), 2); // Both are unused as no predicates defined
    /// ```
    pub fn find_unused_domains(table: &SymbolTable) -> Vec<String> {
        let used_domains: HashSet<&String> = table
            .predicates
            .values()
            .flat_map(|p| &p.arg_domains)
            .collect();

        table
            .domains
            .keys()
            .filter(|d| !used_domains.contains(d))
            .cloned()
            .collect()
    }

    /// Find predicates with a specific arity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo, QueryUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_predicate(PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])).unwrap();
    /// table.add_predicate(PredicateInfo::new("age", vec!["Person".to_string()])).unwrap();
    ///
    /// let binary = QueryUtils::find_predicates_by_arity(&table, 2);
    /// assert_eq!(binary.len(), 1);
    /// assert_eq!(binary[0].name, "knows");
    /// ```
    pub fn find_predicates_by_arity(table: &SymbolTable, arity: usize) -> Vec<PredicateInfo> {
        table
            .predicates
            .values()
            .filter(|p| p.arg_domains.len() == arity)
            .cloned()
            .collect()
    }

    /// Group predicates by their arity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo, QueryUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_predicate(PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])).unwrap();
    /// table.add_predicate(PredicateInfo::new("age", vec!["Person".to_string()])).unwrap();
    ///
    /// let grouped = QueryUtils::group_predicates_by_arity(&table);
    /// assert_eq!(grouped.get(&1).unwrap().len(), 1);
    /// assert_eq!(grouped.get(&2).unwrap().len(), 1);
    /// ```
    pub fn group_predicates_by_arity(table: &SymbolTable) -> HashMap<usize, Vec<PredicateInfo>> {
        let mut groups: HashMap<usize, Vec<PredicateInfo>> = HashMap::new();
        for predicate in table.predicates.values() {
            let arity = predicate.arg_domains.len();
            groups.entry(arity).or_default().push(predicate.clone());
        }
        groups
    }
}

/// Validation utilities for enhanced checking.
pub struct ValidationUtils;

impl ValidationUtils {
    /// Quick validation check (returns true if valid).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, ValidationUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// assert!(ValidationUtils::is_valid(&table));
    /// ```
    pub fn is_valid(table: &SymbolTable) -> bool {
        // Check if all predicates reference existing domains
        for predicate in table.predicates.values() {
            for domain in &predicate.arg_domains {
                if !table.domains.contains_key(domain) {
                    return false;
                }
            }
        }

        // Check if all variables reference existing domains
        for domain in table.variables.values() {
            if !table.domains.contains_key(domain) {
                return false;
            }
        }

        true
    }

    /// Get detailed validation report.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, ValidationUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let report = ValidationUtils::detailed_validation(&table);
    /// assert!(report.is_ok());
    /// ```
    pub fn detailed_validation(table: &SymbolTable) -> Result<ValidationReport> {
        use crate::SchemaValidator;
        let validator = SchemaValidator::new(table);
        validator.validate()
    }

    /// Check if a specific domain is used.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo, ValidationUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_predicate(PredicateInfo::new("knows", vec!["Person".to_string()])).unwrap();
    ///
    /// assert!(ValidationUtils::is_domain_used(&table, "Person"));
    /// assert!(!ValidationUtils::is_domain_used(&table, "Nonexistent"));
    /// ```
    pub fn is_domain_used(table: &SymbolTable, domain_name: &str) -> bool {
        // Check predicates
        for predicate in table.predicates.values() {
            if predicate.arg_domains.contains(&domain_name.to_string()) {
                return true;
            }
        }

        // Check variables
        for domain in table.variables.values() {
            if domain == domain_name {
                return true;
            }
        }

        false
    }
}

/// Statistics utilities for metrics collection.
pub struct StatisticsUtils;

impl StatisticsUtils {
    /// Compute comprehensive statistics for a symbol table.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, StatisticsUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let stats = StatisticsUtils::compute_statistics(&table);
    /// assert!(stats.is_ok());
    /// ```
    pub fn compute_statistics(table: &SymbolTable) -> Result<SchemaStatistics> {
        Ok(SchemaStatistics::compute(table))
    }

    /// Get total domain cardinality.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, StatisticsUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_domain(DomainInfo::new("Organization", 50)).unwrap();
    ///
    /// let total = StatisticsUtils::total_domain_cardinality(&table);
    /// assert_eq!(total, 150);
    /// ```
    pub fn total_domain_cardinality(table: &SymbolTable) -> usize {
        table.domains.values().map(|d| d.cardinality).sum()
    }

    /// Get average predicate arity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo, StatisticsUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_predicate(PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])).unwrap();
    /// table.add_predicate(PredicateInfo::new("age", vec!["Person".to_string()])).unwrap();
    ///
    /// let avg = StatisticsUtils::average_predicate_arity(&table);
    /// assert_eq!(avg, 1.5);
    /// ```
    pub fn average_predicate_arity(table: &SymbolTable) -> f64 {
        if table.predicates.is_empty() {
            return 0.0;
        }

        let total: usize = table.predicates.values().map(|p| p.arg_domains.len()).sum();

        total as f64 / table.predicates.len() as f64
    }

    /// Get domain usage counts.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo, StatisticsUtils};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_predicate(PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])).unwrap();
    ///
    /// let usage = StatisticsUtils::domain_usage_counts(&table);
    /// assert_eq!(usage.get("Person"), Some(&2));
    /// ```
    pub fn domain_usage_counts(table: &SymbolTable) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();

        for predicate in table.predicates.values() {
            for domain in &predicate.arg_domains {
                *counts.entry(domain.clone()).or_insert(0) += 1;
            }
        }

        for domain in table.variables.values() {
            *counts.entry(domain.clone()).or_insert(0) += 1;
        }

        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_table() -> SymbolTable {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table
            .add_domain(DomainInfo::new("Organization", 50))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "knows",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new("age", vec!["Person".to_string()]))
            .unwrap();
        table.bind_variable("x", "Person").unwrap();
        table
    }

    #[test]
    fn test_batch_add_domains() {
        let mut table = SymbolTable::new();
        let domains = vec![
            DomainInfo::new("Person", 100),
            DomainInfo::new("Organization", 50),
        ];

        BatchOperations::add_domains(&mut table, domains).unwrap();
        assert_eq!(table.domains.len(), 2);
    }

    #[test]
    fn test_conversion_summary() {
        let table = create_test_table();
        let summary = ConversionUtils::to_summary(&table);
        assert!(summary.contains("Domains: 2"));
        assert!(summary.contains("Predicates: 2"));
    }

    #[test]
    fn test_query_predicates_using_domain() {
        let table = create_test_table();
        let predicates = QueryUtils::find_predicates_using_domain(&table, "Person");
        assert_eq!(predicates.len(), 2);
    }

    #[test]
    fn test_query_by_arity() {
        let table = create_test_table();
        let unary = QueryUtils::find_predicates_by_arity(&table, 1);
        let binary = QueryUtils::find_predicates_by_arity(&table, 2);
        assert_eq!(unary.len(), 1);
        assert_eq!(binary.len(), 1);
    }

    #[test]
    fn test_validation_is_valid() {
        let table = create_test_table();
        assert!(ValidationUtils::is_valid(&table));
    }

    #[test]
    fn test_statistics_total_cardinality() {
        let table = create_test_table();
        let total = StatisticsUtils::total_domain_cardinality(&table);
        assert_eq!(total, 150);
    }

    #[test]
    fn test_statistics_average_arity() {
        let table = create_test_table();
        let avg = StatisticsUtils::average_predicate_arity(&table);
        assert_eq!(avg, 1.5);
    }

    #[test]
    fn test_domain_usage_counts() {
        let table = create_test_table();
        let counts = StatisticsUtils::domain_usage_counts(&table);
        assert_eq!(counts.get("Person"), Some(&4)); // 2 in knows + 1 in age + 1 variable
    }

    #[test]
    fn test_group_by_arity() {
        let table = create_test_table();
        let groups = QueryUtils::group_predicates_by_arity(&table);
        assert_eq!(groups.len(), 2);
        assert!(groups.contains_key(&1));
        assert!(groups.contains_key(&2));
    }

    #[test]
    fn test_extract_names() {
        let table = create_test_table();
        let domain_names = ConversionUtils::extract_domain_names(&table);
        let predicate_names = ConversionUtils::extract_predicate_names(&table);

        assert_eq!(domain_names.len(), 2);
        assert_eq!(predicate_names.len(), 2);
        assert!(domain_names.contains(&"Person".to_string()));
        assert!(predicate_names.contains(&"knows".to_string()));
    }
}
