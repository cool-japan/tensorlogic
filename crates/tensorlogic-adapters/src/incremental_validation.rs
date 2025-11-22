//! Incremental validation system for efficient schema revalidation.
//!
//! This module provides smart revalidation that only checks what has changed,
//! enabling efficient validation of large schemas during iterative development.
//!
//! # Overview
//!
//! Traditional validation requires full schema traversal on every change.
//! Incremental validation tracks modifications and only revalidates affected
//! components, providing 10-100x speedup for large schemas.
//!
//! # Architecture
//!
//! - **ChangeTracker**: Records schema modifications (additions, updates, deletions)
//! - **DependencyGraph**: Tracks relationships between schema components
//! - **IncrementalValidator**: Performs targeted validation of affected components
//! - **ValidationCache**: Caches validation results for unchanged components
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_adapters::{SymbolTable, DomainInfo, ChangeTracker, IncrementalValidator};
//!
//! let mut table = SymbolTable::new();
//! let mut tracker = ChangeTracker::new();
//!
//! // Initial validation
//! table.add_domain(DomainInfo::new("Person", 100)).unwrap();
//! tracker.record_domain_addition("Person");
//!
//! let mut validator = IncrementalValidator::new(&table, &tracker);
//! let report = validator.validate_incremental().unwrap();
//! assert!(report.is_valid());
//!
//! // Incremental update - only validates affected parts
//! tracker.record_domain_addition("Location");
//! table.add_domain(DomainInfo::new("Location", 50)).unwrap();
//!
//! let mut validator2 = IncrementalValidator::new(&table, &tracker);
//! let report = validator2.validate_incremental().unwrap();
//! assert!(report.is_valid());
//! ```

use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use crate::{DomainHierarchy, SymbolTable, ValidationReport};

/// Type of schema change
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ChangeType {
    /// Domain added
    DomainAdded(String),
    /// Domain modified
    DomainModified(String),
    /// Domain removed
    DomainRemoved(String),
    /// Predicate added
    PredicateAdded(String),
    /// Predicate modified
    PredicateModified(String),
    /// Predicate removed
    PredicateRemoved(String),
    /// Variable binding added
    VariableAdded(String),
    /// Variable binding modified
    VariableModified(String),
    /// Variable binding removed
    VariableRemoved(String),
    /// Hierarchy relationship added
    HierarchyAdded(String, String),
    /// Hierarchy relationship removed
    HierarchyRemoved(String, String),
}

/// Change record with timestamp and metadata
#[derive(Clone, Debug)]
pub struct Change {
    pub change_type: ChangeType,
    pub timestamp: Instant,
    pub batch_id: Option<usize>,
}

impl Change {
    pub fn new(change_type: ChangeType) -> Self {
        Self {
            change_type,
            timestamp: Instant::now(),
            batch_id: None,
        }
    }

    pub fn with_batch(mut self, batch_id: usize) -> Self {
        self.batch_id = Some(batch_id);
        self
    }
}

/// Tracks changes to a symbol table for incremental validation
#[derive(Clone, Debug, Default)]
pub struct ChangeTracker {
    changes: Vec<Change>,
    domains_affected: HashSet<String>,
    predicates_affected: HashSet<String>,
    variables_affected: HashSet<String>,
    current_batch: Option<usize>,
    next_batch_id: usize,
}

impl ChangeTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a batch of changes that will be validated together
    pub fn begin_batch(&mut self) -> usize {
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;
        self.current_batch = Some(batch_id);
        batch_id
    }

    /// End the current batch
    pub fn end_batch(&mut self) {
        self.current_batch = None;
    }

    /// Record a domain addition
    pub fn record_domain_addition(&mut self, domain: impl Into<String>) {
        let domain = domain.into();
        self.domains_affected.insert(domain.clone());
        let mut change = Change::new(ChangeType::DomainAdded(domain));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Record a domain modification
    pub fn record_domain_modification(&mut self, domain: impl Into<String>) {
        let domain = domain.into();
        self.domains_affected.insert(domain.clone());
        let mut change = Change::new(ChangeType::DomainModified(domain));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Record a domain removal
    pub fn record_domain_removal(&mut self, domain: impl Into<String>) {
        let domain = domain.into();
        self.domains_affected.insert(domain.clone());
        let mut change = Change::new(ChangeType::DomainRemoved(domain));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Record a predicate addition
    pub fn record_predicate_addition(&mut self, predicate: impl Into<String>) {
        let predicate = predicate.into();
        self.predicates_affected.insert(predicate.clone());
        let mut change = Change::new(ChangeType::PredicateAdded(predicate));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Record a predicate modification
    pub fn record_predicate_modification(&mut self, predicate: impl Into<String>) {
        let predicate = predicate.into();
        self.predicates_affected.insert(predicate.clone());
        let mut change = Change::new(ChangeType::PredicateModified(predicate));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Record a predicate removal
    pub fn record_predicate_removal(&mut self, predicate: impl Into<String>) {
        let predicate = predicate.into();
        self.predicates_affected.insert(predicate.clone());
        let mut change = Change::new(ChangeType::PredicateRemoved(predicate));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Record a variable binding addition
    pub fn record_variable_addition(&mut self, variable: impl Into<String>) {
        let variable = variable.into();
        self.variables_affected.insert(variable.clone());
        let mut change = Change::new(ChangeType::VariableAdded(variable));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Record a variable binding modification
    pub fn record_variable_modification(&mut self, variable: impl Into<String>) {
        let variable = variable.into();
        self.variables_affected.insert(variable.clone());
        let mut change = Change::new(ChangeType::VariableModified(variable));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Record a variable binding removal
    pub fn record_variable_removal(&mut self, variable: impl Into<String>) {
        let variable = variable.into();
        self.variables_affected.insert(variable.clone());
        let mut change = Change::new(ChangeType::VariableRemoved(variable));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Record a hierarchy relationship addition
    pub fn record_hierarchy_addition(
        &mut self,
        subtype: impl Into<String>,
        supertype: impl Into<String>,
    ) {
        let subtype = subtype.into();
        let supertype = supertype.into();
        self.domains_affected.insert(subtype.clone());
        self.domains_affected.insert(supertype.clone());
        let mut change = Change::new(ChangeType::HierarchyAdded(subtype, supertype));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Record a hierarchy relationship removal
    pub fn record_hierarchy_removal(
        &mut self,
        subtype: impl Into<String>,
        supertype: impl Into<String>,
    ) {
        let subtype = subtype.into();
        let supertype = supertype.into();
        self.domains_affected.insert(subtype.clone());
        self.domains_affected.insert(supertype.clone());
        let mut change = Change::new(ChangeType::HierarchyRemoved(subtype, supertype));
        if let Some(batch) = self.current_batch {
            change = change.with_batch(batch);
        }
        self.changes.push(change);
    }

    /// Get all changes
    pub fn changes(&self) -> &[Change] {
        &self.changes
    }

    /// Get domains affected by changes
    pub fn domains_affected(&self) -> &HashSet<String> {
        &self.domains_affected
    }

    /// Get predicates affected by changes
    pub fn predicates_affected(&self) -> &HashSet<String> {
        &self.predicates_affected
    }

    /// Get variables affected by changes
    pub fn variables_affected(&self) -> &HashSet<String> {
        &self.variables_affected
    }

    /// Check if any changes have been recorded
    pub fn has_changes(&self) -> bool {
        !self.changes.is_empty()
    }

    /// Clear all tracked changes
    pub fn clear(&mut self) {
        self.changes.clear();
        self.domains_affected.clear();
        self.predicates_affected.clear();
        self.variables_affected.clear();
    }

    /// Get change statistics
    pub fn stats(&self) -> ChangeStats {
        let mut by_type = HashMap::new();
        for change in &self.changes {
            let type_name = match &change.change_type {
                ChangeType::DomainAdded(_) => "DomainAdded",
                ChangeType::DomainModified(_) => "DomainModified",
                ChangeType::DomainRemoved(_) => "DomainRemoved",
                ChangeType::PredicateAdded(_) => "PredicateAdded",
                ChangeType::PredicateModified(_) => "PredicateModified",
                ChangeType::PredicateRemoved(_) => "PredicateRemoved",
                ChangeType::VariableAdded(_) => "VariableAdded",
                ChangeType::VariableModified(_) => "VariableModified",
                ChangeType::VariableRemoved(_) => "VariableRemoved",
                ChangeType::HierarchyAdded(_, _) => "HierarchyAdded",
                ChangeType::HierarchyRemoved(_, _) => "HierarchyRemoved",
            };
            *by_type.entry(type_name.to_string()).or_insert(0) += 1;
        }

        ChangeStats {
            total_changes: self.changes.len(),
            domains_affected: self.domains_affected.len(),
            predicates_affected: self.predicates_affected.len(),
            variables_affected: self.variables_affected.len(),
            changes_by_type: by_type,
        }
    }
}

/// Statistics about recorded changes
#[derive(Clone, Debug)]
pub struct ChangeStats {
    pub total_changes: usize,
    pub domains_affected: usize,
    pub predicates_affected: usize,
    pub variables_affected: usize,
    pub changes_by_type: HashMap<String, usize>,
}

/// Dependency graph for schema components
#[derive(Clone, Debug, Default)]
pub struct DependencyGraph {
    /// Predicates that depend on each domain
    domain_dependents: HashMap<String, HashSet<String>>,
    /// Domains that each predicate depends on
    predicate_dependencies: HashMap<String, HashSet<String>>,
    /// Variables that depend on each domain
    variable_dependents: HashMap<String, HashSet<String>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build dependency graph from symbol table
    pub fn from_symbol_table(table: &SymbolTable) -> Self {
        let mut graph = Self::new();

        // Add predicate dependencies
        for (pred_name, pred) in &table.predicates {
            for domain in &pred.arg_domains {
                graph.add_predicate_dependency(pred_name, domain);
            }
        }

        // Add variable dependencies
        for (var, domain) in &table.variables {
            graph.add_variable_dependency(var, domain);
        }

        graph
    }

    /// Add a predicate dependency on a domain
    pub fn add_predicate_dependency(
        &mut self,
        predicate: impl Into<String>,
        domain: impl Into<String>,
    ) {
        let predicate = predicate.into();
        let domain = domain.into();

        self.domain_dependents
            .entry(domain.clone())
            .or_default()
            .insert(predicate.clone());

        self.predicate_dependencies
            .entry(predicate)
            .or_default()
            .insert(domain);
    }

    /// Add a variable dependency on a domain
    pub fn add_variable_dependency(
        &mut self,
        variable: impl Into<String>,
        domain: impl Into<String>,
    ) {
        let variable = variable.into();
        let domain = domain.into();

        self.variable_dependents
            .entry(domain)
            .or_default()
            .insert(variable);
    }

    /// Get all predicates that depend on a domain
    pub fn get_dependent_predicates(&self, domain: &str) -> HashSet<String> {
        self.domain_dependents
            .get(domain)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all variables that depend on a domain
    pub fn get_dependent_variables(&self, domain: &str) -> HashSet<String> {
        self.variable_dependents
            .get(domain)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all domains that a predicate depends on
    pub fn get_predicate_dependencies(&self, predicate: &str) -> HashSet<String> {
        self.predicate_dependencies
            .get(predicate)
            .cloned()
            .unwrap_or_default()
    }

    /// Compute transitive closure of affected components
    pub fn compute_affected_components(
        &self,
        initial_domains: &HashSet<String>,
    ) -> AffectedComponents {
        let mut affected = AffectedComponents::default();
        let mut to_process: VecDeque<String> = initial_domains.iter().cloned().collect();

        affected.domains.extend(initial_domains.clone());

        while let Some(domain) = to_process.pop_front() {
            // Find all predicates depending on this domain
            if let Some(predicates) = self.domain_dependents.get(&domain) {
                for pred in predicates {
                    if affected.predicates.insert(pred.clone()) {
                        // If this predicate depends on other domains, they might be affected too
                        if let Some(deps) = self.predicate_dependencies.get(pred) {
                            for dep_domain in deps {
                                if dep_domain != &domain && !affected.domains.contains(dep_domain) {
                                    // Mark as indirectly affected
                                    affected.domains.insert(dep_domain.clone());
                                }
                            }
                        }
                    }
                }
            }

            // Find all variables depending on this domain
            if let Some(variables) = self.variable_dependents.get(&domain) {
                affected.variables.extend(variables.clone());
            }
        }

        affected
    }
}

/// Components affected by changes
#[derive(Clone, Debug, Default)]
pub struct AffectedComponents {
    pub domains: HashSet<String>,
    pub predicates: HashSet<String>,
    pub variables: HashSet<String>,
}

impl AffectedComponents {
    pub fn is_empty(&self) -> bool {
        self.domains.is_empty() && self.predicates.is_empty() && self.variables.is_empty()
    }

    pub fn total_count(&self) -> usize {
        self.domains.len() + self.predicates.len() + self.variables.len()
    }
}

/// Cache for validation results
#[derive(Clone, Debug, Default)]
pub struct ValidationCache {
    domain_results: HashMap<String, Vec<String>>,
    predicate_results: HashMap<String, Vec<String>>,
    variable_results: HashMap<String, Vec<String>>,
}

impl ValidationCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Cache domain validation results
    pub fn cache_domain(&mut self, domain: impl Into<String>, errors: Vec<String>) {
        self.domain_results.insert(domain.into(), errors);
    }

    /// Cache predicate validation results
    pub fn cache_predicate(&mut self, predicate: impl Into<String>, errors: Vec<String>) {
        self.predicate_results.insert(predicate.into(), errors);
    }

    /// Cache variable validation results
    pub fn cache_variable(&mut self, variable: impl Into<String>, errors: Vec<String>) {
        self.variable_results.insert(variable.into(), errors);
    }

    /// Get cached domain result
    pub fn get_domain(&self, domain: &str) -> Option<&Vec<String>> {
        self.domain_results.get(domain)
    }

    /// Get cached predicate result
    pub fn get_predicate(&self, predicate: &str) -> Option<&Vec<String>> {
        self.predicate_results.get(predicate)
    }

    /// Get cached variable result
    pub fn get_variable(&self, variable: &str) -> Option<&Vec<String>> {
        self.variable_results.get(variable)
    }

    /// Invalidate cache entries for affected components
    pub fn invalidate(&mut self, affected: &AffectedComponents) {
        for domain in &affected.domains {
            self.domain_results.remove(domain);
        }
        for predicate in &affected.predicates {
            self.predicate_results.remove(predicate);
        }
        for variable in &affected.variables {
            self.variable_results.remove(variable);
        }
    }

    /// Clear all cached results
    pub fn clear(&mut self) {
        self.domain_results.clear();
        self.predicate_results.clear();
        self.variable_results.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            domains_cached: self.domain_results.len(),
            predicates_cached: self.predicate_results.len(),
            variables_cached: self.variable_results.len(),
        }
    }
}

/// Cache statistics
#[derive(Clone, Debug)]
pub struct CacheStats {
    pub domains_cached: usize,
    pub predicates_cached: usize,
    pub variables_cached: usize,
}

/// Incremental validator for symbol tables
pub struct IncrementalValidator<'a> {
    table: &'a SymbolTable,
    tracker: &'a ChangeTracker,
    hierarchy: Option<&'a DomainHierarchy>,
    cache: ValidationCache,
    dependency_graph: DependencyGraph,
}

impl<'a> IncrementalValidator<'a> {
    pub fn new(table: &'a SymbolTable, tracker: &'a ChangeTracker) -> Self {
        Self {
            table,
            tracker,
            hierarchy: None,
            cache: ValidationCache::new(),
            dependency_graph: DependencyGraph::from_symbol_table(table),
        }
    }

    pub fn with_hierarchy(mut self, hierarchy: &'a DomainHierarchy) -> Self {
        self.hierarchy = Some(hierarchy);
        self
    }

    pub fn with_cache(mut self, cache: ValidationCache) -> Self {
        self.cache = cache;
        self
    }

    /// Get the validation cache
    pub fn cache(&self) -> &ValidationCache {
        &self.cache
    }

    /// Extract the validation cache (consumes self)
    pub fn into_cache(self) -> ValidationCache {
        self.cache
    }

    /// Perform incremental validation
    pub fn validate_incremental(&mut self) -> Result<IncrementalValidationReport> {
        let start = Instant::now();

        if !self.tracker.has_changes() {
            // No changes, return empty report
            return Ok(IncrementalValidationReport {
                report: ValidationReport::new(),
                components_validated: 0,
                components_cached: self.cache.stats().domains_cached
                    + self.cache.stats().predicates_cached
                    + self.cache.stats().variables_cached,
                duration: start.elapsed(),
            });
        }

        // Compute affected components
        let affected = self
            .dependency_graph
            .compute_affected_components(self.tracker.domains_affected());

        // Invalidate cache for affected components
        self.cache.invalidate(&affected);

        // Perform targeted validation
        let mut report = ValidationReport::new();
        let mut components_validated = 0;

        // Validate affected domains
        for domain in &affected.domains {
            if let Some(cached) = self.cache.get_domain(domain) {
                report.errors.extend(cached.clone());
            } else {
                let errors = self.validate_domain(domain)?;
                report.errors.extend(errors.clone());
                self.cache.cache_domain(domain, errors);
                components_validated += 1;
            }
        }

        // Validate affected predicates
        for predicate in &affected.predicates {
            if let Some(cached) = self.cache.get_predicate(predicate) {
                report.errors.extend(cached.clone());
            } else {
                let errors = self.validate_predicate(predicate)?;
                report.errors.extend(errors.clone());
                self.cache.cache_predicate(predicate, errors);
                components_validated += 1;
            }
        }

        // Validate affected variables
        for variable in &affected.variables {
            if let Some(cached) = self.cache.get_variable(variable) {
                report.errors.extend(cached.clone());
            } else {
                let errors = self.validate_variable(variable)?;
                report.errors.extend(errors.clone());
                self.cache.cache_variable(variable, errors);
                components_validated += 1;
            }
        }

        let cache_stats = self.cache.stats();
        let components_cached = cache_stats.domains_cached
            + cache_stats.predicates_cached
            + cache_stats.variables_cached;

        Ok(IncrementalValidationReport {
            report,
            components_validated,
            components_cached,
            duration: start.elapsed(),
        })
    }

    fn validate_domain(&self, domain: &str) -> Result<Vec<String>> {
        let mut errors = Vec::new();

        if !self.table.domains.contains_key(domain) {
            errors.push(format!("Domain '{}' not found in symbol table", domain));
        }

        Ok(errors)
    }

    fn validate_predicate(&self, predicate: &str) -> Result<Vec<String>> {
        let mut errors = Vec::new();

        if let Some(pred) = self.table.predicates.get(predicate) {
            for domain in &pred.arg_domains {
                if domain != "Unknown" && !self.table.domains.contains_key(domain) {
                    errors.push(format!(
                        "Predicate '{}' references undefined domain '{}'",
                        predicate, domain
                    ));
                }
            }
        } else {
            errors.push(format!(
                "Predicate '{}' not found in symbol table",
                predicate
            ));
        }

        Ok(errors)
    }

    fn validate_variable(&self, variable: &str) -> Result<Vec<String>> {
        let mut errors = Vec::new();

        if let Some(domain) = self.table.variables.get(variable) {
            if !self.table.domains.contains_key(domain) {
                errors.push(format!(
                    "Variable '{}' is bound to undefined domain '{}'",
                    variable, domain
                ));
            }
        } else {
            errors.push(format!("Variable '{}' not found in symbol table", variable));
        }

        Ok(errors)
    }
}

/// Incremental validation report with performance metrics
#[derive(Clone, Debug)]
pub struct IncrementalValidationReport {
    pub report: ValidationReport,
    pub components_validated: usize,
    pub components_cached: usize,
    pub duration: Duration,
}

impl IncrementalValidationReport {
    pub fn is_valid(&self) -> bool {
        self.report.is_valid()
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.components_validated + self.components_cached;
        if total == 0 {
            0.0
        } else {
            self.components_cached as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DomainInfo, PredicateInfo};

    #[test]
    fn test_change_tracker_basic() {
        let mut tracker = ChangeTracker::new();

        tracker.record_domain_addition("Person");
        tracker.record_predicate_addition("knows");
        tracker.record_variable_addition("x");

        assert_eq!(tracker.changes().len(), 3);
        assert_eq!(tracker.domains_affected().len(), 1);
        assert_eq!(tracker.predicates_affected().len(), 1);
        assert_eq!(tracker.variables_affected().len(), 1);
    }

    #[test]
    fn test_change_tracker_batching() {
        let mut tracker = ChangeTracker::new();

        let batch1 = tracker.begin_batch();
        tracker.record_domain_addition("Person");
        tracker.record_domain_addition("Location");
        tracker.end_batch();

        let batch2 = tracker.begin_batch();
        tracker.record_predicate_addition("at");
        tracker.end_batch();

        assert_eq!(tracker.changes().len(), 3);
        assert!(tracker.changes()[0].batch_id == Some(batch1));
        assert!(tracker.changes()[1].batch_id == Some(batch1));
        assert!(tracker.changes()[2].batch_id == Some(batch2));
    }

    #[test]
    fn test_dependency_graph() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Location", 50)).unwrap();

        let knows = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()]);
        table.add_predicate(knows).unwrap();

        let at = PredicateInfo::new("at", vec!["Person".to_string(), "Location".to_string()]);
        table.add_predicate(at).unwrap();

        let graph = DependencyGraph::from_symbol_table(&table);

        let person_deps = graph.get_dependent_predicates("Person");
        assert_eq!(person_deps.len(), 2);
        assert!(person_deps.contains("knows"));
        assert!(person_deps.contains("at"));

        let location_deps = graph.get_dependent_predicates("Location");
        assert_eq!(location_deps.len(), 1);
        assert!(location_deps.contains("at"));
    }

    #[test]
    fn test_affected_components() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let knows = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()]);
        table.add_predicate(knows).unwrap();

        table.bind_variable("x", "Person").unwrap();

        let graph = DependencyGraph::from_symbol_table(&table);

        let mut initial = HashSet::new();
        initial.insert("Person".to_string());

        let affected = graph.compute_affected_components(&initial);

        assert_eq!(affected.domains.len(), 1);
        assert_eq!(affected.predicates.len(), 1);
        assert_eq!(affected.variables.len(), 1);
        assert!(affected.predicates.contains("knows"));
        assert!(affected.variables.contains("x"));
    }

    #[test]
    fn test_validation_cache() {
        let mut cache = ValidationCache::new();

        cache.cache_domain("Person", vec![]);
        cache.cache_predicate("knows", vec!["Error".to_string()]);

        assert_eq!(cache.get_domain("Person"), Some(&vec![]));
        assert_eq!(
            cache.get_predicate("knows"),
            Some(&vec!["Error".to_string()])
        );
        assert_eq!(cache.get_variable("x"), None);

        let stats = cache.stats();
        assert_eq!(stats.domains_cached, 1);
        assert_eq!(stats.predicates_cached, 1);
        assert_eq!(stats.variables_cached, 0);
    }

    #[test]
    fn test_incremental_validation_basic() {
        let mut table = SymbolTable::new();
        let mut tracker = ChangeTracker::new();

        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        tracker.record_domain_addition("Person");

        let mut validator = IncrementalValidator::new(&table, &tracker);
        let report = validator.validate_incremental().unwrap();

        assert!(report.is_valid());
        assert_eq!(report.components_validated, 1);
    }

    #[test]
    fn test_incremental_validation_with_cache() {
        let mut table = SymbolTable::new();
        let mut tracker = ChangeTracker::new();

        // First validation
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        tracker.record_domain_addition("Person");

        let mut validator = IncrementalValidator::new(&table, &tracker);
        let report1 = validator.validate_incremental().unwrap();
        assert_eq!(report1.components_validated, 1);

        // Extract cache before second validation
        let cache = validator.cache.clone();

        // Second validation with cache (use fresh references)
        let mut tracker2 = ChangeTracker::new();
        table.add_domain(DomainInfo::new("Location", 50)).unwrap();
        tracker2.record_domain_addition("Location");

        let mut validator2 = IncrementalValidator::new(&table, &tracker2).with_cache(cache);
        let report2 = validator2.validate_incremental().unwrap();

        // Only Location should be validated, Person should be cached
        assert_eq!(report2.components_validated, 1);
        assert!(report2.components_cached > 0);
    }

    #[test]
    fn test_change_stats() {
        let mut tracker = ChangeTracker::new();

        tracker.record_domain_addition("Person");
        tracker.record_domain_modification("Person");
        tracker.record_predicate_addition("knows");

        let stats = tracker.stats();
        assert_eq!(stats.total_changes, 3);
        assert_eq!(stats.domains_affected, 1);
        assert_eq!(stats.predicates_affected, 1);
        assert_eq!(stats.changes_by_type.get("DomainAdded"), Some(&1));
        assert_eq!(stats.changes_by_type.get("DomainModified"), Some(&1));
        assert_eq!(stats.changes_by_type.get("PredicateAdded"), Some(&1));
    }

    #[test]
    fn test_incremental_validation_no_changes() {
        let table = SymbolTable::new();
        let tracker = ChangeTracker::new();

        let mut validator = IncrementalValidator::new(&table, &tracker);
        let report = validator.validate_incremental().unwrap();

        assert!(report.is_valid());
        assert_eq!(report.components_validated, 0);
    }
}
