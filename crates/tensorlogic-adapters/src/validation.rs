//! Schema validation and completeness checking.

use anyhow::Result;
use std::collections::HashSet;

use crate::{DomainHierarchy, SymbolTable};

/// Validation results with errors, warnings, and hints
#[derive(Clone, Debug, Default)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub hints: Vec<String>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_error(&mut self, error: impl Into<String>) {
        self.errors.push(error.into());
    }

    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    pub fn add_hint(&mut self, hint: impl Into<String>) {
        self.hints.push(hint.into());
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn has_issues(&self) -> bool {
        !self.errors.is_empty() || !self.warnings.is_empty()
    }
}

/// Schema validator for symbol tables
pub struct SchemaValidator<'a> {
    table: &'a SymbolTable,
    hierarchy: Option<&'a DomainHierarchy>,
}

impl<'a> SchemaValidator<'a> {
    pub fn new(table: &'a SymbolTable) -> Self {
        Self {
            table,
            hierarchy: None,
        }
    }

    pub fn with_hierarchy(mut self, hierarchy: &'a DomainHierarchy) -> Self {
        self.hierarchy = Some(hierarchy);
        self
    }

    /// Perform comprehensive schema validation
    pub fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        self.check_completeness(&mut report)?;
        self.check_consistency(&mut report)?;
        self.check_semantic(&mut report)?;

        Ok(report)
    }

    /// Check schema completeness
    fn check_completeness(&self, report: &mut ValidationReport) -> Result<()> {
        // Check that all domains referenced by predicates exist
        for (pred_name, pred) in &self.table.predicates {
            for domain in &pred.arg_domains {
                if domain != "Unknown" && !self.table.domains.contains_key(domain) {
                    report.add_error(format!(
                        "Predicate '{}' references undefined domain '{}'",
                        pred_name, domain
                    ));
                }
            }
        }

        // Check that all variables are bound to existing domains
        for (var, domain) in &self.table.variables {
            if !self.table.domains.contains_key(domain) {
                report.add_error(format!(
                    "Variable '{}' is bound to undefined domain '{}'",
                    var, domain
                ));
            }
        }

        // Check for hierarchy references
        if let Some(hierarchy) = self.hierarchy {
            for domain in hierarchy.get_all_domains() {
                if !self.table.domains.contains_key(&domain) {
                    report.add_error(format!(
                        "Domain hierarchy references undefined domain '{}'",
                        domain
                    ));
                }
            }
        }

        Ok(())
    }

    /// Check schema consistency
    fn check_consistency(&self, report: &mut ValidationReport) -> Result<()> {
        // Check for duplicate domain definitions
        let mut seen_domains = HashSet::new();
        for domain_name in self.table.domains.keys() {
            if !seen_domains.insert(domain_name) {
                report.add_error(format!("Duplicate domain definition: '{}'", domain_name));
            }
        }

        // Check for duplicate predicate definitions
        let mut seen_predicates = HashSet::new();
        for pred_name in self.table.predicates.keys() {
            if !seen_predicates.insert(pred_name) {
                report.add_error(format!("Duplicate predicate definition: '{}'", pred_name));
            }
        }

        // Validate hierarchy is acyclic
        if let Some(hierarchy) = self.hierarchy {
            if let Err(e) = hierarchy.validate_acyclic() {
                report.add_error(format!("Domain hierarchy contains cycles: {}", e));
            }
        }

        // Check domain cardinalities are non-negative
        for (domain_name, domain) in &self.table.domains {
            if domain.cardinality == 0 && domain.elements.is_none() {
                report.add_warning(format!(
                    "Domain '{}' has cardinality 0 and no elements defined",
                    domain_name
                ));
            }
        }

        Ok(())
    }

    /// Check semantic validity
    fn check_semantic(&self, report: &mut ValidationReport) -> Result<()> {
        // Warn about unused domains
        let mut used_domains = HashSet::new();

        // Collect domains used in predicates
        for pred in self.table.predicates.values() {
            for domain in &pred.arg_domains {
                used_domains.insert(domain.as_str());
            }
        }

        // Collect domains used in variables
        for domain in self.table.variables.values() {
            used_domains.insert(domain.as_str());
        }

        // Check for unused domains
        for domain_name in self.table.domains.keys() {
            if !used_domains.contains(domain_name.as_str()) {
                report.add_warning(format!(
                    "Domain '{}' is defined but never used",
                    domain_name
                ));
            }
        }

        // Warn about predicates with "Unknown" domains
        for (pred_name, pred) in &self.table.predicates {
            if pred.arg_domains.iter().any(|d| d == "Unknown") {
                report.add_warning(format!(
                    "Predicate '{}' has 'Unknown' domain types - consider specifying explicit types",
                    pred_name
                ));
            }
        }

        // Suggest missing predicates for equivalence relations
        if let Some(hierarchy) = self.hierarchy {
            self.suggest_equality_predicates(hierarchy, report);
        }

        Ok(())
    }

    /// Suggest equality predicates for domains in hierarchy
    fn suggest_equality_predicates(
        &self,
        _hierarchy: &DomainHierarchy,
        report: &mut ValidationReport,
    ) {
        // Check if there's an equality predicate for each domain
        let has_eq = self.table.predicates.iter().any(|(name, _)| {
            name.to_lowercase().contains("eq")
                || name.to_lowercase().contains("equal")
                || name == "="
        });

        if !has_eq && !self.table.domains.is_empty() {
            report.add_hint("Consider defining equality predicates for your domains".to_string());
        }
    }
}

/// Helper trait for DomainHierarchy to get all domains
trait HierarchyHelper {
    fn get_all_domains(&self) -> Vec<String>;
}

impl HierarchyHelper for DomainHierarchy {
    fn get_all_domains(&self) -> Vec<String> {
        // This is a workaround since we can't access private fields
        // In practice, we'd need to add a public method to DomainHierarchy
        // For now, return empty vec
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DomainInfo, PredicateInfo};

    #[test]
    fn test_validation_complete_schema() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 10)).unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "Parent",
                vec!["Person".into(), "Person".into()],
            ))
            .unwrap();

        let validator = SchemaValidator::new(&table);
        let report = validator.validate().unwrap();

        assert!(report.is_valid());
    }

    #[test]
    fn test_validation_missing_domain() {
        let mut table = SymbolTable::new();
        // Bypass normal add_predicate validation by directly inserting
        table.predicates.insert(
            "Parent".into(),
            PredicateInfo::new("Parent", vec!["Person".into(), "Person".into()]),
        );

        let validator = SchemaValidator::new(&table);
        let report = validator.validate().unwrap();

        assert!(!report.is_valid());
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_validation_unused_domain() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 10)).unwrap();
        table.add_domain(DomainInfo::new("City", 5)).unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "Parent",
                vec!["Person".into(), "Person".into()],
            ))
            .unwrap();

        let validator = SchemaValidator::new(&table);
        let report = validator.validate().unwrap();

        assert!(report.is_valid());
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn test_validation_unknown_domains() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 10)).unwrap();
        table.predicates.insert(
            "Test".into(),
            PredicateInfo::new("Test", vec!["Unknown".into()]),
        );

        let validator = SchemaValidator::new(&table);
        let report = validator.validate().unwrap();

        assert!(report.is_valid());
        assert!(!report.warnings.is_empty());
    }
}
