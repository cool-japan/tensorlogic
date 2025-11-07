//! Integration utilities for tensorlogic-compiler.
//!
//! This module provides utilities for exporting SymbolTable data to
//! tensorlogic-compiler's CompilerContext and for bidirectional synchronization.

use anyhow::Result;
use std::collections::HashMap;

use crate::{DomainInfo, PredicateInfo, SymbolTable};

/// Export utilities for compiler integration.
pub struct CompilerExport;

impl CompilerExport {
    /// Export domain information as a simple map for compiler consumption.
    ///
    /// This returns a HashMap that maps domain names to their cardinalities,
    /// suitable for direct use in compiler contexts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, CompilerExport};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_domain(DomainInfo::new("Location", 50)).unwrap();
    ///
    /// let domain_map = CompilerExport::export_domains(&table);
    /// assert_eq!(domain_map.get("Person"), Some(&100));
    /// assert_eq!(domain_map.get("Location"), Some(&50));
    /// ```
    pub fn export_domains(table: &SymbolTable) -> HashMap<String, usize> {
        table
            .domains
            .iter()
            .map(|(name, info)| (name.clone(), info.cardinality))
            .collect()
    }

    /// Export predicate signatures for type checking.
    ///
    /// Returns a HashMap mapping predicate names to their argument domain lists.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo, CompilerExport};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_predicate(PredicateInfo::new(
    ///     "knows",
    ///     vec!["Person".to_string(), "Person".to_string()]
    /// )).unwrap();
    ///
    /// let signatures = CompilerExport::export_predicate_signatures(&table);
    /// assert_eq!(signatures.get("knows"), Some(&vec!["Person".to_string(), "Person".to_string()]));
    /// ```
    pub fn export_predicate_signatures(table: &SymbolTable) -> HashMap<String, Vec<String>> {
        table
            .predicates
            .iter()
            .map(|(name, info)| (name.clone(), info.arg_domains.clone()))
            .collect()
    }

    /// Export variable bindings for scope analysis.
    ///
    /// Returns a HashMap mapping variable names to their domain types.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, CompilerExport};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.bind_variable("x", "Person").unwrap();
    /// table.bind_variable("y", "Person").unwrap();
    ///
    /// let bindings = CompilerExport::export_variable_bindings(&table);
    /// assert_eq!(bindings.get("x"), Some(&"Person".to_string()));
    /// assert_eq!(bindings.get("y"), Some(&"Person".to_string()));
    /// ```
    pub fn export_variable_bindings(table: &SymbolTable) -> HashMap<String, String> {
        table
            .variables
            .iter()
            .map(|(var, domain)| (var.clone(), domain.clone()))
            .collect()
    }

    /// Create a complete export bundle for compiler initialization.
    ///
    /// Returns all three maps (domains, signatures, bindings) in a single structure.
    pub fn export_all(table: &SymbolTable) -> CompilerExportBundle {
        CompilerExportBundle {
            domains: Self::export_domains(table),
            predicate_signatures: Self::export_predicate_signatures(table),
            variable_bindings: Self::export_variable_bindings(table),
        }
    }
}

/// Complete export bundle for compiler integration.
///
/// This structure contains all information needed to initialize a compiler context
/// from a symbol table.
#[derive(Clone, Debug)]
pub struct CompilerExportBundle {
    /// Domain names mapped to cardinalities.
    pub domains: HashMap<String, usize>,
    /// Predicate names mapped to argument domain lists.
    pub predicate_signatures: HashMap<String, Vec<String>>,
    /// Variable names mapped to domain types.
    pub variable_bindings: HashMap<String, String>,
}

impl CompilerExportBundle {
    /// Create an empty export bundle.
    pub fn new() -> Self {
        Self {
            domains: HashMap::new(),
            predicate_signatures: HashMap::new(),
            variable_bindings: HashMap::new(),
        }
    }

    /// Check if the bundle is empty.
    pub fn is_empty(&self) -> bool {
        self.domains.is_empty()
            && self.predicate_signatures.is_empty()
            && self.variable_bindings.is_empty()
    }
}

impl Default for CompilerExportBundle {
    fn default() -> Self {
        Self::new()
    }
}

/// Import utilities for reverse synchronization.
pub struct CompilerImport;

impl CompilerImport {
    /// Import domain information from a compiler context back into a symbol table.
    ///
    /// This is useful for synchronizing state after compilation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, CompilerImport};
    /// use std::collections::HashMap;
    ///
    /// let mut domains = HashMap::new();
    /// domains.insert("Person".to_string(), 100);
    /// domains.insert("Location".to_string(), 50);
    ///
    /// let mut table = SymbolTable::new();
    /// CompilerImport::import_domains(&mut table, &domains).unwrap();
    ///
    /// assert!(table.get_domain("Person").is_some());
    /// assert!(table.get_domain("Location").is_some());
    /// ```
    pub fn import_domains(table: &mut SymbolTable, domains: &HashMap<String, usize>) -> Result<()> {
        for (name, cardinality) in domains {
            table.add_domain(DomainInfo::new(name.clone(), *cardinality))?;
        }
        Ok(())
    }

    /// Import predicate signatures from a compiler context.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, CompilerImport};
    /// use std::collections::HashMap;
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let mut signatures = HashMap::new();
    /// signatures.insert("knows".to_string(), vec!["Person".to_string(), "Person".to_string()]);
    ///
    /// CompilerImport::import_predicates(&mut table, &signatures).unwrap();
    ///
    /// assert!(table.get_predicate("knows").is_some());
    /// ```
    pub fn import_predicates(
        table: &mut SymbolTable,
        signatures: &HashMap<String, Vec<String>>,
    ) -> Result<()> {
        for (name, arg_domains) in signatures {
            table.add_predicate(PredicateInfo::new(name.clone(), arg_domains.clone()))?;
        }
        Ok(())
    }

    /// Import variable bindings from a compiler context.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, CompilerImport};
    /// use std::collections::HashMap;
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let mut bindings = HashMap::new();
    /// bindings.insert("x".to_string(), "Person".to_string());
    /// bindings.insert("y".to_string(), "Person".to_string());
    ///
    /// CompilerImport::import_variables(&mut table, &bindings).unwrap();
    ///
    /// assert_eq!(table.get_variable_domain("x"), Some("Person"));
    /// assert_eq!(table.get_variable_domain("y"), Some("Person"));
    /// ```
    pub fn import_variables(
        table: &mut SymbolTable,
        bindings: &HashMap<String, String>,
    ) -> Result<()> {
        for (var, domain) in bindings {
            table.bind_variable(var, domain)?;
        }
        Ok(())
    }

    /// Import a complete bundle into a symbol table.
    pub fn import_all(table: &mut SymbolTable, bundle: &CompilerExportBundle) -> Result<()> {
        Self::import_domains(table, &bundle.domains)?;
        Self::import_predicates(table, &bundle.predicate_signatures)?;
        Self::import_variables(table, &bundle.variable_bindings)?;
        Ok(())
    }
}

/// Bidirectional synchronization utilities.
pub struct SymbolTableSync;

impl SymbolTableSync {
    /// Synchronize a symbol table with compiler data, merging information.
    ///
    /// This performs a two-way sync:
    /// 1. Exports current symbol table state
    /// 2. Imports compiler context data
    /// 3. Returns the merged export bundle
    pub fn sync_with_compiler(
        table: &mut SymbolTable,
        compiler_bundle: &CompilerExportBundle,
    ) -> Result<CompilerExportBundle> {
        // First, import compiler data into the table
        CompilerImport::import_all(table, compiler_bundle)?;

        // Then, export the merged state
        Ok(CompilerExport::export_all(table))
    }

    /// Validate that a compiler bundle is compatible with a symbol table.
    ///
    /// Checks that all referenced domains exist.
    pub fn validate_bundle(
        table: &SymbolTable,
        bundle: &CompilerExportBundle,
    ) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check predicate signatures reference existing domains
        for (pred_name, arg_domains) in &bundle.predicate_signatures {
            for domain in arg_domains {
                if !table.domains.contains_key(domain) && !bundle.domains.contains_key(domain) {
                    errors.push(format!(
                        "Predicate '{}' references unknown domain '{}'",
                        pred_name, domain
                    ));
                }
            }
        }

        // Check variable bindings reference existing domains
        for (var_name, domain) in &bundle.variable_bindings {
            if !table.domains.contains_key(domain) && !bundle.domains.contains_key(domain) {
                errors.push(format!(
                    "Variable '{}' references unknown domain '{}'",
                    var_name, domain
                ));
            }
        }

        // Warn about unused domains
        for domain_name in bundle.domains.keys() {
            let used_in_predicates = bundle
                .predicate_signatures
                .values()
                .any(|args| args.contains(domain_name));
            let used_in_variables = bundle.variable_bindings.values().any(|d| d == domain_name);

            if !used_in_predicates && !used_in_variables {
                warnings.push(format!(
                    "Domain '{}' is defined but never used",
                    domain_name
                ));
            }
        }

        Ok(ValidationResult { errors, warnings })
    }
}

/// Result of bundle validation.
#[derive(Clone, Debug)]
pub struct ValidationResult {
    /// Validation errors (must be empty for valid bundles).
    pub errors: Vec<String>,
    /// Validation warnings (non-critical issues).
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Check if validation passed (no errors).
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_domains() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Location", 50)).unwrap();

        let domains = CompilerExport::export_domains(&table);
        assert_eq!(domains.get("Person"), Some(&100));
        assert_eq!(domains.get("Location"), Some(&50));
    }

    #[test]
    fn test_export_predicate_signatures() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "knows",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .unwrap();

        let signatures = CompilerExport::export_predicate_signatures(&table);
        assert_eq!(
            signatures.get("knows"),
            Some(&vec!["Person".to_string(), "Person".to_string()])
        );
    }

    #[test]
    fn test_export_all() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table
            .add_predicate(PredicateInfo::new("knows", vec!["Person".to_string()]))
            .unwrap();
        table.bind_variable("x", "Person").unwrap();

        let bundle = CompilerExport::export_all(&table);
        assert_eq!(bundle.domains.len(), 1);
        assert_eq!(bundle.predicate_signatures.len(), 1);
        assert_eq!(bundle.variable_bindings.len(), 1);
    }

    #[test]
    fn test_import_domains() {
        let mut domains = HashMap::new();
        domains.insert("Person".to_string(), 100);

        let mut table = SymbolTable::new();
        CompilerImport::import_domains(&mut table, &domains).unwrap();

        assert!(table.get_domain("Person").is_some());
    }

    #[test]
    fn test_import_predicates() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let mut signatures = HashMap::new();
        signatures.insert("knows".to_string(), vec!["Person".to_string()]);

        CompilerImport::import_predicates(&mut table, &signatures).unwrap();
        assert!(table.get_predicate("knows").is_some());
    }

    #[test]
    fn test_validation_invalid_domain_reference() {
        let table = SymbolTable::new();
        let mut bundle = CompilerExportBundle::new();
        bundle
            .predicate_signatures
            .insert("knows".to_string(), vec!["UnknownDomain".to_string()]);

        let result = SymbolTableSync::validate_bundle(&table, &bundle).unwrap();
        assert!(!result.is_valid());
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_validation_unused_domain_warning() {
        let table = SymbolTable::new();
        let mut bundle = CompilerExportBundle::new();
        bundle.domains.insert("UnusedDomain".to_string(), 100);

        let result = SymbolTableSync::validate_bundle(&table, &bundle).unwrap();
        assert!(result.is_valid()); // Still valid, just has warnings
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_sync_with_compiler() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let mut bundle = CompilerExportBundle::new();
        bundle.domains.insert("Location".to_string(), 50);

        let result = SymbolTableSync::sync_with_compiler(&mut table, &bundle).unwrap();

        // Table should now have both domains
        assert!(table.get_domain("Person").is_some());
        assert!(table.get_domain("Location").is_some());

        // Result should include both
        assert_eq!(result.domains.len(), 2);
    }
}
