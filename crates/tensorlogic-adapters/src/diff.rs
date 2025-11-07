//! Schema diff utilities for comparing and tracking changes.
//!
//! This module provides tools for comparing two SymbolTables and generating
//! detailed diff reports showing additions, deletions, and modifications.

use std::collections::HashSet;

use crate::{DomainInfo, PredicateInfo, SymbolTable};

/// Comparison result for two symbol tables.
///
/// Contains detailed information about all differences between two schemas.
#[derive(Clone, Debug, Default)]
pub struct SchemaDiff {
    /// Domains added in the new schema.
    pub domains_added: Vec<DomainInfo>,
    /// Domains removed from the old schema.
    pub domains_removed: Vec<DomainInfo>,
    /// Domains modified between schemas.
    pub domains_modified: Vec<DomainModification>,
    /// Predicates added in the new schema.
    pub predicates_added: Vec<PredicateInfo>,
    /// Predicates removed from the old schema.
    pub predicates_removed: Vec<PredicateInfo>,
    /// Predicates modified between schemas.
    pub predicates_modified: Vec<PredicateModification>,
    /// Variables added in the new schema.
    pub variables_added: Vec<(String, String)>,
    /// Variables removed from the old schema.
    pub variables_removed: Vec<(String, String)>,
    /// Variables with changed types.
    pub variables_modified: Vec<VariableModification>,
}

impl SchemaDiff {
    /// Check if there are any differences.
    pub fn has_changes(&self) -> bool {
        !self.domains_added.is_empty()
            || !self.domains_removed.is_empty()
            || !self.domains_modified.is_empty()
            || !self.predicates_added.is_empty()
            || !self.predicates_removed.is_empty()
            || !self.predicates_modified.is_empty()
            || !self.variables_added.is_empty()
            || !self.variables_removed.is_empty()
            || !self.variables_modified.is_empty()
    }

    /// Check if the diff represents backward-compatible changes.
    ///
    /// A change is backward-compatible if it only adds new entities
    /// or expands existing ones without removing or breaking existing definitions.
    pub fn is_backward_compatible(&self) -> bool {
        // Removals break backward compatibility
        if !self.domains_removed.is_empty()
            || !self.predicates_removed.is_empty()
            || !self.variables_removed.is_empty()
        {
            return false;
        }

        // Check domain modifications for breaking changes
        for modification in &self.domains_modified {
            // Cardinality reduction breaks compatibility
            if modification.new_cardinality < modification.old_cardinality {
                return false;
            }
        }

        // Check predicate modifications for breaking changes
        for modification in &self.predicates_modified {
            // Signature changes break compatibility
            if modification.signature_changed {
                return false;
            }
        }

        // Variable type changes break compatibility
        if !self.variables_modified.is_empty() {
            return false;
        }

        true
    }

    /// Get a summary of the changes.
    pub fn summary(&self) -> DiffSummary {
        DiffSummary {
            domains_added: self.domains_added.len(),
            domains_removed: self.domains_removed.len(),
            domains_modified: self.domains_modified.len(),
            predicates_added: self.predicates_added.len(),
            predicates_removed: self.predicates_removed.len(),
            predicates_modified: self.predicates_modified.len(),
            variables_added: self.variables_added.len(),
            variables_removed: self.variables_removed.len(),
            variables_modified: self.variables_modified.len(),
            is_backward_compatible: self.is_backward_compatible(),
        }
    }

    /// Generate a human-readable report.
    pub fn report(&self) -> String {
        let mut output = String::new();

        if !self.has_changes() {
            output.push_str("No changes detected.\n");
            return output;
        }

        let summary = self.summary();
        output.push_str("Schema Diff Summary:\n");
        output.push_str(&format!(
            "  Backward Compatible: {}\n\n",
            summary.is_backward_compatible
        ));

        if !self.domains_added.is_empty() {
            output.push_str(&format!("Domains Added ({}):\n", self.domains_added.len()));
            for domain in &self.domains_added {
                output.push_str(&format!(
                    "  + {} (cardinality: {})\n",
                    domain.name, domain.cardinality
                ));
            }
            output.push('\n');
        }

        if !self.domains_removed.is_empty() {
            output.push_str(&format!(
                "Domains Removed ({}):\n",
                self.domains_removed.len()
            ));
            for domain in &self.domains_removed {
                output.push_str(&format!(
                    "  - {} (cardinality: {})\n",
                    domain.name, domain.cardinality
                ));
            }
            output.push('\n');
        }

        if !self.domains_modified.is_empty() {
            output.push_str(&format!(
                "Domains Modified ({}):\n",
                self.domains_modified.len()
            ));
            for modification in &self.domains_modified {
                output.push_str(&format!("  ~ {}\n", modification.domain_name));
                if modification.old_cardinality != modification.new_cardinality {
                    output.push_str(&format!(
                        "    cardinality: {} -> {}\n",
                        modification.old_cardinality, modification.new_cardinality
                    ));
                }
                if modification.description_changed {
                    output.push_str("    description: changed\n");
                }
            }
            output.push('\n');
        }

        if !self.predicates_added.is_empty() {
            output.push_str(&format!(
                "Predicates Added ({}):\n",
                self.predicates_added.len()
            ));
            for pred in &self.predicates_added {
                output.push_str(&format!(
                    "  + {} (arity: {})\n",
                    pred.name,
                    pred.arg_domains.len()
                ));
            }
            output.push('\n');
        }

        if !self.predicates_removed.is_empty() {
            output.push_str(&format!(
                "Predicates Removed ({}):\n",
                self.predicates_removed.len()
            ));
            for pred in &self.predicates_removed {
                output.push_str(&format!(
                    "  - {} (arity: {})\n",
                    pred.name,
                    pred.arg_domains.len()
                ));
            }
            output.push('\n');
        }

        if !self.predicates_modified.is_empty() {
            output.push_str(&format!(
                "Predicates Modified ({}):\n",
                self.predicates_modified.len()
            ));
            for modification in &self.predicates_modified {
                output.push_str(&format!("  ~ {}\n", modification.predicate_name));
                if modification.signature_changed {
                    output.push_str(&format!(
                        "    signature: {:?} -> {:?}\n",
                        modification.old_signature, modification.new_signature
                    ));
                }
            }
            output.push('\n');
        }

        output
    }
}

/// Modification details for a domain.
#[derive(Clone, Debug)]
pub struct DomainModification {
    /// Name of the modified domain.
    pub domain_name: String,
    /// Old cardinality.
    pub old_cardinality: usize,
    /// New cardinality.
    pub new_cardinality: usize,
    /// Whether description changed.
    pub description_changed: bool,
    /// Whether metadata changed.
    pub metadata_changed: bool,
}

/// Modification details for a predicate.
#[derive(Clone, Debug)]
pub struct PredicateModification {
    /// Name of the modified predicate.
    pub predicate_name: String,
    /// Whether the signature changed.
    pub signature_changed: bool,
    /// Old argument domains.
    pub old_signature: Vec<String>,
    /// New argument domains.
    pub new_signature: Vec<String>,
    /// Whether description changed.
    pub description_changed: bool,
}

/// Modification details for a variable binding.
#[derive(Clone, Debug)]
pub struct VariableModification {
    /// Variable name.
    pub variable_name: String,
    /// Old domain.
    pub old_domain: String,
    /// New domain.
    pub new_domain: String,
}

/// Summary statistics for a schema diff.
#[derive(Clone, Debug)]
pub struct DiffSummary {
    /// Number of domains added.
    pub domains_added: usize,
    /// Number of domains removed.
    pub domains_removed: usize,
    /// Number of domains modified.
    pub domains_modified: usize,
    /// Number of predicates added.
    pub predicates_added: usize,
    /// Number of predicates removed.
    pub predicates_removed: usize,
    /// Number of predicates modified.
    pub predicates_modified: usize,
    /// Number of variables added.
    pub variables_added: usize,
    /// Number of variables removed.
    pub variables_removed: usize,
    /// Number of variables modified.
    pub variables_modified: usize,
    /// Whether changes are backward compatible.
    pub is_backward_compatible: bool,
}

impl DiffSummary {
    /// Total number of changes.
    pub fn total_changes(&self) -> usize {
        self.domains_added
            + self.domains_removed
            + self.domains_modified
            + self.predicates_added
            + self.predicates_removed
            + self.predicates_modified
            + self.variables_added
            + self.variables_removed
            + self.variables_modified
    }
}

/// Compute the difference between two symbol tables.
///
/// # Example
///
/// ```rust
/// use tensorlogic_adapters::{SymbolTable, DomainInfo, compute_diff};
///
/// let mut old_table = SymbolTable::new();
/// old_table.add_domain(DomainInfo::new("Person", 100)).unwrap();
///
/// let mut new_table = SymbolTable::new();
/// new_table.add_domain(DomainInfo::new("Person", 100)).unwrap();
/// new_table.add_domain(DomainInfo::new("Location", 50)).unwrap();
///
/// let diff = compute_diff(&old_table, &new_table);
/// assert_eq!(diff.domains_added.len(), 1);
/// assert!(diff.is_backward_compatible());
/// ```
pub fn compute_diff(old: &SymbolTable, new: &SymbolTable) -> SchemaDiff {
    let mut diff = SchemaDiff::default();

    // Compute domain differences
    let old_domain_names: HashSet<_> = old.domains.keys().collect();
    let new_domain_names: HashSet<_> = new.domains.keys().collect();

    for name in new_domain_names.difference(&old_domain_names) {
        diff.domains_added.push(new.domains[*name].clone());
    }

    for name in old_domain_names.difference(&new_domain_names) {
        diff.domains_removed.push(old.domains[*name].clone());
    }

    for name in old_domain_names.intersection(&new_domain_names) {
        let old_domain = &old.domains[*name];
        let new_domain = &new.domains[*name];

        if old_domain.cardinality != new_domain.cardinality
            || old_domain.description != new_domain.description
            || old_domain.metadata != new_domain.metadata
        {
            diff.domains_modified.push(DomainModification {
                domain_name: (*name).clone(),
                old_cardinality: old_domain.cardinality,
                new_cardinality: new_domain.cardinality,
                description_changed: old_domain.description != new_domain.description,
                metadata_changed: old_domain.metadata != new_domain.metadata,
            });
        }
    }

    // Compute predicate differences
    let old_pred_names: HashSet<_> = old.predicates.keys().collect();
    let new_pred_names: HashSet<_> = new.predicates.keys().collect();

    for name in new_pred_names.difference(&old_pred_names) {
        diff.predicates_added.push(new.predicates[*name].clone());
    }

    for name in old_pred_names.difference(&new_pred_names) {
        diff.predicates_removed.push(old.predicates[*name].clone());
    }

    for name in old_pred_names.intersection(&new_pred_names) {
        let old_pred = &old.predicates[*name];
        let new_pred = &new.predicates[*name];

        let signature_changed = old_pred.arg_domains != new_pred.arg_domains;
        let description_changed = old_pred.description != new_pred.description;

        if signature_changed || description_changed {
            diff.predicates_modified.push(PredicateModification {
                predicate_name: (*name).clone(),
                signature_changed,
                old_signature: old_pred.arg_domains.clone(),
                new_signature: new_pred.arg_domains.clone(),
                description_changed,
            });
        }
    }

    // Compute variable binding differences
    let old_var_names: HashSet<_> = old.variables.keys().collect();
    let new_var_names: HashSet<_> = new.variables.keys().collect();

    for name in new_var_names.difference(&old_var_names) {
        diff.variables_added
            .push(((*name).clone(), new.variables[*name].clone()));
    }

    for name in old_var_names.difference(&new_var_names) {
        diff.variables_removed
            .push(((*name).clone(), old.variables[*name].clone()));
    }

    for name in old_var_names.intersection(&new_var_names) {
        let old_domain = &old.variables[*name];
        let new_domain = &new.variables[*name];

        if old_domain != new_domain {
            diff.variables_modified.push(VariableModification {
                variable_name: (*name).clone(),
                old_domain: old_domain.clone(),
                new_domain: new_domain.clone(),
            });
        }
    }

    diff
}

/// Merge two symbol tables, preferring values from the newer table.
///
/// This is useful for applying schema migrations or updates.
pub fn merge_tables(base: &SymbolTable, update: &SymbolTable) -> SymbolTable {
    let mut merged = base.clone();

    // Merge domains (update overwrites base)
    for (name, domain) in &update.domains {
        merged.domains.insert(name.clone(), domain.clone());
    }

    // Merge predicates (update overwrites base)
    for (name, predicate) in &update.predicates {
        merged.predicates.insert(name.clone(), predicate.clone());
    }

    // Merge variables (update overwrites base)
    for (name, domain) in &update.variables {
        merged.variables.insert(name.clone(), domain.clone());
    }

    merged
}

/// Compute schema compatibility between two versions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompatibilityLevel {
    /// Schemas are identical.
    Identical,
    /// Changes are backward compatible.
    BackwardCompatible,
    /// Changes are forward compatible (old can read new).
    ForwardCompatible,
    /// Changes break compatibility.
    Breaking,
}

/// Determine compatibility level between two schemas.
pub fn check_compatibility(old: &SymbolTable, new: &SymbolTable) -> CompatibilityLevel {
    let diff = compute_diff(old, new);

    if !diff.has_changes() {
        return CompatibilityLevel::Identical;
    }

    if diff.is_backward_compatible() {
        return CompatibilityLevel::BackwardCompatible;
    }

    // Check for forward compatibility (only removals, no additions or modifications)
    if diff.domains_added.is_empty()
        && diff.predicates_added.is_empty()
        && diff.variables_added.is_empty()
        && diff.domains_modified.is_empty()
        && diff.predicates_modified.is_empty()
        && diff.variables_modified.is_empty()
    {
        return CompatibilityLevel::ForwardCompatible;
    }

    CompatibilityLevel::Breaking
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_schemas() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let diff = compute_diff(&table, &table);
        assert!(!diff.has_changes());
        assert!(diff.is_backward_compatible());
        assert_eq!(
            check_compatibility(&table, &table),
            CompatibilityLevel::Identical
        );
    }

    #[test]
    fn test_domain_addition() {
        let mut old_table = SymbolTable::new();
        old_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let mut new_table = old_table.clone();
        new_table
            .add_domain(DomainInfo::new("Location", 50))
            .unwrap();

        let diff = compute_diff(&old_table, &new_table);
        assert_eq!(diff.domains_added.len(), 1);
        assert_eq!(diff.domains_added[0].name, "Location");
        assert!(diff.is_backward_compatible());
        assert_eq!(
            check_compatibility(&old_table, &new_table),
            CompatibilityLevel::BackwardCompatible
        );
    }

    #[test]
    fn test_domain_removal() {
        let mut old_table = SymbolTable::new();
        old_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        old_table
            .add_domain(DomainInfo::new("Location", 50))
            .unwrap();

        let mut new_table = SymbolTable::new();
        new_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let diff = compute_diff(&old_table, &new_table);
        assert_eq!(diff.domains_removed.len(), 1);
        assert_eq!(diff.domains_removed[0].name, "Location");
        assert!(!diff.is_backward_compatible());
        assert_eq!(
            check_compatibility(&old_table, &new_table),
            CompatibilityLevel::ForwardCompatible
        );
    }

    #[test]
    fn test_domain_modification() {
        let mut old_table = SymbolTable::new();
        old_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let mut new_table = SymbolTable::new();
        new_table
            .add_domain(DomainInfo::new("Person", 200))
            .unwrap();

        let diff = compute_diff(&old_table, &new_table);
        assert_eq!(diff.domains_modified.len(), 1);
        assert_eq!(diff.domains_modified[0].old_cardinality, 100);
        assert_eq!(diff.domains_modified[0].new_cardinality, 200);
        assert!(diff.is_backward_compatible()); // Cardinality increase is compatible
    }

    #[test]
    fn test_cardinality_reduction_breaks_compatibility() {
        let mut old_table = SymbolTable::new();
        old_table
            .add_domain(DomainInfo::new("Person", 200))
            .unwrap();

        let mut new_table = SymbolTable::new();
        new_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let diff = compute_diff(&old_table, &new_table);
        assert!(!diff.is_backward_compatible());
        assert_eq!(
            check_compatibility(&old_table, &new_table),
            CompatibilityLevel::Breaking
        );
    }

    #[test]
    fn test_predicate_addition() {
        let mut old_table = SymbolTable::new();
        old_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let mut new_table = old_table.clone();
        new_table
            .add_predicate(PredicateInfo::new("knows", vec!["Person".to_string()]))
            .unwrap();

        let diff = compute_diff(&old_table, &new_table);
        assert_eq!(diff.predicates_added.len(), 1);
        assert!(diff.is_backward_compatible());
    }

    #[test]
    fn test_predicate_signature_change() {
        let mut old_table = SymbolTable::new();
        old_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        old_table
            .add_predicate(PredicateInfo::new("knows", vec!["Person".to_string()]))
            .unwrap();

        let mut new_table = SymbolTable::new();
        new_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        new_table
            .add_predicate(PredicateInfo::new(
                "knows",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .unwrap();

        let diff = compute_diff(&old_table, &new_table);
        assert_eq!(diff.predicates_modified.len(), 1);
        assert!(diff.predicates_modified[0].signature_changed);
        assert!(!diff.is_backward_compatible());
    }

    #[test]
    fn test_merge_tables() {
        let mut base = SymbolTable::new();
        base.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let mut update = SymbolTable::new();
        update.add_domain(DomainInfo::new("Person", 200)).unwrap();
        update.add_domain(DomainInfo::new("Location", 50)).unwrap();

        let merged = merge_tables(&base, &update);
        assert_eq!(merged.domains.len(), 2);
        assert_eq!(merged.get_domain("Person").unwrap().cardinality, 200);
        assert!(merged.get_domain("Location").is_some());
    }

    #[test]
    fn test_diff_report() {
        let mut old_table = SymbolTable::new();
        old_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let mut new_table = old_table.clone();
        new_table
            .add_domain(DomainInfo::new("Location", 50))
            .unwrap();

        let diff = compute_diff(&old_table, &new_table);
        let report = diff.report();
        assert!(report.contains("Domains Added"));
        assert!(report.contains("Location"));
    }

    #[test]
    fn test_summary_total_changes() {
        let mut old_table = SymbolTable::new();
        old_table
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let mut new_table = old_table.clone();
        new_table
            .add_domain(DomainInfo::new("Location", 50))
            .unwrap();
        new_table.add_domain(DomainInfo::new("Event", 30)).unwrap();

        let diff = compute_diff(&old_table, &new_table);
        let summary = diff.summary();
        assert_eq!(summary.total_changes(), 2);
    }
}
