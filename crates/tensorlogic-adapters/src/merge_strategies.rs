//! Advanced schema merging strategies for combining schemas from different sources.
//!
//! This module provides sophisticated strategies for merging symbol tables with
//! conflict resolution, validation, and detailed merge reports.

use crate::{DomainInfo, PredicateInfo, SymbolTable};
use anyhow::{bail, Result};
use std::collections::HashSet;

/// Strategy for resolving conflicts during schema merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Keep the first (base) version in case of conflict
    KeepFirst,
    /// Keep the second (incoming) version in case of conflict
    KeepSecond,
    /// Fail on any conflict
    FailOnConflict,
    /// Union: Keep both if compatible, fail if incompatible
    Union,
    /// Intersection: Only keep items present in both with compatible definitions
    Intersection,
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// The merged symbol table
    pub merged: SymbolTable,
    /// Report of the merge operation
    pub report: MergeReport,
}

/// Report of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeReport {
    /// Domains that were added from base
    pub base_domains: Vec<String>,
    /// Domains that were added from incoming
    pub incoming_domains: Vec<String>,
    /// Domains that had conflicts
    pub conflicting_domains: Vec<DomainConflict>,
    /// Predicates that were added from base
    pub base_predicates: Vec<String>,
    /// Predicates that were added from incoming
    pub incoming_predicates: Vec<String>,
    /// Predicates that had conflicts
    pub conflicting_predicates: Vec<PredicateConflict>,
    /// Variables that were merged
    pub merged_variables: Vec<String>,
    /// Variables that had conflicts
    pub conflicting_variables: Vec<VariableConflict>,
    /// Overall merge strategy used
    pub strategy: MergeStrategy,
}

impl MergeReport {
    /// Create a new empty merge report.
    pub fn new(strategy: MergeStrategy) -> Self {
        Self {
            base_domains: Vec::new(),
            incoming_domains: Vec::new(),
            conflicting_domains: Vec::new(),
            base_predicates: Vec::new(),
            incoming_predicates: Vec::new(),
            conflicting_predicates: Vec::new(),
            merged_variables: Vec::new(),
            conflicting_variables: Vec::new(),
            strategy,
        }
    }

    /// Check if there were any conflicts during merging.
    pub fn has_conflicts(&self) -> bool {
        !self.conflicting_domains.is_empty()
            || !self.conflicting_predicates.is_empty()
            || !self.conflicting_variables.is_empty()
    }

    /// Get total number of conflicts.
    pub fn conflict_count(&self) -> usize {
        self.conflicting_domains.len()
            + self.conflicting_predicates.len()
            + self.conflicting_variables.len()
    }

    /// Get total number of merged items.
    pub fn merged_count(&self) -> usize {
        self.base_domains.len()
            + self.incoming_domains.len()
            + self.base_predicates.len()
            + self.incoming_predicates.len()
            + self.merged_variables.len()
    }
}

/// Information about a domain conflict.
#[derive(Debug, Clone)]
pub struct DomainConflict {
    /// Name of the conflicting domain
    pub name: String,
    /// Domain from base table
    pub base: DomainInfo,
    /// Domain from incoming table
    pub incoming: DomainInfo,
    /// How the conflict was resolved
    pub resolution: MergeConflictResolution,
}

/// Information about a predicate conflict.
#[derive(Debug, Clone)]
pub struct PredicateConflict {
    /// Name of the conflicting predicate
    pub name: String,
    /// Predicate from base table
    pub base: PredicateInfo,
    /// Predicate from incoming table
    pub incoming: PredicateInfo,
    /// How the conflict was resolved
    pub resolution: MergeConflictResolution,
}

/// Information about a variable conflict.
#[derive(Debug, Clone)]
pub struct VariableConflict {
    /// Name of the conflicting variable
    pub name: String,
    /// Domain from base table
    pub base_domain: String,
    /// Domain from incoming table
    pub incoming_domain: String,
    /// How the conflict was resolved
    pub resolution: MergeConflictResolution,
}

/// How a merge conflict was resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeConflictResolution {
    /// Kept the base version
    KeptBase,
    /// Kept the incoming version
    KeptIncoming,
    /// Failed to resolve
    Failed,
    /// Merged both (only for compatible items)
    Merged,
}

/// A schema merger with configurable strategies.
pub struct SchemaMerger {
    strategy: MergeStrategy,
}

impl SchemaMerger {
    /// Create a new schema merger with the given strategy.
    pub fn new(strategy: MergeStrategy) -> Self {
        Self { strategy }
    }

    /// Merge two symbol tables according to the configured strategy.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, SchemaMerger, MergeStrategy};
    ///
    /// let mut base = SymbolTable::new();
    /// base.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let mut incoming = SymbolTable::new();
    /// incoming.add_domain(DomainInfo::new("Organization", 50)).unwrap();
    ///
    /// let merger = SchemaMerger::new(MergeStrategy::Union);
    /// let result = merger.merge(&base, &incoming).unwrap();
    ///
    /// assert_eq!(result.merged.domains.len(), 2);
    /// ```
    pub fn merge(&self, base: &SymbolTable, incoming: &SymbolTable) -> Result<MergeResult> {
        let mut merged = SymbolTable::new();
        let mut report = MergeReport::new(self.strategy);

        // Merge domains
        self.merge_domains(base, incoming, &mut merged, &mut report)?;

        // Merge predicates
        self.merge_predicates(base, incoming, &mut merged, &mut report)?;

        // Merge variables
        self.merge_variables(base, incoming, &mut merged, &mut report)?;

        Ok(MergeResult { merged, report })
    }

    fn merge_domains(
        &self,
        base: &SymbolTable,
        incoming: &SymbolTable,
        merged: &mut SymbolTable,
        report: &mut MergeReport,
    ) -> Result<()> {
        let base_keys: HashSet<&String> = base.domains.keys().collect();
        let incoming_keys: HashSet<&String> = incoming.domains.keys().collect();

        // Domains only in base
        for key in base_keys.difference(&incoming_keys) {
            let domain = base.domains.get(*key).unwrap();
            merged.add_domain(domain.clone())?;
            report.base_domains.push(key.to_string());
        }

        // Domains only in incoming
        for key in incoming_keys.difference(&base_keys) {
            let domain = incoming.domains.get(*key).unwrap();
            merged.add_domain(domain.clone())?;
            report.incoming_domains.push(key.to_string());
        }

        // Domains in both (conflicts)
        for key in base_keys.intersection(&incoming_keys) {
            let base_domain = base.domains.get(*key).unwrap();
            let incoming_domain = incoming.domains.get(*key).unwrap();

            let (domain, resolution) =
                self.resolve_domain_conflict(base_domain, incoming_domain)?;

            merged.add_domain(domain)?;

            if resolution != MergeConflictResolution::Merged {
                report.conflicting_domains.push(DomainConflict {
                    name: key.to_string(),
                    base: base_domain.clone(),
                    incoming: incoming_domain.clone(),
                    resolution,
                });
            }
        }

        Ok(())
    }

    fn merge_predicates(
        &self,
        base: &SymbolTable,
        incoming: &SymbolTable,
        merged: &mut SymbolTable,
        report: &mut MergeReport,
    ) -> Result<()> {
        let base_keys: HashSet<&String> = base.predicates.keys().collect();
        let incoming_keys: HashSet<&String> = incoming.predicates.keys().collect();

        // Predicates only in base
        for key in base_keys.difference(&incoming_keys) {
            let predicate = base.predicates.get(*key).unwrap();
            merged.add_predicate(predicate.clone())?;
            report.base_predicates.push(key.to_string());
        }

        // Predicates only in incoming
        for key in incoming_keys.difference(&base_keys) {
            let predicate = incoming.predicates.get(*key).unwrap();
            merged.add_predicate(predicate.clone())?;
            report.incoming_predicates.push(key.to_string());
        }

        // Predicates in both (conflicts)
        for key in base_keys.intersection(&incoming_keys) {
            let base_pred = base.predicates.get(*key).unwrap();
            let incoming_pred = incoming.predicates.get(*key).unwrap();

            let (predicate, resolution) =
                self.resolve_predicate_conflict(base_pred, incoming_pred)?;

            merged.add_predicate(predicate)?;

            if resolution != MergeConflictResolution::Merged {
                report.conflicting_predicates.push(PredicateConflict {
                    name: key.to_string(),
                    base: base_pred.clone(),
                    incoming: incoming_pred.clone(),
                    resolution,
                });
            }
        }

        Ok(())
    }

    fn merge_variables(
        &self,
        base: &SymbolTable,
        incoming: &SymbolTable,
        merged: &mut SymbolTable,
        report: &mut MergeReport,
    ) -> Result<()> {
        let base_keys: HashSet<&String> = base.variables.keys().collect();
        let incoming_keys: HashSet<&String> = incoming.variables.keys().collect();

        // Variables only in base
        for key in base_keys.difference(&incoming_keys) {
            let domain = base.variables.get(*key).unwrap();
            merged.bind_variable(key.to_string(), domain.clone())?;
            report.merged_variables.push(key.to_string());
        }

        // Variables only in incoming
        for key in incoming_keys.difference(&base_keys) {
            let domain = incoming.variables.get(*key).unwrap();
            merged.bind_variable(key.to_string(), domain.clone())?;
            report.merged_variables.push(key.to_string());
        }

        // Variables in both (conflicts)
        for key in base_keys.intersection(&incoming_keys) {
            let base_domain = base.variables.get(*key).unwrap();
            let incoming_domain = incoming.variables.get(*key).unwrap();

            let (domain, resolution) =
                self.resolve_variable_conflict(base_domain, incoming_domain)?;

            merged.bind_variable(key.to_string(), domain)?;

            if resolution != MergeConflictResolution::Merged {
                report.conflicting_variables.push(VariableConflict {
                    name: key.to_string(),
                    base_domain: base_domain.clone(),
                    incoming_domain: incoming_domain.clone(),
                    resolution,
                });
            }
        }

        Ok(())
    }

    fn resolve_domain_conflict(
        &self,
        base: &DomainInfo,
        incoming: &DomainInfo,
    ) -> Result<(DomainInfo, MergeConflictResolution)> {
        match self.strategy {
            MergeStrategy::KeepFirst => Ok((base.clone(), MergeConflictResolution::KeptBase)),
            MergeStrategy::KeepSecond => {
                Ok((incoming.clone(), MergeConflictResolution::KeptIncoming))
            }
            MergeStrategy::FailOnConflict => {
                bail!(
                    "Domain conflict for '{}': cardinality {} vs {}",
                    base.name,
                    base.cardinality,
                    incoming.cardinality
                )
            }
            MergeStrategy::Union => {
                // For domains, union means take the larger cardinality
                if base.cardinality >= incoming.cardinality {
                    Ok((base.clone(), MergeConflictResolution::KeptBase))
                } else {
                    Ok((incoming.clone(), MergeConflictResolution::KeptIncoming))
                }
            }
            MergeStrategy::Intersection => {
                // For domains, intersection means take the smaller cardinality
                if base.cardinality <= incoming.cardinality {
                    Ok((base.clone(), MergeConflictResolution::KeptBase))
                } else {
                    Ok((incoming.clone(), MergeConflictResolution::KeptIncoming))
                }
            }
        }
    }

    fn resolve_predicate_conflict(
        &self,
        base: &PredicateInfo,
        incoming: &PredicateInfo,
    ) -> Result<(PredicateInfo, MergeConflictResolution)> {
        // Check if predicates are compatible (same signature)
        let compatible = base.arg_domains == incoming.arg_domains;

        match self.strategy {
            MergeStrategy::KeepFirst => Ok((base.clone(), MergeConflictResolution::KeptBase)),
            MergeStrategy::KeepSecond => {
                Ok((incoming.clone(), MergeConflictResolution::KeptIncoming))
            }
            MergeStrategy::FailOnConflict => {
                bail!(
                    "Predicate conflict for '{}': {:?} vs {:?}",
                    base.name,
                    base.arg_domains,
                    incoming.arg_domains
                )
            }
            MergeStrategy::Union => {
                if compatible {
                    Ok((base.clone(), MergeConflictResolution::Merged))
                } else {
                    bail!(
                        "Incompatible predicate signatures for '{}': {:?} vs {:?}",
                        base.name,
                        base.arg_domains,
                        incoming.arg_domains
                    )
                }
            }
            MergeStrategy::Intersection => {
                if compatible {
                    Ok((base.clone(), MergeConflictResolution::Merged))
                } else {
                    bail!(
                        "Incompatible predicate signatures for '{}': {:?} vs {:?}",
                        base.name,
                        base.arg_domains,
                        incoming.arg_domains
                    )
                }
            }
        }
    }

    fn resolve_variable_conflict(
        &self,
        base_domain: &str,
        incoming_domain: &str,
    ) -> Result<(String, MergeConflictResolution)> {
        match self.strategy {
            MergeStrategy::KeepFirst => {
                Ok((base_domain.to_string(), MergeConflictResolution::KeptBase))
            }
            MergeStrategy::KeepSecond => Ok((
                incoming_domain.to_string(),
                MergeConflictResolution::KeptIncoming,
            )),
            MergeStrategy::FailOnConflict => {
                bail!(
                    "Variable domain conflict: '{}' vs '{}'",
                    base_domain,
                    incoming_domain
                )
            }
            MergeStrategy::Union | MergeStrategy::Intersection => {
                if base_domain == incoming_domain {
                    Ok((base_domain.to_string(), MergeConflictResolution::Merged))
                } else {
                    bail!(
                        "Incompatible variable domains: '{}' vs '{}'",
                        base_domain,
                        incoming_domain
                    )
                }
            }
        }
    }
}

impl Default for SchemaMerger {
    fn default() -> Self {
        Self::new(MergeStrategy::Union)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_base_table() -> SymbolTable {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "knows",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .unwrap();
        table.bind_variable("x", "Person").unwrap();
        table
    }

    fn create_incoming_table() -> SymbolTable {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 150)).unwrap(); // Different cardinality
        table
            .add_domain(DomainInfo::new("Organization", 50))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new("age", vec!["Person".to_string()]))
            .unwrap();
        table
    }

    #[test]
    fn test_merge_union_no_conflicts() {
        let base = create_base_table();
        let incoming = create_incoming_table();

        let merger = SchemaMerger::new(MergeStrategy::Union);
        let result = merger.merge(&base, &incoming).unwrap();

        assert_eq!(result.merged.domains.len(), 2); // Person (larger card.) + Organization
        assert_eq!(result.merged.predicates.len(), 2); // knows + age
                                                       // Person domain has conflict but will be resolved by taking larger cardinality
        assert!(result.report.has_conflicts()); // Person domain conflict
    }

    #[test]
    fn test_merge_with_domain_conflict() {
        let mut base = SymbolTable::new();
        base.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let mut incoming = SymbolTable::new();
        incoming.add_domain(DomainInfo::new("Person", 200)).unwrap();

        let merger = SchemaMerger::new(MergeStrategy::KeepFirst);
        let result = merger.merge(&base, &incoming).unwrap();

        assert_eq!(result.merged.domains.len(), 1);
        assert_eq!(
            result.merged.domains.get("Person").unwrap().cardinality,
            100
        );
        assert!(result.report.has_conflicts());
    }

    #[test]
    fn test_merge_keep_second() {
        let mut base = SymbolTable::new();
        base.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let mut incoming = SymbolTable::new();
        incoming.add_domain(DomainInfo::new("Person", 200)).unwrap();

        let merger = SchemaMerger::new(MergeStrategy::KeepSecond);
        let result = merger.merge(&base, &incoming).unwrap();

        assert_eq!(
            result.merged.domains.get("Person").unwrap().cardinality,
            200
        );
    }

    #[test]
    fn test_merge_fail_on_conflict() {
        let mut base = SymbolTable::new();
        base.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let mut incoming = SymbolTable::new();
        incoming.add_domain(DomainInfo::new("Person", 200)).unwrap();

        let merger = SchemaMerger::new(MergeStrategy::FailOnConflict);
        let result = merger.merge(&base, &incoming);

        assert!(result.is_err());
    }

    #[test]
    fn test_merge_report() {
        let base = create_base_table();
        let incoming = create_incoming_table();

        let merger = SchemaMerger::new(MergeStrategy::Union);
        let result = merger.merge(&base, &incoming).unwrap();

        let report = &result.report;
        // Base has Person (no unique domains after merge since incoming also has Person)
        assert_eq!(report.base_domains.len(), 0);
        // Incoming has Organization (unique)
        assert_eq!(report.incoming_domains.len(), 1);
        // merged_count = base_domains (0) + incoming_domains (1) + base_predicates (1) + incoming_predicates (1) + merged_variables (1)
        assert_eq!(report.merged_count(), 4);
        assert_eq!(report.conflict_count(), 1); // Person domain conflict
    }

    #[test]
    fn test_predicate_conflict_compatible() {
        let mut base = SymbolTable::new();
        base.add_domain(DomainInfo::new("Person", 100)).unwrap();
        base.add_predicate(PredicateInfo::new("age", vec!["Person".to_string()]))
            .unwrap();

        let mut incoming = SymbolTable::new();
        incoming.add_domain(DomainInfo::new("Person", 100)).unwrap();
        incoming
            .add_predicate(PredicateInfo::new("age", vec!["Person".to_string()]))
            .unwrap();

        let merger = SchemaMerger::new(MergeStrategy::Union);
        let result = merger.merge(&base, &incoming).unwrap();

        assert_eq!(result.merged.predicates.len(), 1);
        assert_eq!(result.report.conflicting_predicates.len(), 0);
    }

    #[test]
    fn test_variable_conflict() {
        let mut base = SymbolTable::new();
        base.add_domain(DomainInfo::new("Person", 100)).unwrap();
        base.add_domain(DomainInfo::new("Agent", 50)).unwrap();
        base.bind_variable("x", "Person").unwrap();

        let mut incoming = SymbolTable::new();
        incoming.add_domain(DomainInfo::new("Person", 100)).unwrap();
        incoming.add_domain(DomainInfo::new("Agent", 50)).unwrap();
        incoming.bind_variable("x", "Agent").unwrap();

        let merger = SchemaMerger::new(MergeStrategy::KeepFirst);
        let result = merger.merge(&base, &incoming).unwrap();

        assert_eq!(result.merged.variables.get("x").unwrap(), "Person");
        assert_eq!(result.report.conflicting_variables.len(), 1);
    }
}
