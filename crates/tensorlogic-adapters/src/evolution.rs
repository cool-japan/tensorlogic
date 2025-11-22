//! Schema evolution analysis and migration planning.
//!
//! This module provides sophisticated analysis of schema changes to detect
//! breaking changes, suggest migrations, and ensure backward compatibility.
//!
//! # Overview
//!
//! Schema evolution is critical for production systems. This module goes
//! beyond simple diffs to provide:
//!
//! - **Breaking change detection**: Identify changes that break existing code
//! - **Semantic versioning guidance**: Suggest version bumps (major/minor/patch)
//! - **Migration path generation**: Create executable migration plans
//! - **Compatibility analysis**: Assess forward/backward compatibility
//! - **Deprecation tracking**: Manage deprecated features
//!
//! # Architecture
//!
//! - **EvolutionAnalyzer**: Analyzes schema changes
//! - **BreakingChange**: Categorizes breaking changes by severity
//! - **MigrationPlan**: Executable migration steps
//! - **CompatibilityReport**: Detailed compatibility analysis
//! - **DeprecationPolicy**: Manage deprecation lifecycle
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_adapters::{SymbolTable, DomainInfo, EvolutionAnalyzer};
//!
//! let mut old_schema = SymbolTable::new();
//! old_schema.add_domain(DomainInfo::new("Person", 100)).unwrap();
//!
//! let mut new_schema = SymbolTable::new();
//! new_schema.add_domain(DomainInfo::new("Person", 200)).unwrap();
//!
//! let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
//! let report = analyzer.analyze().unwrap();
//!
//! if report.has_breaking_changes() {
//!     println!("Breaking changes detected!");
//!     for change in &report.breaking_changes {
//!         println!("  - {}", change.description);
//!     }
//! }
//!
//! println!("Suggested version: {}", report.suggested_version_bump());
//! ```

use anyhow::Result;
use std::collections::HashSet;

use crate::SymbolTable;

/// Type of schema change
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChangeKind {
    /// Domain added
    DomainAdded,
    /// Domain removed
    DomainRemoved,
    /// Domain cardinality increased
    DomainCardinalityIncreased,
    /// Domain cardinality decreased
    DomainCardinalityDecreased,
    /// Predicate added
    PredicateAdded,
    /// Predicate removed
    PredicateRemoved,
    /// Predicate arity changed
    PredicateArityChanged,
    /// Predicate signature changed
    PredicateSignatureChanged,
    /// Variable added
    VariableAdded,
    /// Variable removed
    VariableRemoved,
    /// Variable type changed
    VariableTypeChanged,
}

impl ChangeKind {
    /// Check if this change is potentially breaking
    pub fn is_potentially_breaking(&self) -> bool {
        matches!(
            self,
            ChangeKind::DomainRemoved
                | ChangeKind::DomainCardinalityDecreased
                | ChangeKind::PredicateRemoved
                | ChangeKind::PredicateArityChanged
                | ChangeKind::PredicateSignatureChanged
                | ChangeKind::VariableRemoved
                | ChangeKind::VariableTypeChanged
        )
    }
}

/// Impact level of a change
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChangeImpact {
    /// No impact (internal change)
    None,
    /// Minor impact (new features, backward compatible)
    Minor,
    /// Moderate impact (deprecations, behavior changes)
    Moderate,
    /// Major impact (breaking changes)
    Major,
    /// Critical impact (data loss possible)
    Critical,
}

impl ChangeImpact {
    pub fn max(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }
}

/// Breaking change with detailed information
#[derive(Clone, Debug)]
pub struct BreakingChange {
    pub kind: ChangeKind,
    pub impact: ChangeImpact,
    pub description: String,
    pub affected_components: Vec<String>,
    pub migration_hint: Option<String>,
}

impl BreakingChange {
    pub fn new(kind: ChangeKind, impact: ChangeImpact, description: impl Into<String>) -> Self {
        Self {
            kind,
            impact,
            description: description.into(),
            affected_components: Vec::new(),
            migration_hint: None,
        }
    }

    pub fn with_affected(mut self, components: Vec<String>) -> Self {
        self.affected_components = components;
        self
    }

    pub fn with_migration_hint(mut self, hint: impl Into<String>) -> Self {
        self.migration_hint = Some(hint.into());
        self
    }
}

/// Semantic version bump recommendation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VersionBump {
    /// No version change needed
    None,
    /// Patch version bump (bug fixes, internal changes)
    Patch,
    /// Minor version bump (new features, backward compatible)
    Minor,
    /// Major version bump (breaking changes)
    Major,
}

impl std::fmt::Display for VersionBump {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VersionBump::None => write!(f, "none"),
            VersionBump::Patch => write!(f, "patch"),
            VersionBump::Minor => write!(f, "minor"),
            VersionBump::Major => write!(f, "major"),
        }
    }
}

/// Migration step
#[derive(Clone, Debug)]
pub enum MigrationStep {
    /// Add a domain
    AddDomain(String, usize),
    /// Remove a domain
    RemoveDomain(String),
    /// Resize a domain
    ResizeDomain(String, usize),
    /// Add a predicate
    AddPredicate(String, Vec<String>),
    /// Remove a predicate
    RemovePredicate(String),
    /// Rename a predicate
    RenamePredicate(String, String),
    /// Add a variable binding
    AddVariable(String, String),
    /// Remove a variable binding
    RemoveVariable(String),
    /// Update variable type
    UpdateVariableType(String, String),
    /// Custom migration code
    Custom(String),
}

impl MigrationStep {
    pub fn description(&self) -> String {
        match self {
            MigrationStep::AddDomain(name, size) => {
                format!("Add domain '{}' with cardinality {}", name, size)
            }
            MigrationStep::RemoveDomain(name) => format!("Remove domain '{}'", name),
            MigrationStep::ResizeDomain(name, new_size) => {
                format!("Resize domain '{}' to cardinality {}", name, new_size)
            }
            MigrationStep::AddPredicate(name, domains) => {
                format!("Add predicate '{}' with signature {:?}", name, domains)
            }
            MigrationStep::RemovePredicate(name) => format!("Remove predicate '{}'", name),
            MigrationStep::RenamePredicate(old, new) => {
                format!("Rename predicate '{}' to '{}'", old, new)
            }
            MigrationStep::AddVariable(name, domain) => {
                format!("Add variable '{}' of type '{}'", name, domain)
            }
            MigrationStep::RemoveVariable(name) => format!("Remove variable '{}'", name),
            MigrationStep::UpdateVariableType(name, new_type) => {
                format!("Update variable '{}' to type '{}'", name, new_type)
            }
            MigrationStep::Custom(desc) => desc.clone(),
        }
    }

    pub fn is_reversible(&self) -> bool {
        matches!(
            self,
            MigrationStep::AddDomain(_, _)
                | MigrationStep::AddPredicate(_, _)
                | MigrationStep::AddVariable(_, _)
                | MigrationStep::RenamePredicate(_, _)
        )
    }
}

/// Migration plan
#[derive(Clone, Debug)]
pub struct MigrationPlan {
    pub steps: Vec<MigrationStep>,
    pub estimated_complexity: usize,
    pub requires_manual_intervention: bool,
}

impl MigrationPlan {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            estimated_complexity: 0,
            requires_manual_intervention: false,
        }
    }

    pub fn add_step(&mut self, step: MigrationStep) {
        self.estimated_complexity += 1;
        if !step.is_reversible() {
            self.requires_manual_intervention = true;
        }
        self.steps.push(step);
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn is_automatic(&self) -> bool {
        !self.requires_manual_intervention
    }
}

impl Default for MigrationPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility analysis result
#[derive(Clone, Debug)]
pub struct CompatibilityReport {
    /// Breaking changes
    pub breaking_changes: Vec<BreakingChange>,
    /// Backward compatible changes
    pub backward_compatible_changes: Vec<String>,
    /// Forward compatible changes
    pub forward_compatible_changes: Vec<String>,
    /// Deprecations
    pub deprecations: Vec<String>,
    /// Migration plan
    pub migration_plan: MigrationPlan,
}

impl CompatibilityReport {
    pub fn new() -> Self {
        Self {
            breaking_changes: Vec::new(),
            backward_compatible_changes: Vec::new(),
            forward_compatible_changes: Vec::new(),
            deprecations: Vec::new(),
            migration_plan: MigrationPlan::new(),
        }
    }

    pub fn has_breaking_changes(&self) -> bool {
        !self.breaking_changes.is_empty()
    }

    pub fn is_backward_compatible(&self) -> bool {
        !self.has_breaking_changes()
    }

    pub fn suggested_version_bump(&self) -> VersionBump {
        if self.has_breaking_changes() {
            VersionBump::Major
        } else if !self.backward_compatible_changes.is_empty() {
            VersionBump::Minor
        } else {
            VersionBump::Patch
        }
    }

    pub fn max_impact(&self) -> ChangeImpact {
        self.breaking_changes
            .iter()
            .map(|bc| bc.impact.clone())
            .max()
            .unwrap_or(ChangeImpact::None)
    }
}

impl Default for CompatibilityReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Schema evolution analyzer
pub struct EvolutionAnalyzer<'a> {
    old_schema: &'a SymbolTable,
    new_schema: &'a SymbolTable,
}

impl<'a> EvolutionAnalyzer<'a> {
    pub fn new(old_schema: &'a SymbolTable, new_schema: &'a SymbolTable) -> Self {
        Self {
            old_schema,
            new_schema,
        }
    }

    /// Analyze schema evolution
    pub fn analyze(&self) -> Result<CompatibilityReport> {
        let mut report = CompatibilityReport::new();

        self.analyze_domains(&mut report)?;
        self.analyze_predicates(&mut report)?;
        self.analyze_variables(&mut report)?;
        self.generate_migration_plan(&mut report)?;

        Ok(report)
    }

    fn analyze_domains(&self, report: &mut CompatibilityReport) -> Result<()> {
        let old_domains: HashSet<_> = self.old_schema.domains.keys().collect();
        let new_domains: HashSet<_> = self.new_schema.domains.keys().collect();

        // Removed domains (breaking)
        for removed in old_domains.difference(&new_domains) {
            let affected = self.find_predicates_using_domain(removed);
            report.breaking_changes.push(
                BreakingChange::new(
                    ChangeKind::DomainRemoved,
                    ChangeImpact::Major,
                    format!("Domain '{}' was removed", removed),
                )
                .with_affected(affected)
                .with_migration_hint(
                    "Replace references to this domain or remove dependent predicates",
                ),
            );
        }

        // Added domains (backward compatible)
        for added in new_domains.difference(&old_domains) {
            report
                .backward_compatible_changes
                .push(format!("Domain '{}' was added", added));
        }

        // Modified domains
        for domain in old_domains.intersection(&new_domains) {
            let old_info = &self.old_schema.domains[*domain];
            let new_info = &self.new_schema.domains[*domain];

            if new_info.cardinality > old_info.cardinality {
                report.backward_compatible_changes.push(format!(
                    "Domain '{}' cardinality increased from {} to {}",
                    domain, old_info.cardinality, new_info.cardinality
                ));
            } else if new_info.cardinality < old_info.cardinality {
                report.breaking_changes.push(
                    BreakingChange::new(
                        ChangeKind::DomainCardinalityDecreased,
                        ChangeImpact::Critical,
                        format!(
                            "Domain '{}' cardinality decreased from {} to {} (possible data loss)",
                            domain, old_info.cardinality, new_info.cardinality
                        ),
                    )
                    .with_migration_hint(
                        "Ensure all existing data fits within the new cardinality",
                    ),
                );
            }
        }

        Ok(())
    }

    fn analyze_predicates(&self, report: &mut CompatibilityReport) -> Result<()> {
        let old_predicates: HashSet<_> = self.old_schema.predicates.keys().collect();
        let new_predicates: HashSet<_> = self.new_schema.predicates.keys().collect();

        // Removed predicates (breaking)
        for removed in old_predicates.difference(&new_predicates) {
            report.breaking_changes.push(
                BreakingChange::new(
                    ChangeKind::PredicateRemoved,
                    ChangeImpact::Major,
                    format!("Predicate '{}' was removed", removed),
                )
                .with_migration_hint("Remove or replace usages of this predicate"),
            );
        }

        // Added predicates (backward compatible)
        for added in new_predicates.difference(&old_predicates) {
            report
                .backward_compatible_changes
                .push(format!("Predicate '{}' was added", added));
        }

        // Modified predicates
        for predicate in old_predicates.intersection(&new_predicates) {
            let old_pred = &self.old_schema.predicates[*predicate];
            let new_pred = &self.new_schema.predicates[*predicate];

            // Check arity
            if old_pred.arg_domains.len() != new_pred.arg_domains.len() {
                report.breaking_changes.push(
                    BreakingChange::new(
                        ChangeKind::PredicateArityChanged,
                        ChangeImpact::Major,
                        format!(
                            "Predicate '{}' arity changed from {} to {}",
                            predicate,
                            old_pred.arg_domains.len(),
                            new_pred.arg_domains.len()
                        ),
                    )
                    .with_migration_hint("Update all usages to match the new arity"),
                );
            }
            // Check signature
            else if old_pred.arg_domains != new_pred.arg_domains {
                report.breaking_changes.push(
                    BreakingChange::new(
                        ChangeKind::PredicateSignatureChanged,
                        ChangeImpact::Major,
                        format!(
                            "Predicate '{}' signature changed from {:?} to {:?}",
                            predicate, old_pred.arg_domains, new_pred.arg_domains
                        ),
                    )
                    .with_migration_hint("Update argument types in all usages"),
                );
            }
        }

        Ok(())
    }

    fn analyze_variables(&self, report: &mut CompatibilityReport) -> Result<()> {
        let old_variables: HashSet<_> = self.old_schema.variables.keys().collect();
        let new_variables: HashSet<_> = self.new_schema.variables.keys().collect();

        // Removed variables (breaking)
        for removed in old_variables.difference(&new_variables) {
            report.breaking_changes.push(
                BreakingChange::new(
                    ChangeKind::VariableRemoved,
                    ChangeImpact::Moderate,
                    format!("Variable '{}' was removed", removed),
                )
                .with_migration_hint("Remove or replace usages of this variable"),
            );
        }

        // Added variables (backward compatible)
        for added in new_variables.difference(&old_variables) {
            report
                .backward_compatible_changes
                .push(format!("Variable '{}' was added", added));
        }

        // Modified variables
        for variable in old_variables.intersection(&new_variables) {
            let old_type = &self.old_schema.variables[*variable];
            let new_type = &self.new_schema.variables[*variable];

            if old_type != new_type {
                report.breaking_changes.push(
                    BreakingChange::new(
                        ChangeKind::VariableTypeChanged,
                        ChangeImpact::Major,
                        format!(
                            "Variable '{}' type changed from '{}' to '{}'",
                            variable, old_type, new_type
                        ),
                    )
                    .with_migration_hint("Update usages to match the new type"),
                );
            }
        }

        Ok(())
    }

    fn generate_migration_plan(&self, report: &mut CompatibilityReport) -> Result<()> {
        let mut plan = MigrationPlan::new();

        // Generate steps for domain changes
        let old_domains: HashSet<_> = self.old_schema.domains.keys().cloned().collect();
        let new_domains: HashSet<_> = self.new_schema.domains.keys().cloned().collect();

        for added in new_domains.difference(&old_domains) {
            let info = &self.new_schema.domains[added];
            plan.add_step(MigrationStep::AddDomain(added.clone(), info.cardinality));
        }

        for removed in old_domains.difference(&new_domains) {
            plan.add_step(MigrationStep::RemoveDomain(removed.clone()));
        }

        for domain in old_domains.intersection(&new_domains) {
            let old_info = &self.old_schema.domains[domain];
            let new_info = &self.new_schema.domains[domain];

            if old_info.cardinality != new_info.cardinality {
                plan.add_step(MigrationStep::ResizeDomain(
                    domain.clone(),
                    new_info.cardinality,
                ));
            }
        }

        // Generate steps for predicate changes
        let old_predicates: HashSet<_> = self.old_schema.predicates.keys().cloned().collect();
        let new_predicates: HashSet<_> = self.new_schema.predicates.keys().cloned().collect();

        for added in new_predicates.difference(&old_predicates) {
            let pred = &self.new_schema.predicates[added];
            plan.add_step(MigrationStep::AddPredicate(
                added.clone(),
                pred.arg_domains.clone(),
            ));
        }

        for removed in old_predicates.difference(&new_predicates) {
            plan.add_step(MigrationStep::RemovePredicate(removed.clone()));
        }

        // Generate steps for variable changes
        let old_variables: HashSet<_> = self.old_schema.variables.keys().cloned().collect();
        let new_variables: HashSet<_> = self.new_schema.variables.keys().cloned().collect();

        for added in new_variables.difference(&old_variables) {
            let domain = &self.new_schema.variables[added];
            plan.add_step(MigrationStep::AddVariable(added.clone(), domain.clone()));
        }

        for removed in old_variables.difference(&new_variables) {
            plan.add_step(MigrationStep::RemoveVariable(removed.clone()));
        }

        for variable in old_variables.intersection(&new_variables) {
            let old_type = &self.old_schema.variables[variable];
            let new_type = &self.new_schema.variables[variable];

            if old_type != new_type {
                plan.add_step(MigrationStep::UpdateVariableType(
                    variable.clone(),
                    new_type.clone(),
                ));
            }
        }

        report.migration_plan = plan;
        Ok(())
    }

    fn find_predicates_using_domain(&self, domain: &str) -> Vec<String> {
        self.old_schema
            .predicates
            .iter()
            .filter(|(_, pred)| pred.arg_domains.contains(&domain.to_string()))
            .map(|(name, _)| name.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DomainInfo, PredicateInfo};

    #[test]
    fn test_domain_removal_breaking() {
        let mut old_schema = SymbolTable::new();
        old_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let new_schema = SymbolTable::new();

        let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
        let report = analyzer.analyze().unwrap();

        assert!(report.has_breaking_changes());
        assert_eq!(report.suggested_version_bump(), VersionBump::Major);
    }

    #[test]
    fn test_domain_addition_compatible() {
        let old_schema = SymbolTable::new();

        let mut new_schema = SymbolTable::new();
        new_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
        let report = analyzer.analyze().unwrap();

        assert!(!report.has_breaking_changes());
        assert_eq!(report.suggested_version_bump(), VersionBump::Minor);
    }

    #[test]
    fn test_cardinality_increase_compatible() {
        let mut old_schema = SymbolTable::new();
        old_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let mut new_schema = SymbolTable::new();
        new_schema
            .add_domain(DomainInfo::new("Person", 200))
            .unwrap();

        let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
        let report = analyzer.analyze().unwrap();

        assert!(!report.has_breaking_changes());
        assert!(!report.backward_compatible_changes.is_empty());
    }

    #[test]
    fn test_cardinality_decrease_breaking() {
        let mut old_schema = SymbolTable::new();
        old_schema
            .add_domain(DomainInfo::new("Person", 200))
            .unwrap();

        let mut new_schema = SymbolTable::new();
        new_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
        let report = analyzer.analyze().unwrap();

        assert!(report.has_breaking_changes());
        assert_eq!(report.max_impact(), ChangeImpact::Critical);
    }

    #[test]
    fn test_predicate_removal_breaking() {
        let mut old_schema = SymbolTable::new();
        old_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        old_schema
            .add_predicate(PredicateInfo::new(
                "knows",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .unwrap();

        let mut new_schema = SymbolTable::new();
        new_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
        let report = analyzer.analyze().unwrap();

        assert!(report.has_breaking_changes());
        assert_eq!(
            report.breaking_changes[0].kind,
            ChangeKind::PredicateRemoved
        );
    }

    #[test]
    fn test_predicate_signature_change_breaking() {
        let mut old_schema = SymbolTable::new();
        old_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        old_schema
            .add_domain(DomainInfo::new("Location", 50))
            .unwrap();
        old_schema
            .add_predicate(PredicateInfo::new(
                "at",
                vec!["Person".to_string(), "Location".to_string()],
            ))
            .unwrap();

        let mut new_schema = SymbolTable::new();
        new_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        new_schema
            .add_domain(DomainInfo::new("Location", 50))
            .unwrap();
        new_schema
            .add_predicate(PredicateInfo::new(
                "at",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .unwrap();

        let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
        let report = analyzer.analyze().unwrap();

        assert!(report.has_breaking_changes());
    }

    #[test]
    fn test_migration_plan_generation() {
        let mut old_schema = SymbolTable::new();
        old_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let mut new_schema = SymbolTable::new();
        new_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        new_schema
            .add_domain(DomainInfo::new("Location", 50))
            .unwrap();

        let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
        let report = analyzer.analyze().unwrap();

        assert!(!report.migration_plan.is_empty());
        assert_eq!(report.migration_plan.steps.len(), 1);
    }

    #[test]
    fn test_variable_type_change_breaking() {
        let mut old_schema = SymbolTable::new();
        old_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        old_schema
            .add_domain(DomainInfo::new("Location", 50))
            .unwrap();
        old_schema.bind_variable("x", "Person").unwrap();

        let mut new_schema = SymbolTable::new();
        new_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();
        new_schema
            .add_domain(DomainInfo::new("Location", 50))
            .unwrap();
        new_schema.bind_variable("x", "Location").unwrap();

        let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
        let report = analyzer.analyze().unwrap();

        assert!(report.has_breaking_changes());
    }

    #[test]
    fn test_no_changes() {
        let mut old_schema = SymbolTable::new();
        old_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let mut new_schema = SymbolTable::new();
        new_schema
            .add_domain(DomainInfo::new("Person", 100))
            .unwrap();

        let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
        let report = analyzer.analyze().unwrap();

        assert!(!report.has_breaking_changes());
        assert_eq!(report.suggested_version_bump(), VersionBump::Patch);
    }
}
