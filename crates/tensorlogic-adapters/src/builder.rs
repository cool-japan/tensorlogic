//! Builder patterns for convenient schema construction.
//!
//! This module provides fluent builder APIs for constructing symbol tables,
//! making it easier to programmatically create complex schemas.

use anyhow::{bail, Result};

use crate::{DomainHierarchy, DomainInfo, Metadata, PredicateInfo, SymbolTable};

/// Builder for constructing SymbolTable instances.
///
/// Provides a fluent API for building schemas step by step.
///
/// # Example
///
/// ```rust
/// use tensorlogic_adapters::SchemaBuilder;
///
/// let table = SchemaBuilder::new()
///     .domain("Person", 100)
///     .domain("Location", 50)
///     .predicate("at", vec!["Person", "Location"])
///     .build()
///     .unwrap();
///
/// assert_eq!(table.domains.len(), 2);
/// assert_eq!(table.predicates.len(), 1);
/// ```
#[derive(Clone, Debug, Default)]
pub struct SchemaBuilder {
    domains: Vec<DomainInfo>,
    predicates: Vec<PredicateInfo>,
    variables: Vec<(String, String)>,
    hierarchy: Option<DomainHierarchy>,
}

impl SchemaBuilder {
    /// Create a new schema builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a domain with the given name and cardinality.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::SchemaBuilder;
    ///
    /// let builder = SchemaBuilder::new()
    ///     .domain("Person", 100)
    ///     .domain("Location", 50);
    /// ```
    pub fn domain(mut self, name: impl Into<String>, cardinality: usize) -> Self {
        self.domains.push(DomainInfo::new(name.into(), cardinality));
        self
    }

    /// Add a domain with description.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::SchemaBuilder;
    ///
    /// let builder = SchemaBuilder::new()
    ///     .domain_with_desc("Person", 100, "Human entities");
    /// ```
    pub fn domain_with_desc(
        mut self,
        name: impl Into<String>,
        cardinality: usize,
        description: impl Into<String>,
    ) -> Self {
        self.domains
            .push(DomainInfo::new(name.into(), cardinality).with_description(description.into()));
        self
    }

    /// Add a domain with metadata.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SchemaBuilder, Metadata};
    ///
    /// let mut meta = Metadata::new();
    /// meta.add_tag("core");
    ///
    /// let builder = SchemaBuilder::new()
    ///     .domain_with_metadata("Person", 100, meta);
    /// ```
    pub fn domain_with_metadata(
        mut self,
        name: impl Into<String>,
        cardinality: usize,
        metadata: Metadata,
    ) -> Self {
        self.domains
            .push(DomainInfo::new(name.into(), cardinality).with_metadata(metadata));
        self
    }

    /// Add a predicate with the given name and argument domains.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::SchemaBuilder;
    ///
    /// let builder = SchemaBuilder::new()
    ///     .domain("Person", 100)
    ///     .predicate("knows", vec!["Person", "Person"]);
    /// ```
    pub fn predicate<S: Into<String>>(
        mut self,
        name: impl Into<String>,
        arg_domains: Vec<S>,
    ) -> Self {
        let domains: Vec<String> = arg_domains.into_iter().map(|s| s.into()).collect();
        self.predicates
            .push(PredicateInfo::new(name.into(), domains));
        self
    }

    /// Add a predicate with description.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::SchemaBuilder;
    ///
    /// let builder = SchemaBuilder::new()
    ///     .domain("Person", 100)
    ///     .domain("Location", 50)
    ///     .predicate_with_desc("at", vec!["Person", "Location"], "Person at location");
    /// ```
    pub fn predicate_with_desc<S: Into<String>>(
        mut self,
        name: impl Into<String>,
        arg_domains: Vec<S>,
        description: impl Into<String>,
    ) -> Self {
        let domains: Vec<String> = arg_domains.into_iter().map(|s| s.into()).collect();
        self.predicates
            .push(PredicateInfo::new(name.into(), domains).with_description(description.into()));
        self
    }

    /// Bind a variable to a domain.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::SchemaBuilder;
    ///
    /// let builder = SchemaBuilder::new()
    ///     .domain("Person", 100)
    ///     .variable("x", "Person")
    ///     .variable("y", "Person");
    /// ```
    pub fn variable(mut self, var: impl Into<String>, domain: impl Into<String>) -> Self {
        self.variables.push((var.into(), domain.into()));
        self
    }

    /// Add a domain hierarchy relationship.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::SchemaBuilder;
    ///
    /// let builder = SchemaBuilder::new()
    ///     .domain("Agent", 200)
    ///     .domain("Person", 100)
    ///     .subtype("Person", "Agent");
    /// ```
    pub fn subtype(mut self, subtype: impl Into<String>, supertype: impl Into<String>) -> Self {
        if self.hierarchy.is_none() {
            self.hierarchy = Some(DomainHierarchy::new());
        }

        if let Some(hierarchy) = &mut self.hierarchy {
            hierarchy.add_subtype(subtype.into(), supertype.into());
        }

        self
    }

    /// Build the final SymbolTable.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A predicate references an undefined domain
    /// - A variable is bound to an undefined domain
    /// - The domain hierarchy is cyclic
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::SchemaBuilder;
    ///
    /// let result = SchemaBuilder::new()
    ///     .domain("Person", 100)
    ///     .predicate("person", vec!["Person"])
    ///     .build();
    ///
    /// assert!(result.is_ok());
    /// ```
    pub fn build(self) -> Result<SymbolTable> {
        let mut table = SymbolTable::new();

        // Add all domains first
        for domain in self.domains {
            table.add_domain(domain)?;
        }

        // Add predicates (will validate domain references)
        for predicate in self.predicates {
            table.add_predicate(predicate)?;
        }

        // Bind variables (will validate domain references)
        for (var, domain) in self.variables {
            table.bind_variable(var, domain)?;
        }

        // Note: Hierarchy support is a future enhancement
        // This would require adding a hierarchy field to SymbolTable
        // For now, hierarchy information is validated but not stored
        if let Some(hierarchy) = &self.hierarchy {
            // Validate that all referenced domains exist
            for domain in hierarchy.all_domains() {
                if !table.domains.contains_key(&domain) {
                    bail!("Hierarchy references unknown domain: {}", domain);
                }
            }
            // Validate hierarchy is acyclic
            hierarchy.validate_acyclic()?;
            // Future: Store hierarchy in table once SymbolTable supports it
        }

        Ok(table)
    }

    /// Build and validate the schema.
    ///
    /// This performs additional validation beyond the basic build.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::SchemaBuilder;
    ///
    /// let result = SchemaBuilder::new()
    ///     .domain("Person", 100)
    ///     .predicate("knows", vec!["Person", "Person"])
    ///     .build_and_validate();
    ///
    /// assert!(result.is_ok());
    /// ```
    pub fn build_and_validate(self) -> Result<SymbolTable> {
        let table = self.build()?;

        // Run validation
        let validator = crate::SchemaValidator::new(&table);
        let report = validator.validate()?;

        if !report.errors.is_empty() {
            anyhow::bail!("Schema validation failed: {}", report.errors.join(", "));
        }

        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_builder() {
        let table = SchemaBuilder::new()
            .domain("Person", 100)
            .domain("Location", 50)
            .predicate("at", vec!["Person", "Location"])
            .build()
            .unwrap();

        assert_eq!(table.domains.len(), 2);
        assert_eq!(table.predicates.len(), 1);
        assert!(table.get_domain("Person").is_some());
        assert!(table.get_domain("Location").is_some());
        assert!(table.get_predicate("at").is_some());
    }

    #[test]
    fn test_builder_with_variables() {
        let table = SchemaBuilder::new()
            .domain("Person", 100)
            .variable("x", "Person")
            .variable("y", "Person")
            .build()
            .unwrap();

        assert_eq!(table.variables.len(), 2);
        assert_eq!(table.get_variable_domain("x"), Some("Person"));
        assert_eq!(table.get_variable_domain("y"), Some("Person"));
    }

    #[test]
    fn test_builder_with_descriptions() {
        let table = SchemaBuilder::new()
            .domain_with_desc("Person", 100, "Human entities")
            .predicate_with_desc("knows", vec!["Person", "Person"], "Knows relation")
            .build()
            .unwrap();

        let domain = table.get_domain("Person").unwrap();
        assert_eq!(domain.description.as_ref().unwrap(), "Human entities");

        let predicate = table.get_predicate("knows").unwrap();
        assert_eq!(predicate.description.as_ref().unwrap(), "Knows relation");
    }

    #[test]
    fn test_builder_error_undefined_domain() {
        let result = SchemaBuilder::new()
            .predicate("knows", vec!["UndefinedDomain"])
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_fluent_api() {
        // Test that chaining works smoothly
        let table = SchemaBuilder::new()
            .domain("A", 10)
            .domain("B", 20)
            .domain("C", 30)
            .predicate("p1", vec!["A"])
            .predicate("p2", vec!["A", "B"])
            .predicate("p3", vec!["A", "B", "C"])
            .variable("x", "A")
            .variable("y", "B")
            .variable("z", "C")
            .build()
            .unwrap();

        assert_eq!(table.domains.len(), 3);
        assert_eq!(table.predicates.len(), 3);
        assert_eq!(table.variables.len(), 3);
    }

    #[test]
    fn test_builder_with_metadata() {
        let mut meta = Metadata::new();
        meta.add_tag("core");
        meta.add_tag("reasoning");

        let table = SchemaBuilder::new()
            .domain_with_metadata("Person", 100, meta)
            .build()
            .unwrap();

        let domain = table.get_domain("Person").unwrap();
        assert!(domain.metadata.is_some());
        let metadata = domain.metadata.as_ref().unwrap();
        assert!(metadata.tags.contains("core"));
        assert!(metadata.tags.contains("reasoning"));
    }

    #[test]
    fn test_build_and_validate() {
        // Valid schema should pass
        let result = SchemaBuilder::new()
            .domain("Person", 100)
            .predicate("person", vec!["Person"])
            .build_and_validate();

        assert!(result.is_ok());
    }
}
