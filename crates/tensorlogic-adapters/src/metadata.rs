//! Rich metadata system with provenance tracking, documentation, and tagging.
//!
//! This module provides a comprehensive metadata system for domains and predicates,
//! including:
//! - Provenance tracking (who created/modified, when, from where)
//! - Version history
//! - Long-form documentation and examples
//! - Flexible tagging system for organization and filtering
//! - Custom attributes for extensibility

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Rich metadata container for domains and predicates.
///
/// This structure captures all metadata associated with a symbol, including
/// its provenance, documentation, tags, and custom attributes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    /// Provenance information (who, when, where)
    pub provenance: Option<Provenance>,
    /// Long-form documentation
    pub documentation: Option<Documentation>,
    /// Tags for categorization and filtering
    pub tags: HashSet<String>,
    /// Custom key-value attributes
    pub attributes: HashMap<String, String>,
    /// Version history
    pub version_history: Vec<VersionEntry>,
}

/// Provenance information tracking the origin and history of a symbol.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Who created this symbol (user, system, or tool name)
    pub created_by: String,
    /// When it was created (ISO 8601 timestamp)
    pub created_at: String,
    /// Source file or location where it was defined
    pub source_file: Option<String>,
    /// Source line number
    pub source_line: Option<usize>,
    /// Who last modified this symbol
    pub modified_by: Option<String>,
    /// When it was last modified (ISO 8601 timestamp)
    pub modified_at: Option<String>,
    /// Derivation information (if this symbol was derived from others)
    pub derived_from: Vec<String>,
    /// Additional provenance notes
    pub notes: Option<String>,
}

/// Long-form documentation with examples and usage notes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Documentation {
    /// Brief one-line summary
    pub summary: String,
    /// Detailed description (markdown supported)
    pub description: Option<String>,
    /// Usage examples
    pub examples: Vec<Example>,
    /// Usage notes and best practices
    pub notes: Vec<String>,
    /// Related symbols (predicates, domains)
    pub see_also: Vec<String>,
}

/// An example demonstrating how to use a symbol.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Example {
    /// Example title or description
    pub title: String,
    /// Example code or expression
    pub code: String,
    /// Expected output or result
    pub expected_output: Option<String>,
}

/// A version history entry tracking changes to a symbol.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VersionEntry {
    /// Version number (e.g., "1.0.0", "2.1.3")
    pub version: String,
    /// Timestamp of this version (ISO 8601)
    pub timestamp: String,
    /// Who made this change
    pub author: String,
    /// Description of changes
    pub changes: String,
}

impl Metadata {
    /// Creates a new empty metadata container.
    pub fn new() -> Self {
        Metadata::default()
    }

    /// Creates metadata with provenance information.
    pub fn with_provenance(provenance: Provenance) -> Self {
        Metadata {
            provenance: Some(provenance),
            ..Default::default()
        }
    }

    /// Adds a tag to this metadata.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.insert(tag.into());
    }

    /// Removes a tag from this metadata.
    pub fn remove_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }

    /// Checks if this metadata has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }

    /// Checks if this metadata has all of the given tags.
    pub fn has_all_tags(&self, tags: &[String]) -> bool {
        tags.iter().all(|tag| self.tags.contains(tag))
    }

    /// Checks if this metadata has any of the given tags.
    pub fn has_any_tag(&self, tags: &[String]) -> bool {
        tags.iter().any(|tag| self.tags.contains(tag))
    }

    /// Sets a custom attribute.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Gets a custom attribute by key.
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }

    /// Removes a custom attribute.
    pub fn remove_attribute(&mut self, key: &str) -> Option<String> {
        self.attributes.remove(key)
    }

    /// Adds a version entry to the history.
    pub fn add_version(
        &mut self,
        version: impl Into<String>,
        timestamp: impl Into<String>,
        author: impl Into<String>,
        changes: impl Into<String>,
    ) {
        self.version_history.push(VersionEntry {
            version: version.into(),
            timestamp: timestamp.into(),
            author: author.into(),
            changes: changes.into(),
        });
    }

    /// Gets the latest version from the history.
    pub fn latest_version(&self) -> Option<&VersionEntry> {
        self.version_history.last()
    }

    /// Sets the documentation for this symbol.
    pub fn set_documentation(&mut self, doc: Documentation) {
        self.documentation = Some(doc);
    }

    /// Gets the documentation summary, if available.
    pub fn get_summary(&self) -> Option<&str> {
        self.documentation.as_ref().map(|d| d.summary.as_str())
    }
}

impl Provenance {
    /// Creates a new provenance record.
    pub fn new(created_by: impl Into<String>, created_at: impl Into<String>) -> Self {
        Provenance {
            created_by: created_by.into(),
            created_at: created_at.into(),
            source_file: None,
            source_line: None,
            modified_by: None,
            modified_at: None,
            derived_from: Vec::new(),
            notes: None,
        }
    }

    /// Sets the source file location.
    pub fn with_source(mut self, file: impl Into<String>, line: Option<usize>) -> Self {
        self.source_file = Some(file.into());
        self.source_line = line;
        self
    }

    /// Marks this symbol as modified.
    pub fn mark_modified(
        &mut self,
        modified_by: impl Into<String>,
        modified_at: impl Into<String>,
    ) {
        self.modified_by = Some(modified_by.into());
        self.modified_at = Some(modified_at.into());
    }

    /// Adds a derivation source.
    pub fn add_derivation(&mut self, source: impl Into<String>) {
        self.derived_from.push(source.into());
    }

    /// Sets provenance notes.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

impl Documentation {
    /// Creates new documentation with a summary.
    pub fn new(summary: impl Into<String>) -> Self {
        Documentation {
            summary: summary.into(),
            description: None,
            examples: Vec::new(),
            notes: Vec::new(),
            see_also: Vec::new(),
        }
    }

    /// Sets the detailed description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adds an example.
    pub fn add_example(&mut self, example: Example) {
        self.examples.push(example);
    }

    /// Adds a usage note.
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Adds a related symbol reference.
    pub fn add_see_also(&mut self, symbol: impl Into<String>) {
        self.see_also.push(symbol.into());
    }
}

impl Example {
    /// Creates a new example.
    pub fn new(title: impl Into<String>, code: impl Into<String>) -> Self {
        Example {
            title: title.into(),
            code: code.into(),
            expected_output: None,
        }
    }

    /// Sets the expected output.
    pub fn with_output(mut self, output: impl Into<String>) -> Self {
        self.expected_output = Some(output.into());
        self
    }
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Created by: {} at {}", self.created_by, self.created_at)?;
        if let Some(ref file) = self.source_file {
            write!(f, "Source: {}", file)?;
            if let Some(line) = self.source_line {
                write!(f, ":{}", line)?;
            }
            writeln!(f)?;
        }
        if let Some(ref modified_by) = self.modified_by {
            if let Some(ref modified_at) = self.modified_at {
                writeln!(f, "Modified by: {} at {}", modified_by, modified_at)?;
            }
        }
        if !self.derived_from.is_empty() {
            writeln!(f, "Derived from: {}", self.derived_from.join(", "))?;
        }
        Ok(())
    }
}

/// A tag category for organizing tags into groups.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TagCategory {
    /// Category name (e.g., "domain", "application", "status")
    pub name: String,
    /// Human-readable description
    pub description: Option<String>,
    /// Tags in this category
    pub tags: HashSet<String>,
}

impl TagCategory {
    /// Creates a new tag category.
    pub fn new(name: impl Into<String>) -> Self {
        TagCategory {
            name: name.into(),
            description: None,
            tags: HashSet::new(),
        }
    }

    /// Sets the category description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adds a tag to this category.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.insert(tag.into());
    }

    /// Checks if a tag belongs to this category.
    pub fn contains(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }
}

/// A registry of tag categories for organizing the tag namespace.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TagRegistry {
    categories: HashMap<String, TagCategory>,
}

impl TagRegistry {
    /// Creates a new empty tag registry.
    pub fn new() -> Self {
        TagRegistry::default()
    }

    /// Registers a tag category.
    pub fn register_category(&mut self, category: TagCategory) {
        self.categories.insert(category.name.clone(), category);
    }

    /// Gets a category by name.
    pub fn get_category(&self, name: &str) -> Option<&TagCategory> {
        self.categories.get(name)
    }

    /// Finds which category a tag belongs to.
    pub fn find_category_for_tag(&self, tag: &str) -> Option<&str> {
        self.categories
            .iter()
            .find(|(_, cat)| cat.contains(tag))
            .map(|(name, _)| name.as_str())
    }

    /// Creates a standard tag registry with common categories.
    pub fn standard() -> Self {
        let mut registry = TagRegistry::new();

        let mut domain_cat =
            TagCategory::new("domain").with_description("Tags related to problem domains");
        domain_cat.add_tag("person");
        domain_cat.add_tag("location");
        domain_cat.add_tag("time");
        domain_cat.add_tag("organization");
        registry.register_category(domain_cat);

        let mut status_cat =
            TagCategory::new("status").with_description("Tags related to development status");
        status_cat.add_tag("experimental");
        status_cat.add_tag("stable");
        status_cat.add_tag("deprecated");
        registry.register_category(status_cat);

        let mut application_cat =
            TagCategory::new("application").with_description("Tags related to application areas");
        application_cat.add_tag("reasoning");
        application_cat.add_tag("learning");
        application_cat.add_tag("planning");
        application_cat.add_tag("inference");
        registry.register_category(application_cat);

        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_tags() {
        let mut meta = Metadata::new();
        meta.add_tag("experimental");
        meta.add_tag("reasoning");

        assert!(meta.has_tag("experimental"));
        assert!(meta.has_tag("reasoning"));
        assert!(!meta.has_tag("stable"));

        assert!(meta.has_all_tags(&["experimental".to_string(), "reasoning".to_string()]));
        assert!(!meta.has_all_tags(&["experimental".to_string(), "stable".to_string()]));

        assert!(meta.has_any_tag(&["experimental".to_string(), "stable".to_string()]));
    }

    #[test]
    fn test_metadata_attributes() {
        let mut meta = Metadata::new();
        meta.set_attribute("complexity", "O(n^2)");
        meta.set_attribute("author", "Alice");

        assert_eq!(meta.get_attribute("complexity"), Some("O(n^2)"));
        assert_eq!(meta.get_attribute("author"), Some("Alice"));
        assert_eq!(meta.get_attribute("unknown"), None);

        meta.remove_attribute("complexity");
        assert_eq!(meta.get_attribute("complexity"), None);
    }

    #[test]
    fn test_provenance() {
        let prov = Provenance::new("Alice", "2025-01-15T10:30:00Z")
            .with_source("rules.tl", Some(42))
            .with_notes("Imported from legacy system");

        assert_eq!(prov.created_by, "Alice");
        assert_eq!(prov.created_at, "2025-01-15T10:30:00Z");
        assert_eq!(prov.source_file, Some("rules.tl".to_string()));
        assert_eq!(prov.source_line, Some(42));
    }

    #[test]
    fn test_provenance_modification() {
        let mut prov = Provenance::new("Alice", "2025-01-15T10:30:00Z");
        prov.mark_modified("Bob", "2025-01-16T14:20:00Z");

        assert_eq!(prov.modified_by, Some("Bob".to_string()));
        assert_eq!(prov.modified_at, Some("2025-01-16T14:20:00Z".to_string()));
    }

    #[test]
    fn test_provenance_derivation() {
        let mut prov = Provenance::new("System", "2025-01-15T10:30:00Z");
        prov.add_derivation("BaseRule");
        prov.add_derivation("Optimization");

        assert_eq!(prov.derived_from.len(), 2);
        assert!(prov.derived_from.contains(&"BaseRule".to_string()));
    }

    #[test]
    fn test_documentation() {
        let mut doc = Documentation::new("A predicate for checking person relationships")
            .with_description(
                "This predicate checks if two persons have a specific relationship type",
            );

        doc.add_example(Example::new("Basic usage", "knows(alice, bob)"));
        doc.add_note("This predicate is symmetric");
        doc.add_see_also("friend");
        doc.add_see_also("family");

        assert_eq!(doc.summary, "A predicate for checking person relationships");
        assert_eq!(doc.examples.len(), 1);
        assert_eq!(doc.notes.len(), 1);
        assert_eq!(doc.see_also.len(), 2);
    }

    #[test]
    fn test_example() {
        let example =
            Example::new("Simple query", "Person(x)").with_output("[alice, bob, charlie]");

        assert_eq!(example.title, "Simple query");
        assert_eq!(example.code, "Person(x)");
        assert_eq!(
            example.expected_output,
            Some("[alice, bob, charlie]".to_string())
        );
    }

    #[test]
    fn test_version_history() {
        let mut meta = Metadata::new();
        meta.add_version("1.0.0", "2025-01-15T10:00:00Z", "Alice", "Initial version");
        meta.add_version("1.1.0", "2025-01-20T15:30:00Z", "Bob", "Added constraints");

        assert_eq!(meta.version_history.len(), 2);

        let latest = meta.latest_version().unwrap();
        assert_eq!(latest.version, "1.1.0");
        assert_eq!(latest.author, "Bob");
    }

    #[test]
    fn test_tag_category() {
        let mut category = TagCategory::new("domain").with_description("Problem domain tags");

        category.add_tag("person");
        category.add_tag("location");

        assert_eq!(category.name, "domain");
        assert!(category.contains("person"));
        assert!(!category.contains("experimental"));
    }

    #[test]
    fn test_tag_registry() {
        let mut registry = TagRegistry::new();

        let mut domain_cat = TagCategory::new("domain");
        domain_cat.add_tag("person");
        domain_cat.add_tag("location");
        registry.register_category(domain_cat);

        let category = registry.get_category("domain").unwrap();
        assert!(category.contains("person"));

        let found_category = registry.find_category_for_tag("person");
        assert_eq!(found_category, Some("domain"));
    }

    #[test]
    fn test_standard_tag_registry() {
        let registry = TagRegistry::standard();

        assert!(registry.get_category("domain").is_some());
        assert!(registry.get_category("status").is_some());
        assert!(registry.get_category("application").is_some());

        assert_eq!(
            registry.find_category_for_tag("experimental"),
            Some("status")
        );
        assert_eq!(registry.find_category_for_tag("person"), Some("domain"));
        assert_eq!(
            registry.find_category_for_tag("reasoning"),
            Some("application")
        );
    }
}
