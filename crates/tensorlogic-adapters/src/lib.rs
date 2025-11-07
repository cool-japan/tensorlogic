//! Adapter utilities for the Tensorlogic ecosystem.
//!
//! This crate provides the bridge between logical expressions and tensor execution
//! by managing symbol tables, domain hierarchies, and schema validation.
//!
//! # Overview
//!
//! `tensorlogic-adapters` offers a comprehensive system for managing the metadata
//! and structure of logical systems compiled to tensor operations. It provides:
//!
//! - **Symbol tables** - Central registry for predicates, domains, and variables
//! - **Domain hierarchies** - Type relationships with subtyping and inheritance
//! - **Parametric types** - Generic domains like `List<T>`, `Option<T>`, `Map<K,V>`
//! - **Predicate composition** - Define predicates in terms of other predicates
//! - **Rich metadata** - Provenance tracking, documentation, version history
//! - **Schema validation** - Completeness, consistency, and semantic checks
//!
//! # Quick Start
//!
//! ```rust
//! use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo};
//!
//! // Create a symbol table
//! let mut table = SymbolTable::new();
//!
//! // Add a domain
//! table.add_domain(DomainInfo::new("Person", 100)).unwrap();
//!
//! // Add a predicate
//! let knows = PredicateInfo::new(
//!     "knows",
//!     vec!["Person".to_string(), "Person".to_string()]
//! );
//! table.add_predicate(knows).unwrap();
//!
//! // Bind a variable
//! table.bind_variable("x", "Person").unwrap();
//! ```
//!
//! # Features
//!
//! ## Domain Hierarchies
//!
//! Define type hierarchies with subtype relationships:
//!
//! ```rust
//! use tensorlogic_adapters::DomainHierarchy;
//!
//! let mut hierarchy = DomainHierarchy::new();
//! hierarchy.add_subtype("Student", "Person");
//! hierarchy.add_subtype("Person", "Agent");
//!
//! assert!(hierarchy.is_subtype("Student", "Agent")); // Transitive
//! ```
//!
//! ## Parametric Types
//!
//! Create generic domain types:
//!
//! ```rust
//! use tensorlogic_adapters::{ParametricType, TypeParameter};
//!
//! let list_person = ParametricType::list(
//!     TypeParameter::concrete("Person")
//! );
//! assert_eq!(list_person.to_string(), "List<Person>");
//! ```
//!
//! ## Schema Validation
//!
//! Validate schemas for correctness:
//!
//! ```rust
//! use tensorlogic_adapters::{SymbolTable, SchemaValidator, DomainInfo};
//!
//! let mut table = SymbolTable::new();
//! table.add_domain(DomainInfo::new("Person", 100)).unwrap();
//!
//! let validator = SchemaValidator::new(&table);
//! let report = validator.validate().unwrap();
//!
//! assert!(report.errors.is_empty());
//! ```

mod axis;
mod builder;
mod compact;
mod compiler_integration;
mod composition;
mod computed;
mod constraint;
mod diff;
mod domain;
mod error;
mod hierarchy;
mod lazy;
mod mask;
mod metadata;
mod parametric;
mod performance;
mod predicate;
mod product;
mod schema_analysis;
mod signature_matcher;
mod symbol_table;
mod validation;

#[cfg(test)]
mod tests;

pub use axis::AxisMetadata;
pub use builder::SchemaBuilder;
pub use compact::{CompactSchema, CompressionStats};
pub use compiler_integration::{
    CompilerExport, CompilerExportBundle, CompilerImport, SymbolTableSync, ValidationResult,
};
pub use composition::{CompositePredicate, CompositeRegistry, PredicateBody, PredicateTemplate};
pub use computed::{ComputedDomain, ComputedDomainRegistry, DomainComputation};
pub use constraint::{FunctionalDependency, PredicateConstraints, PredicateProperty, ValueRange};
pub use diff::{
    check_compatibility, compute_diff, merge_tables, CompatibilityLevel, DiffSummary,
    DomainModification, PredicateModification, SchemaDiff, VariableModification,
};
pub use domain::DomainInfo;
pub use error::AdapterError;
pub use hierarchy::DomainHierarchy;
pub use lazy::{FileSchemaLoader, LazyLoadStats, LazySymbolTable, LoadStrategy, SchemaLoader};
pub use mask::DomainMask;
pub use metadata::{
    Documentation, Example, Metadata, Provenance, TagCategory, TagRegistry, VersionEntry,
};
pub use parametric::{BoundConstraint, ParametricType, TypeBound, TypeParameter};
pub use performance::{CacheStats, LookupCache, MemoryStats, StringInterner};
pub use predicate::PredicateInfo;
pub use product::{ProductDomain, ProductDomainExt};
pub use schema_analysis::{SchemaAnalyzer, SchemaIssue, SchemaRecommendations, SchemaStatistics};
pub use signature_matcher::{MatcherStats, SignatureMatcher};
pub use symbol_table::SymbolTable;
pub use validation::{SchemaValidator, ValidationReport};
