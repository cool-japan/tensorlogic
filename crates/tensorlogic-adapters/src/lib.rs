//! Adapter utilities for the Tensorlogic ecosystem.
//!
//! **Version**: 0.1.0-beta.1 | **Status**: Production Ready
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

mod autocompletion;
mod axis;
mod builder;
mod codegen;
mod compact;
mod compiler_integration;
mod composition;
mod computed;
mod constraint;
mod database;
mod dependent;
mod diff;
mod domain;
mod effects;
mod embeddings;
mod error;
mod evolution;
mod hierarchy;
mod incremental_validation;
mod lazy;
mod learning;
mod linear;
mod locking;
mod mask;
mod merge_strategies;
mod metadata;
mod parametric;
mod performance;
mod predicate;
mod product;
mod query_cache;
mod query_planner;
mod recommendation;
mod refinement;
mod schema_analysis;
mod signature_matcher;
mod symbol_table;
mod synchronization;
mod utilities;
mod validation;

#[cfg(test)]
mod tests;

pub use autocompletion::{
    AutoCompleter, AutoCompleterStats, DomainSuggestion, PredicateSuggestion, SuggestionSource,
    VariableSuggestion,
};
pub use axis::AxisMetadata;
pub use builder::SchemaBuilder;
pub use codegen::{GraphQLCodegen, PythonCodegen, RustCodegen, TypeScriptCodegen};
pub use compact::{CompactSchema, CompressionStats};
pub use compiler_integration::{
    AdvancedExportBundle, CompilerExport, CompilerExportAdvanced, CompilerExportBundle,
    CompilerImport, CompleteExportBundle, SymbolTableSync, ValidationResult,
};
pub use composition::{CompositePredicate, CompositeRegistry, PredicateBody, PredicateTemplate};
pub use computed::{ComputedDomain, ComputedDomainRegistry, DomainComputation};
pub use constraint::{FunctionalDependency, PredicateConstraints, PredicateProperty, ValueRange};
pub use database::{
    DatabaseStats, MemoryDatabase, SchemaDatabase, SchemaDatabaseSQL, SchemaId, SchemaMetadata,
    SchemaVersion,
};

#[cfg(feature = "sqlite")]
pub use database::SQLiteDatabase;

#[cfg(feature = "postgres")]
pub use database::PostgreSQLDatabase;
pub use diff::{
    check_compatibility, compute_diff, merge_tables, CompatibilityLevel, DiffSummary,
    DomainModification, PredicateModification, SchemaDiff, VariableModification,
};
pub use domain::DomainInfo;
pub use embeddings::{
    Embedding, EmbeddingWeights, SchemaEmbedder, SimilaritySearch, SimilarityStats, EMBEDDING_DIM,
};
pub use error::AdapterError;
pub use hierarchy::DomainHierarchy;
pub use lazy::{FileSchemaLoader, LazyLoadStats, LazySymbolTable, LoadStrategy, SchemaLoader};
pub use learning::{
    ConfidenceScore, DataSample, InferenceConfig, LearningStatistics, SchemaLearner,
};
pub use locking::{LockStats, LockWithTimeout, LockedSymbolTable, Transaction};
pub use mask::DomainMask;
pub use merge_strategies::{
    DomainConflict, MergeConflictResolution, MergeReport, MergeResult, MergeStrategy,
    PredicateConflict, SchemaMerger, VariableConflict,
};
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

// Re-export new modules (already declared above as mod)
pub use evolution::{
    BreakingChange, ChangeImpact, ChangeKind, CompatibilityReport, EvolutionAnalyzer,
    MigrationPlan, MigrationStep, VersionBump,
};
pub use incremental_validation::{
    AffectedComponents, Change, ChangeStats, ChangeTracker, ChangeType, DependencyGraph,
    IncrementalValidationReport, IncrementalValidator, ValidationCache,
};
pub use query_cache::{
    CacheConfig, CacheKey, CachedResult, QueryCache, QueryCacheStats, SymbolTableCache,
};
pub use query_planner::{
    IndexStrategy, PredicatePattern, PredicateQuery, QueryPlan, QueryPlanner, QueryStatistics,
};
pub use recommendation::{
    PatternMatcher, RecommendationContext, RecommendationStrategy, RecommenderStats,
    SchemaRecommender, SchemaScore,
};

// Advanced Type System modules
pub use dependent::{
    patterns as dependent_patterns, DependentType, DependentTypeContext, DependentTypeRegistry,
    DimConstraint, DimExpr, DimRelation,
};
pub use effects::{
    infer_effects, Effect, EffectContext, EffectHandler, EffectRegistry, EffectRow, EffectSet,
    EffectSignature,
};
pub use linear::{
    LinearContext, LinearError, LinearKind, LinearStatistics, LinearType, LinearTypeRegistry,
    Ownership, Resource,
};
pub use refinement::{
    DependentRelation, RefinementContext, RefinementPredicate, RefinementRegistry, RefinementType,
};
pub use synchronization::{
    ApplyResult, ConflictResolution, EventListener, InMemorySyncProtocol, NodeId, SyncChangeType,
    SyncEvent, SyncProtocol, SyncStatistics, SynchronizationManager, VectorClock,
};
pub use utilities::{
    BatchOperations, ConversionUtils, QueryUtils, StatisticsUtils, ValidationUtils,
};
