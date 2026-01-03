# tensorlogic-adapters

[![Crates.io](https://img.shields.io/crates/v/tensorlogic-adapters.svg)](https://crates.io/crates/tensorlogic-adapters)
[![Documentation](https://docs.rs/tensorlogic-adapters/badge.svg)](https://docs.rs/tensorlogic-adapters)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Adapter utilities for the Tensorlogic ecosystem: symbol tables, axis metadata, domain management, and schema validation.**

The `tensorlogic-adapters` crate provides the bridge between logical expressions and tensor execution by managing:
- **Symbol tables** (predicates, domains, variables)
- **Axis metadata** (variable-to-axis mappings)
- **Domain information** (cardinalities, elements, hierarchies)
- **Domain masks** (filtering, constraints)
- **Parametric types** (generic domains like List<T>, Option<T>)
- **Predicate composition** (define predicates in terms of others)
- **Rich metadata** (provenance tracking, documentation, tagging)
- **Schema validation** (completeness, consistency, semantic checks)
- **Incremental validation** (efficient revalidation with change tracking)
- **Query planning** (optimized predicate lookups with cost-based strategies)
- **Schema evolution** (breaking change detection, migration planning)

## Features

### Core Symbol Management
- **SymbolTable**: Central registry for domains, predicates, and variables
- **DomainInfo**: Domain definitions with cardinality, elements, and metadata
- **PredicateInfo**: Predicate metadata including arity and domain types
- **AxisMetadata**: Variable-to-axis mappings for einsum notation

### Product Domains (NEW)
- **ProductDomain**: Cartesian product types for cross-domain reasoning
- **Binary/Ternary products**: Convenient constructors for common cases
- **Product cardinality**: Automatic computation from components
- **Projection and slicing**: Extract components from products

### Domain Hierarchy
- **Subtype relationships**: Define type hierarchies (e.g., `Person <: Agent`)
- **Transitive checking**: Automatic subtype resolution through inheritance chains
- **Cycle detection**: Prevent circular type definitions
- **Least common supertype**: Find common ancestors in type hierarchies

### Predicate Constraints
- **Logical properties**: Symmetric, transitive, reflexive, antisymmetric, functional
- **Value ranges**: Min/max constraints with inclusive/exclusive bounds
- **Functional dependencies**: Define relationships between predicate arguments

### Parametric Types
- **Generic domains**: List<T>, Option<T>, Pair<A, B>, Map<K, V>
- **Type parameters**: Support for nested parametric types
- **Type bounds**: Constrain type parameters (e.g., `T: Comparable`)
- **Type substitution**: Instantiate generic types with concrete arguments

### Predicate Composition
- **Composite predicates**: Define predicates in terms of other predicates
- **Macro expansion**: Automatic expansion with parameter substitution
- **Predicate templates**: Create parameterized predicate families
- **Composition operators**: AND, OR, NOT for building complex predicates

### Rich Metadata System
- **Provenance tracking**: Who created/modified, when, from where
- **Version history**: Track changes over time with changelog
- **Documentation**: Long-form docs with examples and usage notes
- **Tagging system**: Categorize and filter symbols with tags
- **Custom attributes**: Extensible key-value metadata

### Schema Validation
- **Completeness checks**: Ensure all referenced domains exist
- **Consistency validation**: Check for duplicates and well-formedness
- **Semantic validation**: Detect unused domains, suggest missing predicates
- **Validation reports**: Detailed reports with errors, warnings, and hints

### Computed Domains (NEW)
- **ComputedDomain**: Virtual domains derived from operations
- **DomainComputation**: Filter, union, intersection, difference, product operations
- **Lazy evaluation**: Compute domains on-demand
- **Cardinality bounds**: Automatic bound computation
- **ComputedDomainRegistry**: Manage computed domain dependencies

### Lazy Loading (NEW)
- **LazySymbolTable**: On-demand loading for huge schemas
- **SchemaLoader**: Pluggable loader trait (file, database, remote)
- **FileSchemaLoader**: Built-in file-based loader
- **Load strategies**: Eager, OnDemand, Predictive, Batched
- **Caching**: LRU cache with statistics

### Import/Export
- **JSON support**: Serialize/deserialize symbol tables to JSON
- **YAML support**: Human-readable schema definitions

### Performance Optimizations
- **Signature matcher**: O(1) predicate lookups by arity/signature
- **String interning**: Reduce memory usage of repeated strings
- **Lookup caching**: LRU cache for frequently accessed data
- **Compact representation**: Binary serialization with compression

### Schema Analysis
- **Statistics**: Comprehensive metrics about schema structure
- **Complexity scoring**: Automated complexity analysis
- **Recommendations**: Detect issues and suggest improvements
- **Usage patterns**: Track domain usage frequency

### Incremental Validation (NEW)
- **ChangeTracker**: Record schema modifications for efficient revalidation
- **DependencyGraph**: Track relationships between components
- **IncrementalValidator**: Validate only affected components (10-100x speedup)
- **ValidationCache**: Cache validation results for unchanged components
- **Batch operations**: Group changes for atomic validation

### Query Planning (NEW)
- **QueryPlanner**: Cost-based query optimization for predicate lookups
- **IndexStrategy**: Multiple index types (hash, range, composite, inverted)
- **PredicateQuery**: Rich query language (by name, arity, signature, pattern)
- **QueryStatistics**: Track access patterns and selectivity
- **Plan caching**: Reuse execution plans for repeated queries

### Schema Evolution (NEW)
- **EvolutionAnalyzer**: Detect breaking changes between schema versions
- **BreakingChange**: Categorize changes by severity and impact
- **MigrationPlan**: Generate executable migration steps
- **CompatibilityReport**: Detailed backward/forward compatibility analysis
- **VersionBump**: Semantic versioning guidance (major/minor/patch)

### CLI Tools
- **schema_validate**: Validate schemas with detailed reports
- **schema_migrate**: Convert, merge, diff schemas

### Schema Builder
- **Fluent API**: Build schemas programmatically with method chaining
- **Type safety**: Validation during construction
- **Convenience**: Simplified schema creation

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tensorlogic-adapters = "0.1.0-alpha.2"
```

## Quick Start

### Basic Symbol Table

```rust
use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo};

// Create a symbol table
let mut table = SymbolTable::new();

// Add a domain
table.add_domain(DomainInfo::new("Person", 100))?;

// Add a predicate
let knows = PredicateInfo::new(
    "knows",
    vec!["Person".to_string(), "Person".to_string()]
);
table.add_predicate(knows)?;

// Bind a variable
table.bind_variable("x", "Person")?;
```

### Using Schema Builder (Recommended)

```rust
use tensorlogic_adapters::SchemaBuilder;

// Build a schema with fluent API
let table = SchemaBuilder::new()
    .domain("Person", 100)
    .domain("Location", 50)
    .predicate("at", vec!["Person", "Location"])
    .predicate("knows", vec!["Person", "Person"])
    .variable("x", "Person")
    .build()?;
```

### Domain Hierarchy

```rust
use tensorlogic_adapters::DomainHierarchy;

let mut hierarchy = DomainHierarchy::new();

// Define a type hierarchy
hierarchy.add_subtype("Student", "Person");
hierarchy.add_subtype("Teacher", "Person");
hierarchy.add_subtype("Person", "Agent");

// Check relationships
assert!(hierarchy.is_subtype("Student", "Agent")); // Transitive
assert!(hierarchy.is_subtype("Teacher", "Person"));

// Find common supertype
let common = hierarchy.least_common_supertype("Student", "Teacher");
assert_eq!(common, Some("Person".to_string()));
```

### Parametric Types

```rust
use tensorlogic_adapters::{ParametricType, TypeParameter};

// Create List<Person>
let list_person = ParametricType::list(
    TypeParameter::concrete("Person")
);
println!("{}", list_person); // "List<Person>"

// Create Map<String, Int>
let map_type = ParametricType::map(
    TypeParameter::concrete("String"),
    TypeParameter::concrete("Int")
);
println!("{}", map_type); // "Map<String, Int>"

// Nested types: List<Option<Person>>
let nested = ParametricType::list(
    TypeParameter::parametric(
        ParametricType::option(TypeParameter::concrete("Person"))
    )
);
println!("{}", nested); // "List<Option<Person>>"
```

### Predicate Composition

```rust
use tensorlogic_adapters::{
    CompositePredicate, CompositeRegistry, PredicateBody
};

// Define a composite predicate
let friend = CompositePredicate::new(
    "friend",
    vec!["x".to_string(), "y".to_string()],
    PredicateBody::And(vec![
        PredicateBody::Reference {
            name: "knows".to_string(),
            args: vec!["x".to_string(), "y".to_string()],
        },
        PredicateBody::Reference {
            name: "trusts".to_string(),
            args: vec!["x".to_string(), "y".to_string()],
        },
    ])
);

// Register in a registry
let mut registry = CompositeRegistry::new();
registry.register(friend)?;

// Expand with concrete arguments
let expanded = registry.expand(
    "friend",
    &["alice".to_string(), "bob".to_string()]
)?;
```

### Rich Metadata

```rust
use tensorlogic_adapters::{DomainInfo, Metadata, Provenance, Documentation};

// Create domain with rich metadata
let mut meta = Metadata::new();

// Add provenance
meta.provenance = Some(
    Provenance::new("Alice", "2025-01-15T10:30:00Z")
        .with_source("schema.yaml", Some(42))
);

// Add tags
meta.add_tag("experimental");
meta.add_tag("reasoning");

// Add custom attributes
meta.set_attribute("complexity", "O(n^2)");

// Add documentation
let mut doc = Documentation::new("Domain for persons in the knowledge base");
doc.add_note("This domain supports reasoning about human entities");
meta.documentation = Some(doc);

// Attach to domain
let domain = DomainInfo::new("Person", 100)
    .with_metadata(meta);
```

### Schema Validation

```rust
use tensorlogic_adapters::{SymbolTable, SchemaValidator};

let mut table = SymbolTable::new();
// ... populate table ...

// Validate schema
let validator = SchemaValidator::new(&table);
let report = validator.validate()?;

// Check for issues
if !report.errors.is_empty() {
    for error in &report.errors {
        eprintln!("Error: {}", error);
    }
}

if !report.warnings.is_empty() {
    for warning in &report.warnings {
        println!("Warning: {}", warning);
    }
}
```

### JSON/YAML Import/Export

```rust
use tensorlogic_adapters::SymbolTable;

// Export to JSON
let json = table.to_json()?;
std::fs::write("schema.json", json)?;

// Import from JSON
let json = std::fs::read_to_string("schema.json")?;
let table = SymbolTable::from_json(&json)?;

// Export to YAML
let yaml = table.to_yaml()?;
std::fs::write("schema.yaml", yaml)?;

// Import from YAML
let yaml = std::fs::read_to_string("schema.yaml")?;
let table = SymbolTable::from_yaml(&yaml)?;
```

## Integration with Compiler

The adapters integrate seamlessly with `tensorlogic-compiler`:

```rust
use tensorlogic_adapters::SymbolTable;
use tensorlogic_compiler::CompilerContext;
use tensorlogic_compiler::passes::symbol_integration;

let mut table = SymbolTable::new();
// ... define schema ...

// Sync with compiler context
let mut ctx = CompilerContext::new();
symbol_integration::sync_context_with_symbol_table(&mut ctx, &table)?;

// Build signature registry for type checking
let registry = symbol_integration::build_signature_registry(&table);
```

## Architecture

### Module Structure

```
tensorlogic-adapters/
├── axis.rs               # Axis metadata for einsum notation
├── compact.rs            # Compact schema representation
├── compiler_integration.rs # Export/import for compiler
├── composition.rs        # Predicate composition system
├── constraint.rs         # Predicate constraints and properties
├── diff.rs               # Schema diff and compatibility
├── domain.rs             # Domain information and management
├── error.rs              # Error types
├── hierarchy.rs          # Domain hierarchy and subtyping
├── mask.rs               # Domain masks for filtering
├── metadata.rs           # Rich metadata with provenance/tags
├── parametric.rs         # Parametric types (List<T>, etc.)
├── performance.rs        # String interning and caching
├── predicate.rs          # Predicate metadata
├── schema_analysis.rs    # Schema statistics and analysis
├── signature_matcher.rs  # Fast predicate lookups
├── symbol_table.rs       # Central symbol table
├── validation.rs         # Schema validation
└── bin/
    ├── schema_validate.rs  # CLI validation tool
    └── schema_migrate.rs   # CLI migration tool
```

### Design Principles

1. **Separation of concerns**: Each module has a single, well-defined responsibility
2. **Composability**: Features can be used independently or combined
3. **Type safety**: Rich type system with validation at construction time
4. **Extensibility**: Metadata and attributes allow domain-specific extensions
5. **Performance**: Efficient data structures with minimal overhead

## Use Cases

### Knowledge Graph Integration
Use domain hierarchies and predicate constraints to model ontologies and validate RDF*/SHACL schemas.

### Machine Learning Pipelines
Parametric types and composition enable defining complex feature transformations and model architectures.

### Symbolic Reasoning
Rich metadata and provenance tracking support explainable AI and audit trails.

### Type-Safe DSLs
Schema validation ensures logical expressions are well-typed before compilation.

## CLI Tools

The crate includes two command-line tools:

### Schema Validator

Validate, analyze, and get statistics for schema files:

```bash
# Basic validation
cargo run --bin schema_validate schema.yaml

# With analysis and statistics
cargo run --bin schema_validate --analyze --stats schema.yaml

# Validate JSON schema
cargo run --bin schema_validate --format json schema.json
```

### Schema Migration Tool

Convert, merge, diff, and check compatibility:

```bash
# Convert JSON to YAML
cargo run --bin schema_migrate convert schema.json schema.yaml

# Merge multiple schemas
cargo run --bin schema_migrate merge schema1.yaml schema2.yaml merged.yaml

# Show diff between versions
cargo run --bin schema_migrate diff old.yaml new.yaml

# Check compatibility
cargo run --bin schema_migrate check old.yaml new.yaml
```

### Incremental Validation Example

```rust
use tensorlogic_adapters::incremental_validation::{
    ChangeTracker, IncrementalValidator
};

let mut table = SymbolTable::new();
let mut tracker = ChangeTracker::new();

// Add domain and track change
table.add_domain(DomainInfo::new("Person", 100))?;
tracker.record_domain_addition("Person");

// Incremental validation (only validates changed components)
let mut validator = IncrementalValidator::new(&table, &tracker);
let report = validator.validate_incremental()?;

println!("Validated {} components, {} cached",
    report.components_validated,
    report.components_cached);
```

### Query Planning Example

```rust
use tensorlogic_adapters::query_planner::{QueryPlanner, PredicateQuery};

let mut planner = QueryPlanner::new(&table);

// Query with automatic optimization
let query = PredicateQuery::by_signature(
    vec!["Person".to_string(), "Person".to_string()]
);
let results = planner.execute(&query)?;

// View statistics
println!("Query stats: {:?}", planner.statistics().top_queries(5));
```

### Schema Evolution Example

```rust
use tensorlogic_adapters::evolution::EvolutionAnalyzer;

let analyzer = EvolutionAnalyzer::new(&old_schema, &new_schema);
let report = analyzer.analyze()?;

if report.has_breaking_changes() {
    for change in &report.breaking_changes {
        println!("⚠️  {}: {}", change.impact, change.description);
        if let Some(hint) = &change.migration_hint {
            println!("   Migration: {}", hint);
        }
    }
}

println!("Suggested version bump: {}", report.suggested_version_bump());
println!("Migration steps: {}", report.migration_plan.steps.len());
```

## Testing

Run the test suite:

```bash
cargo nextest run -p tensorlogic-adapters
```

**Current Test Stats**: 447 tests (all passing, zero warnings) ⬆️ +18 new tests
**Lines of Code**: ~23,000 (production code)
**Examples**: 13 complete examples in `examples/`
**Benchmarks**: 6 comprehensive benchmark suites
**Recent Enhancements v0.1.0-alpha.2**:
- ✨ **Semantic Subtyping** - Practical SMT-based implication checking (+8 tests)
- ✨ **Database Property Tests** - 10 comprehensive proptest validations
- ✨ **Refinement Benchmarks** - 6 benchmark suites measuring type checking performance
- ✨ **Modular Database** - Refactored for maintainability (<2000 lines/file)
- Incremental validation with change tracking (10-100x speedup)
- Cost-based query planning and optimization
- Schema evolution with breaking change detection
- Product domains, computed domains, lazy loading
- CLI integration tests for schema tools

## Performance Considerations

- **IndexMap** used for domains/predicates to preserve insertion order while maintaining O(1) lookup
- **HashSet** used for tags and hierarchy tracking for O(1) membership tests
- **Validation** performed eagerly at construction time to catch errors early
- **Serialization** uses efficient binary formats (JSON/YAML) with serde

## Future Enhancements

See [TODO.md](TODO.md) for the complete roadmap. Upcoming features include:

- **Computed domains**: Domains derived from operations
- **Enhanced composition**: Predicate calculus operators
- **Property-based testing**: QuickCheck-style validation
- **Performance optimization**: Caching, lazy evaluation, memory pooling
- **CLI tools**: Schema validation, migration, and diff tools

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.

## References

- **Tensorlogic Project**: https://github.com/cool-japan/tensorlogic
- **SciRS2 (Scientific Computing)**: https://github.com/cool-japan/scirs
- **OxiRS (RDF/SPARQL)**: https://github.com/cool-japan/oxirs
- **Tensor Logic Paper**: https://arxiv.org/abs/2510.12269

---

**Status**: 🎉 Production Ready (v0.1.0-alpha.2)
**Last Updated**: 2025-12-16 (Alpha.2 Release)
**Tests**: 447/447 passing (100%) ⬆️ +18 new tests (8 subtyping + 10 database property tests)
**Lines of Code**: ~23,000 (production code)
**Examples**: 13 comprehensive examples
**Benchmarks**: 6 comprehensive benchmark suites (250+ benchmark functions)
**Completion**: ~99% (Advanced production features)
**CLI Tools**: 2 production-ready binaries with integration tests
**Code Quality**: Zero warnings, zero clippy issues
**New in 0.1.0-alpha.2**:
- ✨ **Semantic Subtyping** - Practical SMT-based subtyping check for refinement types (+8 tests)
- ✨ **Database Property Tests** - 10 comprehensive proptest validations for database operations
- ✨ **Refinement Benchmarks** - 6 benchmark groups measuring type checking, subtyping, and scaling
- ✨ **Modular Database** - Refactored database.rs for better maintainability (<2000 lines/file)
- ✨ **Incremental Validation** - 10-100x faster revalidation
- ✨ **Query Planner** - Cost-based query optimization
- ✨ **Schema Evolution** - Breaking change detection & migration planning
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
