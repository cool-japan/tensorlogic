# tensorlogic-oxirs-bridge
[![Crate](https://img.shields.io/badge/crates.io-tensorlogic-oxirs-bridge-orange)](https://crates.io/crates/tensorlogic-oxirs-bridge)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/tensorlogic-oxirs-bridge)
[![Tests](https://img.shields.io/badge/tests-541%2F541-brightgreen)](#)
[![Production](https://img.shields.io/badge/status-production_ready-success)](#)

Lightweight RDF/SHACL → TensorLogic integration using oxrdf.

## Overview

Bridges semantic web technologies (RDF, RDFS, OWL, SHACL) with TensorLogic tensor-based reasoning:

- **RDF Schema → SymbolTable**: Extract domains (classes) and predicates (properties)
- **SHACL → TLExpr**: Compile constraints to logical rules
- **Provenance Tracking**: Map RDF entities to tensor indices with RDF*
- **Knowledge Embeddings**: Entity and relation embeddings with cosine/euclidean similarity
- **SPARQL Execution**: Query RDF graphs with SPARQL 1.1
- **GraphQL Bridge**: OxiRS GraphQL integration for schema conversion

## Quick Start

```rust
use tensorlogic_oxirs_bridge::SchemaAnalyzer;

let mut analyzer = SchemaAnalyzer::new();

// Load RDF schema in Turtle format
analyzer.load_turtle(r#"
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix ex: <http://example.org/> .

    ex:Person a rdfs:Class ;
              rdfs:label "Person" .

    ex:knows a rdf:Property ;
             rdfs:domain ex:Person ;
             rdfs:range ex:Person .
"#)?;

// Analyze schema
analyzer.analyze()?;

// Convert to SymbolTable
let table = analyzer.to_symbol_table()?;
assert_eq!(table.domains.len(), 1);
assert_eq!(table.predicates.len(), 1);
```

## Key Features

- **Lightweight**: Uses oxrdf (no heavy oxirs-core dependencies)
- **Turtle Parser**: Load RDF schemas from Turtle files
- **Multiple Formats**: N-Triples, N-Quads, and JSON-LD serialization support
- **Class Extraction**: RDF classes → TensorLogic domains
- **Property Extraction**: RDF properties → TensorLogic predicates
- **Provenance Tracking**: Bidirectional entity ↔ tensor mapping
- **RDF* Export**: Generate provenance statements with metadata
- **SHACL Support**: Advanced constraint compilation with 15+ constraint types
- **GraphQL Integration**: Convert GraphQL schemas to TensorLogic symbol tables
- **SPARQL 1.1 Compilation**: Comprehensive query support (SELECT, ASK, DESCRIBE, CONSTRUCT) with OPTIONAL, UNION patterns, aggregates, GROUP BY/HAVING
- **OWL Reasoning**: RDFS/OWL inference with class hierarchies and property characteristics
- **Validation Reports**: SHACL-compliant validation report generation with Turtle/JSON export
- **Knowledge Embeddings**: TransE/DistMult-style entity and relation embeddings
- **SPARQL Execution**: OxirsSparqlExecutor for querying RDF graphs
- **OxiRS GraphQL Bridge**: OxirsGraphQLBridge for OxiRS-backed GraphQL schemas
- **Streaming RDF**: Memory-efficient large graph processing
- **Triple Indexing**: SPO indexes for O(1) lookups
- **Schema Caching**: In-memory and file-based caching with LRU eviction
- **SPARQL Query Generation** (v0.1.19): `SparqlQuery` builder (SELECT/ASK/CONSTRUCT), `GraphPattern` variants (Triple/Optional/Union/Filter/Bind/Values), `expr_to_sparql()` translator from TLExpr
- **JSON-LD Generation** (v0.1.21): `TlJsonLdContext`, `TlJsonLdNode`, `TlJsonLdDocument`, `context_from_expr()` for TLExpr scanning, standard prefix contexts
- **9 Examples**: Comprehensive examples demonstrating all major features

## Architecture

```
RDF Schema (Turtle)
  ↓ [oxttl parser]
oxrdf::Graph
  ↓ [SchemaAnalyzer]
Extract: Classes, Properties, Domains, Ranges
  ↓
SymbolTable (tensorlogic-adapters)
  ↓
Compiler → Tensors → Backend
  ↑
ProvenanceTracker
  ↓
RDF* / JSON provenance export
```

## Provenance Tracking

Track tensor computations back to RDF entities:

```rust
use tensorlogic_oxirs_bridge::ProvenanceTracker;

let mut tracker = ProvenanceTracker::new();

// Track entity-to-tensor mappings
tracker.track_entity("http://example.org/Person".to_string(), 0);
tracker.track_entity("http://example.org/knows".to_string(), 1);

// Track rule-to-shape mappings
tracker.track_shape(
    "http://example.org/shapes#Rule1".to_string(),
    "knows(x,y) → knows(y,x)".to_string(),
    0
);

// Export as RDF* (quoted triples)
let rdf_star = tracker.to_rdf_star();

// Export as JSON
let json = tracker.to_json()?;
```

## Schema Analysis

The `SchemaAnalyzer` extracts semantic information from RDF:

```rust
let mut analyzer = SchemaAnalyzer::new();
analyzer.load_turtle(turtle_data)?;
analyzer.analyze()?;

// Access extracted classes
for (iri, class_info) in &analyzer.classes {
    println!("Class: {}", class_info.label.as_ref().unwrap_or(&iri));
    println!("  Subclasses: {:?}", class_info.subclass_of);
}

// Access extracted properties
for (iri, prop_info) in &analyzer.properties {
    println!("Property: {}", prop_info.label.as_ref().unwrap_or(&iri));
    println!("  Domain: {:?}", prop_info.domain);
    println!("  Range: {:?}", prop_info.range);
}
```

## IRI Handling

Convert IRIs to local names automatically:

```rust
use tensorlogic_oxirs_bridge::SchemaAnalyzer;

assert_eq!(
    SchemaAnalyzer::iri_to_name("http://example.org/Person"),
    "Person"
);
assert_eq!(
    SchemaAnalyzer::iri_to_name("http://xmlns.com/foaf/0.1#knows"),
    "knows"
);
```

## SHACL Support

Compile SHACL shapes to TLExpr rules:

```rust
use tensorlogic_oxirs_bridge::ShaclConverter;

let converter = ShaclConverter::new(symbol_table);
let rules = converter.convert_to_rules(shacl_turtle)?;
```

### Supported SHACL Constraints

**Cardinality Constraints:**
- `sh:minCount N` → ∃y. property(x, y) (at least N values)
- `sh:maxCount 1` → Uniqueness constraint (at most one value)

**Value Constraints:**
- `sh:class C` → property(x, y) → hasType(y, C)
- `sh:datatype D` → property(x, y) → hasDatatype(y, D)
- `sh:pattern P` → property(x, y) → matchesPattern(y, P)
- `sh:minLength N` → property(x, y) → lengthAtLeast(y, N)
- `sh:maxLength N` → property(x, y) → lengthAtMost(y, N)
- `sh:minInclusive N` → property(x, y) → greaterOrEqual(y, N)
- `sh:maxInclusive N` → property(x, y) → lessOrEqual(y, N)
- `sh:in (v1 v2 v3)` → property(x, y) → (y = v1 ∨ y = v2 ∨ y = v3)

**Logical Constraints:**
- `sh:and (S1 S2)` → All shapes must be satisfied (conjunction)
- `sh:or (S1 S2)` → At least one shape must be satisfied (disjunction)
- `sh:not S` → Shape must not be satisfied (negation)
- `sh:xone (S1 S2)` → Exactly one shape must be satisfied (exclusive-or)

**Shape References:**
- `sh:node S` → property(x, y) → nodeConformsTo(y, S)

## GraphQL Integration

Convert GraphQL schemas to TensorLogic symbol tables:

```rust
use tensorlogic_oxirs_bridge::GraphQLConverter;

let mut converter = GraphQLConverter::new();
let symbol_table = converter.parse_schema(schema)?;
```

### GraphQL Features

- **Type Definitions**: GraphQL types → TensorLogic domains
- **Field Definitions**: GraphQL fields → TensorLogic predicates
- **Scalar Types**: Built-in scalars (String, Int, Float, Boolean, ID)
- **List Types**: Array field support with `[Type]` syntax
- **Required Fields**: Non-null type support with `!` syntax
- **Special Types**: Automatic filtering of Query, Mutation, Subscription types

## Knowledge Embeddings

Entity and relation embeddings for knowledge graph completion:

```rust
use tensorlogic_oxirs_bridge::{KnowledgeEmbeddings, EmbeddingConfig, KGTriple};

let config = EmbeddingConfig::default();
let embeddings = KnowledgeEmbeddings::new(config);

// Compute cosine similarity between vectors
let sim = cosine_similarity(&vec_a, &vec_b);

// Compute euclidean distance
let dist = euclidean_distance(&vec_a, &vec_b);
```

## SHACL Validation Reports

Generate SHACL-compliant validation reports from tensor computations:

```rust
use tensorlogic_oxirs_bridge::{ShaclValidator, ValidationResult, ValidationSeverity};

let validator = ShaclValidator::new();

// Build a complete validation report
let mut report = ValidationReport::new();
report.add_result(ValidationResult::new(
    "http://example.org/person/1",
    "http://example.org/PersonShape",
    "http://www.w3.org/ns/shacl#MinCountConstraintComponent",
    "Missing required email property",
).with_path("http://example.org/email"));

// Export as Turtle (SHACL-compliant RDF)
let turtle = report.to_turtle();

// Export as JSON
let json = report.to_json()?;

// Get summary
println!("{}", report.summary());
```

### Validation Features

- **SHACL-Compliant Reports**: Generate validation reports conforming to W3C SHACL spec
- **Multiple Severity Levels**: Violation, Warning, Info
- **Rich Result Details**: Focus node, result path, value, source shape, constraint component
- **Export Formats**: Turtle (RDF), JSON
- **Constraint Validators**: Pre-built validators for minCount, maxCount, datatype, pattern, etc.
- **Report Statistics**: Track violations, warnings, checked shapes and constraints

## Design Decision: Lightweight oxrdf

This crate uses **oxrdf + oxttl** instead of full **oxirs-core** to avoid:
- Heavy build times (COOLJAPAN ecosystem builds are already slow)
- Complex transitive dependencies
- Memory overhead during compilation

For full SPARQL/federation/GraphQL support, use oxirs-core directly.

## Testing

```bash
cargo nextest run -p tensorlogic-oxirs-bridge
# 541 tests, all passing, zero warnings
```

Key test categories:
- **RDF Schema Tests** (7 tests): Schema parsing, class/property extraction, IRI handling
- **N-Triples Tests** (6 tests): Export, import, roundtrip, escaping
- **JSON-LD Tests** (11 tests): Export, context management, IRI compaction, namespace detection
- **SHACL Tests** (17 tests): All constraint types, logical combinations, complex shapes
- **GraphQL Tests** (7 tests): Type parsing, field extraction, scalar handling
- **SPARQL 1.1 Tests** (24 tests): Query types (SELECT/ASK/DESCRIBE/CONSTRUCT), OPTIONAL/UNION patterns, filter conditions, solution modifiers
- **Validation Tests** (10 tests): Report generation, severity levels, export formats
- **RDF* Tests** (18 tests): Provenance tracking, metadata, statistics
- **OWL Tests** (18 tests): Class hierarchies, property characteristics, restrictions
- **Inference Tests** (13 tests): RDFS reasoning, transitive closure
- **SPARQL Gen Tests**: `SparqlQuery` builder, `GraphPattern` variants, `expr_to_sparql()` TLExpr translation
- **JSON-LD Gen Tests**: `TlJsonLdContext`, `TlJsonLdDocument`, `context_from_expr()`, standard prefix contexts

## Examples

The crate includes 9 comprehensive examples demonstrating different features:

```bash
# 1. Basic RDF schema analysis
cargo run --example 01_basic_schema_analysis -p tensorlogic-oxirs-bridge

# 2. SHACL constraints to TensorLogic rules
cargo run --example 02_shacl_constraints -p tensorlogic-oxirs-bridge

# 3. OWL reasoning and inference
cargo run --example 03_owl_reasoning -p tensorlogic-oxirs-bridge

# 4. GraphQL schema integration
cargo run --example 04_graphql_integration -p tensorlogic-oxirs-bridge

# 5. RDF* provenance tracking
cargo run --example 05_rdfstar_provenance -p tensorlogic-oxirs-bridge

# 6. Complete validation pipeline
cargo run --example 06_validation_pipeline -p tensorlogic-oxirs-bridge

# 7. JSON-LD export
cargo run --example 07_jsonld_export -p tensorlogic-oxirs-bridge

# 8. Performance features (caching, indexing, metadata)
cargo run --example 08_performance_features -p tensorlogic-oxirs-bridge

# 9. Advanced SPARQL 1.1 queries
cargo run --example 09_sparql_advanced -p tensorlogic-oxirs-bridge
```

## SPARQL 1.1 Support

Comprehensive SPARQL 1.1 query compilation to TensorLogic operations:

```rust
use tensorlogic_oxirs_bridge::SparqlCompiler;

let mut compiler = SparqlCompiler::new();
compiler.add_predicate_mapping(
    "http://example.org/knows".to_string(),
    "knows".to_string()
);

// SELECT query with OPTIONAL and FILTER
let query = r#"
    SELECT DISTINCT ?x ?y WHERE {
      ?x <http://example.org/knows> ?y .
      OPTIONAL { ?x <http://example.org/age> ?age }
      FILTER(?x > 18)
    } LIMIT 100 ORDER BY ?y
"#;

let sparql_query = compiler.parse_query(query)?;
let tl_expr = compiler.compile_to_tensorlogic(&sparql_query)?;
```

Supported SPARQL 1.1 features:

**Query Types**:
- SELECT queries (with DISTINCT, LIMIT, OFFSET, ORDER BY)
- ASK queries (boolean existence checks)
- DESCRIBE queries (resource descriptions)
- CONSTRUCT queries (RDF graph construction)

**Graph Patterns**:
- Triple patterns with variables and IRIs
- Multiple patterns combined with AND
- OPTIONAL patterns (left-outer join semantics)
- UNION patterns (disjunction)
- Nested graph patterns with braces

**Filter Conditions**:
- Comparison operators: `>`, `<`, `>=`, `<=`, `=`, `!=`
- BOUND(?var) - check if variable is bound
- isIRI(?var) / isURI(?var) - check if value is IRI
- isLiteral(?var) - check if value is literal
- regex(?var, "pattern") - regular expression matching

**Solution Modifiers**:
- DISTINCT - remove duplicate solutions
- LIMIT N - limit number of results
- OFFSET N - skip first N results
- ORDER BY ?var - sort results

**Aggregate Functions** (GROUP BY/HAVING support):
- COUNT, COUNT(DISTINCT), COUNT(*)
- SUM, AVG, MIN, MAX
- GROUP_CONCAT with separator
- SAMPLE

**Planned (FUTURE)**:
- Execute SPARQL via tensor operations (requires SciRS2 backend)
- Federated SPARQL queries
- Property paths (e.g., `?x foaf:knows+ ?y`)
- GRAPH patterns for named graphs
- BIND and VALUES clauses

## N-Triples and N-Quads Support

Export and import RDF data in N-Triples and N-Quads formats:

```rust
use tensorlogic_oxirs_bridge::SchemaAnalyzer;

let mut analyzer = SchemaAnalyzer::new();
analyzer.load_turtle(turtle_data)?;
analyzer.analyze()?;

// Export to N-Triples
let ntriples = analyzer.to_ntriples();

// Import from N-Triples
let mut analyzer2 = SchemaAnalyzer::new();
analyzer2.load_ntriples(&ntriples)?;
analyzer2.analyze()?;
```

## JSON-LD Support

Full bidirectional JSON-LD support for web integration:

```rust
use tensorlogic_oxirs_bridge::{SchemaAnalyzer, JsonLdContext};

let mut analyzer = SchemaAnalyzer::new();
analyzer.load_turtle(turtle_data)?;
analyzer.analyze()?;

// Export with default context
let jsonld = analyzer.to_jsonld()?;

// Export with custom context
let mut context = JsonLdContext::new();
context.add_prefix("ex".to_string(), "http://example.org/".to_string());
let jsonld_custom = analyzer.to_jsonld_with_context(context)?;

// Import from JSON-LD
let mut analyzer2 = SchemaAnalyzer::new();
analyzer2.load_jsonld(jsonld_str)?;
analyzer2.analyze()?;
```

## Performance Features

```rust
use tensorlogic_oxirs_bridge::{SchemaAnalyzer, SchemaCache};

// Enable indexing and metadata preservation
let mut analyzer = SchemaAnalyzer::new()
    .with_indexing()
    .with_metadata();

analyzer.load_turtle(turtle_data)?;

// Fast indexed lookup
if let Some(index) = analyzer.index() {
    let triples = index.find_by_subject("http://example.org/Person");
}

// Schema caching (20-50x speedup on repeated parses)
let mut cache = SchemaCache::new();
if let Some(cached) = cache.get_symbol_table(turtle) {
    // cache hit
} else {
    let table = analyzer.to_symbol_table()?;
    cache.put_symbol_table(turtle, table.clone());
}
```

## Limitations

Current limitations:
- SPARQL: Execute via tensor operations not yet implemented (requires SciRS2 backend)
- N-Triples: Simplified parser, doesn't handle all edge cases
- GraphQL parsing is simplified (use dedicated parser for production)
- RDF list parsing may not work with all Turtle variants

Planned features (FUTURE):
- SPARQL property paths (e.g., `?x foaf:knows+ ?y`)
- Execute SPARQL via tensor backend
- Federated SPARQL queries
- GraphQL directives → constraint rules
- GraphQL interfaces → domain hierarchies
- RDF/XML format support

## License

Apache-2.0

---

**Part of the TensorLogic ecosystem**: [tensorlogic](https://github.com/cool-japan/tensorlogic)

---

**Status**: Production Ready (v0.1.3 Stable)
**Last Updated**: 2026-08-31
**Tests**: 541/541 passing (100%)
**Examples**: 9 comprehensive examples
**Features**: Full SPARQL 1.1 query support (SELECT/ASK/DESCRIBE/CONSTRUCT + OPTIONAL/UNION + aggregates)
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
