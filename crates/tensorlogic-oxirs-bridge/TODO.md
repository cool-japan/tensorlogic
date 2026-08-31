# TensorLogic OxiRS Bridge — TODO

**Status**: Stable | **Version**: 0.1.2 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-30
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

RDF / SPARQL / SHACL / OWL / JSON-LD integration via the OxiRS stack.

## Completed

- [x] Lightweight oxrdf-based implementation (avoiding heavy oxirs-core)
- [x] ProvenanceTracker structure
  - [x] Bidirectional entity ↔ tensor mapping
  - [x] Shape ↔ rule mapping
  - [x] RDF* export for provenance
  - [x] JSON serialization/deserialization
- [x] SchemaAnalyzer structure
  - [x] Turtle parser integration (oxttl)
  - [x] RDF class extraction
  - [x] RDF property extraction
  - [x] Label and comment extraction
  - [x] Domain and range extraction
  - [x] Subclass relationships
  - [x] Optional indexing via `.with_indexing()`
  - [x] Optional metadata preservation via `.with_metadata()`
- [x] SymbolTable generation from RDF schema
- [x] IRI to local name conversion
- [x] **SHACL constraint compilation**
  - [x] Parse SHACL shapes from Turtle format
  - [x] NodeShape extraction with targetClass
  - [x] PropertyShape extraction from blank nodes
  - [x] Convert constraints to TLExpr (15+ constraint types)
  - [x] Basic constraints: `sh:minCount`, `sh:maxCount`, `sh:class`, `sh:datatype`, `sh:pattern`
  - [x] String constraints: `sh:minLength`, `sh:maxLength`
  - [x] Numeric constraints: `sh:minInclusive`, `sh:maxInclusive`
  - [x] Value constraints: `sh:in` (enumeration)
  - [x] Shape references: `sh:node`
  - [x] Logical operators: `sh:and`, `sh:or`, `sh:not`, `sh:xone`
- [x] **GraphQL Integration**
  - [x] GraphQL schema parsing
  - [x] Type definitions → TensorLogic domains
  - [x] Field definitions → TensorLogic predicates
  - [x] Scalar type handling (String, Int, Float, Boolean, ID)
  - [x] List and required field support
  - [x] Automatic filtering of special types (Query, Mutation, Subscription)
- [x] **OxiRS GraphQL Bridge** (`oxirs_graphql` module)
  - [x] OxirsGraphQLBridge for OxiRS-backed schema conversion
  - [x] GraphQLObjectType, GraphQLField, GraphQLType structures
- [x] **SHACL Validation Reports**
  - [x] ValidationResult structure with full SHACL compliance
  - [x] ValidationReport with conforms flag and statistics
  - [x] Severity levels (Violation, Warning, Info)
  - [x] Export to Turtle (SHACL-compliant RDF)
  - [x] Export to JSON
  - [x] ShaclValidator with pre-built constraint checkers
  - [x] End-to-end validation pipeline example
- [x] **OWL support**
  - [x] Parse OWL class definitions (owl:Class, owl:equivalentClass)
  - [x] Handle owl:equivalentClass, disjointWith, complementOf
  - [x] Support owl:unionOf, owl:intersectionOf
  - [x] Parse owl:Restriction (someValuesFrom, allValuesFrom, cardinality constraints)
  - [x] Translate property characteristics (functional, inverse_functional, transitive, symmetric, reflexive, irreflexive, asymmetric)
  - [x] Property inverses and equivalences
  - [x] 18 comprehensive tests
- [x] **RDFS inference**
  - [x] Apply rdfs:subClassOf transitivity
  - [x] Property domain/range inheritance
  - [x] Type propagation
  - [x] Subproperty inference with transitivity
  - [x] Materialized graph generation
  - [x] Query methods (is_subclass_of, is_subproperty_of, get_all_superclasses, get_all_superproperties)
  - [x] Circular hierarchy handling (prevents infinite loops)
  - [x] Integration with SchemaAnalyzer
  - [x] Inference statistics tracking
  - [x] 13 comprehensive tests
- [x] **RDF* (RDF-star) provenance**
  - [x] Parse quoted triples
  - [x] Track statement-level metadata
  - [x] Generate provenance graphs
  - [x] RdfStarProvenanceStore with indexing
  - [x] MetadataBuilder for fluent API
  - [x] Integration with ProvenanceTracker
  - [x] Confidence score tracking
  - [x] Source attribution
  - [x] Rule ID tracking
  - [x] Temporal tracking (timestamps)
  - [x] Custom metadata support
  - [x] Export to RDF* Turtle format
  - [x] Export to JSON format
  - [x] Querying by confidence, source, rule, predicate
  - [x] Provenance statistics
  - [x] 18 comprehensive tests
- [x] **N-Triples Support**
  - [x] N-Triples parser (load_ntriples)
  - [x] N-Triples export (to_ntriples)
  - [x] Escape/unescape literals
  - [x] Round-trip conversion support
  - [x] 6 comprehensive tests
- [x] **JSON-LD Support**
  - [x] JSON-LD export (to_jsonld)
  - [x] JSON-LD import (load_jsonld)
  - [x] Custom context support (to_jsonld_with_context)
  - [x] Context parsing and IRI expansion
  - [x] Language-tagged literal handling
  - [x] Namespace auto-detection
  - [x] IRI compaction with prefixes
  - [x] Valid JSON output with @context and @graph
  - [x] Roundtrip conversion support
  - [x] 18 comprehensive tests
- [x] **N-Quads Support**
  - [x] N-Quads parser (NQuadsProcessor with graph support)
  - [x] N-Quads serialization (to_nquads)
  - [x] Named graph handling
  - [x] 10 comprehensive tests
- [x] **SPARQL 1.1 Comprehensive Support**
  - [x] Parse SPARQL queries (SELECT with WHERE and FILTER)
  - [x] Compile SPARQL to TLExpr
  - [x] Pattern element parsing (variables and constants)
  - [x] Filter conditions (equals, not equals, greater than, less than, regex, BOUND, isIRI, isLiteral)
  - [x] Triple pattern compilation
  - [x] IRI to predicate mapping
  - [x] Query types: SELECT, ASK, DESCRIBE, CONSTRUCT
  - [x] Graph patterns: OPTIONAL (left-outer join), UNION (disjunction)
  - [x] Solution modifiers: LIMIT, OFFSET, ORDER BY, DISTINCT
  - [x] Aggregate functions: COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE
  - [x] GROUP BY clause
  - [x] HAVING conditions
  - [x] SelectElement type for variables and aggregates
  - [x] 28+ comprehensive tests
- [x] **Triple Indexing**
  - [x] Build indexes for fast property lookup (TripleIndex)
  - [x] Subject/predicate/object indexes
  - [x] Prefix-based search
  - [x] Pattern matching (SPO wildcards)
  - [x] Graph analytics (degree, frequency)
  - [x] 13 comprehensive tests
- [x] **Schema Caching**
  - [x] Cache parsed schemas (SchemaCache)
  - [x] Cache SymbolTable generation
  - [x] In-memory caching with TTL
  - [x] Persistent file-based caching (PersistentCache)
  - [x] LRU eviction strategy
  - [x] Cache statistics
  - [x] 7 comprehensive tests
- [x] **Error Handling**
  - [x] ParseLocation with line/column tracking
  - [x] Context-aware error reporting
  - [x] Suggestion system for common errors
  - [x] Structured error types (InvalidIri, MissingField, etc.)
  - [x] Pretty error formatting
- [x] **Metadata Preservation**
  - [x] Keep original IRIs (EntityMetadata)
  - [x] Store labels in multiple languages (LangString)
  - [x] Language-tagged strings with fallback
  - [x] Custom annotation properties
  - [x] Metadata statistics and quality checks
  - [x] Find by label (search functionality)
  - [x] Find missing metadata
  - [x] JSON export/import
  - [x] 8 comprehensive tests
- [x] **Streaming RDF Processing**
  - [x] Memory-efficient streaming for large RDF datasets
  - [x] StreamingRdfLoader with callback-based processing
  - [x] Batch processing with configurable batch sizes
  - [x] Progress tracking and statistics
  - [x] Predicate and subject prefix filtering
  - [x] StreamAnalyzer for on-the-fly dataset analysis
  - [x] N-Triples line-by-line processing
  - [x] 7 comprehensive tests
- [x] **SymbolTable Export/Import**
  - [x] symbol_table_to_turtle() - Export to Turtle format
  - [x] symbol_table_to_json() - Export to JSON format
  - [x] symbol_table_from_json() - Import from JSON format
  - [x] Roundtrip support
  - [x] 4 comprehensive tests
- [x] **Knowledge Embeddings** (`knowledge_embeddings` module)
  - [x] KnowledgeEmbeddings structure
  - [x] EmbeddingConfig with configurable dimensions and model type
  - [x] KGTriple for knowledge graph triples
  - [x] EmbeddingModel enum (TransE, DistMult, etc.)
  - [x] cosine_similarity function
  - [x] euclidean_distance function
- [x] **SPARQL Executor** (`oxirs_executor` module)
  - [x] OxirsSparqlExecutor for querying RDF graphs
  - [x] QueryResults and QueryValue types
  - [x] TripleResult for graph query results
- [x] **Compile RDF→TLExpr→Tensor pipeline**
  - [x] compile_rules function
  - [x] Full end-to-end example (10_end_to_end_pipeline.rs)
  - [x] Handle large schemas efficiently
  - [x] Stream processing for big graphs (StreamingRdfLoader)
- [x] **Real-World Ontology Tests**
  - [x] FOAF (Friend of a Friend) ontology tests
  - [x] Dublin Core metadata vocabulary tests
  - [x] SKOS (Simple Knowledge Organization System) tests
  - [x] Schema.org vocabulary tests
- [x] **Property-Based Testing**
  - [x] Generate random valid RDF schemas
  - [x] Test round-trip conversion (N-Quads, literals)
  - [x] Verify invariants (StreamAnalyzer, graph separation)
  - [x] 12 proptest tests with 100 cases each
- [x] **Benchmarks**
  - [x] Parsing speed (Turtle, N-Quads)
  - [x] Streaming loader performance
  - [x] SPARQL parsing and compilation
  - [x] Schema analysis and SymbolTable conversion
  - [x] Criterion-based benchmarks with throughput metrics
- [x] **Comprehensive Examples** (9 examples)
  - [x] 01_basic_schema_analysis.rs - RDF schema loading with FOAF
  - [x] 02_shacl_constraints.rs - SHACL constraint parsing and categorization
  - [x] 03_owl_reasoning.rs - OWL hierarchies and RDFS inference
  - [x] 04_graphql_integration.rs - GraphQL schema to TensorLogic
  - [x] 05_rdfstar_provenance.rs - RDF* provenance tracking
  - [x] 06_validation_pipeline.rs - End-to-end validation workflow
  - [x] 07_jsonld_export.rs - JSON-LD import/export
  - [x] 08_performance_features.rs - Indexing and caching
  - [x] 09_sparql_advanced.rs - Advanced SPARQL compilation
  - [x] 10_end_to_end_pipeline.rs - Complete RDF→TLExpr pipeline

## High Priority

### Integration
- [x] Execute and validate
  - [x] Run compiled rules with SciRS2 backend
  - [x] Generate validation reports from execution
  - [x] Export results as RDF

## Medium Priority

### Performance
- [ ] Optimize RDF parsing (FUTURE)
  - [x] Use bulk loading for large graphs — `InternedGraph::from_rdf_triples(Vec<RdfTriple>)` + `rdf_bulk_importer_into_interned` bridge
  - [x] Parallel triple processing — `std::thread::scope` chunked parallel N-Triples parsing in `interned_graph.rs`
  - [x] Memory-efficient graph representation — `InternedGraph` with O(1) term dictionary + predicate-indexed adjacency lists (vs. O(n) linear scan in `DomainInfo::get_index`)

### Error Handling
- [x] **Validation warnings** ✅ (v0.1.2) — SchemaWarningAnalyzer (MissingLabel, UnusedClass, SuggestSHACL)
- [ ] Recovery strategies (FUTURE)
  - [ ] Continue parsing after non-fatal errors
  - [ ] Provide partial results on failure
  - [ ] Auto-fix common issues

### Metadata Management
- [ ] Versioning (FUTURE)
  - [ ] Track schema versions
  - [ ] Schema evolution support
  - [ ] Migration scripts

## Low Priority

### Documentation
- [x] Add README.md (COMPLETE)
- [x] API documentation (MOSTLY COMPLETE)
- [ ] Tutorial
  - [ ] Step-by-step RDF schema import tutorial
  - [ ] SHACL constraint compilation tutorial
  - [ ] Provenance tracking guide

### Testing
- [ ] Extended test coverage
  - [ ] Test with real-world ontologies at scale (>1M triples)
  - [ ] Test with complex nested SHACL shapes

### Formats Support
- [ ] Export formats
  - [x] Export SymbolTable as Turtle (.ttl) ✅ (v0.1.2)
  - [ ] RDF/XML parser (via external crate) (FUTURE)

### Tooling
- [ ] CLI tool
  - [ ] Convert RDF to SymbolTable (JSON)
  - [ ] Validate SHACL shapes
  - [ ] Generate TLExpr from SHACL
- [ ] Visualization
  - [ ] Visualize RDF schema as graph
  - [ ] Show SymbolTable structure
  - [ ] Display provenance chains

## Future Enhancements

### Advanced RDF Features
- [x] **Blank node handling** ✅ (v0.1.2) — BlankNodeManager with fresh IRI generation
- [x] `named-graphs-multiple` (planned 2026-04-17)
  - **Goal:** First-class multi-named-graph storage and per-graph query API. Extend `NQuadsProcessor` and add a new `QuadStore` wrapping per-graph `TripleStore`s.
  - **Design:** `pub struct QuadStore { stores: HashMap<Option<String>, TripleStore> }` — `None` = default graph; `Some(iri)` = named graph. Methods: `insert_quad(&mut self, q: Quad)`, `query_subject(graph: Option<&str>, s: &str) -> Vec<Triple>`, `query_predicate(...)`, `query_object(...)`, `iter_graphs() -> impl Iterator<Item = &Option<String>>`. Reuse existing per-graph triple operations on `TripleStore` (`src/property_path.rs:123–195`); TripleStore itself stays graph-unaware. Bridge: `NQuadsProcessor::into_quad_store(self) -> QuadStore`. SCOPE EXPLICITLY EXCLUDES: SPARQL GRAPH pattern, cross-graph queries, any edits to `sparql/types.rs` / `sparql/compiler.rs` / `oxirs_executor.rs` — those are item 8's territory.
  - **Files:** `src/quad_store.rs` (NEW); `src/schema/nquads.rs` (add `into_quad_store`); `src/lib.rs` (re-export `QuadStore`). `src/property_path.rs` — NO changes.
  - **Prerequisites:** none.
  - **Tests:** ingest n-quads with three named graphs + default graph; per-graph query isolation; default-graph isolation; `iter_graphs` enumerates exactly the inputs.
  - **Risk:** `lib.rs` re-export overlap with item 8 — edits target distinct non-adjacent lines; conflict-free.
- [ ] Reification
  - [ ] Handle RDF reification statements
  - [ ] Convert to RDF* where possible

### SPARQL Integration
- [x] Execute SPARQL via tensor operations — `sparql/tensor_eval.rs`: `TensorBgpEvaluator` evaluates conjunctive SELECT/BGP queries as boolean tensor contraction via `EinsumGraph` + `Scirs2Exec::forward`; supported: fixed-pred triple patterns, var+const in S/O, projection; unsupported forms return `BridgeError::ValidationError`
- [ ] Federated SPARQL queries
- [ ] SPARQL property paths (e.g., `?x foaf:knows+ ?y`)
- [ ] GRAPH patterns for named graphs
- [x] `bind-and-values-clauses` (planned 2026-04-17, completed 2026-04-17)
  - **Goal:** Add full first-class support for SPARQL `BIND ( expr AS ?var )` and `VALUES (?v1 ?v2) { (t1 t2) … }` to the parser, AST, IR-lowering compiler, executor, and sparql_gen — so a query string with these clauses parses, lowers, executes, and round-trips through generation.
  - **Design:**
    - **C1. AST** — `sparql/types.rs`: add `GraphPattern::Bind(BindExpr, String)` and `GraphPattern::Values(Vec<String>, Vec<Vec<PatternElement>>)`; add `pub enum BindExpr { Term(PatternElement) }`. Re-export `BindExpr` and `GraphPattern` from `lib.rs`.
    - **C2. Parser** — `sparql/compiler.rs`: `parse_graph_pattern` new branches for `BIND ( <elem> AS ?var )` (single-var and multi-var token shapes) and `VALUES ?v { … }` / `VALUES (?v1 ?v2) { (t1 t2) … }` (multi-var grouped). Fix `split_sparql_statements` to be brace+paren depth-aware (VALUES `{}` blocks must stay as one statement). `compile_graph_pattern`: lower `Bind` to `TLExpr::pred("eq", [var, term])`, lower `Values` to disjunction of equality conjunctions.
    - **C3. Executor** — `oxirs_executor.rs`: `Values` arm produces one binding per row (fully executable, store-independent). `Bind` constant arm produces `{var: QueryValue}`. Bind variable-ref limitation (no per-row context) documented; produces empty binding + filed as follow-up.
    - **C4. sparql_gen** — `sparql_gen/types.rs`: add `Values(Vec<String>, Vec<Vec<SparqlTerm>>)`. `sparql_gen/functions.rs`: render `VALUES (?v1 ?v2) { (t1 t2) … }`.
    - **C5. sparql_builder** — add `WhereClauseItem::ValuesMulti(Vec<String>, Vec<Vec<SparqlTerm>>)` + `values_multi()` method + render arm. Keep existing `Values` variant.
  - **Files:** `sparql/types.rs`, `sparql/compiler.rs`, `oxirs_executor.rs`, `sparql_gen/types.rs`, `sparql_gen/functions.rs`, `sparql_builder.rs`, `lib.rs`, new `tests/bind_values_roundtrip.rs`.
  - **Tests:** in-source parser tests (parse BIND + VALUES into correct AST), executor tests (`test_execute_values_single_var`, `test_execute_values_multi_var`, `test_execute_bind_constant`), builder test for `values_multi`, integration test `tests/bind_values_roundtrip.rs` (parse → AST → render → re-parse → equal AST).
  - **Risk:** `split_sparql_statements` splitter may break on VALUES `{}` body — must fix with brace-depth tracking before adding VALUES parse branch.
  - **Follow-ups (NOT this run):** per-row context for Bind/Filter arithmetic; `BindExpr` variants for arithmetic/function-calls; `UNDEF` in VALUES rows.
- [ ] SPARQL expression constraints (SPARQL-based SHACL-AF)

### Reasoning
- [ ] OWL reasoning integration
  - [ ] Materialize inferred triples
  - [ ] Compile reasoning rules to tensors
  - [ ] Incremental reasoning
- [ ] Rule engines
  - [ ] Support N3 rules
  - [ ] Support SWRL rules
  - [ ] Custom rule languages

### Semantic Web Standards
- [ ] SKOS support (taxonomies)
- [ ] Dublin Core metadata
- [ ] Schema.org vocabulary
- [ ] DCAT (data catalogs)

## v0.1.3 Enhancements (2026-03-30)

- [x] **SHACL Report Export** (`shacl/report_export.rs`): `ShaclReportExporter` with Turtle/NTriples/JSON-LD output formats using W3C SHACL vocabulary (sh:ValidationReport, sh:result, sh:resultSeverity). File I/O support. 12 new tests.
- [x] **Ontology Diff** (`ontology_diff.rs`): `OntologyDiff`, `DiffEntry` (Added/Removed/Modified), `compare_symbol_tables()` compares domain and predicate sets between two SymbolTables. 10 new tests.

## v0.1.5 Enhancements (2026-03-30)

- [x] **SPARQL Property Paths** (`property_path.rs`): `PropertyPath` enum (7 variants: Iri/Sequence/Alternative/ZeroOrMore/OneOrMore/ZeroOrOne/Inverse). `TripleStore` with forward/reverse lookups. `PropertyPathExpander` with fixed-point closure, cycle-safe visited-set tracking, and max-depth guard. `Display` impl for SPARQL syntax. 16 new tests.

## v0.1.8 Enhancements (2026-03-30)

- [x] **RDF Graph Statistics** (`graph_stats.rs`): `GraphStats` (density, degree distributions, self-loops), `PredicateStats` (functional/inverse-functional detection), `DegreeDistribution` (median), `connected_components()` via union-find. 18 new tests.

## v0.1.16

- [x] **RdfBulkImporter** (`rdf_bulk_io.rs`): Streaming multi-format bulk loader accepting Turtle, N-Triples, and N-Quads sources; processes triples in configurable batches with optional progress callbacks and per-batch error recovery
- [x] **RdfBulkExporter** (`rdf_bulk_io.rs`): High-throughput parallel serializer for large `TripleStore` graphs; supports Turtle, N-Triples, and N-Quads output formats with configurable chunk size and a `NamespaceRegistry` for prefix compaction
- [x] **NamespaceRegistry** (`rdf_bulk_io.rs`): Prefix-to-IRI mapping with `register()`, `compact_iri()` (longest-prefix lookup returning `prefix:local`), `expand_iri()`, and bulk `to_turtle_prefixes()` serialization
- [x] **RdfTriple** (`rdf_bulk_io.rs`): Lightweight plain-struct triple (`subject`, `predicate`, `object` as `String`) for zero-copy bulk transfer between loader, store, and exporter without oxrdf allocation overhead
- [x] **BulkIoStats** (`rdf_bulk_io.rs`): Tracks `triples_processed`, `batches_completed`, `elapsed_secs`, and derives `throughput_triples_per_sec()` for import and export performance reporting

## v0.1.19 (2026-04-06)

- [x] **SPARQL Query Generation** (`sparql_gen.rs`): `SparqlQuery` builder (SELECT/ASK/CONSTRUCT) assembling a list of `GraphPattern` nodes; `GraphPattern` enum with `Triple`, `Optional`, `Union`, `Filter`, and `Bind` variants (NOTE: Values variant was ADDED in the bind-and-values-clauses plan item (2026-04-17); the original claim that it already existed was incorrect); `SparqlFilter` covering equality, comparison, `BOUND`, `REGEX`, `NOT EXISTS`, and `IN` operators; `SparqlGenConfig` with IRI prefix map and depth limit; `expr_to_sparql()` top-level translator from `TLExpr` to `SparqlQuery` — `And` becomes a sequence of patterns, `Or` becomes `UNION`, `Not` / `FuzzyNot` emit `FILTER NOT EXISTS`, `Exists` / `SoftExists` introduce fresh variables, `Imply` / `FuzzyImplication` desugar to `Or(Not(premise), conclusion)`.

## v0.1.21 (2026-04-06)

- [x] **JSON-LD Generation** (`json_ld.rs`): Added json_ld.rs — JSON-LD generation: `ContextTerm`, `TlJsonLdContext` (`expand_iri`, prefixes), `TlJsonLdNode` (@id/@type/properties), `TlJsonLdDocument` (compact+pretty serialization); `context_from_predicates()`, `standard_prefixes_context()`, `context_from_expr()` scanning `TLExpr::Pred` nodes, `expr_to_json_ld_node()`.

## v0.1.11

- [x] **SelectQuery + AskQuery + WhereClause** (`sparql_builder.rs`): `SelectQuery` builder with variable projection, DISTINCT flag, LIMIT/OFFSET pagination; `AskQuery` boolean-result form; `WhereClause` composing one or more `TriplePattern` and optional `SparqlFilter` expressions.
- [x] **TriplePattern + SparqlFilter + SparqlTerm** (`sparql_builder.rs`): `TriplePattern` with subject/predicate/object each as a `SparqlTerm` (IRI/Literal/Variable/BlankNode); `SparqlFilter` supporting comparison and logical expressions; `Display` impls emitting valid SPARQL 1.1 syntax.

### 2026-04-14 — File split refactor

- Split `src/sparql.rs` (1846L) into `src/sparql/` directory with 3 files (mod.rs 751L, compiler.rs 992L, types.rs 118L) to stay well under the 2,000-line hard cap and 1,500-line soft target.
- Public API surface preserved via `mod.rs` re-exports; all existing tests still pass (468/468).

---

**Total Items:** 92+ tasks
**Completion:** ~87%

**Status:** Production-ready (v0.1.0 Stable)
**Release Date:** 2026-03-06 (stable: 2026-04-06)

## v0.2.0 / Future Work

- Federated RDF support across multiple endpoints.
- SHACL-SPARQL query translation.
- Bulk streaming import for large knowledge graphs.
- [x] ~~Split `src/sparql_gen.rs` (1,530 L) into a `sparql_gen/` directory.~~ (completed 2026-04-15)
