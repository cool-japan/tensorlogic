//! OxiRS Bridge: RDF*/SHACL/GraphQL → TensorLogic Integration
//!
//! **Version**: 0.1.0-beta.1 | **Status**: Production Ready
//!
//! This crate provides comprehensive bidirectional integration between RDF knowledge graphs
//! and TensorLogic's tensor-based logical reasoning system. It enables semantic web data
//! to be compiled into tensor operations while preserving provenance and validation semantics.
//!
//! # Overview
//!
//! The bridge connects five key components:
//!
//! 1. **Schema Import**: RDF/OWL schemas → TensorLogic [`SymbolTable`](tensorlogic_adapters::SymbolTable)
//! 2. **Constraint Compilation**: SHACL shapes → TensorLogic [`TLExpr`](tensorlogic_ir::TLExpr) rules
//! 3. **Provenance Tracking**: RDF entities ↔ tensor indices with RDF* metadata
//! 4. **Validation**: SHACL-compliant validation reports from tensor outputs
//! 5. **GraphQL Integration**: GraphQL schemas → TensorLogic domains/predicates
//!
//! # Quick Start
//!
//! ```
//! use tensorlogic_oxirs_bridge::SchemaAnalyzer;
//! use anyhow::Result;
//!
//! fn main() -> Result<()> {
//!     // Define an RDF schema
//!     let turtle = r#"
//!         @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
//!         @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
//!         @prefix ex: <http://example.org/> .
//!
//!         ex:Person a rdfs:Class ;
//!                   rdfs:label "Person" .
//!
//!         ex:knows a rdf:Property ;
//!                  rdfs:domain ex:Person ;
//!                  rdfs:range ex:Person .
//!     "#;
//!
//!     // Parse and analyze the schema
//!     let mut analyzer = SchemaAnalyzer::new();
//!     analyzer.load_turtle(turtle)?;
//!     analyzer.analyze()?;
//!
//!     // Convert to TensorLogic symbol table
//!     let symbol_table = analyzer.to_symbol_table()?;
//!     println!("Converted {} classes and {} properties",
//!              symbol_table.domains.len(),
//!              symbol_table.predicates.len());
//!     Ok(())
//! }
//! ```
//!
//! # Key Features
//!
//! ## RDF/OWL Schema Support
//!
//! - **RDFS**: Classes, properties, subclass/subproperty hierarchies
//! - **OWL**: Class expressions, property characteristics, restrictions
//! - **RDFS Inference**: Automatic entailment and materialization
//! - **Formats**: Turtle, N-Triples, JSON-LD
//!
//! ### SHACL Validation
//! - **Constraint Types**: 15+ SHACL constraint types (minCount, pattern, datatype, etc.)
//! - **Logical Operators**: sh:and, sh:or, sh:not, sh:xone
//! - **Validation Reports**: Full SHACL-compliant reports with severity levels
//! - **Export**: Turtle and JSON export formats
//!
//! ### SPARQL 1.1 Query Support
//! - **Query Types**: SELECT, ASK, DESCRIBE, CONSTRUCT
//! - **Graph Patterns**: OPTIONAL (left-outer join), UNION (disjunction)
//! - **Filters**: Comparison operators, BOUND, isIRI, isLiteral, regex
//! - **Solution Modifiers**: DISTINCT, LIMIT, OFFSET, ORDER BY
//! - **Compilation**: Full compilation to TensorLogic expressions
//!
//! ## Performance Features
//!
//! - **Triple Indexing**: O(1) lookups by subject/predicate/object
//! - **Caching**: In-memory and file-based caching (10-50x speedup)
//! - **Metadata**: Multilingual label preservation and quality checking
//!
//! ## Provenance & Tracking
//!
//! - **RDF* Support**: Statement-level metadata with quoted triples
//! - **Bidirectional Mapping**: Entities ↔ tensor indices
//! - **Confidence Tracking**: Confidence scores for inferred statements
//! - **Rule Attribution**: Track which rules produced which results
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     Input Formats                            │
//! │  Turtle │ N-Triples │ JSON-LD │ GraphQL │ SHACL              │
//! └────────────┬────────────────────────────────────────────────┘
//!              │
//!              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  SchemaAnalyzer                              │
//! │  • Parse RDF triples                                         │
//! │  • Extract classes & properties                              │
//! │  • Build indexes (optional)                                  │
//! │  • Preserve metadata (optional)                              │
//! └────────────┬────────────────────────────────────────────────┘
//!              │
//!              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  SymbolTable                                 │
//! │  • Domains (from classes)                                    │
//! │  • Predicates (from properties)                              │
//! │  • Axis metadata                                             │
//! └────────────┬────────────────────────────────────────────────┘
//!              │
//!              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │              TensorLogic Compiler                            │
//! │  • TLExpr (logical expressions)                              │
//! │  • Tensor operations                                         │
//! │  • Provenance tracking                                       │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Examples
//!
//! ## With Performance Features
//!
//! ```
//! use tensorlogic_oxirs_bridge::SchemaAnalyzer;
//! use anyhow::Result;
//!
//! fn main() -> Result<()> {
//!     // Enable indexing and metadata preservation
//!     let mut analyzer = SchemaAnalyzer::new()
//!         .with_indexing()
//!         .with_metadata();
//!
//!     analyzer.load_turtle(r#"
//!         @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
//!         @prefix ex: <http://example.org/> .
//!
//!         ex:Person rdfs:label "Person"@en, "Personne"@fr .
//!     "#)?;
//!
//!     // Fast indexed lookup
//!     if let Some(index) = analyzer.index() {
//!         let triples = index.find_by_subject("http://example.org/Person");
//!         println!("Found {} triples", triples.len());
//!     }
//!
//!     // Multilingual metadata
//!     if let Some(metadata) = analyzer.metadata() {
//!         if let Some(meta) = metadata.get("http://example.org/Person") {
//!             println!("EN: {}", meta.get_label(Some("en")).unwrap_or("N/A"));
//!             println!("FR: {}", meta.get_label(Some("fr")).unwrap_or("N/A"));
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## SHACL Validation
//!
//! ```
//! use tensorlogic_oxirs_bridge::{ShaclConverter, SchemaAnalyzer};
//! use tensorlogic_adapters::SymbolTable;
//! use anyhow::Result;
//!
//! fn main() -> Result<()> {
//!     let shacl = r#"
//!         @prefix sh: <http://www.w3.org/ns/shacl#> .
//!         @prefix ex: <http://example.org/> .
//!         @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
//!
//!         ex:PersonShape a sh:NodeShape ;
//!             sh:targetClass ex:Person ;
//!             sh:property [
//!                 sh:path ex:name ;
//!                 sh:minCount 1 ;
//!                 sh:datatype xsd:string ;
//!             ] .
//!     "#;
//!
//!     let symbol_table = SymbolTable::new();
//!     let mut converter = ShaclConverter::new(symbol_table);
//!     converter.parse_shapes(shacl)?;
//!
//!     println!("Parsed SHACL shapes successfully");
//!     Ok(())
//! }
//! ```
//!
//! ## Caching for Performance
//!
//! ```
//! use tensorlogic_oxirs_bridge::{SchemaCache, SchemaAnalyzer};
//! use anyhow::Result;
//!
//! fn main() -> Result<()> {
//!     let mut cache = SchemaCache::new();
//!     let turtle = "@prefix ex: <http://example.org/> .";
//!
//!     // First access - cache miss
//!     let table = if let Some(cached) = cache.get_symbol_table(turtle) {
//!         cached
//!     } else {
//!         let mut analyzer = SchemaAnalyzer::new();
//!         analyzer.load_turtle(turtle)?;
//!         analyzer.analyze()?;
//!         let table = analyzer.to_symbol_table()?;
//!         cache.put_symbol_table(turtle, table.clone());
//!         table
//!     };
//!
//!     // Second access - cache hit (20-50x faster!)
//!     assert!(cache.get_symbol_table(turtle).is_some());
//!     Ok(())
//! }
//! ```
//!
//! # Module Organization
//!
//! - [`schema`] - Core RDF schema parsing and analysis
//!   - [`cache`](schema::cache) - Caching for performance
//!   - [`index`](schema::index) - Triple indexing for fast lookups
//!   - [`metadata`](schema::metadata) - Multilingual metadata management
//!   - [`owl`](schema::owl) - OWL ontology support
//!   - [`inference`](schema::inference) - RDFS inference engine
//! - [`shacl`] - SHACL constraint compilation and validation
//! - [`rdfstar`] - RDF* provenance tracking
//! - [`graphql`] - GraphQL schema integration
//! - [`sparql`] - SPARQL 1.1 query compilation (SELECT/ASK/DESCRIBE/CONSTRUCT + OPTIONAL/UNION)
//!
//! # See Also
//!
//! - [Examples directory](https://github.com/cool-japan/tensorlogic/tree/main/crates/tensorlogic-oxirs-bridge/examples) - Comprehensive examples
//! - [TensorLogic project](https://github.com/cool-japan/tensorlogic) - Main project repository
//! - [OxiRS](https://github.com/cool-japan/oxirs) - Full RDF/SPARQL support
//!
//! # Implementation Note
//!
//! This is a **lightweight implementation** using only `oxrdf` and `oxttl` for RDF parsing.
//! For full SPARQL query execution, federation, and advanced RDF features, use the `oxirs-core` crate.

mod compilation;
mod error;
pub mod graphql;
pub mod knowledge_embeddings;
pub mod oxirs_executor;
pub mod oxirs_graphql;
mod provenance;
pub mod rdfstar;
pub mod schema;
pub mod shacl;
pub mod sparql;

#[cfg(test)]
mod rdfstar_tests;
#[cfg(test)]
mod tests;

pub use compilation::compile_rules;
pub use error::{BridgeError, ParseLocation};
pub use graphql::{DirectiveValue, GraphQLConverter, GraphQLDirective};
pub use knowledge_embeddings::{
    cosine_similarity, euclidean_distance, EmbeddingConfig, EmbeddingModel, KGTriple,
    KnowledgeEmbeddings,
};
pub use oxirs_executor::{OxirsSparqlExecutor, QueryResults, QueryValue, TripleResult};
pub use oxirs_graphql::{GraphQLField, GraphQLObjectType, GraphQLType, OxirsGraphQLBridge};
pub use provenance::ProvenanceTracker;
pub use rdfstar::{
    MetadataBuilder, ProvenanceStats, QuotedTriple, RdfStarProvenanceStore, StatementMetadata,
};
pub use schema::{
    cache::{CacheStats, PersistentCache, SchemaCache},
    index::{IndexStats, TripleIndex},
    inference::{InferenceStats, RdfsInferenceEngine},
    jsonld::JsonLdContext,
    metadata::{EntityMetadata, LangString, MetadataStats, MetadataStore},
    nquads::{NQuadsProcessor, Quad},
    owl::{OwlClassInfo, OwlPropertyCharacteristics, OwlPropertyInfo, OwlRestriction},
    streaming::{StreamAnalyzer, StreamStats, StreamingRdfLoader},
    ClassInfo, PropertyInfo, SchemaAnalyzer,
};
pub use shacl::{
    validation::{ShaclValidator, ValidationReport, ValidationResult, ValidationSeverity},
    ShaclConverter,
};
pub use sparql::{
    AggregateFunction, FilterCondition, PatternElement, SelectElement, SparqlCompiler, SparqlQuery,
    TriplePattern,
};
