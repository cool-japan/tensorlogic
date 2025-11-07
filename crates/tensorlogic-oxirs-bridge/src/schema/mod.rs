//! RDF schema analysis and conversion to TensorLogic types.

mod analyzer;
pub mod cache;
mod converter;
pub mod index;
pub mod inference;
pub mod jsonld;
pub mod metadata;
mod ntriples;
pub mod owl;

#[cfg(test)]
mod inference_tests;
#[cfg(test)]
mod owl_tests;

use anyhow::{Context, Result};
use indexmap::IndexMap;
use oxrdf::Graph;
use serde::{Deserialize, Serialize};

/// Information about an RDF class extracted from a schema.
///
/// Represents an `rdfs:Class` or `owl:Class` with its associated metadata
/// and relationships. This structure is used to build TensorLogic domain types.
///
/// # Examples
///
/// ```
/// use tensorlogic_oxirs_bridge::ClassInfo;
///
/// let person_class = ClassInfo {
///     iri: "http://xmlns.com/foaf/0.1/Person".to_string(),
///     label: Some("Person".to_string()),
///     comment: Some("A person, living or dead".to_string()),
///     subclass_of: vec!["http://xmlns.com/foaf/0.1/Agent".to_string()],
/// };
/// ```
///
/// # Fields
///
/// - `iri`: The full IRI of the class (e.g., `http://xmlns.com/foaf/0.1/Person`)
/// - `label`: Human-readable label from `rdfs:label`
/// - `comment`: Description from `rdfs:comment`
/// - `subclass_of`: IRIs of parent classes from `rdfs:subClassOf`
///
/// # See Also
///
/// - [`SchemaAnalyzer::extract_classes`] - Extracts `ClassInfo` from RDF
/// - [`SchemaAnalyzer::to_symbol_table`] - Converts to TensorLogic domains
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassInfo {
    pub iri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub subclass_of: Vec<String>,
}

/// Information about an RDF property extracted from a schema.
///
/// Represents an `rdf:Property` or `owl:ObjectProperty`/`owl:DatatypeProperty`
/// with its metadata and domain/range constraints. This structure is used to
/// build TensorLogic predicates.
///
/// # Examples
///
/// ```
/// use tensorlogic_oxirs_bridge::PropertyInfo;
///
/// let knows_property = PropertyInfo {
///     iri: "http://xmlns.com/foaf/0.1/knows".to_string(),
///     label: Some("knows".to_string()),
///     comment: Some("A person known by this person".to_string()),
///     domain: vec!["http://xmlns.com/foaf/0.1/Person".to_string()],
///     range: vec!["http://xmlns.com/foaf/0.1/Person".to_string()],
/// };
/// ```
///
/// # Fields
///
/// - `iri`: The full IRI of the property (e.g., `http://xmlns.com/foaf/0.1/knows`)
/// - `label`: Human-readable label from `rdfs:label`
/// - `comment`: Description from `rdfs:comment`
/// - `domain`: IRIs of domain classes from `rdfs:domain`
/// - `range`: IRIs of range classes from `rdfs:range`
///
/// # See Also
///
/// - [`SchemaAnalyzer::extract_properties`] - Extracts `PropertyInfo` from RDF
/// - [`SchemaAnalyzer::to_symbol_table`] - Converts to TensorLogic predicates
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PropertyInfo {
    pub iri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub domain: Vec<String>,
    pub range: Vec<String>,
}

/// RDF schema analyzer and converter (lightweight oxrdf-based).
///
/// The primary interface for importing RDF schemas and converting them to TensorLogic
/// symbol tables. Supports multiple RDF serialization formats and optional performance
/// features like indexing and metadata preservation.
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
/// use anyhow::Result;
///
/// fn main() -> Result<()> {
///     let turtle = r#"
///         @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
///         @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
///         @prefix ex: <http://example.org/> .
///
///         ex:Person a rdfs:Class ;
///                   rdfs:label "Person" .
///
///         ex:name a rdf:Property ;
///                 rdfs:domain ex:Person .
///     "#;
///
///     let mut analyzer = SchemaAnalyzer::new();
///     analyzer.load_turtle(turtle)?;
///     analyzer.analyze()?;
///
///     let symbol_table = analyzer.to_symbol_table()?;
///     println!("Domains: {}", symbol_table.domains.len());
///     Ok(())
/// }
/// ```
///
/// ## With Indexing and Metadata
///
/// ```
/// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
/// use anyhow::Result;
///
/// fn main() -> Result<()> {
///     let turtle = r#"
///         @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
///         @prefix ex: <http://example.org/> .
///
///         ex:Person rdfs:label "Person"@en, "Personne"@fr .
///     "#;
///
///     let mut analyzer = SchemaAnalyzer::new()
///         .with_indexing()
///         .with_metadata();
///
///     analyzer.load_turtle(turtle)?;
///
///     // Fast lookups via index
///     if let Some(index) = analyzer.index() {
///         let triples = index.find_by_subject("http://example.org/Person");
///         println!("Found {} triples", triples.len());
///     }
///
///     // Access multilingual metadata
///     if let Some(metadata) = analyzer.metadata() {
///         let stats = metadata.stats();
///         println!("Entities with labels: {}", stats.entities_with_labels);
///     }
///
///     Ok(())
/// }
/// ```
///
/// # Supported Formats
///
/// - **Turtle**: [`load_turtle`](Self::load_turtle)
/// - **N-Triples**: [`load_ntriples`](Self::load_ntriples)
/// - **JSON-LD**: [`load_jsonld`](Self::load_jsonld)
///
/// # Performance Features
///
/// - **Indexing**: Enable with [`with_indexing`](Self::with_indexing) for O(1) triple lookups
/// - **Caching**: Use [`SchemaCache`](crate::SchemaCache) for repeated parsing
/// - **Metadata**: Enable with [`with_metadata`](Self::with_metadata) for multilingual labels
///
/// # See Also
///
/// - [`SymbolTable`](tensorlogic_adapters::SymbolTable) - Output format for TensorLogic
/// - [`TripleIndex`](crate::TripleIndex) - Fast RDF triple indexing
/// - [`MetadataStore`](crate::MetadataStore) - Multilingual metadata management
#[derive(Clone, Debug)]
pub struct SchemaAnalyzer {
    pub graph: Graph,
    pub classes: IndexMap<String, ClassInfo>,
    pub properties: IndexMap<String, PropertyInfo>,
    index: Option<index::TripleIndex>,
    metadata_store: Option<metadata::MetadataStore>,
}

impl SchemaAnalyzer {
    /// Creates a new empty schema analyzer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    ///
    /// let analyzer = SchemaAnalyzer::new();
    /// ```
    pub fn new() -> Self {
        SchemaAnalyzer {
            graph: Graph::new(),
            classes: IndexMap::new(),
            properties: IndexMap::new(),
            index: None,
            metadata_store: None,
        }
    }

    /// Creates a schema analyzer from an existing RDF graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - An `oxrdf::Graph` containing RDF triples
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    /// use oxrdf::Graph;
    ///
    /// let graph = Graph::new();
    /// let analyzer = SchemaAnalyzer::from_graph(graph);
    /// ```
    pub fn from_graph(graph: Graph) -> Self {
        SchemaAnalyzer {
            graph,
            classes: IndexMap::new(),
            properties: IndexMap::new(),
            index: None,
            metadata_store: None,
        }
    }

    /// Enables indexing for fast triple lookups.
    ///
    /// Creates a triple index that provides O(1) lookups by subject, predicate, or object.
    /// The index is automatically rebuilt when new data is loaded.
    ///
    /// # Performance
    ///
    /// - **Lookup Time**: O(1) average case
    /// - **Space Overhead**: ~3x memory (S, P, O indexes)
    /// - **Rebuild Time**: O(n) where n = number of triples
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    /// use anyhow::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let mut analyzer = SchemaAnalyzer::new().with_indexing();
    ///
    ///     analyzer.load_turtle(r#"
    ///         @prefix ex: <http://example.org/> .
    ///         ex:Alice ex:knows ex:Bob .
    ///     "#)?;
    ///
    ///     if let Some(index) = analyzer.index() {
    ///         let triples = index.find_by_subject("http://example.org/Alice");
    ///         println!("Found {} triples about Alice", triples.len());
    ///     }
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`TripleIndex`](crate::TripleIndex) - The indexing implementation
    /// - [`index()`](Self::index) - Accessor for the index
    pub fn with_indexing(mut self) -> Self {
        self.index = Some(index::TripleIndex::from_graph(&self.graph));
        self
    }

    /// Enables metadata preservation with multilingual support.
    ///
    /// Creates a metadata store that extracts and manages entity labels, comments,
    /// and custom annotations in multiple languages.
    ///
    /// # Features
    ///
    /// - Multilingual labels via language tags (e.g., `@en`, `@fr`)
    /// - Custom annotation properties
    /// - Metadata quality checks
    /// - Search by label functionality
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    /// use anyhow::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let mut analyzer = SchemaAnalyzer::new().with_metadata();
    ///
    ///     analyzer.load_turtle(r#"
    ///         @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ///         @prefix ex: <http://example.org/> .
    ///
    ///         ex:Person rdfs:label "Person"@en, "Personne"@fr .
    ///     "#)?;
    ///
    ///     if let Some(metadata) = analyzer.metadata() {
    ///         if let Some(meta) = metadata.get("http://example.org/Person") {
    ///             println!("English: {}", meta.get_label(Some("en")).unwrap_or("N/A"));
    ///             println!("French: {}", meta.get_label(Some("fr")).unwrap_or("N/A"));
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`MetadataStore`](crate::MetadataStore) - The metadata storage implementation
    /// - [`EntityMetadata`](crate::EntityMetadata) - Per-entity metadata structure
    pub fn with_metadata(mut self) -> Self {
        self.metadata_store = Some(metadata::MetadataStore::new());
        self
    }

    /// Rebuilds the triple index from the current graph.
    ///
    /// This is automatically called when loading new data, but can be manually
    /// invoked if the graph is modified directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    ///
    /// let mut analyzer = SchemaAnalyzer::new().with_indexing();
    /// // ... modify analyzer.graph directly ...
    /// analyzer.rebuild_index();
    /// ```
    pub fn rebuild_index(&mut self) {
        if let Some(ref mut idx) = self.index {
            idx.build_from_graph(&self.graph);
        }
    }

    /// Returns a reference to the triple index, if enabled.
    ///
    /// # Returns
    ///
    /// - `Some(&TripleIndex)` if indexing was enabled with [`with_indexing()`](Self::with_indexing)
    /// - `None` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    ///
    /// let analyzer = SchemaAnalyzer::new().with_indexing();
    /// assert!(analyzer.index().is_some());
    /// ```
    pub fn index(&self) -> Option<&index::TripleIndex> {
        self.index.as_ref()
    }

    /// Returns a mutable reference to the triple index, if enabled.
    ///
    /// # Returns
    ///
    /// - `Some(&mut TripleIndex)` if indexing was enabled
    /// - `None` otherwise
    pub fn index_mut(&mut self) -> Option<&mut index::TripleIndex> {
        self.index.as_mut()
    }

    /// Returns a reference to the metadata store, if enabled.
    ///
    /// # Returns
    ///
    /// - `Some(&MetadataStore)` if metadata was enabled with [`with_metadata()`](Self::with_metadata)
    /// - `None` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    ///
    /// let analyzer = SchemaAnalyzer::new().with_metadata();
    /// assert!(analyzer.metadata().is_some());
    /// ```
    pub fn metadata(&self) -> Option<&metadata::MetadataStore> {
        self.metadata_store.as_ref()
    }

    /// Returns a mutable reference to the metadata store, if enabled.
    ///
    /// # Returns
    ///
    /// - `Some(&mut MetadataStore)` if metadata was enabled
    /// - `None` otherwise
    pub fn metadata_mut(&mut self) -> Option<&mut metadata::MetadataStore> {
        self.metadata_store.as_mut()
    }

    /// Loads RDF data from Turtle format.
    ///
    /// Parses the Turtle syntax and adds all triples to the internal graph.
    /// Automatically rebuilds the index and extracts metadata if those features are enabled.
    ///
    /// # Arguments
    ///
    /// * `data` - A string containing Turtle-formatted RDF data
    ///
    /// # Errors
    ///
    /// Returns an error if the Turtle syntax is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    /// use anyhow::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let turtle = r#"
    ///         @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    ///         @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ///         @prefix ex: <http://example.org/> .
    ///
    ///         ex:Person a rdfs:Class ;
    ///                   rdfs:label "Person" ;
    ///                   rdfs:comment "A human being" .
    ///
    ///         ex:name a rdf:Property ;
    ///                 rdfs:domain ex:Person ;
    ///                 rdfs:range rdfs:Literal .
    ///     "#;
    ///
    ///     let mut analyzer = SchemaAnalyzer::new();
    ///     analyzer.load_turtle(turtle)?;
    ///     println!("Loaded {} triples", analyzer.graph.len());
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`load_ntriples()`](Self::load_ntriples) - Load from N-Triples format
    /// - [`load_jsonld()`](Self::load_jsonld) - Load from JSON-LD format
    pub fn load_turtle(&mut self, data: &str) -> Result<()> {
        use oxttl::TurtleParser;

        let parser = TurtleParser::new().for_slice(data.as_bytes());

        for result in parser {
            let triple = result.context("Failed to parse Turtle")?;
            self.graph.insert(&triple);
        }

        // Rebuild index if enabled
        self.rebuild_index();

        // Extract metadata if enabled
        if let Some(ref mut store) = self.metadata_store {
            store.extract_from_graph(&self.graph)?;
        }

        Ok(())
    }

    /// Analyzes the loaded RDF schema and extracts classes and properties.
    ///
    /// This method must be called after loading data and before converting to a symbol table.
    /// It extracts:
    /// - RDF/OWL classes into [`ClassInfo`] structures
    /// - RDF/OWL properties into [`PropertyInfo`] structures
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails (rare).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    /// use anyhow::Result;
    ///
    /// fn main() -> Result<()> {
    ///     let turtle = r#"
    ///         @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    ///         @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ///         @prefix ex: <http://example.org/> .
    ///
    ///         ex:Person a rdfs:Class .
    ///         ex:name a rdf:Property .
    ///     "#;
    ///
    ///     let mut analyzer = SchemaAnalyzer::new();
    ///     analyzer.load_turtle(turtle)?;
    ///     analyzer.analyze()?;
    ///
    ///     println!("Found {} classes", analyzer.classes.len());
    ///     println!("Found {} properties", analyzer.properties.len());
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Workflow
    ///
    /// 1. Load data: [`load_turtle()`](Self::load_turtle)
    /// 2. **Analyze**: `analyze()`
    /// 3. Convert: [`to_symbol_table()`](Self::to_symbol_table)
    pub fn analyze(&mut self) -> Result<()> {
        self.extract_classes()?;
        self.extract_properties()?;
        Ok(())
    }
}

impl Default for SchemaAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
