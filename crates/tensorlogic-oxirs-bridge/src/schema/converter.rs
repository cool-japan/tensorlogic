//! Convert RDF schema to TensorLogic symbol tables.
//!
//! This module provides the core conversion logic from RDF schemas (classes and properties)
//! to TensorLogic [`SymbolTable`] format, which contains domains and predicates for
//! tensor-based logical reasoning.
//!
//! # Conversion Rules
//!
//! - **RDF Classes → TensorLogic Domains**: Each `rdfs:Class` or `owl:Class` becomes a domain
//!   with a default cardinality of 100 elements.
//! - **RDF Properties → TensorLogic Predicates**: Each `rdf:Property` becomes a binary predicate
//!   with argument types determined by `rdfs:domain` and `rdfs:range`.
//! - **Standard Types**: Automatically adds `Literal`, `Resource`, and `Entity` base domains.
//!
//! # Example
//!
//! ```
//! use tensorlogic_oxirs_bridge::SchemaAnalyzer;
//! use anyhow::Result;
//!
//! fn main() -> Result<()> {
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
//!     let mut analyzer = SchemaAnalyzer::new();
//!     analyzer.load_turtle(turtle)?;
//!     analyzer.analyze()?;
//!
//!     let symbol_table = analyzer.to_symbol_table()?;
//!     println!("Domains: {:?}", symbol_table.domains.keys().collect::<Vec<_>>());
//!     println!("Predicates: {:?}", symbol_table.predicates.keys().collect::<Vec<_>>());
//!     Ok(())
//! }
//! ```

use anyhow::Result;
use tensorlogic_adapters::{DomainInfo, PredicateInfo, SymbolTable};

use super::SchemaAnalyzer;

impl SchemaAnalyzer {
    /// Converts the analyzed RDF schema to a TensorLogic symbol table.
    ///
    /// This method transforms RDF classes and properties into TensorLogic domains and predicates,
    /// creating a [`SymbolTable`] suitable for compilation into tensor operations.
    ///
    /// # Conversion Details
    ///
    /// ## Standard Domains
    ///
    /// Always includes these base types:
    /// - `Literal` (cardinality: 10000) - For literal values
    /// - `Resource` (cardinality: 10000) - For RDF resources
    /// - `Entity` (cardinality: 1000) - For general entities
    ///
    /// ## Class → Domain Mapping
    ///
    /// For each RDF class:
    /// - **Name**: Extracted from IRI using [`iri_to_name()`](Self::iri_to_name)
    /// - **Cardinality**: Default 100 elements
    /// - **Description**: Uses `rdfs:label` if available, otherwise `rdfs:comment`
    ///
    /// ## Property → Predicate Mapping
    ///
    /// For each RDF property:
    /// - **Name**: Extracted from IRI using [`iri_to_name()`](Self::iri_to_name)
    /// - **Arguments**: Binary predicate with domain types from `rdfs:domain` and `rdfs:range`
    /// - **Default Types**: Uses `Entity` if domain/range not specified
    /// - **Description**: Uses `rdfs:label` if available, otherwise `rdfs:comment`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Duplicate domain or predicate names are encountered
    /// - Invalid domain references in property definitions
    ///
    /// # Examples
    ///
    /// ## Basic Conversion
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
    ///         ex:Organization a rdfs:Class .
    ///
    ///         ex:worksFor a rdf:Property ;
    ///                     rdfs:domain ex:Person ;
    ///                     rdfs:range ex:Organization .
    ///     "#;
    ///
    ///     let mut analyzer = SchemaAnalyzer::new();
    ///     analyzer.load_turtle(turtle)?;
    ///     analyzer.analyze()?;
    ///
    ///     let table = analyzer.to_symbol_table()?;
    ///
    ///     // Check domains
    ///     assert!(table.domains.contains_key("Person"));
    ///     assert!(table.domains.contains_key("Organization"));
    ///     assert!(table.domains.contains_key("Entity")); // Standard type
    ///
    ///     // Check predicates
    ///     assert!(table.predicates.contains_key("worksFor"));
    ///     let works_for = &table.predicates["worksFor"];
    ///     assert_eq!(works_for.arg_domains, vec!["Person", "Organization"]);
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// ## With Descriptions
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
    ///                   rdfs:label "A human being" .
    ///
    ///         ex:name a rdf:Property ;
    ///                 rdfs:label "A person's name" ;
    ///                 rdfs:domain ex:Person .
    ///     "#;
    ///
    ///     let mut analyzer = SchemaAnalyzer::new();
    ///     analyzer.load_turtle(turtle)?;
    ///     analyzer.analyze()?;
    ///
    ///     let table = analyzer.to_symbol_table()?;
    ///
    ///     // Descriptions are preserved
    ///     assert_eq!(table.domains["Person"].description, Some("A human being".to_string()));
    ///     assert_eq!(table.predicates["name"].description, Some("A person's name".to_string()));
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Workflow
    ///
    /// 1. Load data: [`load_turtle()`](Self::load_turtle)
    /// 2. Analyze: [`analyze()`](Self::analyze)
    /// 3. **Convert**: `to_symbol_table()`
    ///
    /// # See Also
    ///
    /// - [`SymbolTable`] - The output format
    /// - [`DomainInfo`] - Domain type structure
    /// - [`PredicateInfo`] - Predicate structure
    ///
    /// [`SymbolTable`]: tensorlogic_adapters::SymbolTable
    /// [`DomainInfo`]: tensorlogic_adapters::DomainInfo
    /// [`PredicateInfo`]: tensorlogic_adapters::PredicateInfo
    pub fn to_symbol_table(&self) -> Result<SymbolTable> {
        let mut table = SymbolTable::new();

        // Add standard RDF/RDFS types as domains
        table.add_domain(DomainInfo::new("Literal", 10000))?;
        table.add_domain(DomainInfo::new("Resource", 10000))?;
        table.add_domain(DomainInfo::new("Entity", 1000))?;

        // Convert classes to domains
        for (class_iri, class_info) in &self.classes {
            let domain_name = Self::iri_to_name(class_iri);
            let mut domain = DomainInfo::new(&domain_name, 100); // Default cardinality

            if let Some(label) = &class_info.label {
                domain = domain.with_description(label);
            } else if let Some(comment) = &class_info.comment {
                domain = domain.with_description(comment);
            }

            table.add_domain(domain)?;
        }

        // Convert properties to predicates
        for (prop_iri, prop_info) in &self.properties {
            let pred_name = Self::iri_to_name(prop_iri);

            // Determine argument domains from property domain/range
            let mut arg_domains = Vec::new();

            if !prop_info.domain.is_empty() {
                arg_domains.push(Self::iri_to_name(&prop_info.domain[0]));
            } else {
                arg_domains.push("Entity".to_string()); // Default domain
            }

            if !prop_info.range.is_empty() {
                arg_domains.push(Self::iri_to_name(&prop_info.range[0]));
            } else {
                arg_domains.push("Entity".to_string()); // Default range
            }

            let mut predicate = PredicateInfo::new(&pred_name, arg_domains);

            if let Some(label) = &prop_info.label {
                predicate = predicate.with_description(label);
            } else if let Some(comment) = &prop_info.comment {
                predicate = predicate.with_description(comment);
            }

            table.add_predicate(predicate)?;
        }

        Ok(table)
    }

    /// Extracts the local name from a full IRI.
    ///
    /// This is a utility function that converts full IRIs into short, human-readable names
    /// for use in TensorLogic symbol tables. It extracts everything after the last `#` or `/`
    /// in the IRI.
    ///
    /// # Algorithm
    ///
    /// 1. If IRI contains `#`, returns everything after the last `#`
    /// 2. Else if IRI contains `/`, returns everything after the last `/`
    /// 3. Otherwise, returns the full IRI unchanged
    ///
    /// # Arguments
    ///
    /// * `iri` - A full IRI string (e.g., `http://example.org/Person`)
    ///
    /// # Returns
    ///
    /// The local name portion of the IRI (e.g., `Person`)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    ///
    /// // Hash-style IRI (RDF/RDFS/OWL standard)
    /// let name = SchemaAnalyzer::iri_to_name("http://www.w3.org/2000/01/rdf-schema#Class");
    /// assert_eq!(name, "Class");
    ///
    /// // Slash-style IRI (common in custom ontologies)
    /// let name = SchemaAnalyzer::iri_to_name("http://example.org/Person");
    /// assert_eq!(name, "Person");
    ///
    /// // FOAF namespace (hash-style)
    /// let name = SchemaAnalyzer::iri_to_name("http://xmlns.com/foaf/0.1/Person");
    /// assert_eq!(name, "Person");
    ///
    /// // No separator - returns full IRI
    /// let name = SchemaAnalyzer::iri_to_name("Person");
    /// assert_eq!(name, "Person");
    /// ```
    ///
    /// # Common Patterns
    ///
    /// | IRI | Extracted Name |
    /// |-----|----------------|
    /// | `http://www.w3.org/2000/01/rdf-schema#Class` | `Class` |
    /// | `http://xmlns.com/foaf/0.1/Person` | `Person` |
    /// | `http://example.org/vocab/Animal` | `Animal` |
    /// | `urn:example:Book` | `Book` |
    ///
    /// # Use Cases
    ///
    /// - Converting RDF class IRIs to TensorLogic domain names
    /// - Converting RDF property IRIs to TensorLogic predicate names
    /// - Generating readable identifiers for error messages
    ///
    /// # See Also
    ///
    /// - [`to_symbol_table()`](Self::to_symbol_table) - Uses this for all conversions
    pub fn iri_to_name(iri: &str) -> String {
        if let Some(hash_pos) = iri.rfind('#') {
            iri[hash_pos + 1..].to_string()
        } else if let Some(slash_pos) = iri.rfind('/') {
            iri[slash_pos + 1..].to_string()
        } else {
            iri.to_string()
        }
    }
}

/// Export a SymbolTable to Turtle format.
///
/// This function converts a TensorLogic SymbolTable back to RDF Turtle format,
/// useful for serialization, debugging, and interoperability with other RDF tools.
///
/// # Arguments
///
/// * `table` - The SymbolTable to export
/// * `base_iri` - Base IRI for generated resources (e.g., `"http://example.org/"`)
///
/// # Returns
///
/// A String containing valid Turtle RDF data.
///
/// # Example
///
/// ```
/// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo};
/// use tensorlogic_oxirs_bridge::schema::converter::symbol_table_to_turtle;
///
/// let mut table = SymbolTable::new();
/// table.add_domain(DomainInfo::new("Person", 100).with_description("A human being")).unwrap();
/// table.add_predicate(PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])).unwrap();
///
/// let turtle = symbol_table_to_turtle(&table, "http://example.org/");
/// assert!(turtle.contains("ex:Person"));
/// assert!(turtle.contains("ex:knows"));
/// ```
pub fn symbol_table_to_turtle(table: &SymbolTable, base_iri: &str) -> String {
    let mut output = String::new();

    // Write prefixes
    output.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
    output.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
    output.push_str("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n");
    output.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n");
    output.push_str(&format!("@prefix ex: <{}> .\n\n", base_iri));

    // Skip standard types
    let standard_types = ["Literal", "Resource", "Entity", "Value"];

    // Export domains as classes
    for (name, domain) in &table.domains {
        if standard_types.contains(&name.as_str()) {
            continue;
        }

        output.push_str(&format!("ex:{} a rdfs:Class", name));

        if let Some(desc) = &domain.description {
            output.push_str(&format!(
                " ;\n    rdfs:label \"{}\"",
                escape_turtle_string(desc)
            ));
        }

        // Add cardinality as custom annotation
        output.push_str(&format!(
            " ;\n    rdfs:comment \"Cardinality: {}\"",
            domain.cardinality
        ));

        output.push_str(" .\n\n");
    }

    // Export predicates as properties
    for (name, predicate) in &table.predicates {
        output.push_str(&format!("ex:{} a rdf:Property", name));

        // Add domain (first argument type)
        if !predicate.arg_domains.is_empty() {
            let domain_name = &predicate.arg_domains[0];
            if !standard_types.contains(&domain_name.as_str()) {
                output.push_str(&format!(" ;\n    rdfs:domain ex:{}", domain_name));
            }
        }

        // Add range (second argument type for binary predicates)
        if predicate.arg_domains.len() > 1 {
            let range_name = &predicate.arg_domains[1];
            if !standard_types.contains(&range_name.as_str()) {
                output.push_str(&format!(" ;\n    rdfs:range ex:{}", range_name));
            }
        }

        if let Some(desc) = &predicate.description {
            output.push_str(&format!(
                " ;\n    rdfs:label \"{}\"",
                escape_turtle_string(desc)
            ));
        }

        output.push_str(" .\n\n");
    }

    output
}

/// Export a SymbolTable to JSON format.
///
/// This provides a JSON serialization of the SymbolTable for use with
/// web services, configuration files, or JavaScript interoperability.
///
/// # Example
///
/// ```
/// use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo};
/// use tensorlogic_oxirs_bridge::schema::converter::symbol_table_to_json;
///
/// let mut table = SymbolTable::new();
/// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
///
/// let json = symbol_table_to_json(&table).unwrap();
/// assert!(json.contains("\"Person\""));
/// ```
pub fn symbol_table_to_json(table: &SymbolTable) -> Result<String> {
    serde_json::to_string_pretty(table)
        .map_err(|e| anyhow::anyhow!("JSON serialization error: {}", e))
}

/// Import a SymbolTable from JSON format.
///
/// # Example
///
/// ```
/// use tensorlogic_adapters::{SymbolTable, DomainInfo};
/// use tensorlogic_oxirs_bridge::schema::converter::{symbol_table_to_json, symbol_table_from_json};
///
/// let mut table = SymbolTable::new();
/// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
///
/// let json = symbol_table_to_json(&table).unwrap();
/// let imported = symbol_table_from_json(&json).unwrap();
///
/// assert!(imported.domains.contains_key("Person"));
/// ```
pub fn symbol_table_from_json(json: &str) -> Result<SymbolTable> {
    serde_json::from_str(json).map_err(|e| anyhow::anyhow!("JSON deserialization error: {}", e))
}

/// Escape a string for use in Turtle literals.
fn escape_turtle_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_adapters::{DomainInfo, PredicateInfo};

    #[test]
    fn test_symbol_table_to_turtle_basic() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A human being"))
            .unwrap();
        table
            .add_domain(DomainInfo::new("Organization", 50))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "worksFor",
                vec!["Person".to_string(), "Organization".to_string()],
            ))
            .unwrap();

        let turtle = symbol_table_to_turtle(&table, "http://example.org/");

        assert!(turtle.contains("@prefix rdf:"));
        assert!(turtle.contains("@prefix rdfs:"));
        assert!(turtle.contains("ex:Person a rdfs:Class"));
        assert!(turtle.contains("ex:Organization a rdfs:Class"));
        assert!(turtle.contains("ex:worksFor a rdf:Property"));
        assert!(turtle.contains("rdfs:domain ex:Person"));
        assert!(turtle.contains("rdfs:range ex:Organization"));
        assert!(turtle.contains("A human being"));
    }

    #[test]
    fn test_symbol_table_to_turtle_skips_standard_types() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Literal", 10000)).unwrap();
        table.add_domain(DomainInfo::new("Entity", 1000)).unwrap();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let turtle = symbol_table_to_turtle(&table, "http://example.org/");

        // Standard types should be skipped
        assert!(!turtle.contains("ex:Literal"));
        assert!(!turtle.contains("ex:Entity"));
        // Custom type should be present
        assert!(turtle.contains("ex:Person"));
    }

    #[test]
    fn test_symbol_table_json_roundtrip() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person"))
            .unwrap();
        table
            .add_predicate(
                PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
                    .with_description("Knows relationship"),
            )
            .unwrap();

        let json = symbol_table_to_json(&table).unwrap();
        let imported = symbol_table_from_json(&json).unwrap();

        assert_eq!(table.domains.len(), imported.domains.len());
        assert_eq!(table.predicates.len(), imported.predicates.len());
        assert!(imported.domains.contains_key("Person"));
        assert!(imported.predicates.contains_key("knows"));
    }

    #[test]
    fn test_escape_turtle_string() {
        assert_eq!(escape_turtle_string("simple"), "simple");
        assert_eq!(
            escape_turtle_string("with \"quotes\""),
            "with \\\"quotes\\\""
        );
        assert_eq!(escape_turtle_string("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_turtle_string("with\\backslash"), "with\\\\backslash");
    }
}
