//! N-Triples serialization support for RDF data export
//!
//! N-Triples is a line-based, plain text format for encoding an RDF graph.
//! Each line encodes one RDF triple. It's simple, easy to parse, and suitable
//! for large datasets.
//!
//! ## Format Specification
//!
//! Each line contains:
//! - Subject (IRI or blank node)
//! - Predicate (IRI)
//! - Object (IRI, blank node, or literal)
//! - Terminating '.' and newline
//!
//! ## Example
//!
//! ```text
//! <http://example.org/Person> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2000/01/rdf-schema#Class> .
//! <http://example.org/Person> <http://www.w3.org/2000/01/rdf-schema#label> "Person" .
//! ```

use super::{ClassInfo, PropertyInfo, SchemaAnalyzer};
use anyhow::Result;

impl SchemaAnalyzer {
    /// Export the schema as N-Triples format
    ///
    /// This generates a simple, line-based RDF serialization suitable for:
    /// - Large-scale data processing
    /// - Streaming applications
    /// - Simple parsing requirements
    ///
    /// ## Example
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    ///
    /// let mut analyzer = SchemaAnalyzer::new();
    /// analyzer.load_turtle(r#"
    ///     @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ///     @prefix ex: <http://example.org/> .
    ///
    ///     ex:Person a rdfs:Class ;
    ///         rdfs:label "Person" .
    /// "#).unwrap();
    /// analyzer.analyze().unwrap();
    ///
    /// let ntriples = analyzer.to_ntriples();
    /// assert!(ntriples.contains("<http://example.org/Person>"));
    /// ```
    pub fn to_ntriples(&self) -> String {
        let mut output = String::new();

        // Export classes
        for (iri, class_info) in &self.classes {
            output.push_str(&self.class_to_ntriples(iri, class_info));
        }

        // Export properties
        for (iri, prop_info) in &self.properties {
            output.push_str(&self.property_to_ntriples(iri, prop_info));
        }

        output
    }

    /// Convert a class to N-Triples format
    fn class_to_ntriples(&self, iri: &str, class_info: &ClassInfo) -> String {
        let mut output = String::new();

        // Class type declaration
        output.push_str(&format!(
            "<{}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2000/01/rdf-schema#Class> .\n",
            iri
        ));

        // Label
        if let Some(label) = &class_info.label {
            output.push_str(&format!(
                "<{}> <http://www.w3.org/2000/01/rdf-schema#label> {} .\n",
                iri,
                escape_literal(label)
            ));
        }

        // Comment
        if let Some(comment) = &class_info.comment {
            output.push_str(&format!(
                "<{}> <http://www.w3.org/2000/01/rdf-schema#comment> {} .\n",
                iri,
                escape_literal(comment)
            ));
        }

        // Subclass relationships
        for parent in &class_info.subclass_of {
            output.push_str(&format!(
                "<{}> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <{}> .\n",
                iri, parent
            ));
        }

        output
    }

    /// Convert a property to N-Triples format
    fn property_to_ntriples(&self, iri: &str, prop_info: &PropertyInfo) -> String {
        let mut output = String::new();

        // Property type declaration
        output.push_str(&format!(
            "<{}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> .\n",
            iri
        ));

        // Label
        if let Some(label) = &prop_info.label {
            output.push_str(&format!(
                "<{}> <http://www.w3.org/2000/01/rdf-schema#label> {} .\n",
                iri,
                escape_literal(label)
            ));
        }

        // Comment
        if let Some(comment) = &prop_info.comment {
            output.push_str(&format!(
                "<{}> <http://www.w3.org/2000/01/rdf-schema#comment> {} .\n",
                iri,
                escape_literal(comment)
            ));
        }

        // Domain
        for domain in &prop_info.domain {
            output.push_str(&format!(
                "<{}> <http://www.w3.org/2000/01/rdf-schema#domain> <{}> .\n",
                iri, domain
            ));
        }

        // Range
        for range in &prop_info.range {
            output.push_str(&format!(
                "<{}> <http://www.w3.org/2000/01/rdf-schema#range> <{}> .\n",
                iri, range
            ));
        }

        output
    }

    /// Load N-Triples data into the schema analyzer
    ///
    /// This parses N-Triples format and adds triples to the graph.
    /// Note: This is a basic implementation. For production use, consider
    /// using a dedicated N-Triples parser.
    ///
    /// ## Example
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    ///
    /// let ntriples = r#"
    /// <http://example.org/Person> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2000/01/rdf-schema#Class> .
    /// <http://example.org/Person> <http://www.w3.org/2000/01/rdf-schema#label> "Person" .
    /// "#;
    ///
    /// let mut analyzer = SchemaAnalyzer::new();
    /// analyzer.load_ntriples(ntriples).unwrap();
    /// analyzer.analyze().unwrap();
    ///
    /// assert_eq!(analyzer.classes.len(), 1);
    /// ```
    pub fn load_ntriples(&mut self, data: &str) -> Result<()> {
        for line in data.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse N-Triple line: <subject> <predicate> <object> .
            let triple = parse_ntriple_line(line)?;

            self.graph.insert(&triple);
        }

        Ok(())
    }
}

/// Escape a string literal for N-Triples format
fn escape_literal(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{}\"", escaped)
}

/// Parse a single N-Triple line into a Triple
///
/// Basic implementation - for production, use oxttl::NTriplesParser
fn parse_ntriple_line(line: &str) -> Result<oxrdf::Triple> {
    use oxrdf::{Literal, NamedNode, NamedOrBlankNode, Term, Triple};

    // Remove trailing ' .' if present
    let line = line.trim_end_matches('.').trim();

    // Find components
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() < 3 {
        anyhow::bail!("Invalid N-Triple line: {}", line);
    }

    // Parse subject
    let subject_str = parts[0];
    let subject = if let Some(iri) = subject_str
        .strip_prefix('<')
        .and_then(|s| s.strip_suffix('>'))
    {
        NamedOrBlankNode::NamedNode(NamedNode::new(iri)?)
    } else if let Some(blank_id) = subject_str.strip_prefix("_:") {
        NamedOrBlankNode::BlankNode(oxrdf::BlankNode::new(blank_id)?)
    } else {
        anyhow::bail!("Invalid subject: {}", subject_str);
    };

    // Parse predicate
    let predicate_str = parts[1];
    let predicate = if let Some(iri) = predicate_str
        .strip_prefix('<')
        .and_then(|s| s.strip_suffix('>'))
    {
        NamedNode::new(iri)?
    } else {
        anyhow::bail!("Invalid predicate: {}", predicate_str);
    };

    // Parse object (rest of the line)
    let object_str = parts[2..].join(" ");
    let object = if let Some(iri) = object_str
        .strip_prefix('<')
        .and_then(|s| s.strip_suffix('>'))
    {
        Term::NamedNode(NamedNode::new(iri)?)
    } else if let Some(blank_id) = object_str.strip_prefix("_:") {
        Term::BlankNode(oxrdf::BlankNode::new(blank_id)?)
    } else if let Some(rest) = object_str.strip_prefix('"') {
        // Parse literal (simplified - doesn't handle language tags or datatypes)
        let end_quote = rest.find('"').unwrap_or(rest.len());
        let literal_value = &rest[..end_quote];
        let unescaped = unescape_literal(literal_value);
        Term::Literal(Literal::new_simple_literal(&unescaped))
    } else {
        anyhow::bail!("Invalid object: {}", object_str);
    };

    Ok(Triple::new(subject, predicate, object))
}

/// Unescape a literal value from N-Triples format
fn unescape_literal(s: &str) -> String {
    s.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace("\\\"", "\"")
        .replace("\\\\", "\\")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_ntriples_basic() {
        let mut analyzer = SchemaAnalyzer::new();
        analyzer
            .load_turtle(
                r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Person a rdfs:Class ;
                rdfs:label "Person" ;
                rdfs:comment "A human being" .
        "#,
            )
            .unwrap();
        analyzer.analyze().unwrap();

        let ntriples = analyzer.to_ntriples();

        assert!(ntriples.contains("<http://example.org/Person>"));
        assert!(ntriples.contains("<http://www.w3.org/2000/01/rdf-schema#Class>"));
        assert!(ntriples.contains("\"Person\""));
        assert!(ntriples.contains("\"A human being\""));
    }

    #[test]
    fn test_to_ntriples_with_hierarchy() {
        let mut analyzer = SchemaAnalyzer::new();
        analyzer
            .load_turtle(
                r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Animal a rdfs:Class .
            ex:Dog a rdfs:Class ;
                rdfs:subClassOf ex:Animal .
        "#,
            )
            .unwrap();
        analyzer.analyze().unwrap();

        let ntriples = analyzer.to_ntriples();

        assert!(ntriples.contains("<http://example.org/Dog>"));
        assert!(ntriples.contains("<http://www.w3.org/2000/01/rdf-schema#subClassOf>"));
        assert!(ntriples.contains("<http://example.org/Animal>"));
    }

    #[test]
    fn test_to_ntriples_properties() {
        let mut analyzer = SchemaAnalyzer::new();
        analyzer
            .load_turtle(
                r#"
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:knows a rdf:Property ;
                rdfs:label "knows" ;
                rdfs:domain ex:Person ;
                rdfs:range ex:Person .
        "#,
            )
            .unwrap();
        analyzer.analyze().unwrap();

        let ntriples = analyzer.to_ntriples();

        assert!(ntriples.contains("<http://example.org/knows>"));
        assert!(ntriples.contains("<http://www.w3.org/1999/02/22-rdf-syntax-ns#Property>"));
        assert!(ntriples.contains("<http://www.w3.org/2000/01/rdf-schema#domain>"));
        assert!(ntriples.contains("<http://www.w3.org/2000/01/rdf-schema#range>"));
    }

    #[test]
    fn test_load_ntriples_basic() {
        let ntriples = r#"
<http://example.org/Person> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2000/01/rdf-schema#Class> .
<http://example.org/Person> <http://www.w3.org/2000/01/rdf-schema#label> "Person" .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_ntriples(ntriples).unwrap();
        analyzer.analyze().unwrap();

        assert_eq!(analyzer.classes.len(), 1);
        assert!(analyzer.classes.contains_key("http://example.org/Person"));

        let person = &analyzer.classes["http://example.org/Person"];
        assert_eq!(person.label, Some("Person".to_string()));
    }

    #[test]
    fn test_roundtrip_ntriples() {
        let turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Person a rdfs:Class ;
                rdfs:label "Person" .
        "#;

        let mut analyzer1 = SchemaAnalyzer::new();
        analyzer1.load_turtle(turtle).unwrap();
        analyzer1.analyze().unwrap();

        let ntriples = analyzer1.to_ntriples();

        let mut analyzer2 = SchemaAnalyzer::new();
        analyzer2.load_ntriples(&ntriples).unwrap();
        analyzer2.analyze().unwrap();

        assert_eq!(analyzer1.classes.len(), analyzer2.classes.len());
        assert_eq!(analyzer1.properties.len(), analyzer2.properties.len());
    }

    #[test]
    fn test_escape_literal() {
        assert_eq!(escape_literal("simple"), "\"simple\"");
        assert_eq!(escape_literal("with \"quotes\""), "\"with \\\"quotes\\\"\"");
        assert_eq!(escape_literal("with\nnewline"), "\"with\\nnewline\"");
        assert_eq!(escape_literal("with\\backslash"), "\"with\\\\backslash\"");
    }

    #[test]
    fn test_unescape_literal() {
        assert_eq!(unescape_literal("simple"), "simple");
        assert_eq!(unescape_literal("with \\\"quotes\\\""), "with \"quotes\"");
        assert_eq!(unescape_literal("with\\nnewline"), "with\nnewline");
        assert_eq!(unescape_literal("with\\\\backslash"), "with\\backslash");
    }
}
