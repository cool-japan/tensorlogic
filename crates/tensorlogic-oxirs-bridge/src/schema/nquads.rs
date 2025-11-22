//! N-Quads serialization support for RDF data with named graphs
//!
//! N-Quads extends N-Triples by adding a fourth element for named graphs.
//! Each line encodes one RDF quad (subject, predicate, object, graph).
//!
//! ## Format Specification
//!
//! Each line contains:
//! - Subject (IRI or blank node)
//! - Predicate (IRI)
//! - Object (IRI, blank node, or literal)
//! - Graph name (IRI) - optional, if omitted, uses default graph
//! - Terminating '.' and newline
//!
//! ## Example
//!
//! ```text
//! <http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> <http://example.org/graph1> .
//! <http://example.org/Bob> <http://example.org/knows> <http://example.org/Charlie> .
//! ```

use anyhow::Result;
use oxrdf::{Literal, NamedNode, NamedOrBlankNode, Term, Triple};
use std::collections::HashMap;

/// A quad represents an RDF statement with an optional named graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Quad {
    /// The subject of the statement
    pub subject: String,
    /// The predicate of the statement
    pub predicate: String,
    /// The object of the statement
    pub object: String,
    /// The graph IRI (None for default graph)
    pub graph: Option<String>,
}

impl Quad {
    /// Create a new quad with a named graph.
    pub fn new(subject: String, predicate: String, object: String, graph: Option<String>) -> Self {
        Quad {
            subject,
            predicate,
            object,
            graph,
        }
    }

    /// Create a quad in the default graph.
    pub fn default_graph(subject: String, predicate: String, object: String) -> Self {
        Quad {
            subject,
            predicate,
            object,
            graph: None,
        }
    }

    /// Convert to N-Quads format string.
    pub fn to_nquads(&self) -> String {
        let graph_part = if let Some(ref g) = self.graph {
            format!(" <{}>", g)
        } else {
            String::new()
        };

        format!(
            "<{}> <{}> {}{} .\n",
            self.subject,
            self.predicate,
            format_object(&self.object),
            graph_part
        )
    }
}

/// N-Quads parser and serializer.
pub struct NQuadsProcessor {
    /// Parsed quads grouped by graph
    graphs: HashMap<Option<String>, Vec<Quad>>,
}

impl NQuadsProcessor {
    /// Create a new N-Quads processor.
    pub fn new() -> Self {
        NQuadsProcessor {
            graphs: HashMap::new(),
        }
    }

    /// Parse N-Quads data.
    ///
    /// ## Example
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::schema::nquads::NQuadsProcessor;
    ///
    /// let nquads = r#"
    /// <http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> <http://example.org/graph1> .
    /// <http://example.org/Bob> <http://example.org/knows> <http://example.org/Charlie> .
    /// "#;
    ///
    /// let mut processor = NQuadsProcessor::new();
    /// processor.load_nquads(nquads).unwrap();
    ///
    /// assert_eq!(processor.total_quads(), 2);
    /// assert_eq!(processor.graph_count(), 2);
    /// ```
    pub fn load_nquads(&mut self, data: &str) -> Result<()> {
        for line in data.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let quad = parse_nquad_line(line)?;

            self.graphs
                .entry(quad.graph.clone())
                .or_default()
                .push(quad);
        }

        Ok(())
    }

    /// Get the total number of quads.
    pub fn total_quads(&self) -> usize {
        self.graphs.values().map(|v| v.len()).sum()
    }

    /// Get the number of graphs (including default graph if present).
    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Get quads for a specific graph (None for default graph).
    pub fn get_graph(&self, graph_iri: Option<&str>) -> Option<&Vec<Quad>> {
        let key = graph_iri.map(|s| s.to_string());
        self.graphs.get(&key)
    }

    /// Get all graph IRIs (excluding default graph).
    pub fn graph_iris(&self) -> Vec<&str> {
        self.graphs.keys().filter_map(|k| k.as_deref()).collect()
    }

    /// Export all quads as N-Quads format string.
    pub fn to_nquads(&self) -> String {
        let mut output = String::new();

        // Output default graph first
        if let Some(quads) = self.graphs.get(&None) {
            for quad in quads {
                output.push_str(&quad.to_nquads());
            }
        }

        // Then named graphs
        for (graph, quads) in &self.graphs {
            if graph.is_some() {
                for quad in quads {
                    output.push_str(&quad.to_nquads());
                }
            }
        }

        output
    }

    /// Get all quads as an iterator.
    pub fn all_quads(&self) -> impl Iterator<Item = &Quad> {
        self.graphs.values().flat_map(|v| v.iter())
    }

    /// Convert to triples (losing graph information).
    pub fn to_triples(&self) -> Result<Vec<Triple>> {
        let mut triples = Vec::new();

        for quad in self.all_quads() {
            let triple = quad_to_triple(quad)?;
            triples.push(triple);
        }

        Ok(triples)
    }

    /// Clear all quads.
    pub fn clear(&mut self) {
        self.graphs.clear();
    }

    /// Add a quad.
    pub fn add_quad(&mut self, quad: Quad) {
        self.graphs
            .entry(quad.graph.clone())
            .or_default()
            .push(quad);
    }

    /// Merge another processor's quads into this one.
    pub fn merge(&mut self, other: NQuadsProcessor) {
        for (graph, quads) in other.graphs {
            self.graphs.entry(graph).or_default().extend(quads);
        }
    }
}

impl Default for NQuadsProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a single N-Quad line.
fn parse_nquad_line(line: &str) -> Result<Quad> {
    // Remove trailing ' .' if present
    let line = line.trim_end_matches('.').trim();

    // Split into components, being careful about quoted strings
    let parts = split_nquad_components(line);

    if parts.len() < 3 {
        anyhow::bail!("Invalid N-Quad line (need at least 3 components): {}", line);
    }

    // Parse subject
    let subject = parse_iri(&parts[0])?;

    // Parse predicate
    let predicate = parse_iri(&parts[1])?;

    // Parse object
    let object = parts[2].to_string();

    // Parse optional graph
    let graph = if parts.len() > 3 {
        Some(parse_iri(&parts[3])?)
    } else {
        None
    };

    Ok(Quad {
        subject,
        predicate,
        object,
        graph,
    })
}

/// Split N-Quad line into components, handling quoted strings.
fn split_nquad_components(line: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut escape_next = false;

    for c in line.chars() {
        if escape_next {
            current.push(c);
            escape_next = false;
            continue;
        }

        match c {
            '\\' if in_quotes => {
                current.push(c);
                escape_next = true;
            }
            '"' => {
                current.push(c);
                in_quotes = !in_quotes;
            }
            ' ' | '\t' if !in_quotes => {
                if !current.is_empty() {
                    parts.push(current);
                    current = String::new();
                }
            }
            _ => {
                current.push(c);
            }
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

/// Parse an IRI from angle brackets.
fn parse_iri(s: &str) -> Result<String> {
    if let Some(iri) = s.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        Ok(iri.to_string())
    } else if s.starts_with("_:") {
        // Blank node
        Ok(s.to_string())
    } else {
        anyhow::bail!("Invalid IRI: {}", s);
    }
}

/// Format an object value for N-Quads output.
fn format_object(object: &str) -> String {
    if object.starts_with('<') || object.starts_with("_:") || object.starts_with('"') {
        object.to_string()
    } else {
        // Assume it's a literal that needs quoting
        format!("\"{}\"", object.replace('\\', "\\\\").replace('"', "\\\""))
    }
}

/// Convert a Quad to an oxrdf Triple.
fn quad_to_triple(quad: &Quad) -> Result<Triple> {
    // Parse subject
    let subject = if quad.subject.starts_with("_:") {
        NamedOrBlankNode::BlankNode(oxrdf::BlankNode::new(&quad.subject[2..])?)
    } else {
        NamedOrBlankNode::NamedNode(NamedNode::new(&quad.subject)?)
    };

    // Parse predicate
    let predicate = NamedNode::new(&quad.predicate)?;

    // Parse object
    let object = if quad.object.starts_with('<') && quad.object.ends_with('>') {
        Term::NamedNode(NamedNode::new(&quad.object[1..quad.object.len() - 1])?)
    } else if quad.object.starts_with("_:") {
        Term::BlankNode(oxrdf::BlankNode::new(&quad.object[2..])?)
    } else if quad.object.starts_with('"') {
        // Parse literal
        let value = extract_literal_value(&quad.object);
        Term::Literal(Literal::new_simple_literal(&value))
    } else {
        // Treat as simple literal
        Term::Literal(Literal::new_simple_literal(&quad.object))
    };

    Ok(Triple::new(subject, predicate, object))
}

/// Extract literal value from quoted string.
fn extract_literal_value(s: &str) -> String {
    if let Some(rest) = s.strip_prefix('"') {
        // Find closing quote
        let mut value = String::new();
        let mut escape_next = false;

        for c in rest.chars() {
            if escape_next {
                match c {
                    'n' => value.push('\n'),
                    'r' => value.push('\r'),
                    't' => value.push('\t'),
                    '"' => value.push('"'),
                    '\\' => value.push('\\'),
                    _ => value.push(c),
                }
                escape_next = false;
                continue;
            }

            if c == '\\' {
                escape_next = true;
                continue;
            }

            if c == '"' {
                break;
            }

            value.push(c);
        }

        value
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nquad_default_graph() {
        let line =
            r#"<http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> ."#;
        let quad = parse_nquad_line(line).unwrap();

        assert_eq!(quad.subject, "http://example.org/Alice");
        assert_eq!(quad.predicate, "http://example.org/knows");
        assert_eq!(quad.object, "<http://example.org/Bob>");
        assert_eq!(quad.graph, None);
    }

    #[test]
    fn test_parse_nquad_named_graph() {
        let line = r#"<http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> <http://example.org/graph1> ."#;
        let quad = parse_nquad_line(line).unwrap();

        assert_eq!(quad.subject, "http://example.org/Alice");
        assert_eq!(quad.predicate, "http://example.org/knows");
        assert_eq!(quad.object, "<http://example.org/Bob>");
        assert_eq!(quad.graph, Some("http://example.org/graph1".to_string()));
    }

    #[test]
    fn test_parse_nquad_with_literal() {
        let line =
            r#"<http://example.org/Alice> <http://www.w3.org/2000/01/rdf-schema#label> "Alice" ."#;
        let quad = parse_nquad_line(line).unwrap();

        assert_eq!(quad.subject, "http://example.org/Alice");
        assert_eq!(quad.object, r#""Alice""#);
    }

    #[test]
    fn test_processor_load() {
        let nquads = r#"
            <http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> <http://example.org/graph1> .
            <http://example.org/Bob> <http://example.org/knows> <http://example.org/Charlie> .
            <http://example.org/Charlie> <http://example.org/knows> <http://example.org/Alice> <http://example.org/graph2> .
        "#;

        let mut processor = NQuadsProcessor::new();
        processor.load_nquads(nquads).unwrap();

        assert_eq!(processor.total_quads(), 3);
        assert_eq!(processor.graph_count(), 3); // default + graph1 + graph2
    }

    #[test]
    fn test_processor_get_graph() {
        let nquads = r#"
            <http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> <http://example.org/graph1> .
            <http://example.org/Bob> <http://example.org/knows> <http://example.org/Charlie> .
        "#;

        let mut processor = NQuadsProcessor::new();
        processor.load_nquads(nquads).unwrap();

        let default = processor.get_graph(None).unwrap();
        assert_eq!(default.len(), 1);

        let graph1 = processor
            .get_graph(Some("http://example.org/graph1"))
            .unwrap();
        assert_eq!(graph1.len(), 1);
    }

    #[test]
    fn test_processor_to_nquads() {
        let mut processor = NQuadsProcessor::new();

        processor.add_quad(Quad::new(
            "http://example.org/Alice".to_string(),
            "http://example.org/knows".to_string(),
            "<http://example.org/Bob>".to_string(),
            Some("http://example.org/graph1".to_string()),
        ));

        let output = processor.to_nquads();
        assert!(output.contains("http://example.org/Alice"));
        assert!(output.contains("http://example.org/graph1"));
    }

    #[test]
    fn test_quad_to_triple() {
        let quad = Quad::new(
            "http://example.org/Alice".to_string(),
            "http://example.org/knows".to_string(),
            "<http://example.org/Bob>".to_string(),
            None,
        );

        let triple = quad_to_triple(&quad).unwrap();
        assert_eq!(triple.predicate.as_str(), "http://example.org/knows");
    }

    #[test]
    fn test_graph_iris() {
        let nquads = r#"
            <http://example.org/a> <http://example.org/p> <http://example.org/b> <http://example.org/g1> .
            <http://example.org/a> <http://example.org/p> <http://example.org/c> <http://example.org/g2> .
            <http://example.org/a> <http://example.org/p> <http://example.org/d> .
        "#;

        let mut processor = NQuadsProcessor::new();
        processor.load_nquads(nquads).unwrap();

        let iris = processor.graph_iris();
        assert!(iris.contains(&"http://example.org/g1"));
        assert!(iris.contains(&"http://example.org/g2"));
    }

    #[test]
    fn test_roundtrip() {
        let original = r#"<http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> <http://example.org/graph1> .
<http://example.org/Bob> <http://example.org/knows> <http://example.org/Charlie> .
"#;

        let mut processor = NQuadsProcessor::new();
        processor.load_nquads(original).unwrap();

        let output = processor.to_nquads();

        let mut processor2 = NQuadsProcessor::new();
        processor2.load_nquads(&output).unwrap();

        assert_eq!(processor.total_quads(), processor2.total_quads());
    }

    #[test]
    fn test_literal_with_escape() {
        let line = r#"<http://example.org/s> <http://example.org/p> "line1\nline2" ."#;
        let quad = parse_nquad_line(line).unwrap();

        let triple = quad_to_triple(&quad).unwrap();
        if let Term::Literal(lit) = triple.object {
            assert!(lit.value().contains('\n'));
        } else {
            panic!("Expected literal");
        }
    }
}
