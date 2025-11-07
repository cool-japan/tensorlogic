//! Metadata and source location tracking.

use serde::{Deserialize, Serialize};

/// Source code location information
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
}

impl SourceLocation {
    pub fn new(file: impl Into<String>, line: usize, column: usize) -> Self {
        SourceLocation {
            file: file.into(),
            line,
            column,
        }
    }

    pub fn unknown() -> Self {
        SourceLocation {
            file: "<unknown>".to_string(),
            line: 0,
            column: 0,
        }
    }
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// Span information covering a range in source code
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start: SourceLocation,
    pub end: SourceLocation,
}

impl SourceSpan {
    pub fn new(start: SourceLocation, end: SourceLocation) -> Self {
        SourceSpan { start, end }
    }

    pub fn single(location: SourceLocation) -> Self {
        SourceSpan {
            start: location.clone(),
            end: location,
        }
    }

    pub fn unknown() -> Self {
        SourceSpan {
            start: SourceLocation::unknown(),
            end: SourceLocation::unknown(),
        }
    }
}

impl std::fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.start.file == self.end.file {
            if self.start.line == self.end.line {
                write!(
                    f,
                    "{}:{}:{}-{}",
                    self.start.file, self.start.line, self.start.column, self.end.column
                )
            } else {
                write!(
                    f,
                    "{} (lines {}-{})",
                    self.start.file, self.start.line, self.end.line
                )
            }
        } else {
            write!(f, "{} to {}", self.start, self.end)
        }
    }
}

/// Provenance information tracking the origin of an IR node
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Source rule ID or identifier
    pub rule_id: Option<String>,
    /// Source file information
    pub source_file: Option<String>,
    /// Source span
    pub span: Option<SourceSpan>,
    /// Additional metadata as key-value pairs
    pub attributes: Vec<(String, String)>,
}

impl Provenance {
    pub fn new() -> Self {
        Provenance {
            rule_id: None,
            source_file: None,
            span: None,
            attributes: Vec::new(),
        }
    }

    pub fn with_rule_id(mut self, rule_id: impl Into<String>) -> Self {
        self.rule_id = Some(rule_id.into());
        self
    }

    pub fn with_source_file(mut self, source_file: impl Into<String>) -> Self {
        self.source_file = Some(source_file.into());
        self
    }

    pub fn with_span(mut self, span: SourceSpan) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.push((key.into(), value.into()));
        self
    }

    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata container that can be attached to IR nodes
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    /// Human-readable name or comment
    pub name: Option<String>,
    /// Source location information
    pub span: Option<SourceSpan>,
    /// Provenance tracking
    pub provenance: Option<Provenance>,
    /// Additional custom attributes
    pub attributes: Vec<(String, String)>,
}

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_span(mut self, span: SourceSpan) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.push((key.into(), value.into()));
        self
    }

    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new("test.tl", 10, 5);
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);
        assert_eq!(loc.to_string(), "test.tl:10:5");
    }

    #[test]
    fn test_source_span() {
        let start = SourceLocation::new("test.tl", 10, 5);
        let end = SourceLocation::new("test.tl", 10, 20);
        let span = SourceSpan::new(start, end);
        assert_eq!(span.to_string(), "test.tl:10:5-20");
    }

    #[test]
    fn test_provenance() {
        let prov = Provenance::new()
            .with_rule_id("rule_1")
            .with_source_file("test.tl")
            .with_attribute("author", "system");

        assert_eq!(prov.rule_id, Some("rule_1".to_string()));
        assert_eq!(prov.get_attribute("author"), Some("system"));
    }

    #[test]
    fn test_metadata() {
        let span = SourceSpan::single(SourceLocation::new("test.tl", 10, 5));
        let prov = Provenance::new().with_rule_id("rule_1");

        let meta = Metadata::new()
            .with_name("transitivity")
            .with_span(span.clone())
            .with_provenance(prov.clone())
            .with_attribute("version", "1.0");

        assert_eq!(meta.name, Some("transitivity".to_string()));
        assert_eq!(meta.span, Some(span));
        assert_eq!(meta.provenance, Some(prov));
        assert_eq!(meta.get_attribute("version"), Some("1.0"));
    }
}
