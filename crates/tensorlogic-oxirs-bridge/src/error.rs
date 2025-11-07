//! Error types for the OxiRS bridge.

use thiserror::Error;

/// Parse location information for better error reporting
#[derive(Debug, Clone)]
pub struct ParseLocation {
    pub line: usize,
    pub column: usize,
    pub context: Option<String>,
}

impl ParseLocation {
    pub fn new(line: usize, column: usize) -> Self {
        Self {
            line,
            column,
            context: None,
        }
    }

    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }
}

impl std::fmt::Display for ParseLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)?;
        if let Some(ctx) = &self.context {
            write!(f, "\n  {}", ctx)?;
        }
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum BridgeError {
    #[error("Invalid RDF schema: {0}")]
    InvalidSchema(String),

    #[error("Parse error at {location}: {message}")]
    ParseError {
        location: ParseLocation,
        message: String,
    },

    #[error("SHACL shape not supported: {0}")]
    UnsupportedShape(String),

    #[error("Property '{property}' not found in schema")]
    PropertyNotFound { property: String },

    #[error("Class '{class}' not found in schema")]
    ClassNotFound { class: String },

    #[error("Provenance tracking error: {0}")]
    ProvenanceError(String),

    #[error("Invalid IRI '{iri}': {reason}")]
    InvalidIri { iri: String, reason: String },

    #[error("Missing required field '{field}' in {context}")]
    MissingField { field: String, context: String },

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Index error: {0}")]
    IndexError(String),

    #[error("Cache error: {0}")]
    CacheError(String),
}

impl BridgeError {
    /// Create a parse error with location
    pub fn parse_error(line: usize, column: usize, message: impl Into<String>) -> Self {
        Self::ParseError {
            location: ParseLocation::new(line, column),
            message: message.into(),
        }
    }

    /// Create a parse error with location and context
    pub fn parse_error_with_context(
        line: usize,
        column: usize,
        message: impl Into<String>,
        context: impl Into<String>,
    ) -> Self {
        Self::ParseError {
            location: ParseLocation::new(line, column).with_context(context.into()),
            message: message.into(),
        }
    }

    /// Suggest a fix for this error
    pub fn suggestion(&self) -> Option<String> {
        match self {
            Self::PropertyNotFound { property } => {
                Some(format!("Did you mean to define a property with IRI '{}'? Add it to your RDF schema with 'a rdf:Property'.", property))
            }
            Self::ClassNotFound { class } => {
                Some(format!("Did you mean to define a class with IRI '{}'? Add it to your RDF schema with 'a rdfs:Class'.", class))
            }
            Self::InvalidIri { iri, reason } => {
                Some(format!("Ensure '{}' is a valid IRI. Reason: {}", iri, reason))
            }
            Self::MissingField { field, context } => {
                Some(format!("Add the required field '{}' to {}", field, context))
            }
            _ => None,
        }
    }
}
