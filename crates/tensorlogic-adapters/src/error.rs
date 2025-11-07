//! Error types for adapters.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum AdapterError {
    #[error("Domain '{0}' not found")]
    DomainNotFound(String),
    #[error("Predicate '{0}' not found")]
    PredicateNotFound(String),
    #[error("Variable '{0}' not bound to any domain")]
    UnboundVariable(String),
    #[error("Arity mismatch for predicate '{name}': expected {expected}, found {found}")]
    ArityMismatch {
        name: String,
        expected: usize,
        found: usize,
    },
    #[error("Invalid domain element: {0}")]
    InvalidDomainElement(String),
    #[error("Invalid parametric type: {0}")]
    InvalidParametricType(String),
    #[error("Unknown domain: {0}")]
    UnknownDomain(String),
    #[error("Unknown predicate: {0}")]
    UnknownPredicate(String),
    #[error("Duplicate domain: {0}")]
    DuplicateDomain(String),
    #[error("Invalid cardinality: {0}")]
    InvalidCardinality(String),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}
