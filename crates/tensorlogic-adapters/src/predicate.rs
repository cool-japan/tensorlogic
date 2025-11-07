//! Predicate metadata.

use serde::{Deserialize, Serialize};
use tensorlogic_ir::Term;

use crate::constraint::PredicateConstraints;
use crate::error::AdapterError;
use crate::metadata::Metadata;

/// Predicate metadata including arity and domain types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredicateInfo {
    pub name: String,
    pub arity: usize,
    pub arg_domains: Vec<String>,
    pub description: Option<String>,
    pub constraints: Option<PredicateConstraints>,
    /// Rich metadata including provenance, documentation, tags
    pub metadata: Option<Metadata>,
}

impl PredicateInfo {
    pub fn new(name: impl Into<String>, arg_domains: Vec<String>) -> Self {
        let arity = arg_domains.len();
        PredicateInfo {
            name: name.into(),
            arity,
            arg_domains,
            description: None,
            constraints: None,
            metadata: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_constraints(mut self, constraints: PredicateConstraints) -> Self {
        self.constraints = Some(constraints);
        self
    }

    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn validate_args(&self, args: &[Term]) -> Result<(), AdapterError> {
        if args.len() != self.arity {
            return Err(AdapterError::ArityMismatch {
                name: self.name.clone(),
                expected: self.arity,
                found: args.len(),
            });
        }
        Ok(())
    }
}
