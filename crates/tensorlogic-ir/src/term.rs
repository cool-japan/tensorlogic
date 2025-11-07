//! Terms: variables and constants.

use serde::{Deserialize, Serialize};

/// Type annotation for a term.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeAnnotation {
    pub type_name: String,
}

impl TypeAnnotation {
    pub fn new(type_name: impl Into<String>) -> Self {
        TypeAnnotation {
            type_name: type_name.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Term {
    Var(String),
    Const(String),
    /// Typed term with explicit type annotation
    Typed {
        value: Box<Term>,
        type_annotation: TypeAnnotation,
    },
}

impl Term {
    pub fn var(name: impl Into<String>) -> Self {
        Term::Var(name.into())
    }

    pub fn constant(name: impl Into<String>) -> Self {
        Term::Const(name.into())
    }

    /// Create a typed variable
    pub fn typed_var(name: impl Into<String>, type_name: impl Into<String>) -> Self {
        Term::Typed {
            value: Box::new(Term::Var(name.into())),
            type_annotation: TypeAnnotation::new(type_name),
        }
    }

    /// Create a typed constant
    pub fn typed_const(name: impl Into<String>, type_name: impl Into<String>) -> Self {
        Term::Typed {
            value: Box::new(Term::Const(name.into())),
            type_annotation: TypeAnnotation::new(type_name),
        }
    }

    /// Attach a type annotation to an existing term
    pub fn with_type(self, type_name: impl Into<String>) -> Self {
        Term::Typed {
            value: Box::new(self),
            type_annotation: TypeAnnotation::new(type_name),
        }
    }

    pub fn is_var(&self) -> bool {
        match self {
            Term::Var(_) => true,
            Term::Typed { value, .. } => value.is_var(),
            _ => false,
        }
    }

    pub fn is_const(&self) -> bool {
        match self {
            Term::Const(_) => true,
            Term::Typed { value, .. } => value.is_const(),
            _ => false,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Term::Var(n) | Term::Const(n) => n,
            Term::Typed { value, .. } => value.name(),
        }
    }

    /// Get the type annotation if present
    pub fn get_type(&self) -> Option<&TypeAnnotation> {
        match self {
            Term::Typed {
                type_annotation, ..
            } => Some(type_annotation),
            _ => None,
        }
    }

    /// Get the underlying untyped term
    pub fn untyped(&self) -> &Term {
        match self {
            Term::Typed { value, .. } => value.untyped(),
            term => term,
        }
    }
}
