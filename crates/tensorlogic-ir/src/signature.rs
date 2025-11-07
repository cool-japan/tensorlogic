//! Predicate signatures and metadata.

use serde::{Deserialize, Serialize};

use crate::TypeAnnotation;

/// Signature for a predicate: defines expected argument types
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateSignature {
    pub name: String,
    pub arg_types: Vec<TypeAnnotation>,
    pub arity: usize,
}

impl PredicateSignature {
    pub fn new(name: impl Into<String>, arg_types: Vec<TypeAnnotation>) -> Self {
        let arity = arg_types.len();
        PredicateSignature {
            name: name.into(),
            arg_types,
            arity,
        }
    }

    /// Create an untyped signature (for backward compatibility)
    pub fn untyped(name: impl Into<String>, arity: usize) -> Self {
        PredicateSignature {
            name: name.into(),
            arg_types: Vec::new(),
            arity,
        }
    }

    /// Check if this signature matches the given number of arguments
    pub fn matches_arity(&self, arg_count: usize) -> bool {
        self.arity == arg_count
    }

    /// Check if the given argument types match this signature
    pub fn matches_types(&self, arg_types: &[Option<&TypeAnnotation>]) -> bool {
        if arg_types.len() != self.arity {
            return false;
        }

        // If signature has no type annotations, accept any types
        if self.arg_types.is_empty() {
            return true;
        }

        // Check each argument type
        for (i, expected) in self.arg_types.iter().enumerate() {
            if let Some(actual) = arg_types[i] {
                if expected != actual {
                    return false;
                }
            }
            // If actual type is None, we accept it (untyped argument)
        }

        true
    }
}

/// Registry of predicate signatures
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SignatureRegistry {
    signatures: Vec<PredicateSignature>,
}

impl SignatureRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new predicate signature
    pub fn register(&mut self, signature: PredicateSignature) {
        self.signatures.push(signature);
    }

    /// Look up a signature by predicate name
    pub fn get(&self, name: &str) -> Option<&PredicateSignature> {
        self.signatures.iter().find(|sig| sig.name == name)
    }

    /// Get all registered signatures
    pub fn all(&self) -> &[PredicateSignature] {
        &self.signatures
    }

    /// Check if a predicate is registered
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_creation() {
        let sig = PredicateSignature::new(
            "knows",
            vec![TypeAnnotation::new("Person"), TypeAnnotation::new("Person")],
        );
        assert_eq!(sig.name, "knows");
        assert_eq!(sig.arity, 2);
        assert_eq!(sig.arg_types.len(), 2);
    }

    #[test]
    fn test_signature_arity_matching() {
        let sig = PredicateSignature::new(
            "knows",
            vec![TypeAnnotation::new("Person"), TypeAnnotation::new("Person")],
        );
        assert!(sig.matches_arity(2));
        assert!(!sig.matches_arity(1));
        assert!(!sig.matches_arity(3));
    }

    #[test]
    fn test_signature_type_matching() {
        let sig = PredicateSignature::new(
            "knows",
            vec![TypeAnnotation::new("Person"), TypeAnnotation::new("Person")],
        );

        let person_type = TypeAnnotation::new("Person");
        let thing_type = TypeAnnotation::new("Thing");

        // Matching types
        assert!(sig.matches_types(&[Some(&person_type), Some(&person_type)]));

        // Mismatched types
        assert!(!sig.matches_types(&[Some(&person_type), Some(&thing_type)]));

        // Untyped arguments (should accept)
        assert!(sig.matches_types(&[None, Some(&person_type)]));
        assert!(sig.matches_types(&[None, None]));
    }

    #[test]
    fn test_signature_registry() {
        let mut registry = SignatureRegistry::new();

        let knows_sig = PredicateSignature::new(
            "knows",
            vec![TypeAnnotation::new("Person"), TypeAnnotation::new("Person")],
        );
        registry.register(knows_sig);

        assert!(registry.contains("knows"));
        assert!(!registry.contains("likes"));

        let retrieved = registry.get("knows").unwrap();
        assert_eq!(retrieved.arity, 2);
    }

    #[test]
    fn test_untyped_signature() {
        let sig = PredicateSignature::untyped("pred", 3);
        assert_eq!(sig.arity, 3);
        assert!(sig.arg_types.is_empty());

        // Untyped signature should accept any types
        let any_type = TypeAnnotation::new("AnyType");
        assert!(sig.matches_types(&[Some(&any_type), None, Some(&any_type)]));
    }
}
