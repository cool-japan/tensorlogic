//! Predicate signatures and metadata.

use serde::{Deserialize, Serialize};

use crate::parametric_types::{unify, ParametricType, TypeSubstitution};
use crate::{IrError, TypeAnnotation};

/// Signature for a predicate: defines expected argument types
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateSignature {
    pub name: String,
    pub arg_types: Vec<TypeAnnotation>,
    pub arity: usize,
    /// Optional parametric type signature for generic predicates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parametric_types: Option<Vec<ParametricType>>,
}

impl PredicateSignature {
    pub fn new(name: impl Into<String>, arg_types: Vec<TypeAnnotation>) -> Self {
        let arity = arg_types.len();
        PredicateSignature {
            name: name.into(),
            arg_types,
            arity,
            parametric_types: None,
        }
    }

    /// Create a parametric signature
    pub fn parametric(name: impl Into<String>, parametric_types: Vec<ParametricType>) -> Self {
        let arity = parametric_types.len();
        PredicateSignature {
            name: name.into(),
            arg_types: Vec::new(), // Populated from parametric types if needed
            arity,
            parametric_types: Some(parametric_types),
        }
    }

    /// Create an untyped signature (for backward compatibility)
    pub fn untyped(name: impl Into<String>, arity: usize) -> Self {
        PredicateSignature {
            name: name.into(),
            arg_types: Vec::new(),
            arity,
            parametric_types: None,
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
        if self.arg_types.is_empty() && self.parametric_types.is_none() {
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

    /// Unify parametric signature with concrete argument types
    pub fn unify_parametric(
        &self,
        arg_types: &[ParametricType],
    ) -> Result<TypeSubstitution, IrError> {
        if arg_types.len() != self.arity {
            return Err(IrError::ArityMismatch {
                name: self.name.clone(),
                expected: self.arity,
                actual: arg_types.len(),
            });
        }

        let Some(ref param_types) = self.parametric_types else {
            // No parametric types, fall back to simple matching
            return Ok(TypeSubstitution::new());
        };

        // Unify each argument type with the parametric signature
        let mut subst = TypeSubstitution::new();
        for (expected, actual) in param_types.iter().zip(arg_types.iter()) {
            let new_subst = unify(expected, actual)?;
            // Compose substitutions
            subst = crate::parametric_types::compose_substitutions(&subst, &new_subst);
        }

        Ok(subst)
    }

    /// Check if this is a parametric signature
    pub fn is_parametric(&self) -> bool {
        self.parametric_types.is_some()
    }

    /// Get the parametric types if present
    pub fn get_parametric_types(&self) -> Option<&[ParametricType]> {
        self.parametric_types.as_deref()
    }

    /// Instantiate a parametric signature with a substitution
    pub fn instantiate(&self, subst: &TypeSubstitution) -> PredicateSignature {
        let parametric_types = self
            .parametric_types
            .as_ref()
            .map(|types| types.iter().map(|ty| ty.substitute(subst)).collect());

        PredicateSignature {
            name: self.name.clone(),
            arg_types: self.arg_types.clone(),
            arity: self.arity,
            parametric_types,
        }
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

    #[test]
    fn test_parametric_signature_creation() {
        let t_var = ParametricType::variable("T");
        let sig = PredicateSignature::parametric(
            "contains",
            vec![ParametricType::list(t_var.clone()), t_var.clone()],
        );

        assert_eq!(sig.name, "contains");
        assert_eq!(sig.arity, 2);
        assert!(sig.is_parametric());
        assert_eq!(sig.get_parametric_types().unwrap().len(), 2);
    }

    #[test]
    fn test_parametric_signature_unification() {
        let t_var = ParametricType::variable("T");
        let sig = PredicateSignature::parametric(
            "contains",
            vec![ParametricType::list(t_var.clone()), t_var.clone()],
        );

        let int_type = ParametricType::concrete("Int");
        let list_int = ParametricType::list(int_type.clone());

        // Unify List<T>, T with List<Int>, Int
        let subst = sig.unify_parametric(&[list_int, int_type.clone()]).unwrap();
        assert_eq!(subst.get("T").unwrap(), &int_type);
    }

    #[test]
    fn test_parametric_signature_instantiation() {
        let t_var = ParametricType::variable("T");
        let sig = PredicateSignature::parametric("identity", vec![t_var.clone(), t_var.clone()]);

        let int_type = ParametricType::concrete("Int");
        let mut subst = TypeSubstitution::new();
        subst.insert("T".to_string(), int_type.clone());

        let instantiated = sig.instantiate(&subst);
        assert!(instantiated.is_parametric());
        let param_types = instantiated.get_parametric_types().unwrap();
        assert_eq!(param_types[0], int_type);
        assert_eq!(param_types[1], int_type);
    }

    #[test]
    fn test_parametric_signature_arity_mismatch() {
        let t_var = ParametricType::variable("T");
        let sig = PredicateSignature::parametric("pred", vec![t_var.clone()]);

        let int_type = ParametricType::concrete("Int");
        // Provide 2 arguments when signature expects 1
        let result = sig.unify_parametric(&[int_type.clone(), int_type]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parametric_signature_complex_types() {
        let t_var = ParametricType::variable("T");
        let u_var = ParametricType::variable("U");

        // map_over: (T -> U, List<T>) -> List<U>
        let sig = PredicateSignature::parametric(
            "map_over",
            vec![
                ParametricType::function(t_var.clone(), u_var.clone()),
                ParametricType::list(t_var.clone()),
                ParametricType::list(u_var.clone()),
            ],
        );

        let int_type = ParametricType::concrete("Int");
        let string_type = ParametricType::concrete("String");

        // Unify with (Int -> String, List<Int>, List<String>)
        let subst = sig
            .unify_parametric(&[
                ParametricType::function(int_type.clone(), string_type.clone()),
                ParametricType::list(int_type.clone()),
                ParametricType::list(string_type.clone()),
            ])
            .unwrap();

        assert_eq!(subst.get("T").unwrap(), &int_type);
        assert_eq!(subst.get("U").unwrap(), &string_type);
    }

    #[test]
    fn test_type_annotation_parametric_conversion() {
        let type_ann = TypeAnnotation::new("Int");
        let param_type = type_ann.to_parametric();
        assert_eq!(param_type, ParametricType::concrete("Int"));

        // Convert back
        let converted_back = TypeAnnotation::from_parametric(&param_type);
        assert_eq!(converted_back, Some(type_ann));

        // Can't convert parametric types back
        let list_int = ParametricType::list(ParametricType::concrete("Int"));
        assert!(TypeAnnotation::from_parametric(&list_int).is_none());
    }
}
