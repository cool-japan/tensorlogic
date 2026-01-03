//! Parameterized domain types for generic type definitions.
//!
//! This module provides support for parameterized domains such as:
//! - `List<T>`: A list of elements of type T
//! - `Option<T>`: An optional value of type T
//! - `Pair<A, B>`: A pair of values of types A and B
//! - `Map<K, V>`: A mapping from keys of type K to values of type V
//!
//! These parameterized types enable rich type definitions in the symbol table
//! and support for complex data structures in logical rules.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::AdapterError;

/// A parameterized domain type with type parameters.
///
/// Examples:
/// - `List<Person>` - A list of persons
/// - `Option<City>` - An optional city
/// - `Pair<Person, City>` - A pair of person and city
/// - `Map<String, Int>` - A mapping from strings to integers
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParametricType {
    /// The type constructor name (e.g., "List", "Option", "Pair")
    pub constructor: String,
    /// The type parameters (e.g., `["Person"]` for `List<Person>`)
    pub parameters: Vec<TypeParameter>,
}

/// A type parameter that can be either a concrete type or another parametric type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeParameter {
    /// A concrete type name (e.g., "Person", "Int")
    Concrete(String),
    /// A nested parametric type (e.g., `List<Option<Person>>`)
    Parametric(Box<ParametricType>),
}

impl ParametricType {
    /// Creates a new parametric type with the given constructor and parameters.
    pub fn new(constructor: impl Into<String>, parameters: Vec<TypeParameter>) -> Self {
        ParametricType {
            constructor: constructor.into(),
            parameters,
        }
    }

    /// Creates a `List<T>` parametric type.
    pub fn list(inner: TypeParameter) -> Self {
        ParametricType::new("List", vec![inner])
    }

    /// Creates an `Option<T>` parametric type.
    pub fn option(inner: TypeParameter) -> Self {
        ParametricType::new("Option", vec![inner])
    }

    /// Creates a Pair<A, B> parametric type.
    pub fn pair(first: TypeParameter, second: TypeParameter) -> Self {
        ParametricType::new("Pair", vec![first, second])
    }

    /// Creates a Map<K, V> parametric type.
    pub fn map(key: TypeParameter, value: TypeParameter) -> Self {
        ParametricType::new("Map", vec![key, value])
    }

    /// Validates that the parametric type is well-formed.
    ///
    /// Checks:
    /// - Constructor name is not empty
    /// - Number of parameters matches expected arity for known constructors
    /// - All nested parametric types are also well-formed
    pub fn validate(&self) -> Result<(), AdapterError> {
        if self.constructor.is_empty() {
            return Err(AdapterError::InvalidParametricType(
                "Constructor name cannot be empty".to_string(),
            ));
        }

        // Validate arity for known constructors
        let expected_arity = match self.constructor.as_str() {
            "List" | "Option" | "Set" => 1,
            "Pair" | "Map" | "Either" => 2,
            _ => return Ok(()), // Unknown constructors are allowed
        };

        if self.parameters.len() != expected_arity {
            return Err(AdapterError::InvalidParametricType(format!(
                "Constructor '{}' expects {} parameters, found {}",
                self.constructor,
                expected_arity,
                self.parameters.len()
            )));
        }

        // Recursively validate nested parametric types
        for param in &self.parameters {
            if let TypeParameter::Parametric(nested) = param {
                nested.validate()?;
            }
        }

        Ok(())
    }

    /// Returns the arity (number of type parameters) of this parametric type.
    pub fn arity(&self) -> usize {
        self.parameters.len()
    }

    /// Checks if this is a monomorphic type (no type parameters).
    pub fn is_monomorphic(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Checks if this parametric type contains the given type parameter name.
    pub fn contains_parameter(&self, name: &str) -> bool {
        self.parameters.iter().any(|param| match param {
            TypeParameter::Concrete(n) => n == name,
            TypeParameter::Parametric(nested) => nested.contains_parameter(name),
        })
    }

    /// Substitutes all occurrences of a type parameter with a concrete type.
    ///
    /// This is used for type instantiation when applying a parametric type
    /// to concrete arguments.
    pub fn substitute(&self, from: &str, to: &TypeParameter) -> ParametricType {
        let new_params = self
            .parameters
            .iter()
            .map(|param| match param {
                TypeParameter::Concrete(name) if name == from => to.clone(),
                TypeParameter::Concrete(_) => param.clone(),
                TypeParameter::Parametric(nested) => {
                    TypeParameter::Parametric(Box::new(nested.substitute(from, to)))
                }
            })
            .collect();

        ParametricType {
            constructor: self.constructor.clone(),
            parameters: new_params,
        }
    }
}

impl TypeParameter {
    /// Creates a concrete type parameter.
    pub fn concrete(name: impl Into<String>) -> Self {
        TypeParameter::Concrete(name.into())
    }

    /// Creates a parametric type parameter.
    pub fn parametric(ptype: ParametricType) -> Self {
        TypeParameter::Parametric(Box::new(ptype))
    }

    /// Returns the name if this is a concrete type parameter.
    pub fn as_concrete(&self) -> Option<&str> {
        match self {
            TypeParameter::Concrete(name) => Some(name),
            TypeParameter::Parametric(_) => None,
        }
    }

    /// Returns the parametric type if this is a parametric type parameter.
    pub fn as_parametric(&self) -> Option<&ParametricType> {
        match self {
            TypeParameter::Concrete(_) => None,
            TypeParameter::Parametric(ptype) => Some(ptype),
        }
    }
}

impl fmt::Display for ParametricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.constructor)?;
        if !self.parameters.is_empty() {
            write!(f, "<")?;
            for (i, param) in self.parameters.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", param)?;
            }
            write!(f, ">")?;
        }
        Ok(())
    }
}

impl fmt::Display for TypeParameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeParameter::Concrete(name) => write!(f, "{}", name),
            TypeParameter::Parametric(ptype) => write!(f, "{}", ptype),
        }
    }
}

/// A type bound that constrains a type parameter.
///
/// Type bounds allow specifying constraints on type parameters, such as:
/// - `T: Comparable` - T must be a comparable type
/// - `T: Numeric` - T must be a numeric type
/// - `T: Subtypes(Person)` - T must be a subtype of Person
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeBound {
    /// The name of the type parameter being constrained
    pub param_name: String,
    /// The constraint kind
    pub constraint: BoundConstraint,
}

/// The kind of constraint in a type bound.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundConstraint {
    /// The type must be a subtype of the given type
    Subtype(String),
    /// The type must implement the given trait
    Trait(String),
    /// The type must be comparable (supports equality, ordering)
    Comparable,
    /// The type must be numeric (supports arithmetic operations)
    Numeric,
}

impl TypeBound {
    /// Creates a new type bound with the given constraint.
    pub fn new(param_name: impl Into<String>, constraint: BoundConstraint) -> Self {
        TypeBound {
            param_name: param_name.into(),
            constraint,
        }
    }

    /// Creates a subtype bound: `T: Subtypes(supertype)`
    pub fn subtype(param_name: impl Into<String>, supertype: impl Into<String>) -> Self {
        TypeBound::new(param_name, BoundConstraint::Subtype(supertype.into()))
    }

    /// Creates a trait bound: `T: Trait(trait_name)`
    pub fn trait_bound(param_name: impl Into<String>, trait_name: impl Into<String>) -> Self {
        TypeBound::new(param_name, BoundConstraint::Trait(trait_name.into()))
    }

    /// Creates a comparable bound: `T: Comparable`
    pub fn comparable(param_name: impl Into<String>) -> Self {
        TypeBound::new(param_name, BoundConstraint::Comparable)
    }

    /// Creates a numeric bound: `T: Numeric`
    pub fn numeric(param_name: impl Into<String>) -> Self {
        TypeBound::new(param_name, BoundConstraint::Numeric)
    }
}

impl fmt::Display for TypeBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.param_name)?;
        match &self.constraint {
            BoundConstraint::Subtype(s) => write!(f, "Subtype({})", s),
            BoundConstraint::Trait(t) => write!(f, "{}", t),
            BoundConstraint::Comparable => write!(f, "Comparable"),
            BoundConstraint::Numeric => write!(f, "Numeric"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parametric_type_list() {
        let list_person = ParametricType::list(TypeParameter::concrete("Person"));
        assert_eq!(list_person.constructor, "List");
        assert_eq!(list_person.arity(), 1);
        assert_eq!(list_person.to_string(), "List<Person>");
    }

    #[test]
    fn test_parametric_type_option() {
        let opt_city = ParametricType::option(TypeParameter::concrete("City"));
        assert_eq!(opt_city.constructor, "Option");
        assert_eq!(opt_city.arity(), 1);
        assert_eq!(opt_city.to_string(), "Option<City>");
    }

    #[test]
    fn test_parametric_type_pair() {
        let pair = ParametricType::pair(
            TypeParameter::concrete("Person"),
            TypeParameter::concrete("City"),
        );
        assert_eq!(pair.constructor, "Pair");
        assert_eq!(pair.arity(), 2);
        assert_eq!(pair.to_string(), "Pair<Person, City>");
    }

    #[test]
    fn test_parametric_type_map() {
        let map = ParametricType::map(
            TypeParameter::concrete("String"),
            TypeParameter::concrete("Int"),
        );
        assert_eq!(map.constructor, "Map");
        assert_eq!(map.arity(), 2);
        assert_eq!(map.to_string(), "Map<String, Int>");
    }

    #[test]
    fn test_nested_parametric_type() {
        let list_option_person = ParametricType::list(TypeParameter::parametric(
            ParametricType::option(TypeParameter::concrete("Person")),
        ));
        assert_eq!(list_option_person.to_string(), "List<Option<Person>>");
        assert!(list_option_person.contains_parameter("Person"));
    }

    #[test]
    fn test_parametric_type_validation() {
        let valid = ParametricType::list(TypeParameter::concrete("Person"));
        assert!(valid.validate().is_ok());

        let invalid = ParametricType::new(
            "List",
            vec![TypeParameter::concrete("A"), TypeParameter::concrete("B")],
        );
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_parametric_type_substitution() {
        let list_t = ParametricType::list(TypeParameter::concrete("T"));
        let list_person = list_t.substitute("T", &TypeParameter::concrete("Person"));
        assert_eq!(list_person.to_string(), "List<Person>");
    }

    #[test]
    fn test_type_bounds() {
        let bound = TypeBound::subtype("T", "Person");
        assert_eq!(bound.param_name, "T");
        assert_eq!(bound.to_string(), "T: Subtype(Person)");

        let comparable = TypeBound::comparable("T");
        assert_eq!(comparable.to_string(), "T: Comparable");

        let numeric = TypeBound::numeric("N");
        assert_eq!(numeric.to_string(), "N: Numeric");
    }
}
