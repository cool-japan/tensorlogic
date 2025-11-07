//! Domain constraints for variables and quantifiers.
//!
//! This module provides:
//! - Domain metadata (size, type, constraints)
//! - Domain registry for managing variable domains
//! - Domain validation and compatibility checking

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::IrError;

/// Represents a domain constraint for a variable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainInfo {
    /// Domain name (e.g., "Person", "Integer", "Real")
    pub name: String,
    /// Domain size (None for infinite domains like Real)
    pub size: Option<usize>,
    /// Domain type category
    pub domain_type: DomainType,
    /// Additional constraints (e.g., "positive", "bounded")
    pub constraints: Vec<String>,
    /// Optional metadata for custom domains
    pub metadata: HashMap<String, String>,
}

/// Type category of a domain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainType {
    /// Finite discrete domain (e.g., enum, categorical)
    Finite,
    /// Integer domain (bounded or unbounded)
    Integer,
    /// Real number domain (continuous)
    Real,
    /// Complex number domain
    Complex,
    /// Boolean domain {true, false}
    Boolean,
    /// Custom domain type
    Custom(String),
}

impl DomainInfo {
    /// Create a new domain with given name and type.
    pub fn new(name: impl Into<String>, domain_type: DomainType) -> Self {
        Self {
            name: name.into(),
            size: None,
            domain_type,
            constraints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a finite domain with a specific size.
    pub fn finite(name: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            size: Some(size),
            domain_type: DomainType::Finite,
            constraints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create an integer domain.
    pub fn integer(name: impl Into<String>) -> Self {
        Self::new(name, DomainType::Integer)
    }

    /// Create a real number domain.
    pub fn real(name: impl Into<String>) -> Self {
        Self::new(name, DomainType::Real)
    }

    /// Create a boolean domain.
    pub fn boolean(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            size: Some(2),
            domain_type: DomainType::Boolean,
            constraints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a constraint to this domain.
    pub fn with_constraint(mut self, constraint: impl Into<String>) -> Self {
        self.constraints.push(constraint.into());
        self
    }

    /// Add metadata to this domain.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the size for this domain.
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// Check if this domain is compatible with another.
    ///
    /// Two domains are compatible if:
    /// - They have the same type category
    /// - Finite domains have compatible sizes
    /// - Constraints don't conflict
    pub fn is_compatible_with(&self, other: &DomainInfo) -> bool {
        // Type must match
        if self.domain_type != other.domain_type {
            return false;
        }

        // For finite domains, sizes must be compatible
        if let (Some(size1), Some(size2)) = (self.size, other.size) {
            if size1 != size2 {
                return false;
            }
        }

        // Check for constraint conflicts (simple check - can be extended)
        for constraint in &self.constraints {
            if let Some(negated) = constraint.strip_prefix("not_") {
                if other.constraints.contains(&negated.to_string()) {
                    return false;
                }
            }
        }

        // Check the reverse direction
        for constraint in &other.constraints {
            if let Some(negated) = constraint.strip_prefix("not_") {
                if self.constraints.contains(&negated.to_string()) {
                    return false;
                }
            }
        }

        true
    }

    /// Check if this domain can be cast to another domain.
    ///
    /// Casting rules:
    /// - Boolean -> Integer -> Real
    /// - Finite -> Integer (if size fits)
    /// - Same type is always compatible
    pub fn can_cast_to(&self, target: &DomainInfo) -> bool {
        if self == target {
            return true;
        }

        match (&self.domain_type, &target.domain_type) {
            // Boolean can cast to anything numeric
            (DomainType::Boolean, DomainType::Integer | DomainType::Real) => true,

            // Integer can cast to Real
            (DomainType::Integer, DomainType::Real) => true,

            // Finite can cast to Integer if small enough
            (DomainType::Finite, DomainType::Integer) => {
                if let Some(size) = self.size {
                    size <= i32::MAX as usize
                } else {
                    false
                }
            }

            // Same type is compatible
            (a, b) if a == b => self.is_compatible_with(target),

            _ => false,
        }
    }
}

/// Registry for managing domain information.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DomainRegistry {
    domains: HashMap<String, DomainInfo>,
}

impl DomainRegistry {
    /// Create a new empty domain registry.
    pub fn new() -> Self {
        Self {
            domains: HashMap::new(),
        }
    }

    /// Create a registry with standard built-in domains.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();

        // Add standard domains
        let _ = registry.register(DomainInfo::boolean("Bool"));
        let _ = registry.register(DomainInfo::integer("Int"));
        let _ = registry.register(DomainInfo::real("Real"));
        let _ = registry.register(
            DomainInfo::integer("Nat")
                .with_constraint("non_negative")
                .with_metadata("min", "0"),
        );
        let _ = registry.register(
            DomainInfo::real("Probability")
                .with_constraint("bounded")
                .with_metadata("min", "0.0")
                .with_metadata("max", "1.0"),
        );

        registry
    }

    /// Register a new domain.
    pub fn register(&mut self, domain: DomainInfo) -> Result<(), IrError> {
        if self.domains.contains_key(&domain.name) {
            return Err(IrError::DomainAlreadyExists {
                name: domain.name.clone(),
            });
        }
        self.domains.insert(domain.name.clone(), domain);
        Ok(())
    }

    /// Register a domain, overwriting if it exists.
    pub fn register_or_replace(&mut self, domain: DomainInfo) {
        self.domains.insert(domain.name.clone(), domain);
    }

    /// Get domain information by name.
    pub fn get(&self, name: &str) -> Option<&DomainInfo> {
        self.domains.get(name)
    }

    /// Check if a domain exists.
    pub fn contains(&self, name: &str) -> bool {
        self.domains.contains_key(name)
    }

    /// Validate that a domain exists.
    pub fn validate_domain(&self, name: &str) -> Result<&DomainInfo, IrError> {
        self.get(name).ok_or_else(|| IrError::DomainNotFound {
            name: name.to_string(),
        })
    }

    /// Check if two domains are compatible.
    pub fn are_compatible(&self, domain1: &str, domain2: &str) -> Result<bool, IrError> {
        let d1 = self.validate_domain(domain1)?;
        let d2 = self.validate_domain(domain2)?;
        Ok(d1.is_compatible_with(d2))
    }

    /// Check if domain1 can be cast to domain2.
    pub fn can_cast(&self, from: &str, to: &str) -> Result<bool, IrError> {
        let d1 = self.validate_domain(from)?;
        let d2 = self.validate_domain(to)?;
        Ok(d1.can_cast_to(d2))
    }

    /// Get all registered domain names.
    pub fn domain_names(&self) -> Vec<String> {
        self.domains.keys().cloned().collect()
    }

    /// Get number of registered domains.
    pub fn len(&self) -> usize {
        self.domains.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.domains.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_info_creation() {
        let domain = DomainInfo::finite("Color", 3);
        assert_eq!(domain.name, "Color");
        assert_eq!(domain.size, Some(3));
        assert_eq!(domain.domain_type, DomainType::Finite);
    }

    #[test]
    fn test_domain_compatibility() {
        let int1 = DomainInfo::integer("Int1");
        let int2 = DomainInfo::integer("Int2");
        assert!(int1.is_compatible_with(&int2));

        let int = DomainInfo::integer("Int");
        let real = DomainInfo::real("Real");
        assert!(!int.is_compatible_with(&real));
    }

    #[test]
    fn test_domain_casting() {
        let bool_dom = DomainInfo::boolean("Bool");
        let int_dom = DomainInfo::integer("Int");
        let real_dom = DomainInfo::real("Real");

        assert!(bool_dom.can_cast_to(&int_dom));
        assert!(bool_dom.can_cast_to(&real_dom));
        assert!(int_dom.can_cast_to(&real_dom));
        assert!(!real_dom.can_cast_to(&int_dom));
    }

    #[test]
    fn test_finite_domain_size_compatibility() {
        let d1 = DomainInfo::finite("D1", 5);
        let d2 = DomainInfo::finite("D2", 5);
        let d3 = DomainInfo::finite("D3", 10);

        assert!(d1.is_compatible_with(&d2));
        assert!(!d1.is_compatible_with(&d3));
    }

    #[test]
    fn test_domain_constraints() {
        let positive = DomainInfo::integer("Positive").with_constraint("positive");
        let negative = DomainInfo::integer("Negative").with_constraint("not_positive");

        assert!(!positive.is_compatible_with(&negative));
    }

    #[test]
    fn test_domain_registry() {
        let mut registry = DomainRegistry::new();
        let domain = DomainInfo::finite("Color", 3);

        registry.register(domain.clone()).unwrap();
        assert!(registry.contains("Color"));
        assert_eq!(registry.get("Color"), Some(&domain));
    }

    #[test]
    fn test_builtin_domains() {
        let registry = DomainRegistry::with_builtins();

        assert!(registry.contains("Bool"));
        assert!(registry.contains("Int"));
        assert!(registry.contains("Real"));
        assert!(registry.contains("Nat"));
        assert!(registry.contains("Probability"));

        let prob = registry.get("Probability").unwrap();
        assert_eq!(prob.metadata.get("min"), Some(&"0.0".to_string()));
        assert_eq!(prob.metadata.get("max"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_registry_compatibility_check() {
        let mut registry = DomainRegistry::new();
        registry.register(DomainInfo::integer("Int1")).unwrap();
        registry.register(DomainInfo::integer("Int2")).unwrap();
        registry.register(DomainInfo::real("Real")).unwrap();

        assert!(registry.are_compatible("Int1", "Int2").unwrap());
        assert!(!registry.are_compatible("Int1", "Real").unwrap());
    }

    #[test]
    fn test_registry_casting() {
        let registry = DomainRegistry::with_builtins();

        assert!(registry.can_cast("Bool", "Int").unwrap());
        assert!(registry.can_cast("Bool", "Real").unwrap());
        assert!(registry.can_cast("Int", "Real").unwrap());
        assert!(!registry.can_cast("Real", "Int").unwrap());
    }

    #[test]
    fn test_domain_metadata() {
        let domain = DomainInfo::real("Temperature")
            .with_metadata("unit", "celsius")
            .with_metadata("min", "-273.15");

        assert_eq!(domain.metadata.get("unit"), Some(&"celsius".to_string()));
        assert_eq!(domain.metadata.get("min"), Some(&"-273.15".to_string()));
    }
}
