//! Predicate composition system for defining predicates in terms of others.
//!
//! This module provides a system for composing predicates from other predicates,
//! enabling:
//! - Macro-like predicate expansion
//! - Predicate templates with parameters
//! - Derived predicates based on existing ones
//! - Complex predicate definitions through composition

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tensorlogic_ir::TLExpr;

use crate::error::AdapterError;

/// A composable predicate definition that can be expanded.
///
/// Composite predicates are defined in terms of other predicates and can
/// include parameters that are substituted during expansion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompositePredicate {
    /// Name of this composite predicate
    pub name: String,
    /// Parameter names (e.g., ["x", "y"])
    pub parameters: Vec<String>,
    /// The body expression defining this predicate
    pub body: PredicateBody,
    /// Optional description
    pub description: Option<String>,
}

/// The body of a composite predicate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PredicateBody {
    /// A TensorLogic expression
    Expression(Box<TLExpr>),
    /// Reference to another composite predicate
    Reference { name: String, args: Vec<String> },
    /// Conjunction of multiple predicates
    And(Vec<PredicateBody>),
    /// Disjunction of multiple predicates
    Or(Vec<PredicateBody>),
    /// Negation
    Not(Box<PredicateBody>),
}

/// A registry of composite predicates for lookup and expansion.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CompositeRegistry {
    predicates: HashMap<String, CompositePredicate>,
}

impl CompositePredicate {
    /// Creates a new composite predicate.
    pub fn new(name: impl Into<String>, parameters: Vec<String>, body: PredicateBody) -> Self {
        CompositePredicate {
            name: name.into(),
            parameters,
            body,
            description: None,
        }
    }

    /// Sets the description for this composite predicate.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Returns the arity (number of parameters) of this predicate.
    pub fn arity(&self) -> usize {
        self.parameters.len()
    }

    /// Validates that this composite predicate is well-formed.
    pub fn validate(&self) -> Result<(), AdapterError> {
        // Check that all parameters are unique
        let mut seen = std::collections::HashSet::new();
        for param in &self.parameters {
            if !seen.insert(param) {
                return Err(AdapterError::InvalidParametricType(format!(
                    "Duplicate parameter '{}' in predicate '{}'",
                    param, self.name
                )));
            }
        }

        // Validate the body
        self.body.validate(&self.parameters)?;

        Ok(())
    }

    /// Expands this composite predicate with the given arguments.
    ///
    /// Substitutes all parameter occurrences in the body with the provided arguments.
    pub fn expand(&self, args: &[String]) -> Result<PredicateBody, AdapterError> {
        if args.len() != self.parameters.len() {
            return Err(AdapterError::ArityMismatch {
                name: self.name.clone(),
                expected: self.parameters.len(),
                found: args.len(),
            });
        }

        // Create substitution map
        let mut substitutions = HashMap::new();
        for (param, arg) in self.parameters.iter().zip(args.iter()) {
            substitutions.insert(param.clone(), arg.clone());
        }

        self.body.substitute(&substitutions)
    }
}

impl PredicateBody {
    /// Validates that this predicate body is well-formed.
    fn validate(&self, parameters: &[String]) -> Result<(), AdapterError> {
        match self {
            PredicateBody::Expression(_) => Ok(()), // TLExpr validation handled elsewhere
            PredicateBody::Reference { args, .. } => {
                // Check that all args reference valid parameters
                for arg in args {
                    if !parameters.contains(arg) && !arg.starts_with('_') {
                        return Err(AdapterError::UnboundVariable(arg.clone()));
                    }
                }
                Ok(())
            }
            PredicateBody::And(bodies) | PredicateBody::Or(bodies) => {
                for body in bodies {
                    body.validate(parameters)?;
                }
                Ok(())
            }
            PredicateBody::Not(body) => body.validate(parameters),
        }
    }

    /// Substitutes parameters with concrete arguments.
    fn substitute(
        &self,
        substitutions: &HashMap<String, String>,
    ) -> Result<PredicateBody, AdapterError> {
        match self {
            PredicateBody::Expression(expr) => {
                // For now, return as-is. Full expression substitution would require
                // walking the TLExpr tree and replacing variable names.
                Ok(PredicateBody::Expression(expr.clone()))
            }
            PredicateBody::Reference { name, args } => {
                let new_args = args
                    .iter()
                    .map(|arg| {
                        substitutions
                            .get(arg)
                            .cloned()
                            .unwrap_or_else(|| arg.clone())
                    })
                    .collect();
                Ok(PredicateBody::Reference {
                    name: name.clone(),
                    args: new_args,
                })
            }
            PredicateBody::And(bodies) => {
                let new_bodies: Result<Vec<_>, _> =
                    bodies.iter().map(|b| b.substitute(substitutions)).collect();
                Ok(PredicateBody::And(new_bodies?))
            }
            PredicateBody::Or(bodies) => {
                let new_bodies: Result<Vec<_>, _> =
                    bodies.iter().map(|b| b.substitute(substitutions)).collect();
                Ok(PredicateBody::Or(new_bodies?))
            }
            PredicateBody::Not(body) => Ok(PredicateBody::Not(Box::new(
                body.substitute(substitutions)?,
            ))),
        }
    }
}

impl CompositeRegistry {
    /// Creates a new empty composite registry.
    pub fn new() -> Self {
        CompositeRegistry::default()
    }

    /// Registers a composite predicate.
    pub fn register(&mut self, predicate: CompositePredicate) -> Result<(), AdapterError> {
        predicate.validate()?;
        self.predicates.insert(predicate.name.clone(), predicate);
        Ok(())
    }

    /// Gets a composite predicate by name.
    pub fn get(&self, name: &str) -> Option<&CompositePredicate> {
        self.predicates.get(name)
    }

    /// Checks if a predicate is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.predicates.contains_key(name)
    }

    /// Expands a composite predicate with the given arguments.
    pub fn expand(&self, name: &str, args: &[String]) -> Result<PredicateBody, AdapterError> {
        let predicate = self
            .get(name)
            .ok_or_else(|| AdapterError::PredicateNotFound(name.to_string()))?;

        predicate.expand(args)
    }

    /// Returns the number of registered composite predicates.
    pub fn len(&self) -> usize {
        self.predicates.len()
    }

    /// Checks if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }

    /// Lists all registered predicate names.
    pub fn list_predicates(&self) -> Vec<String> {
        self.predicates.keys().cloned().collect()
    }
}

/// A template for creating multiple similar predicates.
///
/// Templates allow defining patterns for predicates that can be instantiated
/// with different domains or properties.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredicateTemplate {
    /// Template name
    pub name: String,
    /// Type parameters (e.g., ["T", "U"])
    pub type_params: Vec<String>,
    /// Value parameters (e.g., ["relation"])
    pub value_params: Vec<String>,
    /// The body defining how to construct the predicate
    pub body: PredicateBody,
}

impl PredicateTemplate {
    /// Creates a new predicate template.
    pub fn new(
        name: impl Into<String>,
        type_params: Vec<String>,
        value_params: Vec<String>,
        body: PredicateBody,
    ) -> Self {
        PredicateTemplate {
            name: name.into(),
            type_params,
            value_params,
            body,
        }
    }

    /// Instantiates this template with concrete types and values.
    pub fn instantiate(
        &self,
        type_args: &[String],
        value_args: &[String],
    ) -> Result<CompositePredicate, AdapterError> {
        if type_args.len() != self.type_params.len() {
            return Err(AdapterError::ArityMismatch {
                name: format!("{}[type params]", self.name),
                expected: self.type_params.len(),
                found: type_args.len(),
            });
        }

        if value_args.len() != self.value_params.len() {
            return Err(AdapterError::ArityMismatch {
                name: format!("{}[value params]", self.name),
                expected: self.value_params.len(),
                found: value_args.len(),
            });
        }

        // Create substitution map
        let mut substitutions = HashMap::new();
        for (param, arg) in self.type_params.iter().zip(type_args.iter()) {
            substitutions.insert(param.clone(), arg.clone());
        }
        for (param, arg) in self.value_params.iter().zip(value_args.iter()) {
            substitutions.insert(param.clone(), arg.clone());
        }

        // Generate instance name
        let instance_name = format!("{}<{}>", self.name, type_args.join(", "));

        // Substitute in body
        let instance_body = self.body.substitute(&substitutions)?;

        Ok(CompositePredicate {
            name: instance_name,
            parameters: value_args.to_vec(),
            body: instance_body,
            description: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_predicate_creation() {
        let pred = CompositePredicate::new(
            "friend",
            vec!["x".to_string(), "y".to_string()],
            PredicateBody::Reference {
                name: "knows".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
        );

        assert_eq!(pred.name, "friend");
        assert_eq!(pred.arity(), 2);
    }

    #[test]
    fn test_composite_predicate_validation() {
        let valid = CompositePredicate::new(
            "test",
            vec!["x".to_string(), "y".to_string()],
            PredicateBody::Reference {
                name: "knows".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
        );
        assert!(valid.validate().is_ok());

        let invalid = CompositePredicate::new(
            "test",
            vec!["x".to_string(), "x".to_string()], // Duplicate parameter
            PredicateBody::Reference {
                name: "knows".to_string(),
                args: vec!["x".to_string()],
            },
        );
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_composite_registry() {
        let mut registry = CompositeRegistry::new();

        let pred = CompositePredicate::new(
            "friend",
            vec!["x".to_string(), "y".to_string()],
            PredicateBody::Reference {
                name: "knows".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
        );

        registry.register(pred).unwrap();
        assert!(registry.contains("friend"));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_predicate_expansion() {
        let pred = CompositePredicate::new(
            "friend",
            vec!["x".to_string(), "y".to_string()],
            PredicateBody::Reference {
                name: "knows".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
        );

        let expanded = pred
            .expand(&["alice".to_string(), "bob".to_string()])
            .unwrap();

        match expanded {
            PredicateBody::Reference { name, args } => {
                assert_eq!(name, "knows");
                assert_eq!(args, vec!["alice".to_string(), "bob".to_string()]);
            }
            _ => panic!("Expected Reference"),
        }
    }

    #[test]
    fn test_predicate_template() {
        let template = PredicateTemplate::new(
            "related",
            vec!["T".to_string()],
            vec!["x".to_string(), "y".to_string()],
            PredicateBody::Reference {
                name: "connected".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
        );

        let instance = template
            .instantiate(&["Person".to_string()], &["a".to_string(), "b".to_string()])
            .unwrap();

        assert_eq!(instance.name, "related<Person>");
        assert_eq!(instance.parameters, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_composite_and() {
        let body = PredicateBody::And(vec![
            PredicateBody::Reference {
                name: "knows".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
            PredicateBody::Reference {
                name: "trusts".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
        ]);

        let pred = CompositePredicate::new("friend", vec!["x".to_string(), "y".to_string()], body);

        assert!(pred.validate().is_ok());
    }

    #[test]
    fn test_composite_or() {
        let body = PredicateBody::Or(vec![
            PredicateBody::Reference {
                name: "colleague".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
            PredicateBody::Reference {
                name: "friend".to_string(),
                args: vec!["x".to_string(), "y".to_string()],
            },
        ]);

        let pred =
            CompositePredicate::new("connected", vec!["x".to_string(), "y".to_string()], body);

        assert!(pred.validate().is_ok());
    }

    #[test]
    fn test_composite_not() {
        let body = PredicateBody::Not(Box::new(PredicateBody::Reference {
            name: "enemy".to_string(),
            args: vec!["x".to_string(), "y".to_string()],
        }));

        let pred =
            CompositePredicate::new("not_enemy", vec!["x".to_string(), "y".to_string()], body);

        assert!(pred.validate().is_ok());
    }
}
