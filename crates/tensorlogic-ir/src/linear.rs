//! Linear type system for resource management in TensorLogic.
//!
//! This module implements linear types (also known as affine types), where values
//! must be used exactly once. This is crucial for:
//!
//! - **Memory management**: Ensuring tensors are properly deallocated
//! - **In-place operations**: Tracking when tensors can be mutated safely
//! - **Resource tracking**: Managing GPU memory, file handles, etc.
//! - **Side effect control**: Ensuring operations execute in the correct order
//!
//! # Examples
//!
//! ```
//! use tensorlogic_ir::linear::{LinearType, Multiplicity, LinearContext};
//!
//! // Linear type: must be used exactly once
//! let tensor_handle = LinearType::linear("TensorHandle");
//!
//! // Unrestricted type: can be used multiple times
//! let int_type = LinearType::unrestricted("Int");
//!
//! // Check multiplicity constraints
//! let mut ctx = LinearContext::new();
//! ctx.bind("x", tensor_handle);
//! assert!(ctx.is_linear("x"));
//! ```
//!
//! # Multiplicity System
//!
//! - **Linear (1)**: Must be used exactly once
//! - **Affine (0..1)**: Must be used at most once
//! - **Relevant (1..)**: Must be used at least once
//! - **Unrestricted (0..)**: Can be used any number of times

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::{IrError, ParametricType};

/// Multiplicity: how many times a value can be used.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Multiplicity {
    /// Linear: must be used exactly once (1)
    Linear,
    /// Affine: must be used at most once (0..1)
    Affine,
    /// Relevant: must be used at least once (1..)
    Relevant,
    /// Unrestricted: can be used any number of times (0..)
    Unrestricted,
}

impl Multiplicity {
    /// Check if a value with this multiplicity can be used n times
    pub fn allows(&self, n: usize) -> bool {
        match self {
            Multiplicity::Linear => n == 1,
            Multiplicity::Affine => n <= 1,
            Multiplicity::Relevant => n >= 1,
            Multiplicity::Unrestricted => true,
        }
    }

    /// Check if this is linear (exactly once)
    pub fn is_linear(&self) -> bool {
        matches!(self, Multiplicity::Linear)
    }

    /// Check if this is unrestricted (any number of times)
    pub fn is_unrestricted(&self) -> bool {
        matches!(self, Multiplicity::Unrestricted)
    }

    /// Combine multiplicities (for products/tuples)
    pub fn combine(&self, other: &Multiplicity) -> Multiplicity {
        match (self, other) {
            (Multiplicity::Unrestricted, Multiplicity::Unrestricted) => Multiplicity::Unrestricted,
            (Multiplicity::Linear, Multiplicity::Linear) => Multiplicity::Linear,
            (Multiplicity::Affine, Multiplicity::Affine) => Multiplicity::Affine,
            (Multiplicity::Relevant, Multiplicity::Relevant) => Multiplicity::Relevant,
            // Most restrictive wins
            (Multiplicity::Linear, _) | (_, Multiplicity::Linear) => Multiplicity::Linear,
            (Multiplicity::Affine, _) | (_, Multiplicity::Affine) => Multiplicity::Affine,
            (Multiplicity::Relevant, _) | (_, Multiplicity::Relevant) => Multiplicity::Relevant,
        }
    }

    /// Join multiplicities (for sums/unions)
    pub fn join(&self, other: &Multiplicity) -> Multiplicity {
        match (self, other) {
            (Multiplicity::Unrestricted, _) | (_, Multiplicity::Unrestricted) => {
                Multiplicity::Unrestricted
            }
            (Multiplicity::Relevant, _) | (_, Multiplicity::Relevant) => Multiplicity::Relevant,
            (Multiplicity::Affine, Multiplicity::Affine) => Multiplicity::Affine,
            (Multiplicity::Linear, Multiplicity::Linear) => Multiplicity::Linear,
            (Multiplicity::Affine, Multiplicity::Linear)
            | (Multiplicity::Linear, Multiplicity::Affine) => Multiplicity::Affine,
        }
    }
}

impl fmt::Display for Multiplicity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Multiplicity::Linear => write!(f, "1"),
            Multiplicity::Affine => write!(f, "0..1"),
            Multiplicity::Relevant => write!(f, "1.."),
            Multiplicity::Unrestricted => write!(f, "0.."),
        }
    }
}

/// Linear type: type with multiplicity constraints.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinearType {
    /// Base type
    pub base_type: ParametricType,
    /// Multiplicity constraint
    pub multiplicity: Multiplicity,
}

impl LinearType {
    /// Create a new linear type
    pub fn new(base_type: ParametricType, multiplicity: Multiplicity) -> Self {
        LinearType {
            base_type,
            multiplicity,
        }
    }

    /// Create a linear type (must be used exactly once)
    pub fn linear(type_name: impl Into<String>) -> Self {
        LinearType {
            base_type: ParametricType::concrete(type_name),
            multiplicity: Multiplicity::Linear,
        }
    }

    /// Create an affine type (at most once)
    pub fn affine(type_name: impl Into<String>) -> Self {
        LinearType {
            base_type: ParametricType::concrete(type_name),
            multiplicity: Multiplicity::Affine,
        }
    }

    /// Create a relevant type (at least once)
    pub fn relevant(type_name: impl Into<String>) -> Self {
        LinearType {
            base_type: ParametricType::concrete(type_name),
            multiplicity: Multiplicity::Relevant,
        }
    }

    /// Create an unrestricted type (any number of times)
    pub fn unrestricted(type_name: impl Into<String>) -> Self {
        LinearType {
            base_type: ParametricType::concrete(type_name),
            multiplicity: Multiplicity::Unrestricted,
        }
    }

    /// Check if this is a linear type
    pub fn is_linear(&self) -> bool {
        self.multiplicity.is_linear()
    }

    /// Check if this is unrestricted
    pub fn is_unrestricted(&self) -> bool {
        self.multiplicity.is_unrestricted()
    }

    /// Convert to unrestricted (for copying)
    pub fn make_unrestricted(mut self) -> Self {
        self.multiplicity = Multiplicity::Unrestricted;
        self
    }

    /// Convert to linear
    pub fn make_linear(mut self) -> Self {
        self.multiplicity = Multiplicity::Linear;
        self
    }
}

impl fmt::Display for LinearType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}<{}>", self.base_type, self.multiplicity)
    }
}

/// Usage tracking for linear variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Usage {
    /// Variable name
    pub var_name: String,
    /// Number of times used
    pub use_count: usize,
    /// Expected multiplicity
    pub expected: Multiplicity,
}

impl Usage {
    pub fn new(var_name: impl Into<String>, expected: Multiplicity) -> Self {
        Usage {
            var_name: var_name.into(),
            use_count: 0,
            expected,
        }
    }

    /// Record a use
    pub fn record_use(&mut self) {
        self.use_count += 1;
    }

    /// Check if usage is valid
    pub fn is_valid(&self) -> bool {
        self.expected.allows(self.use_count)
    }

    /// Get error message if invalid
    pub fn error_message(&self) -> Option<String> {
        if self.is_valid() {
            None
        } else {
            Some(format!(
                "Variable '{}' has multiplicity {} but was used {} times",
                self.var_name, self.expected, self.use_count
            ))
        }
    }
}

/// Linear typing context for tracking variable usage.
#[derive(Clone, Debug, Default)]
pub struct LinearContext {
    /// Variable bindings with their linear types
    bindings: HashMap<String, LinearType>,
    /// Usage tracking
    usage: HashMap<String, Usage>,
    /// Consumed variables (used and invalidated)
    consumed: HashSet<String>,
}

impl LinearContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a variable with a linear type
    pub fn bind(&mut self, name: impl Into<String>, linear_type: LinearType) {
        let name = name.into();
        let multiplicity = linear_type.multiplicity.clone();
        self.bindings.insert(name.clone(), linear_type);
        self.usage
            .insert(name.clone(), Usage::new(name, multiplicity));
    }

    /// Use a variable (increment use count)
    pub fn use_var(&mut self, name: &str) -> Result<(), IrError> {
        if self.consumed.contains(name) {
            return Err(IrError::LinearityViolation(format!(
                "Variable '{}' already consumed",
                name
            )));
        }

        if let Some(usage) = self.usage.get_mut(name) {
            usage.record_use();

            // If linear or affine, mark as consumed after use
            #[allow(clippy::collapsible_if)]
            if usage.expected.is_linear() || matches!(usage.expected, Multiplicity::Affine) {
                if usage.use_count >= 1 {
                    self.consumed.insert(name.to_string());
                }
            }

            Ok(())
        } else {
            Err(IrError::UnboundVariable {
                var: name.to_string(),
            })
        }
    }

    /// Check if a variable is linear
    pub fn is_linear(&self, name: &str) -> bool {
        self.bindings
            .get(name)
            .map(|t| t.is_linear())
            .unwrap_or(false)
    }

    /// Check if a variable is consumed
    pub fn is_consumed(&self, name: &str) -> bool {
        self.consumed.contains(name)
    }

    /// Get the linear type of a variable
    pub fn get_type(&self, name: &str) -> Option<&LinearType> {
        self.bindings.get(name)
    }

    /// Validate all usage counts at the end of scope
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        for usage in self.usage.values() {
            if let Some(err) = usage.error_message() {
                errors.push(err);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get all unused variables with relevant or linear multiplicity
    pub fn get_unused_required(&self) -> Vec<String> {
        self.usage
            .values()
            .filter(|u| {
                u.use_count == 0
                    && (u.expected.is_linear() || matches!(u.expected, Multiplicity::Relevant))
            })
            .map(|u| u.var_name.clone())
            .collect()
    }

    /// Merge two contexts (for branching control flow)
    pub fn merge(&self, other: &LinearContext) -> Result<LinearContext, IrError> {
        let mut merged = LinearContext::new();

        // Merge bindings
        for (name, typ) in &self.bindings {
            if let Some(other_typ) = other.bindings.get(name) {
                if typ != other_typ {
                    return Err(IrError::InconsistentTypes {
                        var: name.clone(),
                        type1: format!("{}", typ),
                        type2: format!("{}", other_typ),
                    });
                }
                merged.bindings.insert(name.clone(), typ.clone());
            }
        }

        // Merge usage: both branches must satisfy constraints
        for (name, usage1) in &self.usage {
            if let Some(usage2) = other.usage.get(name) {
                // For linear/relevant: both branches must use the variable
                // For affine/unrestricted: either branch can use it
                let min_uses = usage1.use_count.min(usage2.use_count);
                let max_uses = usage1.use_count.max(usage2.use_count);

                let use_count = match usage1.expected {
                    Multiplicity::Linear | Multiplicity::Relevant => {
                        // Both branches must use it
                        if usage1.use_count == 0 || usage2.use_count == 0 {
                            return Err(IrError::LinearityViolation(format!(
                                "Variable '{}' must be used in both branches",
                                name
                            )));
                        }
                        min_uses
                    }
                    Multiplicity::Affine | Multiplicity::Unrestricted => max_uses,
                };

                let mut merged_usage = Usage::new(name, usage1.expected.clone());
                merged_usage.use_count = use_count;
                merged.usage.insert(name.clone(), merged_usage);
            }
        }

        // Merge consumed sets
        merged.consumed = self
            .consumed
            .intersection(&other.consumed)
            .cloned()
            .collect();

        Ok(merged)
    }

    /// Split context for parallel use (e.g., function arguments)
    pub fn split(&mut self, vars: &[String]) -> Result<LinearContext, IrError> {
        let mut split_ctx = LinearContext::new();

        for var in vars {
            if let Some(typ) = self.bindings.remove(var) {
                if typ.is_linear() {
                    // Linear types can be moved
                    split_ctx.bind(var, typ);
                    self.consumed.insert(var.clone());
                } else if typ.is_unrestricted() {
                    // Unrestricted types can be copied
                    split_ctx.bind(var, typ.clone());
                    self.bindings.insert(var.clone(), typ);
                } else {
                    return Err(IrError::LinearityViolation(format!(
                        "Cannot split variable '{}' with multiplicity {}",
                        var, typ.multiplicity
                    )));
                }
            }
        }

        Ok(split_ctx)
    }
}

/// Linearity checker for expressions.
#[derive(Clone, Debug)]
pub struct LinearityChecker {
    context: LinearContext,
    errors: Vec<String>,
}

impl LinearityChecker {
    pub fn new() -> Self {
        LinearityChecker {
            context: LinearContext::new(),
            errors: Vec::new(),
        }
    }

    /// Add a linear variable binding
    pub fn bind(&mut self, name: impl Into<String>, linear_type: LinearType) {
        self.context.bind(name, linear_type);
    }

    /// Record a variable use
    pub fn use_var(&mut self, name: &str) {
        if let Err(e) = self.context.use_var(name) {
            self.errors.push(format!("{}", e));
        }
    }

    /// Check if all linearity constraints are satisfied
    pub fn check(&self) -> Result<(), Vec<String>> {
        let mut all_errors = self.errors.clone();

        if let Err(mut usage_errors) = self.context.validate() {
            all_errors.append(&mut usage_errors);
        }

        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(all_errors)
        }
    }

    /// Get the current context
    pub fn context(&self) -> &LinearContext {
        &self.context
    }

    /// Get a mutable reference to the context
    pub fn context_mut(&mut self) -> &mut LinearContext {
        &mut self.context
    }
}

impl Default for LinearityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Capability: describes what operations are allowed on a linear resource.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Read access
    Read,
    /// Write access
    Write,
    /// Execute access
    Execute,
    /// Own (can deallocate)
    Own,
}

/// Linear resource with capabilities.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearResource {
    /// Resource type
    pub resource_type: LinearType,
    /// Allowed capabilities
    pub capabilities: HashSet<Capability>,
}

impl LinearResource {
    pub fn new(resource_type: LinearType, capabilities: HashSet<Capability>) -> Self {
        LinearResource {
            resource_type,
            capabilities,
        }
    }

    /// Check if a capability is allowed
    pub fn has_capability(&self, cap: &Capability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Create a read-only resource
    pub fn read_only(resource_type: LinearType) -> Self {
        let mut caps = HashSet::new();
        caps.insert(Capability::Read);
        LinearResource::new(resource_type, caps)
    }

    /// Create a read-write resource
    pub fn read_write(resource_type: LinearType) -> Self {
        let mut caps = HashSet::new();
        caps.insert(Capability::Read);
        caps.insert(Capability::Write);
        LinearResource::new(resource_type, caps)
    }

    /// Create an owned resource (full access)
    pub fn owned(resource_type: LinearType) -> Self {
        let mut caps = HashSet::new();
        caps.insert(Capability::Read);
        caps.insert(Capability::Write);
        caps.insert(Capability::Own);
        LinearResource::new(resource_type, caps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiplicity_allows() {
        assert!(Multiplicity::Linear.allows(1));
        assert!(!Multiplicity::Linear.allows(0));
        assert!(!Multiplicity::Linear.allows(2));

        assert!(Multiplicity::Affine.allows(0));
        assert!(Multiplicity::Affine.allows(1));
        assert!(!Multiplicity::Affine.allows(2));

        assert!(!Multiplicity::Relevant.allows(0));
        assert!(Multiplicity::Relevant.allows(1));
        assert!(Multiplicity::Relevant.allows(2));

        assert!(Multiplicity::Unrestricted.allows(0));
        assert!(Multiplicity::Unrestricted.allows(1));
        assert!(Multiplicity::Unrestricted.allows(100));
    }

    #[test]
    fn test_multiplicity_combine() {
        assert_eq!(
            Multiplicity::Linear.combine(&Multiplicity::Linear),
            Multiplicity::Linear
        );
        assert_eq!(
            Multiplicity::Unrestricted.combine(&Multiplicity::Unrestricted),
            Multiplicity::Unrestricted
        );
        assert_eq!(
            Multiplicity::Linear.combine(&Multiplicity::Unrestricted),
            Multiplicity::Linear
        );
    }

    #[test]
    fn test_linear_type_creation() {
        let linear_tensor = LinearType::linear("Tensor");
        assert!(linear_tensor.is_linear());
        assert!(!linear_tensor.is_unrestricted());

        let unrestricted_int = LinearType::unrestricted("Int");
        assert!(!unrestricted_int.is_linear());
        assert!(unrestricted_int.is_unrestricted());
    }

    #[test]
    fn test_linear_context_basic() {
        let mut ctx = LinearContext::new();
        let tensor_type = LinearType::linear("Tensor");

        ctx.bind("x", tensor_type);
        assert!(ctx.is_linear("x"));
        assert!(!ctx.is_consumed("x"));

        // Use once - should be OK
        assert!(ctx.use_var("x").is_ok());
        assert!(ctx.is_consumed("x"));

        // Use again - should fail
        assert!(ctx.use_var("x").is_err());
    }

    #[test]
    fn test_affine_type_usage() {
        let mut ctx = LinearContext::new();
        let affine_type = LinearType::affine("File");

        ctx.bind("f", affine_type);

        // Using 0 times is OK for affine
        assert!(ctx.validate().is_ok());

        // Using 1 time is OK
        assert!(ctx.use_var("f").is_ok());
        assert!(ctx.validate().is_ok());
    }

    #[test]
    fn test_relevant_type_usage() {
        let mut ctx = LinearContext::new();
        let relevant_type = LinearType::relevant("Resource");

        ctx.bind("r", relevant_type);

        // Not using is NOT OK for relevant
        assert!(ctx.validate().is_err());

        let mut ctx2 = LinearContext::new();
        ctx2.bind("r", LinearType::relevant("Resource"));
        assert!(ctx2.use_var("r").is_ok());
        assert!(ctx2.use_var("r").is_ok()); // Can use multiple times
        assert!(ctx2.validate().is_ok());
    }

    #[test]
    fn test_unrestricted_type_usage() {
        let mut ctx = LinearContext::new();
        let unrestricted_type = LinearType::unrestricted("Int");

        ctx.bind("x", unrestricted_type);

        // Can use any number of times
        for _ in 0..10 {
            assert!(ctx.use_var("x").is_ok());
        }
        assert!(ctx.validate().is_ok());
    }

    #[test]
    fn test_linearity_checker() {
        let mut checker = LinearityChecker::new();

        checker.bind("x", LinearType::linear("Tensor"));
        checker.bind("y", LinearType::unrestricted("Int"));

        // Use x once
        checker.use_var("x");

        // Use y multiple times
        checker.use_var("y");
        checker.use_var("y");

        // Should pass
        assert!(checker.check().is_ok());
    }

    #[test]
    fn test_linearity_checker_violation() {
        let mut checker = LinearityChecker::new();

        checker.bind("x", LinearType::linear("Tensor"));

        // Use x twice - should fail
        checker.use_var("x");
        checker.use_var("x");

        assert!(checker.check().is_err());
    }

    #[test]
    fn test_context_merge() {
        let mut ctx1 = LinearContext::new();
        let mut ctx2 = LinearContext::new();

        // Both contexts have same unrestricted binding
        ctx1.bind("x", LinearType::unrestricted("Int"));
        ctx2.bind("x", LinearType::unrestricted("Int"));

        // Use in different amounts
        ctx1.use_var("x").unwrap();
        ctx2.use_var("x").unwrap();
        ctx2.use_var("x").unwrap();

        // Merge should succeed
        let merged = ctx1.merge(&ctx2);
        assert!(merged.is_ok());
    }

    #[test]
    fn test_linear_resource_capabilities() {
        let tensor_type = LinearType::linear("Tensor");
        let resource = LinearResource::read_only(tensor_type);

        assert!(resource.has_capability(&Capability::Read));
        assert!(!resource.has_capability(&Capability::Write));
        assert!(!resource.has_capability(&Capability::Own));
    }

    #[test]
    fn test_get_unused_required() {
        let mut ctx = LinearContext::new();

        ctx.bind("x", LinearType::linear("Tensor"));
        ctx.bind("y", LinearType::unrestricted("Int"));
        ctx.bind("z", LinearType::relevant("Resource"));

        // x and z are required but unused
        let unused = ctx.get_unused_required();
        assert_eq!(unused.len(), 2);
        assert!(unused.contains(&"x".to_string()));
        assert!(unused.contains(&"z".to_string()));
    }

    #[test]
    fn test_context_split() {
        let mut ctx = LinearContext::new();

        ctx.bind("x", LinearType::linear("Tensor"));
        ctx.bind("y", LinearType::unrestricted("Int"));

        // Split off x
        let split = ctx.split(&["x".to_string()]);
        assert!(split.is_ok());

        let split_ctx = split.unwrap();
        assert!(split_ctx.get_type("x").is_some());
        assert!(ctx.is_consumed("x"));

        // y should still be in both
        assert!(ctx.get_type("y").is_some());
        assert!(!ctx.is_consumed("y"));
    }

    #[test]
    fn test_linear_type_display() {
        let linear = LinearType::linear("Tensor");
        assert_eq!(linear.to_string(), "Tensor<1>");

        let affine = LinearType::affine("File");
        assert_eq!(affine.to_string(), "File<0..1>");

        let unrestricted = LinearType::unrestricted("Int");
        assert_eq!(unrestricted.to_string(), "Int<0..>");
    }
}
