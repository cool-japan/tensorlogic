//! Effect system for tracking computational effects in TensorLogic expressions.
//!
//! This module provides an effect system that tracks various kinds of computational
//! effects in logical expressions and tensor operations, enabling:
//!
//! - **Effect tracking**: Know which operations have side effects
//! - **Differentiability**: Track which operations support gradient computation
//! - **Probabilistic reasoning**: Distinguish deterministic from stochastic operations
//! - **Memory safety**: Track memory access patterns
//! - **Effect polymorphism**: Functions parametric over effects
//!
//! # Examples
//!
//! ```
//! use tensorlogic_ir::effect_system::{Effect, EffectSet, ComputationalEffect};
//!
//! // Pure computation (no side effects)
//! let pure_effect = EffectSet::pure();
//! assert!(pure_effect.is_pure());
//!
//! // Differentiable operation
//! let diff_effect = EffectSet::new()
//!     .with(Effect::Computational(ComputationalEffect::Pure))
//!     .with(Effect::Differentiable);
//!
//! // Combine effects
//! let combined = pure_effect.union(&diff_effect);
//! assert!(combined.contains(&Effect::Differentiable));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

use crate::IrError;

/// Computational purity effects.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputationalEffect {
    /// Pure computation (no side effects, referentially transparent)
    Pure,
    /// Impure computation (may have side effects)
    Impure,
    /// I/O operations (reading/writing external state)
    IO,
}

impl fmt::Display for ComputationalEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputationalEffect::Pure => write!(f, "Pure"),
            ComputationalEffect::Impure => write!(f, "Impure"),
            ComputationalEffect::IO => write!(f, "IO"),
        }
    }
}

/// Memory access effects.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryEffect {
    /// Read-only memory access
    ReadOnly,
    /// Read-write memory access
    ReadWrite,
    /// Memory allocation
    Allocating,
}

impl fmt::Display for MemoryEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryEffect::ReadOnly => write!(f, "ReadOnly"),
            MemoryEffect::ReadWrite => write!(f, "ReadWrite"),
            MemoryEffect::Allocating => write!(f, "Allocating"),
        }
    }
}

/// Probabilistic effects.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProbabilisticEffect {
    /// Deterministic computation (same inputs → same outputs)
    Deterministic,
    /// Stochastic computation (involves randomness)
    Stochastic,
}

impl fmt::Display for ProbabilisticEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbabilisticEffect::Deterministic => write!(f, "Deterministic"),
            ProbabilisticEffect::Stochastic => write!(f, "Stochastic"),
        }
    }
}

/// Individual effect kinds.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Effect {
    /// Computational purity
    Computational(ComputationalEffect),
    /// Memory access pattern
    Memory(MemoryEffect),
    /// Probabilistic behavior
    Probabilistic(ProbabilisticEffect),
    /// Supports automatic differentiation
    Differentiable,
    /// Does not support automatic differentiation
    NonDifferentiable,
    /// Asynchronous computation
    Async,
    /// Parallel computation
    Parallel,
    /// Custom user-defined effect
    Custom(String),
}

impl Effect {
    /// Check if this effect is pure
    pub fn is_pure(&self) -> bool {
        matches!(self, Effect::Computational(ComputationalEffect::Pure))
    }

    /// Check if this effect is impure
    pub fn is_impure(&self) -> bool {
        matches!(
            self,
            Effect::Computational(ComputationalEffect::Impure | ComputationalEffect::IO)
        )
    }

    /// Check if this effect is differentiable
    pub fn is_differentiable(&self) -> bool {
        matches!(self, Effect::Differentiable)
    }

    /// Check if this effect is stochastic
    pub fn is_stochastic(&self) -> bool {
        matches!(self, Effect::Probabilistic(ProbabilisticEffect::Stochastic))
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Effect::Computational(e) => write!(f, "{}", e),
            Effect::Memory(e) => write!(f, "{}", e),
            Effect::Probabilistic(e) => write!(f, "{}", e),
            Effect::Differentiable => write!(f, "Diff"),
            Effect::NonDifferentiable => write!(f, "NonDiff"),
            Effect::Async => write!(f, "Async"),
            Effect::Parallel => write!(f, "Parallel"),
            Effect::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Set of effects for an expression or operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectSet {
    effects: HashSet<Effect>,
}

impl EffectSet {
    /// Create an empty effect set
    pub fn new() -> Self {
        EffectSet {
            effects: HashSet::new(),
        }
    }

    /// Create a pure effect set (pure + deterministic + differentiable)
    pub fn pure() -> Self {
        let mut effects = HashSet::new();
        effects.insert(Effect::Computational(ComputationalEffect::Pure));
        effects.insert(Effect::Probabilistic(ProbabilisticEffect::Deterministic));
        effects.insert(Effect::Memory(MemoryEffect::ReadOnly));
        EffectSet { effects }
    }

    /// Create an impure effect set
    pub fn impure() -> Self {
        let mut effects = HashSet::new();
        effects.insert(Effect::Computational(ComputationalEffect::Impure));
        EffectSet { effects }
    }

    /// Create a differentiable effect set
    pub fn differentiable() -> Self {
        let mut effects = HashSet::new();
        effects.insert(Effect::Differentiable);
        EffectSet { effects }
    }

    /// Create a stochastic effect set
    pub fn stochastic() -> Self {
        let mut effects = HashSet::new();
        effects.insert(Effect::Probabilistic(ProbabilisticEffect::Stochastic));
        EffectSet { effects }
    }

    /// Add an effect to this set
    pub fn with(mut self, effect: Effect) -> Self {
        self.effects.insert(effect);
        self
    }

    /// Add multiple effects
    pub fn with_all(mut self, effects: impl IntoIterator<Item = Effect>) -> Self {
        self.effects.extend(effects);
        self
    }

    /// Check if this set contains a specific effect
    pub fn contains(&self, effect: &Effect) -> bool {
        self.effects.contains(effect)
    }

    /// Check if this effect set is pure (contains Pure computational effect and no impure effects)
    pub fn is_pure(&self) -> bool {
        // Either empty or contains Pure and no impure effects
        if self.effects.is_empty() {
            return true;
        }

        let has_pure = self
            .effects
            .iter()
            .any(|e| matches!(e, Effect::Computational(ComputationalEffect::Pure)));

        let has_impure = self.effects.iter().any(|e| {
            matches!(
                e,
                Effect::Computational(ComputationalEffect::Impure | ComputationalEffect::IO)
            )
        });

        has_pure && !has_impure
    }

    /// Check if this effect set is impure
    pub fn is_impure(&self) -> bool {
        self.effects.iter().any(|e| e.is_impure())
    }

    /// Check if this effect set is differentiable
    pub fn is_differentiable(&self) -> bool {
        self.effects.iter().any(|e| e.is_differentiable())
            && !self
                .effects
                .iter()
                .any(|e| matches!(e, Effect::NonDifferentiable))
    }

    /// Check if this effect set is stochastic
    pub fn is_stochastic(&self) -> bool {
        self.effects.iter().any(|e| e.is_stochastic())
    }

    /// Get all effects in this set
    pub fn effects(&self) -> impl Iterator<Item = &Effect> {
        self.effects.iter()
    }

    /// Union of two effect sets
    pub fn union(&self, other: &EffectSet) -> EffectSet {
        let mut effects = self.effects.clone();
        effects.extend(other.effects.iter().cloned());
        EffectSet { effects }
    }

    /// Intersection of two effect sets
    pub fn intersection(&self, other: &EffectSet) -> EffectSet {
        let effects = self.effects.intersection(&other.effects).cloned().collect();
        EffectSet { effects }
    }

    /// Check if this effect set is a subset of another (subtyping)
    pub fn is_subset_of(&self, other: &EffectSet) -> bool {
        self.effects.is_subset(&other.effects)
    }

    /// Check if two effect sets are compatible
    pub fn is_compatible_with(&self, other: &EffectSet) -> bool {
        // Compatible if no conflicting effects
        !self.has_conflicts_with(other)
    }

    /// Check if there are conflicting effects
    fn has_conflicts_with(&self, other: &EffectSet) -> bool {
        // Pure and Impure conflict
        if (self.contains(&Effect::Computational(ComputationalEffect::Pure)) && other.is_impure())
            || (other.contains(&Effect::Computational(ComputationalEffect::Pure))
                && self.is_impure())
        {
            return true;
        }

        // Differentiable and NonDifferentiable conflict
        if (self.contains(&Effect::Differentiable) && other.contains(&Effect::NonDifferentiable))
            || (other.contains(&Effect::Differentiable)
                && self.contains(&Effect::NonDifferentiable))
        {
            return true;
        }

        false
    }

    /// Number of effects in this set
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Check if effect set is empty
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

impl Default for EffectSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.effects.is_empty() {
            return write!(f, "{{}}");
        }

        write!(f, "{{")?;
        let mut first = true;
        for effect in &self.effects {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}", effect)?;
            first = false;
        }
        write!(f, "}}")
    }
}

/// Effect variable for effect polymorphism
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectVar(pub String);

impl EffectVar {
    pub fn new(name: impl Into<String>) -> Self {
        EffectVar(name.into())
    }
}

impl fmt::Display for EffectVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ε{}", self.0)
    }
}

/// Effect scheme for effect polymorphism (analogous to type schemes)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectScheme {
    /// Concrete effect set
    Concrete(EffectSet),
    /// Effect variable (for polymorphism)
    Variable(EffectVar),
    /// Union of effect schemes
    Union(Box<EffectScheme>, Box<EffectScheme>),
}

impl EffectScheme {
    /// Create a concrete effect scheme
    pub fn concrete(effects: EffectSet) -> Self {
        EffectScheme::Concrete(effects)
    }

    /// Create an effect variable
    pub fn variable(name: impl Into<String>) -> Self {
        EffectScheme::Variable(EffectVar::new(name))
    }

    /// Create a union of two effect schemes
    pub fn union(e1: EffectScheme, e2: EffectScheme) -> Self {
        EffectScheme::Union(Box::new(e1), Box::new(e2))
    }

    /// Substitute effect variables with concrete effect sets
    pub fn substitute(&self, subst: &EffectSubstitution) -> EffectScheme {
        match self {
            EffectScheme::Concrete(effects) => EffectScheme::Concrete(effects.clone()),
            EffectScheme::Variable(var) => {
                if let Some(effects) = subst.get(var) {
                    EffectScheme::Concrete(effects.clone())
                } else {
                    EffectScheme::Variable(var.clone())
                }
            }
            EffectScheme::Union(e1, e2) => {
                let s1 = e1.substitute(subst);
                let s2 = e2.substitute(subst);
                EffectScheme::union(s1, s2)
            }
        }
    }

    /// Evaluate to a concrete effect set (if possible)
    pub fn evaluate(&self, subst: &EffectSubstitution) -> Result<EffectSet, IrError> {
        match self {
            EffectScheme::Concrete(effects) => Ok(effects.clone()),
            EffectScheme::Variable(var) => {
                subst
                    .get(var)
                    .cloned()
                    .ok_or_else(|| IrError::UnboundVariable {
                        var: format!("effect variable {}", var),
                    })
            }
            EffectScheme::Union(e1, e2) => {
                let effects1 = e1.evaluate(subst)?;
                let effects2 = e2.evaluate(subst)?;
                Ok(effects1.union(&effects2))
            }
        }
    }
}

impl fmt::Display for EffectScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EffectScheme::Concrete(effects) => write!(f, "{}", effects),
            EffectScheme::Variable(var) => write!(f, "{}", var),
            EffectScheme::Union(e1, e2) => write!(f, "({} ∪ {})", e1, e2),
        }
    }
}

/// Substitution mapping effect variables to effect sets
pub type EffectSubstitution = std::collections::HashMap<EffectVar, EffectSet>;

/// Effect annotation for expressions
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectAnnotation {
    /// The effect scheme for this expression
    pub scheme: EffectScheme,
    /// Optional description
    pub description: Option<String>,
}

impl EffectAnnotation {
    pub fn new(scheme: EffectScheme) -> Self {
        EffectAnnotation {
            scheme,
            description: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Create a pure effect annotation
    pub fn pure() -> Self {
        EffectAnnotation::new(EffectScheme::concrete(EffectSet::pure()))
    }

    /// Create a differentiable effect annotation
    pub fn differentiable() -> Self {
        EffectAnnotation::new(EffectScheme::concrete(EffectSet::differentiable()))
    }
}

/// Infer effects for common operations
pub fn infer_operation_effects(op_name: &str) -> EffectSet {
    match op_name {
        // Pure logical operations
        "and" | "or" | "not" | "implies" => EffectSet::pure().with(Effect::Differentiable),

        // Arithmetic operations (pure and differentiable)
        "add" | "subtract" | "multiply" | "divide" => {
            EffectSet::pure().with(Effect::Differentiable)
        }

        // Quantifiers (pure but may not be differentiable)
        "exists" | "forall" => EffectSet::pure(),

        // Comparisons (pure but not differentiable)
        "equal" | "less_than" | "greater_than" => EffectSet::pure().with(Effect::NonDifferentiable),

        // Sampling operations (stochastic)
        "sample" | "random" => EffectSet::stochastic().with(Effect::NonDifferentiable),

        // I/O operations
        "read" | "write" => EffectSet::new()
            .with(Effect::Computational(ComputationalEffect::IO))
            .with(Effect::Memory(MemoryEffect::ReadWrite)),

        // Default: conservative (impure, non-differentiable)
        _ => EffectSet::impure().with(Effect::NonDifferentiable),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_creation() {
        let pure = Effect::Computational(ComputationalEffect::Pure);
        assert!(pure.is_pure());
        assert!(!pure.is_impure());

        let impure = Effect::Computational(ComputationalEffect::Impure);
        assert!(!impure.is_pure());
        assert!(impure.is_impure());

        let diff = Effect::Differentiable;
        assert!(diff.is_differentiable());
    }

    #[test]
    fn test_effect_set_pure() {
        let pure_set = EffectSet::pure();
        assert!(pure_set.is_pure());
        assert!(!pure_set.is_impure());
        assert!(pure_set.contains(&Effect::Computational(ComputationalEffect::Pure)));
    }

    #[test]
    fn test_effect_set_differentiable() {
        let diff_set = EffectSet::differentiable();
        assert!(diff_set.is_differentiable());
        assert!(diff_set.contains(&Effect::Differentiable));
    }

    #[test]
    fn test_effect_set_union() {
        let pure = EffectSet::pure();
        let diff = EffectSet::differentiable();
        let combined = pure.union(&diff);

        assert!(combined.contains(&Effect::Computational(ComputationalEffect::Pure)));
        assert!(combined.contains(&Effect::Differentiable));
    }

    #[test]
    fn test_effect_set_intersection() {
        let set1 = EffectSet::pure().with(Effect::Differentiable);
        let set2 = EffectSet::differentiable();
        let intersection = set1.intersection(&set2);

        assert!(intersection.contains(&Effect::Differentiable));
        assert!(!intersection.contains(&Effect::Computational(ComputationalEffect::Pure)));
    }

    #[test]
    fn test_effect_set_subset() {
        let small = EffectSet::pure();
        let large = EffectSet::pure().with(Effect::Differentiable);

        assert!(small.is_subset_of(&large));
        assert!(!large.is_subset_of(&small));
    }

    #[test]
    fn test_effect_conflicts() {
        let pure = EffectSet::pure();
        let impure = EffectSet::impure();

        assert!(!pure.is_compatible_with(&impure));
        assert!(!impure.is_compatible_with(&pure));
    }

    #[test]
    fn test_effect_scheme_concrete() {
        let scheme = EffectScheme::concrete(EffectSet::pure());
        let subst = EffectSubstitution::new();
        let effects = scheme.evaluate(&subst).unwrap();

        assert!(effects.is_pure());
    }

    #[test]
    fn test_effect_scheme_variable() {
        let var = EffectVar::new("e1");
        let scheme = EffectScheme::Variable(var.clone());

        let mut subst = EffectSubstitution::new();
        subst.insert(var, EffectSet::pure());

        let effects = scheme.evaluate(&subst).unwrap();
        assert!(effects.is_pure());
    }

    #[test]
    fn test_effect_scheme_union() {
        let scheme1 = EffectScheme::concrete(EffectSet::pure());
        let scheme2 = EffectScheme::concrete(EffectSet::differentiable());
        let union_scheme = EffectScheme::union(scheme1, scheme2);

        let subst = EffectSubstitution::new();
        let effects = union_scheme.evaluate(&subst).unwrap();

        assert!(effects.is_pure());
        assert!(effects.is_differentiable());
    }

    #[test]
    fn test_effect_annotation() {
        let annotation = EffectAnnotation::pure().with_description("Pure computation");

        assert_eq!(annotation.description.as_deref(), Some("Pure computation"));
    }

    #[test]
    fn test_infer_operation_effects() {
        let and_effects = infer_operation_effects("and");
        assert!(and_effects.is_pure());
        assert!(and_effects.is_differentiable());

        let sample_effects = infer_operation_effects("sample");
        assert!(sample_effects.is_stochastic());

        let io_effects = infer_operation_effects("read");
        assert!(io_effects.is_impure());
    }

    #[test]
    fn test_effect_set_stochastic() {
        let stochastic = EffectSet::stochastic();
        assert!(stochastic.is_stochastic());
        assert!(stochastic.contains(&Effect::Probabilistic(ProbabilisticEffect::Stochastic)));
    }

    #[test]
    fn test_memory_effects() {
        let read_only = Effect::Memory(MemoryEffect::ReadOnly);
        let read_write = Effect::Memory(MemoryEffect::ReadWrite);

        let set1 = EffectSet::new().with(read_only);
        let set2 = EffectSet::new().with(read_write);

        assert_ne!(set1, set2);
    }

    #[test]
    fn test_custom_effect() {
        let custom = Effect::Custom("GPUCompute".to_string());
        let set = EffectSet::new().with(custom.clone());

        assert!(set.contains(&custom));
    }

    #[test]
    fn test_effect_display() {
        let pure = Effect::Computational(ComputationalEffect::Pure);
        assert_eq!(pure.to_string(), "Pure");

        let diff = Effect::Differentiable;
        assert_eq!(diff.to_string(), "Diff");

        let custom = Effect::Custom("MyEffect".to_string());
        assert_eq!(custom.to_string(), "MyEffect");
    }

    #[test]
    fn test_effect_set_display() {
        let set = EffectSet::pure().with(Effect::Differentiable);
        let display = set.to_string();

        assert!(display.contains("Pure") || display.contains("Diff"));
        assert!(display.starts_with('{'));
        assert!(display.ends_with('}'));
    }

    #[test]
    fn test_effect_var_display() {
        let var = EffectVar::new("1");
        assert_eq!(var.to_string(), "ε1");
    }

    #[test]
    fn test_non_differentiable_conflicts() {
        let diff = EffectSet::new().with(Effect::Differentiable);
        let non_diff = EffectSet::new().with(Effect::NonDifferentiable);

        assert!(!diff.is_compatible_with(&non_diff));
    }
}
