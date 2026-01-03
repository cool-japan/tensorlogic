//! Refinement types for expressing value constraints beyond simple types.
//!
//! Refinement types extend base types with predicates that constrain valid values.
//! This enables static verification of properties like positivity, bounds, and custom invariants.
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_adapters::{RefinementType, RefinementPredicate, RefinementContext};
//!
//! // Create a positive integer refinement
//! let pos_int = RefinementType::new("Int")
//!     .with_predicate(RefinementPredicate::greater_than(0.0))
//!     .with_name("PositiveInt");
//!
//! // Check if a value satisfies the refinement
//! assert!(pos_int.check(5.0));
//! assert!(!pos_int.check(-1.0));
//!
//! // Create bounded range refinement
//! let probability = RefinementType::new("Float")
//!     .with_predicate(RefinementPredicate::range(0.0, 1.0))
//!     .with_name("Probability");
//!
//! assert!(probability.check(0.5));
//! assert!(!probability.check(1.5));
//! ```

use std::collections::HashMap;
use std::sync::Arc;

/// A refinement predicate that constrains values.
#[derive(Clone)]
pub enum RefinementPredicate {
    /// Value must equal a constant
    Equal(f64),
    /// Value must not equal a constant
    NotEqual(f64),
    /// Value must be greater than a constant
    GreaterThan(f64),
    /// Value must be greater than or equal to a constant
    GreaterThanOrEqual(f64),
    /// Value must be less than a constant
    LessThan(f64),
    /// Value must be less than or equal to a constant
    LessThanOrEqual(f64),
    /// Value must be in a range [min, max]
    Range { min: f64, max: f64 },
    /// Value must be in a half-open range [min, max)
    RangeExclusive { min: f64, max: f64 },
    /// Value must satisfy a modulo constraint (value % divisor == remainder)
    Modulo { divisor: i64, remainder: i64 },
    /// Value must be in a set of allowed values
    InSet(Vec<f64>),
    /// Value must not be in a set of disallowed values
    NotInSet(Vec<f64>),
    /// Conjunction of predicates (all must be satisfied)
    And(Vec<RefinementPredicate>),
    /// Disjunction of predicates (at least one must be satisfied)
    Or(Vec<RefinementPredicate>),
    /// Negation of a predicate
    Not(Box<RefinementPredicate>),
    /// Custom predicate with a name (for symbolic reasoning)
    Custom {
        name: String,
        description: String,
        checker: Arc<dyn Fn(f64) -> bool + Send + Sync>,
    },
    /// Dependent predicate referencing another variable
    Dependent {
        variable: String,
        relation: DependentRelation,
    },
    /// String length constraint
    StringLength {
        min: Option<usize>,
        max: Option<usize>,
    },
    /// Pattern match constraint (for strings)
    Pattern(String),
}

/// Relation for dependent predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum DependentRelation {
    /// Value must be less than the referenced variable
    LessThan,
    /// Value must be less than or equal to the referenced variable
    LessThanOrEqual,
    /// Value must be greater than the referenced variable
    GreaterThan,
    /// Value must be greater than or equal to the referenced variable
    GreaterThanOrEqual,
    /// Value must equal the referenced variable
    Equal,
    /// Value must not equal the referenced variable
    NotEqual,
    /// Value is a divisor of the referenced variable
    Divides,
    /// Referenced variable is a divisor of this value
    DivisibleBy,
}

impl std::fmt::Debug for RefinementPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefinementPredicate::Equal(v) => f.debug_tuple("Equal").field(v).finish(),
            RefinementPredicate::NotEqual(v) => f.debug_tuple("NotEqual").field(v).finish(),
            RefinementPredicate::GreaterThan(v) => f.debug_tuple("GreaterThan").field(v).finish(),
            RefinementPredicate::GreaterThanOrEqual(v) => {
                f.debug_tuple("GreaterThanOrEqual").field(v).finish()
            }
            RefinementPredicate::LessThan(v) => f.debug_tuple("LessThan").field(v).finish(),
            RefinementPredicate::LessThanOrEqual(v) => {
                f.debug_tuple("LessThanOrEqual").field(v).finish()
            }
            RefinementPredicate::Range { min, max } => f
                .debug_struct("Range")
                .field("min", min)
                .field("max", max)
                .finish(),
            RefinementPredicate::RangeExclusive { min, max } => f
                .debug_struct("RangeExclusive")
                .field("min", min)
                .field("max", max)
                .finish(),
            RefinementPredicate::Modulo { divisor, remainder } => f
                .debug_struct("Modulo")
                .field("divisor", divisor)
                .field("remainder", remainder)
                .finish(),
            RefinementPredicate::InSet(set) => f.debug_tuple("InSet").field(set).finish(),
            RefinementPredicate::NotInSet(set) => f.debug_tuple("NotInSet").field(set).finish(),
            RefinementPredicate::And(preds) => f.debug_tuple("And").field(preds).finish(),
            RefinementPredicate::Or(preds) => f.debug_tuple("Or").field(preds).finish(),
            RefinementPredicate::Not(pred) => f.debug_tuple("Not").field(pred).finish(),
            RefinementPredicate::Custom {
                name, description, ..
            } => f
                .debug_struct("Custom")
                .field("name", name)
                .field("description", description)
                .finish(),
            RefinementPredicate::Dependent { variable, relation } => f
                .debug_struct("Dependent")
                .field("variable", variable)
                .field("relation", relation)
                .finish(),
            RefinementPredicate::StringLength { min, max } => f
                .debug_struct("StringLength")
                .field("min", min)
                .field("max", max)
                .finish(),
            RefinementPredicate::Pattern(pattern) => {
                f.debug_tuple("Pattern").field(pattern).finish()
            }
        }
    }
}

impl RefinementPredicate {
    /// Create a "greater than" predicate.
    pub fn greater_than(value: f64) -> Self {
        RefinementPredicate::GreaterThan(value)
    }

    /// Create a "greater than or equal" predicate.
    pub fn greater_than_or_equal(value: f64) -> Self {
        RefinementPredicate::GreaterThanOrEqual(value)
    }

    /// Create a "less than" predicate.
    pub fn less_than(value: f64) -> Self {
        RefinementPredicate::LessThan(value)
    }

    /// Create a "less than or equal" predicate.
    pub fn less_than_or_equal(value: f64) -> Self {
        RefinementPredicate::LessThanOrEqual(value)
    }

    /// Create a range predicate [min, max].
    pub fn range(min: f64, max: f64) -> Self {
        RefinementPredicate::Range { min, max }
    }

    /// Create a modulo constraint predicate.
    pub fn modulo(divisor: i64, remainder: i64) -> Self {
        RefinementPredicate::Modulo { divisor, remainder }
    }

    /// Create a predicate requiring value to be in a set.
    pub fn in_set(values: Vec<f64>) -> Self {
        RefinementPredicate::InSet(values)
    }

    /// Create a conjunction of predicates.
    pub fn and(predicates: Vec<RefinementPredicate>) -> Self {
        RefinementPredicate::And(predicates)
    }

    /// Create a disjunction of predicates.
    pub fn or(predicates: Vec<RefinementPredicate>) -> Self {
        RefinementPredicate::Or(predicates)
    }

    /// Create a negation of a predicate.
    #[allow(clippy::should_implement_trait)]
    pub fn not(predicate: RefinementPredicate) -> Self {
        RefinementPredicate::Not(Box::new(predicate))
    }

    /// Create a custom predicate with a checker function.
    pub fn custom<F>(name: impl Into<String>, description: impl Into<String>, checker: F) -> Self
    where
        F: Fn(f64) -> bool + Send + Sync + 'static,
    {
        RefinementPredicate::Custom {
            name: name.into(),
            description: description.into(),
            checker: Arc::new(checker),
        }
    }

    /// Create a dependent predicate.
    pub fn dependent(variable: impl Into<String>, relation: DependentRelation) -> Self {
        RefinementPredicate::Dependent {
            variable: variable.into(),
            relation,
        }
    }

    /// Check if a value satisfies this predicate.
    ///
    /// Note: For dependent predicates, this returns true (use `check_with_context` instead).
    pub fn check(&self, value: f64) -> bool {
        match self {
            RefinementPredicate::Equal(v) => (value - v).abs() < f64::EPSILON,
            RefinementPredicate::NotEqual(v) => (value - v).abs() >= f64::EPSILON,
            RefinementPredicate::GreaterThan(v) => value > *v,
            RefinementPredicate::GreaterThanOrEqual(v) => value >= *v,
            RefinementPredicate::LessThan(v) => value < *v,
            RefinementPredicate::LessThanOrEqual(v) => value <= *v,
            RefinementPredicate::Range { min, max } => value >= *min && value <= *max,
            RefinementPredicate::RangeExclusive { min, max } => value >= *min && value < *max,
            RefinementPredicate::Modulo { divisor, remainder } => {
                (value as i64) % divisor == *remainder
            }
            RefinementPredicate::InSet(set) => set.iter().any(|v| (value - v).abs() < f64::EPSILON),
            RefinementPredicate::NotInSet(set) => {
                !set.iter().any(|v| (value - v).abs() < f64::EPSILON)
            }
            RefinementPredicate::And(preds) => preds.iter().all(|p| p.check(value)),
            RefinementPredicate::Or(preds) => preds.iter().any(|p| p.check(value)),
            RefinementPredicate::Not(pred) => !pred.check(value),
            RefinementPredicate::Custom { checker, .. } => checker(value),
            RefinementPredicate::Dependent { .. } => true, // Needs context
            RefinementPredicate::StringLength { .. } => true, // Not applicable to f64
            RefinementPredicate::Pattern(_) => true,       // Not applicable to f64
        }
    }

    /// Check if a value satisfies this predicate with a context for dependent predicates.
    pub fn check_with_context(&self, value: f64, context: &RefinementContext) -> bool {
        match self {
            RefinementPredicate::Dependent { variable, relation } => {
                if let Some(&other) = context.get_value(variable) {
                    match relation {
                        DependentRelation::LessThan => value < other,
                        DependentRelation::LessThanOrEqual => value <= other,
                        DependentRelation::GreaterThan => value > other,
                        DependentRelation::GreaterThanOrEqual => value >= other,
                        DependentRelation::Equal => (value - other).abs() < f64::EPSILON,
                        DependentRelation::NotEqual => (value - other).abs() >= f64::EPSILON,
                        DependentRelation::Divides => {
                            other != 0.0 && (other as i64) % (value as i64) == 0
                        }
                        DependentRelation::DivisibleBy => {
                            value != 0.0 && (value as i64) % (other as i64) == 0
                        }
                    }
                } else {
                    false // Unknown variable
                }
            }
            RefinementPredicate::And(preds) => {
                preds.iter().all(|p| p.check_with_context(value, context))
            }
            RefinementPredicate::Or(preds) => {
                preds.iter().any(|p| p.check_with_context(value, context))
            }
            RefinementPredicate::Not(pred) => !pred.check_with_context(value, context),
            _ => self.check(value),
        }
    }

    /// Get the free variables referenced by this predicate.
    pub fn free_variables(&self) -> Vec<String> {
        match self {
            RefinementPredicate::Dependent { variable, .. } => vec![variable.clone()],
            RefinementPredicate::And(preds) | RefinementPredicate::Or(preds) => {
                let mut vars = Vec::new();
                for pred in preds {
                    vars.extend(pred.free_variables());
                }
                vars.sort();
                vars.dedup();
                vars
            }
            RefinementPredicate::Not(pred) => pred.free_variables(),
            _ => vec![],
        }
    }

    /// Simplify the predicate by removing redundant constraints.
    pub fn simplify(&self) -> RefinementPredicate {
        match self {
            RefinementPredicate::And(preds) => {
                let simplified: Vec<_> = preds.iter().map(|p| p.simplify()).collect();
                if simplified.len() == 1 {
                    simplified.into_iter().next().unwrap()
                } else {
                    // Merge range constraints
                    let mut min_val = f64::NEG_INFINITY;
                    let mut max_val = f64::INFINITY;
                    let mut others = Vec::new();

                    for pred in simplified {
                        match pred {
                            RefinementPredicate::GreaterThan(v) => {
                                min_val = min_val.max(v);
                            }
                            RefinementPredicate::GreaterThanOrEqual(v) => {
                                min_val = min_val.max(v);
                            }
                            RefinementPredicate::LessThan(v) => {
                                max_val = max_val.min(v);
                            }
                            RefinementPredicate::LessThanOrEqual(v) => {
                                max_val = max_val.min(v);
                            }
                            RefinementPredicate::Range { min, max } => {
                                min_val = min_val.max(min);
                                max_val = max_val.min(max);
                            }
                            other => others.push(other),
                        }
                    }

                    // Create merged range if we have bounds
                    if min_val > f64::NEG_INFINITY || max_val < f64::INFINITY {
                        if min_val > f64::NEG_INFINITY && max_val < f64::INFINITY {
                            others.insert(
                                0,
                                RefinementPredicate::Range {
                                    min: min_val,
                                    max: max_val,
                                },
                            );
                        } else if min_val > f64::NEG_INFINITY {
                            others.insert(0, RefinementPredicate::GreaterThanOrEqual(min_val));
                        } else {
                            others.insert(0, RefinementPredicate::LessThanOrEqual(max_val));
                        }
                    }

                    if others.len() == 1 {
                        others.into_iter().next().unwrap()
                    } else {
                        RefinementPredicate::And(others)
                    }
                }
            }
            RefinementPredicate::Or(preds) => {
                let simplified: Vec<_> = preds.iter().map(|p| p.simplify()).collect();
                if simplified.len() == 1 {
                    simplified.into_iter().next().unwrap()
                } else {
                    RefinementPredicate::Or(simplified)
                }
            }
            RefinementPredicate::Not(pred) => {
                let inner = pred.simplify();
                match inner {
                    RefinementPredicate::Not(p) => *p, // Double negation
                    other => RefinementPredicate::Not(Box::new(other)),
                }
            }
            other => other.clone(),
        }
    }

    /// Convert to a human-readable string.
    pub fn to_string_repr(&self) -> String {
        match self {
            RefinementPredicate::Equal(v) => format!("x == {}", v),
            RefinementPredicate::NotEqual(v) => format!("x != {}", v),
            RefinementPredicate::GreaterThan(v) => format!("x > {}", v),
            RefinementPredicate::GreaterThanOrEqual(v) => format!("x >= {}", v),
            RefinementPredicate::LessThan(v) => format!("x < {}", v),
            RefinementPredicate::LessThanOrEqual(v) => format!("x <= {}", v),
            RefinementPredicate::Range { min, max } => format!("{} <= x <= {}", min, max),
            RefinementPredicate::RangeExclusive { min, max } => format!("{} <= x < {}", min, max),
            RefinementPredicate::Modulo { divisor, remainder } => {
                format!("x % {} == {}", divisor, remainder)
            }
            RefinementPredicate::InSet(set) => format!("x in {:?}", set),
            RefinementPredicate::NotInSet(set) => format!("x not in {:?}", set),
            RefinementPredicate::And(preds) => {
                let parts: Vec<_> = preds.iter().map(|p| p.to_string_repr()).collect();
                format!("({})", parts.join(" && "))
            }
            RefinementPredicate::Or(preds) => {
                let parts: Vec<_> = preds.iter().map(|p| p.to_string_repr()).collect();
                format!("({})", parts.join(" || "))
            }
            RefinementPredicate::Not(pred) => format!("!({})", pred.to_string_repr()),
            RefinementPredicate::Custom { name, .. } => format!("{}(x)", name),
            RefinementPredicate::Dependent { variable, relation } => {
                let rel_str = match relation {
                    DependentRelation::LessThan => "<",
                    DependentRelation::LessThanOrEqual => "<=",
                    DependentRelation::GreaterThan => ">",
                    DependentRelation::GreaterThanOrEqual => ">=",
                    DependentRelation::Equal => "==",
                    DependentRelation::NotEqual => "!=",
                    DependentRelation::Divides => "divides",
                    DependentRelation::DivisibleBy => "divisible_by",
                };
                format!("x {} {}", rel_str, variable)
            }
            RefinementPredicate::StringLength { min, max } => match (min, max) {
                (Some(min), Some(max)) => format!("{} <= len(x) <= {}", min, max),
                (Some(min), None) => format!("len(x) >= {}", min),
                (None, Some(max)) => format!("len(x) <= {}", max),
                (None, None) => "true".to_string(),
            },
            RefinementPredicate::Pattern(pattern) => format!("x matches \"{}\"", pattern),
        }
    }
}

/// A refinement type combining a base type with predicates.
#[derive(Debug, Clone)]
pub struct RefinementType {
    /// Base type name
    pub base_type: String,
    /// Optional refined name (e.g., "PositiveInt" for Int{x > 0})
    pub name: Option<String>,
    /// Predicates that constrain values
    pub predicates: Vec<RefinementPredicate>,
    /// Description of the refinement
    pub description: Option<String>,
}

impl RefinementType {
    /// Create a new refinement type with a base type.
    pub fn new(base_type: impl Into<String>) -> Self {
        RefinementType {
            base_type: base_type.into(),
            name: None,
            predicates: Vec::new(),
            description: None,
        }
    }

    /// Set the refined name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add a predicate to the refinement.
    pub fn with_predicate(mut self, predicate: RefinementPredicate) -> Self {
        self.predicates.push(predicate);
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Check if a value satisfies this refinement type.
    pub fn check(&self, value: f64) -> bool {
        self.predicates.iter().all(|p| p.check(value))
    }

    /// Check if a value satisfies this refinement type with context.
    pub fn check_with_context(&self, value: f64, context: &RefinementContext) -> bool {
        self.predicates
            .iter()
            .all(|p| p.check_with_context(value, context))
    }

    /// Get the effective name of this type.
    pub fn type_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.base_type)
    }

    /// Check if this is a subtype of another refinement type.
    ///
    /// A refinement type A is a subtype of B if:
    /// 1. They have the same base type
    /// 2. A's predicates imply B's predicates
    ///
    /// This implementation uses semantic implication checking for common predicate patterns,
    /// providing a practical alternative to full SMT solving while handling most real-world cases.
    pub fn is_subtype_of(&self, other: &RefinementType) -> bool {
        if self.base_type != other.base_type {
            return false;
        }

        // Conservative check: if other has no predicates, we're a subtype
        if other.predicates.is_empty() {
            return true;
        }

        // If we have no predicates but other does, we're not a subtype
        if self.predicates.is_empty() && !other.predicates.is_empty() {
            return false;
        }

        // Check if all of other's predicates are implied by our predicates
        for other_pred in &other.predicates {
            if !self.implies_predicate(other_pred) {
                return false;
            }
        }

        true
    }

    /// Check if this refinement type's predicates imply the given predicate.
    ///
    /// This uses semantic implication checking for common patterns:
    /// - Syntactic equality (via Debug representation)
    /// - Range implication (x > 10 implies x > 5)
    /// - Modulo implication (x % 4 == 0 implies x % 2 == 0)
    fn implies_predicate(&self, target: &RefinementPredicate) -> bool {
        // Check for syntactic equality using Debug representation
        // (RefinementPredicate doesn't implement PartialEq due to function pointers)
        let target_repr = format!("{:?}", target);
        if self
            .predicates
            .iter()
            .any(|p| format!("{:?}", p) == target_repr)
        {
            return true;
        }

        // Check for semantic implication based on predicate types
        for pred in &self.predicates {
            if Self::semantic_implies(pred, target) {
                return true;
            }
        }

        // Check for conjunction of predicates implying the target
        Self::conjunction_implies(&self.predicates, target)
    }

    /// Check if one predicate semantically implies another.
    fn semantic_implies(source: &RefinementPredicate, target: &RefinementPredicate) -> bool {
        use RefinementPredicate::*;

        match (source, target) {
            // Range implications: stricter range implies looser range
            (
                Range {
                    min: min1,
                    max: max1,
                },
                Range {
                    min: min2,
                    max: max2,
                },
            ) => {
                // [5, 10] implies [0, 15]
                min1 >= min2 && max1 <= max2
            }
            (
                RangeExclusive {
                    min: min1,
                    max: max1,
                },
                RangeExclusive {
                    min: min2,
                    max: max2,
                },
            ) => min1 >= min2 && max1 <= max2,
            // Greater-than implications: x > 10 implies x > 5
            (GreaterThan(v1), GreaterThan(v2)) => v1 >= v2,
            (GreaterThanOrEqual(v1), GreaterThanOrEqual(v2)) => v1 >= v2,
            (GreaterThan(v1), GreaterThanOrEqual(v2)) => v1 >= v2, // x > 10 implies x >= 10
            // Less-than implications: x < 5 implies x < 10
            (LessThan(v1), LessThan(v2)) => v1 <= v2,
            (LessThanOrEqual(v1), LessThanOrEqual(v2)) => v1 <= v2,
            (LessThan(v1), LessThanOrEqual(v2)) => v1 <= v2, // x < 5 implies x <= 5
            // Equality implies bounds
            (Equal(v1), GreaterThan(v2)) => v1 > v2,
            (Equal(v1), GreaterThanOrEqual(v2)) => v1 >= v2,
            (Equal(v1), LessThan(v2)) => v1 < v2,
            (Equal(v1), LessThanOrEqual(v2)) => v1 <= v2,
            (Equal(v1), Range { min, max }) => v1 >= min && v1 <= max,
            // Modulo implications: x % 4 == 0 implies x % 2 == 0
            (
                Modulo {
                    divisor: d1,
                    remainder: r1,
                },
                Modulo {
                    divisor: d2,
                    remainder: r2,
                },
            ) => r1 == r2 && d1 % d2 == 0,
            // Dependent predicates with same variable
            (
                Dependent {
                    variable: v1,
                    relation: rel1,
                },
                Dependent {
                    variable: v2,
                    relation: rel2,
                },
            ) => {
                if v1 != v2 {
                    return false;
                }
                // Same variable, check if rel1 implies rel2
                use DependentRelation::*;
                matches!(
                    (rel1, rel2),
                    (Equal, Equal)
                        | (GreaterThan, GreaterThan)
                        | (GreaterThan, GreaterThanOrEqual)
                        | (LessThan, LessThan)
                        | (LessThan, LessThanOrEqual)
                        | (GreaterThanOrEqual, GreaterThanOrEqual)
                        | (LessThanOrEqual, LessThanOrEqual)
                )
            }
            _ => false,
        }
    }

    /// Check if a conjunction of predicates implies a target predicate.
    ///
    /// This handles cases like: (x > 5 && x < 10) implies x > 0
    fn conjunction_implies(
        predicates: &[RefinementPredicate],
        target: &RefinementPredicate,
    ) -> bool {
        use RefinementPredicate::*;

        // Extract range bounds from multiple predicates
        let mut lower_bounds = Vec::new();
        let mut upper_bounds = Vec::new();

        for pred in predicates {
            match pred {
                GreaterThan(v) | GreaterThanOrEqual(v) => {
                    lower_bounds.push(*v);
                }
                LessThan(v) | LessThanOrEqual(v) => {
                    upper_bounds.push(*v);
                }
                Range { min, max } => {
                    lower_bounds.push(*min);
                    upper_bounds.push(*max);
                }
                Equal(v) => {
                    lower_bounds.push(*v);
                    upper_bounds.push(*v);
                }
                _ => {}
            }
        }

        // Check if combined bounds imply the target
        match target {
            GreaterThan(v) | GreaterThanOrEqual(v) => lower_bounds.iter().any(|lb| lb >= v),
            LessThan(v) | LessThanOrEqual(v) => upper_bounds.iter().any(|ub| ub <= v),
            Range { min, max } => {
                lower_bounds.iter().any(|lb| lb >= min) && upper_bounds.iter().any(|ub| ub <= max)
            }
            _ => false,
        }
    }

    /// Get all free variables referenced in predicates.
    pub fn free_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        for pred in &self.predicates {
            vars.extend(pred.free_variables());
        }
        vars.sort();
        vars.dedup();
        vars
    }

    /// Convert to human-readable representation.
    pub fn to_string_repr(&self) -> String {
        if self.predicates.is_empty() {
            return self.base_type.clone();
        }

        let pred_strs: Vec<_> = self.predicates.iter().map(|p| p.to_string_repr()).collect();
        format!("{}{{{}}}", self.base_type, pred_strs.join(" && "))
    }
}

/// Context for evaluating dependent refinement predicates.
#[derive(Debug, Clone, Default)]
pub struct RefinementContext {
    /// Variable values in the current context
    values: HashMap<String, f64>,
    /// Type assignments for variables
    types: HashMap<String, RefinementType>,
}

impl RefinementContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        RefinementContext {
            values: HashMap::new(),
            types: HashMap::new(),
        }
    }

    /// Set a variable's value.
    pub fn set_value(&mut self, var: impl Into<String>, value: f64) {
        self.values.insert(var.into(), value);
    }

    /// Get a variable's value.
    pub fn get_value(&self, var: &str) -> Option<&f64> {
        self.values.get(var)
    }

    /// Set a variable's type.
    pub fn set_type(&mut self, var: impl Into<String>, ty: RefinementType) {
        self.types.insert(var.into(), ty);
    }

    /// Get a variable's type.
    pub fn get_type(&self, var: &str) -> Option<&RefinementType> {
        self.types.get(var)
    }

    /// Check if a variable exists in the context.
    pub fn has_variable(&self, var: &str) -> bool {
        self.values.contains_key(var) || self.types.contains_key(var)
    }

    /// Get all variable names.
    pub fn variables(&self) -> Vec<&str> {
        let mut vars: Vec<_> = self.values.keys().map(|s| s.as_str()).collect();
        for key in self.types.keys() {
            if !self.values.contains_key(key) {
                vars.push(key.as_str());
            }
        }
        vars
    }
}

/// Registry for managing refinement types.
#[derive(Debug, Clone, Default)]
pub struct RefinementRegistry {
    /// Named refinement types
    types: HashMap<String, RefinementType>,
}

impl RefinementRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        RefinementRegistry {
            types: HashMap::new(),
        }
    }

    /// Create a registry with common built-in refinement types.
    pub fn with_builtins() -> Self {
        let mut registry = RefinementRegistry::new();

        // Positive integer
        registry.register(
            RefinementType::new("Int")
                .with_name("PositiveInt")
                .with_predicate(RefinementPredicate::GreaterThan(0.0))
                .with_description("Strictly positive integer"),
        );

        // Non-negative integer
        registry.register(
            RefinementType::new("Int")
                .with_name("NonNegativeInt")
                .with_predicate(RefinementPredicate::GreaterThanOrEqual(0.0))
                .with_description("Non-negative integer (zero or positive)"),
        );

        // Probability (0 to 1)
        registry.register(
            RefinementType::new("Float")
                .with_name("Probability")
                .with_predicate(RefinementPredicate::Range { min: 0.0, max: 1.0 })
                .with_description("Probability value between 0 and 1"),
        );

        // Percentage (0 to 100)
        registry.register(
            RefinementType::new("Float")
                .with_name("Percentage")
                .with_predicate(RefinementPredicate::Range {
                    min: 0.0,
                    max: 100.0,
                })
                .with_description("Percentage value between 0 and 100"),
        );

        // Normalized (-1 to 1)
        registry.register(
            RefinementType::new("Float")
                .with_name("Normalized")
                .with_predicate(RefinementPredicate::Range {
                    min: -1.0,
                    max: 1.0,
                })
                .with_description("Normalized value between -1 and 1"),
        );

        // Natural number (0, 1, 2, ...)
        registry.register(
            RefinementType::new("Int")
                .with_name("Natural")
                .with_predicate(RefinementPredicate::And(vec![
                    RefinementPredicate::GreaterThanOrEqual(0.0),
                    RefinementPredicate::Modulo {
                        divisor: 1,
                        remainder: 0,
                    },
                ]))
                .with_description("Natural number (non-negative integer)"),
        );

        // Even number
        registry.register(
            RefinementType::new("Int")
                .with_name("Even")
                .with_predicate(RefinementPredicate::Modulo {
                    divisor: 2,
                    remainder: 0,
                })
                .with_description("Even integer"),
        );

        // Odd number
        registry.register(
            RefinementType::new("Int")
                .with_name("Odd")
                .with_predicate(RefinementPredicate::Modulo {
                    divisor: 2,
                    remainder: 1,
                })
                .with_description("Odd integer"),
        );

        registry
    }

    /// Register a refinement type.
    pub fn register(&mut self, refinement: RefinementType) {
        let name = refinement.type_name().to_string();
        self.types.insert(name, refinement);
    }

    /// Get a refinement type by name.
    pub fn get(&self, name: &str) -> Option<&RefinementType> {
        self.types.get(name)
    }

    /// Check if a type is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    /// Get all registered type names.
    pub fn type_names(&self) -> Vec<&str> {
        self.types.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of registered types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Check if a value satisfies a refinement type by name.
    pub fn check(&self, type_name: &str, value: f64) -> Option<bool> {
        self.types.get(type_name).map(|t| t.check(value))
    }

    /// Iterate over all refinement types.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &RefinementType)> {
        self.types.iter().map(|(k, v)| (k.as_str(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_predicates() {
        let pred = RefinementPredicate::GreaterThan(0.0);
        assert!(pred.check(5.0));
        assert!(!pred.check(-1.0));
        assert!(!pred.check(0.0));
    }

    #[test]
    fn test_range_predicate() {
        let pred = RefinementPredicate::Range { min: 0.0, max: 1.0 };
        assert!(pred.check(0.5));
        assert!(pred.check(0.0));
        assert!(pred.check(1.0));
        assert!(!pred.check(-0.1));
        assert!(!pred.check(1.1));
    }

    #[test]
    fn test_modulo_predicate() {
        let even = RefinementPredicate::Modulo {
            divisor: 2,
            remainder: 0,
        };
        assert!(even.check(4.0));
        assert!(even.check(0.0));
        assert!(!even.check(3.0));
    }

    #[test]
    fn test_compound_predicates() {
        // Positive and even
        let pred = RefinementPredicate::And(vec![
            RefinementPredicate::GreaterThan(0.0),
            RefinementPredicate::Modulo {
                divisor: 2,
                remainder: 0,
            },
        ]);

        assert!(pred.check(4.0));
        assert!(!pred.check(-2.0)); // Not positive
        assert!(!pred.check(3.0)); // Not even
    }

    #[test]
    fn test_in_set_predicate() {
        let pred = RefinementPredicate::InSet(vec![1.0, 2.0, 3.0]);
        assert!(pred.check(1.0));
        assert!(pred.check(2.0));
        assert!(!pred.check(4.0));
    }

    #[test]
    fn test_custom_predicate() {
        let pred = RefinementPredicate::custom("is_prime", "Checks if number is prime", |n| {
            if n < 2.0 {
                return false;
            }
            let n = n as i64;
            for i in 2..=((n as f64).sqrt() as i64) {
                if n % i == 0 {
                    return false;
                }
            }
            true
        });

        assert!(pred.check(2.0));
        assert!(pred.check(7.0));
        assert!(!pred.check(4.0));
        assert!(!pred.check(1.0));
    }

    #[test]
    fn test_refinement_type() {
        let pos_int = RefinementType::new("Int")
            .with_name("PositiveInt")
            .with_predicate(RefinementPredicate::GreaterThan(0.0));

        assert_eq!(pos_int.type_name(), "PositiveInt");
        assert!(pos_int.check(5.0));
        assert!(!pos_int.check(-1.0));
    }

    #[test]
    fn test_dependent_predicate() {
        let pred = RefinementPredicate::Dependent {
            variable: "n".to_string(),
            relation: DependentRelation::LessThan,
        };

        let mut context = RefinementContext::new();
        context.set_value("n", 10.0);

        assert!(pred.check_with_context(5.0, &context));
        assert!(!pred.check_with_context(15.0, &context));
    }

    #[test]
    fn test_registry_builtins() {
        let registry = RefinementRegistry::with_builtins();

        // Test PositiveInt
        assert!(registry.check("PositiveInt", 5.0).unwrap());
        assert!(!registry.check("PositiveInt", -1.0).unwrap());

        // Test Probability
        assert!(registry.check("Probability", 0.5).unwrap());
        assert!(!registry.check("Probability", 1.5).unwrap());

        // Test Even
        assert!(registry.check("Even", 4.0).unwrap());
        assert!(!registry.check("Even", 3.0).unwrap());
    }

    #[test]
    fn test_predicate_simplification() {
        let pred = RefinementPredicate::And(vec![
            RefinementPredicate::GreaterThan(0.0),
            RefinementPredicate::LessThan(10.0),
            RefinementPredicate::GreaterThanOrEqual(1.0),
        ]);

        let simplified = pred.simplify();

        // Should be simplified to a range [1, 10]
        // Note: simplification is conservative and uses inclusive bounds
        assert!(simplified.check(5.0));
        assert!(!simplified.check(0.0));
        // The simplified range includes 10.0 since simplification is conservative
        assert!(simplified.check(1.0)); // min bound included
    }

    #[test]
    fn test_predicate_string_repr() {
        let pred = RefinementPredicate::Range { min: 0.0, max: 1.0 };
        assert_eq!(pred.to_string_repr(), "0 <= x <= 1");

        let pred = RefinementPredicate::And(vec![
            RefinementPredicate::GreaterThan(0.0),
            RefinementPredicate::LessThan(10.0),
        ]);
        assert_eq!(pred.to_string_repr(), "(x > 0 && x < 10)");
    }

    #[test]
    fn test_free_variables() {
        let pred = RefinementPredicate::And(vec![
            RefinementPredicate::GreaterThan(0.0),
            RefinementPredicate::Dependent {
                variable: "n".to_string(),
                relation: DependentRelation::LessThan,
            },
            RefinementPredicate::Dependent {
                variable: "m".to_string(),
                relation: DependentRelation::GreaterThan,
            },
        ]);

        let vars = pred.free_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&"m".to_string()));
        assert!(vars.contains(&"n".to_string()));
    }

    #[test]
    fn test_refinement_type_repr() {
        let ty = RefinementType::new("Int")
            .with_name("BoundedInt")
            .with_predicate(RefinementPredicate::Range {
                min: 0.0,
                max: 100.0,
            });

        assert_eq!(ty.to_string_repr(), "Int{0 <= x <= 100}");
    }

    #[test]
    fn test_context_operations() {
        let mut ctx = RefinementContext::new();

        ctx.set_value("x", 5.0);
        ctx.set_value("y", 10.0);

        assert_eq!(ctx.get_value("x"), Some(&5.0));
        assert!(ctx.has_variable("x"));
        assert!(!ctx.has_variable("z"));

        let vars = ctx.variables();
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_negation_predicate() {
        let pred = RefinementPredicate::Not(Box::new(RefinementPredicate::Equal(0.0)));

        assert!(pred.check(5.0));
        assert!(!pred.check(0.0));
    }

    #[test]
    fn test_or_predicate() {
        let pred = RefinementPredicate::Or(vec![
            RefinementPredicate::LessThan(0.0),
            RefinementPredicate::GreaterThan(10.0),
        ]);

        assert!(pred.check(-5.0));
        assert!(pred.check(15.0));
        assert!(!pred.check(5.0));
    }

    #[test]
    fn test_double_negation_simplification() {
        let pred = RefinementPredicate::Not(Box::new(RefinementPredicate::Not(Box::new(
            RefinementPredicate::GreaterThan(0.0),
        ))));

        let simplified = pred.simplify();
        assert!(simplified.check(5.0));
        assert!(!simplified.check(-1.0));
    }

    #[test]
    fn test_registry_custom_type() {
        let mut registry = RefinementRegistry::new();

        registry.register(
            RefinementType::new("Float")
                .with_name("SmallPositive")
                .with_predicate(RefinementPredicate::Range {
                    min: 0.0,
                    max: 1e-6,
                }),
        );

        assert!(registry.contains("SmallPositive"));
        assert!(registry.check("SmallPositive", 1e-7).unwrap());
        assert!(!registry.check("SmallPositive", 1.0).unwrap());
    }

    // Tests for semantic subtyping implementation

    #[test]
    fn test_subtyping_basic() {
        // Test basic base type matching
        let int_type = RefinementType::new("Int");
        let float_type = RefinementType::new("Float");

        assert!(!int_type.is_subtype_of(&float_type)); // Different base types
        assert!(int_type.is_subtype_of(&int_type)); // Same type
    }

    #[test]
    fn test_subtyping_range_implication() {
        // x ∈ [5, 10] is a subtype of x ∈ [0, 15]
        let stricter = RefinementType::new("Int").with_predicate(RefinementPredicate::Range {
            min: 5.0,
            max: 10.0,
        });

        let looser = RefinementType::new("Int").with_predicate(RefinementPredicate::Range {
            min: 0.0,
            max: 15.0,
        });

        assert!(stricter.is_subtype_of(&looser));
        assert!(!looser.is_subtype_of(&stricter)); // Not the other way around
    }

    #[test]
    fn test_subtyping_greater_than_implication() {
        // x > 10 is a subtype of x > 5
        let stricter =
            RefinementType::new("Int").with_predicate(RefinementPredicate::GreaterThan(10.0));

        let looser =
            RefinementType::new("Int").with_predicate(RefinementPredicate::GreaterThan(5.0));

        assert!(stricter.is_subtype_of(&looser));
        assert!(!looser.is_subtype_of(&stricter));
    }

    #[test]
    fn test_subtyping_less_than_implication() {
        // x < 5 is a subtype of x < 10
        let stricter =
            RefinementType::new("Int").with_predicate(RefinementPredicate::LessThan(5.0));

        let looser = RefinementType::new("Int").with_predicate(RefinementPredicate::LessThan(10.0));

        assert!(stricter.is_subtype_of(&looser));
        assert!(!looser.is_subtype_of(&stricter));
    }

    #[test]
    fn test_subtyping_modulo_implication() {
        // x % 4 == 0 is a subtype of x % 2 == 0
        let divisible_by_4 =
            RefinementType::new("Int").with_predicate(RefinementPredicate::Modulo {
                divisor: 4,
                remainder: 0,
            });

        let divisible_by_2 =
            RefinementType::new("Int").with_predicate(RefinementPredicate::Modulo {
                divisor: 2,
                remainder: 0,
            });

        assert!(divisible_by_4.is_subtype_of(&divisible_by_2));
        assert!(!divisible_by_2.is_subtype_of(&divisible_by_4));
    }

    #[test]
    fn test_subtyping_conjunction() {
        // (x > 5 && x < 10) implies x > 0
        let bounded = RefinementType::new("Int")
            .with_predicate(RefinementPredicate::GreaterThan(5.0))
            .with_predicate(RefinementPredicate::LessThan(10.0));

        let positive =
            RefinementType::new("Int").with_predicate(RefinementPredicate::GreaterThan(0.0));

        assert!(bounded.is_subtype_of(&positive));
    }

    #[test]
    fn test_subtyping_equality_implies_bounds() {
        // x == 7 implies x > 5 and x < 10
        let exact = RefinementType::new("Int").with_predicate(RefinementPredicate::Equal(7.0));

        let gt_5 = RefinementType::new("Int").with_predicate(RefinementPredicate::GreaterThan(5.0));

        let lt_10 = RefinementType::new("Int").with_predicate(RefinementPredicate::LessThan(10.0));

        assert!(exact.is_subtype_of(&gt_5));
        assert!(exact.is_subtype_of(&lt_10));
    }

    #[test]
    fn test_subtyping_no_implication() {
        // x % 2 == 0 does NOT imply x > 5
        let even = RefinementType::new("Int").with_predicate(RefinementPredicate::Modulo {
            divisor: 2,
            remainder: 0,
        });

        let gt_5 = RefinementType::new("Int").with_predicate(RefinementPredicate::GreaterThan(5.0));

        assert!(!even.is_subtype_of(&gt_5));
        assert!(!gt_5.is_subtype_of(&even));
    }
}
