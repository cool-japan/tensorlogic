//! Predicate constraints and properties.

use serde::{Deserialize, Serialize};

/// Properties that can be associated with predicates
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PredicateProperty {
    /// Predicate is symmetric: P(x,y) ⟺ P(y,x)
    Symmetric,
    /// Predicate is transitive: P(x,y) ∧ P(y,z) ⟹ P(x,z)
    Transitive,
    /// Predicate is reflexive: ∀x. P(x,x)
    Reflexive,
    /// Predicate is irreflexive: ∀x. ¬P(x,x)
    Irreflexive,
    /// Predicate is antisymmetric: P(x,y) ∧ P(y,x) ⟹ x = y
    Antisymmetric,
    /// Predicate is functional: ∀x,y,z. P(x,y) ∧ P(x,z) ⟹ y = z
    Functional,
    /// Predicate is inverse functional: ∀x,y,z. P(y,x) ∧ P(z,x) ⟹ y = z
    InverseFunctional,
}

/// Value range constraint for numeric predicates
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValueRange {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub inclusive_min: bool,
    pub inclusive_max: bool,
}

impl ValueRange {
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            inclusive_min: true,
            inclusive_max: true,
        }
    }

    pub fn with_min(mut self, min: f64, inclusive: bool) -> Self {
        self.min = Some(min);
        self.inclusive_min = inclusive;
        self
    }

    pub fn with_max(mut self, max: f64, inclusive: bool) -> Self {
        self.max = Some(max);
        self.inclusive_max = inclusive;
        self
    }

    pub fn contains(&self, value: f64) -> bool {
        if let Some(min) = self.min {
            if self.inclusive_min {
                if value < min {
                    return false;
                }
            } else if value <= min {
                return false;
            }
        }

        if let Some(max) = self.max {
            if self.inclusive_max {
                if value > max {
                    return false;
                }
            } else if value >= max {
                return false;
            }
        }

        true
    }
}

impl Default for ValueRange {
    fn default() -> Self {
        Self::new()
    }
}

/// Functional dependency between predicate arguments
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionalDependency {
    /// Determining arguments (indices)
    pub determinants: Vec<usize>,
    /// Dependent arguments (indices)
    pub dependents: Vec<usize>,
}

impl FunctionalDependency {
    pub fn new(determinants: Vec<usize>, dependents: Vec<usize>) -> Self {
        Self {
            determinants,
            dependents,
        }
    }
}

/// Constraints associated with a predicate
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PredicateConstraints {
    /// Logical properties of the predicate
    pub properties: Vec<PredicateProperty>,
    /// Value ranges for each argument (None if unconstrained)
    pub value_ranges: Vec<Option<ValueRange>>,
    /// Functional dependencies between arguments
    pub functional_dependencies: Vec<FunctionalDependency>,
}

impl PredicateConstraints {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_property(mut self, property: PredicateProperty) -> Self {
        self.properties.push(property);
        self
    }

    pub fn with_value_range(mut self, arg_index: usize, range: ValueRange) -> Self {
        while self.value_ranges.len() <= arg_index {
            self.value_ranges.push(None);
        }
        self.value_ranges[arg_index] = Some(range);
        self
    }

    pub fn with_functional_dependency(mut self, dependency: FunctionalDependency) -> Self {
        self.functional_dependencies.push(dependency);
        self
    }

    pub fn has_property(&self, property: &PredicateProperty) -> bool {
        self.properties.contains(property)
    }

    pub fn is_symmetric(&self) -> bool {
        self.has_property(&PredicateProperty::Symmetric)
    }

    pub fn is_transitive(&self) -> bool {
        self.has_property(&PredicateProperty::Transitive)
    }

    pub fn is_reflexive(&self) -> bool {
        self.has_property(&PredicateProperty::Reflexive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_range() {
        let range = ValueRange::new().with_min(0.0, true).with_max(1.0, true);

        assert!(range.contains(0.0));
        assert!(range.contains(0.5));
        assert!(range.contains(1.0));
        assert!(!range.contains(-0.1));
        assert!(!range.contains(1.1));
    }

    #[test]
    fn test_value_range_exclusive() {
        let range = ValueRange::new().with_min(0.0, false).with_max(1.0, false);

        assert!(!range.contains(0.0));
        assert!(range.contains(0.5));
        assert!(!range.contains(1.0));
    }

    #[test]
    fn test_predicate_properties() {
        let constraints = PredicateConstraints::new()
            .with_property(PredicateProperty::Symmetric)
            .with_property(PredicateProperty::Transitive);

        assert!(constraints.is_symmetric());
        assert!(constraints.is_transitive());
        assert!(!constraints.is_reflexive());
    }

    #[test]
    fn test_functional_dependency() {
        let fd = FunctionalDependency::new(vec![0], vec![1, 2]);

        assert_eq!(fd.determinants, vec![0]);
        assert_eq!(fd.dependents, vec![1, 2]);
    }
}
