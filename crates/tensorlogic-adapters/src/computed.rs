//! Computed and virtual domains for derived types.
//!
//! Computed domains represent types that are derived from other domains
//! through operations or transformations. They enable lazy evaluation
//! and dynamic domain generation.
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_adapters::{ComputedDomain, DomainComputation};
//!
//! // Create a filtered domain
//! let adults = ComputedDomain::new(
//!     "Adults",
//!     DomainComputation::Filter {
//!         base: "Person".to_string(),
//!         predicate: "is_adult".to_string(),
//!     }
//! );
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::{AdapterError, SymbolTable};

/// Types of domain computations.
///
/// Each variant represents a different way of deriving a new domain
/// from existing domains.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainComputation {
    /// Filter a base domain by a predicate.
    ///
    /// Creates a subset of the base domain containing only elements
    /// satisfying the predicate.
    Filter {
        /// Base domain to filter.
        base: String,
        /// Predicate name for filtering.
        predicate: String,
    },

    /// Union of multiple domains.
    ///
    /// Creates a domain containing elements from all source domains.
    Union {
        /// Source domains to union.
        domains: Vec<String>,
    },

    /// Intersection of multiple domains.
    ///
    /// Creates a domain containing only elements present in all source domains.
    Intersection {
        /// Source domains to intersect.
        domains: Vec<String>,
    },

    /// Difference between two domains (A - B).
    ///
    /// Creates a domain containing elements in the first domain but not in the second.
    Difference {
        /// Base domain.
        base: String,
        /// Domain to subtract.
        subtract: String,
    },

    /// Product of domains.
    ///
    /// Creates a cartesian product of the source domains.
    Product {
        /// Domains to take product of.
        domains: Vec<String>,
    },

    /// Power set of a domain.
    ///
    /// Creates a domain containing all subsets of the base domain.
    PowerSet {
        /// Base domain.
        base: String,
    },

    /// Projection from a product domain.
    ///
    /// Extracts a component from a product domain.
    Projection {
        /// Product domain to project from.
        product: String,
        /// Index of component to project.
        index: usize,
    },

    /// Custom computation with a formula.
    ///
    /// Allows user-defined domain computations with arbitrary logic.
    Custom {
        /// Description of the computation.
        description: String,
        /// Formula or implementation reference.
        formula: String,
    },
}

impl fmt::Display for DomainComputation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Filter { base, predicate } => write!(f, "{{ x ∈ {} | {}(x) }}", base, predicate),
            Self::Union { domains } => write!(f, "{}", domains.join(" ∪ ")),
            Self::Intersection { domains } => write!(f, "{}", domains.join(" ∩ ")),
            Self::Difference { base, subtract } => write!(f, "{} \\ {}", base, subtract),
            Self::Product { domains } => write!(f, "{}", domains.join(" × ")),
            Self::PowerSet { base } => write!(f, "℘({})", base),
            Self::Projection { product, index } => write!(f, "π{}({})", index, product),
            Self::Custom { description, .. } => write!(f, "{}", description),
        }
    }
}

/// A computed domain that is derived from other domains.
///
/// Computed domains are evaluated lazily and can represent complex
/// domain transformations without materializing all elements.
///
/// # Examples
///
/// ```rust
/// use tensorlogic_adapters::{ComputedDomain, DomainComputation};
///
/// // Filter domain
/// let adults = ComputedDomain::new(
///     "Adults",
///     DomainComputation::Filter {
///         base: "Person".to_string(),
///         predicate: "is_adult".to_string(),
///     }
/// );
///
/// // Union domain
/// let entities = ComputedDomain::new(
///     "Entities",
///     DomainComputation::Union {
///         domains: vec!["Person".to_string(), "Organization".to_string()],
///     }
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputedDomain {
    /// Name of the computed domain.
    name: String,
    /// Computation defining this domain.
    computation: DomainComputation,
    /// Optional cardinality estimate or bound.
    cardinality_estimate: Option<usize>,
    /// Whether this domain is materialized (computed and cached).
    materialized: bool,
}

impl ComputedDomain {
    /// Create a new computed domain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::{ComputedDomain, DomainComputation};
    ///
    /// let domain = ComputedDomain::new(
    ///     "FilteredDomain",
    ///     DomainComputation::Filter {
    ///         base: "Base".to_string(),
    ///         predicate: "pred".to_string(),
    ///     }
    /// );
    /// ```
    pub fn new(name: impl Into<String>, computation: DomainComputation) -> Self {
        Self {
            name: name.into(),
            computation,
            cardinality_estimate: None,
            materialized: false,
        }
    }

    /// Set a cardinality estimate for this computed domain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::{ComputedDomain, DomainComputation};
    ///
    /// let domain = ComputedDomain::new(
    ///     "Adults",
    ///     DomainComputation::Filter {
    ///         base: "Person".to_string(),
    ///         predicate: "is_adult".to_string(),
    ///     }
    /// ).with_cardinality_estimate(750);
    /// ```
    pub fn with_cardinality_estimate(mut self, estimate: usize) -> Self {
        self.cardinality_estimate = Some(estimate);
        self
    }

    /// Get the name of this computed domain.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the computation defining this domain.
    pub fn computation(&self) -> &DomainComputation {
        &self.computation
    }

    /// Get the cardinality estimate if available.
    pub fn cardinality_estimate(&self) -> Option<usize> {
        self.cardinality_estimate
    }

    /// Check if this domain is materialized.
    pub fn is_materialized(&self) -> bool {
        self.materialized
    }

    /// Compute the cardinality bounds for this domain.
    ///
    /// Returns (lower_bound, upper_bound) or an error if dependencies are missing.
    pub fn cardinality_bounds(&self, table: &SymbolTable) -> Result<(usize, usize), AdapterError> {
        match &self.computation {
            DomainComputation::Filter { base, .. } => {
                let base_card = table
                    .get_domain(base)
                    .ok_or_else(|| AdapterError::UnknownDomain(base.clone()))?
                    .cardinality;
                // Lower bound is 0 (could filter all), upper bound is base cardinality
                Ok((0, base_card))
            }
            DomainComputation::Union { domains } => {
                let max_card = domains
                    .iter()
                    .map(|d| {
                        table
                            .get_domain(d)
                            .map(|info| info.cardinality)
                            .ok_or_else(|| AdapterError::UnknownDomain(d.clone()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .max()
                    .unwrap_or(0);
                let sum_card: usize = domains
                    .iter()
                    .map(|d| table.get_domain(d).unwrap().cardinality)
                    .sum();
                Ok((max_card, sum_card))
            }
            DomainComputation::Intersection { domains } => {
                let min_card = domains
                    .iter()
                    .map(|d| {
                        table
                            .get_domain(d)
                            .map(|info| info.cardinality)
                            .ok_or_else(|| AdapterError::UnknownDomain(d.clone()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .min()
                    .unwrap_or(0);
                Ok((0, min_card))
            }
            DomainComputation::Difference { base, subtract } => {
                let base_card = table
                    .get_domain(base)
                    .ok_or_else(|| AdapterError::UnknownDomain(base.clone()))?
                    .cardinality;
                let sub_card = table
                    .get_domain(subtract)
                    .ok_or_else(|| AdapterError::UnknownDomain(subtract.clone()))?
                    .cardinality;
                Ok((0, base_card.saturating_sub(sub_card)))
            }
            DomainComputation::Product { domains } => {
                let product: usize = domains
                    .iter()
                    .map(|d| {
                        table
                            .get_domain(d)
                            .map(|info| info.cardinality)
                            .ok_or_else(|| AdapterError::UnknownDomain(d.clone()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .product();
                Ok((product, product))
            }
            DomainComputation::PowerSet { base } => {
                let base_card = table
                    .get_domain(base)
                    .ok_or_else(|| AdapterError::UnknownDomain(base.clone()))?
                    .cardinality;
                let power = 2_usize.pow(base_card as u32);
                Ok((power, power))
            }
            DomainComputation::Projection { product, index: _ } => {
                let prod_domain = table
                    .get_domain(product)
                    .ok_or_else(|| AdapterError::UnknownDomain(product.clone()))?;
                // This is simplified - in real implementation we'd parse product structure
                Ok((0, prod_domain.cardinality))
            }
            DomainComputation::Custom { .. } => {
                // For custom computations, use estimate if available
                if let Some(est) = self.cardinality_estimate {
                    Ok((0, est))
                } else {
                    Err(AdapterError::InvalidOperation(
                        "Custom domain computation requires cardinality estimate".to_string(),
                    ))
                }
            }
        }
    }

    /// Validate that all dependencies exist in the symbol table.
    pub fn validate(&self, table: &SymbolTable) -> Result<(), AdapterError> {
        match &self.computation {
            DomainComputation::Filter { base, predicate } => {
                if table.get_domain(base).is_none() {
                    return Err(AdapterError::UnknownDomain(base.clone()));
                }
                if table.get_predicate(predicate).is_none() {
                    return Err(AdapterError::UnknownPredicate(predicate.clone()));
                }
            }
            DomainComputation::Union { domains } | DomainComputation::Intersection { domains } => {
                for domain in domains {
                    if table.get_domain(domain).is_none() {
                        return Err(AdapterError::UnknownDomain(domain.clone()));
                    }
                }
            }
            DomainComputation::Difference { base, subtract } => {
                if table.get_domain(base).is_none() {
                    return Err(AdapterError::UnknownDomain(base.clone()));
                }
                if table.get_domain(subtract).is_none() {
                    return Err(AdapterError::UnknownDomain(subtract.clone()));
                }
            }
            DomainComputation::Product { domains } => {
                for domain in domains {
                    if table.get_domain(domain).is_none() {
                        return Err(AdapterError::UnknownDomain(domain.clone()));
                    }
                }
            }
            DomainComputation::PowerSet { base } => {
                if table.get_domain(base).is_none() {
                    return Err(AdapterError::UnknownDomain(base.clone()));
                }
            }
            DomainComputation::Projection { product, .. } => {
                if table.get_domain(product).is_none() {
                    return Err(AdapterError::UnknownDomain(product.clone()));
                }
            }
            DomainComputation::Custom { .. } => {
                // Custom computations can't be validated automatically
            }
        }
        Ok(())
    }

    /// Mark this domain as materialized.
    pub fn materialize(&mut self) {
        self.materialized = true;
    }
}

impl fmt::Display for ComputedDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} := {}", self.name, self.computation)
    }
}

/// Registry for managing computed domains.
///
/// The registry tracks computed domains and their dependencies,
/// enabling lazy evaluation and caching.
#[derive(Clone, Debug, Default)]
pub struct ComputedDomainRegistry {
    /// Registered computed domains.
    domains: HashMap<String, ComputedDomain>,
}

impl ComputedDomainRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            domains: HashMap::new(),
        }
    }

    /// Register a computed domain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::{ComputedDomainRegistry, ComputedDomain, DomainComputation};
    ///
    /// let mut registry = ComputedDomainRegistry::new();
    /// let domain = ComputedDomain::new(
    ///     "Adults",
    ///     DomainComputation::Filter {
    ///         base: "Person".to_string(),
    ///         predicate: "is_adult".to_string(),
    ///     }
    /// );
    /// registry.register(domain).unwrap();
    /// ```
    pub fn register(&mut self, domain: ComputedDomain) -> Result<(), AdapterError> {
        let name = domain.name().to_string();
        if self.domains.contains_key(&name) {
            return Err(AdapterError::DuplicateDomain(name));
        }
        self.domains.insert(name, domain);
        Ok(())
    }

    /// Get a computed domain by name.
    pub fn get(&self, name: &str) -> Option<&ComputedDomain> {
        self.domains.get(name)
    }

    /// Get a mutable reference to a computed domain.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ComputedDomain> {
        self.domains.get_mut(name)
    }

    /// List all registered computed domains.
    pub fn list(&self) -> Vec<&ComputedDomain> {
        self.domains.values().collect()
    }

    /// Remove a computed domain from the registry.
    pub fn remove(&mut self, name: &str) -> Option<ComputedDomain> {
        self.domains.remove(name)
    }

    /// Validate all computed domains against a symbol table.
    pub fn validate_all(&self, table: &SymbolTable) -> Result<(), Vec<AdapterError>> {
        let errors: Vec<_> = self
            .domains
            .values()
            .filter_map(|domain| domain.validate(table).err())
            .collect();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get the number of registered domains.
    pub fn len(&self) -> usize {
        self.domains.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.domains.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DomainInfo, PredicateInfo};

    #[test]
    fn test_filter_computation() {
        let comp = DomainComputation::Filter {
            base: "Person".to_string(),
            predicate: "is_adult".to_string(),
        };
        assert!(comp.to_string().contains("Person"));
        assert!(comp.to_string().contains("is_adult"));
    }

    #[test]
    fn test_union_computation() {
        let comp = DomainComputation::Union {
            domains: vec!["A".to_string(), "B".to_string(), "C".to_string()],
        };
        assert_eq!(comp.to_string(), "A ∪ B ∪ C");
    }

    #[test]
    fn test_intersection_computation() {
        let comp = DomainComputation::Intersection {
            domains: vec!["A".to_string(), "B".to_string()],
        };
        assert_eq!(comp.to_string(), "A ∩ B");
    }

    #[test]
    fn test_difference_computation() {
        let comp = DomainComputation::Difference {
            base: "A".to_string(),
            subtract: "B".to_string(),
        };
        assert_eq!(comp.to_string(), "A \\ B");
    }

    #[test]
    fn test_product_computation() {
        let comp = DomainComputation::Product {
            domains: vec!["A".to_string(), "B".to_string()],
        };
        assert_eq!(comp.to_string(), "A × B");
    }

    #[test]
    fn test_powerset_computation() {
        let comp = DomainComputation::PowerSet {
            base: "A".to_string(),
        };
        assert_eq!(comp.to_string(), "℘(A)");
    }

    #[test]
    fn test_computed_domain_creation() {
        let domain = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "is_adult".to_string(),
            },
        );
        assert_eq!(domain.name(), "Adults");
        assert!(!domain.is_materialized());
    }

    #[test]
    fn test_cardinality_estimate() {
        let domain = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "is_adult".to_string(),
            },
        )
        .with_cardinality_estimate(750);
        assert_eq!(domain.cardinality_estimate(), Some(750));
    }

    #[test]
    fn test_cardinality_bounds_filter() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 1000)).unwrap();
        table
            .add_predicate(PredicateInfo::new("is_adult", vec!["Person".to_string()]))
            .unwrap();

        let domain = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "is_adult".to_string(),
            },
        );

        let (lower, upper) = domain.cardinality_bounds(&table).unwrap();
        assert_eq!(lower, 0);
        assert_eq!(upper, 1000);
    }

    #[test]
    fn test_cardinality_bounds_union() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("A", 100)).unwrap();
        table.add_domain(DomainInfo::new("B", 200)).unwrap();

        let domain = ComputedDomain::new(
            "AorB",
            DomainComputation::Union {
                domains: vec!["A".to_string(), "B".to_string()],
            },
        );

        let (lower, upper) = domain.cardinality_bounds(&table).unwrap();
        assert_eq!(lower, 200); // max(100, 200)
        assert_eq!(upper, 300); // 100 + 200
    }

    #[test]
    fn test_cardinality_bounds_product() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("A", 10)).unwrap();
        table.add_domain(DomainInfo::new("B", 20)).unwrap();

        let domain = ComputedDomain::new(
            "AxB",
            DomainComputation::Product {
                domains: vec!["A".to_string(), "B".to_string()],
            },
        );

        let (lower, upper) = domain.cardinality_bounds(&table).unwrap();
        assert_eq!(lower, 200);
        assert_eq!(upper, 200);
    }

    #[test]
    fn test_validate_success() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 1000)).unwrap();
        table
            .add_predicate(PredicateInfo::new("is_adult", vec!["Person".to_string()]))
            .unwrap();

        let domain = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "is_adult".to_string(),
            },
        );

        assert!(domain.validate(&table).is_ok());
    }

    #[test]
    fn test_validate_missing_domain() {
        let table = SymbolTable::new();

        let domain = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "is_adult".to_string(),
            },
        );

        assert!(domain.validate(&table).is_err());
    }

    #[test]
    fn test_registry_register() {
        let mut registry = ComputedDomainRegistry::new();
        let domain = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "is_adult".to_string(),
            },
        );
        assert!(registry.register(domain).is_ok());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_duplicate() {
        let mut registry = ComputedDomainRegistry::new();
        let domain1 = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "is_adult".to_string(),
            },
        );
        let domain2 = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "other".to_string(),
            },
        );
        registry.register(domain1).unwrap();
        assert!(registry.register(domain2).is_err());
    }

    #[test]
    fn test_registry_get() {
        let mut registry = ComputedDomainRegistry::new();
        let domain = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "is_adult".to_string(),
            },
        );
        registry.register(domain).unwrap();

        assert!(registry.get("Adults").is_some());
        assert!(registry.get("Unknown").is_none());
    }

    #[test]
    fn test_registry_remove() {
        let mut registry = ComputedDomainRegistry::new();
        let domain = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "is_adult".to_string(),
            },
        );
        registry.register(domain).unwrap();

        assert!(registry.remove("Adults").is_some());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_display() {
        let domain = ComputedDomain::new(
            "Adults",
            DomainComputation::Filter {
                base: "Person".to_string(),
                predicate: "is_adult".to_string(),
            },
        );
        let s = format!("{}", domain);
        assert!(s.contains("Adults"));
        assert!(s.contains(":="));
    }
}
