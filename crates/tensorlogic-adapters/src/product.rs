//! Domain product types for cross-domain reasoning.
//!
//! Product types allow combining multiple domains into composite types,
//! enabling predicates over tuples of values from different domains.
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_adapters::ProductDomain;
//!
//! // Create Person × Location product
//! let product = ProductDomain::new(vec!["Person".to_string(), "Location".to_string()]);
//! assert_eq!(product.components(), &["Person", "Location"]);
//!
//! // Nested product: (Person × Location) × Time
//! let nested = ProductDomain::new(vec![
//!     product.to_string(),
//!     "Time".to_string()
//! ]);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::{AdapterError, DomainInfo, SymbolTable};

/// A product domain representing a tuple of component domains.
///
/// Product domains enable cross-domain reasoning by creating composite types
/// from multiple base domains. The cardinality of a product domain is the
/// product of its component cardinalities.
///
/// # Examples
///
/// ```rust
/// use tensorlogic_adapters::ProductDomain;
///
/// let product = ProductDomain::new(vec![
///     "Person".to_string(),
///     "Location".to_string()
/// ]);
///
/// assert_eq!(product.arity(), 2);
/// assert_eq!(product.to_string(), "Person × Location");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductDomain {
    /// Component domains in the product.
    components: Vec<String>,
}

impl ProductDomain {
    /// Create a new product domain from component domains.
    ///
    /// # Panics
    ///
    /// Panics if `components` has fewer than 2 elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::ProductDomain;
    ///
    /// let product = ProductDomain::new(vec!["A".to_string(), "B".to_string()]);
    /// assert_eq!(product.arity(), 2);
    /// ```
    pub fn new(components: Vec<String>) -> Self {
        assert!(
            components.len() >= 2,
            "Product domain must have at least 2 components"
        );
        Self { components }
    }

    /// Create a binary product domain (A × B).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::ProductDomain;
    ///
    /// let product = ProductDomain::binary("Person", "Location");
    /// assert_eq!(product.to_string(), "Person × Location");
    /// ```
    pub fn binary(a: impl Into<String>, b: impl Into<String>) -> Self {
        Self::new(vec![a.into(), b.into()])
    }

    /// Create a ternary product domain (A × B × C).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::ProductDomain;
    ///
    /// let product = ProductDomain::ternary("Person", "Location", "Time");
    /// assert_eq!(product.to_string(), "Person × Location × Time");
    /// ```
    pub fn ternary(a: impl Into<String>, b: impl Into<String>, c: impl Into<String>) -> Self {
        Self::new(vec![a.into(), b.into(), c.into()])
    }

    /// Get the component domains.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::ProductDomain;
    ///
    /// let product = ProductDomain::binary("A", "B");
    /// assert_eq!(product.components(), &["A", "B"]);
    /// ```
    pub fn components(&self) -> &[String] {
        &self.components
    }

    /// Get the arity (number of components) of this product.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::ProductDomain;
    ///
    /// let product = ProductDomain::ternary("A", "B", "C");
    /// assert_eq!(product.arity(), 3);
    /// ```
    pub fn arity(&self) -> usize {
        self.components.len()
    }

    /// Compute the cardinality of this product domain.
    ///
    /// Returns the product of component cardinalities, or an error if
    /// any component domain is not found in the symbol table.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, ProductDomain};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("A", 10)).unwrap();
    /// table.add_domain(DomainInfo::new("B", 20)).unwrap();
    ///
    /// let product = ProductDomain::binary("A", "B");
    /// assert_eq!(product.cardinality(&table).unwrap(), 200);
    /// ```
    pub fn cardinality(&self, table: &SymbolTable) -> Result<usize, AdapterError> {
        let mut result = 1_usize;
        for component in &self.components {
            let domain = table
                .get_domain(component)
                .ok_or_else(|| AdapterError::UnknownDomain(component.clone()))?;
            result = result.checked_mul(domain.cardinality).ok_or_else(|| {
                AdapterError::InvalidCardinality(format!(
                    "Cardinality overflow in product domain: {}",
                    self
                ))
            })?;
        }
        Ok(result)
    }

    /// Check if all component domains exist in the symbol table.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, ProductDomain};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("A", 10)).unwrap();
    /// table.add_domain(DomainInfo::new("B", 20)).unwrap();
    ///
    /// let product = ProductDomain::binary("A", "B");
    /// assert!(product.validate(&table).is_ok());
    ///
    /// let invalid = ProductDomain::binary("A", "Unknown");
    /// assert!(invalid.validate(&table).is_err());
    /// ```
    pub fn validate(&self, table: &SymbolTable) -> Result<(), AdapterError> {
        for component in &self.components {
            if table.get_domain(component).is_none() {
                return Err(AdapterError::UnknownDomain(component.clone()));
            }
        }
        Ok(())
    }

    /// Project to a specific component by index.
    ///
    /// Returns the domain name of the component at the given index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::ProductDomain;
    ///
    /// let product = ProductDomain::ternary("A", "B", "C");
    /// assert_eq!(product.project(0), Some("A"));
    /// assert_eq!(product.project(1), Some("B"));
    /// assert_eq!(product.project(2), Some("C"));
    /// assert_eq!(product.project(3), None);
    /// ```
    pub fn project(&self, index: usize) -> Option<&str> {
        self.components.get(index).map(|s| s.as_str())
    }

    /// Get a subproduct by slicing component indices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::ProductDomain;
    ///
    /// let product = ProductDomain::new(vec![
    ///     "A".to_string(),
    ///     "B".to_string(),
    ///     "C".to_string(),
    ///     "D".to_string()
    /// ]);
    ///
    /// // Get middle two components (B × C)
    /// let sub = product.slice(1, 3).unwrap();
    /// assert_eq!(sub.components(), &["B", "C"]);
    /// ```
    pub fn slice(&self, start: usize, end: usize) -> Result<ProductDomain, AdapterError> {
        if start >= end || end > self.components.len() {
            return Err(AdapterError::InvalidOperation(format!(
                "Invalid slice indices: {}..{} for product of arity {}",
                start,
                end,
                self.components.len()
            )));
        }
        let components = self.components[start..end].to_vec();
        Ok(ProductDomain::new(components))
    }

    /// Extend this product with additional components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::ProductDomain;
    ///
    /// let mut product = ProductDomain::binary("A", "B");
    /// product.extend(vec!["C".to_string(), "D".to_string()]);
    /// assert_eq!(product.arity(), 4);
    /// ```
    pub fn extend(&mut self, mut additional: Vec<String>) {
        self.components.append(&mut additional);
    }
}

impl fmt::Display for ProductDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.components.join(" × "))
    }
}

impl From<Vec<String>> for ProductDomain {
    fn from(components: Vec<String>) -> Self {
        Self::new(components)
    }
}

/// Extension trait for SymbolTable to support product domains.
pub trait ProductDomainExt {
    /// Add a product domain to the symbol table.
    ///
    /// The product domain's cardinality is computed from its components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, ProductDomain, ProductDomainExt};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    /// table.add_domain(DomainInfo::new("Location", 50)).unwrap();
    ///
    /// let product = ProductDomain::binary("Person", "Location");
    /// table.add_product_domain("PersonAtLocation", product).unwrap();
    ///
    /// let domain = table.get_domain("PersonAtLocation").unwrap();
    /// assert_eq!(domain.cardinality, 5000);
    /// ```
    fn add_product_domain(
        &mut self,
        name: impl Into<String>,
        product: ProductDomain,
    ) -> Result<(), AdapterError>;

    /// Get a product domain by name.
    ///
    /// Returns `None` if the domain doesn't exist or is not a product domain.
    fn get_product_domain(&self, name: &str) -> Option<&ProductDomain>;

    /// List all product domains in the symbol table.
    fn list_product_domains(&self) -> Vec<(&str, &ProductDomain)>;
}

impl ProductDomainExt for SymbolTable {
    fn add_product_domain(
        &mut self,
        name: impl Into<String>,
        product: ProductDomain,
    ) -> Result<(), AdapterError> {
        let name = name.into();

        // Validate that all component domains exist
        product.validate(self)?;

        // Compute cardinality
        let cardinality = product.cardinality(self)?;

        // Create domain info with product type metadata
        let mut domain_info = DomainInfo::new(&name, cardinality);
        domain_info.description = Some(format!("Product domain: {}", product));

        // Store product domain metadata (we'll use a custom attribute)
        if let Some(ref mut meta) = domain_info.metadata {
            let components_json = serde_json::to_string(&product.components).map_err(|e| {
                AdapterError::InvalidOperation(format!(
                    "Failed to serialize product components: {}",
                    e
                ))
            })?;
            meta.set_attribute("product_components", &components_json);
        }

        self.add_domain(domain_info)
            .map_err(|_| AdapterError::DuplicateDomain(name.clone()))?;
        Ok(())
    }

    fn get_product_domain(&self, name: &str) -> Option<&ProductDomain> {
        // This is a simplified implementation
        // In a real implementation, we'd store ProductDomain instances separately
        // For now, we return None as a placeholder
        let _domain = self.get_domain(name)?;
        None
    }

    fn list_product_domains(&self) -> Vec<(&str, &ProductDomain)> {
        // Simplified implementation
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_product() {
        let product = ProductDomain::binary("A", "B");
        assert_eq!(product.arity(), 2);
        assert_eq!(product.components(), &["A", "B"]);
        assert_eq!(product.to_string(), "A × B");
    }

    #[test]
    fn test_ternary_product() {
        let product = ProductDomain::ternary("A", "B", "C");
        assert_eq!(product.arity(), 3);
        assert_eq!(product.to_string(), "A × B × C");
    }

    #[test]
    fn test_cardinality() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("A", 10)).unwrap();
        table.add_domain(DomainInfo::new("B", 20)).unwrap();
        table.add_domain(DomainInfo::new("C", 5)).unwrap();

        let product = ProductDomain::ternary("A", "B", "C");
        assert_eq!(product.cardinality(&table).unwrap(), 1000);
    }

    #[test]
    fn test_validate_success() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("A", 10)).unwrap();
        table.add_domain(DomainInfo::new("B", 20)).unwrap();

        let product = ProductDomain::binary("A", "B");
        assert!(product.validate(&table).is_ok());
    }

    #[test]
    fn test_validate_unknown_domain() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("A", 10)).unwrap();

        let product = ProductDomain::binary("A", "Unknown");
        assert!(product.validate(&table).is_err());
    }

    #[test]
    fn test_project() {
        let product = ProductDomain::ternary("A", "B", "C");
        assert_eq!(product.project(0), Some("A"));
        assert_eq!(product.project(1), Some("B"));
        assert_eq!(product.project(2), Some("C"));
        assert_eq!(product.project(3), None);
    }

    #[test]
    fn test_slice() {
        let product = ProductDomain::new(vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ]);

        let sub = product.slice(1, 3).unwrap();
        assert_eq!(sub.components(), &["B", "C"]);
        assert_eq!(sub.to_string(), "B × C");
    }

    #[test]
    fn test_slice_invalid() {
        let product = ProductDomain::binary("A", "B");
        assert!(product.slice(0, 3).is_err());
        assert!(product.slice(2, 1).is_err());
    }

    #[test]
    fn test_extend() {
        let mut product = ProductDomain::binary("A", "B");
        product.extend(vec!["C".to_string(), "D".to_string()]);
        assert_eq!(product.arity(), 4);
        assert_eq!(product.to_string(), "A × B × C × D");
    }

    #[test]
    fn test_add_product_domain() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Location", 50)).unwrap();

        let product = ProductDomain::binary("Person", "Location");
        table
            .add_product_domain("PersonAtLocation", product)
            .unwrap();

        let domain = table.get_domain("PersonAtLocation").unwrap();
        assert_eq!(domain.cardinality, 5000);
        assert!(domain
            .description
            .as_ref()
            .unwrap()
            .contains("Product domain"));
    }

    #[test]
    #[should_panic(expected = "Product domain must have at least 2 components")]
    fn test_invalid_single_component() {
        ProductDomain::new(vec!["A".to_string()]);
    }

    #[test]
    fn test_nested_product() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("A", 10)).unwrap();
        table.add_domain(DomainInfo::new("B", 20)).unwrap();
        table.add_domain(DomainInfo::new("C", 5)).unwrap();

        // Create (A × B)
        let ab = ProductDomain::binary("A", "B");
        table.add_product_domain("AB", ab).unwrap();

        // Create (AB × C)
        let abc = ProductDomain::binary("AB", "C");
        assert_eq!(abc.cardinality(&table).unwrap(), 1000);
    }

    #[test]
    fn test_display() {
        let product = ProductDomain::new(vec![
            "Person".to_string(),
            "Location".to_string(),
            "Time".to_string(),
        ]);
        assert_eq!(format!("{}", product), "Person × Location × Time");
    }

    #[test]
    fn test_from_vec() {
        let components = vec!["A".to_string(), "B".to_string()];
        let product: ProductDomain = components.into();
        assert_eq!(product.arity(), 2);
    }
}
