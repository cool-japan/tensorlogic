//! Domain hierarchy and subtype relationships.

use anyhow::{bail, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Domain hierarchy tracking subtype relationships
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainHierarchy {
    /// Map from subdomain to parent domain
    parent_map: IndexMap<String, String>,
}

impl DomainHierarchy {
    pub fn new() -> Self {
        Self {
            parent_map: IndexMap::new(),
        }
    }

    /// Add a subtype relationship: subdomain <: parent
    pub fn add_subtype(&mut self, subdomain: impl Into<String>, parent: impl Into<String>) {
        self.parent_map.insert(subdomain.into(), parent.into());
    }

    /// Check if subdomain is a subtype of parent (directly or transitively)
    pub fn is_subtype(&self, subdomain: &str, parent: &str) -> bool {
        if subdomain == parent {
            return true;
        }

        let mut current = subdomain;
        while let Some(p) = self.parent_map.get(current) {
            if p == parent {
                return true;
            }
            current = p;
        }

        false
    }

    /// Get the direct parent of a domain, if any
    pub fn get_parent(&self, domain: &str) -> Option<&str> {
        self.parent_map.get(domain).map(|s| s.as_str())
    }

    /// Get all ancestors of a domain (parent, grandparent, etc.)
    pub fn get_ancestors(&self, domain: &str) -> Vec<String> {
        let mut ancestors = Vec::new();
        let mut current = domain;

        while let Some(parent) = self.parent_map.get(current) {
            ancestors.push(parent.clone());
            current = parent;
        }

        ancestors
    }

    /// Get all descendants of a domain
    pub fn get_descendants(&self, domain: &str) -> Vec<String> {
        self.parent_map
            .iter()
            .filter_map(|(child, parent)| {
                if parent == domain || self.is_subtype(child, domain) {
                    Some(child.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Validate that there are no cycles in the hierarchy
    pub fn validate_acyclic(&self) -> Result<()> {
        for domain in self.parent_map.keys() {
            let mut visited = std::collections::HashSet::new();
            let mut current = domain.as_str();

            while let Some(parent) = self.parent_map.get(current) {
                if !visited.insert(current) {
                    bail!("Cycle detected in domain hierarchy involving '{}'", domain);
                }
                current = parent;
            }
        }

        Ok(())
    }

    /// Find the least common supertype of two domains
    pub fn least_common_supertype(&self, domain1: &str, domain2: &str) -> Option<String> {
        if domain1 == domain2 {
            return Some(domain1.to_string());
        }

        let ancestors1: std::collections::HashSet<_> =
            self.get_ancestors(domain1).into_iter().collect();

        if ancestors1.contains(domain2) {
            return Some(domain2.to_string());
        }

        self.get_ancestors(domain2)
            .into_iter()
            .find(|ancestor| ancestors1.contains(ancestor))
    }

    /// Get all domains in the hierarchy (both subdomains and their parents).
    pub fn all_domains(&self) -> Vec<String> {
        let mut domains = std::collections::HashSet::new();

        for (subdomain, parent) in &self.parent_map {
            domains.insert(subdomain.clone());
            domains.insert(parent.clone());
        }

        domains.into_iter().collect()
    }
}

impl Default for DomainHierarchy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtype_direct() {
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Student", "Person");

        assert!(hierarchy.is_subtype("Student", "Person"));
        assert!(hierarchy.is_subtype("Student", "Student"));
        assert!(!hierarchy.is_subtype("Person", "Student"));
    }

    #[test]
    fn test_subtype_transitive() {
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Student", "Person");
        hierarchy.add_subtype("Person", "Agent");

        assert!(hierarchy.is_subtype("Student", "Agent"));
        assert!(hierarchy.is_subtype("Student", "Person"));
        assert!(hierarchy.is_subtype("Person", "Agent"));
        assert!(!hierarchy.is_subtype("Agent", "Student"));
    }

    #[test]
    fn test_get_ancestors() {
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Student", "Person");
        hierarchy.add_subtype("Person", "Agent");

        let ancestors = hierarchy.get_ancestors("Student");
        assert_eq!(ancestors, vec!["Person", "Agent"]);
    }

    #[test]
    fn test_get_descendants() {
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Student", "Person");
        hierarchy.add_subtype("Teacher", "Person");

        let descendants = hierarchy.get_descendants("Person");
        assert_eq!(descendants.len(), 2);
        assert!(descendants.contains(&"Student".to_string()));
        assert!(descendants.contains(&"Teacher".to_string()));
    }

    #[test]
    fn test_least_common_supertype() {
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Student", "Person");
        hierarchy.add_subtype("Teacher", "Person");
        hierarchy.add_subtype("Person", "Agent");

        assert_eq!(
            hierarchy.least_common_supertype("Student", "Teacher"),
            Some("Person".to_string())
        );
        assert_eq!(
            hierarchy.least_common_supertype("Student", "Student"),
            Some("Student".to_string())
        );
    }

    #[test]
    fn test_validate_acyclic() {
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("A", "B");
        hierarchy.add_subtype("B", "C");

        assert!(hierarchy.validate_acyclic().is_ok());
    }

    #[test]
    fn test_validate_cycle_detection() {
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("A", "B");
        hierarchy.add_subtype("B", "A");

        assert!(hierarchy.validate_acyclic().is_err());
    }
}
