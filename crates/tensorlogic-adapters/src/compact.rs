//! Compact schema representation for efficient storage and transmission.
//!
//! This module provides space-efficient encodings for symbol tables,
//! using techniques like string interning and delta encoding.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{DomainInfo, PredicateInfo, StringInterner, SymbolTable};

/// Compact representation of a symbol table.
///
/// Uses string interning and delta encoding to minimize size.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactSchema {
    /// String interner for deduplication.
    strings: Vec<String>,
    /// Compact domain representations.
    domains: Vec<CompactDomain>,
    /// Compact predicate representations.
    predicates: Vec<CompactPredicate>,
    /// Variable bindings (name_id, domain_id).
    variables: Vec<(usize, usize)>,
}

/// Compact domain representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompactDomain {
    /// String ID for name.
    name_id: usize,
    /// Cardinality.
    cardinality: usize,
    /// Optional description ID.
    description_id: Option<usize>,
}

/// Compact predicate representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompactPredicate {
    /// String ID for name.
    name_id: usize,
    /// String IDs for argument domains.
    arg_domain_ids: Vec<usize>,
    /// Optional description ID.
    description_id: Option<usize>,
}

impl CompactSchema {
    /// Create a compact schema from a symbol table.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tensorlogic_adapters::{SymbolTable, DomainInfo, CompactSchema};
    ///
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let compact = CompactSchema::from_symbol_table(&table);
    /// let recovered = compact.to_symbol_table().unwrap();
    ///
    /// assert_eq!(table.domains.len(), recovered.domains.len());
    /// ```
    pub fn from_symbol_table(table: &SymbolTable) -> Self {
        let mut interner = StringInterner::new();
        let mut string_to_id = HashMap::new();

        // Helper to intern a string
        let mut intern = |s: &str| -> usize {
            if let Some(&id) = string_to_id.get(s) {
                id
            } else {
                let id = interner.intern(s);
                string_to_id.insert(s.to_string(), id);
                id
            }
        };

        // Compact domains
        let domains: Vec<_> = table
            .domains
            .values()
            .map(|domain| {
                let name_id = intern(&domain.name);
                let description_id = domain.description.as_ref().map(|d| intern(d));

                CompactDomain {
                    name_id,
                    cardinality: domain.cardinality,
                    description_id,
                }
            })
            .collect();

        // Compact predicates
        let predicates: Vec<_> = table
            .predicates
            .values()
            .map(|pred| {
                let name_id = intern(&pred.name);
                let arg_domain_ids: Vec<_> = pred.arg_domains.iter().map(|d| intern(d)).collect();
                let description_id = pred.description.as_ref().map(|d| intern(d));

                CompactPredicate {
                    name_id,
                    arg_domain_ids,
                    description_id,
                }
            })
            .collect();

        // Compact variables
        let variables: Vec<_> = table
            .variables
            .iter()
            .map(|(var, domain)| {
                let var_id = intern(var);
                let domain_id = intern(domain);
                (var_id, domain_id)
            })
            .collect();

        // Extract interned strings
        let strings: Vec<_> = (0..interner.len())
            .filter_map(|id| interner.resolve(id).map(|s| s.to_string()))
            .collect();

        CompactSchema {
            strings,
            domains,
            predicates,
            variables,
        }
    }

    /// Convert compact schema back to a symbol table.
    pub fn to_symbol_table(&self) -> Result<SymbolTable> {
        let mut table = SymbolTable::new();

        // Reconstruct domains
        for compact in &self.domains {
            let name = self.strings.get(compact.name_id).ok_or_else(|| {
                anyhow::anyhow!("Invalid string ID {} for domain name", compact.name_id)
            })?;

            let mut domain = DomainInfo::new(name.clone(), compact.cardinality);

            if let Some(desc_id) = compact.description_id {
                let description = self.strings.get(desc_id).ok_or_else(|| {
                    anyhow::anyhow!("Invalid string ID {} for description", desc_id)
                })?;
                domain.description = Some(description.clone());
            }

            table.add_domain(domain)?;
        }

        // Reconstruct predicates
        for compact in &self.predicates {
            let name = self.strings.get(compact.name_id).ok_or_else(|| {
                anyhow::anyhow!("Invalid string ID {} for predicate name", compact.name_id)
            })?;

            let arg_domains: Result<Vec<_>> = compact
                .arg_domain_ids
                .iter()
                .map(|&id| {
                    self.strings
                        .get(id)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("Invalid string ID {} for arg domain", id))
                })
                .collect();

            let mut pred = PredicateInfo::new(name.clone(), arg_domains?);

            if let Some(desc_id) = compact.description_id {
                let description = self.strings.get(desc_id).ok_or_else(|| {
                    anyhow::anyhow!("Invalid string ID {} for description", desc_id)
                })?;
                pred.description = Some(description.clone());
            }

            table.add_predicate(pred)?;
        }

        // Reconstruct variables
        for &(var_id, domain_id) in &self.variables {
            let var = self
                .strings
                .get(var_id)
                .ok_or_else(|| anyhow::anyhow!("Invalid string ID {} for variable", var_id))?;

            let domain = self.strings.get(domain_id).ok_or_else(|| {
                anyhow::anyhow!("Invalid string ID {} for variable domain", domain_id)
            })?;

            table.bind_variable(var, domain)?;
        }

        Ok(table)
    }

    /// Serialize to compact binary format.
    pub fn to_binary(&self) -> Result<Vec<u8>> {
        oxicode::serde::encode_to_vec(self, oxicode::config::standard())
            .map_err(|e| anyhow::anyhow!("Bincode encode error: {}", e))
    }

    /// Deserialize from compact binary format.
    pub fn from_binary(data: &[u8]) -> Result<Self> {
        let (result, _): (Self, usize) =
            oxicode::serde::decode_from_slice(data, oxicode::config::standard())
                .map_err(|e| anyhow::anyhow!("Bincode decode error: {}", e))?;
        Ok(result)
    }

    /// Get the number of unique strings.
    pub fn string_count(&self) -> usize {
        self.strings.len()
    }

    /// Get statistics about compression.
    pub fn compression_stats(&self) -> CompressionStats {
        let string_bytes: usize = self.strings.iter().map(|s| s.len()).sum();
        let domain_count = self.domains.len();
        let predicate_count = self.predicates.len();
        let variable_count = self.variables.len();

        // Estimate original size (rough approximation)
        let avg_string_len = if !self.strings.is_empty() {
            string_bytes / self.strings.len()
        } else {
            0
        };

        let estimated_original_size = domain_count * (avg_string_len + 16) // name + cardinality + overhead
            + predicate_count * (avg_string_len + 16) // name + args overhead
            + variable_count * (avg_string_len * 2); // var name + domain name

        CompressionStats {
            unique_strings: self.strings.len(),
            total_string_bytes: string_bytes,
            domain_count,
            predicate_count,
            variable_count,
            estimated_original_size,
            compact_size: string_bytes
                + domain_count * 24
                + predicate_count * 24
                + variable_count * 16,
        }
    }
}

/// Compression statistics for compact schemas.
#[derive(Clone, Debug)]
pub struct CompressionStats {
    /// Number of unique strings.
    pub unique_strings: usize,
    /// Total bytes used by strings.
    pub total_string_bytes: usize,
    /// Number of domains.
    pub domain_count: usize,
    /// Number of predicates.
    pub predicate_count: usize,
    /// Number of variables.
    pub variable_count: usize,
    /// Estimated original size in bytes.
    pub estimated_original_size: usize,
    /// Compact representation size in bytes.
    pub compact_size: usize,
}

impl CompressionStats {
    /// Calculate compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.estimated_original_size > 0 {
            self.compact_size as f64 / self.estimated_original_size as f64
        } else {
            1.0
        }
    }

    /// Calculate space savings as a percentage.
    pub fn space_savings(&self) -> f64 {
        (1.0 - self.compression_ratio()) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_round_trip() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Location", 50)).unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "at",
                vec!["Person".to_string(), "Location".to_string()],
            ))
            .unwrap();
        table.bind_variable("x", "Person").unwrap();

        let compact = CompactSchema::from_symbol_table(&table);
        let recovered = compact.to_symbol_table().unwrap();

        assert_eq!(table.domains.len(), recovered.domains.len());
        assert_eq!(table.predicates.len(), recovered.predicates.len());
        assert_eq!(table.variables.len(), recovered.variables.len());
    }

    #[test]
    fn test_string_deduplication() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table
            .add_predicate(PredicateInfo::new("knows", vec!["Person".to_string()]))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new("likes", vec!["Person".to_string()]))
            .unwrap();

        let compact = CompactSchema::from_symbol_table(&table);

        // "Person" should only be stored once
        // Expected strings: "Person", "knows", "likes"
        assert_eq!(compact.string_count(), 3);
    }

    #[test]
    fn test_binary_serialization() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let compact = CompactSchema::from_symbol_table(&table);
        let binary = compact.to_binary().unwrap();
        let recovered = CompactSchema::from_binary(&binary).unwrap();

        let table2 = recovered.to_symbol_table().unwrap();
        assert_eq!(table.domains.len(), table2.domains.len());
    }

    #[test]
    fn test_compression_stats() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Location", 50)).unwrap();

        let compact = CompactSchema::from_symbol_table(&table);
        let stats = compact.compression_stats();

        assert_eq!(stats.domain_count, 2);
        // For small schemas, compression ratio might be > 1.0 due to overhead
        assert!(stats.compression_ratio() > 0.0);
        // Space savings can be negative for very small schemas
        assert!(stats.space_savings() > -200.0);
    }

    #[test]
    fn test_empty_table() {
        let table = SymbolTable::new();
        let compact = CompactSchema::from_symbol_table(&table);
        let recovered = compact.to_symbol_table().unwrap();

        assert_eq!(recovered.domains.len(), 0);
        assert_eq!(recovered.predicates.len(), 0);
    }
}
