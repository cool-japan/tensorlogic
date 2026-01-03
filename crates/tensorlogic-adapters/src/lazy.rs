//! Lazy loading support for huge symbol tables.
//!
//! This module provides on-demand loading of domains, predicates, and other
//! schema elements, enabling efficient handling of very large schemas that
//! don't fit comfortably in memory.
//!
//! # Examples
//!
//! ```rust
//! use tensorlogic_adapters::{LazySymbolTable, FileSchemaLoader};
//! use std::sync::Arc;
//!
//! // Create a lazy symbol table with on-demand loading
//! let loader = Arc::new(FileSchemaLoader::new("/tmp/schema"));
//! let lazy_table = LazySymbolTable::new(loader);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::{DomainInfo, PredicateInfo, SymbolTable};

/// Strategy for loading schema elements.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LoadStrategy {
    /// Load all elements eagerly (no lazy loading).
    Eager,
    /// Load elements on first access.
    #[default]
    OnDemand,
    /// Preload frequently accessed elements.
    Predictive {
        /// Threshold for "frequent" access.
        access_threshold: usize,
    },
    /// Load elements in batches.
    Batched {
        /// Batch size for loading.
        batch_size: usize,
    },
}

/// A loader trait for fetching schema elements on demand.
///
/// Implementations can load from files, databases, or remote services.
pub trait SchemaLoader: Send + Sync {
    /// Load a domain by name.
    fn load_domain(&self, name: &str) -> Result<DomainInfo>;

    /// Load a predicate by name.
    fn load_predicate(&self, name: &str) -> Result<PredicateInfo>;

    /// Check if a domain exists without loading it.
    fn has_domain(&self, name: &str) -> bool;

    /// Check if a predicate exists without loading it.
    fn has_predicate(&self, name: &str) -> bool;

    /// List all available domain names.
    fn list_domains(&self) -> Result<Vec<String>>;

    /// List all available predicate names.
    fn list_predicates(&self) -> Result<Vec<String>>;

    /// Load a batch of domains by name.
    fn load_domains_batch(&self, names: &[String]) -> Result<Vec<DomainInfo>> {
        names.iter().map(|n| self.load_domain(n)).collect()
    }

    /// Load a batch of predicates by name.
    fn load_predicates_batch(&self, names: &[String]) -> Result<Vec<PredicateInfo>> {
        names.iter().map(|n| self.load_predicate(n)).collect()
    }
}

/// File-based schema loader that reads from a directory structure.
///
/// Expected directory layout:
/// ```text
/// schema_dir/
///   domains/
///     domain1.json
///     domain2.json
///   predicates/
///     pred1.json
///     pred2.json
/// ```
#[derive(Clone, Debug)]
pub struct FileSchemaLoader {
    /// Base directory for schema files.
    base_dir: PathBuf,
}

impl FileSchemaLoader {
    /// Create a new file-based schema loader.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::FileSchemaLoader;
    ///
    /// let loader = FileSchemaLoader::new("/path/to/schema");
    /// ```
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    fn domain_path(&self, name: &str) -> PathBuf {
        self.base_dir.join("domains").join(format!("{}.json", name))
    }

    fn predicate_path(&self, name: &str) -> PathBuf {
        self.base_dir
            .join("predicates")
            .join(format!("{}.json", name))
    }
}

impl SchemaLoader for FileSchemaLoader {
    fn load_domain(&self, name: &str) -> Result<DomainInfo> {
        let path = self.domain_path(name);
        let content = std::fs::read_to_string(path)?;
        let domain: DomainInfo = serde_json::from_str(&content)?;
        Ok(domain)
    }

    fn load_predicate(&self, name: &str) -> Result<PredicateInfo> {
        let path = self.predicate_path(name);
        let content = std::fs::read_to_string(path)?;
        let predicate: PredicateInfo = serde_json::from_str(&content)?;
        Ok(predicate)
    }

    fn has_domain(&self, name: &str) -> bool {
        self.domain_path(name).exists()
    }

    fn has_predicate(&self, name: &str) -> bool {
        self.predicate_path(name).exists()
    }

    fn list_domains(&self) -> Result<Vec<String>> {
        let domains_dir = self.base_dir.join("domains");
        if !domains_dir.exists() {
            return Ok(Vec::new());
        }

        let mut names = Vec::new();
        for entry in std::fs::read_dir(domains_dir)? {
            let entry = entry?;
            if let Some(name) = entry.path().file_stem() {
                names.push(name.to_string_lossy().to_string());
            }
        }
        Ok(names)
    }

    fn list_predicates(&self) -> Result<Vec<String>> {
        let predicates_dir = self.base_dir.join("predicates");
        if !predicates_dir.exists() {
            return Ok(Vec::new());
        }

        let mut names = Vec::new();
        for entry in std::fs::read_dir(predicates_dir)? {
            let entry = entry?;
            if let Some(name) = entry.path().file_stem() {
                names.push(name.to_string_lossy().to_string());
            }
        }
        Ok(names)
    }
}

/// Statistics about lazy loading behavior.
#[derive(Clone, Debug, Default)]
pub struct LazyLoadStats {
    /// Number of domain loads.
    pub domain_loads: usize,
    /// Number of predicate loads.
    pub predicate_loads: usize,
    /// Number of cache hits.
    pub cache_hits: usize,
    /// Number of cache misses.
    pub cache_misses: usize,
    /// Number of batch loads.
    pub batch_loads: usize,
}

impl LazyLoadStats {
    /// Get the cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// A symbol table with lazy loading support.
///
/// Elements are loaded on-demand from a SchemaLoader, reducing memory
/// usage for large schemas.
pub struct LazySymbolTable {
    /// Eagerly loaded symbol table (acts as cache).
    loaded: Arc<RwLock<SymbolTable>>,
    /// Loader for on-demand fetching.
    loader: Arc<dyn SchemaLoader>,
    /// Loading strategy.
    strategy: LoadStrategy,
    /// Statistics.
    stats: Arc<RwLock<LazyLoadStats>>,
    /// Set of loaded domain names.
    loaded_domains: Arc<RwLock<HashSet<String>>>,
    /// Set of loaded predicate names.
    loaded_predicates: Arc<RwLock<HashSet<String>>>,
    /// Access counts for predictive loading.
    access_counts: Arc<RwLock<HashMap<String, usize>>>,
}

impl LazySymbolTable {
    /// Create a new lazy symbol table.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tensorlogic_adapters::{LazySymbolTable, FileSchemaLoader};
    /// use std::sync::Arc;
    ///
    /// let loader = Arc::new(FileSchemaLoader::new("/tmp/schema"));
    /// let lazy_table = LazySymbolTable::new(loader);
    /// ```
    pub fn new(loader: Arc<dyn SchemaLoader>) -> Self {
        Self {
            loaded: Arc::new(RwLock::new(SymbolTable::new())),
            loader,
            strategy: LoadStrategy::default(),
            stats: Arc::new(RwLock::new(LazyLoadStats::default())),
            loaded_domains: Arc::new(RwLock::new(HashSet::new())),
            loaded_predicates: Arc::new(RwLock::new(HashSet::new())),
            access_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a lazy table with a specific loading strategy.
    pub fn with_strategy(loader: Arc<dyn SchemaLoader>, strategy: LoadStrategy) -> Self {
        let mut table = Self::new(loader);
        table.strategy = strategy;
        table
    }

    /// Get a domain, loading it if necessary.
    pub fn get_domain(&self, name: &str) -> Result<Option<DomainInfo>> {
        // Check if already loaded
        {
            let loaded_set = self.loaded_domains.read().unwrap();
            if loaded_set.contains(name) {
                let table = self.loaded.read().unwrap();
                let mut stats = self.stats.write().unwrap();
                stats.cache_hits += 1;
                return Ok(table.get_domain(name).cloned());
            }
        }

        // Check if domain exists
        if !self.loader.has_domain(name) {
            let mut stats = self.stats.write().unwrap();
            stats.cache_misses += 1;
            return Ok(None);
        }

        // Load domain
        self.load_domain_internal(name)?;

        let table = self.loaded.read().unwrap();
        Ok(table.get_domain(name).cloned())
    }

    /// Get a predicate, loading it if necessary.
    pub fn get_predicate(&self, name: &str) -> Result<Option<PredicateInfo>> {
        // Check if already loaded
        {
            let loaded_set = self.loaded_predicates.read().unwrap();
            if loaded_set.contains(name) {
                let table = self.loaded.read().unwrap();
                let mut stats = self.stats.write().unwrap();
                stats.cache_hits += 1;
                return Ok(table.get_predicate(name).cloned());
            }
        }

        // Check if predicate exists
        if !self.loader.has_predicate(name) {
            let mut stats = self.stats.write().unwrap();
            stats.cache_misses += 1;
            return Ok(None);
        }

        // Load predicate
        self.load_predicate_internal(name)?;

        let table = self.loaded.read().unwrap();
        Ok(table.get_predicate(name).cloned())
    }

    /// List all available domains (without loading them).
    pub fn list_domains(&self) -> Result<Vec<String>> {
        self.loader.list_domains()
    }

    /// List all available predicates (without loading them).
    pub fn list_predicates(&self) -> Result<Vec<String>> {
        self.loader.list_predicates()
    }

    /// Preload a batch of domains.
    pub fn preload_domains(&self, names: &[String]) -> Result<()> {
        let domains = self.loader.load_domains_batch(names)?;
        let mut table = self.loaded.write().unwrap();
        let mut loaded_set = self.loaded_domains.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        for domain in domains {
            let name = domain.name.clone();
            table.add_domain(domain).map_err(|e| anyhow::anyhow!(e))?;
            loaded_set.insert(name);
            stats.domain_loads += 1;
        }
        stats.batch_loads += 1;

        Ok(())
    }

    /// Preload a batch of predicates.
    pub fn preload_predicates(&self, names: &[String]) -> Result<()> {
        let predicates = self.loader.load_predicates_batch(names)?;
        let mut table = self.loaded.write().unwrap();
        let mut loaded_set = self.loaded_predicates.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        for predicate in predicates {
            let name = predicate.name.clone();
            table
                .add_predicate(predicate)
                .map_err(|e| anyhow::anyhow!(e))?;
            loaded_set.insert(name);
            stats.predicate_loads += 1;
        }
        stats.batch_loads += 1;

        Ok(())
    }

    /// Get loading statistics.
    pub fn stats(&self) -> LazyLoadStats {
        self.stats.read().unwrap().clone()
    }

    /// Clear the cache and force reload.
    pub fn clear_cache(&self) {
        let mut table = self.loaded.write().unwrap();
        *table = SymbolTable::new();
        self.loaded_domains.write().unwrap().clear();
        self.loaded_predicates.write().unwrap().clear();
        self.access_counts.write().unwrap().clear();
    }

    /// Get the number of loaded domains.
    pub fn loaded_domain_count(&self) -> usize {
        self.loaded_domains.read().unwrap().len()
    }

    /// Get the number of loaded predicates.
    pub fn loaded_predicate_count(&self) -> usize {
        self.loaded_predicates.read().unwrap().len()
    }

    /// Get a read-only reference to the loaded symbol table.
    ///
    /// Note: This only includes loaded elements.
    pub fn as_symbol_table(&self) -> Arc<RwLock<SymbolTable>> {
        Arc::clone(&self.loaded)
    }

    fn load_domain_internal(&self, name: &str) -> Result<()> {
        let domain = self.loader.load_domain(name)?;
        let mut table = self.loaded.write().unwrap();
        let mut loaded_set = self.loaded_domains.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        table.add_domain(domain).map_err(|e| anyhow::anyhow!(e))?;
        loaded_set.insert(name.to_string());
        stats.domain_loads += 1;
        stats.cache_misses += 1;

        // Track access for predictive loading
        {
            let mut counts = self.access_counts.write().unwrap();
            *counts.entry(name.to_string()).or_insert(0) += 1;
        }

        Ok(())
    }

    fn load_predicate_internal(&self, name: &str) -> Result<()> {
        let predicate = self.loader.load_predicate(name)?;
        let mut table = self.loaded.write().unwrap();
        let mut loaded_set = self.loaded_predicates.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        table
            .add_predicate(predicate)
            .map_err(|e| anyhow::anyhow!(e))?;
        loaded_set.insert(name.to_string());
        stats.predicate_loads += 1;
        stats.cache_misses += 1;

        // Track access for predictive loading
        {
            let mut counts = self.access_counts.write().unwrap();
            *counts.entry(name.to_string()).or_insert(0) += 1;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Mock loader for testing
    struct MockLoader {
        domains: HashMap<String, DomainInfo>,
        predicates: HashMap<String, PredicateInfo>,
    }

    impl MockLoader {
        fn new() -> Self {
            let mut domains = HashMap::new();
            domains.insert("Person".to_string(), DomainInfo::new("Person", 100));
            domains.insert("Location".to_string(), DomainInfo::new("Location", 50));

            let mut predicates = HashMap::new();
            predicates.insert(
                "at".to_string(),
                PredicateInfo::new("at", vec!["Person".to_string(), "Location".to_string()]),
            );

            Self {
                domains,
                predicates,
            }
        }
    }

    impl SchemaLoader for MockLoader {
        fn load_domain(&self, name: &str) -> Result<DomainInfo> {
            self.domains
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Domain not found: {}", name))
        }

        fn load_predicate(&self, name: &str) -> Result<PredicateInfo> {
            self.predicates
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Predicate not found: {}", name))
        }

        fn has_domain(&self, name: &str) -> bool {
            self.domains.contains_key(name)
        }

        fn has_predicate(&self, name: &str) -> bool {
            self.predicates.contains_key(name)
        }

        fn list_domains(&self) -> Result<Vec<String>> {
            Ok(self.domains.keys().cloned().collect())
        }

        fn list_predicates(&self) -> Result<Vec<String>> {
            Ok(self.predicates.keys().cloned().collect())
        }
    }

    #[test]
    fn test_lazy_load_domain() {
        let loader = Arc::new(MockLoader::new());
        let lazy_table = LazySymbolTable::new(loader);

        let domain = lazy_table.get_domain("Person").unwrap();
        assert!(domain.is_some());
        assert_eq!(domain.unwrap().name, "Person");
    }

    #[test]
    fn test_lazy_load_predicate() {
        let loader = Arc::new(MockLoader::new());
        let lazy_table = LazySymbolTable::new(loader);

        // First load predicates' domains
        lazy_table.get_domain("Person").unwrap();
        lazy_table.get_domain("Location").unwrap();

        let predicate = lazy_table.get_predicate("at").unwrap();
        assert!(predicate.is_some());
        assert_eq!(predicate.unwrap().name, "at");
    }

    #[test]
    fn test_cache_hits() {
        let loader = Arc::new(MockLoader::new());
        let lazy_table = LazySymbolTable::new(loader);

        // First access (miss)
        lazy_table.get_domain("Person").unwrap();

        // Second access (hit)
        lazy_table.get_domain("Person").unwrap();

        let stats = lazy_table.stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_list_domains() {
        let loader = Arc::new(MockLoader::new());
        let lazy_table = LazySymbolTable::new(loader);

        let domains = lazy_table.list_domains().unwrap();
        assert_eq!(domains.len(), 2);
        assert!(domains.contains(&"Person".to_string()));
        assert!(domains.contains(&"Location".to_string()));
    }

    #[test]
    fn test_preload_domains() {
        let loader = Arc::new(MockLoader::new());
        let lazy_table = LazySymbolTable::new(loader);

        let names = vec!["Person".to_string(), "Location".to_string()];
        lazy_table.preload_domains(&names).unwrap();

        assert_eq!(lazy_table.loaded_domain_count(), 2);

        let stats = lazy_table.stats();
        assert_eq!(stats.batch_loads, 1);
    }

    #[test]
    fn test_clear_cache() {
        let loader = Arc::new(MockLoader::new());
        let lazy_table = LazySymbolTable::new(loader);

        lazy_table.get_domain("Person").unwrap();
        assert_eq!(lazy_table.loaded_domain_count(), 1);

        lazy_table.clear_cache();
        assert_eq!(lazy_table.loaded_domain_count(), 0);
    }

    #[test]
    fn test_load_strategy() {
        let loader = Arc::new(MockLoader::new());
        let strategy = LoadStrategy::Predictive {
            access_threshold: 5,
        };
        let lazy_table = LazySymbolTable::with_strategy(loader, strategy);

        lazy_table.get_domain("Person").unwrap();
        assert_eq!(lazy_table.loaded_domain_count(), 1);
    }

    #[test]
    fn test_hit_rate() {
        let mut stats = LazyLoadStats::default();
        assert_eq!(stats.hit_rate(), 0.0);

        stats.cache_hits = 8;
        stats.cache_misses = 2;
        assert!((stats.hit_rate() - 0.8).abs() < 0.01);
    }
}
