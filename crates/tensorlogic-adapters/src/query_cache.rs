//! Advanced query result caching system for performance optimization.
//!
//! This module provides sophisticated caching mechanisms for expensive query operations,
//! including TTL-based expiration, size limits, and cache statistics tracking.

use crate::{PredicateInfo, SymbolTable};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// A cache key for query results.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheKey {
    /// Query by predicate name
    PredicateByName(String),
    /// Query by arity
    PredicatesByArity(usize),
    /// Query by domain
    PredicatesByDomain(String),
    /// Query by signature
    PredicatesBySignature(Vec<String>),
    /// Query by pattern (wildcard matching)
    PredicatesByPattern(String),
    /// Domain usage count
    DomainUsageCount(String),
    /// All domain names
    AllDomainNames,
    /// All predicate names
    AllPredicateNames,
    /// Custom query key
    Custom(String),
}

/// A cached query result with metadata.
#[derive(Debug, Clone)]
pub struct CachedResult<T> {
    /// The cached value
    pub value: T,
    /// When this entry was created
    pub created_at: Instant,
    /// When this entry was last accessed
    pub last_accessed: Instant,
    /// Number of times this entry has been accessed
    pub access_count: u64,
    /// Time-to-live for this entry
    pub ttl: Option<Duration>,
}

impl<T> CachedResult<T> {
    /// Create a new cached result.
    pub fn new(value: T, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            ttl,
        }
    }

    /// Check if this cached result has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    /// Update access statistics.
    pub fn update_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Get the age of this cache entry.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Configuration for the query cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache
    pub max_entries: usize,
    /// Default time-to-live for cache entries
    pub default_ttl: Option<Duration>,
    /// Whether to enable LRU eviction
    pub enable_lru: bool,
    /// Whether to enable statistics tracking
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            default_ttl: Some(Duration::from_secs(300)), // 5 minutes
            enable_lru: true,
            enable_stats: true,
        }
    }
}

impl CacheConfig {
    /// Create a configuration for a small cache.
    pub fn small() -> Self {
        Self {
            max_entries: 100,
            default_ttl: Some(Duration::from_secs(60)),
            enable_lru: true,
            enable_stats: true,
        }
    }

    /// Create a configuration for a large cache.
    pub fn large() -> Self {
        Self {
            max_entries: 10000,
            default_ttl: Some(Duration::from_secs(600)),
            enable_lru: true,
            enable_stats: true,
        }
    }

    /// Create a configuration with no TTL (cache until evicted).
    pub fn no_ttl() -> Self {
        Self {
            max_entries: 1000,
            default_ttl: None,
            enable_lru: true,
            enable_stats: true,
        }
    }
}

/// Statistics for query cache performance.
#[derive(Debug, Clone, Default)]
pub struct QueryCacheStats {
    /// Total number of cache hits
    pub hits: u64,
    /// Total number of cache misses
    pub misses: u64,
    /// Total number of evictions
    pub evictions: u64,
    /// Total number of expirations
    pub expirations: u64,
    /// Total number of invalidations
    pub invalidations: u64,
}

impl QueryCacheStats {
    /// Calculate the hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate the miss rate (0.0 to 1.0).
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }

    /// Get total number of accesses.
    pub fn total_accesses(&self) -> u64 {
        self.hits + self.misses
    }
}

/// A generic query result cache with TTL and LRU eviction.
pub struct QueryCache<T> {
    /// The cache storage
    cache: HashMap<CacheKey, CachedResult<T>>,
    /// LRU queue for eviction
    lru_queue: VecDeque<CacheKey>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: QueryCacheStats,
}

impl<T: Clone> QueryCache<T> {
    /// Create a new query cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new query cache with custom configuration.
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            cache: HashMap::new(),
            lru_queue: VecDeque::new(),
            config,
            stats: QueryCacheStats::default(),
        }
    }

    /// Get a value from the cache.
    pub fn get(&mut self, key: &CacheKey) -> Option<T> {
        // Check if entry exists and not expired
        let is_expired = self
            .cache
            .get(key)
            .map(|entry| entry.is_expired())
            .unwrap_or(false);

        if is_expired {
            self.cache.remove(key);
            if self.config.enable_stats {
                self.stats.expirations += 1;
                self.stats.misses += 1;
            }
            return None;
        }

        // Get mutable entry and update
        if let Some(entry) = self.cache.get_mut(key) {
            // Update access statistics
            entry.update_access();
            if self.config.enable_stats {
                self.stats.hits += 1;
            }

            let value = entry.value.clone();

            // Update LRU queue if enabled
            if self.config.enable_lru {
                self.update_lru(key);
            }

            Some(value)
        } else {
            if self.config.enable_stats {
                self.stats.misses += 1;
            }
            None
        }
    }

    /// Insert a value into the cache.
    pub fn insert(&mut self, key: CacheKey, value: T) {
        self.insert_with_ttl(key, value, self.config.default_ttl);
    }

    /// Insert a value with a custom TTL.
    pub fn insert_with_ttl(&mut self, key: CacheKey, value: T, ttl: Option<Duration>) {
        // Check if we need to evict
        if self.cache.len() >= self.config.max_entries {
            self.evict_one();
        }

        // Insert the new entry
        let entry = CachedResult::new(value, ttl);
        self.cache.insert(key.clone(), entry);

        // Update LRU queue
        if self.config.enable_lru {
            self.lru_queue.push_back(key);
        }
    }

    /// Invalidate a specific cache entry.
    pub fn invalidate(&mut self, key: &CacheKey) -> bool {
        if self.cache.remove(key).is_some() {
            if self.config.enable_stats {
                self.stats.invalidations += 1;
            }
            // Remove from LRU queue
            if self.config.enable_lru {
                self.lru_queue.retain(|k| k != key);
            }
            true
        } else {
            false
        }
    }

    /// Clear all cache entries.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_queue.clear();
    }

    /// Remove expired entries.
    pub fn cleanup_expired(&mut self) -> usize {
        let mut removed = 0;
        let expired_keys: Vec<CacheKey> = self
            .cache
            .iter()
            .filter(|(_, v)| v.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired_keys {
            self.cache.remove(&key);
            self.lru_queue.retain(|k| k != &key);
            removed += 1;
        }

        if self.config.enable_stats {
            self.stats.expirations += removed as u64;
        }

        removed
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &QueryCacheStats {
        &self.stats
    }

    /// Get the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get the cache configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Update the LRU queue when an entry is accessed.
    fn update_lru(&mut self, key: &CacheKey) {
        // Remove the key from its current position
        self.lru_queue.retain(|k| k != key);
        // Add it to the back (most recently used)
        self.lru_queue.push_back(key.clone());
    }

    /// Evict one entry using LRU strategy.
    fn evict_one(&mut self) {
        if let Some(key) = self.lru_queue.pop_front() {
            self.cache.remove(&key);
            if self.config.enable_stats {
                self.stats.evictions += 1;
            }
        }
    }
}

impl<T: Clone> Default for QueryCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A specialized cache for symbol table queries.
pub struct SymbolTableCache {
    /// Cache for predicate queries
    predicate_cache: QueryCache<Vec<PredicateInfo>>,
    /// Cache for domain name queries
    domain_cache: QueryCache<Vec<String>>,
    /// Cache for scalar results
    scalar_cache: QueryCache<usize>,
}

impl SymbolTableCache {
    /// Create a new symbol table cache.
    pub fn new() -> Self {
        Self {
            predicate_cache: QueryCache::new(),
            domain_cache: QueryCache::new(),
            scalar_cache: QueryCache::new(),
        }
    }

    /// Create a new cache with custom configuration.
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            predicate_cache: QueryCache::with_config(config.clone()),
            domain_cache: QueryCache::with_config(config.clone()),
            scalar_cache: QueryCache::with_config(config),
        }
    }

    /// Get predicates by arity (cached).
    pub fn get_predicates_by_arity(
        &mut self,
        table: &SymbolTable,
        arity: usize,
    ) -> Vec<PredicateInfo> {
        let key = CacheKey::PredicatesByArity(arity);

        if let Some(result) = self.predicate_cache.get(&key) {
            return result;
        }

        // Cache miss - compute and cache
        let result: Vec<PredicateInfo> = table
            .predicates
            .values()
            .filter(|p| p.arg_domains.len() == arity)
            .cloned()
            .collect();

        self.predicate_cache.insert(key, result.clone());
        result
    }

    /// Get predicates using a domain (cached).
    pub fn get_predicates_by_domain(
        &mut self,
        table: &SymbolTable,
        domain: &str,
    ) -> Vec<PredicateInfo> {
        let key = CacheKey::PredicatesByDomain(domain.to_string());

        if let Some(result) = self.predicate_cache.get(&key) {
            return result;
        }

        // Cache miss - compute and cache
        let result: Vec<PredicateInfo> = table
            .predicates
            .values()
            .filter(|p| p.arg_domains.contains(&domain.to_string()))
            .cloned()
            .collect();

        self.predicate_cache.insert(key, result.clone());
        result
    }

    /// Get all domain names (cached).
    pub fn get_domain_names(&mut self, table: &SymbolTable) -> Vec<String> {
        let key = CacheKey::AllDomainNames;

        if let Some(result) = self.domain_cache.get(&key) {
            return result;
        }

        // Cache miss - compute and cache
        let mut result: Vec<String> = table.domains.keys().cloned().collect();
        result.sort();

        self.domain_cache.insert(key, result.clone());
        result
    }

    /// Get domain usage count (cached).
    pub fn get_domain_usage_count(&mut self, table: &SymbolTable, domain: &str) -> usize {
        let key = CacheKey::DomainUsageCount(domain.to_string());

        if let Some(result) = self.scalar_cache.get(&key) {
            return result;
        }

        // Cache miss - compute and cache
        let mut count = 0;
        for predicate in table.predicates.values() {
            count += predicate
                .arg_domains
                .iter()
                .filter(|d| d.as_str() == domain)
                .count();
        }

        for var_domain in table.variables.values() {
            if var_domain == domain {
                count += 1;
            }
        }

        self.scalar_cache.insert(key, count);
        count
    }

    /// Invalidate all caches.
    pub fn invalidate_all(&mut self) {
        self.predicate_cache.clear();
        self.domain_cache.clear();
        self.scalar_cache.clear();
    }

    /// Invalidate caches related to a specific domain.
    pub fn invalidate_domain(&mut self, domain: &str) {
        self.predicate_cache
            .invalidate(&CacheKey::PredicatesByDomain(domain.to_string()));
        self.scalar_cache
            .invalidate(&CacheKey::DomainUsageCount(domain.to_string()));
        self.domain_cache.invalidate(&CacheKey::AllDomainNames);
    }

    /// Invalidate caches related to predicates.
    pub fn invalidate_predicates(&mut self) {
        self.predicate_cache.clear();
    }

    /// Get combined statistics from all caches.
    pub fn combined_stats(&self) -> QueryCacheStats {
        let pred_stats = self.predicate_cache.stats();
        let domain_stats = self.domain_cache.stats();
        let scalar_stats = self.scalar_cache.stats();

        QueryCacheStats {
            hits: pred_stats.hits + domain_stats.hits + scalar_stats.hits,
            misses: pred_stats.misses + domain_stats.misses + scalar_stats.misses,
            evictions: pred_stats.evictions + domain_stats.evictions + scalar_stats.evictions,
            expirations: pred_stats.expirations
                + domain_stats.expirations
                + scalar_stats.expirations,
            invalidations: pred_stats.invalidations
                + domain_stats.invalidations
                + scalar_stats.invalidations,
        }
    }

    /// Cleanup expired entries in all caches.
    pub fn cleanup_expired(&mut self) -> usize {
        self.predicate_cache.cleanup_expired()
            + self.domain_cache.cleanup_expired()
            + self.scalar_cache.cleanup_expired()
    }
}

impl Default for SymbolTableCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DomainInfo;

    #[test]
    fn test_cache_basic_operations() {
        let mut cache: QueryCache<String> = QueryCache::new();
        let key = CacheKey::Custom("test".to_string());

        // Insert and retrieve
        cache.insert(key.clone(), "value".to_string());
        assert_eq!(cache.get(&key), Some("value".to_string()));

        // Stats
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache: QueryCache<String> = QueryCache::new();
        let key = CacheKey::Custom("nonexistent".to_string());

        assert_eq!(cache.get(&key), None);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_invalidation() {
        let mut cache: QueryCache<String> = QueryCache::new();
        let key = CacheKey::Custom("test".to_string());

        cache.insert(key.clone(), "value".to_string());
        assert!(cache.invalidate(&key));
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn test_cache_expiration() {
        let config = CacheConfig {
            default_ttl: Some(Duration::from_millis(10)),
            ..Default::default()
        };
        let mut cache: QueryCache<String> = QueryCache::with_config(config);
        let key = CacheKey::Custom("test".to_string());

        cache.insert(key.clone(), "value".to_string());
        std::thread::sleep(Duration::from_millis(20));

        // Should be expired
        assert_eq!(cache.get(&key), None);
        assert_eq!(cache.stats().expirations, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            enable_lru: true,
            ..Default::default()
        };
        let mut cache: QueryCache<String> = QueryCache::with_config(config);

        cache.insert(CacheKey::Custom("key1".to_string()), "value1".to_string());
        cache.insert(CacheKey::Custom("key2".to_string()), "value2".to_string());
        cache.insert(CacheKey::Custom("key3".to_string()), "value3".to_string());

        // key1 should have been evicted
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_symbol_table_cache() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "knows",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new("age", vec!["Person".to_string()]))
            .unwrap();

        let mut cache = SymbolTableCache::new();

        // First call - cache miss
        let predicates = cache.get_predicates_by_arity(&table, 2);
        assert_eq!(predicates.len(), 1);
        assert_eq!(cache.predicate_cache.stats().misses, 1);

        // Second call - cache hit
        let predicates = cache.get_predicates_by_arity(&table, 2);
        assert_eq!(predicates.len(), 1);
        assert_eq!(cache.predicate_cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_config_presets() {
        let small = CacheConfig::small();
        assert_eq!(small.max_entries, 100);

        let large = CacheConfig::large();
        assert_eq!(large.max_entries, 10000);

        let no_ttl = CacheConfig::no_ttl();
        assert!(no_ttl.default_ttl.is_none());
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: QueryCache<String> = QueryCache::new();
        let key1 = CacheKey::Custom("key1".to_string());
        let key2 = CacheKey::Custom("key2".to_string());

        cache.insert(key1.clone(), "value1".to_string());
        cache.get(&key1); // hit
        cache.get(&key2); // miss

        let stats = cache.stats();
        assert_eq!(stats.hit_rate(), 0.5);
        assert_eq!(stats.miss_rate(), 0.5);
        assert_eq!(stats.total_accesses(), 2);
    }

    #[test]
    fn test_cleanup_expired() {
        let config = CacheConfig {
            default_ttl: Some(Duration::from_millis(10)),
            ..Default::default()
        };
        let mut cache: QueryCache<String> = QueryCache::with_config(config);

        cache.insert(CacheKey::Custom("key1".to_string()), "value1".to_string());
        cache.insert(CacheKey::Custom("key2".to_string()), "value2".to_string());

        std::thread::sleep(Duration::from_millis(20));

        let removed = cache.cleanup_expired();
        assert_eq!(removed, 2);
        assert!(cache.is_empty());
    }
}
