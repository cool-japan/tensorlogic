//! Compilation caching for improved performance.
//!
//! This module provides a caching layer for compiled expressions, reducing
//! redundant compilation when the same expressions are compiled multiple times.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, TLExpr};

use crate::config::CompilationConfig;
use crate::CompilerContext;

/// A hash key for caching compiled expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    /// Expression hash
    expr_hash: u64,
    /// Configuration hash
    config_hash: u64,
    /// Domain information hash
    domain_hash: u64,
}

impl CacheKey {
    /// Create a new cache key from an expression, configuration, and context.
    fn new(expr: &TLExpr, config: &CompilationConfig, ctx: &CompilerContext) -> Self {
        use std::collections::hash_map::DefaultHasher;

        // Hash the expression
        let mut expr_hasher = DefaultHasher::new();
        format!("{:?}", expr).hash(&mut expr_hasher);
        let expr_hash = expr_hasher.finish();

        // Hash the configuration
        let mut config_hasher = DefaultHasher::new();
        format!("{:?}", config).hash(&mut config_hasher);
        let config_hash = config_hasher.finish();

        // Hash the domains
        let mut domain_hasher = DefaultHasher::new();
        for (name, domain) in &ctx.domains {
            name.hash(&mut domain_hasher);
            domain.cardinality.hash(&mut domain_hasher);
        }
        let domain_hash = domain_hasher.finish();

        CacheKey {
            expr_hash,
            config_hash,
            domain_hash,
        }
    }
}

/// Cached compilation result.
#[derive(Clone)]
struct CachedResult {
    /// The compiled graph
    graph: EinsumGraph,
    /// Number of times this entry has been hit
    hit_count: usize,
}

/// Compilation cache for storing and retrieving compiled expressions.
///
/// The cache is thread-safe and can be shared across multiple compilation
/// operations. It automatically evicts least-recently-used entries when
/// the cache reaches its maximum size.
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::{CompilationCache, compile_to_einsum_with_context, CompilerContext};
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let cache = CompilationCache::new(100); // Cache up to 100 entries
/// let mut ctx = CompilerContext::new();
/// ctx.add_domain("Person", 100);
///
/// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
///
/// // First compilation: miss (not in cache)
/// let graph1 = cache.get_or_compile(&expr, &mut ctx, |expr, ctx| {
///     compile_to_einsum_with_context(expr, ctx)
/// }).unwrap();
///
/// // Second compilation: hit (cached)
/// let graph2 = cache.get_or_compile(&expr, &mut ctx, |expr, ctx| {
///     compile_to_einsum_with_context(expr, ctx)
/// }).unwrap();
///
/// assert_eq!(graph1, graph2);
/// assert_eq!(cache.stats().hits, 1);
/// ```
pub struct CompilationCache {
    /// Cache storage
    cache: Arc<Mutex<HashMap<CacheKey, CachedResult>>>,
    /// Maximum cache size
    max_size: usize,
    /// Cache statistics
    stats: Arc<Mutex<CacheStats>>,
}

/// Statistics about cache performance.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of entries evicted
    pub evictions: u64,
    /// Current cache size
    pub current_size: usize,
}

impl CacheStats {
    /// Calculate the hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate the total number of lookups.
    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses
    }
}

impl CompilationCache {
    /// Create a new compilation cache with the specified maximum size.
    ///
    /// # Arguments
    ///
    /// * `max_size` - Maximum number of entries to cache (default: 1000)
    ///
    /// # Example
    ///
    /// ```
    /// use tensorlogic_compiler::CompilationCache;
    ///
    /// let cache = CompilationCache::new(100);
    /// assert_eq!(cache.max_size(), 100);
    /// ```
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Create a new cache with default settings (1000 entries).
    pub fn default_size() -> Self {
        Self::new(1000)
    }

    /// Get the maximum cache size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get or compile an expression.
    ///
    /// If the expression is in the cache, returns the cached result.
    /// Otherwise, compiles the expression using the provided function
    /// and caches the result.
    ///
    /// # Arguments
    ///
    /// * `expr` - The expression to compile
    /// * `ctx` - The compiler context
    /// * `compile_fn` - Function to compile the expression if not cached
    ///
    /// # Example
    ///
    /// ```
    /// use tensorlogic_compiler::{CompilationCache, compile_to_einsum_with_context, CompilerContext};
    /// use tensorlogic_ir::{TLExpr, Term};
    ///
    /// let cache = CompilationCache::new(100);
    /// let mut ctx = CompilerContext::new();
    /// ctx.add_domain("Person", 100);
    ///
    /// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    ///
    /// let graph = cache.get_or_compile(&expr, &mut ctx, |expr, ctx| {
    ///     compile_to_einsum_with_context(expr, ctx)
    /// }).unwrap();
    /// ```
    pub fn get_or_compile<F>(
        &self,
        expr: &TLExpr,
        ctx: &mut CompilerContext,
        compile_fn: F,
    ) -> Result<EinsumGraph>
    where
        F: FnOnce(&TLExpr, &mut CompilerContext) -> Result<EinsumGraph>,
    {
        let key = CacheKey::new(expr, &ctx.config, ctx);

        // Try to get from cache
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get_mut(&key) {
                // Cache hit
                cached.hit_count += 1;
                let mut stats = self.stats.lock().unwrap();
                stats.hits += 1;
                return Ok(cached.graph.clone());
            }
        }

        // Cache miss - compile
        let mut stats = self.stats.lock().unwrap();
        stats.misses += 1;
        drop(stats);

        let graph = compile_fn(expr, ctx)?;

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();

            // Evict if necessary
            if cache.len() >= self.max_size {
                // Find least-used entry
                let min_key = cache
                    .iter()
                    .min_by_key(|(_, v)| v.hit_count)
                    .map(|(k, _)| k.clone());

                if let Some(key_to_evict) = min_key {
                    cache.remove(&key_to_evict);
                    let mut stats = self.stats.lock().unwrap();
                    stats.evictions += 1;
                }
            }

            cache.insert(
                key,
                CachedResult {
                    graph: graph.clone(),
                    hit_count: 0,
                },
            );

            let mut stats = self.stats.lock().unwrap();
            stats.current_size = cache.len();
        }

        Ok(graph)
    }

    /// Get current cache statistics.
    ///
    /// # Example
    ///
    /// ```
    /// use tensorlogic_compiler::CompilationCache;
    ///
    /// let cache = CompilationCache::new(100);
    /// let stats = cache.stats();
    /// assert_eq!(stats.hits, 0);
    /// assert_eq!(stats.misses, 0);
    /// ```
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use tensorlogic_compiler::CompilationCache;
    ///
    /// let cache = CompilationCache::new(100);
    /// cache.clear();
    /// assert_eq!(cache.stats().current_size, 0);
    /// ```
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
        let mut stats = self.stats.lock().unwrap();
        stats.current_size = 0;
    }

    /// Get the current number of entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for CompilationCache {
    fn default() -> Self {
        Self::default_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_to_einsum_with_context;
    use tensorlogic_ir::Term;

    #[test]
    fn test_cache_new() {
        let cache = CompilationCache::new(100);
        assert_eq!(cache.max_size(), 100);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_hit() {
        let cache = CompilationCache::new(100);
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

        // First compilation: miss
        let graph1 = cache
            .get_or_compile(&expr, &mut ctx, compile_to_einsum_with_context)
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Second compilation: hit
        let graph2 = cache
            .get_or_compile(&expr, &mut ctx, compile_to_einsum_with_context)
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.hit_rate(), 0.5);

        // Graphs should be identical
        assert_eq!(graph1, graph2);
    }

    #[test]
    fn test_cache_different_expressions() {
        let cache = CompilationCache::new(100);
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr1 = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let expr2 = TLExpr::pred("likes", vec![Term::var("x"), Term::var("y")]);

        // Compile both
        let _graph1 = cache
            .get_or_compile(&expr1, &mut ctx, |e, c| {
                compile_to_einsum_with_context(e, c)
            })
            .unwrap();
        let _graph2 = cache
            .get_or_compile(&expr2, &mut ctx, |e, c| {
                compile_to_einsum_with_context(e, c)
            })
            .unwrap();

        // Both should be misses
        let stats = cache.stats();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 0);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = CompilationCache::new(2); // Small cache
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr1 = TLExpr::pred("p1", vec![Term::var("x")]);
        let expr2 = TLExpr::pred("p2", vec![Term::var("x")]);
        let expr3 = TLExpr::pred("p3", vec![Term::var("x")]);

        // Compile three expressions (should evict one)
        let _ = cache.get_or_compile(&expr1, &mut ctx, |e, c| {
            compile_to_einsum_with_context(e, c)
        });
        let _ = cache.get_or_compile(&expr2, &mut ctx, |e, c| {
            compile_to_einsum_with_context(e, c)
        });
        let _ = cache.get_or_compile(&expr3, &mut ctx, |e, c| {
            compile_to_einsum_with_context(e, c)
        });

        // Should have evicted one entry
        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_clear() {
        let cache = CompilationCache::new(100);
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);

        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

        // Compile and cache
        let _ = cache.get_or_compile(&expr, &mut ctx, compile_to_einsum_with_context);

        assert_eq!(cache.len(), 1);

        // Clear
        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_stats() {
        let cache = CompilationCache::new(100);
        let stats = cache.stats();

        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.current_size, 0);
        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.total_lookups(), 0);
    }
}
