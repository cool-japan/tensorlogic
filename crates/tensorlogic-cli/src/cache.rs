//! Persistent compilation cache for TensorLogic
//!
//! This module provides disk-based caching of compiled graphs to speed up repeated compilations.
//! The cache is based on a hash of the expression and compilation context.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tensorlogic_compiler::CompilerContext;
use tensorlogic_ir::{EinsumGraph, TLExpr};

/// Cache entry containing the compiled graph and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The compiled graph
    pub graph: EinsumGraph,
    /// Strategy used for compilation
    pub strategy: String,
    /// Timestamp when cached
    pub timestamp: i64,
    /// Hash of the expression
    pub expr_hash: u64,
}

/// Persistent compilation cache
pub struct CompilationCache {
    /// Cache directory path
    cache_dir: PathBuf,
    /// Maximum cache size in MB
    max_size_mb: usize,
    /// In-memory index of cache entries
    index: HashMap<u64, CacheEntry>,
    /// Whether caching is enabled
    enabled: bool,
}

impl CompilationCache {
    /// Create a new compilation cache
    pub fn new(cache_dir: Option<PathBuf>, max_size_mb: usize) -> Result<Self> {
        let cache_dir = match cache_dir {
            Some(dir) => dir,
            None => Self::default_cache_dir()?,
        };

        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;
        }

        let mut cache = Self {
            cache_dir,
            max_size_mb,
            index: HashMap::new(),
            enabled: true,
        };

        // Load existing cache index
        cache.load_index()?;

        Ok(cache)
    }

    /// Get the default cache directory
    pub fn default_cache_dir() -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .context("Failed to determine cache directory")?
            .join("tensorlogic")
            .join("compilation");
        Ok(cache_dir)
    }

    /// Compute hash for an expression and context
    pub fn compute_hash(expr: &TLExpr, context: &CompilerContext) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the expression (serialized form)
        let expr_str = format!("{:?}", expr);
        expr_str.hash(&mut hasher);

        // Hash the configuration (serialized form for simplicity)
        let config_str = format!("{:?}", context.config);
        config_str.hash(&mut hasher);

        // Hash domain information
        let mut domains: Vec<_> = context.domains.iter().collect();
        domains.sort_by_key(|(name, _)| *name);
        for (name, info) in domains {
            name.hash(&mut hasher);
            // Hash domain cardinality
            info.cardinality.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Get a cached graph if available
    pub fn get(&self, expr: &TLExpr, context: &CompilerContext) -> Option<EinsumGraph> {
        if !self.enabled {
            return None;
        }

        let hash = Self::compute_hash(expr, context);

        if let Some(entry) = self.index.get(&hash) {
            // Verify strategy matches (using debug format)
            let current_strategy = format!("{:?}", context.config);
            if entry.strategy == current_strategy {
                return Some(entry.graph.clone());
            }
        }

        None
    }

    /// Store a compiled graph in the cache
    pub fn put(
        &mut self,
        expr: &TLExpr,
        context: &CompilerContext,
        graph: &EinsumGraph,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let hash = Self::compute_hash(expr, context);

        let entry = CacheEntry {
            graph: graph.clone(),
            strategy: format!("{:?}", context.config), // Use debug format for strategy
            timestamp: chrono::Utc::now().timestamp(),
            expr_hash: hash,
        };

        // Save to disk
        let cache_file = self.cache_dir.join(format!("{:016x}.json", hash));
        let json = serde_json::to_string_pretty(&entry)?;
        fs::write(&cache_file, json)?;

        // Update index
        self.index.insert(hash, entry);

        // Check and enforce cache size limits
        self.enforce_size_limit()?;

        Ok(())
    }

    /// Load the cache index from disk
    fn load_index(&mut self) -> Result<()> {
        if !self.cache_dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Ok(cache_entry) = serde_json::from_str::<CacheEntry>(&content) {
                        self.index.insert(cache_entry.expr_hash, cache_entry);
                    }
                }
            }
        }

        Ok(())
    }

    /// Enforce cache size limits by removing oldest entries
    fn enforce_size_limit(&mut self) -> Result<()> {
        let current_size = self.get_cache_size_mb()?;

        if current_size > self.max_size_mb {
            // Get entries sorted by timestamp (oldest first)
            let mut entries: Vec<_> = self
                .index
                .iter()
                .map(|(hash, entry)| (*hash, entry.timestamp))
                .collect();
            entries.sort_by_key(|(_, timestamp)| *timestamp);

            // Remove oldest entries until we're under the limit
            let target_size = (self.max_size_mb as f64 * 0.8) as usize; // 80% of max

            for (hash, _) in entries {
                if self.get_cache_size_mb()? <= target_size {
                    break;
                }

                self.remove_entry(hash)?;
            }
        }

        Ok(())
    }

    /// Get current cache size in MB
    fn get_cache_size_mb(&self) -> Result<usize> {
        let mut total_bytes = 0u64;

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            total_bytes += entry.metadata()?.len();
        }

        Ok((total_bytes / 1_000_000) as usize)
    }

    /// Remove a cache entry
    fn remove_entry(&mut self, hash: u64) -> Result<()> {
        let cache_file = self.cache_dir.join(format!("{:016x}.json", hash));

        if cache_file.exists() {
            fs::remove_file(cache_file)?;
        }

        self.index.remove(&hash);
        Ok(())
    }

    /// Clear the entire cache
    pub fn clear(&mut self) -> Result<()> {
        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            fs::remove_file(entry.path())?;
        }

        self.index.clear();
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.index.len(),
            size_mb: self.get_cache_size_mb().unwrap_or(0),
            max_size_mb: self.max_size_mb,
            enabled: self.enabled,
            cache_dir: self.cache_dir.clone(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache entries
    pub entries: usize,
    /// Current cache size in MB
    pub size_mb: usize,
    /// Maximum cache size in MB
    pub max_size_mb: usize,
    /// Whether caching is enabled
    pub enabled: bool,
    /// Cache directory path
    pub cache_dir: PathBuf,
}

impl CacheStats {
    /// Print cache statistics
    pub fn print(&self) {
        use crate::output::{print_header, print_info};

        print_header("Cache Statistics");
        print_info(&format!("  Entries: {}", self.entries));
        print_info(&format!(
            "  Size: {} MB / {} MB",
            self.size_mb, self.max_size_mb
        ));
        print_info(&format!(
            "  Enabled: {}",
            if self.enabled { "yes" } else { "no" }
        ));
        print_info(&format!("  Location: {}", self.cache_dir.display()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_compiler::CompilationConfig;
    use tensorlogic_ir::Term;

    #[test]
    fn test_cache_creation() {
        let temp_dir = std::env::temp_dir().join("tensorlogic-test-cache");
        let cache = CompilationCache::new(Some(temp_dir.clone()), 100);
        assert!(cache.is_ok());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_hash_computation() {
        let expr = TLExpr::Pred {
            name: "test".to_string(),
            args: vec![Term::Var("x".to_string())],
        };

        let ctx1 = CompilerContext::with_config(CompilationConfig::soft_differentiable());
        let ctx2 = CompilerContext::with_config(CompilationConfig::hard_boolean());

        let hash1 = CompilationCache::compute_hash(&expr, &ctx1);
        let hash2 = CompilationCache::compute_hash(&expr, &ctx2);

        // Different strategies should produce different hashes
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_cache_put_get() {
        let temp_dir = std::env::temp_dir().join("tensorlogic-test-cache-putget");
        let mut cache = CompilationCache::new(Some(temp_dir.clone()), 100).unwrap();

        let expr = TLExpr::Pred {
            name: "test".to_string(),
            args: vec![Term::Var("x".to_string())],
        };

        let mut ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());
        ctx.add_domain("D", 10);

        // Create a simple graph
        let graph = EinsumGraph::new();

        // Put in cache
        cache.put(&expr, &ctx, &graph).unwrap();

        // Get from cache
        let retrieved = cache.get(&expr, &ctx);
        assert!(retrieved.is_some());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cache_clear() {
        let temp_dir = std::env::temp_dir().join("tensorlogic-test-cache-clear");
        let mut cache = CompilationCache::new(Some(temp_dir.clone()), 100).unwrap();

        let expr = TLExpr::Pred {
            name: "test".to_string(),
            args: vec![Term::Var("x".to_string())],
        };

        let ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());
        let graph = EinsumGraph::new();

        cache.put(&expr, &ctx, &graph).unwrap();
        assert_eq!(cache.stats().entries, 1);

        cache.clear().unwrap();
        assert_eq!(cache.stats().entries, 0);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }
}
