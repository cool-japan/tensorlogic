//! Persistent compilation cache for TensorLogic
//!
//! This module provides disk-based caching of compiled graphs to speed up repeated compilations.
//! The cache uses LRU (Least Recently Used) eviction policy with compression support.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tensorlogic_compiler::CompilerContext;
use tensorlogic_ir::{EinsumGraph, TLExpr};

/// Cache entry containing the compiled graph and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The compiled graph
    pub graph: EinsumGraph,
    /// Strategy used for compilation
    pub strategy: String,
    /// Timestamp when created
    pub timestamp: i64,
    /// Timestamp when last accessed
    pub last_accessed: i64,
    /// Hash of the expression
    pub expr_hash: u64,
    /// Number of times accessed
    pub access_count: u64,
}

/// Persistent compilation cache with LRU eviction
pub struct CompilationCache {
    /// Cache directory path
    cache_dir: PathBuf,
    /// Maximum cache size in MB
    max_size_mb: usize,
    /// In-memory index of cache entries
    index: HashMap<u64, CacheEntry>,
    /// Whether caching is enabled
    enabled: bool,
    /// Cache hit count
    hits: u64,
    /// Cache miss count
    misses: u64,
    /// Number of entries evicted
    evictions: u64,
    /// Whether to use compression
    use_compression: bool,
}

impl CompilationCache {
    /// Create a new compilation cache with LRU eviction and compression
    pub fn new(cache_dir: Option<PathBuf>, max_size_mb: usize) -> Result<Self> {
        Self::with_compression(cache_dir, max_size_mb, true)
    }

    /// Create a new compilation cache with optional compression
    pub fn with_compression(
        cache_dir: Option<PathBuf>,
        max_size_mb: usize,
        use_compression: bool,
    ) -> Result<Self> {
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
            hits: 0,
            misses: 0,
            evictions: 0,
            use_compression,
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

    /// Get a cached graph if available (updates LRU access time)
    pub fn get(&mut self, expr: &TLExpr, context: &CompilerContext) -> Option<EinsumGraph> {
        if !self.enabled {
            return None;
        }

        let hash = Self::compute_hash(expr, context);

        if let Some(entry) = self.index.get_mut(&hash) {
            // Verify strategy matches (using debug format)
            let current_strategy = format!("{:?}", context.config);
            if entry.strategy == current_strategy {
                // Update LRU statistics
                entry.last_accessed = chrono::Utc::now().timestamp();
                entry.access_count += 1;
                self.hits += 1;

                // Clone entry data for async update
                let entry_clone = entry.clone();
                let graph = entry.graph.clone();

                // Update the entry on disk asynchronously (best effort)
                let _ = self.update_entry_metadata(&entry_clone);

                return Some(graph);
            }
        }

        self.misses += 1;
        None
    }

    /// Update entry metadata on disk
    fn update_entry_metadata(&self, entry: &CacheEntry) -> Result<()> {
        if self.use_compression {
            let compressed = Self::compress_entry(entry)?;
            let cache_file = self.cache_dir.join(format!("{:016x}.bin", entry.expr_hash));
            fs::write(&cache_file, compressed)?;
        } else {
            let cache_file = self
                .cache_dir
                .join(format!("{:016x}.json", entry.expr_hash));
            let json = serde_json::to_string(entry)?;
            fs::write(&cache_file, json)?;
        }

        Ok(())
    }

    /// Compress a cache entry using gzip
    fn compress_entry(entry: &CacheEntry) -> Result<Vec<u8>> {
        use std::io::Write;

        // Serialize to JSON first
        let json = serde_json::to_vec(entry).context("Failed to serialize entry")?;

        // Compress with flate2 (gzip)
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        encoder.write_all(&json).context("Failed to compress")?;
        let compressed = encoder.finish().context("Failed to finish compression")?;

        Ok(compressed)
    }

    /// Decompress a cache entry
    fn decompress_entry(compressed: &[u8]) -> Result<CacheEntry> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        // Decompress
        let mut decoder = GzDecoder::new(compressed);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .context("Failed to decompress")?;

        // Deserialize from JSON
        let entry: CacheEntry =
            serde_json::from_slice(&decompressed).context("Failed to deserialize entry")?;

        Ok(entry)
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
        let now = chrono::Utc::now().timestamp();

        let entry = CacheEntry {
            graph: graph.clone(),
            strategy: format!("{:?}", context.config),
            timestamp: now,
            last_accessed: now,
            expr_hash: hash,
            access_count: 0,
        };

        // Save to disk with optional compression
        if self.use_compression {
            let compressed = Self::compress_entry(&entry)?;
            let cache_file = self.cache_dir.join(format!("{:016x}.bin", hash));
            fs::write(&cache_file, compressed)?;
        } else {
            let cache_file = self.cache_dir.join(format!("{:016x}.json", hash));
            let json = serde_json::to_string_pretty(&entry)?;
            fs::write(&cache_file, json)?;
        }

        // Update index
        self.index.insert(hash, entry);

        // Check and enforce cache size limits with LRU eviction
        self.enforce_size_limit()?;

        Ok(())
    }

    /// Load the cache index from disk (supports both JSON and compressed formats)
    fn load_index(&mut self) -> Result<()> {
        if !self.cache_dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            let ext = path.extension().and_then(|s| s.to_str());

            match ext {
                Some("json") => {
                    // Load uncompressed JSON format
                    if let Ok(content) = fs::read_to_string(&path) {
                        if let Ok(cache_entry) = serde_json::from_str::<CacheEntry>(&content) {
                            self.index.insert(cache_entry.expr_hash, cache_entry);
                        }
                    }
                }
                Some("bin") => {
                    // Load compressed binary format
                    if let Ok(content) = fs::read(&path) {
                        if let Ok(cache_entry) = Self::decompress_entry(&content) {
                            self.index.insert(cache_entry.expr_hash, cache_entry);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Enforce cache size limits using LRU eviction (removes least recently used)
    fn enforce_size_limit(&mut self) -> Result<()> {
        let current_size = self.get_cache_size_mb()?;

        if current_size > self.max_size_mb {
            // Get entries sorted by last_accessed (least recently used first)
            let mut entries: Vec<_> = self
                .index
                .iter()
                .map(|(hash, entry)| (*hash, entry.last_accessed, entry.access_count))
                .collect();

            // Sort by last_accessed (oldest first), then by access_count (least used first)
            entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)));

            // Remove LRU entries until we're under the limit
            let target_size = (self.max_size_mb as f64 * 0.8) as usize; // 80% of max

            for (hash, _, _) in entries {
                if self.get_cache_size_mb()? <= target_size {
                    break;
                }

                self.remove_entry(hash)?;
                self.evictions += 1;
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

    /// Remove a cache entry (handles both JSON and binary formats)
    fn remove_entry(&mut self, hash: u64) -> Result<()> {
        // Try removing both formats
        let json_file = self.cache_dir.join(format!("{:016x}.json", hash));
        let bin_file = self.cache_dir.join(format!("{:016x}.bin", hash));

        if json_file.exists() {
            fs::remove_file(json_file)?;
        }
        if bin_file.exists() {
            fs::remove_file(bin_file)?;
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
        let hit_rate = if self.hits + self.misses > 0 {
            (self.hits as f64 / (self.hits + self.misses) as f64) * 100.0
        } else {
            0.0
        };

        CacheStats {
            entries: self.index.len(),
            size_mb: self.get_cache_size_mb().unwrap_or(0),
            max_size_mb: self.max_size_mb,
            enabled: self.enabled,
            cache_dir: self.cache_dir.clone(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            hit_rate,
            compression_enabled: self.use_compression,
        }
    }

    /// Warm up the cache by precompiling a list of expressions
    ///
    /// This is useful for frequently used expressions that should be cached on startup.
    /// Returns the number of successfully warmed expressions.
    #[allow(dead_code)]
    pub fn warm_up(&mut self, expressions: &[(String, CompilerContext)]) -> Result<usize> {
        use crate::parser::parse_expression;
        use tensorlogic_compiler::compile_to_einsum_with_context;

        let mut warmed = 0;

        for (expr_str, context) in expressions {
            // Parse and compile
            if let Ok(expr) = parse_expression(expr_str) {
                let mut ctx_clone = context.clone();
                if let Ok(graph) = compile_to_einsum_with_context(&expr, &mut ctx_clone) {
                    // Store in cache
                    if self.put(&expr, context, &graph).is_ok() {
                        warmed += 1;
                    }
                }
            }
        }

        Ok(warmed)
    }

    /// Warm up the cache from a file containing expressions (one per line)
    ///
    /// Lines starting with '#' are treated as comments and ignored.
    /// Format: `expression | strategy | domains`
    /// Example: `AND(a, b) | soft_differentiable | Person:100,Item:50`
    #[allow(dead_code)]
    pub fn warm_up_from_file(&mut self, file_path: &std::path::Path) -> Result<CacheWarmupResult> {
        use std::fs;
        use tensorlogic_compiler::CompilationConfig;

        let content = fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read warmup file: {}", file_path.display()))?;

        let mut expressions = Vec::new();
        let mut errors = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse the line format: expression | strategy | domains
            let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();

            if parts.is_empty() {
                continue;
            }

            let expr_str = parts[0].to_string();

            // Determine strategy (default to soft_differentiable)
            let config = if parts.len() > 1 {
                match parts[1] {
                    "hard_boolean" => CompilationConfig::hard_boolean(),
                    "fuzzy_godel" => CompilationConfig::fuzzy_godel(),
                    "fuzzy_product" => CompilationConfig::fuzzy_product(),
                    "fuzzy_lukasiewicz" => CompilationConfig::fuzzy_lukasiewicz(),
                    "probabilistic" => CompilationConfig::probabilistic(),
                    _ => CompilationConfig::soft_differentiable(),
                }
            } else {
                CompilationConfig::soft_differentiable()
            };

            let mut context = CompilerContext::with_config(config);

            // Parse domains if provided
            if parts.len() > 2 {
                for domain_spec in parts[2].split(',') {
                    let domain_parts: Vec<&str> = domain_spec.split(':').collect();
                    if domain_parts.len() == 2 {
                        if let Ok(size) = domain_parts[1].parse::<usize>() {
                            context.add_domain(domain_parts[0], size);
                        }
                    }
                }
            }

            expressions.push((expr_str, context));
        }

        // Warm up the cache
        match self.warm_up(&expressions) {
            Ok(warmed) => Ok(CacheWarmupResult {
                total: expressions.len(),
                warmed,
                errors,
            }),
            Err(e) => {
                errors.push(format!("Warmup error: {}", e));
                Ok(CacheWarmupResult {
                    total: expressions.len(),
                    warmed: 0,
                    errors,
                })
            }
        }
    }
}

/// Result of a cache warmup operation
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CacheWarmupResult {
    /// Total number of expressions in warmup file
    pub total: usize,
    /// Number of expressions successfully warmed
    pub warmed: usize,
    /// Errors encountered during warmup
    pub errors: Vec<String>,
}

impl CacheWarmupResult {
    /// Print warmup results
    #[allow(dead_code)]
    pub fn print(&self) {
        use crate::output::{print_error, print_header, print_info, print_success};

        print_header("Cache Warmup Results");
        print_info(&format!("  Total expressions: {}", self.total));
        print_success(&format!("  Successfully warmed: {}", self.warmed));

        if !self.errors.is_empty() {
            print_error(&format!("  Errors: {}", self.errors.len()));
            for error in &self.errors {
                print_info(&format!("    - {}", error));
            }
        }
    }
}

/// Cache statistics with LRU metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cache entries
    pub entries: usize,
    /// Current cache size in MB
    pub size_mb: usize,
    /// Maximum cache size in MB
    pub max_size_mb: usize,
    /// Whether caching is enabled
    pub enabled: bool,
    /// Cache directory path (serialized as string for JSON compatibility)
    #[serde(
        serialize_with = "serialize_path",
        deserialize_with = "deserialize_path"
    )]
    pub cache_dir: PathBuf,
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Hit rate percentage
    pub hit_rate: f64,
    /// Whether compression is enabled
    pub compression_enabled: bool,
}

// Helper functions for PathBuf serialization
fn serialize_path<S>(path: &Path, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&path.to_string_lossy())
}

fn deserialize_path<'de, D>(deserializer: D) -> Result<PathBuf, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(PathBuf::from(s))
}

impl CacheStats {
    /// Print cache statistics with LRU metrics
    pub fn print(&self) {
        use crate::output::{print_header, print_info, print_success};

        print_header("Cache Statistics");
        print_info(&format!("  Entries: {}", self.entries));
        print_info(&format!(
            "  Size: {} MB / {} MB ({:.1}% full)",
            self.size_mb,
            self.max_size_mb,
            (self.size_mb as f64 / self.max_size_mb as f64) * 100.0
        ));
        print_info(&format!(
            "  Enabled: {}",
            if self.enabled { "yes" } else { "no" }
        ));
        print_info(&format!(
            "  Compression: {}",
            if self.compression_enabled {
                "enabled"
            } else {
                "disabled"
            }
        ));

        // Performance metrics
        print_header("Performance Metrics");
        print_info(&format!("  Cache Hits: {}", self.hits));
        print_info(&format!("  Cache Misses: {}", self.misses));
        print_info(&format!("  Evictions: {}", self.evictions));

        if self.hits + self.misses > 0 {
            if self.hit_rate >= 80.0 {
                print_success(&format!("  Hit Rate: {:.2}% (excellent)", self.hit_rate));
            } else if self.hit_rate >= 50.0 {
                print_info(&format!("  Hit Rate: {:.2}% (good)", self.hit_rate));
            } else {
                print_info(&format!("  Hit Rate: {:.2}% (poor)", self.hit_rate));
            }
        } else {
            print_info("  Hit Rate: N/A (no requests yet)");
        }

        print_info(&format!("  Location: {}", self.cache_dir.display()));
    }

    /// Export cache statistics as JSON
    #[allow(dead_code)]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("Failed to serialize cache statistics to JSON")
    }

    /// Export cache statistics to a JSON file
    #[allow(dead_code)]
    pub fn export_to_file(&self, path: &Path) -> Result<()> {
        let json = self.to_json()?;
        fs::write(path, json).context("Failed to write cache statistics to file")?;
        Ok(())
    }

    /// Get analytics report with derived metrics
    #[allow(dead_code)]
    pub fn get_analytics(&self) -> CacheAnalytics {
        let total_requests = self.hits + self.misses;
        let utilization_pct = if self.max_size_mb > 0 {
            (self.size_mb as f64 / self.max_size_mb as f64) * 100.0
        } else {
            0.0
        };

        let avg_entry_size_kb = if self.entries > 0 {
            (self.size_mb as f64 * 1024.0) / self.entries as f64
        } else {
            0.0
        };

        let eviction_rate = if total_requests > 0 {
            (self.evictions as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        let efficiency_score =
            calculate_efficiency_score(self.hit_rate, utilization_pct, eviction_rate);

        CacheAnalytics {
            total_requests,
            utilization_pct,
            avg_entry_size_kb,
            eviction_rate,
            efficiency_score,
            recommendation: generate_recommendation(
                self.hit_rate,
                utilization_pct,
                eviction_rate,
                self.entries,
            ),
        }
    }
}

/// Cache analytics with derived metrics and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheAnalytics {
    /// Total cache requests (hits + misses)
    pub total_requests: u64,
    /// Cache utilization percentage
    pub utilization_pct: f64,
    /// Average entry size in KB
    pub avg_entry_size_kb: f64,
    /// Eviction rate percentage
    pub eviction_rate: f64,
    /// Overall efficiency score (0-100)
    pub efficiency_score: f64,
    /// Optimization recommendation
    pub recommendation: String,
}

impl CacheAnalytics {
    /// Print analytics report
    #[allow(dead_code)]
    pub fn print(&self) {
        use crate::output::{print_header, print_info, print_success, print_warning};

        print_header("Cache Analytics");
        print_info(&format!("  Total Requests: {}", self.total_requests));
        print_info(&format!("  Utilization: {:.1}%", self.utilization_pct));
        print_info(&format!(
            "  Avg Entry Size: {:.2} KB",
            self.avg_entry_size_kb
        ));
        print_info(&format!("  Eviction Rate: {:.2}%", self.eviction_rate));

        if self.efficiency_score >= 80.0 {
            print_success(&format!(
                "  Efficiency Score: {:.1}/100 (excellent)",
                self.efficiency_score
            ));
        } else if self.efficiency_score >= 60.0 {
            print_info(&format!(
                "  Efficiency Score: {:.1}/100 (good)",
                self.efficiency_score
            ));
        } else {
            print_warning(&format!(
                "  Efficiency Score: {:.1}/100 (needs improvement)",
                self.efficiency_score
            ));
        }

        if !self.recommendation.is_empty() {
            print_header("Recommendation");
            print_info(&format!("  {}", self.recommendation));
        }
    }

    /// Export analytics as JSON
    #[allow(dead_code)]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("Failed to serialize cache analytics to JSON")
    }
}

/// Calculate overall cache efficiency score (0-100)
fn calculate_efficiency_score(hit_rate: f64, utilization: f64, eviction_rate: f64) -> f64 {
    // Weighted scoring:
    // - 60% weight on hit rate
    // - 20% weight on optimal utilization (60-80% is ideal)
    // - 20% weight on low eviction rate

    let hit_score = hit_rate * 0.6;

    let utilization_score = if (60.0..=80.0).contains(&utilization) {
        100.0 * 0.2
    } else if utilization < 60.0 {
        (utilization / 60.0) * 100.0 * 0.2
    } else {
        ((100.0 - utilization) / 20.0) * 100.0 * 0.2
    };

    let eviction_score = if eviction_rate < 1.0 {
        100.0 * 0.2
    } else if eviction_rate < 5.0 {
        ((5.0 - eviction_rate) / 4.0) * 100.0 * 0.2
    } else {
        0.0
    };

    (hit_score + utilization_score + eviction_score).min(100.0)
}

/// Generate optimization recommendation based on metrics
fn generate_recommendation(
    hit_rate: f64,
    utilization: f64,
    eviction_rate: f64,
    entries: usize,
) -> String {
    if entries == 0 {
        return "Cache is empty. Start compiling expressions to populate the cache.".to_string();
    }

    if hit_rate < 50.0 {
        return "Low hit rate detected. Consider using cache warmup with frequently used expressions.".to_string();
    }

    if eviction_rate > 10.0 {
        return "High eviction rate detected. Consider increasing max cache size to reduce thrashing.".to_string();
    }

    if utilization > 90.0 {
        return "Cache is nearly full. Consider increasing max cache size or clearing old entries."
            .to_string();
    }

    if utilization < 30.0 && entries > 10 {
        return "Low cache utilization. Cache size may be larger than needed.".to_string();
    }

    "Cache is performing well. No immediate optimization needed.".to_string()
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

    #[test]
    fn test_cache_warmup() {
        let temp_dir = std::env::temp_dir().join("tensorlogic-test-cache-warmup");
        let mut cache = CompilationCache::new(Some(temp_dir.clone()), 100).unwrap();

        let ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());

        let expressions = vec![
            ("pred(x, y)".to_string(), ctx.clone()),
            ("AND(a, b)".to_string(), ctx.clone()),
        ];

        let warmed = cache.warm_up(&expressions).unwrap();

        assert_eq!(warmed, 2);
        assert_eq!(cache.stats().entries, 2);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cache_warmup_from_file() {
        use std::fs::File;
        use std::io::Write;

        let temp_dir = std::env::temp_dir().join("tensorlogic-test-cache-warmup-file");
        let mut cache = CompilationCache::new(Some(temp_dir.clone()), 100).unwrap();

        // Create a warmup file
        let warmup_file = temp_dir.join("warmup.txt");
        let mut file = File::create(&warmup_file).unwrap();
        writeln!(file, "# This is a comment").unwrap();
        writeln!(file, "pred(x, y) | soft_differentiable | Person:100").unwrap();
        writeln!(file, "AND(a, b)").unwrap();

        let result = cache.warm_up_from_file(&warmup_file).unwrap();

        assert_eq!(result.total, 2);
        assert_eq!(result.warmed, 2);

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cache_analytics() {
        let temp_dir = std::env::temp_dir().join("tensorlogic-test-cache-analytics");
        let stats = CacheStats {
            entries: 50,
            size_mb: 100,
            max_size_mb: 200,
            enabled: true,
            cache_dir: temp_dir.clone(),
            hits: 800,
            misses: 200,
            evictions: 10,
            hit_rate: 80.0,
            compression_enabled: true,
        };

        let analytics = stats.get_analytics();

        assert_eq!(analytics.total_requests, 1000);
        assert_eq!(analytics.utilization_pct, 50.0);
        assert!(analytics.efficiency_score >= 70.0); // Should be good with 80% hit rate
        assert!(!analytics.recommendation.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cache_stats_json_export() {
        let temp_dir = std::env::temp_dir().join("tensorlogic-test-cache-json");
        let stats = CacheStats {
            entries: 10,
            size_mb: 50,
            max_size_mb: 500,
            enabled: true,
            cache_dir: temp_dir.clone(),
            hits: 100,
            misses: 20,
            evictions: 2,
            hit_rate: 83.33,
            compression_enabled: true,
        };

        let json = stats.to_json();
        assert!(json.is_ok());

        let json_str = json.unwrap();
        assert!(json_str.contains("\"entries\""));
        assert!(json_str.contains("\"hits\""));
        assert!(json_str.contains("\"hit_rate\""));

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cache_analytics_json_export() {
        let analytics = CacheAnalytics {
            total_requests: 500,
            utilization_pct: 65.0,
            avg_entry_size_kb: 512.0,
            eviction_rate: 2.5,
            efficiency_score: 85.0,
            recommendation: "Cache is performing well.".to_string(),
        };

        let json = analytics.to_json();
        assert!(json.is_ok());

        let json_str = json.unwrap();
        assert!(json_str.contains("\"total_requests\""));
        assert!(json_str.contains("\"efficiency_score\""));
        assert!(json_str.contains("\"recommendation\""));
    }

    #[test]
    fn test_efficiency_score_calculation() {
        // Excellent cache: high hit rate, good utilization, low evictions
        let score1 = calculate_efficiency_score(90.0, 70.0, 0.5);
        assert!(score1 >= 80.0);

        // Poor cache: low hit rate
        let score2 = calculate_efficiency_score(30.0, 70.0, 0.5);
        assert!(score2 < 60.0);

        // High evictions
        let score3 = calculate_efficiency_score(80.0, 70.0, 15.0);
        assert!(score3 < 80.0);
    }

    #[test]
    fn test_recommendation_generation() {
        // Empty cache
        let rec1 = generate_recommendation(0.0, 0.0, 0.0, 0);
        assert!(rec1.contains("empty"));

        // Low hit rate
        let rec2 = generate_recommendation(30.0, 50.0, 1.0, 100);
        assert!(rec2.contains("hit rate"));

        // High eviction rate
        let rec3 = generate_recommendation(80.0, 70.0, 15.0, 100);
        assert!(rec3.contains("eviction"));

        // Cache nearly full
        let rec4 = generate_recommendation(80.0, 95.0, 1.0, 100);
        assert!(rec4.contains("nearly full") || rec4.contains("full"));

        // Good performance
        let rec5 = generate_recommendation(85.0, 65.0, 1.0, 100);
        assert!(rec5.contains("performing well"));
    }
}
