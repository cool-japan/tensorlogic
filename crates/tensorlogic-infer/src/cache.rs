//! Tensor caching and memory pooling for efficient execution.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

/// Cache key for tensor identification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub graph_id: Option<String>,
    pub node_id: usize,
    pub input_hash: u64,
}

impl CacheKey {
    pub fn new(node_id: usize) -> Self {
        CacheKey {
            graph_id: None,
            node_id,
            input_hash: 0,
        }
    }

    pub fn with_graph(mut self, graph_id: impl Into<String>) -> Self {
        self.graph_id = Some(graph_id.into());
        self
    }

    pub fn with_inputs<T: Hash>(mut self, inputs: &[T]) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for input in inputs {
            input.hash(&mut hasher);
        }
        self.input_hash = hasher.finish();
        self
    }
}

/// Cache eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// First In First Out
    FIFO,
    /// Least Frequently Used
    LFU,
    /// No eviction (cache grows unbounded)
    None,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub current_size: usize,
    pub peak_size: usize,
    pub total_bytes: usize,
}

impl CacheStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Cache Stats:\n\
             - Hits: {} ({:.1}%)\n\
             - Misses: {}\n\
             - Evictions: {}\n\
             - Current size: {} entries\n\
             - Peak size: {} entries\n\
             - Total bytes: {} ({:.2} MB)",
            self.hits,
            self.hit_rate() * 100.0,
            self.misses,
            self.evictions,
            self.current_size,
            self.peak_size,
            self.total_bytes,
            self.total_bytes as f64 / (1024.0 * 1024.0)
        )
    }
}

/// Cached entry metadata
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    size_bytes: usize,
    access_count: usize,
    last_access: usize, // Timestamp
}

/// Tensor cache with configurable eviction policy
pub struct TensorCache<T> {
    cache: HashMap<CacheKey, CacheEntry<T>>,
    eviction_policy: EvictionPolicy,
    max_size: Option<usize>,
    max_bytes: Option<usize>,
    stats: CacheStats,
    access_counter: usize,
    access_order: VecDeque<CacheKey>,
}

impl<T: Clone> TensorCache<T> {
    pub fn new(eviction_policy: EvictionPolicy) -> Self {
        TensorCache {
            cache: HashMap::new(),
            eviction_policy,
            max_size: None,
            max_bytes: None,
            stats: CacheStats::new(),
            access_counter: 0,
            access_order: VecDeque::new(),
        }
    }

    pub fn with_max_size(mut self, max_entries: usize) -> Self {
        self.max_size = Some(max_entries);
        self
    }

    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = Some(max_bytes);
        self
    }

    /// Insert a value into the cache
    pub fn insert(&mut self, key: CacheKey, value: T, size_bytes: usize) {
        // Check if eviction is needed
        while self.should_evict(size_bytes) {
            self.evict_one();
        }

        // Insert or update entry
        if self.cache.contains_key(&key) {
            // Update existing entry
            if let Some(entry) = self.cache.get_mut(&key) {
                self.stats.total_bytes -= entry.size_bytes;
                entry.value = value;
                entry.size_bytes = size_bytes;
                entry.access_count += 1;
                entry.last_access = self.access_counter;
                self.stats.total_bytes += size_bytes;
            }
        } else {
            // Insert new entry
            let entry = CacheEntry {
                value,
                size_bytes,
                access_count: 1,
                last_access: self.access_counter,
            };

            self.cache.insert(key.clone(), entry);
            self.stats.current_size += 1;
            self.stats.peak_size = self.stats.peak_size.max(self.stats.current_size);
            self.stats.total_bytes += size_bytes;

            // Track access order for FIFO/LRU
            self.access_order.push_back(key);
        }

        self.access_counter += 1;
    }

    /// Get a value from the cache
    pub fn get(&mut self, key: &CacheKey) -> Option<T> {
        if let Some(entry) = self.cache.get_mut(key) {
            self.stats.hits += 1;
            entry.access_count += 1;
            entry.last_access = self.access_counter;
            self.access_counter += 1;

            // Update access order for LRU
            if self.eviction_policy == EvictionPolicy::LRU {
                self.access_order.retain(|k| k != key);
                self.access_order.push_back(key.clone());
            }

            Some(entry.value.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Check if a key exists in the cache without updating access stats
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.cache.contains_key(key)
    }

    /// Remove a specific entry
    pub fn remove(&mut self, key: &CacheKey) -> Option<T> {
        if let Some(entry) = self.cache.remove(key) {
            self.stats.current_size -= 1;
            self.stats.total_bytes -= entry.size_bytes;
            self.access_order.retain(|k| k != key);
            Some(entry.value)
        } else {
            None
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.stats.current_size = 0;
        self.stats.total_bytes = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset statistics (keep cached entries)
    pub fn reset_stats(&mut self) {
        self.stats.hits = 0;
        self.stats.misses = 0;
        self.stats.evictions = 0;
    }

    fn should_evict(&self, new_size_bytes: usize) -> bool {
        if self.eviction_policy == EvictionPolicy::None {
            return false;
        }

        let size_exceeded = self
            .max_size
            .map(|max| self.stats.current_size >= max)
            .unwrap_or(false);

        let bytes_exceeded = self
            .max_bytes
            .map(|max| self.stats.total_bytes + new_size_bytes > max)
            .unwrap_or(false);

        size_exceeded || bytes_exceeded
    }

    fn evict_one(&mut self) {
        let key_to_evict = match self.eviction_policy {
            EvictionPolicy::LRU => self.find_lru_key(),
            EvictionPolicy::FIFO => self.find_fifo_key(),
            EvictionPolicy::LFU => self.find_lfu_key(),
            EvictionPolicy::None => return,
        };

        if let Some(key) = key_to_evict {
            self.remove(&key);
            self.stats.evictions += 1;
        }
    }

    fn find_lru_key(&self) -> Option<CacheKey> {
        self.cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, _)| key.clone())
    }

    fn find_fifo_key(&self) -> Option<CacheKey> {
        self.access_order.front().cloned()
    }

    fn find_lfu_key(&self) -> Option<CacheKey> {
        self.cache
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(key, _)| key.clone())
    }

    pub fn len(&self) -> usize {
        self.stats.current_size
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl<T: Clone> Default for TensorCache<T> {
    fn default() -> Self {
        Self::new(EvictionPolicy::LRU)
    }
}

/// Memory pool for tensor allocation reuse
pub struct MemoryPool<T> {
    pools: HashMap<usize, Vec<T>>,
    stats: PoolStats,
    max_pool_size: Option<usize>,
}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub allocations: usize,
    pub reuses: usize,
    pub releases: usize,
    pub peak_allocations: usize,
}

impl PoolStats {
    pub fn reuse_rate(&self) -> f64 {
        let total = self.allocations + self.reuses;
        if total == 0 {
            0.0
        } else {
            self.reuses as f64 / total as f64
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Memory Pool Stats:\n\
             - Allocations: {}\n\
             - Reuses: {} ({:.1}%)\n\
             - Releases: {}\n\
             - Peak allocations: {}",
            self.allocations,
            self.reuses,
            self.reuse_rate() * 100.0,
            self.releases,
            self.peak_allocations
        )
    }
}

impl<T> MemoryPool<T> {
    pub fn new() -> Self {
        MemoryPool {
            pools: HashMap::new(),
            stats: PoolStats::default(),
            max_pool_size: Some(100), // Default max 100 per size class
        }
    }

    pub fn with_max_pool_size(mut self, max_size: usize) -> Self {
        self.max_pool_size = Some(max_size);
        self
    }

    /// Acquire a tensor from the pool or allocate new
    pub fn acquire<F>(&mut self, size_class: usize, allocator: F) -> T
    where
        F: FnOnce() -> T,
    {
        if let Some(pool) = self.pools.get_mut(&size_class) {
            if let Some(tensor) = pool.pop() {
                self.stats.reuses += 1;
                return tensor;
            }
        }

        self.stats.allocations += 1;
        self.stats.peak_allocations = self
            .stats
            .peak_allocations
            .max(self.stats.allocations - self.stats.releases);

        allocator()
    }

    /// Release a tensor back to the pool
    pub fn release(&mut self, size_class: usize, tensor: T) {
        let pool = self.pools.entry(size_class).or_default();

        // Check pool size limit
        if let Some(max_size) = self.max_pool_size {
            if pool.len() >= max_size {
                // Pool is full, drop the tensor
                self.stats.releases += 1;
                return;
            }
        }

        pool.push(tensor);
        self.stats.releases += 1;
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.pools.clear();
        self.stats = PoolStats::default();
    }

    /// Get pool statistics
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Get total number of pooled tensors
    pub fn total_pooled(&self) -> usize {
        self.pools.values().map(|v| v.len()).sum()
    }
}

impl<T> Default for MemoryPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_creation() {
        let key1 = CacheKey::new(0);
        assert_eq!(key1.node_id, 0);
        assert_eq!(key1.input_hash, 0);

        let key2 = CacheKey::new(1).with_graph("graph1");
        assert_eq!(key2.graph_id, Some("graph1".to_string()));

        let inputs = vec![1, 2, 3];
        let key3 = CacheKey::new(2).with_inputs(&inputs);
        assert!(key3.input_hash != 0);
    }

    #[test]
    fn test_cache_basic_operations() {
        let mut cache: TensorCache<i32> = TensorCache::new(EvictionPolicy::LRU);

        let key = CacheKey::new(0);
        cache.insert(key.clone(), 42, 4);

        assert_eq!(cache.get(&key), Some(42));
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);

        let missing_key = CacheKey::new(1);
        assert_eq!(cache.get(&missing_key), None);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache: TensorCache<i32> = TensorCache::new(EvictionPolicy::LRU).with_max_size(2);

        cache.insert(CacheKey::new(0), 1, 4);
        cache.insert(CacheKey::new(1), 2, 4);
        cache.insert(CacheKey::new(2), 3, 4); // Should evict key 0

        assert!(!cache.contains(&CacheKey::new(0)));
        assert!(cache.contains(&CacheKey::new(1)));
        assert!(cache.contains(&CacheKey::new(2)));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_cache_fifo_eviction() {
        let mut cache: TensorCache<i32> = TensorCache::new(EvictionPolicy::FIFO).with_max_size(2);

        cache.insert(CacheKey::new(0), 1, 4);
        cache.insert(CacheKey::new(1), 2, 4);
        cache.insert(CacheKey::new(2), 3, 4); // Should evict key 0 (first in)

        assert!(!cache.contains(&CacheKey::new(0)));
        assert!(cache.contains(&CacheKey::new(1)));
        assert!(cache.contains(&CacheKey::new(2)));
    }

    #[test]
    fn test_cache_lfu_eviction() {
        let mut cache: TensorCache<i32> = TensorCache::new(EvictionPolicy::LFU).with_max_size(2);

        cache.insert(CacheKey::new(0), 1, 4);
        cache.insert(CacheKey::new(1), 2, 4);

        // Access key 0 multiple times
        cache.get(&CacheKey::new(0));
        cache.get(&CacheKey::new(0));

        cache.insert(CacheKey::new(2), 3, 4); // Should evict key 1 (least frequently used)

        assert!(cache.contains(&CacheKey::new(0)));
        assert!(!cache.contains(&CacheKey::new(1)));
        assert!(cache.contains(&CacheKey::new(2)));
    }

    #[test]
    fn test_cache_byte_limit() {
        let mut cache: TensorCache<Vec<u8>> =
            TensorCache::new(EvictionPolicy::LRU).with_max_bytes(20);

        cache.insert(CacheKey::new(0), vec![0; 8], 8);
        cache.insert(CacheKey::new(1), vec![0; 8], 8);
        cache.insert(CacheKey::new(2), vec![0; 8], 8); // Should evict to stay under 20 bytes

        // Should have at most 2 entries to stay under byte limit
        assert!(cache.len() <= 2);
        assert!(cache.stats().total_bytes <= 20);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: TensorCache<i32> = TensorCache::new(EvictionPolicy::LRU);

        cache.insert(CacheKey::new(0), 42, 4);
        cache.get(&CacheKey::new(0));
        cache.get(&CacheKey::new(1));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_cache_remove() {
        let mut cache: TensorCache<i32> = TensorCache::new(EvictionPolicy::LRU);

        cache.insert(CacheKey::new(0), 42, 4);
        assert_eq!(cache.len(), 1);

        let removed = cache.remove(&CacheKey::new(0));
        assert_eq!(removed, Some(42));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache: TensorCache<i32> = TensorCache::new(EvictionPolicy::LRU);

        cache.insert(CacheKey::new(0), 1, 4);
        cache.insert(CacheKey::new(1), 2, 4);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().total_bytes, 0);
    }

    #[test]
    fn test_memory_pool_basic() {
        let mut pool: MemoryPool<Vec<u8>> = MemoryPool::new();

        // Acquire new allocation
        let vec1 = pool.acquire(100, || vec![0u8; 100]);
        assert_eq!(vec1.len(), 100);
        assert_eq!(pool.stats().allocations, 1);

        // Release back to pool
        pool.release(100, vec1);
        assert_eq!(pool.stats().releases, 1);

        // Reuse from pool
        let vec2 = pool.acquire(100, || vec![0u8; 100]);
        assert_eq!(vec2.len(), 100);
        assert_eq!(pool.stats().reuses, 1);
    }

    #[test]
    fn test_memory_pool_size_classes() {
        let mut pool: MemoryPool<Vec<u8>> = MemoryPool::new();

        // Different size classes
        let vec1 = pool.acquire(100, || vec![0u8; 100]);
        let vec2 = pool.acquire(200, || vec![0u8; 200]);

        pool.release(100, vec1);
        pool.release(200, vec2);

        assert_eq!(pool.total_pooled(), 2);
    }

    #[test]
    fn test_memory_pool_max_size() {
        let mut pool: MemoryPool<Vec<u8>> = MemoryPool::new().with_max_pool_size(2);

        // Fill pool
        pool.release(100, vec![0u8; 100]);
        pool.release(100, vec![0u8; 100]);
        pool.release(100, vec![0u8; 100]); // Should be dropped

        assert_eq!(pool.total_pooled(), 2);
    }

    #[test]
    fn test_pool_stats() {
        let mut pool: MemoryPool<Vec<u8>> = MemoryPool::new();

        pool.acquire(100, || vec![0u8; 100]);
        pool.acquire(100, || vec![0u8; 100]);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 2);
        assert!(stats.reuse_rate() == 0.0);
    }
}
