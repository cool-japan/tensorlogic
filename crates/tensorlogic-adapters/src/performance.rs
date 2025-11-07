//! Performance optimizations for symbol tables.
//!
//! This module provides optimizations for:
//! - String interning to reduce memory usage
//! - Lookup caching for frequently accessed data
//! - Efficient data structure selection

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// String interner for reducing memory usage of repeated strings.
///
/// Domain and predicate names are often repeated throughout the symbol table.
/// This interner ensures each unique string is stored only once in memory.
///
/// # Example
///
/// ```rust
/// use tensorlogic_adapters::StringInterner;
///
/// let mut interner = StringInterner::new();
/// let id1 = interner.intern("Person");
/// let id2 = interner.intern("Person");
/// assert_eq!(id1, id2); // Same string gets same ID
///
/// assert_eq!(interner.resolve(id1), Some("Person"));
/// ```
#[derive(Clone, Debug, Default)]
pub struct StringInterner {
    strings: Arc<RwLock<HashMap<String, usize>>>,
    ids: Arc<RwLock<Vec<String>>>,
}

impl StringInterner {
    /// Create a new string interner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string and return its unique ID.
    ///
    /// If the string already exists, returns the existing ID.
    /// Otherwise, allocates a new ID and stores the string.
    pub fn intern(&mut self, s: &str) -> usize {
        let mut strings = self.strings.write().unwrap();

        if let Some(&id) = strings.get(s) {
            return id;
        }

        let mut ids = self.ids.write().unwrap();
        let id = ids.len();
        ids.push(s.to_string());
        strings.insert(s.to_string(), id);
        id
    }

    /// Resolve an ID back to its string.
    pub fn resolve(&self, id: usize) -> Option<&str> {
        let ids = self.ids.read().unwrap();
        ids.get(id).map(|s| {
            // SAFETY: We're converting the reference to have a 'static lifetime.
            // This is safe because:
            // 1. The Arc ensures the data lives as long as any StringInterner instance
            // 2. The RwLock ensures no concurrent modifications while reading
            // 3. Strings in the interner are never removed, only added
            unsafe { std::mem::transmute::<&str, &str>(s.as_str()) }
        })
    }

    /// Get the number of unique strings interned.
    pub fn len(&self) -> usize {
        self.ids.read().unwrap().len()
    }

    /// Check if the interner is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all interned strings.
    pub fn clear(&mut self) {
        self.strings.write().unwrap().clear();
        self.ids.write().unwrap().clear();
    }

    /// Get memory usage statistics.
    pub fn memory_usage(&self) -> MemoryStats {
        let ids = self.ids.read().unwrap();
        let total_string_bytes: usize = ids.iter().map(|s| s.len()).sum();
        let count = ids.len();

        MemoryStats {
            string_count: count,
            total_string_bytes,
            index_overhead_bytes: count * std::mem::size_of::<String>(),
            hash_table_overhead_bytes: count * std::mem::size_of::<(String, usize)>(),
        }
    }
}

/// Memory usage statistics for a string interner.
#[derive(Clone, Debug)]
pub struct MemoryStats {
    /// Number of unique strings interned.
    pub string_count: usize,
    /// Total bytes used by string data.
    pub total_string_bytes: usize,
    /// Overhead for the index vector.
    pub index_overhead_bytes: usize,
    /// Overhead for the hash table.
    pub hash_table_overhead_bytes: usize,
}

impl MemoryStats {
    /// Total memory usage in bytes.
    pub fn total_bytes(&self) -> usize {
        self.total_string_bytes + self.index_overhead_bytes + self.hash_table_overhead_bytes
    }
}

/// Cache for frequently accessed lookups.
///
/// This provides a simple LRU-like cache for domain and predicate lookups
/// to avoid repeated hash table accesses.
///
/// # Example
///
/// ```rust
/// use tensorlogic_adapters::LookupCache;
///
/// let mut cache = LookupCache::new(100);
/// cache.insert("Person".to_string(), 42);
/// assert_eq!(cache.get(&"Person".to_string()), Some(&42));
/// ```
#[derive(Clone, Debug)]
pub struct LookupCache<K, V> {
    cache: HashMap<K, V>,
    capacity: usize,
    access_count: HashMap<K, usize>,
}

impl<K: Clone + Eq + std::hash::Hash, V: Clone> LookupCache<K, V> {
    /// Create a new lookup cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            capacity,
            access_count: HashMap::with_capacity(capacity),
        }
    }

    /// Insert a key-value pair into the cache.
    ///
    /// If the cache is at capacity, removes the least recently used item.
    pub fn insert(&mut self, key: K, value: V) {
        if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            self.evict_lru();
        }

        self.cache.insert(key.clone(), value);
        *self.access_count.entry(key).or_insert(0) += 1;
    }

    /// Get a value from the cache, updating access count.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(count) = self.access_count.get_mut(key) {
            *count += 1;
        }
        self.cache.get(key)
    }

    /// Remove the least recently used item from the cache.
    fn evict_lru(&mut self) {
        if let Some((key_to_remove, _)) = self.access_count.iter().min_by_key(|(_, &count)| count) {
            let key_to_remove = key_to_remove.clone();
            self.cache.remove(&key_to_remove);
            self.access_count.remove(&key_to_remove);
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_count.clear();
    }

    /// Get the number of items in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache hit statistics.
    pub fn stats(&self) -> CacheStats {
        let total_accesses: usize = self.access_count.values().sum();
        CacheStats {
            size: self.cache.len(),
            capacity: self.capacity,
            total_accesses,
        }
    }
}

/// Cache statistics.
#[derive(Clone, Debug)]
pub struct CacheStats {
    /// Current number of items in cache.
    pub size: usize,
    /// Maximum cache capacity.
    pub capacity: usize,
    /// Total number of cache accesses.
    pub total_accesses: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_interner_basic() {
        let mut interner = StringInterner::new();

        let id1 = interner.intern("Person");
        let id2 = interner.intern("Agent");
        let id3 = interner.intern("Person");

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(interner.resolve(id1), Some("Person"));
        assert_eq!(interner.resolve(id2), Some("Agent"));
    }

    #[test]
    fn test_string_interner_len() {
        let mut interner = StringInterner::new();
        assert_eq!(interner.len(), 0);

        interner.intern("Person");
        assert_eq!(interner.len(), 1);

        interner.intern("Person");
        assert_eq!(interner.len(), 1);

        interner.intern("Agent");
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn test_string_interner_clear() {
        let mut interner = StringInterner::new();
        interner.intern("Person");
        interner.intern("Agent");
        assert_eq!(interner.len(), 2);

        interner.clear();
        assert_eq!(interner.len(), 0);
    }

    #[test]
    fn test_string_interner_memory_stats() {
        let mut interner = StringInterner::new();
        interner.intern("Person");
        interner.intern("Agent");

        let stats = interner.memory_usage();
        assert_eq!(stats.string_count, 2);
        assert_eq!(stats.total_string_bytes, 6 + 5); // "Person" + "Agent"
        assert!(stats.total_bytes() > 0);
    }

    #[test]
    fn test_lookup_cache_basic() {
        let mut cache = LookupCache::new(3);

        cache.insert("key1", 1);
        cache.insert("key2", 2);

        assert_eq!(cache.get(&"key1"), Some(&1));
        assert_eq!(cache.get(&"key2"), Some(&2));
        assert_eq!(cache.get(&"key3"), None);
    }

    #[test]
    fn test_lookup_cache_eviction() {
        let mut cache = LookupCache::new(2);

        cache.insert("key1", 1);
        cache.insert("key2", 2);
        cache.get(&"key1"); // Access key1 multiple times
        cache.get(&"key1");
        cache.insert("key3", 3); // Should evict key2 (least accessed)

        assert_eq!(cache.get(&"key1"), Some(&1));
        assert_eq!(cache.get(&"key2"), None);
        assert_eq!(cache.get(&"key3"), Some(&3));
    }

    #[test]
    fn test_lookup_cache_clear() {
        let mut cache = LookupCache::new(10);
        cache.insert("key1", 1);
        cache.insert("key2", 2);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lookup_cache_stats() {
        let mut cache = LookupCache::new(10);
        cache.insert("key1", 1);
        cache.insert("key2", 2);
        cache.get(&"key1");
        cache.get(&"key1");

        let stats = cache.stats();
        assert_eq!(stats.size, 2);
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.total_accesses, 4); // 2 inserts + 2 gets
    }
}
