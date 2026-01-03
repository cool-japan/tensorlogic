//! Example 25: Query Result Caching
//!
//! This example demonstrates the query result caching system for performance optimization.
//! Features:
//! - TTL-based cache expiration
//! - LRU eviction when cache is full
//! - Cache statistics tracking (hits, misses, evictions)
//! - Multiple cache configurations (small, large, no-TTL)
//! - Symbol table-specific caching

use std::thread;
use std::time::Duration;
use tensorlogic_adapters::{
    CacheConfig, CacheKey, DomainInfo, PredicateInfo, QueryCache, SymbolTable, SymbolTableCache,
};

fn main() {
    println!("=== Example 25: Query Result Caching ===\n");

    // Scenario 1: Basic cache operations
    println!("📦 Scenario 1: Basic Cache Operations");
    println!("─────────────────────────────────────");
    scenario_basic_caching();
    println!();

    // Scenario 2: TTL-based expiration
    println!("⏰ Scenario 2: TTL-Based Expiration");
    println!("───────────────────────────────────");
    scenario_ttl_expiration();
    println!();

    // Scenario 3: LRU eviction
    println!("🔄 Scenario 3: LRU Eviction");
    println!("───────────────────────────");
    scenario_lru_eviction();
    println!();

    // Scenario 4: Cache configurations
    println!("⚙️ Scenario 4: Cache Configurations");
    println!("────────────────────────────────────");
    scenario_cache_configs();
    println!();

    // Scenario 5: Symbol table caching
    println!("🔍 Scenario 5: Symbol Table Caching");
    println!("────────────────────────────────────");
    scenario_symbol_table_cache();
    println!();

    // Scenario 6: Performance comparison
    println!("⚡ Scenario 6: Performance Comparison");
    println!("─────────────────────────────────────");
    scenario_performance();
    println!();

    println!("✅ All query caching scenarios completed!");
}

fn scenario_basic_caching() {
    // Create a cache with default configuration
    let mut cache: QueryCache<Vec<String>> = QueryCache::with_config(CacheConfig::default());

    println!("Creating cache with default config:");
    println!("  - Max entries: 1000");
    println!("  - Default TTL: 5 minutes");
    println!("  - LRU eviction: enabled");
    println!();

    // Add some entries
    let key1 = CacheKey::PredicateByName("knows".to_string());
    let key2 = CacheKey::PredicatesByArity(2);
    let key3 = CacheKey::AllDomainNames;

    cache.insert(
        key1.clone(),
        vec!["Person".to_string(), "Person".to_string()],
    );
    cache.insert(key2.clone(), vec!["knows".to_string(), "likes".to_string()]);
    cache.insert(
        key3.clone(),
        vec!["Person".to_string(), "Organization".to_string()],
    );

    println!("Added 3 entries to cache");

    // Test cache hits
    assert!(cache.get(&key1).is_some());
    assert!(cache.get(&key2).is_some());
    assert!(cache.get(&key3).is_some());

    println!("✓ All 3 entries retrieved successfully (cache hits)");

    // Test cache miss
    let key4 = CacheKey::PredicateByName("unknown".to_string());
    assert!(cache.get(&key4).is_none());

    println!("✓ Cache miss for non-existent key");

    // Check statistics
    let stats = cache.stats();
    println!("\nCache statistics:");
    println!("  - Total entries: {}", cache.len());
    println!("  - Hits: {}", stats.hits);
    println!("  - Misses: {}", stats.misses);
    println!("  - Hit rate: {:.2}%", stats.hit_rate() * 100.0);
}

fn scenario_ttl_expiration() {
    // Create a cache with short TTL for testing
    let config = CacheConfig {
        max_entries: 100,
        default_ttl: Some(Duration::from_millis(100)),
        enable_lru: true,
        enable_stats: true,
    };

    let mut cache: QueryCache<String> = QueryCache::with_config(config);

    println!("Creating cache with 100ms TTL");
    println!();

    // Add an entry
    let key = CacheKey::PredicateByName("temporary".to_string());
    cache.insert(key.clone(), "This will expire soon".to_string());

    println!("Added entry to cache");

    // Immediately retrieve (should succeed)
    assert!(cache.get(&key).is_some());
    println!("✓ Entry retrieved immediately (before expiration)");

    // Wait for expiration
    println!("⏳ Waiting 150ms for expiration...");
    thread::sleep(Duration::from_millis(150));

    // Try to retrieve again (should fail due to expiration)
    assert!(cache.get(&key).is_none());
    println!("✓ Entry expired and cannot be retrieved");

    // Check expiration count
    let stats = cache.stats();
    println!("\nCache statistics after expiration:");
    println!("  - Total entries: {}", cache.len());
    println!("  - Expirations: {}", stats.expirations);
}

fn scenario_lru_eviction() {
    // Create a small cache to trigger eviction
    let config = CacheConfig {
        max_entries: 3, // Very small to trigger eviction quickly
        default_ttl: None,
        enable_lru: true,
        enable_stats: true,
    };

    let mut cache: QueryCache<usize> = QueryCache::with_config(config);

    println!("Creating cache with max 3 entries (no TTL)");
    println!();

    // Add entries up to capacity
    for i in 0..3 {
        let key = CacheKey::Custom(format!("key{}", i));
        cache.insert(key, i);
        println!("Added entry {} to cache", i);
    }

    // All entries should be present
    for i in 0..3 {
        let key = CacheKey::Custom(format!("key{}", i));
        assert!(cache.get(&key).is_some());
    }
    println!("✓ All 3 entries present in cache");

    // Add one more entry (should trigger LRU eviction)
    println!("\nAdding 4th entry (should evict least recently used)...");
    let key4 = CacheKey::Custom("key3".to_string());
    cache.insert(key4, 3);

    // First entry should be evicted
    let key0 = CacheKey::Custom("key0".to_string());
    assert!(cache.get(&key0).is_none());
    println!("✓ Entry 0 evicted (was least recently used)");

    // Other entries should still be present
    for i in 1..=3 {
        let key = CacheKey::Custom(format!("key{}", i));
        assert!(cache.get(&key).is_some());
    }
    println!("✓ Entries 1-3 still present");

    let stats = cache.stats();
    println!("\nCache statistics after eviction:");
    println!("  - Total entries: {}", cache.len());
    println!("  - Evictions: {}", stats.evictions);
}

fn scenario_cache_configs() {
    println!("Testing different cache configurations:");
    println!();

    // Small cache
    let small_config = CacheConfig::small();
    println!("Small cache:");
    println!("  - Max entries: {}", small_config.max_entries);
    println!(
        "  - TTL: {:?}",
        small_config.default_ttl.map(|d| format!("{:?}", d))
    );

    // Large cache
    let large_config = CacheConfig::large();
    println!("\nLarge cache:");
    println!("  - Max entries: {}", large_config.max_entries);
    println!(
        "  - TTL: {:?}",
        large_config.default_ttl.map(|d| format!("{:?}", d))
    );

    // No TTL cache
    let no_ttl_config = CacheConfig::no_ttl();
    println!("\nNo-TTL cache:");
    println!("  - Max entries: {}", no_ttl_config.max_entries);
    println!("  - TTL: None (entries never expire by time)");

    println!("\n✓ All configurations created successfully");
}

fn scenario_symbol_table_cache() {
    // Create a symbol table with some data
    let mut table = SymbolTable::new();

    // Add domains
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table
        .add_domain(DomainInfo::new("Organization", 50))
        .unwrap();
    table.add_domain(DomainInfo::new("Event", 200)).unwrap();

    // Add predicates
    table
        .add_predicate(PredicateInfo::new(
            "knows",
            vec!["Person".to_string(), "Person".to_string()],
        ))
        .unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "works_at",
            vec!["Person".to_string(), "Organization".to_string()],
        ))
        .unwrap();
    table
        .add_predicate(PredicateInfo::new(
            "attends",
            vec!["Person".to_string(), "Event".to_string()],
        ))
        .unwrap();
    table
        .add_predicate(PredicateInfo::new("age", vec!["Person".to_string()]))
        .unwrap();

    println!("Created symbol table:");
    println!("  - 3 domains: Person, Organization, Event");
    println!("  - 4 predicates: knows, works_at, attends, age");
    println!();

    // Create a cached symbol table
    let mut cache = SymbolTableCache::new();

    println!("Performing cached queries:");
    println!();

    // Query 1: Get predicates by arity (first time - cache miss)
    let binary_predicates = cache.get_predicates_by_arity(&table, 2);
    println!("Query 1: Binary predicates (arity=2)");
    println!("  Found {} predicates", binary_predicates.len());
    for pred in &binary_predicates {
        println!("    - {}", pred.name);
    }

    // Query 2: Get predicates using domain (first time - cache miss)
    let person_predicates = cache.get_predicates_by_domain(&table, "Person");
    println!("\nQuery 2: Predicates using 'Person' domain");
    println!("  Found {} predicates", person_predicates.len());
    for pred in &person_predicates {
        println!("    - {}", pred.name);
    }

    // Query 3: Get domain names (first time - cache miss)
    let domain_names = cache.get_domain_names(&table);
    println!("\nQuery 3: All domain names");
    println!("  Found {} domains: {:?}", domain_names.len(), domain_names);

    // Query 4: Get domain usage count (first time - cache miss)
    let person_usage = cache.get_domain_usage_count(&table, "Person");
    println!("\nQuery 4: Usage count for 'Person' domain");
    println!("  Used in {} predicates", person_usage);

    println!();

    // Repeat queries (should hit cache)
    println!("Repeating queries (should hit cache):");
    let _ = cache.get_predicates_by_arity(&table, 2);
    let _ = cache.get_predicates_by_domain(&table, "Person");
    let _ = cache.get_domain_names(&table);
    let _ = cache.get_domain_usage_count(&table, "Person");
    println!("✓ All queries completed");

    // Check cache statistics
    let stats = cache.combined_stats();
    println!("\nCache statistics:");
    println!("  - Hits: {}", stats.hits);
    println!("  - Misses: {}", stats.misses);
    println!("  - Hit rate: {:.2}%", stats.hit_rate() * 100.0);
    println!("  - Miss rate: {:.2}%", stats.miss_rate() * 100.0);
}

fn scenario_performance() {
    use std::time::Instant;

    println!("Measuring cache performance benefits:");
    println!();

    // Create a large symbol table
    let mut table = SymbolTable::new();

    // Add many domains
    for i in 0..100 {
        table
            .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
            .unwrap();
    }

    // Add many predicates
    for i in 0..500 {
        let arity = (i % 4) + 1; // Arity from 1 to 4
        let domains: Vec<String> = (0..arity)
            .map(|j| format!("Domain{}", (i + j) % 100))
            .collect();
        table
            .add_predicate(PredicateInfo::new(format!("pred{}", i), domains))
            .unwrap();
    }

    println!("Created large symbol table:");
    println!("  - 100 domains");
    println!("  - 500 predicates");
    println!();

    // Test without cache
    let start = Instant::now();
    for _ in 0..100 {
        let _ = table
            .predicates
            .values()
            .filter(|p| p.arg_domains.len() == 2)
            .map(|p| p.name.clone())
            .collect::<Vec<_>>();
    }
    let no_cache_time = start.elapsed();
    println!("Without cache (100 queries): {:?}", no_cache_time);

    // Test with cache
    let mut cache = SymbolTableCache::with_config(CacheConfig::large());
    let start = Instant::now();
    for _ in 0..100 {
        let _ = cache.get_predicates_by_arity(&table, 2);
    }
    let with_cache_time = start.elapsed();
    println!("With cache (100 queries): {:?}", with_cache_time);

    // Calculate speedup
    let speedup = no_cache_time.as_micros() as f64 / with_cache_time.as_micros() as f64;
    println!("\n⚡ Speedup: {:.2}x faster with caching", speedup);

    let stats = cache.combined_stats();
    println!("\nFinal cache statistics:");
    println!("  - Hit rate: {:.2}%", stats.hit_rate() * 100.0);
    println!("  - Total hits: {}", stats.hits);
    println!("  - Total misses: {}", stats.misses);
}
