//! Example 11: Lazy Loading for Huge Schemas
//!
//! This example demonstrates how to use lazy loading to efficiently
//! handle very large schemas that don't fit comfortably in memory.
//!
//! Run with: cargo run --example 11_lazy_loading

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tensorlogic_adapters::{
    DomainInfo, LazyLoadStats, LazySymbolTable, LoadStrategy, PredicateInfo, SchemaLoader,
};

/// Mock loader for demonstration
struct MockSchemaLoader {
    domains: HashMap<String, DomainInfo>,
    predicates: HashMap<String, PredicateInfo>,
}

impl MockSchemaLoader {
    fn new_large_schema() -> Self {
        let mut domains = HashMap::new();
        let mut predicates = HashMap::new();

        // Simulate a large schema with 1000 domains
        for i in 0..1000 {
            let name = format!("Domain{}", i);
            domains.insert(name.clone(), DomainInfo::new(&name, 100 + i));
        }

        // Simulate predicates
        for i in 0..500 {
            let name = format!("predicate{}", i);
            let arg_domains = vec![
                format!("Domain{}", i * 2 % 1000),
                format!("Domain{}", (i * 2 + 1) % 1000),
            ];
            predicates.insert(name.clone(), PredicateInfo::new(&name, arg_domains));
        }

        Self {
            domains,
            predicates,
        }
    }
}

impl SchemaLoader for MockSchemaLoader {
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

fn main() -> Result<()> {
    println!("=== Lazy Loading Example ===\n");

    // Example 1: Basic Lazy Loading
    println!("1. Basic Lazy Loading with On-Demand Strategy");
    println!("{}", "=".repeat(50));

    let loader: Arc<dyn SchemaLoader> = Arc::new(MockSchemaLoader::new_large_schema());
    let lazy_table = LazySymbolTable::new(Arc::clone(&loader));

    println!("Schema size: 1000 domains, 500 predicates");
    println!(
        "Initially loaded: {} domains",
        lazy_table.loaded_domain_count()
    );
    println!();

    // Access a domain (triggers lazy load)
    println!("Accessing Domain0...");
    let domain = lazy_table.get_domain("Domain0")?;
    println!("✓ Loaded: {:?}", domain.map(|d| d.name));
    println!("Loaded domains: {}", lazy_table.loaded_domain_count());

    // Access same domain again (cache hit)
    println!("\nAccessing Domain0 again...");
    let _domain = lazy_table.get_domain("Domain0")?;
    let stats = lazy_table.stats();
    println!("Cache hits: {}", stats.cache_hits);
    println!("Cache misses: {}", stats.cache_misses);
    println!("Hit rate: {:.2}%\n", stats.hit_rate() * 100.0);

    // Example 2: Batch Preloading
    println!("2. Batch Preloading for Frequently Used Domains");
    println!("{}", "=".repeat(50));

    let loader2: Arc<dyn SchemaLoader> = Arc::new(MockSchemaLoader::new_large_schema());
    let lazy_table2 = LazySymbolTable::new(loader2);

    println!(
        "Before preload: {} domains loaded",
        lazy_table2.loaded_domain_count()
    );

    // Preload frequently used domains
    let frequently_used = vec![
        "Domain0".to_string(),
        "Domain1".to_string(),
        "Domain2".to_string(),
        "Domain3".to_string(),
        "Domain4".to_string(),
    ];

    lazy_table2.preload_domains(&frequently_used)?;

    println!(
        "After preload: {} domains loaded",
        lazy_table2.loaded_domain_count()
    );

    let stats2 = lazy_table2.stats();
    println!("Batch loads: {}", stats2.batch_loads);
    println!("Total domain loads: {}\n", stats2.domain_loads);

    // Example 3: Load Strategies
    println!("3. Different Loading Strategies");
    println!("{}", "=".repeat(50));

    // On-Demand Strategy
    let loader_ondemand: Arc<dyn SchemaLoader> = Arc::new(MockSchemaLoader::new_large_schema());
    let _lazy_ondemand = LazySymbolTable::with_strategy(loader_ondemand, LoadStrategy::OnDemand);
    println!("✓ Created table with OnDemand strategy");

    // Predictive Strategy
    let loader_predictive: Arc<dyn SchemaLoader> = Arc::new(MockSchemaLoader::new_large_schema());
    let _lazy_predictive = LazySymbolTable::with_strategy(
        loader_predictive,
        LoadStrategy::Predictive {
            access_threshold: 5,
        },
    );
    println!("✓ Created table with Predictive strategy (threshold=5)");

    // Batched Strategy
    let loader_batched: Arc<dyn SchemaLoader> = Arc::new(MockSchemaLoader::new_large_schema());
    let _lazy_batched =
        LazySymbolTable::with_strategy(loader_batched, LoadStrategy::Batched { batch_size: 10 });
    println!("✓ Created table with Batched strategy (size=10)");
    println!();

    // Example 4: Listing Available Elements
    println!("4. Listing Available Elements (Without Loading)");
    println!("{}", "=".repeat(50));

    let domains = lazy_table.list_domains()?;
    println!("Total available domains: {}", domains.len());
    println!("First 10 domains: {:?}", &domains[0..10]);

    let predicates = lazy_table.list_predicates()?;
    println!("\nTotal available predicates: {}", predicates.len());
    println!("First 10 predicates: {:?}", &predicates[0..10]);
    println!();

    // Example 5: Memory Management
    println!("5. Memory Management and Cache Control");
    println!("{}", "=".repeat(50));

    println!("Loading multiple domains...");
    for i in 0..20 {
        lazy_table.get_domain(&format!("Domain{}", i))?;
    }

    println!("Loaded domains: {}", lazy_table.loaded_domain_count());

    // Clear cache to free memory
    println!("\nClearing cache...");
    lazy_table.clear_cache();
    println!(
        "Loaded domains after clear: {}",
        lazy_table.loaded_domain_count()
    );
    println!();

    // Example 6: Performance Monitoring
    println!("6. Performance Monitoring with Statistics");
    println!("{}", "=".repeat(50));

    let loader_perf: Arc<dyn SchemaLoader> = Arc::new(MockSchemaLoader::new_large_schema());
    let lazy_perf = LazySymbolTable::new(loader_perf);

    // Simulate workload
    println!("Simulating workload...");
    for i in 0..10 {
        // Access some domains multiple times
        lazy_perf.get_domain(&format!("Domain{}", i % 5))?;
    }

    let perf_stats = lazy_perf.stats();
    print_stats(&perf_stats);
    println!();

    // Example 7: Predicate Loading
    println!("7. Lazy Loading of Predicates");
    println!("{}", "=".repeat(50));

    // First load required domains
    lazy_table.get_domain("Domain0")?;
    lazy_table.get_domain("Domain1")?;

    println!("Loading predicate...");
    let predicate = lazy_table.get_predicate("predicate0")?;
    println!("✓ Loaded: {:?}", predicate.map(|p| p.name));

    let stats = lazy_table.stats();
    println!("Predicate loads: {}\n", stats.predicate_loads);

    // Example 8: Batch Predicate Loading
    println!("8. Batch Predicate Loading");
    println!("{}", "=".repeat(50));

    let predicates_to_load = vec![
        "predicate1".to_string(),
        "predicate2".to_string(),
        "predicate3".to_string(),
    ];

    lazy_table.preload_predicates(&predicates_to_load)?;
    println!(
        "✓ Preloaded {} predicates",
        lazy_table.loaded_predicate_count()
    );
    println!();

    // Example 9: Real-World Use Case - Progressive Schema Loading
    println!("9. Real-World: Progressive Schema Loading");
    println!("{}", "=".repeat(50));

    let loader_progressive: Arc<dyn SchemaLoader> = Arc::new(MockSchemaLoader::new_large_schema());
    let lazy_progressive = LazySymbolTable::new(loader_progressive);

    println!("Stage 1: Load core domains");
    let core_domains = vec!["Domain0".to_string(), "Domain1".to_string()];
    lazy_progressive.preload_domains(&core_domains)?;
    println!(
        "  Loaded: {} domains",
        lazy_progressive.loaded_domain_count()
    );

    println!("\nStage 2: Load related predicates");
    let core_predicates = vec!["predicate0".to_string(), "predicate1".to_string()];
    lazy_progressive.preload_predicates(&core_predicates)?;
    println!(
        "  Loaded: {} predicates",
        lazy_progressive.loaded_predicate_count()
    );

    println!("\nStage 3: Load additional domains as needed");
    for i in 2..5 {
        lazy_progressive.get_domain(&format!("Domain{}", i))?;
    }
    println!(
        "  Total loaded: {} domains",
        lazy_progressive.loaded_domain_count()
    );

    let final_stats = lazy_progressive.stats();
    println!("\nFinal statistics:");
    print_stats(&final_stats);
    println!();

    // Example 10: Comparison - Eager vs Lazy
    println!("10. Performance Comparison: Eager vs Lazy");
    println!("{}", "=".repeat(50));

    println!("Scenario: Large schema (1000 domains, 500 predicates)");
    println!("Task: Access only 10 domains");
    println!();

    println!("Eager Loading:");
    println!("  - Load all 1000 domains at startup");
    println!("  - Memory: ~100KB (all domains)");
    println!("  - Startup time: ~100ms");
    println!("  - First access: O(1)");
    println!();

    println!("Lazy Loading:");
    println!("  - Load 0 domains at startup");
    println!("  - Memory: ~1KB (10 domains)");
    println!("  - Startup time: <1ms");
    println!("  - First access: O(1) + I/O");
    println!();

    println!("Winner: Lazy loading for sparse access patterns!");
    println!();

    println!("=== Summary ===");
    println!("✓ Basic lazy loading with on-demand strategy");
    println!("✓ Batch preloading for frequently used elements");
    println!("✓ Multiple loading strategies (OnDemand, Predictive, Batched)");
    println!("✓ Listing available elements without loading");
    println!("✓ Memory management and cache control");
    println!("✓ Performance monitoring with statistics");
    println!("✓ Lazy loading of predicates");
    println!("✓ Batch predicate loading");
    println!("✓ Progressive schema loading use case");
    println!("✓ Performance comparison (Eager vs Lazy)");

    Ok(())
}

fn print_stats(stats: &LazyLoadStats) {
    println!("  Domain loads: {}", stats.domain_loads);
    println!("  Predicate loads: {}", stats.predicate_loads);
    println!("  Cache hits: {}", stats.cache_hits);
    println!("  Cache misses: {}", stats.cache_misses);
    println!("  Hit rate: {:.2}%", stats.hit_rate() * 100.0);
    println!("  Batch loads: {}", stats.batch_loads);
}
