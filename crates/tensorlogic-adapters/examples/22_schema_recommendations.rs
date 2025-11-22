//! Schema Recommendation System Example
//!
//! This example demonstrates the schema recommendation engine with various
//! recommendation strategies.

use tensorlogic_adapters::{
    DomainInfo, PredicateInfo, RecommendationContext, RecommendationStrategy, SchemaRecommender,
    SymbolTable,
};

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic Adapters: Schema Recommendation System ===\n");

    // Example 1: Similarity-based recommendations
    example_similarity_recommendations()?;

    // Example 2: Pattern-based recommendations
    example_pattern_recommendations()?;

    // Example 3: Collaborative filtering
    example_collaborative_recommendations()?;

    // Example 4: Context-aware recommendations
    example_context_recommendations()?;

    // Example 5: Use-case specific recommendations
    example_usecase_recommendations()?;

    Ok(())
}

fn example_similarity_recommendations() -> anyhow::Result<()> {
    println!("--- Example 1: Similarity-Based Recommendations ---");

    let mut recommender = SchemaRecommender::new();

    // Add user management schema
    let mut users_schema = SymbolTable::new();
    users_schema.add_domain(DomainInfo::new("User", 1000))?;
    users_schema.add_domain(DomainInfo::new("Role", 10))?;
    users_schema.add_predicate(PredicateInfo::new(
        "hasRole",
        vec!["User".into(), "Role".into()],
    ))?;
    recommender.add_schema("user_management", users_schema);

    // Add customer schema (similar to users)
    let mut customers_schema = SymbolTable::new();
    customers_schema.add_domain(DomainInfo::new("Customer", 5000))?;
    customers_schema.add_domain(DomainInfo::new("Tier", 5))?;
    customers_schema.add_predicate(PredicateInfo::new(
        "hasTier",
        vec!["Customer".into(), "Tier".into()],
    ))?;
    recommender.add_schema("customer_database", customers_schema);

    // Add product catalog (different domain)
    let mut products_schema = SymbolTable::new();
    products_schema.add_domain(DomainInfo::new("Product", 10000))?;
    products_schema.add_domain(DomainInfo::new("Category", 50))?;
    products_schema.add_predicate(PredicateInfo::new(
        "inCategory",
        vec!["Product".into(), "Category".into()],
    ))?;
    recommender.add_schema("product_catalog", products_schema);

    // Query: Looking for a schema similar to user management
    let mut query = SymbolTable::new();
    query.add_domain(DomainInfo::new("Person", 800))?;
    query.add_domain(DomainInfo::new("Group", 20))?;
    query.add_predicate(PredicateInfo::new(
        "belongs",
        vec!["Person".into(), "Group".into()],
    ))?;

    let recommendations = recommender.recommend(&query, RecommendationStrategy::Similarity, 3)?;

    println!("Query: Person management schema");
    println!("\nTop Recommendations:");
    for (i, rec) in recommendations.iter().enumerate() {
        println!(
            "  {}. {} (score: {:.2}) - {}",
            i + 1,
            rec.schema_id,
            rec.score,
            rec.reasoning
        );
        if !rec.factors.is_empty() {
            println!("     Factors: {:?}", rec.factors);
        }
    }

    println!();
    Ok(())
}

fn example_pattern_recommendations() -> anyhow::Result<()> {
    println!("--- Example 2: Pattern-Based Recommendations ---");

    let mut recommender = SchemaRecommender::new();

    // Add various schemas with different sizes
    let mut small1 = SymbolTable::new();
    small1.add_domain(DomainInfo::new("Item", 100))?;
    small1.add_domain(DomainInfo::new("Type", 5))?;
    recommender.add_schema("simple_inventory", small1);

    let mut medium1 = SymbolTable::new();
    for i in 0..8 {
        medium1.add_domain(DomainInfo::new(format!("Domain{}", i), 100))?;
    }
    recommender.add_schema("moderate_system", medium1);

    let mut large1 = SymbolTable::new();
    for i in 0..20 {
        large1.add_domain(DomainInfo::new(format!("Entity{}", i), 100))?;
    }
    recommender.add_schema("enterprise_erp", large1);

    // Query with medium complexity
    let mut query = SymbolTable::new();
    for i in 0..7 {
        query.add_domain(DomainInfo::new(format!("QueryDomain{}", i), 100))?;
    }

    let recommendations = recommender.recommend(&query, RecommendationStrategy::Pattern, 3)?;

    println!("Query: Medium-sized schema (7 domains)");
    println!("\nPattern-Matched Recommendations:");
    for (i, rec) in recommendations.iter().enumerate() {
        println!(
            "  {}. {} (score: {:.2}) - {}",
            i + 1,
            rec.schema_id,
            rec.score,
            rec.reasoning
        );
    }

    println!();
    Ok(())
}

fn example_collaborative_recommendations() -> anyhow::Result<()> {
    println!("--- Example 3: Collaborative Filtering ---");

    let mut recommender = SchemaRecommender::new();

    // Add schemas
    let mut popular = SymbolTable::new();
    popular.add_domain(DomainInfo::new("PopularDomain", 1000))?;
    recommender.add_schema("popular_schema", popular);

    let mut trendy = SymbolTable::new();
    trendy.add_domain(DomainInfo::new("TrendyDomain", 500))?;
    recommender.add_schema("trending_schema", trendy);

    let mut niche = SymbolTable::new();
    niche.add_domain(DomainInfo::new("NicheDomain", 100))?;
    recommender.add_schema("niche_schema", niche);

    // Simulate usage patterns
    println!("Recording usage patterns...");
    for _ in 0..50 {
        recommender.record_usage("popular_schema");
    }
    for _ in 0..30 {
        recommender.record_usage("trending_schema");
    }
    for _ in 0..5 {
        recommender.record_usage("niche_schema");
    }

    let query = SymbolTable::new();
    let recommendations =
        recommender.recommend(&query, RecommendationStrategy::Collaborative, 3)?;

    println!("\nMost Popular Schemas (Collaborative Filtering):");
    for (i, rec) in recommendations.iter().enumerate() {
        println!(
            "  {}. {} (score: {:.2}) - {}",
            i + 1,
            rec.schema_id,
            rec.score,
            rec.reasoning
        );
    }

    let stats = recommender.stats();
    println!("\nRecommender Statistics:");
    println!("  Total schemas: {}", stats.total_schemas);
    println!("  Total usage records: {}", stats.total_usage_records);
    println!(
        "  Most used: {:?}",
        stats.most_used_schema.unwrap_or_default()
    );

    println!();
    Ok(())
}

fn example_context_recommendations() -> anyhow::Result<()> {
    println!("--- Example 4: Context-Aware Recommendations ---");

    let mut recommender = SchemaRecommender::new();

    // Add schemas
    let mut ecommerce = SymbolTable::new();
    ecommerce.add_domain(DomainInfo::new("Product", 5000))?;
    ecommerce.add_domain(DomainInfo::new("Order", 10000))?;
    ecommerce.add_domain(DomainInfo::new("Customer", 3000))?;
    recommender.add_schema("ecommerce_platform", ecommerce);

    let mut analytics = SymbolTable::new();
    analytics.add_domain(DomainInfo::new("Event", 100000))?;
    analytics.add_domain(DomainInfo::new("User", 5000))?;
    analytics.add_domain(DomainInfo::new("Metric", 100))?;
    recommender.add_schema("analytics_system", analytics);

    let mut crm = SymbolTable::new();
    crm.add_domain(DomainInfo::new("Contact", 4000))?;
    crm.add_domain(DomainInfo::new("Deal", 2000))?;
    crm.add_domain(DomainInfo::new("Activity", 8000))?;
    recommender.add_schema("crm_system", crm);

    // Create user context with preferences and history
    let context = RecommendationContext::new()
        .with_preference("ecommerce", 0.9)
        .with_preference("Customer", 0.85)
        .with_history("analytics_system")
        .with_history("crm_system")
        .with_rating("ecommerce_platform", 0.95)
        .with_interest("sales")
        .with_interest("customers");

    println!("User Context:");
    println!("  Preferences: {:?}", context.preferences);
    println!("  History: {:?}", context.history);
    println!("  Ratings: {:?}", context.ratings);
    println!("  Interests: {:?}", context.interests);

    let mut query = SymbolTable::new();
    query.add_domain(DomainInfo::new("Sale", 5000))?;
    query.add_domain(DomainInfo::new("Client", 3000))?;

    let recommendations = recommender.recommend_with_context(&query, &context, 3)?;

    println!("\nContext-Aware Recommendations:");
    for (i, rec) in recommendations.iter().enumerate() {
        println!(
            "  {}. {} (score: {:.2}) - {}",
            i + 1,
            rec.schema_id,
            rec.score,
            rec.reasoning
        );
        if !rec.factors.is_empty() {
            println!("     Contributing factors:");
            for (factor, value) in &rec.factors {
                println!("       - {}: {:.2}", factor, value);
            }
        }
    }

    println!();
    Ok(())
}

fn example_usecase_recommendations() -> anyhow::Result<()> {
    println!("--- Example 5: Use-Case Specific Recommendations ---");

    let mut recommender = SchemaRecommender::new();

    // Simple schema
    let mut simple = SymbolTable::new();
    simple.add_domain(DomainInfo::new("Entity", 100))?;
    simple.add_predicate(PredicateInfo::new("hasProperty", vec!["Entity".into()]))?;
    recommender.add_schema("simple_model", simple);

    // Complex relational schema
    let mut relational = SymbolTable::new();
    relational.add_domain(DomainInfo::new("Table1", 1000))?;
    relational.add_domain(DomainInfo::new("Table2", 2000))?;
    relational.add_domain(DomainInfo::new("Table3", 1500))?;
    relational.add_predicate(PredicateInfo::new(
        "relates1_2",
        vec!["Table1".into(), "Table2".into()],
    ))?;
    relational.add_predicate(PredicateInfo::new(
        "relates2_3",
        vec!["Table2".into(), "Table3".into()],
    ))?;
    relational.add_predicate(PredicateInfo::new(
        "relates1_3",
        vec!["Table1".into(), "Table3".into()],
    ))?;
    recommender.add_schema("relational_db", relational);

    // Large schema
    let mut large = SymbolTable::new();
    for i in 0..25 {
        large.add_domain(DomainInfo::new(format!("Domain{}", i), 500))?;
    }
    recommender.add_schema("enterprise_system", large);

    println!("Testing different use cases:\n");

    // Use case 1: Simple
    let mut query = SymbolTable::new();
    query.add_domain(DomainInfo::new("SimpleEntity", 50))?;

    let simple_recs =
        recommender.recommend(&query, RecommendationStrategy::UseCase("simple".into()), 2)?;
    println!("Use Case: Simple");
    for rec in simple_recs {
        println!("  → {} (score: {:.2})", rec.schema_id, rec.score);
    }

    // Use case 2: Large
    let large_recs =
        recommender.recommend(&query, RecommendationStrategy::UseCase("large".into()), 2)?;
    println!("\nUse Case: Large");
    for rec in large_recs {
        println!("  → {} (score: {:.2})", rec.schema_id, rec.score);
    }

    // Use case 3: Relational
    let relational_recs = recommender.recommend(
        &query,
        RecommendationStrategy::UseCase("relational".into()),
        2,
    )?;
    println!("\nUse Case: Relational");
    for rec in relational_recs {
        println!("  → {} (score: {:.2})", rec.schema_id, rec.score);
    }

    println!();
    Ok(())
}
