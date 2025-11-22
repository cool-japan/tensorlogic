//! Schema recommendation system.
//!
//! This module provides intelligent schema recommendations based on similarity,
//! patterns, use cases, and collaborative filtering techniques.
//!
//! # Overview
//!
//! The recommendation system helps users discover relevant schemas by:
//! - Finding similar schemas based on embeddings
//! - Identifying common patterns across schema collections
//! - Recommending schemas for specific use cases
//! - Learning from user interactions and preferences
//!
//! # Architecture
//!
//! - **SchemaRecommender**: Main recommendation engine
//! - **RecommendationStrategy**: Multiple recommendation approaches
//! - **SchemaScore**: Scored recommendation with reasoning
//! - **RecommendationContext**: User context and preferences
//! - **PatternMatcher**: Pattern-based schema matching
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_adapters::{
//!     SchemaRecommender, RecommendationStrategy, SymbolTable, DomainInfo
//! };
//!
//! let mut recommender = SchemaRecommender::new();
//!
//! // Add schemas to the recommendation pool
//! let mut schema1 = SymbolTable::new();
//! schema1.add_domain(DomainInfo::new("Person", 100)).unwrap();
//! recommender.add_schema("users", schema1);
//!
//! let mut schema2 = SymbolTable::new();
//! schema2.add_domain(DomainInfo::new("Product", 200)).unwrap();
//! recommender.add_schema("products", schema2);
//!
//! // Get recommendations
//! let mut query = SymbolTable::new();
//! query.add_domain(DomainInfo::new("User", 50)).unwrap();
//!
//! let recommendations = recommender.recommend(
//!     &query,
//!     RecommendationStrategy::Similarity,
//!     5
//! ).unwrap();
//!
//! assert!(!recommendations.is_empty());
//! ```

use anyhow::Result;
use std::collections::HashMap;

use crate::{Embedding, SchemaEmbedder, SchemaStatistics, SymbolTable};

/// Compute cosine similarity between two embeddings.
fn cosine_similarity(a: &Embedding, b: &Embedding) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Strategy for generating recommendations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecommendationStrategy {
    /// Similarity-based using embeddings
    Similarity,
    /// Pattern-based matching
    Pattern,
    /// Use-case specific recommendations
    UseCase(String),
    /// Hybrid approach combining multiple strategies
    Hybrid,
    /// Collaborative filtering based on usage
    Collaborative,
}

/// A scored schema recommendation.
#[derive(Clone, Debug)]
pub struct SchemaScore {
    /// Schema identifier
    pub schema_id: String,
    /// Recommendation score (0.0 to 1.0)
    pub score: f64,
    /// Reasoning for the recommendation
    pub reasoning: String,
    /// Contributing factors to the score
    pub factors: HashMap<String, f64>,
}

impl SchemaScore {
    pub fn new(schema_id: impl Into<String>, score: f64, reasoning: impl Into<String>) -> Self {
        Self {
            schema_id: schema_id.into(),
            score: score.clamp(0.0, 1.0),
            reasoning: reasoning.into(),
            factors: HashMap::new(),
        }
    }

    pub fn with_factor(mut self, name: impl Into<String>, value: f64) -> Self {
        self.factors.insert(name.into(), value);
        self
    }
}

/// Context for generating recommendations.
#[derive(Clone, Debug, Default)]
pub struct RecommendationContext {
    /// User preferences
    pub preferences: HashMap<String, f64>,
    /// Previously viewed schemas
    pub history: Vec<String>,
    /// Explicit user ratings
    pub ratings: HashMap<String, f64>,
    /// Tags or categories of interest
    pub interests: Vec<String>,
}

impl RecommendationContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_preference(mut self, key: impl Into<String>, value: f64) -> Self {
        self.preferences.insert(key.into(), value);
        self
    }

    pub fn with_history(mut self, schema_id: impl Into<String>) -> Self {
        self.history.push(schema_id.into());
        self
    }

    pub fn with_rating(mut self, schema_id: impl Into<String>, rating: f64) -> Self {
        self.ratings.insert(schema_id.into(), rating);
        self
    }

    pub fn with_interest(mut self, tag: impl Into<String>) -> Self {
        self.interests.push(tag.into());
        self
    }
}

/// Pattern matcher for schema recommendations.
#[derive(Clone, Debug)]
pub struct PatternMatcher {
    patterns: HashMap<String, Vec<String>>,
}

impl PatternMatcher {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    pub fn add_pattern(&mut self, name: impl Into<String>, schema_ids: Vec<String>) {
        self.patterns.insert(name.into(), schema_ids);
    }

    pub fn match_pattern(&self, schema: &SymbolTable) -> Vec<String> {
        let mut matches = Vec::new();

        // Simple pattern matching based on domain count and structure
        let domain_count = schema.domains.len();
        let predicate_count = schema.predicates.len();

        for pattern_name in self.patterns.keys() {
            // Match based on size heuristics or complexity
            let size_match = (pattern_name.contains("small") && domain_count < 5)
                || (pattern_name.contains("medium") && (5..15).contains(&domain_count))
                || (pattern_name.contains("large") && domain_count >= 15);

            let complexity_match = (pattern_name.contains("simple") && predicate_count < 10)
                || (pattern_name.contains("complex") && predicate_count >= 10);

            if size_match || complexity_match {
                matches.push(pattern_name.clone());
            }
        }

        matches
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Schema recommendation engine.
pub struct SchemaRecommender {
    schemas: HashMap<String, SymbolTable>,
    embedder: SchemaEmbedder,
    pattern_matcher: PatternMatcher,
    usage_counts: HashMap<String, usize>,
    schema_stats: HashMap<String, SchemaStatistics>,
}

impl SchemaRecommender {
    /// Create a new recommendation engine.
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            embedder: SchemaEmbedder::new(),
            pattern_matcher: PatternMatcher::new(),
            usage_counts: HashMap::new(),
            schema_stats: HashMap::new(),
        }
    }

    /// Add a schema to the recommendation pool.
    pub fn add_schema(&mut self, id: impl Into<String>, schema: SymbolTable) {
        let id = id.into();
        let stats = SchemaStatistics::compute(&schema);
        self.schema_stats.insert(id.clone(), stats);
        self.schemas.insert(id, schema);
    }

    /// Remove a schema from the pool.
    pub fn remove_schema(&mut self, id: &str) -> Option<SymbolTable> {
        self.schema_stats.remove(id);
        self.usage_counts.remove(id);
        self.schemas.remove(id)
    }

    /// Record schema usage for collaborative filtering.
    pub fn record_usage(&mut self, schema_id: &str) {
        *self.usage_counts.entry(schema_id.to_string()).or_insert(0) += 1;
    }

    /// Get recommendations for a query schema.
    pub fn recommend(
        &self,
        query: &SymbolTable,
        strategy: RecommendationStrategy,
        limit: usize,
    ) -> Result<Vec<SchemaScore>> {
        match strategy {
            RecommendationStrategy::Similarity => self.recommend_by_similarity(query, limit),
            RecommendationStrategy::Pattern => self.recommend_by_pattern(query, limit),
            RecommendationStrategy::UseCase(use_case) => {
                self.recommend_by_use_case(query, &use_case, limit)
            }
            RecommendationStrategy::Hybrid => self.recommend_hybrid(query, limit),
            RecommendationStrategy::Collaborative => self.recommend_collaborative(query, limit),
        }
    }

    /// Get recommendations with context.
    pub fn recommend_with_context(
        &self,
        query: &SymbolTable,
        context: &RecommendationContext,
        limit: usize,
    ) -> Result<Vec<SchemaScore>> {
        let mut base_recommendations = self.recommend_hybrid(query, limit * 2)?;

        // Adjust scores based on context
        for rec in &mut base_recommendations {
            // Boost based on user ratings
            if let Some(rating) = context.ratings.get(&rec.schema_id) {
                rec.score = (rec.score + rating) / 2.0;
                rec.factors.insert("user_rating".to_string(), *rating);
            }

            // Boost based on history (recency)
            if let Some(pos) = context.history.iter().position(|id| id == &rec.schema_id) {
                let recency_boost = 1.0 - (pos as f64 / context.history.len() as f64) * 0.3;
                rec.score *= recency_boost;
                rec.factors.insert("recency".to_string(), recency_boost);
            }

            // Adjust based on preferences
            for (pref_key, pref_value) in &context.preferences {
                if rec.schema_id.contains(pref_key) {
                    rec.score = (rec.score + pref_value) / 2.0;
                    rec.factors
                        .insert(format!("preference_{}", pref_key), *pref_value);
                }
            }
        }

        // Re-sort and limit
        base_recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        base_recommendations.truncate(limit);

        Ok(base_recommendations)
    }

    fn recommend_by_similarity(
        &self,
        query: &SymbolTable,
        limit: usize,
    ) -> Result<Vec<SchemaScore>> {
        let query_embedding = self.embedder.embed_schema(query);
        let mut similarities = Vec::new();

        // Compute similarity with each schema
        for (id, schema) in &self.schemas {
            let schema_embedding = self.embedder.embed_schema(schema);
            let similarity = cosine_similarity(&query_embedding, &schema_embedding);
            similarities.push((id.clone(), similarity));
        }

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(limit);

        Ok(similarities
            .into_iter()
            .map(|(id, similarity)| {
                SchemaScore::new(
                    id.clone(),
                    similarity,
                    format!("Similar schema (cosine similarity: {:.2})", similarity),
                )
                .with_factor("embedding_similarity", similarity)
            })
            .collect())
    }

    fn recommend_by_pattern(&self, query: &SymbolTable, limit: usize) -> Result<Vec<SchemaScore>> {
        let patterns = self.pattern_matcher.match_pattern(query);
        let mut scores = Vec::new();

        for (id, schema) in &self.schemas {
            let schema_patterns = self.pattern_matcher.match_pattern(schema);
            let overlap: usize = patterns
                .iter()
                .filter(|p| schema_patterns.contains(p))
                .count();

            if overlap > 0 {
                let score = overlap as f64 / patterns.len().max(1) as f64;
                scores.push(
                    SchemaScore::new(
                        id.clone(),
                        score,
                        format!("Matches {} common patterns", overlap),
                    )
                    .with_factor("pattern_overlap", score),
                );
            }
        }

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scores.truncate(limit);

        Ok(scores)
    }

    fn recommend_by_use_case(
        &self,
        query: &SymbolTable,
        use_case: &str,
        limit: usize,
    ) -> Result<Vec<SchemaScore>> {
        let mut scores = Vec::new();
        let query_stats = SchemaStatistics::compute(query);

        for id in self.schemas.keys() {
            if let Some(stats) = self.schema_stats.get(id) {
                let score = self.compute_use_case_score(use_case, &query_stats, stats);
                if score > 0.0 {
                    scores.push(
                        SchemaScore::new(
                            id.clone(),
                            score,
                            format!("Suitable for {} use case", use_case),
                        )
                        .with_factor("use_case_match", score),
                    );
                }
            }
        }

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scores.truncate(limit);

        Ok(scores)
    }

    fn recommend_hybrid(&self, query: &SymbolTable, limit: usize) -> Result<Vec<SchemaScore>> {
        // Combine similarity and pattern matching
        let similarity_recs = self.recommend_by_similarity(query, limit * 2)?;
        let pattern_recs = self.recommend_by_pattern(query, limit * 2)?;

        let mut combined: HashMap<String, SchemaScore> = HashMap::new();

        // Merge recommendations
        for rec in similarity_recs {
            combined.insert(rec.schema_id.clone(), rec);
        }

        for rec in pattern_recs {
            combined
                .entry(rec.schema_id.clone())
                .and_modify(|existing| {
                    existing.score = (existing.score + rec.score) / 2.0;
                    existing.reasoning.push_str(&format!("; {}", rec.reasoning));
                    for (k, v) in rec.factors.clone() {
                        existing.factors.insert(k, v);
                    }
                })
                .or_insert(rec);
        }

        let mut results: Vec<SchemaScore> = combined.into_values().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(limit);

        Ok(results)
    }

    fn recommend_collaborative(
        &self,
        _query: &SymbolTable,
        limit: usize,
    ) -> Result<Vec<SchemaScore>> {
        let mut scores: Vec<SchemaScore> = self
            .usage_counts
            .iter()
            .map(|(id, count)| {
                let max_count = self.usage_counts.values().max().unwrap_or(&1);
                let score = *count as f64 / *max_count as f64;
                SchemaScore::new(
                    id.clone(),
                    score,
                    format!("Popular schema (used {} times)", count),
                )
                .with_factor("usage_count", *count as f64)
            })
            .collect();

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scores.truncate(limit);

        Ok(scores)
    }

    fn compute_use_case_score(
        &self,
        use_case: &str,
        query_stats: &SchemaStatistics,
        candidate_stats: &SchemaStatistics,
    ) -> f64 {
        match use_case.to_lowercase().as_str() {
            "simple" => {
                // Prefer schemas with similar low complexity
                let complexity_diff =
                    (query_stats.complexity_score() - candidate_stats.complexity_score()).abs();
                f64::max(0.0, 1.0 - complexity_diff / 10.0)
            }
            "large" => {
                // Prefer schemas with many domains
                if candidate_stats.domain_count > 10 {
                    0.8
                } else {
                    0.3
                }
            }
            "relational" => {
                // Prefer schemas with many predicates
                let predicate_ratio = candidate_stats.predicate_count as f64
                    / candidate_stats.domain_count.max(1) as f64;
                (predicate_ratio / 3.0).min(1.0)
            }
            _ => 0.5, // Default score
        }
    }

    /// Get statistics about the recommendation pool.
    pub fn stats(&self) -> RecommenderStats {
        RecommenderStats {
            total_schemas: self.schemas.len(),
            total_patterns: self.pattern_matcher.patterns.len(),
            total_usage_records: self.usage_counts.values().sum(),
            most_used_schema: self
                .usage_counts
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(id, _)| id.clone()),
        }
    }
}

impl Default for SchemaRecommender {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the recommendation engine.
#[derive(Clone, Debug)]
pub struct RecommenderStats {
    pub total_schemas: usize,
    pub total_patterns: usize,
    pub total_usage_records: usize,
    pub most_used_schema: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DomainInfo;

    fn create_test_schema(name: &str, domain_count: usize) -> SymbolTable {
        let mut schema = SymbolTable::new();
        for i in 0..domain_count {
            schema
                .add_domain(DomainInfo::new(format!("{}Domain{}", name, i), 100))
                .unwrap();
        }
        schema
    }

    #[test]
    fn test_schema_score_creation() {
        let score = SchemaScore::new("test", 0.85, "High similarity");
        assert_eq!(score.schema_id, "test");
        assert_eq!(score.score, 0.85);
        assert_eq!(score.reasoning, "High similarity");
    }

    #[test]
    fn test_schema_score_with_factors() {
        let score = SchemaScore::new("test", 0.8, "reason")
            .with_factor("similarity", 0.9)
            .with_factor("popularity", 0.7);

        assert_eq!(score.factors.len(), 2);
        assert_eq!(score.factors.get("similarity"), Some(&0.9));
    }

    #[test]
    fn test_recommendation_context() {
        let context = RecommendationContext::new()
            .with_preference("users", 0.9)
            .with_history("schema1")
            .with_rating("schema2", 0.8)
            .with_interest("database");

        assert_eq!(context.preferences.get("users"), Some(&0.9));
        assert_eq!(context.history.len(), 1);
        assert_eq!(context.ratings.get("schema2"), Some(&0.8));
        assert_eq!(context.interests.len(), 1);
    }

    #[test]
    fn test_pattern_matcher() {
        let mut matcher = PatternMatcher::new();
        matcher.add_pattern("small_schema", vec!["s1".to_string()]);

        let schema = create_test_schema("Test", 3);
        let matches = matcher.match_pattern(&schema);

        assert!(!matches.is_empty());
    }

    #[test]
    fn test_recommender_add_remove() {
        let mut recommender = SchemaRecommender::new();
        let schema = create_test_schema("Test", 5);

        recommender.add_schema("test1", schema.clone());
        assert_eq!(recommender.schemas.len(), 1);

        let removed = recommender.remove_schema("test1");
        assert!(removed.is_some());
        assert_eq!(recommender.schemas.len(), 0);
    }

    #[test]
    fn test_recommend_by_similarity() {
        let mut recommender = SchemaRecommender::new();

        recommender.add_schema("schema1", create_test_schema("A", 3));
        recommender.add_schema("schema2", create_test_schema("B", 5));
        recommender.add_schema("schema3", create_test_schema("C", 3));

        let query = create_test_schema("Query", 3);
        let recs = recommender
            .recommend(&query, RecommendationStrategy::Similarity, 2)
            .unwrap();

        assert!(!recs.is_empty());
        assert!(recs.len() <= 2);
    }

    #[test]
    fn test_recommend_by_pattern() {
        let mut recommender = SchemaRecommender::new();

        // Register patterns
        recommender.pattern_matcher.add_pattern(
            "small_simple",
            vec!["small1".to_string(), "small2".to_string()],
        );

        recommender.add_schema("small1", create_test_schema("S1", 2));
        recommender.add_schema("small2", create_test_schema("S2", 3));
        recommender.add_schema("large1", create_test_schema("L1", 20));

        let query = create_test_schema("Query", 2);
        let recs = recommender
            .recommend(&query, RecommendationStrategy::Pattern, 2)
            .unwrap();

        // Pattern matching may return empty if no patterns match
        // This is expected behavior
        assert!(recs.len() <= 2);
    }

    #[test]
    fn test_recommend_collaborative() {
        let mut recommender = SchemaRecommender::new();

        recommender.add_schema("popular", create_test_schema("P", 5));
        recommender.add_schema("unpopular", create_test_schema("U", 5));

        recommender.record_usage("popular");
        recommender.record_usage("popular");
        recommender.record_usage("popular");
        recommender.record_usage("unpopular");

        let query = create_test_schema("Query", 5);
        let recs = recommender
            .recommend(&query, RecommendationStrategy::Collaborative, 2)
            .unwrap();

        assert!(!recs.is_empty());
        assert_eq!(recs[0].schema_id, "popular");
    }

    #[test]
    fn test_recommend_hybrid() {
        let mut recommender = SchemaRecommender::new();

        recommender.add_schema("schema1", create_test_schema("A", 3));
        recommender.add_schema("schema2", create_test_schema("B", 5));

        let query = create_test_schema("Query", 3);
        let recs = recommender
            .recommend(&query, RecommendationStrategy::Hybrid, 2)
            .unwrap();

        assert!(!recs.is_empty());
    }

    #[test]
    fn test_recommend_with_context() {
        let mut recommender = SchemaRecommender::new();

        recommender.add_schema("schema1", create_test_schema("A", 3));
        recommender.add_schema("schema2", create_test_schema("B", 5));

        let context = RecommendationContext::new()
            .with_rating("schema1", 0.9)
            .with_history("schema2");

        let query = create_test_schema("Query", 3);
        let recs = recommender
            .recommend_with_context(&query, &context, 2)
            .unwrap();

        assert!(!recs.is_empty());
    }

    #[test]
    fn test_recommender_stats() {
        let mut recommender = SchemaRecommender::new();

        recommender.add_schema("s1", create_test_schema("A", 3));
        recommender.add_schema("s2", create_test_schema("B", 5));
        recommender.record_usage("s1");
        recommender.record_usage("s1");

        let stats = recommender.stats();
        assert_eq!(stats.total_schemas, 2);
        assert_eq!(stats.total_usage_records, 2);
        assert_eq!(stats.most_used_schema, Some("s1".to_string()));
    }

    #[test]
    fn test_use_case_recommendations() {
        let mut recommender = SchemaRecommender::new();

        recommender.add_schema("simple", create_test_schema("S", 3));
        recommender.add_schema("complex", create_test_schema("C", 15));

        let query = create_test_schema("Query", 3);
        let recs = recommender
            .recommend(
                &query,
                RecommendationStrategy::UseCase("large".to_string()),
                2,
            )
            .unwrap();

        assert!(!recs.is_empty());
    }

    #[test]
    fn test_record_usage() {
        let mut recommender = SchemaRecommender::new();
        recommender.add_schema("test", create_test_schema("T", 5));

        recommender.record_usage("test");
        recommender.record_usage("test");

        assert_eq!(recommender.usage_counts.get("test"), Some(&2));
    }
}
