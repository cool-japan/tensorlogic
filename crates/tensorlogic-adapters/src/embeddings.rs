//! Schema embeddings for similarity search and ML applications.
//!
//! This module provides functionality to generate vector embeddings for domains,
//! predicates, and entire schemas. These embeddings can be used for:
//! - Similarity search (find similar domains/predicates)
//! - Schema recommendation
//! - Clustering and analysis
//! - ML-based schema completion
//!
//! The embeddings are based on structural and semantic features of the schema elements.

use std::collections::HashMap;

use crate::{DomainInfo, PredicateInfo, SymbolTable};

/// Dimensionality of the embedding vectors.
///
/// Using 64 dimensions provides a good balance between expressiveness
/// and computational efficiency for typical schema sizes.
pub const EMBEDDING_DIM: usize = 64;

/// Vector embedding representation.
pub type Embedding = Vec<f64>;

/// Schema element embedding generator.
///
/// Generates vector embeddings for domains, predicates, and schemas
/// based on their structural and semantic properties.
pub struct SchemaEmbedder {
    /// Whether to normalize embeddings to unit length
    normalize: bool,
    /// Feature weights for embedding computation
    weights: EmbeddingWeights,
}

/// Weights for different embedding features.
#[derive(Clone, Debug)]
pub struct EmbeddingWeights {
    /// Weight for cardinality-based features
    pub cardinality_weight: f64,
    /// Weight for arity-based features
    pub arity_weight: f64,
    /// Weight for name-based features
    pub name_weight: f64,
    /// Weight for structural features
    pub structural_weight: f64,
}

impl Default for EmbeddingWeights {
    fn default() -> Self {
        Self {
            cardinality_weight: 1.0,
            arity_weight: 1.0,
            name_weight: 0.5,
            structural_weight: 0.8,
        }
    }
}

impl SchemaEmbedder {
    /// Create a new schema embedder with default settings.
    pub fn new() -> Self {
        Self {
            normalize: true,
            weights: EmbeddingWeights::default(),
        }
    }

    /// Set whether to normalize embeddings.
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set custom feature weights.
    pub fn with_weights(mut self, weights: EmbeddingWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Generate embedding for a domain.
    pub fn embed_domain(&self, domain: &DomainInfo) -> Embedding {
        let mut embedding = vec![0.0; EMBEDDING_DIM];

        // Cardinality-based features (dimensions 0-15)
        let log_card = (domain.cardinality as f64).ln();
        embedding[0] = log_card * self.weights.cardinality_weight;
        embedding[1] = (domain.cardinality as f64).sqrt() * self.weights.cardinality_weight;
        embedding[2] = (domain.cardinality as f64).cbrt() * self.weights.cardinality_weight;

        // Cardinality ranges (binary features)
        embedding[3] = if domain.cardinality < 10 { 1.0 } else { 0.0 };
        embedding[4] = if domain.cardinality < 100 { 1.0 } else { 0.0 };
        embedding[5] = if domain.cardinality < 1000 { 1.0 } else { 0.0 };
        embedding[6] = if domain.cardinality < 10000 { 1.0 } else { 0.0 };

        // Name-based features (dimensions 16-31)
        self.add_name_features(&mut embedding, &domain.name, 16);

        // Description features (dimensions 32-39)
        if let Some(ref desc) = domain.description {
            embedding[32] = (desc.len() as f64).ln() * self.weights.structural_weight;
            embedding[33] =
                (desc.split_whitespace().count() as f64).ln() * self.weights.structural_weight;
            embedding[34] = if desc.contains("person") || desc.contains("user") {
                1.0
            } else {
                0.0
            };
            embedding[35] = if desc.contains("time") || desc.contains("temporal") {
                1.0
            } else {
                0.0
            };
        }

        // Metadata features (dimensions 40-47)
        if let Some(ref metadata) = domain.metadata {
            embedding[40] = if metadata.provenance.is_some() {
                1.0
            } else {
                0.0
            };
            embedding[41] = metadata.version_history.len() as f64;
            embedding[42] = metadata.tags.len() as f64;
        }

        if self.normalize {
            self.normalize_embedding(&mut embedding);
        }

        embedding
    }

    /// Generate embedding for a predicate.
    pub fn embed_predicate(&self, predicate: &PredicateInfo) -> Embedding {
        let mut embedding = vec![0.0; EMBEDDING_DIM];

        // Arity-based features (dimensions 0-15)
        let arity = predicate.arg_domains.len();
        embedding[0] = arity as f64 * self.weights.arity_weight;
        embedding[1] = (arity as f64).sqrt() * self.weights.arity_weight;

        // Arity ranges (binary features)
        embedding[2] = if arity == 0 { 1.0 } else { 0.0 }; // Nullary
        embedding[3] = if arity == 1 { 1.0 } else { 0.0 }; // Unary
        embedding[4] = if arity == 2 { 1.0 } else { 0.0 }; // Binary
        embedding[5] = if arity == 3 { 1.0 } else { 0.0 }; // Ternary
        embedding[6] = if arity > 3 { 1.0 } else { 0.0 }; // N-ary

        // Name-based features (dimensions 16-31)
        self.add_name_features(&mut embedding, &predicate.name, 16);

        // Constraint features (dimensions 32-47)
        if let Some(ref constraints) = predicate.constraints {
            embedding[32] = constraints.properties.len() as f64 * self.weights.structural_weight;
            embedding[33] = if constraints.properties.iter().any(|p| {
                matches!(
                    p,
                    crate::PredicateProperty::Symmetric | crate::PredicateProperty::Transitive
                )
            }) {
                1.0
            } else {
                0.0
            };
            embedding[34] =
                constraints.functional_dependencies.len() as f64 * self.weights.structural_weight;

            // Count non-None value ranges
            let num_ranges = constraints
                .value_ranges
                .iter()
                .filter(|r| r.is_some())
                .count();
            embedding[35] = num_ranges as f64;
        }

        // Description features (dimensions 48-55)
        if let Some(ref desc) = predicate.description {
            embedding[48] = (desc.len() as f64).ln() * self.weights.structural_weight;
            embedding[49] =
                (desc.split_whitespace().count() as f64).ln() * self.weights.structural_weight;
        }

        if self.normalize {
            self.normalize_embedding(&mut embedding);
        }

        embedding
    }

    /// Generate embedding for an entire schema.
    pub fn embed_schema(&self, table: &SymbolTable) -> Embedding {
        let mut embedding = vec![0.0; EMBEDDING_DIM];

        // Schema size features (dimensions 0-15)
        // Use max(1, len) to avoid ln(0) = -inf
        embedding[0] = ((table.domains.len().max(1)) as f64).ln() * self.weights.structural_weight;
        embedding[1] =
            ((table.predicates.len().max(1)) as f64).ln() * self.weights.structural_weight;
        embedding[2] =
            ((table.variables.len().max(1)) as f64).ln() * self.weights.structural_weight;

        // Total cardinality
        let total_card: usize = table.domains.values().map(|d| d.cardinality).sum();
        embedding[3] = ((total_card.max(1)) as f64).ln() * self.weights.cardinality_weight;

        // Average arity
        let avg_arity: f64 = if table.predicates.is_empty() {
            0.0
        } else {
            table
                .predicates
                .values()
                .map(|p| p.arg_domains.len())
                .sum::<usize>() as f64
                / table.predicates.len() as f64
        };
        embedding[4] = avg_arity * self.weights.arity_weight;

        // Domain histogram (dimensions 16-23)
        for domain in table.domains.values() {
            let log_card = (domain.cardinality as f64).ln();
            let idx = ((log_card / 10.0).min(7.0) as usize).min(7);
            embedding[16 + idx] += 1.0;
        }

        // Arity histogram (dimensions 24-31)
        for predicate in table.predicates.values() {
            let arity = predicate.arg_domains.len().min(7);
            embedding[24 + arity] += 1.0;
        }

        // Graph density (dimension 32)
        let max_edges = table.domains.len() * table.domains.len();
        let actual_edges = table
            .predicates
            .values()
            .filter(|p| p.arg_domains.len() == 2)
            .count();
        embedding[32] = if max_edges > 0 {
            actual_edges as f64 / max_edges as f64
        } else {
            0.0
        };

        if self.normalize {
            self.normalize_embedding(&mut embedding);
        }

        embedding
    }

    /// Compute cosine similarity between two embeddings.
    pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f64 {
        assert_eq!(a.len(), b.len(), "Embeddings must have same dimension");

        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Compute Euclidean distance between two embeddings.
    pub fn euclidean_distance(a: &Embedding, b: &Embedding) -> f64 {
        assert_eq!(a.len(), b.len(), "Embeddings must have same dimension");

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Add name-based features to embedding.
    fn add_name_features(&self, embedding: &mut [f64], name: &str, start_idx: usize) {
        let name_lower = name.to_lowercase();

        // Length features
        embedding[start_idx] = (name.len() as f64).ln() * self.weights.name_weight;
        embedding[start_idx + 1] =
            name.chars().filter(|c| c.is_uppercase()).count() as f64 * self.weights.name_weight;

        // Character distribution
        let vowels = name_lower.chars().filter(|c| "aeiou".contains(*c)).count();
        embedding[start_idx + 2] = vowels as f64 / name.len().max(1) as f64;

        // Common patterns
        embedding[start_idx + 3] = if name_lower.contains('_') { 1.0 } else { 0.0 };
        embedding[start_idx + 4] = if name_lower.starts_with("is") || name_lower.starts_with("has")
        {
            1.0
        } else {
            0.0
        };

        // Domain-specific keywords
        embedding[start_idx + 5] = if name_lower.contains("person")
            || name_lower.contains("user")
            || name_lower.contains("agent")
        {
            1.0
        } else {
            0.0
        };
        embedding[start_idx + 6] = if name_lower.contains("time")
            || name_lower.contains("date")
            || name_lower.contains("temporal")
        {
            1.0
        } else {
            0.0
        };
        embedding[start_idx + 7] = if name_lower.contains("value")
            || name_lower.contains("number")
            || name_lower.contains("count")
        {
            1.0
        } else {
            0.0
        };
    }

    /// Normalize embedding to unit length.
    fn normalize_embedding(&self, embedding: &mut [f64]) {
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }
    }
}

impl Default for SchemaEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

/// Schema similarity search engine.
///
/// Provides functionality to find similar domains, predicates, or schemas
/// based on their embeddings.
pub struct SimilaritySearch {
    embedder: SchemaEmbedder,
    domain_embeddings: HashMap<String, Embedding>,
    predicate_embeddings: HashMap<String, Embedding>,
}

impl SimilaritySearch {
    /// Create a new similarity search engine.
    pub fn new() -> Self {
        Self {
            embedder: SchemaEmbedder::new(),
            domain_embeddings: HashMap::new(),
            predicate_embeddings: HashMap::new(),
        }
    }

    /// Create with custom embedder.
    pub fn with_embedder(embedder: SchemaEmbedder) -> Self {
        Self {
            embedder,
            domain_embeddings: HashMap::new(),
            predicate_embeddings: HashMap::new(),
        }
    }

    /// Index a symbol table for similarity search.
    pub fn index_table(&mut self, table: &SymbolTable) {
        // Index domains
        for (name, domain) in &table.domains {
            let embedding = self.embedder.embed_domain(domain);
            self.domain_embeddings.insert(name.clone(), embedding);
        }

        // Index predicates
        for (name, predicate) in &table.predicates {
            let embedding = self.embedder.embed_predicate(predicate);
            self.predicate_embeddings.insert(name.clone(), embedding);
        }
    }

    /// Find most similar domains to a query domain.
    pub fn find_similar_domains(&self, query: &DomainInfo, top_k: usize) -> Vec<(String, f64)> {
        let query_emb = self.embedder.embed_domain(query);
        self.find_top_k(&self.domain_embeddings, &query_emb, top_k)
    }

    /// Find most similar predicates to a query predicate.
    pub fn find_similar_predicates(
        &self,
        query: &PredicateInfo,
        top_k: usize,
    ) -> Vec<(String, f64)> {
        let query_emb = self.embedder.embed_predicate(query);
        self.find_top_k(&self.predicate_embeddings, &query_emb, top_k)
    }

    /// Find most similar domains by name.
    pub fn find_similar_domains_by_name(&self, name: &str, top_k: usize) -> Vec<(String, f64)> {
        if let Some(query_emb) = self.domain_embeddings.get(name) {
            self.find_top_k(&self.domain_embeddings, query_emb, top_k + 1)
                .into_iter()
                .filter(|(n, _)| n != name)
                .take(top_k)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find most similar predicates by name.
    pub fn find_similar_predicates_by_name(&self, name: &str, top_k: usize) -> Vec<(String, f64)> {
        if let Some(query_emb) = self.predicate_embeddings.get(name) {
            self.find_top_k(&self.predicate_embeddings, query_emb, top_k + 1)
                .into_iter()
                .filter(|(n, _)| n != name)
                .take(top_k)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get statistics about indexed elements.
    pub fn stats(&self) -> SimilarityStats {
        SimilarityStats {
            num_domains: self.domain_embeddings.len(),
            num_predicates: self.predicate_embeddings.len(),
            embedding_dim: EMBEDDING_DIM,
        }
    }

    /// Internal: Find top-k similar items from a set of embeddings.
    fn find_top_k(
        &self,
        embeddings: &HashMap<String, Embedding>,
        query: &Embedding,
        k: usize,
    ) -> Vec<(String, f64)> {
        let mut similarities: Vec<(String, f64)> = embeddings
            .iter()
            .map(|(name, emb)| {
                let sim = SchemaEmbedder::cosine_similarity(query, emb);
                (name.clone(), sim)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return top k
        similarities.into_iter().take(k).collect()
    }
}

impl Default for SimilaritySearch {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about indexed elements in similarity search.
#[derive(Clone, Debug)]
pub struct SimilarityStats {
    /// Number of indexed domains
    pub num_domains: usize,
    /// Number of indexed predicates
    pub num_predicates: usize,
    /// Embedding dimensionality
    pub embedding_dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_embedding_generation() {
        let domain = DomainInfo::new("Person", 100);
        let embedder = SchemaEmbedder::new();
        let embedding = embedder.embed_domain(&domain);

        assert_eq!(embedding.len(), EMBEDDING_DIM);
        // Normalized embeddings should have unit length
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_predicate_embedding_generation() {
        let predicate =
            PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()]);
        let embedder = SchemaEmbedder::new();
        let embedding = embedder.embed_predicate(&predicate);

        assert_eq!(embedding.len(), EMBEDDING_DIM);
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_schema_embedding_generation() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Course", 50)).unwrap();

        let embedder = SchemaEmbedder::new();
        let embedding = embedder.embed_schema(&table);

        assert_eq!(embedding.len(), EMBEDDING_DIM);
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((SchemaEmbedder::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((SchemaEmbedder::cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];

        let dist = SchemaEmbedder::euclidean_distance(&a, &b);
        assert!((dist - 3.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_search_indexing() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Student", 50)).unwrap();
        table.add_domain(DomainInfo::new("Course", 30)).unwrap();

        let mut search = SimilaritySearch::new();
        search.index_table(&table);

        let stats = search.stats();
        assert_eq!(stats.num_domains, 3);
        assert_eq!(stats.embedding_dim, EMBEDDING_DIM);
    }

    #[test]
    fn test_find_similar_domains() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Student", 80)).unwrap();
        table.add_domain(DomainInfo::new("Course", 50)).unwrap();

        let mut search = SimilaritySearch::new();
        search.index_table(&table);

        let query = DomainInfo::new("Teacher", 90);
        let similar = search.find_similar_domains(&query, 2);

        assert_eq!(similar.len(), 2);
        // Teacher (90) should be most similar to Person (100) and Student (80)
        assert!(similar[0].1 > 0.5); // High similarity
    }

    #[test]
    fn test_find_similar_predicates() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let knows = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()]);
        let likes = PredicateInfo::new("likes", vec!["Person".to_string(), "Person".to_string()]);
        let teaches =
            PredicateInfo::new("teaches", vec!["Person".to_string(), "Person".to_string()]);

        table.add_predicate(knows).unwrap();
        table.add_predicate(likes).unwrap();
        table.add_predicate(teaches).unwrap();

        let mut search = SimilaritySearch::new();
        search.index_table(&table);

        let query = PredicateInfo::new("loves", vec!["Person".to_string(), "Person".to_string()]);
        let similar = search.find_similar_predicates(&query, 3);

        assert_eq!(similar.len(), 3);
        // All binary predicates should have high similarity
        for (_, sim) in &similar {
            assert!(*sim > 0.8);
        }
    }

    #[test]
    fn test_similar_domains_by_name() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Student", 80)).unwrap();
        table.add_domain(DomainInfo::new("Course", 50)).unwrap();

        let mut search = SimilaritySearch::new();
        search.index_table(&table);

        let similar = search.find_similar_domains_by_name("Person", 2);

        assert_eq!(similar.len(), 2);
        // Should not include "Person" itself
        assert!(!similar.iter().any(|(n, _)| n == "Person"));
    }

    #[test]
    fn test_unnormalized_embeddings() {
        let embedder = SchemaEmbedder::new().with_normalization(false);
        let domain = DomainInfo::new("Person", 100);
        let embedding = embedder.embed_domain(&domain);

        assert_eq!(embedding.len(), EMBEDDING_DIM);
        // Unnormalized embeddings may not have unit length
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        // But should have non-zero length
        assert!(norm > 0.0);
    }

    #[test]
    fn test_custom_weights() {
        let weights = EmbeddingWeights {
            cardinality_weight: 2.0,
            arity_weight: 1.0,
            name_weight: 0.5,
            structural_weight: 0.8,
        };

        let embedder = SchemaEmbedder::new().with_weights(weights);
        let domain = DomainInfo::new("Person", 100);
        let embedding = embedder.embed_domain(&domain);

        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_empty_schema_embedding() {
        let table = SymbolTable::new();
        let embedder = SchemaEmbedder::new();
        let embedding = embedder.embed_schema(&table);

        assert_eq!(embedding.len(), EMBEDDING_DIM);
        // Empty schema should still produce valid embedding
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm >= 0.0);
    }

    #[test]
    fn test_similarity_transitivity() {
        let embedder = SchemaEmbedder::new();

        let d1 = DomainInfo::new("Person", 100);
        let d2 = DomainInfo::new("Student", 90);
        let d3 = DomainInfo::new("Teacher", 95);

        let e1 = embedder.embed_domain(&d1);
        let e2 = embedder.embed_domain(&d2);
        let e3 = embedder.embed_domain(&d3);

        let sim_12 = SchemaEmbedder::cosine_similarity(&e1, &e2);
        let sim_13 = SchemaEmbedder::cosine_similarity(&e1, &e3);
        let sim_23 = SchemaEmbedder::cosine_similarity(&e2, &e3);

        // All should be highly similar (same cardinality range)
        assert!(sim_12 > 0.8);
        assert!(sim_13 > 0.8);
        assert!(sim_23 > 0.8);
    }
}
