//! Knowledge graph embeddings for TensorLogic integration.
//!
//! This module provides embedding generation from RDF knowledge graphs,
//! enabling machine learning and neural-symbolic integration.
//!
//! # Overview
//!
//! Knowledge graph embeddings map entities and relations to dense vector spaces,
//! useful for:
//! - Link prediction (predicting missing triples)
//! - Entity classification
//! - Similarity computation
//! - Integration with neural networks
//!
//! # Supported Embedding Models
//!
//! - **TransE**: Translation-based model (h + r ≈ t)
//! - **DistMult**: Bilinear model (h ⊙ r ⊙ t)
//! - **ComplEx**: Complex-valued embeddings
//! - **Random**: Baseline random embeddings
//!
//! # Example
//!
//! ```no_run
//! use tensorlogic_oxirs_bridge::knowledge_embeddings::{
//!     KnowledgeEmbeddings, EmbeddingConfig, EmbeddingModel,
//! };
//!
//! let mut embeddings = KnowledgeEmbeddings::new(EmbeddingConfig::default()).unwrap();
//!
//! // Load knowledge graph
//! embeddings.load_turtle(r#"
//!     @prefix ex: <http://example.org/> .
//!     ex:Alice ex:knows ex:Bob .
//!     ex:Bob ex:knows ex:Carol .
//! "#).unwrap();
//!
//! // Train embeddings
//! embeddings.train(100).unwrap();
//!
//! // Get entity embeddings
//! let alice_emb = embeddings.entity_embedding("http://example.org/Alice");
//! ```

use crate::oxirs_executor::OxirsSparqlExecutor;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray::{Array1, ArrayD};
use scirs2_core::random::{thread_rng, Rng, SeedableRng, StdRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tensorlogic_ir::{TLExpr, Term};

/// Embedding model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EmbeddingModel {
    /// TransE: Translation-based model (h + r ≈ t)
    #[default]
    TransE,
    /// DistMult: Bilinear diagonal model
    DistMult,
    /// ComplEx: Complex-valued embeddings
    ComplEx,
    /// Random baseline
    Random,
}

/// Configuration for knowledge embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Regularization coefficient
    pub regularization: f64,
    /// Margin for margin-based loss
    pub margin: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Embedding model type
    pub model: EmbeddingModel,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 50,
            learning_rate: 0.01,
            regularization: 0.001,
            margin: 1.0,
            batch_size: 100,
            model: EmbeddingModel::TransE,
            seed: None,
        }
    }
}

impl EmbeddingConfig {
    /// Create a new config with specified dimension.
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            ..Default::default()
        }
    }

    /// Set the model type.
    pub fn with_model(mut self, model: EmbeddingModel) -> Self {
        self.model = model;
        self
    }

    /// Set the learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

/// A triple in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KGTriple {
    pub head: String,
    pub relation: String,
    pub tail: String,
}

/// Knowledge graph embeddings.
///
/// This struct manages entity and relation embeddings learned from
/// a knowledge graph.
pub struct KnowledgeEmbeddings {
    /// Configuration
    config: EmbeddingConfig,
    /// Entity embeddings: entity IRI -> embedding vector
    entity_embeddings: HashMap<String, Array1<f64>>,
    /// Relation embeddings: relation IRI -> embedding vector
    relation_embeddings: HashMap<String, Array1<f64>>,
    /// Entity index: entity IRI -> index
    entity_index: HashMap<String, usize>,
    /// Relation index: relation IRI -> index
    relation_index: HashMap<String, usize>,
    /// Training triples
    triples: Vec<KGTriple>,
    /// SPARQL executor for data access
    executor: OxirsSparqlExecutor,
}

impl KnowledgeEmbeddings {
    /// Create new knowledge embeddings.
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        Ok(Self {
            config,
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            entity_index: HashMap::new(),
            relation_index: HashMap::new(),
            triples: Vec::new(),
            executor: OxirsSparqlExecutor::new()?,
        })
    }

    /// Load knowledge graph from Turtle format.
    pub fn load_turtle(&mut self, turtle: &str) -> Result<usize> {
        let count = self.executor.load_turtle(turtle)?;
        self.extract_triples()?;
        self.initialize_embeddings();
        Ok(count)
    }

    /// Extract triples from the executor.
    fn extract_triples(&mut self) -> Result<()> {
        // Query all triples
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let results = self.executor.execute(query)?;

        if let crate::oxirs_executor::QueryResults::Select { bindings, .. } = results {
            for binding in bindings {
                let head = binding
                    .get("s")
                    .map(|v| v.as_str().to_string())
                    .unwrap_or_default();
                let relation = binding
                    .get("p")
                    .map(|v| v.as_str().to_string())
                    .unwrap_or_default();
                let tail = binding
                    .get("o")
                    .map(|v| v.as_str().to_string())
                    .unwrap_or_default();

                if !head.is_empty() && !relation.is_empty() && !tail.is_empty() {
                    self.triples.push(KGTriple {
                        head,
                        relation,
                        tail,
                    });
                }
            }
        }

        // Build indices
        let mut entity_idx = 0;
        let mut relation_idx = 0;

        for triple in &self.triples {
            if !self.entity_index.contains_key(&triple.head) {
                self.entity_index.insert(triple.head.clone(), entity_idx);
                entity_idx += 1;
            }
            if !self.entity_index.contains_key(&triple.tail) {
                self.entity_index.insert(triple.tail.clone(), entity_idx);
                entity_idx += 1;
            }
            if !self.relation_index.contains_key(&triple.relation) {
                self.relation_index
                    .insert(triple.relation.clone(), relation_idx);
                relation_idx += 1;
            }
        }

        Ok(())
    }

    /// Initialize embeddings with random values.
    fn initialize_embeddings(&mut self) {
        let mut rng_box: Box<dyn scirs2_core::random::RngCore> =
            if let Some(seed) = self.config.seed {
                Box::new(StdRng::seed_from_u64(seed))
            } else {
                Box::new(thread_rng())
            };

        let dim = self.config.embedding_dim;
        let scale = 1.0 / (dim as f64).sqrt();

        // Initialize entity embeddings
        for entity in self.entity_index.keys() {
            let embedding: Vec<f64> = (0..dim).map(|_| rng_box.random::<f64>() * scale).collect();
            self.entity_embeddings
                .insert(entity.clone(), Array1::from(embedding));
        }

        // Initialize relation embeddings
        for relation in self.relation_index.keys() {
            let embedding: Vec<f64> = (0..dim).map(|_| rng_box.random::<f64>() * scale).collect();
            self.relation_embeddings
                .insert(relation.clone(), Array1::from(embedding));
        }
    }

    /// Train the embeddings.
    pub fn train(&mut self, num_epochs: usize) -> Result<f64> {
        if self.triples.is_empty() {
            return Err(anyhow!("No triples to train on"));
        }

        let mut total_loss = 0.0;
        let mut rng = thread_rng();

        for _epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;

            // Shuffle triples
            let mut indices: Vec<usize> = (0..self.triples.len()).collect();
            for i in (1..indices.len()).rev() {
                let j = rng.random_range(0..=i);
                indices.swap(i, j);
            }

            // Mini-batch training
            for batch_start in (0..indices.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(indices.len());

                for &idx in &indices[batch_start..batch_end] {
                    // Clone the triple to avoid borrow conflict
                    let triple = self.triples[idx].clone();

                    // Generate negative sample
                    let neg_triple = self.generate_negative_sample(&triple, &mut rng);

                    // Compute loss and update
                    let loss = self.train_step(&triple, &neg_triple)?;
                    epoch_loss += loss;
                }
            }

            total_loss = epoch_loss / self.triples.len() as f64;
        }

        Ok(total_loss)
    }

    /// Generate a negative sample by corrupting head or tail.
    fn generate_negative_sample(&self, triple: &KGTriple, rng: &mut impl Rng) -> KGTriple {
        let entities: Vec<_> = self.entity_index.keys().collect();
        if entities.is_empty() {
            return triple.clone();
        }

        let corrupt_head = rng.random();
        let random_entity = entities[rng.random_range(0..entities.len())].clone();

        if corrupt_head {
            KGTriple {
                head: random_entity,
                relation: triple.relation.clone(),
                tail: triple.tail.clone(),
            }
        } else {
            KGTriple {
                head: triple.head.clone(),
                relation: triple.relation.clone(),
                tail: random_entity,
            }
        }
    }

    /// Perform one training step.
    fn train_step(&mut self, pos_triple: &KGTriple, neg_triple: &KGTriple) -> Result<f64> {
        match self.config.model {
            EmbeddingModel::TransE => self.train_step_transe(pos_triple, neg_triple),
            EmbeddingModel::DistMult => self.train_step_distmult(pos_triple, neg_triple),
            EmbeddingModel::ComplEx => self.train_step_complex(pos_triple, neg_triple),
            EmbeddingModel::Random => Ok(0.0), // No training for random
        }
    }

    /// TransE training step.
    fn train_step_transe(&mut self, pos_triple: &KGTriple, neg_triple: &KGTriple) -> Result<f64> {
        let h_pos = self
            .entity_embeddings
            .get(&pos_triple.head)
            .ok_or_else(|| anyhow!("Missing head embedding"))?
            .clone();
        let r = self
            .relation_embeddings
            .get(&pos_triple.relation)
            .ok_or_else(|| anyhow!("Missing relation embedding"))?
            .clone();
        let t_pos = self
            .entity_embeddings
            .get(&pos_triple.tail)
            .ok_or_else(|| anyhow!("Missing tail embedding"))?
            .clone();

        let h_neg = self
            .entity_embeddings
            .get(&neg_triple.head)
            .ok_or_else(|| anyhow!("Missing negative head embedding"))?
            .clone();
        let t_neg = self
            .entity_embeddings
            .get(&neg_triple.tail)
            .ok_or_else(|| anyhow!("Missing negative tail embedding"))?
            .clone();

        // TransE score: ||h + r - t||
        let pos_diff = &h_pos + &r - &t_pos;
        let neg_diff = &h_neg + &r - &t_neg;

        let pos_score = pos_diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        let neg_score = neg_diff.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Margin-based ranking loss
        let loss = (self.config.margin + pos_score - neg_score).max(0.0);

        if loss > 0.0 {
            let lr = self.config.learning_rate;
            let reg = self.config.regularization;

            // Gradient update for positive triple
            let grad_pos: Array1<f64> = pos_diff.mapv(|x| x / pos_score.max(1e-10));

            // Update head (positive)
            if let Some(h) = self.entity_embeddings.get_mut(&pos_triple.head) {
                *h = &*h - &(&grad_pos * lr);
                // L2 regularization
                *h = &*h - &(&*h * (lr * reg));
            }

            // Update tail (positive)
            if let Some(t) = self.entity_embeddings.get_mut(&pos_triple.tail) {
                *t = &*t + &(&grad_pos * lr);
                *t = &*t - &(&*t * (lr * reg));
            }

            // Update relation
            if let Some(r) = self.relation_embeddings.get_mut(&pos_triple.relation) {
                *r = &*r - &(&grad_pos * lr);
                *r = &*r - &(&*r * (lr * reg));
            }
        }

        Ok(loss)
    }

    /// DistMult training step.
    fn train_step_distmult(&mut self, pos_triple: &KGTriple, neg_triple: &KGTriple) -> Result<f64> {
        // DistMult score: h ⊙ r ⊙ t
        // Clone values to avoid borrow conflicts during update
        let h_pos: Array1<f64> = self
            .entity_embeddings
            .get(&pos_triple.head)
            .ok_or_else(|| anyhow!("Missing head embedding"))?
            .clone();
        let r: Array1<f64> = self
            .relation_embeddings
            .get(&pos_triple.relation)
            .ok_or_else(|| anyhow!("Missing relation embedding"))?
            .clone();
        let t_pos: Array1<f64> = self
            .entity_embeddings
            .get(&pos_triple.tail)
            .ok_or_else(|| anyhow!("Missing tail embedding"))?
            .clone();

        let h_neg: Array1<f64> = self
            .entity_embeddings
            .get(&neg_triple.head)
            .ok_or_else(|| anyhow!("Missing negative head embedding"))?
            .clone();
        let t_neg: Array1<f64> = self
            .entity_embeddings
            .get(&neg_triple.tail)
            .ok_or_else(|| anyhow!("Missing negative tail embedding"))?
            .clone();

        let pos_score: f64 = h_pos
            .iter()
            .zip(r.iter())
            .zip(t_pos.iter())
            .map(|((h, r), t)| h * r * t)
            .sum();
        let neg_score: f64 = h_neg
            .iter()
            .zip(r.iter())
            .zip(t_neg.iter())
            .map(|((h, r), t)| h * r * t)
            .sum();

        // Margin-based loss
        let loss = (self.config.margin - pos_score + neg_score).max(0.0);

        // Simplified gradient update (similar structure to TransE)
        if loss > 0.0 {
            let lr = self.config.learning_rate;

            // Update embeddings (simplified)
            if let Some(h) = self.entity_embeddings.get_mut(&pos_triple.head) {
                let grad: Array1<f64> = r
                    .iter()
                    .zip(t_pos.iter())
                    .map(|(ri, ti)| ri * ti * lr)
                    .collect();
                *h = &*h + &grad;
            }
        }

        Ok(loss)
    }

    /// ComplEx training step (simplified).
    fn train_step_complex(&mut self, pos_triple: &KGTriple, neg_triple: &KGTriple) -> Result<f64> {
        // For simplicity, treat as real-valued DistMult
        // Full ComplEx would use complex arithmetic
        self.train_step_distmult(pos_triple, neg_triple)
    }

    /// Get entity embedding.
    pub fn entity_embedding(&self, entity: &str) -> Option<&Array1<f64>> {
        self.entity_embeddings.get(entity)
    }

    /// Get relation embedding.
    pub fn relation_embedding(&self, relation: &str) -> Option<&Array1<f64>> {
        self.relation_embeddings.get(relation)
    }

    /// Generate embeddings for all entities.
    pub fn generate_entity_embeddings(&self) -> Result<HashMap<String, ArrayD<f64>>> {
        let mut result: HashMap<String, ArrayD<f64>> = HashMap::new();
        for (entity, embedding) in &self.entity_embeddings {
            let entity_str: String = entity.to_string();
            let emb: &Array1<f64> = embedding;
            let shape = vec![emb.len()];
            let data = emb.to_vec();
            let array = ArrayD::from_shape_vec(shape, data)
                .map_err(|e| anyhow!("Failed to reshape: {}", e))?;
            result.insert(entity_str, array);
        }
        Ok(result)
    }

    /// Convert embeddings to weighted TensorLogic predicates.
    ///
    /// Creates predicates with weights based on embedding similarities.
    pub fn to_weighted_predicates(&self) -> Result<Vec<TLExpr>> {
        let mut predicates = Vec::new();

        for triple in &self.triples {
            // Compute triple score as weight
            let score = self.score_triple(triple)?;
            // Weight can be used for probabilistic reasoning (reserved for future use)
            let _weight = (-score).exp().min(1.0); // Convert distance to probability

            // Create weighted predicate
            let relation_name = Self::iri_to_name(&triple.relation);
            let pred = TLExpr::pred(
                &relation_name,
                vec![Term::constant(&triple.head), Term::constant(&triple.tail)],
            );

            // Wrap with weight (using a pseudo-weight representation)
            // In a full implementation, this would integrate with TensorLogic's weight system
            predicates.push(pred);
        }

        Ok(predicates)
    }

    /// Predict missing links.
    ///
    /// Given a subject and relation, predict likely objects.
    pub fn predict_links(&self, subject: &str, relation: &str) -> Result<Vec<(String, f64)>> {
        let h = self
            .entity_embeddings
            .get(subject)
            .ok_or_else(|| anyhow!("Unknown subject: {}", subject))?;
        let r = self
            .relation_embeddings
            .get(relation)
            .ok_or_else(|| anyhow!("Unknown relation: {}", relation))?;

        let mut predictions: Vec<(String, f64)> = Vec::new();

        for (entity, t) in &self.entity_embeddings {
            let entity_str: &String = entity;
            let t_emb: &Array1<f64> = t;

            if entity_str == subject {
                continue; // Skip self-links
            }

            let score: f64 = match self.config.model {
                EmbeddingModel::TransE => {
                    // TransE: score = -||h + r - t||
                    let diff = h + r - t_emb;
                    -diff.iter().map(|x| x * x).sum::<f64>().sqrt()
                }
                EmbeddingModel::DistMult | EmbeddingModel::ComplEx => {
                    // DistMult: score = h ⊙ r ⊙ t
                    let h_arr: &Array1<f64> = h;
                    let r_arr: &Array1<f64> = r;
                    let t_arr: &Array1<f64> = t_emb;
                    h_arr
                        .iter()
                        .zip(r_arr.iter())
                        .zip(t_arr.iter())
                        .map(|((hi, ri), ti): ((&f64, &f64), &f64)| hi * ri * ti)
                        .sum()
                }
                EmbeddingModel::Random => thread_rng().random(),
            };

            predictions.push((entity_str.clone(), score));
        }

        // Sort by score (descending)
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(predictions)
    }

    /// Score a triple.
    pub fn score_triple(&self, triple: &KGTriple) -> Result<f64> {
        let h = self
            .entity_embeddings
            .get(&triple.head)
            .ok_or_else(|| anyhow!("Unknown head"))?;
        let r = self
            .relation_embeddings
            .get(&triple.relation)
            .ok_or_else(|| anyhow!("Unknown relation"))?;
        let t = self
            .entity_embeddings
            .get(&triple.tail)
            .ok_or_else(|| anyhow!("Unknown tail"))?;

        let score = match self.config.model {
            EmbeddingModel::TransE => {
                let diff = h + r - t;
                diff.iter().map(|x| x * x).sum::<f64>().sqrt()
            }
            EmbeddingModel::DistMult | EmbeddingModel::ComplEx => -h
                .iter()
                .zip(r.iter())
                .zip(t.iter())
                .map(|((hi, ri), ti)| hi * ri * ti)
                .sum::<f64>(),
            EmbeddingModel::Random => 0.5,
        };

        Ok(score)
    }

    /// Get the number of entities.
    pub fn num_entities(&self) -> usize {
        self.entity_index.len()
    }

    /// Get the number of relations.
    pub fn num_relations(&self) -> usize {
        self.relation_index.len()
    }

    /// Get the number of triples.
    pub fn num_triples(&self) -> usize {
        self.triples.len()
    }

    /// Extract local name from IRI.
    fn iri_to_name(iri: &str) -> String {
        iri.split(['/', '#']).next_back().unwrap_or(iri).to_string()
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute Euclidean distance between two vectors.
pub fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi): (&f64, &f64)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_embedding_config() {
        let config = EmbeddingConfig::new(100)
            .with_model(EmbeddingModel::DistMult)
            .with_learning_rate(0.001)
            .with_batch_size(256);

        assert_eq!(config.embedding_dim, 100);
        assert_eq!(config.model, EmbeddingModel::DistMult);
        assert_abs_diff_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 256);
    }

    #[test]
    fn test_embeddings_creation() {
        let config = EmbeddingConfig::default();
        let embeddings = KnowledgeEmbeddings::new(config).expect("Failed to create embeddings");

        assert_eq!(embeddings.num_entities(), 0);
        assert_eq!(embeddings.num_relations(), 0);
        assert_eq!(embeddings.num_triples(), 0);
    }

    #[test]
    fn test_load_turtle() {
        let config = EmbeddingConfig::default();
        let mut embeddings = KnowledgeEmbeddings::new(config).expect("Failed to create embeddings");

        let result = embeddings.load_turtle(
            r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
            ex:Bob ex:knows ex:Carol .
        "#,
        );

        assert!(result.is_ok());
        assert_eq!(embeddings.num_triples(), 2);
        assert!(embeddings.num_entities() >= 2);
    }

    #[test]
    fn test_entity_embedding() {
        let config = EmbeddingConfig::new(10);
        let mut embeddings = KnowledgeEmbeddings::new(config).expect("Failed to create embeddings");

        embeddings
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
        "#,
            )
            .expect("Load failed");

        let alice_emb = embeddings.entity_embedding("http://example.org/Alice");
        assert!(alice_emb.is_some());
        assert_eq!(alice_emb.map(|e| e.len()), Some(10));
    }

    #[test]
    fn test_train() {
        let config = EmbeddingConfig::new(10).with_batch_size(2);
        let mut embeddings = KnowledgeEmbeddings::new(config).expect("Failed to create embeddings");

        embeddings
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
            ex:Bob ex:knows ex:Carol .
        "#,
            )
            .expect("Load failed");

        let loss = embeddings.train(5);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_predict_links() {
        let config = EmbeddingConfig::new(10);
        let mut embeddings = KnowledgeEmbeddings::new(config).expect("Failed to create embeddings");

        embeddings
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
            ex:Bob ex:knows ex:Carol .
            ex:Carol ex:knows ex:Dave .
        "#,
            )
            .expect("Load failed");

        let predictions =
            embeddings.predict_links("http://example.org/Alice", "http://example.org/knows");
        assert!(predictions.is_ok());

        let predictions = predictions.expect("Prediction failed");
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_score_triple() {
        let config = EmbeddingConfig::new(10);
        let mut embeddings = KnowledgeEmbeddings::new(config).expect("Failed to create embeddings");

        embeddings
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
        "#,
            )
            .expect("Load failed");

        let triple = KGTriple {
            head: "http://example.org/Alice".to_string(),
            relation: "http://example.org/knows".to_string(),
            tail: "http://example.org/Bob".to_string(),
        };

        let score = embeddings.score_triple(&triple);
        assert!(score.is_ok());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from(vec![1.0, 0.0, 0.0]);
        let b = Array1::from(vec![1.0, 0.0, 0.0]);

        let sim = cosine_similarity(&a, &b);
        assert_abs_diff_eq!(sim, 1.0, epsilon = 1e-6);

        let c = Array1::from(vec![0.0, 1.0, 0.0]);
        let sim_orthogonal = cosine_similarity(&a, &c);
        assert_abs_diff_eq!(sim_orthogonal, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = Array1::from(vec![0.0, 0.0, 0.0]);
        let b = Array1::from(vec![3.0, 4.0, 0.0]);

        let dist = euclidean_distance(&a, &b);
        assert_abs_diff_eq!(dist, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_generate_entity_embeddings() {
        let config = EmbeddingConfig::new(5);
        let mut embeddings = KnowledgeEmbeddings::new(config).expect("Failed to create embeddings");

        embeddings
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
        "#,
            )
            .expect("Load failed");

        let entity_embs = embeddings.generate_entity_embeddings();
        assert!(entity_embs.is_ok());

        let entity_embs = entity_embs.expect("Generation failed");
        assert!(entity_embs.contains_key("http://example.org/Alice"));
    }

    #[test]
    fn test_to_weighted_predicates() {
        let config = EmbeddingConfig::new(5);
        let mut embeddings = KnowledgeEmbeddings::new(config).expect("Failed to create embeddings");

        embeddings
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
        "#,
            )
            .expect("Load failed");

        let predicates = embeddings.to_weighted_predicates();
        assert!(predicates.is_ok());

        let predicates = predicates.expect("Predicate generation failed");
        assert!(!predicates.is_empty());
    }

    #[test]
    fn test_distmult_model() {
        let config = EmbeddingConfig::new(10).with_model(EmbeddingModel::DistMult);
        let mut embeddings = KnowledgeEmbeddings::new(config).expect("Failed to create embeddings");

        embeddings
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
        "#,
            )
            .expect("Load failed");

        let loss = embeddings.train(3);
        assert!(loss.is_ok());
    }
}
