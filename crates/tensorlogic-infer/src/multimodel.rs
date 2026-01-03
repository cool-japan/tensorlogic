//! Multi-model coordination for ensemble and multi-task inference.
//!
//! This module provides coordination capabilities for running multiple models:
//! - Ensemble inference (voting, averaging, stacking)
//! - Multi-task model coordination
//! - Model cascades (early-exit strategies)
//! - Model routing (dynamic model selection)
//! - Resource sharing across models
//! - Load balancing for model serving

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tensorlogic_ir::EinsumGraph;
use thiserror::Error;

/// Multi-model coordination errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MultiModelError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Incompatible model outputs")]
    IncompatibleOutputs,

    #[error("Invalid ensemble configuration: {0}")]
    InvalidEnsemble(String),

    #[error("Model routing failed: {0}")]
    RoutingFailed(String),

    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
}

/// Ensemble aggregation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnsembleStrategy {
    /// Simple average of predictions
    Average,
    /// Weighted average with learned weights
    WeightedAverage,
    /// Majority voting (for classification)
    MajorityVote,
    /// Soft voting with probabilities
    SoftVote,
    /// Stacking with meta-learner
    Stacking,
    /// Boosting-style weighted combination
    Boosting,
    /// Maximum confidence prediction
    MaxConfidence,
}

/// Model metadata for coordination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Expected input shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Expected output shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Model weight (for ensemble)
    pub weight: f64,
    /// Priority (for routing)
    pub priority: u32,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Resource requirements for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Memory required (bytes)
    pub memory_bytes: usize,
    /// GPU memory required (bytes)
    pub gpu_memory_bytes: Option<usize>,
    /// Estimated FLOPS
    pub estimated_flops: f64,
    /// Estimated latency (milliseconds)
    pub estimated_latency_ms: f64,
}

/// Ensemble configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Ensemble strategy
    pub strategy: EnsembleStrategy,
    /// Model weights (for weighted averaging)
    pub model_weights: HashMap<String, f64>,
    /// Minimum models for consensus
    pub min_models: usize,
    /// Enable parallel execution
    pub parallel_execution: bool,
    /// Timeout for individual models
    pub model_timeout_ms: Option<u64>,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            strategy: EnsembleStrategy::Average,
            model_weights: HashMap::new(),
            min_models: 1,
            parallel_execution: true,
            model_timeout_ms: None,
        }
    }
}

impl EnsembleConfig {
    /// Create configuration for voting ensemble.
    pub fn voting() -> Self {
        Self {
            strategy: EnsembleStrategy::MajorityVote,
            min_models: 3,
            ..Default::default()
        }
    }

    /// Create configuration for weighted averaging.
    pub fn weighted_average(weights: HashMap<String, f64>) -> Self {
        Self {
            strategy: EnsembleStrategy::WeightedAverage,
            model_weights: weights,
            ..Default::default()
        }
    }

    /// Create configuration for stacking ensemble.
    pub fn stacking() -> Self {
        Self {
            strategy: EnsembleStrategy::Stacking,
            parallel_execution: true,
            ..Default::default()
        }
    }
}

/// Model routing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route to model with highest priority
    Priority,
    /// Route to model with lowest latency
    LowestLatency,
    /// Route to model with best accuracy (requires profiling)
    BestAccuracy,
    /// Round-robin across models
    RoundRobin,
    /// Random selection
    Random,
    /// Cascade (try fast model first, fallback to accurate)
    Cascade,
    /// Route based on input features
    ContentBased,
}

/// Model cascade configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeConfig {
    /// Ordered list of model IDs (fast to accurate)
    pub model_sequence: Vec<String>,
    /// Confidence thresholds for each model
    pub confidence_thresholds: Vec<f64>,
    /// Enable early exit
    pub enable_early_exit: bool,
    /// Maximum models to try
    pub max_models: usize,
}

impl CascadeConfig {
    /// Create a two-tier cascade (fast + accurate).
    pub fn two_tier(fast_model: String, accurate_model: String, threshold: f64) -> Self {
        Self {
            model_sequence: vec![fast_model, accurate_model],
            confidence_thresholds: vec![threshold],
            enable_early_exit: true,
            max_models: 2,
        }
    }

    /// Create a three-tier cascade.
    pub fn three_tier(
        fast: String,
        medium: String,
        accurate: String,
        thresholds: (f64, f64),
    ) -> Self {
        Self {
            model_sequence: vec![fast, medium, accurate],
            confidence_thresholds: vec![thresholds.0, thresholds.1],
            enable_early_exit: true,
            max_models: 3,
        }
    }
}

/// Multi-model coordinator.
pub struct MultiModelCoordinator {
    models: HashMap<String, EinsumGraph>,
    metadata: HashMap<String, ModelMetadata>,
    ensemble_config: Option<EnsembleConfig>,
    routing_strategy: RoutingStrategy,
    stats: CoordinationStats,
}

impl MultiModelCoordinator {
    /// Create a new multi-model coordinator.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            metadata: HashMap::new(),
            ensemble_config: None,
            routing_strategy: RoutingStrategy::Priority,
            stats: CoordinationStats::default(),
        }
    }

    /// Register a model.
    pub fn register_model(
        &mut self,
        graph: EinsumGraph,
        metadata: ModelMetadata,
    ) -> Result<(), MultiModelError> {
        let id = metadata.id.clone();
        self.models.insert(id.clone(), graph);
        self.metadata.insert(id, metadata);
        self.stats.total_models += 1;
        Ok(())
    }

    /// Unregister a model.
    pub fn unregister_model(&mut self, model_id: &str) -> Result<(), MultiModelError> {
        self.models
            .remove(model_id)
            .ok_or_else(|| MultiModelError::ModelNotFound(model_id.to_string()))?;
        self.metadata.remove(model_id);
        self.stats.total_models = self.models.len();
        Ok(())
    }

    /// Set ensemble configuration.
    pub fn set_ensemble_config(&mut self, config: EnsembleConfig) {
        self.ensemble_config = Some(config);
    }

    /// Set routing strategy.
    pub fn set_routing_strategy(&mut self, strategy: RoutingStrategy) {
        self.routing_strategy = strategy;
    }

    /// Select model based on routing strategy.
    pub fn select_model(
        &mut self,
        _input_features: Option<&[f64]>,
    ) -> Result<String, MultiModelError> {
        if self.models.is_empty() {
            return Err(MultiModelError::RoutingFailed(
                "No models registered".to_string(),
            ));
        }

        let selected = match self.routing_strategy {
            RoutingStrategy::Priority => self.select_by_priority(),
            RoutingStrategy::LowestLatency => self.select_by_latency(),
            RoutingStrategy::BestAccuracy => self.select_by_accuracy(),
            RoutingStrategy::RoundRobin => self.select_round_robin(),
            RoutingStrategy::Random => self.select_random(),
            RoutingStrategy::Cascade => self.select_cascade(),
            RoutingStrategy::ContentBased => self.select_content_based(_input_features),
        };

        if let Ok(ref id) = selected {
            self.stats.total_routings += 1;
            *self.stats.model_usage.entry(id.clone()).or_insert(0) += 1;
        }

        selected
    }

    fn select_by_priority(&self) -> Result<String, MultiModelError> {
        self.metadata
            .iter()
            .max_by_key(|(_, meta)| meta.priority)
            .map(|(id, _)| id.clone())
            .ok_or_else(|| MultiModelError::RoutingFailed("No models available".to_string()))
    }

    fn select_by_latency(&self) -> Result<String, MultiModelError> {
        self.metadata
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.resource_requirements
                    .estimated_latency_ms
                    .partial_cmp(&b.resource_requirements.estimated_latency_ms)
                    .unwrap()
            })
            .map(|(id, _)| id.clone())
            .ok_or_else(|| MultiModelError::RoutingFailed("No models available".to_string()))
    }

    fn select_by_accuracy(&self) -> Result<String, MultiModelError> {
        // Would use historical accuracy data
        // For now, just select highest priority
        self.select_by_priority()
    }

    fn select_round_robin(&mut self) -> Result<String, MultiModelError> {
        let model_ids: Vec<_> = self.models.keys().cloned().collect();
        if model_ids.is_empty() {
            return Err(MultiModelError::RoutingFailed(
                "No models available".to_string(),
            ));
        }

        let idx = self.stats.total_routings % model_ids.len();
        Ok(model_ids[idx].clone())
    }

    fn select_random(&self) -> Result<String, MultiModelError> {
        // In real implementation, use proper RNG
        let model_ids: Vec<_> = self.models.keys().cloned().collect();
        if model_ids.is_empty() {
            return Err(MultiModelError::RoutingFailed(
                "No models available".to_string(),
            ));
        }

        Ok(model_ids[0].clone())
    }

    fn select_cascade(&self) -> Result<String, MultiModelError> {
        // Start with fastest model
        self.select_by_latency()
    }

    fn select_content_based(&self, _features: Option<&[f64]>) -> Result<String, MultiModelError> {
        // Would analyze input features to select best model
        // For now, fallback to priority
        self.select_by_priority()
    }

    /// Get model by ID.
    pub fn get_model(&self, model_id: &str) -> Option<&EinsumGraph> {
        self.models.get(model_id)
    }

    /// Get model metadata.
    pub fn get_metadata(&self, model_id: &str) -> Option<&ModelMetadata> {
        self.metadata.get(model_id)
    }

    /// Get all registered model IDs.
    pub fn model_ids(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Get statistics.
    pub fn stats(&self) -> &CoordinationStats {
        &self.stats
    }

    /// Check if ensemble is configured.
    pub fn has_ensemble(&self) -> bool {
        self.ensemble_config.is_some()
    }

    /// Get ensemble configuration.
    pub fn ensemble_config(&self) -> Option<&EnsembleConfig> {
        self.ensemble_config.as_ref()
    }

    /// Estimate total resource requirements.
    pub fn total_resource_requirements(&self) -> ResourceRequirements {
        let mut total = ResourceRequirements {
            memory_bytes: 0,
            gpu_memory_bytes: Some(0),
            estimated_flops: 0.0,
            estimated_latency_ms: 0.0,
        };

        for metadata in self.metadata.values() {
            let req = &metadata.resource_requirements;
            total.memory_bytes += req.memory_bytes;
            if let (Some(total_gpu), Some(req_gpu)) = (total.gpu_memory_bytes, req.gpu_memory_bytes)
            {
                total.gpu_memory_bytes = Some(total_gpu + req_gpu);
            }
            total.estimated_flops += req.estimated_flops;
            // For latency, use max if parallel, sum if sequential
            total.estimated_latency_ms = total.estimated_latency_ms.max(req.estimated_latency_ms);
        }

        total
    }
}

impl Default for MultiModelCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Coordination statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoordinationStats {
    /// Total models registered
    pub total_models: usize,
    /// Total routing decisions
    pub total_routings: usize,
    /// Total ensemble executions
    pub total_ensemble_executions: usize,
    /// Model usage counts
    pub model_usage: HashMap<String, usize>,
}

impl CoordinationStats {
    /// Get most used model.
    pub fn most_used_model(&self) -> Option<(String, usize)> {
        self.model_usage
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(id, &count)| (id.clone(), count))
    }

    /// Get model usage distribution.
    pub fn usage_distribution(&self) -> HashMap<String, f64> {
        let total = self.model_usage.values().sum::<usize>() as f64;
        if total == 0.0 {
            return HashMap::new();
        }

        self.model_usage
            .iter()
            .map(|(id, &count)| (id.clone(), count as f64 / total))
            .collect()
    }
}

/// Trait for multi-model ensemble execution.
pub trait TlEnsembleExecutor {
    /// Output type
    type Output;
    /// Error type
    type Error;

    /// Execute ensemble with given strategy.
    fn execute_ensemble(
        &self,
        models: &[&EinsumGraph],
        inputs: &[Self::Output],
        strategy: EnsembleStrategy,
    ) -> Result<Self::Output, Self::Error>;

    /// Aggregate predictions from multiple models.
    fn aggregate_predictions(
        &self,
        predictions: &[Self::Output],
        strategy: EnsembleStrategy,
    ) -> Result<Self::Output, Self::Error>;
}

/// Trait for model routing.
pub trait TlModelRouter {
    /// Select appropriate model based on input.
    fn route_to_model(&self, input: &[f64]) -> Result<String, MultiModelError>;

    /// Get routing confidence score.
    fn routing_confidence(&self, input: &[f64], model_id: &str) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{EinsumNode, OpType};

    fn create_test_graph(_id: &str) -> EinsumGraph {
        let mut graph = EinsumGraph::new();
        graph.nodes.push(EinsumNode {
            op: OpType::Einsum {
                spec: "ij->ij".to_string(),
            },
            inputs: vec![],
            outputs: vec![0],
            metadata: Default::default(),
        });
        graph
    }

    fn create_test_metadata(id: &str, priority: u32) -> ModelMetadata {
        ModelMetadata {
            id: id.to_string(),
            name: format!("Model {}", id),
            version: "1.0".to_string(),
            input_shapes: vec![vec![10, 10]],
            output_shapes: vec![vec![10, 10]],
            weight: 1.0,
            priority,
            resource_requirements: ResourceRequirements {
                memory_bytes: 1024 * 1024,
                gpu_memory_bytes: Some(512 * 1024),
                estimated_flops: 1e9,
                estimated_latency_ms: 10.0,
            },
        }
    }

    #[test]
    fn test_ensemble_strategy() {
        let config = EnsembleConfig::voting();
        assert_eq!(config.strategy, EnsembleStrategy::MajorityVote);

        let mut weights = HashMap::new();
        weights.insert("model1".to_string(), 0.6);
        weights.insert("model2".to_string(), 0.4);
        let config = EnsembleConfig::weighted_average(weights);
        assert_eq!(config.strategy, EnsembleStrategy::WeightedAverage);
    }

    #[test]
    fn test_cascade_config() {
        let config = CascadeConfig::two_tier("fast".to_string(), "accurate".to_string(), 0.9);
        assert_eq!(config.model_sequence.len(), 2);
        assert_eq!(config.confidence_thresholds[0], 0.9);

        let config = CascadeConfig::three_tier(
            "fast".to_string(),
            "medium".to_string(),
            "accurate".to_string(),
            (0.8, 0.95),
        );
        assert_eq!(config.model_sequence.len(), 3);
    }

    #[test]
    fn test_coordinator_creation() {
        let coordinator = MultiModelCoordinator::new();
        assert_eq!(coordinator.models.len(), 0);
        assert_eq!(coordinator.stats.total_models, 0);
    }

    #[test]
    fn test_model_registration() {
        let mut coordinator = MultiModelCoordinator::new();

        let graph = create_test_graph("model1");
        let metadata = create_test_metadata("model1", 1);

        assert!(coordinator.register_model(graph, metadata).is_ok());
        assert_eq!(coordinator.stats.total_models, 1);
        assert!(coordinator.get_model("model1").is_some());
    }

    #[test]
    fn test_model_unregistration() {
        let mut coordinator = MultiModelCoordinator::new();

        let graph = create_test_graph("model1");
        let metadata = create_test_metadata("model1", 1);
        coordinator.register_model(graph, metadata).unwrap();

        assert!(coordinator.unregister_model("model1").is_ok());
        assert_eq!(coordinator.stats.total_models, 0);
        assert!(coordinator.get_model("model1").is_none());
    }

    #[test]
    fn test_routing_by_priority() {
        let mut coordinator = MultiModelCoordinator::new();

        coordinator
            .register_model(
                create_test_graph("model1"),
                create_test_metadata("model1", 1),
            )
            .unwrap();
        coordinator
            .register_model(
                create_test_graph("model2"),
                create_test_metadata("model2", 5),
            )
            .unwrap();

        coordinator.set_routing_strategy(RoutingStrategy::Priority);
        let selected = coordinator.select_model(None).unwrap();
        assert_eq!(selected, "model2"); // Higher priority
    }

    #[test]
    fn test_routing_by_latency() {
        let mut coordinator = MultiModelCoordinator::new();

        let mut meta1 = create_test_metadata("model1", 1);
        meta1.resource_requirements.estimated_latency_ms = 20.0;
        let mut meta2 = create_test_metadata("model2", 1);
        meta2.resource_requirements.estimated_latency_ms = 5.0;

        coordinator
            .register_model(create_test_graph("model1"), meta1)
            .unwrap();
        coordinator
            .register_model(create_test_graph("model2"), meta2)
            .unwrap();

        coordinator.set_routing_strategy(RoutingStrategy::LowestLatency);
        let selected = coordinator.select_model(None).unwrap();
        assert_eq!(selected, "model2"); // Lower latency
    }

    #[test]
    fn test_ensemble_configuration() {
        let mut coordinator = MultiModelCoordinator::new();
        assert!(!coordinator.has_ensemble());

        coordinator.set_ensemble_config(EnsembleConfig::voting());
        assert!(coordinator.has_ensemble());
        assert_eq!(
            coordinator.ensemble_config().unwrap().strategy,
            EnsembleStrategy::MajorityVote
        );
    }

    #[test]
    fn test_total_resource_requirements() {
        let mut coordinator = MultiModelCoordinator::new();

        coordinator
            .register_model(
                create_test_graph("model1"),
                create_test_metadata("model1", 1),
            )
            .unwrap();
        coordinator
            .register_model(
                create_test_graph("model2"),
                create_test_metadata("model2", 1),
            )
            .unwrap();

        let total = coordinator.total_resource_requirements();
        assert_eq!(total.memory_bytes, 2 * 1024 * 1024);
        assert_eq!(total.gpu_memory_bytes, Some(2 * 512 * 1024));
    }

    #[test]
    fn test_coordination_stats() {
        let mut stats = CoordinationStats::default();
        stats.model_usage.insert("model1".to_string(), 10);
        stats.model_usage.insert("model2".to_string(), 5);

        let (id, count) = stats.most_used_model().unwrap();
        assert_eq!(id, "model1");
        assert_eq!(count, 10);

        let dist = stats.usage_distribution();
        assert_eq!(dist.get("model1").unwrap(), &(10.0 / 15.0));
    }

    #[test]
    fn test_round_robin_routing() {
        let mut coordinator = MultiModelCoordinator::new();

        coordinator
            .register_model(
                create_test_graph("model1"),
                create_test_metadata("model1", 1),
            )
            .unwrap();
        coordinator
            .register_model(
                create_test_graph("model2"),
                create_test_metadata("model2", 1),
            )
            .unwrap();

        coordinator.set_routing_strategy(RoutingStrategy::RoundRobin);

        let id1 = coordinator.select_model(None).unwrap();
        let id2 = coordinator.select_model(None).unwrap();

        // Should alternate (though order may vary)
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_model_ids() {
        let mut coordinator = MultiModelCoordinator::new();

        coordinator
            .register_model(
                create_test_graph("model1"),
                create_test_metadata("model1", 1),
            )
            .unwrap();
        coordinator
            .register_model(
                create_test_graph("model2"),
                create_test_metadata("model2", 1),
            )
            .unwrap();

        let ids = coordinator.model_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"model1".to_string()));
        assert!(ids.contains(&"model2".to_string()));
    }
}
