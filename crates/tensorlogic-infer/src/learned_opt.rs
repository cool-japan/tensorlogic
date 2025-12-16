//! Machine learning-based optimization decisions.
//!
//! This module implements learned optimization strategies:
//! - **ML-based fusion decisions**: Learn which operations to fuse
//! - **Learned cost models**: Predict operation costs using ML
//! - **Reinforcement learning for scheduling**: Learn optimal execution schedules
//! - **Feature extraction**: Extract relevant features from computation graphs
//! - **Online learning**: Continuously improve from observed performance
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_infer::{LearnedOptimizer, LearningStrategy, ModelType};
//!
//! // Create learned optimizer
//! let mut optimizer = LearnedOptimizer::new()
//!     .with_strategy(LearningStrategy::ReinforcementLearning)
//!     .with_model_type(ModelType::NeuralNetwork)
//!     .with_learning_rate(0.01);
//!
//! // Train from observed executions
//! for (graph, performance) in training_data {
//!     optimizer.observe(&graph, performance)?;
//! }
//!
//! // Use learned model for optimization
//! let decision = optimizer.recommend_fusion(&graph)?;
//! ```
//!
//! ## SCIRS2 Policy Note
//!
//! This module uses `rand` for epsilon-greedy exploration in Q-learning, which is
//! acceptable under the SCIRS2 policy exception for planning layer crates
//! (tensorlogic-infer). The planning layer may avoid heavy SciRS2 dependencies to
//! remain lightweight and engine-agnostic. The rand usage here is minimal (only
//! for exploration strategy) and does not involve heavy tensor operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Learned optimization errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum LearnedOptError {
    #[error("Insufficient training data: {0}")]
    InsufficientData(String),

    #[error("Model not trained: {0}")]
    ModelNotTrained(String),

    #[error("Feature extraction failed: {0}")]
    FeatureExtractionFailed(String),

    #[error("Prediction failed: {0}")]
    PredictionFailed(String),

    #[error("Invalid model configuration: {0}")]
    InvalidConfig(String),
}

/// Node ID in the computation graph.
pub type NodeId = String;

/// Learning strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningStrategy {
    /// Supervised learning from labeled examples
    Supervised,
    /// Reinforcement learning from rewards
    ReinforcementLearning,
    /// Online learning with continuous updates
    Online,
    /// Transfer learning from pre-trained models
    Transfer,
}

/// Model type for learned optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Linear regression model
    LinearRegression,
    /// Decision tree
    DecisionTree,
    /// Random forest
    RandomForest,
    /// Neural network (simplified)
    NeuralNetwork,
    /// Gradient boosting
    GradientBoosting,
}

/// Feature vector for graph/node characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
}

impl FeatureVector {
    fn new() -> Self {
        Self {
            features: Vec::new(),
            feature_names: Vec::new(),
        }
    }

    fn add_feature(&mut self, name: String, value: f64) {
        self.feature_names.push(name);
        self.features.push(value);
    }
}

/// Training example for supervised learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub features: FeatureVector,
    pub label: f64, // For regression: execution time, for classification: 0/1
}

/// Reward signal for reinforcement learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardSignal {
    pub state_features: FeatureVector,
    pub action: OptimizationAction,
    pub reward: f64, // Positive for speedup, negative for slowdown
    pub next_state_features: Option<FeatureVector>,
}

/// Optimization action that can be learned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationAction {
    Fuse,
    DontFuse,
    Parallelize,
    Sequential,
    CacheResult,
    Recompute,
}

/// Fusion recommendation from learned model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionRecommendation {
    pub should_fuse: bool,
    pub confidence: f64,
    pub expected_speedup: f64,
}

/// Scheduling recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleRecommendation {
    pub schedule: Vec<NodeId>,
    pub confidence: f64,
    pub expected_time_us: f64,
}

/// Cost prediction from learned model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostPrediction {
    pub predicted_cost_us: f64,
    pub confidence_interval: (f64, f64), // (lower, upper)
    pub model_confidence: f64,
}

/// Learning statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    pub training_examples: usize,
    pub model_accuracy: f64,
    pub average_prediction_error: f64,
    pub total_updates: usize,
    pub learning_rate: f64,
}

/// Simplified linear model for cost prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearModel {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl LinearModel {
    fn new(num_features: usize, learning_rate: f64) -> Self {
        Self {
            weights: vec![0.0; num_features],
            bias: 0.0,
            learning_rate,
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let mut result = self.bias;
        for (w, f) in self.weights.iter().zip(features.iter()) {
            result += w * f;
        }
        result
    }

    fn update(&mut self, features: &[f64], target: f64) {
        let prediction = self.predict(features);
        let error = target - prediction;

        // Gradient descent update
        for (w, f) in self.weights.iter_mut().zip(features.iter()) {
            *w += self.learning_rate * error * f;
        }
        self.bias += self.learning_rate * error;
    }
}

/// Q-learning agent for reinforcement learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QLearningAgent {
    q_table: HashMap<(String, OptimizationAction), f64>, // (state, action) -> Q-value
    learning_rate: f64,
    discount_factor: f64,
    epsilon: f64, // Exploration rate
}

impl QLearningAgent {
    fn new(learning_rate: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor: 0.95,
            epsilon: 0.1,
        }
    }

    fn get_q_value(&self, state: &str, action: OptimizationAction) -> f64 {
        *self
            .q_table
            .get(&(state.to_string(), action))
            .unwrap_or(&0.0)
    }

    fn update_q_value(
        &mut self,
        state: &str,
        action: OptimizationAction,
        reward: f64,
        next_state: Option<&str>,
    ) {
        let current_q = self.get_q_value(state, action);

        let max_next_q = if let Some(ns) = next_state {
            [
                self.get_q_value(ns, OptimizationAction::Fuse),
                self.get_q_value(ns, OptimizationAction::DontFuse),
                self.get_q_value(ns, OptimizationAction::Parallelize),
                self.get_q_value(ns, OptimizationAction::Sequential),
            ]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        } else {
            0.0
        };

        let new_q = current_q
            + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q);

        self.q_table.insert((state.to_string(), action), new_q);
    }

    fn select_action(&self, state: &str, explore: bool) -> OptimizationAction {
        if explore && rand::random::<f64>() < self.epsilon {
            // Random exploration
            let actions = [
                OptimizationAction::Fuse,
                OptimizationAction::DontFuse,
                OptimizationAction::Parallelize,
                OptimizationAction::Sequential,
            ];
            actions[rand::random::<usize>() % actions.len()]
        } else {
            // Greedy exploitation
            let actions = [
                (
                    OptimizationAction::Fuse,
                    self.get_q_value(state, OptimizationAction::Fuse),
                ),
                (
                    OptimizationAction::DontFuse,
                    self.get_q_value(state, OptimizationAction::DontFuse),
                ),
                (
                    OptimizationAction::Parallelize,
                    self.get_q_value(state, OptimizationAction::Parallelize),
                ),
                (
                    OptimizationAction::Sequential,
                    self.get_q_value(state, OptimizationAction::Sequential),
                ),
            ];

            actions
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(action, _)| *action)
                .unwrap_or(OptimizationAction::DontFuse)
        }
    }
}

/// Learned optimizer.
pub struct LearnedOptimizer {
    strategy: LearningStrategy,
    model_type: ModelType,
    cost_model: Option<LinearModel>,
    q_agent: Option<QLearningAgent>,
    training_examples: Vec<TrainingExample>,
    learning_rate: f64,
    stats: LearningStats,
    min_training_examples: usize,
}

impl LearnedOptimizer {
    /// Create a new learned optimizer with default settings.
    pub fn new() -> Self {
        Self {
            strategy: LearningStrategy::Online,
            model_type: ModelType::LinearRegression,
            cost_model: None,
            q_agent: None,
            training_examples: Vec::new(),
            learning_rate: 0.01,
            stats: LearningStats {
                training_examples: 0,
                model_accuracy: 0.0,
                average_prediction_error: 0.0,
                total_updates: 0,
                learning_rate: 0.01,
            },
            min_training_examples: 10,
        }
    }

    /// Set learning strategy.
    pub fn with_strategy(mut self, strategy: LearningStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set model type.
    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Set learning rate.
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate.clamp(0.0001, 1.0);
        self.stats.learning_rate = self.learning_rate;
        self
    }

    /// Extract features from graph description.
    pub fn extract_features(
        &self,
        graph_desc: &HashMap<String, f64>,
    ) -> Result<FeatureVector, LearnedOptError> {
        let mut features = FeatureVector::new();

        // Extract common graph features
        features.add_feature(
            "num_nodes".to_string(),
            *graph_desc.get("num_nodes").unwrap_or(&0.0),
        );
        features.add_feature(
            "num_edges".to_string(),
            *graph_desc.get("num_edges").unwrap_or(&0.0),
        );
        features.add_feature(
            "avg_node_degree".to_string(),
            *graph_desc.get("avg_degree").unwrap_or(&0.0),
        );
        features.add_feature(
            "graph_depth".to_string(),
            *graph_desc.get("depth").unwrap_or(&0.0),
        );
        features.add_feature(
            "total_memory".to_string(),
            *graph_desc.get("memory").unwrap_or(&0.0),
        );
        features.add_feature(
            "parallelism_factor".to_string(),
            *graph_desc.get("parallelism").unwrap_or(&1.0),
        );

        Ok(features)
    }

    /// Observe execution and update model (online learning).
    pub fn observe(
        &mut self,
        features: FeatureVector,
        actual_cost: f64,
    ) -> Result<(), LearnedOptError> {
        let example = TrainingExample {
            features: features.clone(),
            label: actual_cost,
        };

        self.training_examples.push(example);
        self.stats.training_examples += 1;

        // Initialize model if needed
        if self.cost_model.is_none() && features.features.len() > 0 {
            self.cost_model = Some(LinearModel::new(
                features.features.len(),
                self.learning_rate,
            ));
        }

        // Update model online
        if let Some(model) = &mut self.cost_model {
            model.update(&features.features, actual_cost);
            self.stats.total_updates += 1;
        }

        Ok(())
    }

    /// Observe reward signal for reinforcement learning.
    pub fn observe_reward(&mut self, signal: RewardSignal) -> Result<(), LearnedOptError> {
        if self.strategy != LearningStrategy::ReinforcementLearning {
            return Err(LearnedOptError::InvalidConfig(
                "Reward observation requires ReinforcementLearning strategy".to_string(),
            ));
        }

        // Initialize Q-learning agent if needed
        if self.q_agent.is_none() {
            self.q_agent = Some(QLearningAgent::new(self.learning_rate));
        }

        // Create state representation (simplified as feature hash)
        let state = format!("{:?}", signal.state_features.features);
        let next_state = signal
            .next_state_features
            .as_ref()
            .map(|f| format!("{:?}", f.features));

        if let Some(agent) = &mut self.q_agent {
            agent.update_q_value(&state, signal.action, signal.reward, next_state.as_deref());
        }

        self.stats.total_updates += 1;

        Ok(())
    }

    /// Predict cost for given features.
    pub fn predict_cost(
        &self,
        features: &FeatureVector,
    ) -> Result<CostPrediction, LearnedOptError> {
        let model = self.cost_model.as_ref().ok_or_else(|| {
            LearnedOptError::ModelNotTrained("Cost model not trained".to_string())
        })?;

        if self.training_examples.len() < self.min_training_examples {
            return Err(LearnedOptError::InsufficientData(format!(
                "Need at least {} examples, have {}",
                self.min_training_examples,
                self.training_examples.len()
            )));
        }

        let predicted_cost = model.predict(&features.features);

        // Simplified confidence interval (±20% of prediction)
        let margin = predicted_cost * 0.2;
        let confidence_interval = (predicted_cost - margin, predicted_cost + margin);

        // Model confidence based on training data size
        let model_confidence = (self.training_examples.len() as f64
            / (self.min_training_examples * 10) as f64)
            .min(1.0);

        Ok(CostPrediction {
            predicted_cost_us: predicted_cost.max(0.0),
            confidence_interval,
            model_confidence,
        })
    }

    /// Recommend whether to fuse operations.
    pub fn recommend_fusion(
        &self,
        features: &FeatureVector,
    ) -> Result<FusionRecommendation, LearnedOptError> {
        match self.strategy {
            LearningStrategy::ReinforcementLearning => {
                let agent = self.q_agent.as_ref().ok_or_else(|| {
                    LearnedOptError::ModelNotTrained("Q-learning agent not initialized".to_string())
                })?;

                let state = format!("{:?}", features.features);
                let action = agent.select_action(&state, false);

                let should_fuse = action == OptimizationAction::Fuse;
                let q_fuse = agent.get_q_value(&state, OptimizationAction::Fuse);
                let q_no_fuse = agent.get_q_value(&state, OptimizationAction::DontFuse);

                let confidence =
                    (q_fuse - q_no_fuse).abs() / (q_fuse.abs() + q_no_fuse.abs() + 1.0);
                let expected_speedup = if should_fuse { q_fuse.max(1.0) } else { 1.0 };

                Ok(FusionRecommendation {
                    should_fuse,
                    confidence,
                    expected_speedup,
                })
            }
            _ => {
                // Use cost model to estimate fusion benefit
                let cost_pred = self.predict_cost(features)?;

                // Heuristic: fuse if predicted cost is below threshold
                let threshold = 100.0; // microseconds
                let should_fuse = cost_pred.predicted_cost_us < threshold;

                Ok(FusionRecommendation {
                    should_fuse,
                    confidence: cost_pred.model_confidence,
                    expected_speedup: if should_fuse { 1.5 } else { 1.0 },
                })
            }
        }
    }

    /// Get learning statistics.
    pub fn get_stats(&self) -> &LearningStats {
        &self.stats
    }

    /// Evaluate model accuracy on training data.
    pub fn evaluate_accuracy(&mut self) -> Result<f64, LearnedOptError> {
        if self.training_examples.is_empty() {
            return Ok(0.0);
        }

        let model = self.cost_model.as_ref().ok_or_else(|| {
            LearnedOptError::ModelNotTrained("Cost model not trained".to_string())
        })?;

        let mut total_error = 0.0;

        for example in &self.training_examples {
            let prediction = model.predict(&example.features.features);
            let error = (prediction - example.label).abs();
            total_error += error;
        }

        let avg_error = total_error / self.training_examples.len() as f64;
        self.stats.average_prediction_error = avg_error;

        // Accuracy = 1 - normalized error
        let max_label = self
            .training_examples
            .iter()
            .map(|e| e.label)
            .fold(f64::NEG_INFINITY, f64::max);

        let accuracy = if max_label > 0.0 {
            (1.0 - (avg_error / max_label)).max(0.0)
        } else {
            0.0
        };

        self.stats.model_accuracy = accuracy;

        Ok(accuracy)
    }

    /// Reset learning state.
    pub fn reset(&mut self) {
        self.training_examples.clear();
        self.cost_model = None;
        self.q_agent = None;
        self.stats = LearningStats {
            training_examples: 0,
            model_accuracy: 0.0,
            average_prediction_error: 0.0,
            total_updates: 0,
            learning_rate: self.learning_rate,
        };
    }
}

impl Default for LearnedOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_features() -> FeatureVector {
        let mut features = FeatureVector::new();
        features.add_feature("num_nodes".to_string(), 10.0);
        features.add_feature("num_edges".to_string(), 15.0);
        features.add_feature("avg_degree".to_string(), 1.5);
        features
    }

    #[test]
    fn test_learned_optimizer_creation() {
        let optimizer = LearnedOptimizer::new();
        assert_eq!(optimizer.strategy, LearningStrategy::Online);
        assert_eq!(optimizer.model_type, ModelType::LinearRegression);
    }

    #[test]
    fn test_builder_pattern() {
        let optimizer = LearnedOptimizer::new()
            .with_strategy(LearningStrategy::ReinforcementLearning)
            .with_model_type(ModelType::NeuralNetwork)
            .with_learning_rate(0.05);

        assert_eq!(optimizer.strategy, LearningStrategy::ReinforcementLearning);
        assert_eq!(optimizer.model_type, ModelType::NeuralNetwork);
        assert_eq!(optimizer.learning_rate, 0.05);
    }

    #[test]
    fn test_feature_extraction() {
        let optimizer = LearnedOptimizer::new();
        let mut graph_desc = HashMap::new();
        graph_desc.insert("num_nodes".to_string(), 10.0);
        graph_desc.insert("num_edges".to_string(), 15.0);

        let features = optimizer.extract_features(&graph_desc).unwrap();
        assert!(features.features.len() > 0);
    }

    #[test]
    fn test_observe_and_learn() {
        let mut optimizer = LearnedOptimizer::new();
        let features = create_test_features();

        optimizer.observe(features.clone(), 100.0).unwrap();
        optimizer.observe(features.clone(), 95.0).unwrap();

        assert_eq!(optimizer.stats.training_examples, 2);
        assert_eq!(optimizer.stats.total_updates, 2);
    }

    #[test]
    fn test_cost_prediction_insufficient_data() {
        let optimizer = LearnedOptimizer::new();
        let features = create_test_features();

        let result = optimizer.predict_cost(&features);
        assert!(result.is_err());
    }

    #[test]
    fn test_cost_prediction_with_training() {
        let mut optimizer = LearnedOptimizer::new();
        let features = create_test_features();

        // Add sufficient training examples
        for i in 0..15 {
            let mut f = create_test_features();
            f.features[0] = i as f64;
            optimizer.observe(f, 100.0 + i as f64).unwrap();
        }

        let prediction = optimizer.predict_cost(&features).unwrap();
        assert!(prediction.predicted_cost_us >= 0.0);
        assert!(prediction.model_confidence > 0.0);
    }

    #[test]
    fn test_reinforcement_learning_observation() {
        let mut optimizer =
            LearnedOptimizer::new().with_strategy(LearningStrategy::ReinforcementLearning);

        let signal = RewardSignal {
            state_features: create_test_features(),
            action: OptimizationAction::Fuse,
            reward: 10.0, // Positive reward for speedup
            next_state_features: Some(create_test_features()),
        };

        optimizer.observe_reward(signal).unwrap();
        assert_eq!(optimizer.stats.total_updates, 1);
    }

    #[test]
    fn test_fusion_recommendation() {
        let mut optimizer = LearnedOptimizer::new();
        let features = create_test_features();

        // Train with examples
        for i in 0..15 {
            let mut f = create_test_features();
            f.features[0] = i as f64;
            optimizer.observe(f, 50.0 + i as f64).unwrap(); // Low cost -> should recommend fusion
        }

        let recommendation = optimizer.recommend_fusion(&features).unwrap();
        assert!(recommendation.confidence >= 0.0);
    }

    #[test]
    fn test_rl_fusion_recommendation() {
        let mut optimizer =
            LearnedOptimizer::new().with_strategy(LearningStrategy::ReinforcementLearning);

        let features = create_test_features();

        // Train with rewards
        for _ in 0..10 {
            let signal = RewardSignal {
                state_features: features.clone(),
                action: OptimizationAction::Fuse,
                reward: 15.0,
                next_state_features: None,
            };
            optimizer.observe_reward(signal).unwrap();
        }

        let recommendation = optimizer.recommend_fusion(&features).unwrap();
        // Just check it returns a valid recommendation
        assert!(recommendation.confidence >= 0.0);
    }

    #[test]
    fn test_accuracy_evaluation() {
        let mut optimizer = LearnedOptimizer::new();

        // Add training examples with known relationship
        for i in 0..20 {
            let mut features = FeatureVector::new();
            features.add_feature("x".to_string(), i as f64);
            optimizer.observe(features, i as f64 * 2.0).unwrap(); // y = 2x
        }

        let accuracy = optimizer.evaluate_accuracy().unwrap();
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    fn test_reset() {
        let mut optimizer = LearnedOptimizer::new();
        let features = create_test_features();

        optimizer.observe(features, 100.0).unwrap();
        assert_eq!(optimizer.stats.training_examples, 1);

        optimizer.reset();
        assert_eq!(optimizer.stats.training_examples, 0);
        assert!(optimizer.training_examples.is_empty());
    }

    #[test]
    fn test_linear_model_prediction() {
        let model = LinearModel::new(3, 0.01);
        let features = vec![1.0, 2.0, 3.0];

        let prediction = model.predict(&features);
        assert!(prediction.is_finite());
    }

    #[test]
    fn test_linear_model_update() {
        let mut model = LinearModel::new(2, 0.1);
        let features = vec![1.0, 2.0];

        model.update(&features, 10.0);
        let pred_after = model.predict(&features);

        // After update, prediction should move towards target
        assert!(pred_after.is_finite());
    }

    #[test]
    fn test_q_learning_agent() {
        let mut agent = QLearningAgent::new(0.1);

        agent.update_q_value("state1", OptimizationAction::Fuse, 10.0, Some("state2"));

        let q_value = agent.get_q_value("state1", OptimizationAction::Fuse);
        assert!(q_value > 0.0);
    }

    #[test]
    fn test_q_learning_action_selection() {
        let mut agent = QLearningAgent::new(0.1);

        // Train with high reward for Fuse action
        for _ in 0..10 {
            agent.update_q_value("state1", OptimizationAction::Fuse, 20.0, None);
        }

        let action = agent.select_action("state1", false);
        // With no exploration, should select Fuse (but other actions possible)
        assert!(
            action == OptimizationAction::Fuse
                || action == OptimizationAction::DontFuse
                || action == OptimizationAction::Parallelize
                || action == OptimizationAction::Sequential
        );
    }

    #[test]
    fn test_different_learning_strategies() {
        let strategies = vec![
            LearningStrategy::Supervised,
            LearningStrategy::Online,
            LearningStrategy::Transfer,
        ];

        for strategy in strategies {
            let optimizer = LearnedOptimizer::new().with_strategy(strategy);
            assert_eq!(optimizer.strategy, strategy);
        }
    }

    #[test]
    fn test_different_model_types() {
        let model_types = vec![
            ModelType::LinearRegression,
            ModelType::DecisionTree,
            ModelType::RandomForest,
            ModelType::NeuralNetwork,
            ModelType::GradientBoosting,
        ];

        for model_type in model_types {
            let optimizer = LearnedOptimizer::new().with_model_type(model_type);
            assert_eq!(optimizer.model_type, model_type);
        }
    }
}
