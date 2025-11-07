//! Hyperparameter optimization utilities.
//!
//! This module provides various hyperparameter search strategies:
//! - Grid search (exhaustive search over parameter grid)
//! - Random search (random sampling from parameter space)
//! - Parameter space definition
//! - Result tracking and comparison

use crate::{TrainError, TrainResult};
use scirs2_core::random::{Rng, SeedableRng, StdRng};
use std::collections::HashMap;

/// Hyperparameter value type.
#[derive(Debug, Clone, PartialEq)]
pub enum HyperparamValue {
    /// Floating-point value.
    Float(f64),
    /// Integer value.
    Int(i64),
    /// Boolean value.
    Bool(bool),
    /// String value.
    String(String),
}

impl HyperparamValue {
    /// Get as f64, if possible.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            HyperparamValue::Float(v) => Some(*v),
            HyperparamValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get as i64, if possible.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            HyperparamValue::Int(v) => Some(*v),
            HyperparamValue::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Get as bool, if possible.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            HyperparamValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as string, if possible.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            HyperparamValue::String(v) => Some(v),
            _ => None,
        }
    }
}

/// Hyperparameter space definition.
#[derive(Debug, Clone)]
pub enum HyperparamSpace {
    /// Discrete choices.
    Discrete(Vec<HyperparamValue>),
    /// Continuous range [min, max].
    Continuous { min: f64, max: f64 },
    /// Log-uniform distribution [min, max].
    LogUniform { min: f64, max: f64 },
    /// Integer range [min, max].
    IntRange { min: i64, max: i64 },
}

impl HyperparamSpace {
    /// Create a discrete choice space.
    pub fn discrete(values: Vec<HyperparamValue>) -> TrainResult<Self> {
        if values.is_empty() {
            return Err(TrainError::InvalidParameter(
                "Discrete space cannot be empty".to_string(),
            ));
        }
        Ok(Self::Discrete(values))
    }

    /// Create a continuous range space.
    pub fn continuous(min: f64, max: f64) -> TrainResult<Self> {
        if min >= max {
            return Err(TrainError::InvalidParameter(
                "min must be less than max".to_string(),
            ));
        }
        Ok(Self::Continuous { min, max })
    }

    /// Create a log-uniform distribution space.
    pub fn log_uniform(min: f64, max: f64) -> TrainResult<Self> {
        if min <= 0.0 || max <= 0.0 || min >= max {
            return Err(TrainError::InvalidParameter(
                "min and max must be positive and min < max".to_string(),
            ));
        }
        Ok(Self::LogUniform { min, max })
    }

    /// Create an integer range space.
    pub fn int_range(min: i64, max: i64) -> TrainResult<Self> {
        if min >= max {
            return Err(TrainError::InvalidParameter(
                "min must be less than max".to_string(),
            ));
        }
        Ok(Self::IntRange { min, max })
    }

    /// Sample a value from this space.
    pub fn sample(&self, rng: &mut StdRng) -> HyperparamValue {
        match self {
            HyperparamSpace::Discrete(values) => {
                let idx = rng.gen_range(0..values.len());
                values[idx].clone()
            }
            HyperparamSpace::Continuous { min, max } => {
                let value = min + (max - min) * rng.random::<f64>();
                HyperparamValue::Float(value)
            }
            HyperparamSpace::LogUniform { min, max } => {
                let log_min = min.ln();
                let log_max = max.ln();
                let log_value = log_min + (log_max - log_min) * rng.random::<f64>();
                HyperparamValue::Float(log_value.exp())
            }
            HyperparamSpace::IntRange { min, max } => {
                let value = rng.gen_range(*min..=*max);
                HyperparamValue::Int(value)
            }
        }
    }

    /// Get all possible values for grid search (for discrete/int spaces).
    pub fn grid_values(&self, num_samples: usize) -> Vec<HyperparamValue> {
        match self {
            HyperparamSpace::Discrete(values) => values.clone(),
            HyperparamSpace::IntRange { min, max } => {
                let range_size = (max - min + 1) as usize;
                let step = (range_size / num_samples).max(1);
                (*min..=*max)
                    .step_by(step)
                    .map(HyperparamValue::Int)
                    .collect()
            }
            HyperparamSpace::Continuous { min, max } => {
                let step = (max - min) / (num_samples as f64);
                (0..num_samples)
                    .map(|i| HyperparamValue::Float(min + step * i as f64))
                    .collect()
            }
            HyperparamSpace::LogUniform { min, max } => {
                let log_min = min.ln();
                let log_max = max.ln();
                let log_step = (log_max - log_min) / (num_samples as f64);
                (0..num_samples)
                    .map(|i| HyperparamValue::Float((log_min + log_step * i as f64).exp()))
                    .collect()
            }
        }
    }
}

/// Hyperparameter configuration (a single point in parameter space).
pub type HyperparamConfig = HashMap<String, HyperparamValue>;

/// Result of a hyperparameter evaluation.
#[derive(Debug, Clone)]
pub struct HyperparamResult {
    /// Hyperparameter configuration used.
    pub config: HyperparamConfig,
    /// Evaluation score (higher is better).
    pub score: f64,
    /// Additional metrics.
    pub metrics: HashMap<String, f64>,
}

impl HyperparamResult {
    /// Create a new result.
    pub fn new(config: HyperparamConfig, score: f64) -> Self {
        Self {
            config,
            score,
            metrics: HashMap::new(),
        }
    }

    /// Add a metric to the result.
    pub fn with_metric(mut self, name: String, value: f64) -> Self {
        self.metrics.insert(name, value);
        self
    }
}

/// Grid search strategy for hyperparameter optimization.
///
/// Exhaustively searches over a grid of hyperparameter values.
#[derive(Debug)]
pub struct GridSearch {
    /// Parameter space definition.
    param_space: HashMap<String, HyperparamSpace>,
    /// Number of grid points per continuous parameter.
    num_grid_points: usize,
    /// Results from all evaluations.
    results: Vec<HyperparamResult>,
}

impl GridSearch {
    /// Create a new grid search.
    ///
    /// # Arguments
    /// * `param_space` - Hyperparameter space definition
    /// * `num_grid_points` - Number of points for continuous parameters
    pub fn new(param_space: HashMap<String, HyperparamSpace>, num_grid_points: usize) -> Self {
        Self {
            param_space,
            num_grid_points,
            results: Vec::new(),
        }
    }

    /// Generate all parameter configurations for grid search.
    pub fn generate_configs(&self) -> Vec<HyperparamConfig> {
        if self.param_space.is_empty() {
            return vec![HashMap::new()];
        }

        let mut param_names: Vec<String> = self.param_space.keys().cloned().collect();
        param_names.sort(); // Ensure deterministic order

        let mut all_values: Vec<Vec<HyperparamValue>> = Vec::new();
        for name in &param_names {
            let space = &self.param_space[name];
            all_values.push(space.grid_values(self.num_grid_points));
        }

        // Generate Cartesian product
        let mut configs = Vec::new();
        self.generate_cartesian_product(
            &param_names,
            &all_values,
            0,
            &mut HashMap::new(),
            &mut configs,
        );

        configs
    }

    /// Recursively generate Cartesian product of parameter values.
    #[allow(clippy::only_used_in_recursion)]
    fn generate_cartesian_product(
        &self,
        param_names: &[String],
        all_values: &[Vec<HyperparamValue>],
        depth: usize,
        current_config: &mut HyperparamConfig,
        configs: &mut Vec<HyperparamConfig>,
    ) {
        if depth == param_names.len() {
            configs.push(current_config.clone());
            return;
        }

        let param_name = &param_names[depth];
        let values = &all_values[depth];

        for value in values {
            current_config.insert(param_name.clone(), value.clone());
            self.generate_cartesian_product(
                param_names,
                all_values,
                depth + 1,
                current_config,
                configs,
            );
        }

        current_config.remove(param_name);
    }

    /// Add a result from evaluating a configuration.
    pub fn add_result(&mut self, result: HyperparamResult) {
        self.results.push(result);
    }

    /// Get the best result found so far.
    pub fn best_result(&self) -> Option<&HyperparamResult> {
        self.results.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get all results sorted by score (descending).
    pub fn sorted_results(&self) -> Vec<&HyperparamResult> {
        let mut results: Vec<&HyperparamResult> = self.results.iter().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Get all results.
    pub fn results(&self) -> &[HyperparamResult] {
        &self.results
    }

    /// Get total number of configurations to evaluate.
    pub fn total_configs(&self) -> usize {
        self.generate_configs().len()
    }
}

/// Random search strategy for hyperparameter optimization.
///
/// Randomly samples from the hyperparameter space.
#[derive(Debug)]
pub struct RandomSearch {
    /// Parameter space definition.
    param_space: HashMap<String, HyperparamSpace>,
    /// Number of random samples to evaluate.
    num_samples: usize,
    /// Random number generator.
    rng: StdRng,
    /// Results from all evaluations.
    results: Vec<HyperparamResult>,
}

impl RandomSearch {
    /// Create a new random search.
    ///
    /// # Arguments
    /// * `param_space` - Hyperparameter space definition
    /// * `num_samples` - Number of random configurations to try
    /// * `seed` - Random seed for reproducibility
    pub fn new(
        param_space: HashMap<String, HyperparamSpace>,
        num_samples: usize,
        seed: u64,
    ) -> Self {
        Self {
            param_space,
            num_samples,
            rng: StdRng::seed_from_u64(seed),
            results: Vec::new(),
        }
    }

    /// Generate random parameter configurations.
    pub fn generate_configs(&mut self) -> Vec<HyperparamConfig> {
        let mut configs = Vec::with_capacity(self.num_samples);

        for _ in 0..self.num_samples {
            let mut config = HashMap::new();

            for (name, space) in &self.param_space {
                let value = space.sample(&mut self.rng);
                config.insert(name.clone(), value);
            }

            configs.push(config);
        }

        configs
    }

    /// Add a result from evaluating a configuration.
    pub fn add_result(&mut self, result: HyperparamResult) {
        self.results.push(result);
    }

    /// Get the best result found so far.
    pub fn best_result(&self) -> Option<&HyperparamResult> {
        self.results.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get all results sorted by score (descending).
    pub fn sorted_results(&self) -> Vec<&HyperparamResult> {
        let mut results: Vec<&HyperparamResult> = self.results.iter().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Get all results.
    pub fn results(&self) -> &[HyperparamResult] {
        &self.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperparam_value() {
        let float_val = HyperparamValue::Float(3.5);
        assert_eq!(float_val.as_float(), Some(3.5));
        assert_eq!(float_val.as_int(), Some(3));

        let int_val = HyperparamValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));

        let bool_val = HyperparamValue::Bool(true);
        assert_eq!(bool_val.as_bool(), Some(true));

        let string_val = HyperparamValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some("test"));
    }

    #[test]
    fn test_hyperparam_space_discrete() {
        let space = HyperparamSpace::discrete(vec![
            HyperparamValue::Float(0.1),
            HyperparamValue::Float(0.01),
        ])
        .unwrap();

        let values = space.grid_values(10);
        assert_eq!(values.len(), 2);

        let mut rng = StdRng::seed_from_u64(42);
        let sampled = space.sample(&mut rng);
        assert!(matches!(sampled, HyperparamValue::Float(_)));
    }

    #[test]
    fn test_hyperparam_space_continuous() {
        let space = HyperparamSpace::continuous(0.0, 1.0).unwrap();

        let values = space.grid_values(5);
        assert_eq!(values.len(), 5);

        let mut rng = StdRng::seed_from_u64(42);
        let sampled = space.sample(&mut rng);
        if let HyperparamValue::Float(v) = sampled {
            assert!((0.0..=1.0).contains(&v));
        } else {
            panic!("Expected Float value");
        }
    }

    #[test]
    fn test_hyperparam_space_log_uniform() {
        let space = HyperparamSpace::log_uniform(1e-4, 1e-1).unwrap();

        let values = space.grid_values(3);
        assert_eq!(values.len(), 3);

        let mut rng = StdRng::seed_from_u64(42);
        let sampled = space.sample(&mut rng);
        if let HyperparamValue::Float(v) = sampled {
            assert!((1e-4..=1e-1).contains(&v));
        } else {
            panic!("Expected Float value");
        }
    }

    #[test]
    fn test_hyperparam_space_int_range() {
        let space = HyperparamSpace::int_range(1, 10).unwrap();

        let values = space.grid_values(5);
        assert!(!values.is_empty());

        let mut rng = StdRng::seed_from_u64(42);
        let sampled = space.sample(&mut rng);
        if let HyperparamValue::Int(v) = sampled {
            assert!((1..=10).contains(&v));
        } else {
            panic!("Expected Int value");
        }
    }

    #[test]
    fn test_hyperparam_space_invalid() {
        assert!(HyperparamSpace::discrete(vec![]).is_err());
        assert!(HyperparamSpace::continuous(1.0, 0.0).is_err());
        assert!(HyperparamSpace::log_uniform(0.0, 1.0).is_err());
        assert!(HyperparamSpace::log_uniform(1.0, 0.5).is_err());
        assert!(HyperparamSpace::int_range(10, 5).is_err());
    }

    #[test]
    fn test_grid_search() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "lr".to_string(),
            HyperparamSpace::discrete(vec![
                HyperparamValue::Float(0.1),
                HyperparamValue::Float(0.01),
            ])
            .unwrap(),
        );
        param_space.insert(
            "batch_size".to_string(),
            HyperparamSpace::int_range(16, 64).unwrap(),
        );

        let grid_search = GridSearch::new(param_space, 3);

        let configs = grid_search.generate_configs();
        assert!(!configs.is_empty());

        // Should have 2 (lr values) * grid_points (batch_size values) configs
        assert!(configs.len() >= 2);
    }

    #[test]
    fn test_grid_search_results() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "lr".to_string(),
            HyperparamSpace::discrete(vec![HyperparamValue::Float(0.1)]).unwrap(),
        );

        let mut grid_search = GridSearch::new(param_space, 3);

        let mut config = HashMap::new();
        config.insert("lr".to_string(), HyperparamValue::Float(0.1));

        grid_search.add_result(HyperparamResult::new(config.clone(), 0.9));
        grid_search.add_result(HyperparamResult::new(config.clone(), 0.95));
        grid_search.add_result(HyperparamResult::new(config, 0.85));

        let best = grid_search.best_result().unwrap();
        assert_eq!(best.score, 0.95);

        let sorted = grid_search.sorted_results();
        assert_eq!(sorted[0].score, 0.95);
        assert_eq!(sorted[1].score, 0.9);
        assert_eq!(sorted[2].score, 0.85);
    }

    #[test]
    fn test_random_search() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "lr".to_string(),
            HyperparamSpace::continuous(1e-4, 1e-1).unwrap(),
        );
        param_space.insert(
            "dropout".to_string(),
            HyperparamSpace::continuous(0.0, 0.5).unwrap(),
        );

        let mut random_search = RandomSearch::new(param_space, 10, 42);

        let configs = random_search.generate_configs();
        assert_eq!(configs.len(), 10);

        // Check that each config has all parameters
        for config in &configs {
            assert!(config.contains_key("lr"));
            assert!(config.contains_key("dropout"));
        }
    }

    #[test]
    fn test_random_search_results() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "lr".to_string(),
            HyperparamSpace::discrete(vec![HyperparamValue::Float(0.1)]).unwrap(),
        );

        let mut random_search = RandomSearch::new(param_space, 5, 42);

        let mut config = HashMap::new();
        config.insert("lr".to_string(), HyperparamValue::Float(0.1));

        random_search.add_result(HyperparamResult::new(config.clone(), 0.8));
        random_search.add_result(HyperparamResult::new(config, 0.9));

        let best = random_search.best_result().unwrap();
        assert_eq!(best.score, 0.9);

        assert_eq!(random_search.results().len(), 2);
    }

    #[test]
    fn test_hyperparam_result_with_metrics() {
        let mut config = HashMap::new();
        config.insert("lr".to_string(), HyperparamValue::Float(0.1));

        let result = HyperparamResult::new(config, 0.95)
            .with_metric("accuracy".to_string(), 0.95)
            .with_metric("loss".to_string(), 0.05);

        assert_eq!(result.score, 0.95);
        assert_eq!(result.metrics.get("accuracy"), Some(&0.95));
        assert_eq!(result.metrics.get("loss"), Some(&0.05));
    }

    #[test]
    fn test_grid_search_empty_space() {
        let grid_search = GridSearch::new(HashMap::new(), 3);
        let configs = grid_search.generate_configs();
        assert_eq!(configs.len(), 1); // One empty config
        assert!(configs[0].is_empty());
    }

    #[test]
    fn test_grid_search_total_configs() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "lr".to_string(),
            HyperparamSpace::discrete(vec![
                HyperparamValue::Float(0.1),
                HyperparamValue::Float(0.01),
            ])
            .unwrap(),
        );

        let grid_search = GridSearch::new(param_space, 3);
        assert_eq!(grid_search.total_configs(), 2);
    }
}
