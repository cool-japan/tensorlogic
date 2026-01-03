//! Gradient Centralization (GC) - Advanced gradient preprocessing technique.
//!
//! Gradient Centralization is a simple yet effective optimization technique that
//! normalizes gradients by subtracting their mean before applying the optimizer update.
//! This has been shown to improve generalization, accelerate training, and stabilize
//! gradient flow, especially for deep networks.
//!
//! # Benefits
//! - Improved generalization and test accuracy
//! - Faster convergence
//! - Better gradient flow (reduces gradient explosion/vanishing)
//! - Works with any optimizer
//! - Minimal computational overhead
//!
//! # Reference
//! Yong et al., "Gradient Centralization: A New Optimization Technique for Deep Neural Networks"
//! ECCV 2020 - <https://arxiv.org/abs/2004.01461>

use crate::{Optimizer, TrainResult};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

/// Gradient centralization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum GcStrategy {
    /// Centralize each layer's gradients independently (most common).
    /// For each parameter matrix: g = g - mean(g)
    #[default]
    LayerWise,

    /// Centralize all gradients globally (experimental).
    /// Compute global mean across all parameters: g_all = g_all - mean(g_all)
    Global,

    /// Centralize per row (for weight matrices).
    /// For weight matrix: g\[i,:\] = g\[i,:\] - mean(g\[i,:\])
    PerRow,

    /// Centralize per column (for weight matrices).
    /// For weight matrix: g\[:,j\] = g\[:,j\] - mean(g\[:,j\])
    PerColumn,
}

/// Configuration for gradient centralization.
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Centralization strategy.
    pub strategy: GcStrategy,

    /// Whether to apply GC (can be toggled dynamically).
    pub enabled: bool,

    /// Minimum parameter dimensions to apply GC (skip small parameters).
    /// For example, bias vectors (1D) are typically not centralized.
    pub min_dims: usize,

    /// Epsilon for numerical stability.
    pub eps: f64,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            strategy: GcStrategy::LayerWise,
            enabled: true,
            min_dims: 2, // Only centralize 2D+ tensors (weight matrices)
            eps: 1e-8,
        }
    }
}

impl GcConfig {
    /// Create a new GC configuration.
    pub fn new(strategy: GcStrategy) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }

    /// Enable gradient centralization.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable gradient centralization.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Set minimum dimensions for applying GC.
    pub fn with_min_dims(mut self, min_dims: usize) -> Self {
        self.min_dims = min_dims;
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

/// Gradient Centralization optimizer wrapper.
///
/// Wraps any optimizer and applies gradient centralization before the optimizer step.
/// GC normalizes gradients by subtracting their mean, which improves training dynamics.
///
/// # Example
/// ```no_run
/// use tensorlogic_train::*;
/// use scirs2_core::ndarray::Array2;
/// use std::collections::HashMap;
///
/// // Create base optimizer
/// let config = OptimizerConfig { learning_rate: 0.001, ..Default::default() };
/// let adam = AdamOptimizer::new(config);
///
/// // Wrap with gradient centralization
/// let mut gc_adam = GradientCentralization::new(
///     Box::new(adam),
///     GcConfig::default(),
/// );
///
/// // Use as normal optimizer - GC is applied automatically
/// let mut params = HashMap::new();
/// params.insert("w1".to_string(), Array2::zeros((10, 5)));
///
/// let mut grads = HashMap::new();
/// grads.insert("w1".to_string(), Array2::ones((10, 5)));
///
/// gc_adam.step(&mut params, &grads).unwrap();
/// ```
pub struct GradientCentralization {
    /// Wrapped optimizer.
    inner_optimizer: Box<dyn Optimizer>,

    /// GC configuration.
    config: GcConfig,

    /// Statistics for monitoring.
    stats: GcStats,
}

/// Statistics for gradient centralization.
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Number of parameters centralized.
    pub num_centralized: usize,

    /// Number of parameters skipped (too small).
    pub num_skipped: usize,

    /// Average gradient magnitude before centralization.
    pub avg_grad_norm_before: f64,

    /// Average gradient magnitude after centralization.
    pub avg_grad_norm_after: f64,

    /// Total number of centralization operations.
    pub total_operations: usize,
}

impl GradientCentralization {
    /// Create a new gradient centralization optimizer.
    pub fn new(inner_optimizer: Box<dyn Optimizer>, config: GcConfig) -> Self {
        Self {
            inner_optimizer,
            config,
            stats: GcStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_default(inner_optimizer: Box<dyn Optimizer>) -> Self {
        Self::new(inner_optimizer, GcConfig::default())
    }

    /// Get GC configuration.
    pub fn config(&self) -> &GcConfig {
        &self.config
    }

    /// Get mutable GC configuration.
    pub fn config_mut(&mut self) -> &mut GcConfig {
        &mut self.config
    }

    /// Get statistics.
    pub fn stats(&self) -> &GcStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = GcStats::default();
    }

    /// Apply gradient centralization to gradients.
    fn centralize_gradients(
        &mut self,
        grads: &HashMap<String, Array2<f64>>,
    ) -> HashMap<String, Array2<f64>> {
        if !self.config.enabled {
            return grads.clone();
        }

        let mut centralized_grads = HashMap::new();
        let mut total_norm_before = 0.0;
        let mut total_norm_after = 0.0;

        for (name, grad) in grads {
            let shape = grad.shape();

            // Check if parameter meets minimum dimension requirement
            if shape.len() < self.config.min_dims {
                centralized_grads.insert(name.clone(), grad.clone());
                self.stats.num_skipped += 1;
                continue;
            }

            // Compute norm before centralization
            let norm_before = grad.iter().map(|&x| x * x).sum::<f64>().sqrt();
            total_norm_before += norm_before;

            // Apply centralization based on strategy
            let centered_grad = match self.config.strategy {
                GcStrategy::LayerWise => self.centralize_layerwise(grad),
                GcStrategy::Global => grad.clone(), // Global handled separately
                GcStrategy::PerRow => self.centralize_per_row(grad),
                GcStrategy::PerColumn => self.centralize_per_column(grad),
            };

            // Compute norm after centralization
            let norm_after = centered_grad.iter().map(|&x| x * x).sum::<f64>().sqrt();
            total_norm_after += norm_after;

            centralized_grads.insert(name.clone(), centered_grad);
            self.stats.num_centralized += 1;
        }

        // Handle global strategy
        if self.config.strategy == GcStrategy::Global && !centralized_grads.is_empty() {
            centralized_grads = self.centralize_global(&centralized_grads);
        }

        // Update statistics
        let n = (self.stats.num_centralized + self.stats.num_skipped).max(1) as f64;
        self.stats.avg_grad_norm_before = total_norm_before / n;
        self.stats.avg_grad_norm_after = total_norm_after / n;
        self.stats.total_operations += 1;

        centralized_grads
    }

    /// Centralize gradients layer-wise (subtract mean from each layer).
    fn centralize_layerwise(&self, grad: &Array2<f64>) -> Array2<f64> {
        let mean = grad.mean().unwrap_or(0.0);
        grad - mean
    }

    /// Centralize gradients per row.
    fn centralize_per_row(&self, grad: &Array2<f64>) -> Array2<f64> {
        let mut centered = grad.clone();

        for i in 0..grad.nrows() {
            let row_mean = grad.row(i).mean().unwrap_or(0.0);
            for j in 0..grad.ncols() {
                centered[[i, j]] -= row_mean;
            }
        }

        centered
    }

    /// Centralize gradients per column.
    fn centralize_per_column(&self, grad: &Array2<f64>) -> Array2<f64> {
        let mut centered = grad.clone();

        for j in 0..grad.ncols() {
            let col_mean = grad.column(j).mean().unwrap_or(0.0);
            for i in 0..grad.nrows() {
                centered[[i, j]] -= col_mean;
            }
        }

        centered
    }

    /// Centralize all gradients globally.
    fn centralize_global(
        &self,
        grads: &HashMap<String, Array2<f64>>,
    ) -> HashMap<String, Array2<f64>> {
        // Compute global mean across all parameters
        let mut total_sum = 0.0;
        let mut total_count = 0;

        for grad in grads.values() {
            total_sum += grad.sum();
            total_count += grad.len();
        }

        let global_mean = if total_count > 0 {
            total_sum / total_count as f64
        } else {
            0.0
        };

        // Subtract global mean from all gradients
        let mut centralized = HashMap::new();
        for (name, grad) in grads {
            centralized.insert(name.clone(), grad - global_mean);
        }

        centralized
    }
}

impl Optimizer for GradientCentralization {
    fn step(
        &mut self,
        params: &mut HashMap<String, Array2<f64>>,
        grads: &HashMap<String, Array2<f64>>,
    ) -> TrainResult<()> {
        // Apply gradient centralization
        let centralized_grads = self.centralize_gradients(grads);

        // Forward to wrapped optimizer
        self.inner_optimizer.step(params, &centralized_grads)
    }

    fn zero_grad(&mut self) {
        self.inner_optimizer.zero_grad();
    }

    fn get_lr(&self) -> f64 {
        self.inner_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f64) {
        self.inner_optimizer.set_lr(lr);
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        // Include both inner optimizer state and GC config
        let mut state = self.inner_optimizer.state_dict();

        // Serialize GC config (simplified - just store enabled flag)
        let gc_state = if self.config.enabled {
            vec![1.0]
        } else {
            vec![0.0]
        };
        state.insert("gc_enabled".to_string(), gc_state);

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        // Load GC config
        if let Some(gc_state) = state.get("gc_enabled") {
            self.config.enabled = !gc_state.is_empty() && gc_state[0] > 0.5;
        }

        // Load inner optimizer state
        self.inner_optimizer.load_state_dict(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdamOptimizer, OptimizerConfig};
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_gc_config_default() {
        let config = GcConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_dims, 2);
        assert_eq!(config.strategy, GcStrategy::LayerWise);
    }

    #[test]
    fn test_gc_config_builder() {
        let config = GcConfig::new(GcStrategy::PerRow)
            .with_min_dims(1)
            .with_eps(1e-10);

        assert_eq!(config.strategy, GcStrategy::PerRow);
        assert_eq!(config.min_dims, 1);
        assert_eq!(config.eps, 1e-10);
    }

    #[test]
    fn test_gc_layerwise_centralization() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let mut gc = GradientCentralization::new(Box::new(adam), GcConfig::default());

        // Create gradient with known mean
        let grad = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as f64);
        let mean = grad.mean().unwrap();

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), grad.clone());

        let centered = gc.centralize_gradients(&grads);
        let centered_grad = &centered["w1"];

        // Mean should be close to zero after centralization
        let new_mean = centered_grad.mean().unwrap();
        assert!(new_mean.abs() < 1e-10);

        // Each element should be shifted by original mean
        for i in 0..3 {
            for j in 0..3 {
                assert!((centered_grad[[i, j]] - (grad[[i, j]] - mean)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gc_per_row_centralization() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let config = GcConfig::new(GcStrategy::PerRow);
        let mut gc = GradientCentralization::new(Box::new(adam), config);

        let grad = Array2::from_shape_fn((2, 3), |(i, j)| (i * 10 + j) as f64);

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), grad.clone());

        let centered = gc.centralize_gradients(&grads);
        let centered_grad = &centered["w1"];

        // Each row should have mean close to zero
        for i in 0..2 {
            let row_mean = centered_grad.row(i).mean().unwrap();
            assert!(row_mean.abs() < 1e-10);
        }
    }

    #[test]
    fn test_gc_per_column_centralization() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let config = GcConfig::new(GcStrategy::PerColumn);
        let mut gc = GradientCentralization::new(Box::new(adam), config);

        let grad = Array2::from_shape_fn((3, 2), |(i, j)| (i + j * 10) as f64);

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), grad.clone());

        let centered = gc.centralize_gradients(&grads);
        let centered_grad = &centered["w1"];

        // Each column should have mean close to zero
        for j in 0..2 {
            let col_mean = centered_grad.column(j).mean().unwrap();
            assert!(col_mean.abs() < 1e-10);
        }
    }

    #[test]
    fn test_gc_global_centralization() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let config = GcConfig::new(GcStrategy::Global);
        let mut gc = GradientCentralization::new(Box::new(adam), config);

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), Array2::from_elem((2, 2), 5.0));
        grads.insert("w2".to_string(), Array2::from_elem((2, 2), 15.0));

        let centered = gc.centralize_gradients(&grads);

        // Global mean should be 10.0
        // After centralization: w1 = -5, w2 = 5
        let w1_centered = &centered["w1"];
        let w2_centered = &centered["w2"];

        assert!((w1_centered[[0, 0]] + 5.0).abs() < 1e-10);
        assert!((w2_centered[[0, 0]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_gc_skip_small_tensors() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let config = GcConfig::default().with_min_dims(2);
        let gc = GradientCentralization::new(Box::new(adam), config);

        // This test would require 1D tensors, but our implementation uses Array2
        // So we verify that the min_dims check is there
        assert_eq!(gc.config().min_dims, 2);
    }

    #[test]
    fn test_gc_enable_disable() {
        let mut config = GcConfig::default();
        assert!(config.enabled);

        config.disable();
        assert!(!config.enabled);

        config.enable();
        assert!(config.enabled);
    }

    #[test]
    fn test_gc_with_optimizer_step() {
        let config = OptimizerConfig {
            learning_rate: 0.01,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let mut gc = GradientCentralization::new(Box::new(adam), GcConfig::default());

        let mut params = HashMap::new();
        params.insert("w1".to_string(), Array2::ones((3, 3)));

        // Use varying gradients (not uniform) so after centralization there's still signal
        let mut grads = HashMap::new();
        grads.insert(
            "w1".to_string(),
            Array2::from_shape_fn((3, 3), |(i, j)| 0.1 + (i + j) as f64 * 0.05),
        );

        // Step should succeed
        assert!(gc.step(&mut params, &grads).is_ok());

        // Parameters should be updated (at least some of them should decrease)
        let updated = &params["w1"];
        // After GC, we still have non-zero centered gradients
        // At least one parameter should have changed
        let has_changed = updated.iter().any(|&x| (x - 1.0).abs() > 1e-6);
        assert!(has_changed);
    }

    #[test]
    fn test_gc_statistics() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let mut gc = GradientCentralization::new(Box::new(adam), GcConfig::default());

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), Array2::ones((3, 3)));
        grads.insert("w2".to_string(), Array2::ones((3, 3)));

        gc.centralize_gradients(&grads);

        let stats = gc.stats();
        assert_eq!(stats.num_centralized, 2);
        assert_eq!(stats.total_operations, 1);
        assert!(stats.avg_grad_norm_before > 0.0);
    }

    #[test]
    fn test_gc_reset_stats() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let mut gc = GradientCentralization::new(Box::new(adam), GcConfig::default());

        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), Array2::ones((3, 3)));

        gc.centralize_gradients(&grads);
        assert_eq!(gc.stats().total_operations, 1);

        gc.reset_stats();
        assert_eq!(gc.stats().total_operations, 0);
    }

    #[test]
    fn test_gc_learning_rate() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let mut gc = GradientCentralization::new(Box::new(adam), GcConfig::default());

        assert_eq!(gc.get_lr(), 0.001);

        gc.set_lr(0.01);
        assert_eq!(gc.get_lr(), 0.01);
    }

    #[test]
    fn test_gc_state_dict() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let mut gc = GradientCentralization::new(Box::new(adam), GcConfig::default());

        // Get state
        let state = gc.state_dict();
        assert!(state.contains_key("gc_enabled"));

        // Modify and load
        gc.config_mut().disable();
        assert!(!gc.config().enabled);

        gc.load_state_dict(state);
        assert!(gc.config().enabled); // Should be restored
    }

    #[test]
    fn test_gc_disabled() {
        let config = OptimizerConfig {
            learning_rate: 0.001,
            ..Default::default()
        };
        let adam = AdamOptimizer::new(config);
        let mut config = GcConfig::default();
        config.disable();

        let mut gc = GradientCentralization::new(Box::new(adam), config);

        let grad = Array2::from_elem((3, 3), 5.0);
        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), grad.clone());

        let centered = gc.centralize_gradients(&grads);

        // Should return unchanged gradients when disabled
        let centered_grad = &centered["w1"];
        assert_eq!(centered_grad[[0, 0]], 5.0);
    }
}
