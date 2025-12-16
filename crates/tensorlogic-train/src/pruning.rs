//! Model pruning for compression and acceleration.
//!
//! This module provides various pruning strategies to reduce model size and computational cost:
//! - **Unstructured pruning**: Remove individual weights based on magnitude, gradient, or other criteria
//! - **Structured pruning**: Remove entire neurons, channels, or filters
//! - **Iterative pruning**: Gradually increase pruning ratio during training
//! - **Dynamic pruning**: Adaptively prune based on runtime statistics
//!
//! # Pruning Strategies
//!
//! ## Magnitude-based Pruning
//! Prune weights with smallest absolute values (most common and effective):
//! ```rust
//! use tensorlogic_train::{MagnitudePruner, Pruner, PruningConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! let weights = Array2::from_shape_vec((3, 3), vec![
//!     0.1, 0.5, 0.9,
//!     0.2, 0.6, 0.01,
//!     0.3, 0.7, 0.8,
//! ]).unwrap();
//!
//! let config = PruningConfig {
//!     pruning_ratio: 0.3, // Remove 30% of weights
//!     structured: false,
//!     iterative: false,
//!     ..Default::default()
//! };
//!
//! let pruner = MagnitudePruner::new(config);
//! let (pruned_weights, mask) = pruner.prune(&weights).unwrap();
//! ```
//!
//! ## Gradient-based Pruning
//! Prune weights with smallest gradient magnitudes (less sensitive to training):
//! ```rust
//! use tensorlogic_train::{GradientPruner, PruningConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! let gradients = Array2::<f64>::zeros((3, 3));
//! let config = PruningConfig::default();
//! let pruner = GradientPruner::new(config);
//! ```
//!
//! ## Structured Pruning
//! Remove entire neurons, channels, or filters:
//! ```rust
//! use tensorlogic_train::{StructuredPruner, PruningConfig, StructuredPruningAxis};
//!
//! let config = PruningConfig {
//!     pruning_ratio: 0.5,
//!     structured: true,
//!     ..Default::default()
//! };
//!
//! let pruner = StructuredPruner::new(config, StructuredPruningAxis::Rows);
//! ```

use scirs2_core::ndarray::{Array2, ArrayD, Axis, Ix2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{TrainError, TrainResult};

/// Configuration for pruning strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Fraction of weights to prune (0.0 to 1.0)
    pub pruning_ratio: f64,
    /// Whether to use structured pruning (entire neurons/channels)
    pub structured: bool,
    /// Use iterative pruning (gradually increase pruning ratio)
    pub iterative: bool,
    /// Number of iterations for iterative pruning
    pub num_iterations: usize,
    /// Initial pruning ratio for iterative pruning
    pub initial_ratio: f64,
    /// Final pruning ratio for iterative pruning
    pub final_ratio: f64,
    /// Pruning schedule: "linear", "exponential", "cosine"
    pub schedule: String,
    /// Minimum weight magnitude threshold (weights below this are always pruned)
    pub min_threshold: f64,
    /// Whether to use global pruning (across all layers) or local (per-layer)
    pub global_pruning: bool,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            pruning_ratio: 0.5,
            structured: false,
            iterative: false,
            num_iterations: 10,
            initial_ratio: 0.0,
            final_ratio: 0.9,
            schedule: "linear".to_string(),
            min_threshold: 1e-8,
            global_pruning: false,
        }
    }
}

/// Axis for structured pruning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredPruningAxis {
    /// Prune rows (output neurons)
    Rows,
    /// Prune columns (input neurons)
    Columns,
    /// Prune both (for convolutional filters)
    Both,
}

/// Pruning mask indicating which weights are kept (1.0) or removed (0.0).
pub type PruningMask = ArrayD<f64>;

/// Statistics about pruned model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningStats {
    /// Total number of parameters before pruning
    pub total_params: usize,
    /// Number of parameters after pruning
    pub active_params: usize,
    /// Pruning ratio achieved
    pub pruning_ratio: f64,
    /// Number of pruning iterations performed
    pub iterations: usize,
    /// Per-layer pruning statistics
    pub per_layer_stats: HashMap<String, LayerPruningStats>,
}

/// Pruning statistics for a single layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPruningStats {
    /// Layer name
    pub name: String,
    /// Original parameter count
    pub original_params: usize,
    /// Active parameter count after pruning
    pub active_params: usize,
    /// Pruning ratio for this layer
    pub ratio: f64,
}

impl PruningStats {
    /// Calculate compression ratio (original size / pruned size).
    pub fn compression_ratio(&self) -> f64 {
        if self.active_params == 0 {
            0.0
        } else {
            self.total_params as f64 / self.active_params as f64
        }
    }

    /// Calculate FLOPs reduction (approximate).
    pub fn flops_reduction(&self) -> f64 {
        self.pruning_ratio
    }

    /// Pretty print pruning statistics.
    pub fn summary(&self) -> String {
        format!(
            "Pruning Stats:\n\
             - Total params: {}\n\
             - Active params: {}\n\
             - Pruned: {} ({:.2}%)\n\
             - Compression: {:.2}x\n\
             - Est. FLOPs reduction: {:.2}%",
            self.total_params,
            self.active_params,
            self.total_params - self.active_params,
            self.pruning_ratio * 100.0,
            self.compression_ratio(),
            self.flops_reduction() * 100.0
        )
    }
}

/// Trait for pruning strategies.
pub trait Pruner {
    /// Prune weights and return pruned weights and mask.
    fn prune(&self, weights: &Array2<f64>) -> TrainResult<(Array2<f64>, PruningMask)>;

    /// Generate pruning mask without modifying weights.
    fn generate_mask(&self, weights: &Array2<f64>) -> TrainResult<PruningMask>;

    /// Apply existing mask to weights.
    fn apply_mask(&self, weights: &Array2<f64>, mask: &PruningMask) -> TrainResult<Array2<f64>>;

    /// Get pruning configuration.
    fn config(&self) -> &PruningConfig;

    /// Update pruning ratio for iterative pruning.
    fn update_ratio(&mut self, iteration: usize);
}

/// Magnitude-based pruning (prune smallest weights).
pub struct MagnitudePruner {
    config: PruningConfig,
    current_ratio: f64,
}

impl MagnitudePruner {
    /// Create a new magnitude-based pruner.
    pub fn new(config: PruningConfig) -> Self {
        let current_ratio = if config.iterative {
            config.initial_ratio
        } else {
            config.pruning_ratio
        };
        Self {
            config,
            current_ratio,
        }
    }

    /// Calculate pruning threshold based on weight distribution.
    fn calculate_threshold(&self, weights: &Array2<f64>) -> f64 {
        let mut abs_weights: Vec<f64> = weights.iter().map(|w| w.abs()).collect();
        abs_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let prune_count = (abs_weights.len() as f64 * self.current_ratio) as usize;
        if prune_count >= abs_weights.len() {
            abs_weights.last().copied().unwrap_or(0.0)
        } else {
            abs_weights[prune_count]
        }
    }
}

impl Pruner for MagnitudePruner {
    fn prune(&self, weights: &Array2<f64>) -> TrainResult<(Array2<f64>, PruningMask)> {
        let mask = self.generate_mask(weights)?;
        let pruned = self.apply_mask(weights, &mask)?;
        Ok((pruned, mask))
    }

    fn generate_mask(&self, weights: &Array2<f64>) -> TrainResult<PruningMask> {
        let threshold = self
            .calculate_threshold(weights)
            .max(self.config.min_threshold);

        let mask = weights.mapv(|w| if w.abs() >= threshold { 1.0 } else { 0.0 });
        Ok(mask.into_dyn())
    }

    fn apply_mask(&self, weights: &Array2<f64>, mask: &PruningMask) -> TrainResult<Array2<f64>> {
        let mask_2d = mask
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|e| TrainError::ConfigError(format!("Mask shape mismatch: {}", e)))?;

        if weights.shape() != mask_2d.shape() {
            return Err(TrainError::ConfigError(format!(
                "Weight and mask shapes do not match: {:?} vs {:?}",
                weights.shape(),
                mask_2d.shape()
            )));
        }

        Ok(weights * &mask_2d)
    }

    fn config(&self) -> &PruningConfig {
        &self.config
    }

    fn update_ratio(&mut self, iteration: usize) {
        if !self.config.iterative || iteration >= self.config.num_iterations {
            return;
        }

        let progress = iteration as f64 / (self.config.num_iterations - 1) as f64;
        self.current_ratio = match self.config.schedule.as_str() {
            "linear" => {
                self.config.initial_ratio
                    + (self.config.final_ratio - self.config.initial_ratio) * progress
            }
            "exponential" => {
                let log_initial = self.config.initial_ratio.max(1e-8).ln();
                let log_final = self.config.final_ratio.ln();
                (log_initial + (log_final - log_initial) * progress).exp()
            }
            "cosine" => {
                let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                self.config.final_ratio
                    + (self.config.initial_ratio - self.config.final_ratio) * cosine_decay
            }
            _ => self.config.pruning_ratio,
        };
    }
}

/// Gradient-based pruning (prune weights with smallest gradients).
pub struct GradientPruner {
    config: PruningConfig,
    current_ratio: f64,
    gradient_history: HashMap<String, Vec<Array2<f64>>>,
}

impl GradientPruner {
    /// Create a new gradient-based pruner.
    pub fn new(config: PruningConfig) -> Self {
        let current_ratio = if config.iterative {
            config.initial_ratio
        } else {
            config.pruning_ratio
        };
        Self {
            config,
            current_ratio,
            gradient_history: HashMap::new(),
        }
    }

    /// Update gradient history for a layer.
    pub fn update_gradients(&mut self, layer_name: &str, gradients: Array2<f64>) {
        self.gradient_history
            .entry(layer_name.to_string())
            .or_default()
            .push(gradients);
    }

    /// Calculate average gradient magnitude for a layer.
    fn average_gradient_magnitude(&self, layer_name: &str) -> Option<Array2<f64>> {
        let gradients = self.gradient_history.get(layer_name)?;
        if gradients.is_empty() {
            return None;
        }

        let mut sum = gradients[0].mapv(|g| g.abs());
        for grad in &gradients[1..] {
            sum = sum + grad.mapv(|g| g.abs());
        }
        Some(sum / gradients.len() as f64)
    }

    /// Calculate pruning threshold based on gradient distribution.
    fn calculate_threshold(&self, gradients: &Array2<f64>) -> f64 {
        let mut abs_grads: Vec<f64> = gradients.iter().map(|g| g.abs()).collect();
        abs_grads.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let prune_count = (abs_grads.len() as f64 * self.current_ratio) as usize;
        if prune_count >= abs_grads.len() {
            abs_grads.last().copied().unwrap_or(0.0)
        } else {
            abs_grads[prune_count]
        }
    }

    /// Prune based on gradient history.
    pub fn prune_with_history(
        &self,
        weights: &Array2<f64>,
        layer_name: &str,
    ) -> TrainResult<(Array2<f64>, PruningMask)> {
        if let Some(avg_grads) = self.average_gradient_magnitude(layer_name) {
            let threshold = self
                .calculate_threshold(&avg_grads)
                .max(self.config.min_threshold);
            let mask = avg_grads.mapv(|g| if g >= threshold { 1.0 } else { 0.0 });
            let pruned = weights * &mask;
            Ok((pruned, mask.into_dyn()))
        } else {
            // Fall back to magnitude pruning if no gradient history
            let magnitude_pruner = MagnitudePruner::new(self.config.clone());
            magnitude_pruner.prune(weights)
        }
    }
}

impl Pruner for GradientPruner {
    fn prune(&self, weights: &Array2<f64>) -> TrainResult<(Array2<f64>, PruningMask)> {
        // Without gradient information, fall back to magnitude pruning
        let magnitude_pruner = MagnitudePruner::new(self.config.clone());
        magnitude_pruner.prune(weights)
    }

    fn generate_mask(&self, weights: &Array2<f64>) -> TrainResult<PruningMask> {
        let magnitude_pruner = MagnitudePruner::new(self.config.clone());
        magnitude_pruner.generate_mask(weights)
    }

    fn apply_mask(&self, weights: &Array2<f64>, mask: &PruningMask) -> TrainResult<Array2<f64>> {
        let magnitude_pruner = MagnitudePruner::new(self.config.clone());
        magnitude_pruner.apply_mask(weights, mask)
    }

    fn config(&self) -> &PruningConfig {
        &self.config
    }

    fn update_ratio(&mut self, iteration: usize) {
        if !self.config.iterative || iteration >= self.config.num_iterations {
            return;
        }

        let progress = iteration as f64 / (self.config.num_iterations - 1) as f64;
        self.current_ratio = match self.config.schedule.as_str() {
            "linear" => {
                self.config.initial_ratio
                    + (self.config.final_ratio - self.config.initial_ratio) * progress
            }
            "exponential" => {
                let log_initial = self.config.initial_ratio.max(1e-8).ln();
                let log_final = self.config.final_ratio.ln();
                (log_initial + (log_final - log_initial) * progress).exp()
            }
            "cosine" => {
                let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                self.config.final_ratio
                    + (self.config.initial_ratio - self.config.final_ratio) * cosine_decay
            }
            _ => self.config.pruning_ratio,
        };
    }
}

/// Structured pruning (remove entire neurons/channels/filters).
pub struct StructuredPruner {
    config: PruningConfig,
    axis: StructuredPruningAxis,
    current_ratio: f64,
}

impl StructuredPruner {
    /// Create a new structured pruner.
    pub fn new(config: PruningConfig, axis: StructuredPruningAxis) -> Self {
        let current_ratio = if config.iterative {
            config.initial_ratio
        } else {
            config.pruning_ratio
        };
        Self {
            config,
            axis,
            current_ratio,
        }
    }

    /// Calculate importance scores for rows or columns.
    fn calculate_importance(&self, weights: &Array2<f64>, axis: Axis) -> Vec<f64> {
        let axis_len = weights.len_of(axis);
        (0..axis_len)
            .map(|i| {
                let slice = weights.index_axis(axis, i);
                // L2 norm as importance metric
                slice.iter().map(|&w| w * w).sum::<f64>().sqrt()
            })
            .collect()
    }

    /// Determine which units to prune based on importance scores.
    fn select_units_to_prune(&self, importance: &[f64]) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = importance.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let prune_count = (importance.len() as f64 * self.current_ratio) as usize;
        indexed
            .iter()
            .take(prune_count)
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// Generate mask for structured pruning.
    fn generate_structured_mask(&self, weights: &Array2<f64>) -> TrainResult<PruningMask> {
        let (nrows, ncols) = weights.dim();
        let mut mask = Array2::ones((nrows, ncols));

        match self.axis {
            StructuredPruningAxis::Rows => {
                let importance = self.calculate_importance(weights, Axis(0));
                let to_prune = self.select_units_to_prune(&importance);
                for &row_idx in &to_prune {
                    mask.row_mut(row_idx).fill(0.0);
                }
            }
            StructuredPruningAxis::Columns => {
                let importance = self.calculate_importance(weights, Axis(1));
                let to_prune = self.select_units_to_prune(&importance);
                for &col_idx in &to_prune {
                    mask.column_mut(col_idx).fill(0.0);
                }
            }
            StructuredPruningAxis::Both => {
                // Prune both rows and columns
                let row_importance = self.calculate_importance(weights, Axis(0));
                let col_importance = self.calculate_importance(weights, Axis(1));

                let rows_to_prune = self.select_units_to_prune(&row_importance);
                let cols_to_prune = self.select_units_to_prune(&col_importance);

                for &row_idx in &rows_to_prune {
                    mask.row_mut(row_idx).fill(0.0);
                }
                for &col_idx in &cols_to_prune {
                    mask.column_mut(col_idx).fill(0.0);
                }
            }
        }

        Ok(mask.into_dyn())
    }
}

impl Pruner for StructuredPruner {
    fn prune(&self, weights: &Array2<f64>) -> TrainResult<(Array2<f64>, PruningMask)> {
        let mask = self.generate_structured_mask(weights)?;
        let pruned = self.apply_mask(weights, &mask)?;
        Ok((pruned, mask))
    }

    fn generate_mask(&self, weights: &Array2<f64>) -> TrainResult<PruningMask> {
        self.generate_structured_mask(weights)
    }

    fn apply_mask(&self, weights: &Array2<f64>, mask: &PruningMask) -> TrainResult<Array2<f64>> {
        let mask_2d = mask
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|e| TrainError::ConfigError(format!("Mask shape mismatch: {}", e)))?;

        if weights.shape() != mask_2d.shape() {
            return Err(TrainError::ConfigError(format!(
                "Weight and mask shapes do not match: {:?} vs {:?}",
                weights.shape(),
                mask_2d.shape()
            )));
        }

        Ok(weights * &mask_2d)
    }

    fn config(&self) -> &PruningConfig {
        &self.config
    }

    fn update_ratio(&mut self, iteration: usize) {
        if !self.config.iterative || iteration >= self.config.num_iterations {
            return;
        }

        let progress = iteration as f64 / (self.config.num_iterations - 1) as f64;
        self.current_ratio = match self.config.schedule.as_str() {
            "linear" => {
                self.config.initial_ratio
                    + (self.config.final_ratio - self.config.initial_ratio) * progress
            }
            "exponential" => {
                let log_initial = self.config.initial_ratio.max(1e-8).ln();
                let log_final = self.config.final_ratio.ln();
                (log_initial + (log_final - log_initial) * progress).exp()
            }
            "cosine" => {
                let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                self.config.final_ratio
                    + (self.config.initial_ratio - self.config.final_ratio) * cosine_decay
            }
            _ => self.config.pruning_ratio,
        };
    }
}

/// Global pruning across multiple layers.
pub struct GlobalPruner {
    config: PruningConfig,
    layer_weights: HashMap<String, Array2<f64>>,
}

impl GlobalPruner {
    /// Create a new global pruner.
    pub fn new(config: PruningConfig) -> Self {
        Self {
            config,
            layer_weights: HashMap::new(),
        }
    }

    /// Add a layer to the global pruning pool.
    pub fn add_layer(&mut self, name: &str, weights: Array2<f64>) {
        self.layer_weights.insert(name.to_string(), weights);
    }

    /// Calculate global threshold across all layers.
    fn calculate_global_threshold(&self) -> f64 {
        let mut all_weights: Vec<f64> = self
            .layer_weights
            .values()
            .flat_map(|w| w.iter().map(|x| x.abs()))
            .collect();

        all_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let total_params = all_weights.len();
        let prune_count = (total_params as f64 * self.config.pruning_ratio) as usize;

        if prune_count >= total_params {
            all_weights.last().copied().unwrap_or(0.0)
        } else {
            all_weights[prune_count]
        }
    }

    /// Prune all layers using global threshold.
    pub fn prune_all(&self) -> TrainResult<HashMap<String, (Array2<f64>, PruningMask)>> {
        let threshold = self
            .calculate_global_threshold()
            .max(self.config.min_threshold);

        let mut results = HashMap::new();
        for (name, weights) in &self.layer_weights {
            let mask = weights.mapv(|w| if w.abs() >= threshold { 1.0 } else { 0.0 });
            let pruned = weights * &mask;
            results.insert(name.clone(), (pruned, mask.into_dyn()));
        }

        Ok(results)
    }

    /// Generate pruning statistics.
    pub fn statistics(&self, pruned: &HashMap<String, (Array2<f64>, PruningMask)>) -> PruningStats {
        let mut total_params = 0;
        let mut active_params = 0;
        let mut per_layer_stats = HashMap::new();

        for (name, weights) in &self.layer_weights {
            let layer_total = weights.len();
            total_params += layer_total;

            if let Some((_, mask)) = pruned.get(name) {
                let layer_active = mask.iter().filter(|&&m| m > 0.5).count();
                active_params += layer_active;

                per_layer_stats.insert(
                    name.clone(),
                    LayerPruningStats {
                        name: name.clone(),
                        original_params: layer_total,
                        active_params: layer_active,
                        ratio: 1.0 - (layer_active as f64 / layer_total as f64),
                    },
                );
            }
        }

        PruningStats {
            total_params,
            active_params,
            pruning_ratio: 1.0 - (active_params as f64 / total_params as f64),
            iterations: 1,
            per_layer_stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_magnitude_pruner() {
        let weights =
            Array2::from_shape_vec((3, 3), vec![0.1, 0.5, 0.9, 0.2, 0.6, 0.01, 0.3, 0.7, 0.8])
                .unwrap();

        let config = PruningConfig {
            pruning_ratio: 0.3,
            structured: false,
            iterative: false,
            ..Default::default()
        };

        let pruner = MagnitudePruner::new(config);
        let (pruned, mask) = pruner.prune(&weights).unwrap();

        // Check that smallest weights are pruned
        let active_count = mask.iter().filter(|&&m| m > 0.5).count();
        // With 30% pruning ratio, prune_count = (9 * 0.3) = 2.7 -> 2
        // So we keep 9 - 2 = 7 weights (approximately 78% kept)
        let prune_count = (9.0 * 0.3) as usize;
        let expected_active = 9 - prune_count;
        assert_eq!(active_count, expected_active);

        // Check that pruned weights are zeroed
        for ((&p, &m), &w) in pruned.iter().zip(mask.iter()).zip(weights.iter()) {
            if m < 0.5 {
                assert_abs_diff_eq!(p, 0.0, epsilon = 1e-10);
            } else {
                assert_abs_diff_eq!(p, w, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_iterative_pruning() {
        let mut pruner = MagnitudePruner::new(PruningConfig {
            pruning_ratio: 0.0,
            structured: false,
            iterative: true,
            num_iterations: 5,
            initial_ratio: 0.0,
            final_ratio: 0.5,
            schedule: "linear".to_string(),
            ..Default::default()
        });

        assert_abs_diff_eq!(pruner.current_ratio, 0.0, epsilon = 1e-10);

        pruner.update_ratio(0);
        assert_abs_diff_eq!(pruner.current_ratio, 0.0, epsilon = 1e-3);

        pruner.update_ratio(2);
        assert_abs_diff_eq!(pruner.current_ratio, 0.25, epsilon = 1e-3);

        pruner.update_ratio(4);
        assert_abs_diff_eq!(pruner.current_ratio, 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_structured_pruner_rows() {
        let weights = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.1, 0.1, 0.1, // Row 0: low magnitude
                0.9, 0.9, 0.9, // Row 1: high magnitude
                0.2, 0.2, 0.2, // Row 2: low magnitude
                0.8, 0.8, 0.8, // Row 3: high magnitude
            ],
        )
        .unwrap();

        let config = PruningConfig {
            pruning_ratio: 0.5, // Prune 50% of rows (2 out of 4)
            structured: true,
            ..Default::default()
        };

        let pruner = StructuredPruner::new(config, StructuredPruningAxis::Rows);
        let (pruned, _mask) = pruner.prune(&weights).unwrap();

        // Check that 2 rows are completely zeroed
        let zero_rows = (0..4)
            .filter(|&i| pruned.row(i).iter().all(|&x| x.abs() < 1e-10))
            .count();
        assert_eq!(zero_rows, 2);

        // Check that the low magnitude rows (0 and 2) are pruned
        assert!(pruned.row(0).iter().all(|&x| x.abs() < 1e-10));
        assert!(pruned.row(2).iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_structured_pruner_columns() {
        let weights = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.1, 0.9, 0.2, 0.8, // Each column has varying magnitudes
                0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8,
            ],
        )
        .unwrap();

        let config = PruningConfig {
            pruning_ratio: 0.5,
            structured: true,
            ..Default::default()
        };

        let pruner = StructuredPruner::new(config, StructuredPruningAxis::Columns);
        let (pruned, _mask) = pruner.prune(&weights).unwrap();

        // Check that 2 columns are completely zeroed
        let zero_cols = (0..4)
            .filter(|&i| pruned.column(i).iter().all(|&x| x.abs() < 1e-10))
            .count();
        assert_eq!(zero_cols, 2);
    }

    #[test]
    fn test_global_pruner() {
        let mut global_pruner = GlobalPruner::new(PruningConfig {
            pruning_ratio: 0.5,
            global_pruning: true,
            ..Default::default()
        });

        let layer1 = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let layer2 = Array2::from_shape_vec((2, 2), vec![0.5, 0.6, 0.7, 0.8]).unwrap();

        global_pruner.add_layer("layer1", layer1);
        global_pruner.add_layer("layer2", layer2);

        let pruned = global_pruner.prune_all().unwrap();
        let stats = global_pruner.statistics(&pruned);

        assert_eq!(stats.total_params, 8);
        assert_eq!(stats.active_params, 4);
        assert_abs_diff_eq!(stats.pruning_ratio, 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_pruning_stats() {
        let stats = PruningStats {
            total_params: 1000,
            active_params: 200,
            pruning_ratio: 0.8,
            iterations: 5,
            per_layer_stats: HashMap::new(),
        };

        assert_abs_diff_eq!(stats.compression_ratio(), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.flops_reduction(), 0.8, epsilon = 1e-10);

        let summary = stats.summary();
        assert!(summary.contains("1000"));
        assert!(summary.contains("200"));
        assert!(summary.contains("5.00x"));
    }

    #[test]
    fn test_gradient_pruner_fallback() {
        let weights =
            Array2::from_shape_vec((3, 3), vec![0.1, 0.5, 0.9, 0.2, 0.6, 0.01, 0.3, 0.7, 0.8])
                .unwrap();

        let config = PruningConfig {
            pruning_ratio: 0.3,
            ..Default::default()
        };

        let pruner = GradientPruner::new(config);
        let (_pruned, mask) = pruner.prune(&weights).unwrap();

        // Without gradient history, should fall back to magnitude pruning
        let active_count = mask.iter().filter(|&&m| m > 0.5).count();
        // With 30% pruning ratio, prune_count = (9 * 0.3) = 2.7 -> 2
        // So we keep 9 - 2 = 7 weights
        let prune_count = (9.0 * 0.3) as usize;
        let expected_active = 9 - prune_count;
        assert_eq!(active_count, expected_active);
    }

    #[test]
    fn test_gradient_pruner_with_history() {
        let weights = Array2::from_shape_vec((2, 2), vec![0.5, 0.6, 0.7, 0.8]).unwrap();

        let grads1 = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        let grads2 = Array2::from_shape_vec((2, 2), vec![0.15, 0.25, 0.35, 0.45]).unwrap();

        let config = PruningConfig {
            pruning_ratio: 0.5,
            ..Default::default()
        };

        let mut pruner = GradientPruner::new(config);
        pruner.update_gradients("layer1", grads1);
        pruner.update_gradients("layer1", grads2);

        let (pruned, _mask) = pruner.prune_with_history(&weights, "layer1").unwrap();

        // Weights with smallest average gradients should be pruned
        // Average gradients: [0.125, 0.225, 0.325, 0.425]
        // Should prune the two smallest (0.125, 0.225)
        assert_abs_diff_eq!(pruned[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pruned[[0, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_schedule() {
        let mut pruner = MagnitudePruner::new(PruningConfig {
            pruning_ratio: 0.0,
            iterative: true,
            num_iterations: 5,
            initial_ratio: 0.1,
            final_ratio: 0.9,
            schedule: "exponential".to_string(),
            ..Default::default()
        });

        pruner.update_ratio(0);
        let ratio_0 = pruner.current_ratio;
        pruner.update_ratio(2);
        let ratio_2 = pruner.current_ratio;
        pruner.update_ratio(4);
        let ratio_4 = pruner.current_ratio;

        // Exponential schedule should have larger jumps later
        assert!(ratio_0 < ratio_2);
        assert!(ratio_2 < ratio_4);
        assert_abs_diff_eq!(ratio_0, 0.1, epsilon = 1e-2);
        assert_abs_diff_eq!(ratio_4, 0.9, epsilon = 1e-2);
    }

    #[test]
    fn test_cosine_schedule() {
        let mut pruner = MagnitudePruner::new(PruningConfig {
            pruning_ratio: 0.0,
            iterative: true,
            num_iterations: 5,
            initial_ratio: 0.1,
            final_ratio: 0.9,
            schedule: "cosine".to_string(),
            ..Default::default()
        });

        pruner.update_ratio(0);
        let ratio_0 = pruner.current_ratio;
        pruner.update_ratio(4);
        let ratio_4 = pruner.current_ratio;

        assert_abs_diff_eq!(ratio_0, 0.1, epsilon = 1e-2);
        assert_abs_diff_eq!(ratio_4, 0.9, epsilon = 1e-2);
    }

    #[test]
    fn test_min_threshold() {
        let weights = Array2::from_shape_vec((2, 2), vec![1e-10, 1e-9, 1e-8, 0.5]).unwrap();

        let config = PruningConfig {
            pruning_ratio: 0.0,  // Don't prune by ratio
            min_threshold: 1e-7, // But prune by threshold
            ..Default::default()
        };

        let pruner = MagnitudePruner::new(config);
        let (pruned, _mask) = pruner.prune(&weights).unwrap();

        // All weights below threshold should be pruned
        assert_abs_diff_eq!(pruned[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pruned[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pruned[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pruned[[1, 1]], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_mask() {
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let mask = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 1.0, 0.0])
            .unwrap()
            .into_dyn();

        let config = PruningConfig::default();
        let pruner = MagnitudePruner::new(config);
        let pruned = pruner.apply_mask(&weights, &mask).unwrap();

        assert_abs_diff_eq!(pruned[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pruned[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pruned[[1, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pruned[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_structured_both_axes() {
        let weights = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.9, 0.9, 0.1, 0.9,
            ],
        )
        .unwrap();

        let config = PruningConfig {
            pruning_ratio: 0.25,
            structured: true,
            ..Default::default()
        };

        let pruner = StructuredPruner::new(config, StructuredPruningAxis::Both);
        let (_pruned, _mask) = pruner.prune(&weights).unwrap();

        // Should prune both rows and columns based on L2 norms
        // This is a complex test; we just verify it doesn't panic
    }
}
