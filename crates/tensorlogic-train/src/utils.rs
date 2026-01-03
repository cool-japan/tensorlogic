//! Utility functions for model introspection, training analysis, and debugging.
//!
//! This module provides tools for:
//! - Model parameter analysis and visualization
//! - Gradient statistics computation
//! - Training time estimation
//! - Model summary generation

use crate::error::{TrainError, TrainResult};
use crate::model::Model;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// Model parameter statistics for a single layer or the entire model.
#[derive(Debug, Clone)]
pub struct ParameterStats {
    /// Number of parameters
    pub count: usize,
    /// Mean of parameter values
    pub mean: f64,
    /// Standard deviation of parameter values
    pub std: f64,
    /// Minimum parameter value
    pub min: f64,
    /// Maximum parameter value
    pub max: f64,
    /// Percentage of zero parameters
    pub sparsity: f64,
}

impl ParameterStats {
    /// Compute statistics from a parameter array.
    pub fn from_array(params: &Array1<f64>) -> Self {
        let count = params.len();
        if count == 0 {
            return Self {
                count: 0,
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                sparsity: 0.0,
            };
        }

        let mean = params.mean().unwrap_or(0.0);
        let variance = params.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std = variance.sqrt();

        let min = params.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = params.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let zeros = params.iter().filter(|&&x| x.abs() < 1e-10).count();
        let sparsity = zeros as f64 / count as f64 * 100.0;

        Self {
            count,
            mean,
            std,
            min,
            max,
            sparsity,
        }
    }

    /// Pretty print the statistics.
    pub fn summary(&self) -> String {
        format!(
            "Parameters: {}\n\
             Mean: {:.6}, Std: {:.6}\n\
             Min: {:.6}, Max: {:.6}\n\
             Sparsity: {:.2}%",
            self.count, self.mean, self.std, self.min, self.max, self.sparsity
        )
    }
}

/// Model summary containing layer-wise parameter information.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    /// Total number of parameters
    pub total_params: usize,
    /// Trainable parameters
    pub trainable_params: usize,
    /// Layer-wise statistics
    pub layer_stats: HashMap<String, ParameterStats>,
    /// Overall model statistics
    pub overall_stats: ParameterStats,
}

impl ModelSummary {
    /// Generate a model summary from a model's state dict.
    pub fn from_model<M: Model>(model: &M) -> TrainResult<Self> {
        let state_dict = model.state_dict();
        let mut total_params = 0;
        let mut layer_stats = HashMap::new();
        let mut all_params = Vec::new();

        for (name, params) in state_dict.iter() {
            total_params += params.len();
            all_params.extend(params.iter());
            let params_array = Array1::from_vec(params.clone());
            layer_stats.insert(name.clone(), ParameterStats::from_array(&params_array));
        }

        let overall_stats = ParameterStats::from_array(&Array1::from_vec(all_params));
        let trainable_params = total_params; // Assuming all params are trainable by default

        Ok(Self {
            total_params,
            trainable_params,
            layer_stats,
            overall_stats,
        })
    }

    /// Print a formatted summary of the model.
    pub fn print(&self) {
        println!("=================================================================");
        println!("Model Summary");
        println!("=================================================================");
        println!("Total Parameters: {}", self.total_params);
        println!("Trainable Parameters: {}", self.trainable_params);
        println!("-----------------------------------------------------------------");
        println!("Overall Statistics:");
        println!("{}", self.overall_stats.summary());
        println!("-----------------------------------------------------------------");
        println!("Layer-wise Statistics:");
        for (name, stats) in &self.layer_stats {
            println!("\n{}: {} parameters", name, stats.count);
            println!("  Mean: {:.6}, Std: {:.6}", stats.mean, stats.std);
            println!("  Range: [{:.6}, {:.6}]", stats.min, stats.max);
            if stats.sparsity > 0.0 {
                println!("  Sparsity: {:.2}%", stats.sparsity);
            }
        }
        println!("=================================================================");
    }
}

/// Gradient statistics for monitoring gradient flow.
#[derive(Debug, Clone)]
pub struct GradientStats {
    /// Layer name
    pub layer_name: String,
    /// L2 norm of gradients
    pub norm: f64,
    /// Mean gradient value
    pub mean: f64,
    /// Standard deviation of gradients
    pub std: f64,
    /// Maximum absolute gradient
    pub max_abs: f64,
}

impl GradientStats {
    /// Compute gradient statistics from a gradient array.
    pub fn compute(layer_name: String, grads: &Array1<f64>) -> Self {
        let norm = grads.iter().map(|&g| g * g).sum::<f64>().sqrt();
        let mean = grads.mean().unwrap_or(0.0);
        let variance = grads.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / grads.len() as f64;
        let std = variance.sqrt();
        let max_abs = grads.iter().map(|&g| g.abs()).fold(0.0, f64::max);

        Self {
            layer_name,
            norm,
            mean,
            std,
            max_abs,
        }
    }

    /// Check if gradients are vanishing (too small).
    pub fn is_vanishing(&self, threshold: f64) -> bool {
        self.norm < threshold
    }

    /// Check if gradients are exploding (too large).
    pub fn is_exploding(&self, threshold: f64) -> bool {
        self.norm > threshold
    }
}

/// Compute gradient statistics for all layers in a gradient dictionary.
pub fn compute_gradient_stats(gradients: &HashMap<String, Array1<f64>>) -> Vec<GradientStats> {
    gradients
        .iter()
        .map(|(name, grads)| GradientStats::compute(name.clone(), grads))
        .collect()
}

/// Print a formatted report of gradient statistics.
pub fn print_gradient_report(stats: &[GradientStats]) {
    println!("=================================================================");
    println!("Gradient Statistics");
    println!("=================================================================");
    for stat in stats {
        println!("Layer: {}", stat.layer_name);
        println!("  Norm: {:.6}", stat.norm);
        println!("  Mean: {:.6}, Std: {:.6}", stat.mean, stat.std);
        println!("  Max(abs): {:.6}", stat.max_abs);

        if stat.is_vanishing(1e-7) {
            println!("  ⚠️  WARNING: Vanishing gradients detected!");
        }
        if stat.is_exploding(1e3) {
            println!("  ⚠️  WARNING: Exploding gradients detected!");
        }
        println!();
    }
    println!("=================================================================");
}

/// Training time estimation based on iteration timing.
#[derive(Debug, Clone)]
pub struct TimeEstimator {
    /// Number of samples processed so far
    samples_processed: usize,
    /// Total time elapsed (in seconds)
    time_elapsed: f64,
    /// Total number of samples to process
    total_samples: usize,
}

impl TimeEstimator {
    /// Create a new time estimator.
    pub fn new(total_samples: usize) -> Self {
        Self {
            samples_processed: 0,
            time_elapsed: 0.0,
            total_samples,
        }
    }

    /// Update with the number of samples processed in this iteration and time taken.
    pub fn update(&mut self, samples: usize, time_seconds: f64) {
        self.samples_processed += samples;
        self.time_elapsed += time_seconds;
    }

    /// Get the current throughput (samples per second).
    pub fn throughput(&self) -> f64 {
        if self.time_elapsed > 0.0 {
            self.samples_processed as f64 / self.time_elapsed
        } else {
            0.0
        }
    }

    /// Estimate remaining time in seconds.
    pub fn remaining_time(&self) -> f64 {
        let throughput = self.throughput();
        if throughput > 0.0 {
            let remaining_samples = self.total_samples.saturating_sub(self.samples_processed);
            remaining_samples as f64 / throughput
        } else {
            0.0
        }
    }

    /// Format remaining time as a human-readable string.
    pub fn remaining_time_formatted(&self) -> String {
        let seconds = self.remaining_time();
        format_duration(seconds)
    }

    /// Get progress percentage.
    pub fn progress(&self) -> f64 {
        if self.total_samples > 0 {
            (self.samples_processed as f64 / self.total_samples as f64 * 100.0).min(100.0)
        } else {
            0.0
        }
    }
}

/// Format a duration in seconds to a human-readable string.
pub fn format_duration(seconds: f64) -> String {
    let total_seconds = seconds as u64;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let secs = total_seconds % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

/// Compare two models and report differences in parameters.
pub fn compare_models<M: Model>(
    model1: &M,
    model2: &M,
) -> TrainResult<HashMap<String, ParameterDifference>> {
    let state1 = model1.state_dict();
    let state2 = model2.state_dict();

    let mut differences = HashMap::new();

    for (name, params1) in state1.iter() {
        if let Some(params2) = state2.get(name) {
            if params1.len() != params2.len() {
                return Err(TrainError::ModelError(format!(
                    "Parameter size mismatch for layer '{}': {} vs {}",
                    name,
                    params1.len(),
                    params2.len()
                )));
            }

            let params1_array = Array1::from_vec(params1.clone());
            let params2_array = Array1::from_vec(params2.clone());
            let diff = ParameterDifference::compute(&params1_array, &params2_array);
            differences.insert(name.clone(), diff);
        } else {
            return Err(TrainError::ModelError(format!(
                "Layer '{}' not found in second model",
                name
            )));
        }
    }

    Ok(differences)
}

/// Statistics about parameter differences between two models.
#[derive(Debug, Clone)]
pub struct ParameterDifference {
    /// Mean absolute difference
    pub mean_abs_diff: f64,
    /// Maximum absolute difference
    pub max_abs_diff: f64,
    /// Relative change (mean abs diff / mean abs value)
    pub relative_change: f64,
    /// Cosine similarity between parameter vectors
    pub cosine_similarity: f64,
}

impl ParameterDifference {
    /// Compute parameter difference statistics.
    pub fn compute(params1: &Array1<f64>, params2: &Array1<f64>) -> Self {
        let diff: Array1<f64> = params1 - params2;
        let abs_diff = diff.mapv(f64::abs);

        let mean_abs_diff = abs_diff.mean().unwrap_or(0.0);
        let max_abs_diff = abs_diff.iter().cloned().fold(0.0, f64::max);

        let mean_abs_value = params1.mapv(f64::abs).mean().unwrap_or(1.0);
        let relative_change = if mean_abs_value > 0.0 {
            mean_abs_diff / mean_abs_value
        } else {
            0.0
        };

        // Cosine similarity
        let dot_product = params1
            .iter()
            .zip(params2.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>();
        let norm1 = params1.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let norm2 = params2.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let cosine_similarity = if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        };

        Self {
            mean_abs_diff,
            max_abs_diff,
            relative_change,
            cosine_similarity,
        }
    }
}

/// Learning rate range test analyzer for finding optimal learning rates.
#[derive(Debug, Clone)]
pub struct LrRangeTestAnalyzer {
    /// Learning rates tested
    pub learning_rates: Vec<f64>,
    /// Losses observed at each learning rate
    pub losses: Vec<f64>,
}

impl LrRangeTestAnalyzer {
    /// Create a new analyzer.
    pub fn new(learning_rates: Vec<f64>, losses: Vec<f64>) -> TrainResult<Self> {
        if learning_rates.len() != losses.len() {
            return Err(TrainError::ConfigError(
                "Learning rates and losses must have the same length".to_string(),
            ));
        }

        Ok(Self {
            learning_rates,
            losses,
        })
    }

    /// Find the learning rate with the steepest loss decrease.
    pub fn suggest_lr(&self) -> Option<f64> {
        if self.losses.len() < 2 {
            return None;
        }

        // Compute gradients (rate of loss change)
        let mut max_gradient = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for i in 1..self.losses.len() {
            let gradient = (self.losses[i - 1] - self.losses[i])
                / (self.learning_rates[i] - self.learning_rates[i - 1]).abs();

            if gradient > max_gradient {
                max_gradient = gradient;
                best_idx = i;
            }
        }

        Some(self.learning_rates[best_idx])
    }

    /// Find the learning rate at minimum loss.
    pub fn lr_at_min_loss(&self) -> Option<f64> {
        self.losses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| self.learning_rates[idx])
    }

    /// Plot the LR range test results (returns a simple ASCII plot).
    pub fn plot_ascii(&self, width: usize, height: usize) -> String {
        if self.losses.is_empty() {
            return "No data to plot".to_string();
        }

        let min_loss = self.losses.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_loss = self
            .losses
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let loss_range = max_loss - min_loss;

        let mut plot = vec![vec![' '; width]; height];

        // Plot points
        for (i, &loss) in self.losses.iter().enumerate() {
            let x = (i * width) / self.losses.len().max(1);
            let normalized = if loss_range > 0.0 {
                (max_loss - loss) / loss_range
            } else {
                0.5
            };
            let y = ((normalized * (height - 1) as f64) as usize).min(height - 1);

            if x < width && y < height {
                plot[y][x] = '*';
            }
        }

        // Convert to string
        let mut result = String::new();
        result.push_str(&format!(
            "Learning Rate Range Test (Loss: {:.4} - {:.4})\n",
            min_loss, max_loss
        ));
        result.push_str(&format!(
            "Suggested LR: {:.2e}\n\n",
            self.suggest_lr().unwrap_or(0.0)
        ));

        for row in plot {
            result.push_str(&row.iter().collect::<String>());
            result.push('\n');
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_parameter_stats() {
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = ParameterStats::from_array(&params);

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!(stats.std > 0.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_parameter_stats_with_zeros() {
        let params = Array1::from_vec(vec![0.0, 0.0, 1.0, 2.0]);
        let stats = ParameterStats::from_array(&params);

        assert_eq!(stats.count, 4);
        assert_eq!(stats.sparsity, 50.0); // 2 out of 4 are zeros
    }

    #[test]
    fn test_gradient_stats() {
        let grads = Array1::from_vec(vec![0.1, 0.2, -0.1, 0.3]);
        let stats = GradientStats::compute("test_layer".to_string(), &grads);

        assert_eq!(stats.layer_name, "test_layer");
        assert!(stats.norm > 0.0);
        assert!(!stats.is_vanishing(1e-8));
        assert!(!stats.is_exploding(1e3));
    }

    #[test]
    fn test_gradient_stats_vanishing() {
        let grads = Array1::from_vec(vec![1e-10, 1e-9, -1e-10]);
        let stats = GradientStats::compute("vanishing".to_string(), &grads);

        assert!(stats.is_vanishing(1e-7));
    }

    #[test]
    fn test_gradient_stats_exploding() {
        let grads = Array1::from_vec(vec![1e5, 1e6, -1e5]);
        let stats = GradientStats::compute("exploding".to_string(), &grads);

        assert!(stats.is_exploding(1e3));
    }

    #[test]
    fn test_time_estimator() {
        let mut estimator = TimeEstimator::new(1000);

        estimator.update(100, 10.0); // 10 seconds for 100 samples
        assert!((estimator.throughput() - 10.0).abs() < 0.1); // 10 samples/sec
        assert!((estimator.progress() - 10.0).abs() < 0.1); // 10% progress

        let remaining = estimator.remaining_time();
        assert!((remaining - 90.0).abs() < 1.0); // ~90 seconds remaining
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3665.0), "1h 1m 5s");
    }

    #[test]
    fn test_parameter_difference() {
        let params1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let params2 = Array1::from_vec(vec![1.1, 2.1, 3.1]);

        let diff = ParameterDifference::compute(&params1, &params2);

        assert!((diff.mean_abs_diff - 0.1).abs() < 1e-6);
        assert!((diff.max_abs_diff - 0.1).abs() < 1e-6);
        assert!(diff.cosine_similarity > 0.99); // Very similar vectors
    }

    #[test]
    fn test_lr_range_test_analyzer() {
        let lrs = vec![1e-4, 1e-3, 1e-2, 1e-1];
        let losses = vec![1.0, 0.5, 0.3, 0.8]; // Min at 1e-2

        let analyzer = LrRangeTestAnalyzer::new(lrs.clone(), losses).unwrap();

        let min_lr = analyzer.lr_at_min_loss();
        assert_eq!(min_lr, Some(1e-2));

        let suggested = analyzer.suggest_lr();
        assert!(suggested.is_some());
    }

    #[test]
    fn test_lr_range_test_analyzer_invalid() {
        let lrs = vec![1e-4, 1e-3];
        let losses = vec![1.0]; // Mismatched length

        let result = LrRangeTestAnalyzer::new(lrs, losses);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_gradient_stats() {
        let mut gradients = HashMap::new();
        gradients.insert("layer1".to_string(), Array1::from_vec(vec![0.1, 0.2, 0.3]));
        gradients.insert("layer2".to_string(), Array1::from_vec(vec![1e-10, 1e-9]));

        let stats = compute_gradient_stats(&gradients);
        assert_eq!(stats.len(), 2);

        // Find the vanishing layer
        let vanishing = stats.iter().find(|s| s.layer_name == "layer2").unwrap();
        assert!(vanishing.is_vanishing(1e-7));
    }
}
