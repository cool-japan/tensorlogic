//! Online kernel updates for streaming and incremental learning.
//!
//! This module provides kernel matrices that can be efficiently updated
//! incrementally as new samples arrive, without recomputing the entire matrix.
//!
//! ## Features
//!
//! - **OnlineKernelMatrix** - Incrementally add samples with O(n) updates
//! - **WindowedKernelMatrix** - Sliding window for bounded memory in time series
//! - **ForgetfulKernelMatrix** - Exponential decay for concept drift adaptation
//!
//! ## Use Cases
//!
//! - Streaming data classification
//! - Online learning with kernel methods
//! - Time series with non-stationarity
//! - Memory-constrained environments

use crate::error::{KernelError, Result};
use crate::types::Kernel;
use std::collections::VecDeque;
use std::sync::Arc;

/// Configuration for online kernel matrix updates.
#[derive(Debug, Clone)]
pub struct OnlineConfig {
    /// Initial capacity for samples
    pub initial_capacity: usize,
    /// Whether to compute full matrix or just needed entries
    pub compute_full_matrix: bool,
}

impl Default for OnlineConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 64,
            compute_full_matrix: true,
        }
    }
}

impl OnlineConfig {
    /// Create a new configuration with specified initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            initial_capacity: capacity,
            ..Default::default()
        }
    }
}

/// Statistics for online kernel updates.
#[derive(Debug, Clone, Default)]
pub struct OnlineStats {
    /// Number of samples added
    pub samples_added: usize,
    /// Number of samples removed (for windowed)
    pub samples_removed: usize,
    /// Total kernel computations performed
    pub kernel_computations: usize,
    /// Number of matrix resizes
    pub resizes: usize,
}

/// Incrementally updatable kernel matrix.
///
/// Supports efficient O(n) updates when adding new samples,
/// avoiding O(n²) recomputation of the entire matrix.
///
/// # Example
///
/// ```
/// use tensorlogic_sklears_kernels::{OnlineKernelMatrix, RbfKernel, RbfKernelConfig, Kernel};
///
/// let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
/// let mut online = OnlineKernelMatrix::new(Box::new(kernel));
///
/// // Add samples incrementally
/// online.add_sample(vec![1.0, 2.0, 3.0]).unwrap();
/// online.add_sample(vec![4.0, 5.0, 6.0]).unwrap();
/// online.add_sample(vec![7.0, 8.0, 9.0]).unwrap();
///
/// // Get the kernel matrix
/// let matrix = online.get_matrix();
/// assert_eq!(matrix.len(), 3);
/// ```
pub struct OnlineKernelMatrix {
    /// The underlying kernel function
    kernel: Box<dyn Kernel>,
    /// Stored samples
    samples: Vec<Vec<f64>>,
    /// Current kernel matrix (upper triangular stored as full for simplicity)
    matrix: Vec<Vec<f64>>,
    /// Configuration
    config: OnlineConfig,
    /// Statistics
    stats: OnlineStats,
}

impl OnlineKernelMatrix {
    /// Create a new online kernel matrix.
    pub fn new(kernel: Box<dyn Kernel>) -> Self {
        Self::with_config(kernel, OnlineConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(kernel: Box<dyn Kernel>, config: OnlineConfig) -> Self {
        Self {
            kernel,
            samples: Vec::with_capacity(config.initial_capacity),
            matrix: Vec::with_capacity(config.initial_capacity),
            config,
            stats: OnlineStats::default(),
        }
    }

    /// Add a new sample and update the kernel matrix.
    ///
    /// Time complexity: O(n) where n is current number of samples.
    pub fn add_sample(&mut self, sample: Vec<f64>) -> Result<()> {
        // Validate dimensions
        if let Some(first) = self.samples.first() {
            if sample.len() != first.len() {
                return Err(KernelError::DimensionMismatch {
                    expected: vec![first.len()],
                    got: vec![sample.len()],
                    context: "online kernel matrix".to_string(),
                });
            }
        }

        let n = self.samples.len();

        // Compute kernel values between new sample and all existing samples
        let mut new_row = Vec::with_capacity(n + 1);
        for existing in &self.samples {
            let k = self.kernel.compute(&sample, existing)?;
            new_row.push(k);
            self.stats.kernel_computations += 1;
        }

        // Self-similarity (usually 1.0 for normalized kernels)
        let k_self = self.kernel.compute(&sample, &sample)?;
        new_row.push(k_self);
        self.stats.kernel_computations += 1;

        // Update existing rows with new column
        for (i, row) in self.matrix.iter_mut().enumerate() {
            row.push(new_row[i]);
        }

        // Add new row
        self.matrix.push(new_row);
        self.samples.push(sample);
        self.stats.samples_added += 1;

        Ok(())
    }

    /// Add multiple samples at once (batch update).
    ///
    /// More efficient than adding one by one due to better cache utilization.
    pub fn add_samples(&mut self, samples: Vec<Vec<f64>>) -> Result<()> {
        for sample in samples {
            self.add_sample(sample)?;
        }
        Ok(())
    }

    /// Remove a sample by index and update the matrix.
    ///
    /// Time complexity: O(n²) due to matrix restructuring.
    pub fn remove_sample(&mut self, index: usize) -> Result<Vec<f64>> {
        if index >= self.samples.len() {
            return Err(KernelError::ComputationError(format!(
                "Index {} out of bounds for {} samples",
                index,
                self.samples.len()
            )));
        }

        // Remove from samples
        let removed = self.samples.remove(index);

        // Remove row
        self.matrix.remove(index);

        // Remove column from all remaining rows
        for row in &mut self.matrix {
            row.remove(index);
        }

        self.stats.samples_removed += 1;
        Ok(removed)
    }

    /// Get the current kernel matrix.
    pub fn get_matrix(&self) -> &Vec<Vec<f64>> {
        &self.matrix
    }

    /// Get the current samples.
    pub fn get_samples(&self) -> &Vec<Vec<f64>> {
        &self.samples
    }

    /// Get a specific kernel value.
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        self.matrix.get(i).and_then(|row| row.get(j).copied())
    }

    /// Get the number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get statistics.
    pub fn stats(&self) -> &OnlineStats {
        &self.stats
    }

    /// Reset the matrix.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.matrix.clear();
        self.stats = OnlineStats::default();
    }

    /// Get the underlying kernel.
    pub fn kernel(&self) -> &dyn Kernel {
        self.kernel.as_ref()
    }

    /// Get the configuration.
    pub fn config(&self) -> &OnlineConfig {
        &self.config
    }

    /// Compute kernel value between a query point and stored sample.
    pub fn compute_with_sample(&self, query: &[f64], sample_idx: usize) -> Result<f64> {
        let sample = self.samples.get(sample_idx).ok_or_else(|| {
            KernelError::ComputationError(format!("Sample index {} not found", sample_idx))
        })?;
        self.kernel.compute(query, sample)
    }

    /// Compute kernel values between query and all stored samples.
    pub fn compute_with_all(&self, query: &[f64]) -> Result<Vec<f64>> {
        let mut result = Vec::with_capacity(self.samples.len());
        for sample in &self.samples {
            let k = self.kernel.compute(query, sample)?;
            result.push(k);
        }
        Ok(result)
    }

    /// Clone the current matrix as a standalone 2D vector.
    pub fn to_matrix(&self) -> Vec<Vec<f64>> {
        self.matrix.clone()
    }
}

/// Sliding window kernel matrix for bounded-memory streaming.
///
/// Maintains only the most recent `window_size` samples, automatically
/// removing oldest samples when the window is full.
///
/// # Example
///
/// ```
/// use tensorlogic_sklears_kernels::{WindowedKernelMatrix, LinearKernel, Kernel};
///
/// let kernel = LinearKernel::new();
/// let mut windowed = WindowedKernelMatrix::new(Box::new(kernel), 3);
///
/// // Add samples (window size = 3)
/// windowed.add_sample(vec![1.0]).unwrap();
/// windowed.add_sample(vec![2.0]).unwrap();
/// windowed.add_sample(vec![3.0]).unwrap();
/// windowed.add_sample(vec![4.0]).unwrap(); // First sample evicted
///
/// assert_eq!(windowed.len(), 3);
/// ```
pub struct WindowedKernelMatrix {
    /// The underlying kernel function
    kernel: Box<dyn Kernel>,
    /// Window size
    window_size: usize,
    /// Stored samples in window (circular buffer)
    samples: VecDeque<Vec<f64>>,
    /// Current kernel matrix
    matrix: Vec<Vec<f64>>,
    /// Statistics
    stats: OnlineStats,
}

impl WindowedKernelMatrix {
    /// Create a new windowed kernel matrix.
    pub fn new(kernel: Box<dyn Kernel>, window_size: usize) -> Self {
        assert!(window_size > 0, "Window size must be positive");
        Self {
            kernel,
            window_size,
            samples: VecDeque::with_capacity(window_size),
            matrix: Vec::with_capacity(window_size),
            stats: OnlineStats::default(),
        }
    }

    /// Add a sample, evicting oldest if window is full.
    pub fn add_sample(&mut self, sample: Vec<f64>) -> Result<Option<Vec<f64>>> {
        // Validate dimensions
        if let Some(first) = self.samples.front() {
            if sample.len() != first.len() {
                return Err(KernelError::DimensionMismatch {
                    expected: vec![first.len()],
                    got: vec![sample.len()],
                    context: "windowed kernel matrix".to_string(),
                });
            }
        }

        let evicted = if self.samples.len() >= self.window_size {
            // Remove oldest sample
            let removed = self.samples.pop_front();

            // Update matrix: remove first row and first column
            self.matrix.remove(0);
            for row in &mut self.matrix {
                row.remove(0);
            }

            self.stats.samples_removed += 1;
            removed
        } else {
            None
        };

        // Compute kernel values with existing samples
        let n = self.samples.len();
        let mut new_row = Vec::with_capacity(n + 1);

        for existing in &self.samples {
            let k = self.kernel.compute(&sample, existing)?;
            new_row.push(k);
            self.stats.kernel_computations += 1;
        }

        // Self-similarity
        let k_self = self.kernel.compute(&sample, &sample)?;
        new_row.push(k_self);
        self.stats.kernel_computations += 1;

        // Update existing rows
        for (i, row) in self.matrix.iter_mut().enumerate() {
            row.push(new_row[i]);
        }

        // Add new row
        self.matrix.push(new_row);
        self.samples.push_back(sample);
        self.stats.samples_added += 1;

        Ok(evicted)
    }

    /// Get the current kernel matrix.
    pub fn get_matrix(&self) -> &Vec<Vec<f64>> {
        &self.matrix
    }

    /// Get samples in the window.
    pub fn get_samples(&self) -> &VecDeque<Vec<f64>> {
        &self.samples
    }

    /// Get the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get current number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if window is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Check if window is full.
    pub fn is_full(&self) -> bool {
        self.samples.len() >= self.window_size
    }

    /// Get statistics.
    pub fn stats(&self) -> &OnlineStats {
        &self.stats
    }

    /// Clear the window.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.matrix.clear();
        self.stats = OnlineStats::default();
    }

    /// Compute kernel values between query and all samples in window.
    pub fn compute_with_all(&self, query: &[f64]) -> Result<Vec<f64>> {
        let mut result = Vec::with_capacity(self.samples.len());
        for sample in &self.samples {
            let k = self.kernel.compute(query, sample)?;
            result.push(k);
        }
        Ok(result)
    }
}

/// Configuration for forgetful kernel matrix.
#[derive(Debug, Clone)]
pub struct ForgetfulConfig {
    /// Forgetting factor (0 < λ <= 1)
    /// λ = 1: no forgetting (infinite memory)
    /// λ < 1: older samples weighted less
    pub lambda: f64,
    /// Threshold below which samples are removed
    pub removal_threshold: Option<f64>,
    /// Maximum number of samples to keep
    pub max_samples: Option<usize>,
}

impl Default for ForgetfulConfig {
    fn default() -> Self {
        Self {
            lambda: 0.99,
            removal_threshold: Some(0.01),
            max_samples: None,
        }
    }
}

impl ForgetfulConfig {
    /// Create configuration with specified forgetting factor.
    pub fn with_lambda(lambda: f64) -> Result<Self> {
        if lambda <= 0.0 || lambda > 1.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "lambda".to_string(),
                value: lambda.to_string(),
                reason: "lambda must be in (0, 1]".to_string(),
            });
        }
        Ok(Self {
            lambda,
            ..Default::default()
        })
    }

    /// Set maximum samples limit.
    pub fn with_max_samples(mut self, max: usize) -> Self {
        self.max_samples = Some(max);
        self
    }

    /// Set removal threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.removal_threshold = Some(threshold);
        self
    }
}

/// Kernel matrix with exponential forgetting for concept drift adaptation.
///
/// Older samples are weighted by λ^age, allowing the model to adapt
/// to changing data distributions.
///
/// # Example
///
/// ```
/// use tensorlogic_sklears_kernels::{ForgetfulKernelMatrix, ForgetfulConfig, RbfKernel, RbfKernelConfig, Kernel};
///
/// let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
/// let config = ForgetfulConfig::with_lambda(0.95).unwrap();
/// let mut forgetful = ForgetfulKernelMatrix::new(Box::new(kernel), config);
///
/// // Add samples (older ones get downweighted)
/// forgetful.add_sample(vec![1.0, 2.0]).unwrap();
/// forgetful.add_sample(vec![3.0, 4.0]).unwrap();
/// forgetful.add_sample(vec![5.0, 6.0]).unwrap();
///
/// // Get weighted kernel matrix
/// let weighted = forgetful.get_weighted_matrix();
/// ```
pub struct ForgetfulKernelMatrix {
    /// The underlying kernel function
    kernel: Box<dyn Kernel>,
    /// Configuration
    config: ForgetfulConfig,
    /// Stored samples
    samples: Vec<Vec<f64>>,
    /// Sample weights (λ^age)
    weights: Vec<f64>,
    /// Raw kernel matrix (unweighted)
    matrix: Vec<Vec<f64>>,
    /// Statistics
    stats: OnlineStats,
}

impl ForgetfulKernelMatrix {
    /// Create a new forgetful kernel matrix.
    pub fn new(kernel: Box<dyn Kernel>, config: ForgetfulConfig) -> Self {
        Self {
            kernel,
            config,
            samples: Vec::new(),
            weights: Vec::new(),
            matrix: Vec::new(),
            stats: OnlineStats::default(),
        }
    }

    /// Add a sample with forgetting.
    pub fn add_sample(&mut self, sample: Vec<f64>) -> Result<()> {
        // Validate dimensions
        if let Some(first) = self.samples.first() {
            if sample.len() != first.len() {
                return Err(KernelError::DimensionMismatch {
                    expected: vec![first.len()],
                    got: vec![sample.len()],
                    context: "forgetful kernel matrix".to_string(),
                });
            }
        }

        // Age existing samples (multiply weights by lambda)
        for weight in &mut self.weights {
            *weight *= self.config.lambda;
        }

        // Remove samples below threshold
        if let Some(threshold) = self.config.removal_threshold {
            let mut i = 0;
            while i < self.weights.len() {
                if self.weights[i] < threshold {
                    self.remove_at(i);
                } else {
                    i += 1;
                }
            }
        }

        // Enforce max samples
        if let Some(max) = self.config.max_samples {
            while self.samples.len() >= max && !self.samples.is_empty() {
                // Remove the lowest weight sample
                if let Some((min_idx, _)) = self
                    .weights
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                {
                    self.remove_at(min_idx);
                }
            }
        }

        // Compute kernel values with existing samples
        let n = self.samples.len();
        let mut new_row = Vec::with_capacity(n + 1);

        for existing in &self.samples {
            let k = self.kernel.compute(&sample, existing)?;
            new_row.push(k);
            self.stats.kernel_computations += 1;
        }

        // Self-similarity
        let k_self = self.kernel.compute(&sample, &sample)?;
        new_row.push(k_self);
        self.stats.kernel_computations += 1;

        // Update existing rows
        for (i, row) in self.matrix.iter_mut().enumerate() {
            row.push(new_row[i]);
        }

        // Add new row and sample
        self.matrix.push(new_row);
        self.samples.push(sample);
        self.weights.push(1.0); // New sample has full weight
        self.stats.samples_added += 1;

        Ok(())
    }

    /// Remove sample at specific index.
    fn remove_at(&mut self, index: usize) {
        self.samples.remove(index);
        self.weights.remove(index);
        self.matrix.remove(index);
        for row in &mut self.matrix {
            row.remove(index);
        }
        self.stats.samples_removed += 1;
    }

    /// Get the raw (unweighted) kernel matrix.
    pub fn get_matrix(&self) -> &Vec<Vec<f64>> {
        &self.matrix
    }

    /// Get the weighted kernel matrix.
    ///
    /// Each entry `K[i,j]` is multiplied by `sqrt(w_i * w_j)` to maintain PSD property.
    pub fn get_weighted_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.matrix.len();
        let mut weighted = vec![vec![0.0; n]; n];

        for (i, (row, &weight_i)) in self.matrix.iter().zip(&self.weights).enumerate() {
            let sqrt_wi = weight_i.sqrt();
            for (j, (&k_val, &weight_j)) in row.iter().zip(&self.weights).enumerate() {
                let sqrt_wj = weight_j.sqrt();
                weighted[i][j] = k_val * sqrt_wi * sqrt_wj;
            }
        }

        weighted
    }

    /// Get sample weights.
    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }

    /// Get the current samples.
    pub fn get_samples(&self) -> &Vec<Vec<f64>> {
        &self.samples
    }

    /// Get the number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get statistics.
    pub fn stats(&self) -> &OnlineStats {
        &self.stats
    }

    /// Get the forgetting factor.
    pub fn lambda(&self) -> f64 {
        self.config.lambda
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.weights.clear();
        self.matrix.clear();
        self.stats = OnlineStats::default();
    }

    /// Compute weighted kernel values between query and all stored samples.
    pub fn compute_weighted(&self, query: &[f64]) -> Result<Vec<f64>> {
        let mut result = Vec::with_capacity(self.samples.len());
        for (sample, weight) in self.samples.iter().zip(&self.weights) {
            let k = self.kernel.compute(query, sample)?;
            result.push(k * weight.sqrt());
        }
        Ok(result)
    }

    /// Get effective sample size (sum of weights).
    pub fn effective_size(&self) -> f64 {
        self.weights.iter().sum()
    }
}

/// Adaptive kernel with automatic bandwidth adjustment.
///
/// Updates kernel parameters based on incoming data statistics.
pub struct AdaptiveKernelMatrix {
    /// Base kernel (must be RBF-like with adjustable bandwidth)
    kernel: Arc<dyn Fn(f64) -> Box<dyn Kernel + Send + Sync> + Send + Sync>,
    /// Current bandwidth parameter
    current_bandwidth: f64,
    /// Online mean of pairwise distances
    distance_sum: f64,
    /// Count of distance observations
    distance_count: usize,
    /// Inner online matrix
    inner: OnlineKernelMatrix,
    /// Adaptation rate
    adaptation_rate: f64,
}

impl AdaptiveKernelMatrix {
    /// Create adaptive kernel with bandwidth factory function.
    pub fn new<F>(kernel_factory: F, initial_bandwidth: f64, adaptation_rate: f64) -> Self
    where
        F: Fn(f64) -> Box<dyn Kernel + Send + Sync> + Send + Sync + 'static,
    {
        let factory = Arc::new(kernel_factory);
        let kernel = factory(initial_bandwidth);

        Self {
            kernel: factory,
            current_bandwidth: initial_bandwidth,
            distance_sum: 0.0,
            distance_count: 0,
            inner: OnlineKernelMatrix::new(kernel),
            adaptation_rate,
        }
    }

    /// Add sample with adaptive bandwidth update.
    pub fn add_sample(&mut self, sample: Vec<f64>) -> Result<()> {
        // Compute distances to existing samples for bandwidth adaptation
        for existing in self.inner.get_samples() {
            let dist_sq: f64 = sample
                .iter()
                .zip(existing.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let dist = dist_sq.sqrt();
            self.distance_sum += dist;
            self.distance_count += 1;
        }

        // Update bandwidth using median heuristic approximation
        if self.distance_count > 0 {
            let mean_dist = self.distance_sum / self.distance_count as f64;
            let new_bandwidth = mean_dist / 2.0_f64.sqrt();

            // Exponential moving average update
            self.current_bandwidth = (1.0 - self.adaptation_rate) * self.current_bandwidth
                + self.adaptation_rate * new_bandwidth;

            // Rebuild kernel with new bandwidth
            let new_kernel = (self.kernel)(self.current_bandwidth);

            // Rebuild matrix with new kernel (expensive but necessary for adaptation)
            let samples: Vec<Vec<f64>> = self.inner.get_samples().clone();
            self.inner = OnlineKernelMatrix::new(new_kernel);
            for s in samples {
                self.inner.add_sample(s)?;
            }
        }

        self.inner.add_sample(sample)
    }

    /// Get current bandwidth.
    pub fn bandwidth(&self) -> f64 {
        self.current_bandwidth
    }

    /// Get the underlying matrix.
    pub fn get_matrix(&self) -> &Vec<Vec<f64>> {
        self.inner.get_matrix()
    }

    /// Get number of samples.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::{LinearKernel, RbfKernel, RbfKernelConfig};

    // ===== OnlineKernelMatrix Tests =====

    #[test]
    fn test_online_kernel_matrix_basic() {
        let kernel = LinearKernel::new();
        let mut online = OnlineKernelMatrix::new(Box::new(kernel));

        assert!(online.is_empty());

        online.add_sample(vec![1.0, 2.0]).unwrap();
        assert_eq!(online.len(), 1);

        online.add_sample(vec![3.0, 4.0]).unwrap();
        assert_eq!(online.len(), 2);

        let matrix = online.get_matrix();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
    }

    #[test]
    fn test_online_kernel_matrix_values() {
        let kernel = LinearKernel::new();
        let mut online = OnlineKernelMatrix::new(Box::new(kernel));

        online.add_sample(vec![1.0, 0.0]).unwrap();
        online.add_sample(vec![0.0, 1.0]).unwrap();

        let matrix = online.get_matrix();

        // K[0,0] = [1,0]·[1,0] = 1
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        // K[1,1] = [0,1]·[0,1] = 1
        assert!((matrix[1][1] - 1.0).abs() < 1e-10);
        // K[0,1] = [1,0]·[0,1] = 0
        assert!((matrix[0][1]).abs() < 1e-10);
        // K[1,0] = K[0,1]
        assert!((matrix[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_online_kernel_matrix_symmetry() {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let mut online = OnlineKernelMatrix::new(Box::new(kernel));

        online.add_sample(vec![1.0, 2.0, 3.0]).unwrap();
        online.add_sample(vec![4.0, 5.0, 6.0]).unwrap();
        online.add_sample(vec![7.0, 8.0, 9.0]).unwrap();

        let matrix = online.get_matrix();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 1e-10,
                    "Matrix not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_online_kernel_matrix_remove() {
        let kernel = LinearKernel::new();
        let mut online = OnlineKernelMatrix::new(Box::new(kernel));

        online.add_sample(vec![1.0]).unwrap();
        online.add_sample(vec![2.0]).unwrap();
        online.add_sample(vec![3.0]).unwrap();

        let removed = online.remove_sample(1).unwrap();
        assert_eq!(removed, vec![2.0]);
        assert_eq!(online.len(), 2);

        let matrix = online.get_matrix();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
    }

    #[test]
    fn test_online_kernel_matrix_dimension_mismatch() {
        let kernel = LinearKernel::new();
        let mut online = OnlineKernelMatrix::new(Box::new(kernel));

        online.add_sample(vec![1.0, 2.0]).unwrap();
        let result = online.add_sample(vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_online_kernel_matrix_compute_with_all() {
        let kernel = LinearKernel::new();
        let mut online = OnlineKernelMatrix::new(Box::new(kernel));

        online.add_sample(vec![1.0, 0.0]).unwrap();
        online.add_sample(vec![0.0, 1.0]).unwrap();

        let query = vec![1.0, 1.0];
        let result = online.compute_with_all(&query).unwrap();

        // [1,1]·[1,0] = 1
        assert!((result[0] - 1.0).abs() < 1e-10);
        // [1,1]·[0,1] = 1
        assert!((result[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_online_kernel_matrix_stats() {
        let kernel = LinearKernel::new();
        let mut online = OnlineKernelMatrix::new(Box::new(kernel));

        online.add_sample(vec![1.0]).unwrap();
        online.add_sample(vec![2.0]).unwrap();
        online.add_sample(vec![3.0]).unwrap();

        let stats = online.stats();
        assert_eq!(stats.samples_added, 3);
        // 1 + 2 + 3 = 6 kernel computations (including self)
        assert_eq!(stats.kernel_computations, 6);
    }

    // ===== WindowedKernelMatrix Tests =====

    #[test]
    fn test_windowed_kernel_matrix_basic() {
        let kernel = LinearKernel::new();
        let mut windowed = WindowedKernelMatrix::new(Box::new(kernel), 3);

        assert_eq!(windowed.window_size(), 3);
        assert!(!windowed.is_full());

        windowed.add_sample(vec![1.0]).unwrap();
        windowed.add_sample(vec![2.0]).unwrap();
        windowed.add_sample(vec![3.0]).unwrap();

        assert!(windowed.is_full());
        assert_eq!(windowed.len(), 3);
    }

    #[test]
    fn test_windowed_kernel_matrix_eviction() {
        let kernel = LinearKernel::new();
        let mut windowed = WindowedKernelMatrix::new(Box::new(kernel), 2);

        windowed.add_sample(vec![1.0]).unwrap();
        windowed.add_sample(vec![2.0]).unwrap();

        // This should evict [1.0]
        let evicted = windowed.add_sample(vec![3.0]).unwrap();
        assert_eq!(evicted, Some(vec![1.0]));
        assert_eq!(windowed.len(), 2);

        // Check samples
        let samples: Vec<_> = windowed.get_samples().iter().cloned().collect();
        assert_eq!(samples, vec![vec![2.0], vec![3.0]]);
    }

    #[test]
    fn test_windowed_kernel_matrix_values() {
        let kernel = LinearKernel::new();
        let mut windowed = WindowedKernelMatrix::new(Box::new(kernel), 2);

        windowed.add_sample(vec![1.0, 0.0]).unwrap();
        windowed.add_sample(vec![0.0, 1.0]).unwrap();

        let matrix = windowed.get_matrix();

        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        assert!((matrix[1][1] - 1.0).abs() < 1e-10);
        assert!((matrix[0][1]).abs() < 1e-10);

        // Evict first and add new
        windowed.add_sample(vec![1.0, 1.0]).unwrap();

        let matrix = windowed.get_matrix();
        // Now have [0,1] and [1,1]
        // K[0,0] = 1, K[1,1] = 2, K[0,1] = 1
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        assert!((matrix[1][1] - 2.0).abs() < 1e-10);
        assert!((matrix[0][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_kernel_matrix_dimension_mismatch() {
        let kernel = LinearKernel::new();
        let mut windowed = WindowedKernelMatrix::new(Box::new(kernel), 3);

        windowed.add_sample(vec![1.0, 2.0]).unwrap();
        let result = windowed.add_sample(vec![1.0]);
        assert!(result.is_err());
    }

    // ===== ForgetfulKernelMatrix Tests =====

    #[test]
    fn test_forgetful_kernel_matrix_basic() {
        let kernel = LinearKernel::new();
        let config = ForgetfulConfig::with_lambda(0.9).unwrap();
        let mut forgetful = ForgetfulKernelMatrix::new(Box::new(kernel), config);

        forgetful.add_sample(vec![1.0]).unwrap();
        forgetful.add_sample(vec![2.0]).unwrap();

        assert_eq!(forgetful.len(), 2);
        assert!((forgetful.lambda() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_forgetful_kernel_matrix_weights() {
        let kernel = LinearKernel::new();
        let config = ForgetfulConfig {
            lambda: 0.8,
            removal_threshold: None,
            max_samples: None,
        };
        let mut forgetful = ForgetfulKernelMatrix::new(Box::new(kernel), config);

        forgetful.add_sample(vec![1.0]).unwrap();
        forgetful.add_sample(vec![2.0]).unwrap();
        forgetful.add_sample(vec![3.0]).unwrap();

        let weights = forgetful.get_weights();
        // Newest has weight 1.0
        assert!((weights[2] - 1.0).abs() < 1e-10);
        // Middle has weight 0.8
        assert!((weights[1] - 0.8).abs() < 1e-10);
        // Oldest has weight 0.8^2 = 0.64
        assert!((weights[0] - 0.64).abs() < 1e-10);
    }

    #[test]
    fn test_forgetful_kernel_matrix_weighted_matrix() {
        let kernel = LinearKernel::new();
        let config = ForgetfulConfig {
            lambda: 0.5,
            removal_threshold: None,
            max_samples: None,
        };
        let mut forgetful = ForgetfulKernelMatrix::new(Box::new(kernel), config);

        forgetful.add_sample(vec![1.0]).unwrap();
        forgetful.add_sample(vec![1.0]).unwrap();

        let weighted = forgetful.get_weighted_matrix();

        // w[0] = 0.5, w[1] = 1.0
        // K[0,0] = 1 * sqrt(0.5) * sqrt(0.5) = 0.5
        // K[1,1] = 1 * sqrt(1.0) * sqrt(1.0) = 1.0
        // K[0,1] = 1 * sqrt(0.5) * sqrt(1.0) = sqrt(0.5)
        assert!((weighted[0][0] - 0.5).abs() < 1e-10);
        assert!((weighted[1][1] - 1.0).abs() < 1e-10);
        assert!((weighted[0][1] - 0.5_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_forgetful_kernel_matrix_removal_threshold() {
        let kernel = LinearKernel::new();
        let config = ForgetfulConfig {
            lambda: 0.5,
            removal_threshold: Some(0.3),
            max_samples: None,
        };
        let mut forgetful = ForgetfulKernelMatrix::new(Box::new(kernel), config);

        forgetful.add_sample(vec![1.0]).unwrap();
        forgetful.add_sample(vec![2.0]).unwrap();
        // First sample now has weight 0.5

        forgetful.add_sample(vec![3.0]).unwrap();
        // First sample would have weight 0.25 < 0.3, should be removed

        assert_eq!(forgetful.len(), 2);
    }

    #[test]
    fn test_forgetful_kernel_matrix_max_samples() {
        let kernel = LinearKernel::new();
        let config = ForgetfulConfig {
            lambda: 1.0, // No decay
            removal_threshold: None,
            max_samples: Some(2),
        };
        let mut forgetful = ForgetfulKernelMatrix::new(Box::new(kernel), config);

        forgetful.add_sample(vec![1.0]).unwrap();
        forgetful.add_sample(vec![2.0]).unwrap();
        forgetful.add_sample(vec![3.0]).unwrap();

        assert_eq!(forgetful.len(), 2);
    }

    #[test]
    fn test_forgetful_kernel_matrix_effective_size() {
        let kernel = LinearKernel::new();
        let config = ForgetfulConfig {
            lambda: 0.9,
            removal_threshold: None,
            max_samples: None,
        };
        let mut forgetful = ForgetfulKernelMatrix::new(Box::new(kernel), config);

        forgetful.add_sample(vec![1.0]).unwrap();
        forgetful.add_sample(vec![2.0]).unwrap();
        forgetful.add_sample(vec![3.0]).unwrap();

        // Weights: 0.81, 0.9, 1.0
        let eff_size = forgetful.effective_size();
        assert!((eff_size - 2.71).abs() < 1e-10);
    }

    #[test]
    fn test_forgetful_kernel_matrix_invalid_lambda() {
        let result = ForgetfulConfig::with_lambda(0.0);
        assert!(result.is_err());

        let result = ForgetfulConfig::with_lambda(1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_forgetful_kernel_matrix_dimension_mismatch() {
        let kernel = LinearKernel::new();
        let config = ForgetfulConfig::with_lambda(0.9).unwrap();
        let mut forgetful = ForgetfulKernelMatrix::new(Box::new(kernel), config);

        forgetful.add_sample(vec![1.0, 2.0]).unwrap();
        let result = forgetful.add_sample(vec![1.0]);
        assert!(result.is_err());
    }

    // ===== AdaptiveKernelMatrix Tests =====

    #[test]
    fn test_adaptive_kernel_matrix_basic() {
        let mut adaptive = AdaptiveKernelMatrix::new(
            |gamma| Box::new(RbfKernel::new(RbfKernelConfig::new(gamma)).unwrap()),
            1.0,
            0.1,
        );

        adaptive.add_sample(vec![1.0, 2.0]).unwrap();
        adaptive.add_sample(vec![3.0, 4.0]).unwrap();
        adaptive.add_sample(vec![5.0, 6.0]).unwrap();

        assert_eq!(adaptive.len(), 3);
        assert!(adaptive.bandwidth() > 0.0);
    }

    #[test]
    fn test_adaptive_kernel_matrix_bandwidth_update() {
        let mut adaptive = AdaptiveKernelMatrix::new(
            |gamma| Box::new(RbfKernel::new(RbfKernelConfig::new(gamma)).unwrap()),
            1.0,
            0.5, // High adaptation rate
        );

        let initial = adaptive.bandwidth();

        adaptive.add_sample(vec![0.0]).unwrap();
        adaptive.add_sample(vec![10.0]).unwrap();

        // Bandwidth should have changed
        let after = adaptive.bandwidth();
        assert_ne!(initial, after);
    }

    // ===== Edge case tests =====

    #[test]
    fn test_online_empty_operations() {
        let kernel = LinearKernel::new();
        let online = OnlineKernelMatrix::new(Box::new(kernel));

        assert!(online.is_empty());
        assert!(online.get_matrix().is_empty());
        assert!(online.get_samples().is_empty());
    }

    #[test]
    fn test_windowed_clear() {
        let kernel = LinearKernel::new();
        let mut windowed = WindowedKernelMatrix::new(Box::new(kernel), 3);

        windowed.add_sample(vec![1.0]).unwrap();
        windowed.add_sample(vec![2.0]).unwrap();
        windowed.clear();

        assert!(windowed.is_empty());
        assert_eq!(windowed.len(), 0);
    }

    #[test]
    fn test_forgetful_clear() {
        let kernel = LinearKernel::new();
        let config = ForgetfulConfig::with_lambda(0.9).unwrap();
        let mut forgetful = ForgetfulKernelMatrix::new(Box::new(kernel), config);

        forgetful.add_sample(vec![1.0]).unwrap();
        forgetful.add_sample(vec![2.0]).unwrap();
        forgetful.clear();

        assert!(forgetful.is_empty());
        assert_eq!(forgetful.len(), 0);
    }
}
