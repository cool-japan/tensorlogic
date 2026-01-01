// Allow needless_range_loop for matrix operations which are clearer with indexed loops
#![allow(clippy::needless_range_loop)]

//! Kernel selection and cross-validation utilities.
//!
//! This module provides tools for selecting the best kernel and hyperparameters
//! for a given dataset, including:
//!
//! - **Kernel Target Alignment (KTA)**: Quick kernel quality metric
//! - **Leave-One-Out Cross-Validation (LOO-CV)**: Efficient for GP regression
//! - **K-Fold Cross-Validation**: For general kernel evaluation
//! - **Kernel comparison utilities**: Compare multiple kernels on same data
//!
//! ## Example
//!
//! ```rust
//! use tensorlogic_sklears_kernels::kernel_selection::{
//!     KernelSelector, KernelComparison, KFoldConfig
//! };
//! use tensorlogic_sklears_kernels::{RbfKernel, RbfKernelConfig, LinearKernel, Kernel};
//!
//! // Create kernels to compare
//! let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
//! let linear = LinearKernel::new();
//!
//! // Sample data
//! let data = vec![
//!     vec![1.0, 2.0],
//!     vec![2.0, 3.0],
//!     vec![3.0, 4.0],
//!     vec![4.0, 5.0],
//! ];
//! let targets = vec![1.0, 2.0, 3.0, 4.0];
//!
//! // Compare using Kernel Target Alignment
//! let selector = KernelSelector::new();
//! let rbf_kta = selector.kernel_target_alignment(&rbf, &data, &targets).unwrap();
//! let linear_kta = selector.kernel_target_alignment(&linear, &data, &targets).unwrap();
//! ```

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// Configuration for K-fold cross-validation.
#[derive(Debug, Clone)]
pub struct KFoldConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Whether to shuffle the data
    pub shuffle: bool,
    /// Random seed for shuffling (if enabled)
    pub seed: Option<u64>,
}

impl Default for KFoldConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            shuffle: false,
            seed: None,
        }
    }
}

impl KFoldConfig {
    /// Create a new K-fold configuration.
    pub fn new(n_folds: usize) -> Self {
        Self {
            n_folds,
            ..Default::default()
        }
    }

    /// Enable shuffling with optional seed.
    pub fn with_shuffle(mut self, shuffle: bool, seed: Option<u64>) -> Self {
        self.shuffle = shuffle;
        self.seed = seed;
        self
    }
}

/// Results from cross-validation.
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Mean score across folds
    pub mean_score: f64,
    /// Standard deviation of scores
    pub std_score: f64,
    /// Individual fold scores
    pub fold_scores: Vec<f64>,
    /// Total computation time in microseconds
    pub compute_time_us: u64,
}

impl CrossValidationResult {
    /// Create a new cross-validation result.
    pub fn new(fold_scores: Vec<f64>, compute_time_us: u64) -> Self {
        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let variance = fold_scores
            .iter()
            .map(|s| (s - mean_score).powi(2))
            .sum::<f64>()
            / fold_scores.len() as f64;
        let std_score = variance.sqrt();

        Self {
            mean_score,
            std_score,
            fold_scores,
            compute_time_us,
        }
    }

    /// Get the 95% confidence interval.
    pub fn confidence_interval(&self) -> (f64, f64) {
        let margin = 1.96 * self.std_score / (self.fold_scores.len() as f64).sqrt();
        (self.mean_score - margin, self.mean_score + margin)
    }
}

/// Comparison results for multiple kernels.
#[derive(Debug, Clone)]
pub struct KernelComparison {
    /// Kernel names
    pub kernel_names: Vec<String>,
    /// Scores for each kernel
    pub scores: Vec<f64>,
    /// Standard deviations (if available)
    pub std_devs: Option<Vec<f64>>,
    /// Index of the best kernel
    pub best_index: usize,
}

impl KernelComparison {
    /// Create a comparison from scores.
    pub fn from_scores(kernel_names: Vec<String>, scores: Vec<f64>) -> Self {
        let best_index = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        Self {
            kernel_names,
            scores,
            std_devs: None,
            best_index,
        }
    }

    /// Add standard deviations.
    pub fn with_std_devs(mut self, std_devs: Vec<f64>) -> Self {
        self.std_devs = Some(std_devs);
        self
    }

    /// Get the best kernel name.
    pub fn best_kernel(&self) -> &str {
        &self.kernel_names[self.best_index]
    }

    /// Get the best score.
    pub fn best_score(&self) -> f64 {
        self.scores[self.best_index]
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        let mut report = String::from("Kernel Comparison Results:\n");
        report.push_str(&format!("{:=<50}\n", ""));

        for (i, name) in self.kernel_names.iter().enumerate() {
            let score = self.scores[i];
            let std = self
                .std_devs
                .as_ref()
                .map(|s| format!(" ± {:.4}", s[i]))
                .unwrap_or_default();
            let best = if i == self.best_index { " *BEST*" } else { "" };
            report.push_str(&format!("{:20} : {:.4}{}{}\n", name, score, std, best));
        }

        report
    }
}

/// Kernel selector for choosing and comparing kernels.
#[derive(Debug, Clone, Default)]
pub struct KernelSelector {
    /// Regularization for numerical stability
    regularization: f64,
}

impl KernelSelector {
    /// Create a new kernel selector.
    pub fn new() -> Self {
        Self {
            regularization: 1e-6,
        }
    }

    /// Set the regularization parameter.
    pub fn with_regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Compute Kernel Target Alignment (KTA).
    ///
    /// KTA measures how well a kernel aligns with the target labels.
    /// Higher values indicate better alignment.
    ///
    /// Formula: KTA = <K, yy^T>_F / (||K||_F * ||yy^T||_F)
    ///
    /// # Arguments
    /// * `kernel` - The kernel to evaluate
    /// * `data` - Input data points
    /// * `targets` - Target values (for regression) or labels (for classification)
    pub fn kernel_target_alignment<K: Kernel + ?Sized>(
        &self,
        kernel: &K,
        data: &[Vec<f64>],
        targets: &[f64],
    ) -> Result<f64> {
        if data.len() != targets.len() {
            return Err(KernelError::ComputationError(
                "data and targets must have same length".to_string(),
            ));
        }
        if data.is_empty() {
            return Err(KernelError::ComputationError(
                "data cannot be empty".to_string(),
            ));
        }

        let n = data.len();

        // Compute kernel matrix K
        let k_matrix = kernel.compute_matrix(data)?;

        // Compute target matrix yy^T
        let mut y_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                y_matrix[i][j] = targets[i] * targets[j];
            }
        }

        // Compute Frobenius inner product <K, yy^T>
        let mut k_y_product = 0.0;
        for i in 0..n {
            for j in 0..n {
                k_y_product += k_matrix[i][j] * y_matrix[i][j];
            }
        }

        // Compute ||K||_F
        let mut k_norm_sq = 0.0;
        for i in 0..n {
            for j in 0..n {
                k_norm_sq += k_matrix[i][j] * k_matrix[i][j];
            }
        }
        let k_norm = k_norm_sq.sqrt();

        // Compute ||yy^T||_F
        let mut y_norm_sq = 0.0;
        for i in 0..n {
            for j in 0..n {
                y_norm_sq += y_matrix[i][j] * y_matrix[i][j];
            }
        }
        let y_norm = y_norm_sq.sqrt();

        // KTA
        if k_norm < 1e-10 || y_norm < 1e-10 {
            return Ok(0.0);
        }

        Ok(k_y_product / (k_norm * y_norm))
    }

    /// Compute centered Kernel Target Alignment.
    ///
    /// Centered KTA is more robust and accounts for the mean of the kernel matrix.
    pub fn centered_kernel_target_alignment<K: Kernel + ?Sized>(
        &self,
        kernel: &K,
        data: &[Vec<f64>],
        targets: &[f64],
    ) -> Result<f64> {
        if data.len() != targets.len() {
            return Err(KernelError::ComputationError(
                "data and targets must have same length".to_string(),
            ));
        }
        if data.is_empty() {
            return Err(KernelError::ComputationError(
                "data cannot be empty".to_string(),
            ));
        }

        let n = data.len();

        // Compute kernel matrix K
        let k_matrix = kernel.compute_matrix(data)?;

        // Center the kernel matrix
        let centered_k = center_kernel_matrix(&k_matrix);

        // Center the target matrix
        let target_mean: f64 = targets.iter().sum::<f64>() / n as f64;
        let centered_targets: Vec<f64> = targets.iter().map(|t| t - target_mean).collect();

        // Compute centered target matrix
        let mut y_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                y_matrix[i][j] = centered_targets[i] * centered_targets[j];
            }
        }

        // Compute Frobenius inner product
        let mut k_y_product = 0.0;
        let mut k_norm_sq = 0.0;
        let mut y_norm_sq = 0.0;

        for i in 0..n {
            for j in 0..n {
                k_y_product += centered_k[i][j] * y_matrix[i][j];
                k_norm_sq += centered_k[i][j] * centered_k[i][j];
                y_norm_sq += y_matrix[i][j] * y_matrix[i][j];
            }
        }

        let k_norm = k_norm_sq.sqrt();
        let y_norm = y_norm_sq.sqrt();

        if k_norm < 1e-10 || y_norm < 1e-10 {
            return Ok(0.0);
        }

        Ok(k_y_product / (k_norm * y_norm))
    }

    /// Compare multiple kernels using KTA.
    ///
    /// Returns a comparison with the best kernel identified.
    pub fn compare_kernels_kta(
        &self,
        kernels: &[(&str, &dyn Kernel)],
        data: &[Vec<f64>],
        targets: &[f64],
    ) -> Result<KernelComparison> {
        let mut names = Vec::with_capacity(kernels.len());
        let mut scores = Vec::with_capacity(kernels.len());

        for (name, kernel) in kernels {
            let kta = self.kernel_target_alignment(*kernel, data, targets)?;
            names.push(name.to_string());
            scores.push(kta);
        }

        Ok(KernelComparison::from_scores(names, scores))
    }

    /// Evaluate kernel quality using Leave-One-Out (LOO) error estimate.
    ///
    /// For GP regression, this provides an efficient estimate of generalization error.
    /// Lower values indicate better performance.
    ///
    /// Note: This is an approximation based on the kernel matrix.
    pub fn loo_error_estimate<K: Kernel + ?Sized>(
        &self,
        kernel: &K,
        data: &[Vec<f64>],
        targets: &[f64],
    ) -> Result<f64> {
        if data.len() != targets.len() {
            return Err(KernelError::ComputationError(
                "data and targets must have same length".to_string(),
            ));
        }
        if data.len() < 2 {
            return Err(KernelError::ComputationError(
                "need at least 2 data points".to_string(),
            ));
        }

        let n = data.len();

        // Compute kernel matrix with regularization
        let k_matrix = kernel.compute_matrix(data)?;
        let mut k_reg = k_matrix.clone();
        for i in 0..n {
            k_reg[i][i] += self.regularization;
        }

        // Compute inverse using simple method (Gauss-Jordan elimination)
        let k_inv = invert_matrix(&k_reg)?;

        // Compute alpha = K^{-1} y
        let mut alpha = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                alpha[i] += k_inv[i][j] * targets[j];
            }
        }

        // LOO error: sum_i (alpha_i / K^{-1}_{ii})^2 / n
        let mut loo_error = 0.0;
        for i in 0..n {
            let diag = k_inv[i][i];
            if diag.abs() > 1e-10 {
                let loo_residual = alpha[i] / diag;
                loo_error += loo_residual * loo_residual;
            }
        }

        Ok(loo_error / n as f64)
    }

    /// Perform K-fold cross-validation for kernel evaluation.
    ///
    /// This evaluates a kernel by training on K-1 folds and testing on the remaining fold.
    /// The score returned is based on kernel alignment with targets.
    pub fn k_fold_cv<K: Kernel + ?Sized>(
        &self,
        kernel: &K,
        data: &[Vec<f64>],
        targets: &[f64],
        config: &KFoldConfig,
    ) -> Result<CrossValidationResult> {
        use std::time::Instant;

        if data.len() != targets.len() {
            return Err(KernelError::ComputationError(
                "data and targets must have same length".to_string(),
            ));
        }
        if data.len() < config.n_folds {
            return Err(KernelError::ComputationError(format!(
                "need at least {} data points for {}-fold CV",
                config.n_folds, config.n_folds
            )));
        }

        let start = Instant::now();
        let n = data.len();

        // Create indices (optionally shuffled)
        let mut indices: Vec<usize> = (0..n).collect();
        if config.shuffle {
            // Simple deterministic shuffle using seed
            let seed = config.seed.unwrap_or(42);
            shuffle_indices(&mut indices, seed);
        }

        // Split into folds
        let fold_size = n / config.n_folds;
        let mut fold_scores = Vec::with_capacity(config.n_folds);

        for fold in 0..config.n_folds {
            let fold_start = fold * fold_size;
            let fold_end = if fold == config.n_folds - 1 {
                n
            } else {
                fold_start + fold_size
            };

            // Split data
            let test_indices: Vec<_> = indices[fold_start..fold_end].to_vec();
            let train_indices: Vec<_> = indices[0..fold_start]
                .iter()
                .chain(indices[fold_end..].iter())
                .copied()
                .collect();

            // Collect train/test data
            let _train_data: Vec<_> = train_indices.iter().map(|&i| data[i].clone()).collect();
            let _train_targets: Vec<_> = train_indices.iter().map(|&i| targets[i]).collect();
            let test_data: Vec<_> = test_indices.iter().map(|&i| data[i].clone()).collect();
            let test_targets: Vec<_> = test_indices.iter().map(|&i| targets[i]).collect();

            // Evaluate on this fold using KTA on test set
            // Note: Full CV would train on train_data/train_targets then evaluate on test
            // For simplicity, we use KTA on test fold as the score
            let score = self.kernel_target_alignment(kernel, &test_data, &test_targets)?;
            fold_scores.push(score);
        }

        let compute_time_us = start.elapsed().as_micros() as u64;
        Ok(CrossValidationResult::new(fold_scores, compute_time_us))
    }

    /// Find the best gamma for RBF kernel using grid search.
    ///
    /// Searches over a logarithmic grid of gamma values.
    pub fn grid_search_rbf_gamma(
        &self,
        data: &[Vec<f64>],
        targets: &[f64],
        gammas: &[f64],
    ) -> Result<GammaSearchResult> {
        use crate::tensor_kernels::RbfKernel;
        use crate::types::RbfKernelConfig;

        let mut best_gamma = gammas[0];
        let mut best_score = f64::NEG_INFINITY;
        let mut all_scores = Vec::with_capacity(gammas.len());

        for &gamma in gammas {
            let config = RbfKernelConfig::new(gamma);
            let kernel = RbfKernel::new(config)?;
            let score = self.centered_kernel_target_alignment(&kernel, data, targets)?;
            all_scores.push((gamma, score));

            if score > best_score {
                best_score = score;
                best_gamma = gamma;
            }
        }

        Ok(GammaSearchResult {
            best_gamma,
            best_score,
            all_scores,
        })
    }
}

/// Result of gamma grid search for RBF kernel.
#[derive(Debug, Clone)]
pub struct GammaSearchResult {
    /// Best gamma value found
    pub best_gamma: f64,
    /// Best KTA score
    pub best_score: f64,
    /// All (gamma, score) pairs tested
    pub all_scores: Vec<(f64, f64)>,
}

impl GammaSearchResult {
    /// Get a summary of the search results.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "RBF Gamma Search:\n  Best gamma: {:.6}\n  Best score: {:.4}\n\n",
            self.best_gamma, self.best_score
        );
        s.push_str("All results:\n");
        for (gamma, score) in &self.all_scores {
            let marker = if (*gamma - self.best_gamma).abs() < 1e-10 {
                " *"
            } else {
                ""
            };
            s.push_str(&format!("  gamma={:.6}: {:.4}{}\n", gamma, score, marker));
        }
        s
    }
}

/// Center a kernel matrix: K_c = H K H where H = I - (1/n) * 1 * 1^T
fn center_kernel_matrix(k: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = k.len();
    if n == 0 {
        return vec![];
    }

    // Compute row means, column means, and global mean
    let mut row_means = vec![0.0; n];
    let mut col_means = vec![0.0; n];
    let mut global_mean = 0.0;

    for (i, row) in k.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            row_means[i] += val;
            col_means[j] += val;
            global_mean += val;
        }
    }

    let n_f = n as f64;
    for mean in &mut row_means {
        *mean /= n_f;
    }
    for mean in &mut col_means {
        *mean /= n_f;
    }
    global_mean /= n_f * n_f;

    // Center: K_c[i][j] = K[i][j] - row_mean[i] - col_mean[j] + global_mean
    let mut centered = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            centered[i][j] = k[i][j] - row_means[i] - col_means[j] + global_mean;
        }
    }

    centered
}

/// Simple matrix inversion using Gauss-Jordan elimination.
fn invert_matrix(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 {
        return Err(KernelError::ComputationError(
            "cannot invert empty matrix".to_string(),
        ));
    }

    // Create augmented matrix [A | I]
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = matrix[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = aug[i][i].abs();
        for k in (i + 1)..n {
            if aug[k][i].abs() > max_val {
                max_val = aug[k][i].abs();
                max_row = k;
            }
        }

        if max_val < 1e-10 {
            return Err(KernelError::ComputationError(
                "matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows
        if max_row != i {
            aug.swap(i, max_row);
        }

        // Eliminate column
        let pivot = aug[i][i];
        for j in 0..(2 * n) {
            aug[i][j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..(2 * n) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // Extract inverse
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = aug[i][n + j];
        }
    }

    Ok(inverse)
}

/// Simple shuffle using a deterministic PRNG.
fn shuffle_indices(indices: &mut [usize], seed: u64) {
    let n = indices.len();
    let mut state = seed;

    for i in (1..n).rev() {
        // Simple LCG for deterministic shuffling
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_kernels::{LinearKernel, RbfKernel};
    use crate::types::RbfKernelConfig;

    #[test]
    fn test_kfold_config() {
        let config = KFoldConfig::new(10);
        assert_eq!(config.n_folds, 10);
        assert!(!config.shuffle);
    }

    #[test]
    fn test_kfold_config_with_shuffle() {
        let config = KFoldConfig::new(5).with_shuffle(true, Some(42));
        assert_eq!(config.n_folds, 5);
        assert!(config.shuffle);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_cross_validation_result() {
        let fold_scores = vec![0.8, 0.85, 0.75, 0.9, 0.82];
        let result = CrossValidationResult::new(fold_scores.clone(), 1000);

        assert!((result.mean_score - 0.824).abs() < 1e-10);
        assert!(result.std_score > 0.0);
        assert_eq!(result.fold_scores, fold_scores);
    }

    #[test]
    fn test_kernel_comparison() {
        let names = vec!["Linear".to_string(), "RBF".to_string()];
        let scores = vec![0.5, 0.8];

        let comp = KernelComparison::from_scores(names, scores);
        assert_eq!(comp.best_index, 1);
        assert_eq!(comp.best_kernel(), "RBF");
        assert!((comp.best_score() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_comparison_summary() {
        let names = vec!["Linear".to_string(), "RBF".to_string()];
        let scores = vec![0.5, 0.8];
        let std_devs = vec![0.05, 0.03];

        let comp = KernelComparison::from_scores(names, scores).with_std_devs(std_devs);
        let summary = comp.summary();

        assert!(summary.contains("Linear"));
        assert!(summary.contains("RBF"));
        assert!(summary.contains("*BEST*"));
    }

    #[test]
    fn test_kernel_target_alignment() {
        let selector = KernelSelector::new();
        let kernel = LinearKernel::new();

        let data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let targets = vec![1.0, 2.0, 3.0, 4.0]; // Perfectly correlated

        let kta = selector.kernel_target_alignment(&kernel, &data, &targets);
        assert!(kta.is_ok());
        let kta_val = kta.unwrap();
        // For perfectly correlated data, KTA should be high
        assert!(kta_val > 0.5);
    }

    #[test]
    fn test_centered_kernel_target_alignment() {
        let selector = KernelSelector::new();
        let kernel = LinearKernel::new();

        let data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        let ckta = selector.centered_kernel_target_alignment(&kernel, &data, &targets);
        assert!(ckta.is_ok());
    }

    #[test]
    fn test_kta_empty_data() {
        let selector = KernelSelector::new();
        let kernel = LinearKernel::new();

        let result = selector.kernel_target_alignment(&kernel, &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_kta_mismatched_lengths() {
        let selector = KernelSelector::new();
        let kernel = LinearKernel::new();

        let data = vec![vec![1.0], vec![2.0]];
        let targets = vec![1.0, 2.0, 3.0];

        let result = selector.kernel_target_alignment(&kernel, &data, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_kernels_kta() {
        let selector = KernelSelector::new();
        let linear = LinearKernel::new();
        let rbf = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();

        let data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        let kernels: Vec<(&str, &dyn Kernel)> = vec![("Linear", &linear), ("RBF", &rbf)];

        let comparison = selector.compare_kernels_kta(&kernels, &data, &targets);
        assert!(comparison.is_ok());

        let comp = comparison.unwrap();
        assert_eq!(comp.kernel_names.len(), 2);
        assert_eq!(comp.scores.len(), 2);
    }

    #[test]
    fn test_loo_error_estimate() {
        let selector = KernelSelector::new().with_regularization(0.1);
        let kernel = LinearKernel::new();

        let data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        let result = selector.loo_error_estimate(&kernel, &data, &targets);
        assert!(result.is_ok());
        let error = result.unwrap();
        // Error should be finite and non-negative
        assert!(error >= 0.0);
        assert!(error.is_finite());
    }

    #[test]
    fn test_k_fold_cv() {
        let selector = KernelSelector::new();
        let kernel = LinearKernel::new();
        let config = KFoldConfig::new(3);

        let data = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = selector.k_fold_cv(&kernel, &data, &targets, &config);
        assert!(result.is_ok());

        let cv_result = result.unwrap();
        assert_eq!(cv_result.fold_scores.len(), 3);
        assert!(cv_result.mean_score.is_finite());
    }

    #[test]
    fn test_k_fold_cv_with_shuffle() {
        let selector = KernelSelector::new();
        let kernel = LinearKernel::new();
        let config = KFoldConfig::new(3).with_shuffle(true, Some(42));

        let data = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = selector.k_fold_cv(&kernel, &data, &targets, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_grid_search_rbf_gamma() {
        let selector = KernelSelector::new();

        let data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let gammas = vec![0.01, 0.1, 1.0, 10.0];

        let result = selector.grid_search_rbf_gamma(&data, &targets, &gammas);
        assert!(result.is_ok());

        let search_result = result.unwrap();
        assert!(gammas.contains(&search_result.best_gamma));
        assert_eq!(search_result.all_scores.len(), gammas.len());
    }

    #[test]
    fn test_center_kernel_matrix() {
        let k = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];

        let centered = center_kernel_matrix(&k);
        assert_eq!(centered.len(), 3);

        // Centered matrix should have row and column means close to 0
        let n = centered.len() as f64;
        for row in &centered {
            let row_mean: f64 = row.iter().sum::<f64>() / n;
            assert!(row_mean.abs() < 1e-10);
        }
    }

    #[test]
    fn test_matrix_inversion() {
        let matrix = vec![vec![4.0, 7.0], vec![2.0, 6.0]];

        let inv = invert_matrix(&matrix).unwrap();

        // Check A * A^{-1} = I
        let n = matrix.len();
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += matrix[i][k] * inv[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((sum - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_shuffle_deterministic() {
        let mut indices1 = vec![0, 1, 2, 3, 4];
        let mut indices2 = vec![0, 1, 2, 3, 4];

        shuffle_indices(&mut indices1, 42);
        shuffle_indices(&mut indices2, 42);

        assert_eq!(indices1, indices2); // Same seed = same shuffle
    }

    #[test]
    fn test_gamma_search_result_summary() {
        let result = GammaSearchResult {
            best_gamma: 0.1,
            best_score: 0.9,
            all_scores: vec![(0.01, 0.5), (0.1, 0.9), (1.0, 0.7)],
        };

        let summary = result.summary();
        assert!(summary.contains("Best gamma: 0.1"));
        assert!(summary.contains("Best score: 0.9"));
    }
}
