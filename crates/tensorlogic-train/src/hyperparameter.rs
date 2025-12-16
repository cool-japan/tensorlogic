//! Hyperparameter optimization utilities.
//!
//! This module provides various hyperparameter search strategies:
//! - Grid search (exhaustive search over parameter grid)
//! - Random search (random sampling from parameter space)
//! - Bayesian optimization (Gaussian Process-based optimization with acquisition functions)
//! - Parameter space definition
//! - Result tracking and comparison

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{s, Array1, Array2};
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

// ============================================================================
// Bayesian Optimization
// ============================================================================

/// Acquisition function type for Bayesian Optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcquisitionFunction {
    /// Expected Improvement - balances exploration and exploitation.
    ExpectedImprovement { xi: f64 },
    /// Upper Confidence Bound - uses uncertainty to guide exploration.
    UpperConfidenceBound { kappa: f64 },
    /// Probability of Improvement - probability of improving over best.
    ProbabilityOfImprovement { xi: f64 },
}

impl Default for AcquisitionFunction {
    fn default() -> Self {
        Self::ExpectedImprovement { xi: 0.01 }
    }
}

/// Gaussian Process kernel for Bayesian Optimization.
#[derive(Debug, Clone, Copy)]
pub enum GpKernel {
    /// Radial Basis Function (RBF) / Squared Exponential kernel.
    /// K(x, x') = σ² * exp(-||x - x'||² / (2 * l²))
    Rbf {
        /// Signal variance (output scale).
        sigma: f64,
        /// Length scale (input scale).
        length_scale: f64,
    },
    /// Matérn kernel with ν = 3/2.
    /// K(x, x') = σ² * (1 + √3 * r / l) * exp(-√3 * r / l)
    Matern32 {
        /// Signal variance.
        sigma: f64,
        /// Length scale.
        length_scale: f64,
    },
}

impl Default for GpKernel {
    fn default() -> Self {
        Self::Rbf {
            sigma: 1.0,
            length_scale: 1.0,
        }
    }
}

impl GpKernel {
    /// Compute kernel matrix K(X, X').
    fn compute_kernel(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1_row = x1.row(i);
                let x2_row = x2.row(j);
                let dist_sq = x1_row
                    .iter()
                    .zip(x2_row.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();

                k[[i, j]] = match self {
                    Self::Rbf {
                        sigma,
                        length_scale,
                    } => sigma.powi(2) * (-dist_sq / (2.0 * length_scale.powi(2))).exp(),
                    Self::Matern32 {
                        sigma,
                        length_scale,
                    } => {
                        let r = dist_sq.sqrt();
                        let sqrt3_r_l = (3.0_f64).sqrt() * r / length_scale;
                        sigma.powi(2) * (1.0 + sqrt3_r_l) * (-sqrt3_r_l).exp()
                    }
                };
            }
        }

        k
    }

    /// Compute kernel vector k(X, x).
    fn compute_kernel_vector(&self, x_train: &Array2<f64>, x_test: &Array1<f64>) -> Array1<f64> {
        let n = x_train.nrows();
        let mut k = Array1::zeros(n);

        for i in 0..n {
            let x_train_row = x_train.row(i);
            let dist_sq = x_train_row
                .iter()
                .zip(x_test.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();

            k[i] = match self {
                Self::Rbf {
                    sigma,
                    length_scale,
                } => sigma.powi(2) * (-dist_sq / (2.0 * length_scale.powi(2))).exp(),
                Self::Matern32 {
                    sigma,
                    length_scale,
                } => {
                    let r = dist_sq.sqrt();
                    let sqrt3_r_l = (3.0_f64).sqrt() * r / length_scale;
                    sigma.powi(2) * (1.0 + sqrt3_r_l) * (-sqrt3_r_l).exp()
                }
            };
        }

        k
    }
}

/// Gaussian Process regressor for Bayesian Optimization.
///
/// Provides probabilistic predictions with uncertainty estimates.
#[derive(Debug)]
pub struct GaussianProcess {
    /// Kernel function.
    kernel: GpKernel,
    /// Noise variance (observation noise).
    noise_variance: f64,
    /// Training inputs (normalized to [0, 1]).
    x_train: Option<Array2<f64>>,
    /// Training outputs (standardized).
    y_train: Option<Array1<f64>>,
    /// Mean of training outputs (for standardization).
    y_mean: f64,
    /// Std of training outputs (for standardization).
    y_std: f64,
    /// Cholesky decomposition of K + σ²I (cached for efficiency).
    l_matrix: Option<Array2<f64>>,
    /// Alpha = L^T \ (L \ y) (cached).
    alpha: Option<Array1<f64>>,
}

impl GaussianProcess {
    /// Create a new Gaussian Process.
    pub fn new(kernel: GpKernel, noise_variance: f64) -> Self {
        Self {
            kernel,
            noise_variance,
            x_train: None,
            y_train: None,
            y_mean: 0.0,
            y_std: 1.0,
            l_matrix: None,
            alpha: None,
        }
    }

    /// Fit the GP to training data.
    pub fn fit(&mut self, x: Array2<f64>, y: Array1<f64>) -> TrainResult<()> {
        if x.nrows() != y.len() {
            return Err(TrainError::InvalidParameter(
                "X and y must have same number of samples".to_string(),
            ));
        }

        // Standardize y
        let y_mean = y.mean().unwrap_or(0.0);
        let y_std = y.std(0.0).max(1e-8);
        let y_standardized = (&y - y_mean) / y_std;

        // Compute kernel matrix
        let k = self.kernel.compute_kernel(&x, &x);

        // Add noise: K + σ²I
        let mut k_noisy = k;
        for i in 0..k_noisy.nrows() {
            k_noisy[[i, i]] += self.noise_variance;
        }

        // Cholesky decomposition
        let l = self.cholesky(&k_noisy)?;

        // Solve L * alpha' = y
        let alpha_prime = self.forward_substitution(&l, &y_standardized)?;
        // Solve L^T * alpha = alpha'
        let alpha = self.backward_substitution(&l, &alpha_prime)?;

        self.x_train = Some(x);
        self.y_train = Some(y_standardized);
        self.y_mean = y_mean;
        self.y_std = y_std;
        self.l_matrix = Some(l);
        self.alpha = Some(alpha);

        Ok(())
    }

    /// Predict mean and standard deviation at test points.
    pub fn predict(&self, x_test: &Array2<f64>) -> TrainResult<(Array1<f64>, Array1<f64>)> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| TrainError::InvalidParameter("GP not fitted".to_string()))?;
        let l_matrix = self.l_matrix.as_ref().unwrap();
        let alpha = self.alpha.as_ref().unwrap();

        let n_test = x_test.nrows();
        let mut means = Array1::zeros(n_test);
        let mut stds = Array1::zeros(n_test);

        for i in 0..n_test {
            let x = x_test.row(i).to_owned();

            // Compute k(X*, x)
            let k_star = self.kernel.compute_kernel_vector(x_train, &x);

            // Mean: k(X*, x)^T * alpha
            let mean_standardized = k_star.dot(alpha);
            means[i] = mean_standardized * self.y_std + self.y_mean;

            // Variance: k(x, x) - k(X*, x)^T * (K + σ²I)^(-1) * k(X*, x)
            let k_star_star = self
                .kernel
                .compute_kernel_vector(&x_test.slice(s![i..i + 1, ..]).to_owned(), &x)[0];
            let v = self
                .forward_substitution(l_matrix, &k_star)
                .unwrap_or_else(|_| Array1::zeros(k_star.len()));
            let variance_standardized = k_star_star - v.dot(&v);
            stds[i] = (variance_standardized.max(1e-10) * self.y_std.powi(2)).sqrt();
        }

        Ok((means, stds))
    }

    /// Cholesky decomposition: K = L * L^T.
    fn cholesky(&self, k: &Array2<f64>) -> TrainResult<Array2<f64>> {
        let n = k.nrows();
        let mut l = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k_idx in 0..j {
                    sum += l[[i, k_idx]] * l[[j, k_idx]];
                }

                if i == j {
                    let val = k[[i, i]] - sum;
                    if val <= 0.0 {
                        // Add jitter for numerical stability
                        l[[i, j]] = (k[[i, i]] - sum + 1e-6).sqrt();
                    } else {
                        l[[i, j]] = val.sqrt();
                    }
                } else {
                    l[[i, j]] = (k[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        Ok(l)
    }

    /// Forward substitution: solve L * x = b.
    fn forward_substitution(&self, l: &Array2<f64>, b: &Array1<f64>) -> TrainResult<Array1<f64>> {
        let n = l.nrows();
        let mut x = Array1::zeros(n);

        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * x[j];
            }
            x[i] = (b[i] - sum) / l[[i, i]];
        }

        Ok(x)
    }

    /// Backward substitution: solve L^T * x = b.
    fn backward_substitution(&self, l: &Array2<f64>, b: &Array1<f64>) -> TrainResult<Array1<f64>> {
        let n = l.nrows();
        let mut x = Array1::zeros(n);

        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l[[j, i]] * x[j];
            }
            x[i] = (b[i] - sum) / l[[i, i]];
        }

        Ok(x)
    }
}

/// Bayesian Optimization for hyperparameter tuning.
///
/// Uses Gaussian Processes to model the objective function and acquisition
/// functions to intelligently select the next hyperparameters to evaluate.
///
/// # Algorithm
/// 1. Initialize with random samples
/// 2. Fit Gaussian Process to observed data
/// 3. Optimize acquisition function to find next point
/// 4. Evaluate objective at new point
/// 5. Repeat steps 2-4 until budget exhausted
///
/// # Example
/// ```
/// use tensorlogic_train::*;
/// use std::collections::HashMap;
///
/// let mut param_space = HashMap::new();
/// param_space.insert(
///     "lr".to_string(),
///     HyperparamSpace::log_uniform(1e-4, 1e-1).unwrap(),
/// );
///
/// let mut bayes_opt = BayesianOptimization::new(
///     param_space,
///     10,  // n_iterations
///     5,   // n_initial_points
///     42,  // seed
/// );
///
/// // In practice, you would evaluate your model here
/// // bayes_opt.add_result(result);
/// ```
pub struct BayesianOptimization {
    /// Parameter space definition.
    param_space: HashMap<String, HyperparamSpace>,
    /// Number of optimization iterations.
    n_iterations: usize,
    /// Number of random initial points.
    n_initial_points: usize,
    /// Acquisition function.
    acquisition_fn: AcquisitionFunction,
    /// Gaussian Process kernel.
    kernel: GpKernel,
    /// Observation noise.
    noise_variance: f64,
    /// Random number generator.
    rng: StdRng,
    /// Results from all evaluations.
    results: Vec<HyperparamResult>,
    /// Bounds for normalization [min, max] per dimension.
    bounds: Vec<(f64, f64)>,
    /// Parameter names in order.
    param_names: Vec<String>,
}

impl BayesianOptimization {
    /// Create a new Bayesian Optimization instance.
    ///
    /// # Arguments
    /// * `param_space` - Hyperparameter space definition
    /// * `n_iterations` - Number of optimization iterations
    /// * `n_initial_points` - Number of random initialization points
    /// * `seed` - Random seed for reproducibility
    pub fn new(
        param_space: HashMap<String, HyperparamSpace>,
        n_iterations: usize,
        n_initial_points: usize,
        seed: u64,
    ) -> Self {
        let mut param_names: Vec<String> = param_space.keys().cloned().collect();
        param_names.sort(); // Ensure deterministic order

        let bounds = Self::extract_bounds(&param_space, &param_names);

        Self {
            param_space,
            n_iterations,
            n_initial_points,
            acquisition_fn: AcquisitionFunction::default(),
            kernel: GpKernel::default(),
            noise_variance: 1e-6,
            rng: StdRng::seed_from_u64(seed),
            results: Vec::new(),
            bounds,
            param_names,
        }
    }

    /// Set acquisition function.
    pub fn with_acquisition(mut self, acquisition_fn: AcquisitionFunction) -> Self {
        self.acquisition_fn = acquisition_fn;
        self
    }

    /// Set kernel.
    pub fn with_kernel(mut self, kernel: GpKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set noise variance.
    pub fn with_noise(mut self, noise_variance: f64) -> Self {
        self.noise_variance = noise_variance;
        self
    }

    /// Extract bounds from parameter space.
    fn extract_bounds(
        param_space: &HashMap<String, HyperparamSpace>,
        param_names: &[String],
    ) -> Vec<(f64, f64)> {
        param_names
            .iter()
            .map(|name| {
                match &param_space[name] {
                    HyperparamSpace::Continuous { min, max } => (*min, *max),
                    HyperparamSpace::LogUniform { min, max } => (min.ln(), max.ln()),
                    HyperparamSpace::IntRange { min, max } => (*min as f64, *max as f64),
                    HyperparamSpace::Discrete(values) => {
                        // For discrete, we'll use indices
                        (0.0, (values.len() - 1) as f64)
                    }
                }
            })
            .collect()
    }

    /// Suggest next hyperparameter configuration to evaluate.
    pub fn suggest(&mut self) -> TrainResult<HyperparamConfig> {
        // Use random sampling for initial points
        if self.results.len() < self.n_initial_points {
            return Ok(self.random_sample());
        }

        // Build GP from observed data
        let (x_observed, y_observed) = self.get_observations();
        let mut gp = GaussianProcess::new(self.kernel, self.noise_variance);
        gp.fit(x_observed, y_observed)?;

        // Optimize acquisition function
        let best_x = self.optimize_acquisition(&gp)?;

        // Convert to hyperparameter configuration
        self.vector_to_config(&best_x)
    }

    /// Get observations as (X, y) matrices.
    fn get_observations(&self) -> (Array2<f64>, Array1<f64>) {
        let n_samples = self.results.len();
        let n_dims = self.param_names.len();

        let mut x = Array2::zeros((n_samples, n_dims));
        let mut y = Array1::zeros(n_samples);

        for (i, result) in self.results.iter().enumerate() {
            let x_vec = self.config_to_vector(&result.config);
            for (j, &val) in x_vec.iter().enumerate() {
                x[[i, j]] = val;
            }
            y[i] = result.score;
        }

        (x, y)
    }

    /// Optimize acquisition function to find next point.
    fn optimize_acquisition(&mut self, gp: &GaussianProcess) -> TrainResult<Array1<f64>> {
        let n_dims = self.param_names.len();
        let n_candidates = 1000;
        let n_restarts = 10;

        let mut best_acq_value = f64::NEG_INFINITY;
        let mut best_x = Array1::zeros(n_dims);

        // Random search with multiple restarts
        for _ in 0..n_restarts {
            for _ in 0..(n_candidates / n_restarts) {
                // Generate random candidate
                let mut x_candidate = Array1::zeros(n_dims);
                for (i, (min, max)) in self.bounds.iter().enumerate() {
                    x_candidate[i] = min + (max - min) * self.rng.random::<f64>();
                }

                // Evaluate acquisition
                let acq_value = self.evaluate_acquisition(gp, &x_candidate)?;

                if acq_value > best_acq_value {
                    best_acq_value = acq_value;
                    best_x = x_candidate;
                }
            }
        }

        Ok(best_x)
    }

    /// Evaluate acquisition function at a point.
    fn evaluate_acquisition(&self, gp: &GaussianProcess, x: &Array1<f64>) -> TrainResult<f64> {
        let x_mat = x.clone().into_shape_with_order((1, x.len())).unwrap();
        let (mean, std) = gp.predict(&x_mat)?;
        let mu = mean[0];
        let sigma = std[0];

        if sigma < 1e-10 {
            return Ok(0.0);
        }

        let f_best = self
            .results
            .iter()
            .map(|r| r.score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let acq = match self.acquisition_fn {
            AcquisitionFunction::ExpectedImprovement { xi } => {
                let z = (mu - f_best - xi) / sigma;
                let phi = Self::normal_cdf(z);
                let pdf = Self::normal_pdf(z);
                (mu - f_best - xi) * phi + sigma * pdf
            }
            AcquisitionFunction::UpperConfidenceBound { kappa } => mu + kappa * sigma,
            AcquisitionFunction::ProbabilityOfImprovement { xi } => {
                let z = (mu - f_best - xi) / sigma;
                Self::normal_cdf(z)
            }
        };

        Ok(acq)
    }

    /// Standard normal CDF (cumulative distribution function).
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
    }

    /// Standard normal PDF (probability density function).
    fn normal_pdf(x: f64) -> f64 {
        (-0.5 * x.powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    /// Error function approximation.
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Convert configuration to normalized vector [0, 1]^d.
    fn config_to_vector(&self, config: &HyperparamConfig) -> Array1<f64> {
        let n_dims = self.param_names.len();
        let mut x = Array1::zeros(n_dims);

        for (i, name) in self.param_names.iter().enumerate() {
            let value = &config[name];
            let (min, max) = self.bounds[i];

            x[i] = match &self.param_space[name] {
                HyperparamSpace::Continuous { .. } => {
                    let v = value.as_float().unwrap();
                    (v - min) / (max - min)
                }
                HyperparamSpace::LogUniform { .. } => {
                    let v = value.as_float().unwrap();
                    let log_v = v.ln();
                    (log_v - min) / (max - min)
                }
                HyperparamSpace::IntRange { .. } => {
                    let v = value.as_int().unwrap() as f64;
                    (v - min) / (max - min)
                }
                HyperparamSpace::Discrete(values) => {
                    let idx = values.iter().position(|v| v == value).unwrap_or(0);
                    (idx as f64 - min) / (max - min)
                }
            };
        }

        x
    }

    /// Convert normalized vector to configuration.
    fn vector_to_config(&self, x: &Array1<f64>) -> TrainResult<HyperparamConfig> {
        let mut config = HashMap::new();

        for (i, name) in self.param_names.iter().enumerate() {
            let normalized = x[i].clamp(0.0, 1.0);
            let (min, max) = self.bounds[i];
            let value_raw = min + normalized * (max - min);

            let value = match &self.param_space[name] {
                HyperparamSpace::Continuous { .. } => HyperparamValue::Float(value_raw),
                HyperparamSpace::LogUniform { .. } => HyperparamValue::Float(value_raw.exp()),
                HyperparamSpace::IntRange { .. } => HyperparamValue::Int(value_raw.round() as i64),
                HyperparamSpace::Discrete(values) => {
                    let idx = value_raw.round() as usize;
                    values[idx.min(values.len() - 1)].clone()
                }
            };

            config.insert(name.clone(), value);
        }

        Ok(config)
    }

    /// Generate a random sample from parameter space.
    fn random_sample(&mut self) -> HyperparamConfig {
        let mut config = HashMap::new();

        for (name, space) in &self.param_space {
            let value = space.sample(&mut self.rng);
            config.insert(name.clone(), value);
        }

        config
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

    /// Check if optimization is complete.
    pub fn is_complete(&self) -> bool {
        self.results.len() >= self.n_iterations + self.n_initial_points
    }

    /// Get current iteration number.
    pub fn current_iteration(&self) -> usize {
        self.results.len()
    }

    /// Get total budget (initial + iterations).
    pub fn total_budget(&self) -> usize {
        self.n_iterations + self.n_initial_points
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

    // ============================================================================
    // Bayesian Optimization Tests
    // ============================================================================

    #[test]
    fn test_gp_kernel_rbf() {
        let kernel = GpKernel::Rbf {
            sigma: 1.0,
            length_scale: 1.0,
        };

        let x1 = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let x2 = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.5, 0.5]).unwrap();

        let k = kernel.compute_kernel(&x1, &x2);
        assert_eq!(k.shape(), &[2, 2]);

        // K(x, x) should be sigma^2
        assert!((k[[0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gp_kernel_matern() {
        let kernel = GpKernel::Matern32 {
            sigma: 1.0,
            length_scale: 1.0,
        };

        let x = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let k = kernel.compute_kernel(&x, &x);

        // K(x, x) should be sigma^2 for Matern kernel at same point
        assert!((k[[0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gp_fit_and_predict() {
        let kernel = GpKernel::Rbf {
            sigma: 1.0,
            length_scale: 0.5,
        };
        let mut gp = GaussianProcess::new(kernel, 1e-6);

        // Training data: y = x^2
        let x_train = Array2::from_shape_vec((5, 1), vec![0.0, 0.5, 1.0, 1.5, 2.0]).unwrap();
        let y_train = Array1::from_vec(vec![0.0, 0.25, 1.0, 2.25, 4.0]);

        gp.fit(x_train, y_train).unwrap();

        // Test prediction
        let x_test = Array2::from_shape_vec((2, 1), vec![0.75, 1.25]).unwrap();
        let (means, _stds) = gp.predict(&x_test).unwrap();

        assert_eq!(means.len(), 2);
        // Predictions should be reasonable (between 0 and 4)
        assert!(means[0] >= 0.0 && means[0] <= 4.0);
        assert!(means[1] >= 0.0 && means[1] <= 4.0);
    }

    #[test]
    fn test_gp_predict_error_not_fitted() {
        let kernel = GpKernel::default();
        let gp = GaussianProcess::new(kernel, 1e-6);

        let x_test = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        let result = gp.predict(&x_test);

        assert!(result.is_err());
    }

    #[test]
    fn test_gp_fit_dimension_mismatch() {
        let kernel = GpKernel::default();
        let mut gp = GaussianProcess::new(kernel, 1e-6);

        let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0]); // Wrong size

        let result = gp.fit(x, y);
        assert!(result.is_err());
    }

    #[test]
    fn test_acquisition_function_ei() {
        let acq = AcquisitionFunction::ExpectedImprovement { xi: 0.01 };
        assert!(matches!(
            acq,
            AcquisitionFunction::ExpectedImprovement { .. }
        ));
    }

    #[test]
    fn test_acquisition_function_ucb() {
        let acq = AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 };
        assert!(matches!(
            acq,
            AcquisitionFunction::UpperConfidenceBound { .. }
        ));
    }

    #[test]
    fn test_acquisition_function_pi() {
        let acq = AcquisitionFunction::ProbabilityOfImprovement { xi: 0.01 };
        assert!(matches!(
            acq,
            AcquisitionFunction::ProbabilityOfImprovement { .. }
        ));
    }

    #[test]
    fn test_bayesian_optimization_creation() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "lr".to_string(),
            HyperparamSpace::log_uniform(1e-4, 1e-1).unwrap(),
        );

        let bayes_opt = BayesianOptimization::new(param_space, 10, 5, 42);

        assert_eq!(bayes_opt.total_budget(), 15);
        assert_eq!(bayes_opt.current_iteration(), 0);
        assert!(!bayes_opt.is_complete());
    }

    #[test]
    fn test_bayesian_optimization_suggest_initial() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "lr".to_string(),
            HyperparamSpace::continuous(0.0, 1.0).unwrap(),
        );

        let mut bayes_opt = BayesianOptimization::new(param_space, 5, 3, 42);

        // First suggestions should be random (initial phase)
        for _ in 0..3 {
            let config = bayes_opt.suggest().unwrap();
            assert!(config.contains_key("lr"));

            // Simulate adding a result
            bayes_opt.add_result(HyperparamResult::new(config, 0.5));
        }

        assert_eq!(bayes_opt.current_iteration(), 3);
    }

    #[test]
    fn test_bayesian_optimization_suggest_gp_phase() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "x".to_string(),
            HyperparamSpace::continuous(0.0, 1.0).unwrap(),
        );

        let mut bayes_opt = BayesianOptimization::new(param_space, 5, 2, 42);

        // Add initial observations
        let mut config1 = HashMap::new();
        config1.insert("x".to_string(), HyperparamValue::Float(0.25));
        bayes_opt.add_result(HyperparamResult::new(config1, 0.5));

        let mut config2 = HashMap::new();
        config2.insert("x".to_string(), HyperparamValue::Float(0.75));
        bayes_opt.add_result(HyperparamResult::new(config2, 0.8));

        // Next suggestion should use GP
        let config = bayes_opt.suggest().unwrap();
        assert!(config.contains_key("x"));
    }

    #[test]
    fn test_bayesian_optimization_with_acquisition() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "lr".to_string(),
            HyperparamSpace::log_uniform(1e-4, 1e-1).unwrap(),
        );

        let bayes_opt = BayesianOptimization::new(param_space, 10, 5, 42)
            .with_acquisition(AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 })
            .with_kernel(GpKernel::Matern32 {
                sigma: 1.0,
                length_scale: 0.5,
            })
            .with_noise(1e-5);

        assert!(bayes_opt.total_budget() == 15);
    }

    #[test]
    fn test_bayesian_optimization_best_result() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "x".to_string(),
            HyperparamSpace::continuous(0.0, 1.0).unwrap(),
        );

        let mut bayes_opt = BayesianOptimization::new(param_space, 5, 2, 42);

        let mut config1 = HashMap::new();
        config1.insert("x".to_string(), HyperparamValue::Float(0.3));
        bayes_opt.add_result(HyperparamResult::new(config1, 0.6));

        let mut config2 = HashMap::new();
        config2.insert("x".to_string(), HyperparamValue::Float(0.7));
        bayes_opt.add_result(HyperparamResult::new(config2, 0.9));

        let best = bayes_opt.best_result().unwrap();
        assert_eq!(best.score, 0.9);
    }

    #[test]
    fn test_bayesian_optimization_is_complete() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "x".to_string(),
            HyperparamSpace::continuous(0.0, 1.0).unwrap(),
        );

        let mut bayes_opt = BayesianOptimization::new(param_space, 2, 1, 42);

        assert!(!bayes_opt.is_complete());

        // Add results up to budget
        for i in 0..3 {
            let mut config = HashMap::new();
            config.insert("x".to_string(), HyperparamValue::Float(i as f64 * 0.3));
            bayes_opt.add_result(HyperparamResult::new(config, i as f64 * 0.2));
        }

        assert!(bayes_opt.is_complete());
    }

    #[test]
    fn test_bayesian_optimization_multivariate() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "lr".to_string(),
            HyperparamSpace::log_uniform(1e-4, 1e-1).unwrap(),
        );
        param_space.insert(
            "batch_size".to_string(),
            HyperparamSpace::int_range(16, 128).unwrap(),
        );
        param_space.insert(
            "dropout".to_string(),
            HyperparamSpace::continuous(0.0, 0.5).unwrap(),
        );

        let mut bayes_opt = BayesianOptimization::new(param_space, 10, 3, 42);

        let config = bayes_opt.suggest().unwrap();
        assert_eq!(config.len(), 3);
        assert!(config.contains_key("lr"));
        assert!(config.contains_key("batch_size"));
        assert!(config.contains_key("dropout"));
    }

    #[test]
    fn test_bayesian_optimization_discrete_space() {
        let mut param_space = HashMap::new();
        param_space.insert(
            "optimizer".to_string(),
            HyperparamSpace::discrete(vec![
                HyperparamValue::String("adam".to_string()),
                HyperparamValue::String("sgd".to_string()),
                HyperparamValue::String("rmsprop".to_string()),
            ])
            .unwrap(),
        );

        let mut bayes_opt = BayesianOptimization::new(param_space, 5, 2, 42);

        let config = bayes_opt.suggest().unwrap();
        assert!(config.contains_key("optimizer"));

        let optimizer = config.get("optimizer").unwrap();
        assert!(matches!(optimizer, HyperparamValue::String(_)));
    }

    #[test]
    fn test_normal_cdf() {
        // Test standard normal CDF at common points
        let cdf_0 = BayesianOptimization::normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 1e-4);

        let cdf_pos = BayesianOptimization::normal_cdf(1.96);
        assert!((cdf_pos - 0.975).abs() < 1e-2);

        let cdf_neg = BayesianOptimization::normal_cdf(-1.96);
        assert!((cdf_neg - 0.025).abs() < 1e-2);
    }

    #[test]
    fn test_normal_pdf() {
        // Test standard normal PDF at 0
        let pdf_0 = BayesianOptimization::normal_pdf(0.0);
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((pdf_0 - expected).abs() < 1e-6);

        // PDF should be symmetric
        let pdf_pos = BayesianOptimization::normal_pdf(1.0);
        let pdf_neg = BayesianOptimization::normal_pdf(-1.0);
        assert!((pdf_pos - pdf_neg).abs() < 1e-10);
    }

    #[test]
    fn test_erf() {
        // Test error function at known points
        assert!((BayesianOptimization::erf(0.0) - 0.0).abs() < 1e-6);
        assert!((BayesianOptimization::erf(1.0) - 0.8427).abs() < 1e-3);
        assert!((BayesianOptimization::erf(-1.0) + 0.8427).abs() < 1e-3);
    }
}
