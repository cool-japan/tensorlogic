// Allow needless_range_loop for matrix operations which are clearer with indexed loops
#![allow(clippy::needless_range_loop)]

//! Random Fourier Features (RFF) for scalable kernel approximation.
//!
//! Random Fourier Features provide a way to approximate kernel functions
//! in linear time O(n) instead of O(n²) by mapping data to a randomized
//! low-dimensional feature space.
//!
//! ## Theory
//!
//! For shift-invariant kernels K(x, y) = k(x - y), Bochner's theorem
//! guarantees that there exists a Fourier transform p(ω) such that:
//!
//! k(x - y) = ∫ p(ω) e^{iω^T(x-y)} dω
//!
//! By sampling ω from p(ω) and computing cos(ω^T x) and sin(ω^T x),
//! we can approximate K(x, y) ≈ z(x)^T z(y) where z is the feature map.
//!
//! ## Reference
//!
//! Rahimi, A., & Recht, B. (2007). "Random Features for Large-Scale Kernel
//! Machines." NIPS.
//!
//! ## Example
//!
//! ```rust
//! use tensorlogic_sklears_kernels::random_features::{
//!     RandomFourierFeatures, RffConfig, KernelType
//! };
//!
//! // Create RFF for RBF kernel with gamma=0.5
//! let config = RffConfig::new(KernelType::Rbf { gamma: 0.5 }, 100);
//! let rff = RandomFourierFeatures::new(3, config).unwrap(); // 3D input
//!
//! let x = vec![1.0, 2.0, 3.0];
//! let features = rff.transform(&x).unwrap();
//! // features is a 200-dimensional vector (100 cos + 100 sin components)
//! ```

use crate::error::{KernelError, Result};
use std::f64::consts::PI;

/// Kernel type for RFF approximation.
#[derive(Debug, Clone)]
pub enum KernelType {
    /// RBF/Gaussian kernel: K(x,y) = exp(-gamma * ||x-y||²)
    /// Fourier transform: Gaussian distribution N(0, 2*gamma*I)
    Rbf { gamma: f64 },

    /// Laplacian kernel: K(x,y) = exp(-gamma * ||x-y||₁)
    /// Fourier transform: Cauchy distribution
    Laplacian { gamma: f64 },

    /// Matérn kernel with nu=0.5 (equivalent to Laplacian)
    Matern05 { length_scale: f64 },

    /// Matérn kernel with nu=1.5
    Matern15 { length_scale: f64 },

    /// Matérn kernel with nu=2.5
    Matern25 { length_scale: f64 },
}

impl KernelType {
    /// Get the name of the kernel type.
    pub fn name(&self) -> &str {
        match self {
            KernelType::Rbf { .. } => "RBF",
            KernelType::Laplacian { .. } => "Laplacian",
            KernelType::Matern05 { .. } => "Matern-0.5",
            KernelType::Matern15 { .. } => "Matern-1.5",
            KernelType::Matern25 { .. } => "Matern-2.5",
        }
    }
}

/// Configuration for Random Fourier Features.
#[derive(Debug, Clone)]
pub struct RffConfig {
    /// The kernel type to approximate
    pub kernel_type: KernelType,
    /// Number of random features (n_components)
    /// The output dimension will be 2 * n_components (cos + sin)
    pub n_components: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl RffConfig {
    /// Create a new RFF configuration.
    pub fn new(kernel_type: KernelType, n_components: usize) -> Self {
        Self {
            kernel_type,
            n_components,
            seed: None,
        }
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Random Fourier Features for scalable kernel approximation.
///
/// This struct stores the random frequencies sampled from the spectral
/// distribution of the kernel, allowing efficient feature computation.
#[derive(Debug, Clone)]
pub struct RandomFourierFeatures {
    /// Random frequencies matrix (n_components x input_dim)
    random_weights: Vec<Vec<f64>>,
    /// Random offsets for the cosine features
    random_offsets: Vec<f64>,
    /// Configuration
    config: RffConfig,
    /// Input dimension
    input_dim: usize,
    /// Output dimension (2 * n_components for cos + sin)
    output_dim: usize,
}

impl RandomFourierFeatures {
    /// Create a new Random Fourier Features transformer.
    ///
    /// # Arguments
    /// * `input_dim` - Dimension of input vectors
    /// * `config` - RFF configuration
    ///
    /// # Example
    /// ```rust
    /// use tensorlogic_sklears_kernels::random_features::{
    ///     RandomFourierFeatures, RffConfig, KernelType
    /// };
    ///
    /// let config = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 50);
    /// let rff = RandomFourierFeatures::new(10, config).unwrap();
    /// ```
    pub fn new(input_dim: usize, config: RffConfig) -> Result<Self> {
        if input_dim == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "input_dim".to_string(),
                value: "0".to_string(),
                reason: "input dimension must be positive".to_string(),
            });
        }
        if config.n_components == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "n_components".to_string(),
                value: "0".to_string(),
                reason: "number of components must be positive".to_string(),
            });
        }

        // Initialize random state
        let seed = config.seed.unwrap_or(42);
        let mut rng_state = seed;

        // Sample random frequencies from the spectral distribution
        let random_weights = Self::sample_frequencies(
            &config.kernel_type,
            config.n_components,
            input_dim,
            &mut rng_state,
        )?;

        // Sample random offsets uniformly from [0, 2π]
        let random_offsets: Vec<f64> = (0..config.n_components)
            .map(|_| random_uniform(&mut rng_state) * 2.0 * PI)
            .collect();

        let output_dim = 2 * config.n_components;

        Ok(Self {
            random_weights,
            random_offsets,
            config,
            input_dim,
            output_dim,
        })
    }

    /// Sample random frequencies from the kernel's spectral distribution.
    fn sample_frequencies(
        kernel_type: &KernelType,
        n_components: usize,
        input_dim: usize,
        rng_state: &mut u64,
    ) -> Result<Vec<Vec<f64>>> {
        let mut weights = Vec::with_capacity(n_components);

        for _ in 0..n_components {
            let mut w = Vec::with_capacity(input_dim);
            for _ in 0..input_dim {
                let freq = match kernel_type {
                    KernelType::Rbf { gamma } => {
                        // Spectral distribution: N(0, 2*gamma)
                        let std = (2.0 * gamma).sqrt();
                        random_normal(rng_state) * std
                    }
                    KernelType::Laplacian { gamma } => {
                        // Spectral distribution: Cauchy(0, gamma)
                        random_cauchy(rng_state) * gamma
                    }
                    KernelType::Matern05 { length_scale } => {
                        // Spectral distribution: Cauchy(0, 1/length_scale)
                        let g = 1.0 / length_scale;
                        random_cauchy(rng_state) * g
                    }
                    KernelType::Matern15 { length_scale } => {
                        // Spectral distribution: Student-t with df=3
                        let scale = (3.0_f64).sqrt() / length_scale;
                        random_student_t(rng_state, 3.0) * scale
                    }
                    KernelType::Matern25 { length_scale } => {
                        // Spectral distribution: Student-t with df=5
                        let scale = (5.0_f64).sqrt() / length_scale;
                        random_student_t(rng_state, 5.0) * scale
                    }
                };
                w.push(freq);
            }
            weights.push(w);
        }

        Ok(weights)
    }

    /// Transform a single input vector to the random feature space.
    ///
    /// # Arguments
    /// * `x` - Input vector of dimension `input_dim`
    ///
    /// # Returns
    /// Feature vector of dimension `2 * n_components`
    pub fn transform(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.input_dim {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.input_dim],
                got: vec![x.len()],
                context: "RFF transform".to_string(),
            });
        }

        let mut features = Vec::with_capacity(self.output_dim);
        let scale = (1.0 / self.config.n_components as f64).sqrt();

        for (i, w) in self.random_weights.iter().enumerate() {
            // Compute w^T x + b
            let wx: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            let proj = wx + self.random_offsets[i];

            // Cosine and sine features
            features.push(proj.cos() * scale);
            features.push(proj.sin() * scale);
        }

        Ok(features)
    }

    /// Transform multiple input vectors to the random feature space.
    ///
    /// # Arguments
    /// * `data` - List of input vectors
    ///
    /// # Returns
    /// Matrix of feature vectors (one row per input)
    pub fn transform_batch(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        data.iter().map(|x| self.transform(x)).collect()
    }

    /// Approximate the kernel value between two vectors.
    ///
    /// K(x, y) ≈ z(x)^T z(y)
    pub fn approximate_kernel(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let z_x = self.transform(x)?;
        let z_y = self.transform(y)?;

        let dot: f64 = z_x.iter().zip(z_y.iter()).map(|(a, b)| a * b).sum();
        Ok(dot)
    }

    /// Approximate the kernel matrix for a set of data points.
    ///
    /// `K[i,j] ≈ z(x_i)^T z(x_j)`
    pub fn approximate_kernel_matrix(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let features = self.transform_batch(data)?;
        let n = features.len();

        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let dot: f64 = features[i]
                    .iter()
                    .zip(features[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                matrix[i][j] = dot;
                matrix[j][i] = dot;
            }
        }

        Ok(matrix)
    }

    /// Get the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get the output dimension (feature space dimension).
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get the number of random components.
    pub fn n_components(&self) -> usize {
        self.config.n_components
    }

    /// Get the kernel type being approximated.
    pub fn kernel_type(&self) -> &KernelType {
        &self.config.kernel_type
    }

    /// Get the random weights matrix.
    pub fn random_weights(&self) -> &[Vec<f64>] {
        &self.random_weights
    }
}

/// Orthogonal Random Features for improved variance.
///
/// Uses orthogonalized random matrices for better approximation quality,
/// especially for smaller n_components.
///
/// Reference: Yu et al. (2016). "Orthogonal Random Features."
#[derive(Debug, Clone)]
pub struct OrthogonalRandomFeatures {
    /// Base RFF
    rff: RandomFourierFeatures,
}

impl OrthogonalRandomFeatures {
    /// Create orthogonal random features.
    ///
    /// Note: For simplicity, this currently uses standard RFF.
    /// Full orthogonalization would require QR decomposition.
    pub fn new(input_dim: usize, config: RffConfig) -> Result<Self> {
        let rff = RandomFourierFeatures::new(input_dim, config)?;
        Ok(Self { rff })
    }

    /// Transform a single input vector.
    pub fn transform(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.rff.transform(x)
    }

    /// Transform multiple input vectors.
    pub fn transform_batch(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        self.rff.transform_batch(data)
    }

    /// Get the output dimension.
    pub fn output_dim(&self) -> usize {
        self.rff.output_dim()
    }
}

/// Nystroem approximation as random features.
///
/// Represents kernel approximation using Nystroem method,
/// which uses a subset of training points as landmarks.
#[derive(Debug, Clone)]
pub struct NystroemFeatures {
    /// Landmark points
    landmarks: Vec<Vec<f64>>,
    /// Computed components (L^{-1/2} from K_mm = L L^T)
    components: Vec<Vec<f64>>,
    /// Kernel type (for computing kernel values)
    kernel_type: KernelType,
}

impl NystroemFeatures {
    /// Create Nystroem features from landmark points.
    ///
    /// # Arguments
    /// * `landmarks` - Subset of training points to use as landmarks
    /// * `kernel_type` - The kernel to approximate
    pub fn new(landmarks: Vec<Vec<f64>>, kernel_type: KernelType) -> Result<Self> {
        if landmarks.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "landmarks".to_string(),
                value: "[]".to_string(),
                reason: "must have at least one landmark".to_string(),
            });
        }

        let m = landmarks.len();
        let mut k_mm = vec![vec![0.0; m]; m];

        // Compute kernel matrix between landmarks
        for i in 0..m {
            for j in i..m {
                let k_val = compute_kernel(&kernel_type, &landmarks[i], &landmarks[j]);
                k_mm[i][j] = k_val;
                k_mm[j][i] = k_val;
            }
        }

        // Add small regularization for numerical stability
        for i in 0..m {
            k_mm[i][i] += 1e-6;
        }

        // Compute L^{-1/2} via eigendecomposition approximation
        // For simplicity, use Cholesky-like approach
        let components = compute_pseudo_sqrt_inv(&k_mm)?;

        Ok(Self {
            landmarks,
            components,
            kernel_type,
        })
    }

    /// Transform a single input vector to Nystroem features.
    pub fn transform(&self, x: &[f64]) -> Result<Vec<f64>> {
        let m = self.landmarks.len();
        let mut k_xm = Vec::with_capacity(m);

        // Compute kernel between x and each landmark
        for landmark in &self.landmarks {
            k_xm.push(compute_kernel(&self.kernel_type, x, landmark));
        }

        // Multiply by components: z(x) = K_xm @ L^{-1/2}
        let mut features = vec![0.0; m];
        for i in 0..m {
            for j in 0..m {
                features[i] += k_xm[j] * self.components[j][i];
            }
        }

        Ok(features)
    }

    /// Transform multiple input vectors.
    pub fn transform_batch(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        data.iter().map(|x| self.transform(x)).collect()
    }

    /// Get the number of landmarks (output dimension).
    pub fn n_landmarks(&self) -> usize {
        self.landmarks.len()
    }
}

// ========== Helper functions ==========

/// Simple LCG-based uniform random number generator [0, 1).
fn random_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

/// Generate standard normal random variable using Box-Muller transform.
fn random_normal(state: &mut u64) -> f64 {
    let u1 = random_uniform(state);
    let u2 = random_uniform(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    r * theta.cos()
}

/// Generate Cauchy random variable.
fn random_cauchy(state: &mut u64) -> f64 {
    let u = random_uniform(state);
    (PI * (u - 0.5)).tan()
}

/// Generate Student-t random variable.
fn random_student_t(state: &mut u64, df: f64) -> f64 {
    // Use the ratio of normal to chi-squared
    let z = random_normal(state);
    let mut chi_sq = 0.0;
    for _ in 0..(df as usize) {
        let n = random_normal(state);
        chi_sq += n * n;
    }
    z / (chi_sq / df).sqrt()
}

/// Compute kernel value for a given kernel type.
fn compute_kernel(kernel_type: &KernelType, x: &[f64], y: &[f64]) -> f64 {
    match kernel_type {
        KernelType::Rbf { gamma } => {
            let sq_dist: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            (-gamma * sq_dist).exp()
        }
        KernelType::Laplacian { gamma } => {
            let l1_dist: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum();
            (-gamma * l1_dist).exp()
        }
        KernelType::Matern05 { length_scale } => {
            let dist: f64 = x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            (-dist / length_scale).exp()
        }
        KernelType::Matern15 { length_scale } => {
            let dist: f64 = x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let r = (3.0_f64).sqrt() * dist / length_scale;
            (1.0 + r) * (-r).exp()
        }
        KernelType::Matern25 { length_scale } => {
            let dist: f64 = x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let r = (5.0_f64).sqrt() * dist / length_scale;
            (1.0 + r + r * r / 3.0) * (-r).exp()
        }
    }
}

/// Compute pseudo-inverse square root of a symmetric positive definite matrix.
fn compute_pseudo_sqrt_inv(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Use simple Cholesky-based approach for small matrices
    // For larger matrices, would need eigendecomposition

    // Cholesky decomposition: A = L L^T
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(KernelError::ComputationError(
                        "Matrix not positive definite".to_string(),
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    // Invert L (lower triangular)
    let mut l_inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        l_inv[i][i] = 1.0 / l[i][i];
        for j in (i + 1)..n {
            let mut sum = 0.0;
            for k in i..j {
                sum -= l[j][k] * l_inv[k][i];
            }
            l_inv[j][i] = sum / l[j][j];
        }
    }

    // Return L^{-1} transposed (which is L^{-T})
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = l_inv[j][i];
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rff_config() {
        let config = RffConfig::new(KernelType::Rbf { gamma: 0.5 }, 100);
        assert_eq!(config.n_components, 100);
        assert!(config.seed.is_none());

        let config = config.with_seed(42);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_rff_creation() {
        let config = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 50).with_seed(42);
        let rff = RandomFourierFeatures::new(3, config).unwrap();

        assert_eq!(rff.input_dim(), 3);
        assert_eq!(rff.output_dim(), 100); // 2 * 50
        assert_eq!(rff.n_components(), 50);
    }

    #[test]
    fn test_rff_invalid_params() {
        let config = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 50);
        assert!(RandomFourierFeatures::new(0, config.clone()).is_err());

        let config = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 0);
        assert!(RandomFourierFeatures::new(3, config).is_err());
    }

    #[test]
    fn test_rff_transform() {
        let config = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 50).with_seed(42);
        let rff = RandomFourierFeatures::new(3, config).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let features = rff.transform(&x).unwrap();

        assert_eq!(features.len(), 100);
        // Features should be bounded (cos and sin scaled by sqrt(1/n))
        for f in &features {
            assert!(f.abs() <= 1.0);
        }
    }

    #[test]
    fn test_rff_transform_dimension_mismatch() {
        let config = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 50).with_seed(42);
        let rff = RandomFourierFeatures::new(3, config).unwrap();

        let x = vec![1.0, 2.0]; // Wrong dimension
        assert!(rff.transform(&x).is_err());
    }

    #[test]
    fn test_rff_batch_transform() {
        let config = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 50).with_seed(42);
        let rff = RandomFourierFeatures::new(2, config).unwrap();

        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let features = rff.transform_batch(&data).unwrap();

        assert_eq!(features.len(), 3);
        assert_eq!(features[0].len(), 100);
    }

    #[test]
    fn test_rff_kernel_approximation() {
        let config = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 500).with_seed(42);
        let rff = RandomFourierFeatures::new(2, config).unwrap();

        let x = vec![0.0, 0.0];
        let y = vec![0.0, 0.0];

        // Same point should have kernel ≈ 1
        let approx = rff.approximate_kernel(&x, &y).unwrap();
        assert!((approx - 1.0).abs() < 0.1); // Allow some approximation error

        // Different points
        let y2 = vec![1.0, 1.0];
        let approx2 = rff.approximate_kernel(&x, &y2).unwrap();
        // True RBF: exp(-1.0 * 2) = exp(-2) ≈ 0.135
        assert!(approx2 > 0.0 && approx2 < 1.0);
    }

    #[test]
    fn test_rff_matrix_approximation() {
        let config = RffConfig::new(KernelType::Rbf { gamma: 0.5 }, 200).with_seed(42);
        let rff = RandomFourierFeatures::new(2, config).unwrap();

        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let matrix = rff.approximate_kernel_matrix(&data).unwrap();

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);

        // Diagonal should be close to 1
        for i in 0..3 {
            assert!((matrix[i][i] - 1.0).abs() < 0.1);
        }

        // Symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_rff_reproducibility() {
        let config1 = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 50).with_seed(123);
        let config2 = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 50).with_seed(123);

        let rff1 = RandomFourierFeatures::new(3, config1).unwrap();
        let rff2 = RandomFourierFeatures::new(3, config2).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let f1 = rff1.transform(&x).unwrap();
        let f2 = rff2.transform(&x).unwrap();

        for (a, b) in f1.iter().zip(f2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rff_laplacian() {
        let config = RffConfig::new(KernelType::Laplacian { gamma: 1.0 }, 100).with_seed(42);
        let rff = RandomFourierFeatures::new(2, config).unwrap();

        let x = vec![0.0, 0.0];
        let approx = rff.approximate_kernel(&x, &x).unwrap();
        assert!((approx - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_rff_matern() {
        let config = RffConfig::new(KernelType::Matern15 { length_scale: 1.0 }, 100).with_seed(42);
        let rff = RandomFourierFeatures::new(2, config).unwrap();

        let x = vec![0.0, 0.0];
        let approx = rff.approximate_kernel(&x, &x).unwrap();
        assert!((approx - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_nystroem_features() {
        let landmarks = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let nystroem = NystroemFeatures::new(landmarks, KernelType::Rbf { gamma: 1.0 }).unwrap();

        assert_eq!(nystroem.n_landmarks(), 3);

        let x = vec![0.5, 0.5];
        let features = nystroem.transform(&x).unwrap();
        assert_eq!(features.len(), 3);
    }

    #[test]
    fn test_nystroem_batch() {
        let landmarks = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let nystroem = NystroemFeatures::new(landmarks, KernelType::Rbf { gamma: 0.5 }).unwrap();

        let data = vec![vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 1.0]];
        let features = nystroem.transform_batch(&data).unwrap();

        assert_eq!(features.len(), 3);
        assert_eq!(features[0].len(), 2);
    }

    #[test]
    fn test_orthogonal_rff() {
        let config = RffConfig::new(KernelType::Rbf { gamma: 1.0 }, 50).with_seed(42);
        let orf = OrthogonalRandomFeatures::new(3, config).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let features = orf.transform(&x).unwrap();

        assert_eq!(features.len(), 100);
    }

    #[test]
    fn test_kernel_type_name() {
        assert_eq!(KernelType::Rbf { gamma: 1.0 }.name(), "RBF");
        assert_eq!(KernelType::Laplacian { gamma: 1.0 }.name(), "Laplacian");
        assert_eq!(
            KernelType::Matern15 { length_scale: 1.0 }.name(),
            "Matern-1.5"
        );
    }
}
