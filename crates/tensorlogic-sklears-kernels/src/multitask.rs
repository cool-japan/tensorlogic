//! Multi-task kernel learning for related tasks with shared representations.
//!
//! This module provides kernels that can learn multiple related tasks simultaneously,
//! sharing information between tasks through structured covariance matrices.
//!
//! ## Features
//!
//! - **TaskKernel** - Wraps any kernel with task indices
//! - **ICMKernel** - Intrinsic Coregionalization Model (B ⊗ K)
//! - **LMCKernel** - Linear Model of Coregionalization (Σ B_q ⊗ K_q)
//! - **IndexKernel** - Purely task-based similarity
//!
//! ## Use Cases
//!
//! - Multi-output regression
//! - Transfer learning between related domains
//! - Hierarchical task structures
//! - Heterogeneous data fusion

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// Input for multi-task kernels: features + task index.
#[derive(Debug, Clone)]
pub struct TaskInput {
    /// Feature vector
    pub features: Vec<f64>,
    /// Task index (0-based)
    pub task: usize,
}

impl TaskInput {
    /// Create a new task input.
    pub fn new(features: Vec<f64>, task: usize) -> Self {
        Self { features, task }
    }

    /// Create from slice with task index.
    pub fn from_slice(features: &[f64], task: usize) -> Self {
        Self {
            features: features.to_vec(),
            task,
        }
    }
}

/// Configuration for multi-task kernels.
#[derive(Debug, Clone)]
pub struct MultiTaskConfig {
    /// Number of tasks
    pub num_tasks: usize,
    /// Whether to normalize task covariance matrix
    pub normalize: bool,
}

impl MultiTaskConfig {
    /// Create configuration with specified number of tasks.
    pub fn new(num_tasks: usize) -> Self {
        Self {
            num_tasks,
            normalize: false,
        }
    }

    /// Enable normalization.
    pub fn with_normalization(mut self) -> Self {
        self.normalize = true;
        self
    }
}

/// Index kernel: K(i, j) = B[i, j] where B is task covariance matrix.
///
/// Pure task-based similarity without feature component.
/// Useful as a building block for more complex multi-task kernels.
#[derive(Debug, Clone)]
pub struct IndexKernel {
    /// Task covariance matrix (num_tasks x num_tasks)
    task_covariance: Vec<Vec<f64>>,
    /// Number of tasks
    num_tasks: usize,
}

impl IndexKernel {
    /// Create an index kernel from task covariance matrix.
    ///
    /// The covariance matrix should be symmetric positive semi-definite.
    pub fn new(task_covariance: Vec<Vec<f64>>) -> Result<Self> {
        let num_tasks = task_covariance.len();
        if num_tasks == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "task_covariance".to_string(),
                value: "empty".to_string(),
                reason: "must have at least one task".to_string(),
            });
        }

        // Validate square matrix
        for (i, row) in task_covariance.iter().enumerate() {
            if row.len() != num_tasks {
                return Err(KernelError::InvalidParameter {
                    parameter: "task_covariance".to_string(),
                    value: format!("row {} has {} elements", i, row.len()),
                    reason: format!("expected {} elements (square matrix)", num_tasks),
                });
            }
        }

        Ok(Self {
            task_covariance,
            num_tasks,
        })
    }

    /// Create with identity covariance (independent tasks).
    pub fn identity(num_tasks: usize) -> Result<Self> {
        let mut cov = vec![vec![0.0; num_tasks]; num_tasks];
        for (i, row) in cov.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Self::new(cov)
    }

    /// Create with uniform covariance (all tasks equally similar).
    pub fn uniform(num_tasks: usize, correlation: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&correlation) {
            return Err(KernelError::InvalidParameter {
                parameter: "correlation".to_string(),
                value: correlation.to_string(),
                reason: "must be in [0, 1]".to_string(),
            });
        }

        let mut cov = vec![vec![correlation; num_tasks]; num_tasks];
        for (i, row) in cov.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Self::new(cov)
    }

    /// Get task covariance value.
    pub fn get_task_covariance(&self, task_i: usize, task_j: usize) -> Result<f64> {
        if task_i >= self.num_tasks || task_j >= self.num_tasks {
            return Err(KernelError::ComputationError(format!(
                "Task index out of bounds: ({}, {}) for {} tasks",
                task_i, task_j, self.num_tasks
            )));
        }
        Ok(self.task_covariance[task_i][task_j])
    }

    /// Get number of tasks.
    pub fn num_tasks(&self) -> usize {
        self.num_tasks
    }

    /// Get the full covariance matrix.
    pub fn covariance_matrix(&self) -> &Vec<Vec<f64>> {
        &self.task_covariance
    }
}

/// Intrinsic Coregionalization Model (ICM) kernel.
///
/// K((x, i), (y, j)) = B[i, j] * k(x, y)
///
/// where:
/// - B is the task covariance matrix (num_tasks x num_tasks)
/// - k is the base kernel on features
///
/// This model assumes all tasks share the same underlying kernel
/// but with different task-specific scales captured in B.
pub struct ICMKernel {
    /// Base kernel for features
    base_kernel: Box<dyn Kernel>,
    /// Task covariance/similarity matrix
    task_covariance: Vec<Vec<f64>>,
    /// Number of tasks
    num_tasks: usize,
}

impl ICMKernel {
    /// Create a new ICM kernel.
    ///
    /// # Arguments
    /// * `base_kernel` - Kernel for feature similarity
    /// * `task_covariance` - Positive semi-definite task covariance matrix
    pub fn new(base_kernel: Box<dyn Kernel>, task_covariance: Vec<Vec<f64>>) -> Result<Self> {
        let num_tasks = task_covariance.len();
        if num_tasks == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "task_covariance".to_string(),
                value: "empty".to_string(),
                reason: "must have at least one task".to_string(),
            });
        }

        // Validate square matrix
        for (i, row) in task_covariance.iter().enumerate() {
            if row.len() != num_tasks {
                return Err(KernelError::InvalidParameter {
                    parameter: "task_covariance".to_string(),
                    value: format!("row {} has {} elements", i, row.len()),
                    reason: format!("expected {} elements", num_tasks),
                });
            }
        }

        Ok(Self {
            base_kernel,
            task_covariance,
            num_tasks,
        })
    }

    /// Create ICM with identity task covariance (independent tasks).
    pub fn independent(base_kernel: Box<dyn Kernel>, num_tasks: usize) -> Result<Self> {
        let mut cov = vec![vec![0.0; num_tasks]; num_tasks];
        for (i, row) in cov.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Self::new(base_kernel, cov)
    }

    /// Create ICM with uniform task correlation.
    pub fn uniform(
        base_kernel: Box<dyn Kernel>,
        num_tasks: usize,
        correlation: f64,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&correlation) {
            return Err(KernelError::InvalidParameter {
                parameter: "correlation".to_string(),
                value: correlation.to_string(),
                reason: "must be in [0, 1]".to_string(),
            });
        }

        let mut cov = vec![vec![correlation; num_tasks]; num_tasks];
        for (i, row) in cov.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Self::new(base_kernel, cov)
    }

    /// Create ICM from rank-1 decomposition B = v * v^T.
    ///
    /// This is useful when you have task-specific variances.
    pub fn from_rank1(base_kernel: Box<dyn Kernel>, task_variances: Vec<f64>) -> Result<Self> {
        let num_tasks = task_variances.len();
        let mut cov = vec![vec![0.0; num_tasks]; num_tasks];
        for i in 0..num_tasks {
            for j in 0..num_tasks {
                cov[i][j] = task_variances[i].sqrt() * task_variances[j].sqrt();
            }
        }
        Self::new(base_kernel, cov)
    }

    /// Compute ICM kernel value for task inputs.
    pub fn compute_tasks(&self, x: &TaskInput, y: &TaskInput) -> Result<f64> {
        if x.task >= self.num_tasks || y.task >= self.num_tasks {
            return Err(KernelError::ComputationError(format!(
                "Task index out of bounds: ({}, {}) for {} tasks",
                x.task, y.task, self.num_tasks
            )));
        }

        let k_features = self.base_kernel.compute(&x.features, &y.features)?;
        let b_tasks = self.task_covariance[x.task][y.task];

        Ok(b_tasks * k_features)
    }

    /// Get number of tasks.
    pub fn num_tasks(&self) -> usize {
        self.num_tasks
    }

    /// Get task covariance matrix.
    pub fn task_covariance(&self) -> &Vec<Vec<f64>> {
        &self.task_covariance
    }

    /// Compute full kernel matrix for multiple task inputs.
    pub fn compute_task_matrix(&self, inputs: &[TaskInput]) -> Result<Vec<Vec<f64>>> {
        let n = inputs.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                let k = self.compute_tasks(&inputs[i], &inputs[j])?;
                matrix[i][j] = k;
                matrix[j][i] = k;
            }
        }

        Ok(matrix)
    }
}

/// A single latent process component for LMC.
struct LMCComponent {
    /// Base kernel for this component
    kernel: Box<dyn Kernel>,
    /// Task covariance matrix for this component
    task_covariance: Vec<Vec<f64>>,
}

/// Linear Model of Coregionalization (LMC) kernel.
///
/// K((x, i), (y, j)) = Σ_q B_q[i, j] * k_q(x, y)
///
/// where:
/// - Each (B_q, k_q) pair represents a latent process
/// - B_q is a task covariance matrix
/// - k_q is a kernel function
///
/// LMC is more expressive than ICM as it allows different
/// kernels to capture different aspects of task relationships.
pub struct LMCKernel {
    /// Latent process components
    components: Vec<LMCComponent>,
    /// Number of tasks
    num_tasks: usize,
}

impl LMCKernel {
    /// Create a new LMC kernel.
    pub fn new(num_tasks: usize) -> Self {
        Self {
            components: Vec::new(),
            num_tasks,
        }
    }

    /// Add a latent process component.
    pub fn add_component(
        &mut self,
        kernel: Box<dyn Kernel>,
        task_covariance: Vec<Vec<f64>>,
    ) -> Result<()> {
        // Validate task covariance dimensions
        if task_covariance.len() != self.num_tasks {
            return Err(KernelError::InvalidParameter {
                parameter: "task_covariance".to_string(),
                value: format!("{} rows", task_covariance.len()),
                reason: format!("expected {} rows", self.num_tasks),
            });
        }

        for (i, row) in task_covariance.iter().enumerate() {
            if row.len() != self.num_tasks {
                return Err(KernelError::InvalidParameter {
                    parameter: "task_covariance".to_string(),
                    value: format!("row {} has {} elements", i, row.len()),
                    reason: format!("expected {} elements", self.num_tasks),
                });
            }
        }

        self.components.push(LMCComponent {
            kernel,
            task_covariance,
        });

        Ok(())
    }

    /// Compute LMC kernel value for task inputs.
    pub fn compute_tasks(&self, x: &TaskInput, y: &TaskInput) -> Result<f64> {
        if x.task >= self.num_tasks || y.task >= self.num_tasks {
            return Err(KernelError::ComputationError(format!(
                "Task index out of bounds: ({}, {}) for {} tasks",
                x.task, y.task, self.num_tasks
            )));
        }

        let mut result = 0.0;
        for component in &self.components {
            let k_features = component.kernel.compute(&x.features, &y.features)?;
            let b_tasks = component.task_covariance[x.task][y.task];
            result += b_tasks * k_features;
        }

        Ok(result)
    }

    /// Get number of components.
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Get number of tasks.
    pub fn num_tasks(&self) -> usize {
        self.num_tasks
    }

    /// Compute full kernel matrix for multiple task inputs.
    pub fn compute_task_matrix(&self, inputs: &[TaskInput]) -> Result<Vec<Vec<f64>>> {
        let n = inputs.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                let k = self.compute_tasks(&inputs[i], &inputs[j])?;
                matrix[i][j] = k;
                matrix[j][i] = k;
            }
        }

        Ok(matrix)
    }
}

/// Wrapper to use ICM kernel with standard Kernel trait.
///
/// Encodes task index in the first element of the input vector.
pub struct ICMKernelWrapper {
    inner: ICMKernel,
}

impl ICMKernelWrapper {
    /// Create wrapper from ICM kernel.
    pub fn new(inner: ICMKernel) -> Self {
        Self { inner }
    }

    /// Get the inner ICM kernel.
    pub fn inner(&self) -> &ICMKernel {
        &self.inner
    }
}

impl Kernel for ICMKernelWrapper {
    /// Compute kernel where first element is task index.
    ///
    /// Input format: [task_index, feature_1, feature_2, ...]
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.is_empty() || y.is_empty() {
            return Err(KernelError::ComputationError(
                "Input must have at least task index".to_string(),
            ));
        }

        let task_x = x[0] as usize;
        let task_y = y[0] as usize;
        let features_x = &x[1..];
        let features_y = &y[1..];

        let input_x = TaskInput::from_slice(features_x, task_x);
        let input_y = TaskInput::from_slice(features_y, task_y);

        self.inner.compute_tasks(&input_x, &input_y)
    }

    fn name(&self) -> &str {
        "ICM"
    }
}

/// Wrapper to use LMC kernel with standard Kernel trait.
pub struct LMCKernelWrapper {
    inner: LMCKernel,
}

impl LMCKernelWrapper {
    /// Create wrapper from LMC kernel.
    pub fn new(inner: LMCKernel) -> Self {
        Self { inner }
    }

    /// Get the inner LMC kernel.
    pub fn inner(&self) -> &LMCKernel {
        &self.inner
    }
}

impl Kernel for LMCKernelWrapper {
    /// Compute kernel where first element is task index.
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.is_empty() || y.is_empty() {
            return Err(KernelError::ComputationError(
                "Input must have at least task index".to_string(),
            ));
        }

        let task_x = x[0] as usize;
        let task_y = y[0] as usize;
        let features_x = &x[1..];
        let features_y = &y[1..];

        let input_x = TaskInput::from_slice(features_x, task_x);
        let input_y = TaskInput::from_slice(features_y, task_y);

        self.inner.compute_tasks(&input_x, &input_y)
    }

    fn name(&self) -> &str {
        "LMC"
    }
}

/// Hadamard (element-wise) product of multiple task kernels.
///
/// K((x, i), (y, j)) = Π_q K_q((x, i), (y, j))
///
/// Useful for combining different aspects of task similarity.
pub struct HadamardTaskKernel {
    /// Component kernels
    kernels: Vec<ICMKernel>,
}

impl HadamardTaskKernel {
    /// Create a new Hadamard task kernel.
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
        }
    }

    /// Add a component kernel.
    pub fn add_kernel(&mut self, kernel: ICMKernel) -> Result<()> {
        if !self.kernels.is_empty() && kernel.num_tasks() != self.kernels[0].num_tasks() {
            return Err(KernelError::InvalidParameter {
                parameter: "num_tasks".to_string(),
                value: kernel.num_tasks().to_string(),
                reason: format!("expected {}", self.kernels[0].num_tasks()),
            });
        }
        self.kernels.push(kernel);
        Ok(())
    }

    /// Compute kernel value.
    pub fn compute_tasks(&self, x: &TaskInput, y: &TaskInput) -> Result<f64> {
        if self.kernels.is_empty() {
            return Err(KernelError::ComputationError(
                "No component kernels added".to_string(),
            ));
        }

        let mut result = 1.0;
        for kernel in &self.kernels {
            result *= kernel.compute_tasks(x, y)?;
        }
        Ok(result)
    }

    /// Get number of tasks.
    pub fn num_tasks(&self) -> Option<usize> {
        self.kernels.first().map(|k| k.num_tasks())
    }
}

impl Default for HadamardTaskKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating multi-task kernels.
pub struct MultiTaskKernelBuilder {
    num_tasks: usize,
    base_kernels: Vec<Box<dyn Kernel>>,
    task_covariances: Vec<Vec<Vec<f64>>>,
}

impl MultiTaskKernelBuilder {
    /// Create a new builder.
    pub fn new(num_tasks: usize) -> Self {
        Self {
            num_tasks,
            base_kernels: Vec::new(),
            task_covariances: Vec::new(),
        }
    }

    /// Add a component with its kernel and task covariance.
    pub fn add_component(
        mut self,
        kernel: Box<dyn Kernel>,
        task_covariance: Vec<Vec<f64>>,
    ) -> Self {
        self.base_kernels.push(kernel);
        self.task_covariances.push(task_covariance);
        self
    }

    /// Build an ICM kernel (single component).
    pub fn build_icm(self) -> Result<ICMKernel> {
        if self.base_kernels.len() != 1 {
            return Err(KernelError::InvalidParameter {
                parameter: "components".to_string(),
                value: self.base_kernels.len().to_string(),
                reason: "ICM requires exactly one component".to_string(),
            });
        }

        let kernel = self.base_kernels.into_iter().next().unwrap();
        let cov = self.task_covariances.into_iter().next().unwrap();
        ICMKernel::new(kernel, cov)
    }

    /// Build an LMC kernel (multiple components).
    pub fn build_lmc(self) -> Result<LMCKernel> {
        let mut lmc = LMCKernel::new(self.num_tasks);

        for (kernel, cov) in self.base_kernels.into_iter().zip(self.task_covariances) {
            lmc.add_component(kernel, cov)?;
        }

        Ok(lmc)
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::{LinearKernel, RbfKernel, RbfKernelConfig};

    // ===== IndexKernel Tests =====

    #[test]
    fn test_index_kernel_basic() {
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let kernel = IndexKernel::new(cov).unwrap();

        assert_eq!(kernel.num_tasks(), 2);
        assert!((kernel.get_task_covariance(0, 1).unwrap() - 0.5).abs() < 1e-10);
        assert!((kernel.get_task_covariance(1, 1).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_index_kernel_identity() {
        let kernel = IndexKernel::identity(3).unwrap();

        assert!((kernel.get_task_covariance(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((kernel.get_task_covariance(0, 1).unwrap()).abs() < 1e-10);
        assert!((kernel.get_task_covariance(1, 2).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_index_kernel_uniform() {
        let kernel = IndexKernel::uniform(3, 0.5).unwrap();

        assert!((kernel.get_task_covariance(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((kernel.get_task_covariance(0, 1).unwrap() - 0.5).abs() < 1e-10);
        assert!((kernel.get_task_covariance(1, 2).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_index_kernel_invalid() {
        // Empty
        let result = IndexKernel::new(vec![]);
        assert!(result.is_err());

        // Non-square
        let result = IndexKernel::new(vec![vec![1.0, 0.5]]);
        assert!(result.is_err());

        // Invalid correlation
        let result = IndexKernel::uniform(3, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_kernel_out_of_bounds() {
        let kernel = IndexKernel::identity(2).unwrap();
        assert!(kernel.get_task_covariance(2, 0).is_err());
    }

    // ===== ICMKernel Tests =====

    #[test]
    fn test_icm_kernel_basic() {
        let base = LinearKernel::new();
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let icm = ICMKernel::new(Box::new(base), cov).unwrap();

        assert_eq!(icm.num_tasks(), 2);
    }

    #[test]
    fn test_icm_kernel_compute() {
        let base = LinearKernel::new();
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let icm = ICMKernel::new(Box::new(base), cov).unwrap();

        let x = TaskInput::new(vec![1.0, 2.0], 0);
        let y = TaskInput::new(vec![3.0, 4.0], 1);

        let k = icm.compute_tasks(&x, &y).unwrap();
        // Linear: 1*3 + 2*4 = 11
        // Task covariance: 0.5
        // Result: 0.5 * 11 = 5.5
        assert!((k - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_icm_kernel_same_task() {
        let base = LinearKernel::new();
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let icm = ICMKernel::new(Box::new(base), cov).unwrap();

        let x = TaskInput::new(vec![1.0, 2.0], 0);
        let y = TaskInput::new(vec![3.0, 4.0], 0);

        let k = icm.compute_tasks(&x, &y).unwrap();
        // Linear: 11, Task: 1.0, Result: 11.0
        assert!((k - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_icm_kernel_independent() {
        let base = LinearKernel::new();
        let icm = ICMKernel::independent(Box::new(base), 3).unwrap();

        let x = TaskInput::new(vec![1.0], 0);
        let y = TaskInput::new(vec![1.0], 1);

        // Different tasks, independent => 0
        let k = icm.compute_tasks(&x, &y).unwrap();
        assert!(k.abs() < 1e-10);

        // Same task => base kernel value
        let z = TaskInput::new(vec![1.0], 0);
        let k = icm.compute_tasks(&x, &z).unwrap();
        assert!((k - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_icm_kernel_uniform() {
        let base = LinearKernel::new();
        let icm = ICMKernel::uniform(Box::new(base), 2, 0.8).unwrap();

        let x = TaskInput::new(vec![1.0], 0);
        let y = TaskInput::new(vec![1.0], 1);

        let k = icm.compute_tasks(&x, &y).unwrap();
        // Linear: 1.0, Task: 0.8, Result: 0.8
        assert!((k - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_icm_kernel_rank1() {
        let base = LinearKernel::new();
        let variances = vec![1.0, 4.0]; // sqrt gives [1.0, 2.0]
        let icm = ICMKernel::from_rank1(Box::new(base), variances).unwrap();

        // B[0,1] = sqrt(1) * sqrt(4) = 2.0
        let x = TaskInput::new(vec![1.0], 0);
        let y = TaskInput::new(vec![1.0], 1);

        let k = icm.compute_tasks(&x, &y).unwrap();
        assert!((k - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_icm_kernel_matrix() {
        let base = LinearKernel::new();
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let icm = ICMKernel::new(Box::new(base), cov).unwrap();

        let inputs = vec![
            TaskInput::new(vec![1.0], 0),
            TaskInput::new(vec![1.0], 1),
            TaskInput::new(vec![2.0], 0),
        ];

        let matrix = icm.compute_task_matrix(&inputs).unwrap();

        assert_eq!(matrix.len(), 3);
        // Symmetric
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
    fn test_icm_kernel_invalid_task() {
        let base = LinearKernel::new();
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let icm = ICMKernel::new(Box::new(base), cov).unwrap();

        let x = TaskInput::new(vec![1.0], 0);
        let y = TaskInput::new(vec![1.0], 5); // Out of bounds

        assert!(icm.compute_tasks(&x, &y).is_err());
    }

    // ===== LMCKernel Tests =====

    #[test]
    fn test_lmc_kernel_basic() {
        let mut lmc = LMCKernel::new(2);

        let base1 = LinearKernel::new();
        let cov1 = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        lmc.add_component(Box::new(base1), cov1).unwrap();

        assert_eq!(lmc.num_tasks(), 2);
        assert_eq!(lmc.num_components(), 1);
    }

    #[test]
    fn test_lmc_kernel_compute() {
        let mut lmc = LMCKernel::new(2);

        // Component 1: Linear with correlation
        let base1 = LinearKernel::new();
        let cov1 = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        lmc.add_component(Box::new(base1), cov1).unwrap();

        // Component 2: RBF with different correlation
        let base2 = RbfKernel::new(RbfKernelConfig::new(1.0)).unwrap();
        let cov2 = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        lmc.add_component(Box::new(base2), cov2).unwrap();

        let x = TaskInput::new(vec![1.0, 0.0], 0);
        let y = TaskInput::new(vec![1.0, 0.0], 1);

        let k = lmc.compute_tasks(&x, &y).unwrap();
        // Linear: 1.0, Task cov1: 0.5 => 0.5
        // RBF: 1.0, Task cov2: 1.0 => 1.0
        // Sum: 1.5
        assert!((k - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_lmc_kernel_matrix() {
        let mut lmc = LMCKernel::new(2);

        let base = LinearKernel::new();
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        lmc.add_component(Box::new(base), cov).unwrap();

        let inputs = vec![TaskInput::new(vec![1.0], 0), TaskInput::new(vec![1.0], 1)];

        let matrix = lmc.compute_task_matrix(&inputs).unwrap();
        assert_eq!(matrix.len(), 2);

        // Check symmetry
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_lmc_kernel_invalid_dimensions() {
        let mut lmc = LMCKernel::new(2);

        let base = LinearKernel::new();
        let cov = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];

        // Wrong number of tasks
        assert!(lmc.add_component(Box::new(base), cov).is_err());
    }

    // ===== Wrapper Tests =====

    #[test]
    fn test_icm_wrapper() {
        let base = LinearKernel::new();
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let icm = ICMKernel::new(Box::new(base), cov).unwrap();
        let wrapper = ICMKernelWrapper::new(icm);

        // [task, features...]
        let x = vec![0.0, 1.0, 2.0]; // Task 0
        let y = vec![1.0, 3.0, 4.0]; // Task 1

        let k = wrapper.compute(&x, &y).unwrap();
        // Linear: 11, Task: 0.5, Result: 5.5
        assert!((k - 5.5).abs() < 1e-10);

        assert_eq!(wrapper.name(), "ICM");
    }

    #[test]
    fn test_lmc_wrapper() {
        let mut lmc = LMCKernel::new(2);
        let base = LinearKernel::new();
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        lmc.add_component(Box::new(base), cov).unwrap();

        let wrapper = LMCKernelWrapper::new(lmc);

        let x = vec![0.0, 1.0]; // Task 0
        let y = vec![1.0, 1.0]; // Task 1

        let k = wrapper.compute(&x, &y).unwrap();
        assert!((k - 0.5).abs() < 1e-10);

        assert_eq!(wrapper.name(), "LMC");
    }

    #[test]
    fn test_wrapper_empty_input() {
        let base = LinearKernel::new();
        let cov = vec![vec![1.0]];
        let icm = ICMKernel::new(Box::new(base), cov).unwrap();
        let wrapper = ICMKernelWrapper::new(icm);

        assert!(wrapper.compute(&[], &[0.0, 1.0]).is_err());
    }

    // ===== HadamardTaskKernel Tests =====

    #[test]
    fn test_hadamard_task_kernel() {
        let mut hadamard = HadamardTaskKernel::new();

        let base1 = LinearKernel::new();
        let cov1 = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let icm1 = ICMKernel::new(Box::new(base1), cov1).unwrap();
        hadamard.add_kernel(icm1).unwrap();

        let base2 = LinearKernel::new();
        let cov2 = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let icm2 = ICMKernel::new(Box::new(base2), cov2).unwrap();
        hadamard.add_kernel(icm2).unwrap();

        let x = TaskInput::new(vec![1.0], 0);
        let y = TaskInput::new(vec![1.0], 1);

        let k = hadamard.compute_tasks(&x, &y).unwrap();
        // ICM1: 0.5 * 1.0 = 0.5
        // ICM2: 1.0 * 1.0 = 1.0
        // Product: 0.5
        assert!((k - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_task_kernel_empty() {
        let hadamard = HadamardTaskKernel::new();
        let x = TaskInput::new(vec![1.0], 0);
        let y = TaskInput::new(vec![1.0], 0);

        assert!(hadamard.compute_tasks(&x, &y).is_err());
    }

    #[test]
    fn test_hadamard_mismatched_tasks() {
        let mut hadamard = HadamardTaskKernel::new();

        let base1 = LinearKernel::new();
        let cov1 = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let icm1 = ICMKernel::new(Box::new(base1), cov1).unwrap();
        hadamard.add_kernel(icm1).unwrap();

        // Different number of tasks
        let base2 = LinearKernel::new();
        let cov2 = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];
        let icm2 = ICMKernel::new(Box::new(base2), cov2).unwrap();

        assert!(hadamard.add_kernel(icm2).is_err());
    }

    // ===== Builder Tests =====

    #[test]
    fn test_builder_icm() {
        let base = LinearKernel::new();
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let icm = MultiTaskKernelBuilder::new(2)
            .add_component(Box::new(base), cov)
            .build_icm()
            .unwrap();

        assert_eq!(icm.num_tasks(), 2);
    }

    #[test]
    fn test_builder_lmc() {
        let base1 = LinearKernel::new();
        let cov1 = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let base2 = RbfKernel::new(RbfKernelConfig::new(1.0)).unwrap();
        let cov2 = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        let lmc = MultiTaskKernelBuilder::new(2)
            .add_component(Box::new(base1), cov1)
            .add_component(Box::new(base2), cov2)
            .build_lmc()
            .unwrap();

        assert_eq!(lmc.num_tasks(), 2);
        assert_eq!(lmc.num_components(), 2);
    }

    #[test]
    fn test_builder_icm_wrong_components() {
        let base1 = LinearKernel::new();
        let cov1 = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let base2 = LinearKernel::new();
        let cov2 = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let result = MultiTaskKernelBuilder::new(2)
            .add_component(Box::new(base1), cov1)
            .add_component(Box::new(base2), cov2)
            .build_icm();

        assert!(result.is_err());
    }

    // ===== Integration Tests =====

    #[test]
    fn test_multitask_with_rbf() {
        let base = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        let cov = vec![
            vec![1.0, 0.8, 0.6],
            vec![0.8, 1.0, 0.7],
            vec![0.6, 0.7, 1.0],
        ];
        let icm = ICMKernel::new(Box::new(base), cov).unwrap();

        // Same point, same task
        let x = TaskInput::new(vec![1.0, 2.0], 0);
        let k = icm.compute_tasks(&x, &x).unwrap();
        assert!((k - 1.0).abs() < 1e-10);

        // Same point, different task
        let y = TaskInput::new(vec![1.0, 2.0], 1);
        let k = icm.compute_tasks(&x, &y).unwrap();
        assert!((k - 0.8).abs() < 1e-10);

        // Different point, same task
        let z = TaskInput::new(vec![1.0, 3.0], 0);
        let k = icm.compute_tasks(&x, &z).unwrap();
        // RBF with distance 1.0: exp(-0.5 * 1) ≈ 0.6065
        assert!(k > 0.5 && k < 0.7);
    }

    #[test]
    fn test_task_input_creation() {
        let input = TaskInput::new(vec![1.0, 2.0, 3.0], 0);
        assert_eq!(input.features, vec![1.0, 2.0, 3.0]);
        assert_eq!(input.task, 0);

        let input = TaskInput::from_slice(&[4.0, 5.0], 2);
        assert_eq!(input.features, vec![4.0, 5.0]);
        assert_eq!(input.task, 2);
    }
}
