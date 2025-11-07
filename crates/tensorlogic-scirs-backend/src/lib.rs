//! SciRS2-backed executor (CPU/SIMD/GPU via features).
//!
//! This crate provides a production-ready implementation of the TensorLogic execution
//! traits using the SciRS2 scientific computing library.
//!
//! ## Core Features
//!
//! ### Execution Engine
//! - **Forward pass**: Tensor operations (einsum, element-wise, reductions)
//! - **Backward pass**: Automatic differentiation with stored intermediate values
//! - **Gradient checking**: Numeric verification for correctness
//! - **Batch execution**: Parallel processing support for multiple inputs
//!
//! ### Performance
//! - **Memory pooling**: Efficient tensor allocation with shape-based reuse
//! - **Operation fusion**: Analysis and optimization opportunities
//! - **SIMD support**: Vectorized operations via feature flags
//! - **Profiling**: Detailed performance monitoring and tracing
//!
//! ### Reliability
//! - **Error handling**: Comprehensive error types with detailed context
//! - **Execution tracing**: Multi-level debugging and operation tracking
//! - **Numerical stability**: Fallback mechanisms for NaN/Inf handling
//! - **Shape validation**: Runtime shape inference and verification
//!
//! ### Testing
//! - **104 tests**: Including unit, integration, and property-based tests
//! - **Property tests**: Mathematical properties verified with proptest
//! - **Gradient tests**: Numeric gradient checking for autodiff correctness
//!
//! ## Module Organization
//!
//! - `executor`: Core Scirs2Exec implementation
//! - `autodiff`: Backward pass and gradient computation
//! - `gradient_ops`: Advanced gradient operations (STE, Gumbel-Softmax, soft quantifiers)
//! - `error`: Comprehensive error types and validation
//! - `fallback`: Numerical stability and NaN/Inf handling
//! - `tracing`: Execution debugging and performance tracking
//! - `memory_pool`: Efficient tensor allocation
//! - `fusion`: Operation fusion analysis
//! - `gradient_check`: Numeric gradient verification
//! - `shape_inference`: Runtime shape validation
//! - `batch_executor`: Parallel batch processing
//! - `profiled_executor`: Performance profiling wrapper
//! - `capabilities`: Runtime capability detection
//! - `dependency_analyzer`: Graph dependency analysis for parallel execution
//! - `parallel_executor`: Multi-threaded parallel execution using Rayon
//! - `device`: Device management (CPU/GPU selection)
//! - `execution_mode`: Execution mode abstractions (Eager/Graph/JIT)
//! - `precision`: Precision control (f32/f64/mixed)

pub(crate) mod autodiff;
pub mod batch_executor;
pub mod capabilities;
mod conversion;
pub mod dependency_analyzer;
pub mod device;
pub(crate) mod einsum_grad;
pub mod error;
pub mod execution_mode;
mod executor;
pub mod fallback;
pub mod fusion;
pub mod gradient_check;
pub mod gradient_ops;
pub mod memory_pool;
mod ops;
pub mod parallel_executor;
pub mod precision;
pub mod profiled_executor;
pub mod shape_inference;
pub mod tracing;

#[cfg(test)]
mod tests;

use scirs2_core::ndarray::ArrayD;

pub type Scirs2Tensor = ArrayD<f64>;

pub use autodiff::ForwardTape;
pub use batch_executor::ParallelBatchExecutor;
pub use dependency_analyzer::{DependencyAnalysis, DependencyStats, OperationDependency};
pub use device::{Device, DeviceError, DeviceManager, DeviceType};
pub use error::{
    NumericalError, NumericalErrorKind, ShapeMismatchError, TlBackendError, TlBackendResult,
};
pub use execution_mode::{
    CompilationStats, CompiledGraph, ExecutionConfig, ExecutionMode, MemoryPlan,
};
pub use executor::Scirs2Exec;
pub use fallback::{is_valid, sanitize_tensor, FallbackConfig};
pub use gradient_ops::{
    gumbel_softmax, gumbel_softmax_backward, soft_exists, soft_exists_backward, soft_forall,
    soft_forall_backward, ste_threshold, ste_threshold_backward, GumbelSoftmaxConfig,
    QuantifierMode, SteConfig,
};
pub use parallel_executor::{ParallelConfig, ParallelScirs2Exec, ParallelStats};
pub use precision::{ComputePrecision, Precision, PrecisionConfig, Scalar};
pub use profiled_executor::ProfiledScirs2Exec;
pub use shape_inference::{validate_tensor_shapes, Scirs2ShapeInference};
pub use tracing::{ExecutionTracer, TraceEvent, TraceLevel};
