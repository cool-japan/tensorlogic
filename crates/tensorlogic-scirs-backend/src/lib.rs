//! SciRS2-backed executor (CPU/SIMD/GPU via features).
//!
//! **Version**: 0.1.3 | **Status**: Production Ready
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
//! - `probabilistic`: Monte Carlo sampling, uncertainty quantification, and mean-field VI

pub mod activations;
pub mod attention;
pub mod attention_grad;
pub(crate) mod autodiff;
pub mod batch_executor;
pub mod blocked_sparse;
pub mod capabilities;
pub mod checkpoint;
pub mod comparison;
mod conversion;
pub mod convolution;
pub mod cuda_detect;
pub mod custom_ops;
pub mod decomposition;
pub mod dependency_analyzer;
pub mod device;
pub mod device_manager;
pub(crate) mod einsum_grad;
pub mod error;
pub mod execution_mode;
mod executor;
pub mod executor_f32;
pub mod fallback;
pub mod fusion;
pub mod gather_scatter;
pub mod geometric_ops;
pub mod gpu_readiness;
pub mod gradient_check;
pub mod gradient_ops;
pub mod graph_optimizer;
pub mod inplace_ops;
pub mod lazy;
pub mod memory_pool;
pub mod memory_profiler;
pub mod metrics;
mod ops;
pub mod parallel_executor;
pub mod pooling;
pub mod precision;
pub mod precision_cast;
pub mod probabilistic;
pub mod profiled_executor;
pub mod quantization;
pub mod recurrent;
pub mod scoring;
pub mod shape_inference;
pub mod signal_ops;
pub mod temporal_ops;
pub mod tensor_io;
pub mod tensor_loss;
pub mod tracing;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::{
    create_gpu_backend, CudaStub, GpuBackend, GpuBuffer, GpuDevice, GpuError, GpuMemoryPool,
    KernelConfig,
};

#[cfg(feature = "torsh")]
pub mod torsh_interop;

#[cfg(test)]
mod tests;

use scirs2_core::ndarray::ArrayD;

pub type Scirs2Tensor = ArrayD<f64>;

pub use activations::{
    elu, gelu, gelu_approx, gelu_scalar, hardsigmoid, hardswish, leaky_relu, log_softmax, mish,
    prelu, relu, relu6, relu_grad, relu_scalar, selu, sigmoid, sigmoid_grad, sigmoid_scalar, silu,
    softmax, softplus, softsign, swish, swish_scalar, tanh_activation, tanh_grad,
    ActivationBenchmark, ActivationError, ActivationType,
};
pub use attention::{
    attention_entropy, chunked_attention, scaled_dot_product_attention, stable_softmax,
    AttentionConfig, AttentionError, AttentionOutput, MultiHeadAttention,
};
pub use attention_grad::{
    attention_backward, multihead_attention_backward, softmax_backward, AttentionGradients,
    MultiHeadAttentionGrad,
};
pub use autodiff::ForwardTape;
pub use batch_executor::ParallelBatchExecutor;
pub use blocked_sparse::{
    blocked_sparse_add, blocked_sparse_dense_mm, blocked_sparse_mm, blocked_sparse_scale,
    BlockSparsityPattern, BlockSparsityStats, BlockedSparseDynTensor, BlockedSparseError,
    BlockedSparseTensor,
};
pub use checkpoint::{Checkpoint, CheckpointConfig, CheckpointManager, CheckpointMetadata};
pub use comparison::{
    abs_diff, assert_tensors_close, compare_tensors, count_non_finite, is_finite, ComparisonError,
    ComparisonResult, Tolerance,
};
pub use convolution::{
    col2im, conv1d, conv2d, conv_transpose2d, depthwise_conv2d, im2col, ConvConfig, ConvError,
    ConvStats,
};
pub use cuda_detect::{
    cuda_device_count, cuda_devices_to_device_list, detect_cuda_devices, is_cuda_available,
    CudaDeviceInfo,
};
pub use custom_ops::{
    BinaryCustomOp, CustomOp, CustomOpContext, EluOp, GeluOp, HardSigmoidOp, HardSwishOp,
    LeakyReluOp, MishOp, OpRegistry, SoftplusOp, SwishOp,
};
pub use decomposition::{
    cp_als, fold, hosvd, truncated_svd, tucker1, unfold, CpDecomposition, DecompositionError,
    HosvdResult, TruncatedSvd, Tucker1Result,
};
pub use dependency_analyzer::{DependencyAnalysis, DependencyStats, OperationDependency};
pub use device::{Device, DeviceError, DeviceType, SystemDeviceManager};
pub use device_manager::{
    DeviceConfig, DeviceManager, DeviceSelector, HeuristicSelector, OpDescriptor, OpKind,
};
pub use error::{
    NumericalError, NumericalErrorKind, ShapeMismatchError, TlBackendError, TlBackendResult,
};
pub use execution_mode::{
    CompilationStats, CompiledGraph, ExecutionConfig, ExecutionMode, MemoryPlan, OptimizationConfig,
};
pub use executor::Scirs2Exec;
pub use executor_f32::{Scirs2Exec32, Scirs2Tensor32};
pub use fallback::{is_valid, sanitize_tensor, FallbackConfig};
pub use gather_scatter::{
    gather, gather_nd, masked_fill, masked_select, scatter_add, scatter_max, scatter_min, top_k,
    GatherScatterError, IndexStats,
};
pub use geometric_ops::{
    gcn_layer, graph_laplacian, mat_mul, sph_harm, spherical_harmonics, AdjacencyMatrix,
    GcnActivation, GeoError, LaplacianMatrix, LaplacianType, Rotation3,
};
pub use gpu_readiness::{
    assess_gpu_readiness, generate_recommendations, recommend_batch_size, GpuCapability,
    GpuReadinessReport, WorkloadProfile,
};
pub use gradient_ops::{
    gumbel_softmax, gumbel_softmax_backward, soft_exists, soft_exists_backward, soft_forall,
    soft_forall_backward, ste_threshold, ste_threshold_backward, GumbelSoftmaxConfig,
    QuantifierMode, SteConfig,
};
pub use graph_optimizer::{
    GraphOptimizer, GraphOptimizerBuilder, OptimizationPass, OptimizationStats,
};
pub use inplace_ops::{can_execute_inplace, is_shape_preserving, InplaceExecutor, InplaceStats};
pub use lazy::{
    EvaluationPlan, LazyEinsumGraph, LazyExecutor, LazyStats, LazyTensor, NodeMemoryEstimate,
};
pub use memory_profiler::{
    AllocationRecord, AtomicMemoryCounter, MemoryProfiler, MemoryStats as ProfilerMemoryStats,
};
pub use metrics::{
    format_bytes, shared_metrics, AtomicMetrics, MemoryStats, MetricsCollector, MetricsConfig,
    MetricsSummary, OperationRecord, OperationStats, SharedMetrics, ThroughputStats,
};
pub use parallel_executor::{ParallelConfig, ParallelScirs2Exec, ParallelStats};
pub use pooling::{
    adaptive_avg_pool, avg_pool, global_avg_pool, global_max_pool, lp_pool, max_pool,
    max_pool_with_indices, max_unpool, PoolConfig, PoolingError, PoolingStats,
};
pub use precision::{ComputePrecision, Precision, PrecisionConfig, Scalar};
pub use precision_cast::{cast_f32_to_f64, cast_f64_to_f32, DualPrecisionBridge};
pub use probabilistic::{
    bald_epistemic_uncertainty, mc_integrate, predictive_entropy, sample_bernoulli,
    sample_categorical, sample_normal, sample_uniform, MeanFieldGaussian, MonteCarloConfig,
    MonteCarloEstimator, UncertaintyEstimate, VariationalConfig, VariationalInference,
};
pub use profiled_executor::ProfiledScirs2Exec;
pub use quantization::{
    calibrate_quantization, QatConfig, QuantizationGranularity, QuantizationParams,
    QuantizationScheme, QuantizationStats, QuantizationType, QuantizedTensor,
};
pub use recurrent::{
    gru_sequence, lstm_sequence, rnn_sequence, GruCell, LstmCell, LstmState, RecurrentError,
    RecurrentStats, RnnCell,
};
pub use scoring::{
    log_sum_exp, weighted_soft_exists, weighted_soft_forall, LogSpaceAggregator, ScoringConfig,
    ScoringError, ScoringMode, WeightedQuantifier,
};
pub use shape_inference::{validate_tensor_shapes, Scirs2ShapeInference};
pub use signal_ops::{
    apply_window, dct, dft, fir_filter, hz_to_mel, idct, idft, istft, mel_filterbank, mel_to_hz,
    stft, window, Complex, FirFilter, SignalError, StftResult, WindowType,
};
pub use tensor_io::{
    load_tensor, load_tensors, read_header, read_tensor, save_tensor, save_tensors, write_tensor,
    TensorHeader, TensorIoError,
};
pub use tensor_loss::{
    LossReduction, TensorBCELoss, TensorCosineEmbeddingLoss, TensorCrossEntropyLoss,
    TensorFocalLoss, TensorHuberLoss, TensorKLDivLoss, TensorLoss, TensorLossConfig,
    TensorLossError, TensorLossOutput, TensorLossRegistry, TensorMseLoss,
};
pub use tracing::{ExecutionTracer, TraceEvent, TraceLevel};
