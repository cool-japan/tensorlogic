//! Engine-agnostic traits and execution planning API.
//!
//! This crate defines the abstract execution interfaces and optimization utilities for TensorLogic:
//!
//! ## Core Execution Traits
//! - **TlExecutor**: Core tensor operations (einsum, element-wise, reductions)
//! - **TlAutodiff**: Forward/backward pass for automatic differentiation
//! - **TlEnhancedAutodiff**: Enhanced autodiff with gradient accumulation, clipping, scaling
//! - **TlBatchExecutor**: Batch execution support
//! - **TlStreamingExecutor**: Streaming execution for large datasets
//! - **TlRecoverableExecutor**: Execution with error recovery and checkpointing
//! - **TlCapabilities**: Backend capability queries
//! - **TlProfiledExecutor**: Execution profiling
//! - **TlJitExecutor**: Just-In-Time compilation support
//! - **TlDistributedExecutor**: Distributed multi-device execution
//!
//! ## Optimization Utilities
//! - **GraphOptimizer**: Fusion detection, dead node elimination, redundancy analysis
//! - **FusionPlanner**: Planning and validation of fusion transformations
//! - **Scheduler**: Execution scheduling with multiple strategies (sequential, parallel, cost-based)
//! - **PlacementOptimizer**: Device placement and multi-device coordination
//! - **TensorCache**: Result caching with LRU/FIFO/LFU eviction policies
//! - **MemoryPool**: Tensor memory pooling for allocation reuse
//! - **ExecutionStrategy**: Complete strategy configuration (mode, precision, memory, parallelism)
//! - **ExecutionContext**: State management and lifecycle tracking with hooks
//! - **GraphCompiler**: Ahead-of-time graph compilation with optimization passes
//! - **CompilationCache**: Caching of compiled graphs to avoid recompilation
//!
//! ## JIT Compilation
//! - **JitCompiler**: Runtime compilation with hot path detection
//! - **JitCache**: Specialized caching for JIT-compiled graphs
//! - **HotPathDetector**: Identifies frequently executed code paths
//! - **AdaptiveOptimizer**: Progressively optimizes based on runtime profiling
//!
//! ## Distributed Execution
//! - **DistributedExecutor**: Multi-device execution coordination
//! - **DataParallelCoordinator**: Data-parallel training across devices
//! - **ModelParallelCoordinator**: Model-parallel execution with tensor sharding
//! - **PipelineParallelCoordinator**: Pipeline parallelism across stages
//! - **CommunicationBackend**: Abstract interface for device communication
//!
//! ## Zero-Copy Operations (Alpha.2) 🆕
//! - **TensorView**: Zero-copy tensor views and slicing
//! - **SliceSpec**: Flexible slicing specifications
//! - **ViewBuilder**: Ergonomic view construction
//! - **TensorViewable**: Trait for zero-copy tensor operations
//!
//! ## Async Execution (Alpha.2) 🆕
//! - **TlAsyncExecutor**: Async/await-based non-blocking execution
//! - **TlAsyncBatchExecutor**: Asynchronous batch processing
//! - **TlAsyncStreamExecutor**: Async streaming with backpressure
//! - **AsyncExecutorPool**: Load-balanced executor pool
//!
//! ## Enhanced Diagnostics (Alpha.2) 🆕
//! - **Diagnostic**: Rich error messages with suggestions
//! - **DiagnosticCollector**: Error aggregation and reporting
//! - **ShapeMismatchDiagnostic**: Helpful shape error messages
//! - **PerformanceDiagnostic**: Performance issue detection
//!
//! ## Analysis and Validation
//! - **GraphValidator**: Graph validation and diagnostics
//! - **MemoryEstimator**: Memory usage estimation and lifetime analysis
//! - **ShapeInferenceContext**: Tensor shape inference for optimization
//!
//! ## Debugging Utilities
//! - **ExecutionTracer**: Record execution flow through computation graphs
//! - **TensorInspector**: Examine intermediate tensor values and statistics
//! - **BreakpointManager**: Pause execution at specific nodes for inspection
//! - **ExecutionRecorder**: Record full execution history for replay and analysis
//!
//! ## Visualization Utilities
//! - **TimelineVisualizer**: ASCII/DOT/JSON timeline visualization
//! - **GraphVisualizer**: Computation graph visualization
//! - **TensorStatsVisualizer**: Tensor statistics and histograms
//! - **ExportFormat**: Export to various formats for external tools
//!
//! ## Testing and Development
//! - **DummyExecutor**: Minimal implementation for testing and prototyping
//! - **DummyTensor**: Simple tensor representation for tests
//! - **Backend Tests**: Comprehensive test templates for backend validation
//! - **Gradient Checking**: Numerical gradient verification utilities
//!
//! ## Eager Execution
//! - **TlEagerAutodiff**: Eager mode automatic differentiation
//! - **Variable**: Variables with gradient tracking
//! - **EagerTape**: Dynamic computation graph recording

pub mod async_exec;
pub mod autodiff;
pub mod backend_tests;
pub mod batch;
pub mod cache;
pub mod capabilities;
pub mod compilation;
pub mod context;
pub mod debug;
pub mod diagnostics;
pub mod distributed;
mod dummy_executor;
mod dummy_tensor;
pub mod eager;
mod error;
pub mod gradcheck;
pub mod jit;
pub mod memory;
mod ops;
pub mod optimization;
pub mod perfregression;
pub mod placement;
pub mod profiling;
pub mod recovery;
pub mod scheduling;
pub mod shape;
pub mod strategy;
pub mod streaming;
pub mod tensor_view;
mod traits;
pub mod typesafe;
pub mod validation;
pub mod visualization;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod validation_tests;

#[cfg(test)]
mod memory_tests;

#[cfg(feature = "async")]
pub use async_exec::{
    AsyncConfig, AsyncExecutionError, AsyncExecutionHandle, AsyncExecutorPool, AsyncStats,
    AsyncStreamResults, BoxFuture, TlAsyncBatchExecutor, TlAsyncExecutor, TlAsyncStreamExecutor,
};
pub use autodiff::{
    AccumulationConfig, ClippingStrategy, CustomGradientRegistry, GradientAccumulationStrategy,
    GradientAccumulator, GradientClipper, GradientConfig, GradientScaler, GradientScaling,
    GradientStats, TlEnhancedAutodiff,
};
pub use backend_tests::{
    assert_vec_close, print_test_summary, run_all_basic_tests, run_all_performance_tests,
    test_backend_edge_cases, test_backend_einsum, test_backend_elem_binary,
    test_backend_elem_unary, test_backend_forward, test_backend_large_tensors,
    test_backend_memory_efficiency, test_backend_reduce, test_backend_shapes, BackendTestAdapter,
    TestResult, DEFAULT_TOLERANCE,
};
pub use batch::{BatchResult, TlBatchExecutor};
pub use cache::{CacheKey, CacheStats, EvictionPolicy, MemoryPool, PoolStats, TensorCache};
pub use capabilities::{BackendCapabilities, DType, DeviceType, Feature, TlCapabilities};
pub use compilation::{
    CacheStats as CompilationCacheStats, CompilationCache, CompilationConfig, CompilationKey,
    CompilationStats, CompiledGraph, GraphCompiler, OptimizationLevel, TlCompilableExecutor,
};
pub use context::{ExecutionContext, ExecutionHook, ExecutionPhase, ExecutionState, LoggingHook};
pub use debug::{
    Breakpoint, BreakpointHit, BreakpointManager, ExecutionRecorder, ExecutionReport,
    ExecutionTrace, ExecutionTracer, OperationHandle, TensorInspector, TensorStats,
    TraceEntry as DebugTraceEntry, TraceSummary,
};
pub use diagnostics::{
    Diagnostic, DiagnosticCollector, MemoryDiagnostic, NodeExecutionDiagnostic,
    PerformanceDiagnostic, Severity, ShapeMismatchDiagnostic, SourceLocation,
    TypeMismatchDiagnostic,
};
pub use distributed::{
    CommunicationBackend, CommunicationOp, DataParallelCoordinator, DistributedConfig,
    DistributedExecutor, DistributedPlacementPlan, DistributedStats, DummyCommunicationBackend,
    ModelParallelCoordinator, ParallelismStrategy as DistributedParallelismStrategy,
    PipelineParallelCoordinator, ReductionOp, ShardingSpec, TlDistributedExecutor,
};
pub use dummy_executor::DummyExecutor;
pub use dummy_tensor::DummyTensor;
pub use eager::{EagerOp, EagerOps, EagerTape, TlEagerAutodiff, Variable, VariableGrad};
pub use error::ExecutorError;
pub use gradcheck::{
    compare_gradients, numerical_gradient_central, numerical_gradient_forward, quick_check,
    GradCheckConfig, GradCheckResult, GradientChecker, GradientError,
};
pub use jit::{
    AdaptiveOptimizationPlan, AdaptiveOptimizer, HotPathDetector, JitCache, JitCacheEntry,
    JitCacheStats, JitCompiler, JitConfig, JitEntryStats, JitKey, JitStats, SpecializationContext,
    TlJitExecutor,
};
pub use memory::{MemoryEstimate, MemoryEstimator, TensorMemory};
pub use ops::{ElemOp, ReduceOp};
pub use optimization::{
    FusionOpportunity, FusionPlanner, FusionType, GraphOptimizer, OptimizationResult,
};
pub use perfregression::{
    BenchmarkBaseline, BenchmarkComparison, BenchmarkConfig, BenchmarkStats, PerfRegression,
    RegressionReport,
};
pub use placement::{Device, PlacementOptimizer, PlacementPlan, PlacementStrategy};
pub use profiling::{
    Bottleneck, BottleneckAnalyzer, BottleneckReport, PerformanceBaseline, PerformanceComparison,
    ProfileData, ProfileStatistics, Profiler, ProfilerHook, TimelineProfiler, TlProfiledExecutor,
    TraceEntry,
};
pub use recovery::{
    Checkpoint, CheckpointManager, DegradationPolicy, FailureInfo, FallbackStrategy,
    RecoveryConfig, RecoveryMetadata, RecoveryResult, RecoveryStats, RecoveryStrategy, RetryPolicy,
    TlRecoverableExecutor,
};
pub use scheduling::{ExecutionSchedule, NodeCost, Scheduler, SchedulingStrategy};
pub use shape::{DimSize, ShapeInferenceContext, TensorShape};
pub use strategy::{
    ExecutionMode, ExecutionStrategy, GradientStrategy, MemoryStrategy, ParallelismStrategy,
    PrecisionMode, StrategyOptimizer,
};
pub use streaming::{
    ChunkIterator, ChunkMetadata, StreamProcessor, StreamResult, StreamingConfig, StreamingMode,
    TlStreamingExecutor,
};
pub use tensor_view::{
    InPlaceMode, InPlaceOps, SliceSpec, TensorView, TensorViewable, ViewBuilder,
};
pub use traits::{TlAutodiff, TlExecutor};
pub use typesafe::{
    BroadcastShape, Dim, DimMul, DimOp, DimSize as TypesafeDimSize, Dyn, EinsumSpec, FixedShape,
    Matrix, MatrixOps, Nat, Scalar, ShapeConstraint, ShapedTensor, Static, Tensor3D, Tensor4D,
    TensorBuilder, TypedBatch, TypedInputs, TypedOutputs, TypedTensor, TypedTensorOps, Vector, D1,
    D2, D3, D4, D5, D6, S, Z,
};
pub use validation::{GraphValidator, ValidationResult};
pub use visualization::{
    ExportFormat, GraphConfig, GraphVisualizer, TensorStatsVisualizer, TimelineConfig,
    TimelineVisualizer, VisualizationFormat,
};
