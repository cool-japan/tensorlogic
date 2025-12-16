//! Engine-agnostic traits and execution planning API.
//!
//! **Version**: 0.1.0-alpha.2 | **Status**: Production Ready
//!
//! This crate defines the abstract execution interfaces and optimization utilities for TensorLogic:

#![allow(clippy::len_zero)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::overly_complex_bool_expr)]
#![allow(clippy::unwrap_or_default)]
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
//!
//! ## Advanced Quantization (Alpha.2) 🆕
//! - **Quantizer**: Complete quantization pipeline (QAT/PTQ)
//! - **QuantizationType**: INT8, INT4, INT2, FP8, Binary, Ternary support
//! - **CalibrationStrategy**: Multiple calibration methods (MinMax, Percentile, MSE, KL-divergence)
//! - **FakeQuantize**: Quantization simulation for training
//!
//! ## Dynamic Batching (Alpha.2) 🆕
//! - **DynamicBatcher**: Adaptive request batching with priority queues
//! - **RequestQueue**: Priority-based queuing (Low/Normal/High/Critical)
//! - **AdaptiveBatcher**: Automatic batch size optimization
//! - **BatchingStats**: Comprehensive throughput and latency metrics
//!
//! ## Advanced Kernel Fusion (Alpha.2) 🆕
//! - **FusionOptimizer**: Pattern-based fusion detection and optimization
//! - **FusionStrategy**: Conservative/Aggressive/Balanced/Memory-aware modes
//! - **FusionCostModel**: Memory bandwidth-aware cost modeling
//! - **FusionPattern**: Common patterns (MatMul+Bias, MatMul+Activation, etc.)
//!
//! ## Workspace Management (Alpha.2) 🆕
//! - **WorkspacePool**: Memory pool with multiple allocation strategies
//! - **SharedWorkspacePool**: Thread-safe workspace sharing
//! - **AllocationStrategy**: BestFit/FirstFit/ExactFit/PowerOfTwo
//! - **WorkspaceStats**: Efficiency metrics and hit rate tracking
//!
//! ## Multi-Model Coordination (Alpha.2) 🆕
//! - **MultiModelCoordinator**: Ensemble and multi-model management
//! - **EnsembleStrategy**: Averaging/Voting/Stacking/Boosting
//! - **RoutingStrategy**: Priority/Latency/Accuracy-based model selection
//! - **CascadeConfig**: Early-exit model cascades
//!
//! ## Mixed Precision Training (Alpha.2) 🆕
//! - **MixedPrecisionConfig**: FP16/BF16/FP8 configuration
//! - **LossScaler**: Automatic loss scaling with dynamic adjustment
//! - **PrecisionMode**: Multiple precision modes (FP32/FP16/BF16/FP8/FP64)
//! - **GradientCheckpoint**: Memory-efficient gradient checkpointing
//! - **MixedPrecisionState**: Complete training state management
//!
//! ## Sparse Tensor Support (Alpha.2) 🆕
//! - **SparseTensor**: CSR/CSC/COO sparse formats
//! - **SparseCSR**: Compressed Sparse Row format
//! - **SparseCSC**: Compressed Sparse Column format
//! - **SparseCOO**: Coordinate format for construction
//! - **Automatic sparsity detection**: Convert dense to sparse when beneficial
//!
//! ## Parallel Execution (Alpha.2) 🆕
//! - **WorkStealingScheduler**: Dynamic load balancing scheduler
//! - **Task**: Parallel task with dependencies and priorities
//! - **StealStrategy**: Multiple work-stealing strategies
//! - **NumaStrategy**: NUMA-aware memory allocation
//! - **LoadBalanceStats**: Load balancing metrics
//!
//! ## SIMD Optimizations (Alpha.2) 🆕
//! - **SimdCapabilities**: Platform detection (AVX2/AVX-512/NEON/SVE)
//! - **AlignedBuffer**: SIMD-aligned memory allocations
//! - **SimdInstructionSet**: Instruction set abstractions
//! - **SimdOptimizationHints**: Compiler optimization hints
//!
//! ## Graph Rewriting (Alpha.2) 🆕
//! - **RewriteEngine**: Pattern-based graph transformations
//! - **Pattern**: Flexible pattern matching DSL
//! - **RewriteRule**: Custom rewrite rules
//! - **CommonRules**: Standard optimization rules (constant folding, etc.)
//! - **RewriteStrategy**: Application strategies (exhaustive, fixed-point, etc.)
//!
//! ## Profiling-Guided Optimization (Alpha.2) 🆕
//! - **ProfilingOptimizer**: Adaptive performance tuning
//! - **ExecutionProfile**: Runtime performance metrics
//! - **Hotspot**: Performance bottleneck detection
//! - **OptimizationGoal**: Optimization objectives (latency, throughput, memory)
//! - **Auto-tuning**: Automatic configuration selection
//!
//! ## Cache Optimization (Alpha.2) 🆕
//! - **CacheOptimizer**: Memory hierarchy aware optimization
//! - **CacheConfig**: L1/L2/L3 cache configuration
//! - **TilingParams**: Loop tiling for cache efficiency
//! - **CacheMetrics**: Cache performance estimation
//! - **DataLayout**: Cache-friendly data arrangements
//!
//! ## Automatic Parallelization (Experimental) 🧪
//! - **AutoParallelizer**: Automatic detection of parallelism opportunities
//! - **ParallelizationAnalysis**: Analysis of parallel execution potential
//! - **ParallelExecutionPlan**: Generated parallel execution plans
//! - **WorkPartition**: Work distribution across workers
//! - **Cost modeling**: Estimate execution costs and communication overhead
//!
//! ## Speculative Execution (Experimental) 🧪
//! - **SpeculativeExecutor**: Branch prediction and speculative execution
//! - **PredictionStrategy**: Multiple prediction strategies
//! - **RollbackPolicy**: Handling mispredictions
//! - **SpeculationStats**: Track speculation success rates
//! - **Adaptive learning**: Learn from prediction outcomes
//!
//! ## Learned Optimizations (Experimental) 🧪
//! - **LearnedOptimizer**: ML-based optimization decisions
//! - **LearningStrategy**: Supervised, reinforcement, online learning
//! - **CostPrediction**: Learned cost models
//! - **FusionRecommendation**: ML-based fusion decisions
//! - **Reinforcement learning**: Q-learning for scheduling

pub mod async_exec;
pub mod auto_parallel;
pub mod autodiff;
pub mod backend_tests;
pub mod batch;
pub mod cache;
pub mod cache_optimizer;
pub mod capabilities;
pub mod compilation;
pub mod context;
pub mod debug;
pub mod diagnostics;
pub mod distributed;
mod dummy_executor;
mod dummy_tensor;
pub mod dynamic_batching;
pub mod eager;
mod error;
pub mod fusion;
pub mod gradcheck;
pub mod jit;
pub mod learned_opt;
pub mod memory;
pub mod mixed_precision;
pub mod multimodel;
mod ops;
pub mod optimization;
pub mod parallel;
pub mod perfregression;
pub mod placement;
pub mod profiling;
pub mod profiling_optimizer;
pub mod quantization;
pub mod recovery;
pub mod rewrite;
pub mod scheduling;
pub mod shape;
pub mod simd;
pub mod sparse;
pub mod speculative;
pub mod strategy;
pub mod streaming;
pub mod tensor_view;
mod traits;
pub mod typesafe;
pub mod validation;
pub mod visualization;
pub mod workspace;

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
pub use auto_parallel::{
    AutoParallelError, AutoParallelizer, CostModel as AutoParallelCostModel, DependencyType,
    NodeId as AutoParallelNodeId, NodeInfo, ParallelExecutionPlan, ParallelStage,
    ParallelizationAnalysis, ParallelizationStrategy, WorkPartition,
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
pub use cache_optimizer::{
    AccessPattern, CacheConfig, CacheLevel, CacheMetrics, CacheOptimizer, CacheOptimizerError,
    DataLayout, OptimizationStats as CacheOptimizationStats, TilingParams,
};
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
pub use dynamic_batching::{
    AdaptiveBatcher, BatchRequest, BatchingError, BatchingStats, DynamicBatchConfig,
    DynamicBatcher, Priority, RequestMetadata, RequestQueue,
};
pub use eager::{EagerOp, EagerOps, EagerTape, TlEagerAutodiff, Variable, VariableGrad};
pub use error::ExecutorError;
pub use fusion::{
    FusionCandidate, FusionConfig, FusionCostModel, FusionError, FusionOptimizer, FusionPattern,
    FusionStats, FusionStrategy,
};
pub use gradcheck::{
    compare_gradients, numerical_gradient_central, numerical_gradient_forward, quick_check,
    GradCheckConfig, GradCheckResult, GradientChecker, GradientError,
};
pub use jit::{
    AdaptiveOptimizationPlan, AdaptiveOptimizer, HotPathDetector, JitCache, JitCacheEntry,
    JitCacheStats, JitCompiler, JitConfig, JitEntryStats, JitKey, JitStats, SpecializationContext,
    TlJitExecutor,
};
pub use learned_opt::{
    CostPrediction, FeatureVector, FusionRecommendation, LearnedOptError, LearnedOptimizer,
    LearningStats, LearningStrategy, ModelType, NodeId as LearnedOptNodeId, OptimizationAction,
    RewardSignal, ScheduleRecommendation, TrainingExample,
};
pub use memory::{MemoryEstimate, MemoryEstimator, TensorMemory};
pub use mixed_precision::{
    GradientCheckpoint, LossScaler, LossScalerStats, LossScalingStrategy, MixedPrecisionConfig,
    MixedPrecisionError, MixedPrecisionState, MixedPrecisionStats, PrecisionMode,
};
pub use multimodel::{
    CascadeConfig, CoordinationStats, EnsembleConfig, EnsembleStrategy, ModelMetadata,
    MultiModelCoordinator, MultiModelError, ResourceRequirements, RoutingStrategy,
    TlEnsembleExecutor, TlModelRouter,
};
pub use ops::{ElemOp, ReduceOp};
pub use optimization::{
    FusionOpportunity, FusionPlanner, FusionType, GraphOptimizer, OptimizationResult,
};
pub use parallel::{
    LoadBalanceStats, NumaNode, NumaStrategy, ParallelConfig, ParallelError, SchedulerStats,
    StealStrategy, Task, TaskId, TaskPriority, WorkStealingScheduler,
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
pub use profiling_optimizer::{
    ExecutionProfile, Hotspot, OptimizationGoal, OptimizationReport, OptimizationStrategy,
    ProfilingOptimizer, ProfilingOptimizerError, TuningConfig,
};
pub use quantization::{
    CalibrationStats, CalibrationStrategy, FakeQuantize, QuantizationConfig, QuantizationError,
    QuantizationGranularity, QuantizationMode, QuantizationParams, QuantizationSummary,
    QuantizationSymmetry, QuantizationType, Quantizer,
};
pub use recovery::{
    Checkpoint, CheckpointManager, DegradationPolicy, FailureInfo, FallbackStrategy,
    RecoveryConfig, RecoveryMetadata, RecoveryResult, RecoveryStats, RecoveryStrategy, RetryPolicy,
    TlRecoverableExecutor,
};
pub use rewrite::{
    CommonRules, Match, NodeId as RewriteNodeId, Pattern, ReplacementFn, RewriteEngine,
    RewriteError, RewriteRule, RewriteStats, RewriteStrategy,
};
pub use scheduling::{ExecutionSchedule, NodeCost, Scheduler, SchedulingStrategy};
pub use shape::{DimSize, ShapeInferenceContext, TensorShape};
pub use simd::{
    AlignedBuffer, CpuArchitecture, SimdCapabilities, SimdError, SimdInstructionSet,
    SimdOptimizationHints,
};
pub use sparse::{
    detect_sparsity, to_sparse_if_beneficial, SparseCOO, SparseCSC, SparseCSR, SparseError,
    SparseFormat, SparseTensor, SparseTensorBuilder,
};
pub use speculative::{
    BranchOutcome, NodeId as SpeculativeNodeId, PredictionStrategy, RollbackPolicy,
    SpeculationStats, SpeculativeError, SpeculativeExecutor, SpeculativeTask,
};
pub use strategy::{
    ExecutionMode, ExecutionStrategy, GradientStrategy, MemoryStrategy, ParallelismStrategy,
    StrategyOptimizer,
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
pub use workspace::{
    AllocationStrategy, DefragmentationResult, SharedWorkspacePool, Workspace, WorkspaceConfig,
    WorkspaceError, WorkspacePool, WorkspaceStats,
};
