# TensorLogic Infer — TODO

**Status**: Stable | **Version**: 0.1.3 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-31
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

Executor and autodiff traits for tensor-logic inference pipelines.

## Completed

### Core Traits
- [x] TlExecutor trait definition
- [x] TlAutodiff trait definition
- [x] DummyExecutor implementation
- [x] TensorInputs/TensorOutputs types
- [x] Basic test coverage

### Trait Enhancement
- [x] **Batch execution support**
  - [x] BatchResult<T> container with metadata
  - [x] TlBatchExecutor trait
  - [x] Parallel execution support (execute_batch_parallel)
  - [x] Optimal batch size recommendations
- [x] **Backend capability queries**
  - [x] BackendCapabilities descriptor
  - [x] TlCapabilities trait
  - [x] Device/dtype/feature detection (CPU/GPU/TPU)
  - [x] Operation support queries
  - [x] Capability summary generation

### Type System
- [x] Tensor shape inference
  - [x] TensorShape with static/dynamic/symbolic dimensions
  - [x] ShapeInferenceContext for graph-level inference
  - [x] Shape compatibility and broadcasting checks
  - [x] Einsum spec parsing for output shape
- [x] Shape validation
  - [x] DimSize enum (Static/Dynamic/Symbolic)
  - [x] as_static() for runtime checks
  - [x] rank() and is_static() helpers

### Execution Profiling
- [x] Profiling infrastructure
  - [x] OpProfile with timing statistics (count, avg, min, max)
  - [x] MemoryProfile with allocation tracking
  - [x] ProfileData with operation summaries
  - [x] Profiler with automatic timing
- [x] TlProfiledExecutor trait
  - [x] enable_profiling()/disable_profiling()
  - [x] get_profile_data()
  - [x] time_op() for automatic timing

### Streaming Execution
- [x] **Add streaming execution**
  - [x] execute_streaming() for large datasets
  - [x] TlStreamingExecutor trait
  - [x] StreamingConfig with multiple modes (Fixed/Dynamic/Adaptive)
  - [x] ChunkIterator for memory-efficient iteration
  - [x] StreamProcessor with split/merge capabilities
  - [x] Adaptive chunking based on performance metrics
  - [x] Prefetching and checkpoint support

### Error Recovery
- [x] **Error recovery**
  - [x] Partial results on failure (RecoveryResult)
  - [x] Checkpoint/restart (CheckpointManager)
  - [x] Graceful degradation (DegradationPolicy, FallbackStrategy)
  - [x] TlRecoverableExecutor trait
  - [x] RecoveryConfig with multiple strategies
  - [x] RetryPolicy with exponential backoff
  - [x] RecoveryStats for monitoring

### Autodiff Enhancements
- [x] Gradient accumulation strategy
  - [x] Standard accumulation
  - [x] Gradient checkpointing
  - [x] Mixed precision
  - [x] Average accumulation
  - [x] GradientAccumulator implementation
- [x] Custom gradient functions
  - [x] Register custom backward passes (CustomGradientRegistry)
  - [x] Override default gradients
- [x] Gradient clipping/scaling
  - [x] Clip by value/norm (ClippingStrategy)
  - [x] Automatic scaling (GradientScaler)
  - [x] GradientClipper implementation
  - [x] GradientStats for monitoring

### Type Safety Extensions
- [x] Type-safe tensor wrappers
  - [x] Strong typing for inputs/outputs (TypedInputs/TypedOutputs)
  - [x] Compile-time shape checking (TypedTensor with Nat rank)
  - [x] Type-level dimensions (D1-D6, Static, Dyn)
  - [x] Typed aliases (Scalar, Vector, Matrix, Tensor3D, Tensor4D)
  - [x] TensorBuilder for safe construction
  - [x] TypedBatch for batched operations
  - [x] ShapeConstraint trait

### Execution Modes
- [x] **Eager execution** (eager.rs - 14 tests)
- [x] **Graph compilation**
  - [x] Compile to optimized form (GraphCompiler with multiple optimization levels)
  - [x] Cache compiled graphs (CompilationCache with LRU-style eviction)
  - [x] TlCompilableExecutor trait for compilation support
  - [x] Compilation statistics and performance tracking
  - [x] 14 comprehensive tests (100% passing)
- [x] **JIT compilation** (jit.rs)
  - [x] Runtime compilation with hot path detection
  - [x] Adaptive optimization based on profiling
  - [x] Graph specialization for observed shapes
  - [x] JitCompiler with caching support
  - [x] 13 comprehensive tests (100% passing)
- [x] **Distributed execution** (distributed.rs)
  - [x] Multi-device support with communication backends
  - [x] Data parallelism with gradient synchronization
  - [x] Model parallelism with tensor sharding
  - [x] Pipeline parallelism with stage coordination
  - [x] 13 comprehensive tests (100% passing)

### Utilities
- [x] **Execution profiling**
  - [x] Time per operation
  - [x] Memory usage
  - [x] Bottleneck detection
- [x] **Debugging tools**
  - [x] Trace execution (ExecutionTracer)
  - [x] Inspect intermediate tensors (TensorInspector)
  - [x] Breakpoint support (BreakpointManager)
  - [x] Full execution recording (ExecutionRecorder)
- [x] **Visualization**
  - [x] Execution timeline (ASCII, DOT, JSON formats)
  - [x] Tensor flow diagram (ASCII, DOT, JSON, GraphML)
  - [x] Performance visualization
  - [x] Tensor statistics histograms
  - [x] 9 comprehensive tests

### Documentation
- [x] Add README.md
- [x] Trait implementation guide
- [x] Backend development tutorial
- [x] Performance optimization guide

### Debugging Tools
- [x] **Execution tracing and debugging**
  - [x] ExecutionTracer for recording operation flow
  - [x] TensorInspector for examining intermediate values
  - [x] BreakpointManager for pausing execution
  - [x] ExecutionRecorder for full history replay
  - [x] TraceEntry with detailed timing information
  - [x] TraceSummary with performance statistics
  - [x] TensorStats with numerical issue detection
  - [x] Multiple breakpoint types (Node, Operation, NumericalIssue, TimeThreshold)
  - [x] 12 comprehensive tests (100% passing)

### Testing
- [x] Backend compatibility tests (templates for backend developers)
- [x] Stress tests (large graphs)
- [x] Correctness tests (gradient checking)
- [x] Performance regression tests
  - [x] PerfRegression framework with warmup and measurement iterations
  - [x] BenchmarkStats with statistical analysis (mean, median, std_dev, CV)
  - [x] BenchmarkBaseline for save/load baselines (JSON format)
  - [x] RegressionReport with regression detection
  - [x] HTML and text report generation
  - [x] 12 comprehensive tests (100% passing)

### Eager Execution
- [x] **Eager mode automatic differentiation**
  - [x] TlEagerAutodiff trait for dynamic graph building
  - [x] Variable with gradient tracking
  - [x] EagerTape for operation recording
  - [x] EagerOps convenience trait
  - [x] Support for all operations (einsum, elem_op, reduce)
  - [x] 14 comprehensive tests

### Beta.1 / RC.1 Features (All Complete)

#### Zero-Copy Tensor Operations
- [x] **Zero-copy tensor views and slicing**
  - [x] TensorView with flexible SliceSpec
  - [x] ViewBuilder for ergonomic API
  - [x] In-place operation support
  - [x] 10 comprehensive tests

#### Async Execution Support [feature = "async"]
- [x] **Async execution traits**
  - [x] TlAsyncExecutor trait for non-blocking execution
  - [x] TlAsyncBatchExecutor for async batching
  - [x] TlAsyncStreamExecutor for streaming
  - [x] AsyncExecutorPool for load balancing
  - [x] AsyncExecutionHandle for cancellation
  - [x] 4 comprehensive tests

#### Enhanced Diagnostics
- [x] **Rich error messages with suggestions**
  - [x] Diagnostic with severity levels
  - [x] DiagnosticCollector for aggregation
  - [x] ShapeMismatchDiagnostic builder
  - [x] TypeMismatchDiagnostic builder
  - [x] MemoryDiagnostic builder
  - [x] PerformanceDiagnostic builder
  - [x] Source location tracking
  - [x] 10 comprehensive tests

#### Mixed Precision Training
- [x] **Complete mixed precision training support** (mixed_precision.rs)
  - [x] FP16/BF16/FP8/FP32/FP64 precision modes
  - [x] Automatic loss scaling with dynamic adjustment
  - [x] LossScaler with multiple strategies (Static/Dynamic)
  - [x] MixedPrecisionState for training management
  - [x] Gradient checkpointing for memory efficiency
  - [x] 15 comprehensive tests

#### Sparse Tensor Support
- [x] **Comprehensive sparse tensor infrastructure** (sparse.rs)
  - [x] CSR (Compressed Sparse Row) format
  - [x] CSC (Compressed Sparse Column) format
  - [x] COO (Coordinate) format for construction
  - [x] Automatic sparsity detection and conversion
  - [x] Sparse-dense hybrid operations
  - [x] 14 comprehensive tests

#### Parallel Execution
- [x] **Work-stealing scheduler and parallel infrastructure** (parallel.rs)
  - [x] WorkStealingScheduler with dynamic load balancing
  - [x] Multiple work-stealing strategies (Random/MaxLoad/LRU/RoundRobin)
  - [x] Task dependencies and priority levels
  - [x] NUMA-aware memory allocation
  - [x] Load balancing statistics and metrics
  - [x] 13 comprehensive tests

#### SIMD Optimizations
- [x] **Platform-specific SIMD optimization utilities** (simd.rs)
  - [x] SimdCapabilities detection (AVX2/AVX-512/NEON/SVE)
  - [x] AlignedBuffer for SIMD-aligned memory
  - [x] SimdInstructionSet abstractions
  - [x] SimdOptimizationHints for compiler
  - [x] 13 comprehensive tests

#### Advanced Quantization
- [x] **Complete quantization pipeline** (quantization.rs)
  - [x] INT8, INT4, INT2, FP8, Binary, Ternary quantization types
  - [x] QAT and PTQ with multiple calibration strategies
  - [x] Per-tensor and per-channel granularity
  - [x] Symmetric and asymmetric modes
  - [x] Comprehensive compression analysis

#### Dynamic Batching
- [x] **Adaptive request batching for inference serving** (dynamic_batching.rs)
  - [x] 4 priority levels (Low/Normal/High/Critical)
  - [x] Adaptive batch size optimization
  - [x] Request timeout and queueing
  - [x] Latency and throughput optimization strategies

#### Advanced Kernel Fusion
- [x] **Pattern-based fusion optimization** (fusion.rs)
  - [x] MatMul+Bias, MatMul+Activation, BatchNorm+ReLU patterns
  - [x] Vertical and horizontal fusion detection
  - [x] Memory bandwidth-aware cost modeling
  - [x] Conservative/Aggressive/Balanced/Memory-aware strategies

#### Workspace Management
- [x] **Memory pool for efficient allocation reuse** (workspace.rs)
  - [x] BestFit/FirstFit/ExactFit/PowerOfTwo allocation strategies
  - [x] Automatic expansion and defragmentation
  - [x] Thread-safe shared workspace pools
  - [x] Comprehensive efficiency metrics

#### Multi-Model Coordination
- [x] **Ensemble and multi-model management** (multimodel.rs)
  - [x] Ensemble strategies: Averaging, Voting, Stacking, Boosting
  - [x] Model routing: Priority, Latency, Accuracy, Round-robin, Cascade
  - [x] Early-exit cascade support
  - [x] Resource tracking and usage statistics

#### Graph Rewriting
- [x] **Pattern-based graph transformation engine** (rewrite.rs)
  - [x] Pattern matching DSL with flexible combinators
  - [x] RewriteEngine with multiple application strategies
  - [x] Common optimization rules (identity elimination, constant folding)
  - [x] Exhaustive, fixed-point, and prioritized rewrite strategies
  - [x] 23 comprehensive tests

#### Profiling-Guided Optimization
- [x] **Adaptive performance tuning infrastructure** (profiling_optimizer.rs)
  - [x] Runtime profiling and execution profile collection
  - [x] Hotspot detection and performance bottleneck analysis
  - [x] Multiple optimization goals (latency, throughput, memory, energy)
  - [x] Auto-tuning with A/B testing support
  - [x] 21 comprehensive tests

#### Cache Optimization
- [x] **Memory hierarchy aware optimization** (cache_optimizer.rs)
  - [x] L1/L2/L3 cache configuration and modeling
  - [x] Loop tiling parameter computation
  - [x] Cache metrics estimation (hit rate, latency, bandwidth)
  - [x] Data layout recommendations for different access patterns
  - [x] 20 comprehensive tests

### Experimental Features (All Complete)

#### Automatic Parallelization
- [x] **Automatic detection of parallelism opportunities** (auto_parallel.rs)
  - [x] Graph-level parallelism detection with dependency analysis
  - [x] Cost model for parallel execution with communication overhead estimation
  - [x] Dynamic work partitioning across workers with load balancing
  - [x] Multiple parallelization strategies (Conservative/Balanced/Aggressive/CostBased)
  - [x] 19 comprehensive tests

#### Speculative Execution
- [x] **Branch prediction and speculative execution** (speculative.rs)
  - [x] Multiple prediction strategies (HistoryBased/AlwaysTrue/MostFrequent/Adaptive)
  - [x] Prefetching for likely future operations
  - [x] Rollback mechanisms (Immediate/Lazy/Checkpoint-based)
  - [x] Confidence scoring and success rate tracking
  - [x] 19 comprehensive tests

#### Learned Optimizations
- [x] **ML-based optimization decisions** (learned_opt.rs)
  - [x] ML-based fusion decisions with reinforcement learning
  - [x] Learned cost models using linear regression
  - [x] Q-learning for scheduling optimization
  - [x] Multiple learning strategies (Supervised/Online/Reinforcement/Transfer)
  - [x] Feature extraction and online learning
  - [x] 21 comprehensive tests

---

## v0.1.1 Additions (2026-03-29) ✅
- [x] **Low-Rank Approximation** (src/low_rank/)
  - [x] TruncatedSvd with power-iteration (from scratch, no external SVD)
  - [x] SvdResult: reconstruct(), relative_error(), energy_fraction()
  - [x] LowRankApproximation: matrix/matmul approx, is_candidate(), optimal_rank()
  - [x] LowRankInferencePass: EinsumGraph candidate scanning
  - [x] 15 tests passing
- [x] **Partitioned Reductions** (src/partitioned/)
  - [x] PartitionedReducer: reduce_all(), reduce_axis(), log_sum_exp()
  - [x] AccumulationStrategy: Sum/Max/Min/Mean/Product/LogSumExp
  - [x] PartitionedStats tracking
  - [x] 10 tests passing
- [x] **Enhanced Streaming (StreamingConfigV2)** (streaming.rs extensions)
  - [x] BackpressureConfig with 4 strategies (Block/DropOldest/DropNewest/ErrorOnFull)
  - [x] WatermarkConfig for out-of-order event handling
  - [x] StreamingStats: latency, drop rate, throughput
  - [x] 15 tests passing
- [x] **Windowed Aggregation** (src/windowed_aggregation.rs)
  - [x] WindowedAggregation: tumbling, sliding, session, count windows
  - [x] WindowAggregation: Sum/Mean/Max/Min/Count/LastValue/FirstValue
  - [x] WindowConfig builder API
  - [x] 12 tests passing

## v0.1.3 Enhancements (2026-03-30)

- [x] **Symbolic Shape Support** (`symbolic_shape.rs`): `SymbolicDim` (Fixed/Symbolic/Product), `SymbolicShape = Vec<SymbolicDim>`, `ShapeConstraint`, `SymbolicShapeEnv` with unification engine. `propagate_einsum_shapes()` infers output shapes from einsum specs. Handles batch dims, chained ops, contradiction detection. 25 new tests.

## v0.1.4 Enhancements (2026-03-30)

- [x] **Structured Sparsity + Pruning** (`pruning.rs`): `SparsityPattern` (Unstructured/Block/Row/Column/N:M), `MagnitudePruner::prune_2d()` and `prune()` (N-D), `SparsityStats::compute()` with theoretical speedup, `PruningConfig` with rescale option. Free functions: `compute_sparsity()`, `row_norms()`. 20 new tests.

## v0.1.5 Enhancements (2026-03-30)

- [x] **Tensor Statistics + Anomaly Detection** (`tensor_stats.rs`): `TensorStatsSummary::compute()` (mean/std/min/max/percentiles with NaN/Inf handling), `AnomalyDetector` (z-score outliers, IQR-based, constant detection), `ActivationStatistics` (per-tensor history with trend_mean/trend_std). 20 new tests.

## v0.1.7 Enhancements (2026-03-30)

- [x] **Execution Plan Formatter** (`execution_plan.rs`): `ExecutionPlan` with `PlanStep` builder, `PlanFormatter::format_table()` and `format_tree()`, `compute_memory_timeline()`, parallel speedup analysis, critical path computation. 18 new tests.

## v0.1.8 Enhancements (2026-03-30)

- [x] **Execution Trace Recording** (`execution_trace.rs`): `TraceRecorder` (real-time recording), `RecordedExecutionTrace` (JSON export, slowest-ops, peak memory), `TraceAnalyzer` (operation summary, memory hotspots, avg duration). 18 new tests.

## v0.1.10 Enhancements (2026-03-31)

- [x] **FLOP Estimation** (`cost_model.rs`): `FlopEstimate` with per-node FLOP counting — einsum uses product of all dimension sizes, reduce uses element count, elem-wise ops use tensor size; free function `estimate_node_flops()`
- [x] **Memory Cost Estimation** (`cost_model.rs`): `MemoryCostEstimate` simulating peak live-set memory across a topological execution schedule; accounts for tensor lifetimes and simultaneous live buffers
- [x] **Graph Cost Summary** (`cost_model.rs`): `GraphCostSummary` aggregating total FLOPs, peak memory bytes, critical-path cost, and per-node breakdown for profiling-guided optimization decisions
- [x] **Cost Model** (`cost_model.rs`): `CostModel` struct with `estimate_graph()` producing a full `GraphCostSummary` and `cheapest_node()`/`most_expensive_node()` query helpers
- [x] **Cost-Aware Schedule** (`cost_model.rs`): `CostAwareSchedule` producing a topologically valid execution order ranked by descending per-node cost, enabling schedulers to prioritize critical-path operations first

## v0.1.11

- [x] **BeamSearchDecoder + BeamHypothesis + BeamState** (`beam_search.rs`): `BeamSearchDecoder` with configurable beam width, length penalty (alpha), and repetition penalty; `BeamHypothesis` stores a token sequence with cumulative log-probability; `BeamState` holds the pruned beam set at each step with top-k selection.
- [x] **BeamSearchResult + BeamSearchStats** (`beam_search.rs`): `BeamSearchResult` returns top-k finished sequences sorted by length-normalized score; `BeamSearchStats` tracks total expansion steps, hypotheses pruned by beam width, and EOS-triggered completions.

## v0.1.14

- [x] **JoinOrderOptimizer** (`join_order.rs`): Configurable join ordering engine supporting greedy and dynamic-programming strategies with cost-based optimization
- [x] **greedy_order** (`join_order.rs`): Greedy join ordering that iteratively selects the lowest-cost pair of relations to join based on estimated selectivity
- [x] **dp_order** (`join_order.rs`): Optimal dynamic-programming join ordering with memoization over relation subsets, guaranteeing minimum estimated cost plans
- [x] **JoinPlan** (`join_order.rs`): Tree-structured execution plan representing the join order with estimated total cost and per-node cost breakdown
- [x] **JoinPlanNode** (`join_order.rs`): Individual node in a join plan tree, either a leaf (base relation) or an inner join of two sub-plans with local cost estimate
- [x] **JoinStats** (`join_order.rs`): Statistics for join optimization including number of relations, strategy used, estimated cost, and planning wall-clock time

## v0.1.16

- [x] **UncertaintyEstimate** (`uncertainty.rs`): Result type holding `mean` prediction tensor, `aleatoric_variance` (data noise), and `epistemic_variance` (model uncertainty) components with `total_variance()` and `std_dev()` convenience methods
- [x] **MonteCarloEstimator** (`uncertainty.rs`): Dropout-based MC sampling engine that runs `n_samples` stochastic forward passes, aggregates ensemble statistics, and decomposes total variance into aleatoric and epistemic contributions
- [x] **CalibrationMetrics** (`uncertainty.rs`): Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and reliability diagram data (confidence bins vs accuracy) for assessing how well predicted confidence matches empirical accuracy
- [x] **ConfidenceInterval** (`uncertainty.rs`): Lower/upper bound pair at a configurable coverage probability (e.g. 90%, 95%) derived from the predictive distribution; constructed via `from_normal()` and `from_quantiles()` helpers
- [x] **PredictionInterval** (`uncertainty.rs`): Combines a point estimate with calibrated prediction intervals, distinguishing in-distribution from out-of-distribution samples using a configurable OOD threshold on epistemic variance

## v0.1.12

- [x] **GreedyDecoder** (`sampling.rs`): deterministic token selection via argmax over the logit/probability vector; handles equal-value ties by returning the lowest index.
- [x] **TemperatureSampler** (`sampling.rs`): scales logits by `1/temperature` before softmax then draws a token using a `SimpleRng` LCG; temperature=1.0 reproduces the original distribution.
- [x] **TopKSampler** (`sampling.rs`): truncates the distribution to the top-k highest-logit tokens, re-normalizes, then applies temperature sampling; falls back to greedy when k=1.
- [x] **TopPSampler** (`sampling.rs`): nucleus sampling — sorts tokens by descending probability, includes the minimal prefix whose cumulative probability exceeds `p`, then samples uniformly from that nucleus.
- [x] **ConfigurableSampler** (`sampling.rs`): unified enum (`Greedy` / `Temperature` / `TopK` / `TopP`) dispatching the appropriate strategy from a single `SamplingConfig` struct containing temperature, k, and p parameters.
- [x] **SimpleRng** (`sampling.rs`): lightweight LCG pseudo-random number generator seeded at construction; provides `next_f64()` and `next_usize_below()` used internally by the probabilistic samplers.

## v0.1.17

- [x] **MemoCache** (`memo_cache.rs`): Generic LRU-style expression memoization store keyed by `MemoKey`; configurable capacity with eviction triggered automatically when capacity is exceeded
- [x] **MemoKey** (`memo_cache.rs`): Structural SHA-256 hash key for `TLExpr` sub-trees enabling equality-based cache lookup without full expression comparison; `from_expr()` constructor
- [x] **MemoEvictionPolicy** (`memo_cache.rs`): Enum selecting eviction strategy — `LRU` (least-recently-used), `LFU` (least-frequently-used), `FIFO` (insertion order) — applied consistently on every cache miss when at capacity
- [x] **MemoStats** (`memo_cache.rs`): Hit count, miss count, eviction count, and `hit_rate()` convenience method for monitoring memoization effectiveness at runtime
- [x] **MemoLookupResult** (`memo_cache.rs`): Typed enum `Hit(value)` / `Miss` returned by `MemoCache::get()` eliminating the need for `Option` unwrapping at call sites
- [x] **ExprMemoCache** (`memo_cache.rs`): Pre-configured `MemoCache<TLExpr, EinsumGraph>` specialisation for the most common caching pattern in the inference pipeline
- [x] **MemoCacheBuilder** (`memo_cache.rs`): Builder API for constructing `MemoCache` with capacity, eviction policy, and optional pre-warming from a sequence of `(key, value)` pairs

---

## Open / Future Work

### Distributed Improvements
- [ ] **Advanced communication backends**
  - [ ] NCCL integration for multi-GPU
  - [ ] Gloo backend for CPU clusters
  - [ ] Custom collective operations
- [ ] **Fault tolerance enhancements**
  - [ ] Automatic failover and recovery
  - [ ] Elastic training (dynamic worker scaling)
  - [ ] Distributed checkpointing
- [ ] **Performance monitoring**
  - [x] Per-device profiling
  - [x] Communication bottleneck detection
  - [x] Load balancing metrics

### Developer Experience
- [x] **Improved error messages** ✅ (v0.1.2) — ShapeMismatchDiagnostic with transpose/broadcast suggestions, PerformanceDiagnostic recommendations
- [x] **Enhanced debugging** ✅ (v0.1.2) — StepExecutor with IntermediateValue logging (min/max/mean/nan/inf), BreakpointCondition (NodeIndex, OnNaN, OnInf, Always)
- [ ] **Performance profiling tools**
  - [x] Flamegraph generation
  - [x] `critical-path-analysis` (completed 2026-04-17)
    - **Goal:** Dedicated analysis API extracting the longest dependency chain (critical path) of an inference graph and reporting the bottleneck node + total estimated latency.
    - **Design:** Topological sort + DAG longest-path (dynamic programming over reverse topo order). `pub struct CriticalPathReport { pub nodes: Vec<NodeId>, pub total_latency_ns: u64, pub bottleneck: NodeId }`. `pub fn critical_path(graph: &InferenceGraph) -> CriticalPathReport`. Node latency comes from existing per-node cost estimate (or 1 unit if absent — emit `MissingCost` warning).
    - **Files:** `src/critical_path.rs` (NEW); `src/lib.rs` (export `CriticalPathReport`, `critical_path`).
    - **Prerequisites:** reuse existing `InferenceGraph` / `NodeId` types in this crate — no new graph abstraction.
    - **Tests:** unit tests with hand-rolled DAGs: linear chain (entire graph is critical); diamond (longer branch wins); isolated nodes (empty critical path with MissingCost warning).
    - **Risk:** Graph-type interop — if shapes don't fit, narrow to `pub trait CriticalPathInput` and adapt internally.
  - [ ] Memory bandwidth profiling

### Hardware-Specific Backends
- [ ] Apple Silicon optimizations (Metal)
- [ ] AMD ROCm support
- [ ] Intel oneAPI integration

### Cloud Execution
- [ ] AWS SageMaker integration
- [ ] Google TPU support
- [ ] Azure ML integration

### Advanced Optimizations
- [x] **Higher-order derivatives** ✅ (v0.1.2) — JacobianComputer, HessianComputer (finite differences)
- [x] **Enhanced diagnostics** ✅ (v0.1.2) — ShapeMismatchDiagnostic suggestions, PerformanceDiagnostic recommendations
- [ ] Sparse gradient support
- [ ] Cross-operator fusion enhancements
- [ ] Template-based kernel generation

### Documentation & Testing
- [ ] Property-based testing for all traits
- [ ] Fuzz testing for robustness
- [ ] Integration tests with real backends

---

**Total Completed Items:** 55 tasks (including 3 experimental research directions)
**Production Ready Features:**
- Batch Execution & Parallel Processing
- Shape Inference & Type Checking
- Backend Capabilities & Feature Detection
- Execution Profiling & Performance Analysis
- Streaming Execution & Memory-Efficient Processing
- Error Recovery & Fault Tolerance
- Autodiff Enhancements (Gradient Accumulation, Clipping, Scaling, Custom Gradients)
- Type-Safe Tensor Wrappers & Compile-Time Checking
- Graph Optimization (Fusion Planning, Dead Code Elimination)
- Execution Scheduling (Sequential, Parallel, Cost-Based, Memory-Efficient)
- Device Placement Optimization
- Memory Management (Caching, Pooling, Estimation, Workspace)
- Execution Context & Lifecycle Hooks
- Debugging Tools (Trace, Inspect, Breakpoints)
- Visualization Utilities (Timeline, Graph, Statistics)
- Graph Compilation & Caching
- Eager Mode Autodiff
- Backend Test Templates
- Gradient Checking
- Performance Regression Testing
- JIT Compilation with Hot Path Detection
- Distributed Execution (Data/Model/Pipeline Parallelism)
- Zero-Copy Tensor Views
- Async Execution [feature = "async"]
- Enhanced Diagnostics
- Mixed Precision Training
- Sparse Tensor Support
- Work-Stealing Parallel Scheduler
- SIMD Optimization Utilities
- Advanced Quantization
- Dynamic Batching
- Advanced Kernel Fusion
- Workspace Management
- Multi-Model Coordination
- Graph Rewriting Engine
- Profiling-Guided Optimization
- Cache Optimization
- Automatic Parallelization (Experimental)
- Speculative Execution (Experimental)
- Learned Optimizations (Experimental)

**Test Coverage:** 909 tests (all passing)
**Build Status:** ZERO ERRORS, ZERO WARNINGS
**Total Lines of Code:** ~26,000 lines Rust code
**Examples:** 3 working examples (jit_demo.rs, distributed_demo.rs, recovery_demo.rs)

**Version**: 0.1.1
**Release Date**: 2026-04-06
**Backward Compatibility**: Maintained

## v0.1.21 (2026-04-05)

- [x] **MCMC Sampling** (`mcmc.rs`): Added mcmc.rs — MCMC sampling: `MetropolisHastings` (pluggable `LogProb` + `Proposal`), `HamiltonianMonteCarlo` (leapfrog + finite-diff gradients); `McmcRng` (LCG+Box-Muller); `GaussianProposal`/`IndependentGaussianProposal`; `effective_sample_size`, `gelman_rubin`, `autocorrelation` diagnostics.

## v0.1.19 (2026-04-05)

- [x] **Causal Inference** (`causal.rs`): `CausalGraph` with directed-edge adjacency representation; d-separation via the Bayes-Ball algorithm (`d_separated()`); backdoor criterion check (`satisfies_backdoor_criterion()`) identifying valid adjustment sets that block all back-door paths from treatment to outcome; frontdoor criterion check (`satisfies_frontdoor_criterion()`); do-calculus intervention operator (`do_intervention()`) returning the post-intervention DAG; `ate_backdoor()` estimator of the Average Treatment Effect under backdoor adjustment; `ate_instrumental_variable()` IV estimator using a provided instrument; `propensity_score()` logistic-approximation propensity scoring; `ObservationalData` struct for passing empirical covariate/treatment/outcome samples.

## v0.1.18 (2026-04-05)

- [x] **Constraint Propagation** (`constraint_propagation.rs`): `ConstraintNetwork` manages variables, `Domain` (ordered set of `f64` values), and `BinaryConstraint` arcs; AC-3 arc-consistency algorithm (`propagate()`) with a workqueue driving domain reduction to a fixed point; `ConstraintRelation` enum with 7 variants (`Equal`, `NotEqual`, `LessThan`, `LessEqual`, `GreaterThan`, `GreaterEqual`, `CustomFn`); `CspSolver` backtracking CSP solver with configurable `VarOrdering` (Lexicographic, `MinRemainingValues` / MRV, `DegreeHeuristic`) and optional forward-checking that runs AC-3 after every variable assignment to prune sibling domains early; `SolveResult` carrying all solutions found (or first-solution-only mode).

## v0.2.0 / Future Work

- Lazy batched execution (fold compatible `forward` calls across a batch).
- Memoization cache for repeated graph sub-patterns.
- Streaming inference API for large graphs.
- Zero-copy tensor sharing between executor and backend.
- [x] ~~Split `src/causal.rs` (1,589 L) into a `causal/` directory.~~ (completed 2026-04-15)
