# TensorLogic SciRS2 Backend — TODO

**Status**: Stable | **Version**: 0.1.3 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-31
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

SciRS2-backed execution engine with SIMD acceleration and autograd.

## Completed

- [x] Basic Scirs2Exec structure implementing TlExecutor trait
- [x] Integration with SciRS2 dependencies
  - [x] scirs2-core for tensor operations
  - [x] scirs2-linalg for einsum
- [x] Test infrastructure setup
- [x] Workspace dependencies configured
- [x] **Real einsum execution**
  - [x] Parse einsum specs from EinsumGraph
  - [x] Execute with scirs2_linalg::einsum
  - [x] Handle multiple operations in sequence
- [x] **Core execution fully implemented**
  - [x] Implement TlAutodiff::forward() method
  - [x] Load input tensors from EinsumGraph
  - [x] Execute each EinsumNode in topological order
  - [x] Handle all OpType variants:
    - [x] Einsum (tensor contraction)
    - [x] ElemUnary (relu, sigmoid, tanh, oneminus, abs, neg, exp, log, sqrt, square, clip)
    - [x] ElemBinary operations:
      - [x] Arithmetic: add, subtract, multiply, divide
      - [x] Comparisons: eq, lt, gt, lte, gte (return 0.0 or 1.0)
    - [x] Reduce (sum, max, min, mean, product over axes)
  - [x] Collect output tensors and return results
- [x] **Tensor management**
  - [x] Store intermediate tensors efficiently
  - [x] Hashmap-based tensor storage
- [x] **Shape validation**
  - [x] Validate einsum specs against tensor shapes
  - [x] Check dimension compatibility
  - [x] Clear error messages on shape mismatch
- [x] **Conversion utilities**
  - [x] from_vec() with shape validation
  - [x] zeros() and ones() tensor constructors
- [x] **Integration tests**
  - [x] End-to-end TLExpr → EinsumGraph → Execution
  - [x] Simple predicate execution
  - [x] EXISTS quantifier with reduction
  - [x] AND operation with shared variables
  - [x] IMPLY operation execution
- [x] **Module refactoring**
  - [x] Separate modules for executor, conversion, ops, autodiff
  - [x] Clean public API

## High Priority - Completed

### Autodiff Support (Backward Pass)
- [x] Implement TlAutodiff::backward() method with proper gradient computation
  - [x] Store forward pass intermediate values in ForwardTape
  - [x] Correct gradient computation for arithmetic operations (multiply, divide with actual input values)
  - [x] Proper gradient computation for unary operations (relu, sigmoid, oneminus)
  - [x] Zero gradients for comparison operations (non-differentiable)
  - [x] Gradient accumulation when tensors used multiple times
  - [x] Proper gradient tracking with node output indices
  - [x] ReLU gradient: grad * (input > 0) with element-wise check
  - [x] Sigmoid gradient: grad * sigmoid(x) * (1 - sigmoid(x)) with proper computation
  - [x] Broadcast gradients back through reduction operations
- [x] Gradient accumulation
  - [x] Support multiple backward passes
  - [x] Accumulate gradients for parameters when used multiple times
  - [x] Proper gradient addition for shared tensors
- [x] Einsum gradient computation
  - [x] Parse einsum specifications to determine gradient contraction patterns
  - [x] Proper gradient computation for matrix multiplication (ij,jk->ik)
  - [x] Proper gradient computation for element-wise operations with explicit indices
  - [x] Automatic gradient spec generation for arbitrary einsum operations
  - [x] Fallback to passthrough for unsupported patterns
- [x] Gradient verification
  - [x] Numeric gradient checking utility with finite differences
  - [x] Configurable epsilon, rtol, atol for gradient comparison
  - [x] Per-tensor gradient comparison with max abs/rel diff reporting
  - [x] Comprehensive gradient verification tests
  - [x] Verified accuracy: gradients match within 10^-10 to 10^-11
- [x] Advanced autodiff features
  - [x] Straight-Through Estimator (STE) for non-differentiable operations
  - [x] Gumbel-Softmax for differentiable categorical sampling
  - [x] Soft quantifiers: differentiable exists and forall
    - [x] Hard mode (max/min), Smooth mode (log-sum-exp), Probabilistic mode
  - [x] 11 comprehensive tests (all passing)
  - [x] Full backward pass support for all gradient estimators
  - [ ] Optional: Integrate scirs2_autograd::Variable for alternative implementation (FUTURE)

### Additional Operations
- [x] Extend logical operations
  - [x] OR: max (OrMax) and probabilistic sum (OrProbSum): 1 - (1-a)(1-b) = a + b - ab
  - [x] NAND: 1 - (a * b)
  - [x] NOR: 1 - max(a, b)
  - [x] XOR (soft): a + b - 2ab
  - [x] Full gradient support for all operations
- [x] Advanced quantifiers
  - [x] FORALL: product reduction implemented (ReduceOp::Product)
  - [x] Product reduction with proper gradient support
  - [ ] Min reduction variant (FUTURE)
  - [x] **Weighted quantifiers** ✅ (v0.1.2) — weighted_exists, weighted_forall with gradients
- [x] Fuzzy/soft logic module (fuzzy_logic.rs)
  - [x] FuzzyFamily enum (Lukasiewicz, Godel, Product, Nilpotent)
  - [x] FuzzyConfig with temperature control
  - [x] soft_and, soft_or, soft_not, soft_imply functions
  - [x] FuzzyLogic struct with full operation set
  - [x] Gradient functions for soft_and/soft_or
  - [x] AnnealingSchedule for temperature annealing
  - [x] 21 comprehensive tests
- [x] **Scoring aggregation** ✅ (v0.1.2) — LogSpaceAggregator, WeightedQuantifier

## Medium Priority - Completed

### Parallelization
- [x] Dependency analysis
  - [x] Graph dependency analyzer (DependencyAnalysis)
  - [x] Topological sorting for execution levels
  - [x] Independent operation detection
  - [x] Parallelism opportunity identification
  - [x] Estimated speedup calculation
  - [x] 8 comprehensive tests (all passing)
- [x] Parallel executor
  - [x] Rayon-based parallel execution
  - [x] Level-by-level parallel processing
  - [x] Thread pool management (configurable via ParallelConfig)
  - [x] Performance comparison benchmarks (parallel_performance.rs)
  - [x] 8 comprehensive tests (all passing)
  - [x] Automatic parallelization based on dependency levels
  - [x] Configurable min_parallel_ops threshold
  - [x] ParallelStats tracking (parallel vs sequential op counts)
  - [x] Full TlAutodiff support (forward and backward passes)
- [ ] Additional parallelization (FUTURE)
  - [ ] Batch execution parallelization
  - [ ] Work stealing for load balancing

### Performance Optimization
- [x] Operation fusion analysis
  - [x] FusionOpportunity detection for consecutive operations
  - [x] Pattern matching (UnaryUnary, BinaryUnary, UnaryBinary, BinaryBinary)
  - [x] FusionStats with estimated speedup calculation
  - [x] 6 tests covering various fusion patterns
  - Note: Analysis-only; actual kernel fusion at execution time is FUTURE
- [x] Memory pooling
  - [x] TensorPool with shape-based reuse
  - [x] Statistics tracking (allocations, reuses, reuse_rate)
  - [x] Zero tensors before reuse to prevent data leakage
  - [x] Integration with Scirs2Exec (enable/disable pooling, pool_stats)
  - [x] 6 tests covering basic pooling, different shapes, statistics
- [x] SIMD support
  - [x] SIMD features configured in Cargo.toml
  - [x] Enable SIMD features in scirs2 (via feature flag)
  - [x] Vectorized element-wise operations (via scirs2)
  - [x] Optimized reductions (via scirs2)
  - [x] Builds successfully with --features simd
  - [ ] SIMD-specific benchmarks (FUTURE)
- [ ] Additional optimizations (FUTURE)
  - [ ] Fuse consecutive einsum operations (requires kernel changes)
  - [x] **Lazy evaluation** ✅ (v0.1.2) — LazyTensor, LazyEinsumGraph, LazyExecutor with memoization

### Backend Features
- [x] Multiple execution modes
  - [x] Eager execution (default)
  - [x] Graph compilation infrastructure
  - [x] ExecutionMode enum with Eager/Graph/JIT modes
  - [x] CompiledGraph with optimization passes
  - [x] ExecutionConfig for mode configuration
  - [x] 18 comprehensive tests (all passing)
  - [ ] JIT compilation (FUTURE)
- [x] Device management
  - [x] CPU backend (default, fully functional)
  - [x] DeviceType enum (CPU/CUDA/Metal/Vulkan/ROCm)
  - [x] Device abstraction with type and index
  - [x] DeviceManager for querying available devices
  - [x] Device selection API
  - [x] CUDA device detection via nvidia-smi (cuda_detect module)
  - [x] GPU readiness assessment (gpu_readiness module)
  - [x] 10 comprehensive tests (all passing)
  - [ ] GPU backend execution (FUTURE, via scirs2 GPU features)
- [x] Precision control
  - [x] Precision enum (F32/F64/Mixed16/BFloat16)
  - [x] Scalar trait for generic f32/f64 operations
  - [x] PrecisionConfig with mixed precision support
  - [x] Loss scaling for mixed precision training
  - [x] 10 comprehensive tests (all passing)
  - [ ] Mixed precision execution (FUTURE)
  - [ ] f32 tensor backend (FUTURE, currently f64 only)

### Error Handling
- [x] Comprehensive error types
  - [x] ShapeMismatchError with details and context
  - [x] InvalidEinsumSpec errors
  - [x] DeviceError (GPU unavailable, allocation failed, sync failed)
  - [x] OutOfMemory errors
  - [x] NumericalError (NaN, Inf, overflow, underflow, division by zero)
  - [x] GradientError, GraphError, ExecutionError
  - [x] Unsupported feature errors
  - [x] Helper functions for creating common errors
  - [x] 7 comprehensive tests covering all error types
- [x] Execution tracing
  - [x] TraceLevel system (None, Error, Warn, Info, Debug, Trace)
  - [x] TraceEvent with timestamps and operation metadata
  - [x] ExecutionTracer with handle-based operation tracking
  - [x] TraceStats for operation counts and performance analysis
  - [x] 6 comprehensive tests for tracing functionality
- [x] Fallback mechanisms
  - [x] FallbackConfig with configurable replacement values
  - [x] Handle NaN/Inf gracefully with sanitize_tensor()
  - [x] Numeric stability checks (contains_nan, contains_inf, is_valid)
  - [x] Value clamping and safe operations (safe_div, safe_log, safe_sqrt)
  - [x] Numerical issue detection with detailed reports
  - [x] Strict and permissive modes
  - [x] 13 comprehensive tests for fallback mechanisms

## Low Priority - Completed

### Checkpoint Support
- [x] Checkpoint infrastructure
  - [x] CheckpointConfig for training/inference modes
  - [x] Checkpoint serialization (JSON format)
  - [x] Checksum verification for data integrity
  - [x] CheckpointMetadata with custom metadata support
  - [x] CheckpointManager for multiple checkpoints
  - [x] Automatic cleanup of old checkpoints
  - [x] 11 comprehensive tests (all passing)

### In-Place Operations
- [x] In-place execution
  - [x] InplaceExecutor with aliasing tracking
  - [x] All unary ops (relu, sigmoid, oneminus, tanh, abs, neg, exp, log, sqrt, square, clip)
  - [x] All binary ops (add, subtract, multiply, divide, min, max)
  - [x] Scalar operations (add_scalar, mul_scalar, pow, clamp_min/max)
  - [x] InplaceStats for memory savings tracking
  - [x] Shape preservation verification
  - [x] 16 comprehensive tests (all passing)

### Monitoring and Profiling
- [x] Performance metrics (metrics module)
  - [x] MetricsCollector with comprehensive tracking
  - [x] Per-operation timing with statistics (min/avg/max)
  - [x] Memory usage tracking (current/peak)
  - [x] Throughput measurement (ops/sec, elements/sec)
  - [x] AtomicMetrics for thread-safe tracking
  - [x] SharedMetrics for concurrent access
  - [x] Export to JSON and CSV formats
  - [x] 19 comprehensive tests (all passing)
- [x] Profiling integration
  - [x] ProfiledScirs2Exec wrapper
  - [x] Operation-level profiling with TraceLevel
  - [x] ExecutionTracer with timestamps and metadata
  - [x] 3 comprehensive tests
- [x] Memory profiler (memory_profiler module)
  - [x] AllocationRecord with detailed tracking
  - [x] AtomicMemoryCounter for thread-safe counting
  - [x] MemoryProfiler with configurable profiling
  - [x] MemoryStats reporting
  - [x] 8 comprehensive tests
- [ ] Additional telemetry (FUTURE)
  - [ ] Integration with external monitoring systems
  - [ ] Real-time streaming metrics

### Benchmarking
- [x] Operation benchmarks
  - [x] Einsum benchmarks (matmul, batch_matmul, transpose, trace)
  - [x] Unary operation benchmarks (relu, sigmoid, tanh)
  - [x] Binary operation benchmarks (add, sub, mul, div)
  - [x] Reduce operation benchmarks (sum, max, min, mean)
  - [x] Logical operation benchmarks (or, nand, nor, xor)
  - [x] Memory pool comparison benchmarks
  - [x] Complex einsum patterns (attention, outer product)
- [ ] End-to-end benchmarks (FUTURE)
  - [ ] Benchmark realistic TLExpr graphs
  - [ ] Memory usage profiling
  - [ ] Scaling tests (graph size)
- [ ] Regression tracking (FUTURE)
  - [ ] Track performance over commits
  - [ ] Automated benchmark CI
  - [ ] Performance alerts

### Advanced Features
- [x] Custom operations
  - [x] CustomOp trait for user-defined operations
  - [x] OpRegistry for dynamic registration
  - [x] Standard ops: softplus, leaky_relu, elu, swish, mish, gelu, hard_sigmoid, hard_swish
  - [x] CustomOpContext for intermediate value storage
  - [x] BinaryCustomOp for element-wise binary operations
  - [x] Forward and backward pass support
  - [x] Input validation and shape inference
  - [x] 16 comprehensive tests (all passing)
- [x] Graph optimization
  - [x] GraphOptimizer with configurable passes
  - [x] Constant folding pass
  - [x] Subgraph caching with hash-based deduplication
  - [x] Algebraic simplification (identity einsum detection)
  - [x] Dead code elimination
  - [x] Operation reordering infrastructure
  - [x] OptimizationStats with reduction percentage
  - [x] GraphOptimizerBuilder for configuration
  - [x] 13 comprehensive tests (all passing)
- [x] Quantization (quantization module)
  - [x] QuantizationType: Int8, Int4, Int2
  - [x] QuantizationScheme: Symmetric, Asymmetric
  - [x] QuantizationGranularity: PerTensor, PerChannel
  - [x] QuantizationParams with scale/zero-point
  - [x] QuantizedTensor with dequantization
  - [x] QatConfig for quantization-aware training
  - [x] calibrate_quantization() for PTQ
  - [x] QuantizationStats for error monitoring
  - [x] 10 comprehensive tests (all passing)
- [ ] Distributed execution (FUTURE)
  - [ ] Split graphs across devices
  - [ ] Data parallelism
  - [ ] Model parallelism

## v0.1.3 Enhancements (2026-03-30)

- [x] **F32 Tensor Executor** (`executor_f32.rs`): `Scirs2Exec32` with full `TlExecutor` impl for `ArrayD<f32>`. All ops: einsum, elem_op (Relu/Sigmoid/OneMinus), binary_op, reduce (Sum/Max/Min/Mean/Product). 15 new tests.
- [x] **Precision Casting** (`precision_cast.rs`): `cast_f64_to_f32()`, `cast_f32_to_f64()`, `DualPrecisionBridge` — f32 compute with f64 gradient accumulation. 10 new tests.

## v0.1.4 Enhancements (2026-03-30)

- [x] **Attention Operations** (`attention.rs`): `MultiHeadAttention::forward()` and `forward_with_weights()`, `scaled_dot_product_attention()`, `chunked_attention()`, `stable_softmax()` (max-subtraction), `attention_entropy()`. `AttentionConfig` builder with n_heads/head_dim/chunk_size/scale/causal. 20 new tests.

## v0.1.5 Enhancements (2026-03-30)

- [x] **Attention Backward Pass** (`attention_grad.rs`): `softmax_backward()` (row-wise Jacobian-vector product), `attention_backward()` (dQ=d_scaled@K, dK=d_scaled^T@Q, dV=W^T@dout), `multihead_attention_backward()` (batch+head loop). Finite-difference numerical verification. 20 new tests.

## v0.1.7 Enhancements (2026-03-30)

- [x] **Tensor Comparison Utilities** (`tensor_compare.rs`): `compare_tensors()` with configurable `Tolerance` (rtol/atol), `ComparisonResult` (max/mean diff, mismatch count, NaN/Inf detection), `assert_tensors_close()`, `abs_diff()`, `is_finite()`, `count_non_finite()`. 18 new tests.

## v0.1.8 Enhancements (2026-03-30)

- [x] **Tensor Binary I/O** (`tensor_io.rs`): Custom `TLTF` binary format, `save_tensor`/`load_tensor` (single), `save_tensors`/`load_tensors` (multi-named), `read_header` for metadata inspection. NaN/Inf preserving. 18 new tests.

## v0.1.1 Additions (2026-03-29)

### GPU Infrastructure Framework ✅ COMPLETE
- [x] **GPU Infrastructure Framework** ✅ COMPLETE (2026-03-29)
  - [x] `src/gpu/` module: GpuDevice, GpuBuffer<T>, GpuMemoryPool, GpuBackend trait
  - [x] CudaStub: graceful fallback (GpuError::NotAvailable)
  - [x] KernelConfig builder pattern, KernelLaunchResult
  - [x] Feature-gated: `#[cfg(feature = "gpu")]`, Pure Rust
  - [x] 24 tests passing
- [ ] GPU kernel implementations (actual CUDA/ROCm kernels) (FUTURE)

## v0.1.10 Enhancements (2026-03-31)

- [x] **Block-Sparse Tensor Format** (`blocked_sparse.rs`): `BlockedSparseTensor` storing non-zero blocks by (block_row, block_col) index with configurable `block_size`; `to_dense()` and `from_dense()` conversion utilities
- [x] **Blocked Sparse Matrix Multiply** (`blocked_sparse.rs`): `blocked_sparse_mm()` for sparse × dense matrix multiplication with zero-block skipping for computational savings
- [x] **Block Sparsity Statistics** (`blocked_sparse.rs`): `BlockSparsityStats` reporting density, total blocks, non-zero blocks, and estimated speedup factor from block skipping
- [x] **Block Sparsity Patterns** (`blocked_sparse.rs`): `BlockSparsityPattern` enum with Diagonal, BandedDiagonal, Random (configurable density), and Strided pattern generators for structured sparsity exploration

## v0.1.11

- [x] **gather + scatter_add + scatter_max/min** (`gather_scatter.rs`): `gather()` extracts elements along a given axis using an index tensor, `scatter_add()` accumulates values at index positions, `scatter_max()` and `scatter_min()` compute segment maxima/minima with neutral-element initialization.
- [x] **top_k + masked_select/fill + IndexStats** (`gather_scatter.rs`): `top_k()` partial sort returning top-k values and their original indices, `masked_select()` boolean-mask filter producing a flat 1-D result, `masked_fill()` conditional in-place fill, `IndexStats` reporting coverage ratio and per-index collision count.

## v0.1.12

- [x] **TruncatedSvd** (`decomposition.rs`): `TruncatedSvd` struct holding left-singular vectors `u` (m×k), singular values `s` (k), and right-singular vectors `vt` (k×n) computed via power-iteration with configurable rank and iteration count.
- [x] **tucker1** (`decomposition.rs`): `tucker1()` mode-n Tucker-1 decomposition returning `Tucker1Result` with a factor matrix (rows = original mode size, cols = rank), a compressed core tensor, and `compression_ratio`.
- [x] **cp_als (ALS)** (`decomposition.rs`): `cp_als()` CP/PARAFAC decomposition via alternating-least-squares returning `CpDecomposition` (factor matrices per mode, weights, iterations, final error) with `reconstruct()` and `explained_variance()` methods.
- [x] **hosvd** (`decomposition.rs`): `hosvd()` higher-order SVD producing `HosvdResult` containing one truncated factor matrix per mode plus a Tucker core, with `compression_ratio` and `reconstruction_error()`.
- [x] **unfold / fold** (`decomposition.rs`): `unfold()` mode-n matricization mapping an N-D tensor to a 2-D matrix, `fold()` inverse operation restoring the original shape; both return `DecompositionError` on invalid inputs.
- [x] **DecompositionError** (`decomposition.rs`): typed error enum covering `NonMatrixInput`, `EmptyTensor`, `RankTooLarge`, `InvalidMode`, `InvalidShape`, `SingularMatrix`, and `ConvergenceFailure` variants.

## v0.1.13

- [x] **max_pool** (`pooling.rs`): Sliding-window maximum pooling with configurable kernel size, stride, and padding; supports 1-D and 2-D inputs
- [x] **avg_pool** (`pooling.rs`): Average pooling with configurable kernel/stride/padding, proper handling of padded regions
- [x] **lp_pool** (`pooling.rs`): Lp-norm pooling for arbitrary p values (p=1 Manhattan, p=2 Euclidean, etc.) with kernel/stride configuration
- [x] **global_max_pool / global_avg_pool** (`pooling.rs`): Reduce all spatial dimensions to single values, returning batch×channel output
- [x] **adaptive_avg_pool** (`pooling.rs`): Target output-size-based average pooling that automatically computes kernel and stride to achieve desired spatial dimensions
- [x] **max_unpool** (`pooling.rs`): Inverse max-pooling using stored argmax indices to place values back into an expanded tensor
- [x] **PoolingStats** (`pooling.rs`): Operation statistics including output shape computation, element count, and receptive field analysis

## v0.1.14

- [x] **conv1d** (`convolution.rs`): 1-D cross-correlation with configurable stride, padding, and dilation; supports multi-batch multi-channel inputs
- [x] **conv2d** (`convolution.rs`): 2-D convolution via im2col matrix multiplication; handles stride, padding, dilation, and groups
- [x] **conv_transpose2d** (`convolution.rs`): Transposed (deconvolution) 2-D convolution for upsampling with configurable output padding
- [x] **depthwise_conv2d** (`convolution.rs`): Channel-wise independent convolution where each input channel is convolved with its own kernel
- [x] **im2col** (`convolution.rs`): Image-to-column transform extracting sliding-window patches into a 2-D matrix for efficient GEMM-based convolution
- [x] **col2im** (`convolution.rs`): Column-to-image inverse transform that accumulates column patches back into the spatial tensor layout
- [x] **ConvStats** (`convolution.rs`): Convolution operation statistics including FLOPs count, parameter count, output shape computation, and summary formatting

## v0.1.16

- [x] **RnnCell** (`recurrent.rs`): Single-step Elman RNN cell computing `h' = tanh(W_ih * x + b_ih + W_hh * h + b_hh)` with configurable input and hidden sizes; `from_weights()` constructor for loading pre-trained parameters
- [x] **LstmCell** (`recurrent.rs`): Long Short-Term Memory cell with input, forget, output, and cell gates; `LstmState` container holding paired hidden state `h` and cell state `c`; `from_weights()` for weight-matrix initialization
- [x] **GruCell** (`recurrent.rs`): Gated Recurrent Unit cell with reset and update gates computing `h' = (1 - z) * n + z * h`; `from_weights()` accepting separate input and hidden weight matrices and biases
- [x] **LstmState** (`recurrent.rs`): Value struct pairing hidden state `h` and cell state `c` as `Array1<f64>` fields; `zeros()` constructor for batch initialization
- [x] **rnn_sequence** (`recurrent.rs`): Unrolls `RnnCell` over a `&[Array1<f64>]` sequence of inputs, returning all intermediate hidden states and the final hidden state
- [x] **lstm_sequence** (`recurrent.rs`): Unrolls `LstmCell` over a variable-length input sequence accumulating `LstmState` at each step, returning the full hidden-state history and final `LstmState`
- [x] **gru_sequence** (`recurrent.rs`): Unrolls `GruCell` over a sequence of inputs, returning all intermediate hidden states and the final hidden vector

## v0.1.15

- [x] **relu** (`activations.rs`): Rectified Linear Unit `max(0, x)` with in-place variant and element-wise application over arbitrary-shaped tensors
- [x] **gelu** (`activations.rs`): Gaussian Error Linear Unit approximation `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))` for transformer feed-forward layers
- [x] **swish** (`activations.rs`): Self-gated activation `x * sigmoid(x)` (also known as SiLU), smooth alternative to ReLU used in EfficientNet and LLaMA
- [x] **mish** (`activations.rs`): `x * tanh(softplus(x))` smooth non-monotonic activation with unbounded positive and bounded negative range
- [x] **selu** (`activations.rs`): Scaled Exponential Linear Unit with fixed `alpha` and `scale` constants enabling self-normalizing networks
- [x] **elu** (`activations.rs`): Exponential Linear Unit `x if x > 0 else alpha*(exp(x)-1)` with configurable alpha parameter
- [x] **softmax** (`activations.rs`): Numerically stable softmax `exp(x - max(x)) / sum(exp(x - max(x)))` with configurable axis parameter
- [x] **prelu** (`activations.rs`): Parametric ReLU with per-channel learnable negative slopes stored as a 1-D tensor; supports broadcasting over batch and spatial dimensions
- [x] **ActivationType** (`activations.rs`): Enum dispatching all activation variants through a unified `activate()` function and an `activate_inplace()` variant for zero-copy execution

## v0.1.17

- [x] **TensorMseLoss** (`tensor_loss.rs`): Mean Squared Error loss `∑(pred - target)² / n` with per-element gradient `2(pred - target) / n`; supports `reduction` mode (mean/sum/none)
- [x] **TensorBCELoss** (`tensor_loss.rs`): Binary Cross-Entropy loss with numerical stability clamp `pred = clamp(pred, ε, 1-ε)`; gradient `-target/pred + (1-target)/(1-pred)`
- [x] **TensorCrossEntropyLoss** (`tensor_loss.rs`): Multi-class cross-entropy combining log-softmax and NLL; label-smoothing support via `smoothing` parameter
- [x] **TensorFocalLoss** (`tensor_loss.rs`): Focal loss `-(1-p)^γ * log(p)` for class-imbalanced datasets; configurable `gamma` focusing parameter and `alpha` class weighting
- [x] **TensorHuberLoss** (`tensor_loss.rs`): Huber (smooth L1) loss `0.5*(pred-target)²` if `|e| ≤ δ`, else `δ*(|e| - δ/2)`; configurable `delta` threshold blending L1/L2 regimes
- [x] **TensorKLDivLoss** (`tensor_loss.rs`): Kullback-Leibler divergence `∑ target * log(target/pred)`; supports `log_input` flag for inputs already in log-space
- [x] **TensorCosineEmbeddingLoss** (`tensor_loss.rs`): Cosine embedding loss `1 - cos(x1, x2)` for positive pairs, `max(0, cos(x1, x2) - margin)` for negative pairs; configurable `margin`
- [x] **TensorLossRegistry** (`tensor_loss.rs`): Dynamic loss lookup by string name with `register()`, `compute()`, `contains()`, and `names()` API; pre-populated with all 7 built-in loss types

## v0.1.19 (2026-04-05)

- [x] **Geometric Deep Learning** (`geometric_ops.rs`): `AdjacencyMatrix` dense graph representation with `add_edge()` / `degree()` / `num_nodes()`; `graph_laplacian()` computing four variants (`Unnormalized`: `D - A`, `Symmetric`: `D^{-1/2}(D-A)D^{-1/2}`, `RandomWalk`: `D^{-1}(D-A)`, `AddSelfLoops`: `D̃ - Ã` after adding self-loops); `gcn_layer()` executing a single Graph Convolutional Network forward pass `σ(Ã_norm · X · W)` with trainable weight matrix; `Rotation3` SO(3) rigid-body rotation with construction from `axis_angle()` (Rodrigues' formula), `euler_xyz()`, and `from_quaternion()`; `spherical_harmonics()` computing real spherical harmonics `Y_l^m` up to degree 2 on a unit-sphere direction vector.

## v0.1.20 (2026-04-05)

- [x] **Signal Processing** (`signal_ops.rs`): `Complex` type with magnitude, phase, conjugate, and `std::ops::{Add,Mul}` trait impls; `dft()` / `idft()` direct O(N²) Discrete Fourier Transform and inverse; `stft()` Short-Time Fourier Transform returning `StftResult` (frames × freqs magnitude/phase matrix) with configurable window type, window size, and hop length; `istft()` overlap-add reconstruction from `StftResult`; `dct()` / `idct()` Discrete Cosine Transform (Type II) and its exact inverse (DCT-III normalised); six `WindowType` variants (`Rectangular`, `Hann`, `Hamming`, `Blackman`, `Triangular`, `FlatTop`); `FirFilter` with `low_pass()` windowed-sinc design and `fir_filter()` direct-convolution application; `MelFilterbank` computing triangular Mel-scale filter weights; `hz_to_mel()` / `mel_to_hz()` frequency conversion helpers.

## v0.2.0 / Future Work

### GPU Acceleration
- [ ] CUDA backend via scirs2
- [ ] Metal backend (macOS)
- [ ] Vulkan compute shaders
- [ ] ROCm for AMD GPUs
- [x] `auto-device-selection` (planned 2026-04-17)
  - **Goal:** Add a `DeviceManager` that picks `Device::Cpu` vs `Device::Gpu(idx)` per op based on tensor size + op type + device availability, behind a stable selection trait so callers can inject overrides.
  - **Design:** `pub trait DeviceSelector { fn select(&self, op: &OpDescriptor, shape: &[usize]) -> Device; }`. Default impl `HeuristicSelector { gpu_threshold_elems: usize, gpu_available: bool }`: GPU iff `gpu_available && shape.iter().product::<usize>() >= threshold && op.kind.is_gpu_friendly()`. `DeviceConfig` exposes `with_gpu_threshold`, `force_cpu`, `force_gpu(idx)`. `OpDescriptor` is a small enum (MatMul, Elementwise, Reduce, Other) — gpu-friendliness defined per variant.
  - **Files:** `src/device_manager.rs` (NEW); `src/lib.rs` (export DeviceManager, DeviceSelector, HeuristicSelector, DeviceConfig, OpDescriptor).
  - **Prerequisites:** none — `Device` enum already exists in this crate.
  - **Tests:** unit tests in `device_manager.rs` covering: tiny tensor → CPU; large tensor → GPU when available; large tensor → CPU when GPU unavailable; force overrides; per-op-kind selection.
  - **Risk:** Heuristic correctness drift over time — mitigation: conservative default thresholds, explicit override API, doc the heuristic.

### Advanced Backends
- [ ] TPU support
- [ ] WebGPU for browser execution
- [ ] FPGA acceleration
- [ ] Custom ASIC integration

### Quantization (Extended)
- [ ] INT8 quantization execution (currently analysis/PTQ only)
- [ ] Mixed precision (FP16/BF16) execution
- [ ] Dynamic quantization at runtime
- [ ] Quantization-aware training integration with optimizer

### Compiler Integration
- [ ] XLA integration
- [ ] TVM integration
- [ ] Custom IR lowering
- [ ] Kernel fusion at execution time

### Interoperability
- [ ] Export to ONNX Runtime
- [ ] Execute with TensorFlow
- [ ] Execute with PyTorch
- [ ] Execute with JAX

### Probabilistic Execution
- [x] Monte Carlo sampling — `probabilistic/sampling.rs`: `sample_bernoulli`, `sample_uniform`, `sample_normal`, `sample_categorical` (Gumbel-max), `mc_integrate`; `MonteCarloConfig`
- [x] Variational inference — `probabilistic/variational.rs`: `VariationalInference::fit` — mean-field Gaussian VI with reparameterization trick + Adam SGA, ELBO maximization; recovers Gaussian targets
- [x] Probabilistic programming integration — `probabilistic/mod.rs`: unified `probabilistic` module integrated into `Scirs2Exec` ecosystem via re-exports in `lib.rs`
- [x] Uncertainty quantification — `probabilistic/uncertainty.rs`: `MonteCarloEstimator` (mean/var/CI), `predictive_entropy`, `bald_epistemic_uncertainty` (BALD = H(mean)−mean(H))

---

**Total Tests:** 669 (all passing, 100%)
**Overall Completion:** ~96% - Core execution complete, production-ready autodiff, parallel execution, backend features, comprehensive monitoring/profiling, checkpointing, in-place operations, custom operations, graph optimization, quantization, fuzzy logic, loss functions (v0.1.17), geometric deep learning (v0.1.19), signal processing (v0.1.20), error handling, testing, and documentation
**Remaining:** GPU execution, JIT compilation, Mixed precision execution, Distributed execution
