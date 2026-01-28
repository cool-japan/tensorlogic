# tensorlogic-infer

[![Crate](https://img.shields.io/badge/crates.io-tensorlogic--infer-orange)](https://crates.io/crates/tensorlogic-infer)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/tensorlogic-infer)
[![Tests](https://img.shields.io/badge/tests-285-brightgreen)](#)
[![Production](https://img.shields.io/badge/status-production_ready-success)](#)
[![Completion](https://img.shields.io/badge/completion-100%25-success)](#)

Engine-agnostic execution traits, optimization utilities, and planning API for TensorLogic.

## Overview

`tensorlogic-infer` provides the abstract execution interface and comprehensive optimization infrastructure for TensorLogic backends. This crate defines **traits** that backends must implement, along with powerful utilities for optimization, scheduling, profiling, and memory management.

### Key Components

#### Core Execution Traits
- **TlExecutor**: Basic forward execution of compiled graphs
- **TlAutodiff**: Forward/backward pass for automatic differentiation
- **TlEagerAutodiff**: 🆕 Eager mode autodiff with dynamic graph building
- **TlBatchExecutor**: Efficient batch execution with parallel support
- **TlStreamingExecutor**: Streaming execution for large datasets
- **TlCompilableExecutor**: Ahead-of-time graph compilation support
- **TlJitExecutor**: 🆕 Just-In-Time compilation with hot path detection
- **TlDistributedExecutor**: 🆕 Multi-device distributed execution
- **TlRecoverableExecutor**: Execution with error recovery and checkpointing
- **TlCapabilities**: Backend capability queries (devices, dtypes, features)
- **TlProfiledExecutor**: Execution profiling and performance analysis

#### Optimization Infrastructure
- **GraphOptimizer**: Fusion detection, dead node elimination, redundancy analysis
- **FusionPlanner**: Planning and validation of operation fusion
- **Scheduler**: Execution scheduling (sequential, parallel, cost-based)
- **PlacementOptimizer**: Multi-device placement and coordination
- **GraphCompiler**: AOT graph compilation with multiple optimization levels
- **CompilationCache**: Caching of compiled graphs to avoid recompilation
- **MemoryEstimator**: Memory usage estimation and lifetime analysis
- **ShapeInferenceContext**: Tensor shape inference for optimization

#### Runtime Utilities
- **TensorCache**: Result caching with LRU/FIFO/LFU eviction
- **MemoryPool**: Tensor memory pooling for allocation reuse
- **ExecutionStrategy**: Complete strategy configuration
- **ExecutionContext**: State management with lifecycle hooks
- **GraphValidator**: Graph validation and diagnostics

#### Testing & Development Tools 🆕
- **BackendTestAdapter**: Comprehensive test templates for backend validation
- **GradientChecker**: Numerical gradient checking for autodiff verification
- **PerfRegression**: Performance regression testing with baseline comparison
- **Variable & EagerTape**: Eager mode execution with gradient tracking

## Quick Start

```rust
use tensorlogic_infer::{TlExecutor, TlAutodiff};
use tensorlogic_scirs_backend::Scirs2Exec;
use tensorlogic_ir::EinsumGraph;

// Create executor
let mut executor = Scirs2Exec::new();

// Forward pass
let outputs = executor.forward(&graph, &inputs)?;

// Backward pass
executor.backward(&outputs, &gradients)?;
let param_grads = executor.get_gradients()?;
```

## Core Traits

### TlExecutor

Basic execution interface for forward passes:

```rust
pub trait TlExecutor {
    type Tensor;
    type Error;

    fn execute(
        &self,
        graph: &EinsumGraph,
        inputs: &HashMap<String, Self::Tensor>,
    ) -> Result<Vec<Self::Tensor>, Self::Error>;
}
```

### TlAutodiff

Automatic differentiation support:

```rust
pub trait TlAutodiff: TlExecutor {
    fn forward(
        &mut self,
        graph: &EinsumGraph,
        inputs: &HashMap<String, Self::Tensor>,
    ) -> Result<Vec<Self::Tensor>, Self::Error>;

    fn backward(
        &mut self,
        outputs: &[Self::Tensor],
        output_grads: &[Self::Tensor],
    ) -> Result<(), Self::Error>;

    fn get_gradients(&self) -> Result<HashMap<String, Self::Tensor>, Self::Error>;
}
```

### TlBatchExecutor

Efficient batch execution with parallel support:

```rust
pub trait TlBatchExecutor: TlExecutor {
    fn execute_batch(
        &mut self,
        graph: &EinsumGraph,
        batch_inputs: Vec<HashMap<String, Self::Tensor>>,
    ) -> Result<BatchResult<Self::Tensor>, Self::Error>;

    fn execute_batch_parallel(
        &mut self,
        graph: &EinsumGraph,
        batch_inputs: Vec<HashMap<String, Self::Tensor>>,
        num_threads: Option<usize>,
    ) -> Result<BatchResult<Self::Tensor>, Self::Error>;

    fn optimal_batch_size(&self, graph: &EinsumGraph) -> usize;
}
```

### TlStreamingExecutor

Streaming execution for large datasets:

```rust
pub trait TlStreamingExecutor {
    type Tensor;
    type Error;

    fn execute_stream(
        &mut self,
        graph: &EinsumGraph,
        input_stream: Vec<Vec<Vec<Self::Tensor>>>,
        config: &StreamingConfig,
    ) -> Result<Vec<StreamResult<Self::Tensor>>, Self::Error>;

    fn execute_chunk(
        &mut self,
        graph: &EinsumGraph,
        chunk_inputs: Vec<Self::Tensor>,
        metadata: &ChunkMetadata,
    ) -> Result<StreamResult<Self::Tensor>, Self::Error>;
}
```

**Streaming Modes:**
```rust
use tensorlogic_infer::{StreamingMode, StreamingConfig};

// Fixed chunk size
let config = StreamingConfig::new(StreamingMode::FixedChunk(64))
    .with_prefetch(2)
    .with_checkpointing(100);

// Dynamic chunk sizing based on memory
let config = StreamingConfig::new(StreamingMode::DynamicChunk {
    target_memory_mb: 512,
});

// Adaptive chunking based on performance
let config = StreamingConfig::new(StreamingMode::Adaptive {
    initial_chunk: 32,
});
```

### TlCapabilities

Query backend capabilities:

```rust
pub trait TlCapabilities {
    fn capabilities(&self) -> BackendCapabilities;
}

// Example usage
let caps = executor.capabilities();
println!("Devices: {:?}", caps.devices);
println!("DTypes: {:?}", caps.dtypes);
println!("Features: {:?}", caps.features);
```

### TlProfiledExecutor

Execution profiling and performance analysis:

```rust
pub trait TlProfiledExecutor: TlExecutor {
    fn enable_profiling(&mut self);
    fn disable_profiling(&mut self);
    fn get_profile_data(&self) -> ProfileData;
}

// Example usage
executor.enable_profiling();
executor.execute(&graph, &inputs)?;
let profile = executor.get_profile_data();

for (op_name, stats) in &profile.op_profiles {
    println!("{}: avg={}ms, count={}",
        op_name, stats.avg_time_ms, stats.count);
}
```

### TlJitExecutor

Just-In-Time compilation with hot path detection and adaptive optimization:

```rust
pub trait TlJitExecutor: TlExecutor {
    fn execute_jit(
        &mut self,
        graph: &EinsumGraph,
        inputs: &HashMap<String, Self::Tensor>,
        config: &JitConfig,
    ) -> Result<Vec<Self::Tensor>, Self::Error>;

    fn get_jit_stats(&self) -> JitStats;
    fn clear_jit_cache(&mut self);
}

// Example usage
use tensorlogic_infer::{TlJitExecutor, JitConfig};

let config = JitConfig::default()
    .with_hot_path_threshold(10)
    .with_max_cache_size(100);

let outputs = executor.execute_jit(&graph, &inputs, &config)?;
let stats = executor.get_jit_stats();

println!("Hot paths detected: {}", stats.hot_paths_detected);
println!("Cache hit rate: {:.2}%", stats.cache_hit_rate * 100.0);
```

**JIT Features:**
- **Hot Path Detection**: Automatically identifies frequently executed code paths
- **Adaptive Optimization**: Progressively optimizes based on runtime profiling
- **Graph Specialization**: Specializes graphs for observed tensor shapes
- **Intelligent Caching**: LRU-based cache for compiled graphs

### TlDistributedExecutor

Multi-device distributed execution with data/model/pipeline parallelism:

```rust
pub trait TlDistributedExecutor {
    type Tensor;
    type Error;

    fn execute_distributed(
        &mut self,
        graph: &EinsumGraph,
        inputs: &HashMap<String, Self::Tensor>,
        config: &DistributedConfig,
    ) -> Result<Vec<Self::Tensor>, Self::Error>;

    fn get_distributed_stats(&self) -> DistributedStats;
}

// Example usage - Data Parallelism
use tensorlogic_infer::{
    DistributedConfig, DistributedParallelismStrategy, Device
};

let devices = vec![Device::GPU(0), Device::GPU(1), Device::GPU(2), Device::GPU(3)];
let config = DistributedConfig::new(devices)
    .with_strategy(DistributedParallelismStrategy::DataParallel {
        num_replicas: 4,
    });

let outputs = executor.execute_distributed(&graph, &inputs, &config)?;
let stats = executor.get_distributed_stats();

println!("Communication time: {}ms", stats.communication_time_ms);
println!("Computation time: {}ms", stats.computation_time_ms);
println!("Efficiency: {:.2}%", stats.efficiency * 100.0);
```

**Distributed Parallelism Strategies:**

**Data Parallelism**: Replicate model across devices, split data
```rust
DistributedParallelismStrategy::DataParallel {
    num_replicas: 4,  // 4 GPUs
}
```

**Model Parallelism**: Split model across devices
```rust
DistributedParallelismStrategy::ModelParallel {
    sharding_spec: ShardingSpec::new()
        .shard_tensor("weights", 0, 4),  // Shard along dimension 0
}
```

**Pipeline Parallelism**: Split model into stages
```rust
DistributedParallelismStrategy::PipelineParallel {
    num_stages: 4,
    micro_batch_size: 32,
}
```

**Hybrid Parallelism**: Combine multiple strategies
```rust
DistributedParallelismStrategy::Hybrid {
    data_parallel_groups: 2,
    model_parallel_size: 2,
    pipeline_stages: 2,
}
```

### TlRecoverableExecutor

Execution with error recovery, checkpointing, and fault tolerance:

```rust
pub trait TlRecoverableExecutor: TlExecutor {
    fn execute_with_recovery(
        &mut self,
        graph: &EinsumGraph,
        inputs: &HashMap<String, Self::Tensor>,
        config: &RecoveryConfig,
    ) -> RecoveryResult<Vec<Self::Tensor>, Self::Error>;

    fn save_checkpoint(&mut self, path: &str) -> Result<(), Self::Error>;
    fn load_checkpoint(&mut self, path: &str) -> Result<(), Self::Error>;
}

// Example usage
use tensorlogic_infer::{RecoveryConfig, RecoveryStrategy, RetryPolicy};

let config = RecoveryConfig::default()
    .with_strategy(RecoveryStrategy::RetryWithBackoff)
    .with_retry_policy(RetryPolicy::exponential(3, 100))
    .with_checkpointing(true);

match executor.execute_with_recovery(&graph, &inputs, &config)? {
    RecoveryResult::Success { result, stats } => {
        println!("Success after {} retries", stats.retries);
    }
    RecoveryResult::PartialSuccess { result, failed_nodes, stats } => {
        println!("Partial success: {} nodes failed", failed_nodes.len());
    }
    RecoveryResult::Failure { error, stats } => {
        println!("Failed after {} retries", stats.retries);
    }
}
```

**Recovery Strategies:**
- **RetryWithBackoff**: Exponential backoff retry
- **Checkpoint**: Periodic checkpointing with restart
- **FallbackExecution**: Fall back to alternative execution path
- **GracefulDegradation**: Continue with reduced functionality

## Beta.1 Features 🆕

### Zero-Copy Tensor Operations

Efficient memory-safe tensor views and slicing without data duplication:

```rust
use tensorlogic_infer::{TensorView, SliceSpec, ViewBuilder, TensorViewable};

// Create a tensor view
let view = TensorView::new(base_tensor_id, vec![
    SliceSpec::Range(10..50),
    SliceSpec::Full,
]);

// Check properties
println!("Is contiguous: {}", view.is_contiguous());
println!("Rank: {}", view.rank());

// Ergonomic view builder
let view = ViewBuilder::new(tensor_id, 3)
    .range_dim(0, 10, 20)  // Slice dimension 0
    .index_dim(1, 5)       // Index dimension 1
    .with_offset(100)
    .build();

// Compose views (create view of a view)
let composed = view1.compose(&view2)?;

// Slice specifications
let specs = vec![
    SliceSpec::Full,                              // Full dimension
    SliceSpec::Range(0..100),                     // Range slice
    SliceSpec::Index(42),                         // Single index
    SliceSpec::Strided { start: 0, end: 100, stride: 2 },  // Every 2nd element
    SliceSpec::Reverse,                           // Reverse order
];
```

**Key Features:**
- **Zero-copy views**: No data duplication
- **Flexible slicing**: Range, index, strided, and reverse slices
- **View composition**: Create views of views
- **Contiguity checks**: Optimize based on memory layout
- **In-place operations**: Safe in-place computation support

**Use Cases:**
- Large tensor slicing without memory overhead
- Windowed operations on sequences
- Batch processing with tensor views
- Memory-efficient data augmentation

### Async Execution

Non-blocking execution with async/await support (feature-gated):

```rust
use tensorlogic_infer::{
    TlAsyncExecutor, TlAsyncBatchExecutor,
    AsyncExecutorPool, AsyncConfig
};

// Enable async feature in Cargo.toml
// [dependencies]
// tensorlogic-infer = { version = "*", features = ["async"] }

// Async execution
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut executor = MyAsyncExecutor::new();

    let outputs = executor.execute_async(&graph, &inputs).await?;
    println!("Got {} outputs", outputs.len());

    Ok(())
}

// Async batch processing
let batch_outputs = executor.execute_batch_async(&graph, batch_inputs).await?;

// Async streaming with backpressure
let config = AsyncConfig::default()
    .with_max_concurrent(4)
    .with_backpressure_threshold(100);

let stream_results = executor
    .execute_stream_async(&graph, input_stream, &config)
    .await?;

// Load-balanced executor pool
let pool = AsyncExecutorPool::new(vec![
    executor1,
    executor2,
    executor3,
    executor4,
]);

// Pool automatically distributes work
let output = pool.execute(&graph, &inputs).await?;

// Cancellable execution
let handle = executor.execute_async(&graph, &inputs);
// ... later ...
handle.cancel();

let stats = pool.stats();
println!("Total executions: {}", stats.total_executions);
println!("Average queue time: {}ms", stats.avg_queue_time_ms);
```

**Key Features:**
- **Non-blocking execution**: Use async/await for concurrency
- **Async batch processing**: Process multiple inputs concurrently
- **Async streaming**: Stream processing with backpressure control
- **Executor pooling**: Load-balanced execution across multiple backends
- **Cancellation support**: Cancel long-running operations
- **Feature-gated**: Optional async support to minimize dependencies

**Use Cases:**
- Web services with concurrent requests
- Real-time inference pipelines
- Distributed training coordination
- Resource-efficient batch processing

### Enhanced Diagnostics

Rich error messages with helpful suggestions and context:

```rust
use tensorlogic_infer::{
    Diagnostic, DiagnosticCollector, Severity,
    ShapeMismatchDiagnostic, MemoryDiagnostic,
    PerformanceDiagnostic, SourceLocation,
};

// Create diagnostic with context
let diag = Diagnostic::error("Tensor operation failed")
    .with_code("E001")
    .with_context("Expected shape [64, 128], got [64, 256]")
    .with_suggestion("Use tensor.reshape([64, 128]) to match expected shape")
    .with_suggestion("Check input tensor dimensions")
    .with_location(
        SourceLocation::new()
            .with_file("model.rs".to_string())
            .with_line(42)
    );

println!("{}", diag.format());

// Shape mismatch diagnostics
let expected = TensorShape::static_shape(vec![64, 128]);
let actual = TensorShape::static_shape(vec![64, 256]);
let diag = ShapeMismatchDiagnostic::create(&expected, &actual, "matmul");

// Memory diagnostics
let diag = MemoryDiagnostic::out_of_memory(
    1024 * 1024 * 1024,  // 1 GB requested
    512 * 1024 * 1024     // 512 MB available
);
println!("{}", diag);  // Includes helpful suggestions

// Performance diagnostics
let diag = PerformanceDiagnostic::slow_operation(
    "einsum",
    150.0,  // actual: 150ms
    50.0    // expected: 50ms
);

// Diagnostic collector
let mut collector = DiagnosticCollector::new();
collector.add(diag1);
collector.add(diag2);
collector.add(diag3);

if collector.has_errors() {
    println!("{}", collector.format_all());
    println!("Errors: {}, Warnings: {}",
        collector.error_count(),
        collector.warning_count()
    );
}
```

**Example Output:**
```
[ERROR] Shape mismatch in matmul operation
  at model.rs:42
  code: E001

Context:
  Expected shape: [64, 128], but got: [64, 256]
  Dimension 1 mismatch: expected Static(128), got Static(256)

Suggestions:
  1. Check your input tensor shapes match the expected dimensions
  2. Use tensor.reshape([64, 128]) to match the expected shape

Summary: 1 error(s), 0 warning(s)
```

**Diagnostic Types:**
- **Shape mismatch**: Detailed shape error analysis
- **Type mismatch**: Type conversion suggestions
- **Memory errors**: Out-of-memory with mitigation strategies
- **Performance warnings**: Slow operations with optimization hints
- **Node execution errors**: Failed operations with graph context

**Severity Levels:**
- **Info**: Informational messages
- **Warning**: Non-fatal issues
- **Error**: Fatal errors preventing execution
- **Critical**: System-level issues

## Graph Compilation

### TlCompilableExecutor

Ahead-of-time graph compilation with multiple optimization levels:

```rust
pub trait TlCompilableExecutor: TlExecutor {
    fn compile_graph(
        &mut self,
        graph: &EinsumGraph,
        config: &CompilationConfig,
    ) -> Result<CompiledGraph, Self::Error>;

    fn execute_compiled(
        &mut self,
        compiled: &CompiledGraph,
        inputs: &HashMap<String, Self::Tensor>,
    ) -> Result<Vec<Self::Tensor>, Self::Error>;
}

// Example usage
use tensorlogic_infer::{
    TlCompilableExecutor, CompilationConfig, OptimizationLevel, GraphCompiler
};

let config = CompilationConfig::default()
    .with_optimization_level(OptimizationLevel::Aggressive)
    .with_fusion_enabled(true)
    .with_constant_folding(true);

// Compile once
let compiled = executor.compile_graph(&graph, &config)?;

// Execute multiple times with different inputs
let outputs1 = executor.execute_compiled(&compiled, &inputs1)?;
let outputs2 = executor.execute_compiled(&compiled, &inputs2)?;
let outputs3 = executor.execute_compiled(&compiled, &inputs3)?;

// Check compilation statistics
let stats = compiled.compilation_stats();
println!("Nodes before: {}", stats.nodes_before_optimization);
println!("Nodes after: {}", stats.nodes_after_optimization);
println!("Reduction: {:.2}%", stats.reduction_percentage);
```

**Optimization Levels:**
- **None**: No optimization, fastest compilation
- **Basic**: Dead code elimination only
- **Standard**: DCE + common subexpression elimination
- **Aggressive**: All optimizations + fusion planning

**Compilation Cache:**
```rust
use tensorlogic_infer::{CompilationCache, CompilationKey};

let mut cache = CompilationCache::new(100); // Cache up to 100 graphs

// Automatic caching
let key = CompilationKey::from_graph(&graph, &config);
if let Some(compiled) = cache.get(&key) {
    println!("Cache hit!");
} else {
    let compiled = executor.compile_graph(&graph, &config)?;
    cache.insert(key, compiled);
}

let stats = cache.stats();
println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
```

## Optimization Utilities

### GraphOptimizer

Analyze and optimize computation graphs:

```rust
use tensorlogic_infer::{GraphOptimizer, OptimizationResult};

let optimizer = GraphOptimizer::new();
let result: OptimizationResult = optimizer.analyze(&graph);

println!("Fusion opportunities: {}", result.fusion_opportunities.len());
println!("Dead nodes: {}", result.dead_nodes.len());
println!("Estimated speedup: {:.2}x", result.estimated_speedup);
```

### FusionPlanner

Plan operation fusion:

```rust
use tensorlogic_infer::{FusionPlanner, FusionType};

let planner = FusionPlanner::new();
let opportunities = planner.find_fusion_opportunities(&graph);

for opp in &opportunities {
    match opp.fusion_type {
        FusionType::ElementWise => println!("Can fuse element-wise ops"),
        FusionType::Reduction => println!("Can fuse reduction ops"),
        FusionType::Einsum => println!("Can merge einsum operations"),
    }
}
```

### Scheduler

Execution scheduling with multiple strategies:

```rust
use tensorlogic_infer::{Scheduler, SchedulingStrategy};

let scheduler = Scheduler::new(SchedulingStrategy::CostBased {
    cost_threshold: 1000,
});

let schedule = scheduler.schedule(&graph)?;
println!("Execution order: {:?}", schedule.node_order);
println!("Parallel groups: {:?}", schedule.parallel_groups);
```

**Scheduling Strategies:**
- `Sequential`: Simple topological order
- `Parallel`: Maximize parallelism across independent nodes
- `CostBased`: Balance parallelism with execution cost

### PlacementOptimizer

Multi-device placement optimization:

```rust
use tensorlogic_infer::{PlacementOptimizer, PlacementStrategy, Device};

let devices = vec![Device::CPU(0), Device::GPU(0)];
let optimizer = PlacementOptimizer::new(devices, PlacementStrategy::LoadBalance);

let plan = optimizer.optimize(&graph)?;
for (node_id, device) in &plan.node_placements {
    println!("Node {} -> {:?}", node_id, device);
}
```

### Memory Management

**TensorCache**: Cache computation results

```rust
use tensorlogic_infer::{TensorCache, EvictionPolicy};

let mut cache = TensorCache::new(EvictionPolicy::LRU, 1000); // 1000 MB limit

// Cache usage is automatic when integrated with executor
cache.insert(key, tensor);
if let Some(tensor) = cache.get(&key) {
    // Cache hit
}
```

**MemoryPool**: Reuse tensor allocations

```rust
use tensorlogic_infer::MemoryPool;

let mut pool = MemoryPool::new();

// Allocate or reuse
let tensor = pool.allocate(shape)?;

// Return to pool
pool.deallocate(tensor);

// Stats
let stats = pool.stats();
println!("Reuse rate: {:.2}%", stats.reuse_rate * 100.0);
```

### ExecutionStrategy

Configure complete execution strategy:

```rust
use tensorlogic_infer::{
    ExecutionStrategy, ExecutionMode, PrecisionMode,
    MemoryStrategy, ParallelismStrategy, GradientStrategy,
};

let strategy = ExecutionStrategy {
    mode: ExecutionMode::Graph,  // Graph, Eager, or JIT
    precision: PrecisionMode::FP32,
    memory: MemoryStrategy::Optimize,
    parallelism: ParallelismStrategy::Auto,
    gradient: GradientStrategy::Eager,
};

let optimizer = StrategyOptimizer::new();
let optimized = optimizer.optimize_for_throughput(&graph, &strategy);
```

### ExecutionContext

Manage execution state with lifecycle hooks:

```rust
use tensorlogic_infer::{ExecutionContext, LoggingHook, ExecutionPhase};

let mut context = ExecutionContext::new();
context.add_hook(Box::new(LoggingHook::new()));

context.notify(ExecutionPhase::GraphLoad);
context.notify(ExecutionPhase::Execution);
context.notify(ExecutionPhase::Complete);
```

## Validation and Analysis

### GraphValidator

Validate computation graphs:

```rust
use tensorlogic_infer::GraphValidator;

let validator = GraphValidator::new();
let result = validator.validate(&graph);

if !result.is_valid() {
    for error in &result.errors {
        println!("Error: {}", error);
    }
}
```

### MemoryEstimator

Estimate memory usage:

```rust
use tensorlogic_infer::MemoryEstimator;

let estimator = MemoryEstimator::new();
let estimate = estimator.estimate(&graph);

println!("Peak memory: {} MB", estimate.peak_memory_mb);
println!("Tensor lifetimes: {:?}", estimate.lifetimes);
```

### ShapeInferenceContext

Infer tensor shapes:

```rust
use tensorlogic_infer::ShapeInferenceContext;

let mut ctx = ShapeInferenceContext::new();
ctx.set_input_shape("x", vec![64, 10]);

let inferred = ctx.infer_shapes(&graph)?;
for (tensor_id, shape) in &inferred {
    println!("{}: {:?}", tensor_id, shape);
}
```

## Debugging Tools

### ExecutionTracer

Record and analyze execution flow:

```rust
use tensorlogic_infer::debug::ExecutionTracer;

let mut tracer = ExecutionTracer::new();
tracer.enable();
tracer.start_trace(Some(graph_id));

// Execute operations...
let handle = tracer.record_operation_start(node_id, "einsum", input_ids);
// ... operation execution ...
tracer.record_operation_end(handle, node_id, "einsum", input_ids, output_ids, metadata);

// Get trace
let trace = tracer.get_trace();
let summary = trace.summary();
println!("Total operations: {}", summary.total_operations);
println!("Total time: {:.2}ms", summary.total_time_ms);

// Find slowest operations
let slowest = trace.slowest_operations(5);
for entry in slowest {
    println!("Node {}: {:.2}ms", entry.node_id, entry.duration_ms());
}
```

### TensorInspector

Examine intermediate tensor values:

```rust
use tensorlogic_infer::debug::{TensorInspector, TensorStats};

let mut inspector = TensorInspector::new();
inspector.enable();
inspector.watch(tensor_id); // Watch specific tensor

// Record statistics
let stats = TensorStats::new(tensor_id, vec![64, 128], "f64")
    .with_statistics(min, max, mean, std_dev, num_nans, num_infs);
inspector.record_stats(stats);

// Check for numerical issues
let problematic = inspector.find_problematic_tensors();
for tensor in problematic {
    println!("Tensor {} has {} NaNs, {} Infs",
        tensor.tensor_id,
        tensor.num_nans.unwrap_or(0),
        tensor.num_infs.unwrap_or(0)
    );
}
```

### BreakpointManager

Pause execution for debugging:

```rust
use tensorlogic_infer::debug::{BreakpointManager, Breakpoint};

let mut breakpoints = BreakpointManager::new();
breakpoints.enable();

// Add various breakpoint types
breakpoints.add_node_breakpoint(node_id);
breakpoints.add_operation_breakpoint("matmul");
breakpoints.add_numerical_issue_breakpoint();
breakpoints.add_time_threshold_breakpoint(5000); // 5ms

// Check during execution
if let Some(hit) = breakpoints.should_break(node_id, op_name, elapsed_us, has_nan) {
    println!("Breakpoint hit at node {}", hit.node_id);
    // Inspect state, then continue
    breakpoints.continue_execution();
}
```

### ExecutionRecorder

Full execution recording for replay:

```rust
use tensorlogic_infer::debug::ExecutionRecorder;

let mut recorder = ExecutionRecorder::new();
recorder.enable();

// All debugging features enabled
recorder.tracer().start_trace(Some(graph_id));
recorder.inspector().watch(tensor_id);
recorder.breakpoints().add_node_breakpoint(5);

// Generate comprehensive report
let report = recorder.generate_report();
println!("{}", report);
```

## Advanced Profiling

### TimelineProfiler

Create detailed execution timelines:

```rust
use tensorlogic_infer::{TimelineProfiler, ProfilerHook};

let mut profiler = TimelineProfiler::new();
let hook = ProfilerHook::new(&mut profiler);

// Attach to context
context.add_hook(Box::new(hook));

// Execute
executor.execute(&graph, &inputs)?;

// Analyze timeline
let entries = profiler.entries();
for entry in entries {
    println!("{}: {}ms", entry.name, entry.duration_ms);
}
```

### BottleneckAnalyzer

Identify performance bottlenecks:

```rust
use tensorlogic_infer::BottleneckAnalyzer;

let analyzer = BottleneckAnalyzer::new();
let report = analyzer.analyze(&profile_data);

println!("Bottlenecks:");
for bottleneck in &report.bottlenecks {
    println!("  {}: {:.2}% of total time",
        bottleneck.operation,
        bottleneck.percentage);
}

println!("\nRecommendations:");
for rec in &report.recommendations {
    println!("  - {}", rec);
}
```

### PerformanceComparison

Compare execution strategies:

```rust
use tensorlogic_infer::PerformanceComparison;

let baseline = PerformanceBaseline::from_profile(&profile1);
let comparison = PerformanceComparison::new(baseline, &profile2);

println!("Speedup: {:.2}x", comparison.speedup);
println!("Memory reduction: {:.2}%", comparison.memory_reduction_pct);
```

## Testing Support

### DummyExecutor

Minimal executor for testing:

```rust
use tensorlogic_infer::DummyExecutor;

let executor = DummyExecutor::new();
let outputs = executor.execute(&graph, &inputs)?;
// Returns empty outputs for testing
```

## Examples

### Basic Execution

```rust
use tensorlogic_infer::TlExecutor;
use tensorlogic_scirs_backend::Scirs2Exec;
use std::collections::HashMap;

let executor = Scirs2Exec::new();
let mut inputs = HashMap::new();
inputs.insert("x".to_string(), tensor_x);

let outputs = executor.execute(&graph, &inputs)?;
```

### Batch Processing

```rust
use tensorlogic_infer::TlBatchExecutor;

let batch_inputs = vec![inputs1, inputs2, inputs3];
let result = executor.execute_batch_parallel(&graph, batch_inputs, Some(4))?;

println!("Processed {} items", result.len());
println!("Batch time: {}ms", result.total_time_ms);
```

### Streaming Large Datasets

```rust
use tensorlogic_infer::{TlStreamingExecutor, StreamingConfig, StreamingMode};

let config = StreamingConfig::new(StreamingMode::Adaptive {
    initial_chunk: 64,
}).with_prefetch(2);

let results = executor.execute_stream(&graph, input_stream, &config)?;

for result in results {
    println!("Chunk {}: {} items in {}ms",
        result.metadata.chunk_id,
        result.metadata.size,
        result.processing_time_ms);
}
```

### Training with Autodiff

```rust
use tensorlogic_infer::TlAutodiff;

// Forward pass
let outputs = executor.forward(&graph, &inputs)?;

// Compute loss gradients
let loss_grads = compute_loss_gradients(&outputs, &targets);

// Backward pass
executor.backward(&outputs, &loss_grads)?;

// Get parameter gradients
let grads = executor.get_gradients()?;

// Update parameters
for (param_name, grad) in grads {
    update_parameter(&param_name, &grad);
}
```

## Architecture

```
tensorlogic-infer
├── Core Traits
│   ├── TlExecutor (basic execution)
│   ├── TlAutodiff (training with gradients)
│   ├── TlEagerAutodiff (eager mode autodiff) 🆕
│   ├── TlAsyncExecutor (async/await execution) 🆕 Alpha.2
│   ├── TlAsyncBatchExecutor (async batching) 🆕 Alpha.2
│   ├── TlAsyncStreamExecutor (async streaming) 🆕 Alpha.2
│   ├── TlBatchExecutor (batch processing)
│   ├── TlStreamingExecutor (streaming for large datasets)
│   ├── TlCompilableExecutor (AOT graph compilation)
│   ├── TlJitExecutor (JIT compilation) 🆕
│   ├── TlDistributedExecutor (multi-device) 🆕
│   ├── TlRecoverableExecutor (error recovery) 🆕
│   ├── TlCapabilities (backend queries)
│   └── TlProfiledExecutor (profiling & analysis)
├── Compilation & Optimization
│   ├── GraphCompiler (AOT compilation)
│   ├── CompilationCache (compiled graph caching)
│   ├── JitCompiler (runtime compilation) 🆕
│   ├── JitCache (JIT-specific caching) 🆕
│   ├── HotPathDetector (hot path identification) 🆕
│   ├── AdaptiveOptimizer (adaptive optimization) 🆕
│   ├── GraphOptimizer (fusion, DCE, redundancy)
│   ├── FusionPlanner (operation fusion)
│   ├── Scheduler (execution ordering)
│   └── PlacementOptimizer (device placement)
├── Distributed Execution 🆕
│   ├── DistributedExecutor (multi-device coordinator)
│   ├── DataParallelCoordinator (data parallelism)
│   ├── ModelParallelCoordinator (model parallelism)
│   ├── PipelineParallelCoordinator (pipeline parallelism)
│   └── CommunicationBackend (device communication)
├── Runtime & Memory
│   ├── TensorCache (result caching)
│   ├── MemoryPool (allocation pooling)
│   ├── TensorView (zero-copy views) 🆕 Alpha.2
│   ├── ViewBuilder (ergonomic view API) 🆕 Alpha.2
│   ├── ExecutionStrategy (strategy config)
│   ├── ExecutionContext (state management)
│   ├── AsyncExecutorPool (async load balancing) 🆕 Alpha.2
│   ├── CheckpointManager (checkpointing) 🆕
│   └── StreamProcessor (streaming processing)
├── Analysis & Validation
│   ├── GraphValidator (graph validation)
│   ├── MemoryEstimator (memory estimation)
│   ├── ShapeInferenceContext (shape inference)
│   └── BottleneckAnalyzer (performance analysis)
├── Debugging & Profiling 🆕
│   ├── ExecutionTracer (execution recording)
│   ├── TensorInspector (tensor inspection)
│   ├── BreakpointManager (execution breakpoints)
│   ├── ExecutionRecorder (full history recording)
│   ├── TimelineProfiler (timeline visualization)
│   └── Visualization (DOT, JSON, GraphML export)
├── Enhanced Diagnostics 🆕 Alpha.2
│   ├── Diagnostic (rich error messages)
│   ├── DiagnosticCollector (error aggregation)
│   ├── ShapeMismatchDiagnostic (shape errors)
│   ├── MemoryDiagnostic (memory issues)
│   ├── PerformanceDiagnostic (performance warnings)
│   └── SourceLocation (error tracking)
└── Testing Support 🆕
    ├── DummyExecutor (test executor)
    ├── BackendTestAdapter (backend test templates)
    ├── GradientChecker (numerical gradient checking)
    └── PerfRegression (performance regression testing)
```

## Integration with Other Crates

**tensorlogic-scirs-backend**: Reference implementation using SciRS2
```rust
use tensorlogic_scirs_backend::Scirs2Exec;
let executor = Scirs2Exec::new();
```

**tensorlogic-train**: Training infrastructure
```rust
use tensorlogic_train::{Trainer, TrainerConfig};
let trainer = Trainer::new(executor, config);
```

**tensorlogic-compiler**: Compile TLExpr to EinsumGraph
```rust
use tensorlogic_compiler::compile;
let graph = compile(&expr, &context)?;
let outputs = executor.execute(&graph, &inputs)?;
```

## Performance Considerations

### Optimization Checklist

1. **Enable fusion** for consecutive operations
2. **Use batch execution** for multiple inputs
3. **Enable memory pooling** to reduce allocations
4. **Use streaming** for large datasets that don't fit in memory
5. **Profile execution** to identify bottlenecks
6. **Optimize placement** for multi-device execution
7. **Cache results** for repeated computations

### Benchmarking

```bash
cargo bench -p tensorlogic-infer
```

## Testing

```bash
# Run all tests
cargo test -p tensorlogic-infer

# Run with output
cargo test -p tensorlogic-infer -- --nocapture

# Run specific test
cargo test -p tensorlogic-infer test_streaming
```

**Test Coverage**: 368 tests covering all traits and utilities (100% passing)

### New Beta.1 Modules

The following production-grade modules have been added in Beta.1:

#### Advanced Quantization (`quantization.rs`)
Complete quantization pipeline for model compression:
- INT8, INT4, INT2, FP8, Binary, Ternary quantization types
- QAT and PTQ with multiple calibration strategies
- Per-tensor and per-channel granularity
- Symmetric and asymmetric modes
- Comprehensive compression analysis

#### Dynamic Batching (`dynamic_batching.rs`)
Adaptive request batching for inference serving:
- 4 priority levels (Low/Normal/High/Critical)
- Adaptive batch size optimization
- Request timeout and queueing
- Latency and throughput optimization strategies

#### Advanced Kernel Fusion (`fusion.rs`)
Pattern-based fusion optimization:
- MatMul+Bias, MatMul+Activation, BatchNorm+ReLU patterns
- Vertical and horizontal fusion detection
- Memory bandwidth-aware cost modeling
- Conservative/Aggressive/Balanced/Memory-aware strategies

#### Workspace Management (`workspace.rs`)
Memory pool for efficient allocation reuse:
- BestFit/FirstFit/ExactFit/PowerOfTwo allocation strategies
- Automatic expansion and defragmentation
- Thread-safe shared workspace pools
- Comprehensive efficiency metrics

#### Multi-Model Coordination (`multimodel.rs`)
Ensemble and multi-model management:
- Ensemble strategies: Averaging, Voting, Stacking, Boosting
- Model routing: Priority, Latency, Accuracy, Round-robin, Cascade
- Early-exit cascade support
- Resource tracking and usage statistics

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Apache-2.0

---

**Status**: 🎉 Production Ready (v0.1.0-beta.1)
**Last Updated**: 2025-12-10
**Tests**: 368 passing (100%)
**Code**: 46 files, 19,921 lines
**Completeness**: 100%
**Previous Features**: JIT Compilation, Distributed Execution, Comprehensive Debugging Tools
**Beta.1 Features**: Zero-Copy Tensor Views, Async Execution, Enhanced Diagnostics, Advanced Quantization, Dynamic Batching, Kernel Fusion, Workspace Management, Multi-Model Coordination 🆕
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
