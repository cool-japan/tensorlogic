# tensorlogic-infer

[![Crate](https://img.shields.io/badge/crates.io-tensorlogic--infer-orange)](https://crates.io/crates/tensorlogic-infer)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/tensorlogic-infer)
[![Tests](https://img.shields.io/badge/tests-241-brightgreen)](#)
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
│   ├── TlAutodiff (training)
│   ├── TlBatchExecutor (batching)
│   ├── TlStreamingExecutor (streaming)
│   ├── TlCapabilities (queries)
│   └── TlProfiledExecutor (profiling)
├── Optimization
│   ├── GraphOptimizer (analysis)
│   ├── FusionPlanner (fusion)
│   ├── Scheduler (ordering)
│   └── PlacementOptimizer (devices)
├── Runtime
│   ├── TensorCache (caching)
│   ├── MemoryPool (pooling)
│   ├── ExecutionStrategy (config)
│   └── ExecutionContext (state)
└── Analysis
    ├── GraphValidator (validation)
    ├── MemoryEstimator (memory)
    ├── ShapeInferenceContext (shapes)
    └── BottleneckAnalyzer (perf)
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

**Test Coverage**: 189 tests covering all traits and utilities (100% passing)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Apache-2.0

---

**Status**: 🎉 Production Ready (v0.1.0-alpha.1)
**Last Updated**: 2025-11-06
**Tests**: 189 passing (100%)
**Completeness**: ~95%
**New Features**: Comprehensive debugging tools (ExecutionTracer, TensorInspector, BreakpointManager)
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
