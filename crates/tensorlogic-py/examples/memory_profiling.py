"""
Memory Profiling and Performance Monitoring Demo

Demonstrates the new performance monitoring features in TensorLogic:
- Profiler for tracking execution times
- Timer context manager
- Memory snapshots
- Streaming execution for large datasets
- Cancellation support for async operations
"""

import time
import numpy as np
import pytensorlogic as tl

print("=" * 70)
print("TensorLogic Memory Profiling and Performance Monitoring Demo")
print("=" * 70)


# ============================================================================
# 1. Using the Timer Context Manager
# ============================================================================

print("\n1. Timer Context Manager")
print("-" * 70)

# Create a simple expression
x = tl.var("x")
expr = tl.not_(tl.pred("data", [x]))
graph = tl.compile(expr)

# Use Timer as a context manager
with tl.timer("compilation") as t:
    graph = tl.compile(expr)

print(f"Compilation time: {t.elapsed():.4f} ms")

# Use Timer manually
timer = tl.timer("execution")
timer.start()
result = tl.execute(graph, {"data": np.random.rand(1000)})
elapsed = timer.stop()
print(f"Execution time: {elapsed:.4f} ms")


# ============================================================================
# 2. Using the Profiler
# ============================================================================

print("\n2. Profiler for Multiple Operations")
print("-" * 70)

# Create profiler
prof = tl.profiler()
prof.start()

# Profile multiple operations
for i in range(100):
    # Time compilation
    start = time.perf_counter()
    graph = tl.compile(expr)
    compile_time = (time.perf_counter() - start) * 1000
    prof.record_time("compilation", compile_time)

    # Time execution
    start = time.perf_counter()
    result = tl.execute(graph, {"data": np.random.rand(100)})
    exec_time = (time.perf_counter() - start) * 1000
    prof.record_time("execution", exec_time)

prof.stop()

# Get statistics
compile_stats = prof.get_timing_stats("compilation")
exec_stats = prof.get_timing_stats("execution")

print(f"Compilation Statistics:")
print(f"  Mean: {compile_stats['mean']:.4f} ms")
print(f"  Std:  {compile_stats['std']:.4f} ms")
print(f"  Min:  {compile_stats['min']:.4f} ms")
print(f"  Max:  {compile_stats['max']:.4f} ms")

print(f"\nExecution Statistics:")
print(f"  Mean: {exec_stats['mean']:.4f} ms")
print(f"  Std:  {exec_stats['std']:.4f} ms")

# Print summary
print(f"\nTotal elapsed: {prof.elapsed_ms():.2f} ms")
print(f"Operations profiled: {prof.get_operation_names()}")


# ============================================================================
# 3. Memory Snapshots
# ============================================================================

print("\n3. Memory Snapshots")
print("-" * 70)

# Reset memory tracking
tl.reset_memory_tracking()

# Take initial snapshot
prof2 = tl.profiler()
prof2.start()
prof2.snapshot("initial")

# Create some tensors
graphs = []
for i in range(10):
    expr = tl.not_(tl.pred(f"data_{i}", [tl.var("x")]))
    graphs.append(tl.compile(expr))

prof2.snapshot("after_compilation")

# Execute with large data
results = []
for graph in graphs:
    result = tl.execute(graph, {f"data_{i}": np.random.rand(1000) for i in range(10)})
    results.append(result)

prof2.snapshot("after_execution")

# Print memory snapshots
print("Memory Snapshots:")
for label, snapshot in prof2.get_memory_snapshots():
    print(f"  {label}: {snapshot.current_mb:.2f} MB (peak: {snapshot.peak_mb:.2f} MB)")


# ============================================================================
# 4. Memory Info
# ============================================================================

print("\n4. System Memory Info")
print("-" * 70)

mem_info = tl.get_memory_info()
print(f"Tracked memory: {mem_info['tracked_mb']:.2f} MB")
print(f"Peak memory: {mem_info['peak_mb']:.2f} MB")
print(f"Allocation count: {mem_info['allocation_count']}")


# ============================================================================
# 5. Streaming Execution for Large Datasets
# ============================================================================

print("\n5. Streaming Execution")
print("-" * 70)

# Create graph
x = tl.var("x")
expr = tl.not_(tl.pred("data", [x]))
graph = tl.compile(expr)

# Create streaming executor
streamer = tl.streaming_executor(graph, chunk_size=500, overlap=0)
print(f"Created: {streamer}")

# Create large dataset
large_data = np.random.rand(2000, 10)
inputs = {"data": large_data}

# Execute in streaming fashion
print(f"Processing {large_data.shape[0]} samples in chunks of {streamer.chunk_size}...")

with tl.timer("streaming_execution") as t:
    chunk_results = streamer.execute_streaming(inputs, output_key="data")

print(f"Streaming execution time: {t.elapsed():.2f} ms")
print(f"Number of chunks: {len(chunk_results)}")

# Compare with regular execution
print("\nComparing with regular execution...")
with tl.timer("regular_execution") as t:
    regular_result = tl.execute(graph, inputs)

print(f"Regular execution time: {t.elapsed():.2f} ms")


# ============================================================================
# 6. Result Accumulator
# ============================================================================

print("\n6. Result Accumulator")
print("-" * 70)

# Create accumulator
accumulator = tl.result_accumulator()

# Accumulate results from streaming
for chunk in chunk_results:
    accumulator.add(chunk)

stats = accumulator.stats()
print(f"Accumulated {accumulator.count()} chunks")
print(f"Total elements: {accumulator.total_elements()}")
print(f"Mean chunk size: {stats.get('mean_chunk_size', 0):.1f}")

# Combine results
combined = accumulator.combine()
print(f"Combined result shape: {combined.shape}")


# ============================================================================
# 7. Async Cancellation Support
# ============================================================================

print("\n7. Async Cancellation Support")
print("-" * 70)

# Create a computation
x = tl.var("x")
expr = tl.not_(tl.pred("data", [x]))
graph = tl.compile(expr)
large_input = {"data": np.random.rand(10000)}

# Start async execution
future = tl.execute_async(graph, large_input)
print(f"Started async execution: {future}")

# Check cancellation token
token = future.get_cancellation_token()
if token:
    print(f"Cancellation token: {token}")
    print(f"Is cancelled: {token.is_cancelled()}")

# Wait for result
result = future.result()
print(f"Result shape: {result['output'].shape}")

# Demonstrate cancellation (won't actually cancel since computation is fast)
future2 = tl.execute_async(graph, large_input)
future2.cancel()
print(f"\nCancelled future: {future2}")
print(f"Is cancelled: {future2.is_cancelled()}")


# ============================================================================
# 8. Profiler Summary
# ============================================================================

print("\n8. Complete Profiler Summary")
print("-" * 70)

# Create comprehensive profile
prof_complete = tl.profiler()
prof_complete.start()

# Warm up
for _ in range(5):
    tl.execute(graph, {"data": np.random.rand(100)})

prof_complete.snapshot("warmup_complete")

# Compilation benchmark
for _ in range(50):
    start = time.perf_counter()
    g = tl.compile(expr)
    prof_complete.record_time("compile", (time.perf_counter() - start) * 1000)

prof_complete.snapshot("compilation_complete")

# Small tensor execution
for _ in range(100):
    start = time.perf_counter()
    tl.execute(graph, {"data": np.random.rand(10)})
    prof_complete.record_time("exec_small", (time.perf_counter() - start) * 1000)

# Medium tensor execution
for _ in range(50):
    start = time.perf_counter()
    tl.execute(graph, {"data": np.random.rand(100)})
    prof_complete.record_time("exec_medium", (time.perf_counter() - start) * 1000)

# Large tensor execution
for _ in range(20):
    start = time.perf_counter()
    tl.execute(graph, {"data": np.random.rand(1000)})
    prof_complete.record_time("exec_large", (time.perf_counter() - start) * 1000)

prof_complete.snapshot("execution_complete")
prof_complete.stop()

# Print full summary
print(prof_complete.summary())


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary: Performance Monitoring Features")
print("=" * 70)
print("""
Available Features:

Performance Monitoring:
  • timer(name) - Create a timer (context manager support)
  • profiler() - Create a profiler for multiple operations
  • memory_snapshot() - Get current memory snapshot
  • get_memory_info() - Get detailed memory statistics
  • reset_memory_tracking() - Reset memory counters

Streaming Execution:
  • streaming_executor(graph, chunk_size) - Create streaming executor
  • result_accumulator() - Accumulate streaming results
  • process_stream() - Process iterator through graph

Async Cancellation:
  • cancellation_token() - Create cancellation token
  • AsyncResult.cancel() - Cancel async operation
  • AsyncResult.is_cancelled() - Check cancellation status

Use Cases:
  • Performance benchmarking and optimization
  • Memory usage monitoring in production
  • Processing large datasets that don't fit in memory
  • Cancelling long-running async computations
  • Building data pipelines with streaming
""")
print("=" * 70)
