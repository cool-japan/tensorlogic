# TensorLogic CLI - FFI Bindings

This document describes the FFI (Foreign Function Interface) bindings for the TensorLogic CLI library, enabling integration with C/C++ and Python projects.

## Overview

The FFI bindings provide a C-compatible interface to the TensorLogic CLI library, allowing you to:
- Compile logical expressions to tensor graphs
- Execute compiled graphs with multiple backends
- Optimize graphs for better performance
- Benchmark compilation performance
- Query backend availability and version information

## Building the Shared Library

To build the shared library:

```bash
cargo build --release -p tensorlogic-cli
```

The shared library will be located at:
- **Linux**: `target/release/libtensorlogic_cli.so`
- **macOS**: `target/release/libtensorlogic_cli.dylib`
- **Windows**: `target/release/tensorlogic_cli.dll`

## C/C++ Integration

### Header File

The C header file is located at `tensorlogic.h`. Include it in your C/C++ project:

```c
#include "tensorlogic.h"
```

### Basic Example (C)

```c
#include <stdio.h>
#include "tensorlogic.h"

int main() {
    // Get version
    char* version = tl_version();
    printf("TensorLogic version: %s\n", version);
    tl_free_string(version);

    // Check backend availability
    if (tl_is_backend_available("cpu")) {
        printf("CPU backend is available\n");
    }

    // Compile an expression
    TLGraphResult* result = tl_compile_expr("AND(pred1(x), pred2(x, y))");
    if (result->error_message != NULL) {
        fprintf(stderr, "Error: %s\n", result->error_message);
        tl_free_graph_result(result);
        return 1;
    }

    printf("Compiled graph with %zu tensors and %zu nodes\n",
           result->tensor_count, result->node_count);

    // Execute the graph
    TLExecutionResult* exec = tl_execute_graph(result->graph_data, "cpu");
    if (exec->error_message != NULL) {
        fprintf(stderr, "Execution error: %s\n", exec->error_message);
    } else {
        printf("Execution time: %lu microseconds\n", exec->execution_time_us);
        printf("Output: %s\n", exec->output_data);
    }

    // Clean up
    tl_free_execution_result(exec);
    tl_free_graph_result(result);

    return 0;
}
```

### Compilation

Link against the TensorLogic library:

**Linux/macOS**:
```bash
gcc -o myapp myapp.c -L./target/release -ltensorlogic_cli -Wl,-rpath,./target/release
```

**Windows**:
```bash
cl myapp.c tensorlogic_cli.lib
```

## Python Integration

### Using the Python Wrapper

The Python wrapper is located at `python/tensorlogic_ffi.py`. It provides a high-level Pythonic interface:

```python
from tensorlogic_ffi import TensorLogic

# Initialize
tl = TensorLogic()

# Get version
print(f"Version: {tl.version()}")

# Check backend availability
if tl.is_backend_available("cpu"):
    print("CPU backend is available")

# Compile an expression
result = tl.compile("AND(pred1(x), pred2(x, y))")
if result.is_success():
    print(f"Compiled graph with {result.tensor_count} tensors")
    print(f"Graph JSON: {result.graph_data}")
else:
    print(f"Error: {result.error_message}")

# Execute the graph
if result.is_success():
    exec_result = tl.execute(result.graph_data, backend="cpu")
    if exec_result.is_success():
        print(f"Execution time: {exec_result.execution_time_ms:.3f}ms")
        print(f"Output: {exec_result.output_data}")

# Optimize a graph
if result.is_success():
    opt_result = tl.optimize(result.graph_data, level=2)
    if opt_result.is_success():
        print(f"Removed {opt_result.nodes_removed} nodes")

# Benchmark compilation
bench = tl.benchmark("AND(pred1(x), pred2(x, y))", iterations=100)
if bench.is_success():
    print(f"Mean: {bench.mean_ms:.3f}ms ± {bench.std_dev_us/1000:.3f}ms")
```

### Environment Setup

Set the `TENSORLOGIC_LIB_PATH` environment variable to point to the shared library:

```bash
export TENSORLOGIC_LIB_PATH=/path/to/tensorlogic/target/release/libtensorlogic_cli.so
python your_script.py
```

Or place the library in a standard location (e.g., `/usr/local/lib`).

## API Reference

### Compilation

- **`tl_compile_expr(expr)`**: Compile a logical expression to a tensor graph
  - Returns: `TLGraphResult*` (must be freed with `tl_free_graph_result`)

### Execution

- **`tl_execute_graph(graph_json, backend)`**: Execute a compiled graph
  - Parameters:
    - `graph_json`: Graph as JSON string
    - `backend`: Backend name ("cpu", "parallel", "profiled")
  - Returns: `TLExecutionResult*` (must be freed with `tl_free_execution_result`)

### Optimization

- **`tl_optimize_graph(graph_json, level)`**: Optimize a graph
  - Parameters:
    - `graph_json`: Graph as JSON string
    - `level`: Optimization level (0=none, 1=basic, 2=standard, 3=aggressive)
  - Returns: `TLOptimizationResult*` (must be freed with `tl_free_optimization_result`)

### Benchmarking

- **`tl_benchmark_compilation(expr, iterations)`**: Benchmark compilation
  - Parameters:
    - `expr`: Expression to benchmark
    - `iterations`: Number of iterations
  - Returns: `TLBenchmarkResult*` (must be freed with `tl_free_benchmark_result`)

### Utilities

- **`tl_version()`**: Get version string (must be freed with `tl_free_string`)
- **`tl_is_backend_available(backend)`**: Check if backend is available (returns 1 if available, 0 if not)

### Memory Management

All strings and result structures returned by the library must be freed:

- **`tl_free_string(str)`**: Free a string
- **`tl_free_graph_result(result)`**: Free a graph result
- **`tl_free_execution_result(result)`**: Free an execution result
- **`tl_free_optimization_result(result)`**: Free an optimization result
- **`tl_free_benchmark_result(result)`**: Free a benchmark result

All free functions are NULL-safe and will automatically free any strings contained in result structures.

## Error Handling

All operations return result structures with two key fields:
- **`error_message`**: NULL on success, error description on failure
- **`*_data`**: NULL on error, valid data on success

Always check `error_message` first before accessing data fields.

### Example Error Handling (C)

```c
TLGraphResult* result = tl_compile_expr("invalid expression");
if (result->error_message != NULL) {
    fprintf(stderr, "Compilation failed: %s\n", result->error_message);
    tl_free_graph_result(result);
    return 1;
}

// Safe to use result->graph_data here
printf("Graph: %s\n", result->graph_data);
tl_free_graph_result(result);
```

### Example Error Handling (Python)

```python
result = tl.compile("invalid expression")
if not result.is_success():
    print(f"Compilation failed: {result.error_message}")
    return

# Safe to use result.graph_data here
print(f"Graph: {result.graph_data}")
```

## Thread Safety

The FFI functions are thread-safe for independent operations. However, sharing result structures between threads requires external synchronization.

## Performance Considerations

- Minimize FFI boundary crossings by batching operations
- Reuse compiled graphs when possible
- Use the profiled backend for performance analysis
- Optimization level 2 (standard) provides good balance between compile time and runtime performance

## Examples

Complete examples can be found in:
- C examples: Create a C file using the examples shown above
- Python examples: See `python/tensorlogic_ffi.py` (run as `python tensorlogic_ffi.py` for demo)

## Testing

The FFI bindings include comprehensive tests:

```bash
cargo test -p tensorlogic-cli ffi
```

All 5 FFI tests pass with zero warnings.

## Troubleshooting

### Library Not Found

**Linux**:
```bash
export LD_LIBRARY_PATH=/path/to/tensorlogic/target/release:$LD_LIBRARY_PATH
```

**macOS**:
```bash
export DYLD_LIBRARY_PATH=/path/to/tensorlogic/target/release:$DYLD_LIBRARY_PATH
```

**Windows**:
Add the directory containing `tensorlogic_cli.dll` to your PATH.

### Python Import Errors

Make sure the shared library is built:
```bash
cargo build --release -p tensorlogic-cli
```

Set `TENSORLOGIC_LIB_PATH` or place the library in a standard location.

### Memory Leaks

Always call the appropriate `tl_free_*` function for every result structure returned by the library.

## Support

For issues and questions:
- GitHub Issues: https://github.com/cool-japan/tensorlogic/issues
- Documentation: See main README.md and TODO.md

## License

Same as TensorLogic CLI (see main repository LICENSE).
