"""
TensorLogic CLI - Python FFI Wrapper

This module provides Python bindings for the TensorLogic CLI library using ctypes.
It allows you to compile, optimize, and execute logical expressions from Python.

Example:
    >>> from tensorlogic_ffi import TensorLogic
    >>> tl = TensorLogic()
    >>> result = tl.compile("friend(alice, bob)")
    >>> if result.is_success():
    ...     print(f"Compiled graph with {result.tensor_count} tensors")
    ... else:
    ...     print(f"Error: {result.error_message}")

Requirements:
    - The TensorLogic CLI library must be built and accessible
    - Set TENSORLOGIC_LIB_PATH environment variable to library location, or
    - Place the library in a standard location (LD_LIBRARY_PATH on Linux, etc.)

Build the library with:
    cargo build --release -p tensorlogic-cli

The shared library will be located at:
    - Linux: target/release/libtensorlogic_cli.so
    - macOS: target/release/libtensorlogic_cli.dylib
    - Windows: target/release/tensorlogic_cli.dll
"""

import ctypes
import os
import platform
import sys
from typing import Optional
from pathlib import Path


class GraphResult:
    """Result of compiling a logical expression to a tensor graph"""

    def __init__(self, ptr):
        self._ptr = ptr
        self._result = ptr.contents if ptr else None

    def is_success(self) -> bool:
        """Check if compilation succeeded"""
        return self._result and not self._result.error_message

    @property
    def graph_data(self) -> Optional[str]:
        """Get the compiled graph as JSON (None on error)"""
        if not self._result or not self._result.graph_data:
            return None
        return self._result.graph_data.decode('utf-8')

    @property
    def error_message(self) -> Optional[str]:
        """Get error message (None on success)"""
        if not self._result or not self._result.error_message:
            return None
        return self._result.error_message.decode('utf-8')

    @property
    def tensor_count(self) -> int:
        """Get number of tensors in the graph"""
        return self._result.tensor_count if self._result else 0

    @property
    def node_count(self) -> int:
        """Get number of nodes in the graph"""
        return self._result.node_count if self._result else 0

    def __str__(self) -> str:
        if self.is_success():
            return f"GraphResult(tensors={self.tensor_count}, nodes={self.node_count})"
        else:
            return f"GraphResult(error='{self.error_message}')"


class ExecutionResult:
    """Result of executing a tensor graph"""

    def __init__(self, ptr):
        self._ptr = ptr
        self._result = ptr.contents if ptr else None

    def is_success(self) -> bool:
        """Check if execution succeeded"""
        return self._result and not self._result.error_message

    @property
    def output_data(self) -> Optional[str]:
        """Get the execution output as JSON (None on error)"""
        if not self._result or not self._result.output_data:
            return None
        return self._result.output_data.decode('utf-8')

    @property
    def error_message(self) -> Optional[str]:
        """Get error message (None on success)"""
        if not self._result or not self._result.error_message:
            return None
        return self._result.error_message.decode('utf-8')

    @property
    def execution_time_us(self) -> int:
        """Get execution time in microseconds"""
        return self._result.execution_time_us if self._result else 0

    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds"""
        return self.execution_time_us / 1000.0

    def __str__(self) -> str:
        if self.is_success():
            return f"ExecutionResult(time={self.execution_time_ms:.3f}ms)"
        else:
            return f"ExecutionResult(error='{self.error_message}')"


class OptimizationResult:
    """Result of optimizing a tensor graph"""

    def __init__(self, ptr):
        self._ptr = ptr
        self._result = ptr.contents if ptr else None

    def is_success(self) -> bool:
        """Check if optimization succeeded"""
        return self._result and not self._result.error_message

    @property
    def graph_data(self) -> Optional[str]:
        """Get the optimized graph as JSON (None on error)"""
        if not self._result or not self._result.graph_data:
            return None
        return self._result.graph_data.decode('utf-8')

    @property
    def error_message(self) -> Optional[str]:
        """Get error message (None on success)"""
        if not self._result or not self._result.error_message:
            return None
        return self._result.error_message.decode('utf-8')

    @property
    def tensors_removed(self) -> int:
        """Get number of tensors removed by optimization"""
        return self._result.tensors_removed if self._result else 0

    @property
    def nodes_removed(self) -> int:
        """Get number of nodes removed by optimization"""
        return self._result.nodes_removed if self._result else 0

    def __str__(self) -> str:
        if self.is_success():
            return f"OptimizationResult(tensors_removed={self.tensors_removed}, nodes_removed={self.nodes_removed})"
        else:
            return f"OptimizationResult(error='{self.error_message}')"


class BenchmarkResult:
    """Result of benchmarking compilation"""

    def __init__(self, ptr):
        self._ptr = ptr
        self._result = ptr.contents if ptr else None

    def is_success(self) -> bool:
        """Check if benchmark succeeded"""
        return self._result and not self._result.error_message

    @property
    def error_message(self) -> Optional[str]:
        """Get error message (None on success)"""
        if not self._result or not self._result.error_message:
            return None
        return self._result.error_message.decode('utf-8')

    @property
    def mean_us(self) -> float:
        """Get mean execution time in microseconds"""
        return self._result.mean_us if self._result else 0.0

    @property
    def mean_ms(self) -> float:
        """Get mean execution time in milliseconds"""
        return self.mean_us / 1000.0

    @property
    def std_dev_us(self) -> float:
        """Get standard deviation in microseconds"""
        return self._result.std_dev_us if self._result else 0.0

    @property
    def min_us(self) -> int:
        """Get minimum execution time in microseconds"""
        return self._result.min_us if self._result else 0

    @property
    def max_us(self) -> int:
        """Get maximum execution time in microseconds"""
        return self._result.max_us if self._result else 0

    @property
    def iterations(self) -> int:
        """Get number of iterations"""
        return self._result.iterations if self._result else 0

    def __str__(self) -> str:
        if self.is_success():
            return f"BenchmarkResult(mean={self.mean_ms:.3f}ms±{self.std_dev_us/1000:.3f}ms, n={self.iterations})"
        else:
            return f"BenchmarkResult(error='{self.error_message}')"


# C structures
class _TLGraphResult(ctypes.Structure):
    _fields_ = [
        ("graph_data", ctypes.c_char_p),
        ("error_message", ctypes.c_char_p),
        ("tensor_count", ctypes.c_size_t),
        ("node_count", ctypes.c_size_t),
    ]


class _TLExecutionResult(ctypes.Structure):
    _fields_ = [
        ("output_data", ctypes.c_char_p),
        ("error_message", ctypes.c_char_p),
        ("execution_time_us", ctypes.c_uint64),
    ]


class _TLOptimizationResult(ctypes.Structure):
    _fields_ = [
        ("graph_data", ctypes.c_char_p),
        ("error_message", ctypes.c_char_p),
        ("tensors_removed", ctypes.c_size_t),
        ("nodes_removed", ctypes.c_size_t),
    ]


class _TLBenchmarkResult(ctypes.Structure):
    _fields_ = [
        ("error_message", ctypes.c_char_p),
        ("mean_us", ctypes.c_double),
        ("std_dev_us", ctypes.c_double),
        ("min_us", ctypes.c_uint64),
        ("max_us", ctypes.c_uint64),
        ("iterations", ctypes.c_size_t),
    ]


def _find_library() -> Path:
    """Find the TensorLogic CLI library"""
    # Check environment variable first
    if "TENSORLOGIC_LIB_PATH" in os.environ:
        lib_path = Path(os.environ["TENSORLOGIC_LIB_PATH"])
        if lib_path.exists():
            return lib_path

    # Determine library name based on platform
    system = platform.system()
    if system == "Linux":
        lib_name = "libtensorlogic_cli.so"
    elif system == "Darwin":
        lib_name = "libtensorlogic_cli.dylib"
    elif system == "Windows":
        lib_name = "tensorlogic_cli.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    # Search common locations
    search_paths = [
        # Relative to this file (development)
        Path(__file__).parent.parent.parent.parent / "target" / "release" / lib_name,
        # Standard installation paths
        Path(f"/usr/local/lib/{lib_name}"),
        Path(f"/usr/lib/{lib_name}"),
        # Current directory
        Path(lib_name),
    ]

    for path in search_paths:
        if path.exists():
            return path

    raise RuntimeError(
        f"Could not find TensorLogic CLI library ({lib_name}). "
        "Set TENSORLOGIC_LIB_PATH environment variable or build with: "
        "cargo build --release -p tensorlogic-cli"
    )


class TensorLogic:
    """
    Python wrapper for TensorLogic CLI library.

    This class provides a high-level Python interface to the TensorLogic CLI
    library, allowing you to compile, optimize, and execute logical expressions.

    Example:
        >>> tl = TensorLogic()
        >>> result = tl.compile("AND(pred1(x), pred2(x, y))")
        >>> if result.is_success():
        ...     print(f"Graph: {result.graph_data}")
        >>> exec_result = tl.execute(result.graph_data, "cpu")
        >>> print(f"Output: {exec_result.output_data}")
    """

    def __init__(self, lib_path: Optional[Path] = None):
        """
        Initialize the TensorLogic wrapper.

        Args:
            lib_path: Optional path to the library. If None, will search standard locations.
        """
        if lib_path is None:
            lib_path = _find_library()

        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_functions()

    def _setup_functions(self):
        """Setup function signatures for the FFI functions"""
        # tl_compile_expr
        self._lib.tl_compile_expr.argtypes = [ctypes.c_char_p]
        self._lib.tl_compile_expr.restype = ctypes.POINTER(_TLGraphResult)

        # tl_execute_graph
        self._lib.tl_execute_graph.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.tl_execute_graph.restype = ctypes.POINTER(_TLExecutionResult)

        # tl_optimize_graph
        self._lib.tl_optimize_graph.argtypes = [ctypes.c_char_p, ctypes.c_int32]
        self._lib.tl_optimize_graph.restype = ctypes.POINTER(_TLOptimizationResult)

        # tl_benchmark_compilation
        self._lib.tl_benchmark_compilation.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self._lib.tl_benchmark_compilation.restype = ctypes.POINTER(_TLBenchmarkResult)

        # Memory management functions
        self._lib.tl_free_string.argtypes = [ctypes.c_char_p]
        self._lib.tl_free_string.restype = None

        self._lib.tl_free_graph_result.argtypes = [ctypes.POINTER(_TLGraphResult)]
        self._lib.tl_free_graph_result.restype = None

        self._lib.tl_free_execution_result.argtypes = [ctypes.POINTER(_TLExecutionResult)]
        self._lib.tl_free_execution_result.restype = None

        self._lib.tl_free_optimization_result.argtypes = [ctypes.POINTER(_TLOptimizationResult)]
        self._lib.tl_free_optimization_result.restype = None

        self._lib.tl_free_benchmark_result.argtypes = [ctypes.POINTER(_TLBenchmarkResult)]
        self._lib.tl_free_benchmark_result.restype = None

        # Utility functions
        self._lib.tl_version.argtypes = []
        self._lib.tl_version.restype = ctypes.c_char_p

        self._lib.tl_is_backend_available.argtypes = [ctypes.c_char_p]
        self._lib.tl_is_backend_available.restype = ctypes.c_int32

    def compile(self, expr: str) -> GraphResult:
        """
        Compile a logical expression to a tensor graph.

        Args:
            expr: The logical expression to compile

        Returns:
            GraphResult containing the compiled graph or error

        Example:
            >>> result = tl.compile("AND(pred1(x), pred2(x, y))")
            >>> if result.is_success():
            ...     print(f"Compiled successfully with {result.tensor_count} tensors")
        """
        ptr = self._lib.tl_compile_expr(expr.encode('utf-8'))
        result = GraphResult(ptr)
        # Note: Caller must free the result manually with _lib.tl_free_graph_result(ptr)
        # For automatic cleanup, consider using a context manager
        return result

    def execute(self, graph_json: str, backend: str = "cpu") -> ExecutionResult:
        """
        Execute a compiled graph.

        Args:
            graph_json: The graph as JSON string (from GraphResult.graph_data)
            backend: The backend to use ("cpu", "parallel", "profiled")

        Returns:
            ExecutionResult containing the output or error

        Example:
            >>> compile_result = tl.compile("pred(x, y)")
            >>> exec_result = tl.execute(compile_result.graph_data)
            >>> print(f"Execution time: {exec_result.execution_time_ms:.3f}ms")
        """
        ptr = self._lib.tl_execute_graph(
            graph_json.encode('utf-8'),
            backend.encode('utf-8')
        )
        result = ExecutionResult(ptr)
        return result

    def optimize(self, graph_json: str, level: int = 2) -> OptimizationResult:
        """
        Optimize a compiled graph.

        Args:
            graph_json: The graph as JSON string
            level: Optimization level (0=none, 1=basic, 2=standard, 3=aggressive)

        Returns:
            OptimizationResult containing the optimized graph or error

        Example:
            >>> compile_result = tl.compile("AND(a, b)")
            >>> opt_result = tl.optimize(compile_result.graph_data, level=2)
            >>> print(f"Removed {opt_result.nodes_removed} nodes")
        """
        ptr = self._lib.tl_optimize_graph(graph_json.encode('utf-8'), level)
        result = OptimizationResult(ptr)
        return result

    def benchmark(self, expr: str, iterations: int = 100) -> BenchmarkResult:
        """
        Benchmark compilation of an expression.

        Args:
            expr: The logical expression to benchmark
            iterations: Number of iterations to run

        Returns:
            BenchmarkResult containing timing statistics or error

        Example:
            >>> result = tl.benchmark("pred(x, y)", iterations=1000)
            >>> print(f"Mean: {result.mean_ms:.3f}ms ± {result.std_dev_us/1000:.3f}ms")
        """
        ptr = self._lib.tl_benchmark_compilation(expr.encode('utf-8'), iterations)
        result = BenchmarkResult(ptr)
        return result

    def version(self) -> str:
        """
        Get the TensorLogic CLI version.

        Returns:
            Version string

        Example:
            >>> print(f"TensorLogic version: {tl.version()}")
        """
        version_ptr = self._lib.tl_version()
        version = version_ptr.decode('utf-8')
        self._lib.tl_free_string(version_ptr)
        return version

    def is_backend_available(self, backend: str) -> bool:
        """
        Check if a backend is available.

        Args:
            backend: The backend name ("cpu", "parallel", "simd", "gpu")

        Returns:
            True if available, False otherwise

        Example:
            >>> if tl.is_backend_available("cpu"):
            ...     print("CPU backend is available")
        """
        return self._lib.tl_is_backend_available(backend.encode('utf-8')) == 1


if __name__ == "__main__":
    # Example usage
    tl = TensorLogic()

    print(f"TensorLogic version: {tl.version()}")
    print(f"CPU backend available: {tl.is_backend_available('cpu')}")
    print()

    # Compile an expression
    print("Compiling expression: friend(alice, bob)")
    result = tl.compile("friend(alice, bob)")
    if result.is_success():
        print(f"✓ Compiled successfully!")
        print(f"  Tensors: {result.tensor_count}, Nodes: {result.node_count}")
    else:
        print(f"✗ Compilation failed: {result.error_message}")

    # Benchmark
    print("\nBenchmarking compilation (100 iterations)...")
    bench = tl.benchmark("AND(pred1(x), pred2(x, y))", iterations=100)
    if bench.is_success():
        print(f"✓ Benchmark complete!")
        print(f"  Mean: {bench.mean_ms:.3f}ms ± {bench.std_dev_us/1000:.3f}ms")
        print(f"  Range: {bench.min_us/1000:.3f}ms - {bench.max_us/1000:.3f}ms")
    else:
        print(f"✗ Benchmark failed: {bench.error_message}")
