/**
 * TensorLogic CLI - C FFI Header
 *
 * This header provides C-compatible bindings for the TensorLogic CLI library,
 * enabling integration with C/C++ projects and other languages via FFI.
 *
 * # Memory Management
 *
 * - All functions that return owned strings use allocated C strings that must be freed with tl_free_string()
 * - All result structures must be freed with their respective free functions
 * - Error messages are allocated and must be freed with tl_free_string()
 *
 * # Example Usage (C)
 *
 * ```c
 * // Compile an expression
 * TLGraphResult* result = tl_compile_expr("friend(alice, bob)");
 * if (result->error_message != NULL) {
 *     fprintf(stderr, "Error: %s\n", result->error_message);
 *     tl_free_graph_result(result);
 *     return 1;
 * }
 *
 * printf("Graph: %s\n", result->graph_data);
 * printf("Tensors: %zu, Nodes: %zu\n", result->tensor_count, result->node_count);
 * tl_free_graph_result(result);
 * ```
 *
 * # Linking
 *
 * To link against the TensorLogic library:
 * - Linux: -ltensorlogic_cli
 * - macOS: -ltensorlogic_cli
 * - Windows: tensorlogic_cli.lib
 *
 * # Build Requirements
 *
 * Build the shared library with:
 * ```sh
 * cargo build --release -p tensorlogic-cli
 * ```
 *
 * The shared library will be located at:
 * - Linux: target/release/libtensorlogic_cli.so
 * - macOS: target/release/libtensorlogic_cli.dylib
 * - Windows: target/release/tensorlogic_cli.dll
 */

#ifndef TENSORLOGIC_H
#define TENSORLOGIC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * FFI-compatible result type for graph compilation
 */
typedef struct {
    /** The compiled graph as JSON string (NULL on error) */
    char* graph_data;
    /** Error message (NULL on success) */
    char* error_message;
    /** Number of tensors in the graph (0 on error) */
    size_t tensor_count;
    /** Number of nodes in the graph (0 on error) */
    size_t node_count;
} TLGraphResult;

/**
 * FFI-compatible result type for execution
 */
typedef struct {
    /** The execution output as JSON string (NULL on error) */
    char* output_data;
    /** Error message (NULL on success) */
    char* error_message;
    /** Execution time in microseconds */
    uint64_t execution_time_us;
} TLExecutionResult;

/**
 * FFI-compatible result type for optimization
 */
typedef struct {
    /** The optimized graph as JSON string (NULL on error) */
    char* graph_data;
    /** Error message (NULL on success) */
    char* error_message;
    /** Number of tensors removed */
    size_t tensors_removed;
    /** Number of nodes removed */
    size_t nodes_removed;
} TLOptimizationResult;

/**
 * FFI-compatible benchmark results
 */
typedef struct {
    /** Error message (NULL on success) */
    char* error_message;
    /** Mean execution time in microseconds */
    double mean_us;
    /** Standard deviation in microseconds */
    double std_dev_us;
    /** Minimum execution time in microseconds */
    uint64_t min_us;
    /** Maximum execution time in microseconds */
    uint64_t max_us;
    /** Number of iterations */
    size_t iterations;
} TLBenchmarkResult;

/**
 * Compile a logical expression to a tensor graph
 *
 * @param expr The logical expression as a C string
 * @return A pointer to TLGraphResult that must be freed with tl_free_graph_result()
 *
 * Example:
 * ```c
 * TLGraphResult* result = tl_compile_expr("AND(pred1(x), pred2(x, y))");
 * if (result->error_message != NULL) {
 *     fprintf(stderr, "Error: %s\n", result->error_message);
 *     tl_free_graph_result(result);
 *     return 1;
 * }
 * printf("Compiled graph with %zu tensors and %zu nodes\n",
 *        result->tensor_count, result->node_count);
 * tl_free_graph_result(result);
 * ```
 */
TLGraphResult* tl_compile_expr(const char* expr);

/**
 * Execute a compiled graph
 *
 * @param graph_json The graph as JSON string
 * @param backend The backend name (e.g., "cpu", "parallel", "profiled")
 * @return A pointer to TLExecutionResult that must be freed with tl_free_execution_result()
 *
 * Example:
 * ```c
 * TLGraphResult* compile_result = tl_compile_expr("pred(x, y)");
 * if (compile_result->error_message != NULL) {
 *     // Handle error...
 * }
 *
 * TLExecutionResult* exec_result = tl_execute_graph(compile_result->graph_data, "cpu");
 * if (exec_result->error_message != NULL) {
 *     fprintf(stderr, "Execution error: %s\n", exec_result->error_message);
 * } else {
 *     printf("Execution time: %lu us\n", exec_result->execution_time_us);
 *     printf("Output: %s\n", exec_result->output_data);
 * }
 *
 * tl_free_execution_result(exec_result);
 * tl_free_graph_result(compile_result);
 * ```
 */
TLExecutionResult* tl_execute_graph(const char* graph_json, const char* backend);

/**
 * Optimize a compiled graph
 *
 * @param graph_json The graph as JSON string
 * @param level Optimization level (0=none, 1=basic, 2=standard, 3=aggressive)
 * @return A pointer to TLOptimizationResult that must be freed with tl_free_optimization_result()
 *
 * Example:
 * ```c
 * TLGraphResult* compile_result = tl_compile_expr("AND(a, b)");
 * if (compile_result->error_message != NULL) {
 *     // Handle error...
 * }
 *
 * TLOptimizationResult* opt_result = tl_optimize_graph(compile_result->graph_data, 2);
 * if (opt_result->error_message != NULL) {
 *     fprintf(stderr, "Optimization error: %s\n", opt_result->error_message);
 * } else {
 *     printf("Removed %zu tensors and %zu nodes\n",
 *            opt_result->tensors_removed, opt_result->nodes_removed);
 * }
 *
 * tl_free_optimization_result(opt_result);
 * tl_free_graph_result(compile_result);
 * ```
 */
TLOptimizationResult* tl_optimize_graph(const char* graph_json, int32_t level);

/**
 * Benchmark compilation of an expression
 *
 * @param expr The logical expression as a C string
 * @param iterations Number of iterations to run
 * @return A pointer to TLBenchmarkResult that must be freed with tl_free_benchmark_result()
 *
 * Example:
 * ```c
 * TLBenchmarkResult* bench = tl_benchmark_compilation("pred(x, y)", 100);
 * if (bench->error_message != NULL) {
 *     fprintf(stderr, "Benchmark error: %s\n", bench->error_message);
 * } else {
 *     printf("Mean: %.2f us, StdDev: %.2f us\n", bench->mean_us, bench->std_dev_us);
 *     printf("Min: %lu us, Max: %lu us\n", bench->min_us, bench->max_us);
 * }
 * tl_free_benchmark_result(bench);
 * ```
 */
TLBenchmarkResult* tl_benchmark_compilation(const char* expr, size_t iterations);

/**
 * Free a string allocated by tensorlogic
 *
 * @param s The string to free
 *
 * Note: This function is NULL-safe and can be called with NULL pointers.
 */
void tl_free_string(char* s);

/**
 * Free a graph result
 *
 * @param result The result to free
 *
 * Note: This function is NULL-safe and can be called with NULL pointers.
 * It will also free any strings contained in the result structure.
 */
void tl_free_graph_result(TLGraphResult* result);

/**
 * Free an execution result
 *
 * @param result The result to free
 *
 * Note: This function is NULL-safe and can be called with NULL pointers.
 * It will also free any strings contained in the result structure.
 */
void tl_free_execution_result(TLExecutionResult* result);

/**
 * Free an optimization result
 *
 * @param result The result to free
 *
 * Note: This function is NULL-safe and can be called with NULL pointers.
 * It will also free any strings contained in the result structure.
 */
void tl_free_optimization_result(TLOptimizationResult* result);

/**
 * Free a benchmark result
 *
 * @param result The result to free
 *
 * Note: This function is NULL-safe and can be called with NULL pointers.
 * It will also free any strings contained in the result structure.
 */
void tl_free_benchmark_result(TLBenchmarkResult* result);

/**
 * Get the version string
 *
 * @return A pointer to a C string that must be freed with tl_free_string()
 *
 * Example:
 * ```c
 * char* version = tl_version();
 * printf("TensorLogic version: %s\n", version);
 * tl_free_string(version);
 * ```
 */
char* tl_version(void);

/**
 * Check if a backend is available
 *
 * @param backend The backend name (e.g., "cpu", "parallel", "simd", "gpu")
 * @return 1 if available, 0 if not
 *
 * Example:
 * ```c
 * if (tl_is_backend_available("cpu")) {
 *     printf("CPU backend is available\n");
 * }
 * if (tl_is_backend_available("gpu")) {
 *     printf("GPU backend is available\n");
 * } else {
 *     printf("GPU backend is not available\n");
 * }
 * ```
 */
int32_t tl_is_backend_available(const char* backend);

#ifdef __cplusplus
}
#endif

#endif /* TENSORLOGIC_H */
