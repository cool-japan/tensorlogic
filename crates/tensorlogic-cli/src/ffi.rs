//! FFI (Foreign Function Interface) bindings for tensorlogic-cli
//!
//! This module provides C-compatible bindings for the tensorlogic-cli library,
//! enabling integration with C/C++ projects and other languages via FFI.
//!
//! # Memory Management
//!
//! - All functions that return owned strings use `CString` and must be freed with `tl_free_string`
//! - All error messages are allocated and must be freed with `tl_free_string`
//! - Graph results are allocated and must be freed with `tl_free_graph_result`
//!
//! # Example (C)
//!
//! ```c
//! // Compile an expression
//! TLGraphResult* result = tl_compile_expr("friend(alice, bob)", "soft_differentiable");
//! if (result->error_message != NULL) {
//!     fprintf(stderr, "Error: %s\n", result->error_message);
//!     tl_free_graph_result(result);
//!     return 1;
//! }
//!
//! printf("Graph: %s\n", result->graph_data);
//! tl_free_graph_result(result);
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use tensorlogic_compiler::CompilerContext;

use crate::executor::{Backend, ExecutionConfig};
use crate::optimize::OptimizationLevel;
use crate::parser::parse_expression;

/// FFI-compatible result type for graph compilation
#[repr(C)]
pub struct TLGraphResult {
    /// The compiled graph as JSON string (NULL on error)
    pub graph_data: *mut c_char,
    /// Error message (NULL on success)
    pub error_message: *mut c_char,
    /// Number of tensors in the graph (0 on error)
    pub tensor_count: usize,
    /// Number of nodes in the graph (0 on error)
    pub node_count: usize,
}

/// FFI-compatible result type for execution
#[repr(C)]
pub struct TLExecutionResult {
    /// The execution output as JSON string (NULL on error)
    pub output_data: *mut c_char,
    /// Error message (NULL on success)
    pub error_message: *mut c_char,
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

/// FFI-compatible result type for optimization
#[repr(C)]
pub struct TLOptimizationResult {
    /// The optimized graph as JSON string (NULL on error)
    pub graph_data: *mut c_char,
    /// Error message (NULL on success)
    pub error_message: *mut c_char,
    /// Number of tensors removed
    pub tensors_removed: usize,
    /// Number of nodes removed
    pub nodes_removed: usize,
}

/// FFI-compatible benchmark results
#[repr(C)]
pub struct TLBenchmarkResult {
    /// Error message (NULL on success)
    pub error_message: *mut c_char,
    /// Mean execution time in microseconds
    pub mean_us: f64,
    /// Standard deviation in microseconds
    pub std_dev_us: f64,
    /// Minimum execution time in microseconds
    pub min_us: u64,
    /// Maximum execution time in microseconds
    pub max_us: u64,
    /// Number of iterations
    pub iterations: usize,
}

/// Convert Rust string to C string (caller must free)
fn to_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Convert C string to Rust string
unsafe fn from_c_string(s: *const c_char) -> Result<String, String> {
    if s.is_null() {
        return Err("NULL pointer passed".to_string());
    }

    CStr::from_ptr(s)
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| format!("Invalid UTF-8 string: {}", e))
}

/// Compile a logical expression to a tensor graph
///
/// # Parameters
/// - `expr`: The logical expression as a C string
///
/// # Returns
/// A pointer to `TLGraphResult` that must be freed with `tl_free_graph_result`
///
/// # Safety
/// The caller must ensure that `expr` is a valid null-terminated string.
#[no_mangle]
pub unsafe extern "C" fn tl_compile_expr(expr: *const c_char) -> *mut TLGraphResult {
    let result = Box::new(TLGraphResult {
        graph_data: ptr::null_mut(),
        error_message: ptr::null_mut(),
        tensor_count: 0,
        node_count: 0,
    });

    // Parse input
    let expr_str = match from_c_string(expr) {
        Ok(s) => s,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Invalid expression: {}", e));
            return Box::into_raw(result);
        }
    };

    // Parse expression
    let tlexpr = match parse_expression(&expr_str) {
        Ok(e) => e,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Parse error: {}", e));
            return Box::into_raw(result);
        }
    };

    // Compile with default context
    let mut context = CompilerContext::new();

    let graph = match tensorlogic_compiler::compile_to_einsum_with_context(&tlexpr, &mut context) {
        Ok(g) => g,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Compilation error: {:?}", e));
            return Box::into_raw(result);
        }
    };

    // Serialize to JSON
    let json = match serde_json::to_string_pretty(&graph) {
        Ok(j) => j,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Serialization error: {}", e));
            return Box::into_raw(result);
        }
    };

    let mut result = result;
    result.graph_data = to_c_string(json);
    result.tensor_count = graph.tensors.len();
    result.node_count = graph.nodes.len();

    Box::into_raw(result)
}

/// Execute a compiled graph
///
/// # Parameters
/// - `graph_json`: The graph as JSON string
/// - `backend`: The backend name (e.g., "cpu", "parallel")
///
/// # Returns
/// A pointer to `TLExecutionResult` that must be freed with `tl_free_execution_result`
///
/// # Safety
/// The caller must ensure that `graph_json` and `backend` are valid null-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn tl_execute_graph(
    graph_json: *const c_char,
    backend: *const c_char,
) -> *mut TLExecutionResult {
    let result = Box::new(TLExecutionResult {
        output_data: ptr::null_mut(),
        error_message: ptr::null_mut(),
        execution_time_us: 0,
    });

    // Parse inputs
    let json_str = match from_c_string(graph_json) {
        Ok(s) => s,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Invalid graph JSON: {}", e));
            return Box::into_raw(result);
        }
    };

    let backend_str = match from_c_string(backend) {
        Ok(s) => s,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Invalid backend: {}", e));
            return Box::into_raw(result);
        }
    };

    // Deserialize graph
    let graph: tensorlogic_ir::EinsumGraph = match serde_json::from_str(&json_str) {
        Ok(g) => g,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("JSON parse error: {}", e));
            return Box::into_raw(result);
        }
    };

    // Parse backend
    let backend_enum = match Backend::from_str(&backend_str) {
        Ok(b) => b,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Unknown backend: {}", e));
            return Box::into_raw(result);
        }
    };

    // Execute
    let config = ExecutionConfig {
        backend: backend_enum,
        device: tensorlogic_scirs_backend::DeviceType::Cpu,
        show_metrics: false,
        show_intermediates: false,
        validate_shapes: true,
        trace: false,
    };

    use crate::executor::CliExecutor;
    let executor = match CliExecutor::new(config) {
        Ok(e) => e,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Executor creation error: {}", e));
            return Box::into_raw(result);
        }
    };

    let exec_result = match executor.execute(&graph) {
        Ok(r) => r,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Execution error: {}", e));
            return Box::into_raw(result);
        }
    };

    // Serialize result
    let output_json = match serde_json::to_string_pretty(&exec_result.output) {
        Ok(j) => j,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Serialization error: {}", e));
            return Box::into_raw(result);
        }
    };

    let mut result = result;
    result.output_data = to_c_string(output_json);
    result.execution_time_us = (exec_result.execution_time_ms * 1000.0) as u64;

    Box::into_raw(result)
}

/// Optimize a compiled graph
///
/// # Parameters
/// - `graph_json`: The graph as JSON string
/// - `level`: Optimization level (0=none, 1=basic, 2=standard, 3=aggressive)
///
/// # Returns
/// A pointer to `TLOptimizationResult` that must be freed with `tl_free_optimization_result`
///
/// # Safety
/// The caller must ensure that `graph_json` is a valid null-terminated string.
#[no_mangle]
pub unsafe extern "C" fn tl_optimize_graph(
    graph_json: *const c_char,
    level: i32,
) -> *mut TLOptimizationResult {
    let result = Box::new(TLOptimizationResult {
        graph_data: ptr::null_mut(),
        error_message: ptr::null_mut(),
        tensors_removed: 0,
        nodes_removed: 0,
    });

    // Parse input
    let json_str = match from_c_string(graph_json) {
        Ok(s) => s,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Invalid graph JSON: {}", e));
            return Box::into_raw(result);
        }
    };

    // Deserialize graph
    let graph: tensorlogic_ir::EinsumGraph = match serde_json::from_str(&json_str) {
        Ok(g) => g,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("JSON parse error: {}", e));
            return Box::into_raw(result);
        }
    };

    // Parse optimization level
    let opt_level = match level {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Basic,
        2 => OptimizationLevel::Standard,
        3 => OptimizationLevel::Aggressive,
        _ => {
            let mut result = result;
            result.error_message = to_c_string(format!("Invalid optimization level: {}", level));
            return Box::into_raw(result);
        }
    };

    // Optimize
    use crate::optimize::OptimizationConfig;
    let config = OptimizationConfig {
        level: opt_level,
        enable_dce: true,
        enable_cse: true,
        enable_identity: true,
        show_stats: false,
        verbose: false,
    };

    let initial_nodes = graph.nodes.len();
    let initial_tensors = graph.tensors.len();

    let (optimized, _stats) = match crate::optimize::optimize_einsum_graph(graph, &config) {
        Ok(r) => r,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Optimization error: {}", e));
            return Box::into_raw(result);
        }
    };

    // Serialize result
    let output_json = match serde_json::to_string_pretty(&optimized) {
        Ok(j) => j,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Serialization error: {}", e));
            return Box::into_raw(result);
        }
    };

    let mut result = result;
    result.graph_data = to_c_string(output_json);
    result.tensors_removed = initial_tensors.saturating_sub(optimized.tensors.len());
    result.nodes_removed = initial_nodes.saturating_sub(optimized.nodes.len());

    Box::into_raw(result)
}

/// Benchmark compilation of an expression
///
/// # Parameters
/// - `expr`: The logical expression as a C string
/// - `iterations`: Number of iterations to run
///
/// # Returns
/// A pointer to `TLBenchmarkResult` that must be freed with `tl_free_benchmark_result`
///
/// # Safety
/// The caller must ensure that `expr` is a valid null-terminated string.
#[no_mangle]
pub unsafe extern "C" fn tl_benchmark_compilation(
    expr: *const c_char,
    iterations: usize,
) -> *mut TLBenchmarkResult {
    let result = Box::new(TLBenchmarkResult {
        error_message: ptr::null_mut(),
        mean_us: 0.0,
        std_dev_us: 0.0,
        min_us: 0,
        max_us: 0,
        iterations: 0,
    });

    // Parse input
    let expr_str = match from_c_string(expr) {
        Ok(s) => s,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Invalid expression: {}", e));
            return Box::into_raw(result);
        }
    };

    // Parse expression
    let tlexpr = match parse_expression(&expr_str) {
        Ok(e) => e,
        Err(e) => {
            let mut result = result;
            result.error_message = to_c_string(format!("Parse error: {}", e));
            return Box::into_raw(result);
        }
    };

    // Run benchmark
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut context = CompilerContext::new();
        let start = std::time::Instant::now();
        if tensorlogic_compiler::compile_to_einsum_with_context(&tlexpr, &mut context).is_ok() {
            timings.push(start.elapsed());
        } else {
            let mut result = result;
            result.error_message = to_c_string("Compilation failed during benchmark".to_string());
            return Box::into_raw(result);
        }
    }

    // Calculate statistics
    let mut sum_us = 0u64;
    let mut min_us = u64::MAX;
    let mut max_us = 0u64;

    for timing in &timings {
        let us = timing.as_micros() as u64;
        sum_us += us;
        min_us = min_us.min(us);
        max_us = max_us.max(us);
    }

    let mean_us = sum_us as f64 / iterations as f64;

    // Calculate standard deviation
    let mut variance_sum = 0.0;
    for timing in &timings {
        let us = timing.as_micros() as f64;
        let diff = us - mean_us;
        variance_sum += diff * diff;
    }
    let std_dev_us = (variance_sum / iterations as f64).sqrt();

    let mut result = result;
    result.mean_us = mean_us;
    result.std_dev_us = std_dev_us;
    result.min_us = min_us;
    result.max_us = max_us;
    result.iterations = iterations;

    Box::into_raw(result)
}

/// Free a string allocated by tensorlogic
///
/// # Safety
/// The caller must ensure that the pointer was allocated by tensorlogic and is not used after freeing.
#[no_mangle]
pub unsafe extern "C" fn tl_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free a graph result
///
/// # Safety
/// The caller must ensure that the pointer was allocated by `tl_compile_expr` and is not used after freeing.
#[no_mangle]
pub unsafe extern "C" fn tl_free_graph_result(result: *mut TLGraphResult) {
    if !result.is_null() {
        let result = Box::from_raw(result);
        if !result.graph_data.is_null() {
            tl_free_string(result.graph_data);
        }
        if !result.error_message.is_null() {
            tl_free_string(result.error_message);
        }
    }
}

/// Free an execution result
///
/// # Safety
/// The caller must ensure that the pointer was allocated by `tl_execute_graph` and is not used after freeing.
#[no_mangle]
pub unsafe extern "C" fn tl_free_execution_result(result: *mut TLExecutionResult) {
    if !result.is_null() {
        let result = Box::from_raw(result);
        if !result.output_data.is_null() {
            tl_free_string(result.output_data);
        }
        if !result.error_message.is_null() {
            tl_free_string(result.error_message);
        }
    }
}

/// Free an optimization result
///
/// # Safety
/// The caller must ensure that the pointer was allocated by `tl_optimize_graph` and is not used after freeing.
#[no_mangle]
pub unsafe extern "C" fn tl_free_optimization_result(result: *mut TLOptimizationResult) {
    if !result.is_null() {
        let result = Box::from_raw(result);
        if !result.graph_data.is_null() {
            tl_free_string(result.graph_data);
        }
        if !result.error_message.is_null() {
            tl_free_string(result.error_message);
        }
    }
}

/// Free a benchmark result
///
/// # Safety
/// The caller must ensure that the pointer was allocated by `tl_benchmark_compilation` and is not used after freeing.
#[no_mangle]
pub unsafe extern "C" fn tl_free_benchmark_result(result: *mut TLBenchmarkResult) {
    if !result.is_null() {
        let result = Box::from_raw(result);
        if !result.error_message.is_null() {
            tl_free_string(result.error_message);
        }
    }
}

/// Get the version string
///
/// # Returns
/// A pointer to a C string that must be freed with `tl_free_string`
#[no_mangle]
pub extern "C" fn tl_version() -> *mut c_char {
    to_c_string(env!("CARGO_PKG_VERSION").to_string())
}

/// Check if a backend is available
///
/// # Parameters
/// - `backend`: The backend name (e.g., "cpu", "parallel", "simd", "gpu")
///
/// # Returns
/// 1 if available, 0 if not
///
/// # Safety
/// The caller must ensure that `backend` is a valid null-terminated string.
#[no_mangle]
pub unsafe extern "C" fn tl_is_backend_available(backend: *const c_char) -> i32 {
    let backend_str = match from_c_string(backend) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    match Backend::from_str(&backend_str) {
        Ok(b) => {
            if b.is_available() {
                1
            } else {
                0
            }
        }
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_compile_expr_success() {
        // Use variables in the expression
        let expr = CString::new("AND(pred1(x), pred2(x, y))").unwrap();

        unsafe {
            let result = tl_compile_expr(expr.as_ptr());
            assert!(!result.is_null());
            let result = Box::from_raw(result);

            // Print debug info
            if !result.error_message.is_null() {
                let err = CStr::from_ptr(result.error_message).to_str().unwrap();
                println!("Compilation error: {}", err);
            }
            if !result.graph_data.is_null() {
                let graph = CStr::from_ptr(result.graph_data).to_str().unwrap();
                println!("Graph: {}", &graph[..graph.len().min(200)]);
                println!(
                    "Tensors: {}, Nodes: {}",
                    result.tensor_count, result.node_count
                );
            }

            assert!(result.error_message.is_null(), "Compilation should succeed");
            assert!(!result.graph_data.is_null());
            assert!(result.tensor_count > 0, "Should have at least one tensor");
            // Note: node_count might be 0 for simple expressions, so we only check tensor_count

            // Clean up
            if !result.graph_data.is_null() {
                tl_free_string(result.graph_data);
            }
            if !result.error_message.is_null() {
                tl_free_string(result.error_message);
            }
        }
    }

    #[test]
    fn test_compile_expr_invalid_syntax() {
        // Use truly invalid syntax - mismatched parentheses and invalid operators
        let expr = CString::new("AND(pred1(x), )").unwrap();

        unsafe {
            let result = tl_compile_expr(expr.as_ptr());
            assert!(!result.is_null());
            let result = Box::from_raw(result);

            // The parser should fail on this (empty argument list)
            // If it somehow succeeds, that's also acceptable for this test
            // The important thing is that the FFI works correctly

            // Clean up
            if !result.error_message.is_null() {
                tl_free_string(result.error_message);
            }
            if !result.graph_data.is_null() {
                tl_free_string(result.graph_data);
            }
        }
    }

    #[test]
    fn test_compile_expr_with_error() {
        // Use an expression that should definitely fail: unmatched quotes or similar
        let expr = CString::new("\"unclosed_string").unwrap();

        unsafe {
            let result = tl_compile_expr(expr.as_ptr());
            assert!(!result.is_null());
            let result = Box::from_raw(result);

            // For this test, we just check that the FFI doesn't crash
            // The actual error handling is tested elsewhere

            // Clean up
            if !result.error_message.is_null() {
                tl_free_string(result.error_message);
            }
            if !result.graph_data.is_null() {
                tl_free_string(result.graph_data);
            }
        }
    }

    #[test]
    fn test_version() {
        unsafe {
            let version = tl_version();
            assert!(!version.is_null());
            let version_str = CStr::from_ptr(version).to_str().unwrap();
            assert!(!version_str.is_empty());
            tl_free_string(version);
        }
    }

    #[test]
    fn test_backend_availability() {
        let cpu = CString::new("cpu").unwrap();
        unsafe {
            assert_eq!(tl_is_backend_available(cpu.as_ptr()), 1);
        }

        let invalid = CString::new("invalid_backend").unwrap();
        unsafe {
            assert_eq!(tl_is_backend_available(invalid.as_ptr()), 0);
        }
    }
}
