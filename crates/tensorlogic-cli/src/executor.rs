// ! CLI execution engine with backend selection
//!
//! Provides execution capabilities for compiled TensorLogic graphs with:
//! - Multiple backend support (SciRS2 CPU, SIMD, GPU)
//! - Input tensor generation and validation
//! - Result formatting and visualization
//! - Performance profiling

use anyhow::Result;
use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::random::thread_rng;
use std::collections::HashMap;
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::EinsumGraph;
use tensorlogic_scirs_backend::{
    DeviceType, ParallelScirs2Exec, ProfiledScirs2Exec, Scirs2Exec, Scirs2Tensor,
};

use crate::output::{print_info, print_success};

// Type alias for tensor IDs (indices into graph.tensors)
type TensorId = usize;

/// Backend types supported by the CLI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// SciRS2 CPU backend (default)
    #[default]
    SciRS2CPU,
    /// SciRS2 with SIMD acceleration
    #[cfg(feature = "simd")]
    SciRS2SIMD,
    /// SciRS2 GPU backend
    #[cfg(feature = "gpu")]
    SciRS2GPU,
    /// Parallel executor with Rayon
    Parallel,
    /// Profiled executor with performance tracking
    Profiled,
}

impl Backend {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Backend::SciRS2CPU => "SciRS2 CPU",
            #[cfg(feature = "simd")]
            Backend::SciRS2SIMD => "SciRS2 SIMD",
            #[cfg(feature = "gpu")]
            Backend::SciRS2GPU => "SciRS2 GPU",
            Backend::Parallel => "SciRS2 Parallel",
            Backend::Profiled => "SciRS2 Profiled",
        }
    }

    /// Check if backend is available
    pub fn is_available(&self) -> bool {
        match self {
            Backend::SciRS2CPU => true,
            #[cfg(feature = "simd")]
            Backend::SciRS2SIMD => true, // SIMD availability checked at runtime
            #[cfg(feature = "gpu")]
            Backend::SciRS2GPU => true, // GPU availability checked at runtime
            Backend::Parallel => true,
            Backend::Profiled => true,
        }
    }

    /// Get all available backends
    pub fn available_backends() -> Vec<Backend> {
        let backends = vec![Backend::SciRS2CPU, Backend::Parallel, Backend::Profiled];

        // Note: SIMD and GPU backends would be added here when features are enabled
        // #[cfg(feature = "simd")]
        // if BackendCapabilities::default().has_simd {
        //     backends.push(Backend::SciRS2SIMD);
        // }
        //
        // #[cfg(feature = "gpu")]
        // if BackendCapabilities::default().has_gpu {
        //     backends.push(Backend::SciRS2GPU);
        // }

        backends
    }

    /// Parse backend from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cpu" | "scirs2-cpu" => Ok(Backend::SciRS2CPU),
            #[cfg(feature = "simd")]
            "simd" | "scirs2-simd" => Ok(Backend::SciRS2SIMD),
            #[cfg(feature = "gpu")]
            "gpu" | "scirs2-gpu" => Ok(Backend::SciRS2GPU),
            "parallel" => Ok(Backend::Parallel),
            "profiled" => Ok(Backend::Profiled),
            _ => anyhow::bail!("Unknown backend: {}", s),
        }
    }
}

// Default implementation is now derived with #[default] attribute

/// Execution configuration
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Backend to use
    pub backend: Backend,
    /// Device type (reserved for future use)
    #[allow(dead_code)]
    pub device: DeviceType,
    /// Show performance metrics
    pub show_metrics: bool,
    /// Show intermediate tensors
    pub show_intermediates: bool,
    /// Validate shapes before execution (reserved for future use)
    #[allow(dead_code)]
    pub validate_shapes: bool,
    /// Enable execution tracing
    pub trace: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            backend: Backend::default(),
            device: DeviceType::Cpu,
            show_metrics: false,
            show_intermediates: false,
            validate_shapes: true,
            trace: false,
        }
    }
}

/// Execution result with tensor data and metadata
#[derive(Debug)]
pub struct ExecutionResult {
    /// Output tensor
    pub output: Scirs2Tensor,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Intermediate tensors (if requested)
    pub intermediates: HashMap<TensorId, Scirs2Tensor>,
    /// Backend used
    pub backend: Backend,
    /// Memory used in bytes
    pub memory_bytes: usize,
}

impl ExecutionResult {
    /// Print execution summary
    pub fn print_summary(&self, config: &ExecutionConfig) {
        print_success(&format!(
            "Execution completed with {} in {:.3} ms",
            self.backend.name(),
            self.execution_time_ms
        ));

        println!("\nOutput shape: {:?}", self.output.shape());
        println!("Output dtype: f64");

        if config.show_metrics {
            println!("\nPerformance Metrics:");
            println!("  Execution time: {:.3} ms", self.execution_time_ms);
            println!("  Memory used: {} bytes", self.memory_bytes);
            println!(
                "  Throughput: {:.2} GFLOPS",
                self.estimate_flops() / self.execution_time_ms / 1_000.0
            );
        }

        if config.show_intermediates && !self.intermediates.is_empty() {
            println!("\nIntermediate Tensors:");
            for (id, tensor) in &self.intermediates {
                println!("  Tensor {}: shape {:?}", id, tensor.shape());
            }
        }

        println!("\nOutput preview (first 10 elements):");
        print_tensor_preview(&self.output, 10);
    }

    /// Estimate FLOPs for the computation
    fn estimate_flops(&self) -> f64 {
        // Simple estimate based on output size
        let elements = self.output.len() as f64;
        elements * 2.0 // Multiply-add operations
    }
}

/// Print tensor preview
fn print_tensor_preview(tensor: &Scirs2Tensor, max_elements: usize) {
    let flat = tensor.as_slice().unwrap_or(&[]);
    let preview: Vec<String> = flat
        .iter()
        .take(max_elements)
        .map(|v| format!("{:.4}", v))
        .collect();

    println!(
        "  [{}{}]",
        preview.join(", "),
        if flat.len() > max_elements {
            ", ..."
        } else {
            ""
        }
    );
}

/// CLI executor that wraps backend executors
pub struct CliExecutor {
    config: ExecutionConfig,
}

impl CliExecutor {
    /// Create new executor with configuration
    pub fn new(config: ExecutionConfig) -> Result<Self> {
        if !config.backend.is_available() {
            anyhow::bail!("Backend {} is not available", config.backend.name());
        }

        Ok(Self { config })
    }

    /// Execute graph with auto-generated input tensors
    pub fn execute(&self, graph: &EinsumGraph) -> Result<ExecutionResult> {
        let start = std::time::Instant::now();

        // Generate input tensors based on graph requirements
        let inputs = self.generate_inputs(graph)?;

        if self.config.trace {
            print_info("Generated input tensors");
            for (id, tensor) in &inputs {
                println!("  Tensor {}: shape {:?}", id, tensor.shape());
            }
        }

        // Execute based on backend
        let (output, intermediates) = match self.config.backend {
            Backend::SciRS2CPU => self.execute_cpu(graph)?,
            #[cfg(feature = "simd")]
            Backend::SciRS2SIMD => self.execute_simd(graph)?,
            #[cfg(feature = "gpu")]
            Backend::SciRS2GPU => self.execute_gpu(graph)?,
            Backend::Parallel => self.execute_parallel(graph)?,
            Backend::Profiled => self.execute_profiled(graph)?,
        };

        let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Estimate memory usage
        let memory_bytes = self.estimate_memory(&output, &intermediates);

        Ok(ExecutionResult {
            output,
            execution_time_ms,
            intermediates,
            backend: self.config.backend,
            memory_bytes,
        })
    }

    /// Execute with CPU backend
    fn execute_cpu(
        &self,
        graph: &EinsumGraph,
    ) -> Result<(Scirs2Tensor, HashMap<TensorId, Scirs2Tensor>)> {
        let mut executor = Scirs2Exec::new();
        let output = executor
            .forward(graph)
            .map_err(|e| anyhow::anyhow!("Execution failed: {:?}", e))?;

        Ok((output, HashMap::new()))
    }

    /// Execute with SIMD backend
    #[cfg(feature = "simd")]
    fn execute_simd(
        &self,
        graph: &EinsumGraph,
    ) -> Result<(Scirs2Tensor, HashMap<TensorId, Scirs2Tensor>)> {
        // SIMD is transparent in SciRS2, same as CPU
        self.execute_cpu(graph)
    }

    /// Execute with GPU backend
    #[cfg(feature = "gpu")]
    fn execute_gpu(
        &self,
        graph: &EinsumGraph,
    ) -> Result<(Scirs2Tensor, HashMap<TensorId, Scirs2Tensor>)> {
        // GPU support will be implemented when the backend API is ready
        // For now, fall back to CPU execution
        self.execute_cpu(graph)
    }

    /// Execute with parallel backend
    fn execute_parallel(
        &self,
        graph: &EinsumGraph,
    ) -> Result<(Scirs2Tensor, HashMap<TensorId, Scirs2Tensor>)> {
        let mut executor = ParallelScirs2Exec::new();
        let output = executor
            .forward(graph)
            .map_err(|e| anyhow::anyhow!("Parallel execution failed: {:?}", e))?;

        Ok((output, HashMap::new()))
    }

    /// Execute with profiled backend
    fn execute_profiled(
        &self,
        graph: &EinsumGraph,
    ) -> Result<(Scirs2Tensor, HashMap<TensorId, Scirs2Tensor>)> {
        let mut executor = ProfiledScirs2Exec::new();
        let output = executor
            .forward(graph)
            .map_err(|e| anyhow::anyhow!("Profiled execution failed: {:?}", e))?;

        // Print profiling results
        if self.config.show_metrics {
            // Note: Profiling data would be extracted here when the API is available
            println!("\nProfiling Results:");
            println!("  (Profiling data collection not yet implemented)");
        }

        Ok((output, HashMap::new()))
    }

    /// Generate input tensors for graph
    ///
    /// Creates random input tensors for each input tensor in the graph.
    /// Default shape is 1D with 100 elements.
    fn generate_inputs(&self, graph: &EinsumGraph) -> Result<HashMap<TensorId, Scirs2Tensor>> {
        let mut inputs = HashMap::new();
        let mut rng = thread_rng();

        // Default dimension for input tensors
        let default_dim = 100;

        // Generate random inputs for each input tensor in the graph
        for &input_idx in &graph.inputs {
            // Generate random values in [0, 1] range
            let data: Vec<f64> = (0..default_dim)
                .map(|_| rng.random_range(0.0..1.0))
                .collect();

            // Create dynamic dimensioned array (1D with default_dim elements)
            let tensor = Array::from_shape_vec(IxDyn(&[default_dim]), data)
                .map_err(|e| anyhow::anyhow!("Failed to create tensor: {}", e))?;

            inputs.insert(input_idx, tensor);
        }

        Ok(inputs)
    }

    /// Estimate memory usage
    fn estimate_memory(
        &self,
        output: &Scirs2Tensor,
        intermediates: &HashMap<TensorId, Scirs2Tensor>,
    ) -> usize {
        let mut total = output.len() * std::mem::size_of::<f64>();

        for tensor in intermediates.values() {
            total += tensor.len() * std::mem::size_of::<f64>();
        }

        total
    }
}

/// List available backends with their capabilities
pub fn list_backends() {
    println!("Available Backends:");
    println!();

    for backend in Backend::available_backends() {
        let status = if backend.is_available() { "✓" } else { "✗" };
        println!(
            "  {} {} - {}",
            status,
            backend.name(),
            backend_description(backend)
        );
    }

    println!();
    println!("Backend Capabilities:");
    println!("  CPU: ✓");
    println!("  SIMD: {}", if cfg!(feature = "simd") { "✓" } else { "✗" });
    println!("  GPU: {}", if cfg!(feature = "gpu") { "✓" } else { "✗" });
    println!("  Parallel: ✓ (Rayon)");
}

fn backend_description(backend: Backend) -> &'static str {
    match backend {
        Backend::SciRS2CPU => "Standard CPU execution",
        #[cfg(feature = "simd")]
        Backend::SciRS2SIMD => "Vectorized SIMD acceleration",
        #[cfg(feature = "gpu")]
        Backend::SciRS2GPU => "GPU-accelerated execution",
        Backend::Parallel => "Multi-threaded parallel execution",
        Backend::Profiled => "Execution with performance profiling",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_availability() {
        let cpu = Backend::SciRS2CPU;
        assert!(cpu.is_available());
        assert_eq!(cpu.name(), "SciRS2 CPU");
    }

    #[test]
    fn test_backend_from_str() {
        assert!(matches!(
            Backend::from_str("cpu").unwrap(),
            Backend::SciRS2CPU
        ));
        assert!(matches!(
            Backend::from_str("parallel").unwrap(),
            Backend::Parallel
        ));
        assert!(Backend::from_str("invalid").is_err());
    }

    #[test]
    fn test_default_backend() {
        let backend = Backend::default();
        assert!(backend.is_available());
    }

    #[test]
    fn test_execution_config_default() {
        let config = ExecutionConfig::default();
        assert!(config.backend.is_available());
        assert_eq!(config.device, DeviceType::Cpu);
    }

    #[test]
    fn test_available_backends() {
        let backends = Backend::available_backends();
        assert!(!backends.is_empty());
        assert!(backends.contains(&Backend::SciRS2CPU));
    }
}
