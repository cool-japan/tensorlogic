//! TensorLogic CLI Library
//!
//! **Version**: 0.1.0-alpha.2 | **Status**: Production Ready
//!
//! This library provides programmatic access to the TensorLogic CLI functionality,
//! allowing you to use the parser, executor, optimizer, and other components
//! directly from Rust code without shelling out to the command-line interface.
//!
//! # Features
//!
//! - **Expression Parsing**: Parse logical expressions into TensorLogic IR
//! - **Compilation**: Compile expressions to einsum graphs with various strategies
//! - **Execution**: Execute compiled graphs with multiple backends
//! - **Optimization**: Apply optimization passes to improve performance
//! - **Analysis**: Analyze graph complexity and computational costs
//! - **Benchmarking**: Measure compilation and execution performance
//! - **Format Conversion**: Convert between different input/output formats
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use tensorlogic_cli::{parser, CompilationContext};
//! use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationConfig};
//!
//! // Parse an expression
//! let expr = parser::parse_expression("pred1(x) AND pred2(x, y)").unwrap();
//!
//! // Create compiler context with configuration
//! let config = CompilationConfig::soft_differentiable();
//! let mut ctx = CompilationContext::with_config(config);
//! ctx.add_domain("D", 100);
//!
//! // Compile to einsum graph
//! let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
//!
//! // Execute the graph (requires appropriate setup)
//! // let result = executor::execute_graph(&graph, &backend_config).unwrap();
//! ```
//!
//! # Module Overview
//!
//! - [`parser`]: Parse logical expressions from strings
//! - [`executor`]: Execute compiled einsum graphs
//! - [`optimize`]: Apply optimization passes
//! - [`benchmark`]: Performance benchmarking utilities
//! - [`analysis`]: Graph analysis and metrics
//! - [`conversion`]: Format conversion utilities
//! - `config`: Configuration management (internal)
//! - [`output`]: Output formatting and colors
//!
//! # Library Mode Benefits
//!
//! Using TensorLogic as a library instead of a CLI provides:
//!
//! - **Better Performance**: No process spawning overhead
//! - **Type Safety**: Compile-time error checking
//! - **Integration**: Direct embedding in Rust applications
//! - **Flexibility**: Fine-grained control over compilation and execution
//! - **Testing**: Easier unit and integration testing
//!
//! # Example: Full Compilation Pipeline
//!
//! ```rust,no_run
//! use tensorlogic_cli::{parser, analysis, CompilationContext};
//! use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationConfig};
//!
//! // Parse expression
//! let expr = parser::parse_expression(
//!     "EXISTS x IN Person. (pred1(x) AND pred2(x, y))"
//! ).unwrap();
//!
//! // Setup compilation context
//! let config = CompilationConfig::soft_differentiable();
//! let mut ctx = CompilationContext::with_config(config);
//! ctx.add_domain("Person", 100);
//! ctx.add_domain("D", 100);
//!
//! // Compile
//! let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
//!
//! // Analyze complexity
//! let metrics = analysis::GraphMetrics::analyze(&graph);
//! println!("Graph has {} tensors and {} nodes", metrics.tensor_count, metrics.node_count);
//! println!("Estimated FLOPs: {}", metrics.estimated_flops);
//! ```

// Re-export core TensorLogic types for convenience
pub use tensorlogic_adapters;
pub use tensorlogic_compiler;
pub use tensorlogic_infer;
pub use tensorlogic_ir;
pub use tensorlogic_scirs_backend;

// Export public modules
pub mod analysis;
pub mod benchmark;
pub mod cache;
pub mod conversion;
pub mod error_suggestions;
pub mod executor;
pub mod ffi;
pub mod macros;
pub mod optimize;
pub mod output;
pub mod parser;
pub mod simplify;
pub mod snapshot;

// Re-export config types (but keep internal config logic private)
pub use config::{CacheConfig, Config, ReplConfig, WatchConfig};

// Internal modules (not part of public API)
#[allow(dead_code)]
mod batch;
#[allow(dead_code)]
mod cli;
pub(crate) mod completion;
pub(crate) mod config;
#[allow(dead_code)]
pub(crate) mod profile;
#[allow(dead_code)]
pub(crate) mod repl;
#[allow(dead_code)]
pub(crate) mod watch;

/// Type alias for compilation context used throughout the library
pub type CompilationContext = tensorlogic_compiler::CompilerContext;

/// Type alias for einsum graphs
pub type EinsumGraph = tensorlogic_ir::EinsumGraph;

/// Type alias for TensorLogic expressions
pub type TLExpr = tensorlogic_ir::TLExpr;

/// Library result type
pub type Result<T> = anyhow::Result<T>;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_const() {
        // VERSION is a compile-time constant from CARGO_PKG_VERSION
        assert!(VERSION.contains('.'), "Version should be in semver format");
        assert_eq!(NAME, "tensorlogic-cli");
    }

    #[test]
    fn test_parse_and_compile() {
        let expr = parser::parse_expression("pred(x, y)").unwrap();
        assert!(matches!(expr, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_format_conversion() {
        let expr = parser::parse_expression("AND(a, b)").unwrap();
        let formatted = conversion::format_expression(&expr, false);
        assert!(formatted.contains("AND"));
    }
}
