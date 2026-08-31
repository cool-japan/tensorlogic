//! Tensorlogic - Logic-as-Tensor planning layer
//!
//! **Version**: 0.1.2 | **Status**: Production Ready
//!
//! This is the top-level umbrella crate that re-exports all TensorLogic components.
//!
//! ## Overview
//!
//! TensorLogic compiles logical rules (predicates, quantifiers, implications) into
//! **tensor equations (einsum graphs)** with a minimal DSL + IR, enabling neural/symbolic/
//! probabilistic models within a unified tensor computation framework.
//!
//! ## Key Features
//!
//! - 🧠 **Logic-to-Tensor Compilation**: Compile complex logical rules into optimized tensor operations
//! - ⚡ **High Performance**: SciRS2 backend with SIMD acceleration (2-4x speedup)
//! - 🔧 **Multiple Backends**: CPU, SIMD-accelerated CPU, GPU (future)
//! - 🧪 **Extensively Tested**: 4,287 tests with 100% pass rate
//! - 📊 **Comprehensive Benchmarks**: 24 benchmark groups across 5 suites
//!
//! ## Architecture
//!
//! - **Planning Layer**: `ir`, `compiler`, `infer`, `adapters`
//! - **Execution Layer**: `scirs_backend`, `train`
//! - **Integration Layer**: `oxirs_bridge`, `sklears_kernels`, `quantrs_hooks`, `trustformers`
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use tensorlogic::prelude::*;
//!
//! // Define a logical rule: knows(x, y) ∧ knows(y, z) → knows(x, z)
//! let x = Term::var("x");
//! let y = Term::var("y");
//! let z = Term::var("z");
//!
//! let knows_xy = TLExpr::pred("knows", vec![x.clone(), y.clone()]);
//! let knows_yz = TLExpr::pred("knows", vec![y.clone(), z.clone()]);
//! let premise = TLExpr::and(knows_xy, knows_yz);
//!
//! // Compile to tensor graph
//! let graph = compile_to_einsum(&premise)?;
//!
//! // Execute with SciRS2 backend
//! let mut executor = Scirs2Exec::new();
//! let result = executor.forward(&graph)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Core planning layer (engine-agnostic) — always available
pub use tensorlogic_adapters as adapters;
pub use tensorlogic_compiler as compiler;
pub use tensorlogic_infer as infer;
pub use tensorlogic_ir as ir;

// Execution layer — scirs-backend is mandatory; train is feature-gated
pub use tensorlogic_scirs_backend as scirs_backend;
#[cfg(feature = "train")]
pub use tensorlogic_train as train;

// Integration layer — each crate is gated behind its own feature flag
#[cfg(feature = "oxirs")]
pub use tensorlogic_oxirs_bridge as oxirs_bridge;
#[cfg(feature = "quantrs")]
pub use tensorlogic_quantrs_hooks as quantrs_hooks;
#[cfg(feature = "sklears")]
pub use tensorlogic_sklears_kernels as sklears_kernels;
#[cfg(feature = "trustformers")]
pub use tensorlogic_trustformers as trustformers;

// Round 5 sub-feature crates
#[cfg(feature = "rng")]
pub use tensorlogic_oxicuda_rng as oxicuda_rng;
#[cfg(feature = "solver")]
pub use tensorlogic_oxicuda_solver as oxicuda_solver;
#[cfg(feature = "sparse")]
pub use tensorlogic_oxicuda_sparse as oxicuda_sparse;

/// Prelude module for convenient imports.
///
/// Re-exports the smallest set of types needed to build, compile, and
/// execute a TensorLogic expression end-to-end.
pub mod prelude {
    // Core construction + compilation + execution entry points.
    pub use crate::compiler::compile_to_einsum;
    pub use crate::infer::{TlAutodiff, TlExecutor};
    pub use crate::ir::{TLExpr, Term};
    pub use crate::scirs_backend::Scirs2Exec;

    /// Convenience: tunable compilation strategies (AND/OR/quantifier semantics).
    pub use crate::compiler::CompilationConfig;
    /// Convenience: the compiled execution graph type returned by `compile_to_einsum`.
    pub use crate::ir::EinsumGraph;

    // Canonical top-level error types from each mandatory sub-crate.
    pub use crate::adapters::AdapterError;
    pub use crate::compiler::CompileError;
    pub use crate::infer::ExecutorError;
    pub use crate::ir::IrError;
    pub use crate::scirs_backend::TlBackendError;

    // Canonical error types from optional sub-crates (feature-gated).
    #[cfg(feature = "oxirs")]
    pub use crate::oxirs_bridge::BridgeError;
    #[cfg(feature = "quantrs")]
    pub use crate::quantrs_hooks::PgmError;
    #[cfg(feature = "sklears")]
    pub use crate::sklears_kernels::KernelError;
    #[cfg(feature = "train")]
    pub use crate::train::TrainError;
    #[cfg(feature = "trustformers")]
    pub use crate::trustformers::TrustformerError;
}
