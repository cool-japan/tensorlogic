//! Tensorlogic - Logic-as-Tensor planning layer
//!
//! **Version**: 0.1.0-beta.1 | **Status**: Production Ready
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

// Core planning layer (engine-agnostic)
pub use tensorlogic_adapters as adapters;
pub use tensorlogic_compiler as compiler;
pub use tensorlogic_infer as infer;
pub use tensorlogic_ir as ir;

// Execution layer (SciRS2-powered)
pub use tensorlogic_scirs_backend as scirs_backend;
pub use tensorlogic_train as train;

// Integration layer
pub use tensorlogic_oxirs_bridge as oxirs_bridge;
pub use tensorlogic_quantrs_hooks as quantrs_hooks;
pub use tensorlogic_sklears_kernels as sklears_kernels;
pub use tensorlogic_trustformers as trustformers;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::compiler::compile_to_einsum;
    pub use crate::infer::{TlAutodiff, TlExecutor};
    pub use crate::ir::{TLExpr, Term};
    pub use crate::scirs_backend::Scirs2Exec;
}
