//! Tensorlogic - Logic-as-Tensor planning layer
//!
//! This is the top-level umbrella crate that re-exports all TensorLogic components.
//!
//! # Architecture
//!
//! - **Planning Layer**: `ir`, `compiler`, `infer`, `adapters`
//! - **Execution Layer**: `scirs_backend`, `train`
//! - **Integration Layer**: `oxirs_bridge`, `sklears_kernels`, `quantrs_hooks`, `trustformers`

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
