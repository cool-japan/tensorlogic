//! Expression optimization passes.
//!
//! This module provides various optimization passes for TLExpr expressions:
//!
//! - **Negation optimization**: Apply De Morgan's laws and eliminate double negations
//! - **Constant folding**: Evaluate constant expressions at compile time
//! - **Algebraic simplification**: Apply mathematical identities (x+0=x, x*1=x, etc.)
//! - **Pipeline**: Multi-pass optimization combining all passes intelligently
//!
//! # Quick Start
//!
//! For most use cases, use the unified optimization pipeline:
//!
//! ```
//! use tensorlogic_compiler::optimize::{OptimizationPipeline, PipelineConfig};
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let pipeline = OptimizationPipeline::new();
//! let expr = TLExpr::add(
//!     TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
//!     TLExpr::Constant(0.0)
//! );
//! let (optimized, stats) = pipeline.optimize(&expr);
//! ```
//!
//! For fine-grained control, use individual passes:
//!
//! ```
//! use tensorlogic_compiler::optimize::{fold_constants, simplify_algebraic};
//! use tensorlogic_ir::TLExpr;
//!
//! let expr = TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0));
//! let (step1, _) = fold_constants(&expr);
//! let (optimized, _) = simplify_algebraic(&step1);
//! ```

pub mod algebraic;
pub mod constant_folding;
pub mod negation;
pub mod pipeline;

pub use algebraic::{simplify_algebraic, AlgebraicSimplificationStats};
pub use constant_folding::{fold_constants, ConstantFoldingStats};
pub use negation::{optimize_negations, NegationOptStats};
pub use pipeline::{IterationStats, OptimizationPipeline, PipelineConfig, PipelineStats};
