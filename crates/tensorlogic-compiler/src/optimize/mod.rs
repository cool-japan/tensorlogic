//! Expression optimization passes.
//!
//! This module provides various optimization passes for TLExpr expressions:
//!
//! - **Negation optimization**: Apply De Morgan's laws and eliminate double negations
//! - **Constant folding**: Evaluate constant expressions at compile time
//! - **Algebraic simplification**: Apply mathematical identities (x+0=x, x*1=x, etc.)
//! - **Strength reduction**: Replace expensive operations with cheaper equivalents
//! - **Distributivity**: Factor or expand expressions based on cost analysis
//! - **Quantifier optimization**: Hoist loop-invariant code, reorder quantifiers
//! - **Dead code elimination**: Remove unreachable code and simplify constant conditions
//! - **Complexity analysis**: Estimate computational cost and memory usage
//! - **Cost-based optimization**: Explore rewrites and select optimal execution plan
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
//! use tensorlogic_compiler::optimize::{fold_constants, simplify_algebraic, reduce_strength};
//! use tensorlogic_ir::TLExpr;
//!
//! let expr = TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0));
//! let (step1, _) = fold_constants(&expr);
//! let (step2, _) = simplify_algebraic(&step1);
//! let (optimized, _) = reduce_strength(&step2);
//! ```
//!
//! # Analysis Tools
//!
//! The module also provides analysis tools for expressions:
//!
//! ```
//! use tensorlogic_compiler::optimize::analyze_complexity;
//! use tensorlogic_compiler::CompilerContext;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let expr = TLExpr::mul(
//!     TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0)),
//!     TLExpr::Constant(3.0)
//! );
//! let complexity = analyze_complexity(&expr);
//! println!("Operations: {}", complexity.total_operations());
//! println!("Cost: {:.2}", complexity.total_cost());
//! ```

pub mod algebraic;
pub mod complexity;
pub mod constant_folding;
pub mod cost_based;
pub mod dead_code;
pub mod distributivity;
pub mod memory_estimation;
pub mod negation;
pub mod pipeline;
pub mod quantifier_opt;
pub mod strength_reduction;

pub use algebraic::{simplify_algebraic, AlgebraicSimplificationStats};
pub use complexity::{analyze_complexity, compare_complexity, CostWeights, ExpressionComplexity};
pub use constant_folding::{fold_constants, ConstantFoldingStats};
pub use cost_based::{optimize_by_cost, optimize_by_cost_with_config, CostBasedStats, RewriteRule};
pub use dead_code::{eliminate_dead_code, DeadCodeStats};
pub use distributivity::{optimize_distributivity, DistributivityStats};
pub use memory_estimation::{estimate_batch_memory, estimate_memory, MemoryEstimate};
pub use negation::{optimize_negations, NegationOptStats};
pub use pipeline::{IterationStats, OptimizationPipeline, PipelineConfig, PipelineStats};
pub use quantifier_opt::{optimize_quantifiers, QuantifierOptStats};
pub use strength_reduction::{reduce_strength, StrengthReductionStats};
