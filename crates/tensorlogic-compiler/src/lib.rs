//! TLExpr → EinsumGraph compiler (planning only).
//!
//! **Version**: 0.1.2 | **Status**: Production Ready
//!
//! This crate compiles logical expressions into tensor computation graphs
//! represented as einsum operations. It provides a bridge between symbolic
//! logic and numeric tensor computations.
//!
//! # Overview
//!
//! The tensorlogic-compiler translates high-level logical expressions (predicates,
//! quantifiers, implications) into low-level tensor operations that can be executed
//! efficiently on various backends (CPU, GPU, etc.).
//!
//! **Key Features:**
//! - Logic-to-tensor mapping with configurable strategies
//! - Type checking and scope analysis
//! - Optimization passes (negation, CSE, einsum optimization)
//! - Enhanced diagnostics with helpful error messages
//! - Support for arithmetic, comparison, and conditional expressions
//!
//! # Quick Start
//!
//! ```rust
//! use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let mut ctx = CompilerContext::new();
//! ctx.add_domain("Person", 100);
//!
//! // Define a logic rule: ∃y. knows(x, y)
//! // "Find all persons x who know someone"
//! let rule = TLExpr::exists(
//!     "y",
//!     "Person",
//!     TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
//! );
//!
//! // Compile to tensor operations
//! let graph = compile_to_einsum_with_context(&rule, &mut ctx).unwrap();
//! ```
//!
//! # Compilation Pipeline
//!
//! The compiler follows a multi-stage pipeline:
//!
//! 1. **Pre-compilation passes**:
//!    - Scope analysis (detect unbound variables)
//!    - Type checking (validate predicate arity and types)
//!    - Expression optimization (negation optimization, CSE)
//!
//! 2. **Compilation**:
//!    - Variable axis assignment
//!    - Logic-to-tensor mapping (using configurable strategies)
//!    - Einsum graph construction
//!
//! 3. **Post-compilation passes**:
//!    - Dead code elimination
//!    - Einsum operation merging
//!    - Identity elimination
//!
//! # Modules
//!
//! - [`config`]: Compilation configuration and strategy selection
//! - [`optimize`]: Expression-level optimization passes
//! - [`passes`]: Analysis and validation passes
//!
//! # Examples
//!
//! ## Basic Predicate Compilation
//!
//! ```rust
//! use tensorlogic_compiler::compile_to_einsum;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
//! let graph = compile_to_einsum(&expr).unwrap();
//! ```
//!
//! ## Compilation with Context
//!
//! ```rust
//! use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let mut ctx = CompilerContext::new();
//! ctx.add_domain("Person", 100);
//!
//! let expr = TLExpr::exists(
//!     "y",
//!     "Person",
//!     TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
//! );
//!
//! let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
//! ```
//!
//! ## Using Optimization Passes
//!
//! ### Unified Pipeline (Recommended)
//!
//! ```rust
//! use tensorlogic_compiler::optimize::{OptimizationPipeline, PipelineConfig};
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let pipeline = OptimizationPipeline::new();
//! let expr = TLExpr::add(
//!     TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
//!     TLExpr::Constant(0.0)
//! );
//! let (optimized, stats) = pipeline.optimize(&expr);
//! println!("Applied {} optimizations", stats.total_optimizations());
//! ```
//!
//! ### Individual Passes
//!
//! ```rust
//! use tensorlogic_compiler::optimize::optimize_negations;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let expr = TLExpr::negate(TLExpr::negate(
//!     TLExpr::pred("p", vec![Term::var("x")])
//! ));
//!
//! let (optimized, stats) = optimize_negations(&expr);
//! assert_eq!(stats.double_negations_eliminated, 1);
//! ```

pub mod bytecode;
pub mod cache;
pub mod compile;
pub mod complexity;
pub mod config;
pub mod const_prop;
mod context;
pub mod dead_code;
pub mod debug;
pub mod error_recovery;
pub mod export;
pub mod expr_diff;
pub mod import;
pub mod incremental;
pub mod inline;
pub mod jit;
pub mod optimize;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod partial_eval;
pub mod passes;
pub mod pipeline;
pub mod profiling;
pub mod rewrite;
pub mod symbolic_diff;
pub mod type_infer;

#[cfg(test)]
mod property_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_math_ops;

use anyhow::Result;
use tensorlogic_ir::{EinsumGraph, TLExpr};

pub use cache::{
    CacheStats, CachedResult, CachingCompiler, CompilationCache, ExprFingerprint,
    LruCompilationCache,
};
pub use complexity::{
    check_complexity, BatchComplexityStats, ComplexityComparison, ComplexityThresholds,
    ComplexityWarning, ExprComplexity, WarningSeverity,
};
pub use config::{
    AndStrategy, CompilationConfig, CompilationConfigBuilder, ExistsStrategy, ForallStrategy,
    ImplicationStrategy, ModalStrategy, NotStrategy, OrStrategy, TemporalStrategy,
};
pub use const_prop::{ConstPropConfig, ConstPropStats, ConstantPropagator};
pub use context::{CompilerContext, DomainInfo};
pub use dead_code::{DceConfig, DceStats, DeadCodeEliminator};
pub use error_recovery::{
    compile_tolerant, compile_tolerant_with_strategy, Diagnostic, DiagnosticCollector,
    PartialCompilationResult, RecoveryAction, RecoveryStrategy, Severity, SourceSpan,
    TolerantCompiler,
};
pub use expr_diff::{expr_diff, DiffEntry, DiffKind, ExprDiff};
pub use inline::{InlineConfig, InlineStats, LetInliner};
pub use jit::{JitCompiler, JitError, JitStats};
pub use partial_eval::{
    partially_evaluate, specialize, specialize_batch, PEConfig, PEEnv, PEResult, PEStats, PEValue,
};
pub use pipeline::{
    CompilerPassId, CompilerPassOrder, CompilerPassStats, CompilerPipeline, CompilerPipelineConfig,
    CompilerPipelineResult, CompilerPipelineStats, PassBenchmark,
};
pub use profiling::{profile_phase, CompilationProfiler, ProfileEntry, ProfileReport};
pub use rewrite::{
    builtin_rules, EliminateAndTrue, EliminateDoubleNeg, EliminateOrFalse, FlattenNestedAnd,
    FlattenNestedOr, RewriteEngine, RewriteRule, RewriteStats,
};
pub use symbolic_diff::{
    differentiate, jacobian, simplify_derivative, DiffConfig, DiffError, DiffResult,
};
pub use type_infer::{
    annotate, infer_type, Substitution as TypeSubstitution, TLType, TyVarCounter, TypeEnv,
    TypeInferError, TypeInferResult, TypedExpr,
};

// Re-export adapter types for convenience
pub use passes::validate_arity;
pub use tensorlogic_adapters::{PredicateInfo, SymbolTable};

// Re-export bytecode VM types and functions
pub use bytecode::{
    compile, compile_with_config, execute, execute_with_stats, BytecodeProgram, CompileError,
    Instruction, VmEnv, VmError, VmStats, VmValue,
};

use compile::{compile_expr, infer_domain};

/// Compile a TLExpr into an EinsumGraph with an empty context.
///
/// This is the simplest entry point for compilation. It creates a new
/// compiler context automatically and infers domains where possible.
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::compile_to_einsum;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
/// let graph = compile_to_einsum(&expr).unwrap();
/// ```
pub fn compile_to_einsum(expr: &TLExpr) -> Result<EinsumGraph> {
    let mut ctx = CompilerContext::new();
    compile_to_einsum_with_context(expr, &mut ctx)
}

/// Compile a TLExpr into an EinsumGraph with a custom compilation configuration.
///
/// This allows you to control how logical operations are compiled to tensor operations,
/// using different strategies for AND, OR, NOT, quantifiers, and other logic operators.
///
/// # Arguments
///
/// * `expr` - The logical expression to compile
/// * `config` - Compilation configuration specifying strategies
///
/// # Returns
///
/// An `EinsumGraph` representing the compiled tensor computation.
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::{compile_to_einsum_with_config, CompilationConfig};
/// use tensorlogic_ir::{TLExpr, Term};
///
/// // Use Łukasiewicz fuzzy logic
/// let config = CompilationConfig::fuzzy_lukasiewicz();
/// let expr = TLExpr::and(
///     TLExpr::pred("P", vec![Term::var("x")]),
///     TLExpr::pred("Q", vec![Term::var("x")]),
/// );
/// let graph = compile_to_einsum_with_config(&expr, &config).unwrap();
///
/// // Use hard Boolean logic
/// let config = CompilationConfig::hard_boolean();
/// let graph = compile_to_einsum_with_config(&expr, &config).unwrap();
///
/// // Use probabilistic logic
/// let config = CompilationConfig::probabilistic();
/// let graph = compile_to_einsum_with_config(&expr, &config).unwrap();
/// ```
pub fn compile_to_einsum_with_config(
    expr: &TLExpr,
    config: &CompilationConfig,
) -> Result<EinsumGraph> {
    let mut ctx = CompilerContext::with_config(config.clone());
    compile_to_einsum_with_context(expr, &mut ctx)
}

/// Compile a TLExpr into an EinsumGraph with an existing context.
///
/// Use this when you need fine-grained control over domains, variable bindings,
/// or when compiling multiple related expressions with shared context.
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
/// use tensorlogic_ir::{TLExpr, Term};
///
/// let mut ctx = CompilerContext::new();
/// ctx.add_domain("Person", 100);
///
/// let expr = TLExpr::exists(
///     "y",
///     "Person",
///     TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
/// );
///
/// let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
/// ```
pub fn compile_to_einsum_with_context(
    expr: &TLExpr,
    ctx: &mut CompilerContext,
) -> Result<EinsumGraph> {
    let mut graph = EinsumGraph::new();

    let free_vars = expr.free_vars();
    for var in free_vars.iter() {
        if !ctx.var_to_domain.contains_key(var) {
            if let Some(domain) = infer_domain(expr, var) {
                ctx.bind_var(var, &domain)?;
            }
        }
        ctx.assign_axis(var);
    }

    let result = compile_expr(expr, ctx, &mut graph)?;

    // Mark the result tensor as an output
    graph.outputs.push(result.tensor_idx);

    Ok(graph)
}
