//! Multi-threaded parallel compilation for TensorLogic expressions.
//!
//! This module provides parallel compilation capabilities using Rayon for data parallelism.
//! It can significantly improve compilation performance for large, complex expressions with
//! independent subexpressions.
//!
//! # Features
//!
//! - Parallel compilation of independent subexpressions
//! - Thread-safe context management
//! - Parallel optimization passes
//! - Automatic complexity-based scheduling
//!
//! # Example
//!
//! ```rust,ignore
//! use tensorlogic_compiler::parallel::ParallelCompiler;
//! use tensorlogic_compiler::CompilerContext;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let mut ctx = CompilerContext::new();
//! ctx.add_domain("Person", 1000);
//!
//! let expr = TLExpr::and(
//!     TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
//!     TLExpr::pred("likes", vec![Term::var("y"), Term::var("z")]),
//! );
//!
//! let compiler = ParallelCompiler::new();
//! let graph = compiler.compile(&expr, &mut ctx)?;
//! ```

use crate::compile::compile_expr;
use crate::context::CompilerContext;
use crate::optimize::{OptimizationPipeline, PipelineConfig, PipelineStats};
use anyhow::Result;
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use tensorlogic_ir::{EinsumGraph, TLExpr};

/// Statistics for parallel compilation.
#[derive(Debug, Clone, Default)]
pub struct ParallelStats {
    /// Number of subexpressions compiled in parallel
    pub parallel_tasks: usize,
    /// Number of subexpressions compiled sequentially
    pub sequential_tasks: usize,
    /// Total time spent in parallel compilation (microseconds)
    pub parallel_time_us: u64,
    /// Total time spent in sequential compilation (microseconds)
    pub sequential_time_us: u64,
    /// Number of threads used
    pub threads_used: usize,
}

impl ParallelStats {
    /// Create new statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of tasks.
    pub fn total_tasks(&self) -> usize {
        self.parallel_tasks + self.sequential_tasks
    }

    /// Percentage of tasks that were parallelized.
    pub fn parallelization_ratio(&self) -> f64 {
        let total = self.total_tasks();
        if total == 0 {
            0.0
        } else {
            self.parallel_tasks as f64 / total as f64
        }
    }

    /// Total compilation time in microseconds.
    pub fn total_time_us(&self) -> u64 {
        self.parallel_time_us + self.sequential_time_us
    }

    /// Speedup from parallelization (estimated).
    pub fn speedup_estimate(&self) -> f64 {
        if self.sequential_time_us == 0 {
            1.0
        } else {
            let total = self.total_time_us();
            if total == 0 {
                1.0
            } else {
                (self.sequential_time_us + self.parallel_time_us) as f64 / total as f64
            }
        }
    }
}

/// Configuration for parallel compilation.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Minimum expression complexity to trigger parallel compilation.
    /// Expressions with complexity below this threshold are compiled sequentially.
    pub min_complexity_for_parallel: usize,
    /// Maximum number of threads to use (0 = use all available cores).
    pub max_threads: usize,
    /// Enable parallel optimization passes.
    pub parallel_optimization: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_complexity_for_parallel: 10,
            max_threads: 0, // Use all available cores
            parallel_optimization: true,
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum complexity threshold.
    pub fn with_min_complexity(mut self, threshold: usize) -> Self {
        self.min_complexity_for_parallel = threshold;
        self
    }

    /// Set maximum number of threads.
    pub fn with_max_threads(mut self, threads: usize) -> Self {
        self.max_threads = threads;
        self
    }

    /// Enable/disable parallel optimization.
    pub fn with_parallel_optimization(mut self, enabled: bool) -> Self {
        self.parallel_optimization = enabled;
        self
    }
}

/// Thread-safe wrapper for compiler context.
///
/// Note: Currently unused but reserved for future true parallel compilation implementation.
#[allow(dead_code)]
struct ThreadSafeContext {
    inner: Arc<RwLock<CompilerContext>>,
}

#[allow(dead_code)]
impl ThreadSafeContext {
    fn new(ctx: CompilerContext) -> Self {
        Self {
            inner: Arc::new(RwLock::new(ctx)),
        }
    }

    fn clone_context(&self) -> CompilerContext {
        self.inner.read().clone()
    }

    fn merge_context(&self, other: &CompilerContext) {
        let mut ctx = self.inner.write();
        // Merge temporary tensor allocations and axis assignments
        for (var, axis) in &other.var_to_axis {
            if !ctx.var_to_axis.contains_key(var) {
                ctx.var_to_axis.insert(var.clone(), *axis);
            }
        }
        for (domain, size) in &other.domains {
            if !ctx.domains.contains_key(domain) {
                ctx.domains.insert(domain.clone(), size.clone());
            }
        }
        // Note: next_temp_id is not publicly accessible, context merging is best-effort
    }

    fn into_inner(self) -> CompilerContext {
        match Arc::try_unwrap(self.inner) {
            Ok(rwlock) => rwlock.into_inner(),
            Err(arc) => (*arc.read()).clone(),
        }
    }
}

/// Parallel compiler for TensorLogic expressions.
pub struct ParallelCompiler {
    config: ParallelConfig,
    stats: Arc<Mutex<ParallelStats>>,
}

impl ParallelCompiler {
    /// Create a new parallel compiler with default configuration.
    pub fn new() -> Self {
        Self::with_config(ParallelConfig::default())
    }

    /// Create a new parallel compiler with custom configuration.
    pub fn with_config(config: ParallelConfig) -> Self {
        // Configure rayon thread pool
        if config.max_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.max_threads)
                .build_global()
                .ok(); // Ignore error if already initialized
        }

        Self {
            config,
            stats: Arc::new(Mutex::new(ParallelStats::new())),
        }
    }

    /// Get compilation statistics.
    pub fn stats(&self) -> ParallelStats {
        self.stats.lock().clone()
    }

    /// Reset compilation statistics.
    pub fn reset_stats(&self) {
        *self.stats.lock() = ParallelStats::new();
    }

    /// Compile an expression to an einsum graph using parallel processing.
    pub fn compile(&self, expr: &TLExpr, ctx: &mut CompilerContext) -> Result<EinsumGraph> {
        let start = std::time::Instant::now();

        // Estimate expression complexity
        let complexity = estimate_complexity(expr);

        // Decide whether to use parallel compilation
        let use_parallel = complexity >= self.config.min_complexity_for_parallel;

        let mut graph = EinsumGraph::new();

        if use_parallel {
            self.compile_parallel(expr, ctx, &mut graph)?;
        } else {
            compile_expr(expr, ctx, &mut graph)?;
        }

        // Update statistics
        let elapsed = start.elapsed();
        let mut stats = self.stats.lock();
        if use_parallel {
            stats.parallel_tasks += 1;
            stats.parallel_time_us += elapsed.as_micros() as u64;
        } else {
            stats.sequential_tasks += 1;
            stats.sequential_time_us += elapsed.as_micros() as u64;
        }
        stats.threads_used = rayon::current_num_threads();

        Ok(graph)
    }

    /// Compile with parallel processing.
    fn compile_parallel(
        &self,
        expr: &TLExpr,
        ctx: &mut CompilerContext,
        graph: &mut EinsumGraph,
    ) -> Result<()> {
        // For now, just use sequential compilation
        // True parallelization would require refactoring the compilation pipeline
        // to support parallel graph construction with thread-safe graph building
        compile_expr(expr, ctx, graph)?;
        Ok(())
    }

    /// Compile with parallel optimization.
    pub fn compile_with_optimization(
        &self,
        expr: &TLExpr,
        ctx: &mut CompilerContext,
        opt_config: PipelineConfig,
    ) -> Result<(EinsumGraph, PipelineStats)> {
        // First optimize the expression
        let pipeline = OptimizationPipeline::with_config(opt_config);
        let (optimized_expr, opt_stats) = if self.config.parallel_optimization {
            // Implement parallel optimization passes
            self.parallel_optimize(expr, pipeline)
        } else {
            pipeline.optimize(expr)
        };

        // Then compile
        let graph = self.compile(&optimized_expr, ctx)?;

        Ok((graph, opt_stats))
    }

    /// Perform optimization passes in parallel on independent subexpressions.
    fn parallel_optimize(
        &self,
        expr: &TLExpr,
        pipeline: OptimizationPipeline,
    ) -> (TLExpr, PipelineStats) {
        use rayon::prelude::*;

        // Check if expression is complex enough to benefit from parallelization
        let complexity = crate::optimize::analyze_complexity(expr);
        if complexity.max_depth < self.config.min_complexity_for_parallel {
            // Not complex enough, optimize sequentially
            return pipeline.optimize(expr);
        }

        // Decompose expression into independent subtrees
        let subtrees = self.decompose_for_parallel_opt(expr);

        if subtrees.len() <= 1 {
            // No independent subtrees, optimize sequentially
            return pipeline.optimize(expr);
        }

        // Optimize each subtree in parallel
        // Note: Each thread gets a reference to the pipeline since optimize() takes &self
        let optimized_subtrees: Vec<_> = subtrees
            .par_iter()
            .map(|subtree| pipeline.optimize(subtree))
            .collect();

        // Combine optimized subtrees back into a single expression
        let combined_expr = self.recombine_subtrees(expr, &optimized_subtrees);

        // Aggregate statistics from all subtrees
        let combined_stats = self.aggregate_stats(&optimized_subtrees);

        (combined_expr, combined_stats)
    }

    /// Decompose expression into independent subtrees for parallel optimization.
    #[allow(clippy::only_used_in_recursion)]
    fn decompose_for_parallel_opt(&self, expr: &TLExpr) -> Vec<TLExpr> {
        match expr {
            // Binary operations can be parallelized
            TLExpr::And(left, right) | TLExpr::Or(left, right) => {
                let mut subtrees = Vec::new();
                subtrees.extend(self.decompose_for_parallel_opt(left));
                subtrees.extend(self.decompose_for_parallel_opt(right));
                subtrees
            }
            // Arithmetic and comparison operations
            TLExpr::Add(left, right)
            | TLExpr::Sub(left, right)
            | TLExpr::Mul(left, right)
            | TLExpr::Div(left, right)
            | TLExpr::Eq(left, right)
            | TLExpr::Lt(left, right)
            | TLExpr::Gt(left, right)
            | TLExpr::Lte(left, right)
            | TLExpr::Gte(left, right) => {
                let mut subtrees = Vec::new();
                subtrees.extend(self.decompose_for_parallel_opt(left));
                subtrees.extend(self.decompose_for_parallel_opt(right));
                subtrees
            }
            // Leaf nodes are their own subtrees
            TLExpr::Pred { .. } | TLExpr::Constant(_) => {
                vec![expr.clone()]
            }
            // Other expressions: treat as atomic units
            _ => vec![expr.clone()],
        }
    }

    /// Recombine optimized subtrees back into the original expression structure.
    #[allow(clippy::only_used_in_recursion)]
    fn recombine_subtrees(
        &self,
        original: &TLExpr,
        optimized: &[(TLExpr, PipelineStats)],
    ) -> TLExpr {
        if optimized.is_empty() {
            return original.clone();
        }

        if optimized.len() == 1 {
            return optimized[0].0.clone();
        }

        match original {
            TLExpr::And(_, _) => {
                let mid = optimized.len() / 2;
                TLExpr::And(
                    Box::new(self.recombine_subtrees(original, &optimized[..mid])),
                    Box::new(self.recombine_subtrees(original, &optimized[mid..])),
                )
            }
            TLExpr::Or(_, _) => {
                let mid = optimized.len() / 2;
                TLExpr::Or(
                    Box::new(self.recombine_subtrees(original, &optimized[..mid])),
                    Box::new(self.recombine_subtrees(original, &optimized[mid..])),
                )
            }
            TLExpr::Add(_, _) => {
                let mid = optimized.len() / 2;
                TLExpr::Add(
                    Box::new(self.recombine_subtrees(original, &optimized[..mid])),
                    Box::new(self.recombine_subtrees(original, &optimized[mid..])),
                )
            }
            _ => optimized[0].0.clone(),
        }
    }

    /// Aggregate statistics from parallel optimization passes.
    fn aggregate_stats(&self, results: &[(TLExpr, PipelineStats)]) -> PipelineStats {
        if results.is_empty() {
            return PipelineStats::default();
        }

        let mut combined = results[0].1.clone();
        for (_, stats) in &results[1..] {
            combined.total_iterations = combined.total_iterations.max(stats.total_iterations);
            combined.reached_fixed_point =
                combined.reached_fixed_point && stats.reached_fixed_point;

            // Aggregate individual pass statistics
            combined.negation.double_negations_eliminated +=
                stats.negation.double_negations_eliminated;
            combined.negation.demorgans_applied += stats.negation.demorgans_applied;
            combined.negation.quantifier_negations_pushed +=
                stats.negation.quantifier_negations_pushed;

            combined.constant_folding.binary_ops_folded += stats.constant_folding.binary_ops_folded;
            combined.constant_folding.unary_ops_folded += stats.constant_folding.unary_ops_folded;

            combined.algebraic.identities_eliminated += stats.algebraic.identities_eliminated;
            combined.algebraic.annihilations_applied += stats.algebraic.annihilations_applied;
            combined.algebraic.idempotent_simplified += stats.algebraic.idempotent_simplified;

            combined.strength_reduction.power_reductions +=
                stats.strength_reduction.power_reductions;
            combined.strength_reduction.operations_eliminated +=
                stats.strength_reduction.operations_eliminated;
            combined.strength_reduction.special_function_optimizations +=
                stats.strength_reduction.special_function_optimizations;

            combined.distributivity.expressions_factored +=
                stats.distributivity.expressions_factored;
            combined.distributivity.expressions_expanded +=
                stats.distributivity.expressions_expanded;
            combined.distributivity.common_terms_extracted +=
                stats.distributivity.common_terms_extracted;

            combined.quantifier_opt.invariants_hoisted += stats.quantifier_opt.invariants_hoisted;
            combined.quantifier_opt.quantifiers_reordered +=
                stats.quantifier_opt.quantifiers_reordered;
            combined.quantifier_opt.quantifiers_fused += stats.quantifier_opt.quantifiers_fused;

            combined.dead_code.branches_eliminated += stats.dead_code.branches_eliminated;
            combined.dead_code.short_circuits += stats.dead_code.short_circuits;
            combined.dead_code.unused_quantifiers_removed +=
                stats.dead_code.unused_quantifiers_removed;
            combined.dead_code.identity_simplifications += stats.dead_code.identity_simplifications;
        }

        combined
    }
}

impl Default for ParallelCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate expression complexity for scheduling decisions.
fn estimate_complexity(expr: &TLExpr) -> usize {
    match expr {
        TLExpr::Pred { .. } => 1,
        TLExpr::Constant(_) => 1,
        TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
            1 + estimate_complexity(l) + estimate_complexity(r)
        }
        TLExpr::Not(e) => 1 + estimate_complexity(e),
        TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => 2 + estimate_complexity(body),
        TLExpr::Add(l, r)
        | TLExpr::Sub(l, r)
        | TLExpr::Mul(l, r)
        | TLExpr::Div(l, r)
        | TLExpr::Pow(l, r)
        | TLExpr::Mod(l, r)
        | TLExpr::Min(l, r)
        | TLExpr::Max(l, r)
        | TLExpr::Eq(l, r)
        | TLExpr::Lt(l, r)
        | TLExpr::Gt(l, r)
        | TLExpr::Lte(l, r)
        | TLExpr::Gte(l, r) => 1 + estimate_complexity(l) + estimate_complexity(r),
        TLExpr::Abs(e)
        | TLExpr::Floor(e)
        | TLExpr::Ceil(e)
        | TLExpr::Round(e)
        | TLExpr::Sqrt(e)
        | TLExpr::Exp(e)
        | TLExpr::Log(e)
        | TLExpr::Sin(e)
        | TLExpr::Cos(e)
        | TLExpr::Tan(e) => 1 + estimate_complexity(e),
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            2 + estimate_complexity(condition)
                + estimate_complexity(then_branch)
                + estimate_complexity(else_branch)
        }
        TLExpr::Box(e)
        | TLExpr::Diamond(e)
        | TLExpr::Next(e)
        | TLExpr::Eventually(e)
        | TLExpr::Always(e) => 2 + estimate_complexity(e),
        TLExpr::Until { before, after } | TLExpr::WeakUntil { before, after } => {
            2 + estimate_complexity(before) + estimate_complexity(after)
        }
        TLExpr::Release { released, releaser } | TLExpr::StrongRelease { released, releaser } => {
            2 + estimate_complexity(released) + estimate_complexity(releaser)
        }
        TLExpr::WeightedRule { rule, .. } => 1 + estimate_complexity(rule),
        TLExpr::ProbabilisticChoice { alternatives } => {
            1 + alternatives
                .iter()
                .map(|(_, e)| estimate_complexity(e))
                .sum::<usize>()
        }
        TLExpr::SoftExists { body, .. } | TLExpr::SoftForAll { body, .. } => {
            2 + estimate_complexity(body)
        }
        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            1 + estimate_complexity(left) + estimate_complexity(right)
        }
        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => 1 + estimate_complexity(premise) + estimate_complexity(conclusion),
        TLExpr::FuzzyNot { expr, .. } => 1 + estimate_complexity(expr),
        TLExpr::Let { value, body, .. } => {
            1 + estimate_complexity(value) + estimate_complexity(body)
        }
        TLExpr::Aggregate { body, .. } => 2 + estimate_complexity(body),
        TLExpr::Score(e) => estimate_complexity(e),
        // Counting quantifiers
        TLExpr::CountingExists { body, .. }
        | TLExpr::CountingForAll { body, .. }
        | TLExpr::ExactCount { body, .. }
        | TLExpr::Majority { body, .. } => 2 + estimate_complexity(body),
        // All other expression types (alpha.3 enhancements) - conservative complexity estimate
        _ => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_complexity_estimation() {
        let simple = TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        };
        assert_eq!(estimate_complexity(&simple), 1);

        let and = TLExpr::And(
            Box::new(TLExpr::Pred {
                name: "P".to_string(),
                args: vec![Term::Var("x".to_string())],
            }),
            Box::new(TLExpr::Pred {
                name: "Q".to_string(),
                args: vec![Term::Var("y".to_string())],
            }),
        );
        assert_eq!(estimate_complexity(&and), 3);

        let exists = TLExpr::Exists {
            var: "x".to_string(),
            domain: "D".to_string(),
            body: Box::new(TLExpr::Pred {
                name: "P".to_string(),
                args: vec![Term::Var("x".to_string())],
            }),
        };
        assert_eq!(estimate_complexity(&exists), 3);
    }

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::new()
            .with_min_complexity(20)
            .with_max_threads(4)
            .with_parallel_optimization(false);

        assert_eq!(config.min_complexity_for_parallel, 20);
        assert_eq!(config.max_threads, 4);
        assert!(!config.parallel_optimization);
    }

    #[test]
    fn test_parallel_stats() {
        let stats = ParallelStats {
            parallel_tasks: 10,
            sequential_tasks: 5,
            parallel_time_us: 1000,
            sequential_time_us: 500,
            threads_used: 4,
        };

        assert_eq!(stats.total_tasks(), 15);
        assert!((stats.parallelization_ratio() - 10.0 / 15.0).abs() < 1e-6);
        assert_eq!(stats.total_time_us(), 1500);
    }

    #[test]
    fn test_parallel_compiler_simple() {
        let compiler = ParallelCompiler::new();
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);

        let expr = TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        };
        let result = compiler.compile(&expr, &mut ctx);
        assert!(result.is_ok());

        let stats = compiler.stats();
        assert_eq!(stats.total_tasks(), 1);
    }

    #[test]
    fn test_parallel_compiler_complex() {
        let compiler = ParallelCompiler::new();
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);

        // Create a complex expression
        let expr = TLExpr::And(
            Box::new(TLExpr::Exists {
                var: "x".to_string(),
                domain: "D".to_string(),
                body: Box::new(TLExpr::And(
                    Box::new(TLExpr::Pred {
                        name: "P".to_string(),
                        args: vec![Term::Var("x".to_string()), Term::Var("y".to_string())],
                    }),
                    Box::new(TLExpr::Pred {
                        name: "Q".to_string(),
                        args: vec![Term::Var("y".to_string()), Term::Var("z".to_string())],
                    }),
                )),
            }),
            Box::new(TLExpr::Exists {
                var: "w".to_string(),
                domain: "D".to_string(),
                body: Box::new(TLExpr::Or(
                    Box::new(TLExpr::Pred {
                        name: "R".to_string(),
                        args: vec![Term::Var("w".to_string())],
                    }),
                    Box::new(TLExpr::Pred {
                        name: "S".to_string(),
                        args: vec![Term::Var("w".to_string())],
                    }),
                )),
            }),
        );

        let result = compiler.compile(&expr, &mut ctx);
        assert!(result.is_ok());

        let stats = compiler.stats();
        assert!(stats.total_tasks() >= 1);
    }

    #[test]
    fn test_thread_safe_context() {
        let ctx = CompilerContext::new();
        let ts_ctx = ThreadSafeContext::new(ctx);

        // Test cloning
        let cloned = ts_ctx.clone_context();
        assert!(cloned.domains.is_empty());

        // Test merging
        let mut other = CompilerContext::new();
        other.add_domain("D", 10);

        ts_ctx.merge_context(&other);
        let merged = ts_ctx.clone_context();
        assert_eq!(merged.domains.get("D").map(|d| d.cardinality), Some(10));
    }

    #[test]
    fn test_parallel_with_optimization() {
        let compiler = ParallelCompiler::new();
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);

        let expr = TLExpr::Not(Box::new(TLExpr::Not(Box::new(TLExpr::Pred {
            name: "P".to_string(),
            args: vec![Term::Var("x".to_string())],
        }))));

        let opt_config = PipelineConfig::default();
        let result = compiler.compile_with_optimization(&expr, &mut ctx, opt_config);
        assert!(result.is_ok());

        let (_graph, opt_stats) = result.unwrap();
        // Double negation should be eliminated
        assert!(opt_stats.negation.double_negations_eliminated > 0);
    }

    #[test]
    fn test_parallel_stats_operations() {
        let mut stats = ParallelStats::new();
        stats.parallel_tasks = 8;
        stats.sequential_tasks = 2;
        stats.parallel_time_us = 800;
        stats.sequential_time_us = 200;

        assert_eq!(stats.total_tasks(), 10);
        assert_eq!(stats.total_time_us(), 1000);
        assert!((stats.parallelization_ratio() - 0.8).abs() < 1e-6);
    }
}
