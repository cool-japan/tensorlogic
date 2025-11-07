//! Optimization pipeline orchestration for expressions.
//!
//! This module provides a high-level optimization pipeline that automatically
//! orders and applies multiple optimization passes to expressions, tracking
//! metrics and ensuring convergence.
//!
//! # Architecture
//!
//! The pipeline consists of:
//! - **Optimization passes**: Individual transformation functions
//! - **Pass ordering**: Automatic determination of pass order
//! - **Convergence detection**: Stopping when no more changes occur
//! - **Metrics tracking**: Recording improvements and performance
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_ir::{TLExpr, Term, OptimizationPipeline, OptimizationLevel};
//!
//! let expr = TLExpr::and(
//!     TLExpr::constant(1.0),
//!     TLExpr::pred("P", vec![Term::var("x")])
//! );
//!
//! let pipeline = OptimizationPipeline::default();
//! let (optimized, metrics) = pipeline.optimize(expr);
//! println!("Applied {} passes", metrics.passes_applied);
//! ```

use std::collections::HashMap;

use super::{
    distributive_laws::{apply_distributive_laws, DistributiveStrategy},
    modal_equivalences::apply_modal_equivalences,
    normal_forms::to_nnf,
    optimization::{algebraic_simplify, constant_fold, propagate_constants},
    temporal_equivalences::apply_temporal_equivalences,
    TLExpr,
};

/// Optimization level controlling aggressiveness of optimizations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum OptimizationLevel {
    /// No optimizations (O0)
    None,
    /// Basic optimizations (O1): constant folding, simple simplifications
    Basic,
    /// Standard optimizations (O2): includes algebraic laws, normal forms
    #[default]
    Standard,
    /// Aggressive optimizations (O3): all transformations, multiple passes
    Aggressive,
}

/// A single optimization pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationPass {
    /// Constant folding
    ConstantFolding,
    /// Constant propagation
    ConstantPropagation,
    /// Algebraic simplification
    AlgebraicSimplification,
    /// Convert to negation normal form
    NegationNormalForm,
    /// Apply modal logic equivalences
    ModalEquivalences,
    /// Apply temporal logic equivalences
    TemporalEquivalences,
    /// Apply distributive laws (AND over OR)
    DistributiveAndOverOr,
    /// Apply distributive laws (OR over AND)
    DistributiveOrOverAnd,
    /// Apply distributive laws for quantifiers
    DistributiveQuantifiers,
    /// Apply distributive laws for modal operators
    DistributiveModal,
}

impl OptimizationPass {
    /// Get the name of this pass.
    pub fn name(&self) -> &'static str {
        match self {
            OptimizationPass::ConstantFolding => "constant_folding",
            OptimizationPass::ConstantPropagation => "constant_propagation",
            OptimizationPass::AlgebraicSimplification => "algebraic_simplification",
            OptimizationPass::NegationNormalForm => "negation_normal_form",
            OptimizationPass::ModalEquivalences => "modal_equivalences",
            OptimizationPass::TemporalEquivalences => "temporal_equivalences",
            OptimizationPass::DistributiveAndOverOr => "distributive_and_over_or",
            OptimizationPass::DistributiveOrOverAnd => "distributive_or_over_and",
            OptimizationPass::DistributiveQuantifiers => "distributive_quantifiers",
            OptimizationPass::DistributiveModal => "distributive_modal",
        }
    }

    /// Apply this pass to an expression.
    pub fn apply(&self, expr: TLExpr) -> TLExpr {
        match self {
            OptimizationPass::ConstantFolding => constant_fold(&expr),
            OptimizationPass::ConstantPropagation => propagate_constants(&expr),
            OptimizationPass::AlgebraicSimplification => algebraic_simplify(&expr),
            OptimizationPass::NegationNormalForm => to_nnf(&expr),
            OptimizationPass::ModalEquivalences => apply_modal_equivalences(&expr),
            OptimizationPass::TemporalEquivalences => apply_temporal_equivalences(&expr),
            OptimizationPass::DistributiveAndOverOr => {
                apply_distributive_laws(&expr, DistributiveStrategy::AndOverOr)
            }
            OptimizationPass::DistributiveOrOverAnd => {
                apply_distributive_laws(&expr, DistributiveStrategy::OrOverAnd)
            }
            OptimizationPass::DistributiveQuantifiers => {
                apply_distributive_laws(&expr, DistributiveStrategy::Quantifiers)
            }
            OptimizationPass::DistributiveModal => {
                apply_distributive_laws(&expr, DistributiveStrategy::Modal)
            }
        }
    }

    /// Get the priority of this pass (lower = earlier in pipeline).
    pub fn priority(&self) -> u32 {
        match self {
            // Early passes: normalize and fold constants
            OptimizationPass::ConstantFolding => 10,
            OptimizationPass::ConstantPropagation => 20,
            OptimizationPass::NegationNormalForm => 30,
            // Middle passes: apply equivalences and simplifications
            OptimizationPass::AlgebraicSimplification => 40,
            OptimizationPass::ModalEquivalences => 50,
            OptimizationPass::TemporalEquivalences => 60,
            // Late passes: distributive laws (can expand expressions)
            OptimizationPass::DistributiveQuantifiers => 70,
            OptimizationPass::DistributiveModal => 80,
            OptimizationPass::DistributiveAndOverOr => 90,
            OptimizationPass::DistributiveOrOverAnd => 100,
        }
    }

    /// Get all passes for a given optimization level.
    pub fn for_level(level: OptimizationLevel) -> Vec<OptimizationPass> {
        match level {
            OptimizationLevel::None => vec![],
            OptimizationLevel::Basic => vec![
                OptimizationPass::ConstantFolding,
                OptimizationPass::ConstantPropagation,
                OptimizationPass::AlgebraicSimplification,
            ],
            OptimizationLevel::Standard => vec![
                OptimizationPass::ConstantFolding,
                OptimizationPass::ConstantPropagation,
                OptimizationPass::NegationNormalForm,
                OptimizationPass::AlgebraicSimplification,
                OptimizationPass::ModalEquivalences,
                OptimizationPass::TemporalEquivalences,
            ],
            OptimizationLevel::Aggressive => vec![
                OptimizationPass::ConstantFolding,
                OptimizationPass::ConstantPropagation,
                OptimizationPass::NegationNormalForm,
                OptimizationPass::AlgebraicSimplification,
                OptimizationPass::ModalEquivalences,
                OptimizationPass::TemporalEquivalences,
                OptimizationPass::DistributiveQuantifiers,
                OptimizationPass::DistributiveModal,
                OptimizationPass::DistributiveAndOverOr,
            ],
        }
    }
}

/// Metrics collected during optimization.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OptimizationMetrics {
    /// Number of passes applied
    pub passes_applied: usize,
    /// Number of iterations until convergence
    pub iterations: usize,
    /// Whether the pipeline converged
    pub converged: bool,
    /// Per-pass application counts
    pub pass_counts: HashMap<String, usize>,
    /// Initial expression size (node count)
    pub initial_size: usize,
    /// Final expression size (node count)
    pub final_size: usize,
    /// Size reduction ratio
    pub reduction_ratio: f64,
}

impl OptimizationMetrics {
    /// Create new empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a pass was applied.
    pub fn record_pass(&mut self, pass: OptimizationPass) {
        self.passes_applied += 1;
        *self.pass_counts.entry(pass.name().to_string()).or_insert(0) += 1;
    }

    /// Compute final metrics.
    pub fn finalize(&mut self, initial_size: usize, final_size: usize) {
        self.initial_size = initial_size;
        self.final_size = final_size;
        self.reduction_ratio = if initial_size > 0 {
            1.0 - (final_size as f64 / initial_size as f64)
        } else {
            0.0
        };
    }
}

/// Configuration for the optimization pipeline.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Optimization level
    pub level: OptimizationLevel,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Custom pass ordering (if None, uses default ordering by priority)
    pub custom_passes: Option<Vec<OptimizationPass>>,
    /// Whether to enable convergence detection
    pub enable_convergence: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::Standard,
            max_iterations: 10,
            custom_passes: None,
            enable_convergence: true,
        }
    }
}

impl PipelineConfig {
    /// Create a new configuration with the given optimization level.
    pub fn with_level(level: OptimizationLevel) -> Self {
        Self {
            level,
            ..Default::default()
        }
    }

    /// Set custom passes.
    pub fn with_custom_passes(mut self, passes: Vec<OptimizationPass>) -> Self {
        self.custom_passes = Some(passes);
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Disable convergence detection.
    pub fn without_convergence(mut self) -> Self {
        self.enable_convergence = false;
        self
    }
}

/// Optimization pipeline that orchestrates multiple passes.
#[derive(Default)]
pub struct OptimizationPipeline {
    config: PipelineConfig,
}

impl OptimizationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Create a pipeline with a specific optimization level.
    pub fn with_level(level: OptimizationLevel) -> Self {
        Self::new(PipelineConfig::with_level(level))
    }

    /// Optimize an expression using the configured pipeline.
    ///
    /// Returns the optimized expression and metrics about the optimization process.
    pub fn optimize(&self, expr: TLExpr) -> (TLExpr, OptimizationMetrics) {
        let mut current = expr;
        let mut metrics = OptimizationMetrics::new();
        let initial_size = count_nodes(&current);

        // Get passes to apply (either custom or default for level)
        let passes = self
            .config
            .custom_passes
            .clone()
            .unwrap_or_else(|| OptimizationPass::for_level(self.config.level));

        // Sort passes by priority
        let mut sorted_passes = passes.clone();
        sorted_passes.sort_by_key(|p| p.priority());

        // Apply passes iteratively until convergence or max iterations
        for iteration in 0..self.config.max_iterations {
            metrics.iterations = iteration + 1;
            let previous = current.clone();

            // Apply each pass in order
            for pass in &sorted_passes {
                let before = current.clone();
                current = pass.apply(current);

                // Record if the pass made changes
                if before != current {
                    metrics.record_pass(*pass);
                }
            }

            // Check for convergence
            if self.config.enable_convergence && current == previous {
                metrics.converged = true;
                break;
            }
        }

        let final_size = count_nodes(&current);
        metrics.finalize(initial_size, final_size);

        (current, metrics)
    }

    /// Apply a single pass to an expression.
    pub fn apply_pass(&self, expr: TLExpr, pass: OptimizationPass) -> TLExpr {
        pass.apply(expr)
    }

    /// Get the configuration of this pipeline.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

/// Count the number of nodes in an expression (for metrics).
fn count_nodes(expr: &TLExpr) -> usize {
    match expr {
        TLExpr::Pred { .. } | TLExpr::Constant(_) => 1,
        TLExpr::And(l, r)
        | TLExpr::Or(l, r)
        | TLExpr::Imply(l, r)
        | TLExpr::Add(l, r)
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
        | TLExpr::Gte(l, r) => 1 + count_nodes(l) + count_nodes(r),
        TLExpr::Not(e)
        | TLExpr::Score(e)
        | TLExpr::Abs(e)
        | TLExpr::Floor(e)
        | TLExpr::Ceil(e)
        | TLExpr::Round(e)
        | TLExpr::Sqrt(e)
        | TLExpr::Exp(e)
        | TLExpr::Log(e)
        | TLExpr::Sin(e)
        | TLExpr::Cos(e)
        | TLExpr::Tan(e)
        | TLExpr::Box(e)
        | TLExpr::Diamond(e)
        | TLExpr::Next(e)
        | TLExpr::Eventually(e)
        | TLExpr::Always(e) => 1 + count_nodes(e),
        TLExpr::Until { before, after }
        | TLExpr::Release {
            released: before,
            releaser: after,
        }
        | TLExpr::WeakUntil { before, after }
        | TLExpr::StrongRelease {
            released: before,
            releaser: after,
        } => 1 + count_nodes(before) + count_nodes(after),
        TLExpr::Exists { body, .. }
        | TLExpr::ForAll { body, .. }
        | TLExpr::SoftExists { body, .. }
        | TLExpr::SoftForAll { body, .. }
        | TLExpr::Aggregate { body, .. }
        | TLExpr::WeightedRule { rule: body, .. }
        | TLExpr::FuzzyNot { expr: body, .. } => 1 + count_nodes(body),
        TLExpr::TNorm { left, right, .. }
        | TLExpr::TCoNorm { left, right, .. }
        | TLExpr::FuzzyImplication {
            premise: left,
            conclusion: right,
            ..
        } => 1 + count_nodes(left) + count_nodes(right),
        TLExpr::ProbabilisticChoice { alternatives } => {
            1 + alternatives
                .iter()
                .map(|(_, e)| count_nodes(e))
                .sum::<usize>()
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => 1 + count_nodes(condition) + count_nodes(then_branch) + count_nodes(else_branch),
        TLExpr::Let { value, body, .. } => 1 + count_nodes(value) + count_nodes(body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_optimization_level_ordering() {
        assert!(OptimizationLevel::None < OptimizationLevel::Basic);
        assert!(OptimizationLevel::Basic < OptimizationLevel::Standard);
        assert!(OptimizationLevel::Standard < OptimizationLevel::Aggressive);
    }

    #[test]
    fn test_pass_priority_ordering() {
        let passes = OptimizationPass::for_level(OptimizationLevel::Aggressive);
        let priorities: Vec<u32> = passes.iter().map(|p| p.priority()).collect();

        // Verify that constant folding comes first
        assert_eq!(passes[0], OptimizationPass::ConstantFolding);
        assert_eq!(priorities[0], 10);
    }

    #[test]
    fn test_pipeline_basic_optimization() {
        // (1.0 AND P(x)) should simplify to P(x)
        let expr = TLExpr::and(
            TLExpr::constant(1.0),
            TLExpr::pred("P", vec![Term::var("x")]),
        );

        let pipeline = OptimizationPipeline::with_level(OptimizationLevel::Basic);
        let (_optimized, metrics) = pipeline.optimize(expr);

        // Should have simplified
        assert!(metrics.passes_applied > 0);
        assert!(metrics.reduction_ratio > 0.0);
        assert!(metrics.converged);
    }

    #[test]
    fn test_pipeline_no_optimization() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        let pipeline = OptimizationPipeline::with_level(OptimizationLevel::None);
        let (optimized, metrics) = pipeline.optimize(expr.clone());

        // Should not change
        assert_eq!(optimized, expr);
        assert_eq!(metrics.passes_applied, 0);
    }

    #[test]
    fn test_pipeline_convergence() {
        // Expression that requires multiple passes
        let expr = TLExpr::and(
            TLExpr::or(TLExpr::constant(1.0), TLExpr::constant(0.0)),
            TLExpr::pred("P", vec![Term::var("x")]),
        );

        let pipeline = OptimizationPipeline::with_level(OptimizationLevel::Standard);
        let (_, metrics) = pipeline.optimize(expr);

        // Should converge
        assert!(metrics.converged);
        assert!(metrics.iterations > 0);
    }

    #[test]
    fn test_pipeline_max_iterations() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        let config = PipelineConfig::default().with_max_iterations(5);
        let pipeline = OptimizationPipeline::new(config);
        let (_, metrics) = pipeline.optimize(expr);

        // Should not exceed max iterations
        assert!(metrics.iterations <= 5);
    }

    #[test]
    fn test_custom_passes() {
        let expr = TLExpr::constant(42.0);

        let custom_passes = vec![
            OptimizationPass::ConstantFolding,
            OptimizationPass::AlgebraicSimplification,
        ];

        let config = PipelineConfig::default().with_custom_passes(custom_passes);
        let pipeline = OptimizationPipeline::new(config);
        let (_, metrics) = pipeline.optimize(expr);

        // Verify only specified passes were used
        assert!(metrics.pass_counts.len() <= 2);
    }

    #[test]
    fn test_metrics_tracking() {
        let expr = TLExpr::and(
            TLExpr::constant(1.0),
            TLExpr::pred("P", vec![Term::var("x")]),
        );

        let pipeline = OptimizationPipeline::with_level(OptimizationLevel::Standard);
        let (_, metrics) = pipeline.optimize(expr);

        assert!(metrics.initial_size > metrics.final_size);
        assert!(metrics.reduction_ratio > 0.0);
        assert!(metrics.reduction_ratio <= 1.0);
    }

    #[test]
    fn test_count_nodes_simple() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        assert_eq!(count_nodes(&expr), 1);
    }

    #[test]
    fn test_count_nodes_complex() {
        let expr = TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::or(
                TLExpr::pred("Q", vec![Term::var("y")]),
                TLExpr::pred("R", vec![Term::var("z")]),
            ),
        );
        // 1 (AND) + 1 (P) + 1 (OR) + 1 (Q) + 1 (R) = 5
        assert_eq!(count_nodes(&expr), 5);
    }

    #[test]
    fn test_pipeline_aggressive_level() {
        let expr = TLExpr::and(
            TLExpr::or(
                TLExpr::pred("P", vec![Term::var("x")]),
                TLExpr::pred("Q", vec![Term::var("x")]),
            ),
            TLExpr::pred("R", vec![Term::var("x")]),
        );

        let pipeline = OptimizationPipeline::with_level(OptimizationLevel::Aggressive);
        let (_, metrics) = pipeline.optimize(expr);

        // Aggressive level should apply more passes
        assert!(metrics.passes_applied > 0);
    }

    #[test]
    fn test_pass_application() {
        let expr = TLExpr::constant(1.0);
        let pipeline = OptimizationPipeline::default();

        let result = pipeline.apply_pass(expr.clone(), OptimizationPass::ConstantFolding);
        assert_eq!(result, expr); // Constants don't change
    }
}
