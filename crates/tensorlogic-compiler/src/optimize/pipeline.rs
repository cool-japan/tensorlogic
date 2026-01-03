//! Multi-pass optimization pipeline for TLExpr expressions.
//!
//! This module provides a unified optimization pipeline that combines multiple
//! optimization passes and applies them iteratively until a fixed point is reached.
//!
//! # Architecture
//!
//! The pipeline applies 7 optimization passes in this order:
//! 1. **Negation optimization**: Push negations inward using De Morgan's laws
//! 2. **Constant folding**: Evaluate constant expressions at compile time
//! 3. **Algebraic simplification**: Apply mathematical identities (x+0=x, x*1=x, etc.)
//! 4. **Strength reduction**: Replace expensive operations with cheaper equivalents (x^2→x*x)
//! 5. **Distributivity**: Factor common subexpressions (a*b + a*c → a*(b+c))
//! 6. **Quantifier optimization**: Loop-invariant code motion (∃x.(a+p(x)) → a + ∃x.p(x))
//! 7. **Dead code elimination**: Remove unreachable code and constant branches
//!
//! This order is chosen because:
//! - Negation optimization can expose more opportunities for other passes
//! - Constant folding creates simpler expressions for subsequent passes
//! - Algebraic simplification can create new constants and identity patterns
//! - Strength reduction makes operations more efficient
//! - Distributivity reduces redundant computation
//! - Quantifier optimization hoists loop-invariant code
//! - Dead code elimination removes unreachable branches created by earlier passes
//!
//! # Examples
//!
//! ```
//! use tensorlogic_compiler::optimize::{OptimizationPipeline, PipelineConfig};
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! // Create a pipeline with default configuration
//! let pipeline = OptimizationPipeline::new();
//!
//! // Optimize an expression: NOT(AND(x + 0, 2.0 * 3.0))
//! let x = TLExpr::pred("x", vec![Term::var("i")]);
//! let expr = TLExpr::negate(TLExpr::and(
//!     TLExpr::add(x, TLExpr::Constant(0.0)),
//!     TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
//! ));
//!
//! let (optimized, stats) = pipeline.optimize(&expr);
//!
//! // Pipeline applies multiple passes and reports statistics
//! assert!(stats.total_iterations > 0);
//! assert!(stats.constant_folding.binary_ops_folded > 0);
//! assert!(stats.algebraic.identities_eliminated > 0);
//! ```

use super::{
    algebraic::{simplify_algebraic, AlgebraicSimplificationStats},
    constant_folding::{fold_constants, ConstantFoldingStats},
    dead_code::{eliminate_dead_code, DeadCodeStats},
    distributivity::{optimize_distributivity, DistributivityStats},
    negation::{optimize_negations, NegationOptStats},
    quantifier_opt::{optimize_quantifiers, QuantifierOptStats},
    strength_reduction::{reduce_strength, StrengthReductionStats},
};
use tensorlogic_ir::TLExpr;

/// Configuration for the optimization pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable negation optimization pass
    pub enable_negation_opt: bool,
    /// Enable constant folding pass
    pub enable_constant_folding: bool,
    /// Enable algebraic simplification pass
    pub enable_algebraic_simplification: bool,
    /// Enable strength reduction pass
    pub enable_strength_reduction: bool,
    /// Enable distributivity optimization pass
    pub enable_distributivity: bool,
    /// Enable quantifier optimization pass
    pub enable_quantifier_opt: bool,
    /// Enable dead code elimination pass
    pub enable_dead_code_elimination: bool,
    /// Maximum number of iterations before stopping
    pub max_iterations: usize,
    /// Stop early if an iteration makes no changes
    pub stop_on_fixed_point: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_negation_opt: true,
            enable_constant_folding: true,
            enable_algebraic_simplification: true,
            enable_strength_reduction: true,
            enable_distributivity: true,
            enable_quantifier_opt: true,
            enable_dead_code_elimination: true,
            max_iterations: 10,
            stop_on_fixed_point: true,
        }
    }
}

impl PipelineConfig {
    /// Create a configuration with all optimizations enabled.
    pub fn all() -> Self {
        Self::default()
    }

    /// Create a configuration with all optimizations disabled.
    pub fn none() -> Self {
        Self {
            enable_negation_opt: false,
            enable_constant_folding: false,
            enable_algebraic_simplification: false,
            enable_strength_reduction: false,
            enable_distributivity: false,
            enable_quantifier_opt: false,
            enable_dead_code_elimination: false,
            max_iterations: 1,
            stop_on_fixed_point: true,
        }
    }

    /// Create a configuration with only constant folding enabled.
    pub fn constant_folding_only() -> Self {
        Self {
            enable_negation_opt: false,
            enable_constant_folding: true,
            enable_algebraic_simplification: false,
            enable_strength_reduction: false,
            enable_distributivity: false,
            enable_quantifier_opt: false,
            enable_dead_code_elimination: false,
            max_iterations: 1,
            stop_on_fixed_point: true,
        }
    }

    /// Create a configuration with only algebraic simplification enabled.
    pub fn algebraic_only() -> Self {
        Self {
            enable_negation_opt: false,
            enable_constant_folding: false,
            enable_algebraic_simplification: true,
            enable_strength_reduction: false,
            enable_distributivity: false,
            enable_quantifier_opt: false,
            enable_dead_code_elimination: false,
            max_iterations: 1,
            stop_on_fixed_point: true,
        }
    }

    /// Create a configuration for aggressive optimization (more iterations).
    pub fn aggressive() -> Self {
        Self {
            enable_negation_opt: true,
            enable_constant_folding: true,
            enable_algebraic_simplification: true,
            enable_strength_reduction: true,
            enable_distributivity: true,
            enable_quantifier_opt: true,
            enable_dead_code_elimination: true,
            max_iterations: 20,
            stop_on_fixed_point: true,
        }
    }

    /// Builder method to enable/disable negation optimization.
    pub fn with_negation_opt(mut self, enable: bool) -> Self {
        self.enable_negation_opt = enable;
        self
    }

    /// Builder method to enable/disable constant folding.
    pub fn with_constant_folding(mut self, enable: bool) -> Self {
        self.enable_constant_folding = enable;
        self
    }

    /// Builder method to enable/disable algebraic simplification.
    pub fn with_algebraic_simplification(mut self, enable: bool) -> Self {
        self.enable_algebraic_simplification = enable;
        self
    }

    /// Builder method to set maximum iterations.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Builder method to enable/disable fixed-point detection.
    pub fn with_stop_on_fixed_point(mut self, stop: bool) -> Self {
        self.stop_on_fixed_point = stop;
        self
    }

    /// Builder method to enable/disable strength reduction.
    pub fn with_strength_reduction(mut self, enable: bool) -> Self {
        self.enable_strength_reduction = enable;
        self
    }

    /// Builder method to enable/disable distributivity optimization.
    pub fn with_distributivity(mut self, enable: bool) -> Self {
        self.enable_distributivity = enable;
        self
    }

    /// Builder method to enable/disable quantifier optimization.
    pub fn with_quantifier_opt(mut self, enable: bool) -> Self {
        self.enable_quantifier_opt = enable;
        self
    }

    /// Builder method to enable/disable dead code elimination.
    pub fn with_dead_code_elimination(mut self, enable: bool) -> Self {
        self.enable_dead_code_elimination = enable;
        self
    }
}

/// Statistics from a single pipeline iteration.
#[derive(Debug, Clone, Default)]
pub struct IterationStats {
    /// Negation optimization statistics
    pub negation: NegationOptStats,
    /// Constant folding statistics
    pub constant_folding: ConstantFoldingStats,
    /// Algebraic simplification statistics
    pub algebraic: AlgebraicSimplificationStats,
    /// Strength reduction statistics
    pub strength_reduction: StrengthReductionStats,
    /// Distributivity optimization statistics
    pub distributivity: DistributivityStats,
    /// Quantifier optimization statistics
    pub quantifier_opt: QuantifierOptStats,
    /// Dead code elimination statistics
    pub dead_code: DeadCodeStats,
}

impl IterationStats {
    /// Check if this iteration made any changes.
    pub fn made_changes(&self) -> bool {
        self.negation.double_negations_eliminated > 0
            || self.negation.demorgans_applied > 0
            || self.negation.quantifier_negations_pushed > 0
            || self.constant_folding.binary_ops_folded > 0
            || self.constant_folding.unary_ops_folded > 0
            || self.algebraic.identities_eliminated > 0
            || self.algebraic.annihilations_applied > 0
            || self.algebraic.idempotent_simplified > 0
            || self.strength_reduction.total_optimizations() > 0
            || self.distributivity.total_optimizations() > 0
            || self.quantifier_opt.total_optimizations() > 0
            || self.dead_code.total_optimizations() > 0
    }

    /// Get total number of optimizations applied in this iteration.
    pub fn total_optimizations(&self) -> usize {
        self.negation.double_negations_eliminated
            + self.negation.demorgans_applied
            + self.negation.quantifier_negations_pushed
            + self.constant_folding.binary_ops_folded
            + self.constant_folding.unary_ops_folded
            + self.algebraic.identities_eliminated
            + self.algebraic.annihilations_applied
            + self.algebraic.idempotent_simplified
            + self.strength_reduction.total_optimizations()
            + self.distributivity.total_optimizations()
            + self.quantifier_opt.total_optimizations()
            + self.dead_code.total_optimizations()
    }
}

/// Cumulative statistics from all pipeline iterations.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total number of iterations performed
    pub total_iterations: usize,
    /// Negation optimization statistics (accumulated)
    pub negation: NegationOptStats,
    /// Constant folding statistics (accumulated)
    pub constant_folding: ConstantFoldingStats,
    /// Algebraic simplification statistics (accumulated)
    pub algebraic: AlgebraicSimplificationStats,
    /// Strength reduction statistics (accumulated)
    pub strength_reduction: StrengthReductionStats,
    /// Distributivity optimization statistics (accumulated)
    pub distributivity: DistributivityStats,
    /// Quantifier optimization statistics (accumulated)
    pub quantifier_opt: QuantifierOptStats,
    /// Dead code elimination statistics (accumulated)
    pub dead_code: DeadCodeStats,
    /// Statistics per iteration
    pub iterations: Vec<IterationStats>,
    /// Whether the pipeline reached a fixed point
    pub reached_fixed_point: bool,
    /// Whether the pipeline was stopped due to max iterations
    pub stopped_at_max_iterations: bool,
}

impl PipelineStats {
    /// Get total number of optimizations applied across all iterations.
    pub fn total_optimizations(&self) -> usize {
        self.negation.double_negations_eliminated
            + self.negation.demorgans_applied
            + self.negation.quantifier_negations_pushed
            + self.constant_folding.binary_ops_folded
            + self.constant_folding.unary_ops_folded
            + self.algebraic.identities_eliminated
            + self.algebraic.annihilations_applied
            + self.algebraic.idempotent_simplified
            + self.strength_reduction.total_optimizations()
            + self.distributivity.total_optimizations()
            + self.quantifier_opt.total_optimizations()
            + self.dead_code.total_optimizations()
    }

    /// Get the most productive iteration (one with most optimizations).
    pub fn most_productive_iteration(&self) -> Option<(usize, &IterationStats)> {
        self.iterations
            .iter()
            .enumerate()
            .max_by_key(|(_, stats)| stats.total_optimizations())
    }
}

impl std::fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pipeline Statistics:")?;
        writeln!(f, "  Iterations: {}", self.total_iterations)?;
        writeln!(f, "  Reached fixed point: {}", self.reached_fixed_point)?;
        writeln!(f, "  Total optimizations: {}", self.total_optimizations())?;
        writeln!(f, "\nNegation Optimization:")?;
        writeln!(
            f,
            "  Double negations eliminated: {}",
            self.negation.double_negations_eliminated
        )?;
        writeln!(
            f,
            "  De Morgan's laws applied: {}",
            self.negation.demorgans_applied
        )?;
        writeln!(
            f,
            "  Quantifier negations pushed: {}",
            self.negation.quantifier_negations_pushed
        )?;
        writeln!(f, "\nConstant Folding:")?;
        writeln!(
            f,
            "  Binary ops folded: {}",
            self.constant_folding.binary_ops_folded
        )?;
        writeln!(
            f,
            "  Unary ops folded: {}",
            self.constant_folding.unary_ops_folded
        )?;
        writeln!(f, "\nAlgebraic Simplification:")?;
        writeln!(
            f,
            "  Identities eliminated: {}",
            self.algebraic.identities_eliminated
        )?;
        writeln!(
            f,
            "  Annihilations applied: {}",
            self.algebraic.annihilations_applied
        )?;
        writeln!(
            f,
            "  Idempotent simplified: {}",
            self.algebraic.idempotent_simplified
        )?;
        writeln!(f, "\nStrength Reduction:")?;
        writeln!(
            f,
            "  Power reductions: {}",
            self.strength_reduction.power_reductions
        )?;
        writeln!(
            f,
            "  Operations eliminated: {}",
            self.strength_reduction.operations_eliminated
        )?;
        writeln!(
            f,
            "  Special function optimizations: {}",
            self.strength_reduction.special_function_optimizations
        )?;
        writeln!(f, "\nDistributivity:")?;
        writeln!(
            f,
            "  Expressions factored: {}",
            self.distributivity.expressions_factored
        )?;
        writeln!(
            f,
            "  Expressions expanded: {}",
            self.distributivity.expressions_expanded
        )?;
        writeln!(f, "\nQuantifier Optimization:")?;
        writeln!(
            f,
            "  Invariants hoisted: {}",
            self.quantifier_opt.invariants_hoisted
        )?;
        writeln!(
            f,
            "  Quantifiers reordered: {}",
            self.quantifier_opt.quantifiers_reordered
        )?;
        writeln!(f, "\nDead Code Elimination:")?;
        writeln!(
            f,
            "  Branches eliminated: {}",
            self.dead_code.branches_eliminated
        )?;
        writeln!(f, "  Short circuits: {}", self.dead_code.short_circuits)?;
        writeln!(
            f,
            "  Unused quantifiers removed: {}",
            self.dead_code.unused_quantifiers_removed
        )?;
        Ok(())
    }
}

/// Multi-pass optimization pipeline for TLExpr expressions.
///
/// The pipeline applies 7 optimization passes in sequence, iterating
/// until a fixed point is reached or the maximum number of iterations is hit.
///
/// # Pass Order
///
/// 1. **Negation optimization**: Applies De Morgan's laws and eliminates
///    double negations. This exposes more opportunities for subsequent passes.
///
/// 2. **Constant folding**: Evaluates constant expressions at compile time.
///    This creates simpler expressions with fewer operations.
///
/// 3. **Algebraic simplification**: Applies mathematical identities like
///    x + 0 = x, x * 1 = x, etc. This can create new constants for folding.
///
/// 4. **Strength reduction**: Replaces expensive operations with cheaper
///    equivalents (e.g., x^2 → x*x, exp(log(x)) → x).
///
/// 5. **Distributivity**: Factors common subexpressions to reduce redundant
///    computation (e.g., a*b + a*c → a*(b+c)).
///
/// 6. **Quantifier optimization**: Hoists loop-invariant expressions out of
///    quantifiers (e.g., ∃x.(a + p(x)) → a + ∃x.p(x)).
///
/// 7. **Dead code elimination**: Removes unreachable code and eliminates
///    branches with constant conditions (e.g., if true then A else B → A).
///
/// # Fixed Point Detection
///
/// The pipeline tracks whether each pass makes changes. If an entire iteration
/// produces no changes (i.e., the expression is unchanged), a fixed point has
/// been reached and optimization stops early.
///
/// # Examples
///
/// ```
/// use tensorlogic_compiler::optimize::{OptimizationPipeline, PipelineConfig};
/// use tensorlogic_ir::{TLExpr, Term};
///
/// // Default pipeline
/// let pipeline = OptimizationPipeline::new();
///
/// // Custom configuration
/// let config = PipelineConfig::default()
///     .with_max_iterations(5)
///     .with_constant_folding(true);
/// let pipeline = OptimizationPipeline::with_config(config);
///
/// // Optimize an expression
/// let expr = TLExpr::add(
///     TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
///     TLExpr::Constant(0.0)
/// );
/// let (optimized, stats) = pipeline.optimize(&expr);
/// ```
pub struct OptimizationPipeline {
    config: PipelineConfig,
}

impl OptimizationPipeline {
    /// Create a new optimization pipeline with default configuration.
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }

    /// Create a new optimization pipeline with custom configuration.
    pub fn with_config(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Optimize an expression using the configured pipeline.
    ///
    /// Returns the optimized expression and statistics about the optimizations applied.
    pub fn optimize(&self, expr: &TLExpr) -> (TLExpr, PipelineStats) {
        let mut current = expr.clone();
        let mut stats = PipelineStats::default();

        for iteration in 0..self.config.max_iterations {
            let mut iter_stats = IterationStats::default();
            let mut changed = false;

            // Pass 1: Negation optimization
            if self.config.enable_negation_opt {
                let (optimized, neg_stats) = optimize_negations(&current);
                iter_stats.negation = neg_stats;

                if optimized != current {
                    current = optimized;
                    changed = true;
                }
            }

            // Pass 2: Constant folding
            if self.config.enable_constant_folding {
                let (optimized, fold_stats) = fold_constants(&current);
                iter_stats.constant_folding = fold_stats;

                if optimized != current {
                    current = optimized;
                    changed = true;
                }
            }

            // Pass 3: Algebraic simplification
            if self.config.enable_algebraic_simplification {
                let (optimized, alg_stats) = simplify_algebraic(&current);
                iter_stats.algebraic = alg_stats;

                if optimized != current {
                    current = optimized;
                    changed = true;
                }
            }

            // Pass 4: Strength reduction
            if self.config.enable_strength_reduction {
                let (optimized, sr_stats) = reduce_strength(&current);
                iter_stats.strength_reduction = sr_stats;

                if optimized != current {
                    current = optimized;
                    changed = true;
                }
            }

            // Pass 5: Distributivity optimization
            if self.config.enable_distributivity {
                let (optimized, dist_stats) = optimize_distributivity(&current);
                iter_stats.distributivity = dist_stats;

                if optimized != current {
                    current = optimized;
                    changed = true;
                }
            }

            // Pass 6: Quantifier optimization
            if self.config.enable_quantifier_opt {
                let (optimized, quant_stats) = optimize_quantifiers(&current);
                iter_stats.quantifier_opt = quant_stats;

                if optimized != current {
                    current = optimized;
                    changed = true;
                }
            }

            // Pass 7: Dead code elimination
            if self.config.enable_dead_code_elimination {
                let (optimized, dead_stats) = eliminate_dead_code(&current);
                iter_stats.dead_code = dead_stats;

                if optimized != current {
                    current = optimized;
                    changed = true;
                }
            }

            // Accumulate statistics
            stats.total_iterations = iteration + 1;
            stats.negation.double_negations_eliminated +=
                iter_stats.negation.double_negations_eliminated;
            stats.negation.demorgans_applied += iter_stats.negation.demorgans_applied;
            stats.negation.quantifier_negations_pushed +=
                iter_stats.negation.quantifier_negations_pushed;
            stats.constant_folding.binary_ops_folded +=
                iter_stats.constant_folding.binary_ops_folded;
            stats.constant_folding.unary_ops_folded += iter_stats.constant_folding.unary_ops_folded;
            stats.constant_folding.total_processed += iter_stats.constant_folding.total_processed;
            stats.algebraic.identities_eliminated += iter_stats.algebraic.identities_eliminated;
            stats.algebraic.annihilations_applied += iter_stats.algebraic.annihilations_applied;
            stats.algebraic.idempotent_simplified += iter_stats.algebraic.idempotent_simplified;
            stats.algebraic.total_processed += iter_stats.algebraic.total_processed;
            stats.strength_reduction.power_reductions +=
                iter_stats.strength_reduction.power_reductions;
            stats.strength_reduction.operations_eliminated +=
                iter_stats.strength_reduction.operations_eliminated;
            stats.strength_reduction.special_function_optimizations +=
                iter_stats.strength_reduction.special_function_optimizations;
            stats.strength_reduction.total_processed +=
                iter_stats.strength_reduction.total_processed;
            stats.distributivity.expressions_factored +=
                iter_stats.distributivity.expressions_factored;
            stats.distributivity.expressions_expanded +=
                iter_stats.distributivity.expressions_expanded;
            stats.distributivity.common_terms_extracted +=
                iter_stats.distributivity.common_terms_extracted;
            stats.distributivity.total_processed += iter_stats.distributivity.total_processed;
            stats.quantifier_opt.invariants_hoisted += iter_stats.quantifier_opt.invariants_hoisted;
            stats.quantifier_opt.quantifiers_reordered +=
                iter_stats.quantifier_opt.quantifiers_reordered;
            stats.quantifier_opt.quantifiers_fused += iter_stats.quantifier_opt.quantifiers_fused;
            stats.quantifier_opt.total_processed += iter_stats.quantifier_opt.total_processed;
            stats.dead_code.branches_eliminated += iter_stats.dead_code.branches_eliminated;
            stats.dead_code.short_circuits += iter_stats.dead_code.short_circuits;
            stats.dead_code.unused_quantifiers_removed +=
                iter_stats.dead_code.unused_quantifiers_removed;
            stats.dead_code.identity_simplifications +=
                iter_stats.dead_code.identity_simplifications;
            stats.dead_code.total_processed += iter_stats.dead_code.total_processed;
            stats.iterations.push(iter_stats);

            // Check for fixed point
            if self.config.stop_on_fixed_point && !changed {
                stats.reached_fixed_point = true;
                break;
            }

            // Check if we've hit max iterations
            if iteration + 1 >= self.config.max_iterations {
                stats.stopped_at_max_iterations = true;
            }
        }

        (current, stats)
    }

    /// Get the current configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

impl Default for OptimizationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_pipeline_with_all_passes() {
        // Expression: NOT(AND(x + 0, 2.0 * 3.0))
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::negate(TLExpr::and(
            TLExpr::add(x, TLExpr::Constant(0.0)),
            TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
        ));

        let pipeline = OptimizationPipeline::new();
        let (optimized, stats) = pipeline.optimize(&expr);

        // Should apply multiple optimizations
        assert!(stats.total_iterations > 0);
        assert!(stats.constant_folding.binary_ops_folded > 0);
        assert!(stats.algebraic.identities_eliminated > 0);
        assert!(stats.negation.demorgans_applied > 0);

        // Should not be the same as original
        assert!(optimized != expr);
    }

    #[test]
    fn test_constant_folding_only() {
        let expr = TLExpr::add(
            TLExpr::Constant(2.0),
            TLExpr::mul(TLExpr::Constant(3.0), TLExpr::Constant(4.0)),
        );

        let config = PipelineConfig::constant_folding_only();
        let pipeline = OptimizationPipeline::with_config(config);
        let (optimized, stats) = pipeline.optimize(&expr);

        // Should fold to 2.0 + 12.0 = 14.0
        assert!(matches!(optimized, TLExpr::Constant(_)));
        assert_eq!(stats.constant_folding.binary_ops_folded, 2);
        assert_eq!(stats.algebraic.identities_eliminated, 0);
        assert_eq!(stats.negation.demorgans_applied, 0);
    }

    #[test]
    fn test_algebraic_only() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::mul(TLExpr::add(x, TLExpr::Constant(0.0)), TLExpr::Constant(1.0));

        let config = PipelineConfig::algebraic_only();
        let pipeline = OptimizationPipeline::with_config(config);
        let (_optimized, stats) = pipeline.optimize(&expr);

        // Should eliminate both identities: x + 0 = x, x * 1 = x
        assert_eq!(stats.algebraic.identities_eliminated, 2);
        assert_eq!(stats.constant_folding.binary_ops_folded, 0);
    }

    #[test]
    fn test_fixed_point_detection() {
        // Expression that's already optimal
        let x = TLExpr::pred("x", vec![Term::var("i")]);

        let config = PipelineConfig::default().with_max_iterations(10);
        let pipeline = OptimizationPipeline::with_config(config);
        let (optimized, stats) = pipeline.optimize(&x);

        // Should stop after first iteration (no changes)
        assert_eq!(stats.total_iterations, 1);
        assert!(stats.reached_fixed_point);
        assert!(!stats.stopped_at_max_iterations);
        assert_eq!(optimized, x);
    }

    #[test]
    fn test_max_iterations_limit() {
        // Create an expression that could benefit from more iterations
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::negate(TLExpr::negate(TLExpr::add(x, TLExpr::Constant(0.0))));

        let config = PipelineConfig::default().with_max_iterations(1);
        let pipeline = OptimizationPipeline::with_config(config);
        let (_, stats) = pipeline.optimize(&expr);

        assert_eq!(stats.total_iterations, 1);
        assert!(stats.stopped_at_max_iterations);
    }

    #[test]
    fn test_aggressive_optimization() {
        // Complex nested expression that requires multiple optimization passes
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        // Expression: NOT(AND(NOT(x + 0), NOT((2.0 * 3.0) * x))) + (1.0 * 1.0)
        let expr = TLExpr::add(
            TLExpr::negate(TLExpr::and(
                TLExpr::negate(TLExpr::add(x.clone(), TLExpr::Constant(0.0))),
                TLExpr::negate(TLExpr::mul(
                    TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
                    x,
                )),
            )),
            TLExpr::mul(TLExpr::Constant(1.0), TLExpr::Constant(1.0)),
        );

        let config = PipelineConfig::aggressive();
        let pipeline = OptimizationPipeline::with_config(config);
        let (_, stats) = pipeline.optimize(&expr);

        // Should apply multiple optimizations (negation, folding, algebraic)
        // At least: De Morgan's law, double negations, constant folding, identity elimination
        assert!(
            stats.total_optimizations() >= 4,
            "Expected at least 4 optimizations, got {}",
            stats.total_optimizations()
        );
        assert!(stats.total_iterations >= 1);
    }

    #[test]
    fn test_no_optimization() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::add(x.clone(), TLExpr::Constant(1.0));

        let config = PipelineConfig::none();
        let pipeline = OptimizationPipeline::with_config(config);
        let (optimized, stats) = pipeline.optimize(&expr);

        // Should make no changes
        assert_eq!(optimized, expr);
        assert_eq!(stats.total_optimizations(), 0);
    }

    #[test]
    fn test_iteration_stats() {
        let expr = TLExpr::add(
            TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
            TLExpr::Constant(0.0),
        );

        let pipeline = OptimizationPipeline::new();
        let (_, stats) = pipeline.optimize(&expr);

        // Check per-iteration statistics
        assert!(!stats.iterations.is_empty());
        assert!(stats.iterations[0].made_changes());
        assert!(stats.iterations[0].total_optimizations() > 0);
    }

    #[test]
    fn test_most_productive_iteration() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::negate(TLExpr::negate(TLExpr::add(
            TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
            TLExpr::mul(x, TLExpr::Constant(1.0)),
        )));

        let pipeline = OptimizationPipeline::new();
        let (_, stats) = pipeline.optimize(&expr);

        // Should identify the most productive iteration
        let (iter_idx, iter_stats) = stats.most_productive_iteration().unwrap();
        assert!(iter_stats.total_optimizations() > 0);
        assert!(iter_idx < stats.total_iterations);
    }

    #[test]
    fn test_pipeline_display() {
        let expr = TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0));
        let pipeline = OptimizationPipeline::new();
        let (_, stats) = pipeline.optimize(&expr);

        // Test Display implementation
        let output = format!("{}", stats);
        assert!(output.contains("Pipeline Statistics"));
        assert!(output.contains("Iterations:"));
        assert!(output.contains("Total optimizations:"));
    }

    #[test]
    fn test_builder_pattern() {
        let config = PipelineConfig::default()
            .with_negation_opt(false)
            .with_constant_folding(true)
            .with_algebraic_simplification(false)
            .with_max_iterations(5)
            .with_stop_on_fixed_point(false);

        assert!(!config.enable_negation_opt);
        assert!(config.enable_constant_folding);
        assert!(!config.enable_algebraic_simplification);
        assert_eq!(config.max_iterations, 5);
        assert!(!config.stop_on_fixed_point);
    }

    #[test]
    fn test_complex_real_world_expression() {
        // Softmax-like expression: exp((x - max) / 1.0) where temperature = 1.0
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let max_val = TLExpr::pred("max", vec![]);
        let temp = TLExpr::Constant(1.0);

        let expr = TLExpr::exp(TLExpr::div(TLExpr::sub(x, max_val), temp));

        let pipeline = OptimizationPipeline::new();
        let (optimized, stats) = pipeline.optimize(&expr);

        // Should eliminate division by 1.0
        assert!(stats.algebraic.identities_eliminated > 0);
        assert!(optimized != expr);
    }

    #[test]
    fn test_dead_code_elimination_integration() {
        // Expression with dead branches: if true then A else B → A
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("i")]);
        let expr = TLExpr::IfThenElse {
            condition: Box::new(TLExpr::Constant(1.0)), // Always true
            then_branch: Box::new(a.clone()),
            else_branch: Box::new(b),
        };

        let pipeline = OptimizationPipeline::new();
        let (optimized, stats) = pipeline.optimize(&expr);

        // Should eliminate the dead else branch
        assert!(stats.dead_code.branches_eliminated > 0);
        // Should be simplified to just 'a'
        assert!(matches!(optimized, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_all_passes_together() {
        // Complex expression benefiting from all passes
        // if true then (NOT(NOT(x^2 + 0)) AND (a*b + a*c)) else FALSE
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let a = TLExpr::pred("a", vec![Term::var("i")]);
        let b = TLExpr::pred("b", vec![Term::var("i")]);
        let c = TLExpr::pred("c", vec![Term::var("i")]);

        let expr = TLExpr::IfThenElse {
            condition: Box::new(TLExpr::Constant(1.0)),
            then_branch: Box::new(TLExpr::and(
                TLExpr::negate(TLExpr::negate(TLExpr::add(
                    TLExpr::pow(x, TLExpr::Constant(2.0)),
                    TLExpr::Constant(0.0),
                ))),
                TLExpr::add(
                    TLExpr::mul(a.clone(), b.clone()),
                    TLExpr::mul(a.clone(), c.clone()),
                ),
            )),
            else_branch: Box::new(TLExpr::Constant(0.0)),
        };

        let pipeline = OptimizationPipeline::new();
        let (_, stats) = pipeline.optimize(&expr);

        // Should apply multiple passes:
        // - Dead code elimination (remove else branch)
        // - Negation optimization (double negation)
        // - Algebraic simplification (x + 0 → x)
        // - Strength reduction (x^2 → x*x)
        // - Distributivity (a*b + a*c → a*(b+c))
        assert!(
            stats.dead_code.branches_eliminated > 0,
            "Dead code elimination should apply"
        );
        assert!(
            stats.negation.double_negations_eliminated > 0,
            "Negation optimization should apply"
        );
        assert!(
            stats.algebraic.identities_eliminated > 0,
            "Algebraic simplification should apply"
        );
        assert!(
            stats.strength_reduction.power_reductions > 0,
            "Strength reduction should apply"
        );
        assert!(
            stats.distributivity.expressions_factored > 0,
            "Distributivity should apply"
        );

        // Total should be significant
        assert!(
            stats.total_optimizations() >= 5,
            "Should apply at least 5 optimizations"
        );
    }
}
