//! Automatic optimization strategy selection based on expression characteristics.
//!
//! This module provides heuristics for automatically selecting the best optimization
//! strategy based on the structure and complexity of expressions.

use super::{
    advanced_analysis::{ComplexityMetrics, OperatorCounts},
    optimization_pipeline::{OptimizationLevel, OptimizationPass, PipelineConfig},
    TLExpr,
};

/// Characteristics of an expression used for strategy selection.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpressionProfile {
    /// Operator counts by category
    pub operator_counts: OperatorCounts,
    /// Complexity metrics
    pub complexity: ComplexityMetrics,
    /// Whether the expression has quantifiers
    pub has_quantifiers: bool,
    /// Whether the expression has modal operators
    pub has_modal: bool,
    /// Whether the expression has temporal operators
    pub has_temporal: bool,
    /// Whether the expression has fuzzy operators
    pub has_fuzzy: bool,
    /// Whether the expression has constants
    pub has_constants: bool,
    /// Expression size (node count)
    pub size: usize,
}

impl ExpressionProfile {
    /// Analyze an expression to create a profile.
    pub fn analyze(expr: &TLExpr) -> Self {
        let operator_counts = OperatorCounts::from_expr(expr);
        let complexity = ComplexityMetrics::from_expr(expr);

        Self {
            has_quantifiers: operator_counts.quantifiers > 0,
            has_modal: operator_counts.modal > 0,
            has_temporal: operator_counts.temporal > 0,
            has_fuzzy: operator_counts.fuzzy > 0,
            has_constants: operator_counts.constants > 0,
            size: operator_counts.total,
            operator_counts,
            complexity,
        }
    }

    /// Check if the expression is simple (few operators, shallow depth).
    pub fn is_simple(&self) -> bool {
        self.size <= 10 && self.complexity.max_depth <= 3
    }

    /// Check if the expression is complex (many operators, deep nesting).
    pub fn is_complex(&self) -> bool {
        self.size > 50 || self.complexity.max_depth > 10
    }

    /// Check if the expression would benefit from distributive laws.
    pub fn needs_distribution(&self) -> bool {
        // Check for patterns like A ∧ (B ∨ C) or A ∨ (B ∧ C)
        // Heuristic: many logical operators might benefit
        self.operator_counts.logical > 5
    }

    /// Check if the expression has significant constant folding opportunities.
    pub fn has_constant_opportunities(&self) -> bool {
        self.has_constants && self.operator_counts.arithmetic > 0
    }
}

/// Strategy selector that recommends optimization configurations.
#[derive(Clone, Copy, Debug)]
pub struct StrategySelector {
    /// Default optimization level for fallback
    _default_level: OptimizationLevel,
}

impl Default for StrategySelector {
    fn default() -> Self {
        Self {
            _default_level: OptimizationLevel::Standard,
        }
    }
}

impl StrategySelector {
    /// Create a new strategy selector with a default optimization level.
    pub fn new(default_level: OptimizationLevel) -> Self {
        Self {
            _default_level: default_level,
        }
    }

    /// Select an optimization level based on expression profile.
    pub fn select_level(&self, profile: &ExpressionProfile) -> OptimizationLevel {
        // Simple expressions: use basic optimizations
        if profile.is_simple() {
            return OptimizationLevel::Basic;
        }

        // Complex expressions with specific features: use aggressive
        if profile.is_complex()
            && (profile.has_modal || profile.has_temporal || profile.has_quantifiers)
        {
            return OptimizationLevel::Aggressive;
        }

        // Default to standard for most cases
        OptimizationLevel::Standard
    }

    /// Select specific optimization passes based on expression profile.
    pub fn select_passes(&self, profile: &ExpressionProfile) -> Vec<OptimizationPass> {
        let mut passes = Vec::new();

        // Always include constant folding if there are constants
        if profile.has_constants {
            passes.push(OptimizationPass::ConstantFolding);
            passes.push(OptimizationPass::ConstantPropagation);
        }

        // Always include basic simplifications
        passes.push(OptimizationPass::AlgebraicSimplification);

        // Add NNF conversion for complex logical expressions
        if profile.operator_counts.logical > 3 {
            passes.push(OptimizationPass::NegationNormalForm);
        }

        // Add modal equivalences if modal operators present
        if profile.has_modal {
            passes.push(OptimizationPass::ModalEquivalences);
            passes.push(OptimizationPass::DistributiveModal);
        }

        // Add temporal equivalences if temporal operators present
        if profile.has_temporal {
            passes.push(OptimizationPass::TemporalEquivalences);
        }

        // Add quantifier distribution if quantifiers present
        if profile.has_quantifiers && profile.operator_counts.quantifiers > 2 {
            passes.push(OptimizationPass::DistributiveQuantifiers);
        }

        // Add distributive laws for complex logical expressions
        if profile.needs_distribution() {
            passes.push(OptimizationPass::DistributiveAndOverOr);
        }

        passes
    }

    /// Create a recommended pipeline configuration for an expression.
    pub fn recommend_config(&self, expr: &TLExpr) -> PipelineConfig {
        let profile = ExpressionProfile::analyze(expr);
        let level = self.select_level(&profile);
        let custom_passes = self.select_passes(&profile);

        let max_iterations = if profile.is_complex() { 15 } else { 10 };

        PipelineConfig::with_level(level)
            .with_custom_passes(custom_passes)
            .with_max_iterations(max_iterations)
    }

    /// Quick optimization recommendation: returns whether aggressive optimization is recommended.
    pub fn should_optimize_aggressively(&self, expr: &TLExpr) -> bool {
        let profile = ExpressionProfile::analyze(expr);
        matches!(self.select_level(&profile), OptimizationLevel::Aggressive)
    }
}

/// Convenience function to automatically select and apply the best optimization strategy.
pub fn auto_optimize(expr: TLExpr) -> (TLExpr, super::optimization_pipeline::OptimizationMetrics) {
    let selector = StrategySelector::default();
    let config = selector.recommend_config(&expr);

    let pipeline = super::optimization_pipeline::OptimizationPipeline::new(config);
    pipeline.optimize(expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_profile_simple_expression() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        let profile = ExpressionProfile::analyze(&expr);

        assert!(profile.is_simple());
        assert!(!profile.is_complex());
        assert!(!profile.has_constants);
    }

    #[test]
    fn test_profile_complex_expression() {
        // Build a deeply nested expression
        let mut expr = TLExpr::pred("P", vec![Term::var("x")]);
        for _ in 0..15 {
            expr = TLExpr::and(expr.clone(), TLExpr::pred("Q", vec![Term::var("y")]));
        }

        let profile = ExpressionProfile::analyze(&expr);
        assert!(profile.is_complex());
        assert!(!profile.is_simple());
    }

    #[test]
    fn test_profile_with_constants() {
        let expr = TLExpr::add(TLExpr::constant(1.0), TLExpr::constant(2.0));
        let profile = ExpressionProfile::analyze(&expr);

        assert!(profile.has_constants);
        assert!(profile.has_constant_opportunities());
    }

    #[test]
    fn test_profile_with_quantifiers() {
        let expr = TLExpr::forall("x", "D", TLExpr::pred("P", vec![Term::var("x")]));
        let profile = ExpressionProfile::analyze(&expr);

        assert!(profile.has_quantifiers);
    }

    #[test]
    fn test_profile_with_modal() {
        let expr = TLExpr::modal_box(TLExpr::pred("P", vec![Term::var("x")]));
        let profile = ExpressionProfile::analyze(&expr);

        assert!(profile.has_modal);
    }

    #[test]
    fn test_profile_with_temporal() {
        let expr = TLExpr::eventually(TLExpr::pred("P", vec![Term::var("x")]));
        let profile = ExpressionProfile::analyze(&expr);

        assert!(profile.has_temporal);
    }

    #[test]
    fn test_selector_simple_expression() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        let selector = StrategySelector::default();
        let profile = ExpressionProfile::analyze(&expr);

        let level = selector.select_level(&profile);
        assert_eq!(level, OptimizationLevel::Basic);
    }

    #[test]
    fn test_selector_complex_modal_expression() {
        // Build a complex modal expression
        let mut expr = TLExpr::modal_box(TLExpr::pred("P", vec![Term::var("x")]));
        for _ in 0..12 {
            expr = TLExpr::and(expr.clone(), TLExpr::modal_box(TLExpr::pred("Q", vec![])));
        }

        let selector = StrategySelector::default();
        let profile = ExpressionProfile::analyze(&expr);

        let level = selector.select_level(&profile);
        assert_eq!(level, OptimizationLevel::Aggressive);
    }

    #[test]
    fn test_selector_pass_selection() {
        let expr = TLExpr::and(
            TLExpr::constant(1.0),
            TLExpr::modal_box(TLExpr::pred("P", vec![Term::var("x")])),
        );

        let selector = StrategySelector::default();
        let profile = ExpressionProfile::analyze(&expr);
        let passes = selector.select_passes(&profile);

        // Should include constant folding and modal equivalences
        assert!(passes.contains(&OptimizationPass::ConstantFolding));
        assert!(passes.contains(&OptimizationPass::ModalEquivalences));
    }

    #[test]
    fn test_recommend_config() {
        let expr = TLExpr::and(
            TLExpr::constant(1.0),
            TLExpr::pred("P", vec![Term::var("x")]),
        );

        let selector = StrategySelector::default();
        let config = selector.recommend_config(&expr);

        assert_eq!(config.level, OptimizationLevel::Basic);
        assert!(config.custom_passes.is_some());
    }

    #[test]
    fn test_auto_optimize() {
        let expr = TLExpr::and(
            TLExpr::constant(1.0),
            TLExpr::pred("P", vec![Term::var("x")]),
        );

        let (optimized, metrics) = auto_optimize(expr);

        // Should have applied optimizations
        assert!(metrics.passes_applied > 0);
        // Should have simplified the expression
        assert_eq!(optimized, TLExpr::pred("P", vec![Term::var("x")]));
    }

    #[test]
    fn test_should_optimize_aggressively() {
        let simple_expr = TLExpr::pred("P", vec![Term::var("x")]);
        let selector = StrategySelector::default();

        assert!(!selector.should_optimize_aggressively(&simple_expr));

        // Build complex expression
        let mut complex_expr = TLExpr::modal_box(TLExpr::pred("P", vec![Term::var("x")]));
        for _ in 0..12 {
            complex_expr = TLExpr::and(
                complex_expr.clone(),
                TLExpr::modal_box(TLExpr::pred("Q", vec![])),
            );
        }

        assert!(selector.should_optimize_aggressively(&complex_expr));
    }

    #[test]
    fn test_needs_distribution() {
        // Expression with many logical operators
        let mut expr = TLExpr::pred("P", vec![Term::var("x")]);
        for i in 0..7 {
            expr = TLExpr::and(
                expr,
                TLExpr::or(
                    TLExpr::pred("Q", vec![Term::var(format!("x{}", i))]),
                    TLExpr::pred("R", vec![Term::var(format!("y{}", i))]),
                ),
            );
        }

        let profile = ExpressionProfile::analyze(&expr);
        assert!(profile.needs_distribution());
    }
}
