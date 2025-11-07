//! Automatic strategy selection based on expression analysis.
//!
//! This module analyzes TLExpr expressions to recommend optimal compilation
//! strategies based on expression characteristics, usage context, and optimization goals.

use crate::config::*;
use tensorlogic_ir::TLExpr;

/// Optimization goals for strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationGoal {
    /// Prioritize differentiability for gradient-based training
    Differentiable,
    /// Prioritize discrete/Boolean reasoning accuracy
    DiscreteReasoning,
    /// Prioritize computational efficiency
    Performance,
    /// Balance between differentiability and accuracy
    Balanced,
}

/// Expression characteristics that influence strategy selection.
#[derive(Debug, Clone, Default)]
pub struct ExpressionProfile {
    /// Number of AND operations
    pub and_count: usize,
    /// Number of OR operations
    pub or_count: usize,
    /// Number of NOT operations
    pub not_count: usize,
    /// Number of EXISTS quantifiers
    pub exists_count: usize,
    /// Number of FORALL quantifiers
    pub forall_count: usize,
    /// Number of implications
    pub implication_count: usize,
    /// Number of arithmetic operations
    pub arithmetic_count: usize,
    /// Number of comparisons
    pub comparison_count: usize,
    /// Maximum nesting depth
    pub max_depth: usize,
    /// Has nested quantifiers
    pub has_nested_quantifiers: bool,
    /// Has negated quantifiers
    pub has_negated_quantifiers: bool,
    /// Has arithmetic mixed with logic
    pub has_mixed_operations: bool,
}

impl ExpressionProfile {
    /// Analyze an expression to build its profile.
    pub fn analyze(expr: &TLExpr) -> Self {
        let mut profile = Self::default();
        Self::analyze_recursive(expr, 0, &mut profile);
        profile
    }

    fn analyze_recursive(expr: &TLExpr, depth: usize, profile: &mut ExpressionProfile) {
        profile.max_depth = profile.max_depth.max(depth);

        match expr {
            TLExpr::And(left, right) => {
                profile.and_count += 1;

                // Check for mixed operations
                if Self::contains_arithmetic_op(left) || Self::contains_arithmetic_op(right) {
                    profile.has_mixed_operations = true;
                }

                Self::analyze_recursive(left, depth + 1, profile);
                Self::analyze_recursive(right, depth + 1, profile);
            }
            TLExpr::Or(left, right) => {
                profile.or_count += 1;

                // Check for mixed operations
                if Self::contains_arithmetic_op(left) || Self::contains_arithmetic_op(right) {
                    profile.has_mixed_operations = true;
                }

                Self::analyze_recursive(left, depth + 1, profile);
                Self::analyze_recursive(right, depth + 1, profile);
            }
            TLExpr::Not(inner) => {
                profile.not_count += 1;

                // Check for negated quantifiers
                if matches!(**inner, TLExpr::Exists { .. } | TLExpr::ForAll { .. }) {
                    profile.has_negated_quantifiers = true;
                }

                Self::analyze_recursive(inner, depth + 1, profile);
            }
            TLExpr::Exists { body, .. } => {
                profile.exists_count += 1;

                // Check for nested quantifiers
                if Self::contains_quantifier(body) {
                    profile.has_nested_quantifiers = true;
                }

                Self::analyze_recursive(body, depth + 1, profile);
            }
            TLExpr::ForAll { body, .. } => {
                profile.forall_count += 1;

                // Check for nested quantifiers
                if Self::contains_quantifier(body) {
                    profile.has_nested_quantifiers = true;
                }

                Self::analyze_recursive(body, depth + 1, profile);
            }
            TLExpr::Imply(premise, conclusion) => {
                profile.implication_count += 1;
                Self::analyze_recursive(premise, depth + 1, profile);
                Self::analyze_recursive(conclusion, depth + 1, profile);
            }
            TLExpr::Add(left, right)
            | TLExpr::Sub(left, right)
            | TLExpr::Mul(left, right)
            | TLExpr::Div(left, right)
            | TLExpr::Pow(left, right)
            | TLExpr::Mod(left, right) => {
                profile.arithmetic_count += 1;

                // Check for mixed operations
                if Self::contains_logical_op(left) || Self::contains_logical_op(right) {
                    profile.has_mixed_operations = true;
                }

                Self::analyze_recursive(left, depth + 1, profile);
                Self::analyze_recursive(right, depth + 1, profile);
            }
            TLExpr::Min(left, right) | TLExpr::Max(left, right) => {
                profile.arithmetic_count += 1;
                Self::analyze_recursive(left, depth + 1, profile);
                Self::analyze_recursive(right, depth + 1, profile);
            }
            TLExpr::Eq(left, right)
            | TLExpr::Lt(left, right)
            | TLExpr::Gt(left, right)
            | TLExpr::Lte(left, right)
            | TLExpr::Gte(left, right) => {
                profile.comparison_count += 1;
                Self::analyze_recursive(left, depth + 1, profile);
                Self::analyze_recursive(right, depth + 1, profile);
            }
            TLExpr::Abs(inner)
            | TLExpr::Floor(inner)
            | TLExpr::Ceil(inner)
            | TLExpr::Round(inner)
            | TLExpr::Sqrt(inner)
            | TLExpr::Exp(inner)
            | TLExpr::Log(inner)
            | TLExpr::Sin(inner)
            | TLExpr::Cos(inner)
            | TLExpr::Tan(inner) => {
                profile.arithmetic_count += 1;
                Self::analyze_recursive(inner, depth + 1, profile);
            }
            TLExpr::Let {
                var: _,
                value,
                body,
            } => {
                Self::analyze_recursive(value, depth + 1, profile);
                Self::analyze_recursive(body, depth + 1, profile);
            }
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                Self::analyze_recursive(condition, depth + 1, profile);
                Self::analyze_recursive(then_branch, depth + 1, profile);
                Self::analyze_recursive(else_branch, depth + 1, profile);
            }
            TLExpr::Score(operand) => {
                Self::analyze_recursive(operand, depth + 1, profile);
            }
            TLExpr::Aggregate { body, .. } => {
                Self::analyze_recursive(body, depth + 1, profile);
            }

            // Modal/temporal logic operators - not yet implemented, pass through with recursion
            TLExpr::Box(inner)
            | TLExpr::Diamond(inner)
            | TLExpr::Next(inner)
            | TLExpr::Eventually(inner)
            | TLExpr::Always(inner) => {
                Self::analyze_recursive(inner, depth + 1, profile);
            }
            TLExpr::Until { before, after } => {
                Self::analyze_recursive(before, depth + 1, profile);
                Self::analyze_recursive(after, depth + 1, profile);
            }

            // Fuzzy logic operators
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                Self::analyze_recursive(left, depth + 1, profile);
                Self::analyze_recursive(right, depth + 1, profile);
            }
            TLExpr::FuzzyNot { expr, .. } => {
                Self::analyze_recursive(expr, depth + 1, profile);
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                Self::analyze_recursive(premise, depth + 1, profile);
                Self::analyze_recursive(conclusion, depth + 1, profile);
            }
            TLExpr::SoftExists { body, .. } | TLExpr::SoftForAll { body, .. } => {
                profile.exists_count += 1;
                Self::analyze_recursive(body, depth + 1, profile);
            }
            TLExpr::WeightedRule { rule, .. } => {
                Self::analyze_recursive(rule, depth + 1, profile);
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_prob, expr) in alternatives {
                    Self::analyze_recursive(expr, depth + 1, profile);
                }
            }
            TLExpr::Release { released, releaser }
            | TLExpr::WeakUntil {
                before: released,
                after: releaser,
            }
            | TLExpr::StrongRelease { released, releaser } => {
                Self::analyze_recursive(released, depth + 1, profile);
                Self::analyze_recursive(releaser, depth + 1, profile);
            }

            TLExpr::Pred { .. } | TLExpr::Constant(_) => {
                // Leaf nodes
            }
        }
    }

    fn contains_quantifier(expr: &TLExpr) -> bool {
        match expr {
            TLExpr::Exists { .. } | TLExpr::ForAll { .. } => true,
            TLExpr::And(l, r)
            | TLExpr::Or(l, r)
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
            | TLExpr::Gte(l, r) => Self::contains_quantifier(l) || Self::contains_quantifier(r),
            TLExpr::Imply(premise, conclusion) => {
                Self::contains_quantifier(premise) || Self::contains_quantifier(conclusion)
            }
            TLExpr::Not(inner)
            | TLExpr::Score(inner)
            | TLExpr::Abs(inner)
            | TLExpr::Floor(inner)
            | TLExpr::Ceil(inner)
            | TLExpr::Round(inner)
            | TLExpr::Sqrt(inner)
            | TLExpr::Exp(inner)
            | TLExpr::Log(inner)
            | TLExpr::Sin(inner)
            | TLExpr::Cos(inner)
            | TLExpr::Tan(inner) => Self::contains_quantifier(inner),
            TLExpr::Let {
                var: _,
                value,
                body,
            } => Self::contains_quantifier(value) || Self::contains_quantifier(body),
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                Self::contains_quantifier(condition)
                    || Self::contains_quantifier(then_branch)
                    || Self::contains_quantifier(else_branch)
            }

            // Modal/temporal logic operators
            TLExpr::Box(inner)
            | TLExpr::Diamond(inner)
            | TLExpr::Next(inner)
            | TLExpr::Eventually(inner)
            | TLExpr::Always(inner) => Self::contains_quantifier(inner),
            TLExpr::Until { before, after } => {
                Self::contains_quantifier(before) || Self::contains_quantifier(after)
            }

            // Fuzzy logic operators
            TLExpr::SoftExists { .. } | TLExpr::SoftForAll { .. } => true,
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                Self::contains_quantifier(left) || Self::contains_quantifier(right)
            }
            TLExpr::FuzzyNot { expr, .. } => Self::contains_quantifier(expr),
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => Self::contains_quantifier(premise) || Self::contains_quantifier(conclusion),
            TLExpr::WeightedRule { rule, .. } => Self::contains_quantifier(rule),
            TLExpr::ProbabilisticChoice { alternatives } => alternatives
                .iter()
                .any(|(_prob, expr)| Self::contains_quantifier(expr)),
            TLExpr::Release { released, releaser }
            | TLExpr::WeakUntil {
                before: released,
                after: releaser,
            }
            | TLExpr::StrongRelease { released, releaser } => {
                Self::contains_quantifier(released) || Self::contains_quantifier(releaser)
            }

            TLExpr::Pred { .. } | TLExpr::Constant(_) | TLExpr::Aggregate { .. } => false,
        }
    }

    fn contains_logical_op(expr: &TLExpr) -> bool {
        match expr {
            TLExpr::And(..)
            | TLExpr::Or(..)
            | TLExpr::Not(..)
            | TLExpr::Exists { .. }
            | TLExpr::ForAll { .. }
            | TLExpr::Imply(..) => true,
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
            | TLExpr::Gte(l, r) => Self::contains_logical_op(l) || Self::contains_logical_op(r),
            TLExpr::Score(operand)
            | TLExpr::Abs(operand)
            | TLExpr::Floor(operand)
            | TLExpr::Ceil(operand)
            | TLExpr::Round(operand)
            | TLExpr::Sqrt(operand)
            | TLExpr::Exp(operand)
            | TLExpr::Log(operand)
            | TLExpr::Sin(operand)
            | TLExpr::Cos(operand)
            | TLExpr::Tan(operand) => Self::contains_logical_op(operand),
            TLExpr::Let {
                var: _,
                value,
                body,
            } => Self::contains_logical_op(value) || Self::contains_logical_op(body),
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                Self::contains_logical_op(condition)
                    || Self::contains_logical_op(then_branch)
                    || Self::contains_logical_op(else_branch)
            }

            // Modal/temporal logic operators - these are logical operators
            TLExpr::Box(..)
            | TLExpr::Diamond(..)
            | TLExpr::Next(..)
            | TLExpr::Eventually(..)
            | TLExpr::Always(..)
            | TLExpr::Until { .. } => true,

            // Fuzzy logic operators - these are logical operators
            TLExpr::TNorm { .. }
            | TLExpr::TCoNorm { .. }
            | TLExpr::FuzzyNot { .. }
            | TLExpr::FuzzyImplication { .. }
            | TLExpr::SoftExists { .. }
            | TLExpr::SoftForAll { .. }
            | TLExpr::Release { .. }
            | TLExpr::WeakUntil { .. }
            | TLExpr::StrongRelease { .. } => true,
            TLExpr::WeightedRule { rule, .. } => Self::contains_logical_op(rule),
            TLExpr::ProbabilisticChoice { alternatives } => alternatives
                .iter()
                .any(|(_prob, expr)| Self::contains_logical_op(expr)),

            TLExpr::Pred { .. } | TLExpr::Constant(_) | TLExpr::Aggregate { .. } => false,
        }
    }

    fn contains_arithmetic_op(expr: &TLExpr) -> bool {
        match expr {
            TLExpr::Add(..)
            | TLExpr::Sub(..)
            | TLExpr::Mul(..)
            | TLExpr::Div(..)
            | TLExpr::Pow(..)
            | TLExpr::Mod(..)
            | TLExpr::Min(..)
            | TLExpr::Max(..)
            | TLExpr::Abs(..)
            | TLExpr::Floor(..)
            | TLExpr::Ceil(..)
            | TLExpr::Round(..)
            | TLExpr::Sqrt(..)
            | TLExpr::Exp(..)
            | TLExpr::Log(..)
            | TLExpr::Sin(..)
            | TLExpr::Cos(..)
            | TLExpr::Tan(..) => true,
            TLExpr::And(l, r)
            | TLExpr::Or(l, r)
            | TLExpr::Eq(l, r)
            | TLExpr::Lt(l, r)
            | TLExpr::Gt(l, r)
            | TLExpr::Lte(l, r)
            | TLExpr::Gte(l, r) => {
                Self::contains_arithmetic_op(l) || Self::contains_arithmetic_op(r)
            }
            TLExpr::Imply(premise, conclusion) => {
                Self::contains_arithmetic_op(premise) || Self::contains_arithmetic_op(conclusion)
            }
            TLExpr::Not(inner) | TLExpr::Score(inner) => Self::contains_arithmetic_op(inner),
            TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
                Self::contains_arithmetic_op(body)
            }
            TLExpr::Let {
                var: _,
                value,
                body,
            } => Self::contains_arithmetic_op(value) || Self::contains_arithmetic_op(body),
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                Self::contains_arithmetic_op(condition)
                    || Self::contains_arithmetic_op(then_branch)
                    || Self::contains_arithmetic_op(else_branch)
            }

            // Modal/temporal logic operators - check recursively
            TLExpr::Box(inner)
            | TLExpr::Diamond(inner)
            | TLExpr::Next(inner)
            | TLExpr::Eventually(inner)
            | TLExpr::Always(inner) => Self::contains_arithmetic_op(inner),
            TLExpr::Until { before, after } => {
                Self::contains_arithmetic_op(before) || Self::contains_arithmetic_op(after)
            }

            // Fuzzy logic operators - check recursively
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                Self::contains_arithmetic_op(left) || Self::contains_arithmetic_op(right)
            }
            TLExpr::FuzzyNot { expr, .. } => Self::contains_arithmetic_op(expr),
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => Self::contains_arithmetic_op(premise) || Self::contains_arithmetic_op(conclusion),
            TLExpr::SoftExists { body, .. } | TLExpr::SoftForAll { body, .. } => {
                Self::contains_arithmetic_op(body)
            }
            TLExpr::WeightedRule { rule, .. } => Self::contains_arithmetic_op(rule),
            TLExpr::ProbabilisticChoice { alternatives } => alternatives
                .iter()
                .any(|(_prob, expr)| Self::contains_arithmetic_op(expr)),
            TLExpr::Release { released, releaser }
            | TLExpr::WeakUntil {
                before: released,
                after: releaser,
            }
            | TLExpr::StrongRelease { released, releaser } => {
                Self::contains_arithmetic_op(released) || Self::contains_arithmetic_op(releaser)
            }

            TLExpr::Pred { .. } | TLExpr::Constant(_) | TLExpr::Aggregate { .. } => false,
        }
    }

    /// Calculate a complexity score for the expression.
    pub fn complexity_score(&self) -> f64 {
        let op_count = self.and_count
            + self.or_count
            + self.not_count
            + self.exists_count
            + self.forall_count
            + self.implication_count
            + self.arithmetic_count
            + self.comparison_count;

        let base_score = op_count as f64;
        let depth_penalty = self.max_depth as f64 * 0.5;
        let quantifier_penalty = if self.has_nested_quantifiers {
            10.0
        } else {
            0.0
        };
        let mixed_penalty = if self.has_mixed_operations { 5.0 } else { 0.0 };

        base_score + depth_penalty + quantifier_penalty + mixed_penalty
    }
}

/// Strategy recommendation with confidence score.
#[derive(Debug, Clone)]
pub struct StrategyRecommendation {
    /// Recommended configuration
    pub config: CompilationConfig,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Explanation for the recommendation
    pub rationale: String,
    /// Alternative configurations (if any)
    pub alternatives: Vec<(CompilationConfig, f64, String)>,
}

/// Analyze an expression and recommend compilation strategy.
///
/// # Examples
///
/// ```
/// use tensorlogic_compiler::passes::recommend_strategy;
/// use tensorlogic_compiler::passes::OptimizationGoal;
/// use tensorlogic_ir::{TLExpr, Term};
///
/// // Expression with nested quantifiers
/// let expr = TLExpr::exists(
///     "y",
///     "Person",
///     TLExpr::forall(
///         "z",
///         "Person",
///         TLExpr::imply(
///             TLExpr::pred("knows", vec![Term::var("y"), Term::var("z")]),
///             TLExpr::pred("likes", vec![Term::var("x"), Term::var("z")]),
///         ),
///     ),
/// );
///
/// let recommendation = recommend_strategy(&expr, OptimizationGoal::Differentiable);
/// assert!(recommendation.confidence > 0.5);
/// println!("Recommendation: {}", recommendation.rationale);
/// ```
pub fn recommend_strategy(expr: &TLExpr, goal: OptimizationGoal) -> StrategyRecommendation {
    let profile = ExpressionProfile::analyze(expr);

    match goal {
        OptimizationGoal::Differentiable => recommend_for_differentiable(&profile),
        OptimizationGoal::DiscreteReasoning => recommend_for_discrete(&profile),
        OptimizationGoal::Performance => recommend_for_performance(&profile),
        OptimizationGoal::Balanced => recommend_for_balanced(&profile),
    }
}

fn recommend_for_differentiable(profile: &ExpressionProfile) -> StrategyRecommendation {
    let complexity = profile.complexity_score();

    // For expressions with nested quantifiers, use Łukasiewicz for better gradient stability
    if profile.has_nested_quantifiers {
        StrategyRecommendation {
            config: CompilationConfig::fuzzy_lukasiewicz(),
            confidence: 0.85,
            rationale: format!(
                "Łukasiewicz fuzzy logic recommended for nested quantifiers \
                 (complexity: {:.1}). Fully differentiable with stable gradients.",
                complexity
            ),
            alternatives: vec![(
                CompilationConfig::soft_differentiable(),
                0.75,
                "Standard soft differentiable config (simpler but may have gradient issues)"
                    .to_string(),
            )],
        }
    }
    // For expressions with mixed arithmetic and logic
    else if profile.has_mixed_operations {
        StrategyRecommendation {
            config: CompilationConfig::soft_differentiable(),
            confidence: 0.9,
            rationale:
                "Soft differentiable config recommended for mixed arithmetic-logic expressions. \
                       Provides smooth gradients for both operation types."
                    .to_string(),
            alternatives: vec![(
                CompilationConfig::fuzzy_product(),
                0.7,
                "Product fuzzy logic (alternative probabilistic interpretation)".to_string(),
            )],
        }
    }
    // For simple expressions, standard soft config is best
    else if complexity < 10.0 {
        StrategyRecommendation {
            config: CompilationConfig::soft_differentiable(),
            confidence: 0.95,
            rationale: format!(
                "Standard soft differentiable config recommended for simple expression \
                 (complexity: {:.1}). Efficient and well-tested.",
                complexity
            ),
            alternatives: vec![],
        }
    }
    // Default to soft differentiable
    else {
        StrategyRecommendation {
            config: CompilationConfig::soft_differentiable(),
            confidence: 0.8,
            rationale: format!(
                "Standard soft differentiable config recommended (complexity: {:.1}).",
                complexity
            ),
            alternatives: vec![(
                CompilationConfig::fuzzy_lukasiewicz(),
                0.7,
                "Łukasiewicz fuzzy logic (alternative for more complex scenarios)".to_string(),
            )],
        }
    }
}

fn recommend_for_discrete(profile: &ExpressionProfile) -> StrategyRecommendation {
    // For expressions with negated quantifiers, use Gödel
    if profile.has_negated_quantifiers {
        StrategyRecommendation {
            config: CompilationConfig::fuzzy_godel(),
            confidence: 0.9,
            rationale: "Gödel fuzzy logic recommended for negated quantifiers. \
                       Provides crisp Boolean-like semantics while handling fuzzy values."
                .to_string(),
            alternatives: vec![(
                CompilationConfig::hard_boolean(),
                0.75,
                "Pure Boolean logic (strictly discrete but may lose gradient info)".to_string(),
            )],
        }
    }
    // For simple discrete reasoning
    else if profile.complexity_score() < 15.0 {
        StrategyRecommendation {
            config: CompilationConfig::hard_boolean(),
            confidence: 0.95,
            rationale: format!(
                "Hard Boolean logic recommended for simple discrete reasoning \
                 (complexity: {:.1}). Provides crisp true/false semantics.",
                profile.complexity_score()
            ),
            alternatives: vec![],
        }
    }
    // For complex discrete reasoning, use Gödel as it's more robust
    else {
        StrategyRecommendation {
            config: CompilationConfig::fuzzy_godel(),
            confidence: 0.85,
            rationale: format!(
                "Gödel fuzzy logic recommended for complex discrete reasoning \
                 (complexity: {:.1}). More robust than pure Boolean for edge cases.",
                profile.complexity_score()
            ),
            alternatives: vec![(
                CompilationConfig::hard_boolean(),
                0.7,
                "Hard Boolean logic (simpler but less robust)".to_string(),
            )],
        }
    }
}

fn recommend_for_performance(profile: &ExpressionProfile) -> StrategyRecommendation {
    let complexity = profile.complexity_score();

    // For simple expressions, use Product fuzzy (efficient)
    if complexity < 10.0 {
        StrategyRecommendation {
            config: CompilationConfig::fuzzy_product(),
            confidence: 0.9,
            rationale: format!(
                "Product fuzzy logic recommended for efficient execution \
                 (complexity: {:.1}). Uses fast multiplication instead of min/max.",
                complexity
            ),
            alternatives: vec![(
                CompilationConfig::soft_differentiable(),
                0.85,
                "Soft differentiable (similar performance, better gradients)".to_string(),
            )],
        }
    }
    // For complex expressions with many quantifiers, use probabilistic
    else if profile.exists_count + profile.forall_count > 3 {
        StrategyRecommendation {
            config: CompilationConfig::probabilistic(),
            confidence: 0.85,
            rationale: "Probabilistic config recommended for many quantifiers. \
                       Uses efficient mean reductions instead of sum/product."
                .to_string(),
            alternatives: vec![(
                CompilationConfig::fuzzy_product(),
                0.75,
                "Product fuzzy logic (alternative with product reductions)".to_string(),
            )],
        }
    }
    // Default to soft differentiable (good balance)
    else {
        StrategyRecommendation {
            config: CompilationConfig::soft_differentiable(),
            confidence: 0.8,
            rationale: format!(
                "Soft differentiable config recommended for balanced performance \
                 (complexity: {:.1}).",
                complexity
            ),
            alternatives: vec![(
                CompilationConfig::fuzzy_product(),
                0.75,
                "Product fuzzy logic (slight performance edge)".to_string(),
            )],
        }
    }
}

fn recommend_for_balanced(profile: &ExpressionProfile) -> StrategyRecommendation {
    let complexity = profile.complexity_score();

    // For very simple expressions, use soft differentiable
    if complexity < 8.0 {
        StrategyRecommendation {
            config: CompilationConfig::soft_differentiable(),
            confidence: 0.9,
            rationale: format!(
                "Soft differentiable config recommended for simple balanced use \
                 (complexity: {:.1}).",
                complexity
            ),
            alternatives: vec![],
        }
    }
    // For moderately complex, use Łukasiewicz (good balance)
    else if complexity < 25.0 {
        StrategyRecommendation {
            config: CompilationConfig::fuzzy_lukasiewicz(),
            confidence: 0.85,
            rationale: format!(
                "Łukasiewicz fuzzy logic recommended for balanced complexity \
                 (complexity: {:.1}). Good mix of differentiability and accuracy.",
                complexity
            ),
            alternatives: vec![
                (
                    CompilationConfig::soft_differentiable(),
                    0.75,
                    "Soft differentiable (simpler, slightly better performance)".to_string(),
                ),
                (
                    CompilationConfig::fuzzy_godel(),
                    0.7,
                    "Gödel fuzzy logic (more discrete semantics)".to_string(),
                ),
            ],
        }
    }
    // For very complex, use probabilistic (robust)
    else {
        StrategyRecommendation {
            config: CompilationConfig::probabilistic(),
            confidence: 0.8,
            rationale: format!(
                "Probabilistic config recommended for complex balanced reasoning \
                 (complexity: {:.1}). Robust handling of uncertainty.",
                complexity
            ),
            alternatives: vec![(
                CompilationConfig::fuzzy_lukasiewicz(),
                0.75,
                "Łukasiewicz fuzzy logic (more structured reasoning)".to_string(),
            )],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_simple_predicate_profile() {
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let profile = ExpressionProfile::analyze(&expr);

        assert_eq!(profile.and_count, 0);
        assert_eq!(profile.exists_count, 0);
        assert_eq!(profile.max_depth, 0);
        assert!(!profile.has_nested_quantifiers);
    }

    #[test]
    fn test_and_or_profile() {
        let expr = TLExpr::And(
            Box::new(TLExpr::pred("p", vec![Term::var("x")])),
            Box::new(TLExpr::Or(
                Box::new(TLExpr::pred("q", vec![Term::var("y")])),
                Box::new(TLExpr::pred("r", vec![Term::var("z")])),
            )),
        );
        let profile = ExpressionProfile::analyze(&expr);

        assert_eq!(profile.and_count, 1);
        assert_eq!(profile.or_count, 1);
        assert_eq!(profile.max_depth, 2);
    }

    #[test]
    fn test_nested_quantifiers_profile() {
        let expr = TLExpr::exists(
            "y",
            "D",
            TLExpr::forall(
                "z",
                "D",
                TLExpr::pred("p", vec![Term::var("x"), Term::var("y"), Term::var("z")]),
            ),
        );
        let profile = ExpressionProfile::analyze(&expr);

        assert_eq!(profile.exists_count, 1);
        assert_eq!(profile.forall_count, 1);
        assert!(profile.has_nested_quantifiers);
        assert_eq!(profile.max_depth, 2);
    }

    #[test]
    fn test_negated_quantifier_profile() {
        let expr = TLExpr::Not(Box::new(TLExpr::exists(
            "y",
            "D",
            TLExpr::pred("p", vec![Term::var("x"), Term::var("y")]),
        )));
        let profile = ExpressionProfile::analyze(&expr);

        assert_eq!(profile.not_count, 1);
        assert_eq!(profile.exists_count, 1);
        assert!(profile.has_negated_quantifiers);
    }

    #[test]
    fn test_mixed_operations_profile() {
        let expr = TLExpr::And(
            Box::new(TLExpr::pred("p", vec![Term::var("x")])),
            Box::new(TLExpr::Add(
                Box::new(TLExpr::Constant(1.0)),
                Box::new(TLExpr::Constant(2.0)),
            )),
        );
        let profile = ExpressionProfile::analyze(&expr);

        assert_eq!(profile.arithmetic_count, 1);
        assert!(profile.has_mixed_operations);
    }

    #[test]
    fn test_recommend_simple_differentiable() {
        let expr = TLExpr::And(
            Box::new(TLExpr::pred("p", vec![Term::var("x")])),
            Box::new(TLExpr::pred("q", vec![Term::var("y")])),
        );

        let rec = recommend_strategy(&expr, OptimizationGoal::Differentiable);
        assert_eq!(rec.config, CompilationConfig::soft_differentiable());
        assert!(rec.confidence > 0.8);
    }

    #[test]
    fn test_recommend_complex_differentiable() {
        let expr = TLExpr::exists(
            "y",
            "D",
            TLExpr::forall(
                "z",
                "D",
                TLExpr::And(
                    Box::new(TLExpr::pred("p", vec![Term::var("y")])),
                    Box::new(TLExpr::pred("q", vec![Term::var("z")])),
                ),
            ),
        );

        let rec = recommend_strategy(&expr, OptimizationGoal::Differentiable);
        // Should recommend Łukasiewicz for nested quantifiers
        assert_eq!(rec.config, CompilationConfig::fuzzy_lukasiewicz());
        assert!(rec.confidence > 0.8);
    }

    #[test]
    fn test_recommend_discrete() {
        let expr = TLExpr::And(
            Box::new(TLExpr::pred("p", vec![Term::var("x")])),
            Box::new(TLExpr::pred("q", vec![Term::var("y")])),
        );

        let rec = recommend_strategy(&expr, OptimizationGoal::DiscreteReasoning);
        assert_eq!(rec.config, CompilationConfig::hard_boolean());
        assert!(rec.confidence > 0.9);
    }

    #[test]
    fn test_recommend_performance() {
        let expr = TLExpr::And(
            Box::new(TLExpr::pred("p", vec![Term::var("x")])),
            Box::new(TLExpr::pred("q", vec![Term::var("y")])),
        );

        let rec = recommend_strategy(&expr, OptimizationGoal::Performance);
        assert_eq!(rec.config, CompilationConfig::fuzzy_product());
        assert!(rec.confidence > 0.8);
    }

    #[test]
    fn test_recommend_balanced() {
        let expr = TLExpr::And(
            Box::new(TLExpr::pred("p", vec![Term::var("x")])),
            Box::new(TLExpr::pred("q", vec![Term::var("y")])),
        );

        let rec = recommend_strategy(&expr, OptimizationGoal::Balanced);
        assert_eq!(rec.config, CompilationConfig::soft_differentiable());
        assert!(rec.confidence > 0.8);
    }

    #[test]
    fn test_complexity_score() {
        // Simple expression
        let simple = TLExpr::pred("p", vec![Term::var("x")]);
        let simple_profile = ExpressionProfile::analyze(&simple);
        let simple_score = simple_profile.complexity_score();
        assert!(simple_score < 5.0);

        // Complex expression
        let complex = TLExpr::exists(
            "y",
            "D",
            TLExpr::forall(
                "z",
                "D",
                TLExpr::And(
                    Box::new(TLExpr::Or(
                        Box::new(TLExpr::pred("p", vec![Term::var("y")])),
                        Box::new(TLExpr::pred("q", vec![Term::var("z")])),
                    )),
                    Box::new(TLExpr::Not(Box::new(TLExpr::pred(
                        "r",
                        vec![Term::var("x")],
                    )))),
                ),
            ),
        );
        let complex_profile = ExpressionProfile::analyze(&complex);
        let complex_score = complex_profile.complexity_score();
        assert!(complex_score > simple_score);
        assert!(complex_score > 15.0); // Has nested quantifiers penalty
    }
}
