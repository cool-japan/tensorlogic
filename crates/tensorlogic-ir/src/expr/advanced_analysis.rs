//! Advanced expression analysis for operator counting and complexity metrics.
//!
//! This module provides detailed analysis capabilities for TensorLogic expressions,
//! including operator type classification, complexity metrics, and pattern detection.

use std::collections::HashMap;

use super::TLExpr;

/// Detailed operator counts categorized by type.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperatorCounts {
    /// Total number of operators
    pub total: usize,
    /// Basic logical operators (And, Or, Not, Imply)
    pub logical: usize,
    /// Arithmetic operators (Add, Sub, Mul, Div, Pow, Mod, Min, Max)
    pub arithmetic: usize,
    /// Comparison operators (Eq, Lt, Gt, Lte, Gte)
    pub comparison: usize,
    /// Unary mathematical functions (Abs, Floor, Ceil, Round, Sqrt, Exp, Log, Sin, Cos, Tan)
    pub mathematical: usize,
    /// Quantifiers (Exists, ForAll, SoftExists, SoftForAll)
    pub quantifiers: usize,
    /// Modal logic operators (Box, Diamond)
    pub modal: usize,
    /// Temporal logic operators (Next, Eventually, Always, Until, Release, WeakUntil, StrongRelease)
    pub temporal: usize,
    /// Fuzzy logic operators (TNorm, TCoNorm, FuzzyNot, FuzzyImplication)
    pub fuzzy: usize,
    /// Probabilistic operators (WeightedRule, ProbabilisticChoice)
    pub probabilistic: usize,
    /// Aggregation operators
    pub aggregation: usize,
    /// Control flow (IfThenElse, Let)
    pub control_flow: usize,
    /// Predicates
    pub predicates: usize,
    /// Constants
    pub constants: usize,
}

impl OperatorCounts {
    /// Create a new empty operator count.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute operator counts from an expression.
    pub fn from_expr(expr: &TLExpr) -> Self {
        let mut counts = Self::new();
        counts.count_recursive(expr);
        counts
    }

    /// Recursively count operators in an expression.
    fn count_recursive(&mut self, expr: &TLExpr) {
        self.total += 1;

        match expr {
            // Predicates
            TLExpr::Pred { .. } => {
                self.predicates += 1;
            }

            // Constants
            TLExpr::Constant(_) => {
                self.constants += 1;
            }

            // Basic logical operators
            TLExpr::And(l, r) | TLExpr::Or(l, r) | TLExpr::Imply(l, r) => {
                self.logical += 1;
                self.count_recursive(l);
                self.count_recursive(r);
            }
            TLExpr::Not(e) | TLExpr::Score(e) => {
                self.logical += 1;
                self.count_recursive(e);
            }

            // Arithmetic operators
            TLExpr::Add(l, r)
            | TLExpr::Sub(l, r)
            | TLExpr::Mul(l, r)
            | TLExpr::Div(l, r)
            | TLExpr::Pow(l, r)
            | TLExpr::Mod(l, r)
            | TLExpr::Min(l, r)
            | TLExpr::Max(l, r) => {
                self.arithmetic += 1;
                self.count_recursive(l);
                self.count_recursive(r);
            }

            // Comparison operators
            TLExpr::Eq(l, r)
            | TLExpr::Lt(l, r)
            | TLExpr::Gt(l, r)
            | TLExpr::Lte(l, r)
            | TLExpr::Gte(l, r) => {
                self.comparison += 1;
                self.count_recursive(l);
                self.count_recursive(r);
            }

            // Mathematical functions
            TLExpr::Abs(e)
            | TLExpr::Floor(e)
            | TLExpr::Ceil(e)
            | TLExpr::Round(e)
            | TLExpr::Sqrt(e)
            | TLExpr::Exp(e)
            | TLExpr::Log(e)
            | TLExpr::Sin(e)
            | TLExpr::Cos(e)
            | TLExpr::Tan(e) => {
                self.mathematical += 1;
                self.count_recursive(e);
            }

            // Quantifiers
            TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
                self.quantifiers += 1;
                self.count_recursive(body);
            }
            TLExpr::SoftExists { body, .. } | TLExpr::SoftForAll { body, .. } => {
                self.quantifiers += 1;
                self.probabilistic += 1; // Also counts as probabilistic
                self.count_recursive(body);
            }

            // Modal logic
            TLExpr::Box(e) | TLExpr::Diamond(e) => {
                self.modal += 1;
                self.count_recursive(e);
            }

            // Temporal logic
            TLExpr::Next(e) | TLExpr::Eventually(e) | TLExpr::Always(e) => {
                self.temporal += 1;
                self.count_recursive(e);
            }
            TLExpr::Until { before, after }
            | TLExpr::Release {
                released: before,
                releaser: after,
            }
            | TLExpr::WeakUntil { before, after }
            | TLExpr::StrongRelease {
                released: before,
                releaser: after,
            } => {
                self.temporal += 1;
                self.count_recursive(before);
                self.count_recursive(after);
            }

            // Fuzzy logic
            TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
                self.fuzzy += 1;
                self.count_recursive(left);
                self.count_recursive(right);
            }
            TLExpr::FuzzyNot { expr, .. } => {
                self.fuzzy += 1;
                self.count_recursive(expr);
            }
            TLExpr::FuzzyImplication {
                premise,
                conclusion,
                ..
            } => {
                self.fuzzy += 1;
                self.count_recursive(premise);
                self.count_recursive(conclusion);
            }

            // Probabilistic operators
            TLExpr::WeightedRule { rule, .. } => {
                self.probabilistic += 1;
                self.count_recursive(rule);
            }
            TLExpr::ProbabilisticChoice { alternatives } => {
                self.probabilistic += 1;
                for (_, e) in alternatives {
                    self.count_recursive(e);
                }
            }

            // Aggregation
            TLExpr::Aggregate { body, .. } => {
                self.aggregation += 1;
                self.count_recursive(body);
            }

            // Control flow
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.control_flow += 1;
                self.count_recursive(condition);
                self.count_recursive(then_branch);
                self.count_recursive(else_branch);
            }
            TLExpr::Let { value, body, .. } => {
                self.control_flow += 1;
                self.count_recursive(value);
                self.count_recursive(body);
            }
        }
    }
}

/// Complexity metrics for an expression.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexityMetrics {
    /// Maximum depth of the expression tree
    pub max_depth: usize,
    /// Average depth of leaf nodes
    pub avg_depth: f64,
    /// Total number of nodes
    pub node_count: usize,
    /// Number of leaf nodes (predicates and constants)
    pub leaf_count: usize,
    /// Branching factor (average number of children per non-leaf node)
    pub branching_factor: f64,
    /// Cyclomatic complexity (number of decision points + 1)
    pub cyclomatic_complexity: usize,
    /// Quantifier depth (maximum nesting level of quantifiers)
    pub quantifier_depth: usize,
    /// Modal depth (maximum nesting level of modal operators)
    pub modal_depth: usize,
    /// Temporal depth (maximum nesting level of temporal operators)
    pub temporal_depth: usize,
}

impl ComplexityMetrics {
    /// Compute complexity metrics from an expression.
    pub fn from_expr(expr: &TLExpr) -> Self {
        let mut metrics = Self {
            max_depth: 0,
            avg_depth: 0.0,
            node_count: 0,
            leaf_count: 0,
            branching_factor: 0.0,
            cyclomatic_complexity: 1, // Start with 1
            quantifier_depth: 0,
            modal_depth: 0,
            temporal_depth: 0,
        };

        let mut depth_sum = 0;
        let mut non_leaf_count = 0;
        let mut child_count = 0;

        metrics.compute_recursive(
            expr,
            0,
            &mut depth_sum,
            &mut non_leaf_count,
            &mut child_count,
            0,
            0,
            0,
        );

        // Compute averages
        if metrics.leaf_count > 0 {
            metrics.avg_depth = depth_sum as f64 / metrics.leaf_count as f64;
        }
        if non_leaf_count > 0 {
            metrics.branching_factor = child_count as f64 / non_leaf_count as f64;
        }

        metrics
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_recursive(
        &mut self,
        expr: &TLExpr,
        depth: usize,
        depth_sum: &mut usize,
        non_leaf_count: &mut usize,
        child_count: &mut usize,
        quantifier_depth: usize,
        modal_depth: usize,
        temporal_depth: usize,
    ) {
        self.node_count += 1;
        self.max_depth = self.max_depth.max(depth);
        self.quantifier_depth = self.quantifier_depth.max(quantifier_depth);
        self.modal_depth = self.modal_depth.max(modal_depth);
        self.temporal_depth = self.temporal_depth.max(temporal_depth);

        match expr {
            // Leaves
            TLExpr::Pred { .. } | TLExpr::Constant(_) => {
                self.leaf_count += 1;
                *depth_sum += depth;
            }

            // Decision points (add to cyclomatic complexity)
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.cyclomatic_complexity += 1; // Decision point
                *non_leaf_count += 1;
                *child_count += 3;
                self.compute_recursive(
                    condition,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
                self.compute_recursive(
                    then_branch,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
                self.compute_recursive(
                    else_branch,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
            }

            // Logical operators (decision points)
            TLExpr::And(l, r) | TLExpr::Or(l, r) => {
                self.cyclomatic_complexity += 1; // Decision point
                *non_leaf_count += 1;
                *child_count += 2;
                self.compute_recursive(
                    l,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
                self.compute_recursive(
                    r,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
            }

            // Quantifiers (increase quantifier depth)
            TLExpr::Exists { body, .. }
            | TLExpr::ForAll { body, .. }
            | TLExpr::SoftExists { body, .. }
            | TLExpr::SoftForAll { body, .. } => {
                *non_leaf_count += 1;
                *child_count += 1;
                self.compute_recursive(
                    body,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth + 1,
                    modal_depth,
                    temporal_depth,
                );
            }

            // Modal operators (increase modal depth)
            TLExpr::Box(e) | TLExpr::Diamond(e) => {
                *non_leaf_count += 1;
                *child_count += 1;
                self.compute_recursive(
                    e,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth + 1,
                    temporal_depth,
                );
            }

            // Temporal operators (increase temporal depth)
            TLExpr::Next(e) | TLExpr::Eventually(e) | TLExpr::Always(e) => {
                *non_leaf_count += 1;
                *child_count += 1;
                self.compute_recursive(
                    e,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth + 1,
                );
            }

            // Binary temporal operators
            TLExpr::Until { before, after }
            | TLExpr::Release {
                released: before,
                releaser: after,
            }
            | TLExpr::WeakUntil { before, after }
            | TLExpr::StrongRelease {
                released: before,
                releaser: after,
            } => {
                *non_leaf_count += 1;
                *child_count += 2;
                self.compute_recursive(
                    before,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth + 1,
                );
                self.compute_recursive(
                    after,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth + 1,
                );
            }

            // Unary operators
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
            | TLExpr::Tan(e) => {
                *non_leaf_count += 1;
                *child_count += 1;
                self.compute_recursive(
                    e,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
            }

            // Binary operators
            TLExpr::Imply(l, r)
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
            | TLExpr::Gte(l, r) => {
                *non_leaf_count += 1;
                *child_count += 2;
                self.compute_recursive(
                    l,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
                self.compute_recursive(
                    r,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
            }

            // Fuzzy operators
            TLExpr::TNorm { left, right, .. }
            | TLExpr::TCoNorm { left, right, .. }
            | TLExpr::FuzzyImplication {
                premise: left,
                conclusion: right,
                ..
            } => {
                *non_leaf_count += 1;
                *child_count += 2;
                self.compute_recursive(
                    left,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
                self.compute_recursive(
                    right,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
            }

            TLExpr::FuzzyNot { expr, .. } => {
                *non_leaf_count += 1;
                *child_count += 1;
                self.compute_recursive(
                    expr,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
            }

            // Probabilistic operators
            TLExpr::WeightedRule { rule, .. } => {
                *non_leaf_count += 1;
                *child_count += 1;
                self.compute_recursive(
                    rule,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
            }

            TLExpr::ProbabilisticChoice { alternatives } => {
                self.cyclomatic_complexity += alternatives.len(); // Multiple decision points
                *non_leaf_count += 1;
                *child_count += alternatives.len();
                for (_, e) in alternatives {
                    self.compute_recursive(
                        e,
                        depth + 1,
                        depth_sum,
                        non_leaf_count,
                        child_count,
                        quantifier_depth,
                        modal_depth,
                        temporal_depth,
                    );
                }
            }

            // Aggregation
            TLExpr::Aggregate { body, .. } => {
                *non_leaf_count += 1;
                *child_count += 1;
                self.compute_recursive(
                    body,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
            }

            // Let binding
            TLExpr::Let { value, body, .. } => {
                *non_leaf_count += 1;
                *child_count += 2;
                self.compute_recursive(
                    value,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
                self.compute_recursive(
                    body,
                    depth + 1,
                    depth_sum,
                    non_leaf_count,
                    child_count,
                    quantifier_depth,
                    modal_depth,
                    temporal_depth,
                );
            }
        }
    }
}

/// Pattern detection results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatternAnalysis {
    /// Detected De Morgan's law patterns: ¬(A ∧ B) or ¬(A ∨ B)
    pub de_morgan_patterns: usize,
    /// Double negation patterns: ¬¬A
    pub double_negation: usize,
    /// Modal duality patterns: ◇P with ¬□¬P nearby
    pub modal_duality: usize,
    /// Temporal duality patterns: FP with ¬G¬P nearby
    pub temporal_duality: usize,
    /// Redundant quantifier patterns (nested same quantifiers)
    pub redundant_quantifiers: usize,
    /// Tautologies (always true expressions)
    pub tautologies: usize,
    /// Contradictions (always false expressions)
    pub contradictions: usize,
}

impl PatternAnalysis {
    /// Detect patterns in an expression.
    pub fn from_expr(expr: &TLExpr) -> Self {
        let mut analysis = Self {
            de_morgan_patterns: 0,
            double_negation: 0,
            modal_duality: 0,
            temporal_duality: 0,
            redundant_quantifiers: 0,
            tautologies: 0,
            contradictions: 0,
        };

        analysis.detect_recursive(expr, &HashMap::new());
        analysis
    }

    fn detect_recursive(&mut self, expr: &TLExpr, _context: &HashMap<String, TLExpr>) {
        match expr {
            // Double negation and De Morgan's patterns
            TLExpr::Not(e) => {
                // Double negation: ¬¬A
                if matches!(**e, TLExpr::Not(_)) {
                    self.double_negation += 1;
                }
                // De Morgan's patterns: ¬(A ∧ B) or ¬(A ∨ B)
                if matches!(**e, TLExpr::And(_, _) | TLExpr::Or(_, _)) {
                    self.de_morgan_patterns += 1;
                }
                self.detect_recursive(e, _context);
            }

            // Tautologies: A ∨ ¬A, A → A, A ∨ TRUE
            TLExpr::Or(l, r) => {
                if let (TLExpr::Not(inner), other) | (other, TLExpr::Not(inner)) = (&**l, &**r) {
                    if **inner == *other {
                        self.tautologies += 1;
                    }
                }
                if matches!(**l, TLExpr::Constant(v) if v >= 1.0)
                    || matches!(**r, TLExpr::Constant(v) if v >= 1.0)
                {
                    self.tautologies += 1;
                }
                self.detect_recursive(l, _context);
                self.detect_recursive(r, _context);
            }

            TLExpr::Imply(l, r) => {
                if l == r {
                    self.tautologies += 1;
                }
                self.detect_recursive(l, _context);
                self.detect_recursive(r, _context);
            }

            // Contradictions: A ∧ ¬A, A ∧ FALSE
            TLExpr::And(l, r) => {
                if let (TLExpr::Not(inner), other) | (other, TLExpr::Not(inner)) = (&**l, &**r) {
                    if **inner == *other {
                        self.contradictions += 1;
                    }
                }
                if matches!(**l, TLExpr::Constant(v) if v <= 0.0)
                    || matches!(**r, TLExpr::Constant(v) if v <= 0.0)
                {
                    self.contradictions += 1;
                }
                self.detect_recursive(l, _context);
                self.detect_recursive(r, _context);
            }

            // Redundant quantifiers: ∃x.∃x.P or ∀x.∀x.P
            TLExpr::Exists { var, body, .. } => {
                if let TLExpr::Exists { var: inner_var, .. } = &**body {
                    if var == inner_var {
                        self.redundant_quantifiers += 1;
                    }
                }
                self.detect_recursive(body, _context);
            }

            TLExpr::ForAll { var, body, .. } => {
                if let TLExpr::ForAll { var: inner_var, .. } = &**body {
                    if var == inner_var {
                        self.redundant_quantifiers += 1;
                    }
                }
                self.detect_recursive(body, _context);
            }

            // Modal duality: ◇P when we see ¬□¬P pattern
            TLExpr::Diamond(e) => {
                if let TLExpr::Not(inner) = &**e {
                    if let TLExpr::Box(inner_inner) = &**inner {
                        if matches!(**inner_inner, TLExpr::Not(_)) {
                            self.modal_duality += 1;
                        }
                    }
                }
                self.detect_recursive(e, _context);
            }

            // Temporal duality: FP when we see ¬G¬P pattern
            TLExpr::Eventually(e) => {
                if let TLExpr::Not(inner) = &**e {
                    if let TLExpr::Always(inner_inner) = &**inner {
                        if matches!(**inner_inner, TLExpr::Not(_)) {
                            self.temporal_duality += 1;
                        }
                    }
                }
                self.detect_recursive(e, _context);
            }

            // Recursive cases (And and Or are handled above for pattern detection)
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
            | TLExpr::Gte(l, r) => {
                self.detect_recursive(l, _context);
                self.detect_recursive(r, _context);
            }

            TLExpr::Score(e)
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
            | TLExpr::Next(e)
            | TLExpr::Always(e) => {
                self.detect_recursive(e, _context);
            }

            TLExpr::Until { before, after }
            | TLExpr::Release {
                released: before,
                releaser: after,
            }
            | TLExpr::WeakUntil { before, after }
            | TLExpr::StrongRelease {
                released: before,
                releaser: after,
            } => {
                self.detect_recursive(before, _context);
                self.detect_recursive(after, _context);
            }

            TLExpr::TNorm { left, right, .. }
            | TLExpr::TCoNorm { left, right, .. }
            | TLExpr::FuzzyImplication {
                premise: left,
                conclusion: right,
                ..
            } => {
                self.detect_recursive(left, _context);
                self.detect_recursive(right, _context);
            }

            TLExpr::FuzzyNot { expr, .. } => {
                self.detect_recursive(expr, _context);
            }

            TLExpr::SoftExists { body, .. }
            | TLExpr::SoftForAll { body, .. }
            | TLExpr::Aggregate { body, .. } => {
                self.detect_recursive(body, _context);
            }

            TLExpr::WeightedRule { rule, .. } => {
                self.detect_recursive(rule, _context);
            }

            TLExpr::ProbabilisticChoice { alternatives } => {
                for (_, e) in alternatives {
                    self.detect_recursive(e, _context);
                }
            }

            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.detect_recursive(condition, _context);
                self.detect_recursive(then_branch, _context);
                self.detect_recursive(else_branch, _context);
            }

            TLExpr::Let { value, body, .. } => {
                self.detect_recursive(value, _context);
                self.detect_recursive(body, _context);
            }

            // Leaves
            TLExpr::Pred { .. } | TLExpr::Constant(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_operator_counts_basic() {
        // Simple: P(x) AND Q(x)
        let expr = TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("x")]),
        );

        let counts = OperatorCounts::from_expr(&expr);
        assert_eq!(counts.total, 3); // AND + 2 predicates
        assert_eq!(counts.logical, 1);
        assert_eq!(counts.predicates, 2);
    }

    #[test]
    fn test_operator_counts_modal() {
        // □(P(x))
        let expr = TLExpr::modal_box(TLExpr::pred("P", vec![Term::var("x")]));

        let counts = OperatorCounts::from_expr(&expr);
        assert_eq!(counts.modal, 1);
        assert_eq!(counts.predicates, 1);
    }

    #[test]
    fn test_operator_counts_temporal() {
        // F(P(x))
        let expr = TLExpr::eventually(TLExpr::pred("P", vec![Term::var("x")]));

        let counts = OperatorCounts::from_expr(&expr);
        assert_eq!(counts.temporal, 1);
        assert_eq!(counts.predicates, 1);
    }

    #[test]
    fn test_operator_counts_fuzzy() {
        // P(x) ⊤_min Q(x)
        let expr = TLExpr::fuzzy_and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("x")]),
        );

        let counts = OperatorCounts::from_expr(&expr);
        assert_eq!(counts.fuzzy, 1);
        assert_eq!(counts.predicates, 2);
    }

    #[test]
    fn test_complexity_metrics_simple() {
        // P(x)
        let expr = TLExpr::pred("P", vec![Term::var("x")]);

        let metrics = ComplexityMetrics::from_expr(&expr);
        assert_eq!(metrics.max_depth, 0);
        assert_eq!(metrics.node_count, 1);
        assert_eq!(metrics.leaf_count, 1);
    }

    #[test]
    fn test_complexity_metrics_nested() {
        // ((P AND Q) OR (R AND S))
        let expr = TLExpr::or(
            TLExpr::and(
                TLExpr::pred("P", vec![Term::var("x")]),
                TLExpr::pred("Q", vec![Term::var("x")]),
            ),
            TLExpr::and(
                TLExpr::pred("R", vec![Term::var("x")]),
                TLExpr::pred("S", vec![Term::var("x")]),
            ),
        );

        let metrics = ComplexityMetrics::from_expr(&expr);
        assert_eq!(metrics.max_depth, 2); // OR at 0, AND at 1, predicates at 2
        assert_eq!(metrics.node_count, 7); // 1 OR + 2 AND + 4 predicates
        assert_eq!(metrics.leaf_count, 4);
        assert_eq!(metrics.cyclomatic_complexity, 4); // 1 + 3 decision points (OR, 2 ANDs)
    }

    #[test]
    fn test_complexity_quantifier_depth() {
        // ∃x.∀y.P(x,y)
        let expr = TLExpr::exists(
            "x",
            "D",
            TLExpr::forall(
                "y",
                "D",
                TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]),
            ),
        );

        let metrics = ComplexityMetrics::from_expr(&expr);
        assert_eq!(metrics.quantifier_depth, 2);
    }

    #[test]
    fn test_complexity_modal_depth() {
        // □◇P(x)
        let expr = TLExpr::modal_box(TLExpr::modal_diamond(TLExpr::pred(
            "P",
            vec![Term::var("x")],
        )));

        let metrics = ComplexityMetrics::from_expr(&expr);
        assert_eq!(metrics.modal_depth, 2);
    }

    #[test]
    fn test_pattern_double_negation() {
        // ¬¬P(x)
        let expr = TLExpr::negate(TLExpr::negate(TLExpr::pred("P", vec![Term::var("x")])));

        let patterns = PatternAnalysis::from_expr(&expr);
        assert_eq!(patterns.double_negation, 1);
    }

    #[test]
    fn test_pattern_de_morgan() {
        // ¬(P ∧ Q)
        let expr = TLExpr::negate(TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("x")]),
        ));

        let patterns = PatternAnalysis::from_expr(&expr);
        assert_eq!(patterns.de_morgan_patterns, 1);
    }

    #[test]
    fn test_pattern_tautology() {
        // P(x) ∨ ¬P(x)
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let expr = TLExpr::or(p.clone(), TLExpr::negate(p));

        let patterns = PatternAnalysis::from_expr(&expr);
        assert_eq!(patterns.tautologies, 1);
    }

    #[test]
    fn test_pattern_contradiction() {
        // P(x) ∧ ¬P(x)
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let expr = TLExpr::and(p.clone(), TLExpr::negate(p));

        let patterns = PatternAnalysis::from_expr(&expr);
        assert_eq!(patterns.contradictions, 1);
    }

    #[test]
    fn test_pattern_redundant_quantifier() {
        // ∃x.∃x.P(x) - redundant nesting of same variable
        let expr = TLExpr::exists(
            "x",
            "D",
            TLExpr::exists("x", "D", TLExpr::pred("P", vec![Term::var("x")])),
        );

        let patterns = PatternAnalysis::from_expr(&expr);
        assert_eq!(patterns.redundant_quantifiers, 1);
    }
}
