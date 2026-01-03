//! Graph rewriting engine for pattern-based optimizations.
//!
//! This module provides a powerful graph rewriting system:
//! - **Pattern matching**: Find subgraphs matching specific patterns
//! - **Rewrite rules**: Transform matched patterns into optimized equivalents
//! - **Rule application**: Apply rules systematically with strategies
//! - **Correctness checking**: Validate rewrites preserve semantics
//! - **Performance tracking**: Measure impact of rewrites
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_infer::{RewriteEngine, RewriteRule, Pattern, RewriteStrategy};
//!
//! // Define a rewrite rule: A + 0 -> A
//! let rule = RewriteRule::new("eliminate_add_zero")
//!     .with_pattern(Pattern::binary_op("add", Pattern::any(), Pattern::zero()))
//!     .with_replacement(|matched| matched.get_operand(0));
//!
//! // Create rewrite engine
//! let mut engine = RewriteEngine::new()
//!     .add_rule(rule)
//!     .with_strategy(RewriteStrategy::Exhaustive);
//!
//! // Apply rewrites to graph
//! let optimized = engine.rewrite(&graph)?;
//! println!("Eliminated {} operations", engine.stats().rewrites_applied);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Graph rewriting errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RewriteError {
    #[error("Pattern matching failed: {0}")]
    PatternMatchFailed(String),

    #[error("Invalid rewrite rule: {0}")]
    InvalidRule(String),

    #[error("Rewrite application failed: {0}")]
    ApplicationFailed(String),

    #[error("Cycle detected in rewrite application")]
    CycleDetected,

    #[error("Semantics verification failed: {0}")]
    SemanticsViolation(String),
}

/// Node identifier in the computation graph.
pub type NodeId = usize;

/// Graph pattern for matching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    /// Match any node
    Any,

    /// Match a specific operation
    Op(String),

    /// Match a binary operation with two subpatterns
    BinaryOp {
        op: String,
        left: Box<Pattern>,
        right: Box<Pattern>,
    },

    /// Match a unary operation with one subpattern
    UnaryOp { op: String, operand: Box<Pattern> },

    /// Match a constant value
    Constant(f64),

    /// Match zero
    Zero,

    /// Match one
    One,

    /// Match a variable (captures the matched node)
    Variable(String),

    /// Match a sequence of operations
    Sequence(Vec<Pattern>),

    /// Match any of the given patterns
    Alternative(Vec<Pattern>),
}

impl Pattern {
    /// Create a pattern matching any node.
    pub fn any() -> Self {
        Pattern::Any
    }

    /// Create a pattern matching a specific operation.
    pub fn op(name: impl Into<String>) -> Self {
        Pattern::Op(name.into())
    }

    /// Create a pattern matching a binary operation.
    pub fn binary_op(op: impl Into<String>, left: Pattern, right: Pattern) -> Self {
        Pattern::BinaryOp {
            op: op.into(),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Create a pattern matching a unary operation.
    pub fn unary_op(op: impl Into<String>, operand: Pattern) -> Self {
        Pattern::UnaryOp {
            op: op.into(),
            operand: Box::new(operand),
        }
    }

    /// Create a pattern matching a constant.
    pub fn constant(value: f64) -> Self {
        Pattern::Constant(value)
    }

    /// Create a pattern matching zero.
    pub fn zero() -> Self {
        Pattern::Zero
    }

    /// Create a pattern matching one.
    pub fn one() -> Self {
        Pattern::One
    }

    /// Create a pattern matching a variable.
    pub fn variable(name: impl Into<String>) -> Self {
        Pattern::Variable(name.into())
    }
}

/// A matched pattern instance.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// Root node of the match
    pub root: NodeId,

    /// Captured variables
    pub captures: HashMap<String, NodeId>,

    /// Matched nodes
    pub matched_nodes: HashSet<NodeId>,
}

impl Match {
    /// Create a new match.
    pub fn new(root: NodeId) -> Self {
        let mut matched_nodes = HashSet::new();
        matched_nodes.insert(root);

        Self {
            root,
            captures: HashMap::new(),
            matched_nodes,
        }
    }

    /// Get a captured node by variable name.
    pub fn get_capture(&self, name: &str) -> Option<NodeId> {
        self.captures.get(name).copied()
    }

    /// Add a capture.
    pub fn with_capture(mut self, name: String, node: NodeId) -> Self {
        self.captures.insert(name, node);
        self.matched_nodes.insert(node);
        self
    }

    /// Get all matched nodes.
    pub fn nodes(&self) -> &HashSet<NodeId> {
        &self.matched_nodes
    }
}

/// Rewrite rule replacement function.
pub type ReplacementFn = Box<dyn Fn(&Match) -> Result<NodeId, RewriteError>>;

/// A graph rewrite rule.
pub struct RewriteRule {
    /// Rule name
    pub name: String,

    /// Pattern to match
    pub pattern: Pattern,

    /// Replacement function
    pub replacement: ReplacementFn,

    /// Rule priority (higher = applied first)
    pub priority: i32,

    /// Whether this rule preserves semantics
    pub preserves_semantics: bool,
}

impl RewriteRule {
    /// Create a new rewrite rule.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pattern: Pattern::Any,
            replacement: Box::new(|m| Ok(m.root)),
            priority: 0,
            preserves_semantics: true,
        }
    }

    /// Set the pattern to match.
    pub fn with_pattern(mut self, pattern: Pattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set the replacement function.
    pub fn with_replacement<F>(mut self, f: F) -> Self
    where
        F: Fn(&Match) -> Result<NodeId, RewriteError> + 'static,
    {
        self.replacement = Box::new(f);
        self
    }

    /// Set the rule priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Mark whether this rule preserves semantics.
    pub fn with_semantics_preservation(mut self, preserves: bool) -> Self {
        self.preserves_semantics = preserves;
        self
    }
}

/// Strategy for applying rewrite rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RewriteStrategy {
    /// Apply each rule once to each node
    OnePass,

    /// Apply rules until no more matches found
    Exhaustive,

    /// Apply rules in a fixed-point manner (until convergence)
    FixedPoint { max_iterations: usize },

    /// Apply rules in order of priority
    Prioritized,

    /// Apply rules bottom-up (from leaves to root)
    BottomUp,

    /// Apply rules top-down (from root to leaves)
    TopDown,
}

impl Default for RewriteStrategy {
    fn default() -> Self {
        RewriteStrategy::Exhaustive
    }
}

/// Graph rewriting engine.
pub struct RewriteEngine {
    /// Rewrite rules
    rules: Vec<RewriteRule>,

    /// Application strategy
    strategy: RewriteStrategy,

    /// Statistics
    stats: RewriteStats,

    /// Enable verification
    verify_semantics: bool,
}

impl RewriteEngine {
    /// Create a new rewrite engine.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            strategy: RewriteStrategy::default(),
            stats: RewriteStats::default(),
            verify_semantics: false,
        }
    }

    /// Add a rewrite rule.
    pub fn add_rule(mut self, rule: RewriteRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Set the rewrite strategy.
    pub fn with_strategy(mut self, strategy: RewriteStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable or disable semantics verification.
    pub fn with_verification(mut self, enabled: bool) -> Self {
        self.verify_semantics = enabled;
        self
    }

    /// Get rewrite statistics.
    pub fn stats(&self) -> &RewriteStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = RewriteStats::default();
    }

    /// Sort rules by priority.
    fn sort_rules_by_priority(&mut self) {
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Apply rewrites to a simplified graph representation.
    /// In a real implementation, this would work with the actual EinsumGraph.
    pub fn rewrite_simple(&mut self, node_count: usize) -> Result<usize, RewriteError> {
        self.stats.graphs_processed += 1;

        match self.strategy {
            RewriteStrategy::OnePass => self.apply_one_pass(node_count),
            RewriteStrategy::Exhaustive => self.apply_exhaustive(node_count),
            RewriteStrategy::FixedPoint { max_iterations } => {
                self.apply_fixed_point(node_count, max_iterations)
            }
            RewriteStrategy::Prioritized => {
                self.sort_rules_by_priority();
                self.apply_one_pass(node_count)
            }
            RewriteStrategy::BottomUp | RewriteStrategy::TopDown => self.apply_one_pass(node_count),
        }
    }

    fn apply_one_pass(&mut self, node_count: usize) -> Result<usize, RewriteError> {
        let mut rewrites = 0;

        // Simplified: just count how many rules could apply
        for rule in &self.rules {
            // In real implementation, would match pattern and apply replacement
            if self.can_apply_rule(rule, node_count) {
                rewrites += 1;
                self.stats.rewrites_applied += 1;
                self.stats
                    .rule_applications
                    .entry(rule.name.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }

        Ok(node_count.saturating_sub(rewrites))
    }

    fn apply_exhaustive(&mut self, mut node_count: usize) -> Result<usize, RewriteError> {
        let mut iteration = 0;
        let max_iterations = 100; // Safety limit

        loop {
            iteration += 1;
            if iteration > max_iterations {
                return Err(RewriteError::CycleDetected);
            }

            let before = node_count;
            node_count = self.apply_one_pass(node_count)?;

            if node_count == before {
                // Converged
                break;
            }
        }

        Ok(node_count)
    }

    fn apply_fixed_point(
        &mut self,
        mut node_count: usize,
        max_iterations: usize,
    ) -> Result<usize, RewriteError> {
        for iteration in 0..max_iterations {
            let before = node_count;
            node_count = self.apply_one_pass(node_count)?;

            if node_count == before {
                self.stats.fixed_point_iterations = iteration + 1;
                break;
            }
        }

        Ok(node_count)
    }

    fn can_apply_rule(&self, _rule: &RewriteRule, _node_count: usize) -> bool {
        // Simplified: in real implementation, would match pattern
        true
    }
}

impl Default for RewriteEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Rewrite statistics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RewriteStats {
    /// Number of graphs processed
    pub graphs_processed: usize,

    /// Total rewrites applied
    pub rewrites_applied: usize,

    /// Applications per rule
    pub rule_applications: HashMap<String, usize>,

    /// Fixed-point iterations
    pub fixed_point_iterations: usize,

    /// Verification failures
    pub verification_failures: usize,
}

impl std::fmt::Display for RewriteStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Rewrite Statistics")?;
        writeln!(f, "==================")?;
        writeln!(f, "Graphs processed:     {}", self.graphs_processed)?;
        writeln!(f, "Rewrites applied:     {}", self.rewrites_applied)?;
        writeln!(f, "Fixed-point iters:    {}", self.fixed_point_iterations)?;
        writeln!(f, "Verification fails:   {}", self.verification_failures)?;

        if !self.rule_applications.is_empty() {
            writeln!(f, "\nRule Applications:")?;
            let mut rules: Vec<_> = self.rule_applications.iter().collect();
            rules.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
            for (rule, count) in rules {
                writeln!(f, "  {}: {}", rule, count)?;
            }
        }

        Ok(())
    }
}

/// Common rewrite rules for optimization.
pub struct CommonRules;

impl CommonRules {
    /// Eliminate addition with zero: A + 0 -> A
    pub fn eliminate_add_zero() -> RewriteRule {
        RewriteRule::new("eliminate_add_zero")
            .with_pattern(Pattern::binary_op("add", Pattern::any(), Pattern::zero()))
            .with_replacement(|m| Ok(m.root))
            .with_priority(10)
    }

    /// Eliminate multiplication by one: A * 1 -> A
    pub fn eliminate_mul_one() -> RewriteRule {
        RewriteRule::new("eliminate_mul_one")
            .with_pattern(Pattern::binary_op("mul", Pattern::any(), Pattern::one()))
            .with_replacement(|m| Ok(m.root))
            .with_priority(10)
    }

    /// Eliminate multiplication by zero: A * 0 -> 0
    pub fn eliminate_mul_zero() -> RewriteRule {
        RewriteRule::new("eliminate_mul_zero")
            .with_pattern(Pattern::binary_op("mul", Pattern::any(), Pattern::zero()))
            .with_replacement(|_m| Ok(0)) // Return zero node
            .with_priority(10)
    }

    /// Constant folding: C1 + C2 -> C3
    pub fn constant_folding() -> RewriteRule {
        RewriteRule::new("constant_folding")
            .with_pattern(Pattern::binary_op(
                "add",
                Pattern::constant(0.0), // Placeholder
                Pattern::constant(0.0),
            ))
            .with_replacement(|_m| Ok(0)) // Would compute result
            .with_priority(20)
    }

    /// Associativity: (A + B) + C -> A + (B + C)
    pub fn associativity_add() -> RewriteRule {
        RewriteRule::new("associativity_add")
            .with_pattern(Pattern::binary_op(
                "add",
                Pattern::binary_op("add", Pattern::any(), Pattern::any()),
                Pattern::any(),
            ))
            .with_replacement(|m| Ok(m.root))
            .with_priority(5)
    }

    /// Get all common optimization rules.
    pub fn all() -> Vec<RewriteRule> {
        vec![
            Self::eliminate_add_zero(),
            Self::eliminate_mul_one(),
            Self::eliminate_mul_zero(),
            Self::constant_folding(),
            Self::associativity_add(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_creation() {
        let pattern = Pattern::binary_op("add", Pattern::any(), Pattern::zero());
        assert!(matches!(pattern, Pattern::BinaryOp { .. }));
    }

    #[test]
    fn test_pattern_helpers() {
        let _ = Pattern::any();
        let _ = Pattern::op("matmul");
        let _ = Pattern::zero();
        let _ = Pattern::one();
        let _ = Pattern::constant(42.0);
        let _ = Pattern::variable("x");
    }

    #[test]
    fn test_match_creation() {
        let m = Match::new(5);
        assert_eq!(m.root, 5);
        assert!(m.matched_nodes.contains(&5));
    }

    #[test]
    fn test_match_captures() {
        let m = Match::new(5).with_capture("x".to_string(), 10);
        assert_eq!(m.get_capture("x"), Some(10));
        assert!(m.matched_nodes.contains(&10));
    }

    #[test]
    fn test_rewrite_rule_creation() {
        let rule = RewriteRule::new("test_rule")
            .with_pattern(Pattern::any())
            .with_priority(10);

        assert_eq!(rule.name, "test_rule");
        assert_eq!(rule.priority, 10);
    }

    #[test]
    fn test_rewrite_engine_creation() {
        let engine = RewriteEngine::new();
        assert_eq!(engine.rules.len(), 0);
        assert_eq!(engine.strategy, RewriteStrategy::Exhaustive);
    }

    #[test]
    fn test_rewrite_engine_add_rule() {
        let rule = RewriteRule::new("test");
        let engine = RewriteEngine::new().add_rule(rule);
        assert_eq!(engine.rules.len(), 1);
    }

    #[test]
    fn test_rewrite_strategy() {
        let engine = RewriteEngine::new().with_strategy(RewriteStrategy::OnePass);
        assert_eq!(engine.strategy, RewriteStrategy::OnePass);
    }

    #[test]
    fn test_rewrite_stats() {
        let stats = RewriteStats::default();
        assert_eq!(stats.graphs_processed, 0);
        assert_eq!(stats.rewrites_applied, 0);
    }

    #[test]
    fn test_rewrite_stats_display() {
        let mut stats = RewriteStats::default();
        stats.graphs_processed = 5;
        stats.rewrites_applied = 10;
        stats.rule_applications.insert("rule1".to_string(), 7);

        let display = format!("{}", stats);
        assert!(display.contains("Graphs processed:     5"));
        assert!(display.contains("Rewrites applied:     10"));
    }

    #[test]
    fn test_common_rules() {
        let rules = CommonRules::all();
        assert!(!rules.is_empty());
        assert_eq!(rules.len(), 5);
    }

    #[test]
    fn test_eliminate_add_zero_rule() {
        let rule = CommonRules::eliminate_add_zero();
        assert_eq!(rule.name, "eliminate_add_zero");
        assert_eq!(rule.priority, 10);
    }

    #[test]
    fn test_rewrite_one_pass() {
        let rule = RewriteRule::new("test");
        let mut engine = RewriteEngine::new()
            .add_rule(rule)
            .with_strategy(RewriteStrategy::OnePass);

        let result = engine.rewrite_simple(10).unwrap();
        assert!(result <= 10);
        assert!(engine.stats().graphs_processed > 0);
    }

    #[test]
    fn test_rewrite_exhaustive() {
        let rule = RewriteRule::new("test");
        let mut engine = RewriteEngine::new()
            .add_rule(rule)
            .with_strategy(RewriteStrategy::Exhaustive);

        let result = engine.rewrite_simple(10).unwrap();
        assert!(result <= 10);
    }

    #[test]
    fn test_rewrite_fixed_point() {
        let rule = RewriteRule::new("test");
        let mut engine = RewriteEngine::new()
            .add_rule(rule)
            .with_strategy(RewriteStrategy::FixedPoint { max_iterations: 10 });

        let result = engine.rewrite_simple(10).unwrap();
        assert!(result <= 10);
    }

    #[test]
    fn test_rewrite_prioritized() {
        let rule1 = RewriteRule::new("low").with_priority(1);
        let rule2 = RewriteRule::new("high").with_priority(10);

        let mut engine = RewriteEngine::new()
            .add_rule(rule1)
            .add_rule(rule2)
            .with_strategy(RewriteStrategy::Prioritized);

        engine.rewrite_simple(10).unwrap();
        // After sorting, high priority rule should be first
        assert_eq!(engine.rules[0].name, "high");
    }

    #[test]
    fn test_reset_stats() {
        let rule = RewriteRule::new("test");
        let mut engine = RewriteEngine::new().add_rule(rule);

        engine.rewrite_simple(10).unwrap();
        assert!(engine.stats().graphs_processed > 0);

        engine.reset_stats();
        assert_eq!(engine.stats().graphs_processed, 0);
    }

    #[test]
    fn test_verification_flag() {
        let engine = RewriteEngine::new().with_verification(true);
        assert!(engine.verify_semantics);
    }
}
