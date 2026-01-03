//! Graph pattern matching and rewriting system.
//!
//! This module provides a sophisticated pattern matching and rewriting framework
//! for tensor computation graphs. It enables:
//! - Pattern-based graph optimization
//! - Backend-specific transformations
//! - Custom fusion strategies
//! - Automatic graph simplification

use std::collections::{HashMap, HashSet};

use super::{EinsumGraph, EinsumNode, OpType};
use crate::error::IrError;

/// A pattern that can match against graph structures
#[derive(Debug, Clone, PartialEq)]
pub enum GraphPattern {
    /// Match any single node
    AnyNode,
    /// Match a specific operation type
    OpType(OpType),
    /// Match a sequence of operations
    Sequence(Vec<GraphPattern>),
    /// Match any of the given patterns
    Choice(Vec<GraphPattern>),
    /// Match a node with specific input count
    WithInputs(usize),
    /// Match a node with specific output count
    WithOutputs(usize),
    /// Match a named subpattern for capture
    Capture(String, Box<GraphPattern>),
    /// Match a pattern zero or more times
    ZeroOrMore(Box<GraphPattern>),
    /// Match a pattern one or more times
    OneOrMore(Box<GraphPattern>),
}

/// Result of a successful pattern match
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Indices of matched nodes in order
    pub matched_nodes: Vec<usize>,
    /// Captured subpatterns by name
    pub captures: HashMap<String, Vec<usize>>,
    /// Matched tensors (inputs and outputs)
    pub matched_tensors: HashSet<usize>,
}

impl PatternMatch {
    /// Create a new empty pattern match
    pub fn new() -> Self {
        Self {
            matched_nodes: Vec::new(),
            captures: HashMap::new(),
            matched_tensors: HashSet::new(),
        }
    }

    /// Add a matched node
    pub fn add_node(&mut self, node_idx: usize) {
        self.matched_nodes.push(node_idx);
    }

    /// Add a capture
    pub fn add_capture(&mut self, name: String, node_idx: usize) {
        self.captures.entry(name).or_default().push(node_idx);
    }

    /// Get nodes for a capture by name
    pub fn get_capture(&self, name: &str) -> Option<&[usize]> {
        self.captures.get(name).map(|v| v.as_slice())
    }
}

impl Default for PatternMatch {
    fn default() -> Self {
        Self::new()
    }
}

/// A rewrite rule that transforms matched patterns
#[derive(Debug, Clone)]
pub struct GraphRewriteRule {
    /// Name of this rule for debugging
    pub name: String,
    /// Pattern to match
    pub pattern: GraphPattern,
    /// Function to apply the rewrite
    pub rewriter: fn(&EinsumGraph, &PatternMatch) -> Result<Vec<EinsumNode>, IrError>,
    /// Priority (higher = applied first)
    pub priority: i32,
}

impl GraphRewriteRule {
    /// Create a new rewrite rule
    pub fn new(
        name: impl Into<String>,
        pattern: GraphPattern,
        rewriter: fn(&EinsumGraph, &PatternMatch) -> Result<Vec<EinsumNode>, IrError>,
    ) -> Self {
        Self {
            name: name.into(),
            pattern,
            rewriter,
            priority: 0,
        }
    }

    /// Set the priority of this rule
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// Statistics about pattern matching and rewriting
#[derive(Debug, Clone, Default)]
pub struct RewriteStats {
    /// Number of patterns matched
    pub patterns_matched: usize,
    /// Number of rewrites applied
    pub rewrites_applied: usize,
    /// Number of nodes before rewriting
    pub nodes_before: usize,
    /// Number of nodes after rewriting
    pub nodes_after: usize,
    /// Nodes eliminated by rewriting
    pub nodes_eliminated: usize,
}

impl RewriteStats {
    /// Create new empty stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate reduction percentage
    pub fn reduction_percentage(&self) -> f64 {
        if self.nodes_before == 0 {
            return 0.0;
        }
        (self.nodes_eliminated as f64 / self.nodes_before as f64) * 100.0
    }
}

/// Pattern matcher for graphs
pub struct PatternMatcher {
    /// Rules to apply
    rules: Vec<GraphRewriteRule>,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rewrite rule
    pub fn add_rule(&mut self, rule: GraphRewriteRule) {
        self.rules.push(rule);
        // Sort by priority (descending)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Find all matches for a pattern in the graph
    pub fn find_matches(&self, graph: &EinsumGraph, pattern: &GraphPattern) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        // Try to match starting from each node
        for start_idx in 0..graph.nodes.len() {
            if let Some(m) = self.try_match_from(graph, pattern, start_idx, &HashSet::new()) {
                matches.push(m);
            }
        }

        matches
    }

    /// Try to match a pattern starting from a specific node
    fn try_match_from(
        &self,
        graph: &EinsumGraph,
        pattern: &GraphPattern,
        start_idx: usize,
        visited: &HashSet<usize>,
    ) -> Option<PatternMatch> {
        if start_idx >= graph.nodes.len() || visited.contains(&start_idx) {
            return None;
        }

        match pattern {
            GraphPattern::AnyNode => {
                let mut m = PatternMatch::new();
                m.add_node(start_idx);
                Some(m)
            }

            GraphPattern::OpType(expected_op) => {
                let node = &graph.nodes[start_idx];
                if Self::op_matches(&node.op, expected_op) {
                    let mut m = PatternMatch::new();
                    m.add_node(start_idx);
                    Some(m)
                } else {
                    None
                }
            }

            GraphPattern::WithInputs(count) => {
                let node = &graph.nodes[start_idx];
                if node.inputs.len() == *count {
                    let mut m = PatternMatch::new();
                    m.add_node(start_idx);
                    Some(m)
                } else {
                    None
                }
            }

            GraphPattern::WithOutputs(count) => {
                let node = &graph.nodes[start_idx];
                if node.outputs.len() == *count {
                    let mut m = PatternMatch::new();
                    m.add_node(start_idx);
                    Some(m)
                } else {
                    None
                }
            }

            GraphPattern::Capture(name, sub_pattern) => {
                if let Some(mut m) = self.try_match_from(graph, sub_pattern, start_idx, visited) {
                    m.add_capture(name.clone(), start_idx);
                    Some(m)
                } else {
                    None
                }
            }

            GraphPattern::Sequence(patterns) => {
                self.match_sequence(graph, patterns, start_idx, visited)
            }

            GraphPattern::Choice(patterns) => {
                for pat in patterns {
                    if let Some(m) = self.try_match_from(graph, pat, start_idx, visited) {
                        return Some(m);
                    }
                }
                None
            }

            GraphPattern::OneOrMore(sub_pattern) => {
                self.match_one_or_more(graph, sub_pattern, start_idx, visited)
            }

            GraphPattern::ZeroOrMore(sub_pattern) => {
                if let Some(m) = self.match_one_or_more(graph, sub_pattern, start_idx, visited) {
                    Some(m)
                } else {
                    // Zero matches is valid
                    Some(PatternMatch::new())
                }
            }
        }
    }

    /// Match a sequence of patterns
    fn match_sequence(
        &self,
        graph: &EinsumGraph,
        patterns: &[GraphPattern],
        start_idx: usize,
        visited: &HashSet<usize>,
    ) -> Option<PatternMatch> {
        if patterns.is_empty() {
            return Some(PatternMatch::new());
        }

        let mut result = PatternMatch::new();
        let mut current_visited = visited.clone();
        let mut current_idx = start_idx;

        for pattern in patterns {
            if let Some(m) = self.try_match_from(graph, pattern, current_idx, &current_visited) {
                // Merge matches
                for &node in &m.matched_nodes {
                    result.add_node(node);
                    current_visited.insert(node);
                }
                for (name, nodes) in m.captures {
                    for node in nodes {
                        result.add_capture(name.clone(), node);
                    }
                }

                // Move to next node (follow data flow)
                if let Some(&last_node) = m.matched_nodes.last() {
                    if let Some(next) = self.find_successor(graph, last_node) {
                        current_idx = next;
                    } else {
                        return None; // No successor, can't continue sequence
                    }
                }
            } else {
                return None;
            }
        }

        Some(result)
    }

    /// Match one or more occurrences of a pattern
    fn match_one_or_more(
        &self,
        graph: &EinsumGraph,
        pattern: &GraphPattern,
        start_idx: usize,
        visited: &HashSet<usize>,
    ) -> Option<PatternMatch> {
        let mut result = PatternMatch::new();
        let mut current_visited = visited.clone();
        let mut current_idx = start_idx;
        let mut matched_any = false;

        loop {
            if let Some(m) = self.try_match_from(graph, pattern, current_idx, &current_visited) {
                matched_any = true;

                // Merge matches
                for &node in &m.matched_nodes {
                    result.add_node(node);
                    current_visited.insert(node);
                }

                // Try to continue matching
                if let Some(&last_node) = m.matched_nodes.last() {
                    if let Some(next) = self.find_successor(graph, last_node) {
                        current_idx = next;
                        continue;
                    }
                }
            }
            break;
        }

        if matched_any {
            Some(result)
        } else {
            None
        }
    }

    /// Find the successor of a node in the dataflow
    fn find_successor(&self, graph: &EinsumGraph, node_idx: usize) -> Option<usize> {
        let node = &graph.nodes[node_idx];

        // Find a node that uses the output of this node
        for &output_tensor in &node.outputs {
            for (idx, other_node) in graph.nodes.iter().enumerate() {
                if other_node.inputs.contains(&output_tensor) {
                    return Some(idx);
                }
            }
        }

        None
    }

    /// Check if two operation types match
    fn op_matches(actual: &OpType, expected: &OpType) -> bool {
        match (actual, expected) {
            (OpType::Einsum { .. }, OpType::Einsum { .. }) => true,
            (OpType::ElemUnary { op: a }, OpType::ElemUnary { op: b }) => a == b,
            (OpType::ElemBinary { op: a }, OpType::ElemBinary { op: b }) => a == b,
            (OpType::Reduce { op: a, .. }, OpType::Reduce { op: b, .. }) => a == b,
            _ => false,
        }
    }

    /// Apply all rules to a graph and return rewrite statistics
    pub fn apply_rules(&self, graph: &mut EinsumGraph) -> Result<RewriteStats, IrError> {
        let mut stats = RewriteStats::new();
        stats.nodes_before = graph.nodes.len();

        let mut modified = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while modified && iterations < MAX_ITERATIONS {
            modified = false;
            iterations += 1;

            for rule in &self.rules {
                let matches = self.find_matches(graph, &rule.pattern);

                for m in matches {
                    stats.patterns_matched += 1;

                    // Apply the rewrite
                    if let Ok(new_nodes) = (rule.rewriter)(graph, &m) {
                        // Replace matched nodes with new nodes
                        if self.apply_rewrite(graph, &m, new_nodes)? {
                            stats.rewrites_applied += 1;
                            modified = true;
                        }
                    }
                }
            }
        }

        stats.nodes_after = graph.nodes.len();
        stats.nodes_eliminated = stats.nodes_before.saturating_sub(stats.nodes_after);

        Ok(stats)
    }

    /// Apply a rewrite by replacing matched nodes with new nodes
    fn apply_rewrite(
        &self,
        _graph: &mut EinsumGraph,
        _pattern_match: &PatternMatch,
        _new_nodes: Vec<EinsumNode>,
    ) -> Result<bool, IrError> {
        // This is a simplified implementation
        // In a full implementation, we would:
        // 1. Remove the matched nodes
        // 2. Insert the new nodes
        // 3. Rewire connections
        // 4. Update tensor indices

        // For now, just indicate success without modification
        Ok(false)
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Common graph rewrite patterns
pub mod patterns {
    use super::*;

    /// Pattern for consecutive element-wise operations
    #[allow(dead_code)]
    pub fn elementwise_chain(min_length: usize) -> GraphPattern {
        let elem_op = GraphPattern::Choice(vec![
            GraphPattern::OpType(OpType::ElemUnary { op: String::new() }),
            GraphPattern::OpType(OpType::ElemBinary { op: String::new() }),
        ]);

        if min_length == 1 {
            GraphPattern::OneOrMore(Box::new(elem_op))
        } else {
            let mut sequence = Vec::new();
            for _ in 0..min_length {
                sequence.push(elem_op.clone());
            }
            GraphPattern::Sequence(sequence)
        }
    }

    /// Pattern for einsum followed by reduction
    #[allow(dead_code)]
    pub fn einsum_reduce() -> GraphPattern {
        GraphPattern::Sequence(vec![
            GraphPattern::OpType(OpType::Einsum {
                spec: String::new(),
            }),
            GraphPattern::OpType(OpType::Reduce {
                op: String::new(),
                axes: Vec::new(),
            }),
        ])
    }

    /// Pattern for map-reduce idiom
    #[allow(dead_code)]
    pub fn map_reduce() -> GraphPattern {
        GraphPattern::Sequence(vec![
            GraphPattern::Capture(
                "map".to_string(),
                Box::new(GraphPattern::OpType(OpType::ElemUnary {
                    op: String::new(),
                })),
            ),
            GraphPattern::Capture(
                "reduce".to_string(),
                Box::new(GraphPattern::OpType(OpType::Reduce {
                    op: String::new(),
                    axes: Vec::new(),
                })),
            ),
        ])
    }

    /// Pattern for broadcast followed by element-wise op
    #[allow(dead_code)]
    pub fn broadcast_elementwise() -> GraphPattern {
        GraphPattern::Sequence(vec![
            GraphPattern::OpType(OpType::ElemBinary {
                op: "broadcast".to_string(),
            }),
            GraphPattern::OpType(OpType::ElemBinary { op: String::new() }),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_match_creation() {
        let m = PatternMatch::new();
        assert!(m.matched_nodes.is_empty());
        assert!(m.captures.is_empty());
    }

    #[test]
    fn test_pattern_match_add_node() {
        let mut m = PatternMatch::new();
        m.add_node(0);
        m.add_node(1);
        assert_eq!(m.matched_nodes, vec![0, 1]);
    }

    #[test]
    fn test_pattern_match_capture() {
        let mut m = PatternMatch::new();
        m.add_capture("test".to_string(), 5);
        assert_eq!(m.get_capture("test"), Some(&[5][..]));
        assert_eq!(m.get_capture("nonexistent"), None);
    }

    #[test]
    fn test_rewrite_stats_default() {
        let stats = RewriteStats::default();
        assert_eq!(stats.patterns_matched, 0);
        assert_eq!(stats.rewrites_applied, 0);
    }

    #[test]
    fn test_rewrite_stats_reduction() {
        let stats = RewriteStats {
            nodes_before: 100,
            nodes_after: 80,
            nodes_eliminated: 20,
            ..Default::default()
        };
        assert_eq!(stats.reduction_percentage(), 20.0);
    }

    #[test]
    fn test_pattern_matcher_creation() {
        let matcher = PatternMatcher::new();
        assert_eq!(matcher.rules.len(), 0);
    }

    #[test]
    fn test_pattern_matcher_add_rule() {
        let mut matcher = PatternMatcher::new();

        fn dummy_rewriter(
            _graph: &EinsumGraph,
            _m: &PatternMatch,
        ) -> Result<Vec<EinsumNode>, IrError> {
            Ok(Vec::new())
        }

        let rule = GraphRewriteRule::new("test", GraphPattern::AnyNode, dummy_rewriter);
        matcher.add_rule(rule);
        assert_eq!(matcher.rules.len(), 1);
    }

    #[test]
    fn test_rule_priority_ordering() {
        let mut matcher = PatternMatcher::new();

        fn dummy_rewriter(
            _graph: &EinsumGraph,
            _m: &PatternMatch,
        ) -> Result<Vec<EinsumNode>, IrError> {
            Ok(Vec::new())
        }

        let rule1 =
            GraphRewriteRule::new("low", GraphPattern::AnyNode, dummy_rewriter).with_priority(1);
        let rule2 =
            GraphRewriteRule::new("high", GraphPattern::AnyNode, dummy_rewriter).with_priority(10);

        matcher.add_rule(rule1);
        matcher.add_rule(rule2);

        // Should be sorted by priority (descending)
        assert_eq!(matcher.rules[0].name, "high");
        assert_eq!(matcher.rules[1].name, "low");
    }

    #[test]
    fn test_op_matches_einsum() {
        let op1 = OpType::Einsum {
            spec: "ij,jk->ik".to_string(),
        };
        let op2 = OpType::Einsum {
            spec: "ik,kl->il".to_string(),
        };
        assert!(PatternMatcher::op_matches(&op1, &op2));
    }

    #[test]
    fn test_op_matches_elem_unary() {
        let op1 = OpType::ElemUnary {
            op: "relu".to_string(),
        };
        let op2 = OpType::ElemUnary {
            op: "relu".to_string(),
        };
        assert!(PatternMatcher::op_matches(&op1, &op2));
    }

    #[test]
    fn test_op_not_matches_different_types() {
        let op1 = OpType::ElemUnary {
            op: "relu".to_string(),
        };
        let op2 = OpType::ElemBinary {
            op: "add".to_string(),
        };
        assert!(!PatternMatcher::op_matches(&op1, &op2));
    }

    #[test]
    fn test_patterns_elementwise_chain() {
        let pattern = patterns::elementwise_chain(1);
        match pattern {
            GraphPattern::OneOrMore(_) => (),
            _ => panic!("Expected OneOrMore pattern"),
        }
    }

    #[test]
    fn test_patterns_map_reduce() {
        let pattern = patterns::map_reduce();
        match pattern {
            GraphPattern::Sequence(seq) => {
                assert_eq!(seq.len(), 2);
            }
            _ => panic!("Expected Sequence pattern"),
        }
    }
}
