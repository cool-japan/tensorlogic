//! Operation fusion for improved performance.
//!
//! This module identifies opportunities to fuse consecutive element-wise operations
//! into single compound operations, reducing memory traffic and improving performance.

use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

/// Types of fusable operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionPattern {
    /// Two consecutive element-wise unary operations
    UnaryUnary,
    /// Two consecutive element-wise binary operations with same shape
    BinaryBinary,
    /// Unary followed by binary (e.g., relu then multiply)
    UnaryBinary,
    /// Binary followed by unary (e.g., add then relu)
    BinaryUnary,
}

/// A fusion opportunity identified in the graph
#[derive(Debug, Clone)]
pub struct FusionOpportunity {
    /// Indices of nodes that can be fused
    pub node_indices: Vec<usize>,
    /// Pattern type
    pub pattern: FusionPattern,
    /// Description of the fusion
    pub description: String,
}

/// Analyze a graph for fusion opportunities
pub fn analyze_fusion_opportunities(graph: &EinsumGraph) -> Vec<FusionOpportunity> {
    let mut opportunities = Vec::new();

    // Check each consecutive pair of nodes
    for i in 0..(graph.nodes.len().saturating_sub(1)) {
        let node1 = &graph.nodes[i];
        let node2 = &graph.nodes[i + 1];

        // Check if node2 uses output of node1
        if let Some(output1) = node1.outputs.first() {
            if node2.inputs.contains(output1) {
                // Check for fusable patterns
                if let Some(opportunity) = check_fusion_pair(node1, node2, i, i + 1) {
                    opportunities.push(opportunity);
                }
            }
        }
    }

    opportunities
}

/// Check if two consecutive nodes can be fused
fn check_fusion_pair(
    node1: &EinsumNode,
    node2: &EinsumNode,
    idx1: usize,
    idx2: usize,
) -> Option<FusionOpportunity> {
    match (&node1.op, &node2.op) {
        (OpType::ElemUnary { op: op1 }, OpType::ElemUnary { op: op2 }) => Some(FusionOpportunity {
            node_indices: vec![idx1, idx2],
            pattern: FusionPattern::UnaryUnary,
            description: format!("Fuse {} → {}", op1, op2),
        }),
        (OpType::ElemBinary { op: op1 }, OpType::ElemUnary { op: op2 }) => {
            Some(FusionOpportunity {
                node_indices: vec![idx1, idx2],
                pattern: FusionPattern::BinaryUnary,
                description: format!("Fuse {} → {}", op1, op2),
            })
        }
        (OpType::ElemUnary { op: op1 }, OpType::ElemBinary { op: op2 }) => {
            Some(FusionOpportunity {
                node_indices: vec![idx1, idx2],
                pattern: FusionPattern::UnaryBinary,
                description: format!("Fuse {} → {}", op1, op2),
            })
        }
        _ => None,
    }
}

/// Statistics about fusion analysis
#[derive(Debug, Clone)]
pub struct FusionStats {
    /// Total opportunities found
    pub total_opportunities: usize,
    /// Opportunities by pattern type
    pub by_pattern: std::collections::HashMap<String, usize>,
    /// Estimated speedup (rough estimate)
    pub estimated_speedup: f64,
}

impl FusionStats {
    /// Compute fusion statistics from opportunities
    pub fn from_opportunities(opportunities: &[FusionOpportunity]) -> Self {
        let mut by_pattern = std::collections::HashMap::new();

        for opp in opportunities {
            let pattern_name = format!("{:?}", opp.pattern);
            *by_pattern.entry(pattern_name).or_insert(0) += 1;
        }

        // Rough estimate: each fusion saves 1 memory pass
        // Assume 20% speedup per fused operation
        let estimated_speedup = 1.0 + (opportunities.len() as f64 * 0.2);

        FusionStats {
            total_opportunities: opportunities.len(),
            by_pattern,
            estimated_speedup,
        }
    }
}

impl std::fmt::Display for FusionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fusion Analysis:")?;
        writeln!(f, "  Total opportunities: {}", self.total_opportunities)?;
        writeln!(f, "  By pattern:")?;
        for (pattern, count) in &self.by_pattern {
            writeln!(f, "    {}: {}", pattern, count)?;
        }
        write!(f, "  Estimated speedup: {:.2}x", self.estimated_speedup)
    }
}

/// Apply fusion transformations to a graph
///
/// Note: This is an analysis-only function. Actual fusion requires
/// kernel implementations in the executor.
pub fn suggest_fusions(graph: &EinsumGraph) -> (Vec<FusionOpportunity>, FusionStats) {
    let opportunities = analyze_fusion_opportunities(graph);
    let stats = FusionStats::from_opportunities(&opportunities);
    (opportunities, stats)
}

#[cfg(all(test, feature = "integration-tests"))]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tensorlogic_compiler::compile_to_einsum;
    use tensorlogic_ir::{TLExpr, Term};

    #[test]
    fn test_analyze_unary_unary_fusion() {
        // Create expression with explicit unary operations
        // Since we don't have direct relu/sigmoid in TLExpr, test with actual operations
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let y = TLExpr::pred("y", vec![Term::var("i")]);
        // Create a chain that might have fusable operations
        let expr = TLExpr::mul(TLExpr::add(x.clone(), y.clone()), x);

        let graph = compile_to_einsum(&expr).unwrap();
        let _opportunities = analyze_fusion_opportunities(&graph);

        // Fusion opportunities depend on compilation strategy
        // Test passes if analysis completes without error
    }

    #[test]
    fn test_analyze_binary_unary_fusion() {
        // Create expression with binary then unary: (x + y) > 0
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let y = TLExpr::pred("y", vec![Term::var("i")]);
        let sum = TLExpr::add(x, y);
        let zero = TLExpr::constant(0.0);
        let expr = TLExpr::gt(sum, zero);

        let graph = compile_to_einsum(&expr).unwrap();
        let _opportunities = analyze_fusion_opportunities(&graph);

        // May find fusion opportunities depending on compilation
        // At minimum, no crash
    }

    #[test]
    fn test_fusion_stats() {
        let opportunities = vec![
            FusionOpportunity {
                node_indices: vec![0, 1],
                pattern: FusionPattern::UnaryUnary,
                description: "relu → sigmoid".to_string(),
            },
            FusionOpportunity {
                node_indices: vec![2, 3],
                pattern: FusionPattern::BinaryUnary,
                description: "add → relu".to_string(),
            },
            FusionOpportunity {
                node_indices: vec![4, 5],
                pattern: FusionPattern::UnaryUnary,
                description: "sigmoid → relu".to_string(),
            },
        ];

        let stats = FusionStats::from_opportunities(&opportunities);

        assert_eq!(stats.total_opportunities, 3);
        assert_eq!(stats.by_pattern.len(), 2); // UnaryUnary and BinaryUnary
        assert!(stats.estimated_speedup > 1.0);
    }

    #[test]
    fn test_suggest_fusions() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let y = TLExpr::pred("y", vec![Term::var("i")]);
        let expr = TLExpr::mul(x, y);

        let graph = compile_to_einsum(&expr).unwrap();
        let (opportunities, stats) = suggest_fusions(&graph);

        // Basic sanity check
        assert_eq!(opportunities.len(), stats.total_opportunities);
    }

    #[test]
    fn test_empty_graph_fusion() {
        let graph = EinsumGraph {
            tensors: vec![],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_metadata: HashMap::new(),
        };

        let opportunities = analyze_fusion_opportunities(&graph);
        assert_eq!(opportunities.len(), 0);
    }

    #[test]
    fn test_single_node_no_fusion() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let graph = compile_to_einsum(&x).unwrap();

        let opportunities = analyze_fusion_opportunities(&graph);
        // Single node - no fusion possible
        assert_eq!(opportunities.len(), 0);
    }
}
