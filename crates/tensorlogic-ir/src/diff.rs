//! IR diff tool for comparing graphs and expressions.
//!
//! This module provides utilities to compare two IR structures and
//! identify differences, useful for debugging and validation.

use crate::{EinsumGraph, EinsumNode, OpType, TLExpr};
use std::collections::HashSet;

/// Difference between two expressions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprDiff {
    /// Expressions are identical
    Identical,
    /// Different expression types
    TypeMismatch { left: String, right: String },
    /// Different predicate names or arities
    PredicateMismatch { left: String, right: String },
    /// Different subexpressions
    SubexprMismatch {
        path: Vec<String>,
        left: String,
        right: String,
    },
    /// Different quantifier variables or domains
    QuantifierMismatch {
        left_var: String,
        right_var: String,
        left_domain: String,
        right_domain: String,
    },
}

/// Difference between two graphs
#[derive(Debug, Clone)]
pub struct GraphDiff {
    /// Tensors only in left graph
    pub left_only_tensors: Vec<String>,
    /// Tensors only in right graph
    pub right_only_tensors: Vec<String>,
    /// Nodes only in left graph
    pub left_only_nodes: usize,
    /// Nodes only in right graph
    pub right_only_nodes: usize,
    /// Different node operations
    pub node_differences: Vec<NodeDiff>,
    /// Output differences
    pub output_differences: Vec<String>,
}

/// Difference in a specific node
#[derive(Debug, Clone)]
pub struct NodeDiff {
    pub node_index: usize,
    pub description: String,
}

impl ExprDiff {
    /// Check if expressions are identical
    pub fn is_identical(&self) -> bool {
        matches!(self, ExprDiff::Identical)
    }

    /// Get human-readable description
    pub fn description(&self) -> String {
        match self {
            ExprDiff::Identical => "Expressions are identical".to_string(),
            ExprDiff::TypeMismatch { left, right } => {
                format!("Type mismatch: left={}, right={}", left, right)
            }
            ExprDiff::PredicateMismatch { left, right } => {
                format!("Predicate mismatch: left={}, right={}", left, right)
            }
            ExprDiff::SubexprMismatch { path, left, right } => {
                format!(
                    "Subexpression mismatch at {}: left={}, right={}",
                    path.join("/"),
                    left,
                    right
                )
            }
            ExprDiff::QuantifierMismatch {
                left_var,
                right_var,
                left_domain,
                right_domain,
            } => {
                format!(
                    "Quantifier mismatch: left=({}, {}), right=({}, {})",
                    left_var, left_domain, right_var, right_domain
                )
            }
        }
    }
}

impl GraphDiff {
    /// Check if graphs are identical
    pub fn is_identical(&self) -> bool {
        self.left_only_tensors.is_empty()
            && self.right_only_tensors.is_empty()
            && self.left_only_nodes == 0
            && self.right_only_nodes == 0
            && self.node_differences.is_empty()
            && self.output_differences.is_empty()
    }

    /// Get summary of differences
    pub fn summary(&self) -> String {
        if self.is_identical() {
            return "Graphs are identical".to_string();
        }

        let mut parts = Vec::new();

        if !self.left_only_tensors.is_empty() {
            parts.push(format!(
                "{} tensors only in left",
                self.left_only_tensors.len()
            ));
        }
        if !self.right_only_tensors.is_empty() {
            parts.push(format!(
                "{} tensors only in right",
                self.right_only_tensors.len()
            ));
        }
        if self.left_only_nodes > 0 {
            parts.push(format!("{} nodes only in left", self.left_only_nodes));
        }
        if self.right_only_nodes > 0 {
            parts.push(format!("{} nodes only in right", self.right_only_nodes));
        }
        if !self.node_differences.is_empty() {
            parts.push(format!("{} node differences", self.node_differences.len()));
        }
        if !self.output_differences.is_empty() {
            parts.push(format!(
                "{} output differences",
                self.output_differences.len()
            ));
        }

        parts.join(", ")
    }
}

/// Compare two expressions
pub fn diff_exprs(left: &TLExpr, right: &TLExpr) -> ExprDiff {
    diff_exprs_impl(left, right, &mut Vec::new())
}

fn diff_exprs_impl(left: &TLExpr, right: &TLExpr, path: &mut Vec<String>) -> ExprDiff {
    match (left, right) {
        (TLExpr::Pred { name: n1, args: a1 }, TLExpr::Pred { name: n2, args: a2 }) => {
            if n1 != n2 || a1.len() != a2.len() {
                ExprDiff::PredicateMismatch {
                    left: format!("{}({})", n1, a1.len()),
                    right: format!("{}({})", n2, a2.len()),
                }
            } else {
                ExprDiff::Identical
            }
        }
        (TLExpr::And(l1, r1), TLExpr::And(l2, r2))
        | (TLExpr::Or(l1, r1), TLExpr::Or(l2, r2))
        | (TLExpr::Imply(l1, r1), TLExpr::Imply(l2, r2))
        | (TLExpr::Add(l1, r1), TLExpr::Add(l2, r2))
        | (TLExpr::Sub(l1, r1), TLExpr::Sub(l2, r2))
        | (TLExpr::Mul(l1, r1), TLExpr::Mul(l2, r2))
        | (TLExpr::Div(l1, r1), TLExpr::Div(l2, r2))
        | (TLExpr::Pow(l1, r1), TLExpr::Pow(l2, r2))
        | (TLExpr::Mod(l1, r1), TLExpr::Mod(l2, r2))
        | (TLExpr::Min(l1, r1), TLExpr::Min(l2, r2))
        | (TLExpr::Max(l1, r1), TLExpr::Max(l2, r2))
        | (TLExpr::Eq(l1, r1), TLExpr::Eq(l2, r2))
        | (TLExpr::Lt(l1, r1), TLExpr::Lt(l2, r2))
        | (TLExpr::Gt(l1, r1), TLExpr::Gt(l2, r2))
        | (TLExpr::Lte(l1, r1), TLExpr::Lte(l2, r2))
        | (TLExpr::Gte(l1, r1), TLExpr::Gte(l2, r2)) => {
            path.push("left".to_string());
            let left_diff = diff_exprs_impl(l1, l2, path);
            path.pop();

            if !left_diff.is_identical() {
                return left_diff;
            }

            path.push("right".to_string());
            let right_diff = diff_exprs_impl(r1, r2, path);
            path.pop();

            right_diff
        }
        (TLExpr::Not(e1), TLExpr::Not(e2))
        | (TLExpr::Score(e1), TLExpr::Score(e2))
        | (TLExpr::Abs(e1), TLExpr::Abs(e2))
        | (TLExpr::Floor(e1), TLExpr::Floor(e2))
        | (TLExpr::Ceil(e1), TLExpr::Ceil(e2))
        | (TLExpr::Round(e1), TLExpr::Round(e2))
        | (TLExpr::Sqrt(e1), TLExpr::Sqrt(e2))
        | (TLExpr::Exp(e1), TLExpr::Exp(e2))
        | (TLExpr::Log(e1), TLExpr::Log(e2))
        | (TLExpr::Sin(e1), TLExpr::Sin(e2))
        | (TLExpr::Cos(e1), TLExpr::Cos(e2))
        | (TLExpr::Tan(e1), TLExpr::Tan(e2)) => {
            path.push("inner".to_string());
            let diff = diff_exprs_impl(e1, e2, path);
            path.pop();
            diff
        }
        (
            TLExpr::Exists {
                var: v1,
                domain: d1,
                body: b1,
            },
            TLExpr::Exists {
                var: v2,
                domain: d2,
                body: b2,
            },
        )
        | (
            TLExpr::ForAll {
                var: v1,
                domain: d1,
                body: b1,
            },
            TLExpr::ForAll {
                var: v2,
                domain: d2,
                body: b2,
            },
        ) => {
            if v1 != v2 || d1 != d2 {
                return ExprDiff::QuantifierMismatch {
                    left_var: v1.clone(),
                    right_var: v2.clone(),
                    left_domain: d1.clone(),
                    right_domain: d2.clone(),
                };
            }

            path.push("body".to_string());
            let diff = diff_exprs_impl(b1, b2, path);
            path.pop();
            diff
        }
        (
            TLExpr::IfThenElse {
                condition: c1,
                then_branch: t1,
                else_branch: e1,
            },
            TLExpr::IfThenElse {
                condition: c2,
                then_branch: t2,
                else_branch: e2,
            },
        ) => {
            path.push("condition".to_string());
            let cond_diff = diff_exprs_impl(c1, c2, path);
            path.pop();

            if !cond_diff.is_identical() {
                return cond_diff;
            }

            path.push("then".to_string());
            let then_diff = diff_exprs_impl(t1, t2, path);
            path.pop();

            if !then_diff.is_identical() {
                return then_diff;
            }

            path.push("else".to_string());
            let else_diff = diff_exprs_impl(e1, e2, path);
            path.pop();

            else_diff
        }
        (TLExpr::Constant(c1), TLExpr::Constant(c2)) => {
            if (c1 - c2).abs() < f64::EPSILON {
                ExprDiff::Identical
            } else {
                ExprDiff::SubexprMismatch {
                    path: path.clone(),
                    left: format!("{}", c1),
                    right: format!("{}", c2),
                }
            }
        }
        _ => ExprDiff::TypeMismatch {
            left: format!("{:?}", left).split('(').next().unwrap().to_string(),
            right: format!("{:?}", right)
                .split('(')
                .next()
                .unwrap()
                .to_string(),
        },
    }
}

/// Compare two graphs
pub fn diff_graphs(left: &EinsumGraph, right: &EinsumGraph) -> GraphDiff {
    let left_tensors: HashSet<_> = left.tensors.iter().collect();
    let right_tensors: HashSet<_> = right.tensors.iter().collect();

    let left_only_tensors: Vec<String> = left_tensors
        .difference(&right_tensors)
        .map(|s| s.to_string())
        .collect();
    let right_only_tensors: Vec<String> = right_tensors
        .difference(&left_tensors)
        .map(|s| s.to_string())
        .collect();

    let node_differences = diff_nodes(&left.nodes, &right.nodes);

    let left_only_nodes = if left.nodes.len() > right.nodes.len() {
        left.nodes.len() - right.nodes.len()
    } else {
        0
    };
    let right_only_nodes = if right.nodes.len() > left.nodes.len() {
        right.nodes.len() - left.nodes.len()
    } else {
        0
    };

    let output_differences = diff_outputs(&left.outputs, &right.outputs);

    GraphDiff {
        left_only_tensors,
        right_only_tensors,
        left_only_nodes,
        right_only_nodes,
        node_differences,
        output_differences,
    }
}

fn diff_nodes(left: &[EinsumNode], right: &[EinsumNode]) -> Vec<NodeDiff> {
    let mut differences = Vec::new();
    let min_len = left.len().min(right.len());

    for i in 0..min_len {
        if let Some(diff) = diff_node(&left[i], &right[i], i) {
            differences.push(diff);
        }
    }

    differences
}

fn diff_node(left: &EinsumNode, right: &EinsumNode, index: usize) -> Option<NodeDiff> {
    if left.inputs != right.inputs {
        return Some(NodeDiff {
            node_index: index,
            description: format!("Different inputs: {:?} vs {:?}", left.inputs, right.inputs),
        });
    }

    if left.outputs != right.outputs {
        return Some(NodeDiff {
            node_index: index,
            description: format!(
                "Different outputs: {:?} vs {:?}",
                left.outputs, right.outputs
            ),
        });
    }

    if !ops_equal(&left.op, &right.op) {
        return Some(NodeDiff {
            node_index: index,
            description: format!("Different operations: {:?} vs {:?}", left.op, right.op),
        });
    }

    None
}

fn ops_equal(left: &OpType, right: &OpType) -> bool {
    // Simple discriminant comparison
    std::mem::discriminant(left) == std::mem::discriminant(right)
}

fn diff_outputs(left: &[usize], right: &[usize]) -> Vec<String> {
    let mut differences = Vec::new();

    if left.len() != right.len() {
        differences.push(format!(
            "Different number of outputs: {} vs {}",
            left.len(),
            right.len()
        ));
    }

    for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        if l != r {
            differences.push(format!("Output {} differs: {} vs {}", i, l, r));
        }
    }

    differences
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Term;

    #[test]
    fn test_identical_exprs() {
        let expr1 = TLExpr::pred("p", vec![Term::var("x")]);
        let expr2 = TLExpr::pred("p", vec![Term::var("x")]);

        let diff = diff_exprs(&expr1, &expr2);
        assert!(diff.is_identical());
    }

    #[test]
    fn test_different_predicates() {
        let expr1 = TLExpr::pred("p", vec![Term::var("x")]);
        let expr2 = TLExpr::pred("q", vec![Term::var("x")]);

        let diff = diff_exprs(&expr1, &expr2);
        assert!(!diff.is_identical());
        assert!(matches!(diff, ExprDiff::PredicateMismatch { .. }));
    }

    #[test]
    fn test_different_types() {
        let expr1 = TLExpr::pred("p", vec![Term::var("x")]);
        let expr2 = TLExpr::constant(1.0);

        let diff = diff_exprs(&expr1, &expr2);
        assert!(!diff.is_identical());
        assert!(matches!(diff, ExprDiff::TypeMismatch { .. }));
    }

    #[test]
    fn test_nested_and_difference() {
        let expr1 = TLExpr::and(
            TLExpr::pred("p", vec![Term::var("x")]),
            TLExpr::pred("q", vec![Term::var("y")]),
        );
        let expr2 = TLExpr::and(
            TLExpr::pred("p", vec![Term::var("x")]),
            TLExpr::pred("r", vec![Term::var("y")]),
        );

        let diff = diff_exprs(&expr1, &expr2);
        assert!(!diff.is_identical());
    }

    #[test]
    fn test_quantifier_difference() {
        let expr1 = TLExpr::exists("x", "Domain1", TLExpr::pred("p", vec![Term::var("x")]));
        let expr2 = TLExpr::exists("y", "Domain2", TLExpr::pred("p", vec![Term::var("y")]));

        let diff = diff_exprs(&expr1, &expr2);
        assert!(!diff.is_identical());
        assert!(matches!(diff, ExprDiff::QuantifierMismatch { .. }));
    }

    #[test]
    fn test_identical_graphs() {
        let graph1 = EinsumGraph {
            tensors: vec!["t0".to_string()],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![0],
            tensor_metadata: std::collections::HashMap::new(),
        };
        let graph2 = EinsumGraph {
            tensors: vec!["t0".to_string()],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![0],
            tensor_metadata: std::collections::HashMap::new(),
        };

        let diff = diff_graphs(&graph1, &graph2);
        assert!(diff.is_identical());
    }

    #[test]
    fn test_different_tensor_count() {
        let graph1 = EinsumGraph {
            tensors: vec!["t0".to_string(), "t1".to_string()],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_metadata: std::collections::HashMap::new(),
        };
        let graph2 = EinsumGraph {
            tensors: vec!["t0".to_string()],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_metadata: std::collections::HashMap::new(),
        };

        let diff = diff_graphs(&graph1, &graph2);
        assert!(!diff.is_identical());
        assert_eq!(diff.left_only_tensors.len(), 1);
    }

    #[test]
    fn test_different_outputs() {
        let graph1 = EinsumGraph {
            tensors: vec!["t0".to_string()],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![0],
            tensor_metadata: std::collections::HashMap::new(),
        };
        let graph2 = EinsumGraph {
            tensors: vec!["t0".to_string()],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![1],
            tensor_metadata: std::collections::HashMap::new(),
        };

        let diff = diff_graphs(&graph1, &graph2);
        assert!(!diff.is_identical());
        assert!(!diff.output_differences.is_empty());
    }

    #[test]
    fn test_diff_summary() {
        let diff = GraphDiff {
            left_only_tensors: vec!["t1".to_string()],
            right_only_tensors: vec!["t2".to_string()],
            left_only_nodes: 0,
            right_only_nodes: 0,
            node_differences: vec![],
            output_differences: vec![],
        };

        let summary = diff.summary();
        assert!(summary.contains("tensors only in left"));
        assert!(summary.contains("tensors only in right"));
    }
}
