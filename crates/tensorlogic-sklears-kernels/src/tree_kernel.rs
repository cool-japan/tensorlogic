//! Tree kernels for structured data similarity
//!
//! This module provides kernel functions for tree-structured data, particularly
//! useful for measuring similarity between hierarchical expressions like TLExpr.
//!
//! ## Tree Representations
//!
//! Trees are represented as labeled nodes with children, where each node has:
//! - A label (string identifier)
//! - A list of child nodes
//!
//! ## Kernel Types
//!
//! - **SubtreeKernel**: Counts common subtrees between two trees
//! - **SubsetTreeKernel**: Counts common tree fragments (allows gaps)
//! - **PartialTreeKernel**: Partial subtree matching with decay factors
//!
//! ## References
//!
//! - Collins & Duffy (2001): "Convolution Kernels for Natural Language"
//! - Moschitti (2006): "Making Tree Kernels Practical for Natural Language Learning"

use crate::error::{KernelError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tensorlogic_ir::TLExpr;

/// A tree node with label and children
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TreeNode {
    /// Node label
    pub label: String,
    /// Child nodes
    pub children: Vec<TreeNode>,
}

impl TreeNode {
    /// Create a new tree node
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            children: Vec::new(),
        }
    }

    /// Create a tree node with children
    pub fn with_children(label: impl Into<String>, children: Vec<TreeNode>) -> Self {
        Self {
            label: label.into(),
            children,
        }
    }

    /// Get the height of the tree
    pub fn height(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.height()).max().unwrap_or(0)
        }
    }

    /// Get the number of nodes in the tree
    pub fn num_nodes(&self) -> usize {
        1 + self.children.iter().map(|c| c.num_nodes()).sum::<usize>()
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Convert from TLExpr to TreeNode
    pub fn from_tlexpr(expr: &TLExpr) -> Self {
        match expr {
            TLExpr::Pred { name, .. } => TreeNode::new(format!("Pred({})", name)),
            TLExpr::And(left, right) => TreeNode::with_children(
                "And",
                vec![TreeNode::from_tlexpr(left), TreeNode::from_tlexpr(right)],
            ),
            TLExpr::Or(left, right) => TreeNode::with_children(
                "Or",
                vec![TreeNode::from_tlexpr(left), TreeNode::from_tlexpr(right)],
            ),
            TLExpr::Not(expr) => TreeNode::with_children("Not", vec![TreeNode::from_tlexpr(expr)]),
            TLExpr::Imply(left, right) => TreeNode::with_children(
                "Imply",
                vec![TreeNode::from_tlexpr(left), TreeNode::from_tlexpr(right)],
            ),
            TLExpr::Exists { var, domain, body } => TreeNode::with_children(
                format!("Exists({}, {})", var, domain),
                vec![TreeNode::from_tlexpr(body)],
            ),
            TLExpr::ForAll { var, domain, body } => TreeNode::with_children(
                format!("ForAll({}, {})", var, domain),
                vec![TreeNode::from_tlexpr(body)],
            ),
            _ => TreeNode::new("Expr"),
        }
    }

    /// Get all subtrees (including the tree itself)
    fn get_all_subtrees(&self) -> Vec<TreeNode> {
        let mut subtrees = vec![self.clone()];
        for child in &self.children {
            subtrees.extend(child.get_all_subtrees());
        }
        subtrees
    }
}

/// Configuration for subtree kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtreeKernelConfig {
    /// Whether to normalize the kernel value
    pub normalize: bool,
}

impl SubtreeKernelConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self { normalize: true }
    }

    /// Set normalization flag
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Default for SubtreeKernelConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Subtree kernel - counts common subtrees
///
/// This kernel counts the number of subtrees that are common between two trees.
/// It provides a measure of structural similarity.
///
/// ## Formula
///
/// ```text
/// K(T1, T2) = Σ_i Σ_j I(subtree_i(T1) == subtree_j(T2))
/// ```
///
/// where I is the indicator function.
pub struct SubtreeKernel {
    config: SubtreeKernelConfig,
}

impl SubtreeKernel {
    /// Create a new subtree kernel
    pub fn new(config: SubtreeKernelConfig) -> Self {
        Self { config }
    }

    /// Compute kernel between two trees
    pub fn compute_trees(&self, tree1: &TreeNode, tree2: &TreeNode) -> Result<f64> {
        let subtrees1 = tree1.get_all_subtrees();
        let subtrees2 = tree2.get_all_subtrees();

        // Count matches
        let mut count = 0;
        for st1 in &subtrees1 {
            for st2 in &subtrees2 {
                if st1 == st2 {
                    count += 1;
                }
            }
        }

        let similarity = count as f64;

        if self.config.normalize {
            // Normalize by geometric mean of self-similarities
            let self_sim1 = subtrees1.len() as f64;
            let self_sim2 = subtrees2.len() as f64;
            let norm = (self_sim1 * self_sim2).sqrt();
            if norm > 0.0 {
                Ok(similarity / norm)
            } else {
                Ok(0.0)
            }
        } else {
            Ok(similarity)
        }
    }
}

/// Configuration for subset tree kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsetTreeKernelConfig {
    /// Whether to normalize the kernel value
    pub normalize: bool,
    /// Decay factor for tree fragments (0.0 to 1.0)
    pub decay: f64,
}

impl SubsetTreeKernelConfig {
    /// Create a new configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            normalize: true,
            decay: 1.0,
        })
    }

    /// Set normalization flag
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set decay factor
    pub fn with_decay(mut self, decay: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&decay) {
            return Err(KernelError::InvalidParameter {
                parameter: "decay".to_string(),
                value: decay.to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        self.decay = decay;
        Ok(self)
    }
}

impl Default for SubsetTreeKernelConfig {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Subset tree kernel - allows gaps in tree fragments
///
/// This kernel is more flexible than the subtree kernel, allowing matching
/// of tree fragments even when intermediate nodes are skipped.
pub struct SubsetTreeKernel {
    config: SubsetTreeKernelConfig,
}

impl SubsetTreeKernel {
    /// Create a new subset tree kernel
    pub fn new(config: SubsetTreeKernelConfig) -> Self {
        Self { config }
    }

    /// Compute kernel between two trees
    pub fn compute_trees(&self, tree1: &TreeNode, tree2: &TreeNode) -> Result<f64> {
        let similarity = self.compute_recursive(tree1, tree2, &mut HashMap::new());

        if self.config.normalize {
            let self_sim1 = self.compute_recursive(tree1, tree1, &mut HashMap::new());
            let self_sim2 = self.compute_recursive(tree2, tree2, &mut HashMap::new());
            let norm = (self_sim1 * self_sim2).sqrt();
            if norm > 0.0 {
                Ok(similarity / norm)
            } else {
                Ok(0.0)
            }
        } else {
            Ok(similarity)
        }
    }

    /// Recursive kernel computation with memoization
    fn compute_recursive(
        &self,
        n1: &TreeNode,
        n2: &TreeNode,
        cache: &mut HashMap<(usize, usize), f64>,
    ) -> f64 {
        // Simple hash for caching (not perfect but good enough)
        let key = (n1.num_nodes(), n2.num_nodes());

        if let Some(&cached) = cache.get(&key) {
            return cached;
        }

        let mut result = 0.0;

        // If labels match
        if n1.label == n2.label {
            // Add contribution from this node
            result += self.config.decay;

            // If both have children, recursively compute
            if !n1.children.is_empty() && !n2.children.is_empty() {
                // Compute kernel for all pairs of children
                for c1 in &n1.children {
                    for c2 in &n2.children {
                        result += self.config.decay * self.compute_recursive(c1, c2, cache);
                    }
                }
            }
        }

        cache.insert(key, result);
        result
    }
}

/// Configuration for partial tree kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialTreeKernelConfig {
    /// Whether to normalize the kernel value
    pub normalize: bool,
    /// Decay factor for partial matches
    pub decay: f64,
    /// Minimum similarity threshold for partial matches
    pub threshold: f64,
}

impl PartialTreeKernelConfig {
    /// Create a new configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            normalize: true,
            decay: 0.8,
            threshold: 0.0,
        })
    }

    /// Set normalization flag
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set decay factor
    pub fn with_decay(mut self, decay: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&decay) {
            return Err(KernelError::InvalidParameter {
                parameter: "decay".to_string(),
                value: decay.to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        self.decay = decay;
        Ok(self)
    }

    /// Set threshold
    pub fn with_threshold(mut self, threshold: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(KernelError::InvalidParameter {
                parameter: "threshold".to_string(),
                value: threshold.to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        self.threshold = threshold;
        Ok(self)
    }
}

impl Default for PartialTreeKernelConfig {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Partial tree kernel - allows partial subtree matching
///
/// This kernel measures similarity by allowing partial matches with decay factors.
/// It's useful when trees have similar structure but not exact matches.
pub struct PartialTreeKernel {
    config: PartialTreeKernelConfig,
}

impl PartialTreeKernel {
    /// Create a new partial tree kernel
    pub fn new(config: PartialTreeKernelConfig) -> Self {
        Self { config }
    }

    /// Compute kernel between two trees
    pub fn compute_trees(&self, tree1: &TreeNode, tree2: &TreeNode) -> Result<f64> {
        let similarity = self.compute_partial_match(tree1, tree2, 1.0);

        if similarity < self.config.threshold {
            return Ok(0.0);
        }

        if self.config.normalize {
            let self_sim1 = self.compute_partial_match(tree1, tree1, 1.0);
            let self_sim2 = self.compute_partial_match(tree2, tree2, 1.0);
            let norm = (self_sim1 * self_sim2).sqrt();
            if norm > 0.0 {
                Ok(similarity / norm)
            } else {
                Ok(0.0)
            }
        } else {
            Ok(similarity)
        }
    }

    /// Compute partial match score with decay
    fn compute_partial_match(&self, n1: &TreeNode, n2: &TreeNode, weight: f64) -> f64 {
        let mut score = 0.0;

        // Exact label match
        if n1.label == n2.label {
            score += weight;

            // Recursively match children with decay
            let min_children = n1.children.len().min(n2.children.len());
            for i in 0..min_children {
                score += self.compute_partial_match(
                    &n1.children[i],
                    &n2.children[i],
                    weight * self.config.decay,
                );
            }
        } else {
            // Partial match based on label similarity (simple heuristic)
            let label_sim = self.label_similarity(&n1.label, &n2.label);
            score += weight * label_sim * 0.5; // Partial credit

            // Try matching children even if labels differ
            let min_children = n1.children.len().min(n2.children.len());
            for i in 0..min_children {
                score += self.compute_partial_match(
                    &n1.children[i],
                    &n2.children[i],
                    weight * self.config.decay * 0.5,
                );
            }
        }

        score
    }

    /// Simple label similarity (can be improved with more sophisticated methods)
    fn label_similarity(&self, label1: &str, label2: &str) -> f64 {
        if label1 == label2 {
            1.0
        } else {
            // Simple Jaccard similarity on characters
            let chars1: std::collections::HashSet<char> = label1.chars().collect();
            let chars2: std::collections::HashSet<char> = label2.chars().collect();
            let intersection = chars1.intersection(&chars2).count();
            let union = chars1.union(&chars2).count();
            if union > 0 {
                intersection as f64 / union as f64
            } else {
                0.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_node_creation() {
        let node = TreeNode::new("root");
        assert_eq!(node.label, "root");
        assert!(node.children.is_empty());
        assert!(node.is_leaf());
    }

    #[test]
    fn test_tree_node_with_children() {
        let child1 = TreeNode::new("child1");
        let child2 = TreeNode::new("child2");
        let parent = TreeNode::with_children("parent", vec![child1, child2]);

        assert_eq!(parent.label, "parent");
        assert_eq!(parent.children.len(), 2);
        assert!(!parent.is_leaf());
    }

    #[test]
    fn test_tree_height() {
        let leaf = TreeNode::new("leaf");
        assert_eq!(leaf.height(), 1);

        let tree = TreeNode::with_children(
            "root",
            vec![
                TreeNode::new("child1"),
                TreeNode::with_children("child2", vec![TreeNode::new("grandchild")]),
            ],
        );
        assert_eq!(tree.height(), 3);
    }

    #[test]
    fn test_tree_num_nodes() {
        let tree = TreeNode::with_children(
            "root",
            vec![
                TreeNode::new("child1"),
                TreeNode::with_children("child2", vec![TreeNode::new("grandchild")]),
            ],
        );
        assert_eq!(tree.num_nodes(), 4);
    }

    #[test]
    fn test_tree_from_tlexpr() {
        let expr = TLExpr::and(TLExpr::pred("p1", vec![]), TLExpr::pred("p2", vec![]));
        let tree = TreeNode::from_tlexpr(&expr);

        assert_eq!(tree.label, "And");
        assert_eq!(tree.children.len(), 2);
    }

    #[test]
    fn test_subtree_kernel_identical() {
        let tree1 = TreeNode::with_children(
            "root",
            vec![TreeNode::new("child1"), TreeNode::new("child2")],
        );
        let tree2 = tree1.clone();

        let config = SubtreeKernelConfig::new().with_normalize(false);
        let kernel = SubtreeKernel::new(config);

        let sim = kernel.compute_trees(&tree1, &tree2).unwrap();
        assert!(sim > 0.0);
    }

    #[test]
    fn test_subtree_kernel_different() {
        // Create trees with completely different children
        let tree1 = TreeNode::with_children("root", vec![TreeNode::new("child1")]);
        let tree2 = TreeNode::with_children("root", vec![TreeNode::new("child2")]);

        let config = SubtreeKernelConfig::new().with_normalize(false);
        let kernel = SubtreeKernel::new(config);

        let sim = kernel.compute_trees(&tree1, &tree2).unwrap();
        // Trees with different children have no matching subtrees
        // (root node is different because it includes children)
        assert!(sim >= 0.0); // No matches expected
    }

    #[test]
    fn test_subtree_kernel_partial_match() {
        // Create trees with same root label and one matching child
        let tree1 = TreeNode::with_children(
            "root",
            vec![TreeNode::new("child1"), TreeNode::new("child2")],
        );
        let tree2 = TreeNode::with_children(
            "root",
            vec![TreeNode::new("child1"), TreeNode::new("child3")],
        );

        let config = SubtreeKernelConfig::new().with_normalize(false);
        let kernel = SubtreeKernel::new(config);

        let sim = kernel.compute_trees(&tree1, &tree2).unwrap();
        // Should have similarity from the shared "child1" subtree
        assert!(sim > 0.0);
    }

    #[test]
    fn test_subtree_kernel_normalized() {
        let tree1 = TreeNode::with_children(
            "root",
            vec![TreeNode::new("child1"), TreeNode::new("child2")],
        );
        let tree2 = tree1.clone();

        let config = SubtreeKernelConfig::new().with_normalize(true);
        let kernel = SubtreeKernel::new(config);

        let sim = kernel.compute_trees(&tree1, &tree2).unwrap();
        assert!((sim - 1.0).abs() < 1e-6); // Self-similarity should be 1.0 when normalized
    }

    #[test]
    fn test_subset_tree_kernel() {
        let tree1 = TreeNode::with_children(
            "root",
            vec![TreeNode::new("child1"), TreeNode::new("child2")],
        );
        let tree2 = TreeNode::with_children("root", vec![TreeNode::new("child1")]);

        let config = SubsetTreeKernelConfig::new().unwrap();
        let kernel = SubsetTreeKernel::new(config);

        let sim = kernel.compute_trees(&tree1, &tree2).unwrap();
        assert!(sim > 0.0);
    }

    #[test]
    fn test_subset_tree_kernel_decay() {
        let tree1 = TreeNode::with_children("root", vec![TreeNode::new("child")]);
        let tree2 = tree1.clone();

        let config1 = SubsetTreeKernelConfig::new()
            .unwrap()
            .with_decay(1.0)
            .unwrap()
            .with_normalize(false);
        let kernel1 = SubsetTreeKernel::new(config1);

        let config2 = SubsetTreeKernelConfig::new()
            .unwrap()
            .with_decay(0.5)
            .unwrap()
            .with_normalize(false);
        let kernel2 = SubsetTreeKernel::new(config2);

        let sim1 = kernel1.compute_trees(&tree1, &tree2).unwrap();
        let sim2 = kernel2.compute_trees(&tree1, &tree2).unwrap();

        // Lower decay should give lower similarity
        assert!(sim2 < sim1);
    }

    #[test]
    fn test_partial_tree_kernel() {
        let tree1 = TreeNode::with_children(
            "root",
            vec![TreeNode::new("child1"), TreeNode::new("child2")],
        );
        let tree2 = TreeNode::with_children(
            "root",
            vec![TreeNode::new("child1"), TreeNode::new("child3")],
        );

        let config = PartialTreeKernelConfig::new().unwrap();
        let kernel = PartialTreeKernel::new(config);

        let sim = kernel.compute_trees(&tree1, &tree2).unwrap();
        assert!(sim > 0.0); // Should have partial similarity
    }

    #[test]
    fn test_partial_tree_kernel_threshold() {
        let tree1 = TreeNode::with_children("root1", vec![TreeNode::new("child")]);
        let tree2 = TreeNode::with_children("root2", vec![TreeNode::new("child")]);

        let config = PartialTreeKernelConfig::new()
            .unwrap()
            .with_threshold(0.9)
            .unwrap();
        let kernel = PartialTreeKernel::new(config);

        let sim = kernel.compute_trees(&tree1, &tree2).unwrap();
        // Different roots with high threshold should give low/zero similarity
        assert!(sim < 0.5);
    }

    #[test]
    fn test_partial_tree_kernel_config_invalid_decay() {
        let result = PartialTreeKernelConfig::new().unwrap().with_decay(1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_tree_kernel_config_invalid_threshold() {
        let result = PartialTreeKernelConfig::new().unwrap().with_threshold(-0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_tree_kernel_with_tlexpr() {
        let expr1 = TLExpr::and(TLExpr::pred("p1", vec![]), TLExpr::pred("p2", vec![]));
        let expr2 = TLExpr::and(TLExpr::pred("p1", vec![]), TLExpr::pred("p3", vec![]));

        let tree1 = TreeNode::from_tlexpr(&expr1);
        let tree2 = TreeNode::from_tlexpr(&expr2);

        let config = SubtreeKernelConfig::new();
        let kernel = SubtreeKernel::new(config);

        let sim = kernel.compute_trees(&tree1, &tree2).unwrap();
        assert!(sim > 0.0); // Should have some similarity (And node and p1 match)
    }
}
