//! Tree Kernels Demonstration
//!
//! This example shows how to use the three types of tree kernels:
//! 1. SubtreeKernel - Exact subtree matching
//! 2. SubsetTreeKernel - Fragment matching with decay
//! 3. PartialTreeKernel - Partial matching with thresholds
//!
//! Run with: cargo run --example tree_kernels_demo

use tensorlogic_ir::TLExpr;
use tensorlogic_sklears_kernels::{
    PartialTreeKernel, PartialTreeKernelConfig, SubsetTreeKernel, SubsetTreeKernelConfig,
    SubtreeKernel, SubtreeKernelConfig, TreeNode,
};

fn main() -> anyhow::Result<()> {
    println!("=== Tree Kernels Demonstration ===\n");

    // Create sample TLExpr trees
    println!("1. Creating sample expression trees...");

    // Tree 1: AND(Pred(a), Pred(b))
    let expr1 = TLExpr::and(TLExpr::pred("a", vec![]), TLExpr::pred("b", vec![]));

    // Tree 2: AND(Pred(a), Pred(c))
    let expr2 = TLExpr::and(TLExpr::pred("a", vec![]), TLExpr::pred("c", vec![]));

    // Tree 3: OR(AND(Pred(a), Pred(b)), Pred(d))
    let expr3 = TLExpr::or(
        TLExpr::and(TLExpr::pred("a", vec![]), TLExpr::pred("b", vec![])),
        TLExpr::pred("d", vec![]),
    );

    // Convert to TreeNodes
    let tree1 = TreeNode::from_tlexpr(&expr1);
    let tree2 = TreeNode::from_tlexpr(&expr2);
    let tree3 = TreeNode::from_tlexpr(&expr3);

    println!("  Tree 1: AND(Pred(a), Pred(b))");
    println!("  Tree 2: AND(Pred(a), Pred(c))");
    println!("  Tree 3: OR(AND(Pred(a), Pred(b)), Pred(d))");
    println!();

    // Demonstrate SubtreeKernel
    println!("2. SubtreeKernel (Exact Subtree Matching)");
    println!("   Counts exact matching subtrees between two trees.");
    println!();

    let subtree_config = SubtreeKernelConfig::new().with_normalize(true);
    let subtree_kernel = SubtreeKernel::new(subtree_config);

    let sim_12 = subtree_kernel.compute_trees(&tree1, &tree2)?;
    let sim_13 = subtree_kernel.compute_trees(&tree1, &tree3)?;
    let sim_23 = subtree_kernel.compute_trees(&tree2, &tree3)?;

    println!("   Similarity(tree1, tree2): {:.4}", sim_12);
    println!("   Similarity(tree1, tree3): {:.4}", sim_13);
    println!("   Similarity(tree2, tree3): {:.4}", sim_23);
    println!("   → Trees 1 & 3 share exact subtrees (both have AND with Pred(a) and Pred(b))");
    println!();

    // Demonstrate SubsetTreeKernel
    println!("3. SubsetTreeKernel (Fragment Matching with Decay)");
    println!("   Counts matching fragments with exponential decay for depth.");
    println!();

    let subset_config = SubsetTreeKernelConfig::new()?
        .with_decay(0.8)?
        .with_normalize(true);
    let subset_kernel = SubsetTreeKernel::new(subset_config);

    let sim_12_subset = subset_kernel.compute_trees(&tree1, &tree2)?;
    let sim_13_subset = subset_kernel.compute_trees(&tree1, &tree3)?;
    let sim_23_subset = subset_kernel.compute_trees(&tree2, &tree3)?;

    println!("   Similarity(tree1, tree2): {:.4}", sim_12_subset);
    println!("   Similarity(tree1, tree3): {:.4}", sim_13_subset);
    println!("   Similarity(tree2, tree3): {:.4}", sim_23_subset);
    println!("   → Decay=0.8 weights deeper matches less heavily");
    println!();

    // Demonstrate PartialTreeKernel
    println!("4. PartialTreeKernel (Partial Matching with Thresholds)");
    println!("   Allows partial matches based on child similarity threshold.");
    println!();

    let partial_config = PartialTreeKernelConfig::new()?
        .with_threshold(0.5)?
        .with_normalize(true);
    let partial_kernel = PartialTreeKernel::new(partial_config);

    let sim_12_partial = partial_kernel.compute_trees(&tree1, &tree2)?;
    let sim_13_partial = partial_kernel.compute_trees(&tree1, &tree3)?;
    let sim_23_partial = partial_kernel.compute_trees(&tree2, &tree3)?;

    println!("   Similarity(tree1, tree2): {:.4}", sim_12_partial);
    println!("   Similarity(tree1, tree3): {:.4}", sim_13_partial);
    println!("   Similarity(tree2, tree3): {:.4}", sim_23_partial);
    println!("   → Threshold=0.5 allows nodes to match if children are 50%+ similar");
    println!();

    // Compare all three approaches
    println!("5. Comparison of All Three Kernels");
    println!("   ┌────────────────────┬─────────┬─────────┬─────────┐");
    println!("   │ Tree Pair          │ Subtree │ Subset  │ Partial │");
    println!("   ├────────────────────┼─────────┼─────────┼─────────┤");
    println!(
        "   │ Tree 1 vs Tree 2   │  {:.4}  │  {:.4}  │  {:.4}  │",
        sim_12, sim_12_subset, sim_12_partial
    );
    println!(
        "   │ Tree 1 vs Tree 3   │  {:.4}  │  {:.4}  │  {:.4}  │",
        sim_13, sim_13_subset, sim_13_partial
    );
    println!(
        "   │ Tree 2 vs Tree 3   │  {:.4}  │  {:.4}  │  {:.4}  │",
        sim_23, sim_23_subset, sim_23_partial
    );
    println!("   └────────────────────┴─────────┴─────────┴─────────┘");
    println!();

    // Practical use case
    println!("6. Practical Use Case: Rule Similarity");
    println!("   Tree kernels are useful for measuring similarity between:");
    println!("   • Logic rules (TLExpr structures)");
    println!("   • Parse trees (code similarity)");
    println!("   • Decision trees (ML model comparison)");
    println!("   • Hierarchical data (XML, JSON structures)");
    println!();

    // Create more complex rules
    let rule1 = TLExpr::imply(
        TLExpr::and(
            TLExpr::pred("temperature_high", vec![]),
            TLExpr::pred("humidity_low", vec![]),
        ),
        TLExpr::pred("fire_risk", vec![]),
    );

    let rule2 = TLExpr::imply(
        TLExpr::and(
            TLExpr::pred("temperature_high", vec![]),
            TLExpr::pred("wind_strong", vec![]),
        ),
        TLExpr::pred("fire_risk", vec![]),
    );

    let tree_rule1 = TreeNode::from_tlexpr(&rule1);
    let tree_rule2 = TreeNode::from_tlexpr(&rule2);

    let rule_similarity = subset_kernel.compute_trees(&tree_rule1, &tree_rule2)?;

    println!("   Rule 1: temperature_high ∧ humidity_low → fire_risk");
    println!("   Rule 2: temperature_high ∧ wind_strong → fire_risk");
    println!("   Similarity: {:.4}", rule_similarity);
    println!("   → Rules share structure and some conditions, giving high similarity");
    println!();

    println!("=== Demo Complete ===");

    Ok(())
}
