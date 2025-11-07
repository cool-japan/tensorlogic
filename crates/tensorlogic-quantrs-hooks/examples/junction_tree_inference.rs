//! Junction tree exact inference example.
//!
//! This example demonstrates the junction tree algorithm for exact probabilistic inference.
//! We'll use a classic "Student Network" Bayesian network to show how junction trees
//! provide exact marginal probabilities efficiently.
//!
//! # Network Structure
//!
//! ```text
//!     Difficulty
//!         |
//!         v
//!      Grade  <--- Intelligence
//!         |
//!         v
//!      Letter
//! ```
//!
//! This example shows:
//! 1. Building a factor graph from a Bayesian network
//! 2. Constructing a junction tree
//! 3. Calibrating the tree for exact inference
//! 4. Querying marginal probabilities
//! 5. Analyzing treewidth and running intersection property

use scirs2_core::ndarray::Array;
use tensorlogic_quantrs_hooks::{Factor, FactorGraph, JunctionTree};

fn main() -> anyhow::Result<()> {
    println!("=== Junction Tree Exact Inference Example ===\n");

    // Build the Student Network
    let graph = build_student_network()?;

    println!("Factor Graph Statistics:");
    println!("  Variables: {}", graph.num_variables());
    println!("  Factors: {}", graph.num_factors());
    println!();

    // Construct junction tree
    println!("Constructing junction tree...");
    let mut tree = JunctionTree::from_factor_graph(&graph)?;

    println!("Junction Tree Structure:");
    println!("  Number of cliques: {}", tree.cliques.len());
    println!("  Number of edges: {}", tree.edges.len());
    println!("  Treewidth: {}", tree.treewidth());
    println!();

    // Print cliques
    println!("Cliques:");
    for (i, clique) in tree.cliques.iter().enumerate() {
        let vars: Vec<&String> = clique.variables.iter().collect();
        println!("  Clique {}: {:?}", i, vars);
    }
    println!();

    // Verify running intersection property
    println!(
        "Running Intersection Property: {}",
        if tree.verify_running_intersection_property() {
            "✓ Satisfied"
        } else {
            "✗ Not satisfied"
        }
    );
    println!();

    // Calibrate the tree
    println!("Calibrating junction tree...");
    tree.calibrate()?;
    println!("✓ Calibration complete\n");

    // Query marginal probabilities
    println!("=== Marginal Queries ===\n");

    // Query: P(Intelligence)
    println!("Query: P(Intelligence)");
    let p_intelligence = tree.query_marginal("Intelligence")?;
    println!("  P(Intelligence = Low)  = {:.4}", p_intelligence[[0]]);
    println!("  P(Intelligence = High) = {:.4}", p_intelligence[[1]]);
    println!();

    // Query: P(Difficulty)
    println!("Query: P(Difficulty)");
    let p_difficulty = tree.query_marginal("Difficulty")?;
    println!("  P(Difficulty = Easy) = {:.4}", p_difficulty[[0]]);
    println!("  P(Difficulty = Hard) = {:.4}", p_difficulty[[1]]);
    println!();

    // Query: P(Grade)
    println!("Query: P(Grade)");
    let p_grade = tree.query_marginal("Grade")?;
    println!("  P(Grade = A) = {:.4}", p_grade[[0]]);
    println!("  P(Grade = B) = {:.4}", p_grade[[1]]);
    println!("  P(Grade = C) = {:.4}", p_grade[[2]]);
    println!();

    // Query: P(Letter)
    println!("Query: P(Letter)");
    let p_letter = tree.query_marginal("Letter")?;
    println!("  P(Letter = Weak)   = {:.4}", p_letter[[0]]);
    println!("  P(Letter = Strong) = {:.4}", p_letter[[1]]);
    println!();

    // Joint query
    println!("=== Joint Query ===\n");
    println!("Query: P(Intelligence, Difficulty)");
    let p_joint =
        tree.query_joint_marginal(&["Intelligence".to_string(), "Difficulty".to_string()])?;
    println!("Shape: {:?}", p_joint.shape());
    println!(
        "  P(Intelligence=Low,  Difficulty=Easy) = {:.4}",
        p_joint[[0, 0]]
    );
    println!(
        "  P(Intelligence=Low,  Difficulty=Hard) = {:.4}",
        p_joint[[0, 1]]
    );
    println!(
        "  P(Intelligence=High, Difficulty=Easy) = {:.4}",
        p_joint[[1, 0]]
    );
    println!(
        "  P(Intelligence=High, Difficulty=Hard) = {:.4}",
        p_joint[[1, 1]]
    );
    println!();

    println!("=== Performance Analysis ===\n");

    // Compare junction tree properties
    println!("Complexity Analysis:");
    println!("  Treewidth: {}", tree.treewidth());
    println!(
        "  Max clique size: {}",
        tree.cliques
            .iter()
            .map(|c| c.variables.len())
            .max()
            .unwrap_or(0)
    );
    println!(
        "  Avg separator size: {:.2}",
        if tree.edges.is_empty() {
            0.0
        } else {
            tree.edges
                .iter()
                .map(|e| e.separator.variables.len())
                .sum::<usize>() as f64
                / tree.edges.len() as f64
        }
    );
    println!();

    println!("Advantages of Junction Tree Algorithm:");
    println!("  ✓ Exact inference (no approximation)");
    println!("  ✓ Efficient message passing on tree structure");
    println!("  ✓ Handles any query without recomputation");
    println!("  ✓ Guarantees consistency across marginals");
    println!();

    println!("✓ Example completed successfully!");

    Ok(())
}

/// Build the Student Network factor graph.
///
/// # Network Description
///
/// - **Intelligence**: Prior probability of student intelligence (Low/High)
/// - **Difficulty**: Prior probability of course difficulty (Easy/Hard)
/// - **Grade**: Student's grade depends on Intelligence and Difficulty (A/B/C)
/// - **Letter**: Recommendation letter quality depends on Grade (Weak/Strong)
fn build_student_network() -> anyhow::Result<FactorGraph> {
    let mut graph = FactorGraph::new();

    // Add variables
    graph.add_variable_with_card("Intelligence".to_string(), "Binary".to_string(), 2);
    graph.add_variable_with_card("Difficulty".to_string(), "Binary".to_string(), 2);
    graph.add_variable_with_card("Grade".to_string(), "Ternary".to_string(), 3);
    graph.add_variable_with_card("Letter".to_string(), "Binary".to_string(), 2);

    // P(Intelligence): Prior probability
    // [Low, High] = [0.7, 0.3]
    let p_intelligence = Factor::new(
        "P(Intelligence)".to_string(),
        vec!["Intelligence".to_string()],
        Array::from_shape_vec(vec![2], vec![0.7, 0.3])?.into_dyn(),
    )?;
    graph.add_factor(p_intelligence)?;

    // P(Difficulty): Prior probability
    // [Easy, Hard] = [0.6, 0.4]
    let p_difficulty = Factor::new(
        "P(Difficulty)".to_string(),
        vec!["Difficulty".to_string()],
        Array::from_shape_vec(vec![2], vec![0.6, 0.4])?.into_dyn(),
    )?;
    graph.add_factor(p_difficulty)?;

    // P(Grade | Intelligence, Difficulty): Conditional probability table
    // Rows: Intelligence (Low=0, High=1)
    // Cols: Difficulty (Easy=0, Hard=1)
    // Values: [A, B, C] for each combination
    //
    // Intelligence=Low, Difficulty=Easy:  [0.3, 0.4, 0.3]
    // Intelligence=Low, Difficulty=Hard:  [0.05, 0.25, 0.7]
    // Intelligence=High, Difficulty=Easy: [0.9, 0.08, 0.02]
    // Intelligence=High, Difficulty=Hard: [0.5, 0.3, 0.2]
    #[rustfmt::skip]
    let grade_values = vec![
        // Intelligence=Low
        0.3, 0.4, 0.3,    // Difficulty=Easy  -> [A, B, C]
        0.05, 0.25, 0.7,  // Difficulty=Hard  -> [A, B, C]
        // Intelligence=High
        0.9, 0.08, 0.02,  // Difficulty=Easy  -> [A, B, C]
        0.5, 0.3, 0.2,    // Difficulty=Hard  -> [A, B, C]
    ];

    let p_grade = Factor::new(
        "P(Grade|Intelligence,Difficulty)".to_string(),
        vec![
            "Intelligence".to_string(),
            "Difficulty".to_string(),
            "Grade".to_string(),
        ],
        Array::from_shape_vec(vec![2, 2, 3], grade_values)?.into_dyn(),
    )?;
    graph.add_factor(p_grade)?;

    // P(Letter | Grade): Conditional probability table
    // Grade values: A=0, B=1, C=2
    // Letter values: Weak=0, Strong=1
    //
    // Grade=A: [0.1, 0.9]  (mostly strong letters)
    // Grade=B: [0.4, 0.6]
    // Grade=C: [0.99, 0.01] (mostly weak letters)
    let p_letter = Factor::new(
        "P(Letter|Grade)".to_string(),
        vec!["Grade".to_string(), "Letter".to_string()],
        Array::from_shape_vec(
            vec![3, 2],
            vec![
                0.1, 0.9, // Grade=A -> [Weak, Strong]
                0.4, 0.6, // Grade=B -> [Weak, Strong]
                0.99, 0.01, // Grade=C -> [Weak, Strong]
            ],
        )?
        .into_dyn(),
    )?;
    graph.add_factor(p_letter)?;

    Ok(graph)
}
