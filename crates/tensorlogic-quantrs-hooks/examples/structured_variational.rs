//! Structured variational inference example.
//!
//! This example demonstrates advanced variational inference methods that leverage
//! the factor graph structure, comparing them to standard mean-field approximation.
//!
//! # Scenario
//!
//! We create a 3x3 grid-structured Markov Random Field (MRF) with:
//! - Binary variables at each grid position
//! - Pairwise interactions between neighboring nodes
//! - External field (bias) at each node
//!
//! We compare three variational methods:
//! 1. **Mean-Field**: Assumes complete independence (fully factorized)
//! 2. **Bethe Approximation**: Uses factor graph structure for better accuracy
//! 3. **Tree-Reweighted BP**: Provides upper bounds on the partition function

use tensorlogic_quantrs_hooks::{
    BetheApproximation, FactorGraph, MeanFieldInference, TreeReweightedBP,
};

fn main() -> anyhow::Result<()> {
    println!("=== Structured Variational Inference Example ===\n");

    // Create a grid MRF
    println!("Creating 3x3 grid Markov Random Field...");
    let graph = create_grid_mrf(3, 3);
    println!("  - 9 binary variables (grid positions)");
    println!(
        "  - {} factors (node potentials + edge potentials)\n",
        graph.factor_ids().count()
    );

    // Method 1: Mean-Field Variational Inference
    println!("=== Method 1: Mean-Field Approximation ===");
    println!("(Assumes complete independence between variables)\n");

    let mean_field = MeanFieldInference::new(100, 1e-6);
    let mf_start = std::time::Instant::now();
    let mf_beliefs = mean_field.run(&graph)?;
    let mf_time = mf_start.elapsed();

    println!("Converged in {:?}", mf_time);
    println!("\nMean-Field Marginals (first 3 variables):");
    for var_name in graph.variable_names().take(3) {
        if let Some(belief) = mf_beliefs.get(var_name) {
            println!("  P({}=1) = {:.4}", var_name, belief[[1]]);
        }
    }

    // Compute ELBO
    let mf_elbo = mean_field.compute_elbo(&graph, &mf_beliefs)?;
    println!("\nMean-Field ELBO: {:.4}\n", mf_elbo);

    // Method 2: Bethe Approximation
    println!("=== Method 2: Bethe Approximation ===");
    println!("(Uses factor graph structure for structured approximation)\n");

    let bethe = BetheApproximation::new(100, 1e-6, 0.0);
    let bethe_start = std::time::Instant::now();
    let bethe_beliefs = bethe.run(&graph)?;
    let bethe_time = bethe_start.elapsed();

    println!("Converged in {:?}", bethe_time);
    println!("\nBethe Marginals (first 3 variables):");
    for var_name in graph.variable_names().take(3) {
        if let Some(belief) = bethe_beliefs.get(var_name) {
            println!("  P({}=1) = {:.4}", var_name, belief[[1]]);
        }
    }

    // Compute Bethe free energy
    let factor_beliefs = bethe.compute_factor_beliefs(&graph, &bethe_beliefs)?;
    let bethe_free_energy = bethe.compute_free_energy(&graph, &bethe_beliefs, &factor_beliefs)?;
    println!("\nBethe Free Energy: {:.4}\n", bethe_free_energy);

    // Method 3: Tree-Reweighted Belief Propagation
    println!("=== Method 3: Tree-Reweighted BP ===");
    println!("(Provides upper bounds on log partition function)\n");

    let mut trw_bp = TreeReweightedBP::new(100, 1e-6);
    let trw_start = std::time::Instant::now();
    let trw_beliefs = trw_bp.run(&graph)?;
    let trw_time = trw_start.elapsed();

    println!("Converged in {:?}", trw_time);
    println!("\nTRW-BP Marginals (first 3 variables):");
    for var_name in graph.variable_names().take(3) {
        if let Some(belief) = trw_beliefs.get(var_name) {
            println!("  P({}=1) = {:.4}", var_name, belief[[1]]);
        }
    }
    println!();

    // Comparison Summary
    println!("=== Comparison Summary ===\n");

    println!("Convergence Times:");
    println!("  Mean-Field: {:?}", mf_time);
    println!("  Bethe:      {:?}", bethe_time);
    println!("  TRW-BP:     {:?}", trw_time);
    println!();

    println!("Variational Objectives:");
    println!("  Mean-Field ELBO:       {:.4}", mf_elbo);
    println!("  Bethe Free Energy:     {:.4}", bethe_free_energy);
    println!();

    // Compare marginal differences
    println!("Marginal Differences (L1 norm):");
    let mf_bethe_diff = compute_marginal_difference(&mf_beliefs, &bethe_beliefs);
    let mf_trw_diff = compute_marginal_difference(&mf_beliefs, &trw_beliefs);
    let bethe_trw_diff = compute_marginal_difference(&bethe_beliefs, &trw_beliefs);

    println!("  Mean-Field vs Bethe:  {:.4}", mf_bethe_diff);
    println!("  Mean-Field vs TRW-BP: {:.4}", mf_trw_diff);
    println!("  Bethe vs TRW-BP:      {:.4}", bethe_trw_diff);
    println!();

    // Analysis
    println!("=== Analysis ===\n");
    println!("1. **Mean-Field** assumes complete independence, which is");
    println!("   inappropriate for the grid structure. This leads to");
    println!("   potential inaccuracies in marginal estimates.");
    println!();
    println!("2. **Bethe Approximation** respects the factor graph structure,");
    println!("   providing more accurate marginals at similar computational cost.");
    println!("   The Bethe free energy is typically tighter than mean-field ELBO.");
    println!();
    println!("3. **TRW-BP** uses edge reweighting to provide upper bounds on");
    println!("   the log partition function, making it particularly robust for");
    println!("   loopy graphs. Convergence is guaranteed for convex combinations");
    println!("   of spanning trees.");
    println!();

    println!("For grid-structured MRFs with loops, Bethe and TRW-BP typically");
    println!("outperform mean-field in terms of accuracy, while maintaining");
    println!("similar computational efficiency.\n");

    println!("✓ Structured variational inference demonstration completed!");

    Ok(())
}

/// Create a grid-structured Markov Random Field.
///
/// Grid has binary variables with pairwise potentials favoring agreement
/// between neighbors (ferromagnetic Ising model).
fn create_grid_mrf(rows: usize, cols: usize) -> FactorGraph {
    use scirs2_core::ndarray::Array;

    let mut graph = FactorGraph::new();

    // Add variables
    for i in 0..rows {
        for j in 0..cols {
            let var_name = format!("x_{}_{}", i, j);
            graph.add_variable_with_card(var_name, "Binary".to_string(), 2);
        }
    }

    // Add node potentials (external field)
    let node_potential_values = vec![0.4, 0.6]; // Slight bias toward 1
    for i in 0..rows {
        for j in 0..cols {
            let var_name = format!("x_{}_{}", i, j);
            let factor_name = format!("node_{}_{}", i, j);

            let potential = Array::from_shape_vec(vec![2], node_potential_values.clone())
                .unwrap()
                .into_dyn();

            use tensorlogic_quantrs_hooks::Factor;
            let factor = Factor {
                name: factor_name,
                variables: vec![var_name],
                values: potential,
            };
            graph.add_factor(factor).unwrap();
        }
    }

    // Add horizontal edge potentials
    for i in 0..rows {
        for j in 0..(cols - 1) {
            let var1 = format!("x_{}_{}", i, j);
            let var2 = format!("x_{}_{}", i, j + 1);
            let factor_name = format!("h_edge_{}_{}_{}", i, j, j + 1);

            // Ferromagnetic coupling: favor agreement
            // [0,0], [0,1], [1,0], [1,1]
            let edge_values = vec![
                0.8, // both 0
                0.2, // differ
                0.2, // differ
                0.8, // both 1
            ];

            let potential = Array::from_shape_vec(vec![2, 2], edge_values)
                .unwrap()
                .into_dyn();

            use tensorlogic_quantrs_hooks::Factor;
            let factor = Factor {
                name: factor_name,
                variables: vec![var1, var2],
                values: potential,
            };
            graph.add_factor(factor).unwrap();
        }
    }

    // Add vertical edge potentials
    for i in 0..(rows - 1) {
        for j in 0..cols {
            let var1 = format!("x_{}_{}", i, j);
            let var2 = format!("x_{}_{}", i + 1, j);
            let factor_name = format!("v_edge_{}_{}_{}", i, i + 1, j);

            let edge_values = vec![0.8, 0.2, 0.2, 0.8];

            let potential = Array::from_shape_vec(vec![2, 2], edge_values)
                .unwrap()
                .into_dyn();

            use tensorlogic_quantrs_hooks::Factor;
            let factor = Factor {
                name: factor_name,
                variables: vec![var1, var2],
                values: potential,
            };
            graph.add_factor(factor).unwrap();
        }
    }

    graph
}

/// Compute L1 distance between two sets of marginals.
fn compute_marginal_difference(
    beliefs1: &std::collections::HashMap<String, scirs2_core::ndarray::ArrayD<f64>>,
    beliefs2: &std::collections::HashMap<String, scirs2_core::ndarray::ArrayD<f64>>,
) -> f64 {
    let mut total_diff = 0.0;
    let mut count = 0;

    for (var, belief1) in beliefs1 {
        if let Some(belief2) = beliefs2.get(var) {
            let diff: f64 = (belief1 - belief2).mapv(|x| x.abs()).sum();
            total_diff += diff;
            count += 1;
        }
    }

    if count > 0 {
        total_diff / count as f64
    } else {
        0.0
    }
}
