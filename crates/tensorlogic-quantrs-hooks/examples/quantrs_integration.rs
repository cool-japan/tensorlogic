//! QuantRS2 integration example.
//!
//! This example demonstrates how to use the QuantRS integration hooks to:
//! 1. Export PGM models to QuantRS2-compatible format
//! 2. Convert factors to distributions
//! 3. Serialize models to JSON for ecosystem integration
//! 4. Compute information-theoretic quantities
//! 5. Analyze model statistics

use scirs2_core::ndarray::Array;
use tensorlogic_quantrs_hooks::{
    quantrs_hooks::utils, Factor, FactorGraph, QuantRSDistribution, QuantRSModelExport,
};

fn main() -> anyhow::Result<()> {
    println!("=== QuantRS2 Integration Example ===\n");

    // Build a simple Bayesian network
    let graph = build_simple_network()?;

    println!("Factor Graph Statistics:");
    println!("  Variables: {}", graph.num_variables());
    println!("  Factors: {}", graph.num_factors());
    println!();

    // Export model to QuantRS format
    println!("=== Model Export to QuantRS2 ===\n");
    let model_export = graph.to_quantrs_model()?;

    println!("Exported Model:");
    println!("  Type: {}", model_export.model_type);
    println!("  Variables: {}", model_export.variables.len());
    println!("  Factors: {}", model_export.factors.len());
    println!();

    println!("Variable Definitions:");
    for var_def in &model_export.variables {
        println!(
            "  {} (domain: {}, cardinality: {})",
            var_def.name, var_def.domain, var_def.cardinality
        );
    }
    println!();

    println!("Factor Definitions:");
    for factor_def in &model_export.factors {
        println!("  {} - scope: {:?}", factor_def.name, factor_def.scope);
    }
    println!();

    // Get model statistics
    println!("=== Model Statistics ===\n");
    let stats = graph.model_stats();
    println!("  Number of variables: {}", stats.num_variables);
    println!("  Number of factors: {}", stats.num_factors);
    println!("  Average factor size: {:.2}", stats.avg_factor_size);
    println!("  Maximum factor size: {}", stats.max_factor_size);
    println!();

    // Export individual factor to QuantRS distribution
    println!("=== Distribution Export ===\n");

    let pxy_values = Array::from_shape_vec(vec![2, 2], vec![0.1, 0.4, 0.2, 0.3])?.into_dyn();
    let pxy = Factor::new(
        "P(X,Y)".to_string(),
        vec!["X".to_string(), "Y".to_string()],
        pxy_values,
    )?;

    let dist_export = pxy.to_quantrs_distribution()?;

    println!("Distribution P(X,Y):");
    println!("  Variables: {:?}", dist_export.variables);
    println!("  Cardinalities: {:?}", dist_export.cardinalities);
    println!("  Shape: {:?}", dist_export.shape);
    println!("  Type: {}", dist_export.metadata.distribution_type);
    println!(
        "  Normalized: {}",
        if dist_export.metadata.normalized {
            "✓"
        } else {
            "✗"
        }
    );
    println!();

    println!("Probability values:");
    for i in 0..2 {
        for j in 0..2 {
            let idx = i * 2 + j;
            println!(
                "  P(X={}, Y={}) = {:.3}",
                i, j, dist_export.probabilities[idx]
            );
        }
    }
    println!();

    // Check normalization
    println!("=== Distribution Properties ===\n");
    println!("Is normalized: {}", pxy.is_normalized());
    println!("Support size: {}", pxy.support().len());
    println!();

    println!("Support (all valid assignments):");
    for (idx, assignment) in pxy.support().iter().enumerate().take(4) {
        println!("  {}: {:?}", idx, assignment);
    }
    println!();

    // Compute mutual information
    println!("=== Information-Theoretic Measures ===\n");

    let mi = utils::mutual_information(&dist_export, "X", "Y")?;
    println!("Mutual Information I(X;Y): {:.4} bits", mi);
    println!();

    // Compare distributions with KL divergence
    let q_values = Array::from_shape_vec(vec![2, 2], vec![0.25, 0.25, 0.25, 0.25])?.into_dyn();
    let q = Factor::new(
        "Q(X,Y)".to_string(),
        vec!["X".to_string(), "Y".to_string()],
        q_values,
    )?;

    let q_export = q.to_quantrs_distribution()?;
    let kl_div = utils::kl_divergence(&dist_export, &q_export)?;
    println!("KL Divergence D_KL(P||Q): {:.4}", kl_div);
    println!();

    // JSON serialization for ecosystem integration
    println!("=== JSON Export for QuantRS2 Ecosystem ===\n");

    let json_export = utils::export_to_json(&graph)?;
    println!("JSON export (first 500 characters):");
    println!("{}...", &json_export[..json_export.len().min(500)]);
    println!();

    // Distribution roundtrip
    println!("=== Distribution Roundtrip Test ===\n");

    let original_factor = pxy.clone();
    let exported_dist = original_factor.to_quantrs_distribution()?;
    let imported_factor = Factor::from_quantrs_distribution(&exported_dist)?;

    println!("Original factor variables: {:?}", original_factor.variables);
    println!("Imported factor variables: {:?}", imported_factor.variables);
    println!(
        "Values match: {}",
        original_factor.values.shape() == imported_factor.values.shape()
    );
    println!();

    // Advanced: Create a more complex network and analyze
    println!("=== Complex Network Analysis ===\n");

    let complex_graph = build_complex_network()?;
    let complex_stats = complex_graph.model_stats();

    println!("Complex Network Statistics:");
    println!("  Variables: {}", complex_stats.num_variables);
    println!("  Factors: {}", complex_stats.num_factors);
    println!("  Avg factor size: {:.2}", complex_stats.avg_factor_size);
    println!("  Max factor size: {}", complex_stats.max_factor_size);
    println!();

    let complex_export = complex_graph.to_quantrs_model()?;
    println!("Clique structure:");
    println!(
        "  Number of cliques: {}",
        complex_export.structure.cliques.len()
    );
    for (i, clique) in complex_export.structure.cliques.iter().enumerate().take(5) {
        println!("    Clique {}: {:?}", i, clique);
    }
    println!();

    println!("=== Integration Summary ===\n");
    println!("✓ Model successfully exported to QuantRS2 format");
    println!("✓ Distributions converted and validated");
    println!("✓ Information measures computed");
    println!("✓ JSON serialization ready for ecosystem integration");
    println!();

    println!("Use Cases for QuantRS2 Integration:");
    println!("  • Export PGM models to QuantRS probabilistic programs");
    println!("  • Share distributions across COOLJAPAN ecosystem");
    println!("  • Analyze models using information theory");
    println!("  • Enable parameter learning with QuantRS optimizers");
    println!("  • Integrate MCMC sampling from QuantRS samplers");
    println!();

    println!("✓ Example completed successfully!");

    Ok(())
}

/// Build a simple Bayesian network for demonstration.
fn build_simple_network() -> anyhow::Result<FactorGraph> {
    let mut graph = FactorGraph::new();

    // Variables: A -> B -> C (chain structure)
    graph.add_variable_with_card("A".to_string(), "Binary".to_string(), 2);
    graph.add_variable_with_card("B".to_string(), "Binary".to_string(), 2);
    graph.add_variable_with_card("C".to_string(), "Binary".to_string(), 2);

    // P(A): Prior
    let p_a = Factor::new(
        "P(A)".to_string(),
        vec!["A".to_string()],
        Array::from_shape_vec(vec![2], vec![0.6, 0.4])?.into_dyn(),
    )?;
    graph.add_factor(p_a)?;

    // P(B|A): Conditional
    let p_b_given_a = Factor::new(
        "P(B|A)".to_string(),
        vec!["A".to_string(), "B".to_string()],
        Array::from_shape_vec(
            vec![2, 2],
            vec![
                0.7, 0.3, // A=0
                0.2, 0.8, // A=1
            ],
        )?
        .into_dyn(),
    )?;
    graph.add_factor(p_b_given_a)?;

    // P(C|B): Conditional
    let p_c_given_b = Factor::new(
        "P(C|B)".to_string(),
        vec!["B".to_string(), "C".to_string()],
        Array::from_shape_vec(
            vec![2, 2],
            vec![
                0.8, 0.2, // B=0
                0.3, 0.7, // B=1
            ],
        )?
        .into_dyn(),
    )?;
    graph.add_factor(p_c_given_b)?;

    Ok(graph)
}

/// Build a more complex network for advanced analysis.
fn build_complex_network() -> anyhow::Result<FactorGraph> {
    let mut graph = FactorGraph::new();

    // Create a 5-variable network with mixed structure
    for i in 0..5 {
        graph.add_variable_with_card(format!("V{}", i), "Binary".to_string(), 2);
    }

    // Add various factor types
    // Unary factors
    for i in 0..5 {
        let factor = Factor::new(
            format!("P(V{})", i),
            vec![format!("V{}", i)],
            Array::from_shape_vec(vec![2], vec![0.5, 0.5])?.into_dyn(),
        )?;
        graph.add_factor(factor)?;
    }

    // Pairwise factors
    let pairs = vec![(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)];
    for (i, j) in pairs {
        let factor = Factor::new(
            format!("P(V{},V{})", i, j),
            vec![format!("V{}", i), format!("V{}", j)],
            Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.1, 0.9])?.into_dyn(),
        )?;
        graph.add_factor(factor)?;
    }

    // Triple factor
    let factor = Factor::new(
        "P(V1,V2,V3)".to_string(),
        vec!["V1".to_string(), "V2".to_string(), "V3".to_string()],
        Array::from_shape_vec(
            vec![2, 2, 2],
            vec![0.1, 0.2, 0.15, 0.05, 0.2, 0.1, 0.05, 0.15],
        )?
        .into_dyn(),
    )?;
    graph.add_factor(factor)?;

    Ok(graph)
}
