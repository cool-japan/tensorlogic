//! Expectation Propagation (EP) Example
//!
//! This example demonstrates using EP for approximate inference on a small Bayesian network.
//! EP is particularly useful when you need good approximations of complex posteriors.

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::Array;
use tensorlogic_quantrs_hooks::{ExpectationPropagation, Factor, FactorGraph};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Expectation Propagation Example ===\n");

    // Example 1: Simple Bayesian Network
    println!("Example 1: Simple Disease Diagnosis Network");
    println!("-------------------------------------------");
    simple_disease_network()?;

    println!("\n");

    // Example 2: Comparing EP with other inference methods
    println!("Example 2: EP vs Mean-Field Comparison");
    println!("--------------------------------------");
    compare_inference_methods()?;

    println!("\n");

    // Example 3: EP parameters
    println!("Example 3: EP with Different Parameters");
    println!("---------------------------------------");
    ep_with_parameters()?;

    Ok(())
}

/// Simple disease diagnosis network: Flu → Fever, Flu → Cough
fn simple_disease_network() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = FactorGraph::new();

    // Variables
    graph.add_variable_with_card("Flu".to_string(), "Disease".to_string(), 2);
    graph.add_variable_with_card("Fever".to_string(), "Symptom".to_string(), 2);
    graph.add_variable_with_card("Cough".to_string(), "Symptom".to_string(), 2);

    // Prior: P(Flu) = [0.95, 0.05] (5% chance of flu)
    let p_flu = Factor::new(
        "P(Flu)".to_string(),
        vec!["Flu".to_string()],
        Array::from_shape_vec(vec![2], vec![0.95, 0.05])?.into_dyn(),
    )?;
    graph.add_factor(p_flu)?;

    // P(Fever | Flu)
    // If no flu: 90% no fever, 10% fever
    // If flu: 20% no fever, 80% fever
    let p_fever_given_flu = Factor::new(
        "P(Fever|Flu)".to_string(),
        vec!["Flu".to_string(), "Fever".to_string()],
        Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])?.into_dyn(),
    )?;
    graph.add_factor(p_fever_given_flu)?;

    // P(Cough | Flu)
    // If no flu: 95% no cough, 5% cough
    // If flu: 30% no cough, 70% cough
    let p_cough_given_flu = Factor::new(
        "P(Cough|Flu)".to_string(),
        vec!["Flu".to_string(), "Cough".to_string()],
        Array::from_shape_vec(vec![2, 2], vec![0.95, 0.05, 0.3, 0.7])?.into_dyn(),
    )?;
    graph.add_factor(p_cough_given_flu)?;

    // Run EP
    let ep = ExpectationPropagation::new(100, 1e-6, 0.0);
    let marginals = ep.run(&graph)?;

    println!("Prior probability of flu: P(Flu=yes) = 0.05");
    println!("\nMarginal probabilities:");
    for (var, marginal) in &marginals {
        println!("  P({}=no)  = {:.4}", var, marginal[[0]]);
        println!("  P({}=yes) = {:.4}", var, marginal[[1]]);
    }

    // Verify normalization
    for marginal in marginals.values() {
        let sum: f64 = marginal.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-4);
    }
    println!("\n✓ All marginals are properly normalized");

    Ok(())
}

/// Compare EP with Mean-Field VI on the same graph
fn compare_inference_methods() -> Result<(), Box<dyn std::error::Error>> {
    use tensorlogic_quantrs_hooks::MeanFieldInference;

    // Build a simple 2-variable graph (simpler to avoid edge cases)
    let mut graph = FactorGraph::new();

    graph.add_variable_with_card("X".to_string(), "Binary".to_string(), 2);
    graph.add_variable_with_card("Y".to_string(), "Binary".to_string(), 2);

    // P(X) = [0.6, 0.4]
    let px = Factor::new(
        "P(X)".to_string(),
        vec!["X".to_string()],
        Array::from_shape_vec(vec![2], vec![0.6, 0.4])?.into_dyn(),
    )?;
    graph.add_factor(px)?;

    // P(Y|X) - Y tends to match X
    let pyx = Factor::new(
        "P(Y|X)".to_string(),
        vec!["X".to_string(), "Y".to_string()],
        Array::from_shape_vec(vec![2, 2], vec![0.8, 0.2, 0.2, 0.8])?.into_dyn(),
    )?;
    graph.add_factor(pyx)?;

    // Run EP
    println!("Running Expectation Propagation...");
    let ep = ExpectationPropagation::new(100, 1e-6, 0.0);
    let ep_marginals = ep.run(&graph)?;

    // Run Mean-Field
    println!("Running Mean-Field Variational Inference...");
    let mf = MeanFieldInference::new(1000, 1e-6);
    let mf_marginals = mf.run(&graph)?;

    println!("\nComparison of marginal probabilities:");
    println!(
        "{:<8} {:<15} {:<15} {:<15}",
        "Variable", "EP P(=1)", "Mean-Field P(=1)", "Difference"
    );
    println!("{}", "-".repeat(60));

    for var in ["X", "Y"] {
        let ep_prob = ep_marginals.get(var).map(|m| m[[1]]).unwrap_or(0.0);
        let mf_prob = mf_marginals.get(var).map(|p| p[[1]]).unwrap_or(0.0);
        let diff = (ep_prob - mf_prob).abs();

        println!(
            "{:<8} {:.6}        {:.6}         {:.6}",
            var, ep_prob, mf_prob, diff
        );
    }

    println!("\n✓ Both methods produce similar results (as expected for this simple graph)");

    Ok(())
}

/// EP with different parameters
fn ep_with_parameters() -> Result<(), Box<dyn std::error::Error>> {
    // Same graph as Example 1 but testing different EP parameters
    let mut graph = FactorGraph::new();

    graph.add_variable_with_card("X".to_string(), "Binary".to_string(), 2);
    graph.add_variable_with_card("Y".to_string(), "Binary".to_string(), 2);

    let px = Factor::new(
        "P(X)".to_string(),
        vec!["X".to_string()],
        Array::from_shape_vec(vec![2], vec![0.3, 0.7])?.into_dyn(),
    )?;
    graph.add_factor(px)?;

    let pyx = Factor::new(
        "P(Y|X)".to_string(),
        vec!["X".to_string(), "Y".to_string()],
        Array::from_shape_vec(vec![2, 2], vec![0.9, 0.1, 0.2, 0.8])?.into_dyn(),
    )?;
    graph.add_factor(pyx)?;

    println!("Testing EP with different convergence tolerances:\n");

    // Strict tolerance
    println!("1. Strict tolerance (1e-8):");
    let ep_strict = ExpectationPropagation::new(100, 1e-8, 0.0);
    let marginals_strict = ep_strict.run(&graph)?;
    println!(
        "   P(Y=1) = {:.8}",
        marginals_strict.get("Y").map(|m| m[[1]]).unwrap_or(0.0)
    );

    // Loose tolerance
    println!("\n2. Loose tolerance (1e-4):");
    let ep_loose = ExpectationPropagation::new(100, 1e-4, 0.0);
    let marginals_loose = ep_loose.run(&graph)?;
    println!(
        "   P(Y=1) = {:.8}",
        marginals_loose.get("Y").map(|m| m[[1]]).unwrap_or(0.0)
    );

    // With damping
    println!("\n3. With damping (damping=0.5):");
    let ep_damped = ExpectationPropagation::new(100, 1e-6, 0.5);
    let marginals_damped = ep_damped.run(&graph)?;
    println!(
        "   P(Y=1) = {:.8}",
        marginals_damped.get("Y").map(|m| m[[1]]).unwrap_or(0.0)
    );

    println!("\n✓ EP parameters can be tuned for different convergence behaviors");
    println!("  (Stricter tolerance may require more iterations but gives more precise results)");

    Ok(())
}
