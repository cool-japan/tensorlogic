//! Bayesian Network Example: Student Performance Model
//!
//! This example demonstrates building a Bayesian Network to model student performance
//! and using various inference algorithms to answer queries.

use scirs2_core::ndarray::Array;
use tensorlogic_quantrs_hooks::{
    BayesianNetwork, InferenceEngine, MarginalizationQuery, SumProductAlgorithm,
    VariableElimination,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Bayesian Network: Student Performance Model ===\n");

    // Create a Bayesian Network
    // Structure: Difficulty → Grade ← Intelligence
    //                          ↓
    //                        SAT
    let mut bn = BayesianNetwork::new();

    // Add variables
    bn.add_variable("Difficulty".to_string(), 2); // Easy=0, Hard=1
    bn.add_variable("Intelligence".to_string(), 2); // Low=0, High=1
    bn.add_variable("Grade".to_string(), 3); // A=0, B=1, C=2
    bn.add_variable("SAT".to_string(), 2); // Low=0, High=1

    println!("Building network structure...");

    // P(Difficulty) - Prior over course difficulty
    let p_difficulty = Array::from_shape_vec(vec![2], vec![0.6, 0.4])?.into_dyn();
    bn.add_prior("Difficulty".to_string(), p_difficulty)?;

    // P(Intelligence) - Prior over student intelligence
    let p_intelligence = Array::from_shape_vec(vec![2], vec![0.7, 0.3])?.into_dyn();
    bn.add_prior("Intelligence".to_string(), p_intelligence)?;

    // P(Grade | Difficulty, Intelligence)
    // Order: [Difficulty, Intelligence, Grade]
    // D=0,I=0: [0.3, 0.4, 0.3] (Easy, Low -> mostly B)
    // D=0,I=1: [0.9, 0.08, 0.02] (Easy, High -> mostly A)
    // D=1,I=0: [0.05, 0.25, 0.7] (Hard, Low -> mostly C)
    // D=1,I=1: [0.5, 0.3, 0.2] (Hard, High -> mixed)
    let p_grade = Array::from_shape_vec(
        vec![2, 2, 3],
        vec![
            0.3, 0.4, 0.3, // D=0, I=0
            0.9, 0.08, 0.02, // D=0, I=1
            0.05, 0.25, 0.7, // D=1, I=0
            0.5, 0.3, 0.2, // D=1, I=1
        ],
    )?
    .into_dyn();
    bn.add_cpd(
        "Grade".to_string(),
        vec!["Difficulty".to_string(), "Intelligence".to_string()],
        p_grade,
    )?;

    // P(SAT | Intelligence)
    // I=0: [0.95, 0.05] (Low intelligence -> mostly low SAT)
    // I=1: [0.2, 0.8] (High intelligence -> mostly high SAT)
    let p_sat = Array::from_shape_vec(
        vec![2, 2],
        vec![
            0.95, 0.05, // I=0
            0.2, 0.8, // I=1
        ],
    )?
    .into_dyn();
    bn.add_cpd("SAT".to_string(), vec!["Intelligence".to_string()], p_sat)?;

    // Verify network structure
    println!(
        "Network has {} variables and {} factors",
        bn.graph().num_variables(),
        bn.graph().num_factors()
    );
    assert!(bn.is_acyclic(), "Network must be acyclic!");
    println!("✓ Network is a valid DAG\n");

    // Get topological order
    let topo_order = bn.topological_order()?;
    println!("Topological order: {:?}\n", topo_order);

    // === Query 1: Marginal probabilities using Sum-Product ===
    println!("=== Query 1: What's the probability distribution over Grades? ===");
    let algorithm = Box::new(SumProductAlgorithm::default());
    let engine = InferenceEngine::new(bn.graph().clone(), algorithm);

    let query = MarginalizationQuery {
        variable: "Grade".to_string(),
    };
    let grade_marginal = engine.marginalize(&query)?;

    println!("P(Grade):");
    println!("  P(Grade=A) = {:.3}", grade_marginal[[0]]);
    println!("  P(Grade=B) = {:.3}", grade_marginal[[1]]);
    println!("  P(Grade=C) = {:.3}\n", grade_marginal[[2]]);

    // === Query 2: Using Variable Elimination ===
    println!("=== Query 2: What's the probability distribution over SAT scores? ===");
    let ve = VariableElimination::new();
    let sat_marginal = ve.marginalize(bn.graph(), "SAT")?;

    println!("P(SAT):");
    println!("  P(SAT=Low) = {:.3}", sat_marginal[[0]]);
    println!("  P(SAT=High) = {:.3}\n", sat_marginal[[1]]);

    // === Query 3: Conditional Probability with Evidence ===
    println!("=== Query 3: If we observe SAT=High, what's the distribution over Intelligence? ===");

    // To compute P(Intelligence | SAT=High), we need to add evidence
    // This is done by multiplying with an indicator factor
    use tensorlogic_quantrs_hooks::Factor;
    let mut evidence_graph = bn.graph().clone();

    // Add evidence: SAT=High (value 1)
    let evidence_values = Array::from_shape_vec(vec![2], vec![0.0, 1.0])?.into_dyn();
    let evidence = Factor::new(
        "Evidence_SAT".to_string(),
        vec!["SAT".to_string()],
        evidence_values,
    )?;
    evidence_graph.add_factor(evidence)?;

    let ve = VariableElimination::new();
    let intel_given_sat = ve.marginalize(&evidence_graph, "Intelligence")?;

    println!("P(Intelligence | SAT=High):");
    println!(
        "  P(Intelligence=Low | SAT=High) = {:.3}",
        intel_given_sat[[0]]
    );
    println!(
        "  P(Intelligence=High | SAT=High) = {:.3}\n",
        intel_given_sat[[1]]
    );

    // === Query 4: Joint Probability ===
    println!("=== Query 4: Computing joint probability ===");
    let joint = engine.joint()?;
    println!("Joint distribution shape: {:?}", joint.shape());
    let joint_sum: f64 = joint.iter().sum();
    println!(
        "Joint distribution sums to: {:.6} (should be ~1.0)\n",
        joint_sum
    );

    // === Summary ===
    println!("=== Summary ===");
    println!("✓ Built a 4-variable Bayesian Network");
    println!("✓ Verified DAG property and computed topological order");
    println!("✓ Performed marginal inference using Sum-Product");
    println!("✓ Performed marginal inference using Variable Elimination");
    println!("✓ Computed conditional probabilities with evidence");
    println!("✓ Computed joint distribution over all variables");

    Ok(())
}
