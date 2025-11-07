//! Hidden Markov Model Example: Weather Prediction
//!
//! This example demonstrates using HMMs for temporal inference:
//! - Filtering: P(state_t | obs_0:t)
//! - Smoothing: P(state_t | obs_0:T)
//! - Viterbi: Most likely state sequence

use scirs2_core::ndarray::Array;
use tensorlogic_quantrs_hooks::HiddenMarkovModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Hidden Markov Model: Weather Prediction ===\n");

    // Problem: Predict weather (Sunny/Rainy) based on umbrella observations
    // Hidden states: Weather (Sunny=0, Rainy=1)
    // Observations: Umbrella (No=0, Yes=1)

    let num_states = 2; // Sunny, Rainy
    let num_obs = 2; // No umbrella, Has umbrella
    let time_steps = 5; // 5 day sequence

    let mut hmm = HiddenMarkovModel::new(num_states, num_obs, time_steps);

    println!(
        "Building HMM with {} states, {} observations, {} time steps",
        num_states, num_obs, time_steps
    );

    // Initial state distribution: P(Weather_0)
    // More likely to start sunny
    let initial = Array::from_shape_vec(
        vec![2],
        vec![0.6, 0.4], // P(Sunny)=0.6, P(Rainy)=0.4
    )?
    .into_dyn();
    hmm.set_initial_distribution(initial)?;
    println!("✓ Initial distribution: P(Sunny)=0.6, P(Rainy)=0.4");

    // Transition matrix: P(Weather_t | Weather_{t-1})
    // [from_state, to_state]
    // Weather tends to persist
    let transition = Array::from_shape_vec(
        vec![2, 2],
        vec![
            0.7, 0.3, // Sunny -> [Sunny=0.7, Rainy=0.3]
            0.4, 0.6, // Rainy -> [Sunny=0.4, Rainy=0.6]
        ],
    )?
    .into_dyn();
    hmm.set_transition_matrix(transition)?;
    println!("✓ Transition: Sunny→Sunny=0.7, Sunny→Rainy=0.3");
    println!("              Rainy→Sunny=0.4, Rainy→Rainy=0.6");

    // Emission matrix: P(Umbrella_t | Weather_t)
    // [state, observation]
    // People usually bring umbrellas when rainy
    let emission = Array::from_shape_vec(
        vec![2, 2],
        vec![
            0.9, 0.1, // Sunny -> [No=0.9, Yes=0.1]
            0.2, 0.8, // Rainy -> [No=0.2, Yes=0.8]
        ],
    )?
    .into_dyn();
    hmm.set_emission_matrix(emission)?;
    println!("✓ Emission: Sunny→NoUmbrella=0.9, Sunny→Umbrella=0.1");
    println!("            Rainy→NoUmbrella=0.2, Rainy→Umbrella=0.8\n");

    // Observation sequence: [Yes, Yes, No, Yes, Yes]
    // Someone brought umbrella on days 0,1,3,4 but not day 2
    let observations = vec![1, 1, 0, 1, 1];
    println!("Observation sequence (5 days):");
    println!("  Day 0: Umbrella");
    println!("  Day 1: Umbrella");
    println!("  Day 2: No umbrella");
    println!("  Day 3: Umbrella");
    println!("  Day 4: Umbrella\n");

    // === Filtering: P(Weather_t | obs_0:t) ===
    println!("=== Filtering (online inference) ===");
    println!("What's the weather distribution at each time given observations so far?");
    for t in 0..time_steps {
        let filtered = hmm.filter(&observations, t)?;
        println!(
            "Day {}: P(Sunny|obs_0:{}) = {:.3}, P(Rainy|obs_0:{}) = {:.3}",
            t,
            t,
            filtered[[0]],
            t,
            filtered[[1]]
        );
    }
    println!();

    // === Smoothing: P(Weather_t | obs_0:T) ===
    println!("=== Smoothing (offline inference with all observations) ===");
    println!("What's the weather distribution at each time given ALL observations?");
    for t in 0..time_steps {
        let smoothed = hmm.smooth(&observations, t)?;
        println!(
            "Day {}: P(Sunny|all obs) = {:.3}, P(Rainy|all obs) = {:.3}",
            t,
            smoothed[[0]],
            smoothed[[1]]
        );
    }
    println!();

    // === Viterbi: Most likely state sequence ===
    println!("=== Viterbi (most likely weather sequence) ===");
    let viterbi_path = hmm.viterbi(&observations)?;
    println!("Most probable weather sequence:");
    for (t, &state) in viterbi_path.iter().enumerate() {
        let weather = if state == 0 { "Sunny" } else { "Rainy" };
        let umbrella = if observations[t] == 0 { "No" } else { "Yes" };
        println!("  Day {}: {} (observed: {})", t, weather, umbrella);
    }
    println!();

    // === Analysis ===
    println!("=== Analysis ===");
    println!("Filtering vs Smoothing:");
    println!("- Filtering uses only past observations (causal)");
    println!("- Smoothing uses all observations (more accurate)");
    println!();
    println!("Notice:");
    let day2_filtered = hmm.filter(&observations, 2)?;
    let day2_smoothed = hmm.smooth(&observations, 2)?;
    println!("Day 2 (no umbrella observed):");
    println!("  Filtered:  P(Sunny) = {:.3}", day2_filtered[[0]]);
    println!("  Smoothed:  P(Sunny) = {:.3}", day2_smoothed[[0]]);
    println!("  → Smoothing gives higher confidence in Sunny (uses future obs)");

    println!("\n✓ Demonstrated filtering, smoothing, and Viterbi on HMM");

    Ok(())
}
