//! Parameter learning example for Hidden Markov Models.
//!
//! This example demonstrates the Baum-Welch algorithm (EM for HMMs) by learning
//! the parameters of a weather prediction model from observed sequences.
//!
//! # Scenario
//!
//! We have a simple weather model with:
//! - 2 hidden states: Sunny (0), Rainy (1)
//! - 3 observations: Walk (0), Shop (1), Clean (2)
//!
//! We'll generate synthetic data from a known HMM, then attempt to recover the
//! parameters using only the observation sequences (Baum-Welch algorithm).

use tensorlogic_quantrs_hooks::{BaumWelchLearner, SimpleHMM};

fn main() -> anyhow::Result<()> {
    println!("=== HMM Parameter Learning Example ===\n");

    // Create the "true" HMM (the model we'll try to learn)
    println!("Creating true weather model...");
    let true_hmm = create_true_weather_model();

    println!("True Model Parameters:");
    println!("---------------------");
    print_hmm_parameters(&true_hmm);
    println!();

    // Generate observation sequences from the true model
    println!("Generating observation sequences from true model...");
    let num_sequences = 50;
    let sequence_length = 20;
    let observation_sequences = generate_observations(&true_hmm, num_sequences, sequence_length);

    println!(
        "Generated {} sequences of length {}",
        num_sequences, sequence_length
    );
    println!("Sample sequence: {:?}", observation_sequences[0]);
    println!();

    // Initialize a random HMM to learn
    println!("Initializing random HMM for learning...");
    let mut learned_hmm = SimpleHMM::new_random(2, 3);

    println!("Initial (Random) Parameters:");
    println!("---------------------------");
    print_hmm_parameters(&learned_hmm);
    println!();

    // Learn parameters using Baum-Welch
    println!("=== Learning Parameters with Baum-Welch ===\n");
    let learner = BaumWelchLearner::with_verbose(100, 1e-4);

    let final_log_likelihood = learner.learn(&mut learned_hmm, &observation_sequences)?;

    println!("\nFinal log-likelihood: {:.4}", final_log_likelihood);
    println!();

    // Display learned parameters
    println!("Learned Model Parameters:");
    println!("------------------------");
    print_hmm_parameters(&learned_hmm);
    println!();

    // Compare with true parameters
    println!("=== Parameter Comparison ===\n");
    compare_parameters(&true_hmm, &learned_hmm);

    // Test the learned model
    println!("\n=== Testing Learned Model ===\n");
    test_model_predictions(&true_hmm, &learned_hmm);

    println!("\n✓ Parameter learning completed successfully!");

    Ok(())
}

/// Create the true weather HMM we want to learn.
fn create_true_weather_model() -> SimpleHMM {
    use scirs2_core::ndarray::{Array1, Array2};

    let mut hmm = SimpleHMM::new(2, 3);

    // Initial distribution: P(Sunny) = 0.6, P(Rainy) = 0.4
    hmm.initial_distribution = Array1::from_vec(vec![0.6, 0.4]);

    // Transition probabilities:
    // From Sunny: 80% stay sunny, 20% become rainy
    // From Rainy: 40% become sunny, 60% stay rainy
    hmm.transition_probabilities = Array2::from_shape_vec(
        (2, 2),
        vec![
            0.8, 0.2, // From Sunny
            0.4, 0.6, // From Rainy
        ],
    )
    .unwrap();

    // Emission probabilities:
    // Sunny:  60% walk, 30% shop, 10% clean
    // Rainy:  10% walk, 20% shop, 70% clean
    hmm.emission_probabilities = Array2::from_shape_vec(
        (2, 3),
        vec![
            0.6, 0.3, 0.1, // Sunny -> Walk, Shop, Clean
            0.1, 0.2, 0.7, // Rainy -> Walk, Shop, Clean
        ],
    )
    .unwrap();

    hmm
}

/// Generate observation sequences from an HMM.
fn generate_observations(hmm: &SimpleHMM, num_sequences: usize, length: usize) -> Vec<Vec<usize>> {
    use scirs2_core::random::thread_rng;

    let mut rng = thread_rng();
    let mut sequences = Vec::new();

    for _ in 0..num_sequences {
        let mut sequence = Vec::new();
        let mut state = sample_discrete(&hmm.initial_distribution.to_vec(), &mut rng);

        for _ in 0..length {
            // Emit observation from current state
            let emission_probs = hmm.emission_probabilities.row(state).to_vec();
            let observation = sample_discrete(&emission_probs, &mut rng);
            sequence.push(observation);

            // Transition to next state
            let transition_probs = hmm.transition_probabilities.row(state).to_vec();
            state = sample_discrete(&transition_probs, &mut rng);
        }

        sequences.push(sequence);
    }

    sequences
}

/// Sample from a discrete distribution.
fn sample_discrete(probs: &[f64], rng: &mut impl scirs2_core::Rng) -> usize {
    let u: f64 = rng.random();
    let mut cumsum = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i;
        }
    }

    probs.len() - 1
}

/// Print HMM parameters in a readable format.
fn print_hmm_parameters(hmm: &SimpleHMM) {
    println!("Initial Distribution:");
    println!("  P(Sunny) = {:.3}", hmm.initial_distribution[0]);
    println!("  P(Rainy) = {:.3}", hmm.initial_distribution[1]);

    println!("\nTransition Probabilities:");
    println!(
        "  Sunny -> Sunny: {:.3}",
        hmm.transition_probabilities[[0, 0]]
    );
    println!(
        "  Sunny -> Rainy: {:.3}",
        hmm.transition_probabilities[[0, 1]]
    );
    println!(
        "  Rainy -> Sunny: {:.3}",
        hmm.transition_probabilities[[1, 0]]
    );
    println!(
        "  Rainy -> Rainy: {:.3}",
        hmm.transition_probabilities[[1, 1]]
    );

    println!("\nEmission Probabilities:");
    println!(
        "  Sunny -> Walk:  {:.3}",
        hmm.emission_probabilities[[0, 0]]
    );
    println!(
        "  Sunny -> Shop:  {:.3}",
        hmm.emission_probabilities[[0, 1]]
    );
    println!(
        "  Sunny -> Clean: {:.3}",
        hmm.emission_probabilities[[0, 2]]
    );
    println!(
        "  Rainy -> Walk:  {:.3}",
        hmm.emission_probabilities[[1, 0]]
    );
    println!(
        "  Rainy -> Shop:  {:.3}",
        hmm.emission_probabilities[[1, 1]]
    );
    println!(
        "  Rainy -> Clean: {:.3}",
        hmm.emission_probabilities[[1, 2]]
    );
}

/// Compare learned parameters with true parameters.
fn compare_parameters(true_hmm: &SimpleHMM, learned_hmm: &SimpleHMM) {
    println!("Parameter Errors (Absolute Difference):\n");

    // Initial distribution error
    let init_error_0 =
        (true_hmm.initial_distribution[0] - learned_hmm.initial_distribution[0]).abs();
    let init_error_1 =
        (true_hmm.initial_distribution[1] - learned_hmm.initial_distribution[1]).abs();

    println!("Initial Distribution:");
    println!(
        "  P(Sunny): true={:.3}, learned={:.3}, error={:.3}",
        true_hmm.initial_distribution[0], learned_hmm.initial_distribution[0], init_error_0
    );
    println!(
        "  P(Rainy): true={:.3}, learned={:.3}, error={:.3}",
        true_hmm.initial_distribution[1], learned_hmm.initial_distribution[1], init_error_1
    );

    // Transition probabilities error
    println!("\nTransition Probabilities:");
    for i in 0..2 {
        for j in 0..2 {
            let state_names = ["Sunny", "Rainy"];
            let true_val = true_hmm.transition_probabilities[[i, j]];
            let learned_val = learned_hmm.transition_probabilities[[i, j]];
            let error = (true_val - learned_val).abs();

            println!(
                "  {} -> {}: true={:.3}, learned={:.3}, error={:.3}",
                state_names[i], state_names[j], true_val, learned_val, error
            );
        }
    }

    // Emission probabilities error
    println!("\nEmission Probabilities:");
    let states = ["Sunny", "Rainy"];
    let observations = ["Walk", "Shop", "Clean"];

    for (i, state) in states.iter().enumerate() {
        for (j, obs) in observations.iter().enumerate() {
            let true_val = true_hmm.emission_probabilities[[i, j]];
            let learned_val = learned_hmm.emission_probabilities[[i, j]];
            let error = (true_val - learned_val).abs();

            println!(
                "  {} -> {}: true={:.3}, learned={:.3}, error={:.3}",
                state, obs, true_val, learned_val, error
            );
        }
    }

    // Compute average error
    let mut total_error = 0.0;
    let mut count = 0;

    for i in 0..2 {
        total_error +=
            (true_hmm.initial_distribution[i] - learned_hmm.initial_distribution[i]).abs();
        count += 1;
    }

    for i in 0..2 {
        for j in 0..2 {
            total_error += (true_hmm.transition_probabilities[[i, j]]
                - learned_hmm.transition_probabilities[[i, j]])
            .abs();
            count += 1;
        }
    }

    for i in 0..2 {
        for j in 0..3 {
            total_error += (true_hmm.emission_probabilities[[i, j]]
                - learned_hmm.emission_probabilities[[i, j]])
            .abs();
            count += 1;
        }
    }

    let avg_error = total_error / count as f64;
    println!("\nAverage absolute error: {:.4}", avg_error);
}

/// Test model predictions.
fn test_model_predictions(true_hmm: &SimpleHMM, learned_hmm: &SimpleHMM) {
    let test_sequence = vec![0, 0, 1, 2, 2, 0]; // Walk, Walk, Shop, Clean, Clean, Walk

    println!("Test sequence: {:?}", test_sequence);
    println!("(0=Walk, 1=Shop, 2=Clean)\n");

    // Compute forward probabilities for both models
    println!("Comparing likelihood under both models:");

    let true_likelihood = compute_sequence_likelihood(true_hmm, &test_sequence);
    let learned_likelihood = compute_sequence_likelihood(learned_hmm, &test_sequence);

    println!(
        "  True model log-likelihood:    {:.4}",
        true_likelihood.ln()
    );
    println!(
        "  Learned model log-likelihood: {:.4}",
        learned_likelihood.ln()
    );
    println!(
        "  Difference:                   {:.4}",
        (true_likelihood - learned_likelihood).ln()
    );
}

/// Compute the likelihood of a sequence under an HMM.
fn compute_sequence_likelihood(hmm: &SimpleHMM, sequence: &[usize]) -> f64 {
    let num_states = hmm.num_states;
    let mut alpha = vec![0.0; num_states];

    // Initialize
    for (s, alpha_val) in alpha.iter_mut().enumerate().take(num_states) {
        *alpha_val = hmm.initial_distribution[s] * hmm.emission_probabilities[[s, sequence[0]]];
    }

    // Forward recursion
    for &obs in sequence.iter().skip(1) {
        let mut new_alpha = vec![0.0; num_states];
        for (s2, new_val) in new_alpha.iter_mut().enumerate().take(num_states) {
            let mut sum = 0.0;
            for (s1, &alpha_val) in alpha.iter().enumerate().take(num_states) {
                sum += alpha_val * hmm.transition_probabilities[[s1, s2]];
            }
            *new_val = sum * hmm.emission_probabilities[[s2, obs]];
        }
        alpha = new_alpha;
    }

    alpha.iter().sum()
}
