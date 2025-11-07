//! Linear-chain CRF Example: Part-of-Speech Tagging
//!
//! This example demonstrates using linear-chain CRFs for sequence labeling,
//! specifically a simplified part-of-speech tagging scenario.

use scirs2_core::ndarray::Array;
use tensorlogic_quantrs_hooks::{
    EmissionFeature, FeatureFunction, LinearChainCRF, TransitionFeature,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Linear-chain CRF Examples ===\n");

    // Example 1: Simple POS tagging
    println!("Example 1: Simplified Part-of-Speech Tagging");
    println!("--------------------------------------------");
    pos_tagging_example()?;

    println!("\n");

    // Example 2: Named Entity Recognition
    println!("Example 2: Named Entity Recognition (NER)");
    println!("-----------------------------------------");
    ner_example()?;

    println!("\n");

    // Example 3: Custom features
    println!("Example 3: Custom Feature Functions");
    println!("-----------------------------------");
    custom_features_example()?;

    Ok(())
}

/// Example 1: Simplified POS tagging
/// States: 0=Noun, 1=Verb, 2=Adjective
/// Observations: simplified word indices
fn pos_tagging_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Scenario: Tagging a simple sentence");
    println!("States: 0=Noun, 1=Verb, 2=Adjective");
    println!();

    // Create CRF with 3 states
    let mut crf = LinearChainCRF::new(3);

    // Set transition weights (learned from training data)
    // Rows: from state, Columns: to state
    // These weights encode linguistic patterns:
    // - Nouns often follow adjectives
    // - Verbs often follow nouns
    // - Adjectives often precede nouns
    let transition_weights = Array::from_shape_vec(
        vec![3, 3],
        vec![
            // From Noun:  to N,  V,  Adj
            0.2, 0.5, 0.3, // Noun → Verb is common
            // From Verb:
            0.6, 0.1, 0.3, // Verb → Noun is common
            // From Adj:
            0.7, 0.1, 0.2, // Adj → Noun is very common
        ],
    )?
    .into_dimensionality::<scirs2_core::ndarray::Ix2>()?;

    crf.set_transition_weights(transition_weights)?;

    // Set emission weights (word → POS probabilities)
    // For simplicity, we use a small vocabulary
    let emission_weights = Array::from_shape_vec(
        vec![3, 5], // 3 states × 5 words
        vec![
            // Word 0  1    2    3    4
            0.8, 0.1, 0.1, 0.7, 0.2, // Noun probabilities
            0.1, 0.8, 0.1, 0.1, 0.7, // Verb probabilities
            0.1, 0.1, 0.8, 0.2, 0.1, // Adjective probabilities
        ],
    )?
    .into_dimensionality::<scirs2_core::ndarray::Ix2>()?;

    crf.set_emission_weights(emission_weights)?;

    // Test sentence: "The quick brown fox jumps"
    // Encoded as: [3, 4, 4, 3, 1]
    let sentence = vec![3, 4, 4, 3, 1];
    let words = ["The", "quick", "brown", "fox", "jumps"];

    println!("Input sentence: {}", words.join(" "));
    println!("Word indices:   {:?}", sentence);
    println!();

    // Viterbi decoding (most likely sequence)
    let (best_tags, score) = crf.viterbi(&sentence)?;

    let tag_names = ["Noun", "Verb", "Adj"];
    println!("Viterbi decoding (most likely sequence):");
    println!("Score: {:.4}", score);
    println!();
    for (i, (&word, &tag)) in words.iter().zip(best_tags.iter()).enumerate() {
        println!("  Position {}: {:10} → {}", i, word, tag_names[tag]);
    }

    // Compute marginal probabilities
    println!("\nMarginal probabilities for each position:");
    let marginals = crf.marginals(&sentence)?;

    for (i, word) in words.iter().enumerate() {
        println!("\n  Position {} ({}):", i, word);
        for (tag_idx, tag_name) in tag_names.iter().enumerate() {
            let prob = marginals[[i, tag_idx]];
            let bar = "█".repeat((prob * 20.0) as usize);
            println!("    {:<10} {:.4} {}", tag_name, prob, bar);
        }
    }

    Ok(())
}

/// Example 2: Named Entity Recognition
/// States: 0=Outside, 1=Person, 2=Location, 3=Organization
fn ner_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Scenario: Recognizing named entities in text");
    println!("States: 0=Outside, 1=Person, 2=Location, 3=Organization");
    println!();

    let mut crf = LinearChainCRF::new(4);

    // NER-specific transition patterns
    // Entities tend to cluster, and transitions between entity types are rare
    let transition_weights = Array::from_shape_vec(
        vec![4, 4],
        vec![
            // From Outside:
            0.7, 0.1, 0.1, 0.1, // Usually stays outside
            // From Person:
            0.4, 0.5, 0.05, 0.05, // Person tags often cluster
            // From Location:
            0.4, 0.05, 0.5, 0.05, // Location tags cluster
            // From Organization:
            0.4, 0.05, 0.05, 0.5, // Org tags cluster
        ],
    )?
    .into_dimensionality::<scirs2_core::ndarray::Ix2>()?;

    crf.set_transition_weights(transition_weights)?;

    // Simplified emission weights
    let emission_weights = Array::from_shape_vec(
        vec![4, 6], // 4 states × 6 words
        vec![
            // Common words tend to be Outside
            0.8, 0.1, 0.2, 0.7, 0.2, 0.9, // Outside
            0.05, 0.7, 0.1, 0.1, 0.1, 0.05, // Person (word 1 is likely a name)
            0.1, 0.1, 0.6, 0.1, 0.6, 0.05, // Location (words 2,4 are places)
            0.05, 0.1, 0.1, 0.1, 0.1, 0.0, // Organization
        ],
    )?
    .into_dimensionality::<scirs2_core::ndarray::Ix2>()?;

    crf.set_emission_weights(emission_weights)?;

    // Test sentence: "John visited Paris last week"
    let sentence = vec![1, 5, 2, 5, 5]; // 1=John, 2=Paris, 5=common words
    let words = ["John", "visited", "Paris", "last", "week"];

    println!("Input: {}", words.join(" "));
    println!();

    let (tags, score) = crf.viterbi(&sentence)?;

    let tag_names = ["O", "PER", "LOC", "ORG"];
    println!("NER tagging (Viterbi):");
    println!("Score: {:.4}", score);
    println!();
    for (&word, &tag) in words.iter().zip(tags.iter()) {
        println!("  {:10} → {}", word, tag_names[tag]);
    }

    Ok(())
}

/// Example 3: Custom feature functions
fn custom_features_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Building CRF with custom feature functions");
    println!();

    // Define a custom feature: word starts with capital letter
    struct CapitalLetterFeature {
        state: usize,
        observation: usize,
    }

    impl FeatureFunction for CapitalLetterFeature {
        fn compute(
            &self,
            _prev_label: Option<usize>,
            curr_label: usize,
            input_sequence: &[usize],
            position: usize,
        ) -> f64 {
            if curr_label == self.state
                && position < input_sequence.len()
                && input_sequence[position] == self.observation
            {
                1.0
            } else {
                0.0
            }
        }

        fn name(&self) -> &str {
            "capital_letter"
        }
    }

    // Create CRF
    let mut crf = LinearChainCRF::new(3);

    // Add transition features
    println!("Adding transition features:");
    for from_state in 0..3 {
        for to_state in 0..3 {
            let feature = Box::new(TransitionFeature::new(from_state, to_state));
            let weight = if from_state == to_state { 0.5 } else { 0.1 };
            println!(
                "  Transition {}→{}: weight = {:.2}",
                from_state, to_state, weight
            );
            crf.add_feature(feature, weight);
        }
    }

    println!();
    println!("Adding emission features:");
    // Add emission features
    for state in 0..3 {
        for obs in 0..4 {
            let feature = Box::new(EmissionFeature::new(state, obs));
            let weight = if state == obs % 3 { 0.8 } else { 0.2 };
            println!(
                "  Emission state={}, obs={}: weight = {:.2}",
                state, obs, weight
            );
            crf.add_feature(feature, weight);
        }
    }

    println!();
    println!("Adding custom capital letter features:");
    // Add custom features
    for state in 0..2 {
        let feature = Box::new(CapitalLetterFeature {
            state,
            observation: 5, // Special observation for capital letters
        });
        println!("  Capital letter → state {}: weight = 1.5", state);
        crf.add_feature(feature, 1.5);
    }

    // Test sequence
    let test_sequence = vec![0, 1, 2, 1, 0];
    println!();
    println!("Testing on sequence: {:?}", test_sequence);

    let (path, score) = crf.viterbi(&test_sequence)?;
    println!();
    println!("Predicted path: {:?}", path);
    println!("Path score: {:.4}", score);

    // Show marginals
    let marginals = crf.marginals(&test_sequence)?;
    println!();
    println!("Marginal probabilities:");
    for t in 0..test_sequence.len() {
        print!("  Position {}: ", t);
        for s in 0..3 {
            print!("P({}={:.3}) ", s, marginals[[t, s]]);
        }
        println!();
    }

    println!();
    println!("✓ Custom features allow domain-specific modeling");

    Ok(())
}
