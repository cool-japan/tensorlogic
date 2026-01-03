//! Example: Few-Shot Learning
//!
//! This example demonstrates how to use few-shot learning helpers to learn
//! from minimal examples. This is useful when you have very limited labeled
//! data for new classes.
//!
//! Run with: cargo run --example 17_few_shot_learning

use scirs2_core::ndarray::{Array1, Array2};
use tensorlogic_train::{
    DistanceMetric, EpisodeSampler, FewShotAccuracy, MatchingNetwork, PrototypicalDistance,
    ShotType, SupportSet,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Few-Shot Learning Examples ===\n");

    // Example 1: Basic Support Set
    println!("1. Creating Support Set:");
    println!("   (Small labeled dataset for adaptation)\n");

    // Create a simple support set: 2 classes, 3 examples each
    // Class 0: points near [1, 1]
    // Class 1: points near [5, 5]
    let support_features = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 1.0, // Class 0
            1.2, 0.9, // Class 0
            0.9, 1.1, // Class 0
            5.0, 5.0, // Class 1
            5.1, 4.9, // Class 1
            4.9, 5.1, // Class 1
        ],
    )?;

    let support_labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

    let support_set = SupportSet::new(support_features, support_labels)?;
    println!(
        "   ✓ Support set created with {} examples",
        support_set.size()
    );
    println!("   ✓ Number of classes: {}", support_set.num_classes);

    // Example 2: Prototypical Networks
    println!("\n2. Prototypical Networks:");
    println!("   (Classify by distance to class prototypes)\n");

    let mut proto_net = PrototypicalDistance::euclidean();
    proto_net.compute_prototypes(&support_set);

    // Test queries
    let query1 = Array1::from_vec(vec![1.1, 1.0]); // Should be class 0
    let query2 = Array1::from_vec(vec![4.9, 5.0]); // Should be class 1

    let pred1 = proto_net.predict(&query1.view())?;
    let pred2 = proto_net.predict(&query2.view())?;

    println!("   Query [1.1, 1.0] → Class {} (expected 0)", pred1);
    println!("   Query [4.9, 5.0] → Class {} (expected 1)", pred2);

    // Get probability distributions
    let probs1 = proto_net.predict_proba(&query1.view(), 1.0)?;
    let probs2 = proto_net.predict_proba(&query2.view(), 1.0)?;

    println!(
        "   Probabilities for [1.1, 1.0]: [{:.3}, {:.3}]",
        probs1[0], probs1[1]
    );
    println!(
        "   Probabilities for [4.9, 5.0]: [{:.3}, {:.3}]",
        probs2[0], probs2[1]
    );

    // Example 3: Different Distance Metrics
    println!("\n3. Distance Metrics Comparison:");
    println!("   (Euclidean vs Cosine vs Manhattan)\n");

    let query = Array1::from_vec(vec![1.5, 1.5]);

    for metric in &[
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::Manhattan,
        DistanceMetric::SquaredEuclidean,
    ] {
        let mut proto = PrototypicalDistance::new(*metric);
        proto.compute_prototypes(&support_set);
        let pred = proto.predict(&query.view())?;
        println!("   {:?}: Class {}", metric, pred);
    }

    // Example 4: Matching Networks
    println!("\n4. Matching Networks:");
    println!("   (Attention-based matching to support examples)\n");

    let mut matcher = MatchingNetwork::new(DistanceMetric::Euclidean);
    matcher.set_support(support_set.clone());

    let query = Array1::from_vec(vec![2.0, 2.0]);
    let attention = matcher.compute_attention(&query.view())?;

    println!("   Query [2.0, 2.0]:");
    println!(
        "   Attention weights over {} support examples:",
        attention.len()
    );
    for (i, &weight) in attention.iter().enumerate() {
        println!("     Example {}: {:.3}", i, weight);
    }

    let pred = matcher.predict(&query.view())?;
    let probs = matcher.predict_proba(&query.view())?;
    println!("   Predicted class: {}", pred);
    println!("   Class probabilities: [{:.3}, {:.3}]", probs[0], probs[1]);

    // Example 5: N-way K-shot Episode Sampling
    println!("\n5. Episode Sampling (N-way K-shot):");
    println!("   (Task generation for episodic training)\n");

    let samplers = vec![
        (
            "5-way 1-shot",
            EpisodeSampler::new(5, ShotType::OneShot, 15),
        ),
        (
            "3-way 5-shot",
            EpisodeSampler::new(3, ShotType::FewShot(5), 10),
        ),
        (
            "10-way 3-shot",
            EpisodeSampler::new(10, ShotType::Custom(3), 20),
        ),
    ];

    for (name, sampler) in samplers {
        println!("   {}:", name);
        println!("     Support set size: {} examples", sampler.support_size());
        println!("     Query set size: {} examples", sampler.query_size());
        println!("     Description: {}", sampler.description());
        println!();
    }

    // Example 6: Few-Shot Accuracy Tracking
    println!("6. Few-Shot Accuracy Evaluation:");
    println!("   (Track performance on few-shot tasks)\n");

    let mut accuracy = FewShotAccuracy::new();

    // Create a larger support set for testing
    let test_support_features = Array2::from_shape_vec(
        (12, 2),
        vec![
            // Class 0: around [1, 1]
            1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.2, 1.0, // Class 1: around [5, 5]
            5.0, 5.0, 5.1, 4.9, 4.9, 5.1, 5.0, 5.2, // Class 2: around [1, 5]
            1.0, 5.0, 1.1, 4.9, 0.9, 5.1, 1.0, 5.1,
        ],
    )?;

    let test_support_labels = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);

    let test_support = SupportSet::new(test_support_features, test_support_labels)?;

    let mut proto = PrototypicalDistance::euclidean();
    proto.compute_prototypes(&test_support);

    // Test queries with known labels
    let test_queries = vec![
        (vec![1.0, 1.0], 0),
        (vec![5.0, 5.0], 1),
        (vec![1.0, 5.0], 2),
        (vec![1.2, 1.2], 0),
        (vec![4.8, 4.8], 1),
        (vec![0.9, 4.9], 2),
    ];

    for (features, true_label) in test_queries {
        let query = Array1::from_vec(features.clone());
        let pred = proto.predict(&query.view())?;
        accuracy.update(pred, true_label);
        println!(
            "   Query {:?} → Predicted: {}, Actual: {} {}",
            features,
            pred,
            true_label,
            if pred == true_label { "✓" } else { "✗" }
        );
    }

    let (correct, total) = accuracy.counts();
    println!("\n   Final accuracy: {:.2}%", accuracy.accuracy() * 100.0);
    println!("   ({} correct out of {} queries)", correct, total);

    // Example 7: Practical Use Case
    println!("\n7. Practical Example: Image Classification");
    println!("   (Simulate classifying new object categories)\n");

    // Simulate feature vectors from a pre-trained network
    // In practice, these would come from a CNN feature extractor
    println!("   Scenario: Classify 3 new animal species with 5 examples each");
    println!();

    let species_support = Array2::from_shape_vec(
        (15, 512), // 15 examples × 512-dim features (typical CNN output)
        (0..15 * 512)
            .map(|i| {
                // Simulate features: each species has a different pattern
                let species = i / (512 * 5); // Which species (0, 1, or 2)
                let dim = i % 512; // Which dimension

                // Create patterns that cluster by species
                match species {
                    0 => (dim as f64 / 100.0).sin(), // Species 0: sinusoidal pattern
                    1 => (dim as f64 / 100.0).cos(), // Species 1: cosinusoidal pattern
                    2 => ((dim as f64 / 50.0).sin() + (dim as f64 / 50.0).cos()) / 2.0, // Species 2: mixed
                    _ => 0.0,
                }
            })
            .collect(),
    )?;

    let species_labels = Array1::from_vec(
        (0..15)
            .map(|i| i / 5) // 5 examples per species
            .collect(),
    );

    let species_support_set = SupportSet::new(species_support, species_labels)?;

    println!("   ✓ Created 3-way 5-shot support set");
    println!("   ✓ Feature dimension: 512 (from CNN)");
    println!("   ✓ Ready for classification of new query images");

    // Create prototypes
    let mut species_classifier = PrototypicalDistance::cosine(); // Cosine works well for high-dim
    species_classifier.compute_prototypes(&species_support_set);

    println!("\n   ✓ Computed class prototypes");
    println!("   ✓ Model ready for inference on new examples");

    println!("\n=== Summary ===");
    println!("Few-shot learning enables:");
    println!("  • Learning from minimal labeled examples (1-5 per class)");
    println!("  • Rapid adaptation to new classes");
    println!("  • Efficient use of limited annotation budget");
    println!("  • Transfer learning from rich feature representations");
    println!();
    println!("Use cases:");
    println!("  • New product category classification");
    println!("  • Rare disease diagnosis");
    println!("  • Personalized recommendations");
    println!("  • Robot adaptation to new objects");

    Ok(())
}
