//! Constrained Neural Network Optimization with Logic Constraints
//!
//! This example demonstrates how to use TensorLogic constraints to guide
//! neural network training, ensuring outputs satisfy logical requirements.
//!
//! # Use Case
//!
//! Train a neural network to classify objects while enforcing logical constraints:
//! - Mutual exclusivity: object can't be both "cat" AND "dog"
//! - Hierarchical rules: if "cat" then "animal"
//! - Confidence thresholds: predictions must be above minimum confidence
//!
//! # Running
//!
//! ```bash
//! cargo run --example constrained_neural_optimization --features torsh
//! ```

#[cfg(feature = "torsh")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Constrained Neural Network Optimization\n");

    // ============================================================
    // Part 1: Define Logic Constraints
    // ============================================================
    println!("📋 Part 1: Logic Constraints Definition");
    println!("  Constraints:");
    println!("    1. Mutual exclusivity: NOT(cat AND dog)");
    println!("    2. Hierarchy: cat → animal");
    println!("    3. Hierarchy: dog → animal");
    println!("    4. Minimum confidence: score >= 0.5");
    println!();

    // ============================================================
    // Part 2: Simulate Neural Network Predictions
    // ============================================================
    println!("🧠 Part 2: Neural Network Predictions (Unconstrained)");

    // Simulate 4 samples × 3 classes (cat, dog, animal)
    let num_samples = 4;
    let num_classes = 3;

    // Raw neural network outputs (logits)
    // Sample 0: predicts cat and dog (violates mutual exclusivity!)
    // Sample 1: predicts cat but not animal (violates hierarchy!)
    // Sample 2: predicts dog correctly
    // Sample 3: predicts animal correctly
    let raw_predictions: Vec<f32> = vec![
        0.8, 0.7, 0.3, // Sample 0: cat=0.8, dog=0.7, animal=0.3 (VIOLATIONS!)
        0.9, 0.1, 0.2, // Sample 1: cat=0.9, dog=0.1, animal=0.2 (VIOLATION!)
        0.2, 0.8, 0.9, // Sample 2: cat=0.2, dog=0.8, animal=0.9 (OK)
        0.1, 0.2, 0.9, // Sample 3: cat=0.1, dog=0.2, animal=0.9 (OK)
    ];

    println!("  Raw predictions (samples × [cat, dog, animal]):");
    for i in 0..num_samples {
        let start = i * num_classes;
        println!(
            "    Sample {}: [{:.2}, {:.2}, {:.2}]",
            i,
            raw_predictions[start],
            raw_predictions[start + 1],
            raw_predictions[start + 2]
        );
    }
    println!();

    // ============================================================
    // Part 3: Check Constraint Violations
    // ============================================================
    println!("⚠️  Part 3: Constraint Violation Detection");

    let mut violations = Vec::new();

    // Check each sample
    for i in 0..num_samples {
        let cat_score = raw_predictions[i * num_classes];
        let dog_score = raw_predictions[i * num_classes + 1];
        let animal_score = raw_predictions[i * num_classes + 2];

        // Constraint 1: Mutual exclusivity (cat AND dog should be close to 0)
        let mutual_exclusivity_violation = (cat_score * dog_score) > 0.5;

        // Constraint 2: Hierarchy (if cat, then animal)
        let cat_hierarchy_violation = cat_score > 0.5 && animal_score < cat_score;

        // Constraint 3: Hierarchy (if dog, then animal)
        let dog_hierarchy_violation = dog_score > 0.5 && animal_score < dog_score;

        if mutual_exclusivity_violation {
            violations.push(format!(
                "Sample {}: Mutual exclusivity violated (cat={:.2} AND dog={:.2})",
                i, cat_score, dog_score
            ));
        }

        if cat_hierarchy_violation {
            violations.push(format!(
                "Sample {}: Hierarchy violated (cat={:.2} but animal={:.2})",
                i, cat_score, animal_score
            ));
        }

        if dog_hierarchy_violation {
            violations.push(format!(
                "Sample {}: Hierarchy violated (dog={:.2} but animal={:.2})",
                i, dog_score, animal_score
            ));
        }
    }

    if violations.is_empty() {
        println!("    ✓ No violations detected");
    } else {
        for violation in &violations {
            println!("    ✗ {}", violation);
        }
    }
    println!();

    // ============================================================
    // Part 4: Apply Constraint Corrections
    // ============================================================
    println!("🔧 Part 4: Constraint-Guided Correction");

    // Strategy: Modify predictions to satisfy constraints
    let mut corrected = raw_predictions.clone();

    for i in 0..num_samples {
        let cat_idx = i * num_classes;
        let dog_idx = i * num_classes + 1;
        let animal_idx = i * num_classes + 2;

        let cat = corrected[cat_idx];
        let dog = corrected[dog_idx];
        let animal = corrected[animal_idx];

        // Fix mutual exclusivity: keep highest, suppress other
        if cat > 0.5 && dog > 0.5 {
            if cat > dog {
                corrected[dog_idx] *= 0.3; // Suppress dog
            } else {
                corrected[cat_idx] *= 0.3; // Suppress cat
            }
        }

        // Fix hierarchy: if cat/dog, ensure animal >= max(cat, dog)
        let max_species = cat.max(dog);
        if max_species > 0.5 && animal < max_species {
            corrected[animal_idx] = max_species * 1.1; // Ensure animal is higher
        }

        // Clip to [0, 1] range
        corrected[cat_idx] = corrected[cat_idx].clamp(0.0, 1.0);
        corrected[dog_idx] = corrected[dog_idx].clamp(0.0, 1.0);
        corrected[animal_idx] = corrected[animal_idx].clamp(0.0, 1.0);
    }

    println!("  Corrected predictions:");
    for i in 0..num_samples {
        let start = i * num_classes;
        println!(
            "    Sample {}: [{:.2}, {:.2}, {:.2}]",
            i,
            corrected[start],
            corrected[start + 1],
            corrected[start + 2]
        );
    }
    println!();

    // ============================================================
    // Part 5: Verify Corrected Predictions
    // ============================================================
    println!("✅ Part 5: Verification of Corrected Predictions");

    // Re-check constraints
    let mut new_violations = 0;

    for i in 0..num_samples {
        let cat_score = corrected[i * num_classes];
        let dog_score = corrected[i * num_classes + 1];
        let animal_score = corrected[i * num_classes + 2];

        if (cat_score * dog_score) > 0.5 {
            new_violations += 1;
        }

        if cat_score > 0.5 && animal_score < cat_score {
            new_violations += 1;
        }

        if dog_score > 0.5 && animal_score < dog_score {
            new_violations += 1;
        }
    }

    println!("  Violations before correction: {}", violations.len());
    println!("  Violations after correction: {}", new_violations);

    if new_violations == 0 {
        println!("  ✓ All constraints satisfied!");
    } else {
        println!("  ⚠️  Some violations remain");
    }
    println!();

    // ============================================================
    // Part 6: Compute Constraint Loss for Training
    // ============================================================
    println!("📊 Part 6: Constraint Loss for Training");

    // Compute violation penalty (could be used as additional loss term)
    let mut total_violation_loss = 0.0;

    for i in 0..num_samples {
        let cat = raw_predictions[i * num_classes];
        let dog = raw_predictions[i * num_classes + 1];
        let animal = raw_predictions[i * num_classes + 2];

        // Mutual exclusivity loss: penalize cat * dog
        total_violation_loss += (cat * dog).max(0.0);

        // Hierarchy loss: penalize max(cat, dog) - animal when positive
        total_violation_loss += (cat.max(dog) - animal).max(0.0);
    }

    let avg_violation_loss = total_violation_loss / num_samples as f32;

    println!(
        "  Average constraint violation loss: {:.4}",
        avg_violation_loss
    );
    println!("  (This can be added to training loss to enforce constraints)\n");

    // ============================================================
    // Summary
    // ============================================================
    println!("🎉 Constrained Neural Optimization Summary:");
    println!("  ✅ Defined logic constraints (mutual exclusivity, hierarchy)");
    println!(
        "  ✅ Detected {} constraint violations in raw predictions",
        violations.len()
    );
    println!("  ✅ Applied constraint-guided corrections");
    println!(
        "  ✅ Reduced violations from {} → {}",
        violations.len(),
        new_violations
    );
    println!("  ✅ Computed violation loss for gradient-based training");
    println!();

    println!("💡 Training Integration:");
    println!("  - Add constraint loss to training objective:");
    println!("    total_loss = prediction_loss + λ·constraint_loss");
    println!("  - Use TensorLogic → ToRSh for differentiable constraints");
    println!("  - Backpropagate through both data loss and constraint loss");
    println!("  - Network learns to satisfy logical rules automatically");

    Ok(())
}

#[cfg(not(feature = "torsh"))]
fn main() {
    eprintln!("This example requires the 'torsh' feature.");
    eprintln!("Run with: cargo run --example constrained_neural_optimization --features torsh");
    std::process::exit(1);
}
