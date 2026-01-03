//! Knowledge Graph Reasoning with Neurosymbolic AI
//!
//! This example demonstrates how to combine:
//! - Symbolic logic rules for knowledge graph reasoning
//! - Neural embeddings for similarity-based inference
//! - ToRSh for differentiable tensor operations
//!
//! # Use Case
//!
//! Given a knowledge graph with entities and relations, we want to:
//! 1. Apply logical rules (e.g., transitivity: friendOf(A,B) ∧ friendOf(B,C) → friendOf(A,C))
//! 2. Use neural embeddings to score potential facts
//! 3. Combine symbolic and neural reasoning for knowledge completion
//!
//! # Running
//!
//! ```bash
//! cargo run --example knowledge_graph_reasoning --features torsh
//! ```

#[cfg(feature = "torsh")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_core::ndarray::ArrayD;
    use tensorlogic_scirs_backend::torsh_interop::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::Tensor;

    println!("🔗 Knowledge Graph Reasoning with Neurosymbolic AI\n");

    // ============================================================
    // Part 1: Symbolic Logic Rules
    // ============================================================
    println!("📚 Part 1: Symbolic Logic Rules for Knowledge Graph");
    println!("  Rules:");
    println!("    1. Transitivity: friendOf(A,B) ∧ friendOf(B,C) → friendOf(A,C)");
    println!("    2. Symmetry: friendOf(A,B) → friendOf(B,A)");
    println!("    3. Type constraint: person(X) ∧ person(Y) → friendOf(X,Y)");
    println!();

    // Knowledge graph: (Alice, friendOf, Bob), (Bob, friendOf, Charlie)
    // Adjacency matrix representation: 1 if relation exists, 0 otherwise
    // Entities: [Alice=0, Bob=1, Charlie=2]
    let num_entities = 3;

    // friendOf relation (adjacency matrix)
    let friend_of_data = vec![
        0.0, 1.0, 0.0, // Alice -> Bob
        1.0, 0.0, 1.0, // Bob -> Alice (symmetric), Bob -> Charlie
        0.0, 1.0, 0.0, // Charlie -> Bob (symmetric)
    ];

    let friend_of_matrix =
        ArrayD::from_shape_vec(vec![num_entities, num_entities], friend_of_data)?;

    println!("  Initial friendOf matrix:");
    println!(
        "  {:?}\n",
        friend_of_matrix.iter().copied().collect::<Vec<_>>()
    );

    // Apply transitivity rule: A·B where A and B are adjacency matrices
    // This computes 2-hop friendships
    let friend_of_2hop = {
        let a: Vec<f64> = friend_of_matrix.iter().copied().collect();
        let b: Vec<f64> = friend_of_matrix.iter().copied().collect();

        // Matrix multiplication for transitivity
        let mut result = vec![0.0; num_entities * num_entities];
        for i in 0..num_entities {
            for j in 0..num_entities {
                for k in 0..num_entities {
                    result[i * num_entities + j] +=
                        a[i * num_entities + k] * b[k * num_entities + j];
                }
            }
        }

        // Threshold: if there's a 2-hop path, create friendship
        for val in &mut result {
            *val = if *val > 0.0 { 1.0 } else { 0.0 };
        }

        ArrayD::from_shape_vec(vec![num_entities, num_entities], result)?
    };

    println!("  2-hop friendships (transitivity):");
    println!(
        "  {:?}\n",
        friend_of_2hop.iter().copied().collect::<Vec<_>>()
    );

    // Combine direct and transitive friendships
    let combined_friends = {
        let direct = friend_of_matrix.iter().copied().collect::<Vec<_>>();
        let indirect = friend_of_2hop.iter().copied().collect::<Vec<_>>();

        let mut result = vec![0.0; num_entities * num_entities];
        for i in 0..result.len() {
            result[i] = if direct[i] > 0.0 || indirect[i] > 0.0 {
                1.0
            } else {
                0.0
            };
        }

        ArrayD::from_shape_vec(vec![num_entities, num_entities], result)?
    };

    println!("  Combined friendships (direct + transitive):");
    println!(
        "  {:?}\n",
        combined_friends.iter().copied().collect::<Vec<_>>()
    );

    // ============================================================
    // Part 2: Neural Embeddings for Entity Similarity
    // ============================================================
    println!("🧠 Part 2: Neural Embeddings for Similarity-based Reasoning");
    println!("  Converting logic results to ToRSh for neural processing\n");

    // Convert to ToRSh tensor
    let torsh_friends = tl_to_torsh_f32(&combined_friends, DeviceType::Cpu)?;

    println!(
        "  ToRSh friendship tensor: {:?}",
        torsh_friends.shape().dims()
    );

    // Simulate entity embeddings (3 entities × 4 dimensions)
    // In practice, these would be learned by a neural network
    let entity_embeddings = vec![
        // Alice embedding
        0.8, 0.2, 0.1, 0.5, // Bob embedding
        0.7, 0.3, 0.2, 0.4, // Charlie embedding
        0.6, 0.4, 0.3, 0.3,
    ];

    let embeddings_tensor = Tensor::from_data(entity_embeddings, vec![3, 4], DeviceType::Cpu)?;

    println!(
        "  Entity embeddings shape: {:?}",
        embeddings_tensor.shape().dims()
    );

    // Compute embedding similarity (dot product)
    // This gives us neural scores for potential friendships
    let embedding_sim = {
        let emb_t = embeddings_tensor.transpose(0, 1)?; // [3, 4] → [4, 3]
        let sim = embeddings_tensor.matmul(&emb_t)?; // [3, 4] × [4, 3] = [3, 3]
        sim.sigmoid()? // Apply sigmoid for probability-like scores
    };

    println!("  Embedding similarity scores:");
    println!("  {:?}\n", embedding_sim.to_vec()?);

    // ============================================================
    // Part 3: Hybrid Reasoning (Logic + Neural)
    // ============================================================
    println!("⚡ Part 3: Hybrid Reasoning (Combining Logic and Neural Scores)");
    println!("  Formula: final_score = α·logic_score + (1-α)·neural_score\n");

    let alpha = 0.7; // Weight for logic vs neural (0.7 = 70% logic, 30% neural)

    // Logic scores are already f32 from tl_to_torsh_f32
    let logic_scores = &torsh_friends;

    // Combine logic and neural scores
    let hybrid_scores = logic_scores
        .mul_scalar(alpha)?
        .add(&embedding_sim.mul_scalar(1.0 - alpha)?)?;

    println!("  Hybrid friendship scores (α={}):", alpha);
    println!("  {:?}\n", hybrid_scores.to_vec()?);

    // Threshold to get binary predictions
    let threshold = 0.5;
    let predictions = hybrid_scores.to_vec()?;

    println!("  Final predictions (threshold={}):", threshold);
    for i in 0..num_entities {
        for j in 0..num_entities {
            let score = predictions[i * num_entities + j];
            let is_friend = score > threshold;
            if is_friend && i != j {
                let names = ["Alice", "Bob", "Charlie"];
                println!(
                    "    ✓ {} and {} are friends (score: {:.3})",
                    names[i], names[j], score
                );
            }
        }
    }
    println!();

    // ============================================================
    // Part 4: Convert Back to Logic for Constraint Checking
    // ============================================================
    println!("✅ Part 4: Constraint Validation (Neural → Logic)");
    println!("  Converting neural predictions back to logic for verification\n");

    // Convert predictions back to TensorLogic
    let predictions_tl = torsh_f32_to_tl(&hybrid_scores)?;

    println!("  TensorLogic predictions: {:?}", predictions_tl.shape());

    // Verify constraints: friendOf should be symmetric
    let is_symmetric = {
        let pred_vec: Vec<f64> = predictions_tl.iter().copied().collect();
        let mut symmetric = true;
        for i in 0..num_entities {
            for j in 0..num_entities {
                let forward = pred_vec[i * num_entities + j];
                let backward = pred_vec[j * num_entities + i];
                if (forward - backward).abs() > 0.1 {
                    symmetric = false;
                    println!(
                        "    ⚠️  Asymmetry detected: ({}, {}) = {:.2}, ({}, {}) = {:.2}",
                        i, j, forward, j, i, backward
                    );
                }
            }
        }
        symmetric
    };

    if is_symmetric {
        println!("    ✓ Symmetry constraint satisfied!");
    } else {
        println!("    ✗ Symmetry constraint violated (could apply correction)");
    }
    println!();

    // ============================================================
    // Summary
    // ============================================================
    println!("🎉 Neurosymbolic Knowledge Graph Reasoning Summary:");
    println!("  ✅ Applied symbolic logic rules (transitivity, symmetry)");
    println!("  ✅ Computed neural embedding similarities");
    println!(
        "  ✅ Combined logic and neural scores (α={}, 1-α={})",
        alpha,
        1.0 - alpha
    );
    println!("  ✅ Made hybrid predictions with constraint checking");
    println!();

    println!("💡 Key Benefits:");
    println!("  - Logic rules provide interpretability and hard constraints");
    println!("  - Neural embeddings capture soft similarities and learn patterns");
    println!("  - Hybrid approach combines strengths of both paradigms");
    println!("  - Bidirectional conversion enables constraint verification");

    Ok(())
}

#[cfg(not(feature = "torsh"))]
fn main() {
    eprintln!("This example requires the 'torsh' feature.");
    eprintln!("Run with: cargo run --example knowledge_graph_reasoning --features torsh");
    std::process::exit(1);
}
