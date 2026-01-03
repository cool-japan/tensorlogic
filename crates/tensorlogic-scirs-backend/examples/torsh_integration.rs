//! ToRSh Integration Example
//!
//! This example demonstrates seamless interoperability between TensorLogic
//! and ToRSh (pure Rust PyTorch alternative) for neurosymbolic AI applications.
//!
//! # Use Cases
//!
//! 1. **Logic → Neural**: Use logic execution results as neural network inputs
//! 2. **Neural → Logic**: Convert neural network outputs into logic constraints
//! 3. **Hybrid Training**: Combine symbolic reasoning with gradient descent
//!
//! # Running
//!
//! ```bash
//! cargo run --example torsh_integration --features torsh
//! ```

#[cfg(feature = "torsh")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_core::ndarray::ArrayD;
    use tensorlogic_scirs_backend::torsh_interop::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::Tensor;

    println!("🧠 TensorLogic ↔ ToRSh Interoperability Demo\n");

    // ============================================================
    // Scenario 1: Logic Results → ToRSh Neural Network
    // ============================================================
    println!("📊 Scenario 1: TensorLogic → ToRSh");
    println!("  Converting logic execution results to ToRSh tensors\n");

    // Simulate logic execution results (e.g., predicate satisfaction scores)
    let logic_results = vec![0.9, 0.8, 0.95, 0.85, 0.7, 0.92];
    let tl_tensor = ArrayD::from_shape_vec(vec![2, 3], logic_results.clone())?;

    println!("  TensorLogic tensor shape: {:?}", tl_tensor.shape());
    println!("  TensorLogic data: {:?}\n", &logic_results);

    // Convert to ToRSh tensor (f32 for neural networks)
    let torsh_tensor = tl_to_torsh_f32(&tl_tensor, DeviceType::Cpu)?;

    println!("  ToRSh tensor shape: {:?}", torsh_tensor.shape().dims());
    println!("  ToRSh data: {:?}\n", torsh_tensor.to_vec()?);

    // Simulate neural network operation (e.g., activation function)
    let activated = torsh_tensor.sigmoid()?;
    println!("  After sigmoid activation: {:?}\n", activated.to_vec()?);

    // ============================================================
    // Scenario 2: ToRSh Neural Output → Logic Constraints
    // ============================================================
    println!("📊 Scenario 2: ToRSh → TensorLogic");
    println!("  Converting neural network outputs to logic constraints\n");

    // Simulate neural network output (classification probabilities)
    let nn_output = vec![0.2f64, 0.7, 0.1, 0.9, 0.3, 0.8];
    let torsh_output = Tensor::from_data(nn_output.clone(), vec![2, 3], DeviceType::Cpu)?;

    println!("  ToRSh output shape: {:?}", torsh_output.shape().dims());
    println!("  ToRSh data: {:?}\n", nn_output);

    // Convert to TensorLogic for constraint checking
    let tl_constraints = torsh_to_tl(&torsh_output)?;

    println!("  TensorLogic tensor shape: {:?}", tl_constraints.shape());
    println!(
        "  TensorLogic data: {:?}\n",
        tl_constraints.iter().copied().collect::<Vec<_>>()
    );

    // Check constraint: all values should be > 0.5 (thresholding)
    let satisfies_constraint = tl_constraints.iter().all(|&x| x > 0.5);
    println!("  Constraint check (all > 0.5): {}\n", satisfies_constraint);

    // ============================================================
    // Scenario 3: Roundtrip Conversion (Lossless for f64)
    // ============================================================
    println!("📊 Scenario 3: Roundtrip Conversion Test");
    println!("  TensorLogic → ToRSh → TensorLogic (f64 precision)\n");

    let original_data = vec![1.5, 2.5, 3.5, 4.5];
    let original = ArrayD::from_shape_vec(vec![2, 2], original_data.clone())?;

    println!("  Original TensorLogic: {:?}", original_data);

    // Roundtrip: TL → ToRSh → TL
    let torsh_intermediate = tl_to_torsh(&original, DeviceType::Cpu)?;
    let roundtrip = torsh_to_tl(&torsh_intermediate)?;

    println!(
        "  After roundtrip: {:?}",
        roundtrip.iter().copied().collect::<Vec<_>>()
    );

    // Verify lossless conversion
    let original_vec: Vec<f64> = original.iter().copied().collect();
    let roundtrip_vec: Vec<f64> = roundtrip.iter().copied().collect();
    assert_eq!(original_vec, roundtrip_vec, "Roundtrip should be lossless!");

    println!("  ✅ Roundtrip conversion is lossless!\n");

    // ============================================================
    // Scenario 4: Batch Processing with Type Conversion
    // ============================================================
    println!("📊 Scenario 4: Batch Processing");
    println!("  Processing multiple logic results through neural network\n");

    let batch_size = 3;
    let feature_dim = 4;

    // Create batch of logic results
    let batch_data: Vec<f64> = (0..batch_size * feature_dim)
        .map(|i| (i as f64 * 0.1) % 1.0)
        .collect();
    let tl_batch = ArrayD::from_shape_vec(vec![batch_size, feature_dim], batch_data.clone())?;

    println!("  Batch shape: {:?}", tl_batch.shape());
    println!("  Batch data: {:?}\n", batch_data);

    // Convert to ToRSh f32 for efficient neural processing
    let torsh_batch = tl_to_torsh_f32(&tl_batch, DeviceType::Cpu)?;

    // Simulate neural network layer (element-wise operation)
    let processed = torsh_batch.mul_scalar(2.0)?.add_scalar(0.5)?;

    println!("  After neural processing: {:?}\n", processed.to_vec()?);

    // Convert back to TensorLogic for logic integration
    let tl_processed = torsh_f32_to_tl(&processed)?;

    println!(
        "  Back to TensorLogic: {:?}\n",
        tl_processed.iter().copied().collect::<Vec<_>>()
    );

    // ============================================================
    // Summary
    // ============================================================
    println!("🎉 ToRSh Integration Summary:");
    println!("  ✅ TensorLogic → ToRSh (f32/f64)");
    println!("  ✅ ToRSh → TensorLogic (f32/f64)");
    println!("  ✅ Roundtrip conversion (lossless for f64)");
    println!("  ✅ Batch processing with type conversion");
    println!("  ✅ Pure Rust neurosymbolic AI integration\n");

    println!("💡 Use Cases:");
    println!("  - Neurosymbolic AI (logic + neural networks)");
    println!("  - Differentiable logic programming");
    println!("  - Hybrid symbolic-connectionist systems");
    println!("  - Explainable AI with logic constraints");

    Ok(())
}

#[cfg(not(feature = "torsh"))]
fn main() {
    eprintln!("This example requires the 'torsh' feature.");
    eprintln!("Run with: cargo run --example torsh_integration --features torsh");
    std::process::exit(1);
}
