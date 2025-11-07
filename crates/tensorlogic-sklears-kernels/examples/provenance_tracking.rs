//! # Provenance Tracking Example
//!
//! This example demonstrates how to use provenance tracking to:
//! - Monitor kernel computations for debugging
//! - Audit ML pipeline operations
//! - Analyze performance patterns
//! - Export provenance data for reproducibility
//!
//! Run with: cargo run --example provenance_tracking

use tensorlogic_sklears_kernels::{
    ComputationResult, Kernel, LinearKernel, ProvenanceConfig, ProvenanceKernel, ProvenanceTracker,
    RbfKernel, RbfKernelConfig,
};

fn main() {
    println!("=== Provenance Tracking Example ===\n");

    // Example 1: Basic provenance tracking
    println!("1. Basic Provenance Tracking");
    println!("{}", "-".repeat(50));
    basic_tracking();
    println!();

    // Example 2: Tracking with configuration
    println!("2. Configurable Tracking");
    println!("{}", "-".repeat(50));
    configured_tracking();
    println!();

    // Example 3: Filtering and querying
    println!("3. Filtering and Querying");
    println!("{}", "-".repeat(50));
    filtering_example();
    println!();

    // Example 4: Performance analysis
    println!("4. Performance Analysis");
    println!("{}", "-".repeat(50));
    performance_analysis();
    println!();

    // Example 5: JSON export/import
    println!("5. JSON Export/Import");
    println!("{}", "-".repeat(50));
    json_export_example();
    println!();

    // Example 6: Tagged Experiments
    println!("6. Tagged Experiments");
    println!("{}", "-".repeat(50));
    tagged_experiments();
    println!();
}

/// Example 1: Basic provenance tracking
fn basic_tracking() {
    // Create tracker and kernel
    let tracker = ProvenanceTracker::new();
    let base_kernel = Box::new(LinearKernel::new());
    let kernel = ProvenanceKernel::new(base_kernel, tracker.clone());

    // Perform some computations
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    println!("Computing kernel(x, y)...");
    let result = kernel.compute(&x, &y).unwrap();
    println!("Result: {:.4}", result);

    // Check tracked records
    println!("\nTracked computations: {}", tracker.count());

    let records = tracker.get_all_records();
    for record in &records {
        println!("  - ID: {}", record.id);
        println!("    Kernel: {}", record.kernel_name);
        println!("    Input dim: {}", record.input_dimension);
        println!("    Time: {:?}", record.computation_time);
    }
}

/// Example 2: Configurable tracking
fn configured_tracking() {
    // Configure tracking with limits
    let config = ProvenanceConfig::new()
        .with_max_records(100)
        .with_sample_rate(1.0)
        .unwrap()
        .with_timing(true);

    let tracker = ProvenanceTracker::with_config(config);
    let base_kernel = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap());
    let kernel = ProvenanceKernel::new(base_kernel, tracker.clone());

    // Generate some data
    let data = generate_random_data(20, 5);

    println!("Computing kernel matrix for {} samples...", data.len());
    let matrix = kernel.compute_matrix(&data).unwrap();

    println!("Matrix dimension: {}x{}", matrix.len(), matrix[0].len());
    println!("Tracked records: {}", tracker.count());

    // Show matrix computation details
    let records = tracker.get_all_records();
    if let Some(record) = records.first() {
        if let ComputationResult::Matrix {
            dimension,
            trace,
            frobenius_norm,
        } = &record.result
        {
            println!("\nMatrix statistics:");
            println!("  Dimension: {}", dimension);
            println!("  Trace: {:.4}", trace);
            println!("  Frobenius norm: {:.4}", frobenius_norm);
        }
    }
}

/// Example 3: Filtering and querying
fn filtering_example() {
    let tracker = ProvenanceTracker::new();

    // Track multiple kernel types
    let linear_kernel = Box::new(LinearKernel::new());
    let rbf_kernel = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap());

    let linear_prov = ProvenanceKernel::new(linear_kernel, tracker.clone());
    let rbf_prov = ProvenanceKernel::new(rbf_kernel, tracker.clone());

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    // Perform computations with both kernels
    println!("Computing with Linear kernel...");
    linear_prov.compute(&x, &y).unwrap();

    println!("Computing with RBF kernel...");
    rbf_prov.compute(&x, &y).unwrap();

    println!("\nTotal tracked: {}", tracker.count());

    // Filter by kernel type
    let linear_records = tracker.get_records_by_kernel("Linear");
    let rbf_records = tracker.get_records_by_kernel("RBF");

    println!("Linear kernel computations: {}", linear_records.len());
    println!("RBF kernel computations: {}", rbf_records.len());
}

/// Example 4: Performance analysis
fn performance_analysis() {
    let tracker = ProvenanceTracker::new();
    let base_kernel = Box::new(LinearKernel::new());
    let kernel = ProvenanceKernel::new(base_kernel, tracker.clone());

    // Perform multiple computations
    println!("Performing 10 pairwise computations...");
    for i in 0..10 {
        let x = vec![i as f64, (i + 1) as f64, (i + 2) as f64];
        let y = vec![(i + 3) as f64, (i + 4) as f64, (i + 5) as f64];
        kernel.compute(&x, &y).unwrap();
    }

    // Get statistics
    let stats = tracker.statistics();

    println!("\nPerformance Statistics:");
    println!("  Total computations: {}", stats.total_computations);
    println!("  Successful: {}", stats.successful_computations);
    println!("  Failed: {}", stats.failed_computations);

    if let Some(avg_time) = stats.average_computation_time {
        println!("  Average time: {:?}", avg_time);
    }

    println!("\nKernel usage breakdown:");
    for (kernel_name, count) in &stats.kernel_counts {
        println!("  {}: {} computations", kernel_name, count);
    }
}

/// Example 5: JSON export/import
fn json_export_example() {
    let tracker = ProvenanceTracker::new();
    let base_kernel = Box::new(LinearKernel::new());
    let kernel = ProvenanceKernel::new(base_kernel, tracker.clone());

    // Perform some computations
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];
    kernel.compute(&x, &y).unwrap();

    // Export to JSON
    println!("Exporting provenance to JSON...");
    let json = tracker.to_json().unwrap();
    println!("JSON length: {} bytes", json.len());

    // Show first 200 chars
    if json.len() > 200 {
        println!("Preview:\n{}\n...", &json[..200]);
    } else {
        println!("Content:\n{}", json);
    }

    // Import into new tracker
    let tracker2 = ProvenanceTracker::new();
    tracker2.from_json(&json).unwrap();

    println!("\nImported {} records", tracker2.count());
}

/// Example 6: Tagged experiments
fn tagged_experiments() {
    let tracker = ProvenanceTracker::new();

    // Experiment 1: Baseline
    println!("Running baseline experiment...");
    let mut baseline = ProvenanceKernel::new(Box::new(LinearKernel::new()), tracker.clone());
    baseline.add_tag("experiment:baseline".to_string());
    baseline.add_tag("phase:1".to_string());

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];
    baseline.compute(&x, &y).unwrap();

    // Experiment 2: RBF variant
    println!("Running RBF variant experiment...");
    let mut rbf_variant = ProvenanceKernel::new(
        Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()),
        tracker.clone(),
    );
    rbf_variant.add_tag("experiment:rbf_variant".to_string());
    rbf_variant.add_tag("phase:2".to_string());

    rbf_variant.compute(&x, &y).unwrap();

    // Query by tags
    println!("\nTotal records: {}", tracker.count());

    let baseline_records = tracker.get_records_by_tag("experiment:baseline");
    println!("Baseline records: {}", baseline_records.len());

    let rbf_records = tracker.get_records_by_tag("experiment:rbf_variant");
    println!("RBF variant records: {}", rbf_records.len());

    let phase1_records = tracker.get_records_by_tag("phase:1");
    println!("Phase 1 records: {}", phase1_records.len());

    // Show tags for each record
    println!("\nRecord tags:");
    for record in tracker.get_all_records() {
        println!("  Record {}: {:?}", record.id, record.tags);
    }
}

/// Helper function to generate random data
fn generate_random_data(n_samples: usize, n_features: usize) -> Vec<Vec<f64>> {
    (0..n_samples)
        .map(|i| {
            (0..n_features)
                .map(|j| ((i * n_features + j) as f64) * 0.1)
                .collect()
        })
        .collect()
}
