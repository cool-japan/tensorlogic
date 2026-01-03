//! GPU Readiness Assessment Example
//!
//! This example demonstrates the GPU readiness framework for TensorLogic.
//! It shows:
//!
//! 1. Assessing GPU availability and capabilities
//! 2. Getting detailed GPU information (memory, compute capability, bandwidth)
//! 3. Receiving execution recommendations
//! 4. Workload profiling and batch size recommendations
//! 5. Theoretical speedup estimation
//!
//! This framework helps you prepare for GPU execution even before full
//! GPU support is implemented in the backend.

use tensorlogic_scirs_backend::{
    assess_gpu_readiness, generate_recommendations, recommend_batch_size, WorkloadProfile,
};

fn main() {
    println!("=== TensorLogic GPU Readiness Assessment ===\n");

    // Assess GPU readiness
    let report = assess_gpu_readiness();

    println!("1. GPU Availability");
    println!("   ----------------");
    println!("   GPU Available: {}", report.gpu_available);
    println!("   GPU Count: {}", report.gpu_count);
    println!("   Recommended Device: {}", report.recommended_device);

    if let Some(speedup) = report.estimated_speedup {
        println!("   Estimated Speedup: {:.1}x over CPU\n", speedup);
    } else {
        println!("   Estimated Speedup: N/A (CPU only)\n");
    }

    // Show recommendation reasons
    println!("2. Recommendation Reasons");
    println!("   ----------------------");
    for (i, reason) in report.recommendation_reasons.iter().enumerate() {
        println!("   {}. {}", i + 1, reason);
    }
    println!();

    // Detailed GPU information
    if !report.gpus.is_empty() {
        println!("3. Detailed GPU Capabilities");
        println!("   -------------------------");

        for (idx, gpu) in report.gpus.iter().enumerate() {
            println!("\n   GPU {} ({}): {}", idx, gpu.device, gpu.name);
            println!("   ─────────────────────────");
            println!("     Memory: {} GB", gpu.memory_mb / 1024);
            println!(
                "     Memory Bandwidth: {:.0} GB/s",
                gpu.memory_bandwidth_gbs
            );

            if let Some((major, minor)) = gpu.compute_capability {
                println!("     Compute Capability: {}.{}", major, minor);
            }

            if let Some(cores) = gpu.cuda_cores {
                println!("     CUDA Cores: ~{}", cores);
            }

            println!(
                "     Tensor Cores: {}",
                if gpu.has_tensor_cores { "Yes" } else { "No" }
            );
            println!(
                "     FP16 Support: {}",
                if gpu.supports_fp16 { "Yes" } else { "No" }
            );
            println!(
                "     INT8 Support: {}",
                if gpu.supports_int8 { "Yes" } else { "No" }
            );
            println!("     Capability Score: {:.1}", gpu.capability_score());
            println!(
                "     Recommended: {}",
                if gpu.recommended { "★ YES ★" } else { "No" }
            );
        }
        println!();
    } else {
        println!("3. No GPUs Detected");
        println!("   ----------------");
        println!("   Running in CPU-only mode\n");
    }

    // General recommendations
    println!("4. General Recommendations");
    println!("   -----------------------");
    let recommendations = generate_recommendations(&report, None);
    for (i, rec) in recommendations.iter().enumerate() {
        println!("   {}. {}", i + 1, rec);
    }
    println!();

    // Workload-specific recommendations
    println!("5. Workload-Specific Analysis");
    println!("   --------------------------");

    // Example workload profiles
    let small_workload = WorkloadProfile {
        operation_count: 100,
        avg_tensor_size: 10000,
        peak_memory_mb: 64,
        compute_intensity: 15.0,
    };

    let medium_workload = WorkloadProfile {
        operation_count: 1000,
        avg_tensor_size: 100000,
        peak_memory_mb: 512,
        compute_intensity: 50.0,
    };

    let large_workload = WorkloadProfile {
        operation_count: 10000,
        avg_tensor_size: 1000000,
        peak_memory_mb: 4096,
        compute_intensity: 100.0,
    };

    println!("   Small Workload (64 MB):");
    println!("     Operations: {}", small_workload.operation_count);
    println!(
        "     Compute Intensity: {:.1} FLOPs/byte",
        small_workload.compute_intensity
    );

    if !report.gpus.is_empty() {
        let batch_size = recommend_batch_size(&report.gpus[0], &small_workload);
        println!("     Recommended Batch Size: {}", batch_size);
    }

    let recs = generate_recommendations(&report, Some(&small_workload));
    for rec in recs {
        println!("     → {}", rec);
    }

    println!("\n   Medium Workload (512 MB):");
    println!("     Operations: {}", medium_workload.operation_count);
    println!(
        "     Compute Intensity: {:.1} FLOPs/byte",
        medium_workload.compute_intensity
    );

    if !report.gpus.is_empty() {
        let batch_size = recommend_batch_size(&report.gpus[0], &medium_workload);
        println!("     Recommended Batch Size: {}", batch_size);
    }

    let recs = generate_recommendations(&report, Some(&medium_workload));
    for rec in recs {
        println!("     → {}", rec);
    }

    println!("\n   Large Workload (4 GB):");
    println!("     Operations: {}", large_workload.operation_count);
    println!(
        "     Compute Intensity: {:.1} FLOPs/byte",
        large_workload.compute_intensity
    );

    if !report.gpus.is_empty() {
        let batch_size = recommend_batch_size(&report.gpus[0], &large_workload);
        println!("     Recommended Batch Size: {}", batch_size);
    }

    let recs = generate_recommendations(&report, Some(&large_workload));
    for rec in recs {
        println!("     → {}", rec);
    }

    println!("\n6. Future GPU Support");
    println!("   ------------------");
    println!("   This framework is ready for future GPU execution:");
    println!("   • Device detection and capability assessment");
    println!("   • Workload profiling and optimization recommendations");
    println!("   • Batch size tuning for GPU memory constraints");
    println!("   • Performance estimation and planning");
    println!();
    println!("   When scirs2-core adds GPU support, these tools will help");
    println!("   you optimize your TensorLogic workflows for maximum performance!");

    println!("\n=== End of GPU Readiness Assessment ===");
}
