//! GPU Readiness Framework
//!
//! This module provides utilities for assessing GPU readiness and
//! planning for GPU execution. It helps determine optimal execution
//! strategies based on available hardware and workload characteristics.

use crate::cuda_detect::{detect_cuda_devices, CudaDeviceInfo};
use crate::device::Device;

/// GPU readiness assessment result.
#[derive(Debug, Clone)]
pub struct GpuReadinessReport {
    /// Whether any GPU is available
    pub gpu_available: bool,

    /// Number of available GPUs
    pub gpu_count: usize,

    /// Detected GPU devices
    pub gpus: Vec<GpuCapability>,

    /// Recommended execution device
    pub recommended_device: Device,

    /// Reasons for recommendation
    pub recommendation_reasons: Vec<String>,

    /// Estimated speedup over CPU (if GPU available)
    pub estimated_speedup: Option<f64>,
}

/// Detailed GPU capability information.
#[derive(Debug, Clone)]
pub struct GpuCapability {
    /// Device information
    pub device: Device,

    /// GPU name
    pub name: String,

    /// Total memory in MB
    pub memory_mb: u64,

    /// Memory bandwidth in GB/s (estimated)
    pub memory_bandwidth_gbs: f64,

    /// Compute capability
    pub compute_capability: Option<(u32, u32)>,

    /// CUDA cores (estimated based on architecture)
    pub cuda_cores: Option<u32>,

    /// Tensor cores available
    pub has_tensor_cores: bool,

    /// FP16 support
    pub supports_fp16: bool,

    /// INT8 support
    pub supports_int8: bool,

    /// Recommended for this workload
    pub recommended: bool,
}

impl GpuCapability {
    /// Create GPU capability from CUDA device info.
    pub fn from_cuda_device(info: &CudaDeviceInfo) -> Self {
        let compute_capability = info.compute_capability;
        let has_tensor_cores = compute_capability
            .map(|(major, _minor)| major >= 7)
            .unwrap_or(false);

        let supports_fp16 = compute_capability
            .map(|(major, _)| major >= 6)
            .unwrap_or(false);

        let supports_int8 = compute_capability
            .map(|(major, _)| major >= 6)
            .unwrap_or(false);

        // Estimate memory bandwidth based on GPU name and memory size
        let memory_bandwidth_gbs = estimate_memory_bandwidth(&info.name, info.memory_mb);

        // Estimate CUDA cores based on compute capability
        let cuda_cores = compute_capability
            .and_then(|(major, minor)| estimate_cuda_cores(&info.name, major, minor));

        Self {
            device: Device::cuda(info.index),
            name: info.name.clone(),
            memory_mb: info.memory_mb,
            memory_bandwidth_gbs,
            compute_capability,
            cuda_cores,
            has_tensor_cores,
            supports_fp16,
            supports_int8,
            recommended: false,
        }
    }

    /// Get a capability score (higher is better).
    pub fn capability_score(&self) -> f64 {
        let mut score = 0.0;

        // Memory bandwidth contribution
        score += self.memory_bandwidth_gbs * 0.5;

        // Memory size contribution (GB)
        score += (self.memory_mb as f64 / 1024.0) * 2.0;

        // Compute capability contribution
        if let Some((major, minor)) = self.compute_capability {
            score += (major as f64 * 100.0) + (minor as f64 * 10.0);
        }

        // Tensor cores bonus
        if self.has_tensor_cores {
            score += 200.0;
        }

        // FP16/INT8 support
        if self.supports_fp16 {
            score += 50.0;
        }
        if self.supports_int8 {
            score += 30.0;
        }

        score
    }
}

/// Estimate memory bandwidth based on GPU name and memory size.
fn estimate_memory_bandwidth(name: &str, memory_mb: u64) -> f64 {
    let name_lower = name.to_lowercase();

    // Known GPU families and their typical bandwidth
    if name_lower.contains("a100") {
        1555.0 // A100 40GB/80GB
    } else if name_lower.contains("a6000") {
        768.0 // RTX A6000
    } else if name_lower.contains("rtx 3090") {
        936.0 // RTX 3090
    } else if name_lower.contains("rtx 3080") {
        760.0 // RTX 3080
    } else if name_lower.contains("rtx 3070") {
        448.0 // RTX 3070
    } else if name_lower.contains("v100") {
        900.0 // V100
    } else if name_lower.contains("h100") {
        3000.0 // H100
    } else {
        // Rough estimate based on memory size
        // Assume ~30 GB/s per GB of memory (very rough heuristic)
        (memory_mb as f64 / 1024.0) * 30.0
    }
}

/// Estimate CUDA cores based on GPU name and compute capability.
fn estimate_cuda_cores(name: &str, major: u32, minor: u32) -> Option<u32> {
    let name_lower = name.to_lowercase();

    // Known GPU models
    if name_lower.contains("a100") {
        Some(6912)
    } else if name_lower.contains("a6000") {
        Some(10752)
    } else if name_lower.contains("rtx 3090") {
        Some(10496)
    } else if name_lower.contains("rtx 3080") {
        Some(8704)
    } else if name_lower.contains("rtx 3070") {
        Some(5888)
    } else if name_lower.contains("v100") {
        Some(5120)
    } else if name_lower.contains("h100") {
        Some(14592)
    } else {
        // Rough estimate based on compute capability
        match (major, minor) {
            (8, 6) => Some(8192), // Ampere
            (8, 0) => Some(6912), // Ampere
            (7, 5) => Some(4608), // Turing
            (7, 0) => Some(5120), // Volta
            _ => None,
        }
    }
}

/// Assess GPU readiness for TensorLogic execution.
pub fn assess_gpu_readiness() -> GpuReadinessReport {
    let cuda_devices = detect_cuda_devices();
    let gpu_count = cuda_devices.len();
    let gpu_available = gpu_count > 0;

    let mut gpus: Vec<GpuCapability> = cuda_devices
        .iter()
        .map(GpuCapability::from_cuda_device)
        .collect();

    // Rank GPUs by capability score
    gpus.sort_by(|a, b| {
        b.capability_score()
            .partial_cmp(&a.capability_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Mark the best GPU as recommended
    if let Some(best_gpu) = gpus.first_mut() {
        best_gpu.recommended = true;
    }

    let mut recommendation_reasons = Vec::new();
    let recommended_device = if gpu_available {
        let best = &gpus[0];
        recommendation_reasons.push(format!(
            "GPU {} has highest capability score: {:.1}",
            best.name,
            best.capability_score()
        ));

        if best.has_tensor_cores {
            recommendation_reasons
                .push("GPU has Tensor Cores for accelerated matrix operations".to_string());
        }

        recommendation_reasons.push(format!(
            "GPU memory: {} GB ({:.0} GB/s bandwidth)",
            best.memory_mb / 1024,
            best.memory_bandwidth_gbs
        ));

        best.device.clone()
    } else {
        recommendation_reasons.push("No GPU detected, using CPU".to_string());
        recommendation_reasons.push("CPU is currently the only supported backend".to_string());
        Device::cpu()
    };

    let estimated_speedup = if gpu_available {
        Some(estimate_gpu_speedup(&gpus[0]))
    } else {
        None
    };

    GpuReadinessReport {
        gpu_available,
        gpu_count,
        gpus,
        recommended_device,
        recommendation_reasons,
        estimated_speedup,
    }
}

/// Estimate theoretical GPU speedup over CPU.
fn estimate_gpu_speedup(gpu: &GpuCapability) -> f64 {
    let mut speedup = 1.0;

    // Base speedup from memory bandwidth (GPU vs CPU ~30 GB/s)
    speedup *= gpu.memory_bandwidth_gbs / 30.0;

    // Compute capability contribution
    if let Some((major, _)) = gpu.compute_capability {
        speedup *= 1.0 + (major as f64 * 0.2);
    }

    // Tensor cores provide significant speedup for matrix operations
    if gpu.has_tensor_cores {
        speedup *= 1.5;
    }

    // Cap at realistic values (typical GPU speedup is 5-50x)
    speedup.clamp(1.0, 50.0)
}

/// Workload characteristics for optimization recommendations.
#[derive(Debug, Clone)]
pub struct WorkloadProfile {
    /// Number of tensor operations
    pub operation_count: usize,

    /// Average tensor size in elements
    pub avg_tensor_size: usize,

    /// Peak memory usage in MB
    pub peak_memory_mb: u64,

    /// Compute intensity (FLOPs per byte)
    pub compute_intensity: f64,
}

/// Recommend optimal batch size for GPU execution.
pub fn recommend_batch_size(gpu: &GpuCapability, workload: &WorkloadProfile) -> usize {
    let available_memory_mb = (gpu.memory_mb as f64 * 0.8) as u64; // Use 80% of GPU memory

    // Calculate memory per sample
    let memory_per_sample_mb = workload.peak_memory_mb;

    if memory_per_sample_mb == 0 {
        return 1;
    }

    // Maximum batch size based on memory
    let max_batch = (available_memory_mb / memory_per_sample_mb).max(1) as usize;

    // Adjust based on compute capability
    let compute_adjusted = if gpu.has_tensor_cores {
        max_batch.min(256) // Tensor cores work well with medium batches
    } else {
        max_batch.min(128)
    };

    // Ensure batch size is a power of 2 for optimal GPU utilization
    compute_adjusted.next_power_of_two() / 2
}

/// Generate optimization recommendations based on GPU capabilities.
pub fn generate_recommendations(
    report: &GpuReadinessReport,
    workload: Option<&WorkloadProfile>,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    if !report.gpu_available {
        recommendations.push(
            "Consider using SIMD optimizations with the 'simd' feature for CPU acceleration"
                .to_string(),
        );
        recommendations.push("Use the 'parallel' feature for multi-threaded execution".to_string());
        return recommendations;
    }

    let best_gpu = &report.gpus[0];

    // GPU-specific recommendations
    if best_gpu.has_tensor_cores {
        recommendations.push(
            "Enable FP16 mixed precision to utilize Tensor Cores (future feature)".to_string(),
        );
    }

    if best_gpu.supports_int8 {
        recommendations.push(
            "Consider INT8 quantization for inference workloads (future feature)".to_string(),
        );
    }

    // Memory recommendations
    if best_gpu.memory_mb < 8192 {
        recommendations
            .push("GPU has <8GB memory: Use gradient checkpointing for training".to_string());
    } else if best_gpu.memory_mb >= 40960 {
        recommendations.push("Large GPU memory available: Can use larger batch sizes".to_string());
    }

    // Workload-specific recommendations
    if let Some(wl) = workload {
        let batch_size = recommend_batch_size(best_gpu, wl);
        recommendations.push(format!(
            "Recommended batch size for GPU: {} (based on {} MB memory per sample)",
            batch_size, wl.peak_memory_mb
        ));

        if wl.compute_intensity < 10.0 {
            recommendations
                .push("Low compute intensity: Memory bandwidth is bottleneck".to_string());
        } else {
            recommendations.push("High compute intensity: Good for GPU acceleration".to_string());
        }
    }

    recommendations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_memory_bandwidth() {
        assert_eq!(estimate_memory_bandwidth("NVIDIA A100", 40960), 1555.0);
        assert_eq!(
            estimate_memory_bandwidth("NVIDIA GeForce RTX 3090", 24576),
            936.0
        );
        assert!(estimate_memory_bandwidth("Unknown GPU", 16384) > 0.0);
    }

    #[test]
    fn test_estimate_cuda_cores() {
        assert_eq!(estimate_cuda_cores("NVIDIA A100", 8, 0), Some(6912));
        assert_eq!(
            estimate_cuda_cores("NVIDIA GeForce RTX 3090", 8, 6),
            Some(10496)
        );
    }

    #[test]
    fn test_gpu_capability_score() {
        let cuda_info = CudaDeviceInfo {
            index: 0,
            name: "NVIDIA A100".to_string(),
            memory_mb: 40960,
            compute_capability: Some((8, 0)),
        };

        let cap = GpuCapability::from_cuda_device(&cuda_info);
        let score = cap.capability_score();

        // Should have high score due to tensor cores, memory, compute capability
        assert!(score > 1000.0);
        assert!(cap.has_tensor_cores);
        assert!(cap.supports_fp16);
        assert!(cap.supports_int8);
    }

    #[test]
    fn test_assess_gpu_readiness() {
        // Test GPU readiness assessment - behavior depends on actual hardware
        let report = assess_gpu_readiness();

        // Validate internal consistency regardless of GPU presence
        assert_eq!(report.gpu_count, report.gpus.len());
        assert_eq!(report.gpu_available, report.gpu_count > 0);

        if report.gpu_available {
            // If GPU is available, should have estimated speedup and recommend GPU
            assert!(report.estimated_speedup.is_some());
            assert_ne!(report.recommended_device, Device::cpu());
            // At least one GPU should be marked as recommended
            assert!(report.gpus.iter().any(|g| g.recommended));
        } else {
            // If no GPU, should recommend CPU and have no estimated speedup
            assert_eq!(report.recommended_device, Device::cpu());
            assert!(report.estimated_speedup.is_none());
        }

        // Should always have recommendation reasons
        assert!(!report.recommendation_reasons.is_empty());
    }

    #[test]
    fn test_recommend_batch_size() {
        let gpu = GpuCapability {
            device: Device::cuda(0),
            name: "Test GPU".to_string(),
            memory_mb: 16384,
            memory_bandwidth_gbs: 500.0,
            compute_capability: Some((8, 0)),
            cuda_cores: Some(8192),
            has_tensor_cores: true,
            supports_fp16: true,
            supports_int8: true,
            recommended: true,
        };

        let workload = WorkloadProfile {
            operation_count: 1000,
            avg_tensor_size: 100000,
            peak_memory_mb: 128,
            compute_intensity: 50.0,
        };

        let batch_size = recommend_batch_size(&gpu, &workload);

        // Should recommend reasonable batch size
        assert!(batch_size > 0);
        assert!(batch_size <= 256);
        // Should be power of 2
        assert_eq!(batch_size.count_ones(), 1);
    }

    #[test]
    fn test_generate_recommendations() {
        let report = GpuReadinessReport {
            gpu_available: false,
            gpu_count: 0,
            gpus: vec![],
            recommended_device: Device::cpu(),
            recommendation_reasons: vec![],
            estimated_speedup: None,
        };

        let recommendations = generate_recommendations(&report, None);

        assert!(!recommendations.is_empty());
        assert!(recommendations
            .iter()
            .any(|r| r.contains("SIMD") || r.contains("parallel")));
    }

    #[test]
    fn test_estimate_gpu_speedup() {
        let gpu = GpuCapability {
            device: Device::cuda(0),
            name: "High-end GPU".to_string(),
            memory_mb: 40960,
            memory_bandwidth_gbs: 1500.0,
            compute_capability: Some((8, 0)),
            cuda_cores: Some(10000),
            has_tensor_cores: true,
            supports_fp16: true,
            supports_int8: true,
            recommended: true,
        };

        let speedup = estimate_gpu_speedup(&gpu);

        // Should estimate significant speedup
        assert!(speedup > 1.0);
        assert!(speedup <= 50.0); // Capped at 50x
    }

    #[test]
    fn test_workload_profile_creation() {
        let profile = WorkloadProfile {
            operation_count: 5000,
            avg_tensor_size: 250000,
            peak_memory_mb: 512,
            compute_intensity: 75.0,
        };

        assert_eq!(profile.operation_count, 5000);
        assert_eq!(profile.peak_memory_mb, 512);
        assert!(profile.compute_intensity > 50.0);
    }
}
