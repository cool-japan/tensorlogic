//! CUDA device detection utilities.
//!
//! This module provides utilities for detecting available CUDA devices
//! without requiring full CUDA runtime support. It uses system queries
//! to check for NVIDIA GPUs and CUDA installations.

use crate::device::Device;
use std::process::Command;

/// Information about a detected CUDA device.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Device index
    pub index: usize,
    /// Device name
    pub name: String,
    /// Total memory in MB
    pub memory_mb: u64,
    /// Compute capability (major, minor)
    pub compute_capability: Option<(u32, u32)>,
}

/// Detect available CUDA devices using nvidia-smi.
///
/// This function attempts to query CUDA devices using the nvidia-smi command.
/// Returns a list of detected devices, or an empty vec if detection fails.
///
/// # Example
///
/// ```no_run
/// use tensorlogic_scirs_backend::cuda_detect::detect_cuda_devices;
///
/// let devices = detect_cuda_devices();
/// println!("Found {} CUDA devices", devices.len());
/// for device in devices {
///     println!("  GPU {}: {} ({} MB)", device.index, device.name, device.memory_mb);
/// }
/// ```
pub fn detect_cuda_devices() -> Vec<CudaDeviceInfo> {
    // Try to run nvidia-smi to detect CUDA devices
    run_nvidia_smi().unwrap_or_default()
}

/// Check if CUDA is available on the system.
///
/// This checks for the presence of nvidia-smi and CUDA environment variables.
pub fn is_cuda_available() -> bool {
    // Check for CUDA_VISIBLE_DEVICES or other CUDA env vars
    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
        || std::env::var("CUDA_HOME").is_ok()
        || std::env::var("CUDA_PATH").is_ok()
    {
        return true;
    }

    // Try to run nvidia-smi
    Command::new("nvidia-smi")
        .arg("--query-gpu=count")
        .arg("--format=csv,noheader")
        .output()
        .is_ok()
}

/// Run nvidia-smi to query device information.
fn run_nvidia_smi() -> Result<Vec<CudaDeviceInfo>, String> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=index,name,memory.total,compute_cap")
        .arg("--format=csv,noheader,nounits")
        .output()
        .map_err(|e| format!("Failed to run nvidia-smi: {}", e))?;

    if !output.status.success() {
        return Err("nvidia-smi command failed".to_string());
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|e| format!("Failed to parse nvidia-smi output: {}", e))?;

    parse_nvidia_smi_output(&stdout)
}

/// Parse nvidia-smi CSV output.
fn parse_nvidia_smi_output(output: &str) -> Result<Vec<CudaDeviceInfo>, String> {
    let mut devices = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 3 {
            continue;
        }

        let index = parts[0]
            .parse::<usize>()
            .map_err(|e| format!("Failed to parse device index: {}", e))?;

        let name = parts[1].to_string();

        let memory_mb = parts[2]
            .parse::<u64>()
            .map_err(|e| format!("Failed to parse memory: {}", e))?;

        // Parse compute capability if available (format: "major.minor")
        let compute_capability = if parts.len() > 3 {
            parse_compute_capability(parts[3])
        } else {
            None
        };

        devices.push(CudaDeviceInfo {
            index,
            name,
            memory_mb,
            compute_capability,
        });
    }

    Ok(devices)
}

/// Parse compute capability string (e.g., "8.6" -> (8, 6))
fn parse_compute_capability(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 2 {
        return None;
    }

    let major = parts[0].parse::<u32>().ok()?;
    let minor = parts[1].parse::<u32>().ok()?;

    Some((major, minor))
}

/// Convert detected CUDA devices to Device structs.
pub fn cuda_devices_to_device_list(cuda_devices: &[CudaDeviceInfo]) -> Vec<Device> {
    cuda_devices
        .iter()
        .map(|info| Device::cuda(info.index))
        .collect()
}

/// Get the number of available CUDA devices.
pub fn cuda_device_count() -> usize {
    detect_cuda_devices().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_compute_capability() {
        assert_eq!(parse_compute_capability("8.6"), Some((8, 6)));
        assert_eq!(parse_compute_capability("7.5"), Some((7, 5)));
        assert_eq!(parse_compute_capability("3.5"), Some((3, 5)));
        assert_eq!(parse_compute_capability("invalid"), None);
        assert_eq!(parse_compute_capability("8"), None);
    }

    #[test]
    fn test_parse_nvidia_smi_output_single_device() {
        let output = "0, NVIDIA GeForce RTX 3090, 24576, 8.6\n";
        let devices = parse_nvidia_smi_output(output).unwrap();

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].index, 0);
        assert_eq!(devices[0].name, "NVIDIA GeForce RTX 3090");
        assert_eq!(devices[0].memory_mb, 24576);
        assert_eq!(devices[0].compute_capability, Some((8, 6)));
    }

    #[test]
    fn test_parse_nvidia_smi_output_multiple_devices() {
        let output = "0, NVIDIA GeForce RTX 3090, 24576, 8.6\n1, NVIDIA A100, 40960, 8.0\n";
        let devices = parse_nvidia_smi_output(output).unwrap();

        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].name, "NVIDIA GeForce RTX 3090");
        assert_eq!(devices[1].name, "NVIDIA A100");
        assert_eq!(devices[1].memory_mb, 40960);
    }

    #[test]
    fn test_parse_nvidia_smi_output_no_compute_cap() {
        let output = "0, NVIDIA GeForce RTX 3090, 24576\n";
        let devices = parse_nvidia_smi_output(output).unwrap();

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].compute_capability, None);
    }

    #[test]
    fn test_parse_nvidia_smi_output_empty() {
        let output = "";
        let devices = parse_nvidia_smi_output(output).unwrap();
        assert_eq!(devices.len(), 0);
    }

    #[test]
    fn test_cuda_devices_to_device_list() {
        let cuda_devices = vec![
            CudaDeviceInfo {
                index: 0,
                name: "GPU 0".to_string(),
                memory_mb: 8192,
                compute_capability: Some((8, 6)),
            },
            CudaDeviceInfo {
                index: 1,
                name: "GPU 1".to_string(),
                memory_mb: 16384,
                compute_capability: Some((8, 0)),
            },
        ];

        let devices = cuda_devices_to_device_list(&cuda_devices);

        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0], Device::cuda(0));
        assert_eq!(devices[1], Device::cuda(1));
    }

    #[test]
    fn test_is_cuda_available_with_env() {
        // This test will only pass if CUDA environment variables are set
        // or if nvidia-smi is available on the system
        let available = is_cuda_available();

        // We can't assert a specific value since it depends on the system
        // Just verify the function runs without panicking
        let _ = available;
    }

    #[test]
    fn test_cuda_device_count() {
        // Just verify this runs without panicking
        let _count = cuda_device_count();
        // Count is always >= 0 (usize), so no assertion needed
    }
}
