//! Device management for tensor computations.
//!
//! This module provides abstractions for managing compute devices
//! (CPU, GPU, etc.) and tensor placement.

use std::fmt;

/// Compute device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU device (default)
    Cpu,

    /// CUDA GPU device
    Cuda,

    /// Metal GPU device (Apple)
    Metal,

    /// Vulkan compute device
    Vulkan,

    /// ROCm GPU device (AMD)
    Rocm,
}

impl DeviceType {
    /// Returns true if this is a GPU device.
    pub fn is_gpu(&self) -> bool {
        matches!(
            self,
            DeviceType::Cuda | DeviceType::Metal | DeviceType::Vulkan | DeviceType::Rocm
        )
    }

    /// Returns true if this is a CPU device.
    pub fn is_cpu(&self) -> bool {
        matches!(self, DeviceType::Cpu)
    }

    /// Returns the name of this device type.
    pub fn name(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "CPU",
            DeviceType::Cuda => "CUDA",
            DeviceType::Metal => "Metal",
            DeviceType::Vulkan => "Vulkan",
            DeviceType::Rocm => "ROCm",
        }
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A specific compute device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Device {
    /// Device type
    pub device_type: DeviceType,

    /// Device index (for multi-GPU systems)
    pub index: usize,
}

impl Device {
    /// Create a CPU device.
    pub fn cpu() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            index: 0,
        }
    }

    /// Create a CUDA device with the given index.
    pub fn cuda(index: usize) -> Self {
        Self {
            device_type: DeviceType::Cuda,
            index,
        }
    }

    /// Create a Metal device.
    pub fn metal() -> Self {
        Self {
            device_type: DeviceType::Metal,
            index: 0,
        }
    }

    /// Create a Vulkan device with the given index.
    pub fn vulkan(index: usize) -> Self {
        Self {
            device_type: DeviceType::Vulkan,
            index,
        }
    }

    /// Create a ROCm device with the given index.
    pub fn rocm(index: usize) -> Self {
        Self {
            device_type: DeviceType::Rocm,
            index,
        }
    }

    /// Returns true if this is a CPU device.
    pub fn is_cpu(&self) -> bool {
        self.device_type.is_cpu()
    }

    /// Returns true if this is a GPU device.
    pub fn is_gpu(&self) -> bool {
        self.device_type.is_gpu()
    }

    /// Returns the device type.
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Returns the device index.
    pub fn index(&self) -> usize {
        self.index
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.index == 0 && self.is_cpu() {
            write!(f, "{}", self.device_type)
        } else {
            write!(f, "{}:{}", self.device_type, self.index)
        }
    }
}

/// Device manager for querying available devices.
#[derive(Debug, Clone)]
pub struct DeviceManager {
    /// List of available devices
    available_devices: Vec<Device>,

    /// Default device
    default_device: Device,
}

impl DeviceManager {
    /// Create a new device manager.
    ///
    /// This queries the system for available devices, including CUDA GPUs
    /// if available via nvidia-smi.
    pub fn new() -> Self {
        #[cfg(test)] // In tests, only CPU is available
        let available_devices = vec![Device::cpu()];

        #[cfg(not(test))] // In production, detect CUDA devices
        let available_devices = {
            let mut devices = vec![Device::cpu()];
            let cuda_devices = crate::cuda_detect::detect_cuda_devices();
            for cuda_info in cuda_devices {
                devices.push(Device::cuda(cuda_info.index));
            }
            devices
        };

        Self {
            available_devices: available_devices.clone(),
            default_device: available_devices[0].clone(),
        }
    }

    /// Get the list of available devices.
    pub fn available_devices(&self) -> &[Device] {
        &self.available_devices
    }

    /// Get the default device.
    pub fn default_device(&self) -> &Device {
        &self.default_device
    }

    /// Set the default device.
    pub fn set_default_device(&mut self, device: Device) -> Result<(), DeviceError> {
        if !self.available_devices.contains(&device) {
            return Err(DeviceError::DeviceNotAvailable(device));
        }
        self.default_device = device;
        Ok(())
    }

    /// Check if a device is available.
    pub fn is_available(&self, device: &Device) -> bool {
        self.available_devices.contains(device)
    }

    /// Get a device by type and index.
    pub fn get_device(&self, device_type: DeviceType, index: usize) -> Option<&Device> {
        self.available_devices
            .iter()
            .find(|d| d.device_type == device_type && d.index == index)
    }

    /// Count devices of a specific type.
    pub fn count_devices(&self, device_type: DeviceType) -> usize {
        self.available_devices
            .iter()
            .filter(|d| d.device_type == device_type)
            .count()
    }

    /// Check if any GPU devices are available.
    pub fn has_gpu(&self) -> bool {
        self.available_devices.iter().any(|d| d.is_gpu())
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Device-related errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum DeviceError {
    /// Device is not available
    #[error("Device not available: {0}")]
    DeviceNotAvailable(Device),

    /// Device memory allocation failed
    #[error("Device memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Device synchronization failed
    #[error("Device synchronization failed: {0}")]
    SyncFailed(String),

    /// Unsupported device operation
    #[error("Unsupported operation on device {device}: {operation}")]
    UnsupportedOperation { device: Device, operation: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_properties() {
        assert!(DeviceType::Cpu.is_cpu());
        assert!(!DeviceType::Cpu.is_gpu());

        assert!(DeviceType::Cuda.is_gpu());
        assert!(!DeviceType::Cuda.is_cpu());

        assert!(DeviceType::Metal.is_gpu());
        assert!(DeviceType::Vulkan.is_gpu());
        assert!(DeviceType::Rocm.is_gpu());
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::Cpu.to_string(), "CPU");
        assert_eq!(DeviceType::Cuda.to_string(), "CUDA");
        assert_eq!(DeviceType::Metal.to_string(), "Metal");
    }

    #[test]
    fn test_device_creation() {
        let cpu = Device::cpu();
        assert!(cpu.is_cpu());
        assert_eq!(cpu.index(), 0);

        let cuda = Device::cuda(1);
        assert!(cuda.is_gpu());
        assert_eq!(cuda.index(), 1);
        assert_eq!(cuda.device_type(), DeviceType::Cuda);
    }

    #[test]
    fn test_device_default() {
        let device = Device::default();
        assert!(device.is_cpu());
        assert_eq!(device.index(), 0);
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::cpu().to_string(), "CPU");
        assert_eq!(Device::cuda(0).to_string(), "CUDA:0");
        assert_eq!(Device::cuda(1).to_string(), "CUDA:1");
        assert_eq!(Device::metal().to_string(), "Metal:0");
    }

    #[test]
    fn test_device_manager_creation() {
        let manager = DeviceManager::new();
        assert!(!manager.available_devices().is_empty());
        assert!(manager.default_device().is_cpu());
    }

    #[test]
    fn test_device_manager_queries() {
        let manager = DeviceManager::new();

        // CPU should always be available
        assert!(manager.is_available(&Device::cpu()));
        assert_eq!(manager.count_devices(DeviceType::Cpu), 1);

        // Check default device
        assert_eq!(manager.default_device(), &Device::cpu());
    }

    #[test]
    fn test_device_manager_set_default() {
        let mut manager = DeviceManager::new();
        let cpu = Device::cpu();

        // Setting to an available device should succeed
        assert!(manager.set_default_device(cpu.clone()).is_ok());
        assert_eq!(manager.default_device(), &cpu);

        // Setting to an unavailable device should fail
        let cuda = Device::cuda(99);
        assert!(manager.set_default_device(cuda).is_err());
    }

    #[test]
    fn test_device_manager_get_device() {
        let manager = DeviceManager::new();

        // Should find CPU
        let cpu = manager.get_device(DeviceType::Cpu, 0);
        assert!(cpu.is_some());
        assert_eq!(cpu.unwrap(), &Device::cpu());

        // Should not find non-existent devices
        let cuda = manager.get_device(DeviceType::Cuda, 0);
        assert!(cuda.is_none());
    }

    #[test]
    fn test_device_error_display() {
        let err = DeviceError::DeviceNotAvailable(Device::cuda(0));
        assert!(err.to_string().contains("not available"));

        let err = DeviceError::AllocationFailed("out of memory".to_string());
        assert!(err.to_string().contains("allocation failed"));
    }
}
