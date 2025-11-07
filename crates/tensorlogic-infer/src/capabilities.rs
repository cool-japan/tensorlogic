//! Backend capability queries and feature detection.

use std::collections::HashSet;

use crate::ops::{ElemOp, ReduceOp};

/// Device types that a backend can execute on
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    CPU,
    GPU,
    TPU,
    Custom(u32),
}

impl DeviceType {
    pub fn as_str(&self) -> &str {
        match self {
            DeviceType::CPU => "CPU",
            DeviceType::GPU => "GPU",
            DeviceType::TPU => "TPU",
            DeviceType::Custom(_) => "Custom",
        }
    }
}

/// Precision/data type support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
    Custom(u32),
}

impl DType {
    pub fn as_str(&self) -> &str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::Bool => "bool",
            DType::Custom(_) => "custom",
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::Bool => 1,
            DType::Custom(_) => 0,
        }
    }
}

/// Backend feature flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Feature {
    /// Supports automatic differentiation
    Autodiff,
    /// Supports batched execution
    BatchExecution,
    /// Supports sparse tensors
    SparseTensors,
    /// Supports mixed precision
    MixedPrecision,
    /// Supports SIMD acceleration
    SIMDAcceleration,
    /// Supports GPU execution
    GPUAcceleration,
    /// Supports distributed execution
    DistributedExecution,
    /// Supports JIT compilation
    JIT,
    /// Custom feature
    Custom(u32),
}

impl Feature {
    pub fn as_str(&self) -> &str {
        match self {
            Feature::Autodiff => "Autodiff",
            Feature::BatchExecution => "BatchExecution",
            Feature::SparseTensors => "SparseTensors",
            Feature::MixedPrecision => "MixedPrecision",
            Feature::SIMDAcceleration => "SIMDAcceleration",
            Feature::GPUAcceleration => "GPUAcceleration",
            Feature::DistributedExecution => "DistributedExecution",
            Feature::JIT => "JIT",
            Feature::Custom(_) => "Custom",
        }
    }
}

/// Backend capabilities descriptor
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub name: String,
    pub version: String,
    pub supported_devices: HashSet<DeviceType>,
    pub supported_dtypes: HashSet<DType>,
    pub features: HashSet<Feature>,
    pub max_tensor_dims: usize,
    pub max_tensor_size: Option<usize>,
}

impl BackendCapabilities {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        BackendCapabilities {
            name: name.into(),
            version: version.into(),
            supported_devices: HashSet::new(),
            supported_dtypes: HashSet::new(),
            features: HashSet::new(),
            max_tensor_dims: 8, // Default max rank
            max_tensor_size: None,
        }
    }

    pub fn with_device(mut self, device: DeviceType) -> Self {
        self.supported_devices.insert(device);
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.supported_dtypes.insert(dtype);
        self
    }

    pub fn with_feature(mut self, feature: Feature) -> Self {
        self.features.insert(feature);
        self
    }

    pub fn with_max_dims(mut self, max_dims: usize) -> Self {
        self.max_tensor_dims = max_dims;
        self
    }

    pub fn supports_device(&self, device: DeviceType) -> bool {
        self.supported_devices.contains(&device)
    }

    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.supported_dtypes.contains(&dtype)
    }

    pub fn supports_feature(&self, feature: Feature) -> bool {
        self.features.contains(&feature)
    }

    pub fn can_execute_on(&self, device: DeviceType, dtype: DType) -> bool {
        self.supports_device(device) && self.supports_dtype(dtype)
    }

    /// Generate a summary of capabilities
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Backend: {} v{}\n", self.name, self.version));
        summary.push_str("Devices: ");
        for device in &self.supported_devices {
            summary.push_str(&format!("{} ", device.as_str()));
        }
        summary.push('\n');
        summary.push_str("Data Types: ");
        for dtype in &self.supported_dtypes {
            summary.push_str(&format!("{} ", dtype.as_str()));
        }
        summary.push('\n');
        summary.push_str("Features: ");
        for feature in &self.features {
            summary.push_str(&format!("{} ", feature.as_str()));
        }
        summary.push('\n');
        summary.push_str(&format!("Max Tensor Dims: {}\n", self.max_tensor_dims));
        summary
    }
}

/// Trait for backends to advertise their capabilities
pub trait TlCapabilities {
    /// Get backend capabilities
    fn capabilities(&self) -> &BackendCapabilities;

    /// Check if a specific operation is supported
    fn supports_elem_op(&self, op: ElemOp) -> bool {
        let _ = op;
        true // Default: support all ops
    }

    /// Check if a specific reduction operation is supported
    fn supports_reduce_op(&self, op: ReduceOp) -> bool {
        let _ = op;
        true // Default: support all ops
    }

    /// Check if einsum is supported with the given spec
    fn supports_einsum(&self, spec: &str) -> bool {
        let _ = spec;
        true // Default: support all einsum specs
    }

    /// Get available devices
    fn available_devices(&self) -> Vec<DeviceType> {
        self.capabilities()
            .supported_devices
            .iter()
            .copied()
            .collect()
    }

    /// Get default device
    fn default_device(&self) -> DeviceType {
        self.available_devices()
            .first()
            .copied()
            .unwrap_or(DeviceType::CPU)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type() {
        let cpu = DeviceType::CPU;
        assert_eq!(cpu.as_str(), "CPU");

        let gpu = DeviceType::GPU;
        assert_eq!(gpu.as_str(), "GPU");
    }

    #[test]
    fn test_dtype() {
        let f32 = DType::F32;
        assert_eq!(f32.as_str(), "f32");
        assert_eq!(f32.byte_size(), 4);

        let f64 = DType::F64;
        assert_eq!(f64.byte_size(), 8);
    }

    #[test]
    fn test_feature() {
        let autodiff = Feature::Autodiff;
        assert_eq!(autodiff.as_str(), "Autodiff");
    }

    #[test]
    fn test_backend_capabilities() {
        let caps = BackendCapabilities::new("TestBackend", "1.0")
            .with_device(DeviceType::CPU)
            .with_device(DeviceType::GPU)
            .with_dtype(DType::F32)
            .with_dtype(DType::F64)
            .with_feature(Feature::Autodiff)
            .with_max_dims(10);

        assert!(caps.supports_device(DeviceType::CPU));
        assert!(caps.supports_device(DeviceType::GPU));
        assert!(!caps.supports_device(DeviceType::TPU));

        assert!(caps.supports_dtype(DType::F32));
        assert!(!caps.supports_dtype(DType::I32));

        assert!(caps.supports_feature(Feature::Autodiff));
        assert!(!caps.supports_feature(Feature::BatchExecution));

        assert_eq!(caps.max_tensor_dims, 10);
    }

    #[test]
    fn test_can_execute_on() {
        let caps = BackendCapabilities::new("TestBackend", "1.0")
            .with_device(DeviceType::CPU)
            .with_dtype(DType::F32);

        assert!(caps.can_execute_on(DeviceType::CPU, DType::F32));
        assert!(!caps.can_execute_on(DeviceType::GPU, DType::F32));
        assert!(!caps.can_execute_on(DeviceType::CPU, DType::F64));
    }

    #[test]
    fn test_capabilities_summary() {
        let caps = BackendCapabilities::new("TestBackend", "1.0")
            .with_device(DeviceType::CPU)
            .with_dtype(DType::F32)
            .with_feature(Feature::Autodiff);

        let summary = caps.summary();
        assert!(summary.contains("TestBackend"));
        assert!(summary.contains("1.0"));
        assert!(summary.contains("CPU"));
        assert!(summary.contains("f32"));
        assert!(summary.contains("Autodiff"));
    }
}
