//! Backend capability detection and reporting.

use tensorlogic_infer::{BackendCapabilities, DType, DeviceType, Feature, TlCapabilities};

use crate::Scirs2Exec;

impl Scirs2Exec {
    fn build_capabilities() -> BackendCapabilities {
        let mut caps = BackendCapabilities::new("SciRS2 Backend", env!("CARGO_PKG_VERSION"));

        // Add supported devices
        caps.supported_devices.insert(DeviceType::CPU);
        if cfg!(feature = "gpu") {
            caps.supported_devices.insert(DeviceType::GPU);
        }

        // Add supported dtypes
        caps.supported_dtypes.insert(DType::F32);
        caps.supported_dtypes.insert(DType::F64);

        // Add features
        caps.features.insert(Feature::Autodiff);
        caps.features.insert(Feature::BatchExecution);

        // SIMD acceleration
        #[cfg(feature = "simd")]
        {
            caps.features.insert(Feature::SIMDAcceleration);
        }

        // GPU acceleration
        #[cfg(feature = "gpu")]
        {
            caps.features.insert(Feature::GPUAcceleration);
        }

        // Set tensor limits
        caps.max_tensor_dims = 16; // ndarray limit

        caps
    }
}

impl TlCapabilities for Scirs2Exec {
    fn capabilities(&self) -> &BackendCapabilities {
        // Use OnceLock for safe lazy initialization (Rust 2024 edition)
        use std::sync::OnceLock;
        static CAPS: OnceLock<BackendCapabilities> = OnceLock::new();

        CAPS.get_or_init(Self::build_capabilities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_capabilities() {
        let executor = Scirs2Exec::new();
        let caps = executor.capabilities();

        assert_eq!(caps.name, "SciRS2 Backend");
        assert!(!caps.version.is_empty());
        assert!(caps.supported_devices.contains(&DeviceType::CPU));
        assert!(caps.max_tensor_dims > 0);
    }

    #[test]
    fn test_dtype_support() {
        let executor = Scirs2Exec::new();
        let caps = executor.capabilities();

        assert!(caps.supported_dtypes.contains(&DType::F64));
        assert!(caps.supported_dtypes.contains(&DType::F32));
    }

    #[test]
    fn test_features() {
        let executor = Scirs2Exec::new();
        let caps = executor.capabilities();

        assert!(caps.features.contains(&Feature::Autodiff));
        assert!(caps.features.contains(&Feature::BatchExecution));
    }

    #[test]
    fn test_available_devices() {
        let executor = Scirs2Exec::new();
        let devices = executor.available_devices();

        assert!(devices.contains(&DeviceType::CPU));
    }
}
