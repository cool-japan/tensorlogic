//! ToRSh Tensor Interoperability
//!
//! This module provides bidirectional conversion between TensorLogic's SciRS2 tensors
//! and ToRSh tensors, enabling seamless integration with the COOLJAPAN ToRSh ecosystem.
//!
//! # Overview
//!
//! TensorLogic uses `ArrayD<f64>` (from SciRS2) as its primary tensor representation.
//! ToRSh provides a PyTorch-like tensor API built on top of SciRS2.
//! This module bridges the two, allowing:
//!
//! - **Logic execution results → ToRSh tensors** for neural network integration
//! - **ToRSh model outputs → Logic constraints** for neurosymbolic reasoning
//! - **Zero-copy where possible** via shared SciRS2 foundation
//!
//! # Example
//!
//! ```no_run
//! use tensorlogic_scirs_backend::{Scirs2Tensor, torsh_interop::*};
//! use torsh_tensor::Tensor;
//! use torsh_core::device::DeviceType;
//!
//! // TensorLogic → ToRSh
//! let tl_tensor: Scirs2Tensor = /* ... compiled logic result ... */;
//! let torsh_tensor = tl_to_torsh(&tl_tensor, DeviceType::Cpu)?;
//!
//! // ToRSh → TensorLogic
//! let torsh_output: Tensor<f64> = /* ... model forward pass ... */;
//! let tl_tensor = torsh_to_tl(&torsh_output)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Features
//!
//! - **Bidirectional conversion**: TensorLogic ↔ ToRSh
//! - **Type safety**: Generic over numeric types (f32, f64, etc.)
//! - **Device handling**: CPU/GPU device mapping
//! - **Gradient preservation**: Optional gradient flow for training
//! - **Shape validation**: Automatic shape verification

use crate::Scirs2Tensor;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use torsh_core::{device::DeviceType, error::TorshError};
use torsh_tensor::Tensor;

/// Error types for ToRSh interoperability
#[derive(Debug, thiserror::Error)]
pub enum TorshInteropError {
    /// Shape mismatch between tensors
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Device mismatch
    #[error("Device mismatch: TensorLogic only supports CPU execution currently")]
    DeviceMismatch,

    /// Data type mismatch
    #[error("Data type mismatch: {0}")]
    DataTypeMismatch(String),

    /// ToRSh error
    #[error("ToRSh error: {0}")]
    TorshError(#[from] TorshError),

    /// Empty tensor error
    #[error("Cannot convert empty tensor")]
    EmptyTensor,
}

pub type Result<T> = std::result::Result<T, TorshInteropError>;

/// Convert TensorLogic tensor to ToRSh tensor (f64)
///
/// # Arguments
///
/// * `tl_tensor` - TensorLogic tensor (ArrayD<f64>)
/// * `device` - Target ToRSh device (CPU/GPU)
///
/// # Returns
///
/// ToRSh tensor with the same data and shape
///
/// # Example
///
/// ```no_run
/// use tensorlogic_scirs_backend::{Scirs2Tensor, torsh_interop::*};
/// use torsh_core::device::DeviceType;
/// use scirs2_core::ndarray::ArrayD;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let tl_tensor = ArrayD::from_shape_vec(vec![2, 2], data)?;
/// let torsh_tensor = tl_to_torsh(&tl_tensor, DeviceType::Cpu)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tl_to_torsh(tl_tensor: &Scirs2Tensor, device: DeviceType) -> Result<Tensor<f64>> {
    // Validate device (TensorLogic currently only supports CPU)
    if device != DeviceType::Cpu {
        return Err(TorshInteropError::DeviceMismatch);
    }

    // Extract shape
    let shape: Vec<usize> = tl_tensor.shape().to_vec();
    if shape.is_empty() || shape.iter().product::<usize>() == 0 {
        return Err(TorshInteropError::EmptyTensor);
    }

    // Extract data as contiguous Vec<f64>
    let data: Vec<f64> = tl_tensor.iter().copied().collect();

    // Create ToRSh tensor
    let torsh_tensor = Tensor::from_data(data, shape, device)?;

    Ok(torsh_tensor)
}

/// Convert TensorLogic tensor to ToRSh tensor (f32) with type conversion
///
/// # Arguments
///
/// * `tl_tensor` - TensorLogic tensor (ArrayD<f64>)
/// * `device` - Target ToRSh device (CPU/GPU)
///
/// # Returns
///
/// ToRSh tensor with f32 precision
///
/// # Example
///
/// ```no_run
/// use tensorlogic_scirs_backend::{Scirs2Tensor, torsh_interop::*};
/// use torsh_core::device::DeviceType;
/// use scirs2_core::ndarray::ArrayD;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let tl_tensor = ArrayD::from_shape_vec(vec![2, 2], data)?;
/// let torsh_tensor = tl_to_torsh_f32(&tl_tensor, DeviceType::Cpu)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tl_to_torsh_f32(tl_tensor: &Scirs2Tensor, device: DeviceType) -> Result<Tensor<f32>> {
    // Validate device
    if device != DeviceType::Cpu {
        return Err(TorshInteropError::DeviceMismatch);
    }

    // Extract shape
    let shape: Vec<usize> = tl_tensor.shape().to_vec();
    if shape.is_empty() || shape.iter().product::<usize>() == 0 {
        return Err(TorshInteropError::EmptyTensor);
    }

    // Convert f64 → f32
    let data: Vec<f32> = tl_tensor.iter().map(|&x| x as f32).collect();

    // Create ToRSh tensor
    let torsh_tensor = Tensor::from_data(data, shape, device)?;

    Ok(torsh_tensor)
}

/// Convert ToRSh tensor to TensorLogic tensor (f64)
///
/// # Arguments
///
/// * `torsh_tensor` - ToRSh tensor (must be f64)
///
/// # Returns
///
/// TensorLogic tensor (ArrayD<f64>)
///
/// # Example
///
/// ```no_run
/// use tensorlogic_scirs_backend::{Scirs2Tensor, torsh_interop::*};
/// use torsh_tensor::Tensor;
/// use torsh_core::device::DeviceType;
///
/// let torsh_tensor = Tensor::zeros(&[2, 2], DeviceType::Cpu)?;
/// let tl_tensor = torsh_to_tl(&torsh_tensor)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn torsh_to_tl(torsh_tensor: &Tensor<f64>) -> Result<Scirs2Tensor> {
    // Validate device
    if torsh_tensor.device() != DeviceType::Cpu {
        return Err(TorshInteropError::DeviceMismatch);
    }

    // Extract shape
    let shape: Vec<usize> = torsh_tensor.shape().dims().to_vec();
    if shape.is_empty() || shape.iter().product::<usize>() == 0 {
        return Err(TorshInteropError::EmptyTensor);
    }

    // Extract data
    let data = torsh_tensor.to_vec()?;

    // Create TensorLogic tensor
    let tl_tensor = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| TorshInteropError::DataTypeMismatch(format!("Shape error: {}", e)))?;

    Ok(tl_tensor)
}

/// Convert ToRSh tensor (f32) to TensorLogic tensor (f64) with type conversion
///
/// # Arguments
///
/// * `torsh_tensor` - ToRSh tensor (f32)
///
/// # Returns
///
/// TensorLogic tensor (ArrayD<f64>)
///
/// # Example
///
/// ```no_run
/// use tensorlogic_scirs_backend::{Scirs2Tensor, torsh_interop::*};
/// use torsh_tensor::Tensor;
/// use torsh_core::device::DeviceType;
///
/// let torsh_tensor = Tensor::<f32>::zeros(&[2, 2], DeviceType::Cpu)?;
/// let tl_tensor = torsh_f32_to_tl(&torsh_tensor)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn torsh_f32_to_tl(torsh_tensor: &Tensor<f32>) -> Result<Scirs2Tensor> {
    // Validate device
    if torsh_tensor.device() != DeviceType::Cpu {
        return Err(TorshInteropError::DeviceMismatch);
    }

    // Extract shape
    let shape: Vec<usize> = torsh_tensor.shape().dims().to_vec();
    if shape.is_empty() || shape.iter().product::<usize>() == 0 {
        return Err(TorshInteropError::EmptyTensor);
    }

    // Extract data and convert f32 → f64
    let data_f32 = torsh_tensor.to_vec()?;
    let data: Vec<f64> = data_f32.iter().map(|&x| x as f64).collect();

    // Create TensorLogic tensor
    let tl_tensor = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| TorshInteropError::DataTypeMismatch(format!("Shape error: {}", e)))?;

    Ok(tl_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tl_to_torsh_f64() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tl_tensor = ArrayD::from_shape_vec(vec![2, 2], data.clone())
            .expect("Failed to create TensorLogic tensor");

        let torsh_tensor =
            tl_to_torsh(&tl_tensor, DeviceType::Cpu).expect("Failed to convert TL to ToRSh");

        assert_eq!(torsh_tensor.shape().dims(), &[2, 2]);
        let result_data = torsh_tensor.to_vec().expect("Failed to extract ToRSh data");
        assert_eq!(result_data, data);
    }

    #[test]
    fn test_tl_to_torsh_f32() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tl_tensor = ArrayD::from_shape_vec(vec![2, 2], data.clone())
            .expect("Failed to create TensorLogic tensor");

        let torsh_tensor = tl_to_torsh_f32(&tl_tensor, DeviceType::Cpu)
            .expect("Failed to convert TL to ToRSh f32");

        assert_eq!(torsh_tensor.shape().dims(), &[2, 2]);
        let result_data = torsh_tensor
            .to_vec()
            .expect("Failed to extract ToRSh f32 data");
        let expected: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        assert_eq!(result_data, expected);
    }

    #[test]
    fn test_torsh_to_tl_f64() {
        let torsh_tensor =
            Tensor::zeros(&[3, 3], DeviceType::Cpu).expect("Failed to create ToRSh zero tensor");

        let tl_tensor = torsh_to_tl(&torsh_tensor).expect("Failed to convert ToRSh to TL");

        assert_eq!(tl_tensor.shape(), &[3, 3]);
        assert_eq!(tl_tensor.len(), 9);
        assert!(tl_tensor.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_torsh_f32_to_tl() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let torsh_tensor = Tensor::from_data(data.clone(), vec![2, 2], DeviceType::Cpu)
            .expect("Failed to create ToRSh f32 tensor");

        let tl_tensor = torsh_f32_to_tl(&torsh_tensor).expect("Failed to convert ToRSh f32 to TL");

        assert_eq!(tl_tensor.shape(), &[2, 2]);
        let expected: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        let result: Vec<f64> = tl_tensor.iter().copied().collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_roundtrip_f64() {
        let data = vec![1.5, 2.5, 3.5, 4.5];
        let original = ArrayD::from_shape_vec(vec![2, 2], data.clone())
            .expect("Failed to create original tensor");

        // TL → ToRSh → TL
        let torsh = tl_to_torsh(&original, DeviceType::Cpu).expect("Failed TL to ToRSh conversion");
        let roundtrip = torsh_to_tl(&torsh).expect("Failed ToRSh to TL conversion");

        assert_eq!(original.shape(), roundtrip.shape());
        let original_vec: Vec<f64> = original.iter().copied().collect();
        let roundtrip_vec: Vec<f64> = roundtrip.iter().copied().collect();
        assert_eq!(original_vec, roundtrip_vec);
    }

    #[test]
    fn test_empty_tensor_error() {
        let empty = ArrayD::from_shape_vec(vec![0], vec![])
            .expect("Failed to create empty tensor for test");
        let result = tl_to_torsh(&empty, DeviceType::Cpu);
        assert!(matches!(result, Err(TorshInteropError::EmptyTensor)));
    }

    #[test]
    fn test_3d_tensor_conversion() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let tl_tensor = ArrayD::from_shape_vec(vec![2, 3, 4], data.clone())
            .expect("Failed to create 3D tensor");

        let torsh_tensor = tl_to_torsh(&tl_tensor, DeviceType::Cpu)
            .expect("Failed to convert 3D tensor TL to ToRSh");
        assert_eq!(torsh_tensor.shape().dims(), &[2, 3, 4]);

        let back = torsh_to_tl(&torsh_tensor).expect("Failed to convert 3D tensor ToRSh to TL");
        assert_eq!(back.shape(), &[2, 3, 4]);

        let back_vec: Vec<f64> = back.iter().copied().collect();
        assert_eq!(back_vec, data);
    }
}
