//! Quantization Infrastructure for TensorLogic
//!
//! This module provides utilities for quantizing tensors to lower precision
//! formats (INT8, FP16, BF16) for improved memory efficiency and performance.
//! While full quantized execution requires backend support, this infrastructure
//! prepares the framework for future quantization-aware training and inference.

use crate::{Scirs2Tensor, TlBackendError, TlBackendResult};
use serde::{Deserialize, Serialize};

/// Quantization data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 8-bit signed integer quantization
    Int8,
    /// 16-bit floating point (IEEE 754 half precision)
    Fp16,
    /// 16-bit brain floating point (truncated FP32)
    BFloat16,
    /// 4-bit integer quantization (experimental)
    Int4,
    /// No quantization (full precision)
    None,
}

impl QuantizationType {
    /// Get the number of bits used by this quantization type.
    pub fn bits(&self) -> usize {
        match self {
            QuantizationType::Int4 => 4,
            QuantizationType::Int8 => 8,
            QuantizationType::Fp16 | QuantizationType::BFloat16 => 16,
            QuantizationType::None => 64, // Assuming f64 for full precision
        }
    }

    /// Get the memory compression ratio compared to FP64.
    pub fn compression_ratio(&self) -> f64 {
        64.0 / self.bits() as f64
    }

    /// Check if this is a floating-point quantization.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            QuantizationType::Fp16 | QuantizationType::BFloat16 | QuantizationType::None
        )
    }

    /// Check if this is an integer quantization.
    pub fn is_integer(&self) -> bool {
        matches!(self, QuantizationType::Int8 | QuantizationType::Int4)
    }
}

/// Quantization scheme (symmetric vs asymmetric).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// Symmetric quantization: range is [-max, max]
    Symmetric,
    /// Asymmetric quantization: range is [min, max]
    Asymmetric,
}

/// Quantization granularity (per-tensor vs per-channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationGranularity {
    /// Single scale and zero-point for entire tensor
    PerTensor,
    /// Separate scale and zero-point per output channel
    PerChannel,
}

/// Quantization parameters for a tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Quantization data type
    pub qtype: QuantizationType,

    /// Quantization scheme
    pub scheme: QuantizationScheme,

    /// Quantization granularity
    pub granularity: QuantizationGranularity,

    /// Scale factor(s) for dequantization
    pub scale: Vec<f64>,

    /// Zero point(s) for asymmetric quantization
    pub zero_point: Vec<i32>,

    /// Minimum value(s) in original tensor
    pub min_val: Vec<f64>,

    /// Maximum value(s) in original tensor
    pub max_val: Vec<f64>,
}

impl QuantizationParams {
    /// Create symmetric per-tensor quantization parameters.
    pub fn symmetric_per_tensor(qtype: QuantizationType, tensor: &Scirs2Tensor) -> Self {
        let abs_max = tensor.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);

        let scale = match qtype {
            QuantizationType::Int8 => abs_max / 127.0,
            QuantizationType::Int4 => abs_max / 7.0,
            QuantizationType::Fp16 | QuantizationType::BFloat16 => 1.0,
            QuantizationType::None => 1.0,
        };

        Self {
            qtype,
            scheme: QuantizationScheme::Symmetric,
            granularity: QuantizationGranularity::PerTensor,
            scale: vec![scale],
            zero_point: vec![0],
            min_val: vec![-abs_max],
            max_val: vec![abs_max],
        }
    }

    /// Create asymmetric per-tensor quantization parameters.
    pub fn asymmetric_per_tensor(qtype: QuantizationType, tensor: &Scirs2Tensor) -> Self {
        let min_val = tensor.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = tensor.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let (scale, zero_point) = match qtype {
            QuantizationType::Int8 => {
                let scale = (max_val - min_val) / 255.0;
                let zero_point = (-min_val / scale).round() as i32;
                (scale, zero_point)
            }
            QuantizationType::Int4 => {
                let scale = (max_val - min_val) / 15.0;
                let zero_point = (-min_val / scale).round() as i32;
                (scale, zero_point)
            }
            QuantizationType::Fp16 | QuantizationType::BFloat16 | QuantizationType::None => {
                (1.0, 0)
            }
        };

        Self {
            qtype,
            scheme: QuantizationScheme::Asymmetric,
            granularity: QuantizationGranularity::PerTensor,
            scale: vec![scale],
            zero_point: vec![zero_point],
            min_val: vec![min_val],
            max_val: vec![max_val],
        }
    }

    /// Get the dynamic range of this quantization.
    pub fn dynamic_range(&self) -> f64 {
        self.max_val[0] - self.min_val[0]
    }

    /// Get the quantization error bound.
    pub fn quantization_error_bound(&self) -> f64 {
        self.scale[0] / 2.0
    }
}

/// Simulated quantized tensor (stored as f64 but representing quantized values).
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// The quantized data (stored as f64 for compatibility)
    pub data: Scirs2Tensor,

    /// Quantization parameters
    pub params: QuantizationParams,
}

impl QuantizedTensor {
    /// Quantize a tensor using the given parameters.
    pub fn quantize(tensor: &Scirs2Tensor, params: QuantizationParams) -> Self {
        let quantized_data = match params.qtype {
            QuantizationType::Int8 => quantize_int8(tensor, &params),
            QuantizationType::Int4 => quantize_int4(tensor, &params),
            QuantizationType::Fp16 => quantize_fp16(tensor),
            QuantizationType::BFloat16 => quantize_bf16(tensor),
            QuantizationType::None => tensor.clone(),
        };

        Self {
            data: quantized_data,
            params,
        }
    }

    /// Dequantize the tensor back to full precision.
    pub fn dequantize(&self) -> Scirs2Tensor {
        match self.params.qtype {
            QuantizationType::Int8 | QuantizationType::Int4 => {
                dequantize_integer(&self.data, &self.params)
            }
            QuantizationType::Fp16 | QuantizationType::BFloat16 => {
                // Already in f64, just return
                self.data.clone()
            }
            QuantizationType::None => self.data.clone(),
        }
    }

    /// Get the memory size reduction ratio.
    pub fn memory_reduction(&self) -> f64 {
        self.params.qtype.compression_ratio()
    }

    /// Calculate the quantization error (MSE).
    pub fn quantization_error(&self, original: &Scirs2Tensor) -> f64 {
        let dequantized = self.dequantize();
        let diff = &dequantized - original;
        let squared_error: f64 = diff.iter().map(|&x| x * x).sum();
        squared_error / original.len() as f64
    }
}

/// Quantize tensor to INT8.
fn quantize_int8(tensor: &Scirs2Tensor, params: &QuantizationParams) -> Scirs2Tensor {
    let scale = params.scale[0];
    let zero_point = params.zero_point[0];

    tensor.mapv(|x| {
        let quantized = (x / scale).round() + zero_point as f64;
        quantized.clamp(-128.0, 127.0)
    })
}

/// Quantize tensor to INT4.
fn quantize_int4(tensor: &Scirs2Tensor, params: &QuantizationParams) -> Scirs2Tensor {
    let scale = params.scale[0];
    let zero_point = params.zero_point[0];

    tensor.mapv(|x| {
        let quantized = (x / scale).round() + zero_point as f64;
        quantized.clamp(-8.0, 7.0)
    })
}

/// Simulate FP16 quantization (with rounding to FP16 precision).
fn quantize_fp16(tensor: &Scirs2Tensor) -> Scirs2Tensor {
    tensor.mapv(|x| {
        // Simulate FP16 by limiting mantissa precision
        // FP16 has 10 mantissa bits vs FP64's 52 bits
        let scaled = x * (1024.0f64).powi(2);
        (scaled.round() / (1024.0f64).powi(2)).clamp(-65504.0, 65504.0)
    })
}

/// Simulate BFloat16 quantization.
fn quantize_bf16(tensor: &Scirs2Tensor) -> Scirs2Tensor {
    tensor.mapv(|x| {
        // BF16 has 7 mantissa bits vs FP64's 52 bits
        let scaled = x * (128.0f64).powi(2);
        scaled.round() / (128.0f64).powi(2)
    })
}

/// Dequantize integer-quantized tensor.
fn dequantize_integer(tensor: &Scirs2Tensor, params: &QuantizationParams) -> Scirs2Tensor {
    let scale = params.scale[0];
    let zero_point = params.zero_point[0];

    tensor.mapv(|q| (q - zero_point as f64) * scale)
}

/// Quantization-aware training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QatConfig {
    /// Target quantization type
    pub target_qtype: QuantizationType,

    /// Quantization scheme
    pub scheme: QuantizationScheme,

    /// Number of warmup epochs before enabling quantization
    pub warmup_epochs: usize,

    /// Whether to use straight-through estimator for gradients
    pub use_ste: bool,

    /// Whether to learn scale and zero-point parameters
    pub learnable_params: bool,
}

impl Default for QatConfig {
    fn default() -> Self {
        Self {
            target_qtype: QuantizationType::Int8,
            scheme: QuantizationScheme::Symmetric,
            warmup_epochs: 2,
            use_ste: true,
            learnable_params: false,
        }
    }
}

/// Quantization statistics for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationStats {
    /// Number of quantized tensors
    pub num_tensors: usize,

    /// Total memory saved (in bytes)
    pub memory_saved: u64,

    /// Average quantization error (MSE)
    pub avg_error: f64,

    /// Maximum quantization error
    pub max_error: f64,

    /// Distribution of quantization types used
    pub type_distribution: Vec<(QuantizationType, usize)>,
}

impl QuantizationStats {
    /// Create empty statistics.
    pub fn new() -> Self {
        Self {
            num_tensors: 0,
            memory_saved: 0,
            avg_error: 0.0,
            max_error: 0.0,
            type_distribution: Vec::new(),
        }
    }

    /// Update statistics with a new quantized tensor.
    pub fn update(&mut self, original_size: u64, compression_ratio: f64, error: f64) {
        self.num_tensors += 1;
        self.memory_saved += (original_size as f64 * (1.0 - 1.0 / compression_ratio)) as u64;

        // Update running average error
        let n = self.num_tensors as f64;
        self.avg_error = (self.avg_error * (n - 1.0) + error) / n;
        self.max_error = self.max_error.max(error);
    }

    /// Get memory reduction percentage.
    pub fn memory_reduction_pct(&self, total_memory: u64) -> f64 {
        if total_memory == 0 {
            0.0
        } else {
            (self.memory_saved as f64 / total_memory as f64) * 100.0
        }
    }
}

impl Default for QuantizationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Calibrate quantization parameters using sample data.
pub fn calibrate_quantization(
    samples: &[Scirs2Tensor],
    qtype: QuantizationType,
    scheme: QuantizationScheme,
) -> TlBackendResult<QuantizationParams> {
    if samples.is_empty() {
        return Err(TlBackendError::GraphError(
            "Cannot calibrate with empty samples".to_string(),
        ));
    }

    // Collect statistics across all samples
    let mut global_min = f64::INFINITY;
    let mut global_max = f64::NEG_INFINITY;
    let mut global_abs_max = 0.0f64;

    for sample in samples {
        let sample_min = sample.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let sample_max = sample.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sample_abs_max = sample.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);

        global_min = global_min.min(sample_min);
        global_max = global_max.max(sample_max);
        global_abs_max = global_abs_max.max(sample_abs_max);
    }

    let params = match scheme {
        QuantizationScheme::Symmetric => {
            let scale = match qtype {
                QuantizationType::Int8 => global_abs_max / 127.0,
                QuantizationType::Int4 => global_abs_max / 7.0,
                _ => 1.0,
            };

            QuantizationParams {
                qtype,
                scheme,
                granularity: QuantizationGranularity::PerTensor,
                scale: vec![scale],
                zero_point: vec![0],
                min_val: vec![-global_abs_max],
                max_val: vec![global_abs_max],
            }
        }
        QuantizationScheme::Asymmetric => {
            let (scale, zero_point) = match qtype {
                QuantizationType::Int8 => {
                    let scale = (global_max - global_min) / 255.0;
                    let zero_point = (-global_min / scale).round() as i32;
                    (scale, zero_point)
                }
                QuantizationType::Int4 => {
                    let scale = (global_max - global_min) / 15.0;
                    let zero_point = (-global_min / scale).round() as i32;
                    (scale, zero_point)
                }
                _ => (1.0, 0),
            };

            QuantizationParams {
                qtype,
                scheme,
                granularity: QuantizationGranularity::PerTensor,
                scale: vec![scale],
                zero_point: vec![zero_point],
                min_val: vec![global_min],
                max_val: vec![global_max],
            }
        }
    };

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    #[test]
    fn test_quantization_type_properties() {
        assert_eq!(QuantizationType::Int8.bits(), 8);
        assert_eq!(QuantizationType::Int4.bits(), 4);
        assert_eq!(QuantizationType::Fp16.bits(), 16);
        assert_eq!(QuantizationType::BFloat16.bits(), 16);

        assert_eq!(QuantizationType::Int8.compression_ratio(), 8.0);
        assert_eq!(QuantizationType::Int4.compression_ratio(), 16.0);

        assert!(QuantizationType::Int8.is_integer());
        assert!(QuantizationType::Fp16.is_float());
    }

    #[test]
    fn test_symmetric_quantization_int8() {
        let data = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        let tensor = ArrayD::from_shape_vec(vec![5], data.clone()).unwrap();

        let params = QuantizationParams::symmetric_per_tensor(QuantizationType::Int8, &tensor);

        assert_eq!(params.scheme, QuantizationScheme::Symmetric);
        assert_eq!(params.zero_point[0], 0);
        assert!(params.scale[0] > 0.0);
    }

    #[test]
    fn test_asymmetric_quantization_int8() {
        let data = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let tensor = ArrayD::from_shape_vec(vec![5], data).unwrap();

        let params = QuantizationParams::asymmetric_per_tensor(QuantizationType::Int8, &tensor);

        assert_eq!(params.scheme, QuantizationScheme::Asymmetric);
        assert!(params.zero_point[0] >= 0);
        assert!(params.scale[0] > 0.0);
    }

    #[test]
    fn test_quantize_dequantize_int8() {
        let data = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        let tensor = ArrayD::from_shape_vec(vec![5], data.clone()).unwrap();

        let params = QuantizationParams::symmetric_per_tensor(QuantizationType::Int8, &tensor);
        let quantized = QuantizedTensor::quantize(&tensor, params);
        let dequantized = quantized.dequantize();

        // Check that dequantized values are close to original
        for (orig, deq) in tensor.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.1,
                "Original: {}, Dequantized: {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_quantization_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = ArrayD::from_shape_vec(vec![5], data).unwrap();

        let params = QuantizationParams::symmetric_per_tensor(QuantizationType::Int8, &tensor);
        let quantized = QuantizedTensor::quantize(&tensor, params);

        let error = quantized.quantization_error(&tensor);
        assert!(error >= 0.0);
        assert!(error < 1.0); // Error should be small for this simple case
    }

    #[test]
    fn test_memory_reduction() {
        let tensor = ArrayD::from_shape_vec(vec![100], vec![1.0; 100]).unwrap();
        let params = QuantizationParams::symmetric_per_tensor(QuantizationType::Int8, &tensor);
        let quantized = QuantizedTensor::quantize(&tensor, params);

        assert_eq!(quantized.memory_reduction(), 8.0); // 64-bit to 8-bit = 8x compression
    }

    #[test]
    fn test_calibrate_quantization() {
        let sample1 = ArrayD::from_shape_vec(vec![3], vec![-10.0, 0.0, 10.0]).unwrap();
        let sample2 = ArrayD::from_shape_vec(vec![3], vec![-8.0, 2.0, 12.0]).unwrap();
        let samples = vec![sample1, sample2];

        let params = calibrate_quantization(
            &samples,
            QuantizationType::Int8,
            QuantizationScheme::Symmetric,
        )
        .unwrap();

        assert!(params.scale[0] > 0.0);
        assert_eq!(params.zero_point[0], 0); // Symmetric
    }

    #[test]
    fn test_quantization_stats() {
        let mut stats = QuantizationStats::new();

        stats.update(1000, 8.0, 0.01);
        stats.update(2000, 8.0, 0.02);

        assert_eq!(stats.num_tensors, 2);
        assert!(stats.memory_saved > 0);
        assert!(stats.avg_error > 0.0);
        assert_eq!(stats.max_error, 0.02);
    }

    #[test]
    fn test_fp16_quantization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = ArrayD::from_shape_vec(vec![5], data.clone()).unwrap();

        let quantized = quantize_fp16(&tensor);

        // FP16 should preserve values reasonably well for small numbers
        for (orig, quant) in tensor.iter().zip(quantized.iter()) {
            assert!((orig - quant).abs() < 0.001);
        }
    }

    #[test]
    fn test_qat_config_default() {
        let config = QatConfig::default();

        assert_eq!(config.target_qtype, QuantizationType::Int8);
        assert_eq!(config.scheme, QuantizationScheme::Symmetric);
        assert!(config.use_ste);
    }
}
