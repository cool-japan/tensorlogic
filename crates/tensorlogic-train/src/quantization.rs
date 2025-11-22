//! Model quantization utilities for compression and acceleration.
//!
//! This module provides quantization techniques to reduce model size and improve inference speed:
//! - Post-training quantization (PTQ) for immediate deployment
//! - Quantization-aware training (QAT) for better accuracy
//! - Multiple bit-width support (int8, int4, int2)
//! - Per-tensor and per-channel quantization
//!
//! # Examples
//!
//! ```
//! use tensorlogic_train::{QuantizationConfig, Quantizer, QuantizationMode};
//! use scirs2_core::ndarray::Array2;
//!
//! // Post-training quantization (PTQ)
//! let weights = Array2::<f32>::zeros((10, 10));
//! let config = QuantizationConfig::int8_symmetric();
//! let quantized = Quantizer::quantize_tensor(&weights.view(), &config);
//!
//! // Dequantize for inference
//! let dequantized = Quantizer::dequantize_tensor(&quantized);
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantization mode determines the quantization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// Symmetric quantization: range is [-max, max]
    Symmetric,
    /// Asymmetric quantization: range is [min, max]
    Asymmetric,
}

/// Bit-width for quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitWidth {
    /// 8-bit integer quantization (most common)
    Int8,
    /// 4-bit integer quantization (higher compression)
    Int4,
    /// 2-bit integer quantization (extreme compression)
    Int2,
}

impl BitWidth {
    /// Returns the number of quantization levels for this bit-width.
    pub fn levels(&self) -> i32 {
        match self {
            BitWidth::Int8 => 256, // 2^8
            BitWidth::Int4 => 16,  // 2^4
            BitWidth::Int2 => 4,   // 2^2
        }
    }

    /// Returns the minimum quantized value.
    pub fn qmin(&self) -> i32 {
        match self {
            BitWidth::Int8 => -128,
            BitWidth::Int4 => -8,
            BitWidth::Int2 => -2,
        }
    }

    /// Returns the maximum quantized value.
    pub fn qmax(&self) -> i32 {
        match self {
            BitWidth::Int8 => 127,
            BitWidth::Int4 => 7,
            BitWidth::Int2 => 1,
        }
    }
}

/// Quantization granularity (per-tensor or per-channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Granularity {
    /// Single scale/zero-point for entire tensor
    PerTensor,
    /// Separate scale/zero-point per output channel (axis 0)
    PerChannel,
}

/// Configuration for quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization mode (symmetric or asymmetric)
    pub mode: QuantizationMode,
    /// Bit-width for quantization
    pub bit_width: BitWidth,
    /// Granularity (per-tensor or per-channel)
    pub granularity: Granularity,
    /// Small epsilon to avoid division by zero
    pub eps: f32,
}

impl QuantizationConfig {
    /// Creates a default int8 symmetric per-tensor configuration.
    pub fn int8_symmetric() -> Self {
        Self {
            mode: QuantizationMode::Symmetric,
            bit_width: BitWidth::Int8,
            granularity: Granularity::PerTensor,
            eps: 1e-8,
        }
    }

    /// Creates a default int8 asymmetric per-tensor configuration.
    pub fn int8_asymmetric() -> Self {
        Self {
            mode: QuantizationMode::Asymmetric,
            bit_width: BitWidth::Int8,
            granularity: Granularity::PerTensor,
            eps: 1e-8,
        }
    }

    /// Creates a default int4 symmetric per-channel configuration.
    pub fn int4_per_channel() -> Self {
        Self {
            mode: QuantizationMode::Symmetric,
            bit_width: BitWidth::Int4,
            granularity: Granularity::PerChannel,
            eps: 1e-8,
        }
    }

    /// Creates a custom configuration.
    pub fn new(mode: QuantizationMode, bit_width: BitWidth, granularity: Granularity) -> Self {
        Self {
            mode,
            bit_width,
            granularity,
            eps: 1e-8,
        }
    }
}

/// Quantization parameters (scale and zero-point).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor(s) for quantization
    pub scale: Array1<f32>,
    /// Zero-point(s) for asymmetric quantization
    pub zero_point: Array1<i32>,
    /// Original configuration used
    pub config: QuantizationConfig,
}

/// Quantized tensor representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// Quantized integer values
    pub data: Array2<i8>,
    /// Quantization parameters
    pub params: QuantizationParams,
}

/// Main quantizer for model compression.
pub struct Quantizer;

impl Quantizer {
    /// Quantizes a 2D tensor using the specified configuration.
    ///
    /// # Arguments
    /// * `tensor` - Input floating-point tensor
    /// * `config` - Quantization configuration
    ///
    /// # Returns
    /// Quantized tensor with parameters
    pub fn quantize_tensor(
        tensor: &ArrayView2<f32>,
        config: &QuantizationConfig,
    ) -> QuantizedTensor {
        match config.granularity {
            Granularity::PerTensor => Self::quantize_per_tensor(tensor, config),
            Granularity::PerChannel => Self::quantize_per_channel(tensor, config),
        }
    }

    /// Per-tensor quantization (single scale/zero-point).
    fn quantize_per_tensor(
        tensor: &ArrayView2<f32>,
        config: &QuantizationConfig,
    ) -> QuantizedTensor {
        let (scale, zero_point) = Self::compute_params_tensor(tensor, config);

        let quantized = tensor.mapv(|x| {
            let q = (x / scale).round() + zero_point as f32;
            Self::clamp_to_qrange(q as i32, config.bit_width) as i8
        });

        QuantizedTensor {
            data: quantized,
            params: QuantizationParams {
                scale: Array1::from_vec(vec![scale]),
                zero_point: Array1::from_vec(vec![zero_point]),
                config: config.clone(),
            },
        }
    }

    /// Per-channel quantization (separate scale/zero-point per channel).
    fn quantize_per_channel(
        tensor: &ArrayView2<f32>,
        config: &QuantizationConfig,
    ) -> QuantizedTensor {
        let num_channels = tensor.shape()[0];
        let mut scales = Vec::with_capacity(num_channels);
        let mut zero_points = Vec::with_capacity(num_channels);

        // Compute parameters per channel
        for i in 0..num_channels {
            let channel = tensor.index_axis(Axis(0), i);
            let (scale, zero_point) = Self::compute_params_channel(&channel, config);
            scales.push(scale);
            zero_points.push(zero_point);
        }

        // Quantize each channel
        let mut quantized = Array2::<i8>::zeros(tensor.dim());
        for (i, mut row) in quantized.axis_iter_mut(Axis(0)).enumerate() {
            let channel = tensor.index_axis(Axis(0), i);
            let scale = scales[i];
            let zero_point = zero_points[i];

            for (j, &val) in channel.iter().enumerate() {
                let q = (val / scale).round() + zero_point as f32;
                row[j] = Self::clamp_to_qrange(q as i32, config.bit_width) as i8;
            }
        }

        QuantizedTensor {
            data: quantized,
            params: QuantizationParams {
                scale: Array1::from_vec(scales),
                zero_point: Array1::from_vec(zero_points),
                config: config.clone(),
            },
        }
    }

    /// Computes quantization parameters for entire tensor.
    fn compute_params_tensor(tensor: &ArrayView2<f32>, config: &QuantizationConfig) -> (f32, i32) {
        let min = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        Self::compute_scale_zero_point(min, max, config)
    }

    /// Computes quantization parameters for a single channel.
    fn compute_params_channel(
        channel: &ArrayView1<f32>,
        config: &QuantizationConfig,
    ) -> (f32, i32) {
        let min = channel.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = channel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        Self::compute_scale_zero_point(min, max, config)
    }

    /// Computes scale and zero-point from min/max values.
    fn compute_scale_zero_point(min: f32, max: f32, config: &QuantizationConfig) -> (f32, i32) {
        let qmin = config.bit_width.qmin() as f32;
        let qmax = config.bit_width.qmax() as f32;

        match config.mode {
            QuantizationMode::Symmetric => {
                let abs_max = min.abs().max(max.abs());
                let scale = (2.0 * abs_max / (qmax - qmin)).max(config.eps);
                (scale, 0)
            }
            QuantizationMode::Asymmetric => {
                let scale = ((max - min) / (qmax - qmin)).max(config.eps);
                let zero_point = (qmin - min / scale).round() as i32;
                let zero_point = Self::clamp_to_qrange(zero_point, config.bit_width);
                (scale, zero_point)
            }
        }
    }

    /// Clamps a value to the quantization range.
    fn clamp_to_qrange(value: i32, bit_width: BitWidth) -> i32 {
        value.max(bit_width.qmin()).min(bit_width.qmax())
    }

    /// Dequantizes a quantized tensor back to float32.
    ///
    /// # Arguments
    /// * `quantized` - Quantized tensor with parameters
    ///
    /// # Returns
    /// Dequantized floating-point tensor
    pub fn dequantize_tensor(quantized: &QuantizedTensor) -> Array2<f32> {
        match quantized.params.config.granularity {
            Granularity::PerTensor => {
                let scale = quantized.params.scale[0];
                let zero_point = quantized.params.zero_point[0];
                quantized
                    .data
                    .mapv(|q| scale * (q as f32 - zero_point as f32))
            }
            Granularity::PerChannel => {
                let mut result = Array2::<f32>::zeros(quantized.data.dim());
                for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
                    let scale = quantized.params.scale[i];
                    let zero_point = quantized.params.zero_point[i];
                    let q_row = quantized.data.index_axis(Axis(0), i);

                    for (j, &q) in q_row.iter().enumerate() {
                        row[j] = scale * (q as f32 - zero_point as f32);
                    }
                }
                result
            }
        }
    }

    /// Computes the compression ratio achieved by quantization.
    pub fn compression_ratio(config: &QuantizationConfig) -> f32 {
        let original_bits = 32.0; // f32
        let quantized_bits = match config.bit_width {
            BitWidth::Int8 => 8.0,
            BitWidth::Int4 => 4.0,
            BitWidth::Int2 => 2.0,
        };
        original_bits / quantized_bits
    }

    /// Estimates the quantization error (MSE) for a tensor.
    pub fn quantization_error(original: &ArrayView2<f32>, quantized: &QuantizedTensor) -> f32 {
        let dequantized = Self::dequantize_tensor(quantized);
        let diff = original - &dequantized.view();
        diff.mapv(|x| x * x).mean().unwrap_or(0.0)
    }
}

/// Quantization-aware training (QAT) utilities.
pub struct QuantizationAwareTraining {
    /// Layer name to quantization config mapping
    layer_configs: HashMap<String, QuantizationConfig>,
    /// Whether to simulate quantization during training
    simulate_quantization: bool,
}

impl QuantizationAwareTraining {
    /// Creates a new QAT instance.
    pub fn new(simulate_quantization: bool) -> Self {
        Self {
            layer_configs: HashMap::new(),
            simulate_quantization,
        }
    }

    /// Registers a layer for quantization-aware training.
    pub fn register_layer(&mut self, layer_name: String, config: QuantizationConfig) {
        self.layer_configs.insert(layer_name, config);
    }

    /// Simulates quantization during forward pass (straight-through estimator).
    ///
    /// This applies fake quantization: quantize then immediately dequantize,
    /// allowing gradients to flow through.
    pub fn fake_quantize(&self, tensor: &Array2<f32>, layer_name: &str) -> Array2<f32> {
        if !self.simulate_quantization {
            return tensor.clone();
        }

        if let Some(config) = self.layer_configs.get(layer_name) {
            let quantized = Quantizer::quantize_tensor(&tensor.view(), config);
            Quantizer::dequantize_tensor(&quantized)
        } else {
            tensor.clone()
        }
    }

    /// Gets the quantization config for a layer.
    pub fn get_config(&self, layer_name: &str) -> Option<&QuantizationConfig> {
        self.layer_configs.get(layer_name)
    }

    /// Returns all registered layer names.
    pub fn registered_layers(&self) -> Vec<&String> {
        self.layer_configs.keys().collect()
    }
}

/// Dynamic range calibration for post-training quantization.
pub struct DynamicRangeCalibrator {
    /// Collected min/max statistics per layer
    statistics: HashMap<String, (f32, f32)>,
    /// Number of samples collected
    num_samples: usize,
}

impl DynamicRangeCalibrator {
    /// Creates a new calibrator.
    pub fn new() -> Self {
        Self {
            statistics: HashMap::new(),
            num_samples: 0,
        }
    }

    /// Collects statistics from a batch of activations.
    pub fn collect(&mut self, layer_name: String, tensor: &ArrayView2<f32>) {
        let min = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        self.statistics
            .entry(layer_name)
            .and_modify(|(prev_min, prev_max)| {
                *prev_min = prev_min.min(min);
                *prev_max = prev_max.max(max);
            })
            .or_insert((min, max));

        self.num_samples += 1;
    }

    /// Finalizes calibration and returns quantization configs.
    pub fn finalize(
        &self,
        default_config: &QuantizationConfig,
    ) -> HashMap<String, QuantizationConfig> {
        self.statistics
            .keys()
            .map(|name| (name.clone(), default_config.clone()))
            .collect()
    }

    /// Gets the collected range for a layer.
    pub fn get_range(&self, layer_name: &str) -> Option<(f32, f32)> {
        self.statistics.get(layer_name).copied()
    }

    /// Resets all collected statistics.
    pub fn reset(&mut self) {
        self.statistics.clear();
        self.num_samples = 0;
    }
}

impl Default for DynamicRangeCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_int8_symmetric_quantization() {
        let tensor = Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, -2.0, 2.0, 0.5]).unwrap();
        let config = QuantizationConfig::int8_symmetric();

        let quantized = Quantizer::quantize_tensor(&tensor.view(), &config);
        let dequantized = Quantizer::dequantize_tensor(&quantized);

        // Check shape preserved
        assert_eq!(dequantized.dim(), tensor.dim());

        // Check approximate reconstruction
        for (orig, deq) in tensor.iter().zip(dequantized.iter()) {
            assert_relative_eq!(orig, deq, epsilon = 0.1);
        }
    }

    #[test]
    fn test_int8_asymmetric_quantization() {
        let tensor = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let config = QuantizationConfig::int8_asymmetric();

        let quantized = Quantizer::quantize_tensor(&tensor.view(), &config);
        assert_eq!(quantized.params.config.mode, QuantizationMode::Asymmetric);

        let dequantized = Quantizer::dequantize_tensor(&quantized);
        assert_relative_eq!(dequantized[[0, 0]], 0.0, epsilon = 0.05);
        assert_relative_eq!(dequantized[[1, 1]], 3.0, epsilon = 0.05);
    }

    #[test]
    fn test_int4_per_channel_quantization() {
        let tensor =
            Array2::from_shape_vec((2, 4), vec![-1.0, 0.0, 1.0, 2.0, -10.0, -5.0, 5.0, 10.0])
                .unwrap();
        let config = QuantizationConfig::int4_per_channel();

        let quantized = Quantizer::quantize_tensor(&tensor.view(), &config);

        // Should have 2 scales (one per channel)
        assert_eq!(quantized.params.scale.len(), 2);
        assert_eq!(quantized.params.zero_point.len(), 2);

        let dequantized = Quantizer::dequantize_tensor(&quantized);
        assert_eq!(dequantized.dim(), tensor.dim());
    }

    #[test]
    fn test_bit_width_levels() {
        assert_eq!(BitWidth::Int8.levels(), 256);
        assert_eq!(BitWidth::Int4.levels(), 16);
        assert_eq!(BitWidth::Int2.levels(), 4);
    }

    #[test]
    fn test_bit_width_ranges() {
        assert_eq!(BitWidth::Int8.qmin(), -128);
        assert_eq!(BitWidth::Int8.qmax(), 127);
        assert_eq!(BitWidth::Int4.qmin(), -8);
        assert_eq!(BitWidth::Int4.qmax(), 7);
    }

    #[test]
    fn test_compression_ratio() {
        let config_int8 = QuantizationConfig::int8_symmetric();
        assert_eq!(Quantizer::compression_ratio(&config_int8), 4.0);

        let config_int4 = QuantizationConfig::new(
            QuantizationMode::Symmetric,
            BitWidth::Int4,
            Granularity::PerTensor,
        );
        assert_eq!(Quantizer::compression_ratio(&config_int4), 8.0);
    }

    #[test]
    fn test_quantization_error() {
        let tensor = Array2::from_shape_vec((3, 3), vec![1.0; 9]).unwrap();
        let config = QuantizationConfig::int8_symmetric();

        let quantized = Quantizer::quantize_tensor(&tensor.view(), &config);
        let error = Quantizer::quantization_error(&tensor.view(), &quantized);

        // Error should be small for uniform values
        assert!(error < 0.01);
    }

    #[test]
    fn test_qat_registration() {
        let mut qat = QuantizationAwareTraining::new(true);
        qat.register_layer("layer1".to_string(), QuantizationConfig::int8_symmetric());
        qat.register_layer("layer2".to_string(), QuantizationConfig::int4_per_channel());

        assert_eq!(qat.registered_layers().len(), 2);
        assert!(qat.get_config("layer1").is_some());
        assert!(qat.get_config("layer3").is_none());
    }

    #[test]
    fn test_fake_quantization() {
        let mut qat = QuantizationAwareTraining::new(true);
        qat.register_layer("fc1".to_string(), QuantizationConfig::int8_symmetric());

        let tensor = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let fake_quantized = qat.fake_quantize(&tensor, "fc1");

        // Should be similar but not identical due to quantization
        assert_eq!(fake_quantized.dim(), tensor.dim());
    }

    #[test]
    fn test_dynamic_range_calibrator() {
        let mut calibrator = DynamicRangeCalibrator::new();

        let tensor1 = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let tensor2 = Array2::from_shape_vec((2, 2), vec![-1.0, 0.0, 1.0, 4.0]).unwrap();

        calibrator.collect("layer1".to_string(), &tensor1.view());
        calibrator.collect("layer1".to_string(), &tensor2.view());

        let (min, max) = calibrator.get_range("layer1").unwrap();
        assert_eq!(min, -1.0);
        assert_eq!(max, 4.0);
    }

    #[test]
    fn test_calibrator_finalize() {
        let mut calibrator = DynamicRangeCalibrator::new();
        let tensor = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();

        calibrator.collect("layer1".to_string(), &tensor.view());
        calibrator.collect("layer2".to_string(), &tensor.view());

        let config = QuantizationConfig::int8_symmetric();
        let configs = calibrator.finalize(&config);

        assert_eq!(configs.len(), 2);
        assert!(configs.contains_key("layer1"));
        assert!(configs.contains_key("layer2"));
    }

    #[test]
    fn test_calibrator_reset() {
        let mut calibrator = DynamicRangeCalibrator::new();
        let tensor = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();

        calibrator.collect("layer1".to_string(), &tensor.view());
        assert_eq!(calibrator.num_samples, 1);

        calibrator.reset();
        assert_eq!(calibrator.num_samples, 0);
        assert!(calibrator.get_range("layer1").is_none());
    }

    #[test]
    fn test_zero_tensor_quantization() {
        let tensor = Array2::zeros((3, 3));
        let config = QuantizationConfig::int8_symmetric();

        let quantized = Quantizer::quantize_tensor(&tensor.view(), &config);
        let dequantized = Quantizer::dequantize_tensor(&quantized);

        assert_eq!(dequantized, tensor);
    }

    #[test]
    fn test_extreme_values_quantization() {
        let tensor = Array2::from_shape_vec(
            (2, 2),
            vec![f32::MIN / 1e6, f32::MAX / 1e6, -1000.0, 1000.0],
        )
        .unwrap();
        let config = QuantizationConfig::int8_symmetric();

        let quantized = Quantizer::quantize_tensor(&tensor.view(), &config);
        let dequantized = Quantizer::dequantize_tensor(&quantized);

        // Should handle extreme values without panicking
        assert_eq!(dequantized.dim(), tensor.dim());
    }
}
