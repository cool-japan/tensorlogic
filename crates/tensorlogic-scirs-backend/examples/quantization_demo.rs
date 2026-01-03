//! Quantization Demonstration
//!
//! This example demonstrates tensor quantization for memory efficiency:
//!
//! 1. INT8 symmetric and asymmetric quantization
//! 2. FP16 and BFloat16 quantization
//! 3. Quantization calibration with multiple samples
//! 4. Memory savings analysis
//! 5. Quantization error measurement

use scirs2_core::ndarray::ArrayD;
use tensorlogic_scirs_backend::{
    calibrate_quantization, QuantizationParams, QuantizationScheme, QuantizationType,
    QuantizedTensor,
};

fn main() {
    println!("=== TensorLogic Quantization Demonstration ===\n");

    // Create sample tensors
    let data: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.1).collect();
    let tensor = ArrayD::from_shape_vec(vec![100], data).unwrap();

    println!("1. Original Tensor");
    println!("   --------------");
    println!("   Shape: {:?}", tensor.shape());
    println!("   Size: {} elements", tensor.len());
    println!("   Memory: {} bytes (FP64)", tensor.len() * 8);
    println!(
        "   Range: [{:.2}, {:.2}]\n",
        tensor.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        tensor.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // INT8 Symmetric Quantization
    println!("2. INT8 Symmetric Quantization");
    println!("   ---------------------------");
    let params_int8_sym = QuantizationParams::symmetric_per_tensor(QuantizationType::Int8, &tensor);
    let quantized_int8_sym = QuantizedTensor::quantize(&tensor, params_int8_sym);

    println!("   Scale: {:.6}", quantized_int8_sym.params.scale[0]);
    println!("   Zero Point: {}", quantized_int8_sym.params.zero_point[0]);
    println!("   Memory: {} bytes", tensor.len());
    println!(
        "   Compression: {:.1}x",
        quantized_int8_sym.memory_reduction()
    );

    let error_int8_sym = quantized_int8_sym.quantization_error(&tensor);
    println!("   Quantization Error (MSE): {:.6}", error_int8_sym);

    let dequantized = quantized_int8_sym.dequantize();
    println!(
        "   Max Absolute Error: {:.6}\n",
        (&tensor - &dequantized)
            .iter()
            .map(|&x| x.abs())
            .fold(0.0, f64::max)
    );

    // INT8 Asymmetric Quantization
    println!("3. INT8 Asymmetric Quantization");
    println!("   ----------------------------");
    let params_int8_asym =
        QuantizationParams::asymmetric_per_tensor(QuantizationType::Int8, &tensor);
    let quantized_int8_asym = QuantizedTensor::quantize(&tensor, params_int8_asym);

    println!("   Scale: {:.6}", quantized_int8_asym.params.scale[0]);
    println!(
        "   Zero Point: {}",
        quantized_int8_asym.params.zero_point[0]
    );
    println!("   Memory: {} bytes", tensor.len());
    println!(
        "   Compression: {:.1}x",
        quantized_int8_asym.memory_reduction()
    );

    let error_int8_asym = quantized_int8_asym.quantization_error(&tensor);
    println!("   Quantization Error (MSE): {:.6}\n", error_int8_asym);

    // INT4 Quantization
    println!("4. INT4 Quantization (Ultra-Compressed)");
    println!("   ------------------------------------");
    let params_int4 = QuantizationParams::symmetric_per_tensor(QuantizationType::Int4, &tensor);
    let quantized_int4 = QuantizedTensor::quantize(&tensor, params_int4);

    println!("   Scale: {:.6}", quantized_int4.params.scale[0]);
    println!("   Compression: {:.1}x", quantized_int4.memory_reduction());

    let error_int4 = quantized_int4.quantization_error(&tensor);
    println!("   Quantization Error (MSE): {:.6}", error_int4);
    println!("   Note: Higher error due to extreme compression\n");

    // FP16 Quantization
    println!("5. FP16 Quantization");
    println!("   -----------------");
    let params_fp16 = QuantizationParams::symmetric_per_tensor(QuantizationType::Fp16, &tensor);
    let quantized_fp16 = QuantizedTensor::quantize(&tensor, params_fp16);

    println!("   Compression: {:.1}x", quantized_fp16.memory_reduction());
    let error_fp16 = quantized_fp16.quantization_error(&tensor);
    println!("   Quantization Error (MSE): {:.10}", error_fp16);
    println!("   Note: Much lower error than integer quantization\n");

    // BFloat16 Quantization
    println!("6. BFloat16 Quantization");
    println!("   ---------------------");
    let params_bf16 = QuantizationParams::symmetric_per_tensor(QuantizationType::BFloat16, &tensor);
    let quantized_bf16 = QuantizedTensor::quantize(&tensor, params_bf16);

    println!("   Compression: {:.1}x", quantized_bf16.memory_reduction());
    let error_bf16 = quantized_bf16.quantization_error(&tensor);
    println!("   Quantization Error (MSE): {:.10}\n", error_bf16);

    // Calibration with multiple samples
    println!("7. Calibration with Multiple Samples");
    println!("   ---------------------------------");

    let sample1 = ArrayD::from_shape_vec(
        vec![50],
        (0..50).map(|i| (i as f64 - 25.0) * 0.05).collect(),
    )
    .unwrap();
    let sample2 = ArrayD::from_shape_vec(
        vec![50],
        (0..50).map(|i| (i as f64 - 20.0) * 0.08).collect(),
    )
    .unwrap();
    let sample3 = ArrayD::from_shape_vec(
        vec![50],
        (0..50).map(|i| (i as f64 - 30.0) * 0.06).collect(),
    )
    .unwrap();

    let samples = vec![sample1, sample2, sample3];

    let calibrated_params = calibrate_quantization(
        &samples,
        QuantizationType::Int8,
        QuantizationScheme::Symmetric,
    )
    .unwrap();

    println!("   Calibrated with {} samples", samples.len());
    println!("   Calibrated Scale: {:.6}", calibrated_params.scale[0]);
    println!(
        "   Calibrated Range: [{:.2}, {:.2}]",
        calibrated_params.min_val[0], calibrated_params.max_val[0]
    );
    println!("   Dynamic Range: {:.2}", calibrated_params.dynamic_range());
    println!(
        "   Error Bound: {:.6}\n",
        calibrated_params.quantization_error_bound()
    );

    // Comparison Table
    println!("8. Quantization Methods Comparison");
    println!("   --------------------------------");
    println!("   Method          | Compression | Error (MSE)");
    println!("   ----------------|-------------|-------------");
    println!(
        "   INT8 Symmetric  | {:<11.1}x | {:.6}",
        quantized_int8_sym.memory_reduction(),
        error_int8_sym
    );
    println!(
        "   INT8 Asymmetric | {:<11.1}x | {:.6}",
        quantized_int8_asym.memory_reduction(),
        error_int8_asym
    );
    println!(
        "   INT4            | {:<11.1}x | {:.6}",
        quantized_int4.memory_reduction(),
        error_int4
    );
    println!(
        "   FP16            | {:<11.1}x | {:.10}",
        quantized_fp16.memory_reduction(),
        error_fp16
    );
    println!(
        "   BFloat16        | {:<11.1}x | {:.10}\n",
        quantized_bf16.memory_reduction(),
        error_bf16
    );

    // Memory savings for a large model
    println!("9. Practical Example: Large Model");
    println!("   -------------------------------");
    let model_size_gb = 10.0; // 10 GB model

    println!("   Original Model Size: {:.1} GB", model_size_gb);
    println!("\n   With INT8 Quantization:");
    println!("     Quantized Size: {:.2} GB", model_size_gb / 8.0);
    println!(
        "     Memory Saved: {:.2} GB ({:.0}%)",
        model_size_gb * (1.0 - 1.0 / 8.0),
        87.5
    );

    println!("\n   With INT4 Quantization:");
    println!("     Quantized Size: {:.2} GB", model_size_gb / 16.0);
    println!(
        "     Memory Saved: {:.2} GB ({:.0}%)",
        model_size_gb * (1.0 - 1.0 / 16.0),
        93.75
    );

    println!("\n   With FP16 Quantization:");
    println!("     Quantized Size: {:.2} GB", model_size_gb / 4.0);
    println!(
        "     Memory Saved: {:.2} GB ({:.0}%)",
        model_size_gb * (1.0 - 1.0 / 4.0),
        75.0
    );

    println!("\n=== End of Quantization Demonstration ===");
}
