//! Parser for criterion benchmark output

use crate::models::{BenchmarkResult, BenchmarkSuite, CriterionEstimates};
use anyhow::{Context, Result};
use chrono::Utc;
use std::fs;
use std::path::Path;

/// Parse all benchmarks from criterion output directory
pub fn parse_criterion_output(criterion_dir: &Path) -> Result<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    // Iterate through all benchmark groups
    for entry in fs::read_dir(criterion_dir)
        .with_context(|| format!("Failed to read criterion directory: {:?}", criterion_dir))?
    {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        // Skip report directory
        if path.file_name().and_then(|n| n.to_str()) == Some("report") {
            continue;
        }

        let benchmark_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Check if this is a parameterized benchmark
        if let Ok(param_entries) = fs::read_dir(&path) {
            let mut has_params = false;

            for param_entry in param_entries {
                let param_entry = param_entry?;
                let param_path = param_entry.path();

                if !param_path.is_dir() {
                    continue;
                }

                let param_name = param_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Try to parse estimates
                if let Some(result) =
                    parse_benchmark_estimates(&benchmark_name, Some(&param_name), &param_path)?
                {
                    results.push(result);
                    has_params = true;
                }
            }

            // If no parameters found, try to parse the benchmark directly
            if !has_params {
                if let Some(result) = parse_benchmark_estimates(&benchmark_name, None, &path)? {
                    results.push(result);
                }
            }
        }
    }

    Ok(results)
}

/// Parse estimates for a single benchmark
fn parse_benchmark_estimates(
    benchmark_name: &str,
    parameter: Option<&str>,
    benchmark_path: &Path,
) -> Result<Option<BenchmarkResult>> {
    let estimates_path = benchmark_path.join("new/estimates.json");

    if !estimates_path.exists() {
        // Try base estimates if new doesn't exist
        let base_estimates_path = benchmark_path.join("base/estimates.json");
        if !base_estimates_path.exists() {
            return Ok(None);
        }
        return parse_estimates_file(&base_estimates_path, benchmark_name, parameter);
    }

    parse_estimates_file(&estimates_path, benchmark_name, parameter)
}

fn parse_estimates_file(
    path: &Path,
    benchmark_name: &str,
    parameter: Option<&str>,
) -> Result<Option<BenchmarkResult>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read estimates file: {:?}", path))?;

    let estimates: CriterionEstimates = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse estimates JSON: {:?}", path))?;

    Ok(Some(BenchmarkResult {
        name: benchmark_name.to_string(),
        parameter: parameter.map(|s| s.to_string()),
        estimates,
        timestamp: Utc::now(),
    }))
}

/// Parse baseline file
pub fn parse_baseline(path: &Path) -> Result<BenchmarkSuite> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read baseline file: {:?}", path))?;

    let suite: BenchmarkSuite = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse baseline JSON: {:?}", path))?;

    Ok(suite)
}

/// Save benchmark suite to file
pub fn save_suite(suite: &BenchmarkSuite, path: &Path) -> Result<()> {
    // Create parent directory if it doesn't exist
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {:?}", parent))?;
    }

    let content = serde_json::to_string_pretty(suite)
        .with_context(|| "Failed to serialize benchmark suite")?;

    fs::write(path, content)
        .with_context(|| format!("Failed to write baseline file: {:?}", path))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_parse_estimates_file() {
        let test_json = r#"{
            "mean": {
                "point_estimate": 113.48351458953826,
                "standard_error": 1.470715141259493,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 110.96928564380244,
                    "upper_bound": 116.6974108016696
                }
            },
            "median": {
                "point_estimate": 109.21776304688808,
                "standard_error": 0.2281010629762013,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 108.72222222222223,
                    "upper_bound": 109.77119818035072
                }
            },
            "median_abs_dev": {
                "point_estimate": 2.2009518863019446,
                "standard_error": 0.33367693241762536,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 1.554802408527372,
                    "upper_bound": 2.9213887983836933
                }
            },
            "slope": {
                "point_estimate": 113.70123216673805,
                "standard_error": 1.4803311717294008,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 111.0567713753329,
                    "upper_bound": 116.81901327923268
                }
            },
            "std_dev": {
                "point_estimate": 14.770132527845913,
                "standard_error": 4.0937669066701945,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 6.569289856790614,
                    "upper_bound": 21.98764320967657
                }
            }
        }"#;

        let temp_dir = env::temp_dir();
        let test_file = temp_dir.join("test_estimates.json");
        fs::write(&test_file, test_json).expect("Failed to write test file");

        let result =
            parse_estimates_file(&test_file, "test_bench", Some("10")).expect("Failed to parse");

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.name, "test_bench");
        assert_eq!(result.parameter, Some("10".to_string()));
        assert!((result.estimates.mean.point_estimate - 113.48351458953826).abs() < 1e-10);

        fs::remove_file(&test_file).ok();
    }
}
