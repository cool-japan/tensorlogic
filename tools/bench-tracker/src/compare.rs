//! Benchmark comparison logic

use crate::models::ComparisonResult;
use crate::parser;
use crate::report;
use anyhow::{Context, Result};
use std::path::Path;

/// Compare current benchmarks against baseline
pub fn compare_benchmarks(
    criterion_dir: &Path,
    baseline_path: &Path,
    threshold: f64,
    format: &str,
) -> Result<()> {
    // Parse current results
    let current_results = parser::parse_criterion_output(criterion_dir)?;
    if current_results.is_empty() {
        anyhow::bail!("No current benchmark results found in {:?}", criterion_dir);
    }

    // Load baseline
    let baseline = parser::parse_baseline(baseline_path)
        .with_context(|| format!("Failed to load baseline from {:?}", baseline_path))?;

    // Compare results
    let mut comparisons = Vec::new();
    let mut missing_baselines = Vec::new();
    let mut missing_current = Vec::new();

    for current in &current_results {
        if let Some(baseline_result) =
            baseline.get_result(&current.name, current.parameter.as_deref())
        {
            let comparison = ComparisonResult::new(baseline_result, current, threshold);
            comparisons.push(comparison);
        } else {
            missing_baselines.push(current.name.clone());
        }
    }

    // Find benchmarks in baseline but not in current
    for key in baseline.results.keys() {
        let parts: Vec<&str> = key.split('/').collect();
        let name = parts[0];
        let parameter = parts.get(1).copied();

        let found = current_results
            .iter()
            .any(|r| r.name == name && r.parameter.as_deref() == parameter);

        if !found {
            missing_current.push(key.clone());
        }
    }

    // Generate report
    match format {
        "json" => report::generate_json_report(&comparisons, &baseline)?,
        "html" => report::generate_html_report(&comparisons, &baseline)?,
        _ => report::generate_text_report(
            &comparisons,
            &baseline,
            threshold,
            &missing_baselines,
            &missing_current,
        )?,
    }

    // Exit with error if regressions found
    let has_regressions = comparisons.iter().any(|c| c.is_regression);
    if has_regressions {
        anyhow::bail!("Performance regressions detected!");
    }

    Ok(())
}
