//! Baseline management

use crate::models::BenchmarkSuite;
use crate::parser;
use anyhow::{Context, Result};
use colored::Colorize;
use std::path::Path;

/// Save current benchmark results as baseline
pub fn save_baseline(criterion_dir: &Path, output: &Path, name: Option<&str>) -> Result<()> {
    println!("Parsing benchmark results from {:?}...", criterion_dir);
    let results = parser::parse_criterion_output(criterion_dir)?;

    if results.is_empty() {
        anyhow::bail!("No benchmark results found in {:?}", criterion_dir);
    }

    let baseline_name = name.unwrap_or("baseline");
    let mut suite = BenchmarkSuite::new(baseline_name);

    for result in results {
        suite.add_result(result);
    }

    println!(
        "Saving {} benchmark results to {:?}...",
        suite.results.len(),
        output
    );

    parser::save_suite(&suite, output)?;

    println!("{}", "Baseline saved successfully!".green().bold());
    println!("  Name: {}", suite.name);
    println!("  Benchmarks: {}", suite.results.len());
    if let Some(commit) = &suite.commit {
        println!("  Commit: {}", commit);
    }
    println!("  Created: {}", suite.created_at);

    Ok(())
}

/// List all saved baselines
pub fn list_baselines(baseline_path: &Path) -> Result<()> {
    let suite = parser::parse_baseline(baseline_path)
        .with_context(|| format!("Failed to load baseline from {:?}", baseline_path))?;

    println!("{}", "Baseline Information:".bold());
    println!("  Name: {}", suite.name.cyan());
    println!("  Created: {}", suite.created_at);
    if let Some(commit) = &suite.commit {
        println!("  Commit: {}", commit);
    }
    println!();

    println!("{}", "Benchmarks:".bold());
    let mut results: Vec<_> = suite.results.values().collect();
    results.sort_by(|a, b| a.name.cmp(&b.name));

    for result in results {
        let full_name = match &result.parameter {
            Some(param) => format!("{}/{}", result.name, param),
            None => result.name.clone(),
        };
        let mean_ns = result.estimates.mean.point_estimate;
        println!("  {} - {:.2} ns", full_name, mean_ns);
    }

    println!();
    println!("Total: {} benchmarks", suite.results.len());

    Ok(())
}
