//! Report generation

use crate::models::{BenchmarkSuite, ComparisonResult};
use crate::parser;
use anyhow::Result;
use colored::Colorize;
use std::path::Path;
use tabled::{settings::Style, Table, Tabled};

#[derive(Tabled)]
struct ComparisonRow {
    #[tabled(rename = "Benchmark")]
    name: String,
    #[tabled(rename = "Baseline (ns)")]
    baseline: String,
    #[tabled(rename = "Current (ns)")]
    current: String,
    #[tabled(rename = "Change")]
    change: String,
    #[tabled(rename = "Status")]
    status: String,
}

/// Generate text report
pub fn generate_text_report(
    comparisons: &[ComparisonResult],
    baseline: &BenchmarkSuite,
    threshold: f64,
    missing_baselines: &[String],
    missing_current: &[String],
) -> Result<()> {
    println!("{}", "=== Benchmark Comparison Report ===".bold());
    println!();
    println!(
        "Baseline: {} ({})",
        baseline.name.cyan(),
        baseline.created_at
    );
    if let Some(commit) = &baseline.commit {
        println!("Baseline commit: {}", commit);
    }
    println!("Regression threshold: {}%", threshold);
    println!();

    if comparisons.is_empty() {
        println!("{}", "No benchmarks to compare!".yellow());
        return Ok(());
    }

    // Prepare table data
    let mut rows = Vec::new();
    let mut regressions = 0;
    let mut improvements = 0;
    let mut stable = 0;

    for comp in comparisons {
        let full_name = match &comp.parameter {
            Some(param) => format!("{}/{}", comp.name, param),
            None => comp.name.clone(),
        };

        let change_str = if comp.change_percent >= 0.0 {
            format!("+{:.2}%", comp.change_percent)
        } else {
            format!("{:.2}%", comp.change_percent)
        };

        let (status, colored_change) = if comp.is_regression {
            regressions += 1;
            ("REGRESSION".red().to_string(), change_str.red().to_string())
        } else if comp.is_improvement {
            improvements += 1;
            (
                "IMPROVEMENT".green().to_string(),
                change_str.green().to_string(),
            )
        } else {
            stable += 1;
            ("STABLE".blue().to_string(), change_str.blue().to_string())
        };

        rows.push(ComparisonRow {
            name: full_name,
            baseline: format!("{:.2}", comp.baseline_mean),
            current: format!("{:.2}", comp.current_mean),
            change: colored_change,
            status,
        });
    }

    // Sort by status (regressions first) then by name
    rows.sort_by(|a, b| {
        let a_priority = if a.status.contains("REGRESSION") {
            0
        } else if a.status.contains("IMPROVEMENT") {
            2
        } else {
            1
        };
        let b_priority = if b.status.contains("REGRESSION") {
            0
        } else if b.status.contains("IMPROVEMENT") {
            2
        } else {
            1
        };
        a_priority
            .cmp(&b_priority)
            .then_with(|| a.name.cmp(&b.name))
    });

    let table = Table::new(rows).with(Style::modern()).to_string();
    println!("{}", table);
    println!();

    // Summary
    println!("{}", "Summary:".bold());
    println!(
        "  {} Regressions",
        if regressions > 0 {
            format!("{}", regressions).red()
        } else {
            format!("{}", regressions).normal()
        }
    );
    println!(
        "  {} Improvements",
        if improvements > 0 {
            format!("{}", improvements).green()
        } else {
            format!("{}", improvements).normal()
        }
    );
    println!("  {} Stable", stable);
    println!("  {} Total", comparisons.len());
    println!();

    // Missing benchmarks
    if !missing_baselines.is_empty() {
        println!("{}", "New benchmarks (not in baseline):".yellow());
        for name in missing_baselines {
            println!("  - {}", name);
        }
        println!();
    }

    if !missing_current.is_empty() {
        println!(
            "{}",
            "Missing benchmarks (in baseline but not current):".red()
        );
        for name in missing_current {
            println!("  - {}", name);
        }
        println!();
    }

    Ok(())
}

/// Generate JSON report
pub fn generate_json_report(
    comparisons: &[ComparisonResult],
    baseline: &BenchmarkSuite,
) -> Result<()> {
    use serde_json::json;

    let report = json!({
        "baseline": {
            "name": baseline.name,
            "created_at": baseline.created_at,
            "commit": baseline.commit,
        },
        "comparisons": comparisons.iter().map(|c| {
            json!({
                "name": c.name,
                "parameter": c.parameter,
                "baseline_mean_ns": c.baseline_mean,
                "current_mean_ns": c.current_mean,
                "change_percent": c.change_percent,
                "is_regression": c.is_regression,
                "is_improvement": c.is_improvement,
            })
        }).collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Generate HTML report
pub fn generate_html_report(
    comparisons: &[ComparisonResult],
    baseline: &BenchmarkSuite,
) -> Result<()> {
    let mut html = String::from(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .regression { color: red; font-weight: bold; }
        .improvement { color: green; font-weight: bold; }
        .stable { color: blue; }
        .summary { margin-top: 20px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Benchmark Comparison Report</h1>
"#,
    );

    html.push_str(&format!(
        "<p><strong>Baseline:</strong> {} ({})</p>",
        baseline.name, baseline.created_at
    ));
    if let Some(commit) = &baseline.commit {
        html.push_str(&format!("<p><strong>Commit:</strong> {}</p>", commit));
    }

    html.push_str("<table><tr><th>Benchmark</th><th>Baseline (ns)</th><th>Current (ns)</th><th>Change</th><th>Status</th></tr>");

    let mut regressions = 0;
    let mut improvements = 0;
    let mut stable = 0;

    for comp in comparisons {
        let full_name = match &comp.parameter {
            Some(param) => format!("{}/{}", comp.name, param),
            None => comp.name.clone(),
        };

        let change_str = if comp.change_percent >= 0.0 {
            format!("+{:.2}%", comp.change_percent)
        } else {
            format!("{:.2}%", comp.change_percent)
        };

        let (status, class) = if comp.is_regression {
            regressions += 1;
            ("REGRESSION", "regression")
        } else if comp.is_improvement {
            improvements += 1;
            ("IMPROVEMENT", "improvement")
        } else {
            stable += 1;
            ("STABLE", "stable")
        };

        html.push_str(&format!(
            "<tr><td>{}</td><td>{:.2}</td><td>{:.2}</td><td class=\"{}\">{}</td><td class=\"{}\">{}</td></tr>",
            full_name, comp.baseline_mean, comp.current_mean, class, change_str, class, status
        ));
    }

    html.push_str("</table>");

    html.push_str(&format!(
        r#"<div class="summary">
        <h2>Summary</h2>
        <p><span class="regression">Regressions: {}</span></p>
        <p><span class="improvement">Improvements: {}</span></p>
        <p><span class="stable">Stable: {}</span></p>
        <p><strong>Total: {}</strong></p>
    </div>
    </body>
    </html>"#,
        regressions,
        improvements,
        stable,
        comparisons.len()
    ));

    println!("{}", html);
    Ok(())
}

/// Show detailed statistics for a benchmark
pub fn show_stats(name: &str, criterion_dir: &Path) -> Result<()> {
    let results = parser::parse_criterion_output(criterion_dir)?;

    let matching: Vec<_> = results.iter().filter(|r| r.name == name).collect();

    if matching.is_empty() {
        anyhow::bail!("No benchmark found with name: {}", name);
    }

    println!("{}", format!("=== Statistics for {} ===", name).bold());
    println!();

    for result in matching {
        let param_str = result
            .parameter
            .as_ref()
            .map(|p| format!(" (parameter: {})", p))
            .unwrap_or_default();

        println!("{}{}", result.name.cyan().bold(), param_str);
        println!("  Timestamp: {}", result.timestamp);
        println!();

        let est = &result.estimates;
        println!("  {}", "Mean:".bold());
        println!("    Point estimate: {:.2} ns", est.mean.point_estimate);
        println!("    Standard error: {:.2} ns", est.mean.standard_error);
        println!(
            "    95% CI: [{:.2}, {:.2}] ns",
            est.mean.confidence_interval.lower_bound, est.mean.confidence_interval.upper_bound
        );
        println!();

        println!("  {}", "Median:".bold());
        println!("    Point estimate: {:.2} ns", est.median.point_estimate);
        println!(
            "    95% CI: [{:.2}, {:.2}] ns",
            est.median.confidence_interval.lower_bound, est.median.confidence_interval.upper_bound
        );
        println!();

        println!("  {}", "Standard Deviation:".bold());
        println!("    Point estimate: {:.2} ns", est.std_dev.point_estimate);
        println!(
            "    95% CI: [{:.2}, {:.2}] ns",
            est.std_dev.confidence_interval.lower_bound,
            est.std_dev.confidence_interval.upper_bound
        );
        println!();
    }

    Ok(())
}
