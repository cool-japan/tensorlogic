mod baseline;
mod compare;
mod models;
mod parser;
mod report;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "bench-tracker")]
#[command(about = "Benchmark regression tracking tool for TensorLogic", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Save current benchmark results as baseline
    Save {
        /// Path to criterion output directory
        #[arg(short, long, default_value = "target/criterion")]
        criterion_dir: PathBuf,

        /// Output path for baseline file
        #[arg(short, long, default_value = "benchmarks/baseline.json")]
        output: PathBuf,

        /// Baseline name/tag
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Compare current results against baseline
    Compare {
        /// Path to criterion output directory
        #[arg(short, long, default_value = "target/criterion")]
        criterion_dir: PathBuf,

        /// Path to baseline file
        #[arg(short, long, default_value = "benchmarks/baseline.json")]
        baseline: PathBuf,

        /// Regression threshold (percentage)
        #[arg(short, long, default_value = "5.0")]
        threshold: f64,

        /// Output format (text, json, html)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// List all saved baselines
    List {
        /// Path to baseline file
        #[arg(short, long, default_value = "benchmarks/baseline.json")]
        baseline: PathBuf,
    },

    /// Show detailed statistics for a benchmark
    Stats {
        /// Benchmark name
        #[arg(short, long)]
        name: String,

        /// Path to criterion output directory
        #[arg(short, long, default_value = "target/criterion")]
        criterion_dir: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Save {
            criterion_dir,
            output,
            name,
        } => {
            baseline::save_baseline(&criterion_dir, &output, name.as_deref())?;
        }
        Commands::Compare {
            criterion_dir,
            baseline,
            threshold,
            format,
        } => {
            compare::compare_benchmarks(&criterion_dir, &baseline, threshold, &format)?;
        }
        Commands::List { baseline } => {
            baseline::list_baselines(&baseline)?;
        }
        Commands::Stats {
            name,
            criterion_dir,
        } => {
            report::show_stats(&name, &criterion_dir)?;
        }
    }

    Ok(())
}
