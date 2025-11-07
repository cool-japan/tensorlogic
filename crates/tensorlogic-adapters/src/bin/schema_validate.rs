//! CLI tool for validating TensorLogic schema files.
//!
//! This tool validates JSON or YAML schema files and reports errors,
//! warnings, and suggestions for improvement.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin schema_validate -- schema.yaml
//! cargo run --bin schema_validate -- --format json schema.json
//! cargo run --bin schema_validate -- --analyze schema.yaml
//! ```

use std::fs;
use std::path::PathBuf;
use std::process;

use anyhow::{Context, Result};

use tensorlogic_adapters::{SchemaAnalyzer, SchemaStatistics, SchemaValidator, SymbolTable};

/// Main entry point for the schema validation CLI.
fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    // Parse arguments
    let mut format = Format::Auto;
    let mut analyze = false;
    let mut stats = false;
    let mut file_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--format" | "-f" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --format requires an argument (json or yaml)");
                    process::exit(1);
                }
                format = match args[i].to_lowercase().as_str() {
                    "json" => Format::Json,
                    "yaml" | "yml" => Format::Yaml,
                    _ => {
                        eprintln!("Error: Invalid format '{}'. Use 'json' or 'yaml'", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--analyze" | "-a" => {
                analyze = true;
            }
            "--stats" | "-s" => {
                stats = true;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            arg if !arg.starts_with('-') => {
                file_path = Some(PathBuf::from(arg));
            }
            _ => {
                eprintln!("Error: Unknown option '{}'", args[i]);
                print_usage();
                process::exit(1);
            }
        }
        i += 1;
    }

    let file_path = file_path.context("No input file specified")?;

    // Auto-detect format from extension
    if matches!(format, Format::Auto) {
        format = match file_path.extension().and_then(|e| e.to_str()) {
            Some("json") => Format::Json,
            Some("yaml") | Some("yml") => Format::Yaml,
            _ => {
                eprintln!("Error: Cannot detect format from file extension. Use --format");
                process::exit(1);
            }
        };
    }

    println!("Validating schema: {}", file_path.display());
    println!("Format: {:?}", format);
    println!();

    // Load schema
    let content = fs::read_to_string(&file_path)
        .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

    let table = match format {
        Format::Json => SymbolTable::from_json(&content)?,
        Format::Yaml => SymbolTable::from_yaml(&content)?,
        Format::Auto => unreachable!(),
    };

    println!("✓ Schema loaded successfully");
    println!("  Domains: {}", table.domains.len());
    println!("  Predicates: {}", table.predicates.len());
    println!("  Variables: {}", table.variables.len());
    println!();

    // Validate schema
    let validator = SchemaValidator::new(&table);
    let report = validator.validate()?;

    if report.errors.is_empty() {
        println!("✓ Schema validation passed");
    } else {
        println!("✗ Schema validation failed");
        println!();
        println!("Errors:");
        for error in &report.errors {
            println!("  • {}", error);
        }
    }

    if !report.warnings.is_empty() {
        println!();
        println!("Warnings:");
        for warning in &report.warnings {
            println!("  • {}", warning);
        }
    }

    if !report.hints.is_empty() {
        println!();
        println!("Hints:");
        for hint in &report.hints {
            println!("  • {}", hint);
        }
    }

    // Analyze schema if requested
    if analyze {
        println!();
        println!("=== Schema Analysis ===");
        let recommendations = SchemaAnalyzer::analyze(&table);

        if !recommendations.issues.is_empty() {
            println!();
            println!("Issues detected:");
            for issue in &recommendations.issues {
                let severity = match issue.severity() {
                    1 => "INFO",
                    2 => "WARN",
                    3 => "ERROR",
                    _ => "UNKNOWN",
                };
                println!("  [{}] {}", severity, issue.description());
            }
        }

        if !recommendations.suggestions.is_empty() {
            println!();
            println!("Suggestions:");
            for suggestion in &recommendations.suggestions {
                println!("  • {}", suggestion);
            }
        }
    }

    // Show statistics if requested
    if stats {
        println!();
        println!("=== Schema Statistics ===");
        let statistics = SchemaStatistics::compute(&table);

        println!("Domains: {}", statistics.domain_count);
        println!("Predicates: {}", statistics.predicate_count);
        println!("Variables: {}", statistics.variable_count);
        println!("Total cardinality: {}", statistics.total_cardinality);
        println!("Average cardinality: {:.2}", statistics.avg_cardinality);
        println!("Max cardinality: {}", statistics.max_cardinality);
        println!("Min cardinality: {}", statistics.min_cardinality);
        println!("Complexity score: {:.2}", statistics.complexity_score());

        if !statistics.arity_distribution.is_empty() {
            println!();
            println!("Arity distribution:");
            let mut arities: Vec<_> = statistics.arity_distribution.iter().collect();
            arities.sort_by_key(|(arity, _)| *arity);
            for (arity, count) in arities {
                println!("  Arity {}: {} predicate(s)", arity, count);
            }
        }

        if !statistics.domain_usage_frequency.is_empty() {
            println!();
            println!("Most used domains:");
            let top_domains = statistics.most_used_domains(5);
            for (domain, count) in top_domains {
                println!("  {} (used {} times)", domain, count);
            }
        }

        if !statistics.unused_domains.is_empty() {
            println!();
            println!("Unused domains:");
            for domain in &statistics.unused_domains {
                println!("  {}", domain);
            }
        }
    }

    // Exit with error code if validation failed
    if !report.errors.is_empty() {
        process::exit(1);
    }

    Ok(())
}

fn print_usage() {
    println!("TensorLogic Schema Validator");
    println!();
    println!("USAGE:");
    println!("    schema_validate [OPTIONS] <FILE>");
    println!();
    println!("OPTIONS:");
    println!("    -f, --format <FORMAT>    Specify format (json or yaml). Auto-detected if not specified.");
    println!("    -a, --analyze            Perform deep analysis and show recommendations");
    println!("    -s, --stats              Show schema statistics");
    println!("    -h, --help               Print this help message");
    println!();
    println!("EXAMPLES:");
    println!("    schema_validate schema.yaml");
    println!("    schema_validate --format json schema.json");
    println!("    schema_validate --analyze --stats schema.yaml");
}

#[derive(Debug)]
enum Format {
    Auto,
    Json,
    Yaml,
}
