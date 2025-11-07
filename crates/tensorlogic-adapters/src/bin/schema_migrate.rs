//! CLI tool for migrating and transforming TensorLogic schema files.
//!
//! This tool supports:
//! - Converting between JSON and YAML formats
//! - Merging multiple schemas
//! - Computing diffs between schema versions
//! - Checking compatibility between schemas
//!
//! # Usage
//!
//! ```bash
//! # Convert JSON to YAML
//! cargo run --bin schema_migrate -- convert schema.json schema.yaml
//!
//! # Merge two schemas
//! cargo run --bin schema_migrate -- merge schema1.yaml schema2.yaml merged.yaml
//!
//! # Compute diff
//! cargo run --bin schema_migrate -- diff old.yaml new.yaml
//!
//! # Check compatibility
//! cargo run --bin schema_migrate -- check old.yaml new.yaml
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use anyhow::{Context, Result};

use tensorlogic_adapters::{
    check_compatibility, compute_diff, merge_tables, CompatibilityLevel, SymbolTable,
};

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

    match args[1].as_str() {
        "convert" => cmd_convert(&args[2..])?,
        "merge" => cmd_merge(&args[2..])?,
        "diff" => cmd_diff(&args[2..])?,
        "check" => cmd_check(&args[2..])?,
        "--help" | "-h" | "help" => {
            print_usage();
        }
        cmd => {
            eprintln!("Error: Unknown command '{}'", cmd);
            print_usage();
            process::exit(1);
        }
    }

    Ok(())
}

fn cmd_convert(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        eprintln!("Usage: schema_migrate convert <input> <output>");
        process::exit(1);
    }

    let input_path = PathBuf::from(&args[0]);
    let output_path = PathBuf::from(&args[1]);

    println!("Converting schema...");
    println!("  Input:  {}", input_path.display());
    println!("  Output: {}", output_path.display());

    // Load input
    let table = load_schema(&input_path)?;
    println!(
        "✓ Loaded schema with {} domains, {} predicates",
        table.domains.len(),
        table.predicates.len()
    );

    // Save output
    save_schema(&table, &output_path)?;
    println!("✓ Converted successfully");

    Ok(())
}

fn cmd_merge(args: &[String]) -> Result<()> {
    if args.len() < 3 {
        eprintln!("Usage: schema_migrate merge <schema1> <schema2> [<schema3> ...] <output>");
        process::exit(1);
    }

    let output_path = PathBuf::from(&args[args.len() - 1]);
    let input_paths: Vec<PathBuf> = args[..args.len() - 1].iter().map(PathBuf::from).collect();

    println!("Merging {} schemas...", input_paths.len());

    // Load first schema as base
    let mut merged = load_schema(&input_paths[0])?;
    println!(
        "  [1/{}] {}: {} domains, {} predicates",
        input_paths.len(),
        input_paths[0].display(),
        merged.domains.len(),
        merged.predicates.len()
    );

    // Merge remaining schemas
    for (i, path) in input_paths.iter().enumerate().skip(1) {
        let other = load_schema(path)?;
        println!(
            "  [{}/{}] {}: {} domains, {} predicates",
            i + 1,
            input_paths.len(),
            path.display(),
            other.domains.len(),
            other.predicates.len()
        );

        merged = merge_tables(&merged, &other);
    }

    println!();
    println!(
        "Merged result: {} domains, {} predicates",
        merged.domains.len(),
        merged.predicates.len()
    );

    // Save merged schema
    save_schema(&merged, &output_path)?;
    println!("✓ Merged schema saved to {}", output_path.display());

    Ok(())
}

fn cmd_diff(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        eprintln!("Usage: schema_migrate diff <old_schema> <new_schema>");
        process::exit(1);
    }

    let old_path = PathBuf::from(&args[0]);
    let new_path = PathBuf::from(&args[1]);

    println!("Computing diff...");
    println!("  Old: {}", old_path.display());
    println!("  New: {}", new_path.display());
    println!();

    let old_table = load_schema(&old_path)?;
    let new_table = load_schema(&new_path)?;

    let diff = compute_diff(&old_table, &new_table);
    let summary = diff.summary();

    // Print summary
    println!("=== Diff Summary ===");
    println!("Total changes: {}", summary.total_changes());
    println!();

    // Domain changes
    if !diff.domains_added.is_empty() {
        println!("Added domains ({}): ", diff.domains_added.len());
        for domain in &diff.domains_added {
            println!("  + {} (cardinality: {})", domain.name, domain.cardinality);
        }
        println!();
    }

    if !diff.domains_removed.is_empty() {
        println!("Removed domains ({}):", diff.domains_removed.len());
        for domain in &diff.domains_removed {
            println!("  - {} (cardinality: {})", domain.name, domain.cardinality);
        }
        println!();
    }

    if !diff.domains_modified.is_empty() {
        println!("Modified domains ({}):", diff.domains_modified.len());
        for modification in &diff.domains_modified {
            if modification.old_cardinality != modification.new_cardinality {
                println!(
                    "  ~ {}: cardinality {} → {}",
                    modification.domain_name,
                    modification.old_cardinality,
                    modification.new_cardinality
                );
            }
        }
        println!();
    }

    // Predicate changes
    if !diff.predicates_added.is_empty() {
        println!("Added predicates ({}):", diff.predicates_added.len());
        for pred in &diff.predicates_added {
            println!("  + {} (arity: {})", pred.name, pred.arg_domains.len());
        }
        println!();
    }

    if !diff.predicates_removed.is_empty() {
        println!("Removed predicates ({}):", diff.predicates_removed.len());
        for pred in &diff.predicates_removed {
            println!("  - {} (arity: {})", pred.name, pred.arg_domains.len());
        }
        println!();
    }

    if !diff.predicates_modified.is_empty() {
        println!("Modified predicates ({}):", diff.predicates_modified.len());
        for modification in &diff.predicates_modified {
            if modification.signature_changed {
                println!(
                    "  ~ {}: signature {:?} → {:?}",
                    modification.predicate_name,
                    modification.old_signature,
                    modification.new_signature
                );
            }
        }
        println!();
    }

    // Variable changes
    if !diff.variables_added.is_empty() {
        println!("Added variables ({}):", diff.variables_added.len());
        for (var, domain) in &diff.variables_added {
            println!("  + {}: {}", var, domain);
        }
        println!();
    }

    if !diff.variables_removed.is_empty() {
        println!("Removed variables ({}):", diff.variables_removed.len());
        for (var, domain) in &diff.variables_removed {
            println!("  - {}: {}", var, domain);
        }
        println!();
    }

    if !diff.variables_modified.is_empty() {
        println!("Modified variables ({}):", diff.variables_modified.len());
        for modification in &diff.variables_modified {
            println!(
                "  ~ {}: {} → {}",
                modification.variable_name, modification.old_domain, modification.new_domain
            );
        }
        println!();
    }

    // Compatibility
    let compat = check_compatibility(&old_table, &new_table);
    println!("=== Compatibility ===");
    match compat {
        CompatibilityLevel::Identical => {
            println!("✓ Schemas are identical");
        }
        CompatibilityLevel::BackwardCompatible => {
            println!("✓ New schema is backward compatible");
            println!("  (Old clients can work with new schema)");
        }
        CompatibilityLevel::ForwardCompatible => {
            println!("⚠ New schema is forward compatible only");
            println!("  (New clients can work with old schema, but not vice versa)");
        }
        CompatibilityLevel::Breaking => {
            println!("✗ Breaking changes detected");
            println!("  (Migration required for existing clients)");
        }
    }

    Ok(())
}

fn cmd_check(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        eprintln!("Usage: schema_migrate check <old_schema> <new_schema>");
        process::exit(1);
    }

    let old_path = PathBuf::from(&args[0]);
    let new_path = PathBuf::from(&args[1]);

    println!("Checking compatibility...");
    println!("  Old: {}", old_path.display());
    println!("  New: {}", new_path.display());
    println!();

    let old_table = load_schema(&old_path)?;
    let new_table = load_schema(&new_path)?;

    let compat = check_compatibility(&old_table, &new_table);

    match compat {
        CompatibilityLevel::Identical => {
            println!("✓ Schemas are identical");
            println!("  No migration required.");
        }
        CompatibilityLevel::BackwardCompatible => {
            println!("✓ Backward compatible");
            println!("  Safe to deploy. Old clients will continue to work.");
        }
        CompatibilityLevel::ForwardCompatible => {
            println!("⚠ Forward compatible only");
            println!("  New clients can handle old schema, but old clients");
            println!("  may not work with new schema. Consider migration.");
        }
        CompatibilityLevel::Breaking => {
            println!("✗ Breaking changes detected");
            println!("  Migration required. Incompatible changes found.");
            process::exit(1);
        }
    }

    Ok(())
}

fn load_schema(path: &Path) -> Result<SymbolTable> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    let format = detect_format(path)?;
    match format {
        Format::Json => SymbolTable::from_json(&content),
        Format::Yaml => SymbolTable::from_yaml(&content),
    }
    .with_context(|| format!("Failed to parse schema: {}", path.display()))
}

fn save_schema(table: &SymbolTable, path: &Path) -> Result<()> {
    let format = detect_format(path)?;
    let content = match format {
        Format::Json => table.to_json()?,
        Format::Yaml => table.to_yaml()?,
    };

    fs::write(path, content)
        .with_context(|| format!("Failed to write file: {}", path.display()))?;

    Ok(())
}

fn detect_format(path: &Path) -> Result<Format> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => Ok(Format::Json),
        Some("yaml") | Some("yml") => Ok(Format::Yaml),
        _ => {
            anyhow::bail!("Cannot detect format from extension: {}", path.display())
        }
    }
}

fn print_usage() {
    println!("TensorLogic Schema Migration Tool");
    println!();
    println!("USAGE:");
    println!("    schema_migrate <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("    convert <input> <output>              Convert schema between formats");
    println!("    merge <s1> <s2> [...] <output>       Merge multiple schemas");
    println!("    diff <old> <new>                      Show differences between schemas");
    println!("    check <old> <new>                     Check compatibility");
    println!("    help                                  Print this help message");
    println!();
    println!("EXAMPLES:");
    println!("    # Convert JSON to YAML");
    println!("    schema_migrate convert schema.json schema.yaml");
    println!();
    println!("    # Merge two schemas");
    println!("    schema_migrate merge base.yaml extension.yaml merged.yaml");
    println!();
    println!("    # Show diff between versions");
    println!("    schema_migrate diff v1.yaml v2.yaml");
    println!();
    println!("    # Check compatibility");
    println!("    schema_migrate check old.yaml new.yaml");
}

#[derive(Debug)]
enum Format {
    Json,
    Yaml,
}
