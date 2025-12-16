//! TensorLogic CLI - Command-line interface for TensorLogic compilation
//!
//! Provides comprehensive tooling for compiling logical expressions to tensor graphs.

mod analysis;
mod batch;
mod benchmark;
mod cache;
mod cli;
mod completion;
mod config;
mod conversion;
mod error_suggestions;
mod executor;
mod macros;
mod optimize;
mod output;
mod parser;
mod profile;
mod repl;
mod simplify;
mod snapshot;
mod watch;

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationConfig, CompilerContext};
use tensorlogic_ir::{export_to_dot, validate_graph, EinsumGraph, TLExpr};

use analysis::GraphMetrics;
use batch::BatchProcessor;
use benchmark::{BenchmarkResults, Benchmarker};
use cache::CompilationCache;
use cli::{Cli, Commands, ExecutionOutputFormat, InputFormat, OutputFormat};
use config::Config;
use executor::{Backend, CliExecutor, ExecutionConfig};
use optimize::{optimize_einsum_graph, OptimizationConfig, OptimizationLevel};
use output::{enable_colors, print_compilation_success, print_error, print_header, print_success};
use profile::{ProfileConfig, Profiler};
use repl::Repl;
use watch::FileWatcher;

fn main() {
    if let Err(e) = run() {
        print_error(&format!("{:#}", e));
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    // Load configuration
    let mut config = if cli.no_config {
        Config::default()
    } else {
        Config::load_default()
    };

    // Override config with CLI options
    if let Some(strategy) = &cli.strategy {
        config.strategy = strategy.clone();
    }
    if cli.validate {
        config.validate = true;
    }
    if cli.debug {
        config.debug = true;
    }
    if cli.no_color {
        config.colored = false;
    }

    // Set color mode
    enable_colors(config.colored);

    // Handle commands
    match &cli.command {
        Some(Commands::Repl) => {
            let context = create_context(&config, &cli.domains)?;
            let mut repl = Repl::new(config, context)?;
            repl.run()
        }
        Some(Commands::Batch { files }) => {
            let context = create_context(&config, &cli.domains)?;
            let mut processor = BatchProcessor::new(context, config.validate);

            for file in files {
                print_header(&format!("Processing: {}", file.display()));
                let result = processor.process_file(file)?;
                result.print_summary();
            }
            Ok(())
        }
        Some(Commands::Watch { file }) => {
            let context = create_context(&config, &cli.domains)?;
            let mut watcher = FileWatcher::new(
                context,
                config.validate,
                config.watch.clear_screen,
                config.watch.show_timestamps,
            );
            watcher.watch(file)
        }
        Some(Commands::Completion { shell }) => {
            completion::generate_for_shell(*shell);
            Ok(())
        }
        Some(Commands::Config { command }) => handle_config_command(command),
        Some(Commands::Convert {
            input,
            from,
            to,
            output,
            pretty,
        }) => {
            let result = conversion::convert(input, *from, *to, *pretty)?;

            match output {
                Some(path) => {
                    fs::write(path, &result).context("Failed to write output file")?;
                    print_success(&format!("Converted to: {}", path.display()));
                }
                None => {
                    println!("{}", result);
                }
            }
            Ok(())
        }
        Some(Commands::Execute {
            input,
            input_format,
            backend,
            metrics,
            intermediates,
            trace,
            output_format,
        }) => handle_execute_command(
            input,
            input_format,
            backend,
            *metrics,
            *intermediates,
            *trace,
            output_format,
            &config,
            &cli.domains,
        ),
        Some(Commands::Optimize {
            input,
            input_format,
            level,
            stats,
            verbose,
            output,
            output_format,
        }) => handle_optimize_command(
            input,
            input_format,
            level,
            *stats,
            *verbose,
            output,
            output_format,
            &config,
            &cli.domains,
        ),
        Some(Commands::Backends) => {
            executor::list_backends();
            Ok(())
        }
        Some(Commands::Benchmark {
            input,
            input_format,
            iterations,
            backend,
            execute,
            optimize,
            verbose,
            json,
        }) => handle_benchmark_command(
            input,
            input_format,
            *iterations,
            backend,
            *execute,
            *optimize,
            *verbose,
            *json,
            &config,
            &cli.domains,
        ),
        Some(Commands::Profile {
            input,
            input_format,
            no_optimize,
            opt_level,
            validate,
            execute,
            exec_backend,
            warmup,
            runs,
            json,
        }) => handle_profile_command(
            input,
            input_format,
            !no_optimize,
            opt_level,
            *validate,
            *execute,
            exec_backend,
            *warmup,
            *runs,
            *json,
            &config,
            &cli.domains,
        ),
        Some(Commands::Cache { command }) => handle_cache_command(command, &config),
        Some(Commands::Snapshot { command }) => handle_snapshot_command(command, &config),
        None => {
            // Main compilation mode
            compile_mode(&cli, &config)
        }
    }
}

fn compile_mode(cli: &Cli, config: &Config) -> Result<()> {
    // Read input
    let input = cli
        .input
        .as_ref()
        .context("Input is required for compilation mode")?;
    let expr = read_input(input, &cli.input_format)?;

    if config.debug {
        eprintln!("Parsed expression: {:?}", expr);
    }

    // Create compiler context
    let mut context = create_context(config, &cli.domains)?;

    if config.debug {
        eprintln!("Context: {} domains", context.domains.len());
    }

    // Create or load cache if enabled
    let mut cache = if config.cache.disk_cache_enabled {
        Some(CompilationCache::new(
            config.cache.disk_cache_dir.clone(),
            config.cache.disk_cache_max_size_mb,
        )?)
    } else {
        None
    };

    // Try to get from cache first
    let graph = if let Some(ref mut cache_instance) = cache {
        if let Some(cached_graph) = cache_instance.get(&expr, &context) {
            if config.debug {
                eprintln!("Using cached compilation result");
            }
            cached_graph
        } else {
            // Compile if not in cache
            let compiled_graph = compile_to_einsum_with_context(&expr, &mut context)
                .context("Compilation failed")?;

            // Store in cache
            cache_instance.put(&expr, &context, &compiled_graph)?;

            compiled_graph
        }
    } else {
        // No cache, just compile
        compile_to_einsum_with_context(&expr, &mut context).context("Compilation failed")?
    };

    if config.debug {
        eprintln!(
            "Compiled: {} tensors, {} nodes",
            graph.tensors.len(),
            graph.nodes.len()
        );
    }

    // Validate if requested
    if config.validate {
        let report = validate_graph(&graph);
        if !report.is_valid() {
            print_error("Validation failed:");
            for error in &report.errors {
                eprintln!("  - {}", error.message);
            }
            anyhow::bail!("Graph validation failed");
        }
        if config.debug && !report.warnings.is_empty() {
            eprintln!("Validation warnings:");
            for warning in &report.warnings {
                eprintln!("  - {}", warning.message);
            }
        }
    }

    // Show metrics if requested
    if cli.analyze {
        let metrics = GraphMetrics::analyze(&graph);
        metrics.print();
        println!();
    }

    // Generate output
    let output = generate_output(&graph, &cli.output_format)?;

    // Write output
    match &cli.output {
        Some(path) => {
            fs::write(path, output).context("Failed to write output file")?;
            if config.debug {
                print_success(&format!("Output written to: {}", path.display()));
            }
        }
        None => {
            // Don't print status for structured formats (they need clean stdout for parsing)
            let is_structured = matches!(cli.output_format, OutputFormat::Json);
            if !cli.quiet && !is_structured {
                print_compilation_success(&graph);
                println!();
            }
            println!("{}", output);
        }
    }

    Ok(())
}

fn read_input(input: &str, format: &InputFormat) -> Result<TLExpr> {
    match format {
        InputFormat::Expr => parser::parse_expression(input),
        InputFormat::Json => {
            let content = if input == "-" {
                use std::io::Read;
                let mut buffer = String::new();
                std::io::stdin().read_to_string(&mut buffer)?;
                buffer
            } else {
                fs::read_to_string(input).context("Failed to read input file")?
            };
            serde_json::from_str(&content).context("Failed to parse JSON")
        }
        InputFormat::Yaml => {
            let content = fs::read_to_string(input).context("Failed to read input file")?;
            serde_yaml::from_str(&content).context("Failed to parse YAML")
        }
    }
}

fn create_context(config: &Config, cli_domains: &[(String, usize)]) -> Result<CompilerContext> {
    let compilation_config = match config.strategy.as_str() {
        "soft_differentiable" => CompilationConfig::soft_differentiable(),
        "hard_boolean" => CompilationConfig::hard_boolean(),
        "fuzzy_godel" => CompilationConfig::fuzzy_godel(),
        "fuzzy_product" => CompilationConfig::fuzzy_product(),
        "fuzzy_lukasiewicz" => CompilationConfig::fuzzy_lukasiewicz(),
        "probabilistic" => CompilationConfig::probabilistic(),
        _ => anyhow::bail!("Unknown compilation strategy: {}", config.strategy),
    };

    let mut ctx = CompilerContext::with_config(compilation_config);

    // Add domains from config
    for (name, size) in &config.domains {
        ctx.add_domain(name, *size);
    }

    // Add domains from CLI (override config)
    for (name, size) in cli_domains {
        ctx.add_domain(name, *size);
    }

    // Ensure default domain exists
    if !ctx.domains.contains_key("D") {
        ctx.add_domain("D", 100);
    }

    Ok(ctx)
}

fn generate_output(graph: &EinsumGraph, format: &OutputFormat) -> Result<String> {
    match format {
        OutputFormat::Graph => Ok(format!("{:#?}", graph)),
        OutputFormat::Dot => Ok(export_to_dot(graph)),
        OutputFormat::Json => {
            serde_json::to_string_pretty(graph).context("Failed to serialize to JSON")
        }
        OutputFormat::Stats => {
            let metrics = GraphMetrics::analyze(graph);
            let mut output = String::new();
            use std::fmt::Write;
            writeln!(&mut output, "Graph Statistics:")?;
            writeln!(&mut output, "  Tensors: {}", metrics.tensor_count)?;
            writeln!(&mut output, "  Nodes: {}", metrics.node_count)?;
            writeln!(&mut output, "  Inputs: {}", metrics.input_count)?;
            writeln!(&mut output, "  Outputs: {}", metrics.output_count)?;
            writeln!(&mut output, "  Depth: {}", metrics.depth)?;
            writeln!(&mut output, "  Avg Fanout: {:.2}", metrics.avg_fanout)?;
            writeln!(&mut output, "\nOperation Breakdown:")?;
            for (op, count) in &metrics.op_breakdown {
                writeln!(&mut output, "  {}: {}", op, count)?;
            }
            writeln!(&mut output, "\nEstimated Complexity:")?;
            writeln!(&mut output, "  FLOPs: {}", metrics.estimated_flops)?;
            writeln!(&mut output, "  Memory: {} bytes", metrics.estimated_memory)?;
            Ok(output)
        }
    }
}

fn handle_config_command(command: &cli::ConfigCommand) -> Result<()> {
    use cli::ConfigCommand;

    match command {
        ConfigCommand::Show => {
            let config = Config::load_default();
            let toml_str = toml::to_string_pretty(&config)?;
            println!("{}", toml_str);
        }
        ConfigCommand::Path => {
            let path = Config::config_path();
            println!("{}", path.display());
        }
        ConfigCommand::Init => {
            let path = Config::create_default()?;
            print_success(&format!("Created config file: {}", path.display()));
        }
        ConfigCommand::Edit => {
            let path = Config::config_path();
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
            std::process::Command::new(editor)
                .arg(&path)
                .status()
                .context("Failed to open editor")?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_execute_command(
    input: &str,
    input_format: &InputFormat,
    backend_str: &str,
    metrics: bool,
    intermediates: bool,
    trace: bool,
    output_format: &ExecutionOutputFormat,
    config: &Config,
    domains: &[(String, usize)],
) -> Result<()> {
    // Parse expression
    let expr = read_input(input, input_format)?;

    // Compile to graph
    let mut context = create_context(config, domains)?;
    print_header("Compiling expression...");
    let graph =
        compile_to_einsum_with_context(&expr, &mut context).context("Compilation failed")?;

    print_compilation_success(&graph);

    // Parse backend
    let backend = Backend::from_str(backend_str)?;

    // Create execution config
    let exec_config = ExecutionConfig {
        backend,
        device: tensorlogic_scirs_backend::DeviceType::Cpu,
        show_metrics: metrics,
        show_intermediates: intermediates,
        validate_shapes: true,
        trace,
    };

    // Execute
    print_header(&format!("Executing with {}...", backend.name()));
    let executor = CliExecutor::new(exec_config.clone())?;
    let result = executor.execute(&graph)?;

    // Output results
    match output_format {
        ExecutionOutputFormat::Table => {
            result.print_summary(&exec_config);
        }
        ExecutionOutputFormat::Json => {
            let output_json = serde_json::json!({
                "output": result.output.as_slice().unwrap_or(&[]).to_vec(),
                "shape": result.output.shape(),
                "execution_time_ms": result.execution_time_ms,
                "backend": backend.name(),
                "memory_bytes": result.memory_bytes,
            });
            println!("{}", serde_json::to_string_pretty(&output_json)?);
        }
        ExecutionOutputFormat::Csv => {
            let flat = result.output.as_slice().unwrap_or(&[]);
            for val in flat {
                println!("{:.6}", val);
            }
        }
        ExecutionOutputFormat::Numpy => {
            println!("# shape: {:?}", result.output.shape());
            println!("# dtype: f64");
            let flat = result.output.as_slice().unwrap_or(&[]);
            for (i, val) in flat.iter().enumerate() {
                if i > 0 && i % 10 == 0 {
                    println!();
                }
                print!("{:.6} ", val);
            }
            println!();
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_optimize_command(
    input: &str,
    input_format: &InputFormat,
    level_str: &str,
    show_stats: bool,
    verbose: bool,
    output: &Option<std::path::PathBuf>,
    output_format: &OutputFormat,
    config: &Config,
    domains: &[(String, usize)],
) -> Result<()> {
    // Parse expression
    let expr = read_input(input, input_format)?;

    // Compile to graph
    let mut context = create_context(config, domains)?;
    print_header("Compiling expression...");
    let graph =
        compile_to_einsum_with_context(&expr, &mut context).context("Compilation failed")?;

    print_compilation_success(&graph);

    // Parse optimization level
    let level = OptimizationLevel::from_str(level_str)?;

    // Create optimization config
    let opt_config = OptimizationConfig {
        level,
        enable_dce: true,
        enable_cse: true,
        enable_identity: true,
        show_stats,
        verbose,
    };

    // Optimize
    print_header("Optimizing graph...");
    let (optimized_graph, _stats) = optimize_einsum_graph(graph, &opt_config)?;

    // Generate output
    let output_str = generate_output(&optimized_graph, output_format)?;

    // Write output
    match output {
        Some(path) => {
            fs::write(path, output_str).context("Failed to write output file")?;
            print_success(&format!("Optimized graph written to: {}", path.display()));
        }
        None => {
            println!("\nOptimized Graph:");
            println!("{}", output_str);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_benchmark_command(
    input: &str,
    input_format: &InputFormat,
    iterations: usize,
    backend_str: &str,
    execute: bool,
    optimize: bool,
    verbose: bool,
    json: bool,
    config: &Config,
    domains: &[(String, usize)],
) -> Result<()> {
    // Parse expression
    let expr = read_input(input, input_format)?;

    // Create context
    let context = create_context(config, domains)?;

    // Create benchmarker (quiet mode for JSON output)
    let benchmarker = Benchmarker::with_quiet(iterations, verbose, json);

    // Results
    let mut results = BenchmarkResults::new();

    // Compilation benchmark (always run)
    results.compilation_times = benchmarker.benchmark_compilation(&expr, &context)?;

    // Execution benchmark (if requested)
    if execute {
        let mut ctx = context.clone();
        let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;
        let backend = Backend::from_str(backend_str)?;
        results.execution_times = benchmarker.benchmark_execution(&graph, backend)?;
    }

    // Optimization benchmark (if requested)
    if optimize {
        results.optimization_times = benchmarker.benchmark_optimization(&expr, &context)?;
    }

    // Output results
    if json {
        println!("{}", serde_json::to_string_pretty(&results.to_json())?);
    } else {
        results.print_summary();
        print_success("\nBenchmark complete");
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_profile_command(
    input: &str,
    input_format: &InputFormat,
    optimize: bool,
    opt_level: &str,
    validate: bool,
    execute: bool,
    exec_backend: &str,
    warmup: usize,
    runs: usize,
    json: bool,
    config: &Config,
    domains: &[(String, usize)],
) -> Result<()> {
    // Parse expression
    let expr = read_input(input, input_format)?;

    // Create context
    let context = create_context(config, domains)?;

    // Parse optimization level
    let optimization_level = OptimizationLevel::from_str(opt_level)?;

    // Create profile config
    let profile_config = ProfileConfig {
        include_optimization: optimize,
        optimization_level,
        include_validation: validate,
        detailed: true,
        warmup_runs: warmup,
        profile_runs: runs,
        include_execution: execute,
        execution_backend: exec_backend.to_string(),
    };

    // Run profiler
    let profiler = Profiler::new(profile_config);
    let profile_data = profiler.profile(&expr, &context)?;

    // Output results
    if json {
        println!("{}", serde_json::to_string_pretty(&profile_data.to_json())?);
    } else {
        profile_data.print();
    }

    Ok(())
}

fn handle_cache_command(command: &cli::CacheCommand, config: &Config) -> Result<()> {
    use cli::CacheCommand;

    // Create cache instance
    let cache_dir = config.cache.disk_cache_dir.clone();
    let max_size_mb = config.cache.disk_cache_max_size_mb;

    match command {
        CacheCommand::Stats => {
            let cache = CompilationCache::new(cache_dir, max_size_mb)?;
            let stats = cache.stats();
            stats.print();
        }
        CacheCommand::Clear => {
            let mut cache = CompilationCache::new(cache_dir, max_size_mb)?;
            cache.clear()?;
            print_success("Cache cleared successfully");
        }
        CacheCommand::Enable => {
            print_success("Caching enabled (update config to persist)");
            println!("Edit your .tensorlogicrc file and set:");
            println!("  [cache]");
            println!("  disk_cache_enabled = true");
        }
        CacheCommand::Disable => {
            print_success("Caching disabled (update config to persist)");
            println!("Edit your .tensorlogicrc file and set:");
            println!("  [cache]");
            println!("  disk_cache_enabled = false");
        }
        CacheCommand::Path => {
            let cache_dir = match cache_dir {
                Some(dir) => dir,
                None => CompilationCache::default_cache_dir()?,
            };
            println!("{}", cache_dir.display());
        }
    }

    Ok(())
}

fn handle_snapshot_command(command: &cli::SnapshotCommand, config: &Config) -> Result<()> {
    use crate::snapshot::SnapshotSuite;
    use cli::SnapshotCommand;
    use std::env;

    // Determine snapshot directory
    let snapshot_dir = env::var("TENSORLOGIC_SNAPSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".tensorlogic_snapshots")
        });

    let suite = SnapshotSuite::new("tensorlogic", snapshot_dir.clone());

    match command {
        SnapshotCommand::Record {
            name,
            expression,
            strategy,
            domains,
        } => {
            let expr = parser::parse_expression(expression)?;
            let context = if let Some(strat) = strategy {
                let mut cfg = config.clone();
                cfg.strategy = strat.clone();
                create_context(&cfg, domains)?
            } else {
                create_context(config, domains)?
            };
            suite.record(name, &expr, &context, expression)?;
            print_success(&format!("Recorded snapshot: {}", name));
        }
        SnapshotCommand::Verify {
            name,
            expression,
            strategy,
            domains,
        } => {
            let expr = parser::parse_expression(expression)?;
            let context = if let Some(strat) = strategy {
                let mut cfg = config.clone();
                cfg.strategy = strat.clone();
                create_context(&cfg, domains)?
            } else {
                create_context(config, domains)?
            };
            let diff = suite.verify(name, &expr, &context, expression)?;
            if diff.is_match() {
                print_success(&format!("✓ Snapshot matches: {}", name));
            } else {
                print_error(&format!("✗ Snapshot differs: {}", name));
                for d in &diff.differences {
                    eprintln!("  - {}", d);
                }
                std::process::exit(1);
            }
        }
        SnapshotCommand::Update {
            name,
            expression,
            strategy,
            domains,
        } => {
            let expr = parser::parse_expression(expression)?;
            let context = if let Some(strat) = strategy {
                let mut cfg = config.clone();
                cfg.strategy = strat.clone();
                create_context(&cfg, domains)?
            } else {
                create_context(config, domains)?
            };
            suite.update(name, &expr, &context, expression)?;
            print_success(&format!("Updated snapshot: {}", name));
        }
        SnapshotCommand::List => {
            let snapshots = suite.list_snapshots()?;
            if snapshots.is_empty() {
                println!("No snapshots found in {}", snapshot_dir.display());
            } else {
                println!("Snapshots in {}:", snapshot_dir.display());
                for snapshot in &snapshots {
                    println!("  - {}", snapshot);
                }
                println!("\nTotal: {} snapshot(s)", snapshots.len());
            }
        }
        SnapshotCommand::Path => {
            println!("{}", snapshot_dir.display());
        }
    }

    Ok(())
}
