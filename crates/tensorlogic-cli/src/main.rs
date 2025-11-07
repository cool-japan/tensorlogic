//! TensorLogic CLI - Command-line interface for TensorLogic compilation
//!
//! Provides comprehensive tooling for compiling logical expressions to tensor graphs.

mod analysis;
mod batch;
mod cli;
mod completion;
mod config;
mod output;
mod parser;
mod repl;
mod watch;

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationConfig, CompilerContext};
use tensorlogic_ir::{export_to_dot, validate_graph, EinsumGraph, TLExpr};

use analysis::GraphMetrics;
use batch::BatchProcessor;
use cli::{Cli, Commands, InputFormat, OutputFormat};
use config::Config;
use output::{enable_colors, print_compilation_success, print_error, print_header, print_success};
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
        None => {
            // Main compilation mode
            compile_mode(&cli, &config)
        }
    }
}

fn compile_mode(cli: &Cli, config: &Config) -> Result<()> {
    // Read input
    let expr = read_input(&cli.input, &cli.input_format)?;

    if config.debug {
        eprintln!("Parsed expression: {:?}", expr);
    }

    // Create compiler context
    let mut context = create_context(config, &cli.domains)?;

    if config.debug {
        eprintln!("Context: {} domains", context.domains.len());
    }

    // Compile
    let graph =
        compile_to_einsum_with_context(&expr, &mut context).context("Compilation failed")?;

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
            if !cli.quiet {
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
