//! CLI argument definitions using clap

use clap::{Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "tensorlogic")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Input expression or file path (required for compilation mode)
    #[arg(value_name = "INPUT")]
    pub input: Option<String>,

    /// Input format
    #[arg(short = 'f', long, value_enum, default_value = "expr")]
    pub input_format: InputFormat,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output format
    #[arg(short = 'F', long, value_enum, default_value = "graph")]
    pub output_format: OutputFormat,

    /// Compilation strategy
    #[arg(short, long)]
    pub strategy: Option<String>,

    /// Domain definitions (name:size pairs, can be specified multiple times)
    #[arg(short, long, value_parser = parse_domain)]
    pub domains: Vec<(String, usize)>,

    /// Enable validation
    #[arg(long)]
    pub validate: bool,

    /// Enable debug output
    #[arg(long)]
    pub debug: bool,

    /// Analyze graph and show metrics
    #[arg(short, long)]
    pub analyze: bool,

    /// Quiet mode (minimal output)
    #[arg(short, long)]
    pub quiet: bool,

    /// Disable colored output
    #[arg(long)]
    pub no_color: bool,

    /// Don't load configuration file
    #[arg(long)]
    pub no_config: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start interactive REPL mode
    Repl,

    /// Batch process multiple expressions from files
    Batch {
        /// Input files containing expressions (one per line)
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },

    /// Watch a file and recompile on changes
    Watch {
        /// File to watch
        file: PathBuf,
    },

    /// Generate shell completion scripts
    Completion {
        /// Shell type
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Configuration file management
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },

    /// Convert between formats
    Convert {
        /// Input file or expression
        input: String,

        /// Input format
        #[arg(short = 'f', long, value_enum)]
        from: ConvertFormat,

        /// Output format
        #[arg(short = 't', long, value_enum)]
        to: ConvertFormat,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Pretty-print the output
        #[arg(short, long)]
        pretty: bool,
    },

    /// Execute compiled expressions with backend selection
    Execute {
        /// Input expression or file path
        input: String,

        /// Input format
        #[arg(short = 'f', long, value_enum, default_value = "expr")]
        input_format: InputFormat,

        /// Backend to use (cpu, simd, gpu, parallel, profiled)
        #[arg(short, long, default_value = "cpu")]
        backend: String,

        /// Show performance metrics
        #[arg(long)]
        metrics: bool,

        /// Show intermediate tensors
        #[arg(long)]
        intermediates: bool,

        /// Enable execution tracing
        #[arg(long)]
        trace: bool,

        /// Output format for results
        #[arg(short = 'F', long, value_enum, default_value = "table")]
        output_format: ExecutionOutputFormat,
    },

    /// Optimize compiled graphs
    Optimize {
        /// Input expression or file path
        input: String,

        /// Input format
        #[arg(short = 'f', long, value_enum, default_value = "expr")]
        input_format: InputFormat,

        /// Optimization level (none, basic, standard, aggressive)
        #[arg(short, long, default_value = "standard")]
        level: String,

        /// Show optimization statistics
        #[arg(long)]
        stats: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format
        #[arg(short = 'F', long, value_enum, default_value = "graph")]
        output_format: OutputFormat,
    },

    /// List available backends and their capabilities
    Backends,

    /// Benchmark compilation and execution performance
    Benchmark {
        /// Input expression or file path
        input: String,

        /// Input format
        #[arg(short = 'f', long, value_enum, default_value = "expr")]
        input_format: InputFormat,

        /// Number of iterations
        #[arg(short = 'n', long, default_value = "10")]
        iterations: usize,

        /// Backend to benchmark (for execution benchmarks)
        #[arg(short, long, default_value = "cpu")]
        backend: String,

        /// Include execution benchmarks
        #[arg(long)]
        execute: bool,

        /// Include optimization benchmarks
        #[arg(long)]
        optimize: bool,

        /// Show detailed timing for each iteration
        #[arg(short, long)]
        verbose: bool,

        /// Export results as JSON
        #[arg(long)]
        json: bool,
    },

    /// Profile compilation with detailed phase breakdown
    Profile {
        /// Input expression or file path
        input: String,

        /// Input format
        #[arg(short = 'f', long, value_enum, default_value = "expr")]
        input_format: InputFormat,

        /// Skip optimization in profile
        #[arg(long)]
        no_optimize: bool,

        /// Optimization level (none, basic, standard, aggressive)
        #[arg(long, default_value = "standard")]
        opt_level: String,

        /// Include validation in profile
        #[arg(long)]
        validate: bool,

        /// Include execution profiling
        #[arg(long)]
        execute: bool,

        /// Backend for execution profiling (cpu, parallel, profiled)
        #[arg(long, default_value = "cpu")]
        exec_backend: String,

        /// Number of warmup runs
        #[arg(long, default_value = "1")]
        warmup: usize,

        /// Number of profiling runs to average
        #[arg(long, default_value = "3")]
        runs: usize,

        /// Export results as JSON
        #[arg(long)]
        json: bool,
    },

    /// Manage persistent compilation cache
    Cache {
        #[command(subcommand)]
        command: CacheCommand,
    },

    /// Snapshot testing for output consistency
    Snapshot {
        #[command(subcommand)]
        command: SnapshotCommand,
    },
}

#[derive(Subcommand)]
pub enum CacheCommand {
    /// Show cache statistics
    Stats,
    /// Clear the entire cache
    Clear,
    /// Enable caching
    Enable,
    /// Disable caching
    Disable,
    /// Show cache directory path
    Path,
}

#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Show current configuration
    Show,
    /// Show configuration file path
    Path,
    /// Initialize default configuration file
    Init,
    /// Edit configuration file
    Edit,
}

#[derive(Subcommand)]
pub enum SnapshotCommand {
    /// Record a new snapshot
    Record {
        /// Test name
        name: String,
        /// Input expression
        expression: String,
        /// Compilation strategy
        #[arg(short, long)]
        strategy: Option<String>,
        /// Domain definitions (name:size pairs)
        #[arg(short, long, value_parser = parse_domain)]
        domains: Vec<(String, usize)>,
    },
    /// Verify expression against recorded snapshot
    Verify {
        /// Test name
        name: String,
        /// Input expression
        expression: String,
        /// Compilation strategy
        #[arg(short, long)]
        strategy: Option<String>,
        /// Domain definitions (name:size pairs)
        #[arg(short, long, value_parser = parse_domain)]
        domains: Vec<(String, usize)>,
    },
    /// Update an existing snapshot
    Update {
        /// Test name
        name: String,
        /// Input expression
        expression: String,
        /// Compilation strategy
        #[arg(short, long)]
        strategy: Option<String>,
        /// Domain definitions (name:size pairs)
        #[arg(short, long, value_parser = parse_domain)]
        domains: Vec<(String, usize)>,
    },
    /// List all snapshots
    List,
    /// Show snapshot directory path
    Path,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum InputFormat {
    /// Direct expression string
    Expr,
    /// JSON file or stdin
    Json,
    /// YAML file
    Yaml,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    /// Debug graph representation
    Graph,
    /// Graphviz DOT format
    Dot,
    /// JSON serialization
    Json,
    /// Statistics only
    Stats,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ConvertFormat {
    /// Expression string
    Expr,
    /// JSON format
    Json,
    /// YAML format
    Yaml,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ExecutionOutputFormat {
    /// Human-readable table
    Table,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// NumPy text format
    Numpy,
}

fn parse_domain(s: &str) -> Result<(String, usize), String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid domain format '{}'. Expected name:size", s));
    }

    let name = parts[0].to_string();
    let size = parts[1]
        .parse::<usize>()
        .map_err(|e| format!("Invalid domain size: {}", e))?;

    Ok((name, size))
}
