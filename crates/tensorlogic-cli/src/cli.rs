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
