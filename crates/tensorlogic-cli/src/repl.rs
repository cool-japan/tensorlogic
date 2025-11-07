//! Interactive REPL mode for TensorLogic CLI

use anyhow::{Context, Result};
use colored::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::validate_graph;

use crate::analysis::GraphMetrics;
use crate::config::Config;
use crate::output::{print_error, print_header, print_info, print_success};
use crate::parser::parse_expression;

pub struct Repl {
    context: CompilerContext,
    config: Config,
    history_path: PathBuf,
    editor: DefaultEditor,
}

impl Repl {
    pub fn new(config: Config, context: CompilerContext) -> Result<Self> {
        let history_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(&config.repl.history_file);

        let mut editor = DefaultEditor::new()?;

        // Load history if it exists
        if history_path.exists() {
            let _ = editor.load_history(&history_path);
        }

        Ok(Self {
            context,
            config,
            history_path,
            editor,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        print_header("TensorLogic Interactive REPL");
        println!("Type '.help' for available commands, '.exit' to quit\n");

        loop {
            let readline = self.editor.readline(&self.config.repl.prompt);

            match readline {
                Ok(line) => {
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    // Add to history
                    let _ = self.editor.add_history_entry(line);

                    // Handle commands
                    if line.starts_with('.') {
                        if let Err(e) = self.handle_command(line) {
                            print_error(&format!("Command error: {}", e));
                        }
                        continue;
                    }

                    // Compile and execute expression
                    if let Err(e) = self.process_expression(line) {
                        print_error(&format!("Error: {}", e));
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("^D");
                    break;
                }
                Err(err) => {
                    print_error(&format!("Read error: {}", err));
                    break;
                }
            }
        }

        // Save history
        if self.config.repl.auto_save {
            let _ = self.editor.save_history(&self.history_path);
        }

        println!("\nGoodbye!");
        Ok(())
    }

    fn handle_command(&mut self, cmd: &str) -> Result<()> {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        let command = parts[0];

        match command {
            ".help" | ".h" => self.show_help(),
            ".exit" | ".quit" | ".q" => std::process::exit(0),
            ".clear" | ".cls" => {
                print!("\x1B[2J\x1B[1;1H"); // Clear screen
                Ok(())
            }
            ".context" | ".ctx" => self.show_context(),
            ".domain" => {
                if parts.len() < 3 {
                    print_error("Usage: .domain <name> <size>");
                    return Ok(());
                }
                let name = parts[1];
                let size: usize = parts[2].parse().context("Invalid domain size")?;
                self.context.add_domain(name, size);
                print_success(&format!("Added domain '{}' with size {}", name, size));
                Ok(())
            }
            ".strategy" | ".strat" => {
                if parts.len() < 2 {
                    println!("Current strategy: {}", self.config.strategy);
                    return Ok(());
                }
                self.config.strategy = parts[1].to_string();
                print_success(&format!("Strategy set to: {}", self.config.strategy));
                Ok(())
            }
            ".validate" => {
                self.config.validate = !self.config.validate;
                print_info(&format!(
                    "Validation: {}",
                    if self.config.validate { "ON" } else { "OFF" }
                ));
                Ok(())
            }
            ".debug" => {
                self.config.debug = !self.config.debug;
                print_info(&format!(
                    "Debug mode: {}",
                    if self.config.debug { "ON" } else { "OFF" }
                ));
                Ok(())
            }
            ".history" => {
                for (i, entry) in self.editor.history().iter().enumerate() {
                    println!("{:4}: {}", i + 1, entry);
                }
                Ok(())
            }
            _ => {
                print_error(&format!("Unknown command: {}", command));
                println!("Type '.help' for available commands");
                Ok(())
            }
        }
    }

    fn show_help(&self) -> Result<()> {
        println!("\n{}", "Available Commands:".cyan().bold());
        println!("  .help, .h           Show this help message");
        println!("  .exit, .quit, .q    Exit the REPL");
        println!("  .clear, .cls        Clear the screen");
        println!("  .context, .ctx      Show compiler context");
        println!("  .domain <name> <size> Add a domain");
        println!("  .strategy [name]    Show or set compilation strategy");
        println!("  .validate           Toggle validation mode");
        println!("  .debug              Toggle debug mode");
        println!("  .history            Show command history");
        println!("\n{}", "Expression Syntax:".cyan().bold());
        println!("  Predicates:   knows(x, y)");
        println!("  Logical:      p AND q, p OR q, NOT p, p -> q");
        println!("  Quantifiers:  EXISTS x IN D. p(x)");
        println!("              FORALL x IN D. p(x)");
        println!("  Arithmetic:   x + y, x - y, x * y, x / y");
        println!("  Comparisons:  x < y, x <= y, x = y, x >= y, x > y");
        println!("  Conditional:  IF cond THEN x ELSE y");
        println!();
        Ok(())
    }

    fn show_context(&self) -> Result<()> {
        println!("\n{}", "Compiler Context:".cyan().bold());
        println!("Strategy: {}", self.config.strategy.green());
        println!("Domains:");
        for (name, domain_info) in &self.context.domains {
            println!(
                "  {} -> cardinality: {}",
                name.yellow(),
                domain_info.cardinality
            );
        }
        println!();
        Ok(())
    }

    fn process_expression(&mut self, input: &str) -> Result<()> {
        // Parse expression
        let expr = parse_expression(input)?;

        if self.config.debug {
            println!("{}", "Parsed expression:".bright_black());
            println!("  {:?}", expr);
        }

        // Compile
        let graph = compile_to_einsum_with_context(&expr, &mut self.context)
            .context("Compilation failed")?;

        if self.config.debug {
            println!("{}", "Compiled graph:".bright_black());
            println!(
                "  {} tensors, {} nodes",
                graph.tensors.len(),
                graph.nodes.len()
            );
        }

        // Validate if enabled
        if self.config.validate {
            let report = validate_graph(&graph);
            if !report.is_valid() {
                print_error("Validation failed:");
                for error in &report.errors {
                    println!("  - {}", error.message.red());
                }
                return Ok(());
            }
        }

        // Print success
        print_success("Compilation successful");

        // Show metrics
        let metrics = GraphMetrics::analyze(&graph);
        println!(
            "  {} tensors, {} nodes, depth {}",
            metrics.tensor_count.to_string().green(),
            metrics.node_count.to_string().cyan(),
            metrics.depth.to_string().yellow()
        );

        Ok(())
    }
}
