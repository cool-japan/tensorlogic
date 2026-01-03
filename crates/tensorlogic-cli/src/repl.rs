//! Interactive REPL mode for TensorLogic CLI

use anyhow::{Context, Result};
use colored::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use std::str::FromStr;
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationCache, CompilerContext};
use tensorlogic_ir::{validate_graph, EinsumGraph};

use crate::analysis::GraphMetrics;
use crate::config::Config;
use crate::executor::{Backend, CliExecutor, ExecutionConfig};
use crate::macros::{parse_macro_definition, MacroRegistry};
use crate::optimize::{optimize_einsum_graph, OptimizationConfig, OptimizationLevel};
use crate::output::{print_error, print_header, print_info, print_success};
use crate::parser::parse_expression;
use crate::profile::{ProfileConfig, Profiler};

pub struct Repl {
    context: CompilerContext,
    config: Config,
    history_path: PathBuf,
    editor: DefaultEditor,
    last_graph: Option<EinsumGraph>,
    backend: Backend,
    cache: Option<CompilationCache>,
    macros: MacroRegistry,
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

        // Initialize compilation cache if enabled
        let cache = if config.cache.enabled {
            Some(CompilationCache::new(config.cache.max_entries))
        } else {
            None
        };

        // Initialize macro registry with built-ins and config macros
        let mut macros = MacroRegistry::with_builtins();
        for macro_def in &config.macros {
            let _ = macros.define(macro_def.clone());
        }

        Ok(Self {
            context,
            config,
            history_path,
            editor,
            last_graph: None,
            backend: Backend::default(),
            cache,
            macros,
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
            ".backend" => {
                if parts.len() < 2 {
                    println!("Current backend: {}", self.backend.name());
                    println!("\nAvailable backends:");
                    for backend in Backend::available_backends() {
                        let mark = if backend == self.backend { "* " } else { "  " };
                        println!("{}  {}", mark, backend.name());
                    }
                    return Ok(());
                }
                match Backend::from_str(parts[1]) {
                    Ok(backend) => {
                        self.backend = backend;
                        print_success(&format!("Backend set to: {}", self.backend.name()));
                    }
                    Err(e) => {
                        print_error(&format!("Invalid backend: {}", e));
                    }
                }
                Ok(())
            }
            ".cache" => {
                if let Some(cache) = &self.cache {
                    let stats = cache.stats();
                    println!("\nCache Statistics:");
                    println!("  Current size: {}", stats.current_size);
                    println!("  Hits: {}", stats.hits);
                    println!("  Misses: {}", stats.misses);
                    println!("  Evictions: {}", stats.evictions);
                    println!("  Hit rate: {:.1}%", stats.hit_rate() * 100.0);
                } else {
                    println!("Cache is disabled");
                }
                Ok(())
            }
            ".clearcache" => {
                if let Some(cache) = &self.cache {
                    cache.clear();
                    print_success("Cache cleared");
                } else {
                    println!("Cache is disabled");
                }
                Ok(())
            }
            ".macro" => {
                if parts.len() < 2 {
                    print_error("Usage: .macro DEFINE MACRO name(params) = body");
                    println!("Example: .macro DEFINE MACRO trans(R, x, z) = EXISTS y. (R(x, y) AND R(y, z))");
                    return Ok(());
                }

                // Join all parts after ".macro" to get the full definition
                let definition = parts[1..].join(" ");

                match parse_macro_definition(&definition) {
                    Ok(macro_def) => {
                        let name = macro_def.name.clone();
                        self.macros.define(macro_def)?;
                        print_success(&format!("Macro '{}' defined successfully", name));
                    }
                    Err(e) => {
                        print_error(&format!("Failed to define macro: {}", e));
                    }
                }
                Ok(())
            }
            ".macros" => {
                let macros = self.macros.list();
                if macros.is_empty() {
                    println!("No macros defined");
                } else {
                    println!("\n{}", "Defined Macros:".cyan().bold());
                    for macro_def in macros {
                        println!(
                            "  {}({}) = {}",
                            macro_def.name.yellow(),
                            macro_def.params.join(", "),
                            macro_def.body.bright_black()
                        );
                    }
                    println!();
                }
                Ok(())
            }
            ".delmacro" => {
                if parts.len() < 2 {
                    print_error("Usage: .delmacro <name>");
                    return Ok(());
                }
                let name = parts[1];
                if self.macros.undefine(name).is_some() {
                    print_success(&format!("Macro '{}' removed", name));
                } else {
                    print_error(&format!("Macro '{}' not found", name));
                }
                Ok(())
            }
            ".expandmacro" | ".expand" => {
                if parts.len() < 2 {
                    print_error("Usage: .expandmacro <expression>");
                    return Ok(());
                }
                let expr = parts[1..].join(" ");
                match self.macros.expand_all(&expr) {
                    Ok(expanded) => {
                        println!("\n{}:", "Original".cyan());
                        println!("  {}", expr);
                        println!("\n{}:", "Expanded".green());
                        println!("  {}", expanded);
                        println!();
                    }
                    Err(e) => {
                        print_error(&format!("Expansion failed: {}", e));
                    }
                }
                Ok(())
            }
            ".execute" | ".exec" | ".run" => {
                if self.last_graph.is_none() {
                    print_error("No compiled graph available. Compile an expression first.");
                    return Ok(());
                }

                let show_metrics = parts.len() > 1 && parts[1] == "--metrics";
                self.execute_last_graph(show_metrics)
            }
            ".optimize" | ".opt" => {
                if self.last_graph.is_none() {
                    print_error("No compiled graph available. Compile an expression first.");
                    return Ok(());
                }

                let level_str = if parts.len() > 1 {
                    parts[1]
                } else {
                    "standard"
                };
                let show_stats = parts.contains(&"--stats");
                self.optimize_last_graph(level_str, show_stats)
            }
            ".profile" | ".prof" => {
                if parts.len() < 2 {
                    print_error("Usage: .profile <expression>");
                    println!("  Options: --json, --no-opt");
                    return Ok(());
                }

                // Parse options
                let json_output = parts.contains(&"--json");
                let include_opt = !parts.contains(&"--no-opt");

                // Get expression (skip command and options)
                let expr_parts: Vec<&str> = parts[1..]
                    .iter()
                    .filter(|p| !p.starts_with("--"))
                    .copied()
                    .collect();
                let expr_str = expr_parts.join(" ");

                self.profile_expression(&expr_str, json_output, include_opt)
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
        println!("  .help, .h              Show this help message");
        println!("  .exit, .quit, .q       Exit the REPL");
        println!("  .clear, .cls           Clear the screen");
        println!("  .context, .ctx         Show compiler context");
        println!("  .domain <name> <size>  Add a domain");
        println!("  .strategy [name]       Show or set compilation strategy");
        println!("  .validate              Toggle validation mode");
        println!("  .debug                 Toggle debug mode");
        println!("  .history               Show command history");
        println!("\n{}", "Execution & Optimization:".cyan().bold());
        println!("  .backend [name]        Show or set execution backend");
        println!("                         (cpu, parallel, profiled)");
        println!("  .execute [--metrics]   Execute last compiled graph");
        println!("  .exec, .run            Aliases for .execute");
        println!("  .optimize [level]      Optimize last compiled graph");
        println!("  .opt                   Alias for .optimize");
        println!("                         levels: none, basic, standard, aggressive");
        println!("  .profile <expr>        Profile compilation of expression");
        println!("  .prof                  Alias for .profile");
        println!("                         options: --json, --no-opt");
        println!("\n{}", "Cache:".cyan().bold());
        println!("  .cache                 Show cache statistics");
        println!("  .clearcache            Clear the compilation cache");
        println!("\n{}", "Macros:".cyan().bold());
        println!("  .macro <definition>    Define a new macro");
        println!("                         Example: .macro DEFINE MACRO trans(R,x,z) = EXISTS y. (R(x,y) AND R(y,z))");
        println!("  .macros                List all defined macros");
        println!("  .delmacro <name>       Remove a macro definition");
        println!("  .expandmacro <expr>    Show macro expansion of expression");
        println!("  .expand                Alias for .expandmacro");
        println!("\n{}", "Expression Syntax:".cyan().bold());
        println!("  Predicates:   knows(x, y)");
        println!("  Logical:      p AND q, p OR q, NOT p, p -> q");
        println!("  Quantifiers:  EXISTS x IN D. p(x)");
        println!("                FORALL x IN D. p(x)");
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
        // Expand macros first
        let expanded = self.macros.expand_all(input)?;

        if self.config.debug && expanded != input {
            println!("{}", "After macro expansion:".bright_black());
            println!("  {}", expanded);
        }

        // Parse expression
        let expr = parse_expression(&expanded)?;

        if self.config.debug {
            println!("{}", "Parsed expression:".bright_black());
            println!("  {:?}", expr);
        }

        // Compile (with cache if enabled)
        let prev_hits = self.cache.as_ref().map(|c| c.stats().hits).unwrap_or(0);

        let graph = if let Some(cache) = &self.cache {
            cache
                .get_or_compile(&expr, &mut self.context, |expr, ctx| {
                    compile_to_einsum_with_context(expr, ctx)
                })
                .context("Compilation failed")?
        } else {
            // Compile without cache
            compile_to_einsum_with_context(&expr, &mut self.context)
                .context("Compilation failed")?
        };

        let cache_hit = self
            .cache
            .as_ref()
            .map(|c| c.stats().hits > prev_hits)
            .unwrap_or(false);

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
        if cache_hit {
            print_success("Cache hit - using cached result");
        } else {
            print_success("Compilation successful");
        }

        // Show metrics
        let metrics = GraphMetrics::analyze(&graph);
        println!(
            "  {} tensors, {} nodes, depth {}",
            metrics.tensor_count.to_string().green(),
            metrics.node_count.to_string().cyan(),
            metrics.depth.to_string().yellow()
        );

        // Save graph for execute/optimize commands
        self.last_graph = Some(graph);

        Ok(())
    }

    fn execute_last_graph(&mut self, show_metrics: bool) -> Result<()> {
        let graph = self.last_graph.as_ref().unwrap();

        print_info(&format!(
            "Executing with {} backend...",
            self.backend.name()
        ));

        let exec_config = ExecutionConfig {
            backend: self.backend,
            device: tensorlogic_scirs_backend::DeviceType::Cpu,
            show_metrics,
            show_intermediates: false,
            validate_shapes: true,
            trace: false,
        };

        let executor = CliExecutor::new(exec_config.clone())?;
        let result = executor.execute(graph)?;

        result.print_summary(&exec_config);

        Ok(())
    }

    fn optimize_last_graph(&mut self, level_str: &str, show_stats: bool) -> Result<()> {
        let graph = self.last_graph.take().unwrap();

        let level = OptimizationLevel::from_str(level_str)?;

        print_info(&format!("Optimizing with {} level...", level_str));

        let opt_config = OptimizationConfig {
            level,
            enable_dce: true,
            enable_cse: true,
            enable_identity: true,
            show_stats,
            verbose: self.config.debug,
        };

        let (optimized_graph, stats) = optimize_einsum_graph(graph, &opt_config)?;

        // Show improvement metrics
        let metrics = GraphMetrics::analyze(&optimized_graph);
        println!(
            "  {} tensors, {} nodes, depth {}",
            metrics.tensor_count.to_string().green(),
            metrics.node_count.to_string().cyan(),
            metrics.depth.to_string().yellow()
        );

        if show_stats || self.config.debug {
            println!("\nOptimization Impact:");
            println!("  Identity eliminated: {}", stats.identity_simplifications);
            println!("  Einsums merged: {}", stats.merged_einsums);
            println!("  Operations reordered: {}", stats.reordered_ops);
            if stats.estimated_speedup > 1.0 {
                println!("  Estimated speedup: {:.2}x", stats.estimated_speedup);
            }
        }

        // Save optimized graph
        self.last_graph = Some(optimized_graph);

        Ok(())
    }

    fn profile_expression(
        &self,
        expr_str: &str,
        json_output: bool,
        include_opt: bool,
    ) -> Result<()> {
        // Parse expression
        let expr = parse_expression(expr_str)?;

        // Create profile config
        let profile_config = ProfileConfig {
            include_optimization: include_opt,
            optimization_level: OptimizationLevel::Standard,
            include_validation: self.config.validate,
            detailed: true,
            warmup_runs: 1,
            profile_runs: 3,
            include_execution: false,
            execution_backend: "cpu".to_string(),
        };

        // Run profiler
        let profiler = Profiler::new(profile_config);
        let profile_data = profiler.profile(&expr, &self.context)?;

        // Output results
        if json_output {
            println!("{}", serde_json::to_string_pretty(&profile_data.to_json())?);
        } else {
            profile_data.print();
        }

        Ok(())
    }
}
