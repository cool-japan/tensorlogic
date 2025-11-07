//! Watch mode for auto-recompilation on file changes

use anyhow::{Context, Result};
use colored::*;
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::mpsc::channel;
use std::time::Duration;
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::validate_graph;

use crate::output::{print_error, print_info, print_success};
use crate::parser::parse_expression;

pub struct FileWatcher {
    context: CompilerContext,
    validate: bool,
    clear_screen: bool,
    show_timestamps: bool,
}

impl FileWatcher {
    pub fn new(
        context: CompilerContext,
        validate: bool,
        clear_screen: bool,
        show_timestamps: bool,
    ) -> Self {
        Self {
            context,
            validate,
            clear_screen,
            show_timestamps,
        }
    }

    pub fn watch(&mut self, file_path: &Path) -> Result<()> {
        print_info(&format!("Watching file: {}", file_path.display()));
        println!("Press Ctrl+C to stop\n");

        // Initial compilation
        self.compile_file(file_path)?;

        // Set up file watcher
        let (tx, rx) = channel();
        let mut watcher = RecommendedWatcher::new(tx, Config::default())?;
        watcher.watch(file_path, RecursiveMode::NonRecursive)?;

        // Watch for changes
        loop {
            match rx.recv() {
                Ok(Ok(Event { kind: _, paths, .. })) => {
                    if paths.contains(&file_path.to_path_buf()) {
                        // Debounce: wait a bit for multiple events
                        std::thread::sleep(Duration::from_millis(200));

                        // Clear pending events
                        while rx.try_recv().is_ok() {}

                        // Recompile
                        if self.clear_screen {
                            print!("\x1B[2J\x1B[1;1H");
                        }

                        if self.show_timestamps {
                            let now = chrono::Local::now();
                            println!("{}", format!("[{}]", now.format("%H:%M:%S")).bright_black());
                        }

                        if let Err(e) = self.compile_file(file_path) {
                            print_error(&format!("Compilation failed: {}", e));
                        }
                        println!();
                    }
                }
                Ok(Err(e)) => print_error(&format!("Watch error: {}", e)),
                Err(e) => {
                    print_error(&format!("Channel error: {}", e));
                    break;
                }
            }
        }

        Ok(())
    }

    fn compile_file(&mut self, file_path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        let expr = parse_expression(&content)?;
        let graph = compile_to_einsum_with_context(&expr, &mut self.context)?;

        if self.validate {
            let report = validate_graph(&graph);
            if !report.is_valid() {
                print_error("Validation failed:");
                for error in &report.errors {
                    println!("  - {}", error.message);
                }
                anyhow::bail!("Validation errors found");
            }
        }

        print_success(&format!(
            "Compiled: {} tensors, {} nodes",
            graph.tensors.len(),
            graph.nodes.len()
        ));

        Ok(())
    }
}
