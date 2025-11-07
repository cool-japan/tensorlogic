//! Batch processing mode for TensorLogic CLI

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::Path;
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::validate_graph;

use crate::output::{print_error, print_success};
use crate::parser::parse_expression;

pub struct BatchProcessor {
    context: CompilerContext,
    validate: bool,
}

impl BatchProcessor {
    pub fn new(context: CompilerContext, validate: bool) -> Self {
        Self { context, validate }
    }

    pub fn process_file(&mut self, input_path: &Path) -> Result<BatchResult> {
        let content = fs::read_to_string(input_path)
            .with_context(|| format!("Failed to read file: {}", input_path.display()))?;

        let expressions: Vec<String> = content
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
            .map(|s| s.to_string())
            .collect();

        self.process_expressions(&expressions)
    }

    pub fn process_expressions(&mut self, expressions: &[String]) -> Result<BatchResult> {
        let total = expressions.len();
        let mut successes = 0;
        let mut failures = Vec::new();

        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        for (i, expr_str) in expressions.iter().enumerate() {
            pb.set_message(format!("Processing expression {}", i + 1));

            match self.process_one(expr_str) {
                Ok(_) => successes += 1,
                Err(e) => failures.push((i + 1, expr_str.clone(), e.to_string())),
            }

            pb.inc(1);
        }

        pb.finish_with_message("Done");

        Ok(BatchResult {
            total,
            successes,
            failures,
        })
    }

    fn process_one(&mut self, expr_str: &str) -> Result<()> {
        let expr = parse_expression(expr_str)?;
        let graph = compile_to_einsum_with_context(&expr, &mut self.context)?;

        if self.validate {
            let report = validate_graph(&graph);
            if !report.is_valid() {
                anyhow::bail!("Validation failed: {:?}", report.errors);
            }
        }

        Ok(())
    }
}

pub struct BatchResult {
    pub total: usize,
    pub successes: usize,
    pub failures: Vec<(usize, String, String)>,
}

impl BatchResult {
    pub fn print_summary(&self) {
        println!("\nBatch Processing Summary:");
        println!("  Total: {}", self.total);
        print_success(&format!("Successes: {}", self.successes));

        if !self.failures.is_empty() {
            print_error(&format!("Failures: {}", self.failures.len()));
            println!("\nFailed expressions:");
            for (line_num, expr, error) in &self.failures {
                println!("  Line {}: {}", line_num, expr);
                println!("    Error: {}", error);
            }
        }
    }
}
