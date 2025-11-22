//! Batch processing mode for TensorLogic CLI with parallel execution support

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::validate_graph;

use crate::output::{print_error, print_success};
use crate::parser::parse_expression;

pub struct BatchProcessor {
    context: CompilerContext,
    validate: bool,
    parallel: bool,
    num_threads: Option<usize>,
}

impl BatchProcessor {
    /// Create a new batch processor with sequential processing
    pub fn new(context: CompilerContext, validate: bool) -> Self {
        Self {
            context,
            validate,
            parallel: false,
            num_threads: None,
        }
    }

    /// Create a new batch processor with parallel processing enabled
    #[allow(dead_code)]
    pub fn with_parallelism(
        context: CompilerContext,
        validate: bool,
        num_threads: Option<usize>,
    ) -> Self {
        Self {
            context,
            validate,
            parallel: true,
            num_threads,
        }
    }

    /// Enable parallel processing
    #[allow(dead_code)]
    pub fn enable_parallel(&mut self, num_threads: Option<usize>) {
        self.parallel = true;
        self.num_threads = num_threads;
    }

    /// Disable parallel processing
    #[allow(dead_code)]
    pub fn disable_parallel(&mut self) {
        self.parallel = false;
        self.num_threads = None;
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
        if self.parallel {
            self.process_expressions_parallel(expressions)
        } else {
            self.process_expressions_sequential(expressions)
        }
    }

    /// Process expressions sequentially (original implementation)
    fn process_expressions_sequential(&mut self, expressions: &[String]) -> Result<BatchResult> {
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

    /// Process expressions in parallel using rayon
    fn process_expressions_parallel(&self, expressions: &[String]) -> Result<BatchResult> {
        let total = expressions.len();

        // Configure thread pool if requested
        if let Some(num_threads) = self.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .ok(); // Ignore error if already initialized
        }

        let pb = Arc::new(ProgressBar::new(total as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} (parallel)",
                )
                .unwrap()
                .progress_chars("##-"),
        );

        // Use Arc<Mutex<>> for thread-safe accumulation
        let successes = Arc::new(Mutex::new(0usize));
        let failures = Arc::new(Mutex::new(Vec::new()));

        // Process in parallel
        expressions
            .par_iter()
            .enumerate()
            .for_each(|(i, expr_str)| {
                let pb_clone = Arc::clone(&pb);
                pb_clone.set_message(format!("Processing expression {}", i + 1));

                // Each thread needs its own context
                let mut local_context = self.context.clone();

                match self.process_one_with_context(expr_str, &mut local_context) {
                    Ok(_) => {
                        let mut succ = successes.lock().unwrap();
                        *succ += 1;
                    }
                    Err(e) => {
                        let mut fails = failures.lock().unwrap();
                        fails.push((i + 1, expr_str.clone(), e.to_string()));
                    }
                }

                pb_clone.inc(1);
            });

        pb.finish_with_message("Done");

        let final_successes = *successes.lock().unwrap();
        let final_failures = failures.lock().unwrap().clone();

        Ok(BatchResult {
            total,
            successes: final_successes,
            failures: final_failures,
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

    fn process_one_with_context(
        &self,
        expr_str: &str,
        context: &mut CompilerContext,
    ) -> Result<()> {
        let expr = parse_expression(expr_str)?;
        let graph = compile_to_einsum_with_context(&expr, context)?;

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
        print_success(&format!("  Successes: {}", self.successes));

        if !self.failures.is_empty() {
            print_error(&format!("  Failures: {}", self.failures.len()));
            println!("\nFailed expressions:");
            for (line_num, expr, error) in &self.failures {
                println!("  Line {}: {}", line_num, expr);
                println!("    Error: {}", error);
            }
        }
    }

    /// Check if all expressions succeeded
    #[allow(dead_code)]
    pub fn all_succeeded(&self) -> bool {
        self.failures.is_empty()
    }

    /// Get success rate as percentage
    #[allow(dead_code)]
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.successes as f64 / self.total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_compiler::CompilationConfig;

    #[test]
    fn test_batch_processor_sequential() {
        let context = CompilerContext::with_config(CompilationConfig::soft_differentiable());
        let mut processor = BatchProcessor::new(context, false);

        let expressions = vec!["pred(x, y)".to_string(), "AND(a, b)".to_string()];

        let result = processor.process_expressions(&expressions).unwrap();

        assert_eq!(result.total, 2);
        assert!(result.successes > 0);
    }

    #[test]
    fn test_batch_processor_parallel() {
        let context = CompilerContext::with_config(CompilationConfig::soft_differentiable());
        let mut processor = BatchProcessor::with_parallelism(context, false, Some(2));

        let expressions = vec![
            "pred(x, y)".to_string(),
            "AND(a, b)".to_string(),
            "OR(p, q)".to_string(),
        ];

        let result = processor.process_expressions(&expressions).unwrap();

        assert_eq!(result.total, 3);
        assert!(result.successes > 0);
    }

    #[test]
    fn test_batch_result_metrics() {
        let result = BatchResult {
            total: 10,
            successes: 8,
            failures: vec![(1, "bad".to_string(), "error".to_string())],
        };

        assert!(!result.all_succeeded());
        assert_eq!(result.success_rate(), 80.0);
    }
}
