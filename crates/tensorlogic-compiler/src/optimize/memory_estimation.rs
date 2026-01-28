//! Memory estimation for tensor expressions.
//!
//! This module provides tools for estimating the memory footprint of
//! compiled tensor expressions. This is useful for:
//!
//! - Planning batch sizes
//! - Deciding on execution strategies
//! - GPU memory allocation
//! - Optimization decisions
//!
//! # Usage
//!
//! ```
//! use tensorlogic_compiler::optimize::{estimate_memory, MemoryEstimate};
//! use tensorlogic_compiler::CompilerContext;
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! let mut ctx = CompilerContext::new();
//! ctx.add_domain("Person", 1000);
//! ctx.add_domain("Thing", 500);
//!
//! let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
//! let estimate = estimate_memory(&expr, &ctx);
//!
//! println!("Estimated memory: {} bytes", estimate.total_bytes);
//! println!("Peak memory: {} bytes", estimate.peak_bytes);
//! ```

use crate::CompilerContext;
use tensorlogic_ir::TLExpr;

/// Detailed memory estimate for an expression.
#[derive(Debug, Clone, Default)]
pub struct MemoryEstimate {
    /// Total memory needed for all tensors (in bytes)
    pub total_bytes: usize,
    /// Peak memory usage during execution (in bytes)
    pub peak_bytes: usize,
    /// Number of intermediate tensors
    pub intermediate_count: usize,
    /// Maximum tensor size (in elements)
    pub max_tensor_size: usize,
    /// Total number of elements across all tensors
    pub total_elements: usize,
}

impl MemoryEstimate {
    /// Get total memory in megabytes.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory in megabytes.
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Check if this exceeds a memory limit (in bytes).
    pub fn exceeds_limit(&self, limit_bytes: usize) -> bool {
        self.peak_bytes > limit_bytes
    }

    /// Suggest optimal batch size given a memory budget.
    pub fn suggest_batch_size(&self, budget_bytes: usize, current_batch: usize) -> usize {
        if self.peak_bytes == 0 {
            return current_batch;
        }

        let memory_per_item = self.peak_bytes / current_batch.max(1);
        if memory_per_item == 0 {
            return current_batch;
        }

        (budget_bytes / memory_per_item).max(1)
    }
}

impl std::fmt::Display for MemoryEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Estimate:")?;
        writeln!(
            f,
            "  Total: {:.2} MB ({} bytes)",
            self.total_mb(),
            self.total_bytes
        )?;
        writeln!(
            f,
            "  Peak: {:.2} MB ({} bytes)",
            self.peak_mb(),
            self.peak_bytes
        )?;
        writeln!(f, "  Intermediates: {}", self.intermediate_count)?;
        writeln!(f, "  Max tensor size: {} elements", self.max_tensor_size)?;
        writeln!(f, "  Total elements: {}", self.total_elements)?;
        Ok(())
    }
}

/// Estimate memory usage for an expression.
///
/// This function analyzes the expression and estimates memory usage
/// based on domain sizes in the compiler context.
///
/// # Arguments
///
/// * `expr` - The expression to analyze
/// * `ctx` - The compiler context containing domain sizes
///
/// # Returns
///
/// A memory estimate
pub fn estimate_memory(expr: &TLExpr, ctx: &CompilerContext) -> MemoryEstimate {
    let mut estimate = MemoryEstimate::default();
    let mut current_memory = 0usize;

    estimate_memory_impl(expr, ctx, &mut estimate, &mut current_memory);

    // Ensure peak is at least total
    estimate.peak_bytes = estimate.peak_bytes.max(estimate.total_bytes);

    estimate
}

fn estimate_memory_impl(
    expr: &TLExpr,
    ctx: &CompilerContext,
    estimate: &mut MemoryEstimate,
    current_memory: &mut usize,
) -> usize {
    // Size of f64 in bytes
    const ELEM_SIZE: usize = 8;

    match expr {
        TLExpr::Pred { args, .. } => {
            // Estimate tensor size from argument domains
            let mut size = 1usize;
            for arg in args {
                if let tensorlogic_ir::Term::Var(v) = arg {
                    // Get domain size for this variable
                    if let Some(domain_name) = ctx.var_to_domain.get(v) {
                        if let Some(info) = ctx.domains.get(domain_name) {
                            size = size.saturating_mul(info.cardinality);
                        }
                    } else {
                        // Unknown domain, assume default size
                        size = size.saturating_mul(100);
                    }
                }
            }

            let bytes = size.saturating_mul(ELEM_SIZE);
            estimate.total_bytes = estimate.total_bytes.saturating_add(bytes);
            estimate.total_elements = estimate.total_elements.saturating_add(size);
            estimate.max_tensor_size = estimate.max_tensor_size.max(size);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            bytes
        }

        TLExpr::Constant(_) => {
            // Scalar constant: 8 bytes
            let bytes = ELEM_SIZE;
            estimate.total_bytes = estimate.total_bytes.saturating_add(bytes);
            estimate.total_elements = estimate.total_elements.saturating_add(1);
            estimate.max_tensor_size = estimate.max_tensor_size.max(1);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            bytes
        }

        // Binary operations: result has same shape as operands (broadcast assumed)
        TLExpr::Add(lhs, rhs)
        | TLExpr::Sub(lhs, rhs)
        | TLExpr::Mul(lhs, rhs)
        | TLExpr::Div(lhs, rhs)
        | TLExpr::Min(lhs, rhs)
        | TLExpr::Max(lhs, rhs) => {
            let lhs_bytes = estimate_memory_impl(lhs, ctx, estimate, current_memory);
            let rhs_bytes = estimate_memory_impl(rhs, ctx, estimate, current_memory);

            // Result tensor: max of operand sizes
            let result_bytes = lhs_bytes.max(rhs_bytes);
            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            // Free intermediate tensors
            *current_memory = current_memory.saturating_sub(lhs_bytes);
            *current_memory = current_memory.saturating_sub(rhs_bytes);

            result_bytes
        }

        // Logic operations
        TLExpr::And(lhs, rhs) | TLExpr::Or(lhs, rhs) | TLExpr::Imply(lhs, rhs) => {
            let lhs_bytes = estimate_memory_impl(lhs, ctx, estimate, current_memory);
            let rhs_bytes = estimate_memory_impl(rhs, ctx, estimate, current_memory);

            let result_bytes = lhs_bytes.max(rhs_bytes);
            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(lhs_bytes);
            *current_memory = current_memory.saturating_sub(rhs_bytes);

            result_bytes
        }

        // Comparison operations
        TLExpr::Eq(lhs, rhs)
        | TLExpr::Lt(lhs, rhs)
        | TLExpr::Lte(lhs, rhs)
        | TLExpr::Gt(lhs, rhs)
        | TLExpr::Gte(lhs, rhs) => {
            let lhs_bytes = estimate_memory_impl(lhs, ctx, estimate, current_memory);
            let rhs_bytes = estimate_memory_impl(rhs, ctx, estimate, current_memory);

            let result_bytes = lhs_bytes.max(rhs_bytes);
            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(lhs_bytes);
            *current_memory = current_memory.saturating_sub(rhs_bytes);

            result_bytes
        }

        // Unary operations: same shape as input
        TLExpr::Not(inner)
        | TLExpr::Abs(inner)
        | TLExpr::Sqrt(inner)
        | TLExpr::Exp(inner)
        | TLExpr::Log(inner)
        | TLExpr::Score(inner)
        | TLExpr::Floor(inner)
        | TLExpr::Ceil(inner)
        | TLExpr::Round(inner)
        | TLExpr::Sin(inner)
        | TLExpr::Cos(inner)
        | TLExpr::Tan(inner) => {
            let inner_bytes = estimate_memory_impl(inner, ctx, estimate, current_memory);

            // Result tensor: same size as input
            estimate.total_bytes = estimate.total_bytes.saturating_add(inner_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(inner_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(inner_bytes);

            inner_bytes
        }

        TLExpr::Pow(base, exp) | TLExpr::Mod(base, exp) => {
            let base_bytes = estimate_memory_impl(base, ctx, estimate, current_memory);
            let exp_bytes = estimate_memory_impl(exp, ctx, estimate, current_memory);

            let result_bytes = base_bytes.max(exp_bytes);
            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(base_bytes);
            *current_memory = current_memory.saturating_sub(exp_bytes);

            result_bytes
        }

        // Quantifiers: reduce along one dimension
        TLExpr::Exists { var, domain, body } | TLExpr::ForAll { var, domain, body } => {
            // Get domain size for reduction
            let domain_size = ctx
                .domains
                .get(domain)
                .map(|info| info.cardinality)
                .unwrap_or(100);

            let body_bytes = estimate_memory_impl(body, ctx, estimate, current_memory);

            // Result is reduced: divide by domain size
            let result_bytes = body_bytes / domain_size.max(1);
            let result_bytes = result_bytes.max(ELEM_SIZE); // At least one element

            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            // Account for domain dimension in variable
            let _ = var; // Just to silence unused warning

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(body_bytes);

            result_bytes
        }

        TLExpr::Let { value, body, .. } => {
            let value_bytes = estimate_memory_impl(value, ctx, estimate, current_memory);
            let body_bytes = estimate_memory_impl(body, ctx, estimate, current_memory);

            // Let keeps value alive during body evaluation
            *current_memory = current_memory.saturating_sub(value_bytes);

            body_bytes
        }

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond_bytes = estimate_memory_impl(condition, ctx, estimate, current_memory);
            let then_bytes = estimate_memory_impl(then_branch, ctx, estimate, current_memory);
            let else_bytes = estimate_memory_impl(else_branch, ctx, estimate, current_memory);

            // Result is max of branches
            let result_bytes = then_bytes.max(else_bytes);
            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(cond_bytes);
            *current_memory = current_memory.saturating_sub(then_bytes);
            *current_memory = current_memory.saturating_sub(else_bytes);

            result_bytes
        }

        // Modal/Temporal: similar to quantifiers
        TLExpr::Box(inner)
        | TLExpr::Diamond(inner)
        | TLExpr::Next(inner)
        | TLExpr::Eventually(inner)
        | TLExpr::Always(inner)
        | TLExpr::FuzzyNot { expr: inner, .. }
        | TLExpr::WeightedRule { rule: inner, .. } => {
            let inner_bytes = estimate_memory_impl(inner, ctx, estimate, current_memory);

            // Typically reduces one dimension
            let result_bytes = inner_bytes;
            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(inner_bytes);

            result_bytes
        }

        TLExpr::Until { before, after } | TLExpr::WeakUntil { before, after } => {
            let lhs_bytes = estimate_memory_impl(before, ctx, estimate, current_memory);
            let rhs_bytes = estimate_memory_impl(after, ctx, estimate, current_memory);

            let result_bytes = lhs_bytes.max(rhs_bytes);
            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(lhs_bytes);
            *current_memory = current_memory.saturating_sub(rhs_bytes);

            result_bytes
        }

        TLExpr::Release { released, releaser } | TLExpr::StrongRelease { released, releaser } => {
            let lhs_bytes = estimate_memory_impl(released, ctx, estimate, current_memory);
            let rhs_bytes = estimate_memory_impl(releaser, ctx, estimate, current_memory);

            let result_bytes = lhs_bytes.max(rhs_bytes);
            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(lhs_bytes);
            *current_memory = current_memory.saturating_sub(rhs_bytes);

            result_bytes
        }

        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            let lhs_bytes = estimate_memory_impl(left, ctx, estimate, current_memory);
            let rhs_bytes = estimate_memory_impl(right, ctx, estimate, current_memory);

            let result_bytes = lhs_bytes.max(rhs_bytes);
            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(lhs_bytes);
            *current_memory = current_memory.saturating_sub(rhs_bytes);

            result_bytes
        }

        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => {
            let lhs_bytes = estimate_memory_impl(premise, ctx, estimate, current_memory);
            let rhs_bytes = estimate_memory_impl(conclusion, ctx, estimate, current_memory);

            let result_bytes = lhs_bytes.max(rhs_bytes);
            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(lhs_bytes);
            *current_memory = current_memory.saturating_sub(rhs_bytes);

            result_bytes
        }

        TLExpr::Aggregate {
            var, domain, body, ..
        }
        | TLExpr::SoftExists {
            var, domain, body, ..
        }
        | TLExpr::SoftForAll {
            var, domain, body, ..
        } => {
            let domain_size = ctx
                .domains
                .get(domain)
                .map(|info| info.cardinality)
                .unwrap_or(100);

            let body_bytes = estimate_memory_impl(body, ctx, estimate, current_memory);

            let result_bytes = body_bytes / domain_size.max(1);
            let result_bytes = result_bytes.max(ELEM_SIZE);

            estimate.total_bytes = estimate.total_bytes.saturating_add(result_bytes);
            estimate.intermediate_count += 1;

            let _ = var;

            *current_memory = current_memory.saturating_add(result_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            *current_memory = current_memory.saturating_sub(body_bytes);

            result_bytes
        }

        TLExpr::ProbabilisticChoice { alternatives } => {
            let mut max_bytes = 0;
            for (_, expr) in alternatives {
                let bytes = estimate_memory_impl(expr, ctx, estimate, current_memory);
                max_bytes = max_bytes.max(bytes);
            }

            estimate.total_bytes = estimate.total_bytes.saturating_add(max_bytes);
            estimate.intermediate_count += 1;

            *current_memory = current_memory.saturating_add(max_bytes);
            estimate.peak_bytes = estimate.peak_bytes.max(*current_memory);

            max_bytes
        }

        // All other expression types (enhancements)
        _ => {
            const ELEM_SIZE: usize = 8;
            ELEM_SIZE
        }
    }
}

/// Estimate memory for a batch of similar expressions.
///
/// Useful for planning batch execution.
pub fn estimate_batch_memory(
    expr: &TLExpr,
    ctx: &CompilerContext,
    batch_size: usize,
) -> MemoryEstimate {
    let single = estimate_memory(expr, ctx);

    MemoryEstimate {
        total_bytes: single.total_bytes.saturating_mul(batch_size),
        peak_bytes: single.peak_bytes.saturating_mul(batch_size),
        intermediate_count: single.intermediate_count,
        max_tensor_size: single.max_tensor_size.saturating_mul(batch_size),
        total_elements: single.total_elements.saturating_mul(batch_size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    #[test]
    fn test_constant_memory() {
        let ctx = CompilerContext::new();
        let expr = TLExpr::Constant(1.0);
        let estimate = estimate_memory(&expr, &ctx);

        assert_eq!(estimate.total_bytes, 8); // One f64
        assert_eq!(estimate.total_elements, 1);
    }

    #[test]
    fn test_predicate_memory() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        ctx.add_domain("Thing", 50);
        ctx.bind_var("x", "Person").unwrap();
        ctx.bind_var("y", "Thing").unwrap();

        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let estimate = estimate_memory(&expr, &ctx);

        // 100 * 50 = 5000 elements * 8 bytes = 40000 bytes
        assert_eq!(estimate.total_bytes, 40000);
        assert_eq!(estimate.max_tensor_size, 5000);
    }

    #[test]
    fn test_binary_operation_memory() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 100);
        ctx.bind_var("x", "D").unwrap();

        let a = TLExpr::pred("a", vec![Term::var("x")]);
        let b = TLExpr::pred("b", vec![Term::var("x")]);
        let expr = TLExpr::add(a, b);

        let estimate = estimate_memory(&expr, &ctx);

        // Two inputs (100 each) + one output (100) = 300 elements
        // Actually more due to intermediate tracking
        assert!(estimate.total_bytes > 0);
        assert!(estimate.intermediate_count >= 3);
    }

    #[test]
    fn test_quantifier_reduction() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("Person", 100);
        ctx.add_domain("Thing", 50);
        ctx.bind_var("x", "Person").unwrap();
        ctx.bind_var("y", "Thing").unwrap();

        let pred = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let expr = TLExpr::exists("y", "Thing", pred);

        let estimate = estimate_memory(&expr, &ctx);

        // Input: 100*50 = 5000 elements
        // Output after reduction: 100 elements
        assert!(estimate.total_bytes > 0);
        assert!(estimate.intermediate_count >= 2);
    }

    #[test]
    fn test_peak_memory() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 1000);
        ctx.bind_var("x", "D").unwrap();

        // Build deep expression tree
        let mut expr = TLExpr::pred("a", vec![Term::var("x")]);
        for i in 0..5 {
            let name = format!("b{}", i);
            let other = TLExpr::pred(&name, vec![Term::var("x")]);
            expr = TLExpr::add(expr, other);
        }

        let estimate = estimate_memory(&expr, &ctx);

        // Peak should be >= some input size
        assert!(estimate.peak_bytes > 0);
        assert!(estimate.peak_bytes >= 1000 * 8); // At least one tensor
    }

    #[test]
    fn test_batch_estimation() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 100);
        ctx.bind_var("x", "D").unwrap();

        let expr = TLExpr::pred("a", vec![Term::var("x")]);
        let single = estimate_memory(&expr, &ctx);
        let batch = estimate_batch_memory(&expr, &ctx, 10);

        assert_eq!(batch.total_bytes, single.total_bytes * 10);
        assert_eq!(batch.total_elements, single.total_elements * 10);
    }

    #[test]
    fn test_memory_limit_check() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10000);
        ctx.bind_var("x", "D").unwrap();
        ctx.bind_var("y", "D").unwrap();

        let expr = TLExpr::pred("big", vec![Term::var("x"), Term::var("y")]);
        let estimate = estimate_memory(&expr, &ctx);

        // 10000 * 10000 * 8 = 800MB
        let mb = 1024 * 1024;
        assert!(estimate.exceeds_limit(100 * mb)); // Should exceed 100MB
        assert!(!estimate.exceeds_limit(1000 * mb)); // Should not exceed 1GB
    }

    #[test]
    fn test_batch_size_suggestion() {
        let estimate = MemoryEstimate {
            total_bytes: 1000,
            peak_bytes: 1000,
            intermediate_count: 5,
            max_tensor_size: 100,
            total_elements: 500,
        };

        // With 5000 byte budget and 1000 bytes per item (assuming batch=1)
        let suggested = estimate.suggest_batch_size(5000, 1);
        assert_eq!(suggested, 5); // Can fit 5 items
    }

    #[test]
    fn test_display() {
        let estimate = MemoryEstimate {
            total_bytes: 1024 * 1024,    // 1MB
            peak_bytes: 2 * 1024 * 1024, // 2MB
            intermediate_count: 10,
            max_tensor_size: 10000,
            total_elements: 50000,
        };

        let display = format!("{}", estimate);
        assert!(display.contains("Memory Estimate"));
        assert!(display.contains("MB"));
    }

    #[test]
    fn test_nested_quantifiers() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("A", 100);
        ctx.add_domain("B", 50);
        ctx.bind_var("x", "A").unwrap();
        ctx.bind_var("y", "B").unwrap();

        let pred = TLExpr::pred("rel", vec![Term::var("x"), Term::var("y")]);
        let expr = TLExpr::exists("x", "A", TLExpr::forall("y", "B", pred));

        let estimate = estimate_memory(&expr, &ctx);

        // Should reduce both dimensions
        assert!(estimate.total_bytes > 0);
        assert!(estimate.intermediate_count >= 3);
    }

    #[test]
    fn test_mb_conversion() {
        let estimate = MemoryEstimate {
            total_bytes: 1024 * 1024 * 10, // 10MB
            peak_bytes: 1024 * 1024 * 20,  // 20MB
            ..Default::default()
        };

        assert!((estimate.total_mb() - 10.0).abs() < 0.001);
        assert!((estimate.peak_mb() - 20.0).abs() < 0.001);
    }
}
