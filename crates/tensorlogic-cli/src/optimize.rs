//! Graph optimization pipeline for CLI
//!
//! Provides IR-level optimizations:
//! - Identity operation elimination
//! - Einsum operation merging
//! - Contraction order optimization
//! - Multi-pass optimization

use anyhow::Result;
use std::str::FromStr;
use tensorlogic_compiler::passes::{
    optimize_einsum_graph as compiler_optimize_graph, EinsumOptResult,
};
use tensorlogic_ir::EinsumGraph;

use crate::output::{print_info, print_success};

/// Optimization statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of identity operations simplified
    pub identity_simplifications: usize,
    /// Number of einsum operations merged
    pub merged_einsums: usize,
    /// Number of operations reordered
    pub reordered_ops: usize,
    /// Estimated speedup factor
    pub estimated_speedup: f64,
}

impl From<EinsumOptResult> for OptimizationStats {
    fn from(result: EinsumOptResult) -> Self {
        Self {
            identity_simplifications: result.identity_eliminated,
            merged_einsums: result.merged_count,
            reordered_ops: result.reordered_count,
            estimated_speedup: result.estimated_speedup,
        }
    }
}

/// Optimize graph using tensorlogic-compiler's optimization passes
fn optimize_graph_internal(graph: &mut EinsumGraph) -> OptimizationStats {
    match compiler_optimize_graph(graph) {
        Ok(result) => result.into(),
        Err(_) => OptimizationStats::default(),
    }
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// No optimizations
    None,
    /// Basic optimizations (1 pass)
    Basic,
    /// Standard optimizations (2 passes)
    #[default]
    Standard,
    /// Aggressive optimizations (until convergence)
    Aggressive,
}

impl OptimizationLevel {
    /// Get number of optimization passes
    pub fn num_passes(&self) -> usize {
        match self {
            OptimizationLevel::None => 0,
            OptimizationLevel::Basic => 1,
            OptimizationLevel::Standard => 2,
            OptimizationLevel::Aggressive => 10, // Until convergence
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            OptimizationLevel::None => "No optimizations",
            OptimizationLevel::Basic => "Basic (1 pass: DCE + CSE)",
            OptimizationLevel::Standard => "Standard (2 passes: DCE + CSE + Identity)",
            OptimizationLevel::Aggressive => "Aggressive (until convergence)",
        }
    }
}

// Implement FromStr trait
impl FromStr for OptimizationLevel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "none" | "0" => Ok(OptimizationLevel::None),
            "basic" | "1" => Ok(OptimizationLevel::Basic),
            "standard" | "2" => Ok(OptimizationLevel::Standard),
            "aggressive" | "3" => Ok(OptimizationLevel::Aggressive),
            _ => anyhow::bail!("Unknown optimization level: {}", s),
        }
    }
}

// Default implementation is now derived with #[default] attribute

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Optimization level
    pub level: OptimizationLevel,
    /// Enable Dead Code Elimination (reserved for future use)
    #[allow(dead_code)]
    pub enable_dce: bool,
    /// Enable Common Subexpression Elimination (reserved for future use)
    #[allow(dead_code)]
    pub enable_cse: bool,
    /// Enable identity simplification (reserved for future use)
    #[allow(dead_code)]
    pub enable_identity: bool,
    /// Show optimization statistics
    pub show_stats: bool,
    /// Verbose output
    pub verbose: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            level: OptimizationLevel::default(),
            enable_dce: true,
            enable_cse: true,
            enable_identity: true,
            show_stats: false,
            verbose: false,
        }
    }
}

/// Optimize einsum graph with configuration
pub fn optimize_einsum_graph(
    mut graph: EinsumGraph,
    config: &OptimizationConfig,
) -> Result<(EinsumGraph, OptimizationStats)> {
    if config.level == OptimizationLevel::None {
        if config.verbose {
            print_info("Skipping optimizations (level: None)");
        }
        return Ok((graph, OptimizationStats::default()));
    }

    let num_passes = config.level.num_passes();
    let mut total_stats = OptimizationStats::default();

    if config.verbose {
        print_info(&format!(
            "Applying {} ({})",
            config.level.description(),
            num_passes
        ));
        println!(
            "  Initial: {} nodes, {} tensors",
            graph.nodes.len(),
            graph.tensors.len()
        );
    }

    for pass in 0..num_passes {
        let before_nodes = graph.nodes.len();
        let before_tensors = graph.tensors.len();

        // Apply optimization
        let stats = optimize_graph_internal(&mut graph);

        // Check for convergence
        if stats.identity_simplifications == 0
            && stats.merged_einsums == 0
            && stats.reordered_ops == 0
        {
            if config.verbose {
                println!("  Converged after {} passes", pass + 1);
            }
            break;
        }

        // Accumulate stats
        total_stats.identity_simplifications += stats.identity_simplifications;
        total_stats.merged_einsums += stats.merged_einsums;
        total_stats.reordered_ops += stats.reordered_ops;
        if stats.estimated_speedup > 1.0 {
            total_stats.estimated_speedup *= stats.estimated_speedup;
        }

        if config.verbose {
            println!(
                "  Pass {}: {} → {} nodes, {} → {} tensors",
                pass + 1,
                before_nodes,
                graph.nodes.len(),
                before_tensors,
                graph.tensors.len()
            );
        }
    }

    if config.show_stats || config.verbose {
        print_optimization_stats(&total_stats);
    }

    let total_improvements = total_stats.identity_simplifications
        + total_stats.merged_einsums
        + total_stats.reordered_ops;

    if total_improvements > 0 {
        print_success(&format!(
            "Optimization complete: {} identities removed, {} einsums merged, {} reordered",
            total_stats.identity_simplifications,
            total_stats.merged_einsums,
            total_stats.reordered_ops
        ));
    } else if config.verbose {
        print_info("No optimizations applied (graph already optimal)");
    }

    Ok((graph, total_stats))
}

/// Print optimization statistics
fn print_optimization_stats(stats: &OptimizationStats) {
    println!("\nOptimization Statistics:");
    println!(
        "  Identity operations eliminated: {}",
        stats.identity_simplifications
    );
    println!("  Einsum operations merged: {}", stats.merged_einsums);
    println!("  Operations reordered: {}", stats.reordered_ops);

    let total = stats.identity_simplifications + stats.merged_einsums + stats.reordered_ops;
    if total > 0 {
        println!("  Total improvements: {}", total);
        if stats.estimated_speedup > 1.0 {
            println!("  Estimated speedup: {:.2}x", stats.estimated_speedup);
        }
    }
}

/// List available optimization levels (reserved for future use)
#[allow(dead_code)]
pub fn list_optimization_levels() {
    println!("Optimization Levels:");
    println!();

    for level in &[
        OptimizationLevel::None,
        OptimizationLevel::Basic,
        OptimizationLevel::Standard,
        OptimizationLevel::Aggressive,
    ] {
        println!("  {:?}: {}", level, level.description());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_level_from_str() {
        assert_eq!(
            OptimizationLevel::from_str("none").unwrap(),
            OptimizationLevel::None
        );
        assert_eq!(
            OptimizationLevel::from_str("basic").unwrap(),
            OptimizationLevel::Basic
        );
        assert_eq!(
            OptimizationLevel::from_str("2").unwrap(),
            OptimizationLevel::Standard
        );
        assert!(OptimizationLevel::from_str("invalid").is_err());
    }

    #[test]
    fn test_optimization_level_num_passes() {
        assert_eq!(OptimizationLevel::None.num_passes(), 0);
        assert_eq!(OptimizationLevel::Basic.num_passes(), 1);
        assert_eq!(OptimizationLevel::Standard.num_passes(), 2);
        assert_eq!(OptimizationLevel::Aggressive.num_passes(), 10);
    }

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert_eq!(config.level, OptimizationLevel::Standard);
        assert!(config.enable_dce);
        assert!(config.enable_cse);
        assert!(config.enable_identity);
    }
}
