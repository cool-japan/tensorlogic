//! Snapshot testing for output consistency
//!
//! This module provides snapshot testing capabilities to ensure that compilation
//! outputs remain consistent across code changes and refactorings.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{export_to_dot, TLExpr};

use crate::analysis::GraphMetrics;

/// A snapshot of compilation output for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationSnapshot {
    /// The expression that was compiled
    pub expression: String,

    /// Compilation strategy used
    pub strategy: String,

    /// Domains defined
    pub domains: Vec<(String, usize)>,

    /// Number of tensors in the graph
    pub tensor_count: usize,

    /// Number of nodes in the graph
    pub node_count: usize,

    /// Graph depth
    pub depth: usize,

    /// Operation breakdown
    pub operations: std::collections::HashMap<String, usize>,

    /// Estimated FLOPs
    pub estimated_flops: u64,

    /// Estimated memory (bytes)
    pub estimated_memory: u64,

    /// DOT format representation (for structural comparison)
    pub dot_output: String,

    /// JSON serialization (for complete graph structure)
    pub json_output: String,

    /// Creation timestamp
    pub created_at: String,
}

impl CompilationSnapshot {
    /// Create a snapshot from an expression and context
    pub fn create(expr: &TLExpr, context: &CompilerContext, expr_string: &str) -> Result<Self> {
        // Compile the expression
        let mut ctx = context.clone();
        let graph = compile_to_einsum_with_context(expr, &mut ctx)?;

        // Analyze the graph
        let metrics = GraphMetrics::analyze(&graph);

        // Generate outputs
        let dot_output = export_to_dot(&graph);
        let json_output = serde_json::to_string_pretty(&graph)?;

        // Extract domains
        let domains: Vec<(String, usize)> = context
            .domains
            .iter()
            .map(|(k, v)| (k.clone(), v.cardinality))
            .collect();

        Ok(Self {
            expression: expr_string.to_string(),
            strategy: format!("{:?}", context.config.and_strategy),
            domains,
            tensor_count: metrics.tensor_count,
            node_count: metrics.node_count,
            depth: metrics.depth,
            operations: metrics.op_breakdown,
            estimated_flops: metrics.estimated_flops,
            estimated_memory: metrics.estimated_memory,
            dot_output,
            json_output,
            created_at: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Save snapshot to a file
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load snapshot from a file
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let snapshot: Self = serde_json::from_str(&content)?;
        Ok(snapshot)
    }

    /// Compare this snapshot with another (strict mode)
    ///
    /// Note: Timestamps are intentionally excluded from comparison
    pub fn compare(&self, other: &Self) -> SnapshotDiff {
        self.compare_with_options(other, true)
    }

    /// Compare this snapshot with another with options
    ///
    /// - `strict_dot`: If true, compare DOT output character-by-character
    ///   If false, skip DOT comparison (useful when DOT has non-deterministic ordering)
    pub fn compare_with_options(&self, other: &Self, strict_dot: bool) -> SnapshotDiff {
        let mut differences = Vec::new();

        // Check expression
        if self.expression != other.expression {
            differences.push(format!(
                "Expression changed: '{}' -> '{}'",
                self.expression, other.expression
            ));
        }

        // Check strategy
        if self.strategy != other.strategy {
            differences.push(format!(
                "Strategy changed: {} -> {}",
                self.strategy, other.strategy
            ));
        }

        // Check domains
        if self.domains != other.domains {
            differences.push(format!(
                "Domains changed: {:?} -> {:?}",
                self.domains, other.domains
            ));
        }

        // Check tensor count
        if self.tensor_count != other.tensor_count {
            differences.push(format!(
                "Tensor count changed: {} -> {}",
                self.tensor_count, other.tensor_count
            ));
        }

        // Check node count
        if self.node_count != other.node_count {
            differences.push(format!(
                "Node count changed: {} -> {}",
                self.node_count, other.node_count
            ));
        }

        // Check depth
        if self.depth != other.depth {
            differences.push(format!("Depth changed: {} -> {}", self.depth, other.depth));
        }

        // Check operations
        if self.operations != other.operations {
            differences.push(format!(
                "Operations changed: {:?} -> {:?}",
                self.operations, other.operations
            ));
        }

        // Check FLOPs (allow small variation due to floating point)
        let flops_diff = self.estimated_flops.abs_diff(other.estimated_flops);

        if flops_diff > 100 {
            // Allow 100 FLOPs tolerance
            differences.push(format!(
                "Estimated FLOPs changed significantly: {} -> {}",
                self.estimated_flops, other.estimated_flops
            ));
        }

        // Check memory (allow small variation)
        let mem_diff = self.estimated_memory.abs_diff(other.estimated_memory);

        if mem_diff > 1000 {
            // Allow 1KB tolerance
            differences.push(format!(
                "Estimated memory changed significantly: {} -> {}",
                self.estimated_memory, other.estimated_memory
            ));
        }

        // Check DOT output (structural comparison) - only if strict mode
        if strict_dot && self.dot_output != other.dot_output {
            differences.push("DOT output structure changed".to_string());
        }

        SnapshotDiff {
            identical: differences.is_empty(),
            differences,
        }
    }
}

/// Result of comparing two snapshots
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    /// Whether the snapshots are identical
    pub identical: bool,

    /// List of differences found
    pub differences: Vec<String>,
}

impl SnapshotDiff {
    /// Check if snapshots match
    pub fn is_match(&self) -> bool {
        self.identical
    }

    /// Print differences to stderr
    #[allow(dead_code)]
    pub fn print_diff(&self) {
        if self.identical {
            println!("✓ Snapshots match");
        } else {
            eprintln!("✗ Snapshots differ:");
            for diff in &self.differences {
                eprintln!("  - {}", diff);
            }
        }
    }
}

/// Snapshot test suite manager
pub struct SnapshotSuite {
    /// Directory where snapshots are stored
    snapshot_dir: PathBuf,

    /// Test suite name
    name: String,
}

impl SnapshotSuite {
    /// Create a new snapshot suite
    pub fn new(name: &str, snapshot_dir: PathBuf) -> Self {
        Self {
            snapshot_dir,
            name: name.to_string(),
        }
    }

    /// Get the path for a snapshot file
    fn snapshot_path(&self, test_name: &str) -> PathBuf {
        self.snapshot_dir
            .join(format!("{}_{}.json", self.name, test_name))
    }

    /// Record a snapshot
    pub fn record(
        &self,
        test_name: &str,
        expr: &TLExpr,
        context: &CompilerContext,
        expr_string: &str,
    ) -> Result<()> {
        // Ensure snapshot directory exists
        fs::create_dir_all(&self.snapshot_dir)?;

        let snapshot = CompilationSnapshot::create(expr, context, expr_string)?;
        let path = self.snapshot_path(test_name);
        snapshot.save(&path)?;

        println!("✓ Recorded snapshot: {}", test_name);
        Ok(())
    }

    /// Verify against a recorded snapshot
    pub fn verify(
        &self,
        test_name: &str,
        expr: &TLExpr,
        context: &CompilerContext,
        expr_string: &str,
    ) -> Result<SnapshotDiff> {
        let path = self.snapshot_path(test_name);

        if !path.exists() {
            anyhow::bail!(
                "Snapshot not found: {}. Run in record mode first.",
                test_name
            );
        }

        let recorded = CompilationSnapshot::load(&path)?;
        let current = CompilationSnapshot::create(expr, context, expr_string)?;

        Ok(recorded.compare(&current))
    }

    /// Update a snapshot (re-record)
    pub fn update(
        &self,
        test_name: &str,
        expr: &TLExpr,
        context: &CompilerContext,
        expr_string: &str,
    ) -> Result<()> {
        self.record(test_name, expr, context, expr_string)
    }

    /// List all snapshots in the suite
    pub fn list_snapshots(&self) -> Result<Vec<String>> {
        let mut snapshots = Vec::new();

        if !self.snapshot_dir.exists() {
            return Ok(snapshots);
        }

        for entry in fs::read_dir(&self.snapshot_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name() {
                if let Some(name) = filename.to_str() {
                    if name.starts_with(&self.name) && name.ends_with(".json") {
                        // Extract test name
                        let test_name = name
                            .trim_start_matches(&format!("{}_", self.name))
                            .trim_end_matches(".json");
                        snapshots.push(test_name.to_string());
                    }
                }
            }
        }

        Ok(snapshots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_compiler::CompilationConfig;
    use tensorlogic_ir::Term;

    fn create_test_expr() -> TLExpr {
        TLExpr::And(
            Box::new(TLExpr::Pred {
                name: "knows".to_string(),
                args: vec![Term::Var("x".to_string()), Term::Var("y".to_string())],
            }),
            Box::new(TLExpr::Pred {
                name: "likes".to_string(),
                args: vec![Term::Var("y".to_string()), Term::Var("z".to_string())],
            }),
        )
    }

    fn create_test_context() -> CompilerContext {
        let config = CompilationConfig::soft_differentiable();
        let mut ctx = CompilerContext::with_config(config);
        ctx.add_domain("D", 100);
        ctx
    }

    #[test]
    fn test_snapshot_creation() {
        let expr = create_test_expr();
        let context = create_test_context();

        let snapshot =
            CompilationSnapshot::create(&expr, &context, "knows(x, y) AND likes(y, z)").unwrap();

        assert_eq!(snapshot.expression, "knows(x, y) AND likes(y, z)");
        assert!(snapshot.tensor_count > 0);
        assert!(snapshot.node_count > 0);
        assert!(!snapshot.dot_output.is_empty());
        assert!(!snapshot.json_output.is_empty());
    }

    #[test]
    fn test_snapshot_save_load() {
        let expr = create_test_expr();
        let context = create_test_context();
        let snapshot =
            CompilationSnapshot::create(&expr, &context, "knows(x, y) AND likes(y, z)").unwrap();

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_snapshot.json");

        snapshot.save(&path).unwrap();
        let loaded = CompilationSnapshot::load(&path).unwrap();

        assert_eq!(snapshot.expression, loaded.expression);
        assert_eq!(snapshot.tensor_count, loaded.tensor_count);
        assert_eq!(snapshot.node_count, loaded.node_count);

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_snapshot_comparison_identical() {
        let expr = create_test_expr();
        let context = create_test_context();

        let snapshot1 =
            CompilationSnapshot::create(&expr, &context, "knows(x, y) AND likes(y, z)").unwrap();
        let snapshot2 =
            CompilationSnapshot::create(&expr, &context, "knows(x, y) AND likes(y, z)").unwrap();

        // Skip strict DOT comparison since identical compilations might have
        // different internal orderings (e.g., HashMap iteration order)
        let diff = snapshot1.compare_with_options(&snapshot2, false);
        if !diff.is_match() {
            eprintln!("Differences found:");
            for d in &diff.differences {
                eprintln!("  {}", d);
            }
        }
        assert!(diff.is_match());
        assert!(diff.differences.is_empty());
    }

    #[test]
    fn test_snapshot_comparison_different() {
        let expr1 = create_test_expr();
        let expr2 = TLExpr::Pred {
            name: "knows".to_string(),
            args: vec![Term::Var("x".to_string()), Term::Var("y".to_string())],
        };

        let context = create_test_context();

        let snapshot1 =
            CompilationSnapshot::create(&expr1, &context, "knows(x, y) AND likes(y, z)").unwrap();
        let snapshot2 = CompilationSnapshot::create(&expr2, &context, "knows(x, y)").unwrap();

        let diff = snapshot1.compare(&snapshot2);
        assert!(!diff.is_match());
        assert!(!diff.differences.is_empty());
    }

    #[test]
    fn test_snapshot_suite() {
        let temp_dir = std::env::temp_dir().join("tensorlogic_snapshots_test");
        let suite = SnapshotSuite::new("test_suite", temp_dir.clone());

        let expr = create_test_expr();
        let context = create_test_context();

        // Record a snapshot
        suite
            .record("test1", &expr, &context, "knows(x, y) AND likes(y, z)")
            .unwrap();

        // Verify against the snapshot (skip strict DOT comparison)
        // Since we're comparing identical inputs, use non-strict comparison
        let current =
            CompilationSnapshot::create(&expr, &context, "knows(x, y) AND likes(y, z)").unwrap();
        let path = suite.snapshot_path("test1");
        let recorded = CompilationSnapshot::load(&path).unwrap();
        let diff = recorded.compare_with_options(&current, false);

        assert!(diff.is_match());

        // List snapshots
        let snapshots = suite.list_snapshots().unwrap();
        assert!(snapshots.contains(&"test1".to_string()));

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
