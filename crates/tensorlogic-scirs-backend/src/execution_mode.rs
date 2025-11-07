//! Execution mode abstractions for different execution strategies.
//!
//! This module provides infrastructure for multiple execution modes:
//! - **Eager**: Immediate execution (default, already implemented)
//! - **Graph**: Graph compilation and optimization
//! - **JIT**: Just-in-time compilation (future)

use tensorlogic_ir::EinsumGraph;

/// Execution mode for the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    /// Eager execution: operations execute immediately as they're called.
    /// This is the default mode and provides the best debugging experience.
    #[default]
    Eager,

    /// Graph mode: operations are compiled into an optimized graph before execution.
    /// This mode enables graph-level optimizations like operation fusion and memory planning.
    Graph,

    /// JIT mode: operations are compiled to native code at runtime.
    /// This mode provides the best performance but has compilation overhead.
    /// Currently not implemented.
    Jit,
}

impl ExecutionMode {
    /// Returns true if this mode is eager execution.
    pub fn is_eager(&self) -> bool {
        matches!(self, ExecutionMode::Eager)
    }

    /// Returns true if this mode requires graph compilation.
    pub fn requires_compilation(&self) -> bool {
        matches!(self, ExecutionMode::Graph | ExecutionMode::Jit)
    }

    /// Returns a human-readable description of this mode.
    pub fn description(&self) -> &'static str {
        match self {
            ExecutionMode::Eager => "Immediate execution with no compilation overhead",
            ExecutionMode::Graph => "Graph compilation with optimization passes",
            ExecutionMode::Jit => "Just-in-time compilation to native code",
        }
    }
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionMode::Eager => write!(f, "Eager"),
            ExecutionMode::Graph => write!(f, "Graph"),
            ExecutionMode::Jit => write!(f, "JIT"),
        }
    }
}

/// Compiled graph for optimized execution.
///
/// In Graph mode, the EinsumGraph is analyzed and optimized before execution.
/// This structure holds the compiled representation.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    /// Original graph
    pub original: EinsumGraph,

    /// Optimized graph (after passes like fusion, CSE, DCE)
    pub optimized: EinsumGraph,

    /// Memory plan for tensor allocation
    pub memory_plan: Option<MemoryPlan>,

    /// Compilation statistics
    pub stats: CompilationStats,
}

/// Memory allocation plan for optimized execution.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Maximum number of tensors alive at any point
    pub max_live_tensors: usize,

    /// Peak memory usage estimate (in bytes)
    pub peak_memory_bytes: usize,

    /// Tensor reuse opportunities
    pub reuse_opportunities: Vec<(usize, usize)>, // (source_tensor, dest_tensor)
}

/// Statistics from graph compilation.
#[derive(Debug, Clone, Default)]
pub struct CompilationStats {
    /// Number of operations in original graph
    pub original_ops: usize,

    /// Number of operations after optimization
    pub optimized_ops: usize,

    /// Number of operations eliminated
    pub eliminated_ops: usize,

    /// Number of operations fused
    pub fused_ops: usize,

    /// Compilation time in milliseconds
    pub compilation_time_ms: f64,
}

impl CompiledGraph {
    /// Create a new compiled graph from an EinsumGraph.
    ///
    /// This performs optimization passes on the graph.
    pub fn compile(graph: EinsumGraph) -> Self {
        let start = std::time::Instant::now();
        let original_ops = graph.nodes.len();

        // For now, just use the original graph
        // In the future, this would apply optimization passes:
        // - Dead code elimination
        // - Common subexpression elimination
        // - Operation fusion
        // - Memory planning
        let optimized = graph.clone();
        let optimized_ops = optimized.nodes.len();

        let compilation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        CompiledGraph {
            original: graph,
            optimized,
            memory_plan: None,
            stats: CompilationStats {
                original_ops,
                optimized_ops,
                eliminated_ops: original_ops.saturating_sub(optimized_ops),
                fused_ops: 0,
                compilation_time_ms,
            },
        }
    }

    /// Get the graph to execute (optimized version).
    pub fn graph(&self) -> &EinsumGraph {
        &self.optimized
    }

    /// Get compilation statistics.
    pub fn stats(&self) -> &CompilationStats {
        &self.stats
    }
}

impl std::fmt::Display for CompilationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CompilationStats {{ original: {}, optimized: {}, eliminated: {}, fused: {}, time: {:.2}ms }}",
            self.original_ops,
            self.optimized_ops,
            self.eliminated_ops,
            self.fused_ops,
            self.compilation_time_ms
        )
    }
}

/// Execution configuration combining mode and device settings.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Execution mode
    pub mode: ExecutionMode,

    /// Enable graph optimizations (only applies to Graph mode)
    pub enable_optimizations: bool,

    /// Enable memory planning (only applies to Graph mode)
    pub enable_memory_planning: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Eager,
            enable_optimizations: true,
            enable_memory_planning: true,
        }
    }
}

impl ExecutionConfig {
    /// Create a new configuration with eager mode.
    pub fn eager() -> Self {
        Self {
            mode: ExecutionMode::Eager,
            enable_optimizations: false,
            enable_memory_planning: false,
        }
    }

    /// Create a new configuration with graph mode.
    pub fn graph() -> Self {
        Self {
            mode: ExecutionMode::Graph,
            enable_optimizations: true,
            enable_memory_planning: true,
        }
    }

    /// Enable or disable optimizations.
    pub fn with_optimizations(mut self, enable: bool) -> Self {
        self.enable_optimizations = enable;
        self
    }

    /// Enable or disable memory planning.
    pub fn with_memory_planning(mut self, enable: bool) -> Self {
        self.enable_memory_planning = enable;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_mode_default() {
        let mode = ExecutionMode::default();
        assert_eq!(mode, ExecutionMode::Eager);
        assert!(mode.is_eager());
        assert!(!mode.requires_compilation());
    }

    #[test]
    fn test_execution_mode_properties() {
        assert!(ExecutionMode::Eager.is_eager());
        assert!(!ExecutionMode::Graph.is_eager());
        assert!(!ExecutionMode::Jit.is_eager());

        assert!(!ExecutionMode::Eager.requires_compilation());
        assert!(ExecutionMode::Graph.requires_compilation());
        assert!(ExecutionMode::Jit.requires_compilation());
    }

    #[test]
    fn test_execution_mode_display() {
        assert_eq!(ExecutionMode::Eager.to_string(), "Eager");
        assert_eq!(ExecutionMode::Graph.to_string(), "Graph");
        assert_eq!(ExecutionMode::Jit.to_string(), "JIT");
    }

    #[test]
    fn test_execution_config_default() {
        let config = ExecutionConfig::default();
        assert_eq!(config.mode, ExecutionMode::Eager);
        assert!(config.enable_optimizations);
        assert!(config.enable_memory_planning);
    }

    #[test]
    fn test_execution_config_eager() {
        let config = ExecutionConfig::eager();
        assert_eq!(config.mode, ExecutionMode::Eager);
        assert!(!config.enable_optimizations);
        assert!(!config.enable_memory_planning);
    }

    #[test]
    fn test_execution_config_graph() {
        let config = ExecutionConfig::graph();
        assert_eq!(config.mode, ExecutionMode::Graph);
        assert!(config.enable_optimizations);
        assert!(config.enable_memory_planning);
    }

    #[test]
    fn test_execution_config_builder() {
        let config = ExecutionConfig::graph()
            .with_optimizations(false)
            .with_memory_planning(false);

        assert_eq!(config.mode, ExecutionMode::Graph);
        assert!(!config.enable_optimizations);
        assert!(!config.enable_memory_planning);
    }

    #[test]
    fn test_compiled_graph_basic() {
        use tensorlogic_ir::{EinsumNode, OpType};

        let mut graph = EinsumGraph::new();
        let a_idx = graph.add_tensor("a");
        let b_idx = graph.add_tensor("b");

        graph.add_input(a_idx).unwrap();
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "relu".to_string(),
                },
                inputs: vec![a_idx],
                outputs: vec![b_idx],
                metadata: None,
            })
            .unwrap();
        graph.add_output(b_idx).unwrap();

        let compiled = CompiledGraph::compile(graph);

        assert_eq!(compiled.stats.original_ops, 1);
        assert_eq!(compiled.stats.optimized_ops, 1);
        assert_eq!(compiled.stats.eliminated_ops, 0);
    }

    #[test]
    fn test_compilation_stats_display() {
        let stats = CompilationStats {
            original_ops: 10,
            optimized_ops: 8,
            eliminated_ops: 2,
            fused_ops: 1,
            compilation_time_ms: 1.5,
        };

        let display = stats.to_string();
        assert!(display.contains("original: 10"));
        assert!(display.contains("optimized: 8"));
        assert!(display.contains("eliminated: 2"));
    }
}
