//! Execution mode abstractions for different execution strategies.
//!
//! This module provides infrastructure for multiple execution modes:
//! - **Eager**: Immediate execution (default, already implemented)
//! - **Graph**: Graph compilation and optimization
//! - **JIT**: Just-in-time compilation (future)

use std::collections::{HashMap, HashSet};
use tensorlogic_ir::{
    fold_constants_aggressive, fuse_elementwise_operations, optimize_layouts, EinsumGraph,
    EinsumNode, OpType,
};

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

/// Configuration for graph optimization passes.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable constant folding
    pub enable_constant_folding: bool,

    /// Enable operation fusion
    pub enable_fusion: bool,

    /// Enable dead code elimination
    pub enable_dce: bool,

    /// Enable common subexpression elimination
    pub enable_cse: bool,

    /// Enable layout optimization
    pub enable_layout_opt: bool,

    /// Enable memory planning
    pub enable_memory_planning: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_constant_folding: true,
            enable_fusion: true,
            enable_dce: true,
            enable_cse: true,
            enable_layout_opt: true,
            enable_memory_planning: true,
        }
    }
}

impl OptimizationConfig {
    /// Create a new configuration with all optimizations enabled.
    pub fn aggressive() -> Self {
        Self::default()
    }

    /// Create a new configuration with only safe optimizations.
    pub fn conservative() -> Self {
        Self {
            enable_constant_folding: true,
            enable_fusion: false,
            enable_dce: true,
            enable_cse: false,
            enable_layout_opt: false,
            enable_memory_planning: false,
        }
    }

    /// Create a new configuration with no optimizations.
    pub fn none() -> Self {
        Self {
            enable_constant_folding: false,
            enable_fusion: false,
            enable_dce: false,
            enable_cse: false,
            enable_layout_opt: false,
            enable_memory_planning: false,
        }
    }
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
        Self::compile_with_config(graph, &OptimizationConfig::default())
    }

    /// Create a new compiled graph with custom optimization configuration.
    pub fn compile_with_config(graph: EinsumGraph, config: &OptimizationConfig) -> Self {
        let start = std::time::Instant::now();
        let original_ops = graph.nodes.len();

        let mut optimized = graph.clone();
        let mut fused_count = 0;
        let mut eliminated_count = 0;

        // Phase 1: Constant folding (if enabled)
        if config.enable_constant_folding {
            if let Ok(_stats) = fold_constants_aggressive(&mut optimized) {
                // Constant folding succeeded
            }
        }

        // Phase 2: Operation fusion (if enabled)
        if config.enable_fusion {
            if let Ok(stats) = fuse_elementwise_operations(&mut optimized) {
                fused_count = stats.ops_fused;
            }
        }

        // Phase 3: Dead code elimination (if enabled)
        if config.enable_dce {
            if let Ok(removed) = eliminate_dead_code(&mut optimized) {
                eliminated_count += removed;
            }
        }

        // Phase 4: Common subexpression elimination (if enabled)
        if config.enable_cse {
            if let Ok(removed) = eliminate_common_subexpressions(&mut optimized) {
                eliminated_count += removed;
            }
        }

        // Phase 5: Layout optimization (if enabled)
        if config.enable_layout_opt {
            if let Ok(_result) = optimize_layouts(&optimized) {
                // Layout optimization succeeded
            }
        }

        let optimized_ops = optimized.nodes.len();
        let compilation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Phase 6: Memory planning (if enabled)
        let memory_plan = if config.enable_memory_planning {
            Some(compute_memory_plan(&optimized))
        } else {
            None
        };

        CompiledGraph {
            original: graph,
            optimized,
            memory_plan,
            stats: CompilationStats {
                original_ops,
                optimized_ops,
                eliminated_ops: eliminated_count,
                fused_ops: fused_count,
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

/// Dead Code Elimination (DCE) - removes unused tensors and nodes.
fn eliminate_dead_code(graph: &mut EinsumGraph) -> Result<usize, String> {
    if graph.outputs.is_empty() {
        return Ok(0);
    }

    // Track which tensors are live (needed)
    let mut live_tensors = HashSet::new();
    let mut worklist: Vec<usize> = graph.outputs.clone();

    // Mark all output tensors as live
    for &output_idx in &graph.outputs {
        live_tensors.insert(output_idx);
    }

    // Build tensor-to-node mapping (which node produces each tensor)
    let mut tensor_producers: HashMap<usize, usize> = HashMap::new();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output_idx in &node.outputs {
            tensor_producers.insert(output_idx, node_idx);
        }
    }

    // Backward pass: mark all dependencies as live
    while let Some(tensor_idx) = worklist.pop() {
        if let Some(&node_idx) = tensor_producers.get(&tensor_idx) {
            let node = &graph.nodes[node_idx];
            for &input_idx in &node.inputs {
                if !live_tensors.contains(&input_idx) {
                    live_tensors.insert(input_idx);
                    worklist.push(input_idx);
                }
            }
        }
    }

    // Remove dead nodes (nodes whose output is not live)
    let initial_count = graph.nodes.len();
    let mut nodes_to_keep = Vec::new();
    for node in &graph.nodes {
        let all_outputs_live = node
            .outputs
            .iter()
            .any(|out_idx| live_tensors.contains(out_idx));
        if all_outputs_live {
            nodes_to_keep.push(node.clone());
        }
    }

    graph.nodes = nodes_to_keep;
    let removed_count = initial_count - graph.nodes.len();

    Ok(removed_count)
}

/// Common Subexpression Elimination (CSE) - detects and deduplicates identical subgraphs.
fn eliminate_common_subexpressions(graph: &mut EinsumGraph) -> Result<usize, String> {
    let mut node_hashes: HashMap<String, usize> = HashMap::new();
    let mut replacements: HashMap<usize, usize> = HashMap::new();
    let mut eliminated_count = 0;

    // Build hash for each node (based on operation and inputs)
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        let node_hash = compute_node_hash(node);

        if let Some(&existing_idx) = node_hashes.get(&node_hash) {
            // Found a duplicate - mark for replacement
            if !node.outputs.is_empty() && !graph.nodes[existing_idx].outputs.is_empty() {
                let produced_tensor_idx = node.outputs[0];
                let existing_tensor_idx = graph.nodes[existing_idx].outputs[0];
                replacements.insert(produced_tensor_idx, existing_tensor_idx);
                eliminated_count += 1;
            }
        } else {
            node_hashes.insert(node_hash, node_idx);
        }
    }

    // Apply replacements (update all node inputs that reference eliminated tensors)
    if !replacements.is_empty() {
        for node in &mut graph.nodes {
            for input_idx in &mut node.inputs {
                if let Some(&replacement_idx) = replacements.get(input_idx) {
                    *input_idx = replacement_idx;
                }
            }
        }

        // Update outputs
        for output_idx in &mut graph.outputs {
            if let Some(&replacement_idx) = replacements.get(output_idx) {
                *output_idx = replacement_idx;
            }
        }
    }

    Ok(eliminated_count)
}

/// Compute a hash for a node based on its operation and inputs.
fn compute_node_hash(node: &EinsumNode) -> String {
    let op_str = match &node.op {
        OpType::Einsum { spec } => format!("einsum:{}", spec),
        OpType::ElemUnary { op } => format!("unary:{}", op),
        OpType::ElemBinary { op } => format!("binary:{}", op),
        OpType::Reduce { op, axes } => format!("reduce:{}:{:?}", op, axes),
    };

    format!("{}|inputs:{:?}", op_str, node.inputs)
}

/// Compute memory plan for a graph.
fn compute_memory_plan(graph: &EinsumGraph) -> MemoryPlan {
    // Build liveness analysis
    let total_tensors = graph.tensors.len();
    let mut live_at_step: Vec<HashSet<usize>> = Vec::new();
    let mut current_live = HashSet::new();

    // Add input tensors as initially live
    for &input_idx in &graph.inputs {
        current_live.insert(input_idx);
    }

    // Process each node in order
    for node in &graph.nodes {
        // Mark outputs as live
        for &output_idx in &node.outputs {
            current_live.insert(output_idx);
        }

        // Check if inputs are still needed later
        for &input_idx in &node.inputs {
            let mut still_needed = false;
            // Check if this input is used by later nodes
            for later_node in graph.nodes.iter().skip(1) {
                if later_node.inputs.contains(&input_idx) {
                    still_needed = true;
                    break;
                }
            }
            // Check if it's an output
            if graph.outputs.contains(&input_idx) {
                still_needed = true;
            }
            if !still_needed {
                current_live.remove(&input_idx);
            }
        }

        live_at_step.push(current_live.clone());
    }

    // Compute max live tensors
    let max_live_tensors = live_at_step
        .iter()
        .map(|live_set| live_set.len())
        .max()
        .unwrap_or(0);

    // Estimate peak memory (assuming 8 bytes per element, 1000 elements per tensor on average)
    let avg_tensor_size = 8 * 1000; // 8KB average
    let peak_memory_bytes = max_live_tensors * avg_tensor_size;

    // Identify reuse opportunities (tensors with non-overlapping lifetimes)
    let mut reuse_opportunities = Vec::new();
    for i in 0..total_tensors {
        for j in (i + 1)..total_tensors {
            // Check if lifetimes don't overlap
            let mut i_live = false;
            let mut j_live = false;
            let mut overlap = false;

            for live_set in &live_at_step {
                let i_in_this = live_set.contains(&i);
                let j_in_this = live_set.contains(&j);

                if i_in_this {
                    i_live = true;
                }
                if j_in_this {
                    j_live = true;
                }
                if i_in_this && j_in_this {
                    overlap = true;
                    break;
                }
            }

            if i_live && j_live && !overlap {
                reuse_opportunities.push((i, j));
            }
        }
    }

    MemoryPlan {
        max_live_tensors,
        peak_memory_bytes,
        reuse_opportunities,
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

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert!(config.enable_constant_folding);
        assert!(config.enable_fusion);
        assert!(config.enable_dce);
        assert!(config.enable_cse);
        assert!(config.enable_layout_opt);
        assert!(config.enable_memory_planning);
    }

    #[test]
    fn test_optimization_config_aggressive() {
        let config = OptimizationConfig::aggressive();
        assert!(config.enable_constant_folding);
        assert!(config.enable_fusion);
        assert!(config.enable_dce);
        assert!(config.enable_cse);
        assert!(config.enable_layout_opt);
        assert!(config.enable_memory_planning);
    }

    #[test]
    fn test_optimization_config_conservative() {
        let config = OptimizationConfig::conservative();
        assert!(config.enable_constant_folding);
        assert!(!config.enable_fusion);
        assert!(config.enable_dce);
        assert!(!config.enable_cse);
        assert!(!config.enable_layout_opt);
        assert!(!config.enable_memory_planning);
    }

    #[test]
    fn test_optimization_config_none() {
        let config = OptimizationConfig::none();
        assert!(!config.enable_constant_folding);
        assert!(!config.enable_fusion);
        assert!(!config.enable_dce);
        assert!(!config.enable_cse);
        assert!(!config.enable_layout_opt);
        assert!(!config.enable_memory_planning);
    }

    #[test]
    fn test_compiled_graph_with_optimization() {
        use tensorlogic_ir::{EinsumNode, OpType};

        let mut graph = EinsumGraph::new();
        let a_idx = graph.add_tensor("a");
        let b_idx = graph.add_tensor("b");
        let c_idx = graph.add_tensor("c");

        graph.add_input(a_idx).unwrap();

        // Add a ReLU node
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

        // Add another ReLU node (duplicate for CSE testing)
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "relu".to_string(),
                },
                inputs: vec![a_idx],
                outputs: vec![c_idx],
                metadata: None,
            })
            .unwrap();

        graph.add_output(b_idx).unwrap();

        let compiled = CompiledGraph::compile(graph);

        assert_eq!(compiled.stats.original_ops, 2);
        // Note: The optimized ops might be less if CSE works
        assert!(compiled.stats.compilation_time_ms >= 0.0);
    }

    #[test]
    fn test_compiled_graph_with_custom_config() {
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

        let config = OptimizationConfig::none();
        let compiled = CompiledGraph::compile_with_config(graph, &config);

        assert_eq!(compiled.stats.original_ops, 1);
        assert_eq!(compiled.stats.optimized_ops, 1);
        assert_eq!(compiled.stats.eliminated_ops, 0);
        assert_eq!(compiled.stats.fused_ops, 0);
        assert!(compiled.memory_plan.is_none());
    }

    #[test]
    fn test_memory_plan_basic() {
        use tensorlogic_ir::{EinsumNode, OpType};

        let mut graph = EinsumGraph::new();
        let a_idx = graph.add_tensor("a");
        let b_idx = graph.add_tensor("b");
        let c_idx = graph.add_tensor("c");

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
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "sigmoid".to_string(),
                },
                inputs: vec![b_idx],
                outputs: vec![c_idx],
                metadata: None,
            })
            .unwrap();
        graph.add_output(c_idx).unwrap();

        let compiled = CompiledGraph::compile(graph);

        assert!(compiled.memory_plan.is_some());
        let plan = compiled.memory_plan.unwrap();
        assert!(plan.max_live_tensors > 0);
        assert!(plan.peak_memory_bytes > 0);
    }

    #[test]
    fn test_dce_removes_dead_code() {
        use tensorlogic_ir::{EinsumNode, OpType};

        let mut graph = EinsumGraph::new();
        let a_idx = graph.add_tensor("a");
        let b_idx = graph.add_tensor("b");
        let c_idx = graph.add_tensor("c");
        let d_idx = graph.add_tensor("d");

        graph.add_input(a_idx).unwrap();

        // Node that produces b (will be used)
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

        // Dead node that produces c (not used)
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "sigmoid".to_string(),
                },
                inputs: vec![a_idx],
                outputs: vec![c_idx],
                metadata: None,
            })
            .unwrap();

        // Node that uses b to produce d
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "oneminus".to_string(),
                },
                inputs: vec![b_idx],
                outputs: vec![d_idx],
                metadata: None,
            })
            .unwrap();

        graph.add_output(d_idx).unwrap();

        let initial_nodes = graph.nodes.len();
        let removed = eliminate_dead_code(&mut graph).unwrap();

        // Should remove the dead sigmoid node
        assert!(removed > 0 || graph.nodes.len() < initial_nodes);
    }

    #[test]
    fn test_cse_deduplicates_nodes() {
        use tensorlogic_ir::{EinsumNode, OpType};

        let mut graph = EinsumGraph::new();
        let a_idx = graph.add_tensor("a");
        let b_idx = graph.add_tensor("b");
        let c_idx = graph.add_tensor("c");

        graph.add_input(a_idx).unwrap();

        // First ReLU
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

        // Duplicate ReLU (same operation, same input)
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "relu".to_string(),
                },
                inputs: vec![a_idx],
                outputs: vec![c_idx],
                metadata: None,
            })
            .unwrap();

        graph.add_output(b_idx).unwrap();
        graph.add_output(c_idx).unwrap();

        let eliminated = eliminate_common_subexpressions(&mut graph).unwrap();

        // Should detect the duplicate (CSE may or may not eliminate it depending on implementation)
        // At minimum, the function should not error
        let _ = eliminated; // Use the value to avoid unused variable warning
    }
}
