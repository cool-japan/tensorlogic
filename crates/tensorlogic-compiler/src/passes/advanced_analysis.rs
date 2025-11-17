//! Advanced graph analysis integration.
//!
//! This module integrates tensorlogic-ir's analysis capabilities into the compiler pipeline.
//! It provides:
//! - Memory usage estimation
//! - Parallelization analysis
//! - Optimization recommendations

use anyhow::Result;
use tensorlogic_ir::{analyze_memory, analyze_parallelization, EinsumGraph, OpType};

/// Result of advanced graph analysis
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Memory-intensive operations (node indices)
    pub memory_intensive_ops: Vec<usize>,
    /// Potential memory savings (bytes)
    pub potential_memory_savings: usize,
    /// Parallelization opportunities
    pub parallel_opportunities: Vec<ParallelOpportunity>,
    /// Recommended optimizations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Description of a parallelization opportunity
#[derive(Debug, Clone)]
pub struct ParallelOpportunity {
    /// Nodes that can be executed in parallel
    pub parallel_nodes: Vec<usize>,
    /// Estimated speedup from parallelization
    pub estimated_speedup: f64,
    /// Description of the opportunity
    pub description: String,
}

/// Optimization recommendation based on analysis
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Priority level (higher = more important)
    pub priority: u8,
    /// Category of recommendation
    pub category: RecommendationCategory,
    /// Human-readable description
    pub description: String,
    /// Estimated improvement (as a ratio, e.g., 1.5 = 50% improvement)
    pub estimated_improvement: f64,
}

/// Categories of optimization recommendations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendationCategory {
    /// Recommendations related to operation fusion
    Fusion,
    /// Recommendations related to memory optimization
    Memory,
    /// Recommendations related to parallelization
    Parallelization,
    /// Recommendations related to layout optimization
    Layout,
    /// Recommendations related to numerical stability
    Numerical,
    /// General optimization recommendations
    General,
}

impl AnalysisReport {
    /// Create a new empty analysis report
    pub fn new() -> Self {
        Self {
            peak_memory_bytes: 0,
            memory_intensive_ops: Vec::new(),
            potential_memory_savings: 0,
            parallel_opportunities: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    /// Check if there are high-priority recommendations
    pub fn has_critical_recommendations(&self) -> bool {
        self.recommendations.iter().any(|r| r.priority >= 8)
    }

    /// Get high-priority recommendations (priority >= 7)
    pub fn high_priority_recommendations(&self) -> Vec<&OptimizationRecommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.priority >= 7)
            .collect()
    }

    /// Estimate total potential speedup from all recommendations
    pub fn estimated_total_speedup(&self) -> f64 {
        self.recommendations
            .iter()
            .map(|r| r.estimated_improvement)
            .fold(1.0, |acc, x| acc * x)
    }
}

impl Default for AnalysisReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Perform comprehensive analysis of a compiled graph
///
/// This analyzes the graph structure, identifies optimization opportunities,
/// and provides actionable recommendations for improving performance.
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::passes::advanced_analysis::analyze_graph;
/// use tensorlogic_ir::EinsumGraph;
///
/// let graph = EinsumGraph::new();
/// let report = analyze_graph(&graph).unwrap();
///
/// println!("Peak memory: {} bytes", report.peak_memory_bytes);
///
/// for rec in report.high_priority_recommendations() {
///     println!("Priority {}: {}", rec.priority, rec.description);
/// }
/// ```
pub fn analyze_graph(graph: &EinsumGraph) -> Result<AnalysisReport> {
    let mut report = AnalysisReport::new();

    // Memory analysis
    let memory_analysis = analyze_memory(graph, 8)?; // 8 bytes for f64
    report.peak_memory_bytes = memory_analysis.peak_memory_bytes;
    report.potential_memory_savings = memory_analysis
        .total_memory_bytes
        .saturating_sub(memory_analysis.peak_memory_bytes);

    // Parallelization analysis
    let parallel_analysis = analyze_parallelization(graph)?;
    for group in parallel_analysis.parallel_groups {
        if group.nodes.len() > 1 {
            report.parallel_opportunities.push(ParallelOpportunity {
                parallel_nodes: group.nodes.clone(),
                estimated_speedup: (group.nodes.len() as f64).sqrt(),
                description: format!("{} operations can execute in parallel", group.nodes.len()),
            });
        }
    }

    // Generate recommendations
    generate_recommendations(graph, &mut report);

    Ok(report)
}

/// Generate optimization recommendations based on analysis
fn generate_recommendations(graph: &EinsumGraph, report: &mut AnalysisReport) {
    // Recommendation 1: Fusion opportunities
    if has_fusible_operations(graph) {
        report.recommendations.push(OptimizationRecommendation {
            priority: 9,
            category: RecommendationCategory::Fusion,
            description: "Enable operation fusion to reduce kernel launches".to_string(),
            estimated_improvement: 1.3,
        });
    }

    // Recommendation 2: Memory optimization
    if report.peak_memory_bytes > 100_000_000 {
        // > 100MB
        report.recommendations.push(OptimizationRecommendation {
            priority: 8,
            category: RecommendationCategory::Memory,
            description: "Enable memory optimization to reduce peak usage".to_string(),
            estimated_improvement: 1.2,
        });
    }

    // Recommendation 3: Parallelization
    if !report.parallel_opportunities.is_empty() {
        let max_speedup = report
            .parallel_opportunities
            .iter()
            .map(|p| p.estimated_speedup)
            .fold(0.0, f64::max);
        report.recommendations.push(OptimizationRecommendation {
            priority: 7,
            category: RecommendationCategory::Parallelization,
            description: format!(
                "Parallelize {} independent operation groups",
                report.parallel_opportunities.len()
            ),
            estimated_improvement: max_speedup,
        });
    }

    // Recommendation 4: Layout optimization for large graphs
    if graph.nodes.len() > 50 {
        report.recommendations.push(OptimizationRecommendation {
            priority: 6,
            category: RecommendationCategory::Layout,
            description: "Apply layout optimization for better cache locality".to_string(),
            estimated_improvement: 1.15,
        });
    }

    // Recommendation 5: Tiling for memory-intensive graphs
    if report.peak_memory_bytes > 500_000_000 {
        // > 500MB
        report.recommendations.push(OptimizationRecommendation {
            priority: 5,
            category: RecommendationCategory::General,
            description: "Consider tiling to reduce memory pressure".to_string(),
            estimated_improvement: 1.1,
        });
    }

    // Sort recommendations by priority (descending)
    report
        .recommendations
        .sort_by(|a, b| b.priority.cmp(&a.priority));
}

/// Check if the graph has fusible operations
fn has_fusible_operations(graph: &EinsumGraph) -> bool {
    let mut consecutive_elementwise = 0;

    for node in &graph.nodes {
        match &node.op {
            OpType::ElemUnary { op: _ } | OpType::ElemBinary { op: _ } => {
                consecutive_elementwise += 1;
                if consecutive_elementwise >= 2 {
                    return true;
                }
            }
            _ => {
                consecutive_elementwise = 0;
            }
        }
    }

    false
}

/// Quick analysis for fast feedback during compilation
///
/// Provides essential metrics without deep analysis.
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::passes::advanced_analysis::quick_analyze;
/// use tensorlogic_ir::EinsumGraph;
///
/// let graph = EinsumGraph::new();
/// let (peak_memory, parallel_groups) = quick_analyze(&graph).unwrap();
/// println!("Memory: {} bytes, Parallelism: {}", peak_memory, parallel_groups);
/// ```
pub fn quick_analyze(graph: &EinsumGraph) -> Result<(usize, usize)> {
    let memory = analyze_memory(graph, 8)?;
    let parallel = analyze_parallelization(graph)?;
    let parallel_groups = parallel
        .parallel_groups
        .iter()
        .filter(|g| g.nodes.len() > 1)
        .count();
    Ok((memory.peak_memory_bytes, parallel_groups))
}

/// Print a human-readable analysis report
///
/// # Example
///
/// ```
/// use tensorlogic_compiler::passes::advanced_analysis::{analyze_graph, print_report};
/// use tensorlogic_ir::EinsumGraph;
///
/// let graph = EinsumGraph::new();
/// let report = analyze_graph(&graph).unwrap();
/// print_report(&report);
/// ```
pub fn print_report(report: &AnalysisReport) {
    println!("=== Graph Analysis Report ===");
    println!(
        "Peak Memory Usage: {:.2} MB",
        report.peak_memory_bytes as f64 / 1_048_576.0
    );
    println!(
        "Potential Memory Savings: {:.2} MB",
        report.potential_memory_savings as f64 / 1_048_576.0
    );

    if !report.parallel_opportunities.is_empty() {
        println!(
            "\nParallelization Opportunities: {}",
            report.parallel_opportunities.len()
        );
        for (i, opp) in report.parallel_opportunities.iter().enumerate() {
            println!(
                "  {}. {} (speedup: {:.2}x)",
                i + 1,
                opp.description,
                opp.estimated_speedup
            );
        }
    }

    if !report.recommendations.is_empty() {
        println!("\nOptimization Recommendations:");
        for (i, rec) in report.recommendations.iter().enumerate() {
            println!(
                "  {}. [Priority {}] {} (improvement: {:.2}x)",
                i + 1,
                rec.priority,
                rec.description,
                rec.estimated_improvement
            );
        }

        println!(
            "\nEstimated Total Speedup: {:.2}x",
            report.estimated_total_speedup()
        );
    }

    println!("============================");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::EinsumNode;

    #[test]
    fn test_analysis_report_new() {
        let report = AnalysisReport::new();
        assert_eq!(report.peak_memory_bytes, 0);
        assert!(report.parallel_opportunities.is_empty());
    }

    #[test]
    fn test_has_critical_recommendations() {
        let mut report = AnalysisReport::new();
        assert!(!report.has_critical_recommendations());

        report.recommendations.push(OptimizationRecommendation {
            priority: 9,
            category: RecommendationCategory::Fusion,
            description: "Test".to_string(),
            estimated_improvement: 1.5,
        });
        assert!(report.has_critical_recommendations());
    }

    #[test]
    fn test_high_priority_recommendations() {
        let mut report = AnalysisReport::new();

        report.recommendations.push(OptimizationRecommendation {
            priority: 9,
            category: RecommendationCategory::Fusion,
            description: "High priority".to_string(),
            estimated_improvement: 1.5,
        });

        report.recommendations.push(OptimizationRecommendation {
            priority: 5,
            category: RecommendationCategory::General,
            description: "Low priority".to_string(),
            estimated_improvement: 1.1,
        });

        let high_priority = report.high_priority_recommendations();
        assert_eq!(high_priority.len(), 1);
        assert_eq!(high_priority[0].priority, 9);
    }

    #[test]
    fn test_estimated_total_speedup() {
        let mut report = AnalysisReport::new();

        report.recommendations.push(OptimizationRecommendation {
            priority: 9,
            category: RecommendationCategory::Fusion,
            description: "Test 1".to_string(),
            estimated_improvement: 1.5,
        });

        report.recommendations.push(OptimizationRecommendation {
            priority: 8,
            category: RecommendationCategory::Memory,
            description: "Test 2".to_string(),
            estimated_improvement: 1.2,
        });

        let speedup = report.estimated_total_speedup();
        assert!((speedup - 1.8).abs() < 0.01); // 1.5 * 1.2 = 1.8
    }

    #[test]
    fn test_analyze_empty_graph() {
        let graph = EinsumGraph::new();
        let result = analyze_graph(&graph);
        assert!(result.is_ok());

        let report = result.unwrap();
        assert_eq!(report.peak_memory_bytes, 0);
    }

    #[test]
    fn test_quick_analyze_empty() {
        let graph = EinsumGraph::new();
        let result = quick_analyze(&graph);
        assert!(result.is_ok());

        let (memory, parallel_groups) = result.unwrap();
        assert_eq!(memory, 0);
        assert_eq!(parallel_groups, 0);
    }

    #[test]
    fn test_has_fusible_operations_no_fusion() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("x");
        let t1 = graph.add_tensor("x_relu");
        let _ = graph.add_node(EinsumNode::elem_unary("relu", t0, t1));

        assert!(!has_fusible_operations(&graph));
    }

    #[test]
    fn test_has_fusible_operations_with_fusion() {
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("x");
        let t1 = graph.add_tensor("y");
        let t2 = graph.add_tensor("x_relu");
        let t3 = graph.add_tensor("y_tanh");
        let _ = graph.add_node(EinsumNode::elem_unary("relu", t0, t2));
        let _ = graph.add_node(EinsumNode::elem_unary("tanh", t1, t3));

        // Two consecutive element-wise operations
        assert!(has_fusible_operations(&graph));
    }

    #[test]
    fn test_recommendation_category_equality() {
        assert_eq!(
            RecommendationCategory::Fusion,
            RecommendationCategory::Fusion
        );
        assert_ne!(
            RecommendationCategory::Fusion,
            RecommendationCategory::Memory
        );
    }

    #[test]
    fn test_parallel_opportunity_creation() {
        let opp = ParallelOpportunity {
            parallel_nodes: vec![1, 2, 3],
            estimated_speedup: 1.7,
            description: "Test opportunity".to_string(),
        };

        assert_eq!(opp.parallel_nodes.len(), 3);
        assert!((opp.estimated_speedup - 1.7).abs() < 0.01);
    }
}
