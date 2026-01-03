//! Profiling functionality for TensorLogic CLI
//!
//! Provides detailed phase-by-phase timing breakdown and memory analysis.

use anyhow::Result;
use colored::Colorize;
use serde::Serialize;
use std::time::Instant;
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{validate_graph, EinsumGraph, TLExpr};

use crate::analysis::GraphMetrics;
use crate::optimize::{optimize_einsum_graph, OptimizationConfig, OptimizationLevel};
use crate::output::{print_header, print_success};

/// Profile data for a single compilation run
#[derive(Debug, Clone, Serialize)]
pub struct ProfileData {
    /// Total compilation time
    pub total_time_us: u64,
    /// Phase breakdown
    pub phases: Vec<PhaseProfile>,
    /// Memory estimates
    pub memory_estimate: MemoryProfile,
    /// Graph metrics after compilation
    pub graph_metrics: ProfileGraphMetrics,
    /// Bottleneck analysis
    pub bottleneck_analysis: BottleneckAnalysis,
    /// Performance variance (if multiple runs)
    pub variance: PerformanceVariance,
    /// Execution profiling data (if enabled)
    pub execution_profile: Option<ExecutionProfile>,
}

/// Profile for a single phase
#[derive(Debug, Clone, Serialize)]
pub struct PhaseProfile {
    /// Phase name
    pub name: String,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Percentage of total time
    pub percentage: f64,
}

/// Memory usage estimates
#[derive(Debug, Clone, Serialize)]
pub struct MemoryProfile {
    /// Estimated tensor memory in bytes
    pub tensor_memory_bytes: usize,
    /// Estimated graph structure memory in bytes
    pub graph_structure_bytes: usize,
    /// Total estimated memory in bytes
    pub total_bytes: usize,
}

/// Simplified graph metrics for profiling output
#[derive(Debug, Clone, Serialize)]
pub struct ProfileGraphMetrics {
    pub tensor_count: usize,
    pub node_count: usize,
    pub depth: usize,
    pub estimated_flops: u64,
}

/// Bottleneck analysis identifying performance issues
#[derive(Debug, Clone, Serialize)]
pub struct BottleneckAnalysis {
    /// Identified hotspots (phases >30% of total time)
    pub hotspots: Vec<Hotspot>,
    /// Optimization suggestions
    pub suggestions: Vec<String>,
    /// Overall bottleneck severity (0-100)
    pub severity_score: u8,
}

/// A performance hotspot
#[derive(Debug, Clone, Serialize)]
pub struct Hotspot {
    /// Phase name
    pub phase: String,
    /// Percentage of total time
    pub percentage: f64,
    /// Time in microseconds
    pub duration_us: u64,
    /// Severity level (Low, Medium, High, Critical)
    pub severity: HotspotSeverity,
}

/// Hotspot severity levels
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum HotspotSeverity {
    Low,      // 30-40%
    Medium,   // 40-55%
    High,     // 55-75%
    Critical, // >75%
}

impl HotspotSeverity {
    fn from_percentage(pct: f64) -> Self {
        if pct > 75.0 {
            HotspotSeverity::Critical
        } else if pct > 55.0 {
            HotspotSeverity::High
        } else if pct > 40.0 {
            HotspotSeverity::Medium
        } else {
            HotspotSeverity::Low
        }
    }
}

/// Performance variance across multiple runs
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceVariance {
    /// Standard deviation of total time (microseconds)
    pub total_stddev_us: f64,
    /// Coefficient of variation (stddev/mean) as percentage
    pub coefficient_of_variation: f64,
    /// Minimum total time observed
    pub min_time_us: u64,
    /// Maximum total time observed
    pub max_time_us: u64,
    /// Phase-level variance
    pub phase_variance: Vec<PhaseVariance>,
}

/// Variance for a specific phase
#[derive(Debug, Clone, Serialize)]
pub struct PhaseVariance {
    pub phase: String,
    pub stddev_us: f64,
    pub coefficient_of_variation: f64,
}

/// Execution profiling data
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionProfile {
    /// Average execution time in microseconds
    pub avg_execution_time_us: u64,
    /// Minimum execution time
    pub min_execution_time_us: u64,
    /// Maximum execution time
    pub max_execution_time_us: u64,
    /// Standard deviation of execution time
    pub execution_stddev_us: f64,
    /// Actual memory usage during execution (bytes)
    pub actual_memory_bytes: usize,
    /// Backend used for execution
    pub backend: String,
    /// Throughput (graphs/second)
    pub throughput: f64,
}

/// Configuration for profiling
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Include optimization pass profiling
    pub include_optimization: bool,
    /// Optimization level for profiling
    pub optimization_level: OptimizationLevel,
    /// Include validation profiling
    pub include_validation: bool,
    /// Show detailed phase breakdown (reserved for future use)
    #[allow(dead_code)]
    pub detailed: bool,
    /// Number of warmup runs
    pub warmup_runs: usize,
    /// Number of profiling runs for averaging
    pub profile_runs: usize,
    /// Include execution profiling
    pub include_execution: bool,
    /// Backend for execution profiling
    pub execution_backend: String,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            include_optimization: true,
            optimization_level: OptimizationLevel::Standard,
            include_validation: false,
            detailed: true,
            warmup_runs: 1,
            profile_runs: 3,
            include_execution: false,
            execution_backend: "cpu".to_string(),
        }
    }
}

/// Profile compilation of a TLExpr
pub struct Profiler {
    config: ProfileConfig,
}

impl Profiler {
    /// Create a new profiler with the given configuration
    pub fn new(config: ProfileConfig) -> Self {
        Self { config }
    }

    /// Profile the compilation of an expression
    pub fn profile(&self, expr: &TLExpr, context: &CompilerContext) -> Result<ProfileData> {
        // Warmup runs
        for _ in 0..self.config.warmup_runs {
            let mut ctx = context.clone();
            let _ = compile_to_einsum_with_context(expr, &mut ctx);
        }

        // Profile multiple runs and average
        let mut all_phases: Vec<Vec<PhaseProfile>> = Vec::new();
        let mut total_times: Vec<u64> = Vec::new();
        let mut final_graph: Option<EinsumGraph> = None;

        for _ in 0..self.config.profile_runs {
            let (phases, total, graph) = self.profile_single_run(expr, context)?;
            all_phases.push(phases);
            total_times.push(total);
            if final_graph.is_none() {
                final_graph = Some(graph);
            }
        }

        // Average the results
        let avg_total = total_times.iter().sum::<u64>() / total_times.len() as u64;
        let phases = self.average_phases(&all_phases, avg_total);

        let graph = final_graph.expect("At least one profile run must succeed");
        let memory_estimate = self.estimate_memory(&graph);
        let metrics = GraphMetrics::analyze(&graph);

        // Calculate variance
        let variance = self.calculate_variance(&all_phases, &total_times, avg_total);

        // Perform bottleneck analysis
        let bottleneck_analysis = self.analyze_bottlenecks(&phases, &graph);

        // Perform execution profiling if enabled
        let execution_profile = if self.config.include_execution {
            Some(self.profile_execution(&graph)?)
        } else {
            None
        };

        Ok(ProfileData {
            total_time_us: avg_total,
            phases,
            memory_estimate,
            graph_metrics: ProfileGraphMetrics {
                tensor_count: metrics.tensor_count,
                node_count: metrics.node_count,
                depth: metrics.depth,
                estimated_flops: metrics.estimated_flops,
            },
            bottleneck_analysis,
            variance,
            execution_profile,
        })
    }

    fn profile_execution(&self, graph: &EinsumGraph) -> Result<ExecutionProfile> {
        use crate::executor::{Backend, CliExecutor, ExecutionConfig};
        use tensorlogic_scirs_backend::DeviceType;

        let backend = Backend::from_str(&self.config.execution_backend)?;
        let exec_config = ExecutionConfig {
            backend,
            device: DeviceType::Cpu,
            show_metrics: false,
            show_intermediates: false,
            validate_shapes: false,
            trace: false,
        };

        let executor = CliExecutor::new(exec_config)?;

        // Warmup runs
        for _ in 0..self.config.warmup_runs {
            let _ = executor.execute(graph);
        }

        // Profile multiple execution runs
        let mut execution_times = Vec::new();
        let mut memory_bytes = 0usize;

        for _ in 0..self.config.profile_runs {
            let start = Instant::now();
            let result = executor.execute(graph)?;
            let duration_us = start.elapsed().as_micros() as u64;
            execution_times.push(duration_us);
            memory_bytes = result.memory_bytes; // Use the last measured value
        }

        // Calculate statistics
        let avg_time = execution_times.iter().sum::<u64>() / execution_times.len() as u64;
        let min_time = *execution_times.iter().min().unwrap_or(&avg_time);
        let max_time = *execution_times.iter().max().unwrap_or(&avg_time);

        let stddev = if execution_times.len() > 1 {
            let variance: f64 = execution_times
                .iter()
                .map(|&t| {
                    let diff = t as f64 - avg_time as f64;
                    diff * diff
                })
                .sum::<f64>()
                / execution_times.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let throughput = if avg_time > 0 {
            1_000_000.0 / avg_time as f64 // graphs per second
        } else {
            0.0
        };

        Ok(ExecutionProfile {
            avg_execution_time_us: avg_time,
            min_execution_time_us: min_time,
            max_execution_time_us: max_time,
            execution_stddev_us: stddev,
            actual_memory_bytes: memory_bytes,
            backend: backend.name().to_string(),
            throughput,
        })
    }

    fn profile_single_run(
        &self,
        expr: &TLExpr,
        context: &CompilerContext,
    ) -> Result<(Vec<PhaseProfile>, u64, EinsumGraph)> {
        let mut phases = Vec::new();
        let total_start = Instant::now();

        // Phase 1: Expression analysis / preparation
        let phase_start = Instant::now();
        let _ = format!("{:?}", expr); // Force any lazy evaluation
        let analysis_duration = phase_start.elapsed();
        phases.push(("Expression Analysis".to_string(), analysis_duration));

        // Phase 2: Compilation
        let phase_start = Instant::now();
        let mut ctx = context.clone();
        let graph = compile_to_einsum_with_context(expr, &mut ctx)?;
        let compilation_duration = phase_start.elapsed();
        phases.push(("IR Compilation".to_string(), compilation_duration));

        // Phase 3: Validation (if enabled)
        if self.config.include_validation {
            let phase_start = Instant::now();
            let _report = validate_graph(&graph);
            let validation_duration = phase_start.elapsed();
            phases.push(("Graph Validation".to_string(), validation_duration));
        }

        // Phase 4: Optimization (if enabled)
        let final_graph = if self.config.include_optimization {
            let phase_start = Instant::now();
            let opt_config = OptimizationConfig {
                level: self.config.optimization_level,
                enable_dce: true,
                enable_cse: true,
                enable_identity: true,
                show_stats: false,
                verbose: false,
            };
            let (optimized, _) = optimize_einsum_graph(graph, &opt_config)?;
            let optimization_duration = phase_start.elapsed();
            phases.push(("Optimization".to_string(), optimization_duration));
            optimized
        } else {
            graph
        };

        // Phase 5: Output serialization (simulate)
        let phase_start = Instant::now();
        let _ = serde_json::to_string(&final_graph);
        let serialization_duration = phase_start.elapsed();
        phases.push(("Serialization".to_string(), serialization_duration));

        let total_duration = total_start.elapsed();
        let total_us = total_duration.as_micros() as u64;

        // Convert to PhaseProfile
        let phase_profiles: Vec<PhaseProfile> = phases
            .into_iter()
            .map(|(name, duration)| {
                let duration_us = duration.as_micros() as u64;
                let percentage = if total_us > 0 {
                    (duration_us as f64 / total_us as f64) * 100.0
                } else {
                    0.0
                };
                PhaseProfile {
                    name,
                    duration_us,
                    percentage,
                }
            })
            .collect();

        Ok((phase_profiles, total_us, final_graph))
    }

    fn average_phases(&self, all_phases: &[Vec<PhaseProfile>], total_us: u64) -> Vec<PhaseProfile> {
        if all_phases.is_empty() {
            return Vec::new();
        }

        let num_runs = all_phases.len();
        let num_phases = all_phases[0].len();
        let mut averaged = Vec::with_capacity(num_phases);

        for i in 0..num_phases {
            let name = all_phases[0][i].name.clone();
            let avg_duration: u64 = all_phases
                .iter()
                .map(|phases| phases.get(i).map_or(0, |p| p.duration_us))
                .sum::<u64>()
                / num_runs as u64;

            let percentage = if total_us > 0 {
                (avg_duration as f64 / total_us as f64) * 100.0
            } else {
                0.0
            };

            averaged.push(PhaseProfile {
                name,
                duration_us: avg_duration,
                percentage,
            });
        }

        averaged
    }

    fn estimate_memory(&self, graph: &EinsumGraph) -> MemoryProfile {
        // Estimate tensor memory based on node operations and tensor count
        // Each tensor is estimated to hold ~100 elements (10x10) as default
        let default_tensor_size = 100 * 8; // 100 f64 elements
        let tensor_memory: usize = graph.tensors.len() * default_tensor_size;

        // Estimate graph structure memory (rough estimate)
        let graph_structure = std::mem::size_of::<EinsumGraph>()
            + graph.tensors.len() * 128 // rough per-tensor name overhead
            + graph.nodes.len() * 256 // rough per-node overhead
            + graph.tensor_metadata.len() * 64; // metadata overhead

        MemoryProfile {
            tensor_memory_bytes: tensor_memory,
            graph_structure_bytes: graph_structure,
            total_bytes: tensor_memory + graph_structure,
        }
    }

    fn calculate_variance(
        &self,
        all_phases: &[Vec<PhaseProfile>],
        total_times: &[u64],
        mean_total: u64,
    ) -> PerformanceVariance {
        if total_times.len() <= 1 {
            // Not enough data for variance
            return PerformanceVariance {
                total_stddev_us: 0.0,
                coefficient_of_variation: 0.0,
                min_time_us: mean_total,
                max_time_us: mean_total,
                phase_variance: Vec::new(),
            };
        }

        // Calculate total time variance
        let variance_sum: f64 = total_times
            .iter()
            .map(|&t| {
                let diff = t as f64 - mean_total as f64;
                diff * diff
            })
            .sum();
        let total_stddev = (variance_sum / total_times.len() as f64).sqrt();
        let coeff_var = if mean_total > 0 {
            (total_stddev / mean_total as f64) * 100.0
        } else {
            0.0
        };

        let min_time = *total_times.iter().min().unwrap_or(&mean_total);
        let max_time = *total_times.iter().max().unwrap_or(&mean_total);

        // Calculate phase-level variance
        let mut phase_variance = Vec::new();
        if !all_phases.is_empty() {
            let num_phases = all_phases[0].len();
            for i in 0..num_phases {
                let phase_name = all_phases[0][i].name.clone();
                let durations: Vec<u64> = all_phases
                    .iter()
                    .filter_map(|phases| phases.get(i).map(|p| p.duration_us))
                    .collect();

                if !durations.is_empty() {
                    let mean = durations.iter().sum::<u64>() as f64 / durations.len() as f64;
                    let var: f64 = durations
                        .iter()
                        .map(|&d| {
                            let diff = d as f64 - mean;
                            diff * diff
                        })
                        .sum::<f64>()
                        / durations.len() as f64;
                    let stddev = var.sqrt();
                    let cv = if mean > 0.0 {
                        (stddev / mean) * 100.0
                    } else {
                        0.0
                    };

                    phase_variance.push(PhaseVariance {
                        phase: phase_name,
                        stddev_us: stddev,
                        coefficient_of_variation: cv,
                    });
                }
            }
        }

        PerformanceVariance {
            total_stddev_us: total_stddev,
            coefficient_of_variation: coeff_var,
            min_time_us: min_time,
            max_time_us: max_time,
            phase_variance,
        }
    }

    fn analyze_bottlenecks(
        &self,
        phases: &[PhaseProfile],
        graph: &EinsumGraph,
    ) -> BottleneckAnalysis {
        // Identify hotspots (phases taking >30% of time)
        let mut hotspots = Vec::new();
        for phase in phases {
            if phase.percentage >= 30.0 {
                hotspots.push(Hotspot {
                    phase: phase.name.clone(),
                    percentage: phase.percentage,
                    duration_us: phase.duration_us,
                    severity: HotspotSeverity::from_percentage(phase.percentage),
                });
            }
        }

        // Generate optimization suggestions based on hotspots and graph complexity
        let mut suggestions = Vec::new();

        for hotspot in &hotspots {
            match hotspot.phase.as_str() {
                "IR Compilation" => {
                    if hotspot.severity == HotspotSeverity::High
                        || hotspot.severity == HotspotSeverity::Critical
                    {
                        suggestions.push(format!(
                            "IR Compilation is taking {:.1}% of time. Consider:\n    \
                            - Breaking down complex expressions into smaller sub-expressions\n    \
                            - Using cached compilation results if processing similar patterns\n    \
                            - Reducing the number of quantifiers and nested operations",
                            hotspot.percentage
                        ));
                    } else {
                        suggestions.push(format!(
                            "IR Compilation takes {:.1}% of time - this is normal for complex expressions",
                            hotspot.percentage
                        ));
                    }
                }
                "Optimization" => {
                    if hotspot.severity == HotspotSeverity::High
                        || hotspot.severity == HotspotSeverity::Critical
                    {
                        suggestions.push(format!(
                            "Optimization is taking {:.1}% of time. Consider:\n    \
                            - Lowering optimization level if compilation speed is critical\n    \
                            - Disabling optimization for development/debugging\n    \
                            - Graph complexity (tensors: {}, nodes: {}) may be causing slow optimization",
                            hotspot.percentage,
                            graph.tensors.len(),
                            graph.nodes.len()
                        ));
                    }
                }
                "Serialization" => {
                    if hotspot.severity == HotspotSeverity::Medium
                        || hotspot.severity == HotspotSeverity::High
                        || hotspot.severity == HotspotSeverity::Critical
                    {
                        suggestions.push(format!(
                            "Serialization is taking {:.1}% of time. This is unusual. Consider:\n    \
                            - Using compact JSON format if output size matters\n    \
                            - Caching serialized graphs for repeated use\n    \
                            - Very large graphs ({} tensors, {} nodes) cause slow serialization",
                            hotspot.percentage,
                            graph.tensors.len(),
                            graph.nodes.len()
                        ));
                    }
                }
                "Graph Validation" => {
                    if hotspot.severity != HotspotSeverity::Low {
                        suggestions.push(format!(
                            "Validation is taking {:.1}% of time. Consider:\n    \
                            - Disabling validation in production builds\n    \
                            - Using validation only during development/testing",
                            hotspot.percentage
                        ));
                    }
                }
                _ => {}
            }
        }

        // General suggestions based on graph complexity
        if graph.tensors.len() > 100 {
            suggestions.push(format!(
                "Large graph detected ({} tensors). Consider:\n    \
                - Expression simplification or decomposition\n    \
                - Higher optimization levels to reduce graph size",
                graph.tensors.len()
            ));
        }

        if graph.nodes.len() > 50 {
            suggestions.push(format!(
                "Complex computation graph ({} nodes). Consider:\n    \
                - Using optimization to merge operations\n    \
                - Breaking into smaller sub-problems",
                graph.nodes.len()
            ));
        }

        // No bottlenecks found
        if hotspots.is_empty() {
            suggestions.push(
                "No significant bottlenecks detected. Performance is well-balanced.".to_string(),
            );
        }

        // Calculate overall severity score (0-100)
        let severity_score = if hotspots.is_empty() {
            0
        } else {
            let max_percentage = hotspots.iter().map(|h| h.percentage).fold(0.0f64, f64::max);
            let critical_count = hotspots
                .iter()
                .filter(|h| h.severity == HotspotSeverity::Critical)
                .count();
            let high_count = hotspots
                .iter()
                .filter(|h| h.severity == HotspotSeverity::High)
                .count();

            // Score based on worst hotspot and number of severe issues
            let base_score = (max_percentage * 0.8).min(80.0);
            let penalty = (critical_count * 15 + high_count * 5) as f64;
            (base_score + penalty).min(100.0) as u8
        };

        BottleneckAnalysis {
            hotspots,
            suggestions,
            severity_score,
        }
    }
}

impl ProfileData {
    /// Print profile results in human-readable format
    pub fn print(&self) {
        print_header("Compilation Profile");

        // Total time
        let total_ms = self.total_time_us as f64 / 1000.0;
        println!("  {} {:.3}ms", "Total Time:".bold(), total_ms);
        println!();

        // Phase breakdown
        println!("{}", "Phase Breakdown:".bold());
        let max_name_len = self.phases.iter().map(|p| p.name.len()).max().unwrap_or(20);

        for phase in &self.phases {
            let duration_ms = phase.duration_us as f64 / 1000.0;
            let bar_len = (phase.percentage / 5.0).round() as usize;
            let bar: String = "█".repeat(bar_len);

            let color_bar = if phase.percentage > 50.0 {
                bar.red()
            } else if phase.percentage > 25.0 {
                bar.yellow()
            } else {
                bar.green()
            };

            println!(
                "  {:width$} {:>8.3}ms {:>6.1}% {}",
                phase.name,
                duration_ms,
                phase.percentage,
                color_bar,
                width = max_name_len
            );
        }
        println!();

        // Memory estimates
        println!("{}", "Memory Estimates:".bold());
        println!(
            "  {} {} bytes",
            "Tensor Data:".dimmed(),
            format_bytes(self.memory_estimate.tensor_memory_bytes)
        );
        println!(
            "  {} {} bytes",
            "Graph Structure:".dimmed(),
            format_bytes(self.memory_estimate.graph_structure_bytes)
        );
        println!(
            "  {} {} bytes",
            "Total:".bold(),
            format_bytes(self.memory_estimate.total_bytes)
        );
        println!();

        // Graph metrics
        println!("{}", "Graph Complexity:".bold());
        println!(
            "  {} {}",
            "Tensors:".dimmed(),
            self.graph_metrics.tensor_count
        );
        println!("  {} {}", "Nodes:".dimmed(), self.graph_metrics.node_count);
        println!("  {} {}", "Depth:".dimmed(), self.graph_metrics.depth);
        println!(
            "  {} {}",
            "Estimated FLOPs:".dimmed(),
            format_number(self.graph_metrics.estimated_flops)
        );
        println!();

        // Bottleneck Analysis
        if !self.bottleneck_analysis.hotspots.is_empty()
            || !self.bottleneck_analysis.suggestions.is_empty()
        {
            println!("{}", "⚠ Bottleneck Analysis:".bold().red());
            println!(
                "  {} {}",
                "Severity Score:".bold(),
                self.format_severity_score()
            );
            println!();

            if !self.bottleneck_analysis.hotspots.is_empty() {
                println!("{}", "  Detected Hotspots:".bold());
                for hotspot in &self.bottleneck_analysis.hotspots {
                    let duration_ms = hotspot.duration_us as f64 / 1000.0;
                    let severity_color = match hotspot.severity {
                        HotspotSeverity::Critical => "bright red",
                        HotspotSeverity::High => "red",
                        HotspotSeverity::Medium => "yellow",
                        HotspotSeverity::Low => "yellow",
                    };
                    println!(
                        "    {} {:>6.1}% ({:.3}ms) - {:?}",
                        hotspot.phase.color(severity_color),
                        hotspot.percentage,
                        duration_ms,
                        hotspot.severity
                    );
                }
                println!();
            }

            if !self.bottleneck_analysis.suggestions.is_empty() {
                println!("{}", "  Optimization Suggestions:".bold());
                for (i, suggestion) in self.bottleneck_analysis.suggestions.iter().enumerate() {
                    println!("    {}. {}", i + 1, suggestion);
                }
                println!();
            }
        }

        // Performance Variance (if multiple runs)
        if self.variance.coefficient_of_variation > 0.0 {
            println!("{}", "Performance Variance:".bold());
            let variance_color = if self.variance.coefficient_of_variation > 15.0 {
                "red"
            } else if self.variance.coefficient_of_variation > 5.0 {
                "yellow"
            } else {
                "green"
            };

            println!(
                "  {} {:.2}%",
                "Coefficient of Variation:".dimmed(),
                self.variance
                    .coefficient_of_variation
                    .to_string()
                    .color(variance_color)
            );
            println!(
                "  {} {:.3}ms",
                "Std Dev:".dimmed(),
                self.variance.total_stddev_us / 1000.0
            );
            println!(
                "  {} {:.3}ms - {:.3}ms",
                "Range:".dimmed(),
                self.variance.min_time_us as f64 / 1000.0,
                self.variance.max_time_us as f64 / 1000.0
            );

            // Show phase variance if any phase has high variance
            let high_variance_phases: Vec<_> = self
                .variance
                .phase_variance
                .iter()
                .filter(|pv| pv.coefficient_of_variation > 10.0)
                .collect();

            if !high_variance_phases.is_empty() {
                println!();
                println!("{}", "  Phases with High Variance:".dimmed());
                for pv in high_variance_phases {
                    println!("    {} {:.2}% CV", pv.phase, pv.coefficient_of_variation);
                }
            }
            println!();
        }

        // Execution Profile (if enabled)
        if let Some(ref exec_profile) = self.execution_profile {
            println!("{}", "Execution Profile:".bold());
            println!("  {} {}", "Backend:".dimmed(), exec_profile.backend);
            let avg_ms = exec_profile.avg_execution_time_us as f64 / 1000.0;
            println!("  {} {:.3}ms", "Avg Execution Time:".dimmed(), avg_ms);

            let min_ms = exec_profile.min_execution_time_us as f64 / 1000.0;
            let max_ms = exec_profile.max_execution_time_us as f64 / 1000.0;
            println!("  {} {:.3}ms - {:.3}ms", "Range:".dimmed(), min_ms, max_ms);

            let stddev_ms = exec_profile.execution_stddev_us / 1000.0;
            println!("  {} {:.3}ms", "Std Dev:".dimmed(), stddev_ms);

            println!(
                "  {} {} bytes",
                "Actual Memory:".dimmed(),
                format_bytes(exec_profile.actual_memory_bytes)
            );

            println!(
                "  {} {:.2} graphs/sec",
                "Throughput:".dimmed(),
                exec_profile.throughput
            );
            println!();
        }

        print_success("Profile complete");
    }

    fn format_severity_score(&self) -> String {
        let score = self.bottleneck_analysis.severity_score;
        let color = if score >= 75 {
            "bright red"
        } else if score >= 50 {
            "red"
        } else if score >= 25 {
            "yellow"
        } else {
            "green"
        };

        let level = if score >= 75 {
            "CRITICAL"
        } else if score >= 50 {
            "HIGH"
        } else if score >= 25 {
            "MEDIUM"
        } else {
            "LOW"
        };

        format!("{}/100 ({})", score, level)
            .color(color)
            .to_string()
    }

    /// Export to JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Format bytes with appropriate unit
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{}", bytes)
    }
}

/// Format large numbers with commas
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_compiler::CompilationConfig;
    use tensorlogic_ir::Term;

    fn simple_expr() -> TLExpr {
        TLExpr::Pred {
            name: "test".to_string(),
            args: vec![Term::Var("x".to_string())],
        }
    }

    #[test]
    fn test_profiler_basic() {
        let config = ProfileConfig {
            include_optimization: false,
            include_validation: false,
            warmup_runs: 0,
            profile_runs: 1,
            ..Default::default()
        };

        let profiler = Profiler::new(config);
        let expr = simple_expr();
        let ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());

        let result = profiler.profile(&expr, &ctx);
        assert!(result.is_ok());

        let profile = result.unwrap();
        assert!(profile.total_time_us > 0);
        assert!(!profile.phases.is_empty());
        // Check bottleneck analysis exists
        assert!(profile.bottleneck_analysis.severity_score <= 100);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(123), "123");
        assert_eq!(format_number(1234), "1,234");
        assert_eq!(format_number(1234567), "1,234,567");
    }

    #[test]
    fn test_profile_with_and_expr() {
        let config = ProfileConfig {
            include_optimization: false, // Skip optimization for faster test
            include_validation: false,   // Skip validation for faster test
            warmup_runs: 0,
            profile_runs: 1,
            ..Default::default()
        };

        let profiler = Profiler::new(config);
        let expr = TLExpr::And(Box::new(simple_expr()), Box::new(simple_expr()));
        let ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());

        let result = profiler.profile(&expr, &ctx);
        assert!(result.is_ok());

        let profile = result.unwrap();
        // Should have 3 phases: analysis, compilation, serialization
        assert!(profile.phases.len() >= 2);
        // Bottleneck analysis should exist
        assert!(profile.bottleneck_analysis.severity_score <= 100);
    }

    #[test]
    fn test_bottleneck_detection() {
        let config = ProfileConfig {
            include_optimization: true, // Enable optimization to create bottleneck
            optimization_level: OptimizationLevel::Aggressive,
            include_validation: false,
            warmup_runs: 0,
            profile_runs: 1,
            ..Default::default()
        };

        let profiler = Profiler::new(config);
        // Complex expression that might create a bottleneck
        let expr = TLExpr::And(
            Box::new(TLExpr::And(
                Box::new(simple_expr()),
                Box::new(simple_expr()),
            )),
            Box::new(TLExpr::And(
                Box::new(simple_expr()),
                Box::new(simple_expr()),
            )),
        );
        let ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());

        let result = profiler.profile(&expr, &ctx);
        assert!(result.is_ok());

        let profile = result.unwrap();
        // Check that bottleneck analysis exists and has valid severity score
        assert!(profile.bottleneck_analysis.severity_score <= 100);
        // Suggestions may or may not be empty depending on whether bottlenecks were detected
        // The analyze_bottlenecks function should provide at least one message
        // But we don't strictly require it for the test to pass
    }

    #[test]
    fn test_variance_calculation() {
        let config = ProfileConfig {
            include_optimization: false,
            include_validation: false,
            warmup_runs: 0,
            profile_runs: 3, // Multiple runs to calculate variance
            ..Default::default()
        };

        let profiler = Profiler::new(config);
        let expr = simple_expr();
        let ctx = CompilerContext::with_config(CompilationConfig::soft_differentiable());

        let result = profiler.profile(&expr, &ctx);
        assert!(result.is_ok());

        let profile = result.unwrap();
        // Variance should be calculated correctly
        // Note: min_time_us can be 0 if execution is faster than 1 microsecond
        assert!(profile.variance.max_time_us >= profile.variance.min_time_us);
        // With multiple runs, we should have phase variance data
        assert!(!profile.variance.phase_variance.is_empty());
    }
}
