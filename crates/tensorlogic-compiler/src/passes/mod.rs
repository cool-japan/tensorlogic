//! Compiler passes (validation, optimization, etc.).

pub mod advanced_analysis;
pub mod contraction_opt;
pub mod cse;
pub mod dataflow;
pub mod diagnostics;
pub mod einsum_opt;
pub mod graph_opt_integration;
pub mod loop_fusion;
pub mod metadata_propagation;
pub mod post_compilation;
pub mod reachability;
pub mod scope_analysis;
pub mod strategy_selection;
pub mod symbol_integration;
pub mod type_checking;
pub mod validation;

pub use advanced_analysis::{
    analyze_graph, print_report, quick_analyze, AnalysisReport, OptimizationRecommendation,
    ParallelOpportunity, RecommendationCategory,
};
pub use contraction_opt::{
    analyze_contraction_path, optimize_contractions, optimize_contractions_with_config,
    ContractionOptConfig, ContractionOptStats, ContractionPath, TensorShape,
};
pub use cse::{eliminate_common_subexpressions, CseResult};
pub use dataflow::{
    analyze_dataflow, analyze_dataflow_with_config, analyze_graph_dataflow, DataflowAnalysis,
    DataflowConfig, GraphDataflow,
};
pub use diagnostics::{
    diagnose_expression, enhance_error, Diagnostic, DiagnosticBuilder, DiagnosticLevel,
};
pub use einsum_opt::{optimize_einsum_graph, EinsumOptResult};
pub use graph_opt_integration::{
    apply_graph_optimizations, apply_pattern_optimizations, quick_optimize,
    recommend_optimizations, GraphOptConfig, GraphOptStats,
};
pub use loop_fusion::{
    estimate_fusion_benefit, fuse_loops, fuse_loops_with_config, LoopFusionConfig, LoopFusionStats,
};
pub use metadata_propagation::{
    attach_expr_metadata, propagate_metadata, MetadataBuilder, MetadataCompilationResult,
};
pub use post_compilation::{
    post_compilation_passes, quick_validate, PostCompilationOptions, PostCompilationResult,
};
pub use reachability::{
    analyze_dominance, analyze_reachability, DominanceAnalysis, ReachabilityAnalysis,
};
pub use scope_analysis::{
    analyze_scopes, suggest_quantifiers, validate_scopes, ScopeAnalysisResult,
};
pub use strategy_selection::{
    recommend_strategy, ExpressionProfile, OptimizationGoal, StrategyRecommendation,
};
pub use symbol_integration::{
    build_signature_registry, export_domains, import_domains, sync_context_with_symbol_table,
};
pub use type_checking::{infer_types, TypeChecker};
pub use validation::{
    validate_arity, validate_expression, validate_expression_with_types, ValidationResult,
};
