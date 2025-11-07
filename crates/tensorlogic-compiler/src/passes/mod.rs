//! Compiler passes (validation, optimization, etc.).

pub mod cse;
pub mod diagnostics;
pub mod einsum_opt;
pub mod metadata_propagation;
pub mod post_compilation;
pub mod scope_analysis;
pub mod strategy_selection;
pub mod symbol_integration;
pub mod type_checking;
pub mod validation;

pub use cse::{eliminate_common_subexpressions, CseResult};
pub use diagnostics::{
    diagnose_expression, enhance_error, Diagnostic, DiagnosticBuilder, DiagnosticLevel,
};
pub use einsum_opt::{optimize_einsum_graph, EinsumOptResult};
pub use metadata_propagation::{
    attach_expr_metadata, propagate_metadata, MetadataBuilder, MetadataCompilationResult,
};
pub use post_compilation::{
    post_compilation_passes, quick_validate, PostCompilationOptions, PostCompilationResult,
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
