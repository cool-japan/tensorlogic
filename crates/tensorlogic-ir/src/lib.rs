//! # TensorLogic IR
//!
//! **Engine-agnostic Abstract Syntax Tree & Intermediate Representation for TensorLogic**
//!
//! **Version**: 0.1.0-alpha.1 | **Status**: Production Ready
//!
//! This crate provides the core data structures and operations for representing logic-as-tensor
//! computations in the TensorLogic framework. It serves as the foundational layer that all other
//! TensorLogic components build upon.
//!
//! ## Overview
//!
//! TensorLogic IR enables the compilation of logical rules (predicates, quantifiers, implications)
//! into **tensor equations (einsum graphs)** with a minimal DSL + IR. This approach allows for:
//!
//! - Neural, symbolic, and probabilistic models within a unified tensor computation framework
//! - Engine-agnostic representation that can be executed on different backends
//! - Static analysis and optimization of logical expressions before execution
//! - Type-safe construction and validation of logical expressions
//!
//! ## Core Components
//!
//! ### Terms ([`Term`])
//! Variables and constants that appear in logical expressions:
//! - **Variables**: Free or bound variables (e.g., `x`, `y`, `Person`)
//! - **Constants**: Concrete values (e.g., `alice`, `bob`, `42`)
//! - **Typed terms**: Terms with explicit type annotations
//!
//! ### Logical Expressions ([`TLExpr`])
//! The DSL for expressing logical rules:
//! - **Predicates**: Atomic propositions like `Person(x)` or `knows(alice, bob)`
//! - **Logical connectives**: AND (∧), OR (∨), NOT (¬), Implication (→)
//! - **Quantifiers**: Existential (∃) and Universal (∀) with domain constraints
//! - **Arithmetic**: Addition, subtraction, multiplication, division
//! - **Comparisons**: Equality, less than, greater than, etc.
//! - **Control flow**: If-then-else conditionals
//!
//! ### Tensor Computation Graphs ([`EinsumGraph`])
//! The compiled IR representing tensor operations:
//! - **Tensors**: Named tensor values with indices
//! - **Nodes**: Operations on tensors (einsum, element-wise, reductions)
//! - **Outputs**: Designated output tensors
//! - **Validation**: Comprehensive graph validation and error checking
//!
//! ## Features
//!
//! ### Type System
//! - Static type checking with [`PredicateSignature`] and [`TypeAnnotation`]
//! - Arity validation ensures consistent predicate usage
//! - Type inference and compatibility checking
//!
//! ### Domain Constraints
//! - Domain management via [`DomainRegistry`] and [`DomainInfo`]
//! - Built-in domains: Bool, Int, Real, Nat, Probability
//! - Custom finite and infinite domains
//! - Domain validation for quantified variables
//!
//! ### Metadata & Provenance
//! - Source location tracking with [`SourceLocation`] and [`SourceSpan`]
//! - Provenance information via [`Provenance`] for rule tracking
//! - Custom metadata support for nodes and expressions
//!
//! ### Serialization
//! - Full serde support for JSON and binary formats
//! - Versioned serialization with [`VersionedExpr`] and [`VersionedGraph`]
//! - Backward compatibility checking
//!
//! ### Analysis & Utilities
//! - Free variable analysis
//! - Predicate extraction and counting
//! - Graph statistics with [`GraphStats`]
//! - Expression statistics with [`ExprStats`]
//! - Pretty printing and DOT export for visualization
//! - Expression and graph diffing with [`diff_exprs`] and [`diff_graphs`]
//!
//! ## Quick Start
//!
//! ### Creating Logical Expressions
//!
//! ```rust
//! use tensorlogic_ir::{TLExpr, Term};
//!
//! // Simple predicate: Person(x)
//! let person = TLExpr::pred("Person", vec![Term::var("x")]);
//!
//! // Logical rule: ∀x. Person(x) → Mortal(x)
//! let mortal = TLExpr::pred("Mortal", vec![Term::var("x")]);
//! let rule = TLExpr::forall("x", "Entity", TLExpr::imply(person, mortal));
//!
//! // Verify no free variables (all bound by quantifier)
//! assert!(rule.free_vars().is_empty());
//! ```
//!
//! ### Building Computation Graphs
//!
//! ```rust
//! use tensorlogic_ir::{EinsumGraph, EinsumNode};
//!
//! let mut graph = EinsumGraph::new();
//!
//! // Matrix multiplication: C = A @ B
//! let a = graph.add_tensor("A");
//! let b = graph.add_tensor("B");
//! let c = graph.add_tensor("C");
//!
//! graph.add_node(EinsumNode::einsum("ik,kj->ij", vec![a, b], vec![c])).unwrap();
//! graph.add_output(c).unwrap();
//!
//! // Validate the graph
//! assert!(graph.validate().is_ok());
//! ```
//!
//! ### Domain Management
//!
//! ```rust
//! use tensorlogic_ir::{DomainRegistry, DomainInfo, TLExpr, Term};
//!
//! // Create registry with built-in domains
//! let mut registry = DomainRegistry::with_builtins();
//!
//! // Add custom domain
//! registry.register(DomainInfo::finite("Person", 100)).unwrap();
//!
//! // Create and validate expression
//! let expr = TLExpr::exists("x", "Person", TLExpr::pred("P", vec![Term::var("x")]));
//! assert!(expr.validate_domains(&registry).is_ok());
//! ```
//!
//! ## Examples
//!
//! See the `examples/` directory for comprehensive demonstrations:
//! - `00_basic_expressions`: Simple predicates and logical connectives
//! - `01_quantifiers`: Existential and universal quantifiers
//! - `02_arithmetic`: Arithmetic and comparison operations
//! - `03_graph_construction`: Building computation graphs
//! - `05_serialization`: JSON and binary serialization
//! - `06_visualization`: Pretty printing and DOT export
//!
//! ## Architecture
//!
//! The crate is organized into focused modules:
//! - **term**: Variables, constants, and type annotations
//! - **expr**: Logical expressions and builders
//! - **graph**: Tensor computation graphs and nodes
//! - **domain**: Domain constraints and validation
//! - **signature**: Type signatures for predicates
//! - **metadata**: Provenance and source tracking
//! - **[`serialization`]**: Versioned JSON/binary formats
//! - **[`util`]**: Pretty printing and statistics
//! - **[`diff`]**: Expression and graph comparison
//! - **error**: Comprehensive error types
//!
//! ## Logic-to-Tensor Mapping
//!
//! | Logic Operation | Tensor Equivalent | Notes |
//! |----------------|-------------------|-------|
//! | `AND(a, b)` | `a * b` | Hadamard product (element-wise) |
//! | `OR(a, b)` | `max(a, b)` | Or soft variant (configurable) |
//! | `NOT(a)` | `1 - a` | Complement operation |
//! | `∃x. P(x)` | `sum(P, axis=x)` | Or `max` for hard quantification |
//! | `∀x. P(x)` | `NOT(∃x. NOT(P(x)))` | Dual of existential |
//! | `a → b` | `ReLU(b - a)` | Soft implication |
//!
//! ## Performance
//!
//! - **Lazy validation**: Operations are validated only when explicitly requested
//! - **Zero-copy indices**: Graph operations use tensor indices instead of cloning
//! - **Incremental building**: Graphs can be built step-by-step efficiently
//! - **Property tests**: 30 randomized tests ensure correctness
//! - **Benchmarks**: 40+ performance tests measure all operations
//!
//! ## See Also
//!
//! - **tensorlogic-compiler**: Compiles TLExpr → EinsumGraph
//! - **tensorlogic-infer**: Execution and autodiff traits
//! - **tensorlogic-scirs-backend**: SciRS2-powered runtime execution
//! - **tensorlogic-adapters**: Symbol tables and axis metadata

pub mod diff;
mod display;
mod domain;
mod error;
mod expr;
pub mod fuzzing;
mod graph;
mod metadata;
pub mod serialization;
mod signature;
mod term;
pub mod util;

#[cfg(test)]
mod tests;

pub use diff::{diff_exprs, diff_graphs, ExprDiff, GraphDiff, NodeDiff};
pub use domain::{DomainInfo, DomainRegistry, DomainType};
pub use error::IrError;
pub use expr::ac_matching::{
    ac_equivalent, flatten_ac, normalize_ac, ACOperator, ACPattern, Multiset,
};
pub use expr::advanced_analysis::{ComplexityMetrics, OperatorCounts, PatternAnalysis};
pub use expr::advanced_rewriting::{
    AdvancedRewriteSystem, ConditionalRule, RewriteConfig, RewriteStats, RewriteStrategy,
    RulePriority,
};
pub use expr::confluence::{
    are_joinable, normalize, ConfluenceChecker, ConfluenceReport, CriticalPair,
};
pub use expr::defuzzification::{
    bisector, centroid, defuzzify, largest_of_maximum, mean_of_maximum, smallest_of_maximum,
    weighted_average, DefuzzificationMethod, FuzzySet, SingletonFuzzySet,
};
pub use expr::distributive_laws::{apply_distributive_laws, DistributiveStrategy};
pub use expr::ltl_ctl_utilities::{
    apply_advanced_ltl_equivalences, classify_temporal_formula, compute_temporal_complexity,
    decompose_safety_liveness, extract_state_predicates, extract_temporal_subformulas,
    identify_temporal_pattern, is_temporal, is_temporal_nnf, TemporalClass, TemporalComplexity,
    TemporalPattern,
};
pub use expr::modal_axioms::{
    apply_axiom_k, apply_axiom_t, extract_modal_subformulas, is_modal_free, is_theorem_in_system,
    modal_depth, normalize_s5, verify_axiom_4, verify_axiom_5, verify_axiom_b, verify_axiom_d,
    verify_axiom_k, verify_axiom_t, ModalSystem,
};
pub use expr::modal_equivalences::apply_modal_equivalences;
pub use expr::normal_forms::{is_cnf, is_dnf, to_cnf, to_dnf, to_nnf};
pub use expr::optimization::{
    algebraic_simplify, constant_fold, optimize_expr, propagate_constants,
};
pub use expr::optimization_pipeline::{
    OptimizationLevel, OptimizationMetrics, OptimizationPass, OptimizationPipeline, PipelineConfig,
};
pub use expr::probabilistic_reasoning::{
    compute_tight_bounds, extract_probabilistic_semantics, mln_probability,
    propagate_probabilities, CredalSet, ProbabilityInterval,
};
pub use expr::rewriting::{Pattern, RewriteRule, RewriteSystem};
pub use expr::strategy_selector::{auto_optimize, ExpressionProfile, StrategySelector};
pub use expr::temporal_equivalences::apply_temporal_equivalences;
pub use expr::{
    AggregateOp, FuzzyImplicationKind, FuzzyNegationKind, TCoNormKind, TLExpr, TNormKind,
};
pub use graph::cost_model::{
    auto_annotate_costs, estimate_graph_cost, estimate_operation_cost, CostSummary, GraphCostModel,
    OperationCost,
};
pub use graph::{
    are_graphs_equivalent, canonical_hash, canonicalize_graph, export_to_dot,
    export_to_dot_with_options, validate_graph, DotExportOptions, EinsumGraph, EinsumNode,
    GraphValidationStats, OpType, ValidationError, ValidationErrorKind, ValidationReport,
    ValidationWarning, ValidationWarningKind,
};
pub use metadata::{Metadata, Provenance, SourceLocation, SourceSpan};
pub use serialization::{VersionedExpr, VersionedGraph, FORMAT_VERSION};
pub use signature::{PredicateSignature, SignatureRegistry};
pub use term::{Term, TypeAnnotation};
pub use util::{pretty_print_expr, pretty_print_graph, ExprStats, GraphStats};
