# TensorLogic IR — TODO

**Status**: Stable | **Version**: 0.1.3 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-31
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

Intermediate representation, DSL types, and einsum graph for logical rules.

## Completed

### Core Infrastructure
- [x] Core AST types (Term, TLExpr)
- [x] EinsumGraph structure
- [x] EinsumNode with OpType variants
- [x] Graph validation
- [x] Free variable analysis
- [x] Arity validation
- [x] Builder methods for TLExpr
- [x] Serialization (serde)

### Type System Enhancement - PRODUCTION READY
- [x] Add type annotations to Terms
  - [x] Term::Typed { value, type_annotation }
  - [x] TypeAnnotation struct with type metadata
  - [x] Helper methods (typed_var, typed_const, with_type, get_type)
- [x] Predicate signatures
  - [x] PredicateSignature with arity and type validation
  - [x] SignatureRegistry for managing predicate metadata
  - [x] Type matching and compatibility checking
- [x] Enhanced error types
  - [x] ArityMismatch, TypeMismatch errors
  - [x] UnboundVariable, InconsistentTypes errors

### Graph Optimization - PRODUCTION READY
- [x] Dead code elimination in EinsumGraph
  - [x] Remove unused tensors
  - [x] Prune unreachable nodes
  - [x] Backward pass liveness analysis
- [x] Common subexpression detection
  - [x] Find duplicate subgraphs
  - [x] Node hashing for deduplication
  - [x] Replacement mapping
- [x] Graph simplification
  - [x] Eliminate identity operations
  - [x] Multi-pass optimization pipeline
  - [x] OptimizationStats tracking

### Metadata Support - PRODUCTION READY
- [x] Source location tracking
  - [x] SourceLocation with file/line/column
  - [x] SourceSpan for ranges
  - [x] Display formatting
- [x] Provenance metadata
  - [x] Provenance struct with rule IDs
  - [x] Source file tracking
  - [x] Custom attributes support
- [x] Debug information
  - [x] Metadata container for IR nodes
  - [x] Human-readable names
  - [x] Attribute key-value pairs

## High Priority - DONE

### Domain Constraints - PRODUCTION READY
- [x] Attach domain info to quantifiers (already in Exists/ForAll)
- [x] DomainInfo struct with type categories and constraints
- [x] DomainRegistry for managing domains
- [x] Domain validation methods
- [x] Domain compatibility and casting checks
- [x] Built-in domains (Bool, Int, Real, Nat, Probability)
- [x] Domain validation in TLExpr (validate_domains, referenced_domains)

## Medium Priority - DONE

### Expression Extensions - IMPLEMENTED
- [x] Arithmetic operations
  - [x] Add, Subtract, Multiply, Divide
  - [x] Mixed logical/arithmetic expressions
  - [x] Element-wise tensor operations
- [x] Comparison operations
  - [x] Equal, LessThan, GreaterThan, LessThanOrEqual, GreaterThanOrEqual
  - [x] Integration with logical ops
- [x] Conditional expressions
  - [x] If-then-else with soft probabilistic semantics
  - [x] Compiles to: cond * then + (1-cond) * else
- [x] Numeric constants
  - [x] Constant(f64) variant for scalar literals
  - [x] `ir-pattern-matching-deferred` (planned 2026-04-17)
    - **Goal:** Add pattern matching to the IR — `Pattern` enum + `TLExpr::Match { scrutinee, arms }` — and lower to nested `IfThenElse` in the compiler. Scope locked to design (A) minimal: `Pattern::ConstSymbol(String) | ConstNumber(f64) | Wildcard`. Last arm MUST be `Wildcard` (validation enforced). Variable-binding patterns and the alpha-rename fix are out of scope this run — design (A) has no binders.
    - **Design:** New module `src/pattern.rs`: `pub enum Pattern { ConstSymbol(String), ConstNumber(f64), Wildcard }` + Display + serde shim. `TLExpr::Match { scrutinee: Box<TLExpr>, arms: Vec<(Pattern, Box<TLExpr>)> }` added to `src/expr/mod.rs`. Validation rule: `arms.last().0 == Pattern::Wildcard` else `ValidationError::NonExhaustivePatternMatch`. Lowering in `tensorlogic-compiler/src/compile/pattern_match.rs` (NEW): translate Match into nested IfThenElse using compile_if_then_else as template; each arm condition is Eq(scrutinee, pattern_constant); wildcard arm is final else. Sequencing — stub-all-then-implement: add Match arm returning UnsupportedExpr/Unknown in every exhaustive match site first, keep compilation green, then add real lowering/display/serde/analysis.
    - **Files:** IR (~14): `src/expr/mod.rs`, `src/expr/analysis.rs`, `src/expr/validation.rs`, `src/expr/optimization/substitution.rs`, `src/expr_serialize/binary.rs` (TAG_75=Match, TAG_76=Pattern), `src/expr_serialize/sexpr.rs`, `src/expr_serialize/mod.rs`, `src/display.rs`, `src/pretty_print.rs`, `src/diff.rs`, `src/pattern.rs` (NEW), `src/lib.rs`. Compiler (~14): `src/compile/mod.rs`, `src/compile/pattern_match.rs` (NEW), `src/bytecode.rs`, `src/type_infer.rs`, `src/const_prop.rs`, `src/expr_diff.rs`, `src/complexity.rs`, `src/inline/substitute.rs`.
    - **Prerequisites:** none. Substitution alpha-rename gap is a separate future plan item (irrelevant for design (A) which has no binders).
    - **Tests:** `crates/tensorlogic-ir/tests/pattern_smoke.rs` (NEW): construct Match, validate exhaustiveness, round-trip binary serde, round-trip s-expression serde, display + pretty-print snapshots. `crates/tensorlogic-compiler/tests/pattern_lowering.rs` (NEW): Symbol equality, Number equality, wildcard fallback, three-arm cascade.
    - **Risk:** ~28-file cascade is tedious — mitigation: stub-all-then-implement keeps every intermediate state compilable.
  - **Refinement (2026-04-17):** Audit confirms implementation is complete across all sites. Only the two integration test files remain. No production-code changes needed.
    - Create `crates/tensorlogic-ir/tests/pattern_smoke.rs`: construct `match_expr` instances, assert validation rejects no-arms and non-Wildcard tail, binary serde round-trip, s-expr serde round-trip, Display + pretty_print golden strings.
    - Create `crates/tensorlogic-compiler/tests/pattern_lowering.rs`: compile and execute several scrutinee/arm combinations (ConstSymbol + Wildcard, multi-arm Symbol cascade, ConstNumber + Wildcard, three-arm cascade), assert returned `VmValue` matches correct arm. Test wildcard fallthrough. Test binary serde round-trip of the lowered IR.
- [x] Aggregations - INFRASTRUCTURE READY (temporarily disabled)
  - [x] AggregateOp enum (Count, Sum, Average, Max, Min, Product, Any, All)
  - [x] Aggregate variant with group-by support
  - [x] Builder methods (aggregate, count, sum, average, max, min)
  - Note: Temporarily disabled pending compiler integration

### Graph Features - PRODUCTION READY
- [x] Subgraph extraction
  - [x] extract_subgraph method
  - [x] Dependency tracking
  - [x] Tensor remapping
- [x] Graph merging
  - [x] merge method with tensor reuse
  - [x] Shared tensor deduplication
  - [x] Output preservation
- [x] Graph transformation API
  - [x] GraphVisitor trait
  - [x] GraphMutVisitor trait
  - [x] apply_rewrite method
  - [x] Utility methods (tensor_consumers, tensor_producer, has_path, dependencies)
  - [x] Node and tensor counting

### Serialization - PRODUCTION READY
- [x] Better JSON format
  - [x] Preserve structure with VersionedExpr/VersionedGraph wrappers
  - [x] Human-readable pretty JSON format
  - [x] Version tagging (semver "1.0.0")
  - [x] ISO 8601 timestamps
  - [x] Custom metadata support
  - [x] Version compatibility checking
- [x] Binary format
  - [x] Fast serialization using bincode
  - [x] Compact representation
  - [x] Roundtrip tests for both JSON and binary
  - [x] 10 comprehensive tests
- [x] Graph exchange formats
  - [x] ONNX text export (10 tests)
  - [x] TorchScript text export (10 tests)
  - [x] Custom export options
  - [x] Example: 18_graph_export.rs

## Low Priority - DONE

### Documentation - COMPLETED
- [x] Add README.md
  - [x] Comprehensive overview with badges
  - [x] Quick start guide with examples
  - [x] All features documented
  - [x] Production-ready status highlighted
  - [x] Ecosystem integration explained
- [x] Examples of IR construction
  - [x] 00_basic_expressions: Simple predicates, logical connectives, free variables
  - [x] 01_quantifiers: Existential and universal quantifiers with domains
  - [x] 02_arithmetic: Arithmetic operations, comparisons, conditionals
  - [x] 03_graph_construction: Building computation graphs
  - [x] 04_optimization: Graph optimization
  - [x] 05_serialization: JSON and binary serialization
  - [x] 06_visualization: Pretty printing and DOT export
- [x] Rustdoc for all types
  - [x] Comprehensive module-level documentation in lib.rs
  - [x] Quick start examples with code
  - [x] Architecture overview
  - [x] Logic-to-tensor mapping reference
  - [x] Links to related crates
  - [x] Zero rustdoc warnings

### Testing - ENHANCED
- [x] Property-based tests
  - [x] Random TLExpr generation with proptest
  - [x] Invariant checking (free vars, predicates, cloning)
  - [x] Serialization roundtrips
  - [x] Normal forms property tests (5 tests: NNF, CNF, DNF idempotency & validity)
  - [x] Modal/temporal logic property tests (9 tests: free var preservation, predicates)
  - [x] Graph canonicalization property tests (2 tests: idempotency, hash equality)
  - [x] 44 property tests total (43 passing, 1 ignored)
  - [x] Coverage: expressions, graphs, domains, terms, normal forms, modal/temporal logic
- [x] Fuzzing
  - [x] FuzzStats for tracking test results
  - [x] Expression operation fuzzing (free_vars, all_predicates, clone, debug, serde)
  - [x] Graph validation fuzzing
  - [x] Stress test generators
  - [x] Edge case testing
  - [x] Invariant checking
  - [x] 7 comprehensive fuzzing tests
- [x] Performance benchmarks
  - [x] Expression construction (5 benchmarks)
  - [x] Free variable analysis (4 benchmarks)
  - [x] Arity validation (3 benchmarks)
  - [x] Graph construction (4 benchmarks)
  - [x] Graph validation (3 benchmarks)
  - [x] Serialization (8 benchmarks)
  - [x] Domain operations (4 benchmarks)
  - [x] Cloning performance (3 benchmarks)
  - [x] Throughput testing (6 benchmarks)

### Utilities - PRODUCTION READY
- [x] Pretty printing
  - [x] TLExpr to readable format (pretty_print_expr)
  - [x] Graph visualization (pretty_print_graph)
  - [x] Indented, structured output
- [x] IR statistics
  - [x] ExprStats: node count, depth, free vars, operator counts
  - [x] GraphStats: tensor/node counts, operation breakdown, averages
  - [x] Complexity metrics
- [x] IR diff tool
  - [x] Compare two expressions (diff_exprs)
  - [x] Compare two graphs (diff_graphs)
  - [x] Show differences with detailed descriptions
  - [x] ExprDiff: TypeMismatch, PredicateMismatch, SubexprMismatch, QuantifierMismatch
  - [x] GraphDiff: tensor/node differences, operation differences, output differences
  - [x] Summary generation for quick overview
  - [x] 9 comprehensive tests

## Normal Forms - PRODUCTION READY
- [x] Negation Normal Form (NNF) transformation
- [x] Conjunctive Normal Form (CNF) transformation
- [x] Disjunctive Normal Form (DNF) transformation
- [x] Implication elimination and De Morgan's laws
- [x] Double negation elimination
- [x] Quantifier negation handling
- [x] Form validation predicates (is_cnf, is_dnf)
- [x] 17 comprehensive tests (all passing)

## Graph Canonicalization - PRODUCTION READY
- [x] Topological sorting of tensors and nodes
- [x] Canonical tensor naming (t0, t1, t2, ...)
- [x] Deterministic graph ordering
- [x] Graph equivalence checking
- [x] Canonical hash computation for deduplication
- [x] Cyclic graph detection
- [x] 10 comprehensive tests (all passing)

## Modal Logic Operators - PRODUCTION READY
- [x] Box operator for necessity
- [x] Diamond operator for possibility
- [x] Builder methods (modal_box, modal_diamond)
- [x] Display implementations
- [x] Full integration with all analysis/optimization passes
- [x] Documentation with formal semantics

## Temporal Logic Operators - PRODUCTION READY
- [x] Next operator (X) for next state
- [x] Eventually operator (F) for future states
- [x] Always operator (G) for all future states
- [x] Until operator (U) for temporal sequences
- [x] Builder methods (next, eventually, always, until)
- [x] Display implementations (X, F, G, U)
- [x] Full integration with all analysis/optimization passes
- [x] Documentation with formal semantics

## Advanced Algebraic Simplification - PRODUCTION READY
- [x] Logical laws (idempotence, absorption, identity, annihilation, complement)
- [x] Implication simplifications
- [x] Comparison simplifications
- [x] Arithmetic simplifications
- [x] Modal logic simplifications
- [x] Temporal logic simplifications
- [x] 39 comprehensive tests for all new simplification rules
- [x] Integration with existing optimization pipeline

## Optimization Pipeline System - PRODUCTION READY
- [x] OptimizationPipeline orchestrator with automatic pass ordering
  - [x] 10 optimization passes
  - [x] Priority-based automatic ordering
  - [x] Convergence detection
  - [x] Maximum iteration control
  - [x] Custom pass sequences
- [x] OptimizationLevel system (None, Basic, Standard, Aggressive)
- [x] OptimizationMetrics tracking
- [x] PipelineConfig for customization
- [x] 12 comprehensive tests for pipeline orchestration

## Automatic Strategy Selection - PRODUCTION READY
- [x] ExpressionProfile analysis
- [x] StrategySelector with intelligent recommendations
- [x] Heuristics for strategy selection
- [x] auto_optimize convenience function
- [x] 13 comprehensive tests for strategy selection

## Distributive Law Transformations - PRODUCTION READY
- [x] AND over OR distribution
- [x] OR over AND distribution
- [x] Quantifier distribution
- [x] Modal operator distribution
- [x] Strategy-based application
- [x] 10 comprehensive tests
- [x] Full integration with expression transformations

## Cost Model Annotations - PRODUCTION READY
- [x] OperationCost structure with multiple cost components
- [x] GraphCostModel for entire graph annotations
- [x] Cost estimation functions
- [x] Auto-annotation with heuristic estimates
- [x] Cost composition (add for sequential, max for parallel)
- [x] CostSummary for reporting
- [x] 10 comprehensive tests

## Advanced Term Rewriting System - PRODUCTION READY
- [x] Conditional rewrite rules with guards and predicates
- [x] Advanced rewriting strategies (Innermost, Outermost, BottomUp, TopDown, etc.)
- [x] Termination detection and cycle prevention
- [x] Associative-Commutative (AC) pattern matching
- [x] Confluence checking and critical pair analysis
- [x] 7 comprehensive tests for advanced rewriting
- [x] 8 tests for AC matching
- [x] 7 tests for confluence checking

## Modal Logic Axiom Systems - PRODUCTION READY
- [x] ModalSystem enum with 6 axiom systems (K, T, S4, S5, D, B)
- [x] Axiom verification functions (axiom_k, axiom_t, axiom_4, axiom_5, axiom_d, axiom_b)
- [x] Modal transformation functions
- [x] Modal analysis utilities
- [x] 13 comprehensive tests for modal axioms

## LTL/CTL Temporal Logic Utilities - PRODUCTION READY
- [x] Formula classification system (TemporalClass enum)
- [x] Temporal pattern recognition (TemporalPattern enum)
- [x] Temporal complexity analysis (TemporalComplexity)
- [x] Safety-liveness decomposition
- [x] Advanced LTL equivalences
- [x] Model checking utilities
- [x] 16 comprehensive tests for LTL/CTL utilities

## Probabilistic Reasoning with Bounds Propagation - PRODUCTION READY
- [x] ProbabilityInterval for imprecise probabilities
- [x] Frechet bounds for interval arithmetic
- [x] Interval operations
- [x] Credal sets for convex probability distributions
- [x] Probability propagation through logical expressions
- [x] Markov Logic Network (MLN) semantics
- [x] Probabilistic semantics extraction
- [x] 17 comprehensive tests for probabilistic reasoning

## Defuzzification Methods for Fuzzy Logic - PRODUCTION READY
- [x] DefuzzificationMethod enum with 6 methods
  - [x] Centroid (Center of Area/Gravity)
  - [x] Bisector of Area
  - [x] Mean of Maximum (MOM)
  - [x] Smallest of Maximum (SOM)
  - [x] Largest of Maximum (LOM)
  - [x] Weighted Average (for singleton sets)
- [x] FuzzySet representation
- [x] Core defuzzification algorithms
- [x] SingletonFuzzySet for discrete inputs
- [x] Area computation with trapezoidal rule
- [x] 14 comprehensive tests for defuzzification

## Effect System - PRODUCTION READY (v0.1.0)
- [x] Effect types (Computational, Memory, Probabilistic, Differentiable, etc.)
- [x] EffectSet for tracking multiple effects
- [x] Effect combination (union, intersection, subset checking)
- [x] Effect compatibility and conflict detection
- [x] Effect polymorphism with EffectVar and EffectScheme
- [x] Effect substitution and evaluation
- [x] Effect annotations for expressions
- [x] Effect inference for common operations
- [x] 19 comprehensive tests for effect system
- [x] Complete example (08_effect_system.rs)

## Parametric Types System - PRODUCTION READY (v0.1.0)
- [x] Kind system for type constructors (*, * -> *, * -> * -> *)
- [x] Type constructors (List, Option, Tuple, Function, Array, Set, Map, Custom)
- [x] Parametric types with type variables and type application
- [x] Type unification using Robinson's algorithm
- [x] Type substitution and composition
- [x] Occurs check for infinite type detection
- [x] Generalization and instantiation
- [x] Integration with TypeAnnotation system
- [x] Parametric PredicateSignature support
- [x] 27 comprehensive tests for parametric types module
- [x] 7 integration tests with PredicateSignature
- [x] Complete example (07_parametric_types.rs)

## Advanced Types - ALL COMPLETE (v0.1.0)
- [x] Parametric types (List<T>)
- [x] Effect system
- [x] Dependent types (Vec<n, T> where n is runtime)
- [x] Linear types (multiplicity tracking, resource capabilities)
- [x] Refinement types (logical predicates, liquid type inference)

## Advanced Operators - ALL COMPLETE (v0.1.0)
- [x] Probabilistic operators with bounds propagation
- [x] Fuzzy logic operators with defuzzification
- [x] Extended temporal logic (LTL/CTL properties, classification, model checking utilities)
- [x] Modal logic axiom systems (K, T, S4, S5, D, B with verification)

## Optimization - ALL COMPLETE (v0.1.0)
- [x] Distributive law transformations
- [x] Cost model annotations
- [x] Automatic optimization pass ordering
- [x] Automatic strategy selection
- [x] Advanced algebraic rewriting with term rewriting systems
- [x] Profile-guided optimization (PGO) based on runtime metrics

## Testing & Quality - COMPLETE (v0.1.0)
- [x] Fuzzing with property-based testing (fuzzing.rs module with 7 comprehensive tests)

## 0.1.0 Release - New Features (v0.1.0, 2026-03-06)

### Advanced Type Systems

**Dependent Types** (dependent.rs) - 864 lines, fully tested
- Value-dependent types (Vec<n, T> where n is runtime)
- Index expressions with arithmetic
- Dimension constraints and relationships
- Dependent function types
- Well-formedness checking
- Examples: 09_dependent_types.rs

**Linear Types** (linear.rs) - 760 lines, fully tested
- Multiplicity system (Linear, Affine, Relevant, Unrestricted)
- Usage tracking and linearity violations
- Resource capabilities (Read, Write, Execute, Own)
- Context merging and splitting
- Examples: 10_linear_types.rs

**Refinement Types** (refinement.rs) - 473 lines, fully tested
- Logical predicates on types
- Built-in refinements (positive_int, nat, probability, non_empty_vec)
- Refinement context and assumptions
- Type strengthening/weakening
- Liquid type inference
- Examples: 11_refinement_types.rs

### Profile-Guided Optimization

**PGO Module** (graph/pgo.rs) - 683 lines, fully tested
- Execution profiling with runtime metrics
- Node and tensor usage statistics
- Performance scoring and hot node identification
- Memory-intensive operation detection
- Optimization hints (fusion, caching, pre-allocation, parallelization)
- Profile merging and JSON serialization
- Examples: 12_profile_guided_optimization.rs

### Automated Theorem Proving

**Unification** (unification.rs) - 826 lines, fully tested
- Robinson's unification algorithm for first-order terms
- Most general unifier (MGU) computation
- Occur-check for infinite structure prevention
- Substitution composition and application
- Anti-unification (least general generalization)
- Variable renaming for quantifier rules
- 26 comprehensive tests covering all unification scenarios

**Resolution-Based Proving** (resolution.rs) - 1,709 lines, fully tested
- Robinson's resolution principle for refutation-based proving
- Literal and clause representation
- Multiple resolution strategies (Saturation, Set-of-Support, Linear, Unit)
- Subsumption checking for clause simplification
- Tautology detection and removal
- Proof reconstruction from resolution derivations
- Comprehensive statistics tracking
- 44 comprehensive tests including strategy comparisons
- Examples: 16_resolution_theorem_proving.rs

**Sequent Calculus** (sequent.rs) - 932 lines, fully tested
- Gentzen's sequent calculus (LK system)
- Structural rules (Identity, Weakening, Contraction, Exchange, Cut)
- Logical rules for connectives (AND, OR, NOT, IMPLY)
- Quantifier rules (EXISTS, FORALL) with proper capture-avoiding substitution
- Proof tree construction and validation
- Automated proof search with multiple strategies (DFS, BFS, Iterative Deepening)
- Cut elimination for proof normalization
- Free variable analysis in sequents
- 23 comprehensive tests covering all inference rules
- Examples: 13_sequent_calculus.rs

**Constraint Logic Programming** (clp.rs) - ~1,000 lines, fully tested
- Constraint satisfaction problems (CSP)
- Domain constraint representation
- Arc consistency (AC-3 algorithm)
- Path consistency checking
- Backtracking search with forward checking
- Constraint propagation
- Examples: 14_constraint_logic_programming.rs

### Advanced Graph Analysis

**Advanced Algorithms** (graph/advanced_algorithms.rs) - enhanced
- Strongly connected components (Tarjan's algorithm)
- Topological sorting for DAG analysis
- Cycle detection and enumeration
- Critical path analysis for optimization scheduling
- Graph diameter computation
- All-paths enumeration between nodes
- Graph isomorphism detection
- Examples: 15_advanced_graph_algorithms.rs

### Status (v0.1.0, 2026-04-06)
- All modules compile without warnings
- **806 tests** passing (806 passing, 1 skipped)
- **7 new examples** added (13-16 for theorem proving, plus enhancements)
- **4 major new modules**: unification, resolution, sequent, enhanced CLP
- **S-expression serialization** module (expr_serialize): to_sexpr/from_sexpr/to_binary/from_binary, expr_fingerprint, BatchStats
- Full API documentation with examples
- Integrated into lib.rs exports
- Zero compiler/clippy warnings
- **New benchmarks**: 50+ benchmarks including theorem proving (11 benchmark groups)
- **Integration tests**: 17 comprehensive cross-module integration tests

---

**Total Items:** 76 tasks
**Completion:** 100% (76/76) - FULLY COMPLETE
**Production Ready Features:**
- Type System (TypeAnnotation, PredicateSignature, SignatureRegistry, Parametric Types, Effect System)
- Graph Optimization (Dead code elimination, CSE, simplification)
- Metadata Support (SourceLocation, Provenance, custom attributes)
- Expression Extensions (Arithmetic, Comparison, Conditional, Constants)
- Domain Constraints (DomainInfo, DomainRegistry, validation)
- Serialization (Versioned JSON/binary, metadata support)
- Utilities (pretty_print_expr, pretty_print_graph, ExprStats, GraphStats, diff tools)
- Documentation (Comprehensive README with examples)
- Normal Forms (NNF, CNF, DNF transformations & validation)
- Graph Canonicalization (canonical ordering, hashing, equivalence)
- Modal Logic (Box/Diamond operators with full integration)
- Temporal Logic (Next/Eventually/Always/Until operators)
- Advanced Algebraic Simplification (comprehensive logical laws, modal/temporal simplifications)
- Parametric Types (Kind system, type constructors, unification, generalization)
- Effect System (Effect tracking, polymorphism, inference, annotations)
- Dependent Types (value-dependent, dimension constraints)
- Linear Types (multiplicity, resource capabilities)
- Refinement Types (logical predicates, liquid type inference)
- Automated Theorem Proving (unification, resolution, sequent calculus, CLP)
- Advanced Graph Analysis (SCC, critical paths, cycle enumeration, isomorphism)
- Profile-Guided Optimization (runtime profiling, hints, merging)
**Infrastructure Ready:**
- Aggregation operations (temporarily disabled pending compiler integration)
- Graph Transformation (Visitor patterns, subgraph extraction, merging - module disabled)
**Test Coverage:** 806 tests total (806 passing, 1 skipped)
  - 762 unit tests (including comprehensive theorem proving and S-expression serialization tests)
  - 44 property tests (43 passing, 1 ignored)

## v0.1.14 (2026-04-06)

- [x] **to_sexpr** (`expr_serialize/`): Convert any `TLExpr` to a human-readable S-expression string covering all variants including modal, temporal, fuzzy, set, and counting operators
- [x] **from_sexpr** (`expr_serialize/`): Parse S-expression strings back into `TLExpr` with full round-trip fidelity for all supported expression types
- [x] **to_binary** (`expr_serialize/`): Compact tag-based binary encoding of `TLExpr` trees for efficient storage and network transfer
- [x] **from_binary** (`expr_serialize/`): Deserialize binary-encoded expressions back into `TLExpr` with validation
- [x] **graph_to_binary** (`expr_serialize/`): Serialize a full `EinsumGraph` (nodes, edges, metadata) into a single binary blob
- [x] **expr_fingerprint** (`expr_serialize/`): SHA-256 structural hash of an expression tree for deduplication and caching
- [x] **Batch serialization** (`expr_serialize/`): `BatchStats` computing sexpr/binary byte sizes, compression ratio, node count, and max depth for expression batches

## v0.2.0 / Future Work

- Advanced type refinement / dependent-type annotations.
- DSL macro sugar for common rule patterns.
- Graph diffing for incremental re-compilation.
- First-class soft constraints (weighted rules).
- [x] ~~Split `src/resolution.rs` (1,712 L) into a `resolution/` directory.~~ (completed 2026-04-15)
