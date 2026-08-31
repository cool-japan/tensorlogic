# TensorLogic Compiler — TODO

**Status**: Stable | **Version**: 0.1.2 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-30
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

Compiler that lowers logical expressions into optimized einsum graphs.

## Completed

### Core Compilation
- [x] Basic predicate compilation to einsum specs
- [x] AND operation with same-axes operands
- [x] OR operation support
- [x] NOT operation support
- [x] EXISTS quantifier compilation (reduction)
- [x] FORALL quantifier compilation (via double negation)
- [x] Implication (->) compilation using ReLU(b - a)
- [x] Score wrapper support
- [x] CompilerContext for domain and variable tracking
- [x] Axis assignment for variables
- [x] Free variable inference
- [x] Arity validation
- [x] Basic test coverage

### AND Operation with Shared Variables - COMPLETE
- [x] Implemented union of axes for output
- [x] Support for variable contraction in einsum
- [x] Test all edge cases (disjoint, overlapping, identical variables)

### Variable Scope Analysis - PRODUCTION READY
- [x] Detect unbound variables
- [x] ScopeAnalysisResult with type conflict tracking
- [x] validate_scopes() for compilation safety
- [x] suggest_quantifiers() for helpful error messages
- [x] Track bound vs free variables
- [x] Nested quantifier support
- [x] Type annotation consistency checking

### Type Safety - PRODUCTION READY
- [x] Domain type checking for predicates
  - [x] TypeChecker with signature registry integration
  - [x] Arity validation against signatures
  - [x] Type inference from predicate applications
  - [x] Type conflict detection across expressions
- [x] Arity consistency enforcement
  - [x] Enhanced arity validation across complex expressions
  - [x] Error messages with predicate signature hints
- [x] Type inference
  - [x] infer_types() with signature registry
  - [x] Automatic variable type inference
  - [x] Type consistency validation

### Optimization - PRODUCTION READY
- [x] Common subexpression elimination (CSE)
  - [x] Expression-level CSE with caching
  - [x] Recursive subexpression detection
  - [x] CseResult with elimination statistics
- [x] Integration with IR graph optimizations
  - [x] DCE, CSE, identity simplification available
  - [x] Multi-pass optimization pipeline

### Integration - PRODUCTION READY
- [x] SymbolTable Integration
  - [x] sync_context_with_symbol_table()
  - [x] build_signature_registry()
  - [x] Bidirectional domain import/export
  - [x] PredicateInfo <-> PredicateSignature conversion

### Enhanced Diagnostics - PRODUCTION READY
- [x] Rich error messages with source locations
  - [x] Diagnostic struct with levels (Error/Warning/Info/Hint)
  - [x] enhance_error() for IrError enrichment
  - [x] Help text and related information
- [x] diagnose_expression() for validation
  - [x] Unbound variable detection with suggestions
  - [x] Unused binding warnings
  - [x] Type conflict reporting
- [x] DiagnosticBuilder for error aggregation

### Expression Compilation - PRODUCTION READY
- [x] Arithmetic operations
  - [x] Add, Subtract, Multiply, Divide
  - [x] Element-wise tensor operations
  - [x] Axis preservation
- [x] Comparison operations
  - [x] Equal, LessThan, GreaterThan, LessThanOrEqual, GreaterThanOrEqual
  - [x] Boolean result tensors
- [x] Conditional expressions
  - [x] If-then-else compilation
  - [x] Soft probabilistic semantics: cond * then + (1-cond) * else
- [x] Numeric constants
  - [x] Constant compilation to scalar tensors
- [x] Updated all compiler passes
  - [x] scope_analysis handles new expression types
  - [x] type_checking handles new expression types
  - [x] cse handles new expression types
  - [x] diagnostics handles new expression types

### Compiler Correctness - COMPLETE
- [x] Fix implication with different free variables
  - [x] Support implicit universal quantification
  - [x] OR align axes through broadcasting/projection
  - [x] Implement explicit axis alignment strategy
  - [x] Handle premise with extra axes (marginalize via sum reduction)
  - [x] Handle conclusion with extra axes (broadcast premise to match)
  - [x] Symmetric broadcasting for both operands

### Advanced Optimizations - COMPLETE
- [x] Einsum simplification module (einsum_opt.rs)
  - [x] Merge consecutive einsum operations
  - [x] Eliminate identity operations (e.g., "ab->ab")
  - [x] Optimize contraction order for multi-input einsums
  - [x] EinsumOptResult with statistics tracking
  - [x] Graph-level optimization pipeline
  - [x] 10 comprehensive unit tests

### Transitivity Rules - COMPLETE
- [x] Proper transitivity rule compilation
  - [x] Handle: forall x,y,z. knows(x,y) AND knows(y,z) -> knows(x,z)
  - [x] Broadcasting ensures premise axes align with conclusion axes
  - [x] Comprehensive test coverage for transitivity patterns
  - [x] Fixed OR axis ordering for consistent broadcasting

### Parameterized Compilation - COMPLETE
- [x] Configuration module (config.rs)
  - [x] AndStrategy: Product, Min, ProbabilisticSum, Godel, ProductTNorm, Lukasiewicz
  - [x] OrStrategy: Max, ProbabilisticSum, Godel, ProbabilisticSNorm, Lukasiewicz
  - [x] NotStrategy: Complement, Sigmoid (with temperature)
  - [x] ExistsStrategy: Sum, Max, LogSumExp, Mean
  - [x] ForallStrategy: DualOfExists, Product, Min, MeanThreshold
  - [x] ImplicationStrategy: ReLU, Material, Godel, Lukasiewicz, Reichenbach
- [x] Preset configurations
  - [x] soft_differentiable (default - neural network training)
  - [x] hard_boolean (discrete reasoning)
  - [x] fuzzy_godel (Godel fuzzy logic)
  - [x] fuzzy_product (Product fuzzy logic)
  - [x] fuzzy_lukasiewicz (Lukasiewicz fuzzy logic)
  - [x] probabilistic (probabilistic interpretation)
- [x] CompilationConfigBuilder for custom configurations
- [x] 7 comprehensive tests for all config presets

## High Priority - DONE

## Medium Priority - DONE

### Advanced Features
- [x] Negation optimization - COMPLETE
  - [x] Optimize double negations (NOT(NOT(x)) -> x)
  - [x] Propagate negations through De Morgan's laws
    - [x] NOT(AND(x, y)) -> OR(NOT(x), NOT(y))
    - [x] NOT(OR(x, y)) -> AND(NOT(x), NOT(y))
  - [x] Push negations through quantifiers
    - [x] NOT(EXISTS x. P(x)) -> FORALL x. NOT(P(x))
    - [x] NOT(FORALL x. P(x)) -> EXISTS x. NOT(P(x))
  - [x] Statistics tracking (NegationOptStats)
  - [x] 8 comprehensive tests covering all optimization patterns
- [x] Quantifier optimization - COMPLETE
  - [x] Configurable quantifier strategies via CompilationConfig
  - [x] Automatic strategy selection based on context (strategy_selection.rs)
- [x] Mixed operation types - COMPLETE
  - [x] Arithmetic operations (Add, Subtract, Multiply, Divide)
  - [x] Comparison operations (Equal, LessThan, etc.)
  - [x] Conditional expressions (if-then-else)
  - [x] Runtime operation mapping registration (custom_ops.rs)
- [x] Parameterized compilation - COMPLETE
  - [x] Configurable AND mapping (6 strategies)
  - [x] Configurable OR mapping (5 strategies)
  - [x] Configurable NOT mapping (2 strategies)
  - [x] Configurable quantifier mappings (8 strategies total)
  - [x] Configurable implication mapping (5 strategies)

### Integration with Adapters - COMPLETE
- [x] Use SymbolTable from tensorlogic-adapters
  - [x] Replace internal DomainInfo with adapter's DomainInfo
  - [x] Query predicate signatures from SymbolTable (symbol_integration.rs)
  - [x] Validate against schema (type_checking.rs, validation.rs)
- [x] Metadata propagation
  - [x] Preserve domain names in compiled graph (tensor_metadata HashMap in EinsumGraph)
  - [x] Track predicate origins (metadata field in EinsumNode)
  - [x] Enable debuggability (MetadataBuilder, propagate_metadata, attach_expr_metadata)
  - [x] Comprehensive test suite (12 tests in metadata_propagation module)

### Error Handling
- [x] Improved error messages - ENHANCED
  - [x] Suggest fixes for common errors (enhance_error function)
  - [x] Pretty-print complex expressions in errors
    - [x] Unicode symbols for logic operators
    - [x] Safe UTF-8 truncation for long expressions
    - [x] Support for all expression types
  - [x] Detailed error creation with context (create_detailed_error)
  - [x] 6 new tests for pretty-printing functionality
  - [ ] Show source location in TLExpr (requires TLExpr metadata extension)
- [x] Error recovery - EXPRESSION-LEVEL (tolerant compilation mode)
  - [x] DiagnosticBuilder collects multiple errors
  - [x] Continue validation after non-fatal warnings
  - [x] Continue compilation after non-fatal errors (via `error_recovery::TolerantCompiler`)
- [x] Validation passes - ENHANCED
  - [x] Pre-compilation validation (validate_expression function)
    - [x] Arity validation
    - [x] Scope analysis integration
    - [x] Enhanced diagnostics integration
    - [x] ValidationResult type with error/warning counts
    - [x] Type checking with predicate signatures (validate_expression_with_types)
    - [x] 7 comprehensive tests
  - [x] Post-compilation graph validation
    - [x] post_compilation_passes function with configurable options
    - [x] Axis consistency validation
    - [x] Shape compatibility checks
    - [x] Cycle detection
    - [x] Integration with IR graph optimization passes
    - [x] PostCompilationOptions for fine-grained control
    - [x] 6 comprehensive tests

## Low Priority - DONE

### Documentation
- [x] Add README.md with usage examples
- [x] Document compilation strategy
  - [x] Explain logic-to-tensor mapping (with default strategy table)
  - [x] Show einsum spec generation rules
  - [x] Provide optimization guidelines
  - [x] Parameterized compilation (26+ strategies, 6 presets)
  - [x] Architecture diagram with all compilation phases
  - [x] Scope analysis & type checking examples
  - [x] Testing & quality metrics
- [x] API documentation
  - [x] Add rustdoc for all public functions
    - [x] Module-level documentation with overview and examples
    - [x] CompilerContext with detailed method documentation
    - [x] DomainInfo struct documentation
    - [x] Validation functions with comprehensive examples
    - [x] 18 passing doc tests
  - [x] Include code examples in docs
  - [x] Document CompilerContext lifecycle
- [x] Tutorial
  - [x] Step-by-step compilation walkthrough (TUTORIAL.md)
  - [x] Common patterns and idioms (10 patterns documented)
  - [x] Debugging guide (validation, tracing, troubleshooting)
  - [x] Advanced features (strategy selection, custom operations)
  - [x] Best practices section with 6 guidelines

### Testing
- [x] Property-based testing
  - [x] Use proptest for random TLExpr generation (21 property tests passing)
  - [x] Verify compilation invariants (17 core + 4 strategy-specific)
  - [x] Check graph validity
- [x] Fuzzing
  - [x] Fuzz complex nested expressions (fuzz_compile_expression)
  - [x] Stress-test axis assignment (fuzz_type_checking)
  - [x] Find edge cases in quantifiers (fuzz_quantifiers)
  - [x] Fuzz optimization passes (fuzz_optimizations)
  - [x] Complete README with usage instructions
  - [x] 4 comprehensive fuzz targets
- [x] Benchmark suite
  - [x] Measure compilation time (compilation_performance.rs)
  - [x] Track graph size vs expression complexity
  - [x] Compare optimization passes
- [x] **Reference Comparison Tests** (tests/reference_comparisons.rs) ✅ NEW (2026-03-29)
  - [x] AND/OR/NOT/IMPLICATION vs analytical formulas (all 6 strategies)
  - [x] De Morgan's Laws numerical verification
  - [x] Distributive and Absorption Laws
  - [x] Cross-strategy consistency at Boolean inputs
  - [x] Łukasiewicz triangle inequality
  - [x] 35 tests passing

### Tooling
- [x] Visualization
  - [x] Export EinsumGraph to DOT format (tensorlogic-ir::export_to_dot)
  - [x] Visualize compilation process (with options: clustering, highlighting, layout)
  - [x] Show axis mappings graphically (via graph visualization)
  - [x] 8 comprehensive tests for DOT export
- [x] Debug utilities
  - [x] Print intermediate compilation states (CompilationTrace)
  - [x] Trace axis assignments (CompilationTracer)
  - [x] Dump context at each step (print_context_state, print_graph_state, print_graph_diff)
  - [x] 7 comprehensive tests for debug utilities
- [x] CLI tool - Moved to tensorlogic-cli crate
  - [x] Compile TLExpr from command line (tensorlogic binary)
  - [x] Output in various formats (graph, JSON, DOT, stats)
  - [x] Input formats (expr string, JSON, YAML)
  - [x] Domain definitions via CLI flags
  - [x] Strategy selection (6 presets)
  - [x] Graph validation
  - [x] Debug mode with detailed output
  - [x] Enhanced features: REPL, batch processing, watch mode, shell completion

## Advanced Logic - ALL COMPLETE (v0.1.0)

### Counting Quantifiers
- [x] CountingExists (exists>=k x. P(x)) - at least k elements satisfy P
- [x] CountingForAll (forall>=k x. P(x)) - at least k elements satisfy P
- [x] ExactCount (exists=k x. P(x)) - exactly k elements satisfy P
- [x] Majority (Majority x. P(x)) - more than half satisfy P
- [x] Implementations using sum reductions and soft thresholding
- [x] 4 comprehensive unit tests (all passing)
- [x] Integration with compiler dispatcher in compile/mod.rs

### Match Exhaustiveness
- [x] Added wildcard patterns to 20+ files for new TLExpr variants
- [x] Optimize directory: 11 files
- [x] Passes directory: 5 files
- [x] Updated symbol_table.rs, scope_analysis.rs, type_checking.rs
- [x] Zero compilation errors, all tests passing

### Higher-Order Logic
- [x] Lambda expressions with type annotations (compile_lambda)
- [x] Apply with beta reduction (compile_apply)
- [x] Compile-time substitution for immediate applications
- [x] Non-lambda application support (predicate application)
- [x] 8 comprehensive unit tests (all passing)
- [x] Supports nested lambda applications

### Set Theory Operations
- [x] SetMembership (elem in set) - element-wise product
- [x] SetUnion (A union B) - element-wise max
- [x] SetIntersection (A intersect B) - element-wise min
- [x] SetDifference (A \ B) - masked multiplication
- [x] SetCardinality (|S|) - sum reduction
- [x] EmptySet - constant zero tensor
- [x] SetComprehension ({ var : domain | condition }) - predicate as characteristic function
- [x] 8 comprehensive unit tests (all passing)
- [x] Example demonstrating usage (19_set_operations.rs)
- [x] Sets represented as characteristic functions (indicator tensors)

### Fixed-Point Operators
- [x] LeastFixpoint (muX.phi(X)) - starts from empty set, iterates upward
- [x] GreatestFixpoint (nuX.phi(X)) - starts from universal set, iterates downward
- [x] Unrolling strategy with configurable depth (default: 5 iterations)
- [x] Domain inference from quantifiers in body
- [x] 8 comprehensive unit tests (all passing)
- [x] Applications: transitive closure, reachability, safety properties

### Hybrid Logic
- [x] Nominal (@i) - one-hot vector over state space
- [x] At operator (@i phi) - evaluates formula at specific state
- [x] Somewhere (E phi) - existential over reachable states (max reduction)
- [x] Everywhere (A phi) - universal over reachable states (min reduction)
- [x] State space representation with default size (10 states)
- [x] Full connectivity assumption for reachability
- [x] 10 comprehensive unit tests (all passing)

### Constraint Programming
- [x] AllDifferent - ensures all variables have distinct values
- [x] GlobalCardinality - bounds occurrences of values
- [x] AllDifferent compiles to: product_{i<j} (xi != xj) as pairwise inequalities
- [x] GlobalCardinality compiles to: count bounds with aggregations
- [x] 9 comprehensive unit tests (all passing)
- [x] Example demonstrating usage (20_constraint_programming.rs)
- [x] Applications: N-Queens, Sudoku, Graph Coloring, Scheduling, Resource Allocation

### Abductive Reasoning
- [x] Abducible(name, cost) - hypothesis literals with associated costs
- [x] Explain(formula) - marks formulas for explanation
- [x] Soft optimization objective: satisfaction - lambda * total_cost
- [x] Cost minimization through gradient descent (backend responsibility)
- [x] Multiple abducibles support with cost aggregation
- [x] 11 comprehensive unit tests (all passing)
- [x] Applications: medical diagnosis, robot planning, fault detection

### Modal Logic
- [x] Box - necessity operator with min/product reduction over worlds
- [x] Diamond - possibility operator with max/sum reduction over worlds
- [x] ModalStrategy configuration (AllWorldsMin, AllWorldsProduct, Threshold)
- [x] Automatic world axis management
- [x] Integration with all 6 compilation presets
- [x] 9 comprehensive tests

### Temporal Logic
- [x] Eventually (F) - temporal eventually with max/sum reduction over time
- [x] Always (G) - temporal always with min/product reduction over time
- [x] TemporalStrategy configuration (Max, Sum, LogSumExp)
- [x] Automatic time axis management
- [x] Next (X) - implemented via `temporal_next:<axis>` unary node; backend uses `shift_next` (shift forward one step, zero-pad last slice); full VJP via `shift_prev` in autodiff backward pass
- [x] Until (U) - implemented via `temporal_until:<tag>:<axis>` binary node; backend uses backward-scan `until_scan`; two semantics: MaxMin (crisp) and ProbSumProduct (smooth, differentiable); full adjoint VJP via `until_scan_vjp`
- [x] Advanced operators (Release, WeakUntil, StrongRelease) — exact finite-trace recurrences via unified `temporal_binary_scan`; see `temporal_ops.rs`
- [x] 9 comprehensive tests

### Probabilistic Logic
- [x] WeightedRule for soft constraints (multiply rule by confidence weight)
- [x] ProbabilisticChoice for stochastic selection (weighted sum of alternatives)
- [x] SoftExists with temperature-controlled log-sum-exp
- [x] SoftForAll as dual of SoftExists
- [x] 5 comprehensive tests

### Fuzzy Logic
- [x] TNorm operators (Minimum, Product, Lukasiewicz, Drastic, Nilpotent, Hamacher)
- [x] TCoNorm operators (Maximum, Probabilistic, Bounded, Drastic, Nilpotent, Hamacher)
- [x] FuzzyNot operators (Standard, Yager, Sugeno)
- [x] FuzzyImplication operators (Kleene-Dienes, Godel, Reichenbach, Lukasiewicz, Goguen, Rescher)
- [x] 6 comprehensive tests (all passing)

## Performance - ALL COMPLETE (v0.1.0)

### Multi-threaded Compilation
- [x] ParallelCompiler with configurable parallelization strategy
- [x] Complexity-based scheduling (min_complexity_for_parallel threshold)
- [x] Thread pool configuration (max_threads setting)
- [x] Parallel optimization passes support
- [x] Comprehensive statistics tracking (ParallelStats)
- [x] 9 comprehensive tests (all passing)
- [x] Example demonstrating usage (14_parallel_compilation.rs)
- [x] Feature flag: parallel (optional dependency on rayon + parking_lot)

### Incremental Compilation
- [x] Expression dependency tracking
- [x] Change detection and invalidation strategies
- [x] IncrementalCompiler with minimal recompilation
- [x] Automatic invalidation on context changes
- [x] 6 comprehensive tests
- [x] Example demonstrating usage (09_incremental_compilation.rs)

### Compilation Caching
- [x] Thread-safe cache with LRU eviction
- [x] Automatic cache key generation
- [x] Cache statistics (hits, misses, evictions, hit rate)
- [x] Configurable cache size
- [x] 6 comprehensive tests

### JIT Compilation
- [x] JIT compilation for hot paths — call-count-based promotion with pre-optimized graph caching

## Interoperability - ALL COMPLETE (v0.1.0)

### Export to ONNX
- [x] OnnxExportConfig with DataType support (Float32, Float64, Int32, Int64, Bool)
- [x] Protobuf message structures for ONNX format
- [x] OnnxConverter translating EinsumGraph operations to ONNX
- [x] Support for Einsum, ElemUnary, ElemBinary, and Reduce operations
- [x] export_to_onnx() and export_to_onnx_with_config() API functions
- [x] 8 comprehensive unit tests (all passing)
- [x] Example demonstrating usage (15_onnx_export.rs)
- [x] Feature flag: onnx (optional dependency on prost + prost-types)

### Export to TensorFlow GraphDef
- [x] TensorFlowExportConfig with TfDataType support (Float32, Float64, Int32, Int64, Bool)
- [x] Protobuf message structures for TensorFlow GraphDef format
- [x] TensorFlowConverter translating EinsumGraph operations to TensorFlow ops
- [x] Support for Einsum, ElemUnary, ElemBinary, and Reduce operations
- [x] Special handling for one_minus operation (1 - x)
- [x] export_to_tensorflow() and export_to_tensorflow_with_config() API functions
- [x] 10 comprehensive unit tests (all passing)
- [x] Example demonstrating usage (16_tensorflow_export.rs)
- [x] Feature flag: tensorflow (optional dependency on prost + prost-types)

### Export to PyTorch Code Generation
- [x] PyTorchExportConfig with PyTorchDtype support (Float32, Float64, Int32, Int64, Bool)
- [x] Python code generator producing PyTorch nn.Module classes
- [x] Support for all operation types (Einsum, ElemUnary, ElemBinary, Reduce)
- [x] Proper input tensor detection and dictionary lookup generation
- [x] TorchScript decorator support (@torch.jit.export)
- [x] Configurable indentation and class naming
- [x] export_to_pytorch() and export_to_pytorch_with_config() API functions
- [x] 11 comprehensive unit tests (all passing)
- [x] Example demonstrating usage (17_pytorch_export.rs)
- [x] Feature flag: pytorch (no additional dependencies)

### Import from Logic Frameworks
- [x] Prolog syntax parser (import/prolog.rs)
  - Facts, rules (:-), conjunctions (,), disjunctions (;)
  - Negation (\+ and not(...) syntax)
  - Variables (uppercase) and constants (lowercase/numeric)
  - Multi-argument predicates
- [x] S-Expression parser (import/sexpr.rs)
  - Nested logical expressions with proper tokenization
  - Operators: and, or, not, =>, exists, forall
  - Quantifier support with domain specification
  - Multi-operand chains (and P Q R)
- [x] TPTP format parser (import/tptp.rs)
  - FOF (First-Order Formula) and CNF support
  - Quantifiers: ![X]: (forall), ?[X]: (exists)
  - Operators: & (and), | (or), ~ (not), => (imply)
  - Multiple variable quantification: ![X, Y]:
- [x] Auto-detection (import/mod.rs)
  - Automatic format detection based on syntax
  - parse_auto() function with pattern matching
- [x] 34 comprehensive unit tests (all passing)
- [x] Example demonstrating usage (18_logic_import.rs)

## Not Yet Implemented

- [ ] First-class functions/predicates
- [ ] Higher-order quantification
- [x] Next (X) temporal operator — see `temporal_ops.rs`; `shift_next`/`shift_prev` primitives
- [x] Until (U) temporal operator — see `temporal_ops.rs`; `until_scan`/`until_scan_vjp` primitives; MaxMin and ProbSumProduct semantics; tag selected by `TemporalStrategy`
- [x] Advanced temporal operators (Release, WeakUntil, StrongRelease) — exact finite-trace recurrences, VJPs; see temporal_ops.rs
- [x] JIT compilation for hot paths — call-count-based promotion with pre-optimized graph caching
- [ ] Source location tracking in TLExpr (requires TLExpr metadata extension)
- [x] Continue compilation after non-fatal errors (partial error recovery) — completed 2026-04-15 via `error_recovery/` module

## v0.1.4 Enhancements (2026-03-30)

- [x] **Compilation Cache** (`cache.rs`): `LruCompilationCache` (LRU eviction via VecDeque, configurable capacity), `ExprFingerprint` (Debug-derived structural hash), `CachingCompiler` wrapper with `compile()`, `compile_batch()`, `invalidate()`. `CacheStats` with `hit_rate()`. 20 new tests.

## v0.1.5 Enhancements (2026-03-30)

- [x] **Expression Canonicalization** (`optimize/canonical.rs`): `Canonicalizer` (double-neg elimination, nested AND/OR flattening, commutative sorting via `canonical_order_key()`), `CanonicalStats` tracking. Handles all 60+ TLExpr variants recursively. 20 new tests.

## v0.1.7 Enhancements (2026-03-30)

- [x] **Compilation Profiling** (`profiling.rs`): `CompilationProfiler` with `begin_phase`/`end_phase`/`finish`, `ProfileReport` (slowest/fastest phase, percentages, compilation speed), `ProfileEntry` with throughput, `profile_phase()` convenience function. 18 new tests.

## v0.1.10 Enhancements (2026-03-31)

- [x] **RewriteRule Trait** (`rewrite.rs`): `RewriteRule` trait with `name()`, `description()`, `applies_to()`, and `apply()` methods enabling pluggable term-rewriting rules over `TLExpr`
- [x] **RewriteEngine** (`rewrite.rs`): `RewriteEngine` with configurable rule sets and three application strategies — exhaustive (apply all matching rules), fixed-point (repeat until no change), single-pass (one traversal)
- [x] **5 Built-in Rewrite Rules** (`rewrite.rs`): `DoubleNegationRule` (NOT(NOT(x)) → x), `IdentityAndRule` (AND(x, TRUE) → x), `IdentityOrRule` (OR(x, FALSE) → x), `DeMorganAndRule` (NOT(AND(x,y)) → OR(NOT(x),NOT(y))), `DeMorganOrRule` (NOT(OR(x,y)) → AND(NOT(x),NOT(y)))
- [x] **RewriteStats** (`rewrite.rs`): `RewriteStats` tracking per-rule application counts, total rewrite passes performed, and total reductions achieved across the full expression tree

## v0.1.12

- [x] **DeadCodeEliminator** (`dead_code.rs`): `DeadCodeEliminator` struct driving multi-pass DCE over `TLExpr` trees, removing structurally unreachable sub-expressions and constant-foldable branches until a fixed point is reached or `DceConfig::max_passes` is exceeded.
- [x] **DceStats** (`dead_code.rs`): `DceStats` recording the number of eliminated nodes, passes completed, and whether the process converged, enabling callers to inspect DCE effectiveness.
- [x] **DceConfig** (`dead_code.rs`): `DceConfig` controlling maximum pass count and enabling/disabling individual fold categories (`fold_constants`, `eliminate_unused_lets`) for fine-grained DCE tuning.
- [x] **fold_and / fold_or / fold_not / fold_if** (`dead_code.rs`): constant-folding helpers propagating literal `TRUE`/`FALSE` through `AND`, `OR`, `NOT`, and `If` expressions (e.g. `AND(FALSE, x) → FALSE`, `IF(TRUE, t, _) → t`).
- [x] **Unused Let elimination** (`dead_code.rs`): detects `Let` bindings whose variable does not appear free in the body and removes them, preventing dead-binding accumulation in generated IR.

## v0.1.13

- [x] **ExprComplexity** (`complexity.rs`): Recursive structural complexity scoring traversing all 55+ `TLExpr` variants, computing depth, node count, quantifier nesting, and weighted complexity score with `is_simple()` threshold check
- [x] **ComplexityThresholds** (`complexity.rs`): Configurable warning/error/critical complexity limits with `check()` returning the highest triggered severity level
- [x] **ComplexityWarning** (`complexity.rs`): Severity-tagged warnings (`Warning`/`Error`/`Critical`) with human-readable messages and threshold context for compiler diagnostics
- [x] **ComplexityComparison** (`complexity.rs`): Diff two expressions' complexity metrics producing delta scores, depth changes, and node-count differences with `is_more_complex()` / `is_simpler()` queries
- [x] **BatchComplexityStats** (`complexity.rs`): Aggregate complexity statistics over expression batches — min/max/mean/median complexity, count above threshold, distribution summary

## v0.1.16

- [x] **LetInliner** (`inline.rs`): Capture-avoiding substitution pass that inlines `Let`-bound variables into their body expressions, driven by `InlineConfig` and producing `InlineStats`
- [x] **InlineConfig** (`inline.rs`): Builder-style configuration controlling `max_depth` for recursive inlining, `inline_once` (only inline single-use bindings), `always_inline` (force inline regardless of use count), and `size_threshold` (skip inlining when the bound expression exceeds the node-count limit)
- [x] **InlineStats** (`inline.rs`): Collects `bindings_inlined` count, total `substitutions_performed`, and a `converged` flag indicating whether the fixed-point was reached within the configured pass limit
- [x] **count_free_occurrences** (`inline.rs`): Recursive function counting the number of free occurrences of a named variable in a `TLExpr` tree, respecting `Let`/`Lambda` binding scopes to avoid counting shadowed uses
- [x] **capture-avoiding substitute** (`inline.rs`): Core alpha-renaming-based substitution function that replaces free occurrences of a variable with an expression, renaming binders to fresh names when substitution would otherwise capture a free variable in the replacement

## v0.1.15

- [x] **ConstantPropagator** (`const_prop.rs`): Bottom-up constant-folding pass over `TLExpr` trees; traverses all variants and substitutes fully-constant sub-expressions with their evaluated `Num` or boolean literal results until a fixed point is reached or `ConstPropConfig::max_passes` is exceeded
- [x] **ConstPropStats** (`const_prop.rs`): Collects per-pass and cumulative counts of folded nodes, total passes performed, and a `converged` flag indicating whether the fixed point was reached within the configured pass limit
- [x] **ConstPropConfig** (`const_prop.rs`): Builder-style configuration controlling `max_passes`, and per-category fold toggles: `fold_arithmetic` (Add/Sub/Mul/Div/Pow), `fold_comparison` (Eq/Lt/Gt/Le/Ge), `fold_boolean` (AND/OR/NOT with short-circuit)
- [x] **Arithmetic folding** (`const_prop.rs`): Folds `Add`, `Sub`, `Mul`, `Div`, `Pow` nodes whose both operands reduce to `Num` literals; division by zero produces `NaN`-literal rather than panicking
- [x] **Comparison folding** (`const_prop.rs`): Evaluates `Eq`, `Lt`, `Gt`, `Le`, `Ge` between constant `Num` operands, replacing the comparison node with `TRUE` or `FALSE` `Atom` literals
- [x] **Boolean folding** (`const_prop.rs`): Short-circuit folds `AND(FALSE, _) → FALSE`, `AND(TRUE, x) → x`, `OR(TRUE, _) → TRUE`, `OR(FALSE, x) → x`, `NOT(TRUE) → FALSE`, `NOT(FALSE) → TRUE`

## v0.1.17

- [x] **CompilerPipeline** (`pipeline.rs`): Composable pass chain driving all compilation phases end-to-end, from parsing through optimisation to code generation, with configurable pass selection via `CompilerPipelineConfig`
- [x] **CompilerPipelineConfig** (`pipeline.rs`): Builder-style configuration with feature-gate toggles for each pass category (scope analysis, type checking, DCE, CSE, constant propagation, inlining, rewriting), enabling minimal or maximal pipeline configurations
- [x] **CompilerPassOrder** (`pipeline.rs`): Enum encoding the canonical ordering of compilation passes; supports dependency-aware ordering to prevent applying downstream passes before their prerequisites
- [x] **CompilerPassStats** (`pipeline.rs`): Per-pass timing (wall-clock nanoseconds) and nodes-affected counters reported after each pass completes, enabling fine-grained profiling of individual pipeline stages
- [x] **CompilerPipelineStats** (`pipeline.rs`): Aggregate pipeline-level metrics combining per-pass `CompilerPassStats`, total compilation time, overall nodes eliminated, and a `slowest_pass()` helper
- [x] **PassBenchmark** (`pipeline.rs`): Micro-benchmark harness for individual passes — runs a pass N times, reports mean/min/max wall-clock duration, and flags regressions against a configurable baseline threshold
- [x] **`inline.rs` sub-module refactor**: `LetInliner` and helpers extracted into a focused `inline/` sub-directory; `inline/mod.rs` re-exports the public API unchanged for backward compatibility
- [x] **`augmentation.rs` sub-module refactor**: `AugmentationPipeline` and transform helpers extracted into `augmentation/` sub-directory with focused modules per transform family (noise, crop, mix, normalize)

---

**Total Items:** 111 tasks (all implemented) + future enhancements
**Completion:** 103/103 (100%) - FULLY COMPLETE as of v0.1.0 (2026-03-06)

**Production Ready Features:**
- Core Compilation: Predicates, AND, OR, NOT, quantifiers, implications
- Modal & Temporal Logic: Box, Diamond, Eventually, Always
- Type Safety: Scope analysis, type checking, arity validation
- Optimization Passes: Negation, CSE, einsum, DCE, contraction, loop fusion
- Enhanced Diagnostics: Rich error messages, pretty-printing, DiagnosticBuilder
- Expression Types: Arithmetic, Comparison, Conditional
- Advanced Features: Transitivity Rules, Parameterized Compilation (26+ strategies, 6 presets)
- Automatic Strategy Selection, Post-Compilation Validation, Runtime Custom Operations
- SymbolTable Integration, Metadata propagation
- Advanced Logic: Counting quantifiers, Higher-order logic, Set theory, Fixed-points
- Hybrid logic, Constraint programming, Abductive reasoning
- Probabilistic logic, Fuzzy logic (all 4 families)
- Import: Prolog, S-Expression, TPTP (34 tests)
- Export: ONNX, TensorFlow GraphDef, PyTorch code generation (29 tests)
- Performance: Parallel compilation, Incremental compilation, Caching
- Analysis: Profiling, Dataflow, Contraction optimization, Loop fusion, Reachability

**Test Coverage:** 862 tests (100% passing)
**Build Status:** Zero errors, zero warnings (strict clippy compliance)

## v0.1.19 (2026-04-05)

- [x] **Partial Evaluation / Specialization** (`partial_eval.rs`): `PEEnv` binding map from variable names to `PEValue` (Constant, Bool, Symbolic); `PEConfig` with toggles for arithmetic folding, boolean folding, branch pruning, let-inlining, and configurable max depth/inline threshold; `PEStats` tracking nodes visited, reduced, and inlined; `partially_evaluate()` single-pass reducer; `specialize()` convenience wrapper binding a named argument; `specialize_batch()` multi-argument specialization; handles all arithmetic/logic/fuzzy/temporal operators with short-circuit branch pruning (IfThenElse under known condition) and let-inlining at or below the inline-size threshold.

## v0.1.18 (2026-04-05)

- [x] **Symbolic Differentiation** (`symbolic_diff.rs`): `differentiate()` computes symbolic derivatives of `TLExpr` with respect to a named variable; `jacobian()` builds the full Jacobian vector for a list of output expressions; `simplify_derivative()` applies algebraic simplification rules (zero/identity elimination, constant folding); `DiffConfig` controls simplification depth and intermediate caching; `DiffResult` carries the differentiated expression plus a `simplified` flag. Supports all arithmetic operators (`Add`, `Sub`, `Mul`, `Div`, `Pow`, `Neg`), logical operators (`And`, `Or`, `Not`, `Implies`), fuzzy operators (`TNorm`, `TCoNorm`), temporal operators (`Eventually`, `Always`, `Until`), and probabilistic expressions.

## v0.1.21 (2026-04-05)

- [x] **Bytecode VM** (`bytecode.rs`): Added bytecode.rs — stack-based bytecode VM: 40-instruction set (arithmetic, comparison, boolean, fuzzy, control flow, variables), `BytecodeProgram` with forward-jump patching, `compile()`/`execute()`/`execute_with_stats()`; short-circuit `And`/`Or` via `JumpIfFalse`/`JumpIfTrue`.

## v0.1.20 (2026-04-05)

- [x] **Type Inference** (`type_infer.rs`): `TLType` enum covering `Bool`, `Numeric`, `Relation(arity)`, `Set`, `Fuzzy`, `Probabilistic`, `Var(name)`, and `Unknown`; unification-based Hindley-Milner-lite inference engine; `Substitution` map with occurs-check to prevent infinite types; `TypeEnv` for binding variable names to `TLType`; `annotate()` building fully type-annotated expression trees from bare `TLExpr` inputs; `unify()` with `UnificationError` reporting mismatched types.

## v0.2.0 / Future Work

- Incremental re-compilation (hash-based graph cache).
- Alternative backends: CUDA codegen, WASM target.
- Cost-model-driven optimization pass scheduling.
- Symbolic differentiation caching across runs.
- [x] ~~Split `src/dead_code.rs` (1,614 L), `src/partial_eval.rs` (1,789 L), and `src/symbolic_diff.rs` (1,524 L) into directory modules.~~ (completed 2026-04-15)
