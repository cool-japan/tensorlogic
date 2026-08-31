# tensorlogic-compiler

**Engine-agnostic compilation of TensorLogic expressions to tensor computation graphs**

[![Crate](https://img.shields.io/badge/crates.io-tensorlogic--compiler-orange)](https://crates.io/crates/tensorlogic-compiler)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/tensorlogic-compiler)
[![Tests](https://img.shields.io/badge/tests-920%2F920_passing-brightgreen)](#)
[![Production](https://img.shields.io/badge/status-stable-success)](#)
[![Version](https://img.shields.io/badge/version-0.1.3-blue)](#)
[![Zero Warnings](https://img.shields.io/badge/warnings-0-success)](#)

## Overview

The compiler translates logical rules with quantifiers into optimized tensor operations using Einstein summation notation. It operates as a **planning layer** only - no execution happens here.

**Input:** `TLExpr` (logical expressions with predicates, quantifiers, implications)
**Output:** `EinsumGraph` (directed graph of tensor operations)

## Key Features

### Core Compilation (Production Ready)
- **Logic-to-Tensor Mapping**: Compiles predicates, AND, OR, NOT, EXISTS, FORALL, IMPLY
- **Arithmetic Operations**: Add, Subtract, Multiply, Divide with element-wise tensor ops
- **Comparison Operations**: Equal, LessThan, GreaterThan with boolean result tensors
- **Conditional Expressions**: If-then-else with soft probabilistic semantics
- **Shared Variable Support**: Handles variable sharing in AND operations via einsum contraction
- **Automatic Axis Marginalization**: Implicitly quantifies extra variables in implications

### Modal & Temporal Logic (Production Ready)
- **Modal Operators**: Box for necessity, Diamond for possibility
- **Temporal Operators**: Eventually (F), Always (G) for temporal reasoning
- **Configurable Strategies**: 3 modal strategies, 3 temporal strategies
- **Automatic Axis Management**: World and time dimensions managed transparently
- **Combined Reasoning**: Support for nested modal/temporal expressions

### Type Safety & Validation (Production Ready)
- **Scope Analysis**: Detects unbound variables with helpful quantifier suggestions
- **Type Checking**: Validates predicate arity and type consistency across expressions
- **Domain Validation**: Ensures variables are bound to valid domains
- **Enhanced Diagnostics**: Rich error messages with source locations and fix suggestions

### Optimization Pipeline (Production Ready)

The compiler features a **7-pass optimization pipeline** that can reduce expression complexity by up to 80%:

1. **Negation Optimization**: Double negation elimination, De Morgan's laws, quantifier negation pushing
2. **Constant Folding**: Compile-time evaluation of constant expressions (2.0 * 3.0 -> 6.0)
3. **Algebraic Simplification**: Identity elimination (x+0=x, x*1=x), annihilation (x*0=0), idempotency
4. **Strength Reduction**: Replace expensive ops with cheaper equivalents (x^2->x*x, exp(log(x))->x)
5. **Distributivity**: Factor common subexpressions (a*b + a*c -> a*(b+c))
6. **Quantifier Optimization**: Loop-invariant code motion (exists x.(a+p(x)) -> a + exists x.p(x))
7. **Dead Code Elimination**: Remove unreachable branches and short-circuit constant conditions

**Additional Graph-Level Optimizations:**
- **Common Subexpression Elimination (CSE)**: Graph-level deduplication of identical operations
- **Einsum Optimization**: Operation merging, identity elimination, contraction order optimization

**Pipeline Features:**
- **Configurable**: Enable/disable individual passes, set iteration limits
- **Fixed-Point Detection**: Automatically stops when no more optimizations are possible
- **Performance Tracking**: Detailed statistics on applied optimizations
- **Hardware-Adaptive**: GPU-optimized, CPU-optimized, and SIMD-optimized cost models

### Parameterized Compilation (Production Ready)
- **26+ Configurable Strategies**: Customize logic-to-tensor mappings for different use cases
- **6 Preset Configurations**: Soft differentiable, hard Boolean, fuzzy logics, probabilistic
- **Fine-Grained Control**: Per-operation strategy selection (AND, OR, NOT, quantifiers, implication)

### Advanced Analysis & Profiling (Production Ready)
- **Compilation Profiling**: Track compilation time, memory usage, cache statistics, and pass-level performance
- **Dataflow Analysis**: Live variable analysis, reaching definitions, use-def chains for optimization
- **Graph Dataflow**: Tensor liveness tracking, dependency analysis for graph optimization
- **Contraction Optimization**: Dynamic programming for optimal einsum contraction order (reduces FLOPs)
- **Loop Fusion**: Fuse multiple loops over the same axes for better cache locality
- **Reachability Analysis**: Compute dominance, strongly connected components, topological ordering
- **Integrated Post-Compilation**: Unified pipeline combining validation and graph-level optimizations

### Advanced Logic (Production Ready)
- **Counting Quantifiers**: CountingExists, CountingForAll, ExactCount, Majority
- **Higher-Order Logic**: Lambda expressions, Apply with beta reduction
- **Set Theory Operations**: Membership, Union, Intersection, Difference, Cardinality, Comprehension
- **Fixed-Point Operators**: LeastFixpoint, GreatestFixpoint with configurable unrolling depth
- **Hybrid Logic**: Nominal (@i), At operator (@i phi), Somewhere (E phi), Everywhere (A phi)
- **Constraint Programming**: AllDifferent, GlobalCardinality
- **Abductive Reasoning**: Abducible with costs, Explain operator
- **Probabilistic Logic**: WeightedRule, ProbabilisticChoice, SoftExists, SoftForAll
- **Fuzzy Logic**: TNorm (6 variants), TCoNorm (6 variants), FuzzyNot (3 variants), FuzzyImplication (6 variants)

### Import/Export (Production Ready)
- **Import**: Prolog syntax, S-Expressions, TPTP format with auto-detection
- **Export to ONNX**: Full protobuf message generation
- **Export to TensorFlow GraphDef**: TensorFlow op translation
- **Export to PyTorch**: Human-readable Python nn.Module code generation

### Compiler Pipeline (Production Ready, v0.1.17)
- **CompilerPipeline**: Composable end-to-end pass chain from parsing through code generation
- **CompilerPassOrder**: Dependency-aware canonical ordering of compilation passes
- **CompilerPipelineConfig**: Feature-gate toggles for scope analysis, DCE, CSE, inlining, rewriting
- **CompilerPassStats / CompilerPipelineStats**: Per-pass timing and aggregate metrics
- **PassBenchmark**: Micro-benchmark harness for individual passes

### Symbolic Differentiation (Production Ready, v0.1.18)
- **differentiate()**: Symbolic derivatives of `TLExpr` w.r.t. a named variable
- **jacobian()**: Full Jacobian vector for a list of output expressions
- **simplify_derivative()**: Algebraic simplification of computed derivatives
- **DiffConfig / DiffResult**: Depth-controlled simplification with intermediate caching
- Supports arithmetic, logical, fuzzy, temporal, and probabilistic operators

### Partial Evaluation (Production Ready, v0.1.19)
- **partially_evaluate()**: Single-pass reducer with `PEEnv` binding map
- **specialize()**: Convenience wrapper for binding a single named argument
- **specialize_batch()**: Multi-argument specialization in one call
- **PEConfig**: Toggles for arithmetic folding, boolean folding, branch pruning, let-inlining
- **PEStats**: Nodes visited, reduced, and inlined counters

### Type Inference (Production Ready, v0.1.20)
- **TLType**: Enum covering Bool, Numeric, Relation(arity), Set, Fuzzy, Probabilistic, Var, Unknown
- **annotate()**: Fully type-annotated expression trees from bare `TLExpr` inputs
- **unify()**: Hindley-Milner-lite unification engine with occurs-check
- **Substitution / TypeEnv**: Variable-to-type binding maps with `UnificationError` reporting

### Bytecode VM (Production Ready, v0.1.21)
- **Stack-based VM**: 40-instruction set (arithmetic, comparison, boolean, fuzzy, control flow)
- **compile()**: Compile `TLExpr` to `BytecodeProgram` with forward-jump patching
- **execute() / execute_with_stats()**: Run programs with optional execution statistics
- **Short-circuit evaluation**: `JumpIfFalse` / `JumpIfTrue` for `And` / `Or`

### Performance Features (Production Ready)
- **Parallel Compilation**: Multi-threaded with configurable parallelization strategy
- **Incremental Compilation**: Expression dependency tracking, change detection
- **Compilation Caching**: Thread-safe LRU cache with statistics

## Quick Start

```rust
use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_ir::{TLExpr, Term};

// Define a logic rule: exists y. knows(x, y)
// "Find all persons x who know someone"
let rule = TLExpr::exists(
    "y",
    "Person",
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
);

// Compile to tensor operations
let graph = compile_to_einsum(&rule)?;

// Graph contains:
// - Tensors: ["knows[ab]", "temp_0"]
// - Operations: [Reduce{op: "sum", axes: [1]}]
// - Outputs: [1]
```

## Logic-to-Tensor Mapping

### Default Strategy (Soft Differentiable)

| Logic Operation | Tensor Equivalent | Notes |
|----------------|-------------------|-------|
| `P(x, y)` | Tensor with axes `ab` | Predicate as multi-dimensional array |
| `P AND Q` | Hadamard product or einsum | Element-wise if same axes, contraction if shared vars |
| `P OR Q` | `max(P, Q)` | Or soft variant (configurable) |
| `NOT P` | `1 - P` | Or temperature-controlled |
| `exists x. P(x)` | `sum(P, axis=x)` | Or `max` for hard quantification |
| `forall x. P(x)` | `NOT(exists x. NOT(P(x)))` | Dual of EXISTS |
| `P -> Q` | `ReLU(Q - P)` | Soft implication |

### Modal & Temporal Logic Operations

| Logic Operation | Tensor Equivalent | Notes |
|----------------|-------------------|-------|
| `Box P` (necessity) | `min(P, axis=world)` or `prod(P, axis=world)` | Necessity over possible worlds |
| `Diamond P` (possibility) | `max(P, axis=world)` or `sum(P, axis=world)` | Possibility over possible worlds |
| `F(P)` (Eventually) | `max(P, axis=time)` or `sum(P, axis=time)` | True in some future state |
| `G(P)` (Always) | `min(P, axis=time)` or `prod(P, axis=time)` | True in all future states |

**Modal Logic Example:**
```rust
use tensorlogic_ir::{TLExpr, Term};

// Box(exists y. knows(x, y)) - "In all possible worlds, x knows someone"
let expr = TLExpr::Box(Box::new(
    TLExpr::exists("y", "Person",
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")])
    )
));
```

**Temporal Logic Example:**
```rust
// F(completed(t)) - "Task t will eventually be completed"
let expr = TLExpr::Eventually(Box::new(
    TLExpr::pred("completed", vec![Term::var("t")])
));

// G(safe(s)) - "System s is always safe"
let expr = TLExpr::Always(Box::new(
    TLExpr::pred("safe", vec![Term::var("s")])
));
```

**Combined Modal & Temporal:**
```rust
// Box(F(goal(a))) - "In all possible worlds, agent a eventually achieves goal"
let expr = TLExpr::Box(Box::new(
    TLExpr::Eventually(Box::new(
        TLExpr::pred("goal", vec![Term::var("a")])
    ))
));
```

See `examples/10_modal_temporal_logic.rs` for comprehensive demonstrations.

### Parameterized Compilation

The compiler defines **6 preset configurations** and **26+ configurable strategies**:

```rust
use tensorlogic_compiler::{CompilationConfig, CompilationConfigBuilder};

// Use preset configurations
let config = CompilationConfig::soft_differentiable();  // Default (neural training)
let config = CompilationConfig::hard_boolean();         // Discrete reasoning
let config = CompilationConfig::fuzzy_godel();          // Godel fuzzy logic
let config = CompilationConfig::probabilistic();        // Probabilistic semantics

// Or build a custom configuration
let config = CompilationConfigBuilder::new()
    .and_strategy(AndStrategy::Product)           // Product t-norm
    .or_strategy(OrStrategy::ProbabilisticSum)    // Probabilistic s-norm
    .not_strategy(NotStrategy::Complement)        // Standard complement
    .exists_strategy(ExistsStrategy::Max)         // Max aggregation
    .build();
```

**Available Strategies:**

| Operation | Strategies | Use Cases |
|-----------|-----------|-----------|
| AND | Product, Min, ProbabilisticSum, Godel, ProductTNorm, Lukasiewicz | T-norms for conjunctions |
| OR | Max, ProbabilisticSum, Godel, ProbabilisticSNorm, Lukasiewicz | S-norms for disjunctions |
| NOT | Complement (1-x), Sigmoid | Negation with or without temperature |
| EXISTS | Sum, Max, LogSumExp, Mean | Different quantifier semantics |
| FORALL | DualOfExists, Product, Min, MeanThreshold | Universal quantification strategies |
| IMPLY | ReLU, Material, Godel, Lukasiewicz, Reichenbach | Various implication operators |
| MODAL | AllWorldsMin, AllWorldsProduct, Threshold | Necessity/possibility operators |
| TEMPORAL | Max, Sum, LogSumExp | Eventually/always operators |

## Advanced: Transitivity Rules

The compiler handles complex rules like transitivity with shared variables:

```rust
// knows(x,y) AND knows(y,z) -> knows(x,z)
let knows_xy = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
let knows_yz = TLExpr::pred("knows", vec![Term::var("y"), Term::var("z")]);
let knows_xz = TLExpr::pred("knows", vec![Term::var("x"), Term::var("z")]);

let premise = TLExpr::and(knows_xy, knows_yz);
let rule = TLExpr::imply(premise, knows_xz);

let graph = compile_to_einsum(&rule)?;

// Generates:
// 1. knows[ab] AND knows[bc] -> einsum("ab,bc->abc") [contraction over shared 'b']
// 2. Marginalize over 'b' to align with conclusion axes 'ac'
// 3. Apply ReLU(knows[ac] - marginalized_premise[ac])
```

## Optimization Pipeline Usage

### Unified Pipeline (Recommended)

```rust
use tensorlogic_compiler::optimize::{OptimizationPipeline, PipelineConfig};
use tensorlogic_ir::{TLExpr, Term};

let pipeline = OptimizationPipeline::new();
let expr = TLExpr::negate(TLExpr::negate(TLExpr::add(
    TLExpr::pow(x, TLExpr::Constant(2.0)),
    TLExpr::Constant(0.0),
)));

let (optimized, stats) = pipeline.optimize(&expr);

println!("Total optimizations: {}", stats.total_optimizations());
println!("  Negation: {}", stats.negation.double_negations_eliminated);
println!("  Constant folding: {}", stats.constant_folding.binary_ops_folded);
println!("  Algebraic: {}", stats.algebraic.identities_eliminated);
println!("  Strength reduction: {}", stats.strength_reduction.power_reductions);
println!("  Iterations: {}", stats.total_iterations);
println!("  Reached fixed point: {}", stats.reached_fixed_point);
```

### Configurable Pipeline

```rust
use tensorlogic_compiler::optimize::PipelineConfig;

// Aggressive optimization (more iterations)
let config = PipelineConfig::aggressive();
let pipeline = OptimizationPipeline::with_config(config);

// Custom configuration
let config = PipelineConfig::default()
    .with_negation_opt(true)
    .with_constant_folding(true)
    .with_algebraic_simplification(true)
    .with_strength_reduction(true)
    .with_distributivity(true)
    .with_quantifier_opt(true)
    .with_dead_code_elimination(true)
    .with_max_iterations(15);

let pipeline = OptimizationPipeline::with_config(config);
let (optimized, stats) = pipeline.optimize(&expr);
```

### Individual Pass Usage

```rust
use tensorlogic_compiler::optimize::{
    optimize_negations, fold_constants, simplify_algebraic,
    reduce_strength, optimize_distributivity, optimize_quantifiers,
    eliminate_dead_code,
};

let (opt1, stats1) = optimize_negations(&expr);
let (opt2, stats2) = fold_constants(&opt1);
let (opt3, stats3) = simplify_algebraic(&opt2);
let (opt4, stats4) = reduce_strength(&opt3);
```

### Complexity Analysis

```rust
use tensorlogic_compiler::optimize::{analyze_complexity, CostWeights};

let complexity = analyze_complexity(&expr);
println!("Max depth: {}", complexity.max_depth);
println!("Total operations: {}", complexity.total_operations());
println!("Total cost: {}", complexity.total_cost());

// Use GPU-optimized cost weights
let gpu_weights = CostWeights::gpu_optimized();
let gpu_cost = complexity.total_cost_with_weights(&gpu_weights);
println!("GPU-optimized cost: {}", gpu_cost);

println!("CSE potential: {}", complexity.cse_potential());
println!("Complexity level: {}", complexity.complexity_level());
```

### Graph-Level Optimizations

```rust
use tensorlogic_ir::graph::optimization::{optimize_graph, OptimizationLevel};

let graph = compile_to_einsum(&expr)?;
let (optimized_graph, stats) = optimize_graph(&graph, OptimizationLevel::Aggressive);
println!("Removed {} nodes", stats.nodes_removed);
```

## Advanced Analysis Features

### Compilation Profiling

```rust
use tensorlogic_compiler::profiling::CompilationProfiler;

let mut profiler = CompilationProfiler::new();
profiler.start();

profiler.start_phase("compilation");
let graph = compile_to_einsum(&expr)?;
profiler.end_phase("compilation");

profiler.record_pass("negation_opt", duration, optimizations_applied);

let report = profiler.generate_report();
println!("{}", report);

let json = profiler.generate_json_report();
```

**Profiling capabilities:**
- Phase-level time tracking with nesting support
- Memory usage snapshots and peak memory detection
- Pass-level statistics (execution count, time, optimizations)
- Cache statistics (hits, misses, evictions, hit rate)
- Performance recommendations based on profiling data

### Dataflow Analysis

```rust
use tensorlogic_compiler::passes::{analyze_dataflow, analyze_graph_dataflow};

let analysis = analyze_dataflow(&expr);
println!("Live variables: {:?}", analysis.live_variables);
println!("Reaching definitions: {:?}", analysis.reaching_defs);
println!("Available expressions: {:?}", analysis.available_exprs);
println!("Use-def chains: {:?}", analysis.use_def_chains);

let graph_analysis = analyze_graph_dataflow(&graph);
println!("Tensor dependencies: {:?}", graph_analysis.dependencies);
println!("Live tensors per node: {:?}", graph_analysis.live_tensors);
```

### Contraction Optimization

```rust
use tensorlogic_compiler::passes::{optimize_contractions, optimize_contractions_with_config};
use tensorlogic_compiler::passes::ContractionOptConfig;

let (optimized_graph, stats) = optimize_contractions(&graph);

println!("Contractions reordered: {}", stats.contractions_reordered);
println!("FLOPs reduction: {:.1}%", stats.flops_reduction_percent);
println!("Memory reduction: {:.1}%", stats.memory_reduction_percent);

let config = ContractionOptConfig {
    max_intermediate_size: 1_000_000,
    prefer_memory_over_flops: false,
};

let (optimized, stats) = optimize_contractions_with_config(&graph, &config);
```

### Loop Fusion

```rust
use tensorlogic_compiler::passes::{fuse_loops, fuse_loops_with_config};
use tensorlogic_compiler::passes::LoopFusionConfig;

let (fused_graph, stats) = fuse_loops(&graph);

println!("Loops fused: {}", stats.loops_fused);
println!("Reductions merged: {}", stats.reductions_merged);
println!("Intermediates eliminated: {}", stats.intermediates_eliminated);
```

### Reachability Analysis

```rust
use tensorlogic_compiler::passes::{analyze_reachability, analyze_dominance};

let reachability = analyze_reachability(&graph);

if reachability.reachable.contains(&(node_a, node_b)) {
    println!("Node {} can reach node {}", node_a, node_b);
}

println!("SCCs: {:?}", reachability.strongly_connected_components);

if let Some(topo) = &reachability.topological_order {
    println!("Topological order: {:?}", topo);
}

let dominance = analyze_dominance(&graph);
println!("Immediate dominators: {:?}", dominance.immediate_dominators);
println!("Dominance frontiers: {:?}", dominance.dominance_frontiers);
```

### Integrated Post-Compilation Pipeline

```rust
use tensorlogic_compiler::passes::{post_compilation_passes, PostCompilationOptions};

let options = PostCompilationOptions {
    validate_graph_structure: true,
    validate_axes: true,
    validate_shapes: true,
    apply_optimizations: true,
    enable_contraction_opt: true,
    enable_loop_fusion: true,
    strict_mode: false,
};

let mut graph = compile_to_einsum(&expr)?;
let result = post_compilation_passes(&mut graph, &ctx, options)?;

if result.is_valid {
    println!("Graph validated successfully");
    println!("  Checks performed: {}", result.validation_report.checks_performed);
    println!("  Optimizations: {}", result.optimizations_applied);
}
```

See `examples/21_profiling_and_optimization.rs` for comprehensive demonstrations of all these features.

## Compiler Architecture

```
TLExpr
  |
[Pre-Compilation Passes]
  - Scope analysis (detect unbound variables)
  - Type checking (validate arity, types)
  - Negation optimization
  - Common subexpression elimination
  |
[Compiler Context]
  - Assign axes to variables
  - Track domains
  - Manage temporary tensors
  - Apply compilation config
  |
[compile_expr recursion]
  - compile_predicate -> tensor with axes
  - compile_and -> einsum contraction (configurable)
  - compile_or -> element-wise max (configurable)
  - compile_not -> 1 - x (configurable)
  - compile_exists -> reduction (configurable)
  - compile_forall -> dual or product (configurable)
  - compile_imply -> marginalize + operator (configurable)
  - compile_arithmetic -> element-wise ops
  - compile_comparison -> boolean tensors
  |
[Post-Compilation Passes]
  - Dead code elimination
  - Einsum operation merging
  - Identity elimination
  - Contraction order optimization
  |
EinsumGraph
  - Tensors: Vec<String>
  - Nodes: Vec<EinsumNode>
  - Outputs: Vec<usize>
```

## Scope Analysis & Type Checking

### Scope Analysis

```rust
use tensorlogic_compiler::passes::scope_analysis::analyze_scopes;

let expr = TLExpr::exists("x", "Person",
    TLExpr::and(
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
        TLExpr::pred("likes", vec![Term::var("x"), Term::var("z")]),
    )
);

let analysis = analyze_scopes(&expr);

if !analysis.unbound_vars.is_empty() {
    println!("Unbound variables: {:?}", analysis.unbound_vars);
    println!("Suggestions: {}", analysis.suggest_quantifiers());
}
```

### Type Checking

```rust
use tensorlogic_compiler::passes::type_checking::TypeChecker;
use tensorlogic_ir::PredicateSignature;

let mut checker = TypeChecker::new();

checker.register_predicate(PredicateSignature {
    name: "knows".to_string(),
    arity: 2,
    arg_types: vec![Some("Person".to_string()), Some("Person".to_string())],
});

let result = checker.check_types(&expr);
if let Some(error) = result.type_errors.first() {
    println!("Type error: {}", error);
}
```

### Enhanced Diagnostics

```rust
use tensorlogic_compiler::passes::diagnostics::{diagnose_expression, DiagnosticLevel};

let diagnostics = diagnose_expression(&expr);

for diag in diagnostics {
    match diag.level {
        DiagnosticLevel::Error => eprintln!("ERROR: {}", diag.message),
        DiagnosticLevel::Warning => eprintln!("WARNING: {}", diag.message),
        DiagnosticLevel::Hint => println!("HINT: {}", diag.message),
        _ => {}
    }

    if let Some(help) = diag.help {
        println!("  Help: {}", help);
    }
}
```

## Compiler Context

The `CompilerContext` manages compilation state:

```rust
use tensorlogic_compiler::CompilerContext;

let mut ctx = CompilerContext::new();

// Register domains
ctx.add_domain("Person", 100);  // 100 possible persons
ctx.add_domain("City", 50);     // 50 cities

// Bind variables to domains
ctx.bind_var("x", "Person")?;
ctx.bind_var("y", "City")?;

// Axes are automatically assigned: x->'a', y->'b', ...
```

## Operation Types

The compiler generates 4 types of operations:

### 1. Einsum (Tensor Contraction)
```rust
// Spec: "ab,bc->ac" (matrix multiplication)
EinsumNode::einsum("ab,bc->ac", vec![tensor0, tensor1])
```

### 2. Element-Wise Unary
```rust
// Operations: not, relu, sigmoid, etc.
EinsumNode::elem_unary("relu", tensor_idx)
```

### 3. Element-Wise Binary
```rust
// Operations: add, subtract, multiply, etc.
EinsumNode::elem_binary("subtract", left_idx, right_idx)
```

### 4. Reduction
```rust
// Reduce over axis 1 (sum/max/min)
EinsumNode::reduce("sum", vec![1], tensor_idx)
```

## Error Handling

The compiler performs extensive validation:

```rust
// Arity validation
let p1 = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
let p2 = TLExpr::pred("P", vec![Term::var("a")]);  // Different arity!
validate_arity(&TLExpr::and(p1, p2))?;  // Error: Predicate 'P' has inconsistent arity

// Domain validation
ctx.bind_var("x", "NonExistent")?;  // Error: Domain 'NonExistent' not found
```

## Integration with Other Crates

### tensorlogic-adapters
Use `SymbolTable` to provide domain and predicate metadata:

```rust
use tensorlogic_adapters::SymbolTable;

let table = SymbolTable::new();
// Add domains and predicates...
// Pass to compiler for enhanced type checking
```

### tensorlogic-scirs-backend
Execute the compiled graph:

```rust
use tensorlogic_scirs_backend::Scirs2Exec;
use tensorlogic_infer::TlExecutor;

let executor = Scirs2Exec::new();
let outputs = executor.execute(&graph, &inputs)?;
```

## Performance Considerations

- **Operation Fusion**: Einsum operation merging (completed)
- **Common Subexpression Elimination**: Expression-level and graph-level CSE (completed)
- **Negation Optimization**: De Morgan's laws and double negation elimination (completed)
- **Dead Code Elimination**: Removes unused operations from the graph (completed)
- **Axis Assignment**: Uses lexicographic order ('a', 'b', 'c', ...) for determinism
- **Temporary Tensors**: Named as `temp_0`, `temp_1`, ... for debugging

## Testing & Quality

The compiler has comprehensive test coverage:

```bash
# Run all tests with nextest (recommended)
cargo nextest run -p tensorlogic-compiler

# Run with standard cargo test
cargo test -p tensorlogic-compiler

# Run with coverage
cargo llvm-cov --package tensorlogic-compiler
```

**Current Test Status (v0.1.3):**
- **920 tests** (100% passing)
- **Zero warnings** (strict clippy compliance)
- Stable quality

## Current Status & Roadmap

### Stable (v0.1.3)
- Core logic compilation (AND, OR, NOT, quantifiers, implications)
- Arithmetic and comparison operations
- Conditional expressions (if-then-else)
- Type checking and scope analysis
- Enhanced diagnostics with helpful error messages
- Parameterized compilation (26+ strategies, 6 presets)
- Optimization passes (negation, CSE, einsum, DCE)
- SymbolTable integration for metadata
- Modal & temporal logic (Box, Diamond, Eventually, Always)
- Advanced logic: counting quantifiers, higher-order logic, set theory, fixed-points
- Hybrid logic, constraint programming, abductive reasoning
- Probabilistic logic, fuzzy logic operators
- Import: Prolog, S-Expression, TPTP formats
- Export: ONNX, TensorFlow GraphDef, PyTorch code generation
- Parallel compilation (feature-gated)
- Incremental compilation and caching
- Compilation profiling
- Dataflow analysis
- Contraction optimization
- Loop fusion
- Reachability analysis
- Property-based testing (21 property tests)
- Fuzzing infrastructure (4 fuzz targets)
- Benchmark suite
- **CompilerPipeline** (v0.1.17): end-to-end composable pass chain with `CompilerPassOrder`
- **Symbolic differentiation** (v0.1.18): `differentiate()`, `jacobian()`, full arithmetic/logic/fuzzy support
- **Partial evaluation** (v0.1.19): `partially_evaluate()`, `specialize()`, branch pruning
- **Type inference** (v0.1.20): `TLType`, `annotate()`, Hindley-Milner-lite unification
- **Bytecode VM** (v0.1.21): 40-instruction stack VM, `compile()`, `execute()`

### Known Limitations
- `Next` (X) temporal operator requires backend shift operations
- `Until` (U) temporal operator requires backend scan operations
- JIT compilation for hot paths: not yet implemented
- First-class functions/predicates: not yet implemented
- Higher-order quantification: not yet implemented

## Examples

See the test suite and examples directory for demonstrations:

```bash
cargo test -p tensorlogic-compiler
```

Key examples:
- `examples/10_modal_temporal_logic.rs`: Box, Diamond, Eventually, Always operators
- `examples/11_fuzzy_logic.rs`: All 19 fuzzy operators with real-world applications
- `examples/14_parallel_compilation.rs`: Multi-threaded compilation
- `examples/15_onnx_export.rs`: ONNX format export
- `examples/16_tensorflow_export.rs`: TensorFlow GraphDef export
- `examples/17_pytorch_export.rs`: PyTorch code generation
- `examples/18_logic_import.rs`: Import from Prolog, S-Expression, TPTP
- `examples/19_set_operations.rs`: Set theory operations
- `examples/20_constraint_programming.rs`: AllDifferent, GlobalCardinality
- `examples/21_profiling_and_optimization.rs`: Profiling and advanced analysis
- `examples/22_hybrid_logic.rs`: Hybrid logic operators
- `examples/23_abductive_reasoning.rs`: Abductive reasoning

Key test cases:
- `test_transitivity_rule_shared_variables`: Transitivity with contraction
- `test_and_with_different_axes`: Partial variable overlap
- `test_and_with_disjoint_variables`: Outer product (no shared vars)
- `test_implication`: Soft implication with ReLU
- `test_exists_quantifier`: Reduction over quantified variables

## Contributing

When adding new features:
1. Update `compile_expr` to handle new TLExpr variants
2. Add tests in the `tests` module
3. Update this README and TODO.md
4. Ensure all tests pass: `cargo nextest run -p tensorlogic-compiler`

## License

Apache-2.0

---

**Status**: Stable (v0.1.3)
**Last Updated**: 2026-08-31
**Tests**: 920/920 passing (100%)
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
