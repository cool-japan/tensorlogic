# TensorLogic Tutorial

Welcome to TensorLogic! This tutorial will guide you through the key concepts and features of TensorLogic, a library for compiling logical expressions into tensor computation graphs.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Concepts](#basic-concepts)
4. [Quick Start](#quick-start)
5. [Predicates and Terms](#predicates-and-terms)
6. [Logical Operations](#logical-operations)
7. [Quantifiers](#quantifiers)
8. [Compilation Strategies](#compilation-strategies)
9. [Using the CLI Tool](#using-the-cli-tool)
10. [Advanced Topics](#advanced-topics)

## Introduction

TensorLogic bridges symbolic logic and numeric tensor computations. It translates high-level logical expressions (predicates, quantifiers, implications) into efficient tensor operations that can be executed on various backends (CPU, GPU, etc.).

### Key Features

- **Logic-to-Tensor Compilation**: Convert logical rules into einsum operations
- **Multiple Strategies**: Choose from 6 compilation strategies (soft/hard/fuzzy/probabilistic)
- **Type Safety**: Comprehensive type checking and scope analysis
- **Optimization**: Automatic optimization passes (CSE, negation, einsum)
- **Multiple Backends**: SciRS2 backend with CPU/SIMD support (GPU coming soon)
- **Python Bindings**: PyO3-based Python API
- **CLI Tool**: Command-line compiler for quick experiments

## Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
tensorlogic-compiler = "0.1.0-beta.1"
tensorlogic-ir = "0.1.0-beta.1"
tensorlogic-scirs-backend = "0.1.0-beta.1"
```

### Python

```bash
pip install tensorlogic-py
```

### CLI Tool

```bash
cargo install tensorlogic --bin tlc
```

## Basic Concepts

### Domains

Domains represent sets of entities. Each variable in your logical expressions ranges over a domain.

```rust
use tensorlogic_compiler::CompilerContext;

let mut ctx = CompilerContext::new();
ctx.add_domain("Person", 100);  // 100 people
ctx.add_domain("City", 50);     // 50 cities
```

### Tensors

Logical predicates are represented as tensors:
- Unary predicate `tall(x)` → 1D tensor of shape `[100]`
- Binary predicate `knows(x, y)` → 2D tensor of shape `[100, 100]`
- Values typically range from 0.0 (false) to 1.0 (true)

### Einsum Notation

TensorLogic compiles logical operations to einsum specifications:
- `knows(x, y) ∧ knows(y, z)` → `"xy,yz->xyz"` (matrix multiplication pattern)
- `∃y. knows(x, y)` → Reduction over y-axis

## Quick Start

### Example 1: Simple Predicate

```rust
use tensorlogic_compiler::{compile_to_einsum, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

// Define: knows(x, y)
let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

// Compile to tensor graph
let mut ctx = CompilerContext::new();
let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;

println!("Compiled graph: {:?}", graph);
```

### Example 2: Logical AND

```rust
// Define: knows(x, y) ∧ likes(x, y)
let expr = TLExpr::and(
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
    TLExpr::pred("likes", vec![Term::var("x"), Term::var("y")])
);

let graph = compile_to_einsum(&expr)?;
```

### Example 3: Quantifier

```rust
// Define: ∃y. knows(x, y)  (x knows someone)
let expr = TLExpr::exists(
    "y",
    "Person",
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")])
);

let mut ctx = CompilerContext::new();
ctx.add_domain("Person", 100);

let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;
```

## Predicates and Terms

### Creating Predicates

```rust
use tensorlogic_ir::{TLExpr, Term};

// Unary predicate: tall(x)
let expr1 = TLExpr::pred("tall", vec![Term::var("x")]);

// Binary predicate: knows(x, y)
let expr2 = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

// Ternary predicate: gave(x, y, z)
let expr3 = TLExpr::pred("gave", vec![
    Term::var("x"),
    Term::var("y"),
    Term::var("z")
]);
```

### Typed Terms

```rust
// Specify types explicitly
let term = Term::typed_var("x", "Person");

let expr = TLExpr::pred("tall", vec![term]);
```

## Logical Operations

### Conjunction (AND)

```rust
// knows(x, y) ∧ likes(x, y)
let expr = TLExpr::and(
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
    TLExpr::pred("likes", vec![Term::var("x"), Term::var("y")])
);
```

**Tensor Semantics** (soft_differentiable strategy):
- Result = `knows(x,y) * likes(x,y)` (element-wise product)

### Disjunction (OR)

```rust
// tall(x) ∨ smart(x)
let expr = TLExpr::or(
    TLExpr::pred("tall", vec![Term::var("x")]),
    TLExpr::pred("smart", vec![Term::var("x")])
);
```

**Tensor Semantics**:
- Result = `max(tall(x), smart(x))` (element-wise max)

### Negation (NOT)

```rust
// ¬knows(x, y)
let expr = TLExpr::negate(
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")])
);
```

**Tensor Semantics**:
- Result = `1 - knows(x,y)` (complement)

### Implication

```rust
// knows(x, y) → likes(x, y)
let expr = TLExpr::imply(
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
    TLExpr::pred("likes", vec![Term::var("x"), Term::var("y")])
);
```

**Tensor Semantics**:
- Result = `ReLU(likes(x,y) - knows(x,y))`

## Quantifiers

### Existential Quantification (∃)

```rust
// ∃y. knows(x, y)  "x knows someone"
let expr = TLExpr::exists(
    "y",
    "Person",
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")])
);
```

**Tensor Semantics**:
- Result = `sum(knows(x, y), axis=y)` (reduce over y-axis)

### Universal Quantification (∀)

```rust
// ∀y. knows(x, y)  "x knows everyone"
let expr = TLExpr::forall(
    "y",
    "Person",
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")])
);
```

**Tensor Semantics** (via dual of exists):
- Result = `NOT(EXISTS y. NOT(knows(x, y)))`

### Nested Quantifiers

```rust
// ∀x. ∃y. knows(x, y)  "everyone knows someone"
let expr = TLExpr::forall(
    "x",
    "Person",
    TLExpr::exists(
        "y",
        "Person",
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")])
    )
);
```

## Compilation Strategies

TensorLogic supports 6 compilation strategies, each with different tensor mappings:

### 1. Soft Differentiable (Default)

Optimized for neural network training with smooth, differentiable operations.

```rust
use tensorlogic_compiler::{CompilerContext, CompilationConfig};

let config = CompilationConfig::soft_differentiable();
let ctx = CompilerContext::with_config(config);
```

**Mappings:**
- AND: Product (`a * b`)
- OR: Max (`max(a, b)`)
- NOT: Complement (`1 - a`)
- EXISTS: Sum reduction (`sum(P, axis=x)`)
- FORALL: Dual (`NOT(EXISTS x. NOT(P))`)

### 2. Hard Boolean

Discrete Boolean logic for exact reasoning.

```rust
let config = CompilationConfig::hard_boolean();
```

**Mappings:**
- AND: Min (`min(a, b)`)
- OR: Max (`max(a, b)`)
- NOT: Complement (`1 - a`)

### 3. Fuzzy Gödel

Gödel fuzzy logic with min/max semantics.

```rust
let config = CompilationConfig::fuzzy_godel();
```

**Mappings:**
- AND: Min (`min(a, b)`)
- OR: Max (`max(a, b)`)
- Implication: Gödel (`if a ≤ b then 1 else b`)

### 4. Fuzzy Product

Product fuzzy logic.

```rust
let config = CompilationConfig::fuzzy_product();
```

**Mappings:**
- AND: Product (`a * b`)
- OR: Probabilistic sum (`a + b - a*b`)

### 5. Fuzzy Łukasiewicz

Łukasiewicz fuzzy logic.

```rust
let config = CompilationConfig::fuzzy_lukasiewicz();
```

**Mappings:**
- AND: Łukasiewicz T-norm (`max(0, a + b - 1)`)
- OR: Łukasiewicz S-norm (`min(1, a + b)`)

### 6. Probabilistic

Probabilistic interpretation.

```rust
let config = CompilationConfig::probabilistic();
```

**Mappings:**
- AND: Probabilistic sum (`a + b - a*b`)
- OR: Probabilistic sum (`a + b - a*b`)
- EXISTS: Mean (`mean(P, axis=x)`)

## Using the CLI Tool

The `tensorlogic` command-line tool provides quick compilation without writing code.

### Basic Usage

```bash
# Compile a simple predicate
tensorlogic "knows(x, y)"

# Compile with explicit domain
tensorlogic -d Person:100 "tall(x)"

# Show statistics
tensorlogic "knows(x, y) & likes(x, y)" --output-format stats
```

### Quantifiers

```bash
# EXISTS
tensorlogic "exists y. knows(x, y)"

# FORALL
tensorlogic "forall x. tall(x) -> smart(x)"
```

### Output Formats

```bash
# DOT format for visualization
tensorlogic "knows(x, y)" --output-format dot > graph.dot
dot -Tpng graph.dot > graph.png

# JSON for programmatic use
tensorlogic "knows(x, y)" --output-format json > graph.json

# Statistics only
tensorlogic "exists y. knows(x, y)" --output-format stats
```

### Compilation Strategies

```bash
# Use fuzzy logic
tensorlogic --strategy fuzzy_godel "knows(x, y) & friends(x, y)"

# Use hard Boolean logic
tensorlogic --strategy hard_boolean "tall(x) | smart(x)"
```

### Validation

```bash
# Validate compiled graph
tensorlogic --validate "forall x. knows(x, y) -> likes(x, y)"
```

### Debug Mode

```bash
# Enable debug output
tensorlogic --debug "exists y. knows(x, y)"
```

## Advanced Topics

### Complex Rules

#### Transitivity

```rust
// ∀x,y,z. knows(x,y) ∧ knows(y,z) → knows(x,z)
let rule = TLExpr::forall("x", "Person",
    TLExpr::forall("y", "Person",
        TLExpr::forall("z", "Person",
            TLExpr::imply(
                TLExpr::and(
                    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
                    TLExpr::pred("knows", vec![Term::var("y"), Term::var("z")])
                ),
                TLExpr::pred("knows", vec![Term::var("x"), Term::var("z")])
            )
        )
    )
);
```

#### Symmetric Relations

```rust
// ∀x,y. friends(x,y) → friends(y,x)
let rule = TLExpr::forall("x", "Person",
    TLExpr::forall("y", "Person",
        TLExpr::imply(
            TLExpr::pred("friends", vec![Term::var("x"), Term::var("y")]),
            TLExpr::pred("friends", vec![Term::var("y"), Term::var("x")])
        )
    )
);
```

### Execution with SciRS2 Backend

```rust
use tensorlogic_scirs_backend::Scirs2Executor;
use tensorlogic_infer::TlExecutor;
use scirs2_core::ndarray::Array2;

// Compile expression
let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
let mut ctx = CompilerContext::new();
let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;

// Create input tensor (100x100 adjacency matrix)
let knows_data = Array2::zeros((100, 100));
let inputs = vec![knows_data.into_dyn()];

// Execute
let executor = Scirs2Executor::new();
let outputs = executor.execute(&graph, &inputs)?;

println!("Output shape: {:?}", outputs[0].shape());
```

### Graph Visualization

```rust
use tensorlogic_ir::export_to_dot;

let graph = compile_to_einsum(&expr)?;
let dot = export_to_dot(&graph);

// Save to file
std::fs::write("graph.dot", dot)?;
```

Then visualize with Graphviz:
```bash
dot -Tpng graph.dot > graph.png
```

### Validation

```rust
use tensorlogic_ir::validate_graph;

let graph = compile_to_einsum(&expr)?;
let report = validate_graph(&graph);

if !report.is_valid() {
    for error in &report.errors {
        eprintln!("Error: {}", error.message);
    }
}

println!("Statistics: {} tensors, {} nodes",
    report.stats.total_tensors,
    report.stats.total_nodes
);
```

### Debug Tracing

```rust
use tensorlogic_compiler::debug::CompilationTracer;

let mut tracer = CompilationTracer::new(true);
tracer.start(&expr);

// During compilation...
let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;

let trace = tracer.finish(&graph).unwrap();
trace.print_summary();
```

## Common Patterns

### Pattern 1: Social Network Analysis

```rust
// Find mutual friends: ∃z. knows(x,z) ∧ knows(y,z)
let mutual_friends = TLExpr::exists("z", "Person",
    TLExpr::and(
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("z")]),
        TLExpr::pred("knows", vec![Term::var("y"), Term::var("z")])
    )
);
```

### Pattern 2: Recommendation Systems

```rust
// Recommend: likes(x, i) ∧ ∃y. (similar(x,y) ∧ likes(y, i))
let recommend = TLExpr::and(
    TLExpr::pred("likes", vec![Term::var("x"), Term::var("i")]),
    TLExpr::exists("y", "User",
        TLExpr::and(
            TLExpr::pred("similar", vec![Term::var("x"), Term::var("y")]),
            TLExpr::pred("likes", vec![Term::var("y"), Term::var("i")])
        )
    )
);
```

### Pattern 3: Knowledge Graph Reasoning

```rust
// Infer relationships: parent(x,y) → ancestor(x,y)
let rule = TLExpr::forall("x", "Person",
    TLExpr::forall("y", "Person",
        TLExpr::imply(
            TLExpr::pred("parent", vec![Term::var("x"), Term::var("y")]),
            TLExpr::pred("ancestor", vec![Term::var("x"), Term::var("y")])
        )
    )
);
```

## Best Practices

1. **Use Type Annotations**: Explicitly specify domains for better error messages
2. **Start Simple**: Begin with simple predicates and gradually add complexity
3. **Validate Early**: Use `validate_graph()` to catch issues early
4. **Choose Strategy Wisely**: Use `soft_differentiable` for learning, `hard_boolean` for exact reasoning
5. **Visualize Graphs**: Use DOT export to understand compilation results
6. **Profile Execution**: Use the profiling APIs to identify bottlenecks

## Troubleshooting

### "Domain not found" Error

**Problem**: Domain referenced in quantifier but not defined in context.

**Solution**:
```rust
ctx.add_domain("Person", 100);  // Add before compilation
```

### "Unbound variable" Error

**Problem**: Variable used but not bound by quantifier.

**Solution**: Wrap in appropriate quantifier or check variable names.

### "Arity mismatch" Error

**Problem**: Predicate used with wrong number of arguments.

**Solution**: Ensure predicate arity is consistent across all uses.

## Next Steps

- Explore the [examples directory](examples/) for more code samples
- Read the [API documentation](https://docs.rs/tensorlogic-compiler)
- Join our [community discussions](https://github.com/cool-japan/tensorlogic/discussions)
- Check out the [Python tutorials](crates/tensorlogic-py/tutorials/)

## Resources

- **Paper**: [Tensor Logic on arXiv](https://arxiv.org/abs/2510.12269)
- **Repository**: https://github.com/cool-japan/tensorlogic
- **Documentation**: https://docs.rs/tensorlogic-compiler
- **Examples**: [examples/](examples/)
- **Python Bindings**: [crates/tensorlogic-py/](crates/tensorlogic-py/)

Happy logical tensor programming! 🎉
