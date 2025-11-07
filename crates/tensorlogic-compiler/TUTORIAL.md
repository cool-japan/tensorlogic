# TensorLogic Compiler Tutorial

This tutorial provides a comprehensive guide to using the tensorlogic-compiler crate to compile logical expressions into tensor computation graphs.

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Step-by-Step Compilation](#step-by-step-compilation)
5. [Compilation Strategies](#compilation-strategies)
6. [Advanced Features](#advanced-features)
7. [Optimization Passes](#optimization-passes)
8. [Custom Operations](#custom-operations)
9. [Debugging and Validation](#debugging-and-validation)
10. [Best Practices](#best-practices)

## Introduction

The tensorlogic-compiler translates high-level logical expressions (predicates, quantifiers, implications) into low-level tensor operations that can be executed efficiently on various backends (CPU, GPU, etc.).

### Key Features

- **Logic-to-tensor mapping** with configurable strategies
- **Type checking** and scope analysis
- **Optimization passes** (negation, CSE, einsum optimization)
- **Enhanced diagnostics** with helpful error messages
- **Support for arithmetic**, comparison, and conditional expressions
- **Custom operation registration** for extensibility

## Quick Start

```rust
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

// Create a compiler context
let mut ctx = CompilerContext::new();
ctx.add_domain("Person", 100);

// Define a logic rule: ∃y. knows(x, y)
// "Find all persons x who know someone"
let rule = TLExpr::exists(
    "y",
    "Person",
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
);

// Compile to tensor operations
let graph = compile_to_einsum_with_context(&rule, &mut ctx).unwrap();

println!("Compiled graph with {} nodes", graph.nodes.len());
```

## Core Concepts

### Logical Expressions (TLExpr)

TensorLogic supports various logical constructs:

```rust
use tensorlogic_ir::{TLExpr, Term};

// Predicates
let knows = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

// Logical operations
let and_expr = TLExpr::And(
    Box::new(TLExpr::pred("p", vec![Term::var("x")])),
    Box::new(TLExpr::pred("q", vec![Term::var("x")])),
);

let or_expr = TLExpr::Or(
    Box::new(TLExpr::pred("p", vec![Term::var("x")])),
    Box::new(TLExpr::pred("q", vec![Term::var("x")])),
);

let not_expr = TLExpr::Not(Box::new(TLExpr::pred("p", vec![Term::var("x")])));

// Quantifiers
let exists_expr = TLExpr::exists(
    "y",
    "Domain",
    TLExpr::pred("p", vec![Term::var("x"), Term::var("y")]),
);

let forall_expr = TLExpr::forall(
    "y",
    "Domain",
    TLExpr::pred("p", vec![Term::var("x"), Term::var("y")]),
);

// Implications
let imply_expr = TLExpr::imply(
    TLExpr::pred("p", vec![Term::var("x")]),
    TLExpr::pred("q", vec![Term::var("x")]),
);
```

### Compiler Context

The `CompilerContext` tracks domains, variable bindings, and compilation configuration:

```rust
use tensorlogic_compiler::{CompilerContext, CompilationConfig};

let mut ctx = CompilerContext::new();

// Add domains
ctx.add_domain("Person", 100);
ctx.add_domain("City", 50);

// Bind variables to domains
ctx.bind_var("x", "Person").unwrap();
ctx.bind_var("c", "City").unwrap();

// Configure compilation strategy
let config = CompilationConfig::soft_differentiable();
let ctx_with_config = CompilerContext::with_config(config);
```

### Einsum Graphs

The output of compilation is an `EinsumGraph`, which represents tensor operations:

```rust
// After compilation
let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

println!("Tensors: {}", graph.tensors.len());
println!("Nodes: {}", graph.nodes.len());
println!("Outputs: {:?}", graph.outputs);
```

## Step-by-Step Compilation

### 1. Define Your Domain

First, define the domains (entity types) in your problem:

```rust
use tensorlogic_compiler::CompilerContext;

let mut ctx = CompilerContext::new();
ctx.add_domain("Person", 100);   // 100 persons
ctx.add_domain("Course", 30);    // 30 courses
```

### 2. Build Your Logical Expression

Construct the logical rule you want to compile:

```rust
use tensorlogic_ir::{TLExpr, Term};

// Rule: ∃c. enrolled(x, c) ∧ difficult(c)
// "Find persons enrolled in a difficult course"
let rule = TLExpr::exists(
    "c",
    "Course",
    TLExpr::And(
        Box::new(TLExpr::pred("enrolled", vec![Term::var("x"), Term::var("c")])),
        Box::new(TLExpr::pred("difficult", vec![Term::var("c")])),
    ),
);
```

### 3. Validate the Expression (Optional but Recommended)

```rust
use tensorlogic_compiler::passes::validate_expression;

let validation_result = validate_expression(&rule, &ctx);

if !validation_result.is_valid() {
    println!("Validation errors:");
    for diag in &validation_result.diagnostics {
        println!("  - {}", diag.message);
    }
}
```

### 4. Compile to Einsum Graph

```rust
use tensorlogic_compiler::compile_to_einsum_with_context;

let graph = compile_to_einsum_with_context(&rule, &mut ctx).unwrap();
```

### 5. Optimize (Optional)

```rust
use tensorlogic_compiler::passes::post_compilation_passes;
use tensorlogic_compiler::passes::PostCompilationOptions;

let options = PostCompilationOptions::default();
let result = post_compilation_passes(&mut graph, &ctx, options).unwrap();

println!("Optimizations applied: {}", result.optimizations_applied);
```

## Compilation Strategies

TensorLogic supports multiple compilation strategies for different use cases:

### 1. Soft Differentiable (Default)

Optimized for neural network training and gradient-based optimization:

```rust
use tensorlogic_compiler::CompilationConfig;

let config = CompilationConfig::soft_differentiable();
let mut ctx = CompilerContext::with_config(config);
```

**Mappings:**
- AND → Product (a * b)
- OR → Probabilistic sum (a + b - a*b)
- NOT → Complement (1 - a)
- EXISTS → Sum reduction
- FORALL → Dual of exists
- Implication → ReLU(b - a)

### 2. Hard Boolean

For discrete reasoning with Boolean-like values:

```rust
let config = CompilationConfig::hard_boolean();
let mut ctx = CompilerContext::with_config(config);
```

**Mappings:**
- AND → Min
- OR → Max
- NOT → Complement
- EXISTS → Max reduction
- FORALL → Min reduction
- Implication → Material (NOT(a) OR b)

### 3. Fuzzy Logic (Gödel)

Standard fuzzy logic with Gödel semantics:

```rust
let config = CompilationConfig::fuzzy_godel();
let mut ctx = CompilerContext::with_config(config);
```

### 4. Fuzzy Logic (Product)

Fuzzy logic with product t-norms:

```rust
let config = CompilationConfig::fuzzy_product();
let mut ctx = CompilerContext::with_config(config);
```

### 5. Fuzzy Logic (Łukasiewicz)

Fully differentiable Łukasiewicz fuzzy logic:

```rust
let config = CompilationConfig::fuzzy_lukasiewicz();
let mut ctx = CompilerContext::with_config(config);
```

### 6. Probabilistic

Interprets logical operations as probabilistic events:

```rust
let config = CompilationConfig::probabilistic();
let mut ctx = CompilerContext::with_config(config);
```

### Custom Configuration

Build a custom configuration:

```rust
use tensorlogic_compiler::{CompilationConfig, AndStrategy, OrStrategy};

let config = CompilationConfig::custom()
    .and_strategy(AndStrategy::Min)
    .or_strategy(OrStrategy::Max)
    .build();

let mut ctx = CompilerContext::with_config(config);
```

## Advanced Features

### Automatic Strategy Selection

Let the compiler recommend a strategy based on expression characteristics:

```rust
use tensorlogic_compiler::passes::{recommend_strategy, OptimizationGoal};

let recommendation = recommend_strategy(&expr, OptimizationGoal::Differentiable);

println!("Recommended: {}", recommendation.rationale);
println!("Confidence: {:.2}", recommendation.confidence);

// Use the recommended config
let mut ctx = CompilerContext::with_config(recommendation.config);
```

**Optimization Goals:**
- `Differentiable` - Prioritize differentiability for training
- `DiscreteReasoning` - Prioritize Boolean accuracy
- `Performance` - Prioritize computational efficiency
- `Balanced` - Balance between differentiability and accuracy

### Arithmetic and Comparisons

Mix logical and arithmetic operations:

```rust
// Rule: price(x) > 100 ∧ inStock(x)
// "Find items that are expensive and in stock"
let rule = TLExpr::And(
    Box::new(TLExpr::GreaterThan(
        Box::new(TLExpr::pred("price", vec![Term::var("x")])),
        Box::new(TLExpr::Constant(100.0)),
    )),
    Box::new(TLExpr::pred("inStock", vec![Term::var("x")])),
);
```

### Conditional Expressions

Use if-then-else for conditional logic:

```rust
// Rule: if score(x) > 90 then "A" else "B"
let rule = TLExpr::If {
    condition: Box::new(TLExpr::GreaterThan(
        Box::new(TLExpr::pred("score", vec![Term::var("x")])),
        Box::new(TLExpr::Constant(90.0)),
    )),
    then_expr: Box::new(TLExpr::Constant(1.0)),  // "A" grade
    else_expr: Box::new(TLExpr::Constant(0.0)),  // "B" grade
};
```

### Complex Nested Quantifiers

Handle complex nested quantification:

```rust
// Rule: ∀x. ∃y. ∀z. (knows(x,y) ∧ knows(y,z)) → knows(x,z)
// "Transitive closure of knows relation"
let rule = TLExpr::forall(
    "x",
    "Person",
    TLExpr::exists(
        "y",
        "Person",
        TLExpr::forall(
            "z",
            "Person",
            TLExpr::imply(
                TLExpr::And(
                    Box::new(TLExpr::pred("knows", vec![
                        Term::var("x"),
                        Term::var("y")
                    ])),
                    Box::new(TLExpr::pred("knows", vec![
                        Term::var("y"),
                        Term::var("z")
                    ])),
                ),
                TLExpr::pred("knows", vec![Term::var("x"), Term::var("z")]),
            ),
        ),
    ),
);
```

## Optimization Passes

### Expression-Level Optimizations

Apply optimizations before compilation:

```rust
use tensorlogic_compiler::optimize::optimize_negations;

// Optimize double negations and De Morgan's laws
let (optimized_expr, stats) = optimize_negations(&expr);
println!("Double negations eliminated: {}", stats.double_negations_eliminated);
println!("De Morgan applications: {}", stats.demorgans_applied);
```

### Common Subexpression Elimination

```rust
use tensorlogic_compiler::passes::eliminate_common_subexpressions;

let (optimized_expr, cse_result) = eliminate_common_subexpressions(&expr);
println!("Subexpressions eliminated: {}", cse_result.eliminations);
```

### Post-Compilation Optimizations

Optimize the compiled graph:

```rust
use tensorlogic_compiler::passes::{post_compilation_passes, PostCompilationOptions};

let options = PostCompilationOptions {
    validate_graph_structure: true,
    validate_axes: true,
    validate_shapes: true,
    apply_optimizations: true,
    strict_mode: false,
};

let mut graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
let result = post_compilation_passes(&mut graph, &ctx, options).unwrap();

println!("Validation: {}", if result.is_valid { "PASSED" } else { "FAILED" });
println!("Optimizations: {}", result.optimizations_applied);
```

## Custom Operations

Register custom logic-to-tensor mappings:

```rust
use tensorlogic_compiler::compile::{
    CustomOpRegistry, CustomOpMetadata, CustomOpData,
    ExtendedCompilerContext
};
use std::sync::Arc;

// Create registry
let mut registry = CustomOpRegistry::new();

// Define custom operation
let metadata = CustomOpMetadata {
    name: "custom_and".to_string(),
    description: "Custom AND with threshold".to_string(),
    expected_arity: Some(2),
    is_differentiable: true,
};

let handler = Arc::new(|expr, ctx, graph, data| {
    // Custom compilation logic
    let threshold = data.get_numeric("threshold").unwrap_or(0.5);
    // ... compile with threshold
    Ok(0) // Return tensor index
});

registry.register("custom_and", metadata, handler).unwrap();

// Use in extended context
let mut ext_ctx = ExtendedCompilerContext::new();
let data = CustomOpData::new().with_numeric("threshold", 0.7);
ext_ctx = ext_ctx.with_custom_data(data);
```

### Using Preset Custom Operations

```rust
use tensorlogic_compiler::compile::custom_ops::presets;

let (metadata, handler) = presets::create_soft_threshold_and(2.0);
registry.register("soft_threshold_and", metadata, handler).unwrap();

let (metadata, handler) = presets::create_weighted_or(0.6, 0.4);
registry.register("weighted_or", metadata, handler).unwrap();
```

## Debugging and Validation

### Pre-Compilation Validation

```rust
use tensorlogic_compiler::passes::{validate_expression_with_types, build_signature_registry};
use tensorlogic_adapters::SymbolTable;

// Build symbol table with predicate signatures
let mut symbol_table = SymbolTable::new();
// ... add predicate definitions

// Build signature registry
let registry = build_signature_registry(&symbol_table);

// Validate with types
let validation = validate_expression_with_types(&expr, &ctx, &registry);

for diagnostic in &validation.diagnostics {
    match diagnostic.level {
        DiagnosticLevel::Error => println!("ERROR: {}", diagnostic.message),
        DiagnosticLevel::Warning => println!("WARNING: {}", diagnostic.message),
        _ => {}
    }
}
```

### Scope Analysis

Check for unbound variables:

```rust
use tensorlogic_compiler::passes::analyze_scopes;

let scope_result = analyze_scopes(&expr);

if !scope_result.unbound_vars.is_empty() {
    println!("Unbound variables: {:?}", scope_result.unbound_vars);

    // Get suggestions
    let suggestions = suggest_quantifiers(&expr);
    println!("Suggestions: {}", suggestions);
}
```

### Debug Tracing

Enable compilation tracing:

```rust
use tensorlogic_compiler::debug::{CompilationTracer, CompilationTrace};

let mut tracer = CompilationTracer::new();
tracer.enable();

// Compile with tracing
// ... compilation code

let trace = tracer.into_trace();
println!("Compilation steps: {}", trace.steps.len());

for step in &trace.steps {
    println!("Step {}: {}", step.step_number, step.description);
}
```

### Graph Visualization

Export to DOT format for visualization:

```rust
use tensorlogic_ir::export_to_dot;

let dot = export_to_dot(&graph);
std::fs::write("graph.dot", dot).unwrap();

// Generate PNG with:
// dot -Tpng graph.dot -o graph.png
```

## Best Practices

### 1. Always Validate First

```rust
// GOOD: Validate before compiling
let validation = validate_expression(&expr, &ctx);
if validation.is_valid() {
    let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
}

// BAD: Compile without validation
let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
```

### 2. Use Appropriate Strategies

```rust
// For training: use soft differentiable
let config = CompilationConfig::soft_differentiable();

// For inference: use hard boolean or fuzzy
let config = CompilationConfig::hard_boolean();

// When unsure: use automatic selection
let recommendation = recommend_strategy(&expr, OptimizationGoal::Balanced);
let config = recommendation.config;
```

### 3. Apply Optimizations

```rust
// Optimize expressions before compilation
let (expr, _) = optimize_negations(&expr);
let (expr, _) = eliminate_common_subexpressions(&expr);

// Optimize graphs after compilation
let options = PostCompilationOptions::default();
post_compilation_passes(&mut graph, &ctx, options).unwrap();
```

### 4. Handle Errors Gracefully

```rust
use anyhow::Result;

fn safe_compile(expr: &TLExpr, ctx: &mut CompilerContext) -> Result<EinsumGraph> {
    // Validate first
    let validation = validate_expression(expr, ctx);
    if !validation.is_valid() {
        anyhow::bail!("Validation failed: {} errors", validation.error_count);
    }

    // Compile
    let graph = compile_to_einsum_with_context(expr, ctx)?;

    // Post-validate
    let options = PostCompilationOptions::default();
    post_compilation_passes(&mut graph, ctx, options)?;

    Ok(graph)
}
```

### 5. Reuse Contexts

```rust
// GOOD: Reuse context for related expressions
let mut ctx = CompilerContext::new();
ctx.add_domain("Person", 100);

let rule1 = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
let graph1 = compile_to_einsum_with_context(&rule1, &mut ctx).unwrap();

let rule2 = TLExpr::pred("likes", vec![Term::var("x"), Term::var("y")]);
let graph2 = compile_to_einsum_with_context(&rule2, &mut ctx).unwrap();

// BAD: Create new context each time (loses domain info)
let rule1 = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
let graph1 = compile_to_einsum(&rule1).unwrap();  // New context each time
```

### 6. Profile Complex Compilations

```rust
use std::time::Instant;

let start = Instant::now();
let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
let duration = start.elapsed();

println!("Compilation took: {:?}", duration);
println!("Graph size: {} nodes, {} tensors", graph.nodes.len(), graph.tensors.len());
```

## Common Patterns

### Pattern 1: Transitive Closure

```rust
// Rule: knows(x,y) ∧ knows(y,z) → knows(x,z)
let transitivity = TLExpr::forall(
    "x", "Person",
    TLExpr::forall(
        "y", "Person",
        TLExpr::forall(
            "z", "Person",
            TLExpr::imply(
                TLExpr::And(
                    Box::new(TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")])),
                    Box::new(TLExpr::pred("knows", vec![Term::var("y"), Term::var("z")])),
                ),
                TLExpr::pred("knows", vec![Term::var("x"), Term::var("z")]),
            ),
        ),
    ),
);
```

### Pattern 2: Existential Constraints

```rust
// Rule: ∃y. enrolled(x, y) ∧ required(y)
// "Students must be enrolled in at least one required course"
let constraint = TLExpr::exists(
    "y",
    "Course",
    TLExpr::And(
        Box::new(TLExpr::pred("enrolled", vec![Term::var("x"), Term::var("y")])),
        Box::new(TLExpr::pred("required", vec![Term::var("y")])),
    ),
);
```

### Pattern 3: Conditional Rules

```rust
// Rule: if score(x) > threshold then passed(x) else failed(x)
let conditional_rule = TLExpr::If {
    condition: Box::new(TLExpr::GreaterThan(
        Box::new(TLExpr::pred("score", vec![Term::var("x")])),
        Box::new(TLExpr::pred("threshold", vec![])),
    )),
    then_expr: Box::new(TLExpr::pred("passed", vec![Term::var("x")])),
    else_expr: Box::new(TLExpr::pred("failed", vec![Term::var("x")])),
};
```

## Troubleshooting

### Issue: "Unbound variable"

**Cause:** Variable used but not bound by a quantifier.

**Solution:**
```rust
// BAD: y is unbound
let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

// GOOD: Bind y with exists
let expr = TLExpr::exists(
    "y",
    "Person",
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
);
```

### Issue: "Arity mismatch"

**Cause:** Predicate called with wrong number of arguments.

**Solution:**
```rust
// Define predicate signature
let mut symbol_table = SymbolTable::new();
symbol_table.add_predicate("knows", 2, vec!["Person", "Person"]);

// Validate
let registry = build_signature_registry(&symbol_table);
validate_expression_with_types(&expr, &ctx, &registry);
```

### Issue: "Graph validation failed"

**Cause:** Invalid graph structure after compilation.

**Solution:**
```rust
// Enable post-compilation validation
let options = PostCompilationOptions {
    validate_graph_structure: true,
    validate_axes: true,
    validate_shapes: true,
    apply_optimizations: true,
    strict_mode: true,  // Fail on warnings too
};

post_compilation_passes(&mut graph, &ctx, options)?;
```

## Next Steps

- Explore the [API documentation](https://docs.rs/tensorlogic-compiler)
- Check out [examples](../examples/) for real-world use cases
- Read about [execution backends](../../tensorlogic-scirs-backend/README.md)
- Learn about [training with TensorLogic](../../tensorlogic-train/README.md)

---

**Last Updated:** 2025-11-04
**Version:** 0.1.0-alpha.1
