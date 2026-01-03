# TensorLogic Library Mode Examples

This directory contains examples demonstrating how to use `tensorlogic-cli` as a Rust library instead of a command-line tool.

## Overview

The `tensorlogic-cli` crate can be used as a library to programmatically:
- Parse logical expressions
- Compile expressions to tensor graphs
- Optimize graphs for performance
- Execute graphs with different backends
- Analyze graph complexity
- Define and expand macros
- Convert between different formats
- Benchmark compilation performance

## Examples

### Basic Usage: `library_basic.rs`

**What it demonstrates:**
- Parsing logical expressions from strings
- Creating and configuring compiler contexts
- Compiling expressions to einsum graphs
- Analyzing graph complexity metrics

**Run with:**
```bash
cargo run --example library_basic
```

**Key concepts:**
```rust
use tensorlogic_cli::{analysis, parser, CompilationContext};

// Parse expression
let expr = parser::parse_expression("AND(knows(Alice, Bob), knows(Bob, Charlie))")?;

// Set up context
let mut context = CompilationContext::new();
context.add_domain("Person", 100);

// Compile
let graph = compile_to_einsum_with_context(&expr, &mut context)?;

// Analyze
let metrics = analysis::analyze_graph(&graph);
```

### Macro System: `library_macros.rs`

**What it demonstrates:**
- Using built-in macros (transitive, symmetric, reflexive, etc.)
- Defining custom macros programmatically
- Expanding macros in expressions
- Composing macros (using macros within macro definitions)

**Run with:**
```bash
cargo run --example library_macros
```

**Key concepts:**
```rust
use tensorlogic_cli::macros::{MacroDef, MacroRegistry};

// Create registry with built-ins
let mut macros = MacroRegistry::with_builtins();

// Define custom macro
let macro_def = MacroDef::new(
    "friendOfFriend".to_string(),
    vec!["x".to_string(), "z".to_string()],
    "EXISTS y. (friend(x, y) AND friend(y, z))".to_string(),
);
macros.define(macro_def)?;

// Expand macros
let expanded = macros.expand_all("friendOfFriend(Alice, Bob)")?;
```

### Advanced Features: `library_advanced.rs`

**What it demonstrates:**
- Parsing complex nested expressions
- Graph optimization with different levels
- Benchmarking compilation performance
- Analyzing optimization impact
- Working with execution backends

**Run with:**
```bash
cargo run --example library_advanced
```

**Key concepts:**
```rust
use tensorlogic_cli::{benchmark, optimize, executor};

// Optimize graph
let opt_config = optimize::OptimizationConfig {
    level: optimize::OptimizationLevel::Aggressive,
    verbose: true,
};
let optimized = optimize::optimize_einsum_graph(&graph, &opt_config)?;

// Benchmark
let mut benchmarker = benchmark::Benchmarker::new(bench_config);
benchmarker.benchmark_compilation(&expr, &mut context)?;

// List backends
for backend in executor::Backend::available_backends() {
    println!("{}: {}", backend.name(), backend.is_available());
}
```

### Format Conversion: `library_conversion.rs`

**What it demonstrates:**
- Pretty-printing expressions
- Converting to/from JSON and YAML
- Round-trip serialization
- Expression normalization
- Formatting different expression types

**Run with:**
```bash
cargo run --example library_conversion
```

**Key concepts:**
```rust
use tensorlogic_cli::conversion;

// Format expression
let pretty = conversion::format_expression(&expr, true);

// Serialize to JSON
let json = serde_json::to_string_pretty(&expr)?;

// Deserialize from JSON
let expr: TLExpr = serde_json::from_str(&json)?;
```

## Integration Patterns

### Pattern 1: One-shot Compilation

For simple, one-time compilation:

```rust
use tensorlogic_cli::{parser, CompilationContext};
use tensorlogic_compiler::compile_to_einsum_with_context;

fn compile_expression(input: &str) -> anyhow::Result<EinsumGraph> {
    let expr = parser::parse_expression(input)?;
    let mut context = CompilationContext::new();
    context.add_domain("D", 100);
    compile_to_einsum_with_context(&expr, &mut context)
}
```

### Pattern 2: Reusable Context

For multiple compilations with shared context:

```rust
use tensorlogic_cli::CompilationContext;

struct LogicCompiler {
    context: CompilationContext,
}

impl LogicCompiler {
    fn new() -> Self {
        let mut context = CompilationContext::new();
        // Configure context...
        Self { context }
    }

    fn compile(&mut self, input: &str) -> anyhow::Result<EinsumGraph> {
        let expr = parser::parse_expression(input)?;
        compile_to_einsum_with_context(&expr, &mut self.context)
    }
}
```

### Pattern 3: With Macro Expansion

Integrate macros into your pipeline:

```rust
use tensorlogic_cli::macros::MacroRegistry;

struct MacroAwareCompiler {
    context: CompilationContext,
    macros: MacroRegistry,
}

impl MacroAwareCompiler {
    fn compile(&mut self, input: &str) -> anyhow::Result<EinsumGraph> {
        // Expand macros first
        let expanded = self.macros.expand_all(input)?;

        // Parse and compile
        let expr = parser::parse_expression(&expanded)?;
        compile_to_einsum_with_context(&expr, &mut self.context)
    }
}
```

### Pattern 4: Optimizing Pipeline

Full pipeline with optimization:

```rust
use tensorlogic_cli::{optimize, analysis};

fn compile_and_optimize(input: &str) -> anyhow::Result<(EinsumGraph, GraphMetrics)> {
    // Compile
    let expr = parser::parse_expression(input)?;
    let mut context = CompilationContext::new();
    let graph = compile_to_einsum_with_context(&expr, &mut context)?;

    // Optimize
    let opt_config = optimize::OptimizationConfig {
        level: optimize::OptimizationLevel::Standard,
        verbose: false,
    };
    let optimized = optimize::optimize_einsum_graph(&graph, &opt_config)?;

    // Analyze
    let metrics = analysis::analyze_graph(&optimized);

    Ok((optimized, metrics))
}
```

## Performance Considerations

1. **Compilation Caching**: For repeated compilations, use `CompilationCache`:
   ```rust
   use tensorlogic_compiler::CompilationCache;

   let cache = CompilationCache::new(100);
   let graph = cache.get_or_compile(&expr, &mut context, |e, ctx| {
       compile_to_einsum_with_context(e, ctx)
   })?;
   ```

2. **Macro Expansion**: Cache expanded expressions when possible to avoid redundant expansion.

3. **Context Reuse**: Reuse `CompilerContext` instances across compilations when domains don't change.

4. **Optimization**: Profile before optimizing. Use `analysis::analyze_graph()` to determine if optimization is beneficial.

## Testing

The library mode makes testing easier:

```rust
#[cfg(test)]
mod tests {
    use tensorlogic_cli::parser;

    #[test]
    fn test_parse_simple_predicate() {
        let expr = parser::parse_expression("knows(Alice, Bob)").unwrap();
        assert!(matches!(expr, TLExpr::Pred { .. }));
    }

    #[test]
    fn test_macro_expansion() {
        use tensorlogic_cli::macros::MacroRegistry;

        let macros = MacroRegistry::with_builtins();
        let expanded = macros.expand_all("transitive(knows, Alice, Bob)").unwrap();
        assert!(expanded.contains("EXISTS"));
    }
}
```

## Error Handling

All library functions return `anyhow::Result<T>`, making error handling straightforward:

```rust
use anyhow::Context;

fn process_logic(input: &str) -> anyhow::Result<GraphMetrics> {
    let expr = parser::parse_expression(input)
        .context("Failed to parse expression")?;

    let mut context = CompilationContext::new();
    let graph = compile_to_einsum_with_context(&expr, &mut context)
        .context("Compilation failed")?;

    Ok(analysis::analyze_graph(&graph))
}
```

## Further Reading

- **Main Documentation**: See [README.md](../README.md) for overall project documentation
- **CLI Usage**: See [COOKBOOK.md](../docs/COOKBOOK.md) for CLI usage patterns
- **API Documentation**: Run `cargo doc --open` to view detailed API documentation
- **Macro System**: See [macros.rs](../src/macros.rs) for macro system details

## Contributing Examples

If you develop useful patterns using the library mode, consider contributing them as examples! See the main repository for contribution guidelines.
