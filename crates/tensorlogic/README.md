# TensorLogic Meta Crate

**Unified access to all TensorLogic components**

[![Crates.io](https://img.shields.io/crates/v/tensorlogic.svg)](https://crates.io/crates/tensorlogic)
[![Documentation](https://docs.rs/tensorlogic/badge.svg)](https://docs.rs/tensorlogic)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)

This is the top-level umbrella crate that re-exports all TensorLogic components for convenient access. Instead of importing individual crates, you can use this meta crate to access the entire TensorLogic ecosystem.

## Overview

TensorLogic compiles logical rules (predicates, quantifiers, implications) into **tensor equations (einsum graphs)** with a minimal DSL + IR, enabling neural/symbolic/probabilistic models within a unified tensor computation framework.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tensorlogic = "0.1.0-alpha.1"
```

### Basic Usage

```rust
use tensorlogic::prelude::*;

// Define logical expressions
let x = Term::var("x");
let y = Term::var("y");
let knows = TLExpr::pred("knows", vec![x.clone(), y.clone()]);

// Compile to tensor graph
let graph = compile_to_einsum(&knows)?;

// Execute with backend
let mut executor = Scirs2Exec::new();
let result = executor.forward(&graph)?;
```

## Architecture

The meta crate provides organized access to three layers:

### Planning Layer (Engine-Agnostic)

```rust
use tensorlogic::ir::*;        // AST and IR types
use tensorlogic::compiler::*;  // Logic → tensor compilation
use tensorlogic::infer::*;     // Execution traits
use tensorlogic::adapters::*;  // Symbol tables, domains
```

**Components:**
- `tensorlogic::ir` - Core IR types (`Term`, `TLExpr`, `EinsumGraph`)
- `tensorlogic::compiler` - Logic-to-tensor mapping with static analysis
- `tensorlogic::infer` - Execution/autodiff traits (`TlExecutor`, `TlAutodiff`)
- `tensorlogic::adapters` - Symbol tables, axis metadata, domain masks

### Execution Layer (SciRS2-Powered)

```rust
use tensorlogic::scirs_backend::*;  // SciRS2 runtime executor
use tensorlogic::train::*;           // Training infrastructure
```

**Components:**
- `tensorlogic::scirs_backend` - Runtime executor with CPU/SIMD/GPU features
- `tensorlogic::train` - Training loops, loss wiring, schedules, callbacks

### Integration Layer

```rust
use tensorlogic::oxirs_bridge::*;      // RDF*/SHACL integration
use tensorlogic::sklears_kernels::*;   // ML kernels
use tensorlogic::quantrs_hooks::*;     // PGM integration
use tensorlogic::trustformers::*;      // Transformer components
```

**Components:**
- `tensorlogic::oxirs_bridge` - RDF*/GraphQL/SHACL → TL rules; provenance binding
- `tensorlogic::sklears_kernels` - Logic-derived similarity kernels for SkleaRS
- `tensorlogic::quantrs_hooks` - PGM/message-passing interop for QuantrS2
- `tensorlogic::trustformers` - Transformer-as-rules (attention/FFN as einsum)

## Prelude Module

For convenience, commonly used types are available through the prelude:

```rust
use tensorlogic::prelude::*;
```

This imports:
- Core types: `Term`, `TLExpr`, `EinsumGraph`, `EinsumNode`
- Compilation: `compile_to_einsum`, `CompilerContext`, `CompilationConfig`
- Execution: `TlExecutor`, `TlAutodiff`, `Scirs2Exec`
- Errors: `IrError`, `CompilerError`

## Examples

This crate includes 5 comprehensive examples demonstrating all features:

```bash
# Basic predicate and compilation
cargo run --example 00_minimal_rule

# Existential quantifier with reduction
cargo run --example 01_exists_reduce

# Full execution with SciRS2 backend
cargo run --example 02_scirs2_execution

# OxiRS bridge with RDF* data
cargo run --example 03_rdf_integration

# Comparing 6 compilation strategy presets
cargo run --example 04_compilation_strategies
```

## Features

### Compilation Strategies

TensorLogic supports 6 preset compilation strategies:

1. **soft_differentiable** - Neural network training (smooth gradients)
2. **hard_boolean** - Discrete Boolean logic (exact semantics)
3. **fuzzy_godel** - Gödel fuzzy logic (min/max operations)
4. **fuzzy_product** - Product fuzzy logic (probabilistic)
5. **fuzzy_lukasiewicz** - Łukasiewicz fuzzy logic (bounded)
6. **probabilistic** - Probabilistic interpretation

```rust
use tensorlogic::compiler::CompilationConfig;

let config = CompilationConfig::soft_differentiable();
let graph = compile_with_config(&expr, config)?;
```

### Logic-to-Tensor Mapping

Default mappings (configurable per use case):

| Logic Operation | Tensor Equivalent | Notes |
|----------------|-------------------|-------|
| `AND(a, b)` | `a * b` (Hadamard) | Element-wise multiplication |
| `OR(a, b)` | `max(a, b)` | Or soft variant |
| `NOT(a)` | `1 - a` | Or temperature-controlled |
| `∃x. P(x)` | `sum(P, axis=x)` | Or `max` for hard |
| `∀x. P(x)` | Dual of ∃ | Or product reduction |
| `a → b` | `max(1-a, b)` | Or ReLU variant |

## Feature Flags

Control which components are included:

```toml
[dependencies]
tensorlogic = { version = "0.1.0-alpha.1", features = ["simd"] }
```

Available features:
- `simd` - Enable SIMD acceleration in SciRS2 backend (2-4x speedup)
- `gpu` - Enable GPU support (future)

## Documentation

- **Project Guide**: [CLAUDE.md](../../CLAUDE.md)
- **API Reference**: [docs.rs/tensorlogic](https://docs.rs/tensorlogic)
- **Main README**: [README.md](../../README.md)
- **Tutorial**: Check individual crate READMEs for detailed guides

### Component Documentation

Each component has comprehensive documentation:

- [tensorlogic-ir](../tensorlogic-ir/README.md) - IR and AST types
- [tensorlogic-compiler](../tensorlogic-compiler/README.md) - Compilation
- [tensorlogic-infer](../tensorlogic-infer/README.md) - Execution traits
- [tensorlogic-scirs-backend](../tensorlogic-scirs-backend/README.md) - SciRS2 backend
- [tensorlogic-train](../tensorlogic-train/README.md) - Training
- [tensorlogic-adapters](../tensorlogic-adapters/README.md) - Symbol tables
- [tensorlogic-oxirs-bridge](../tensorlogic-oxirs-bridge/README.md) - RDF* integration
- [tensorlogic-sklears-kernels](../tensorlogic-sklears-kernels/README.md) - ML kernels
- [tensorlogic-quantrs-hooks](../tensorlogic-quantrs-hooks/README.md) - PGM integration
- [tensorlogic-trustformers](../tensorlogic-trustformers/README.md) - Transformers

## Development

### Building

```bash
# Build the meta crate
cargo build -p tensorlogic

# Build with SIMD support
cargo build -p tensorlogic --features simd

# Run tests
cargo nextest run -p tensorlogic

# Run examples
cargo run -p tensorlogic --example 00_minimal_rule
```

### Testing

The meta crate includes all component tests:

```bash
# Run all tests
cargo test -p tensorlogic --all-features

# Run with nextest (faster)
cargo nextest run -p tensorlogic --all-features
```

## Version Compatibility

This meta crate version `0.1.0-alpha.1` includes:

| Component | Version | Status |
|-----------|---------|--------|
| tensorlogic-ir | 0.1.0-alpha.1 | ✅ Production Ready |
| tensorlogic-compiler | 0.1.0-alpha.1 | ✅ Production Ready |
| tensorlogic-infer | 0.1.0-alpha.1 | ✅ Production Ready |
| tensorlogic-scirs-backend | 0.1.0-alpha.1 | ✅ Production Ready |
| tensorlogic-train | 0.1.0-alpha.1 | ✅ Complete |
| tensorlogic-adapters | 0.1.0-alpha.1 | ✅ Complete |
| tensorlogic-oxirs-bridge | 0.1.0-alpha.1 | ✅ Complete |
| tensorlogic-sklears-kernels | 0.1.0-alpha.1 | ✅ Core Features |
| tensorlogic-quantrs-hooks | 0.1.0-alpha.1 | ✅ Core Features |
| tensorlogic-trustformers | 0.1.0-alpha.1 | ✅ Complete |

All components are synchronized to version `0.1.0-alpha.1`.

## Migration from Individual Crates

If you were using individual crates:

**Before:**
```toml
[dependencies]
tensorlogic-ir = "0.1.0-alpha.1"
tensorlogic-compiler = "0.1.0-alpha.1"
tensorlogic-scirs-backend = "0.1.0-alpha.1"
```

**After:**
```toml
[dependencies]
tensorlogic = "0.1.0-alpha.1"
```

Your code remains the same, just update imports:

**Before:**
```rust
use tensorlogic_ir::{Term, TLExpr};
use tensorlogic_compiler::compile_to_einsum;
```

**After:**
```rust
use tensorlogic::ir::{Term, TLExpr};
use tensorlogic::compiler::compile_to_einsum;
// Or use prelude for common types
use tensorlogic::prelude::*;
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

Licensed under Apache 2.0 License. See [LICENSE](../../LICENSE) for details.

## References

- **Tensor Logic Paper**: https://arxiv.org/abs/2510.12269
- **Project Repository**: https://github.com/cool-japan/tensorlogic
- **Documentation**: https://docs.rs/tensorlogic

---

**Part of the COOLJAPAN Ecosystem**

For questions and support, please open an issue on [GitHub](https://github.com/cool-japan/tensorlogic/issues).
