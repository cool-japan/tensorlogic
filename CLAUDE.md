# Tensorlogic Project Summary

Tensorlogic is a **logic-as-tensor planning layer** for the COOLJAPAN ecosystem. It compiles logical rules (predicates, quantifiers, implications) into **tensor equations (einsum graphs)** with a minimal DSL + IR, enabling neural/symbolic/probabilistic models within a unified tensor computation framework.

## Project Scope

- **DSL → IR → EinsumGraph**: Planning-layer compilation from logic expressions to symbolic tensor graphs
- **Engine-agnostic traits**: Abstract execution interface supporting multiple backends
- **SciRS2 backend**: Reference implementation for CPU/SIMD/GPU execution
- **Data governance**: OxiRS bridge for GraphQL/RDF*/SHACL integration and provenance tracking
- **Training scaffolds**: Loss composition, constraint violations, optimization schedules
- **Python bindings**: PyO3-based API for researchers and notebook workflows

## Architecture Overview

The workspace contains 11 specialized crates organized by responsibility:

### Planning Layer (Engine-Agnostic)
- **tensorlogic-ir**: AST and IR types (`Term`, `TLExpr`, `EinsumGraph`)
- **tensorlogic-compiler**: Logic → tensor mapping with static analysis
- **tensorlogic-infer**: Execution/autodiff traits (`TlExecutor`, `TlAutodiff`)
- **tensorlogic-adapters**: Symbol tables, axis metadata, domain masks

### Execution Layer (SciRS2-Powered)
- **tensorlogic-scirs-backend**: Runtime executor with CPU/SIMD/GPU features
- **tensorlogic-train**: Training loops, loss wiring, schedules, callbacks

### Integration Layer
- **tensorlogic-oxirs-bridge**: RDF*/GraphQL/SHACL → TL rules; provenance binding
- **tensorlogic-sklears-kernels**: Logic-derived similarity kernels for SkleaRS
- **tensorlogic-quantrs-hooks**: PGM/message-passing interop for QuantrS2
- **tensorlogic-trustformers**: Transformer-as-rules (self-attention/FFN as einsum)
- **tensorlogic-py**: PyO3 bindings with `abi3-py39` support

## Critical SciRS2 Integration Policy

**Tensorlogic MUST use SciRS2 as its complete scientific computing and tensor execution foundation.**

### Forbidden Direct Dependencies

❌ **NEVER** import these directly:
```rust
use ndarray::Array2;        // ❌ Wrong
use rand::thread_rng;       // ❌ Wrong
use num_complex::Complex64; // ❌ Wrong
```

✅ **ALWAYS** use SciRS2 equivalents:
```rust
use scirs2_core::ndarray::{Array, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::complex::Complex64;
use scirs2_core::array;  // array! macro
use scirs2_autograd::Variable;  // For training/autodiff
use scirs2_linalg::einsum;
```

### Layer-Specific Rules

**Planning Layer** (`tensorlogic-ir`, `tensorlogic-compiler`, `tensorlogic-infer`):
- MAY avoid heavy SciRS2 dependencies to remain lightweight
- Focus on symbolic representation, NOT runtime execution
- Use trait abstraction to allow alternative backends

**Execution Layer** (`tensorlogic-scirs-backend`, `tensorlogic-train`):
- MUST use SciRS2 for all tensor operations
- Backend: `scirs2-core`, `scirs2-linalg`
- Training: `scirs2-core`, `scirs2-autograd`, `scirs2-optimize`
- Features: `cpu` (default), `simd`, `gpu` (future)

**Integration Layer**:
- Follow respective ecosystem policies (OxiRS for bridge, SkleaRS for kernels, etc.)
- Use SciRS2 where tensor operations occur

See [SCIRS2_INTEGRATION_POLICY.md](SCIRS2_INTEGRATION_POLICY.md) for complete details.

## Development Workflow

### Essential Build Commands

```bash
# Build all crates
cargo build

# Build with specific backend features
cargo build -p tensorlogic-scirs-backend --features simd

# Run tests (using cargo-nextest)
cargo nextest run --no-fail-fast

# Run example
cargo run --example 00_minimal_rule

# Code quality checks
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo audit
```

### Quality Standards

**Code MUST compile without any warnings.** This is enforced via CI and should be verified locally before commits.

**File Size Limit**: Single source files should not exceed **2000 lines**. Use module decomposition or the SplitRS tool for refactoring large implementations.

**Naming Conventions**:
- Variables/functions: `snake_case`
- Types/traits: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`

**Workspace Dependencies**: Use `workspace = true` in Cargo.toml for shared dependencies.

## Logic-to-Tensor Mapping Defaults

The compiler uses these default mappings (configurable per use case):

| Logic Operation | Tensor Equivalent | Notes |
|----------------|-------------------|-------|
| `AND(a, b)` | Hadamard product `a * b` | Element-wise multiplication |
| `OR(a, b)` | `max(a, b)` | Or soft variant (configurable) |
| `NOT(a)` | `1 - a` | Or temperature-controlled sigmoid |
| `∃x. P(x)` | `sum(P, axis=x)` | Or `max` for hard quantification |
| `∀x. P(x)` | Dual of `∃`: `NOT(∃x. NOT(P(x)))` | Or product reduction |
| `a → b` | `ReLU(b - a)` | Or soft implication variants |

## Testing Philosophy

**Use `cargo nextest` exclusively** (not `cargo test`) for faster, parallelized test execution:

```bash
cargo nextest run --no-fail-fast
```

**Test Organization**:
- Unit tests: Within module files (`#[cfg(test)] mod tests { ... }`)
- Integration tests: In `tests/` directory
- Examples: In `examples/` directory (executable demonstrations)

**Coverage**: Aim for high coverage on compiler and inference traits; execution backend may have integration-focused tests.

## Example Workflow: Adding a New Logic Rule

1. **Define in IR** (`tensorlogic-ir/src/lib.rs`):
   ```rust
   pub enum TLExpr {
       // ... existing variants
       Custom { name: String, args: Vec<TLExpr> },
   }
   ```

2. **Implement compiler mapping** (`tensorlogic-compiler/src/lib.rs`):
   ```rust
   fn compile_custom(name: &str, args: &[TLExpr]) -> Result<EinsumNode> {
       // Map to einsum specification
   }
   ```

3. **Add execution support** (if needed in `tensorlogic-scirs-backend`):
   ```rust
   impl TlExecutor for Scirs2Exec {
       // Implement runtime behavior using SciRS2
   }
   ```

4. **Write tests**:
   ```bash
   cargo nextest run -p tensorlogic-compiler
   ```

5. **Add example** (e.g., `examples/02_custom_rule/main.rs`)

6. **Document** in README.md and TODO.md roadmap

## Common Development Tasks

### Adding a New Crate Dependency

1. Add to **workspace** `Cargo.toml` if shared:
   ```toml
   [workspace.dependencies]
   new-crate = "1.0"
   ```

2. Reference in crate's `Cargo.toml`:
   ```toml
   [dependencies]
   new-crate.workspace = true
   ```

3. **Verify** it doesn't violate SciRS2 integration policy

### Implementing a New Backend

1. Implement `TlExecutor` trait from `tensorlogic-infer`
2. Optionally implement `TlAutodiff` for training support
3. Add feature flags in backend crate's `Cargo.toml`
4. Update README.md with backend selection instructions

### Extending OxiRS Bridge

1. Define schema-to-symbols mapping in `tensorlogic-oxirs-bridge`
2. Implement SHACL → `TLExpr` lowering
3. Add provenance tracking (ruleID → nodeID → tensorID)
4. Write integration tests with sample RDF* data

## CI/CD Pipeline

The `.github/workflows/ci.yml` enforces:
- ✅ `cargo fmt --all -- --check`
- ✅ `cargo clippy --all -- -D warnings`
- ✅ `cargo build --no-default-features --features <matrix>`
- ✅ `cargo nextest run --no-fail-fast`

Feature matrix tests `cpu` and `simd` backend modes.

## Python Bindings Usage

The `tensorlogic-py` crate provides PyO3 bindings:

```python
import tensorlogic_py as tl

# Compile logic rules to execution plan
plan = tl.compile(rules)

# Execute with backend
result = tl.run(plan, executor="scirs2-cpu")

# Retrieve provenance
provenance = tl.get_provenance(result)
```

Build with:
```bash
maturin develop -m crates/tensorlogic-py/Cargo.toml
```

## Roadmap Alignment

See [TODO.md](TODO.md) for the 8-phase development plan:
1. Phase 0: Repo hygiene (CI, docs skeleton)
2. Phase 1: Minimal IR & compiler
3. Phase 2: Engine traits & dummy executor
4. Phase 3: SciRS2 backend
5. Phase 4: OxiRS bridge
6. Phase 5: Interop crates (SkleaRS, QuantrS2, TrustformeRS)
7. Phase 6: Training scaffolds
8. Phase 7: Python bindings
9. Phase 8: Validation & scale (property tests, fuzzing, GPU)

## References

- **Tensor Logic Paper**: https://arxiv.org/abs/2510.12269
- **SciRS2**: https://github.com/cool-japan/scirs
- **OxiRS**: https://github.com/cool-japan/oxirs
- **QuantRS2**: https://github.com/cool-japan/quantrs
- **SkleaRS**: https://github.com/cool-japan/sklears

## Quick Start

```bash
# Clone and build
git clone https://github.com/cool-japan/tensorlogic.git
cd tensorlogic
cargo build

# Run minimal example
cargo run --example 00_minimal_rule

# Run tests
cargo nextest run

# Check code quality
cargo fmt --all -- --check
cargo clippy --all -- -D warnings
```

---

**Document Status**: Active
**Maintained By**: COOLJAPAN ecosystem
