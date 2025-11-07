# SciRS2 Integration Policy for Tensorlogic

## Core Policy Statement

**Tensorlogic MUST use SciRS2 as its complete scientific computing and tensor execution foundation.**

This policy ensures consistent, high-performance tensor operations across logic-as-tensor planning, compilation, and execution workflows. All scientific computing primitives (arrays, einsum, autodiff, optimization) MUST be accessed through SciRS2 rather than fragmented direct dependencies.

## Architectural Philosophy

Tensorlogic implements a clean separation between **planning** (IR/compiler) and **execution** (backend):

- **Planning Layer** (`tensorlogic-ir`, `tensorlogic-compiler`, `tensorlogic-infer`): Engine-agnostic traits and symbolic graph construction
- **Execution Layer** (`tensorlogic-scirs-backend`): SciRS2-powered runtime with CPU/SIMD/GPU support
- **Integration Layer** (`tensorlogic-oxirs-bridge`, `tensorlogic-train`): Data governance and training scaffolds

This architecture allows alternative backends while maintaining SciRS2 as the reference implementation and primary execution engine.

## Forbidden Direct Dependencies

The following crates MUST NOT be imported directly. Use SciRS2 equivalents instead:

### Core Scientific Computing
- ❌ `ndarray` → ✅ `scirs2_core::ndarray`
- ❌ `rand` / `rand_distr` → ✅ `scirs2_core::random`
- ❌ `num-complex` → ✅ `scirs2_core::complex`

### Parallelization and Hardware Acceleration
- ❌ `rayon` → ✅ `scirs2_core` (with `parallel` feature)
- ❌ `rayon-core` → ✅ `scirs2_core` (with `parallel` feature)
- ❌ Direct SIMD intrinsics → ✅ `scirs2_core` (with `simd` feature)
- ❌ Direct GPU libraries (CUDA, OpenCL, etc.) → ✅ `scirs2_core` (with `gpu` feature when available)

**Rationale**: SciRS2-core provides unified, optimized parallel and hardware-accelerated operations. Direct use of rayon or hardware-specific libraries would:
1. Bypass SciRS2's unified performance optimization layer
2. Create fragmented parallelization strategies
3. Complicate cross-platform compatibility
4. Prevent centralized performance tuning

**Exception**: The planning layer (`tensorlogic-ir`, `tensorlogic-compiler`, `tensorlogic-infer`) MAY avoid heavy SciRS2 dependencies to remain lightweight and engine-agnostic. Only the backend and training crates require full SciRS2 integration.

## Essential SciRS2 Crates

### Always Required

**`scirs2-core`**
- Foundation for all tensor operations
- Provides `array!` macro for convenient array construction
- **Features**:
  - `random`: Random number generation (replaces `rand`/`rand_distr`)
  - `parallel`: Multi-threaded operations via rayon (replaces direct `rayon` dependency)
  - `simd`: SIMD-accelerated operations (replaces manual SIMD intrinsics)
  - `gpu`: GPU execution support (future, replaces CUDA/OpenCL)
- Used by: `tensorlogic-scirs-backend`, `tensorlogic-train`
- **Critical**: ALL parallelization, SIMD, and GPU features MUST come from scirs2-core

### Highly Likely Required

**`scirs2-linalg`**
- Einsum implementation
- Matrix operations for tensor manipulations
- Used by: `tensorlogic-scirs-backend`

**`scirs2-autograd`**
- Automatic differentiation for training (Phase 6)
- Gradient computation via `Variable` type
- Used by: `tensorlogic-train`

**`scirs2-optimize`**
- Training loop optimizers (Adam, SGD, etc.)
- Loss function composition
- Used by: `tensorlogic-train`

**`scirs2-metrics`**
- Performance monitoring
- Inference throughput tracking
- Used by: `tensorlogic-scirs-backend`

**`scirs2-neural`**
- Transformer-as-rules integration (future)
- Self-attention einsum patterns
- Used by: `tensorlogic-trustformers`

**`scirs2-stats`**
- Probabilistic logic operations
- Soft quantifiers (∃, ∀)
- Used by: `tensorlogic-compiler`

### Future Candidates

Evaluated based on roadmap requirements:

- `scirs2-sparse`: Sparse tensor support for large knowledge bases
- `scirs2-graph`: Graph algorithms for rule dependency analysis
- `scirs2-fft`: Frequency-domain logic operations (research)
- `scirs2-cluster`: Distributed execution across nodes
- `scirs2-signal`: Real-time stream reasoning

### Not Applicable

- `scirs2-text`: NLP handled by other COOLJAPAN crates
- `scirs2-vision`: Computer vision not in scope
- `scirs2-datasets`: Data loading handled by `tensorlogic-adapters`

## Integration Patterns

### Correct Import Style

```rust
// ✅ Correct: Unified imports through SciRS2
use scirs2_core::ndarray::{Array, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_autograd::{array, Variable};
use scirs2_linalg::einsum;
```

### Incorrect Import Style

```rust
// ❌ Wrong: Direct imports violate policy
use ndarray::Array2;
use rand::thread_rng;
```

### Backend Implementation Pattern

```rust
use scirs2_core::ndarray::Array;
use scirs2_linalg::einsum;
use tensorlogic_infer::{TlExecutor, ElemOp};

pub struct Scirs2Backend {
    // SciRS2 session/device handles
}

impl TlExecutor for Scirs2Backend {
    type Tensor = Array<f32, ndarray::IxDyn>;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Self::Tensor {
        einsum(spec, inputs)
    }
}
```

## Feature Flags and Hardware Acceleration

The `tensorlogic-scirs-backend` crate provides execution mode selection through feature flags. **All hardware acceleration and parallelization features delegate to scirs2-core**:

### Feature Mapping

| TensorLogic Feature | SciRS2-Core Feature | Purpose |
|---------------------|---------------------|---------|
| `cpu` (default) | Default scirs2-core | Single-threaded CPU execution |
| `simd` | `simd` | SIMD-accelerated operations (AVX, NEON, etc.) |
| `gpu` (future) | `gpu` | GPU execution (CUDA, OpenCL, Vulkan) |
| `parallel` (implicit) | `parallel` | Multi-threaded via rayon |

### Usage Example

```toml
[dependencies]
tensorlogic-scirs-backend = { version = "0.1", features = ["simd"] }
```

This enables SIMD acceleration through scirs2-core's unified interface, **not** through direct SIMD intrinsics or vendor-specific libraries.

### Critical Policy

**NEVER** add direct dependencies on:
- `rayon` or `rayon-core` for parallelization
- `packed_simd`, `std::arch`, or manual intrinsics for SIMD
- `cuda`, `opencl`, `vulkan`, or direct GPU libraries

**ALWAYS** use scirs2-core's feature flags to access these capabilities. This ensures:
1. Consistent performance characteristics across the codebase
2. Unified optimization strategies
3. Cross-platform compatibility
4. Centralized maintenance of hardware-specific code

## Enforcement Mechanisms

### CI/CD Pipeline Checks

```yaml
# .github/workflows/ci.yml
- name: Check for forbidden direct dependencies
  run: |
    # Core scientific computing
    ! grep -r "use ndarray::" crates/tensorlogic-scirs-backend crates/tensorlogic-train
    ! grep -r "use rand::" crates/tensorlogic-scirs-backend crates/tensorlogic-train

    # Parallelization and hardware acceleration
    ! grep -r "use rayon::" crates/tensorlogic-scirs-backend crates/tensorlogic-train
    ! grep -r "rayon = " crates/tensorlogic-scirs-backend/Cargo.toml crates/tensorlogic-train/Cargo.toml
    ! grep -r "packed_simd\|std::arch" crates/tensorlogic-scirs-backend crates/tensorlogic-train

    # GPU libraries
    ! grep -r "cuda\|opencl\|vulkan" crates/tensorlogic-scirs-backend/Cargo.toml crates/tensorlogic-train/Cargo.toml
```

### Code Review Checklist

#### Core Scientific Computing
- [ ] All array operations use `scirs2_core::ndarray`
- [ ] Random number generation uses `scirs2_core::random`
- [ ] Complex numbers use `scirs2_core::complex`
- [ ] Autodiff uses `scirs2_autograd::Variable`
- [ ] Einsum operations use `scirs2_linalg::einsum`

#### Parallelization and Hardware Acceleration
- [ ] No direct `rayon` or `rayon-core` dependencies
- [ ] Parallel operations use `scirs2_core` with `parallel` feature
- [ ] No manual SIMD intrinsics (`std::arch`, `packed_simd`)
- [ ] SIMD operations use `scirs2_core` with `simd` feature
- [ ] No direct GPU libraries (CUDA, OpenCL, Vulkan)
- [ ] GPU operations (future) use `scirs2_core` with `gpu` feature

#### Documentation
- [ ] New dependencies documented in this policy
- [ ] Feature flag usage documented
- [ ] Performance implications noted

### Quarterly Dependency Audit

Review and prune unused SciRS2 crates, evaluate new candidates from roadmap, and verify compliance across all backend/training code.

## Migration from Legacy Dependencies

If migrating existing code:

1. **Identify**: Search for direct `ndarray`, `rand`, `num-complex` imports
2. **Replace**: Switch to `scirs2_core` equivalents
3. **Test**: Verify numerical equivalence with existing tests
4. **Document**: Note any API differences in migration comments

## Exemptions and Justification

Planning-layer crates (`tensorlogic-ir`, `tensorlogic-compiler`, `tensorlogic-infer`) are exempt from heavy SciRS2 dependencies to maintain:

- **Lightweight compilation**: Minimal dependencies for IR/compiler
- **Engine portability**: Traits allow non-SciRS2 backends
- **Fast build times**: Avoid full SciRS2 rebuild during planning-only changes

All execution and training code MUST use SciRS2.

## Version Alignment

Tensorlogic targets:
- **SciRS2**: Latest stable (2.x series)
- **Rust**: MSRV 1.77+ (aligned with workspace)

## References

- [SciRS2 Documentation](https://github.com/cool-japan/scirs)
- [Tensor Logic Paper](https://arxiv.org/abs/2510.12269)
- Related policies: [QuantRS2](https://github.com/cool-japan/quantrs/blob/master/SCIRS2_INTEGRATION_POLICY.md), [OxiRS](https://github.com/cool-japan/oxirs/blob/master/SCIRS2_INTEGRATION_POLICY.md)

---

**Document Status**: Active
**Review Cycle**: Quarterly
**Major Updates**:
 Added explicit policy for rayon, SIMD, GPU, and parallelization (must use scirs2-core)
 Initial version with core scientific computing dependencies
