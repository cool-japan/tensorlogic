# Changelog

All notable changes to TensorLogic will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-08-30

### Added

- **GPU sparse shape validation** (tensorlogic-oxicuda-sparse): new `check_spmv_shapes` / `check_spmm_shapes` helpers validate operand buffer lengths against the CSR matrix dimensions before dispatching to device kernels, returning `SparseError::ShapeMismatch` instead of letting an undersized buffer be indexed out of bounds on the device.
- `count_for_schema` (tensorlogic-adapters): row-count helper for the SQLite-backed schema store.

### Changed

- **SQLite backend migrated from rusqlite to OxiSQL** (tensorlogic-adapters): the `sqlite` feature now depends on `oxisql-core` + `oxisql-sqlite-compat` instead of `rusqlite` (which bundled a C SQLite build), removing the crate's last C dependency for full Pure Rust compliance. SQL placeholder syntax changed from `?` to numbered `$1, $2, ...` to match OxiSQL's parameter binding; `initialize_schema` no longer requires `&mut self`; reads/writes now run through a dedicated Tokio runtime with errors mapped via a new `map_oxi` helper.
- **RDF/Turtle stack migrated from oxrdf/oxttl to OxiXML** (tensorlogic-oxirs-bridge): `oxrdf` / `oxttl` are now workspace aliases for the COOLJAPAN `oxixml-model` / `oxixml-turtle` crates (0.1.2) instead of the upstream Oxigraph crates; the now-dead `oxirs-ttl` dependency was dropped. Call sites in `schema/inference.rs`, `schema/metadata.rs`, `schema/nquads.rs`, `schema/ntriples.rs`, and `shacl/mod.rs` were updated for OxiXML's borrowed-reference iterator API.
- Dependency updates: `scirs2-{core,linalg,autograd,optimize,sparse}` 0.5.0 → 0.6.5; `oxicuda-{backend,blas,driver,fft,memory,rand,solver,sparse}` 0.1.8 → 0.5.5; `torsh-{core,tensor}` 0.1.2 → 0.2.0; `sklears-{core,kernel-approximation}` 0.1.1 → 0.2.0; `quantrs2-{core,circuit,sim}` 0.2.0 → 0.2.1; `oxirs-{core,gql}` 0.3.1 → 0.4.1; `oxicode` → 0.2.6; `oxiarc-deflate` 0.3.3 → 0.4.1; `tabled` 0.20 → 0.21; new `oxisql-core` / `oxisql-sqlite-compat` workspace dependencies at 0.4.1.

### Fixed

- GPU sparse SpMV/SpMM (tensorlogic-oxicuda-sparse) previously trusted caller-supplied buffer lengths when dispatching to device kernels; mismatched `x`/`y` (SpMV) or `b`/`c` (SpMM) lengths could index device memory out of bounds. Both entry points now validate shapes up front and return an error on mismatch instead.

## [0.1.1] - 2026-06-09

### Added — Round 7 (2026-06-02)

- **Exact LTL Next (X) operator** (tensorlogic-compiler): `compile_next` now emits a `temporal_next:<axis>` unary node dispatched via `tensorlogic_scirs_backend::temporal_ops::shift_next`; previously returned an unimplemented error.
- **Exact LTL Until (U) operator** (tensorlogic-compiler): `compile_until` now emits a `temporal_until:<tag>:<axis>` binary node; previously returned an unimplemented error.
- **Exact LTL Release (R), WeakUntil (W), StrongRelease (M) operators** (tensorlogic-compiler + tensorlogic-scirs-backend): replaced three mathematically incorrect single-step approximations with exact finite-trace backward-scan recurrences. Unified `temporal_binary_scan` / `temporal_binary_scan_vjp` generalise the `until_scan` from Round 6. `OUTER`/`INNER` parametrised over `TemporalBinaryForm` × `UntilSemantics`; `boundary_val` ∈ {0.0, 1.0} distinguishes strong/weak variants.
- **Temporal ops backend** (tensorlogic-scirs-backend): new `temporal_ops.rs` module exposing `UntilSemantics` (MaxMin / ProbSumProduct), `TemporalBinaryForm` (Until / WeakUntil / Release / StrongRelease), `shift_next`, `shift_prev`, `temporal_binary_scan`, `temporal_binary_scan_vjp`. +16 tests (10 unit + 6 integration).
- **OxiCUDA solver f64 + PCG + Thomas algorithm** (tensorlogic-oxicuda-solver): generic `solve_lu_f64`, `solve_cholesky_f64`, `solve_qr_lstsq_f64`, `cg_solve_f64`; preconditioned CG (`pcg_solve` / `pcg_solve_f64`) with `Precond::Jacobi` and `Precond::IncompleteCholesky`; `solve_tridiagonal` / `solve_tridiagonal_f64` via Thomas LU (new `banded.rs`). 12 new integration tests; 47 total in crate.
- **Generic OxiCUDA sparse: SparseCsr<T> + SparseCsc<T>** (tensorlogic-oxicuda-sparse): `SparseCsr<T>` generalised over `T: Float` (backward-compatible `f32` default); new `SparseCsc<T>` (column-histogram build, `csc_spmv`, `to_csr`, `from_dense`); `SparseCsr::transpose()`, `to_csc()`, `from_dense()`; `spmv_f64`, `spmm_f64`, `spmv_batched`. 14 new tests; 27 total in crate.
- **OxiCUDA RNG f64 + streaming** (tensorlogic-oxicuda-rng): `uniform_f64` / `normal_f64` using 52-bit mantissa extraction + Box-Muller on f64; streaming `fill_uniform_chunked` / `fill_uniform_chunked_f64` / `fill_normal_chunked` callbacks; CPU builds now auto-derive `Sync` (PhantomData guard moved behind `#[cfg(feature="gpu")]`). 12 new integration tests; 60 total in crate.

### Added — GPU Autodiff + Multi-output Kernels (2026-05-29)

- **Tape-based autodiff for OxiCudaExecutor** (tensorlogic-oxicuda-backend): `TlAutodiff` implementation recording a tape during `forward` and replaying it in reverse for `backward`. Supports Matmul2D, BatchedMatmul3D, Identity, Unary, Binary, and Reduce ops. `OxiCudaTape` with `gradients: HashMap<usize, OxiCudaTensor>` exposed for inspection. Optional `native-broadcast` feature routes gradient broadcasts through GPU kernels.
- **Multi-output Gaussian Processes** (tensorlogic-sklears-kernels): new `multi_output/` module — `MultiOutputKernel` trait (`compute_block`, `block_gram_matrix`), ICM (Intrinsic Coregionalization Model), LMC (Linear Model of Coregionalization), VVGP (vector-valued GP regression). Integration tests for Swiss-roll and block-structure scenarios.
- **Variational Bayes Gaussian Mixture** (tensorlogic-quantrs-hooks): new `vmp/mixture.rs` — `VariationalGaussianMixture` implementing the VBEM algorithm (Bishop 2006 §10.2 / Attias 1999) with `VgmmConfig`, Dirichlet-prior mixing proportions, Gaussian-Normal component means, coordinate-ascent E-step / M-step, ELBO monitoring.

### Added — Round 8 (2026-06-03)

- **Probabilistic execution** (tensorlogic-scirs-backend): new `probabilistic/` sub-module — Monte Carlo samplers (`sample_bernoulli`, `sample_uniform`, `sample_normal`, `sample_categorical` via Gumbel-max trick), `mc_integrate`; `MonteCarloEstimator` with mean/variance/percentile credible intervals; `predictive_entropy`, `bald_epistemic_uncertainty`; `VariationalInference::fit` — mean-field Gaussian VI with reparameterization trick + Adam SGA maximising the ELBO. +15 tests; 641 total in crate.
- **SPARQL tensor evaluation** (tensorlogic-oxirs-bridge): new `InternedGraph` — O(1) term dictionary, predicate-indexed adjacency, parallel N-Triples bulk loading via `std::thread::scope`, `from_rdf_triples` / `into_quad_store`; new `TensorBgpEvaluator` evaluates conjunctive SELECT/BGP queries as boolean tensor contractions over `EinsumGraph` + `Scirs2Exec::forward`, decoding non-zero entries back to variable bindings. +21 tests (12 unit + 9 integration); 541 total in crate.
- **Neural Architecture Search** (tensorlogic-train): new `nas/` module — `ArchSearchSpace` / `Architecture` / `LayerSpec` with `param_count()` + `HyperparamConfig` interop; `ArchSampler` with 4-operator mutation (change op / width / activation, add/remove layer); `RegularizedEvolution` (Real et al. 2019 aging evolution with tournament selection and oldest eviction, ask/tell API); `RandomArchSearch` baseline; `NasResult`. +15 tests; 741 total in crate.
- **SVM via SMO** (tensorlogic-sklears-kernels): new `svm/` module — `SvcModel` / `SvcFitted` (C-SVM binary + one-vs-rest multiclass); `SvrModel` / `SvrFitted` (ε-SVR via 2N-variable augmented dual); `smo_svc` implementing Platt 1998 SMO with Keerthi two-loop heuristics, error cache, KKT convergence guard. All built on `Arc<dyn Kernel>`. +24 tests; 557 total in crate.

### Changed

- `scirs2-{core,linalg,autograd,optimize,sparse}` updated 0.4.2 → 0.5.0
- `oxiarc-deflate` updated 0.2.7 → 0.3.3
- `quantrs2-{core,circuit,sim}` updated 0.1.3 → 0.2.0
- `oxirs-{core,gql,ttl}` updated 0.2.4 → 0.3.1
- `oxicuda-{backend,blas,driver,fft,memory,rand,solver,sparse}` updated to 0.1.8; `oxicuda-backend` and `oxicuda-fft` added as new workspace dependencies
- `oxicode` updated 0.2 → 0.2.4
- `sklears-{core,kernel-approximation}` updated 0.1.0 → 0.1.1
- Test count: 6,407 → 7,178 (+771 new tests, 100% pass rate; 37 GPU-only tests skipped)

## [0.1.0] - 2026-04-27

### Changed - Stable Release

#### Version Bump
- **Version bump from 0.1.0-rc.1 to 0.1.0** across all workspace crates
  - Workspace version updated in root Cargo.toml
  - All 10 internal crate references updated to 0.1.0 stable (tensorlogic-ir, tensorlogic-adapters, tensorlogic-infer, tensorlogic-compiler, tensorlogic-scirs-backend, tensorlogic-train, tensorlogic-oxirs-bridge, tensorlogic-sklears-kernels, tensorlogic-quantrs-hooks, tensorlogic-trustformers)
  - tensorlogic-py version aligned to 0.1.0

#### Dependency Upgrades
- **SciRS2 ecosystem**: 0.3.0 → 0.3.4 (scirs2-core, scirs2-linalg, scirs2-autograd, scirs2-optimize)
- **SkleaRS ecosystem**: 0.1.0-rc.1 → 0.1.0 stable (sklears-core, sklears-kernel-approximation)
- **ToRSh ecosystem**: 0.1.0 → 0.1.1 (torsh-core, torsh-tensor)
- **OxiRS ecosystem**: 0.1.0 → 0.2.2 (oxirs-core, oxirs-gql, oxirs-ttl) — major API upgrade
- **oxicode**: 0.1.1 → 0.2 (serialization/codec library)
- **clap**: 4.5 → 4.6
- **clap_complete**: 4.5 → 4.6
- **assert_cmd**: 2.1 → 2.2
- **rand**: Added `rand_09` (rand 0.9) as explicit workspace alias for backward-compat crates

#### Dependency Fixes
- **oxirs-core**: Fixed RngExt trait import compatibility with scirs2-core 0.3.2
- **tensorlogic-sklears-kernels**: Fixed `StdRng` type mismatch with `sklears-kernel-approximation` — migrated from `rand_09`/`rand_distr_05` to `scirs2_core::rand_prelude`/`scirs2_core::rand_distributions` (rand 0.10 alignment, sklears-kernel-approximation uses scirs2-core's rand 0.10)
- **tensorlogic-train**: Removed unused Rng imports

#### Code Quality
- Fixed all clippy warnings (sort_by → sort_by_key, collapsible match guards)
- Zero warnings policy enforced across all crates
- 6397 tests passing (100% success rate)
- Eliminated 7 `unwrap()` calls in `sklears_integration.rs` `sample_frequencies` implementations — replaced with `expect("Normal distribution parameters must be valid")` per no-unwrap policy

## [0.1.0-rc.1] - 2026-03-06

### Changed - Release Candidate 1

#### Version Bump
- **Version bump from 0.1.0-beta.1 to 0.1.0-rc.1** across all workspace crates
  - Workspace version updated in root Cargo.toml
  - All internal crate references updated (tensorlogic-ir, tensorlogic-adapters, tensorlogic-infer, tensorlogic-compiler, tensorlogic-scirs-backend, tensorlogic-train, tensorlogic-oxirs-bridge, tensorlogic-sklears-kernels, tensorlogic-quantrs-hooks, tensorlogic-trustformers)
  - tensorlogic-py version aligned to 0.1.0-rc.1
  - Version strings updated in lib.rs doc comments (tensorlogic, tensorlogic-train, tensorlogic-trustformers)

#### Dependency Upgrades
- **SciRS2 ecosystem**: 0.1.3 -> 0.3.0 (scirs2-core, scirs2-linalg, scirs2-autograd, scirs2-optimize)
- **SkleaRS ecosystem**: 0.1.0-beta.1 -> 0.1.0-rc.1 (sklears-core, sklears-kernel-approximation)
- **ToRSh ecosystem**: 0.1.0-beta.1 -> 0.1.0 (torsh-core, torsh-tensor)
- **rand**: 0.9 -> 0.10
- **toml**: 0.9 -> 1.0
- **tokio**: 1.49 -> 1.50
- **oxrdf**: 0.3.2 -> 0.3.3
- **oxttl**: 0.2.2 -> 0.2.3
- **tempfile**: 3.24 -> 3.26

### Fixed
- **rand 0.10 compatibility**: Changed `rand::Rng` to `rand::RngExt` in learned_opt.rs for rand 0.10 API
- **Doc test in torsh_interop.rs**: Changed `no_run` to `ignore` to prevent doc test build failures

## [0.1.0-beta.1] - 2026-01-28

### Added - Beta.1 Release

#### Production Release
- **First Beta Release** - All alpha.2 features stabilized
- **4,415 tests passing** (100% pass rate)
- **Zero warnings** across all build configurations
- **Complete documentation** and examples
- **Production-ready** for real-world use

## [0.1.0-alpha.2] - 2025-12-16

### Added - Alpha.2 Release

#### CUDA/GPU Infrastructure (Experimental)
- **Device management infrastructure** (device.rs)
  - DeviceType enum (CPU, CUDA, Metal, Vulkan, ROCm)
  - Device abstraction with multi-device support
  - DeviceManager for device discovery and management
  - Future-ready for GPU backend implementation via scirs2
- **Benchmark enhancements** for GPU profiling
  - Updated all benchmark suites with device metrics
  - Preparation for GPU performance comparisons

#### Comprehensive Benchmark Suite
- **memory_footprint benchmark** (149 lines, 3 groups)
  - Memory allocation patterns for simple/matrix/complex expressions
  - Size scaling analysis (100 to 10,000 elements)
- **gradient_stability benchmark** (207 lines, 5 groups)
  - Gradient computation performance measurement
  - Simple ops, nested ops, matrix ops, quantifiers, complex expressions
  - Numerical stability testing
- **throughput benchmark** (235 lines, 5 groups)
  - Operations per second measurement
  - Element-wise, matrix, reduction, complex, and batch operations
  - Throughput tracking with Criterion
- **Fixed simd_comparison benchmark** (rewritten, 203 lines)
  - Migrated to compiler API for maintainability
  - 5 benchmark groups for SIMD comparison
- **Complete benchmark coverage**: 24 groups across 5 suites (991 total lines)

#### Packaging Infrastructure
- **PACKAGING.md** (500+ lines comprehensive guide)
  - Complete Maturin packaging documentation
  - Development setup and workflow
  - Cross-platform build instructions (Linux/macOS/Windows)
  - PyPI publishing guide (TestPyPI → PyPI)
  - CI/CD integration (GitHub Actions + GitLab CI)
  - Troubleshooting section (6 common issues)
  - Advanced optimization topics
- **GitHub Actions workflow template** (280+ lines)
  - Multi-platform wheel builds (Linux x86_64/aarch64, macOS, Windows)
  - Python version matrix (3.9-3.12)
  - Automated testing and publishing
  - SIMD builds support
- **Makefile for Python development** (100+ lines)
  - 15 common tasks automated
  - Development, building, testing, publishing targets
- **Comprehensive README.md** (500+ lines)
  - Modern documentation with badges
  - Complete feature overview
  - Quick start guides (Rust + Python)
  - Architecture diagrams
  - Performance benchmarks
  - Project status table

### Changed
- Phase 3 (SciRS2 Backend): 95% → **100% Production Ready**
- Phase 7 (Python Bindings): 95% → **98% Production Ready**
- All benchmarks now use compiler API (consistent, maintainable)

### Status
- **4,287/4,287 tests passing (100%)** - Significant test coverage expansion
- **12 tests intentionally skipped** (strategy-specific edge cases)
- **Zero warnings, zero errors**
- **Complete benchmark infrastructure** (24 groups across 5 suites)
- **Production-ready packaging**
- **272,370+ lines of Rust code** (216,811 source + 32,749 docs)

## [0.1.0-alpha.0] - 2025-11-04

### Added - Session 3

#### SIMD Acceleration Support
- **Feature flag configuration** in tensorlogic-scirs-backend
  - `simd` feature passes through to scirs2-core/scirs2-linalg
  - Transparent SIMD acceleration (2-4x speedup)
  - No code changes required
- **Capability detection** infrastructure
  - `SIMDAcceleration` feature enum
  - Runtime capability queries
  - Backend reporting to Python bindings
- **Python backend selection** (backend.rs - 480+ lines)
  - `Backend.SciRS2SIMD` enum variant
  - Smart default selection (prefers SIMD when available)
  - 4 backend query functions
  - Backend capabilities API
- **SIMD benchmark suite** (simd_comparison.rs - 450+ lines)
  - 5 benchmark groups (element-wise, reduction, matrix, logical, einsum)
  - Comprehensive SIMD vs non-SIMD comparison

#### Backend Selection API
- **PyBackend enum** (Auto, SciRS2CPU, SciRS2SIMD, SciRS2GPU)
- **PyBackendCapabilities class** with full queries
- **Backend functions**:
  - `get_backend_capabilities()` - Query backend features
  - `list_available_backends()` - List all backends
  - `get_default_backend()` - Get system default
  - `get_system_info()` - Comprehensive system info
- **Comprehensive test suite** (test_backend.py - 380+ lines, 30+ tests)
- **Type stubs** updated with backend types
- **Python example** (backend_selection.py - 280+ lines)

### Changed
- Phase 3: 80% → **95% Enhanced**
- Phase 7: 90% → **95% Production Ready**

### Status
- **783/783 tests passing (100%)**
- **SIMD acceleration functional**
- **Backend selection complete**

## [0.1.0-dev.2] - 2025-11-04

### Added - Session 2

#### Integration Tests
- **End-to-end integration tests** (end_to_end.rs - 428 lines, 18 tests)
  - Basic logical operations with execution
  - Complex nested expressions
  - Multi-arity predicates
  - Strategy comparison tests
  - Graph structure validation
  - Constant tensor handling

#### Compilation Benchmarks
- **Compilation performance benchmarks** (compilation_performance.rs - 410+ lines)
  - Simple expression benchmarks
  - Complex expression benchmarks
  - Quantifier benchmarks
  - Strategy comparison benchmarks
  - Multi-arity predicate benchmarks
  - Criterion-based infrastructure

### Changed
- Phase 8: 85% → **100% Complete**

## [0.1.0-dev.1] - 2025-11-03

### Added - Session 1

#### Python Bindings Enhancements
- **Test Suite Enhancement**
  - test_adapters.py (350+ lines) - Adapter type tests
  - test_strategies.py (470+ lines) - Strategy & property tests
  - pytest.ini configuration
  - requirements-dev.txt for dependencies
  - pyproject.toml with project metadata
- **Type Stubs** (.pyi files)
  - Complete type annotations for tensorlogic_py
  - IDE support (autocomplete, type checking)
  - mypy configuration
- **Tutorial Notebooks**
  - 01_getting_started.ipynb (800+ lines) - Beginner tutorial
  - 02_advanced_topics.ipynb (900+ lines) - Advanced tutorial
  - tutorials/README.md - Complete guide

### Changed
- Phase 7: 60% → **90% Production Ready**

## [0.1.0-dev.0] - 2025-11-03

### Added - Initial Development Phase

#### Core Infrastructure (Phases 0-2)
- **Repository Hygiene** (Phase 0)
  - LICENSE (Apache-2.0), CODEOWNERS, CONTRIBUTING.md, SECURITY.md
  - Documentation skeleton (DSL.md, IR.md, PROVENANCE.md)
  - CI configuration (fmt, clippy, tests)
- **IR & Compiler** (Phase 1)
  - `tensorlogic-ir`: AST and IR types (Term, TLExpr, EinsumGraph)
  - `tensorlogic-compiler`: Logic → tensor mapping with static analysis
  - Logic operation defaults (AND→Hadamard, OR→max, NOT→1-x, etc.)
- **Engine Traits** (Phase 2)
  - `tensorlogic-infer`: TlExecutor/TlAutodiff traits
  - Dummy executor for testing
  - Examples: 00_minimal_rule, 01_exists_reduce, 02_scirs2_execution

#### SciRS2 Backend (Phase 3)
- **Runtime Executor** (tensorlogic-scirs-backend)
  - TlExecutor trait implementation
  - TlAutodiff::forward() with full EinsumGraph execution
  - TlAutodiff::backward() with gradient computation
  - All OpType variants support (Einsum, ElemUnary, ElemBinary, Reduce)
  - Features: `cpu` (default), `simd`, `gpu` (future)
  - Integration tests: end-to-end TLExpr → Execution
  - Backward pass tests for autodiff
  - Modular structure (executor, conversion, ops, autodiff)
- **Forward pass benchmark** (forward_pass.rs - 197 lines)
  - 6 benchmark groups
  - Simple predicate, AND/OR operations, quantifiers, complex expressions

#### Core Enhancements (Phase 4.5)
- **Type System** (tensorlogic-ir)
  - Term::Typed with TypeAnnotation
  - PredicateSignature with arity/type validation
  - SignatureRegistry for predicate metadata
  - Enhanced error types
- **Graph Optimizations** (tensorlogic-ir)
  - Dead Code Elimination (DCE) with liveness analysis
  - Common Subexpression Elimination (CSE)
  - Identity operation simplification
  - Multi-pass optimization pipeline
- **Metadata & Provenance** (tensorlogic-ir)
  - SourceLocation and SourceSpan for error reporting
  - Provenance tracking (rule IDs, source files, attributes)
  - Metadata container for IR nodes
- **Compiler Enhancements** (tensorlogic-compiler)
  - Variable scope analysis
  - Type checking & inference
  - Expression-level CSE
  - SymbolTable integration
  - Enhanced diagnostics
- **Execution Enhancements** (tensorlogic-infer)
  - Batch execution
  - Shape inference
  - Backend capabilities
  - Execution profiling

#### OxiRS Bridge (Phase 4)
- **Schema Integration** (tensorlogic-oxirs-bridge)
  - Symbol tables from RDF* schema analysis
  - SchemaAnalyzer for extracting classes and properties
  - Provenance tracking infrastructure
  - SHACL constraint parser (Turtle format)
  - SHACL → TLExpr conversion
  - Support for sh:minCount, sh:maxCount, sh:class, sh:datatype, sh:pattern

#### Interop Crates (Phase 5)
- **SkleaRS Kernels** (tensorlogic-sklears-kernels)
  - Rule similarity kernel
  - Predicate overlap kernel
  - Tensor kernels (Linear, RBF, Polynomial, Cosine)
  - 24 comprehensive tests
- **QuantrS2 Hooks** (tensorlogic-quantrs-hooks)
  - Factor representation with normalization
  - Factor graph with adjacency tracking
  - Message passing algorithms (sum-product, max-product)
  - TLExpr → Factor graph conversion
  - 15 comprehensive tests
- **TrustformeRS** (tensorlogic-trustformers)
  - Self-attention as einsum operations
  - Multi-head attention
  - Feed-forward networks (standard + gated GLU)
  - Position encodings (sinusoidal, learned, relative, RoPE, ALiBi)
  - Layer normalization (LayerNorm + RMSNorm)
  - Transformer encoder/decoder layers
  - Rule-based attention patterns
  - Sparse attention patterns
  - Extended model presets (GPT-2/3, LLaMA, BLOOM, T5)
  - 123 comprehensive tests

#### Training Scaffolds (Phase 6)
- **Training Infrastructure** (tensorlogic-train)
  - Loss functions (cross-entropy, MSE, rule satisfaction, constraint violations)
  - Optimizers (SGD, Adam, AdamW with gradient clipping)
  - Learning rate schedulers (Step, Exponential, Cosine, Warmup)
  - Batch management (iterator, shuffling, stratified sampling)
  - Training loop (Trainer with epoch/batch iteration)
  - Callbacks (early stopping, checkpointing, LR plateau reduction)
  - Metrics (accuracy, precision, recall, F1 score)
  - 28 unit tests

#### Python Bindings (Phase 7)
- **Core Type Bindings** (types.rs - 331 lines)
  - PyTerm: Variables and constants
  - PyTLExpr: Full logical expression API (13 operations)
  - PyEinsumGraph: Compiled tensor graphs
- **Compilation API** (compiler.rs - 153 lines)
  - compile(expr) - Default compilation
  - compile_with_config(expr, config) - Custom strategies
  - 6 compilation strategy presets
- **Execution API** (executor.rs - 72 lines)
  - execute(graph, inputs) - NumPy array execution
  - Dynamic tensor shape handling
- **NumPy Integration** (numpy_conversion.rs - 63 lines)
  - Bidirectional conversion (NumPy ↔ SciRS2)
  - Safe memory management
- **Adapter Bindings** (adapters.rs)
  - PySymbolTable, PyCompilerContext
  - PyDomainInfo, PyPredicateInfo
- **Examples**: 5 Rust examples + 10 Python examples
- **Test Coverage**: 30 Rust tests

#### Validation & Scale (Phase 8)
- **Property Tests** (property_tests.rs - 900+ lines, 21/21 passing)
  - CompilationConfig integration
  - Strategy mapping module (180+ lines)
  - 17 core property tests (symmetry, associativity, monotonicity, etc.)
  - 4 strategy-specific tests
- **CompilationConfig Integration**
  - Updated logic operations to use config strategies
  - Added Min/Max element-wise operations to backend
  - Optimized Product AND with einsum fusion
  - Support for 26+ compilation strategies

### Testing
- **783 tests** across all crates
- **100% pass rate**
- **Zero warnings** in release builds
- Coverage: unit tests, integration tests, property tests, Python tests

### Documentation
- Complete project guide (CLAUDE.md)
- SciRS2 integration policy
- Security policy
- Contributing guidelines
- Tutorial notebooks
- 15+ examples (Rust + Python)

[0.1.2]: https://github.com/cool-japan/tensorlogic/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/tensorlogic/releases/tag/v0.1.1
