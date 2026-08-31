# TensorLogic — Flagship Meta-Crate — TODO

**Status**: Stable | **Version**: 0.1.3 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-31
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

Umbrella crate re-exporting planning, execution, and integration layers.

## Completed

### Core Functionality
- [x] Re-export all planning layer components
  - [x] tensorlogic-ir
  - [x] tensorlogic-compiler
  - [x] tensorlogic-infer
  - [x] tensorlogic-adapters
- [x] Re-export all execution layer components
  - [x] tensorlogic-scirs-backend
  - [x] tensorlogic-train
- [x] Re-export all integration layer components
  - [x] tensorlogic-oxirs-bridge
  - [x] tensorlogic-sklears-kernels
  - [x] tensorlogic-quantrs-hooks
  - [x] tensorlogic-trustformers

### Prelude Module - COMPLETE
- [x] `Term` (from tensorlogic::ir)
- [x] `TLExpr` (from tensorlogic::ir)
- [x] `compile_to_einsum` (from tensorlogic::compiler)
- [x] `TlExecutor` (from tensorlogic::infer)
- [x] `TlAutodiff` (from tensorlogic::infer)
- [x] `Scirs2Exec` (from tensorlogic::scirs_backend)

Note: `EinsumGraph`, `CompilerContext`, `CompilationConfig`, `IrError`, `CompilerError`
are available via their respective sub-modules but are not in the prelude.

### Documentation - COMPLETE
- [x] Comprehensive README.md
  - [x] Overview and quick start
  - [x] Architecture explanation
  - [x] Component documentation links
  - [x] Examples with commands
  - [x] Feature flags documentation
  - [x] Migration guide from individual crates
  - [x] Accurate prelude contents
- [x] Module organization
  - [x] Planning layer exports
  - [x] Execution layer exports
  - [x] Integration layer exports

### Examples - COMPLETE
- [x] 00_minimal_rule - Basic predicate and compilation
- [x] 01_exists_reduce - Existential quantifier with reduction
- [x] 02_scirs2_execution - Full execution with SciRS2 backend
- [x] 03_rdf_integration - OxiRS bridge with RDF* data
- [x] 04_compilation_strategies - All 6 strategy presets compared

All examples work correctly and demonstrate meta crate usage.

### Workspace Integration - COMPLETE
- [x] Proper Cargo.toml structure
  - [x] All component dependencies specified
  - [x] Workspace inheritance (version, edition, license, etc.)
  - [x] Descriptive metadata (keywords, categories)
  - [x] Example declarations
- [x] Clean crate structure
  - [x] lib.rs with organized re-exports
  - [x] examples/ directory with all examples
  - [x] README.md and TODO.md
- [x] Moved from root to crates/tensorlogic/
- [x] All examples migrated successfully
- [x] Build and test infrastructure working
- [x] Documentation references updated

## Future Enhancements

### Prelude Improvements
- [x] Add more convenience re-exports based on user feedback (completed 2026-04-14)
  - [x] `EinsumGraph` - frequently needed alongside `TLExpr` (completed 2026-04-14)
  - [x] `CompilationConfig` - needed for most non-default compilations (completed 2026-04-14)
  - [x] `IrError`, `CompilerError` - for error handling in user code (completed 2026-04-14 — expanded to all 10 canonical sub-crate error types)
- [ ] Group exports by common use cases
- [ ] Trait extension methods for ergonomic API

### Additional Examples
- [ ] Complex nested expressions example
- [ ] Performance optimization example
- [ ] Multi-backend comparison example
- [ ] Training workflow example
- [ ] Real-world application examples

### Documentation
- [ ] Add tutorial for meta crate usage patterns
- [ ] Create cookbook with common recipes
- [ ] Document best practices for feature selection
- [ ] Add performance comparison guide

### Feature Flags
- [x] Fine-grained feature control (completed 2026-04-14)
  - [x] Individual component features (completed 2026-04-14 — `train`, `oxirs`, `quantrs`, `sklears`, `trustformers` gates)
  - [ ] Backend selection features
  - [x] Integration layer opt-in (completed 2026-04-14)
- [x] Performance features (completed 2026-04-14)
  - [x] `full` feature for all components (completed 2026-04-14)
  - [x] `minimal` feature for core only (completed 2026-04-14)
  - [ ] `no-std` support investigation

### Tooling
- [ ] Meta crate version sync checker
- [ ] Component dependency graph visualization
- [ ] Automatic re-export generation tool

## Low Priority

### Optimization
- [ ] Compile time optimization
  - [ ] Feature-gated dependencies
  - [ ] Conditional compilation
- [ ] Binary size optimization
  - [ ] Strip unused components
  - [ ] Link-time optimization settings

### Testing
- [x] Integration tests for meta crate (completed 2026-04-14)
  - [x] Verify all re-exports work (completed 2026-04-14 — `tests/reexport_surface.rs`, 7 tests)
  - [x] Test prelude imports (completed 2026-04-14 — `tests/prelude_smoke.rs`, 3 tests)
  - [ ] Example compilation tests
- [ ] Documentation tests
  - [ ] All code examples in README
  - [ ] API usage patterns

---

**Completion**: 100% (All planned features for 0.1.0)
**Release**: v0.1.0 (2026-03-06)

**Production Ready Features:**
- Complete re-export of all 10 component crates
- Organized module structure (planning/execution/integration layers)
- Prelude module with 6 core items
- 5 comprehensive examples
- Complete documentation
- Virtual workspace integration

**Notes:**
- This is a pure re-export crate with no implementation code
- All functionality is provided by component crates
- Examples serve as integration tests
- Version is synchronized with all components (0.1.0)
- Prelude intentionally kept minimal; users should import from sub-modules for additional types

## 2026-04-14 — Post-stable hardening

First pass of workspace-level integration tests and the long-staged v0.1.1 convenience work, landed under the stable 0.1.0 branch without any breaking change:

- **Integration tests** — added `tests/prelude_smoke.rs` (3 tests) exercising a minimal end-to-end flow through the re-exported prelude items (`Term`, `TLExpr`, `compile_to_einsum`, `Scirs2Exec`, `.forward()`), and `tests/reexport_surface.rs` (7 tests, 5 feature-gated) verifying every re-exported sub-crate module (`tensorlogic::ir`, `::compiler`, `::infer`, `::adapters`, `::scirs_backend`, `::train`, `::oxirs_bridge`, `::quantrs_hooks`, `::sklears_kernels`, `::trustformers`) resolves and type-checks.
- **Prelude extension** — `EinsumGraph`, `CompilationConfig`, and the 10 canonical sub-crate error types (`AdapterError`, `CompileError`, `ExecutorError`, `IrError`, `TlBackendError`, plus feature-gated `BridgeError`, `PgmError`, `KernelError`, `TrainError`, `TrustformerError`) are now in `tensorlogic::prelude` so downstream users don't have to know which sub-crate owns each type.
- **Feature flags** — `default = ["full"]`, `full = ["train", "oxirs", "quantrs", "sklears", "trustformers"]`, `minimal = []`, plus one gate per optional integration. The 5 integration-layer sub-crate deps are marked `optional = true`; each corresponding `pub use` in `lib.rs` is `#[cfg]`-gated. `--features minimal` builds only the 5 mandatory-core crates (ir / infer / adapters / compiler / scirs-backend).
- **Verification** — `cargo check --workspace --all-features` (0 warnings), `cargo clippy --workspace --all-targets --all-features -- -D warnings` (clean), `cargo nextest run --workspace --all-features` (7,178 tests, all green).

## v0.2.0 / Future Work

- Stabilize all feature-flag combinations in CI (`--features minimal` through `--all-features`).
- Doc-book chapters for every sub-crate.
- End-to-end tutorial examples covering the `full` → `minimal` feature spectrum.
- Benchmarks at the flagship level (currently only sub-crates have benches/).
