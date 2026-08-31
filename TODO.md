# TensorLogic — TODO

**Status**: Stable | **Version**: 0.1.2 | **Released**: 2026-08-30 | **Last Updated**: 2026-08-30

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [ ] `tensorlogic-infer`: `crates/tensorlogic-infer/src/backend_kind.rs:139` — implement Metal, Vulkan, ROCm, WebGPU, LevelZero backends (currently doc-hidden stubs)
  - Priority: P2 | Scope: large | Hint: none
**History**: See [CHANGELOG.md](./CHANGELOG.md) for release history.

## Round 5 (2026-04-17) — Complete

### Deliverables
- [x] **BackendKind enum** — `crates/tensorlogic-infer/src/backend_kind.rs`: 7-variant enum (`Scirs`, `OxiCuda` live; `Metal`/`Vulkan`/`Rocm`/`Webgpu`/`Levelzero` doc-hidden stubs). Implements `std::str::FromStr`. Methods: `default_backend()`, `from_env()`, `is_gpu()`, `supports_autodiff()`, `validate()`, `as_str()`.
- [x] **tensorlogic-oxicuda-sparse** — new wrapper crate: `SparseCsr`, `spmv`, `spmm`. CPU path: manual CSR arithmetic. GPU path: stubbed (`#[cfg(feature="gpu")]`), wired in Round 6.
- [x] **tensorlogic-oxicuda-solver** — new wrapper crate: `solve_lu`, `solve_cholesky`, `solve_qr_lstsq`, `cg_solve`. Pure-Rust CPU solvers (Doolittle LU, Cholesky-Banachiewicz, Modified Gram-Schmidt QR, CG). GPU path: stubbed.
- [x] **tensorlogic-oxicuda-rng** — new wrapper crate: `RngEngine` (PCG-XSH-RR, Box-Muller), `uniform_f32`, `normal_f32`, `bernoulli`. `Send+!Sync`. GPU path: stubbed.
- [x] **Umbrella feature flags** — `sparse`, `solver`, `rng`, `full-advanced` added to `crates/tensorlogic/Cargo.toml` and re-exported in `crates/tensorlogic/src/lib.rs`.
- [x] **`[patch.crates-io]` extended** — `oxicuda-sparse`, `oxicuda-solver`, `oxicuda-rand` patched to `$HOME/work/oxicuda/crates/*`.
- [x] **Native `fill` kernel** — `oxicuda_blas::elementwise::fill<T>()` in `$HOME/work/oxicuda/crates/oxicuda-blas/src/elementwise/fill.rs`. PTX via `ElementwiseOp::Fill` (added to oxicuda-ptx).
- [x] **Native `broadcast_axes` kernel** — `oxicuda_blas::elementwise::broadcast_axes<T>()` in `.../broadcast.rs`. `BroadcastTemplate` in oxicuda-ptx uses stride-zero trick; 28-param PTX kernel supports rank ≤ 8.
- [x] **Autodiff rewiring** — `crates/tensorlogic-oxicuda-backend`: `native-broadcast = ["gpu"]` feature. Under this feature, `broadcast_to_shape` and Mean-divisor `fill` use native GPU kernels (upload→kernel→readback). Removed 4 `TODO(perf)` Round-5-follow-up comments.
- [x] **LRU cache correctness fix** — `tensorlogic-infer/src/memo_cache.rs`: `find_lru_key` now uses the deque-front strategy (O(1), tie-safe) instead of `min_by_key(last_accessed)` which was racy under nanosecond-level Instant precision.

### Round 5 Test Coverage
- `tensorlogic-infer/tests/backend_kind_smoke.rs`: 13 tests
- `tensorlogic-oxicuda-sparse/tests/smoke.rs`: 13 tests (2 GPU ignored)
- `tensorlogic-oxicuda-solver/tests/smoke.rs`: 30 tests
- `tensorlogic-oxicuda-rng/tests/smoke.rs`: 47 tests
- `tensorlogic-oxicuda-backend/tests/native_broadcast_smoke.rs`: 6 tests (2 GPU ignored)
- Full workspace: **6784 tests, 6784 passed, 21 skipped (GPU-only), 0 failed**

TensorLogic compiles logical rules into tensor equations (einsum graphs) with a minimal DSL + IR, enabling neural / symbolic / probabilistic models in a unified tensor framework.

## At a glance

- Phase 0 — Repo Hygiene. **Complete.**
- Phase 1 — Minimal IR & Compiler. **Complete.**
- Phase 2 — Engine Traits & Dummy Executor. **Complete.**
- Phase 3 — SciRS2 Backend. **Complete.**
- Phase 4 — OxiRS Bridge. **Complete.**
- Phase 5 — Interop Crates. **Complete.**
- Phase 6 — Training Scaffolds. **Complete.**
- Phase 7 — Python Bindings. **Complete.**
- Phase 8 — Validation & Scale. **Complete.**
- Stable release 0.1.0 — 2026-04-06. **Complete.**
- Post-stable hardening — 2026-04-14: three oversize files split (`kv_cache.rs`, `codegen.rs`, `sparql.rs`), flagship integration tests added, feature flags introduced. **Complete.**
- Documentation hygiene — 2026-04-15: extracted release history to CHANGELOG.md, unified TODO header template, added forward-looking v0.2.0 sections to every crate TODO. **Complete.**
- v0.2.0 research preview — 2026-04-15: six features delivered (rule-guided decoding, learned kernel composition, variational message passing, deep kernel learning, speculative decoding, partial error recovery) + two pre-existing items audited complete (incremental recompilation, streaming inference). **Complete.**

See [CHANGELOG.md](./CHANGELOG.md) for the full record of each phase, RC, and stable-release deliverable.

## Post-stable Roadmap

### v0.1.x Stabilization (in-repo, unblocked)

#### Completed 2026-04-15

- [x] ~~Split the nine remaining files > 1,500 lines using the same `dir/{mod,...}.rs` pattern used 2026-04-14:~~ (completed 2026-04-15)
  - [x] ~~`crates/tensorlogic-adapters/src/database.rs` (1,613 L)~~ (completed 2026-04-15)
  - [x] ~~`crates/tensorlogic-compiler/src/dead_code.rs` (1,614 L)~~ (completed 2026-04-15)
  - [x] ~~`crates/tensorlogic-compiler/src/partial_eval.rs` (1,789 L)~~ (completed 2026-04-15)
  - [x] ~~`crates/tensorlogic-compiler/src/symbolic_diff.rs` (1,524 L)~~ (completed 2026-04-15)
  - [x] ~~`crates/tensorlogic-infer/src/causal.rs` (1,589 L)~~ (completed 2026-04-15)
  - [x] ~~`crates/tensorlogic-ir/src/resolution.rs` (1,712 L)~~ (completed 2026-04-15)
  - [x] ~~`crates/tensorlogic-oxirs-bridge/src/sparql_gen.rs` (1,530 L)~~ (completed 2026-04-15)
  - [x] ~~`crates/tensorlogic-quantrs-hooks/src/loopy_bp.rs` (1,744 L)~~ (completed 2026-04-15)
  - [x] ~~`crates/tensorlogic-train/src/hyperparameter.rs` (1,641 L) and `loss.rs` (1,551 L)~~ (completed 2026-04-15)
- [x] ~~Fix the doc-build warning: `cargo doc --workspace --no-deps --all-features` reports a bin/lib output-path collision for `tensorlogic`. Rename the CLI `[[bin]]` `name` in `crates/tensorlogic-cli/Cargo.toml` to `tensorlogic-cli` (Cargo issue #6313).~~ (completed 2026-04-15)

#### Remaining

- Coverage-gate per-crate test counts (nextest + tarpaulin or llvm-cov) into CI once SciRS2 stabilizes.

### v0.2.0 (blocked on upstream SciRS2 / ecosystem)

- Distributed training (NCCL / multi-GPU scheduling).
- PyPI release of `tensorlogic-py` via maturin; requires release-ready `abi3-py39` manylinux wheels.

### v0.2.0 / Research (in-repo, forward-looking)

#### Delivered 2026-04-15 (Research Preview)

- [x] ~~Rule-guided sampling decoder in `tensorlogic-trustformers`~~ — `rule_guided_decoder/` module: TLExpr-constrained beam search with hard-masking and soft-penalty modes.
- [x] ~~Learned kernel composition in `tensorlogic-sklears-kernels`~~ — `learned_composition/` module: differentiable softmax-gated mixture over a kernel library.
- [x] ~~Variational Message Passing in `tensorlogic-quantrs-hooks`~~ — `vmp/` module: VMP over conjugate exponential families (Gaussian, Categorical, Dirichlet).
- [x] ~~Deep Kernel Learning in `tensorlogic-sklears-kernels`~~ — `deep_kernel/` module: MLP feature extractor composed with a base kernel (Wilson et al., 2016); Xavier init via SciRS2-Core RNG.
- [x] ~~Speculative decoding in `tensorlogic-trustformers`~~ — `speculative_decoding/` module: DraftModel + TargetModel traits with rejection-sampling acceptance (Leviathan et al., 2023); empirical distribution preservation chi-square-verified.
- [x] ~~Partial error recovery in `tensorlogic-compiler`~~ — `error_recovery/` module: TolerantCompiler with configurable RecoveryStrategy; compiles well-formed expressions around non-fatal errors instead of aborting.
- [x] ~~OxiCUDA GPU backend scaffold in `tensorlogic-oxicuda-backend`~~ — new crate, feature-gated (`gpu`), MVP matmul via `oxicuda-blas`. Supersedes the "blocked on SciRS2 GPU stabilization" status (OxiCUDA is the COOLJAPAN Pure Rust CUDA replacement: ~/work/oxicuda/).
- [x] ~~Expanded VMP family catalogue in `tensorlogic-quantrs-hooks`~~ — `vmp/gamma.rs` and `vmp/beta.rs`: Gamma-Poisson and Beta-Bernoulli conjugate families with ExponentialFamily trait, closed-form KL, and posterior update helpers.
- [x] ~~Kernel PCA in `tensorlogic-sklears-kernels`~~ — `kernel_pca/` module: Scholkopf-Smola-Muller (1998) implementation with double-centering, scirs2-linalg eigendecomp, fit/transform API generic over Kernel + Clone + 'static.
- [x] ~~Numerical MoE layer in `tensorlogic-trustformers`~~ — `moe/` research-preview submodules: TopKGate, LinearExpert, MoELayer with capacity-factor dropping and importance/load auxiliary losses.

#### Audited complete 2026-04-15 (pre-existing implementations)

- [x] ~~Incremental re-compilation in `tensorlogic-compiler`~~ — `incremental.rs` (`IncrementalCompiler`, `ChangeDetector`, `ChangeSet`, `IncrementalStats`) + `cache.rs` hash-based `CompileCache`.
- [x] ~~Streaming inference API in `tensorlogic-infer`~~ — `streaming.rs` (`TlStreamingExecutor` trait, `StreamingConfig`, `StreamProcessor`, `ChunkIterator`, `StreamingConfigV2` with backpressure + watermarks).

#### Delivered 2026-04-16

- [x] ~~Real SGEMM wiring in `tensorlogic-oxicuda-backend`~~ — `GpuState` (Context + BlasHandle), host↔device copies via `oxicuda-memory`, `gemm_api::gemm` SGEMM dispatch, stream sync, `DimensionOverflow` / `From<CudaError>` / `From<BlasError>` error conversions. GPU integration test gated behind `TENSORLOGIC_GPU_TESTS=1`.
- [x] ~~LoRA adapter in `tensorlogic-train`~~ — `lora/` module: Low-Rank Adaptation (Hu et al., 2021) with ΔW=BA decomposition, merge/unmerge, effective_weight, dropout, compression ratio. `LoraAdapter` multi-layer manager with summary statistics. 12 unit tests + 2 integration tests.
- [x] ~~Sparse attention in `tensorlogic-trustformers`~~ — `sparse_attention/` module: Longformer-style (Beltagy et al., 2020) sliding-window + global-token attention. Dense boolean mask with sparsity metrics, multi-head forward pass, causal masking support. 12 unit tests + 4 integration tests.

#### Remaining

All initial v0.2.0 research-preview items for in-repo work are delivered. Next enhancements tracked per-crate: see each `crates/*/TODO.md`.

## Round 6 (2026-05-29) — Complete

### Deliverables

- [x] **Temporal `Next (X)` operator** — `crates/tensorlogic-compiler/src/compile/modal_temporal.rs`: replaces the `bail!` stub with a real shift-forward implementation. Compiler emits `ElemUnary { op: "temporal_next:<axis>" }`. Backend `crates/tensorlogic-scirs-backend/src/temporal_ops.rs`: `shift_next` (forward shift along any axis, T=1 edge case handled), `shift_prev` (VJP). Wired into `autodiff.rs` (forward + backward) and `parallel_executor.rs`.
- [x] **Temporal `Until (U)` operator** — same files as Next. Backward-in-time recurrence `u[T-1]=b[T-1]`, `u[t]=b[t]⊕(a[t]⊗u[t+1])`. Two semantics: MaxMin (⊕=max, ⊗=min) for `TemporalStrategy::Max/LogSumExp`; ProbSumProduct (⊕=a+b−ab, ⊗=a·b) for `Sum`. Encoded in op string `"temporal_until:<tag>:<axis>"`. VJP `until_scan_vjp` uses forward-in-time adjoint accumulation. 17 new tests.
- [x] **Variational Bayes GMM** — `crates/tensorlogic-quantrs-hooks/src/vmp/mixture.rs` (NEW, 570 lines): `VariationalGaussianMixture` (standalone VBEM, following gamma/beta pattern — NOT wired into engine enums). Full Bishop PRML §10.2 VBEM: Dirichlet prior on weights, Gaussian-mean posteriors with known precision. Divergence-guard ELBO loop matching `engine.rs`. `VgmmConfig` builder, `VgmmResult` with `mixing_weights`, `hard_assignments`, `component_counts`. 11 unit tests + 1 end-to-end integration test (3-cluster recovery).
- [x] **Multi-output / vector-valued kernels** — `crates/tensorlogic-sklears-kernels/src/multi_output/` (NEW directory module): `MultiOutputKernel` trait (`compute_block → Array2<f64>`, default `block_gram_matrix`), `KroneckerICMKernel` and `KroneckerLMCKernel` (matrix-valued wrappers over existing scalar multi-task kernels), `VvgpModel` / `VvgpFitted` (vector-valued GP regression via Cholesky solve + posterior prediction). 20 new tests (10 unit + 10 integration).

### Round 6 Test Coverage
- `tensorlogic-scirs-backend/src/temporal_ops.rs`: 10 unit tests; `tests/temporal_grad.rs`: 7 integration tests
- `tensorlogic-quantrs-hooks/src/vmp/mixture.rs`: 10 unit tests; `tests/vmp_integration.rs`: +1 integration test
- `tensorlogic-sklears-kernels/src/multi_output/tests.rs`: 10 unit tests; `tests/multi_output_integration.rs`: 10 integration tests

## Round 7 (2026-06-02) — Complete

### Deliverables

- [x] **Exact LTL Release (R), WeakUntil (W), StrongRelease (M) operators** — `crates/tensorlogic-scirs-backend/src/temporal_ops.rs` + `crates/tensorlogic-compiler/src/compile/modal_temporal.rs`: replaced three mathematically incorrect single-step approximations with exact finite-trace backward-scan recurrences. Unified `temporal_binary_scan` / `temporal_binary_scan_vjp` (generalize Round 6's `until_scan`): OUTER/INNER closed over `TemporalBinaryForm` × `UntilSemantics`; boundary_val ∈ {0.0,1.0} distinguishes strong/weak variants. Op strings `temporal_weakuntil:<tag>:<axis>`, `temporal_release:`, `temporal_strongrelease:`. VJPs wired into all 6 dispatch sites in `autodiff.rs` + `parallel_executor.rs`. +10 unit tests + 6 integration tests; all 17 Round 6 temporal tests still pass.
- [x] **OxiCUDA solver: f64 variants + Preconditioned CG + Thomas tridiagonal LU** — `crates/tensorlogic-oxicuda-solver/`: generic `lu_core<T:Float>` / `cholesky_core` / `qr_core` / `cg_core` extracted; `solve_lu_f64`, `solve_cholesky_f64`, `solve_qr_lstsq_f64`, `cg_solve_f64` added. New `pcg_solve` / `pcg_solve_f64` with `Precond::Jacobi` (diagonal scaling) and `Precond::IncompleteCholesky` (exact dense Cholesky for IC(0) preconditioner). New `banded.rs`: Thomas algorithm `solve_tridiagonal` / `solve_tridiagonal_f64`. 12 new integration tests in `tests/advanced_solver.rs`; 47 total.
- [x] **OxiCUDA sparse: generic SparseCsr<T> + SparseCsc + transpose + f64 + batched SpMV** — `crates/tensorlogic-oxicuda-sparse/`: `SparseCsr<T=f32>` generalized over `T:Float` (backward-compatible default). New `csc.rs`: `SparseCsc<T>` (column-histogram build, `csc_spmv`, `to_csr`, `from_dense`). `SparseCsr::transpose()`, `SparseCsr::to_csc()`, `SparseCsr::from_dense()`. `spmv_f64`, `spmm_f64`, `spmv_batched`. 14 new tests in `tests/advanced_sparse.rs`; 27 total.
- [x] **OxiCUDA rng: f64 sampling + streaming API + Send+Sync for CPU builds** — `crates/tensorlogic-oxicuda-rng/`: `uniform_f64` / `normal_f64` using 52-bit mantissa extraction + Box–Muller on f64. `fill_uniform_chunked` / `fill_uniform_chunked_f64` / `fill_normal_chunked` streaming callbacks (chunk-size-agnostic determinism). `PhantomData<*const ()>` made `#[cfg(feature="gpu")]`-only so CPU builds auto-derive `Sync`. 12 new integration tests in `tests/f64_and_stream.rs`; 60 total.

### Round 7 Test Coverage
- `tensorlogic-scirs-backend/src/temporal_ops.rs`: +10 unit tests; `tests/temporal_grad.rs`: +6 integration tests
- `tensorlogic-oxicuda-solver/tests/advanced_solver.rs`: 12 new integration tests; 47 total in crate
- `tensorlogic-oxicuda-sparse/tests/advanced_sparse.rs`: 14 new tests; 27 total in crate
- `tensorlogic-oxicuda-rng/tests/f64_and_stream.rs`: 12 new tests; 60 total in crate
- **Full workspace: 6938 tests, 6938 passed, 21 skipped (GPU-only), 0 failed**

## Round 8 (2026-06-03) — Complete

### Deliverables

- [x] **Probabilistic Execution in scirs-backend** — new `crates/tensorlogic-scirs-backend/src/probabilistic/` module: `sample_bernoulli`/`sample_uniform`/`sample_normal`/`sample_categorical` (Gumbel-max trick) + `mc_integrate`; `MonteCarloEstimator` (mean/variance/percentile CI) + `predictive_entropy` + BALD epistemic uncertainty; `VariationalInference::fit` — mean-field Gaussian VI with reparameterization trick and Adam SGA maximizing the reparameterized ELBO. RNG via `scirs2_core::random`. 15 new tests; 641 total in crate.
- [x] **SPARQL via tensor operations in oxirs-bridge** — `src/interned_graph.rs`: `InternedGraph` O(1) term dictionary + predicate-indexed adjacency, parallel N-Triples bulk loading via `std::thread::scope`, `from_rdf_triples`/`into_quad_store` bridges; `src/sparql/tensor_eval.rs`: `TensorBgpEvaluator` evaluates conjunctive SELECT/BGP queries as boolean tensor contraction over `EinsumGraph` + `Scirs2Exec::forward`, decoding nonzero output entries back to variable bindings. 21 new tests (12 unit + 9 integration); 541 total in crate.
- [x] **Neural Architecture Search in tensorlogic-train** — `src/nas/`: `ArchSearchSpace`/`Architecture`/`LayerSpec` with `param_count()` + `HyperparamConfig` interop; `ArchSampler` (random generation + 4-operator mutation); `RegularizedEvolution` (Real et al. 2019 aging evolution with tournament selection, aging eviction, ask/tell API); `RandomArchSearch` baseline. 15 new tests; 741 total in crate.
- [x] **SVM via SMO in tensorlogic-sklears-kernels** — `src/svm/`: `SvcModel`/`SvcFitted` (C-SVM binary + one-vs-rest multiclass), `SvrModel`/`SvrFitted` (ε-SVR via 2N-variable augmented dual); `smo_svc` implementing Platt 1998 SMO with Keerthi two-loop heuristics, error cache, KKT convergence guard, non-convergence error. All built on the existing `Arc<dyn Kernel>` trait. 24 new tests; 557 total in crate.

### Round 8 Test Coverage
- `tensorlogic-scirs-backend/src/probabilistic/`: 15 new tests; 641 total in crate
- `tensorlogic-oxirs-bridge/src/interned_graph.rs`: 12 unit tests; `tests/tensor_sparql.rs`: 9 integration tests; 541 total in crate
- `tensorlogic-train/src/nas/tests.rs`: 15 new tests; 741 total in crate
- `tensorlogic-sklears-kernels/src/svm/tests.rs`: 24 new tests; 557 total in crate
- **Full workspace: 7019 tests, 7018 passed, 21 skipped (GPU-only), 0 failures** (1 pre-existing resource-exhaustion flake in tensorlogic-adapters::integration_tests, passes in isolation)

## Round 9 (2026-08-30) — Complete

### Deliverables

- [x] **SQLite backend migrated from rusqlite to OxiSQL** (tensorlogic-adapters) — the `sqlite` feature now depends on `oxisql-core` + `oxisql-sqlite-compat` instead of `rusqlite` (which bundled a C SQLite build), removing the crate's last C dependency for full Pure Rust compliance. SQL placeholder syntax changed from `?` to numbered `$1, $2, ...`; `initialize_schema` no longer requires `&mut self`; reads/writes now run through a dedicated Tokio runtime with errors mapped via a new `map_oxi` helper; new `count_for_schema` row-count helper.
- [x] **RDF/Turtle stack migrated from oxrdf/oxttl to OxiXML** (tensorlogic-oxirs-bridge) — `oxrdf`/`oxttl` are now workspace aliases for the COOLJAPAN `oxixml-model`/`oxixml-turtle` crates (0.1.2) instead of the upstream Oxigraph crates; the dead `oxirs-ttl` dependency was dropped. Call sites in `schema/inference.rs`, `schema/metadata.rs`, `schema/nquads.rs`, `schema/ntriples.rs`, and `shacl/mod.rs` updated for OxiXML's borrowed-reference iterator API.
- [x] **GPU sparse shape validation** (tensorlogic-oxicuda-sparse) — new `check_spmv_shapes`/`check_spmm_shapes` helpers validate operand buffer lengths against CSR matrix dimensions before dispatching to device kernels, closing an out-of-bounds device-memory risk on mismatched `x`/`y` (SpMV) or `b`/`c` (SpMM) lengths.
- [x] Dependency updates: `scirs2-{core,linalg,autograd,optimize,sparse}` 0.5.0 → 0.6.5; `oxicuda-{backend,blas,driver,fft,memory,rand,solver,sparse}` 0.1.8 → 0.5.5; `torsh-{core,tensor}` 0.1.2 → 0.2.0; `sklears-{core,kernel-approximation}` 0.1.1 → 0.2.0; `quantrs2-{core,circuit,sim}` 0.2.0 → 0.2.1; `oxirs-{core,gql}` 0.3.1 → 0.4.1; `oxicode` → 0.2.6; `oxiarc-deflate` 0.3.3 → 0.4.1.

### Round 9 Test Coverage
- **Full workspace: 7019 tests (default features), 7178 tests (--all-features), 0 failures** — net test count unchanged from Round 8 (this round was migrations/hardening, not new surface area).

### Known non-blocking items (tracked, not resolved this round)
- Two HIGH-severity RUSTSEC advisories in `quick-xml 0.37.5`, reachable transitively via `oxirs-core` → `oxrdfxml` (this exposure already shipped in 0.1.1). Fix belongs upstream in `oxirs-core`, not here; scoped in `deny.toml` pending that migration.
- `tensorlogic-py` (standalone PyO3 crate, not part of the crates.io publish set) currently fails to build standalone due to an upstream `numpy 0.27` (ndarray 0.16) vs `scirs2-core 0.6.5` (ndarray 0.17) conflict. Unrelated to this round's changes.

## Per-crate TODOs

- [crates/tensorlogic/TODO.md](./crates/tensorlogic/TODO.md) — Umbrella / flagship meta-crate.
- [crates/tensorlogic-ir/TODO.md](./crates/tensorlogic-ir/TODO.md) — IR and DSL.
- [crates/tensorlogic-infer/TODO.md](./crates/tensorlogic-infer/TODO.md) — Executor traits + autodiff scaffolding.
- [crates/tensorlogic-compiler/TODO.md](./crates/tensorlogic-compiler/TODO.md) — Logic → einsum compilation.
- [crates/tensorlogic-adapters/TODO.md](./crates/tensorlogic-adapters/TODO.md) — External data + codegen adapters.
- [crates/tensorlogic-scirs-backend/TODO.md](./crates/tensorlogic-scirs-backend/TODO.md) — SciRS2 execution backend.
- [crates/tensorlogic-train/TODO.md](./crates/tensorlogic-train/TODO.md) — Training loop + loss / optimizer / scheduler.
- [crates/tensorlogic-oxirs-bridge/TODO.md](./crates/tensorlogic-oxirs-bridge/TODO.md) — OxiRS / RDF / SPARQL bridge.
- [crates/tensorlogic-quantrs-hooks/TODO.md](./crates/tensorlogic-quantrs-hooks/TODO.md) — QuantRS2 probabilistic hooks.
- [crates/tensorlogic-sklears-kernels/TODO.md](./crates/tensorlogic-sklears-kernels/TODO.md) — Kernel methods.
- [crates/tensorlogic-trustformers/TODO.md](./crates/tensorlogic-trustformers/TODO.md) — Transformer adapters.
- [crates/tensorlogic-oxicuda-backend/TODO.md](./crates/tensorlogic-oxicuda-backend/TODO.md) — OxiCUDA GPU backend (Pure Rust CUDA).
- [crates/tensorlogic-cli/TODO.md](./crates/tensorlogic-cli/TODO.md) — CLI.
- [crates/tensorlogic-py/TODO.md](./crates/tensorlogic-py/TODO.md) — Python bindings.

## References

- [README.md](./README.md) — Project overview.
- [CHANGELOG.md](./CHANGELOG.md) — Release history.
- [CLAUDE.md](./CLAUDE.md) — Project conventions.
