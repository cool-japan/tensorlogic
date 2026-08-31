# TensorLogic OxiCUDA Backend — TODO

**Status**: Research Preview | **Version**: 0.1.3 | **Last Updated**: 2026-08-31
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

GPU tensor execution backend for TensorLogic, built on the COOLJAPAN **OxiCUDA** ecosystem
(`~/work/oxicuda/`, crates.io `oxicuda-*` 0.1.2). Pure Rust. Only requires the NVIDIA driver
at runtime — no CUDA SDK, no nvcc, no C/C++ toolchain.

## Completed

- [x] Crate scaffold registered in workspace.
- [x] `default-features = []` (pure Rust) — no GPU dependency on the default build path.
- [x] `gpu` feature wiring: `oxicuda-backend` + `oxicuda-blas` (crates.io 0.1.2).
- [x] `OxiCudaExecutor` / `OxiCudaTensor` / `OxiCudaBackendError` skeleton.
- [x] `TlExecutor` trait implementation (stub + gpu split; disabled path returns `BackendDisabled`, gpu path validates shapes and routes `einsum("ij,jk->ik", ...)` to a matmul helper).
- [x] Compile-smoke tests for both feature states.

## v0.1.x Stabilization

- [x] **Wire real SGEMM through `oxicuda-blas`.** Replaced the `Unsupported` stub in `matmul_ij_jk_ik` with the full pipeline:
  - [x] `GpuState` holding `Arc<oxicuda_driver::Context>` + `Stream` + `BlasHandle`.
  - [x] Host → device copy for both operands (`oxicuda_memory::DeviceBuffer::from_host`).
  - [x] `MatrixDesc` / `MatrixDescMut` construction for A (M×K), B (K×N), C (M×N), row-major.
  - [x] `oxicuda_blas::level3::gemm_api::gemm` SGEMM call (alpha=1.0, beta=0.0).
  - [x] Stream synchronize + device → host copy of C.
  - [x] `#[ignore]` integration test gated behind `TENSORLOGIC_GPU_TESTS=1` env var (tests/gpu_sgemm.rs).
  - [x] `From<CudaError>` and `From<BlasError>` conversions in error.rs; `DimensionOverflow` variant for safe u32 casts.
- [x] **Per-crate README + CHANGELOG content pass** (planned 2026-04-17)
  - **Goal:** Bring `README.md` up to post-Round-3 state (remove stale "stub" references, add supported-ops table, feature-flag table with `fft` row, performance pointer). Add `CHANGELOG.md` in Keep-a-Changelog format covering 0.1.0–0.1.3.
  - **Design:** README rewrite ~80 LoC. New `CHANGELOG.md`: `[Unreleased]` (Items E/F/H), `[0.1.3]` (Round 3 ops), `[0.1.2]` (Round 2 einsum+elem), `[0.1.1]` (Round 1 SGEMM), `[0.1.0]` (scaffold).
  - **Files:** `README.md` (rewrite), `CHANGELOG.md` (NEW)
  - **Tests:** `cargo doc --no-deps -p tensorlogic-oxicuda-backend`
  - **Risk:** Run G-writer after Items E/F/H land so the README accurately reflects shipped code.

## v0.2.0 / Future Work

- [x] Combined: `oxicuda-broader-einsum` + `oxicuda-elementwise-ops` + `oxicuda-reductions` (planned 2026-04-17)
  - **Goal:** Replace the hard-coded `"ij,jk->ik"` einsum MVP with a parser that dispatches into the actual upstream `oxicuda-blas` surface; wire the elementwise/reduction methods that currently return `Unsupported` to the kernels upstream **really** ships today; add real per-axis reduction in the wrapper since upstream is scalar-only.
  - **Design (Option A — wrapper-only, honest):** Einsum dispatch: parse spec into (in_labels, out_labels); map `"ij,jk->ik"` → gemm (MatrixDesc/MatrixDescMut); `"bij,bjk->bik"` → gemm_strided_batched (raw CUdeviceptr — care at FFI boundary); identity/pure-transpose/single-sum-axis → dedicated helpers; anything else → UnsupportedSpec error. Elementwise unary (elem_op): wire only the 6 functions upstream exposes: relu, gelu, sigmoid, silu, tanh_activation, scale. Anything else → typed Unsupported. Elementwise binary (elem_op_binary): wire only add, mul, fused_add_relu, fused_scale_add. Anything else → typed Unsupported. Reductions (reduce): upstream is scalar-only. Implement per-axis in the wrapper via transpose-pack-loop: (1) transpose target axis to last dim, (2) pack contiguous rows into temp DeviceBuffer, (3) loop rows calling upstream scalar reduction, (4) write each scalar back. No upstream changes this run; upstream coverage gap filed as Proposed follow-up requiring user approval.
  - **Files:** `src/executor.rs` (replace stubs at lines 172–208; keep GPU-disabled path at 130–163 untouched); `src/einsum.rs` (NEW — spec parser + dispatch); `src/elem_ops.rs` (NEW — unary + binary dispatch tables); `src/reduce.rs` (NEW — per-axis transpose-pack-loop); `src/lib.rs` (declare new modules); `src/error.rs` (add UnsupportedSpec, UnsupportedUnary, UnsupportedBinary variants); `Cargo.toml` (no new deps — oxicuda-blas 0.1.2 already present).
  - **Prerequisites:** none in this repo.
  - **Tests:** `tests/elem_ops_smoke.rs` (NEW): feature-disabled → BackendDisabled; with TENSORLOGIC_GPU_TESTS=1 + #[ignore] round-trips each supported unary/binary vs CPU oracle. `tests/reduce_smoke.rs` (NEW): per-axis sum/max over 3D [2,3,4] tensor, three axis choices, CPU oracle. `tests/einsum_specs.rs` (NEW): bij,bjk->bik batched matmul [2,3,4]×[2,4,5]; identity spec; rejected unknown spec → UnsupportedSpec.
  - **Risk:** Transpose-pack-loop adds extra device memory + kernel launches. Mitigation: document perf in module docstring; native per-axis kernel is the proper fix and filed as follow-up. FFI on gemm_strided_batched (raw CUdeviceptr) — extract a typed helper converting &DeviceBuffer once.
  - **Proposed follow-ups (require user approval to touch ~/work/oxicuda):** Add upstream unary kernels: OneMinus, Exp, Log, Sqrt, Abs, Neg. Add upstream binary kernels: Sub, Div, Min, Max, Eq, Lt, Gt, Lte, Gte, OrMax, OrProbSum, Nand, Nor, Xor. Add native per-axis reduction kernel to oxicuda-blas.
- [x] **`TlAutodiff` impl over EinsumGraph (forward + backward)** (planned 2026-04-17)
  - **Goal:** Full `TlAutodiff` trait impl on `OxiCudaExecutor` — real forward topo-walk + backward wengert tape. Coverage: matmul 2D/3D, identity, transpose, ReduceSum/Max per-axis, and elementwise ops with known derivatives (Relu, Sigmoid, Tanh, Silu, Exp, Log, Sqrt, Neg, Abs, OneMinus, Add, Sub, Mul, Div). Unsupported ops return typed `UnsupportedAutodiffOp` error.
  - **Design:** `OxiCudaTape { entries: Vec<TapeEntry>, gradients: HashMap<NodeId, OxiCudaTensor> }`. `TapeEntry` saves `op`, `inputs`, `output_shape`, `saved_inputs`. `forward` topo-walks EinsumGraph. `backward` reverse-iterates tape; uses existing Round-3 kernels for all gradient expressions; uses host-side broadcast for ReduceSum/ReduceMax gradients (perf TODO, document). New `src/autodiff.rs` ~600 LoC.
  - **Files:** `src/autodiff.rs` (NEW), `src/lib.rs`, `src/error.rs`, `src/einsum.rs` (additive gradient specs only)
  - **Tests:** `tests/autodiff_smoke.rs` — CUDA-gated: forward matmul, backward matmul (finite-diff check), backward Relu, backward Sigmoid scalar, unsupported-op error, ReduceSum gradient.
  - **Risk:** `EinsumGraph` traversal API — read `tensorlogic-scirs-backend/src/autodiff.rs` first, mirror pattern. Host broadcast is correct but slow; document explicitly.
- [ ] **Multi-GPU scheduling hook** — integrate with `oxicuda-dist-infer` / `oxicuda-dist-train` for Distributed training (unblocks the v0.2.0 "NCCL / multi-GPU scheduling" TODO at the root level).
- [ ] **LLM/inference fast path** — route `tensorlogic-trustformers` inference through `oxicuda-infer` / `oxicuda-lm` for production-grade LLM hosting.
- [x] **FFT sub-feature: `oxicuda-fft` wrapper** (planned 2026-04-17)
  - **Goal:** Wire `oxicuda-fft` through `tensorlogic-oxicuda-backend` behind a `fft` sub-feature (off by default; `fft` implies `gpu`). Provides `forward_c2c_1d` / `inverse_c2c_1d` accepting host-slice `&[Complex<f32>]`.
  - **Design:** New `fft` Cargo feature (`fft = ["gpu", "dep:oxicuda-fft"]`). New `src/fft.rs` with `OxiCudaFftPlan`, `forward_c2c_1d`, `inverse_c2c_1d` — cfg-gated on both `gpu` and `fft`; stub path returns `FftDisabled`. Add `FftDisabled` + `Fft(String)` + `From<FftError>` in `error.rs`. Expose `gpu_state()` accessor on `OxiCudaExecutor` for shared context. Extend workspace `[patch.crates-io]` with `oxicuda-fft` path.
  - **Files:** `src/fft.rs` (NEW), `src/error.rs`, `src/executor.rs`, `src/lib.rs`, `Cargo.toml` (crate + workspace)
  - **Tests:** `tests/fft_smoke.rs` — CUDA-gated `#[ignore]` round-trip + disabled-feature test.
  - **Risk:** Upstream `FftHandle` constructor signature may differ; implement to actual API, not assumed.
- [x] **Non-NVIDIA backends / BackendKind enum.** `BackendKind` (7-variant) landed in `tensorlogic-infer` with `Scirs`/`OxiCuda` live and 5 doc-hidden stubs for Round 6 (`Metal`, `Vulkan`, `Rocm`, `Webgpu`, `Levelzero`). Implements `std::str::FromStr`. Methods: `default_backend()`, `from_env()`, `is_gpu()`, `supports_autodiff()`, `validate()`.
- [x] **Benchmark suite: OxiCUDA GPU vs scirs CPU matmul** (planned 2026-04-17)
  - **Goal:** `criterion` bench comparing GPU SGEMM vs CPU matmul across shapes `[64, 256, 1024, 2048]`. GPU group skips gracefully if no NVIDIA driver.
  - **Design:** Add `criterion` + `tensorlogic-scirs-backend` dev-deps. `[[bench]] name = "gpu_vs_cpu_matmul" harness = false`. Two groups in bench file: `gpu_matmul_square` (feature-gated, runtime-skipped if no GPU) and `cpu_matmul_square`. `Throughput::Elements(n³)` for FLOPS proxy.
  - **Files:** `benches/gpu_vs_cpu_matmul.rs` (NEW), `Cargo.toml` (dev-deps + `[[bench]]`), `src/lib.rs` (one doc note)
  - **Tests:** `cargo bench --no-run --features gpu` compile-check.
  - **Risk:** `tensorlogic-scirs-backend` dev-dep must not create a circular dep; verify with `cargo metadata`.
- [ ] **Quantization path** — integrate with `oxicuda-quant` for int8 / fp8 inference once the upstream API stabilizes.

### Pending

- [x] **sparse/solver/rng sub-feature crates** — `tensorlogic-oxicuda-sparse`, `tensorlogic-oxicuda-solver`, `tensorlogic-oxicuda-rng` created. CPU paths complete; GPU paths stubbed. Umbrella features `sparse`, `solver`, `rng`, `full-advanced` wired in `crates/tensorlogic/Cargo.toml`.
- [x] **native-broadcast / fill kernels in autodiff** — `native-broadcast = ["gpu"]` feature in this crate. Under this feature, `broadcast_to_shape` and Mean-divisor `fill` use native GPU kernels from `oxicuda-blas` (upload→kernel→readback). `BroadcastTemplate` PTX uses stride-zero trick, MAX_RANK=8.

### Round 5 (2026-04-17) Achievements
- BackendKind enum landed in tensorlogic-infer (Scirs/OxiCuda live; 5 non-NVIDIA stubs for Round 6)
- oxicuda-blas: native `fill` + `broadcast_axes` kernels (BroadcastTemplate, stride-zero trick, MAX_RANK=8)
- tensorlogic-oxicuda-sparse/solver/rng wrapper crates created (CPU paths complete; GPU paths stubbed)
- Umbrella features: `sparse`, `solver`, `rng`, `full-advanced`
- Autodiff: `native-broadcast` feature gates GPU kernel paths for ReduceSum/Mean backward
- LRU memo cache correctness fix in tensorlogic-infer (deque-front strategy, O(1), tie-safe)
- No-warnings pass: 14 clippy issues fixed across 7 files (cfg-gating, range contains, needless return, dead code, should_implement_trait)
- Full workspace: **7,178 tests, 7,178 passed**

- [x] `tensorlogic-backend-wire-new-kernels` (planned 2026-04-17)
  - **Goal:** Wire all new OxiCUDA elementwise and reduction ops into the TensorLogic GPU backend. In `src/elem_ops.rs`, extend `dispatch_unary` to cover `Gelu, Silu, Tanh, Neg, Abs, Sqrt, Rsqrt, Exp, Log, Ceil, Floor, HardSigmoid, HardSwish, Softplus, LeakyRelu, OneMinus` and `dispatch_binary` to cover `Sub, Div, Pow, Min, Max, CmpEq, CmpNe, CmpLt, CmpGt, CmpLe, CmpGe, OrMax, OrProbSum, Nand, Nor, Xor`. In `src/reduce.rs`, delete `cpu_move_axis_to_last` and the CPU loop in `reduce_one_axis`; compute `(outer, axis_len, inner)` from input shape and axis index; call `oxicuda_blas::reduction::axis::reduce_axis`.
  - **Prerequisites:** `oxicuda-unary-wire-and-extend`, `oxicuda-binary-extensions`, `oxicuda-axis-reduction` (all in ~/work/oxicuda/) must be complete.
  - **Files:** `src/elem_ops.rs`, `src/reduce.rs`
  - **Tests:** Add `tests/elem_ops_native.rs` (CUDA-gated, one test per new op); `tests/reduce_axis_native.rs` (CUDA-gated, Sum/Max/Min/Mean per axis of 2D and 3D tensors).

## Policy notes

- **Pure Rust Policy**: Default build must have zero GPU/native deps. `gpu` feature is the single opt-in.
- **No `unwrap()` / `expect()`**: enforced by `#![deny(clippy::unwrap_used, clippy::expect_used)]` in `src/lib.rs`.
- **Edition**: Crate uses `edition.workspace = true` (2021) currently; OxiCUDA's 2024 public API does not leak through the wrapper. If a future OxiCUDA upgrade requires 2024 at the call site, switch this crate's manifest to `edition = "2024"` (downstream crates stay on 2021).
- **No raw CUDA SDK / cudarc / scirs2-core GPU** — OxiCUDA only, per COOLJAPAN Pure Rust Policy extension (user directive, 2026-04-15).
