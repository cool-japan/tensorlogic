# tensorlogic-oxicuda-solver

GPU-accelerated linear system solvers and matrix decompositions for TensorLogic with pure-Rust CPU fallback.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/status-Alpha-yellow.svg)]()
[![Tests](https://img.shields.io/badge/tests-51%2F51-brightgreen.svg)]()

Provides direct and iterative solvers for dense and sparse linear systems. All solvers
have a pure-Rust CPU implementation (default) that switches to OxiCUDA GPU acceleration
when built with `--features gpu`.

## Feature flags

| Feature | Default | Effect |
|---------|---------|--------|
| `cpu`   | yes     | Pure-Rust CPU solvers (Doolittle LU, Cholesky-Banachiewicz, Modified Gram-Schmidt QR, Conjugate Gradient). Always compiled. |
| `gpu`   | no      | Enables `oxicuda-solver`, `oxicuda-driver`, `oxicuda-memory`. Requires an NVIDIA driver at runtime — no CUDA SDK needed. |

## Quick start

```rust
use tensorlogic_oxicuda_solver::{solve_lu, solve_cholesky, solve_qr_lstsq, cg_solve, SolverError};

fn main() -> Result<(), SolverError> {
    // Solve Ax = b with LU decomposition
    // A = [[2, 1], [5, 7]], b = [11, 13]
    let a = vec![2.0f32, 1.0, 5.0, 7.0];
    let b = vec![11.0f32, 13.0];
    let x = solve_lu(&a, 2, &b)?;
    // x ≈ [7.0, -3.0]

    // Positive-definite system via Cholesky
    let a_pd = vec![4.0f32, 2.0, 2.0, 3.0]; // symmetric positive-definite
    let b2 = vec![8.0f32, 7.0];
    let x2 = solve_cholesky(&a_pd, 2, &b2)?;

    // Overdetermined least-squares via QR (m=3, n=2)
    let a_over = vec![1.0f32, 1.0, 1.0, 2.0, 1.0, 3.0];
    let b3 = vec![6.0f32, 5.0, 7.0];
    let x3 = solve_qr_lstsq(&a_over, 3, 2, &b3)?;

    Ok(())
}
```

## API

| Function | Description |
|----------|-------------|
| `solve_lu(a, n, b)` | Solve n×n system Ax=b via Doolittle LU decomposition |
| `solve_cholesky(a, n, b)` | Solve n×n symmetric positive-definite system via Cholesky-Banachiewicz |
| `solve_qr_lstsq(a, m, n, b)` | Solve m×n least-squares problem via Modified Gram-Schmidt QR |
| `cg_solve(...)` | Conjugate Gradient iterative solver for large sparse SPD systems |

## Requirements

- CPU features work on all platforms (pure Rust, no native deps)
- GPU features require `--features gpu` and an NVIDIA GPU with CUDA driver at runtime

## License

Apache-2.0 — see [LICENSE](../../LICENSE) for details.
