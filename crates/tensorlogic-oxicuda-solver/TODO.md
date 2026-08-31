# tensorlogic-oxicuda-solver — TODO

**Status**: Alpha | **Version**: 0.1.2 | **Last Updated**: 2026-08-30

## Completed

- [x] `solve_lu` — Doolittle LU decomposition (CPU)
- [x] `solve_cholesky` — Cholesky-Banachiewicz decomposition (CPU)
- [x] `solve_qr_lstsq` — Modified Gram-Schmidt QR least-squares (CPU)
- [x] `cg_solve` — Conjugate Gradient iterative solver (CPU)
- [x] `SolverError` enum with thiserror integration
- [x] 35 passing tests
- [x] Pure-Rust CPU path (default features)
- [x] GPU stub path (feature-gated)

## Planned

- [ ] GPU path wired to real `oxicuda-solver` kernels (currently stubbed)
- [ ] `solve_lu_f64` / `solve_cholesky_f64` — f64 variants
- [ ] Sparse system support (CG already supports sparse via SpMV)
- [x] Preconditioned CG (diagonal/incomplete Cholesky preconditioner)
- [ ] Banded matrix LU for tridiagonal systems
