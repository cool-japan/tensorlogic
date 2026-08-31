# tensorlogic-oxicuda-sparse — TODO

**Status**: Alpha | **Version**: 0.1.2 | **Last Updated**: 2026-08-30

## Completed

- [x] `SparseCsr` — Compressed Sparse Row matrix storage
- [x] `SparseCsr::from_triplets` — COO → CSR construction
- [x] `SparseCsr::nnz()`, `shape()`, `to_dense()`
- [x] `spmv` — Sparse matrix-vector multiply (CPU)
- [x] `spmm` — Sparse matrix-matrix multiply (CPU)
- [x] `SparseError` enum with thiserror integration
- [x] 13 passing tests (2 skipped pending CUDA hardware)
- [x] Pure-Rust CPU path (default features)
- [x] GPU stub path (feature-gated)

## Planned

- [ ] GPU path wired to real `oxicuda-sparse` kernels (currently stubbed)
- [x] `SparseCsc` — Compressed Sparse Column format
- [x] `SparseCsr::transpose()` → CSR of transposed matrix
- [ ] f64 variants for `spmv` / `spmm`
- [ ] `spmv_batched` for multiple RHS vectors
- [ ] Sparse-dense conversion utilities
