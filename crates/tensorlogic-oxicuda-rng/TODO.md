# tensorlogic-oxicuda-rng — TODO

**Status**: Alpha | **Version**: 0.1.3 | **Last Updated**: 2026-08-31

## Completed

- [x] `RngEngine` with `Cpu` / `Gpu` variants
- [x] `uniform_f32` — uniform [0,1) via PCG-XSH-RR
- [x] `normal_f32` — Gaussian via Box-Muller
- [x] `bernoulli` — 0/1 Bernoulli samples
- [x] `RngEngineKind::as_str()` for display
- [x] `is_gpu()` query
- [x] 47 passing tests
- [x] Pure-Rust CPU path (default features)
- [x] GPU stub path (feature-gated)

## Completed (Round 7 Track D)

- [x] `uniform_f64` / `normal_f64` — f64 variants with 52-bit mantissa precision
- [x] Streaming API for large buffers (`fill_uniform_chunked`, `fill_uniform_chunked_f64`, `fill_normal_chunked`)
- [x] `RngEngine: Send + Sync` on CPU path; GPU path remains `Send + !Sync`

## Planned

- [ ] GPU path wired to real `oxicuda-rand` kernel (currently stubbed)
