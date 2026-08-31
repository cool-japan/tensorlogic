# tensorlogic-oxicuda-rng

GPU-accelerated random number generation for TensorLogic with pure-Rust CPU fallback.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/status-Alpha-yellow.svg)]()
[![Tests](https://img.shields.io/badge/tests-59%2F59-brightgreen.svg)]()

Provides a unified `RngEngine` that seamlessly dispatches to either a pure-Rust CPU
backend (PCG-XSH-RR + Box-Muller) or an OxiCUDA GPU backend — chosen at construction
time with no API change.

## Feature flags

| Feature | Default | Effect |
|---------|---------|--------|
| `cpu`   | yes     | Pure-Rust CPU RNG via `scirs2-core::random` (PCG-XSH-RR + Box-Muller). |
| `gpu`   | no      | Enables `oxicuda-rand`, `oxicuda-driver`, `oxicuda-memory`. Requires an NVIDIA driver at runtime — no CUDA SDK needed. |

## Quick start

```rust
use tensorlogic_oxicuda_rng::{RngEngine, RngEngineKind, RngError};

fn main() -> Result<(), RngError> {
    // CPU path (no NVIDIA driver required)
    let mut rng = RngEngine::new(RngEngineKind::Cpu, 42)?;

    let mut uniform = vec![0f32; 1024];
    rng.uniform_f32(&mut uniform)?;

    let mut normal = vec![0f32; 1024];
    rng.normal_f32(&mut normal, 0.0, 1.0)?;

    let mut mask = vec![0u8; 1024];
    rng.bernoulli(&mut mask, 0.5)?;

    println!("is_gpu: {}", rng.is_gpu()); // false
    Ok(())
}
```

## API

### `RngEngine`

| Method | Description |
|--------|-------------|
| `new(kind, seed)` | Create engine for `Cpu` or `Gpu` backend |
| `kind()` | Returns `RngEngineKind` of this engine |
| `is_gpu()` | Returns `true` if running on GPU |
| `uniform_f32(out)` | Fill slice with samples ∈ [0, 1) |
| `normal_f32(out, mean, std_dev)` | Fill slice with Gaussian samples |
| `bernoulli(out, p)` | Fill `u8` slice with 0/1 Bernoulli samples |

### `RngEngineKind`

| Variant | Description |
|---------|-------------|
| `Cpu` | Pure-Rust PCG-XSH-RR generator |
| `Gpu` | OxiCUDA GPU RNG (requires `gpu` feature and NVIDIA driver) |

## Requirements

- CPU features work on all platforms (pure Rust, no native deps)
- GPU features require `--features gpu` and an NVIDIA GPU with CUDA driver at runtime

## License

Apache-2.0 — see [LICENSE](../../LICENSE) for details.
