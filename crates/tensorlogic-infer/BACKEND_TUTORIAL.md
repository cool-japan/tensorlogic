# Backend Development Tutorial

**Build Your First TensorLogic Backend in 30 Minutes**

This hands-on tutorial walks you through creating a minimal but functional TensorLogic backend from scratch.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Tutorial Overview](#tutorial-overview)
- [Part 1: Project Setup](#part-1-project-setup)
- [Part 2: Define Core Types](#part-2-define-core-types)
- [Part 3: Implement TlExecutor](#part-3-implement-tlexecutor)
- [Part 4: Testing](#part-4-testing)
- [Part 5: Optimization](#part-5-optimization)
- [Part 6: Advanced Features](#part-6-advanced-features)
- [Next Steps](#next-steps)

## Prerequisites

- Rust 1.70+ installed
- Basic understanding of Rust traits and generics
- Familiarity with tensor operations (optional but helpful)

**Estimated Time**: 30-45 minutes

## Tutorial Overview

We'll build **SimpleTensor**, a minimal CPU-based backend using `ndarray`. By the end, you'll have:

- ✅ A working `TlExecutor` implementation
- ✅ Support for basic operations (einsum, element-wise, reduce)
- ✅ Comprehensive tests
- ✅ Integration with the TensorLogic ecosystem

**What We Won't Cover** (but can be added later):
- GPU acceleration
- Automatic differentiation
- Distributed execution

## Part 1: Project Setup

### Step 1.1: Create the Project

```bash
cargo new --lib simple-tensor-backend
cd simple-tensor-backend
```

### Step 1.2: Add Dependencies

Edit `Cargo.toml`:

```toml
[package]
name = "simple-tensor-backend"
version = "0.1.0"
edition = "2021"

[dependencies]
tensorlogic-ir = "0.1"
tensorlogic-infer = "0.1"
ndarray = "0.15"
thiserror = "1.0"

[dev-dependencies]
tensorlogic-compiler = "0.1"
```

### Step 1.3: Set Up Module Structure

Create `src/lib.rs`:

```rust
//! SimpleTensor - A minimal TensorLogic backend using ndarray

mod tensor;
mod executor;
mod error;

pub use tensor::SimpleTensor;
pub use executor::SimpleExecutor;
pub use error::SimpleError;
```

## Part 2: Define Core Types

### Step 2.1: Define the Tensor Type

Create `src/tensor.rs`:

```rust
use ndarray::ArrayD;

/// A simple tensor backed by ndarray
#[derive(Clone, Debug)]
pub struct SimpleTensor {
    /// The tensor data
    pub data: ArrayD<f64>,
    /// Unique identifier for debugging
    pub id: String,
}

impl SimpleTensor {
    /// Create a new tensor with the given data
    pub fn new(id: impl Into<String>, data: ArrayD<f64>) -> Self {
        Self {
            data,
            id: id.into(),
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(id: impl Into<String>, shape: &[usize]) -> Self {
        Self::new(id, ArrayD::zeros(shape))
    }

    /// Create a tensor filled with ones
    pub fn ones(id: impl Into<String>, shape: &[usize]) -> Self {
        Self::new(id, ArrayD::ones(shape))
    }

    /// Create a tensor with specific data
    pub fn with_data(id: impl Into<String>, shape: &[usize], data: Vec<f64>) -> Self {
        let array = ArrayD::from_shape_vec(shape, data)
            .expect("Shape and data length must match");
        Self::new(id, array)
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get the number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }
}
```

### Step 2.2: Define the Error Type

Create `src/error.rs`:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SimpleError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid einsum specification: {0}")]
    InvalidEinsum(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Computation error: {0}")]
    ComputationError(String),
}
```

### Step 2.3: Define the Executor Type

Create `src/executor.rs`:

```rust
use crate::{SimpleTensor, SimpleError};

/// A simple executor for TensorLogic operations
#[derive(Default)]
pub struct SimpleExecutor {
    /// Optional: State for caching, profiling, etc.
}

impl SimpleExecutor {
    /// Create a new executor
    pub fn new() -> Self {
        Self::default()
    }
}
```

## Part 3: Implement TlExecutor

Now the fun part - implementing the trait!

### Step 3.1: Implement Element-wise Operations

Add to `src/executor.rs`:

```rust
use tensorlogic_infer::{TlExecutor, ElemOp, ReduceOp};
use ndarray::{Array, Axis, Zip};

impl TlExecutor for SimpleExecutor {
    type Tensor = SimpleTensor;
    type Error = SimpleError;

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor)
        -> Result<Self::Tensor, Self::Error>
    {
        let result_data = match op {
            ElemOp::Relu => x.data.mapv(|v| v.max(0.0)),
            ElemOp::OneMinus => x.data.mapv(|v| 1.0 - v),
            ElemOp::Sigmoid => x.data.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            _ => return Err(SimpleError::UnsupportedOperation(
                format!("Element-wise operation {:?} not supported", op)
            )),
        };

        Ok(SimpleTensor::new(
            format!("{}_op", x.id),
            result_data
        ))
    }

    fn elem_op_binary(&mut self, op: ElemOp, x: &Self::Tensor, y: &Self::Tensor)
        -> Result<Self::Tensor, Self::Error>
    {
        // Validate shapes match
        if x.shape() != y.shape() {
            return Err(SimpleError::ShapeMismatch {
                expected: x.shape().to_vec(),
                actual: y.shape().to_vec(),
            });
        }

        let result_data = match op {
            ElemOp::Add => &x.data + &y.data,
            ElemOp::Multiply => &x.data * &y.data,
            ElemOp::Max => {
                let mut result = x.data.clone();
                Zip::from(&mut result)
                    .and(&y.data)
                    .for_each(|a, &b| *a = a.max(b));
                result
            },
            ElemOp::Min => {
                let mut result = x.data.clone();
                Zip::from(&mut result)
                    .and(&y.data)
                    .for_each(|a, &b| *a = a.min(b));
                result
            },
            _ => return Err(SimpleError::UnsupportedOperation(
                format!("Binary operation {:?} not supported", op)
            )),
        };

        Ok(SimpleTensor::new(
            format!("{}_{}_op", x.id, y.id),
            result_data
        ))
    }

    // We'll add reduce and einsum next...
    fn reduce(&mut self, op: ReduceOp, x: &Self::Tensor, axes: &[usize])
        -> Result<Self::Tensor, Self::Error>
    {
        todo!("Implement in next step")
    }

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor])
        -> Result<Self::Tensor, Self::Error>
    {
        todo!("Implement in next step")
    }
}
```

### Step 3.2: Implement Reduce Operations

Add to the `TlExecutor` impl:

```rust
fn reduce(&mut self, op: ReduceOp, x: &Self::Tensor, axes: &[usize])
    -> Result<Self::Tensor, Self::Error>
{
    // Validate axes
    for &axis in axes {
        if axis >= x.data.ndim() {
            return Err(SimpleError::InvalidInput(
                format!("Axis {} out of bounds for tensor with {} dimensions",
                    axis, x.data.ndim())
            ));
        }
    }

    let mut result = x.data.clone();

    // Reduce along each axis (in reverse order to maintain axis indices)
    for &axis in axes.iter().rev() {
        result = match op {
            ReduceOp::Sum => result.sum_axis(Axis(axis)),
            ReduceOp::Max => result.map_axis(Axis(axis), |view| {
                view.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            }),
            ReduceOp::Min => result.map_axis(Axis(axis), |view| {
                view.iter().fold(f64::INFINITY, |a, &b| a.min(b))
            }),
            ReduceOp::Product => result.map_axis(Axis(axis), |view| {
                view.iter().fold(1.0, |a, &b| a * b)
            }),
        };
    }

    Ok(SimpleTensor::new(
        format!("{}_reduce", x.id),
        result
    ))
}
```

### Step 3.3: Implement Einsum (Simplified)

For this tutorial, we'll implement a simplified einsum that handles common cases:

```rust
fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor])
    -> Result<Self::Tensor, Self::Error>
{
    // Parse einsum spec
    let parts: Vec<&str> = spec.split("->").collect();
    if parts.len() != 2 {
        return Err(SimpleError::InvalidEinsum(
            format!("Invalid einsum spec: {}", spec)
        ));
    }

    let input_specs: Vec<&str> = parts[0].split(',').collect();
    let output_spec = parts[1];

    // Validate input count
    if inputs.len() != input_specs.len() {
        return Err(SimpleError::InvalidEinsum(
            format!("Expected {} inputs, got {}", input_specs.len(), inputs.len())
        ));
    }

    // Handle common cases
    match (input_specs.as_slice(), output_spec) {
        // Identity: "ij->ij"
        (["ij"], "ij") if inputs.len() == 1 => {
            Ok(inputs[0].clone())
        },

        // Matrix multiplication: "ik,kj->ij"
        (["ik", "kj"], "ij") if inputs.len() == 2 => {
            let a = &inputs[0].data;
            let b = &inputs[1].data;

            if a.ndim() != 2 || b.ndim() != 2 {
                return Err(SimpleError::ShapeMismatch {
                    expected: vec![2, 2],
                    actual: vec![a.ndim(), b.ndim()],
                });
            }

            let result = a.dot(b);
            Ok(SimpleTensor::new("matmul", result))
        },

        // Batch matrix multiplication: "bik,bkj->bij"
        (["bik", "bkj"], "bij") if inputs.len() == 2 => {
            let a = &inputs[0].data;
            let b = &inputs[1].data;

            if a.ndim() != 3 || b.ndim() != 3 {
                return Err(SimpleError::ShapeMismatch {
                    expected: vec![3, 3],
                    actual: vec![a.ndim(), b.ndim()],
                });
            }

            // Simplified batch matmul
            let batch_size = a.shape()[0];
            let m = a.shape()[1];
            let k = a.shape()[2];
            let n = b.shape()[2];

            let mut result = Array::zeros((batch_size, m, n));

            for b_idx in 0..batch_size {
                let a_slice = a.index_axis(Axis(0), b_idx);
                let b_slice = b.index_axis(Axis(0), b_idx);
                let prod = a_slice.dot(&b_slice);
                result.index_axis_mut(Axis(0), b_idx).assign(&prod);
            }

            Ok(SimpleTensor::new("batch_matmul", result))
        },

        // Element-wise product: "i,i->i"
        (["i", "i"], "i") if inputs.len() == 2 => {
            self.elem_op_binary(ElemOp::Multiply, &inputs[0], &inputs[1])
        },

        // Add more patterns as needed...
        _ => Err(SimpleError::UnsupportedOperation(
            format!("Einsum pattern '{}' not yet supported", spec)
        )),
    }
}
```

## Part 4: Testing

### Step 4.1: Write Unit Tests

Add to `src/executor.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elem_op_relu() {
        let mut exec = SimpleExecutor::new();
        let tensor = SimpleTensor::with_data(
            "test",
            &[4],
            vec![-2.0, -1.0, 0.0, 1.0]
        );

        let result = exec.elem_op(ElemOp::Relu, &tensor).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_elem_op_binary_add() {
        let mut exec = SimpleExecutor::new();
        let t1 = SimpleTensor::with_data("t1", &[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let t2 = SimpleTensor::with_data("t2", &[2, 2], vec![5.0, 6.0, 7.0, 8.0]);

        let result = exec.elem_op_binary(ElemOp::Add, &t1, &t2).unwrap();

        assert_eq!(result.data.as_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_reduce_sum() {
        let mut exec = SimpleExecutor::new();
        let tensor = SimpleTensor::with_data(
            "test",
            &[2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        // Sum along axis 0 (columns)
        let result = exec.reduce(ReduceOp::Sum, &tensor, &[0]).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_einsum_matmul() {
        let mut exec = SimpleExecutor::new();
        let a = SimpleTensor::with_data("a", &[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = SimpleTensor::with_data("b", &[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let result = exec.einsum("ik,kj->ij", &[a, b]).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        // [[1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6],
        //  [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6]]
        // = [[22, 28], [49, 64]]
        assert_eq!(result.data.as_slice().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let mut exec = SimpleExecutor::new();
        let t1 = SimpleTensor::zeros("t1", &[2, 3]);
        let t2 = SimpleTensor::zeros("t2", &[3, 2]);

        let result = exec.elem_op_binary(ElemOp::Add, &t1, &t2);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SimpleError::ShapeMismatch { .. }));
    }
}
```

### Step 4.2: Run Tests

```bash
cargo test
```

You should see all tests passing! 🎉

## Part 5: Optimization

### Step 5.1: Add Benchmarks

Create `benches/benchmarks.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use simple_tensor_backend::{SimpleExecutor, SimpleTensor};
use tensorlogic_infer::{TlExecutor, ElemOp};

fn bench_matmul(c: &mut Criterion) {
    let mut exec = SimpleExecutor::new();
    let a = SimpleTensor::ones("a", &[100, 100]);
    let b = SimpleTensor::ones("b", &[100, 100]);

    c.bench_function("matmul_100x100", |bencher| {
        bencher.iter(|| {
            exec.einsum("ik,kj->ij", &[a.clone(), b.clone()]).unwrap()
        });
    });
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
```

Add to `Cargo.toml`:

```toml
[[bench]]
name = "benchmarks"
harness = false

[dev-dependencies]
criterion = "0.5"
```

Run benchmarks:

```bash
cargo bench
```

### Step 5.2: Profile and Optimize

Use `cargo flamegraph` to find hotspots:

```bash
cargo install flamegraph
cargo flamegraph --bench benchmarks
```

Common optimizations:
- Use `ndarray`'s parallel features
- Implement memory pooling for large tensors
- Cache einsum spec parsing

## Part 6: Advanced Features

### Step 6.1: Add Profiling Support

```rust
use tensorlogic_infer::{TlProfiledExecutor, ProfileData, OpProfile};
use std::collections::HashMap;
use std::time::Instant;

impl TlProfiledExecutor for SimpleExecutor {
    fn enable_profiling(&mut self) {
        self.profiling_enabled = true;
    }

    fn disable_profiling(&mut self) {
        self.profiling_enabled = false;
    }

    fn get_profile_data(&self) -> ProfileData {
        ProfileData {
            op_profiles: self.profiles.clone(),
            memory_profile: Default::default(),
        }
    }
}
```

### Step 6.2: Add Capability Queries

```rust
use tensorlogic_infer::{TlCapabilities, BackendCapabilities, DeviceType, DType, Feature};

impl TlCapabilities for SimpleExecutor {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            devices: vec![DeviceType::CPU],
            dtypes: vec![DType::F64],
            features: vec![
                Feature::Einsum,
                Feature::ElementWise,
                Feature::Reduction,
            ],
            max_tensor_size: 1_000_000_000, // 1GB
            supports_sparse: false,
            supports_complex: false,
        }
    }
}
```

## Next Steps

Congratulations! You've built a working TensorLogic backend. Here's what to do next:

### Immediate Next Steps

1. **Add More Einsum Patterns**
   - Implement a general einsum parser
   - Support arbitrary contractions
   - Handle broadcasting

2. **Implement TlAutodiff**
   - Add gradient tracking
   - Implement backward passes
   - Support common neural network operations

3. **Optimize Performance**
   - Enable BLAS/LAPACK for matrix operations
   - Add SIMD support
   - Implement memory pooling

### Long-term Improvements

1. **GPU Support**
   - Use `cudarc` or `wgpu` for GPU operations
   - Implement device placement
   - Handle data transfers

2. **Distributed Execution**
   - Add MPI support
   - Implement tensor sharding
   - Support model parallelism

3. **Production Features**
   - Comprehensive error recovery
   - Checkpointing
   - Monitoring and observability

### Resources

- **TensorLogic Documentation**: https://docs.rs/tensorlogic-infer
- **Reference Backend**: `tensorlogic-scirs-backend` crate
- **Community**: https://github.com/cool-japan/tensorlogic/discussions

## Troubleshooting

### Common Issues

**Issue**: Tests failing with shape mismatches
```rust
// Solution: Add shape validation
if x.shape() != expected_shape {
    return Err(SimpleError::ShapeMismatch { ... });
}
```

**Issue**: Out of memory errors
```rust
// Solution: Implement chunked processing
for chunk in inputs.chunks(batch_size) {
    process_chunk(chunk)?;
}
```

**Issue**: Slow performance
```rust
// Solution: Enable ndarray's parallel features
use ndarray::parallel::prelude::*;
```

## Conclusion

You now have a solid foundation for a TensorLogic backend! The patterns you've learned here scale to more complex backends with GPU support, distributed execution, and advanced optimizations.

Happy coding! 🚀

---

**Version**: 1.0
****Last Updated**: 2025-12-16
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
