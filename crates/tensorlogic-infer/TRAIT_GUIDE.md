# Trait Implementation Guide

This guide provides comprehensive instructions for implementing the TensorLogic execution traits in your backend.

## Table of Contents

- [Overview](#overview)
- [Core Traits](#core-traits)
- [Implementation Checklist](#implementation-checklist)
- [Step-by-Step Guide](#step-by-step-guide)
- [Best Practices](#best-practices)
- [Testing Your Implementation](#testing-your-implementation)
- [Common Pitfalls](#common-pitfalls)
- [Examples](#examples)

## Overview

TensorLogic defines several traits that backends must implement:

1. **TlExecutor** - Core tensor operations (required)
2. **TlAutodiff** - Automatic differentiation (optional, for training)
3. **TlBatchExecutor** - Batch processing (optional, for efficiency)
4. **TlStreamingExecutor** - Streaming execution (optional, for large datasets)
5. **TlCapabilities** - Backend capability queries (recommended)
6. **TlProfiledExecutor** - Execution profiling (optional, for debugging)
7. **TlRecoverableExecutor** - Error recovery (optional, for fault tolerance)

## Core Traits

### TlExecutor (Required)

The fundamental trait that all backends must implement.

```rust
pub trait TlExecutor {
    type Tensor;
    type Error;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor])
        -> Result<Self::Tensor, Self::Error>;

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor)
        -> Result<Self::Tensor, Self::Error>;

    fn elem_op_binary(&mut self, op: ElemOp, x: &Self::Tensor, y: &Self::Tensor)
        -> Result<Self::Tensor, Self::Error>;

    fn reduce(&mut self, op: ReduceOp, x: &Self::Tensor, axes: &[usize])
        -> Result<Self::Tensor, Self::Error>;
}
```

**Key Points:**
- `Tensor` type should represent your tensor data structure
- `Error` type should capture all possible errors
- All methods take `&mut self` to allow state tracking
- Operations should be pure (no side effects beyond state tracking)

### TlAutodiff (Optional, for Training)

Extends `TlExecutor` with automatic differentiation.

```rust
pub trait TlAutodiff: TlExecutor {
    type Tape;

    fn forward(&mut self, graph: &EinsumGraph)
        -> Result<Self::Tensor, Self::Error>;

    fn backward(&mut self, graph: &EinsumGraph, loss: &Self::Tensor)
        -> Result<Self::Tape, Self::Error>;
}
```

**Key Points:**
- `Tape` represents recorded computation for backpropagation
- `forward` executes the graph and records operations
- `backward` computes gradients using the tape
- Must track intermediate values for gradient computation

## Implementation Checklist

### Minimum Viable Implementation (TlExecutor only)

- [ ] Define `Tensor` type (e.g., wrapper around ndarray)
- [ ] Define `Error` type (use thiserror for clean errors)
- [ ] Implement `einsum` (at least basic cases)
- [ ] Implement `elem_op` (Relu, OneMinus, at minimum)
- [ ] Implement `elem_op_binary` (Add, Multiply, at minimum)
- [ ] Implement `reduce` (Sum, Max, at minimum)
- [ ] Write unit tests for each operation
- [ ] Handle edge cases (empty tensors, mismatched shapes)

### Production-Ready Implementation

- [ ] All minimum viable items ✓
- [ ] Implement `TlAutodiff` for training support
- [ ] Implement `TlBatchExecutor` for batch processing
- [ ] Implement `TlCapabilities` for feature detection
- [ ] Comprehensive error handling
- [ ] Performance optimization (SIMD, parallelization)
- [ ] Memory efficiency (pooling, caching)
- [ ] Integration tests with real graphs
- [ ] Benchmarks against reference implementation

## Step-by-Step Guide

### Step 1: Set Up Your Crate

```bash
cargo new --lib my-tensorlogic-backend
cd my-tensorlogic-backend
```

Add dependencies to `Cargo.toml`:

```toml
[dependencies]
tensorlogic-ir = "0.1"
tensorlogic-infer = "0.1"
thiserror = "1.0"
# Your tensor library (e.g., ndarray)
ndarray = "0.15"
```

### Step 2: Define Core Types

```rust
use ndarray::{Array, ArrayD};
use thiserror::Error;

/// Your tensor type
#[derive(Clone, Debug)]
pub struct MyTensor {
    data: ArrayD<f64>,
    id: String,
}

/// Your error type
#[derive(Error, Debug)]
pub enum MyError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Invalid einsum specification: {0}")]
    InvalidEinsum(String),

    #[error("Operation failed: {0}")]
    OperationFailed(String),
}

/// Your executor
pub struct MyExecutor {
    // State tracking, caching, etc.
}
```

### Step 3: Implement TlExecutor

```rust
use tensorlogic_infer::{TlExecutor, ElemOp, ReduceOp};

impl TlExecutor for MyExecutor {
    type Tensor = MyTensor;
    type Error = MyError;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor])
        -> Result<Self::Tensor, Self::Error>
    {
        // Parse einsum specification
        let (input_specs, output_spec) = parse_einsum_spec(spec)?;

        // Validate inputs
        if inputs.len() != input_specs.len() {
            return Err(MyError::InvalidEinsum(
                format!("Expected {} inputs, got {}", input_specs.len(), inputs.len())
            ));
        }

        // Execute einsum operation
        let result = execute_einsum_op(input_specs, output_spec, inputs)?;

        Ok(result)
    }

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor)
        -> Result<Self::Tensor, Self::Error>
    {
        let result_data = match op {
            ElemOp::Relu => x.data.mapv(|v| v.max(0.0)),
            ElemOp::OneMinus => x.data.mapv(|v| 1.0 - v),
            ElemOp::Sigmoid => x.data.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            // Add other operations...
            _ => return Err(MyError::OperationFailed(
                format!("Unsupported operation: {:?}", op)
            )),
        };

        Ok(MyTensor {
            data: result_data,
            id: format!("{}_op", x.id)
        })
    }

    fn elem_op_binary(&mut self, op: ElemOp, x: &Self::Tensor, y: &Self::Tensor)
        -> Result<Self::Tensor, Self::Error>
    {
        // Validate shapes are compatible
        if x.data.shape() != y.data.shape() {
            return Err(MyError::ShapeMismatch {
                expected: x.data.shape().to_vec(),
                actual: y.data.shape().to_vec(),
            });
        }

        let result_data = match op {
            ElemOp::Add => &x.data + &y.data,
            ElemOp::Multiply => &x.data * &y.data,
            ElemOp::Max => {
                let mut result = x.data.clone();
                ndarray::Zip::from(&mut result)
                    .and(&y.data)
                    .for_each(|a, &b| *a = a.max(b));
                result
            },
            // Add other operations...
            _ => return Err(MyError::OperationFailed(
                format!("Unsupported binary operation: {:?}", op)
            )),
        };

        Ok(MyTensor {
            data: result_data,
            id: format!("{}_{}_op", x.id, y.id)
        })
    }

    fn reduce(&mut self, op: ReduceOp, x: &Self::Tensor, axes: &[usize])
        -> Result<Self::Tensor, Self::Error>
    {
        let result_data = match op {
            ReduceOp::Sum => {
                let mut result = x.data.clone();
                for &axis in axes.iter().rev() {
                    result = result.sum_axis(ndarray::Axis(axis));
                }
                result
            },
            ReduceOp::Max => {
                let mut result = x.data.clone();
                for &axis in axes.iter().rev() {
                    result = result.map_axis(ndarray::Axis(axis), |view| {
                        view.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                    });
                }
                result
            },
            // Add other operations...
            _ => return Err(MyError::OperationFailed(
                format!("Unsupported reduce operation: {:?}", op)
            )),
        };

        Ok(MyTensor {
            data: result_data,
            id: format!("{}_reduce", x.id)
        })
    }
}
```

### Step 4: Implement TlAutodiff (Optional)

```rust
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::EinsumGraph;

pub struct ComputationTape {
    // Store intermediate values and operations
    operations: Vec<TapeEntry>,
}

impl TlAutodiff for MyExecutor {
    type Tape = ComputationTape;

    fn forward(&mut self, graph: &EinsumGraph)
        -> Result<Self::Tensor, Self::Error>
    {
        // Enable gradient tracking
        self.gradient_mode = true;

        // Execute graph and record operations
        let mut tape = ComputationTape::new();

        for node in &graph.nodes {
            let result = match &node.op {
                OpType::Einsum { spec } => {
                    let inputs = self.get_inputs(&node.inputs)?;
                    let output = self.einsum(spec, &inputs)?;
                    tape.record(TapeEntry::Einsum {
                        spec: spec.clone(),
                        inputs: inputs.clone(),
                        output: output.clone()
                    });
                    output
                },
                // Handle other operation types...
                _ => unimplemented!(),
            };

            self.store_result(node.id, result)?;
        }

        self.tape = Some(tape);
        self.get_output()
    }

    fn backward(&mut self, graph: &EinsumGraph, loss: &Self::Tensor)
        -> Result<Self::Tape, Self::Error>
    {
        let tape = self.tape.take()
            .ok_or(MyError::OperationFailed("No forward pass recorded".into()))?;

        // Initialize gradient with respect to loss
        let mut gradients = HashMap::new();
        gradients.insert(loss.id.clone(), loss.clone());

        // Backpropagate through tape in reverse order
        for entry in tape.operations.iter().rev() {
            match entry {
                TapeEntry::Einsum { spec, inputs, output } => {
                    let grad_output = gradients.get(&output.id)
                        .ok_or(MyError::OperationFailed("Missing gradient".into()))?;

                    // Compute gradients for inputs using einsum derivatives
                    let input_grads = self.einsum_backward(spec, inputs, grad_output)?;

                    for (input, grad) in inputs.iter().zip(input_grads.iter()) {
                        gradients.insert(input.id.clone(), grad.clone());
                    }
                },
                // Handle other operation types...
                _ => unimplemented!(),
            }
        }

        self.gradients = Some(gradients);
        Ok(tape)
    }
}
```

### Step 5: Implement TlBatchExecutor (Optional)

```rust
use tensorlogic_infer::{TlBatchExecutor, BatchResult};
use std::collections::HashMap;

impl TlBatchExecutor for MyExecutor {
    fn execute_batch(
        &mut self,
        graph: &EinsumGraph,
        batch_inputs: Vec<HashMap<String, Self::Tensor>>,
    ) -> Result<BatchResult<Self::Tensor>, Self::Error> {
        let start = std::time::Instant::now();
        let mut results = Vec::new();

        for inputs in batch_inputs {
            // Execute graph for each input
            let output = self.forward_with_inputs(graph, &inputs)?;
            results.push(output);
        }

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(BatchResult {
            outputs: results,
            total_time_ms,
            metadata: HashMap::new(),
        })
    }

    fn execute_batch_parallel(
        &mut self,
        graph: &EinsumGraph,
        batch_inputs: Vec<HashMap<String, Self::Tensor>>,
        num_threads: Option<usize>,
    ) -> Result<BatchResult<Self::Tensor>, Self::Error> {
        use rayon::prelude::*;

        let num_threads = num_threads.unwrap_or_else(num_cpus::get);
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let start = std::time::Instant::now();

        // Parallel execution
        let results: Result<Vec<_>, _> = batch_inputs
            .par_iter()
            .map(|inputs| {
                // Each thread needs its own executor
                let mut exec = self.clone();
                exec.forward_with_inputs(graph, inputs)
            })
            .collect();

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(BatchResult {
            outputs: results?,
            total_time_ms,
            metadata: HashMap::new(),
        })
    }

    fn optimal_batch_size(&self, _graph: &EinsumGraph) -> usize {
        // Heuristic: balance memory and parallelism
        let available_threads = num_cpus::get();
        let memory_per_sample_mb = 10.0; // Estimate
        let available_memory_mb = 1000.0; // Estimate

        let max_by_memory = (available_memory_mb / memory_per_sample_mb) as usize;
        let max_by_threads = available_threads * 2; // 2x for pipelining

        max_by_memory.min(max_by_threads).max(1)
    }
}
```

## Best Practices

### Error Handling

1. **Use thiserror for clean error types**:
```rust
#[derive(Error, Debug)]
pub enum MyError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
}
```

2. **Provide detailed error messages**:
```rust
if inputs.is_empty() {
    return Err(MyError::InvalidInput(
        "Expected at least one input tensor".into()
    ));
}
```

3. **Handle all edge cases**:
- Empty tensors
- Mismatched shapes
- Invalid specifications
- Out of memory

### Performance Optimization

1. **Use SIMD when possible**:
```rust
#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;
```

2. **Implement memory pooling**:
```rust
struct TensorPool {
    pool: Vec<ArrayD<f64>>,
}

impl TensorPool {
    fn allocate(&mut self, shape: &[usize]) -> ArrayD<f64> {
        self.pool.pop().unwrap_or_else(|| ArrayD::zeros(shape))
    }

    fn deallocate(&mut self, tensor: ArrayD<f64>) {
        self.pool.push(tensor);
    }
}
```

3. **Cache einsum parsing**:
```rust
use std::collections::HashMap;

struct EinsumCache {
    cache: HashMap<String, ParsedEinsum>,
}
```

### Memory Management

1. **Use Cow for zero-copy when possible**:
```rust
use std::borrow::Cow;

fn process<'a>(&self, data: Cow<'a, ArrayD<f64>>) -> Cow<'a, ArrayD<f64>> {
    if needs_modification {
        Cow::Owned(data.into_owned().mapv(|x| x * 2.0))
    } else {
        data // Zero-copy
    }
}
```

2. **Implement Drop for resource cleanup**:
```rust
impl Drop for MyExecutor {
    fn drop(&mut self) {
        // Clean up GPU resources, file handles, etc.
        self.cleanup_resources();
    }
}
```

## Testing Your Implementation

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_execution() {
        let mut exec = MyExecutor::new();
        let t1 = MyTensor::ones(vec![2, 3]);
        let t2 = MyTensor::ones(vec![2, 3]);

        let result = exec.elem_op_binary(ElemOp::Add, &t1, &t2).unwrap();

        assert_eq!(result.data.shape(), &[2, 3]);
        assert!(result.data.iter().all(|&x| (x - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_shape_mismatch_error() {
        let mut exec = MyExecutor::new();
        let t1 = MyTensor::ones(vec![2, 3]);
        let t2 = MyTensor::ones(vec![3, 2]);

        let result = exec.elem_op_binary(ElemOp::Add, &t1, &t2);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MyError::ShapeMismatch { .. }));
    }
}
```

### Integration Tests

Create `tests/integration_test.rs`:

```rust
use my_backend::MyExecutor;
use tensorlogic_infer::TlExecutor;
use tensorlogic_compiler::compile;

#[test]
fn test_full_graph_execution() {
    // Compile a TLExpr to EinsumGraph
    let expr = /* ... */;
    let graph = compile(&expr, &context).unwrap();

    // Execute with your backend
    let mut executor = MyExecutor::new();
    let inputs = /* ... */;
    let outputs = executor.forward(&graph, &inputs).unwrap();

    // Verify results
    assert_eq!(outputs.shape(), expected_shape);
}
```

## Common Pitfalls

### 1. **Forgetting to clone tensors**

❌ **Wrong**:
```rust
fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
    x.data.mapv_inplace(|v| v.max(0.0)); // Mutates input!
    Ok(x.clone())
}
```

✅ **Correct**:
```rust
fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
    let result_data = x.data.mapv(|v| v.max(0.0)); // Creates new array
    Ok(MyTensor { data: result_data, id: format!("{}_relu", x.id) })
}
```

### 2. **Not handling broadcast correctly**

❌ **Wrong**:
```rust
fn elem_op_binary(&mut self, op: ElemOp, x: &Self::Tensor, y: &Self::Tensor)
    -> Result<Self::Tensor, Self::Error>
{
    // Assumes shapes are identical
    Ok(MyTensor { data: &x.data + &y.data, id: "result".into() })
}
```

✅ **Correct**:
```rust
fn elem_op_binary(&mut self, op: ElemOp, x: &Self::Tensor, y: &Self::Tensor)
    -> Result<Self::Tensor, Self::Error>
{
    // Check if broadcast is needed
    if !are_shapes_compatible(&x.data.shape(), &y.data.shape()) {
        return Err(MyError::ShapeMismatch { /* ... */ });
    }

    let result_data = broadcast_and_apply(&x.data, &y.data, |a, b| a + b)?;
    Ok(MyTensor { data: result_data, id: "result".into() })
}
```

### 3. **Memory leaks in gradient computation**

✅ **Always clean up**:
```rust
impl Drop for MyExecutor {
    fn drop(&mut self) {
        self.clear_tape();
        self.clear_gradients();
    }
}
```

### 4. **Not validating einsum specifications**

✅ **Validate early**:
```rust
fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor])
    -> Result<Self::Tensor, Self::Error>
{
    // Validate specification format
    if !is_valid_einsum_spec(spec) {
        return Err(MyError::InvalidEinsum(format!("Invalid spec: {}", spec)));
    }

    // Validate input count
    let expected_inputs = count_einsum_inputs(spec);
    if inputs.len() != expected_inputs {
        return Err(MyError::InvalidEinsum(
            format!("Expected {} inputs, got {}", expected_inputs, inputs.len())
        ));
    }

    // ... rest of implementation
}
```

## Examples

### Complete Minimal Backend

See `examples/minimal_backend.rs` in the repository for a complete minimal implementation.

### Production Backend

See the `tensorlogic-scirs-backend` crate for a full production implementation using SciRS2.

## Next Steps

1. Implement the basic `TlExecutor` trait
2. Write comprehensive tests
3. Benchmark against reference implementation
4. Add optional traits (TlAutodiff, TlBatchExecutor, etc.)
5. Optimize for your target platform
6. Submit your backend to the TensorLogic ecosystem!

## Getting Help

- **Documentation**: https://docs.rs/tensorlogic-infer
- **Examples**: https://github.com/cool-japan/tensorlogic/tree/main/examples
- **Issues**: https://github.com/cool-japan/tensorlogic/issues
- **Discussions**: https://github.com/cool-japan/tensorlogic/discussions

---

**Version**: 1.0
**Last Updated**: 2025-11-06
**Part of**: [TensorLogic Ecosystem](https://github.com/cool-japan/tensorlogic)
