# Performance Guide for tensorlogic-train

This document provides guidelines for optimizing training performance in tensorlogic-train.

## Benchmark Suite

The crate includes comprehensive benchmarks to measure performance across different components:

### Available Benchmarks

1. **training_performance.rs** - End-to-end training throughput
2. **scheduler_performance.rs** - Learning rate scheduler overhead
3. **loss_performance.rs** - Loss function computation speed
4. **callback_overhead.rs** - Callback execution overhead
5. **metrics_performance.rs** - Metric computation performance

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench -p tensorlogic-train

# Run specific benchmark
cargo bench -p tensorlogic-train --bench training_performance

# Run with specific features
cargo bench -p tensorlogic-train --features structured-logging
```

## Optimization Strategies

### 1. Batch Size Optimization

**Impact**: Batch size directly affects training throughput and memory usage.

```rust
use tensorlogic_train::BatchConfig;

// Small batches: Better for limited memory, slower overall
let config = BatchConfig {
    batch_size: 16,
    ..Default::default()
};

// Large batches: Better throughput, requires more memory
let config = BatchConfig {
    batch_size: 128,
    ..Default::default()
};
```

**Guidelines**:
- Start with batch size 32 as a baseline
- Increase batch size until memory is saturated
- Use gradient accumulation for larger effective batch sizes
- Monitor gradient statistics when changing batch size

### 2. Optimizer Selection

Different optimizers have different computational costs:

| Optimizer | Memory Overhead | Computation Cost | Best For |
|-----------|----------------|------------------|----------|
| SGD | Low | Low | Fast inference, simple problems |
| Adam | Medium | Medium | General purpose, good default |
| AdamW | Medium | Medium | When weight decay is needed |
| LAMB | High | High | Large batch training |
| SAM | Very High | Very High | Best generalization (slower) |
| Sophia | High | High | Large models, fewer steps |

**Example**:
```rust
use tensorlogic_train::{SgdOptimizer, AdamOptimizer, OptimizerConfig};

// Fast training with SGD
let sgd = SgdOptimizer::new(OptimizerConfig {
    learning_rate: 0.01,
    momentum: 0.9,
    ..Default::default()
});

// Better convergence with Adam
let adam = AdamOptimizer::new(OptimizerConfig {
    learning_rate: 0.001,
    ..Default::default()
});
```

### 3. Callback Overhead Management

Callbacks add overhead to training. Use them judiciously:

```rust
use tensorlogic_train::{CallbackList, CheckpointCallback, EarlyStoppingCallback};

// Minimal callbacks for maximum speed
let mut callbacks = CallbackList::new();

// Add only essential callbacks
callbacks.add(Box::new(EarlyStoppingCallback::new(5, 1e-4)));

// Checkpoint every N epochs, not every batch
callbacks.add(Box::new(CheckpointCallback::new("checkpoints")
    .with_frequency(10)));  // Save every 10 epochs
```

**Callback Performance Characteristics**:
- **Low overhead**: EarlyStoppingCallback, ValidationCallback
- **Medium overhead**: CheckpointCallback, ReduceLROnPlateauCallback
- **High overhead**: HistogramCallback, ProfilingCallback, GradientMonitor

### 4. Loss Function Performance

Different loss functions have different computational costs:

| Loss Function | Relative Cost | Notes |
|---------------|--------------|-------|
| MSE | 1.0x | Baseline, very fast |
| CrossEntropy | 1.2x | Requires log computation |
| Focal | 1.5x | Additional exponentiation |
| Dice | 1.3x | Intersection-over-union calculation |
| PolyLoss | 1.4x | Polynomial terms |
| Triplet | 2.0x | Pairwise distance computation |

**Tip**: Use simpler loss functions during prototyping, switch to more sophisticated ones when needed.

### 5. Metric Computation

Metrics add overhead to validation:

```rust
use tensorlogic_train::{Accuracy, MetricTracker};

// Compute only essential metrics during training
let mut tracker = MetricTracker::new();
tracker.add_metric("accuracy", Box::new(Accuracy::default()));

// Add more metrics only for final evaluation
// F1, Precision, Recall can be computed after training
```

**Performance Tips**:
- Compute expensive metrics (ROC, mAP) only during validation, not training
- Use `MetricTracker` to batch metric computations
- Consider computing detailed metrics only at the end of training

### 6. Learning Rate Scheduler Overhead

Scheduler overhead is typically negligible, but some are faster:

**Fast Schedulers**: StepLR, ExponentialLR (O(1) per step)
**Medium Schedulers**: CosineAnnealingLR, PolynomialDecayLR (O(1) but more computation)
**Slower Schedulers**: ReduceLROnPlateau (requires validation metric)

### 7. Memory-Efficient Training

For large models or limited memory:

```rust
use tensorlogic_train::{GradientCheckpointConfig, MemoryEfficientTraining};

// Enable gradient checkpointing
let config = GradientCheckpointConfig {
    checkpoint_frequency: 2,  // Checkpoint every 2 layers
    preserve_rng_state: false,  // Disable for more memory savings
};

// Or use memory budget manager
use tensorlogic_train::{MemoryBudgetManager, MemorySettings};

let settings = MemorySettings {
    max_memory_mb: 4096,  // 4GB limit
    enable_garbage_collection: true,
    gc_frequency: 100,  // GC every 100 batches
};
```

### 8. Data Loading and Preprocessing

**Efficient Data Loading**:
```rust
use tensorlogic_train::BatchConfig;

// Use shuffling only when necessary
let config = BatchConfig {
    batch_size: 64,
    shuffle: false,  // Disable if data is pre-shuffled
    drop_last: true,  // Avoid partial batches
    seed: Some(42),  // Deterministic shuffling when needed
};
```

**Preprocessing Tips**:
- Pre-normalize data once before training
- Use deterministic shuffling with fixed seed
- Consider caching preprocessed data

### 9. Parallel Training Strategies

While single-GPU/CPU training is the focus, here are strategies for better parallelism:

**Gradient Accumulation** (effective large batches):
```rust
use tensorlogic_train::GradientAccumulationCallback;

// Simulate batch size of 256 with 4 accumulation steps
let callback = GradientAccumulationCallback::new(4);
// Actual batch size: 64
// Effective batch size: 64 * 4 = 256
```

**Data Parallelism** (future):
- Multiple workers for data loading
- Batch processing across devices
- Model parallelism for very large models

### 10. Profiling and Debugging

Use the built-in profiling tools:

```rust
use tensorlogic_train::{ProfilingCallback, ProfilingStats};

// Enable detailed profiling
let profiler = ProfilingCallback::new(true);  // detailed=true

// After training, analyze results:
// - Forward pass time
// - Backward pass time
// - Optimizer step time
// - Callback overhead
// - Memory usage patterns
```

## Performance Checklist

Before deploying to production, verify:

- [ ] Batch size is optimized for available memory
- [ ] Appropriate optimizer selected for the problem
- [ ] Only essential callbacks are enabled
- [ ] Metrics computed at appropriate intervals
- [ ] Gradient clipping enabled if training is unstable
- [ ] Learning rate scheduler chosen appropriately
- [ ] Memory usage profiled and optimized
- [ ] Benchmarks run to establish baseline performance

## Expected Performance

Typical training throughput (samples/second) on modern hardware:

| Model Size | Batch Size | SGD | Adam | AdamW |
|------------|-----------|-----|------|-------|
| Small (1M params) | 32 | 5000 | 4500 | 4400 |
| Small (1M params) | 128 | 15000 | 13000 | 12800 |
| Medium (10M params) | 32 | 2000 | 1800 | 1750 |
| Medium (10M params) | 128 | 6000 | 5200 | 5100 |
| Large (100M params) | 32 | 500 | 450 | 440 |
| Large (100M params) | 128 | 1500 | 1300 | 1280 |

*Note: These are approximate values for reference. Actual performance depends on model architecture, hardware, and data characteristics.*

## Common Performance Issues

### Issue: Training is Slow

**Diagnosis**:
1. Check batch size (increase if memory allows)
2. Profile callback overhead (disable unnecessary callbacks)
3. Verify data loading is not bottleneck
4. Check if expensive metrics are computed every batch

### Issue: High Memory Usage

**Diagnosis**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use gradient accumulation instead of large batches
4. Check for memory leaks in custom callbacks

### Issue: Poor Convergence

**Diagnosis**:
1. Verify learning rate is appropriate
2. Check gradient statistics for vanishing/exploding gradients
3. Try different optimizer (Adam often more stable than SGD)
4. Enable gradient clipping
5. Use learning rate warmup

## Advanced Optimization

### Mixed Precision Training

For faster training with reduced memory:

```rust
use tensorlogic_train::{MixedPrecisionTrainer, PrecisionMode};

let trainer = MixedPrecisionTrainer::new(
    PrecisionMode::FP16,  // or BF16
    loss_scale: 1024.0,
);

// Automatic mixed precision with gradient scaling
```

### Model Compression

For deployment efficiency:

```rust
use tensorlogic_train::{MagnitudePruner, Pruner, PruningConfig};

// Prune 30% of weights
let config = PruningConfig {
    pruning_ratio: 0.3,
    structured: false,
    ..Default::default()
};

let pruner = MagnitudePruner::new(config);
let (pruned_weights, mask) = pruner.prune(&weights)?;
```

### Quantization

For reduced model size:

```rust
use tensorlogic_train::{Quantizer, QuantizationConfig, BitWidth};

let config = QuantizationConfig {
    bit_width: BitWidth::Int8,
    symmetric: true,
    per_channel: true,
};

let quantizer = Quantizer::new(config)?;
let quantized = quantizer.quantize(&weights)?;
```

## Conclusion

Performance optimization is a continuous process. Start with sensible defaults, profile your training, and optimize bottlenecks iteratively. The built-in profiling and monitoring tools help identify performance issues quickly.

For questions or performance-related issues, please refer to the examples or open an issue on GitHub.
