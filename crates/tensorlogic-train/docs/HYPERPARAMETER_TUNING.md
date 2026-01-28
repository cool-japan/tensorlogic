# Hyperparameter Tuning Guide

Practical guide to tuning hyperparameters for optimal training performance.

## Overview

This guide provides systematic approaches to tuning the most impactful hyperparameters in `tensorlogic-train`. We focus on practical, empirically-validated strategies rather than exhaustive grid searches.

## Table of Contents

1. [Learning Rate](#learning-rate)
2. [Batch Size](#batch-size)
3. [Optimizer Selection](#optimizer-selection)
4. [Learning Rate Schedules](#learning-rate-schedules)
5. [Regularization](#regularization)
6. [Early Stopping](#early-stopping)
7. [Practical Workflow](#practical-workflow)

---

## Learning Rate

**Impact**: ⭐⭐⭐⭐⭐ (Highest)
**Difficulty**: Medium
**Time to tune**: 30-60 minutes

### Why It Matters

Learning rate is the single most important hyperparameter. Too high causes divergence, too low wastes time.

### Recommended Approach: LearningRateFinder

Use the built-in LR finder to automatically discover optimal learning rate:

```rust
use tensorlogic_train::{LearningRateFinder, CallbackList};

let mut callbacks = CallbackList::new();
callbacks.add(Box::new(LearningRateFinder::new(
    1e-7,  // start_lr
    1.0,   // end_lr
    100,   // num_steps
).with_exponential_scaling()));

// After training, inspect the LR vs loss curve
// Optimal LR is typically where loss decreases fastest
```

### Manual Tuning Strategy

If not using LR finder, try this sequence:

**1. Start conservatively:**
```rust
let lr = 1e-3; // Good default for Adam/AdamW
```

**2. Test 3-5 values spanning 2 orders of magnitude:**
```rust
let lrs = vec![1e-4, 3e-4, 1e-3, 3e-3, 1e-2];
```

**3. Train for 5-10 epochs each, pick the fastest converging:**

### Optimizer-Specific Recommendations

| Optimizer | Typical Range | Good Default | Notes |
|-----------|--------------|--------------|-------|
| **SGD** | 1e-2 to 1e-1 | 1e-2 | Needs larger LR than adaptive methods |
| **SGD + Momentum** | 1e-2 to 1e-1 | 3e-2 | Momentum allows higher LR |
| **Adam** | 1e-4 to 1e-3 | 1e-3 | Most robust choice |
| **AdamW** | 1e-4 to 1e-3 | 1e-3 | Same as Adam |
| **RMSprop** | 1e-4 to 1e-3 | 1e-3 | Similar to Adam |
| **Adagrad** | 1e-2 to 1e-1 | 1e-2 | Learning rate decays automatically |

### Signs of Incorrect LR

**Too High:**
- Loss increases or oscillates wildly
- NaN/Inf values appear
- Training unstable even with gradient clipping
- Validation loss much worse than training loss

**Too Low:**
- Training very slow (loss barely decreases)
- Stuck in poor local minimum
- Hours of training without progress

### Advanced: Warm Restarts

For long training runs, consider cyclic learning rates:

```rust
use tensorlogic_train::OneCycleLrScheduler;

let scheduler = Box::new(OneCycleLrScheduler::new(
    0.0001,  // initial_lr
    0.01,    // max_lr
    100,     // total_epochs
));
```

---

## Batch Size

**Impact**: ⭐⭐⭐⭐ (High)
**Difficulty**: Easy
**Time to tune**: 15-30 minutes

### Why It Matters

Batch size affects:
- Training speed (larger = faster per epoch)
- Gradient noise (larger = more stable)
- Generalization (smaller often better)
- Memory usage (GPU/RAM constraint)

### Recommended Approach

**1. Start with power of 2:**
```rust
let batch_size = 32; // Good default
```

**2. Increase until memory constrained:**
```rust
// Try: 32, 64, 128, 256, 512
// Stop when out of memory
```

**3. Adjust learning rate with batch size:**

**Linear Scaling Rule**:
- Double batch size → double learning rate
- Caveat: Only works up to ~256-512

```rust
let base_lr = 0.001;
let base_batch = 32;
let new_batch = 128;

let new_lr = base_lr * (new_batch as f64 / base_batch as f64);
// new_lr = 0.004
```

### Dataset Size Considerations

| Dataset Size | Recommended Batch Size | Reason |
|-------------|----------------------|--------|
| < 1000 | 16-32 | Avoid overfitting |
| 1000-10000 | 32-64 | Balanced |
| 10000-100000 | 64-128 | Stable gradients |
| > 100000 | 128-512 | Efficiency |

### Memory Constraints

If you hit memory limits:

**Option 1: Gradient Accumulation** (not yet implemented)
```rust
// Simulate larger batch by accumulating gradients
// effective_batch_size = batch_size * accumulation_steps
```

**Option 2: Reduce batch size, adjust LR**
```rust
let batch_size = 16; // Smaller
let lr = 0.0005;     // Proportionally smaller LR
```

---

## Optimizer Selection

**Impact**: ⭐⭐⭐ (Medium-High)
**Difficulty**: Easy
**Time to tune**: 5-15 minutes

### Quick Decision Matrix

```
1. Do you have enough time to tune?
   ├─ Yes → Try multiple optimizers
   └─ No → Use Adam (safest default)

2. Is your dataset small (<10k samples)?
   ├─ Yes → Try SGD with momentum
   └─ No → Use Adam/AdamW

3. Do you need the best possible performance?
   ├─ Yes → Try AdamW, then tune others
   └─ No → Stick with Adam

4. Is training time critical?
   ├─ Yes → Use Adam (faster convergence)
   └─ No → Try SGD (might generalize better)
```

### Optimizer Comparison

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Adam** | Fast, robust, good default | Can overfit, less interpretable | Most tasks, quick prototyping |
| **AdamW** | Better generalization than Adam | Slightly slower | Production models, long training |
| **SGD** | Simple, generalizes well | Slow, needs tuning | When you have time, final tuning |
| **SGD + Momentum** | Faster than plain SGD | Still needs tuning | Computer vision, proven architectures |
| **RMSprop** | Good for RNNs | Less stable than Adam | Recurrent models |
| **NAdam** | Combines Adam + Nesterov | More complex | Advanced use |
| **LAMB** | Large batch training | Complex, niche | Very large batches |

### Recommended Configuration

**1. Adam (Default):**
```rust
use tensorlogic_train::{AdamOptimizer, OptimizerConfig};

let optimizer = AdamOptimizer::new(OptimizerConfig {
    learning_rate: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    ..Default::default()
});
```

**2. AdamW (Better Regularization):**
```rust
use tensorlogic_train::{AdamWOptimizer, OptimizerConfig};

let optimizer = AdamWOptimizer::new(OptimizerConfig {
    learning_rate: 0.001,
    weight_decay: 0.01, // L2 regularization
    beta1: 0.9,
    beta2: 0.999,
    ..Default::default()
});
```

**3. SGD with Momentum (Classical):**
```rust
use tensorlogic_train::{SgdOptimizer, OptimizerConfig};

let optimizer = SgdOptimizer::new(OptimizerConfig {
    learning_rate: 0.01,  // Higher LR than Adam
    momentum: 0.9,        // Standard value
    ..Default::default()
});
```

### Adam Hyperparameters

Most people never change β₁ and β₂. If you do:

- **β₁ (first moment)**: Usually 0.9
  - Increase to 0.95 for smoother training
  - Decrease to 0.8 for faster adaptation

- **β₂ (second moment)**: Usually 0.999
  - Use 0.99 for small batches
  - Use 0.9999 for very large batches

- **ε (epsilon)**: Usually 1e-8
  - Increase to 1e-7 if encountering NaN
  - Rarely needs tuning

### Weight Decay (AdamW)

Start with 0.01, adjust based on overfitting:

```rust
let weight_decay = if overfitting {
    0.1  // Stronger regularization
} else if underfitting {
    0.001  // Weaker regularization
} else {
    0.01  // Default
};
```

---

## Learning Rate Schedules

**Impact**: ⭐⭐⭐ (Medium)
**Difficulty**: Medium
**Time to tune**: 30-60 minutes

### When to Use Schedules

✅ **Use when:**
- Training for many epochs (>50)
- Learning rate plateaus
- Want to squeeze out final performance

❌ **Skip when:**
- Quick experiments (<10 epochs)
- Already achieving good results
- Limited tuning time

### Schedule Types

**1. Step Decay (Simple, Effective)**

```rust
use tensorlogic_train::StepLrScheduler;

let scheduler = StepLrScheduler::new(
    0.001, // initial_lr
    0.1,   // gamma (multiply by 0.1)
    30,    // step_size (every 30 epochs)
);

// Drops: 0.001 → 0.0001 → 0.00001
```

**Best for**: Known epoch counts, proven pipelines

**2. Cosine Annealing (Smooth, Popular)**

```rust
use tensorlogic_train::CosineAnnealingLrScheduler;

let scheduler = CosineAnnealingLrScheduler::new(
    0.001,   // initial_lr
    0.00001, // min_lr
    100,     // t_max (total epochs)
);
```

**Best for**: Long training, smooth convergence

**3. OneCycle (Fast Training)**

```rust
use tensorlogic_train::OneCycleLrScheduler;

let scheduler = OneCycleLrScheduler::new(
    0.0001, // initial_lr
    0.01,   // max_lr (10-100x higher!)
    50,     // total_epochs
);
```

**Best for**: When training time is limited, proven to work well

**4. ReduceOnPlateau (Adaptive)**

```rust
use tensorlogic_train::ReduceLrOnPlateauCallback;

let callback = ReduceLrOnPlateauCallback::new(
    0.5,    // factor (reduce by 50%)
    10,     // patience (wait 10 epochs)
    0.0001, // min_delta
    1e-6,   // min_lr
);
```

**Best for**: Uncertain optimal LR, automatic tuning

### Schedule Selection Guide

```
Training Duration?
├─ Short (<20 epochs)
│  └─ No schedule or Warmup only
├─ Medium (20-100 epochs)
│  ├─ Known duration → CosineAnnealing
│  └─ Uncertain → ReduceOnPlateau
└─ Long (>100 epochs)
   ├─ Want fastest → OneCycle
   ├─ Want smoothest → CosineAnnealing
   └─ Want automatic → ReduceOnPlateau
```

### Warmup

Always recommended for Adam/AdamW:

```rust
use tensorlogic_train::WarmupScheduler;

let scheduler = WarmupScheduler::new(
    0.0,    // start_lr
    0.001,  // target_lr
    5,      // warmup_epochs
);
```

Prevents instability in early training.

---

## Regularization

**Impact**: ⭐⭐⭐ (Medium)
**Difficulty**: Easy
**Time to tune**: 15-30 minutes

### Techniques Available

**1. Weight Decay (L2 Regularization)**

```rust
use tensorlogic_train::{AdamWOptimizer, OptimizerConfig};

let optimizer = AdamWOptimizer::new(OptimizerConfig {
    weight_decay: 0.01, // Start here
    ..Default::default()
});
```

**Tuning:**
- Too high (>0.1): Underfitting, loss doesn't decrease
- Too low (<0.001): Overfitting, train/val gap large
- Sweet spot: 0.001 to 0.1

**2. Gradient Clipping**

```rust
use tensorlogic_train::{OptimizerConfig, GradClipMode};

let optimizer = AdamOptimizer::new(OptimizerConfig {
    grad_clip: Some(1.0), // Clip at norm=1.0
    grad_clip_mode: GradClipMode::Norm,
    ..Default::default()
});
```

**When to use:**
- Exploding gradients (NaN/Inf)
- Unstable training
- RNNs / transformers

**Typical values:**
- Conservative: 0.5
- Standard: 1.0
- Aggressive: 5.0

**3. Early Stopping**

```rust
use tensorlogic_train::EarlyStoppingCallback;

let callback = EarlyStoppingCallback::new(
    10,    // patience
    0.001, // min_delta
);
```

Always recommended! Prevents wasting time and overfitting.

### Regularization Strategy

```
1. Start with no regularization
2. Train to convergence
3. Check train/val gap:
   ├─ Gap < 5% → No regularization needed
   ├─ Gap 5-20% → Add weight_decay=0.01
   └─ Gap > 20% → Add weight_decay=0.1 + early stopping
4. If still overfitting:
   └─ Reduce model size or get more data
```

---

## Early Stopping

**Impact**: ⭐⭐⭐⭐ (High)
**Difficulty**: Easy
**Time to tune**: 5 minutes

### Configuration

```rust
use tensorlogic_train::EarlyStoppingCallback;

let callback = EarlyStoppingCallback::new(
    patience,  // How many epochs to wait
    min_delta, // Minimum improvement threshold
);
```

### Tuning Patience

| Dataset Size | Recommended Patience | Reason |
|--------------|---------------------|---------|
| < 1000 | 5-10 | Quick overfitting |
| 1000-10000 | 10-20 | Moderate |
| > 10000 | 20-50 | Slow convergence |

### Tuning Min Delta

- **0.001**: Standard, catches meaningful improvements
- **0.0001**: Very sensitive, might stop too late
- **0.01**: Aggressive, stops early

### Best Practices

```rust
// Typical configuration
let patience = 10;
let min_delta = 0.001;

// For long training
let patience = 20;
let min_delta = 0.0001;

// For quick experiments
let patience = 5;
let min_delta = 0.01;
```

---

## Practical Workflow

### 1. Quick Start (10 minutes)

**Goal**: Get something working

```rust
// Use defaults, minimal tuning
let lr = 0.001;
let batch_size = 32;
let optimizer = AdamOptimizer::new(OptimizerConfig::default());
let epochs = 50;
let early_stopping = EarlyStoppingCallback::new(10, 0.001);

// Train and see if it works at all
```

### 2. Initial Tuning (1 hour)

**Goal**: Find good learning rate and batch size

```rust
// Step 1: Use LR Finder (15 min)
let lr_finder = LearningRateFinder::new(1e-7, 1.0, 100);
// → Pick LR where loss decreases fastest

// Step 2: Try batch sizes (30 min)
for batch in [32, 64, 128] {
    // Train for 10 epochs each
    // Pick fastest converging
}

// Step 3: Add early stopping (5 min)
let early_stopping = EarlyStoppingCallback::new(10, 0.001);
```

### 3. Refinement (2-4 hours)

**Goal**: Squeeze out performance

```rust
// Step 1: Try different optimizers (1 hour)
// Compare Adam vs AdamW vs SGD

// Step 2: Add scheduling (1 hour)
// Try CosineAnnealing or OneCycle

// Step 3: Tune regularization (1 hour)
// Adjust weight_decay if overfitting

// Step 4: Final polish (1 hour)
// Fine-tune patience, min_delta
```

### 4. Production (1-2 days)

**Goal**: Best possible model

```rust
// Everything from refinement, plus:
// - Extensive hyperparameter search
// - Multiple random seeds
// - Cross-validation
// - Ensemble methods
```

## Common Pitfalls

### 1. Changing Too Many Things at Once

❌ **Bad:**
```rust
// Changed LR, batch size, optimizer, AND schedule
// Can't tell what helped!
```

✅ **Good:**
```rust
// Change one thing, see effect, then change next
```

### 2. Not Using Validation Set

❌ **Bad:**
```rust
// Only monitoring training loss
// Might be overfitting severely
```

✅ **Good:**
```rust
// Always split data, monitor validation
```

### 3. Training Too Long

❌ **Bad:**
```rust
// Running for 1000 epochs without early stopping
// Wasting time and overfitting
```

✅ **Good:**
```rust
// Use early stopping with reasonable patience
```

### 4. Ignoring Learning Curves

❌ **Bad:**
```rust
// Not plotting loss over time
// Missing obvious issues
```

✅ **Good:**
```rust
// Plot train/val loss
// Look for divergence, plateaus
```

## Hyperparameter Interaction Table

| If you increase... | Then adjust... | Direction |
|-------------------|----------------|-----------|
| Batch size | Learning rate | Increase |
| Learning rate | Gradient clip | Might need to add/increase |
| Model size | Regularization | Increase |
| Dataset size | Everything | Less tuning needed |
| Training time | Patience | Increase |
| Weight decay | Learning rate | Might increase slightly |

## Quick Reference Card

```
┌─────────────────────────────────────┐
│ QUICK START DEFAULTS                │
├─────────────────────────────────────┤
│ Optimizer: Adam                     │
│ Learning Rate: 0.001                │
│ Batch Size: 32                      │
│ Early Stopping: 10 epochs           │
│ Weight Decay: 0 (start)             │
│ Gradient Clip: None (start)         │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ TUNING PRIORITY                     │
├─────────────────────────────────────┤
│ 1. Learning Rate (use LR finder)    │
│ 2. Batch Size (32→64→128)           │
│ 3. Early Stopping (always use!)     │
│ 4. Weight Decay (if overfitting)    │
│ 5. LR Schedule (if time permits)    │
│ 6. Optimizer (try AdamW)            │
└─────────────────────────────────────┘
```

## Further Reading

- [Loss Function Selection Guide](LOSS_FUNCTIONS.md)
- [Training Examples](../examples/)
- [Callback API](../src/callbacks.rs)

---

****Last Updated**: 2026-01-28
**Version**: 0.1.0-beta.1
