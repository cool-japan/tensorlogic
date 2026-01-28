# Loss Function Selection Guide

Comprehensive guide to selecting the right loss function for your training task.

## Overview

The `tensorlogic-train` crate provides 10 different loss functions, each optimized for specific scenarios. This guide helps you choose the right one.

## Quick Reference

| Loss Function | Best For | Key Properties | Output Range |
|--------------|----------|----------------|--------------|
| **CrossEntropyLoss** | Multi-class classification | Probabilistic, smooth gradients | [0, ∞) |
| **BCEWithLogitsLoss** | Binary classification | Numerically stable, combines sigmoid | [0, ∞) |
| **MseLoss** | Regression | Simple, differentiable, outlier sensitive | [0, ∞) |
| **FocalLoss** | Imbalanced classification | Focuses on hard examples | [0, ∞) |
| **HuberLoss** | Robust regression | Outlier resistant, smooth | [0, ∞) |
| **DiceLoss** | Segmentation, overlap | Set similarity, handles imbalance | [0, 1] |
| **TverskyLoss** | Imbalanced segmentation | Adjustable FP/FN trade-off | [0, 1] |
| **RuleSatisfactionLoss** | Logic constraints | Soft logic penalties | [0, ∞) |
| **ConstraintViolationLoss** | Hard constraints | Direct violation measurement | [0, ∞) |
| **LogicalLoss** | Multi-objective | Combines multiple losses | [0, ∞) |

## Detailed Guide

### 1. CrossEntropyLoss

**Use when:**
- Multi-class classification (>2 classes)
- Outputs are class probabilities
- Classes are mutually exclusive

**Advantages:**
- Smooth, well-behaved gradients
- Probabilistic interpretation
- Works well with softmax outputs
- Numerically stable implementation

**Disadvantages:**
- Sensitive to class imbalance
- Can be overconfident on easy examples

**Example:**
```rust
use tensorlogic_train::CrossEntropyLoss;

let loss = CrossEntropyLoss::default();

// predictions: softmax outputs [batch_size, num_classes]
// targets: one-hot encoded [batch_size, num_classes]
let loss_value = loss.compute(&predictions.view(), &targets.view())?;
```

**Best practices:**
- Use with softmax activation
- Consider label smoothing for regularization
- Monitor per-class accuracy for imbalance

**Hyperparameters:**
- `epsilon`: Numerical stability (default: 1e-10)

---

### 2. BCEWithLogitsLoss (Binary Cross-Entropy with Logits)

**Use when:**
- Binary classification (2 classes)
- Multi-label classification
- Need numerical stability

**Advantages:**
- More numerically stable than separate sigmoid + BCE
- Avoids log(0) issues
- Good for probability outputs

**Disadvantages:**
- Assumes independence in multi-label case
- Can be sensitive to class imbalance

**Example:**
```rust
use tensorlogic_train::BCEWithLogitsLoss;

let loss = BCEWithLogitsLoss;

// predictions: raw logits [batch_size, num_labels]
// targets: binary labels [batch_size, num_labels]
let loss_value = loss.compute(&logits.view(), &targets.view())?;
```

**Best practices:**
- Use raw logits as input (no sigmoid needed)
- Consider pos_weight for class imbalance
- Monitor precision/recall for each label

---

### 3. MseLoss (Mean Squared Error)

**Use when:**
- Regression tasks
- Continuous target values
- Targets are roughly Gaussian distributed

**Advantages:**
- Simple, intuitive
- Smooth gradients everywhere
- Well-studied properties

**Disadvantages:**
- Sensitive to outliers (quadratic penalty)
- Can lead to overconfident predictions
- Large errors dominate training

**Example:**
```rust
use tensorlogic_train::MseLoss;

let loss = MseLoss;

// predictions: continuous values [batch_size, output_dim]
// targets: ground truth values [batch_size, output_dim]
let loss_value = loss.compute(&predictions.view(), &targets.view())?;
```

**Best practices:**
- Normalize targets to similar scales
- Consider HuberLoss if outliers present
- Monitor mean absolute error alongside MSE

**Alternatives:**
- Use **HuberLoss** for outlier robustness
- Use **MAE** (L1 loss) for median prediction

---

### 4. FocalLoss

**Use when:**
- Severe class imbalance (1:100 or worse)
- Easy examples dominate training
- Object detection tasks
- Hard example mining needed

**Advantages:**
- Automatically down-weights easy examples
- Focuses learning on hard cases
- Reduces false negatives on rare classes

**Disadvantages:**
- More hyperparameters to tune
- Can be unstable early in training
- Requires careful tuning of α and γ

**Example:**
```rust
use tensorlogic_train::FocalLoss;

let loss = FocalLoss::new(
    2.0,  // gamma: focusing parameter (2.0 is typical)
    0.25, // alpha: class balance (0.25 for 1:3 imbalance)
);

// predictions: class probabilities [batch_size, num_classes]
// targets: one-hot labels [batch_size, num_classes]
let loss_value = loss.compute(&predictions.view(), &targets.view())?;
```

**Best practices:**
- Start with γ=2.0, α=0.25
- Use with class imbalance >10:1
- Monitor per-class recall
- Combine with data augmentation

**Hyperparameters:**
- `gamma`: Focusing parameter (higher = more focus on hard examples)
  - γ=0: equivalent to CrossEntropy
  - γ=2: typical for detection
  - γ=5: very hard focus
- `alpha`: Class weighting (0 to 1)
  - α=0.25: good for 1:3 imbalance
  - α=0.1: good for 1:9 imbalance

---

### 5. HuberLoss (Smooth L1 Loss)

**Use when:**
- Regression with outliers
- Robust estimation needed
- Combining L1 and L2 properties

**Advantages:**
- Robust to outliers
- Smooth gradients near zero
- Balances L1 and L2 benefits

**Disadvantages:**
- Extra hyperparameter (delta)
- Slightly more complex than MSE

**Example:**
```rust
use tensorlogic_train::HuberLoss;

let loss = HuberLoss::new(1.0); // delta=1.0

// predictions: continuous values [batch_size, output_dim]
// targets: ground truth with potential outliers
let loss_value = loss.compute(&predictions.view(), &targets.view())?;
```

**Best practices:**
- Set delta to expected noise level
- Use when data has >5% outliers
- Monitor both MSE and MAE

**Hyperparameters:**
- `delta`: Threshold between L2 and L1
  - Small δ: more robust, less smooth
  - Large δ: more smooth, less robust
  - Typical: δ ∈ [0.5, 2.0]

---

### 6. DiceLoss

**Use when:**
- Segmentation tasks
- Measuring set similarity/overlap
- Class imbalance in segmentation
- Small objects in images

**Advantages:**
- Handles class imbalance naturally
- Directly optimizes overlap metric
- Works well for small objects
- Differentiable approximation of Dice coefficient

**Disadvantages:**
- Can be unstable with empty predictions
- Slower convergence than CrossEntropy
- Sensitive to batch composition

**Example:**
```rust
use tensorlogic_train::DiceLoss;

let loss = DiceLoss::new(1e-6); // smooth: numerical stability

// predictions: pixel probabilities [batch_size, height, width]
// targets: binary masks [batch_size, height, width]
let loss_value = loss.compute(&predictions.view(), &targets.view())?;
```

**Best practices:**
- Combine with CrossEntropy (0.5 * CE + 0.5 * Dice)
- Use smooth=1e-6 for numerical stability
- Good for foreground/background ratios >100:1

**Hyperparameters:**
- `smooth`: Smoothing constant (default: 1e-6)

---

### 7. TverskyLoss

**Use when:**
- Imbalanced segmentation
- Need to control FP vs FN trade-off
- Small object detection
- Medical imaging with rare pathologies

**Advantages:**
- Flexible FP/FN control via α and β
- Generalizes Dice loss
- Handles severe imbalance

**Disadvantages:**
- More hyperparameters than Dice
- Requires domain knowledge to set α, β
- Can be unstable if misconfigured

**Example:**
```rust
use tensorlogic_train::TverskyLoss;

let loss = TverskyLoss::new(
    0.3,  // alpha: weight for false positives
    0.7,  // beta: weight for false negatives
    1e-6, // smooth: numerical stability
);

// predictions: pixel probabilities [batch_size, height, width]
// targets: binary masks [batch_size, height, width]
let loss_value = loss.compute(&predictions.view(), &targets.view())?;
```

**Best practices:**
- α + β = 1 for balanced weighting
- α < β to reduce false negatives (recall priority)
- α > β to reduce false positives (precision priority)
- Start with α=0.3, β=0.7 for rare objects

**Hyperparameters:**
- `alpha`: FP weight (0 to 1)
- `beta`: FN weight (0 to 1)
- Common configurations:
  - α=0.5, β=0.5: equivalent to Dice
  - α=0.3, β=0.7: emphasize recall
  - α=0.7, β=0.3: emphasize precision

---

### 8. RuleSatisfactionLoss

**Use when:**
- Training with logical rules
- Soft constraint enforcement
- Domain knowledge incorporation
- Neuro-symbolic learning

**Advantages:**
- Differentiable rule satisfaction
- Temperature-controlled softness
- Integrates logic with learning

**Disadvantages:**
- Requires rule specification
- Temperature tuning needed
- Soft approximation of hard logic

**Example:**
```rust
use tensorlogic_train::RuleSatisfactionLoss;

let loss = RuleSatisfactionLoss::default();

// predictions: rule truth values [batch_size, num_rules]
// targets: desired satisfaction levels [batch_size, num_rules]
let loss_value = loss.compute(&predictions.view(), &targets.view())?;
```

**Best practices:**
- Combine with supervised loss
- Start with temperature=1.0
- Monitor individual rule satisfaction
- Use with LogicalLoss for multi-objective

---

### 9. ConstraintViolationLoss

**Use when:**
- Hard constraint enforcement
- Physical constraints (e.g., non-negativity)
- Feasibility requirements
- Structured prediction

**Advantages:**
- Direct violation measurement
- Interpretable penalties
- Flexible constraint types

**Disadvantages:**
- Requires explicit constraint specification
- May conflict with supervised objectives
- Can lead to local minima

**Example:**
```rust
use tensorlogic_train::ConstraintViolationLoss;

let loss = ConstraintViolationLoss::default();

// predictions: constraint values [batch_size, num_constraints]
// targets: feasibility thresholds [batch_size, num_constraints]
let loss_value = loss.compute(&predictions.view(), &targets.view())?;
```

**Best practices:**
- Weight carefully vs supervised loss
- Start with low weight, increase gradually
- Monitor violation rates separately
- Consider soft constraints first

---

### 10. LogicalLoss (Multi-Objective)

**Use when:**
- Combining multiple objectives
- Supervised + logical constraints
- Multi-task learning
- Complex training scenarios

**Advantages:**
- Flexible weight control
- Combines supervised + logical objectives
- Extensible framework

**Disadvantages:**
- Many hyperparameters
- Requires careful weight tuning
- Can be complex to debug

**Example:**
```rust
use tensorlogic_train::{LogicalLoss, LossConfig, CrossEntropyLoss,
                         RuleSatisfactionLoss, ConstraintViolationLoss};

let config = LossConfig {
    supervised_weight: 1.0,
    rule_weight: 2.0,
    constraint_weight: 5.0,
    temperature: 1.0,
};

let logical_loss = LogicalLoss::new(
    config,
    Box::new(CrossEntropyLoss::default()),
    vec![Box::new(RuleSatisfactionLoss::default())],
    vec![Box::new(ConstraintViolationLoss::default())],
);

// Use compute_total() with rule_values and constraint_values arrays
```

**Best practices:**
- Start with equal weights, adjust based on convergence
- Monitor each component separately
- Normalize loss scales before weighting
- Use validation to tune weights

---

## Loss Function Selection Decision Tree

```
1. What's your task?
   ├─ Classification → Go to 2
   ├─ Regression → Go to 5
   └─ Segmentation → Go to 7

2. Classification: How many classes?
   ├─ Binary (2) → BCEWithLogitsLoss
   ├─ Multi-class (>2) → Go to 3
   └─ Multi-label → BCEWithLogitsLoss

3. Multi-class: Is there class imbalance?
   ├─ Balanced → CrossEntropyLoss
   ├─ Moderate (1:10) → CrossEntropyLoss + class weights
   └─ Severe (>1:100) → FocalLoss

4. Multi-class: Do you have hard examples?
   ├─ Easy examples dominate → FocalLoss
   └─ Balanced difficulty → CrossEntropyLoss

5. Regression: Do you have outliers?
   ├─ No outliers → MseLoss
   ├─ Some outliers (<10%) → HuberLoss
   └─ Many outliers (>10%) → HuberLoss with small delta

6. Regression: What's your prediction goal?
   ├─ Mean prediction → MseLoss
   ├─ Median prediction → MAE (L1)
   └─ Robust estimation → HuberLoss

7. Segmentation: Class balance?
   ├─ Balanced → CrossEntropyLoss
   ├─ Moderate imbalance → DiceLoss
   └─ Severe imbalance → TverskyLoss

8. Segmentation: Object size?
   ├─ Large objects → CrossEntropyLoss
   ├─ Small objects → DiceLoss
   └─ Tiny objects → TverskyLoss (α=0.3, β=0.7)

9. Do you have logical constraints?
   ├─ Soft constraints → RuleSatisfactionLoss
   ├─ Hard constraints → ConstraintViolationLoss
   └─ Multiple objectives → LogicalLoss
```

## Combining Loss Functions

### Common Combinations

**1. Classification + Logic**
```rust
let config = LossConfig {
    supervised_weight: 1.0,
    rule_weight: 0.5,
    constraint_weight: 1.0,
    temperature: 1.0,
};
```

**2. Segmentation: Dice + CrossEntropy**
```rust
// Manually combine (0.5 * CrossEntropy + 0.5 * Dice)
// Better than either alone for segmentation
```

**3. Regression with Constraints**
```rust
// Use LogicalLoss with MSE + ConstraintViolation
// Ensures physical feasibility
```

## Hyperparameter Tuning Guidelines

### Learning Rate Interaction

- **MSE**: Stable with larger LR (1e-2 to 1e-3)
- **CrossEntropy**: Moderate LR (1e-3 to 1e-4)
- **Focal**: Start with smaller LR (1e-4), unstable early
- **Huber**: Similar to MSE
- **Dice/Tversky**: Smaller LR (1e-4 to 1e-5)

### Batch Size Effects

- **All losses**: Larger batches → more stable gradients
- **Dice/Tversky**: Very sensitive to batch composition
- **Focal**: Benefits from larger batches
- **MSE**: Works well with any batch size

### Optimizer Selection

- **Adam/AdamW**: Good default for all losses
- **SGD**: Works well with MSE, CrossEntropy
- **RMSprop**: Good for RNN tasks
- **AdamW**: Best for long training runs

## Common Pitfalls

### 1. Class Imbalance
**Problem**: CrossEntropy treats all classes equally
**Solution**: Use FocalLoss or class weights

### 2. Outliers
**Problem**: MSE gives huge penalties to outliers
**Solution**: Use HuberLoss

### 3. Small Objects
**Problem**: CrossEntropy ignores spatial overlap
**Solution**: Use DiceLoss or TverskyLoss

### 4. Conflicting Objectives
**Problem**: Supervised and constraint losses fight
**Solution**: Careful weight tuning, start with low constraint weight

### 5. Unstable Training
**Problem**: Loss oscillates or diverges
**Solution**: Lower learning rate, check for NaN/Inf, add gradient clipping

## Debugging Loss Values

### Expected Ranges

- **CrossEntropy**: 0 to -log(1/num_classes)
  - Perfect: 0
  - Random: log(num_classes)
  - Bad: > 2 * log(num_classes)

- **MSE**: 0 to ∞
  - Perfect: 0
  - Good: < 0.1 * target_variance
  - Bad: > target_variance

- **Dice**: 0 to 1
  - Perfect: 0
  - Good: < 0.3
  - Bad: > 0.7

### Warning Signs

- Loss = NaN: Exploding gradients, use gradient clipping
- Loss = 0: Overfitting or bug in implementation
- Loss increasing: Learning rate too high
- Loss plateaus early: Learning rate too low or local minimum

## References

- Focal Loss: https://arxiv.org/abs/1708.02002
- Dice Loss: https://arxiv.org/abs/1606.04797
- Tversky Loss: https://arxiv.org/abs/1706.05721
- Huber Loss: https://en.wikipedia.org/wiki/Huber_loss

## See Also

- [Hyperparameter Tuning Guide](HYPERPARAMETER_TUNING.md)
- [Training Examples](../examples/)
- [API Documentation](../src/loss.rs)

---

****Last Updated**: 2026-01-28
**Version**: 0.1.0-beta.1
