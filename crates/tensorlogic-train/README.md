# tensorlogic-train
[![Crate](https://img.shields.io/badge/crates.io-tensorlogic-train-orange)](https://crates.io/crates/tensorlogic-train)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://docs.rs/tensorlogic-train)
[![Tests](https://img.shields.io/badge/tests-745%2F745-brightgreen)](#)
[![Production](https://img.shields.io/badge/status-production_ready-success)](#)

Training scaffolds for Tensorlogic: loss composition, optimizers, schedulers, and callbacks.

## Overview

`tensorlogic-train` provides comprehensive training infrastructure for Tensorlogic models, combining standard ML training components with logic-specific loss functions for constraint satisfaction and rule adherence.

## Features

### Loss Functions (15 types)
- **Standard Losses**: Cross-entropy, MSE, BCE with logits
- **Robust Losses**: Focal (class imbalance), Huber (outliers)
- **Segmentation**: Dice, Tversky (IoU-based losses)
- **Metric Learning**: Contrastive, Triplet (embedding learning)
- **Classification**: Hinge (SVM-style max-margin)
- **Distribution**: KL Divergence (distribution matching)
- **Advanced**: Poly Loss (polynomial expansion of CE for better generalization)
- **Logical Losses**: Rule satisfaction, constraint violation penalties
- **Multi-objective**: Weighted combination of supervised + logical losses
- **Gradient Computation**: All losses support automatic gradient computation

### Optimizers (18 types)
- **SGD**: Momentum support, gradient clipping (value and L2 norm)
- **Adam**: First/second moment estimation, bias correction
- **AdamW**: Decoupled weight decay for better regularization
- **AdamP**: Adam with projection for better generalization
- **RMSprop**: Adaptive learning rates with moving average
- **Adagrad**: Accumulating gradient normalization
- **NAdam**: Nesterov-accelerated Adam
- **LAMB**: Layer-wise adaptive moments (large-batch training)
- **AdaMax**: Adam variant with infinity norm (robust to large gradients)
- **Lookahead**: Slow/fast weights for improved convergence
- **AdaBelief** (NeurIPS 2020): Adapts stepsizes by gradient belief
- **RAdam** (ICLR 2020): Rectified Adam with variance warmup
- **LARS**: Layer-wise adaptive rate scaling for large batch training
- **SAM** (ICLR 2021): Sharpness aware minimization for better generalization
- **Lion**: Modern memory-efficient optimizer with sign-based updates (EvoLved Sign Momentum)
- **Prodigy** (2024): Auto-tuning learning rate, eliminates manual LR tuning entirely
- **ScheduleFreeAdamW** (2024): Eliminates LR scheduling via parameter averaging (Defazio et al.)
- **Sophia**: Second-order optimizer with Hessian diagonal estimates
- **Gradient Clipping**: By value (element-wise) or by L2 norm (global)
- **Gradient Centralization**: Drop-in optimizer wrapper (LayerWise, Global, PerRow, PerColumn)
- **State Management**: Save/load optimizer state for checkpointing

### Learning Rate Schedulers (12 types)
- **StepLR**: Step decay every N epochs
- **ExponentialLR**: Exponential decay per epoch
- **CosineAnnealingLR**: Cosine annealing with warmup
- **WarmupScheduler**: Linear learning rate warmup
- **OneCycleLR**: Super-convergence with single cycle
- **PolynomialDecayLR**: Polynomial learning rate decay
- **CyclicLR**: Triangular/exponential cyclic schedules
- **WarmupCosineLR**: Warmup + cosine annealing
- **NoamScheduler** (Transformer): Attention is All You Need schedule
- **MultiStepLR**: Decay at specific milestone epochs
- **ReduceLROnPlateau**: Adaptive reduction based on validation metrics
- **SGDR**: Stochastic Gradient Descent with Warm Restarts

### Batch Management
- **BatchIterator**: Configurable batch iteration with shuffling
- **DataShuffler**: Deterministic shuffling with seed control
- **Flexible Configuration**: Drop last, custom batch sizes

### Training Loop
- **Trainer**: Complete training orchestration
- **Epoch/Batch Iteration**: Automated iteration with state tracking
- **Validation**: Built-in validation loop with metrics
- **History Tracking**: Loss and metrics history across epochs

### Callbacks (14+ types)
- **Training Events**: on_train/epoch/batch/validation hooks
- **EarlyStoppingCallback**: Stop training when validation plateaus
- **CheckpointCallback**: Save model checkpoints (best/periodic) with optional compression via `oxiarc-deflate` (Pure Rust)
- **ReduceLrOnPlateauCallback**: Adaptive learning rate reduction
- **LearningRateFinder**: Find optimal learning rate automatically
- **GradientMonitor**: Track gradient flow and detect issues
- **HistogramCallback**: Monitor weight distributions
- **ProfilingCallback**: Track training performance and throughput
- **ModelEMACallback**: Exponential moving average for stable predictions
- **GradientAccumulationCallback**: Simulate large batches with limited memory
- **SWACallback**: Stochastic Weight Averaging for better generalization
- **MemoryProfilerCallback**: Track memory usage during training
- **Custom Callbacks**: Easy-to-implement callback trait

### Metrics (19+ types)
- **Accuracy**: Classification accuracy with argmax
- **Precision/Recall**: Per-class and macro-averaged
- **F1 Score**: Harmonic mean of precision/recall
- **ConfusionMatrix**: Full confusion matrix with per-class analysis
- **ROC/AUC**: ROC curve computation and AUC calculation
- **PerClassMetrics**: Comprehensive per-class reporting with pretty printing
- **MetricTracker**: Multi-metric tracking with history
- **TopKAccuracy**: Top-k classification accuracy
- **NDCG**: Normalized Discounted Cumulative Gain (ranking)
- **BalancedAccuracy**, **CohensKappa**, **MatthewsCorrelationCoefficient**
- **Computer Vision**: IoU, MeanIoU, DiceCoefficient, MeanAveragePrecision
- **Calibration**: ExpectedCalibrationError, MaximumCalibrationError

### Model Interface
- **Model Trait**: Flexible interface for trainable models
- **AutodiffModel**: Integration point for automatic differentiation
- **DynamicModel**: Support for variable-sized inputs
- **LinearModel**: Reference implementation demonstrating the interface

### Regularization (9 types)
- **L1 Regularization**: Lasso with sparsity-inducing penalties
- **L2 Regularization**: Ridge for weight decay
- **Elastic Net**: Combined L1+L2 regularization
- **Composite**: Combine multiple regularization strategies
- **Spectral Normalization**: GAN training stability
- **MaxNorm Constraint**: Gradient stability
- **Orthogonal Regularization**: W^T * W approximates identity
- **Group Lasso**: Group-wise sparsity
- **Full Gradient Support**: All regularizers compute gradients

### Data Augmentation (9+ types)
- **Noise Augmentation**: Gaussian noise with Box-Muller transform
- **Scale Augmentation**: Random scaling within configurable ranges
- **Rotation Augmentation**: Placeholder for future image rotation
- **Mixup**: Zhang et al. (ICLR 2018) for improved generalization
- **CutMix**: CutMix augmentation technique
- **RandomErasing**: Randomly erase rectangular regions (AAAI 2020)
- **CutOut**: Fixed-size random square erasing
- **Composite Pipeline**: Chain multiple augmentations
- **SciRS2 RNG**: Uses SciRS2 for random number generation

### Regularization (Advanced)
- **DropPath / Stochastic Depth**: Randomly drops entire residual paths (ECCV 2016)
  - Linear and Exponential schedulers for depth-based drop probability
  - Widely used in Vision Transformers (ViT, DeiT, Swin)
- **DropBlock**: Structured dropout for CNNs (NeurIPS 2018)
  - Drops contiguous blocks instead of individual neurons
  - Configurable block size with linear scheduler

### Logging and Monitoring
- **Console Logger**: Stdout logging with timestamps
- **File Logger**: Persistent file logging with append/truncate modes
- **TensorBoard Logger**: Real tfevents format with CRC32, scalars, histograms, text
- **CSV Logger**: Machine-readable CSV for pandas/spreadsheet analysis
- **JSONL Logger**: JSON Lines format for programmatic processing
- **Metrics Logger**: Aggregates and logs to multiple backends
- **Structured Logging**: Optional tracing/tracing-subscriber integration (feature flag)

### Memory Management
- **MemoryProfilerCallback**: Track memory usage with reports during training
- **GradientCheckpointConfig**: Strategies for memory-efficient training
- **MemoryBudgetManager**: Allocation tracking and budget enforcement
- **MemoryEfficientTraining**: Optimal batch size and model memory estimation

### Data Loading and Preprocessing
- **Dataset struct**: Unified data container with features/targets
- **Train/val/test splits**: Configurable split ratios with validation
- **CSV Loader**: Configurable CSV data loading with column selection
- **DataPreprocessor**: Standardization, normalization, min-max scaling
- **LabelEncoder**: String to numeric label conversion
- **OneHotEncoder**: Categorical to binary encoding

### Advanced Machine Learning
- **Curriculum Learning**: 5 strategies (Linear, Exponential, SelfPaced, Competence, Task)
- **Transfer Learning**: LayerFreezing, ProgressiveUnfreezing, DiscriminativeFineTuning
- **Hyperparameter Optimization**: Grid search, Random search, Bayesian Optimization (GP-based)
- **Cross-Validation**: KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut
- **Model Ensembling**: Voting, Averaging, Stacking, Bagging
- **Knowledge Distillation**: Standard, Feature, Attention transfer
- **Label Smoothing**: Label smoothing and Mixup regularization
- **Multi-task Learning**: Fixed weights, DTP, PCGrad strategies
- **Few-Shot Learning**: Prototypical networks, Matching networks, N-way K-shot sampling
- **Meta-Learning**: MAML (Model-Agnostic Meta-Learning) and Reptile
- **Model Pruning**: Magnitude, Gradient, Structured, Global pruning with iterative schedules
- **Model Quantization**: INT8/INT4/INT2, Post-Training Quantization, Quantization-Aware Training
- **Mixed Precision Training**: FP16/BF16 support, loss scaling, master weights

### Sampling Strategies (7 techniques)
- **HardNegativeMiner**: TopK, threshold, and focal strategies
- **ImportanceSampler**: With/without replacement
- **FocalSampler**: Emphasize hard examples
- **ClassBalancedSampler**: Handle class imbalance
- **CurriculumSampler**: Progressive difficulty
- **OnlineHardExampleMiner**: Dynamic batch selection
- **BatchReweighter**: Uniform, inverse loss, focal, gradient norm

### Model Utilities
- **ParameterStats / ModelSummary**: Parameter analysis and model introspection
- **GradientStats**: Gradient monitoring and analysis
- **TimeEstimator**: Training time prediction
- **LrRangeTestAnalyzer**: Optimal LR finding analysis
- **compare_models**: Model comparison utilities

### Neural ODE (v0.1.18)
- **OdeFunc trait**: User-defined dynamics `f(t, y, params) -> dy/dt` with `vjp()` for vector-Jacobian products
- **rk4_solve()**: Fixed-step RK4 integrator returning full `OdeSolution` trajectory
- **dopri5_solve()**: Adaptive Dormand-Prince RK45 solver with step-size control and error estimation
- **OdeSolverConfig**: Builder with `rtol`, `atol`, `max_steps`, `dense_output`
- **NeuralOde**: Wraps user dynamics function with `(t0, t1)` integration bounds; `forward()` returns endpoint state
- **adjoint_backward()**: Adjoint sensitivity method — memory-efficient gradient O(1) storage rather than O(T)

### Online Learning (v0.1.20)
- **OnlineLearner trait**: `update()`, `predict()`, `reset()` unified interface
- **Perceptron**: Binary classifier with margin-based weight updates
- **PassiveAggressive**: PA / PA-I / PA-II variants with configurable aggressiveness `C`
- **OnlineGradientDescent**: Squared-loss, hinge-loss, and logistic-loss modes with configurable step size
- **FtrlProximal**: Follow The Regularized Leader with L1/L2 regularization, per-coordinate adaptive learning rates
- **OnlineStats**: Per-step loss, mistake rate, cumulative regret, and `n_updates`
- **online_evaluate()**: Batch helper for evaluating/training over a labelled dataset

### Adversarial Training (v0.1.21)
- **FGSM**: Fast Gradient Sign Method — one-step sign gradient attack
- **PGD**: Projected Gradient Descent — iterative attack with random start and norm-ball projection
- **Norm constraints**: L∞, L2, and L1 perturbation norm support
- **CrossEntropyAttackLoss / MseAttackLoss**: Attack-specific loss functions
- **LinearAttackModel**: Reference model for adversarial experiments
- **adversarial_training_loss()**: Combined clean + adversarial loss for robust training
- **robustness_eval()**: Evaluate model accuracy under adversarial perturbations

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tensorlogic-train = { path = "../tensorlogic-train" }
```

## Quick Start

```rust
use tensorlogic_train::{
    Trainer, TrainerConfig, MseLoss, AdamOptimizer, OptimizerConfig,
    EpochCallback, CallbackList, MetricTracker, Accuracy,
};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

// Create loss function
let loss = Box::new(MseLoss);

// Create optimizer
let optimizer_config = OptimizerConfig {
    learning_rate: 0.001,
    ..Default::default()
};
let optimizer = Box::new(AdamOptimizer::new(optimizer_config));

// Create trainer
let config = TrainerConfig {
    num_epochs: 10,
    ..Default::default()
};
let mut trainer = Trainer::new(config, loss, optimizer);

// Add callbacks
let mut callbacks = CallbackList::new();
callbacks.add(Box::new(EpochCallback::new(true)));
trainer = trainer.with_callbacks(callbacks);

// Add metrics
let mut metrics = MetricTracker::new();
metrics.add(Box::new(Accuracy::default()));
trainer = trainer.with_metrics(metrics);

// Prepare data
let train_data = Array2::zeros((100, 10));
let train_targets = Array2::zeros((100, 2));
let val_data = Array2::zeros((20, 10));
let val_targets = Array2::zeros((20, 2));

// Train model
let mut parameters = HashMap::new();
parameters.insert("weights".to_string(), Array2::zeros((10, 2)));

let history = trainer.train(
    &train_data.view(),
    &train_targets.view(),
    Some(&val_data.view()),
    Some(&val_targets.view()),
    &mut parameters,
).unwrap();

// Access training history
println!("Training losses: {:?}", history.train_loss);
println!("Validation losses: {:?}", history.val_loss);
if let Some((best_epoch, best_loss)) = history.best_val_loss() {
    println!("Best validation loss: {} at epoch {}", best_loss, best_epoch);
}
```

## Logical Loss Functions

Combine supervised learning with logical constraints:

```rust
use tensorlogic_train::{
    LogicalLoss, LossConfig, CrossEntropyLoss,
    RuleSatisfactionLoss, ConstraintViolationLoss,
};

// Configure loss weights
let config = LossConfig {
    supervised_weight: 1.0,
    constraint_weight: 10.0,  // Heavily penalize constraint violations
    rule_weight: 5.0,
    temperature: 1.0,
};

// Create logical loss
let logical_loss = LogicalLoss::new(
    config,
    Box::new(CrossEntropyLoss::default()),
    vec![Box::new(RuleSatisfactionLoss::default())],
    vec![Box::new(ConstraintViolationLoss::default())],
);

// Compute total loss
let total_loss = logical_loss.compute_total(
    &predictions.view(),
    &targets.view(),
    &rule_values,
    &constraint_values,
)?;
```

## Early Stopping

Stop training automatically when validation stops improving:

```rust
use tensorlogic_train::{CallbackList, EarlyStoppingCallback};

let mut callbacks = CallbackList::new();
callbacks.add(Box::new(EarlyStoppingCallback::new(
    5,      // patience: Wait 5 epochs without improvement
    0.001,  // min_delta: Minimum improvement threshold
)));

trainer = trainer.with_callbacks(callbacks);
// Training will stop automatically if validation doesn't improve for 5 epochs
```

## Checkpointing

Save model checkpoints during training:

```rust
use tensorlogic_train::{CallbackList, CheckpointCallback};
use std::path::PathBuf;

let mut callbacks = CallbackList::new();
callbacks.add(Box::new(CheckpointCallback::new(
    PathBuf::from("/tmp/checkpoints"),
    1,    // save_frequency: Save every epoch
    true, // save_best_only: Only save when validation improves
)));

trainer = trainer.with_callbacks(callbacks);
```

## Learning Rate Scheduling

Adjust learning rate during training:

```rust
use tensorlogic_train::{CosineAnnealingLrScheduler, LrScheduler};

let scheduler = Box::new(CosineAnnealingLrScheduler::new(
    0.001,   // initial_lr
    0.00001, // min_lr
    100,     // t_max: Total epochs
));

trainer = trainer.with_scheduler(scheduler);
```

## Gradient Clipping by Norm

Use L2 norm clipping for stable training of deep networks:

```rust
use tensorlogic_train::{AdamOptimizer, OptimizerConfig, GradClipMode};

let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig {
    learning_rate: 0.001,
    grad_clip: Some(5.0),  // Clip if global L2 norm > 5.0
    grad_clip_mode: GradClipMode::Norm,  // Use L2 norm clipping
    ..Default::default()
}));

// Global L2 norm is computed across all parameters:
// norm = sqrt(sum(g_i^2 for all gradients g_i))
// If norm > 5.0, all gradients are scaled by (5.0 / norm)
```

## Enhanced Metrics

### Confusion Matrix

```rust
use tensorlogic_train::ConfusionMatrix;

let cm = ConfusionMatrix::compute(&predictions.view(), &targets.view())?;

// Pretty print the confusion matrix
println!("{}", cm);

// Get per-class metrics
let precision = cm.precision_per_class();
let recall = cm.recall_per_class();
let f1 = cm.f1_per_class();

// Get overall accuracy
println!("Accuracy: {:.4}", cm.accuracy());
```

### ROC Curve and AUC

```rust
use tensorlogic_train::RocCurve;

// Binary classification example
let predictions = vec![0.9, 0.8, 0.3, 0.1];
let targets = vec![true, true, false, false];

let roc = RocCurve::compute(&predictions, &targets)?;

// Compute AUC
println!("AUC: {:.4}", roc.auc());
```

## Regularization

Prevent overfitting with L1, L2, or Elastic Net regularization:

```rust
use tensorlogic_train::{L2Regularization, Regularizer};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

// Create L2 regularization (weight decay)
let regularizer = L2Regularization::new(0.01); // lambda = 0.01

// Compute regularization penalty
let mut parameters = HashMap::new();
parameters.insert("weights".to_string(), Array2::ones((10, 5)));

let penalty = regularizer.compute_penalty(&parameters)?;
let gradients = regularizer.compute_gradient(&parameters)?;

// Add penalty to loss and gradients to parameter updates
total_loss += penalty;
```

## Data Augmentation

Apply on-the-fly data augmentation during training:

```rust
use tensorlogic_train::{NoiseAugmenter, ScaleAugmenter, MixupAugmenter, DataAugmenter};
use scirs2_core::ndarray::Array2;

// Gaussian noise augmentation
let noise_aug = NoiseAugmenter::new(0.0, 0.1); // mean=0, std=0.1
let augmented = noise_aug.augment(&data.view())?;

// Mixup augmentation (Zhang et al., ICLR 2018)
let mixup = MixupAugmenter::new(1.0); // alpha = 1.0 (uniform mixing)
let (mixed_data, mixed_targets) = mixup.mixup(
    &data.view(),
    &targets.view(),
    0.3, // lambda: mixing coefficient
)?;
```

## Logging and Monitoring

Track training progress with multiple logging backends:

```rust
use tensorlogic_train::{ConsoleLogger, FileLogger, MetricsLogger, LoggingBackend};
use std::path::PathBuf;

// Console logging with timestamps
let console = ConsoleLogger::new(true); // with_timestamp = true
console.log_epoch(1, 10, 0.532, Some(0.612))?;

// File logging
let file_logger = FileLogger::new(
    PathBuf::from("/tmp/training.log"),
    true, // append mode
)?;
file_logger.log_batch(1, 100, 0.425)?;

// Aggregate metrics across backends
let mut metrics_logger = MetricsLogger::new();
metrics_logger.add_backend(Box::new(console));
metrics_logger.add_backend(Box::new(file_logger));

// Log to all backends
metrics_logger.log_metric("accuracy", 0.95)?;
metrics_logger.log_epoch(5, 20, 0.234, Some(0.287))?;
```

## Architecture

### Module Structure

```
tensorlogic-train/
├── src/
│   ├── lib.rs                    # Public API exports
│   ├── error.rs                  # Error types
│   ├── loss.rs                   # 15 loss functions
│   ├── optimizer.rs              # Re-exports all optimizers
│   ├── optimizers/               # 18 optimizer implementations
│   │   ├── sgd.rs, adam.rs, adamw.rs, adamp.rs
│   │   ├── rmsprop.rs, adagrad.rs, nadam.rs
│   │   ├── lamb.rs, adamax.rs, lookahead.rs
│   │   ├── adabelief.rs, radam.rs, lars.rs
│   │   ├── sam.rs, lion.rs, prodigy.rs
│   │   ├── schedulefree.rs, sophia.rs
│   │   └── common.rs
│   ├── scheduler.rs              # 12 LR schedulers
│   ├── batch.rs                  # Batch management
│   ├── trainer.rs                # Main training loop
│   ├── callbacks/                # 14+ training callbacks
│   │   ├── core.rs, advanced.rs, checkpoint.rs
│   │   ├── early_stopping.rs, gradient.rs
│   │   ├── histogram.rs, lr_finder.rs, profiling.rs
│   │   └── mod.rs
│   ├── metrics/                  # 19+ evaluation metrics (7 modules)
│   │   ├── basic.rs, advanced.rs, ranking.rs
│   │   ├── vision.rs, calibration.rs, tracker.rs
│   │   └── mod.rs
│   ├── model.rs                  # Model trait interface
│   ├── regularization.rs         # 9 regularization techniques
│   ├── augmentation.rs           # 9 data augmentation types
│   ├── stochastic_depth.rs       # DropPath / Stochastic Depth
│   ├── dropblock.rs              # DropBlock structured dropout
│   ├── logging.rs                # 6 logging backends
│   ├── memory.rs                 # Memory management
│   ├── data.rs                   # Data loading and preprocessing
│   ├── curriculum.rs             # 5 curriculum learning strategies
│   ├── transfer.rs               # Transfer learning utilities
│   ├── hyperparameter.rs         # Grid, Random, Bayesian search
│   ├── crossval.rs               # 4 cross-validation strategies
│   ├── ensemble.rs               # 4 ensembling methods
│   ├── distillation.rs           # Knowledge distillation
│   ├── label_smoothing.rs        # Label smoothing and Mixup
│   ├── multitask.rs              # Multi-task learning
│   ├── pruning.rs                # Model pruning
│   ├── sampling.rs               # 7 advanced sampling strategies
│   ├── quantization.rs           # INT8/INT4/INT2 quantization
│   ├── mixed_precision.rs        # FP16/BF16 mixed precision
│   ├── few_shot.rs               # Prototypical/Matching networks
│   ├── meta_learning.rs          # MAML and Reptile
│   ├── gradient_centralization.rs# Gradient centralization
│   ├── neural_ode.rs             # Neural ODE (RK4, DOPRI5, adjoint sensitivity) (v0.1.18)
│   ├── online_learning.rs        # Online learning (Perceptron, PA, OGD, FTRL) (v0.1.20)
│   ├── adversarial.rs            # Adversarial training (FGSM, PGD, L∞/L2/L1) (v0.1.21)
│   ├── structured_logging.rs     # tracing integration (feature-gated)
│   └── utils.rs                  # Model introspection utilities
```

### Key Traits

- **`Model`**: Forward/backward passes and parameter management
- **`AutodiffModel`**: Automatic differentiation integration (trait extension)
- **`DynamicModel`**: Variable-sized input support
- **`Loss`**: Compute loss and gradients
- **`Optimizer`**: Update parameters with gradients
- **`LrScheduler`**: Adjust learning rate
- **`Callback`**: Hook into training events
- **`Metric`**: Evaluate model performance
- **`Regularizer`**: Compute regularization penalties and gradients
- **`DataAugmenter`**: Apply data transformations
- **`LoggingBackend`**: Log training metrics and events

## Integration with SciRS2

This crate strictly follows the SciRS2 integration policy:

```rust
// Correct: Use SciRS2 types
use scirs2_core::ndarray::{Array, Array2};

// Wrong: Never use these directly
// use ndarray::Array2;  // Never!
// use rand::thread_rng; // Never!
```

All tensor operations use `scirs2_core::ndarray`, ready for seamless integration with `scirs2-autograd` for automatic differentiation.

## Test Coverage

All modules have comprehensive unit tests:

| Module | Tests | Coverage |
|--------|-------|----------|
| `loss.rs` | 15 | All 15 loss functions |
| `optimizer.rs` / `optimizers/` | 79 | All 18 optimizers + ScheduleFreeAdamW, Prodigy, Sophia |
| `scheduler.rs` | 13 | All 12 schedulers |
| `batch.rs` | 5 | Batch iteration and sampling |
| `trainer.rs` | 3 | Training loop |
| `callbacks/` | 28 | All 14+ callbacks |
| `metrics/` | 34 | All 19+ metrics (refactored into 7 modules) |
| `model.rs` | 6 | Model interface and implementations |
| `regularization.rs` | 16 | L1, L2, ElasticNet, Composite, Spectral Norm, MaxNorm, Orthogonal, Group Lasso |
| `augmentation.rs` | 25 | Noise, Scale, Rotation, Mixup, CutMix, RandomErasing, CutOut |
| `stochastic_depth.rs` | 14 | DropPath, Linear/Exponential schedulers |
| `dropblock.rs` | 12 | DropBlock structured dropout |
| `logging.rs` | 15 | Console, File, TensorBoard, CSV, JSONL, MetricsLogger |
| `memory.rs` | 10 | MemoryStats, profiler, budget manager |
| `curriculum.rs` | 11 | Linear, Exponential, SelfPaced, Competence, Task, Manager |
| `transfer.rs` | 13 | Freezing, Progressive, Discriminative, FeatureExtractor |
| `hyperparameter.rs` | 32 | Grid, Random, Bayesian Opt, GP, Acquisition |
| `crossval.rs` | 12 | KFold, Stratified, TimeSeries, LeaveOneOut |
| `ensemble.rs` | 22 | Voting, Averaging, Stacking, Bagging |
| `distillation.rs` | 7 | Standard, Feature, Attention distillation |
| `label_smoothing.rs` | 8 | Label smoothing, Mixup |
| `multitask.rs` | 5 | Fixed, DTP, PCGrad |
| `data.rs` | 12 | Dataset, CSV loader, preprocessor, encoders |
| `utils.rs` | 11 | Model summary, gradient stats, time estimation |
| `quantization.rs` | 14 | INT8/4/2, PTQ, QAT, calibration |
| `mixed_precision.rs` | 14 | FP16/BF16, loss scaling, master weights |
| `gradient_centralization.rs` | 14 | LayerWise, Global, PerRow, PerColumn |
| `structured_logging.rs` | 4 | Builder, formats, levels |
| `few_shot.rs` | 13 | Prototypical, Matching networks, distances |
| `meta_learning.rs` | 15 | MAML, Reptile, task management |
| `pruning.rs` | 13 | Magnitude, Gradient, Structured, Global, Iterative |
| `sampling.rs` | 14 | Hard negative mining, Importance, Focal, Class balanced |
| `neural_ode.rs` | 22 | RK4, DOPRI5, NeuralOde, adjoint sensitivity (v0.1.18) |
| `online_learning.rs` | 28 | Perceptron, PassiveAggressive, OGD, FTRL, OnlineStats (v0.1.20) |
| `adversarial.rs` | 29 | FGSM, PGD, L∞/L2/L1, adversarial training loss (v0.1.21) |
| **Total** | **745** | **100%** |

Run tests with:

```bash
cargo nextest run -p tensorlogic-train --no-fail-fast
```

## Guides and Documentation

Comprehensive guides are available in the [docs/](docs/) directory:

- **[Loss Function Selection Guide](docs/LOSS_FUNCTIONS.md)** - Choose the right loss for your task
  - Decision trees and comparison tables
  - Detailed explanations of all 15 loss functions
  - Metric learning losses (Contrastive, Triplet)
  - Classification losses (Hinge, KL Divergence, Poly)
  - Best practices and common pitfalls

- **[Hyperparameter Tuning Guide](docs/HYPERPARAMETER_TUNING.md)** - Optimize training performance
  - Learning rate tuning (with LR finder)
  - Batch size selection
  - Optimizer comparison and selection
  - Learning rate schedules
  - Regularization strategies

## Examples

The crate includes comprehensive examples demonstrating all features:

1. **01_basic_training.rs** - Simple regression with SGD
2. **02_classification_with_metrics.rs** - Multi-class classification
3. **03_callbacks_and_checkpointing.rs** - Advanced callbacks and training management
4. **04_logical_loss_training.rs** - Constraint-based training
5. **05_profiling_and_monitoring.rs** - Performance profiling
6. **06_curriculum_learning.rs** - Progressive difficulty training
7. **07_transfer_learning.rs** - Fine-tuning strategies
8. **08_hyperparameter_optimization.rs** - Grid/random search
9. **09_cross_validation.rs** - Robust model evaluation
10. **10_ensemble_learning.rs** - Model ensembling
11. **11_advanced_integration.rs** - Complete workflow integration
12. **12_knowledge_distillation.rs** - Model compression
13. **13_label_smoothing.rs** - Regularization techniques
14. **14_multitask_learning.rs** - Multi-task training
15. **15_training_recipes.rs** - Complete end-to-end workflows
16. **16_structured_logging.rs** - Production-grade observability
17. **17_few_shot_learning.rs** - Learning from minimal examples
18. **18_meta_learning.rs** - MAML and Reptile algorithms
19. **21_bayesian_optimization.rs** - Bayesian hyperparameter search
20. **22_gradient_centralization.rs** - Gradient preprocessing

Run any example with:
```bash
cargo run --example 01_basic_training
```

## Benchmarks

Performance benchmarks are available in the `benches/` directory:

```bash
cargo bench -p tensorlogic-train
```

Benchmarks cover:
- Optimizer comparison (SGD, Adam, AdamW)
- Batch size scaling
- Dataset size scaling
- Model size scaling
- Gradient clipping overhead

## License

Apache-2.0

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## References

- [Tensorlogic Project](../../README.md)
- [SciRS2](https://github.com/cool-japan/scirs)
- [Tensor Logic Paper](https://arxiv.org/abs/2510.12269)

---

**Status**: Production Ready (v0.1.3 Stable)
**Last Updated**: 2026-08-31
**Version**: 0.1.3
**Test Coverage**: 745/745 tests passing (100%)
**Code Quality**: Zero warnings, clippy clean
**Features**: 15 losses, 18 optimizers, 12 schedulers, 14+ callbacks, 9 regularization techniques, 9 augmentations, DropPath, DropBlock, quantization, mixed precision, few-shot learning, meta-learning, Bayesian optimization, gradient centralization, Neural ODE (RK4/DOPRI5/adjoint), online learning (Perceptron/PA/OGD/FTRL), adversarial training (FGSM/PGD)
**Examples**: 20 comprehensive training examples
