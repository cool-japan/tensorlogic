# TensorLogic Training Examples

This directory contains 11 comprehensive examples demonstrating the `tensorlogic-train` crate's capabilities, including advanced features like curriculum learning, transfer learning, hyperparameter optimization, cross-validation, ensemble methods, and complete integration workflows.

## Running Examples

Run any example with:

```bash
cargo run --example <example_name>
```

For example:
```bash
cargo run --example 01_basic_training
```

## Examples Overview

### 01_basic_training.rs
**Basic regression training with SGD**

Demonstrates:
- Simple linear regression
- MSE loss function
- SGD optimizer with momentum
- Epoch-level callbacks
- Training history tracking

Key concepts:
- Creating synthetic regression data
- Initializing model parameters
- Basic training loop
- Monitoring training progress

Run with:
```bash
cargo run --example 01_basic_training
```

---

### 02_classification_with_metrics.rs
**Multi-class classification with comprehensive metrics**

Demonstrates:
- 3-class classification problem
- Cross-entropy loss with label smoothing
- Adam optimizer
- Multiple evaluation metrics (Accuracy, Precision, Recall, F1)
- Confusion matrix analysis
- Per-class performance reporting

Key concepts:
- Classification data preparation
- Label smoothing for regularization
- Metric tracking during training
- Post-training analysis with confusion matrices
- Per-class metric computation

Run with:
```bash
cargo run --example 02_classification_with_metrics
```

---

### 03_callbacks_and_checkpointing.rs
**Advanced training with callbacks and state management**

Demonstrates:
- Early stopping to prevent overfitting
- Model checkpointing (save best models)
- Learning rate scheduling (Cosine Annealing)
- Reduce LR on plateau
- Gradient monitoring
- Learning rate finder (optional)
- Training state persistence

Key concepts:
- Callback composition
- Automatic hyperparameter adjustment
- Model persistence
- Gradient health monitoring
- Optimal learning rate discovery

Features:
- **EarlyStoppingCallback**: Stops training when validation stops improving
- **CheckpointCallback**: Saves best model states
- **CosineAnnealingLrScheduler**: Gradually reduces learning rate
- **ReduceLrOnPlateauCallback**: Adaptive LR reduction
- **GradientMonitor**: Detects vanishing/exploding gradients
- **LearningRateFinder**: Find optimal learning rate (uncomment to use)

Run with:
```bash
cargo run --example 03_callbacks_and_checkpointing
```

---

### 04_logical_loss_training.rs
**Training with logical constraints and rules**

Demonstrates:
- Combining supervised loss with logical constraints
- Rule satisfaction loss
- Constraint violation penalties
- Multi-objective loss weighting
- Logic-guided learning with AdamW

Key concepts:
- Multi-objective optimization
- Logical rule enforcement during training
- Constraint satisfaction in neural networks
- Weight decay for regularization
- Domain-specific loss design

Loss composition:
```
L_total = 1.0 × L_supervised + 2.0 × L_rules + 5.0 × L_constraints
```

This is particularly useful when you want your model to:
- Respect domain-specific rules
- Satisfy hard constraints
- Learn from both data and logical knowledge

Run with:
```bash
cargo run --example 04_logical_loss_training
```

---

### 05_profiling_and_monitoring.rs
**Advanced monitoring with profiling and histogram tracking**

Demonstrates:
- Performance profiling during training
- Weight histogram tracking
- Gradient monitoring
- Comprehensive training diagnostics
- Performance optimization insights
- Combining multiple monitoring callbacks

Key concepts:
- ProfilingCallback for timing analysis
- HistogramCallback for weight distributions
- GradientMonitor for training health
- Comprehensive performance reporting
- Training optimization recommendations

Features:
- Batch/epoch timing measurement
- Throughput metrics (samples/sec, batches/sec)
- Weight distribution tracking
- Gradient health monitoring
- Automatic issue detection

Run with:
```bash
cargo run --example 05_profiling_and_monitoring
```

---

### 06_curriculum_learning.rs
**Progressive training with curriculum learning strategies**

Demonstrates:
- LinearCurriculum (gradual difficulty increase)
- ExponentialCurriculum (rapid ramp-up)
- SelfPacedCurriculum (model-driven pace)
- CompetenceCurriculum (adaptive difficulty)
- TaskCurriculum (multi-task progressive training)
- CurriculumManager for state management

Key concepts:
- Difficulty score computation
- Sample selection based on model competence
- Progressive training from easy to hard samples
- Dynamic curriculum adaptation
- Stateful curriculum management

Run with:
```bash
cargo run --example 06_curriculum_learning
```

---

### 07_transfer_learning.rs
**Transfer learning and fine-tuning strategies**

Demonstrates:
- Layer freezing and unfreezing
- Progressive unfreezing (gradual layer activation)
- Discriminative fine-tuning (layer-specific learning rates)
- Feature extraction mode
- TransferLearningManager for unified control

Key concepts:
- Freezing pretrained layers
- Gradual unfreezing to prevent catastrophic forgetting
- Different learning rates for different layers
- Feature extraction vs. fine-tuning
- Multi-phase transfer learning workflow

Run with:
```bash
cargo run --example 07_transfer_learning
```

---

### 08_hyperparameter_optimization.rs
**Automated hyperparameter search**

Demonstrates:
- Grid search (exhaustive exploration)
- Random search (stochastic sampling)
- Parameter space definition (discrete, continuous, log-uniform, int ranges)
- Result tracking and comparison
- Best configuration selection

Key concepts:
- Defining search spaces
- Exhaustive vs. stochastic search
- Log-uniform distributions for learning rates
- Configuration generation and evaluation
- Result analysis and ranking

Run with:
```bash
cargo run --example 08_hyperparameter_optimization
```

---

### 09_cross_validation.rs
**Robust model evaluation with cross-validation**

Demonstrates:
- K-Fold cross-validation
- Stratified K-Fold (class-balanced)
- Time Series Split (temporal-aware)
- Leave-One-Out cross-validation
- Result aggregation and statistical analysis

Key concepts:
- Train/validation split strategies
- Class distribution preservation
- Temporal ordering for time series
- Result aggregation across folds
- Statistical significance testing

Run with:
```bash
cargo run --example 09_cross_validation
```

---

### 10_ensemble_learning.rs
**Model ensembling for improved performance**

Demonstrates:
- Voting ensemble (hard and soft voting)
- Averaging ensemble (regression)
- Weighted voting based on model performance
- Stacking ensemble (meta-learner)
- Bagging utilities (bootstrap aggregating)

Key concepts:
- Model diversity and combination
- Hard vs. soft voting
- Meta-learning with stacking
- Bootstrap sampling for bagging
- Out-of-bag validation

Run with:
```bash
cargo run --example 10_ensemble_learning
```

---

### 11_advanced_integration.rs
**End-to-end production training pipeline**

Demonstrates:
- Complete workflow combining multiple advanced features
- Hyperparameter optimization with random search
- Cross-validation for robust evaluation
- Ensemble learning with bagging
- Production-grade training pipeline

Key concepts:
- Integrating multiple features in a realistic workflow
- Best practices for model development
- Systematic approach to hyperparameter tuning
- Robust evaluation strategies
- Ensemble methods for improved predictions

Workflow:
1. **Hyperparameter Optimization**: Random search to find optimal learning rate and batch size
2. **Cross-Validation**: 5-fold CV for robust performance estimation
3. **Ensemble Learning**: Train 5 models with bagging and soft voting

This example represents a complete, production-ready training pipeline
that can serve as a template for real-world projects.

Run with:
```bash
cargo run --example 11_advanced_integration
```

---

## Common Patterns

### Creating a Trainer

```rust
use tensorlogic_train::{Trainer, TrainerConfig, MseLoss, AdamOptimizer};

let config = TrainerConfig {
    num_epochs: 50,
    batch_size: 32,
    shuffle: true,
    validation_frequency: 1,
    ..Default::default()
};

let loss = Box::new(MseLoss);
let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig::default()));
let trainer = Trainer::new(config, loss, optimizer);
```

### Adding Callbacks

```rust
use tensorlogic_train::{CallbackList, EpochCallback, EarlyStoppingCallback};

let mut callbacks = CallbackList::new();
callbacks.add(Box::new(EpochCallback::new(true)));
callbacks.add(Box::new(EarlyStoppingCallback::new(10, 0.001)));

trainer = trainer.with_callbacks(callbacks);
```

### Adding Metrics

```rust
use tensorlogic_train::{MetricTracker, Accuracy, F1Score};

let mut metrics = MetricTracker::new();
metrics.add(Box::new(Accuracy::default()));
metrics.add(Box::new(F1Score::default()));

trainer = trainer.with_metrics(metrics);
```

### Training

```rust
let history = trainer.train(
    &train_data.view(),
    &train_targets.view(),
    Some(&val_data.view()),
    Some(&val_targets.view()),
    &mut parameters,
)?;
```

## Advanced Topics

### Custom Callbacks

Implement the `Callback` trait to create custom training hooks:

```rust
use tensorlogic_train::{Callback, TrainingState, TrainResult};

struct MyCallback;

impl Callback for MyCallback {
    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState) -> TrainResult<()> {
        println!("Epoch {} completed with loss: {}", epoch, state.train_loss);
        Ok(())
    }
}
```

### Custom Metrics

Implement the `Metric` trait for domain-specific evaluation:

```rust
use tensorlogic_train::{Metric, TrainResult};
use scirs2_core::ndarray::{ArrayView, Ix2};

struct MyMetric;

impl Metric for MyMetric {
    fn name(&self) -> &str {
        "MyMetric"
    }

    fn compute(&mut self, predictions: &ArrayView<f64, Ix2>, targets: &ArrayView<f64, Ix2>) -> TrainResult<f64> {
        // Your metric computation
        Ok(0.0)
    }

    fn reset(&mut self) {}
}
```

### Learning Rate Scheduling

Different schedulers for different scenarios:

- **StepLR**: Step decay at fixed intervals
- **ExponentialLR**: Exponential decay
- **CosineAnnealingLR**: Smooth cosine decay (recommended for most cases)
- **OneCycleLR**: Super-convergence schedule
- **CyclicLR**: Cyclical learning rates
- **WarmupCosine**: Warmup + cosine annealing

Example:
```rust
use tensorlogic_train::{OneCycleLrScheduler, LrScheduler};

let scheduler = Box::new(OneCycleLrScheduler::new(
    0.0001,  // initial_lr
    0.01,    // max_lr
    50,      // total_epochs
));

trainer = trainer.with_scheduler(scheduler);
```

## Tips

### Basic Training (Examples 01-05)

1. **Start Simple**: Begin with `01_basic_training.rs` to understand the basics
2. **Monitor Metrics**: Use `02_classification_with_metrics.rs` to learn comprehensive evaluation
3. **Use Callbacks**: Leverage `03_callbacks_and_checkpointing.rs` for production training
4. **Logical Constraints**: Apply domain knowledge with `04_logical_loss_training.rs`
5. **Performance Monitoring**: Use `05_profiling_and_monitoring.rs` for optimization insights

### Advanced Features (Examples 06-10)

6. **Curriculum Learning** (`06_curriculum_learning.rs`):
   - Start with LinearCurriculum for predictable progression
   - Use SelfPacedCurriculum when model performance varies significantly
   - Combine with early stopping for best results
   - Monitor difficulty distribution across epochs

7. **Transfer Learning** (`07_transfer_learning.rs`):
   - Begin with feature extraction (frozen encoder) for 10 epochs
   - Then use discriminative fine-tuning with smaller LRs
   - Progressive unfreezing helps prevent catastrophic forgetting
   - Use lower learning rates (1e-5 to 1e-4) for fine-tuning

8. **Hyperparameter Optimization** (`08_hyperparameter_optimization.rs`):
   - Use grid search for ≤3 hyperparameters
   - Use random search for ≥4 hyperparameters
   - Always use log-uniform for learning rates
   - Run random search 2-3× longer than grid would take

9. **Cross-Validation** (`09_cross_validation.rs`):
   - Always use stratified K-fold for classification
   - Use time series split for temporal data
   - Report mean ± std dev for transparency
   - 5-10 folds provides good bias-variance tradeoff

10. **Ensemble Learning** (`10_ensemble_learning.rs`):
    - Use 5-10 models for good diversity vs. cost
    - Soft voting > hard voting (when applicable)
    - Stacking can outperform simple voting
    - Ensure model diversity for best results

### General Best Practices

11. **Hyperparameter Tuning**:
    - Use `LearningRateFinder` to find optimal learning rate
    - Start with Adam optimizer for most problems
    - Use AdamW for better generalization
    - Try OneCycleLR for faster convergence

12. **Regularization**:
    - Label smoothing for classification
    - Weight decay (especially with AdamW)
    - Early stopping to prevent overfitting
    - Dropout (implement in your model)
    - Curriculum learning for gradual difficulty

13. **Debugging**:
    - Use `GradientMonitor` to detect training issues
    - Check confusion matrices for classification problems
    - Monitor per-class metrics for imbalanced datasets
    - Save checkpoints frequently
    - Profile with `ProfilingCallback` to find bottlenecks

14. **Production Workflow**:
    - Use cross-validation for robust evaluation
    - Hyperparameter search on dev set
    - Curriculum learning for complex tasks
    - Transfer learning when pretrained models available
    - Ensemble top models for final deployment

## Further Reading

- [Main README](../../README.md) - Project overview
- [API Documentation](../src/lib.rs) - Complete API reference
- [TODO](../TODO.md) - Upcoming features

## Contributing

Found a bug or have an example idea? See [CONTRIBUTING.md](../../../CONTRIBUTING.md).

## License

Apache-2.0
