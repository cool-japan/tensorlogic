# TensorLogic Train — TODO

**Status**: Stable | **Version**: 0.1.3 | **Released**: 2026-04-06 | **Last Updated**: 2026-08-31
**History**: See [CHANGELOG.md](../../CHANGELOG.md) for release history.

Training loop, loss functions, optimizers, schedulers, and callbacks.

## Completed

**Phase 6.1 - Core Training Infrastructure** - 100% COMPLETE

### Module Structure
- [x] Error types (`error.rs`)
- [x] Loss functions (`loss.rs`)
- [x] Optimizers (`optimizer.rs` + `optimizers/`)
- [x] Learning rate schedulers (`scheduler.rs`)
- [x] Batch management (`batch.rs`)
- [x] Training loop (`trainer.rs`)
- [x] Callbacks (`callbacks/`)
- [x] Metrics (`metrics/`)

### Loss Functions
- [x] **Standard losses**
  - [x] Cross-entropy loss with numerical stability
  - [x] MSE loss for regression
  - [x] Loss trait with compute() and gradient() methods
- [x] **Logical losses**
  - [x] Rule satisfaction loss (soft penalties with temperature)
  - [x] Constraint violation loss (penalty-based)
  - [x] Logical loss composer (multi-objective with weights)
- [x] **Robust losses**: Focal (class imbalance), Huber (outliers)
- [x] **Segmentation losses**: Dice, Tversky (IoU-based)
- [x] **Metric learning**: Contrastive, Triplet
- [x] **Classification**: Hinge (SVM-style), KL Divergence
- [x] **Advanced**: BCE with logits, Poly Loss
- [x] **Test coverage**: 15 unit tests passing

### Optimizers
- [x] **SGD with momentum**
  - [x] Momentum buffers
  - [x] Gradient clipping support
- [x] **Adam optimizer**
  - [x] First and second moment estimation
  - [x] Bias correction
  - [x] Gradient clipping
- [x] **AdamW optimizer** - Decoupled weight decay
- [x] **AdamP optimizer** - Adam with projection
- [x] **RMSprop** - Adaptive learning rates
- [x] **Adagrad** - Accumulating gradient normalization
- [x] **NAdam** - Nesterov-accelerated Adam
- [x] **LAMB** - Layer-wise adaptive moments
- [x] **AdaMax** - Adam with infinity norm
- [x] **Lookahead** - Slow/fast weight method
- [x] **AdaBelief** (NeurIPS 2020) - Gradient belief adaptation
- [x] **RAdam** (ICLR 2020) - Rectified Adam
- [x] **LARS** - Layer-wise adaptive rate scaling
- [x] **SAM** (ICLR 2021) - Sharpness aware minimization
- [x] **Lion** - Modern sign-based optimizer (EvoLved Sign Momentum)
- [x] **Prodigy** (2024) - Auto-tuning learning rate
- [x] **ScheduleFreeAdamW** (2024) - No LR schedule needed (Defazio et al., arXiv:2405.15682)
- [x] **Sophia** - Second-order optimizer with Hessian estimates (GNB variant)
- [x] **Optimizer trait** with state_dict/load_state_dict
- [x] **Gradient Centralization** wrapper (GcStrategy: LayerWise, Global, PerRow, PerColumn)
- [x] **Test coverage**: ~79 tests passing across all optimizers

### Learning Rate Schedulers
- [x] **StepLR**: Decay by gamma every N epochs
- [x] **ExponentialLR**: Exponential decay every epoch
- [x] **CosineAnnealingLR**: Cosine annealing schedule
- [x] **WarmupScheduler**: Linear warmup phase
- [x] **OneCycleLR**: Super-convergence single cycle
- [x] **PolynomialDecayLR**: Polynomial decay
- [x] **CyclicLR**: Triangular/exponential cyclic
- [x] **WarmupCosineLR**: Warmup + cosine annealing
- [x] **NoamScheduler**: Attention is All You Need schedule
- [x] **MultiStepLR**: Decay at milestone epochs
- [x] **ReduceLROnPlateau**: Adaptive reduction
- [x] **SgdrScheduler**: SGD with Warm Restarts
- [x] **LrScheduler trait**: Unified interface with state_dict/load_state_dict
- [x] **Test coverage**: 13 unit tests passing

### Batch Management
- [x] **BatchIterator**: Configurable batch iteration
  - [x] Shuffling support (deterministic and random)
  - [x] Drop last incomplete batch option
  - [x] Batch size configuration
- [x] **DataShuffler**: Deterministic shuffling with seed
- [x] **extract_batch()**: Efficient batch extraction from arrays
- [x] **Test coverage**: 5 unit tests passing

### Training Loop
- [x] **Trainer struct**: Main training orchestrator
  - [x] Epoch iteration with state tracking
  - [x] Batch iteration with callbacks
  - [x] Parameter updates via optimizer
  - [x] Validation loop
  - [x] Metrics computation
- [x] **TrainerConfig**: Comprehensive configuration
- [x] **TrainingState**: State tracking for callbacks
- [x] **TrainingHistory**: Loss and metrics history
- [x] **Test coverage**: 3 unit tests passing

### Callbacks
- [x] **Callback trait**: Unified callback interface
  - [x] on_train_begin/end
  - [x] on_epoch_begin/end
  - [x] on_batch_begin/end
  - [x] on_validation_end
  - [x] should_stop() for early termination
- [x] **CallbackList**: Callback orchestration
- [x] **EpochCallback**: Epoch-level logging
- [x] **BatchCallback**: Batch-level logging with frequency
- [x] **ValidationCallback**: Validation frequency control
- [x] **CheckpointCallback**: Model checkpointing with optional gzip compression
- [x] **EarlyStoppingCallback**: Early stopping with patience
- [x] **ReduceLrOnPlateauCallback**: Adaptive LR reduction
- [x] **LearningRateFinder**: Exponential/linear LR range test
- [x] **GradientMonitor**: Gradient norm tracking, vanishing/exploding detection
- [x] **HistogramCallback**: Weight distribution monitoring with ASCII visualization
- [x] **ProfilingCallback**: Training speed and throughput tracking
- [x] **ModelEMACallback**: Exponential moving average
- [x] **GradientAccumulationCallback**: Simulate large batches with multiple scaling strategies
- [x] **SWACallback**: Stochastic Weight Averaging
- [x] **MemoryProfilerCallback**: Track memory usage during training
- [x] **Test coverage**: 28 tests passing

### Metrics (7 modules)
- [x] **Accuracy**, **Precision**, **Recall**, **F1Score** (basic.rs)
- [x] **ConfusionMatrix**, **RocCurve**, **PerClassMetrics**, **BalancedAccuracy**, **CohensKappa**, **MatthewsCorrelationCoefficient** (advanced.rs)
- [x] **TopKAccuracy**, **NDCG** (ranking.rs)
- [x] **IoU**, **MeanIoU**, **DiceCoefficient**, **MeanAveragePrecision** (vision.rs)
- [x] **ExpectedCalibrationError**, **MaximumCalibrationError** (calibration.rs)
- [x] **MetricTracker** (tracker.rs)
- [x] Metrics module refactored: 2340-line metrics.rs split into 7 focused files
- [x] **Test coverage**: 34 tests passing

### Integration with SciRS2
- [x] Use scirs2-core for ndarray operations
- [x] Workspace dependencies configured
- [x] Follows SCIRS2 integration policy
- [x] Ready for scirs2-autograd integration

### Build and Quality
- [x] Zero compilation errors
- [x] Zero warnings (all unused imports fixed)
- [x] Cargo.toml configured with all dependencies
- [x] All 499 unit tests implemented and passing

---

**Phase 6.2 - Advanced Training Features** - 100% COMPLETE

### Model Integration
- [x] Define model interface/trait (Model, AutodiffModel, DynamicModel)
- [x] Create LinearModel as reference implementation
- [x] Integrate autodiff trait (placeholder for future scirs2-autograd)
- [x] Replace forward/backward placeholders in Trainer (Model trait used)
- [x] Parameter management (state_dict, load_state_dict)
- [x] **Test coverage**: 6 new tests (all passing)

### Advanced Training Features
- [x] Gradient clipping by norm (L2 norm via GradClipMode::Norm)
- [x] compute_gradient_norm() helper function
- [x] Updated all optimizers (SGD, Adam, AdamW) to support both Value and Norm modes
- [x] GradClipMode enum exported
- [ ] Distributed training support (FUTURE)
- [ ] GPU acceleration via SciRS2 (FUTURE)

### Enhanced Metrics
- [x] Confusion matrix with per-class analysis
- [x] ROC/AUC curves (binary classification)
- [x] Per-class metrics reporting (PerClassMetrics struct)
- [x] Display trait implementations for pretty printing
- [x] **Test coverage**: 8 new tests (all passing)

---

**Phase 6.3 - Advanced Callbacks and Tooling** - 100% COMPLETE

### Advanced Callbacks
- [x] Learning rate finder (LearningRateFinder)
- [x] Gradient flow monitoring (GradientMonitor)
- [x] Weight histogram tracking (HistogramCallback)
- [x] Profiling callback (ProfilingCallback)

### Enhanced Checkpointing
- [x] TrainingCheckpoint struct with full state serialization
- [x] Save full model state (parameters + optimizer + scheduler)
- [x] Load checkpoint and restore training state
- [x] Resume training from checkpoint (train_from_checkpoint)
- [x] Scheduler state_dict/load_state_dict for all schedulers
- [x] Compression support (via `oxiarc-deflate`, Pure Rust — replaces flate2)
- [ ] Cloud storage backends (FUTURE)

### Logging Integration
- [x] TensorBoard writer (real tfevents format with CRC32)
- [x] CSV logger for analysis
- [x] JSONL logger for programmatic access
- [x] Structured logging (tracing/tracing-subscriber, optional feature)
- [ ] Weights and Biases integration (FUTURE)
- [ ] MLflow tracking (FUTURE)

### Performance Benchmarking
- [x] Criterion-based benchmark suite
- [x] Optimizer comparison benchmarks
- [x] Batch size scaling benchmarks
- [x] Dataset scaling benchmarks
- [x] Model size scaling benchmarks
- [x] Gradient clipping overhead benchmarks

---

**Phase 6.4 through 6.11 - All Complete**

### Curriculum Learning
- [x] LinearCurriculum, ExponentialCurriculum
- [x] SelfPacedCurriculum, CompetenceCurriculum, TaskCurriculum
- [x] CurriculumManager for state management
- [x] 11 comprehensive tests

### Transfer Learning
- [x] LayerFreezingConfig, ProgressiveUnfreezing
- [x] DiscriminativeFineTuning, FeatureExtractorMode
- [x] TransferLearningManager (unified management)
- [x] 13 comprehensive tests

### Hyperparameter Optimization
- [x] LearningRateFinder (automatic LR tuning)
- [x] Grid search (HyperparamSpace, Cartesian product)
- [x] Random search (stochastic, reproducible with seeding)
- [x] Bayesian Optimization (GP surrogate model with RBF, Matern 3/2 kernels)
  - [x] Acquisition functions: Expected Improvement, UCB, Probability of Improvement
  - [x] Cholesky decomposition for efficient GP inference
  - [x] Multi-dimensional optimization, continuous/discrete/log-uniform/integer spaces
  - [x] 32 comprehensive tests
- [x] Neural architecture search — `src/nas/`: `ArchSearchSpace`, `ArchSampler`, `RegularizedEvolution` (Real et al. 2019 aging evolution with tournament selection + ask/tell), `RandomArchSearch` baseline; `Architecture`/`LayerSpec` with `param_count()` and `HyperparamConfig` interop; 15 tests

### Cross-Validation
- [x] KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut
- [x] CrossValidationResults (result aggregation)
- [x] 12 comprehensive tests

### Model Ensembling
- [x] VotingEnsemble (hard and soft voting)
- [x] AveragingEnsemble (weighted averaging)
- [x] StackingEnsemble (meta-learner)
- [x] BaggingHelper (bootstrap sampling)
- [x] ModelSoup and SoupRecipe
- [x] 22 comprehensive tests

### Knowledge Distillation
- [x] DistillationLoss (temperature-scaled CE)
- [x] FeatureDistillationLoss
- [x] AttentionTransferLoss
- [x] 7 comprehensive tests

### Label Smoothing and Mixup
- [x] LabelSmoothingLoss
- [x] MixupLoss
- [x] 8 comprehensive tests

### Multi-task Learning
- [x] MultiTaskLoss with fixed weights
- [x] DTP (Dynamic Task Prioritization)
- [x] PCGrad (Projecting Conflicting Gradients)
- [x] TaskWeightingStrategy enum
- [x] 5 comprehensive tests

### Data Loading and Preprocessing
- [x] Dataset struct with train/val/test splits
- [x] CsvLoader with column configuration
- [x] DataPreprocessor (standardize, normalize, min-max)
- [x] LabelEncoder and OneHotEncoder
- [x] 12 comprehensive tests

### Model Pruning
- [x] MagnitudePruner (prune smallest weights)
- [x] GradientPruner (prune weights with smallest gradients)
- [x] StructuredPruner (remove entire neurons/channels/filters)
- [x] GlobalPruner (across all layers)
- [x] Iterative pruning with linear/exponential/cosine schedules
- [x] PruningMask and PruningStats
- [x] 13 comprehensive tests

### Advanced Sampling
- [x] HardNegativeMiner (TopK, threshold, focal strategies)
- [x] ImportanceSampler (with/without replacement)
- [x] FocalSampler (emphasize hard examples)
- [x] ClassBalancedSampler (handle imbalance)
- [x] CurriculumSampler (progressive difficulty)
- [x] OnlineHardExampleMiner (dynamic batch selection)
- [x] BatchReweighter (uniform, inverse loss, focal, gradient norm)
- [x] 14 comprehensive tests

### Model Quantization
- [x] BitWidth: Int8, Int4, Int2
- [x] QuantizationMode: PostTraining (PTQ), QuantizationAwareTraining (QAT)
- [x] Granularity: PerTensor, PerChannel
- [x] QuantizationParams with scale and zero-point
- [x] QuantizedTensor with dequantization
- [x] DynamicRangeCalibrator
- [x] QuantizationConfig with full options
- [x] 14 comprehensive tests

### Mixed Precision Training
- [x] PrecisionMode: F32, F16, BF16
- [x] LossScaler (static and dynamic)
- [x] GradientScaler with overflow detection
- [x] MixedPrecisionTrainer
- [x] AutocastContext for automatic precision management
- [x] MixedPrecisionStats (overflow events, scaling factor)
- [x] Master weight tracking for numerical stability
- [x] 14 comprehensive tests

### Enhanced Gradient Accumulation
- [x] Multiple scaling strategies (Average, Sum, Dynamic)
- [x] Gradient overflow detection (NaN/Inf protection)
- [x] Optional gradient clipping during accumulation
- [x] Memory usage tracking and estimation
- [x] Statistics collection (cycles, max norm)
- [x] Manual reset for error recovery
- [x] 11 comprehensive tests

### Memory Management
- [x] MemoryStats reporting
- [x] MemoryProfilerCallback
- [x] GradientCheckpointConfig
- [x] MemoryBudgetManager
- [x] MemoryEfficientTraining utilities
- [x] 10 comprehensive tests

### Structured Logging (optional feature)
- [x] tracing/tracing-subscriber integration
- [x] Multiple output formats (Pretty, Compact, JSON)
- [x] Configurable log levels and environment filters
- [x] Span-based hierarchical logging
- [x] Zero overhead when feature disabled
- [x] 4 unit tests

### Few-Shot Learning
- [x] SupportSet management
- [x] EpisodeSampler for N-way K-shot tasks
- [x] PrototypicalDistance (prototype-based classification)
- [x] MatchingNetwork (attention-based matching)
- [x] DistanceMetric: Euclidean, Cosine, Manhattan, SquaredEuclidean
- [x] FewShotAccuracy tracker
- [x] 13 comprehensive tests

### Meta-Learning
- [x] MAML (Model-Agnostic Meta-Learning)
- [x] Reptile algorithm (first-order alternative)
- [x] MAMLConfig and ReptileConfig
- [x] MetaTask representation and batching
- [x] MetaStats tracking
- [x] First-order and second-order MAML variants
- [x] 15 comprehensive tests

### Gradient Centralization
- [x] GcStrategy: LayerWise, Global, PerRow, PerColumn
- [x] GcConfig with builder pattern
- [x] GradientCentralization optimizer wrapper (works with any optimizer)
- [x] GcStats (norms before/after, centralized/skipped counts)
- [x] Dynamic enable/disable during training
- [x] State dict save/load support
- [x] 14 comprehensive tests

### Regularization (Advanced)
- [x] DropPath / Stochastic Depth (ECCV 2016)
  - [x] DropPath: randomly drops entire residual paths
  - [x] LinearStochasticDepth: linearly increasing drop probability
  - [x] ExponentialStochasticDepth: exponentially increasing drop probability
  - [x] 14 comprehensive tests
- [x] DropBlock (NeurIPS 2018)
  - [x] DropBlock: structured dropout for CNNs (contiguous block dropping)
  - [x] LinearDropBlockScheduler: linearly increase drop probability
  - [x] 12 comprehensive tests

### Model Utilities
- [x] ParameterStats and ModelSummary
- [x] GradientStats for monitoring
- [x] TimeEstimator for training time prediction
- [x] LrRangeTestAnalyzer
- [x] compare_models utility
- [x] format_duration, print_gradient_report helpers
- [x] 11 comprehensive tests

---

## Test Coverage Summary

| Module | Tests | Status |
|--------|-------|--------|
| loss.rs | 15 | All passing |
| optimizer.rs / optimizers/ | ~79 | All passing (SGD, Adam, AdamW, AdamP, RMSprop, Adagrad, NAdam, LAMB, Lion, ScheduleFreeAdamW, Prodigy, Sophia, ...) |
| scheduler.rs | 13 | All passing |
| batch.rs | 5 | All passing |
| trainer.rs | 3 | All passing |
| callbacks/ | 28 | All passing |
| metrics/ | 34 | All passing (refactored into 7 modules) |
| model.rs | 6 | All passing |
| regularization.rs | 16 | All passing |
| pruning.rs | 13 | All passing |
| sampling.rs | 14 | All passing |
| augmentation.rs | 25 | All passing |
| stochastic_depth.rs | 14 | All passing |
| dropblock.rs | 12 | All passing |
| logging.rs | 15 | All passing |
| memory.rs | 10 | All passing |
| curriculum.rs | 11 | All passing |
| transfer.rs | 13 | All passing |
| hyperparameter.rs | 32 | All passing (Grid, Random, Bayesian Opt, GP, Acquisition) |
| crossval.rs | 12 | All passing |
| ensemble.rs | 22 | All passing |
| distillation.rs | 7 | All passing |
| label_smoothing.rs | 8 | All passing |
| multitask.rs | 5 | All passing |
| data.rs | 12 | All passing |
| utils.rs | 11 | All passing |
| quantization.rs | 14 | All passing |
| mixed_precision.rs | 14 | All passing |
| gradient_centralization.rs | 14 | All passing |
| structured_logging.rs | 4 | All passing |
| few_shot.rs | 13 | All passing |
| meta_learning.rs | 15 | All passing |
| neural_ode.rs | 22 | All passing |
| online_learning.rs | 28 | All passing |
| adversarial.rs | 29 | All passing |
| **Total** | **716** | **100%** |

---

**Total Items Completed:** 200+ features
**Overall Completion:** 100% of core functionality implemented
**Only FUTURE items remaining:** GPU acceleration, distributed training, cloud storage backends, neural architecture search, W&B/MLflow integration, mixed precision execution on GPU

**SCIRS2 Policy:** Fully compliant - all proper scirs2_core::ndarray imports, no direct ndarray/rand imports
**Code Quality:** All files comply with 2000-line limit
**Total source lines:** ~23,000+ (across 36 modules + examples + docs)

## v0.1.7 Enhancements (2026-03-30)

- [x] **Gradient Accumulation** (`gradient_accumulator.rs`): `GradientAccumulator` with `AccumulationConfig` (micro-batch steps, normalization, gradient clipping), `GradientBuffer` with L2 norm, `step()` returns update trigger, `AccumulationStats`. 18 new tests.

## v0.1.14

- [x] **xavier_uniform** (`weight_init.rs`): Glorot uniform initialization sampling from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
- [x] **xavier_normal** (`weight_init.rs`): Glorot normal initialization sampling from N(0, gain^2 * 2 / (fan_in + fan_out))
- [x] **kaiming_uniform** (`weight_init.rs`): He uniform initialization for ReLU networks, U(-bound, bound) with bound = gain * sqrt(3 / fan_in)
- [x] **kaiming_normal** (`weight_init.rs`): He normal initialization sampling from N(0, gain^2 / fan_in)
- [x] **lecun** (`weight_init.rs`): LeCun normal initialization sampling from N(0, 1/fan_in), suitable for SELU activations
- [x] **orthogonal_init** (`weight_init.rs`): QR-decomposition-based orthogonal matrix initialization with configurable gain scaling
- [x] **InitRng** (`weight_init.rs`): Dedicated LCG-based pseudo-random number generator for reproducible weight initialization with `next_f64()` and `next_normal()` methods
- [x] **InitStats** (`weight_init.rs`): Statistical analysis of initialized weights including mean, variance, min, max, histogram bin counts, and formatted summary output

## v0.1.15

- [x] **gaussian_noise** (`augmentation.rs`): Adds element-wise zero-mean Gaussian noise with configurable `stddev`; uses `AugRng` for reproducibility and supports optional clipping to keep values in a valid range
- [x] **dropout** (`augmentation.rs`): Element-wise random zeroing at probability `p` with `1/(1-p)` rescaling to preserve expected activations; produces a companion boolean `dropout_mask` for replay
- [x] **mixup** (`augmentation.rs`): Convex interpolation of two input tensors and their one-hot label vectors with a Beta-distributed mixing coefficient `lambda`; returns mixed sample and mixed labels
- [x] **cutmix** (`augmentation.rs`): Rectangular patch swap between two samples — randomly samples a bounding box and blends labels proportionally to the swapped area fraction
- [x] **random_crop** (`augmentation.rs`): Uniform random sub-tensor crop to a target spatial size; `random_crop_2d` variant handles H×W crops with configurable padding before sampling
- [x] **normalize** (`augmentation.rs`): Channel-wise mean subtraction and standard-deviation division with optional per-channel `mean`/`std` override; `denormalize` is the exact inverse for visualization
- [x] **AugmentationPipeline** (`augmentation.rs`): Composable ordered chain of `AugmentationStep` closures; `apply()` runs the chain in sequence, collecting per-step timing into `AugStats` (total ops, cumulative duration, rejection rate for conditional steps)
- [x] **AugStats** (`augmentation.rs`): Aggregated augmentation statistics tracking total operations applied, cumulative wall-clock duration, and conditional-step rejection rate across a pipeline run

## v0.1.13

- [x] **EarlyStoppingMonitor** (`early_stopping.rs`): Patience-based training termination with configurable `min_delta` improvement threshold, `should_stop()` API returning boolean, tracks best metric value and steps since last improvement
- [x] **MultiMetricMonitor** (`early_stopping.rs`): Track multiple named metrics simultaneously with independent patience per metric, `should_stop_any()` / `should_stop_all()` aggregation, metric history retrieval
- [x] **PlateauDetector** (`early_stopping.rs`): Detect loss plateaus using windowed variance analysis with configurable window size and variance threshold, `is_plateau()` check, supports both minimization and maximization objectives
- [x] **TrainingProgress** (`early_stopping.rs`): Unified progress tracking combining epoch/step counts with elapsed time, ETA estimation, steps-per-second throughput, and `summary()` formatted output

## v0.1.17

- [x] **`augmentation.rs` sub-module refactor**: `augmentation.rs` (single-file, 700+ lines) refactored into `augmentation/` sub-directory with focused modules per transform family: `augmentation/noise.rs` (gaussian_noise, dropout), `augmentation/mix.rs` (mixup, cutmix), `augmentation/spatial.rs` (random_crop), `augmentation/normalize.rs` (normalize, denormalize), and `augmentation/pipeline.rs` (`AugmentationPipeline`, `AugStats`). Public API exported from `augmentation/mod.rs` is fully backward-compatible.

## v0.1.11

- [x] **OptimizerCheckpoint + CheckpointManager + LossTracker** (`checkpoint.rs`): `OptimizerCheckpoint` serializable optimizer state with step/epoch/loss metadata, `CheckpointManager` with configurable keep-last-N policy, best-checkpoint tracking by validation loss, and `load_at_step()` lookup, `LossTracker` windowed moving-average and plateau detection with configurable patience.

## v0.1.4 Enhancements (2026-03-30)

- [x] **Learning Rate Schedulers** (`lr_scheduler.rs`): `LrSchedulerV2` trait (`step`, `current_lr`, `reset`, `steps_taken`, `completed_cycle`). Five implementations: `StepDecayScheduler`, `CosineAnnealingScheduler` (with `with_warm_restarts()`), `WarmupScheduler` (linear warmup wrapping any inner scheduler), `CyclicalScheduler` (triangular CLR), `OneCycleLrScheduler` (linear ramp + cosine decay). `SchedulerConfig` builder for StepDecay/Cosine/OneCycle. 20 new tests.

**Key implementation highlights:**
- 18 optimizers including cutting-edge 2024 methods (Prodigy, ScheduleFreeAdamW)
- 15 loss functions including logical constraint losses
- 12 LR schedulers with full state persistence
- 34 metrics across 7 focused modules
- 9 regularization techniques
- 9 data augmentation types + DropPath + DropBlock
- Complete few-shot and meta-learning infrastructure
- Bayesian optimization with GP surrogate model
- Model quantization (INT8/4/2, PTQ, QAT)
- Mixed precision training (FP16/BF16)
- 20+ comprehensive training examples (6000+ lines)

## v0.1.18 (2026-04-05)

- [x] **Neural ODE** (`neural_ode.rs`): `OdeFunc` trait for user-defined dynamics `f(t, y, params) -> dy/dt` plus `vjp()` for vector-Jacobian products; `rk4_solve()` fixed-step RK4 integrator returning a full `OdeSolution` trajectory; `dopri5_solve()` adaptive Dormand-Prince RK45 solver with step-size control, error estimation (DOPRI5 Butcher tableau), step rejection, and dense output via `AdaptiveSolution`; `OdeSolverConfig` builder (`rtol`, `atol`, `max_steps`, `dense_output`); `NeuralOde<F: OdeFunc>` wrapping a user dynamics function with `(t0, t1)` integration bounds; `NeuralOde::forward()` runs the forward pass and returns the endpoint state; `adjoint_backward()` implements the adjoint sensitivity method — integrates the adjoint ODE backwards through the stored forward trajectory for memory-efficient gradient computation proportional to O(1) state storage rather than O(T).

## v0.1.21 (2026-04-05)

- [x] **Adversarial Training** (`adversarial.rs`): Added adversarial.rs — adversarial training: FGSM (one-step sign gradient), PGD (iterative with random start + projection), L∞/L2/L1 norm constraints; `CrossEntropyAttackLoss`/`MseAttackLoss`; `LinearAttackModel`; `adversarial_training_loss()`, `robustness_eval()`.

## v0.1.20 (2026-04-05)

- [x] **Online Learning** (`online_learning.rs`): `OnlineLearner` trait with `update()`, `predict()`, `reset()`; `Perceptron` (binary, margin-based weight updates); `PassiveAggressive` (PA / PA-I / PA-II variants with configurable aggressiveness `C`); `OnlineGradientDescent` (squared-loss, hinge-loss, and logistic-loss modes with configurable step size); `FtrlProximal` (Follow The Regularized Leader with L1/L2 regularization, per-coordinate adaptive learning rates); `OnlineStats` collecting per-step loss, mistake rate, cumulative regret, and `n_updates`; `online_evaluate()` batch helper running the learner over a labelled dataset with optional training.

## v0.2.0 / Future Work

- [x] **LoRA adapter support** (`lora/`): `LoraLayer` with low-rank A/B decomposition (Hu et al., 2021), merge/unmerge, effective_weight, compression ratio; `LoraAdapter` multi-layer manager with per-layer summary; `LoraConfig` (rank, alpha, dropout, target_modules, seed); `LoraError` with InvalidRank, DimensionMismatch, MergeError, FrozenWeights. 12 unit tests + 2 integration tests. (completed 2026-04-16)
- Quantization-aware training.
- Mixed-precision loops.
- [x] ~~Split `src/hyperparameter.rs` (1,641 L) and `src/loss.rs` (1,551 L) into directory modules.~~ (completed 2026-04-15)
